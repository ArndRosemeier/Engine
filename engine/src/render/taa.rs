//! Temporal anti-aliasing.
//!
//! The camera is nudged by a sub-pixel amount every frame (a Halton 2/3
//! sequence), and this pass blends each new frame into a history buffer that
//! is reprojected from the depth buffer. The history is clamped to the new
//! frame's local colour range, so moving things do not smear. Edges converge
//! to many samples per pixel, and the dithered LOD cross-fade, whose pattern
//! also shifts every frame, averages into a smooth fade.
//!
//! Reprojection follows the camera only (no per-object motion vectors), so
//! fast-moving objects lean on the colour clamp.
//!
//! With a render scale below 1 the scene is drawn smaller than the window
//! and this pass also upscales (like FSR 2 or Unreal's TSR): the history is
//! window-sized, and each frame's few, jittered samples are splatted into it
//! with a small Gaussian at the spot each one really covers. Over the jitter
//! sequence every window pixel gathers samples close to its own centre, so
//! the picture converges to nearly full detail when the view is still.

use glam::{Mat4, Vec2};

use super::postprocess::HDR_FORMAT;

/// Share of the new frame in each blend; the rest comes from history.
const BLEND: f32 = 0.1;
/// Jitter positions before the sequence repeats.
const JITTER_PERIOD: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaUniform {
    inv_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    /// This frame's jitter in NDC.
    jitter: [f32; 2],
    reset: f32,
    blend: f32,
}

/// Lowest render scale: below half the window a 3x3 neighbourhood of
/// samples no longer covers a window pixel's footprint.
pub const MIN_RENDER_SCALE: f32 = 0.5;

/// The scene's size for a window size and render scale (at least 1 x 1).
pub fn scaled_size(width: u32, height: u32, scale: f32) -> (u32, u32) {
    let s = scale.clamp(MIN_RENDER_SCALE, 1.0);
    if s >= 1.0 {
        return (width.max(1), height.max(1));
    }
    let f = |n: u32| ((n as f32 * s).round() as u32).clamp(1, n.max(1));
    (f(width), f(height))
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Sub-pixel offset for `frame`, in pixels within (-0.5, 0.5).
pub fn jitter_pixels(frame: u32) -> Vec2 {
    let i = frame % JITTER_PERIOD + 1;
    Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5)
}

/// Shift a view-projection by `jitter` (NDC units) after projection.
pub fn jittered(vp: Mat4, jitter: Vec2) -> Mat4 {
    Mat4::from_translation(glam::Vec3::new(jitter.x, jitter.y, 0.0)) * vp
}

struct History {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub struct Taa {
    pipeline: wgpu::RenderPipeline,
    /// The upscaling resolve, used when the scene is smaller than the window.
    upscale: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    history: [History; 2],
    /// Bind groups reading history `i` and writing `1 - i`.
    binds: Option<[wgpu::BindGroup; 2]>,
    /// History buffer that holds the last result.
    current: usize,
    /// Whether `history[current]` holds a usable frame.
    valid: bool,
    prev_vp: Option<Mat4>,
    /// Depth-only view of the scene depth (stencil cannot be bound).
    depth_view: wgpu::TextureView,
    /// Window (output and history) size.
    width: u32,
    height: u32,
    /// Whether the scene is drawn smaller than the window.
    upscaling: bool,
}

impl Taa {
    /// `depth` is the scene's depth (scene-sized); `width` x `height` is the
    /// window, the size the result and its history have.
    pub fn new(device: &wgpu::Device, depth: &wgpu::Texture, width: u32, height: u32) -> Self {
        let multisampled = depth.sample_count() > 1;
        let depth_type = if multisampled {
            "texture_depth_multisampled_2d"
        } else {
            "texture_depth_2d"
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.replace("DEPTH_TYPE", depth_type).into()),
        });
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa-layout"),
            entries: &[
                tex(0),
                tex(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let make = |label: &str, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            })
        };
        let pipeline = make("taa-resolve", "fs_taa");
        let upscale = make("taa-upscale", "fs_taau");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa-uniform"),
            size: std::mem::size_of::<TaaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let upscaling = depth.width() != width.max(1) || depth.height() != height.max(1);
        Self {
            pipeline,
            upscale,
            layout,
            sampler,
            uniform,
            history: [
                history(device, width, height),
                history(device, width, height),
            ],
            binds: None,
            current: 0,
            valid: false,
            prev_vp: None,
            depth_view: depth_only(depth),
            width,
            height,
            upscaling,
        }
    }

    /// New window size: fresh history (bind groups follow in `encode`).
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        depth: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        self.depth_view = depth_only(depth);
        self.history = [
            history(device, width, height),
            history(device, width, height),
        ];
        self.binds = None;
        self.valid = false;
        self.width = width;
        self.height = height;
        self.upscaling = depth.width() != width.max(1) || depth.height() != height.max(1);
    }

    /// Whether the scene is smaller than the window, so this pass must run
    /// every frame (with TAA off too) to bring it up to the window's size.
    pub fn upscaling(&self) -> bool {
        self.upscaling
    }

    /// Forget the history (camera cut, TAA switched off).
    pub fn reset(&mut self) {
        self.valid = false;
        self.prev_vp = None;
    }

    /// Resolve `scene` against the history and copy the result into
    /// `output` (the scene texture itself, or the window-sized target when
    /// upscaling), so later passes read the anti-aliased frame.
    /// `vp` is this frame's unjittered view-projection. With `temporal`
    /// off (TAA switched off while upscaling) no history is used or kept:
    /// the frame is only filtered up to the window's size.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        output: &wgpu::Texture,
        vp: Mat4,
        jitter_ndc: Vec2,
        temporal: bool,
    ) {
        if self.binds.is_none() {
            let make = |read: usize| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("taa-bind"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(scene_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.history[read].view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.uniform.as_entire_binding(),
                        },
                    ],
                })
            };
            self.binds = Some([make(0), make(1)]);
        }
        let prev_vp = self.prev_vp.unwrap_or(vp);
        let reset = !temporal || !self.valid || !mat_finite(vp) || !mat_finite(vp.inverse());
        let params = TaaUniform {
            inv_vp: vp.inverse().to_cols_array_2d(),
            prev_vp: prev_vp.to_cols_array_2d(),
            jitter: jitter_ndc.into(),
            reset: if reset { 1.0 } else { 0.0 },
            blend: BLEND,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        let read = self.current;
        let write = 1 - read;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa-resolve-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.history[write].view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(if self.upscaling {
                &self.upscale
            } else {
                &self.pipeline
            });
            let binds = self.binds.as_ref().expect("taa binds");
            pass.set_bind_group(0, &binds[read], &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            self.history[write].texture.as_image_copy(),
            output.as_image_copy(),
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.current = write;
        self.valid = temporal;
        self.prev_vp = temporal.then_some(vp);
    }
}

fn depth_only(depth: &wgpu::Texture) -> wgpu::TextureView {
    depth.create_view(&wgpu::TextureViewDescriptor {
        label: Some("taa-depth"),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    })
}

fn mat_finite(m: Mat4) -> bool {
    m.to_cols_array().iter().all(|v| v.is_finite())
}

fn history(device: &wgpu::Device, width: u32, height: u32) -> History {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("taa-history"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    History { texture, view }
}

/// CPU mirror of the shader's reprojection, for tests.
#[cfg(test)]
fn reproject(inv_vp: Mat4, prev_vp: Mat4, ndc: Vec2, depth: f32) -> Option<Vec2> {
    let world = inv_vp * glam::Vec4::new(ndc.x, ndc.y, depth, 1.0);
    let prev = prev_vp * world;
    (prev.w.abs() > 1e-8).then(|| Vec2::new(prev.x, prev.y) / prev.w)
}

const SHADER: &str = r#"
struct Taa {
    inv_vp: mat4x4<f32>,
    prev_vp: mat4x4<f32>,
    jitter: vec2<f32>,
    reset: f32,
    blend: f32,
};
@group(0) @binding(0) var cur_tex: texture_2d<f32>;
@group(0) @binding(1) var hist_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: DEPTH_TYPE;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var<uniform> t: Taa;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VOut {
    let pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    o.clip = vec4(pos[i], 0.0, 1.0);
    o.uv = pos[i] * vec2(0.5, -0.5) + vec2(0.5);
    return o;
}

// A colour with a NaN or infinite channel (a broken shader input upstream)
// becomes black, so the history never keeps it and the neighbourhood range
// never spreads it to the pixels around. Tested on the bits: a compiler
// may assume floats are finite and drop an `x != x` test.
fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn clean(c: vec3<f32>) -> vec3<f32> {
    if finite(c.r) && finite(c.g) && finite(c.b) {
        return c;
    }
    return vec3<f32>(0.0);
}

// Blend in a tone-mapped space so one hot pixel cannot dominate the history.
fn squash(c: vec3<f32>) -> vec3<f32> {
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}
fn unsquash(c: vec3<f32>) -> vec3<f32> {
    return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-4);
}

// Bicubic Catmull-Rom history fetch in five bilinear taps (Jimenez 2016):
// keeps the history sharp where plain bilinear would blur it every frame.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(hist_tex));
    let pos = uv * size;
    let c1 = floor(pos - 0.5) + 0.5;
    let f = pos - c1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let tc12 = (c1 + w2 / w12) / size;
    let tc0 = (c1 - 1.0) / size;
    let tc3 = (c1 + 2.0) / size;
    var c = textureSampleLevel(hist_tex, lin, vec2(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y)
        + textureSampleLevel(hist_tex, lin, vec2(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y)
        + textureSampleLevel(hist_tex, lin, tc12, 0.0).rgb * (w12.x * w12.y)
        + textureSampleLevel(hist_tex, lin, vec2(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y)
        + textureSampleLevel(hist_tex, lin, vec2(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);
    let wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(clean(c / wsum), vec3<f32>(0.0));
}

@fragment
fn fs_taa(i: VOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(cur_tex));
    let px = vec2<i32>(i.clip.xy);
    let cur = clean(textureLoad(cur_tex, px, 0).rgb);
    if t.reset > 0.5 {
        return vec4(cur, 1.0);
    }
    // Colour range of the 3x3 neighbourhood (mean +- 1.25 sigma), and the
    // nearest depth in it, so edges reproject with the foreground.
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var depth = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(px + vec2<i32>(x, y), vec2<i32>(0), size - 1);
            let c = squash(clean(textureLoad(cur_tex, q, 0).rgb));
            m1 += c;
            m2 += c * c;
            // Reversed Z: larger is nearer.
            depth = max(depth, textureLoad(depth_tex, q, 0));
        }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let lo = mean - sigma * 1.25;
    let hi = mean + sigma * 1.25;

    let ndc = vec2<f32>(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0) - t.jitter;
    let world = t.inv_vp * vec4<f32>(ndc, depth, 1.0);
    let prev = t.prev_vp * world;
    if abs(prev.w) < 1e-8 {
        return vec4(cur, 1.0);
    }
    let prev_uv = (prev.xy / prev.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)) {
        return vec4(cur, 1.0);
    }
    let hist = clamp(squash(sample_history(prev_uv)), lo, hi);
    let out = mix(hist, squash(cur), t.blend);
    return vec4(unsquash(out), 1.0);
}

// Upscaling resolve: the scene is smaller than the window (and history).
// Each scene pixel q was drawn at its centre shifted by the jitter, so it
// shows the unjittered spot q + 0.5 - jitter (in scene pixels). The window
// pixel takes the samples of the 3x3 scene pixels around its own centre,
// weighted by a Gaussian of their distance, and gives the new frame more
// weight in the history the closer its nearest sample lies.
@fragment
fn fs_taau(i: VOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(cur_tex));
    let in_size = vec2<f32>(size);
    // Jitter in scene pixels, y down like the texture.
    let jit = t.jitter * vec2<f32>(0.5, -0.5) * in_size;
    let p = i.uv * in_size;
    let q0 = vec2<i32>(floor(p + jit));
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    var wmax = 0.0;
    var depth = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(q0 + vec2<i32>(x, y), vec2<i32>(0), size - 1);
            let c = squash(clean(textureLoad(cur_tex, q, 0).rgb));
            let d = vec2<f32>(q) + 0.5 - jit - p;
            let w = exp(-2.29 * dot(d, d));
            sum += c * w;
            wsum += w;
            wmax = max(wmax, w);
            m1 += c;
            m2 += c * c;
            depth = max(depth, textureLoad(depth_tex, q, 0));
        }
    }
    let cur = sum / max(wsum, 1e-5);
    if t.reset > 0.5 {
        return vec4(unsquash(cur), 1.0);
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    let lo = min(mean - sigma * 1.25, cur);
    let hi = max(mean + sigma * 1.25, cur);
    // This pixel's own unjittered centre.
    let ndc = vec2<f32>(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
    let world = t.inv_vp * vec4<f32>(ndc, depth, 1.0);
    let prev = t.prev_vp * world;
    if abs(prev.w) < 1e-8 {
        return vec4(unsquash(cur), 1.0);
    }
    let prev_uv = (prev.xy / prev.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)) {
        return vec4(unsquash(cur), 1.0);
    }
    let hist = clamp(squash(sample_history(prev_uv)), lo, hi);
    let alpha = clamp(t.blend * mix(0.3, 1.5, wmax), 0.02, 0.5);
    return vec4(unsquash(mix(hist, cur, alpha)), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn the_resolve_shader_is_valid_wgsl() {
        for depth in ["texture_depth_2d", "texture_depth_multisampled_2d"] {
            crate::render::assert_valid_wgsl(&super::SHADER.replace("DEPTH_TYPE", depth));
        }
    }
    use super::*;
    use glam::{Vec3, Vec4};

    #[test]
    fn scaled_sizes_round_and_clamp() {
        assert_eq!(scaled_size(1920, 1080, 1.0), (1920, 1080));
        assert_eq!(scaled_size(1920, 1080, 2.0), (1920, 1080));
        assert_eq!(scaled_size(3840, 2160, 0.5), (1920, 1080));
        assert_eq!(scaled_size(3840, 2160, 0.1), (1920, 1080));
        assert_eq!(scaled_size(2560, 1440, 0.75), (1920, 1080));
        assert_eq!(scaled_size(1, 1, 0.5), (1, 1));
    }

    #[test]
    fn jitter_stays_inside_the_pixel_and_covers_it() {
        let pts: Vec<Vec2> = (0..JITTER_PERIOD).map(jitter_pixels).collect();
        assert!(pts.iter().all(|p| p.x.abs() < 0.5 && p.y.abs() < 0.5));
        let mean = pts.iter().copied().sum::<Vec2>() / pts.len() as f32;
        assert!(mean.length() < 0.1, "jitter is centred: {mean}");
        assert_ne!(jitter_pixels(0), jitter_pixels(1));
        assert_eq!(jitter_pixels(3), jitter_pixels(3 + JITTER_PERIOD));
    }

    #[test]
    fn reprojection_follows_the_camera() {
        let proj =
            glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 16.0 / 9.0, 0.1);
        let view_a = glam::camera::rh::view::look_at_mat4(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, -10.0),
            Vec3::Y,
        );
        let view_b = glam::camera::rh::view::look_at_mat4(
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 2.0, -10.0),
            Vec3::Y,
        );
        let (a, b) = (proj * view_a, proj * view_b);
        let p = Vec3::new(0.5, 1.5, -6.0);
        let clip = b * p.extend(1.0);
        let ndc = Vec2::new(clip.x, clip.y) / clip.w;
        let depth = clip.z / clip.w;
        let then = a * p.extend(1.0);
        let want = Vec2::new(then.x, then.y) / then.w;
        let got = reproject(b.inverse(), a, ndc, depth).expect("reprojects");
        assert!((got - want).length() < 1e-4, "{got} vs {want}");
        // The sky (depth 0) reprojects by rotation alone.
        assert!(reproject(b.inverse(), a, Vec2::ZERO, 0.0).is_some());
    }

    #[test]
    fn jitter_shifts_by_whole_ndc_offset() {
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.1);
        let j = Vec2::new(0.01, -0.02);
        let p = Vec4::new(0.3, -0.2, -5.0, 1.0);
        let (c0, c1) = (proj * p, jittered(proj, j) * p);
        let d = Vec2::new(c1.x / c1.w - c0.x / c0.w, c1.y / c1.w - c0.y / c0.w);
        assert!((d - j).length() < 1e-5);
    }
}
