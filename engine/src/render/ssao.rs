//! Screen-space ambient occlusion from the depth prepass.
//!
//! Half-resolution, in three full-screen passes:
//! 1. `fs_ao` reads the scene depth (sample 0 when multisampled), rebuilds the
//!    view-space position and normal, and checks a rotated spiral of
//!    neighbours inside `radius` metres for geometry above the surface. It
//!    writes occlusion and linear depth.
//! 2. and 3. a separable bilateral blur that only mixes pixels at similar
//!    depth, so contact shadows do not bleed across silhouettes.
//!
//! The result multiplies sky and ambient light only (see `pbr_shade`), never
//! the sun, which has its own shadows.

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

/// Occlusion and linear depth, filterable everywhere.
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const SAMPLE_COUNT: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AoParams {
    inv_proj: [[f32; 4]; 4],
    full_size: [f32; 2],
    half_size: [f32; 2],
    /// Full-resolution pixels covered by one metre at one metre depth.
    proj_scale: f32,
    radius: f32,
    intensity: f32,
    /// Metres a neighbour must rise above the surface to occlude it, against
    /// depth noise on flat surfaces.
    bias: f32,
}

const _: () = assert!(std::mem::size_of::<AoParams>() == 96);

/// What the occlusion should look like, from [`crate::world::AmbientOcclusion`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoFrame {
    pub projection: glam::Mat4,
    pub fov_y_radians: f32,
    pub radius_m: f32,
    pub intensity: f32,
}

struct Targets {
    /// Raw occlusion, then the final blurred result.
    a_view: wgpu::TextureView,
    b_view: wgpu::TextureView,
    ao_bind: wgpu::BindGroup,
    blur_h_bind: wgpu::BindGroup,
    blur_v_bind: wgpu::BindGroup,
    full: [u32; 2],
    half: [u32; 2],
}

pub struct Ssao {
    ao_pipeline: wgpu::RenderPipeline,
    blur_h: wgpu::RenderPipeline,
    blur_v: wgpu::RenderPipeline,
    ao_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    targets: Targets,
}

impl Ssao {
    pub fn new(device: &wgpu::Device, depth: &wgpu::Texture, width: u32, height: u32) -> Self {
        let multisampled = depth.sample_count() > 1;
        let depth_type = if multisampled {
            "texture_depth_multisampled_2d"
        } else {
            "texture_depth_2d"
        };
        let ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{COMMON_WGSL}{AO_WGSL}")
                    .replace("DEPTH_TYPE", depth_type)
                    .replace("SAMPLE_COUNT", &format!("{SAMPLE_COUNT}"))
                    .into(),
            ),
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao-blur-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{BLUR_WGSL}").into()),
        });
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ao_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao-layout"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
            ],
        });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssao-blur-layout"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let make = |label: &str,
                    shader: &wgpu::ShaderModule,
                    layout: &wgpu::BindGroupLayout,
                    entry: &str| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_full"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: AO_FORMAT,
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
        let ao_pipeline = make("ssao", &ao_shader, &ao_layout, "fs_ao");
        let blur_h = make("ssao-blur-h", &blur_shader, &blur_layout, "fs_blur_h");
        let blur_v = make("ssao-blur-v", &blur_shader, &blur_layout, "fs_blur_v");
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssao-params"),
            contents: bytemuck::bytes_of(&AoParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let targets = create_targets(
            device,
            &ao_layout,
            &blur_layout,
            &params,
            depth,
            width,
            height,
        );
        Self {
            ao_pipeline,
            blur_h,
            blur_v,
            ao_layout,
            blur_layout,
            params,
            targets,
        }
    }

    /// Follow the scene depth texture after a resize.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        depth: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        self.targets = create_targets(
            device,
            &self.ao_layout,
            &self.blur_layout,
            &self.params,
            depth,
            width,
            height,
        );
    }

    /// The blurred occlusion (red) the lit shaders sample.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.targets.a_view
    }

    pub fn prepare(&self, queue: &wgpu::Queue, frame: AoFrame) {
        let [w, h] = self.targets.full;
        let [hw, hh] = self.targets.half;
        let params = AoParams {
            inv_proj: frame.projection.inverse().to_cols_array_2d(),
            full_size: [w as f32, h as f32],
            half_size: [hw as f32, hh as f32],
            proj_scale: h as f32 / (2.0 * (frame.fov_y_radians * 0.5).tan()),
            radius: frame.radius_m,
            intensity: frame.intensity,
            bias: 0.01,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let t = &self.targets;
        let passes: [(
            &str,
            &wgpu::RenderPipeline,
            &wgpu::BindGroup,
            &wgpu::TextureView,
        ); 3] = [
            ("ssao", &self.ao_pipeline, &t.ao_bind, &t.a_view),
            ("ssao-blur-h", &self.blur_h, &t.blur_h_bind, &t.b_view),
            ("ssao-blur-v", &self.blur_v, &t.blur_v_bind, &t.a_view),
        ];
        for (label, pipeline, bind, target) in passes {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn create_targets(
    device: &wgpu::Device,
    ao_layout: &wgpu::BindGroupLayout,
    blur_layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    depth: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Targets {
    let half = [width.div_ceil(2).max(1), height.div_ceil(2).max(1)];
    let make = |label: &str| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: half[0],
                height: half[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: AO_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let a = make("ssao-a");
    let b = make("ssao-b");
    let a_view = a.create_view(&wgpu::TextureViewDescriptor::default());
    let b_view = b.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
        label: Some("ssao-depth"),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    let bind = |label: &str, layout: &wgpu::BindGroupLayout, view: &wgpu::TextureView| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        })
    };
    Targets {
        ao_bind: bind("ssao-bind", ao_layout, &depth_view),
        blur_h_bind: bind("ssao-blur-h-bind", blur_layout, &a_view),
        blur_v_bind: bind("ssao-blur-v-bind", blur_layout, &b_view),
        a_view,
        b_view,
        full: [width, height],
        half,
    }
}

const COMMON_WGSL: &str = r#"
struct AoParams {
    inv_proj: mat4x4<f32>,
    full_size: vec2<f32>,
    half_size: vec2<f32>,
    proj_scale: f32,
    radius: f32,
    intensity: f32,
    bias: f32,
};

@group(0) @binding(0) var<uniform> p: AoParams;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Linear depth stored for the sky, beyond any surface.
const FAR_DEPTH: f32 = 60000.0;
"#;

const AO_WGSL: &str = r#"
@group(0) @binding(1) var depth_tex: DEPTH_TYPE;

fn load_depth(px: vec2<i32>) -> f32 {
    let c = clamp(px, vec2<i32>(0), vec2<i32>(p.full_size) - vec2<i32>(1));
    return textureLoad(depth_tex, c, 0);
}

fn view_pos(px: vec2<i32>, d: f32) -> vec3<f32> {
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / p.full_size;
    let v = p.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    return v.xyz / v.w;
}

fn view_at(px: vec2<i32>) -> vec3<f32> {
    return view_pos(px, load_depth(px));
}

@fragment
fn fs_ao(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let cp = vec2<i32>(frag.xy) * 2;
    let d = load_depth(cp);
    // Reversed-Z: the sky clears to 0 and nothing occludes it.
    if d <= 0.0 {
        return vec4<f32>(1.0, FAR_DEPTH, 0.0, 1.0);
    }
    let pos = view_pos(cp, d);
    let depth = -pos.z;

    // Normal from the neighbour on the flatter side, so silhouettes do not
    // tilt it toward the background.
    let pr = view_at(cp + vec2<i32>(1, 0));
    let pl = view_at(cp - vec2<i32>(1, 0));
    let pd = view_at(cp + vec2<i32>(0, 1));
    let pu = view_at(cp - vec2<i32>(0, 1));
    let dx = select(pr - pos, pos - pl, abs(pl.z - pos.z) < abs(pr.z - pos.z));
    let dy = select(pd - pos, pos - pu, abs(pu.z - pos.z) < abs(pd.z - pos.z));
    var n = normalize(cross(dx, dy));
    if dot(n, pos) > 0.0 {
        n = -n;
    }

    let radius_px = min(p.radius * p.proj_scale / depth, p.full_size.y * 0.12);
    if radius_px < 1.5 {
        return vec4<f32>(1.0, depth, 0.0, 1.0);
    }
    // Interleaved gradient noise rotates the spiral per pixel; the blur
    // removes the pattern.
    let noise = fract(52.9829189 * fract(dot(frag.xy, vec2<f32>(0.06711056, 0.00583715))));
    let spin = noise * 6.2831853;
    let r2 = p.radius * p.radius;
    var sum = 0.0;
    for (var i = 0; i < SAMPLE_COUNT; i++) {
        let a = (f32(i) + 0.5) / f32(SAMPLE_COUNT);
        let angle = a * 7.0 * 6.2831853 + spin;
        let offset = vec2<f32>(cos(angle), sin(angle)) * (a * radius_px);
        let q = cp + vec2<i32>(round(offset));
        let v = view_at(q) - pos;
        let vv = dot(v, v);
        // Alchemy AO: height above the surface over squared distance, so
        // close occluders count most. The bias grows with depth, as the
        // reconstruction error does.
        let falloff = max(1.0 - vv / r2, 0.0);
        let rise = dot(v, n) - (p.bias + depth * 0.002);
        sum += max(rise, 0.0) / (vv + 0.01 * r2) * falloff;
    }
    let ao = clamp(1.0 - p.intensity * p.radius * sum / f32(SAMPLE_COUNT), 0.0, 1.0);
    return vec4<f32>(ao, depth, 0.0, 1.0);
}
"#;

const BLUR_WGSL: &str = r#"
@group(0) @binding(1) var src: texture_2d<f32>;

fn blur(frag: vec4<f32>, dir: vec2<i32>) -> vec4<f32> {
    let c = vec2<i32>(frag.xy);
    let last = vec2<i32>(p.half_size) - vec2<i32>(1);
    let centre = textureLoad(src, c, 0);
    let z = centre.g;
    var sum = centre.r;
    var weight = 1.0;
    for (var i = -4; i <= 4; i++) {
        if i == 0 {
            continue;
        }
        let s = textureLoad(src, clamp(c + dir * i, vec2<i32>(0), last), 0);
        let spatial = exp(-f32(i * i) / 12.5);
        let similar = max(1.0 - abs(s.g - z) / max(z, 0.1) * 25.0, 0.0);
        let w = spatial * similar;
        sum += s.r * w;
        weight += w;
    }
    return vec4<f32>(sum / weight, z, 0.0, 1.0);
}

@fragment
fn fs_blur_h(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return blur(frag, vec2<i32>(1, 0));
}

@fragment
fn fs_blur_v(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return blur(frag, vec2<i32>(0, 1));
}
"#;
