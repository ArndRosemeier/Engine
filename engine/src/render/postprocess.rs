//! HDR scene post-processing: wide bloom from a mip chain, then filmic tone
//! mapping (AgX by default, ACES on request).
//!
//! Bloom follows the "next generation post processing" recipe from Call of
//! Duty: Advanced Warfare: the bright part of the scene is filtered down a
//! chain of half-size images (13-tap filter, Karis-weighted on the first step
//! so single hot pixels do not flicker), then added back up with a tent
//! filter. Each level spreads the light twice as far, so the sum is a soft,
//! wide glow instead of one tight halo.

use crate::grading::ColorGrade;
use crate::world::{BloomSettings, ToneMap};
use wgpu::util::DeviceExt;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Most bloom levels below the half-size first level. Six levels reach about
/// a sixtieth of the screen, a wide enough glow without washing the image.
const MAX_BLOOM_LEVELS: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostprocessUniform {
    threshold: f32,
    knee: f32,
    intensity: f32,
    exposure: f32,
    /// One texel of the image this pass reads.
    texel_size: [f32; 2],
    /// 0 = ACES, 1 = AgX.
    tone_map: f32,
    _pad: f32,
    /// White balance multiplier (from temperature and tint).
    balance: [f32; 3],
    contrast: f32,
    shadows: [f32; 3],
    saturation: f32,
    highlights: [f32; 3],
    vignette: f32,
}

struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn with_samples(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        samples: u32,
    ) -> Self {
        let usage = if samples > 1 {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        } else {
            // Copy destination: the TAA pass writes its result back here.
            // Copy source: water copies the scene to refract it.
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

/// The bloom mip chain: level 0 is half the window, each next level half again.
struct BloomChain {
    _texture: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    sizes: Vec<(u32, u32)>,
}

impl BloomChain {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let sizes = bloom_sizes(width, height);
        let (w0, h0) = sizes[0];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom-chain"),
            size: wgpu::Extent3d {
                width: w0,
                height: h0,
                depth_or_array_layers: 1,
            },
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let views = (0..sizes.len() as u32)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("bloom-level"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Self {
            _texture: texture,
            views,
            sizes,
        }
    }

    fn levels(&self) -> usize {
        self.views.len()
    }
}

/// Sizes of the bloom levels: half the window, then halving while both sides
/// keep at least 4 pixels (always at least one level).
fn bloom_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut w = (width / 2).max(1);
    let mut h = (height / 2).max(1);
    let mut sizes = vec![(w, h)];
    while sizes.len() < MAX_BLOOM_LEVELS as usize && w >= 8 && h >= 8 {
        w /= 2;
        h /= 2;
        sizes.push((w, h));
    }
    sizes
}

pub struct Postprocess {
    /// The world pass's colour, at the scene's (possibly scaled) size.
    scene: Target,
    /// Window-sized colour the upscaling resolve writes, when the scene is
    /// drawn smaller than the window; bloom and tone mapping read it.
    upscaled: Option<Target>,
    /// Multisampled world-pass colour, resolved into `scene`. None without MSAA.
    scene_msaa: Option<Target>,
    bloom: BloomChain,
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    /// Passes in order: one per downsample step, one per upsample step, then
    /// the composite. Each has its own uniform and bind group.
    passes: Vec<Pass>,
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    /// Window size (bloom and tone mapping).
    width: u32,
    height: u32,
    /// Scene size.
    scene_size: (u32, u32),
}

struct Pass {
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl Postprocess {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("postprocess-linear-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postprocess-layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postprocess-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let make = |label: &str,
                    entry: &str,
                    format: wgpu::TextureFormat,
                    blend: Option<wgpu::BlendState>| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
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
                        format,
                        blend,
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
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let prefilter = make("bloom-prefilter", "fs_prefilter", HDR_FORMAT, None);
        let down = make("bloom-down", "fs_down", HDR_FORMAT, None);
        let up = make("bloom-up", "fs_up", HDR_FORMAT, Some(additive));
        let composite = make("tone-map-composite", "fs_composite", surface_format, None);
        let scene = Target::with_samples(device, "hdr-scene-color", width, height, 1);
        let bloom = BloomChain::new(device, width, height);
        let passes = create_passes(device, &layout, &sampler, &scene, &bloom);
        Self {
            scene,
            upscaled: None,
            scene_msaa: create_msaa_target(device, width, height),
            bloom,
            sampler,
            layout,
            passes,
            prefilter,
            down,
            up,
            composite,
            width: width.max(1),
            height: height.max(1),
            scene_size: (width.max(1), height.max(1)),
        }
    }

    /// New sizes: the scene's (`scene`) and the window's. When the scene is
    /// smaller, a window-sized target receives the upscaled frame.
    pub fn resize(&mut self, device: &wgpu::Device, scene: (u32, u32), width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        let scene = (scene.0.max(1), scene.1.max(1));
        if self.width == width && self.height == height && self.scene_size == scene {
            return;
        }
        self.scene = Target::with_samples(device, "hdr-scene-color", scene.0, scene.1, 1);
        self.scene_msaa = create_msaa_target(device, scene.0, scene.1);
        self.upscaled = (scene != (width, height))
            .then(|| Target::with_samples(device, "hdr-upscaled-color", width, height, 1));
        self.bloom = BloomChain::new(device, width, height);
        self.passes = create_passes(
            device,
            &self.layout,
            &self.sampler,
            self.upscaled.as_ref().unwrap_or(&self.scene),
            &self.bloom,
        );
        self.width = width;
        self.height = height;
        self.scene_size = scene;
    }

    /// Where the TAA / upscaling resolve writes: the scene itself, or the
    /// window-sized target when the scene is drawn smaller.
    pub fn resolve_target(&self) -> &wgpu::Texture {
        &self.upscaled.as_ref().unwrap_or(&self.scene)._texture
    }

    /// The resolved HDR scene colour (what bloom and tone mapping read).
    pub fn scene_target(&self) -> (&wgpu::TextureView, &wgpu::Texture) {
        (&self.scene.view, &self.scene._texture)
    }

    /// Colour attachment for the world pass: `(view, resolve_target)`.
    pub fn scene_attachment(&self) -> (wgpu::TextureView, Option<wgpu::TextureView>) {
        match &self.scene_msaa {
            Some(msaa) => (msaa.view.clone(), Some(self.scene.view.clone())),
            None => (self.scene.view.clone(), None),
        }
    }

    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        settings: BloomSettings,
        grade: ColorGrade,
        tone_map: ToneMap,
    ) {
        let levels = self.bloom.levels();
        let bloom_on = settings.enabled() && settings.intensity() > 0.0;
        let texel = |(w, h): (u32, u32)| [1.0 / w as f32, 1.0 / h as f32];
        let base = PostprocessUniform {
            threshold: settings.threshold(),
            knee: settings.soft_knee(),
            // The upsampled chain is the sum of every level; dividing keeps
            // the overall strength independent of the level count.
            intensity: if bloom_on {
                settings.intensity() / levels as f32
            } else {
                0.0
            },
            exposure: settings.exposure() * grade.exposure.exp2(),
            texel_size: [0.0, 0.0],
            tone_map: match tone_map {
                ToneMap::Aces => 0.0,
                ToneMap::Agx => 1.0,
            },
            _pad: 0.0,
            balance: white_balance(grade.temperature, grade.tint).to_array(),
            contrast: grade.contrast,
            shadows: grade.shadows.to_array(),
            saturation: grade.saturation,
            highlights: grade.highlights.to_array(),
            vignette: grade.vignette,
        };
        let write = |pass: &Pass, texel_size: [f32; 2]| {
            let uniform = PostprocessUniform { texel_size, ..base };
            queue.write_buffer(&pass.uniform, 0, bytemuck::bytes_of(&uniform));
        };
        if bloom_on {
            // Down: scene -> level 0 -> level 1 -> ...
            for level in 0..levels {
                let pass = &self.passes[level];
                let source = if level == 0 {
                    (self.width, self.height)
                } else {
                    self.bloom.sizes[level - 1]
                };
                write(pass, texel(source));
                let pipeline = if level == 0 {
                    &self.prefilter
                } else {
                    &self.down
                };
                draw_fullscreen(
                    encoder,
                    "bloom-down-pass",
                    &self.bloom.views[level],
                    pipeline,
                    &pass.bind,
                    true,
                );
            }
            // Up: add each smaller level onto the next larger one.
            for step in 0..levels - 1 {
                let target = levels - 2 - step;
                let pass = &self.passes[levels + step];
                write(pass, texel(self.bloom.sizes[target + 1]));
                draw_fullscreen(
                    encoder,
                    "bloom-up-pass",
                    &self.bloom.views[target],
                    &self.up,
                    &pass.bind,
                    false,
                );
            }
        }
        let composite = self.passes.last().expect("composite pass");
        write(composite, texel((self.width, self.height)));
        draw_fullscreen(
            encoder,
            "tone-map-composite-pass",
            output,
            &self.composite,
            &composite.bind,
            true,
        );
    }
}

/// A colour multiplier for white balance: temperature warms (red up, blue
/// down) or cools, tint greens or magentas; overall brightness is kept.
pub(crate) fn white_balance(temperature: f32, tint: f32) -> glam::Vec3 {
    let c = glam::Vec3::new(
        1.0 + 0.22 * temperature - 0.08 * tint,
        1.0 + 0.16 * tint,
        1.0 - 0.22 * temperature - 0.08 * tint,
    )
    .max(glam::Vec3::splat(0.05));
    let luma = c.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
    c / luma
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_msaa_target(device: &wgpu::Device, width: u32, height: u32) -> Option<Target> {
    let samples = super::stencil::scene_samples();
    (samples > 1)
        .then(|| Target::with_samples(device, "hdr-scene-color-msaa", width, height, samples))
}

/// Uniforms and bind groups for every pass, in `encode`'s order.
fn create_passes(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    scene: &Target,
    bloom: &BloomChain,
) -> Vec<Pass> {
    let make = |label: &str, first: &wgpu::TextureView, second: &wgpu::TextureView| {
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&PostprocessUniform {
                threshold: 1.0,
                knee: 0.5,
                intensity: 0.0,
                exposure: 1.0,
                texel_size: [1.0, 1.0],
                tone_map: 1.0,
                _pad: 0.0,
                balance: [1.0; 3],
                contrast: 1.0,
                shadows: [1.0; 3],
                saturation: 1.0,
                highlights: [1.0; 3],
                vignette: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(first),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(second),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        Pass { uniform, bind }
    };
    let levels = bloom.levels();
    let mut passes = Vec::with_capacity(levels * 2);
    for level in 0..levels {
        let source = if level == 0 {
            &scene.view
        } else {
            &bloom.views[level - 1]
        };
        passes.push(make("bloom-down", source, source));
    }
    for step in 0..levels - 1 {
        let source = &bloom.views[levels - 1 - step];
        passes.push(make("bloom-up", source, source));
    }
    passes.push(make("tone-map-composite", &scene.view, &bloom.views[0]));
    passes
}

fn draw_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    output: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    clear: bool,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            resolve_target: None,
            ops: wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                } else {
                    wgpu::LoadOp::Load
                },
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

#[cfg(test)]
/// CPU copy of the shader's ACES curve (Narkowicz fit), for tests and tools.
pub(crate) fn aces_tone_map(color: glam::Vec3, exposure: f32) -> glam::Vec3 {
    let x = color * exposure;
    ((x * (2.51 * x + glam::Vec3::splat(0.03)))
        / (x * (2.43 * x + glam::Vec3::splat(0.59)) + glam::Vec3::splat(0.14)))
    .clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

#[cfg(test)]
/// AgX keeps bright colours from turning into flat cyan, yellow or magenta
/// blobs: they fade towards white the way film does. Without a correction it
/// is a little darker and greyer than ACES in mid tones, so the engine adds
/// this much exposure and a gentle "punchy" look on top.
const AGX_EXPOSURE: f32 = 1.25;

#[cfg(test)]
/// CPU copy of the shader's AgX curve, for tests and tools. Returns linear
/// display colour.
pub(crate) fn agx_tone_map(color: glam::Vec3, exposure: f32) -> glam::Vec3 {
    use glam::{Mat3, Vec3};
    // Inset into the AgX working space (columns, as in the shader).
    let inset = Mat3::from_cols(
        Vec3::new(0.842_479_06, 0.042_328_24, 0.042_375_655),
        Vec3::new(0.078_433_6, 0.878_468_64, 0.078_433_6),
        Vec3::new(0.079_223_745, 0.079_166_13, 0.879_143),
    );
    let outset = Mat3::from_cols(
        Vec3::new(1.196_879, -0.052_896_85, -0.052_971_635),
        Vec3::new(-0.098_020_88, 1.151_903_1, -0.098_043_45),
        Vec3::new(-0.099_029_74, -0.098_961_18, 1.151_073_7),
    );
    let (min_ev, max_ev) = (-12.473_93_f32, 4.026_069_f32);
    let v = inset * (color * exposure * AGX_EXPOSURE);
    let curve = |x: f32| {
        let x = ((x.max(1e-10).log2().clamp(min_ev, max_ev)) - min_ev) / (max_ev - min_ev);
        let x2 = x * x;
        let x4 = x2 * x2;
        15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
            - 0.00232
    };
    let v = Vec3::new(curve(v.x), curve(v.y), curve(v.z));
    let v = agx_look(v);
    let v = outset * v;
    Vec3::new(
        v.x.max(0.0).powf(2.2),
        v.y.max(0.0).powf(2.2),
        v.z.max(0.0).powf(2.2),
    )
    .clamp(Vec3::ZERO, Vec3::ONE)
}

#[cfg(test)]
/// The look applied in AgX's display space: a touch more contrast and
/// saturation than the neutral base (between Blender's "Base" and "Punchy").
fn agx_look(v: glam::Vec3) -> glam::Vec3 {
    let luma = v.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
    let v = glam::Vec3::new(
        v.x.max(0.0).powf(1.12),
        v.y.max(0.0).powf(1.12),
        v.z.max(0.0).powf(1.12),
    );
    let luma_c = luma.max(0.0).powf(1.12);
    glam::Vec3::splat(luma_c) + (v - glam::Vec3::splat(luma_c)) * 1.15
}

const SHADER: &str = r#"
const AGX_EXPOSURE:f32=1.25;
struct Params { threshold:f32, knee:f32, intensity:f32, exposure:f32, texel_size:vec2<f32>, tone_map:f32, pad:f32,
    balance:vec3<f32>, contrast:f32, shadows:vec3<f32>, saturation:f32, highlights:vec3<f32>, vignette:f32 };
@group(0) @binding(0) var primary_tex:texture_2d<f32>;
@group(0) @binding(1) var secondary_tex:texture_2d<f32>;
@group(0) @binding(2) var linear_sampler:sampler;
@group(0) @binding(3) var<uniform> p:Params;
struct VOut { @builtin(position) clip:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) i:u32)->VOut {
    let pos=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
    var o:VOut; o.clip=vec4(pos[i],0.0,1.0); o.uv=pos[i]*vec2(0.5,-0.5)+vec2(0.5); return o;
}

fn tap(uv:vec2<f32>, x:f32, y:f32)->vec3<f32> {
    return textureSampleLevel(primary_tex, linear_sampler, uv + vec2(x, y) * p.texel_size, 0.0).rgb;
}
fn luma(c:vec3<f32>)->f32 { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// Karis average weight: hot pixels count less, so they cannot flicker.
fn karis(c:vec3<f32>)->f32 { return 1.0 / (1.0 + luma(c)); }

// 13 taps in five overlapping 2x2 boxes (Jimenez 2014).
struct Taps { a:vec3<f32>, b:vec3<f32>, c:vec3<f32>, d:vec3<f32>, e:vec3<f32>, f:vec3<f32>, g:vec3<f32>,
              h:vec3<f32>, i:vec3<f32>, j:vec3<f32>, k:vec3<f32>, l:vec3<f32>, m:vec3<f32> };
fn taps13(uv:vec2<f32>)->Taps {
    var t:Taps;
    t.a=tap(uv,-2.0,-2.0); t.b=tap(uv,0.0,-2.0); t.c=tap(uv,2.0,-2.0);
    t.d=tap(uv,-2.0,0.0);  t.e=tap(uv,0.0,0.0);  t.f=tap(uv,2.0,0.0);
    t.g=tap(uv,-2.0,2.0);  t.h=tap(uv,0.0,2.0);  t.i=tap(uv,2.0,2.0);
    t.j=tap(uv,-1.0,-1.0); t.k=tap(uv,1.0,-1.0); t.l=tap(uv,-1.0,1.0); t.m=tap(uv,1.0,1.0);
    return t;
}

// First step: threshold the scene with a soft knee, Karis-weighted.
@fragment fn fs_prefilter(v:VOut)->@location(0) vec4<f32> {
    let t=taps13(v.uv);
    let g0=(t.j+t.k+t.l+t.m)*0.25;
    let g1=(t.a+t.b+t.d+t.e)*0.25;
    let g2=(t.b+t.c+t.e+t.f)*0.25;
    let g3=(t.d+t.e+t.g+t.h)*0.25;
    let g4=(t.e+t.f+t.h+t.i)*0.25;
    let w0=karis(g0)*0.5; let w1=karis(g1)*0.125; let w2=karis(g2)*0.125; let w3=karis(g3)*0.125; let w4=karis(g4)*0.125;
    var c=(g0*w0+g1*w1+g2*w2+g3*w3+g4*w4)/max(w0+w1+w2+w3+w4, 1e-5);
    c=min(c, vec3(64.0));
    let brightness=max(c.r,max(c.g,c.b));
    let knee=max(p.knee*p.threshold,1e-4);
    let soft=clamp(brightness-p.threshold+knee,0.0,2.0*knee);
    let contribution=max(brightness-p.threshold, soft*soft/(4.0*knee));
    return vec4(c*(contribution/max(brightness,1e-4)),1.0);
}

@fragment fn fs_down(v:VOut)->@location(0) vec4<f32> {
    let t=taps13(v.uv);
    let c=t.e*0.125+(t.a+t.c+t.g+t.i)*0.03125+(t.b+t.d+t.f+t.h)*0.0625+(t.j+t.k+t.l+t.m)*0.125;
    return vec4(c,1.0);
}

// 3x3 tent from the smaller level, added onto the larger one by blending.
@fragment fn fs_up(v:VOut)->@location(0) vec4<f32> {
    var c=tap(v.uv,0.0,0.0)*4.0;
    c+=(tap(v.uv,-1.0,0.0)+tap(v.uv,1.0,0.0)+tap(v.uv,0.0,-1.0)+tap(v.uv,0.0,1.0))*2.0;
    c+=tap(v.uv,-1.0,-1.0)+tap(v.uv,1.0,-1.0)+tap(v.uv,-1.0,1.0)+tap(v.uv,1.0,1.0);
    return vec4(c/16.0,1.0);
}

fn aces(x:vec3<f32>)->vec3<f32>{return clamp((x*(2.51*x+vec3(0.03)))/(x*(2.43*x+vec3(0.59))+vec3(0.14)),vec3(0.0),vec3(1.0));}

// AgX (Troy Sobotka), with the polynomial fit by Benjamin Wrensch.
fn agx_curve(x:vec3<f32>)->vec3<f32> {
    let x2=x*x; let x4=x2*x2;
    return 15.5*x4*x2-40.14*x4*x+31.96*x4-6.868*x2*x+0.4298*x2+0.1191*x-0.00232;
}
fn agx(color:vec3<f32>)->vec3<f32> {
    let inset=mat3x3<f32>(
        vec3(0.842479062253094,0.0423282422610123,0.0423756549057051),
        vec3(0.0784335999999992,0.878468636469772,0.0784336),
        vec3(0.0792237451477643,0.0791661274605434,0.879142973793104));
    let outset=mat3x3<f32>(
        vec3(1.19687900512017,-0.0528968517574562,-0.0529716355144438),
        vec3(-0.0980208811401368,1.15190312990417,-0.0980434501171241),
        vec3(-0.0990297440797205,-0.0989611768448433,1.15107367264116));
    let min_ev=-12.47393; let max_ev=4.026069;
    var v=inset*color;
    v=(clamp(log2(max(v,vec3(1e-10))),vec3(min_ev),vec3(max_ev))-min_ev)/(max_ev-min_ev);
    v=agx_curve(v);
    // Look: a touch more contrast and saturation than the neutral base.
    let l=pow(max(luma(v),0.0),1.12);
    v=vec3(l)+(pow(max(v,vec3(0.0)),vec3(1.12))-vec3(l))*1.15;
    v=outset*v;
    // AgX ends in display encoding; the surface wants linear.
    return clamp(pow(max(v,vec3(0.0)),vec3(2.2)),vec3(0.0),vec3(1.0));
}

@fragment fn fs_composite(v:VOut)->@location(0) vec4<f32> {
    let scene=textureSampleLevel(primary_tex,linear_sampler,v.uv,0.0).rgb;
    let bloom=textureSampleLevel(secondary_tex,linear_sampler,v.uv,0.0).rgb;
    let hdr=(scene+bloom*p.intensity)*p.exposure*p.balance;
    var c:vec3<f32>;
    if p.tone_map>0.5 { c=agx(hdr*AGX_EXPOSURE); } else { c=aces(hdr); }
    return vec4(grade(c,v.uv),1.0);
}
// The look after tone mapping (linear 0..1): contrast around mid grey,
// saturation, shadow and highlight tints, vignette. Neutral leaves c as is.
fn grade(c_in:vec3<f32>, uv:vec2<f32>)->vec3<f32> {
    var c=max(c_in,vec3(0.0));
    c=0.18*pow(c/0.18,vec3(p.contrast));
    let l=dot(c,vec3(0.2126,0.7152,0.0722));
    c=max(mix(vec3(l),c,p.saturation),vec3(0.0));
    let w=smoothstep(0.05,0.6,l);
    c=c*mix(p.shadows,p.highlights,w);
    let d=length((uv-0.5)*vec2(1.0,0.75))*1.6;
    c=c*(1.0-p.vignette*smoothstep(0.35,1.0,d));
    return clamp(c,vec3(0.0),vec3(1.0));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn bounded_and_monotonic(curve: fn(glam::Vec3, f32) -> glam::Vec3) {
        let black = curve(glam::Vec3::ZERO, 1.0);
        let grey = curve(glam::Vec3::splat(0.18), 1.0);
        let middle = curve(glam::Vec3::splat(1.0), 1.0);
        let hot = curve(glam::Vec3::splat(16.0), 1.0);
        assert!(black.cmple(glam::Vec3::splat(1e-3)).all(), "{black}");
        assert!(grey.cmpgt(black).all());
        assert!(middle.cmpgt(grey).all());
        assert!(hot.cmpgt(middle).all());
        assert!(hot.cmple(glam::Vec3::ONE).all());
    }

    #[test]
    fn aces_is_bounded_and_monotonic() {
        bounded_and_monotonic(aces_tone_map);
        assert_eq!(aces_tone_map(glam::Vec3::ZERO, 1.0), glam::Vec3::ZERO);
    }

    #[test]
    fn shader_and_cpu_agx_share_the_exposure() {
        assert!(SHADER.contains(&format!("const AGX_EXPOSURE:f32={AGX_EXPOSURE};")));
    }

    #[test]
    fn agx_is_bounded_and_monotonic() {
        bounded_and_monotonic(agx_tone_map);
    }

    #[test]
    fn agx_fades_saturated_highlights_towards_white() {
        // A very bright pure red ends up less saturated than a dim one.
        let sat = |c: glam::Vec3| (c.max_element() - c.min_element()) / c.max_element().max(1e-5);
        let dim = agx_tone_map(glam::Vec3::new(0.3, 0.0, 0.0), 1.0);
        let hot = agx_tone_map(glam::Vec3::new(40.0, 0.0, 0.0), 1.0);
        assert!(sat(hot) < sat(dim) * 0.8, "dim {dim} hot {hot}");
    }

    #[test]
    fn bloom_levels_halve_down_to_a_few_pixels() {
        let sizes = bloom_sizes(1920, 1080);
        assert_eq!(sizes[0], (960, 540));
        assert_eq!(sizes.len(), MAX_BLOOM_LEVELS as usize);
        assert_eq!(sizes[1], (480, 270));
        assert_eq!(bloom_sizes(3, 3), vec![(1, 1)]);
        assert_eq!(bloom_sizes(20, 20).len(), 2);
    }
}

#[cfg(test)]
mod grade_tests {
    #[test]
    fn white_balance_is_neutral_at_zero_and_keeps_brightness() {
        let n = super::white_balance(0.0, 0.0);
        assert!((n - glam::Vec3::ONE).length() < 1e-5);
        let warm = super::white_balance(0.8, 0.2);
        assert!(warm.x > warm.z);
        let luma = warm.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
        assert!((luma - 1.0).abs() < 1e-5);
        assert_eq!(std::mem::size_of::<super::PostprocessUniform>(), 80);
    }
}
