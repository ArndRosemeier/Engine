//! Mip levels made on the GPU.
//!
//! A material map used to get its mip chain on the CPU, in the frame that
//! first drew it: a 2048² map took tens of milliseconds, and a character
//! with several maps stopped a frame for a fifth of a second. Now only
//! level 0 is copied up; each smaller level is drawn from the one above by
//! a tiny render pass. Colour is averaged in linear light (the sRGB view
//! decodes on read and encodes on write), normals are renormalised.
//!
//! The pipelines are built with the renderer (before the first frame) and
//! registered per device, like the pipeline cache, so `upload_map` finds
//! them without new arguments. Without them (tools, tests) the CPU chain
//! is used as before.

use std::sync::{Arc, RwLock};

use super::material::MapKind;

const SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

// The linear filter at the centre of a target texel averages the 2x2 source
// texels under it.
@fragment
fn fs_average(v: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, smp, v.uv, 0.0);
}

@fragment
fn fs_normal(v: VsOut) -> @location(0) vec4<f32> {
    let s = textureSampleLevel(src, smp, v.uv, 0.0);
    var n = s.xyz * 2.0 - 1.0;
    let len = length(n);
    if (len > 1e-5) {
        n = n / len;
    } else {
        n = vec3<f32>(0.0, 0.0, 1.0);
    }
    return vec4<f32>(n * 0.5 + 0.5, s.w);
}
"#;

/// The three ways a map is reduced: sRGB colour, linear data, normals.
pub(crate) struct MipGen {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    color: wgpu::RenderPipeline,
    data: wgpu::RenderPipeline,
    normal: wgpu::RenderPipeline,
}

static GENERATORS: RwLock<Vec<(wgpu::Device, Arc<MipGen>)>> = RwLock::new(Vec::new());

/// The generator registered for `device`, if any.
pub(crate) fn get(device: &wgpu::Device) -> Option<Arc<MipGen>> {
    let all = GENERATORS.read().unwrap_or_else(|e| e.into_inner());
    all.iter()
        .find(|(d, _)| d == device)
        .map(|(_, g)| g.clone())
}

/// Build the pipelines for `device` and register them until the returned
/// guard is dropped (with the renderer).
pub(crate) fn register(device: &wgpu::Device) -> Registration {
    let generator = Arc::new(MipGen::new(device));
    GENERATORS
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .push((device.clone(), generator));
    Registration {
        device: device.clone(),
    }
}

/// Keeps a device's generator registered; dropping it forgets it.
pub(crate) struct Registration {
    device: wgpu::Device,
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut all = GENERATORS.write().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = all.iter().position(|(d, _)| d == &self.device) {
            all.remove(i);
        }
    }
}

impl MipGen {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mipgen-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mipgen-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mipgen-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let make = |label: &str, entry: &str, format: wgpu::TextureFormat| {
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mipgen-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            color: make(
                "mipgen-color",
                "fs_average",
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ),
            data: make("mipgen-data", "fs_average", wgpu::TextureFormat::Rgba8Unorm),
            normal: make(
                "mipgen-normal",
                "fs_normal",
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            layout,
            sampler,
        }
    }

    /// Fill levels 1.. of `texture` (level 0 already written) and submit the
    /// work. Queued writes made before this call land first.
    pub(crate) fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        kind: MapKind,
    ) {
        let levels = texture.mip_level_count();
        if levels <= 1 {
            return;
        }
        let pipeline = match kind {
            MapKind::Color => &self.color,
            MapKind::Data => &self.data,
            MapKind::Normal => &self.normal,
        };
        let view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mipgen-level"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mipgen"),
        });
        let mut src = view(0);
        for level in 1..levels {
            let dst = view(level);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mipgen-bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mipgen-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &dst,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
            }
            src = dst;
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::AlbedoMap;
    use crate::render::material::{mip_chain, upload_map};

    fn read_level(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        level: u32,
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        let row = (w * 4).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));
        buf.slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::Wait).expect("poll");
        let data = buf.slice(..).get_mapped_range();
        let mut out = Vec::new();
        for y in 0..h {
            let s = (y * row) as usize;
            out.extend_from_slice(&data[s..s + (w * 4) as usize]);
        }
        out
    }

    /// The GPU's levels match the CPU chain they replace (within rounding).
    #[test]
    fn gpu_mips_match_the_cpu_chain() {
        if !crate::gpu_field::verify::gpu_tests_enabled() {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");
        let _reg = register(&device);
        let (w, h) = (64u32, 32u32);
        let mut seed = 12345u32;
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            for c in 0..4 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Normals: keep z positive so none cancel to zero.
                let v = (seed >> 24) as u8;
                rgba.push(if c == 2 { 128 + v / 2 } else { v });
            }
        }
        let map = AlbedoMap {
            width: w,
            height: h,
            rgba: rgba.into(),
        };
        for kind in [MapKind::Color, MapKind::Data, MapKind::Normal] {
            let texture = upload_map(&device, &queue, &map, kind);
            let cpu = mip_chain(&map, kind);
            assert_eq!(texture.mip_level_count() as usize, cpu.len());
            for (level, (lw, lh, want)) in cpu.iter().enumerate() {
                let got = read_level(&device, &queue, &texture, level as u32, *lw, *lh);
                let worst = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (*a as i32 - *b as i32).abs())
                    .max()
                    .unwrap_or(0);
                // Normals are renormalised after 8-bit filtering: a little more slack.
                let allowed = if kind == MapKind::Normal { 6 } else { 3 };
                assert!(worst <= allowed, "{kind:?} level {level}: off by {worst}");
            }
        }
    }
}
