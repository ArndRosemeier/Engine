//! HDR scene post-processing: bloom extraction, blur, and filmic tone mapping.

use crate::world::BloomSettings;
use wgpu::util::DeviceExt;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostprocessUniform {
    threshold: f32,
    knee: f32,
    intensity: f32,
    exposure: f32,
    texel_size: [f32; 2],
    direction: [f32; 2],
}

struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn new(device: &wgpu::Device, label: &str, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

pub struct Postprocess {
    scene: Target,
    bloom_a: Target,
    bloom_b: Target,
    sampler: wgpu::Sampler,
    uniforms: [wgpu::Buffer; 4],
    layout: wgpu::BindGroupLayout,
    binds: [wgpu::BindGroup; 4],
    extract: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    width: u32,
    height: u32,
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
        let uniforms = std::array::from_fn(|index| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("postprocess-uniform-{index}")),
                contents: bytemuck::bytes_of(&PostprocessUniform {
                    threshold: 1.0,
                    knee: 0.5,
                    intensity: 0.35,
                    exposure: 1.0,
                    texel_size: [1.0, 1.0],
                    direction: [1.0, 0.0],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postprocess-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let make = |label: &str, entry: &str, format: wgpu::TextureFormat| {
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
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let extract = make("bloom-extract", "fs_extract", HDR_FORMAT);
        let blur = make("bloom-blur", "fs_blur", HDR_FORMAT);
        let composite = make("tone-map-composite", "fs_composite", surface_format);
        let (scene, bloom_a, bloom_b) = create_targets(device, width, height);
        let binds = create_binds(
            device, &layout, &sampler, &uniforms, &scene, &bloom_a, &bloom_b,
        );
        Self {
            scene,
            bloom_a,
            bloom_b,
            sampler,
            uniforms,
            layout,
            binds,
            extract,
            blur,
            composite,
            width: width.max(1),
            height: height.max(1),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.width == width && self.height == height {
            return;
        }
        (self.scene, self.bloom_a, self.bloom_b) = create_targets(device, width, height);
        self.binds = create_binds(
            device,
            &self.layout,
            &self.sampler,
            &self.uniforms,
            &self.scene,
            &self.bloom_a,
            &self.bloom_b,
        );
        self.width = width;
        self.height = height;
    }

    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.scene.view
    }

    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        settings: BloomSettings,
    ) {
        let bloom_width = (self.width / 2).max(1);
        let bloom_height = (self.height / 2).max(1);
        let base = PostprocessUniform {
            threshold: settings.threshold(),
            knee: settings.soft_knee(),
            intensity: if settings.enabled() {
                settings.intensity()
            } else {
                0.0
            },
            exposure: settings.exposure(),
            texel_size: [1.0 / bloom_width as f32, 1.0 / bloom_height as f32],
            direction: [0.0, 0.0],
        };
        let params = [
            base,
            PostprocessUniform {
                direction: [1.0, 0.0],
                ..base
            },
            PostprocessUniform {
                direction: [0.0, 1.0],
                ..base
            },
            base,
        ];
        for (buffer, params) in self.uniforms.iter().zip(params) {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&params));
        }
        draw_fullscreen(
            encoder,
            "bloom-extract-pass",
            &self.bloom_a.view,
            &self.extract,
            &self.binds[0],
        );
        draw_fullscreen(
            encoder,
            "bloom-horizontal-pass",
            &self.bloom_b.view,
            &self.blur,
            &self.binds[1],
        );
        draw_fullscreen(
            encoder,
            "bloom-vertical-pass",
            &self.bloom_a.view,
            &self.blur,
            &self.binds[2],
        );
        draw_fullscreen(
            encoder,
            "tone-map-composite-pass",
            output,
            &self.composite,
            &self.binds[3],
        );
    }
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

fn create_targets(device: &wgpu::Device, width: u32, height: u32) -> (Target, Target, Target) {
    let half_w = (width / 2).max(1);
    let half_h = (height / 2).max(1);
    (
        Target::new(device, "hdr-scene-color", width, height),
        Target::new(device, "bloom-half-a", half_w, half_h),
        Target::new(device, "bloom-half-b", half_w, half_h),
    )
}

fn create_binds(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniforms: &[wgpu::Buffer; 4],
    scene: &Target,
    bloom_a: &Target,
    bloom_b: &Target,
) -> [wgpu::BindGroup; 4] {
    let make = |label: &str,
                first: &wgpu::TextureView,
                second: &wgpu::TextureView,
                uniform: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        })
    };
    [
        make("bloom-extract-bind", &scene.view, &scene.view, &uniforms[0]),
        make(
            "bloom-horizontal-bind",
            &bloom_a.view,
            &bloom_a.view,
            &uniforms[1],
        ),
        make(
            "bloom-vertical-bind",
            &bloom_b.view,
            &bloom_b.view,
            &uniforms[2],
        ),
        make(
            "tone-map-composite-bind",
            &scene.view,
            &bloom_a.view,
            &uniforms[3],
        ),
    ]
}

fn draw_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    output: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
pub(crate) fn aces_tone_map(color: glam::Vec3, exposure: f32) -> glam::Vec3 {
    let x = color * exposure;
    ((x * (2.51 * x + glam::Vec3::splat(0.03)))
        / (x * (2.43 * x + glam::Vec3::splat(0.59)) + glam::Vec3::splat(0.14)))
    .clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}

const SHADER: &str = r#"
struct Params { threshold:f32, knee:f32, intensity:f32, exposure:f32, texel_size:vec2<f32>, direction:vec2<f32> };
@group(0) @binding(0) var primary_tex:texture_2d<f32>;
@group(0) @binding(1) var secondary_tex:texture_2d<f32>;
@group(0) @binding(2) var linear_sampler:sampler;
@group(0) @binding(3) var<uniform> p:Params;
struct VOut { @builtin(position) clip:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vs_main(@builtin(vertex_index) i:u32)->VOut { let pos=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0)); var o:VOut; o.clip=vec4(pos[i],0.0,1.0); o.uv=pos[i]*vec2(0.5,-0.5)+vec2(0.5); return o; }
@fragment fn fs_extract(i:VOut)->@location(0) vec4<f32> { let c=textureSample(primary_tex,linear_sampler,i.uv).rgb; let brightness=max(c.r,max(c.g,c.b)); let knee=max(p.knee,0.0001); let soft=clamp((brightness-p.threshold+knee)/(2.0*knee),0.0,1.0); let contribution=max(brightness-p.threshold,0.0)+soft*soft*knee; return vec4(c*(contribution/max(brightness,0.0001)),1.0); }
@fragment fn fs_blur(i:VOut)->@location(0) vec4<f32> { let d=p.direction*p.texel_size; var c=textureSample(primary_tex,linear_sampler,i.uv).rgb*0.227027; c+=(textureSample(primary_tex,linear_sampler,i.uv+d*1.384615).rgb+textureSample(primary_tex,linear_sampler,i.uv-d*1.384615).rgb)*0.316216; c+=(textureSample(primary_tex,linear_sampler,i.uv+d*3.230769).rgb+textureSample(primary_tex,linear_sampler,i.uv-d*3.230769).rgb)*0.070270; return vec4(c,1.0); }
fn aces(x:vec3<f32>)->vec3<f32>{return clamp((x*(2.51*x+vec3(0.03)))/(x*(2.43*x+vec3(0.59))+vec3(0.14)),vec3(0.0),vec3(1.0));}
@fragment fn fs_composite(i:VOut)->@location(0) vec4<f32> { let hdr=textureSample(primary_tex,linear_sampler,i.uv).rgb+textureSample(secondary_tex,linear_sampler,i.uv).rgb*p.intensity; return vec4(aces(hdr*p.exposure),1.0); }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aces_is_bounded_and_monotonic() {
        let black = aces_tone_map(glam::Vec3::ZERO, 1.0);
        let middle = aces_tone_map(glam::Vec3::splat(1.0), 1.0);
        let hot = aces_tone_map(glam::Vec3::splat(16.0), 1.0);
        assert_eq!(black, glam::Vec3::ZERO);
        assert!(middle.cmpgt(black).all());
        assert!(hot.cmpgt(middle).all());
        assert!(hot.cmple(glam::Vec3::ONE).all());
    }
}
