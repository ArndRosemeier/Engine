//! Off-screen destination view, composited through an opening in screen space.

use super::pipeline::{Uniforms, SCENE_WGSL};
use super::shadow::ShadowGpu;
use crate::mesh::{InstanceRaw, Vertex};
use wgpu::util::DeviceExt;

pub struct PortalGpu {
    pub pipeline: wgpu::RenderPipeline,
    color: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
    /// Own scene uniforms so the destination pass can keep a virtual camera
    /// after the main pass overwrites the shared frame buffer.
    pub scene_uniform_buf: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
}

impl PortalGpu {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        scene_layout: &wgpu::BindGroupLayout,
        shadow: &ShadowGpu,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("portal-color-layout"),
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("portal-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("portal-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{SCENE_WGSL}{SHADER}").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("portal-pipeline-layout"),
            bind_group_layouts: &[scene_layout, &layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("portal-composite"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: super::DEPTH_COMPARE,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let (color, color_view, depth, depth_view, bind_group) =
            create_targets(device, format, width, height, &layout, &sampler);
        let scene_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("portal-scene-uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::empty()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("portal-scene-bind"),
            layout: scene_layout,
            entries: &shadow.scene_bind_entries(scene_uniform_buf.as_entire_binding()),
        });
        Self {
            pipeline,
            color,
            color_view,
            depth,
            depth_view,
            bind_group,
            scene_uniform_buf,
            scene_bind_group,
            sampler,
            layout,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        let (color, color_view, depth, depth_view, bind_group) =
            create_targets(device, format, width, height, &self.layout, &self.sampler);
        self.color = color;
        self.color_view = color_view;
        self.depth = depth;
        self.depth_view = depth_view;
        self.bind_group = bind_group;
    }
}

fn create_targets(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::BindGroup,
) {
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("portal-color"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("portal-depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: super::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("portal-color-bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (color, color_view, depth, depth_view, bind_group)
}

const SHADER: &str = r#"
@group(1) @binding(0) var portal_tex: texture_2d<f32>;
@group(1) @binding(1) var portal_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.position, 1.0);
    var out: VsOut;
    out.clip = u.view_proj * world;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(portal_tex));
    let uv = in.clip.xy / dims;
    return textureSample(portal_tex, portal_samp, uv);
}
"#;
