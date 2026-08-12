use crate::mesh::{InstanceRaw, Vertex};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub ambient: f32,
    pub light_color: [f32; 3],
    pub _pad: f32,
    pub eye: [f32; 3],
    /// Seconds since start, for materials that animate.
    pub time: f32,
}

const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _pad: f32,
    eye: vec3<f32>,
    time: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.position, 1.0);
    var out: VsOut;
    out.clip = u.view_proj * world;
    // Assume uniform scale for normals (friendly default).
    out.world_n = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    out.color = v.color;
    out.world_p = world.xyz;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_n);
    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    // Soft wrap lighting — enough contrast for smooth heightfields to read as
    // hills. Sky and sun share one budget, so raising ambient fills the shadows
    // instead of blowing out everything the sun already reaches.
    let wrap = ndl * 0.65 + 0.35;
    let lit = in.color.rgb * (u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color);
    // Soft fresnel rim for translucent surfaces (keep grazing alpha modest so
    // water stays see-through from typical third-person angles).
    var alpha = in.color.a;
    if alpha < 0.999 {
        let view = normalize(u.eye - in.world_p);
        let fresnel = pow(1.0 - max(dot(n, view), 0.0), 2.0);
        alpha = mix(alpha, min(alpha + 0.18, 0.55), fresnel * 0.65);
    } else {
        alpha = 1.0;
    }
    return vec4<f32>(lit, alpha);
}
"#;

pub struct Pipelines {
    pub opaque: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_layout: wgpu::BindGroupLayout,
}

pub fn create_pipelines(device: &wgpu::Device, format: wgpu::TextureFormat) -> Pipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lit-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_dir: [0.0, 1.0, 0.0],
            ambient: 0.2,
            light_color: [1.0, 1.0, 1.0],
            _pad: 0.0,
            eye: [0.0, 0.0, 0.0],
            time: 0.0,
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform-bind"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let make =
        |label: &str, blend: wgpu::BlendState, depth_write: bool, cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: depth_write,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

    let opaque = make(
        "opaque-pipeline",
        wgpu::BlendState::REPLACE,
        true,
        Some(wgpu::Face::Back),
    );
    let transparent = make(
        "transparent-pipeline",
        wgpu::BlendState::ALPHA_BLENDING,
        false,
        None, // water/glass readable from both sides
    );

    Pipelines {
        opaque,
        transparent,
        uniform_buf,
        bind_group,
        bind_layout,
    }
}
