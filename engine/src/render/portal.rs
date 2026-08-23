//! Stencil-masked portal passes: depth write, stencil incr/decr, depth clear.

use super::pipeline::scene_shader_prefix;
use super::stencil::{
    depth_clear_depth_stencil, portal_depth_write_depth_stencil, stencil_decr_depth_stencil,
    stencil_incr_depth_stencil,
};
use crate::mesh::{InstanceRaw, Vertex};

pub struct PortalGpu {
    #[expect(dead_code)]
    pub depth_write: wgpu::RenderPipeline,
    pub stencil_incr: wgpu::RenderPipeline,
    pub stencil_decr: wgpu::RenderPipeline,
    pub depth_clear: wgpu::RenderPipeline,
}

const MESH_FS: &str = r#"
@fragment
fn fs_main() {}
"#;

const MESH_VS: &str = r#"
/// Pull the opening slightly along its local +Z so coplanar floors do not
/// block stencil marking when depth testing is enabled.
const STENCIL_NUDGE_M: f32 = 0.025;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let outward = normalize(model[2].xyz);
    let world = model * vec4<f32>(v.position, 1.0) + vec4<f32>(outward * STENCIL_NUDGE_M, 0.0);
    var out: VsOut;
    out.clip = u.view_proj * world;
    return out;
}
"#;

const DEPTH_CLEAR_FS: &str = r#"
@fragment
fn fs_main() {}
"#;

const DEPTH_CLEAR_WGSL: &str = r#"
struct VsOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  var out: VsOut;
  out.clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
  return out;
}
"#;

impl PortalGpu {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        scene_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("portal-mesh-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}{MESH_VS}{MESH_FS}", scene_shader_prefix()).into(),
            ),
        });
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("portal-depth-clear-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{DEPTH_CLEAR_WGSL}{DEPTH_CLEAR_FS}").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("portal-pass-layout"),
            bind_group_layouts: &[scene_layout],
            push_constant_ranges: &[],
        });
        let mesh_vertex = wgpu::VertexState {
            module: &mesh_shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
            compilation_options: Default::default(),
        };
        Self {
            depth_write: mesh_pipeline(
                device,
                &layout,
                &mesh_vertex,
                format,
                portal_depth_write_depth_stencil(),
                wgpu::ColorWrites::empty(),
                "portal-depth-write",
            ),
            stencil_incr: mesh_pipeline(
                device,
                &layout,
                &mesh_vertex,
                format,
                stencil_incr_depth_stencil(),
                wgpu::ColorWrites::empty(),
                "portal-stencil-incr",
            ),
            stencil_decr: mesh_pipeline(
                device,
                &layout,
                &mesh_vertex,
                format,
                stencil_decr_depth_stencil(),
                wgpu::ColorWrites::empty(),
                "portal-stencil-decr",
            ),
            depth_clear: clear_pipeline(
                device,
                &clear_shader,
                format,
                depth_clear_depth_stencil(),
                "portal-depth-clear",
            ),
        }
    }

    pub fn resize(&mut self) {}
}

fn mesh_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vertex: &wgpu::VertexState,
    format: wgpu::TextureFormat,
    depth_stencil: wgpu::DepthStencilState,
    color_mask: wgpu::ColorWrites,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: vertex.clone(),
        fragment: Some(wgpu::FragmentState {
            module: vertex.module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: color_mask,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(depth_stencil),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn clear_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    depth_stencil: wgpu::DepthStencilState,
    label: &str,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("portal-clear-layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_stencil),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}
