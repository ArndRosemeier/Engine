//! GPU skinning pipeline and mesh upload.

use crate::anim::{SkinMesh, MAX_JOINTS};
use crate::mesh::InstanceRaw;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

impl SkinnedVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SkinnedVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x4,
            3 => Uint16x4,
            4 => Float32x4,
        ],
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct JointPalette {
    pub joints: [[[f32; 4]; 4]; MAX_JOINTS],
}

impl JointPalette {
    pub fn from_matrices(mats: &[Mat4]) -> Self {
        let mut joints = [[[0.0; 4]; 4]; MAX_JOINTS];
        for (i, m) in mats.iter().take(MAX_JOINTS).enumerate() {
            joints[i] = m.to_cols_array_2d();
        }
        // Unused slots identity so stray weights don't explode.
        for slot in joints.iter_mut().skip(mats.len().min(MAX_JOINTS)) {
            *slot = Mat4::IDENTITY.to_cols_array_2d();
        }
        Self { joints }
    }
}

const SKINNED_SHADER: &str = r#"
struct Joints {
    m: array<mat4x4<f32>, 128>,
};

@group(1) @binding(0) var<uniform> bones: Joints;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
    @location(5) m0: vec4<f32>,
    @location(6) m1: vec4<f32>,
    @location(7) m2: vec4<f32>,
    @location(8) m3: vec4<f32>,
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
    var skin = mat4x4<f32>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
    );
    skin += bones.m[v.joints.x] * v.weights.x;
    skin += bones.m[v.joints.y] * v.weights.y;
    skin += bones.m[v.joints.z] * v.weights.z;
    skin += bones.m[v.joints.w] * v.weights.w;

    let skinned_pos = skin * vec4<f32>(v.position, 1.0);
    let skinned_n = skin * vec4<f32>(v.normal, 0.0);
    let world = model * skinned_pos;
    var out: VsOut;
    out.clip = u.view_proj * world;
    out.world_n = normalize((model * skinned_n).xyz);
    out.color = v.color;
    out.world_p = world.xyz;
    return out;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_n);
    if (!front) {
        n = -n;
    }
    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.5 + 0.5;
    let lit = in.color.rgb * (u.ambient + wrap * wrap * u.light_color);
    return vec4<f32>(haze(lit, in.world_p), 1.0);
}
"#;

pub struct SkinnedPipelines {
    pub opaque: wgpu::RenderPipeline,
    pub joint_bind_layout: wgpu::BindGroupLayout,
}

pub fn create_skinned_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    scene_bind_layout: &wgpu::BindGroupLayout,
) -> SkinnedPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinned-shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}{SKINNED_SHADER}", super::pipeline::SCENE_WGSL).into(),
        ),
    });

    let joint_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("joint-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("skinned-pipeline-layout"),
        bind_group_layouts: &[scene_bind_layout, &joint_bind_layout],
        push_constant_ranges: &[],
    });

    // Instance matrix at locations 5-8 (after joints/weights).
    let instance_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            5 => Float32x4,
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4,
        ],
    };

    // No backface cull: Quaternius materials are doubleSided.
    let opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("skinned-opaque"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[SkinnedVertex::LAYOUT, instance_layout],
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
            cull_mode: None,
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

    SkinnedPipelines {
        opaque,
        joint_bind_layout,
    }
}

pub struct GpuSkinnedMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
}

impl GpuSkinnedMesh {
    pub fn upload(device: &wgpu::Device, mesh: &SkinMesh) -> Self {
        let mut vertices = Vec::with_capacity(mesh.positions.len());
        for i in 0..mesh.positions.len() {
            vertices.push(SkinnedVertex {
                position: mesh.positions[i].into(),
                normal: mesh.normals[i].into(),
                color: mesh.colors[i].into(),
                joints: mesh.joints[i],
                weights: mesh.weights[i],
            });
        }
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned-vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned-indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
        }
    }
}

pub struct GpuSkinnedEntity {
    pub meshes: Vec<GpuSkinnedMesh>,
    pub instance_buf: wgpu::Buffer,
    pub joint_buf: wgpu::Buffer,
    pub joint_bind: wgpu::BindGroup,
}

impl GpuSkinnedEntity {
    pub fn upload(
        device: &wgpu::Device,
        joint_layout: &wgpu::BindGroupLayout,
        meshes: &[SkinMesh],
        transform: Mat4,
        joints: &[Mat4],
    ) -> Self {
        let gpu_meshes = meshes
            .iter()
            .map(|m| GpuSkinnedMesh::upload(device, m))
            .collect();
        let instance = InstanceRaw::from_matrix(transform);
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned-instance"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let palette = JointPalette::from_matrices(joints);
        let joint_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("joint-palette"),
            contents: bytemuck::bytes_of(&palette),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let joint_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint-bind"),
            layout: joint_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: joint_buf.as_entire_binding(),
            }],
        });
        Self {
            meshes: gpu_meshes,
            instance_buf,
            joint_buf,
            joint_bind,
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, transform: Mat4, joints: &[Mat4]) {
        let instance = InstanceRaw::from_matrix(transform);
        queue.write_buffer(&self.instance_buf, 0, bytemuck::bytes_of(&instance));
        let palette = JointPalette::from_matrices(joints);
        queue.write_buffer(&self.joint_buf, 0, bytemuck::bytes_of(&palette));
    }
}
