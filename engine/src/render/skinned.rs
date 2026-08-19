//! GPU skinning pipeline and mesh upload.

use crate::anim::{SkinMesh, MAX_JOINTS};
use crate::mesh::InstanceRaw;
use crate::render::terrain_pipeline::upload_texture;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
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
            3 => Float32x2,
            4 => Uint16x4,
            5 => Float32x4,
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
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
    @location(3) uv: vec2<f32>,
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
    out.uv = v.uv;
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
    let vis = sun_visibility(in.world_p, n, u.eye);
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let base = in.color * texel;
    let lit = base.rgb * (u.ambient + wrap * wrap * u.light_color * vis);
    return vec4<f32>(haze(lit, in.world_p), base.a);
}
"#;

pub struct SkinnedPipelines {
    pub opaque: wgpu::RenderPipeline,
    pub joint_bind_layout: wgpu::BindGroupLayout,
}

pub fn joint_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

pub fn create_skinned_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    scene_bind_layout: &wgpu::BindGroupLayout,
    joint_bind_layout: wgpu::BindGroupLayout,
    albedo_layout: &wgpu::BindGroupLayout,
) -> SkinnedPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinned-shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}{SKINNED_SHADER}", super::pipeline::scene_shader_prefix()).into(),
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("skinned-pipeline-layout"),
        bind_group_layouts: &[scene_bind_layout, &joint_bind_layout, albedo_layout],
        push_constant_ranges: &[],
    });

    // Instance matrix at locations 6-9 (after uv / joints / weights).
    let instance_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4,
            9 => Float32x4,
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

#[derive(Clone)]
pub struct GpuSkinnedMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
    pub albedo_bind: wgpu::BindGroup,
    /// Keeps the uploaded albedo texture alive for `albedo_bind`.
    _albedo_tex: Option<super::terrain_pipeline::GpuTexture>,
}

impl GpuSkinnedMesh {
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        albedo_layout: &wgpu::BindGroupLayout,
        albedo_sampler: &wgpu::Sampler,
        white_albedo: &wgpu::BindGroup,
        mesh: &SkinMesh,
    ) -> Self {
        let mut vertices = Vec::with_capacity(mesh.positions.len());
        for i in 0..mesh.positions.len() {
            vertices.push(SkinnedVertex {
                position: mesh.positions[i].into(),
                normal: mesh.normals[i].into(),
                color: mesh.colors[i].into(),
                uv: mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
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
        let (albedo_bind, _albedo_tex) = if let Some(map) = &mesh.albedo {
            let gpu = upload_texture(device, queue, map.width, map.height, &map.rgba);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skinned-albedo"),
                layout: albedo_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(albedo_sampler),
                    },
                ],
            });
            (bind, Some(gpu))
        } else {
            (white_albedo.clone(), None)
        };
        Self {
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
            albedo_bind,
            _albedo_tex,
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
    /// Same GPU vertices as another animal of this model; only pose buffers are new.
    pub fn from_shared_meshes(
        device: &wgpu::Device,
        joint_layout: &wgpu::BindGroupLayout,
        meshes: Vec<GpuSkinnedMesh>,
        transform: Mat4,
        joints: &[Mat4],
    ) -> Self {
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
            meshes,
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
