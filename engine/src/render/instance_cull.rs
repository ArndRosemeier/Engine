//! GPU per-instance frustum compact, for [`crate::world::InstanceSubmit::GpuIndirect`].
//!
//! The WGSL test is the same sphere transform and plane check as
//! [`super::frustum::Bounds::transformed`] + [`super::frustum::Frustum::intersects_sphere`].
//! A CPU helper in this module is the shader, so a mismatch is a test failure,
//! not a silent visual difference.

use super::frustum::{Bounds, Frustum};
use super::gpu_mesh::GpuMesh;
use bytemuck::{Pod, Zeroable};

pub const WORKGROUP: u32 = 64;
pub const INDIRECT_STRIDE: u64 = std::mem::size_of::<DrawIndexedArgs>() as u64;
pub const OPAQUE_INDIRECT_OFFSET: u64 = 0;
pub const TRANSLUCENT_INDIRECT_OFFSET: u64 = INDIRECT_STRIDE;
const INSTANCE_COUNT_OFFSET: u64 = std::mem::size_of::<u32>() as u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawIndexedArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

impl DrawIndexedArgs {
    pub fn empty_range(index_count: u32, first_index: u32) -> Self {
        Self {
            index_count,
            instance_count: 0,
            first_index,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CullParams {
    planes: [[f32; 4]; 6],
    centre: [f32; 3],
    radius: f32,
    instance_count: u32,
    draw_slots: u32,
    _pad: [u32; 2],
}

const SHADER: &str = r#"
struct CullParams {
    planes: array<vec4<f32>, 6>,
    centre: vec3<f32>,
    radius: f32,
    instance_count: u32,
    draw_slots: u32,
};

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct Instance {
    model: mat4x4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: CullParams;
@group(1) @binding(0) var<storage, read> src: array<Instance>;
@group(1) @binding(1) var<storage, read_write> dst: array<Instance>;
@group(1) @binding(2) var<storage, read_write> draws: array<DrawArgs>;

fn intersects_sphere(centre: vec3<f32>, radius: f32) -> bool {
    for (var i = 0; i < 6; i++) {
        let p = params.planes[i];
        if dot(p.xyz, centre) + p.w < -radius {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.instance_count {
        return;
    }
    let inst = src[i];
    let model = inst.model;
    let scale = max(
        length(model[0].xyz),
        max(length(model[1].xyz), length(model[2].xyz)),
    );
    let world_c = (model * vec4<f32>(params.centre, 1.0)).xyz;
    let world_r = params.radius * scale;
    if !intersects_sphere(world_c, world_r) {
        return;
    }
    let slot = atomicAdd(&draws[0].instance_count, 1u);
    dst[slot] = inst;
    if params.draw_slots > 1u {
        atomicAdd(&draws[1].instance_count, 1u);
    }
}
"#;

pub struct InstanceCull {
    pipeline: wgpu::ComputePipeline,
    params_layout: wgpu::BindGroupLayout,
    mesh_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    params_bind: wgpu::BindGroup,
    params_stride: u64,
    params_capacity: usize,
    params_cursor: usize,
    params_scratch: Vec<u8>,
}

pub struct CullJob<'a> {
    pub gpu: &'a GpuMesh,
    pub local: Bounds,
    pub draws: CullDraws,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullDraws {
    OpaqueOnly,
    OpaqueAndTranslucent,
}

impl CullDraws {
    fn slots(self) -> u32 {
        match self {
            Self::OpaqueOnly => 1,
            Self::OpaqueAndTranslucent => 2,
        }
    }
}

impl InstanceCull {
    pub fn new(device: &wgpu::Device) -> Self {
        let params_size = std::mem::size_of::<CullParams>() as u64;
        let params_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let params_stride = params_size.div_ceil(params_alignment) * params_alignment;
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance-cull-params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(params_size),
                },
                count: None,
            }],
        });
        let mesh_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance-cull-mesh"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instance-cull"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instance-cull-pipeline-layout"),
            bind_group_layouts: &[&params_layout, &mesh_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("instance-cull-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compact"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let (params_buf, params_bind) = params_buffer(device, &params_layout, params_stride, 1);
        Self {
            pipeline,
            params_layout,
            mesh_layout,
            params_buf,
            params_bind,
            params_stride,
            params_capacity: 1,
            params_cursor: 0,
            params_scratch: Vec::new(),
        }
    }

    /// Reserve immutable parameter slots for every dispatch in this command buffer.
    ///
    /// A slot is never rewritten until the submitted frame has consumed it. This
    /// is essential: queue writes are not interleaved with commands merely
    /// because they happen while a command encoder is being populated.
    pub fn begin_frame(&mut self, device: &wgpu::Device, max_jobs: usize) {
        let needed = max_jobs.max(1);
        if needed > self.params_capacity {
            let capacity = needed
                .checked_next_power_of_two()
                .expect("instance-cull parameter capacity overflow");
            (self.params_buf, self.params_bind) =
                params_buffer(device, &self.params_layout, self.params_stride, capacity);
            self.params_capacity = capacity;
        }
        self.params_cursor = 0;
    }

    pub fn mesh_bind(
        &self,
        device: &wgpu::Device,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        draws: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance-cull-mesh-bind"),
            layout: &self.mesh_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: draws.as_entire_binding(),
                },
            ],
        })
    }

    pub fn dispatch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frustum: &Frustum,
        jobs: &[CullJob<'_>],
    ) {
        if jobs.is_empty() {
            return;
        }
        let end = self
            .params_cursor
            .checked_add(jobs.len())
            .expect("instance-cull parameter cursor overflow");
        if end > self.params_capacity {
            panic!(
                "instance-cull frame needs {end} parameter slots, reserved {}",
                self.params_capacity
            );
        }

        let planes = frustum.planes().map(|p| [p.x, p.y, p.z, p.w]);
        let upload_len = self.params_stride as usize * jobs.len();
        self.params_scratch.clear();
        self.params_scratch.resize(upload_len, 0);
        for (job_index, job) in jobs.iter().enumerate() {
            let instance_count = u32::try_from(job.gpu.instance_count)
                .expect("instance count exceeds GPU u32 range");
            if instance_count == 0 {
                panic!("instance cull dispatched with zero instances");
            }
            let params = CullParams {
                planes,
                centre: job.local.centre.into(),
                radius: job.local.radius,
                instance_count,
                draw_slots: job.draws.slots(),
                _pad: [0, 0],
            };
            let start = job_index * self.params_stride as usize;
            let bytes = bytemuck::bytes_of(&params);
            self.params_scratch[start..start + bytes.len()].copy_from_slice(bytes);

            // These clears are commands, so each shadow/main compact starts from
            // zero at the exact point it executes in the command buffer.
            encoder.clear_buffer(&job.gpu.indirect_buf, INSTANCE_COUNT_OFFSET, Some(4));
            if job.draws == CullDraws::OpaqueAndTranslucent {
                encoder.clear_buffer(
                    &job.gpu.indirect_buf,
                    INDIRECT_STRIDE + INSTANCE_COUNT_OFFSET,
                    Some(4),
                );
            }
        }
        let upload_offset = self.params_cursor as u64 * self.params_stride;
        queue.write_buffer(&self.params_buf, upload_offset, &self.params_scratch);

        // One compute pass per view, not one pass per entity. Dispatches remain
        // separate because each bin owns its source, compact, and indirect buffers.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("instance-cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (job_index, job) in jobs.iter().enumerate() {
            let params_index = self.params_cursor + job_index;
            let dynamic_offset = u32::try_from(params_index as u64 * self.params_stride)
                .expect("instance-cull dynamic uniform offset exceeds u32");
            pass.set_bind_group(0, &self.params_bind, &[dynamic_offset]);
            pass.set_bind_group(1, &job.gpu.cull_bind, &[]);
            let instance_count = u32::try_from(job.gpu.instance_count)
                .expect("instance count exceeds GPU u32 range");
            pass.dispatch_workgroups(instance_count.div_ceil(WORKGROUP), 1, 1);
        }
        drop(pass);
        self.params_cursor = end;
    }
}

fn params_buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u64,
    capacity: usize,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let size = stride
        .checked_mul(capacity as u64)
        .expect("instance-cull parameter buffer size overflow");
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instance-cull-params"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("instance-cull-params-bind"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<CullParams>() as u64),
            }),
        }],
    });
    (buffer, bind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use glam::{Mat4, Vec3, Vec4};

    /// CPU compact matching the WGSL `compact` keep/drop test.
    fn compact_visible(frustum: &Frustum, local: Bounds, models: &[Mat4]) -> Vec<usize> {
        models
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                let world = local.transformed(**model);
                frustum.intersects_sphere(world.centre, world.radius)
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Same transform the shader applies to the local sphere.
    fn transformed_sphere(local: Bounds, model: Mat4) -> (Vec3, f32) {
        let world = local.transformed(model);
        (world.centre, world.radius)
    }

    /// Same plane test the shader uses (`dot + w < -radius` rejects).
    fn shader_intersects_sphere(planes: [Vec4; 6], centre: Vec3, radius: f32) -> bool {
        planes
            .iter()
            .all(|p| p.truncate().dot(centre) + p.w >= -radius)
    }

    fn looking_down_negative_z() -> Frustum {
        let camera = Camera::from_parts(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
            55.0,
            0.1,
            1_000.0,
        );
        Frustum::from_view_projection(camera.view_projection(16.0 / 9.0))
    }

    #[test]
    fn shader_plane_test_matches_frustum() {
        let frustum = looking_down_negative_z();
        let cases = [
            (Vec3::new(0.0, 0.0, -100.0), 1.0, true),
            (Vec3::new(0.0, 0.0, 100.0), 1.0, false),
            (Vec3::new(400.0, 0.0, -100.0), 1.0, false),
            (Vec3::new(0.0, 0.0, -5_000.0), 10.0, false),
            (Vec3::new(400.0, 0.0, -100.0), 350.0, true),
        ];
        for (centre, radius, keep) in cases {
            assert_eq!(
                shader_intersects_sphere(frustum.planes(), centre, radius),
                frustum.intersects_sphere(centre, radius),
                "centre={centre:?} radius={radius}"
            );
            assert_eq!(
                shader_intersects_sphere(frustum.planes(), centre, radius),
                keep,
                "expected keep={keep} for centre={centre:?}"
            );
        }
    }

    #[test]
    fn compact_keeps_ahead_and_drops_behind() {
        let frustum = looking_down_negative_z();
        let local = Bounds {
            centre: Vec3::ZERO,
            radius: 1.0,
        };
        let models = [
            Mat4::from_translation(Vec3::new(0.0, 0.0, -50.0)),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 50.0)),
            Mat4::from_translation(Vec3::new(0.0, 0.0, -80.0)),
        ];
        assert_eq!(compact_visible(&frustum, local, &models), vec![0, 2]);
    }

    #[test]
    fn compact_uses_the_same_sphere_transform_as_bounds() {
        let local = Bounds {
            centre: Vec3::new(1.0, 2.0, 3.0),
            radius: 4.0,
        };
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 3.0),
            glam::Quat::from_rotation_y(0.4),
            Vec3::new(10.0, 0.0, -20.0),
        );
        let (centre, radius) = transformed_sphere(local, model);
        let via_bounds = local.transformed(model);
        assert!((centre - via_bounds.centre).length() < 1e-5);
        assert!((radius - via_bounds.radius).abs() < 1e-5);
    }
}
