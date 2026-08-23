use super::frustum::Bounds;
use super::instance_cull::{DrawIndexedArgs, InstanceCull};
use crate::mesh::{BuiltMesh, InstanceRaw};
use glam::Mat4;
use wgpu::util::DeviceExt;

const INSTANCE_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::COPY_DST);

const INDIRECT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::INDIRECT
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::COPY_DST);

/// Vertex/index buffers and counts one `GpuMesh` is built from.
struct MeshGpuSource {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    opaque_index_count: usize,
    vertex_count: usize,
    local_bounds: Option<Bounds>,
}

pub struct GpuMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub instance_buf: wgpu::Buffer,
    pub compact_buf: wgpu::Buffer,
    pub indirect_buf: wgpu::Buffer,
    pub cull_bind: wgpu::BindGroup,
    pub index_count: usize,
    pub opaque_index_count: usize,
    pub vertex_count: usize,
    pub instance_count: usize,
    pub instance_capacity: usize,
    /// Local sphere around the vertices, before any instance transform.
    local_bounds: Option<Bounds>,
    /// What the draw actually covers: the local sphere at every instance.
    pub bounds: Option<Bounds>,
    /// `Entity::xform_rev` at the last instance upload.
    pub xform_rev: u64,
}

impl GpuMesh {
    pub fn upload(
        device: &wgpu::Device,
        mesh: &BuiltMesh,
        instances: &[InstanceRaw],
        cull: &InstanceCull,
    ) -> Self {
        let vertices = mesh.to_interleaved();
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let local_bounds = Bounds::around(&mesh.positions);
        Self::with_buffers(
            device,
            cull,
            MeshGpuSource {
                vertex_buf,
                index_buf,
                index_count: mesh.indices.len(),
                opaque_index_count: mesh.opaque_index_count.min(mesh.indices.len()),
                vertex_count: mesh.positions.len(),
                local_bounds,
            },
            instances,
        )
    }

    /// Same vertex/index buffers as `src`; a new instance buffer.
    pub fn share_vertices(
        device: &wgpu::Device,
        src: &Self,
        instances: &[InstanceRaw],
        cull: &InstanceCull,
    ) -> Self {
        Self::with_buffers(
            device,
            cull,
            MeshGpuSource {
                vertex_buf: src.vertex_buf.clone(),
                index_buf: src.index_buf.clone(),
                index_count: src.index_count,
                opaque_index_count: src.opaque_index_count,
                vertex_count: src.vertex_count,
                local_bounds: src.local_bounds,
            },
            instances,
        )
    }

    fn with_buffers(
        device: &wgpu::Device,
        cull: &InstanceCull,
        mesh: MeshGpuSource,
        instances: &[InstanceRaw],
    ) -> Self {
        let MeshGpuSource {
            vertex_buf,
            index_buf,
            index_count,
            opaque_index_count,
            vertex_count,
            local_bounds,
        } = mesh;
        let instance_buf = instance_buffer(device, instances, "instances");
        let compact_buf = instance_buffer(device, instances, "instances-compact");
        let indirect_buf = indirect_buffer(device, opaque_index_count, index_count);
        let cull_bind = cull.mesh_bind(device, &instance_buf, &compact_buf, &indirect_buf);
        Self {
            vertex_buf,
            index_buf,
            instance_buf,
            compact_buf,
            indirect_buf,
            cull_bind,
            index_count,
            opaque_index_count,
            vertex_count,
            instance_count: instances.len(),
            instance_capacity: instances.len().max(1),
            local_bounds,
            bounds: spread_over(local_bounds, instances),
            xform_rev: 0,
        }
    }

    pub fn local_sphere(&self) -> Option<Bounds> {
        self.local_bounds
    }

    pub fn bounds_for_instances(&self, instances: &[InstanceRaw]) -> Option<Bounds> {
        spread_over(self.local_bounds, instances)
    }

    pub fn write_instances_at(&self, queue: &wgpu::Queue, start: usize, instances: &[InstanceRaw]) {
        let end = start
            .checked_add(instances.len())
            .expect("partial instance write range overflow");
        if end > self.instance_capacity {
            panic!(
                "partial instance write {start}..{end} exceeds capacity {}",
                self.instance_capacity
            );
        }
        if instances.is_empty() {
            return;
        }
        let byte_offset = start
            .checked_mul(std::mem::size_of::<InstanceRaw>())
            .and_then(|offset| u64::try_from(offset).ok())
            .expect("partial instance write byte offset overflow");
        queue.write_buffer(
            &self.instance_buf,
            byte_offset,
            bytemuck::cast_slice(instances),
        );
    }

    /// Draw nothing without touching the (non-empty) instance buffer.
    pub fn clear_instances(&mut self) {
        self.instance_count = 0;
        self.bounds = None;
    }

    pub fn update_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceRaw],
        cull: &InstanceCull,
    ) -> bool {
        let reallocated = instances.len() > self.instance_capacity;
        if reallocated {
            self.instance_buf = instance_buffer(device, instances, "instances");
            self.compact_buf = instance_buffer(device, instances, "instances-compact");
            self.cull_bind = cull.mesh_bind(
                device,
                &self.instance_buf,
                &self.compact_buf,
                &self.indirect_buf,
            );
            self.instance_capacity = instances.len();
        } else if !instances.is_empty() {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(instances));
        }
        self.instance_count = instances.len();
        self.bounds = spread_over(self.local_bounds, instances);
        reallocated
    }
}

fn instance_buffer(device: &wgpu::Device, instances: &[InstanceRaw], label: &str) -> wgpu::Buffer {
    // wgpu rejects a zero-sized buffer. A prototype with nothing placed still
    // needs its vertex mesh uploaded, so keep a dummy slot and draw zero instances.
    if instances.is_empty() {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&InstanceRaw::from_matrix(Mat4::IDENTITY)),
            usage: INSTANCE_USAGES,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(instances),
            usage: INSTANCE_USAGES,
        })
    }
}

fn indirect_buffer(
    device: &wgpu::Device,
    opaque_index_count: usize,
    index_count: usize,
) -> wgpu::Buffer {
    let opaque =
        u32::try_from(opaque_index_count).expect("opaque index count exceeds GPU u32 range");
    let total = u32::try_from(index_count).expect("index count exceeds GPU u32 range");
    let empty = [
        DrawIndexedArgs::empty_range(opaque, 0),
        DrawIndexedArgs::empty_range(total - opaque, opaque),
    ];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instances-indirect"),
        contents: bytemuck::bytes_of(&empty),
        usage: INDIRECT_USAGES,
    })
}

/// The sphere covering `local` placed at each instance.
fn spread_over(local: Option<Bounds>, instances: &[InstanceRaw]) -> Option<Bounds> {
    let local = local?;
    instances
        .iter()
        .map(|i| local.transformed(Mat4::from_cols_array_2d(&i.model)))
        .reduce(Bounds::union)
}
