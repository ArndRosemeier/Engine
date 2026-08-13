use super::frustum::Bounds;
use crate::mesh::{BuiltMesh, InstanceRaw};
use glam::Mat4;
use wgpu::util::DeviceExt;

pub struct GpuMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub instance_buf: wgpu::Buffer,
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
    pub fn upload(device: &wgpu::Device, mesh: &BuiltMesh, instances: &[InstanceRaw]) -> Self {
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
            vertex_buf,
            index_buf,
            mesh.indices.len(),
            mesh.opaque_index_count.min(mesh.indices.len()),
            mesh.positions.len(),
            local_bounds,
            instances,
        )
    }

    /// Same vertex/index buffers as `src`; a new instance buffer.
    pub fn share_vertices(device: &wgpu::Device, src: &Self, instances: &[InstanceRaw]) -> Self {
        Self::with_buffers(
            device,
            src.vertex_buf.clone(),
            src.index_buf.clone(),
            src.index_count,
            src.opaque_index_count,
            src.vertex_count,
            src.local_bounds,
            instances,
        )
    }

    fn with_buffers(
        device: &wgpu::Device,
        vertex_buf: wgpu::Buffer,
        index_buf: wgpu::Buffer,
        index_count: usize,
        opaque_index_count: usize,
        vertex_count: usize,
        local_bounds: Option<Bounds>,
        instances: &[InstanceRaw],
    ) -> Self {
        let instance_buf = instance_buffer(device, instances);
        Self {
            vertex_buf,
            index_buf,
            instance_buf,
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
    ) {
        if instances.len() > self.instance_capacity {
            self.instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("instances"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.instance_capacity = instances.len();
        } else {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(instances));
        }
        self.instance_count = instances.len();
        self.bounds = spread_over(self.local_bounds, instances);
    }
}

fn instance_buffer(device: &wgpu::Device, instances: &[InstanceRaw]) -> wgpu::Buffer {
    // wgpu rejects a zero-sized buffer. A prototype with nothing placed still
    // needs its vertex mesh uploaded, so keep a dummy slot and draw zero instances.
    if instances.is_empty() {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::bytes_of(&InstanceRaw {
                model: Mat4::IDENTITY.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        })
    }
}

/// The sphere covering `local` placed at each instance.
fn spread_over(local: Option<Bounds>, instances: &[InstanceRaw]) -> Option<Bounds> {
    let local = local?;
    instances
        .iter()
        .map(|i| local.transformed(Mat4::from_cols_array_2d(&i.model)))
        .reduce(Bounds::union)
}
