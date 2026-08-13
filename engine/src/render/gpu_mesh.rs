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
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let local_bounds = Bounds::around(&mesh.positions);
        Self {
            vertex_buf,
            index_buf,
            instance_buf,
            index_count: mesh.indices.len(),
            opaque_index_count: mesh.opaque_index_count.min(mesh.indices.len()),
            vertex_count: mesh.positions.len(),
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

/// The sphere covering `local` placed at each instance.
fn spread_over(local: Option<Bounds>, instances: &[InstanceRaw]) -> Option<Bounds> {
    let local = local?;
    instances
        .iter()
        .map(|i| local.transformed(Mat4::from_cols_array_2d(&i.model)))
        .reduce(Bounds::union)
}
