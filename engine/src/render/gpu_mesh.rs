use crate::mesh::{BuiltMesh, InstanceRaw};
use wgpu::util::DeviceExt;

pub struct GpuMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub instance_buf: wgpu::Buffer,
    pub index_count: usize,
    pub vertex_count: usize,
    pub instance_count: usize,
    pub instance_capacity: usize,
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
        Self {
            vertex_buf,
            index_buf,
            instance_buf,
            index_count: mesh.indices.len(),
            vertex_count: mesh.positions.len(),
            instance_count: instances.len(),
            instance_capacity: instances.len().max(1),
        }
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
    }
}
