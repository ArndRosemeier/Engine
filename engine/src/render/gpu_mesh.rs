use super::frustum::{Aabb, Bounds};
use super::geometry::{GpuGeometry, GpuInstances, MeshArenas};
use super::instance_cull::{DrawIndexedArgs, InstanceCull};
use crate::mesh::{BuiltMesh, InstanceRaw};
use glam::Mat4;
use std::ops::Range;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const INSTANCE_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::VERTEX
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::COPY_DST);

const INDIRECT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::INDIRECT
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::COPY_DST);

/// Geometry and counts one `GpuMesh` is built from.
struct MeshGpuSource {
    geometry: Arc<GpuGeometry>,
    index_count: usize,
    opaque_index_count: usize,
    vertex_count: usize,
    local_bounds: Option<Bounds>,
    local_box: Option<Aabb>,
    sways: bool,
}

/// Whether a vertex sways in the wind (vegetation profiles 8 to 10, as the
/// scene and shadow vertex shaders test `surface3.w`).
pub(crate) fn vertex_sways(profile: f32) -> bool {
    profile > 7.5 && profile < 10.5
}

pub struct GpuMesh {
    /// The arena page's vertex buffer: draw with [`Self::base_vertex`].
    pub vertex_buf: wgpu::Buffer,
    /// The arena page's index buffer: draw with [`Self::indices`].
    pub index_buf: wgpu::Buffer,
    /// Where this mesh's vertices start in `vertex_buf`.
    pub base_vertex: i32,
    /// Where this mesh's indices start in `index_buf`.
    pub first_index: u32,
    /// Its place in the shared geometry arena (shared by meshes made with
    /// [`Self::share_vertices`]).
    pub geometry: Arc<GpuGeometry>,
    /// The instance arena page holding this mesh's instances, from slot
    /// [`Self::first_instance`]: bind [`Self::instance_slice`] for a draw
    /// of its own, or the whole page with `first_instance`.
    pub instance_buf: wgpu::Buffer,
    pub first_instance: u32,
    instance_slots: GpuInstances,
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
    /// Local box around the vertices, before any instance transform.
    local_box: Option<Aabb>,
    /// The local box at every instance: what occlusion culling tests.
    pub world_box: Option<Aabb>,
    /// `Entity::xform_rev` at the last instance upload.
    pub xform_rev: u64,
    /// Some vertices sway in the wind: its shadow changes every frame.
    pub sways: bool,
}

impl GpuMesh {
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        arenas: &MeshArenas,
        mesh: &BuiltMesh,
        instances: &[InstanceRaw],
        cull: &InstanceCull,
    ) -> Self {
        let vertices = mesh.to_interleaved();
        let geometry = arenas
            .geometry
            .upload(device, queue, &vertices, &mesh.indices);
        let local_bounds = Bounds::around(&mesh.positions);
        let local_box = Aabb::around(&mesh.positions);
        let sways = vertices.iter().any(|v| vertex_sways(v.surface3[3]));
        Self::with_buffers(
            device,
            queue,
            arenas,
            cull,
            MeshGpuSource {
                geometry,
                index_count: mesh.indices.len(),
                opaque_index_count: mesh.opaque_index_count.min(mesh.indices.len()),
                vertex_count: mesh.positions.len(),
                local_bounds,
                local_box,
                sways,
            },
            instances,
        )
    }

    /// Same vertex/index buffers as `src`; a new instance buffer.
    pub fn share_vertices(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        arenas: &MeshArenas,
        src: &Self,
        instances: &[InstanceRaw],
        cull: &InstanceCull,
    ) -> Self {
        Self::with_buffers(
            device,
            queue,
            arenas,
            cull,
            MeshGpuSource {
                geometry: src.geometry.clone(),
                index_count: src.index_count,
                opaque_index_count: src.opaque_index_count,
                vertex_count: src.vertex_count,
                local_bounds: src.local_bounds,
                local_box: src.local_box,
                sways: src.sways,
            },
            instances,
        )
    }

    fn with_buffers(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        arenas: &MeshArenas,
        cull: &InstanceCull,
        mesh: MeshGpuSource,
        instances: &[InstanceRaw],
    ) -> Self {
        let MeshGpuSource {
            geometry,
            index_count,
            opaque_index_count,
            vertex_count,
            local_bounds,
            local_box,
            sways,
        } = mesh;
        let instance_slots = arenas
            .instances
            .alloc(device, queue, instances, instances.len());
        let compact_buf = instance_buffer(device, instances, "instances-compact");
        let indirect_buf = indirect_buffer(device, &geometry, opaque_index_count, index_count);
        let cull_bind = cull.mesh_bind(device, &instance_slots.buffer, &compact_buf, &indirect_buf);
        Self {
            vertex_buf: geometry.vertex_buf.clone(),
            index_buf: geometry.index_buf.clone(),
            base_vertex: geometry.base_vertex,
            first_index: geometry.first_index,
            geometry,
            instance_buf: instance_slots.buffer.clone(),
            first_instance: instance_slots.first,
            instance_slots,
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
            local_box,
            world_box: box_over(local_box, instances),
            xform_rev: 0,
            sways,
        }
    }

    /// Which instance arena page its instances are in.
    pub fn instance_page(&self) -> u32 {
        self.instance_slots.page
    }

    /// This mesh's own instance slots, to bind for a draw of its own.
    pub fn instance_slice(&self) -> wgpu::BufferSlice<'_> {
        self.instance_slots.slice()
    }

    /// Its instance slots from `slot` on.
    pub fn instance_slice_from(&self, slot: usize) -> wgpu::BufferSlice<'_> {
        self.instance_buf.slice(
            self.instance_slots.byte_offset(slot)
                ..self
                    .instance_slots
                    .byte_offset(self.instance_slots.capacity as usize),
        )
    }

    /// `range` of this mesh's own indices, as a range of `index_buf`.
    pub fn indices(&self, range: Range<u32>) -> Range<u32> {
        range.start + self.first_index..range.end + self.first_index
    }

    /// Its opaque indices in `index_buf`.
    pub fn opaque_indices(&self) -> Range<u32> {
        self.indices(0..self.opaque_index_count as u32)
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
        queue.write_buffer(
            &self.instance_buf,
            self.instance_slots.byte_offset(start),
            bytemuck::cast_slice(instances),
        );
    }

    /// Draw nothing without touching the (non-empty) instance buffer.
    pub fn clear_instances(&mut self) {
        self.instance_count = 0;
        self.bounds = None;
        self.world_box = None;
    }

    pub fn update_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        arenas: &MeshArenas,
        instances: &[InstanceRaw],
        cull: &InstanceCull,
    ) -> bool {
        let reallocated = instances.len() > self.instance_capacity;
        if reallocated {
            self.instance_slots = arenas
                .instances
                .alloc(device, queue, instances, instances.len());
            self.instance_buf = self.instance_slots.buffer.clone();
            self.first_instance = self.instance_slots.first;
            self.compact_buf = instance_buffer(device, instances, "instances-compact");
            self.cull_bind = cull.mesh_bind(
                device,
                &self.instance_buf,
                &self.compact_buf,
                &self.indirect_buf,
            );
            self.instance_capacity = instances.len();
        } else if !instances.is_empty() {
            queue.write_buffer(
                &self.instance_buf,
                self.instance_slots.byte_offset(0),
                bytemuck::cast_slice(instances),
            );
        }
        self.instance_count = instances.len();
        self.bounds = spread_over(self.local_bounds, instances);
        self.world_box = box_over(self.local_box, instances);
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
    geometry: &GpuGeometry,
    opaque_index_count: usize,
    index_count: usize,
) -> wgpu::Buffer {
    let opaque =
        u32::try_from(opaque_index_count).expect("opaque index count exceeds GPU u32 range");
    let total = u32::try_from(index_count).expect("index count exceeds GPU u32 range");
    let (first, base) = (geometry.first_index, geometry.base_vertex);
    let empty = [
        DrawIndexedArgs::empty_range(opaque, first, base),
        DrawIndexedArgs::empty_range(total - opaque, first + opaque, base),
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

/// The box covering `local` placed at each instance.
fn box_over(local: Option<Aabb>, instances: &[InstanceRaw]) -> Option<Aabb> {
    let local = local?;
    instances
        .iter()
        .map(|i| local.transformed(Mat4::from_cols_array_2d(&i.model)))
        .reduce(Aabb::union)
}
