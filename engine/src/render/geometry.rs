//! Shared vertex and index buffers for every mesh.
//!
//! Professional engines keep their meshes in a few big buffers instead of
//! two small ones each: a draw then names a range (`first_index`,
//! `base_vertex`) instead of binding its own buffers, so neighbouring draws
//! keep the same bindings and many meshes can go out in one multi-draw.
//!
//! The arena is a list of pages, each a vertex buffer and an index buffer.
//! Pages never move or grow, so a mesh's range stays valid for its whole
//! life and nothing recorded has to be redone when another mesh arrives. A
//! mesh lives in one page; new pages are larger than the last (up to
//! [`MAX_PAGE_VERTICES`]), and a mesh too big for any page gets a page of
//! its own. A mesh's range goes back to its page when the last
//! [`GpuGeometry`] holding it is dropped.

use crate::mesh::{InstanceRaw, Vertex};
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// First page: room for this many vertices (about 4 MB).
const FIRST_PAGE_VERTICES: u64 = 32 * 1024;
/// Pages stop growing here (about 66 MB of vertices).
const MAX_PAGE_VERTICES: u64 = 512 * 1024;
/// Index room per vertex room in a page.
const INDICES_PER_VERTEX: u64 = 4;

const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;
const INDEX_SIZE: u64 = std::mem::size_of::<u32>() as u64;

/// First-fit range allocator with coalescing frees.
#[derive(Debug, Default)]
pub(crate) struct RangeAlloc {
    pub capacity: u64,
    pub free: Vec<Range<u64>>,
}

impl RangeAlloc {
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            free: std::iter::once(0..capacity)
                .filter(|r| !r.is_empty())
                .collect(),
        }
    }

    pub fn alloc(&mut self, len: u64) -> Option<Range<u64>> {
        if len == 0 {
            return Some(0..0);
        }
        let i = self.free.iter().position(|r| r.end - r.start >= len)?;
        let start = self.free[i].start;
        self.free[i].start += len;
        if self.free[i].start == self.free[i].end {
            self.free.remove(i);
        }
        Some(start..start + len)
    }

    pub fn release(&mut self, r: Range<u64>) {
        if r.start == r.end {
            return;
        }
        let i = self.free.partition_point(|f| f.start < r.start);
        self.free.insert(i, r);
        if i + 1 < self.free.len() && self.free[i].end == self.free[i + 1].start {
            self.free[i].end = self.free[i + 1].end;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].end == self.free[i].start {
            self.free[i - 1].end = self.free[i].end;
            self.free.remove(i);
        }
    }

    /// Add room at the end, up to `new_capacity` units.
    pub fn grow(&mut self, new_capacity: u64) {
        let old = self.capacity;
        self.capacity = new_capacity;
        self.release(old..new_capacity);
    }

    /// Nothing allocated.
    pub fn is_empty(&self) -> bool {
        self.capacity == 0 || (self.free.len() == 1 && self.free[0] == (0..self.capacity))
    }
}

struct Page {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    vertices: RangeAlloc,
    indices: RangeAlloc,
}

#[derive(Default)]
struct Pages {
    pages: Vec<Option<Page>>,
    /// Vertex room of the next new page.
    next_vertices: u64,
}

impl Pages {
    /// Give a page that has emptied back to the device, keeping one empty
    /// page so a mesh that comes and goes doesn't make a page each time.
    fn trim(&mut self, emptied: usize) {
        let spare = self
            .pages
            .iter()
            .enumerate()
            .any(|(i, p)| i != emptied && p.as_ref().is_some_and(|p| p.vertices.is_empty()));
        if spare {
            self.pages[emptied] = None;
        }
    }
}

/// The shared buffers meshes are drawn from.
#[derive(Clone, Default)]
pub(crate) struct MeshArenas {
    pub geometry: GeometryArena,
    pub instances: InstanceArena,
}

impl MeshArenas {
    /// Bytes of GPU memory their pages take, used or not.
    pub fn reserved_bytes(&self) -> u64 {
        self.geometry.reserved_bytes() + self.instances.reserved_bytes()
    }
}

/// Every mesh's vertices and indices. Cheap to clone: clones share pages.
#[derive(Clone, Default)]
pub(crate) struct GeometryArena {
    pages: Arc<Mutex<Pages>>,
}

/// One mesh's place in the arena. Its range is freed when the last clone of
/// the `Arc` holding it is dropped.
pub(crate) struct GpuGeometry {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    /// Page the ranges are in: draws in the same page share bindings.
    pub page: u32,
    pub base_vertex: i32,
    pub first_index: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pages: Arc<Mutex<Pages>>,
}

impl Drop for GpuGeometry {
    fn drop(&mut self) {
        let Ok(mut pages) = self.pages.lock() else {
            return;
        };
        let page = self.page as usize;
        let Some(Some(p)) = pages.pages.get_mut(page) else {
            return;
        };
        let v = self.base_vertex as u64;
        let i = u64::from(self.first_index);
        p.vertices.release(v..v + u64::from(self.vertex_count));
        p.indices.release(i..i + u64::from(self.index_count));
        if p.vertices.is_empty() && p.indices.is_empty() {
            pages.trim(page);
        }
    }
}

impl GeometryArena {
    /// Put a mesh in the arena. Its data reaches the GPU with the queue's
    /// next submit, ahead of the commands in it.
    pub fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Arc<GpuGeometry> {
        let vlen = vertices.len() as u64;
        let ilen = indices.len() as u64;
        let mut pages = self.pages.lock().expect("geometry arena lock poisoned");
        let found = pages.pages.iter_mut().enumerate().find_map(|(at, p)| {
            let p = p.as_mut()?;
            let v = p.vertices.alloc(vlen)?;
            match p.indices.alloc(ilen) {
                Some(i) => Some((at, v, i)),
                None => {
                    p.vertices.release(v);
                    None
                }
            }
        });
        let (at, v, i) = match found {
            Some(found) => found,
            None => {
                let room = pages.next_vertices.max(FIRST_PAGE_VERTICES);
                pages.next_vertices = (room * 2).min(MAX_PAGE_VERTICES);
                let room = room.max(vlen).max(ilen.div_ceil(INDICES_PER_VERTEX));
                let page = new_page(device, room, (room * INDICES_PER_VERTEX).max(ilen));
                let at = match pages.pages.iter().position(Option::is_none) {
                    Some(free) => {
                        pages.pages[free] = Some(page);
                        free
                    }
                    None => {
                        pages.pages.push(Some(page));
                        pages.pages.len() - 1
                    }
                };
                let p = pages.pages[at].as_mut().expect("page just made");
                let v = p.vertices.alloc(vlen).expect("new page holds the mesh");
                let i = p.indices.alloc(ilen).expect("new page holds the mesh");
                (at, v, i)
            }
        };
        let p = pages.pages[at].as_ref().expect("allocated page is live");
        if !vertices.is_empty() {
            queue.write_buffer(
                &p.vertex_buf,
                v.start * VERTEX_SIZE,
                bytemuck::cast_slice(vertices),
            );
        }
        if !indices.is_empty() {
            queue.write_buffer(
                &p.index_buf,
                i.start * INDEX_SIZE,
                bytemuck::cast_slice(indices),
            );
        }
        Arc::new(GpuGeometry {
            vertex_buf: p.vertex_buf.clone(),
            index_buf: p.index_buf.clone(),
            page: u32::try_from(at).expect("page count exceeds u32"),
            base_vertex: i32::try_from(v.start).expect("vertex offset exceeds i32"),
            first_index: u32::try_from(i.start).expect("index offset exceeds u32"),
            vertex_count: u32::try_from(vlen).expect("vertex count exceeds u32"),
            index_count: u32::try_from(ilen).expect("index count exceeds u32"),
            pages: self.pages.clone(),
        })
    }

    /// Pages in use.
    pub fn page_count(&self) -> usize {
        let pages = self.pages.lock().expect("geometry arena lock poisoned");
        pages.pages.iter().flatten().count()
    }

    /// Bytes of GPU memory the pages take, used or not.
    pub fn reserved_bytes(&self) -> u64 {
        let pages = self.pages.lock().expect("geometry arena lock poisoned");
        pages
            .pages
            .iter()
            .flatten()
            .map(|p| p.vertex_buf.size() + p.index_buf.size())
            .sum()
    }
}

/// First instance page: room for this many instances (about 1.3 MB).
const FIRST_INSTANCE_PAGE: u64 = 16 * 1024;
/// Instance pages stop growing here (about 21 MB).
const MAX_INSTANCE_PAGE: u64 = 256 * 1024;
const INSTANCE_SIZE: u64 = std::mem::size_of::<InstanceRaw>() as u64;

struct InstancePage {
    buffer: wgpu::Buffer,
    slots: RangeAlloc,
}

#[derive(Default)]
struct InstancePages {
    pages: Vec<Option<InstancePage>>,
    next: u64,
}

/// Every mesh's instance transforms, in shared pages like
/// [`GeometryArena`]: a draw of several meshes reads them all from one
/// buffer through `first_instance`.
#[derive(Clone, Default)]
pub(crate) struct InstanceArena {
    pages: Arc<Mutex<InstancePages>>,
}

/// One mesh's instance slots. Freed on drop.
pub(crate) struct GpuInstances {
    pub buffer: wgpu::Buffer,
    pub page: u32,
    pub first: u32,
    pub capacity: u32,
    pages: Arc<Mutex<InstancePages>>,
}

impl GpuInstances {
    /// Byte offset of slot `slot` in `buffer`.
    pub fn byte_offset(&self, slot: usize) -> u64 {
        (u64::from(self.first) + slot as u64) * INSTANCE_SIZE
    }

    /// The mesh's own slots of the page.
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer
            .slice(self.byte_offset(0)..self.byte_offset(self.capacity as usize))
    }
}

impl Drop for GpuInstances {
    fn drop(&mut self) {
        let Ok(mut pages) = self.pages.lock() else {
            return;
        };
        let at = self.page as usize;
        let Some(Some(p)) = pages.pages.get_mut(at) else {
            return;
        };
        let first = u64::from(self.first);
        p.slots.release(first..first + u64::from(self.capacity));
        if p.slots.is_empty() {
            let spare = pages
                .pages
                .iter()
                .enumerate()
                .any(|(i, q)| i != at && q.as_ref().is_some_and(|q| q.slots.is_empty()));
            if spare {
                pages.pages[at] = None;
            }
        }
    }
}

impl InstanceArena {
    /// Room for `capacity` instances (at least one), holding `instances`
    /// at its start; written with the queue's next submit.
    pub fn alloc(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceRaw],
        capacity: usize,
    ) -> GpuInstances {
        let len = capacity.max(instances.len()).max(1) as u64;
        let mut pages = self.pages.lock().expect("instance arena lock poisoned");
        let found = pages
            .pages
            .iter_mut()
            .enumerate()
            .find_map(|(at, p)| Some((at, p.as_mut()?.slots.alloc(len)?)));
        let (at, r) = match found {
            Some(found) => found,
            None => {
                let room = pages.next.max(FIRST_INSTANCE_PAGE);
                pages.next = (room * 2).min(MAX_INSTANCE_PAGE);
                let room = room.max(len);
                let page = InstancePage {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mesh-instance-page"),
                        size: room * INSTANCE_SIZE,
                        usage: wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    slots: RangeAlloc::new(room),
                };
                let at = match pages.pages.iter().position(Option::is_none) {
                    Some(free) => {
                        pages.pages[free] = Some(page);
                        free
                    }
                    None => {
                        pages.pages.push(Some(page));
                        pages.pages.len() - 1
                    }
                };
                let r = pages.pages[at]
                    .as_mut()
                    .expect("page just made")
                    .slots
                    .alloc(len)
                    .expect("new page holds the instances");
                (at, r)
            }
        };
        let buffer = pages.pages[at]
            .as_ref()
            .expect("allocated page is live")
            .buffer
            .clone();
        // A mesh with nothing placed still owns one slot: keep it harmless.
        let identity = [InstanceRaw::from_matrix(glam::Mat4::IDENTITY)];
        let data: &[InstanceRaw] = if instances.is_empty() {
            &identity
        } else {
            instances
        };
        queue.write_buffer(&buffer, r.start * INSTANCE_SIZE, bytemuck::cast_slice(data));
        GpuInstances {
            buffer,
            page: u32::try_from(at).expect("page count exceeds u32"),
            first: u32::try_from(r.start).expect("instance offset exceeds u32"),
            capacity: u32::try_from(len).expect("instance capacity exceeds u32"),
            pages: self.pages.clone(),
        }
    }

    /// Bytes of GPU memory the pages take, used or not.
    pub fn reserved_bytes(&self) -> u64 {
        let pages = self.pages.lock().expect("instance arena lock poisoned");
        pages.pages.iter().flatten().map(|p| p.buffer.size()).sum()
    }
}

fn new_page(device: &wgpu::Device, vertices: u64, indices: u64) -> Page {
    let copy = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    Page {
        vertex_buf: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-vertex-page"),
            size: vertices.max(1) * VERTEX_SIZE,
            usage: wgpu::BufferUsages::VERTEX | copy,
            mapped_at_creation: false,
        }),
        index_buf: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-index-page"),
            size: indices.max(1) * INDEX_SIZE,
            usage: wgpu::BufferUsages::INDEX | copy,
            mapped_at_creation: false,
        }),
        vertices: RangeAlloc::new(vertices),
        indices: RangeAlloc::new(indices),
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAlloc;

    #[test]
    fn ranges_come_back_and_join() {
        let mut a = RangeAlloc::new(100);
        let x = a.alloc(30).unwrap();
        let y = a.alloc(30).unwrap();
        let z = a.alloc(40).unwrap();
        assert_eq!((x.clone(), y.clone(), z.clone()), (0..30, 30..60, 60..100));
        assert!(a.alloc(1).is_none());
        assert_eq!(a.alloc(0), Some(0..0));
        a.release(y);
        a.release(x);
        assert_eq!(a.alloc(60), Some(0..60));
        a.release(0..60);
        a.release(z);
        assert!(a.is_empty());
        a.grow(150);
        assert_eq!(a.alloc(150), Some(0..150));
    }
}
