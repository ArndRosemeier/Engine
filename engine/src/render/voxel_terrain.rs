//! GPU surface nets for [`crate::voxel_terrain::VoxelTerrain`].
//!
//! The CPU copies the density around each mesh region into an upload buffer
//! and counts what the region will produce (exact vertex and index counts, so
//! arena space is never guessed). A compute pass then places one vertex per
//! surface cell (crossing average, gradient normal) and one quad per crossing
//! edge, straight into shared vertex and index arenas that stay on the GPU.
//!
//! Regions form an octree: near the camera one region is one chunk at full
//! resolution; further away one region covers 2^L chunks per axis, sampled
//! every 2^L voxels. Where two levels meet, both sides extend one cell into
//! the other so the seam is covered. A new region set is only shown once all
//! its regions are meshed, so the view never has holes while moving.

use super::frustum::{Bounds, Frustum};
use crate::voxel_terrain::VoxelTerrain;
use glam::{IVec3, Mat4, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

const CHUNK: i32 = crate::volume::CHUNK_SIZE;
/// Samples per axis that one region uploads: corners -3..=34 (in region steps).
const GRID: i32 = CHUNK + 6;
/// Cells per axis of the sample grid.
const CELLS: i32 = GRID - 1;
/// Region corner 0 sits at grid index PAD.
const PAD: i32 = 3;
/// Regions meshed per frame at most (edits first, then new regions).
const JOBS_PER_FRAME: usize = 32;
/// u32 words per vertex: position xyz (f32) + normal (snorm16 x4).
const VERTEX_WORDS: u64 = 5;
const VERTEX_BYTES: u64 = VERTEX_WORDS * 4;

/// Per-vertex buffer: position and packed normal.
pub const VOXEL_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: VERTEX_BYTES,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        },
        wgpu::VertexAttribute {
            offset: 12,
            shader_location: 1,
            format: wgpu::VertexFormat::Snorm16x4,
        },
        // uv.y = how much of the surface an edit exposed (0 natural, 1 bare).
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 3,
            format: wgpu::VertexFormat::Snorm16x2,
        },
    ],
};

/// Per-region instance: transform plus everything the lit shader reads per
/// vertex for a regular mesh (colour, uv, surface material), constant here.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelInstance {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
    color: [f32; 4],
    uv: [f32; 2],
    _pad: [f32; 2],
    surface: [[f32; 4]; 5],
}

const fn attr(offset: u64, location: u32, format: wgpu::VertexFormat) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        offset,
        shader_location: location,
        format,
    }
}

pub const VOXEL_INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<VoxelInstance>() as u64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[
        attr(0, 6, wgpu::VertexFormat::Float32x4),
        attr(16, 7, wgpu::VertexFormat::Float32x4),
        attr(32, 8, wgpu::VertexFormat::Float32x4),
        attr(48, 9, wgpu::VertexFormat::Float32x4),
        attr(64, 10, wgpu::VertexFormat::Float32x4),
        attr(80, 2, wgpu::VertexFormat::Float32x4),
        attr(112, 4, wgpu::VertexFormat::Float32x4),
        attr(128, 5, wgpu::VertexFormat::Float32x4),
        attr(144, 11, wgpu::VertexFormat::Float32x4),
        attr(160, 12, wgpu::VertexFormat::Float32x4),
        attr(176, 13, wgpu::VertexFormat::Float32x4),
    ],
};

/// Both voxel buffers, for pipelines that draw voxel terrain.
pub const VOXEL_BUFFERS: [wgpu::VertexBufferLayout<'static>; 2] =
    [VOXEL_VERTEX_LAYOUT, VOXEL_INSTANCE_LAYOUT];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct JobRaw {
    cell: f32,
    vertex_base: u32,
    index_base: u32,
    faces: u32,
    vertex_count: u32,
    index_count: u32,
    _pad: [u32; 2],
}

const MESH_WGSL: &str = r#"
const GRID: i32 = 38;
const CELLS: i32 = 37;
const PAD: i32 = 3;
const CHUNK: i32 = 32;

struct Job {
    cell: f32,
    vertex_base: u32,
    index_base: u32,
    faces: u32,
    vertex_count: u32,
    index_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> density: array<f32>;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read_write> cell_vertex: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> vertices: array<u32>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;

fn d(job: u32, g: vec3<i32>) -> f32 {
    let c = clamp(g, vec3<i32>(0), vec3<i32>(GRID - 1));
    return density[job * u32(GRID * GRID * GRID) + u32(c.x + c.y * GRID + c.z * GRID * GRID)];
}

// Region-local range of edge corners this job owns, per axis.
fn owned_lo(faces: u32) -> vec3<i32> {
    return vec3<i32>(
        select(0, -1, (faces & 1u) != 0u),
        select(0, -1, (faces & 4u) != 0u),
        select(0, -1, (faces & 16u) != 0u));
}

fn owned_hi(faces: u32) -> vec3<i32> {
    return vec3<i32>(
        select(CHUNK, CHUNK + 1, (faces & 2u) != 0u),
        select(CHUNK, CHUNK + 1, (faces & 8u) != 0u),
        select(CHUNK, CHUNK + 1, (faces & 32u) != 0u));
}

fn grid_of(id: u32) -> vec3<i32> {
    let i = i32(id);
    return vec3<i32>(i % CELLS, (i / CELLS) % CELLS, i / (CELLS * CELLS));
}

fn gradient(job: u32, g: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(
        d(job, g + vec3<i32>(1, 0, 0)) - d(job, g - vec3<i32>(1, 0, 0)),
        d(job, g + vec3<i32>(0, 1, 0)) - d(job, g - vec3<i32>(0, 1, 0)),
        d(job, g + vec3<i32>(0, 0, 1)) - d(job, g - vec3<i32>(0, 0, 1)));
}

fn corner_offset(i: u32) -> vec3<i32> {
    return vec3<i32>(i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
}

@compute @workgroup_size(64)
fn cs_vertices(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(workgroup_id) wg: vec3<u32>) {
    let job_id = wg.y;
    if gid.x >= u32(CELLS * CELLS * CELLS) {
        return;
    }
    let job = jobs[job_id];
    let g = grid_of(gid.x);
    let local = g - vec3<i32>(PAD);
    let lo = owned_lo(job.faces) - vec3<i32>(1);
    let hi = owned_hi(job.faces);
    if any(local < lo) || any(local >= hi) {
        return;
    }
    var v: array<f32, 8>;
    var mask = 0u;
    var exposed = 0.0;
    for (var i = 0u; i < 8u; i++) {
        v[i] = d(job_id, g + corner_offset(i));
        exposed += select(0.0, 1.0, (bitcast<u32>(v[i]) & 3u) == 1u);
        if v[i] > 0.0 {
            mask |= 1u << i;
        }
    }
    if mask == 0u || mask == 255u {
        return;
    }
    // Average of the edge crossings inside the cell.
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var i = 0u; i < 8u; i++) {
        for (var axis = 0u; axis < 3u; axis++) {
            let bit = 1u << axis;
            if (i & bit) != 0u {
                continue;
            }
            let j = i | bit;
            let si = (mask >> i) & 1u;
            let sj = (mask >> j) & 1u;
            if si == sj {
                continue;
            }
            let t = clamp(v[i] / (v[i] - v[j]), 0.0, 1.0);
            sum += mix(vec3<f32>(corner_offset(i)), vec3<f32>(corner_offset(j)), t);
            count += 1.0;
        }
    }
    let f = sum / count;
    // Trilinear blend of the corner gradients at the vertex.
    var grad = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let o = vec3<f32>(corner_offset(i));
        let w = mix(1.0 - f.x, f.x, o.x) * mix(1.0 - f.y, f.y, o.y) * mix(1.0 - f.z, f.z, o.z);
        grad += gradient(job_id, g + corner_offset(i)) * w;
    }
    var n = -grad;
    if dot(n, n) < 1e-12 {
        n = vec3<f32>(0.0, 1.0, 0.0);
    }
    n = normalize(n);

    let slot = atomicAdd(&counters[job_id * 2u], 1u);
    if slot >= job.vertex_count {
        return;
    }
    let p = (vec3<f32>(local) + f) * job.cell;
    let base = (job.vertex_base + slot) * 5u;
    vertices[base] = bitcast<u32>(p.x);
    vertices[base + 1u] = bitcast<u32>(p.y);
    vertices[base + 2u] = bitcast<u32>(p.z);
    vertices[base + 3u] = pack2x16snorm(n.xy);
    vertices[base + 4u] = pack2x16snorm(vec2<f32>(n.z, min(exposed * 0.25, 1.0)));
    cell_vertex[job_id * u32(CELLS * CELLS * CELLS) + gid.x] = slot;
}

fn cell_index(job: u32, g: vec3<i32>) -> u32 {
    return cell_vertex[job * u32(CELLS * CELLS * CELLS) + u32(g.x + g.y * CELLS + g.z * CELLS * CELLS)];
}

@compute @workgroup_size(64)
fn cs_indices(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(workgroup_id) wg: vec3<u32>) {
    let job_id = wg.y;
    if gid.x >= u32(CELLS * CELLS * CELLS) {
        return;
    }
    let job = jobs[job_id];
    let g = grid_of(gid.x);
    let local = g - vec3<i32>(PAD);
    let lo = owned_lo(job.faces);
    let hi = owned_hi(job.faces);
    if any(local < lo) || any(local >= hi) {
        return;
    }
    let d0 = d(job_id, g);
    for (var axis = 0; axis < 3; axis++) {
        var ea = vec3<i32>(0);
        ea[axis] = 1;
        let d1 = d(job_id, g + ea);
        if (d0 > 0.0) == (d1 > 0.0) {
            continue;
        }
        var eu = vec3<i32>(0);
        eu[(axis + 1) % 3] = 1;
        var ev = vec3<i32>(0);
        ev[(axis + 2) % 3] = 1;
        let c00 = cell_index(job_id, g - eu - ev);
        let c10 = cell_index(job_id, g - ev);
        let c11 = cell_index(job_id, g);
        let c01 = cell_index(job_id, g - eu);
        let at = atomicAdd(&counters[job_id * 2u + 1u], 6u);
        if at + 6u > job.index_count {
            continue;
        }
        let o = job.index_base + at;
        if d0 > 0.0 {
            indices[o] = c00; indices[o + 1u] = c10; indices[o + 2u] = c11;
            indices[o + 3u] = c00; indices[o + 4u] = c11; indices[o + 5u] = c01;
        } else {
            indices[o] = c00; indices[o + 1u] = c11; indices[o + 2u] = c10;
            indices[o + 3u] = c00; indices[o + 4u] = c01; indices[o + 5u] = c11;
        }
    }
}
"#;

/// First-fit range allocator with coalescing frees.
#[derive(Debug, Default)]
struct RangeAlloc {
    capacity: u64,
    free: Vec<Range<u64>>,
}

impl RangeAlloc {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            free: std::iter::once(0..capacity).collect(),
        }
    }

    fn alloc(&mut self, len: u64) -> Option<Range<u64>> {
        let i = self.free.iter().position(|r| r.end - r.start >= len)?;
        let start = self.free[i].start;
        self.free[i].start += len;
        if self.free[i].start == self.free[i].end {
            self.free.remove(i);
        }
        Some(start..start + len)
    }

    fn release(&mut self, r: Range<u64>) {
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

    /// Add `extra` units at the end.
    fn grow(&mut self, new_capacity: u64) {
        let old = self.capacity;
        self.capacity = new_capacity;
        self.release(old..new_capacity);
    }
}

/// A growable GPU buffer managed in element units.
struct Arena {
    buffer: wgpu::Buffer,
    alloc: RangeAlloc,
    unit: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl Arena {
    fn new(
        device: &wgpu::Device,
        label: &'static str,
        units: u64,
        unit: u64,
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: units * unit,
                usage,
                mapped_at_creation: false,
            }),
            alloc: RangeAlloc::new(units),
            unit,
            usage,
            label,
        }
    }

    /// Allocate, growing the buffer (and copying it) when full. Returns
    /// `None` only when the device cannot hold a buffer that large.
    fn alloc(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        len: u64,
        max_bytes: u64,
    ) -> Option<(Range<u64>, bool)> {
        if let Some(r) = self.alloc.alloc(len) {
            return Some((r, false));
        }
        let mut units = self.alloc.capacity.max(1);
        while units < self.alloc.capacity + len {
            units *= 2;
        }
        units = units.max(self.alloc.capacity * 2);
        if units * self.unit > max_bytes {
            units = max_bytes / self.unit;
            if units < self.alloc.capacity + len {
                return None;
            }
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: units * self.unit,
            usage: self.usage,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, self.buffer.size());
        self.buffer = buffer;
        self.alloc.grow(units);
        self.alloc.alloc(len).map(|r| (r, true))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    level: u32,
    /// Chunk coordinate of the region's minimum corner (aligned to 2^level).
    key: IVec3,
}

impl NodeKey {
    fn stride(self) -> i32 {
        1 << self.level
    }

    fn chunks(self) -> i32 {
        1 << self.level
    }

    fn aabb(self, chunk_world: f32) -> (Vec3, Vec3) {
        let min = self.key.as_vec3() * chunk_world;
        (min, min + Vec3::splat(chunk_world * self.chunks() as f32))
    }

    /// Chunk keys whose samples this region reads (its own plus padding).
    fn reads_chunk(self, chunk: IVec3) -> bool {
        let s = self.stride();
        let lo = (self.key * CHUNK - IVec3::splat(PAD * s)).div_euclid(IVec3::splat(CHUNK));
        let hi = (self.key * CHUNK + IVec3::splat((CHUNK + PAD - 1) * s))
            .div_euclid(IVec3::splat(CHUNK));
        chunk.cmpge(lo).all() && chunk.cmple(hi).all()
    }
}

struct NodeMesh {
    vertices: Range<u64>,
    indices: Range<u64>,
    slot: u32,
    faces: u32,
    bounds: Option<Bounds>,
}

/// CPU side of one meshing job: the samples plus exact output counts.
struct Prepared {
    node: NodeKey,
    faces: u32,
    samples: Vec<f32>,
    vertex_count: u32,
    index_count: u32,
    /// Region-local cell range holding vertices, for culling bounds.
    cells: Option<(IVec3, IVec3)>,
}

fn owned_range(faces: u32) -> (IVec3, IVec3) {
    let bit = |b: u32| faces & (1 << b) != 0;
    let lo = IVec3::new(
        if bit(0) { -1 } else { 0 },
        if bit(2) { -1 } else { 0 },
        if bit(4) { -1 } else { 0 },
    );
    let hi = IVec3::new(
        if bit(1) { CHUNK + 1 } else { CHUNK },
        if bit(3) { CHUNK + 1 } else { CHUNK },
        if bit(5) { CHUNK + 1 } else { CHUNK },
    );
    (lo, hi)
}

/// Count exactly what the compute pass will emit (same tests, same order of
/// float comparisons), so arena ranges are sized right.
fn count_surface(samples: &[f32], faces: u32) -> (u32, u32, Option<(IVec3, IVec3)>) {
    let solid: Vec<bool> = samples.iter().map(|&d| d > 0.0).collect();
    let at = |g: IVec3| solid[(g.x + g.y * GRID + g.z * GRID * GRID) as usize];
    let (lo, hi) = owned_range(faces);
    let mut verts = 0u32;
    let mut quads = 0u32;
    let (mut cmin, mut cmax) = (IVec3::splat(i32::MAX), IVec3::splat(i32::MIN));
    for z in lo.z - 1..hi.z {
        for y in lo.y - 1..hi.y {
            for x in lo.x - 1..hi.x {
                let local = IVec3::new(x, y, z);
                let g = local + IVec3::splat(PAD);
                let first = at(g);
                let mut mixed = false;
                for i in 1..8 {
                    let o = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
                    if at(g + o) != first {
                        mixed = true;
                        break;
                    }
                }
                if mixed {
                    verts += 1;
                    cmin = cmin.min(local);
                    cmax = cmax.max(local);
                }
                if local.cmpge(lo).all() {
                    for axis in [IVec3::X, IVec3::Y, IVec3::Z] {
                        if at(g) != at(g + axis) {
                            quads += 1;
                        }
                    }
                }
            }
        }
    }
    let cells = (verts > 0).then_some((cmin, cmax));
    (verts, quads * 6, cells)
}

pub struct VoxelTerrainGpu {
    vertices_pipeline: wgpu::ComputePipeline,
    indices_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    upload: wgpu::Buffer,
    jobs: wgpu::Buffer,
    cell_vertex: wgpu::Buffer,
    counters: wgpu::Buffer,
    vertex_arena: Arena,
    index_arena: Arena,
    instances: wgpu::Buffer,
    instance_capacity: u32,
    free_slots: Vec<u32>,
    next_slot: u32,
    bind: Option<wgpu::BindGroup>,
    nodes: HashMap<NodeKey, NodeMesh>,
    /// Regions drawn this frame.
    drawn: Vec<NodeKey>,
    /// Volume revision already folded into node dirtiness.
    seen_rev: u64,
    occupancy: Vec<HashSet<IVec3>>,
    dirty: HashSet<NodeKey>,
    instance_key: Option<[u32; 8]>,
    voxel_size: f32,
    /// Bumped whenever what is drawn changes (for the GI rebuild).
    epoch: u64,
    present: bool,
}

impl VoxelTerrainGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-surface-nets"),
            source: wgpu::ShaderSource::Wgsl(MESH_WGSL.into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-mesh-layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-mesh-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let jobs = JOBS_PER_FRAME as u64;
        let grid3 = (GRID * GRID * GRID) as u64;
        let cells3 = (CELLS * CELLS * CELLS) as u64;
        let arena_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let instance_capacity = 64;
        Self {
            vertices_pipeline: compute("cs_vertices"),
            indices_pipeline: compute("cs_indices"),
            layout,
            upload: buffer("voxel-density-upload", jobs * grid3 * 4, storage_dst),
            jobs: buffer(
                "voxel-mesh-jobs",
                jobs * std::mem::size_of::<JobRaw>() as u64,
                storage_dst,
            ),
            cell_vertex: buffer(
                "voxel-cell-vertex",
                jobs * cells3 * 4,
                wgpu::BufferUsages::STORAGE,
            ),
            counters: buffer("voxel-mesh-counters", jobs * 8, storage_dst),
            vertex_arena: Arena::new(
                device,
                "voxel-vertex-arena",
                1 << 18,
                VERTEX_BYTES,
                arena_usage | wgpu::BufferUsages::VERTEX,
            ),
            index_arena: Arena::new(
                device,
                "voxel-index-arena",
                3 << 19,
                4,
                arena_usage | wgpu::BufferUsages::INDEX,
            ),
            instances: buffer(
                "voxel-instances",
                instance_capacity as u64 * std::mem::size_of::<VoxelInstance>() as u64,
                wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            instance_capacity,
            free_slots: Vec::new(),
            next_slot: 0,
            bind: None,
            nodes: HashMap::new(),
            drawn: Vec::new(),
            seen_rev: 0,
            occupancy: Vec::new(),
            dirty: HashSet::new(),
            instance_key: None,
            voxel_size: 0.0,
            epoch: 0,
            present: false,
        }
    }

    /// Changes whenever the drawn voxel surface changes.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn is_drawing(&self) -> bool {
        self.present && !self.drawn.is_empty()
    }

    fn clear(&mut self) {
        let keys: Vec<NodeKey> = self.nodes.keys().copied().collect();
        for key in keys {
            self.free_node(key);
        }
        self.drawn.clear();
        self.dirty.clear();
        self.occupancy.clear();
        self.seen_rev = 0;
        self.instance_key = None;
        if self.present {
            self.epoch += 1;
        }
        self.present = false;
    }

    fn free_node(&mut self, key: NodeKey) {
        if let Some(node) = self.nodes.remove(&key) {
            self.vertex_arena.alloc.release(node.vertices);
            self.index_arena.alloc.release(node.indices);
            self.free_slots.push(node.slot);
        }
    }

    fn instance_for(terrain: &VoxelTerrain, origin: Vec3) -> VoxelInstance {
        let m = terrain.material();
        VoxelInstance {
            model: Mat4::from_translation(origin).to_cols_array_2d(),
            tint: [1.0; 4],
            color: terrain.color().to_vec4().into(),
            uv: [0.0, 0.0],
            _pad: [0.0; 2],
            surface: [
                [m.roughness, m.metallic, m.emission, m.detail_scale],
                [m.strata_strength, m.seed, m.warp_strength, m.noise_gain],
                [
                    m.orientation[0],
                    m.orientation[1],
                    m.orientation[2],
                    m.profile,
                ],
                [
                    m.coverage_direction[0],
                    m.coverage_direction[1],
                    m.coverage_direction[2],
                    m.coverage_level,
                ],
                [
                    m.coverage_softness,
                    m.coverage_strength,
                    m.overlay_roughness,
                    m.overlay_metallic,
                ],
            ],
        }
    }

    fn take_slot(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> u32 {
        if let Some(slot) = self.free_slots.pop() {
            return slot;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if slot >= self.instance_capacity {
            let capacity = self.instance_capacity * 2;
            let stride = std::mem::size_of::<VoxelInstance>() as u64;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel-instances"),
                size: capacity as u64 * stride,
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&self.instances, 0, &buffer, 0, self.instances.size());
            self.instances = buffer;
            self.instance_capacity = capacity;
        }
        slot
    }

    fn rebuild_occupancy(&mut self, terrain: &VoxelTerrain) {
        let levels = terrain.max_lod() as usize + 1;
        self.occupancy = vec![HashSet::new(); levels];
        for key in terrain.volume().chunk_keys() {
            for (level, set) in self.occupancy.iter_mut().enumerate() {
                let size = IVec3::splat(1 << level);
                set.insert(key.div_euclid(size) * size);
            }
        }
    }

    /// The octree cut for this camera position.
    fn select(&self, terrain: &VoxelTerrain, eye: Vec3) -> Vec<NodeKey> {
        let chunk_world = terrain.volume().chunk_world_size();
        let detail = terrain.detail_distance();
        let Some(top) = self.occupancy.last() else {
            return Vec::new();
        };
        let top_level = self.occupancy.len() as u32 - 1;
        let mut out = Vec::new();
        let mut stack: Vec<NodeKey> = top
            .iter()
            .map(|&key| NodeKey {
                level: top_level,
                key,
            })
            .collect();
        while let Some(node) = stack.pop() {
            let (min, max) = node.aabb(chunk_world);
            let dist = (eye.clamp(min, max) - eye).length();
            let size = chunk_world * node.chunks() as f32;
            if node.level == 0 || dist > detail * size {
                out.push(node);
                continue;
            }
            let child_level = node.level - 1;
            let half = 1 << child_level;
            for i in 0..8 {
                let key = node.key + IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1) * half;
                if self.occupancy[child_level as usize].contains(&key) {
                    stack.push(NodeKey {
                        level: child_level,
                        key,
                    });
                }
            }
        }
        out.sort_unstable_by_key(|n| (n.level, n.key.x, n.key.y, n.key.z));
        out
    }

    /// Faces of `node` whose neighbour is drawn at another level.
    fn seam_faces(node: NodeKey, selected: &HashSet<NodeKey>, max_level: u32) -> u32 {
        let mut faces = 0;
        let n = node.chunks();
        for axis in 0..3 {
            for side in 0..2 {
                let mut probe = node.key + IVec3::splat(n / 2);
                probe[axis] = if side == 0 {
                    node.key[axis] - 1
                } else {
                    node.key[axis] + n
                };
                for level in 0..=max_level {
                    let size = IVec3::splat(1 << level);
                    let key = probe.div_euclid(size) * size;
                    if selected.contains(&NodeKey { level, key }) {
                        if level != node.level {
                            faces |= 1 << (axis * 2 + side);
                        }
                        break;
                    }
                }
            }
        }
        faces
    }

    /// Mesh what changed or newly came into view, then pick what to draw.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        terrain: Option<&VoxelTerrain>,
        eye: Vec3,
    ) {
        let Some(terrain) = terrain else {
            if self.present || !self.nodes.is_empty() {
                self.clear();
            }
            return;
        };
        let volume = terrain.volume();
        if volume.voxel_size() != self.voxel_size
            || self.occupancy.len() != terrain.max_lod() as usize + 1
        {
            self.clear();
            self.voxel_size = volume.voxel_size();
        }
        self.present = true;
        if volume.rev() != self.seen_rev {
            let (changed, removed) = volume.changed_since(self.seen_rev);
            let new_chunk = changed.iter().any(|k| {
                self.occupancy
                    .first()
                    .is_none_or(|level0| !level0.contains(k))
            });
            if removed || new_chunk || self.occupancy.is_empty() {
                self.rebuild_occupancy(terrain);
            }
            if removed {
                self.dirty.extend(self.nodes.keys().copied());
            } else {
                for &key in self.nodes.keys() {
                    if changed.iter().any(|&c| key.reads_chunk(c)) {
                        self.dirty.insert(key);
                    }
                }
            }
            self.seen_rev = volume.rev();
        }

        let instance_key = {
            let m = terrain.material();
            let c = terrain.color().to_vec4();
            [
                m.roughness.to_bits(),
                m.profile.to_bits(),
                m.seed.to_bits(),
                m.metallic.to_bits(),
                c.x.to_bits(),
                c.y.to_bits(),
                c.z.to_bits(),
                m.detail_scale.to_bits() ^ m.coverage_level.to_bits(),
            ]
        };
        let rewrite_instances = self.instance_key != Some(instance_key);
        self.instance_key = Some(instance_key);

        let desired = self.select(terrain, eye);
        let desired_set: HashSet<NodeKey> = desired.iter().copied().collect();
        let max_level = terrain.max_lod();

        // Edited regions on screen first, nearest first; then regions the new
        // cut needs; then seam changes of regions already meshed.
        let chunk_world = volume.chunk_world_size();
        let distance = |n: &NodeKey| {
            let (min, max) = n.aabb(chunk_world);
            (eye.clamp(min, max) - eye).length_squared()
        };
        let mut work: Vec<(u8, f32, NodeKey, u32)> = Vec::new();
        for &node in &desired {
            let faces = Self::seam_faces(node, &desired_set, max_level);
            let priority = match self.nodes.get(&node) {
                None => 1,
                Some(_) if self.dirty.contains(&node) => 0,
                Some(mesh) if mesh.faces != faces => 2,
                Some(_) => continue,
            };
            work.push((priority, distance(&node), node, faces));
        }
        for &node in &self.drawn {
            if !desired_set.contains(&node) && self.dirty.contains(&node) {
                let faces = self.nodes.get(&node).map_or(0, |m| m.faces);
                work.push((0, distance(&node), node, faces));
            }
        }
        work.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        work.truncate(JOBS_PER_FRAME);

        let prepared: Vec<Prepared> = work
            .par_iter()
            .map(|&(_, _, node, faces)| {
                let s = node.stride();
                let origin = node.key * CHUNK - IVec3::splat(PAD * s);
                let mut samples = Vec::new();
                volume.gather(origin, s, GRID, &mut samples);
                let (vertex_count, index_count, cells) = count_surface(&samples, faces);
                Prepared {
                    node,
                    faces,
                    samples,
                    vertex_count,
                    index_count,
                    cells,
                }
            })
            .collect();

        let limits = device.limits();
        let max_bytes = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let mut jobs_raw = Vec::new();
        let mut grew = false;
        for job in prepared {
            self.free_node(job.node);
            self.dirty.remove(&job.node);
            let mut vertices = 0..0;
            let mut indices = 0..0;
            if job.index_count > 0 {
                let v =
                    self.vertex_arena
                        .alloc(device, encoder, job.vertex_count as u64, max_bytes);
                let i = self
                    .index_arena
                    .alloc(device, encoder, job.index_count as u64, max_bytes);
                match (v, i) {
                    (Some((v, gv)), Some((i, gi))) => {
                        grew |= gv || gi;
                        vertices = v;
                        indices = i;
                    }
                    (v, i) => {
                        if let Some((v, _)) = v {
                            self.vertex_arena.alloc.release(v);
                        }
                        if let Some((i, _)) = i {
                            self.index_arena.alloc.release(i);
                        }
                        eprintln!("voxel terrain: GPU mesh arena is full");
                        continue;
                    }
                }
            }
            let slot = self.take_slot(device, encoder);
            let cell = volume.voxel_size() * job.node.stride() as f32;
            let origin = job.node.key.as_vec3() * chunk_world;
            queue.write_buffer(
                &self.instances,
                slot as u64 * std::mem::size_of::<VoxelInstance>() as u64,
                bytemuck::bytes_of(&Self::instance_for(terrain, origin)),
            );
            let bounds = job.cells.map(|(lo, hi)| {
                let min = origin + lo.as_vec3() * cell;
                let max = origin + (hi + IVec3::ONE).as_vec3() * cell;
                Bounds {
                    centre: (min + max) * 0.5,
                    radius: (max - min).length() * 0.5,
                }
            });
            if !indices.is_empty() {
                let j = jobs_raw.len() as u64;
                queue.write_buffer(
                    &self.upload,
                    j * (GRID * GRID * GRID) as u64 * 4,
                    bytemuck::cast_slice(&job.samples),
                );
                jobs_raw.push(JobRaw {
                    cell,
                    vertex_base: vertices.start as u32,
                    index_base: indices.start as u32,
                    faces: job.faces,
                    vertex_count: job.vertex_count,
                    index_count: job.index_count,
                    _pad: [0; 2],
                });
            }
            self.nodes.insert(
                job.node,
                NodeMesh {
                    vertices,
                    indices,
                    slot,
                    faces: job.faces,
                    bounds,
                },
            );
            self.epoch += 1;
        }

        if rewrite_instances {
            for (key, node) in &self.nodes {
                let origin = key.key.as_vec3() * chunk_world;
                queue.write_buffer(
                    &self.instances,
                    node.slot as u64 * std::mem::size_of::<VoxelInstance>() as u64,
                    bytemuck::bytes_of(&Self::instance_for(terrain, origin)),
                );
            }
            self.epoch += 1;
        }

        if grew || self.bind.is_none() {
            self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel-mesh-bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.upload.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.jobs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cell_vertex.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.counters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.vertex_arena.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.index_arena.buffer.as_entire_binding(),
                    },
                ],
            }));
        }

        if !jobs_raw.is_empty() {
            queue.write_buffer(&self.jobs, 0, bytemuck::cast_slice(&jobs_raw));
            encoder.clear_buffer(&self.counters, 0, None);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxel-surface-nets"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, self.bind.as_ref(), &[]);
            let groups = ((CELLS * CELLS * CELLS) as u32).div_ceil(64);
            pass.set_pipeline(&self.vertices_pipeline);
            pass.dispatch_workgroups(groups, jobs_raw.len() as u32, 1);
            pass.set_pipeline(&self.indices_pipeline);
            pass.dispatch_workgroups(groups, jobs_raw.len() as u32, 1);
        }

        // Show the new cut once all of it is meshed; until then keep the old
        // one (edited regions in it were re-meshed in place above).
        let ready = desired.iter().all(|n| self.nodes.contains_key(n));
        if ready && self.drawn != desired {
            self.drawn = desired;
            self.epoch += 1;
            if std::env::var_os("ENGINE_VOXEL_STATS").is_some() {
                let mut per_level = vec![0usize; max_level as usize + 1];
                let mut triangles = 0u64;
                for key in &self.drawn {
                    per_level[key.level as usize] += 1;
                    if let Some(node) = self.nodes.get(key) {
                        triangles += (node.indices.end - node.indices.start) / 3;
                    }
                }
                eprintln!(
                    "voxel_terrain regions per level {per_level:?} triangles={triangles} \
                     arena vertices={} indices={}",
                    self.vertex_arena.alloc.capacity, self.index_arena.alloc.capacity
                );
            }
        } else if self.drawn.is_empty() {
            // First fill: show what is ready rather than nothing.
            let partial: Vec<NodeKey> = desired
                .iter()
                .copied()
                .filter(|n| self.nodes.contains_key(n))
                .collect();
            if !partial.is_empty() {
                self.drawn = partial;
                self.epoch += 1;
            }
        }
        let keep: HashSet<NodeKey> = self
            .drawn
            .iter()
            .chain(desired_set.iter())
            .copied()
            .collect();
        let stale: Vec<NodeKey> = self
            .nodes
            .keys()
            .copied()
            .filter(|k| !keep.contains(k))
            .collect();
        for key in stale {
            self.free_node(key);
            self.dirty.remove(&key);
        }
    }

    /// Regions in `frustum` (all drawn ones when `None`) with geometry.
    pub fn visible(&self, test: impl Fn(Bounds) -> bool) -> Vec<(Range<u32>, i32, u32)> {
        if !self.present {
            return Vec::new();
        }
        self.drawn
            .iter()
            .filter_map(|key| {
                let node = self.nodes.get(key)?;
                let bounds = node.bounds?;
                if node.indices.is_empty() || !test(bounds) {
                    return None;
                }
                Some((
                    node.indices.start as u32..node.indices.end as u32,
                    node.vertices.start as i32,
                    node.slot,
                ))
            })
            .collect()
    }

    pub fn visible_in(&self, frustum: &Frustum) -> Vec<(Range<u32>, i32, u32)> {
        self.visible(|b| frustum.intersects(b))
    }

    /// Bind the voxel buffers and draw `draws` with the pipeline already set.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, draws: &[(Range<u32>, i32, u32)]) {
        if draws.is_empty() {
            return;
        }
        pass.set_vertex_buffer(0, self.vertex_arena.buffer.slice(..));
        pass.set_vertex_buffer(1, self.instances.slice(..));
        pass.set_index_buffer(self.index_arena.buffer.slice(..), wgpu::IndexFormat::Uint32);
        for (indices, base_vertex, slot) in draws {
            pass.draw_indexed(indices.clone(), *base_vertex, *slot..*slot + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_alloc_reuses_and_coalesces() {
        let mut a = RangeAlloc::new(100);
        let x = a.alloc(40).unwrap();
        let y = a.alloc(40).unwrap();
        assert!(a.alloc(40).is_none());
        a.release(x);
        a.release(y);
        assert_eq!(a.free, [0..100]);
        a.grow(200);
        assert_eq!(a.alloc(200), Some(0..200));
    }

    #[test]
    fn flat_ground_counts_one_quad_per_column() {
        // Solid below grid y index PAD + 10.5: one horizontal sheet.
        let mut samples = Vec::new();
        for z in 0..GRID {
            for y in 0..GRID {
                for _x in 0..GRID {
                    let _ = z;
                    samples.push(if y <= PAD + 10 { 1.0 } else { -1.0 });
                }
            }
        }
        let (verts, indices, cells) = count_surface(&samples, 0);
        let n = CHUNK as u32;
        // Vertices for cells -1..32 in x and z, quads for edges 0..32.
        assert_eq!(verts, (n + 1) * (n + 1));
        assert_eq!(indices, n * n * 6);
        let (lo, hi) = cells.unwrap();
        assert_eq!((lo.y, hi.y), (10, 10));
        // A seam face on +x adds one column of quads.
        let (_, wider, _) = count_surface(&samples, 2);
        assert_eq!(wider, (n + 1) * n * 6);
    }
}
