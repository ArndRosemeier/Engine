//! GPU per-instance cull, for [`crate::world::InstanceSubmit::GpuIndirect`].
//!
//! Every culled instance batch writes the instances it keeps into a compact
//! page that mirrors its instance arena page slot for slot, and counts them
//! in its own pair of draw args (opaque, translucent) in one shared args
//! buffer. A view culls all of its batches with one reset dispatch and one
//! compact dispatch per instance page: a job table lists every batch range,
//! and each workgroup finds its job by a binary search over the table. So
//! the CPU cost of a view does not grow with the number of batches.
//!
//! The WGSL test is the same sphere transform and plane check as
//! [`super::frustum::Bounds::transformed`] + [`super::frustum::Frustum::intersects_sphere`].
//! A CPU helper in this module is the shader, so a mismatch is a test failure,
//! not a silent visual difference.

use super::frustum::{Bounds, Frustum};
use super::gpu_mesh::GpuMesh;
use crate::world::EntityId;
use bytemuck::{Pod, Zeroable};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

pub const WORKGROUP: u32 = 64;
pub const INDIRECT_STRIDE: u64 = std::mem::size_of::<DrawIndexedArgs>() as u64;
pub const OPAQUE_INDIRECT_OFFSET: u64 = 0;
pub const TRANSLUCENT_INDIRECT_OFFSET: u64 = INDIRECT_STRIDE;
/// Most workgroups one dispatch may start along x (WebGPU's limit).
const MAX_GROUPS: u32 = 65_535;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct DrawIndexedArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

impl DrawIndexedArgs {
    pub fn empty_range(index_count: u32, first_index: u32, base_vertex: i32) -> Self {
        Self {
            index_count,
            instance_count: 0,
            first_index,
            base_vertex,
            first_instance: 0,
        }
    }
}

/// One dispatch: the view's planes and its slice of the job (or reset) table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CullParams {
    planes: [[f32; 4]; 6],
    first: u32,
    count: u32,
    group_base: u32,
    _pad: u32,
}

/// One range of one batch's instances to test.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JobRaw {
    centre: [f32; 3],
    radius: f32,
    /// First instance in the page.
    source: u32,
    count: u32,
    /// The batch's first slot in the page (its compact output starts here).
    out_base: u32,
    /// The batch's pair of draw args.
    args: u32,
    slots: u32,
    /// The first workgroup of this job within the page's dispatch.
    group_start: u32,
    _pad: [u32; 2],
}

/// Copies one batch's opaque draw args to its place in a view's packed
/// args, so batches that share buffers and a texture go out as one
/// multi-draw (the instances start at the batch's compact slots).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PackRaw {
    src_args: u32,
    dst_args: u32,
    first_instance: u32,
    _pad: u32,
}

/// The draw args of one batch, written afresh before a view's compact.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResetRaw {
    args: u32,
    opaque: u32,
    total: u32,
    first_index: u32,
    base_vertex: i32,
    _pad: [u32; 3],
}

const SHADER: &str = r#"
struct CullParams {
    planes: array<vec4<f32>, 6>,
    first: u32,
    count: u32,
    group_base: u32,
    _pad: u32,
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
    // Material slot and padding (`InstanceRaw::material`).
    extra: vec4<u32>,
};

struct Job {
    centre: vec3<f32>,
    radius: f32,
    source: u32,
    count: u32,
    out_base: u32,
    args: u32,
    slots: u32,
    group_start: u32,
    _p0: u32,
    _p1: u32,
};

struct PlainArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct Pack {
    src_args: u32,
    dst_args: u32,
    first_instance: u32,
    _p0: u32,
};

struct Reset {
    args: u32,
    opaque: u32,
    total: u32,
    first_index: u32,
    base_vertex: i32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> params: CullParams;
@group(1) @binding(0) var<storage, read> src: array<Instance>;
@group(1) @binding(1) var<storage, read_write> dst: array<Instance>;
@group(1) @binding(2) var<storage, read_write> draws: array<DrawArgs>;
@group(1) @binding(3) var<storage, read> jobs: array<Job>;
@group(1) @binding(4) var<storage, read> resets: array<Reset>;
@group(1) @binding(5) var<storage, read_write> packed: array<PlainArgs>;
@group(1) @binding(6) var<storage, read> packs: array<Pack>;

fn intersects_sphere(centre: vec3<f32>, radius: f32) -> bool {
    for (var i = 0; i < 6; i++) {
        let p = params.planes[i];
        if dot(p.xyz, centre) + p.w < -radius {
            return false;
        }
    }
    return true;
}

// Empty both draws of each listed batch: full args, zero instances.
@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }
    let r = resets[params.first + i];
    let o = r.args * 2u;
    draws[o].index_count = r.opaque;
    atomicStore(&draws[o].instance_count, 0u);
    draws[o].first_index = r.first_index;
    draws[o].base_vertex = r.base_vertex;
    draws[o].first_instance = 0u;
    draws[o + 1u].index_count = r.total - r.opaque;
    atomicStore(&draws[o + 1u].instance_count, 0u);
    draws[o + 1u].first_index = r.first_index + r.opaque;
    draws[o + 1u].base_vertex = r.base_vertex;
    draws[o + 1u].first_instance = 0u;
}

// After the compact: each listed batch's opaque args to its packed place,
// its instances starting at its compact slots.
@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }
    let p = packs[params.first + i];
    var a: PlainArgs;
    a.index_count = draws[p.src_args].index_count;
    a.instance_count = atomicLoad(&draws[p.src_args].instance_count);
    a.first_index = draws[p.src_args].first_index;
    a.base_vertex = draws[p.src_args].base_vertex;
    a.first_instance = p.first_instance;
    packed[p.dst_args] = a;
}

@compute @workgroup_size(64)
fn compact(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let group = params.group_base + wg.x;
    // The last job starting at or before this workgroup.
    var lo = params.first;
    var hi = params.first + params.count;
    while hi - lo > 1u {
        let mid = (lo + hi) / 2u;
        if jobs[mid].group_start <= group {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let job = jobs[lo];
    let i = (group - job.group_start) * 64u + lane;
    if i >= job.count {
        return;
    }
    let inst = src[job.source + i];
    let model = inst.model;
    let scale = max(
        length(model[0].xyz),
        max(length(model[1].xyz), length(model[2].xyz)),
    );
    let world_c = (model * vec4<f32>(job.centre, 1.0)).xyz;
    let world_r = job.radius * scale;
    if !intersects_sphere(world_c, world_r) {
        return;
    }
    EXTRA_TEST
    let slot = atomicAdd(&draws[job.args * 2u].instance_count, 1u);
    dst[job.out_base + slot] = inst;
    if job.slots > 1u {
        atomicAdd(&draws[job.args * 2u + 1u].instance_count, 1u);
    }
}
"#;

/// The occlusion variant: also drops instances whose sphere's box lies
/// behind this frame's depth pyramid (see `occlusion.rs`).
const HIZ_TEST: &str = r#"
    if !may_show(world_c - vec3<f32>(world_r), world_c + vec3<f32>(world_r)) {
        return;
    }
"#;

const HIZ_BINDINGS: &str = r#"
@group(2) @binding(0) var<uniform> occ: Params;
@group(2) @binding(1) var pyramid: texture_2d<f32>;
"#;

/// Where one batch's culled instances and draw args live.
#[derive(Clone, Debug, PartialEq)]
pub struct CullTarget {
    /// The compact page mirroring the batch's instance page.
    compact: wgpu::Buffer,
    /// The batch's slots in it (the same as in its instance page).
    compact_bytes: Range<u64>,
    /// The batch's first slot in the page.
    out_base: u32,
    page: u32,
    args: wgpu::Buffer,
    /// Which pair of draw args in `args` is the batch's.
    args_index: u32,
}

impl CullTarget {
    /// The kept instances: bind as the instance vertex buffer.
    pub fn compact_slice(&self) -> wgpu::BufferSlice<'_> {
        self.compact.slice(self.compact_bytes.clone())
    }

    /// The compact buffer and where the batch's slots start, for keys.
    pub fn compact_at(&self) -> (&wgpu::Buffer, u64) {
        (&self.compact, self.compact_bytes.start)
    }

    /// Indirect args: [`OPAQUE_INDIRECT_OFFSET`] or
    /// [`TRANSLUCENT_INDIRECT_OFFSET`] into the batch's pair.
    pub fn args(&self, offset: u64) -> (&wgpu::Buffer, u64) {
        (
            &self.args,
            u64::from(self.args_index) * 2 * INDIRECT_STRIDE + offset,
        )
    }
}

pub struct CullJob<'a> {
    pub gpu: &'a GpuMesh,
    pub local: Bounds,
    pub draws: CullDraws,
    pub active_ranges: &'a [ActiveInstanceRange],
    /// Also copy its opaque args to this row of the packed args
    /// ([`InstanceCull::reserve_packed`]), for multi-draws.
    pub pack: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveInstanceRange {
    pub start: u32,
    pub count: u32,
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

/// The compact page mirroring one instance arena page.
struct CompactPage {
    src: wgpu::Buffer,
    dst: wgpu::Buffer,
    /// Its bind group and the [`InstanceCull::bind_gen`] it was made at.
    bind: Option<(u64, wgpu::BindGroup)>,
}

/// A storage table written by cursor during a frame and never rewritten
/// before the frame's commands ran: a full table is replaced by a bigger
/// one (commands recorded earlier keep the old one alive).
struct FrameTable {
    label: &'static str,
    usage: wgpu::BufferUsages,
    stride: u64,
    buf: wgpu::Buffer,
    capacity: usize,
    cursor: usize,
}

impl FrameTable {
    fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
        stride: u64,
    ) -> Self {
        let capacity = 64;
        Self {
            label,
            usage,
            stride,
            buf: Self::buffer(device, label, usage, stride, capacity),
            capacity,
            cursor: 0,
        }
    }

    fn buffer(
        device: &wgpu::Device,
        label: &str,
        usage: wgpu::BufferUsages,
        stride: u64,
        capacity: usize,
    ) -> wgpu::Buffer {
        let size = stride
            .checked_mul(capacity as u64)
            .expect("instance-cull table size overflow");
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Room for `len` more rows. True when the buffer was replaced.
    fn reserve(&mut self, device: &wgpu::Device, len: usize) -> bool {
        let end = self
            .cursor
            .checked_add(len)
            .expect("instance-cull table cursor overflow");
        if end <= self.capacity {
            return false;
        }
        let capacity = len
            .max(self.capacity * 2)
            .checked_next_power_of_two()
            .expect("instance-cull table capacity overflow");
        self.buf = Self::buffer(device, self.label, self.usage, self.stride, capacity);
        self.capacity = capacity;
        self.cursor = 0;
        true
    }

    /// Write `bytes` (whole rows) at the cursor; returns the first row.
    fn push(&mut self, queue: &wgpu::Queue, bytes: &[u8]) -> usize {
        let rows = bytes.len() / self.stride as usize;
        debug_assert_eq!(rows * self.stride as usize, bytes.len());
        assert!(
            self.cursor + rows <= self.capacity,
            "instance-cull table overrun"
        );
        let first = self.cursor;
        queue.write_buffer(&self.buf, first as u64 * self.stride, bytes);
        self.cursor += rows;
        first
    }
}

pub struct InstanceCull {
    pipeline: wgpu::ComputePipeline,
    /// Frustum and occlusion test (main view after the depth pyramid).
    hiz_pipeline: wgpu::ComputePipeline,
    reset_pipeline: wgpu::ComputePipeline,
    pack_pipeline: wgpu::ComputePipeline,
    hiz_layout: wgpu::BindGroupLayout,
    params_layout: wgpu::BindGroupLayout,
    tables_layout: wgpu::BindGroupLayout,
    /// Uniform rows, one per dispatch (dynamic offsets).
    params: FrameTable,
    params_bind: wgpu::BindGroup,
    params_scratch: Vec<u8>,
    jobs: FrameTable,
    resets: FrameTable,
    /// Views' packed opaque args (written by the GPU) and the copies
    /// that fill them.
    packed: FrameTable,
    packs: FrameTable,
    pack_scratch: Vec<PackRaw>,
    job_scratch: Vec<JobRaw>,
    reset_scratch: Vec<ResetRaw>,
    /// Compact pages by instance arena page.
    pages: HashMap<u32, CompactPage>,
    /// Every batch's pair of draw args.
    args_buf: wgpu::Buffer,
    args_capacity: u32,
    args_slots: HashMap<EntityId, u32>,
    args_free: Vec<u32>,
    args_next: u32,
    /// Raised whenever a buffer the page bind groups hold is replaced.
    bind_gen: u64,
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
        let tables_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance-cull-tables"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                storage(3, true),
                storage(4, true),
                storage(5, false),
                storage(6, true),
            ],
        });
        let hiz_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance-cull-hiz"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let build = |label: &str, src: String, entry: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            })
        };
        let plain_src = SHADER.replace("EXTRA_TEST", "");
        let pipeline = build(
            "instance-cull-pipeline",
            plain_src.clone(),
            "compact",
            &[&params_layout, &tables_layout],
        );
        let reset_pipeline = build(
            "instance-cull-reset-pipeline",
            plain_src.clone(),
            "reset",
            &[&params_layout, &tables_layout],
        );
        let pack_pipeline = build(
            "instance-cull-pack-pipeline",
            plain_src,
            "pack",
            &[&params_layout, &tables_layout],
        );
        let hiz_src = format!(
            "{}{HIZ_BINDINGS}{}",
            SHADER.replace("EXTRA_TEST", HIZ_TEST),
            super::occlusion::BOX_TEST_WGSL
        );
        let hiz_pipeline = build(
            "instance-cull-hiz-pipeline",
            hiz_src,
            "compact",
            &[&params_layout, &tables_layout, &hiz_layout],
        );
        let params = FrameTable::new(
            device,
            "instance-cull-params",
            wgpu::BufferUsages::UNIFORM,
            params_stride,
        );
        let params_bind = params_bind(device, &params_layout, &params.buf);
        let storage_usage = wgpu::BufferUsages::STORAGE;
        let jobs = FrameTable::new(
            device,
            "instance-cull-jobs",
            storage_usage,
            std::mem::size_of::<JobRaw>() as u64,
        );
        let resets = FrameTable::new(
            device,
            "instance-cull-resets",
            storage_usage,
            std::mem::size_of::<ResetRaw>() as u64,
        );
        let packed = FrameTable::new(
            device,
            "instance-cull-packed-args",
            wgpu::BufferUsages::INDIRECT | storage_usage,
            INDIRECT_STRIDE,
        );
        let packs = FrameTable::new(
            device,
            "instance-cull-packs",
            storage_usage,
            std::mem::size_of::<PackRaw>() as u64,
        );
        let args_capacity = 64;
        Self {
            pipeline,
            hiz_pipeline,
            reset_pipeline,
            pack_pipeline,
            hiz_layout,
            params_layout,
            tables_layout,
            params,
            params_bind,
            params_scratch: Vec::new(),
            jobs,
            resets,
            packed,
            packs,
            pack_scratch: Vec::new(),
            job_scratch: Vec::new(),
            reset_scratch: Vec::new(),
            pages: HashMap::new(),
            args_buf: args_buffer(device, args_capacity),
            args_capacity,
            args_slots: HashMap::new(),
            args_free: Vec::new(),
            args_next: 0,
            bind_gen: 0,
        }
    }

    /// Start a frame's culls. `views`, `batches` and `ranges` size the
    /// frame's tables up front (a table that still fills up is replaced by
    /// a bigger one mid-frame, which is safe but costs a buffer).
    pub fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        views: usize,
        batches: usize,
        ranges: usize,
    ) {
        self.params.cursor = 0;
        self.jobs.cursor = 0;
        self.resets.cursor = 0;
        self.packed.cursor = 0;
        self.packs.cursor = 0;
        let mut replaced = false;
        // A reset and about one compact dispatch per page and view.
        let pages = self.pages.len().max(1);
        if self.params.reserve(device, views * (pages + 2)) {
            self.params_bind = params_bind(device, &self.params_layout, &self.params.buf);
        }
        replaced |= self.jobs.reserve(device, views * ranges);
        replaced |= self.resets.reserve(device, views * batches);
        if replaced {
            self.bind_gen += 1;
        }
    }

    /// Give every GPU-culled batch its compact page and draw args, and set
    /// each one's [`GpuMesh::cull_target`]. True when a batch that had a
    /// target got another one: bundles that bake it must be recorded anew.
    pub fn place(
        &mut self,
        device: &wgpu::Device,
        batches: &mut [(EntityId, &mut GpuMesh)],
    ) -> bool {
        let live: HashSet<EntityId> = batches.iter().map(|(id, _)| *id).collect();
        let free = &mut self.args_free;
        self.args_slots.retain(|id, slot| {
            let keep = live.contains(id);
            if !keep {
                free.push(*slot);
            }
            keep
        });
        for (id, _) in batches.iter() {
            if !self.args_slots.contains_key(id) {
                let slot = self.args_free.pop().unwrap_or_else(|| {
                    let slot = self.args_next;
                    self.args_next += 1;
                    slot
                });
                self.args_slots.insert(*id, slot);
            }
        }
        if self.args_next > self.args_capacity {
            self.args_capacity = self
                .args_next
                .checked_next_power_of_two()
                .expect("instance-cull args capacity overflow");
            self.args_buf = args_buffer(device, self.args_capacity);
            self.bind_gen += 1;
        }

        let mut used = HashSet::new();
        for (_, gpu) in batches.iter() {
            let page = gpu.instance_page();
            used.insert(page);
            let fresh = self
                .pages
                .get(&page)
                .is_none_or(|p| p.src != gpu.instance_buf);
            if fresh {
                let dst = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instance-cull-compact-page"),
                    size: gpu.instance_buf.size(),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                self.pages.insert(
                    page,
                    CompactPage {
                        src: gpu.instance_buf.clone(),
                        dst,
                        bind: None,
                    },
                );
            }
        }
        self.pages.retain(|page, _| used.contains(page));

        let mut moved = false;
        for (id, gpu) in batches.iter_mut() {
            let page = gpu.instance_page();
            let target = CullTarget {
                compact: self.pages[&page].dst.clone(),
                compact_bytes: gpu.instance_bytes(),
                out_base: gpu.first_instance,
                page,
                args: self.args_buf.clone(),
                args_index: self.args_slots[id],
            };
            if gpu.cull_target.as_ref() != Some(&target) {
                moved |= gpu.cull_target.is_some();
                gpu.cull_target = Some(target);
            }
        }
        moved
    }

    /// `rows` packed draw args for a view this frame: their buffer and
    /// first row. The GPU fills them ([`CullJob::pack`]); a multi-draw
    /// reads a run of them. Not rewritten until the next frame.
    pub fn reserve_packed(&mut self, device: &wgpu::Device, rows: usize) -> (wgpu::Buffer, u32) {
        if self.packed.reserve(device, rows) {
            self.bind_gen += 1;
        }
        let first = self.packed.cursor;
        self.packed.cursor += rows;
        (
            self.packed.buf.clone(),
            u32::try_from(first).expect("packed args row exceeds u32"),
        )
    }

    /// Video memory the cull holds (bytes): compact pages, args, tables.
    pub fn reserved_bytes(&self) -> u64 {
        self.pages.values().map(|p| p.dst.size()).sum::<u64>()
            + self.args_buf.size()
            + self.params.buf.size()
            + self.jobs.buf.size()
            + self.resets.buf.size()
            + self.packed.buf.size()
            + self.packs.buf.size()
    }

    /// The occlusion pass's uniform and depth pyramid, for [`Self::dispatch`].
    pub fn hiz_bind(
        &self,
        device: &wgpu::Device,
        test: &wgpu::Buffer,
        pyramid: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance-cull-hiz"),
            layout: &self.hiz_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: test.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(pyramid),
                },
            ],
        })
    }

    /// The bind group of `page`'s compact page with this frame's tables.
    fn page_bind(&mut self, device: &wgpu::Device, page: u32) -> wgpu::BindGroup {
        let gen = self.bind_gen;
        let p = self
            .pages
            .get_mut(&page)
            .unwrap_or_else(|| panic!("instance page {page} was not placed for the cull"));
        if let Some((at, bind)) = &p.bind {
            if *at == gen {
                return bind.clone();
            }
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("instance-cull-tables"),
            layout: &self.tables_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p.src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p.dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.args_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.jobs.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.resets.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.packed.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.packs.buf.as_entire_binding(),
                },
            ],
        });
        p.bind = Some((gen, bind.clone()));
        bind
    }

    /// Compact each job's instances to those in `frustum` and, with `hiz`
    /// (a [`Self::hiz_bind`] group), not hidden behind the depth pyramid.
    /// Each batch must appear once. Returns the dispatches recorded.
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frustum: &Frustum,
        jobs: &[CullJob<'_>],
        hiz: Option<&wgpu::BindGroup>,
    ) -> u32 {
        if jobs.is_empty() {
            return 0;
        }
        let planes = frustum.planes().map(|p| [p.x, p.y, p.z, p.w]);
        let plan: Vec<PlanJob<'_>> = jobs
            .iter()
            .map(|job| {
                let target = target_of(job.gpu);
                PlanJob {
                    page: target.page,
                    args: target.args_index,
                    out_base: target.out_base,
                    first_instance: job.gpu.first_instance,
                    opaque: u32::try_from(job.gpu.opaque_index_count)
                        .expect("opaque index count exceeds GPU u32 range"),
                    total: u32::try_from(job.gpu.index_count)
                        .expect("index count exceeds GPU u32 range"),
                    first_index: job.gpu.first_index,
                    base_vertex: job.gpu.base_vertex,
                    slots: job.draws.slots(),
                    local: job.local,
                    ranges: job.active_ranges,
                }
            })
            .collect();
        let page_runs = plan_tables(&plan, &mut self.reset_scratch, &mut self.job_scratch);
        self.pack_scratch.clear();
        for job in jobs {
            if let Some(row) = job.pack {
                let target = target_of(job.gpu);
                self.pack_scratch.push(PackRaw {
                    src_args: target.args_index * 2,
                    dst_args: row,
                    first_instance: target.out_base,
                    _pad: 0,
                });
            }
        }
        let pack_count = u32::try_from(self.pack_scratch.len()).expect("pack rows exceed u32");

        // Room for the rows; a replaced table needs new bind groups.
        let compacts = page_runs
            .iter()
            .map(|run| run.groups.div_ceil(MAX_GROUPS) as usize)
            .sum::<usize>();
        let dispatches = 1 + compacts + usize::from(pack_count > 0);
        let mut replaced = self.jobs.reserve(device, self.job_scratch.len());
        replaced |= self.resets.reserve(device, self.reset_scratch.len());
        replaced |= self.packs.reserve(device, self.pack_scratch.len());
        if replaced {
            self.bind_gen += 1;
        }
        if self.params.reserve(device, dispatches) {
            self.params_bind = params_bind(device, &self.params_layout, &self.params.buf);
        }
        let job_first = self
            .jobs
            .push(queue, bytemuck::cast_slice(&self.job_scratch));
        let reset_first = self
            .resets
            .push(queue, bytemuck::cast_slice(&self.reset_scratch));

        // One parameter row per dispatch: the reset, then each page's chunks.
        let stride = self.params.stride as usize;
        self.params_scratch.clear();
        self.params_scratch.resize(stride * dispatches, 0);
        let mut rows: Vec<(u32, u32)> = Vec::with_capacity(dispatches); // (page, groups)
        let write_row = |scratch: &mut Vec<u8>, at: usize, row: CullParams| {
            let bytes = bytemuck::bytes_of(&row);
            scratch[at * stride..at * stride + bytes.len()].copy_from_slice(bytes);
        };
        let reset_count = u32::try_from(self.reset_scratch.len()).expect("reset rows exceed u32");
        write_row(
            &mut self.params_scratch,
            0,
            CullParams {
                planes,
                first: u32::try_from(reset_first).expect("reset row exceeds u32"),
                count: reset_count,
                group_base: 0,
                _pad: 0,
            },
        );
        for &PageRun {
            page,
            first,
            rows: count,
            groups,
        } in &page_runs
        {
            let mut base = 0;
            while base < groups {
                let chunk = (groups - base).min(MAX_GROUPS);
                write_row(
                    &mut self.params_scratch,
                    1 + rows.len(),
                    CullParams {
                        planes,
                        first: u32::try_from(job_first + first).expect("job row exceeds u32"),
                        count: u32::try_from(count).expect("job rows exceed u32"),
                        group_base: base,
                        _pad: 0,
                    },
                );
                rows.push((page, chunk));
                base += chunk;
            }
        }
        if pack_count > 0 {
            let pack_first = self
                .packs
                .push(queue, bytemuck::cast_slice(&self.pack_scratch));
            write_row(
                &mut self.params_scratch,
                1 + compacts,
                CullParams {
                    planes,
                    first: u32::try_from(pack_first).expect("pack row exceeds u32"),
                    count: pack_count,
                    group_base: 0,
                    _pad: 0,
                },
            );
        }
        let params_first = self.params.push(queue, &self.params_scratch);
        let offset = |row: usize| {
            u32::try_from((params_first + row) as u64 * self.params.stride)
                .expect("instance-cull dynamic uniform offset exceeds u32")
        };
        let offsets: Vec<u32> = (0..dispatches).map(offset).collect();
        let binds: Vec<(u32, wgpu::BindGroup)> = {
            let mut binds = Vec::new();
            for run in &page_runs {
                binds.push((run.page, self.page_bind(device, run.page)));
            }
            if binds.is_empty() {
                binds.push((plan[0].page, self.page_bind(device, plan[0].page)));
            }
            binds
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("instance-cull"),
            timestamp_writes: None,
        });
        // Reset every listed batch's args. WebGPU orders the dispatches of a
        // pass: each sees the storage writes of the ones before it.
        pass.set_pipeline(&self.reset_pipeline);
        pass.set_bind_group(0, &self.params_bind, &[offsets[0]]);
        pass.set_bind_group(1, &binds[0].1, &[]);
        pass.dispatch_workgroups(reset_count.div_ceil(WORKGROUP), 1, 1);
        match hiz {
            Some(bind) => {
                pass.set_pipeline(&self.hiz_pipeline);
                pass.set_bind_group(2, bind, &[]);
            }
            None => pass.set_pipeline(&self.pipeline),
        }
        let mut bound = None;
        for (row, &(page, groups)) in rows.iter().enumerate() {
            if bound != Some(page) {
                let bind = &binds
                    .iter()
                    .find(|(p, _)| *p == page)
                    .expect("every page run has a bind group")
                    .1;
                pass.set_bind_group(1, bind, &[]);
                bound = Some(page);
            }
            pass.set_bind_group(0, &self.params_bind, &[offsets[1 + row]]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if pack_count > 0 {
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &self.params_bind, &[offsets[1 + compacts]]);
            pass.dispatch_workgroups(pack_count.div_ceil(WORKGROUP), 1, 1);
        }
        drop(pass);
        u32::try_from(dispatches).expect("dispatch count exceeds u32")
    }
}

/// What [`plan_tables`] needs of one batch's cull.
struct PlanJob<'a> {
    page: u32,
    args: u32,
    out_base: u32,
    first_instance: u32,
    opaque: u32,
    total: u32,
    first_index: u32,
    base_vertex: i32,
    slots: u32,
    local: Bounds,
    ranges: &'a [ActiveInstanceRange],
}

/// One instance page's slice of the job table and its workgroups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PageRun {
    page: u32,
    /// First row in the job table (of this view's rows).
    first: usize,
    rows: usize,
    groups: u32,
}

/// A view's tables: one reset row per batch, and the job rows grouped by
/// instance page with each row's first workgroup, ascending within the
/// page (the shader's binary search relies on it).
fn plan_tables(
    jobs: &[PlanJob<'_>],
    resets: &mut Vec<ResetRaw>,
    rows: &mut Vec<JobRaw>,
) -> Vec<PageRun> {
    resets.clear();
    rows.clear();
    let mut order: Vec<(u32, usize)> = jobs.iter().enumerate().map(|(i, j)| (j.page, i)).collect();
    order.sort_unstable();
    let mut runs: Vec<PageRun> = Vec::new();
    for &(page, i) in &order {
        let job = &jobs[i];
        resets.push(ResetRaw {
            args: job.args,
            opaque: job.opaque,
            total: job.total,
            first_index: job.first_index,
            base_vertex: job.base_vertex,
            _pad: [0; 3],
        });
        if runs.last().is_none_or(|run| run.page != page) {
            runs.push(PageRun {
                page,
                first: rows.len(),
                rows: 0,
                groups: 0,
            });
        }
        let run = runs.last_mut().expect("a run was just pushed");
        for range in job.ranges {
            assert!(
                range.count > 0,
                "instance cull active range must not be empty"
            );
            rows.push(JobRaw {
                centre: job.local.centre.into(),
                radius: job.local.radius,
                source: job.first_instance + range.start,
                count: range.count,
                out_base: job.out_base,
                args: job.args,
                slots: job.slots,
                group_start: run.groups,
                _pad: [0; 2],
            });
            run.rows += 1;
            run.groups = run
                .groups
                .checked_add(range.count.div_ceil(WORKGROUP))
                .expect("instance-cull workgroup count overflow");
        }
    }
    runs.retain(|run| run.rows > 0);
    runs
}

fn target_of(gpu: &GpuMesh) -> &CullTarget {
    gpu.cull_target
        .as_ref()
        .expect("a GPU-culled batch was placed (InstanceCull::place) before its cull")
}

fn args_buffer(device: &wgpu::Device, pairs: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instance-cull-args"),
        size: u64::from(pairs) * 2 * INDIRECT_STRIDE,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn params_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("instance-cull-params-bind"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<CullParams>() as u64),
            }),
        }],
    })
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

    /// The shader's job lookup on the CPU: the last row of the run whose
    /// first workgroup is at or before `group`.
    fn find_job(rows: &[JobRaw], run: PageRun, group: u32) -> usize {
        let (mut lo, mut hi) = (run.first, run.first + run.rows);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if rows[mid].group_start <= group {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    #[test]
    fn every_instance_is_tested_once_by_the_right_batch() {
        let local = Bounds {
            centre: Vec3::ZERO,
            radius: 1.0,
        };
        let r = |start, count| ActiveInstanceRange { start, count };
        let a = [r(0, 1), r(5, 130), r(200, 64)];
        let b = [r(0, 65)];
        let c = [r(3, 7), r(40, 1000)];
        let d = [r(10, 64)];
        let job = |page, args, first_instance, ranges| PlanJob {
            page,
            args,
            out_base: first_instance,
            first_instance,
            opaque: 6,
            total: 12,
            first_index: 30,
            base_vertex: 4,
            slots: 1,
            local,
            ranges,
        };
        // Pages out of order, so the plan must group them.
        let jobs = [
            job(1, 0, 0, &a[..]),
            job(0, 1, 500, &b[..]),
            job(1, 2, 4000, &c[..]),
            job(0, 3, 9000, &d[..]),
        ];
        let (mut resets, mut rows) = (Vec::new(), Vec::new());
        let runs = plan_tables(&jobs, &mut resets, &mut rows);
        assert_eq!(resets.len(), jobs.len());
        assert_eq!(runs.iter().map(|r| r.page).collect::<Vec<_>>(), vec![0, 1]);
        let mut seen: HashMap<(u32, u32), u32> = HashMap::new();
        for run in &runs {
            for group in 0..run.groups {
                let row = rows[find_job(&rows, *run, group)];
                for lane in 0..WORKGROUP {
                    let i = (group - row.group_start) * WORKGROUP + lane;
                    if i < row.count {
                        *seen.entry((run.page, row.source + i)).or_default() += 1;
                        let owner = jobs
                            .iter()
                            .find(|j| j.args == row.args)
                            .expect("row names a batch");
                        assert_eq!(owner.page, run.page);
                        assert_eq!(row.out_base, owner.out_base);
                    }
                }
            }
        }
        let mut expected = 0;
        for j in &jobs {
            for range in j.ranges {
                for k in 0..range.count {
                    expected += 1;
                    let at = (j.page, j.first_instance + range.start + k);
                    assert_eq!(seen.get(&at), Some(&1), "instance {at:?} tested once");
                }
            }
        }
        assert_eq!(seen.len(), expected);
    }

    #[test]
    fn gpu_structs_match_their_wgsl_layout() {
        // WGSL: vec3 + 9 scalars round to 48; 8 scalars are 32; planes + 4.
        assert_eq!(std::mem::size_of::<JobRaw>(), 48);
        assert_eq!(std::mem::size_of::<ResetRaw>(), 32);
        assert_eq!(std::mem::size_of::<CullParams>(), 112);
        assert_eq!(std::mem::size_of::<PackRaw>(), 16);
        assert_eq!(INDIRECT_STRIDE, 20);
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
