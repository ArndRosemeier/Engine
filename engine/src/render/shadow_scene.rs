//! The GPU shadow scene: every shadow-casting mesh of the live space as a
//! row kept on the GPU, like the opaque GPU scene (`gpu_scene.rs`). One
//! compute pass tests every row against every shadow view drawn this frame
//! (sun cascades, their static-cache layers, lamp faces) and writes each
//! view's draws, packed and counted per multi-draw group, so the shadow
//! passes draw with `multi_draw_indexed_indirect_count` and the CPU no
//! longer walks the casters for each view.
//!
//! The view test is the CPU one (`Frustum::classify` on the mesh's bounding
//! sphere), so the same casters draw. Rows carry whether a mesh counts as
//! static for the sun's shadow cache; a cached cascade's live view draws
//! only the moving rows and its cache layer only the static ones.

use super::frustum::{Bounds, Frustum};
use super::instance_cull::DrawIndexedArgs;
use crate::portal::SpaceId;
use crate::world::EntityId;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

/// Whether GPU shadow culling is switched off (`ENGINE_GPU_SHADOWS=0`).
pub fn disabled_by_env() -> bool {
    std::env::var("ENGINE_GPU_SHADOWS").is_ok_and(|v| v.trim() == "0")
}

/// Shadow views a frame can draw: the bundle slots of `mod.rs` (cascades,
/// lamp layers, then the cascades' static layers).
pub const MAX_VIEWS: usize = 6 + super::lights::SHADOW_LAYERS;

/// What a view draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ViewMode {
    All = 0,
    /// A cached cascade's live view: casters not counted as static.
    Moving = 1,
    /// A cascade's cache layer: the static casters.
    Static = 2,
}

/// One row as the GPU reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct GpuRow {
    sphere: [f32; 4],
    index_count: u32,
    /// 0 when a room hides the mesh.
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
    /// 1 when the mesh counts as static for the sun's shadow cache.
    fixed: u32,
    group: u32,
    group_start: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    rows: u32,
    views: u32,
    capacity: u32,
    _pad: u32,
    planes: [[f32; 4]; MAX_VIEWS * 6],
    /// Per view its `ViewMode`, or 3: not drawn this frame.
    modes: [[u32; 4]; MAX_VIEWS.div_ceil(4)],
}

const SKIP: u32 = 3;
const GROUP: u32 = 64;

const CULL: &str = r#"
struct Row {
    sphere: vec4<f32>,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
    fixed: u32,
    group: u32,
    group_start: u32,
};

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct Params {
    rows: u32,
    views: u32,
    capacity: u32,
    _pad: u32,
    planes: array<vec4<f32>, PLANES>,
    modes: array<vec4<u32>, MODES>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> rows: array<Row>;
@group(0) @binding(2) var<storage, read_write> draws: array<DrawArgs>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= p.rows {
        return;
    }
    let r = rows[i];
    if r.instance_count == 0u {
        return;
    }
    for (var v = 0u; v < p.views; v++) {
        let mode = p.modes[v / 4u][v % 4u];
        if mode == 3u || (mode == 1u && r.fixed == 1u) || (mode == 2u && r.fixed == 0u) {
            continue;
        }
        var outside = false;
        for (var k = 0u; k < 6u; k++) {
            let plane = p.planes[v * 6u + k];
            if dot(plane.xyz, r.sphere.xyz) + plane.w < -r.sphere.w {
                outside = true;
            }
        }
        if outside {
            continue;
        }
        let at = atomicAdd(&counts[v * p.capacity + r.group], 1u);
        draws[v * p.capacity + r.group_start + at] = DrawArgs(
            r.index_count, r.instance_count, r.first_index, r.base_vertex, r.first_instance);
    }
}
"#;

/// What one shadow multi-draw shares.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ShadowKey {
    /// Through the alpha-tested pipeline with `albedo`.
    pub cutout: bool,
    pub albedo: Option<wgpu::BindGroup>,
    pub vertex: wgpu::Buffer,
    pub index: wgpu::Buffer,
    pub instances: wgpu::Buffer,
}

/// A caster as the CPU worked it out this frame.
pub struct ShadowRow {
    pub id: EntityId,
    /// Its place among the shadow sources.
    pub order: u32,
    pub key: ShadowKey,
    pub bounds: Bounds,
    pub args: DrawIndexedArgs,
    pub visible: bool,
    pub fixed: bool,
}

#[derive(Clone, Debug)]
pub struct ShadowGroup {
    pub key: ShadowKey,
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShadowStamp {
    pub render: u64,
    pub instance: u64,
    pub resource: u64,
    pub space: SpaceId,
}

pub struct ShadowScene {
    pub stamp: Option<ShadowStamp>,
    ids: Vec<EntityId>,
    /// Bounds and static flag per row, for the cache's change test.
    meta: Vec<(Bounds, bool)>,
    gpu_rows: Vec<GpuRow>,
    pub groups: Vec<ShadowGroup>,
    capacity: usize,
    rows_buf: wgpu::Buffer,
    pub args: wgpu::Buffer,
    pub counts: wgpu::Buffer,
    params: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    bind: Option<wgpu::BindGroup>,
}

impl ShadowScene {
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
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
            label: Some("shadow-scene-cull"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Params>() as u64
                        ),
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
            ],
        });
        let src = CULL
            .replace("PLANES", &(MAX_VIEWS * 6).to_string())
            .replace("MODES", &MAX_VIEWS.div_ceil(4).to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-scene-cull"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-scene-cull"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shadow-scene-cull"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-scene-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (rows_buf, args, counts) = buffers(device, 1);
        Self {
            stamp: None,
            ids: Vec::new(),
            meta: Vec::new(),
            gpu_rows: Vec::new(),
            groups: Vec::new(),
            capacity: 1,
            rows_buf,
            args,
            counts,
            params,
            layout,
            pipeline,
            bind: None,
        }
    }

    /// Take this frame's casters (any order); write the rows that changed.
    /// Returns the bounds of static casters that came, went or changed (the
    /// cached cascade layers they touch are out of date).
    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rows: Vec<ShadowRow>,
    ) -> Vec<Bounds> {
        // Plain casters first, then the cutouts; groups in the order of
        // their first caster, casters in source order.
        #[allow(clippy::mutable_key_type)]
        let mut first: HashMap<&ShadowKey, u32> = HashMap::new();
        for r in &rows {
            let at = first.entry(&r.key).or_insert(r.order);
            *at = (*at).min(r.order);
        }
        let mut sort: Vec<(bool, u32, u32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.key.cutout, first[&r.key], r.order, i))
            .collect();
        drop(first);
        sort.sort_unstable();
        let mut taken: Vec<Option<ShadowRow>> = rows.into_iter().map(Some).collect();
        let rows: Vec<ShadowRow> = sort
            .into_iter()
            .map(|(.., i)| taken[i].take().expect("each row once"))
            .collect();

        let mut groups: Vec<ShadowGroup> = Vec::new();
        let mut group_of = Vec::with_capacity(rows.len());
        for (i, r) in rows.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if g.key == r.key => g.len += 1,
                _ => groups.push(ShadowGroup {
                    key: r.key.clone(),
                    start: i as u32,
                    len: 1,
                }),
            }
            let g = groups.len() - 1;
            group_of.push((g as u32, groups[g].start));
        }
        let gpu_rows: Vec<GpuRow> = rows
            .iter()
            .zip(&group_of)
            .map(|(r, &(group, group_start))| GpuRow {
                sphere: [
                    r.bounds.centre.x,
                    r.bounds.centre.y,
                    r.bounds.centre.z,
                    r.bounds.radius,
                ],
                index_count: r.args.index_count,
                instance_count: if r.visible { r.args.instance_count } else { 0 },
                first_index: r.args.first_index,
                base_vertex: r.args.base_vertex,
                first_instance: r.args.first_instance,
                fixed: u32::from(r.fixed),
                group,
                group_start,
            })
            .collect();

        // Static casters that changed, by id: old and new places.
        let old: HashMap<EntityId, (GpuRow, (Bounds, bool))> = self
            .ids
            .iter()
            .zip(self.gpu_rows.iter().zip(&self.meta))
            .map(|(id, (row, meta))| (*id, (*row, *meta)))
            .collect();
        let mut changed = Vec::new();
        let mut kept = 0;
        for (r, row) in rows.iter().zip(&gpu_rows) {
            match old.get(&r.id) {
                Some((was, (bounds, fixed))) => {
                    kept += 1;
                    let same = drawn_same(was, row);
                    if (*fixed || r.fixed) && (!same || *fixed != r.fixed) {
                        changed.push(*bounds);
                        changed.push(r.bounds);
                    }
                }
                None if r.fixed => changed.push(r.bounds),
                None => {}
            }
        }
        if kept < old.len() {
            let now: std::collections::HashSet<EntityId> = rows.iter().map(|r| r.id).collect();
            for (id, (_, (bounds, fixed))) in &old {
                if *fixed && !now.contains(id) {
                    changed.push(*bounds);
                }
            }
        }

        let all = if gpu_rows.len() > self.capacity {
            self.capacity = gpu_rows
                .len()
                .checked_next_power_of_two()
                .expect("shadow row capacity overflow");
            (self.rows_buf, self.args, self.counts) = buffers(device, self.capacity);
            self.bind = None;
            true
        } else {
            false
        };
        let stride = std::mem::size_of::<GpuRow>();
        let mut i = 0;
        while i < gpu_rows.len() {
            if !all && self.gpu_rows.get(i) == Some(&gpu_rows[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < gpu_rows.len() && (all || self.gpu_rows.get(i) != Some(&gpu_rows[i])) {
                i += 1;
            }
            queue.write_buffer(
                &self.rows_buf,
                (start * stride) as u64,
                bytemuck::cast_slice(&gpu_rows[start..i]),
            );
        }
        self.ids = rows.iter().map(|r| r.id).collect();
        self.meta = rows.iter().map(|r| (r.bounds, r.fixed)).collect();
        self.gpu_rows = gpu_rows;
        self.groups = groups;
        changed
    }

    /// Recheck which rows count as static; returns the bounds of those that
    /// changed (see [`Self::apply`]).
    pub fn refresh_fixed(
        &mut self,
        queue: &wgpu::Queue,
        mut fixed: impl FnMut(EntityId) -> bool,
    ) -> Vec<Bounds> {
        let mut changed = Vec::new();
        let stride = std::mem::size_of::<GpuRow>();
        for (i, id) in self.ids.iter().enumerate() {
            let now = fixed(*id);
            if now != self.meta[i].1 {
                self.meta[i].1 = now;
                self.gpu_rows[i].fixed = u32::from(now);
                changed.push(self.meta[i].0);
                queue.write_buffer(
                    &self.rows_buf,
                    (i * stride) as u64,
                    bytemuck::bytes_of(&self.gpu_rows[i]),
                );
            }
        }
        changed
    }

    /// Test every row against the views (`views[slot]`: its frustum and
    /// mode, `None` when not drawn) and write their draws.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        views: &[Option<(Frustum, ViewMode)>],
    ) {
        let mut params = Params::zeroed();
        params.rows = u32::try_from(self.ids.len()).expect("shadow rows exceed u32");
        params.views = u32::try_from(views.len().min(MAX_VIEWS)).expect("view count");
        params.capacity = u32::try_from(self.capacity).expect("shadow capacity exceeds u32");
        for (v, view) in views.iter().take(MAX_VIEWS).enumerate() {
            let mode = match view {
                Some((frustum, mode)) => {
                    for (k, plane) in frustum.planes().iter().enumerate() {
                        params.planes[v * 6 + k] = plane.to_array();
                    }
                    *mode as u32
                }
                None => SKIP,
            };
            params.modes[v / 4][v % 4] = mode;
        }
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(&self.counts, 0, None);
        if self.ids.is_empty() {
            return;
        }
        let bind = self.bind.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow-scene-cull"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.rows_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.args.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.counts.as_entire_binding(),
                    },
                ],
            })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shadow-scene-cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*bind, &[]);
        pass.dispatch_workgroups(params.rows.div_ceil(GROUP), 1, 1);
    }

    /// Byte offsets of view `view`'s group `g`: its first draw and its count.
    pub fn offsets(&self, view: usize, g: &ShadowGroup, index: usize) -> (u64, u64) {
        let base = (view * self.capacity) as u64;
        let stride = std::mem::size_of::<DrawIndexedArgs>() as u64;
        (
            (base + u64::from(g.start)) * stride,
            (base + index as u64) * 4,
        )
    }
}

/// Whether two rows draw the same (the group fields aside).
fn drawn_same(a: &GpuRow, b: &GpuRow) -> bool {
    a.sphere == b.sphere
        && a.index_count == b.index_count
        && a.instance_count == b.instance_count
        && a.first_index == b.first_index
        && a.base_vertex == b.base_vertex
        && a.first_instance == b.first_instance
}

fn buffers(device: &wgpu::Device, capacity: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let rows = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow-scene-rows"),
        size: (capacity * std::mem::size_of::<GpuRow>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let args = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow-scene-draws"),
        size: (capacity * MAX_VIEWS * std::mem::size_of::<DrawIndexedArgs>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });
    let counts = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow-scene-counts"),
        size: (capacity * MAX_VIEWS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (rows, args, counts)
}
