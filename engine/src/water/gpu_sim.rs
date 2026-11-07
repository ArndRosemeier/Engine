//! The water simulation on the GPU: the same scheme as [`Water::update`]
//! (local inertial shallow water on the layered cells, outflow limiter, sea
//! edges, sources), run as compute passes, many sub-steps per frame without
//! waiting for the CPU.
//!
//! Per frame: apply queued CPU edits, then for each sub-step pick dt from
//! the CFL limit of the previous one (on the GPU), compute heads, link
//! discharge, sea exchange, the outflow limit and the new depths; then foam
//! and wetness. The state is copied to a read-back buffer that reaches the
//! CPU copy a couple of frames later.

use super::backend::{GpuEdit, Snapshot};
use super::sim::{Water, B_SIDE, NONE};
use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const WORKGROUP: u32 = 64;
/// CPU edits applied per frame (more wait for the next frame).
const MAX_EDITS: usize = 256;
const MAX_SOURCES: usize = 64;
/// Read-back buffers in flight.
const READBACKS: usize = 3;
/// Foam and wetness are read back every this many frames.
const MARKS_EVERY: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// cells, links, boundary entries, adjacency offset of the link lists.
    counts: [u32; 4],
    /// dx, stiffness, manning², theta.
    a: [f32; 4],
    /// settle, settle speed, breaking slope, courant.
    b: [f32; 4],
    /// sea level, sea on (0/1), ramp seconds, max dt.
    sea: [f32; 4],
    /// height, period, direction degrees, 0.
    swell: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Frame {
    /// frame dt, rain (m), sea clock at frame start, simulation time at end.
    a: [f32; 4],
    /// edits, sources, 0, 0.
    counts: [u32; 4],
    /// kind (0 depth, 1 flow, 2 foam), index, amount, 0.
    edits: [[f32; 4]; MAX_EDITS],
    /// cell, rate (m/s of depth), 0, 0.
    sources: [[f32; 4]; MAX_SOURCES],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLinkData {
    a: u32,
    b: u32,
    lo: f32,
    hi: f32,
    prev: u32,
    next: u32,
    axis: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBoundary {
    cell: u32,
    x: f32,
    z: f32,
    q: f32,
}

/// Per frame: remaining, dt, max speed (bits), next dt, steps, simulated.
const SIM_WORDS: usize = 8;

struct Buffers {
    cells: usize,
    links: usize,
    boundary: usize,
    statics: wgpu::Buffer,
    dynamic: wgpu::Buffer,
    adjacency: wgpu::Buffer,
    links_buf: wgpu::Buffer,
    q: [wgpu::Buffer; 2],
    boundary_buf: wgpu::Buffer,
    sim: wgpu::Buffer,
    marks: wgpu::Buffer,
    /// [q_old = A, q_new = B] and the swapped one.
    sim_bind: [wgpu::BindGroup; 2],
    marks_bind: wgpu::BindGroup,
}

struct Readback {
    buffer: wgpu::Buffer,
    /// Bytes: depth, then q, then marks (if `with_marks`).
    depth_bytes: u64,
    q_bytes: u64,
    with_marks: bool,
    generation: u64,
    state: ReadbackState,
    ready: Arc<AtomicBool>,
}

#[derive(PartialEq, Eq)]
enum ReadbackState {
    Free,
    /// Copied into in this frame's commands; map on the next frame.
    Copied,
    Mapping,
}

pub(crate) struct GpuWaterSim {
    sim_layout: wgpu::BindGroupLayout,
    marks_layout: wgpu::BindGroupLayout,
    frame_begin: wgpu::ComputePipeline,
    rain: wgpu::ComputePipeline,
    step_begin: wgpu::ComputePipeline,
    heads: wgpu::ComputePipeline,
    links: wgpu::ComputePipeline,
    sea: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    limit: wgpu::ComputePipeline,
    depth: wgpu::ComputePipeline,
    step_end: wgpu::ComputePipeline,
    marks: wgpu::ComputePipeline,
    state_layout: wgpu::BindGroupLayout,
    list_layout: wgpu::BindGroupLayout,
    render_state: wgpu::ComputePipeline,
    draw_lists: wgpu::ComputePipeline,
    /// (targets rev, buffers rev, state bind, list bind).
    draw_binds: Option<(u64, u64, wgpu::BindGroup, wgpu::BindGroup)>,
    params: wgpu::Buffer,
    frame: wgpu::Buffer,
    buffers: Option<Buffers>,
    synced_topology: u64,
    synced_state: u64,
    synced_boundary: u64,
    /// Bumped whenever `buffers` are replaced.
    buffers_gen: u64,
    readbacks: Vec<Readback>,
    frame_no: u64,
    /// Last CFL step size read back, to choose how many sub-steps to encode.
    last_dt: f32,
}

impl GpuWaterSim {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-sim"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{SIM_WGSL_BODY}").into()),
        });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = wgpu::BufferBindingType::Uniform;
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-sim-layout"),
            entries: &[
                entry(0, uniform),
                entry(1, uniform),
                entry(2, read),
                entry(3, rw),
                entry(4, read),
                entry(5, read),
                entry(6, rw),
                entry(7, rw),
                entry(8, rw),
                entry(9, rw),
            ],
        });
        let marks_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-marks-layout"),
            entries: &[
                entry(0, uniform),
                entry(1, uniform),
                entry(2, read),
                entry(3, rw),
                entry(4, read),
                entry(5, read),
                entry(6, read),
                entry(10, rw),
            ],
        });
        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water-sim-pipeline-layout"),
            bind_group_layouts: &[&sim_layout],
            push_constant_ranges: &[],
        });
        let marks_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("water-marks-pipeline-layout"),
                bind_group_layouts: &[&marks_layout],
                push_constant_ranges: &[],
            });
        let marks_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-marks"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{MARKS_WGSL_BODY}").into()),
        });
        let make = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let sim = |e: &str| make(&sim_pipeline_layout, &shader, e);
        let state_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-render-state-layout"),
            entries: &[
                entry(0, uniform),
                entry(1, uniform),
                entry(2, read),
                entry(3, rw),
                entry(4, read),
                entry(5, read),
                entry(6, read),
                entry(11, rw),
            ],
        });
        let list_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-draw-list-layout"),
            entries: &[
                entry(0, uniform),
                entry(1, uniform),
                entry(11, read),
                entry(12, read),
                entry(13, rw),
                entry(14, rw),
                entry(15, rw),
                entry(16, read),
            ],
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-render-state"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{DRAW_WGSL_BODY}").into()),
        });
        let list_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-draw-lists"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{LIST_WGSL_BODY}").into()),
        });
        let with_layout = |layout: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, e: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(e),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            make(&pl, module, e)
        };
        let render_state = with_layout(&state_layout, &draw_shader, "k_render_state");
        let draw_lists = with_layout(&list_layout, &list_shader, "k_draw_lists");
        let uniform_buffer = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            frame_begin: sim("k_frame_begin"),
            rain: sim("k_rain"),
            step_begin: sim("k_step_begin"),
            heads: sim("k_heads"),
            links: sim("k_links"),
            sea: sim("k_sea"),
            scale: sim("k_scale"),
            limit: sim("k_limit"),
            depth: sim("k_depth"),
            step_end: sim("k_step_end"),
            marks: make(&marks_pipeline_layout, &marks_shader, "k_marks"),
            sim_layout,
            marks_layout,
            state_layout,
            list_layout,
            render_state,
            draw_lists,
            draw_binds: None,
            params: uniform_buffer("water-sim-params", std::mem::size_of::<Params>()),
            frame: uniform_buffer("water-sim-frame", std::mem::size_of::<Frame>()),
            buffers: None,
            synced_topology: u64::MAX,
            synced_state: u64::MAX,
            synced_boundary: u64::MAX,
            buffers_gen: 0,
            readbacks: Vec::new(),
            frame_no: 0,
            last_dt: 1.0 / 240.0,
        }
    }

    pub fn marks_buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffers.as_ref().map(|b| &b.marks)
    }

    /// Bumped when buffers are replaced (bind groups using them must be
    /// rebuilt).
    pub fn buffers_rev(&self) -> u64 {
        self.buffers_gen
    }

    /// Upload what changed on the CPU and encode one frame of simulation.
    pub fn encode_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        water: &Water,
    ) {
        self.collect_readbacks(device, water);
        let fresh = self.ensure_buffers(device, water);
        if fresh {
            self.upload_topology(queue, water);
        }
        if fresh || self.synced_boundary != water.boundary_rev {
            self.upload_boundary(queue, water);
        }
        let buffers = self.buffers.as_ref().expect("buffers");
        if fresh || self.synced_state != water.state_rev {
            let dynamic: Vec<[f32; 4]> = water.depth.iter().map(|&d| [d, 0.0, 1.0, 0.0]).collect();
            queue.write_buffer(&buffers.dynamic, 0, bytemuck::cast_slice(&dynamic));
            if !water.q.is_empty() {
                queue.write_buffer(&buffers.q[0], 0, bytemuck::cast_slice(&water.q));
            }
            queue.write_buffer(&buffers.marks, 0, bytemuck::cast_slice(&water.marks));
            self.synced_state = water.state_rev;
            // Everything queued so far is already in the uploaded state.
            let mut link = water.gpu.0.lock().unwrap_or_else(|e| e.into_inner());
            link.edits.clear();
            link.rain = 0.0;
        }

        // What the CPU queued since the last frame.
        let (dt, rain, edits) = {
            let mut link = water.gpu.0.lock().unwrap_or_else(|e| e.into_inner());
            let dt = std::mem::take(&mut link.pending_dt);
            let rain = std::mem::take(&mut link.rain);
            let take = link.edits.len().min(MAX_EDITS);
            let edits: Vec<GpuEdit> = link.edits.drain(..take).collect();
            (dt, rain, edits)
        };
        let s = water.settings;
        let sea = water.sea.as_ref();
        let mut swell = [[0.0; 4]; 4];
        if let Some(sea) = sea {
            for (o, w) in swell.iter_mut().zip(&sea.swell) {
                *o = [w.height, w.period, w.direction_deg, 0.0];
            }
        }
        let params = Params {
            counts: [
                buffers.cells as u32,
                buffers.links as u32,
                water.boundary.len().min(buffers.boundary) as u32,
                water.cell_link_start.len() as u32,
            ],
            a: [
                water.region.cell_size,
                s.stiffness,
                s.manning * s.manning,
                s.theta.clamp(0.0, 1.0),
            ],
            b: [
                s.settle.max(0.0),
                s.settle_speed.max(1e-3),
                s.breaking_slope.max(1e-3),
                s.courant,
            ],
            sea: [
                sea.map_or(0.0, |s| s.level),
                if sea.is_some() { 1.0 } else { 0.0 },
                sea.map_or(1.0, |s| s.ramp_seconds.max(1e-3)),
                1.0 / 30.0,
            ],
            swell,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut frame = Frame::zeroed();
        // The CPU clocks already include this frame's dt.
        frame.a = [
            dt,
            rain,
            (water.sea_clock - dt as f64) as f32,
            water.time as f32,
        ];
        for (o, e) in frame.edits.iter_mut().zip(&edits) {
            *o = match *e {
                GpuEdit::Depth { cell, add } => [0.0, f32::from_bits(cell), add, 0.0],
                GpuEdit::Flow { link, add } => [1.0, f32::from_bits(link), add, 0.0],
                GpuEdit::Foam { cell, amount } => [2.0, f32::from_bits(cell), amount, 0.0],
            };
        }
        let area = water.region.cell_size * water.region.cell_size;
        let mut sources = 0;
        for (_, src) in water.sources.iter().take(MAX_SOURCES) {
            let cell = water
                .cell_at(src.position)
                .or_else(|| water.cell_above(src.position));
            if let Some(cell) = cell {
                frame.sources[sources] = [f32::from_bits(cell), src.rate / area, 0.0, 0.0];
                sources += 1;
            }
        }
        frame.counts = [edits.len() as u32, sources as u32, 0, 0];
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));

        // Enough sub-steps for this frame at the last known CFL step.
        let max_steps = s.max_substeps.max(1);
        let steps = if dt > 0.0 {
            ((dt / self.last_dt.max(1e-4)) * 1.25).ceil() as u32 + 1
        } else {
            0
        }
        .clamp(if dt > 0.0 { 1 } else { 0 }, max_steps);

        let cells_groups = groups(buffers.cells);
        let link_groups = groups(buffers.links);
        let boundary_groups = groups(params.counts[2] as usize);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("water-sim"),
                timestamp_writes: None,
            });
            let dispatch =
                |pass: &mut wgpu::ComputePass<'_>, p: &wgpu::ComputePipeline, n: (u32, u32)| {
                    if n.0 > 0 {
                        pass.set_pipeline(p);
                        pass.dispatch_workgroups(n.0, n.1, 1);
                    }
                };
            pass.set_bind_group(0, &buffers.sim_bind[0], &[]);
            dispatch(&mut pass, &self.frame_begin, (1, 1));
            if rain != 0.0 {
                dispatch(&mut pass, &self.rain, cells_groups);
            }
            for k in 0..steps as usize {
                pass.set_bind_group(0, &buffers.sim_bind[k % 2], &[]);
                dispatch(&mut pass, &self.step_begin, (1, 1));
                dispatch(&mut pass, &self.heads, cells_groups);
                dispatch(&mut pass, &self.links, link_groups);
                dispatch(&mut pass, &self.sea, boundary_groups);
                dispatch(&mut pass, &self.scale, cells_groups);
                dispatch(&mut pass, &self.limit, link_groups);
                dispatch(&mut pass, &self.depth, cells_groups);
                dispatch(&mut pass, &self.step_end, (1, 1));
            }
        }
        // Keep the current discharge in q[0].
        if steps % 2 == 1 && buffers.links > 0 {
            encoder.copy_buffer_to_buffer(
                &buffers.q[1],
                0,
                &buffers.q[0],
                0,
                (buffers.links * 4) as u64,
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("water-marks"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &buffers.marks_bind, &[]);
            pass.set_pipeline(&self.marks);
            pass.dispatch_workgroups(cells_groups.0, cells_groups.1, 1);
        }
        self.queue_readback(device, encoder, water);
        self.frame_no += 1;
    }

    /// After [`Self::encode_frame`]: the per-cell drawing state and the
    /// lists of cells to draw, straight from the GPU state.
    pub fn encode_draw(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        targets: DrawTargets<'_>,
    ) {
        let Some(b) = self.buffers.as_ref() else {
            return;
        };
        let stale = self
            .draw_binds
            .as_ref()
            .is_none_or(|d| d.0 != targets.rev || d.1 != self.buffers_gen);
        if stale {
            let state = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("water-render-state-bind"),
                layout: &self.state_layout,
                entries: &[
                    bg(0, &self.params),
                    bg(1, &self.frame),
                    bg(2, &b.statics),
                    bg(3, &b.dynamic),
                    bg(4, &b.adjacency),
                    bg(5, &b.links_buf),
                    bg(6, &b.q[0]),
                    bg(11, targets.state),
                ],
            });
            let list = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("water-draw-list-bind"),
                layout: &self.list_layout,
                entries: &[
                    bg(0, &self.params),
                    bg(1, &self.frame),
                    bg(11, targets.state),
                    bg(12, targets.corners),
                    bg(13, targets.water_list),
                    bg(14, targets.ground_list),
                    bg(15, targets.indirect),
                    bg(16, &b.marks),
                ],
            });
            self.draw_binds = Some((targets.rev, self.buffers_gen, state, list));
        }
        let (_, _, state, list) = self.draw_binds.as_ref().expect("draw binds");
        let g = groups(b.cells);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("water-draw-lists"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, state, &[]);
        pass.set_pipeline(&self.render_state);
        pass.dispatch_workgroups(g.0, g.1, 1);
        pass.set_bind_group(0, list, &[]);
        pass.set_pipeline(&self.draw_lists);
        pass.dispatch_workgroups(g.0, g.1, 1);
    }

    fn ensure_buffers(&mut self, device: &wgpu::Device, water: &Water) -> bool {
        let same = self.buffers.as_ref().is_some_and(|b| {
            b.cells == water.floor.len().max(1)
                && b.links == water.links.len().max(1)
                && b.boundary >= water.boundary.len().max(1)
        }) && self.synced_topology == water.topology_rev;
        if same {
            return false;
        }
        let cells = water.floor.len().max(1);
        let links = water.links.len().max(1);
        let boundary = water.boundary.len().max(1).next_power_of_two();
        let storage = |label: &str, size: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | extra,
                mapped_at_creation: false,
            })
        };
        let none = wgpu::BufferUsages::empty();
        let adjacency_len = water.cell_link_start.len() + water.cell_links.len();
        let statics = storage("water-sim-static", cells * 16, none);
        let dynamic = storage("water-sim-dynamic", cells * 16, none);
        let adjacency = storage("water-sim-adjacency", adjacency_len * 4, none);
        let links_buf = storage("water-sim-links", links * 32, none);
        let q = [
            storage("water-sim-q0", links * 4, none),
            storage("water-sim-q1", links * 4, none),
        ];
        let boundary_buf = storage("water-sim-boundary", boundary * 16, none);
        let sim = storage("water-sim-state", SIM_WORDS * 4, none);
        let marks = storage("water-sim-marks", cells * 16, none);
        let bind = |q_old: &wgpu::Buffer, q_new: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("water-sim-bind"),
                layout: &self.sim_layout,
                entries: &[
                    bg(0, &self.params),
                    bg(1, &self.frame),
                    bg(2, &statics),
                    bg(3, &dynamic),
                    bg(4, &adjacency),
                    bg(5, &links_buf),
                    bg(6, q_old),
                    bg(7, q_new),
                    bg(8, &boundary_buf),
                    bg(9, &sim),
                ],
            })
        };
        let sim_bind = [bind(&q[0], &q[1]), bind(&q[1], &q[0])];
        let marks_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-marks-bind"),
            layout: &self.marks_layout,
            entries: &[
                bg(0, &self.params),
                bg(1, &self.frame),
                bg(2, &statics),
                bg(3, &dynamic),
                bg(4, &adjacency),
                bg(5, &links_buf),
                bg(6, &q[0]),
                bg(10, &marks),
            ],
        });
        self.buffers = Some(Buffers {
            cells,
            links,
            boundary,
            statics,
            dynamic,
            adjacency,
            links_buf,
            q,
            boundary_buf,
            sim,
            marks,
            sim_bind,
            marks_bind,
        });
        self.synced_topology = water.topology_rev;
        self.buffers_gen += 1;
        // Read-backs of the old layout are useless now.
        self.readbacks.clear();
        true
    }

    /// Sea edge entries, grouped by cell, and the per-cell statics that
    /// point at them.
    fn upload_boundary(&mut self, queue: &wgpu::Queue, water: &Water) {
        let b = self.buffers.as_ref().expect("buffers");
        let mut order: Vec<usize> = (0..water.boundary.len().min(b.boundary)).collect();
        order.sort_by_key(|&k| water.boundary[k].cell);
        let mut sea_first = vec![0u32; water.floor.len()];
        let mut sea_count = vec![0u32; water.floor.len()];
        let mut list = Vec::with_capacity(order.len());
        for (n, &k) in order.iter().enumerate() {
            let e = water.boundary[k];
            let c = e.cell as usize;
            if sea_count[c] == 0 {
                sea_first[c] = n as u32;
            }
            sea_count[c] += 1;
            list.push(GpuBoundary {
                cell: e.cell,
                x: e.at.x,
                z: e.at.y,
                q: 0.0,
            });
        }
        if !list.is_empty() {
            queue.write_buffer(&b.boundary_buf, 0, bytemuck::cast_slice(&list));
        }
        let ncols = water.region.column_count();
        let statics: Vec<[f32; 4]> = (0..water.floor.len())
            .map(|i| {
                let col = water.cell_col[i] as usize;
                let top = col < ncols && i + 1 == water.col_start[col + 1] as usize;
                // A cell touches the sea through at most two faces (a corner).
                let packed = (sea_first[i] << 2) | sea_count[i].min(3);
                [
                    water.floor[i],
                    water.ceil[i].min(1e30),
                    f32::from_bits(top as u32),
                    f32::from_bits(packed),
                ]
            })
            .collect();
        if !statics.is_empty() {
            queue.write_buffer(&b.statics, 0, bytemuck::cast_slice(&statics));
        }
        self.synced_boundary = water.boundary_rev;
    }

    fn upload_topology(&mut self, queue: &wgpu::Queue, water: &Water) {
        let b = self.buffers.as_ref().expect("buffers");
        let mut adjacency =
            Vec::with_capacity(water.cell_link_start.len() + water.cell_links.len());
        adjacency.extend_from_slice(&water.cell_link_start);
        adjacency.extend_from_slice(&water.cell_links);
        queue.write_buffer(&b.adjacency, 0, bytemuck::cast_slice(&adjacency));
        let links: Vec<GpuLinkData> = water
            .links
            .iter()
            .enumerate()
            .map(|(i, l)| GpuLinkData {
                a: l.a,
                b: l.b,
                lo: l.lo,
                hi: l.hi.min(1e30),
                prev: water.link_prev[i],
                next: water.link_next[i],
                axis: l.axis as u32,
                _pad: 0,
            })
            .collect();
        if !links.is_empty() {
            queue.write_buffer(&b.links_buf, 0, bytemuck::cast_slice(&links));
        }
        let sim = [0.0f32, 0.0, 0.0, self.last_dt, 0.0, 0.0, 0.0, 0.0];
        queue.write_buffer(&b.sim, 0, bytemuck::cast_slice(&sim));
    }

    fn queue_readback(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        water: &Water,
    ) {
        let b = self.buffers.as_ref().expect("buffers");
        let with_marks = self.frame_no.is_multiple_of(MARKS_EVERY);
        let Some(slot) = self
            .readbacks
            .iter()
            .position(|r| r.state == ReadbackState::Free)
            .or_else(|| {
                (self.readbacks.len() < READBACKS).then(|| {
                    let depth_bytes = (b.cells * 16) as u64;
                    let q_bytes = (b.links * 4) as u64;
                    let marks_bytes = (b.cells * 16) as u64;
                    self.readbacks.push(Readback {
                        buffer: device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("water-readback"),
                            size: depth_bytes + q_bytes + marks_bytes + 32,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                        depth_bytes,
                        q_bytes,
                        with_marks: false,
                        generation: 0,
                        state: ReadbackState::Free,
                        ready: Arc::new(AtomicBool::new(false)),
                    });
                    self.readbacks.len() - 1
                })
            })
        else {
            return; // all in flight: skip this frame's read-back
        };
        let r = &mut self.readbacks[slot];
        encoder.copy_buffer_to_buffer(&b.dynamic, 0, &r.buffer, 0, r.depth_bytes);
        encoder.copy_buffer_to_buffer(&b.q[0], 0, &r.buffer, r.depth_bytes, r.q_bytes);
        if with_marks {
            encoder.copy_buffer_to_buffer(
                &b.marks,
                0,
                &r.buffer,
                r.depth_bytes + r.q_bytes,
                (b.cells * 16) as u64,
            );
        }
        let sim_at = r.depth_bytes + r.q_bytes + (b.cells * 16) as u64;
        encoder.copy_buffer_to_buffer(&b.sim, 0, &r.buffer, sim_at, (SIM_WORDS * 4) as u64);
        r.with_marks = with_marks;
        r.generation = water.state_rev;
        r.state = ReadbackState::Copied;
    }

    /// Start mapping last frame's copies; hand finished ones to the CPU.
    fn collect_readbacks(&mut self, device: &wgpu::Device, water: &Water) {
        let _ = device.poll(wgpu::PollType::Poll);
        let cells = self.buffers.as_ref().map_or(0, |b| b.cells);
        let mut newest: Option<Snapshot> = None;
        for r in &mut self.readbacks {
            match r.state {
                ReadbackState::Copied => {
                    let ready = r.ready.clone();
                    ready.store(false, Ordering::Release);
                    r.buffer
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |res| {
                            if res.is_ok() {
                                ready.store(true, Ordering::Release);
                            }
                        });
                    r.state = ReadbackState::Mapping;
                }
                ReadbackState::Mapping if r.ready.load(Ordering::Acquire) => {
                    {
                        let data = r.buffer.slice(..).get_mapped_range();
                        let dynamic: &[[f32; 4]] =
                            bytemuck::cast_slice(&data[..r.depth_bytes as usize]);
                        let q: &[f32] = bytemuck::cast_slice(
                            &data[r.depth_bytes as usize..(r.depth_bytes + r.q_bytes) as usize],
                        );
                        let marks_at = (r.depth_bytes + r.q_bytes) as usize;
                        let sim_at = marks_at + cells * 16;
                        let sim: &[f32] =
                            bytemuck::cast_slice(&data[sim_at..sim_at + SIM_WORDS * 4]);
                        self.last_dt = sim[3].clamp(1e-4, 1.0 / 30.0);
                        let snapshot = Snapshot {
                            generation: r.generation,
                            depth: dynamic.iter().map(|d| d[0]).collect(),
                            q: q[..water.links.len().min(q.len())].to_vec(),
                            marks: r.with_marks.then(|| {
                                bytemuck::cast_slice::<u8, [f32; 4]>(
                                    &data[marks_at..marks_at + cells * 16],
                                )
                                .to_vec()
                            }),
                        };
                        // Keep marks from an older snapshot if the newest has none.
                        let marks = snapshot
                            .marks
                            .is_none()
                            .then(|| newest.as_mut().and_then(|n| n.marks.take()))
                            .flatten();
                        let mut snapshot = snapshot;
                        if snapshot.marks.is_none() {
                            snapshot.marks = marks;
                        }
                        newest = Some(snapshot);
                    }
                    r.buffer.unmap();
                    r.state = ReadbackState::Free;
                }
                _ => {}
            }
        }
        if let Some(mut s) = newest {
            s.depth.truncate(water.floor.len());
            let mut link = water.gpu.0.lock().unwrap_or_else(|e| e.into_inner());
            if s.marks.is_none() {
                s.marks = link.inbox.take().and_then(|old| old.marks);
            }
            link.inbox = Some(s);
        }
    }
}

/// Where [`GpuWaterSim::encode_draw`] writes. The indirect buffer holds two
/// draws (water, then wet ground) and must be reset to `[6, 0, 0, 0, 6, 0,
/// 0, 0]` before.
pub(crate) struct DrawTargets<'a> {
    /// Per cell: surface, depth, current x, current z.
    pub state: &'a wgpu::Buffer,
    pub corners: &'a wgpu::Buffer,
    pub water_list: &'a wgpu::Buffer,
    pub ground_list: &'a wgpu::Buffer,
    pub indirect: &'a wgpu::Buffer,
    /// Changes whenever any of these buffers is replaced.
    pub rev: u64,
}

fn groups(n: usize) -> (u32, u32) {
    let g = (n as u32).div_ceil(WORKGROUP);
    if g <= 65535 {
        (g, 1)
    } else {
        (65535, g.div_ceil(65535))
    }
}

fn bg(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = assert!(B_SIDE == 1 << 31);
const _: () = assert!(NONE == u32::MAX);

/// Shared declarations of the simulation and marks shaders.
const COMMON_WGSL: &str = r#"
struct Params {
    counts: vec4<u32>,
    a: vec4<f32>,
    b: vec4<f32>,
    sea: vec4<f32>,
    swell: array<vec4<f32>, 4>,
};

struct Frame {
    a: vec4<f32>,
    counts: vec4<u32>,
    edits: array<vec4<f32>, 256>,
    sources: array<vec4<f32>, 64>,
};

struct Link {
    a: u32,
    b: u32,
    lo: f32,
    hi: f32,
    prev: u32,
    next: u32,
    axis: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> F: Frame;
// floor, ceil, top (bits), sea entries (bits: first << 2 | count).
@group(0) @binding(2) var<storage, read> statics: array<vec4<f32>>;
// depth, head, outflow scale, 0.
@group(0) @binding(3) var<storage, read_write> dyn_state: array<vec4<f32>>;
// Cell link list starts (cells + 1), then the lists (bit 31 = b side).
@group(0) @binding(4) var<storage, read> adjacency: array<u32>;
@group(0) @binding(5) var<storage, read> links: array<Link>;

const G: f32 = 9.81;
const DRY: f32 = 1e-4;
const NONE: u32 = 0xffffffffu;
const B_SIDE: u32 = 0x80000000u;

fn idx(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65535u * 64u;
}

fn head_of(floor: f32, ceil: f32, depth: f32) -> f32 {
    let cap = ceil - floor;
    if depth <= cap {
        return floor + depth;
    }
    return ceil + (depth - cap) * P.a.y;
}
"#;

const SIM_WGSL_BODY: &str = r#"
@group(0) @binding(6) var<storage, read_write> q_old: array<f32>;
@group(0) @binding(7) var<storage, read_write> q_new: array<f32>;
struct Boundary {
    cell: u32,
    x: f32,
    z: f32,
    q: f32,
};
@group(0) @binding(8) var<storage, read_write> boundary: array<Boundary>;
struct Sim {
    remaining: f32,
    dt: f32,
    max_speed: atomic<u32>,
    next_dt: f32,
    steps: u32,
    simulated: f32,
    pad0: f32,
    pad1: f32,
};
@group(0) @binding(9) var<storage, read_write> sim: Sim;

@compute @workgroup_size(1)
fn k_frame_begin() {
    sim.remaining = F.a.x;
    sim.steps = 0u;
    sim.simulated = 0.0;
    if sim.next_dt <= 0.0 {
        sim.next_dt = 1.0 / 240.0;
    }
    // Edits from the CPU, in order.
    for (var i = 0u; i < F.counts.x; i++) {
        let e = F.edits[i];
        let at = bitcast<u32>(e.y);
        if e.x == 0.0 {
            dyn_state[at].x = max(dyn_state[at].x + e.z, 0.0);
        } else if e.x == 1.0 {
            q_old[at] += e.z;
        }
    }
}

@compute @workgroup_size(64)
fn k_rain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x {
        return;
    }
    if bitcast<u32>(statics[i].z) != 0u {
        dyn_state[i].x = max(dyn_state[i].x + F.a.y, 0.0);
    }
}

@compute @workgroup_size(1)
fn k_step_begin() {
    var dt = min(sim.next_dt, sim.remaining);
    if sim.remaining <= 1e-6 {
        dt = 0.0;
    }
    sim.dt = dt;
    sim.remaining -= dt;
    sim.simulated += dt;
    atomicStore(&sim.max_speed, 0u);
    for (var i = 0u; i < F.counts.y; i++) {
        let s = F.sources[i];
        let c = bitcast<u32>(s.x);
        dyn_state[c].x = max(dyn_state[c].x + s.y * dt, 0.0);
    }
}

@compute @workgroup_size(64)
fn k_heads(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x || sim.dt <= 0.0 {
        return;
    }
    let s = statics[i];
    dyn_state[i].y = head_of(s.x, s.y, dyn_state[i].x);
}

@compute @workgroup_size(64)
fn k_links(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.y {
        return;
    }
    let dt = sim.dt;
    let q0 = q_old[i];
    if dt <= 0.0 {
        q_new[i] = q0;
        return;
    }
    let l = links[i];
    let dx = P.a.x;
    let ha = dyn_state[l.a].y;
    let hb = dyn_state[l.b].y;
    let eta = max(ha, hb);
    if eta <= l.lo + DRY {
        q_new[i] = 0.0;
        return;
    }
    let h = max(min(eta, l.hi) - l.lo, DRY);
    let ea = max(ha, l.lo);
    let eb = max(hb, l.lo);
    var qp = q0;
    if l.prev != NONE {
        qp = q_old[l.prev];
    }
    var qn = q0;
    if l.next != NONE {
        qn = q_old[l.next];
    }
    let theta = P.a.w;
    let qt = theta * q0 + (1.0 - theta) * 0.5 * (qp + qn);
    let speed = abs(qt) / h;
    let settle = P.b.x * max(1.0 - speed / P.b.y, 0.0);
    let slope = abs(eb - ea) / dx;
    let both_wet = ha > l.lo + DRY && hb > l.lo + DRY;
    var breaking = 0.0;
    if both_wet {
        breaking = max(slope - P.b.z, 0.0) * 4.0;
    }
    let drag = G * P.a.z * abs(qt) / pow(h, 7.0 / 3.0) + settle + breaking;
    var fall = eb - ea;
    if abs(fall) <= max(abs(eta), 1.0) * 2e-6 {
        fall = 0.0;
    }
    var q = (qt - G * h * dt * fall / dx) / (1.0 + dt * drag);
    let vmax = sqrt(2.0 * G * abs(ea - eb)) + sqrt(G * h);
    q = clamp(q, -h * vmax, h * vmax);
    var donor_excess = hb - l.lo;
    if q > 0.0 {
        donor_excess = ha - l.lo;
    }
    if donor_excess <= DRY {
        q = 0.0;
    } else {
        let cap = donor_excess * dx / dt;
        q = clamp(q, -cap, cap);
    }
    q_new[i] = q;
    let sa = statics[l.a];
    let sb = statics[l.b];
    let pressurised = ha > sa.y || hb > sb.y;
    var wave_depth = min(h, max(sa.y - sa.x, sb.y - sb.x));
    if pressurised {
        wave_depth = h * P.a.y;
    }
    let signal = abs(q) / h + sqrt(G * wave_depth);
    atomicMax(&sim.max_speed, bitcast<u32>(signal));
}

fn incoming(p: vec2<f32>, t: f32, depth: f32) -> f32 {
    let r = clamp(t / P.sea.z, 0.0, 1.0);
    let ramp = r * r * (3.0 - 2.0 * r);
    let c = sqrt(G * max(depth, 0.5));
    var eta = 0.0;
    for (var i = 0; i < 4; i++) {
        let s = P.swell[i];
        if s.x == 0.0 {
            continue;
        }
        let omega = 6.2831853 / max(s.y, 0.5);
        let a = radians(s.z);
        let k = vec2<f32>(sin(a), cos(a)) * (omega / c);
        eta += 0.5 * s.x * cos(omega * t - dot(k, p) + f32(i) * 1.3);
    }
    return eta * ramp;
}

@compute @workgroup_size(64)
fn k_sea(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = idx(gid);
    if k >= P.counts.z {
        return;
    }
    let dt = sim.dt;
    if dt <= 0.0 || P.sea.y == 0.0 {
        boundary[k].q = 0.0;
        return;
    }
    let b = boundary[k];
    let s = statics[b.cell];
    let floor = s.x;
    let level = P.sea.x;
    let still = max(level - floor, 0.0);
    let head = min(dyn_state[b.cell].y, s.y);
    // Sea time at the start of this sub-step.
    let t = F.a.z + sim.simulated - dt;
    let eta_in = incoming(vec2<f32>(b.x, b.z), t, still);
    if still <= 0.05 && head <= floor + 1e-4 && level + eta_in <= floor {
        boundary[k].q = 0.0;
        return;
    }
    let c = sqrt(G * max(still, 0.05));
    var q = c * (head - level - 2.0 * eta_in);
    if q > 0.0 {
        q = min(q, max(head - floor, 0.0) * P.a.x / dt);
    }
    boundary[k].q = q;
    atomicMax(&sim.max_speed, bitcast<u32>(c));
}

fn sea_entries(i: u32) -> vec2<u32> {
    let packed = bitcast<u32>(statics[i].w);
    return vec2<u32>(packed >> 2u, packed & 3u);
}

@compute @workgroup_size(64)
fn k_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x || sim.dt <= 0.0 {
        return;
    }
    var outflow = 0.0;
    let e0 = adjacency[i];
    let e1 = adjacency[i + 1u];
    for (var e = e0; e < e1; e++) {
        let entry = adjacency[P.counts.w + e];
        let qv = q_new[entry & ~B_SIDE];
        var give = qv;
        if (entry & B_SIDE) != 0u {
            give = -qv;
        }
        outflow += max(give, 0.0);
    }
    let se = sea_entries(i);
    for (var k = 0u; k < se.y; k++) {
        outflow += max(boundary[se.x + k].q, 0.0);
    }
    outflow *= sim.dt / P.a.x;
    let d = dyn_state[i].x;
    var sc = 1.0;
    if outflow > d && outflow > 0.0 {
        sc = d / outflow;
    }
    dyn_state[i].z = sc;
}

@compute @workgroup_size(64)
fn k_limit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.y || sim.dt <= 0.0 {
        return;
    }
    let q = q_new[i];
    let l = links[i];
    var donor = l.b;
    if q > 0.0 {
        donor = l.a;
    }
    q_new[i] = q * dyn_state[donor].z;
}

@compute @workgroup_size(64)
fn k_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x || sim.dt <= 0.0 {
        return;
    }
    var net = 0.0;
    let e0 = adjacency[i];
    let e1 = adjacency[i + 1u];
    for (var e = e0; e < e1; e++) {
        let entry = adjacency[P.counts.w + e];
        let qv = q_new[entry & ~B_SIDE];
        if (entry & B_SIDE) != 0u {
            net += qv;
        } else {
            net -= qv;
        }
    }
    let se = sea_entries(i);
    for (var k = 0u; k < se.y; k++) {
        var q = boundary[se.x + k].q;
        if q > 0.0 {
            q *= dyn_state[i].z;
        }
        net -= q;
    }
    dyn_state[i].x = max(dyn_state[i].x + net * sim.dt / P.a.x, 0.0);
}

@compute @workgroup_size(1)
fn k_step_end() {
    if sim.dt <= 0.0 {
        return;
    }
    let speed = max(bitcast<f32>(atomicLoad(&sim.max_speed)), 1e-3);
    sim.next_dt = clamp(P.b.w * P.a.x / speed, 1e-4, P.sea.w);
    sim.steps += 1u;
}
"#;

const DRAW_WGSL_BODY: &str = r#"
@group(0) @binding(6) var<storage, read> q: array<f32>;
@group(0) @binding(11) var<storage, read_write> state: array<vec4<f32>>;

// Same as Water::render_state.
@compute @workgroup_size(64)
fn k_render_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x {
        return;
    }
    let s = statics[i];
    let d = dyn_state[i].x;
    if d <= DRY {
        state[i] = vec4<f32>(s.x, 0.0, 0.0, 0.0);
        return;
    }
    let head = head_of(s.x, s.y, d);
    let h = max(min(head, s.y) - s.x, 0.05);
    var sum = vec2<f32>(0.0);
    var count = vec2<f32>(0.0);
    let e0 = adjacency[i];
    let e1 = adjacency[i + 1u];
    for (var e = e0; e < e1; e++) {
        let id = adjacency[P.counts.w + e] & ~B_SIDE;
        if links[id].axis == 0u {
            sum.x += q[id];
            count.x += 1.0;
        } else {
            sum.y += q[id];
            count.y += 1.0;
        }
    }
    let cur = sum / max(count, vec2<f32>(1.0)) / h;
    if head >= s.y {
        state[i] = vec4<f32>(s.y, -1.0, cur);
    } else {
        state[i] = vec4<f32>(head, head - s.x, cur);
    }
}
"#;

const LIST_WGSL_BODY: &str = r#"
@group(0) @binding(11) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> corners: array<array<u32, 16>>;
@group(0) @binding(13) var<storage, read_write> water_list: array<u32>;
@group(0) @binding(14) var<storage, read_write> ground_list: array<u32>;
@group(0) @binding(15) var<storage, read_write> indirect: array<atomic<u32>, 8>;
@group(0) @binding(16) var<storage, read> marks: array<vec4<f32>>;

const DRAW_DEPTH: f32 = 0.004;
const DRYING_SECONDS: f32 = 30.0;
const SAND_BIT: u32 = 0x80000000u;

fn recent(i: u32) -> bool {
    return F.a.w - marks[i].z < DRYING_SECONDS * 4.0;
}

fn open_wet(n: u32) -> bool {
    return n != NONE && state[n].y > DRAW_DEPTH;
}

// Same as Water::render_draw_list.
@compute @workgroup_size(64)
fn k_draw_lists(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x {
        return;
    }
    let d = state[i].y;
    if d < 0.0 {
        return;
    }
    var water = d > DRAW_DEPTH;
    if !water {
        let c = corners[i];
        for (var k = 0u; k < 16u; k++) {
            if open_wet(c[k]) {
                water = true;
                break;
            }
        }
    }
    if water {
        water_list[atomicAdd(&indirect[1], 1u)] = i;
    }
    if d > DRAW_DEPTH {
        return;
    }
    // Wet ground and its dry neighbours (the darkening fades across them).
    var ground = recent(i);
    if !ground {
        let c = corners[i];
        for (var k = 0u; k < 16u; k++) {
            let n = c[k];
            if n != NONE && state[n].y >= 0.0 && state[n].y <= DRAW_DEPTH && recent(n) {
                ground = true;
                break;
            }
        }
    }
    if ground {
        ground_list[atomicAdd(&indirect[5], 1u)] = i | SAND_BIT;
    }
}
"#;

const MARKS_WGSL_BODY: &str = r#"
@group(0) @binding(6) var<storage, read> q: array<f32>;
@group(0) @binding(10) var<storage, read_write> marks: array<vec4<f32>>;

const FOAM_SECONDS: f32 = 5.0;
const DRAW_DEPTH: f32 = 0.004;

// Same as Water::update_marks.
@compute @workgroup_size(64)
fn k_marks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = idx(gid);
    if i >= P.counts.x {
        return;
    }
    // A CPU splash, queued as an edit.
    var m = marks[i];
    let t = F.a.w;
    for (var e = 0u; e < F.counts.x; e++) {
        let ed = F.edits[e];
        if ed.x == 2.0 && bitcast<u32>(ed.y) == i {
            let now = m.x * exp(-(t - m.y) / FOAM_SECONDS);
            m = vec4<f32>(max(now, ed.z), t, m.z, m.w);
        }
    }
    let d = dyn_state[i].x;
    let s = statics[i];
    // Heads may be stale after a frame without sub-steps.
    let head = head_of(s.x, s.y, d);
    if d <= DRY {
        marks[i] = m;
        return;
    }
    if d > DRAW_DEPTH {
        m.z = t;
    }
    if head >= s.y {
        marks[i] = m;
        return;
    }
    var make = 0.0;
    let dx = P.a.x;
    let e0 = adjacency[i];
    let e1 = adjacency[i + 1u];
    for (var e = e0; e < e1; e++) {
        let id = adjacency[P.counts.w + e] & ~B_SIDE;
        let l = links[id];
        var other = l.a;
        if l.a == i {
            other = l.b;
        }
        let so = statics[other];
        let head_o = head_of(so.x, so.y, dyn_state[other].x);
        let h = max(min(max(head, head_o), l.hi) - l.lo, 0.02);
        if dyn_state[other].x > DRY {
            let slope = abs(head - head_o) / dx;
            make = max(make, smoothstep(P.b.z, P.b.z * 3.0, slope));
        }
        let froude = abs(q[id]) / h / sqrt(G * h);
        make = max(make, smoothstep(0.9, 1.8, froude) * smoothstep(0.03, 0.2, h) * 0.8);
    }
    let now = m.x * exp(-(t - m.y) / FOAM_SECONDS);
    m.x = max(now, make);
    m.y = t;
    marks[i] = m;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::{HeightMapTerrain, Sea, SeaEdges, Swell, WaterBackend, WaterRegion};
    use glam::{Vec2, Vec3};

    fn gpu_tests_enabled() -> bool {
        std::env::var("ENGINE_GPU_TESTS").is_ok_and(|v| v == "1")
    }

    fn device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("adapter");
        pollster::block_on(adapter.request_device(&Default::default())).expect("device")
    }

    /// Run `water` (on the GPU backend) for `frames` frames of `dt`, then
    /// flush the read-backs into the CPU copy.
    fn run_gpu(
        sim: &mut GpuWaterSim,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        water: &mut Water,
        frames: usize,
        dt: f32,
    ) {
        let mut frame = |water: &mut Water, dt: f32| {
            water.update(dt);
            let mut encoder = device.create_command_encoder(&Default::default());
            sim.encode_frame(device, queue, &mut encoder, water);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::Wait).expect("poll");
        };
        for _ in 0..frames {
            frame(water, dt);
        }
        // Read-backs trail by a couple of frames (one more for the marks).
        for _ in 0..MARKS_EVERY as usize + 3 {
            frame(water, 0.0);
        }
        water.update(0.0);
    }

    fn dam(backend: WaterBackend) -> Water {
        let flat = HeightMapTerrain(|_, _| 0.0);
        let region = WaterRegion::new(Vec2::ZERO, Vec2::new(40.0, 8.0), 1.0, -1.0, 20.0);
        let mut water = Water::new(region, &flat).with_backend(backend);
        for z in 0..8 {
            for x in 0..20 {
                water.pour(Vec3::new(x as f32 + 0.5, 0.5, z as f32 + 0.5), 2.0);
            }
        }
        water.activity.never_sleep = true;
        water
    }

    #[test]
    fn the_gpu_dam_break_matches_the_cpu_one() {
        if !gpu_tests_enabled() {
            return;
        }
        let (device, queue) = device();
        let mut sim = GpuWaterSim::new(&device);
        let mut cpu = dam(WaterBackend::Cpu);
        let mut gpu = dam(WaterBackend::Gpu);
        let v0 = cpu.total_volume();
        let frames = 90;
        for _ in 0..frames {
            cpu.update(1.0 / 60.0);
        }
        run_gpu(&mut sim, &device, &queue, &mut gpu, frames, 1.0 / 60.0);
        let vg = gpu.total_volume();
        assert!((vg - v0).abs() / v0 < 1e-4, "volume {vg} vs {v0}");
        // The front has moved and both agree on where the water is.
        let worst = cpu
            .depth
            .iter()
            .zip(&gpu.depth)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let at = cpu.cell_at(Vec3::new(26.5, 0.1, 4.5)).unwrap() as usize;
        let moved = gpu.depth[at];
        eprintln!("worst depth difference {worst} m, depth at x=26: {moved}");
        assert!(moved > 0.05, "the front should reach x = 26, depth {moved}");
        assert!(worst < 0.1, "GPU and CPU differ by {worst} m");
    }

    #[test]
    fn gpu_swell_runs_up_the_beach_and_pours_and_rain_arrive() {
        if !gpu_tests_enabled() {
            return;
        }
        let (device, queue) = device();
        let mut sim = GpuWaterSim::new(&device);
        let ground = |x: f32| (x - 100.0) * 0.05;
        let beach = HeightMapTerrain(move |x: f32, _| ground(x));
        let make = |backend| {
            let region = WaterRegion::new(Vec2::ZERO, Vec2::new(160.0, 2.0), 1.0, -8.0, 20.0);
            let mut water = Water::new(region, &beach).with_backend(backend);
            water.flood_sea(0.0);
            water.set_sea(Some(
                Sea::calm(0.0)
                    .with_edges(SeaEdges {
                        min_x: true,
                        ..Default::default()
                    })
                    .with_swell(Swell {
                        height: 0.8,
                        period: 8.0,
                        direction_deg: 90.0,
                    }),
            ));
            water.activity.never_sleep = true;
            water
        };
        let mut cpu = make(WaterBackend::Cpu);
        let mut gpu = make(WaterBackend::Gpu);
        let frames = 60 * 12;
        for _ in 0..frames {
            cpu.update(1.0 / 60.0);
        }
        run_gpu(&mut sim, &device, &queue, &mut gpu, frames, 1.0 / 60.0);
        let worst = cpu
            .depth
            .iter()
            .zip(&gpu.depth)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (vc, vg) = (cpu.total_volume(), gpu.total_volume());
        eprintln!("swell: worst {worst} m, volume cpu {vc} gpu {vg}");
        assert!(worst < 0.15, "GPU and CPU differ by {worst} m");
        assert!((vc - vg).abs() / vc < 0.01);

        // Edits made on the CPU reach the GPU state.
        let before = gpu.total_volume();
        gpu.pour(Vec3::new(130.5, 2.0, 0.5), 3.0);
        gpu.rain(0.01);
        run_gpu(&mut sim, &device, &queue, &mut gpu, 1, 1.0 / 60.0);
        let added = gpu.total_volume() - before;
        // 3 m³ poured + 160·2·0.01 rained, minus whatever the sea took.
        assert!((added - 6.2).abs() < 1.0, "added {added}");
        // Foam forms on the breaking swell.
        let foam = gpu.marks.iter().fold(0.0f32, |m, v| m.max(v[0]));
        assert!(foam > 0.1, "no foam on the GPU, max {foam}");
    }

    /// GPU vs CPU on a 1 km² landscape, first mostly still, then with
    /// everything flowing after heavy rain. Run with
    /// `ENGINE_GPU_TESTS=1 cargo test --release -p engine --lib bench_gpu_water -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_gpu_water() {
        if !gpu_tests_enabled() {
            return;
        }
        let (device, queue) = device();
        let noise = crate::proc::Noise::new(11);
        let terrain = HeightMapTerrain(move |x: f32, z: f32| {
            noise.fbm2(x * 0.004, z * 0.004, 5, 2.0, 0.5) * 40.0 + 10.0
        });
        for size in [512.0f32, 1024.0] {
            let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(size), 1.0, -40.0, 80.0);
            let base = Water::new(region, &terrain);
            let setup = |backend| {
                let mut w = base.clone();
                w.flood_sea(0.0);
                w.add_source(crate::water::WaterSource {
                    position: Vec3::new(size * 0.5, 70.0, size * 0.5),
                    rate: 5.0,
                });
                w.set_backend(backend);
                w
            };
            let mut cpu = setup(WaterBackend::Cpu);
            let mut gpu = setup(WaterBackend::Gpu);
            let mut sim = GpuWaterSim::new(&device);
            let gpu_frame = |sim: &mut GpuWaterSim, w: &mut Water| {
                w.update(1.0 / 60.0);
                let mut encoder = device.create_command_encoder(&Default::default());
                sim.encode_frame(&device, &queue, &mut encoder, w);
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::Wait).expect("poll");
            };
            for phase in ["mostly still", "everything flowing"] {
                if phase == "everything flowing" {
                    cpu.rain(0.05);
                    gpu.rain(0.05);
                }
                for _ in 0..30 {
                    cpu.update(1.0 / 60.0);
                    gpu_frame(&mut sim, &mut gpu);
                }
                let frames = 120;
                let t = std::time::Instant::now();
                for _ in 0..frames {
                    cpu.update(1.0 / 60.0);
                }
                let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
                let t = std::time::Instant::now();
                for _ in 0..frames {
                    gpu_frame(&mut sim, &mut gpu);
                }
                let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
                eprintln!(
                    "{size} m, {} cells, {phase}: CPU {cpu_ms:.2} ms/frame ({} substeps), \
                     GPU {gpu_ms:.2} ms/frame incl. read-back (x{:.1})",
                    cpu.cell_count(),
                    cpu.stats().substeps,
                    cpu_ms / gpu_ms
                );
            }
        }
    }
}
