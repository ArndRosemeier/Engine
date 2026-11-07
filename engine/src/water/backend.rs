//! Where the water simulation runs, and the hand-over to the GPU.
//!
//! With [`WaterBackend::Gpu`] the renderer steps the simulation on the GPU
//! every frame. The [`Water`] on the CPU keeps a copy that trails the GPU by
//! a frame or two, refreshed from read-backs, so queries (`sample`,
//! swimmers, floating debris) keep working. Edits made on the CPU (pour,
//! disturb, rain, sources) are queued and applied on the GPU; flooding,
//! clearing and terrain changes upload the whole state again.

use super::sim::Water;
use std::sync::Mutex;

/// Where [`Water::update`] does its work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaterBackend {
    /// On the CPU, in [`Water::update`]. Works without a renderer.
    #[default]
    Cpu,
    /// On the GPU, in the renderer, each frame. Much faster for large
    /// regions with a lot of moving water. Without a renderer nothing moves.
    Gpu,
}

impl WaterBackend {
    /// GPU, unless the `ENGINE_WATER_SIM` environment variable says `cpu`.
    pub fn preferred() -> Self {
        match std::env::var("ENGINE_WATER_SIM").as_deref() {
            Ok("cpu") => Self::Cpu,
            _ => Self::Gpu,
        }
    }
}

/// A change made on the CPU that the GPU must apply too.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GpuEdit {
    Depth { cell: u32, add: f32 },
    Flow { link: u32, add: f32 },
    Foam { cell: u32, amount: f32 },
}

/// What changed on the GPU in one frame, read back: the areas that ran,
/// plus a slice of everything (so the whole copy refreshes every second).
/// The values stay in one block as read; runs point into it.
#[derive(Debug, Default)]
pub(crate) struct Snapshot {
    /// The CPU state revision the GPU state descends from.
    pub generation: u64,
    pub topology: u64,
    pub raw: Vec<f32>,
    /// `[first element, offset in raw, count]` runs of depths and currents.
    pub depth: Vec<[u32; 3]>,
    pub q: Vec<[u32; 3]>,
    /// The slice of everything: cells as `[depth, head, scale, 0]`, and
    /// marks, both `[first cell, offset in raw, count]`.
    pub cells: [u32; 3],
    pub marks: [u32; 3],
}

impl Snapshot {
    fn apply(&self, water: &mut Water) {
        let raw = &self.raw;
        let fits = |to: usize, [d, s, n]: [u32; 3], stride: usize| {
            let (d, s, n) = (d as usize, s as usize, n as usize);
            (d + n <= to && s + n * stride <= raw.len()).then_some((d, s, n))
        };
        for (runs, to) in [(&self.depth, &mut water.depth), (&self.q, &mut water.q)] {
            for &run in runs {
                if let Some((d, s, n)) = fits(to.len(), run, 1) {
                    to[d..d + n].copy_from_slice(&raw[s..s + n]);
                }
            }
        }
        if let Some((d, s, n)) = fits(water.depth.len(), self.cells, 4) {
            for (o, v) in water.depth[d..d + n]
                .iter_mut()
                .zip(raw[s..].chunks_exact(4))
            {
                *o = v[0];
            }
        }
        if let Some((d, s, n)) = fits(water.marks.len(), self.marks, 4) {
            let m: &[[f32; 4]] = bytemuck::cast_slice(&raw[s..s + n * 4]);
            water.marks[d..d + n].copy_from_slice(m);
        }
    }
}

/// Read-backs waiting for the CPU copy (older ones are dropped).
pub(crate) const MAX_INBOX: usize = 120;

#[derive(Debug, Default)]
pub(crate) struct GpuLink {
    /// Seconds to simulate on the next GPU frame.
    pub pending_dt: f32,
    pub edits: Vec<GpuEdit>,
    pub rain: f32,
    /// Oldest first.
    pub inbox: Vec<Snapshot>,
    /// Tiles that ran on the GPU in the newest frame read back.
    pub running_tiles: Option<usize>,
}

/// Shared between `Water` (CPU side) and the renderer, which only sees the
/// water through `&World`.
#[derive(Debug, Default)]
pub(crate) struct GpuLinkCell(pub Mutex<GpuLink>);

impl Clone for GpuLinkCell {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl Water {
    pub fn backend(&self) -> WaterBackend {
        self.backend
    }

    /// Run the simulation on the CPU or on the GPU (see [`WaterBackend`]).
    pub fn set_backend(&mut self, backend: WaterBackend) {
        if backend != self.backend {
            self.backend = backend;
            self.state_rev += 1;
            *self.gpu.0.get_mut().unwrap_or_else(|e| e.into_inner()) = GpuLink::default();
        }
    }

    /// Builder form of [`Water::set_backend`].
    pub fn with_backend(mut self, backend: WaterBackend) -> Self {
        self.set_backend(backend);
        self
    }

    pub(crate) fn on_gpu(&self) -> bool {
        self.backend == WaterBackend::Gpu
    }

    pub(crate) fn gpu_edit(&mut self, edit: GpuEdit) {
        if self.on_gpu() {
            self.gpu
                .0
                .get_mut()
                .unwrap_or_else(|e| e.into_inner())
                .edits
                .push(edit);
        }
    }

    /// GPU backend: apply the read-backs and queue `dt` for the GPU.
    pub(crate) fn update_gpu(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        let link = self.gpu.0.get_mut().unwrap_or_else(|e| e.into_inner());
        link.pending_dt = (link.pending_dt + dt).min(0.1);
        let running = link.running_tiles.take();
        let inbox = std::mem::take(&mut link.inbox);
        if let Some(n) = running {
            self.gpu_running = n;
        }
        for s in inbox {
            if s.generation == self.state_rev && s.topology == self.topology_rev {
                s.apply(self);
            }
        }
        self.time += dt as f64;
        self.sea_clock += dt as f64;
    }
}
