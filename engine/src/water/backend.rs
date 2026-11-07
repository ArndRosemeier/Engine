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

/// The GPU state read back after a frame.
#[derive(Debug, Default)]
pub(crate) struct Snapshot {
    /// The CPU state revision the GPU state descends from.
    pub generation: u64,
    pub depth: Vec<f32>,
    pub q: Vec<f32>,
    pub marks: Option<Vec<[f32; 4]>>,
}

#[derive(Debug, Default)]
pub(crate) struct GpuLink {
    /// Seconds to simulate on the next GPU frame.
    pub pending_dt: f32,
    pub edits: Vec<GpuEdit>,
    pub rain: f32,
    pub inbox: Option<Snapshot>,
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

    /// GPU backend: take the newest read-back and queue `dt` for the GPU.
    pub(crate) fn update_gpu(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        let link = self.gpu.0.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = link.inbox.take() {
            if s.generation == self.state_rev
                && s.depth.len() == self.depth.len()
                && s.q.len() == self.q.len()
            {
                self.depth = s.depth;
                self.q = s.q;
                if let Some(m) = s.marks.filter(|m| m.len() == self.marks.len()) {
                    self.marks = m;
                }
            }
        }
        link.pending_dt = (link.pending_dt + dt).min(0.1);
        self.time += dt as f64;
        self.sea_clock += dt as f64;
    }
}
