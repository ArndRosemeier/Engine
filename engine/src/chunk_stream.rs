//! Asynchronous streaming of multi-layer, globally anchored chunks.
//!
//! The scheduler owns *when* chunks are built and uploaded; a [`ChunkBuilder`]
//! owns *what* they contain. A build produces one [`ChunkPayload`]: any number
//! of typed mesh layers plus the CPU contact grid for the same samples, so land,
//! water, and feet can never come from different bakes.
//!
//! Failures are propagated, not swallowed: a builder error surfaces from
//! [`ChunkStream::sync`] on the frame it is observed.

use crate::contact::{ContactGrid, ContactSnapshot};
use crate::error::{EngineError, EngineResult};
use crate::mesh::BuiltMesh;
use crate::space::{
    ChunkCoord, ChunkId, ChunkLayer, ChunkLevel, ChunkSpan, GlobalPosition, GlobalXZ,
};
use crate::world::World;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// Everything one chunk bake produced.
#[derive(Debug)]
pub struct ChunkPayload {
    anchor: GlobalPosition,
    layers: Vec<(ChunkLayer, BuiltMesh)>,
    contact: Option<ContactGrid>,
}

impl ChunkPayload {
    /// `anchor` is the global position that chunk-local vertices are relative to.
    pub fn new(anchor: GlobalPosition) -> Self {
        Self {
            anchor,
            layers: Vec::new(),
            contact: None,
        }
    }

    /// Add a mesh layer. A layer may only be supplied once per chunk.
    pub fn with_layer(mut self, layer: ChunkLayer, mesh: BuiltMesh) -> EngineResult<Self> {
        if self.layers.iter().any(|(l, _)| *l == layer) {
            return Err(EngineError::InvalidValue(format!(
                "chunk layer {layer:?} supplied twice"
            )));
        }
        self.layers.push((layer, mesh));
        Ok(self)
    }

    pub fn with_contact(mut self, contact: ContactGrid) -> Self {
        self.contact = Some(contact);
        self
    }

    pub fn anchor(&self) -> GlobalPosition {
        self.anchor
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, layer: ChunkLayer) -> Option<&BuiltMesh> {
        self.layers
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, mesh)| mesh)
    }
}

/// Produces chunk content for the scheduler. Called on worker threads.
pub trait ChunkBuilder: Send + Sync + 'static {
    /// Edge length of one chunk.
    fn span(&self) -> ChunkSpan;

    /// Build `coord`, or `Ok(None)` when that address deliberately has no
    /// content (outside the generated world). `Err` means the bake is broken.
    fn build(&self, coord: ChunkCoord) -> EngineResult<Option<ChunkPayload>>;
}

struct ResidentChunk {
    layers: Vec<ChunkLayer>,
    /// Shared so a snapshot can be handed to a worker thread without copying
    /// every height in the ring.
    contact: Option<Arc<ContactGrid>>,
}

struct JobResult {
    coord: ChunkCoord,
    epoch: u64,
    payload: EngineResult<Option<ChunkPayload>>,
}

/// Streams [`ChunkBuilder`] output around a moving focus.
pub struct ChunkStream {
    builder: Arc<dyn ChunkBuilder>,
    span: ChunkSpan,
    level: ChunkLevel,
    /// Async load ring (Chebyshev radius in chunks).
    pub radius: i32,
    /// Ring that must be resident before gameplay may start or continue.
    pub required_radius: i32,
    /// Chunks this close to the focus are left to a finer tier.
    pub hole_radius: Option<i32>,
    /// Extra chunks kept beyond the load ring before unloading (hysteresis).
    pub keep_margin: i32,
    pub max_jobs_per_frame: usize,
    pub max_uploads_per_frame: usize,
    epoch: u64,
    resident: HashMap<ChunkCoord, ResidentChunk>,
    inflight: HashSet<ChunkCoord>,
    ready: VecDeque<(ChunkCoord, ChunkPayload)>,
    tx: Sender<JobResult>,
    rx: Receiver<JobResult>,
    load: HashSet<ChunkCoord>,
    keep: HashSet<ChunkCoord>,
    stale: Vec<ChunkCoord>,
    missing: Vec<ChunkCoord>,
    upload_scratch: Vec<(ChunkCoord, ChunkPayload)>,
    /// Resident chunks whose mesh is stale. Still drawn until the replacement lands.
    dirty: HashSet<ChunkCoord>,
}

impl ChunkStream {
    pub fn new(builder: Arc<dyn ChunkBuilder>, radius: i32) -> Self {
        let span = builder.span();
        let (tx, rx) = mpsc::channel();
        Self {
            builder,
            span,
            level: ChunkLevel::FINEST,
            radius: radius.max(1),
            required_radius: 1,
            hole_radius: None,
            keep_margin: 1,
            max_jobs_per_frame: 6,
            max_uploads_per_frame: 2,
            epoch: 1,
            resident: HashMap::new(),
            inflight: HashSet::new(),
            ready: VecDeque::new(),
            tx,
            rx,
            load: HashSet::new(),
            keep: HashSet::new(),
            stale: Vec::new(),
            missing: Vec::new(),
            upload_scratch: Vec::new(),
            dirty: HashSet::new(),
        }
    }

    pub fn with_required_radius(mut self, radius: i32) -> Self {
        self.required_radius = radius.max(0);
        self
    }

    /// Which detail tier these chunks belong to.
    ///
    /// Two streams at the same level would fight over the same chunk ids in
    /// the world, so every tier past the finest needs its own.
    pub fn with_level(mut self, level: ChunkLevel) -> Self {
        self.level = level;
        self
    }

    /// Leave the chunks within `radius` of the focus to a finer tier.
    ///
    /// The hole must be no larger than what that finer tier is *guaranteed* to
    /// cover, or the ground will be missing between them: a hole of radius `h`
    /// reaches `(h + 1) × span` from the player in the worst case, since the
    /// player may stand at the edge of their own chunk.
    pub fn with_hole_radius(mut self, radius: i32) -> Self {
        self.hole_radius = Some(radius.max(0));
        self
    }

    pub fn with_keep_margin(mut self, margin: i32) -> Self {
        self.keep_margin = margin.max(0);
        self
    }

    pub fn with_budgets(mut self, jobs: usize, uploads: usize) -> Self {
        self.max_jobs_per_frame = jobs.max(1);
        self.max_uploads_per_frame = uploads.max(1);
        self
    }

    pub fn span(&self) -> ChunkSpan {
        self.span
    }

    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    pub fn pending_count(&self) -> usize {
        self.inflight.len() + self.ready.len() + self.dirty.len()
    }

    /// Bakes running on worker threads right now.
    pub fn inflight_count(&self) -> usize {
        self.inflight.len()
    }

    /// Finished bakes waiting for an upload slot.
    pub fn ready_count(&self) -> usize {
        self.ready.len()
    }

    pub fn is_resident(&self, coord: ChunkCoord) -> bool {
        self.resident.contains_key(&coord)
    }

    pub fn focus_chunk(&self, focus: GlobalXZ) -> ChunkCoord {
        ChunkCoord::containing(focus, self.span)
    }

    /// Mark these resident chunks stale so the next sync bakes them again.
    ///
    /// The current mesh stays drawn until the replacement uploads: dropping it
    /// first leaves a hole (bright sky through the ground) for as long as the
    /// bake takes. Bumps the epoch so in-flight results from the old content
    /// cannot land.
    pub fn invalidate(&mut self, _world: &mut World, coords: &[ChunkCoord]) {
        if coords.is_empty() {
            return;
        }
        self.epoch += 1;
        self.inflight.clear();
        self.ready.clear();
        while self.rx.try_recv().is_ok() {}
        for &coord in coords {
            if self.resident.contains_key(&coord) {
                self.dirty.insert(coord);
            }
        }
    }

    /// Invalidate every in-flight bake and drop all resident chunks.
    ///
    /// Results from the previous epoch are discarded when they arrive, so a
    /// world change can never upload geometry from the world we just left.
    pub fn reset(&mut self, world: &mut World) {
        self.epoch += 1;
        let level = self.level;
        for (coord, chunk) in self.resident.drain() {
            for layer in chunk.layers {
                world.clear_anchored_chunk(ChunkId::at_level(coord, layer, level));
            }
        }
        self.inflight.clear();
        self.ready.clear();
        self.load.clear();
        self.keep.clear();
        self.stale.clear();
        self.missing.clear();
        self.upload_scratch.clear();
        self.dirty.clear();
        while self.rx.try_recv().is_ok() {}
    }

    /// Are all chunks in the required ring resident?
    pub fn required_ready(&self, focus: GlobalXZ) -> bool {
        let center = self.focus_chunk(focus);
        Self::ring(center, self.required_radius)
            .into_iter()
            .filter(|c| !self.in_hole(*c, center))
            .all(|c| self.resident.contains_key(&c))
    }

    pub fn level(&self) -> ChunkLevel {
        self.level
    }

    /// Height of the drawn land surface under `p`, if that chunk is resident.
    pub fn contact_height(&self, p: GlobalXZ) -> Option<f32> {
        let coord = ChunkCoord::containing(p, self.span);
        self.resident
            .get(&coord)
            .and_then(|c| c.contact.as_ref())
            .and_then(|g| g.height_at(p))
    }

    /// The ground bakes resident right now, for a worker thread to stand things on.
    pub fn contact_snapshot(&self) -> ContactSnapshot {
        ContactSnapshot::new(
            self.span,
            self.resident
                .iter()
                .filter_map(|(coord, chunk)| Some((*coord, Arc::clone(chunk.contact.as_ref()?))))
                .collect(),
        )
    }

    fn in_hole(&self, coord: ChunkCoord, center: ChunkCoord) -> bool {
        self.hole_radius
            .is_some_and(|hole| coord.ring_distance(center) <= hole)
    }

    fn fill_ring(
        center: ChunkCoord,
        radius: i32,
        hole_radius: Option<i32>,
        out: &mut HashSet<ChunkCoord>,
    ) {
        out.clear();
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let c = center.offset(dx, dz);
                let in_hole = hole_radius.is_some_and(|hole| c.ring_distance(center) <= hole);
                if !in_hole {
                    out.insert(c);
                }
            }
        }
    }

    fn ring(center: ChunkCoord, radius: i32) -> Vec<ChunkCoord> {
        let mut keys = Vec::new();
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                keys.push(center.offset(dx, dz));
            }
        }
        keys
    }

    /// Stream around `focus`, optionally prioritising the chunk under `priority`
    /// (the movement leading edge).
    pub fn sync(
        &mut self,
        world: &mut World,
        focus: GlobalXZ,
        priority: Option<GlobalXZ>,
    ) -> EngineResult<()> {
        let center = self.focus_chunk(focus);
        let radius = self.radius;
        let keep_r = self.radius + self.keep_margin;
        let hole = self.hole_radius;
        Self::fill_ring(center, radius, hole, &mut self.load);
        Self::fill_ring(center, keep_r, hole, &mut self.keep);

        self.stale.clear();
        self.stale.extend(
            self.resident
                .keys()
                .copied()
                .filter(|c| !self.keep.contains(c)),
        );
        let mut stale = std::mem::take(&mut self.stale);
        for coord in stale.drain(..) {
            self.unload(world, coord);
        }
        self.stale = stale;

        self.drain_ready()?;
        self.upload_ready(world, center)?;
        self.spawn_jobs(center, priority.map(|p| self.focus_chunk(p)));
        Ok(())
    }

    /// Build and upload the required ring on the calling thread.
    ///
    /// Chunks are built **sequentially** on purpose: each bake may already use a
    /// parallel sample grid, and nesting a parallel iterator around that
    /// deadlocks the global pool (it looks like a hard freeze at startup).
    pub fn ensure_required_blocking(
        &mut self,
        world: &mut World,
        focus: GlobalXZ,
    ) -> EngineResult<()> {
        let center = self.focus_chunk(focus);
        let mut missing: Vec<ChunkCoord> = Self::ring(center, self.required_radius)
            .into_iter()
            .filter(|c| !self.resident.contains_key(c) && !self.in_hole(*c, center))
            .collect();
        missing.sort_by_key(|c| (c.walk_distance(center), c.x, c.z));
        for coord in missing {
            // A bake may already be in flight; take ownership here and let the
            // stale result be discarded rather than uploading the chunk twice.
            self.inflight.remove(&coord);
            let payload = self.builder.build(coord)?;
            self.install(world, coord, payload)?;
        }
        Ok(())
    }

    fn unload(&mut self, world: &mut World, coord: ChunkCoord) {
        self.dirty.remove(&coord);
        if let Some(chunk) = self.resident.remove(&coord) {
            for layer in chunk.layers {
                world.clear_anchored_chunk(ChunkId::at_level(coord, layer, self.level));
            }
        }
    }

    fn drain_ready(&mut self) -> EngineResult<()> {
        while let Ok(result) = self.rx.try_recv() {
            self.inflight.remove(&result.coord);
            if result.epoch != self.epoch {
                continue;
            }
            match result.payload? {
                Some(payload) => self.ready.push_back((result.coord, payload)),
                None => {
                    self.ready
                        .push_back((result.coord, ChunkPayload::new(GlobalPosition::ORIGIN)));
                }
            }
        }
        Ok(())
    }

    fn upload_ready(&mut self, world: &mut World, center: ChunkCoord) -> EngineResult<()> {
        let mut batch = std::mem::take(&mut self.upload_scratch);
        batch.clear();
        batch.extend(self.ready.drain(..));
        batch.sort_by_key(|(c, _)| self.upload_priority(*c, center));
        let mut uploaded = 0usize;
        let mut rest = VecDeque::new();
        for (coord, payload) in batch.drain(..) {
            if !self.keep.contains(&coord) {
                continue;
            }
            if self.resident.contains_key(&coord) && !self.dirty.contains(&coord) {
                continue;
            }
            // Required chunks always upload: gameplay is blocked until they land.
            let required = coord.ring_distance(center) <= self.required_radius;
            if required || uploaded < self.max_uploads_per_frame {
                self.install(world, coord, Some(payload))?;
                uploaded += 1;
            } else {
                rest.push_back((coord, payload));
            }
        }
        self.upload_scratch = batch;
        self.ready = rest;
        Ok(())
    }

    fn upload_priority(&self, coord: ChunkCoord, center: ChunkCoord) -> (i32, i32) {
        let required = if coord.ring_distance(center) <= self.required_radius {
            0
        } else {
            1
        };
        (required, coord.walk_distance(center))
    }

    fn install(
        &mut self,
        world: &mut World,
        coord: ChunkCoord,
        payload: Option<ChunkPayload>,
    ) -> EngineResult<()> {
        let Some(payload) = payload else {
            // Deliberately empty address (outside the world) — resident, no mesh.
            self.dirty.remove(&coord);
            self.resident.insert(
                coord,
                ResidentChunk {
                    layers: Vec::new(),
                    contact: None,
                },
            );
            return Ok(());
        };
        let ChunkPayload {
            anchor,
            layers,
            contact,
        } = payload;
        self.dirty.remove(&coord);
        if let Some(old) = self.resident.remove(&coord) {
            for layer in &old.layers {
                if !layers.iter().any(|(l, _)| l == layer) {
                    world.clear_anchored_chunk(ChunkId::at_level(coord, *layer, self.level));
                }
            }
        }
        let mut installed = Vec::with_capacity(layers.len());
        for (layer, mesh) in layers {
            world.set_anchored_chunk(ChunkId::at_level(coord, layer, self.level), anchor, mesh)?;
            installed.push(layer);
        }
        self.resident.insert(
            coord,
            ResidentChunk {
                layers: installed,
                contact: contact.map(Arc::new),
            },
        );
        Ok(())
    }

    fn spawn_jobs(&mut self, center: ChunkCoord, priority: Option<ChunkCoord>) {
        let mut missing = std::mem::take(&mut self.missing);
        missing.clear();
        missing.extend(self.load.iter().copied().filter(|c| {
            !self.inflight.contains(c) && (!self.resident.contains_key(c) || self.dirty.contains(c))
        }));
        // Required ring first, then the movement leading edge, then nearest.
        // Never block the main thread on the leading edge — that froze the
        // window on the first movement key.
        missing.sort_by_key(|c| {
            let required = if c.ring_distance(center) <= self.required_radius {
                0
            } else {
                1
            };
            let lead = match priority {
                Some(p) if *c == p => 0,
                _ => 1,
            };
            (required, lead, c.walk_distance(center), c.x, c.z)
        });
        let budget = self.max_jobs_per_frame.min(missing.len());
        for coord in missing.iter().copied().take(budget) {
            self.inflight.insert(coord);
            let builder = Arc::clone(&self.builder);
            let tx = self.tx.clone();
            let epoch = self.epoch;
            rayon::spawn(move || {
                let payload = builder.build(coord);
                let _ = tx.send(JobResult {
                    coord,
                    epoch,
                    payload,
                });
            });
        }
        self.missing = missing;
    }
}

impl std::fmt::Debug for ChunkStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkStream")
            .field("span_m", &self.span.metres())
            .field("level", &self.level.index())
            .field("radius", &self.radius)
            .field("required_radius", &self.required_radius)
            .field("resident", &self.resident.len())
            .field("inflight", &self.inflight.len())
            .field("ready", &self.ready.len())
            .field("epoch", &self.epoch)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::space::GlobalXZ;
    use glam::Vec3;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    const SPAN: f64 = 100.0;

    struct FlatBuilder {
        builds: Arc<AtomicUsize>,
        /// Chunks outside this Chebyshev radius report "no content".
        domain: i32,
        fail_at: Option<ChunkCoord>,
    }

    impl FlatBuilder {
        fn new() -> Self {
            Self {
                builds: Arc::new(AtomicUsize::new(0)),
                domain: 100,
                fail_at: None,
            }
        }
    }

    impl ChunkBuilder for FlatBuilder {
        fn span(&self) -> ChunkSpan {
            ChunkSpan::new(SPAN).expect("span")
        }

        fn build(&self, coord: ChunkCoord) -> EngineResult<Option<ChunkPayload>> {
            self.builds.fetch_add(1, Ordering::Relaxed);
            if Some(coord) == self.fail_at {
                return Err(EngineError::InvalidValue("bad chunk".into()));
            }
            if coord.ring_distance(ChunkCoord::new(0, 0)) > self.domain {
                return Ok(None);
            }
            let span = self.span();
            let origin = coord.origin(span);
            let anchor = origin.with_height(0.0)?;
            let mut mesh = Mesh::new();
            let s = SPAN as f32;
            let a = mesh.add_point(Vec3::new(0.0, 0.0, 0.0))?;
            let b = mesh.add_point(Vec3::new(s, 0.0, 0.0))?;
            let c = mesh.add_point(Vec3::new(s, 0.0, s))?;
            let d = mesh.add_point(Vec3::new(0.0, 0.0, s))?;
            mesh.add_quad(a, d, c, b)?;
            let contact = ContactGrid::new(origin, SPAN, 2, vec![0.0; 4])?;
            Ok(Some(
                ChunkPayload::new(anchor)
                    .with_layer(ChunkLayer::Land, mesh.build())?
                    .with_contact(contact),
            ))
        }
    }

    fn pump(stream: &mut ChunkStream, world: &mut World, focus: GlobalXZ, frames: usize) {
        for _ in 0..frames {
            stream.sync(world, focus, None).expect("sync");
            if stream.pending_count() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn required_ring_is_resident_after_blocking_prepare() {
        let mut world = World::new();
        let mut stream = ChunkStream::new(Arc::new(FlatBuilder::new()), 3).with_required_radius(1);
        let focus = GlobalXZ::at(50.0, 50.0);
        stream
            .ensure_required_blocking(&mut world, focus)
            .expect("prepare");
        assert!(stream.required_ready(focus));
        assert_eq!(stream.resident_count(), 9);
        assert!(world.has_anchored_chunk(ChunkId::new(ChunkCoord::new(0, 0), ChunkLayer::Land)));
    }

    #[test]
    fn contact_height_comes_from_the_resident_bake() {
        let mut world = World::new();
        let mut stream = ChunkStream::new(Arc::new(FlatBuilder::new()), 2).with_required_radius(0);
        let focus = GlobalXZ::at(5_050.0, 25.0);
        stream
            .ensure_required_blocking(&mut world, focus)
            .expect("prepare");
        assert_eq!(stream.contact_height(focus), Some(0.0));
        // Far outside the resident ring there is no contact to stand on.
        assert_eq!(stream.contact_height(GlobalXZ::at(9_000_000.0, 0.0)), None);
    }

    #[test]
    fn async_ring_fills_without_blocking_the_caller() {
        let mut world = World::new();
        let mut stream = ChunkStream::new(Arc::new(FlatBuilder::new()), 2)
            .with_required_radius(0)
            .with_budgets(8, 8);
        let focus = GlobalXZ::at(0.0, 0.0);
        let t0 = Instant::now();
        stream.sync(&mut world, focus, None).expect("sync");
        assert!(
            t0.elapsed() < Duration::from_millis(200),
            "first sync must not bake the whole ring inline"
        );
        pump(&mut stream, &mut world, focus, 400);
        assert_eq!(stream.resident_count(), 25, "5x5 load ring should fill");
    }

    #[test]
    fn keep_margin_holds_chunks_behind_the_walker() {
        let mut world = World::new();
        let mut stream = ChunkStream::new(Arc::new(FlatBuilder::new()), 1)
            .with_required_radius(0)
            .with_keep_margin(2)
            .with_budgets(8, 8);
        let focus = GlobalXZ::at(50.0, 50.0);
        pump(&mut stream, &mut world, focus, 400);
        assert!(stream.is_resident(ChunkCoord::new(0, 0)));
        stream
            .sync(&mut world, GlobalXZ::at(150.0, 50.0), None)
            .expect("sync");
        assert!(
            stream.is_resident(ChunkCoord::new(0, 0)),
            "origin chunk should survive one chunk of movement"
        );
    }

    #[test]
    fn empty_addresses_count_as_resident() {
        let mut world = World::new();
        let builder = FlatBuilder {
            domain: 0,
            ..FlatBuilder::new()
        };
        let mut stream = ChunkStream::new(Arc::new(builder), 1).with_required_radius(1);
        let focus = GlobalXZ::at(50.0, 50.0);
        stream
            .ensure_required_blocking(&mut world, focus)
            .expect("prepare");
        assert!(stream.required_ready(focus));
        assert!(!world.has_anchored_chunk(ChunkId::new(ChunkCoord::new(1, 0), ChunkLayer::Land)));
    }

    #[test]
    fn builder_failures_surface_from_sync() {
        let mut world = World::new();
        let builder = FlatBuilder {
            fail_at: Some(ChunkCoord::new(0, 0)),
            ..FlatBuilder::new()
        };
        let mut stream = ChunkStream::new(Arc::new(builder), 1)
            .with_required_radius(0)
            .with_budgets(8, 8);
        let focus = GlobalXZ::at(50.0, 50.0);
        let mut saw_error = false;
        for _ in 0..400 {
            if stream.sync(&mut world, focus, None).is_err() {
                saw_error = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(saw_error, "a failing bake must not be silently dropped");
    }

    #[test]
    fn a_coarse_tier_leaves_a_hole_and_does_not_evict_the_fine_one() {
        // Both tiers address chunk (0, 0). Before the level was part of the
        // identity, whichever uploaded last owned the entity and the other
        // tier's ground vanished.
        let mut world = World::new();
        let mut fine = ChunkStream::new(Arc::new(FlatBuilder::new()), 1)
            .with_required_radius(1)
            .with_budgets(8, 8);
        let coarse = ChunkLevel::new(1);
        let mut wide = ChunkStream::new(Arc::new(FlatBuilder::new()), 2)
            .with_level(coarse)
            .with_required_radius(0)
            .with_hole_radius(0)
            .with_budgets(8, 8);

        let focus = GlobalXZ::at(50.0, 50.0);
        pump(&mut fine, &mut world, focus, 400);
        pump(&mut wide, &mut world, focus, 400);

        let home = ChunkCoord::new(0, 0);
        assert!(world.has_anchored_chunk(ChunkId::new(home, ChunkLayer::Land)));
        assert!(
            !world.has_anchored_chunk(ChunkId::at_level(home, ChunkLayer::Land, coarse)),
            "the chunk under the player belongs to the fine tier"
        );
        assert!(
            world.has_anchored_chunk(ChunkId::at_level(
                ChunkCoord::new(2, 0),
                ChunkLayer::Land,
                coarse
            )),
            "the coarse tier still covers everything outside its hole"
        );
        assert_eq!(wide.resident_count(), 24, "5x5 minus the hole");
    }

    #[test]
    fn walking_out_of_a_hole_fills_it_in() {
        // The hole follows the player, so ground the fine tier has left behind
        // has to come back at the coarse resolution rather than stay missing.
        let mut world = World::new();
        let mut wide = ChunkStream::new(Arc::new(FlatBuilder::new()), 2)
            .with_level(ChunkLevel::new(1))
            .with_required_radius(0)
            .with_hole_radius(0)
            .with_budgets(8, 8);
        pump(&mut wide, &mut world, GlobalXZ::at(50.0, 50.0), 400);
        assert!(!wide.is_resident(ChunkCoord::new(0, 0)));

        pump(&mut wide, &mut world, GlobalXZ::at(250.0, 50.0), 400);
        assert!(wide.is_resident(ChunkCoord::new(0, 0)));
        assert!(!wide.is_resident(ChunkCoord::new(2, 0)));
    }

    #[test]
    fn stale_results_from_a_previous_world_are_discarded() {
        let mut world = World::new();
        let builder = Arc::new(FlatBuilder::new());
        let builds = Arc::clone(&builder.builds);
        let mut stream = ChunkStream::new(builder, 2)
            .with_required_radius(0)
            .with_budgets(8, 8);
        let focus = GlobalXZ::at(50.0, 50.0);
        stream.sync(&mut world, focus, None).expect("sync");
        stream.reset(&mut world);
        assert_eq!(stream.resident_count(), 0);
        assert_eq!(stream.pending_count(), 0);
        // Let any in-flight jobs land; they belong to the previous epoch.
        std::thread::sleep(Duration::from_millis(50));
        stream.sync(&mut world, focus, None).expect("sync");
        let before = builds.load(Ordering::Relaxed);
        assert!(before > 0);
        assert!(
            stream.resident_count() <= 25,
            "stale uploads must not leak into the new session"
        );
    }
}
