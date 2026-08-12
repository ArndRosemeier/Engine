//! Asynchronous streaming of multi-layer, globally anchored chunks.
//!
//! The scheduler owns *when* chunks are built and uploaded; a [`ChunkBuilder`]
//! owns *what* they contain. A build produces one [`ChunkPayload`]: any number
//! of typed mesh layers plus the CPU contact grid for the same samples, so land,
//! water, and feet can never come from different bakes.
//!
//! Failures are propagated, not swallowed: a builder error surfaces from
//! [`ChunkStream::sync`] on the frame it is observed.

use crate::contact::ContactGrid;
use crate::error::{EngineError, EngineResult};
use crate::mesh::BuiltMesh;
use crate::space::{ChunkCoord, ChunkId, ChunkLayer, ChunkSpan, GlobalPosition, GlobalXZ};
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
    contact: Option<ContactGrid>,
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
    /// Async load ring (Chebyshev radius in chunks).
    pub radius: i32,
    /// Ring that must be resident before gameplay may start or continue.
    pub required_radius: i32,
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
}

impl ChunkStream {
    pub fn new(builder: Arc<dyn ChunkBuilder>, radius: i32) -> Self {
        let span = builder.span();
        let (tx, rx) = mpsc::channel();
        Self {
            builder,
            span,
            radius: radius.max(1),
            required_radius: 1,
            keep_margin: 1,
            max_jobs_per_frame: 6,
            max_uploads_per_frame: 2,
            epoch: 1,
            resident: HashMap::new(),
            inflight: HashSet::new(),
            ready: VecDeque::new(),
            tx,
            rx,
        }
    }

    pub fn with_required_radius(mut self, radius: i32) -> Self {
        self.required_radius = radius.max(0);
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
        self.inflight.len() + self.ready.len()
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

    /// Invalidate every in-flight bake and drop all resident chunks.
    ///
    /// Results from the previous epoch are discarded when they arrive, so a
    /// world change can never upload geometry from the world we just left.
    pub fn reset(&mut self, world: &mut World) {
        self.epoch += 1;
        for (coord, chunk) in self.resident.drain() {
            for layer in chunk.layers {
                world.clear_anchored_chunk(ChunkId::new(coord, layer));
            }
        }
        self.inflight.clear();
        self.ready.clear();
        while self.rx.try_recv().is_ok() {}
    }

    /// Are all chunks in the required ring resident?
    pub fn required_ready(&self, focus: GlobalXZ) -> bool {
        let center = self.focus_chunk(focus);
        Self::ring(center, self.required_radius)
            .into_iter()
            .all(|c| self.resident.contains_key(&c))
    }

    /// Height of the drawn land surface under `p`, if that chunk is resident.
    pub fn contact_height(&self, p: GlobalXZ) -> Option<f32> {
        let coord = ChunkCoord::containing(p, self.span);
        self.resident
            .get(&coord)
            .and_then(|c| c.contact.as_ref())
            .and_then(|g| g.height_at(p))
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
        let load: HashSet<ChunkCoord> = Self::ring(center, self.radius).into_iter().collect();
        let keep: HashSet<ChunkCoord> = Self::ring(center, self.radius + self.keep_margin)
            .into_iter()
            .collect();

        let stale: Vec<ChunkCoord> = self
            .resident
            .keys()
            .copied()
            .filter(|c| !keep.contains(c))
            .collect();
        for coord in stale {
            self.unload(world, coord);
        }

        self.drain_ready()?;
        self.upload_ready(world, center, &keep)?;
        self.spawn_jobs(center, &load, priority.map(|p| self.focus_chunk(p)));
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
            .filter(|c| !self.resident.contains_key(c))
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
        if let Some(chunk) = self.resident.remove(&coord) {
            for layer in chunk.layers {
                world.clear_anchored_chunk(ChunkId::new(coord, layer));
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

    fn upload_ready(
        &mut self,
        world: &mut World,
        center: ChunkCoord,
        keep: &HashSet<ChunkCoord>,
    ) -> EngineResult<()> {
        let mut batch: Vec<(ChunkCoord, ChunkPayload)> = self.ready.drain(..).collect();
        batch.sort_by_key(|(c, _)| self.upload_priority(*c, center));
        let mut uploaded = 0usize;
        let mut rest = VecDeque::new();
        for (coord, payload) in batch {
            if !keep.contains(&coord) || self.resident.contains_key(&coord) {
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
        let mut installed = Vec::with_capacity(layers.len());
        for (layer, mesh) in layers {
            world.set_anchored_chunk(ChunkId::new(coord, layer), anchor, mesh)?;
            installed.push(layer);
        }
        self.resident.insert(
            coord,
            ResidentChunk {
                layers: installed,
                contact,
            },
        );
        Ok(())
    }

    fn spawn_jobs(
        &mut self,
        center: ChunkCoord,
        load: &HashSet<ChunkCoord>,
        priority: Option<ChunkCoord>,
    ) {
        let mut missing: Vec<ChunkCoord> = load
            .iter()
            .copied()
            .filter(|c| !self.resident.contains_key(c) && !self.inflight.contains(c))
            .collect();
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
        for coord in missing.into_iter().take(budget) {
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
    }
}

impl std::fmt::Debug for ChunkStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkStream")
            .field("span_m", &self.span.metres())
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
