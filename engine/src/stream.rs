//! World streaming: huge procedural worlds loaded around the camera.
//!
//! You write a [`WorldGenerator`]: the ground height (and, if you like, 3D
//! shape for caves and overhangs), and what stands on each tile of ground
//! (trees, rocks, buildings, lamps). [`WorldStream`] runs it on worker
//! threads around the camera and puts the results into the [`World`] a little
//! at a time, so moving never stalls a frame:
//!
//! - **Diggable ground near the camera.** Voxel terrain in square blocks
//!   around the focus, generated from your height and shape. Digging and
//!   building work as usual, and edits are kept when a block unloads and come
//!   back when you return (and go into save files).
//! - **Ground to the horizon.** Beyond the voxels, height-field tiles in a
//!   quadtree (bigger and coarser with distance), drawn with the same shader
//!   so the far ground looks like the near ground.
//! - **Things on the ground.** [`WorldGenerator::populate`] fills one tile at
//!   a time. Models you place there are drawn with per-instance LOD, in the
//!   same GPU batches for every tile.
//! - **Water.** With [`WorldGenerator::water_level`], a water simulation
//!   window follows the camera and fills seas and lakes as you go.
//!
//! GI, shadows, physics (the ground near moving bodies), navigation and the
//! water read the streamed voxel terrain like any other.
//!
//! ```no_run
//! use engine::prelude::*;
//! use engine::stream::{StreamSettings, Tile, WorldGenerator, WorldStream};
//!
//! struct Hills {
//!     noise: Noise,
//!     oak: LodId,
//! }
//!
//! impl WorldGenerator for Hills {
//!     fn height(&self, x: f32, z: f32) -> f32 {
//!         self.noise.fbm2(x * 0.002, z * 0.002, 5, 2.0, 0.5) * 120.0
//!     }
//!
//!     fn populate(&self, tile: &mut Tile) -> EngineResult<()> {
//!         let (min, max) = (tile.min(), tile.max());
//!         let places = Scatter::new(tile.seed())
//!             .rect(min.x, min.y, max.x, max.y)
//!             .spacing(9.0)
//!             .on_ground(|x, z| self.height(x, z))
//!             .places()?;
//!         tile.place_all(self.oak, places);
//!         Ok(())
//!     }
//! }
//!
//! fn main() -> EngineResult<()> {
//!     Engine::run("hills", |world, frame| {
//!         if frame.first {
//!             // A model spawned with no places is a template the tiles copy.
//!             let oak = world.spawn_lod(tree_model(TreeSpecies::Oak, 0)?.lod_levels())?;
//!             let generator = Hills { noise: Noise::new(7), oak };
//!             world.set_stream(Some(WorldStream::new(generator, StreamSettings::default())))?;
//!             world.prepare_stream(10.0); // ground under the camera first
//!         }
//!         Ok(())
//!     })
//! }
//! ```

use crate::error::{EngineError, EngineResult};
use crate::lamps::{Lamp, LampId};
use crate::lod::LodId;
use crate::mesh::{BuiltMesh, Mesh};
use crate::place::Place;
use crate::volume::{EditedChunk, Volume, VolumeSource, CHUNK_SIZE};
use crate::voxel_terrain::{FarTileKey, FarTileMesh, VoxelTerrain};
use crate::world::{EntityId, World};
use glam::{IVec2, IVec3, Vec2, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

/// Your world: its ground and what stands on it. Called on worker threads,
/// so it must be cheap to share (`Send + Sync`) and the same input must give
/// the same output (tiles are rebuilt whenever you come back).
pub trait WorldGenerator: Send + Sync + 'static {
    /// Ground height in metres at a point. Drawn to the horizon, and the
    /// base of [`Self::density`].
    fn height(&self, x: f32, z: f32) -> f32;

    /// Solid (positive) or open (negative) at `p`, in metres; `ground` is
    /// [`Self::height`] at `p`. Override for caves, overhangs and arches; the
    /// default is plain ground. Only near the camera (the diggable voxels)
    /// uses it; far ground is drawn from the height.
    fn density(&self, p: Vec3, ground: f32) -> f32 {
        ground - p.y
    }

    /// Put things on one tile of ground: models (with LOD), meshes, lamps.
    /// See [`Tile`]. Runs on a worker thread whenever the tile comes into
    /// range, so it may take a while; keep it deterministic.
    fn populate(&self, tile: &mut Tile) -> EngineResult<()> {
        let _ = tile;
        Ok(())
    }

    /// The ocean, if the world has one: its level and swell. It is drawn
    /// out to the horizon, and the water window around the camera opens
    /// onto it.
    fn sea(&self) -> Option<crate::water::Sea> {
        None
    }

    /// Surface height of standing water (sea or lake) at a point, or `None`
    /// for dry land. The water window fills open ground below it. Defaults
    /// to the sea level everywhere.
    fn water_level(&self, x: f32, z: f32) -> Option<f32> {
        let _ = (x, z);
        self.sea().map(|s| s.level)
    }

    /// Whether this world has water at all (skips the water window when
    /// false). Defaults to having a sea, or water at the origin.
    fn has_water(&self) -> bool {
        self.sea().is_some() || self.water_level(0.0, 0.0).is_some()
    }
}

/// How far and how fine a [`WorldStream`] loads.
#[derive(Clone, Debug)]
pub struct StreamSettings {
    /// Edge of one voxel in metres (diggable ground resolution).
    pub voxel_size: f32,
    /// Coarsest voxel level of detail: one voxel block is
    /// `32 * 2^voxel_lod` voxels across (128 m with the defaults).
    pub voxel_lod: u32,
    /// Voxel blocks loaded in each direction around the camera's block
    /// (1: a 3 x 3 square). One more ring is kept before unloading.
    pub voxel_blocks: u32,
    /// Voxels are generated from this far below the lowest ground of a
    /// column to this far above the highest; widen for caves and overhangs.
    pub voxel_below: f32,
    pub voxel_above: f32,
    /// How far the ground is drawn, in metres.
    pub view_distance: f32,
    /// Far tiles are split while closer than `far_detail` times their size:
    /// higher draws more detail further out.
    pub far_detail: f32,
    /// Cells along one side of a far tile.
    pub far_cells: usize,
    /// Edge of one populated tile, in metres.
    pub tile_size: f32,
    /// Tiles are populated out to this distance, in metres. Keep it at or
    /// beyond the farthest LOD distance of your models.
    pub content_distance: f32,
    /// Water simulation window edge (metres) and cell size.
    pub water_window: f32,
    pub water_cell: f32,
    /// Worker threads for generation (0: all cores but two).
    pub threads: usize,
    /// Main-thread milliseconds per frame for putting finished work into
    /// the world. At least one item goes in each frame.
    pub install_budget_ms: f32,
}

impl Default for StreamSettings {
    fn default() -> Self {
        Self {
            voxel_size: 1.0,
            voxel_lod: 2,
            voxel_blocks: 1,
            voxel_below: 6.0,
            voxel_above: 6.0,
            view_distance: 6000.0,
            far_detail: 2.0,
            far_cells: 32,
            tile_size: 128.0,
            content_distance: 1200.0,
            water_window: 384.0,
            water_cell: 1.0,
            threads: 0,
            install_budget_ms: 2.0,
        }
    }
}

impl StreamSettings {
    fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| {
            Err(EngineError::InvalidValue(format!(
                "stream settings: {what}"
            )))
        };
        if !(self.voxel_size.is_finite() && self.voxel_size > 0.0) {
            return bad("voxel_size must be > 0");
        }
        if self.voxel_lod > crate::voxel_terrain::VOXEL_MAX_LOD {
            return bad("voxel_lod is above VOXEL_MAX_LOD");
        }
        if !(self.view_distance.is_finite() && self.view_distance > 0.0) {
            return bad("view_distance must be > 0");
        }
        if !(self.far_detail.is_finite() && self.far_detail >= 1.0) {
            return bad("far_detail must be >= 1");
        }
        if !(2..=256).contains(&self.far_cells) {
            return bad("far_cells must be within 2..=256");
        }
        if !(self.tile_size.is_finite() && self.tile_size >= 4.0) {
            return bad("tile_size must be >= 4 m");
        }
        if !(self.content_distance.is_finite() && self.content_distance >= 0.0) {
            return bad("content_distance must be >= 0");
        }
        if !(self.water_window.is_finite()
            && self.water_cell.is_finite()
            && self.water_cell > 0.0
            && self.water_window >= self.water_cell * 16.0)
        {
            return bad("water_window must be at least 16 water cells");
        }
        if !(self.voxel_below >= 0.0 && self.voxel_above >= 0.0) {
            return bad("voxel_below and voxel_above must be >= 0");
        }
        Ok(())
    }

    /// Edge of one voxel block in metres.
    pub fn block_size(&self) -> f32 {
        self.voxel_size * CHUNK_SIZE as f32 * (1u32 << self.voxel_lod) as f32
    }
}

/// One tile of ground being populated by [`WorldGenerator::populate`].
pub struct Tile {
    coord: IVec2,
    min: Vec2,
    size: f32,
    models: HashMap<LodId, Vec<Place>>,
    meshes: Vec<TileMesh>,
    lamps: Vec<Lamp>,
    effects: Vec<crate::particles::ParticleEmitter>,
}

struct TileMesh {
    built: BuiltMesh,
    albedo: (
        Option<crate::mesh::AlbedoMap>,
        Option<crate::mesh::MaterialMaps>,
    ),
    place: Place,
    solid: bool,
}

impl Tile {
    fn new(coord: IVec2, size: f32) -> Self {
        Self {
            coord,
            min: coord.as_vec2() * size,
            size,
            models: HashMap::new(),
            meshes: Vec::new(),
            lamps: Vec::new(),
            effects: Vec::new(),
        }
    }

    /// Tile index on the tile grid.
    pub fn coord(&self) -> IVec2 {
        self.coord
    }

    /// Lowest corner (x, z) in metres.
    pub fn min(&self) -> Vec2 {
        self.min
    }

    /// Highest corner (x, z) in metres.
    pub fn max(&self) -> Vec2 {
        self.min + Vec2::splat(self.size)
    }

    pub fn centre(&self) -> Vec2 {
        self.min + Vec2::splat(self.size * 0.5)
    }

    pub fn size(&self) -> f32 {
        self.size
    }

    /// Whether `(x, z)` lies on this tile (min edge inclusive): place each
    /// thing on exactly one tile.
    pub fn contains(&self, x: f32, z: f32) -> bool {
        let max = self.max();
        x >= self.min.x && x < max.x && z >= self.min.y && z < max.y
    }

    /// A seed for this tile (the same every time it is built). Mix in your
    /// world seed for different worlds.
    pub fn seed(&self) -> u32 {
        let h = (self.coord.x as u32).wrapping_mul(0x9E37_79B1)
            ^ (self.coord.y as u32).wrapping_mul(0x85EB_CA77);
        let h = (h ^ (h >> 15)).wrapping_mul(0x2C1B_3C6D);
        h ^ (h >> 12)
    }

    /// Place one instance of a model. `model` is a LOD set spawned with
    /// [`World::spawn_lod`] (no places needed): every tile's instances share
    /// its meshes and draw in the same batches.
    pub fn place(&mut self, model: LodId, place: Place) {
        self.models.entry(model).or_default().push(place);
    }

    pub fn place_all(&mut self, model: LodId, places: impl IntoIterator<Item = Place>) {
        self.models.entry(model).or_default().extend(places);
    }

    /// Spawn a one-off mesh (a building, a bridge). It is built on the
    /// worker thread.
    pub fn spawn(&mut self, mesh: Mesh, place: Place) {
        self.push_mesh(mesh, place, false);
    }

    /// Like [`Self::spawn`], with a fixed physics body of the exact mesh, so
    /// bodies and characters collide with it.
    pub fn spawn_solid(&mut self, mesh: Mesh, place: Place) {
        self.push_mesh(mesh, place, true);
    }

    fn push_mesh(&mut self, mesh: Mesh, place: Place, solid: bool) {
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        self.meshes.push(TileMesh {
            built: mesh.build(),
            albedo,
            place,
            solid,
        });
    }

    /// Add a lamp (street lamp, camp fire) that lives with the tile.
    pub fn add_lamp(&mut self, lamp: Lamp) {
        self.lamps.push(lamp);
    }

    /// Add a running particle emitter (chimney smoke, a waterfall's mist)
    /// that lives with the tile.
    pub fn add_effect(&mut self, emitter: crate::particles::ParticleEmitter) {
        self.effects.push(emitter);
    }

    /// A camp fire `size` metres across: flames, smoke, embers and its light.
    pub fn add_fire(&mut self, at: Vec3, size: f32) {
        use crate::particles::ParticleEmitter as E;
        self.effects.push(E::flames(at, size));
        self.effects.push(E::smoke(at, size));
        self.effects.push(E::embers(at, size));
        self.lamps.push(
            Lamp::fire(at + Vec3::Y * 0.5 * size)
                .intensity(5.0 * size.max(0.1).sqrt())
                .flicker(0.35),
        );
    }

    /// How many things the tile holds so far.
    pub fn len(&self) -> usize {
        self.models.values().map(Vec::len).sum::<usize>()
            + self.meshes.len()
            + self.lamps.len()
            + self.effects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Counters for an overlay or a test. See [`WorldStream::stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamStats {
    /// Voxel blocks resident, and chunks they hold.
    pub voxel_blocks: usize,
    pub voxel_chunks: usize,
    /// Edited chunks kept while their block is unloaded.
    pub stored_edits: usize,
    /// Far terrain tiles shown, and built ones held.
    pub far_shown: usize,
    pub far_built: usize,
    /// Populated tiles in the world, and things in them.
    pub tiles: usize,
    pub tile_things: usize,
    /// Jobs running on the workers, and finished ones waiting to go in.
    pub jobs: usize,
    pub waiting: usize,
    /// What went into the world this frame, and how long that took.
    pub installed: usize,
    pub install_ms: f32,
    /// Whether the water window is up.
    pub water: bool,
}

// ------------------------------------------------------------------ workers

enum JobOut {
    Block {
        block: IVec2,
        chunks: Vec<(IVec3, Vec<f32>)>,
    },
    Far {
        key: FarTileKey,
        mesh: Arc<FarTileMesh>,
    },
    Tile {
        coord: IVec2,
        tile: EngineResult<Tile>,
    },
    Water {
        centre: IVec2,
        water: Box<crate::water::Water>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum JobKey {
    Block(IVec2),
    Far(FarTileKey),
    Tile(IVec2),
    Water,
}

/// The generator as a [`VolumeSource`]: density in voxels, columns sampled
/// once per (x, z).
struct GenSource {
    gen: Arc<dyn WorldGenerator>,
    voxel: f32,
}

impl VolumeSource for GenSource {
    fn density(&self, p: Vec3) -> f32 {
        self.gen.density(p, self.gen.height(p.x, p.z)) / self.voxel
    }

    fn fill(&self, origin: Vec3, step: f32, n: usize, out: &mut Vec<f32>) {
        let mut ground = Vec::with_capacity(n * n);
        for z in 0..n {
            for x in 0..n {
                let (px, pz) = (origin.x + x as f32 * step, origin.z + z as f32 * step);
                ground.push(self.gen.height(px, pz));
            }
        }
        fill_with_ground(&*self.gen, origin, step, n, &ground, self.voxel, out);
    }
}

fn fill_with_ground(
    gen: &dyn WorldGenerator,
    origin: Vec3,
    step: f32,
    n: usize,
    ground: &[f32],
    voxel: f32,
    out: &mut Vec<f32>,
) {
    out.clear();
    out.reserve(n * n * n);
    let inv = 1.0 / voxel;
    for y in 0..n {
        let py = origin.y + y as f32 * step;
        for z in 0..n {
            let pz = origin.z + z as f32 * step;
            for x in 0..n {
                let px = origin.x + x as f32 * step;
                let d = gen.density(Vec3::new(px, py, pz), ground[z * n + x]) * inv;
                out.push(if d.is_finite() { d } else { -1.0 });
            }
        }
    }
}

/// Generate the voxel chunks of one block that hold some surface.
fn build_block(
    gen: &dyn WorldGenerator,
    settings: &StreamSettings,
    block: IVec2,
) -> Vec<(IVec3, Vec<f32>)> {
    let vs = settings.voxel_size;
    let cw = vs * CHUNK_SIZE as f32;
    let per = 1i32 << settings.voxel_lod;
    let n = (CHUNK_SIZE + 1) as usize;
    let mut out = Vec::new();
    let mut ground = vec![0.0; n * n];
    let mut samples = Vec::new();
    for cz in 0..per {
        for cx in 0..per {
            let col = block * per + IVec2::new(cx, cz);
            let ox = col.x as f32 * cw;
            let oz = col.y as f32 * cw;
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for z in 0..n {
                for x in 0..n {
                    let h = gen.height(ox + x as f32 * vs, oz + z as f32 * vs);
                    let h = if h.is_finite() { h } else { 0.0 };
                    ground[z * n + x] = h;
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            let y0 = ((lo - settings.voxel_below - 2.0 * vs) / cw).floor() as i32;
            let y1 = ((hi + settings.voxel_above + 2.0 * vs) / cw).floor() as i32;
            for cy in y0..=y1 {
                let origin = Vec3::new(ox, cy as f32 * cw, oz);
                fill_with_ground(gen, origin, vs, n, &ground, vs, &mut samples);
                let solid = samples.iter().any(|&d| d > 0.0);
                let open = samples.iter().any(|&d| d <= 0.0);
                if solid && open {
                    out.push((IVec3::new(col.x, cy, col.y), std::mem::take(&mut samples)));
                }
            }
        }
    }
    out
}

fn build_far(gen: &dyn WorldGenerator, settings: &StreamSettings, key: FarTileKey) -> FarTileMesh {
    let size = settings.block_size() * (1u32 << key.level) as f32;
    let n = settings.far_cells;
    let cell = size / n as f32;
    let (x0, z0) = (key.x as f32 * size, key.z as f32 * size);
    let w = n + 3;
    let mut heights = Vec::with_capacity(w * w);
    for z in 0..w {
        for x in 0..w {
            let h = gen.height(x0 + (x as f32 - 1.0) * cell, z0 + (z as f32 - 1.0) * cell);
            heights.push(if h.is_finite() { h } else { 0.0 });
        }
    }
    // Sunk a little so voxels win wherever both draw; skirts cover the
    // steps between tiles of different detail.
    let sink = 0.04 * cell;
    let skirt = 2.0 + cell * 1.5;
    FarTileMesh::from_heights(x0, z0, cell, n, &heights, sink, skirt)
}

fn build_tile(gen: &dyn WorldGenerator, coord: IVec2, size: f32) -> EngineResult<Tile> {
    let mut tile = Tile::new(coord, size);
    gen.populate(&mut tile)?;
    Ok(tile)
}

// ------------------------------------------------------------------ resident state

struct ResidentTile {
    lods: Vec<LodId>,
    entities: Vec<EntityId>,
    lamps: Vec<LampId>,
    emitters: Vec<crate::particles::EmitterId>,
    things: usize,
}

/// Generated chunks of one voxel block: key and density samples.
type BlockChunks = Vec<(IVec3, Vec<f32>)>;

/// Streams a [`WorldGenerator`] around the camera. Hand it to
/// [`World::set_stream`]; the engine updates it every frame.
pub struct WorldStream {
    gen: Arc<dyn WorldGenerator>,
    settings: StreamSettings,
    terrain_style: Option<Box<dyn Fn(VoxelTerrain) -> VoxelTerrain + Send + Sync>>,
    pool: rayon::ThreadPool,
    tx: Sender<JobOut>,
    rx: std::sync::Mutex<Receiver<JobOut>>,
    inflight: HashSet<JobKey>,
    focus: Option<Vec3>,
    last_focus: Vec3,
    // Voxel blocks.
    blocks: HashSet<IVec2>,
    block_chunks: HashMap<IVec2, usize>,
    ready_blocks: Vec<(IVec2, BlockChunks)>,
    edits: HashMap<IVec3, EditedChunk>,
    seen_layout: u64,
    // Far terrain.
    far_built: HashMap<FarTileKey, Arc<FarTileMesh>>,
    far_shown: HashSet<FarTileKey>,
    far_wanted: Vec<FarTileKey>,
    // Populated tiles.
    tiles: HashMap<IVec2, ResidentTile>,
    ready_tiles: Vec<(IVec2, Tile)>,
    // Water window.
    water_centre: Option<IVec2>,
    water_pending: Option<(IVec2, Box<crate::water::Water>)>,
    errors: Vec<EngineError>,
    stats: StreamStats,
}

impl std::fmt::Debug for WorldStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorldStream")
            .field("settings", &self.settings)
            .field("stats", &self.stats)
            .finish()
    }
}

impl WorldStream {
    /// Stream `generator` with `settings`. Nothing happens until it is set
    /// on a world with [`World::set_stream`].
    pub fn new(generator: impl WorldGenerator, settings: StreamSettings) -> Self {
        Self::from_arc(Arc::new(generator), settings)
    }

    pub fn from_arc(generator: Arc<dyn WorldGenerator>, settings: StreamSettings) -> Self {
        let threads = if settings.threads > 0 {
            settings.threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2))
                .unwrap_or(2)
                .max(1)
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("engine-stream-{i}"))
            .start_handler(|_| crate::parallel::lower_thread_priority())
            .build()
            .expect("stream worker pool");
        let (tx, rx) = mpsc::channel();
        Self {
            gen: generator,
            settings,
            terrain_style: None,
            pool,
            tx,
            rx: std::sync::Mutex::new(rx),
            inflight: HashSet::new(),
            focus: None,
            last_focus: Vec3::splat(f32::NAN),
            blocks: HashSet::new(),
            block_chunks: HashMap::new(),
            ready_blocks: Vec::new(),
            edits: HashMap::new(),
            seen_layout: u64::MAX,
            far_built: HashMap::new(),
            far_shown: HashSet::new(),
            far_wanted: Vec::new(),
            tiles: HashMap::new(),
            ready_tiles: Vec::new(),
            water_centre: None,
            water_pending: None,
            errors: Vec::new(),
            stats: StreamStats::default(),
        }
    }

    /// Change the look of the streamed voxel terrain (material, colour,
    /// beaches): `stream.with_terrain(|t| t.with_beach(1.5))`.
    pub fn with_terrain(
        mut self,
        style: impl Fn(VoxelTerrain) -> VoxelTerrain + Send + Sync + 'static,
    ) -> Self {
        self.terrain_style = Some(Box::new(style));
        self
    }

    pub fn settings(&self) -> &StreamSettings {
        &self.settings
    }

    pub fn generator(&self) -> &Arc<dyn WorldGenerator> {
        &self.gen
    }

    /// Ground height from the generator (anywhere, loaded or not).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.gen.height(x, z)
    }

    /// Load around `focus` instead of the camera (a player far below a high
    /// camera, say). `None` follows the camera again.
    pub fn set_focus(&mut self, focus: Option<Vec3>) {
        self.focus = focus.filter(|f| f.is_finite());
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    /// Errors from [`WorldGenerator::populate`] since the last call. The
    /// tile that failed stays empty.
    pub fn take_errors(&mut self) -> Vec<EngineError> {
        std::mem::take(&mut self.errors)
    }

    /// Edited voxel chunks kept while their block is unloaded.
    pub fn stored_edits(&self) -> impl Iterator<Item = &EditedChunk> {
        self.edits.values()
    }

    /// Whether the ground around the focus is loaded: the voxel block under
    /// it and its neighbours, and the far ground.
    pub fn is_ready(&self) -> bool {
        let centre = self.block_of(self.last_focus);
        let r = self.settings.voxel_blocks as i32;
        self.last_focus.is_finite()
            && (-r..=r)
                .all(|dz| (-r..=r).all(|dx| self.blocks.contains(&(centre + IVec2::new(dx, dz)))))
            && !self.far_shown.is_empty()
            && self
                .far_wanted
                .iter()
                .all(|k| self.far_built.contains_key(k))
    }

    fn block_of(&self, p: Vec3) -> IVec2 {
        if !p.is_finite() {
            return IVec2::ZERO;
        }
        (Vec2::new(p.x, p.z) / self.settings.block_size())
            .floor()
            .as_ivec2()
    }

    /// Set up the voxel terrain this stream fills.
    pub(crate) fn attach(&mut self, world: &mut World) -> EngineResult<()> {
        self.settings.validate()?;
        let source = Arc::new(GenSource {
            gen: self.gen.clone(),
            voxel: self.settings.voxel_size,
        });
        let volume = Volume::try_new(self.settings.voxel_size)?.with_source(source);
        let mut terrain = VoxelTerrain::new(volume).with_max_lod(self.settings.voxel_lod);
        if let Some(style) = &self.terrain_style {
            terrain = style(terrain);
        }
        world.set_voxel_terrain(Some(terrain));
        self.blocks.clear();
        self.block_chunks.clear();
        self.far_shown.clear();
        self.tiles.clear();
        self.seen_layout = u64::MAX;
        self.water_centre = None;
        Ok(())
    }

    /// Remove everything this stream put into the world.
    pub(crate) fn detach(&mut self, world: &mut World) {
        let coords: Vec<IVec2> = self.tiles.keys().copied().collect();
        for c in coords {
            self.unload_tile(world, c);
        }
        world.set_voxel_terrain(None);
        if self.water_centre.is_some() {
            world.set_water(None);
        }
        self.blocks.clear();
        self.far_shown.clear();
    }

    /// One frame: take finished work, start new work, put things in and
    /// take far things out. The engine calls this every frame.
    pub fn update(&mut self, world: &mut World) {
        let started = Instant::now();
        let focus = self.focus.unwrap_or_else(|| world.camera().eye());
        if !focus.is_finite() {
            return;
        }
        self.last_focus = focus;
        self.receive();

        let wanted_blocks = self.wanted_blocks(focus);
        let far = self.far_tree(focus, &wanted_blocks);
        let wanted_tiles = self.wanted_tiles(focus);

        let t0 = Instant::now();
        self.schedule(focus, &wanted_blocks, &far, &wanted_tiles);
        self.stats.installed = 0;
        let t1 = Instant::now();
        self.install_blocks(world, focus, &wanted_blocks);
        let t2 = Instant::now();
        self.update_far(world, &far);
        let t3 = Instant::now();
        self.install_tiles(world, focus, started);
        let t4 = Instant::now();
        self.unload(world, focus);
        let t5 = Instant::now();
        self.update_water(world, focus);
        self.store_stray_edits(world);
        let t6 = Instant::now();
        if std::env::var_os("ENGINE_STREAM_TRACE").is_some() && (t6 - started).as_secs_f32() > 0.003
        {
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f32() * 1000.0;
            eprintln!(
                "stream slow frame: wanted {:.2} schedule {:.2} blocks {:.2} far {:.2} tiles {:.2} unload {:.2} water {:.2}",
                ms(started, t0), ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t5, t6)
            );
        }

        self.stats.voxel_blocks = self.blocks.len();
        self.stats.voxel_chunks = self.block_chunks.values().sum();
        self.stats.stored_edits = self.edits.len();
        self.stats.far_built = self.far_built.len();
        self.stats.far_shown = self.far_shown.len();
        self.stats.tiles = self.tiles.len();
        self.stats.tile_things = self.tiles.values().map(|t| t.things).sum();
        self.stats.jobs = self.inflight.len();
        self.stats.waiting = self.ready_blocks.len() + self.ready_tiles.len();
        self.stats.install_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.stats.water = self.water_centre.is_some();
    }

    /// Run updates (waiting for the workers) until the ground around the
    /// focus is loaded, or `timeout_s` passed. For the first frame, so the
    /// player does not start in the void or fall through the ground.
    pub fn prepare(&mut self, world: &mut World, timeout_s: f32) -> bool {
        let start = Instant::now();
        let budget = self.settings.install_budget_ms;
        self.settings.install_budget_ms = 1.0e6;
        loop {
            self.update(world);
            if self.is_ready() && self.inflight.is_empty() {
                break;
            }
            if start.elapsed().as_secs_f32() > timeout_s {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        self.settings.install_budget_ms = budget;
        self.is_ready()
    }

    fn receive(&mut self) {
        let rx = self.rx.get_mut().unwrap_or_else(|e| e.into_inner());
        let mut outs = Vec::new();
        while let Ok(out) = rx.try_recv() {
            outs.push(out);
        }
        for out in outs {
            match out {
                JobOut::Block { block, chunks } => {
                    self.inflight.remove(&JobKey::Block(block));
                    self.ready_blocks.push((block, chunks));
                }
                JobOut::Far { key, mesh } => {
                    self.inflight.remove(&JobKey::Far(key));
                    self.far_built.insert(key, mesh);
                }
                JobOut::Tile { coord, tile } => {
                    self.inflight.remove(&JobKey::Tile(coord));
                    match tile {
                        Ok(tile) => self.ready_tiles.push((coord, tile)),
                        Err(e) => {
                            eprintln!("world stream: tile {coord} failed: {e}");
                            self.errors.push(e);
                            // Leave it empty rather than retrying every frame.
                            self.ready_tiles
                                .push((coord, Tile::new(coord, self.settings.tile_size)));
                        }
                    }
                }
                JobOut::Water { centre, water } => {
                    self.inflight.remove(&JobKey::Water);
                    self.water_pending = Some((centre, water));
                }
            }
        }
    }

    // -------------------------------------------------------------- wanted sets

    fn wanted_blocks(&self, focus: Vec3) -> Vec<IVec2> {
        let c = self.block_of(focus);
        let r = self.settings.voxel_blocks as i32;
        let mut out = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
        for dz in -r..=r {
            for dx in -r..=r {
                out.push(c + IVec2::new(dx, dz));
            }
        }
        out.sort_by_key(|b| (*b - c).abs().max_element());
        out
    }

    fn far_tree(&self, focus: Vec3, voxel: &[IVec2]) -> FarTree {
        let base = self.settings.block_size();
        let view = self.settings.view_distance;
        let mut top = 0u8;
        while base * ((1u64 << top) as f32) < view * 0.5 && top < 20 {
            top += 1;
        }
        let size = base * (1u64 << top) as f32;
        let f = Vec2::new(focus.x, focus.z);
        let lo = ((f - view) / size).floor().as_ivec2();
        let hi = ((f + view) / size).floor().as_ivec2();
        let (vlo, vhi) = voxel.iter().fold(
            (IVec2::splat(i32::MAX), IVec2::splat(i32::MIN)),
            |(a, b), v| (a.min(*v), b.max(*v)),
        );
        let mut tree = FarTree::default();
        for z in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let key = FarTileKey { level: top, x, z };
                if self.far_rect_distance(key, f) <= view {
                    tree.roots.push(key);
                    self.far_expand(key, f, vlo, vhi, &mut tree);
                }
            }
        }
        tree
    }

    fn far_size(&self, level: u8) -> f32 {
        self.settings.block_size() * (1u64 << level) as f32
    }

    fn far_has_built_ancestor(&self, key: FarTileKey) -> bool {
        let mut k = key;
        while k.level < 16 {
            k = FarTileKey {
                level: k.level + 1,
                x: k.x.div_euclid(2),
                z: k.z.div_euclid(2),
            };
            if self.far_built.contains_key(&k) {
                return true;
            }
        }
        false
    }

    fn block_rect_distance(&self, block: IVec2, f: Vec2) -> f32 {
        tile_distance(block, self.settings.block_size(), f)
    }

    fn far_rect_distance(&self, key: FarTileKey, f: Vec2) -> f32 {
        let size = self.far_size(key.level);
        let min = Vec2::new(key.x as f32, key.z as f32) * size;
        let max = min + Vec2::splat(size);
        (f.clamp(min, max) - f).length()
    }

    fn far_expand(&self, key: FarTileKey, f: Vec2, vlo: IVec2, vhi: IVec2, tree: &mut FarTree) {
        let size = self.far_size(key.level);
        // Block range this tile covers.
        let per = 1i32 << key.level;
        let blo = IVec2::new(key.x, key.z) * per;
        let bhi = blo + IVec2::splat(per - 1);
        let over_voxels = blo.cmple(vhi).all() && bhi.cmpge(vlo).all();
        let split = key.level > 0
            && (self.far_rect_distance(key, f) < self.settings.far_detail * size || over_voxels);
        if !split {
            tree.leaves.insert(key);
            return;
        }
        tree.split.insert(key);
        for child in far_children(key) {
            if self.far_rect_distance(child, f) <= self.settings.view_distance {
                self.far_expand(child, f, vlo, vhi, tree);
            }
        }
    }

    fn wanted_tiles(&self, focus: Vec3) -> Vec<IVec2> {
        let size = self.settings.tile_size;
        let reach = self.settings.content_distance;
        if reach <= 0.0 {
            return Vec::new();
        }
        let f = Vec2::new(focus.x, focus.z);
        let lo = ((f - reach) / size).floor().as_ivec2();
        let hi = ((f + reach) / size).floor().as_ivec2();
        let mut out = Vec::new();
        for z in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let c = IVec2::new(x, z);
                if tile_distance(c, size, f) <= reach {
                    out.push(c);
                }
            }
        }
        out.sort_by(|a, b| tile_distance(*a, size, f).total_cmp(&tile_distance(*b, size, f)));
        out
    }

    // -------------------------------------------------------------- scheduling

    fn schedule(&mut self, focus: Vec3, blocks: &[IVec2], far: &FarTree, tiles: &[IVec2]) {
        let max = self.pool.current_num_threads() * 2 + 2;
        let f = Vec2::new(focus.x, focus.z);
        let ready_blocks: HashSet<IVec2> = self.ready_blocks.iter().map(|(b, _)| *b).collect();
        let ready_tiles: HashSet<IVec2> = self.ready_tiles.iter().map(|(t, _)| *t).collect();
        // Candidates by urgency: the voxel block under the camera, then far
        // ground (cheap, and holes there show sky), then the other blocks
        // and tiles by distance.
        // Everything else by distance, so near tiles are not starved while
        // flying fast. Far ground that only refines a tile already shown (an
        // ancestor is built) counts as twice as far; far ground with nothing
        // to stand in for it is urgent (it would show a hole).
        let mut ranked: Vec<(f32, JobKey)> = Vec::new();
        for k in far.leaves.iter().copied() {
            if self.far_built.contains_key(&k) {
                continue;
            }
            let d = self.far_rect_distance(k, f);
            let rank = if self.far_has_built_ancestor(k) {
                d * 2.0
            } else {
                -1.0e6 + d
            };
            ranked.push((rank, JobKey::Far(k)));
        }
        self.far_wanted = far.leaves.iter().copied().collect();
        let mut jobs: Vec<JobKey> = Vec::new();
        let mut first_block = true;
        for b in blocks {
            if self.blocks.contains(b) || ready_blocks.contains(b) {
                continue;
            }
            if first_block {
                // The block under the camera goes before anything.
                jobs.push(JobKey::Block(*b));
                first_block = false;
            } else {
                ranked.push((self.block_rect_distance(*b, f), JobKey::Block(*b)));
            }
        }
        let size = self.settings.tile_size;
        for t in tiles {
            if !self.tiles.contains_key(t) && !ready_tiles.contains(t) {
                ranked.push((tile_distance(*t, size, f), JobKey::Tile(*t)));
            }
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        jobs.extend(ranked.into_iter().map(|(_, j)| j));
        for job in jobs {
            if self.inflight.len() >= max {
                break;
            }
            if self.inflight.contains(&job) {
                continue;
            }
            self.spawn_job(job);
        }
    }

    fn spawn_job(&mut self, job: JobKey) {
        self.inflight.insert(job);
        let gen = self.gen.clone();
        let settings = self.settings.clone();
        let tx = self.tx.clone();
        self.pool.spawn(move || {
            let out = match job {
                JobKey::Block(block) => JobOut::Block {
                    block,
                    chunks: build_block(&*gen, &settings, block),
                },
                JobKey::Far(key) => JobOut::Far {
                    key,
                    mesh: Arc::new(build_far(&*gen, &settings, key)),
                },
                JobKey::Tile(coord) => JobOut::Tile {
                    coord,
                    tile: build_tile(&*gen, coord, settings.tile_size),
                },
                JobKey::Water => unreachable!("water jobs are spawned by update_water"),
            };
            let _ = tx.send(out);
        });
    }

    // -------------------------------------------------------------- voxels

    fn install_blocks(&mut self, world: &mut World, focus: Vec3, wanted: &[IVec2]) {
        if self.ready_blocks.is_empty() {
            return;
        }
        let keep = self.block_keep_range(focus);
        let ready = std::mem::take(&mut self.ready_blocks);
        let wanted: HashSet<IVec2> = wanted.iter().copied().collect();
        let Some(terrain) = world.voxel_terrain_mut() else {
            return;
        };
        let volume = terrain.volume_mut();
        let per = 1i32 << self.settings.voxel_lod;
        for (block, chunks) in ready {
            if !wanted.contains(&block) && !keep(block) {
                continue; // walked away while it was being built
            }
            let mut count = 0;
            for (key, samples) in chunks {
                if volume.insert_generated(key, samples) {
                    count += 1;
                }
            }
            // Put back edits made here before it unloaded.
            let mine: Vec<IVec3> = self
                .edits
                .keys()
                .copied()
                .filter(|k| IVec2::new(k.x, k.z).div_euclid(IVec2::splat(per)) == block)
                .collect();
            for key in mine {
                if let Some(edit) = self.edits.remove(&key) {
                    volume.restore_edited(edit);
                    count += 1;
                }
            }
            self.blocks.insert(block);
            self.block_chunks.insert(block, count);
            self.stats.installed += 1;
        }
    }

    fn block_keep_range(&self, focus: Vec3) -> impl Fn(IVec2) -> bool {
        let c = self.block_of(focus);
        let r = self.settings.voxel_blocks as i32 + 1;
        move |b: IVec2| (b - c).abs().max_element() <= r
    }

    fn unload_block(&mut self, world: &mut World, block: IVec2) {
        self.blocks.remove(&block);
        self.block_chunks.remove(&block);
        let Some(terrain) = world.voxel_terrain_mut() else {
            return;
        };
        let per = 1i32 << self.settings.voxel_lod;
        let volume = terrain.volume_mut();
        let keys: Vec<IVec3> = volume
            .chunk_keys()
            .filter(|k| IVec2::new(k.x, k.z).div_euclid(IVec2::splat(per)) == block)
            .collect();
        for key in keys {
            if let Some(edit) = volume.evict(key) {
                self.edits.insert(key, edit);
            }
        }
    }

    /// Edits made outside the loaded blocks (a far explosion) go to the
    /// store too, or they would float there on their own.
    fn store_stray_edits(&mut self, world: &mut World) {
        let per = 1i32 << self.settings.voxel_lod;
        let Some(terrain) = world.voxel_terrain_mut() else {
            return;
        };
        let volume = terrain.volume_mut();
        if volume.layout_rev() == self.seen_layout {
            return;
        }
        let stray: Vec<IVec3> = volume
            .edited_chunks()
            .filter(|k| {
                !self
                    .blocks
                    .contains(&IVec2::new(k.x, k.z).div_euclid(IVec2::splat(per)))
            })
            .collect();
        for key in stray {
            if let Some(edit) = volume.evict(key) {
                self.edits.insert(key, edit);
            }
        }
        self.seen_layout = volume.layout_rev();
    }

    // -------------------------------------------------------------- far

    fn update_far(&mut self, world: &mut World, tree: &FarTree) {
        // What to show: each wanted tile once built; until then its built
        // parent (or its built children when merging back).
        let mut shown: Vec<FarTileKey> = Vec::new();
        for &root in &tree.roots {
            if let Some(cover) = self.far_cover(root, tree) {
                shown.extend(cover);
            }
        }
        let shown_set: HashSet<FarTileKey> = shown.into_iter().collect();
        let Some(terrain) = world.voxel_terrain_mut() else {
            return;
        };
        for key in self.far_shown.difference(&shown_set) {
            terrain.remove_far_tile(*key);
        }
        for key in shown_set.difference(&self.far_shown) {
            if let Some(mesh) = self.far_built.get(key) {
                terrain.set_far_tile(*key, mesh.clone());
            }
        }
        self.far_shown = shown_set;
        // Forget built tiles nobody wants or shows.
        let keep = |k: &FarTileKey| {
            tree.leaves.contains(k) || tree.split.contains(k) || self.far_shown.contains(k)
        };
        let drop: Vec<FarTileKey> = self
            .far_built
            .keys()
            .copied()
            .filter(|k| !keep(k))
            .collect();
        for k in drop {
            self.far_built.remove(&k);
        }
    }

    fn far_cover(&self, key: FarTileKey, tree: &FarTree) -> Option<Vec<FarTileKey>> {
        let built = |k: &FarTileKey| self.far_built.contains_key(k);
        if tree.leaves.contains(&key) {
            if built(&key) {
                return Some(vec![key]);
            }
            // Merging back: the finer tiles still shown can stand in.
            if key.level > 0 {
                let children = far_children(key);
                if children
                    .iter()
                    .all(|c| self.far_shown.contains(c) && built(c))
                {
                    return Some(children.to_vec());
                }
            }
            return None;
        }
        if !tree.split.contains(&key) {
            return Some(Vec::new()); // beyond the view distance
        }
        let mut out = Vec::new();
        for child in far_children(key) {
            if !tree.leaves.contains(&child) && !tree.split.contains(&child) {
                continue;
            }
            match self.far_cover(child, tree) {
                Some(c) => out.extend(c),
                None => return built(&key).then(|| vec![key]),
            }
        }
        Some(out)
    }

    // -------------------------------------------------------------- tiles

    fn install_tiles(&mut self, world: &mut World, focus: Vec3, started: Instant) {
        if self.ready_tiles.is_empty() {
            return;
        }
        let f = Vec2::new(focus.x, focus.z);
        let size = self.settings.tile_size;
        let reach = self.settings.content_distance + size;
        let budget = self.settings.install_budget_ms;
        // Nearest first.
        self.ready_tiles
            .sort_by(|a, b| tile_distance(b.0, size, f).total_cmp(&tile_distance(a.0, size, f)));
        let mut first = self.stats.installed == 0;
        while let Some((coord, tile)) = self.ready_tiles.pop() {
            if tile_distance(coord, size, f) > reach || self.tiles.contains_key(&coord) {
                continue;
            }
            let resident = self.install_tile(world, tile);
            self.tiles.insert(coord, resident);
            self.stats.installed += 1;
            if !first && started.elapsed().as_secs_f32() * 1000.0 > budget {
                break;
            }
            first = false;
        }
    }

    fn install_tile(&mut self, world: &mut World, tile: Tile) -> ResidentTile {
        let things = tile.len();
        let mut resident = ResidentTile {
            lods: Vec::new(),
            entities: Vec::new(),
            lamps: Vec::new(),
            emitters: Vec::new(),
            things,
        };
        let mut models: Vec<(LodId, Vec<Place>)> = tile.models.into_iter().collect();
        models.sort_by_key(|(id, _)| *id);
        for (model, places) in models {
            if places.is_empty() {
                continue;
            }
            match world.spawn_lod_like(model, &places) {
                Ok(id) => resident.lods.push(id),
                Err(e) => {
                    eprintln!("world stream: model {model:?} on tile {}: {e}", tile.coord);
                    self.errors.push(e);
                }
            }
        }
        for m in tile.meshes {
            match world.spawn_prebuilt(m.built, m.albedo, m.place) {
                Ok(id) => {
                    if m.solid {
                        if let Err(e) = world.add_body(id, crate::physics::Body::fixed()) {
                            self.errors.push(e);
                        }
                    }
                    resident.entities.push(id);
                }
                Err(e) => self.errors.push(e),
            }
        }
        for lamp in tile.lamps {
            match world.add_lamp(lamp) {
                Ok(id) => resident.lamps.push(id),
                Err(e) => self.errors.push(e),
            }
        }
        for effect in tile.effects {
            // Out of emitter slots: the tile just goes without.
            if let Ok(id) = world.particles_mut().start(effect) {
                resident.emitters.push(id);
            }
        }
        resident
    }

    fn unload_tile(&mut self, world: &mut World, coord: IVec2) {
        if let Some(t) = self.tiles.remove(&coord) {
            for id in t.lods {
                world.despawn_lod(id);
            }
            for e in t.entities {
                world.despawn(e);
            }
            for l in t.lamps {
                world.remove_lamp(l);
            }
            world.particles_mut().stop_all(&t.emitters);
        }
    }

    // -------------------------------------------------------------- unloading

    fn unload(&mut self, world: &mut World, focus: Vec3) {
        let keep = self.block_keep_range(focus);
        let gone: Vec<IVec2> = self.blocks.iter().copied().filter(|b| !keep(*b)).collect();
        for b in gone {
            self.unload_block(world, b);
        }
        let f = Vec2::new(focus.x, focus.z);
        let size = self.settings.tile_size;
        let reach = self.settings.content_distance + size;
        let gone: Vec<IVec2> = self
            .tiles
            .keys()
            .copied()
            .filter(|c| tile_distance(*c, size, f) > reach)
            .collect();
        // Unloading is cheap but not free: a few per frame is plenty.
        for c in gone.into_iter().take(16) {
            self.unload_tile(world, c);
        }
    }

    // -------------------------------------------------------------- water

    fn update_water(&mut self, world: &mut World, focus: Vec3) {
        if !self.gen.has_water() {
            return;
        }
        let cell = self.settings.water_cell;
        let window = self.settings.water_window;
        // The window moves in steps of a quarter of its size.
        let step = (window * 0.25 / cell).round().max(1.0) * cell;
        let centre = (Vec2::new(focus.x, focus.z) / step).round().as_ivec2();
        if let Some((c, water)) = self.water_pending.take() {
            let mut water = *water;
            if let Some(old) = world.water() {
                water.adopt_state(old);
            }
            world.set_water(Some(water));
            // Edits inside the window: let the water see them.
            if let Some(t) = world.voxel_terrain() {
                let size = t.volume().chunk_world_size();
                let region = *world.water().expect("just set").region();
                let mut changed = vec![false; region.column_count()];
                let mut any = false;
                for key in t.volume().edited_chunks() {
                    let min = Vec2::new(key.x as f32, key.z as f32) * size;
                    crate::water::mark_columns(&region, min, min + Vec2::splat(size), &mut changed);
                    any = true;
                }
                if any {
                    world.resync_water_columns(&changed);
                }
            }
            self.water_centre = Some(c);
        }
        let wanted = centre;
        if self.water_centre == Some(wanted) || self.inflight.contains(&JobKey::Water) {
            return;
        }
        if self.water_pending.is_some() {
            return;
        }
        // Build the new window on a worker from the generator's shape; the
        // old one keeps running until it is ready.
        self.inflight.insert(JobKey::Water);
        let gen = self.gen.clone();
        let tx = self.tx.clone();
        let settings = self.settings.clone();
        let look = world.water().map(|w| *w.look());
        self.pool.spawn(move || {
            let mid = wanted.as_vec2() * step;
            let half = Vec2::splat(window * 0.5);
            let (lo, hi) = water_height_range(&*gen, mid - half, mid + half);
            let region = crate::water::WaterRegion::new(
                mid - half,
                mid + half,
                settings.water_cell,
                lo - 4.0,
                hi + 8.0,
            );
            let shape = GenShape { gen: &*gen };
            let mut water = crate::water::Water::new(region, &shape)
                .with_backend(crate::water::WaterBackend::preferred());
            if let Some(look) = look {
                water.set_look(look);
            }
            water.fill_levels(|x, z| gen.water_level(x, z));
            if let Some(sea) = gen.sea() {
                water.set_sea(Some(sea));
            }
            let _ = tx.send(JobOut::Water {
                centre: wanted,
                water: Box::new(water),
            });
        });
    }
}

/// The generator's solid shape, for building water columns on a worker.
struct GenShape<'a> {
    gen: &'a dyn WorldGenerator,
}

impl crate::water::WaterTerrain for GenShape<'_> {
    fn density(&self, p: Vec3) -> f32 {
        self.gen.density(p, self.gen.height(p.x, p.z))
    }
}

fn water_height_range(gen: &dyn WorldGenerator, min: Vec2, max: Vec2) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let n = 24;
    for z in 0..=n {
        for x in 0..=n {
            let p = min + (max - min) * Vec2::new(x as f32, z as f32) / n as f32;
            let h = gen.height(p.x, p.y);
            if h.is_finite() {
                lo = lo.min(h);
                hi = hi.max(h);
            }
            if let Some(w) = gen.water_level(p.x, p.y) {
                hi = hi.max(w);
            }
        }
    }
    if !lo.is_finite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

#[derive(Default)]
struct FarTree {
    roots: Vec<FarTileKey>,
    leaves: HashSet<FarTileKey>,
    split: HashSet<FarTileKey>,
}

fn far_children(key: FarTileKey) -> [FarTileKey; 4] {
    let l = key.level - 1;
    let (x, z) = (key.x * 2, key.z * 2);
    [
        FarTileKey { level: l, x, z },
        FarTileKey {
            level: l,
            x: x + 1,
            z,
        },
        FarTileKey {
            level: l,
            x,
            z: z + 1,
        },
        FarTileKey {
            level: l,
            x: x + 1,
            z: z + 1,
        },
    ]
}

fn tile_distance(c: IVec2, size: f32, f: Vec2) -> f32 {
    let min = c.as_vec2() * size;
    let max = min + Vec2::splat(size);
    (f.clamp(min, max) - f).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb;
    use crate::mesh::Shape;

    /// Gentle hills with a sea below 0 and a model on every tile.
    struct TestWorld {
        model: Option<LodId>,
        water: bool,
    }

    impl WorldGenerator for TestWorld {
        fn height(&self, x: f32, z: f32) -> f32 {
            (x * 0.01).sin() * 6.0 + (z * 0.013).cos() * 4.0 + 2.0
        }

        fn populate(&self, tile: &mut Tile) -> EngineResult<()> {
            if let Some(model) = self.model {
                let c = tile.centre();
                tile.place(model, Place::new(c.x, self.height(c.x, c.y), c.y));
            }
            Ok(())
        }

        fn water_level(&self, _x: f32, _z: f32) -> Option<f32> {
            self.water.then_some(6.0)
        }
    }

    fn small() -> StreamSettings {
        StreamSettings {
            view_distance: 1500.0,
            content_distance: 200.0,
            tile_size: 64.0,
            threads: 2,
            water_window: 64.0,
            ..StreamSettings::default()
        }
    }

    fn look(world: &mut World, x: f32, z: f32) {
        world.look_first_person((x, 20.0, z), 0.0, 0.0);
    }

    fn pump(world: &mut World) {
        assert!(world.prepare_stream(30.0), "stream got ready");
    }

    #[test]
    fn gathering_unloaded_ground_matches_sampling_it_point_by_point() {
        let gen: Arc<dyn WorldGenerator> = Arc::new(TestWorld {
            model: None,
            water: false,
        });
        let source = Arc::new(GenSource {
            gen: gen.clone(),
            voxel: 1.0,
        });
        let mut volume = crate::volume::Volume::new(1.0).with_source(source.clone());
        // One chunk loaded (and edited), the rest of the region not.
        volume.dig_sphere(Vec3::new(8.0, 2.0, 8.0), 3.0);
        // A block with no chunk loaded is read from the source alone (on
        // another thread, in the renderer) and comes out the same.
        let far = IVec3::new(300, -6, 300);
        let job = volume.source_gather(far, 2, 12).expect("no chunk there");
        let mut here = Vec::new();
        volume.gather(far, 2, 12, &mut here);
        assert_eq!(job.run(), here);
        // A block that touches the edited chunk: the source's part from the
        // job, the loaded chunk laid over it.
        let near = IVec3::new(-5, -6, -7);
        let job = volume.source_gather(near, 1, 12).expect("partly loaded");
        let mut block = job.run();
        volume.gather(near, 1, 12, &mut here);
        assert_ne!(block, here, "the dig is not in the source");
        assert!(volume.overlay_loaded(near, 1, 12, &mut block));
        assert_eq!(block, here);
        assert!(!volume.overlay_loaded(near, 1, 12, &mut block));
        for stride in [1, 2, 4] {
            let origin = IVec3::new(-5, -6, -7) * stride;
            let mut fast = Vec::new();
            volume.gather(origin, stride, 12, &mut fast);
            let mut i = 0;
            for z in 0..12 {
                for y in 0..12 {
                    for x in 0..12 {
                        let g = origin + IVec3::new(x, y, z) * stride;
                        let want = volume.corner(g);
                        let got = fast[i];
                        // Packing keeps the material in the low mantissa bits.
                        assert!((want - got).abs() < 1e-3, "{g}: {want} vs {got}");
                        i += 1;
                    }
                }
            }
        }
    }

    #[test]
    fn ground_loads_around_the_camera_and_unloads_behind_it() {
        let mut world = World::new();
        look(&mut world, 10.0, 10.0);
        let gen = TestWorld {
            model: None,
            water: false,
        };
        world
            .set_stream(Some(WorldStream::new(gen, small())))
            .unwrap();
        pump(&mut world);
        let stats = world.stream().unwrap().stats();
        assert_eq!(stats.voxel_blocks, 9, "3 x 3 blocks around the camera");
        assert!(
            stats.voxel_chunks >= 9 * 16,
            "every column has surface chunks"
        );
        assert!(stats.far_shown > 0);
        // The volume answers below and above the loaded chunks from the
        // generator: deep is solid, high is open.
        let t = world.voxel_terrain().unwrap();
        assert!(t.is_solid(Vec3::new(10.0, -200.0, 10.0)));
        assert!(!t.is_solid(Vec3::new(10.0, 200.0, 10.0)));
        assert!(t.volume().has_source());

        // 2 km away: the old blocks go, new ones come.
        look(&mut world, 2000.0, 10.0);
        pump(&mut world);
        let s = world.stream().unwrap();
        assert_eq!(s.stats().voxel_blocks, 9);
        let block = s.settings().block_size();
        let t = world.voxel_terrain().unwrap();
        assert!(t
            .volume()
            .chunk_keys()
            .all(|k| (k.x as f32 * 32.0 - 2000.0).abs() < block * 3.0));
    }

    #[test]
    fn far_tiles_shown_never_overlap() {
        let mut world = World::new();
        look(&mut world, 300.0, -40.0);
        world
            .set_stream(Some(WorldStream::new(
                TestWorld {
                    model: None,
                    water: false,
                },
                small(),
            )))
            .unwrap();
        pump(&mut world);
        let s = world.stream().unwrap();
        let shown: Vec<FarTileKey> = s.far_shown.iter().copied().collect();
        let rect = |k: &FarTileKey| {
            let size = s.far_size(k.level);
            let min = Vec2::new(k.x as f32, k.z as f32) * size;
            (min, min + Vec2::splat(size))
        };
        for (i, a) in shown.iter().enumerate() {
            for b in &shown[i + 1..] {
                let (a0, a1) = rect(a);
                let (b0, b1) = rect(b);
                let overlap = a0.cmplt(b1).all() && b0.cmplt(a1).all();
                assert!(!overlap, "{a:?} overlaps {b:?}");
            }
        }
        // The camera's own ground is covered by base-size tiles.
        let f = Vec2::new(300.0, -40.0);
        assert!(shown.iter().any(|k| {
            let (a, b) = rect(k);
            k.level == 0 && f.cmpge(a).all() && f.cmplt(b).all()
        }));
    }

    #[test]
    fn edits_survive_unloading_and_come_back() {
        let mut world = World::new();
        look(&mut world, 0.0, 0.0);
        world
            .set_stream(Some(WorldStream::new(
                TestWorld {
                    model: None,
                    water: false,
                },
                small(),
            )))
            .unwrap();
        pump(&mut world);
        let hole = Vec3::new(5.0, 3.0, 5.0);
        let ground = world.stream().unwrap().height_at(hole.x, hole.z);
        let probe = Vec3::new(hole.x, ground - 1.0, hole.z);
        assert!(world.voxel_terrain().unwrap().is_solid(probe));
        let changed = world
            .voxel_terrain_mut()
            .unwrap()
            .dig_sphere(Vec3::new(hole.x, ground, hole.z), 3.0);
        assert!(changed > 0);
        assert!(!world.voxel_terrain().unwrap().is_solid(probe));

        look(&mut world, 3000.0, 0.0);
        pump(&mut world);
        let s = world.stream().unwrap();
        assert!(s.stats().stored_edits > 0, "the dug chunks were kept");
        // Out of range, the volume falls back to the untouched generator.
        assert!(world.voxel_terrain().unwrap().is_solid(probe));

        look(&mut world, 0.0, 0.0);
        pump(&mut world);
        assert!(
            !world.voxel_terrain().unwrap().is_solid(probe),
            "the hole is back"
        );
        assert_eq!(world.stream().unwrap().stats().stored_edits, 0);
    }

    #[test]
    fn saved_games_keep_edits_of_unloaded_ground() {
        let gen = || TestWorld {
            model: None,
            water: false,
        };
        let mut world = World::new();
        look(&mut world, 0.0, 0.0);
        world
            .set_stream(Some(WorldStream::new(gen(), small())))
            .unwrap();
        pump(&mut world);
        let ground = world.stream().unwrap().height_at(5.0, 5.0);
        let probe = Vec3::new(5.0, ground - 1.0, 5.0);
        world
            .voxel_terrain_mut()
            .unwrap()
            .dig_sphere(Vec3::new(5.0, ground, 5.0), 3.0);
        // Walk away so the dug chunks sit in the stream's store, then save.
        look(&mut world, 3000.0, 0.0);
        pump(&mut world);
        assert!(world.stream().unwrap().stats().stored_edits > 0);
        let bytes = world.save_to_bytes(&()).unwrap();

        let mut fresh = World::new();
        look(&mut fresh, 3000.0, 0.0);
        fresh
            .set_stream(Some(WorldStream::new(gen(), small())))
            .unwrap();
        pump(&mut fresh);
        fresh.load_from_bytes(&bytes).unwrap();
        look(&mut fresh, 0.0, 0.0);
        pump(&mut fresh);
        assert!(
            !fresh.voxel_terrain().unwrap().is_solid(probe),
            "the hole came back from the save"
        );
    }

    #[test]
    fn tiles_place_models_and_take_them_away() {
        let mut world = World::new();
        let cube = Shape::box_at((0.0, 0.5, 0.0), (1.0, 1.0, 1.0), rgb(200, 100, 50)).unwrap();
        let model = world
            .spawn_lod(vec![crate::lod::LodLevel::new(cube, 500.0)])
            .unwrap();
        look(&mut world, 0.0, 0.0);
        let gen = TestWorld {
            model: Some(model),
            water: false,
        };
        world
            .set_stream(Some(WorldStream::new(gen, small())))
            .unwrap();
        pump(&mut world);
        // Install the remaining tiles (the prepare waits for ground only).
        for _ in 0..400 {
            world.tick_stream();
            let s = world.stream().unwrap().stats();
            if s.jobs == 0 && s.waiting == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let s = world.stream().unwrap().stats();
        // Tiles within 200 m of the camera: about pi * (200 + 32)^2 / 64^2.
        assert!(s.tiles >= 30, "tiles {}", s.tiles);
        assert_eq!(s.tile_things, s.tiles);
        let levels = world.lod_levels(model).unwrap();
        // Every like-set draws the template's meshes.
        let likes = world
            .entities()
            .filter(|(_, e)| e.instance_of == Some(levels[0]))
            .count();
        assert_eq!(likes, s.tiles);

        look(&mut world, 5000.0, 0.0);
        for _ in 0..40 {
            world.tick_stream();
        }
        let likes = world
            .entities()
            .filter(|(_, e)| e.instance_of == Some(levels[0]))
            .count();
        assert!(likes < s.tiles, "far tiles unloaded");
    }

    #[test]
    fn water_window_fills_the_sea_and_follows() {
        let mut world = World::new();
        look(&mut world, 0.0, 0.0);
        let gen = TestWorld {
            model: None,
            water: true,
        };
        world
            .set_stream(Some(WorldStream::new(gen, small())))
            .unwrap();
        pump(&mut world);
        for _ in 0..400 {
            world.tick_stream();
            if world.water().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let water = world.water().expect("water window");
        assert!(water.total_volume() > 0.0, "the low ground is flooded");
        let first = water.region().min_xz;
        look(&mut world, 200.0, 0.0);
        for _ in 0..400 {
            world.tick_stream();
            if world.water().is_some_and(|w| w.region().min_xz != first) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_ne!(
            world.water().unwrap().region().min_xz,
            first,
            "the window moved"
        );
    }

    #[test]
    fn volume_cache_keeps_edits_only() {
        struct Flat;
        impl VolumeSource for Flat {
            fn density(&self, p: Vec3) -> f32 {
                -p.y
            }
        }
        let mut v = Volume::new(1.0).with_source(Arc::new(Flat));
        let n = ((CHUNK_SIZE + 1) as usize).pow(3);
        let mut samples = Vec::new();
        Flat.fill(
            Vec3::new(0.0, -16.0, 0.0),
            1.0,
            (CHUNK_SIZE + 1) as usize,
            &mut samples,
        );
        assert_eq!(samples.len(), n);
        let rev = v.rev();
        let layout = v.layout_rev();
        assert!(v.insert_generated(IVec3::new(0, -1, 0), samples.clone()));
        assert_eq!(v.rev(), rev, "a cached copy is not a change");
        assert!(v.layout_rev() > layout);
        assert!(!v.is_edited(IVec3::new(0, -1, 0)));
        assert!(v.evict(IVec3::new(0, -1, 0)).is_none());

        v.insert_generated(IVec3::new(0, -1, 0), samples);
        v.dig_sphere(Vec3::new(10.0, 0.0, 10.0), 2.0);
        assert!(v.is_edited(IVec3::new(0, -1, 0)));
        let edit = v.evict(IVec3::new(0, -1, 0)).expect("kept");
        assert_eq!(v.removed_since(rev).unwrap(), vec![IVec3::new(0, -1, 0)]);
        assert!(
            v.is_solid(Vec3::new(10.0, -1.0, 10.0)),
            "back to the source"
        );
        v.restore_edited(edit);
        assert!(!v.is_solid(Vec3::new(10.0, -1.0, 10.0)), "the edit is back");
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::color::rgb;
    use crate::mesh::Shape;

    struct Rolling {
        models: Vec<LodId>,
    }

    impl WorldGenerator for Rolling {
        fn height(&self, x: f32, z: f32) -> f32 {
            (x * 0.004).sin() * 30.0 + (z * 0.003).cos() * 25.0 + (x * 0.03 + z * 0.02).sin() * 3.0
        }

        fn populate(&self, tile: &mut Tile) -> EngineResult<()> {
            let (min, max) = (tile.min(), tile.max());
            let places = crate::scatter::Scatter::new(tile.seed())
                .rect(min.x, min.y, max.x, max.y)
                .spacing(9.0)
                .on_ground(|x, z| self.height(x, z))
                .places()?;
            for (i, p) in places.into_iter().enumerate() {
                tile.place(self.models[i % self.models.len()], p);
            }
            Ok(())
        }
    }

    /// World-side cost of flying over a streamed world (no renderer):
    /// `cargo test --release -p engine --lib stream::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn flight_costs() {
        let mut world = World::new();
        let mut models = Vec::new();
        for k in 0..12 {
            let cube = Shape::box_at(
                (0.0, 0.5, 0.0),
                (1.0, 1.0 + k as f32 * 0.1, 1.0),
                rgb(90, 140, 60),
            )
            .unwrap();
            models.push(
                world
                    .spawn_lod(vec![
                        crate::lod::LodLevel::new(cube.clone(), 60.0),
                        crate::lod::LodLevel::new(cube.clone(), 200.0),
                        crate::lod::LodLevel::new(cube, 1100.0),
                    ])
                    .unwrap(),
            );
        }
        world.look_first_person((0.0, 40.0, 0.0), 90.0, 0.0);
        world
            .set_stream(Some(WorldStream::new(
                Rolling { models },
                StreamSettings::default(),
            )))
            .unwrap();
        let t = Instant::now();
        assert!(world.prepare_stream(60.0));
        println!(
            "prepare {:.0} ms {:?}",
            t.elapsed().as_secs_f32() * 1000.0,
            world.stream().unwrap().stats()
        );
        for speed in [15.0f32, 120.0, 400.0] {
            let mut x = world.camera().eye().x;
            let mut stream_ms = Vec::new();
            let mut lod_ms = Vec::new();
            for _ in 0..600 {
                x += speed / 60.0;
                world.look_first_person((x, 40.0, 0.0), 90.0, 0.0);
                let a = Instant::now();
                world.tick_stream();
                let b = Instant::now();
                world.tick_lods();
                let c = Instant::now();
                stream_ms.push((b - a).as_secs_f32() * 1000.0);
                lod_ms.push((c - b).as_secs_f32() * 1000.0);
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            let stat = |v: &mut Vec<f32>| {
                v.sort_by(f32::total_cmp);
                (v[v.len() / 2], v[v.len() * 99 / 100], v[v.len() - 1])
            };
            println!(
                "{speed} m/s: stream p50/p99/max {:?} ms, lod {:?} ms, entities {}, {:?}",
                stat(&mut stream_ms),
                stat(&mut lod_ms),
                world.entity_count(),
                world.stream().unwrap().stats()
            );
        }
    }
}
