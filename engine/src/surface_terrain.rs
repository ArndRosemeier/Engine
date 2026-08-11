//! Chunk meshing from a [`SurfaceSource`] — land + water from one sample.
//!
//! **Land** is a heightfield mesh (its resolution is only for ground LOD).
//! **Water** is a separate isosurface of the continuous wetness field
//! (`water_top - ground >= CLEARANCE`), extracted at [`SurfaceMeshStyle::water_iso_cell`].
//! Water is never “paint this land cell blue” — that always makes rectangular shores.

use crate::color::{rgb, Color};
use crate::mesh::{BuiltMesh, Mesh, PointId};
use crate::surface::{SharedSurface, SurfaceSample, SurfaceSource, WATER_CLEARANCE};
use crate::world::World;
use glam::{IVec3, Vec2, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// Visual knobs for [`SurfaceTerrain`] meshes (sampling comes from [`SurfaceSource`]).
#[derive(Clone, Debug)]
pub struct SurfaceMeshStyle {
    pub chunk_cells: i32,
    pub cell_size: f32,
    /// Independent sample spacing for the water wetness isosurface (metres).
    /// Decoupled from [`Self::cell_size`] on purpose.
    pub water_iso_cell: f32,
    pub grass: Color,
    pub sand: Color,
    pub rock: Color,
    pub water: Color,
    pub bed: Color,
    /// Heights above this (approx) tint as rock.
    pub rock_height: f32,
}

impl Default for SurfaceMeshStyle {
    fn default() -> Self {
        Self {
            chunk_cells: 48,
            cell_size: 4.0,
            water_iso_cell: 1.0,
            grass: rgb(92, 140, 70),
            sand: rgb(194, 178, 128),
            rock: rgb(120, 118, 112),
            water: crate::color::rgba(40, 120, 175, 90),
            bed: rgb(110, 125, 95),
            rock_height: 400.0,
        }
    }
}

/// Builds land/water chunk meshes from a pluggable surface sample.
#[derive(Clone)]
pub struct SurfaceTerrain {
    source: SharedSurface,
    style: SurfaceMeshStyle,
}

impl SurfaceTerrain {
    pub fn new(source: SharedSurface, style: SurfaceMeshStyle) -> Self {
        Self { source, style }
    }

    pub fn source(&self) -> &dyn SurfaceSource {
        self.source.as_ref()
    }

    pub fn style(&self) -> &SurfaceMeshStyle {
        &self.style
    }

    pub fn sample(&self, x: f32, z: f32) -> SurfaceSample {
        self.source.sample(x, z)
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z).walk_height()
    }

    pub fn chunk_key_for(&self, x: f32, z: f32) -> IVec3 {
        let cells = self.style.chunk_cells as f32;
        let cs = self.style.cell_size;
        let span = cells * cs;
        IVec3::new((x / span).floor() as i32, 0, (z / span).floor() as i32)
    }

    pub fn build_chunk(&self, cx: i32, cz: i32) -> Mesh {
        let s = &self.style;
        let cells = s.chunk_cells;
        let cs = s.cell_size;
        let origin_x = cx as f32 * cells as f32 * cs;
        let origin_z = cz as f32 * cells as f32 * cs;
        let span = cells as f32 * cs;
        let verts = (cells + 1) as usize;

        let samples: Vec<SurfaceSample> = (0..verts * verts)
            .into_par_iter()
            .map(|i| {
                let ix = (i % verts) as i32;
                let iz = (i / verts) as i32;
                let x = origin_x + ix as f32 * cs;
                let z = origin_z + iz as f32 * cs;
                self.source.sample(x, z)
            })
            .collect();

        let mut mesh = Mesh::new();
        let mut ids = Vec::with_capacity(verts * verts);

        for (i, sample) in samples.iter().enumerate() {
            let ix = (i % verts) as i32;
            let iz = (i / verts) as i32;
            let x = origin_x + ix as f32 * cs;
            let z = origin_z + iz as f32 * cs;
            let id = mesh
                .add_point(Vec3::new(x, sample.ground, z))
                .expect("terrain point");
            let color = if sample.is_wet() {
                s.bed
            } else if sample.water_top.is_finite() && sample.ground < sample.water_top + 1.0
            {
                s.sand
            } else if sample.ground > s.rock_height {
                s.rock
            } else {
                s.grass
            };
            mesh.set_point_color(id, color).expect("terrain color");
            ids.push(id);
        }

        for iz in 0..cells {
            for ix in 0..cells {
                let i00 = (iz * (cells + 1) + ix) as usize;
                let i10 = i00 + 1;
                let i01 = i00 + (cells as usize + 1);
                let i11 = i01 + 1;
                mesh.add_quad(ids[i00], ids[i01], ids[i11], ids[i10])
                    .expect("terrain quad");
            }
        }

        // Water: continuous wetness isosurface — not the land grid.
        append_water_isosurface(
            &mut mesh,
            self.source.as_ref(),
            origin_x,
            origin_z,
            span,
            s.water_iso_cell.max(0.25),
            s.water,
        );

        mesh
    }

    pub fn build_chunk_built(&self, cx: i32, cz: i32) -> BuiltMesh {
        self.build_chunk(cx, cz).build()
    }
}

/// Signed wetness: >= 0 means wet (same predicate as [`SurfaceSample::is_wet`]).
fn wet_excess(s: SurfaceSample) -> f32 {
    if !s.water_top.is_finite() {
        return -1.0;
    }
    (s.water_top - s.ground) - WATER_CLEARANCE
}

fn edge_crossing_xz(a: Vec2, ea: f32, b: Vec2, eb: f32) -> (Vec2, f32) {
    let denom = ea - eb;
    let t = if denom.abs() < 1e-8 {
        0.5
    } else {
        (ea / denom).clamp(0.0, 1.0)
    };
    (a + (b - a) * t, t)
}

/// Extract water as the zero-contour of wetness over an independent fine lattice.
fn append_water_isosurface(
    mesh: &mut Mesh,
    source: &dyn SurfaceSource,
    origin_x: f32,
    origin_z: f32,
    span: f32,
    iso_cell: f32,
    water_color: Color,
) {
    let n = ((span / iso_cell).ceil() as i32).max(2);
    let step = span / n as f32;
    let verts = (n + 1) as usize;

    // Sample continuous field on the iso lattice (parallel).
    let field: Vec<(f32, f32)> = (0..verts * verts)
        .into_par_iter()
        .map(|i| {
            let ix = (i % verts) as i32;
            let iz = (i / verts) as i32;
            let x = origin_x + ix as f32 * step;
            let z = origin_z + iz as f32 * step;
            let s = source.sample(x, z);
            let e = wet_excess(s);
            let sheet = if s.water_top.is_finite() {
                s.water_top
            } else {
                0.0
            };
            (e, sheet)
        })
        .collect();

    for iz in 0..n {
        for ix in 0..n {
            let i00 = (iz * (n + 1) + ix) as usize;
            let i10 = i00 + 1;
            let i01 = i00 + (n as usize + 1);
            let i11 = i01 + 1;
            let x0 = origin_x + ix as f32 * step;
            let z0 = origin_z + iz as f32 * step;
            let x1 = x0 + step;
            let z1 = z0 + step;

            let corners = [
                (Vec2::new(x0, z0), field[i00].0, field[i00].1),
                (Vec2::new(x1, z0), field[i10].0, field[i10].1),
                (Vec2::new(x1, z1), field[i11].0, field[i11].1),
                (Vec2::new(x0, z1), field[i01].0, field[i01].1),
            ];
            emit_water_iso_cell(mesh, water_color, corners);
        }
    }
}

/// Max vertical jump (metres) allowed across one water iso cell. Larger deltas
/// are body seams (e.g. highland river vs sea) — bridging them draws sky walls.
const MAX_WATER_SHEET_DELTA: f32 = 2.5;

/// Marching-squares on wetness; vertex Y from interpolated `water_top`.
fn emit_water_iso_cell(
    mesh: &mut Mesh,
    water_color: Color,
    corners: [(Vec2, f32, f32); 4],
) {
    let excess = [corners[0].1, corners[1].1, corners[2].1, corners[3].1];
    if !excess.iter().any(|e| *e >= 0.0) {
        return;
    }

    let mut sheet_min = f32::INFINITY;
    let mut sheet_max = f32::NEG_INFINITY;
    for i in 0..4 {
        if excess[i] >= 0.0 {
            sheet_min = sheet_min.min(corners[i].2);
            sheet_max = sheet_max.max(corners[i].2);
        }
    }
    if sheet_max - sheet_min > MAX_WATER_SHEET_DELTA {
        // Do not bridge disconnected vertical sheets.
        return;
    }

    // Polygon of (xz, sheet_z) along the wet side of the contour.
    let mut poly: Vec<(Vec2, f32)> = Vec::with_capacity(6);
    for i in 0..4 {
        let j = (i + 1) % 4;
        let (pi, ei, si) = corners[i];
        let (pj, ej, sj) = corners[j];
        if ei >= 0.0 {
            poly.push((pi, si));
        }
        if (ei >= 0.0) != (ej >= 0.0) {
            let (xz, t) = edge_crossing_xz(pi, ei, pj, ej);
            let sheet = si + (sj - si) * t;
            poly.push((xz, sheet));
        }
    }
    if poly.len() < 3 {
        return;
    }

    let mut ids: Vec<PointId> = Vec::with_capacity(poly.len());
    for (p, sheet) in &poly {
        let y = *sheet + WATER_CLEARANCE * 0.5;
        let id = mesh
            .add_point(Vec3::new(p.x, y, p.y))
            .expect("water point");
        mesh.set_point_color(id, water_color).expect("water color");
        ids.push(id);
    }
    for k in 1..ids.len() - 1 {
        mesh.add_triangle(ids[0], ids[k], ids[k + 1])
            .expect("water tri");
    }
}

type ChunkKey = (i32, i32);

/// Streams [`SurfaceTerrain`] chunks around a focus (same budgets as heightfield stream).
pub struct SurfaceStream {
    terrain: Arc<SurfaceTerrain>,
    pub radius: i32,
    pub max_jobs_per_frame: usize,
    pub max_uploads_per_frame: usize,
    loaded: HashMap<ChunkKey, ()>,
    inflight: HashSet<ChunkKey>,
    ready_queue: VecDeque<(ChunkKey, BuiltMesh)>,
    tx: Sender<(ChunkKey, BuiltMesh)>,
    rx: Receiver<(ChunkKey, BuiltMesh)>,
}

impl SurfaceStream {
    pub fn new(terrain: SurfaceTerrain, radius: i32) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            terrain: Arc::new(terrain),
            radius: radius.max(1),
            max_jobs_per_frame: 6,
            max_uploads_per_frame: 2,
            loaded: HashMap::new(),
            inflight: HashSet::new(),
            ready_queue: VecDeque::new(),
            tx,
            rx,
        }
    }

    pub fn with_budgets(mut self, jobs: usize, uploads: usize) -> Self {
        self.max_jobs_per_frame = jobs.max(1);
        self.max_uploads_per_frame = uploads.max(1);
        self
    }

    pub fn terrain(&self) -> &SurfaceTerrain {
        &self.terrain
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.terrain.height_at(x, z)
    }

    pub fn sample(&self, x: f32, z: f32) -> SurfaceSample {
        self.terrain.sample(x, z)
    }

    fn desired_ring(&self, focus: Vec3) -> (ChunkKey, HashSet<ChunkKey>) {
        let center = self.terrain.chunk_key_for(focus.x, focus.z);
        let center_key = (center.x, center.z);
        let r = self.radius;
        let mut desired = HashSet::new();
        for dz in -r..=r {
            for dx in -r..=r {
                desired.insert((center.x + dx, center.z + dz));
            }
        }
        (center_key, desired)
    }

    fn drain_ready(&mut self) {
        while let Ok(item) = self.rx.try_recv() {
            self.inflight.remove(&item.0);
            self.ready_queue.push_back(item);
        }
    }

    fn upload_ready(&mut self, world: &mut World, focus: Vec3, desired: &HashSet<ChunkKey>) {
        let center = self.terrain.chunk_key_for(focus.x, focus.z);
        let mut batch: Vec<(ChunkKey, BuiltMesh)> = self.ready_queue.drain(..).collect();
        batch.sort_by_key(|(k, _)| (k.0 - center.x).abs() + (k.1 - center.z).abs());
        let mut uploaded = 0;
        let mut rest = VecDeque::new();
        for (key, built) in batch {
            if !desired.contains(&key) || self.loaded.contains_key(&key) {
                continue;
            }
            if uploaded < self.max_uploads_per_frame {
                world.set_chunk_built(IVec3::new(key.0, 0, key.1), built);
                self.loaded.insert(key, ());
                uploaded += 1;
            } else {
                rest.push_back((key, built));
            }
        }
        self.ready_queue = rest;
    }

    fn spawn_jobs(&mut self, focus: Vec3, desired: &HashSet<ChunkKey>) {
        let center = self.terrain.chunk_key_for(focus.x, focus.z);
        let mut missing: Vec<ChunkKey> = desired
            .iter()
            .copied()
            .filter(|k| !self.loaded.contains_key(k) && !self.inflight.contains(k))
            .collect();
        missing.sort_by_key(|k| (k.0 - center.x).abs() + (k.1 - center.z).abs());
        let budget = self.max_jobs_per_frame.min(missing.len());
        for key in missing.into_iter().take(budget) {
            self.inflight.insert(key);
            let terrain = Arc::clone(&self.terrain);
            let tx = self.tx.clone();
            rayon::spawn(move || {
                let built = terrain.build_chunk_built(key.0, key.1);
                let _ = tx.send((key, built));
            });
        }
    }

    fn ensure_focus_chunk(&mut self, world: &mut World, center_key: ChunkKey) {
        if self.loaded.contains_key(&center_key) {
            return;
        }
        self.inflight.remove(&center_key);
        let built = self.terrain.build_chunk_built(center_key.0, center_key.1);
        world.set_chunk_built(IVec3::new(center_key.0, 0, center_key.1), built);
        self.loaded.insert(center_key, ());
    }

    pub fn sync(&mut self, world: &mut World, focus: Vec3) {
        let (center_key, desired) = self.desired_ring(focus);
        let stale: Vec<ChunkKey> = self
            .loaded
            .keys()
            .copied()
            .filter(|k| !desired.contains(k))
            .collect();
        for key in stale {
            world.clear_chunk(IVec3::new(key.0, 0, key.1));
            self.loaded.remove(&key);
        }
        self.drain_ready();
        self.ensure_focus_chunk(world, center_key);
        self.upload_ready(world, focus, &desired);
        self.spawn_jobs(focus, &desired);
    }

    pub fn sync_blocking(&mut self, world: &mut World, focus: Vec3) {
        let (_center_key, desired) = self.desired_ring(focus);
        let stale: Vec<ChunkKey> = self
            .loaded
            .keys()
            .copied()
            .filter(|k| !desired.contains(k))
            .collect();
        for key in stale {
            world.clear_chunk(IVec3::new(key.0, 0, key.1));
            self.loaded.remove(&key);
        }
        self.drain_ready();
        self.ready_queue.clear();
        self.inflight.clear();
        let mut keys: Vec<ChunkKey> = desired.into_iter().collect();
        keys.sort_by_key(|k| (k.0, k.1));
        let built: Vec<(ChunkKey, BuiltMesh)> = keys
            .par_iter()
            .map(|key| (*key, self.terrain.build_chunk_built(key.0, key.1)))
            .collect();
        for (key, mesh) in built {
            if self.loaded.contains_key(&key) {
                continue;
            }
            world.set_chunk_built(IVec3::new(key.0, 0, key.1), mesh);
            self.loaded.insert(key, ());
        }
    }
}
