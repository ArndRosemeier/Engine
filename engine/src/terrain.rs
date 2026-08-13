//! Infinite heightfield terrain (hills + lakes). No caves — use [`crate::volume`] for those.

use crate::color::{rgb, rgba, Color};
use crate::mesh::{BuiltMesh, Mesh};
use crate::proc::Noise;
use crate::surface::WATER_CLEARANCE;
use crate::world::World;
use glam::{IVec3, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// Surface sample at a world XZ position.
#[derive(Clone, Copy, Debug)]
pub struct TerrainSample {
    /// Walkable / mesh surface height.
    pub height: f32,
    /// Solid ground under the surface (below the waterline inside lakes).
    pub ground: f32,
    /// Sheet height when this column holds water.
    pub water_top: Option<f32>,
    pub water: bool,
}

/// Tunables for rolling hills with occasional lakes.
#[derive(Clone, Debug)]
pub struct TerrainRules {
    pub seed: u32,
    /// Cells along one chunk edge (quads = cells, verts = cells+1).
    pub chunk_cells: i32,
    pub cell_size: f32,
    pub base_height: f32,
    pub hill_height: f32,
    pub hill_scale: f32,
    pub lake_scale: f32,
    /// Higher → fewer lakes (roughly 0.55..=0.85).
    pub lake_threshold: f32,
    pub water_level: f32,
    pub grass: Color,
    pub sand: Color,
    pub water: Color,
    pub rock: Color,
}

impl Default for TerrainRules {
    fn default() -> Self {
        Self {
            seed: 42,
            chunk_cells: 32,
            cell_size: 1.0,
            base_height: 6.0,
            hill_height: 14.0,
            hill_scale: 0.018,
            lake_scale: 0.012,
            lake_threshold: 0.68,
            water_level: 5.5,
            grass: rgb(92, 140, 70),
            sand: rgb(194, 178, 128),
            // Translucent — engine draws alpha < 1 in the transparent pass.
            water: rgba(40, 120, 175, 90),
            rock: rgb(120, 118, 112),
        }
    }
}

/// Deterministic heightfield sampler (infinite in XZ).
#[derive(Clone, Debug)]
pub struct HeightTerrain {
    rules: TerrainRules,
    hills: Noise,
    lakes: Noise,
}

impl HeightTerrain {
    pub fn new(rules: TerrainRules) -> Self {
        let hills = Noise::new(rules.seed);
        let lakes = Noise::new(rules.seed ^ 0xC0FFEE);
        Self {
            rules,
            hills,
            lakes,
        }
    }

    pub fn rules(&self) -> &TerrainRules {
        &self.rules
    }

    pub fn sample(&self, x: f32, z: f32) -> TerrainSample {
        let r = &self.rules;
        let n = self.hills.fbm3(
            Vec3::new(x * r.hill_scale, 0.0, z * r.hill_scale),
            5,
            2.1,
            0.5,
        );
        let h_raw = r.base_height + r.hill_height * n;

        let lake = self.lakes.fbm3(
            Vec3::new(x * r.lake_scale, 3.0, z * r.lake_scale),
            3,
            2.0,
            0.55,
        );
        let lake_t = lake * 0.5 + 0.5;
        let span = (1.0 - r.lake_threshold).max(1e-3);
        let basin = ((lake_t - r.lake_threshold) / span).clamp(0.0, 1.0);

        // Raise valleys to the waterline so dry land never undercuts water (mesa cause).
        // Carve lake bowls through that floor; the mesh surface stays at the waterline
        // over wet cells so shores meet flush (no separate overhanging water deck).
        // Plains/valleys sit on a waterline floor; hills keep their height.
        let floor = h_raw.max(r.water_level);
        let carved = floor - basin * 3.5;
        // Lakes only on the coastal shelf (floor near waterline).
        let near_shore = floor <= r.water_level + 1.5;
        let in_basin = basin > 0.25 && near_shore;
        let (ground, water_top) = if in_basin {
            let water_top = r.water_level;
            // Epsilon so float rounding still satisfies the clearance contract.
            let ground = carved.min(water_top - WATER_CLEARANCE - 1e-3);
            (ground, Some(water_top))
        } else {
            (floor, None)
        };
        // Soft apron: land in the lake field eases down to the waterline so
        // neighboring hill verts don't spike through the water surface.
        let height = if let Some(top) = water_top {
            top
        } else if basin > 0.0 && near_shore {
            let t = (basin / 0.25).clamp(0.0, 1.0);
            let t = t * t * (3.0 - 2.0 * t);
            floor * (1.0 - t) + r.water_level * t
        } else {
            ground
        };

        TerrainSample {
            height,
            ground,
            water_top,
            water: water_top.is_some(),
        }
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z).height
    }

    /// Build a chunk mesh: opaque land at ground height + translucent water surface.
    pub fn build_chunk(&self, cx: i32, cz: i32) -> Mesh {
        let r = &self.rules;
        let cells = r.chunk_cells;
        let cs = r.cell_size;
        let origin_x = cx as f32 * cells as f32 * cs;
        let origin_z = cz as f32 * cells as f32 * cs;
        let verts = (cells + 1) as usize;

        // Sample the heightfield in parallel — this is the hot path for dense chunks.
        let samples: Vec<TerrainSample> = (0..verts * verts)
            .into_par_iter()
            .map(|i| {
                let ix = (i % verts) as i32;
                let iz = (i / verts) as i32;
                let x = origin_x + ix as f32 * cs;
                let z = origin_z + iz as f32 * cs;
                self.sample(x, z)
            })
            .collect();

        let mut mesh = Mesh::new();
        let mut ids = Vec::with_capacity(verts * verts);

        for (i, s) in samples.iter().enumerate() {
            let ix = (i % verts) as i32;
            let iz = (i / verts) as i32;
            let x = origin_x + ix as f32 * cs;
            let z = origin_z + iz as f32 * cs;
            // Land always uses ground so lake beds show through translucent water.
            let id = mesh
                .add_point(Vec3::new(x, s.ground, z))
                .expect("terrain point");
            let color = if s.ground < r.water_level - 0.15 {
                // Darker bed so translucent water reads clearly.
                rgb(110, 125, 95)
            } else if s.ground < r.water_level + 1.0 {
                r.sand
            } else if s.ground > r.base_height + r.hill_height * 0.55 {
                r.rock
            } else {
                r.grass
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
                // Outward (+Y) when viewed from above.
                mesh.add_quad(ids[i00], ids[i01], ids[i11], ids[i10])
                    .expect("terrain quad");

                let corners = [samples[i00], samples[i10], samples[i01], samples[i11]];
                let Some(mut water_y) = corners
                    .iter()
                    .filter_map(|c| c.water_top)
                    .fold(None, |acc: Option<f32>, top| {
                        Some(acc.map_or(top, |a| a.max(top)))
                    })
                else {
                    continue;
                };
                water_y += WATER_CLEARANCE * 0.5;
                let x0 = origin_x + ix as f32 * cs;
                let z0 = origin_z + iz as f32 * cs;
                let x1 = x0 + cs;
                let z1 = z0 + cs;
                let w00 = mesh.add_point(Vec3::new(x0, water_y, z0)).expect("water");
                let w10 = mesh.add_point(Vec3::new(x1, water_y, z0)).expect("water");
                let w01 = mesh.add_point(Vec3::new(x0, water_y, z1)).expect("water");
                let w11 = mesh.add_point(Vec3::new(x1, water_y, z1)).expect("water");
                for id in [w00, w10, w01, w11] {
                    mesh.set_point_color(id, r.water).expect("water color");
                }
                mesh.add_quad(w00, w01, w11, w10).expect("water quad");
            }
        }

        mesh
    }

    /// Build + triangulate on a worker-friendly path.
    pub fn build_chunk_built(&self, cx: i32, cz: i32) -> BuiltMesh {
        self.build_chunk(cx, cz).build_smooth()
    }

    pub fn chunk_key_for(&self, x: f32, z: f32) -> IVec3 {
        let cells = self.rules.chunk_cells as f32;
        let cs = self.rules.cell_size;
        let span = cells * cs;
        IVec3::new((x / span).floor() as i32, 0, (z / span).floor() as i32)
    }
}

type ChunkKey = (i32, i32);

/// Keeps a ring of heightfield chunks loaded around a focus point.
///
/// Generation runs on a background pool (`rayon`); each [`sync`] call only
/// uploads a small budget of finished meshes so the frame thread never stalls
/// on a full ring rebuild.
pub struct TerrainStream {
    terrain: Arc<HeightTerrain>,
    /// Chunk radius (Chebyshev) to keep loaded.
    pub radius: i32,
    /// Max background jobs to start per [`sync`] call.
    pub max_jobs_per_frame: usize,
    /// Max finished chunks to upload to the GPU per [`sync`] call.
    pub max_uploads_per_frame: usize,
    loaded: HashMap<ChunkKey, ()>,
    inflight: HashSet<ChunkKey>,
    ready_queue: VecDeque<(ChunkKey, BuiltMesh)>,
    tx: Sender<(ChunkKey, BuiltMesh)>,
    rx: Receiver<(ChunkKey, BuiltMesh)>,
}

impl std::fmt::Debug for TerrainStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerrainStream")
            .field("radius", &self.radius)
            .field("loaded", &self.loaded.len())
            .field("inflight", &self.inflight.len())
            .field("ready_queue", &self.ready_queue.len())
            .field("max_jobs_per_frame", &self.max_jobs_per_frame)
            .field("max_uploads_per_frame", &self.max_uploads_per_frame)
            .finish()
    }
}

impl TerrainStream {
    pub fn new(rules: TerrainRules, radius: i32) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            terrain: Arc::new(HeightTerrain::new(rules)),
            radius: radius.max(1),
            // Keep workers busy without flooding the GPU upload path.
            max_jobs_per_frame: 6,
            max_uploads_per_frame: 2,
            loaded: HashMap::new(),
            inflight: HashSet::new(),
            ready_queue: VecDeque::new(),
            tx,
            rx,
        }
    }

    pub fn with_budgets(mut self, jobs_per_frame: usize, uploads_per_frame: usize) -> Self {
        self.max_jobs_per_frame = jobs_per_frame.max(1);
        self.max_uploads_per_frame = uploads_per_frame.max(1);
        self
    }

    pub fn terrain(&self) -> &HeightTerrain {
        &self.terrain
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.terrain.height_at(x, z)
    }

    pub fn sample(&self, x: f32, z: f32) -> TerrainSample {
        self.terrain.sample(x, z)
    }

    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn pending_count(&self) -> usize {
        self.inflight.len() + self.ready_queue.len()
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

    /// Ensure the focus chunk exists immediately (one chunk, main thread).
    fn ensure_focus_chunk(&mut self, world: &mut World, center_key: ChunkKey) {
        if self.loaded.contains_key(&center_key) {
            return;
        }
        self.inflight.remove(&center_key);
        let built = self.terrain.build_chunk_built(center_key.0, center_key.1);
        world.set_chunk_built(IVec3::new(center_key.0, 0, center_key.1), built);
        self.loaded.insert(center_key, ());
    }

    /// Stream chunks around `focus`: background build, budgeted GPU upload.
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

    /// Build the whole ring on the calling thread (tests / screenshots).
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

        // Drop any async leftovers — we're filling synchronously.
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
