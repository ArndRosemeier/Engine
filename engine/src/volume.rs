use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::marching_cubes::{triangulate_cell, CornerValues};
use crate::mesh::{BuiltMesh, Mesh};
use crate::place::{ensure_finite, ensure_finite3};
use glam::{IVec3, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Number of voxels along one axis of a chunk (exclusive upper bound of cell grid).
pub const CHUNK_SIZE: i32 = 32;

#[derive(Clone, Debug)]
struct Chunk {
    /// Density samples at cell corners: (CHUNK_SIZE+1)^3
    /// Positive = solid, negative = empty. Surface at 0.
    samples: Vec<f32>,
    dirty: bool,
}

impl Chunk {
    fn new() -> Self {
        let n = (CHUNK_SIZE + 1) as usize;
        Self {
            samples: vec![-1.0; n * n * n],
            dirty: true,
        }
    }

    fn index(x: i32, y: i32, z: i32) -> usize {
        let n = CHUNK_SIZE + 1;
        debug_assert!((0..n).contains(&x) && (0..n).contains(&y) && (0..n).contains(&z));
        (y * n * n + z * n + x) as usize
    }

    fn get(&self, x: i32, y: i32, z: i32) -> f32 {
        self.samples[Self::index(x, y, z)]
    }

    fn set(&mut self, x: i32, y: i32, z: i32, value: f32) {
        self.samples[Self::index(x, y, z)] = value;
        self.dirty = true;
    }
}

/// Chunked scalar volume for caves, overhangs, and landscapes.
#[derive(Clone, Debug)]
pub struct Volume {
    pub voxel_size: f32,
    chunks: HashMap<IVec3, Chunk>,
}

impl Default for Volume {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Volume {
    pub fn new(voxel_size: f32) -> Self {
        Self::try_new(voxel_size).expect("voxel_size must be finite and > 0")
    }

    pub fn try_new(voxel_size: f32) -> EngineResult<Self> {
        ensure_finite(voxel_size, "voxel_size")?;
        if voxel_size <= 0.0 {
            return Err(EngineError::InvalidValue(
                "voxel_size must be > 0".into(),
            ));
        }
        Ok(Self {
            voxel_size,
            chunks: HashMap::new(),
        })
    }

    fn estimate_samples(min: Vec3, max: Vec3, step: f32) -> u64 {
        let nx = ((max.x - min.x) / step).floor().max(0.0) as u64 + 1;
        let ny = ((max.y - min.y) / step).floor().max(0.0) as u64 + 1;
        let nz = ((max.z - min.z) / step).floor().max(0.0) as u64 + 1;
        nx.saturating_mul(ny).saturating_mul(nz)
    }

    fn check_paint_budget(
        &self,
        min: Vec3,
        max: Vec3,
        limits: &EngineLimits,
    ) -> EngineResult<()> {
        ensure_finite3(min, "paint min")?;
        ensure_finite3(max, "paint max")?;
        let samples = Self::estimate_samples(min, max, self.voxel_size);
        if samples > limits.max_volume_samples_per_paint {
            return Err(EngineError::ResourceLimit(format!(
                "volume paint would touch {samples} samples (limit {})",
                limits.max_volume_samples_per_paint
            )));
        }
        Ok(())
    }

    pub fn chunk_keys(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.chunks.keys().copied()
    }

    pub fn has_chunk(&self, key: IVec3) -> bool {
        self.chunks.contains_key(&key)
    }

    pub fn remove_chunk_data(&mut self, key: IVec3) {
        self.chunks.remove(&key);
    }

    pub fn dirty_chunk_keys(&self) -> Vec<IVec3> {
        self.chunks
            .iter()
            .filter(|(_, c)| c.dirty)
            .map(|(k, _)| *k)
            .collect()
    }

    fn world_to_grid(&self, p: Vec3) -> (IVec3, IVec3) {
        let g = (p / self.voxel_size).floor();
        let gx = g.x as i32;
        let gy = g.y as i32;
        let gz = g.z as i32;
        let chunk = IVec3::new(
            gx.div_euclid(CHUNK_SIZE),
            gy.div_euclid(CHUNK_SIZE),
            gz.div_euclid(CHUNK_SIZE),
        );
        let local = IVec3::new(
            gx.rem_euclid(CHUNK_SIZE),
            gy.rem_euclid(CHUNK_SIZE),
            gz.rem_euclid(CHUNK_SIZE),
        );
        (chunk, local)
    }

    fn ensure_chunk(&mut self, key: IVec3) -> &mut Chunk {
        self.chunks.entry(key).or_insert_with(Chunk::new)
    }

    /// Set density at a world-space point (snapped to voxel grid).
    pub fn set(&mut self, world: impl Into<Vec3>, density: f32) {
        let (chunk_key, local) = self.world_to_grid(world.into());
        let chunk = self.ensure_chunk(chunk_key);
        chunk.set(local.x, local.y, local.z, density);

        // Duplicate samples on positive faces into neighboring chunks for seamless MC.
        if local.x == 0 {
            let c = self.ensure_chunk(chunk_key - IVec3::X);
            c.set(CHUNK_SIZE, local.y, local.z, density);
        }
        if local.y == 0 {
            let c = self.ensure_chunk(chunk_key - IVec3::Y);
            c.set(local.x, CHUNK_SIZE, local.z, density);
        }
        if local.z == 0 {
            let c = self.ensure_chunk(chunk_key - IVec3::Z);
            c.set(local.x, local.y, CHUNK_SIZE, density);
        }
    }

    pub fn get(&self, world: impl Into<Vec3>) -> f32 {
        let (chunk_key, local) = self.world_to_grid(world.into());
        match self.chunks.get(&chunk_key) {
            Some(c) => c.get(local.x, local.y, local.z),
            None => -1.0,
        }
    }

    /// Fill an axis-aligned box of voxels with a density value.
    pub fn fill_box(&mut self, min: impl Into<Vec3>, max: impl Into<Vec3>, density: f32) {
        let min = min.into();
        let max = max.into();
        let step = self.voxel_size;
        let mut y = min.y;
        while y <= max.y + 1e-5 {
            let mut z = min.z;
            while z <= max.z + 1e-5 {
                let mut x = min.x;
                while x <= max.x + 1e-5 {
                    self.set(Vec3::new(x, y, z), density);
                    x += step;
                }
                z += step;
            }
            y += step;
        }
    }

    /// Carve (set empty) a sphere.
    pub fn carve_sphere(&mut self, center: impl Into<Vec3>, radius: f32) {
        self.paint_sphere(center, radius, -1.0);
    }

    /// Fill a sphere with solid density.
    pub fn fill_sphere(&mut self, center: impl Into<Vec3>, radius: f32) {
        self.paint_sphere(center, radius, 1.0);
    }

    fn paint_sphere(&mut self, center: impl Into<Vec3>, radius: f32, density: f32) {
        let center = center.into();
        let min = center - Vec3::splat(radius);
        let max = center + Vec3::splat(radius);
        let step = self.voxel_size;
        let r2 = radius * radius;
        let mut y = min.y;
        while y <= max.y + 1e-5 {
            let mut z = min.z;
            while z <= max.z + 1e-5 {
                let mut x = min.x;
                while x <= max.x + 1e-5 {
                    let p = Vec3::new(x, y, z);
                    if p.distance_squared(center) <= r2 {
                        self.set(p, density);
                    }
                    x += step;
                }
                z += step;
            }
            y += step;
        }
    }

    /// Apply a density function over a world-space AABB (inclusive).
    ///
    /// Panics if the paint exceeds default [`EngineLimits`]. Prefer [`paint_fn_limited`].
    pub fn paint_fn(
        &mut self,
        min: impl Into<Vec3>,
        max: impl Into<Vec3>,
        f: impl FnMut(Vec3) -> f32,
    ) {
        self.paint_fn_limited(min, max, &EngineLimits::default(), f)
            .expect("volume paint exceeded resource limits");
    }

    /// Apply a density function with an explicit sample budget.
    pub fn paint_fn_limited(
        &mut self,
        min: impl Into<Vec3>,
        max: impl Into<Vec3>,
        limits: &EngineLimits,
        mut f: impl FnMut(Vec3) -> f32,
    ) -> EngineResult<()> {
        let min = min.into();
        let max = max.into();
        self.check_paint_budget(min, max, limits)?;
        let step = self.voxel_size;
        let mut y = min.y;
        while y <= max.y + 1e-5 {
            let mut z = min.z;
            while z <= max.z + 1e-5 {
                let mut x = min.x;
                while x <= max.x + 1e-5 {
                    let p = Vec3::new(x, y, z);
                    let d = f(p);
                    if !d.is_finite() {
                        return Err(EngineError::InvalidValue(
                            "density function returned non-finite value".into(),
                        ));
                    }
                    self.set(p, d);
                    x += step;
                }
                z += step;
            }
            y += step;
        }
        Ok(())
    }

    /// Extract a single chunk to a [`Mesh`].
    pub fn extract_chunk(&self, chunk_key: IVec3, color: Color) -> Mesh {
        let Some(chunk) = self.chunks.get(&chunk_key) else {
            return Mesh::new();
        };

        let mut mesh = Mesh::new();
        let origin = chunk_key.as_vec3() * (CHUNK_SIZE as f32) * self.voxel_size;

        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let corners = CornerValues([
                        chunk.get(x, y, z),
                        chunk.get(x + 1, y, z),
                        chunk.get(x + 1, y, z + 1),
                        chunk.get(x, y, z + 1),
                        chunk.get(x, y + 1, z),
                        chunk.get(x + 1, y + 1, z),
                        chunk.get(x + 1, y + 1, z + 1),
                        chunk.get(x, y + 1, z + 1),
                    ]);
                    let cell_origin =
                        origin + Vec3::new(x as f32, y as f32, z as f32) * self.voxel_size;
                    let tris = triangulate_cell(corners, cell_origin, self.voxel_size);
                    for tri in tris {
                        let i0 = mesh.add_point(tri[0]).expect("mc point");
                        let i1 = mesh.add_point(tri[1]).expect("mc point");
                        let i2 = mesh.add_point(tri[2]).expect("mc point");
                        mesh.set_point_color(i0, color).expect("mc color");
                        mesh.set_point_color(i1, color).expect("mc color");
                        mesh.set_point_color(i2, color).expect("mc color");
                        mesh.add_face(&[i0, i1, i2]).expect("mc face");
                    }
                }
            }
        }
        mesh
    }

    /// Extract all dirty chunks in parallel and clear their dirty flags.
    pub fn extract_dirty(&mut self, color: Color) -> Vec<(IVec3, BuiltMesh)> {
        let keys: Vec<IVec3> = self.dirty_chunk_keys();
        let built: Vec<(IVec3, BuiltMesh)> = keys
            .par_iter()
            .map(|key| {
                let mesh = self.extract_chunk(*key, color);
                (*key, mesh.build())
            })
            .collect();

        for key in keys {
            if let Some(c) = self.chunks.get_mut(&key) {
                c.dirty = false;
            }
        }
        built
    }

    /// Extract every chunk into one combined mesh (fine for small demos).
    pub fn extract_all(&self, color: Color) -> BuiltMesh {
        let mut combined = BuiltMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
            opaque_index_count: 0,
        };
        let mut keys: Vec<IVec3> = self.chunks.keys().copied().collect();
        keys.sort_by_key(|k| (k.y, k.z, k.x));
        for key in keys {
            let built = self.extract_chunk(key, color).build();
            combined.append_translated(&built, Vec3::ZERO);
        }
        combined
    }

    /// Extract as a friendly [`Mesh`].
    pub fn extract_mesh(&self, color: Color) -> Mesh {
        let built = self.extract_all(color);
        built_to_editable_mesh(&built)
    }

    /// Keep only chunks whose centers are within `radius` of `focus` (world space).
    /// Returns removed chunk keys.
    pub fn retain_around(&mut self, focus: Vec3, radius: f32) -> Vec<IVec3> {
        let r2 = radius * radius;
        let half = (CHUNK_SIZE as f32) * self.voxel_size * 0.5;
        let mut removed = Vec::new();
        self.chunks.retain(|key, _| {
            let center = key.as_vec3() * (CHUNK_SIZE as f32) * self.voxel_size + Vec3::splat(half);
            let keep = center.distance_squared(focus) <= r2;
            if !keep {
                removed.push(*key);
            }
            keep
        });
        removed
    }

    /// Ensure chunks covering the AABB exist (filled empty by default).
    pub fn ensure_bounds(&mut self, min: Vec3, max: Vec3) {
        let step = self.voxel_size * CHUNK_SIZE as f32;
        let mut y = min.y;
        while y <= max.y {
            let mut z = min.z;
            while z <= max.z {
                let mut x = min.x;
                while x <= max.x {
                    let (key, _) = self.world_to_grid(Vec3::new(x, y, z));
                    self.ensure_chunk(key);
                    x += step;
                }
                z += step;
            }
            y += step;
        }
    }

    pub fn mark_all_dirty(&mut self) {
        for c in self.chunks.values_mut() {
            c.dirty = true;
        }
    }

    /// Lower-detail extraction: sample every `stride` cells (LOD helper).
    pub fn extract_chunk_lod(&self, chunk_key: IVec3, color: Color, stride: i32) -> Mesh {
        assert!(stride >= 1, "stride must be >= 1");
        if stride == 1 {
            return self.extract_chunk(chunk_key, color);
        }
        let Some(chunk) = self.chunks.get(&chunk_key) else {
            return Mesh::new();
        };

        let mut mesh = Mesh::new();
        let origin = chunk_key.as_vec3() * (CHUNK_SIZE as f32) * self.voxel_size;
        let cell = self.voxel_size * stride as f32;

        let mut y = 0;
        while y < CHUNK_SIZE {
            let mut z = 0;
            while z < CHUNK_SIZE {
                let mut x = 0;
                while x < CHUNK_SIZE {
                    let x1 = (x + stride).min(CHUNK_SIZE);
                    let y1 = (y + stride).min(CHUNK_SIZE);
                    let z1 = (z + stride).min(CHUNK_SIZE);
                    let corners = CornerValues([
                        chunk.get(x, y, z),
                        chunk.get(x1, y, z),
                        chunk.get(x1, y, z1),
                        chunk.get(x, y, z1),
                        chunk.get(x, y1, z),
                        chunk.get(x1, y1, z),
                        chunk.get(x1, y1, z1),
                        chunk.get(x, y1, z1),
                    ]);
                    let cell_origin =
                        origin + Vec3::new(x as f32, y as f32, z as f32) * self.voxel_size;
                    let tris = triangulate_cell(corners, cell_origin, cell);
                    for tri in tris {
                        let i0 = mesh.add_point(tri[0]).expect("mc point");
                        let i1 = mesh.add_point(tri[1]).expect("mc point");
                        let i2 = mesh.add_point(tri[2]).expect("mc point");
                        mesh.set_point_color(i0, color).expect("mc color");
                        mesh.set_point_color(i1, color).expect("mc color");
                        mesh.set_point_color(i2, color).expect("mc color");
                        mesh.add_face(&[i0, i1, i2]).expect("mc face");
                    }
                    x += stride;
                }
                z += stride;
            }
            y += stride;
        }
        mesh
    }
}

fn built_to_editable_mesh(built: &BuiltMesh) -> Mesh {
    let mut mesh = Mesh::new();
    let mut ids = Vec::with_capacity(built.positions.len());
    for (i, p) in built.positions.iter().enumerate() {
        let id = mesh.add_point(*p).expect("built point");
        let c = built.colors[i];
        let color = Color::rgba01(c.x, c.y, c.z, c.w).expect("color");
        mesh.set_point_color(id, color).expect("built color");
        ids.push(id);
    }
    for tri in built.indices.chunks_exact(3) {
        mesh.add_face(&[ids[tri[0] as usize], ids[tri[1] as usize], ids[tri[2] as usize]])
            .expect("built face");
    }
    mesh
}

/// Streaming helper: generate/extract chunks around a focus point.
#[derive(Debug)]
pub struct ChunkStreamer {
    pub radius_chunks: i32,
    pub lod_near: f32,
    pub lod_far: f32,
    loaded: HashSet<IVec3>,
}

impl ChunkStreamer {
    pub fn new(radius_chunks: i32) -> Self {
        Self {
            radius_chunks,
            lod_near: 2.0,
            lod_far: 4.0,
            loaded: HashSet::new(),
        }
    }

    pub fn desired_chunks(&self, focus: Vec3, voxel_size: f32) -> Vec<IVec3> {
        let chunk_world = CHUNK_SIZE as f32 * voxel_size;
        let center = IVec3::new(
            (focus.x / chunk_world).floor() as i32,
            (focus.y / chunk_world).floor() as i32,
            (focus.z / chunk_world).floor() as i32,
        );
        let r = self.radius_chunks;
        let mut keys = Vec::new();
        for y in -r..=r {
            for z in -r..=r {
                for x in -r..=r {
                    if x * x + y * y + z * z <= r * r {
                        keys.push(center + IVec3::new(x, y, z));
                    }
                }
            }
        }
        keys
    }

    pub fn lod_stride(&self, chunk_key: IVec3, focus: Vec3, voxel_size: f32) -> i32 {
        let chunk_world = CHUNK_SIZE as f32 * voxel_size;
        let half = chunk_world * 0.5;
        let center = chunk_key.as_vec3() * chunk_world + Vec3::splat(half);
        let dist_chunks = center.distance(focus) / chunk_world;
        if dist_chunks < self.lod_near {
            1
        } else if dist_chunks < self.lod_far {
            2
        } else {
            4
        }
    }

    pub fn mark_loaded(&mut self, key: IVec3) {
        self.loaded.insert(key);
    }

    pub fn unload_outside(&mut self, desired: &HashSet<IVec3>) -> Vec<IVec3> {
        let removed: Vec<IVec3> = self
            .loaded
            .difference(desired)
            .copied()
            .collect();
        for k in &removed {
            self.loaded.remove(k);
        }
        removed
    }

    pub fn is_loaded(&self, key: IVec3) -> bool {
        self.loaded.contains(&key)
    }
}
