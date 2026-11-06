use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::marching_cubes::{triangulate_cell, CornerValues};
use crate::mesh::{BuiltMesh, Mesh};
use crate::place::{ensure_finite, ensure_finite3};
use glam::{IVec3, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Sample material: surface uncovered or built by an edit (bare earth/rock).
pub const MATERIAL_EXPOSED: u8 = 1;

/// Blend width (in density units, about voxels) of smooth dig/add edits.
const SMOOTH_EDIT: f32 = 1.5;

/// Polynomial smooth minimum: like `a.min(b)` but rounded within `k`.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b + (a - b) * h - k * h * (1.0 - h)
}

/// Number of voxels along one axis of a chunk (exclusive upper bound of cell grid).
pub const CHUNK_SIZE: i32 = 32;

#[derive(Clone, Debug)]
struct Chunk {
    /// Density samples at cell corners: (CHUNK_SIZE+1)^3
    /// Positive = solid, negative = empty. Surface at 0.
    samples: Vec<f32>,
    /// Per-sample surface material (0 natural, 1 exposed by an edit),
    /// allocated on the first edit.
    materials: Option<Vec<u8>>,
    dirty: bool,
    /// `Volume::rev` at this chunk's last change.
    rev: u64,
}

impl Chunk {
    fn new() -> Self {
        let n = (CHUNK_SIZE + 1) as usize;
        Self {
            samples: vec![-1.0; n * n * n],
            materials: None,
            dirty: true,
            rev: 0,
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
    voxel_size: f32,
    chunks: HashMap<IVec3, Chunk>,
    /// Bumped on every change; chunks remember the value of their last one.
    rev: u64,
    /// Revision of the last chunk removal (removals leave no chunk to stamp).
    removed_rev: u64,
}

impl Default for Volume {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Volume {
    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    pub fn new(voxel_size: f32) -> Self {
        Self::try_new(voxel_size).expect("voxel_size must be finite and > 0")
    }

    pub fn try_new(voxel_size: f32) -> EngineResult<Self> {
        ensure_finite(voxel_size, "voxel_size")?;
        if voxel_size <= 0.0 {
            return Err(EngineError::InvalidValue("voxel_size must be > 0".into()));
        }
        Ok(Self {
            voxel_size,
            chunks: HashMap::new(),
            rev: 1,
            removed_rev: 0,
        })
    }

    fn estimate_samples(min: Vec3, max: Vec3, step: f32) -> EngineResult<u64> {
        let nx = ((max.x - min.x) / step).floor().max(0.0) as u64 + 1;
        let ny = ((max.y - min.y) / step).floor().max(0.0) as u64 + 1;
        let nz = ((max.z - min.z) / step).floor().max(0.0) as u64 + 1;
        nx.checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .ok_or_else(|| {
                EngineError::ResourceLimit(format!(
                    "volume paint sample count overflow for dimensions {nx}x{ny}x{nz}"
                ))
            })
    }

    fn check_paint_budget(&self, min: Vec3, max: Vec3, limits: &EngineLimits) -> EngineResult<()> {
        ensure_finite3(min, "paint min")?;
        ensure_finite3(max, "paint max")?;
        let samples = Self::estimate_samples(min, max, self.voxel_size)?;
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
        if self.chunks.remove(&key).is_some() {
            self.rev += 1;
            self.removed_rev = self.rev;
        }
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
        self.rev += 1;
        let rev = self.rev;
        let chunk = self.chunks.entry(key).or_insert_with(Chunk::new);
        chunk.rev = rev;
        chunk
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
            uvs: Vec::new(),
            indices: Vec::new(),
            opaque_index_count: 0,
            surface: Default::default(),
            surface_overrides: Vec::new(),
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
        if !removed.is_empty() {
            self.rev += 1;
            self.removed_rev = self.rev;
        }
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

    /// Edge length of one chunk in world units.
    pub fn chunk_world_size(&self) -> f32 {
        CHUNK_SIZE as f32 * self.voxel_size
    }

    /// Change counter: every edit, new chunk or removal raises it.
    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// Chunks changed after revision `since`, and whether any chunk was removed
    /// since then (a removal leaves no key behind to report).
    pub fn changed_since(&self, since: u64) -> (Vec<IVec3>, bool) {
        let keys = self
            .chunks
            .iter()
            .filter(|(_, c)| c.rev > since)
            .map(|(k, _)| *k)
            .collect();
        (keys, self.removed_rev > since)
    }

    /// Fill whole chunks from a density function, in parallel.
    ///
    /// `f` gets world positions of grid corners. Samples shared by
    /// neighbouring chunks are evaluated at the same position, so seams stay
    /// consistent. Existing chunks with these keys are replaced.
    pub fn generate_chunks(&mut self, keys: &[IVec3], f: impl Fn(Vec3) -> f32 + Sync) {
        let n = CHUNK_SIZE + 1;
        let vs = self.voxel_size;
        let built: Vec<(IVec3, Vec<f32>)> = keys
            .par_iter()
            .map(|&key| {
                let base = key * CHUNK_SIZE;
                let mut samples = Vec::with_capacity((n * n * n) as usize);
                for y in 0..n {
                    for z in 0..n {
                        for x in 0..n {
                            let g = base + IVec3::new(x, y, z);
                            let d = f(g.as_vec3() * vs);
                            samples.push(if d.is_finite() { d } else { -1.0 });
                        }
                    }
                }
                (key, samples)
            })
            .collect();
        self.rev += 1;
        for (key, samples) in built {
            self.chunks.insert(
                key,
                Chunk {
                    samples,
                    materials: None,
                    dirty: true,
                    rev: self.rev,
                },
            );
        }
    }

    /// Smoothly dig a sphere out of the volume (the surface follows the
    /// sphere, not the voxel grid). Returns how many chunks changed.
    pub fn dig_sphere(&mut self, center: impl Into<Vec3>, radius: f32) -> usize {
        let center = center.into();
        let vs = self.voxel_size;
        self.paint_smooth(center, radius, false, |p| {
            (p.distance(center) - radius) / vs
        })
    }

    /// Smoothly add a solid sphere. Returns how many chunks changed.
    pub fn add_sphere(&mut self, center: impl Into<Vec3>, radius: f32) -> usize {
        let center = center.into();
        let vs = self.voxel_size;
        self.paint_smooth(center, radius, true, |p| (radius - p.distance(center)) / vs)
    }

    /// Combine a signed distance (in voxels) into every sample near a sphere:
    /// `max` when adding solid, `min` when digging.
    fn paint_smooth(
        &mut self,
        center: Vec3,
        radius: f32,
        add: bool,
        sdf: impl Fn(Vec3) -> f32,
    ) -> usize {
        if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return 0;
        }
        let vs = self.voxel_size;
        let reach = radius + 3.0 * vs;
        let lo = ((center - Vec3::splat(reach)) / vs).floor().as_ivec3();
        let hi = ((center + Vec3::splat(reach)) / vs).ceil().as_ivec3();
        // Each chunk owns local corners 0..=CHUNK_SIZE; the last one is the
        // next chunk's first, so the same global corner is painted in both.
        let ck_lo = (lo - IVec3::ONE).div_euclid(IVec3::splat(CHUNK_SIZE));
        let ck_hi = hi.div_euclid(IVec3::splat(CHUNK_SIZE));
        let mut changed = 0;
        for cy in ck_lo.y..=ck_hi.y {
            for cz in ck_lo.z..=ck_hi.z {
                for cx in ck_lo.x..=ck_hi.x {
                    let key = IVec3::new(cx, cy, cz);
                    if !add && !self.chunks.contains_key(&key) {
                        continue; // missing chunks are already empty
                    }
                    let base = key * CHUNK_SIZE;
                    let from = (lo - base).max(IVec3::ZERO);
                    let to = (hi - base).min(IVec3::splat(CHUNK_SIZE));
                    if from.cmpgt(to).any() {
                        continue;
                    }
                    let mut touched = false;
                    let chunk = self.chunks.entry(key).or_insert_with(Chunk::new);
                    for y in from.y..=to.y {
                        for z in from.z..=to.z {
                            for x in from.x..=to.x {
                                let p = (base + IVec3::new(x, y, z)).as_vec3() * vs;
                                let d = sdf(p).clamp(-3.0, 3.0);
                                let i = Chunk::index(x, y, z);
                                let old = chunk.samples[i];
                                // Smooth union / subtraction rounds the rim
                                // where the sphere meets the old surface, so
                                // no paper-thin slivers are left behind.
                                let new = if add {
                                    -smooth_min(-old, -d, SMOOTH_EDIT)
                                } else {
                                    smooth_min(old, d, SMOOTH_EDIT)
                                };
                                if new != old {
                                    chunk.samples[i] = new;
                                    touched = true;
                                    if new > -2.0 && new < 2.0 {
                                        let len = chunk.samples.len();
                                        chunk.materials.get_or_insert_with(|| vec![0; len])[i] =
                                            MATERIAL_EXPOSED;
                                    }
                                }
                            }
                        }
                    }
                    if touched {
                        self.rev += 1;
                        chunk.rev = self.rev;
                        chunk.dirty = true;
                        changed += 1;
                    }
                }
            }
        }
        changed
    }

    /// Density at a global grid corner (outside any chunk: empty, -1).
    pub fn corner(&self, g: IVec3) -> f32 {
        let key = g.div_euclid(IVec3::splat(CHUNK_SIZE));
        let local = g.rem_euclid(IVec3::splat(CHUNK_SIZE));
        self.chunks
            .get(&key)
            .map_or(-1.0, |c| c.get(local.x, local.y, local.z))
    }

    /// Trilinear density at a world position.
    pub fn sample(&self, p: Vec3) -> f32 {
        let g = p / self.voxel_size;
        let base = g.floor();
        let f = g - base;
        let b = base.as_ivec3();
        let c = |x: i32, y: i32, z: i32| self.corner(b + IVec3::new(x, y, z));
        let x00 = c(0, 0, 0) + (c(1, 0, 0) - c(0, 0, 0)) * f.x;
        let x10 = c(0, 1, 0) + (c(1, 1, 0) - c(0, 1, 0)) * f.x;
        let x01 = c(0, 0, 1) + (c(1, 0, 1) - c(0, 0, 1)) * f.x;
        let x11 = c(0, 1, 1) + (c(1, 1, 1) - c(0, 1, 1)) * f.x;
        let y0 = x00 + (x10 - x00) * f.y;
        let y1 = x01 + (x11 - x01) * f.y;
        y0 + (y1 - y0) * f.z
    }

    /// Whether a world position is inside solid material.
    pub fn is_solid(&self, p: Vec3) -> bool {
        self.sample(p) > 0.0
    }

    /// First point along a ray where the volume turns solid.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<Vec3> {
        let dir = dir.try_normalize()?;
        let step = self.voxel_size * 0.5;
        let mut t = 0.0;
        let mut prev_t = 0.0;
        if self.sample(origin) > 0.0 {
            return Some(origin);
        }
        while t < max_dist {
            t = (t + step).min(max_dist);
            let d = self.sample(origin + dir * t);
            if d > 0.0 {
                // Refine between the last empty and the first solid sample.
                let (mut a, mut b) = (prev_t, t);
                for _ in 0..8 {
                    let m = (a + b) * 0.5;
                    if self.sample(origin + dir * m) > 0.0 {
                        b = m;
                    } else {
                        a = m;
                    }
                }
                return Some(origin + dir * b);
            }
            prev_t = t;
        }
        None
    }

    /// Copy an `n`³ block of corners starting at global grid corner `origin`,
    /// taking every `stride`-th corner, x fastest then y then z. Each value
    /// carries its material in the two lowest mantissa bits (see
    /// [`pack_material`]); the sign, and so the surface, is unchanged.
    pub(crate) fn gather(&self, origin: IVec3, stride: i32, n: i32, out: &mut Vec<f32>) {
        out.clear();
        out.reserve((n * n * n) as usize);
        let cs = IVec3::splat(CHUNK_SIZE);
        let mut cached_key = IVec3::splat(i32::MIN);
        let mut cached: Option<&Chunk> = None;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let g = origin + IVec3::new(x, y, z) * stride;
                    let key = g.div_euclid(cs);
                    if key != cached_key {
                        cached_key = key;
                        cached = self.chunks.get(&key);
                    }
                    out.push(match cached {
                        Some(c) => {
                            let l = g - key * cs;
                            let i = Chunk::index(l.x, l.y, l.z);
                            let d = c.samples[i];
                            match &c.materials {
                                Some(m) => pack_material(d, m[i]),
                                None => pack_material(d, 0),
                            }
                        }
                        None => pack_material(-1.0, 0),
                    });
                }
            }
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
        mesh.add_face(&[
            ids[tri[0] as usize],
            ids[tri[1] as usize],
            ids[tri[2] as usize],
        ])
        .expect("built face");
    }
    mesh
}

/// Streaming helper: generate/extract chunks around a focus point.
#[derive(Debug)]
pub struct ChunkStreamer {
    radius_chunks: i32,
    lod_near: f32,
    lod_far: f32,
    loaded: HashSet<IVec3>,
}

impl ChunkStreamer {
    pub fn radius_chunks(&self) -> i32 {
        self.radius_chunks
    }
    pub fn lod_near(&self) -> f32 {
        self.lod_near
    }
    pub fn lod_far(&self) -> f32 {
        self.lod_far
    }

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
        let removed: Vec<IVec3> = self.loaded.difference(desired).copied().collect();
        for k in &removed {
            self.loaded.remove(k);
        }
        removed
    }

    pub fn is_loaded(&self, key: IVec3) -> bool {
        self.loaded.contains(&key)
    }
}

/// Store a 2-bit material in the lowest mantissa bits of a density. The
/// change is below 1e-6 relative and never touches the sign; exact zero is
/// nudged to a tiny negative value first so it stays "empty".
pub(crate) fn pack_material(density: f32, material: u8) -> f32 {
    let d = if density == 0.0 {
        -f32::MIN_POSITIVE
    } else {
        density
    };
    f32::from_bits((d.to_bits() & !3) | u32::from(material & 3))
}
