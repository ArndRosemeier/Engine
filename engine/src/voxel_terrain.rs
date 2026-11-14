//! Smooth, destructible voxel terrain.
//!
//! The density volume stays on the CPU, where gameplay can query and edit it
//! (digging, building, ray picks, "is this solid"). The renderer meshes the
//! changed parts on the GPU with surface nets, so the surface is smooth rock
//! and earth rather than blocks, and keeps those meshes on the GPU. Far away
//! it meshes larger regions at coarser steps (level of detail).

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::mesh::SurfaceMaterial;
use crate::volume::Volume;
use glam::{IVec3, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Coarsest level of detail: one mesh covers 2^MAX_LOD chunks per axis.
pub const VOXEL_MAX_LOD: u32 = 4;

#[derive(Clone, Debug)]
pub struct VoxelTerrain {
    volume: Volume,
    material: SurfaceMaterial,
    color: Color,
    detail_distance: f32,
    max_lod: u32,
    beach: Option<f32>,
    dry_ground: Color,
    flowers: f32,
    painted_rock: f32,
    cover: Option<GroundCoverFn>,
    far: FarTiles,
}

/// What grows on the natural ground at a point (see
/// [`VoxelTerrain::with_ground_cover`] and
/// [`crate::stream::WorldGenerator::ground_cover`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundCover {
    /// 1: the usual grass. 0: bare dry ground in the terrain's dry colour
    /// ([`VoxelTerrain::with_dry_ground`]): sand, steppe, scree. Between: a mix.
    pub green: f32,
    /// Height (world y) above which the ground lies under snow; a very
    /// large value for no snow.
    pub snow_line: f32,
}

impl GroundCover {
    /// Grass everywhere and no snow: what a terrain without cover shows.
    pub const GRASS: Self = Self {
        green: 1.0,
        snow_line: 1.0e9,
    };
}

/// The colours wildflowers come in ([`VoxelTerrain::with_flowers`]), sRGB.
pub const FLOWER_COLORS: [[f32; 3]; 5] = [
    [0.93, 0.2, 0.1],
    [0.98, 0.78, 0.1],
    [0.52, 0.3, 0.9],
    [0.98, 0.45, 0.68],
    [0.97, 0.94, 0.8],
];

fn flower_hash(x: i32, z: i32, salt: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (z as u32).wrapping_mul(0x85EB_CA77)
        ^ salt.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

/// Value noise on the integer lattice, 0 to 1 (the shader's `flower_noise`).
fn flower_noise(x: f32, z: f32, salt: u32) -> f32 {
    let (fx, fz) = (x.floor(), z.floor());
    let (ix, iz) = (fx as i32, fz as i32);
    let (tx, tz) = (x - fx, z - fz);
    let (sx, sz) = (tx * tx * (3.0 - 2.0 * tx), tz * tz * (3.0 - 2.0 * tz));
    let a = flower_hash(ix, iz, salt);
    let b = flower_hash(ix + 1, iz, salt);
    let c = flower_hash(ix, iz + 1, salt);
    let d = flower_hash(ix + 1, iz + 1, salt);
    (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sz
}

/// Where wildflowers grow ([`VoxelTerrain::with_flowers`]) and in which
/// colour: how thick they stand at ground position `(x, z)`, 0 (none) to 1
/// (before the terrain's `amount` and its grass), and their colour (sRGB).
/// The terrain shader draws from the same numbers, so flowers a game plants
/// by this stand in the coloured fields.
pub fn flower_field(x: f32, z: f32) -> (f32, [f32; 3]) {
    let field =
        0.65 * flower_noise(x / 64.0, z / 64.0, 1) + 0.35 * flower_noise(x / 23.0, z / 23.0, 2);
    // Fields here and there in the grass, not half the land.
    let amount = ((field - 0.57) / 0.13).clamp(0.0, 1.0);
    let amount = amount * amount * (3.0 - 2.0 * amount);
    // One colour to a stretch of country: cells some 170 m across with
    // wandering edges, so a field is one colour and never a rainbow.
    let wx = x + 60.0 * (flower_noise(x / 90.0, z / 90.0, 3) - 0.5);
    let wz = z + 60.0 * (flower_noise(x / 90.0, z / 90.0, 4) - 0.5);
    let cell = flower_hash((wx / 170.0).floor() as i32, (wz / 170.0).floor() as i32, 5);
    let i = ((cell * 5.0) as usize).min(4);
    (amount, FLOWER_COLORS[i])
}

/// A function from ground position `(x, z)` to its cover.
#[derive(Clone)]
pub struct GroundCoverFn(pub Arc<dyn Fn(f32, f32) -> GroundCover + Send + Sync>);

impl std::fmt::Debug for GroundCoverFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GroundCoverFn")
    }
}

/// Address of a far terrain tile: quadtree level and tile index. A tile at
/// `level` is `base * 2^level` metres across.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FarTileKey {
    pub level: u8,
    pub x: i32,
    pub z: i32,
}

/// A height-field patch drawn beyond the voxels with the voxel shader, so
/// distant ground looks like the near ground. Vertices are packed the way
/// the voxel renderer stores them (position, normal, exposure).
#[derive(Debug)]
pub struct FarTileMesh {
    /// World position the vertices are relative to.
    pub(crate) origin: Vec3,
    /// 5 words per vertex: position xyz (f32), normal (snorm16 x4), uv (snorm16 x2).
    pub(crate) vertices: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    /// World-space bounds of the geometry.
    pub(crate) min: Vec3,
    pub(crate) max: Vec3,
    /// Ground area it covers (x, z min and max), for hiding under voxels.
    pub(crate) footprint: [f32; 4],
    /// Lakes and rivers that stand on this ground (see [`Self::with_water`]).
    pub(crate) water: Option<FarWater>,
}

/// Water surfaces of a far tile: what is drawn of lakes and rivers beyond
/// the simulated water. World-space triangles.
#[derive(Debug)]
pub(crate) struct FarWater {
    /// Position xyz and the water's depth there.
    pub(crate) vertices: Vec<[f32; 4]>,
    pub(crate) indices: Vec<u32>,
}

impl FarTileMesh {
    /// Build a patch from a `(n + 3)`² grid of heights: corners `-1..=n+1`
    /// of an `n` x `n` cell grid starting at `(x0, z0)` with `cell` metres per
    /// cell (the outer ring only shapes the normals). The patch is lowered by
    /// `sink` and gets a skirt `skirt` metres deep around its edge, so tiles
    /// of different detail (and the voxels next to them) meet without gaps.
    pub fn from_heights(
        x0: f32,
        z0: f32,
        cell: f32,
        n: usize,
        heights: &[f32],
        sink: f32,
        skirt: f32,
    ) -> Self {
        Self::from_heights_and_cover(x0, z0, cell, n, heights, None, sink, skirt)
    }

    /// [`Self::from_heights`] with how green the ground is at each of the
    /// same `(n + 3)`² corners (1 grass .. 0 bare dry ground, see
    /// [`GroundCover::green`]), stored per vertex.
    #[allow(clippy::too_many_arguments)]
    pub fn from_heights_and_cover(
        x0: f32,
        z0: f32,
        cell: f32,
        n: usize,
        heights: &[f32],
        green: Option<&[f32]>,
        sink: f32,
        skirt: f32,
    ) -> Self {
        let w = n + 3;
        assert_eq!(heights.len(), w * w, "far tile needs (n + 3)^2 heights");
        if let Some(g) = green {
            assert_eq!(g.len(), w * w, "far tile needs (n + 3)^2 cover values");
        }
        let h = |x: usize, z: usize| heights[z * w + x] - sink;
        // The vertex's spare half word: 0 for grass, down to -1 for bare
        // ground (the shader reads uv.y >= 0 as "dug open").
        let bare_at = |x: usize, z: usize| -> f32 {
            green.map_or(0.0, |g| {
                let v = g[z * w + x];
                -(1.0
                    - if v.is_finite() {
                        v.clamp(0.0, 1.0)
                    } else {
                        1.0
                    })
            })
        };
        let origin = Vec3::new(x0, 0.0, z0);
        let mut vertices = Vec::with_capacity(((n + 1) * (n + 1) + 4 * (n + 1)) * 5);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut push = |p: Vec3, normal: Vec3, bare: f32, vertices: &mut Vec<u32>| {
            min = min.min(p + origin);
            max = max.max(p + origin);
            let q = |v: f32| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16) as u16 as u32;
            vertices.push(p.x.to_bits());
            vertices.push(p.y.to_bits());
            vertices.push(p.z.to_bits());
            vertices.push(q(normal.x) | (q(normal.y) << 16));
            vertices.push(q(normal.z) | (q(bare) << 16));
        };
        let normal_at = |x: usize, z: usize| {
            let dx = h(x + 2, z + 1) - h(x, z + 1);
            let dz = h(x + 1, z + 2) - h(x + 1, z);
            Vec3::new(-dx, 2.0 * cell, -dz).normalize_or(Vec3::Y)
        };
        for z in 0..=n {
            for x in 0..=n {
                let p = Vec3::new(x as f32 * cell, h(x + 1, z + 1), z as f32 * cell);
                push(p, normal_at(x, z), bare_at(x + 1, z + 1), &mut vertices);
            }
        }
        let mut indices = Vec::with_capacity(n * n * 6 + 4 * n * 6);
        let row = (n + 1) as u32;
        for z in 0..n as u32 {
            for x in 0..n as u32 {
                let a = z * row + x;
                let b = a + 1;
                let c = a + row;
                let d = c + 1;
                // Counter-clockwise seen from above (front faces up).
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        // Skirts: each edge gets a copy of its vertices pushed down, joined
        // to the edge by a strip that faces outward.
        let edges: [Vec<(usize, usize)>; 4] = [
            (0..=n).map(|x| (x, 0)).collect(),
            (0..=n).map(|z| (n, z)).collect(),
            (0..=n).rev().map(|x| (x, n)).collect(),
            (0..=n).rev().map(|z| (0, z)).collect(),
        ];
        for edge in edges.iter() {
            let base = (vertices.len() / 5) as u32;
            for &(x, z) in edge {
                let p = Vec3::new(x as f32 * cell, h(x + 1, z + 1) - skirt, z as f32 * cell);
                push(p, normal_at(x, z), bare_at(x + 1, z + 1), &mut vertices);
            }
            for i in 0..n {
                let (x, z) = edge[i];
                let (x1, z1) = edge[i + 1];
                let top0 = (z as u32) * row + x as u32;
                let top1 = (z1 as u32) * row + x1 as u32;
                let bot0 = base + i as u32;
                let bot1 = bot0 + 1;
                // Edges run the same way round the tile, so this winding
                // faces outward on all four.
                indices.extend_from_slice(&[top0, top1, bot0, top1, bot1, bot0]);
            }
        }
        let size = n as f32 * cell;
        Self {
            origin,
            vertices,
            indices,
            min,
            max,
            footprint: [x0, z0, x0 + size, z0 + size],
            water: None,
        }
    }

    /// Add the water that stands on this ground: `levels` is the surface
    /// height of the water at each of the same `(n + 3)`² corners the patch
    /// was built from (`None` for dry land; leave the sea out, it is drawn
    /// on its own). A sheet is laid wherever a level lies above the ground;
    /// neighbouring corners at different levels (a waterfall, a river
    /// running downhill) are joined by a sloping sheet.
    pub fn with_water(
        mut self,
        cell: f32,
        n: usize,
        heights: &[f32],
        levels: &[Option<f32>],
    ) -> Self {
        let w = n + 3;
        assert_eq!(heights.len(), w * w, "far tile needs (n + 3)^2 heights");
        assert_eq!(levels.len(), w * w, "far tile needs (n + 3)^2 water levels");
        let [x0, z0, ..] = self.footprint;
        let wet = |x: usize, z: usize| {
            levels[z * w + x].filter(|l| l.is_finite() && *l > heights[z * w + x])
        };
        if !(0..w * w).any(|i| wet(i % w, i / w).is_some()) {
            return self;
        }
        // Each corner's sheet height: its own water, or for a dry corner the
        // water next to it, kept under the ground where the ground is lower
        // than that (the far side of a dam or a bank).
        let mut sheet: Vec<Option<[f32; 2]>> = Vec::with_capacity((n + 1) * (n + 1));
        for z in 1..=n + 1 {
            for x in 1..=n + 1 {
                let h = heights[z * w + x];
                sheet.push(match wet(x, z) {
                    Some(l) => Some([l, l - h]),
                    None => {
                        let (mut sum, mut count) = (0.0, 0);
                        for (nx, nz) in [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)] {
                            if let Some(l) = wet(nx, nz) {
                                sum += l;
                                count += 1;
                            }
                        }
                        (count > 0).then(|| [(sum / count as f32).min(h - 0.3), 0.0])
                    }
                });
            }
        }
        let row = n + 1;
        let mut slot = vec![u32::MAX; row * row];
        let mut water = FarWater {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        for z in 0..n {
            for x in 0..n {
                let corners = [
                    z * row + x,
                    z * row + x + 1,
                    (z + 1) * row + x,
                    (z + 1) * row + x + 1,
                ];
                let any_wet = corners
                    .iter()
                    .any(|&c| sheet[c].is_some_and(|s| s[1] > 0.0));
                if !any_wet || corners.iter().any(|&c| sheet[c].is_none()) {
                    continue;
                }
                let mut ids = [0u32; 4];
                for (id, &c) in ids.iter_mut().zip(&corners) {
                    if slot[c] == u32::MAX {
                        slot[c] = water.vertices.len() as u32;
                        let [y, depth] = sheet[c].expect("checked");
                        let (cx, cz) = ((c % row) as f32, (c / row) as f32);
                        water
                            .vertices
                            .push([x0 + cx * cell, y, z0 + cz * cell, depth]);
                        self.max.y = self.max.y.max(y);
                    }
                    *id = slot[c];
                }
                let [a, b, c, d] = ids;
                water.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        if !water.indices.is_empty() {
            self.water = Some(water);
        }
        self
    }

    /// Triangles of lake and river surface on this tile.
    pub fn water_triangle_count(&self) -> usize {
        self.water.as_ref().map_or(0, |w| w.indices.len() / 3)
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 5
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// The far tiles a terrain currently shows (see [`VoxelTerrain::set_far_tile`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct FarTiles {
    pub tiles: HashMap<FarTileKey, Arc<FarTileMesh>>,
    pub rev: u64,
}

impl VoxelTerrain {
    pub fn new(volume: Volume) -> Self {
        Self {
            volume,
            material: SurfaceMaterial::DEFAULT,
            color: Color::rgb(142, 128, 112),
            detail_distance: 1.0,
            max_lod: VOXEL_MAX_LOD,
            beach: None,
            dry_ground: Color::rgb(196, 170, 120),
            flowers: 0.0,
            painted_rock: 0.0,
            cover: None,
            far: FarTiles::default(),
        }
    }

    /// Show a far height-field tile (replacing one with the same key). Far
    /// tiles are drawn with the voxel shader wherever no voxel region covers
    /// their whole area, so the ground runs on to the horizon past the
    /// voxels. [`crate::stream::WorldStream`] manages them for you.
    pub fn set_far_tile(&mut self, key: FarTileKey, mesh: Arc<FarTileMesh>) {
        self.far.tiles.insert(key, mesh);
        self.far.rev += 1;
    }

    pub fn remove_far_tile(&mut self, key: FarTileKey) -> bool {
        let removed = self.far.tiles.remove(&key).is_some();
        if removed {
            self.far.rev += 1;
        }
        removed
    }

    pub fn has_far_tile(&self, key: FarTileKey) -> bool {
        self.far.tiles.contains_key(&key)
    }

    pub fn far_tile_count(&self) -> usize {
        self.far.tiles.len()
    }

    pub(crate) fn far_tiles(&self) -> &FarTiles {
        &self.far
    }

    /// Material for the whole surface (rock, dirt, cave profiles...).
    pub fn with_material(mut self, material: SurfaceMaterial) -> Self {
        self.material = material;
        self
    }

    /// Sand beaches: flat and gentle ground up to height `top_y` (a little
    /// above the sea) shows sand; grass, earth and rock above it as usual.
    pub fn with_beach(mut self, top_y: f32) -> Self {
        self.beach = top_y.is_finite().then_some(top_y);
        self
    }

    /// Wildflowers on the grass: fields of them some tens of metres across,
    /// each mostly one colour (scarlet, gold, violet, pink, cream), washing
    /// the ground with colour from afar and speckling it close by.
    /// `amount` 0 (none, the default) to 1 (as much as there is); where they
    /// grow and in which colour is [`flower_field`], the same on the CPU, so
    /// a game can stand real flowers in them. They follow the grass: none on
    /// dry ground, snow, sand or rock.
    pub fn with_flowers(mut self, amount: f32) -> Self {
        self.flowers = if amount.is_finite() {
            amount.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    /// How much the grass flowers ([`Self::with_flowers`]).
    pub fn flowers(&self) -> f32 {
        self.flowers
    }

    /// Rock in the colours of a painted desert: its level beds banded
    /// red, ochre and cream. `amount` 0 (the terrain's own rock colour, the
    /// default) to 1.
    pub fn with_painted_rock(mut self, amount: f32) -> Self {
        self.painted_rock = if amount.is_finite() {
            amount.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    /// How much the rock is painted ([`Self::with_painted_rock`]).
    pub fn painted_rock(&self) -> f32 {
        self.painted_rock
    }

    /// Top of the sand beaches, if set ([`Self::with_beach`]).
    pub fn beach(&self) -> Option<f32> {
        self.beach
    }

    /// What grows where: `cover(x, z)` says how green the natural ground is
    /// and where the snow line lies. Without it, grass everywhere.
    /// [`crate::stream::WorldStream`] sets it from its generator.
    pub fn with_ground_cover(
        mut self,
        cover: impl Fn(f32, f32) -> GroundCover + Send + Sync + 'static,
    ) -> Self {
        self.cover = Some(GroundCoverFn(Arc::new(cover)));
        self
    }

    /// The ground cover at a point, if the terrain has one.
    pub fn ground_cover(&self, x: f32, z: f32) -> Option<GroundCover> {
        self.cover.as_ref().map(|c| (c.0)(x, z))
    }

    /// Colour of bare dry ground, shown where the cover is not green
    /// (sand by default).
    pub fn with_dry_ground(mut self, color: Color) -> Self {
        self.dry_ground = color;
        self
    }

    pub fn dry_ground(&self) -> Color {
        self.dry_ground
    }

    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// How far full detail reaches, in multiples of a mesh region's size.
    ///
    /// A region is split into finer ones while the camera is closer than
    /// `distance * region size`. Larger keeps more detail (and costs more).
    pub fn with_detail_distance(mut self, distance: f32) -> EngineResult<Self> {
        if !distance.is_finite() || distance < 0.5 {
            return Err(EngineError::InvalidValue(
                "voxel detail distance must be finite and >= 0.5".into(),
            ));
        }
        self.detail_distance = distance;
        Ok(self)
    }

    /// Coarsest level of detail, 0 (always full detail) to [`VOXEL_MAX_LOD`].
    pub fn with_max_lod(mut self, max_lod: u32) -> Self {
        self.max_lod = max_lod.min(VOXEL_MAX_LOD);
        self
    }

    pub fn volume(&self) -> &Volume {
        &self.volume
    }

    /// Direct access for bulk edits. The renderer re-meshes whatever changed.
    pub fn volume_mut(&mut self) -> &mut Volume {
        &mut self.volume
    }

    pub fn material(&self) -> SurfaceMaterial {
        self.material
    }

    pub fn color(&self) -> Color {
        self.color
    }

    pub fn detail_distance(&self) -> f32 {
        self.detail_distance
    }

    pub fn max_lod(&self) -> u32 {
        self.max_lod
    }

    /// Dig a smooth spherical hole. Returns how many chunks changed.
    pub fn dig_sphere(&mut self, center: Vec3, radius: f32) -> usize {
        self.volume.dig_sphere(center, radius)
    }

    /// Add a smooth solid sphere. Returns how many chunks changed.
    pub fn add_sphere(&mut self, center: Vec3, radius: f32) -> usize {
        self.volume.add_sphere(center, radius)
    }

    /// First solid point along a ray.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<Vec3> {
        self.volume.raycast(origin, dir, max_dist)
    }

    pub fn is_solid(&self, p: Vec3) -> bool {
        self.volume.is_solid(p)
    }

    /// Outward surface normal at `p` (from the density gradient).
    pub fn normal_at(&self, p: Vec3) -> Vec3 {
        let e = self.volume.voxel_size() * 0.5;
        let d = |o: Vec3| self.volume.sample(p + o);
        let g = Vec3::new(
            d(Vec3::X * e) - d(-Vec3::X * e),
            d(Vec3::Y * e) - d(-Vec3::Y * e),
            d(Vec3::Z * e) - d(-Vec3::Z * e),
        );
        (-g).try_normalize().unwrap_or(Vec3::Y)
    }

    /// Fill chunks covering a world box from a density function, in parallel
    /// (positive = solid). Use a signed distance in voxels for smooth results.
    pub fn generate(&mut self, min: Vec3, max: Vec3, f: impl Fn(Vec3) -> f32 + Sync) {
        let size = self.volume.chunk_world_size();
        let lo = (min / size).floor().as_ivec3();
        let hi = ((max / size).ceil().as_ivec3() - IVec3::ONE).max(lo);
        let mut keys = Vec::new();
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    keys.push(IVec3::new(x, y, z));
                }
            }
        }
        self.volume.generate_chunks(&keys, f);
    }
}
