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
    far: FarTiles,
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
        let w = n + 3;
        assert_eq!(heights.len(), w * w, "far tile needs (n + 3)^2 heights");
        let h = |x: usize, z: usize| heights[z * w + x] - sink;
        let origin = Vec3::new(x0, 0.0, z0);
        let mut vertices = Vec::with_capacity(((n + 1) * (n + 1) + 4 * (n + 1)) * 5);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut push = |p: Vec3, normal: Vec3, vertices: &mut Vec<u32>| {
            min = min.min(p + origin);
            max = max.max(p + origin);
            let q = |v: f32| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16) as u16 as u32;
            vertices.push(p.x.to_bits());
            vertices.push(p.y.to_bits());
            vertices.push(p.z.to_bits());
            vertices.push(q(normal.x) | (q(normal.y) << 16));
            vertices.push(q(normal.z));
        };
        let normal_at = |x: usize, z: usize| {
            let dx = h(x + 2, z + 1) - h(x, z + 1);
            let dz = h(x + 1, z + 2) - h(x + 1, z);
            Vec3::new(-dx, 2.0 * cell, -dz).normalize_or(Vec3::Y)
        };
        for z in 0..=n {
            for x in 0..=n {
                let p = Vec3::new(x as f32 * cell, h(x + 1, z + 1), z as f32 * cell);
                push(p, normal_at(x, z), &mut vertices);
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
                push(p, normal_at(x, z), &mut vertices);
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
        }
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

    /// Top of the sand beaches, if set ([`Self::with_beach`]).
    pub fn beach(&self) -> Option<f32> {
        self.beach
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
