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

/// Coarsest level of detail: one mesh covers 2^MAX_LOD chunks per axis.
pub const VOXEL_MAX_LOD: u32 = 4;

#[derive(Clone, Debug)]
pub struct VoxelTerrain {
    volume: Volume,
    material: SurfaceMaterial,
    color: Color,
    detail_distance: f32,
    max_lod: u32,
}

impl VoxelTerrain {
    pub fn new(volume: Volume) -> Self {
        Self {
            volume,
            material: SurfaceMaterial::DEFAULT,
            color: Color::rgb(142, 128, 112),
            detail_distance: 1.0,
            max_lod: VOXEL_MAX_LOD,
        }
    }

    /// Material for the whole surface (rock, dirt, cave profiles...).
    pub fn with_material(mut self, material: SurfaceMaterial) -> Self {
        self.material = material;
        self
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
