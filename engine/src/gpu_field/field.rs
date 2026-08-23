//! GPU scalar-field session types (paint + extract orchestration).

use glam::{IVec3, Vec3};

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::mesh::{BuiltMesh, Mesh};

use super::context::FieldGpuContext;
use super::grid::FieldGrid;
use super::kernel::FieldKernel;

/// Opaque parameter block for a user density kernel (WGSL + `bytemuck::Pod` on the caller side).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldParamsBlob(pub Vec<u8>);

impl FieldParamsBlob {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// World-space axis-aligned region to fill and extract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldBounds {
    pub min: Vec3,
    pub max: Vec3,
    pub voxel_size: f32,
}

impl FieldBounds {
    pub fn try_new(min: Vec3, max: Vec3, voxel_size: f32) -> EngineResult<Self> {
        crate::place::ensure_finite3(min, "field bounds min")?;
        crate::place::ensure_finite3(max, "field bounds max")?;
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Err(EngineError::InvalidValue(
                "field voxel_size must be finite and > 0".into(),
            ));
        }
        if min.x >= max.x || min.y >= max.y || min.z >= max.z {
            return Err(EngineError::InvalidValue(
                "field bounds min must be strictly less than max on every axis".into(),
            ));
        }
        Ok(Self {
            min,
            max,
            voxel_size,
        })
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

/// Chunk index in field grid space (32³ cells per chunk, matching CPU `Volume`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FieldChunkKey(pub IVec3);

/// Axis-aligned bounds of an extracted chunk mesh in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldMeshBounds {
    pub min: Vec3,
    pub max: Vec3,
}

/// One chunk's extracted surface mesh.
#[derive(Clone, Debug)]
pub struct FieldChunkMesh {
    pub key: FieldChunkKey,
    pub mesh: BuiltMesh,
    pub bounds: FieldMeshBounds,
}

/// Chunk edge length in cells (exclusive upper bound of the cell grid), shared with CPU `Volume`.
pub const FIELD_CHUNK_SIZE: i32 = 32;

/// Painted scalar field ready for sampling and extraction.
#[derive(Clone, Debug)]
pub struct PaintedField {
    pub bounds: FieldBounds,
    pub kernel: FieldKernel,
    pub grid: FieldGrid,
    pub density: Vec<f32>,
}

impl PaintedField {
    pub fn sample_density(&self, world: Vec3) -> f32 {
        self.grid.sample_density(&self.density, world)
    }

    pub fn reference_density(&self, world: Vec3) -> f32 {
        self.kernel.density_at(world, self.bounds)
    }

    pub fn extract_mesh(&self, ctx: &FieldGpuContext, color: Color) -> EngineResult<BuiltMesh> {
        ctx.extract_mesh(&self.grid, &self.density, color)
    }

    pub fn to_mesh(&self, ctx: &FieldGpuContext, color: Color) -> EngineResult<Mesh> {
        Ok(built_to_mesh(self.extract_mesh(ctx, color)?))
    }
}

/// GPU scalar-field session configuration.
#[derive(Clone, Debug)]
pub struct GpuField {
    voxel_size: f32,
    bounds: Option<FieldBounds>,
    kernel: Option<FieldKernel>,
    painted: Option<PaintedField>,
}

impl GpuField {
    pub fn new(voxel_size: f32) -> Self {
        Self::try_new(voxel_size).expect("voxel_size must be finite and > 0")
    }

    pub fn try_new(voxel_size: f32) -> EngineResult<Self> {
        if !voxel_size.is_finite() || voxel_size <= 0.0 {
            return Err(EngineError::InvalidValue(
                "gpu_field voxel_size must be finite and > 0".into(),
            ));
        }
        Ok(Self {
            voxel_size,
            bounds: None,
            kernel: None,
            painted: None,
        })
    }

    pub fn voxel_size(&self) -> f32 {
        self.voxel_size
    }

    pub fn bounds(&self) -> Option<FieldBounds> {
        self.bounds
    }

    pub fn kernel(&self) -> Option<FieldKernel> {
        self.kernel.clone()
    }

    pub fn painted(&self) -> Option<&PaintedField> {
        self.painted.as_ref()
    }

    /// Record bounds and kernel for the next paint pass.
    pub fn set_session(&mut self, bounds: FieldBounds, kernel: FieldKernel) -> EngineResult<()> {
        if bounds.voxel_size != self.voxel_size {
            return Err(EngineError::InvalidValue(format!(
                "field bounds voxel_size {} does not match session {}",
                bounds.voxel_size, self.voxel_size
            )));
        }
        self.bounds = Some(bounds);
        self.kernel = Some(kernel);
        self.painted = None;
        Ok(())
    }

    /// Fill density samples over the active bounds on the GPU.
    pub fn paint(&mut self, ctx: &FieldGpuContext) -> EngineResult<&PaintedField> {
        let bounds = self.bounds.ok_or_else(|| {
            EngineError::InvalidValue("gpu_field::paint requires set_session bounds".into())
        })?;
        let kernel = self.kernel.as_ref().ok_or_else(|| {
            EngineError::InvalidValue("gpu_field::paint requires set_session kernel".into())
        })?;
        let grid = FieldGrid::from_bounds(bounds);
        let density = ctx.paint_density(&grid, kernel)?;
        self.painted = Some(PaintedField {
            bounds,
            kernel: kernel.clone(),
            grid,
            density,
        });
        Ok(self.painted.as_ref().expect("just stored painted field"))
    }

    /// Extract the full painted volume to a mesh via GPU marching cubes.
    pub fn extract_mesh(&self, ctx: &FieldGpuContext, color: Color) -> EngineResult<BuiltMesh> {
        let painted = self.painted.as_ref().ok_or_else(|| {
            EngineError::InvalidValue("gpu_field::extract_mesh requires paint first".into())
        })?;
        painted.extract_mesh(ctx, color)
    }

    /// Chunk keys covering the active bounds (for future chunked streaming).
    pub fn chunk_keys(&self) -> EngineResult<Vec<FieldChunkKey>> {
        let bounds = self.bounds.ok_or_else(|| {
            EngineError::InvalidValue("gpu_field::chunk_keys requires set_session bounds".into())
        })?;
        let cell = self.voxel_size * FIELD_CHUNK_SIZE as f32;
        let min_key = IVec3::new(
            (bounds.min.x / cell).floor() as i32,
            (bounds.min.y / cell).floor() as i32,
            (bounds.min.z / cell).floor() as i32,
        );
        let max_key = IVec3::new(
            (bounds.max.x / cell).floor() as i32,
            (bounds.max.y / cell).floor() as i32,
            (bounds.max.z / cell).floor() as i32,
        );
        let mut keys = Vec::new();
        for y in min_key.y..=max_key.y {
            for z in min_key.z..=max_key.z {
                for x in min_key.x..=max_key.x {
                    keys.push(FieldChunkKey(IVec3::new(x, y, z)));
                }
            }
        }
        Ok(keys)
    }
}

fn built_to_mesh(built: BuiltMesh) -> Mesh {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_rejects_inverted_axes() {
        let err = FieldBounds::try_new(Vec3::ONE, Vec3::ZERO, 0.5).unwrap_err();
        assert!(matches!(err, EngineError::InvalidValue(_)));
    }

    #[test]
    fn chunk_keys_cover_a_small_box() {
        let mut field = GpuField::new(0.5);
        let bounds = FieldBounds::try_new(Vec3::ZERO, Vec3::splat(32.0), 0.5).unwrap();
        field
            .set_session(bounds, FieldKernel::demo_sphere_void(bounds))
            .unwrap();
        let keys = field.chunk_keys().unwrap();
        assert!(!keys.is_empty());
        assert!(keys.contains(&FieldChunkKey(IVec3::ZERO)));
    }
}
