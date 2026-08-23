//! GPU scalar-field paint and isosurface extraction.
//!
//! Cave-agnostic infrastructure: callers supply density kernels; this module owns
//! chunked GPU storage, compute dispatch, mesh readback, and CPU verification helpers.

mod context;
mod custom;
mod extract_cpu;
mod field;
mod grid;
mod kernel;
mod mirror;
mod verify;

pub mod shaders {
    pub const FIELD_COMMON_WGSL: &str = include_str!("shaders/field_common.wgsl");
    pub const MARCHING_CUBES_WGSL: &str = include_str!("shaders/marching_cubes.wgsl");
}

pub use context::{FieldGpuContext, MAX_FIELD_UNIFORM_BYTES};
pub use custom::{CustomFieldKernel, DensityReferenceFn};
pub use extract_cpu::extract_mesh_cpu;
pub use field::{
    FieldBounds, FieldChunkKey, FieldChunkMesh, FieldMeshBounds, FieldParamsBlob, GpuField,
    PaintedField, FIELD_CHUNK_SIZE,
};
pub use grid::FieldGrid;
pub use kernel::{demo_sphere_density, FieldKernel};
pub use mirror::{DensityMirror, ParamsMirror};
pub use verify::gpu_tests_enabled;
