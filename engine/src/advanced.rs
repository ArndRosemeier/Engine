//! Power-user escape hatch. Prefer the friendly prelude API when possible.

pub use crate::gpu_field::{
    demo_sphere_density, extract_mesh_cpu, gpu_tests_enabled, CustomFieldKernel, DensityMirror,
    DensityReferenceFn, FieldBounds, FieldChunkKey, FieldChunkMesh, FieldGpuContext, FieldGrid,
    FieldKernel, FieldMeshBounds, FieldParamsBlob, GpuField, PaintedField, ParamsMirror,
    FIELD_CHUNK_SIZE, MAX_FIELD_UNIFORM_BYTES,
};
pub use crate::limits::EngineLimits;
pub use crate::mesh::BuiltMesh;
pub use crate::proc::{carve_tunnel_x, terrain, Noise, TerrainRules as VolumeTerrainRules};
pub use crate::volume::{ChunkStreamer, Volume, CHUNK_SIZE};
