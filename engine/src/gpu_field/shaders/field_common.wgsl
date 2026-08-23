// Shared helpers for GPU scalar-field evaluation.
// Engine-owned; user density kernels are stitched in at compile time.

struct FieldParamsHeader {
    bounds_min: vec3<f32>,
    voxel_size: f32,
    bounds_max: vec3<f32>,
    _pad: f32,
}

// Chunk-local index → world position (metres).
fn field_world_pos(chunk_origin: vec3<i32>, local: vec3<u32>, voxel_size: f32) -> vec3<f32> {
    let g = chunk_origin + vec3<i32>(local);
    return vec3<f32>(g) * voxel_size;
}
