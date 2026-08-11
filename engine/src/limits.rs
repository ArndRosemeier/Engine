/// Resource caps that protect against accidental DoS (huge paints, huge models).
#[derive(Clone, Debug)]
pub struct EngineLimits {
    /// Max density samples touched by a single volume paint operation.
    pub max_volume_samples_per_paint: u64,
    /// Max decoded glTF buffer bytes.
    pub max_gltf_buffer_bytes: u64,
    /// Max triangles accepted from one model.
    pub max_model_triangles: u64,
    /// Max GPU instances in one `spawn_many` / instanced spawn.
    pub max_instances_per_spawn: u64,
    /// Max joints in one skinned model.
    pub max_joints: u32,
}

impl Default for EngineLimits {
    fn default() -> Self {
        Self {
            max_volume_samples_per_paint: 8_000_000,
            max_gltf_buffer_bytes: 64 * 1024 * 1024,
            max_model_triangles: 2_000_000,
            max_instances_per_spawn: 100_000,
            max_joints: 128,
        }
    }
}

impl EngineLimits {
    pub fn permissive() -> Self {
        Self {
            max_volume_samples_per_paint: u64::MAX / 4,
            max_gltf_buffer_bytes: u64::MAX / 4,
            max_model_triangles: u64::MAX / 4,
            max_instances_per_spawn: u64::MAX / 4,
            max_joints: 256,
        }
    }
}
