//! User-supplied WGSL density kernels (cave-agnostic plumbing).

use super::field::FieldParamsBlob;

/// CPU mirror for a custom GPU density kernel. Must match the WGSL `user_density` logic.
pub type DensityReferenceFn = fn(glam::Vec3, super::field::FieldBounds, &[u8]) -> f32;

/// Custom paint kernel compiled and cached by [`super::context::FieldGpuContext`].
#[derive(Clone, Debug)]
pub struct CustomFieldKernel {
    pub shader_key: &'static str,
    pub wgsl: &'static str,
    pub params: FieldParamsBlob,
    /// WGSL uniform struct size in bytes (multiple of 16).
    pub uniform_size: u32,
    pub reference: DensityReferenceFn,
}

impl CustomFieldKernel {
    pub fn new(
        shader_key: &'static str,
        wgsl: &'static str,
        params: FieldParamsBlob,
        uniform_size: u32,
        reference: DensityReferenceFn,
    ) -> Self {
        assert!(
            uniform_size > 0 && uniform_size.is_multiple_of(16),
            "custom field uniform_size must be a positive multiple of 16"
        );
        assert!(
            params.as_bytes().len() <= uniform_size as usize,
            "custom field params exceed uniform_size"
        );
        Self {
            shader_key,
            wgsl,
            params,
            uniform_size,
            reference,
        }
    }
}
