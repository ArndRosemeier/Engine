//! Built-in density kernels and CPU mirrors that must match WGSL.

use glam::Vec3;

use super::custom::{CustomFieldKernel, DensityReferenceFn};
use super::field::{FieldBounds, FieldParamsBlob};

/// Cave-agnostic kernels shipped with the engine for demos and tests.
#[derive(Clone, Debug)]
pub enum FieldKernel {
    /// Solid axis-aligned box with a spherical void carved out.
    DemoSphereVoid {
        sphere_center: Vec3,
        sphere_radius: f32,
    },
    /// Caller-supplied WGSL compute shader (`paint` entry) and uniform params.
    Custom(CustomFieldKernel),
}
impl FieldKernel {
    /// Default demo: 16 m cube with a 4 m radius void at the centre.
    pub fn demo_sphere_void(bounds: FieldBounds) -> Self {
        let centre = (bounds.min + bounds.max) * 0.5;
        Self::DemoSphereVoid {
            sphere_center: centre,
            sphere_radius: 4.0,
        }
    }
}

/// CPU reference for [`FieldKernel::DemoSphereVoid`]. Must match `PAINT_WGSL::demo_density`.
pub fn demo_sphere_density(
    world: Vec3,
    bounds: FieldBounds,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> f32 {
    if world.x < bounds.min.x
        || world.y < bounds.min.y
        || world.z < bounds.min.z
        || world.x > bounds.max.x
        || world.y > bounds.max.y
        || world.z > bounds.max.z
    {
        return -1.0;
    }
    let mut d = 1.0_f32;
    let sd = (world - sphere_center).length() - sphere_radius;
    d = d.min(sd);
    d
}

impl FieldKernel {
    pub fn density_at(&self, world: Vec3, bounds: FieldBounds) -> f32 {
        match self {
            Self::DemoSphereVoid {
                sphere_center,
                sphere_radius,
            } => demo_sphere_density(world, bounds, *sphere_center, *sphere_radius),
            Self::Custom(custom) => (custom.reference)(world, bounds, custom.params.as_bytes()),
        }
    }

    pub fn custom(
        shader_key: &'static str,
        wgsl: &'static str,
        params: FieldParamsBlob,
        uniform_size: u32,
        reference: DensityReferenceFn,
    ) -> Self {
        Self::Custom(CustomFieldKernel::new(
            shader_key,
            wgsl,
            params,
            uniform_size,
            reference,
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_interior_is_empty() {
        let bounds = FieldBounds::try_new(Vec3::ZERO, Vec3::splat(16.0), 0.5).unwrap();
        let centre = Vec3::splat(8.0);
        let d = demo_sphere_density(centre, bounds, centre, 4.0);
        assert!(d < 0.0, "sphere centre must be empty, got {d}");
    }

    #[test]
    fn box_corner_is_solid() {
        let bounds = FieldBounds::try_new(Vec3::ZERO, Vec3::splat(16.0), 0.5).unwrap();
        let centre = Vec3::splat(8.0);
        let d = demo_sphere_density(Vec3::ZERO, bounds, centre, 4.0);
        assert!(d > 0.0, "box corner must be solid, got {d}");
    }
}
