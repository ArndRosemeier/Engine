//! CPU-side density sampling that must match the active GPU kernel.

use glam::Vec3;

use super::field::PaintedField;
use crate::error::EngineResult;

/// Sample signed density at a world-space point.
///
/// Positive = solid, negative = empty; the surface lies at zero.
pub trait DensityMirror {
    fn sample_density(&self, world: Vec3) -> EngineResult<f32>;
}

impl DensityMirror for PaintedField {
    fn sample_density(&self, world: Vec3) -> EngineResult<f32> {
        Ok(self.sample_density(world))
    }
}

/// Portable mirror backed by the same parameter blob passed to the GPU kernel.
#[derive(Clone, Debug)]
pub struct ParamsMirror {
    painted: PaintedField,
}

impl ParamsMirror {
    pub fn new(painted: PaintedField) -> Self {
        Self { painted }
    }
}

impl DensityMirror for ParamsMirror {
    fn sample_density(&self, world: Vec3) -> EngineResult<f32> {
        Ok(self.painted.sample_density(world))
    }
}
