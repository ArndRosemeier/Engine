//! Rock-solid ground + water column sample.
//!
//! Land, water draw, feet, and tests must all use [`SurfaceSample::is_wet`].
//! The shoreline is the iso-surface where wetness flips — not a separate mesh.

use std::sync::Arc;

/// Bed must sit this far below `water_top` for a column to count as wet.
/// Draw, walk, and contract tests share this constant.
pub const WATER_CLEARANCE: f32 = 0.02;

/// One column of the continuous surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    /// Solid ground / bed height (metres).
    pub ground: f32,
    /// Water sheet height. Use `f32::NEG_INFINITY` when no body applies.
    pub water_top: f32,
}

impl SurfaceSample {
    pub fn dry(ground: f32) -> Self {
        Self {
            ground,
            water_top: f32::NEG_INFINITY,
        }
    }

    pub fn wet_body(ground: f32, water_top: f32) -> Self {
        Self { ground, water_top }
    }

    /// Single wetness predicate for draw / walk / tests.
    #[inline]
    pub fn is_wet(self) -> bool {
        self.water_top.is_finite() && (self.water_top - self.ground) >= WATER_CLEARANCE
    }

    /// How far the bed sits above the required clearance floor (0 = ok).
    #[inline]
    pub fn contract_error(self) -> f32 {
        if !self.is_wet() {
            return 0.0;
        }
        (self.ground - (self.water_top - WATER_CLEARANCE)).max(0.0)
    }

    /// Walkable height: water sheet when wet, else ground.
    #[inline]
    pub fn walk_height(self) -> f32 {
        if self.is_wet() {
            self.water_top
        } else {
            self.ground
        }
    }
}

/// Pluggable continent / demo height source (CPU feet + chunk bake).
pub trait SurfaceSource: Send + Sync {
    fn sample(&self, x: f32, z: f32) -> SurfaceSample;
}

impl<F> SurfaceSource for F
where
    F: Fn(f32, f32) -> SurfaceSample + Send + Sync,
{
    fn sample(&self, x: f32, z: f32) -> SurfaceSample {
        self(x, z)
    }
}

pub type SharedSurface = Arc<dyn SurfaceSource>;

/// Dense-grid contract check: every wet column must clear the bed floor.
pub fn max_contract_error(source: &dyn SurfaceSource, samples: &[(f32, f32)]) -> f32 {
    let mut max_err = 0.0_f32;
    for &(x, z) in samples {
        max_err = max_err.max(source.sample(x, z).contract_error());
    }
    max_err
}
