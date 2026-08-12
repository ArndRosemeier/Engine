//! Rock-solid ground + water column sample.
//!
//! Wetness is part of the type, not a sentinel float: a column either carries a
//! [`WaterSurface`] or it does not. A water body may only be constructed when
//! the bed sits at least [`WATER_CLEARANCE`] below the sheet, so draw, walk, and
//! collision can never disagree about where the shoreline is.

use crate::error::{EngineError, EngineResult};
use std::sync::Arc;

/// Bed must sit this far below the sheet for a column to hold water.
/// Draw, walk, and generation share this constant.
pub const WATER_CLEARANCE: f32 = 0.02;

/// Standing water on top of a column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSurface {
    top: f32,
    depth: f32,
}

impl WaterSurface {
    /// Sheet height in metres.
    #[inline]
    pub fn top(self) -> f32 {
        self.top
    }

    /// Distance from bed to sheet — always `>= WATER_CLEARANCE`.
    #[inline]
    pub fn depth(self) -> f32 {
        self.depth
    }
}

/// One column of the continuous surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    ground: f32,
    water: Option<WaterSurface>,
}

impl SurfaceSample {
    /// Dry column. Panics on a non-finite height — a broken generator must be
    /// loud here, not silently produce holes in the mesh.
    pub fn dry(ground: f32) -> Self {
        Self::try_dry(ground).expect("dry surface sample")
    }

    /// Column under standing water.
    ///
    /// Panics unless the bed clears the sheet by [`WATER_CLEARANCE`]. Callers
    /// carve the bed first; they must not hand over an unresolved column and
    /// hope the sample decides for them.
    pub fn wet(ground: f32, water_top: f32) -> Self {
        Self::try_wet(ground, water_top).expect("wet surface sample")
    }

    pub fn try_dry(ground: f32) -> EngineResult<Self> {
        if !ground.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "ground height must be finite, got {ground}"
            )));
        }
        Ok(Self {
            ground,
            water: None,
        })
    }

    pub fn try_wet(ground: f32, water_top: f32) -> EngineResult<Self> {
        if !ground.is_finite() || !water_top.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "wet column heights must be finite, got ground {ground} / top {water_top}"
            )));
        }
        let depth = water_top - ground;
        if depth < WATER_CLEARANCE {
            return Err(EngineError::InvalidValue(format!(
                "water sheet {water_top} sits {depth} above bed {ground}; \
                 carve the bed to at least {WATER_CLEARANCE} clearance or report the column dry"
            )));
        }
        Ok(Self {
            ground,
            water: Some(WaterSurface {
                top: water_top,
                depth,
            }),
        })
    }

    /// Solid ground / bed height in metres.
    #[inline]
    pub fn ground(self) -> f32 {
        self.ground
    }

    #[inline]
    pub fn water(self) -> Option<WaterSurface> {
        self.water
    }

    #[inline]
    pub fn water_top(self) -> Option<f32> {
        self.water.map(|w| w.top)
    }

    #[inline]
    pub fn depth(self) -> f32 {
        self.water.map(|w| w.depth).unwrap_or(0.0)
    }

    /// Single wetness predicate for draw / walk / tests.
    #[inline]
    pub fn is_wet(self) -> bool {
        self.water.is_some()
    }

    /// Walkable height: water sheet when wet, else ground.
    #[inline]
    pub fn walk_height(self) -> f32 {
        match self.water {
            Some(w) => w.top,
            None => self.ground,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_column_has_no_water() {
        let s = SurfaceSample::dry(12.5);
        assert!(!s.is_wet());
        assert_eq!(s.water_top(), None);
        assert_eq!(s.walk_height(), 12.5);
    }

    #[test]
    fn wet_column_reports_sheet_and_depth() {
        let s = SurfaceSample::wet(3.0, 9.0);
        assert!(s.is_wet());
        assert_eq!(s.water_top(), Some(9.0));
        assert_eq!(s.depth(), 6.0);
        assert_eq!(s.walk_height(), 9.0);
    }

    #[test]
    fn sheet_without_clearance_is_rejected() {
        let err = SurfaceSample::try_wet(1.0, 1.0 + WATER_CLEARANCE * 0.5).unwrap_err();
        assert!(matches!(err, EngineError::InvalidValue(_)));
    }

    #[test]
    fn non_finite_heights_are_rejected() {
        assert!(SurfaceSample::try_dry(f32::NAN).is_err());
        assert!(SurfaceSample::try_wet(0.0, f32::INFINITY).is_err());
    }
}
