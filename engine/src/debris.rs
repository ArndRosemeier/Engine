//! Flying pieces from [`crate::world::World::shatter`].
//!
//! Each piece becomes a rigid body in [`crate::physics`]: pieces tumble,
//! hit each other and anything else with a body, pile up on the ground,
//! float or sink in water by their density, and sink away after
//! [`Shatter::keep_seconds`].

use crate::error::{EngineError, EngineResult};
use crate::fracture::FractureSettings;

/// How an object breaks and how hard its pieces fly.
#[derive(Clone, Copy, Debug)]
pub struct Shatter {
    pub pieces: usize,
    /// Outward speed of the pieces nearest the impact, in m/s.
    pub speed: f32,
    /// Seconds the rubble stays before it sinks away; `None` keeps it.
    pub keep_seconds: Option<f32>,
    pub seed: u32,
    /// Colour multiplier for the broken inside faces.
    pub interior_shade: f32,
    /// Density relative to water: stone (2.4) sinks, wood (0.6) floats.
    pub density: f32,
}

impl Default for Shatter {
    fn default() -> Self {
        Self {
            pieces: 14,
            speed: 7.0,
            keep_seconds: Some(25.0),
            seed: 1,
            interior_shade: 0.6,
            density: 2.4,
        }
    }
}

impl Shatter {
    pub fn validate(&self) -> EngineResult<()> {
        if self.pieces == 0 || self.pieces > 64 {
            return Err(EngineError::InvalidValue(
                "shatter pieces must be 1..=64".into(),
            ));
        }
        if !self.speed.is_finite() || self.speed < 0.0 {
            return Err(EngineError::InvalidValue(
                "shatter speed must be finite and >= 0".into(),
            ));
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(EngineError::InvalidValue(
                "shatter density must be finite and > 0".into(),
            ));
        }
        if self.keep_seconds.is_some_and(|s| !s.is_finite() || s < 0.0) {
            return Err(EngineError::InvalidValue(
                "shatter keep_seconds must be finite and >= 0".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn fracture_settings(&self) -> FractureSettings {
        FractureSettings {
            pieces: self.pieces,
            seed: self.seed,
            interior_shade: self.interior_shade,
        }
    }
}
