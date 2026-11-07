//! Lamps: point lights and spotlights placed in the world (fires, street
//! lamps, lanterns, torches on walls, headlights), with optional shadows.
//!
//! ```ignore
//! let fire = world.add_lamp(
//!     Lamp::point((4.0, 1.0, 2.0), rgb(255, 150, 70), 6.0, 12.0).shadows(true).flicker(0.4),
//! );
//! world.set_lamp_position(fire, (4.0, 1.2, 2.0))?; // carry it around
//! ```
//!
//! Hundreds of lamps are fine: each pixel only looks at the lamps whose reach
//! covers it. Shadows cost a little more, so only the lamps that ask for them
//! and matter most this frame (bright and near the camera) get them; see
//! [`crate::world::World::set_lamp_shadow_budget`].

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use glam::Vec3;

/// Most lamps that can have shadows in one frame.
pub const MAX_SHADOW_LAMPS: usize = 8;

/// Handle to a lamp from [`crate::world::World::add_lamp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LampId(pub(crate) u32);

/// Which way a lamp shines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LampShape {
    /// Every direction: bulbs, fires, lanterns.
    Point,
    /// A cone: spotlights, headlights, flashlights. `angle_deg` is the full
    /// opening angle; `softness` (0..1) how much of it fades out at the rim.
    Spot {
        direction: Vec3,
        angle_deg: f32,
        softness: f32,
    },
}

/// A light placed in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lamp {
    pub position: Vec3,
    pub color: Color,
    /// Brightness one metre away, on the same scale as the sun's brightness
    /// (a midday sun is about 3; a camp fire 4 to 8, a street lamp 10 to 20).
    pub intensity: f32,
    /// Metres at which the light has faded to nothing. Smaller is cheaper.
    pub range: f32,
    pub shape: LampShape,
    /// Cast shadows (walls, trees, people block it). Costs a few draws per
    /// frame per lamp, so leave it off for small decorative lights.
    pub shadows: bool,
    /// 0 steady, 1 wildly flickering, like a fire in the wind.
    pub flicker: f32,
}

impl Lamp {
    /// A light shining in every direction.
    pub fn point(position: impl Into<Vec3>, color: Color, intensity: f32, range: f32) -> Self {
        Self {
            position: position.into(),
            color,
            intensity,
            range,
            shape: LampShape::Point,
            shadows: false,
            flicker: 0.0,
        }
    }

    /// A cone of light along `direction`, `angle_deg` wide.
    pub fn spot(
        position: impl Into<Vec3>,
        direction: impl Into<Vec3>,
        angle_deg: f32,
        color: Color,
        intensity: f32,
        range: f32,
    ) -> Self {
        Self {
            shape: LampShape::Spot {
                direction: direction.into(),
                angle_deg,
                softness: 0.3,
            },
            ..Self::point(position, color, intensity, range)
        }
    }

    /// A camp fire: warm, flickering, casting shadows.
    pub fn fire(position: impl Into<Vec3>) -> Self {
        Self::point(position, Color::rgb(255, 147, 64), 6.0, 14.0)
            .shadows(true)
            .flicker(0.45)
    }

    /// A street lamp's bulb: a bright warm white with a long reach.
    pub fn street_lamp(position: impl Into<Vec3>) -> Self {
        Self::point(position, Color::rgb(255, 214, 170), 14.0, 18.0).shadows(true)
    }

    pub fn shadows(mut self, on: bool) -> Self {
        self.shadows = on;
        self
    }

    pub fn flicker(mut self, amount: f32) -> Self {
        self.flicker = amount;
        self
    }

    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn range(mut self, range: f32) -> Self {
        self.range = range;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("lamp {what}")));
        if !self.position.is_finite() {
            return bad("position must be finite");
        }
        if !self.intensity.is_finite() || self.intensity < 0.0 {
            return bad("intensity must be finite and >= 0");
        }
        if !self.range.is_finite() || !(0.1..=1000.0).contains(&self.range) {
            return bad("range must be in 0.1..=1000 metres");
        }
        if !self.flicker.is_finite() || !(0.0..=1.0).contains(&self.flicker) {
            return bad("flicker must be in 0..=1");
        }
        if let LampShape::Spot {
            direction,
            angle_deg,
            softness,
        } = self.shape
        {
            if !direction.is_finite() || direction.length_squared() < 1e-12 {
                return bad("spot direction must be a non-zero vector");
            }
            if !angle_deg.is_finite() || !(1.0..=170.0).contains(&angle_deg) {
                return bad("spot angle must be in 1..=170 degrees");
            }
            if !softness.is_finite() || !(0.0..=1.0).contains(&softness) {
                return bad("spot softness must be in 0..=1");
            }
        }
        Ok(())
    }

    /// Brightness this frame, with flicker applied (`time` in seconds, `seed`
    /// keeps neighbouring fires out of step).
    pub(crate) fn intensity_at(&self, time: f32, seed: u32) -> f32 {
        if self.flicker <= 0.0 {
            return self.intensity;
        }
        let s = seed as f32 * 1.618;
        // A few incommensurate waves read as a fire, not a strobe.
        let wave = (time * 7.3 + s).sin() * 0.45
            + (time * 13.1 + s * 2.3).sin() * 0.3
            + (time * 23.7 + s * 0.7).sin() * 0.25;
        self.intensity * (1.0 - self.flicker * 0.35 * (1.0 - wave))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid_and_flicker_stays_positive() {
        for lamp in [
            Lamp::fire(Vec3::ZERO),
            Lamp::street_lamp(Vec3::Y * 5.0),
            Lamp::spot(Vec3::ZERO, -Vec3::Y, 40.0, Color::WHITE, 10.0, 20.0),
        ] {
            lamp.validate().unwrap();
            for i in 0..200 {
                let v = lamp.intensity_at(i as f32 * 0.037, 3);
                assert!(v > 0.0 && v <= lamp.intensity * 1.01, "{v}");
            }
        }
        assert!(Lamp::point(Vec3::ZERO, Color::WHITE, 1.0, 0.0)
            .validate()
            .is_err());
    }
}
