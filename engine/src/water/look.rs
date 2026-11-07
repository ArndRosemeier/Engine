//! How water looks: colour of the body, clarity, surface detail, foam.

/// Optical and surface settings for drawing [`super::Water`].
///
/// Colour comes from physics rather than a painted tint: light travelling
/// through water loses red first, then green (`absorption`), and some of it
/// scatters back toward the eye (`scatter`). Shallow water is clear, deep
/// water takes the scattered colour, and the same settings work above and
/// below the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterLook {
    /// Light lost per metre for red, green and blue (1/m).
    pub absorption: [f32; 3],
    /// Light scattered back per metre (1/m), as a colour: sets the hue of
    /// deep water and the underwater fog.
    pub scatter: [f32; 3],
    /// Surface roughness for the sun highlight and reflections.
    pub roughness: f32,
    /// Strength of the small waves drawn on top of the simulated surface
    /// (0 = mirror). Wind ([`crate::world::World::set_wind`]) adds to it.
    pub ripples: f32,
    /// Length of the longest drawn ripples, metres.
    pub ripple_length_m: f32,
    /// How far the view through the surface bends.
    pub refraction: f32,
    /// Foam from breaking fronts, fast flow and shorelines (0 = none).
    pub foam: f32,
    /// Strength of the light patterns (caustics) on lit underwater ground.
    pub caustics: f32,
}

impl Default for WaterLook {
    fn default() -> Self {
        Self::lake()
    }
}

impl WaterLook {
    /// Clear, slightly green fresh water.
    pub fn lake() -> Self {
        Self {
            absorption: [0.42, 0.10, 0.08],
            scatter: [0.008, 0.028, 0.030],
            roughness: 0.04,
            ripples: 0.6,
            ripple_length_m: 4.0,
            refraction: 1.0,
            foam: 1.0,
            caustics: 1.0,
        }
    }

    /// Deep blue sea water.
    pub fn ocean() -> Self {
        Self {
            absorption: [0.45, 0.07, 0.035],
            scatter: [0.004, 0.020, 0.040],
            roughness: 0.05,
            ripples: 1.0,
            ripple_length_m: 9.0,
            refraction: 1.0,
            foam: 1.2,
            caustics: 1.0,
        }
    }

    /// Turquoise water over pale sand.
    pub fn tropical() -> Self {
        Self {
            absorption: [0.38, 0.05, 0.04],
            scatter: [0.006, 0.040, 0.045],
            roughness: 0.035,
            ripples: 0.7,
            ripple_length_m: 5.0,
            refraction: 1.0,
            foam: 1.0,
            caustics: 1.4,
        }
    }

    /// Silty, brownish river water.
    pub fn river() -> Self {
        Self {
            absorption: [0.9, 0.8, 0.9],
            scatter: [0.06, 0.05, 0.03],
            roughness: 0.06,
            ripples: 0.8,
            ripple_length_m: 2.5,
            refraction: 0.7,
            foam: 1.3,
            caustics: 0.3,
        }
    }

    /// Dark peat-stained swamp water.
    pub fn swamp() -> Self {
        Self {
            absorption: [1.6, 1.4, 2.0],
            scatter: [0.025, 0.022, 0.008],
            roughness: 0.03,
            ripples: 0.25,
            ripple_length_m: 2.0,
            refraction: 0.5,
            foam: 0.4,
            caustics: 0.1,
        }
    }
}
