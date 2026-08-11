use crate::error::{EngineError, EngineResult};
use glam::{Vec3, Vec4};

/// Display color stored as linear RGBA in 0..1 (`a = 1` is fully opaque).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Friendly byte RGB (0..=255), opaque.
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 255)
    }

    /// Friendly byte RGBA (0..=255).
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    /// Float RGB in 0..1, opaque. Rejects non-finite values.
    pub fn rgb01(r: f32, g: f32, b: f32) -> EngineResult<Self> {
        Self::rgba01(r, g, b, 1.0)
    }

    /// Float RGBA in 0..1. Rejects non-finite values.
    pub fn rgba01(r: f32, g: f32, b: f32, a: f32) -> EngineResult<Self> {
        if !r.is_finite() || !g.is_finite() || !b.is_finite() || !a.is_finite() {
            return Err(EngineError::InvalidColor(
                "color channels must be finite".into(),
            ));
        }
        Ok(Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
            a: a.clamp(0.0, 1.0),
        })
    }

    /// Like [`rgb01`] but panics on invalid input (internal/tests only).
    pub fn rgb01_unchecked(r: f32, g: f32, b: f32) -> Self {
        Self::rgb01(r, g, b).expect("invalid color")
    }

    pub fn with_alpha(mut self, a: f32) -> Self {
        self.a = a.clamp(0.0, 1.0);
        self
    }

    pub fn is_transparent(self) -> bool {
        self.a < 0.999
    }

    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.r, self.g, self.b)
    }

    pub fn to_vec4(self) -> Vec4 {
        Vec4::new(self.r, self.g, self.b, self.a)
    }
}

impl From<Color> for Vec3 {
    fn from(c: Color) -> Self {
        c.to_vec3()
    }
}

impl From<Color> for Vec4 {
    fn from(c: Color) -> Self {
        c.to_vec4()
    }
}

/// Shorthand for [`Color::rgb`].
pub fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::rgb(r, g, b)
}

/// Shorthand for [`Color::rgba`].
pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Color {
    Color::rgba(r, g, b, a)
}
