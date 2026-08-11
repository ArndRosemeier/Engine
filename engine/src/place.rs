use crate::error::{EngineError, EngineResult};
use glam::{Mat4, Vec3};

/// Friendly transform: position + yaw (degrees) + uniform scale.
///
/// Hides raw matrices from everyday scene setup.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub position: Vec3,
    pub yaw_degrees: f32,
    pub scale: f32,
}

impl Default for Place {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw_degrees: 0.0,
            scale: 1.0,
        }
    }
}

impl Place {
    pub fn at(x: f32, y: f32, z: f32) -> EngineResult<Self> {
        let position = Vec3::new(x, y, z);
        ensure_finite3(position, "position")?;
        Ok(Self {
            position,
            ..Self::default()
        })
    }

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self::at(x, y, z).expect("Place::new requires finite coordinates")
    }

    pub fn yaw_deg(mut self, degrees: f32) -> EngineResult<Self> {
        if !degrees.is_finite() {
            return Err(EngineError::InvalidValue("yaw must be finite".into()));
        }
        self.yaw_degrees = degrees;
        Ok(self)
    }

    pub fn scale(mut self, scale: f32) -> EngineResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(EngineError::InvalidValue(
                "scale must be finite and > 0".into(),
            ));
        }
        self.scale = scale;
        Ok(self)
    }

    pub fn with_yaw_deg(mut self, degrees: f32) -> Self {
        self.yaw_degrees = degrees;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn to_matrix(self) -> Mat4 {
        let t = Mat4::from_translation(self.position);
        let r = Mat4::from_rotation_y(self.yaw_degrees.to_radians());
        let s = Mat4::from_scale(Vec3::splat(self.scale));
        t * r * s
    }
}

pub(crate) fn ensure_finite3(v: Vec3, what: &str) -> EngineResult<()> {
    if !v.is_finite() {
        return Err(EngineError::InvalidValue(format!("{what} must be finite")));
    }
    Ok(())
}

pub(crate) fn ensure_finite(v: f32, what: &str) -> EngineResult<()> {
    if !v.is_finite() {
        return Err(EngineError::InvalidValue(format!("{what} must be finite")));
    }
    Ok(())
}
