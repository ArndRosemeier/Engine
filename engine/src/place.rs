use crate::error::{EngineError, EngineResult};
use crate::space::{GlobalPosition, RenderOrigin};
use glam::{Mat4, Vec3};

/// Friendly transform: position + yaw/pitch (degrees) + scale.
///
/// [`Self::scale`] is uniform. [`Self::stretch`] is the per-axis factor, default
/// `Vec3::ONE`, so a foundation skirt can be one unit cube instanced at many
/// footprint × height sizes without a unique mesh per house.
///
/// Pitch is around local X after yaw. Zero keeps every existing yaw-only
/// caller upright; −90° lays a [`crate::mesh::Mesh::opening`] on the floor
/// facing +Y.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub position: Vec3,
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub scale: f32,
    pub stretch: Vec3,
}

impl Default for Place {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            scale: 1.0,
            stretch: Vec3::ONE,
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

    pub fn pitch_deg(mut self, degrees: f32) -> EngineResult<Self> {
        if !degrees.is_finite() {
            return Err(EngineError::InvalidValue("pitch must be finite".into()));
        }
        self.pitch_degrees = degrees;
        Ok(self)
    }

    pub fn with_pitch_deg(mut self, degrees: f32) -> Self {
        self.pitch_degrees = degrees;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_stretch(mut self, stretch: Vec3) -> Self {
        self.stretch = stretch;
        self
    }

    pub fn stretch(mut self, stretch: Vec3) -> EngineResult<Self> {
        ensure_finite3(stretch, "stretch")?;
        if stretch.x <= 0.0 || stretch.y <= 0.0 || stretch.z <= 0.0 {
            return Err(EngineError::InvalidValue(
                "stretch components must be > 0".into(),
            ));
        }
        self.stretch = stretch;
        Ok(self)
    }

    pub fn to_matrix(self) -> Mat4 {
        let t = Mat4::from_translation(self.position);
        let r = Mat4::from_rotation_y(self.yaw_degrees.to_radians())
            * Mat4::from_rotation_x(self.pitch_degrees.to_radians());
        let s = Mat4::from_scale(self.stretch * self.scale);
        t * r * s
    }
}

/// A [`Place`] anchored in absolute world metres.
///
/// Anchored transforms survive [`crate::world::World::set_render_origin`]: the
/// engine re-derives the render transform from this global anchor instead of
/// shifting an already-shifted position.
#[derive(Clone, Copy, Debug)]
pub struct GlobalPlace {
    pub position: GlobalPosition,
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
    pub scale: f32,
    pub stretch: Vec3,
}

impl GlobalPlace {
    pub fn at(position: GlobalPosition) -> Self {
        Self {
            position,
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            scale: 1.0,
            stretch: Vec3::ONE,
        }
    }

    pub fn with_yaw_deg(mut self, degrees: f32) -> Self {
        self.yaw_degrees = degrees;
        self
    }

    pub fn with_pitch_deg(mut self, degrees: f32) -> Self {
        self.pitch_degrees = degrees;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_stretch(mut self, stretch: Vec3) -> Self {
        self.stretch = stretch;
        self
    }

    /// Resolve into render space against the active origin.
    pub fn to_place(self, origin: RenderOrigin) -> EngineResult<Place> {
        if !self.yaw_degrees.is_finite() {
            return Err(EngineError::InvalidValue("yaw must be finite".into()));
        }
        if !self.pitch_degrees.is_finite() {
            return Err(EngineError::InvalidValue("pitch must be finite".into()));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(EngineError::InvalidValue(
                "scale must be finite and > 0".into(),
            ));
        }
        ensure_finite3(self.stretch, "stretch")?;
        if self.stretch.x <= 0.0 || self.stretch.y <= 0.0 || self.stretch.z <= 0.0 {
            return Err(EngineError::InvalidValue(
                "stretch components must be > 0".into(),
            ));
        }
        Ok(Place {
            position: self.position.to_render(origin)?.vec3(),
            yaw_degrees: self.yaw_degrees,
            pitch_degrees: self.pitch_degrees,
            scale: self.scale,
            stretch: self.stretch,
        })
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
