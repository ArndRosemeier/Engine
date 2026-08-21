use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::space::{GlobalPosition, RenderOrigin};
use glam::{Mat4, Vec3};

/// Friendly transform: position + yaw/pitch (degrees) + scale + tint.
///
/// [`Self::scale`] is uniform. [`Self::stretch`] is the per-axis factor, default
/// `Vec3::ONE`, so a foundation skirt can be one unit cube instanced at many
/// footprint × height sizes without a unique mesh per house.
///
/// [`Self::tint`] multiplies vertex color × albedo in the mesh shader (linear
/// RGBA). White leaves the authored look unchanged — used for house paint
/// variety and other shared-mesh recolors without cloning GPU prototypes.
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
    pub tint: Color,
}

impl Default for Place {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            scale: 1.0,
            stretch: Vec3::ONE,
            tint: Color::WHITE,
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

    pub fn with_tint(mut self, tint: Color) -> Self {
        self.tint = tint;
        self
    }

    pub fn tint(mut self, tint: Color) -> EngineResult<Self> {
        ensure_finite(tint.r, "tint.r")?;
        ensure_finite(tint.g, "tint.g")?;
        ensure_finite(tint.b, "tint.b")?;
        ensure_finite(tint.a, "tint.a")?;
        self.tint = tint;
        Ok(self)
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

    /// Transform + tint for an instanced draw.
    pub fn to_instance(self) -> MeshInstance {
        MeshInstance {
            transform: self.to_matrix(),
            tint: self.tint,
        }
    }
}

/// One GPU instance: model matrix plus linear RGBA multiply tint.
#[derive(Clone, Copy, Debug)]
pub struct MeshInstance {
    pub transform: Mat4,
    pub tint: Color,
}

impl MeshInstance {
    pub fn from_matrix(transform: Mat4) -> Self {
        Self {
            transform,
            tint: Color::WHITE,
        }
    }

    pub fn with_tint(mut self, tint: Color) -> Self {
        self.tint = tint;
        self
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
    pub tint: Color,
}

impl GlobalPlace {
    pub fn at(position: GlobalPosition) -> Self {
        Self {
            position,
            yaw_degrees: 0.0,
            pitch_degrees: 0.0,
            scale: 1.0,
            stretch: Vec3::ONE,
            tint: Color::WHITE,
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

    pub fn with_tint(mut self, tint: Color) -> Self {
        self.tint = tint;
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
        ensure_finite(self.tint.r, "tint.r")?;
        ensure_finite(self.tint.g, "tint.g")?;
        ensure_finite(self.tint.b, "tint.b")?;
        ensure_finite(self.tint.a, "tint.a")?;
        Ok(Place {
            position: self.position.to_render(origin)?.vec3(),
            yaw_degrees: self.yaw_degrees,
            pitch_degrees: self.pitch_degrees,
            scale: self.scale,
            stretch: self.stretch,
            tint: self.tint,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;

    #[test]
    fn place_tint_defaults_white_and_survives_to_instance() {
        let p = Place::new(1.0, 2.0, 3.0).with_yaw_deg(45.0);
        assert_eq!(p.tint, Color::WHITE);
        let want = Color::rgb(200, 180, 160);
        let tinted = p.with_tint(want);
        let inst = tinted.to_instance();
        assert!((inst.tint.r - want.r).abs() < 1e-5);
        assert_eq!(inst.transform, tinted.to_matrix());
    }

    #[test]
    fn instance_raw_packs_tint_after_the_model_matrix() {
        use crate::mesh::InstanceRaw;
        use bytemuck::bytes_of;
        use glam::Mat4;

        assert_eq!(std::mem::size_of::<InstanceRaw>(), 80);
        let red = Color::rgb01_unchecked(1.0, 0.25, 0.125);
        let raw = InstanceRaw::from_matrix_tint(Mat4::IDENTITY, red);
        let bytes = bytes_of(&raw);
        // mat4 = 64 bytes, then linear RGBA.
        assert_eq!(&bytes[64..80], bytemuck::bytes_of(&[1.0_f32, 0.25, 0.125, 1.0]));
        let white = InstanceRaw::from_matrix(Mat4::IDENTITY);
        assert_ne!(&bytes[64..80], &bytemuck::bytes_of(&white)[64..80]);
    }

    /// Mesh shader contract: `base = vertex_color * instance_tint * albedo`.
    #[test]
    fn shader_multiply_separates_instance_tints() {
        let albedo = [0.85_f32, 0.80, 0.72]; // cream plaster sample
        let vertex = [1.0_f32, 1.0, 1.0];
        let warm = Color {
            r: 1.0,
            g: 0.82,
            b: 0.62,
            a: 1.0,
        };
        let cool = Color {
            r: 0.72,
            g: 0.78,
            b: 0.92,
            a: 1.0,
        };
        let shade = |tint: Color| {
            [
                vertex[0] * tint.r * albedo[0],
                vertex[1] * tint.g * albedo[1],
                vertex[2] * tint.b * albedo[2],
            ]
        };
        let a = shade(warm);
        let b = shade(cool);
        let dist = (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs();
        assert!(
            dist > 0.25,
            "palettes must diverge on cream albedo (dist={dist})"
        );
    }
}
