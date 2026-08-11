use glam::{Mat4, Vec3};

/// Simple perspective camera. Y-up, angles in degrees where helpers use degrees.
#[derive(Clone, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_degrees: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(8.0, 6.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_degrees: 55.0,
            near: 0.1,
            far: 500.0,
        }
    }
}

impl Camera {
    pub fn look_at(eye: impl Into<Vec3>, target: impl Into<Vec3>) -> Self {
        Self {
            eye: eye.into(),
            target: target.into(),
            ..Self::default()
        }
    }

    /// Orbit around `target` at `distance`, with yaw/pitch in degrees.
    pub fn orbit(target: impl Into<Vec3>, distance: f32, yaw_degrees: f32, pitch_degrees: f32) -> Self {
        let target = target.into();
        let yaw = yaw_degrees.to_radians();
        let pitch = pitch_degrees.to_radians();
        let eye = target
            + Vec3::new(
                distance * yaw.cos() * pitch.cos(),
                distance * pitch.sin(),
                distance * yaw.sin() * pitch.cos(),
            );
        Self {
            eye,
            target,
            ..Self::default()
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_degrees.to_radians(), aspect, self.near, self.far)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}
