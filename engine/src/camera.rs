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

    /// Third-person camera behind a walker.
    ///
    /// `yaw_degrees` is the walker's facing (0 = +Z). Camera sits behind and above.
    pub fn follow(
        target: impl Into<Vec3>,
        yaw_degrees: f32,
        distance: f32,
        height: f32,
    ) -> Self {
        let target = target.into();
        let yaw = yaw_degrees.to_radians();
        let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let eye = target - forward * distance + Vec3::Y * height;
        let look = target + Vec3::Y * 1.4;
        Self {
            eye,
            target: look,
            far: 800.0,
            ..Self::default()
        }
    }

    /// Unit facing vector on XZ for a yaw in degrees (0 = +Z).
    pub fn facing_xz(yaw_degrees: f32) -> Vec3 {
        let yaw = yaw_degrees.to_radians();
        Vec3::new(yaw.sin(), 0.0, yaw.cos())
    }

    /// Screen-right strafe vector on XZ for a yaw in degrees.
    ///
    /// Matches glam/`look_at_rh` (forward × up): when facing +Z, that is −X, so
    /// pressing D moves toward the right edge of a third-person follow view.
    pub fn right_xz(yaw_degrees: f32) -> Vec3 {
        let forward = Self::facing_xz(yaw_degrees);
        Vec3::new(-forward.z, 0.0, forward.x)
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
