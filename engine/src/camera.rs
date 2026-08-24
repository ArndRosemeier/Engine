use glam::{Mat4, Vec3};

/// How far a first-person view may tilt before the look direction would fold
/// onto the up axis.
pub const MAX_PITCH_DEGREES: f32 = 89.0;

/// Simple perspective camera. Y-up, angles in degrees where helpers use degrees.
#[derive(Clone, Debug)]
pub struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y_degrees: f32,
    near: f32,
    far: f32,
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
    pub(crate) fn from_parts(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_degrees: f32,
        near: f32,
        far: f32,
    ) -> Self {
        if !eye.is_finite() || !target.is_finite() || !up.is_finite() || up.length_squared() <= 0.0
        {
            panic!("camera parts must contain finite positions and a non-zero up vector");
        }
        if !(fov_y_degrees.is_finite() && fov_y_degrees > 0.0 && fov_y_degrees < 180.0) {
            panic!("camera field of view must be finite and in (0, 180), got {fov_y_degrees}");
        }
        if !(near.is_finite() && near > 0.0 && far.is_finite() && far > near) {
            panic!("camera clipping planes must be finite with 0 < near < far, got {near}, {far}");
        }
        Self {
            eye,
            target,
            up: up.normalize(),
            fov_y_degrees,
            near,
            far,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.eye
    }
    pub fn target(&self) -> Vec3 {
        self.target
    }
    pub fn up(&self) -> Vec3 {
        self.up
    }
    pub fn fov_y_degrees(&self) -> f32 {
        self.fov_y_degrees
    }
    pub fn near(&self) -> f32 {
        self.near
    }
    pub fn far(&self) -> f32 {
        self.far
    }

    pub(crate) fn with_lens(mut self, fov_y_degrees: f32, near: f32, far: f32) -> Self {
        let checked = Self::from_parts(self.eye, self.target, self.up, fov_y_degrees, near, far);
        self.fov_y_degrees = checked.fov_y_degrees;
        self.near = checked.near;
        self.far = checked.far;
        self
    }

    pub fn look_at(eye: impl Into<Vec3>, target: impl Into<Vec3>) -> Self {
        Self {
            eye: eye.into(),
            target: target.into(),
            ..Self::default()
        }
    }

    /// Orbit around `target` at `distance`, with yaw/pitch in degrees.
    pub fn orbit(
        target: impl Into<Vec3>,
        distance: f32,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) -> Self {
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
    pub fn follow(target: impl Into<Vec3>, yaw_degrees: f32, distance: f32, height: f32) -> Self {
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

    /// First-person camera: the eye *is* the viewpoint, no offset.
    ///
    /// `yaw_degrees` is the facing (0 = +Z), `pitch_degrees` is positive
    /// looking up. Pitch is clamped short of straight up/down so the view
    /// direction never becomes parallel to `up`.
    pub fn first_person(eye: impl Into<Vec3>, yaw_degrees: f32, pitch_degrees: f32) -> Self {
        let eye = eye.into();
        Self {
            eye,
            target: eye + Self::direction(yaw_degrees, pitch_degrees),
            far: 800.0,
            ..Self::default()
        }
    }

    /// Unit look direction for a yaw/pitch pair in degrees.
    pub fn direction(yaw_degrees: f32, pitch_degrees: f32) -> Vec3 {
        let yaw = yaw_degrees.to_radians();
        let pitch = pitch_degrees
            .clamp(-MAX_PITCH_DEGREES, MAX_PITCH_DEGREES)
            .to_radians();
        let flat = pitch.cos();
        Vec3::new(yaw.sin() * flat, pitch.sin(), yaw.cos() * flat)
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

    /// Perspective projection with **reversed depth**: near maps to 1, far to 0.
    ///
    /// Swapping the planes is what makes a horizon-scale `far` usable. A
    /// conventional mapping spends almost all of a float's precision in the
    /// first few metres, so at `far = 40 km` distant hills z-fight into mush;
    /// reversed depth pairs the float's dense range near zero with the far
    /// distance and holds up over the whole view. The pipelines compare with
    /// [`wgpu::CompareFunction::Greater`] and the pass clears depth to 0 to
    /// match — all three have to agree or nothing draws.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_degrees.to_radians(), aspect, self.far, self.near)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}
