//! A ready-made free-fly camera for tools, viewers and prototypes.
//!
//! ```no_run
//! use engine::prelude::*;
//!
//! fn main() -> EngineResult<()> {
//!     let mut camera = FlyCamera::new((0.0, 2.0, -10.0), 0.0, 0.0);
//!     Engine::run("fly", move |world, frame| {
//!         camera.update(world, frame);
//!         Ok(())
//!     })
//! }
//! ```
//!
//! Controls: hold the right mouse button (or lock the pointer with
//! [`World::set_pointer_lock`]) and move the mouse to look; WASD or the
//! arrows move along the view; Space / E rise, Ctrl / Q sink; Shift is fast.

use crate::camera::Camera;
use crate::input::{Key, MouseButton};
use crate::world::{Frame, World};
use glam::Vec3;

/// A free-fly camera. Its fields are public, so a game can move it too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyCamera {
    pub position: Vec3,
    /// Degrees, 0 looks along +Z, 90 along +X.
    pub yaw_deg: f32,
    /// Degrees, positive looks up.
    pub pitch_deg: f32,
    /// Metres per second.
    pub speed: f32,
    /// Speed multiplier while Shift is held.
    pub fast: f32,
    /// Degrees per mouse count.
    pub look_sensitivity: f32,
    /// Lowest height above the ground, if a ground function is set with
    /// [`FlyCamera::update_above`].
    pub min_clearance: f32,
}

impl FlyCamera {
    pub fn new(position: impl Into<Vec3>, yaw_deg: f32, pitch_deg: f32) -> Self {
        Self {
            position: position.into(),
            yaw_deg,
            pitch_deg,
            speed: 10.0,
            fast: 5.0,
            look_sensitivity: 0.15,
            min_clearance: 1.5,
        }
    }

    pub fn with_speed(mut self, metres_per_second: f32) -> Self {
        self.speed = metres_per_second;
        self
    }

    /// Unit view direction.
    pub fn forward(&self) -> Vec3 {
        Camera::direction(self.yaw_deg, self.pitch_deg)
    }

    /// Read input, move, and set the world camera.
    pub fn update(&mut self, world: &mut World, frame: &Frame) {
        self.step(world, frame);
        world.look_first_person(self.position, self.yaw_deg, self.pitch_deg);
    }

    /// Like [`Self::update`], but never closer than `min_clearance` to the
    /// ground `height(x, z)` (for example `|x, z| world.proc_height_at(x, z)`
    /// computed beforehand, or your own terrain function).
    pub fn update_above(
        &mut self,
        world: &mut World,
        frame: &Frame,
        height: impl Fn(f32, f32) -> f32,
    ) {
        self.step(world, frame);
        let floor = height(self.position.x, self.position.z) + self.min_clearance;
        if floor.is_finite() && self.position.y < floor {
            self.position.y = floor;
        }
        world.look_first_person(self.position, self.yaw_deg, self.pitch_deg);
    }

    fn step(&mut self, world: &World, frame: &Frame) {
        let input = &frame.input;
        let dt = frame.dt.clamp(0.0, 0.1);
        if input.mouse_down(MouseButton::Right) || world.pointer_lock() {
            let (yaw, pitch) = input.look_delta(self.look_sensitivity);
            self.yaw_deg = (self.yaw_deg + yaw).rem_euclid(360.0);
            self.pitch_deg = (self.pitch_deg + pitch).clamp(-89.0, 89.0);
        }
        let speed = self.speed
            * if input.down(Key::Shift) {
                self.fast
            } else {
                1.0
            };
        let mut wish = input.move_xz();
        if autopilot() {
            // ENGINE_AUTOFLY=1: fly ahead in a slow curve (smoothness and
            // performance checks without anyone at the controls).
            wish = glam::Vec2::new(0.0, 1.0);
            self.yaw_deg = (self.yaw_deg - 6.0 * dt).rem_euclid(360.0);
        }
        let up = f32::from(input.down(Key::Space) || input.down(Key::E))
            - f32::from(input.down(Key::Ctrl) || input.down(Key::Q));
        let mv = self.forward() * wish.y + Camera::right_xz(self.yaw_deg) * wish.x + Vec3::Y * up;
        let step = mv.clamp_length_max(1.0) * speed * dt;
        if step.is_finite() {
            self.position += step;
        }
    }
}

fn autopilot() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_AUTOFLY").is_ok_and(|v| v.trim() != "0"))
}

#[cfg(test)]
mod tests {
    use crate::camera::Camera;

    #[test]
    fn turning_right_lowers_yaw() {
        // Facing +Z, the screen's right is -X (right-handed, Y up), and yaw
        // 90 faces +X: so a right turn must make yaw smaller.
        assert!(Camera::right_xz(0.0).x < -0.99);
        assert!(Camera::direction(-90.0, 0.0).x < -0.99);
    }
}
