//! Friendly keyboard input for the frame callback.

use crate::camera::Camera;
use glam::Vec3;
use std::collections::HashSet;
use winit::keyboard::{KeyCode, PhysicalKey};

/// Keys the engine tracks for gameplay (not Escape — that still quits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    W,
    A,
    S,
    D,
    Q,
    E,
    Space,
    Shift,
    Up,
    Down,
    Left,
    Right,
}

impl Key {
    fn from_code(code: KeyCode) -> Option<Self> {
        Some(match code {
            KeyCode::KeyW => Self::W,
            KeyCode::KeyA => Self::A,
            KeyCode::KeyS => Self::S,
            KeyCode::KeyD => Self::D,
            KeyCode::KeyQ => Self::Q,
            KeyCode::KeyE => Self::E,
            KeyCode::Space => Self::Space,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => Self::Shift,
            KeyCode::ArrowUp => Self::Up,
            KeyCode::ArrowDown => Self::Down,
            KeyCode::ArrowLeft => Self::Left,
            KeyCode::ArrowRight => Self::Right,
            _ => return None,
        })
    }
}

/// Snapshot of held keys for one frame.
#[derive(Clone, Debug, Default)]
pub struct Input {
    down: HashSet<Key>,
}

impl Input {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn set_key(&mut self, physical: PhysicalKey, pressed: bool) {
        let PhysicalKey::Code(code) = physical else {
            return;
        };
        let Some(key) = Key::from_code(code) else {
            return;
        };
        if pressed {
            self.down.insert(key);
        } else {
            self.down.remove(&key);
        }
    }

    pub fn down(&self, key: Key) -> bool {
        self.down.contains(&key)
    }

    /// Horizontal move intent from WASD / arrows (x = strafe right, y = forward).
    /// Length is at most 1.
    pub fn move_xz(&self) -> glam::Vec2 {
        let mut x = 0.0;
        let mut y = 0.0;
        if self.down(Key::D) || self.down(Key::Right) {
            x += 1.0;
        }
        if self.down(Key::A) || self.down(Key::Left) {
            x -= 1.0;
        }
        if self.down(Key::W) || self.down(Key::Up) {
            y += 1.0;
        }
        if self.down(Key::S) || self.down(Key::Down) {
            y -= 1.0;
        }
        let v = glam::Vec2::new(x, y);
        if v.length_squared() > 1.0 {
            v.normalize()
        } else {
            v
        }
    }

    /// World-space XZ move direction for a walker/camera yaw (degrees, 0 = +Z).
    ///
    /// Prefer this over combining [`move_xz`] with hand-rolled basis vectors —
    /// the strafe axis must match the follow camera's screen-right.
    pub fn move_dir_xz(&self, yaw_degrees: f32) -> Vec3 {
        let wish = self.move_xz();
        if wish.length_squared() == 0.0 {
            return Vec3::ZERO;
        }
        let forward = Camera::facing_xz(yaw_degrees);
        let right = Camera::right_xz(yaw_degrees);
        (right * wish.x + forward * wish.y).normalize_or_zero()
    }

    /// Q/E yaw rate sign: −1 = left (Q), +1 = right (E).
    pub fn yaw_sign(&self) -> f32 {
        let mut s = 0.0;
        if self.down(Key::E) {
            s += 1.0;
        }
        if self.down(Key::Q) {
            s -= 1.0;
        }
        s
    }
}
