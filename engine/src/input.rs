//! Friendly keyboard and mouse-look input for the frame callback.

use crate::camera::Camera;
use glam::{Vec2, Vec3};
use std::collections::HashSet;
use winit::keyboard::{KeyCode, PhysicalKey};

/// Keys the engine tracks for gameplay (not Escape — that still quits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    Tab,
    Space,
    Shift,
    Ctrl,
    Up,
    Down,
    Left,
    Right,
}

impl Key {
    fn from_code(code: KeyCode) -> Option<Self> {
        Some(match code {
            KeyCode::KeyA => Self::A,
            KeyCode::KeyB => Self::B,
            KeyCode::KeyC => Self::C,
            KeyCode::KeyD => Self::D,
            KeyCode::KeyE => Self::E,
            KeyCode::KeyF => Self::F,
            KeyCode::KeyG => Self::G,
            KeyCode::KeyH => Self::H,
            KeyCode::KeyI => Self::I,
            KeyCode::KeyJ => Self::J,
            KeyCode::KeyK => Self::K,
            KeyCode::KeyL => Self::L,
            KeyCode::KeyM => Self::M,
            KeyCode::KeyN => Self::N,
            KeyCode::KeyO => Self::O,
            KeyCode::KeyP => Self::P,
            KeyCode::KeyQ => Self::Q,
            KeyCode::KeyR => Self::R,
            KeyCode::KeyS => Self::S,
            KeyCode::KeyT => Self::T,
            KeyCode::KeyU => Self::U,
            KeyCode::KeyV => Self::V,
            KeyCode::KeyW => Self::W,
            KeyCode::KeyX => Self::X,
            KeyCode::KeyY => Self::Y,
            KeyCode::KeyZ => Self::Z,
            KeyCode::Digit0 | KeyCode::Numpad0 => Self::Digit0,
            KeyCode::Digit1 | KeyCode::Numpad1 => Self::Digit1,
            KeyCode::Digit2 | KeyCode::Numpad2 => Self::Digit2,
            KeyCode::Digit3 | KeyCode::Numpad3 => Self::Digit3,
            KeyCode::Digit4 | KeyCode::Numpad4 => Self::Digit4,
            KeyCode::Digit5 | KeyCode::Numpad5 => Self::Digit5,
            KeyCode::Digit6 | KeyCode::Numpad6 => Self::Digit6,
            KeyCode::Digit7 | KeyCode::Numpad7 => Self::Digit7,
            KeyCode::Digit8 | KeyCode::Numpad8 => Self::Digit8,
            KeyCode::Digit9 | KeyCode::Numpad9 => Self::Digit9,
            KeyCode::Tab => Self::Tab,
            KeyCode::Space => Self::Space,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => Self::Shift,
            KeyCode::ControlLeft | KeyCode::ControlRight => Self::Ctrl,
            KeyCode::ArrowUp => Self::Up,
            KeyCode::ArrowDown => Self::Down,
            KeyCode::ArrowLeft => Self::Left,
            KeyCode::ArrowRight => Self::Right,
            _ => return None,
        })
    }

    /// Stable bind name stored in settings (`"1"`, `"R"`, `"Tab"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::E => "E",
            Self::F => "F",
            Self::G => "G",
            Self::H => "H",
            Self::I => "I",
            Self::J => "J",
            Self::K => "K",
            Self::L => "L",
            Self::M => "M",
            Self::N => "N",
            Self::O => "O",
            Self::P => "P",
            Self::Q => "Q",
            Self::R => "R",
            Self::S => "S",
            Self::T => "T",
            Self::U => "U",
            Self::V => "V",
            Self::W => "W",
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
            Self::Digit0 => "0",
            Self::Digit1 => "1",
            Self::Digit2 => "2",
            Self::Digit3 => "3",
            Self::Digit4 => "4",
            Self::Digit5 => "5",
            Self::Digit6 => "6",
            Self::Digit7 => "7",
            Self::Digit8 => "8",
            Self::Digit9 => "9",
            Self::Tab => "Tab",
            Self::Space => "Space",
            Self::Shift => "Shift",
            Self::Ctrl => "Ctrl",
            Self::Up => "Up",
            Self::Down => "Down",
            Self::Left => "Left",
            Self::Right => "Right",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "A" | "a" => Self::A,
            "B" | "b" => Self::B,
            "C" | "c" => Self::C,
            "D" | "d" => Self::D,
            "E" | "e" => Self::E,
            "F" | "f" => Self::F,
            "G" | "g" => Self::G,
            "H" | "h" => Self::H,
            "I" | "i" => Self::I,
            "J" | "j" => Self::J,
            "K" | "k" => Self::K,
            "L" | "l" => Self::L,
            "M" | "m" => Self::M,
            "N" | "n" => Self::N,
            "O" | "o" => Self::O,
            "P" | "p" => Self::P,
            "Q" | "q" => Self::Q,
            "R" | "r" => Self::R,
            "S" | "s" => Self::S,
            "T" | "t" => Self::T,
            "U" | "u" => Self::U,
            "V" | "v" => Self::V,
            "W" | "w" => Self::W,
            "X" | "x" => Self::X,
            "Y" | "y" => Self::Y,
            "Z" | "z" => Self::Z,
            "0" => Self::Digit0,
            "1" => Self::Digit1,
            "2" => Self::Digit2,
            "3" => Self::Digit3,
            "4" => Self::Digit4,
            "5" => Self::Digit5,
            "6" => Self::Digit6,
            "7" => Self::Digit7,
            "8" => Self::Digit8,
            "9" => Self::Digit9,
            "Tab" | "tab" => Self::Tab,
            "Space" | "space" => Self::Space,
            "Shift" | "shift" => Self::Shift,
            "Ctrl" | "ctrl" | "Control" => Self::Ctrl,
            "Up" | "up" => Self::Up,
            "Down" | "down" => Self::Down,
            "Left" | "left" => Self::Left,
            "Right" | "right" => Self::Right,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Mouse buttons the engine tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

impl MouseButton {
    fn from_winit(button: winit::event::MouseButton) -> Option<Self> {
        Some(match button {
            winit::event::MouseButton::Left => Self::Left,
            winit::event::MouseButton::Right => Self::Right,
            winit::event::MouseButton::Middle => Self::Middle,
            _ => return None,
        })
    }
}

/// Snapshot of held keys, fresh presses, and mouse motion for one frame.
#[derive(Clone, Debug, Default)]
pub struct Input {
    down: HashSet<Key>,
    pressed: HashSet<Key>,
    mouse_down: HashSet<MouseButton>,
    mouse_clicked: HashSet<MouseButton>,
    mouse_delta: Vec2,
    last_key_down: Option<Key>,
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
            if !self.down.contains(&key) {
                self.last_key_down = Some(key);
            }
            self.down.insert(key);
            self.pressed.insert(key);
        } else {
            self.down.remove(&key);
        }
    }

    pub(crate) fn set_mouse_button(&mut self, button: winit::event::MouseButton, pressed: bool) {
        let Some(button) = MouseButton::from_winit(button) else {
            return;
        };
        if pressed {
            self.mouse_down.insert(button);
            self.mouse_clicked.insert(button);
        } else {
            self.mouse_down.remove(&button);
        }
    }

    /// Accumulate raw pointer motion (device counts, not window pixels).
    pub(crate) fn add_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta += Vec2::new(dx, dy);
    }

    /// Forget one frame's edges and motion, keeping what is still held.
    pub(crate) fn end_frame(&mut self) {
        self.pressed.clear();
        self.mouse_clicked.clear();
        self.mouse_delta = Vec2::ZERO;
        self.last_key_down = None;
    }

    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_down.contains(&button)
    }

    /// True on the frame a mouse button goes down.
    pub fn mouse_clicked(&self, button: MouseButton) -> bool {
        self.mouse_clicked.contains(&button)
    }

    pub fn down(&self, key: Key) -> bool {
        self.down.contains(&key)
    }

    /// True on the frame a key goes down, for toggles like fly mode.
    pub fn pressed(&self, key: Key) -> bool {
        self.pressed.contains(&key)
    }

    /// Last physical key-down this frame (repeats ignored). For Settings bind-listen.
    pub fn last_key_down(&self) -> Option<Key> {
        self.last_key_down
    }

    /// Raw pointer motion since the last frame (x right, y down).
    ///
    /// Only meaningful while the pointer is locked
    /// ([`crate::world::World::set_pointer_lock`]); otherwise it is whatever
    /// motion the window happened to see.
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// Signed axis from two keys, for one-line bindings.
    pub fn axis(&self, negative: Key, positive: Key) -> f32 {
        f32::from(self.down(positive)) - f32::from(self.down(negative))
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
