//! Text in the world: a name over an entity ([`Label`]) and numbers that
//! float up and fade (`World::float_text`), drawn over the scene each frame.

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use glam::Vec3;

/// Text that hangs over an entity: a name, a price, "Press E". Put it on
/// with `world.set_label(id, Label::new("Shop"))`; it goes away with the
/// entity. It keeps its size on screen and fades out with distance.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub text: String,
    pub color: Color,
    /// Text size in points at close range.
    pub size: f32,
    /// Metres above the entity's place (its origin), default 2.2.
    pub above: f32,
    /// Hidden past this distance from the camera (m), fading out before.
    pub max_distance: f32,
    /// Hide it when a wall or the ground is between the camera and it
    /// (one physics ray a frame per label). Default off.
    pub hidden_by_walls: bool,
}

impl Label {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            color: Color::WHITE,
            size: 18.0,
            above: 2.2,
            max_distance: 40.0,
            hidden_by_walls: false,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn size(mut self, points: f32) -> Self {
        self.size = points;
        self
    }

    pub fn above(mut self, metres: f32) -> Self {
        self.above = metres;
        self
    }

    pub fn max_distance(mut self, metres: f32) -> Self {
        self.max_distance = metres;
        self
    }

    pub fn hidden_by_walls(mut self, on: bool) -> Self {
        self.hidden_by_walls = on;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let ok = self.size.is_finite()
            && self.size > 0.0
            && self.above.is_finite()
            && self.max_distance.is_finite()
            && self.max_distance > 0.0;
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidValue(format!(
                "label {:?}: size and max_distance must be > 0 and every number finite \
                 (size {}, above {}, max_distance {})",
                self.text, self.size, self.above, self.max_distance
            )))
        }
    }
}

/// How long floating text lives (s) and how far it rises (m).
pub(crate) const FLOAT_LIFE: f32 = 1.2;
pub(crate) const FLOAT_RISE: f32 = 1.2;

/// Text that rises from a point and fades: damage numbers, "+10".
#[derive(Clone, Debug)]
pub(crate) struct FloatText {
    pub at: Vec3,
    pub text: String,
    pub color: Color,
    pub age: f32,
}

impl FloatText {
    /// Where it is now and how visible (1 = solid, 0 = gone).
    pub fn now(&self) -> (Vec3, f32) {
        let t = (self.age / FLOAT_LIFE).clamp(0.0, 1.0);
        let rise = FLOAT_RISE * (1.0 - (1.0 - t) * (1.0 - t));
        let alpha = if t < 0.5 { 1.0 } else { 1.0 - (t - 0.5) * 2.0 };
        (self.at + Vec3::Y * rise, alpha)
    }
}
