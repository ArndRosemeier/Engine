//! Text in the world: labels over entities and floating text.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;
use crate::labels::{FloatText, Label, FLOAT_LIFE};

/// Most floating texts alive at once; the oldest go first.
const MAX_FLOAT_TEXTS: usize = 256;

impl World {
    /// Hang `label` over `id` (replacing its label). It moves with the
    /// entity and is removed when the entity is despawned.
    pub fn set_label(&mut self, id: EntityId, label: Label) -> EngineResult<()> {
        label.validate()?;
        self.insert(id, label)
    }

    pub fn remove_label(&mut self, id: EntityId) {
        self.remove::<Label>(id);
    }

    pub fn label(&self, id: EntityId) -> Option<&Label> {
        self.get::<Label>(id)
    }

    /// Text that rises from `at` and fades over about a second: damage
    /// numbers, "+10", "Locked".
    pub fn float_text(&mut self, at: impl Into<Vec3>, text: impl Into<String>, color: Color) {
        let at = at.into();
        if !at.is_finite() {
            return;
        }
        if self.float_texts.len() >= MAX_FLOAT_TEXTS {
            self.float_texts.remove(0);
        }
        self.float_texts.push(FloatText {
            at,
            text: text.into(),
            color,
            age: 0.0,
        });
    }

    /// Floating texts still rising.
    pub fn float_text_count(&self) -> usize {
        self.float_texts.len()
    }

    /// Labels drawn on the last frame (in view, near enough, not hidden).
    pub fn labels_on_screen(&self) -> usize {
        self.labels_on_screen
    }

    /// Where the world point `at` shows on a `width` x `height` view, in
    /// the same units (pixels from the top left), or `None` when it is
    /// behind the camera.
    pub fn screen_point(&self, at: impl Into<Vec3>, width: f32, height: f32) -> Option<Vec2> {
        let (w, h) = (width.max(1.0), height.max(1.0));
        let clip = self.camera.view_projection(w / h) * at.into().extend(1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new((ndc.x + 1.0) * 0.5 * w, (1.0 - ndc.y) * 0.5 * h))
    }

    pub(crate) fn age_float_texts(&mut self, dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        for f in &mut self.float_texts {
            f.age += dt;
        }
        self.float_texts.retain(|f| f.age < FLOAT_LIFE);
    }

    /// Draw labels and floating text over the scene (after the game's
    /// update, so they use this frame's camera).
    pub(crate) fn draw_labels(&mut self, ctx: &egui::Context, dt: f32) {
        self.age_float_texts(dt);
        let screen = ctx.screen_rect();
        let (w, h) = (screen.width(), screen.height());
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("engine_labels"),
        ));
        let eye = self.camera.eye();
        let to_screen = |p: Vec2| screen.min + egui::vec2(p.x, p.y);
        let on_screen = |p: Vec2| p.x > -200.0 && p.x < w + 200.0 && p.y > -50.0 && p.y < h + 50.0;
        let mut shown = 0;
        for (id, label) in self.query::<Label>() {
            let Some(m) = self.matrix_of(id) else {
                continue;
            };
            let at = m.w_axis.truncate() + Vec3::Y * label.above;
            let dist = at.distance(eye);
            if dist > label.max_distance {
                continue;
            }
            let Some(p) = self.screen_point(at, w, h).filter(|p| on_screen(*p)) else {
                continue;
            };
            if label.hidden_by_walls {
                let blocked = self
                    .cast_ray(eye, at - eye, dist)
                    .is_some_and(|hit| hit.entity != Some(id) && hit.distance < dist - 0.3);
                if blocked {
                    continue;
                }
            }
            // Fade out over the last quarter of the range; shrink a little far away.
            let fade = ((label.max_distance - dist) / (label.max_distance * 0.25)).clamp(0.0, 1.0);
            let size = label.size * (12.0 / dist.max(1.0)).clamp(0.7, 1.0);
            crate::hud::shadowed_text(
                &painter,
                to_screen(p),
                egui::Align2::CENTER_BOTTOM,
                &label.text,
                size,
                label.color,
                fade,
            );
            shown += 1;
        }
        for f in &self.float_texts {
            let (at, alpha) = f.now();
            if let Some(p) = self.screen_point(at, w, h).filter(|p| on_screen(*p)) {
                crate::hud::shadowed_text(
                    &painter,
                    to_screen(p),
                    egui::Align2::CENTER_CENTER,
                    &f.text,
                    22.0,
                    f.color,
                    alpha,
                );
            }
        }
        self.labels_on_screen = shown;
    }
}
