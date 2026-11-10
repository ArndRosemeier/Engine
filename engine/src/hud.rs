//! A game HUD on top of egui: text in a corner of the screen, bars, a menu
//! and a crosshair, each one call on `frame.ui`. Things put on the same
//! corner stack instead of drawing over each other.
//!
//! ```ignore
//! frame.ui.text(Anchor::TopLeft, format!("Coins: {coins}"));
//! frame.ui.bar(Anchor::BottomLeft, "Health", hp, 100.0, rgb(200, 60, 60));
//! frame.ui.crosshair();
//! if paused {
//!     match frame.ui.menu("Paused", &["Resume", "Quit"]) {
//!         Some(0) => paused = false,
//!         Some(1) => world.request_exit(),
//!         _ => {}
//!     }
//! }
//! ```

use crate::color::Color;
use crate::ui::UiFrame;
use egui::{
    Align2, Color32, CornerRadius, FontId, Id, LayerId, Order, Painter, Pos2, Rect, Stroke,
    StrokeKind, Vec2,
};

/// Where on the screen a HUD item sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Anchor {
    TopLeft,
    Top,
    TopRight,
    Left,
    Center,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl Anchor {
    fn align(self) -> Align2 {
        match self {
            Anchor::TopLeft => Align2::LEFT_TOP,
            Anchor::Top => Align2::CENTER_TOP,
            Anchor::TopRight => Align2::RIGHT_TOP,
            Anchor::Left => Align2::LEFT_CENTER,
            Anchor::Center => Align2::CENTER_CENTER,
            Anchor::Right => Align2::RIGHT_CENTER,
            Anchor::BottomLeft => Align2::LEFT_BOTTOM,
            Anchor::Bottom => Align2::CENTER_BOTTOM,
            Anchor::BottomRight => Align2::RIGHT_BOTTOM,
        }
    }

    /// Bottom anchors stack upwards, the rest downwards.
    fn grows_up(self) -> bool {
        matches!(
            self,
            Anchor::BottomLeft | Anchor::Bottom | Anchor::BottomRight
        )
    }
}

/// Space between the screen edge and HUD items, in points.
const MARGIN: f32 = 16.0;
/// Space between stacked items.
const GAP: f32 = 6.0;
/// HUD text size when none is given.
pub const HUD_TEXT_SIZE: f32 = 22.0;

pub(crate) fn color32(c: Color, alpha: f32) -> Color32 {
    let [r, g, b, a] = c.to_srgba8();
    Color32::from_rgba_unmultiplied(r, g, b, (a as f32 * alpha.clamp(0.0, 1.0)) as u8)
}

/// Text with a soft shadow so it reads on a bright sky and a dark wall.
/// `at` is where `align` puts the text. Returns the rectangle it took.
pub(crate) fn shadowed_text(
    painter: &Painter,
    at: Pos2,
    align: Align2,
    text: &str,
    size: f32,
    color: Color,
    alpha: f32,
) -> Rect {
    let font = FontId::proportional(size);
    let shadow = Color32::from_black_alpha((170.0 * alpha.clamp(0.0, 1.0)) as u8);
    let off = Vec2::splat((size / 14.0).max(1.0));
    painter.text(at + off, align, text, font.clone(), shadow);
    painter.text(at, align, text, font, color32(color, alpha))
}

impl UiFrame {
    fn hud_painter(&self) -> Painter {
        self.ctx()
            .layer_painter(LayerId::new(Order::Foreground, Id::new("engine_hud")))
    }

    /// The rectangle for the next item of `size` at `anchor`, below (or for
    /// bottom anchors above) what is already there this frame.
    pub(crate) fn hud_slot(&self, anchor: Anchor, size: Vec2) -> Rect {
        let screen = self.ctx().screen_rect().shrink(MARGIN);
        let align = anchor.align();
        let mut point = align.pos_in_rect(&screen);
        let mut stacks = self.hud_stacks.borrow_mut();
        let used = stacks.entry(anchor).or_insert(0.0);
        point.y += if anchor.grows_up() { -*used } else { *used };
        *used += size.y + GAP;
        align.anchor_size(point, size)
    }

    /// White HUD text at a corner or edge of the screen.
    pub fn text(&self, anchor: Anchor, text: impl AsRef<str>) {
        self.text_styled(anchor, text, HUD_TEXT_SIZE, Color::WHITE);
    }

    /// HUD text with a size (points) and colour.
    pub fn text_styled(&self, anchor: Anchor, text: impl AsRef<str>, size: f32, color: Color) {
        let text = text.as_ref();
        let size = if size.is_finite() {
            size.clamp(4.0, 200.0)
        } else {
            HUD_TEXT_SIZE
        };
        let painter = self.hud_painter();
        let galley =
            painter.layout_no_wrap(text.to_owned(), FontId::proportional(size), Color32::WHITE);
        let slot = self.hud_slot(anchor, galley.size());
        let align = anchor.align();
        shadowed_text(
            &painter,
            align.pos_in_rect(&slot),
            align,
            text,
            size,
            color,
            1.0,
        );
    }

    /// A bar that is `value / max` full (health, stamina, loading), with
    /// `label` and the numbers written on it.
    pub fn bar(&self, anchor: Anchor, label: &str, value: f32, max: f32, color: Color) {
        let full = if max > 0.0 && value.is_finite() {
            (value / max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let slot = self.hud_slot(anchor, Vec2::new(240.0, 24.0));
        let painter = self.hud_painter();
        let round = CornerRadius::same(5);
        painter.rect_filled(slot, round, Color32::from_black_alpha(150));
        let mut fill = slot.shrink(2.0);
        fill.set_width(fill.width() * full);
        if full > 0.0 {
            painter.rect_filled(fill, CornerRadius::same(4), color32(color, 1.0));
        }
        painter.rect_stroke(
            slot,
            round,
            Stroke::new(1.0_f32, Color32::from_white_alpha(90)),
            StrokeKind::Inside,
        );
        let words = format!("{label}  {:.0} / {:.0}", value.max(0.0), max.max(0.0));
        shadowed_text(
            &painter,
            slot.center(),
            Align2::CENTER_CENTER,
            &words,
            15.0,
            Color::WHITE,
            1.0,
        );
    }

    /// A small cross in the middle of the screen, for aiming.
    pub fn crosshair(&self) {
        let painter = self.hud_painter();
        let c = self.ctx().screen_rect().center();
        for (stroke, len) in [
            (Stroke::new(4.0_f32, Color32::from_black_alpha(140)), 9.0),
            (Stroke::new(2.0_f32, Color32::WHITE), 8.0),
        ] {
            painter.line_segment([c - Vec2::new(len, 0.0), c + Vec2::new(len, 0.0)], stroke);
            painter.line_segment([c - Vec2::new(0.0, len), c + Vec2::new(0.0, len)], stroke);
        }
    }

    /// A menu in the middle of a dimmed screen: a title and one big button
    /// per choice. Returns the index of the choice clicked this frame.
    /// Show it every frame while it should be open.
    pub fn menu(&self, title: &str, choices: &[&str]) -> Option<usize> {
        let ctx = self.ctx();
        ctx.layer_painter(LayerId::new(Order::Background, Id::new("engine_menu_dim")))
            .rect_filled(
                ctx.screen_rect(),
                CornerRadius::ZERO,
                Color32::from_black_alpha(140),
            );
        let mut picked = None;
        egui::Area::new(Id::new(("engine_menu", title)))
            .order(Order::Foreground)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                egui::Frame::window(ui.style())
                    .inner_margin(24)
                    .show(ui, |ui| {
                        ui.set_min_width(260.0);
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new(title).size(30.0).strong());
                            ui.add_space(14.0);
                            for (i, choice) in choices.iter().enumerate() {
                                let button =
                                    egui::Button::new(egui::RichText::new(*choice).size(20.0))
                                        .min_size(Vec2::new(220.0, 40.0));
                                if ui.add(button).clicked() {
                                    picked = Some(i);
                                }
                                ui.add_space(6.0);
                            }
                        });
                    });
            });
        picked
    }
}
