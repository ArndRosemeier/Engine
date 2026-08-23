//! Immediate-mode UI (egui) for modals, buttons, labels, and images.

use egui::{ColorImage, Context, TextureHandle, TextureOptions, Vec2};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

pub use egui;

/// Per-frame UI entry point passed on [`crate::world::Frame`].
#[derive(Clone)]
pub struct UiFrame {
    ctx: Context,
    textures: Rc<RefCell<HashMap<String, TextureHandle>>>,
    modal_open: Rc<Cell<bool>>,
    bind_listen: Rc<Cell<bool>>,
}

impl std::fmt::Debug for UiFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiFrame")
            .field("modal_open", &self.modal_open.get())
            .field("bind_listen", &self.bind_listen.get())
            .finish_non_exhaustive()
    }
}

impl Default for UiFrame {
    fn default() -> Self {
        Self::new(Context::default(), Rc::new(RefCell::new(HashMap::new())))
    }
}

impl UiFrame {
    pub(crate) fn new(ctx: Context, textures: Rc<RefCell<HashMap<String, TextureHandle>>>) -> Self {
        Self {
            ctx,
            textures,
            modal_open: Rc::new(Cell::new(false)),
            bind_listen: Rc::new(Cell::new(false)),
        }
    }

    /// Full egui context for custom widgets, menus, tables, etc.
    pub fn ctx(&self) -> &Context {
        &self.ctx
    }

    /// True if a [`Self::modal`] was shown this frame.
    pub fn modal_was_open(&self) -> bool {
        self.modal_open.get()
    }

    /// While true, Escape does not close [`Self::modal`].
    pub fn set_bind_listen(&self, listening: bool) {
        self.bind_listen.set(listening);
    }

    pub fn bind_listen(&self) -> bool {
        self.bind_listen.get()
    }

    /// Centered modal overlay. Builds content with [`UiPanel`].
    ///
    /// Closing: set `open` to false from the content closure, or let Escape /
    /// backdrop close it (`open` is cleared when egui reports should-close).
    pub fn modal(
        &self,
        title: &str,
        open: &mut bool,
        add_contents: impl FnOnce(&mut UiPanel<'_>, &mut bool),
    ) {
        if !*open {
            return;
        }
        self.modal_open.set(true);

        let modal = egui::Modal::new(egui::Id::new(title));
        let response = modal.show(&self.ctx, |ui| {
            ui.set_min_width(320.0);
            ui.heading(title);
            ui.separator();
            let mut panel = UiPanel {
                ui,
                ctx: self.ctx.clone(),
                textures: Rc::clone(&self.textures),
            };
            add_contents(&mut panel, open);
        });

        if response.should_close() {
            let escape = self.ctx.input(|i| i.key_pressed(egui::Key::Escape));
            if !(self.bind_listen.get() && escape) {
                *open = false;
            }
        }
    }
}

/// Widgets drawn inside a [`UiFrame::modal`] (or any egui `Ui` you pass in).
pub struct UiPanel<'a> {
    ui: &'a mut egui::Ui,
    ctx: Context,
    textures: Rc<RefCell<HashMap<String, TextureHandle>>>,
}

impl UiPanel<'_> {
    pub fn ui(&mut self) -> &mut egui::Ui {
        self.ui
    }

    pub fn label(&mut self, text: impl Into<egui::WidgetText>) {
        self.ui.label(text);
    }

    /// Returns true the frame the button is clicked.
    pub fn button(&mut self, text: impl Into<egui::WidgetText>) -> bool {
        self.ui.button(text).clicked()
    }

    /// Show an RGBA8 image (`width * height * 4` bytes, unmultiplied).
    ///
    /// `id` is stable across frames so the GPU texture is reused/updated.
    pub fn image(&mut self, id: &str, width: u32, height: u32, rgba: &[u8]) {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .expect("image dimensions overflow");
        assert_eq!(
            rgba.len(),
            expected,
            "image `{id}` expected {expected} RGBA bytes, got {}",
            rgba.len()
        );
        assert!(width > 0 && height > 0, "image `{id}` must be non-empty");

        let color = ColorImage::from_rgba_unmultiplied([width as usize, height as usize], rgba);
        let mut textures = self.textures.borrow_mut();
        let handle = textures
            .entry(id.to_string())
            .and_modify(|tex| tex.set(color.clone(), TextureOptions::NEAREST))
            .or_insert_with(|| {
                self.ctx
                    .load_texture(id.to_string(), color, TextureOptions::NEAREST)
            });

        let max_side = self.ui.available_width().clamp(64.0, 512.0);
        let scale = max_side / width.max(height) as f32;
        let size = Vec2::new(width as f32 * scale, height as f32 * scale);
        self.ui.image((handle.id(), size));
    }
}
