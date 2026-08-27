//! Modal overlay demo: Open Atlas → RGBA preview + Close.
//!
//! Escape closes the modal first; Escape again quits.

use engine::egui;
use engine::prelude::*;

fn checker_rgba(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let cell = ((x / 8) ^ (y / 8)) & 1;
            let i = ((y * w + x) * 4) as usize;
            if cell == 0 {
                px[i] = 40;
                px[i + 1] = 90;
                px[i + 2] = 50;
            } else {
                px[i] = 30;
                px[i + 1] = 55;
                px[i + 2] = 110;
            }
            px[i + 3] = 255;
        }
    }
    px
}

fn main() -> EngineResult<()> {
    let mut show_atlas = true;
    let atlas = checker_rgba(128, 96);

    Engine::run("ui_modal", move |world, frame| {
        if frame.first {
            world.spawn(
                Shape::box_at(Vec3::ZERO, Vec3::new(2.0, 0.4, 2.0), rgb(180, 140, 90)).unwrap(),
            );
        }
        world.look_orbit(Vec3::ZERO, 8.0, frame.time * 25.0, 28.0);
        world.set_sun(Vec3::new(0.4, 1.0, 0.2), 0.22);

        if !show_atlas {
            // Tiny HUD to reopen.
            egui::Window::new("Controls")
                .anchor(egui::Align2::LEFT_TOP, [12.0, 12.0])
                .resizable(false)
                .show(frame.ui.ctx(), |ui| {
                    if ui.button("Open Atlas").clicked() {
                        show_atlas = true;
                    }
                    ui.label("Escape quits when no modal is open.");
                });
        }

        frame.ui.modal("Atlas", &mut show_atlas, |ui, open| {
            ui.label("Placeholder map (RGBA image).");
            ui.image("atlas_preview", 128, 96, &atlas);
            if ui.button("Close") {
                *open = false;
            }
        });
        Ok(())
    })
}
