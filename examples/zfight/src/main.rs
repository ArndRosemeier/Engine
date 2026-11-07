//! Z-fighting finder demo: a small yard with four classic mistakes (a rug
//! lying exactly on the floor, a poster flush with a wall, a tablecloth at
//! the table top's height, a path drawn on the ground plane). Each one is
//! marked with a glowing magenta post and listed in the console.
//!
//! F switches to the fixed yard (every face lifted 1-2 cm off its surface)
//! and the posts go away (`ENGINE_ZFIGHT_FIXED=1` starts there). M toggles
//! the posts. Fly with WASD and the mouse.
//!
//! Any app gets the same check with `ENGINE_ZFIGHT=1`; in code it is
//! `world.check_z_fighting()`, and in a test `ZFightCheck` (see
//! `engine::zfight`).
use engine::prelude::*;

/// The yard's parts. `lift` is how far the thin things sit off their
/// surfaces: 0 is the bug, a centimetre or two is the fix.
fn yard(lift: f32) -> EngineResult<Vec<(&'static str, Mesh)>> {
    let wall = rgb(214, 196, 168);
    let mut out = vec![
        (
            "ground",
            Shape::box_at((0.0, -0.05, 0.0), (30.0, 0.1, 30.0), rgb(98, 128, 76))?,
        ),
        (
            "wall",
            Shape::box_at((0.0, 1.5, 4.0), (8.0, 3.0, 0.3), wall)?,
        ),
        (
            "floor slab",
            Shape::box_at((0.0, 0.05, 1.5), (8.0, 0.1, 4.0), rgb(150, 146, 140))?,
        ),
    ];
    // Rug on the slab.
    out.push((
        "rug",
        Shape::box_at(
            (-2.0, 0.1 + lift, 1.5),
            (2.4, 0.0002, 1.6),
            rgb(150, 36, 40),
        )?,
    ));
    // Poster on the wall's front face (z = 3.85).
    out.push((
        "poster",
        Shape::box_at(
            (2.2, 1.7, 3.85 - lift),
            (1.0, 1.4, 0.0002),
            rgb(40, 90, 170),
        )?,
    ));
    // Table and a cloth on its top (top at y = 0.8).
    let mut table = Mesh::new();
    table.add_box((1.8, 0.775, 1.0), (1.6, 0.05, 1.0), rgb(120, 82, 50))?;
    for (x, z) in [(1.1, 0.6), (2.5, 0.6), (1.1, 1.4), (2.5, 1.4)] {
        table.add_box((x, 0.375, z), (0.08, 0.75, 0.08), rgb(100, 68, 40))?;
    }
    out.push(("table", table));
    out.push((
        "tablecloth",
        Shape::box_at(
            (1.8, 0.8 + lift, 1.0),
            (1.2, 0.0002, 0.7),
            rgb(235, 235, 225),
        )?,
    ));
    // A gravel path on the ground (top at y = 0).
    out.push((
        "path",
        Shape::box_at((0.0, lift, -4.5), (1.6, 0.0002, 9.0), rgb(176, 160, 128))?,
    ));
    Ok(out)
}

fn main() -> EngineResult<()> {
    let mut camera = FlyCamera::new((-3.5, 3.2, -7.5), 22.0, -16.0).with_speed(6.0);
    let mut fixed = std::env::var("ENGINE_ZFIGHT_FIXED").is_ok_and(|v| v == "1");
    let mut marks = true;
    let mut spawned: Vec<EntityId> = Vec::new();
    let mut report: Option<ZFightReport> = None;
    let mut rebuild = true;

    Engine::run("zfight", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.0)));
        }
        if frame.input.pressed(Key::F) {
            fixed = !fixed;
            rebuild = true;
        }
        if frame.input.pressed(Key::M) {
            marks = !marks;
            match (&report, marks) {
                (Some(r), true) => {
                    world.show_z_fighting(r);
                }
                _ => world.clear_z_fighting_marks(),
            }
        }
        if rebuild {
            rebuild = false;
            for id in spawned.drain(..) {
                world.despawn(id);
            }
            for (name, mesh) in yard(if fixed { 0.015 } else { 0.0 })? {
                let id = world.spawn(mesh);
                // Names make the report readable ("rug / floor slab").
                world.set_name(id, name);
                spawned.push(id);
            }
            // The check itself: everything in the world, the bottoms of
            // things resting on the ground ignored.
            let r = world.find_z_fighting_with(&WorldScan {
                options: engine::zfight::Options::ground_at(0.0),
                ..WorldScan::default()
            });
            println!("{} yard:\n{r}\n", if fixed { "fixed" } else { "broken" });
            if marks {
                world.show_z_fighting(&r);
            }
            report = Some(r);
        }
        camera.update(world, frame);

        let r = report.as_ref().expect("checked");
        engine::egui::Window::new("z-fighting")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{} yard: {} flickering spots, {} doubled faces",
                    if fixed { "fixed" } else { "broken" },
                    r.spots(2.0).len(),
                    r.doubled().count()
                ));
                for s in r.spots(2.0).iter().take(6) {
                    ui.label(format!(
                        "{:.2} m² at ({:.1}, {:.2}, {:.1})  {} / {}",
                        s.area,
                        s.at.x,
                        s.at.y,
                        s.at.z,
                        r.source_name(s.sources[0]),
                        r.source_name(s.sources[1])
                    ));
                }
                ui.label("F: fixed / broken   M: markers   WASD + mouse: fly");
            });
        Ok(())
    })
}
