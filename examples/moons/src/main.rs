//! Several moons: a night under three moons of different sizes, colours,
//! cycles and paths. Their light adds up on the ground, tinted by their
//! colours and coming from where the brightest stand; the nearer (bigger)
//! moon covers the others when they pass.
//!
//! Controls: right mouse + move to look, WASD to fly.
//! - Space: time runs (one hour every 4 seconds) or stops.
//! - Up / Down: an hour later or earlier. Left / Right: a day earlier or later
//!   (the phases move on).
//! - M: three moons or the one usual moon (compare).
//!
//! `ENGINE_MOONS_HOUR=h` and `ENGINE_MOONS_DAY=d` start at that time;
//! `ENGINE_MOONS_ONE=1` starts with the one usual moon.
use engine::prelude::*;
use glam::Vec3;

fn rules() -> TerrainRules {
    TerrainRules {
        base_height: 0.0,
        hill_height: 6.0,
        hill_scale: 0.012,
        lake_threshold: 2.0,
        water_level: -50.0,
        ..demo_terrain_rules()
    }
}

fn three_moons() -> Vec<MoonSpec> {
    vec![
        // A big pale moon on the usual path.
        MoonSpec::new(2.2, 29.5, 0.38),
        // A small red one, racing through its phases on a low path.
        MoonSpec::new(0.8, 3.1, 0.27)
            .tilt(-10.0)
            .color(rgb(255, 150, 120))
            .brightness(1.4),
        // A blue-white one riding high.
        MoonSpec::new(1.2, 11.0, 0.42)
            .tilt(22.0)
            .color(rgb(170, 200, 255)),
    ]
}

fn main() -> EngineResult<()> {
    let env = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let mut hour = env("ENGINE_MOONS_HOUR", 23.5);
    let mut day = env("ENGINE_MOONS_DAY", 0.0);
    let ground = HeightField::new(rules());
    let mut camera = FlyCamera::new(
        (0.0, ground.height_at(0.0, -30.0) + 6.0, -30.0),
        -12.0,
        26.0,
    )
    .with_speed(10.0);
    let mut running = true;
    let mut three = std::env::var("ENGINE_MOONS_ONE").map_or(true, |v| v != "1");

    Engine::run("moons", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_proc_terrain(ProcTerrain::gpu_clipmap(rules(), ClipmapConfig::default()));
            world.set_moons(if three { three_moons() } else { Vec::new() })?;
            // Pale stones and a pillar to catch the moonlight and cast shadows.
            for i in 0..12 {
                let a = i as f32 * 0.52;
                let (x, z) = (a.cos() * 14.0, a.sin() * 14.0 + 10.0);
                world.spawn(Mesh::box_at(
                    Vec3::new(x, ground.height_at(x, z) + 1.5, z),
                    Vec3::new(1.4, 3.0 + (i % 3) as f32, 1.4),
                    rgb(215, 210, 200),
                )?);
            }
            world.spawn(Mesh::box_at(
                Vec3::new(0.0, ground.height_at(0.0, 10.0) + 5.0, 10.0),
                Vec3::new(2.0, 10.0, 2.0),
                rgb(230, 225, 215),
            )?);
        }
        if frame.input.pressed(Key::Space) {
            running = !running;
        }
        if frame.input.pressed(Key::Up) {
            hour += 1.0;
        }
        if frame.input.pressed(Key::Down) {
            hour -= 1.0;
        }
        if frame.input.pressed(Key::Right) {
            day += 1.0;
        }
        if frame.input.pressed(Key::Left) {
            day -= 1.0;
        }
        if frame.input.pressed(Key::M) {
            three = !three;
            world.set_moons(if three { three_moons() } else { Vec::new() })?;
        }
        if running {
            hour += frame.dt / 4.0;
        }
        if hour >= 24.0 {
            hour -= 24.0;
            day += 1.0;
        } else if hour < 0.0 {
            hour += 24.0;
            day -= 1.0;
        }
        let mut sky = PhysicalSky::at_hour(hour);
        sky.day = day;
        world.set_physical_sky(Some(sky));
        camera.update_above(world, frame, |x, z| ground.height_at(x, z));
        world.set_proc_focus(camera.position);

        engine::egui::Window::new("moons")
            .anchor(engine::egui::Align2::LEFT_BOTTOM, (12.0, -12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   day {:.0}  {:02}:{:02}   {}",
                    frame.fps,
                    day,
                    hour as u32,
                    ((hour.fract()) * 60.0) as u32,
                    if three { "three moons" } else { "one moon" }
                ));
                ui.label(format!(
                    "Space: time {}   Up/Down: hour   Left/Right: day   M: moons",
                    if running { "runs" } else { "stopped" }
                ));
            });
        Ok(())
    })
}
