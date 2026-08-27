//! Procedural landscape demo: hills + caves via the Landscape recipe.
use engine::prelude::*;

fn main() -> EngineResult<()> {
    let mut yaw = 48.0_f32;

    Engine::run("procedural_caves", move |world, frame| {
        if frame.first {
            world.spawn(
                Landscape::new(7)
                    .area(56.0, 56.0)
                    .height(20.0)
                    .caves(true)
                    .tunnel(true)
                    .color(rgb(122, 148, 92))
                    .build()
                    .expect("landscape"),
            );
            world.set_clear_color(rgb(140, 191, 242));
            world.set_sun((0.45, 1.0, 0.35), 0.24);
        }

        yaw += frame.dt * 8.0;
        let yaw = if std::env::var_os("ENGINE_SCREENSHOT").is_some() {
            48.0
        } else {
            yaw
        };
        world.look_orbit((0.0, 5.0, 0.0), 36.0, yaw, 26.0);
        Ok(())
    })
}
