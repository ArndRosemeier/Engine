//! The physical sky: a day passing over a small forest in two minutes.
//!
//! Sun and moon follow their paths, the sky scatters (blue noon, red dusk,
//! starry night with a phased moon), clouds drift, and the scene's sun,
//! ambient light, fog and exposure follow the sky.
//!
//! `ENGINE_SKY_HOUR=18.5` holds a fixed hour; `ENGINE_SKY_WEATHER` picks
//! `clear` (default), `overcast` or `storm`; `ENGINE_SKY_YAW` turns the
//! camera (degrees).
use engine::prelude::*;
use engine::tree::{TreeAsset, TreeLod, TreeSettings};

fn main() -> EngineResult<()> {
    let hour: Option<f32> = std::env::var("ENGINE_SKY_HOUR")
        .ok()
        .and_then(|v| v.parse().ok());
    let yaw: f32 = std::env::var("ENGINE_SKY_YAW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0);
    let weather = match std::env::var("ENGINE_SKY_WEATHER").as_deref() {
        Ok("overcast") => SkyWeather::OVERCAST,
        Ok("storm") => SkyWeather::STORM,
        _ => SkyWeather::CLEAR,
    };
    let positions = scatter_on_xz(
        5,
        (-120.0, 0.0, -120.0).into(),
        (120.0, 0.0, 120.0).into(),
        9.0,
        0.9,
        0.0,
    );
    let places = scatter_places(&positions, 1.0, |i| i as f32 * 47.0);

    Engine::run("sky_day", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (1600.0, 0.1, 1600.0),
                rgb(84, 120, 62),
            )?);
            for (species, settings) in [
                (0usize, TreeSettings::broadleaf(4)),
                (1, TreeSettings::conifer(3)),
            ] {
                let mine: Vec<Place> = places
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == species)
                    .map(|(_, p)| *p)
                    .collect();
                let mesh = |lod| {
                    TreeAsset::generate_lod(Vec3::ZERO, settings.with_lod(lod), lod)
                        .mesh()
                        .expect("tree mesh")
                };
                let id = world.spawn_lod(vec![
                    LodLevel::new(mesh(TreeLod::Hero), 40.0),
                    LodLevel::new(mesh(TreeLod::Mid), 160.0),
                    LodLevel::new(mesh(TreeLod::Far), 600.0),
                ])?;
                world.set_lod_places(id, &mine)?;
            }
            world.set_view_distance(800.0)?;
        }
        // 24 h in 120 s, starting just before sunrise.
        let h = hour.unwrap_or((5.0 + frame.time * 0.2) % 24.0);
        world.set_physical_sky(Some(PhysicalSky::at_hour(h).with_weather(weather)));
        // From a meadow south of the forest, looking north-east past it.
        world.look_first_person(Vec3::new(0.0, 3.0, -200.0), yaw, 12.0);
        Ok(())
    })
}
