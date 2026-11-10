//! See-through things in front of each other: coloured glass panes, a glass
//! ball and a row of glass cubes. The engine draws them far to near, so
//! from any side the nearer glass tints what is behind it, never the other
//! way round. The camera circles the scene.
//!
//! `ENGINE_GLASS_YAW=<degrees>` holds the camera at one angle.
use engine::prelude::*;

fn main() -> EngineResult<()> {
    let held: Option<f32> = std::env::var("ENGINE_GLASS_YAW")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut time = 0.0f32;

    Engine::run("glass", move |world, frame| {
        if frame.first {
            world.set_vsync(false);
            world.set_physical_sky(Some(PhysicalSky::at_hour(14.0)));
            world.spawn(Mesh::box_at(
                (0.0, -0.05, 0.0),
                (30.0, 0.1, 30.0),
                rgb(120, 118, 110),
            )?);
            // A solid post behind the panes, to see through them.
            let post = world.spawn(Mesh::box_at(Vec3::ZERO, (0.6, 3.0, 0.6), rgb(40, 40, 46))?);
            world.set_place(post, Place::new(0.0, 1.5, 0.0))?;

            // Five panes in a row along z, each its own entity and colour.
            // Spawned near to far as seen from -z, so drawing them in spawn
            // order would be wrong from that side.
            let tints = [
                rgba(230, 60, 60, 110),
                rgba(240, 170, 40, 110),
                rgba(60, 200, 90, 110),
                rgba(60, 130, 240, 110),
                rgba(170, 80, 220, 110),
            ];
            for (i, tint) in tints.into_iter().enumerate() {
                let pane = world.spawn(Mesh::box_at(Vec3::ZERO, (3.0, 2.2, 0.04), tint)?);
                world.set_place(pane, Place::new(0.0, 1.2, -4.0 + i as f32 * 2.0))?;
            }

            // A glass ball and a row of glass cubes (one instanced spawn).
            let ball = world.spawn(Mesh::ball(Vec3::ZERO, 0.8, rgba(200, 240, 255, 90))?);
            world.set_place(ball, Place::new(3.5, 0.8, 0.0))?;
            let cubes = world.spawn_instanced(Mesh::box_at(
                Vec3::ZERO,
                (0.7, 0.7, 0.7),
                rgba(255, 255, 255, 80),
            )?);
            let places: Vec<Place> = (0..6)
                .map(|i| Place::new(-3.5, 0.35, -5.0 + i as f32 * 2.0))
                .collect();
            world.set_instances(cubes, &places)?;
        }
        time += frame.dt;
        let yaw = held.unwrap_or(90.0 + time * 20.0);
        world.look_orbit(Vec3::new(0.0, 1.2, 0.0), 11.0, yaw, 18.0);
        world.report("camera yaw", yaw);
        Ok(())
    })
}
