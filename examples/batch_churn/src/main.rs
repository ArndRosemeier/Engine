//! Instance batches that change, vanish and come back (a city size switch).
//!
//! A streamed city moves and refills its instance lists, then a new city size
//! despawns every instanced thing at once. The renderer must drop the batches
//! of what is gone (it once kept them and panicked on the next shadow pass).
use engine::prelude::*;

const SPAWN: u32 = 5;
const CHANGE: u32 = 15;
const CLEAR: u32 = 25;
const AGAIN: u32 = 35;
const END: u32 = 60;

fn lamp() -> EngineResult<Mesh> {
    Shape::box_at((0.0, 1.0, 0.0), (0.3, 2.0, 0.3), rgb(40, 44, 50))
}

fn row(count: usize, z: f32) -> Vec<Place> {
    (0..count)
        .map(|i| Place::new(-10.0 + i as f32 * 2.0, 0.0, z))
        .collect()
}

fn main() -> EngineResult<()> {
    let mut n = 0u32;
    let mut batch: Option<EntityId> = None;
    let mut rounds = 0u32;
    Engine::run("batch churn", move |world, frame| {
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
            let ground = world.spawn(Shape::box_at(
                (0.0, 0.0, 0.0),
                (40.0, 0.2, 40.0),
                rgb(90, 120, 70),
            )?);
            world.set_place(ground, Place::new(0.0, -0.1, 0.0))?;
        }
        world.look_orbit(Vec3::new(0.0, 1.0, 5.0), 18.0, 20.0, 25.0);
        n += 1;
        match n {
            SPAWN | AGAIN => {
                let proto = world.spawn_instanced(lamp()?);
                world.set_instances(proto, &row(8, 5.0))?;
                batch = Some(proto);
            }
            // A touch: the same batch gets other places (updated in place).
            CHANGE => {
                if let Some(proto) = batch {
                    world.set_instances(proto, &row(8, 7.0))?;
                }
            }
            // A new city size: everything instanced goes at once.
            CLEAR => {
                if let Some(proto) = batch.take() {
                    world.despawn(proto);
                    rounds += 1;
                }
            }
            _ => {}
        }
        world.check(
            "the run got past the despawn and the respawn",
            n < END || rounds >= 1,
        );
        world.report("frames", n);
        Ok(())
    })
}
