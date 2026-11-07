//! A fixed 30 Hz simulation drawn at whatever the screen manages.
//!
//! The game logic here (balls on a carousel, a bouncing row, a spinning bar)
//! runs at a fixed tick like a game server would. The front set is drawn
//! between the last two steps and moves smoothly at 144 FPS and beyond. The
//! back set (dimmer) shows the raw steps: it stutters at 30 Hz.
//!
//! Controls: right mouse + move to look, WASD to fly.
//! - I: interpolation on/off for the front set.
//! - 1 / 2 / 3: simulation at 10, 30 or 60 Hz.
//!
//! `ENGINE_SIM_HZ` sets the starting rate (default 30).
use engine::prelude::*;
use glam::Vec3;

const BALLS: usize = 10;

/// The game's own state, stepped only at the fixed tick.
#[derive(Default)]
struct Sim {
    t: f32,
}

impl Sim {
    fn step(&mut self, dt: f32) {
        self.t += dt;
    }

    /// Carousel ball `i`: orbiting and bobbing.
    fn ball(&self, i: usize) -> Vec3 {
        let a = self.t * 1.6 + i as f32 * std::f32::consts::TAU / BALLS as f32;
        Vec3::new(
            a.cos() * 5.0,
            1.2 + (self.t * 3.0 + i as f32).sin().abs() * 1.5,
            a.sin() * 5.0,
        )
    }

    /// A box sliding back and forth fast along x.
    fn slider(&self) -> Vec3 {
        Vec3::new((self.t * 2.4).sin() * 7.0, 0.5, -8.0)
    }

    fn bar_yaw(&self) -> f32 {
        self.t * 120.0
    }
}

struct Things {
    smooth: Vec<EntityId>,
    raw: Vec<EntityId>,
}

fn spawn(world: &mut World, dim: f32) -> EngineResult<Vec<EntityId>> {
    let mut out = Vec::new();
    for i in 0..BALLS {
        let hue = i as f32 / BALLS as f32;
        let c = |o: f32| {
            ((((hue + o) * std::f32::consts::TAU).sin() * 0.5 + 0.5) * 200.0 * dim + 30.0) as u8
        };
        out.push(world.spawn(Mesh::ball(Vec3::ZERO, 0.45, rgb(c(0.0), c(0.33), c(0.66)))?));
    }
    out.push(world.spawn(Mesh::box_at(
        Vec3::ZERO,
        Vec3::splat(1.0),
        rgb((220.0 * dim) as u8, (180.0 * dim) as u8, 40),
    )?));
    out.push(world.spawn(Mesh::box_at(
        Vec3::ZERO,
        Vec3::new(9.0, 0.3, 0.5),
        rgb(60, (140.0 * dim) as u8, (220.0 * dim) as u8),
    )?));
    Ok(out)
}

/// Pose of thing `k` in the simulation, with the set's offset.
fn pose(sim: &Sim, k: usize, offset: Vec3) -> Place {
    let p = if k < BALLS {
        sim.ball(k)
    } else if k == BALLS {
        sim.slider()
    } else {
        Vec3::new(0.0, 4.0, 0.0)
    };
    let p = p + offset;
    let place = Place::new(p.x, p.y, p.z);
    if k == BALLS + 1 {
        place.with_yaw_deg(sim.bar_yaw())
    } else {
        place
    }
}

const BACK: Vec3 = Vec3::new(0.0, 0.0, 18.0);

fn main() -> EngineResult<()> {
    let hz: f32 = std::env::var("ENGINE_SIM_HZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0);
    let mut camera = FlyCamera::new((0.0, 9.0, -22.0), 0.0, -14.0).with_speed(10.0);
    let mut sim = Sim::default();
    let mut things: Option<Things> = None;
    let mut interpolate = true;

    Engine::run("fixed_tick", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_sim_rate(Some(hz))?;
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
            world.spawn(Shape::box_at(
                (0.0, -0.05, 6.0),
                (60.0, 0.1, 60.0),
                rgb(96, 110, 90),
            )?);
            things = Some(Things {
                smooth: spawn(world, 1.0)?,
                raw: spawn(world, 0.55)?,
            });
        }
        camera.update(world, frame);
        let t = things.as_ref().expect("spawned");

        for (key, rate) in [
            (Key::Digit1, 10.0),
            (Key::Digit2, 30.0),
            (Key::Digit3, 60.0),
        ] {
            if frame.input.pressed(key) {
                world.set_sim_rate(Some(rate))?;
            }
        }
        if frame.input.pressed(Key::I) {
            interpolate = !interpolate;
            if !interpolate {
                for id in &t.smooth {
                    world.clear_sim_pose(*id);
                }
            }
        }

        // The fixed tick: as many steps as are due this frame.
        for _ in 0..frame.sim.steps {
            world.begin_sim_step();
            sim.step(frame.sim.dt);
            for (k, id) in t.smooth.iter().enumerate() {
                if interpolate {
                    world.set_sim_place(*id, pose(&sim, k, Vec3::ZERO))?;
                } else {
                    world.set_place(*id, pose(&sim, k, Vec3::ZERO))?;
                }
            }
            for (k, id) in t.raw.iter().enumerate() {
                world.set_place(*id, pose(&sim, k, BACK))?;
            }
        }

        engine::egui::Window::new("fixed tick")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "simulation {:.0} Hz, drawing {:.0} FPS",
                    world.sim_rate().unwrap_or(0.0),
                    frame.fps
                ));
                ui.label(format!(
                    "front: {}   back: raw steps",
                    if interpolate {
                        "interpolated"
                    } else {
                        "raw steps"
                    }
                ));
                ui.label("I: interpolation   1/2/3: 10/30/60 Hz");
            });
        Ok(())
    })
}
