//! Crowds: a thousand people walking about a town square. Far away they
//! are instanced and posed on the GPU from baked walk and idle clips (one
//! draw per mesh for everybody); the nearest become full characters at the
//! same pose and step, and drop back into the crowd as you move on.
//!
//! The camera strolls through the crowd by itself; take over with the mouse
//! (right button + move to look) and WASD.
//! - Space: automatic stroll on/off.
//! - N: full characters near the camera on/off (compare: everyone instanced).
//! - F: far mesh: simplified (5%) or the full character mesh.
//!
//! `ENGINE_CROWD_COUNT=n` sets the number of people (1000 by default);
//! `ENGINE_CROWD_VIEW=above` starts parked above the square;
//! `ENGINE_CROWD_NEAR=0` starts with everyone instanced.
use engine::prelude::*;
use glam::{Vec2, Vec3};
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);
const HALF: f32 = 90.0;

fn rules() -> TerrainRules {
    TerrainRules {
        base_height: 0.0,
        hill_height: 1.5,
        hill_scale: 0.008,
        lake_threshold: 2.0,
        water_level: -50.0,
        ..demo_terrain_rules()
    }
}

/// A tiny deterministic random sequence.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
}

struct Person {
    id: CrowdAgentId,
    at: Vec2,
    goal: Vec2,
    yaw: f32,
    speed: f32,
    /// Seconds left standing still.
    wait: f32,
}

fn profile(near: bool, full_far: bool) -> CrowdProfile {
    let p = CrowdProfile::new("idle", "walk")
        .walk_speed(1.2)
        .near(if near { 18.0 } else { 0.0 }, 24);
    if full_far {
        p.far_detail(1.0)
    } else {
        p
    }
}

fn main() -> EngineResult<()> {
    let model = Arc::new(AnimatedModel::load(CHARACTER)?);
    let ground = HeightField::new(rules());
    let count: usize = std::env::var("ENGINE_CROWD_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let mut camera = FlyCamera::new((-HALF + 5.0, 0.0, -20.0), 70.0, -6.0).with_speed(6.0);
    camera.position.y = ground.height_at(camera.position.x, camera.position.z) + 2.2;
    let above = std::env::var("ENGINE_CROWD_VIEW").is_ok_and(|v| v == "above");
    if above {
        camera = FlyCamera::new((0.0, 30.0, -125.0), 0.0, -14.0).with_speed(12.0);
    }
    let mut stroll = !above;
    let mut near = std::env::var("ENGINE_CROWD_NEAR").map_or(true, |v| v != "0");
    let mut full_far = false;
    let mut crowd: Option<CrowdId> = None;
    let mut people: Vec<Person> = Vec::new();
    let mut rng = Rng(3);

    Engine::run("crowds", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        let respawn = frame.input.pressed(Key::N) || frame.input.pressed(Key::F);
        if frame.input.pressed(Key::N) {
            near = !near;
        }
        if frame.input.pressed(Key::F) {
            full_far = !full_far;
        }
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
            world.set_proc_terrain(ProcTerrain::gpu_clipmap(rules(), ClipmapConfig::default()));
            world.set_grass_scale(0.25)?;
            // A paved square and a few landmarks to walk around.
            world.spawn(Mesh::box_at(
                Vec3::new(0.0, ground.height_at(0.0, 0.0) + 2.5, 0.0),
                Vec3::new(6.0, 5.0, 6.0),
                rgb(190, 180, 160),
            )?);
            for (x, z) in [(-45.0, -40.0), (50.0, 30.0), (-30.0, 55.0), (40.0, -50.0)] {
                world.spawn(Mesh::box_at(
                    Vec3::new(x, ground.height_at(x, z) + 6.0, z),
                    Vec3::new(10.0, 12.0, 10.0),
                    rgb(200, 170, 140),
                )?);
            }
            for i in 0..count {
                let at = Vec2::new(rng.range(-HALF, HALF), rng.range(-HALF, HALF));
                let goal = Vec2::new(rng.range(-HALF, HALF), rng.range(-HALF, HALF));
                let wait = if i % 5 == 0 { rng.range(0.0, 6.0) } else { 0.0 };
                people.push(Person {
                    id: CrowdAgentId(0),
                    at,
                    goal,
                    yaw: 0.0,
                    speed: rng.range(1.0, 1.5),
                    wait,
                });
            }
        }
        if frame.first || respawn {
            if let Some(c) = crowd.take() {
                world.remove_crowd(c);
            }
            let c = world.add_crowd(model.clone(), profile(near, full_far))?;
            for p in people.iter_mut() {
                let y = ground.height_at(p.at.x, p.at.y);
                p.id = world.add_crowd_agent(c, Place::new(p.at.x, y, p.at.y))?;
            }
            crowd = Some(c);
        }
        let Some(c) = crowd else { return Ok(()) };

        // Wander: walk to a goal, sometimes stand a while, pick a new goal.
        for p in people.iter_mut() {
            let mut speed = 0.0;
            if p.wait > 0.0 {
                p.wait -= frame.dt;
            } else {
                let to = p.goal - p.at;
                let d = to.length();
                if d < 0.5 {
                    p.goal = Vec2::new(rng.range(-HALF, HALF), rng.range(-HALF, HALF));
                    if rng.next() < 0.3 {
                        p.wait = rng.range(1.5, 6.0);
                    }
                } else {
                    let dir = to / d;
                    p.at += dir * p.speed * frame.dt;
                    let goal_yaw = dir.x.atan2(dir.y).to_degrees();
                    let turn = (goal_yaw - p.yaw + 540.0).rem_euclid(360.0) - 180.0;
                    p.yaw += turn * (frame.dt * 4.0).min(1.0);
                    speed = p.speed;
                }
            }
            let y = ground.height_at(p.at.x, p.at.y);
            world.set_crowd_agent(
                c,
                p.id,
                Place::new(p.at.x, y, p.at.y).with_yaw_deg(p.yaw),
                speed,
            )?;
        }

        if frame.input.pressed(Key::Space) {
            stroll = !stroll;
        }
        if [Key::W, Key::A, Key::S, Key::D]
            .iter()
            .any(|k| frame.input.down(*k))
        {
            stroll = false;
        }
        if stroll {
            // A slow loop through the square at walking pace.
            let t = 2.2 + world.time() * 0.025;
            let x = t.cos() * 55.0;
            let z = t.sin() * 55.0;
            let ahead = Vec2::new(-t.sin(), t.cos());
            camera.position = Vec3::new(x, ground.height_at(x, z) + 2.2, z);
            camera.yaw_deg = ahead.x.atan2(ahead.y).to_degrees() + 20.0;
        }
        camera.update_above(world, frame, |x, z| ground.height_at(x, z));
        world.set_proc_focus(camera.position);

        engine::egui::Window::new("crowds")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   people: {}   full characters: {}",
                    frame.fps,
                    world.crowd_agent_count(c),
                    world.crowd_near_count(c)
                ));
                ui.label(format!(
                    "Space: stroll {}   N: near characters {}   F: far mesh {}",
                    if stroll { "on" } else { "off" },
                    if near { "on" } else { "off" },
                    if full_far { "full" } else { "simplified" }
                ));
            });
        Ok(())
    })
}
