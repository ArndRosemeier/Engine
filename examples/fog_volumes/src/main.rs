//! Colour grading and fog by zone: a warm, golden old town on the west and a
//! smog-choked factory district on the east. The camera flies back and forth
//! between them by itself; crossing the edge, the light, colour and air blend
//! over a few seconds instead of switching at a line.
//!
//! Controls: right mouse + move to look, WASD to fly (stops the automatic
//! flight), Shift fast.
//! - Space: automatic flight on/off.
//! - T: cycle the transition time (0.5, 2.5, 6 seconds).
//! - G: volumes on/off (compare with the plain scene).
//!
//! `ENGINE_FOG_AT=warm|edge|smog` starts parked in one place.
use engine::prelude::*;
use glam::Vec3;

const WEST: f32 = -160.0;
const EAST: f32 = 200.0;
const HEIGHT: f32 = 26.0;
const SPEED: f32 = 18.0;

fn rules() -> TerrainRules {
    TerrainRules {
        base_height: 0.0,
        hill_height: 3.0,
        hill_scale: 0.01,
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

fn warm_volume() -> FogVolume {
    FogVolume::new((WEST, 60.0, 0.0), (140.0, 120.0, 160.0))
        .grade(ColorGrade::warm())
        .fog(rgb(255, 214, 160), 900.0)
        .blend(40.0)
}

fn smog_volume() -> FogVolume {
    FogVolume::new((EAST + 40.0, 60.0, 0.0), (160.0, 120.0, 160.0))
        .grade(ColorGrade::smog())
        .fog(rgb(128, 112, 84), 110.0)
        .blend(50.0)
}

fn main() -> EngineResult<()> {
    let ground = HeightField::new(rules());
    let parked = std::env::var("ENGINE_FOG_AT").ok();
    let start_x = match parked.as_deref() {
        Some("warm") => WEST - 40.0,
        Some("smog") => EAST + 40.0,
        Some("edge") => 20.0,
        _ => WEST,
    };
    let mut auto = parked.is_none();
    let mut camera = FlyCamera::new(
        (start_x, ground.height_at(start_x, -60.0) + HEIGHT, -60.0),
        90.0,
        -14.0,
    )
    .with_speed(14.0);
    let mut heading = 1.0f32;
    let transitions = [0.5f32, 2.5, 6.0];
    let mut transition = 1;
    let mut volumes: Vec<FogVolumeId> = Vec::new();

    Engine::run("fog_volumes", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.5)));
            world.set_proc_terrain(ProcTerrain::gpu_clipmap(rules(), ClipmapConfig::default()));
            world.set_zone_transition(transitions[transition])?;
            volumes.push(world.add_fog_volume(warm_volume())?);
            volumes.push(world.add_fog_volume(smog_volume())?);
            let mut rng = Rng(11);
            // The old town: low houses in warm plaster and terracotta.
            for gx in -6..=6 {
                for gz in -6..=6 {
                    if rng.next() < 0.25 {
                        continue;
                    }
                    let x = WEST + gx as f32 * 18.0 + rng.range(-3.0, 3.0);
                    let z = gz as f32 * 18.0 + rng.range(-3.0, 3.0);
                    let (w, d, h) = (
                        rng.range(7.0, 12.0),
                        rng.range(7.0, 12.0),
                        rng.range(5.0, 11.0),
                    );
                    let y = ground.height_at(x, z);
                    let walls = [rgb(232, 206, 160), rgb(220, 180, 130), rgb(240, 225, 190)];
                    world.spawn(Mesh::box_at(
                        Vec3::new(x, y + h * 0.5, z),
                        Vec3::new(w, h, d),
                        walls[(rng.next() * 3.0) as usize % 3],
                    )?);
                    world.spawn(Mesh::box_at(
                        Vec3::new(x, y + h + 0.6, z),
                        Vec3::new(w + 0.6, 1.2, d + 0.6),
                        rgb(178, 84, 52),
                    )?);
                }
            }
            // The factory district: dark halls, tanks and tall chimneys.
            for gx in -5..=5 {
                for gz in -5..=5 {
                    if rng.next() < 0.2 {
                        continue;
                    }
                    let x = EAST + 40.0 + gx as f32 * 26.0 + rng.range(-4.0, 4.0);
                    let z = gz as f32 * 26.0 + rng.range(-4.0, 4.0);
                    let y = ground.height_at(x, z);
                    let pick = rng.next();
                    if pick < 0.6 {
                        let (w, d, h) = (
                            rng.range(14.0, 22.0),
                            rng.range(10.0, 18.0),
                            rng.range(8.0, 16.0),
                        );
                        world.spawn(Mesh::box_at(
                            Vec3::new(x, y + h * 0.5, z),
                            Vec3::new(w, h, d),
                            rgb(92, 88, 82),
                        )?);
                    } else if pick < 0.8 {
                        let h = rng.range(35.0, 60.0);
                        world.spawn(Mesh::box_at(
                            Vec3::new(x, y + h * 0.5, z),
                            Vec3::new(3.0, h, 3.0),
                            rgb(120, 70, 55),
                        )?);
                    } else {
                        let r = rng.range(5.0, 8.0);
                        world.spawn(Mesh::box_at(
                            Vec3::new(x, y + r, z),
                            Vec3::new(r * 2.0, r * 2.0, r * 2.0),
                            rgb(150, 150, 140),
                        )?);
                    }
                }
            }
        }

        if frame.input.pressed(Key::Space) {
            auto = !auto;
        }
        if [Key::W, Key::A, Key::S, Key::D]
            .iter()
            .any(|k| frame.input.down(*k))
        {
            auto = false;
        }
        if frame.input.pressed(Key::T) {
            transition = (transition + 1) % transitions.len();
            world.set_zone_transition(transitions[transition])?;
        }
        if frame.input.pressed(Key::G) {
            if volumes.is_empty() {
                volumes.push(world.add_fog_volume(warm_volume())?);
                volumes.push(world.add_fog_volume(smog_volume())?);
            } else {
                for id in volumes.drain(..) {
                    world.remove_fog_volume(id);
                }
            }
        }
        if auto {
            // Back and forth from the old town to the factories.
            let x = camera.position.x + heading * SPEED * frame.dt;
            if (heading > 0.0 && x > EAST + 60.0) || (heading < 0.0 && x < WEST - 40.0) {
                heading = -heading;
            }
            camera.position.x = x;
            camera.position.y = ground.height_at(x, camera.position.z) + HEIGHT;
            let target_yaw = if heading > 0.0 { 90.0 } else { -90.0 };
            camera.yaw_deg += (target_yaw - camera.yaw_deg) * (frame.dt * 1.5).min(1.0);
        }
        camera.update_above(world, frame, |x, z| ground.height_at(x, z));
        world.set_proc_focus(camera.position);

        let zone = |id: Option<&FogVolumeId>| id.map_or(0.0, |id| world.fog_volume_blend(*id));
        engine::egui::Window::new("fog volumes")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   warm {:.0}%   smog {:.0}%",
                    frame.fps,
                    zone(volumes.first()) * 100.0,
                    zone(volumes.get(1)) * 100.0
                ));
                ui.label(format!(
                    "Space: auto flight {}   T: transition {:.1} s   G: volumes {}",
                    if auto { "on" } else { "off" },
                    transitions[transition],
                    if volumes.is_empty() { "off" } else { "on" }
                ));
            });
        Ok(())
    })
}
