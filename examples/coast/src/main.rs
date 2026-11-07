//! Coast: the open sea sends swell onto a beach. The waves are part of the
//! water simulation, so they physically run up the sand and pull back,
//! wash into a sea cave and over anything you dig or build. Past the
//! simulated area the sea goes on to the horizon.
//!
//! Right mouse + move to look, WASD to move, Shift faster. G switches between
//! flying (Space/E up, Ctrl/Q down) and a body that walks, swims and dives.
//! Left click digs, F builds rock, 1/2/3 sets a calm, moderate or rough sea.
//!
//! `ENGINE_COAST_VIEW=beach|sea|cave|surf` picks a starting view,
//! `ENGINE_COAST_SEA=calm|moderate|rough` the swell and `ENGINE_WATER_BODY=1`
//! starts as the swimming body.
use engine::advanced::Volume;
use engine::prelude::*;
use engine::water::{Sea, SeaEdges, Swell, SwimState, Swimmer, WaterLook};
use glam::{Vec2, Vec3};

const SIZE: f32 = 256.0;
const SEA_LEVEL: f32 = 0.0;

/// Where the dry beach starts at `z`.
fn shoreline(noise: &Noise, z: f32) -> f32 {
    170.0 + (z * 0.025).sin() * 12.0 + noise.fbm2(z * 0.02, 3.3, 3, 2.0, 0.5) * 8.0
}

fn ground_height(noise: &Noise, x: f32, z: f32) -> f32 {
    let s = x - shoreline(noise, z);
    // A sandy slope out to sea, a steeper rise to the dunes inland.
    let mut h = if s < 0.0 {
        s * 0.06
    } else {
        s * 0.06 + (s * 0.02).powi(2) * 6.0
    };
    h = h.max(-12.0);
    h += noise.fbm2(x * 0.05, z * 0.05, 3, 2.0, 0.5) * 0.4;
    // A rocky headland running out to sea.
    let ridge = ((z - 60.0).abs() / 16.0).min(1.0);
    let reach = ((x - 110.0) / 30.0).clamp(0.0, 1.0);
    if reach > 0.0 {
        let top = reach * (15.0 + noise.fbm2(x * 0.08, z * 0.08, 3, 2.0, 0.5) * 3.0) - 6.0;
        h = h.max((1.0 - ridge * ridge) * top - 8.0 * ridge * ridge);
    }
    h
}

fn density(noise: &Noise, p: Vec3) -> f32 {
    let ground = ground_height(noise, p.x, p.z) - p.y;
    // A sea cave through the headland, its floor just below the sea.
    let t = ((p.x - 112.0) / 60.0).clamp(0.0, 1.0);
    let axis = Vec3::new(112.0 + t * 60.0, 0.8, 58.0 + t * 6.0);
    let d = p - axis;
    let cave = (Vec2::new(d.y * 1.3, d.z).length() - 3.2) * 2.0;
    ground.min(cave).clamp(-6.0, 6.0)
}

fn sea(kind: &str) -> Sea {
    let edges = SeaEdges {
        min_x: true,
        max_x: false,
        min_z: true,
        max_z: true,
    };
    let base = Sea::calm(SEA_LEVEL).with_edges(edges);
    match kind {
        "calm" => base.with_swell(Swell {
            height: 0.3,
            period: 7.0,
            direction_deg: 95.0,
        }),
        "rough" => base
            .with_swell(Swell {
                height: 1.8,
                period: 11.0,
                direction_deg: 90.0,
            })
            .with_swell(Swell {
                height: 0.7,
                period: 6.5,
                direction_deg: 70.0,
            }),
        _ => base
            .with_swell(Swell {
                height: 1.0,
                period: 9.0,
                direction_deg: 90.0,
            })
            .with_swell(Swell {
                height: 0.35,
                period: 6.0,
                direction_deg: 70.0,
            }),
    }
}

/// Starting views: eye, yaw, pitch.
fn view(noise: &Noise, name: &str) -> (Vec3, f32, f32) {
    let above =
        |x: f32, z: f32, h: f32| Vec3::new(x, ground_height(noise, x, z).max(SEA_LEVEL) + h, z);
    match name {
        "beach" => (
            above(shoreline(noise, 150.0) + 4.0, 150.0, 1.8),
            -95.0,
            -9.0,
        ),
        "surf" => (
            above(shoreline(noise, 110.0) - 22.0, 110.0, 3.0),
            30.0,
            -10.0,
        ),
        "cave" => (above(92.0, 70.0, 2.5), 100.0, -4.0),
        _ => (above(60.0, 200.0, 40.0), 120.0, -20.0),
    }
}

fn main() -> EngineResult<()> {
    let view_name = std::env::var("ENGINE_COAST_VIEW").unwrap_or_default();
    let sea_kind = std::env::var("ENGINE_COAST_SEA").unwrap_or_default();
    let (eye, yaw, pitch) = view(&Noise::new(11), &view_name);
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(12.0);
    let mut body: Option<Swimmer> = std::env::var_os("ENGINE_WATER_BODY").map(|_| {
        let mut b = Swimmer::new(eye - Vec3::Y * 1.65, yaw);
        b.pitch_deg = pitch;
        b
    });

    Engine::run("coast", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            let noise = Noise::new(11);
            let mut terrain = VoxelTerrain::new(Volume::new(1.0))
                .with_material(SurfaceMaterial::SAND)
                .with_color(rgb(214, 196, 158));
            terrain.generate(
                // The sea floor goes on past the simulated area.
                Vec3::new(-192.0, -16.0, -192.0),
                Vec3::new(SIZE + 64.0, 48.0, SIZE + 192.0),
                |p| density(&noise, p),
            );
            let started = std::time::Instant::now();
            let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(SIZE), 1.0, -16.0, 48.0);
            let mut water = Water::new(region, &terrain).with_look(WaterLook::ocean());
            water.flood_sea(SEA_LEVEL);
            water.set_sea(Some(sea(&sea_kind)));
            // Let the swell build up and reach the beach before the first frame.
            for _ in 0..60 * 30 {
                water.update(1.0 / 60.0);
            }
            eprintln!(
                "coast: {} cells, warmed up in {:.1} s",
                water.cell_count(),
                started.elapsed().as_secs_f32()
            );
            world.set_voxel_terrain(Some(terrain));
            world.set_water(Some(water));
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.0)));
            world.set_view_distance(4000.0)?;
        }

        for (key, kind) in [
            (Key::Digit1, "calm"),
            (Key::Digit2, "moderate"),
            (Key::Digit3, "rough"),
        ] {
            if frame.input.pressed(key) {
                if let Some(water) = world.water_mut() {
                    water.set_sea(Some(sea(kind)));
                }
            }
        }
        if frame.input.pressed(Key::G) {
            body = match body.take() {
                Some(b) => {
                    camera.position = b.eye();
                    camera.yaw_deg = b.yaw_deg;
                    camera.pitch_deg = b.pitch_deg;
                    None
                }
                None => {
                    let mut b = Swimmer::new(camera.position - Vec3::Y * 1.65, camera.yaw_deg);
                    b.pitch_deg = camera.pitch_deg;
                    Some(b)
                }
            };
        }
        match body.as_mut() {
            Some(b) => {
                b.update(world, frame);
            }
            None => camera.update(world, frame),
        }
        // Dig, build and pour under the mouse cursor.
        let (eye, aim) = world.cursor_ray(frame);
        let input = &frame.input;
        if let Some(hit) = world
            .voxel_terrain()
            .and_then(|t| t.raycast(eye, aim, 300.0))
        {
            if input.mouse_clicked(MouseButton::Left) {
                world
                    .voxel_terrain_mut()
                    .expect("terrain")
                    .dig_sphere(hit, 3.5);
            }
            if input.pressed(Key::F) {
                world
                    .voxel_terrain_mut()
                    .expect("terrain")
                    .add_sphere(hit, 3.0);
            }
        }
        if let Some(b) = body.as_ref() {
            let state = match b.state() {
                SwimState::OnGround => "walking",
                SwimState::InAir => "in the air",
                SwimState::Wading => "wading",
                SwimState::Swimming => "swimming",
                SwimState::Diving => "diving",
            };
            engine::egui::Window::new("body")
                .title_bar(false)
                .resizable(false)
                .anchor(engine::egui::Align2::LEFT_BOTTOM, [12.0, -12.0])
                .show(frame.ui.ctx(), |ui| {
                    ui.label(state);
                    ui.add(
                        engine::egui::ProgressBar::new(b.breath_fraction())
                            .desired_width(160.0)
                            .text(format!("air {:.0} s", b.breath())),
                    );
                });
        }
        Ok(())
    })
}
