//! Water: a valley with a lake, a spring that finds its own way down the
//! hills, and a sealed cave beside the lake. Everything is one simulation.
//!
//! Right mouse + move to look, WASD to move, Shift faster. G switches between
//! flying (Space/E up, Ctrl/Q down) and a body that walks, swims and dives
//! (Space jumps or rises, Ctrl dives; look down to dive while swimming).
//! Left click digs under the mouse cursor (dig from the lake toward the
//! cave and watch it flood), F builds rock, R pours a big bucket of water,
//! B drops a wooden crate that breaks into floating planks, T toggles rain.
//!
//! `ENGINE_WATER_VIEW=lake|shore|under|bed|river|cave` picks a starting view
//! (for screenshots), `ENGINE_WATER_LOOK=lake|ocean|tropical|river|swamp`
//! the water colour, `ENGINE_WATER_BODY=1` starts as the swimming body and
//! `ENGINE_WATER_CRATE=x,y,z` drops a crate there on the third frame.
use engine::advanced::Volume;
use engine::prelude::*;
use engine::water::{SwimState, Swimmer, WaterLook};
use glam::{Vec2, Vec3};

const SIZE: f32 = 256.0;
const LAKE_LEVEL: f32 = 14.0;
const LAKE: Vec2 = Vec2::new(128.0, 150.0);

fn ground_height(noise: &Noise, x: f32, z: f32) -> f32 {
    let hills = noise.fbm2(x * 0.008, z * 0.008, 5, 2.0, 0.5);
    let mut h = 24.0 + hills * 14.0;
    // A wide basin for the lake.
    let d = Vec2::new(x, z).distance(LAKE) / 70.0;
    let basin = (1.0 - d * d).max(0.0);
    h -= basin * basin * 22.0;
    // Hills rise toward the far edge, where the spring is.
    h += ((60.0 - z) / 60.0).max(0.0) * 16.0;
    h
}

/// Distance in XZ from `p` to the stream bed, which winds from the spring
/// down to the lake.
fn stream_distance(noise: &Noise, x: f32, z: f32) -> f32 {
    let (a, b) = (Vec2::new(64.0, 22.0), Vec2::new(118.0, 96.0));
    let ab = b - a;
    let t = ((Vec2::new(x, z) - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    let meander = noise.fbm2(t * 3.0, 7.7, 2, 2.0, 0.5) * 14.0 * (t * (1.0 - t) * 4.0);
    let side = Vec2::new(-ab.y, ab.x).normalize();
    (Vec2::new(x, z) - (a + ab * t + side * meander)).length()
}

fn density(noise: &Noise, p: Vec3) -> f32 {
    // A stream bed: a rounded channel 2.5 m deep.
    let s = stream_distance(noise, p.x, p.z) / 5.0;
    let channel = (1.0 - s * s).max(0.0) * 2.5;
    let ground = ground_height(noise, p.x, p.z) - channel - p.y;
    // A sealed cave beside the lake, its floor below the lake level.
    let c = (p - Vec3::new(196.0, 9.0, 150.0)) / Vec3::new(14.0, 5.0, 10.0);
    let cave = (c.length() - 1.0) * 5.0;
    ground.min(cave).clamp(-6.0, 6.0)
}

/// Starting views: (x, z, height, yaw, pitch). Height is above the ground,
/// except under water, where it is below the lake surface.
fn view(noise: &Noise, name: &str) -> (Vec3, f32, f32) {
    let above =
        |x: f32, z: f32, h: f32| Vec3::new(x, ground_height(noise, x, z).max(LAKE_LEVEL) + h, z);
    match name {
        "shore" => (above(96.0, 112.0, 1.8), 45.0, -10.0),
        "under" => (Vec3::new(112.0, LAKE_LEVEL - 1.5, 118.0), 40.0, 30.0),
        "bed" => {
            // Over the slope where the lake is about 5 m deep, facing the shore.
            let dir = Vec2::new(-0.6, -0.8);
            let r = (0..70)
                .map(|r| r as f32)
                .find(|&r| {
                    let p = LAKE + dir * r;
                    ground_height(noise, p.x, p.y) > LAKE_LEVEL - 5.0
                })
                .unwrap_or(30.0);
            let p = LAKE + dir * (r - 6.0);
            let yaw = dir.x.atan2(dir.y).to_degrees();
            (Vec3::new(p.x, LAKE_LEVEL - 1.2, p.y), yaw, -22.0)
        }
        "river" => (above(40.0, 4.0, 14.0), 40.0, -25.0),
        "cave" => (above(176.0, 128.0, 8.0), 50.0, -20.0),
        _ => (above(60.0, 70.0, 16.0), 50.0, -20.0),
    }
}

/// "x,y,z" from an environment variable.
#[derive(Clone, Copy)]
struct Vec3Text(Vec3);

impl std::str::FromStr for Vec3Text {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
        (v.len() == 3)
            .then(|| Vec3Text(Vec3::new(v[0], v[1], v[2])))
            .ok_or(())
    }
}

fn main() -> EngineResult<()> {
    let view_name = std::env::var("ENGINE_WATER_VIEW").unwrap_or_default();
    let (eye, yaw, pitch) = view(&Noise::new(5), &view_name);
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(12.0);
    let mut body: Option<Swimmer> = std::env::var_os("ENGINE_WATER_BODY").map(|_| {
        let mut b = Swimmer::new(eye - Vec3::Y * 1.65, yaw);
        b.pitch_deg = pitch;
        b
    });
    let drop_crate_at = std::env::var("ENGINE_WATER_CRATE")
        .ok()
        .and_then(|v| v.parse::<Vec3Text>().ok());
    let mut frame_no = 0u32;
    let mut raining = false;
    let mut crates = 0u32;

    Engine::run("water", move |world, frame| {
        if frame.first {
            let noise = Noise::new(5);
            let mut terrain = VoxelTerrain::new(Volume::new(1.0))
                .with_material(SurfaceMaterial::STONE)
                .with_color(rgb(140, 128, 108));
            terrain.generate(
                Vec3::new(0.0, -16.0, 0.0),
                Vec3::new(SIZE, 80.0, SIZE),
                |p| density(&noise, p),
            );
            let started = std::time::Instant::now();
            let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(SIZE), 1.0, -16.0, 80.0);
            let look = match std::env::var("ENGINE_WATER_LOOK").as_deref() {
                Ok("ocean") => WaterLook::ocean(),
                Ok("tropical") => WaterLook::tropical(),
                Ok("river") => WaterLook::river(),
                Ok("swamp") => WaterLook::swamp(),
                _ => WaterLook::lake(),
            };
            let mut water = Water::new(region, &terrain).with_look(look);
            water.flood(Vec3::new(LAKE.x, 5.0, LAKE.y), LAKE_LEVEL);
            // A spring high in the hills; the water picks its own path down.
            let spring = Vec3::new(64.0, 60.0, 22.0);
            water.add_source(WaterSource {
                position: spring,
                rate: 2.5,
            });
            // Let the stream reach the lake before the first frame.
            for _ in 0..60 * 40 {
                water.update(1.0 / 60.0);
            }
            eprintln!(
                "water: {} cells, {:.0} m3, warmed up in {:.1} s",
                water.cell_count(),
                water.total_volume(),
                started.elapsed().as_secs_f32()
            );
            world.set_voxel_terrain(Some(terrain));
            world.set_water(Some(water));
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.5)));
            world.set_view_distance(700.0)?;
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
        let hit = world
            .voxel_terrain()
            .and_then(|t| t.raycast(eye, aim, 300.0));
        if let Some(hit) = hit {
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
            if input.pressed(Key::R) {
                if let Some(water) = world.water_mut() {
                    water.pour(hit + Vec3::Y * 1.0, 400.0);
                }
            }
        }
        frame_no += 1;
        let scripted_crate = drop_crate_at.filter(|_| frame_no == 3).map(|t| t.0);
        if input.pressed(Key::B) || scripted_crate.is_some() {
            // A crate high above the aim point, broken into light planks.
            let target = scripted_crate.or(hit).unwrap_or(eye + aim * 20.0);
            let at = Vec3::new(target.x, target.y.max(LAKE_LEVEL) + 8.0, target.z);
            let crate_id = world.spawn(Mesh::box_at(at, Vec3::splat(1.6), rgb(150, 110, 70))?);
            crates += 1;
            world.shatter(
                crate_id,
                at + Vec3::Y,
                Shatter {
                    pieces: 10,
                    speed: 2.0,
                    keep_seconds: Some(90.0),
                    seed: crates,
                    density: 0.55,
                    ..Shatter::default()
                },
            )?;
        }
        if input.pressed(Key::T) {
            raining = !raining;
        }
        if raining {
            if let Some(water) = world.water_mut() {
                // 60 mm per minute: a cloudburst.
                water.rain(0.001 * frame.dt);
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
