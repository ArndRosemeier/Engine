//! Physics playground: crates, a brick wall, dominoes, boulders rolling off a
//! hill and a spinning sweeper, all on voxel terrain you can blow holes in.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast.
//! - Left click: throw a ball at the mouse cursor.
//! - F: explosion at the cursor (crater, blast, shatters what it hits).
//! - X: shatter whatever is under the cursor.
//! - R: build the playground again.
//!
//! `ENGINE_PHYSICS_DEMO=1` plays a short script (a throw, the dominoes, an
//! explosion) so a screenshot run shows things moving; it's on by default
//! when `ENGINE_SCREENSHOT` is set.
use engine::advanced::Volume;
use engine::prelude::*;
use glam::Vec3;

/// Half the side of the voxel area, in metres.
const HALF: f32 = 48.0;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Ground height: a flat playground in a shallow bowl, with a hill to the
/// north that boulders roll down.
fn height(noise: &Noise, x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let rim = smoothstep(30.0, 46.0, r) * 7.0;
    let hill = 11.0 * (-((x * x + (z - 36.0).powi(2)) / 220.0)).exp();
    let bumps = noise.fbm2(x * 0.05, z * 0.05, 3, 2.0, 0.5) * 0.35 * smoothstep(12.0, 30.0, r);
    rim + hill + bumps
}

#[derive(Default)]
struct Playground {
    /// Things explosions and X can break.
    breakable: Vec<EntityId>,
    dominoes: Vec<EntityId>,
    sweeper: Option<EntityId>,
    everything: Vec<EntityId>,
}

fn spawn_body(
    world: &mut World,
    things: &mut Playground,
    mesh: Mesh,
    place: Place,
    body: Body,
    breakable: bool,
) -> EngineResult<EntityId> {
    let id = world.spawn(mesh);
    world.set_place(id, place)?;
    world.add_body(id, body)?;
    things.everything.push(id);
    if breakable {
        things.breakable.push(id);
    }
    Ok(id)
}

fn build(world: &mut World, noise: &Noise) -> EngineResult<Playground> {
    let mut p = Playground::default();
    let ground = |x: f32, z: f32| height(noise, x, z);
    let wood = [rgb(168, 120, 72), rgb(150, 104, 60), rgb(182, 136, 84)];
    let crate_body = Body::dynamic().density(0.6).friction(0.7);

    // A pyramid of crates in the boulders' path.
    let size = 1.0;
    for row in 0..5 {
        for i in 0..(5 - row) {
            let x = (i as f32 - (4 - row) as f32 * 0.5) * (size + 0.02);
            let z = 9.0;
            let y = ground(x, z) + size * 0.5 + row as f32 * size + 0.01;
            let mesh = Mesh::box_at(Vec3::ZERO, Vec3::splat(size), wood[(row + i) % 3])?;
            spawn_body(
                world,
                &mut p,
                mesh,
                Place::new(x, y, z),
                crate_body.clone(),
                true,
            )?;
        }
    }

    // A brick wall, joints staggered.
    let stone = [rgb(170, 160, 146), rgb(150, 142, 130), rgb(184, 172, 156)];
    let brick = Vec3::new(1.0, 0.5, 0.5);
    let wall_x = 9.0;
    let wall_z = 2.0;
    let base = ground(wall_x, wall_z) + 0.01;
    for row in 0..8 {
        let shift = if row % 2 == 0 { 0.0 } else { 0.5 };
        for i in 0..7 {
            let z = wall_z + (i as f32 - 3.0 + shift) * (brick.x + 0.01);
            let y = base + brick.y * 0.5 + row as f32 * (brick.y + 0.005);
            let mesh = Mesh::box_at(Vec3::ZERO, brick, stone[(row * 7 + i) % 3])?;
            spawn_body(
                world,
                &mut p,
                mesh,
                Place::new(wall_x, y, z).yaw_deg(90.0)?,
                Body::dynamic().density(2.0).friction(0.8),
                true,
            )?;
        }
    }

    // Dominoes along a curve in front of the camera.
    let paint = [rgb(220, 220, 214), rgb(40, 40, 44)];
    for i in 0..22 {
        let t = i as f32 / 21.0;
        let angle = -0.9 + t * 1.8;
        let (x, z) = (-9.0 + angle.sin() * 7.0, -4.0 + angle.cos() * 7.0 - 7.0);
        let tangent = Vec3::new(angle.cos(), 0.0, -angle.sin());
        let yaw = tangent.x.atan2(tangent.z).to_degrees();
        let y = ground(x, z) + 0.81;
        let mesh = Mesh::box_at(Vec3::ZERO, Vec3::new(0.8, 1.6, 0.2), paint[i % 2])?;
        let id = spawn_body(
            world,
            &mut p,
            mesh,
            Place::new(x, y, z).yaw_deg(yaw)?,
            Body::dynamic().density(0.7).friction(0.5),
            true,
        )?;
        p.dominoes.push(id);
    }

    // Boulders resting on the hill; a nudge (or a ball) sends them down.
    for i in 0..5 {
        let x = -6.0 + i as f32 * 3.0;
        let z = 30.0 + (i % 2) as f32 * 2.0;
        let r = 0.7 + (i % 3) as f32 * 0.25;
        let mesh = Mesh::ball(Vec3::ZERO, r, rgb(120, 116, 110))?;
        spawn_body(
            world,
            &mut p,
            mesh,
            Place::new(x, ground(x, z) + r + 0.05, z),
            Body::dynamic()
                .shape(BodyShape::Ball(r))
                .density(2.4)
                .friction(0.9),
            true,
        )?;
    }

    // A sweeper arm turning on a post: a kinematic body that shoves crates.
    let (sx, sz) = (-14.0, 8.0);
    let post = world.spawn(Mesh::box_at(
        Vec3::ZERO,
        Vec3::new(0.5, 1.6, 0.5),
        rgb(70, 70, 76),
    )?);
    world.set_place(post, Place::new(sx, ground(sx, sz) + 0.8, sz))?;
    world.add_body(post, Body::fixed())?;
    p.everything.push(post);
    let arm = world.spawn(Mesh::box_at(
        Vec3::ZERO,
        Vec3::new(9.0, 0.35, 0.35),
        rgb(200, 60, 40),
    )?);
    world.set_place(arm, Place::new(sx, ground(sx, sz) + 0.55, sz))?;
    world.add_body(arm, Body::kinematic().shape(BodyShape::BoundingBox))?;
    p.everything.push(arm);
    p.sweeper = Some(arm);
    for i in 0..10 {
        let a = i as f32 * 0.63;
        let (x, z) = (sx + a.cos() * 3.2, sz + a.sin() * 3.2);
        let mesh = Mesh::box_at(Vec3::ZERO, Vec3::splat(0.6), wood[i % 3])?;
        spawn_body(
            world,
            &mut p,
            mesh,
            Place::new(x, ground(x, z) + 0.31, z),
            crate_body.clone(),
            true,
        )?;
    }
    Ok(p)
}

fn main() -> EngineResult<()> {
    let noise = Noise::new(5);
    let demo = std::env::var("ENGINE_PHYSICS_DEMO")
        .map(|v| v != "0")
        .unwrap_or_else(|_| std::env::var_os("ENGINE_SCREENSHOT").is_some());
    let hour = std::env::var("ENGINE_SKY_HOUR")
        .ok()
        .and_then(|h| h.parse::<f32>().ok())
        .unwrap_or(15.5);
    let eye = Vec3::new(-3.0, height(&noise, -3.0, -22.0) + 6.5, -22.0);
    let mut camera = FlyCamera::new(eye, 8.0, -12.0).with_speed(10.0);
    let mut things = Playground::default();
    let mut time = 0.0_f32;
    let mut frame_no = 0u32;
    let mut shots = 0u32;
    let mut report = 0.0_f32;

    Engine::run("physics", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            let started = std::time::Instant::now();
            let mut terrain = VoxelTerrain::new(Volume::new(0.5))
                .with_material(SurfaceMaterial::STONE)
                .with_color(rgb(132, 124, 104));
            terrain.generate(
                Vec3::new(-HALF, -8.0, -HALF),
                Vec3::new(HALF, 24.0, HALF),
                |p| ((height(&noise, p.x, p.z) - p.y) * 2.0).clamp(-6.0, 6.0),
            );
            world.set_voxel_terrain(Some(terrain));
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            world.set_view_distance(600.0)?;
            things = build(world, &noise)?;
            eprintln!(
                "physics: {} bodies, ready in {:.1} s",
                world.body_count().0,
                started.elapsed().as_secs_f32()
            );
        }
        frame_no += 1;
        time += frame.dt;
        camera.update(world, frame);

        // The sweeper turns at a steady 50 degrees a second.
        if let Some(arm) = things.sweeper {
            let at = world.entity(arm)?.transform().w_axis;
            world.set_place(arm, Place::new(at.x, at.y, at.z).yaw_deg(time * 50.0)?)?;
        }

        let (eye, mut aim) = world.cursor_ray(frame);
        let mut throw = frame.input.mouse_clicked(MouseButton::Left);
        let mut explode = frame.input.pressed(Key::F);
        let shatter = frame.input.pressed(Key::X);
        if demo {
            // A ball at the wall, the first domino, then a blast by the crates.
            match frame_no {
                20 => {
                    throw = true;
                    aim = (Vec3::new(9.0, 2.0, 2.0) - eye).normalize();
                }
                24 => {
                    if let Some(&first) = things.dominoes.first() {
                        let fwd = world.entity(first)?.transform().z_axis.truncate();
                        let m = world.body_mass(first).unwrap_or(1.0);
                        world.push_at(
                            first,
                            fwd * 2.0 * m,
                            world.entity(first)?.transform().w_axis.truncate() + Vec3::Y * 0.6,
                        )?;
                    }
                }
                30 => {
                    explode = true;
                    aim = (Vec3::new(-1.0, 1.0, 8.0) - eye).normalize();
                }
                _ => {}
            }
        }

        if throw {
            shots += 1;
            let ball = world.spawn(Mesh::ball(Vec3::ZERO, 0.3, rgb(200, 40, 30))?);
            let start = eye + aim * 1.0;
            world.set_place(ball, Place::new(start.x, start.y, start.z))?;
            world.add_body(
                ball,
                Body::dynamic()
                    .shape(BodyShape::Ball(0.3))
                    .density(7.8)
                    .bounce(0.3)
                    .velocity(aim * 32.0)
                    .ccd(true),
            )?;
            things.everything.push(ball);
        }

        if explode || shatter {
            if let Some(hit) = world.cast_ray(eye, aim, 300.0) {
                shots += 1;
                let settings = Shatter {
                    pieces: 10,
                    speed: if explode { 9.0 } else { 4.0 },
                    seed: shots,
                    density: 1.5,
                    ..Shatter::default()
                };
                if shatter {
                    if let Some(id) = hit.entity.filter(|id| things.breakable.contains(id)) {
                        things.breakable.retain(|b| *b != id);
                        world.shatter(id, hit.point, settings)?;
                    }
                } else {
                    // Break what is right at the blast, then throw everything
                    // nearby outward and dig a crater.
                    let near: Vec<EntityId> = things
                        .breakable
                        .iter()
                        .copied()
                        .filter(|id| {
                            world.entity(*id).is_ok_and(|e| {
                                e.transform().w_axis.truncate().distance(hit.point) < 1.6
                            })
                        })
                        .collect();
                    for id in near {
                        things.breakable.retain(|b| *b != id);
                        world.shatter(id, hit.point, settings)?;
                    }
                    world.blast(hit.point, 7.0, 12.0);
                    if hit.entity.is_none() {
                        if let Some(t) = world.voxel_terrain_mut() {
                            t.dig_sphere(hit.point, 2.2);
                        }
                    }
                }
            }
        }

        if frame.input.pressed(Key::R) {
            for id in things.everything.drain(..) {
                world.despawn(id);
            }
            things = build(world, &noise)?;
        }

        // Hard knocks: a line in the log now and then (hook sounds up here).
        let hits = world.take_impacts();
        report -= frame.dt;
        if report <= 0.0 {
            if let Some(hardest) = hits.first() {
                let (bodies, awake) = world.body_count();
                eprintln!(
                    "physics: {bodies} bodies, {awake} moving; hardest hit {:.1} m/s",
                    hardest.speed
                );
                report = 2.0;
            }
        }
        Ok(())
    })
}
