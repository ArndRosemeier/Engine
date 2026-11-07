//! GPU particles: fire, smoke, sparks, dust and debris from destruction.
//!
//! A meadow of diggable ground with a camp fire, a forge throwing sparks
//! and stone towers. Explosions go off on their own every few seconds
//! (P stops them); every one digs a crater, throws clods, dust and sparks,
//! and shatters a tower it hits.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast.
//! - Left click: an explosion where the mouse points.
//! - 1: sparks, 2: a dust cloud, 3: a handful of debris, at the mouse.
//! - F: light or put out another fire at the mouse.
//! - P: automatic explosions on or off.
//!
//! `ENGINE_SKY_HOUR=19.5` for dusk (fire and sparks glow), and
//! `ENGINE_PARTICLES_VIEW=fire` starts by the fire.
use engine::advanced::Volume;
use engine::glam::Vec3;
use engine::prelude::*;

const SIZE: f32 = 256.0;

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn density(noise: &Noise, p: Vec3) -> f32 {
    let hills = noise.fbm2(p.x * 0.008, p.z * 0.008, 4, 2.0, 0.5);
    let height = 20.0 + hills * 6.0;
    (height - p.y).clamp(-6.0, 6.0)
}

/// A square stone tower standing on `base`.
fn tower(base: Vec3, width: f32, height: f32, color: Color) -> EngineResult<Mesh> {
    let mut mesh = Mesh::new();
    mesh.add_box(
        base + Vec3::new(0.0, height * 0.5 - 1.0, 0.0),
        (width, height, width),
        color,
    )?;
    let step = width / 3.0;
    for i in [0.0, 2.0] {
        for j in [0.0, 2.0] {
            mesh.add_box(
                base + Vec3::new(
                    (i - 1.0) * step,
                    height - 1.0 + step * 0.4,
                    (j - 1.0) * step,
                ),
                (step, step * 0.8, step),
                color,
            )?;
        }
    }
    Ok(mesh)
}

/// Logs and a ring of stones for a camp fire.
fn fire_pit(color_logs: Color, color_stones: Color) -> EngineResult<Mesh> {
    let mut m = Mesh::new();
    for k in 0..5 {
        let a = k as f32 * 1.2566;
        m.add_box(
            Vec3::new(a.cos() * 0.35, 0.12, a.sin() * 0.35),
            Vec3::new(0.9, 0.16, 0.16),
            color_logs,
        )?;
    }
    for k in 0..10 {
        let a = k as f32 * 0.628;
        m.add_box(
            Vec3::new(a.cos() * 0.9, 0.1, a.sin() * 0.9),
            Vec3::splat(0.25),
            color_stones,
        )?;
    }
    Ok(m)
}

struct Tower {
    id: EntityId,
    centre: Vec3,
    half: Vec3,
}

fn ray_box(origin: Vec3, dir: Vec3, t: &Tower) -> Option<f32> {
    let inv = dir.recip();
    let t0 = (t.centre - t.half - origin) * inv;
    let t1 = (t.centre + t.half - origin) * inv;
    let near = t0.min(t1).max_element();
    let far = t0.max(t1).min_element();
    (far >= near.max(0.0)).then_some(near.max(0.0))
}

fn main() -> EngineResult<()> {
    let hour: f32 = setting("ENGINE_SKY_HOUR")
        .and_then(|h| h.parse().ok())
        .unwrap_or(16.5);
    let screenshot = setting("ENGINE_SCREENSHOT").is_some();
    let centre = Vec3::new(SIZE * 0.5, 0.0, SIZE * 0.5);
    let (eye, yaw, pitch) = match setting("ENGINE_PARTICLES_VIEW").as_deref() {
        // Height is above the ground there.
        Some("fire") => (centre + Vec3::new(-2.0, 2.2, -2.5), 32.0, -18.0),
        _ => (centre + Vec3::new(-2.0, 4.5, -16.0), 4.0, -9.0),
    };
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(12.0);
    let mut towers: Vec<Tower> = Vec::new();
    let mut fires: Vec<Fire> = Vec::new();
    let mut auto = true;
    let mut next_blast = 1.5_f32;
    let mut blasts = 0u32;
    let mut frame_no = 0u32;

    Engine::run("particles", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        frame_no += 1;
        if frame.first {
            let noise = Noise::new(5);
            let mut terrain = VoxelTerrain::new(Volume::new(1.0))
                .with_material(SurfaceMaterial::STONE)
                .with_color(rgb(150, 138, 120));
            terrain.generate(
                Vec3::new(0.0, -16.0, 0.0),
                Vec3::new(SIZE, 64.0, SIZE),
                |p| density(&noise, p),
            );
            let ground = |t: &VoxelTerrain, x: f32, z: f32| {
                t.raycast(Vec3::new(x, 80.0, z), -Vec3::Y, 200.0)
                    .unwrap_or(Vec3::new(x, 20.0, z))
            };
            camera.position.y += ground(&terrain, eye.x, eye.z).y;
            // The camp fire.
            let fire_at = ground(&terrain, centre.x + 2.0, centre.z + 4.0);
            let pit = world.spawn(fire_pit(rgb(70, 44, 28), rgb(110, 106, 100))?);
            world.set_place(pit, Place::new(fire_at.x, fire_at.y, fire_at.z))?;
            fires.push(world.start_fire(fire_at + Vec3::Y * 0.1, 1.0)?);
            // A forge: sparks keep flying off an anvil.
            let anvil_at = ground(&terrain, centre.x - 7.0, centre.z + 6.0);
            world.spawn(Mesh::box_at(
                anvil_at + Vec3::Y * 0.4,
                Vec3::new(0.9, 0.8, 0.5),
                rgb(60, 60, 64),
            )?);
            world.particles_mut().start(
                ParticleEmitter::sparks(anvil_at + Vec3::Y * 0.85, Vec3::new(0.8, 3.5, 0.4), 1)
                    .with_mode(ParticleMode::Continuous)
                    .with_rate(90.0),
            )?;
            world.add_lamp(
                Lamp::point(anvil_at + Vec3::Y * 1.0, rgb(255, 150, 70), 2.5, 6.0).flicker(0.6),
            )?;
            // Chimney-like smoke from far off, to see it drift.
            let smoke_at = ground(&terrain, centre.x + 30.0, centre.z + 50.0);
            world
                .particles_mut()
                .start(ParticleEmitter::smoke(smoke_at, 3.0).with_rate(14.0))?;
            // Towers to blow up.
            let spots = [(-16.0, 18.0), (14.0, 22.0), (-2.0, 34.0), (28.0, 36.0)];
            for (i, (dx, dz)) in spots.into_iter().enumerate() {
                let base = ground(&terrain, centre.x + dx, centre.z + dz);
                let (w, h) = (3.0 + (i % 2) as f32, 7.0 + (i * 3 % 4) as f32);
                let color = [rgb(176, 166, 150), rgb(150, 120, 96)][i % 2];
                let id = world.spawn(tower(base, w, h, color)?);
                towers.push(Tower {
                    id,
                    centre: base + Vec3::Y * h * 0.5,
                    half: Vec3::new(w * 0.5, h * 0.5 + 1.0, w * 0.5),
                });
            }
            world.set_voxel_terrain(Some(terrain));
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            world.set_view_distance(900.0)?;
        }

        camera.update(world, frame);
        let input = &frame.input;
        let (origin, dir) = world.cursor_ray(frame);
        let ground_hit = world
            .voxel_terrain()
            .and_then(|t| t.raycast(origin, dir, 500.0));

        let mut blast_at: Option<(Vec3, Vec3)> = None;
        if input.mouse_clicked(MouseButton::Left) {
            if let Some(hit) = ground_hit {
                blast_at = Some((hit, dir));
            }
        }
        if input.pressed(Key::P) {
            auto = !auto;
        }
        // Screenshots: the first blast goes off on frame 6.
        let due = if screenshot {
            frame_no == 6
        } else {
            frame.time > next_blast
        };
        if auto && due {
            next_blast = frame.time + 3.0;
            blasts += 1;
            // Walk the blasts round the towers and the open ground.
            let a = blasts as f32 * 2.4;
            let target = towers
                .get((blasts as usize) % 3)
                .filter(|_| blasts % 2 == 1)
                .map(|t| t.centre - Vec3::Y * t.half.y * 0.6)
                .unwrap_or(centre + Vec3::new(a.cos() * 14.0, 0.0, 14.0 + a.sin() * 8.0));
            if let Some(hit) = world
                .voxel_terrain()
                .and_then(|t| t.raycast(target.with_y(80.0), -Vec3::Y, 200.0))
            {
                blast_at = Some((hit, (hit - camera.position).normalize_or(Vec3::Z)));
            }
        }
        if let Some((hit, from)) = blast_at {
            let radius = 3.0;
            world.dig(hit, radius);
            world.explosion(hit + Vec3::Y * 0.5, radius)?;
            // Towers close by break apart.
            let mut i = 0;
            while i < towers.len() {
                let t = &towers[i];
                let near =
                    (t.centre.with_y(0.0) - hit.with_y(0.0)).length() < radius + t.half.x + 1.5;
                let shot = ray_box(origin, dir, t).is_some() && blast_at.is_some() && !auto;
                if near || (shot && input.mouse_clicked(MouseButton::Left)) {
                    let t = towers.remove(i);
                    world.shatter(
                        t.id,
                        hit.max(t.centre - t.half) - from * 0.5,
                        Shatter {
                            pieces: 20,
                            speed: 8.0,
                            seed: blasts + i as u32,
                            ..Shatter::default()
                        },
                    )?;
                } else {
                    i += 1;
                }
            }
        }
        if let Some(hit) = ground_hit {
            let up = Vec3::Y * 0.3;
            if input.pressed(Key::Digit1) {
                world.particles_mut().burst(ParticleEmitter::sparks(
                    hit + up,
                    Vec3::Y * 5.0,
                    200,
                ))?;
            }
            if input.pressed(Key::Digit2) {
                world.particles_mut().burst(ParticleEmitter::dust(
                    hit + up,
                    2.5,
                    rgb(140, 120, 95),
                ))?;
            }
            if input.pressed(Key::Digit3) {
                world.particles_mut().burst(ParticleEmitter::debris(
                    hit + up,
                    rgb(120, 112, 100),
                    60,
                    5.0,
                ))?;
            }
            if input.pressed(Key::F) {
                let near = fires.len() > 1;
                if near {
                    let f = fires.pop().expect("a fire");
                    world.stop_fire(f);
                } else {
                    fires.push(world.start_fire(hit, 1.4)?);
                }
            }
        }

        if !screenshot {
            engine::egui::Window::new("particles")
                .anchor(engine::egui::Align2::LEFT_TOP, [12.0, 12.0])
                .resizable(false)
                .show(frame.ui.ctx(), |ui| {
                    ui.label(format!(
                        "{} emitters running, up to {} particles",
                        world.particles_mut().running(),
                        MAX_PARTICLES
                    ));
                    ui.label("left click: explosion   1 sparks   2 dust   3 debris");
                    ui.label(format!(
                        "F: fire on/off   P: automatic explosions ({})",
                        if auto { "on" } else { "off" }
                    ));
                });
        }
        Ok(())
    })
}
