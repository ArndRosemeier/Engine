//! Destruction demo: smooth voxel terrain you can dig into, and ordinary
//! meshes (stone towers) that break into solid pieces.
//!
//! WASD / arrows fly, Space / Shift up and down, Ctrl faster, hold the right
//! mouse button to look around. Left click: shoot where the crosshair points
//! (towers shatter, the ground gets a crater). F adds rock back.
use engine::advanced::Volume;
use engine::prelude::*;
use glam::Vec3;

const SIZE: f32 = 384.0;

fn density(noise: &Noise, p: Vec3) -> f32 {
    // Rolling hills plus ridges, as a signed distance to the surface (in voxels).
    let hills = noise.fbm2(p.x * 0.006, p.z * 0.006, 5, 2.0, 0.5);
    let ridge = 1.0
        - noise
            .fbm2(p.x * 0.013 + 31.0, p.z * 0.013, 3, 2.1, 0.5)
            .abs();
    let height = 34.0 + hills * 26.0 + ridge * ridge * 14.0;
    let ground = height - p.y;
    // Worm caves: hollow where two 3D noises both cross zero.
    let cave = noise.sample3(p * 0.035).abs() + noise.sample3(p * 0.035 + Vec3::splat(17.0)).abs();
    // Caves fade out below the surface so they open up only where dug into.
    let fade = ((p.y - (height - 14.0)) / 8.0).clamp(0.0, 1.0);
    let cave = (cave - 0.11) * 14.0 + fade * 8.0;
    ground.min(cave).clamp(-6.0, 6.0)
}

/// A square stone tower with a crenellated top, standing on `base`.
fn tower(base: Vec3, width: f32, height: f32, color: Color) -> Mesh {
    let mut mesh = Mesh::new();
    let h = height;
    mesh.add_box(
        base + Vec3::new(0.0, h * 0.5 - 1.0, 0.0),
        (width, h, width),
        color,
    )
    .expect("tower body");
    let step = width / 3.0;
    for i in 0..3 {
        for j in 0..3 {
            if (i + j) % 2 == 0 && !(i == 1 && j == 1) {
                let at = base
                    + Vec3::new(
                        (i as f32 - 1.0) * step,
                        h - 1.0 + step * 0.4,
                        (j as f32 - 1.0) * step,
                    );
                mesh.add_box(at, (step, step * 0.8, step), color)
                    .expect("merlon");
            }
        }
    }
    mesh
}

struct Prop {
    id: EntityId,
    centre: Vec3,
    half: Vec3,
}

/// Ray against a prop's box: distance along the ray, if it hits.
fn ray_box(origin: Vec3, dir: Vec3, prop: &Prop) -> Option<f32> {
    let inv = dir.recip();
    let t0 = (prop.centre - prop.half - origin) * inv;
    let t1 = (prop.centre + prop.half - origin) * inv;
    let near = t0.min(t1).max_element();
    let far = t0.max(t1).min_element();
    (far >= near.max(0.0)).then_some(near.max(0.0))
}

fn main() -> EngineResult<()> {
    let mut eye = Vec3::new(SIZE * 0.5, 62.0, SIZE * 0.5 - 40.0);
    let mut yaw = 20.0_f32;
    let mut pitch = -18.0_f32;
    let screenshot = std::env::var_os("ENGINE_SCREENSHOT").is_some();
    let dig_every: u32 = std::env::var("ENGINE_VOXEL_DIG_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut frame_no = 0u32;
    let mut props: Vec<Prop> = Vec::new();
    let mut shots = 0u32;

    Engine::run("voxel_destruction", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            let noise = Noise::new(11);
            let started = std::time::Instant::now();
            let mut terrain = VoxelTerrain::new(Volume::new(1.0))
                .with_material(SurfaceMaterial::STONE)
                .with_color(rgb(150, 138, 120))
                .with_max_lod(
                    std::env::var("ENGINE_VOXEL_MAX_LOD")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(engine::VOXEL_MAX_LOD),
                );
            terrain.generate(
                Vec3::new(0.0, -32.0, 0.0),
                Vec3::new(SIZE, 96.0, SIZE),
                |p| density(&noise, p),
            );
            eprintln!(
                "voxel terrain: {} chunks generated in {:.0} ms",
                terrain.volume().chunk_keys().count(),
                started.elapsed().as_secs_f32() * 1000.0
            );
            // Towers on the slope in front of the camera.
            let spots = [(-14.0, 30.0), (6.0, 38.0), (24.0, 26.0), (-30.0, 52.0)];
            for (i, (dx, dz)) in spots.into_iter().enumerate() {
                let top = Vec3::new(eye.x + dx, 120.0, eye.z + dz);
                let Some(ground) = terrain.raycast(top, -Vec3::Y, 200.0) else {
                    continue;
                };
                let width = 4.0 + (i % 2) as f32;
                let height = 9.0 + (i * 3 % 5) as f32;
                let color = [rgb(176, 166, 150), rgb(150, 120, 96)][i % 2];
                let id = world.spawn(tower(ground, width, height, color));
                props.push(Prop {
                    id,
                    centre: ground + Vec3::new(0.0, height * 0.5, 0.0),
                    half: Vec3::new(width * 0.5, height * 0.5 + 1.0, width * 0.5),
                });
            }
            world.set_voxel_terrain(Some(terrain));
            world.set_clear_color(rgb(140, 191, 242));
            world.set_sun((0.45, 0.9, 0.3), 0.24);
            world.set_view_distance(900.0).expect("view distance");
        }
        frame_no += 1;

        let input = &frame.input;
        if input.mouse_down(MouseButton::Right) {
            let d = input.mouse_delta();
            yaw += d.x * 0.15;
            pitch = (pitch - d.y * 0.15).clamp(-85.0, 85.0);
        }
        let speed = if input.down(Key::Ctrl) { 60.0 } else { 18.0 };
        let forward = Camera::direction(yaw, pitch);
        let right = Camera::right_xz(yaw);
        let mv = input.move_xz();
        eye += (forward * mv.y + right * mv.x) * speed * frame.dt;
        eye.y += input.axis(Key::Shift, Key::Space) * speed * frame.dt;
        world.look_first_person(eye, yaw, pitch);

        // Scripted shots for screenshots and timing runs.
        let mut aim = forward;
        let scripted = if screenshot && (frame_no == 4 || frame_no == 6) {
            // First the tower straight ahead, then the ground left of it.
            aim = if frame_no == 4 {
                props
                    .get(1)
                    .map_or(forward, |p| (p.centre - eye).normalize())
            } else {
                (forward + right * -0.35 + Vec3::Y * -0.05).normalize()
            };
            true
        } else {
            dig_every > 0 && frame_no % dig_every == 0
        };
        let shoot = input.mouse_clicked(MouseButton::Left) || scripted;
        let build = input.pressed(Key::F);
        if shoot || build {
            shots += 1;
            let ground_hit = world
                .voxel_terrain()
                .and_then(|t| t.raycast(eye, aim, 400.0))
                .map(|hit| (hit.distance(eye), hit));
            let prop_hit = props
                .iter()
                .enumerate()
                .filter_map(|(i, p)| ray_box(eye, aim, p).map(|t| (t, i)))
                .min_by(|a, b| a.0.total_cmp(&b.0));
            match (prop_hit, ground_hit) {
                (Some((t, i)), g) if shoot && g.is_none_or(|(gt, _)| t < gt) => {
                    let prop = props.remove(i);
                    let impact = eye + aim * t;
                    world.shatter(
                        prop.id,
                        impact,
                        Shatter {
                            pieces: 24,
                            speed: 9.0,
                            seed: shots,
                            ..Shatter::default()
                        },
                    )?;
                }
                (_, Some((_, hit))) => {
                    let terrain = world.voxel_terrain_mut().expect("terrain");
                    if build {
                        terrain.add_sphere(hit, 3.5);
                    } else {
                        let radius = if screenshot { 7.0 } else { 4.0 };
                        terrain.dig_sphere(hit, radius);
                        // Rock and earth thrown out of the crater.
                        let clod = world.spawn(Mesh::box_at(
                            hit,
                            Vec3::splat(radius * 0.7),
                            rgb(112, 92, 70),
                        )?);
                        world.shatter(
                            clod,
                            hit - aim * radius * 0.5,
                            Shatter {
                                pieces: 16,
                                speed: 11.0,
                                keep_seconds: Some(12.0),
                                seed: shots,
                                ..Shatter::default()
                            },
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    })
}
