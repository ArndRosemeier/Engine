//! Smooth destructible voxel terrain, meshed on the GPU.
//!
//! WASD / arrows fly, Space / Shift up and down, hold the right mouse button
//! to look around. Left click digs a crater where the crosshair points,
//! middle click (or F) adds rock back.
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
    // Worm caves: hollow where a 3D noise crosses zero.
    let cave = noise.sample3(p * 0.035).abs() + noise.sample3(p * 0.035 + Vec3::splat(17.0)).abs();
    // Caves fade out below the surface so they open up only where dug into.
    let fade = ((p.y - (height - 14.0)) / 8.0).clamp(0.0, 1.0);
    let cave = (cave - 0.11) * 14.0 + fade * 8.0;
    ground.min(cave).clamp(-6.0, 6.0)
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

    Engine::run("voxel_destruction", move |world, frame| {
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

        let dig = input.mouse_clicked(MouseButton::Left)
            || (screenshot && frame_no == 5)
            || (dig_every > 0 && frame_no % dig_every == 0);
        let build = input.mouse_clicked(MouseButton::Middle) || input.pressed(Key::F);
        if dig || build {
            let terrain = world.voxel_terrain_mut().expect("terrain");
            if let Some(hit) = terrain.raycast(eye, forward, 400.0) {
                if build {
                    terrain.add_sphere(hit, 3.5);
                } else if screenshot {
                    // A row of craters so a still frame shows the destruction.
                    for i in 0..4 {
                        let at = hit + Vec3::new(i as f32 * 7.0 - 10.0, 0.0, i as f32 * 2.0);
                        terrain.dig_sphere(at, 5.0 + i as f32);
                    }
                } else {
                    terrain.dig_sphere(hit, 4.0);
                }
            }
        }
        Ok(())
    })
}
