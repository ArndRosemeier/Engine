//! Sound: 3D placed sounds, reverb that follows the space you are in, and
//! music in layers that follow the action. No audio files needed: every sound here
//! is built into the engine or made in code.
//!
//! A meadow with a hill. Walk into the cave in the hill and the sound turns
//! long and ringing; step into the stone hall and it echoes like a hall;
//! the camp fire crackles from its direction; walls muffle what is behind
//! them. Explosions boom, dig and crumble by themselves.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast.
//! - Left click: an explosion where the mouse points (also digs a crater).
//! - 1 / 2 / 3 / 4: music intensity calm / bass / arpeggio / drums.
//! - M: music on or off.   K: a knock at the mouse.   F: fire on or off.
//!
//! `ENGINE_AUDIO_VIEW=cave` starts in the cave, `hall` in the hall.
//! `ENGINE_AUDIO=0` runs silent.
use engine::advanced::Volume;
use engine::glam::Vec3;
use engine::prelude::*;

const SIZE: f32 = 256.0;

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Hills with a big round hill in the middle and a cave dug into it.
fn density(noise: &Noise, p: Vec3, hill: Vec3, hall: Vec3) -> f32 {
    let hills = noise.fbm2(p.x * 0.008, p.z * 0.008, 4, 2.0, 0.5);
    let r = (p.with_y(0.0) - hill.with_y(0.0)).length();
    let mut height = 20.0 + hills * 5.0 + 34.0 * (-(r / 38.0).powi(2)).exp();
    // Level ground for the hall.
    let out = ((p.x - hall.x).abs() - 12.0)
        .max((p.z - hall.z).abs() - 15.0)
        .max(0.0);
    let w = (out / 8.0).min(1.0);
    height = 21.0 + (height - 21.0) * w * w * (3.0 - 2.0 * w);
    let mut d = (height - p.y).clamp(-6.0, 6.0);
    // A tunnel from the south side into a chamber in the hill.
    let mouth = hill + Vec3::new(0.0, 23.0, -40.0);
    let inside = hill + Vec3::new(0.0, 25.0, -4.0);
    let axis = inside - mouth;
    let t = ((p - mouth).dot(axis) / axis.length_squared()).clamp(0.0, 1.0);
    let tunnel = (p - (mouth + axis * t)).length() - 3.5;
    let chamber = (p - (inside + Vec3::Y * 3.0)).length() - 11.0;
    d = d.min(tunnel.min(chamber).clamp(-6.0, 6.0));
    d
}

/// A stone hall: four walls with a doorway and a roof.
fn hall(centre: Vec3, half: Vec3, color: Color) -> EngineResult<Mesh> {
    let mut m = Mesh::new();
    let t = 0.6;
    let (hx, hy, hz) = (half.x, half.y, half.z);
    let c = centre + Vec3::Y * hy;
    m.add_box(c + Vec3::new(0.0, 0.0, hz), (2.0 * hx, 2.0 * hy, t), color)?;
    m.add_box(c + Vec3::new(hx, 0.0, 0.0), (t, 2.0 * hy, 2.0 * hz), color)?;
    m.add_box(c + Vec3::new(-hx, 0.0, 0.0), (t, 2.0 * hy, 2.0 * hz), color)?;
    // Front wall with a doorway 3 m wide.
    let side = hx - 1.5;
    m.add_box(
        c + Vec3::new(-(1.5 + side * 0.5), 0.0, -hz),
        (side, 2.0 * hy, t),
        color,
    )?;
    m.add_box(
        c + Vec3::new(1.5 + side * 0.5, 0.0, -hz),
        (side, 2.0 * hy, t),
        color,
    )?;
    m.add_box(c + Vec3::new(0.0, hy - 1.0, -hz), (3.0, 2.0, t), color)?;
    m.add_box(
        c + Vec3::new(0.0, hy, 0.0),
        (2.0 * hx + t, t, 2.0 * hz + t),
        color,
    )?;
    Ok(m)
}

fn main() -> EngineResult<()> {
    let centre = Vec3::new(SIZE * 0.5, 0.0, SIZE * 0.5);
    let hill = centre + Vec3::new(0.0, 0.0, 30.0);
    let hall_at = centre + Vec3::new(-30.0, 0.0, -10.0);
    let hall_half = Vec3::new(9.0, 6.0, 12.0);
    let (eye, yaw, pitch) = match setting("ENGINE_AUDIO_VIEW").as_deref() {
        Some("cave") => (hill + Vec3::new(0.0, 26.0, -10.0), 0.0, -5.0),
        Some("hall") => (hall_at + Vec3::new(0.0, 2.5, -6.0), 0.0, -5.0),
        _ => (centre + Vec3::new(-6.0, 9.0, -34.0), -10.0, -8.0),
    };
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(10.0);
    let mut fires: Vec<Fire> = Vec::new();
    let mut fire_at = Vec3::ZERO;
    let mut music_on = true;
    let mut last_space = String::new();
    let mut ground_y = 0.0;
    let bell = Sound::knock(3, 1.0);
    let mut frame_no = 0usize;

    Engine::run("audio", move |world, frame| {
        world.set_vsync(false);
        if frame.first {
            let noise = Noise::new(9);
            let mut terrain = VoxelTerrain::new(Volume::new(1.0))
                .with_material(SurfaceMaterial::STONE)
                .with_color(rgb(140, 132, 116));
            terrain.generate(
                Vec3::new(0.0, -16.0, 0.0),
                Vec3::new(SIZE, 80.0, SIZE),
                |p| density(&noise, p, hill, hall_at),
            );
            let ground = |t: &VoxelTerrain, x: f32, z: f32| {
                t.raycast(Vec3::new(x, 120.0, z), -Vec3::Y, 300.0)
                    .unwrap_or(Vec3::new(x, 20.0, z))
            };
            // Views given relative to the ground.
            if setting("ENGINE_AUDIO_VIEW").as_deref() != Some("cave") {
                camera.position.y += ground(&terrain, eye.x, eye.z).y;
            }
            ground_y = ground(&terrain, hall_at.x, hall_at.z).y;
            // The hall, with a room so the engine knows it is indoors.
            let floor = Vec3::new(hall_at.x, ground_y - 0.5, hall_at.z);
            world.spawn(hall(floor, hall_half, rgb(170, 160, 146))?);
            world.add_room(Room::new(floor + Vec3::Y * hall_half.y, hall_half))?;
            world.add_lamp(Lamp::point(
                floor + Vec3::new(0.0, hall_half.y * 1.5, 0.0),
                rgb(255, 210, 160),
                6.0,
                22.0,
            ))?;
            // The camp fire between the hall and the hill crackles by itself.
            fire_at = ground(&terrain, centre.x + 6.0, centre.z - 8.0);
            fires.push(world.start_fire(fire_at + Vec3::Y * 0.1, 1.2)?);
            world.set_voxel_terrain(Some(terrain));
            world.set_physical_sky(Some(PhysicalSky::at_hour(17.0)));
            world.set_view_distance(900.0)?;
            // A breeze everywhere, and music.
            world.play_sound(
                &Sound::stock(Stock::WindLoop, 0),
                PlaySound::ambient().looped().volume(0.25),
            );
            world.play_music(MusicTrack::procedural(7, 92.0), 3.0);
            world.set_music_intensity(0.3);
        }
        // A bell in the hall rings every 3 seconds: muffled outside,
        // ringing inside.
        if (frame.time % 3.0) < frame.dt {
            world.play_sound(
                &bell,
                PlaySound::at(Vec3::new(hall_at.x, ground_y + 2.5, hall_at.z + 8.0))
                    .volume(0.9)
                    .range(2.0, 150.0),
            );
        }

        camera.update(world, frame);
        let input = &frame.input;
        let (origin, dir) = world.cursor_ray(frame);
        let hit = world
            .voxel_terrain()
            .and_then(|t| t.raycast(origin, dir, 500.0));
        if input.mouse_clicked(MouseButton::Left) {
            if let Some(hit) = hit {
                world.dig(hit, 2.5);
                world.explosion(hit + Vec3::Y * 0.4, 2.5)?;
            }
        }
        if input.pressed(Key::K) {
            if let Some(hit) = hit {
                world.play_sound(
                    &Sound::stock(Stock::KnockHard, frame_no),
                    PlaySound::at(hit).volume(2.0),
                );
                frame_no += 1;
            }
        }
        if input.pressed(Key::F) {
            if let Some(f) = fires.pop() {
                world.stop_fire(f);
            } else {
                fires.push(world.start_fire(fire_at + Vec3::Y * 0.1, 1.2)?);
            }
        }
        for (key, level) in [
            (Key::Digit1, 0.0),
            (Key::Digit2, 0.33),
            (Key::Digit3, 0.55),
            (Key::Digit4, 1.0),
        ] {
            if input.pressed(key) {
                world.set_music_intensity(level);
            }
        }
        if input.pressed(Key::M) {
            music_on = !music_on;
            if music_on {
                world.play_music(MusicTrack::procedural(7, 92.0), 2.0);
            } else {
                world.stop_music(2.0);
            }
        }

        // Name the space the ears are in.
        let a = world.audio().acoustics();
        let space = if a.muffle_hz < 2000.0 {
            "under water"
        } else if a.reverb.size > 0.6 && a.reverb.damping < 0.33 {
            "cave"
        } else if a.reverb.size > 0.6 {
            "hall"
        } else if a.reverb.size > 0.35 {
            "room / enclosed"
        } else {
            "open air"
        };
        if space != last_space {
            last_space = space.to_string();
            println!("acoustics: {space}");
        }

        if setting("ENGINE_SCREENSHOT").is_none() {
            engine::egui::Window::new("audio")
                .anchor(engine::egui::Align2::LEFT_TOP, [12.0, 12.0])
                .resizable(false)
                .show(frame.ui.ctx(), |ui| {
                    ui.label(format!(
                        "space: {space}   sounds playing: {}   output: {}",
                        world.audio().voice_count(),
                        if world.audio().has_output() {
                            "on"
                        } else {
                            "none"
                        }
                    ));
                    ui.label(format!(
                        "music intensity {:.2} (1-4)   M music   K knock   F fire",
                        world.audio().music_intensity()
                    ));
                    ui.label(
                        "left click: explosion   fly into the cave (north hill) or the hall (west)",
                    );
                });
        }
        Ok(())
    })
}
