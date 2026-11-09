//! Occlusion culling in a dense city: 2 000 towers, each its own mesh, and
//! 4 000 instanced street lamps on a grid of streets. From the street almost all of them hide behind the first
//! walls; the title bar and the console show how many the GPU skipped.
//!
//! - `OCCLUSION_CITY_VIEW=street` (default), `roof` or `air` picks the camera.
//! - `OCCLUSION_CITY_AO=0` turns ambient occlusion off (occlusion culling then
//!   draws only its own short depth prepass).
//! - `ENGINE_OCCLUSION=0` turns occlusion culling off to compare frame times;
//!   screenshots with it on and off must match pixel for pixel.
//! - `ENGINE_SCREENSHOT=out.png` holds the camera still for a picture.
//! - `OCCLUSION_CITY_TEXTURES=16` gives the towers that many different
//!   facade textures (default 0: plain colours). With texture arrays the
//!   towers still go out in a handful of multi-draws;
//!   `ENGINE_TEXTURE_ARRAYS=0` compares one group per texture.
use engine::prelude::*;
use glam::Vec3;

const BLOCKS: i32 = 24;
const BLOCK_M: f32 = 60.0;
const STREET_M: f32 = 20.0;
const PITCH_M: f32 = BLOCK_M + STREET_M;

/// Small deterministic hash so every run builds the same city.
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn unit(seed: u32) -> f32 {
    (hash(seed) & 0xffff) as f32 / 65_535.0
}

/// A facade texture: a grid of windows, lit or dark, over a tinted wall.
fn facade(variant: u32) -> (u32, u32, Vec<u8>) {
    const SIZE: u32 = 64;
    let tint = [
        200 + (hash(variant * 3 + 1) % 56) as u8,
        190 + (hash(variant * 3 + 2) % 56) as u8,
        170 + (hash(variant * 3 + 3) % 70) as u8,
    ];
    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (cx, cy) = (x / 8, y / 8);
            let window = (2..7).contains(&(x % 8)) && (2..6).contains(&(y % 8));
            let lit = hash(variant * 7919 + cx * 31 + cy * 977) % 3 == 0;
            let c = match (window, lit) {
                (true, true) => [255, 230, 160],
                (true, false) => [60, 75, 95],
                _ => tint,
            };
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    (SIZE, SIZE, rgba)
}

fn tower(x: f32, z: f32, w: f32, d: f32, h: f32, seed: u32) -> EngineResult<Mesh> {
    let shade = 150 + (hash(seed) % 70) as u8;
    let wall = rgb(shade, shade - 12, shade - 30);
    let mut mesh = Mesh::new();
    mesh.add_box((x, h * 0.5, z), (w, h, d), wall)?;
    // Window bands, so the street view has something to look at.
    let floors = (h / 3.5) as i32;
    for f in 1..floors {
        let y = f as f32 * 3.5;
        mesh.add_box((x, y, z), (w + 0.1, 0.9, d + 0.1), rgb(70, 90, 110))?;
    }
    Ok(mesh)
}

fn build(world: &mut World) -> EngineResult<()> {
    let textures: u32 = std::env::var("OCCLUSION_CITY_TEXTURES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let half = BLOCKS as f32 * PITCH_M * 0.5;
    world.spawn(Shape::box_at(
        (0.0, -0.05, 0.0),
        (half * 2.0 + 400.0, 0.1, half * 2.0 + 400.0),
        rgb(70, 72, 76),
    )?);
    for bx in 0..BLOCKS {
        for bz in 0..BLOCKS {
            let cx = -half + (bx as f32 + 0.5) * PITCH_M;
            let cz = -half + (bz as f32 + 0.5) * PITCH_M;
            // Four towers per block, 28 m square each with a 4 m gap.
            for (i, (ox, oz)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                .into_iter()
                .enumerate()
            {
                let seed = ((bx * BLOCKS + bz) * 4 + i as i32) as u32;
                let h = 12.0 + 60.0 * unit(seed).powi(2);
                let mut mesh = tower(cx + ox * 15.0, cz + oz * 15.0, 26.0, 26.0, h, seed)?;
                if textures > 0 {
                    let (w, h, rgba) = facade(hash(seed) % textures);
                    mesh.set_albedo_rgba(w, h, rgba)?;
                }
                world.spawn(mesh);
            }
        }
    }
    // Lamps every 25 m along both kerbs of every street, as one instanced
    // prototype: the occlusion test runs per instance on the GPU.
    let mut lamp = Mesh::new();
    lamp.add_box((0.0, 2.5, 0.0), (0.25, 5.0, 0.25), rgb(40, 44, 50))?;
    lamp.add_box((0.0, 5.1, 0.6), (0.4, 0.25, 1.4), rgb(240, 220, 160))?;
    let prototype = world.spawn_instanced(lamp);
    let mut places = Vec::new();
    for street in 0..=BLOCKS {
        let x = -half + street as f32 * PITCH_M;
        for side in [-1.0f32, 1.0] {
            let mut z = -half;
            while z < half {
                places.push(Place {
                    position: Vec3::new(x + side * (STREET_M * 0.5 - 1.0), 0.0, z),
                    yaw_degrees: if side < 0.0 { 90.0 } else { -90.0 },
                    ..Place::default()
                });
                z += 25.0;
            }
        }
    }
    world.set_instances(prototype, &places)?;
    Ok(())
}

fn main() -> EngineResult<()> {
    let view = std::env::var("OCCLUSION_CITY_VIEW").unwrap_or_else(|_| "street".into());
    let still = std::env::var_os("ENGINE_SCREENSHOT").is_some();
    let mut t = 0.0f32;
    let mut report = 0.0f32;
    let mut frames = 0u32;
    Engine::run("occlusion_city", move |world, frame| {
        if frame.first {
            world.set_vsync(false);
            world.set_view_distance(2_500.0)?;
            world.set_sun((0.5, 1.0, -0.3), 0.25);
            world.set_shadows(Some(ShadowSettings::default()));
            world.set_occlusion_culling(true);
            if std::env::var("OCCLUSION_CITY_AO").is_ok_and(|v| v == "0") {
                world.set_ambient_occlusion(None);
            }
            build(world)?;
        }
        if !still {
            t += frame.dt;
        }
        // The street between the two middle block columns.
        let street_x = 0.0;
        let along = -BLOCKS as f32 * PITCH_M * 0.5 + 30.0 + (t * 12.0) % 600.0;
        let (eye, yaw, pitch) = match view.as_str() {
            "roof" => (Vec3::new(street_x, 40.0, along), 20.0 + t * 5.0, -8.0),
            "air" => (Vec3::new(street_x, 250.0, along), 10.0, -25.0),
            _ => (Vec3::new(street_x, 1.8, along), 25.0 * (t * 0.2).sin(), 0.0),
        };
        world.look_first_person(eye, yaw, pitch);

        frames += 1;
        report += frame.dt;
        if report >= 2.0 {
            let s = world.occlusion_stats();
            println!(
                "occlusion_city view={view} fps={:.0} tested={} hidden={} drawn={}",
                frames as f32 / report,
                s.tested,
                s.hidden,
                s.tested - s.hidden
            );
            report = 0.0;
            frames = 0;
        }
        Ok(())
    })
}
