//! Lamps at night: a hamlet with street lamps, a camp fire, lit windows, a
//! sweeping searchlight and someone walking around with a lantern, all
//! casting shadows.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast.
//! - L: carry your own lantern (it casts shadows too).
//! - T: lamp shadows on or off, to compare.
//! - N and M: an hour earlier or later.
//!
//! `ENGINE_SKY_HOUR=21.5` sets the time (dusk is around 19.5),
//! `ENGINE_LAMPS_VIEW=street|fire|above` picks a starting view.
use engine::prelude::*;
use glam::Vec3;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);

/// Middle of the hamlet.
const HOME: (f32, f32) = (-30.0, -60.0);

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// An emissive material for bulbs, flames and windows.
fn glow(strength: f32) -> SurfaceMaterial {
    SurfaceMaterial {
        emission: strength,
        ..SurfaceMaterial::DEFAULT
    }
}

/// A cottage: walls, a darker roof block and four glowing windows.
fn cottage(world: &mut World, at: Vec3, yaw: f32, size: Vec3) -> EngineResult<()> {
    // Walls reach 2 m below the floor so slopes never show a gap underneath.
    let mut walls = Mesh::box_at(
        Vec3::new(0.0, size.y * 0.5 - 1.0, 0.0),
        size + Vec3::Y * 2.0,
        rgb(196, 182, 160),
    )?;
    walls.add_box(
        Vec3::new(0.0, size.y + 0.6, 0.0),
        Vec3::new(size.x + 0.5, 1.2, size.z + 0.5),
        rgb(92, 52, 40),
    )?;
    let id = world.spawn(walls);
    world.set_place(id, Place::new(at.x, at.y, at.z).yaw_deg(yaw)?)?;
    let mut windows = Mesh::new();
    for side in [-1.0, 1.0] {
        for along in [-0.25, 0.25] {
            windows.add_box(
                Vec3::new(size.x * along, size.y * 0.55, side * (size.z * 0.5 + 0.02)),
                Vec3::new(0.8, 0.9, 0.05),
                rgb(255, 196, 120),
            )?;
        }
    }
    windows.set_surface_material(glow(2.5));
    let w = world.spawn(windows);
    world.set_place(w, Place::new(at.x, at.y, at.z).yaw_deg(yaw)?)?;
    Ok(())
}

/// A street lamp post with a glowing bulb; returns where the bulb is.
fn street_lamp(world: &mut World, foot: Vec3) -> EngineResult<Vec3> {
    let mut post = Mesh::box_at(
        Vec3::new(0.0, 2.2, 0.0),
        Vec3::new(0.16, 4.4, 0.16),
        rgb(40, 42, 46),
    )?;
    post.add_box(
        Vec3::new(0.35, 4.35, 0.0),
        Vec3::new(0.8, 0.1, 0.1),
        rgb(40, 42, 46),
    )?;
    let id = world.spawn(post);
    world.set_place(id, Place::new(foot.x, foot.y, foot.z))?;
    let bulb_at = foot + Vec3::new(0.7, 4.1, 0.0);
    let mut bulb = Mesh::ball(Vec3::ZERO, 0.14, rgb(255, 230, 190))?;
    bulb.set_surface_material(glow(6.0));
    let b = world.spawn(bulb);
    world.set_place(b, Place::new(bulb_at.x, bulb_at.y, bulb_at.z))?;
    world.set_casts_shadow(b, false)?;
    Ok(bulb_at - Vec3::Y * 0.2)
}

fn main() -> EngineResult<()> {
    let rules = demo_terrain_rules();
    let ground = HeightField::new(rules.clone());
    let gy = |x: f32, z: f32| ground.height_at(x, z);
    let home = Vec3::new(HOME.0, gy(HOME.0, HOME.1), HOME.1);
    let mut hour = setting("ENGINE_SKY_HOUR")
        .and_then(|h| h.parse::<f32>().ok())
        .unwrap_or(21.5);
    let (eye, yaw, pitch) = match setting("ENGINE_LAMPS_VIEW").as_deref() {
        Some("fire") => (home + Vec3::new(5.0, 1.8, -4.0), -34.0, -12.0),
        Some("above") => (home + Vec3::new(-22.0, 18.0, -22.0), 45.0, -32.0),
        _ => (home + Vec3::new(-14.0, 3.0, -16.0), 38.0, -8.0),
    };
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(8.0);
    let character = AnimatedModel::load(CHARACTER)
        .map_err(|e| eprintln!("lamps: no character ({e}), the lantern walks alone"))
        .ok();
    let mut character = character;
    let mut walker = None;
    let mut walker_lamp = None;
    let mut searchlight = None;
    let mut own_lantern: Option<LampId> = None;
    let mut shadows_on = true;
    let mut time = 0.0_f32;

    Engine::run("lamps", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        let gy = |x: f32, z: f32| ground.height_at(x, z);
        if frame.first {
            world.set_quality(Quality::High);
            world.set_view_distance(1200.0)?;
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            world.set_proc_terrain(
                ProcTerrain::gpu_clipmap(rules.clone(), ClipmapConfig::default())
                    .with_focus(camera.position)
                    // A trodden village green: short grass, so the fire shows.
                    .with_grass(Some(Grass {
                        height: 0.18,
                        ..Grass::default()
                    })),
            );

            // Three cottages around a little square.
            for (dx, dz, yaw, size) in [
                (9.0, 6.0, 15.0, Vec3::new(6.0, 3.2, 5.0)),
                (-8.0, 10.0, -20.0, Vec3::new(5.0, 3.0, 4.5)),
                (2.0, 16.0, 5.0, Vec3::new(7.0, 3.6, 5.5)),
            ] {
                let (x, z) = (home.x + dx, home.z + dz);
                // Stand on the lowest corner so the downhill side doesn't float.
                let r = size.x.max(size.z) * 0.6;
                let low = [(-r, -r), (r, -r), (-r, r), (r, r), (0.0, 0.0)]
                    .iter()
                    .map(|(a, b)| gy(x + a, z + b))
                    .fold(f32::MAX, f32::min);
                cottage(world, Vec3::new(x, low - 0.1, z), yaw, size)?;
            }

            // A row of street lamps along the path into the square.
            for i in 0..4 {
                let (x, z) = (
                    home.x - 4.0 + i as f32 * 0.6,
                    home.z - 18.0 + i as f32 * 8.0,
                );
                let bulb = street_lamp(world, Vec3::new(x, gy(x, z), z))?;
                world.add_lamp(Lamp::street_lamp(bulb))?;
            }

            // The camp fire: logs, stones, flames and a flickering light.
            let fire = home + Vec3::new(1.0, 0.0, 2.0);
            let fire = Vec3::new(fire.x, gy(fire.x, fire.z), fire.z);
            let mut logs = Mesh::new();
            for k in 0..5 {
                let a = k as f32 * 1.2566;
                logs.add_box(
                    Vec3::new(a.cos() * 0.35, 0.12, a.sin() * 0.35),
                    Vec3::new(0.9, 0.16, 0.16),
                    rgb(70, 44, 28),
                )?;
            }
            for k in 0..10 {
                let a = k as f32 * 0.628;
                logs.add_box(
                    Vec3::new(a.cos() * 0.9, 0.1, a.sin() * 0.9),
                    Vec3::splat(0.25),
                    rgb(110, 106, 100),
                )?;
            }
            let id = world.spawn(logs);
            world.set_place(id, Place::new(fire.x, fire.y, fire.z))?;
            world.particles_mut().start_group([
                ParticleEmitter::flames(fire + Vec3::Y * 0.15, 0.9),
                ParticleEmitter::smoke(fire, 0.9),
                ParticleEmitter::embers(fire, 0.9),
            ])?;
            world.add_lamp(Lamp::fire(fire + Vec3::Y * 0.7))?;

            // Trees and rocks around the square, for shadows to fall from.
            let trees = build_all(&[0usize, 1, 2], |&k| {
                tree_model(
                    [TreeSpecies::Oak, TreeSpecies::Birch, TreeSpecies::Pine][k],
                    k as u32,
                )
            })?;
            let spots = [
                (-12.0, -2.0),
                (14.0, -6.0),
                (-16.0, 18.0),
                (16.0, 20.0),
                (-3.0, 26.0),
                (20.0, 8.0),
            ];
            for (k, tree) in trees.iter().enumerate() {
                let places: Vec<Place> = spots
                    .iter()
                    .skip(k)
                    .step_by(3)
                    .map(|(dx, dz)| {
                        let (x, z) = (home.x + dx, home.z + dz);
                        Place::new(x, gy(x, z) - 0.15, z)
                    })
                    .collect();
                world.spawn_lod_at(tree.lod_levels(), &places)?;
            }

            // A searchlight on a mast at the edge of the hamlet.
            let mast = home + Vec3::new(-14.0, 0.0, 12.0);
            let mast_y = gy(mast.x, mast.z);
            let pole = world.spawn(Mesh::box_at(
                Vec3::new(0.0, 4.0, 0.0),
                Vec3::new(0.3, 8.0, 0.3),
                rgb(60, 60, 64),
            )?);
            world.set_place(pole, Place::new(mast.x, mast_y, mast.z))?;
            let top = Vec3::new(mast.x, mast_y + 8.3, mast.z);
            searchlight = Some((
                world.add_lamp(
                    Lamp::spot(
                        top,
                        Vec3::new(1.0, -0.6, 0.0),
                        28.0,
                        rgb(220, 232, 255),
                        120.0,
                        45.0,
                    )
                    .shadows(true),
                )?,
                top,
            ));

            // Someone walking a round with a lantern.
            if let Some(model) = character.take() {
                let id = world.spawn_animated(model, Place::new(home.x, home.y, home.z))?;
                world.play_animation(id, "walk")?;
                walker = Some(id);
            }
            walker_lamp = Some(
                world.add_lamp(
                    Lamp::point(home, rgb(255, 200, 120), 3.0, 9.0)
                        .shadows(true)
                        .flicker(0.15),
                )?,
            );
        }
        time += frame.dt;
        camera.update_above(world, frame, |x, z| gy(x, z));
        world.set_proc_focus(camera.position);

        // The walker circles the square, lantern in hand.
        let angle = time * 0.25 - 1.0;
        let r = 4.5;
        let (x, z) = (
            home.x + 1.0 + angle.cos() * r,
            home.z + 2.0 + angle.sin() * r,
        );
        let feet = Vec3::new(x, gy(x, z), z);
        let heading = Vec3::new(-angle.sin(), 0.0, angle.cos());
        if let Some(id) = walker {
            let yaw = heading.x.atan2(heading.z).to_degrees();
            world.set_place(id, Place::new(feet.x, feet.y, feet.z).yaw_deg(yaw)?)?;
        }
        if let Some(lamp) = walker_lamp {
            let side = Vec3::new(heading.z, 0.0, -heading.x);
            let sway = (time * 3.6).sin() * 0.05;
            world.set_lamp_position(
                lamp,
                feet + Vec3::Y * 0.95 + side * 0.32 + heading * (0.15 + sway),
            )?;
        }

        // The searchlight sweeps back and forth over the square.
        if let Some((id, top)) = searchlight {
            // Centred on the square, which lies to the south-east of the mast.
            let a = (time * 0.35).sin() * 0.7 - 0.46;
            let dir = Vec3::new(a.cos(), -0.5, a.sin());
            let mut lamp = *world.lamp(id).expect("searchlight");
            lamp.position = top;
            lamp.shape = LampShape::Spot {
                direction: dir,
                angle_deg: 28.0,
                softness: 0.35,
            };
            world.set_lamp(id, lamp)?;
        }

        let input = &frame.input;
        if input.pressed(Key::L) {
            own_lantern = match own_lantern.take() {
                Some(id) => {
                    world.remove_lamp(id);
                    None
                }
                None => Some(world.add_lamp(
                    Lamp::point(camera.position, rgb(255, 205, 140), 2.5, 12.0).shadows(true),
                )?),
            };
        }
        if let Some(id) = own_lantern {
            // Held low and to the right, so shadows fall away from the view.
            let right = Camera::right_xz(camera.yaw_deg);
            world.set_lamp_position(id, camera.position + right * 0.4 - Vec3::Y * 0.5)?;
        }
        if input.pressed(Key::T) {
            shadows_on = !shadows_on;
            let ids: Vec<LampId> = world.lamps().map(|(id, _, _)| id).collect();
            for id in ids {
                let lamp = *world.lamp(id).expect("lamp");
                world.set_lamp(id, lamp.shadows(shadows_on))?;
            }
            eprintln!("lamps: shadows {}", if shadows_on { "on" } else { "off" });
        }
        let earlier = input.pressed(Key::N);
        let later = input.pressed(Key::M);
        if earlier || later {
            hour = (hour + if later { 1.0 } else { -1.0 }).rem_euclid(24.0);
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            eprintln!("lamps: {hour:.1} h");
        }
        Ok(())
    })
}
