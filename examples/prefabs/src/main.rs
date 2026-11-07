//! Prefabs and parent/child objects: carts built from parts drive around
//! with spinning wheels and a lantern on a post, a windmill turns its sails,
//! and someone walks by with a lantern in their hand.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast.
//! - C: drop a cart from the sky (its body is a physics body; the wheels and
//!   lantern stay on it).
//! - X: remove the last dropped cart, parts, lamp and all.
//!
//! `ENGINE_PREFABS_VIEW=yard|mill|walker` picks a starting view,
//! `ENGINE_SKY_HOUR` the time (default 18.0, early evening),
//! `ENGINE_PREFABS_DROP=1` drops a few carts right away.
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);

/// Middle of the yard.
const HOME: (f32, f32) = (-30.0, -60.0);
const WHEEL_RADIUS: f32 = 0.38;

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn glow(strength: f32) -> SurfaceMaterial {
    SurfaceMaterial {
        emission: strength,
        ..SurfaceMaterial::DEFAULT
    }
}

/// A farm cart: a box on four wheels with a lantern on a post. The body is
/// the part everything hangs on, so a physics body on it carries the rest.
fn cart(name: &str, color: Color, falls: bool) -> EngineResult<Prefab> {
    let mut cart = Prefab::new(name);
    let mut box_mesh = Mesh::box_at(Vec3::ZERO, Vec3::new(1.3, 0.45, 2.2), color)?;
    box_mesh.add_box(
        Vec3::new(0.0, 0.0, 1.6),
        Vec3::new(0.12, 0.12, 1.2),
        rgb(90, 60, 40),
    )?;
    let body = cart.mesh("body", box_mesh, Place::new(0.0, WHEEL_RADIUS + 0.3, 0.0));
    let wheel = Mesh::cylinder(Vec3::ZERO, Vec3::X, WHEEL_RADIUS, 0.12, 20, rgb(70, 50, 36))?;
    for (part, x, z) in [
        ("wheel_fl", -0.75, 0.7),
        ("wheel_fr", 0.75, 0.7),
        ("wheel_bl", -0.75, -0.7),
        ("wheel_br", 0.75, -0.7),
    ] {
        cart.mesh_on(body, part, wheel.clone(), Place::new(x, -0.3, z));
    }
    let post = cart.mesh_on(
        body,
        "post",
        Mesh::box_at(
            Vec3::new(0.0, 0.6, 0.0),
            Vec3::new(0.06, 1.2, 0.06),
            rgb(50, 40, 34),
        )?,
        Place::new(0.5, 0.2, 1.0),
    );
    let mut bulb = Mesh::ball(Vec3::ZERO, 0.09, rgb(255, 214, 150))?;
    bulb.set_surface_material(glow(5.0));
    let bulb = cart.mesh_on(post, "bulb", bulb, Place::new(0.0, 1.25, 0.0));
    cart.no_shadow(bulb);
    cart.lamp_on(
        post,
        Lamp::point((0.0, 1.25, 0.0), rgb(255, 196, 120), 3.0, 10.0)
            .shadows(true)
            .flicker(0.1),
    );
    if falls {
        cart.body(body, Body::dynamic().density(0.6));
    }
    Ok(cart)
}

/// A windmill whose sails hang on a hub that turns.
fn windmill() -> EngineResult<Prefab> {
    let mut mill = Prefab::new("windmill");
    let mut tower = Mesh::box_at(
        Vec3::new(0.0, 4.0, 0.0),
        Vec3::new(3.2, 8.0, 3.2),
        rgb(214, 204, 186),
    )?;
    tower.add_box(
        Vec3::new(0.0, 8.5, 0.0),
        Vec3::new(3.6, 1.2, 3.6),
        rgb(110, 60, 44),
    )?;
    tower.add_box(
        // The door sits just outside the wall, not half inside it.
        Vec3::new(0.0, 1.0, 1.63),
        Vec3::new(1.0, 2.0, 0.05),
        rgb(80, 56, 40),
    )?;
    let tower = mill.mesh("tower", tower, Place::default());
    let hub = mill.pivot_on(tower, "hub", Place::new(0.0, 7.6, 1.9));
    mill.mesh_on(
        hub,
        "axle",
        Mesh::cylinder(Vec3::ZERO, Vec3::Z, 0.25, 0.6, 12, rgb(60, 50, 44))?,
        Place::default(),
    );
    let mut sail = Mesh::box_at(
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(0.12, 6.0, 0.12),
        rgb(90, 70, 50),
    )?;
    sail.add_box(
        Vec3::new(0.55, 3.6, 0.0),
        Vec3::new(1.0, 4.4, 0.04),
        rgb(232, 226, 210),
    )?;
    for k in 0..4 {
        mill.mesh_on(
            hub,
            &format!("sail_{k}"),
            sail.clone(),
            Place::new(0.0, 0.0, 0.3).with_roll_deg(k as f32 * 90.0),
        );
    }
    Ok(mill)
}

struct Driving {
    cart: Spawned,
    center: Vec3,
    radius: f32,
    speed: f32,
    angle: f32,
    rolled: f32,
}

fn main() -> EngineResult<()> {
    let rules = demo_terrain_rules();
    let ground = HeightField::new(rules.clone());
    let gy = |x: f32, z: f32| ground.height_at(x, z);
    let home = Vec3::new(HOME.0, gy(HOME.0, HOME.1), HOME.1);
    let hour = setting("ENGINE_SKY_HOUR")
        .and_then(|h| h.parse::<f32>().ok())
        .unwrap_or(18.0);
    let (eye, yaw, pitch) = match setting("ENGINE_PREFABS_VIEW").as_deref() {
        Some("mill") => (home + Vec3::new(-4.0, 6.0, -18.0), 12.0, 6.0),
        Some("walker") => (home + Vec3::new(4.0, 2.2, -14.0), 0.0, -6.0),
        _ => (home + Vec3::new(-12.0, 6.0, -16.0), 35.0, -16.0),
    };
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(8.0);
    let mut character = AnimatedModel::load(CHARACTER)
        .map_err(|e| eprintln!("prefabs: no character ({e})"))
        .ok();
    let mut driving: Vec<Driving> = Vec::new();
    let mut hub = None;
    let mut walker = None;
    let mut dropped: Vec<EntityId> = Vec::new();
    let mut time = 0.0_f32;

    Engine::run("prefabs", move |world, frame| {
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
                    .with_grass(Some(Grass {
                        height: 0.2,
                        ..Grass::default()
                    })),
            );

            // Build each prefab once, place it as often as you like.
            world.register_prefab(cart("cart", rgb(150, 96, 56), false)?)?;
            world.register_prefab(cart("falling cart", rgb(120, 130, 150), true)?)?;
            for (k, (dx, dz, r, speed)) in [
                (0.0, 6.0, 7.0, 0.18),
                (0.0, 6.0, 11.0, -0.12),
                (14.0, -4.0, 4.5, 0.25),
            ]
            .into_iter()
            .enumerate()
            {
                let center = home + Vec3::new(dx, 0.0, dz);
                driving.push(Driving {
                    cart: world.spawn_named("cart", Place::new(center.x, center.y, center.z))?,
                    center,
                    radius: r,
                    speed,
                    angle: k as f32 * 2.1,
                    rolled: 0.0,
                });
            }

            let at = home + Vec3::new(-6.0, 0.0, 22.0);
            let mill = world.spawn_prefab(
                &windmill()?,
                Place::new(at.x, gy(at.x, at.z) - 0.3, at.z).with_yaw_deg(160.0),
            )?;
            hub = mill.part("hub");

            // Someone walking by, a lantern in the left hand.
            if let Some(model) = character.take() {
                let id = world
                    .spawn_animated_shared(Arc::new(model), Place::new(home.x, home.y, home.z))?;
                world.play_animation(id, "walk")?;
                let mut lantern = Mesh::ball(Vec3::ZERO, 0.07, rgb(255, 210, 140))?;
                lantern.set_surface_material(glow(5.0));
                let lantern = world.spawn(lantern);
                world.set_casts_shadow(lantern, false)?;
                world.attach_to_bone(lantern, id, "hand_l", Place::default())?;
                let light = world.add_lamp(
                    Lamp::point(Vec3::ZERO, rgb(255, 200, 130), 2.5, 9.0).shadows(true),
                )?;
                world.attach_lamp_to_bone(light, id, "hand_l", Vec3::ZERO)?;
                walker = Some(id);
            }
        }
        time += frame.dt;
        camera.update_above(world, frame, |x, z| gy(x, z));
        world.set_proc_focus(camera.position);

        // Drive the carts: move the root, roll the wheels (relative to the body).
        for d in &mut driving {
            d.angle += d.speed * frame.dt;
            let p = d.center + Vec3::new(d.angle.cos(), 0.0, d.angle.sin()) * d.radius;
            let heading = Vec3::new(-d.angle.sin(), 0.0, d.angle.cos()) * d.speed.signum();
            let yaw = heading.x.atan2(heading.z).to_degrees();
            world.set_place(
                d.cart.root,
                Place::new(p.x, gy(p.x, p.z), p.z).with_yaw_deg(yaw),
            )?;
            d.rolled += (d.speed * d.radius * frame.dt).abs() / WHEEL_RADIUS;
            for (name, x, z) in [
                ("wheel_fl", -0.75, 0.7),
                ("wheel_fr", 0.75, 0.7),
                ("wheel_bl", -0.75, -0.7),
                ("wheel_br", 0.75, -0.7),
            ] {
                if let Some(wheel) = d.cart.part(name) {
                    world.set_place(
                        wheel,
                        Place::new(x, -0.3, z).with_pitch_deg(d.rolled.to_degrees()),
                    )?;
                }
            }
        }

        // Turn the sails: the hub's place is relative to the tower.
        if let Some(hub) = hub {
            world.set_place(hub, Place::new(0.0, 7.6, 1.9).with_roll_deg(time * 35.0))?;
        }

        // The walker strolls up and down the yard.
        if let Some(id) = walker {
            let s = (time * 0.09).sin();
            let x = home.x + 4.0 + s * 9.0;
            let z = home.z - 2.0;
            let yaw = if (time * 0.09).cos() >= 0.0 {
                90.0
            } else {
                -90.0
            };
            world.set_place(id, Place::new(x, gy(x, z), z).with_yaw_deg(yaw))?;
        }

        let input = &frame.input;
        let auto_drop = frame.first && setting("ENGINE_PREFABS_DROP").is_some();
        for k in 0..if auto_drop { 4 } else { 0 } {
            let at = home + Vec3::new(-6.0 + k as f32 * 2.5, 0.0, -2.0 + k as f32 * 1.5);
            let cart = world.spawn_named(
                "falling cart",
                Place::new(at.x, gy(at.x, at.z) + 3.0 + k as f32 * 2.0, at.z)
                    .with_yaw_deg(k as f32 * 50.0)
                    .with_pitch_deg(k as f32 * 15.0),
            )?;
            dropped.push(cart.root);
        }
        if input.pressed(Key::C) {
            let ahead = camera.position + Camera::facing_xz(camera.yaw_deg) * 8.0;
            let spin = time * 57.0;
            let cart = world.spawn_named(
                "falling cart",
                Place::new(ahead.x, gy(ahead.x, ahead.z) + 6.0, ahead.z).with_yaw_deg(spin),
            )?;
            dropped.push(cart.root);
        }
        if input.pressed(Key::X) {
            if let Some(root) = dropped.pop() {
                world.despawn(root);
            }
        }
        Ok(())
    })
}
