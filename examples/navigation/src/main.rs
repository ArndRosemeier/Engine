//! Navigation: villagers find their own way around houses, walls, a
//! fountain and crates, and up the stairs to a terrace.
//!
//! - Left click: send everyone to that spot.
//! - N: show or hide the walkable area (green; upper floors blue).
//! - Right mouse + move: turn the camera.
//!
//! When left alone, each villager picks a new place to go when it gets
//! there. `ENGINE_NAV_VIEW=overview|terrace` picks a camera;
//! `ENGINE_NAV_OVERLAY=1` starts with the walkable area shown.
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Small deterministic random numbers, so every run looks the same.
struct Dice(u64);

impl Dice {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn solid(world: &mut World, mesh: Mesh) -> EngineResult<EntityId> {
    let id = world.spawn(mesh);
    world.add_body(id, Body::fixed())?;
    Ok(id)
}

/// The village: everything here is a fixed body, which is what
/// navigation walks on and around.
fn build_village(world: &mut World) -> EngineResult<()> {
    let plaster = [rgb(222, 208, 182), rgb(205, 190, 165), rgb(230, 220, 200)];
    let roof = rgb(150, 72, 52);
    // Houses around the square.
    let houses = [
        (Vec3::new(-11.0, 0.0, 6.0), Vec3::new(6.0, 3.2, 5.0)),
        (Vec3::new(-11.5, 0.0, -5.0), Vec3::new(5.0, 3.6, 6.0)),
        (Vec3::new(1.0, 0.0, 12.0), Vec3::new(7.0, 3.0, 4.5)),
        (Vec3::new(12.0, 0.0, 8.0), Vec3::new(5.0, 3.4, 5.0)),
    ];
    for (k, (at, size)) in houses.into_iter().enumerate() {
        let mut house = Mesh::box_at(at + Vec3::Y * size.y * 0.5, size, plaster[k % 3])?;
        house.add_box(
            at + Vec3::Y * (size.y + 0.15),
            Vec3::new(size.x + 0.6, 0.3, size.z + 0.6),
            roof,
        )?;
        solid(world, house)?;
    }
    // A garden wall with a gate.
    let stone = rgb(150, 144, 134);
    solid(
        world,
        Mesh::box_at(Vec3::new(-3.0, 0.6, -9.0), Vec3::new(10.0, 1.2, 0.4), stone)?,
    )?;
    solid(
        world,
        Mesh::box_at(Vec3::new(2.0, 0.6, -6.6), Vec3::new(0.4, 1.2, 5.2), stone)?,
    )?;
    solid(
        world,
        Mesh::box_at(Vec3::new(2.0, 0.6, -1.6), Vec3::new(0.4, 1.2, 1.6), stone)?,
    )?;
    // A terrace with stairs.
    let top = 1.2;
    solid(
        world,
        Mesh::box_at(
            Vec3::new(10.0, top * 0.5, -7.0),
            Vec3::new(7.0, top, 6.0),
            rgb(170, 160, 146),
        )?,
    )?;
    let steps = 6;
    for i in 0..steps {
        let h = top * (i + 1) as f32 / steps as f32;
        let x = 6.5 - 0.35 * (steps - i) as f32 + 0.175;
        solid(
            world,
            Mesh::box_at(
                Vec3::new(x, h * 0.5, -5.5),
                Vec3::new(0.35, h, 2.0),
                if i % 2 == 0 {
                    stone
                } else {
                    rgb(140, 134, 124)
                },
            )?,
        )?;
    }
    // A fountain in the middle of the square.
    let mut fountain = Mesh::cylinder(Vec3::new(0.0, 0.3, 1.0), Vec3::Y, 1.6, 0.6, 32, stone)?;
    fountain.append_translated(
        &Mesh::cylinder(Vec3::ZERO, Vec3::Y, 0.25, 1.6, 12, rgb(170, 165, 155))?,
        Vec3::new(0.0, 0.8, 1.0),
    )?;
    solid(world, fountain)?;
    // Crates.
    let wood = rgb(140, 100, 62);
    for (x, z, s) in [
        (5.0, 3.0, 0.9),
        (5.8, 3.4, 0.7),
        (-5.0, -3.0, 0.8),
        (-4.0, 6.5, 1.0),
        (7.5, -2.0, 0.8),
    ] {
        solid(
            world,
            Mesh::box_at(Vec3::new(x, s * 0.5, z), Vec3::splat(s), wood)?,
        )?;
    }
    Ok(())
}

/// Places villagers like to go.
const SPOTS: [(f32, f32, f32); 10] = [
    (-6.0, 0.0, 1.0),
    (6.0, 0.0, 1.0),
    (0.0, 0.0, 7.5),
    (-3.0, 0.0, -12.0),
    (-6.0, 0.0, -6.0),
    (10.0, 1.2, -8.0),
    (12.0, 1.2, -6.0),
    (8.0, 0.0, 13.0),
    (-12.0, 0.0, 12.0),
    (14.0, 0.0, 1.0),
];

fn main() -> EngineResult<()> {
    let model = Arc::new(AnimatedModel::load(CHARACTER)?);
    let view = setting("ENGINE_NAV_VIEW").unwrap_or_default();
    let mut overlay_on = setting("ENGINE_NAV_OVERLAY").as_deref() == Some("1");
    let mut overlay: Option<EntityId> = None;
    let mut villagers: Vec<EntityId> = Vec::new();
    let mut dice = Dice(7);
    let mut cam_yaw = if view == "terrace" { 150.0 } else { 110.0 };

    Engine::run("navigation", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_quality(Quality::High);
            world.set_view_distance(600.0)?;
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.0)));
            world.set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(0.0),
                ..Default::default()
            })?;
            let mut ground = Mesh::box_at(
                Vec3::new(0.0, -0.5, 0.0),
                Vec3::new(80.0, 1.0, 80.0),
                rgb(98, 120, 72),
            )?;
            ground.add_box(
                Vec3::new(0.0, 0.005, 1.0),
                Vec3::new(20.0, 0.01, 18.0),
                rgb(168, 150, 120),
            )?;
            world.spawn(ground);
            build_village(world)?;

            let stats = world.build_nav(
                Vec3::new(-20.0, -1.0, -20.0),
                Vec3::new(20.0, 6.0, 20.0),
                NavSettings::default(),
            )?;
            eprintln!(
                "navigation: {} walkable cells of {} surfaces, built in {:.0} ms",
                stats.walkable, stats.surfaces, stats.build_ms
            );

            let profile = AnimationProfile::new()
                .idle("idle")
                .walk("walk")
                .run("jog")
                .walk_speed(1.4)
                .run_speed(3.2);
            for k in 0..8 {
                let a = k as f32 / 8.0 * std::f32::consts::TAU;
                let start = Vec3::new(4.5 * a.cos(), 0.0, 1.0 + 4.5 * a.sin());
                let id = world.spawn_animated_shared(
                    model.clone(),
                    Place::new(start.x, start.y, start.z).with_yaw_deg(k as f32 * 45.0),
                )?;
                world.configure_animation(id, profile.clone())?;
                world.set_foot_ik(id, Some(FootIk::default()))?;
                // Two in a hurry, the rest strolling.
                let speed = if k % 4 == 0 {
                    3.0
                } else {
                    1.2 + 0.1 * (k % 3) as f32
                };
                world.add_nav_agent(id, NavAgent::new(speed))?;
                villagers.push(id);
            }
        }

        let input = &frame.input;
        // Click: everyone goes there.
        if input.mouse_clicked(MouseButton::Left) {
            let (eye, aim) = world.cursor_ray(frame);
            if let Some(hit) = world.cast_ray(eye, aim, 300.0) {
                for (k, &id) in villagers.iter().enumerate() {
                    // Spread them out a little around the spot.
                    let a = k as f32 * 2.4;
                    let spot =
                        hit.point + Vec3::new(a.cos(), 0.0, a.sin()) * (0.3 + 0.25 * k as f32);
                    world.set_nav_goal(id, Some(spot))?;
                }
            }
        }
        // Left alone, pick somewhere new.
        for &id in &villagers {
            match world.nav_status(id) {
                Some(NavStatus::Moving) => {}
                Some(NavStatus::Arrived) if dice.next() > 0.004 => {} // linger a bit
                _ => {
                    let (x, y, z) =
                        SPOTS[(dice.next() * SPOTS.len() as f32) as usize % SPOTS.len()];
                    let jitter = Vec3::new(dice.next() - 0.5, 0.0, dice.next() - 0.5) * 2.0;
                    world.set_nav_goal(id, Some(Vec3::new(x, y, z) + jitter))?;
                }
            }
        }
        // N: the walkable area.
        if input.pressed(Key::N) {
            overlay_on = !overlay_on;
        }
        match (overlay_on, overlay) {
            (true, None) => {
                if let Some(mesh) = world.nav_debug_mesh() {
                    let id = world.spawn(mesh);
                    world.set_casts_shadow(id, false)?;
                    overlay = Some(id);
                }
            }
            (false, Some(id)) => {
                world.despawn(id);
                overlay = None;
            }
            _ => {}
        }

        if input.mouse_down(MouseButton::Right) {
            cam_yaw += input.look_delta(0.15).0;
        }
        if view == "terrace" {
            world.look_orbit(Vec3::new(8.0, 1.0, -6.0), 11.0, cam_yaw, 28.0);
        } else {
            world.look_orbit(Vec3::new(0.0, 0.0, 1.0), 30.0, cam_yaw, 42.0);
        }
        Ok(())
    })
}
