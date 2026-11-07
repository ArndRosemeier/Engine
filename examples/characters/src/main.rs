//! Character animation: smooth blends between idle, walk and run, playback
//! matched to the real speed, and feet that find the ground (foot IK) on
//! stone steps.
//!
//! The character walks over the steps, runs back, rests, and starts again.
//! - I: foot IK on or off, to compare (without it, feet float and sink).
//! - B: blending on or off (without it, clips jump).
//! - Right mouse + move: turn the camera.
//!
//! `ENGINE_CHARACTERS_VIEW=steps|side` picks a camera; `ENGINE_CHARACTERS_IK=0`
//! starts with foot IK off; `ENGINE_CHARACTERS_AT=4.5` starts that many
//! seconds into the round.
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);

const STEP_H: f32 = 0.17;
const STEP_D: f32 = 0.42;
const STEPS: usize = 6;
const WALK: f32 = 1.4;
const RUN: f32 = 3.2;

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Stone steps up, a landing, steps down: fixed bodies, so feet (and
/// anything else) stand on them.
fn build_steps(world: &mut World) -> EngineResult<()> {
    let stone = rgb(150, 144, 134);
    let width = 2.4;
    let landing = 2.0;
    let add = |world: &mut World, center: Vec3, size: Vec3, color: Color| -> EngineResult<()> {
        let id = world.spawn(Mesh::box_at(center, size, color)?);
        world.add_body(id, Body::fixed())?;
        Ok(())
    };
    for i in 0..STEPS {
        let h = STEP_H * (i + 1) as f32;
        let shade = if i % 2 == 0 {
            stone
        } else {
            rgb(140, 134, 124)
        };
        // Up...
        let x = STEP_D * (i as f32 + 0.5);
        add(
            world,
            Vec3::new(x, h * 0.5, 0.0),
            Vec3::new(STEP_D, h, width),
            shade,
        )?;
        // ... and down again on the far side of the landing.
        let x = STEP_D * STEPS as f32 * 2.0 + landing - STEP_D * (i as f32 + 0.5);
        add(
            world,
            Vec3::new(x, h * 0.5, 0.0),
            Vec3::new(STEP_D, h, width),
            shade,
        )?;
    }
    let top = STEP_H * STEPS as f32;
    add(
        world,
        Vec3::new(STEP_D * STEPS as f32 + landing * 0.5, top * 0.5, 0.0),
        Vec3::new(landing, top, width),
        stone,
    )?;
    // A few flat stones to step over on the way back.
    for (k, x) in [-3.0_f32, -1.6, 9.8, 11.2].into_iter().enumerate() {
        let h = 0.08 + 0.05 * (k % 2) as f32;
        add(
            world,
            Vec3::new(x, h * 0.5, 0.35),
            Vec3::new(0.5, h, 0.6),
            rgb(120, 116, 110),
        )?;
    }
    Ok(())
}

/// Where the character is on its round, and how fast it should be going.
fn route(t: f32) -> (f32, f32, f32) {
    // Walk from x = -5 to 13 (18 m), run back, rest 2.5 s.
    let (walk_s, run_s, rest_s) = (18.0 / WALK, 18.0 / RUN, 2.5);
    let t = t.rem_euclid(walk_s + run_s + rest_s);
    if t < walk_s {
        (-5.0 + WALK * t, WALK, 90.0)
    } else if t < walk_s + run_s {
        (13.0 - RUN * (t - walk_s), RUN, -90.0)
    } else {
        (-5.0, 0.0, 90.0)
    }
}

fn main() -> EngineResult<()> {
    let model = Arc::new(AnimatedModel::load(CHARACTER)?);
    let side = setting("ENGINE_CHARACTERS_VIEW").as_deref() == Some("side");
    let mut ik_on = setting("ENGINE_CHARACTERS_IK").as_deref() != Some("0");
    let mut blend_on = true;
    let mut hero = None;
    let mut time = setting("ENGINE_CHARACTERS_AT")
        .and_then(|t| t.parse::<f32>().ok())
        .unwrap_or(0.0);
    let mut cam_yaw = if side { 90.0 } else { 55.0 };

    Engine::run("characters", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_quality(Quality::High);
            world.set_view_distance(600.0)?;
            world.set_physical_sky(Some(PhysicalSky::at_hour(10.5)));
            world.set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(0.0),
                ..Default::default()
            })?;
            let mut ground = Mesh::box_at(
                Vec3::new(4.0, -0.5, 0.0),
                Vec3::new(60.0, 1.0, 40.0),
                rgb(98, 120, 72),
            )?;
            ground.add_box(
                // Sunk half into the ground so its underside never meets
                // the undersides of things standing on it.
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(24.0, 0.02, 1.2),
                rgb(150, 128, 98),
            )?;
            world.spawn(ground);
            build_steps(world)?;

            let id = world.spawn_animated_shared(model.clone(), Place::default())?;
            world.configure_animation(
                id,
                AnimationProfile::new()
                    .idle("idle")
                    .walk("walk")
                    .run("jog")
                    .walk_speed(WALK)
                    .run_speed(RUN),
            )?;
            if ik_on {
                world.set_foot_ik(id, Some(FootIk::default()))?;
            }
            hero = Some(id);
        }
        time += frame.dt;
        let Some(id) = hero else {
            return Ok(());
        };

        // Follow the route; stand on whatever is under the character.
        let (x, speed, yaw) = route(time);
        let ground_y = world
            .cast_ray(Vec3::new(x, 3.0, 0.0), Vec3::NEG_Y, 6.0)
            .map_or(0.0, |h| h.point.y);
        world.set_place(id, Place::new(x, ground_y, 0.0).with_yaw_deg(yaw))?;
        let state = if speed > 0.0 {
            Locomotion::Moving { speed_mps: speed }
        } else {
            Locomotion::Idle
        };
        world.set_locomotion(id, state)?;

        let input = &frame.input;
        if input.pressed(Key::I) {
            ik_on = !ik_on;
            world.set_foot_ik(id, ik_on.then(FootIk::default))?;
            eprintln!("characters: foot IK {}", if ik_on { "on" } else { "off" });
        }
        if input.pressed(Key::B) {
            blend_on = !blend_on;
            world.set_animation_blend(id, if blend_on { 0.2 } else { 0.0 })?;
            eprintln!(
                "characters: blending {}",
                if blend_on { "on" } else { "off" }
            );
        }
        if input.mouse_down(MouseButton::Right) {
            cam_yaw += input.look_delta(0.15).0;
        }
        let at = Vec3::new(x, ground_y + 0.9, 0.0);
        if side {
            world.look_orbit(at, 4.5, cam_yaw, -4.0);
        } else {
            world.look_orbit(at, 5.5, cam_yaw, -14.0);
        }
        Ok(())
    })
}
