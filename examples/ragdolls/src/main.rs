//! Ragdolls: click people to knock them flying. They tumble, come to rest,
//! and get up again (from the back or face down), back into their walk.
//!
//! Controls: right mouse + move to look, WASD to fly, Shift fast.
//! - Left click: punch whoever is under the cursor.
//! - F: a blast under the cursor (everyone near goes down).
//! - K: knocked-down people stay down (on/off). G: everyone lying gets up.
//!
//! `ENGINE_RAGDOLLS_DEMO=1` knocks people over by itself.
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);
const PEOPLE: usize = 14;
const WALK: f32 = 1.3;

/// One person strolling about.
struct Walker {
    id: EntityId,
    yaw: f32,
    turn: f32,
    walking: bool,
}

fn spawn_people(world: &mut World, model: &Arc<AnimatedModel>) -> EngineResult<Vec<Walker>> {
    let mut out = Vec::new();
    for i in 0..PEOPLE {
        let a = i as f32 / PEOPLE as f32 * std::f32::consts::TAU;
        let r = 4.0 + (i % 3) as f32 * 2.5;
        let yaw = a.to_degrees() + 90.0;
        let id = world.spawn_animated_shared(
            model.clone(),
            Place::new(a.cos() * r, 0.0, a.sin() * r).with_yaw_deg(yaw),
        )?;
        world.configure_animation(
            id,
            AnimationProfile::new()
                .idle("idle")
                .walk("walk")
                .run("jog")
                .walk_speed(WALK)
                .run_speed(3.2),
        )?;
        out.push(Walker {
            id,
            yaw,
            turn: if i % 2 == 0 { 14.0 } else { -11.0 },
            walking: i % 4 != 3,
        });
    }
    Ok(out)
}

/// Walk on; turn back toward the middle when far out.
fn stroll(world: &mut World, w: &mut Walker, dt: f32) -> EngineResult<()> {
    if world.is_ragdoll(w.id) {
        return Ok(());
    }
    let t = world.animated_entity(w.id)?.transform();
    let (_, rot, pos) = t.to_scale_rotation_translation();
    // Pick the heading up from wherever it stood up.
    let f = rot * Vec3::Z;
    w.yaw = f.x.atan2(f.z).to_degrees();
    if !w.walking {
        world.set_locomotion(w.id, Locomotion::Idle)?;
        return Ok(());
    }
    let out = Vec3::new(pos.x, 0.0, pos.z);
    let mut turn = w.turn;
    if out.length() > 11.0 {
        let home = (-out).normalize();
        let want = home.x.atan2(home.z).to_degrees();
        let diff = (want - w.yaw + 540.0).rem_euclid(360.0) - 180.0;
        turn = diff.signum() * 60.0;
    }
    w.yaw += turn * dt;
    let dir = Vec3::new(w.yaw.to_radians().sin(), 0.0, w.yaw.to_radians().cos());
    let p = pos + dir * WALK * dt;
    world.set_place(w.id, Place::new(p.x, 0.0, p.z).with_yaw_deg(w.yaw))?;
    world.set_locomotion(w.id, Locomotion::Moving { speed_mps: WALK })?;
    Ok(())
}

fn main() -> EngineResult<()> {
    let model = Arc::new(AnimatedModel::load(CHARACTER)?);
    let demo = std::env::var("ENGINE_RAGDOLLS_DEMO").is_ok_and(|v| v != "0");
    let mut camera = FlyCamera::new((0.0, 3.5, -13.0), 0.0, -12.0).with_speed(8.0);
    let mut people: Vec<Walker> = Vec::new();
    let mut stay_down = false;
    let mut frame_no = 0u32;

    Engine::run("ragdolls", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.5)));
            world.set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(0.0),
                ..Default::default()
            })?;
            world.spawn(Mesh::box_at(
                Vec3::new(0.0, -0.5, 0.0),
                Vec3::new(80.0, 1.0, 80.0),
                rgb(96, 118, 80),
            )?);
            // Crates and a low wall to tumble over.
            for i in 0..6 {
                let id = world.spawn(Mesh::box_at(
                    Vec3::ZERO,
                    Vec3::splat(0.8),
                    rgb(150, 110, 70),
                )?);
                let a = i as f32 * 1.1;
                world.set_place(id, Place::new(a.cos() * 7.5, 0.4, a.sin() * 7.5 + 2.0))?;
                world.add_body(id, Body::dynamic().density(0.4))?;
            }
            let wall = world.spawn(Mesh::box_at(
                Vec3::new(0.0, 0.4, 13.0),
                Vec3::new(14.0, 0.8, 0.5),
                rgb(170, 160, 150),
            )?);
            world.add_body(wall, Body::fixed())?;
            people = spawn_people(world, &model)?;
        }
        frame_no += 1;
        camera.update(world, frame);
        for w in people.iter_mut() {
            stroll(world, w, frame.dt)?;
        }

        let (eye, dir) = world.cursor_ray(frame);
        if frame.input.mouse_clicked(MouseButton::Left) {
            if let Some((id, at)) = world.pick_character(eye, dir, 200.0) {
                let push = (dir + Vec3::Y * 0.25).normalize() * 520.0;
                world.knock_down(id, Knock::new(push).at(at).stay_down(stay_down))?;
            }
        }
        if frame.input.pressed(Key::F) {
            if let Some(hit) = world.cast_ray(eye, dir, 200.0) {
                world.knock_down_in(hit.point, 5.0, 9.0);
                world.blast(hit.point, 5.0, 8.0);
            }
        }
        if frame.input.pressed(Key::K) {
            stay_down = !stay_down;
        }
        if frame.input.pressed(Key::G) {
            for w in &people {
                world.get_up(w.id)?;
            }
        }
        if demo && matches!(frame_no, 30 | 50 | 90) {
            // Knock over whoever stands nearest the camera.
            let mut near: Vec<(f32, EntityId, Vec3)> = people
                .iter()
                .filter(|w| !world.is_ragdoll(w.id))
                .filter_map(|w| {
                    let p = world
                        .animated_entity(w.id)
                        .ok()?
                        .transform()
                        .w_axis
                        .truncate();
                    Some((p.distance(eye), w.id, p))
                })
                .collect();
            near.sort_by(|a, b| a.0.total_cmp(&b.0));
            if let Some(&(_, id, p)) = near.first() {
                if frame_no == 90 {
                    world.knock_down_in(p + Vec3::new(0.0, 0.0, -1.0), 4.0, 8.0);
                } else {
                    let at = p + Vec3::Y * 1.35;
                    let push = (at - eye).with_y(0.0).normalize().with_y(0.35) * 480.0;
                    world.knock_down(id, Knock::new(push).at(at))?;
                }
            }
        }
        for ev in world.take_ragdoll_events() {
            if let RagdollEvent::Settled { face_up, .. } = ev {
                eprintln!(
                    "ragdolls: came to rest {}",
                    if face_up { "on the back" } else { "face down" }
                );
            }
        }

        let (live, total) = world.ragdoll_count();
        engine::egui::Window::new("ragdolls")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   ragdolls: {live} tumbling, {total} down",
                    frame.fps
                ));
                ui.label(format!(
                    "click: punch   F: blast   K: stay down ({})   G: get up",
                    if stay_down { "on" } else { "off" }
                ));
            });
        Ok(())
    })
}
