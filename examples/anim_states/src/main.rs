//! An animation state machine: a character that idles, walks, jogs, swings
//! a sword and talks, with the clips chosen by a graph of states and rules,
//! not by game code. Footsteps come from clip events and play a sound.
//!
//! Controls: W/S speed up and slow down, A/D turn, Space attacks, T talks,
//! Esc quits.
//!
//! `ENGINE_ANIM_DEMO=1` (on in verify runs) plays a fixed script instead:
//! stand, walk, jog, stop, attack, talk.
use engine::prelude::*;
use std::collections::BTreeSet;
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);
/// How fast the walk and jog clips move the character (m/s).
const WALK: f32 = 1.4;
const JOG: f32 = 3.2;

/// The state machine, with a footstep at each foot-down of walk and jog.
fn graph(model: &AnimatedModel) -> AnimGraph {
    let half = |clip: &str| model.find_clip(clip).map_or(0.5, |c| c.duration * 0.5);
    AnimGraph::new("idle")
        .state(AnimState::looped("idle", "idle"))
        .state(AnimState::looped("walk", "walk").match_speed("speed", WALK))
        .state(AnimState::looped("jog", "jog").match_speed("speed", JOG))
        .state(AnimState::once("attack", "sword attack").then("idle"))
        .state(AnimState::looped("talk", "talk"))
        // An attack from any state, as soon as it is asked for.
        .transition_any("attack", When::trigger("attack"))
        .blend(0.1)
        .transition("idle", "walk", When::above("speed", 0.2))
        .transition("walk", "jog", When::above("speed", 2.4))
        .transition("jog", "walk", When::below("speed", 2.2))
        .transition("walk", "idle", When::below("speed", 0.15))
        .transition(
            "idle",
            "talk",
            When::all([When::flag("talking", true), When::below("speed", 0.15)]),
        )
        .transition("talk", "idle", When::flag("talking", false))
        .transition("talk", "walk", When::above("speed", 0.2))
        .event("walk", 0.0, "footstep")
        .event("walk", half("walk"), "footstep")
        .event("jog", 0.0, "footstep")
        .event("jog", half("jog"), "footstep")
}

/// The demo script: (from second, speed, talking), attack at 7.6 s.
fn script(t: f32) -> (f32, bool) {
    match t {
        t if t < 1.5 => (0.0, false),
        t if t < 4.0 => (WALK, false),
        t if t < 6.5 => (JOG, false),
        t if t < 9.5 => (0.0, false),
        t if t < 11.5 => (0.0, true),
        _ => (0.0, false),
    }
}

#[derive(Default)]
struct Game {
    hero: Option<EntityId>,
    at: Vec3,
    yaw: f32,
    speed: f32,
    talking: bool,
    time: f32,
    attacked: bool,
    visited: BTreeSet<String>,
    steps: u32,
}

fn main() -> EngineResult<()> {
    let demo = std::env::var("ENGINE_ANIM_DEMO")
        .map(|v| v != "0")
        .unwrap_or_else(|_| std::env::var_os("ENGINE_VERIFY").is_some());
    // The model lives in another example's folder: allow loads from `examples/`.
    let model = Arc::new(AnimatedModel::load_with(
        CHARACTER,
        concat!(env!("CARGO_MANIFEST_DIR"), "/.."),
        &EngineLimits::default(),
    )?);
    let step_sound = Sound::stock(Stock::KnockSoft, 0);
    let mut game = Game::default();

    Engine::run("anim states", move |world, frame| {
        if frame.first {
            world.set_vsync(false);
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.0)));
            world.spawn(Mesh::box_at(
                (0.0, -0.5, 0.0),
                (80.0, 1.0, 80.0),
                rgb(98, 120, 72),
            )?);
            let hero = world.spawn_animated_shared(model.clone(), Place::default())?;
            world.set_anim_graph(hero, graph(&model))?;
            // Feet on the ground, also through the jog's flight phase.
            world.set_foot_ik(hero, Some(FootIk::default()))?;
            game.hero = Some(hero);
        }
        let Some(hero) = game.hero else {
            return Ok(());
        };
        game.time += frame.dt;

        // What the player (or the script) wants: a speed, talking, an attack.
        let input = &frame.input;
        if demo {
            (game.speed, game.talking) = script(game.time);
            if game.time >= 7.6 && !game.attacked {
                world.anim_trigger(hero, "attack")?;
                game.attacked = true;
            }
        } else {
            game.speed +=
                (input.down(Key::W) as i32 - input.down(Key::S) as i32) as f32 * 3.0 * frame.dt;
            game.speed = game.speed.clamp(0.0, JOG);
            if input.pressed(Key::T) {
                game.talking = !game.talking;
            }
            if input.pressed(Key::Space) {
                world.anim_trigger(hero, "attack")?;
            }
        }
        game.yaw +=
            (input.down(Key::D) as i32 - input.down(Key::A) as i32) as f32 * 90.0 * frame.dt;

        // Tell the graph; it picks the clip. Stand still while swinging.
        world.set_anim_value(hero, "speed", game.speed)?;
        world.set_anim_flag(hero, "talking", game.talking)?;
        let state = world.anim_state(hero).unwrap_or("").to_owned();
        let moving = if state == "attack" { 0.0 } else { game.speed };
        let facing = Vec3::new(
            game.yaw.to_radians().sin(),
            0.0,
            game.yaw.to_radians().cos(),
        );
        game.at += facing * moving * frame.dt;
        world.set_place(
            hero,
            Place::new(game.at.x, 0.0, game.at.z).with_yaw_deg(game.yaw),
        )?;

        // Clip events: a footstep sound where the character is.
        for e in world.take_anim_events() {
            if e.name == "footstep" {
                game.steps += 1;
                world.play_sound(&step_sound, PlaySound::at(game.at).volume(0.25));
            }
        }

        // Seen from the front, a little to its left (orbit yaw counts from +X).
        world.look_orbit(game.at + Vec3::Y * 1.0, 5.0, 60.0 - game.yaw, 15.0);
        frame.ui.text(Anchor::TopLeft, format!("State: {state}"));
        frame.ui.bar(
            Anchor::TopLeft,
            "Speed m/s",
            game.speed,
            JOG,
            rgb(90, 170, 230),
        );
        frame
            .ui
            .text(Anchor::TopLeft, format!("Footsteps: {}", game.steps));
        frame.ui.text_styled(
            Anchor::Bottom,
            "W/S speed, A/D turn, Space attack, T talk",
            16.0,
            rgb(230, 230, 230),
        );

        // Self-checks for verify runs (`python tools/verify.py anim_states --frames 780`).
        game.visited.insert(state.clone());
        if demo && game.time > 3.0 && game.time < 3.1 {
            world.check("walking plays the walk state", state == "walk");
        }
        if demo && game.time > 6.0 && game.time < 6.1 {
            world.check("jogging plays the jog state", state == "jog");
        }
        if demo && game.time > 10.5 && game.time < 10.6 {
            world.check("talking plays the talk state", state == "talk");
        }
        if demo && game.time > 12.9 && game.time < 13.0 {
            let all = ["idle", "walk", "jog", "attack", "talk"];
            world.check(
                "every state was visited",
                all.iter().all(|s| game.visited.contains(*s)),
            );
            world.check("the attack goes back to idle", state == "idle");
            world.check("walking and jogging make footsteps", game.steps >= 6);
        }
        world.report("state", state);
        world.report("footsteps", game.steps);
        world.report("speed", game.speed);
        Ok(())
    })
}
