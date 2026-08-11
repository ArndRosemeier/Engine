//! Skinned glTF demo (Quaternius deer).
//!
//! Space / E cycles Idle → Walk → Gallop (also auto-cycles every 4s). Esc quits.

use engine::prelude::*;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let asset = root.join("assets/deer.gltf");
    eprintln!("loading {}", asset.display());
    let model = AnimatedModel::load_with(&asset, &root, &EngineLimits::default())
        .unwrap_or_else(|e| panic!("failed to load deer: {e}"));
    eprintln!(
        "deer: {} joints, {} meshes, clips: {:?}",
        model.skeleton.joint_names.len(),
        model.meshes.len(),
        model.clip_names().collect::<Vec<_>>()
    );

    let mut animal = None;
    let mut clip_i = 0usize;
    let mut was_cycle = false;
    let clips = ["Idle", "Walk", "Gallop"];

    Engine::run("animated_animal", move |world, frame| {
        if frame.first {
            world.clear_color = engine::Color::rgb(140, 190, 230);
            world.spawn(
                Shape::box_at((0.0, -0.05, 0.0), (20.0, 0.1, 20.0), rgb(90, 140, 70)).unwrap(),
            );
            let id = world
                .spawn_animated(
                    model.clone(),
                    Place::at(0.0, 0.0, 0.0)
                        .unwrap()
                        .scale(0.45)
                        .unwrap(),
                )
                .expect("spawn deer");
            world.play_animation(id, "Idle").expect("Idle");
            animal = Some(id);
            eprintln!("controls: Space/E cycle clips (Idle/Walk/Gallop)");
        }

        let cycle_held = frame.input.down(Key::Space) || frame.input.down(Key::E);
        let cycle_pressed = cycle_held && !was_cycle;
        was_cycle = cycle_held;
        if cycle_pressed {
            if let Some(id) = animal {
                clip_i = (clip_i + 1) % clips.len();
                let name = clips[clip_i];
                if let Err(e) = world.play_animation(id, name) {
                    eprintln!("play {name}: {e}");
                } else {
                    eprintln!("clip -> {name}");
                }
            }
        }

        let auto = ((frame.time / 4.0) as usize) % clips.len();
        if auto != clip_i {
            if let Some(id) = animal {
                clip_i = auto;
                let _ = world.play_animation(id, clips[clip_i]);
            }
        }

        world.look_orbit(Vec3::new(0.0, 1.2, 0.0), 7.0, frame.time * 18.0, 22.0);
        world.set_sun(Vec3::new(0.45, 1.0, 0.25), 0.24);
    });
}
