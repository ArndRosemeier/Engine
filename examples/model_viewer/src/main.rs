//! Look at any glTF / GLB file with its full PBR materials.
//!
//! `cargo run --release -p model_viewer -- path/to/model.glb`
//! (or an AssetGenerator `.fmpk` prop pack: every prop in a row)
//!
//! Skinned models (characters exported from Actors, animals) play their
//! first clip; Space steps to the next one.
//! `assets/character.glb` is a dressed Actors character with five clips. Static models load one part per
//! material. Hold the right mouse button to orbit, W/S to zoom.
use engine::prelude::*;
use glam::Vec3;
use std::path::PathBuf;

enum Loaded {
    Static(ModelScene),
    /// An AssetGenerator prop pack: every prop, side by side.
    Props(ModelScene),
    Skinned(AnimatedModel),
}

/// Bounds of the first frame of the first clip, after skinning.
fn skinned_bounds(model: &AnimatedModel) -> Option<(Vec3, Vec3)> {
    let joints = engine::anim::sample_joint_matrices(model, 0, 0.0);
    model
        .meshes
        .iter()
        .flat_map(|m| {
            m.positions
                .iter()
                .zip(&m.joints)
                .zip(&m.weights)
                .map(|((p, j), w)| {
                    (0..4)
                        .map(|k| {
                            let mat = joints.get(j[k] as usize).copied().unwrap_or_default();
                            mat.transform_point3(*p) * w[k]
                        })
                        .sum::<Vec3>()
                })
        })
        .fold(None, |acc, p| match acc {
            None => Some((p, p)),
            Some((lo, hi)) => Some((lo.min(p), hi.max(p))),
        })
}

fn main() -> EngineResult<()> {
    let Some(path) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: model_viewer <model.gltf|model.glb>");
        std::process::exit(2);
    };
    let path = std::fs::canonicalize(&path)?;
    let root = path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let limits = EngineLimits {
        max_gltf_buffer_bytes: 512 * 1024 * 1024,
        max_model_triangles: 8_000_000,
        ..EngineLimits::default()
    };
    let started = std::time::Instant::now();
    let is_pack = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fmpk"));
    let loaded = if is_pack {
        let pack = PropPack::load(&path)?;
        let mut parts = Vec::new();
        let mut x = 0.0;
        for model in &pack.models {
            let mut mesh = Mesh::new();
            let width = model.max.x - model.min.x;
            mesh.append_translated(&model.to_mesh()?, Vec3::new(x - model.min.x, 0.0, 0.0))?;
            x += width + 0.3;
            parts.push(mesh);
        }
        eprintln!("props: {}", pack.names().collect::<Vec<_>>().join(", "));
        Loaded::Props(ModelScene { parts })
    } else {
        match AnimatedModel::load_with(&path, &root, &limits) {
            Ok(model) => Loaded::Skinned(model),
            Err(_) => Loaded::Static(Model::load_scene_with(&path, &root, &limits)?),
        }
    };
    let (bounds, what) = match &loaded {
        Loaded::Static(scene) | Loaded::Props(scene) => (
            scene.bounds(),
            format!(
                "{} parts, {} triangles",
                scene.parts.len(),
                scene.triangle_count()
            ),
        ),
        Loaded::Skinned(model) => (
            skinned_bounds(model),
            format!(
                "{} meshes, {} joints, clips {:?}",
                model.meshes.len(),
                model.skeleton.joint_names.len(),
                model.clip_names().collect::<Vec<_>>()
            ),
        ),
    };
    eprintln!(
        "{}: {what} ({:.0} ms)",
        path.display(),
        started.elapsed().as_secs_f32() * 1000.0
    );
    let (lo, hi) = bounds.unwrap_or((Vec3::splat(-1.0), Vec3::splat(1.0)));
    // Stand the model on the ground at the origin (single models about 2 m across).
    let size = (hi - lo).max_element().max(1e-3);
    // Packs are already in real metres: keep them that size.
    let scale = if matches!(loaded, Loaded::Props(_)) {
        1.0
    } else {
        2.0 / size
    };
    let offset = Vec3::new(-(lo.x + hi.x) * 0.5, -lo.y, -(lo.z + hi.z) * 0.5) * scale;
    let height = (hi.y - lo.y) * scale;

    let mut yaw = std::env::var("ENGINE_VIEW_YAW")
        .ok()
        .and_then(|v| v.parse().ok())
        // A pack's row reads best from the front.
        .unwrap_or(if matches!(loaded, Loaded::Props(_)) {
            70.0
        } else {
            30.0_f32
        });
    let mut pitch = 12.0_f32;
    let env = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<f32>().ok());
    // ENGINE_VIEW_DISTANCE (metres) and ENGINE_VIEW_FOCUS (0 = feet, 1 = top)
    // frame close-ups for screenshots.
    let mut distance = env("ENGINE_VIEW_DISTANCE").unwrap_or((size * scale * 1.4).max(2.5));
    let focus = env("ENGINE_VIEW_FOCUS").unwrap_or(0.5);
    let mut clip = 0usize;
    let mut animated = None;

    Engine::run("model_viewer", move |world, frame| {
        world.set_vsync(false);
        if frame.first {
            world.set_clear_color(rgb(150, 180, 210));
            world.set_sun((0.5, 0.8, 0.35), 0.3);
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (12.0, 0.1, 12.0),
                rgb(120, 120, 115),
            )?);
            let place = Place::at(offset.x, offset.y, offset.z)?.scale(scale)?;
            match &loaded {
                Loaded::Static(scene) | Loaded::Props(scene) => {
                    world.spawn_model(scene, place)?;
                }
                Loaded::Skinned(model) => {
                    let id = world.spawn_animated(model.clone(), place)?;
                    let clips: Vec<String> = model.clip_names().map(String::from).collect();
                    // ENGINE_VIEW_CLIP picks the starting clip by name.
                    if let Ok(name) = std::env::var("ENGINE_VIEW_CLIP") {
                        clip = clips.iter().position(|c| *c == name).unwrap_or(0);
                    }
                    if let Some(name) = clips.get(clip) {
                        world.play_animation(id, name)?;
                    }
                    animated = Some((id, clips));
                }
            }
        }
        let input = &frame.input;
        if input.mouse_down(MouseButton::Right) {
            let d = input.mouse_delta();
            yaw += d.x * 0.3;
            pitch = (pitch + d.y * 0.3).clamp(-80.0, 85.0);
        }
        distance = (distance + input.axis(Key::W, Key::S) * frame.dt * 3.0).clamp(0.5, 50.0);
        if input.pressed(Key::Space) {
            if let Some((id, clips)) = &animated {
                if !clips.is_empty() {
                    clip = (clip + 1) % clips.len();
                    world.play_animation(*id, &clips[clip])?;
                    eprintln!("clip {}", clips[clip]);
                }
            }
        }
        world.look_orbit(Vec3::new(0.0, height * focus, 0.0), distance, yaw, pitch);
        Ok(())
    })
}
