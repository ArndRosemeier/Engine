//! Decals: scorch marks, cracks, frost, impacts, paint and glowing power
//! marks sprayed on terrain, a wall, a sliding block and people walking by.
//! Decals on the block and the people move with them; the pool holds a
//! fixed number (the oldest make room) and each fades out after a while.
//!
//! Controls: right mouse + move to look, WASD to fly, Shift fast.
//! - Left click: spray the chosen decal where the cursor points.
//! - 1..6: choose scorch, crack, frost, impact, paint, power mark.
//! - Space: spraying by itself on/off. B: cycle the budget (64, 256, 512).
//!
//! `ENGINE_DECALS_SPRAY=0` starts without the automatic spraying.
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);
const PEOPLE: usize = 8;
const WALK: f32 = 1.2;

fn rules() -> TerrainRules {
    TerrainRules {
        base_height: 0.0,
        hill_height: 2.5,
        hill_scale: 0.02,
        lake_threshold: 2.0,
        water_level: -50.0,
        ..demo_terrain_rules()
    }
}

/// A tiny deterministic random sequence.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
}

fn styled(kind: DecalKind, at: Vec3, normal: Vec3, rng: &mut Rng) -> Decal {
    let size = match kind {
        DecalKind::Scorch => rng.range(1.2, 2.6),
        DecalKind::Crack => rng.range(1.0, 2.2),
        DecalKind::Frost => rng.range(1.0, 2.0),
        DecalKind::Impact => rng.range(0.35, 0.7),
        DecalKind::Splat => rng.range(0.6, 1.4),
        _ => rng.range(1.0, 1.8),
    };
    let mut d = Decal::new(kind, at, normal, size)
        .angle(rng.range(0.0, std::f32::consts::TAU))
        .lifetime(rng.range(25.0, 40.0))
        .fade(0.2, 3.0);
    match kind {
        DecalKind::Splat => {
            let colors = [
                rgb(220, 40, 40),
                rgb(40, 140, 230),
                rgb(250, 200, 30),
                rgb(60, 200, 90),
            ];
            d = d.tint(colors[(rng.next() * 4.0) as usize % 4]);
        }
        DecalKind::Power => {
            let colors = [rgb(90, 220, 255), rgb(255, 120, 230), rgb(255, 190, 60)];
            d = d
                .tint(colors[(rng.next() * 3.0) as usize % 3])
                .emission(4.0);
        }
        _ => {}
    }
    d
}

struct Walker {
    id: EntityId,
    angle: f32,
    radius: f32,
}

fn main() -> EngineResult<()> {
    let model = Arc::new(AnimatedModel::load(CHARACTER)?);
    let ground = HeightField::new(rules());
    let mut spray = std::env::var("ENGINE_DECALS_SPRAY").map_or(true, |v| v != "0");
    let mut camera =
        FlyCamera::new((0.0, ground.height_at(0.0, -9.0) + 3.5, -9.0), 0.0, -16.0).with_speed(8.0);
    let mut people: Vec<Walker> = Vec::new();
    let mut block: Option<EntityId> = None;
    let mut wall: Option<EntityId> = None;
    let mut kind = DecalKind::Scorch;
    let mut rng = Rng(7);
    let mut next_spray = 0.0f32;
    let budgets = [64usize, 256, 512];
    let mut budget = 1;

    Engine::run("decals", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(17.0)));
            world.set_proc_terrain(ProcTerrain::gpu_clipmap(rules(), ClipmapConfig::default()));
            world.set_decal_budget(budgets[budget])?;
            // Bare ground, so the marks on it show.
            world.set_grass_scale(0.0)?;
            let gy = ground.height_at(0.0, 10.0);
            let w = world.spawn(Mesh::box_at(
                Vec3::new(0.0, gy + 2.0, 10.0),
                Vec3::new(18.0, 5.0, 0.6),
                rgb(185, 178, 166),
            )?);
            world.add_body(w, Body::fixed())?;
            wall = Some(w);
            let b = world.spawn(Mesh::box_at(
                Vec3::ZERO,
                Vec3::new(2.5, 2.5, 2.5),
                rgb(120, 130, 150),
            )?);
            world.add_body(b, Body::kinematic())?;
            block = Some(b);
            for i in 0..PEOPLE {
                let angle = i as f32 / PEOPLE as f32 * std::f32::consts::TAU;
                let radius = 4.0 + (i % 2) as f32 * 2.0;
                let id = world.spawn_animated_shared(model.clone(), Place::new(0.0, 0.0, 0.0))?;
                world.configure_animation(
                    id,
                    AnimationProfile::new()
                        .idle("idle")
                        .walk("walk")
                        .walk_speed(WALK),
                )?;
                people.push(Walker { id, angle, radius });
            }
        }
        camera.update_above(world, frame, |x, z| ground.height_at(x, z));
        world.set_proc_focus(camera.position);

        // People walk in circles; the block slides back and forth.
        for w in people.iter_mut() {
            w.angle += WALK / w.radius * frame.dt;
            let (x, z) = (w.angle.cos() * w.radius, w.angle.sin() * w.radius);
            let yaw = (w.angle + std::f32::consts::FRAC_PI_2).to_degrees();
            world.set_place(
                w.id,
                Place::new(x, ground.height_at(x, z), z).with_yaw_deg(-yaw + 180.0),
            )?;
            world.set_locomotion(w.id, Locomotion::Moving { speed_mps: WALK })?;
        }
        if let Some(b) = block {
            let x = 7.0 + (world.time() * 0.5).sin() * 4.0;
            let z = 3.0;
            world.set_place(
                b,
                Place::new(x, ground.height_at(x, z) + 1.25, z).with_yaw_deg(world.time() * 20.0),
            )?;
        }

        for (key, k) in [
            (Key::Digit1, DecalKind::Scorch),
            (Key::Digit2, DecalKind::Crack),
            (Key::Digit3, DecalKind::Frost),
            (Key::Digit4, DecalKind::Impact),
            (Key::Digit5, DecalKind::Splat),
            (Key::Digit6, DecalKind::Power),
        ] {
            if frame.input.pressed(key) {
                kind = k;
            }
        }
        if frame.input.pressed(Key::Space) {
            spray = !spray;
        }
        if frame.input.pressed(Key::B) {
            budget = (budget + 1) % budgets.len();
            world.set_decal_budget(budgets[budget])?;
        }

        // What a ray hits: a person first, then anything else.
        let aim = |world: &World, eye: Vec3, dir: Vec3| -> Option<(Vec3, Vec3, Option<EntityId>)> {
            if let Some((id, at)) = world.pick_character(eye, dir, 100.0) {
                return Some((at, -dir, Some(id)));
            }
            world
                .cast_ray(eye, dir, 200.0)
                .map(|h| (h.point, h.normal, h.entity))
        };
        let (eye, dir) = world.cursor_ray(frame);
        if frame.input.mouse_clicked(MouseButton::Left) {
            if let Some((at, n, on)) = aim(world, eye, dir) {
                world.spawn_decal(styled(kind, at, n, &mut rng).on(on))?;
            }
        }

        // Spraying by itself: a few at the start, then a steady stream at
        // the ground, the wall, the block and the people.
        let shots = if frame.first {
            40
        } else if spray && world.time() >= next_spray {
            1
        } else {
            0
        };
        if shots > 0 {
            next_spray = world.time() + 0.15;
        }
        for _ in 0..shots {
            let k = DecalKind::BUILT_IN[(rng.next() * 6.0) as usize % 6];
            let pick = rng.next();
            let (at, n, on) = if pick < 0.35 {
                // The ground.
                let (x, z) = (rng.range(-9.0, 9.0), rng.range(-6.0, 8.0));
                let y = ground.height_at(x, z);
                let e = 0.2;
                let n = Vec3::new(
                    ground.height_at(x - e, z) - ground.height_at(x + e, z),
                    2.0 * e,
                    ground.height_at(x, z - e) - ground.height_at(x, z + e),
                )
                .normalize();
                (Vec3::new(x, y, z), n, None)
            } else if pick < 0.7 {
                // The wall's face toward the camera.
                let gy = ground.height_at(0.0, 10.0);
                (
                    Vec3::new(rng.range(-8.0, 8.0), gy + rng.range(0.3, 4.2), 9.7),
                    Vec3::NEG_Z,
                    wall,
                )
            } else if pick < 0.85 {
                // The sliding block, on a side facing the camera.
                let Some(b) = block else { continue };
                let t = world.entity(b)?.transform();
                let c = t.w_axis.truncate();
                // The side facing the camera most.
                let n = [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z]
                    .map(|a| t.transform_vector3(a).normalize())
                    .into_iter()
                    .min_by(|a, b| a.z.total_cmp(&b.z))
                    .unwrap_or(Vec3::NEG_Z);
                let side = n.cross(Vec3::Y);
                let at =
                    c + n * 1.25 + side * rng.range(-0.8, 0.8) + Vec3::Y * rng.range(-0.8, 0.8);
                (at, n, Some(b))
            } else {
                // Someone walking by.
                let w = &people[(rng.next() * PEOPLE as f32) as usize % PEOPLE];
                let t = world.animated_entity(w.id)?.transform();
                let c = t.w_axis.truncate() + Vec3::Y * rng.range(0.8, 1.5);
                let from = c + Vec3::new(rng.range(-1.0, 1.0), 0.3, -4.0);
                match world.pick_character(from, (c - from).normalize(), 10.0) {
                    Some((id, at)) => (at, (from - at).normalize(), Some(id)),
                    None => continue,
                }
            };
            let size_scale = if on.is_some_and(|id| people.iter().any(|p| p.id == id)) {
                0.35
            } else {
                1.0
            };
            let d = styled(k, at, n, &mut rng);
            let d = d.size(d.size[0] * size_scale, d.size[1] * size_scale);
            let d = if size_scale < 1.0 { d.depth(0.35) } else { d };
            world.spawn_decal(d.on(on))?;
        }

        engine::egui::Window::new("decals")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   decals: {} of {}",
                    frame.fps,
                    world.decal_count(),
                    world.decal_budget()
                ));
                ui.label(format!(
                    "click: spray {kind:?}   1-6: kind   Space: auto spray {}   B: budget",
                    if spray { "on" } else { "off" }
                ));
            });
        Ok(())
    })
}
