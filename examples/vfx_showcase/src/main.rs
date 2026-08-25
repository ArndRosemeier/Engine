use engine::prelude::*;
use std::{env, path::PathBuf};

fn parse_kind() -> VisualKind {
    match env::var("VFX_KIND")
        .unwrap_or_else(|_| "fire".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "fire" => VisualKind::Fire,
        "frost" => VisualKind::Frost,
        "lightning" => VisualKind::Lightning,
        "poison" => VisualKind::Poison,
        "root" => VisualKind::Root,
        "hold" => VisualKind::Hold,
        "snare" => VisualKind::Snare,
        "charm" => VisualKind::Charm,
        v => panic!("unknown VFX_KIND {v}"),
    }
}
fn parse_delivery() -> Delivery {
    match env::var("VFX_DELIVERY")
        .unwrap_or_else(|_| "single".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "single" | "singletarget" => Delivery::SingleTarget,
        "aoe" => Delivery::Aoe,
        "pbaoe" => Delivery::Pbaoe,
        "cone" => Delivery::Cone,
        v => panic!("unknown VFX_DELIVERY {v}"),
    }
}
fn spec(kind: VisualKind, delivery: Delivery, seed: u32) -> EffectSpec {
    EffectSpec {
        kind,
        delivery,
        origin: Vec3::new(-4.6, 1.45, 0.0),
        target: Vec3::new(3.3, 0.25, 0.0),
        range_m: 6.8,
        radius_m: 2.7,
        angle_deg: 68.0,
        duration_s: 3.2,
        scale: 1.15,
        intensity: 1.0,
        seed,
    }
}
fn dummy(center: Vec3, color: Color) -> Mesh {
    let mut m = Mesh::new();
    let body = color.lerp(Color::BLACK, 0.38);
    m.add_box(center + Vec3::Y * 0.92, (0.25, 1.25, 0.22), body)
        .unwrap();
    m.add_box(
        center + Vec3::Y * 1.68,
        (0.34, 0.34, 0.32),
        color.lerp(Color::BLACK, 0.18),
    )
    .unwrap();
    for x in [-0.19, 0.19] {
        m.add_box(center + Vec3::new(x, 0.30, 0.0), (0.11, 0.58, 0.12), body)
            .unwrap();
        m.add_box(
            center + Vec3::new(x * 1.35, 1.03, 0.0),
            (0.09, 0.72, 0.10),
            body,
        )
        .unwrap();
    }
    m
}
fn main() {
    let mut kind = parse_kind();
    let mut delivery = parse_delivery();
    let fixed_time = env::var("VFX_SCREENSHOT_TIME").ok().map(|v| {
        v.parse::<f32>()
            .expect("VFX_SCREENSHOT_TIME must be seconds")
    });
    let shot = env::var_os("ENGINE_SCREENSHOT").map(PathBuf::from);
    let mut vfx = VfxSystem::new();
    let mut elapsed = 0.0_f32;
    let mut capture_armed = false;
    let mut seed = 1_u32;
    Engine::run("vfx_showcase", move |world, frame| {
        if frame.first {
            world.set_clear_color(rgb(10, 13, 20));
            world.set_sun((-0.45, -1.0, -0.25), 0.78);
            world.set_bloom(
                BloomSettings::default()
                    .with_threshold(1.35)
                    .expect("showcase bloom threshold")
                    .with_soft_knee(0.32)
                    .expect("showcase bloom knee")
                    .with_intensity(0.18)
                    .expect("showcase bloom intensity")
                    .with_exposure(0.82)
                    .expect("showcase exposure"),
            );
            world.spawn(
                Shape::box_at((0.0, -0.18, 0.0), (19.0, 0.2, 10.0), rgb(34, 39, 47))
                    .expect("ground"),
            );
            world.spawn(dummy(Vec3::new(-4.6, 0.0, 0.0), rgb(48, 65, 88)));
            world.spawn(dummy(Vec3::new(3.3, 0.0, 0.0), rgb(70, 73, 80)));
            vfx.spawn(world, spec(kind, delivery, seed))
                .expect("cast VFX");
        }
        if fixed_time.is_none() {
            for (key, value) in [
                (Key::Digit1, VisualKind::Fire),
                (Key::Digit2, VisualKind::Frost),
                (Key::Digit3, VisualKind::Lightning),
                (Key::Digit4, VisualKind::Poison),
                (Key::Digit5, VisualKind::Root),
                (Key::Digit6, VisualKind::Hold),
                (Key::Digit7, VisualKind::Snare),
                (Key::Digit8, VisualKind::Charm),
            ] {
                if frame.input.pressed(key) {
                    kind = value;
                }
            }
            for (key, value) in [
                (Key::Q, Delivery::SingleTarget),
                (Key::W, Delivery::Aoe),
                (Key::E, Delivery::Pbaoe),
                (Key::R, Delivery::Cone),
            ] {
                if frame.input.pressed(key) {
                    delivery = value;
                }
            }
            if frame.input.pressed(Key::Space) && vfx.active_count() < 12 {
                seed = seed.wrapping_add(1);
                vfx.spawn(world, spec(kind, delivery, seed))
                    .expect("cast VFX");
            }
        }
        let dt = if fixed_time.is_some() {
            1.0 / 60.0
        } else {
            frame.dt.min(0.05)
        };
        vfx.update(world, dt).expect("update VFX");
        elapsed += dt;
        world.look_orbit((1.25, 1.25, 0.0), 6.75, -20.0, 15.0);
        if let (Some(at), Some(path)) = (fixed_time, shot.as_ref()) {
            if capture_armed {
                world.queue_screenshot(path);
                world.request_exit();
            } else if elapsed >= at {
                capture_armed = true;
            }
        }
        engine::egui::Window::new("VFX Showcase")
            .anchor(engine::egui::Align2::RIGHT_TOP, [-10.0, 10.0])
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.small(format!("{kind:?} / {delivery:?} · {:.2}s", elapsed));
                ui.small("1-8 kind · Q/W/E/R delivery · Space cast");
            });
    });
}
