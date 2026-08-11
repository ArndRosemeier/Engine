//! Infinite GPU-procgen landscape POC with lakes and a keyboard walker.
//!
//! Controls: WASD / arrows move · Q/E turn · Shift sprint · Esc quit
//!
//! Landscape is a GPU clipmap (formula in the vertex shader) — no CPU mesh bake.
use engine::prelude::*;

fn walker_mesh() -> Mesh {
    let mut m = Mesh::new();
    // Feet at local y=0 so Place.y is the ground contact.
    m.add_box((0.0, 0.55, 0.0), (0.55, 1.1, 0.35), rgb(55, 90, 160))
        .unwrap();
    m.add_box((0.0, 1.35, 0.0), (0.4, 0.4, 0.4), rgb(220, 190, 160))
        .unwrap();
    m
}

fn main() {
    let rules = demo_terrain_rules();
    let field = HeightField::new(rules.clone());
    let mut pos = Vec3::new(0.0, 0.0, 0.0);
    pos.y = field.height_at(pos.x, pos.z) + 0.05;
    let mut yaw = 20.0_f32;
    let mut walker: Option<EntityId> = None;

    let screenshot = std::env::var_os("ENGINE_SCREENSHOT").is_some();
    if screenshot && std::env::var_os("ENGINE_SCREENSHOT_ORIGIN").is_none() {
        let mut best: Option<(f32, f32, f32)> = None;
        for z in -90..90 {
            for x in -90..90 {
                let px = x as f32 * 3.0;
                let pz = z as f32 * 3.0;
                if !field.sample(px, pz).water {
                    continue;
                }
                for (dx, dz) in [(-6.0, -4.0), (-8.0, -2.0), (-5.0, -7.0), (6.0, -5.0)] {
                    let sx = px + dx;
                    let sz = pz + dz;
                    let shore = field.sample(sx, sz);
                    if shore.water {
                        continue;
                    }
                    let score = (shore.height - rules.water_level).abs();
                    if best.map(|(s, _, _)| score < s).unwrap_or(true) {
                        best = Some((score, sx, sz));
                    }
                }
            }
        }
        if let Some((_, sx, sz)) = best {
            pos = Vec3::new(sx, 0.0, sz);
            yaw = 35.0;
        }
    }

    Engine::run("infinite_walker", move |world, frame| {
        if frame.first {
            world.set_clear_color(rgb(145, 195, 235));
            world.set_sun((0.45, 1.0, 0.25), 0.28);
            world.set_proc_terrain(
                ProcTerrain::gpu_clipmap(
                    rules.clone(),
                    ClipmapConfig {
                        rings: 4,
                        resolution: 128,
                        cell_size: 0.5,
                    },
                )
                .with_focus(pos),
            );
            walker = Some(world.spawn(walker_mesh()));
        }

        if !screenshot {
            yaw += frame.input.yaw_sign() * 90.0 * frame.dt;
            let dir = frame.input.move_dir_xz(yaw);
            if dir.length_squared() > 0.0 {
                let speed = if frame.input.down(Key::Shift) {
                    14.0
                } else {
                    7.0
                };
                pos += dir * speed * frame.dt;
            }
        }

        // Focus first so walk height uses the same snapped grid the GPU draws.
        world.set_proc_focus(pos);
        pos.y = world.proc_walk_height(pos.x, pos.z) + 0.02;

        if let Some(id) = walker {
            world
                .set_place(id, Place::new(pos.x, pos.y, pos.z).with_yaw_deg(yaw))
                .expect("walker");
        }

        world.look_follow(pos, yaw, 9.0, 4.5);
    });
}
