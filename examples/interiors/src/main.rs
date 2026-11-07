//! Indoor culling: a city block where every tower is furnished floor by
//! floor (desks, chairs, lamps, shelves of books: about a million triangles
//! of interiors). Only rooms you can see through windows, doorways and holes
//! are drawn, so from the street it runs as fast as the empty block.
//!
//! Controls: right mouse + move to look, WASD to fly, Shift fast.
//! - C: indoor culling on/off (compare the frame rate).
//! - I: interiors on/off.
//! - Left click: punch a hole in the wall under the cursor (rooms behind open up).
//! - F: a blast under the cursor.
//!
//! `ENGINE_INTERIORS_CULL=0` starts with culling off, `ENGINE_INTERIORS_FURNISH=0`
//! without interiors, `ENGINE_INTERIORS_VIEW=inside` inside an office.
use engine::building::{BuildingPlan, TowerSpec};
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

struct Tower {
    at: Vec3,
    plan: Arc<BuildingPlan>,
}

fn block() -> EngineResult<Vec<(Vec3, Arc<BuildingPlan>)>> {
    let mut out = Vec::new();
    let mut seed = 1u64;
    for row in 0..4 {
        for col in 0..4 {
            seed += 1;
            let floors = 4 + ((seed * 7) % 9) as u16;
            let spec = TowerSpec {
                width: 15.0 + ((seed * 3) % 3) as f32 * 3.0,
                depth: 12.0,
                floors,
                glazing: 0.3 + ((seed * 5) % 4) as f32 * 0.12,
                wall_color: rgb(
                    150 + ((seed * 37) % 60) as u8,
                    140 + ((seed * 53) % 50) as u8,
                    130 + ((seed * 29) % 50) as u8,
                ),
                seed,
                ..Default::default()
            };
            let at = Vec3::new(col as f32 * 26.0 - 39.0, 0.0, row as f32 * 22.0 + 14.0);
            out.push((at, Arc::new(BuildingPlan::tower(&spec)?)));
        }
    }
    Ok(out)
}

/// Desks, chairs, desk lamps and shelves filling a room (half size `half`,
/// floor at y = 0), one mesh.
fn furniture(half: Vec3, seed: u64) -> EngineResult<Mesh> {
    let mut m = Mesh::new();
    let wood = rgb(140, 100, 64);
    let metal = rgb(90, 92, 98);
    let lamp = Mesh::ball(Vec3::ZERO, 0.12, rgb(250, 230, 170))?;
    let (hx, hz) = (half.x - 1.0, half.z - 1.0);
    let mut x = -hx;
    let mut k = seed;
    while x <= hx {
        let mut z = -hz;
        while z <= hz {
            k = k
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Desk: top and four legs.
            m.add_box(Vec3::new(x, 0.74, z), Vec3::new(1.4, 0.04, 0.7), wood)?;
            for (dx, dz) in [(-0.65, -0.3), (0.65, -0.3), (-0.65, 0.3), (0.65, 0.3)] {
                m.add_box(
                    Vec3::new(x + dx, 0.36, z + dz),
                    Vec3::new(0.05, 0.72, 0.05),
                    metal,
                )?;
            }
            // Chair.
            m.add_box(
                Vec3::new(x, 0.45, z + 0.65),
                Vec3::new(0.45, 0.05, 0.45),
                metal,
            )?;
            m.add_box(
                Vec3::new(x, 0.75, z + 0.88),
                Vec3::new(0.45, 0.6, 0.05),
                metal,
            )?;
            // Lamp, a monitor and papers.
            m.append_translated(&lamp, Vec3::new(x - 0.5, 0.9, z - 0.15))?;
            m.add_box(
                Vec3::new(x + 0.2, 1.0, z - 0.2),
                Vec3::new(0.6, 0.38, 0.04),
                rgb(20, 22, 26),
            )?;
            for p in 0..4 {
                let c = 200 + ((k >> (p * 8)) % 50) as u8;
                m.add_box(
                    Vec3::new(x - 0.2 + p as f32 * 0.08, 0.77 + p as f32 * 0.004, z + 0.1),
                    Vec3::new(0.21, 0.004, 0.3),
                    rgb(c, c, c - 10),
                )?;
            }
            z += 2.4;
        }
        x += 2.6;
    }
    // A shelf of books along the back wall.
    let mut x = -hx;
    while x <= hx {
        m.add_box(
            Vec3::new(x, 1.0, half.z - 0.35),
            Vec3::new(1.8, 2.0, 0.04),
            wood,
        )?;
        for shelf in 0..4 {
            let y = 0.25 + shelf as f32 * 0.48;
            m.add_box(
                Vec3::new(x, y, half.z - 0.5),
                Vec3::new(1.8, 0.03, 0.3),
                wood,
            )?;
            for b in 0..14 {
                k = k.wrapping_mul(6364136223846793005).wrapping_add(1);
                let h = 0.2 + ((k >> 33) % 18) as f32 * 0.01;
                let col = rgb(
                    60 + ((k >> 8) % 160) as u8,
                    50 + ((k >> 16) % 120) as u8,
                    40 + ((k >> 24) % 140) as u8,
                );
                m.add_box(
                    Vec3::new(
                        x - 0.82 + b as f32 * 0.125,
                        y + 0.015 + h * 0.5,
                        half.z - 0.5,
                    ),
                    Vec3::new(0.1, h, 0.22),
                    col,
                )?;
            }
        }
        x += 2.0;
    }
    Ok(m)
}

/// Furnish every room of every tower; returns the furniture entities.
fn furnish(world: &mut World, towers: &[Tower]) -> EngineResult<Vec<EntityId>> {
    let mut out = Vec::new();
    for t in towers {
        for (i, (c, h)) in t.plan.rooms().iter().enumerate() {
            let floor_y = c.y - h.y;
            let mesh = furniture(*h, t.plan.seed() * 131 + i as u64)?;
            let p = t.at + Vec3::new(c.x, floor_y, c.z);
            let id = world.place(mesh, Place::new(p.x, p.y, p.z))?;
            world.set_room_by_position(id)?;
            out.push(id);
        }
    }
    Ok(out)
}

fn main() -> EngineResult<()> {
    let plans = block()?;
    let inside = setting("ENGINE_INTERIORS_VIEW").as_deref() == Some("inside");
    let mut culling = setting("ENGINE_INTERIORS_CULL").as_deref() != Some("0");
    let mut furnished = setting("ENGINE_INTERIORS_FURNISH").as_deref() != Some("0");
    let mut camera = if inside {
        FlyCamera::new((-37.0, 3.4 + 1.7, 10.0), 15.0, -6.0)
    } else {
        FlyCamera::new((-6.0, 1.8, -4.0), 10.0, 6.0)
    }
    .with_speed(10.0);
    let mut towers: Vec<Tower> = Vec::new();
    let mut stuff: Vec<EntityId> = Vec::new();

    Engine::run("interiors", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.5)));
            world.spawn(Shape::box_at(
                (0.0, -0.05, 40.0),
                (200.0, 0.1, 200.0),
                rgb(70, 72, 76),
            )?);
            for (at, plan) in &plans {
                world.spawn_building(plan.clone(), Place::new(at.x, at.y, at.z))?;
                towers.push(Tower {
                    at: *at,
                    plan: plan.clone(),
                });
            }
            if furnished {
                stuff = furnish(world, &towers)?;
            }
            world.set_room_culling(culling);
            if inside {
                // Ceiling lights in the offices around the camera.
                let eye = camera.position;
                for t in &towers {
                    for (c, h) in t.plan.rooms() {
                        let p = t.at + *c + Vec3::new(0.0, h.y - 0.3, 0.0);
                        if p.distance(eye) < 18.0 {
                            world.add_lamp(Lamp::point(p, rgb(255, 236, 205), 5.0, 9.0))?;
                        }
                    }
                }
            }
        }
        camera.update(world, frame);

        if frame.input.pressed(Key::C) {
            culling = !culling;
            world.set_room_culling(culling);
        }
        if frame.input.pressed(Key::I) {
            furnished = !furnished;
            if furnished {
                stuff = furnish(world, &towers)?;
            } else {
                for id in stuff.drain(..) {
                    world.despawn(id);
                }
            }
        }
        let (eye, dir) = world.cursor_ray(frame);
        let aim = world.cast_ray_buildings(eye, dir, 300.0);
        if frame.input.mouse_clicked(MouseButton::Left) {
            if let Some((id, _, at)) = aim {
                world.hit_building(id, at, 1.4, 900.0)?;
            }
        }
        if frame.input.pressed(Key::F) {
            if let Some((_, _, at)) = aim {
                world.hit_buildings(at, 4.0, 3000.0);
            }
        }

        let (rooms, openings, seen) = world.room_count();
        engine::egui::Window::new("interiors")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   rooms drawn: {} of {rooms} ({openings} openings)",
                    frame.fps,
                    if culling { seen } else { rooms },
                ));
                ui.label(format!(
                    "C: culling {}   I: interiors {}   click: punch   F: blast",
                    if culling { "on" } else { "off" },
                    if furnished { "on" } else { "off" },
                ));
            });
        Ok(())
    })
}
