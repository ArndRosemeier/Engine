//! Breakable buildings: a block of towers made of wall panels, slab tiles,
//! columns and roof pieces. Break panels, and when a storey loses its
//! support the floors above pancake or topple, shatter and leave rubble.
//!
//! Controls: right mouse + move to look, WASD to fly, Shift fast.
//! - Left click: punch the wall under the cursor.
//! - F: a blast under the cursor.
//! - T: blow out one side of the nearest tower's first floor (it topples).
//! - P: knock out most of the nearest tower's ground floor all round (it pancakes).
//! - R: rebuild everything.
//!
//! `ENGINE_BUILDINGS_DEMO=1` plays a topple and a pancake by itself.
use engine::building::{BuildingPlan, ElementId, ElementKind, TowerSpec};
use engine::prelude::*;
use glam::Vec3;
use std::sync::Arc;

struct Block {
    towers: Vec<(EntityId, Vec3)>,
}

fn plans() -> EngineResult<Vec<Arc<BuildingPlan>>> {
    let specs = [
        TowerSpec::default(),
        TowerSpec {
            width: 12.0,
            depth: 12.0,
            floors: 10,
            glazing: 0.6,
            wall_color: rgb(170, 176, 186),
            seed: 2,
            ..Default::default()
        },
        TowerSpec {
            width: 18.0,
            depth: 10.0,
            floors: 4,
            wall_color: rgb(184, 120, 92),
            seed: 3,
            ..Default::default()
        },
        TowerSpec {
            width: 9.0,
            depth: 9.0,
            floors: 14,
            glazing: 0.85,
            wall_color: rgb(120, 130, 140),
            glass_color: rgb(60, 110, 150),
            seed: 4,
            ..Default::default()
        },
    ];
    specs
        .iter()
        .map(|s| BuildingPlan::tower(s).map(Arc::new))
        .collect()
}

fn build(world: &mut World, plans: &[Arc<BuildingPlan>]) -> EngineResult<Block> {
    let spots = [
        Vec3::new(-20.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 14.0),
        Vec3::new(20.0, 0.0, 10.0),
        Vec3::new(4.0, 0.0, 38.0),
    ];
    let mut towers = Vec::new();
    for (plan, at) in plans.iter().zip(spots) {
        let id = world.spawn_building(plan.clone(), Place::new(at.x, at.y, at.z))?;
        towers.push((id, at));
    }
    Ok(Block { towers })
}

fn nearest(block: &Block, world: &World, to: Vec3) -> Option<EntityId> {
    block
        .towers
        .iter()
        .filter(|(id, _)| world.building(*id).is_some_and(|b| b.collapse().is_none()))
        .min_by(|a, b| a.1.distance(to).total_cmp(&b.1.distance(to)))
        .map(|(id, _)| *id)
}

/// The front (−Z) panels of floor 1: losing them topples the tower forward.
fn topple(world: &mut World, id: EntityId) -> EngineResult<()> {
    let Some(b) = world.building(id) else {
        return Ok(());
    };
    let ids: Vec<ElementId> = b
        .plan()
        .elements()
        .iter()
        .enumerate()
        .filter(|(_, e)| e.floor == 1 && e.outward.z < -0.9)
        .map(|(i, _)| ElementId(i as u32))
        .collect();
    world.break_building_elements(id, &ids)?;
    Ok(())
}

/// Two of every three ground-floor panels and the columns: it pancakes.
fn pancake(world: &mut World, id: EntityId) -> EngineResult<()> {
    let Some(b) = world.building(id) else {
        return Ok(());
    };
    let ids: Vec<ElementId> = b
        .plan()
        .elements()
        .iter()
        .enumerate()
        .filter(|(i, e)| {
            e.floor == 0
                && (e.kind == ElementKind::Column || (e.outward != Vec3::ZERO && i % 3 != 0))
        })
        .map(|(i, _)| ElementId(i as u32))
        .collect();
    world.break_building_elements(id, &ids)?;
    Ok(())
}

fn main() -> EngineResult<()> {
    let demo = std::env::var("ENGINE_BUILDINGS_DEMO").is_ok_and(|v| v != "0");
    let plans = plans()?;
    let mut camera = FlyCamera::new((0.0, 12.0, -38.0), 0.0, -8.0).with_speed(14.0);
    let mut block: Option<Block> = None;
    let mut frame_no = 0u32;

    Engine::run("buildings", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.0)));
            world.spawn(Shape::box_at(
                (0.0, -0.05, 20.0),
                (160.0, 0.1, 160.0),
                rgb(92, 96, 88),
            )?);
            world.spawn(Shape::box_at(
                (0.0, 0.0, -6.0),
                (160.0, 0.02, 10.0),
                rgb(60, 60, 64),
            )?);
            block = Some(build(world, &plans)?);
        }
        frame_no += 1;
        camera.update(world, frame);
        let b = block.as_ref().expect("built");

        let (eye, dir) = world.cursor_ray(frame);
        let aim = world.cast_ray_buildings(eye, dir, 300.0);
        if frame.input.mouse_clicked(MouseButton::Left) {
            if let Some((id, _, at)) = aim {
                world.hit_building(id, at, 1.6, 900.0)?;
            }
        }
        if frame.input.pressed(Key::F) {
            if let Some((_, _, at)) = aim {
                world.hit_buildings(at, 4.5, 4000.0);
                world.blast(at, 8.0, 10.0);
            }
        }
        let near = nearest(b, world, eye);
        if frame.input.pressed(Key::T) {
            if let Some(id) = near {
                topple(world, id)?;
            }
        }
        if frame.input.pressed(Key::P) {
            if let Some(id) = near {
                pancake(world, id)?;
            }
        }
        if demo {
            if frame_no == 20 {
                topple(world, b.towers[1].0)?;
            }
            if frame_no == 30 {
                pancake(world, b.towers[0].0)?;
            }
        }
        if frame.input.pressed(Key::R) {
            for (id, _) in &b.towers {
                world.despawn(*id);
            }
            block = Some(build(world, &plans)?);
        }

        let (flying, resting) = world.rubble_count();
        engine::egui::Window::new("buildings")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!(
                    "{:.0} FPS   debris: {flying} flying, {resting} resting",
                    frame.fps
                ));
                ui.label("click: punch   F: blast   T: topple   P: pancake   R: rebuild");
            });
        Ok(())
    })
}
