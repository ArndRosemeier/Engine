//! Starter template: copy this folder to begin a new project.
//!
//! An endless procedural landscape under a physical sky, with a mixed wood,
//! shrubs and rocks scattered over it and a free-fly camera. About a hundred
//! lines, using only `engine::prelude`.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast, Esc quits.
//!
//! `ENGINE_QUALITY=low|medium|high|ultra` picks a preset,
//! `ENGINE_TARGET_FPS=60` lets the engine trade detail for frame rate,
//! `ENGINE_SKY_HOUR=8.5` sets the time of day, `ENGINE_GRASS=0` hides the
//! grass (`0.5` halves its reach).
use engine::prelude::*;

/// Trees of the wood, in the order they are dealt places.
const WOOD: [TreeSpecies; 5] = [
    TreeSpecies::Oak,
    TreeSpecies::Birch,
    TreeSpecies::Pine,
    TreeSpecies::Linden,
    TreeSpecies::Maple,
];

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Put the scenery on the terrain. Everything here is deterministic.
fn populate(world: &mut World, ground: &HeightField, clearing: Vec3) -> EngineResult<()> {
    // Dry, gentle ground only: no trees in lakes or on steep slopes, and a
    // clearing where the camera starts.
    let dry = |x: f32, z: f32| {
        if Vec3::new(x - clearing.x, 0.0, z - clearing.z).length() < 35.0 {
            return false;
        }
        let slope = (ground.height_at(x + 1.0, z) - ground.height_at(x - 1.0, z)).abs()
            + (ground.height_at(x, z + 1.0) - ground.height_at(x, z - 1.0)).abs();
        !ground.sample(x, z).water && slope < 1.6
    };
    // Sink models a little so their roots never float on a slope.
    let height = |x: f32, z: f32| ground.height_at(x, z) - 0.15;

    let trees = Scatter::new(1)
        .disc(0.0, 0.0, 450.0)
        .spacing(11.0)
        .density(0.55)
        .patch_size(70.0)
        .scale_range(0.8, 1.15)
        .on_ground(height)
        .only_where(dry)
        .places()?;
    // Two individuals per species, so neighbours differ.
    // Generating models is the slow part of startup: build them all at once.
    let kinds: Vec<usize> = (0..WOOD.len() * 2).collect();
    let models = build_all(&kinds, |&k| tree_model(WOOD[k % WOOD.len()], k as u32))?;
    for (tree, places) in models.iter().zip(Scatter::split(&trees, kinds.len(), 1)) {
        world.spawn_lod_at(tree.lod_levels(), &places)?;
    }

    let shrubs = Scatter::new(2)
        .disc(0.0, 0.0, 250.0)
        .spacing(7.0)
        .density(0.45)
        .scale_range(0.7, 1.3)
        .on_ground(height)
        .only_where(dry)
        .places()?;
    let models = build_all(&[0, 1], |&v| shrub_model(v))?;
    for (shrub, places) in models.iter().zip(Scatter::split(&shrubs, 2, 2)) {
        world.spawn_lod_at(shrub.lod_levels(), &places)?;
    }

    let rocks = Scatter::new(3)
        .disc(0.0, 0.0, 400.0)
        .spacing(16.0)
        .density(0.35)
        .scale_range(0.5, 2.2)
        .on_ground(|x, z| ground.height_at(x, z) - 0.3)
        .only_where(|x, z| !ground.sample(x, z).water)
        .places()?;
    let models = build_all(&[0, 1, 2, 3], |&k: &usize| {
        rock_model(RockShape::ALL[k], k as u32)
    })?;
    for (rock, places) in models.iter().zip(Scatter::split(&rocks, 4, 3)) {
        world.spawn_lod_at(rock.lod_levels_scaled(1.6), &places)?;
    }
    eprintln!(
        "{} trees, {} shrubs, {} rocks",
        trees.len(),
        shrubs.len(),
        rocks.len()
    );
    Ok(())
}

fn main() -> EngineResult<()> {
    let rules = demo_terrain_rules();
    let ground = HeightField::new(rules.clone());
    let quality = setting("ENGINE_QUALITY")
        .and_then(|q| Quality::from_name(&q))
        .unwrap_or(Quality::High);
    let target_fps = setting("ENGINE_TARGET_FPS").and_then(|f| f.parse::<f32>().ok());
    let hour = setting("ENGINE_SKY_HOUR")
        .and_then(|h| h.parse::<f32>().ok())
        .unwrap_or(16.0);
    let grass = setting("ENGINE_GRASS").and_then(|g| g.parse::<f32>().ok());

    let start = Vec3::new(-30.0, 0.0, -60.0);
    let mut camera = FlyCamera::new(
        start + Vec3::Y * (ground.height_at(start.x, start.z) + 9.0),
        35.0,
        -8.0,
    )
    .with_speed(12.0);

    Engine::run("starter", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_quality(quality);
            if let Some(grass) = grass {
                world.set_grass_scale(grass)?;
            }
            world.set_target_fps(target_fps)?;
            world.set_view_distance(1500.0)?;
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            world.set_proc_terrain(
                ProcTerrain::gpu_clipmap(rules.clone(), ClipmapConfig::default())
                    .with_focus(camera.position),
            );
            populate(world, &ground, start)?;
        }
        camera.update_above(world, frame, |x, z| ground.height_at(x, z));
        world.set_proc_focus(camera.position);
        Ok(())
    })
}
