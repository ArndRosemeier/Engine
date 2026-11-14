//! The vegetation port: fourteen tree species and shrubs, swaying in the wind.
//!
//! A row of every species stands in front of a mixed wood of about 3000
//! trees (per-instance LOD: full detail, mid, far canopy hulls). The camera
//! drifts along the row; rocks of every shape lie in front and in the wood.
//!
//! `ENGINE_WIND=1.5` blows harder (0 calm .. 2 storm), `ENGINE_SEASON=0.8`
//! turns the leaves, `ENGINE_SKY_HOUR=17.5` sets the time of day,
//! `ENGINE_ROCKS_CLOSE=1` looks at the rocks up close, `ENGINE_VEG_NEW=1` at
//! the newest species and `ENGINE_VEG_NEW=2` at the trees in bloom.
use engine::prelude::*;

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> EngineResult<()> {
    let wind = env("ENGINE_WIND", 0.6);
    let season = env("ENGINE_SEASON", 0.0);
    let hour = env("ENGINE_SKY_HOUR", 15.5);
    let screenshot = std::env::var_os("ENGINE_SCREENSHOT").is_some();

    Engine::run("vegetation", move |world, frame| {
        world.set_vsync(false);
        if frame.first {
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (1600.0, 0.1, 1600.0),
                rgb(88, 116, 60),
            )?);
            // The row: every species twice (two variants), then shrubs.
            for (i, species) in TreeSpecies::ALL.iter().enumerate() {
                for v in 0..2u32 {
                    let model = tree_model(*species, v)?;
                    let x = (i as f32 - 4.5) * 16.0 + v as f32 * 7.0;
                    let z = 30.0 + v as f32 * 9.0;
                    let id = world.spawn_lod(model.lod_levels())?;
                    world.set_lod_places(
                        id,
                        &[Place::at(x, 0.0, z)?.yaw_deg(i as f32 * 41.0 + v as f32 * 97.0)?],
                    )?;
                }
            }
            // In bloom: the same species in colour, near the row's right end.
            let blooms = [
                (TreeSpecies::Acacia, rgb(150, 90, 230), 1.0),
                (TreeSpecies::Acacia, rgb(255, 80, 40), 0.8),
                (TreeSpecies::Linden, rgb(255, 150, 190), 0.9),
                (TreeSpecies::Willow, rgb(255, 210, 60), 0.7),
                (TreeSpecies::Baobab, rgb(60, 220, 210), 1.0),
            ];
            for (i, (species, color, share)) in blooms.into_iter().enumerate() {
                let model = tree_model_in_bloom(species, 3, Bloom::new(color, share))?;
                let id = world.spawn_lod(model.lod_levels())?;
                world.set_lod_places(id, &[Place::at(160.0 + i as f32 * 15.0, 0.0, 34.0)?])?;
            }
            let shrubs = (0..24)
                .map(|i| {
                    Place::at((i as f32 - 12.0) * 6.5, 0.0, 18.0 + (i % 3) as f32 * 2.0)?
                        .yaw_deg(i as f32 * 53.0)
                })
                .collect::<EngineResult<Vec<Place>>>()?;
            for v in 0..2u32 {
                let mine: Vec<Place> = shrubs.iter().skip(v as usize).step_by(2).copied().collect();
                let id = world.spawn_lod(shrub_model(v)?.lod_levels())?;
                world.set_lod_places(id, &mine)?;
            }
            // Rocks: every shape twice in front, then strewn through the wood.
            for (i, shape) in RockShape::ALL.iter().enumerate() {
                for seed in 0..2u32 {
                    let rock = rock_model(*shape, seed)?;
                    let scale = 0.8 + seed as f32 * 0.5;
                    let id = world.spawn_lod(rock.lod_levels_scaled(scale))?;
                    let x = -58.0 + (i * 2 + seed as usize) as f32 * 4.5;
                    world.set_lod_places(
                        id,
                        &[Place::at(x, 0.0, 12.0 + seed as f32 * 2.5)?
                            .yaw_deg(i as f32 * 67.0 + seed as f32 * 131.0)?
                            .scale(scale)?],
                    )?;
                }
            }
            // Strewn rocks keep apart, so no two grow into each other (their
            // flattened feet would fight under the ground).
            let mut taken: Vec<(f32, f32, f32)> = Vec::new();
            for k in 0..8u32 {
                let mut rocks = Vec::new();
                for i in 0..60 {
                    let h = |j: u32| (i * 7919 + k * 104_729 + j * 15_485_863) % 1000;
                    let (x, z) = (h(1) as f32 - 500.0, 25.0 + h(2) as f32 * 0.55);
                    let scale = 0.5 + h(4) as f32 / 1000.0 * 1.6;
                    let r = scale * 1.3 * 1.2;
                    if taken
                        .iter()
                        .any(|&(tx, tz, tr)| (tx - x).hypot(tz - z) < tr + r)
                    {
                        continue;
                    }
                    taken.push((x, z, r));
                    rocks.push(
                        Place::at(x, 0.0, z)?
                            .yaw_deg(h(3) as f32 * 0.36)?
                            .scale(scale)?,
                    );
                }
                let rock = rock_model(RockShape::ALL[k as usize % 4], 10 + k)?;
                let id = world.spawn_lod(rock.lod_levels_scaled(1.3))?;
                world.set_lod_places(id, &rocks)?;
            }
            // The wood behind: a few individuals per species, scattered.
            let positions = scatter_on_xz(
                9,
                (-500.0, 0.0, 70.0).into(),
                (500.0, 0.0, 600.0).into(),
                12.0,
                0.9,
                0.0,
            );
            // Shake the scatter off its grid, and vary size a little.
            let places = positions
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let h = |k: u32| {
                        ((i as u32).wrapping_mul(2_654_435_761) ^ k.wrapping_mul(40_503)) % 1000
                    };
                    let jx = h(1) as f32 / 1000.0 - 0.5;
                    let jz = h(2) as f32 / 1000.0 - 0.5;
                    Place::at(p.x + jx * 9.0, 0.0, p.z + jz * 9.0)?
                        .yaw_deg(h(3) as f32 * 0.36)?
                        .scale(0.85 + h(4) as f32 / 1000.0 * 0.3)
                })
                .collect::<EngineResult<Vec<Place>>>()?;
            let wood = [
                TreeSpecies::Oak,
                TreeSpecies::Linden,
                TreeSpecies::Maple,
                TreeSpecies::Birch,
                TreeSpecies::Pine,
                TreeSpecies::Chestnut,
            ];
            let kinds = wood.len() * 2;
            for k in 0..kinds {
                let mine: Vec<Place> = places.iter().skip(k).step_by(kinds).copied().collect();
                let model = tree_model(wood[k % wood.len()], 10 + k as u32)?;
                let id = world.spawn_lod(model.lod_levels())?;
                world.set_lod_places(id, &mine)?;
            }
            eprintln!("{} trees in the wood", places.len());
            world.set_view_distance(900.0)?;
            world.set_wind(wind)?;
            world.set_season(season)?;
        }
        world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
        let t = if screenshot { 0.0 } else { frame.time };
        let close = std::env::var_os("ENGINE_VEG_CLOSE").is_some();
        let x = -70.0 + (t * 0.03).sin() * 60.0;
        if let Some(which) = std::env::var_os("ENGINE_VEG_NEW") {
            // The four newest species (1), or the trees in bloom (2).
            let x = if which == "2" { 190.0 } else { 114.0 };
            world.look_first_person(Vec3::new(x, 5.0, 2.0), 0.0, 4.0);
        } else if std::env::var_os("ENGINE_ROCKS_CLOSE").is_some() {
            world.look_first_person(Vec3::new(-44.0, 1.6, 6.5), 0.0, -12.0);
        } else if close {
            world.look_first_person(Vec3::new(-40.0, 3.0, 8.0), 20.0, 12.0);
        } else {
            world.look_first_person(Vec3::new(x, 4.0, -12.0), 8.0, 4.0);
        }
        Ok(())
    })
}
