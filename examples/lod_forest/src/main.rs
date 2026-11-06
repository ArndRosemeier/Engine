//! Per-instance LOD with a dithered cross-fade: about 12 000 trees over 1.2 km.
//!
//! Each tree draws its detailed mesh up close, a simpler one further out and
//! the coarsest at range, fading between them (and out past 900 m) instead
//! of popping. The camera flies a loop low over the forest.
//!
//! `ENGINE_LOD=off` draws every tree at full detail for comparison;
//! `ENGINE_LOD_BIAS=0.5` halves every LOD distance.
use engine::prelude::*;
use engine::tree::{TreeAsset, TreeLod, TreeSettings};

fn tree(seed: u32, lod: TreeLod) -> Mesh {
    let settings = if seed % 3 == 0 {
        TreeSettings::conifer(seed)
    } else {
        TreeSettings::broadleaf(seed)
    };
    TreeAsset::generate_lod(Vec3::ZERO, settings.with_lod(lod), lod)
        .mesh()
        .expect("tree mesh")
}

fn main() -> EngineResult<()> {
    let lod_on = std::env::var("ENGINE_LOD").map_or(true, |v| v != "off");
    let bias: f32 = std::env::var("ENGINE_LOD_BIAS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let positions = scatter_on_xz(
        7,
        (-600.0, 0.0, -600.0).into(),
        (600.0, 0.0, 600.0).into(),
        11.0,
        0.8,
        0.0,
    );
    let places = scatter_places(&positions, 1.0, |i| i as f32 * 47.0);
    eprintln!(
        "{} trees, LOD {}",
        places.len(),
        if lod_on { "on" } else { "off" }
    );
    let screenshot = std::env::var_os("ENGINE_SCREENSHOT").is_some();

    Engine::run("lod_forest", move |world, frame| {
        world.set_vsync(false);
        if frame.first {
            world.set_clear_color(rgb(160, 190, 220));
            world.set_sun((0.45, 0.8, 0.3), 0.28);
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (1400.0, 0.1, 1400.0),
                rgb(84, 120, 62),
            )?);
            // Two species, split over the scattered places.
            for (species, seed) in [(0usize, 3u32), (1, 4)] {
                let mine: Vec<Place> = places
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == species)
                    .map(|(_, p)| *p)
                    .collect();
                if lod_on {
                    let id = world.spawn_lod(vec![
                        LodLevel::new(tree(seed, TreeLod::Hero), 40.0),
                        LodLevel::new(tree(seed, TreeLod::Mid), 160.0),
                        LodLevel::new(tree(seed, TreeLod::Far), 900.0),
                    ])?;
                    world.set_lod_places(id, &mine)?;
                } else {
                    world.spawn_many(tree(seed, TreeLod::Hero), mine)?;
                }
            }
            world.set_lod_bias(bias)?;
            world.set_view_distance(1000.0)?;
        }
        let t = if screenshot { 3.0 } else { frame.time };
        let a = t * 0.05;
        let eye = Vec3::new(a.cos() * 250.0, 70.0, a.sin() * 250.0);
        let ahead = Vec3::new((a + 0.2).cos() * 250.0, 9.0, (a + 0.2).sin() * 250.0) - eye;
        world.look_first_person(eye, ahead.x.atan2(ahead.z).to_degrees(), -14.0);
        Ok(())
    })
}
