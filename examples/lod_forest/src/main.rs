//! Per-instance LOD with a dithered cross-fade: about 12 000 trees over 1.2 km.
//!
//! Each tree draws its detailed mesh up close, a simpler one further out and
//! the coarsest at range, fading between them (and out past the last level)
//! instead of popping. The camera flies a loop low over the forest.
//!
//! `ENGINE_LOD=off` draws every tree at full detail for comparison;
//! `ENGINE_LOD_BIAS=0.5` halves every LOD distance.
use engine::prelude::*;

/// Six individuals of five species, so the forest does not look cloned.
const KINDS: [(TreeSpecies, u32); 6] = [
    (TreeSpecies::Oak, 0),
    (TreeSpecies::Pine, 1),
    (TreeSpecies::Birch, 2),
    (TreeSpecies::Linden, 3),
    (TreeSpecies::Oak, 4),
    (TreeSpecies::Maple, 5),
];

fn main() -> EngineResult<()> {
    let lod_on = std::env::var("ENGINE_LOD").map_or(true, |v| v != "off");
    let bias: f32 = std::env::var("ENGINE_LOD_BIAS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let places = Scatter::new(7)
        .rect(-600.0, -600.0, 600.0, 600.0)
        .spacing(11.0)
        .jitter(0.8)
        .scale_range(0.85, 1.15)
        .places()?;
    eprintln!(
        "{} trees, LOD {}",
        places.len(),
        if lod_on { "on" } else { "off" }
    );
    let models = build_all(&KINDS, |&(species, variant)| tree_model(species, variant))?;
    let screenshot = std::env::var_os("ENGINE_SCREENSHOT").is_some();

    let mut models = Some(models);
    Engine::run("lod_forest", move |world, frame| {
        world.set_vsync(false);
        if let Some(models) = models.take() {
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.5)));
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (1400.0, 0.1, 1400.0),
                rgb(84, 120, 62),
            )?);
            for (tree, mine) in models.iter().zip(Scatter::split(&places, KINDS.len(), 1)) {
                if lod_on {
                    world.spawn_lod_at(tree.lod_levels(), &mine)?;
                } else {
                    // One level that never switches: full detail everywhere.
                    let full = vec![LodLevel::new(tree.near.clone(), 5000.0)];
                    world.spawn_lod_at(full, &mine)?;
                }
            }
            world.set_lod_bias(bias)?;
            world.set_view_distance(1000.0)?;
        }
        let t = if screenshot { 3.0 } else { frame.time };
        let a = t * 0.05;
        let eye = Vec3::new(a.cos() * 250.0, 45.0, a.sin() * 250.0);
        let ahead = Vec3::new((a + 0.2).cos() * 250.0, 9.0, (a + 0.2).sin() * 250.0) - eye;
        world.look_first_person(eye, ahead.x.atan2(ahead.z).to_degrees(), -10.0);
        Ok(())
    })
}
