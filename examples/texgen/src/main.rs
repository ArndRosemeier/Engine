//! The procedural texture layers ([`engine::texgen`]) under the physical sky.
//!
//! `ENGINE_TEXGEN_FAMILY` picks a family: `natural` (default: ground tiles,
//! each with a boulder wearing it), `ground`, `facade` (standing walls),
//! `roof` (pitched panels) or `medieval`. The layers are projected
//! triplanar: no UVs, and the texture moves with the mesh.
//! `ENGINE_TEXGEN_DUMP=dir` writes every layer's albedo, normal and
//! occlusion-roughness maps as PNGs.
use engine::prelude::*;
use engine::texgen::prepare;

fn main() -> EngineResult<()> {
    let family = std::env::var("ENGINE_TEXGEN_FAMILY").unwrap_or_else(|_| "natural".into());
    let layers: Vec<Layer> = match family.as_str() {
        "ground" => GroundLayer::ALL.iter().map(|l| (*l).into()).collect(),
        "facade" => FacadeLayer::ALL.iter().map(|l| (*l).into()).collect(),
        "roof" => RoofLayer::ALL.iter().map(|l| (*l).into()).collect(),
        "medieval" => MedievalLayer::ALL.iter().map(|l| (*l).into()).collect(),
        _ => NaturalLayer::ALL.iter().map(|l| (*l).into()).collect(),
    };
    if let Some(dir) = std::env::var_os("ENGINE_TEXGEN_DUMP") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        let all = Layer::all();
        prepare(&all);
        for l in &all {
            let t = texture_layer(*l);
            let name = format!("{:?}", l).replace(['(', ')'], "_");
            for (kind, rgba) in [
                ("albedo", &t.albedo),
                ("normal", &t.normal),
                ("orm", &t.orm),
            ] {
                let path = dir.join(format!("{name}{kind}.png"));
                engine::texture::save_rgba8_png(&path, t.size, t.size, rgba)?;
            }
        }
        eprintln!("wrote {} layers to {}", all.len(), dir.display());
    }
    let t0 = std::time::Instant::now();
    prepare(&layers);
    eprintln!(
        "generated {} {family} layers in {:.2?}",
        layers.len(),
        t0.elapsed()
    );
    let walls = family == "facade" || family == "medieval";
    let roofs = family == "roof";
    Engine::run("texgen", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (400.0, 0.1, 400.0),
                rgb(70, 82, 60),
            )?);
            for (i, l) in layers.iter().enumerate() {
                let (col, row) = ((i % 8) as f32, (i / 8) as f32);
                let (x, z) = ((col - 3.5) * 4.2, 10.0 + row * 6.0);
                let tex = texture_layer(*l);
                let mut mesh = if walls {
                    Mesh::box_at((x, 1.8, z), (3.6, 3.6, 0.3), rgb(255, 255, 255))?
                } else {
                    Mesh::box_at((0.0, 0.0, 0.0), (3.6, 0.2, 3.6), rgb(255, 255, 255))?
                };
                tex.apply_triplanar(&mut mesh)?;
                if walls {
                    world.spawn(mesh);
                } else {
                    // Flat tiles, or roof panels pitched 35 degrees up toward the back.
                    let mut place = Place::at(x, if roofs { 1.2 } else { 0.1 }, z)?;
                    if roofs {
                        place = place.pitch_deg(-35.0)?;
                    }
                    let id = world.spawn_lod(vec![LodLevel::new(mesh, 1000.0)])?;
                    world.set_lod_places(id, &[place])?;
                }
                if family == "natural" {
                    if let Layer::NaturalLayer(n) = l {
                        let rock = rock_model_with(RockShape::Boulder, i as u32, Some(*n))?;
                        let id = world.spawn_lod(rock.lod_levels_scaled(0.5))?;
                        world.set_lod_places(
                            id,
                            &[Place::at(x + 1.0, 0.2, z + 0.9)?
                                .yaw_deg(i as f32 * 47.0)?
                                .scale(0.5)?],
                        )?;
                    }
                }
            }
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
        }
        if walls {
            world.look_first_person(Vec3::new(0.0, 3.0, -6.0), 0.0, -8.0);
        } else {
            world.look_first_person(Vec3::new(0.0, 8.0, 1.0), 0.0, -32.0);
        }
        Ok(())
    })
}
