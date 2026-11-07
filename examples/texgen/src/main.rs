//! Every procedural texture layer ([`engine::texgen`]) on a ground tile, with
//! a boulder wearing it, under the physical sky.
//!
//! The layers are projected triplanar: no UVs, and the texture moves with
//! the mesh. `ENGINE_TEXGEN_DUMP=dir` also writes each layer's albedo,
//! normal and occlusion-roughness maps as PNGs.
use engine::prelude::*;

fn main() -> EngineResult<()> {
    if let Some(dir) = std::env::var_os("ENGINE_TEXGEN_DUMP") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        for layer in NaturalLayer::ALL {
            let l = natural_layer(*layer);
            let name = layer.name().replace(' ', "_");
            for (kind, rgba) in [
                ("albedo", &l.albedo),
                ("normal", &l.normal),
                ("orm", &l.orm),
            ] {
                let path = dir.join(format!("{name}_{kind}.png"));
                engine::texture::save_rgba8_png(&path, l.size, l.size, rgba)?;
            }
        }
        eprintln!(
            "wrote {} layers to {}",
            NaturalLayer::ALL.len(),
            dir.display()
        );
    }
    Engine::run("texgen", move |world, frame| {
        if frame.first {
            world.spawn(Shape::box_at(
                (0.0, -0.05, 0.0),
                (400.0, 0.1, 400.0),
                rgb(70, 82, 60),
            )?);
            for (i, layer) in NaturalLayer::ALL.iter().enumerate() {
                let (col, row) = ((i % 7) as f32, (i / 7) as f32);
                let (x, z) = ((col - 3.0) * 4.0, 10.0 + row * 5.0);
                let mut tile = Mesh::box_at((x, 0.15, z), (3.6, 0.3, 3.6), rgb(255, 255, 255))?;
                natural_layer(*layer).apply_triplanar(&mut tile)?;
                world.spawn(tile);
                let rock = rock_model_with(RockShape::Boulder, i as u32, Some(*layer))?;
                let id = world.spawn_lod(rock.lod_levels_scaled(0.5))?;
                world.set_lod_places(
                    id,
                    &[Place::at(x + 1.0, 0.3, z + 0.9)?
                        .yaw_deg(i as f32 * 47.0)?
                        .scale(0.5)?],
                )?;
            }
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
        }
        world.look_first_person(Vec3::new(0.0, 7.5, 1.0), 0.0, -32.0);
        Ok(())
    })
}
