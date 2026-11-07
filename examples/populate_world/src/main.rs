//! Populate a world with a glTF prop (Quaternius-style) + a few rocks.
use engine::prelude::*;
use std::path::PathBuf;

fn fallback_tree() -> Mesh {
    let mut m = Mesh::new();
    m.add_box((0.0, 0.5, 0.0), (0.28, 1.0, 0.28), rgb(115, 71, 31))
        .unwrap();
    m.add_box((0.0, 1.45, 0.0), (1.2, 1.1, 1.2), rgb(51, 140, 56))
        .unwrap();
    m
}

fn main() -> EngineResult<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let asset = root.join("assets/lowpoly_tree.gltf");

    let tree = match Model::load_with(&asset, &root, &EngineLimits::default()) {
        Ok(mesh) => {
            eprintln!("loaded {}", asset.display());
            mesh
        }
        Err(err) => {
            eprintln!("glTF load failed ({err}); using procedural tree");
            fallback_tree()
        }
    };

    let mut yaw = 20.0_f32;

    Engine::run("populate_world", move |world, frame| {
        if frame.first {
            world.spawn(
                Shape::box_at((0.0, -0.05, 0.0), (60.0, 0.1, 60.0), rgb(84, 133, 71)).unwrap(),
            );

            let tree_positions = scatter_on_xz(
                99,
                (-22.0, 0.0, -22.0).into(),
                (22.0, 0.0, 22.0).into(),
                4.5,
                0.45,
                0.0,
            );
            let places = scatter_places(&tree_positions, 1.2, |i| i as f32 * 37.0);
            world.spawn_many(tree.clone(), places).expect("trees");

            for (i, p) in scatter_on_xz(
                11,
                (-16.0, 0.0, -16.0).into(),
                (16.0, 0.0, 16.0).into(),
                10.0,
                0.35,
                0.0,
            )
            .into_iter()
            .enumerate()
            {
                let rock = Shape::box_at(
                    Vec3::ZERO,
                    Vec3::splat(0.7 + i as f32 * 0.12),
                    rgb(128, 128, 122),
                )
                .unwrap();
                world.place(rock, Place::new(p.x, 0.35, p.z)).expect("rock");
            }

            world.set_sun((0.5, 1.0, 0.25), 0.26);
        }

        yaw += frame.dt * 12.0;
        let yaw = if std::env::var_os("ENGINE_SCREENSHOT").is_some() {
            35.0
        } else {
            yaw
        };
        world.look_orbit((0.0, 2.0, 0.0), 38.0, yaw, 28.0);
        Ok(())
    })
}
