//! Reusable surface-material showcase: stone, wet stone, calcite, metal,
//! and emissive mineral surfaces through the ordinary mesh renderer.
use engine::prelude::*;

fn block(
    at: (f32, f32, f32),
    size: (f32, f32, f32),
    color: Color,
    material: SurfaceMaterial,
) -> Mesh {
    let mut mesh = Mesh::new();
    mesh.add_box(at, size, color).expect("showcase block");
    mesh.set_surface_material(material);
    mesh
}

fn main() {
    Engine::run("material_showcase", move |world, frame| {
        if frame.first {
            world.set_clear_color(rgb(10, 12, 18));
            world.set_sun((-0.4, -1.0, -0.25), 0.65);
            world.spawn(block(
                (0.0, -0.3, 0.0),
                (18.0, 0.5, 12.0),
                rgb(118, 106, 94),
                SurfaceMaterial::STONE,
            ));
            world.spawn(block(
                (-5.0, 1.0, 0.0),
                (2.8, 2.8, 2.8),
                rgb(132, 122, 108),
                SurfaceMaterial::STONE.with_variation(101.0, 4.0),
            ));
            world.spawn(block(
                (-1.5, 1.0, 0.0),
                (2.8, 2.8, 2.8),
                rgb(132, 122, 108),
                SurfaceMaterial::STONE.with_variation(203.0, 5.5),
            ));
            world.spawn(block(
                (1.5, 1.0, 0.0),
                (2.8, 2.8, 2.8),
                rgb(104, 116, 119),
                SurfaceMaterial::WET_STONE.with_seed(307.0),
            ));
            world.spawn(block(
                (4.8, 1.0, 0.0),
                (2.8, 2.8, 2.8),
                rgb(205, 198, 177),
                SurfaceMaterial::CALCITE,
            ));
            world.spawn(block(
                (8.1, 1.0, 0.0),
                (2.8, 2.8, 2.8),
                rgb(145, 150, 158),
                SurfaceMaterial::METAL,
            ));
            world.spawn(block(
                (2.0, 1.0, -3.5),
                (2.8, 2.8, 2.8),
                rgb(80, 145, 112),
                SurfaceMaterial::GLOWING,
            ));
            world.spawn(block(
                (-5.0, 1.0, 7.0),
                (2.8, 2.8, 2.8),
                rgb(105, 72, 42),
                SurfaceMaterial::DIRT
                    .with_orientation([0.0, 1.0, 0.0])
                    .with_seed(907.0),
            ));
            world.spawn(block(
                (-1.5, 1.0, 7.0),
                (2.8, 2.8, 2.8),
                rgb(64, 123, 51),
                SurfaceMaterial::GRASS
                    .with_orientation([0.0, 1.0, 0.0])
                    .with_seed(1013.0),
            ));
            world.spawn(block(
                (2.0, 1.0, 7.0),
                (2.8, 2.8, 2.8),
                rgb(64, 123, 51),
                SurfaceMaterial::GRASS
                    .with_orientation([1.0, 0.0, 0.0])
                    .with_seed(1109.0),
            ));
            world.spawn(block(
                (5.0, 1.0, 3.5),
                (2.8, 2.8, 2.8),
                rgb(194, 157, 93),
                SurfaceMaterial::SAND
                    .with_orientation([0.0, 1.0, 0.0])
                    .with_seed(701.0),
            ));
            world.spawn(block(
                (8.2, 1.0, 3.5),
                (2.8, 2.8, 2.8),
                rgb(194, 157, 93),
                SurfaceMaterial::SAND
                    .with_orientation([1.0, 0.0, 0.0])
                    .with_seed(809.0),
            ));
            world.spawn(block(
                (-5.0, 1.0, 3.5),
                (2.8, 2.8, 2.8),
                rgb(142, 91, 48),
                SurfaceMaterial::WOOD
                    .with_orientation([1.0, 0.0, 0.0])
                    .with_seed(401.0),
            ));
            world.spawn(block(
                (-1.5, 1.0, 3.5),
                (2.8, 2.8, 2.8),
                rgb(142, 91, 48),
                SurfaceMaterial::WOOD
                    .with_orientation([0.0, 1.0, 0.0])
                    .with_seed(503.0),
            ));
            world.spawn(block(
                (2.0, 1.0, 3.5),
                (2.8, 2.8, 2.8),
                rgb(142, 91, 48),
                SurfaceMaterial::WOOD
                    .with_orientation([0.0, 0.0, 1.0])
                    .with_seed(607.0),
            ));
            world.spawn(block(
                (5.0, 1.0, 7.0),
                (2.8, 2.8, 2.8),
                rgb(105, 112, 122),
                SurfaceMaterial::SNOW.with_seed(1201.0),
            ));
            world.spawn(block(
                (8.2, 1.0, 7.0),
                (2.8, 2.8, 2.8),
                rgb(105, 112, 122),
                SurfaceMaterial::SNOW
                    .with_orientation([1.0, 0.25, 0.0])
                    .with_seed(1297.0),
            ));
            world.set_torch(Some(TorchLight::lantern()));
            world.mark_ready();
        }
        if std::env::var_os("ENGINE_SCREENSHOT_WAIT").is_some() && frame.first {
            world.queue_screenshot(std::env::var("ENGINE_SCREENSHOT").expect("ENGINE_SCREENSHOT"));
            world.request_exit();
        }
        world.look_orbit((0.0, 1.0, 0.0), 15.0, frame.time * 8.0, 22.0);
    });
}
