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

fn leaf_cluster(at: (f32, f32, f32), scale: f32, seed: f32) -> Mesh {
    let mut mesh = Mesh::new();
    let (x, y, z) = at;
    // Three crossed cards keep the cluster inexpensive. The shader applies a
    // fixed analytic leaf mask; the cards remain opaque so overlapping foliage
    // does not depend on transparent draw order.
    let cards = [
        ([x - scale, y, z], [x + scale, y + scale * 2.0, z]),
        ([x, y, z - scale], [x, y + scale * 2.0, z + scale]),
        (
            [x - scale * 0.7, y + scale * 0.45, z - scale * 0.7],
            [x + scale * 0.7, y + scale * 1.8, z + scale * 0.7],
        ),
    ];
    for (a, b) in cards {
        let ids = [
            mesh.add_point(a).expect("leaf card point"),
            mesh.add_point([b[0], a[1], b[2]]).expect("leaf card point"),
            mesh.add_point(b).expect("leaf card point"),
            mesh.add_point([a[0], b[1], a[2]]).expect("leaf card point"),
        ];
        for (id, uv) in ids
            .into_iter()
            .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        {
            mesh.set_point_uv(id, uv).expect("leaf card uv");
            mesh.set_point_color(id, Color::rgb(255, 255, 255))
                .expect("leaf card alpha");
        }
        mesh.add_face(&ids).expect("leaf card face");
        mesh.add_face(&[ids[3], ids[2], ids[1], ids[0]])
            .expect("leaf card backface");
    }
    mesh.set_surface_material(SurfaceMaterial::FOLIAGE.with_seed(seed));
    mesh
}

fn needle_cluster(at: (f32, f32, f32), scale: f32, seed: f32) -> Mesh {
    let mut mesh = Mesh::new();
    let (x, y, z) = at;
    // Fixed diamond needles: no procedural silhouette or alpha is involved.
    let needles = [
        (0.0, 0.1, 0.0, 0.18, 1.1, [1.0, 0.0, 0.0]),
        (-0.42, 0.48, 0.12, 0.14, 0.88, [0.82, 0.0, 0.57]),
        (0.4, 0.62, -0.08, 0.15, 0.96, [0.86, 0.0, -0.5]),
        (-0.22, 0.88, -0.2, 0.12, 0.78, [0.55, 0.0, 0.84]),
        (0.24, 1.12, 0.08, 0.13, 0.82, [-0.52, 0.0, 0.86]),
        (-0.62, 0.84, 0.0, 0.10, 0.66, [0.95, 0.0, 0.2]),
        (0.6, 0.94, 0.16, 0.11, 0.7, [-0.9, 0.0, 0.25]),
        (-0.1, 1.48, -0.02, 0.10, 0.62, [0.35, 0.0, 0.94]),
        (0.1, 0.36, -0.46, 0.14, 0.9, [0.72, 0.0, 0.7]),
        (-0.3, 0.76, -0.4, 0.11, 0.7, [0.98, 0.0, -0.12]),
        (0.34, 1.02, -0.34, 0.11, 0.72, [-0.82, 0.0, 0.57]),
    ];
    for (ox, oy, oz, width, height, axis) in needles {
        let cx = x + ox * scale;
        let cy = y + oy * scale;
        let cz = z + oz * scale;
        let w = width * scale;
        let h = height * scale;
        let tip = [cx + axis[0] * w, cy + h, cz + axis[2] * w];
        let left = [cx - axis[2] * w, cy, cz + axis[0] * w];
        let right = [cx + axis[2] * w, cy, cz - axis[0] * w];
        let base = [cx, cy - h * 0.18, cz];
        let ids = [
            mesh.add_point(left).expect("needle point"),
            mesh.add_point(base).expect("needle point"),
            mesh.add_point(right).expect("needle point"),
            mesh.add_point(tip).expect("needle point"),
        ];
        for (id, uv) in ids
            .into_iter()
            .zip([[0.0, 0.5], [0.5, 0.0], [1.0, 0.5], [0.5, 1.0]])
        {
            mesh.set_point_uv(id, uv).expect("needle uv");
        }
        mesh.add_face(&[ids[0], ids[1], ids[3]])
            .expect("needle face");
        mesh.add_face(&[ids[1], ids[2], ids[3]])
            .expect("needle face");
        mesh.add_face(&[ids[3], ids[1], ids[0]])
            .expect("needle backface");
        mesh.add_face(&[ids[3], ids[2], ids[1]])
            .expect("needle backface");
    }
    mesh.set_surface_material(SurfaceMaterial::NEEDLED_FOLIAGE.with_seed(seed));
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
            world.spawn(leaf_cluster((-5.0, 1.0, 11.0), 2.2, 1401.0));
            world.spawn(leaf_cluster((1.0, 1.0, 11.0), 2.2, 1511.0));
            world.spawn(leaf_cluster((7.0, 1.0, 11.0), 2.2, 1621.0));
            world.spawn(needle_cluster((-5.0, 1.0, 16.5), 2.3, 1703.0));
            world.spawn(needle_cluster((1.0, 1.0, 16.5), 2.3, 1811.0));
            world.spawn(needle_cluster((7.0, 1.0, 16.5), 2.3, 1933.0));
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
