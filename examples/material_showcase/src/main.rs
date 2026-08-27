//! Reusable surface-material showcase: stone, wet stone, calcite, metal,
//! and emissive mineral surfaces through the ordinary mesh renderer.
use engine::egui;
use engine::prelude::*;
use engine::{load_rgba8_png, TreeAsset, TreeKind, TreeSettings};

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

fn add_foliage_card(mesh: &mut Mesh, center: Vec3, u: Vec3, v: Vec3) {
    let ids = [
        mesh.add_point(center - u - v).expect("leaf card point"),
        mesh.add_point(center + u - v).expect("leaf card point"),
        mesh.add_point(center + u + v).expect("leaf card point"),
        mesh.add_point(center - u + v).expect("leaf card point"),
    ];
    for (id, uv) in ids
        .into_iter()
        .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    {
        mesh.set_point_uv(id, uv).expect("leaf card uv");
        mesh.set_point_color(id, Color::WHITE)
            .expect("leaf card color");
    }
    mesh.add_face(&ids).expect("leaf card face");
    mesh.add_face(&[ids[3], ids[2], ids[1], ids[0]])
        .expect("leaf card backface");
}

fn showcase_trunk(at: Vec3, broadleaf: bool) -> Mesh {
    let mut mesh = Mesh::new();
    let trunk = if broadleaf {
        Color::rgb(92, 58, 34)
    } else {
        Color::rgb(72, 50, 35)
    };
    mesh.add_box(
        at + Vec3::new(0.0, 2.2, 0.0),
        Vec3::new(0.58, 4.4, 0.58),
        trunk,
    )
    .expect("tree trunk");
    for (offset, size) in [
        (Vec3::new(0.0, 2.3, 0.0), Vec3::new(2.8, 0.22, 0.22)),
        (Vec3::new(0.0, 3.1, 0.0), Vec3::new(0.22, 0.20, 2.5)),
        (Vec3::new(0.0, 3.7, 0.0), Vec3::new(2.1, 0.18, 0.18)),
    ] {
        mesh.add_box(at + offset, size, trunk).expect("tree branch");
    }
    mesh.set_surface_material(SurfaceMaterial::WOOD);
    mesh
}

/// SpeedTree-style foliage half: branch-proximal lobes with independently tilted cutouts.
/// provide attachment points and a dense procedural leaf field provides detail.
/// Dense 3D foliage preview with many small independently oriented cards.
/// Hero foliage prototype: irregular low-poly leaf clusters distributed through
/// overlapping canopy lobes. No cluster is a rectangle in silhouette.
fn showcase_speedtree(at: Vec3, seed: f32, kind: TreeKind) -> (Mesh, Mesh, usize, usize) {
    let tree = TreeAsset::generate(at, TreeSettings::new(kind, seed as u32));
    let texture_name = match kind {
        TreeKind::Pine | TreeKind::Spruce => "vegetation_fern_08.png",
        _ => "vegetation_leaf_maple_01.png",
    };
    let texture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/foliage_sources")
        .join(texture_name);
    let (width, height, rgba) = load_rgba8_png(&texture_path).expect("SpeedTree foliage texture");
    let mut foliage = tree.foliage().clone();
    foliage.set_surface_material(match kind {
        TreeKind::Pine | TreeKind::Spruce => SurfaceMaterial::NEEDLED_FOLIAGE.with_seed(seed),
        _ => SurfaceMaterial::FOLIAGE.with_seed(seed),
    });
    foliage
        .set_albedo_rgba(width, height, rgba)
        .expect("SpeedTree foliage albedo");
    let mut bark = tree.bark().clone();
    bark.set_surface_material(SurfaceMaterial::WOOD.with_seed(seed));
    assert!(tree.branch_count() > 20, "generated tree hierarchy");
    assert!(
        tree.foliage_cluster_count() > 10,
        "generated foliage clusters"
    );
    assert!(
        bark.build().triangle_count() > 100,
        "generated branch geometry"
    );
    (
        bark,
        foliage,
        tree.branch_count(),
        tree.foliage_cluster_count(),
    )
}
fn showcase_tree(at: Vec3, seed: f32, broadleaf: bool) -> Mesh {
    let mut mesh = Mesh::new();
    let foliage = if broadleaf {
        SurfaceMaterial::FOLIAGE.with_seed(seed)
    } else {
        SurfaceMaterial::NEEDLED_FOLIAGE.with_seed(seed)
    };
    let (levels, cards_per_level, leaf_w, leaf_h) = if broadleaf {
        (5, 7, 0.42, 0.62)
    } else {
        (6, 7, 0.30, 0.72)
    };
    for level in 0..levels {
        let t = level as f32 / (levels - 1) as f32;
        let y = 3.0 + t * 3.4;
        let radius = (2.15 - t * 1.10).max(0.52);
        for card in 0..cards_per_level {
            let phase = seed * 0.017 + level as f32 * 1.71 + card as f32 * 2.399;
            let angle =
                phase.sin() * 2.4 + card as f32 * std::f32::consts::TAU / cards_per_level as f32;
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let tangent = Vec3::new(-angle.sin(), 0.0, angle.cos());
            let center =
                at + radial * radius * (0.72 + phase.cos().abs() * 0.22) + Vec3::new(0.0, y, 0.0);
            let tilt = (phase * 1.37).sin() * 0.72 + (phase * 0.61).cos() * 0.22;
            let u = (tangent + Vec3::Y * tilt).normalize() * leaf_w;
            let v = (Vec3::Y + radial * tilt * 0.55).normalize() * leaf_h;
            add_foliage_card(&mut mesh, center, u, v);
        }
    }
    mesh.set_surface_material(foliage);
    mesh
}

fn main() -> EngineResult<()> {
    let mut position = Vec3::new(1.0, 4.4, -5.0);
    let mut yaw = 0.0_f32;
    let mut pitch = -4.0_f32;

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
            for (at, seed, broadleaf) in [
                (Vec3::new(-5.0, 1.0, 10.0), 1401.0, true),
                (Vec3::new(1.0, 1.0, 10.0), 1511.0, true),
                (Vec3::new(7.0, 1.0, 10.0), 1621.0, true),
                (Vec3::new(-5.0, 1.0, 15.5), 1703.0, false),
                (Vec3::new(1.0, 1.0, 15.5), 1811.0, false),
                (Vec3::new(7.0, 1.0, 15.5), 1933.0, false),
            ] {
                world.spawn(showcase_trunk(at, broadleaf));
                world.spawn(showcase_tree(at, seed, broadleaf));
            }
            for (at, seed, broadleaf) in [
                (Vec3::new(-5.0, 1.0, 21.0), 2201.0, true),
                (Vec3::new(1.0, 1.0, 21.0), 2311.0, true),
                (Vec3::new(7.0, 1.0, 21.0), 2421.0, true),
                (Vec3::new(-5.0, 1.0, 26.5), 2503.0, false),
                (Vec3::new(1.0, 1.0, 26.5), 2611.0, false),
                (Vec3::new(7.0, 1.0, 26.5), 2733.0, false),
            ] {
                let kind = if broadleaf {
                    TreeKind::Oak
                } else {
                    TreeKind::Pine
                };
                let (bark, foliage, branches, clusters) = showcase_speedtree(at, seed, kind);
                world.spawn(bark);
                world.spawn(foliage);
                assert!(branches > 20 && clusters > 10, "showcase tree structure");
            }
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
        if frame.input.mouse_clicked(MouseButton::Left) {
            world.set_pointer_lock(true);
        }
        if world.pointer_lock() {
            let mouse = frame.input.mouse_delta();
            yaw -= mouse.x * 0.12;
            pitch = (pitch - mouse.y * 0.12).clamp(-88.0, 88.0);
        }
        let move_dir = frame.input.move_dir_xz(yaw);
        let speed = if frame.input.down(Key::Shift) {
            18.0
        } else {
            6.0
        };
        position += move_dir * speed * frame.dt;
        position.y += frame.input.axis(Key::Ctrl, Key::Space) * speed * frame.dt;

        if std::env::var_os("ENGINE_SCREENSHOT_WAIT").is_some() && frame.first {
            world.queue_screenshot(std::env::var("ENGINE_SCREENSHOT").expect("ENGINE_SCREENSHOT"));
            world.request_exit();
        }
        world.look_first_person(position, yaw, pitch);

        egui::Window::new("Material Showcase")
            .anchor(egui::Align2::LEFT_TOP, [12.0, 12.0])
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label("Click to capture mouse");
                ui.label("WASD / arrows: move   Q/E: turn");
                ui.label("Mouse: look   Space/Ctrl: up/down   Shift: sprint");
                ui.label("Upper row: CONTROL - procedural cards");
                ui.label("Lower row: SPEEDTREE - connected branches and imported cutouts");
                ui.label("Leaf sources: maple broadleaf / fern conifer");
            });
        Ok(())
    })
}
