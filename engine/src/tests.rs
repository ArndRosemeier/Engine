use crate::advanced::Volume;
use crate::color::{rgb, Color};
use crate::error::EngineError;
use crate::limits::EngineLimits;
use crate::mesh::Mesh;
use crate::model::Model;
use crate::place::Place;
use crate::world::Frame;
use glam::Vec3;
use std::path::PathBuf;

#[test]
fn mesh_quad_builds_two_triangles() {
    let mut m = Mesh::new();
    let a = m.add_point(Vec3::ZERO).unwrap();
    let b = m.add_point(Vec3::X).unwrap();
    let c = m.add_point(Vec3::new(1.0, 1.0, 0.0)).unwrap();
    let d = m.add_point(Vec3::Y).unwrap();
    m.add_face(&[a, b, c, d]).unwrap();
    let built = m.build();
    assert_eq!(built.index_count(), 6);
    assert_eq!(built.vertex_count(), 6);
    assert!(built.normals.iter().all(|n| n.length() > 0.9));
}

#[test]
fn invalid_face_returns_error() {
    let mut m = Mesh::new();
    let a = m.add_point(Vec3::ZERO).unwrap();
    let err = m.add_face(&[a, a]).unwrap_err();
    assert!(matches!(err, EngineError::InvalidMesh(_)));
}

#[test]
fn box_normals_point_outward() {
    let mut m = Mesh::new();
    m.add_box(Vec3::ZERO, Vec3::ONE, Color::WHITE).unwrap();
    let built = m.build();
    for (p, n) in built.positions.iter().zip(built.normals.iter()) {
        assert!(
            p.dot(*n) > 0.0,
            "normal {n} at {p} should point outward from origin-centered box"
        );
    }
}

#[test]
fn volume_sphere_extracts_triangles() {
    let mut v = Volume::new(0.5);
    v.fill_sphere(Vec3::new(8.0, 8.0, 8.0), 3.0);
    let mesh = v.extract_all(rgb(255, 255, 255));
    assert!(mesh.index_count() > 0);
    assert_eq!(mesh.index_count() % 3, 0);
}

#[test]
fn volume_sphere_normals_point_outward() {
    let center = Vec3::new(8.0, 8.0, 8.0);
    let mut v = Volume::new(0.5);
    v.fill_sphere(center, 3.0);
    let mesh = v.extract_all(rgb(255, 255, 255));
    let mut outward = 0;
    let mut total = 0;
    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];
        let n = (b - a).cross(c - a);
        if n.length_squared() < 1e-10 {
            continue;
        }
        let n = n.normalize();
        let centroid = (a + b + c) / 3.0;
        if (centroid - center).normalize_or_zero().dot(n) > 0.0 {
            outward += 1;
        }
        total += 1;
    }
    assert!(total > 20);
    assert!(
        outward * 100 / total > 80,
        "expected mostly outward faces, got {outward}/{total}"
    );
}

#[test]
fn oversize_paint_hits_resource_limit() {
    let mut v = Volume::new(0.5);
    let limits = EngineLimits {
        max_volume_samples_per_paint: 100,
        ..EngineLimits::default()
    };
    let err = v
        .paint_fn_limited(
            Vec3::ZERO,
            Vec3::new(50.0, 50.0, 50.0),
            &limits,
            |_| 1.0,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::ResourceLimit(_)));
}

#[test]
fn gltf_path_escape_rejected() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/populate_world");
    // Try to escape upward past the allowed root via .. segments after resolve.
    // canonicalize of a file outside base should fail PathNotAllowed when we point
    // at the engine Cargo.toml while base is the example folder.
    let outside = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let err = Model::load_with(&outside, &base, &EngineLimits::default()).unwrap_err();
    assert!(
        matches!(err, EngineError::PathNotAllowed(_)),
        "expected PathNotAllowed, got {err:?}"
    );
}

#[test]
fn gltf_demo_asset_loads() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../examples/populate_world");
    let path = base.join("assets/lowpoly_tree.gltf");
    let mesh = Model::load_with(path, &base, &EngineLimits::default()).expect("demo glTF");
    assert!(mesh.point_count() > 0);
    assert!(mesh.face_count() > 0);
}

#[test]
fn place_builds_matrix() {
    let p = Place::new(1.0, 2.0, 3.0).with_yaw_deg(90.0).with_scale(2.0);
    let m = p.to_matrix();
    let t = m.transform_point3(Vec3::ZERO);
    assert!((t - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
}

#[test]
fn frame_first_flag_exists() {
    let f = Frame {
        dt: 0.016,
        time: 0.0,
        fps: 60.0,
        width: 1,
        height: 1,
        aspect: 1.0,
        first: true,
        input: crate::input::Input::new(),
    };
    assert!(f.first);
}

#[test]
fn height_terrain_has_some_water() {
    let t = crate::terrain::HeightTerrain::new(crate::terrain::TerrainRules {
        seed: 9,
        lake_threshold: 0.60,
        ..Default::default()
    });
    let mut water = 0;
    let mut total = 0;
    for z in -120..120 {
        for x in -120..120 {
            if t.sample(x as f32 * 2.0, z as f32 * 2.0).water {
                water += 1;
            }
            total += 1;
        }
    }
    assert!(water > 0, "expected some lake samples");
    assert!(water < total / 2, "lakes should be occasional, got {water}/{total}");
}

#[test]
fn lakes_carve_below_waterline() {
    let water_level = 5.5;
    let t = crate::terrain::HeightTerrain::new(crate::terrain::TerrainRules {
        seed: 19,
        lake_threshold: 0.60,
        water_level,
        ..Default::default()
    });
    let mut water_cells = 0;
    let mut shore_checks = 0;
    for z in -120..120 {
        for x in -120..120 {
            let px = x as f32 * 2.0;
            let pz = z as f32 * 2.0;
            let s = t.sample(px, pz);
            if !s.water {
                assert!(
                    s.height + 1e-3 >= water_level,
                    "dry surface must not undercut the waterline (mesa risk): {}",
                    s.height
                );
                assert!(
                    s.ground + 1e-3 >= water_level,
                    "dry ground must not undercut the waterline: {}",
                    s.ground
                );
                continue;
            }
            water_cells += 1;
            assert!(
                s.ground < water_level,
                "lake bed must sit below waterline (got ground={})",
                s.ground
            );
            assert!(
                (s.height - water_level).abs() < 1e-3,
                "walkable height on water must be water_level"
            );
            for (dx, dz) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
                let n = t.sample(px + dx, pz + dz);
                if !n.water {
                    assert!(
                        n.ground + 0.05 >= water_level,
                        "shore undercuts water: shore={} water={}",
                        n.ground,
                        water_level
                    );
                    shore_checks += 1;
                }
            }
        }
    }
    assert!(water_cells > 0, "expected some water");
    assert!(shore_checks > 0, "expected shore samples");
}

#[test]
fn camera_right_xz_matches_follow_view() {
    // look_at_rh side vector is forward × up; for yaw 0 (face +Z) that is −X.
    let right0 = crate::camera::Camera::right_xz(0.0);
    assert!(
        right0.dot(-Vec3::X) > 0.9,
        "yaw 0 follow view: screen-right is −X, got {right0}"
    );
    let f = crate::camera::Camera::facing_xz(35.0);
    let r = crate::camera::Camera::right_xz(35.0);
    let expected = Vec3::new(-f.z, 0.0, f.x);
    assert!(
        r.dot(expected) > 0.99,
        "strafe must match look_at_rh screen-right"
    );
}

#[test]
fn walk_height_matches_clipmap_triangle() {
    use crate::proc_terrain::{demo_terrain_rules, ClipmapConfig, HeightField};
    let field = HeightField::new(demo_terrain_rules());
    let cfg = ClipmapConfig {
        rings: 2,
        resolution: 32,
        cell_size: 1.0,
    };
    let focus = Vec3::new(10.0, 0.0, -4.0);
    let x = 12.3;
    let z = -2.7;
    let y = field.walk_height_on_clipmap(x, z, &cfg, focus);
    let cell = cfg.cell_size;
    let cx = (focus.x / cell).floor() * cell;
    let cz = (focus.z / cell).floor() * cell;
    let extent = cell * cfg.resolution as f32;
    let ox = cx - extent * 0.5;
    let oz = cz - extent * 0.5;
    let fx = (x - ox) / cell;
    let fz = (z - oz) / cell;
    let ix = fx.floor();
    let iz = fz.floor();
    let tx = fx - ix;
    let tz = fz - iz;
    let g = |di: f32, dj: f32| {
        field
            .sample(ox + (ix + di) * cell, oz + (iz + dj) * cell)
            .ground
    };
    let h00 = g(0.0, 0.0);
    let h10 = g(1.0, 0.0);
    let h01 = g(0.0, 1.0);
    let h11 = g(1.0, 1.0);
    let expect = if tz >= tx {
        h00 * (1.0 - tz) + h01 * (tz - tx) + h11 * tx
    } else {
        h00 * (1.0 - tx) + h10 * (tx - tz) + h11 * tz
    };
    if !field.sample(x, z).water {
        assert!(
            (y - expect).abs() < 1e-4,
            "walk height {y} != triangle {expect}"
        );
    }
}

#[test]
fn terrain_stream_budgets_main_thread_work() {
    use crate::terrain::{TerrainRules, TerrainStream};
    use crate::world::World;
    use std::time::{Duration, Instant};

    let rules = TerrainRules {
        chunk_cells: 48,
        cell_size: 1.0,
        ..TerrainRules::default()
    };
    // Tiny upload budget so one sync cannot push the whole ring.
    let mut stream = TerrainStream::new(rules, 2).with_budgets(8, 1);
    let mut world = World::new();
    let focus = Vec3::ZERO;

    let t0 = Instant::now();
    stream.sync(&mut world, focus);
    assert!(
        t0.elapsed() < Duration::from_millis(750),
        "first sync should not build the whole ring on the main thread"
    );
    // Focus chunk is forced in; at most one extra upload same frame.
    assert!(stream.loaded_count() <= 2);
    assert!(stream.loaded_count() >= 1);

    let start = Instant::now();
    while stream.loaded_count() < 25 && start.elapsed() < Duration::from_secs(10) {
        stream.sync(&mut world, focus);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        stream.loaded_count(),
        25,
        "ring should fill asynchronously (got {})",
        stream.loaded_count()
    );
}

#[test]
fn height_chunk_builds_quads() {
    let t = crate::terrain::HeightTerrain::new(crate::terrain::TerrainRules::default());
    let mesh = t.build_chunk(0, 0);
    let cells = t.rules().chunk_cells as usize;
    let ground_verts = (cells + 1) * (cells + 1);
    let ground_faces = cells * cells;
    assert!(mesh.point_count() >= ground_verts);
    assert!(mesh.face_count() >= ground_faces);
    let built = mesh.build();
    assert!(built.opaque_index_count > 0);
    assert!(
        built.opaque_index_count <= built.indices.len(),
        "transparent water should partition after opaque land"
    );
}

#[test]
fn color_rgb_bytes() {
    let c = rgb(255, 0, 128);
    assert!((c.r - 1.0).abs() < 1e-5);
    assert!(c.g.abs() < 1e-5);
    assert!((c.b - 128.0 / 255.0).abs() < 1e-5);
    assert!((c.a - 1.0).abs() < 1e-5);
}

#[test]
fn transparent_faces_draw_after_opaque() {
    use crate::color::rgba;
    let mut m = Mesh::new();
    let a = m.add_point((0.0, 0.0, 0.0)).unwrap();
    let b = m.add_point((1.0, 0.0, 0.0)).unwrap();
    let c = m.add_point((0.0, 0.0, 1.0)).unwrap();
    m.set_point_color(a, rgba(0, 0, 255, 128)).unwrap();
    m.set_point_color(b, rgba(0, 0, 255, 128)).unwrap();
    m.set_point_color(c, rgba(0, 0, 255, 128)).unwrap();
    m.add_triangle(a, b, c).unwrap();
    let d = m.add_point((0.0, 1.0, 0.0)).unwrap();
    let e = m.add_point((1.0, 1.0, 0.0)).unwrap();
    let f = m.add_point((0.0, 1.0, 1.0)).unwrap();
    m.set_point_color(d, rgb(255, 0, 0)).unwrap();
    m.set_point_color(e, rgb(255, 0, 0)).unwrap();
    m.set_point_color(f, rgb(255, 0, 0)).unwrap();
    m.add_triangle(d, e, f).unwrap();
    let built = m.build();
    assert_eq!(built.opaque_index_count, 3);
    assert_eq!(built.indices.len(), 6);
    assert!(built.colors[0].w > 0.99);
    assert!(built.colors[3].w < 0.99);
}
