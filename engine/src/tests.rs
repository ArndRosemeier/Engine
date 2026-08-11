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
        width: 1,
        height: 1,
        aspect: 1.0,
        first: true,
    };
    assert!(f.first);
}

#[test]
fn color_rgb_bytes() {
    let c = rgb(255, 0, 128);
    assert!((c.r - 1.0).abs() < 1e-5);
    assert!(c.g.abs() < 1e-5);
    assert!((c.b - 128.0 / 255.0).abs() < 1e-5);
}
