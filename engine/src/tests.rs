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

/// An opaque one-metre floor patch, for tests that only need some geometry.
fn unit_quad() -> Mesh {
    let mut m = Mesh::new();
    let a = m.add_point((0.0, 0.0, 0.0)).unwrap();
    let b = m.add_point((1.0, 0.0, 0.0)).unwrap();
    let c = m.add_point((1.0, 0.0, 1.0)).unwrap();
    let d = m.add_point((0.0, 0.0, 1.0)).unwrap();
    m.add_face(&[a, b, c, d]).unwrap();
    m
}

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
fn mesh_build_smooth_shares_normals_across_ridge() {
    // Two quads meeting at a ridge — flat build has discontinuous normals;
    // smooth build averages at the shared authoring points.
    let mut m = Mesh::new();
    let p00 = m.add_point(Vec3::new(0.0, 0.0, 0.0)).unwrap();
    let p10 = m.add_point(Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let p20 = m.add_point(Vec3::new(2.0, 0.0, 0.0)).unwrap();
    let p01 = m.add_point(Vec3::new(0.0, 0.5, 1.0)).unwrap();
    let p11 = m.add_point(Vec3::new(1.0, 1.0, 1.0)).unwrap();
    let p21 = m.add_point(Vec3::new(2.0, 0.5, 1.0)).unwrap();
    m.add_quad(p00, p01, p11, p10).unwrap();
    m.add_quad(p10, p11, p21, p20).unwrap();
    let flat = m.build();
    let smooth = m.build_smooth();
    assert_eq!(
        smooth.vertex_count(),
        6,
        "smooth bake shares authoring points"
    );
    assert_eq!(smooth.index_count(), 12);
    assert!(
        smooth.vertex_count() < flat.vertex_count(),
        "indexed smooth must emit fewer GPU verts than the faceted bake"
    );
    // Ridge points p10/p11 are used by both quads; their averaged normal
    // must not match either flat face.
    let ridge = Vec3::new(1.0, 0.0, 0.0);
    let ridge_n = smooth
        .positions
        .iter()
        .zip(smooth.normals.iter())
        .find(|(p, _)| (*p - ridge).length() < 1e-5)
        .map(|(_, n)| *n)
        .expect("ridge vertex");
    let mut max_gap = 0.0_f32;
    for n in &flat.normals {
        max_gap = max_gap.max(1.0 - n.dot(ridge_n));
    }
    assert!(
        max_gap > 0.02,
        "smooth ridge normal should differ from flat faces (gap={max_gap})"
    );
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
    v.fill_sphere(Vec3::new(8.0, 20.0, 8.0), 3.0);
    let mesh = v.extract_all(rgb(255, 255, 255));
    assert!(mesh.index_count() > 0);
    assert_eq!(mesh.index_count() % 3, 0);
}

#[test]
fn volume_sphere_normals_point_outward() {
    let center = Vec3::new(8.0, 20.0, 8.0);
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
        .paint_fn_limited(Vec3::ZERO, Vec3::new(50.0, 50.0, 50.0), &limits, |_| 1.0)
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
fn glb_bin_space_fourcc_is_unknown_chunk_type() {
    let dir = std::env::temp_dir().join(format!("engine-glb-bin-space-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bin_space.glb");
    std::fs::write(&path, crate::model::test_glb_with_bin_space_fourcc()).unwrap();
    let err = Model::load_with(&path, &dir, &EngineLimits::default()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.to_ascii_lowercase().contains("unknown chunk type"),
        "expected gltf crate unknown chunk type, got {msg}"
    );
}

#[test]
fn blend_or_mask_without_real_alpha_loads_opaque() {
    for mode in ["BLEND", "MASK"] {
        let dir =
            std::env::temp_dir().join(format!("engine-static-{mode}-rgb-{}", std::process::id()));
        crate::model::write_minimal_static_gltf(&dir, mode, true);
        let mesh =
            Model::load_with(dir.join("model.gltf"), &dir, &EngineLimits::default()).unwrap();
        assert!(mesh.face_count() > 0, "{mode}");
    }
}

#[test]
fn place_builds_matrix() {
    let p = Place::new(1.0, 2.0, 3.0).with_yaw_deg(90.0).with_scale(2.0);
    let m = p.to_matrix();
    let t = m.transform_point3(Vec3::ZERO);
    assert!((t - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
}

#[test]
fn place_stretch_scales_axes_independently() {
    let p = Place::new(0.0, 0.0, 0.0).with_stretch(Vec3::new(2.0, 3.0, 4.0));
    let t = p.to_matrix().transform_point3(Vec3::ONE);
    assert!((t - Vec3::new(2.0, 3.0, 4.0)).length() < 1e-4);
}

#[test]
fn place_pitch_lays_an_opening_on_the_floor() {
    let up = Place::new(0.0, 0.0, 0.0)
        .with_pitch_deg(-90.0)
        .to_matrix()
        .transform_vector3(Vec3::Z);
    assert!(
        up.y > 0.9,
        "pitch −90 must face Mesh::opening +Z at +Y, got {up}"
    );
    let down = Place::new(0.0, 0.0, 0.0)
        .with_pitch_deg(90.0)
        .to_matrix()
        .transform_vector3(Vec3::Z);
    assert!(
        down.y < -0.9,
        "pitch +90 must face Mesh::opening +Z at −Y, got {down}"
    );
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
        ui: crate::ui::UiFrame::default(),
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
    assert!(
        water < total / 2,
        "lakes should be occasional, got {water}/{total}"
    );
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
fn first_person_looks_along_yaw_and_pitch() {
    use crate::camera::{Camera, MAX_PITCH_DEGREES};

    let level = Camera::first_person(Vec3::new(3.0, 2.0, -1.0), 0.0, 0.0);
    assert_eq!(level.eye(), Vec3::new(3.0, 2.0, -1.0));
    let forward = (level.target() - level.eye()).normalize();
    assert!(
        forward.dot(Vec3::Z) > 0.999,
        "yaw 0 must look down +Z, got {forward}"
    );

    let up = Camera::direction(90.0, 45.0);
    assert!(up.y > 0.7 && up.x > 0.7, "yaw 90 pitch 45 goes up and +X");

    // Straight up would collapse the view matrix onto the up axis.
    let steep = Camera::direction(0.0, 200.0);
    assert!(
        steep.y < MAX_PITCH_DEGREES.to_radians().sin() + 1e-6 && steep.y > 0.99,
        "pitch is clamped just short of vertical, got {steep}"
    );
    assert!(steep.is_finite() && steep.length() > 0.99);
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
fn rebasing_moves_anchored_entities_but_not_their_global_anchor() {
    use crate::place::GlobalPlace;
    use crate::space::{GlobalPosition, GlobalXZ, RenderOrigin};
    use crate::world::World;

    let mut world = World::new();
    let anchor = GlobalPosition::at(2_500_000.0, 30.0, -1_250_000.0);
    let id = world
        .spawn_anchored(
            Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(200, 200, 200)).unwrap(),
            GlobalPlace::at(anchor).with_yaw_deg(35.0),
        )
        .unwrap();

    world
        .set_render_origin(RenderOrigin::snapped(anchor.horizontal(), 1_000.0).unwrap())
        .unwrap();
    let near = world.entity(id).unwrap().transform.w_axis.truncate();
    assert!(
        near.length() < 2_000.0,
        "anchored entity should sit near the rebased origin, got {near}"
    );

    // A second, far rebase must be derived from the anchor, not from `near`.
    world
        .set_render_origin(RenderOrigin::new(GlobalXZ::at(0.0, 0.0)))
        .unwrap();
    let far = world.entity(id).unwrap().transform.w_axis.truncate();
    assert!((far.x as f64 - anchor.x).abs() < 1.0);
    assert!((far.z as f64 - anchor.z).abs() < 1.0);
    assert!((far.y - 30.0).abs() < 1e-3, "rebase is horizontal only");
}

#[test]
fn anchored_entities_keep_their_mesh_albedo() {
    use crate::place::GlobalPlace;
    use crate::space::GlobalPosition;
    use crate::world::World;

    let mut mesh = Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(255, 255, 255)).unwrap();
    mesh.set_albedo_rgba(
        2,
        2,
        vec![
            120, 70, 30, 255, 160, 100, 45, 255, 90, 45, 20, 255, 190, 130, 65, 255,
        ],
    )
    .unwrap();

    let mut world = World::new();
    let id = world
        .spawn_anchored(
            mesh,
            GlobalPlace::at(GlobalPosition::at(1_000_000.0, 4.0, -2_000_000.0)),
        )
        .unwrap();

    assert!(
        world.entity(id).unwrap().albedo.is_some(),
        "anchored spawn dropped the mesh albedo and would render white"
    );
}

#[test]
fn repeated_rebases_keep_two_anchors_the_same_distance_apart() {
    use crate::place::GlobalPlace;
    use crate::space::{GlobalPosition, GlobalXZ, RenderOrigin};
    use crate::world::World;

    let mut world = World::new();
    let a = GlobalPosition::at(1_000_000.0, 5.0, 1_000_000.0);
    let b = GlobalPosition::at(1_000_037.5, 5.0, 1_000_012.25);
    let ea = world
        .spawn_anchored(
            Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(1, 1, 1)).unwrap(),
            GlobalPlace::at(a),
        )
        .unwrap();
    let eb = world
        .spawn_anchored(
            Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(1, 1, 1)).unwrap(),
            GlobalPlace::at(b),
        )
        .unwrap();

    let expected = ((b.x - a.x).powi(2) + (b.z - a.z).powi(2)).sqrt() as f32;
    for step in 0..6 {
        let origin = GlobalXZ::at(1_000_000.0 + step as f64 * 250_000.0, 1_000_000.0);
        world
            .set_render_origin(RenderOrigin::snapped(origin, 2_000.0).unwrap())
            .unwrap();
        let pa = world.entity(ea).unwrap().transform.w_axis.truncate();
        let pb = world.entity(eb).unwrap().transform.w_axis.truncate();
        let d = (pb - pa).length();
        assert!(
            (d - expected).abs() < 0.01,
            "rebase {step} drifted: {d} vs {expected}"
        );
    }
}

#[test]
fn terrain_texture_phase_is_continuous_across_rebase() {
    use crate::space::{GlobalXZ, RenderOrigin};

    // Two origins one whole tile apart must produce the same sampling phase.
    let tile = 8.0_f32;
    let a = RenderOrigin::new(GlobalXZ::at(1_000_000.0, -3_000.0));
    let b = RenderOrigin::new(GlobalXZ::at(1_000_000.0 + tile as f64 * 512.0, -3_000.0));
    let pa = a.texture_phase(tile);
    let pb = b.texture_phase(tile);
    assert!((pa[0] - pb[0]).abs() < 1e-3);
    assert!((pa[1] - pb[1]).abs() < 1e-3);

    // A world point keeps its UV: render x shrinks by the same amount the phase grows.
    let world_x = 1_000_050.0_f64;
    let ua = (world_x - a.horizontal().x) as f32 + pa[0];
    let ub = (world_x - b.horizontal().x) as f32 + pb[0];
    assert!(
        ((ua - ub) / tile).fract().abs() < 1e-3,
        "uv phase jumped by a fraction of a tile: {ua} vs {ub}"
    );
}

#[test]
fn a_scatter_layer_reuses_its_mesh_when_the_placements_change() {
    use crate::world::World;

    let mut world = World::new();
    let id = world.spawn_instanced(unit_quad());
    assert_eq!(world.entity(id).unwrap().instances.len(), 0);

    let places: Vec<Place> = (0..3)
        .map(|i| Place::new(i as f32 * 2.0, 0.0, 0.0))
        .collect();
    world.set_instances(id, &places).unwrap();
    assert_eq!(world.entity(id).unwrap().instances.len(), 3);

    // Emptying the set hides the layer instead of dropping it back to one copy
    // sitting at the origin.
    world.set_instances(id, &[]).unwrap();
    let e = world.entity(id).unwrap();
    assert!(e.instanced && e.instances.is_empty());
}

#[test]
fn set_instances_keeps_per_instance_tint() {
    use crate::world::World;

    let mut world = World::new();
    let id = world.spawn_instanced(unit_quad());
    let warm = Color {
        r: 1.0,
        g: 0.82,
        b: 0.62,
        a: 1.0,
    };
    let cool = Color {
        r: 0.72,
        g: 0.78,
        b: 0.92,
        a: 1.0,
    };
    world
        .set_instances(
            id,
            &[
                Place::new(0.0, 0.0, 0.0).with_tint(warm),
                Place::new(2.0, 0.0, 0.0).with_tint(cool),
            ],
        )
        .unwrap();
    let e = world.entity(id).unwrap();
    assert_eq!(e.instances.len(), 2);
    assert!((e.instances[0].tint.r - warm.r).abs() < 1e-5);
    assert!((e.instances[0].tint.b - warm.b).abs() < 1e-5);
    assert!((e.instances[1].tint.r - cool.r).abs() < 1e-5);
    assert!((e.instances[1].tint.b - cool.b).abs() < 1e-5);
    assert!(
        (e.instances[0].tint.r - e.instances[1].tint.r).abs() > 0.2,
        "tints must stay distinct on the entity"
    );
}

#[test]
fn an_instance_slot_can_reserve_gpu_capacity_without_drawing() {
    use crate::world::World;

    let mut world = World::new();
    let id = world.spawn_instanced(unit_quad());
    world.reserve_instances(id, 1_024).expect("reserve");
    let entity = world.entity(id).expect("instanced entity");
    assert_eq!(entity.instance_reserve, 1_024);
    assert!(
        entity.instances.is_empty(),
        "reserved capacity must not create visible instances"
    );
}

#[test]
fn instance_submit_defaults_to_gpu_and_switches_explicitly() {
    use crate::world::{InstanceSubmit, World};

    let mut world = World::new();
    assert_eq!(world.instance_submit(), InstanceSubmit::GpuIndirect);
    world.set_instance_submit(InstanceSubmit::CpuIndexed);
    assert_eq!(world.instance_submit(), InstanceSubmit::CpuIndexed);
    world.set_instance_submit(InstanceSubmit::GpuIndirect);
    assert_eq!(world.instance_submit(), InstanceSubmit::GpuIndirect);
}

#[test]
fn static_render_epoch_changes_only_for_effective_render_mutations() {
    use crate::world::{InstanceSubmit, World};

    let mut world = World::new();
    let initial = world.render_epoch();
    let id = world.spawn_instanced(unit_quad());
    assert!(world.render_epoch() > initial);

    let unchanged = world.render_epoch();
    world.set_casts_shadow(id, true).expect("same shadow state");
    world.set_instance_submit(InstanceSubmit::GpuIndirect);
    assert_eq!(world.render_epoch(), unchanged);

    world
        .set_instances(id, &[Place::new(2.0, 0.0, 0.0)])
        .expect("instance placement");
    assert!(world.render_epoch() > unchanged);
}

#[test]
fn resource_epoch_changes_when_a_texture_is_created() {
    use crate::world::World;

    let mut world = World::new();
    let initial = world.resource_epoch();
    world
        .create_texture_rgba(1, 1, vec![255, 255, 255, 255])
        .expect("one-pixel texture");
    assert!(world.resource_epoch() > initial);
}

#[test]
fn moving_or_scattering_an_entity_bumps_xform_rev() {
    use crate::world::World;

    let mut world = World::new();
    let id = world.spawn(unit_quad());
    let rev = world.entity(id).unwrap().xform_rev;
    world.set_place(id, Place::new(3.0, 0.0, 1.0)).unwrap();
    assert_ne!(world.entity(id).unwrap().xform_rev, rev);

    let scatter = world.spawn_instanced(unit_quad());
    let rev = world.entity(scatter).unwrap().xform_rev;
    world
        .set_instances(scatter, &[Place::new(1.0, 0.0, 0.0)])
        .unwrap();
    assert_ne!(world.entity(scatter).unwrap().xform_rev, rev);
}

#[test]
fn spawn_instanced_like_shares_albedo_and_owns_no_mesh() {
    use crate::world::World;

    let mut world = World::new();
    let proto = world.spawn_instanced(unit_quad());
    let like = world.spawn_instanced_like(proto).expect("like");
    assert_eq!(world.entity(like).unwrap().instance_of(), Some(proto));
    assert_eq!(world.entity(like).unwrap().mesh().vertex_count(), 0);
    assert!(world.entity(like).unwrap().instanced);
    world
        .set_instances(like, &[Place::new(2.0, 0.0, 0.0)])
        .unwrap();
    assert_eq!(world.entity(like).unwrap().instances.len(), 1);
    assert!(world.spawn_instanced_like(like).is_err());
}

#[test]
fn placing_instances_on_a_plain_entity_is_an_error() {
    use crate::world::World;

    let mut world = World::new();
    let id = world.spawn(unit_quad());
    assert!(world
        .set_instances(id, &[Place::new(1.0, 0.0, 0.0)])
        .is_err());
}

#[test]
fn default_mesh_material_is_the_valid_renderer_default() {
    assert_eq!(
        crate::mesh::Mesh::new().surface_material(),
        crate::mesh::SurfaceMaterial::DEFAULT
    );
    assert_eq!(
        crate::mesh::SurfaceMaterial::default(),
        crate::mesh::SurfaceMaterial::DEFAULT
    );
}

#[test]
fn a_translucent_chunk_layer_becomes_water_and_an_opaque_one_ground() {
    use crate::color::rgba;
    use crate::space::{ChunkCoord, ChunkId, ChunkLayer, GlobalXZ};
    use crate::texture::{TerrainAlbedo, TerrainMaterialDesc, WaterMaterialDesc};
    use crate::world::{SurfaceMaterialRef, World};

    let mut world = World::new();
    let grass = world
        .create_terrain_albedo(TerrainAlbedo::Grass, 16, 1)
        .unwrap();
    let grass_dry = world
        .create_terrain_albedo(TerrainAlbedo::GrassDry, 16, 4)
        .unwrap();
    let grass_moor = world
        .create_terrain_albedo(TerrainAlbedo::GrassMoor, 16, 5)
        .unwrap();
    let mud = world
        .create_terrain_albedo(TerrainAlbedo::Mud, 16, 6)
        .unwrap();
    let tundra = world
        .create_terrain_albedo(TerrainAlbedo::Tundra, 16, 7)
        .unwrap();
    let scree = world
        .create_terrain_albedo(TerrainAlbedo::Scree, 16, 8)
        .unwrap();
    let sand = world
        .create_terrain_albedo(TerrainAlbedo::Sand, 16, 2)
        .unwrap();
    let rock = world
        .create_terrain_albedo(TerrainAlbedo::Rock, 16, 3)
        .unwrap();
    let ground = world
        .create_terrain_material(TerrainMaterialDesc {
            grass,
            grass_dry,
            grass_moor,
            mud,
            tundra,
            scree,
            sand,
            rock,
            ..TerrainMaterialDesc::default()
        })
        .unwrap();
    let water = world
        .create_water_material(WaterMaterialDesc::default())
        .unwrap();
    world.set_default_terrain_material(Some(ground));
    world.set_default_water_material(Some(water));

    let anchor = GlobalXZ::at(0.0, 0.0).with_height(0.0).unwrap();
    let coord = ChunkCoord::new(0, 0);

    let land = world
        .set_anchored_chunk(
            ChunkId::new(coord, ChunkLayer::Land),
            anchor,
            unit_quad().build(),
        )
        .unwrap();
    let mut sheet = Mesh::new();
    let a = sheet.add_point((0.0, 0.0, 0.0)).unwrap();
    let b = sheet.add_point((1.0, 0.0, 0.0)).unwrap();
    let c = sheet.add_point((0.0, 0.0, 1.0)).unwrap();
    for p in [a, b, c] {
        sheet.set_point_color(p, rgba(40, 110, 160, 128)).unwrap();
    }
    sheet.add_triangle(a, b, c).unwrap();
    let wet = world
        .set_anchored_chunk(
            ChunkId::new(coord, ChunkLayer::Water),
            anchor,
            sheet.build(),
        )
        .unwrap();

    assert_eq!(
        world.entity(land).unwrap().material(),
        Some(SurfaceMaterialRef::Terrain(ground))
    );
    assert_eq!(
        world.entity(wet).unwrap().material(),
        Some(SurfaceMaterialRef::Water(water))
    );
}

#[test]
fn a_mouse_click_is_an_edge_and_holding_is_not() {
    use crate::input::MouseButton;
    let mut input = crate::input::Input::new();
    input.set_mouse_button(winit::event::MouseButton::Left, true);
    assert!(input.mouse_clicked(MouseButton::Left));
    assert!(input.mouse_down(MouseButton::Left));

    // Holding the button must not read as a click again, or a game that
    // captures the pointer on click would fight the player every frame.
    input.end_frame();
    assert!(!input.mouse_clicked(MouseButton::Left));
    assert!(input.mouse_down(MouseButton::Left));

    input.set_mouse_button(winit::event::MouseButton::Left, false);
    assert!(!input.mouse_down(MouseButton::Left));
}

#[test]
fn byte_colours_arrive_in_the_same_space_textures_do() {
    // Bytes are authored in sRGB; the GPU shades in linear. Handing 128 through
    // as 0.502 made every hand-written colour render pale and washed out next
    // to a texture, which the hardware decodes properly.
    let c = rgb(255, 0, 128);
    assert!((c.r - 1.0).abs() < 1e-5);
    assert!(c.g.abs() < 1e-5);
    assert!((c.b - 0.2158).abs() < 1e-3, "mid grey came out {}", c.b);
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

#[test]
fn a_daylight_sky_is_deeper_at_the_zenith() {
    let sky = crate::Sky::daylight();
    let zenith = sky.zenith.r + sky.zenith.g + sky.zenith.b;
    let horizon = sky.horizon.r + sky.horizon.g + sky.horizon.b;
    assert!(
        horizon > zenith,
        "horizon {horizon} should be paler than zenith {zenith}"
    );
    assert!(sky.zenith.b > sky.zenith.r);
    assert!(sky.sun_size_degrees > 0.0);
    assert!(sky.sun_bloom_degrees > sky.sun_size_degrees);
}

#[test]
fn a_world_can_wear_a_sky() {
    let mut world = crate::World::default();
    assert!(world.sky().is_none());
    world.set_sky(Some(crate::Sky::daylight()));
    assert_eq!(world.sky(), Some(crate::Sky::daylight()));
}

#[test]
fn hitch_spans_are_taken_once() {
    let mut world = crate::World::default();
    world.hitch_span("fauna", 4.5, "agents=12 born=2");
    world.hitch_span("stream", 1.0, "resident=8");
    let notes = world.take_hitch_spans();
    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0].name, "fauna");
    assert_eq!(notes[0].ms, 4.5);
    assert!(world.take_hitch_spans().is_empty());
}

#[test]
fn hitch_log_is_off_until_a_path_is_set() {
    let mut world = crate::World::default();
    assert!(world.hitch_log().is_none());
    world.set_hitch_log(Some(std::path::PathBuf::from("hitch.log")));
    assert_eq!(world.hitch_log().unwrap().as_os_str(), "hitch.log");
    world.set_hitch_log(None);
    assert!(world.hitch_log().is_none());
}

#[test]
fn like_entities_inherit_whether_they_cast_shadow() {
    let mut world = crate::World::default();
    let proto = world.spawn_instanced(unit_quad());
    world.set_casts_shadow(proto, false).expect("proto");
    let like = world.spawn_instanced_like(proto).expect("like");
    assert!(!world.entity(like).expect("like").casts_shadow());
    let caster = world.spawn_instanced(unit_quad());
    let like_cast = world.spawn_instanced_like(caster).expect("caster like");
    assert!(world.entity(like_cast).expect("caster like").casts_shadow());
}

#[test]
fn the_default_space_draws_the_outdoor_environment() {
    let world = crate::World::default();
    assert!(world.space_draws_environment(crate::SpaceId::DEFAULT));
}

#[test]
fn a_named_space_can_opt_into_the_outdoor_environment() {
    let mut world = crate::World::default();
    let travel = world.space("travel").unwrap();
    assert!(
        !world.space_draws_environment(travel),
        "a named space starts without sky and clipmap"
    );
    world.set_space_draws_environment(travel, true).unwrap();
    assert!(world.space_draws_environment(travel));
    world.set_space_draws_environment(travel, false).unwrap();
    assert!(!world.space_draws_environment(travel));
    assert!(
        world.space_draws_environment(crate::SpaceId::DEFAULT),
        "opting a named space in or out must not change the default space"
    );
}

#[test]
fn the_default_space_cannot_drop_the_outdoor_environment() {
    let mut world = crate::World::default();
    let err = world
        .set_space_draws_environment(crate::SpaceId::DEFAULT, false)
        .unwrap_err();
    assert!(matches!(err, EngineError::InvalidValue(_)));
    assert!(world.space_draws_environment(crate::SpaceId::DEFAULT));
}

#[test]
fn environment_opt_in_rejects_an_unknown_space() {
    let mut world = crate::World::default();
    let err = world
        .set_space_draws_environment(crate::SpaceId::from_raw(99), true)
        .unwrap_err();
    assert!(matches!(err, EngineError::InvalidValue(_)));
}

#[test]
fn opening_faces_plus_z() {
    let mesh = Mesh::opening(1.2, 2.2).unwrap();
    let built = mesh.build();
    assert_eq!(built.triangle_count(), 2);
    assert!(built.normals.iter().all(|n| n.z > 0.9));
}

#[test]
fn spaces_keep_spawns_apart() {
    let mut world = crate::World::default();
    let house = world.space("house").unwrap();
    let yard = world.spawn(unit_quad());
    let table = world.spawn_in(house, unit_quad()).unwrap();
    assert_eq!(world.entity(yard).unwrap().space(), crate::SpaceId::DEFAULT);
    assert_eq!(world.entity(table).unwrap().space(), house);
    world.live_in(house).unwrap();
    assert_eq!(world.living_in(), house);
}

#[test]
fn portals_doorway_stays_visible_inches_from_plane() {
    use crate::camera::Camera;
    let mut world = crate::World::default();
    let house = world.space("house").unwrap();
    let door_out = world
        .place(
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -3.0).with_yaw_deg(180.0),
        )
        .unwrap();
    let door_in = world
        .place_in(
            house,
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -4.0),
        )
        .unwrap();
    world
        .create_portal(door_out, door_in, crate::PortalSettings::TELEPORTING)
        .unwrap();
    let look = Camera::direction(180.0, 0.0);
    let plane = world.portal_plane(door_in).unwrap();
    for z in [-3.96_f32, -3.92, -3.88] {
        let eye = Vec3::new(0.0, 1.6, z);
        let dist = plane.signed_distance(eye);
        let portals = world.visible_portals(eye, look, house).unwrap();
        assert!(
            !portals.is_empty(),
            "portal should stay visible at z={z}, dist={dist}"
        );
    }
}

#[test]
fn walking_through_a_door_switches_space() {
    let mut world = crate::World::default();
    let house = world.space("house").unwrap();
    let outside = world
        .place(
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -1.2).with_yaw_deg(180.0),
        )
        .unwrap();
    let inside = world
        .place_in(
            house,
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -3.0),
        )
        .unwrap();
    world
        .create_portal(outside, inside, crate::PortalSettings::TELEPORTING)
        .unwrap();

    let mut pos = Vec3::new(0.0, 1.6, -6.0);
    let mut yaw = 0.0_f32;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.z = 0.0;
    let entered = world.travel(&mut pos, &mut yaw);
    assert_eq!(entered, Some(house));
    assert_eq!(world.living_in(), house);
    assert!(pos.z > -3.0, "should stand inside the room, pos={pos}");
    assert!(yaw.abs() < 5.0, "forward should still be +Z, yaw={yaw}");
}

#[test]
fn view_only_portal_does_not_teleport() {
    let mut world = crate::World::default();
    let house = world.space("house").unwrap();
    let outside = world
        .place(
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -1.2).with_yaw_deg(180.0),
        )
        .unwrap();
    let inside = world
        .place_in(
            house,
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -3.0),
        )
        .unwrap();
    world
        .create_portal(outside, inside, crate::PortalSettings::VIEW_ONLY)
        .unwrap();

    let mut pos = Vec3::new(0.0, 1.6, -6.0);
    let mut yaw = 0.0_f32;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.z = 0.0;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    assert_eq!(world.living_in(), crate::SpaceId::DEFAULT);
    assert!(
        !world.portals().is_empty() && !world.portals()[0].teleports(),
        "portal should still render but not teleport"
    );
}

#[test]
fn same_space_portals_work_in_both_directions() {
    let mut world = crate::World::default();
    let a = world
        .place(Mesh::opening(1.2, 2.2).unwrap(), Place::new(0.0, 1.1, -4.0))
        .unwrap();
    let b = world
        .place(
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, 4.0).with_yaw_deg(180.0),
        )
        .unwrap();
    world
        .create_portal(a, b, crate::PortalSettings::TELEPORTING)
        .unwrap();

    let mut pos = Vec3::new(0.0, 1.6, 0.0);
    let mut yaw = 180.0_f32;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.z = -5.0;
    assert_eq!(
        world.travel(&mut pos, &mut yaw),
        Some(crate::SpaceId::DEFAULT)
    );
    assert!(
        pos.z >= 3.0,
        "walking into A should come out near B, pos={pos}"
    );

    pos = Vec3::new(0.0, 1.6, 2.0);
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.z = 5.0;
    assert_eq!(
        world.travel(&mut pos, &mut yaw),
        Some(crate::SpaceId::DEFAULT)
    );
    assert!(
        pos.z <= -3.0,
        "walking into B should come out near A, pos={pos}"
    );
}

#[test]
fn disabled_portal_openings_remain_stencil_only() {
    let mut world = crate::World::default();
    let house = world.space("disabled-portal-house").unwrap();
    let outside = world
        .place(Mesh::opening(1.2, 2.2).unwrap(), Place::default())
        .unwrap();
    world.in_space(house).unwrap();
    let inside = world
        .place(Mesh::opening(1.2, 2.2).unwrap(), Place::default())
        .unwrap();
    world.in_space(crate::SpaceId::DEFAULT).unwrap();
    let portal = world
        .create_portal(outside, inside, crate::PortalSettings::TELEPORTING)
        .unwrap();
    world.set_portal_enabled(portal, false).unwrap();
    assert!(world.is_portal_surface(outside));
    assert!(world.is_portal_surface(inside));
    assert!(world
        .visible_portals(Vec3::Z, -Vec3::Z, crate::SpaceId::DEFAULT)
        .unwrap()
        .is_empty());
}
#[test]
fn destroy_portal_stops_crossing() {
    let mut world = crate::World::default();
    let house = world.space("house").unwrap();
    let outside = world
        .place(
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -1.2).with_yaw_deg(180.0),
        )
        .unwrap();
    let inside = world
        .place_in(
            house,
            Mesh::opening(1.2, 2.2).unwrap(),
            Place::new(0.0, 1.1, -3.0),
        )
        .unwrap();
    let id = world
        .create_portal(outside, inside, crate::PortalSettings::TELEPORTING)
        .unwrap();
    world.destroy_portal(id).unwrap();
    assert!(world.portals().is_empty());

    let mut pos = Vec3::new(0.0, 1.6, -6.0);
    let mut yaw = 0.0_f32;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.z = 0.0;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    assert_eq!(world.living_in(), crate::SpaceId::DEFAULT);
}

#[test]
fn create_portal_rejects_a_self_door() {
    let mut world = crate::World::default();
    let door = world.spawn(Mesh::opening(1.0, 2.0).unwrap());
    let err = world
        .create_portal(door, door, crate::PortalSettings::TELEPORTING)
        .unwrap_err();
    assert!(matches!(err, EngineError::InvalidValue(_)));
}

#[test]
fn falling_through_a_hatch_switches_space() {
    let mut world = crate::World::default();
    let dungeon = world.space("dungeon").unwrap();
    let outside = world
        .place(
            Mesh::opening(2.0, 2.0).unwrap(),
            Place::new(0.0, 10.0, 0.0).with_pitch_deg(-90.0),
        )
        .unwrap();
    let inside = world
        .place_in(
            dungeon,
            Mesh::opening(2.0, 2.0).unwrap(),
            Place::new(0.0, 4.0, 0.0).with_pitch_deg(90.0),
        )
        .unwrap();
    world
        .create_portal(outside, inside, crate::PortalSettings::TELEPORTING)
        .unwrap();

    let mut pos = Vec3::new(0.0, 11.0, 0.0);
    let mut yaw = 0.0_f32;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.y = 9.0;
    let entered = world.travel(&mut pos, &mut yaw);
    assert_eq!(entered, Some(dungeon));
    assert_eq!(world.living_in(), dungeon);
    assert!(
        pos.y < 4.0,
        "falling in should come out below the mouth, pos={pos}"
    );
    assert!(yaw.abs() < 5.0, "hatch travel must not spin yaw, got {yaw}");

    pos.y = 3.0;
    assert!(world.travel(&mut pos, &mut yaw).is_none());
    pos.y = 5.0;
    let left = world.travel(&mut pos, &mut yaw);
    assert_eq!(left, Some(crate::SpaceId::DEFAULT));
    assert!(
        pos.y > 10.0,
        "climbing out should come out above the world hatch, pos={pos}"
    );
    assert!(yaw.abs() < 5.0, "climb-out must not spin yaw, got {yaw}");
}

#[test]
fn looking_down_a_hatch_sees_the_destination() {
    let mut world = crate::World::default();
    let dungeon = world.space("dungeon").unwrap();
    let outside = world
        .place(
            Mesh::opening(2.0, 2.0).unwrap(),
            Place::new(0.0, 10.0, 0.0).with_pitch_deg(-90.0),
        )
        .unwrap();
    let inside = world
        .place_in(
            dungeon,
            Mesh::opening(2.0, 2.0).unwrap(),
            Place::new(0.0, 4.0, 0.0).with_pitch_deg(90.0),
        )
        .unwrap();
    world
        .create_portal(outside, inside, crate::PortalSettings::TELEPORTING)
        .unwrap();

    let visible = world
        .visible_portal(Vec3::new(0.0, 12.0, 0.0), -Vec3::Y)
        .expect("portal visibility must be valid")
        .expect("looking down at a hatch must see the other space");
    assert_eq!(visible.src, outside);
    assert_eq!(visible.dst, inside);
    assert_eq!(visible.dest_space, dungeon);
}

#[test]
fn lod_set_moves_instances_between_levels_with_the_camera() {
    use crate::mesh::Shape;
    use crate::world::World;
    let mut world = World::new();
    let cube = || Shape::box_at((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), rgb(200, 200, 200)).unwrap();
    let id = world
        .spawn_lod(vec![
            crate::lod::LodLevel::new(cube(), 20.0),
            crate::lod::LodLevel::new(cube(), 100.0),
        ])
        .unwrap();
    let places: Vec<Place> = [5.0, 50.0, 150.0]
        .iter()
        .map(|x| Place::new(*x, 0.0, 0.0))
        .collect();
    world.set_lod_places(id, &places).unwrap();
    let levels = world.lod_levels(id).unwrap();
    let counts = |w: &World| -> Vec<usize> {
        levels
            .iter()
            .map(|e| w.entity(*e).unwrap().instances.len())
            .collect()
    };

    world.look_first_person((0.0, 0.0, 0.0), 90.0, 0.0);
    world.tick_lods();
    assert_eq!(counts(&world), vec![1, 1], "150 m is past the last level");
    let epoch = world.render_epoch();

    // Walk to x = 55: the far tree comes in, the near one moves to level 1;
    // only instance lists change, so the scene epoch stays.
    world.look_first_person((55.0, 0.0, 0.0), 90.0, 0.0);
    world.tick_lods();
    assert_eq!(counts(&world), vec![1, 2]);
    assert_eq!(world.render_epoch(), epoch);

    assert!(world.set_lod_bias(0.0).is_err());
    world.despawn_lod(id);
    assert!(world.entity(levels[0]).is_err());
}

#[test]
fn physical_sky_drives_light_and_sky() {
    use crate::sky::PhysicalSky;
    let mut world = crate::world::World::new();
    world.set_physical_sky(Some(PhysicalSky::at_hour(12.0)));
    world.tick_sky();
    let noon = world.light().clone();
    assert!(noon.direction.y > 0.5, "noon sun is high");
    let sky = world.sky().expect("sky derived");
    assert!(sky.zenith.b > sky.zenith.r, "noon zenith is blue");

    world.set_physical_sky(Some(PhysicalSky::at_hour(1.0)));
    world.tick_sky();
    let night = world.light();
    assert!(
        night.color.max_element() < noon.color.max_element() * 0.2,
        "night is much darker than noon"
    );
}

#[test]
fn wind_and_season_are_validated() {
    let mut world = crate::world::World::new();
    assert_eq!(world.wind(), 0.5);
    world.set_wind(1.5).expect("breeze");
    assert_eq!(world.wind(), 1.5);
    assert!(world.set_wind(-1.0).is_err());
    assert!(world.set_wind(f32::NAN).is_err());
    world.set_season(0.7).expect("autumn");
    assert!(world.set_season(1.5).is_err());
}

#[test]
fn trees_share_one_atlas_texture() {
    use crate::vegetation::{tree_model, TreeSpecies};
    let mut world = crate::world::World::new();
    let oak = tree_model(TreeSpecies::Oak, 0).expect("oak");
    let birch = tree_model(TreeSpecies::Birch, 0).expect("birch");
    let a = world.spawn_lod(oak.lod_levels()).expect("oak lod");
    let b = world.spawn_lod(birch.lod_levels()).expect("birch lod");
    let tex = |w: &crate::world::World, id| {
        let level = w.lod_levels(id).expect("levels")[0];
        w.entity(level).expect("entity").albedo
    };
    assert!(tex(&world, a).is_some());
    assert_eq!(tex(&world, a), tex(&world, b));
}

mod physics_world {
    use crate::color::rgb;
    use crate::mesh::Mesh;
    use crate::physics::{Body, PhysicsGround, PhysicsSettings};
    use crate::place::Place;
    use crate::world::World;
    use glam::Vec3;

    fn crate_at(world: &mut World, p: Vec3) -> crate::world::EntityId {
        let id = world.spawn(Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(150, 100, 50)).unwrap());
        world.set_place(id, Place::new(p.x, p.y, p.z)).unwrap();
        id
    }

    fn run(world: &mut World, seconds: f32) {
        for _ in 0..(seconds * 60.0) as usize {
            world.tick_physics(1.0 / 60.0);
        }
    }

    fn y_of(world: &World, id: crate::world::EntityId) -> f32 {
        world.entity(id).unwrap().transform().w_axis.y
    }

    fn voxel_ground(top: f32) -> crate::voxel_terrain::VoxelTerrain {
        let mut volume = crate::volume::Volume::new(0.5);
        volume.ensure_bounds(Vec3::new(-8.0, -4.0, -8.0), Vec3::new(8.0, 20.0, 8.0));
        let keys: Vec<_> = volume.chunk_keys().collect();
        volume.generate_chunks(&keys, |p| top - p.y);
        crate::voxel_terrain::VoxelTerrain::new(volume)
    }

    #[test]
    fn a_crate_rests_on_voxel_ground_and_falls_into_a_dug_hole() {
        let mut world = World::new();
        world.set_voxel_terrain(Some(voxel_ground(2.0)));
        let id = crate_at(&mut world, Vec3::new(0.3, 6.0, 0.3));
        world.add_body(id, Body::dynamic()).unwrap();
        run(&mut world, 3.0);
        let rest = y_of(&world, id);
        assert!((rest - 2.5).abs() < 0.1, "rest height {rest}");
        assert_eq!(world.body_count().1, 0, "should be asleep");
        world
            .voxel_terrain_mut()
            .unwrap()
            .volume_mut()
            .dig_sphere(Vec3::new(0.3, 2.0, 0.3), 2.5);
        run(&mut world, 3.0);
        let after = y_of(&world, id);
        assert!(
            after < rest - 1.0,
            "crate should drop into the hole: {after}"
        );
    }

    /// A pile of crates and balls tumbling down height terrain, built once in
    /// `World` and once in a standalone `PhysicsWorld`, stepped 10 s.
    #[test]
    fn standalone_physics_matches_the_world() {
        use crate::physics::{BodyShape, PhysicsWorld};
        let rules = crate::proc_terrain::demo_terrain_rules();
        let field = crate::proc_terrain::HeightField::new(rules.clone());
        let mut world = World::new();
        world.set_proc_terrain(crate::proc_terrain::ProcTerrain::gpu_clipmap(
            rules.clone(),
            Default::default(),
        ));
        let mut alone = PhysicsWorld::new();
        alone.set_height_terrain(crate::proc_terrain::HeightField::new(rules));
        let mut pairs = Vec::new();
        for i in 0..12 {
            let x = (i % 4) as f32 * 1.3 - 2.0;
            let z = (i / 4) as f32 * 1.3 + 3.0;
            let p = Vec3::new(x, field.height_at(x, z) + 2.0 + i as f32 * 0.7, z);
            let mesh = Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(150, 100, 50)).unwrap();
            let body = if i % 3 == 0 {
                Body::dynamic().shape(BodyShape::Ball(0.5)).bounce(0.4)
            } else {
                Body::dynamic().density(0.8 + i as f32 * 0.1)
            };
            let id = world.spawn(mesh.clone());
            world.set_place(id, Place::new(p.x, p.y, p.z)).unwrap();
            world.add_body(id, body.clone()).unwrap();
            let transform = world.entity(id).unwrap().transform();
            let b = alone.add_mesh(&body, transform, &mesh).unwrap();
            pairs.push((id, b, p));
        }
        for frame in 0..600 {
            if frame == 120 {
                world.blast(Vec3::new(-1.0, field.height_at(-1.0, 4.0), 4.0), 6.0, 8.0);
                alone.blast(Vec3::new(-1.0, field.height_at(-1.0, 4.0), 4.0), 6.0, 8.0);
            }
            world.tick_physics(1.0 / 60.0);
            alone.step(1.0 / 60.0);
        }
        let mut travelled = 0.0f32;
        for (id, b, start) in pairs {
            let w = world.entity(id).unwrap().transform();
            let a = alone.transform(b).unwrap();
            assert!(
                w.abs_diff_eq(a, 1e-5),
                "body {b:?} differs:\nworld {w:?}\nalone {a:?}"
            );
            travelled = travelled.max(a.w_axis.truncate().distance(start));
        }
        assert!(travelled > 2.0, "the pile should have tumbled: {travelled}");
        assert!(alone.body_count().0 == 12);
    }

    #[test]
    fn standalone_physics_runs_rays_impacts_and_one_step_at_a_time() {
        use crate::physics::{BodyShape, PhysicsWorld};
        let mut p = PhysicsWorld::new();
        let ball = p
            .add(
                &Body::dynamic().shape(BodyShape::Ball(0.5)),
                glam::Mat4::from_translation(Vec3::new(0.0, 3.0, 0.0)),
            )
            .unwrap();
        assert!(
            p.add(&Body::dynamic(), glam::Mat4::IDENTITY).is_err(),
            "a hull needs a mesh"
        );
        for _ in 0..90 {
            p.step_once();
        }
        let (pos, _) = p.pose(ball).unwrap();
        assert!((pos.y - 0.5).abs() < 0.05, "{pos}");
        assert_eq!(
            p.position(ball).unwrap(),
            pos,
            "step_once shows the last step"
        );
        assert!(p
            .take_impacts()
            .iter()
            .any(|i| i.a == ball && i.b.is_none()));
        let hit = p
            .cast_ray(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, 10.0)
            .unwrap();
        assert_eq!(hit.body, Some(ball));
        let floor = p
            .cast_ray(Vec3::new(4.0, 5.0, 0.0), -Vec3::Y, 10.0)
            .unwrap();
        assert_eq!(floor.body, None);
        assert!((floor.point.y).abs() < 1e-4);
        assert_eq!(crate::physics::BodyId::from_raw(ball.raw()), ball);
        p.set_transform(ball, glam::Mat4::from_translation(Vec3::new(5.0, 0.5, 0.0)))
            .unwrap();
        p.step_once();
        assert!((p.position(ball).unwrap().x - 5.0).abs() < 1e-3);
        p.remove(ball);
        assert!(!p.contains(ball));
    }

    #[test]
    fn a_crate_lands_on_height_terrain() {
        let mut world = World::new();
        let rules = crate::proc_terrain::demo_terrain_rules();
        let field = crate::proc_terrain::HeightField::new(rules.clone());
        let ground = field.height_at(3.0, 4.0);
        world.set_proc_terrain(crate::proc_terrain::ProcTerrain::gpu_clipmap(
            rules,
            Default::default(),
        ));
        let id = crate_at(&mut world, Vec3::new(3.0, ground + 5.0, 4.0));
        world
            .add_body(
                id,
                Body::dynamic().shape(crate::physics::BodyShape::Ball(0.5)),
            )
            .unwrap();
        run(&mut world, 1.5);
        let y = y_of(&world, id);
        assert!(
            y > ground - 0.6 && y < ground + 2.0,
            "ball at {y}, ground {ground}"
        );
        let hit = world
            .cast_ray(Vec3::new(10.0, ground + 50.0, 10.0), -Vec3::Y, 200.0)
            .unwrap();
        assert!(hit.entity.is_none());
        assert!((hit.point.y - field.height_at(10.0, 10.0)).abs() < 0.05);
    }

    #[test]
    fn shattered_pieces_pile_up_instead_of_overlapping_the_floor() {
        let mut world = World::new();
        let id = crate_at(&mut world, Vec3::new(0.0, 0.5, 0.0));
        let pieces = world
            .shatter(
                id,
                Vec3::new(0.0, 0.8, 0.4),
                crate::debris::Shatter::default(),
            )
            .unwrap();
        assert!(pieces.len() > 4);
        assert_eq!(world.debris_count(), pieces.len());
        run(&mut world, 5.0);
        for p in &pieces {
            let y = y_of(&world, *p);
            assert!(y > -0.05 && y < 1.5, "piece at {y}");
        }
    }

    #[test]
    fn a_fixed_wall_stops_a_thrown_crate_and_reports_the_impact() {
        let mut world = World::new();
        let wall = world
            .spawn(Mesh::box_at(Vec3::ZERO, Vec3::new(0.4, 3.0, 4.0), rgb(128, 128, 128)).unwrap());
        world.set_place(wall, Place::new(4.0, 1.5, 0.0)).unwrap();
        world.add_body(wall, Body::fixed()).unwrap();
        let id = crate_at(&mut world, Vec3::new(0.0, 0.6, 0.0));
        world
            .add_body(id, Body::dynamic().velocity(Vec3::new(12.0, 0.0, 0.0)))
            .unwrap();
        run(&mut world, 2.0);
        let x = world.entity(id).unwrap().transform().w_axis.x;
        assert!(x < 3.4, "crate went through the wall: {x}");
        let hits = world.take_impacts();
        assert!(
            hits.iter().any(|h| h.b == Some(wall) || h.a == wall),
            "no impact with the wall: {hits:?}"
        );
    }

    #[test]
    fn light_crates_float_and_heavy_ones_sink() {
        use crate::water::{HeightMapTerrain, Water, WaterRegion};
        let mut world = World::new();
        let floor = HeightMapTerrain(|_: f32, _: f32| 0.0);
        let region = WaterRegion::new(glam::Vec2::ZERO, glam::Vec2::splat(16.0), 1.0, -1.0, 10.0);
        let mut water = Water::new(region, &floor);
        water.flood(Vec3::new(8.0, 0.5, 8.0), 3.0);
        world.set_water(Some(water));
        let wood = crate_at(&mut world, Vec3::new(6.0, 6.0, 8.0));
        world.add_body(wood, Body::dynamic().density(0.5)).unwrap();
        let stone = crate_at(&mut world, Vec3::new(10.0, 6.0, 8.0));
        world.add_body(stone, Body::dynamic().density(2.4)).unwrap();
        run(&mut world, 8.0);
        let w = y_of(&world, wood);
        let s = y_of(&world, stone);
        assert!(w > 2.6 && w < 3.4, "wood should float at the surface: {w}");
        assert!(s < 0.7, "stone should sink: {s}");
    }

    #[test]
    fn ground_plane_setting_moves_the_floor() {
        let mut world = World::new();
        world
            .set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(-3.0),
                ..Default::default()
            })
            .unwrap();
        let id = crate_at(&mut world, Vec3::new(0.0, 1.0, 0.0));
        world.add_body(id, Body::dynamic()).unwrap();
        run(&mut world, 3.0);
        assert!((y_of(&world, id) + 2.5).abs() < 0.05);
        world.despawn(id);
        assert_eq!(world.body_count().0, 0);
    }
}

#[test]
#[ignore]
fn bench_voxel_collision_tile() {
    let mut volume = crate::volume::Volume::new(1.0);
    volume.ensure_bounds(Vec3::new(0.0, 0.0, 0.0), Vec3::new(31.0, 31.0, 31.0));
    let keys: Vec<_> = volume.chunk_keys().collect();
    volume.generate_chunks(&keys, |p| {
        16.0 + (p.x * 0.2).sin() * 4.0 + (p.z * 0.15).cos() * 4.0 - p.y
    });
    let t = std::time::Instant::now();
    let (v, tris) = volume.collision_triangles(keys[0]);
    let mesh_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let c = rapier3d::prelude::ColliderBuilder::trimesh_with_flags(
        v.clone(),
        tris.clone(),
        rapier3d::prelude::TriMeshFlags::FIX_INTERNAL_EDGES,
    );
    let build_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "{} tris: marching {mesh_ms:.2} ms, collider {build_ms:.2} ms ok={}",
        tris.len(),
        c.is_ok()
    );
}

mod lamps_world {
    use crate::color::Color;
    use crate::lamps::{Lamp, LampShape};
    use crate::particles::ParticleEmitter;
    use crate::world::World;
    use glam::Vec3;

    #[test]
    fn lamps_can_be_added_moved_changed_and_removed() {
        let mut world = World::default();
        let fire = world.add_lamp(Lamp::fire((1.0, 0.5, 2.0))).unwrap();
        let spot = world
            .add_lamp(Lamp::spot(
                Vec3::Y * 8.0,
                -Vec3::Y,
                30.0,
                Color::WHITE,
                50.0,
                30.0,
            ))
            .unwrap();
        assert_ne!(fire, spot);
        assert_eq!(world.lamps().count(), 2);

        world.set_lamp_position(fire, (3.0, 0.5, 2.0)).unwrap();
        assert_eq!(world.lamp(fire).unwrap().position, Vec3::new(3.0, 0.5, 2.0));
        let dimmer = world.lamp(spot).unwrap().intensity(5.0);
        world.set_lamp(spot, dimmer).unwrap();
        assert_eq!(world.lamp(spot).unwrap().intensity, 5.0);
        assert!(matches!(
            world.lamp(spot).unwrap().shape,
            LampShape::Spot { .. }
        ));

        // Bad values are refused and leave the lamp as it was.
        assert!(world.set_lamp(spot, dimmer.range(-1.0)).is_err());
        assert!(world.set_lamp_position(fire, (f32::NAN, 0.0, 0.0)).is_err());
        assert!(world.add_lamp(Lamp::fire(Vec3::ZERO).flicker(2.0)).is_err());
        assert_eq!(world.lamp(spot).unwrap().range, 30.0);

        world.remove_lamp(fire);
        assert!(world.lamp(fire).is_none());
        assert!(world.set_lamp_position(fire, Vec3::ZERO).is_err());
        assert_eq!(world.lamps().count(), 1);

        world.set_lamp_shadow_budget(100);
        assert_eq!(world.lamp_shadow_budget(), crate::lamps::MAX_SHADOW_LAMPS);
    }

    #[test]
    fn particles_run_by_themselves_unless_the_game_drives_them() {
        let mut world = World::default();
        world
            .particles_mut()
            .start(ParticleEmitter::new(Vec3::ZERO))
            .unwrap();
        world.tick_effects(0.1);
        let (_, dt, time) = world.particles_frame();
        assert!((dt - 0.1).abs() < 1e-6 && (time - 0.1).abs() < 1e-6);

        // A game in slow motion advances them itself; the engine stays out.
        world.particles_mut().advance(0.025).unwrap();
        world.tick_effects(0.1);
        let (_, dt, _) = world.particles_frame();
        assert!((dt - 0.025).abs() < 1e-6, "{dt}");

        // And the next frame without a manual call runs by itself again.
        world.tick_effects(0.1);
        let (_, dt, _) = world.particles_frame();
        assert!((dt - 0.1).abs() < 1e-6, "{dt}");
    }
}

mod hierarchy_world {
    use crate::anim::AnimatedModel;
    use crate::color::{rgb, Color};
    use crate::lamps::{Lamp, LampShape};
    use crate::mesh::Mesh;
    use crate::physics::{Body, PhysicsGround, PhysicsSettings};
    use crate::place::Place;
    use crate::prefab::Prefab;
    use crate::world::{EntityId, World};
    use glam::Vec3;
    use std::sync::Arc;

    fn cube() -> Mesh {
        Mesh::box_at(Vec3::ZERO, Vec3::splat(0.5), rgb(120, 120, 120)).unwrap()
    }

    fn pos(world: &World, id: EntityId) -> Vec3 {
        world.world_transform(id).unwrap().w_axis.truncate()
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-4
    }

    #[test]
    fn children_follow_parents_and_places_are_relative() {
        let mut world = World::default();
        let cart = world.spawn(cube());
        let wheel = world.spawn(cube());
        let hub = world.spawn(cube());
        world
            .attach(wheel, cart, Place::new(1.0, 0.0, 0.0))
            .unwrap();
        world.attach(hub, wheel, Place::new(0.0, 0.5, 0.0)).unwrap();
        assert!(close(pos(&world, hub), Vec3::new(1.0, 0.5, 0.0)));

        // Moving and turning the parent carries the whole chain along.
        world
            .set_place(cart, Place::new(10.0, 0.0, 0.0).with_yaw_deg(90.0))
            .unwrap();
        assert!(
            close(pos(&world, wheel), Vec3::new(10.0, 0.0, -1.0)),
            "{}",
            pos(&world, wheel)
        );
        assert!(close(pos(&world, hub), Vec3::new(10.0, 0.5, -1.0)));

        // A child's own place is relative to its parent.
        world.set_place(wheel, Place::new(2.0, 0.0, 0.0)).unwrap();
        assert!(close(pos(&world, wheel), Vec3::new(10.0, 0.0, -2.0)));
        assert_eq!(world.parent(hub), Some(wheel));
        assert_eq!(world.children(cart), &[wheel]);

        // No loops.
        assert!(world.attach(cart, hub, Place::default()).is_err());
        assert!(world.attach(cart, cart, Place::default()).is_err());

        // Letting go keeps the place on screen.
        world.detach(wheel).unwrap();
        assert!(close(pos(&world, wheel), Vec3::new(10.0, 0.0, -2.0)));
        world.set_place(cart, Place::new(0.0, 0.0, 0.0)).unwrap();
        assert!(close(pos(&world, wheel), Vec3::new(10.0, 0.0, -2.0)));

        // ... and a new parent keeps it too.
        world.set_parent(wheel, Some(cart)).unwrap();
        assert!(close(pos(&world, wheel), Vec3::new(10.0, 0.0, -2.0)));

        // Despawning a parent removes its children.
        world.despawn(cart);
        assert!(world.world_transform(wheel).is_err());
        assert!(world.world_transform(hub).is_err());
    }

    #[test]
    fn pivots_group_things_and_lamps_ride_along() {
        let mut world = World::default();
        let car = world.spawn_pivot(Place::new(0.0, 0.0, 0.0));
        let body = world.spawn(cube());
        world.attach(body, car, Place::new(0.0, 0.5, 0.0)).unwrap();
        let light = world
            .add_lamp(Lamp::spot(
                Vec3::ZERO,
                Vec3::Z,
                40.0,
                Color::WHITE,
                20.0,
                30.0,
            ))
            .unwrap();
        world.attach_lamp(light, car, (0.0, 0.8, 2.0)).unwrap();
        assert!(close(
            world.lamp(light).unwrap().position,
            Vec3::new(0.0, 0.8, 2.0)
        ));

        // Turn the car to face +X: the headlight turns with it.
        world
            .set_place(car, Place::new(5.0, 0.0, 5.0).with_yaw_deg(90.0))
            .unwrap();
        world.tick_hierarchy();
        let lamp = world.lamp(light).unwrap();
        assert!(
            close(lamp.position, Vec3::new(7.0, 0.8, 5.0)),
            "{}",
            lamp.position
        );
        let LampShape::Spot { direction, .. } = lamp.shape else {
            panic!("spot")
        };
        assert!(close(direction, Vec3::X), "{direction}");
        assert!(close(pos(&world, body), Vec3::new(5.0, 0.5, 5.0)));

        world.despawn(car);
        assert!(world.lamp(light).is_none());
        assert!(world.world_transform(body).is_err());
    }

    fn cart() -> Prefab {
        let mut cart = Prefab::new("cart");
        let body = cart.mesh("body", cube(), Place::new(0.0, 0.6, 0.0));
        cart.mesh_on(body, "wheel_l", cube(), Place::new(-0.7, -0.3, 0.0));
        cart.mesh_on(body, "wheel_r", cube(), Place::new(0.7, -0.3, 0.0));
        cart.lamp_on(body, Lamp::point((0.0, 0.5, 1.0), Color::WHITE, 3.0, 8.0));
        cart
    }

    #[test]
    fn prefabs_place_parts_and_despawn_as_one() {
        let mut world = World::default();
        let a = world
            .spawn_prefab(&cart(), Place::new(10.0, 0.0, 0.0))
            .unwrap();
        let b = world
            .spawn_prefab(&cart(), Place::new(-10.0, 0.0, 0.0))
            .unwrap();
        assert!(close(
            pos(&world, a.part("wheel_r").unwrap()),
            Vec3::new(10.7, 0.3, 0.0)
        ));
        assert!(close(
            pos(&world, b.part("wheel_l").unwrap()),
            Vec3::new(-10.7, 0.3, 0.0)
        ));
        assert!(close(
            world.lamp(a.lamps[0]).unwrap().position,
            Vec3::new(10.0, 1.1, 1.0)
        ));
        assert_eq!(a.parts().count(), 3);
        assert_eq!(world.prefab_of(a.root), Some("cart"));

        world.set_place(a.root, Place::new(20.0, 0.0, 0.0)).unwrap();
        assert!(close(
            pos(&world, a.part("body").unwrap()),
            Vec3::new(20.0, 0.6, 0.0)
        ));

        world.despawn(a.root);
        for (_, id) in a.parts() {
            assert!(world.world_transform(id).is_err());
        }
        assert!(world.lamp(a.lamps[0]).is_none());
        assert!(world.world_transform(b.root).is_ok());
        assert_eq!(world.lamps().count(), 1);

        // By name, once registered.
        assert!(world.spawn_named("cart", Place::default()).is_err());
        world.register_prefab(cart()).unwrap();
        let c = world
            .spawn_named("cart", Place::new(0.0, 0.0, 5.0))
            .unwrap();
        assert!(close(
            pos(&world, c.part("body").unwrap()),
            Vec3::new(0.0, 0.6, 5.0)
        ));

        // Broken prefabs are refused up front.
        let mut bad = Prefab::new("bad");
        let pivot = bad.pivot("hinge", Place::default());
        bad.body(pivot, Body::dynamic());
        assert!(world.spawn_prefab(&bad, Place::default()).is_err());
        assert!(world
            .spawn_prefab(&Prefab::new("empty"), Place::default())
            .is_err());
    }

    #[test]
    fn a_dynamic_part_falls_with_its_children_and_still_despawns_with_the_root() {
        let mut world = World::default();
        world
            .set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(0.0),
                ..Default::default()
            })
            .unwrap();
        let mut falling = cart();
        let body = falling.find("body").unwrap();
        falling.body(body, Body::dynamic());
        let s = world
            .spawn_prefab(&falling, Place::new(0.0, 3.0, 0.0))
            .unwrap();
        let (body, wheel) = (s.part("body").unwrap(), s.part("wheel_l").unwrap());
        assert_eq!(world.parent(body), None, "physics owns the body now");
        for _ in 0..120 {
            world.tick_physics(1.0 / 60.0);
            world.tick_hierarchy();
        }
        let b = pos(&world, body);
        assert!(b.y < 1.0, "fell: {b}");
        // The wheel stays where it hangs on the body.
        let expected = world
            .world_transform(body)
            .unwrap()
            .transform_point3(Vec3::new(-0.7, -0.3, 0.0));
        assert!(close(pos(&world, wheel), expected));
        world.despawn(s.root);
        assert!(world.world_transform(body).is_err());
        assert!(world.world_transform(wheel).is_err());
        assert_eq!(world.body_count().0, 0);
    }

    #[test]
    fn things_can_be_held_in_a_hand() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/model_viewer/assets/character.glb"
        );
        let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
        let limits = crate::limits::EngineLimits::default();
        let model = match AnimatedModel::load_with(path, examples, &limits) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skipped: no character at {path}: {e}");
                return;
            }
        };
        let mut world = World::default();
        let walker = world
            .spawn_animated_shared(Arc::new(model), Place::new(3.0, 0.0, 0.0))
            .unwrap();
        world.play_animation(walker, "walk").unwrap();
        let sword = world.spawn(cube());
        world
            .attach_to_bone(sword, walker, "Hand_R", Place::default())
            .unwrap();
        assert!(world
            .attach_to_bone(sword, walker, "tail", Place::default())
            .is_err());
        let lantern = world
            .add_lamp(Lamp::point(Vec3::ZERO, Color::WHITE, 2.0, 8.0))
            .unwrap();
        world
            .attach_lamp_to_bone(lantern, walker, "hand_l", Vec3::ZERO)
            .unwrap();

        let hand = world
            .bone_transform(walker, "hand_r")
            .unwrap()
            .w_axis
            .truncate();
        assert!(close(pos(&world, sword), hand));
        assert!(
            hand.y > 0.4 && hand.y < 1.6,
            "a hand is at hand height: {hand}"
        );

        // The hand swings as the walk plays, and the sword goes with it.
        let before = pos(&world, sword);
        world.tick_animations(0.4);
        world.tick_hierarchy();
        let after = pos(&world, sword);
        assert!(before.distance(after) > 0.02, "{before} {after}");
        let hand = world
            .bone_transform(walker, "hand_r")
            .unwrap()
            .w_axis
            .truncate();
        assert!(close(after, hand));
        let left = world
            .bone_transform(walker, "hand_l")
            .unwrap()
            .w_axis
            .truncate();
        assert!(close(world.lamp(lantern).unwrap().position, left));
    }
}

mod shapes_and_roll {
    use crate::color::rgb;
    use crate::mesh::Mesh;
    use crate::place::Place;
    use glam::Vec3;

    #[test]
    fn cylinders_are_closed_and_face_outwards() {
        let m = Mesh::cylinder(Vec3::ZERO, Vec3::X, 0.5, 2.0, 16, rgb(1, 2, 3))
            .unwrap()
            .build();
        let (lo, hi) = m
            .positions
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(a, b), p| (a.min(*p), b.max(*p)));
        assert!((hi.x - lo.x - 2.0).abs() < 1e-4 && (hi.y - 0.5).abs() < 1e-3);
        // Every triangle's winding agrees with its outward direction.
        for t in m.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| m.positions[t[k] as usize]);
            let n = (b - a).cross(c - a);
            let mid = (a + b + c) / 3.0;
            let out = if (mid.x.abs() - 1.0).abs() < 1e-4 {
                Vec3::X * mid.x
            } else {
                Vec3::new(0.0, mid.y, mid.z)
            };
            assert!(n.dot(out) > 0.0, "inward triangle at {mid}");
        }
        assert!(Mesh::cylinder(Vec3::ZERO, Vec3::ZERO, 1.0, 1.0, 8, rgb(0, 0, 0)).is_err());
        assert!(Mesh::cylinder(Vec3::ZERO, Vec3::Y, 1.0, 1.0, 2, rgb(0, 0, 0)).is_err());
    }

    #[test]
    fn roll_turns_around_forward() {
        let m = Place::new(0.0, 0.0, 0.0).with_roll_deg(90.0).to_matrix();
        assert!(m.transform_vector3(Vec3::X).abs_diff_eq(Vec3::Y, 1e-5));
        assert!(m.transform_vector3(Vec3::Z).abs_diff_eq(Vec3::Z, 1e-5));
        assert!(Place::new(0.0, 0.0, 0.0).roll_deg(f32::NAN).is_err());
    }
}

mod save_load {
    use crate::color::{rgb, Color};
    use crate::lamps::Lamp;
    use crate::mesh::Mesh;
    use crate::physics::{Body, PhysicsGround, PhysicsSettings};
    use crate::place::Place;
    use crate::prefab::Prefab;
    use crate::world::{EntityId, World};
    use glam::Vec3;
    use serde::{Deserialize, Serialize};

    fn cart(falls: bool) -> Prefab {
        let mut p = Prefab::new(if falls { "falling cart" } else { "cart" });
        let body = p.mesh(
            "body",
            Mesh::box_at(Vec3::ZERO, Vec3::new(1.0, 0.4, 2.0), rgb(150, 100, 60)).unwrap(),
            Place::new(0.0, 0.6, 0.0),
        );
        p.mesh_on(
            body,
            "wheel",
            Mesh::cylinder(Vec3::ZERO, Vec3::X, 0.3, 0.1, 10, rgb(60, 40, 30)).unwrap(),
            Place::new(0.6, -0.3, 0.7),
        );
        p.lamp_on(body, Lamp::point((0.0, 1.0, 0.0), Color::WHITE, 3.0, 8.0));
        if falls {
            p.body(body, Body::dynamic());
        }
        p
    }

    fn fresh() -> World {
        let mut world = World::default();
        world
            .set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(0.0),
                ..Default::default()
            })
            .unwrap();
        world.register_prefab(cart(false)).unwrap();
        world.register_prefab(cart(true)).unwrap();
        world
    }

    fn voxel_ground() -> crate::voxel_terrain::VoxelTerrain {
        let mut volume = crate::volume::Volume::new(0.5);
        volume.ensure_bounds(Vec3::new(-8.0, -4.0, -8.0), Vec3::new(8.0, 8.0, 8.0));
        let keys: Vec<_> = volume.chunk_keys().collect();
        volume.generate_chunks(&keys, |p| 2.0 - p.y);
        crate::voxel_terrain::VoxelTerrain::new(volume)
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Game {
        gold: u32,
        hero_cart: EntityId,
        name: String,
    }

    fn pos(world: &World, id: EntityId) -> Vec3 {
        world.world_transform(id).unwrap().w_axis.truncate()
    }

    #[test]
    fn a_world_comes_back_as_it_was_saved() {
        let mut world = fresh();
        world.set_voxel_terrain(Some(voxel_ground()));
        world.set_physical_sky(Some(crate::sky::PhysicalSky::at_hour(17.25)));
        let a = world
            .spawn_named("cart", Place::new(3.0, 0.0, 1.0).with_yaw_deg(30.0))
            .unwrap();
        let wheel = a.part("wheel").unwrap();
        world
            .set_place(wheel, Place::new(0.6, -0.3, 0.7).with_pitch_deg(77.0))
            .unwrap();
        let flying = world
            .spawn_named("falling cart", Place::new(-4.0, 9.0, 0.0))
            .unwrap();
        let fbody = flying.part("body").unwrap();
        world.set_body_velocity(fbody, (2.0, 1.0, 0.0)).unwrap();
        for _ in 0..10 {
            world.tick_physics(1.0 / 60.0);
            world.tick_hierarchy();
        }
        let free = world
            .add_lamp(Lamp::fire((1.0, 0.5, -2.0)).intensity(4.5))
            .unwrap();
        let riding = world
            .add_lamp(Lamp::point(Vec3::ZERO, rgb(10, 200, 30), 2.0, 6.0))
            .unwrap();
        world
            .attach_lamp(riding, a.part("body").unwrap(), (0.0, 2.0, 0.0))
            .unwrap();
        world.lamp(a.lamps[0]).unwrap();
        let mut look = *world.lamp(a.lamps[0]).unwrap();
        look.intensity = 9.0;
        world.set_lamp(a.lamps[0], look).unwrap();
        world
            .voxel_terrain_mut()
            .unwrap()
            .dig_sphere(Vec3::new(0.0, 2.0, 0.0), 1.5);
        assert!(!world
            .voxel_terrain()
            .unwrap()
            .is_solid(Vec3::new(0.0, 1.2, 0.0)));
        assert!(world
            .voxel_terrain()
            .unwrap()
            .is_solid(Vec3::new(5.0, 1.2, 5.0)));

        let game = Game {
            gold: 42,
            hero_cart: a.root,
            name: "Ærnd".into(),
        };
        let bytes = world.save_to_bytes(&game).unwrap();

        // A new session: build the world as at startup, then load.
        let mut world2 = fresh();
        world2.set_voxel_terrain(Some(voxel_ground()));
        let stray = world2
            .spawn_named("cart", Place::new(50.0, 0.0, 0.0))
            .unwrap();
        world2.add_lamp(Lamp::fire(Vec3::ZERO)).unwrap();
        let loaded = world2.load_from_bytes(&bytes).unwrap();
        assert!(
            world2.world_transform(stray.root).is_err(),
            "load replaces what was there"
        );

        let got: Game = loaded.game().unwrap();
        assert_eq!(got.gold, 42);
        assert_eq!(got.name, "Ærnd");
        let a2 = loaded.entity(got.hero_cart).unwrap();
        assert_eq!(world2.prefab_of(a2), Some("cart"));
        assert!(world2
            .world_transform(a2)
            .unwrap()
            .abs_diff_eq(world.world_transform(a.root).unwrap(), 1e-5));
        let wheel2 = loaded.entity(wheel).unwrap();
        assert!(world2
            .world_transform(wheel2)
            .unwrap()
            .abs_diff_eq(world.world_transform(wheel).unwrap(), 1e-5));
        assert_eq!(
            world2.parent(wheel2),
            loaded.entity(a.part("body").unwrap())
        );

        // The flying cart is where it was, still flying the same way.
        let fbody2 = loaded.entity(fbody).unwrap();
        assert!(pos(&world2, fbody2).distance(pos(&world, fbody)) < 1e-4);
        let (v1, v2) = (
            world.body_velocity(fbody).unwrap(),
            world2.body_velocity(fbody2).unwrap(),
        );
        assert!(v1.distance(v2) < 1e-3, "{v1} {v2}");

        // Lamps: the free fire, the one riding on the cart, the prefab's own
        // (with its changed brightness).
        assert_eq!(world2.lamps().count(), 4);
        let free2 = loaded.lamp(free).unwrap();
        assert_eq!(world2.lamp(free2).unwrap().intensity, 4.5);
        assert!(world2.lamp(free2).unwrap().flicker > 0.0);
        let riding2 = loaded.lamp(riding).unwrap();
        world2.set_place(a2, Place::new(10.0, 0.0, 0.0)).unwrap();
        world2.tick_hierarchy();
        assert!(
            world2
                .lamp(riding2)
                .unwrap()
                .position
                .distance(Vec3::new(10.0, 2.6, 0.0))
                < 1e-4
        );
        let prefab_lamp = world2.lamps().find(|(id, _, _)| {
            *id != free2 && *id != riding2 && world2.lamp(*id).unwrap().intensity == 9.0
        });
        assert!(prefab_lamp.is_some());

        // Dug terrain and time of day.
        assert!(!world2
            .voxel_terrain()
            .unwrap()
            .is_solid(Vec3::new(0.0, 1.2, 0.0)));
        assert!(world2
            .voxel_terrain()
            .unwrap()
            .is_solid(Vec3::new(5.0, 1.2, 5.0)));
        assert_eq!(world2.physical_sky().unwrap().hour, 17.25);

        // Saving the loaded world again gives the same world once more.
        let again = world2.save_to_bytes(&got).unwrap();
        let mut world3 = fresh();
        world3.set_voxel_terrain(Some(voxel_ground()));
        world3.load_from_bytes(&again).unwrap();
        assert_eq!(world3.lamps().count(), 4);
        assert!(!world3
            .voxel_terrain()
            .unwrap()
            .is_solid(Vec3::new(0.0, 1.2, 0.0)));
    }

    #[test]
    fn parts_destroyed_before_saving_stay_destroyed() {
        let mut world = fresh();
        let a = world
            .spawn_named("cart", Place::new(1.0, 0.0, 0.0))
            .unwrap();
        world.despawn(a.part("wheel").unwrap());
        let bytes = world.save_to_bytes(&a.part("body").unwrap()).unwrap();
        let mut world2 = fresh();
        let loaded = world2.load_from_bytes(&bytes).unwrap();
        let body = loaded.entity(loaded.game().unwrap()).unwrap();
        assert!(world2.children(body).is_empty(), "the wheel stays gone");
        assert!(world2.world_transform(body).is_ok());
    }

    #[test]
    fn bad_saves_are_refused_without_touching_the_world() {
        let mut world = fresh();
        let keep = world.spawn_named("cart", Place::default()).unwrap();
        assert!(world.load_from_bytes(b"hello").is_err());
        let mut damaged = world.save_to_bytes(&()).unwrap();
        let mid = damaged.len() / 2;
        damaged.truncate(mid);
        assert!(world.load_from_bytes(&damaged).is_err());

        // A save needing a prefab this world doesn't know.
        let mut other = World::default();
        let mut boat = Prefab::new("boat");
        boat.mesh(
            "hull",
            Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(1, 1, 1)).unwrap(),
            Place::default(),
        );
        other.spawn_prefab(&boat, Place::default()).unwrap();
        let bytes = other.save_to_bytes(&()).unwrap();
        let err = world.load_from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("boat"), "{err}");
        assert!(world.world_transform(keep.root).is_ok());

        // Files on disk, written safely.
        let dir = std::env::temp_dir().join(format!("engine-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slot.save");
        world.save(&path, &7u32).unwrap();
        let loaded = fresh().load(&path).unwrap();
        assert_eq!(loaded.game::<u32>().unwrap(), 7);
        assert!(loaded.game::<String>().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}

mod animation_blending {
    use crate::anim::{AnimatedModel, AnimationProfile, FootIk, Locomotion};
    use crate::color::rgb;
    use crate::mesh::Mesh;
    use crate::physics::{Body, PhysicsGround, PhysicsSettings};
    use crate::place::Place;
    use crate::world::{EntityId, World};
    use glam::Vec3;
    use std::sync::Arc;

    fn character() -> Option<Arc<AnimatedModel>> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/model_viewer/assets/character.glb"
        );
        let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
        match AnimatedModel::load_with(path, examples, &crate::limits::EngineLimits::default()) {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => {
                eprintln!("skipped: {e}");
                None
            }
        }
    }

    fn joint(world: &World, id: EntityId, bone: &str) -> Vec3 {
        world.bone_transform(id, bone).unwrap().w_axis.truncate()
    }

    #[test]
    fn switching_clips_fades_instead_of_jumping() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        let id = world
            .spawn_animated_shared(model.clone(), Place::default())
            .unwrap();
        world.play_animation(id, "walk").unwrap();
        world.tick_animations(0.37);
        let before = joint(&world, id, "hand_r");
        world.play_animation(id, "idle").unwrap();
        let right_after = joint(&world, id, "hand_r");
        assert!(
            before.distance(right_after) < 1e-4,
            "no pop: {before} {right_after}"
        );

        // Halfway through the fade the hand is between the two poses.
        let mut idle_only = World::default();
        let solo = idle_only
            .spawn_animated_shared(model.clone(), Place::default())
            .unwrap();
        idle_only.set_animation_blend(solo, 0.0).unwrap();
        idle_only.play_animation(solo, "idle").unwrap();
        world.tick_animations(0.25);
        idle_only.tick_animations(0.25);
        let done = joint(&world, id, "hand_r");
        assert!(
            done.distance(joint(&idle_only, solo, "hand_r")) < 1e-4,
            "fade finished"
        );

        // Blend 0 switches at once.
        let mut snap = World::default();
        let s = snap.spawn_animated_shared(model, Place::default()).unwrap();
        snap.set_animation_blend(s, 0.0).unwrap();
        snap.play_animation(s, "walk").unwrap();
        snap.tick_animations(0.37);
        let a = joint(&snap, s, "hand_r");
        snap.play_animation(s, "idle").unwrap();
        assert!(a.distance(joint(&snap, s, "hand_r")) > 1e-3);
        assert!(snap.set_animation_blend(s, -1.0).is_err());
    }

    #[test]
    fn walk_and_run_follow_the_real_speed() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        let id = world
            .spawn_animated_shared(model, Place::default())
            .unwrap();
        world
            .configure_animation(
                id,
                AnimationProfile::new()
                    .idle("idle")
                    .walk("walk")
                    .run("jog")
                    .walk_speed(1.4)
                    .run_speed(3.0),
            )
            .unwrap();
        world
            .set_locomotion(id, Locomotion::Moving { speed_mps: 0.7 })
            .unwrap();
        let t0 = world.animated_entity(id).unwrap().animator.time();
        world.tick_animations(0.2);
        let walked = world.animated_entity(id).unwrap().animator.time() - t0;
        assert!(
            (walked - 0.1).abs() < 1e-4,
            "half speed walk plays at half rate: {walked}"
        );

        // Switching to the run keeps the step phase.
        world.tick_animations(0.3);
        let a = &world.animated_entity(id).unwrap().animator;
        let phase = a.time() / a.clip_duration().unwrap();
        world
            .set_locomotion(id, Locomotion::Moving { speed_mps: 3.0 })
            .unwrap();
        let a = &world.animated_entity(id).unwrap().animator;
        assert_eq!(a.clip_name(), "jog");
        assert!((a.time() / a.clip_duration().unwrap() - phase).abs() < 1e-4);
    }

    #[test]
    fn feet_stand_on_steps_and_the_hips_drop_for_a_lower_foot() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        world
            .set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(0.0),
                ..Default::default()
            })
            .unwrap();
        let id = world
            .spawn_animated_shared(model.clone(), Place::default())
            .unwrap();
        world.play_animation(id, "idle").unwrap();
        let flat_l = joint(&world, id, "foot_l");
        let flat_r = joint(&world, id, "foot_r");
        let flat_hips = joint(&world, id, "pelvis");

        // A 20 cm step under the left foot only.
        let step = world.spawn(
            Mesh::box_at(
                Vec3::new(flat_l.x, 0.1, flat_l.z),
                Vec3::new(0.18, 0.2, 0.5),
                rgb(90, 90, 90),
            )
            .unwrap(),
        );
        world.add_body(step, Body::fixed()).unwrap();
        assert!(world.set_foot_ik(id, Some(FootIk::default())).is_ok());
        for _ in 0..60 {
            world.tick_physics(1.0 / 60.0);
            world.tick_animations(1.0 / 60.0);
        }
        let (l, r) = (joint(&world, id, "foot_l"), joint(&world, id, "foot_r"));
        assert!(
            (l.y - flat_l.y - 0.2).abs() < 0.02,
            "left foot up the step: {} -> {}",
            flat_l.y,
            l.y
        );
        assert!(
            (r.y - flat_r.y).abs() < 0.02,
            "right foot stays: {} -> {}",
            flat_r.y,
            r.y
        );
        assert!((joint(&world, id, "pelvis").y - flat_hips.y).abs() < 0.02);

        // Now stand on a block with the right foot hanging over an edge 15 cm lower.
        let mut world = World::default();
        world
            .set_physics_settings(PhysicsSettings {
                ground: PhysicsGround::Plane(-0.15),
                ..Default::default()
            })
            .unwrap();
        let id = world
            .spawn_animated_shared(model, Place::default())
            .unwrap();
        world.play_animation(id, "idle").unwrap();
        let block = world.spawn(
            Mesh::box_at(
                Vec3::new(flat_l.x * 0.5 + 0.3 * flat_l.x.signum(), -0.075, 0.0),
                Vec3::new(0.6 + flat_l.x.abs(), 0.15, 1.0),
                rgb(90, 90, 90),
            )
            .unwrap(),
        );
        world.add_body(block, Body::fixed()).unwrap();
        world.set_foot_ik(id, Some(FootIk::default())).unwrap();
        for _ in 0..60 {
            world.tick_physics(1.0 / 60.0);
            world.tick_animations(1.0 / 60.0);
        }
        let (l, r) = (joint(&world, id, "foot_l"), joint(&world, id, "foot_r"));
        assert!(
            (r.y - flat_r.y + 0.15).abs() < 0.02,
            "right foot down: {} -> {}",
            flat_r.y,
            r.y
        );
        assert!(
            (l.y - flat_l.y).abs() < 0.02,
            "left foot stays: {} -> {}",
            flat_l.y,
            l.y
        );
        assert!(
            (joint(&world, id, "pelvis").y - flat_hips.y + 0.15).abs() < 0.02,
            "hips drop"
        );

        world.set_foot_ik(id, None).unwrap();
    }
}
