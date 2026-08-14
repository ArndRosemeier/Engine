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
    assert_eq!(level.eye, Vec3::new(3.0, 2.0, -1.0));
    let forward = (level.target - level.eye).normalize();
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
fn instance_submit_defaults_to_cpu_and_switches_explicitly() {
    use crate::world::{InstanceSubmit, World};

    let mut world = World::new();
    assert_eq!(world.instance_submit(), InstanceSubmit::CpuIndexed);
    world.set_instance_submit(InstanceSubmit::GpuIndirect);
    assert_eq!(world.instance_submit(), InstanceSubmit::GpuIndirect);
    world.set_instance_submit(InstanceSubmit::CpuIndexed);
    assert_eq!(world.instance_submit(), InstanceSubmit::CpuIndexed);
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
