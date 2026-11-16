use super::*;

const DT: f32 = 1.0 / 60.0;

fn tank(cells: u32, h: f32) -> Fluid {
    let size = Vec3::splat(cells as f32 * h);
    Fluid::new(FluidSettings::new(Vec3::ZERO, size, h)).unwrap()
}

/// A tank whose water calms quickly, for tests about where it ends up.
fn calm_tank(cells: u32, h: f32) -> Fluid {
    let size = Vec3::splat(cells as f32 * h);
    Fluid::new(FluidSettings::new(Vec3::ZERO, size, h).drag(1.0)).unwrap()
}

fn run(f: &mut Fluid, seconds: f32, accel: impl Fn(Vec3) -> Vec3) {
    for _ in 0..(seconds / DT).round() as usize {
        f.step(DT, &accel);
    }
}

fn max_along(f: &Fluid, axis: Vec3) -> f32 {
    f.particles().map(|p| p.dot(axis)).fold(f32::MIN, f32::max)
}

#[test]
#[ignore = "full"]
fn a_column_of_water_slumps_and_settles_without_losing_volume() {
    // 3 m tank of 0.25 m cells; a 1 x 2 x 3 m column in one corner.
    let mut f = calm_tank(12, 0.25);
    let added = f.fill_box(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(added, 4 * 8 * 12 * 8);
    let volume = f.volume();
    run(&mut f, 4.0, |_| Vec3::NEG_Y * 9.81);
    assert_eq!(f.particle_count(), added, "no water lost");
    assert_eq!(f.take_lost(), 0);
    assert!((f.volume() - volume).abs() < 1e-3);
    // 6 m³ over a 9 m² floor stands about 0.67 m deep.
    let top = max_along(&f, Vec3::Y);
    assert!(top < 1.2, "surface at {top}");
    assert!(f.max_speed() < 1.5, "still moving at {}", f.max_speed());
    // It spread across the floor.
    assert!(max_along(&f, Vec3::X) > 2.5);
    // And did not squeeze: water cells hold about the particles' volume.
    let wet = f.sim.cells.iter().filter(|c| **c == Cell::Water).count() as f32;
    let cells_of_water = f.particle_count() as f32 / 8.0;
    assert!(
        wet > cells_of_water * 0.8 && wet < cells_of_water * 1.6,
        "{wet} cells for {cells_of_water}"
    );
}

#[test]
#[ignore = "full"]
fn a_sideways_pull_piles_water_against_a_wall() {
    let mut f = calm_tank(10, 0.3);
    // Not sealed against the far walls, so air can follow it.
    f.fill_box(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.2, 1.8, 3.0));
    run(&mut f, 3.0, |_| Vec3::X * 9.81);
    let c = f.center().unwrap();
    assert!(c.x > 2.2, "centre {c}");
    // Nothing pulls in y: the water fills the wall's height, not the floor.
    assert!(max_along(&f, Vec3::Y) > 2.0);
}

#[test]
#[ignore = "full"]
fn without_a_pull_water_drifts_as_one_blob() {
    let mut f = tank(16, 0.25);
    f.fill_ball(Vec3::new(1.5, 2.0, 2.0), 0.8);
    for v in f.sim.vel.iter_mut() {
        *v = Vec3::new(1.0, 0.0, 0.0);
    }
    let start = f.center().unwrap();
    run(&mut f, 1.0, |_| Vec3::ZERO);
    let c = f.center().unwrap();
    assert!(
        (c.x - start.x - 1.0).abs() < 0.15,
        "moved {}",
        c.x - start.x
    );
    assert!((c.y - start.y).abs() < 0.1 && (c.z - start.z).abs() < 0.1);
    let spread = f.particles().map(|p| p.distance(c)).fold(0.0, f32::max);
    assert!(spread < 1.2, "blob spread to {spread}");
}

#[test]
#[ignore = "full"]
fn a_point_pull_gathers_water_into_a_ball() {
    let mut f = calm_tank(16, 0.25);
    let middle = Vec3::splat(2.0);
    f.fill_box(Vec3::new(0.25, 0.25, 0.25), Vec3::new(1.5, 3.75, 3.75));
    run(&mut f, 6.0, |p| (middle - p).normalize_or_zero() * 9.81);
    let c = f.center().unwrap();
    assert!(c.distance(middle) < 0.3, "centre {c}");
    // About 4 m³ gathers into a ball of about 1 m radius.
    let far = f
        .particles()
        .map(|p| p.distance(middle))
        .fold(0.0, f32::max);
    assert!(far < 1.8, "reaches {far}");
}

#[test]
#[ignore = "full"]
fn rock_holds_water_in_a_basin() {
    let mut f = tank(12, 0.25);
    // A raised floor with a pit in the middle third.
    f.mark_solid(|p| p.y < 1.0 && !(1.0..2.0).contains(&p.x));
    f.fill_box(Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.5, 3.0));
    run(&mut f, 3.0, |_| Vec3::NEG_Y * 9.81);
    assert!(f.particles().all(|p| !f.is_solid_at(p)));
    let top = max_along(&f, Vec3::Y);
    assert!(top < 1.0, "the pit holds it, surface at {top}");
}

#[test]
#[ignore = "full"]
fn the_same_water_gives_the_same_frames() {
    let make = || {
        let mut f = tank(10, 0.3);
        f.fill_box(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        run(&mut f, 1.0, |p| Vec3::new(0.0, -9.81, 0.5 * p.x));
        f
    };
    let (a, b) = (make(), make());
    assert!(a.particles().zip(b.particles()).all(|(p, q)| p == q));
}

#[test]
fn the_surface_is_a_closed_mesh_around_the_water() {
    let mut f = tank(12, 0.25);
    f.fill_box(Vec3::new(0.5, 0.5, 0.5), Vec3::new(2.5, 1.5, 2.5));
    let mesh = f.surface_mesh().unwrap();
    assert!(mesh.face_count() > 50);
    let (lo, hi) = mesh.bounds().unwrap();
    assert!(lo.y > 0.25 && hi.y < 1.8, "{lo} {hi}");
    assert!(lo.x > 0.25 && hi.x < 2.75);
    // An empty box has no surface.
    assert_eq!(tank(4, 0.5).surface_mesh().unwrap().face_count(), 0);
}

#[test]
fn bad_settings_are_named() {
    let e = Fluid::new(FluidSettings::new(Vec3::ZERO, Vec3::splat(100.0), 0.1)).unwrap_err();
    assert!(e.to_string().contains("cells per side"), "{e}");
    let e = Fluid::new(FluidSettings::new(Vec3::ZERO, Vec3::ONE, 0.0)).unwrap_err();
    assert!(e.to_string().contains("cell_size"), "{e}");
}

#[test]
#[ignore = "full"]
fn the_world_steps_water_with_its_field_and_floats_wood_in_it() {
    use crate::physics::{Body, BodyShape, FieldBody, FieldProbe, PhysicsGround, PhysicsSettings};
    use crate::prelude::*;
    let mut world = World::new();
    world
        .set_physics_settings(PhysicsSettings {
            ground: PhysicsGround::None,
            ..Default::default()
        })
        .unwrap();
    // Down is -x here, and the field tells water from bodies.
    world.set_gravity_field(Some(std::sync::Arc::new(|p: &FieldProbe| {
        assert!(matches!(p.kind, FieldBody::Water | FieldBody::Body));
        Vec3::NEG_X * 9.81
    })));
    let mut pool = Fluid::new(FluidSettings::new(Vec3::ZERO, Vec3::splat(4.0), 0.25)).unwrap();
    pool.fill_box(Vec3::ZERO, Vec3::new(2.0, 4.0, 4.0));
    let id = world.add_fluid(pool).unwrap();
    assert!(world.fluid_entity(id).is_some(), "the surface is drawn");
    // A wooden block dropped in from the "top" (+x).
    let block =
        world.spawn(Shape::box_at((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), rgb(150, 110, 70)).unwrap());
    world.set_place(block, Place::new(3.2, 2.0, 2.0)).unwrap();
    world
        .add_body(
            block,
            Body::dynamic()
                .density(0.6)
                .shape(BodyShape::Box(Vec3::splat(0.25))),
        )
        .unwrap();
    for _ in 0..(4.0 * 60.0) as usize {
        world.tick_physics(1.0 / 60.0);
    }
    let x = world.world_transform(block).unwrap().w_axis.x;
    // Water ends at x = 2; floating wood rides with part of it above.
    assert!(x > 1.8 && x < 2.4, "block at x = {x}");
    let f = world.fluid(id).unwrap();
    assert!(f.center().unwrap().x < 1.2);
    assert!(world.remove_fluid(id).is_some());
    assert!(world.fluid(id).is_none());
}

#[test]
#[ignore]
fn bench_lake() {
    let mut f = Fluid::new(FluidSettings::new(
        Vec3::ZERO,
        Vec3::new(24.0, 8.0, 24.0),
        0.5,
    ))
    .unwrap();
    f.fill_box(Vec3::ZERO, Vec3::new(12.0, 4.0, 24.0));
    let t = std::time::Instant::now();
    for _ in 0..30 {
        f.step(DT, |_| Vec3::NEG_Y * 9.81);
    }
    let step = t.elapsed().as_secs_f32() / 30.0;
    let t = std::time::Instant::now();
    let m = f.surface_mesh().unwrap();
    eprintln!(
        "particles {} step {:.1} ms mesh {:.1} ms faces {}",
        f.particle_count(),
        step * 1000.0,
        t.elapsed().as_secs_f32() * 1000.0,
        m.face_count()
    );
}
