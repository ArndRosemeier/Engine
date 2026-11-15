//! 3D water that pours, fills a basin, floats crates and follows a sideways pull
//!
//! A glass tank on a stone floor. A block of water slumps across it, a
//! spring pours more in, wooden crates float, and after five seconds the
//! pull tilts sideways (a gravity field) so the water climbs one wall, then
//! it settles back. Runs by itself; Space pours a burst.
use engine::prelude::*;
use std::sync::Arc;

const TANK: Vec3 = Vec3::new(8.0, 5.0, 8.0);
const CELL: f32 = 0.25;

#[derive(Default)]
struct Demo {
    water: Option<FluidId>,
    crates: Vec<EntityId>,
    start_volume: f32,
    tilted: bool,
}

fn glass_wall(world: &mut World, center: Vec3, size: Vec3) -> EngineResult<()> {
    let wall = world.spawn(Shape::box_at(
        (0.0, 0.0, 0.0),
        size,
        rgba(200, 225, 255, 40),
    )?);
    world.set_place(wall, Place::new(center.x, center.y, center.z))?;
    world.add_body(wall, Body::fixed().shape(BodyShape::Box(size * 0.5)))?;
    world.set_casts_shadow(wall, false)
}

fn setup(world: &mut World, demo: &mut Demo) -> EngineResult<()> {
    world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
    world.set_physics_settings(PhysicsSettings {
        ground: PhysicsGround::None,
        ..Default::default()
    })?;
    // Stone floor under the tank.
    let floor = world.spawn(Shape::box_at(
        (0.0, 0.0, 0.0),
        (14.0, 0.4, 14.0),
        rgb(120, 115, 105),
    )?);
    world.set_place(floor, Place::new(0.0, -0.2, 0.0))?;
    world.add_body(
        floor,
        Body::fixed().shape(BodyShape::Box(Vec3::new(7.0, 0.2, 7.0))),
    )?;
    // Glass walls just outside the water box.
    let (hx, hy, hz) = (TANK.x * 0.5, TANK.y * 0.5, TANK.z * 0.5);
    let t = 0.1;
    glass_wall(
        world,
        Vec3::new(-hx - t, hy, 0.0),
        Vec3::new(2.0 * t, TANK.y, TANK.z),
    )?;
    glass_wall(
        world,
        Vec3::new(hx + t, hy, 0.0),
        Vec3::new(2.0 * t, TANK.y, TANK.z),
    )?;
    glass_wall(
        world,
        Vec3::new(0.0, hy, -hz - t),
        Vec3::new(TANK.x, TANK.y, 2.0 * t),
    )?;
    glass_wall(
        world,
        Vec3::new(0.0, hy, hz + t),
        Vec3::new(TANK.x, TANK.y, 2.0 * t),
    )?;
    // A lid, so a crate riding a wave cannot leave: thin glass, with a
    // thick collider above it so nothing squeezes through.
    let lid = world.spawn(Shape::box_at(
        (0.0, 0.0, 0.0),
        (TANK.x + 4.0 * t, 2.0 * t, TANK.z + 4.0 * t),
        rgba(200, 225, 255, 40),
    )?);
    world.set_place(lid, Place::new(0.0, TANK.y + t, 0.0))?;
    world.set_casts_shadow(lid, false)?;
    let lid_body = world.spawn(Shape::box_at(
        (0.0, 0.0, 0.0),
        (TANK.x + 2.0, 2.0, TANK.z + 2.0),
        rgba(0, 0, 0, 0),
    )?);
    world.set_place(lid_body, Place::new(0.0, TANK.y + 1.0, 0.0))?;
    world.set_casts_shadow(lid_body, false)?;
    world.add_body(
        lid_body,
        Body::fixed().shape(BodyShape::Box(Vec3::new(hx + 1.0, 1.0, hz + 1.0))),
    )?;

    // The water: a box exactly filling the tank, a rock ledge inside it and
    // a block of water against the far wall.
    let origin = Vec3::new(-hx, 0.0, -hz);
    let mut water = Fluid::new(FluidSettings::new(origin, TANK, CELL).mesh_every(2))?;
    water.mark_solid(|p| p.x > 1.5 && p.z > 1.5 && p.y < 1.25);
    water.fill_box(Vec3::new(-hx, 0.0, -hz), Vec3::new(-1.0, 3.0, hz));
    demo.start_volume = water.volume();
    let ledge = world.spawn(Shape::box_at(
        (0.0, 0.0, 0.0),
        (2.5, 1.25, 2.5),
        rgb(110, 100, 90),
    )?);
    world.set_place(ledge, Place::new(2.75, 0.625, 2.75))?;
    world.add_body(
        ledge,
        Body::fixed().shape(BodyShape::Box(Vec3::new(1.25, 0.625, 1.25))),
    )?;
    demo.water = Some(world.add_fluid(water)?);

    // Wooden crates that fall in and float.
    for (k, (x, z)) in [(-2.5, -1.0), (0.5, 0.5), (-1.0, 2.5)]
        .into_iter()
        .enumerate()
    {
        let c = world.spawn(Shape::box_at(
            (0.0, 0.0, 0.0),
            (0.6, 0.6, 0.6),
            rgb(160, 115, 70),
        )?);
        world.set_place(
            c,
            Place::new(x, 4.0 + k as f32 * 0.5, z).with_yaw_deg(20.0 * k as f32),
        )?;
        world.add_body(
            c,
            Body::dynamic()
                .density(0.6)
                .ccd(true)
                .shape(BodyShape::Box(Vec3::splat(0.3))),
        )?;
        demo.crates.push(c);
    }
    Ok(())
}

fn main() -> EngineResult<()> {
    let mut demo = Demo::default();
    let settings = Settings::default()
        .window(1280, 720)
        .quality(Quality::High)
        .vsync(false);
    Engine::run_with_settings("fluids", settings, move |world, frame| {
        if frame.first {
            setup(world, &mut demo)?;
        }
        let Some(id) = demo.water else { return Ok(()) };

        // A spring pours for the first three seconds; Space pours a burst.
        if frame.time < 3.0 || frame.input.pressed(Key::Space) {
            let count = if frame.input.pressed(Key::Space) {
                2000
            } else {
                60
            };
            if let Some(f) = world.fluid_mut(id) {
                f.pour(
                    Vec3::new(2.5, 4.3, -2.5),
                    0.3,
                    Vec3::new(-1.5, -2.0, 0.0),
                    count,
                );
            }
        }

        // From second 5 to 10 the pull leans toward +x: a field, which the
        // water, the crates and everything else follow.
        let tilt = (5.0..10.0).contains(&frame.time);
        if tilt != demo.tilted {
            demo.tilted = tilt;
            world.set_gravity_field(if tilt {
                Some(Arc::new(|_: &FieldProbe| Vec3::new(0.6, -0.8, 0.0) * 9.81))
            } else {
                None
            });
        }

        world.look_orbit(
            Vec3::new(0.0, 1.2, 0.0),
            15.0,
            210.0 + frame.time * 3.0,
            28.0,
        );

        let f = world.fluid(id).expect("added");
        let volume = f.volume();
        let particles = f.particle_count();
        let centre = f.center().unwrap_or(Vec3::ZERO);
        frame.ui.text(
            Anchor::TopLeft,
            format!("Water: {volume:.1} m³ ({particles} particles)"),
        );
        frame.ui.text(
            Anchor::TopLeft,
            if tilt {
                "Pull leans sideways"
            } else {
                "Pull straight down"
            },
        );
        frame.ui.text_styled(
            Anchor::Bottom,
            "Space pours a burst",
            16.0,
            rgb(230, 230, 230),
        );

        // Self-checks for verify runs.
        world.check("water is not lost", volume >= demo.start_volume * 0.98);
        let lowest = demo
            .crates
            .iter()
            .filter_map(|c| world.world_transform(*c).ok())
            .map(|m| m.w_axis.y)
            .fold(f32::MAX, f32::min);
        world.check("crates stay in the tank", lowest > 0.2);
        if (4.5..5.0).contains(&frame.time) {
            world.check("crates float above the floor", lowest > 0.5);
        }
        if (8.0..10.0).contains(&frame.time) {
            world.check("water leans toward +x", centre.x > 0.5);
        }
        world.report("water_m3", volume);
        world.report("particles", particles);
        world.report("water_centre_x", centre.x);
        world.report("lowest_crate", lowest);
        Ok(())
    })
}
