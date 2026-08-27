//! Walk through a doorway into a house that is not inside the house mesh.
//!
//! The yard and interior are separate [`SpaceId`] pockets linked by one
//! [`engine::Portal`]. The renderer draws the interior recursively through the
//! door opening (stencil-masked, correct from any angle).
//!
//! Controls: W/S walk · A/D turn · Shift sprint · Esc quit
use engine::prelude::*;

fn wall(center: (f32, f32, f32), size: (f32, f32, f32), color: Color) -> Mesh {
    Shape::box_at(center, size, color).unwrap()
}

/// Must match the gaps left in the wall boxes and lintel above the door.
const DOOR_W: f32 = 1.2;
const DOOR_H: f32 = 2.2;
const DOOR_Y: f32 = DOOR_H * 0.5;

/// `ENGINE_PROBE="x,y,z,yaw[,house]"` — optional fixed pose for screenshot probes.
fn apply_probe(pos: &mut Vec3, yaw: &mut f32, world: &mut World, house: SpaceId) {
    let Ok(raw) = std::env::var("ENGINE_PROBE") else {
        return;
    };
    let mut parts = raw.split(',');
    let Some(x) = parts.next().and_then(|s| s.parse().ok()) else {
        return;
    };
    let Some(y) = parts.next().and_then(|s| s.parse().ok()) else {
        return;
    };
    let Some(z) = parts.next().and_then(|s| s.parse().ok()) else {
        return;
    };
    let Some(yaw_deg) = parts.next().and_then(|s| s.parse().ok()) else {
        return;
    };
    *pos = Vec3::new(x, y, z);
    *yaw = yaw_deg;
    if parts.next().is_some_and(|s| s.trim() == "house") {
        let _ = world.live_in(house);
    } else {
        let _ = world.live_in(SpaceId::DEFAULT);
    }
}

fn main() -> EngineResult<()> {
    let mut pos = Vec3::new(0.0, 1.6, -8.0);
    let mut yaw = 0.0_f32;

    Engine::run("portals", move |world, frame| {
        let house = if frame.first {
            world.set_clear_color(rgb(145, 195, 235));
            world.set_sun((0.45, 1.0, 0.2), 0.26);
            world.set_portal_recursion(3);

            let plaster = rgb(220, 158, 102);
            let grass = rgb(87, 143, 77);
            let wood = rgb(89, 56, 31);
            let stone = rgb(140, 128, 107);

            world.spawn(wall((0.0, -0.05, 0.0), (28.0, 0.1, 28.0), grass));
            world.spawn(wall((0.0, 1.5, 3.0), (6.0, 3.0, 0.2), plaster));
            world.spawn(wall((-3.0, 1.5, 0.0), (0.2, 3.0, 6.0), plaster));
            world.spawn(wall((3.0, 1.5, 0.0), (0.2, 3.0, 6.0), plaster));
            world.spawn(wall((-1.8, 1.5, -3.0), (2.4, 3.0, 0.2), plaster));
            world.spawn(wall((1.8, 1.5, -3.0), (2.4, 3.0, 0.2), plaster));
            world.spawn(wall((0.0, 2.6, -3.0), (DOOR_W, 0.8, 0.2), plaster));
            world.spawn(wall((0.0, 3.15, 0.0), (6.4, 0.2, 6.4), rgb(184, 46, 41)));
            world.spawn(wall((0.0, 0.02, -4.2), (1.4, 0.04, 2.4), stone));

            let door_out = world
                .place(
                    Mesh::opening(DOOR_W, DOOR_H).unwrap(),
                    Place::new(0.0, DOOR_Y, -3.0).with_yaw_deg(180.0),
                )
                .unwrap();

            let house = world.space("house").unwrap();
            world.in_space(house).unwrap();

            let floor = rgb(168, 112, 72);
            let cream = rgb(232, 214, 186);
            world.spawn(wall((0.0, 0.0, 0.0), (8.0, 0.1, 8.0), floor));
            world.spawn(wall((0.0, 3.0, 0.0), (8.0, 0.1, 8.0), cream));
            world.spawn(wall((0.0, 1.5, 4.0), (8.0, 3.0, 0.2), cream));
            world.spawn(wall((-4.0, 1.5, 0.0), (0.2, 3.0, 8.0), cream));
            world.spawn(wall((4.0, 1.5, 0.0), (0.2, 3.0, 8.0), cream));
            world.spawn(wall((-2.3, 1.5, -4.0), (3.4, 3.0, 0.2), cream));
            world.spawn(wall((2.3, 1.5, -4.0), (3.4, 3.0, 0.2), cream));
            world.spawn(wall((0.0, 2.6, -4.0), (DOOR_W, 0.8, 0.2), cream));
            world.spawn(wall((0.0, 0.45, 1.2), (1.4, 0.7, 0.8), wood));
            world.spawn(wall((-1.6, 0.7, 2.4), (0.5, 1.4, 0.5), rgb(70, 90, 140)));
            world.spawn(wall((2.2, 1.2, 0.4), (0.08, 1.2, 1.6), rgb(140, 191, 230)));

            let door_in = world
                .place(
                    Mesh::opening(DOOR_W, DOOR_H).unwrap(),
                    Place::new(0.0, DOOR_Y, -4.0),
                )
                .unwrap();

            world.in_space(SpaceId::DEFAULT).unwrap();
            world
                .create_portal(door_out, door_in, PortalSettings::TELEPORTING)
                .unwrap();
            world.live_in(SpaceId::DEFAULT).unwrap();
            house
        } else {
            world.space("house").unwrap()
        };

        apply_probe(&mut pos, &mut yaw, world, house);

        if frame.first {
            world.look_first_person(pos, yaw, 0.0);
            return Ok(());
        }

        yaw += frame.input.axis(Key::D, Key::A) * 90.0 * frame.dt;
        let along = frame.input.axis(Key::S, Key::W);
        if along != 0.0 {
            let speed = if frame.input.down(Key::Shift) {
                8.0
            } else {
                4.0
            };
            let yaw_rad = yaw.to_radians();
            let facing = Vec3::new(yaw_rad.sin(), 0.0, yaw_rad.cos());
            pos += facing * along * speed * frame.dt;
        }
        pos.y = 1.6;

        world.travel(&mut pos, &mut yaw);
        world.look_first_person(pos, yaw, 0.0);
        Ok(())
    })
}
