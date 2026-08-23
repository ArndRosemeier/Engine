//! Same-room portal pair with recursive views (Portal-style hall of mirrors).
//!
//! One [`engine::Portal`] links two openings on opposite walls. Look through
//! either frame to see the room through the other — including portal-in-portal
//! recursion when they face each other.
//!
//! Controls: W/S walk · A/D turn · Shift sprint · Esc quit
use engine::prelude::*;

fn box_at(center: (f32, f32, f32), size: (f32, f32, f32), color: Color) -> Mesh {
    Shape::box_at(center, size, color).unwrap()
}

fn portal_frame(center: Vec3, color: Color, world: &mut World) {
    let w = 1.36;
    let h = 2.36;
    let t = 0.12;
    world.spawn(box_at((center.x - 0.62, 1.18, center.z), (t, h, t), color));
    world.spawn(box_at((center.x + 0.62, 1.18, center.z), (t, h, t), color));
    world.spawn(box_at((center.x, 2.30, center.z), (w, t, t), color));
    world.spawn(box_at((center.x, 0.06, center.z), (w, t, t), color));
}

fn main() {
    let mut pos = Vec3::new(0.0, 1.6, 0.0);
    let mut yaw = 180.0_f32;

    Engine::run("portal_pair", move |world, frame| {
        if frame.first {
            world.set_clear_color(rgb(36, 40, 48));
            world.set_sun((0.35, 1.0, 0.45), 0.32);
            // Deep enough to see the infinite hallway when portals face each other.
            world.set_portal_recursion(5);

            let concrete = rgb(168, 170, 176);
            let floor = rgb(92, 96, 104);
            let accent = rgb(210, 214, 220);
            let orange = rgb(232, 120, 36);
            let blue = rgb(48, 156, 232);

            world.spawn(box_at((0.0, 0.0, 0.0), (16.0, 0.1, 20.0), floor));
            world.spawn(box_at((0.0, 3.2, 0.0), (16.0, 0.1, 20.0), concrete));
            world.spawn(box_at((-8.0, 1.6, 0.0), (0.2, 3.2, 20.0), concrete));
            world.spawn(box_at((8.0, 1.6, 0.0), (0.2, 3.2, 20.0), concrete));
            world.spawn(box_at((-4.4, 1.6, -10.0), (7.2, 3.2, 0.2), concrete));
            world.spawn(box_at((4.4, 1.6, -10.0), (7.2, 3.2, 0.2), concrete));
            world.spawn(box_at((0.0, 2.72, -10.0), (1.6, 0.96, 0.2), concrete));
            world.spawn(box_at((-4.4, 1.6, 10.0), (7.2, 3.2, 0.2), concrete));
            world.spawn(box_at((4.4, 1.6, 10.0), (7.2, 3.2, 0.2), concrete));
            world.spawn(box_at((0.0, 2.72, 10.0), (1.6, 0.96, 0.2), concrete));

            // Landmarks visible through recursive portal views.
            world.spawn(box_at((-3.2, 0.45, -3.0), (1.2, 0.9, 1.2), orange));
            world.spawn(box_at((3.2, 0.9, 3.0), (0.7, 1.8, 0.7), blue));
            world.spawn(box_at((0.0, 0.02, 0.0), (2.0, 0.04, 2.0), accent));

            let a_at = Vec3::new(0.0, 1.18, -10.0);
            let b_at = Vec3::new(0.0, 1.18, 10.0);
            portal_frame(a_at, orange, world);
            portal_frame(b_at, blue, world);

            let a = world
                .place(
                    Mesh::opening(1.2, 2.2).unwrap(),
                    Place::new(a_at.x, a_at.y, a_at.z),
                )
                .unwrap();
            let b = world
                .place(
                    Mesh::opening(1.2, 2.2).unwrap(),
                    Place::new(b_at.x, b_at.y, b_at.z).with_yaw_deg(180.0),
                )
                .unwrap();
            world
                .create_portal(a, b, PortalSettings::TELEPORTING)
                .unwrap();
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
    });
}
