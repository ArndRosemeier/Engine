//! Friendly mesh demo: build a house from boxes + quads, no GPU types.
use engine::prelude::*;

fn house() -> Mesh {
    let mut mesh = Mesh::new();
    let wall = rgb(220, 158, 102);
    let roof = rgb(184, 46, 41);

    // Body
    mesh.add_box((0.0, 0.6, 0.0), (2.4, 1.2, 1.8), wall)
        .expect("walls");
    // Door + window sit slightly in front of the −Z wall
    mesh.add_box((0.0, 0.45, -0.96), (0.55, 0.9, 0.12), rgb(89, 56, 31))
        .expect("door");
    mesh.add_box((0.7, 0.75, -0.96), (0.45, 0.4, 0.1), rgb(140, 191, 230))
        .expect("window");

    // Gabled roof (CCW from outside)
    let y_eave = 1.22;
    let y_ridge = 2.15;
    let (x0, x1) = (-1.32, 1.32);
    let (z0, z1) = (-1.02, 1.02);

    let fl = mesh.add_point((x0, y_eave, z0)).unwrap();
    let fr = mesh.add_point((x1, y_eave, z0)).unwrap();
    let br = mesh.add_point((x1, y_eave, z1)).unwrap();
    let bl = mesh.add_point((x0, y_eave, z1)).unwrap();
    let rl = mesh.add_point((x0, y_ridge, 0.0)).unwrap();
    let rr = mesh.add_point((x1, y_ridge, 0.0)).unwrap();
    for id in [fl, fr, br, bl, rl, rr] {
        mesh.set_point_color(id, roof).unwrap();
    }
    mesh.add_quad(fr, fl, rl, rr).unwrap(); // front slope
    mesh.add_quad(bl, br, rr, rl).unwrap(); // back slope
    mesh.add_triangle(fl, bl, rl).unwrap(); // left end
    mesh.add_triangle(fr, rr, br).unwrap(); // right end
    mesh.add_quad(fl, fr, rr, rl).unwrap(); // underside
    mesh.add_quad(br, bl, rl, rr).unwrap();

    // Fill wall gables under the roof
    let wl_f = mesh.add_point((-1.2, 1.2, -0.9)).unwrap();
    let wl_b = mesh.add_point((-1.2, 1.2, 0.9)).unwrap();
    let wl_r = mesh.add_point((-1.2, y_ridge, 0.0)).unwrap();
    let wr_f = mesh.add_point((1.2, 1.2, -0.9)).unwrap();
    let wr_b = mesh.add_point((1.2, 1.2, 0.9)).unwrap();
    let wr_r = mesh.add_point((1.2, y_ridge, 0.0)).unwrap();
    for id in [wl_f, wl_b, wl_r, wr_f, wr_b, wr_r] {
        mesh.set_point_color(id, wall).unwrap();
    }
    mesh.add_triangle(wl_f, wl_b, wl_r).unwrap();
    mesh.add_triangle(wr_b, wr_f, wr_r).unwrap();

    mesh
}

fn main() {
    let mut yaw = -70.0_f32;

    Engine::run("hello_mesh", move |world, frame| {
        if frame.first {
            world.spawn(house());
            world.spawn(
                Shape::box_at((0.0, -0.05, 0.0), (14.0, 0.1, 14.0), rgb(87, 143, 77)).unwrap(),
            );
            world.spawn(
                Shape::box_at((0.0, 0.06, -1.95), (0.85, 0.08, 2.2), rgb(140, 128, 107)).unwrap(),
            );
            world.set_sun((0.4, 1.0, -0.55), 0.2);
        }

        yaw += frame.dt * 18.0;
        let yaw = if std::env::var_os("ENGINE_SCREENSHOT").is_some() {
            -55.0
        } else {
            yaw
        };
        world.look_orbit((0.0, 1.0, 0.0), 8.0, yaw, 24.0);
    });
}
