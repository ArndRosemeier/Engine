//! Generic GPU scalar-field demo: solid box with a spherical void.
//!
//! Verifies paint + marching-cubes extract without any cave semantics.
//!
//! Run: `cargo run -p gpu_field_demo`
//! GPU tests: `ENGINE_GPU_TESTS=1 cargo test -p engine gpu_field::verify`

use engine::prelude::*;
use engine::{FieldBounds, FieldGpuContext, FieldKernel, GpuField};
use glam::Vec3;

fn main() {
    let mut mesh_spawned = false;
    let mut status = String::from("initializing GPU…");
    let mut tri_count = 0usize;
    let mut density_err = 0.0_f32;

    let ctx = match FieldGpuContext::try_new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gpu_field_demo: {e}");
            std::process::exit(1);
        }
    };

    Engine::run("gpu_field_demo", move |world, frame| {
        if frame.first {
            world.set_clear_color(rgb(30, 36, 48));
            world.set_sun((0.4, 1.0, 0.25), 0.28);

            let bounds = FieldBounds::try_new(Vec3::ZERO, Vec3::splat(16.0), 0.5).expect("bounds");
            let kernel = FieldKernel::demo_sphere_void(bounds);
            let mut field = GpuField::new(0.5);
            field.set_session(bounds, kernel).expect("session");

            match field.paint(&ctx) {
                Ok(painted) => {
                    let grid = &painted.grid;
                    let mut max_err = 0.0_f32;
                    for z in 0..grid.corners[2] {
                        for y in 0..grid.corners[1] {
                            for x in 0..grid.corners[0] {
                                let p = grid.corner_world(x, y, z);
                                let gpu = painted.density[grid.corner_index(x, y, z)];
                                let cpu = painted.reference_density(p);
                                max_err = max_err.max((gpu - cpu).abs());
                            }
                        }
                    }
                    density_err = max_err;

                    match field.extract_mesh(&ctx, rgb(168, 158, 142)) {
                        Ok(built) => {
                            tri_count = built.triangle_count();
                            let mut mesh = Mesh::new();
                            let mut ids = Vec::with_capacity(built.positions.len());
                            for (i, p) in built.positions.iter().enumerate() {
                                let id = mesh.add_point(*p).expect("point");
                                let c = built.colors[i];
                                mesh.set_point_color(
                                    id,
                                    Color::rgba01(c.x, c.y, c.z, c.w).expect("color"),
                                )
                                .expect("color");
                                ids.push(id);
                            }
                            for tri in built.indices.chunks_exact(3) {
                                mesh.add_face(&[
                                    ids[tri[0] as usize],
                                    ids[tri[1] as usize],
                                    ids[tri[2] as usize],
                                ])
                                .expect("face");
                            }
                            world.spawn(mesh);
                            mesh_spawned = true;
                            status = format!(
                                "GPU paint + extract OK · {tri_count} tris · density err {max_err:.2e}"
                            );
                        }
                        Err(e) => status = format!("extract failed: {e}"),
                    }
                }
                Err(e) => status = format!("paint failed: {e}"),
            }
        }

        engine::egui::Window::new("gpu_field_demo")
            .anchor(engine::egui::Align2::LEFT_TOP, [12.0, 12.0])
            .resizable(true)
            .show(frame.ui.ctx(), |ui| {
                ui.heading("Engine gpu_field — phase 1");
                ui.label(&status);
                ui.label(format!("Triangles: {tri_count}"));
                ui.label(format!("Corner density max |GPU−CPU|: {density_err:.2e}"));
                ui.separator();
                ui.label("Set ENGINE_GPU_TESTS=1 for automated verify tests.");
            });

        if mesh_spawned {
            world.look_orbit((8.0, 8.0, 8.0), 28.0, frame.time * 18.0, 24.0);
        } else {
            world.look_orbit((8.0, 8.0, 8.0), 28.0, 48.0, 24.0);
        }
    });
}
