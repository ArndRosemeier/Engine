//! GPU integration tests and CPU/GPU cross-checks.
//!
//! Run with `ENGINE_GPU_TESTS=1 cargo test -p engine gpu_field::verify`.

pub fn gpu_tests_enabled() -> bool {
    std::env::var_os("ENGINE_GPU_TESTS").is_some()
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::color::Color;
    use crate::error::EngineResult;

    use super::super::context::FieldGpuContext;
    use super::super::extract_cpu::extract_mesh_cpu;
    use super::super::field::{FieldBounds, GpuField};
    use super::super::grid::FieldGrid;
    use super::super::kernel::FieldKernel;
    use super::gpu_tests_enabled;

    fn demo_bounds() -> FieldBounds {
        FieldBounds::try_new(Vec3::ZERO, Vec3::splat(16.0), 0.5).expect("demo bounds")
    }

    fn run_demo_pipeline() -> EngineResult<(FieldGrid, Vec<f32>, usize, usize)> {
        let ctx = FieldGpuContext::try_new()?;
        let bounds = demo_bounds();
        let kernel = FieldKernel::demo_sphere_void(bounds);
        let grid = FieldGrid::from_bounds(bounds);

        let mut field = GpuField::new(bounds.voxel_size);
        field.set_session(bounds, kernel)?;
        let painted = field.paint(&ctx)?;

        let gpu_mesh = painted.extract_mesh(&ctx, Color::rgb(160, 150, 140))?;
        let cpu_mesh = extract_mesh_cpu(&grid, &painted.density, Color::rgb(160, 150, 140));

        Ok((
            grid,
            painted.density.clone(),
            gpu_mesh.triangle_count(),
            cpu_mesh.triangle_count(),
        ))
    }

    #[test]
    fn gpu_paint_matches_cpu_reference_at_corners() {
        if !gpu_tests_enabled() {
            return;
        }
        let ctx = FieldGpuContext::try_new().expect("GPU context");
        let bounds = demo_bounds();
        let kernel = FieldKernel::demo_sphere_void(bounds);
        let grid = FieldGrid::from_bounds(bounds);

        let mut field = GpuField::new(bounds.voxel_size);
        field.set_session(bounds, kernel).expect("session");
        let painted = field.paint(&ctx).expect("paint");

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
        assert!(
            max_err < 1e-5,
            "GPU density must match CPU reference at corners: max error {max_err}"
        );
    }

    #[test]
    fn gpu_extract_produces_sphere_void_mesh() {
        if !gpu_tests_enabled() {
            return;
        }
        let ctx = FieldGpuContext::try_new().expect("GPU context");
        let bounds = demo_bounds();
        let kernel = FieldKernel::demo_sphere_void(bounds);

        let mut field = GpuField::new(bounds.voxel_size);
        field.set_session(bounds, kernel).expect("session");
        field.paint(&ctx).expect("paint");
        let mesh = field
            .extract_mesh(&ctx, Color::rgb(160, 150, 140))
            .expect("extract");

        assert!(
            mesh.triangle_count() > 100,
            "sphere void must produce a substantial surface, got {} tris",
            mesh.triangle_count()
        );
    }

    #[test]
    fn gpu_extract_triangle_count_near_cpu_reference() {
        if !gpu_tests_enabled() {
            return;
        }
        let (_grid, _density, gpu_tris, cpu_tris) =
            run_demo_pipeline().expect("demo pipeline");

        let ratio = gpu_tris as f32 / cpu_tris as f32;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "GPU extract {gpu_tris} tris should be within 15% of CPU reference {cpu_tris} (ratio {ratio:.3})"
        );
    }

    #[test]
    fn painted_field_trilinear_sample_is_finite() {
        if !gpu_tests_enabled() {
            return;
        }
        let ctx = FieldGpuContext::try_new().expect("GPU context");
        let bounds = demo_bounds();
        let kernel = FieldKernel::demo_sphere_void(bounds);

        let mut field = GpuField::new(bounds.voxel_size);
        field.set_session(bounds, kernel).expect("session");
        let painted = field.paint(&ctx).expect("paint");

        let probes = [
            Vec3::splat(8.0),
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::new(14.0, 10.0, 6.0),
        ];
        for p in probes {
            let d = painted.sample_density(p);
            assert!(d.is_finite(), "sample at {p:?} must be finite, got {d}");
        }
    }

    #[test]
    fn lod_extract_decimates_within_reference_shell() {
        if !gpu_tests_enabled() {
            return;
        }
        let ctx = FieldGpuContext::try_new().expect("GPU context");
        let bounds = demo_bounds();
        let kernel = FieldKernel::demo_sphere_void(bounds);

        let mut field = GpuField::new(bounds.voxel_size);
        field.set_session(bounds, kernel).expect("session");
        let painted = field.paint(&ctx).expect("paint");

        let full = ctx
            .extract_mesh_lod(&painted.grid, &painted.density, Color::rgb(160, 150, 140), 1)
            .expect("stride-1 extract");
        let coarse = ctx
            .extract_mesh_lod(&painted.grid, &painted.density, Color::rgb(160, 150, 140), 4)
            .expect("stride-4 extract");

        assert!(
            coarse.triangle_count() < full.triangle_count(),
            "LOD stride 4 ({} tris) must decimate stride 1 ({} tris)",
            coarse.triangle_count(),
            full.triangle_count()
        );
        // Coarse shell must stay within a voxel*stride of the fine shell.
        let slack = bounds.voxel_size * 4.0 + 1e-3;
        for p in &coarse.positions {
            assert!(
                bounds.contains(*p),
                "coarse vertex {p:?} escaped field bounds"
            );
        }
        let fine_min = full
            .positions
            .iter()
            .copied()
            .reduce(Vec3::min)
            .expect("fine min");
        let fine_max = full
            .positions
            .iter()
            .copied()
            .reduce(Vec3::max)
            .expect("fine max");
        let coarse_min = coarse
            .positions
            .iter()
            .copied()
            .reduce(Vec3::min)
            .expect("coarse min");
        let coarse_max = coarse
            .positions
            .iter()
            .copied()
            .reduce(Vec3::max)
            .expect("coarse max");
        assert!(
            coarse_min.x >= fine_min.x - slack
                && coarse_min.y >= fine_min.y - slack
                && coarse_min.z >= fine_min.z - slack
                && coarse_max.x <= fine_max.x + slack
                && coarse_max.y <= fine_max.y + slack
                && coarse_max.z <= fine_max.z + slack,
            "coarse shell must hug the fine shell within {slack} m"
        );
        assert!(coarse.triangle_count() > 10, "LOD mesh must be substantial");
    }

    #[test]
    fn lod_extract_is_deterministic() {
        if !gpu_tests_enabled() {
            return;
        }
        let ctx = FieldGpuContext::try_new().expect("GPU context");
        let bounds = demo_bounds();
        let kernel = FieldKernel::demo_sphere_void(bounds);

        let mut field = GpuField::new(bounds.voxel_size);
        field.set_session(bounds, kernel).expect("session");
        let painted = field.paint(&ctx).expect("paint");

        let a = ctx
            .extract_mesh_lod(&painted.grid, &painted.density, Color::rgb(1, 2, 3), 2)
            .expect("extract a");
        let b = ctx
            .extract_mesh_lod(&painted.grid, &painted.density, Color::rgb(1, 2, 3), 2)
            .expect("extract b");
        assert_eq!(a.positions.len(), b.positions.len());
        for (pa, pb) in a.positions.iter().zip(b.positions.iter()) {
            assert_eq!(pa, pb, "LOD extract must be deterministic");
        }
    }
}
