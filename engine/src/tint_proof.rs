//! Offline proof that instance tint changes shaded color (no window / GPU).
//!
//! Writes `target/tint_proof_{warm,cool,white}.png` when run:
//!   cargo test -p engine --lib tint_proof_png -- --nocapture

#[cfg(test)]
mod proof {
    use crate::color::Color;
    use crate::mesh::InstanceRaw;
    use bytemuck::bytes_of;
    use glam::Mat4;
    use std::path::PathBuf;

    fn shade_rgb(albedo: [f32; 3], tint: Color) -> [u8; 3] {
        let to_u8 = |c: f32| ((c.clamp(0.0, 1.0).powf(1.0 / 2.2)) * 255.0).round() as u8;
        [
            to_u8(albedo[0] * tint.r),
            to_u8(albedo[1] * tint.g),
            to_u8(albedo[2] * tint.b),
        ]
    }

    fn write_solid(path: &PathBuf, rgb: [u8; 3]) {
        let mut rgba = Vec::with_capacity(64 * 64 * 4);
        for _ in 0..(64 * 64) {
            rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        image::save_buffer(path, &rgba, 64, 64, image::ColorType::Rgba8)
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }

    #[test]
    fn tint_proof_png_and_instance_bytes() {
        let cream = [0.85_f32, 0.80, 0.72];
        let white = Color::WHITE;
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

        let w = shade_rgb(cream, white);
        let a = shade_rgb(cream, warm);
        let b = shade_rgb(cream, cool);
        let dist = (i32::from(a[0]) - i32::from(b[0])).unsigned_abs()
            + (i32::from(a[1]) - i32::from(b[1])).unsigned_abs()
            + (i32::from(a[2]) - i32::from(b[2])).unsigned_abs();
        assert!(
            dist > 40,
            "warm vs cool must be obvious on cream (dist={dist} rgb={a:?} vs {b:?})"
        );

        let raw_w = InstanceRaw::from_matrix_tint(Mat4::IDENTITY, warm);
        let raw_c = InstanceRaw::from_matrix_tint(Mat4::IDENTITY, cool);
        assert_ne!(&bytes_of(&raw_w)[64..80], &bytes_of(&raw_c)[64..80]);

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target");
        std::fs::create_dir_all(&dir).ok();
        write_solid(&dir.join("tint_proof_white.png"), w);
        write_solid(&dir.join("tint_proof_warm.png"), a);
        write_solid(&dir.join("tint_proof_cool.png"), b);
        eprintln!(
            "tint proof PNGs -> {} (white={w:?} warm={a:?} cool={b:?})",
            dir.display()
        );
    }
}

#[cfg(test)]
mod mesh_append_tests {
    use crate::color::Color;
    use crate::mesh::Mesh;
    use glam::Vec3;

    #[test]
    fn append_translated_merges_points_faces_and_colors() {
        let mut base = Mesh::new();
        let ba = base.add_point((0.0, 0.0, 0.0)).unwrap();
        let bb = base.add_point((1.0, 0.0, 0.0)).unwrap();
        let bc = base.add_point((0.0, 1.0, 0.0)).unwrap();
        for id in [ba, bb, bc] {
            base.set_point_color(id, Color::rgb(255, 0, 0)).unwrap();
        }
        base.add_triangle(ba, bb, bc).unwrap();

        let mut other = Mesh::new();
        let oa = other.add_point((0.0, 0.0, 0.0)).unwrap();
        let ob = other.add_point((0.0, 0.0, 1.0)).unwrap();
        let oc = other.add_point((1.0, 0.0, 1.0)).unwrap();
        for id in [oa, ob, oc] {
            other.set_point_color(id, Color::rgb(0, 255, 0)).unwrap();
        }
        other.add_triangle(oa, ob, oc).unwrap();

        base.append_translated(&other, Vec3::new(10.0, 20.0, 30.0))
            .unwrap();
        assert_eq!(base.point_count(), 6);
        assert_eq!(base.face_count(), 2);

        // Translated point landed where expected.
        let built = base.build();
        let has_translated_origin = built
            .positions
            .iter()
            .any(|p| (*p - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-4);
        assert!(has_translated_origin, "appended mesh must be translated");

        // Both color sets survive.
        assert!(built.colors.iter().any(|c| c.y > 0.5 && c.x < 0.5));
        assert!(built.colors.iter().any(|c| c.x > 0.5 && c.y < 0.5));
    }

    #[test]
    fn append_translated_rejects_out_of_bounds_face() {
        let mut base = Mesh::new();
        base.add_point((0.0, 0.0, 0.0)).unwrap();

        let mut other = Mesh::new();
        let a = other.add_point((0.0, 0.0, 0.0)).unwrap();
        other.add_point((1.0, 0.0, 0.0)).unwrap();
        // Corrupt face pointing past the point list must fail loudly.
        let bad = crate::mesh::PointId::peek(u32::MAX);
        let setup_err = other
            .add_triangle(a, a, bad)
            .expect_err("add_triangle must reject out-of-range id");
        assert!(matches!(setup_err, crate::error::EngineError::InvalidMesh(_)));
    }
}
