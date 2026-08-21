//! Offline proof that instance tint changes shaded color (no window / GPU).
//!
//! Writes `target/tint_proof_{warm,cool,white}.png` when run:
//!   cargo test -p engine --lib tint_proof_png -- --nocapture

#[cfg(test)]
mod tint_proof {
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
