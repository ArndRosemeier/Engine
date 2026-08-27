//! Vector water meshes — ribbons, polygon fills, shore bands.
//!
//! Geometry is authored polylines/rings in XZ; Y is the water sheet height.
//! No wetness lattices.

use crate::color::Color;
use crate::mesh::{BuiltMesh, Mesh};
use glam::{Vec2, Vec3};

/// Extrude a centerline into a horizontal ribbon (quad strip).
///
/// `half_widths` and `zs` are per-vertex (same length as `centerline`).
pub fn ribbon_mesh(
    centerline: &[Vec2],
    half_widths: &[f32],
    zs: &[f32],
    color: Color,
) -> BuiltMesh {
    assert_eq!(centerline.len(), half_widths.len(), "ribbon half_widths");
    assert_eq!(centerline.len(), zs.len(), "ribbon zs");
    assert!(centerline.len() >= 2, "ribbon needs ≥2 points");

    let mut mesh = Mesh::new();
    let n = centerline.len();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);

    for i in 0..n {
        // Average adjacent segment directions so joins don't leave wedge gaps.
        let t0 = if i > 0 {
            (centerline[i] - centerline[i - 1]).normalize_or_zero()
        } else {
            Vec2::ZERO
        };
        let t1 = if i + 1 < n {
            (centerline[i + 1] - centerline[i]).normalize_or_zero()
        } else {
            Vec2::ZERO
        };
        let mut tangent = t0 + t1;
        if tangent.length_squared() < 1e-8 {
            tangent = if t1.length_squared() > 0.0 { t1 } else { t0 };
        }
        tangent = tangent.normalize_or_zero();
        let normal = Vec2::new(-tangent.y, tangent.x);
        let hw = half_widths[i].max(0.05);
        left.push(centerline[i] + normal * hw);
        right.push(centerline[i] - normal * hw);
    }

    let mut lids = Vec::with_capacity(n);
    let mut rids = Vec::with_capacity(n);
    for i in 0..n {
        let y = zs[i];
        let l = mesh
            .add_point(Vec3::new(left[i].x, y, left[i].y))
            .expect("ribbon left");
        let r = mesh
            .add_point(Vec3::new(right[i].x, y, right[i].y))
            .expect("ribbon right");
        mesh.set_point_color(l, color)
            .expect("new ribbon left point must accept color");
        mesh.set_point_color(r, color)
            .expect("new ribbon right point must accept color");
        lids.push(l);
        rids.push(r);
    }
    for i in 0..n - 1 {
        mesh.add_quad(lids[i], lids[i + 1], rids[i + 1], rids[i])
            .expect("ribbon quad references points created above");
    }
    mesh.build_smooth()
}

/// Fan-triangulate a simple polygon in XZ at constant `z`.
///
/// Ring must be closed (first ≠ last) with ≥3 vertices. Winding should be CCW
/// for upward normals.
pub fn polygon_fill_mesh(ring: &[Vec2], z: f32, color: Color) -> BuiltMesh {
    assert!(ring.len() >= 3, "polygon needs ≥3 verts");
    let mut mesh = Mesh::new();
    let mut ids = Vec::with_capacity(ring.len());
    for p in ring {
        let id = mesh.add_point(Vec3::new(p.x, z, p.y)).expect("poly point");
        mesh.set_point_color(id, color)
            .expect("new water mesh point must accept color");
        ids.push(id);
    }
    for k in 1..ids.len() - 1 {
        mesh.add_triangle(ids[0], ids[k], ids[k + 1])
            .expect("polygon fan triangle references points created above");
    }
    mesh.build_smooth()
}

/// Quad strip between two polylines of equal length (e.g. shore band).
pub fn band_mesh(inner: &[Vec2], outer: &[Vec2], z: f32, color: Color) -> BuiltMesh {
    assert_eq!(inner.len(), outer.len(), "band lengths");
    assert!(inner.len() >= 2, "band needs ≥2");
    let mut mesh = Mesh::new();
    let mut iids = Vec::with_capacity(inner.len());
    let mut oids = Vec::with_capacity(outer.len());
    for p in inner {
        let id = mesh.add_point(Vec3::new(p.x, z, p.y)).expect("band inner");
        mesh.set_point_color(id, color)
            .expect("new water mesh point must accept color");
        iids.push(id);
    }
    for p in outer {
        let id = mesh.add_point(Vec3::new(p.x, z, p.y)).expect("band outer");
        mesh.set_point_color(id, color)
            .expect("new water mesh point must accept color");
        oids.push(id);
    }
    for i in 0..inner.len() - 1 {
        mesh.add_quad(iids[i], iids[i + 1], oids[i + 1], oids[i])
            .expect("band quad references points created above");
    }
    mesh.build_smooth()
}

/// Axis-aligned rectangle fill in XZ (open-ocean / full-cell lake).
pub fn rect_fill_mesh(min: Vec2, max: Vec2, z: f32, color: Color) -> BuiltMesh {
    polygon_fill_mesh(
        &[
            Vec2::new(min.x, min.y),
            Vec2::new(max.x, min.y),
            Vec2::new(max.x, max.y),
            Vec2::new(min.x, max.y),
        ],
        z,
        color,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgba;

    #[test]
    fn ribbon_has_quads() {
        let c = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(20.0, 5.0),
        ];
        let hw = [2.0_f32; 3];
        let zs = [1.0_f32; 3];
        let m = ribbon_mesh(&c, &hw, &zs, rgba(40, 120, 175, 90));
        assert!(m.index_count() >= 12);
        assert!(m.vertex_count() >= 6);
    }

    #[test]
    fn polygon_fan_tris() {
        let ring = [
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(4.0, 3.0),
            Vec2::new(0.0, 3.0),
        ];
        let m = polygon_fill_mesh(&ring, 0.0, rgba(40, 120, 175, 90));
        assert_eq!(m.index_count(), 6); // 2 tris
    }
}
