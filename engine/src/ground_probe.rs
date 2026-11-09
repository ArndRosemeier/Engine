//! What is drawn under a point: the upward faces of the world's meshes,
//! for feet that stand on what the player sees rather than on a simpler
//! collision shape a few centimetres off.
//!
//! Each mesh gets a [`GroundMesh`] the first time something stands near
//! it: its faces that look up (steeper than about 70 degrees are walls),
//! filed in a grid in the mesh's own space, so every instance of a mesh
//! shares one. [`crate::World::ground_below`] asks it, and so does foot IK.

use crate::mesh::BuiltMesh;
use glam::{Mat4, Vec2, Vec3};
use std::sync::Arc;

/// Faces whose normal's up part is at least this count as ground.
const UP: f32 = 0.3;
/// Most grid cells along one side.
const MAX_CELLS: f32 = 256.0;
/// Smallest grid cell (m).
const MIN_CELL: f32 = 0.5;

/// A mesh's upward faces in a grid over its x/z plane.
#[derive(Debug)]
pub(crate) struct GroundMesh {
    lo: Vec2,
    cell: f32,
    nx: usize,
    nz: usize,
    /// `refs[starts[c]..starts[c + 1]]` are the faces over cell `c`.
    starts: Vec<u32>,
    refs: Vec<u32>,
    tris: Vec<[Vec3; 3]>,
}

impl GroundMesh {
    /// The upward faces of `mesh`; `None` if it has none.
    pub(crate) fn build(mesh: &BuiltMesh) -> Option<Self> {
        let mut tris = Vec::new();
        for ix in mesh.indices.chunks_exact(3) {
            let Some(t) = ix
                .iter()
                .map(|&i| mesh.positions.get(i as usize).copied())
                .collect::<Option<Vec<Vec3>>>()
            else {
                continue;
            };
            let n = (t[1] - t[0]).cross(t[2] - t[0]);
            let len = n.length();
            if len.is_nan() || len <= 1e-12 || n.y < UP * len {
                continue;
            }
            tris.push([t[0], t[1], t[2]]);
        }
        if tris.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for t in &tris {
            for p in t {
                lo = lo.min(Vec2::new(p.x, p.z));
                hi = hi.max(Vec2::new(p.x, p.z));
            }
        }
        let size = (hi - lo).max(Vec2::splat(1e-3));
        let cell = (size.max_element() / MAX_CELLS).max(MIN_CELL);
        let nx = (size.x / cell).floor() as usize + 1;
        let nz = (size.y / cell).floor() as usize + 1;
        let span = |t: &[Vec3; 3]| {
            let (mut a, mut b) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            for p in t {
                a = a.min(Vec2::new(p.x, p.z));
                b = b.max(Vec2::new(p.x, p.z));
            }
            let c = |v: f32, n: usize| ((v / cell).floor().max(0.0) as usize).min(n - 1);
            (
                c(a.x - lo.x, nx),
                c(b.x - lo.x, nx),
                c(a.y - lo.y, nz),
                c(b.y - lo.y, nz),
            )
        };
        // Count, then fill (one flat list, no list per cell).
        let mut starts = vec![0u32; nx * nz + 1];
        for t in &tris {
            let (x0, x1, z0, z1) = span(t);
            for z in z0..=z1 {
                for x in x0..=x1 {
                    starts[z * nx + x + 1] += 1;
                }
            }
        }
        for i in 1..starts.len() {
            starts[i] += starts[i - 1];
        }
        let mut fill = starts.clone();
        let mut refs = vec![0u32; starts[nx * nz] as usize];
        for (k, t) in tris.iter().enumerate() {
            let (x0, x1, z0, z1) = span(t);
            for z in z0..=z1 {
                for x in x0..=x1 {
                    let c = z * nx + x;
                    refs[fill[c] as usize] = k as u32;
                    fill[c] += 1;
                }
            }
        }
        Some(Self {
            lo,
            cell,
            nx,
            nz,
            starts,
            refs,
            tris,
        })
    }

    /// The highest face over (x, z) whose height there lies in `lo..=hi`.
    pub(crate) fn height_at(&self, x: f32, z: f32, lo: f32, hi: f32) -> Option<f32> {
        let cx = ((x - self.lo.x) / self.cell).floor();
        let cz = ((z - self.lo.y) / self.cell).floor();
        if cx < 0.0 || cz < 0.0 || cx >= self.nx as f32 || cz >= self.nz as f32 {
            return None;
        }
        let c = cz as usize * self.nx + cx as usize;
        let mut best: Option<f32> = None;
        for &k in &self.refs[self.starts[c] as usize..self.starts[c + 1] as usize] {
            let Some(y) = height_on(&self.tris[k as usize], x, z) else {
                continue;
            };
            if y >= lo && y <= hi && best.is_none_or(|b| y > b) {
                best = Some(y);
            }
        }
        best
    }
}

/// Height of triangle `t` over (x, z), if (x, z) lies on it (edges
/// included, with a hair to spare so seams between faces never leak).
fn height_on(t: &[Vec3; 3], x: f32, z: f32) -> Option<f32> {
    let (a, b, c) = (t[0], t[1], t[2]);
    let d = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
    if d.abs() < 1e-12 {
        return None;
    }
    let u = ((x - a.x) * (c.z - a.z) - (c.x - a.x) * (z - a.z)) / d;
    let v = ((b.x - a.x) * (z - a.z) - (x - a.x) * (b.z - a.z)) / d;
    const EPS: f32 = 1e-4;
    if u < -EPS || v < -EPS || u + v > 1.0 + EPS {
        return None;
    }
    Some(a.y + u * (b.y - a.y) + v * (c.y - a.y))
}

/// One placement of a ground mesh: world to mesh space, and back.
#[derive(Clone, Debug)]
pub(crate) struct Placed {
    pub mesh: Arc<GroundMesh>,
    pub to_local: Mat4,
    pub to_world: Mat4,
}

/// The ground meshes placed around one spot, gathered once and asked
/// many times (both feet, every frame's probes).
#[derive(Clone, Debug, Default)]
pub(crate) struct GroundPatch {
    pub placed: Vec<Placed>,
}

impl GroundPatch {
    /// The highest drawn surface under `p` no higher than `p.y` and no
    /// lower than `p.y - depth`.
    pub(crate) fn height_below(&self, p: Vec3, depth: f32) -> Option<f32> {
        let mut best: Option<f32> = None;
        for m in &self.placed {
            // Only placements that keep up as up (yaw, scale, moves).
            let up = m.to_world.y_axis.truncate();
            if up.x.abs() > 1e-3 || up.z.abs() > 1e-3 || up.y <= 0.0 {
                continue;
            }
            let l = m.to_local.transform_point3(p);
            let sy = up.y;
            let lo = l.y - depth / sy;
            let Some(y) = m.mesh.height_at(l.x, l.z, lo, l.y) else {
                continue;
            };
            let world_y = m.to_world.transform_point3(Vec3::new(l.x, y, l.z)).y;
            if best.is_none_or(|b| world_y > b) {
                best = Some(world_y);
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb;
    use crate::mesh::Mesh;

    #[test]
    fn finds_the_top_of_a_box_and_skips_its_walls() {
        let m = Mesh::box_at(Vec3::new(0.0, 0.5, 0.0), Vec3::ONE, rgb(1, 2, 3))
            .unwrap()
            .build();
        let g = GroundMesh::build(&m).unwrap();
        assert_eq!(g.height_at(0.1, -0.2, -5.0, 5.0), Some(1.0));
        assert_eq!(g.height_at(0.1, -0.2, -5.0, 0.9), None);
        assert_eq!(g.height_at(0.7, 0.0, -5.0, 5.0), None);
        let patch = GroundPatch {
            placed: vec![Placed {
                mesh: Arc::new(g),
                to_local: Mat4::from_translation(Vec3::new(-10.0, -2.0, 0.0)),
                to_world: Mat4::from_translation(Vec3::new(10.0, 2.0, 0.0)),
            }],
        };
        let y = patch.height_below(Vec3::new(10.2, 3.4, 0.0), 1.0).unwrap();
        assert!((y - 3.0).abs() < 1e-5);
        assert!(patch.height_below(Vec3::new(10.2, 4.4, 0.0), 1.0).is_none());
    }
}
