//! Break an ordinary mesh into solid pieces.
//!
//! Voronoi fracture: scatter seed points (denser near the impact), and for
//! each seed cut the mesh with the bisecting plane towards every nearby
//! seed. Each cut is capped with interior faces, so pieces are closed solids
//! with a darker "inside" colour rather than hollow shells. Works on any
//! closed mesh; open meshes still split, just without caps.

use crate::mesh::{BuiltMesh, SurfaceMaterial};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;

/// One piece of a fractured mesh, centred on its own centroid.
#[derive(Clone, Debug)]
pub struct Fragment {
    /// Vertices relative to `centroid`.
    pub mesh: BuiltMesh,
    /// Where the piece sits in the space the source mesh was given in.
    pub centroid: Vec3,
    /// Largest distance from the centroid to a vertex.
    pub radius: f32,
}

#[derive(Clone, Copy, Debug)]
struct V {
    p: Vec3,
    n: Vec3,
    c: Vec4,
    uv: [f32; 2],
    s: SurfaceMaterial,
}

impl V {
    fn lerp(a: &V, b: &V, t: f32) -> V {
        V {
            p: a.p.lerp(b.p, t),
            n: a.n.lerp(b.n, t).normalize_or_zero(),
            c: a.c.lerp(b.c, t),
            uv: [
                a.uv[0] + (b.uv[0] - a.uv[0]) * t,
                a.uv[1] + (b.uv[1] - a.uv[1]) * t,
            ],
            s: if t < 0.5 { a.s } else { b.s },
        }
    }
}

type Tri = [V; 3];

/// Tiny deterministic RNG (xorshift), so a seed always breaks the same way.
struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn triangles(mesh: &BuiltMesh, transform: Mat4) -> Vec<Tri> {
    let normal_m = Mat3::from_mat4(transform).inverse().transpose();
    let verts: Vec<V> = (0..mesh.positions.len())
        .map(|i| V {
            p: transform.transform_point3(mesh.positions[i]),
            n: (normal_m * mesh.normals.get(i).copied().unwrap_or(Vec3::Y)).normalize_or_zero(),
            c: mesh.colors.get(i).copied().unwrap_or(Vec4::ONE),
            uv: mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            s: mesh
                .surface_overrides
                .get(i)
                .copied()
                .unwrap_or(mesh.surface),
        })
        .collect();
    mesh.indices
        .chunks_exact(3)
        .filter_map(|t| {
            let a = *verts.get(t[0] as usize)?;
            let b = *verts.get(t[1] as usize)?;
            let c = *verts.get(t[2] as usize)?;
            Some([a, b, c])
        })
        .collect()
}

/// Keep the parts of `tris` with `dot(p - origin, n) <= 0`, and return the
/// cut segments that lie on the plane.
fn clip(tris: &[Tri], origin: Vec3, n: Vec3) -> (Vec<Tri>, Vec<[Vec3; 2]>) {
    let mut out = Vec::with_capacity(tris.len());
    let mut segments = Vec::new();
    for tri in tris {
        let d = tri.map(|v| (v.p - origin).dot(n));
        let inside = d.map(|x| x <= 0.0);
        let count = inside.iter().filter(|&&i| i).count();
        match count {
            3 => out.push(*tri),
            0 => {}
            _ => {
                // Walk the edges, emitting kept corners and crossing points.
                let mut poly: Vec<V> = Vec::with_capacity(4);
                let mut cut: Vec<Vec3> = Vec::with_capacity(2);
                for i in 0..3 {
                    let j = (i + 1) % 3;
                    if inside[i] {
                        poly.push(tri[i]);
                    }
                    if inside[i] != inside[j] {
                        // Same edge, same order, same floats in both
                        // triangles that share it, so cut points match.
                        let (a, b) = if lex_less(tri[i].p, tri[j].p) {
                            (i, j)
                        } else {
                            (j, i)
                        };
                        let t = d[a] / (d[a] - d[b]);
                        let v = V::lerp(&tri[a], &tri[b], t);
                        cut.push(v.p);
                        poly.push(v);
                    }
                }
                for k in 1..poly.len() - 1 {
                    out.push([poly[0], poly[k], poly[k + 1]]);
                }
                if cut.len() == 2 {
                    segments.push([cut[0], cut[1]]);
                }
            }
        }
    }
    (out, segments)
}

fn lex_less(a: Vec3, b: Vec3) -> bool {
    (a.x, a.y, a.z) < (b.x, b.y, b.z)
}

fn key(p: Vec3) -> [i64; 3] {
    let q = p * 1.0e4;
    [q.x.round() as i64, q.y.round() as i64, q.z.round() as i64]
}

/// Join cut segments into closed loops.
fn loops(segments: &[[Vec3; 2]]) -> Vec<Vec<Vec3>> {
    let mut points: Vec<Vec3> = Vec::new();
    let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
    let mut id = |p: Vec3, points: &mut Vec<Vec3>| {
        *ids.entry(key(p)).or_insert_with(|| {
            points.push(p);
            points.len() - 1
        })
    };
    let mut adj: Vec<Vec<usize>> = Vec::new();
    for s in segments {
        let a = id(s[0], &mut points);
        let b = id(s[1], &mut points);
        if a == b {
            continue;
        }
        adj.resize(points.len(), Vec::new());
        adj[a].push(b);
        adj[b].push(a);
    }
    adj.resize(points.len(), Vec::new());
    let mut used = vec![false; points.len()];
    let mut result = Vec::new();
    for start in 0..points.len() {
        if used[start] || adj[start].len() != 2 {
            continue;
        }
        let mut ring = vec![start];
        used[start] = true;
        let mut prev = start;
        let mut cur = adj[start][0];
        let mut closed = false;
        while ring.len() <= points.len() {
            if cur == start {
                closed = true;
                break;
            }
            if used[cur] || adj[cur].len() != 2 {
                break;
            }
            used[cur] = true;
            ring.push(cur);
            let next = if adj[cur][0] == prev {
                adj[cur][1]
            } else {
                adj[cur][0]
            };
            prev = cur;
            cur = next;
        }
        if closed && ring.len() >= 3 {
            result.push(ring.into_iter().map(|i| points[i]).collect());
        }
    }
    result
}

/// Ear-clip a planar loop (projected on the plane with normal `n`) into
/// triangles wound counter-clockwise around `n`.
fn triangulate(ring: &[Vec3], n: Vec3) -> Vec<[Vec3; 3]> {
    let u = n.any_orthonormal_vector();
    let v = n.cross(u);
    let mut pts: Vec<(Vec3, Vec2)> = ring
        .iter()
        .map(|&p| (p, Vec2::new(p.dot(u), p.dot(v))))
        .collect();
    let area: f32 = (0..pts.len())
        .map(|i| {
            let a = pts[i].1;
            let b = pts[(i + 1) % pts.len()].1;
            a.perp_dot(b)
        })
        .sum();
    if area < 0.0 {
        pts.reverse();
    }
    let mut out = Vec::new();
    let mut guard = 0;
    while pts.len() > 3 && guard < 10_000 {
        guard += 1;
        let m = pts.len();
        let mut clipped = false;
        for i in 0..m {
            let (a, b, c) = (pts[(i + m - 1) % m].1, pts[i].1, pts[(i + 1) % m].1);
            if (b - a).perp_dot(c - b) <= 1e-9 {
                continue; // reflex or degenerate corner
            }
            let inside = pts.iter().any(|&(_, p)| {
                if p == a || p == b || p == c {
                    return false;
                }
                let d1 = (b - a).perp_dot(p - a);
                let d2 = (c - b).perp_dot(p - b);
                let d3 = (a - c).perp_dot(p - c);
                d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
            });
            if inside {
                continue;
            }
            out.push([pts[(i + m - 1) % m].0, pts[i].0, pts[(i + 1) % m].0]);
            pts.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break; // self-intersecting loop: give up on the rest
        }
    }
    if pts.len() == 3 {
        out.push([pts[0].0, pts[1].0, pts[2].0]);
    }
    out
}

/// How a mesh breaks.
#[derive(Clone, Copy, Debug)]
pub struct FractureSettings {
    pub pieces: usize,
    pub seed: u32,
    /// Colour multiplier for the freshly broken inside faces.
    pub interior_shade: f32,
}

impl Default for FractureSettings {
    fn default() -> Self {
        Self {
            pieces: 12,
            seed: 1,
            interior_shade: 0.6,
        }
    }
}

/// Break `mesh` (placed by `transform`) into pieces around `impact`, which is
/// in the same space as the transformed mesh. Pieces near the impact come out
/// smaller.
pub fn fracture(
    mesh: &BuiltMesh,
    transform: Mat4,
    impact: Vec3,
    settings: FractureSettings,
) -> Vec<Fragment> {
    // Work around the mesh's own centre: far from the origin, f32 would
    // lose the precision the cuts need.
    let pivot = transform.transform_point3(
        mesh.positions.iter().copied().sum::<Vec3>() / mesh.positions.len().max(1) as f32,
    );
    let tris = triangles(mesh, Mat4::from_translation(-pivot) * transform);
    if tris.is_empty() {
        return Vec::new();
    }
    let impact = impact - pivot;
    let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for t in &tris {
        for v in t {
            min = min.min(v.p);
            max = max.max(v.p);
        }
    }
    let size = (max - min).max(Vec3::splat(1e-3));
    let pieces = settings.pieces.clamp(1, 64);
    let mut rng = Rng(settings
        .seed
        .wrapping_mul(747_796_405)
        .wrapping_add(2_891_336_453)
        | 1);
    let impact = impact.clamp(min, max);
    let seeds: Vec<Vec3> = (0..pieces)
        .map(|i| {
            let r = Vec3::new(rng.next(), rng.next(), rng.next());
            let anywhere = min + r * size;
            if i % 2 == 0 {
                // Half the seeds crowd the impact: smaller shards there.
                impact.lerp(anywhere, 0.35)
            } else {
                anywhere
            }
        })
        .collect();

    let mean_color = {
        let sum: Vec4 = tris.iter().flat_map(|t| t.iter().map(|v| v.c)).sum();
        sum / (tris.len() * 3) as f32
    };
    let interior = (mean_color.truncate() * settings.interior_shade).extend(1.0);
    let surface = mesh.surface;

    let mut fragments = Vec::new();
    for (i, &seed) in seeds.iter().enumerate() {
        // Nearest seeds first: they cut the most, so later cuts see fewer
        // triangles. Every seed is used, or a cell could overlap another.
        let mut others: Vec<(f32, Vec3)> = seeds
            .iter()
            .enumerate()
            .filter(|&(j, s)| j != i && s.distance_squared(seed) > 1e-10)
            .map(|(_, &s)| (s.distance_squared(seed), s))
            .collect();
        others.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut cell = tris.clone();
        for &(_, other) in &others {
            let n = (other - seed).normalize();
            let origin = (seed + other) * 0.5;
            let (kept, segments) = clip(&cell, origin, n);
            cell = kept;
            if cell.is_empty() {
                break;
            }
            for ring in loops(&segments) {
                for t in triangulate(&ring, n) {
                    cell.push(t.map(|p| V {
                        p,
                        n,
                        c: interior,
                        uv: [0.0, 0.0],
                        s: surface,
                    }));
                }
            }
        }
        if cell.len() < 4 {
            continue;
        }
        let count = (cell.len() * 3) as f32;
        let centroid = cell
            .iter()
            .flat_map(|t| t.iter().map(|v| v.p))
            .sum::<Vec3>()
            / count;
        let mut built = BuiltMesh {
            elements: Vec::new(),
            positions: Vec::with_capacity(cell.len() * 3),
            normals: Vec::with_capacity(cell.len() * 3),
            colors: Vec::with_capacity(cell.len() * 3),
            uvs: Vec::with_capacity(cell.len() * 3),
            indices: Vec::with_capacity(cell.len() * 3),
            opaque_index_count: 0,
            surface,
            surface_overrides: Vec::new(),
        };
        let per_vertex = !mesh.surface_overrides.is_empty();
        let mut radius: f32 = 0.0;
        for t in &cell {
            for v in t {
                let local = v.p - centroid;
                radius = radius.max(local.length());
                built.indices.push(built.positions.len() as u32);
                built.positions.push(local);
                built.normals.push(v.n);
                built.colors.push(v.c);
                built.uvs.push(v.uv);
                if per_vertex {
                    built.surface_overrides.push(v.s);
                }
            }
        }
        built.opaque_index_count = built.indices.len();
        fragments.push(Fragment {
            mesh: built,
            centroid: centroid + pivot,
            radius,
        });
    }
    fragments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::rgb;
    use crate::mesh::Mesh;

    fn cube() -> BuiltMesh {
        Mesh::box_at((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), rgb(200, 180, 150))
            .expect("box")
            .build()
    }

    #[test]
    fn cube_breaks_into_closed_pieces_that_keep_its_volume() {
        let mesh = cube();
        let pieces = fracture(
            &mesh,
            Mat4::IDENTITY,
            Vec3::new(1.0, 0.5, 0.0),
            FractureSettings {
                pieces: 8,
                ..Default::default()
            },
        );
        assert!(pieces.len() >= 4, "got {} pieces", pieces.len());
        // Signed volume of each closed piece; together they fill the cube.
        let mut total = 0.0;
        for f in &pieces {
            let m = &f.mesh;
            let mut vol = 0.0;
            for t in m.indices.chunks_exact(3) {
                let (a, b, c) = (
                    m.positions[t[0] as usize],
                    m.positions[t[1] as usize],
                    m.positions[t[2] as usize],
                );
                vol += a.dot(b.cross(c)) / 6.0;
            }
            assert!(vol > 0.0, "piece is inside out or open: {vol}");
            total += vol;
        }
        assert!((total - 8.0).abs() < 0.05, "volume {total}");
    }
}

#[cfg(test)]
mod tower_tests {
    use super::*;
    use crate::color::rgb;
    use crate::mesh::Mesh;

    #[test]
    fn tall_box_pieces_are_closed() {
        let mesh = Mesh::box_at((0.0, 5.0, 0.0), (4.0, 10.0, 4.0), rgb(200, 180, 150))
            .expect("box")
            .build();
        for seed in 1..6 {
            let pieces = fracture(
                &mesh,
                Mat4::from_translation(Vec3::new(190.0, 40.0, 180.0)),
                Vec3::new(188.0, 45.0, 180.5),
                FractureSettings {
                    pieces: 24,
                    seed,
                    ..Default::default()
                },
            );
            let mut total = 0.0;
            for f in &pieces {
                let m = &f.mesh;
                let mut vol = 0.0;
                for t in m.indices.chunks_exact(3) {
                    let (a, b, c) = (
                        m.positions[t[0] as usize],
                        m.positions[t[1] as usize],
                        m.positions[t[2] as usize],
                    );
                    vol += a.dot(b.cross(c)) / 6.0;
                }
                total += vol;
            }
            assert!((total - 160.0).abs() < 1.0, "seed {seed}: volume {total}");
        }
    }
}
