//! Procedural rocks, ported from AssetGenerator's rocks package: boulders,
//! blocks, slabs and spires.
//!
//! A rock is an icosphere stretched to its shape, split by a few planes
//! (flat faces that keep a little of the lumps), roughened by noise and, for
//! layered shapes, banded by weathered strata. Its foot is pressed flat at
//! `y = 0`, its middle at the origin, so it sits on the ground where it is
//! placed. Rocks are about 2 m across; scale them with the instance transform.
//!
//! The same `(shape, seed)` gives the same rock at every detail, so
//! [`rock_model`] returns three LOD meshes for
//! [`crate::world::World::spawn_lod`]. Rocks are textured with a
//! [`crate::texgen`] layer (weathered rock, or banded strata for slabs and
//! spires) projected triplanar, so the texture follows the rock however it is
//! placed. Vertex colours darken the cut corners and the foot, and vary the
//! tint a little from rock to rock.
//!
//! ```no_run
//! use engine::rocks::{rock_model, RockShape};
//! # fn demo(world: &mut engine::World) -> engine::EngineResult<()> {
//! let rock = rock_model(RockShape::Boulder, 7)?;
//! world.spawn_lod(rock.lod_levels())?;
//! # Ok(()) }
//! ```

use glam::Vec3;

use crate::color::Color;
use crate::error::EngineResult;
use crate::lod::LodLevel;
use crate::mesh::{Mesh, SurfaceMaterial};
use crate::texgen::{natural_layer, NaturalLayer};
use crate::vegetation::rng::{noise3, Rng};

/// The four kinds of rock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RockShape {
    /// Rounded, lumpy, a few broad faces.
    Boulder,
    /// Boxy and many-faced, as broken off a cliff.
    Block,
    /// Flat and wide, split along level strata.
    Slab,
    /// Tall and narrow, upright faces.
    Spire,
}

impl RockShape {
    pub const ALL: [RockShape; 4] = [Self::Boulder, Self::Block, Self::Slab, Self::Spire];

    pub fn name(self) -> &'static str {
        match self {
            Self::Boulder => "boulder",
            Self::Block => "block",
            Self::Slab => "slab",
            Self::Spire => "spire",
        }
    }

    /// The texture a rock of this shape wears by default.
    pub fn default_texture(self) -> NaturalLayer {
        match self {
            Self::Boulder | Self::Block => NaturalLayer::RockFace,
            Self::Slab | Self::Spire => NaturalLayer::CliffStrata,
        }
    }

    fn rule(self) -> Rule {
        match self {
            Self::Boulder => Rule {
                size: [1.05, 0.8, 0.9],
                cuts: (3, 5),
                keep: (0.7, 0.88),
                level: 0.0,
                rough: 0.07,
                buried: 0.25,
                exponent: 2.6,
                strata: 0.0,
            },
            Self::Block => Rule {
                size: [1.0, 0.85, 0.9],
                cuts: (6, 9),
                keep: (0.62, 0.85),
                level: 0.2,
                rough: 0.035,
                buried: 0.15,
                exponent: 4.5,
                strata: 0.03,
            },
            Self::Slab => Rule {
                size: [1.25, 0.42, 0.95],
                cuts: (4, 6),
                keep: (0.65, 0.88),
                level: 0.6,
                rough: 0.04,
                buried: 0.2,
                exponent: 4.0,
                strata: 0.04,
            },
            Self::Spire => Rule {
                size: [0.65, 1.7, 0.62],
                cuts: (5, 7),
                keep: (0.66, 0.86),
                level: -0.5,
                rough: 0.04,
                buried: 0.1,
                exponent: 4.0,
                strata: 0.06,
            },
        }
    }
}

struct Rule {
    /// Half sizes before the cuts (m).
    size: [f32; 3],
    /// How many planes cut it.
    cuts: (u32, u32),
    /// Share of the radius each cut leaves standing (lower cuts deeper).
    keep: (f32, f32),
    /// 1: cuts lie level (strata), -1: they stand upright.
    level: f32,
    /// Noise roughness (share of the size).
    rough: f32,
    /// Share of the height below the ground.
    buried: f32,
    /// Norm exponent: 2 round, 4 a rounded box.
    exponent: f32,
    /// Depth of the weathered bands (share of the size), 0 for none.
    strata: f32,
}

/// One rock mesh with its size.
#[derive(Clone, Debug)]
pub struct RockMesh {
    pub mesh: Mesh,
    /// Farthest any point near the ground lies from the middle (m): what a walker goes round.
    pub foot: f32,
    /// Top above the ground (m).
    pub height: f32,
    /// Bounding sphere radius round `(0, height / 2, 0)`.
    pub radius: f32,
}

/// A rock at three detail levels.
#[derive(Clone, Debug)]
pub struct RockModel {
    /// 1280 triangles.
    pub near: Mesh,
    /// 320 triangles.
    pub mid: Mesh,
    /// 80 triangles.
    pub far: Mesh,
    pub foot: f32,
    pub height: f32,
    pub radius: f32,
}

impl RockModel {
    /// LOD levels for [`crate::world::World::spawn_lod`] at the rock's
    /// natural 2 m size: 25 m, 70 m, 400 m (multiply by `scale` when the rock
    /// is placed larger or smaller).
    pub fn lod_levels(&self) -> Vec<LodLevel> {
        self.lod_levels_scaled(1.0)
    }

    /// LOD levels for a rock placed at `scale` times its natural size.
    pub fn lod_levels_scaled(&self, scale: f32) -> Vec<LodLevel> {
        let k = scale.max(0.05);
        vec![
            LodLevel::new(self.near.clone(), 25.0 * k),
            LodLevel::new(self.mid.clone(), 70.0 * k),
            LodLevel::new(self.far.clone(), 400.0 * k),
        ]
    }
}

/// The rock of `shape` and `seed` at three detail levels, in its
/// [`RockShape::default_texture`].
pub fn rock_model(shape: RockShape, seed: u32) -> EngineResult<RockModel> {
    rock_model_with(shape, seed, Some(shape.default_texture()))
}

/// The rock of `shape` and `seed` in `texture` (`None`: plain grey stone).
pub fn rock_model_with(
    shape: RockShape,
    seed: u32,
    texture: Option<NaturalLayer>,
) -> EngineResult<RockModel> {
    let near = rock_mesh(shape, seed, 2, texture)?;
    let mid = rock_mesh(shape, seed, 1, texture)?;
    let far = rock_mesh(shape, seed, 0, texture)?;
    Ok(RockModel {
        near: near.mesh,
        mid: mid.mesh,
        far: far.mesh,
        foot: near.foot,
        height: near.height,
        radius: near.radius,
    })
}

/// The rock of `shape` and `seed` at `detail` (0..=4 sphere subdivisions:
/// 80, 320, 1280, 5120, 20480 triangles), in `texture` (`None`: plain
/// grey stone).
pub fn rock_mesh(
    shape: RockShape,
    seed: u32,
    detail: u32,
    texture: Option<NaturalLayer>,
) -> EngineResult<RockMesh> {
    let g = generate(shape, seed, detail.min(4));
    // A texture carries the stone's colour; plain rocks get a grey.
    let base = if texture.is_some() { 1.0 } else { 0.235 };
    let mut mesh = Mesh::new();
    let mut ids = Vec::with_capacity(g.pos.len());
    for i in 0..g.pos.len() {
        let id = mesh.add_point(g.pos[i])?;
        mesh.set_point_normal(id, g.normals[i])?;
        let k = base * g.ao[i];
        mesh.set_point_color(
            id,
            Color {
                r: g.tint[0] * k,
                g: g.tint[1] * k,
                b: g.tint[2] * k,
                a: 1.0,
            },
        )?;
        ids.push(id);
    }
    for t in &g.tris {
        mesh.add_triangle(ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize])?;
    }
    let mut material = SurfaceMaterial::STONE.with_seed(17.0 + (seed % 997) as f32);
    if shape == RockShape::Boulder {
        material.strata_strength = 0.1;
    }
    mesh.set_surface_material(material);
    if let Some(layer) = texture {
        let layer = natural_layer(layer);
        // A little finer than the layer's own tile, to suit a 2 m rock.
        layer.apply_scaled(&mut mesh, layer.tile_metres * 0.6)?;
    }
    Ok(RockMesh {
        mesh,
        foot: g.foot,
        height: g.height,
        radius: g.radius,
    })
}

struct Generated {
    pos: Vec<Vec3>,
    normals: Vec<Vec3>,
    ao: Vec<f32>,
    tris: Vec<[u32; 3]>,
    tint: [f32; 3],
    foot: f32,
    height: f32,
    radius: f32,
}

fn fbm3(seed: u32, p: Vec3, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, 1.0);
    for o in 0..octaves {
        let off = Vec3::new(17.31, -9.17, 5.71) * o as f32;
        sum += amp * noise3(seed.wrapping_add(o), p * f + off);
        norm += amp;
        amp *= 0.5;
        f *= 2.0;
    }
    sum / norm
}

fn generate(shape: RockShape, seed: u32, detail: u32) -> Generated {
    let rule = shape.rule();
    let mut rng = Rng::from(&["rock", shape.name()], seed);
    let nseed = (rng.float() * 4_294_967_000.0) as u32;
    let (verts, tris) = icosphere(detail);
    let s = Vec3::new(
        rule.size[0] * rng.range(0.85, 1.15),
        rule.size[1] * rng.range(0.85, 1.15),
        rule.size[2] * rng.range(0.85, 1.15),
    );
    // Cutting planes: unit normal and the share of the radius left standing.
    let mut cuts = Vec::new();
    for _ in 0..rng.int(rule.cuts.0, rule.cuts.1) {
        let (nx, mut ny, nz) = (
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        );
        // Strata lie level, a spire's faces stand upright; the foot is never
        // cut (the ground does that).
        ny = if rule.level >= 0.0 {
            let sign = if ny < 0.0 { -1.0 } else { 1.0 };
            ny * (1.0 - rule.level) + sign * rule.level
        } else {
            ny * (1.0 + rule.level)
        };
        if ny < -0.5 {
            ny = -ny;
        }
        let n = Vec3::new(nx, ny, nz).try_normalize().unwrap_or(Vec3::Y);
        cuts.push((n, rng.range(rule.keep.0, rule.keep.1)));
    }
    // A tint per rock: a little warmer or cooler, lighter or darker.
    let warm = rng.range(-1.0, 1.0);
    let light = rng.range(0.85, 1.15);
    let tint = [
        (1.06 + 0.08 * warm) * light,
        light,
        (0.92 - 0.08 * warm) * light,
    ];

    let m = rule.exponent;
    let mut pos = Vec::with_capacity(verts.len());
    let mut cut_depth = vec![0.0f32; verts.len()];
    for (i, d) in verts.iter().enumerate() {
        // Noise on the unit sphere, so every detail samples the same field.
        let lump = 1.0
            + rule.rough
                * (fbm3(nseed, *d * 1.7, 3) * 2.0
                    + fbm3(nseed ^ 0x5bd1, *d * 5.3 + Vec3::X * 7.0, 2) * 0.6);
        // Onto a rounded box (the unit sphere of the norm), then to size.
        let k = (d.x.abs().powf(m) + d.y.abs().powf(m) + d.z.abs().powf(m)).powf(-1.0 / m);
        let band = if rule.strata > 0.0 {
            1.0 - rule.strata
                * (d.y * s.y * 7.5 + noise3(nseed ^ 0x3c6e, *d * 2.0) * 1.5)
                    .sin()
                    .abs()
        } else {
            1.0
        };
        let mut p = *d * k * s * lump * Vec3::new(band, 1.0, band);
        for (n, keep) in &cuts {
            let reach = (*n * s).length() * keep;
            let over = p.dot(*n) - reach;
            if over > 0.0 {
                // Not quite flat: a little of the lump stays on the face.
                p -= *n * (over * 0.92);
                cut_depth[i] = cut_depth[i].max(over);
            }
        }
        pos.push(p);
    }
    // The foot: press everything below the ground line flat, then lift the
    // rock so it sits at y = 0.
    let (lo, hi) = pos.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
        (lo.min(p.y), hi.max(p.y))
    });
    let ground = lo + (hi - lo) * rule.buried;
    let (mut foot, mut height) = (0.0f32, 0.0f32);
    for p in &mut pos {
        p.y = p.y.max(ground - (ground - p.y) * 0.1) - ground;
        height = height.max(p.y);
        if p.y < 0.3 {
            foot = foot.max(Vec3::new(p.x, 0.0, p.z).length());
        }
    }
    let centre = Vec3::new(0.0, height * 0.5, 0.0);
    let radius = pos
        .iter()
        .fold(0.0f32, |r, p| r.max((*p - centre).length()));
    let normals = vertex_normals(&pos, &tris);
    // Occlusion: darker toward the foot and in the cut corners.
    let ao = pos
        .iter()
        .zip(&normals)
        .zip(&cut_depth)
        .map(|((p, n), cut)| {
            let up = (p.y / (height * 0.5).max(0.3)).clamp(0.0, 1.0);
            ((0.5 + 0.5 * up) * (1.0 - 0.25 * (cut * 1.5).min(1.0)) * (0.85 + 0.15 * n.y.max(0.0)))
                .clamp(0.25, 1.0)
        })
        .collect();
    Generated {
        pos,
        normals,
        ao,
        tris,
        tint,
        foot,
        height,
        radius,
    }
}

/// A unit icosphere subdivided `levels` times.
fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut verts: Vec<Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|v| Vec3::from_array(*v).normalize())
    .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..levels {
        let mut mid = std::collections::HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| {
            let key = (a.min(b), a.max(b));
            *mid.entry(key).or_insert_with(|| {
                verts.push((verts[a as usize] + verts[b as usize]).normalize());
                verts.len() as u32 - 1
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    (verts, faces)
}

/// Smooth normals: each vertex the area-weighted sum of its triangles' normals.
fn vertex_normals(pos: &[Vec3], tris: &[[u32; 3]]) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; pos.len()];
    for t in tris {
        let [a, b, c] = t.map(|i| pos[i as usize]);
        let n = (b - a).cross(c - a);
        for i in t {
            out[*i as usize] += n;
        }
    }
    out.iter()
        .map(|n| n.try_normalize().unwrap_or(Vec3::Y))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shape_sits_on_its_foot_with_outward_normals() {
        for shape in RockShape::ALL {
            for seed in [1, 2, 3] {
                let g = generate(shape, seed, 2);
                let lowest = g.pos.iter().fold(f32::MAX, |m, p| m.min(p.y));
                assert!(lowest > -0.2 && lowest <= 0.0, "{shape:?} lowest {lowest}");
                let centre = Vec3::new(0.0, g.height * 0.5, 0.0);
                let outward = g
                    .pos
                    .iter()
                    .zip(&g.normals)
                    .filter(|(p, n)| (**p - centre).dot(**n) > 0.0)
                    .count();
                assert!(
                    outward as f32 > 0.9 * g.pos.len() as f32,
                    "{shape:?} normals"
                );
                assert!(g.foot > 0.3 && g.foot < 1.6, "{shape:?} foot {}", g.foot);
            }
        }
    }

    #[test]
    fn shapes_have_their_proportions() {
        for seed in 0..6 {
            let spire = generate(RockShape::Spire, seed, 2);
            assert!(spire.height > 1.4 * spire.foot);
            let slab = generate(RockShape::Slab, seed, 2);
            assert!(slab.height < slab.foot);
        }
    }

    #[test]
    fn same_seed_is_the_same_rock_at_every_detail() {
        for shape in RockShape::ALL {
            let (a, b) = (generate(shape, 9, 1), generate(shape, 9, 3));
            assert!((a.height - b.height).abs() < 0.12, "{shape:?} height");
            assert!((a.foot - b.foot).abs() < 0.15, "{shape:?} foot");
            assert_eq!(b.tris.len(), a.tris.len() * 16);
        }
        let m = rock_model(RockShape::Block, 4).expect("rock");
        assert!(m.near.face_count() > m.mid.face_count());
        assert!(m.mid.face_count() > m.far.face_count());
        assert!(crate::lod::LodSet::validate_levels(&m.lod_levels()).is_ok());
    }
}
