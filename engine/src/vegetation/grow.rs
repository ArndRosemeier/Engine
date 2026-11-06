//! Tree and shrub geometry: species-shaped branching wood, leaf-card crowns
//! with baked crown occlusion, and far canopy hulls. A port of
//! AssetGenerator's `packages/vegetation/src/trees.ts`.
//!
//! Every branch is grown inside a species crown envelope, so silhouettes read
//! as the real species. Leaves are alpha-tested cards carrying whole twig
//! clusters from the leaf atlas, with crown-ellipsoid normals for soft canopy
//! lighting.

use super::rng::{noise2, noise3, smoothstep, Rng};
use super::species::{species_cfg, CardMode, Kind, LenRef, SpeciesCfg};
use glam::{Quat, Vec3};
use std::f32::consts::{PI, TAU};

const DEG: f32 = PI / 180.0;
const UP: Vec3 = Vec3::Y;

/// One vertex of a vegetation part, before it becomes a [`crate::mesh::Mesh`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct VegVertex {
    pub p: Vec3,
    pub n: Vec3,
    /// Bark: `strip * 16 + u` around the branch and `v` along it. Leaves:
    /// leaf-atlas uv. Far: unused.
    pub uv: [f32; 2],
    /// Sway weight: 0 at the root, 1 at the outermost twigs.
    pub wind: f32,
    /// Per-tuft random 0..1 (leaf tint variation, flutter phase).
    pub tint: f32,
    /// Baked crown occlusion: 0 in the open, 1 deep inside.
    pub occ: f32,
    /// Linear colour (far hulls; white elsewhere).
    pub color: [f32; 3],
    /// Linear autumn colour (far hulls).
    pub autumn: [f32; 3],
    /// Far hulls: 0 bark, 0.5 palm frond ribbon, 1 canopy. Leaves: unused.
    pub kind: f32,
    /// Far hulls: 1 for deciduous canopies (turn in autumn).
    pub deciduous: f32,
}

#[derive(Default, Clone, Debug)]
pub(crate) struct Part {
    pub verts: Vec<VegVertex>,
    pub tris: Vec<[u32; 3]>,
}

impl Part {
    #[allow(clippy::too_many_arguments)]
    fn vert(&mut self, p: Vec3, n: Vec3, uv: [f32; 2], wind: f32, tint: f32, occ: f32) -> u32 {
        self.verts.push(VegVertex {
            p,
            n,
            uv,
            wind,
            tint,
            occ,
            color: [1.0; 3],
            autumn: [1.0; 3],
            kind: 0.0,
            deciduous: 0.0,
        });
        (self.verts.len() - 1) as u32
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.tris.push([a, b, c]);
    }
    pub fn max_y(&self) -> f32 {
        self.verts.iter().map(|v| v.p.y).fold(0.0, f32::max)
    }
}

/// Wood, leaves and far hull of one tree or shrub.
#[derive(Clone, Debug)]
pub(crate) struct Grown {
    pub wood: Part,
    pub leaves: Part,
    pub far: Part,
    pub height: f32,
    pub trunk_radius: f32,
    pub crown_radius: f32,
}

fn perp_basis(t: Vec3) -> (Vec3, Vec3) {
    let r = if t.y.abs() < 0.92 { UP } else { Vec3::X };
    let p1 = t.cross(r).normalize();
    let p2 = t.cross(p1).normalize();
    (p1, p2)
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn lin3(c: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear(c[0]),
        srgb_to_linear(c[1]),
        srgb_to_linear(c[2]),
    ]
}

// -------------------------------------------------------------------------
// Crown envelope
// -------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Envelope {
    base: f32,
    top: f32,
    rx: f32,
    rz: f32,
    prof: fn(f32) -> f32,
    cx: f32,
    cz: f32,
}

impl Envelope {
    fn cy(&self) -> f32 {
        (self.base + self.top) * 0.5
    }
    fn ry(&self) -> f32 {
        (self.top - self.base) * 0.5
    }
    fn centre(&self) -> Vec3 {
        Vec3::new(self.cx, self.cy(), self.cz)
    }
    /// Normalised radius: about 1 on the envelope surface.
    fn rel(&self, p: Vec3) -> f32 {
        let t = (p.y - self.base) / (self.top - self.base);
        if t > 1.0 {
            return 1.0 + (t - 1.0) * 4.0;
        }
        let f = (self.prof)(t.clamp(0.0, 1.0)).max(0.02);
        let dx = (p.x - self.cx) / (self.rx * f);
        let dz = (p.z - self.cz) / (self.rz * f);
        let r = (dx * dx + dz * dz).sqrt();
        if t < 0.0 {
            r.max(1.0 - t * 3.0)
        } else {
            r
        }
    }
    fn inside(&self, p: Vec3) -> bool {
        self.rel(p) <= 1.0
    }
    /// Ellipsoid normal and depth (0 centre, 1 surface) for lighting.
    fn shade(&self, p: Vec3) -> (Vec3, f32) {
        let o = Vec3::new(
            (p.x - self.cx) / self.rx,
            (p.y - self.cy()) / self.ry(),
            (p.z - self.cz) / self.rz,
        );
        let d = o.length();
        if d < 1e-4 {
            (UP, d)
        } else {
            (o / d, d)
        }
    }
    /// Distance from the centre to the envelope along `dir`.
    fn extent(&self, dir: Vec3) -> f32 {
        let c = self.centre();
        let max_r = self.rx.max(self.rz).max(self.ry()) * 1.6;
        let mut last = 0.0;
        let mut s = 0.05;
        while s < max_r {
            if self.inside(c + dir * s) {
                last = s;
            } else if s > last + 1.0 {
                break;
            }
            s += 0.08;
        }
        last
    }
}

// -------------------------------------------------------------------------
// Branch growth
// -------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Branch {
    pts: Vec<Vec3>,
    rad: Vec<f32>,
    arc: Vec<f32>,
    level: usize,
    /// Path distance from the root at the branch base.
    d0: f32,
    children: usize,
}

impl Branch {
    fn new(pts: Vec<Vec3>, r0: f32, r_tip: f32, level: usize, d0: f32, taper: f32) -> Self {
        let mut arc = vec![0.0];
        for i in 1..pts.len() {
            arc.push(arc[i - 1] + pts[i].distance(pts[i - 1]));
        }
        let total = arc.last().copied().filter(|t| *t > 0.0).unwrap_or(1.0);
        let rad = arc
            .iter()
            .map(|a| r_tip + (r0 - r_tip) * (1.0 - a / total).max(0.0).powf(taper))
            .collect();
        Self {
            pts,
            rad,
            arc,
            level,
            d0,
            children: 0,
        }
    }
    fn len(&self) -> f32 {
        *self.arc.last().unwrap_or(&0.0)
    }
    /// Point, tangent and radius at fraction `t` along the branch.
    fn sample(&self, t: f32) -> (Vec3, Vec3, f32) {
        let total = self.len();
        let s = t.clamp(0.0, 1.0) * total;
        let mut i = 1;
        while i < self.arc.len() - 1 && self.arc[i] < s {
            i += 1;
        }
        let (a0, a1) = (self.arc[i - 1], self.arc[i]);
        let f = if a1 > a0 { (s - a0) / (a1 - a0) } else { 0.0 };
        let p = self.pts[i - 1].lerp(self.pts[i], f);
        let tan = (self.pts[i] - self.pts[i - 1]).normalize_or(UP);
        (
            p,
            tan,
            self.rad[i - 1] + (self.rad[i] - self.rad[i - 1]) * f,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn grow_path(
    rng: &mut Rng,
    start: Vec3,
    dir: Vec3,
    len: f32,
    segs: u32,
    up: f32,
    curve: f32,
    env: Option<&Envelope>,
    min_len: f32,
) -> Vec<Vec3> {
    let mut pts = vec![start];
    let mut d = dir.normalize();
    let mut p = start;
    let seg_l = len / segs as f32;
    let mut acc = 0.0;
    for i in 0..segs {
        if i > 0 {
            d += UP * (up * seg_l);
            d += rng.on_sphere() * (curve * seg_l.sqrt() * 0.45);
            d = d.normalize();
        }
        let np = p + d * seg_l;
        if let Some(env) = env {
            if acc + seg_l > min_len && !env.inside(np) {
                // Bisect to the envelope boundary.
                let (mut lo, mut hi) = (0.0f32, 1.0f32);
                for _ in 0..6 {
                    let m = (lo + hi) / 2.0;
                    if env.inside(p + d * (seg_l * m)) {
                        lo = m;
                    } else {
                        hi = m;
                    }
                }
                if lo * seg_l > 0.08 {
                    pts.push(p + d * (seg_l * lo));
                }
                break;
            }
        }
        p = np;
        acc += seg_l;
        pts.push(p);
    }
    pts
}

struct Skeleton {
    branches: Vec<Branch>,
    env: Envelope,
    height: f32,
    max_path: f32,
}

fn build_skeleton(cfg: &SpeciesCfg, rng: &mut Rng) -> Skeleton {
    let h = rng.range(cfg.height.0, cfg.height.1);
    let w_scale = rng.range(0.88, 1.12);
    let mut branches: Vec<Branch> = Vec::new();

    // Trunk.
    let mut lean_dir = rng.on_sphere();
    lean_dir.y = 0.0;
    let lean_dir = lean_dir.normalize_or(Vec3::X);
    let trunk_dir = (UP + lean_dir * (cfg.lean * rng.range(0.4, 1.2))).normalize();
    let trunk_len = (h * cfg.trunk_top).max(0.05);
    let trunk_segs = if trunk_len < 0.5 {
        1
    } else {
        ((trunk_len / 0.7).round() as u32).max(6)
    };
    let t_pts = grow_path(
        rng,
        Vec3::new(0.0, -0.15, 0.0),
        trunk_dir,
        trunk_len + 0.15,
        trunk_segs,
        cfg.trunk_up,
        cfg.trunk_curve,
        None,
        0.0,
    );
    let excurrent = cfg.trunk_top > 0.8;
    let trunk_r = cfg.trunk_r * rng.range(0.9, 1.1);
    let top = *t_pts.last().expect("trunk has points");
    branches.push(Branch::new(
        t_pts,
        trunk_r * 1.05,
        if excurrent { 0.02 } else { trunk_r * 0.78 },
        0,
        0.0,
        if excurrent { 0.9 } else { 1.0 },
    ));

    let rx = cfg.width * w_scale * rng.range(0.92, 1.08);
    let rz = cfg.width * w_scale * rng.range(0.92, 1.08);
    let cx = if excurrent {
        top.x * 0.4 + rng.range(-0.2, 0.2)
    } else {
        top.x + rng.range(-0.3, 0.3)
    };
    let cz = if excurrent {
        top.z * 0.4 + rng.range(-0.2, 0.2)
    } else {
        top.z + rng.range(-0.3, 0.3)
    };
    let env = Envelope {
        base: h * cfg.crown_base,
        top: h,
        rx,
        rz,
        prof: cfg.prof,
        cx,
        cz,
    };

    // Depth-first like the original, so branch order (and wind paths) match.
    fn spawn(
        cfg: &SpeciesCfg,
        env: &Envelope,
        rng: &mut Rng,
        branches: &mut Vec<Branch>,
        parent: usize,
        li: usize,
    ) {
        let Some(lv) = cfg.levels.get(li).copied() else {
            return;
        };
        let nodes = rng.int(lv.n.0, lv.n.1);
        let per_node = match lv.whorl {
            Some(w) if lv.start < 1.0 => w,
            _ => 1,
        };
        let is_top = lv.start >= 1.0;
        let mut az = rng.range(0.0, TAU);
        let p_len = branches[parent].len();
        for k in 0..nodes {
            let t = if is_top {
                1.0
            } else {
                lv.start
                    + (lv.end - lv.start)
                        * ((k as f32 + 0.5 + rng.range(-0.35, 0.35)) / nodes as f32)
            };
            let (p, tan, pr) = branches[parent].sample(t);
            for w in 0..per_node {
                let mut a = az
                    + if per_node > 1 {
                        (w as f32 / per_node as f32) * TAU + rng.range(-0.25, 0.25)
                    } else {
                        0.0
                    };
                if is_top {
                    a = az + (k as f32 / nodes as f32) * TAU + rng.range(-0.3, 0.3);
                }
                let ang = (lv.angle + rng.range(-lv.angle_var, lv.angle_var)) * DEG;
                let (p1, p2) = perp_basis(tan);
                let mk = |aa: f32| {
                    let radial = p1 * aa.cos() + p2 * aa.sin();
                    (tan * ang.cos() + radial * ang.sin()).normalize()
                };
                let mut dir = mk(a);
                if dir.y < -0.3 && li > 0 {
                    dir = mk(a + PI);
                }
                let mut len = match lv.len_ref {
                    LenRef::Width => lv.len * env.rx,
                    LenRef::Parent => lv.len * p_len * (1.0 - t * 0.45),
                };
                len *= rng.range(0.75, 1.1);
                let segs = lv.segs.max(1);
                let min_len = len.min(match lv.len_ref {
                    LenRef::Width => 0.8,
                    LenRef::Parent => 0.25,
                });
                let pts = grow_path(rng, p, dir, len, segs, lv.up, lv.curve, Some(env), min_len);
                if pts.len() < 2 {
                    continue;
                }
                let r0 = (pr * lv.rr * rng.range(0.85, 1.05)).max(0.008);
                let d0 = branches[parent].d0 + t * p_len;
                branches.push(Branch::new(
                    pts,
                    r0,
                    (r0 * 0.18).max(0.004),
                    li + 1,
                    d0,
                    1.0,
                ));
                branches[parent].children += 1;
                let me = branches.len() - 1;
                spawn(cfg, env, rng, branches, me, li + 1);
            }
            az += lv.phyllo * DEG + rng.range(-0.3, 0.3);
        }
    }
    spawn(cfg, &env, rng, &mut branches, 0, 0);

    let mut max_path: f32 = 1.0;
    let mut max_y: f32 = 0.0;
    for b in &branches {
        max_path = max_path.max(b.d0 + b.len());
        for p in &b.pts {
            max_y = max_y.max(p.y);
        }
    }
    Skeleton {
        branches,
        env,
        height: h.max(max_y),
        max_path,
    }
}

// -------------------------------------------------------------------------
// Wood
// -------------------------------------------------------------------------

/// Bark texture strips in the bark atlas (wood `uv.x = strip * 16 + u`).
pub const BARK_STRIPS: u32 = 8;

#[allow(clippy::too_many_arguments)]
fn tube(
    part: &mut Part,
    b: &Branch,
    radial: u32,
    strip: u32,
    max_path: f32,
    env: Option<&Envelope>,
    trunk: Option<&SpeciesCfg>,
    seed: u32,
) {
    let n = b.pts.len();
    if n < 2 {
        return;
    }
    // Parallel-transport frames.
    let tans: Vec<Vec3> = (0..n)
        .map(|i| (b.pts[(i + 1).min(n - 1)] - b.pts[i.saturating_sub(1)]).normalize_or(UP))
        .collect();
    let (mut nx, mut ny) = perp_basis(tans[0]);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        if i > 0 {
            let q = Quat::from_rotation_arc(tans[i - 1], tans[i]);
            nx = q * nx;
            ny = tans[i].cross(nx).normalize();
            nx = ny.cross(tans[i]).normalize();
        }
        frames.push((nx, ny));
    }
    let r0 = b.rad[0];
    let wraps = ((TAU * r0) / 0.55).round().max(1.0);
    let wrap_width = (TAU * r0.max(0.02)) / wraps;
    let base = part.verts.len() as u32;
    let mut v = (seed % 97) as f32 * 0.173;
    for (i, &(fx, fy)) in frames.iter().enumerate() {
        if i > 0 {
            v += b.pts[i].distance(b.pts[i - 1]) / (wrap_width * 4.0 * (b.rad[i] / r0).max(0.35));
        }
        let c = b.pts[i];
        let mut r = b.rad[i];
        let y = c.y;
        if let Some(t) = trunk {
            r *= 1.0 + t.flare * (-y.max(0.0) / 0.45).exp();
        }
        let w = (b.d0 + b.arc[i]) / max_path;
        let mut occ = 0.0;
        if let Some(env) = env {
            let (_, d) = env.shade(c);
            if d < 1.05 {
                occ = 0.55 * (1.0 - smoothstep(0.2, 1.05, d));
            }
        }
        occ = occ.max(0.5 * (-y.max(0.0) / 0.35).exp());
        for j in 0..=radial {
            let a = (j as f32 / radial as f32) * TAU;
            let (sa, ca) = a.sin_cos();
            let nn = fx * ca + fy * sa;
            let mut rr = r;
            if let Some(t) = trunk.filter(|t| t.lobes > 0.0) {
                let k = noise2(0x7ee5, ca * 1.3 + seed as f32 * 0.01, y * 0.35) * 0.6
                    + noise2(0x7ee5, sa * 2.1, y * 0.9 + 4.2) * 0.4;
                rr *= 1.0 + t.lobes * k * (1.0 + 1.5 * (-y.max(0.0) / 0.8).exp());
            }
            part.vert(
                c + nn * rr,
                nn,
                [strip as f32 * 16.0 + (j as f32 / radial as f32) * wraps, v],
                w,
                0.0,
                occ,
            );
        }
    }
    let rs = radial + 1;
    for i in 0..(n as u32 - 1) {
        for j in 0..radial {
            let a = base + i * rs + j;
            let b2 = a + rs;
            part.tri(a, a + 1, b2);
            part.tri(a + 1, b2 + 1, b2);
        }
    }
}

fn build_wood(cfg: &SpeciesCfg, sk: &Skeleton, seed: u32, max_level: usize) -> Part {
    let mut part = Part::default();
    for (i, b) in sk.branches.iter().enumerate() {
        if b.level > max_level {
            continue;
        }
        let radial = if b.level == 0 {
            if cfg.trunk_r > 0.1 {
                10
            } else {
                6
            }
        } else {
            cfg.levels[b.level - 1].radial
        };
        let (env, trunk) = if b.level == 0 {
            (None, Some(cfg))
        } else {
            (Some(&sk.env), None)
        };
        tube(
            &mut part,
            b,
            radial,
            cfg.bark,
            sk.max_path,
            env,
            trunk,
            seed + i as u32 * 7,
        );
    }
    part
}

// -------------------------------------------------------------------------
// Leaf cards
// -------------------------------------------------------------------------

const ATLAS_GRID: u32 = 4;

/// Leaf-atlas uv for a cell: (u, v) in 0..1, v = 0 at the card base.
fn cell_uv(cell: u32, u: f32, v: f32, wide_cells: f32) -> [f32; 2] {
    let (cx, cy) = ((cell % ATLAS_GRID) as f32, (cell / ATLAS_GRID) as f32);
    let m = 0.006;
    [
        (cx + m + u * (wide_cells - 2.0 * m)) / ATLAS_GRID as f32,
        (cy + 1.0 - m - v * (1.0 - 2.0 * m)) / ATLAS_GRID as f32,
    ]
}

#[allow(clippy::too_many_arguments)]
fn add_card(
    part: &mut Part,
    base: Vec3,
    axis: Vec3,
    side: Vec3,
    w: f32,
    h: f32,
    cell: u32,
    sph_n: Vec3,
    card_n: Vec3,
    wind: f32,
    tint: f32,
    occ: f32,
    flip: bool,
) {
    let s = if card_n.dot(sph_n) >= 0.0 { 1.0 } else { -1.0 };
    let n = (sph_n * 0.78 + card_n * (0.22 * s)).normalize();
    let i0 = part.verts.len() as u32;
    for (a, b) in [(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)] {
        let p = base + side * (a * w) + axis * (b * h);
        let u = if flip { 0.5 - a } else { a + 0.5 };
        part.vert(
            p,
            n,
            cell_uv(cell, u, b, 1.0),
            (wind + b * 0.12).min(1.0),
            tint,
            occ,
        );
    }
    // Both faces: cards are seen from either side.
    part.tri(i0, i0 + 1, i0 + 2);
    part.tri(i0, i0 + 2, i0 + 3);
    part.tri(i0, i0 + 2, i0 + 1);
    part.tri(i0, i0 + 3, i0 + 2);
}

/// `keep`: fraction of tufts kept (coarser levels), each grown to hold the leaf area.
fn build_leaves(cfg: &SpeciesCfg, sk: &Skeleton, rng: &mut Rng, keep: f32) -> Part {
    let mut part = Part::default();
    let env = sk.env;
    let leaf_levels = cfg
        .leaf_levels
        .clone()
        .unwrap_or_else(|| vec![cfg.levels.len()]);
    let max_level = cfg.levels.len();
    let grow = (1.0 / keep).powf(0.4);

    let place_card =
        |part: &mut Part, rng: &mut Rng, pt: Vec3, tan: Vec3, w_path: f32, size_mul: f32| {
            // Draw every random number whether or not the card is kept, so all
            // levels place their cards in the same spots.
            let kept = rng.float() < keep;
            let mut pt = pt;
            let size = cfg.leaf_size
                * 1.3
                * size_mul
                * rng.range(0.8, 1.2)
                * if keep < 1.0 { grow } else { 1.0 };
            if cfg.card_mode != CardMode::Flat {
                // Unmodelled twigs: scatter the tuft round the branch, biased outwards.
                let (q1, q2) = perp_basis(tan);
                let a = rng.range(0.0, TAU);
                let mut off = q1 * a.cos() + q2 * a.sin();
                let to_out = (pt - env.centre()).normalize_or(UP);
                if off.dot(to_out) < 0.0 {
                    off *= -0.6;
                }
                pt += off * rng.range(0.0, cfg.leaf_size * 0.7);
            }
            let (sph, d) = env.shade(pt);
            let mut outward = Vec3::new(pt.x - env.cx, 0.0, pt.z - env.cz);
            if outward.length_squared() < 1e-4 {
                outward = rng.on_sphere();
                outward.y = 0.0;
            }
            let outward = outward.normalize_or(Vec3::X);
            let (axis, mut nrm0) = match cfg.card_mode {
                CardMode::Flat => {
                    let side_dir = tan.cross(UP).normalize_or(Vec3::X) * rng.sign();
                    let axis = (tan + side_dir * rng.range(0.3, 1.0) + UP * rng.range(-0.45, 0.35))
                        .normalize();
                    pt.y += rng.range(-0.25, 0.2);
                    (axis, UP + rng.on_sphere() * 0.25)
                }
                CardMode::Up => {
                    let axis = (UP * 1.2 + outward * rng.range(0.2, 0.7) + rng.on_sphere() * 0.35)
                        .normalize();
                    (axis, outward + rng.on_sphere() * 0.6)
                }
                CardMode::Tuft => {
                    let axis =
                        (UP * 0.9 + tan * 0.6 + outward * 0.3 + rng.on_sphere() * 0.45).normalize();
                    (axis, rng.on_sphere())
                }
                CardMode::Cluster => {
                    let rnd = rng.on_sphere();
                    let axis = (tan * 0.55 + outward * 0.5 + rnd * 0.75 + UP * 0.25).normalize();
                    (axis, UP * 0.7 + sph * 0.5 + rng.on_sphere() * 0.8)
                }
            };
            nrm0 -= axis * nrm0.dot(axis);
            if nrm0.length_squared() < 1e-4 {
                nrm0 = perp_basis(axis).0;
            }
            let nrm0 = nrm0.normalize();
            let side = axis.cross(nrm0).normalize();
            let occ = (0.8 * (1.0 - smoothstep(0.25, 1.0, d)) + 0.25 * (-sph.y).clamp(0.0, 1.0))
                .clamp(0.0, 1.0);
            let tint = rng.float();
            let base = pt - axis * (size * 0.08);
            let n_cross = cfg.cross.max(1);
            for c in 0..n_cross {
                let (mut sd, mut nr) = (side, nrm0);
                if c > 0 {
                    let q = Quat::from_axis_angle(axis, (PI / n_cross as f32) * c as f32);
                    sd = q * side;
                    nr = q * nrm0;
                }
                let flip = rng.chance(0.5);
                if kept {
                    add_card(
                        part,
                        base,
                        axis,
                        sd,
                        size,
                        size,
                        cfg.leaf_cell,
                        sph,
                        nr,
                        w_path,
                        tint,
                        occ,
                        flip,
                    );
                }
            }
        };

    let mut density = cfg.density;
    if let Some(cards) = cfg.cards {
        let (mut total, mut n) = (0.0, 0);
        for b in sk
            .branches
            .iter()
            .filter(|b| leaf_levels.contains(&b.level))
        {
            let terminal = b.level == max_level || b.children == 0;
            let start = if terminal {
                cfg.leaf_start
            } else {
                cfg.leaf_start.max(0.55)
            };
            total += b.len() * (1.0 - start);
            n += 1;
        }
        density = ((cards as f32 - n as f32) / total.max(1.0)).max(0.5);
    }
    for b in sk
        .branches
        .iter()
        .filter(|b| leaf_levels.contains(&b.level))
    {
        let len = b.len();
        let terminal = b.level == max_level || b.children == 0;
        let start = if terminal {
            cfg.leaf_start
        } else {
            cfg.leaf_start.max(0.55)
        };
        let count = ((len * (1.0 - start) * density * rng.range(0.85, 1.15)).round() as u32).max(1);
        for i in 0..count {
            let t = start + (1.0 - start) * ((i as f32 + rng.range(0.1, 0.9)) / count as f32);
            let (p, tan, _) = b.sample(t);
            let w = (b.d0 + t * len) / sk.max_path;
            place_card(&mut part, rng, p, tan, w.max(0.35), 1.0);
        }
        // Tip card.
        let (p, tan, _) = b.sample(1.0);
        place_card(
            &mut part,
            rng,
            p,
            tan,
            ((b.d0 + len) / sk.max_path).max(0.4),
            1.1,
        );
    }

    if let Some(per_m) = cfg.trunk_leaves {
        let trunk = &sk.branches[0];
        let tl = trunk.len();
        let s0 = ((env.base - trunk.pts[0].y) / tl).max(0.02);
        let count = (tl * (1.0 - s0) * per_m).round() as u32;
        for i in 0..count {
            let t = s0 + (1.0 - s0) * ((i as f32 + rng.float()) / count as f32);
            let (mut p, tan, _) = trunk.sample(t);
            let mut out = rng.on_sphere();
            out.y = 0.0;
            let out = out.normalize_or(Vec3::X);
            let r = env.rx
                * (cfg.prof)(((p.y - env.base) / (env.top - env.base)).clamp(0.0, 1.0))
                * rng.range(0.1, 0.6);
            p += out * r;
            let dir = if cfg.card_mode == CardMode::Flat {
                out
            } else {
                tan
            };
            place_card(
                &mut part,
                rng,
                p,
                dir,
                (t * tl / sk.max_path).max(0.3),
                0.95,
            );
        }
    }
    part
}

// -------------------------------------------------------------------------
// Far hull
// -------------------------------------------------------------------------

const BARK_FAR: [[f32; 3]; 8] = [
    [0.33, 0.29, 0.25],
    [0.3, 0.26, 0.22],
    [0.82, 0.8, 0.76],
    [0.55, 0.53, 0.43],
    [0.45, 0.3, 0.22],
    [0.47, 0.42, 0.35],
    [0.4, 0.29, 0.22],
    [0.38, 0.35, 0.31],
];

#[allow(clippy::too_many_arguments)]
fn far_push(
    part: &mut Part,
    p: Vec3,
    n: Vec3,
    kind: f32,
    deciduous: f32,
    col: [f32; 3],
    aut: [f32; 3],
    w: f32,
) -> u32 {
    part.verts.push(VegVertex {
        p,
        n,
        uv: [0.0, 0.0],
        wind: w,
        tint: 0.0,
        occ: 0.0,
        color: lin3(col),
        autumn: lin3(aut),
        kind,
        deciduous,
    });
    (part.verts.len() - 1) as u32
}

/// A tapered prism of bark from `base` to `top`, shaded and swaying per end.
#[allow(clippy::too_many_arguments)]
fn far_trunk(
    part: &mut Part,
    base: Vec3,
    top: Vec3,
    r: f32,
    bark: u32,
    sides: u32,
    r_top: f32,
    shade: [f32; 2],
    w: [f32; 2],
) {
    let c = BARK_FAR[bark as usize % BARK_FAR.len()];
    let cols = [c.map(|v| v * shade[0]), c.map(|v| v * shade[1])];
    let axis = (top - base).normalize_or(UP);
    let a = if axis.y.abs() < 0.9 { UP } else { Vec3::X };
    let u = axis.cross(a).normalize();
    let v = u.cross(axis);
    for i in 0..sides {
        let a0 = (i as f32 / sides as f32) * TAU + 0.4;
        let a1 = ((i + 1) as f32 / sides as f32) * TAU + 0.4;
        let q = |ang: f32, c: Vec3, rr: f32| c + u * (ang.cos() * rr) + v * (ang.sin() * rr);
        let am = (a0 + a1) / 2.0;
        let n = u * am.cos() + v * am.sin();
        let quad = [
            q(a0, base, r),
            q(a1, base, r),
            q(a1, top, r_top),
            q(a0, top, r_top),
        ];
        let idx: Vec<u32> = (0..4)
            .map(|k| {
                let end = usize::from(k >= 2);
                far_push(part, quad[k], n, 0.0, 0.0, cols[end], cols[end], w[end])
            })
            .collect();
        part.tri(idx[0], idx[2], idx[1]);
        part.tri(idx[0], idx[3], idx[2]);
    }
}

/// Unit icosphere (one subdivision): vertices and triangles.
fn icosphere() -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut v: Vec<Vec3> = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let faces: [[u32; 3]; 20] = [
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
    let mut mid = std::collections::HashMap::new();
    let mut out = Vec::new();
    let mut m = |a: u32, b: u32, v: &mut Vec<Vec3>| -> u32 {
        *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
            v.push((v[a as usize] + v[b as usize]).normalize());
            (v.len() - 1) as u32
        })
    };
    for [a, b, c] in faces {
        let ab = m(a, b, &mut v);
        let bc = m(b, c, &mut v);
        let ca = m(c, a, &mut v);
        out.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
    }
    (v, out)
}

/// How many of the trunk's biggest limbs a far hull shows, and how thick
/// (of the trunk) a limb must be to count.
const FAR_LIMBS: usize = 4;
const FAR_LIMB_MIN: f32 = 0.3;

fn build_far(cfg: &SpeciesCfg, sk: &Skeleton) -> Part {
    let mut part = Part::default();
    let env = sk.env;
    let trunk = &sk.branches[0];
    let y0 = trunk.pts[0].y;
    let y_end = trunk.pts.last().expect("trunk").y;
    let at = |y: f32| trunk.sample(((y - y0) / (y_end - y0).max(1e-3)).clamp(0.0, 1.0));
    let (fork, _, r_fork) = at(env.base);
    let (mut mid, _, _) = at(env.cy());
    mid.y = mid.y.max(env.cy());
    let r0 = cfg.trunk_r * 1.1;
    // The trunk reaches into the middle of the crown, so its end never shows
    // below the hull's dissolved rim.
    far_trunk(
        &mut part,
        Vec3::new(0.0, -0.1, 0.0),
        fork,
        r0,
        cfg.bark,
        5,
        r_fork.max(r0 * 0.55),
        [0.8, 0.6],
        [0.0, 0.2],
    );
    far_trunk(
        &mut part,
        fork,
        mid,
        r_fork.max(r0 * 0.55),
        cfg.bark,
        4,
        r0 * 0.25,
        [0.6, 0.45],
        [0.2, 0.4],
    );
    // The main limbs, part way out into the crown, so a far tree forks.
    let mut limbs: Vec<&Branch> = sk
        .branches
        .iter()
        .filter(|b| b.level == 1 && b.rad[0] >= cfg.trunk_r * FAR_LIMB_MIN)
        .collect();
    limbs.sort_by(|a, b| b.rad[0].total_cmp(&a.rad[0]));
    for b in limbs.into_iter().take(FAR_LIMBS) {
        let (end, _, _) = b.sample(0.6);
        far_trunk(
            &mut part,
            b.pts[0],
            end,
            b.rad[0] * 1.1,
            cfg.bark,
            3,
            b.rad[0] * 0.4,
            [0.6, 0.45],
            [0.2, 0.5],
        );
    }
    let (ico_v, ico_t) = icosphere();
    let c = env.centre();
    let decid = if cfg.evergreen { 0.0 } else { 1.0 };
    let base = part.verts.len() as u32;
    for d in ico_v {
        let mut e = env.extent(d) * 0.97;
        e *= 1.0 + 0.12 * noise3(0x7ee5, Vec3::new(d.x * 1.7 + 3.1, d.y * 1.7, d.z * 1.7));
        let p = c + d * e.max(0.3);
        let (n, _) = env.shade(p);
        let n = n.lerp(d, 0.4).normalize();
        let shade = 0.75 + 0.25 * (d.y * 0.5 + 0.5).clamp(0.0, 1.0);
        let col = cfg.far_color.map(|v| v * shade);
        let aut = cfg.autumn.map(|v| v * shade * 0.85);
        far_push(&mut part, p, n, 1.0, decid, col, aut, 0.75);
    }
    for [a, b, c] in ico_t {
        part.tri(base + a, base + b, base + c);
    }
    part
}

// -------------------------------------------------------------------------
// Palm
// -------------------------------------------------------------------------

fn build_palm(variant: u32) -> (Grown, Part) {
    let mut rng = Rng::from(&["tree", "palm"], variant);
    let h = rng.range(8.5, 11.5);
    let mut wood = Part::default();
    let mut leaves = Part::default();
    let mut lean_dir = rng.on_sphere();
    lean_dir.y = 0.0;
    let lean_dir = lean_dir.normalize_or(Vec3::X);
    let lean = rng.range(0.25, 0.5);
    // Curved trunk: leans out at the base and turns up again.
    let mut pts = vec![Vec3::new(0.0, -0.15, 0.0)];
    let mut d = (UP + lean_dir * lean).normalize();
    let segs = 16;
    let seg_l = h / segs as f32;
    let mut p = pts[0];
    for i in 0..segs {
        p += d * seg_l;
        pts.push(p);
        d += UP * (0.05 + 0.02 * (i as f32 / segs as f32));
        d += rng.on_sphere() * 0.015;
        d = d.normalize();
    }
    let mut trunk = Branch::new(pts.clone(), 0.22, 0.2, 0, 0.0, 1.0);
    for (i, (r, p)) in trunk.rad.iter_mut().zip(&pts).enumerate() {
        let y = p.y;
        *r = 0.2 + 0.12 * (-y.max(0.0) / 0.6).exp() + if i >= segs - 1 { 0.03 } else { 0.0 };
    }
    let trunk_len = trunk.len();
    let max_path = trunk_len + 4.5;
    tube(&mut wood, &trunk, 9, 5, max_path, None, None, variant * 31);
    let top = pts[pts.len() - 1];
    let t_tan = (top - pts[pts.len() - 2]).normalize();
    let crown_c = top + Vec3::new(0.0, 0.4, 0.0);

    // Leaf-base "boots" knob under the crown.
    let boot = Branch::new(
        vec![top - t_tan * 0.7, top + t_tan * 0.25],
        0.27,
        0.16,
        1,
        trunk_len - 0.7,
        1.0,
    );
    tube(&mut wood, &boot, 9, 5, max_path, None, None, 5);

    let frond = |rng: &mut Rng,
                 wood: &mut Part,
                 leaves: &mut Part,
                 az: f32,
                 elev: f32,
                 len: f32,
                 dead: bool| {
        let dir0 = Vec3::new(az.cos() * elev.cos(), elev.sin(), az.sin() * elev.cos());
        let start = top + dir0 * 0.12;
        let pet_len = if dead { 0.4 } else { 0.55 };
        let pet = Branch::new(
            vec![start, start + dir0 * pet_len],
            0.045,
            0.03,
            2,
            trunk_len,
            1.0,
        );
        tube(wood, &pet, 4, 5, max_path, None, None, 3);
        // Leaflet strip with a V fold.
        let fs = 11;
        let mut fd = dir0;
        let mut c = pet.pts[1];
        let half_w = if dead { 0.42 } else { 0.62 };
        let cell = if dead { 14 } else { 12 };
        let tint = rng.float();
        let base = leaves.verts.len() as u32;
        let sl = (len - pet_len) / fs as f32;
        for i in 0..=fs {
            let t = i as f32 / fs as f32;
            if i > 0 {
                let droop = if dead {
                    -0.5 * sl
                } else {
                    -(0.12 + 0.25 * t) * sl * 1.6
                };
                fd = (fd + UP * droop).normalize();
                c += fd * sl;
            }
            let mut side = fd.cross(UP);
            if side.length_squared() < 1e-4 {
                side = Vec3::new(az.sin(), 0.0, -az.cos());
            }
            let side = side.normalize();
            let up_p = side.cross(fd).normalize();
            let w = half_w * if i == fs { 0.35 } else { 1.0 } * if dead { 0.8 } else { 1.0 };
            let fold = if dead { -0.5 } else { 0.38 };
            let wind = ((trunk_len + pet_len + t * len) / max_path).clamp(0.5, 1.0);
            let sph = (c - crown_c).normalize();
            let occ = if dead { 0.35 } else { 0.15 * (1.0 - t) };
            let n_l = (sph * 0.55 + up_p * 0.45).normalize();
            let [u0, v_l] = cell_uv(cell, t, 1.0, 2.0);
            let [_, v_c] = cell_uv(cell, t, 0.5, 2.0);
            let [_, v_r] = cell_uv(cell, t, 0.0, 2.0);
            leaves.vert(
                c - side * w + up_p * (w * fold),
                n_l,
                [u0, v_l],
                wind,
                tint,
                occ,
            );
            leaves.vert(c - up_p * 0.02, n_l, [u0, v_c], wind, tint, occ);
            leaves.vert(
                c + side * w + up_p * (w * fold),
                n_l,
                [u0, v_r],
                wind,
                tint,
                occ,
            );
        }
        for i in 0..fs as u32 {
            let a = base + i * 3;
            let b = a + 3;
            for [x, y, z] in [
                [a, b, a + 1],
                [a + 1, b, b + 1],
                [a + 1, b + 1, a + 2],
                [a + 2, b + 1, b + 2],
            ] {
                leaves.tri(x, y, z);
                leaves.tri(x, z, y);
            }
        }
    };
    let n_fronds = rng.int(18, 23);
    let mut az = rng.range(0.0, TAU);
    for i in 0..n_fronds {
        let f = i as f32 / (n_fronds - 1) as f32;
        let elev = (70.0 - f * 95.0 + rng.range(-8.0, 8.0)) * DEG;
        let len = rng.range(3.4, 4.4) * if f < 0.15 { 0.8 } else { 1.0 };
        frond(&mut rng, &mut wood, &mut leaves, az, elev, len, false);
        az += 137.5 * DEG + rng.range(-0.15, 0.15);
    }
    let n_dead = rng.int(3, 6);
    for _ in 0..n_dead {
        let (a, e, l) = (
            rng.range(0.0, TAU),
            rng.range(-82.0, -65.0) * DEG,
            rng.range(2.2, 2.9),
        );
        frond(&mut rng, &mut wood, &mut leaves, a, e, l, true);
    }

    // Far: trunk plus 9 frond ribbons.
    let mut far = Part::default();
    far_trunk(
        &mut far,
        Vec3::new(0.0, -0.1, 0.0),
        top,
        0.24,
        5,
        4,
        0.24 * 0.6,
        [1.0, 1.0],
        [0.0, 0.3],
    );
    let green = [0.3, 0.42, 0.18];
    for i in 0..9 {
        let a = (i as f32 / 9.0) * TAU + 0.3;
        let elev = match i % 3 {
            0 => 30.0,
            1 => 0.0,
            _ => -25.0,
        } * DEG;
        let dir = Vec3::new(a.cos() * elev.cos(), elev.sin(), a.sin() * elev.cos());
        let side = Vec3::new(-a.sin(), 0.0, a.cos());
        let mid = top + dir * 2.0 + Vec3::new(0.0, -0.2, 0.0);
        let tip = top + dir * 3.8 + Vec3::new(0.0, -1.2, 0.0);
        let q = [
            top + side * 0.15,
            mid + side * 0.55,
            tip,
            mid - side * 0.55,
            top - side * 0.15,
        ];
        let idx: Vec<u32> = q
            .iter()
            .map(|p| far_push(&mut far, *p, UP, 0.5, 0.0, green, green, 0.8))
            .collect();
        for [x, y, z] in [[0, 1, 4], [1, 3, 4], [1, 2, 3]] {
            far.tri(idx[x], idx[y], idx[z]);
            far.tri(idx[x], idx[z], idx[y]);
        }
    }
    let height = top.y + 2.2;
    // The palm's mid level keeps the fronds (they are already cheap) and the
    // trunk without the frond stalks.
    let mid_wood = {
        let mut w = Part::default();
        tube(&mut w, &trunk, 6, 5, max_path, None, None, variant * 31);
        w
    };
    (
        Grown {
            wood,
            leaves,
            far,
            height,
            trunk_radius: 0.22,
            crown_radius: 4.2,
        },
        mid_wood,
    )
}

// -------------------------------------------------------------------------
// Public entry points
// -------------------------------------------------------------------------

/// Full-detail and mid-detail parts of one tree or shrub.
pub(crate) struct GrownLods {
    pub near: Grown,
    /// Coarser wood (trunk and main limbs) and every other leaf tuft, larger.
    pub mid_wood: Part,
    pub mid_leaves: Part,
}

pub(crate) fn grow(kind: Kind, variant: u32) -> GrownLods {
    if kind == Kind::Tree(super::TreeSpecies::Palm) {
        let (near, mid_wood) = build_palm(variant);
        let mid_leaves = near.leaves.clone();
        return GrownLods {
            near,
            mid_wood,
            mid_leaves,
        };
    }
    let cfg = species_cfg(kind, variant);
    let rng = Rng::from(&["tree", kind.name()], variant);
    let sk = build_skeleton(&cfg, &mut rng.fork("skeleton"));
    let seed = variant * 131 + 7;
    let wood = build_wood(&cfg, &sk, seed, usize::MAX);
    let leaves = build_leaves(&cfg, &sk, &mut rng.fork("leaves"), 1.0);
    let far = build_far(&cfg, &sk);
    let mid_wood = build_wood(&cfg, &sk, seed, cfg.levels.len().saturating_sub(1).max(1));
    let mid_leaves = build_leaves(&cfg, &sk, &mut rng.fork("leaves"), 0.5);
    let height = sk.height.max(leaves.max_y());
    GrownLods {
        near: Grown {
            wood,
            leaves,
            far,
            height,
            trunk_radius: cfg.trunk_r,
            crown_radius: sk.env.rx.max(sk.env.rz),
        },
        mid_wood,
        mid_leaves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_hull_faces_outwards() {
        let (v, t) = icosphere();
        for [a, b, c] in t {
            let (a, b, c) = (v[a as usize], v[b as usize], v[c as usize]);
            let n = (b - a).cross(c - a);
            assert!(n.dot(a + b + c) > 0.0, "icosphere triangle faces inwards");
        }
    }
}
