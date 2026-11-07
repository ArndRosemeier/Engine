//! Finds z-fighting: triangles that lie in (almost) the same plane, face the
//! same way and overlap, so the depth buffer cannot decide between them and
//! the surface flickers as the camera moves.
//!
//! Three ways to use it, from least to most effort:
//!
//! - **Any app, no code:** run it with `ENGINE_ZFIGHT=1`. The engine scans
//!   what is around the camera, prints the worst spots to the console and
//!   marks each one with a glowing magenta post. It scans again when the
//!   scene changes or the camera moves on. `ENGINE_ZFIGHT=80` scans 80 m
//!   around the camera, `ENGINE_ZFIGHT=all` the whole world.
//! - **In code:** [`World::find_z_fighting`](crate::world::World::find_z_fighting)
//!   returns a [`ZFightReport`];
//!   [`World::show_z_fighting`](crate::world::World::show_z_fighting) marks it
//!   in the scene, [`World::check_z_fighting`](crate::world::World::check_z_fighting)
//!   does both and prints the report.
//! - **In a test, without a window:** put your generated meshes into a
//!   [`ZFightCheck`] and call [`ZFightReport::assert_clean`]:
//!
//! ```
//! use engine::prelude::*;
//! use engine::zfight::ZFightCheck;
//!
//! let ground = Shape::box_at((0.0, -0.05, 0.0), (10.0, 0.1, 10.0), rgb(90, 120, 70)).unwrap();
//! // A rug lying exactly on the ground: its bottom is hidden, its top fights.
//! let rug = Shape::box_at((0.0, 0.0, 0.0), (2.0, 0.0001, 1.0), rgb(150, 30, 30)).unwrap();
//! let report = ZFightCheck::new()
//!     .with("ground", &ground)
//!     .with("rug", &rug)
//!     .run();
//! assert!(!report.is_clean());
//! println!("{report}");
//! ```
//!
//! Clashes between faces that look different (another colour, texture or
//! material) are visible flicker. Clashes between faces that look the same
//! are only doubled geometry: invisible, but wasted, so they are reported
//! apart ([`Clash::same_look`]) and do not make a report unclean.
//!
//! Fix overlaps in the geometry (lift a face a centimetre off the surface,
//! or cut the hidden one away), not by loosening the check.

use crate::mesh::{BuiltMesh, Mesh};
use crate::place::Place;
use glam::{DVec2, DVec3, Mat4, Vec4};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::time::{Duration, Instant};

/// One triangle with how it looks and where it came from.
#[derive(Clone, Copy, Debug)]
pub struct Tri {
    pub p: [DVec3; 3],
    /// sRGB 0..255 (the mesh's linear colour converted back, so it reads
    /// like the hex code it was authored with).
    pub color: [u8; 3],
    /// Anything else that changes the look (texture, material). Faces with
    /// the same colour but a different `surface` still flicker. 0 = plain.
    pub surface: u64,
    /// Caller's label (an index into [`ZFightReport::sources`] when the
    /// triangles came through [`ZFightCheck`]).
    pub source: u32,
}

/// Two overlapping coplanar triangles.
#[derive(Clone, Copy, Debug)]
pub struct Clash {
    /// Middle of the overlap (world position).
    pub at: DVec3,
    pub normal: DVec3,
    /// Overlapping area (m²).
    pub area: f64,
    /// Distance between the two planes (m).
    pub gap: f64,
    pub colors: [[u8; 3]; 2],
    pub sources: [u32; 2],
    /// Indices of the two triangles in the input.
    pub tris: [usize; 2],
    /// Both faces look the same: doubled geometry, not visible flicker.
    pub same_look: bool,
}

/// How close is "the same plane", and how big an overlap counts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Faces closer than this (m) fight; faces exactly this far apart (a
    /// deliberate lift) do not. The default, 4 mm, catches what flickers
    /// with Engine's reversed-Z float depth; raise it for a stricter rule
    /// (HeroWorld keeps its city clean at 1 cm).
    pub gap: f64,
    /// Overlaps smaller than this (m²) are ignored (edges that only touch).
    pub min_area: f64,
    /// Downward faces lying entirely at or below this height are skipped:
    /// the bottoms of things standing on the ground, which nobody sees. Off
    /// by default; see [`Options::ground_at`].
    pub hidden_below: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            gap: 0.004,
            min_area: 1e-3,
            hidden_below: f64::NEG_INFINITY,
        }
    }
}

impl Options {
    /// For scenes standing on a ground at about `height`: downward faces up
    /// to half a metre above it are hidden bottoms and never fight.
    pub fn ground_at(height: f64) -> Self {
        Self {
            hidden_below: height + 0.5,
            ..Self::default()
        }
    }
}

fn to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// The triangles of a built mesh (opaque and transparent) moved by
/// `transform`, labelled `source`.
pub fn triangles(mesh: &BuiltMesh, transform: Mat4, source: u32) -> Vec<Tri> {
    triangles_tinted(mesh, transform, Vec4::ONE, 0, source)
}

/// [`triangles`] with an instance tint and a look key.
pub(crate) fn triangles_tinted(
    mesh: &BuiltMesh,
    transform: Mat4,
    tint: Vec4,
    surface: u64,
    source: u32,
) -> Vec<Tri> {
    let m = transform.as_dmat4();
    mesh.indices
        .chunks_exact(3)
        .filter(|ix| ix.iter().all(|&i| (i as usize) < mesh.positions.len()))
        .map(|ix| {
            let v = |k: usize| m.transform_point3(mesh.positions[ix[k] as usize].as_dvec3());
            let c = mesh
                .colors
                .get(ix[0] as usize)
                .copied()
                .unwrap_or(Vec4::ONE)
                * tint;
            Tri {
                p: [v(0), v(1), v(2)],
                color: [to_srgb(c.x), to_srgb(c.y), to_srgb(c.z)],
                surface,
                source,
            }
        })
        .collect()
}

struct Prepared {
    n: DVec3,
    d: f64,
    index: usize,
}

/// All clashes among `tris`, biggest overlap first.
pub fn find(tris: &[Tri], opt: &Options) -> Vec<Clash> {
    // Group by facing direction (normals rounded to about 0.1°), then by
    // plane distance, then by position inside the plane. Sorting one flat
    // list keeps this fast with millions of differently facing triangles
    // (foliage), where a map of groups would not be.
    let mut prepared: Vec<([i32; 3], Prepared)> = tris
        .par_iter()
        .enumerate()
        .filter_map(|(index, t)| {
            let c = (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]);
            let len = c.length();
            if len.is_nan() || len <= 1e-9 {
                return None;
            }
            let n = c / len;
            if n.y < -0.5 && t.p.iter().all(|p| p.y <= opt.hidden_below) {
                return None;
            }
            let key = (n * 512.0).round().as_ivec3().to_array();
            Some((
                key,
                Prepared {
                    n,
                    d: n.dot(t.p[0]),
                    index,
                },
            ))
        })
        .collect();
    prepared.par_sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.d.total_cmp(&b.1.d))
            .then(a.1.index.cmp(&b.1.index))
    });
    // Runs of planes facing the same way and chained within `gap` of each
    // other. Plane distances from the origin carry rounding error far from
    // it, so runs are cut loosely; pairs are measured exactly later.
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0;
    for end in 1..=prepared.len() {
        let split = end == prepared.len()
            || prepared[end].0 != prepared[end - 1].0
            || prepared[end].1.d - prepared[end - 1].1.d > opt.gap + 0.05;
        if split {
            if end - start > 1 {
                runs.push(start..end);
            }
            start = end;
        }
    }
    let prepared: Vec<Prepared> = prepared.into_iter().map(|(_, p)| p).collect();
    let mut out: Vec<Clash> = runs
        .into_par_iter()
        .flat_map_iter(|run| {
            let mut out = Vec::new();
            clashes_in_run(tris, &prepared[run], opt, &mut out);
            out
        })
        .collect();
    out.sort_by(|a, b| b.area.total_cmp(&a.area).then(a.tris.cmp(&b.tris)));
    out
}

/// Pairs within one run of nearly coplanar triangles, found through a grid
/// in the plane.
fn clashes_in_run(tris: &[Tri], run: &[Prepared], opt: &Options, out: &mut Vec<Clash>) {
    let n = run[0].n;
    let u = n.any_orthonormal_vector();
    let v = n.cross(u);
    // Work relative to a point of the run: precise far from the origin.
    let o = tris[run[0].index].p[0];
    let flat: Vec<[DVec2; 3]> = run
        .iter()
        .map(|r| {
            let mut q = tris[r.index]
                .p
                .map(|p| DVec2::new((p - o).dot(u), (p - o).dot(v)));
            if cross2(q[1] - q[0], q[2] - q[0]) < 0.0 {
                q.swap(1, 2);
            }
            q
        })
        .collect();
    let bounds: Vec<(DVec2, DVec2)> = flat
        .iter()
        .map(|q| (q[0].min(q[1]).min(q[2]), q[0].max(q[1]).max(q[2])))
        .collect();
    const CELL: f64 = 8.0;
    let cell = |x: f64| (x / CELL).floor() as i64;
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (k, (lo, hi)) in bounds.iter().enumerate() {
        for cx in cell(lo.x)..=cell(hi.x) {
            for cy in cell(lo.y)..=cell(hi.y) {
                grid.entry((cx, cy)).or_default().push(k);
            }
        }
    }
    for ((cx, cy), members) in &grid {
        for (a_pos, &a) in members.iter().enumerate() {
            for &b in &members[a_pos + 1..] {
                let (la, ha) = bounds[a];
                let (lb, hb) = bounds[b];
                let lo = la.max(lb);
                let hi = ha.min(hb);
                if lo.x >= hi.x || lo.y >= hi.y {
                    continue;
                }
                // Test each pair once: in the cell holding the overlap's corner.
                if (cell(lo.x), cell(lo.y)) != (*cx, *cy) {
                    continue;
                }
                let (ta, tb) = (&tris[run[a].index], &tris[run[b].index]);
                let gap = plane_gap(ta, run[a].n, tb).min(plane_gap(tb, run[b].n, ta));
                // Tolerance: a face lifted by exactly `gap` is a fix.
                if gap >= opt.gap - 1e-6 {
                    continue;
                }
                let poly = clip(&flat[a], &flat[b]);
                let (area, centre) = area_and_centre(&poly);
                if area < opt.min_area {
                    continue;
                }
                let d = n.dot(ta.p[0] - o);
                out.push(Clash {
                    at: o + u * centre.x + v * centre.y + n * d,
                    normal: n,
                    area,
                    gap,
                    colors: [ta.color, tb.color],
                    sources: [ta.source, tb.source],
                    tris: [run[a].index, run[b].index],
                    same_look: ta.color == tb.color && ta.surface == tb.surface,
                });
            }
        }
    }
}

/// How far triangle `b` is from the plane of triangle `a` (normal `n`),
/// measured near the triangles, where it is precise.
fn plane_gap(a: &Tri, n: DVec3, b: &Tri) -> f64 {
    b.p.iter()
        .map(|p| n.dot(*p - a.p[0]).abs())
        .fold(0.0, f64::max)
}

fn cross2(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// The part of triangle `a` inside triangle `b` (both counter-clockwise).
fn clip(a: &[DVec2; 3], b: &[DVec2; 3]) -> Vec<DVec2> {
    let mut poly: Vec<DVec2> = a.to_vec();
    for k in 0..3 {
        let (e0, e1) = (b[k], b[(k + 1) % 3]);
        let inside = |p: DVec2| cross2(e1 - e0, p - e0) >= -1e-12;
        let mut next = Vec::with_capacity(poly.len() + 1);
        for i in 0..poly.len() {
            let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
            let (ip, iq) = (inside(p), inside(q));
            if ip {
                next.push(p);
            }
            if ip != iq {
                let dp = cross2(e1 - e0, p - e0);
                let dq = cross2(e1 - e0, q - e0);
                next.push(p + (q - p) * (dp / (dp - dq)));
            }
        }
        poly = next;
        if poly.len() < 3 {
            return Vec::new();
        }
    }
    poly
}

fn area_and_centre(poly: &[DVec2]) -> (f64, DVec2) {
    if poly.len() < 3 {
        return (0.0, DVec2::ZERO);
    }
    let mut area = 0.0;
    let mut c = DVec2::ZERO;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let w = cross2(p, q);
        area += w;
        c += (p + q) * w;
    }
    if area.abs() < 1e-15 {
        return (0.0, poly[0]);
    }
    (area * 0.5, c / (3.0 * area))
}

/// Clashes grouped into spots: one per pair of sources and colours, facing
/// and patch (clashes within `size` metres of each other join), with the summed area,
/// biggest first. `at` is the area-weighted middle of the patch.
pub fn spots(clashes: &[Clash], size: f64) -> Vec<Clash> {
    type Key = ([u32; 2], [[u8; 3]; 2], [i32; 3]);
    struct Patch {
        spot: Clash,
        lo: DVec3,
        hi: DVec3,
        weighted: DVec3,
        /// Size of the biggest overlap in the patch (m).
        reach: f64,
    }
    let mut by: BTreeMap<Key, Vec<Patch>> = BTreeMap::new();
    for c in clashes {
        let mut colors = c.colors;
        let mut sources = c.sources;
        if colors[0] > colors[1] || (colors[0] == colors[1] && sources[0] > sources[1]) {
            colors.swap(0, 1);
            sources.swap(0, 1);
        }
        let key = (
            sources,
            colors,
            (c.normal * 64.0).round().as_ivec3().to_array(),
        );
        let patches = by.entry(key).or_default();
        let near = patches.iter_mut().find(|p| {
            let d = (p.lo - c.at).max(c.at - p.hi).max(DVec3::ZERO);
            // Big overlaps have their middles far apart: allow for their size.
            d.length() <= size + c.area.sqrt() + p.reach
        });
        match near {
            Some(p) => {
                p.lo = p.lo.min(c.at);
                p.hi = p.hi.max(c.at);
                p.weighted += c.at * c.area;
                p.reach = p.reach.max(c.area.sqrt());
                p.spot.area += c.area;
                p.spot.gap = p.spot.gap.max(c.gap);
            }
            None => patches.push(Patch {
                spot: Clash {
                    colors,
                    sources,
                    ..*c
                },
                lo: c.at,
                hi: c.at,
                weighted: c.at * c.area,
                reach: c.area.sqrt(),
            }),
        }
    }
    let mut out: Vec<Clash> = by
        .into_values()
        .flatten()
        .map(|p| Clash {
            at: if p.spot.area > 0.0 {
                p.weighted / p.spot.area
            } else {
                p.spot.at
            },
            ..p.spot
        })
        .collect();
    out.sort_by(|a, b| b.area.total_cmp(&a.area).then(a.tris.cmp(&b.tris)));
    out
}

/// `#rrggbb`.
pub fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Collects named meshes and checks them together. No window or GPU needed,
/// so it runs in tests and command-line tools.
///
/// ```
/// use engine::prelude::*;
/// use engine::zfight::{Options, ZFightCheck};
///
/// let crate_mesh = Shape::box_at((0.0, 0.5, 0.0), (1.0, 1.0, 1.0), rgb(140, 100, 60)).unwrap();
/// let mut check = ZFightCheck::new().options(Options::ground_at(0.0));
/// for (i, x) in [0.0, 3.0, 6.0].into_iter().enumerate() {
///     check.add(format!("crate {i}"), &crate_mesh, Place::new(x, 0.0, 0.0));
/// }
/// check.run().assert_clean();
/// ```
#[derive(Clone, Debug, Default)]
pub struct ZFightCheck {
    tris: Vec<Tri>,
    sources: Vec<String>,
    options: Options,
}

impl ZFightCheck {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn options(mut self, options: Options) -> Self {
        self.options = options;
        self
    }

    /// Adds `mesh` where it was built (chainable form of [`Self::add`]).
    pub fn with(mut self, name: impl Into<String>, mesh: &Mesh) -> Self {
        self.add(name, mesh, Place::default());
        self
    }

    /// Adds `mesh` at `place` and returns its source label.
    pub fn add(&mut self, name: impl Into<String>, mesh: &Mesh, place: Place) -> u32 {
        self.add_built(name, &mesh.build(), place.to_matrix())
    }

    /// Adds an already built mesh moved by `transform`.
    pub fn add_built(&mut self, name: impl Into<String>, mesh: &BuiltMesh, transform: Mat4) -> u32 {
        let source = self.source(name);
        self.tris.extend(triangles(mesh, transform, source));
        source
    }

    /// Adds triangles of your own; their `source` is replaced by the label
    /// of `name`.
    pub fn add_triangles(
        &mut self,
        name: impl Into<String>,
        tris: impl IntoIterator<Item = Tri>,
    ) -> u32 {
        let source = self.source(name);
        self.tris
            .extend(tris.into_iter().map(|t| Tri { source, ..t }));
        source
    }

    /// Triangles collected so far.
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    fn source(&mut self, name: impl Into<String>) -> u32 {
        self.sources.push(name.into());
        (self.sources.len() - 1) as u32
    }

    pub fn run(&self) -> ZFightReport {
        let t = Instant::now();
        let clashes = find(&self.tris, &self.options);
        ZFightReport {
            triangles: self.tris.len(),
            clashes,
            sources: self.sources.clone(),
            notes: Vec::new(),
            elapsed: t.elapsed(),
        }
    }
}

/// What a check found.
#[derive(Clone, Debug, Default)]
pub struct ZFightReport {
    /// Triangles checked.
    pub triangles: usize,
    /// Every clash, biggest first, including same-look doubles.
    pub clashes: Vec<Clash>,
    /// Names of the sources, indexed by [`Clash::sources`].
    pub sources: Vec<String>,
    /// What the scan left out, and why.
    pub notes: Vec<String>,
    pub elapsed: Duration,
}

impl ZFightReport {
    /// Clashes that flicker (the faces look different).
    pub fn fighting(&self) -> impl Iterator<Item = &Clash> {
        self.clashes.iter().filter(|c| !c.same_look)
    }

    /// Doubled faces that look the same: invisible, but wasted triangles.
    pub fn doubled(&self) -> impl Iterator<Item = &Clash> {
        self.clashes.iter().filter(|c| c.same_look)
    }

    /// Nothing flickers.
    pub fn is_clean(&self) -> bool {
        self.fighting().next().is_none()
    }

    /// Flickering clashes grouped into spots of `size` metres, worst first.
    pub fn spots(&self, size: f64) -> Vec<Clash> {
        let fighting: Vec<Clash> = self.fighting().copied().collect();
        spots(&fighting, size)
    }

    pub fn source_name(&self, source: u32) -> &str {
        self.sources
            .get(source as usize)
            .map(String::as_str)
            .unwrap_or("?")
    }

    /// The headline and up to `max_spots` spots, one per line.
    pub fn summary(&self, max_spots: usize) -> String {
        let (n_fight, a_fight) = self
            .fighting()
            .fold((0, 0.0), |(n, a), c| (n + 1, a + c.area));
        let (n_double, a_double) = self
            .doubled()
            .fold((0, 0.0), |(n, a), c| (n + 1, a + c.area));
        let mut s = format!(
            "z-fighting: {} triangles checked in {:.0} ms, {} fighting pairs ({:.2} m²), {} doubled faces of one look ({:.2} m²)",
            self.triangles,
            self.elapsed.as_secs_f64() * 1000.0,
            n_fight,
            a_fight,
            n_double,
            a_double,
        );
        for note in &self.notes {
            s.push_str(&format!("\n  note: {note}"));
        }
        let spots = self.spots(2.0);
        if !spots.is_empty() {
            s.push_str(&format!("\n{} spots, worst first:", spots.len()));
        }
        for c in spots.iter().take(max_spots) {
            s.push_str(&format!(
                "\n  {:>8.2} m²  at ({:.2}, {:.2}, {:.2})  facing ({:.2}, {:.2}, {:.2})  {} vs {}  [{} / {}]  gap {:.3} m",
                c.area,
                c.at.x,
                c.at.y,
                c.at.z,
                c.normal.x,
                c.normal.y,
                c.normal.z,
                hex(c.colors[0]),
                hex(c.colors[1]),
                self.source_name(c.sources[0]),
                self.source_name(c.sources[1]),
                c.gap
            ));
        }
        if spots.len() > max_spots {
            s.push_str(&format!("\n  ... and {} more", spots.len() - max_spots));
        }
        s
    }

    /// Panics with the worst spots if anything flickers. For tests.
    #[track_caller]
    pub fn assert_clean(&self) {
        if !self.is_clean() {
            panic!("{}", self.summary(12));
        }
    }

    /// Adds another report's findings (sources are renumbered).
    pub(crate) fn merge(&mut self, other: ZFightReport) {
        let base = self.sources.len() as u32;
        let tri_base = self.triangles;
        self.triangles += other.triangles;
        self.sources.extend(other.sources);
        self.notes.extend(other.notes);
        self.elapsed += other.elapsed;
        self.clashes
            .extend(other.clashes.into_iter().map(|c| Clash {
                sources: c.sources.map(|s| s + base),
                tris: c.tris.map(|t| t + tri_base),
                ..c
            }));
        self.clashes
            .sort_by(|a, b| b.area.total_cmp(&a.area).then(a.tris.cmp(&b.tris)));
    }
}

impl fmt::Display for ZFightReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary(25))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(y: f64, x0: f64, x1: f64, color: [u8; 3], source: u32) -> [Tri; 2] {
        let p = |x: f64, z: f64| DVec3::new(x, y, z);
        [
            Tri {
                p: [p(x0, 0.0), p(x0, 1.0), p(x1, 1.0)],
                color,
                surface: 0,
                source,
            },
            Tri {
                p: [p(x0, 0.0), p(x1, 1.0), p(x1, 0.0)],
                color,
                surface: 0,
                source,
            },
        ]
    }

    #[test]
    fn overlapping_coplanar_faces_clash() {
        let mut t = quad(0.0, 0.0, 2.0, [255, 0, 0], 0).to_vec();
        t.extend(quad(0.002, 1.0, 3.0, [0, 0, 255], 1));
        let c = find(&t, &Options::default());
        let area: f64 = c.iter().map(|c| c.area).sum();
        assert!((area - 1.0).abs() < 1e-9, "overlap {area}");
        assert!(c.iter().all(|c| !c.same_look));
        assert!(c.iter().all(|c| c.sources == [0, 1] || c.sources == [1, 0]));
        assert_eq!(spots(&c, 10.0).len(), 1);
    }

    #[test]
    fn touching_separated_or_opposite_faces_do_not() {
        // Side by side.
        let mut t = quad(0.0, 0.0, 1.0, [255, 0, 0], 0).to_vec();
        t.extend(quad(0.0, 1.0, 2.0, [0, 0, 255], 1));
        // 5 cm above.
        t.extend(quad(0.05, 0.0, 2.0, [0, 255, 0], 2));
        // Same plane, facing down (the bottom of something resting here).
        let mut down = quad(0.0, 0.0, 2.0, [9, 9, 9], 3);
        for tri in &mut down {
            tri.p.swap(1, 2);
        }
        t.extend(down);
        assert!(find(&t, &Options::default()).is_empty());
    }

    #[test]
    fn same_colour_but_other_texture_still_fights() {
        let mut t = quad(0.0, 0.0, 2.0, [200, 200, 200], 0).to_vec();
        let mut other = quad(0.0, 0.0, 2.0, [200, 200, 200], 1);
        for tri in &mut other {
            tri.surface = 7;
        }
        t.extend(other);
        let c = find(&t, &Options::default());
        assert!(!c.is_empty() && c.iter().all(|c| !c.same_look));
        // Without the texture difference it is only doubled geometry.
        for tri in &mut t {
            tri.surface = 0;
        }
        assert!(find(&t, &Options::default()).iter().all(|c| c.same_look));
    }

    #[test]
    fn precise_far_from_the_origin() {
        let mut t = quad(0.0, 0.0, 2.0, [255, 0, 0], 0).to_vec();
        t.extend(quad(0.003, 1.0, 3.0, [0, 0, 255], 1));
        for tri in &mut t {
            for p in &mut tri.p {
                *p += DVec3::new(250_000.0, 40.0, -180_000.0);
            }
        }
        let area: f64 = find(&t, &Options::default()).iter().map(|c| c.area).sum();
        assert!((area - 1.0).abs() < 1e-6, "overlap {area}");
    }

    #[test]
    fn check_names_the_meshes() {
        use crate::color::rgb;
        use crate::mesh::Shape;
        let ground = Shape::box_at((0.0, -0.05, 0.0), (10.0, 0.1, 10.0), rgb(90, 120, 70)).unwrap();
        let poster = Shape::box_at((1.0, 0.0, 1.0), (2.0, 0.0001, 1.0), rgb(200, 30, 30)).unwrap();
        let report = ZFightCheck::new()
            .options(Options::ground_at(0.0))
            .with("ground", &ground)
            .with("poster", &poster)
            .run();
        assert!(!report.is_clean());
        let worst = report.spots(10.0)[0];
        let names = [
            report.source_name(worst.sources[0]),
            report.source_name(worst.sources[1]),
        ];
        assert!(
            names.contains(&"ground") && names.contains(&"poster"),
            "{names:?}"
        );
        assert!((worst.area - 2.0).abs() < 1e-3, "{}", worst.area);
        assert!(report.summary(5).contains("poster"));

        // Lifted 2 cm, nothing fights.
        let lifted = ZFightCheck::new()
            .options(Options::ground_at(0.0))
            .with("ground", &ground)
            .with(
                "poster",
                &Shape::box_at((1.0, 0.02, 1.0), (2.0, 0.0001, 1.0), rgb(200, 30, 30)).unwrap(),
            )
            .run();
        lifted.assert_clean();
    }
}
