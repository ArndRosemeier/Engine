//! Deterministic scattering: where trees, rocks and props stand.
//!
//! [`Scatter`] lays a jittered grid over an area, thins it with a patchy
//! noise mask so things grow in clumps and clearings, gives each point a
//! random yaw and scale, and can drop it onto the ground. The same seed
//! always gives the same places.
//!
//! ```no_run
//! # use engine::prelude::*;
//! # fn f(world: &mut World) -> EngineResult<()> {
//! let places = Scatter::new(7)
//!     .rect(-200.0, -200.0, 200.0, 200.0)
//!     .spacing(10.0)
//!     .density(0.6)
//!     .scale_range(0.8, 1.2)
//!     .places()?;
//! // Share the places between a few tree variants.
//! for (i, mine) in Scatter::split(&places, 3, 7).iter().enumerate() {
//!     let tree = tree_model(TreeSpecies::Oak, i as u32)?;
//!     world.spawn_lod_at(tree.lod_levels(), mine)?;
//! }
//! # Ok(()) }
//! ```

use crate::error::{EngineError, EngineResult};
use crate::place::Place;
use crate::vegetation::rng::{hash2i, hash_to_float, hash_u32, noise2};

/// Most points one scatter may produce, so a typo in `spacing` fails loudly
/// instead of filling memory.
pub const MAX_SCATTER_POINTS: usize = 2_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Area {
    Rect { min: [f32; 2], max: [f32; 2] },
    Disc { centre: [f32; 2], radius: f32 },
}

/// Builder for a deterministic set of [`Place`]s. See the [module docs](self).
pub struct Scatter<'a> {
    seed: u32,
    area: Area,
    spacing: f32,
    jitter: f32,
    density: f32,
    patch_m: f32,
    scale: (f32, f32),
    yaw: Option<f32>,
    ground: Option<Box<dyn Fn(f32, f32) -> f32 + 'a>>,
    keep: Option<Box<dyn Fn(f32, f32) -> bool + 'a>>,
}

impl<'a> Scatter<'a> {
    /// A 100 m square around the origin, one point every 5 m, full density,
    /// random yaw, scale 1.
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            area: Area::Rect {
                min: [-50.0, -50.0],
                max: [50.0, 50.0],
            },
            spacing: 5.0,
            jitter: 0.9,
            density: 1.0,
            patch_m: 40.0,
            scale: (1.0, 1.0),
            yaw: None,
            ground: None,
            keep: None,
        }
    }

    /// Scatter over the rectangle from (`min_x`, `min_z`) to (`max_x`, `max_z`).
    pub fn rect(mut self, min_x: f32, min_z: f32, max_x: f32, max_z: f32) -> Self {
        self.area = Area::Rect {
            min: [min_x.min(max_x), min_z.min(max_z)],
            max: [min_x.max(max_x), min_z.max(max_z)],
        };
        self
    }

    /// Scatter over a disc.
    pub fn disc(mut self, centre_x: f32, centre_z: f32, radius: f32) -> Self {
        self.area = Area::Disc {
            centre: [centre_x, centre_z],
            radius,
        };
        self
    }

    /// Average distance between neighbours in metres.
    pub fn spacing(mut self, metres: f32) -> Self {
        self.spacing = metres;
        self
    }

    /// How far a point may stray from its grid cell centre: 0 is a perfect
    /// grid, 1 anywhere in the cell (default 0.9).
    pub fn jitter(mut self, amount: f32) -> Self {
        self.jitter = amount;
        self
    }

    /// Share of points kept, 0..=1 (default 1). Below 1 the gaps form
    /// clearings and clumps rather than random holes; see [`Self::patch_size`].
    pub fn density(mut self, share: f32) -> Self {
        self.density = share;
        self
    }

    /// Size in metres of the clumps and clearings that [`Self::density`]
    /// makes (default 40).
    pub fn patch_size(mut self, metres: f32) -> Self {
        self.patch_m = metres;
        self
    }

    /// Random uniform scale between `min` and `max` (default 1..1).
    pub fn scale_range(mut self, min: f32, max: f32) -> Self {
        self.scale = (min.min(max), min.max(max));
        self
    }

    /// Give every point this yaw instead of a random one.
    pub fn yaw_deg(mut self, degrees: f32) -> Self {
        self.yaw = Some(degrees);
        self
    }

    /// Stand each point on the ground: `height(x, z)` gives its y.
    /// For a procedural terrain pass `|x, z| world.proc_height_at(x, z)`.
    pub fn on_ground(mut self, height: impl Fn(f32, f32) -> f32 + 'a) -> Self {
        self.ground = Some(Box::new(height));
        self
    }

    /// Keep only points where `keep(x, z)` is true: roads, water, slopes.
    pub fn only_where(mut self, keep: impl Fn(f32, f32) -> bool + 'a) -> Self {
        self.keep = Some(Box::new(keep));
        self
    }

    fn validate(&self) -> EngineResult<([f32; 2], [f32; 2])> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("scatter {what}")));
        if !(self.spacing.is_finite() && self.spacing > 0.0) {
            return bad("spacing must be finite and > 0");
        }
        if !(0.0..=1.0).contains(&self.jitter) {
            return bad("jitter must be within 0..=1");
        }
        if !(0.0..=1.0).contains(&self.density) {
            return bad("density must be within 0..=1");
        }
        if !(self.patch_m.is_finite() && self.patch_m > 0.0) {
            return bad("patch size must be finite and > 0");
        }
        if !(self.scale.0.is_finite() && self.scale.1.is_finite() && self.scale.0 > 0.0) {
            return bad("scale range must be finite and > 0");
        }
        if self.yaw.is_some_and(|y| !y.is_finite()) {
            return bad("yaw must be finite");
        }
        let (min, max) = match self.area {
            Area::Rect { min, max } => (min, max),
            Area::Disc { centre, radius } => {
                if !(radius.is_finite() && radius >= 0.0) {
                    return bad("disc radius must be finite and >= 0");
                }
                (
                    [centre[0] - radius, centre[1] - radius],
                    [centre[0] + radius, centre[1] + radius],
                )
            }
        };
        if !min.iter().chain(&max).all(|v| v.is_finite()) {
            return bad("area must be finite");
        }
        let cells = ((max[0] - min[0]) / self.spacing).ceil().max(1.0) as f64
            * ((max[1] - min[1]) / self.spacing).ceil().max(1.0) as f64;
        if cells > MAX_SCATTER_POINTS as f64 {
            return Err(EngineError::ResourceLimit(format!(
                "scatter would make {cells:.0} points (most {MAX_SCATTER_POINTS}); raise spacing"
            )));
        }
        Ok((min, max))
    }

    /// Build the places.
    pub fn places(&self) -> EngineResult<Vec<Place>> {
        let (min, max) = self.validate()?;
        let s = self.spacing;
        let nx = ((max[0] - min[0]) / s).ceil().max(1.0) as i32;
        let nz = ((max[1] - min[1]) / s).ceil().max(1.0) as i32;
        let seed = hash_u32(self.seed ^ 0x5ca7_7e12);
        let r = |i: i32, j: i32, k: u32| hash_to_float(hash2i(seed.wrapping_add(k), i, j));
        let mut out = Vec::new();
        for j in 0..nz {
            for i in 0..nx {
                let x = min[0] + (i as f32 + 0.5 + (r(i, j, 1) - 0.5) * self.jitter) * s;
                let z = min[1] + (j as f32 + 0.5 + (r(i, j, 2) - 0.5) * self.jitter) * s;
                if x > max[0] || z > max[1] {
                    continue;
                }
                if let Area::Disc { centre, radius } = self.area {
                    let (dx, dz) = (x - centre[0], z - centre[1]);
                    if dx * dx + dz * dz > radius * radius {
                        continue;
                    }
                }
                if self.density < 1.0 {
                    // Patchy mask (two octaves) plus a little per-point noise,
                    // remapped so about `density` of the points survive.
                    let f = 1.0 / self.patch_m;
                    let m = noise2(seed, x * f, z * f) * 0.7
                        + noise2(seed ^ 0x9e37, x * f * 2.3, z * f * 2.3) * 0.3;
                    let m = m * 0.5 + 0.5 + (r(i, j, 3) - 0.5) * 0.25;
                    if m > self.density * 1.25 - 0.125 {
                        continue;
                    }
                }
                if let Some(keep) = &self.keep {
                    if !keep(x, z) {
                        continue;
                    }
                }
                let y = self.ground.as_ref().map_or(0.0, |g| g(x, z));
                if !y.is_finite() {
                    return Err(EngineError::InvalidValue(format!(
                        "scatter ground height at ({x}, {z}) is not finite"
                    )));
                }
                let yaw = self.yaw.unwrap_or_else(|| r(i, j, 4) * 360.0);
                let scale = self.scale.0 + (self.scale.1 - self.scale.0) * r(i, j, 5);
                out.push(Place::new(x, y, z).with_yaw_deg(yaw).with_scale(scale));
            }
        }
        Ok(out)
    }

    /// Deal `places` at random into `n` lists, for spreading one scatter
    /// over several model variants. Deterministic per `seed`.
    pub fn split(places: &[Place], n: usize, seed: u32) -> Vec<Vec<Place>> {
        let n = n.max(1);
        let mut out = vec![Vec::with_capacity(places.len() / n + 1); n];
        let seed = hash_u32(seed ^ 0x51_17);
        for (i, p) in places.iter().enumerate() {
            let k = hash_u32(seed ^ hash_u32(i as u32)) as usize % n;
            out[k].push(*p);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xz(p: &Place) -> (f32, f32) {
        let m = p.to_matrix().w_axis;
        (m.x, m.z)
    }

    #[test]
    fn scatter_is_deterministic_and_fills_the_area() {
        let a = Scatter::new(3)
            .rect(0.0, 0.0, 100.0, 50.0)
            .spacing(5.0)
            .places()
            .unwrap();
        let b = Scatter::new(3)
            .rect(0.0, 0.0, 100.0, 50.0)
            .spacing(5.0)
            .places()
            .unwrap();
        let c = Scatter::new(4)
            .rect(0.0, 0.0, 100.0, 50.0)
            .spacing(5.0)
            .places()
            .unwrap();
        assert_eq!(a.len(), 200);
        assert_eq!(
            a.iter().map(xz).collect::<Vec<_>>(),
            b.iter().map(xz).collect::<Vec<_>>()
        );
        assert_ne!(xz(&a[0]), xz(&c[0]));
        assert!(a
            .iter()
            .map(xz)
            .all(|(x, z)| (0.0..=100.0).contains(&x) && (0.0..=50.0).contains(&z)));
    }

    #[test]
    fn density_ground_disc_and_filters() {
        let half = Scatter::new(1)
            .rect(0.0, 0.0, 400.0, 400.0)
            .spacing(4.0)
            .density(0.5);
        let n = half.places().unwrap().len() as f32 / 10_000.0;
        assert!((0.35..0.65).contains(&n), "kept {n}");
        let disc = Scatter::new(1)
            .disc(10.0, 10.0, 20.0)
            .spacing(1.0)
            .places()
            .unwrap();
        let area = std::f32::consts::PI * 400.0;
        assert!(
            (disc.len() as f32 - area).abs() < area * 0.06,
            "{} points",
            disc.len()
        );
        let ground = Scatter::new(1)
            .spacing(10.0)
            .on_ground(|x, _| x * 0.1)
            .only_where(|x, _| x > 0.0)
            .places()
            .unwrap();
        assert!(ground.iter().all(|p| {
            let t = p.to_matrix().w_axis;
            t.x > 0.0 && (t.y - t.x * 0.1).abs() < 1e-4
        }));
    }

    #[test]
    fn bad_input_fails_and_split_keeps_everything() {
        assert!(Scatter::new(0).spacing(0.0).places().is_err());
        assert!(Scatter::new(0).spacing(0.001).places().is_err());
        assert!(Scatter::new(0).density(1.5).places().is_err());
        let all = Scatter::new(2).places().unwrap();
        let parts = Scatter::split(&all, 3, 9);
        assert_eq!(parts.iter().map(Vec::len).sum::<usize>(), all.len());
        assert!(parts.iter().all(|p| p.len() > all.len() / 5));
    }
}
