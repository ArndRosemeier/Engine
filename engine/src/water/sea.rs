//! Open sea edges: the region joins an ocean that sends in swell and takes
//! outgoing waves away.

use super::sim::{Water, G};
use glam::Vec2;

/// A train of long waves arriving from the open sea.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Swell {
    /// Crest to trough, metres.
    pub height: f32,
    /// Seconds between crests.
    pub period: f32,
    /// Direction the waves travel, degrees (0 = +Z, 90 = +X).
    pub direction_deg: f32,
}

/// Which edges of the region open onto the sea.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeaEdges {
    pub min_x: bool,
    pub max_x: bool,
    pub min_z: bool,
    pub max_z: bool,
}

impl SeaEdges {
    pub const ALL: Self = Self {
        min_x: true,
        max_x: true,
        min_z: true,
        max_z: true,
    };
}

/// The ocean outside the region, from [`Water::set_sea`].
///
/// Along the open edges every cell at sea level is joined to the ocean by a
/// radiating boundary: waves leaving the region pass out instead of bouncing
/// back, and the swell trains come in. The sea keeps its mean level, so the
/// volume inside changes only as waves come and go (and as the region drains
/// into or fills from it). Everything else stays closed.
#[derive(Clone, Debug, PartialEq)]
pub struct Sea {
    /// Still-water level of the ocean.
    pub level: f32,
    pub swell: Vec<Swell>,
    pub edges: SeaEdges,
    /// Seconds over which the swell builds up after it is set, so it does not
    /// start with a jolt.
    pub ramp_seconds: f32,
}

impl Sea {
    /// A calm sea on every edge.
    pub fn calm(level: f32) -> Self {
        Self {
            level,
            swell: Vec::new(),
            edges: SeaEdges::ALL,
            ramp_seconds: 12.0,
        }
    }

    pub fn with_swell(mut self, swell: Swell) -> Self {
        self.swell.push(swell);
        self
    }

    pub fn with_edges(mut self, edges: SeaEdges) -> Self {
        self.edges = edges;
        self
    }

    /// How far the swell has built up at sea time `t`, 0..1.
    pub fn ramp(&self, t: f32) -> f32 {
        let r = (t / self.ramp_seconds.max(1e-3)).clamp(0.0, 1.0);
        r * r * (3.0 - 2.0 * r)
    }

    /// Surface of the incoming swell above `level` at `p` and time `t`, for
    /// still-water depth `depth` (waves travel at √(g·depth)).
    pub fn incoming(&self, p: Vec2, t: f32, depth: f32) -> f32 {
        let ramp = self.ramp(t);
        let c = (G * depth.max(0.5)).sqrt();
        let mut eta = 0.0;
        for (i, s) in self.swell.iter().enumerate() {
            let omega = std::f32::consts::TAU / s.period.max(0.5);
            let dir = Vec2::new(
                s.direction_deg.to_radians().sin(),
                s.direction_deg.to_radians().cos(),
            );
            let k = dir * (omega / c);
            eta += 0.5 * s.height * (omega * t - k.dot(p) + i as f32 * 1.3).cos();
        }
        eta * ramp
    }
}

/// One cell joined to water outside the region, through one face.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Boundary {
    pub cell: u32,
    /// Centre of the face on the region edge.
    pub at: Vec2,
    /// Still level of the water outside this face.
    pub level: f32,
}

impl Water {
    /// Join the region's open edges to an ocean (or close them again with
    /// `None`). Fill it first with [`Water::flood_sea`].
    pub fn set_sea(&mut self, sea: Option<Sea>) {
        self.sea = sea;
        self.sea_clock = 0.0;
        self.rebuild_boundary();
        self.wake_all();
    }

    pub fn sea(&self) -> Option<&Sea> {
        self.sea.as_ref()
    }

    /// Open the region's edges onto the water that lies outside it:
    /// `level(x, z)` is the still level of that water at a point on the edge
    /// (`None` for dry land). A river that crosses the region then flows in
    /// at its upstream edge and out at the other, and a lake that the region
    /// only covers a part of keeps its level. Edges at the level of the sea
    /// ([`Water::set_sea`]) carry its swell. This replaces the sea's own
    /// choice of edges; call it again after the levels outside change.
    pub fn set_open_edges(&mut self, level: impl Fn(f32, f32) -> Option<f32>) {
        let r = self.region;
        let (nx, nz) = (r.cells_x, r.cells_z);
        let cs = r.cell_size;
        let mut faces = Vec::new();
        let mut add = |x: u32, z: u32, at: Vec2| {
            if let Some(level) = level(at.x, at.y).filter(|l| l.is_finite()) {
                faces.push(EdgeFace { x, z, at, level });
            }
        };
        for z in 0..nz {
            let zc = r.min_xz.y + (z as f32 + 0.5) * cs;
            add(0, z, Vec2::new(r.min_xz.x, zc));
            add(nx - 1, z, Vec2::new(r.max_xz().x, zc));
        }
        for x in 0..nx {
            let xc = r.min_xz.x + (x as f32 + 0.5) * cs;
            add(x, 0, Vec2::new(xc, r.min_xz.y));
            add(x, nz - 1, Vec2::new(xc, r.max_xz().y));
        }
        self.open_edges = Some(faces);
        self.rebuild_boundary();
        self.wake_all();
    }

    /// Seconds since the sea was set (the swell's clock).
    pub fn sea_time(&self) -> f32 {
        self.sea_clock as f32
    }

    /// Mean still-water depth along the open edges: how fast the swell
    /// travels in the sea drawn beyond the region.
    pub fn sea_depth(&self) -> f32 {
        self.sea_depth
    }

    pub(crate) fn rebuild_boundary(&mut self) {
        self.boundary.clear();
        self.boundary_rev = super::sim::next_rev();
        let r = self.region;
        let (nx, nz) = (r.cells_x, r.cells_z);
        // The faces that open, each with the level outside it.
        let faces: Vec<EdgeFace> = match (&self.open_edges, &self.sea) {
            (Some(faces), _) => faces.clone(),
            (None, Some(sea)) => {
                let mut faces = Vec::new();
                let cs = r.cell_size;
                let level = sea.level;
                for z in 0..nz {
                    let zc = r.min_xz.y + (z as f32 + 0.5) * cs;
                    if sea.edges.min_x {
                        let at = Vec2::new(r.min_xz.x, zc);
                        faces.push(EdgeFace { x: 0, z, at, level });
                    }
                    if sea.edges.max_x {
                        let (x, at) = (nx - 1, Vec2::new(r.max_xz().x, zc));
                        faces.push(EdgeFace { x, z, at, level });
                    }
                }
                for x in 0..nx {
                    let xc = r.min_xz.x + (x as f32 + 0.5) * cs;
                    if sea.edges.min_z {
                        let at = Vec2::new(xc, r.min_xz.y);
                        faces.push(EdgeFace { x, z: 0, at, level });
                    }
                    if sea.edges.max_z {
                        let (z, at) = (nz - 1, Vec2::new(xc, r.max_xz().y));
                        faces.push(EdgeFace { x, z, at, level });
                    }
                }
                faces
            }
            (None, None) => return,
        };
        let mut list = Vec::new();
        for f in &faces {
            let c = (f.z * nx + f.x) as usize;
            for cell in self.col_start[c]..self.col_start[c + 1] {
                let i = cell as usize;
                // The cell holding that surface (the open-air one on a coast).
                if self.floor[i] < f.level + 2.0 && self.ceil[i] > f.level {
                    list.push(Boundary {
                        cell,
                        at: f.at,
                        level: f.level,
                    });
                }
            }
        }
        if let Some(sea) = self.sea.as_ref() {
            let wet: Vec<f32> = list
                .iter()
                .filter(|b| at_sea(b.level, sea.level))
                .map(|b| sea.level - self.floor[b.cell as usize])
                .filter(|d| *d > 0.0)
                .collect();
            self.sea_depth = if wet.is_empty() {
                10.0
            } else {
                wet.iter().sum::<f32>() / wet.len() as f32
            };
        }
        self.boundary = list;
        self.boundary_q = vec![0.0; self.boundary.len()];
    }

    /// Outward discharge through every open face for this sub-step (radiation
    /// condition with an incoming wave: q = √(g·h)·(η − 2·η_in)). Adds
    /// outflow per cell to `out`. Returns the fastest wave speed.
    pub(crate) fn sea_fluxes(&mut self, dt: f32, out: &mut [f32]) -> f32 {
        if self.boundary.is_empty() {
            return 0.0;
        }
        let sea = self.sea.as_ref();
        let t = self.sea_clock as f32;
        let dx = self.region.cell_size;
        let mut fastest: f32 = 0.0;
        for (k, b) in self.boundary.iter().enumerate() {
            let i = b.cell as usize;
            if !self.cell_runs(i) {
                self.boundary_q[k] = 0.0;
                continue;
            }
            let floor = self.floor[i];
            let still = (b.level - floor).max(0.0);
            let head = self.head[i].min(self.ceil[i]);
            // Only the sea sends swell in; a lake or river outside is still.
            let eta_in = match sea {
                Some(sea) if at_sea(b.level, sea.level) => sea.incoming(b.at, t, still),
                _ => 0.0,
            };
            if still <= 0.05 && head <= floor + 1e-4 && b.level + eta_in <= floor {
                self.boundary_q[k] = 0.0;
                continue;
            }
            let c = (G * still.max(0.05)).sqrt();
            let eta = head - b.level;
            let mut q = c * (eta - 2.0 * eta_in);
            if q > 0.0 {
                // Cannot give more than the cell holds.
                q = q.min((head - floor).max(0.0) * dx / dt);
                out[i] += q;
            }
            self.boundary_q[k] = q;
            fastest = fastest.max(c);
        }
        fastest
    }
}

/// One face on the region's edge that opens onto water at `level`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeFace {
    x: u32,
    z: u32,
    at: Vec2,
    level: f32,
}

/// Whether an open face at `level` is the sea's (and carries its swell).
pub(crate) fn at_sea(level: f32, sea: f32) -> bool {
    (level - sea).abs() < 0.01
}
