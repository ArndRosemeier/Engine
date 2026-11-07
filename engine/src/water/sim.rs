//! Layered shallow-water simulation.
//!
//! The region is a grid of columns. Each column holds one cell per open span
//! (a cave under a hill is a second cell below the open-air one). Neighbouring
//! cells whose spans overlap are joined by a link; water only ever moves along
//! links, and links only exist through open space, so water cannot pass
//! through rock or out of the region.
//!
//! Every link carries a discharge `q` (m²/s per metre of face) updated with
//! the local inertial form of the shallow water equations (Bates et al. 2010,
//! with the θ smoothing of de Almeida et al. 2012). Waves travel at the
//! physical speed √(g·h), flow runs up and down beaches (wetting and drying),
//! and water drops off ledges at the critical rate. A cell that fills to its
//! ceiling becomes pressurised: its head keeps rising stiffly, so water pushes
//! through flooded tunnels and U-bends and rises in the far shaft.
//!
//! Guarantees, by construction rather than by tuning:
//! - Volume is conserved: every link moves water from one cell to another.
//! - Depth never goes negative: outflow is limited to what a cell holds.
//! - No step can blow up: the step follows the CFL limit, and when a frame
//!   would need more sub-steps than allowed, simulated time slows instead.

use super::terrain::{Span, WaterTerrain};
use glam::{Vec2, Vec3};
use rayon::prelude::*;

pub(crate) const G: f32 = 9.81;
/// Depth below which a cell counts as dry (metres).
pub(crate) const DRY: f32 = 1e-4;
const NONE: u32 = u32::MAX;
/// Bit set in a cell's link list when the cell is the link's `b` side.
const B_SIDE: u32 = 1 << 31;

/// Area and grid of the simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterRegion {
    /// World X and Z of the region's low corner.
    pub min_xz: Vec2,
    /// Columns along X and Z.
    pub cells_x: u32,
    pub cells_z: u32,
    /// Column width in metres.
    pub cell_size: f32,
    /// The region's closed floor; nothing below is simulated.
    pub y_min: f32,
    /// Spans reaching this height are open to the sky.
    pub y_max: f32,
}

impl WaterRegion {
    /// A region covering `[min_xz, max_xz]` with columns `cell_size` wide.
    pub fn new(min_xz: Vec2, max_xz: Vec2, cell_size: f32, y_min: f32, y_max: f32) -> Self {
        let cell_size = cell_size.max(0.05);
        let extent = (max_xz - min_xz).max(Vec2::splat(cell_size));
        Self {
            min_xz,
            cells_x: (extent.x / cell_size).round().max(1.0) as u32,
            cells_z: (extent.y / cell_size).round().max(1.0) as u32,
            cell_size,
            y_min,
            y_max,
        }
    }

    pub fn max_xz(&self) -> Vec2 {
        self.min_xz + Vec2::new(self.cells_x as f32, self.cells_z as f32) * self.cell_size
    }

    pub fn column_count(&self) -> usize {
        self.cells_x as usize * self.cells_z as usize
    }

    /// Centre of column `(x, z)`.
    pub fn column_centre(&self, x: u32, z: u32) -> Vec2 {
        self.min_xz + (Vec2::new(x as f32, z as f32) + 0.5) * self.cell_size
    }

    /// Column under a world XZ position.
    pub fn column_at(&self, x: f32, z: f32) -> Option<(u32, u32)> {
        let g = (Vec2::new(x, z) - self.min_xz) / self.cell_size;
        if g.x < 0.0 || g.y < 0.0 {
            return None;
        }
        let (cx, cz) = (g.x as u32, g.y as u32);
        (cx < self.cells_x && cz < self.cells_z).then_some((cx, cz))
    }
}

/// Simulation constants. The defaults suit natural water at 0.5–2 m cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSettings {
    /// Manning roughness of the beds (s/m^⅓). Higher drags flow more.
    pub manning: f32,
    /// How steeply the head of a full (pressurised) cell rises per metre of
    /// extra water. Higher is closer to incompressible but needs smaller steps.
    pub stiffness: f32,
    /// Fraction of the CFL limit each sub-step uses (0.1–0.9).
    pub courant: f32,
    /// Most sub-steps per update; past this, simulated time slows down.
    pub max_substeps: u32,
    /// θ smoothing of the discharge (1 = none). Removes grid-scale ripples.
    pub theta: f32,
    /// Open spans thinner than this (metres) are treated as solid.
    pub min_gap: f32,
    /// Extra drag (1/s) on slow flow, fading out by `settle_speed`. Lets
    /// sloshing die down to a flat surface without touching real waves.
    pub settle: f32,
    /// Flow speed (m/s) above which `settle` no longer applies.
    pub settle_speed: f32,
    /// Surface slope above which a front counts as breaking and loses
    /// energy like a turbulent bore.
    pub breaking_slope: f32,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            manning: 0.018,
            stiffness: 20.0,
            courant: 0.6,
            max_substeps: 24,
            theta: 0.85,
            min_gap: 0.15,
            settle: 0.3,
            settle_speed: 0.4,
            breaking_slope: 0.2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Link {
    /// Cell on the low side (−X or −Z).
    pub a: u32,
    /// Cell on the high side.
    pub b: u32,
    /// Bottom and top of the opening the two spans share.
    pub lo: f32,
    pub hi: f32,
    /// 0 = X face, 1 = Z face.
    pub axis: u8,
}

/// A steady inflow (positive rate) or drain (negative rate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSource {
    pub position: Vec3,
    /// Cubic metres per second.
    pub rate: f32,
}

/// Handle of a [`WaterSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(u64);

/// Water at a point, from [`Water::sample`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSample {
    /// Height of the water surface above the point (smoothly interpolated).
    /// For a flooded cave this is the cave ceiling.
    pub surface_y: f32,
    /// How far the point is below the surface (negative above it).
    pub depth: f32,
    /// Water depth from the bed to the surface here.
    pub water_depth: f32,
    /// The water fills its span to the ceiling: there is no air to surface
    /// into here.
    pub capped: bool,
    /// Pressure head (metres of water) at the point, at least `depth`.
    pub pressure_head: f32,
    /// Horizontal current in m/s (X, Z).
    pub current: Vec2,
}

impl WaterSample {
    pub fn submerged(&self) -> bool {
        self.depth > 0.0
    }
}

/// Counters from the last [`Water::update`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterStats {
    pub substeps: u32,
    /// Seconds simulated in the last update (less than asked when slowed).
    pub simulated_seconds: f32,
    /// Fastest signal (flow plus wave speed), m/s.
    pub max_speed: f32,
    pub cells: usize,
    pub links: usize,
    pub wet_cells: usize,
}

/// The unified water simulation. See the module docs of [`crate::water`].
#[derive(Clone)]
pub struct Water {
    pub(crate) region: WaterRegion,
    pub(crate) settings: WaterSettings,
    look: super::WaterLook,
    // Topology: cells of column c are col_start[c]..col_start[c + 1].
    pub(crate) col_start: Vec<u32>,
    pub(crate) floor: Vec<f32>,
    pub(crate) ceil: Vec<f32>,
    pub(crate) cell_col: Vec<u32>,
    /// Water per unit area (metres). Above `ceil - floor` the cell is pressurised.
    pub(crate) depth: Vec<f32>,
    pub(crate) links: Vec<Link>,
    pub(crate) q: Vec<f32>,
    /// Links of column c along axis k are link_group[2c + k]..link_group[2c + k + 1].
    link_group: Vec<u32>,
    /// Upstream and downstream link along the same axis (θ smoothing).
    link_prev: Vec<u32>,
    link_next: Vec<u32>,
    cell_link_start: Vec<u32>,
    cell_links: Vec<u32>,
    // Scratch.
    pub(crate) head: Vec<f32>,
    q_new: Vec<f32>,
    scale: Vec<f32>,
    sources: Vec<(SourceId, WaterSource)>,
    next_source: u64,
    pub(crate) sea: Option<super::Sea>,
    pub(crate) sea_clock: f64,
    pub(crate) boundary: Vec<super::sea::Boundary>,
    pub(crate) boundary_q: Vec<f32>,
    pub(crate) sea_depth: f32,
    /// Per cell: outflow to the sea this step, then net exchange.
    sea_out: Vec<f32>,
    /// Next sub-step size from the CFL limit of the previous one.
    next_dt: f32,
    time: f64,
    stats: WaterStats,
    /// Bumped whenever cells or links change (renderers rebuild topology).
    pub(crate) topology_rev: u64,
}

impl std::fmt::Debug for Water {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Water")
            .field("region", &self.region)
            .field("cells", &self.floor.len())
            .field("links", &self.links.len())
            .field("volume_m3", &self.total_volume())
            .finish()
    }
}

impl Water {
    /// Build the column cells of `region` from `terrain`. Starts dry.
    pub fn new(region: WaterRegion, terrain: &dyn WaterTerrain) -> Self {
        Self::with_settings(region, terrain, WaterSettings::default())
    }

    pub fn with_settings(
        region: WaterRegion,
        terrain: &dyn WaterTerrain,
        settings: WaterSettings,
    ) -> Self {
        let ncols = region.column_count();
        let spans: Vec<Vec<Span>> = (0..ncols)
            .into_par_iter()
            .map(|c| column_spans(&region, &settings, terrain, c))
            .collect();
        let mut water = Self {
            region,
            settings,
            look: super::WaterLook::default(),
            col_start: Vec::new(),
            floor: Vec::new(),
            ceil: Vec::new(),
            cell_col: Vec::new(),
            depth: Vec::new(),
            links: Vec::new(),
            q: Vec::new(),
            link_group: Vec::new(),
            link_prev: Vec::new(),
            link_next: Vec::new(),
            cell_link_start: Vec::new(),
            cell_links: Vec::new(),
            head: Vec::new(),
            q_new: Vec::new(),
            scale: Vec::new(),
            sources: Vec::new(),
            next_source: 1,
            sea: None,
            sea_clock: 0.0,
            boundary: Vec::new(),
            boundary_q: Vec::new(),
            sea_depth: 10.0,
            sea_out: Vec::new(),
            next_dt: 1.0 / 120.0,
            time: 0.0,
            stats: WaterStats::default(),
            topology_rev: 1,
        };
        water.set_cells(
            spans
                .iter()
                .map(|s| s.iter().map(|sp| (*sp, 0.0)).collect())
                .collect(),
        );
        water.rebuild_links(None, &[]);
        water
    }

    pub fn region(&self) -> &WaterRegion {
        &self.region
    }

    pub fn settings(&self) -> &WaterSettings {
        &self.settings
    }

    pub fn set_settings(&mut self, settings: WaterSettings) {
        self.settings = settings;
    }

    /// How the water is drawn.
    pub fn look(&self) -> &super::WaterLook {
        &self.look
    }

    pub fn set_look(&mut self, look: super::WaterLook) {
        self.look = look;
    }

    /// Builder form of [`Water::set_look`].
    pub fn with_look(mut self, look: super::WaterLook) -> Self {
        self.look = look;
        self
    }

    pub fn stats(&self) -> WaterStats {
        self.stats
    }

    /// Simulated seconds since creation.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Total water in cubic metres.
    pub fn total_volume(&self) -> f64 {
        let area = (self.region.cell_size as f64).powi(2);
        self.depth.iter().map(|&d| d as f64).sum::<f64>() * area
    }

    pub fn cell_count(&self) -> usize {
        self.floor.len()
    }

    // ---------------------------------------------------------------- filling

    /// Fill the body of water connected to `seed` up to `level`: every cell
    /// reachable from the seed through openings below `level`. Sealed caves
    /// stay dry. Returns the cubic metres added.
    pub fn flood(&mut self, seed: Vec3, level: f32) -> f64 {
        match self
            .cell_at(seed)
            .or_else(|| self.cell_below(seed))
            .or_else(|| self.cell_above(seed))
        {
            Some(cell) => self.flood_from(&[cell], level),
            None => 0.0,
        }
    }

    /// Fill everything connected to the region's border below `level`, like
    /// a sea that reaches in from outside. Returns the cubic metres added.
    pub fn flood_sea(&mut self, level: f32) -> f64 {
        let (nx, nz) = (self.region.cells_x, self.region.cells_z);
        let mut seeds = Vec::new();
        for z in 0..nz {
            for x in 0..nx {
                if x == 0 || z == 0 || x == nx - 1 || z == nz - 1 {
                    let c = (z * nx + x) as usize;
                    for cell in self.col_start[c]..self.col_start[c + 1] {
                        if self.floor[cell as usize] < level {
                            seeds.push(cell);
                        }
                    }
                }
            }
        }
        self.flood_from(&seeds, level)
    }

    fn flood_from(&mut self, seeds: &[u32], level: f32) -> f64 {
        let before = self.total_volume();
        let mut seen = vec![false; self.floor.len()];
        let mut stack: Vec<u32> = Vec::new();
        for &s in seeds {
            if self.floor[s as usize] < level && !seen[s as usize] {
                seen[s as usize] = true;
                stack.push(s);
            }
        }
        while let Some(cell) = stack.pop() {
            let i = cell as usize;
            let target = self.depth_for_head(i, level);
            if target > self.depth[i] {
                self.depth[i] = target;
            }
            let (from, to) = (
                self.cell_link_start[i] as usize,
                self.cell_link_start[i + 1] as usize,
            );
            for k in from..to {
                let entry = self.cell_links[k];
                let link = &self.links[(entry & !B_SIDE) as usize];
                if link.lo >= level {
                    continue;
                }
                let other = if entry & B_SIDE != 0 { link.a } else { link.b };
                if !seen[other as usize] {
                    seen[other as usize] = true;
                    stack.push(other);
                }
            }
        }
        self.time_step_reset();
        self.total_volume() - before
    }

    /// Depth that gives cell `i` the head `level` (pressurised if needed).
    fn depth_for_head(&self, i: usize, level: f32) -> f32 {
        let cap = self.ceil[i] - self.floor[i];
        let free = level - self.floor[i];
        if free <= 0.0 {
            0.0
        } else if free <= cap {
            free
        } else {
            cap + (level - self.ceil[i]) / self.settings.stiffness
        }
    }

    /// Add (or with a negative amount, remove) `volume` cubic metres of water
    /// at `at`. Returns the volume actually added or removed.
    pub fn pour(&mut self, at: Vec3, volume: f32) -> f32 {
        let Some(cell) = self.cell_at(at).or_else(|| self.cell_above(at)) else {
            return 0.0;
        };
        let area = self.region.cell_size * self.region.cell_size;
        let i = cell as usize;
        let new = (self.depth[i] + volume / area).max(0.0);
        let moved = (new - self.depth[i]) * area;
        self.depth[i] = new;
        moved
    }

    /// Rain `metres` of water onto every column (onto its top cell).
    pub fn rain(&mut self, metres: f32) {
        for c in 0..self.region.column_count() {
            let end = self.col_start[c + 1];
            if end > self.col_start[c] {
                let top = (end - 1) as usize;
                self.depth[top] = (self.depth[top] + metres).max(0.0);
            }
        }
    }

    /// Push water outward from `at`, as something falling in or a blast
    /// would: a dip that fills back in and a ring of waves running away.
    /// `strength` is the outward flow speed (m/s) at the centre, fading to
    /// zero at `radius`. Mass is untouched; only the flow changes.
    pub fn disturb(&mut self, at: Vec3, radius: f32, strength: f32) {
        if !(radius > 0.0 && strength.is_finite() && at.is_finite()) {
            return;
        }
        let r = self.region;
        let cs = r.cell_size;
        let lo = ((Vec2::new(at.x, at.z) - Vec2::splat(radius) - r.min_xz) / cs).floor();
        let hi = ((Vec2::new(at.x, at.z) + Vec2::splat(radius) - r.min_xz) / cs).ceil();
        let (nx, nz) = (r.cells_x as i64, r.cells_z as i64);
        for z in (lo.y as i64).clamp(0, nz)..(hi.y as i64).clamp(0, nz) {
            for x in (lo.x as i64).clamp(0, nx)..(hi.x as i64).clamp(0, nx) {
                let c = (z * nx + x) as usize;
                for cell in self.col_start[c] as usize..self.col_start[c + 1] as usize {
                    let head = self.cell_head(cell);
                    if self.depth[cell] <= DRY
                        || self.floor[cell] > at.y + radius
                        || head.min(self.ceil[cell]) < at.y - radius
                    {
                        continue;
                    }
                    let (from, to) = (
                        self.cell_link_start[cell] as usize,
                        self.cell_link_start[cell + 1] as usize,
                    );
                    for k in from..to {
                        let entry = self.cell_links[k];
                        if entry & B_SIDE != 0 {
                            continue; // each link once, from its low side
                        }
                        let id = entry as usize;
                        let link = self.links[id];
                        // Centre of the face between the two columns.
                        let centre = r.column_centre(x as u32, z as u32);
                        let face = if link.axis == 0 {
                            centre + Vec2::new(cs * 0.5, 0.0)
                        } else {
                            centre + Vec2::new(0.0, cs * 0.5)
                        };
                        let d = face - Vec2::new(at.x, at.z);
                        let dist = d.length();
                        if dist >= radius {
                            continue;
                        }
                        let w = (1.0 - dist / radius).powi(2);
                        let along = d[link.axis as usize];
                        let h = (head.min(link.hi) - link.lo).max(0.0);
                        self.q[id] += along.signum() * strength * w * h;
                    }
                }
            }
        }
        self.time_step_reset();
    }

    /// Remove all water.
    pub fn clear(&mut self) {
        self.depth.iter_mut().for_each(|d| *d = 0.0);
        self.q.iter_mut().for_each(|q| *q = 0.0);
    }

    pub fn add_source(&mut self, source: WaterSource) -> SourceId {
        let id = SourceId(self.next_source);
        self.next_source += 1;
        self.sources.push((id, source));
        id
    }

    pub fn set_source(&mut self, id: SourceId, source: WaterSource) {
        if let Some(s) = self.sources.iter_mut().find(|(i, _)| *i == id) {
            s.1 = source;
        }
    }

    pub fn remove_source(&mut self, id: SourceId) {
        self.sources.retain(|(i, _)| *i != id);
    }

    // ---------------------------------------------------------------- queries

    /// Cell whose span contains `p`.
    pub(crate) fn cell_at(&self, p: Vec3) -> Option<u32> {
        let (cx, cz) = self.region.column_at(p.x, p.z)?;
        let c = (cz * self.region.cells_x + cx) as usize;
        (self.col_start[c]..self.col_start[c + 1])
            .find(|&i| p.y >= self.floor[i as usize] && p.y <= self.ceil[i as usize])
    }

    /// Highest cell of the column whose floor is below `p`.
    fn cell_below(&self, p: Vec3) -> Option<u32> {
        let (cx, cz) = self.region.column_at(p.x, p.z)?;
        let c = (cz * self.region.cells_x + cx) as usize;
        (self.col_start[c]..self.col_start[c + 1])
            .rev()
            .find(|&i| self.floor[i as usize] <= p.y)
    }

    /// Lowest cell of the column whose floor is above `p`, else the top one.
    fn cell_above(&self, p: Vec3) -> Option<u32> {
        let (cx, cz) = self.region.column_at(p.x, p.z)?;
        let c = (cz * self.region.cells_x + cx) as usize;
        let range = self.col_start[c]..self.col_start[c + 1];
        range
            .clone()
            .find(|&i| self.floor[i as usize] >= p.y)
            .or_else(|| range.last())
    }

    pub(crate) fn cell_head(&self, i: usize) -> f32 {
        head_of(
            self.floor[i],
            self.ceil[i],
            self.depth[i],
            self.settings.stiffness,
        )
    }

    /// Water at `p`, or `None` where `p` is in rock, outside the region, or
    /// in a dry cell.
    pub fn sample(&self, p: Vec3) -> Option<WaterSample> {
        let cell = self.cell_at(p)? as usize;
        if self.depth[cell] <= DRY {
            return None;
        }
        let head = self.cell_head(cell);
        let capped = head >= self.ceil[cell];
        let surface_y = if capped {
            self.ceil[cell]
        } else {
            self.smooth_surface(cell, p)
        };
        let depth = surface_y - p.y;
        Some(WaterSample {
            surface_y,
            depth,
            water_depth: (surface_y - self.floor[cell]).max(0.0),
            capped,
            pressure_head: (head.max(surface_y) - p.y).max(depth),
            current: self.cell_current(cell),
        })
    }

    /// Whether `p` is under water.
    pub fn is_submerged(&self, p: Vec3) -> bool {
        self.sample(p).is_some_and(|s| s.submerged())
    }

    /// Water surface height in the cell containing (or just below) `p`.
    pub fn surface_at(&self, p: Vec3) -> Option<f32> {
        let cell = self.cell_at(p).or_else(|| self.cell_below(p))? as usize;
        (self.depth[cell] > DRY).then(|| self.smooth_surface(cell, p).min(self.ceil[cell]))
    }

    /// Current (m/s, X and Z) in cell `i`, from the discharge on its faces.
    pub(crate) fn cell_current(&self, i: usize) -> Vec2 {
        let h = (self.cell_head(i).min(self.ceil[i]) - self.floor[i]).max(0.05);
        let mut sum = Vec2::ZERO;
        let mut count = Vec2::ZERO;
        let (from, to) = (
            self.cell_link_start[i] as usize,
            self.cell_link_start[i + 1] as usize,
        );
        for k in from..to {
            let id = (self.cell_links[k] & !B_SIDE) as usize;
            let axis = self.links[id].axis as usize;
            sum[axis] += self.q[id];
            count[axis] += 1.0;
        }
        // Mean discharge over the cell's faces per axis, as velocity.
        Vec2::new(sum.x / count.x.max(1.0), sum.y / count.y.max(1.0)) / h
    }

    /// Surface height at `p`, bilinear between this cell and its linked,
    /// wet neighbours, so it is continuous across columns.
    fn smooth_surface(&self, cell: usize, p: Vec3) -> f32 {
        let own = self.cell_head(cell).min(self.ceil[cell]);
        let r = &self.region;
        let g = (Vec2::new(p.x, p.z) - r.min_xz) / r.cell_size - Vec2::splat(0.5);
        let base = g.floor();
        let f = g - base;
        let col = self.cell_col[cell];
        let (cx, cz) = ((col % r.cells_x) as i32, (col / r.cells_x) as i32);
        // Which corner of the 2x2 stencil this column is.
        let ox = cx - base.x as i32; // 0 or 1
        let oz = cz - base.y as i32;
        let height_from = |dx: i32, dz: i32| -> f32 {
            if dx == 0 && dz == 0 {
                return own;
            }
            let neighbour = if dx != 0 && dz != 0 {
                // Diagonal: through either side neighbour.
                self.linked_cell(cell, dx, 0)
                    .and_then(|s| self.linked_cell(s, 0, dz))
                    .or_else(|| {
                        self.linked_cell(cell, 0, dz)
                            .and_then(|s| self.linked_cell(s, dx, 0))
                    })
            } else {
                self.linked_cell(cell, dx, dz)
            };
            match neighbour {
                Some(n) if self.depth[n] > DRY => self.cell_head(n).min(self.ceil[n]),
                _ => own,
            }
        };
        let h = |sx: i32, sz: i32| height_from(sx - ox, sz - oz);
        let h00 = h(0, 0);
        let h10 = h(1, 0);
        let h01 = h(0, 1);
        let h11 = h(1, 1);
        let a = h00 + (h10 - h00) * f.x;
        let b = h01 + (h11 - h01) * f.x;
        a + (b - a) * f.y
    }

    /// The neighbour of `cell` one column over (dx or dz is ±1) joined to it
    /// by the widest link.
    pub(crate) fn linked_cell(&self, cell: usize, dx: i32, dz: i32) -> Option<usize> {
        let axis = if dx != 0 { 0 } else { 1 };
        let positive = dx + dz > 0;
        let (from, to) = (
            self.cell_link_start[cell] as usize,
            self.cell_link_start[cell + 1] as usize,
        );
        let mut best: Option<(f32, usize)> = None;
        for k in from..to {
            let entry = self.cell_links[k];
            let id = (entry & !B_SIDE) as usize;
            let link = &self.links[id];
            if link.axis as usize != axis {
                continue;
            }
            // We are `a` when the neighbour is on the positive side.
            let we_are_a = entry & B_SIDE == 0;
            if we_are_a != positive {
                continue;
            }
            let other = if we_are_a { link.b } else { link.a } as usize;
            let width = link.hi.min(1e6) - link.lo;
            if best.is_none_or(|(w, _)| width > w) {
                best = Some((width, other));
            }
        }
        best.map(|(_, c)| c)
    }

    // ---------------------------------------------------------------- stepping

    /// Advance the simulation by `dt` seconds (clamped to 0.1 s).
    pub fn update(&mut self, dt: f32) {
        let mut remaining = dt.clamp(0.0, 0.1);
        let mut steps = 0;
        let mut simulated = 0.0;
        let mut max_speed: f32 = 0.0;
        while remaining > 1e-6 && steps < self.settings.max_substeps.max(1) {
            let step = self.next_dt.min(remaining);
            let speed = self.step(step);
            max_speed = max_speed.max(speed);
            remaining -= step;
            simulated += step;
            steps += 1;
        }
        self.time += simulated as f64;
        self.stats = WaterStats {
            substeps: steps,
            simulated_seconds: simulated,
            max_speed,
            cells: self.floor.len(),
            links: self.links.len(),
            wet_cells: self.depth.iter().filter(|&&d| d > DRY).count(),
        };
    }

    /// Forget the CFL history after a sudden change (flood, edit).
    fn time_step_reset(&mut self) {
        self.next_dt = self.next_dt.min(1.0 / 240.0);
    }

    /// One explicit sub-step. Returns the fastest signal speed seen.
    fn step(&mut self, dt: f32) -> f32 {
        self.apply_sources(dt);
        let s = self.settings;
        let dx = self.region.cell_size;
        let stiffness = s.stiffness;

        // 1. Heads.
        {
            let (floor, ceil, depth) = (&self.floor, &self.ceil, &self.depth);
            self.head
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, h)| *h = head_of(floor[i], ceil[i], depth[i], stiffness));
        }

        // 2. Discharge on every link.
        let n2 = s.manning * s.manning;
        let theta = s.theta.clamp(0.0, 1.0);
        let (settle, settle_speed, breaking_slope) = (
            s.settle.max(0.0),
            s.settle_speed.max(1e-3),
            s.breaking_slope.max(1e-3),
        );
        let (links, head, q_old, prev, next, ceil, floor) = (
            &self.links,
            &self.head,
            &self.q,
            &self.link_prev,
            &self.link_next,
            &self.ceil,
            &self.floor,
        );
        let max_speed = self
            .q_new
            .par_iter_mut()
            .enumerate()
            .map(|(i, out)| {
                let l = links[i];
                let (a, b) = (l.a as usize, l.b as usize);
                let (ha, hb) = (head[a], head[b]);
                let eta = ha.max(hb);
                if eta <= l.lo + DRY {
                    *out = 0.0;
                    return 0.0;
                }
                // Flow depth through the shared opening.
                let h = (eta.min(l.hi) - l.lo).max(DRY);
                // Below the lip the far side's level does not matter: water
                // falls freely.
                let (ea, eb) = (ha.max(l.lo), hb.max(l.lo));
                let q0 = q_old[i];
                let qp = if prev[i] != NONE {
                    q_old[prev[i] as usize]
                } else {
                    q0
                };
                let qn = if next[i] != NONE {
                    q_old[next[i] as usize]
                } else {
                    q0
                };
                let qt = theta * q0 + (1.0 - theta) * 0.5 * (qp + qn);
                // Drag: bed friction, settling of slow flow, breaking fronts.
                let speed = qt.abs() / h;
                let settle = settle * (1.0 - speed / settle_speed).max(0.0);
                let slope = (eb - ea).abs() / dx;
                let both_wet = ha > l.lo + DRY && hb > l.lo + DRY;
                let breaking = if both_wet {
                    (slope - breaking_slope).max(0.0) * 4.0
                } else {
                    0.0
                };
                let drag = G * n2 * qt.abs() / h.powf(7.0 / 3.0) + settle + breaking;
                let mut q = (qt - G * h * dt * (eb - ea) / dx) / (1.0 + dt * drag);
                // No faster than the head difference can drive (Torricelli)
                // plus the wave speed: keeps falls and wet fronts sane.
                let vmax = (2.0 * G * (ea - eb).abs()).sqrt() + (G * h).sqrt();
                q = q.clamp(-h * vmax, h * vmax);
                // A side can only give water it holds above the opening.
                let donor_excess = if q > 0.0 { ha - l.lo } else { hb - l.lo };
                if donor_excess <= DRY {
                    q = 0.0;
                } else {
                    let cap = donor_excess * dx / dt;
                    q = q.clamp(-cap, cap);
                }
                *out = q;
                // Signal speed for the next step's CFL limit.
                let pressurised = ha > ceil[a] || hb > ceil[b];
                let wave_depth = if pressurised {
                    h * stiffness
                } else {
                    h.min((ceil[a] - floor[a]).max(ceil[b] - floor[b]))
                };
                q.abs() / h + (G * wave_depth).sqrt()
            })
            .reduce(|| 0.0f32, f32::max);

        // Exchange with the open sea.
        let mut sea_out = std::mem::take(&mut self.sea_out);
        let sea_speed = self.sea_fluxes(dt, &mut sea_out);
        let max_speed = max_speed.max(sea_speed);

        // 3. Outflow limit per cell so no cell goes below empty.
        {
            let (cls, cl, q, depth, sea_out) = (
                &self.cell_link_start,
                &self.cell_links,
                &self.q_new,
                &self.depth,
                &sea_out,
            );
            let k = dt / dx;
            self.scale.par_iter_mut().enumerate().for_each(|(i, sc)| {
                let mut out = 0.0;
                for &entry in &cl[cls[i] as usize..cls[i + 1] as usize] {
                    let id = (entry & !B_SIDE) as usize;
                    let qv = q[id];
                    // a gives on positive q, b on negative.
                    let give = if entry & B_SIDE == 0 { qv } else { -qv };
                    if give > 0.0 {
                        out += give;
                    }
                }
                out = (out + sea_out[i]) * k;
                *sc = if out > depth[i] && out > 0.0 {
                    depth[i] / out
                } else {
                    1.0
                };
            });
        }
        {
            let (links, scale) = (&self.links, &self.scale);
            self.q_new.par_iter_mut().enumerate().for_each(|(i, q)| {
                let l = links[i];
                let donor = if *q > 0.0 { l.a } else { l.b } as usize;
                *q *= scale[donor];
            });
        }

        // Sea outflow is limited like any other; then it becomes each
        // boundary cell's net exchange (positive = inflow).
        for b in &self.boundary {
            sea_out[b.cell as usize] = 0.0;
        }
        for (k, b) in self.boundary.iter().enumerate() {
            let i = b.cell as usize;
            let mut q = self.boundary_q[k];
            if q > 0.0 {
                q *= self.scale[i];
                self.boundary_q[k] = q;
            }
            sea_out[i] -= q;
        }

        // 4. Move the water.
        {
            let (cls, cl, q, sea_net) = (
                &self.cell_link_start,
                &self.cell_links,
                &self.q_new,
                &sea_out,
            );
            let k = dt / dx;
            self.depth.par_iter_mut().enumerate().for_each(|(i, d)| {
                let mut net = sea_net[i];
                for &entry in &cl[cls[i] as usize..cls[i + 1] as usize] {
                    let qv = q[(entry & !B_SIDE) as usize];
                    net += if entry & B_SIDE == 0 { -qv } else { qv };
                }
                *d = (*d + net * k).max(0.0);
            });
        }
        for b in &self.boundary {
            sea_out[b.cell as usize] = 0.0;
        }
        self.sea_out = sea_out;
        self.sea_clock += dt as f64;
        std::mem::swap(&mut self.q, &mut self.q_new);

        let speed = max_speed.max(1e-3);
        self.next_dt = (s.courant * dx / speed).clamp(1e-4, 1.0 / 30.0);
        speed
    }

    fn apply_sources(&mut self, dt: f32) {
        if self.sources.is_empty() {
            return;
        }
        let sources: Vec<WaterSource> = self.sources.iter().map(|(_, s)| *s).collect();
        for s in sources {
            self.pour(s.position, s.rate * dt);
        }
    }

    // ---------------------------------------------------------------- topology

    /// Replace all cells: per column, (span, depth) bottom to top.
    fn set_cells(&mut self, columns: Vec<Vec<(Span, f32)>>) {
        let total: usize = columns.iter().map(|c| c.len()).sum();
        self.col_start = Vec::with_capacity(columns.len() + 1);
        self.floor = Vec::with_capacity(total);
        self.ceil = Vec::with_capacity(total);
        self.cell_col = Vec::with_capacity(total);
        self.depth = Vec::with_capacity(total);
        for (c, cells) in columns.into_iter().enumerate() {
            self.col_start.push(self.floor.len() as u32);
            for (span, d) in cells {
                self.floor.push(span.floor);
                self.ceil.push(span.ceil);
                self.cell_col.push(c as u32);
                self.depth.push(d.max(0.0));
            }
        }
        self.col_start.push(self.floor.len() as u32);
        self.head = vec![0.0; total];
        self.scale = vec![1.0; total];
        self.sea_out = vec![0.0; total];
        self.topology_rev += 1;
    }

    /// Rebuild links. With `old` (the previous topology), discharge is kept
    /// on links between columns that did not change.
    fn rebuild_links(&mut self, old: Option<&OldTopology>, changed: &[bool]) {
        let r = self.region;
        let (nx, nz) = (r.cells_x as usize, r.cells_z as usize);
        let min_gap = self.settings.min_gap;
        // Per column and axis, generated in parallel then concatenated.
        let groups: Vec<(Vec<Link>, Vec<f32>)> = (0..nx * nz * 2)
            .into_par_iter()
            .map(|g| {
                let (c, axis) = (g / 2, g % 2);
                let (x, z) = (c % nx, c / nx);
                let n = if axis == 0 {
                    if x + 1 >= nx {
                        return (Vec::new(), Vec::new());
                    }
                    c + 1
                } else {
                    if z + 1 >= nz {
                        return (Vec::new(), Vec::new());
                    }
                    c + nx
                };
                let mut links = Vec::new();
                let (a0, a1) = (self.col_start[c] as usize, self.col_start[c + 1] as usize);
                let (b0, b1) = (self.col_start[n] as usize, self.col_start[n + 1] as usize);
                // Merge-walk the two sorted span lists.
                let (mut i, mut j) = (a0, b0);
                while i < a1 && j < b1 {
                    let lo = self.floor[i].max(self.floor[j]);
                    let hi = self.ceil[i].min(self.ceil[j]);
                    if hi - lo >= min_gap {
                        links.push(Link {
                            a: i as u32,
                            b: j as u32,
                            lo,
                            hi,
                            axis: axis as u8,
                        });
                    }
                    if self.ceil[i] < self.ceil[j] {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }
                // Keep discharge where neither column changed and the same
                // links exist.
                let mut q = vec![0.0; links.len()];
                if let Some(old) = old {
                    if !changed.is_empty() && !changed[c] && !changed[n] {
                        let (o0, o1) = (old.link_group[g] as usize, old.link_group[g + 1] as usize);
                        if o1 - o0 == links.len() {
                            q.copy_from_slice(&old.q[o0..o1]);
                        }
                    }
                }
                (links, q)
            })
            .collect();
        self.links.clear();
        self.q.clear();
        self.link_group = Vec::with_capacity(groups.len() + 1);
        for (links, q) in groups {
            self.link_group.push(self.links.len() as u32);
            self.links.extend(links);
            self.q.extend(q);
        }
        self.link_group.push(self.links.len() as u32);
        self.q_new = vec![0.0; self.links.len()];

        // Per-cell link lists.
        let ncells = self.floor.len();
        let mut count = vec![0u32; ncells + 1];
        for l in &self.links {
            count[l.a as usize] += 1;
            count[l.b as usize] += 1;
        }
        let mut start = vec![0u32; ncells + 1];
        for i in 0..ncells {
            start[i + 1] = start[i] + count[i];
        }
        let mut fill = start.clone();
        let mut cl = vec![0u32; start[ncells] as usize];
        for (id, l) in self.links.iter().enumerate() {
            cl[fill[l.a as usize] as usize] = id as u32;
            fill[l.a as usize] += 1;
            cl[fill[l.b as usize] as usize] = id as u32 | B_SIDE;
            fill[l.b as usize] += 1;
        }
        self.cell_link_start = start;
        self.cell_links = cl;

        // Upstream / downstream link for θ smoothing: the link on the far
        // face of a (or b) along the same axis that overlaps this one most.
        let (links, cls, cl) = (&self.links, &self.cell_link_start, &self.cell_links);
        let pick = |cell: u32, axis: u8, want_b_side: bool, lo: f32, hi: f32| -> u32 {
            let c = cell as usize;
            let mut best = (0.0f32, NONE);
            for &entry in &cl[cls[c] as usize..cls[c + 1] as usize] {
                let id = entry & !B_SIDE;
                let l = &links[id as usize];
                if l.axis != axis || ((entry & B_SIDE) != 0) != want_b_side {
                    continue;
                }
                let overlap = l.hi.min(hi).min(1e6) - l.lo.max(lo);
                if overlap > best.0 {
                    best = (overlap, id);
                }
            }
            best.1
        };
        let (prev, next): (Vec<u32>, Vec<u32>) = links
            .par_iter()
            .map(|l| {
                // Upstream of a: a link where a is the b side.
                (
                    pick(l.a, l.axis, true, l.lo, l.hi),
                    pick(l.b, l.axis, false, l.lo, l.hi),
                )
            })
            .unzip();
        self.link_prev = prev;
        self.link_next = next;
        self.rebuild_boundary();
        self.topology_rev += 1;
    }

    /// Re-read the terrain under the XZ rectangle `[min, max]` (plus one
    /// column of margin) after it was dug or built on. Water is kept.
    pub fn resync_area(&mut self, terrain: &dyn WaterTerrain, min: Vec2, max: Vec2) {
        let mut changed = vec![false; self.region.column_count()];
        mark_columns(&self.region, min, max, &mut changed);
        self.resync_columns(terrain, &changed);
    }

    /// Re-read the terrain under the columns flagged in `changed` (after
    /// digging or building), keeping every drop of water: water now inside
    /// rock is pushed up into the nearest open space above it.
    pub fn resync_columns(&mut self, terrain: &dyn WaterTerrain, changed: &[bool]) {
        let ncols = self.region.column_count();
        assert_eq!(changed.len(), ncols, "one flag per column");
        if !changed.iter().any(|&c| c) {
            return;
        }
        let region = self.region;
        let settings = self.settings;
        let new_spans: Vec<Option<Vec<Span>>> = (0..ncols)
            .into_par_iter()
            .map(|c| changed[c].then(|| column_spans(&region, &settings, terrain, c)))
            .collect();
        let mut stranded = 0.0f64; // water of columns that became solid
        let mut stranded_at: Vec<usize> = Vec::new();
        let mut columns: Vec<Vec<(Span, f32)>> = Vec::with_capacity(ncols);
        for (c, spans) in new_spans.into_iter().enumerate() {
            let (s0, s1) = (self.col_start[c] as usize, self.col_start[c + 1] as usize);
            let old: Vec<(Span, f32)> = (s0..s1)
                .map(|i| {
                    (
                        Span {
                            floor: self.floor[i],
                            ceil: self.ceil[i],
                        },
                        self.depth[i],
                    )
                })
                .collect();
            match spans {
                None => columns.push(old),
                Some(spans) => {
                    let (cells, lost) = remap_column(&old, &spans);
                    if lost > 0.0 {
                        stranded += lost as f64;
                        stranded_at.push(c);
                    }
                    columns.push(cells);
                }
            }
        }
        let old = OldTopology {
            link_group: std::mem::take(&mut self.link_group),
            q: std::mem::take(&mut self.q),
        };
        self.set_cells(columns);
        self.rebuild_links(Some(&old), changed);
        if stranded > 0.0 {
            self.place_stranded(&stranded_at, stranded as f32);
        }
        self.time_step_reset();
    }

    /// Put water from columns that turned fully solid into the nearest
    /// column that still has an open cell.
    fn place_stranded(&mut self, cols: &[usize], depth_total: f32) {
        let r = self.region;
        let (nx, nz) = (r.cells_x as i32, r.cells_z as i32);
        let per = depth_total / cols.len().max(1) as f32;
        for &c in cols {
            let (cx, cz) = ((c as i32) % nx, (c as i32) / nx);
            'search: for radius in 1..nx.max(nz) {
                for dz in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx.abs() != radius && dz.abs() != radius {
                            continue;
                        }
                        let (x, z) = (cx + dx, cz + dz);
                        if x < 0 || z < 0 || x >= nx || z >= nz {
                            continue;
                        }
                        let n = (z * nx + x) as usize;
                        let end = self.col_start[n + 1];
                        if end > self.col_start[n] {
                            self.depth[(end - 1) as usize] += per;
                            break 'search;
                        }
                    }
                }
            }
        }
    }
}

struct OldTopology {
    link_group: Vec<u32>,
    q: Vec<f32>,
}

#[inline]
pub(crate) fn head_of(floor: f32, ceil: f32, depth: f32, stiffness: f32) -> f32 {
    let cap = ceil - floor;
    if depth <= cap {
        floor + depth
    } else {
        ceil + (depth - cap) * stiffness
    }
}

/// Open spans of column `c`, without slivers thinner than `min_gap`.
fn column_spans(
    region: &WaterRegion,
    settings: &WaterSettings,
    terrain: &dyn WaterTerrain,
    c: usize,
) -> Vec<Span> {
    let (x, z) = (c as u32 % region.cells_x, c as u32 / region.cells_x);
    let centre = region.column_centre(x, z);
    let mut spans = Vec::new();
    terrain.column(centre.x, centre.y, region.y_min, region.y_max, &mut spans);
    spans.retain(|s| s.height() >= settings.min_gap);
    for s in &mut spans {
        s.floor = s.floor.max(region.y_min);
    }
    spans
}

/// Move a column's water from its old cells into new spans. Water keeps its
/// height where it is still open; water now inside rock rises to the next
/// open span above. Returns the new cells and any depth left with nowhere
/// to go (the column became solid).
fn remap_column(old: &[(Span, f32)], spans: &[Span]) -> (Vec<(Span, f32)>, f32) {
    let mut out: Vec<(Span, f32)> = spans.iter().map(|s| (*s, 0.0)).collect();
    let mut pending = 0.0f32;
    for &(span, depth) in old {
        if depth <= 0.0 {
            continue;
        }
        let cap = span.height();
        let free = depth.min(cap);
        let extra = depth - free; // pressurised surplus
        let (w0, w1) = (span.floor, span.floor + free);
        let mut placed = 0.0;
        for (new, d) in out.iter_mut() {
            let lo = w0.max(new.floor);
            let hi = w1.min(new.ceil);
            if hi > lo {
                *d += hi - lo;
                placed += hi - lo;
            }
        }
        let lost = free - placed + extra;
        if lost > 0.0 {
            // The first span reaching above the old water's bottom, else the top.
            if let Some((_, d)) = out.iter_mut().find(|(s, _)| s.ceil > w0) {
                *d += lost;
            } else if let Some((_, d)) = out.last_mut() {
                *d += lost;
            } else {
                pending += lost;
            }
        }
    }
    (out, pending)
}

/// Flag the columns under an XZ rectangle, with one column of margin.
pub(crate) fn mark_columns(region: &WaterRegion, min: Vec2, max: Vec2, changed: &mut [bool]) {
    let lo = ((min - region.min_xz) / region.cell_size).floor() - Vec2::ONE;
    let hi = ((max - region.min_xz) / region.cell_size).ceil() + Vec2::ONE;
    let (nx, nz) = (region.cells_x as i64, region.cells_z as i64);
    for z in (lo.y as i64).clamp(0, nz)..(hi.y as i64).clamp(0, nz) {
        for x in (lo.x as i64).clamp(0, nx)..(hi.x as i64).clamp(0, nx) {
            changed[(z * nx + x) as usize] = true;
        }
    }
}
