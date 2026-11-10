//! Navigation: where characters can walk, paths between places, and agents
//! that follow them.
//!
//! [`World::build_nav`] samples an area into a layered grid of walkable
//! surfaces: the ground (voxel terrain, height terrain or flat floor) and the
//! tops of fixed bodies (floors, steps, bridges, roofs). Several surfaces can
//! sit above each other, so a bridge and the path under it both work. A
//! surface is walkable when it is flat enough, has head room, and is at least
//! the agent radius away from walls and drops. Neighbouring cells connect
//! when the step between them is small enough to walk.
//!
//! [`World::find_path`] runs A* over that grid and straightens the result.
//! [`World::add_nav_agent`] and [`World::set_nav_goal`] make an entity walk
//! there by itself, steering around other agents and playing its walk and run
//! clips when it is animated.
//!
//! Voxel digging and building rebuild the affected part automatically. After
//! adding or removing fixed bodies, call [`World::rebuild_nav_area`].

use super::{EntityId, World};
use crate::error::{EngineError, EngineResult};
use crate::mesh::Mesh;
use crate::physics::PhysicsGround;
use glam::Vec3;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// How [`World::build_nav`] decides what is walkable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavSettings {
    /// Grid cell size in metres. Smaller finds narrower gaps but costs more.
    pub cell_m: f32,
    /// Agents keep this far from walls and drops.
    pub agent_radius: f32,
    /// Head room an agent needs.
    pub agent_height: f32,
    /// Highest step an agent walks up or down (a stair step, a kerb).
    pub max_step: f32,
    /// Steepest walkable slope.
    pub max_slope_deg: f32,
}

impl Default for NavSettings {
    fn default() -> Self {
        Self {
            cell_m: 0.3,
            agent_radius: 0.35,
            agent_height: 1.8,
            max_step: 0.45,
            max_slope_deg: 40.0,
        }
    }
}

impl NavSettings {
    fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str, v: f32| {
            Err(EngineError::InvalidValue(format!(
                "nav {what} out of range: {v}"
            )))
        };
        if !(self.cell_m.is_finite() && (0.05..=5.0).contains(&self.cell_m)) {
            return bad("cell_m (0.05..=5)", self.cell_m);
        }
        if !(self.agent_radius.is_finite() && (0.0..=10.0).contains(&self.agent_radius)) {
            return bad("agent_radius (0..=10)", self.agent_radius);
        }
        if !(self.agent_height.is_finite() && (0.0..=50.0).contains(&self.agent_height)) {
            return bad("agent_height (0..=50)", self.agent_height);
        }
        if !(self.max_step.is_finite() && (0.0..=10.0).contains(&self.max_step)) {
            return bad("max_step (0..=10)", self.max_step);
        }
        if !(self.max_slope_deg.is_finite() && (0.0..=89.0).contains(&self.max_slope_deg)) {
            return bad("max_slope_deg (0..=89)", self.max_slope_deg);
        }
        Ok(())
    }
}

/// How a navigation agent moves. See [`World::add_nav_agent`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavAgent {
    /// Cruising speed in m/s (1.4 walks, 3+ runs).
    pub speed: f32,
    /// How fast it speeds up and slows down, m/s².
    pub acceleration: f32,
    /// Personal space: agents push apart when closer than their radii.
    pub radius: f32,
    /// Counts as arrived this close to the goal.
    pub arrive_distance: f32,
    /// Turning speed in degrees per second.
    pub turn_speed_deg: f32,
}

impl Default for NavAgent {
    fn default() -> Self {
        Self {
            speed: 1.4,
            acceleration: 6.0,
            radius: 0.35,
            arrive_distance: 0.25,
            turn_speed_deg: 540.0,
        }
    }
}

impl NavAgent {
    /// An agent cruising at `speed` m/s, everything else default.
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            ..Self::default()
        }
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    fn validate(&self) -> EngineResult<()> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && v >= lo && v <= hi;
        if !(ok(self.speed, 0.0, 100.0)
            && ok(self.acceleration, 0.01, 1000.0)
            && ok(self.radius, 0.0, 10.0)
            && ok(self.arrive_distance, 0.01, 10.0)
            && ok(self.turn_speed_deg, 1.0, 100000.0))
        {
            return Err(EngineError::InvalidValue(format!(
                "nav agent settings out of range: {self:?}"
            )));
        }
        Ok(())
    }
}

/// What a navigation agent is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavStatus {
    /// No goal.
    Idle,
    /// On its way.
    Moving,
    /// Reached its last goal.
    Arrived,
    /// No walkable way to the goal (or no navigation built).
    Unreachable,
}

/// Size of the built navigation, from [`World::nav_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NavStats {
    pub columns: usize,
    /// Surfaces found (ground, floors, steps), walkable or not.
    pub surfaces: usize,
    /// Surfaces an agent can stand on.
    pub walkable: usize,
    /// Time the last build or rebuild took.
    pub build_ms: f32,
}

const NONE: u8 = u8::MAX;
/// Neighbour directions: four sides, then four diagonals.
const DIRS: [(i32, i32); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];
/// Each diagonal as its two sides.
const DIAG: [(usize, usize); 4] = [(0, 2), (0, 3), (1, 2), (1, 3)];
const MAX_LAYERS: usize = 32;
/// A* gives up after visiting this many cells.
const MAX_VISITS: usize = 400_000;

fn dir_index(di: i32, dj: i32) -> Option<usize> {
    DIRS.iter().position(|&d| d == (di, dj))
}

#[derive(Clone, Copy, Debug)]
struct Span {
    y: f32,
    /// Flat enough with head room.
    open: bool,
    /// Open and far enough from edges.
    walk: bool,
    links: [u8; 8],
}

/// A surface in the grid: column index and layer.
type Cell = (u32, u8);

#[derive(Clone, Copy)]
struct Rect {
    i0: i32,
    j0: i32,
    i1: i32,
    j1: i32,
}

impl Rect {
    fn grow(self, n: i32, nx: usize, nz: usize) -> Rect {
        Rect {
            i0: (self.i0 - n).max(0),
            j0: (self.j0 - n).max(0),
            i1: (self.i1 + n).min(nx as i32),
            j1: (self.j1 + n).min(nz as i32),
        }
    }
    fn contains(&self, i: i32, j: i32) -> bool {
        i >= self.i0 && i < self.i1 && j >= self.j0 && j < self.j1
    }
}

pub(crate) struct NavGrid {
    settings: NavSettings,
    min: Vec3,
    max: Vec3,
    nx: usize,
    nz: usize,
    cols: Vec<Vec<Span>>,
}

/// Navigation state kept by the world.
#[derive(Default)]
pub(crate) struct NavState {
    grid: Option<NavGrid>,
    agents: HashMap<EntityId, AgentState>,
    terrain_rev: u64,
    /// Raised on every build, so agents replan.
    rev: u64,
    stats: NavStats,
}

struct AgentState {
    cfg: NavAgent,
    goal: Option<Vec3>,
    path: Vec<Vec3>,
    next: usize,
    status: NavStatus,
    speed: f32,
    yaw: f32,
    scale: f32,
    planned_rev: u64,
    stuck: f32,
}

/// Reads what is solid: fixed bodies and the ground.
struct Sampler<'a> {
    physics: &'a crate::physics::Physics,
    voxels: Option<&'a crate::voxel_terrain::VoxelTerrain>,
    field: Option<&'a crate::proc_terrain::HeightField>,
    ground: PhysicsGround,
}

impl Sampler<'_> {
    fn voxels(&self) -> Option<&crate::voxel_terrain::VoxelTerrain> {
        match self.ground {
            PhysicsGround::Auto => self
                .voxels
                .filter(|v| v.volume().chunk_keys().next().is_some()),
            _ => None,
        }
    }

    /// Inside the ground (not fixed bodies).
    fn in_ground(&self, p: Vec3) -> bool {
        match self.ground {
            PhysicsGround::None => false,
            PhysicsGround::Plane(y) => p.y < y - 1e-3,
            PhysicsGround::Auto => {
                if let Some(v) = self.voxels() {
                    v.is_solid(p)
                } else if let Some(f) = self.field {
                    p.y < f.height_at(p.x, p.z) - 1e-3
                } else {
                    p.y < -1e-3
                }
            }
        }
    }

    /// First ground surface straight below `p`: height and normal.
    fn ground_down(&self, p: Vec3, len: f32) -> Option<(f32, Vec3)> {
        let plane = |y: f32| (y <= p.y && p.y - y <= len).then_some((y, Vec3::Y));
        match self.ground {
            PhysicsGround::None => None,
            PhysicsGround::Plane(y) => plane(y),
            PhysicsGround::Auto => {
                if let Some(v) = self.voxels() {
                    v.raycast(p, Vec3::NEG_Y, len)
                        .map(|q| (q.y, v.normal_at(q)))
                } else if let Some(f) = self.field {
                    let h = f.height_at(p.x, p.z);
                    if h > p.y || p.y - h > len {
                        return None;
                    }
                    let e = 0.25;
                    let n = Vec3::new(
                        f.height_at(p.x - e, p.z) - f.height_at(p.x + e, p.z),
                        2.0 * e,
                        f.height_at(p.x, p.z - e) - f.height_at(p.x, p.z + e),
                    )
                    .normalize();
                    Some((h, n))
                } else {
                    plane(0.0)
                }
            }
        }
    }

    /// First surface straight below `p` (ground or fixed body).
    fn down(&self, p: Vec3, len: f32) -> Option<(f32, Vec3)> {
        let body = self
            .physics
            .cast_ray_fixed(p, Vec3::NEG_Y, len, false)
            .map(|h| (h.point.y, h.normal));
        let ground = self.ground_down(p, len);
        match (body, ground) {
            (Some(b), Some(g)) => Some(if g.0 > b.0 { g } else { b }),
            (b, g) => b.or(g),
        }
    }

    /// Every surface in one column, top down.
    fn column(&self, x: f32, z: f32, lo: f32, hi: f32, s: &NavSettings) -> Vec<Span> {
        let min_ny = s.max_slope_deg.to_radians().cos();
        let mut out = Vec::new();
        let mut y = hi;
        let mut ceiling = f32::INFINITY;
        for _ in 0..(MAX_LAYERS * 3) {
            if y < lo || out.len() >= MAX_LAYERS {
                break;
            }
            let p = Vec3::new(x, y, z);
            if self.in_ground(p) {
                // Only voxel ground has caves to look for below.
                let Some(v) = self.voxels() else { break };
                let step = (v.volume().voxel_size() * 0.5).max(0.02);
                while y >= lo && v.is_solid(Vec3::new(x, y, z)) {
                    y -= step;
                }
                ceiling = y + step;
                continue;
            }
            if let Some(under) = self.physics.fixed_underside(p, lo) {
                // Inside a wall or block: carry on below it.
                ceiling = under;
                y = under - 0.05;
                continue;
            }
            let Some((hit, n)) = self.down(p, y - lo) else {
                break;
            };
            if n.y > 0.3 && hit >= lo {
                let clear = ceiling - hit;
                out.push(Span {
                    y: hit,
                    open: n.y >= min_ny - 1e-4 && clear >= s.agent_height,
                    walk: false,
                    links: [NONE; 8],
                });
            }
            ceiling = hit;
            y = hit - 0.05;
        }
        out
    }
}

impl NavGrid {
    fn cell(&self) -> f32 {
        self.settings.cell_m
    }

    fn idx(&self, i: i32, j: i32) -> Option<usize> {
        (i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.nz)
            .then(|| j as usize * self.nx + i as usize)
    }

    fn ij(&self, col: u32) -> (i32, i32) {
        (
            (col as usize % self.nx) as i32,
            (col as usize / self.nx) as i32,
        )
    }

    fn col_of(&self, x: f32, z: f32) -> (i32, i32) {
        (
            ((x - self.min.x) / self.cell()).floor() as i32,
            ((z - self.min.z) / self.cell()).floor() as i32,
        )
    }

    fn center(&self, i: i32, j: i32) -> (f32, f32) {
        (
            self.min.x + (i as f32 + 0.5) * self.cell(),
            self.min.z + (j as f32 + 0.5) * self.cell(),
        )
    }

    fn span(&self, c: Cell) -> &Span {
        &self.cols[c.0 as usize][c.1 as usize]
    }

    fn pos(&self, c: Cell) -> Vec3 {
        let (i, j) = self.ij(c.0);
        let (x, z) = self.center(i, j);
        Vec3::new(x, self.span(c).y, z)
    }

    /// Walkable neighbour in direction `k`.
    fn step(&self, c: Cell, k: usize) -> Option<Cell> {
        let l = self.span(c).links[k];
        if l == NONE {
            return None;
        }
        let (i, j) = self.ij(c.0);
        let n = self.idx(i + DIRS[k].0, j + DIRS[k].1)? as u32;
        let t = (n, l);
        self.span(t).walk.then_some(t)
    }

    /// Walkable surface in the column under `p`, within reach of its height.
    fn at(&self, p: Vec3, up: f32, down: f32) -> Option<Cell> {
        let (i, j) = self.col_of(p.x, p.z);
        let col = self.idx(i, j)?;
        self.cols[col]
            .iter()
            .enumerate()
            .filter(|(_, s)| s.walk && s.y <= p.y + up && s.y >= p.y - down)
            .min_by(|a, b| (a.1.y - p.y).abs().total_cmp(&(b.1.y - p.y).abs()))
            .map(|(k, _)| (col as u32, k as u8))
    }

    /// Nearest walkable surface to `p`, searching outward.
    fn nearest(&self, p: Vec3, max_cells: i32) -> Option<Cell> {
        let reach = self.settings.agent_height.max(1.0);
        if let Some(c) = self.at(p, self.settings.max_step + 0.1, reach) {
            return Some(c);
        }
        let (ci, cj) = self.col_of(p.x, p.z);
        let mut best: Option<(f32, Cell)> = None;
        for r in 1..=max_cells {
            for j in (cj - r)..=(cj + r) {
                for i in (ci - r)..=(ci + r) {
                    if (i - ci).abs() != r && (j - cj).abs() != r {
                        continue;
                    }
                    let Some(col) = self.idx(i, j) else { continue };
                    for (k, s) in self.cols[col].iter().enumerate() {
                        if !s.walk || (s.y - p.y).abs() > reach + 0.5 {
                            continue;
                        }
                        let c = (col as u32, k as u8);
                        let d = (self.pos(c) - p).length_squared();
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, c));
                        }
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best.map(|b| b.1)
    }

    /// Connect open surfaces to their neighbours. Sides for `sides`,
    /// diagonals (which use the sides) for `diags`.
    fn link(&mut self, sides: Rect, diags: Rect) {
        let max_step = self.settings.max_step;
        for j in sides.j0..sides.j1 {
            for i in sides.i0..sides.i1 {
                let col = self.idx(i, j).unwrap();
                for s in 0..self.cols[col].len() {
                    let me = self.cols[col][s];
                    let mut links = [NONE; 8];
                    if me.open {
                        for (k, link) in links.iter_mut().enumerate().take(4) {
                            let Some(n) = self.idx(i + DIRS[k].0, j + DIRS[k].1) else {
                                continue;
                            };
                            *link = self.cols[n]
                                .iter()
                                .enumerate()
                                .filter(|(_, t)| t.open && (t.y - me.y).abs() <= max_step)
                                .min_by(|a, b| {
                                    (a.1.y - me.y).abs().total_cmp(&(b.1.y - me.y).abs())
                                })
                                .map_or(NONE, |(t, _)| t as u8);
                        }
                    }
                    self.cols[col][s].links = links;
                }
            }
        }
        for j in diags.j0..diags.j1 {
            for i in diags.i0..diags.i1 {
                let col = self.idx(i, j).unwrap();
                for s in 0..self.cols[col].len() {
                    for (d, &(a, b)) in DIAG.iter().enumerate() {
                        let k = 4 + d;
                        let via = |first: usize, second: usize| -> Option<u8> {
                            let l = self.cols[col][s].links[first];
                            if l == NONE {
                                return None;
                            }
                            let n = self.idx(i + DIRS[first].0, j + DIRS[first].1)?;
                            let l2 = self.cols[n][l as usize].links[second];
                            (l2 != NONE).then_some(l2)
                        };
                        let target = match (via(a, b), via(b, a)) {
                            (Some(x), Some(y)) if x == y => x,
                            _ => NONE,
                        };
                        if target != NONE {
                            let n = self.idx(i + DIRS[k].0, j + DIRS[k].1).unwrap();
                            let dy = (self.cols[n][target as usize].y - self.cols[col][s].y).abs();
                            if dy > max_step {
                                self.cols[col][s].links[k] = NONE;
                                continue;
                            }
                        }
                        self.cols[col][s].links[k] = target;
                    }
                }
            }
        }
    }

    /// Erosion reach in cells.
    fn erode_cells(&self) -> i32 {
        (self.settings.agent_radius / self.cell()).ceil() as i32 + 1
    }

    /// Keep agents `agent_radius` away from walls and drops: an open surface
    /// is walkable when its distance to the nearest edge is large enough.
    /// Writes `out`, reading edges up to the erosion reach beyond it.
    fn erode(&mut self, out: Rect) {
        let reach = self.erode_cells();
        let domain = out.grow(reach, self.nx, self.nz);
        let cap = (reach * 10) as u16;
        let w = (domain.i1 - domain.i0) as usize;
        let local = |g: &NavGrid, col: u32| -> usize {
            let (i, j) = g.ij(col);
            (j - domain.j0) as usize * w + (i - domain.i0) as usize
        };
        let h = (domain.j1 - domain.j0) as usize;
        let mut dist: Vec<Vec<u16>> = vec![Vec::new(); w * h];
        let mut heap = BinaryHeap::new();
        for j in domain.j0..domain.j1 {
            for i in domain.i0..domain.i1 {
                let col = self.idx(i, j).unwrap();
                let l = (j - domain.j0) as usize * w + (i - domain.i0) as usize;
                dist[l] = vec![cap; self.cols[col].len()];
                for (s, span) in self.cols[col].iter().enumerate() {
                    if span.open && span.links[..4].contains(&NONE) {
                        dist[l][s] = 0;
                        heap.push(Reverse((0u16, col as u32, s as u8)));
                    }
                }
            }
        }
        while let Some(Reverse((d, col, s))) = heap.pop() {
            if d > dist[local(self, col)][s as usize] {
                continue;
            }
            let (i, j) = self.ij(col);
            for (k, dir) in DIRS.iter().enumerate() {
                let l = self.cols[col as usize][s as usize].links[k];
                if l == NONE {
                    continue;
                }
                let (ni, nj) = (i + dir.0, j + dir.1);
                if !domain.contains(ni, nj) {
                    continue;
                }
                let n = self.idx(ni, nj).unwrap() as u32;
                let nd = d.saturating_add(if k < 4 { 10 } else { 14 }).min(cap);
                let slot = &mut dist[local(self, n)][l as usize];
                if nd < *slot {
                    *slot = nd;
                    heap.push(Reverse((nd, n, l)));
                }
            }
        }
        let need = self.settings.agent_radius - 1e-4;
        let cell = self.cell();
        for j in out.j0..out.j1 {
            for i in out.i0..out.i1 {
                let col = self.idx(i, j).unwrap();
                let l = (j - domain.j0) as usize * w + (i - domain.i0) as usize;
                for (span, &d) in self.cols[col].iter_mut().zip(&dist[l]) {
                    let d = d as f32 / 10.0;
                    span.walk = span.open && (d + 0.5) * cell >= need;
                }
            }
        }
    }

    /// Follow the surface from `from` (standing at `a`) in a straight line
    /// to `b`. Returns the surface reached, or `None` if the line leaves the
    /// walkable area.
    fn walk_line(&self, from: Cell, a: Vec3, b: Vec3) -> Option<Cell> {
        let d = Vec3::new(b.x - a.x, 0.0, b.z - a.z);
        let steps = (d.length() / (self.cell() * 0.25)).ceil().max(1.0) as i32;
        let mut cur = from;
        let (mut ci, mut cj) = self.ij(cur.0);
        for k in 1..=steps {
            let p = a + d * (k as f32 / steps as f32);
            let (pi, pj) = self.col_of(p.x, p.z);
            let mut guard = 0;
            while (ci, cj) != (pi, pj) {
                let di = (pi - ci).signum();
                let dj = (pj - cj).signum();
                cur = self.step(cur, dir_index(di, dj)?)?;
                ci += di;
                cj += dj;
                guard += 1;
                if guard > 8 {
                    return None;
                }
            }
        }
        Some(cur)
    }

    /// A* from `start` to `goal`; cells along the way.
    fn astar(&self, start: Cell, goal: Cell) -> Option<Vec<Cell>> {
        let key = |c: Cell| ((c.0 as u64) << 8) | c.1 as u64;
        let unkey = |k: u64| ((k >> 8) as u32, (k & 0xff) as u8);
        let goal_pos = self.pos(goal);
        let h = |c: Cell| {
            let p = self.pos(c);
            Vec3::new(p.x - goal_pos.x, 0.0, p.z - goal_pos.z).length()
        };
        let mut best: HashMap<u64, (f32, u64)> = HashMap::new();
        let mut heap = BinaryHeap::new();
        best.insert(key(start), (0.0, u64::MAX));
        heap.push(Reverse((h(start).to_bits(), key(start))));
        let cell = self.cell();
        let mut visits = 0;
        while let Some(Reverse((_, k))) = heap.pop() {
            let c = unkey(k);
            if c == goal {
                let mut out = vec![c];
                let mut at = k;
                while let Some(&(_, parent)) = best.get(&at) {
                    if parent == u64::MAX {
                        break;
                    }
                    out.push(unkey(parent));
                    at = parent;
                }
                out.reverse();
                return Some(out);
            }
            visits += 1;
            if visits > MAX_VISITS {
                return None;
            }
            let g = best[&k].0;
            let y = self.span(c).y;
            for d in 0..8 {
                let Some(n) = self.step(c, d) else { continue };
                let flat = if d < 4 {
                    cell
                } else {
                    cell * std::f32::consts::SQRT_2
                };
                let ng = g + flat + (self.span(n).y - y).abs();
                let nk = key(n);
                if best.get(&nk).is_none_or(|b| ng < b.0) {
                    best.insert(nk, (ng, k));
                    heap.push(Reverse(((ng + h(n)).to_bits(), nk)));
                }
            }
        }
        None
    }

    /// Path from `from` to `to` as straightened waypoints (both included).
    fn path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        let reach = (1.5 / self.cell()).ceil() as i32 + 2;
        let start = self.nearest(from, reach)?;
        let goal = self.nearest(to, reach)?;
        let cells = self.astar(start, goal)?;
        let start_pos = Vec3::new(from.x, self.span(start).y, from.z);
        let goal_pos = Vec3::new(to.x, self.span(goal).y, to.z);
        // Points to pass: the start, every cell, the goal. The ends are the
        // real positions when they lie on their cell, else the cell centre.
        let on = |c: Cell, p: Vec3| {
            let (i, j) = self.ij(c.0);
            self.col_of(p.x, p.z) == (i, j)
        };
        let mut pts: Vec<(Cell, Vec3)> = cells.iter().map(|&c| (c, self.pos(c))).collect();
        if on(start, start_pos) {
            pts[0].1 = start_pos;
        }
        let last = pts.len() - 1;
        if on(goal, goal_pos) {
            pts[last].1 = goal_pos;
        }
        // Straighten: from each corner, go as far as a straight walkable
        // line reaches.
        let mut out = vec![pts[0].1];
        let mut at = 0;
        while at < last {
            let mut far = at + 1;
            let mut k = at + 2;
            while k <= last && self.walk_line(pts[at].0, pts[at].1, pts[k].1) == Some(pts[k].0) {
                far = k;
                k += 1;
            }
            out.push(pts[far].1);
            at = far;
        }
        Some(out)
    }

    fn rebuild(&mut self, sampler: &Sampler, rect: Rect) {
        let (lo, hi) = (self.min.y, self.max.y);
        let s = self.settings;
        let cols: Vec<(usize, Vec<Span>)> = (rect.j0..rect.j1)
            .flat_map(|j| (rect.i0..rect.i1).map(move |i| (i, j)))
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(i, j)| {
                let (x, z) = self.center(i, j);
                (self.idx(i, j).unwrap(), sampler.column(x, z, lo, hi, &s))
            })
            .collect();
        for (col, spans) in cols {
            self.cols[col] = spans;
        }
        let sides = rect.grow(1, self.nx, self.nz);
        let diags = rect.grow(2, self.nx, self.nz);
        self.link(sides, diags);
        let reach = self.erode_cells();
        self.erode(diags.grow(reach, self.nx, self.nz));
    }

    fn whole(&self) -> Rect {
        Rect {
            i0: 0,
            j0: 0,
            i1: self.nx as i32,
            j1: self.nz as i32,
        }
    }

    fn rect_of(&self, min: Vec3, max: Vec3) -> Option<Rect> {
        let (i0, j0) = self.col_of(min.x, min.z);
        let (i1, j1) = self.col_of(max.x, max.z);
        let r = Rect {
            i0: i0.max(0),
            j0: j0.max(0),
            i1: (i1 + 1).min(self.nx as i32),
            j1: (j1 + 1).min(self.nz as i32),
        };
        (r.i0 < r.i1 && r.j0 < r.j1).then_some(r)
    }

    fn stats(&self) -> NavStats {
        let spans = self.cols.iter().flatten();
        NavStats {
            columns: self.cols.len(),
            surfaces: self.cols.iter().map(Vec::len).sum(),
            walkable: spans.filter(|s| s.walk).count(),
            build_ms: 0.0,
        }
    }
}

impl std::fmt::Debug for NavState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NavState")
            .field("built", &self.grid.is_some())
            .field("agents", &self.agents.len())
            .field("stats", &self.stats)
            .finish()
    }
}

fn wrap_deg(a: f32) -> f32 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

impl World {
    fn nav_sampler(&self) -> Sampler<'_> {
        Sampler {
            physics: &self.physics,
            voxels: self.voxel_terrain.as_ref(),
            field: self.height_field.as_ref(),
            ground: self.physics.settings().ground,
        }
    }

    /// Work out where agents can walk inside the box `min`..`max` (the
    /// box's height range limits which floors count). Replaces any earlier
    /// navigation. Takes a moment for big areas: about 100 x 100 m at the
    /// default 0.3 m cells is 110k columns.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World) -> EngineResult<()> {
    /// world.build_nav(Vec3::new(-50.0, -5.0, -50.0), Vec3::new(50.0, 20.0, 50.0), NavSettings::default())?;
    /// if let Some(path) = world.find_path(Vec3::new(0.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 5.0)) {
    ///     // walk along `path`
    /// }
    /// # Ok(()) }
    /// ```
    pub fn build_nav(
        &mut self,
        min: Vec3,
        max: Vec3,
        settings: NavSettings,
    ) -> EngineResult<NavStats> {
        settings.validate()?;
        if !min.is_finite() || !max.is_finite() || min.cmpge(max).any() {
            return Err(EngineError::InvalidValue(format!(
                "nav box must have min < max, got {min} .. {max}"
            )));
        }
        let nx = ((max.x - min.x) / settings.cell_m).ceil() as usize;
        let nz = ((max.z - min.z) / settings.cell_m).ceil() as usize;
        if nx * nz > 16_000_000 {
            return Err(EngineError::ResourceLimit(format!(
                "nav grid of {nx} x {nz} cells is too big; use a larger cell_m or a smaller box"
            )));
        }
        let t = std::time::Instant::now();
        let mut grid = NavGrid {
            settings,
            min,
            max,
            nx,
            nz,
            cols: vec![Vec::new(); nx * nz],
        };
        let whole = grid.whole();
        self.physics.refresh_queries();
        grid.rebuild(&self.nav_sampler(), whole);
        let mut stats = grid.stats();
        stats.build_ms = t.elapsed().as_secs_f32() * 1000.0;
        self.nav.grid = Some(grid);
        self.nav.terrain_rev = self.voxel_terrain.as_ref().map_or(0, |v| v.volume().rev());
        self.nav.rev += 1;
        self.nav.stats = stats;
        Ok(stats)
    }

    /// Look again at part of the navigation, after adding, moving or
    /// removing fixed bodies there. Voxel edits do this by themselves.
    pub fn rebuild_nav_area(&mut self, min: Vec3, max: Vec3) {
        let Some(mut grid) = self.nav.grid.take() else {
            return;
        };
        let t = std::time::Instant::now();
        if let Some(rect) = grid.rect_of(min, max) {
            self.physics.refresh_queries();
            grid.rebuild(&self.nav_sampler(), rect);
            self.nav.rev += 1;
        }
        let mut stats = grid.stats();
        stats.build_ms = t.elapsed().as_secs_f32() * 1000.0;
        self.nav.stats = stats;
        self.nav.grid = Some(grid);
    }

    /// Forget the navigation (agents stop).
    pub fn clear_nav(&mut self) {
        self.nav.grid = None;
        self.nav.rev += 1;
        self.nav.stats = NavStats::default();
    }

    /// Surfaces in the column under `p`: (height, open, walkable). For tests.
    #[cfg(test)]
    pub(crate) fn nav_column(&self, p: Vec3) -> Vec<(f32, bool, bool)> {
        let Some(g) = self.nav.grid.as_ref() else {
            return Vec::new();
        };
        let (i, j) = g.col_of(p.x, p.z);
        g.idx(i, j).map_or(Vec::new(), |c| {
            g.cols[c].iter().map(|s| (s.y, s.open, s.walk)).collect()
        })
    }

    pub fn has_nav(&self) -> bool {
        self.nav.grid.is_some()
    }

    pub fn nav_settings(&self) -> Option<NavSettings> {
        self.nav.grid.as_ref().map(|g| g.settings)
    }

    pub fn nav_stats(&self) -> NavStats {
        self.nav.stats
    }

    /// A walking route from `from` to `to`: straight segments over walkable
    /// ground, starting at `from` and ending at `to` (both moved onto the
    /// walkable surface). `None` when there is no way or no navigation.
    pub fn find_path(&self, from: impl Into<Vec3>, to: impl Into<Vec3>) -> Option<Vec<Vec3>> {
        let (from, to) = (from.into(), to.into());
        if !from.is_finite() || !to.is_finite() {
            return None;
        }
        self.nav.grid.as_ref()?.path(from, to)
    }

    /// The walkable spot closest to `p` (within a few metres), on the
    /// surface.
    pub fn nearest_walkable(&self, p: impl Into<Vec3>) -> Option<Vec3> {
        let p = p.into();
        let g = self.nav.grid.as_ref()?;
        let c = g.nearest(p, (3.0 / g.cell()).ceil() as i32)?;
        let (i, j) = g.ij(c.0);
        if g.col_of(p.x, p.z) == (i, j) {
            Some(Vec3::new(p.x, g.span(c).y, p.z))
        } else {
            Some(g.pos(c))
        }
    }

    /// Whether an agent can stand at `p` (a walkable surface within a step
    /// of its height).
    pub fn is_walkable(&self, p: impl Into<Vec3>) -> bool {
        let p = p.into();
        self.nav
            .grid
            .as_ref()
            .is_some_and(|g| g.at(p, g.settings.max_step, g.settings.max_step).is_some())
    }

    /// An overlay of the walkable area (green tiles a little above the
    /// surface; upper floors blue and yellow) to spawn while tuning [`NavSettings`].
    pub fn nav_debug_mesh(&self) -> Option<Mesh> {
        let g = self.nav.grid.as_ref()?;
        let mut mesh = Mesh::new();
        let half = g.cell() * 0.45;
        let colors = [
            crate::color::rgb(70, 220, 110),
            crate::color::rgb(70, 170, 240),
            crate::color::rgb(240, 200, 70),
        ];
        for col in 0..g.cols.len() {
            for (s, span) in g.cols[col].iter().enumerate() {
                if !span.walk {
                    continue;
                }
                let p = g.pos((col as u32, s as u8)) + Vec3::Y * 0.04;
                let color = colors[s.min(colors.len() - 1)];
                let mut ids = Vec::with_capacity(4);
                for (dx, dz) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)] {
                    let id = mesh
                        .add_point(p + Vec3::new(dx * half, 0.0, dz * half))
                        .ok()?;
                    mesh.set_point_color(id, color).ok()?;
                    ids.push(id);
                }
                mesh.add_quad(ids[0], ids[1], ids[2], ids[3]).ok()?;
            }
        }
        Some(mesh)
    }

    /// Let an entity walk by itself: give it goals with
    /// [`World::set_nav_goal`]. It follows the walkable surface, turns to
    /// face where it goes, keeps apart from other agents, and when it is an
    /// animated character with an [`AnimationProfile`](crate::anim::AnimationProfile)
    /// its idle, walk and run clips follow its speed. Use root entities
    /// (not children).
    pub fn add_nav_agent(&mut self, id: EntityId, agent: NavAgent) -> EngineResult<()> {
        agent.validate()?;
        let m = self.matrix_of(id).ok_or(EngineError::UnknownEntity)?;
        let fwd = m.z_axis.truncate();
        let state = AgentState {
            cfg: agent,
            goal: None,
            path: Vec::new(),
            next: 0,
            status: NavStatus::Idle,
            speed: 0.0,
            yaw: fwd.x.atan2(fwd.z).to_degrees(),
            scale: m.x_axis.truncate().length().max(1e-4),
            planned_rev: 0,
            stuck: 0.0,
        };
        self.nav.agents.insert(id, state);
        Ok(())
    }

    /// Change an agent's speed and other settings.
    pub fn set_nav_agent(&mut self, id: EntityId, agent: NavAgent) -> EngineResult<()> {
        agent.validate()?;
        let a = self
            .nav
            .agents
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        a.cfg = agent;
        Ok(())
    }

    /// Stop treating an entity as an agent (it stays where it is).
    pub fn remove_nav_agent(&mut self, id: EntityId) {
        if self.nav.agents.remove(&id).is_some() {
            let _ = self.set_locomotion(id, crate::anim::Locomotion::Idle);
        }
    }

    /// Send an agent somewhere (`None` stops it). Its route is planned at
    /// once; [`World::nav_status`] says whether there is one.
    pub fn set_nav_goal(&mut self, id: EntityId, goal: Option<Vec3>) -> EngineResult<()> {
        if goal.is_some_and(|g| !g.is_finite()) {
            return Err(EngineError::InvalidValue("nav goal must be finite".into()));
        }
        let pos = self
            .matrix_of(id)
            .ok_or(EngineError::UnknownEntity)?
            .w_axis
            .truncate();
        let path = goal.and_then(|g| self.find_path(pos, g));
        let rev = self.nav.rev;
        let a = self
            .nav
            .agents
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        a.goal = goal;
        a.next = 1;
        a.stuck = 0.0;
        a.planned_rev = rev;
        a.status = match (goal, &path) {
            (None, _) => NavStatus::Idle,
            (Some(_), Some(_)) => NavStatus::Moving,
            (Some(_), None) => NavStatus::Unreachable,
        };
        a.path = path.unwrap_or_default();
        Ok(())
    }

    pub fn nav_status(&self, id: EntityId) -> Option<NavStatus> {
        self.nav.agents.get(&id).map(|a| a.status)
    }

    /// The route an agent is following (empty when it has none).
    pub fn nav_path(&self, id: EntityId) -> &[Vec3] {
        self.nav.agents.get(&id).map_or(&[], |a| &a.path)
    }

    /// Ids of all agents.
    pub fn nav_agents(&self) -> Vec<EntityId> {
        let mut ids: Vec<EntityId> = self.nav.agents.keys().copied().collect();
        ids.sort_by_key(|id| id.raw());
        ids
    }

    /// Once a frame before animation: rebuild after voxel edits, move agents.
    pub(crate) fn tick_nav(&mut self, dt: f32) {
        self.refresh_nav_after_digging();
        if self.nav.agents.is_empty() || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(0.1);
        let mut ids = self.nav_agents();
        ids.retain(|id| {
            let alive = self.matrix_of(*id).is_some();
            if !alive {
                self.nav.agents.remove(id);
            }
            alive
        });
        // Knocked-down characters don't walk until they are up again.
        ids.retain(|id| !self.is_ragdoll(*id));
        // Replan when the walkable area changed or an agent got stuck.
        for &id in &ids {
            let a = &self.nav.agents[&id];
            if a.status == NavStatus::Moving && (a.planned_rev != self.nav.rev || a.stuck > 0.6)
                || a.status == NavStatus::Unreachable && a.planned_rev != self.nav.rev
            {
                let goal = a.goal;
                let _ = self.set_nav_goal(id, goal);
            }
        }
        let positions: Vec<(EntityId, Vec3, f32)> = ids
            .iter()
            .map(|id| {
                let p = self.matrix_of(*id).unwrap().w_axis.truncate();
                (*id, p, self.nav.agents[id].cfg.radius)
            })
            .collect();
        // Neighbours by 2 m buckets, so crowds of hundreds stay cheap.
        let bucket_m = positions.iter().map(|p| p.2 * 2.0).fold(2.0_f32, f32::max);
        let bucket = |p: Vec3| {
            (
                (p.x / bucket_m).floor() as i32,
                (p.z / bucket_m).floor() as i32,
            )
        };
        let mut buckets: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (n, p) in positions.iter().enumerate() {
            buckets.entry(bucket(p.1)).or_default().push(n);
        }
        let mut moves = Vec::with_capacity(ids.len());
        for (n, &(id, pos, radius)) in positions.iter().enumerate() {
            let a = &self.nav.agents[&id];
            let cfg = a.cfg;
            // Where to head.
            let mut want = Vec3::ZERO;
            let mut remaining = 0.0;
            let mut next = a.next;
            if a.status == NavStatus::Moving && next < a.path.len() {
                let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
                // Skip waypoints already reached.
                while next + 1 < a.path.len() && flat(a.path[next] - pos).length() < 0.2 {
                    next += 1;
                }
                let to = flat(a.path[next] - pos);
                remaining = to.length()
                    + a.path[next..]
                        .windows(2)
                        .map(|w| flat(w[1] - w[0]).length())
                        .sum::<f32>();
                want = to.normalize_or_zero();
            }
            let arrived = a.status == NavStatus::Moving && remaining <= cfg.arrive_distance;
            // Speed: slow down near the goal and for sharp turns.
            let mut yaw = a.yaw;
            let mut target_speed = 0.0;
            if want != Vec3::ZERO && !arrived {
                let goal_yaw = want.x.atan2(want.z).to_degrees();
                let diff = wrap_deg(goal_yaw - yaw);
                let turn = cfg.turn_speed_deg * dt;
                yaw = wrap_deg(yaw + diff.clamp(-turn, turn));
                let ease = (remaining / 1.0).clamp(0.3, 1.0);
                let facing = if diff.abs() > 70.0 { 0.35 } else { 1.0 };
                target_speed = cfg.speed * ease * facing;
            }
            let dv = cfg.acceleration * dt;
            let speed = if arrived {
                0.0
            } else {
                a.speed + (target_speed - a.speed).clamp(-dv, dv)
            };
            let heading = Vec3::new(yaw.to_radians().sin(), 0.0, yaw.to_radians().cos());
            // Move along the heading, nudged towards the path direction so
            // turning circles stay tight.
            let dir = (heading * 0.6 + want * 0.4).normalize_or(heading);
            let mut step = dir * speed * dt;
            // Keep apart from other agents.
            let (bi, bj) = bucket(pos);
            let near = (-1..=1)
                .flat_map(|di| (-1..=1).map(move |dj| (bi + di, bj + dj)))
                .filter_map(|b| buckets.get(&b))
                .flatten()
                .copied();
            for m in near {
                let (_, other, r2) = positions[m];
                if m == n {
                    continue;
                }
                let d = Vec3::new(pos.x - other.x, 0.0, pos.z - other.z);
                let gap = radius + r2 - d.length();
                if gap > 0.0 && (other.y - pos.y).abs() < 1.5 {
                    let away = if d.length_squared() > 1e-8 {
                        d.normalize()
                    } else {
                        // Exactly on top of each other: split by order.
                        if n < m {
                            Vec3::X
                        } else {
                            Vec3::NEG_X
                        }
                    };
                    step += away * (gap * 0.5).min(2.0 * dt + gap * 4.0 * dt);
                }
            }
            moves.push((id, pos, step, speed, yaw, next, arrived));
        }
        for (id, pos, step, speed, yaw, next, arrived) in moves {
            let new_pos = self.nav_move(pos, step);
            let moved = Vec3::new(new_pos.x - pos.x, 0.0, new_pos.z - pos.z).length();
            let a = self.nav.agents.get_mut(&id).unwrap();
            a.speed = if speed > 0.0 {
                speed.min(moved / dt + 0.05).max(0.0)
            } else {
                0.0
            };
            a.yaw = yaw;
            a.next = next;
            if a.status == NavStatus::Moving && speed > 0.2 && moved < speed * dt * 0.1 {
                a.stuck += dt;
            } else {
                a.stuck = 0.0;
            }
            if arrived {
                a.status = NavStatus::Arrived;
                a.goal = None;
                a.path.clear();
            }
            let (scale, real_speed) = (a.scale, a.speed);
            let place = crate::place::Place::new(new_pos.x, new_pos.y, new_pos.z)
                .with_yaw_deg(yaw)
                .with_scale(scale);
            let _ = self.set_place(id, place);
            let state = if real_speed > 0.05 {
                crate::anim::Locomotion::Moving {
                    speed_mps: real_speed,
                }
            } else {
                crate::anim::Locomotion::Idle
            };
            if self.animated.contains_key(&id) {
                let _ = self.set_locomotion(id, state);
            }
        }
    }

    /// Move from `pos` by `step` over the walkable surface, sliding along
    /// edges; returns the new position with its height on the ground.
    fn nav_move(&self, pos: Vec3, step: Vec3) -> Vec3 {
        let Some(g) = self.nav.grid.as_ref() else {
            return pos + step;
        };
        let s = g.settings;
        let Some(from) = g.at(pos, s.max_step + 0.1, s.max_step + 0.5) else {
            // Off the walkable area (spawned outside it, pushed off): move
            // freely, back towards it.
            return pos + step;
        };
        let tries = [
            step,
            Vec3::new(step.x, 0.0, 0.0),
            Vec3::new(0.0, 0.0, step.z),
        ];
        for t in tries {
            if t.length_squared() < 1e-12 {
                continue;
            }
            let to = pos + t;
            if let Some(c) = g.walk_line(from, pos, to) {
                let coarse = g.span(c).y;
                let sampler = self.nav_sampler();
                let y = sampler
                    .down(Vec3::new(to.x, coarse + s.max_step, to.z), s.max_step * 2.0)
                    .map_or(coarse, |h| h.0);
                // Smooth steps a little so stairs don't jerk.
                let k = 1.0 - (-25.0 * step.length().max(0.01) / 0.05).exp();
                let k = k.clamp(0.3, 1.0);
                return Vec3::new(to.x, pos.y + (y - pos.y) * k, to.z);
            }
        }
        pos
    }

    /// Rebuild the navigation where voxel terrain was dug or built.
    fn refresh_nav_after_digging(&mut self) {
        let Some(v) = self.voxel_terrain.as_ref() else {
            return;
        };
        let rev = v.volume().rev();
        if self.nav.grid.is_none() || rev == self.nav.terrain_rev {
            self.nav.terrain_rev = rev;
            return;
        }
        let (keys, removed) = v.volume().changed_since(self.nav.terrain_rev);
        let size = v.volume().chunk_world_size();
        self.nav.terrain_rev = rev;
        if removed {
            let g = self.nav.grid.as_ref().unwrap();
            let (min, max) = (g.min, g.max);
            self.rebuild_nav_area(min, max);
            return;
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for k in keys {
            let a = k.as_vec3() * size;
            lo = lo.min(a);
            hi = hi.max(a + Vec3::splat(size));
        }
        if lo.x <= hi.x {
            self.rebuild_nav_area(lo, hi);
        }
    }
}
