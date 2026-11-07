//! Only simulate where something happens.
//!
//! The region is cut into tiles of 16 × 16 columns. A tile is awake while
//! water flows in it; still or dry tiles sleep and cost nothing. Every frame
//! the awake tiles plus their neighbours run (so a front or a wave can step
//! into the next tile and wake it). Links between a running and a resting
//! tile carry no flow for that frame, which keeps the volume exact.

use super::sim::{Water, DRY};
use glam::Vec2;
use rayon::prelude::*;

/// Columns along a tile side.
pub(crate) const TILE: usize = 16;
/// Discharge (m²/s) below which a tile counts as still.
const STILL: f32 = 2e-5;
/// Level difference (m) across a resting tile's border that wakes it.
const LEVEL_STEP: f32 = 5e-4;

/// Which tiles run this frame.
#[derive(Clone, Debug, Default)]
pub(crate) struct Activity {
    pub tiles_x: usize,
    pub tiles_z: usize,
    pub awake: Vec<bool>,
    pub run: Vec<bool>,
    /// Per band (one row of tiles): running column ranges `[x0, x1)`.
    pub runs: Vec<Vec<(usize, usize)>>,
    /// Wet cells per tile, as of the last frame it ran.
    pub wet: Vec<u32>,
    /// Keep every tile running (tests compare against this).
    pub never_sleep: bool,
}

impl Activity {
    pub fn new(cells_x: usize, cells_z: usize) -> Self {
        let tiles_x = cells_x.div_ceil(TILE);
        let tiles_z = cells_z.div_ceil(TILE);
        let n = tiles_x * tiles_z;
        Self {
            tiles_x,
            tiles_z,
            awake: vec![true; n],
            run: vec![false; n],
            runs: vec![Vec::new(); tiles_z],
            wet: vec![0; n],
            never_sleep: false,
        }
    }

    pub fn tile_of_column(&self, cells_x: usize, col: usize) -> usize {
        let (x, z) = (col % cells_x, col / cells_x);
        (z / TILE) * self.tiles_x + x / TILE
    }
}

/// Runs a closure over the running cells (or links) band by band in
/// parallel. `start(c)` is the first element of column `c`; elements of a
/// column range are contiguous.
pub(crate) struct Sweep<'a> {
    pub nx: usize,
    pub nz: usize,
    pub runs: &'a [Vec<(usize, usize)>],
}

impl Sweep<'_> {
    pub fn each<T: Send>(
        &self,
        data: &mut [T],
        start: impl Fn(usize) -> usize + Sync,
        f: impl Fn(usize, &mut T) -> f32 + Sync,
    ) -> f32 {
        let (nx, nz) = (self.nx, self.nz);
        let mut slices = Vec::with_capacity(self.runs.len());
        let mut rest = data;
        let mut offset = 0;
        for b in 0..self.runs.len() {
            let end = start(((b + 1) * TILE).min(nz) * nx);
            let (head, tail) = rest.split_at_mut(end - offset);
            slices.push((b, offset, head));
            rest = tail;
            offset = end;
        }
        slices
            .into_par_iter()
            .map(|(b, offset, slice)| {
                let mut m = 0.0f32;
                if self.runs[b].is_empty() {
                    return m;
                }
                for row in b * TILE..((b + 1) * TILE).min(nz) {
                    for &(x0, x1) in &self.runs[b] {
                        let (s, e) = (start(row * nx + x0), start(row * nx + x1));
                        for i in s..e {
                            m = m.max(f(i, &mut slice[i - offset]));
                        }
                    }
                }
                m
            })
            .reduce(|| 0.0f32, f32::max)
    }
}

impl Water {
    pub(crate) fn wake_all(&mut self) {
        self.activity.awake.iter_mut().for_each(|a| *a = true);
    }

    /// Wake the tiles within `radius` of `at`.
    pub(crate) fn wake_near(&mut self, at: Vec2, radius: f32) {
        let r = self.region;
        let a = &mut self.activity;
        let cs = r.cell_size * TILE as f32;
        let lo = ((at - Vec2::splat(radius) - r.min_xz) / cs).floor();
        let hi = ((at + Vec2::splat(radius) - r.min_xz) / cs).floor();
        for tz in (lo.y.max(0.0) as usize)..=(hi.y.max(0.0) as usize).min(a.tiles_z - 1) {
            for tx in (lo.x.max(0.0) as usize)..=(hi.x.max(0.0) as usize).min(a.tiles_x - 1) {
                a.awake[tz * a.tiles_x + tx] = true;
            }
        }
    }

    /// Pick the tiles that run this frame: the awake ones and their
    /// neighbours. Links of tiles that stop running are zeroed.
    pub(crate) fn schedule(&mut self) {
        let nx = self.region.cells_x as usize;
        let nz = self.region.cells_z as usize;
        let a = &mut self.activity;
        let (tx, tz) = (a.tiles_x, a.tiles_z);
        let mut run = vec![false; tx * tz];
        for z in 0..tz {
            for x in 0..tx {
                if !a.awake[z * tx + x] {
                    continue;
                }
                for zz in z.saturating_sub(1)..(z + 2).min(tz) {
                    for xx in x.saturating_sub(1)..(x + 2).min(tx) {
                        run[zz * tx + xx] = true;
                    }
                }
            }
        }
        // Tiles going to rest keep no flow on their links.
        for (t, &runs_now) in run.iter().enumerate() {
            if a.run[t] && !runs_now {
                let (x0, z0) = ((t % tx) * TILE, (t / tx) * TILE);
                let x1 = (x0 + TILE).min(nx);
                for row in z0..(z0 + TILE).min(nz) {
                    let s = self.link_group[2 * (row * nx + x0)] as usize;
                    let e = self.link_group[2 * (row * nx + x1)] as usize;
                    self.q[s..e].iter_mut().for_each(|q| *q = 0.0);
                    self.q_new[s..e].iter_mut().for_each(|q| *q = 0.0);
                }
            }
        }
        for (b, band) in a.runs.iter_mut().enumerate() {
            band.clear();
            for x in 0..tx {
                if !run[b * tx + x] {
                    continue;
                }
                let (x0, x1) = (x * TILE, ((x + 1) * TILE).min(nx));
                match band.last_mut() {
                    Some(last) if last.1 == x0 => last.1 = x1,
                    _ => band.push((x0, x1)),
                }
            }
        }
        a.run = run;
    }

    /// After a frame: which tiles still have flow, and how many wet cells.
    pub(crate) fn settle_activity(&mut self) {
        let nx = self.region.cells_x as usize;
        let nz = self.region.cells_z as usize;
        let (tx, tz) = (self.activity.tiles_x, self.activity.tiles_z);
        let (q, depth, link_group, col_start, run) = (
            &self.q,
            &self.depth,
            &self.link_group,
            &self.col_start,
            &self.activity.run,
        );
        let result: Vec<(bool, u32)> = (0..tx * tz)
            .into_par_iter()
            .map(|t| {
                if !run[t] {
                    return (false, u32::MAX);
                }
                let (x0, z0) = ((t % tx) * TILE, (t / tx) * TILE);
                let x1 = (x0 + TILE).min(nx);
                let mut flowing = false;
                let mut wet = 0;
                for row in z0..(z0 + TILE).min(nz) {
                    let s = link_group[2 * (row * nx + x0)] as usize;
                    let e = link_group[2 * (row * nx + x1)] as usize;
                    flowing |= q[s..e].iter().any(|v| v.abs() > STILL);
                    let s = col_start[row * nx + x0] as usize;
                    let e = col_start[row * nx + x1] as usize;
                    wet += depth[s..e].iter().filter(|&&d| d > DRY).count() as u32;
                }
                (flowing, wet)
            })
            .collect();
        let no_sleep = self.activity.never_sleep;
        for (t, (flowing, wet)) in result.into_iter().enumerate() {
            self.activity.awake[t] = flowing || no_sleep;
            if wet != u32::MAX {
                self.activity.wet[t] = wet;
            }
        }
        // A resting tile whose level no longer matches the running water
        // next to it wakes up; otherwise the closed border between them
        // would leave a step in the surface.
        for t in self.mismatched_resting_tiles() {
            self.activity.awake[t] = true;
        }
        // Sources keep their tiles going; so do sea edges with swell or flow.
        let sources: Vec<Vec2> = self
            .sources
            .iter()
            .map(|(_, s)| Vec2::new(s.position.x, s.position.z))
            .collect();
        for p in sources {
            self.wake_near(p, 0.0);
        }
        let swell = self.sea.as_ref().is_some_and(|s| !s.swell.is_empty());
        let marks: Vec<usize> = self
            .boundary
            .iter()
            .zip(&self.boundary_q)
            .filter(|(_, q)| swell || q.abs() > STILL)
            .map(|(b, _)| {
                self.activity
                    .tile_of_column(nx, self.cell_col[b.cell as usize] as usize)
            })
            .collect();
        for t in marks {
            self.activity.awake[t] = true;
        }
    }

    /// Resting tiles next to running ones whose water level differs across
    /// the border.
    fn mismatched_resting_tiles(&self) -> Vec<usize> {
        let nx = self.region.cells_x as usize;
        let nz = self.region.cells_z as usize;
        let a = &self.activity;
        let tx = a.tiles_x;
        let tile = |cell: u32| a.tile_of_column(nx, self.cell_col[cell as usize] as usize);
        let differs = |id: usize| {
            let l = self.links[id];
            let (ha, hb) = (self.cell_head(l.a as usize), self.cell_head(l.b as usize));
            ha.max(hb) > l.lo + DRY && (ha.max(l.lo) - hb.max(l.lo)).abs() > LEVEL_STEP
        };
        // Links of column c whose other end lies in a resting tile.
        let check_column = |c: usize, out: &mut Vec<usize>| {
            let (s, e) = (
                self.link_group[2 * c] as usize,
                self.link_group[2 * c + 2] as usize,
            );
            for id in s..e {
                let l = self.links[id];
                let (ta, tb) = (tile(l.a), tile(l.b));
                if ta == tb || a.run[ta] == a.run[tb] {
                    continue;
                }
                if differs(id) {
                    out.push(if a.run[ta] { tb } else { ta });
                }
            }
        };
        (0..a.run.len())
            .into_par_iter()
            .filter(|&t| a.run[t])
            .flat_map_iter(|t| {
                let mut out = Vec::new();
                let (x0, z0) = ((t % tx) * TILE, (t / tx) * TILE);
                let (x1, z1) = ((x0 + TILE).min(nx), (z0 + TILE).min(nz));
                // Links leave a column towards +x and +z: the tile's last
                // column and row, plus the column and row just before it.
                for z in z0..z1 {
                    check_column(z * nx + x1 - 1, &mut out);
                    if x0 > 0 {
                        check_column(z * nx + x0 - 1, &mut out);
                    }
                }
                for x in x0..x1 {
                    check_column((z1 - 1) * nx + x, &mut out);
                    if z0 > 0 {
                        check_column((z0 - 1) * nx + x, &mut out);
                    }
                }
                out
            })
            .collect()
    }

    /// Is this cell in a tile that runs this frame?
    pub(crate) fn cell_runs(&self, cell: usize) -> bool {
        let nx = self.region.cells_x as usize;
        self.activity.run[self
            .activity
            .tile_of_column(nx, self.cell_col[cell] as usize)]
    }

    /// Tiles running in the last frame and in total.
    pub fn active_tiles(&self) -> (usize, usize) {
        let running = if self.on_gpu() {
            self.gpu_running
        } else {
            self.activity.run.iter().filter(|r| **r).count()
        };
        (running, self.activity.run.len())
    }
}
