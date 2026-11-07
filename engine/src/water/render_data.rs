//! What the renderer reads from the simulation each frame.

use super::sim::{Water, DRY};
use rayon::prelude::*;

/// Water thinner than this (metres) is not drawn.
pub(crate) const DRAW_DEPTH: f32 = 0.004;
pub(crate) const NO_CELL: u32 = u32::MAX;

impl Water {
    /// Per cell: surface height, water depth (-1 when the cell is full to its
    /// ceiling, 0 when dry), current X and Z.
    pub(crate) fn render_state(&self, out: &mut Vec<[f32; 4]>) {
        out.resize(self.floor.len(), [0.0; 4]);
        out.par_iter_mut().enumerate().for_each(|(i, s)| {
            let d = self.depth[i];
            if d <= DRY {
                *s = [self.floor[i], 0.0, 0.0, 0.0];
                return;
            }
            let head = self.cell_head(i);
            let cur = self.cell_current(i);
            if head >= self.ceil[i] {
                *s = [self.ceil[i], -1.0, cur.x, cur.y];
            } else {
                *s = [head, head - self.floor[i], cur.x, cur.y];
            }
        });
    }

    /// Per cell, its four quad corners, each as the 2x2 cells around that
    /// corner in the order (-x-z, +x-z, -x+z, +x+z): the cell itself, linked
    /// neighbours, or `NO_CELL`. Corners are shared exactly where water
    /// connects, so the surface is continuous across a body of water and
    /// separate across a wall.
    pub(crate) fn render_corners(&self) -> Vec<[u32; 16]> {
        (0..self.floor.len())
            .into_par_iter()
            .map(|c| {
                let mut out = [NO_CELL; 16];
                for k in 0..4 {
                    let (kx, kz) = ((k & 1) as i32, (k >> 1) as i32);
                    // Our position inside this corner's 2x2 stencil.
                    let (ox, oz) = (1 - kx, 1 - kz);
                    for j in 0..4 {
                        let (sx, sz) = ((j & 1) as i32, (j >> 1) as i32);
                        let (dx, dz) = (sx - ox, sz - oz);
                        let cell = match (dx, dz) {
                            (0, 0) => Some(c),
                            (_, 0) => self.linked_cell(c, dx, 0),
                            (0, _) => self.linked_cell(c, 0, dz),
                            _ => self
                                .linked_cell(c, dx, 0)
                                .and_then(|s| self.linked_cell(s, 0, dz))
                                .or_else(|| {
                                    self.linked_cell(c, 0, dz)
                                        .and_then(|s| self.linked_cell(s, dx, 0))
                                }),
                        };
                        out[k * 4 + j] = cell.map_or(NO_CELL, |v| v as u32);
                    }
                }
                out
            })
            .collect()
    }

    /// Column index of every cell.
    pub(crate) fn render_cell_columns(&self) -> &[u32] {
        &self.cell_col
    }

    /// Cells to draw: wet ones and their dry neighbours (whose quads carry
    /// the surface into the bank, so the shoreline is where water meets
    /// ground, with no gap). Full cells have no surface.
    pub(crate) fn render_draw_list(
        &self,
        state: &[[f32; 4]],
        corners: &[[u32; 16]],
        out: &mut Vec<u32>,
    ) {
        let open_wet = |i: u32| {
            i != NO_CELL && {
                let d = state[i as usize][1];
                d > DRAW_DEPTH
            }
        };
        *out = (0..state.len() as u32)
            .into_par_iter()
            .filter(|&i| {
                let d = state[i as usize][1];
                if d < 0.0 {
                    return false;
                }
                d > DRAW_DEPTH || corners[i as usize].iter().any(|&n| open_wet(n))
            })
            .collect();
    }
}
