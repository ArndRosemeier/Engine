//! Foam where water breaks and wet ground where it has just been.

use super::active::Sweep;
use super::sim::{Water, DRY, G};
use glam::{Vec2, Vec3};

/// Seconds for foam to fade to a third.
pub(crate) const FOAM_SECONDS: f32 = 5.0;
/// Seconds for wet ground to dry to a third.
pub(crate) const DRYING_SECONDS: f32 = 30.0;
/// Never wet, no foam.
pub(crate) const NO_MARKS: [f32; 4] = [0.0, 0.0, -1e9, 0.0];

impl Water {
    /// After a frame: foam forms where the surface breaks (steep fronts,
    /// supercritical flow) and fades over a few seconds; wet cells remember
    /// when they were wet. Only running tiles change.
    pub(crate) fn update_marks(&mut self) {
        let t = self.time as f32;
        let dx = self.region.cell_size;
        let breaking = self.settings.breaking_slope.max(1e-3);
        let runs = std::mem::take(&mut self.activity.runs);
        let sweep = Sweep {
            nx: self.region.cells_x as usize,
            nz: self.region.cells_z as usize,
            runs: &runs,
        };
        let (col_start, depth, ceil, head, q, links, cls, cl) = (
            &self.col_start,
            &self.depth,
            &self.ceil,
            &self.head,
            &self.q,
            &self.links,
            &self.cell_link_start,
            &self.cell_links,
        );
        sweep.each(
            &mut self.marks,
            |c| col_start[c] as usize,
            |i, m| {
                if depth[i] <= DRY {
                    return 0.0;
                }
                // Wet as far as the eye can tell: the drawn water.
                if depth[i] > super::render_data::DRAW_DEPTH {
                    m[2] = t;
                }
                if head[i] >= ceil[i] {
                    return 0.0;
                }
                let mut make: f32 = 0.0;
                for &entry in &cl[cls[i] as usize..cls[i + 1] as usize] {
                    let id = (entry & !(1 << 31)) as usize;
                    let l = links[id];
                    let other = if l.a as usize == i { l.b } else { l.a } as usize;
                    let h = (head[i].max(head[other]).min(l.hi) - l.lo).max(0.02);
                    // A step in the surface between wet cells: a breaking front.
                    if depth[other] > DRY {
                        let slope = (head[i] - head[other]).abs() / dx;
                        make = make.max(smoothstep(breaking, breaking * 3.0, slope));
                    }
                    // Faster than waves travel: rapids, bores, swash.
                    let froude = q[id].abs() / h / (G * h).sqrt();
                    // (Thin sheets are always fast; only real depth breaks white.)
                    make = make.max(smoothstep(0.9, 1.8, froude) * smoothstep(0.03, 0.2, h) * 0.8);
                }
                let now = m[0] * (-(t - m[1]) / FOAM_SECONDS).exp();
                m[0] = now.max(make);
                m[1] = t;
                0.0
            },
        );
        self.activity.runs = runs;
    }

    /// Whitewater around a splash.
    pub(crate) fn foam_at(&mut self, at: Vec3, radius: f32) {
        let t = self.time as f32;
        let r = self.region;
        let cs = r.cell_size;
        let lo = ((Vec2::new(at.x, at.z) - Vec2::splat(radius) - r.min_xz) / cs).floor();
        let hi = ((Vec2::new(at.x, at.z) + Vec2::splat(radius) - r.min_xz) / cs).ceil();
        let (nx, nz) = (r.cells_x as i64, r.cells_z as i64);
        for z in (lo.y as i64).clamp(0, nz)..(hi.y as i64).clamp(0, nz) {
            for x in (lo.x as i64).clamp(0, nx)..(hi.x as i64).clamp(0, nx) {
                let c = (z * nx + x) as usize;
                let d = (r.column_centre(x as u32, z as u32) - Vec2::new(at.x, at.z)).length();
                let w = (1.0 - d / radius).clamp(0.0, 1.0);
                for cell in self.col_start[c] as usize..self.col_start[c + 1] as usize {
                    if self.depth[cell] <= DRY
                        || self.floor[cell] > at.y + radius
                        || self.cell_head(cell) < at.y - radius
                    {
                        continue;
                    }
                    let m = &mut self.marks[cell];
                    let now = m[0] * (-(t - m[1]) / FOAM_SECONDS).exp();
                    *m = [now.max(w), t, m[2], m[3]];
                    self.gpu_edit(super::backend::GpuEdit::Foam {
                        cell: cell as u32,
                        amount: w,
                    });
                }
            }
        }
    }

    /// Foam on the surface at `p`, 0..1.
    pub fn foam(&self, p: Vec3) -> f32 {
        let Some(cell) = self
            .cell_at(p)
            .or_else(|| self.cell_below(p))
            .or_else(|| self.cell_above(p))
        else {
            return 0.0;
        };
        let m = self.marks[cell as usize];
        m[0] * (-(self.time as f32 - m[1]) / FOAM_SECONDS).exp()
    }

    /// How wet the ground at `p` still is from water that has left, 0..1
    /// (1 under water).
    pub fn wetness(&self, p: Vec3) -> f32 {
        let Some(cell) = self
            .cell_at(p)
            .or_else(|| self.cell_below(p))
            .or_else(|| self.cell_above(p))
        else {
            return 0.0;
        };
        let m = self.marks[cell as usize];
        (-(self.time as f32 - m[2]).max(0.0) / DRYING_SECONDS).exp()
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
