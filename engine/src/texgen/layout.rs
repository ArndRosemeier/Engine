//! Tileable block layouts (brick bonds, ashlar courses, slabs, shingles):
//! horizontal courses of blocks in a `W` × `W` tile, wrapping both ways.

use super::core::wrap;
use super::paint::rng;

/// The block under a query point.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Hit {
    /// Stable id (course * 4096 + block).
    pub id: u32,
    /// Pattern-specific kind (header / stretcher).
    pub kind: u32,
    /// Block rectangle in the same (unwrapped) frame as `x`, `y`.
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
    /// Wrapped query point.
    pub x: f32,
    pub y: f32,
}

pub(crate) struct Layout {
    w: f32,
    /// Course start heights, `ys[n] = W`.
    ys: Vec<f32>,
    xs: Vec<Vec<f32>>,
    kinds: Vec<Vec<u32>>,
}

impl Layout {
    pub fn new(w: f32, heights: &[f32], xs: Vec<Vec<f32>>, kinds: Vec<Vec<u32>>) -> Self {
        let total: f32 = heights.iter().sum();
        let k = w / total;
        let mut ys = vec![0.0];
        let mut acc = 0.0;
        for h in heights {
            acc += h;
            ys.push(acc * k);
        }
        let mut sx = Vec::new();
        let mut sk = Vec::new();
        for (c, row) in xs.iter().enumerate() {
            let kr = kinds.get(c);
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|a, b| wrap(row[*a], w).total_cmp(&wrap(row[*b], w)));
            sx.push(order.iter().map(|k| wrap(row[*k], w)).collect());
            sk.push(
                order
                    .iter()
                    .map(|k| kr.and_then(|r| r.get(*k)).copied().unwrap_or(0))
                    .collect(),
            );
        }
        Self {
            w,
            ys,
            xs: sx,
            kinds: sk,
        }
    }

    pub fn at(&self, xq: f32, yq: f32) -> Hit {
        let w = self.w;
        let (x, y) = (wrap(xq, w), wrap(yq, w));
        let nc = self.ys.len() - 1;
        let mut c = 0;
        while c < nc - 1 && y >= self.ys[c + 1] {
            c += 1;
        }
        let row = &self.xs[c];
        let m = row.len();
        // Last block start <= x.
        let b = row.partition_point(|v| *v <= x);
        let (b, x0, x1) = if b == 0 {
            (m - 1, row[m - 1] - w, row[0])
        } else {
            let b = b - 1;
            (b, row[b], if b + 1 < m { row[b + 1] } else { row[0] + w })
        };
        Hit {
            id: (c * 4096 + b) as u32,
            kind: self.kinds[c].get(b).copied().unwrap_or(0),
            x0,
            x1,
            y0: self.ys[c],
            y1: self.ys[c + 1],
            x,
            y,
        }
    }
}

/// Running (stretcher) bond with optional head-joint jitter (m).
pub(crate) fn running_bond(
    w: f32,
    courses: usize,
    per_course: usize,
    seed: u32,
    jitter: f32,
    offset: f32,
) -> Layout {
    let mut r = rng(seed);
    let l = w / per_course as f32;
    let xs = (0..courses)
        .map(|c| {
            let off = (c % 2) as f32 * offset * l + if c % 4 >= 2 { 0.03 * l } else { 0.0 };
            (0..per_course)
                .map(|k| off + k as f32 * l + (r.float() - 0.5) * 2.0 * jitter)
                .collect()
        })
        .collect();
    Layout::new(w, &vec![1.0; courses], xs, Vec::new())
}

/// Flemish bond: stretchers (kind 0) and headers (kind 1) alternating, the
/// headers centred on the stretchers of the courses around them.
pub(crate) fn flemish_bond(w: f32, courses: usize, units: usize, seed: u32, jitter: f32) -> Layout {
    let mut r = rng(seed);
    let u = w / units as f32;
    let (s, h) = (u * (2.0 / 3.0), u / 3.0);
    let mut xs = Vec::new();
    let mut kinds = Vec::new();
    for c in 0..courses {
        let (mut row, mut kr) = (Vec::new(), Vec::new());
        for k in 0..units {
            let k = k as f32;
            if c % 2 == 0 {
                row.push(k * u + (r.float() - 0.5) * 2.0 * jitter);
                kr.push(0);
                row.push(k * u + s + (r.float() - 0.5) * 2.0 * jitter);
                kr.push(1);
            } else {
                let off = s / 2.0 - h / 2.0;
                row.push(off + k * u + (r.float() - 0.5) * 2.0 * jitter);
                kr.push(1);
                row.push(off + k * u + h + (r.float() - 0.5) * 2.0 * jitter);
                kr.push(0);
            }
        }
        xs.push(row);
        kinds.push(kr);
    }
    Layout::new(w, &vec![1.0; courses], xs, kinds)
}

/// Random ashlar courses: each course filled with blocks `min_l..max_l`
/// long that sum to `W`, keeping joints off those of the course below.
pub(crate) fn random_courses(w: f32, heights: &[f32], min_l: f32, max_l: f32, seed: u32) -> Layout {
    let mut r = rng(seed);
    let mut xs: Vec<Vec<f32>> = Vec::new();
    let mut prev: Vec<f32> = Vec::new();
    for _ in heights {
        let mut best = Vec::new();
        let mut best_score = -1.0f32;
        for _ in 0..12 {
            let mut lens = Vec::new();
            let mut sum = 0.0;
            while sum < w - min_l * 0.5 {
                let l = r.range(min_l, max_l);
                lens.push(l);
                sum += l;
            }
            let k = w / sum;
            let mut x = r.float() * w;
            let mut row = Vec::with_capacity(lens.len());
            for l in &lens {
                row.push(wrap(x, w));
                x += l * k;
            }
            let mut score = 1e9f32;
            for a in &row {
                for b in &prev {
                    let d = (((a - b + w * 1.5) % w) - w / 2.0).abs();
                    score = score.min(w / 2.0 - d);
                }
            }
            if prev.is_empty() {
                score = 1.0;
            }
            if score > best_score {
                best_score = score;
                best = row;
            }
            if score > min_l * 0.3 {
                break;
            }
        }
        prev = best.clone();
        xs.push(best);
    }
    Layout::new(w, heights, xs, Vec::new())
}

/// Signed inside distance (m) of the hit point to the block shrunk by the
/// half joint, with corners rounded by `cr`.
pub(crate) fn block_dist(o: &Hit, half_joint: f32, cr: f32) -> f32 {
    let dx = (o.x - o.x0).min(o.x1 - o.x) - half_joint;
    let dy = (o.y - o.y0).min(o.y1 - o.y) - half_joint;
    if dx < cr && dy < cr {
        let (ax, ay) = (cr - dx, cr - dy);
        if ax > 0.0 && ay > 0.0 {
            return cr - (ax * ax + ay * ay).sqrt();
        }
    }
    dx.min(dy)
}
