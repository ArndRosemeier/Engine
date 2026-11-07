//! Wrapped stamping of scattered shapes (pebbles, blades, leaves) into a
//! [`Tex`], with the height buffer as z-buffer.

use super::core::{mix, rnd1, smooth, Rgb, TNoise, Tex};
use crate::vegetation::rng::Rng;

/// A seeded stream for a generator.
pub(crate) fn rng(seed: u32) -> Rng {
    Rng::from(&["texgen"], seed)
}

/// Calls `f(t, dx, dy, idx)` for every pixel whose centre lies within `r`
/// metres of `(cx, cy)`, wrapping.
pub(crate) fn splat(
    t: &mut Tex,
    cx: f32,
    cy: f32,
    r: f32,
    mut f: impl FnMut(&mut Tex, f32, f32, usize),
) {
    let (s, ps) = (t.size as i32, t.ps);
    let (ci, cj, rp, r2) = (cx / ps - 0.5, cy / ps - 0.5, r / ps, r * r);
    for j in (cj - rp).floor() as i32..=(cj + rp).ceil() as i32 {
        let dy = (j as f32 - cj) * ps;
        if dy * dy > r2 {
            continue;
        }
        let row = (j.rem_euclid(s) * s) as usize;
        for i in (ci - rp).floor() as i32..=(ci + rp).ceil() as i32 {
            let dx = (i as f32 - ci) * ps;
            if dx * dx + dy * dy > r2 {
                continue;
            }
            f(t, dx, dy, row + i.rem_euclid(s) as usize);
        }
    }
}

/// Walks a thin segment from `(x, y)` in direction `a` for `len` metres;
/// `f(t, s, idx)` with `s` 0..1 along it.
pub(crate) fn stroke(
    t: &mut Tex,
    x: f32,
    y: f32,
    a: f32,
    len: f32,
    mut f: impl FnMut(&mut Tex, f32, usize),
) {
    let (s, ps) = (t.size as i32, t.ps);
    let n = ((len / (ps * 0.7)).ceil() as i32).max(1);
    let (ca, sa) = (a.cos(), a.sin());
    let mut last = usize::MAX;
    for k in 0..=n {
        let q = k as f32 / n as f32;
        let i = ((x + ca * len * q) / ps).floor() as i32;
        let j = ((y + sa * len * q) / ps).floor() as i32;
        let idx = (j.rem_euclid(s) * s + i.rem_euclid(s)) as usize;
        if idx == last {
            continue;
        }
        last = idx;
        f(t, q, idx);
    }
}

pub(crate) struct Pebbles<'a> {
    /// Coverage multiplier.
    pub count: f32,
    pub r_min: f32,
    pub r_max: f32,
    /// Dome height relative to the radius.
    pub height: f32,
    pub palette: &'a [Rgb],
    pub fines: Rgb,
    pub fines_h: f32,
    /// Where pebbles may lie, 0..1 by `(u, v)`.
    pub mask: Option<&'a dyn Fn(f32, f32) -> f32>,
    pub dust: f32,
    /// false: keep the existing contents as the bed.
    pub bed: bool,
}

impl Default for Pebbles<'_> {
    fn default() -> Self {
        Self {
            count: 1.0,
            r_min: 0.01,
            r_max: 0.05,
            height: 0.5,
            palette: &[],
            fines: [0.0; 3],
            fines_h: 0.0,
            mask: None,
            dust: 0.0,
            bed: true,
        }
    }
}

/// Overlapping, z-buffered ellipsoidal pebbles over a bed of fines.
pub(crate) fn pebbles(t: &mut Tex, seed: u32, o: &Pebbles) {
    let w = t.w;
    let nf = TNoise::new(seed + 1, w);
    let nd = TNoise::new(seed + 2, w);
    if o.bed {
        t.each(|t, u, v, _, _, idx| {
            let k = 1.0 + nf.fbm(u, v, 0.02, 3) * 0.15 + (rnd1(seed + 3, idx as u32) - 0.5) * 0.25;
            let h = o.fines_h + nf.fbm(u, v, 0.05, 3) * 0.002;
            t.set(idx, o.fines[0] * k, o.fines[1] * k, o.fines[2] * k, 0.95, h);
        });
    }
    let mut rng = rng(seed + 4);
    let rm = (o.r_min + o.r_max) / 2.0;
    let n = ((w * w) / (std::f32::consts::PI * rm * rm * 0.7) * o.count).round() as u32;
    for _ in 0..n {
        let (cx, cy) = (rng.float() * w, rng.float() * w);
        if let Some(mask) = o.mask {
            if rng.float() > mask(cx / w, cy / w) {
                continue;
            }
        }
        let a = rng.range(o.r_min, o.r_max);
        let b = a * rng.range(0.55, 0.95);
        let th = rng.float() * std::f32::consts::PI;
        let (ct, st) = (th.cos(), th.sin());
        let base = -a * rng.range(0.0, 0.35) * o.height;
        let hk = a * o.height * rng.range(0.75, 1.1);
        let col =
            o.palette[((rng.float() * o.palette.len() as f32) as usize).min(o.palette.len() - 1)];
        let vk = rng.range(0.82, 1.15);
        let sk = rng.int(0, 1_000_000);
        splat(t, cx, cy, a, |t, dx, dy, idx| {
            let px = dx * ct + dy * st;
            let py = -dx * st + dy * ct;
            let q = (px / a).powi(2) + (py / b).powi(2);
            if q >= 1.0 {
                return;
            }
            let z = base + (1.0 - q).sqrt() * hk;
            if z <= t.h[idx] {
                return;
            }
            let speck = 1.0 + (rnd1(sk, idx as u32) - 0.5) * 0.12;
            let rim = 0.8 + 0.2 * (1.0 - q).sqrt();
            let k = vk * speck * rim;
            t.set(
                idx,
                col[0] * k,
                col[1] * k,
                col[2] * k,
                0.75 - 0.15 * (1.0 - q),
                z,
            );
        });
    }
    if o.dust > 0.0 {
        t.each(|t, u, v, _, _, idx| {
            let d = smooth(-0.2, 0.6, nd.fbm(u, v, 0.4, 4)) * o.dust;
            t.r[idx] = mix(t.r[idx], o.fines[0] * 1.1, d);
            t.g[idx] = mix(t.g[idx], o.fines[1] * 1.1, d);
            t.b[idx] = mix(t.b[idx], o.fines[2] * 1.1, d);
        });
    }
}
