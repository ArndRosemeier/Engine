//! Texture synthesis core: tileable noise, periodic Worley cells, wrapped
//! blurs and the packing of albedo, roughness, height and occlusion into
//! RGBA8 layers.
//!
//! Pixel `(i, j)` has tile coordinates `u = (i + 0.5) / size`,
//! `v = (j + 0.5) / size` and lies `x = u * W`, `y = v * W` metres into the
//! tile. Row `j = 0` is `v = 0`, the bottom of the tile ("down" on a wall).

use std::cell::RefCell;

pub(crate) fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

pub(crate) fn hash2i(seed: u32, x: i32, y: i32) -> u32 {
    hash32(
        seed ^ hash32(
            (x as u32).wrapping_mul(0x27d4_eb2d) ^ hash32((y as u32).wrapping_mul(0x1656_67b1)),
        ),
    )
}

pub(crate) fn hash_to_float(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

/// Hash of `(seed, a, b)` to `[0, 1)`.
pub(crate) fn rnd(seed: u32, a: u32, b: u32) -> f32 {
    hash_to_float(hash2i(seed, a as i32, b as i32))
}

pub(crate) fn rnd1(seed: u32, a: u32) -> f32 {
    rnd(seed, a, 0)
}

pub(crate) fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub(crate) fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite step; `e0 > e1` gives a falling step, as in GLSL.
pub(crate) fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp01((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

/// Wrapped signed difference on a period `p` (result in `[-p/2, p/2)`).
pub(crate) fn wrapd(d: f32, p: f32) -> f32 {
    d - (d / p + 0.5).floor() * p
}

pub(crate) fn wrap(x: f32, p: f32) -> f32 {
    let r = x % p;
    if r < 0.0 {
        r + p
    } else {
        r
    }
}

/// sRGB colour, 0..1.
pub(crate) type Rgb = [f32; 3];

pub(crate) const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
}

fn grads() -> ([f32; 32], [f32; 32]) {
    let (mut gx, mut gy) = ([0.0; 32], [0.0; 32]);
    for k in 0..32 {
        let a = k as f32 / 32.0 * std::f32::consts::TAU + 0.21;
        gx[k] = a.cos();
        gy[k] = a.sin();
    }
    (gx, gy)
}

fn wrap_cell(x: i32, p: i32) -> i32 {
    x.rem_euclid(p)
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Permutation-table gradient noise, periodic in `px`, `py`; about -1..1.
#[allow(clippy::too_many_arguments)]
fn tgrad(p: &[u16; 1024], gx: &[f32; 32], gy: &[f32; 32], x: f32, y: f32, px: i32, py: i32) -> f32 {
    let (xf, yf) = (x.floor(), y.floor());
    let (fx, fy) = (x - xf, y - yf);
    let x0 = wrap_cell(xf as i32, px);
    let y0 = wrap_cell(yf as i32, py);
    let x1 = if x0 + 1 == px { 0 } else { x0 + 1 };
    let y1 = if y0 + 1 == py { 0 } else { y0 + 1 };
    let a0 = p[(x0 & 1023) as usize] as i32;
    let a1 = p[(x1 & 1023) as usize] as i32;
    let h00 = (p[((a0 + y0) & 1023) as usize] & 31) as usize;
    let h10 = (p[((a1 + y0) & 1023) as usize] & 31) as usize;
    let h01 = (p[((a0 + y1) & 1023) as usize] & 31) as usize;
    let h11 = (p[((a1 + y1) & 1023) as usize] & 31) as usize;
    let n00 = gx[h00] * fx + gy[h00] * fy;
    let n10 = gx[h10] * (fx - 1.0) + gy[h10] * fy;
    let n01 = gx[h01] * fx + gy[h01] * (fy - 1.0);
    let n11 = gx[h11] * (fx - 1.0) + gy[h11] * (fy - 1.0);
    let (u, v) = (fade(fx), fade(fy));
    let a = n00 + (n10 - n00) * u;
    let b = n01 + (n11 - n01) * u;
    (a + (b - a) * v) * 1.45
}

/// Tileable noise bound to a tile `w` metres square. Sampling takes tile
/// coordinates `u, v` and a wavelength in metres; frequencies round to whole
/// cycles per tile so everything wraps.
pub(crate) struct TNoise {
    seed: u32,
    w: f32,
    tabs: RefCell<Vec<Option<Box<[u16; 1024]>>>>,
    gx: [f32; 32],
    gy: [f32; 32],
}

impl TNoise {
    pub fn new(seed: u32, w: f32) -> Self {
        let (gx, gy) = grads();
        Self {
            seed,
            w,
            tabs: RefCell::new(vec![None; 48]),
            gx,
            gy,
        }
    }

    fn with_tab<R>(&self, k: usize, f: impl FnOnce(&[u16; 1024]) -> R) -> R {
        let mut tabs = self.tabs.borrow_mut();
        if tabs[k].is_none() {
            let mut t = Box::new([0u16; 1024]);
            for (i, v) in t.iter_mut().enumerate() {
                *v = i as u16;
            }
            for i in (1..1024usize).rev() {
                let j = (hash2i(self.seed, k as i32, i as i32) % (i as u32 + 1)) as usize;
                t.swap(i, j);
            }
            tabs[k] = Some(t);
        }
        f(tabs[k].as_ref().expect("table"))
    }

    /// Cycles per tile for a wavelength in metres.
    pub fn f(&self, lambda: f32) -> i32 {
        ((self.w / lambda).round() as i32).max(1)
    }

    /// One octave.
    pub fn n(&self, u: f32, v: f32, l: f32) -> f32 {
        let f = self.f(l);
        self.with_tab(0, |t| {
            tgrad(t, &self.gx, &self.gy, u * f as f32, v * f as f32, f, f)
        })
    }

    /// fBm, about -1..1.
    pub fn fbm(&self, u: f32, v: f32, l: f32, oct: u32) -> f32 {
        self.fbm_g(u, v, l, oct, 0.5, l)
    }

    /// fBm with a gain and per-axis wavelengths (`lx` across, `ly` up).
    pub fn fbm_g(&self, u: f32, v: f32, lx: f32, oct: u32, gain: f32, ly: f32) -> f32 {
        let (mut fx, mut fy) = (self.f(lx), self.f(ly));
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        for k in 0..oct {
            let n = self.with_tab(k as usize, |t| {
                tgrad(
                    t,
                    &self.gx,
                    &self.gy,
                    u * fx as f32 + k as f32 * 0.31,
                    v * fy as f32 + k as f32 * 0.17,
                    fx,
                    fy,
                )
            });
            sum += amp * n;
            norm += amp;
            amp *= gain;
            fx *= 2;
            fy *= 2;
        }
        sum / norm
    }

    /// Ridged fBm in 0..1 (sharp crests = 1).
    pub fn ridged(&self, u: f32, v: f32, l: f32, oct: u32) -> f32 {
        self.ridged_g(u, v, l, oct, 0.5, l)
    }

    pub fn ridged_g(&self, u: f32, v: f32, lx: f32, oct: u32, gain: f32, ly: f32) -> f32 {
        let (mut fx, mut fy) = (self.f(lx), self.f(ly));
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        for k in 0..oct {
            let n = self.with_tab(16 + k as usize, |t| {
                tgrad(t, &self.gx, &self.gy, u * fx as f32, v * fy as f32, fx, fy)
            });
            let r = 1.0 - n.abs();
            sum += amp * r * r;
            norm += amp;
            amp *= gain;
            fx *= 2;
            fy *= 2;
        }
        sum / norm
    }

    /// Turbulence (abs fBm) in 0..1.
    pub fn turb(&self, u: f32, v: f32, l: f32, oct: u32) -> f32 {
        let (mut fx, mut fy) = (self.f(l), self.f(l));
        let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
        for k in 0..oct {
            let n = self.with_tab(32 + k as usize, |t| {
                tgrad(t, &self.gx, &self.gy, u * fx as f32, v * fy as f32, fx, fy)
            });
            sum += amp * n.abs();
            norm += amp;
            amp *= 0.5;
            fx *= 2;
            fy *= 2;
        }
        sum / norm
    }
}

/// Periodic value noise in 0..1 (good for per-area masks).
pub(crate) fn pvalue(seed: u32, x: f32, y: f32, px: i32, py: i32) -> f32 {
    let (xf, yf) = (x.floor(), y.floor());
    let (fx, fy) = (x - xf, y - yf);
    let x0 = wrap_cell(xf as i32, px);
    let y0 = wrap_cell(yf as i32, py);
    let x1 = if x0 + 1 == px { 0 } else { x0 + 1 };
    let y1 = if y0 + 1 == py { 0 } else { y0 + 1 };
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let r = |a: i32, b: i32| hash_to_float(hash2i(seed, a, b));
    mix(
        mix(r(x0, y0), r(x1, y0), u),
        mix(r(x0, y1), r(x1, y1), u),
        v,
    )
}

/// Result of a Worley lookup.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cell {
    pub f1: f32,
    pub f2: f32,
    /// Distance to the border of the nearest cell (lattice units), with `edge`.
    pub edge: f32,
    /// Stable hash of the nearest cell.
    pub id: u32,
    /// Nearest feature point (unwrapped lattice coordinates).
    pub px: f32,
    pub py: f32,
}

/// Periodic Worley noise on a `px` × `py` lattice. With `edge`, also the
/// true distance to the cell border.
#[allow(clippy::too_many_arguments)]
pub(crate) fn worley(seed: u32, x: f32, y: f32, px: i32, py: i32, jitter: f32, edge: bool) -> Cell {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (mut f1, mut f2) = (1e9f32, 1e9f32);
    let (mut bx, mut by, mut bid, mut bcx, mut bcy) = (0.0, 0.0, 0u32, 0, 0);
    let feature = |cx: i32, cy: i32| {
        let h = hash2i(seed, wrap_cell(cx, px), wrap_cell(cy, py));
        let fx = cx as f32 + 0.5 + (hash_to_float(h) - 0.5) * jitter;
        let fy = cy as f32 + 0.5 + (hash_to_float(hash32(h)) - 0.5) * jitter;
        (h, fx, fy)
    };
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (xi + dx, yi + dy);
            let (h, fx, fy) = feature(cx, cy);
            let d = (fx - x) * (fx - x) + (fy - y) * (fy - y);
            if d < f1 {
                f2 = f1;
                f1 = d;
                bx = fx;
                by = fy;
                bid = h;
                bcx = cx;
                bcy = cy;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    let mut out = Cell {
        f1: f1.sqrt(),
        f2: f2.sqrt(),
        edge: 0.0,
        id: bid,
        px: bx,
        py: by,
    };
    if edge {
        let mut md = 1e9f32;
        for dy in -2..=2 {
            for dx in -2..=2 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (_, fx, fy) = feature(bcx + dx, bcy + dy);
                let (nx, ny) = (fx - bx, fy - by);
                let nl = (nx * nx + ny * ny).sqrt();
                if nl < 1e-6 {
                    continue;
                }
                let d = ((bx + fx) * 0.5 - x) * (nx / nl) + ((by + fy) * 0.5 - y) * (ny / nl);
                md = md.min(d);
            }
        }
        out.edge = md;
    }
    out
}

/// Working buffers of one layer.
pub(crate) struct Tex {
    pub size: usize,
    /// Tile size (m).
    pub w: f32,
    /// Metres per pixel.
    pub ps: f32,
    /// Albedo, sRGB 0..1.
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
    pub rough: Vec<f32>,
    /// Relief height (m).
    pub h: Vec<f32>,
}

impl Tex {
    pub fn new(size: usize, w: f32) -> Self {
        let n = size * size;
        Self {
            size,
            w,
            ps: w / size as f32,
            r: vec![0.0; n],
            g: vec![0.0; n],
            b: vec![0.0; n],
            rough: vec![0.0; n],
            h: vec![0.0; n],
        }
    }

    /// Calls `f(self, u, v, x, y, idx)` for every pixel.
    pub fn each(&mut self, mut f: impl FnMut(&mut Self, f32, f32, f32, f32, usize)) {
        let (s, w) = (self.size, self.w);
        for j in 0..s {
            let v = (j as f32 + 0.5) / s as f32;
            for i in 0..s {
                let u = (i as f32 + 0.5) / s as f32;
                f(self, u, v, u * w, v * w, j * s + i);
            }
        }
    }

    pub fn set(&mut self, idx: usize, r: f32, g: f32, b: f32, rough: f32, h: f32) {
        self.r[idx] = r;
        self.g[idx] = g;
        self.b[idx] = b;
        self.rough[idx] = rough;
        self.h[idx] = h;
    }
}

/// Separable wrapped box blur.
fn box_blur(src: &[f32], s: usize, r: usize) -> Vec<f32> {
    let inv = 1.0 / (2 * r + 1) as f32;
    let si = s as isize;
    let w = |k: isize| k.rem_euclid(si) as usize;
    let mut tmp = vec![0.0f32; src.len()];
    for j in 0..s {
        let row = j * s;
        let mut acc: f32 = (-(r as isize)..=r as isize).map(|k| src[row + w(k)]).sum();
        for i in 0..s {
            tmp[row + i] = acc * inv;
            acc +=
                src[row + w(i as isize + r as isize + 1)] - src[row + w(i as isize - r as isize)];
        }
    }
    let mut dst = vec![0.0f32; src.len()];
    for i in 0..s {
        let mut acc: f32 = (-(r as isize)..=r as isize)
            .map(|k| tmp[w(k) * s + i])
            .sum();
        for j in 0..s {
            dst[j * s + i] = acc * inv;
            acc += tmp[w(j as isize + r as isize + 1) * s + i]
                - tmp[w(j as isize - r as isize) * s + i];
        }
    }
    dst
}

/// Approximate gaussian blur (three box passes), wrapped; sigma in pixels.
pub(crate) fn blur(src: &[f32], s: usize, sigma: f32) -> Vec<f32> {
    let r = ((sigma * 0.95).round() as usize).max(1);
    let a = box_blur(src, s, r);
    let b = box_blur(&a, s, r);
    box_blur(&b, s, r)
}

/// How a layer is packed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Finish {
    /// Normal exaggeration over the physical slope (1 = physical).
    pub normal: f32,
    /// Occlusion strength per metre of cavity depth.
    pub ao_k: f32,
    /// Cavity radii (m), small and large.
    pub ao_r: [f32; 2],
}

/// Packed layer, rows bottom-up (row 0 = v = 0).
pub(crate) struct Packed {
    /// sRGB albedo (rgb), roughness (a).
    pub albedo: Vec<u8>,
    /// Normal xy (rg, +x right, +y up the tile), height (b), occlusion (a).
    pub normal: Vec<u8>,
}

pub(crate) fn finish(t: &Tex, o: Finish) -> Packed {
    let (s, n, h) = (t.size, t.size * t.size, &t.h);
    let mut albedo = vec![0u8; n * 4];
    let mut normal = vec![0u8; n * 4];
    let ns = o.normal / (8.0 * t.ps);
    let b1 = blur(h, s, (o.ao_r[0] / t.ps).max(1.0));
    let b2 = blur(h, s, (o.ao_r[1] / t.ps).max(1.5));
    let (hmin, hmax) = h
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let hs = if hmax > hmin {
        255.0 / (hmax - hmin)
    } else {
        0.0
    };
    let q = |x: f32| (x * 255.0).round().clamp(0.0, 255.0) as u8;
    for j in 0..s {
        let (jm, j0, jp) = (((j + s - 1) % s) * s, j * s, ((j + 1) % s) * s);
        for i in 0..s {
            let (im, ip) = ((i + s - 1) % s, (i + 1) % s);
            let gx = (h[jm + ip] + 2.0 * h[j0 + ip] + h[jp + ip])
                - (h[jm + im] + 2.0 * h[j0 + im] + h[jp + im]);
            let gy = (h[jp + im] + 2.0 * h[jp + i] + h[jp + ip])
                - (h[jm + im] + 2.0 * h[jm + i] + h[jm + ip]);
            let (mut nx, mut ny) = (-gx * ns, -gy * ns);
            let l = 1.0 / (nx * nx + ny * ny + 1.0).sqrt();
            nx *= l;
            ny *= l;
            let idx = j0 + i;
            let o4 = idx * 4;
            let cav = (b1[idx] - h[idx]).max(0.0) * 0.55 + (b2[idx] - h[idx]).max(0.0) * 0.45;
            let ao = (-o.ao_k * cav).exp();
            normal[o4] = q(nx * 0.5 + 0.5);
            normal[o4 + 1] = q(ny * 0.5 + 0.5);
            normal[o4 + 2] = ((h[idx] - hmin) * hs).round().clamp(0.0, 255.0) as u8;
            normal[o4 + 3] = q(clamp01(ao));
            albedo[o4] = q(t.r[idx].clamp(0.025, 0.95));
            albedo[o4 + 1] = q(t.g[idx].clamp(0.025, 0.95));
            albedo[o4 + 2] = q(t.b[idx].clamp(0.025, 0.95));
            albedo[o4 + 3] = q(t.rough[idx].clamp(0.02, 1.0));
        }
    }
    Packed { albedo, normal }
}
