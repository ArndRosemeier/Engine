//! The vegetation's procedural textures, one atlas for every species: a 4x4
//! leaf atlas of twig clusters (the top 1024 rows) above a bark atlas of
//! [`BARK_STRIPS`] strips (the bottom 512 rows), plus the matching normal
//! map (flat under the leaves). A port of AssetGenerator's
//! `packages/vegetation/src/textures.ts`; the leaves are drawn with
//! tiny-skia where the original used a browser canvas.

use super::grow::BARK_STRIPS;
use super::rng::{hash2i, hash_to_float, smoothstep as smooth, Rng};
use std::f32::consts::PI;
use std::sync::OnceLock;
use tiny_skia::{
    Color, FillRule, GradientStop, LineCap, LinearGradient, Mask, Paint, PathBuilder, Pixmap,
    Point, Rect, SpreadMode, Stroke, Transform,
};

const DEG: f32 = PI / 180.0;

/// Atlas size in texels.
pub const ATLAS_W: u32 = 1024;
pub const LEAF_H: u32 = 1024;
pub const BARK_H: u32 = 512;
pub const ATLAS_H: u32 = LEAF_H + BARK_H;

/// Where the leaf atlas sits in the combined atlas (fraction of its height).
pub const LEAF_V_SCALE: f32 = LEAF_H as f32 / ATLAS_H as f32;

pub(crate) struct Atlas {
    /// sRGB colour, straight alpha.
    pub color: Vec<u8>,
    /// Tangent-space normals (glTF convention), alpha 255.
    pub normal: Vec<u8>,
}

/// The shared atlas, built on first use.
pub(crate) fn atlas() -> &'static Atlas {
    static ATLAS: OnceLock<Atlas> = OnceLock::new();
    ATLAS.get_or_init(build)
}

fn build() -> Atlas {
    let (bark_col, bark_nrm) = bark_textures();
    let leaves = leaf_atlas();
    let mut color = leaves;
    color.extend_from_slice(&bark_col);
    let mut normal = Vec::with_capacity((ATLAS_W * ATLAS_H * 4) as usize);
    for _ in 0..ATLAS_W * LEAF_H {
        normal.extend_from_slice(&[128, 128, 255, 255]);
    }
    normal.extend_from_slice(&bark_nrm);
    Atlas { color, normal }
}

// -------------------------------------------------------------------------
// Periodic noise
// -------------------------------------------------------------------------

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn imod(a: i32, b: i32) -> i32 {
    a.rem_euclid(b.max(1))
}

/// Periodic value noise in 0..1 (x period `px`, y period `py`).
fn pnoise(seed: u32, x: f32, y: f32, px: i32, py: i32) -> f32 {
    let (xf, yf) = (x.floor(), y.floor());
    let (fx, fy) = (fade(x - xf), fade(y - yf));
    let (xi, yi) = (xf as i32, yf as i32);
    let h = |i: i32, j: i32| hash_to_float(hash2i(seed, imod(i, px), imod(j, py)));
    let (a, b, c, d) = (h(xi, yi), h(xi + 1, yi), h(xi, yi + 1), h(xi + 1, yi + 1));
    a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
}

fn pfbm(seed: u32, x: f32, y: f32, px: i32, py: i32, oct: u32) -> f32 {
    let (mut s, mut amp, mut norm, mut f) = (0.0, 0.5, 0.0, 1.0);
    for o in 0..oct {
        s += amp
            * pnoise(
                seed + o * 101,
                x * f,
                y * f,
                (px as f32 * f) as i32,
                (py as f32 * f) as i32,
            );
        norm += amp;
        amp *= 0.5;
        f *= 2.0;
    }
    s / norm
}

/// Periodic Worley noise: nearest and second distances, nearest cell id.
fn pworley(seed: u32, x: f32, y: f32, px: i32, py: i32, jitter: f32) -> (f32, f32, u32) {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (mut f1, mut f2, mut id) = (9.0f32, 9.0f32, 0u32);
    for j in -1..=1 {
        for i in -1..=1 {
            let (cx, cy) = (xi + i, yi + j);
            let h = hash2i(seed, imod(cx, px), imod(cy, py));
            let fx = cx as f32 + 0.5 + (hash_to_float(h) - 0.5) * jitter;
            let fy =
                cy as f32 + 0.5 + (hash_to_float(h.wrapping_mul(2_654_435_761)) - 0.5) * jitter;
            let dd = (fx - x).hypot(fy - y);
            if dd < f1 {
                f2 = f1;
                f1 = dd;
                id = h;
            } else if dd < f2 {
                f2 = dd;
            }
        }
    }
    (f1, f2, id)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// -------------------------------------------------------------------------
// Bark
// -------------------------------------------------------------------------

/// Colour and height of a bark strip at (u around, v along), both 0..1.
fn bark_pixel(strip: u32, u: f32, v: f32) -> [f32; 4] {
    match strip {
        0 => {
            // Linden / maple: interlaced shallow furrows.
            let n = pfbm(
                11,
                u * 9.0 + pfbm(12, u * 4.0, v * 3.0, 4, 3, 2) * 1.5,
                v * 5.0,
                9,
                5,
                4,
            );
            let ridge = 1.0 - (2.0 * n - 1.0).abs();
            let fine = pfbm(13, u * 40.0, v * 60.0, 40, 60, 2);
            let h = ridge.powf(1.6) * 0.8 + fine * 0.2;
            let lich = smooth(0.62, 0.75, pfbm(14, u * 3.0, v * 5.0, 3, 5, 3));
            let (r, g, b) = (mix(0.16, 0.42, h), mix(0.14, 0.39, h), mix(0.12, 0.35, h));
            [
                mix(r, 0.42, lich * 0.5),
                mix(g, 0.46, lich * 0.5),
                mix(b, 0.3, lich * 0.5),
                h,
            ]
        }
        1 => {
            // Oak / chestnut: deep blocky fissures.
            let warp_u = pfbm(21, u * 5.0, v * 4.0, 5, 4, 2) * 0.6;
            let (f1, f2, _) = pworley(22, u * 7.0 + warp_u, v * 7.0, 7, 7, 0.85);
            let crack = smooth(0.0, 0.16, f2 - f1);
            let fine = pfbm(23, u * 30.0, v * 50.0, 30, 50, 3);
            let plate = pfbm(24, u * 14.0, v * 12.0, 14, 12, 2);
            let h = crack * (0.65 + 0.35 * plate) * 0.85 + fine * 0.15;
            [
                mix(0.09, 0.36, h),
                mix(0.08, 0.33, h),
                mix(0.07, 0.29, h),
                h,
            ]
        }
        2 => {
            // Birch: white with dark lenticels and black patches.
            let lent = smooth(0.66, 0.72, pfbm(31, u * 7.0, v * 140.0, 7, 140, 2));
            let patch = smooth(
                0.66,
                0.7,
                pfbm(
                    32,
                    u * 3.0 + pfbm(35, u * 6.0, v * 20.0, 6, 20, 2) * 0.8,
                    v * 9.0,
                    3,
                    9,
                    4,
                ),
            );
            let fine = pfbm(33, u * 24.0, v * 90.0, 24, 90, 2);
            let warm = pfbm(34, u * 4.0, v * 10.0, 4, 10, 2);
            let (r, g, b) = (
                0.86 - fine * 0.1 + warm * 0.04,
                0.85 - fine * 0.1,
                0.8 - fine * 0.1 - warm * 0.04,
            );
            let dark = (lent * 0.75).max(patch);
            [
                mix(r, 0.12, dark),
                mix(g, 0.11, dark),
                mix(b, 0.11, dark),
                0.7 - dark * 0.45 + fine * 0.15,
            ]
        }
        3 => {
            // Plane: camouflage patches.
            let wu = u * 4.0 + (pfbm(41, u * 6.0, v * 12.0, 6, 12, 3) - 0.5) * 1.3;
            let wv = v * 9.0 + (pfbm(42, u * 6.0, v * 12.0, 6, 12, 3) - 0.5) * 2.6;
            let (f1, f2, id) = pworley(43, wu, wv, 4, 9, 0.95);
            const PAL: [[f32; 3]; 6] = [
                [0.5, 0.5, 0.38],
                [0.8, 0.77, 0.62],
                [0.55, 0.5, 0.4],
                [0.66, 0.67, 0.53],
                [0.42, 0.44, 0.34],
                [0.74, 0.72, 0.6],
            ];
            let c = PAL[(id % 6) as usize];
            let edge = smooth(0.0, 0.07, f2 - f1);
            let fine = pfbm(44, u * 30.0, v * 60.0, 30, 60, 2);
            let k = (0.85 + 0.15 * edge) * (0.92 + fine * 0.16);
            [
                c[0] * k,
                c[1] * k,
                c[2] * k,
                0.5 + (id % 3) as f32 * 0.08 + fine * 0.1 + edge * 0.1,
            ]
        }
        4 => {
            // Pine: reddish plates.
            let (f1, f2, id) = pworley(
                51,
                u * 5.0 + pfbm(52, u * 5.0, v * 5.0, 5, 5, 2) * 0.5,
                v * 8.0,
                5,
                8,
                0.9,
            );
            let crack = smooth(0.0, 0.1, f2 - f1);
            let fine = pfbm(53, u * 30.0, v * 40.0, 30, 40, 2);
            let red = hash_to_float(id) * 0.5 + 0.5;
            let h = crack * (0.7 + fine * 0.3);
            [
                mix(0.12, 0.5 * red + 0.12, h),
                mix(0.09, 0.3 * red + 0.07, h),
                mix(0.08, 0.2 * red + 0.06, h),
                h,
            ]
        }
        5 => {
            // Palm: horizontal leaf-scar rings, fibrous.
            let ring_v = v * 22.0 + (pfbm(61, u * 3.0, v * 22.0, 3, 22, 2) - 0.5) * 0.5;
            let fr = ring_v - ring_v.floor();
            let ring = smooth(0.0, 0.18, fr) * smooth(1.0, 0.75, fr);
            let fib = pfbm(62, u * 60.0, v * 22.0, 60, 22, 2);
            let h = ring * 0.75 + fib * 0.25;
            [
                mix(0.22, 0.52, h),
                mix(0.19, 0.47, h),
                mix(0.15, 0.39, h),
                h,
            ]
        }
        6 => {
            // Cypress: stringy fibrous strips.
            let n = pfbm(
                71,
                u * 26.0 + pfbm(72, u * 4.0, v * 3.0, 4, 3, 2) * 3.0,
                v * 3.0,
                26,
                3,
                3,
            );
            let ridge = 1.0 - (2.0 * n - 1.0).abs();
            let g2 = pfbm(73, u * 5.0, v * 8.0, 5, 8, 2);
            let h = ridge.powf(1.3);
            [
                mix(0.16, 0.46 - g2 * 0.1, h),
                mix(0.11, 0.33, h),
                mix(0.09, 0.27 + g2 * 0.05, h),
                h,
            ]
        }
        _ => {
            // Ginkgo / shrub: grey, fine furrows.
            let n = pfbm(81, u * 14.0, v * 7.0, 14, 7, 4);
            let ridge = 1.0 - (2.0 * n - 1.0).abs();
            let fine = pfbm(82, u * 40.0, v * 60.0, 40, 60, 2);
            let h = ridge.powi(2) * 0.7 + fine * 0.3;
            [mix(0.2, 0.43, h), mix(0.18, 0.4, h), mix(0.16, 0.36, h), h]
        }
    }
}

/// sRGB-encodes a linear-ish value the way the original's canvas did (it
/// wrote the values straight into an sRGB texture).
fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0) as u8
}

fn bark_textures() -> (Vec<u8>, Vec<u8>) {
    let sw = ATLAS_W / BARK_STRIPS;
    let (w, sh) = (ATLAS_W as usize, BARK_H as usize);
    let mut col = vec![0u8; w * sh * 4];
    let mut hgt = vec![0f32; w * sh];
    for s in 0..BARK_STRIPS {
        for y in 0..sh {
            for x in 0..sw as usize {
                let px = bark_pixel(s, x as f32 / sw as f32, y as f32 / sh as f32);
                let i = y * w + s as usize * sw as usize + x;
                col[i * 4] = byte(px[0]);
                col[i * 4 + 1] = byte(px[1]);
                col[i * 4 + 2] = byte(px[2]);
                col[i * 4 + 3] = 255;
                hgt[i] = px[3];
            }
        }
    }
    const STRENGTH: [f32; 8] = [3.5, 5.0, 1.2, 1.2, 4.0, 3.0, 3.5, 2.5];
    let mut nor = vec![0u8; w * sh * 4];
    let sw = sw as i32;
    for s in 0..BARK_STRIPS as i32 {
        for y in 0..sh as i32 {
            for x in 0..sw {
                let at = |xx: i32, yy: i32| {
                    hgt[(imod(yy, sh as i32) as usize) * w + (s * sw + imod(xx, sw)) as usize]
                };
                let k = STRENGTH[s as usize];
                let dx = (at(x + 1, y) - at(x - 1, y)) * k;
                let dy = (at(x, y + 1) - at(x, y - 1)) * k;
                // Tangent space with +Y up the image (glTF): rows run down +v.
                let len = (dx * dx + dy * dy + 1.0).sqrt();
                let i = ((y as usize) * w + (s * sw + x) as usize) * 4;
                nor[i] = byte((-dx / len) * 0.5 + 0.5);
                nor[i + 1] = byte((dy / len) * 0.5 + 0.5);
                nor[i + 2] = byte((1.0 / len) * 0.5 + 0.5);
                nor[i + 3] = 255;
            }
        }
    }
    (col, nor)
}

// -------------------------------------------------------------------------
// A tiny canvas over tiny-skia
// -------------------------------------------------------------------------

type Pt = (f32, f32);
/// Hue (degrees), saturation and lightness (percent).
type Hsl = [f32; 3];

fn hsla(c: Hsl, a: f32, dl: f32) -> Color {
    let (h, s, l) = (
        c[0].rem_euclid(360.0) / 360.0,
        (c[1] / 100.0).clamp(0.0, 1.0),
        ((c[2] + dl) / 100.0).clamp(0.0, 1.0),
    );
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue = |mut t: f32| {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    Color::from_rgba(
        hue(h + 1.0 / 3.0),
        hue(h),
        hue(h - 1.0 / 3.0),
        a.clamp(0.0, 1.0),
    )
    .unwrap_or(Color::BLACK)
}

#[derive(Clone)]
struct State {
    ts: Transform,
    clip: Option<Mask>,
}

struct Canvas {
    pix: Pixmap,
    st: State,
    stack: Vec<State>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Self {
        Self {
            pix: Pixmap::new(w, h).expect("atlas pixmap"),
            st: State {
                ts: Transform::identity(),
                clip: None,
            },
            stack: Vec::new(),
        }
    }
    fn save(&mut self) {
        self.stack.push(self.st.clone());
    }
    fn restore(&mut self) {
        if let Some(s) = self.stack.pop() {
            self.st = s;
        }
    }
    fn translate(&mut self, x: f32, y: f32) {
        self.st.ts = self.st.ts.pre_translate(x, y);
    }
    fn rotate(&mut self, a: f32) {
        self.st.ts = self.st.ts.pre_concat(Transform::from_rotate(a / DEG));
    }
    fn paint(color: Color) -> Paint<'static> {
        let mut p = Paint::default();
        p.set_color(color);
        p.anti_alias = true;
        p
    }
    fn poly_path(pts: &[Pt]) -> Option<tiny_skia::Path> {
        let mut pb = PathBuilder::new();
        pb.move_to(pts.first()?.0, pts[0].1);
        for p in &pts[1..] {
            pb.line_to(p.0, p.1);
        }
        pb.close();
        pb.finish()
    }
    fn fill_path(&mut self, path: &tiny_skia::Path, color: Color) {
        let ts = self.st.ts;
        self.pix.fill_path(
            path,
            &Self::paint(color),
            FillRule::Winding,
            ts,
            self.st.clip.as_ref(),
        );
    }
    fn stroke_path(&mut self, path: &tiny_skia::Path, color: Color, width: f32) {
        let stroke = Stroke {
            width,
            line_cap: LineCap::Round,
            ..Stroke::default()
        };
        let ts = self.st.ts;
        self.pix.stroke_path(
            path,
            &Self::paint(color),
            &stroke,
            ts,
            self.st.clip.as_ref(),
        );
    }
    fn line(&mut self, a: Pt, b: Pt, color: Color, width: f32) {
        let mut pb = PathBuilder::new();
        pb.move_to(a.0, a.1);
        pb.line_to(b.0, b.1);
        if let Some(p) = pb.finish() {
            self.stroke_path(&p, color, width);
        }
    }
    fn clip_path(&mut self, path: &tiny_skia::Path) {
        let ts = self.st.ts;
        match &mut self.st.clip {
            Some(m) => m.intersect_path(path, FillRule::Winding, true, ts),
            None => {
                let mut m = Mask::new(self.pix.width(), self.pix.height()).expect("mask");
                m.fill_path(path, FillRule::Winding, true, ts);
                self.st.clip = Some(m);
            }
        }
    }
    fn clip_rect(&mut self, x: f32, y: f32, w: f32, h: f32) {
        if let Some(r) = Rect::from_xywh(x, y, w, h) {
            self.clip_path(&PathBuilder::from_rect(r));
        }
    }
    /// Fill a rectangle with a linear gradient between two points.
    fn gradient_rect(
        &mut self,
        rect: (f32, f32, f32, f32),
        from: Pt,
        to: Pt,
        stops: &[(f32, Color)],
    ) {
        let Some(r) = Rect::from_xywh(rect.0, rect.1, rect.2, rect.3) else {
            return;
        };
        let stops = stops
            .iter()
            .map(|(t, c)| GradientStop::new(*t, *c))
            .collect();
        let Some(shader) = LinearGradient::new(
            Point::from_xy(from.0, from.1),
            Point::from_xy(to.0, to.1),
            stops,
            SpreadMode::Pad,
            Transform::identity(),
        ) else {
            return;
        };
        let paint = Paint {
            shader,
            anti_alias: true,
            ..Paint::default()
        };
        let ts = self.st.ts;
        self.pix.fill_path(
            &PathBuilder::from_rect(r),
            &paint,
            FillRule::Winding,
            ts,
            self.st.clip.as_ref(),
        );
    }
}

fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color {
    Color::from_rgba8(r, g, b, (a * 255.0) as u8)
}

// -------------------------------------------------------------------------
// Leaf shapes
// -------------------------------------------------------------------------

/// Polygon for an elongated leaf: base (0,0), tip (0,-len).
fn leaf_poly(len: f32, width: f32, hw: fn(f32) -> f32, teeth: f32, tooth_amp: f32) -> Vec<Pt> {
    const N: usize = 36;
    let right: Vec<Pt> = (0..=N)
        .map(|i| {
            let t = i as f32 / N as f32;
            let mut w = hw(t) * width * 0.5;
            if teeth > 0.0 {
                w *= 1.0 + tooth_amp * ((t * teeth).fract() - 0.5);
            }
            (w, -t * len)
        })
        .collect();
    let mut out = right.clone();
    out.extend(right[..N].iter().rev().map(|&(x, y)| (-x, y)));
    out
}

/// Palmate polygon centred on the petiole, lobes pointing up (angle 0).
fn palm_poly(
    r: f32,
    lobes: &[(f32, f32)],
    lobe_w: f32,
    base: f32,
    sharp: f32,
    teeth: f32,
) -> Vec<Pt> {
    const N: usize = 120;
    (0..N)
        .map(|i| {
            let phi = -PI + (i as f32 / N as f32) * PI * 2.0;
            let mut rr = base;
            for &(a, l) in lobes {
                let mut dd = (phi - a).abs();
                dd = dd.min(PI * 2.0 - dd);
                let k = (1.0 - dd / lobe_w).max(0.0);
                rr = rr.max(base + (l - base) * k.powf(sharp));
            }
            let bottom = (phi.abs() - PI).abs();
            if bottom < 0.5 {
                rr *= 0.35 + 0.65 * (bottom / 0.5);
            }
            rr *= 1.0 + 0.035 * (phi * teeth).sin();
            (phi.sin() * rr * r, -phi.cos() * rr * r)
        })
        .collect()
}

type Vein = (Pt, Pt);

fn fill_leaf(g: &mut Canvas, poly: &[Pt], col: Hsl, len: f32, veins: &[Vein], rng: &mut Rng) {
    let Some(path) = Canvas::poly_path(poly) else {
        return;
    };
    let lgt = rng.range(-5.0, 6.0);
    g.fill_path(&path, hsla(col, 1.0, lgt));
    g.save();
    g.clip_path(&path);
    // Fold shading across the midrib and a base-to-tip gradient.
    let max_x = poly.iter().fold(1.0f32, |m, p| m.max(p.0.abs()));
    let rect = (-max_x, -len * 1.5, max_x * 2.0, len * 2.0);
    g.gradient_rect(
        rect,
        (-max_x, 0.0),
        (max_x, 0.0),
        &[
            (0.0, rgba(0, 0, 0, 0.22)),
            (0.48, rgba(0, 0, 0, 0.0)),
            (0.52, rgba(255, 255, 230, 0.06)),
            (1.0, rgba(255, 255, 220, 0.0)),
        ],
    );
    g.gradient_rect(
        rect,
        (0.0, 0.0),
        (0.0, -len),
        &[
            (0.0, rgba(0, 0, 0, 0.12)),
            (0.5, rgba(255, 255, 200, 0.04)),
            (1.0, rgba(0, 0, 0, 0.08)),
        ],
    );
    let vein_col = hsla(col, 0.5, 9.0);
    for &(a, b) in veins {
        g.line(a, b, vein_col, (len * 0.012).max(0.6));
    }
    g.restore();
    g.stroke_path(&path, hsla(col, 0.55, -10.0), 0.9);
}

fn midrib_veins(len: f32, width: f32, n: u32, hw: fn(f32) -> f32) -> Vec<Vein> {
    let mut out = vec![((0.0, 0.0), (0.0, -len * 0.96))];
    for i in 1..=n {
        let t = i as f32 / (n + 1) as f32;
        let w = hw((t + 0.1).min(1.0)) * width * 0.45;
        out.push(((0.0, -t * len), (w, -(t + 0.12) * len)));
        out.push(((0.0, -t * len), (-w, -(t + 0.12) * len)));
    }
    out
}

fn with_jitter(c: Hsl, rng: &mut Rng, dh: f32, ds: f32, dl: f32) -> Hsl {
    [
        c[0] + rng.range(-dh, dh),
        (c[1] + rng.range(-ds, ds)).clamp(0.0, 100.0),
        c[2] + rng.range(-dl, dl),
    ]
}

fn jit(c: Hsl, rng: &mut Rng) -> Hsl {
    with_jitter(c, rng, 6.0, 6.0, 4.0)
}

/// A leaf drawer: colour jittered per leaf, size `s`.
#[derive(Clone)]
enum Leaf {
    Oval {
        col: Hsl,
        len: f32,
        width: f32,
        hw: fn(f32) -> f32,
        teeth: f32,
        tooth_amp: f32,
        veins: u32,
    },
    Palmate {
        col: Hsl,
        r: f32,
        lobes: Vec<(f32, f32)>,
        lobe_w: f32,
        base: f32,
        sharp: f32,
        teeth: f32,
    },
}

impl Leaf {
    fn draw(&self, g: &mut Canvas, s: f32, rng: &mut Rng) {
        match self {
            Leaf::Oval {
                col,
                len,
                width,
                hw,
                teeth,
                tooth_amp,
                veins,
            } => {
                let col = jit(*col, rng);
                let (l, w) = (len * s, width * s);
                fill_leaf(
                    g,
                    &leaf_poly(l, w, *hw, *teeth, *tooth_amp),
                    col,
                    l,
                    &midrib_veins(l, w, *veins, *hw),
                    rng,
                );
            }
            Leaf::Palmate {
                col,
                r,
                lobes,
                lobe_w,
                base,
                sharp,
                teeth,
            } => {
                let col = jit(*col, rng);
                let r = r * s;
                let poly = palm_poly(r, lobes, *lobe_w, *base, *sharp, *teeth);
                let veins: Vec<Vein> = lobes
                    .iter()
                    .map(|&(a, l)| ((0.0, 0.0), (a.sin() * l * r * 0.9, -a.cos() * l * r * 0.9)))
                    .collect();
                fill_leaf(g, &poly, col, r * 1.6, &veins, rng);
            }
        }
    }
}

struct Twig {
    nodes: u32,
    twig_len: f32,
    bend: f32,
    spread: (f32, f32),
    opposite: bool,
    petiole: f32,
    size: (f32, f32),
    leaf: Leaf,
    twig_w: f32,
    tip: bool,
    start_t: f32,
    twig_col: Hsl,
}

const TWIG: Twig = Twig {
    nodes: 5,
    twig_len: 0.5,
    bend: 0.0,
    spread: (40.0, 70.0),
    opposite: false,
    petiole: 0.0,
    size: (0.8, 1.0),
    leaf: Leaf::Oval {
        col: [90.0, 40.0, 30.0],
        len: 30.0,
        width: 17.0,
        hw: ov,
        teeth: 0.0,
        tooth_amp: 0.0,
        veins: 2,
    },
    twig_w: 3.0,
    tip: true,
    start_t: 0.15,
    twig_col: [28.0, 22.0, 26.0],
};

fn ov(t: f32) -> f32 {
    (PI * t.powf(0.75)).sin().max(0.0).powf(0.8)
}

/// A twig with leaves inside a cell. Origin: bottom centre of the cell.
fn draw_twig(g: &mut Canvas, rng: &mut Rng, o: &Twig) {
    const C: f32 = 256.0;
    const N: usize = 20;
    let twig: Vec<Pt> = (0..=N)
        .map(|i| {
            let t = i as f32 / N as f32;
            (
                o.bend * (t * PI * 0.9).sin() * C + (t * 9.0).sin() * 3.0,
                -t * o.twig_len * C,
            )
        })
        .collect();
    let twig_col = hsla(o.twig_col, 1.0, 0.0);
    for i in 0..N {
        g.line(
            twig[i],
            twig[i + 1],
            twig_col,
            o.twig_w * (1.0 - (i as f32 / N as f32) * 0.6),
        );
    }
    let at = |t: f32| -> (Pt, f32) {
        let f = t * N as f32;
        let i = (f.floor() as usize).min(N - 1);
        let k = f - i as f32;
        let p = (
            twig[i].0 + (twig[i + 1].0 - twig[i].0) * k,
            twig[i].1 + (twig[i + 1].1 - twig[i].1) * k,
        );
        let ang = (twig[i + 1].0 - twig[i].0).atan2(-(twig[i + 1].1 - twig[i].1));
        (p, ang)
    };
    let mut to_draw: Vec<(Pt, f32, f32)> = Vec::new();
    for k in 0..o.nodes {
        let denom = (o.nodes as i32 - if o.tip { 0 } else { 1 }).max(1) as f32;
        let t = o.start_t + (1.0 - o.start_t) * (k as f32 / denom) * if o.tip { 0.92 } else { 1.0 };
        let (p, ta) = at(t.min(1.0));
        let sides: &[f32] = if o.opposite {
            &[-1.0, 1.0]
        } else if k % 2 == 0 {
            &[-1.0]
        } else {
            &[1.0]
        };
        for &sd in sides {
            let a = ta + sd * rng.range(o.spread.0, o.spread.1) * DEG;
            let sz = rng.range(o.size.0, o.size.1) * (0.75 + 0.35 * (PI * (0.25 + 0.6 * t)).sin());
            to_draw.push((p, a, sz));
        }
    }
    if o.tip {
        let (p, ta) = at(1.0);
        to_draw.push((p, ta + rng.range(-8.0, 8.0) * DEG, o.size.1));
    }
    for (p, a, s) in to_draw {
        g.save();
        g.translate(p.0, p.1);
        g.rotate(a);
        if o.petiole > 0.0 {
            g.line(
                (0.0, 0.0),
                (0.0, -o.petiole * s),
                hsla([70.0, 30.0, 30.0], 1.0, 0.0),
                1.6,
            );
            g.translate(0.0, -o.petiole * s);
        }
        o.leaf.draw(g, s, rng);
        g.restore();
    }
}

fn draw_needle_tuft(
    g: &mut Canvas,
    rng: &mut Rng,
    col: Hsl,
    len: f32,
    n_needle: u32,
    twig_len: f32,
) {
    g.line(
        (0.0, 0.0),
        (0.0, -twig_len),
        hsla([25.0, 30.0, 24.0], 1.0, 0.0),
        4.0,
    );
    let mut needles: Vec<(f32, f32, f32, f32, Hsl)> = (0..n_needle)
        .map(|_| {
            let t = rng.float().powf(0.7);
            let y = -twig_len * (0.15 + 0.85 * t);
            let a = rng.sign() * rng.range(12.0, 62.0) * DEG * (1.0 - 0.5 * t);
            let x = rng.range(-2.0, 2.0);
            let l = len * rng.range(0.7, 1.05);
            (x, y, a, l, with_jitter(col, rng, 8.0, 8.0, 7.0))
        })
        .collect();
    needles.sort_by(|a, b| a.3.total_cmp(&b.3));
    for (x, y, a, l, c) in needles {
        let mut pb = PathBuilder::new();
        pb.move_to(x, y);
        let (ex, ey) = (x + a.sin() * l, y - a.cos() * l);
        pb.quad_to(x + a.sin() * l * 0.5, y - a.cos() * l * 0.55 - 3.0, ex, ey);
        if let Some(p) = pb.finish() {
            let w = rng.range(1.6, 2.4);
            g.stroke_path(&p, hsla(c, 1.0, 0.0), w);
            g.stroke_path(&p, hsla(c, 0.6, 12.0), 0.7);
        }
    }
}

fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32, rot: f32) -> Vec<Pt> {
    let (s, c) = rot.sin_cos();
    (0..16)
        .map(|i| {
            let a = i as f32 / 16.0 * PI * 2.0;
            let (x, y) = (a.cos() * rx, a.sin() * ry);
            (cx + x * c - y * s, cy + x * s + y * c)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn draw_cypress_spray(
    g: &mut Canvas,
    rng: &mut Rng,
    x: f32,
    y: f32,
    ang: f32,
    len: f32,
    depth: u32,
    col: Hsl,
) {
    let steps = ((len / 5.0).round() as u32).max(3);
    let (mut px, mut py) = (x, y);
    for i in 0..steps {
        let t = i as f32 / steps as f32;
        let nx = px + ang.sin() * (len / steps as f32);
        let ny = py - ang.cos() * (len / steps as f32);
        let w = (2.2 + depth as f32 * 1.5) * (1.0 - t * 0.5);
        let c = with_jitter(col, rng, 5.0, 5.0, 5.0);
        if let Some(p) = Canvas::poly_path(&ellipse(nx, ny, w, w * 1.6, ang)) {
            g.fill_path(&p, hsla(c, 1.0, t * 8.0));
        }
        if depth > 0 && i > 0 && i % 2 == 0 {
            let sd = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
            let a = ang + sd * rng.range(35.0, 55.0) * DEG;
            let l = len * rng.range(0.35, 0.55) * (1.0 - t * 0.6);
            draw_cypress_spray(g, rng, nx, ny, a, l, depth - 1, col);
        }
        px = nx;
        py = ny;
    }
}

fn draw_frond(g: &mut Canvas, rng: &mut Rng, w: f32, h: f32, col: Hsl, dead: bool) {
    let cy = h / 2.0;
    const NL: usize = 62;
    let mut leaflets = Vec::new();
    for i in 0..NL {
        let t = (i as f32 + 0.5) / NL as f32;
        let x = 6.0 + t * (w - 12.0);
        let l = (h / 2.0 - 6.0)
            * (PI * (0.12 + 0.88 * t)).sin().max(0.0).powf(0.55)
            * if t < 0.06 { 0.4 } else { 1.0 }
            * rng.range(0.85, 1.0);
        for side in [-1.0f32, 1.0] {
            let a = if dead {
                rng.range(20.0, 50.0)
            } else {
                rng.range(42.0, 58.0)
            } * DEG;
            leaflets.push((x, side, l, a, with_jitter(col, rng, 5.0, 6.0, 5.0)));
        }
    }
    let leaflet_hw: fn(f32) -> f32 = |t| (PI * (t * 1.05).min(1.0)).sin().max(0.0).powf(0.6);
    for (x, side, l, a, c) in leaflets {
        let (ex, ey) = (x + a.cos() * l, cy + side * a.sin() * l);
        g.save();
        g.translate(x, cy);
        g.rotate((ey - cy).atan2(ex - x) + PI / 2.0);
        let len = (ex - x).hypot(ey - cy);
        if let Some(p) = Canvas::poly_path(&leaf_poly(
            len,
            if dead { 5.0 } else { 7.5 },
            leaflet_hw,
            0.0,
            0.0,
        )) {
            g.fill_path(&p, hsla(c, 1.0, 0.0));
        }
        g.line((0.0, 0.0), (0.0, -len), hsla(c, 0.7, 10.0), 0.8);
        g.restore();
    }
    let rib = if dead {
        [30.0, 25.0, 40.0]
    } else {
        [65.0, 30.0, 42.0]
    };
    g.line((0.0, cy), (w, cy), hsla(rib, 1.0, 0.0), 4.0);
}

fn leaf_atlas() -> Vec<u8> {
    const S: u32 = LEAF_H;
    const C: f32 = 256.0;
    let mut g = Canvas::new(ATLAS_W, S);
    let rng = Rng::from(&["leaf-atlas"], 0);
    let cell = |g: &mut Canvas, i: u32, draw: &mut dyn FnMut(&mut Canvas)| {
        let (cx, cy) = ((i % 4) as f32 * C, (i / 4) as f32 * C);
        g.save();
        g.clip_rect(cx + 3.0, cy + 3.0, C - 6.0, C - 6.0);
        g.translate(cx + C / 2.0, cy + C - 4.0);
        draw(g);
        g.restore();
    };
    let lobes5 = |a1: f32, l1: f32, a2: f32, l2: f32| {
        vec![(0.0, 1.15), (-a1, l1), (a1, l1), (-a2, l2), (a2, l2)]
    };

    // 0 plane: large palmate, shallow lobes.
    let mut r = rng.fork("plane");
    cell(&mut g, 0, &mut |g| {
        draw_twig(
            g,
            &mut r,
            &Twig {
                nodes: 4,
                twig_len: 0.42,
                bend: 0.03,
                spread: (40.0, 70.0),
                petiole: 26.0,
                size: (0.85, 1.0),
                twig_w: 5.0,
                start_t: 0.35,
                leaf: Leaf::Palmate {
                    col: [88.0, 42.0, 32.0],
                    r: 46.0,
                    lobes: lobes5(0.95, 1.0, 1.9, 0.62),
                    lobe_w: 0.62,
                    base: 0.55,
                    sharp: 1.3,
                    teeth: 26.0,
                },
                ..TWIG
            },
        )
    });
    // 1 linden: heart leaves.
    let mut r = rng.fork("linden");
    cell(&mut g, 1, &mut |g| {
        draw_twig(
            g,
            &mut r,
            &Twig {
                nodes: 7,
                twig_len: 0.8,
                bend: 0.06,
                spread: (45.0, 80.0),
                petiole: 12.0,
                size: (0.75, 1.0),
                twig_w: 3.5,
                leaf: Leaf::Oval {
                    col: [88.0, 44.0, 28.0],
                    len: 62.0,
                    width: 60.0,
                    hw: |t| (PI * (0.08 + 0.92 * t).powf(0.7)).sin().max(0.0).powf(0.75),
                    teeth: 22.0,
                    tooth_amp: 0.07,
                    veins: 4,
                },
                ..TWIG
            },
        )
    });
    // 2 maple: sharp 5-lobed, opposite.
    let mut r = rng.fork("maple");
    cell(&mut g, 2, &mut |g| {
        draw_twig(
            g,
            &mut r,
            &Twig {
                nodes: 3,
                twig_len: 0.45,
                bend: 0.02,
                spread: (50.0, 75.0),
                opposite: true,
                petiole: 30.0,
                size: (0.8, 1.0),
                twig_w: 4.5,
                start_t: 0.3,
                leaf: Leaf::Palmate {
                    col: [96.0, 48.0, 28.0],
                    r: 40.0,
                    lobes: vec![
                        (0.0, 1.15),
                        (-0.85, 1.05),
                        (0.85, 1.05),
                        (-1.75, 0.7),
                        (1.75, 0.7),
                    ],
                    lobe_w: 0.42,
                    base: 0.38,
                    sharp: 1.9,
                    teeth: 14.0,
                },
                ..TWIG
            },
        )
    });
    // 3 oak: lobed leaves clustered near the tip.
    let mut r = rng.fork("oak");
    cell(&mut g, 3, &mut |g| {
        draw_twig(
            g,
            &mut r,
            &Twig {
                nodes: 7,
                twig_len: 0.5,
                bend: 0.04,
                spread: (15.0, 75.0),
                petiole: 4.0,
                size: (0.8, 1.0),
                twig_w: 4.5,
                start_t: 0.55,
                leaf: Leaf::Oval {
                    col: [82.0, 38.0, 25.0],
                    len: 92.0,
                    width: 48.0,
                    hw: |t| {
                        (PI * t.powf(0.85)).sin().max(0.0).powf(0.7)
                            * (0.55 + 0.45 * (t * PI * 4.5).sin().abs().powf(0.6))
                    },
                    teeth: 0.0,
                    tooth_amp: 0.0,
                    veins: 4,
                },
                ..TWIG
            },
        )
    });
    // 4 birch: small serrated leaves on drooping twigs.
    let mut r = rng.fork("birch");
    cell(&mut g, 4, &mut |g| {
        for (dx, bend, len) in [(-50.0, -0.12, 0.85), (45.0, 0.1, 0.8), (0.0, 0.02, 0.95)] {
            g.save();
            g.translate(dx, 0.0);
            draw_twig(
                g,
                &mut r,
                &Twig {
                    nodes: 7,
                    twig_len: len,
                    bend,
                    spread: (40.0, 70.0),
                    petiole: 8.0,
                    size: (0.75, 1.0),
                    twig_w: 2.0,
                    twig_col: [20.0, 25.0, 22.0],
                    leaf: Leaf::Oval {
                        col: [84.0, 46.0, 31.0],
                        len: 36.0,
                        width: 26.0,
                        hw: |t| (PI * t.powf(0.55)).sin().max(0.0).powf(0.9),
                        teeth: 14.0,
                        tooth_amp: 0.14,
                        veins: 3,
                    },
                    ..TWIG
                },
            );
            g.restore();
        }
    });
    // 5 chestnut: palmate compound leaves (7 leaflets).
    let mut r = rng.fork("chestnut");
    cell(&mut g, 5, &mut |g| {
        let leaflet_hw: fn(f32) -> f32 = |t| (PI * t.powf(1.35)).sin().max(0.0).powf(0.75);
        let mut compound = |g: &mut Canvas, x: f32, y: f32, a: f32, s: f32| {
            g.save();
            g.translate(x, y);
            g.rotate(a);
            g.line(
                (0.0, 0.0),
                (0.0, -38.0 * s),
                hsla([60.0, 30.0, 32.0], 1.0, 0.0),
                2.4,
            );
            g.translate(0.0, -38.0 * s);
            for i in 0..7 {
                let la = (-1.0 + (2.0 * i as f32) / 6.0) * 1.45;
                let ll = (1.0 - 0.45 * la.abs() / 1.45) * 78.0 * s;
                g.save();
                g.rotate(la);
                Leaf::Oval {
                    col: [90.0, 44.0, 27.0],
                    len: ll,
                    width: ll * 0.42,
                    hw: leaflet_hw,
                    teeth: 24.0,
                    tooth_amp: 0.08,
                    veins: 6,
                }
                .draw(g, 1.0, &mut r);
                g.restore();
            }
            g.restore();
        };
        g.line(
            (0.0, 0.0),
            (0.0, -70.0),
            hsla([28.0, 22.0, 26.0], 1.0, 0.0),
            5.0,
        );
        compound(g, 0.0, -60.0, -0.7, 0.9);
        compound(g, 0.0, -70.0, 0.75, 0.88);
        compound(g, 0.0, -70.0, 0.02, 1.0);
    });
    // 6 ginkgo: fan leaves on spur shoots.
    let mut r = rng.fork("ginkgo");
    cell(&mut g, 6, &mut |g| {
        let fan = |g: &mut Canvas, s: f32, rr: &mut Rng| {
            let rad = 34.0 * s;
            let mut pts: Vec<Pt> = vec![(0.0, 0.0)];
            for i in 0..=30 {
                let a = (-58.0 + (116.0 * i as f32) / 30.0) * DEG;
                let mut rd = rad * (1.0 + 0.04 * (i as f32 * 1.7).sin());
                if a.abs() < 0.12 {
                    rd *= 0.78 + 0.22 * (a.abs() / 0.12);
                }
                pts.push((a.sin() * rd, -a.cos() * rd));
            }
            let veins: Vec<Vein> = (0..9)
                .map(|i| {
                    let a = (-50.0 + (100.0 * i as f32) / 8.0) * DEG;
                    ((0.0, 0.0), (a.sin() * rad * 0.95, -a.cos() * rad * 0.95))
                })
                .collect();
            let col = jit([70.0, 52.0, 40.0], rr);
            fill_leaf(g, &pts, col, rad, &veins, rr);
        };
        let stem = hsla([28.0, 15.0, 32.0], 1.0, 0.0);
        g.line((0.0, 0.0), (6.0, -120.0), stem, 5.0);
        g.line((6.0, -120.0), (2.0, -190.0), stem, 5.0);
        for (sx, sy, side) in [
            (2.0, -60.0, -1.0f32),
            (5.0, -110.0, 1.0),
            (4.0, -160.0, -1.0),
            (2.0, -195.0, 0.0),
        ] {
            let n = r.int(4, 5);
            for k in 0..n {
                let a = side * 0.7
                    + (-0.9 + (1.8 * k as f32) / (n - 1) as f32)
                        * if side == 0.0 { 1.0 } else { 0.8 };
                g.save();
                g.translate(sx, sy);
                g.rotate(a);
                g.line(
                    (0.0, 0.0),
                    (0.0, -26.0),
                    hsla([70.0, 30.0, 35.0], 1.0, 0.0),
                    1.4,
                );
                g.translate(0.0, -26.0);
                let s = r.range(0.85, 1.1);
                fan(g, s, &mut r);
                g.restore();
            }
        }
    });
    // 7 shrub (deciduous): dense small leaves on 3 twigs.
    let mut r = rng.fork("shrub");
    cell(&mut g, 7, &mut |g| {
        for (dx, bend, len) in [(-40.0, -0.1, 0.8), (40.0, 0.1, 0.78), (0.0, 0.0, 0.92)] {
            g.save();
            g.translate(dx, 0.0);
            draw_twig(
                g,
                &mut r,
                &Twig {
                    nodes: 10,
                    twig_len: len,
                    bend,
                    spread: (35.0, 75.0),
                    petiole: 3.0,
                    size: (0.8, 1.0),
                    twig_w: 2.5,
                    leaf: Leaf::Oval {
                        col: [92.0, 38.0, 29.0],
                        len: 30.0,
                        width: 17.0,
                        hw: ov,
                        teeth: 0.0,
                        tooth_amp: 0.0,
                        veins: 2,
                    },
                    ..TWIG
                },
            );
            g.restore();
        }
    });
    // 8 pine needle tufts (umbrella pine).
    let mut r = rng.fork("pine");
    cell(&mut g, 8, &mut |g| {
        for (dx, a, s) in [(-45.0, -0.35, 0.85), (45.0, 0.35, 0.85), (0.0, 0.0, 1.0)] {
            g.save();
            g.translate(dx * 0.6, 0.0);
            g.rotate(a);
            draw_needle_tuft(g, &mut r, [118.0, 28.0, 25.0], 95.0 * s, 170, 150.0 * s);
            g.restore();
        }
    });
    // 9 cypress scale-leaf sprays.
    let mut r = rng.fork("cypress");
    cell(&mut g, 9, &mut |g| {
        for (dx, a) in [
            (-30.0, -0.35),
            (30.0, 0.35),
            (0.0, 0.0),
            (-10.0, -0.12),
            (12.0, 0.15),
        ] {
            let len = r.range(170.0, 230.0);
            draw_cypress_spray(g, &mut r, dx, -4.0, a, len, 2, [118.0, 30.0, 16.0]);
        }
    });
    // 10 spruce / conical pine branchlet (flat, comb needles).
    let mut r = rng.fork("spruce");
    cell(&mut g, 10, &mut |g| {
        fn axis(g: &mut Canvas, r: &mut Rng, x: f32, y: f32, a: f32, len: f32, depth: u32) {
            let col: Hsl = [135.0, 26.0, 22.0];
            let steps = (len / 4.0).round() as u32;
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                let (px, py) = (x + a.sin() * len * t, y - a.cos() * len * t);
                for sd in [-1.0f32, 1.0] {
                    let na = a + sd * r.range(50.0, 80.0) * DEG;
                    let nl = r.range(11.0, 16.0) * (1.0 - t * 0.3);
                    let c = with_jitter(col, r, 6.0, 6.0, 6.0);
                    g.line(
                        (px, py),
                        (px + na.sin() * nl, py - na.cos() * nl),
                        hsla(c, 1.0, 0.0),
                        1.8,
                    );
                }
                if depth > 0 && i > 2 && i % 4 == 0 {
                    let sd = if (i / 4) % 2 == 0 { 1.0 } else { -1.0 };
                    axis(
                        g,
                        r,
                        px,
                        py,
                        a + sd * 50.0 * DEG,
                        len * (1.0 - t) * 0.55,
                        depth - 1,
                    );
                }
            }
            g.line(
                (x, y),
                (x + a.sin() * len, y - a.cos() * len),
                hsla([25.0, 30.0, 28.0], 1.0, 0.0),
                2.0,
            );
        }
        axis(g, &mut r, 0.0, 0.0, 0.0, 236.0, 1);
    });
    // 11 boxwood / evergreen shrub.
    let mut r = rng.fork("box");
    cell(&mut g, 11, &mut |g| {
        for (dx, bend, len) in [
            (-45.0, -0.12, 0.75),
            (45.0, 0.12, 0.75),
            (-15.0, -0.03, 0.9),
            (18.0, 0.04, 0.88),
        ] {
            g.save();
            g.translate(dx, 0.0);
            draw_twig(
                g,
                &mut r,
                &Twig {
                    nodes: 12,
                    twig_len: len,
                    bend,
                    spread: (30.0, 70.0),
                    opposite: true,
                    petiole: 0.0,
                    size: (0.8, 1.0),
                    twig_w: 2.0,
                    leaf: Leaf::Oval {
                        col: [105.0, 42.0, 21.0],
                        len: 19.0,
                        width: 11.0,
                        hw: |t| (PI * t).sin().max(0.0).powf(0.7),
                        teeth: 0.0,
                        tooth_amp: 0.0,
                        veins: 0,
                    },
                    ..TWIG
                },
            );
            g.restore();
        }
    });
    // 12-13 palm frond (two cells wide), 14-15 dead frond.
    for (ci, dead) in [(12u32, false), (14, true)] {
        let (cx, cy) = ((ci % 4) as f32 * C, (ci / 4) as f32 * C);
        g.save();
        g.clip_rect(cx + 3.0, cy + 3.0, C * 2.0 - 6.0, C - 6.0);
        g.translate(cx + 3.0, cy);
        let mut r = rng.fork_n("frond", ci);
        let col = if dead {
            [34.0, 38.0, 36.0]
        } else {
            [82.0, 40.0, 30.0]
        };
        draw_frond(&mut g, &mut r, C * 2.0 - 6.0, C, col, dead);
        g.restore();
    }

    // Straight alpha, then dilate colours into transparent texels so mip
    // levels don't bleed dark fringes.
    let s = S as usize;
    let w = ATLAS_W as usize;
    let mut d: Vec<u8> = Vec::with_capacity(w * s * 4);
    for p in g.pix.pixels() {
        let c = p.demultiply();
        d.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    let mut valid: Vec<bool> = (0..w * s).map(|i| d[i * 4 + 3] > 100).collect();
    for _ in 0..10 {
        let mut next = valid.clone();
        for y in 0..s {
            for x in 0..w {
                let i = y * w + x;
                if valid[i] {
                    continue;
                }
                let (mut r, mut gg, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
                for (ox, oy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (xx, yy) = (x as i32 + ox, y as i32 + oy);
                    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= s as i32 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if !valid[j] {
                        continue;
                    }
                    r += d[j * 4] as u32;
                    gg += d[j * 4 + 1] as u32;
                    b += d[j * 4 + 2] as u32;
                    n += 1;
                }
                if let (Some(r), Some(gg), Some(b)) =
                    (r.checked_div(n), gg.checked_div(n), b.checked_div(n))
                {
                    d[i * 4] = r as u8;
                    d[i * 4 + 1] = gg as u8;
                    d[i * 4 + 2] = b as u8;
                    next[i] = true;
                }
            }
        }
        valid = next;
    }
    d
}

/// Autumn colour per leaf-atlas cell (sRGB; alpha 0 = never turns).
pub(crate) const AUTUMN: [[f32; 4]; 16] = [
    [0.62, 0.42, 0.12, 1.0],
    [0.85, 0.7, 0.18, 1.0],
    [0.85, 0.22, 0.06, 1.0],
    [0.5, 0.3, 0.1, 1.0],
    [0.92, 0.78, 0.2, 1.0],
    [0.72, 0.45, 0.14, 1.0],
    [1.0, 0.82, 0.12, 1.0],
    [0.75, 0.3, 0.12, 1.0],
    [0.0; 4],
    [0.0; 4],
    [0.0; 4],
    [0.0; 4],
    [0.0; 4],
    [0.0; 4],
    [0.0; 4],
    [0.0; 4],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_has_leaves_and_opaque_bark() {
        let a = atlas();
        assert_eq!(a.color.len(), (ATLAS_W * ATLAS_H * 4) as usize);
        assert_eq!(a.normal.len(), a.color.len());
        // Every leaf cell has a drawn twig cluster: some opaque texels.
        for cell in 0..16u32 {
            let (cx, cy) = ((cell % 4) * 256, (cell / 4) * 256);
            let mut opaque = 0;
            for y in cy + 8..cy + 248 {
                for x in cx + 8..cx + 248 {
                    if a.color[((y * ATLAS_W + x) * 4 + 3) as usize] > 128 {
                        opaque += 1;
                    }
                }
            }
            assert!(opaque > 500, "leaf cell {cell} has {opaque} opaque texels");
        }
        // Leaves are mostly green, bark is fully opaque.
        let (mut g, mut n) = (0.0, 0.0);
        for i in 0..(ATLAS_W * LEAF_H) as usize {
            if a.color[i * 4 + 3] > 200 {
                g += a.color[i * 4 + 1] as f32 - a.color[i * 4] as f32;
                n += 1.0;
            }
        }
        assert!(g / n > 5.0, "leaf texels lean green");
        let bark = &a.color[(ATLAS_W * LEAF_H * 4) as usize..];
        assert!(bark.chunks(4).all(|p| p[3] == 255));
    }
}
