//! Shared masonry: brick walls (any bond, optional paint) and dressed stone.

use super::core::{clamp01, mix, rnd, rnd1, smooth, worley, Rgb, TNoise, Tex};
use super::layout::{block_dist, Hit, Layout};

pub(crate) struct Paint {
    pub color: Rgb,
    pub coverage: f32,
}

pub(crate) struct BrickOpts<'a> {
    pub seed: u32,
    pub layout: Layout,
    /// Mortar joint width (m).
    pub joint: f32,
    /// Mortar recess below the brick face (m).
    pub depth: f32,
    /// Arris rounding width (m).
    pub bevel: f32,
    /// Edge raggedness (m).
    pub ragged: f32,
    /// Chipped edges, 0..1.
    pub chips: f32,
    /// Brick colour from two per-brick randoms and the block kind.
    pub palette: &'a dyn Fn(f32, f32, u32) -> Rgb,
    /// Mottling within a brick.
    pub mottle: f32,
    pub mortar: Rgb,
    pub soot: f32,
    pub streaks: f32,
    pub rough: f32,
    pub paint: Option<Paint>,
}

fn lum(r: f32, g: f32, b: f32) -> f32 {
    r * 0.3 + g * 0.59 + b * 0.11
}

pub(crate) fn brick_wall(t: &mut Tex, o: &BrickOpts) {
    let (w, s) = (t.w, o.seed);
    let n_edge = TNoise::new(s + 1, w);
    let n_chip = TNoise::new(s + 2, w);
    let n_mot = TNoise::new(s + 3, w);
    let n_soot = TNoise::new(s + 4, w);
    let n_str = TNoise::new(s + 5, w);
    let n_face = TNoise::new(s + 6, w);
    let n_paint = TNoise::new(s + 7, w);
    let n_mort = TNoise::new(s + 8, w);
    let aa = t.ps * 0.6;
    let fs = n_face.f(0.007);
    let fsf = fs as f32;
    let hj0 = o.joint / 2.0;
    t.each(|t, u, v, x, y, idx| {
        let hit = o.layout.at(x, y);
        let id = hit.id;
        let (ra, rb, rc, rd) = (rnd(s, id, 1), rnd(s, id, 2), rnd(s, id, 3), rnd(s, id, 4));
        let hj = hj0 + (rc - 0.5) * o.joint * 0.35;
        let e_raw = block_dist(&hit, hj, o.bevel * 1.2) + n_edge.fbm(u, v, 0.014, 3) * o.ragged;
        // Chips gather near the arrises.
        let chip_n = n_chip.fbm_g(u, v, 0.05, 4, 0.55, 0.05);
        let chip = smooth(0.3 - o.chips * 0.25, 0.75, chip_n) * o.chips * 0.014 * (0.6 + 0.8 * rd);
        let e = e_raw - chip;
        let mask = smooth(-aa, aa, e);
        let in_chip = if e_raw > 0.0 && e < 0.0 {
            smooth(0.0, 0.003, e_raw)
        } else {
            0.0
        };

        let cxm = (hit.x0 + hit.x1) * 0.5;
        let len = hit.x1 - hit.x0;
        let face =
            n_face.fbm_g(u, v, 0.012, 4, 0.55, 0.012) * 0.0007 + n_face.fbm(u, v, 0.08, 2) * 0.0008;
        let cell = worley(s + 9, u * fsf, v * fsf, fs, fs, 1.0, false);
        let pit = if (cell.id & 255) < 22 {
            -smooth(0.45, 0.1, cell.f1) * 0.0012
        } else {
            0.0
        };
        let mut hb =
            o.depth + (ra - 0.5) * 0.002 + ((hit.x - cxm) / len) * (rb - 0.5) * 0.002 + face + pit;
        hb -= o.depth * 0.45 * (1.0 - smooth(0.0, o.bevel, e));
        let sand = n_mort.fbm(u, v, 0.005, 2) * 0.0006;
        let hm = sand + smooth(0.0, hj * 1.4, e_raw + hj) * 0.0012;
        let h_chip = o.depth * 0.35 + face;
        let mut h = mix(hm, hb, mask);
        if in_chip > 0.0 {
            h = mix(h, h_chip, in_chip);
        }

        let col = (o.palette)(ra, rb, hit.kind);
        let mot = 1.0
            + n_mot.fbm(u, v, 0.06, 4) * o.mottle
            + n_mot.fbm(u + 0.5, v, 0.015, 2) * o.mottle * 0.5;
        let end_d = (hit.x - hit.x0).min(hit.x1 - hit.x);
        let flash = if rd > 0.55 {
            (1.0 - smooth(0.0, len * 0.35, end_d)) * 0.16
        } else {
            0.0
        };
        let sid = (cell.id >> 8) & 255;
        let spec = if sid < 14 {
            mix(1.0, 0.55, smooth(0.5, 0.15, cell.f1))
        } else if sid < 19 {
            mix(1.0, 1.35, smooth(0.4, 0.1, cell.f1))
        } else {
            1.0
        };
        let k = mot * (1.0 - flash) * spec;
        let (mut br, mut bg, mut bb) = (col[0] * k, col[1] * k, col[2] * k);
        if in_chip > 0.0 {
            br *= 0.88;
            bg *= 0.86;
            bb *= 0.86;
        }

        let grain = n_mort.fbm(u, v, 0.004, 2) * 0.08 + (rnd1(s + 11, idx as u32) - 0.5) * 0.07;
        let mdirt = 1.0 - 0.18 * (1.0 - smooth(-hj, 0.0, e_raw - hj));
        let m = |c: f32| c * (1.0 + grain) * mdirt;
        let mut r = mix(m(o.mortar[0]), br, mask);
        let mut g = mix(m(o.mortar[1]), bg, mask);
        let mut b = mix(m(o.mortar[2]), bb, mask);
        let mut rough = mix(0.95, o.rough + n_face.fbm(u, v, 0.03, 2) * 0.05, mask);

        if let Some(p) = &o.paint {
            let pn = n_paint.fbm_g(u, v, 0.3, 6, 0.62, 0.3)
                + n_paint.fbm(u, v, 0.04, 3) * 0.3
                + (1.0 - mask) * 0.1;
            let thr = 0.65 - p.coverage;
            let pm = smooth(thr, thr + 0.03, pn);
            h += pm * 0.0004
                + (smooth(thr - 0.01, thr + 0.01, pn) - smooth(thr + 0.01, thr + 0.04, pn))
                    * 0.0004;
            let pv = 0.94 + n_paint.fbm(u, v, 0.1, 3) * 0.04 - (1.0 - mask) * 0.03;
            let resid = smooth(0.45, 0.6, n_paint.fbm(u + 0.3, v, 0.01, 2)) * 0.6;
            let pmm = pm.max(resid * (1.0 - pm));
            r = mix(r, p.color[0] * pv, pmm);
            g = mix(g, p.color[1] * pv, pmm);
            b = mix(b, p.color[2] * pv, pmm);
            rough = mix(rough, 0.78, pmm);
        }

        let soot = smooth(
            -0.6,
            1.0,
            n_soot.fbm(u, v, 1.1, 3) + n_soot.fbm_g(u, v, 0.25, 3, 0.5, 1.2) * 0.3,
        ) * o.soot;
        let st = smooth(
            0.05,
            0.75,
            n_str.fbm_g(u, v, 0.09, 3, 0.5, 1.6) + n_str.fbm(u, v, 0.5, 2) * 0.3,
        ) * o.streaks;
        let dk = clamp01(1.0 - 0.3 * soot - 0.22 * st - (1.0 - mask) * 0.12 * soot);
        let l = lum(r, g, b);
        let ds = 0.35 * soot + 0.2 * st;
        t.set(
            idx,
            mix(r, l, ds) * dk,
            mix(g, l, ds) * dk,
            mix(b, l * 1.02, ds) * dk,
            rough,
            h,
        );
    });
}

/// A per-pixel hook: `(u, v, inside mask, hit, [r, g, b, rough, h])`.
pub(crate) type StoneExtra<'a> = &'a dyn Fn(f32, f32, f32, &Hit, &mut [f32; 5]);

pub(crate) struct StoneOpts<'a> {
    pub seed: u32,
    pub layout: Layout,
    pub joint: f32,
    /// Joint recess (m).
    pub depth: f32,
    /// Edge rounding width (m).
    pub bevel: f32,
    /// Edge raggedness (m).
    pub ragged: f32,
    /// Face bulge / rustication height (m).
    pub bulge: f32,
    /// Rough-dressed face relief (m).
    pub rough3d: f32,
    pub base: Rgb,
    /// Per-block value variation.
    pub vary: f32,
    /// Per-block hue variation.
    pub hue: f32,
    /// Fine speckle amplitude.
    pub grain: f32,
    /// Bedding anisotropy.
    pub bedding: f32,
    pub mortar: Rgb,
    /// Dirt in the joints and round the edges, 0..1.
    pub joint_dirt: f32,
    pub soot: f32,
    pub streaks: f32,
    pub rough: f32,
    pub rough_joint: f32,
    /// Erosion pits (sandstone), 0..1.
    pub erosion: f32,
    pub extra: Option<StoneExtra<'a>>,
}

/// Dressed stone blocks or slabs with joints.
pub(crate) fn stone_blocks(t: &mut Tex, o: &StoneOpts) {
    let (w, s) = (t.w, o.seed);
    let n_edge = TNoise::new(s + 1, w);
    let n_grain = TNoise::new(s + 2, w);
    let n_mac = TNoise::new(s + 3, w);
    let n_soot = TNoise::new(s + 4, w);
    let n_str = TNoise::new(s + 5, w);
    let n_ero = TNoise::new(s + 6, w);
    let n_rough = TNoise::new(s + 7, w);
    let aa = t.ps * 0.6;
    let fe = n_ero.f(0.02);
    let fef = fe as f32;
    t.each(|t, u, v, x, y, idx| {
        let hit = o.layout.at(x, y);
        let id = hit.id;
        let (ra, rb, rc) = (rnd(s, id, 1), rnd(s, id, 2), rnd(s, id, 3));
        let hj = o.joint / 2.0 * (0.8 + 0.4 * rc);
        let e = block_dist(&hit, hj, o.bevel) + n_edge.fbm(u, v, 0.03, 4) * o.ragged;
        let mask = smooth(-aa, aa, e);
        let (bw, bh) = (hit.x1 - hit.x0, hit.y1 - hit.y0);
        let lx = (hit.x - hit.x0) / bw - 0.5;
        let ly = (hit.y - hit.y0) / bh - 0.5;
        let mut hb = o.depth + (ra - 0.5) * 0.002;
        hb -= o.depth * 0.5 * (1.0 - smooth(0.0, o.bevel, e));
        hb += o.bulge * (1.0 - (lx * lx + ly * ly) * 2.2) * smooth(0.0, o.bevel * 2.0 + 0.01, e);
        if o.rough3d > 0.0 {
            let rr = n_rough.ridged_g(u, v, 0.12, 5, 0.55, 0.12) - 0.5;
            hb += rr * o.rough3d * smooth(0.0, 0.03, e);
        }
        let gr = n_grain.fbm_g(u, v, 0.006 * o.bedding, 3, 0.6, 0.006);
        hb += gr * 0.0004;
        let mut ero = 0.0;
        if o.erosion > 0.0 {
            let cell = worley(s + 8, u * fef, v * fef, fe, fe, 1.0, false);
            let em = smooth(0.1, 0.6, n_ero.fbm(u, v, 0.4, 3)) * o.erosion;
            ero = smooth(0.5, 0.15, cell.f1) * em * if cell.id & 3 == 0 { 1.0 } else { 0.3 };
            hb -= ero * 0.004;
            hb -= em * (1.0 - smooth(0.0, 0.03, e)) * 0.003;
        }
        let hm = n_grain.fbm(u, v, 0.005, 2) * 0.0005;
        let h = mix(hm, hb, mask);

        let vv = 1.0 + (ra - 0.5) * 2.0 * o.vary + n_mac.fbm(u, v, 0.35, 4) * o.vary * 0.6;
        let hu = (rb - 0.5) * 2.0 * o.hue;
        let fine = gr * o.grain + (rnd1(s + 9, idx as u32) - 0.5) * o.grain * 0.6;
        let bedd = n_grain.fbm_g(u + 0.2, v, 0.5 * o.bedding, 3, 0.5, 0.05) * o.grain * 0.8;
        let k = vv * (1.0 + fine + bedd) * (1.0 - ero * 0.2);
        let edge_dirt = 1.0 - (1.0 - smooth(0.0, 0.025, e)) * o.joint_dirt * 0.25;
        let jd = 1.0 - o.joint_dirt * 0.3;
        let mg0 = 1.0 + n_grain.fbm(u, v, 0.004, 2) * 0.08;
        let mut c = [
            mix(
                o.mortar[0] * jd * mg0,
                o.base[0] * k * (1.0 + hu) * edge_dirt,
                mask,
            ),
            mix(o.mortar[1] * jd * mg0, o.base[1] * k * edge_dirt, mask),
            mix(
                o.mortar[2] * jd * mg0,
                o.base[2] * k * (1.0 - hu) * edge_dirt,
                mask,
            ),
            mix(
                o.rough_joint,
                o.rough + n_grain.fbm(u, v, 0.05, 2) * 0.05,
                mask,
            ),
            h,
        ];
        if let Some(extra) = o.extra {
            extra(u, v, mask, &hit, &mut c);
        }
        let soot = smooth(-0.3, 0.8, n_soot.fbm(u, v, 1.3, 5)) * o.soot;
        let st = smooth(
            0.1,
            0.8,
            n_str.fbm_g(u, v, 0.12, 3, 0.5, 1.8) + n_str.fbm(u, v, 0.6, 2) * 0.3,
        ) * o.streaks;
        let dk = clamp01(1.0 - 0.35 * soot - 0.25 * st);
        let l = lum(c[0], c[1], c[2]);
        let f = |x: f32| mix(x, l, 0.25 * soot) * dk;
        t.set(idx, f(c[0]), f(c[1]), f(c[2]), c[3], c[4]);
    });
}
