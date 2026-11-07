//! Facade wall layers: brick, stone, render, concrete, glass, metal, timber.

#![allow(clippy::needless_range_loop)]

use super::core::{mix, rgb, rnd, rnd1, smooth, worley, wrap, wrapd, Rgb, TNoise, Tex};
use super::layout::{flemish_bond, random_courses, running_bond};
use super::masonry::{brick_wall, stone_blocks, BrickOpts, Paint, StoneOpts};
use super::paint::rng;
use crate::vegetation::rng::noise2;

fn lerp3(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)]
}

fn scale(c: Rgb, k: f32) -> Rgb {
    [c[0] * k, c[1] * k, c[2] * k]
}

/// Red brick, 215 × 65 mm with 10 mm joints, running bond.
pub(crate) fn brick_red(t: &mut Tex) {
    let (c1, c2, c3, c4) = (
        rgb(160, 74, 50),
        rgb(128, 54, 42),
        rgb(96, 44, 40),
        rgb(178, 96, 64),
    );
    let palette = move |a: f32, b: f32, _k: u32| {
        let mut o = lerp3(c1, c2, a);
        if b < 0.1 {
            o = lerp3(o, c3, 0.7);
        } else if b > 0.9 {
            o = lerp3(o, c4, 0.7);
        }
        scale(o, 0.9 + 0.2 * ((a * 7.31) % 1.0))
    };
    brick_wall(
        t,
        &BrickOpts {
            seed: 101,
            layout: running_bond(t.w, 30, 10, 11, 0.003, 0.5),
            joint: 0.01,
            depth: 0.007,
            bevel: 0.004,
            ragged: 0.0008,
            chips: 0.35,
            palette: &palette,
            mottle: 0.07,
            mortar: rgb(176, 170, 158),
            soot: 0.55,
            streaks: 0.45,
            rough: 0.82,
            paint: None,
        },
    );
}

/// Flemish bond, purple-brown brick with darker burnt headers.
pub(crate) fn brick_brown(t: &mut Tex) {
    let (c1, c2, c3, hd) = (
        rgb(112, 66, 52),
        rgb(86, 52, 48),
        rgb(92, 58, 66),
        rgb(66, 44, 44),
    );
    let palette = move |a: f32, b: f32, k: u32| {
        let mut o = lerp3(c1, c2, a);
        if b < 0.25 {
            o = lerp3(o, c3, 0.6);
        }
        if k == 1 && b > 0.35 {
            o = lerp3(o, hd, 0.65);
        }
        scale(o, 0.88 + 0.24 * ((a * 5.17) % 1.0))
    };
    brick_wall(
        t,
        &BrickOpts {
            seed: 202,
            layout: flemish_bond(t.w, 36, 8, 22, 0.002),
            joint: 0.01,
            depth: 0.006,
            bevel: 0.003,
            ragged: 0.0007,
            chips: 0.25,
            palette: &palette,
            mottle: 0.06,
            mortar: rgb(150, 144, 134),
            soot: 0.45,
            streaks: 0.35,
            rough: 0.78,
            paint: None,
        },
    );
}

/// London stock: yellow-buff, irregular, soot-darkened.
pub(crate) fn brick_yellow(t: &mut Tex) {
    let (c1, c2, c3, c4) = (
        rgb(196, 168, 118),
        rgb(168, 138, 96),
        rgb(120, 96, 72),
        rgb(186, 140, 108),
    );
    let palette = move |a: f32, b: f32, _k: u32| {
        let o = lerp3(c1, c2, a);
        if b < 0.18 {
            lerp3(o, c3, 0.6 + b)
        } else if b > 0.85 {
            lerp3(o, c4, 0.7)
        } else {
            o
        }
    };
    brick_wall(
        t,
        &BrickOpts {
            seed: 303,
            layout: running_bond(t.w, 30, 10, 33, 0.008, 0.5),
            joint: 0.011,
            depth: 0.008,
            bevel: 0.005,
            ragged: 0.0022,
            chips: 0.8,
            palette: &palette,
            mottle: 0.1,
            mortar: rgb(168, 160, 145),
            soot: 0.6,
            streaks: 0.45,
            rough: 0.88,
            paint: None,
        },
    );
}

/// White paint flaking off red brick.
pub(crate) fn brick_white(t: &mut Tex) {
    let (c1, c2) = (rgb(138, 104, 92), rgb(112, 86, 78));
    let palette = move |a: f32, _b: f32, _k: u32| lerp3(c1, c2, a);
    brick_wall(
        t,
        &BrickOpts {
            seed: 1515,
            layout: running_bond(t.w, 30, 10, 1516, 0.003, 0.5),
            joint: 0.01,
            depth: 0.006,
            bevel: 0.004,
            ragged: 0.001,
            chips: 0.4,
            palette: &palette,
            mottle: 0.07,
            mortar: rgb(170, 165, 155),
            soot: 0.25,
            streaks: 0.3,
            rough: 0.82,
            paint: Some(Paint {
                color: rgb(226, 224, 218),
                coverage: 0.985,
            }),
        },
    );
}

fn stone(seed: u32, layout: super::layout::Layout) -> StoneOpts<'static> {
    StoneOpts {
        seed,
        layout,
        joint: 0.005,
        depth: 0.003,
        bevel: 0.004,
        ragged: 0.001,
        bulge: 0.0,
        rough3d: 0.0,
        base: [0.5; 3],
        vary: 0.05,
        hue: 0.02,
        grain: 0.05,
        bedding: 1.0,
        mortar: [0.4; 3],
        joint_dirt: 0.5,
        soot: 0.3,
        streaks: 0.3,
        rough: 0.8,
        rough_joint: 0.9,
        erosion: 0.0,
        extra: None,
    }
}

pub(crate) fn brownstone(t: &mut Tex) {
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.004,
            depth: 0.003,
            bevel: 0.006,
            ragged: 0.001,
            bulge: 0.001,
            base: rgb(106, 72, 58),
            vary: 0.05,
            hue: 0.03,
            grain: 0.07,
            bedding: 4.0,
            mortar: rgb(92, 66, 56),
            joint_dirt: 0.4,
            soot: 0.4,
            streaks: 0.35,
            rough: 0.8,
            rough_joint: 0.9,
            erosion: 0.35,
            ..stone(404, random_courses(t.w, &[0.4; 6], 0.8, 1.4, 44))
        },
    );
}

pub(crate) fn limestone(t: &mut Tex) {
    let n = TNoise::new(4401, t.w);
    // Fossil shell fragments and calcite specks.
    let extra = |u: f32, v: f32, mask: f32, _: &super::layout::Hit, c: &mut [f32; 5]| {
        let sp = smooth(0.45, 0.6, n.fbm(u, v, 0.01, 2)) * 0.06 * mask;
        c[0] *= 1.0 + sp;
        c[1] *= 1.0 + sp;
        c[2] *= 1.0 + sp * 0.8;
    };
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.005,
            depth: 0.003,
            bevel: 0.004,
            ragged: 0.0006,
            bulge: 0.0008,
            base: rgb(214, 202, 178),
            vary: 0.035,
            hue: 0.02,
            grain: 0.035,
            bedding: 2.0,
            mortar: rgb(186, 176, 156),
            joint_dirt: 0.7,
            soot: 0.3,
            streaks: 0.4,
            rough: 0.78,
            rough_joint: 0.9,
            erosion: 0.15,
            extra: Some(&extra),
            ..stone(505, random_courses(t.w, &[0.6; 5], 0.9, 1.6, 55))
        },
    );
}

pub(crate) fn sandstone(t: &mut Tex) {
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.008,
            depth: 0.006,
            bevel: 0.01,
            ragged: 0.002,
            bulge: 0.0015,
            base: rgb(190, 150, 104),
            vary: 0.05,
            hue: 0.022,
            grain: 0.09,
            bedding: 6.0,
            mortar: rgb(170, 152, 126),
            joint_dirt: 0.5,
            soot: 0.45,
            streaks: 0.35,
            rough: 0.9,
            rough_joint: 0.95,
            erosion: 0.9,
            ..stone(606, random_courses(t.w, &[0.4; 6], 0.5, 1.0, 66))
        },
    );
}

/// Polished granite slabs: feldspar, quartz and mica speckle.
pub(crate) fn granite(t: &mut Tex) {
    let w = t.w;
    let fa = (w / 0.009).round() as i32;
    let fb = (w / 0.022).round() as i32;
    let p = [
        rgb(178, 150, 142),
        rgb(150, 128, 122),
        rgb(140, 140, 146),
        rgb(48, 46, 48),
        rgb(206, 202, 198),
    ];
    let extra = move |u: f32, v: f32, mask: f32, _: &super::layout::Hit, c: &mut [f32; 5]| {
        let cell = worley(9001, u * fa as f32, v * fa as f32, fa, fa, 1.0, false);
        let m = cell.id & 255;
        let mut k = match m {
            0..=119 => 0,
            120..=169 => 1,
            170..=224 => 2,
            225..=247 => 3,
            _ => 4,
        };
        let w1 = smooth(0.62, 0.45, cell.f1);
        let big = worley(9002, u * fb as f32, v * fb as f32, fb, fb, 1.0, false);
        if (big.id & 255) < 40 && big.f1 < 0.42 {
            k = 0; // large feldspar crystals
        }
        let col = p[k];
        let a = mask * (0.75 + 0.25 * w1);
        for i in 0..3 {
            c[i] = mix(c[i], col[i], a);
        }
        if k == 2 {
            c[3] = mix(c[3], 0.12, mask); // glassy quartz
        }
    };
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.004,
            depth: 0.002,
            bevel: 0.0015,
            ragged: 0.0002,
            base: rgb(160, 140, 136),
            vary: 0.03,
            hue: 0.01,
            grain: 0.02,
            bedding: 1.0,
            mortar: rgb(90, 88, 88),
            joint_dirt: 0.3,
            soot: 0.25,
            streaks: 0.25,
            rough: 0.2,
            rough_joint: 0.8,
            extra: Some(&extra),
            ..stone(1313, running_bond(w, 4, 2, 1314, 0.0, 0.5))
        },
    );
}

/// Dirt band at the bottom of the tile, wrapping softly into the top.
fn bottom_band(y: f32, w: f32) -> f32 {
    let dv = y / w;
    (1.0 - smooth(0.0, 0.3, dv)).max(1.0 - smooth(0.0, 0.03, 1.0 - dv))
}

/// Trowelled plaster with hairline cracks (tintable).
pub(crate) fn plaster(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3, nc) = (
        TNoise::new(601, w),
        TNoise::new(602, w),
        TNoise::new(603, w),
        TNoise::new(604, w),
    );
    let fc = (w / 0.35).round() as i32;
    let fc2 = (w / 0.12).round() as i32;
    t.each(|t, u, v, _, y, idx| {
        let wu = u + n1.fbm(u, v, 0.6, 3) * 0.04;
        let wv = v + n1.fbm(u + 0.5, v, 0.6, 3) * 0.04;
        let sweep = n2.ridged_g(wu, wv, 0.35, 3, 0.5, 0.18);
        let mut h =
            sweep * 0.0012 + n2.fbm(u, v, 0.05, 3) * 0.0003 + n3.fbm(u, v, 0.008, 2) * 0.00012;
        let area = smooth(0.15, 0.45, nc.fbm(u, v, 1.2, 3));
        let mut crack = 0.0;
        if area > 0.0 {
            let c = worley(
                6001,
                u * fc as f32 + n3.fbm(u, v, 0.1, 2) * 0.25,
                v * fc as f32,
                fc,
                fc,
                0.9,
                true,
            );
            crack = (1.0 - smooth(0.0, 0.012, c.edge)) * area;
            let a2 = area * smooth(0.2, 0.5, nc.fbm(u, v, 0.3, 2));
            if a2 > 0.0 {
                let c = worley(
                    6002,
                    u * fc2 as f32,
                    v * fc2 as f32 + n3.fbm(u, v, 0.05, 2) * 0.2,
                    fc2,
                    fc2,
                    0.9,
                    true,
                );
                crack = crack.max((1.0 - smooth(0.0, 0.02, c.edge)) * a2);
            }
        }
        h -= crack * 0.0008;
        let dirt_n = smooth(
            -0.3,
            0.6,
            n1.fbm(u, v, 0.4, 4) + n3.fbm_g(u, v, 0.15, 3, 0.5, 0.6) * 0.4,
        );
        let dirt = bottom_band(y, w) * dirt_n * 0.06;
        let streak = smooth(0.2, 0.9, n3.fbm_g(u, v, 0.35, 3, 0.5, 2.0)) * 0.05;
        let tone = 1.0
            + n2.fbm(u, v, 0.8, 4) * 0.025
            + sweep * 0.015
            + (rnd1(66, idx as u32) - 0.5) * 0.015;
        let k = tone * (1.0 - dirt - streak) * (1.0 - crack * 0.25);
        t.set(
            idx,
            0.89 * k,
            0.88 * k,
            0.855 * k * (1.0 - dirt * 0.1),
            0.88 + sweep * 0.05 - streak * 0.1,
            h,
        );
    });
}

/// Knock-down stucco (tintable).
pub(crate) fn stucco(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(701, w), TNoise::new(702, w));
    let f1 = (w / 0.014).round() as i32;
    let f2 = (w / 0.007).round() as i32;
    t.each(|t, u, v, _, y, idx| {
        let c1 = worley(7001, u * f1 as f32, v * f1 as f32, f1, f1, 1.0, false);
        let b1 = smooth(0.62, 0.15, c1.f1) * (0.6 + 0.4 * ((c1.id & 255) as f32 / 255.0));
        let c2 = worley(7002, u * f2 as f32, v * f2 as f32, f2, f2, 1.0, false);
        let b2 = smooth(0.6, 0.1, c2.f1) * if c2.id & 1 == 1 { 1.0 } else { 0.3 };
        let mut h = b1.min(0.7) * 0.0022
            + b2 * 0.0007
            + n1.fbm(u, v, 0.4, 3) * 0.0008
            + n2.fbm(u, v, 0.05, 2) * 0.0003;
        h += smooth(0.2, 0.8, n2.fbm(u, v, 0.1, 3)) * 0.0006;
        let dirt = bottom_band(y, w) * smooth(-0.3, 0.6, n1.fbm(u, v, 0.3, 4)) * 0.035;
        let streak = smooth(0.2, 0.9, n2.fbm_g(u, v, 0.3, 3, 0.5, 1.6)) * 0.05;
        let tone = (1.0 + n1.fbm(u, v, 0.7, 4) * 0.03 + b1 * 0.02) * (1.0 - dirt - streak);
        t.set(idx, 0.88 * tone, 0.87 * tone, 0.845 * tone, 0.93, h);
    });
}

/// Cast-in-place concrete: plywood form panels 1.2 × 0.6 m, tie holes.
pub(crate) fn concrete(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(801, w),
        TNoise::new(802, w),
        TNoise::new(803, w),
    );
    let fb = (w / 0.025).round() as i32;
    let (pw, ph, r0) = (1.2f32, 0.6f32, 0.013f32);
    let panels_x = ((w / pw).round() as i32).max(1);
    t.each(|t, u, v, x, y, idx| {
        let (pi, pj) = ((x / pw).floor(), (y / ph).floor());
        let pid = ((pi as i32).rem_euclid(panels_x) * 16 + pj as i32) as u32;
        let pr = rnd(808, pid, 1);
        let sx = (x - pi * pw).min((pi + 1.0) * pw - x);
        let sy = (y - pj * ph).min((pj + 1.0) * ph - y);
        let seam = 1.0 - smooth(0.0015, 0.004, sx.min(sy));
        let grain = n2.fbm_g(u + pr, v, 0.5, 4, 0.55, 0.012) * 0.00025;
        let mut h = n1.fbm(u, v, 0.3, 3) * 0.0006
            + grain
            + n3.fbm(u, v, 0.006, 2) * 0.00012
            + (pr - 0.5) * 0.001;
        h += seam * 0.0007;
        let (hx, hy) = (wrapd(x - 0.3, 0.6), wrapd(y - 0.3, 0.6));
        let hr = (hx * hx + hy * hy).sqrt();
        let hole = 1.0 - smooth(r0 - 0.0015, r0 + 0.0015, hr);
        let ring = smooth(r0, r0 + 0.004, hr) * (1.0 - smooth(r0 + 0.004, r0 + 0.012, hr));
        h -= hole * 0.012 * (1.0 - (hr / r0) * 0.4);
        let cell = worley(8001, u * fb as f32, v * fb as f32, fb, fb, 1.0, false);
        let pore = if (cell.id & 255) < 35 {
            1.0 - smooth(0.12, 0.3, cell.f1)
        } else {
            0.0
        };
        h -= pore * 0.002;
        // Rust and water runs under the tie holes.
        let below = -hy;
        let st_len = 0.08
            + 0.35
                * rnd(
                    809,
                    ((x - hx) / 0.6).round() as i32 as u32,
                    ((y - hy) / 0.6).round() as i32 as u32,
                );
        let runw = if hx.abs() < 0.018 + below * 0.03 {
            1.0
        } else {
            0.0
        };
        let run = if below > 0.0 && below < st_len {
            runw * (1.0 - below / st_len)
                * smooth(
                    0.012,
                    0.004,
                    hx.abs() + n3.fbm_g(u, v, 0.03, 2, 0.5, 0.2) * 0.008,
                )
        } else {
            0.0
        };
        let mott = n1.fbm_g(u, v, 0.25, 5, 0.55, 0.25) * 0.06
            + (pr - 0.5) * 0.08
            + n3.fbm(u, v, 0.03, 3) * 0.03;
        let streak = smooth(
            0.2,
            0.9,
            n2.fbm_g(u, v, 0.1, 3, 0.5, 1.4) + n1.fbm(u, v, 0.8, 2) * 0.3,
        ) * 0.12;
        let k = (1.0 + mott)
            * (1.0 - streak)
            * (1.0 - seam * 0.08)
            * (1.0 - pore * 0.35)
            * (1.0 - ring * 0.08);
        let (mut r, mut g, mut b) = (0.6 * k, 0.595 * k, 0.58 * k);
        if run > 0.0 {
            r = mix(r, r * 0.88, run);
            g = mix(g, g * 0.82, run);
            b = mix(b, b * 0.75, run);
        }
        if hole > 0.0 {
            r = mix(r, 0.3, hole);
            g = mix(g, 0.29, hole);
            b = mix(b, 0.28, hole);
        }
        t.set(idx, r, g, b, 0.86 + streak * 0.3 - hole * 0.1, h);
    });
}

/// Precast exposed-aggregate panel, one per tile, joint at the edge.
pub(crate) fn concrete_panel(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(901, w), TNoise::new(902, w));
    let fa = (w / 0.011).round() as i32;
    let agg = [
        rgb(196, 188, 172),
        rgb(150, 146, 140),
        rgb(108, 104, 100),
        rgb(178, 160, 136),
        rgb(212, 208, 200),
        rgb(140, 120, 104),
    ];
    let binder = rgb(168, 164, 156);
    t.each(|t, u, v, x, y, idx| {
        let e = x.min(w - x).min(y.min(w - y)) - 0.01;
        let jm = smooth(-0.001, 0.001, e);
        let cham = smooth(0.0, 0.012, e);
        let cell = worley(9003, u * fa as f32, v * fa as f32, fa, fa, 1.0, false);
        let pebble = if (cell.id & 255) < 175 {
            smooth(0.55, 0.3, cell.f1)
        } else {
            0.0
        };
        let pc = agg[((cell.id >> 8) % 6) as usize];
        let h = 0.015 * (0.6 + 0.4 * cham) + pebble * 0.0012 + n1.fbm(u, v, 0.5, 3) * 0.0005;
        let vary = 1.0 + (rnd1(91, cell.id) - 0.5) * 0.15;
        let top = 1.0 - smooth(0.0, 1.2, w - y);
        let st = smooth(0.0, 0.8, n2.fbm_g(u, v, 0.15, 3, 0.5, 1.5)) * (0.35 + 0.65 * top) * 0.22;
        let mac = 1.0 + n1.fbm(u, v, 0.8, 4) * 0.05;
        let k = mac * (1.0 - st) * (0.9 + 0.1 * cham);
        let sealant = 0.27 + n2.fbm(u, v, 0.02, 2) * 0.02;
        let mut c = [0.0f32; 3];
        for i in 0..3 {
            let s = if i == 2 { sealant * 1.03 } else { sealant };
            c[i] = mix(s, mix(binder[i], pc[i] * vary, pebble) * k, jm);
        }
        t.set(idx, c[0], c[1], c[2], mix(0.6, 0.9, jm), mix(0.003, h, jm));
    });
}

/// Curtain wall: mullions every 1.3 m, floor transom at the bottom, spandrel band.
pub(crate) fn glass_curtain(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(1001, w), TNoise::new(1002, w));
    let (mw, sp, span) = (0.03f32, 1.3f32, 1.0f32);
    t.each(|t, u, v, x, y, idx| {
        let mx = wrapd(x, sp).abs();
        let ty = wrapd(y, w).abs().min((y - span).abs());
        let is_floor = wrapd(y, w).abs() < 0.06;
        let pane = ((x / sp).floor() as i32 * 4 + if y < span { 0 } else { 1 }) as u32;
        let pr = rnd1(1010, pane);
        let tw = if is_floor { 0.05 } else { mw };
        let dm = (mx - mw).min(ty - tw);
        let frame = smooth(0.002, -0.002, dm);
        let cap_m = (1.0 - (mx / mw).powi(2)).max(0.0);
        let cap_t = (1.0 - (ty / tw).powi(2)).max(0.0);
        let cap = (if mx < mw { cap_m.sqrt() } else { 0.0 }).max(if ty < tw {
            cap_t.sqrt()
        } else {
            0.0
        });
        let groove = if (mx - mw * 0.55).abs() < 0.002 || (ty - tw * 0.55).abs() < 0.002 {
            1.0
        } else {
            0.0
        };
        let wave = n1.fbm_g(u, v + pr, 0.35, 2, 0.5, 2.5) * 0.0004 + n2.fbm(u, v, 1.5, 2) * 0.0006;
        let h = mix(
            wave,
            0.045 * cap - groove * 0.002 * if frame >= 1.0 { 1.0 } else { 0.0 },
            frame,
        );
        let spandrel = y < span && !is_floor;
        let (r, g, b, rough) = if spandrel {
            let k = 1.0 + n2.fbm(u, v, 0.2, 3) * 0.04 + (pr - 0.5) * 0.06;
            (0.26 * k, 0.285 * k, 0.31 * k, 0.16)
        } else {
            let k = 1.0 + (pr - 0.5) * 0.08 + n1.fbm(u, v, 1.0, 3) * 0.03;
            (0.17 * k, 0.205 * k, 0.245 * k, 0.06)
        };
        let fk = 1.0 + n2.fbm(u, v, 0.05, 2) * 0.03 - groove * 0.3;
        t.set(
            idx,
            mix(r, 0.34 * fk, frame),
            mix(g, 0.35 * fk, frame),
            mix(b, 0.37 * fk, frame),
            mix(rough, 0.35, frame),
            h,
        );
    });
}

/// Box-rib metal cladding, ribs every 0.2 m, end lap at the bottom.
pub(crate) fn metal_panel(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(1101, w), TNoise::new(1102, w));
    let (p, top, side, hh) = (0.2f32, 0.012f32, 0.01f32, 0.022f32);
    t.each(|t, u, v, x, y, idx| {
        let d = wrapd(x, p).abs();
        let mut h = if d < top {
            hh
        } else if d < top + side {
            hh * (1.0 - smooth(0.0, 1.0, (d - top) / side))
        } else {
            0.0
        };
        let pd = (wrapd(x - p / 2.0, p).abs() - 0.045).abs();
        h += (1.0 - smooth(0.0, 0.006, pd)) * 0.0015;
        h += n1.fbm_g(u, v, 0.6, 3, 0.5, 1.2) * 0.0008 * smooth(top + side, top + side + 0.02, d);
        let ly = wrap(y, w);
        if ly < 0.03 {
            h += 0.002;
        }
        let lap_line = 1.0 - smooth(0.0, 0.004, (ly - 0.03).abs().min((ly - w).abs()));
        let facet = if d >= top && d < top + side {
            0.94
        } else {
            1.0
        };
        let streak = smooth(0.15, 0.9, n2.fbm_g(u, v, 0.08, 3, 0.5, 1.5))
            * 0.1
            * if d < top + side { 0.6 } else { 1.0 };
        let chalk = n1.fbm(u, v, 0.4, 4) * 0.03;
        let k = (1.0 + chalk) * (1.0 - streak) * facet * (1.0 - lap_line * 0.25);
        t.set(
            idx,
            0.69 * k,
            0.70 * k,
            0.715 * k,
            0.42 + streak * 0.8 + n2.fbm(u, v, 0.2, 2) * 0.04,
            h,
        );
    });
}

fn fbm2(seed: u32, x: f32, y: f32, oct: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, 1.0);
    for o in 0..oct {
        sum += amp
            * noise2(
                seed.wrapping_add(o * 101),
                x * f + o as f32 * 17.31,
                y * f - o as f32 * 9.17,
            );
        norm += amp;
        amp *= 0.5;
        f *= 2.0;
    }
    sum / norm
}

struct Beam {
    a: (f32, f32),
    b: (f32, f32),
    w: f32,
    id: u32,
}

/// Half-timbering: oak posts every 1.2 m, sill and mid rail, braces, white infill.
pub(crate) fn timber(t: &mut Tex) {
    let w = t.w;
    let ws = 1201u32;
    let (nw, npl, nc) = (
        TNoise::new(1202, w),
        TNoise::new(1203, w),
        TNoise::new(1204, w),
    );
    let beams = [
        Beam {
            a: (0.1, 0.12),
            b: (1.1, 1.09),
            w: 0.17,
            id: 10,
        },
        Beam {
            a: (2.3, 0.12),
            b: (1.3, 1.09),
            w: 0.17,
            id: 11,
        },
        Beam {
            a: (1.2, 1.75),
            b: (1.65, 1.2),
            w: 0.14,
            id: 12,
        },
        Beam {
            a: (2.4, 1.75),
            b: (1.95, 1.2),
            w: 0.14,
            id: 13,
        },
    ];
    let (post_w, rail, mid) = (0.2f32, 0.22f32, 0.18f32);
    // Oak colour along the grain (gx) and across it (gy); returns (colour, check).
    let wood = |gx: f32, gy: f32, id: u32| -> (Rgb, f32) {
        let idf = id as f32;
        let base = 1.0 + (rnd1(1205, id) - 0.5) * 0.18;
        let g1 = fbm2(
            ws,
            gx * 1.2 + idf * 13.1,
            gy * 70.0 + noise2(ws, gx * 2.0, idf) * 3.0,
            4,
        );
        let ring = (gy * 120.0 + fbm2(ws + 1, gx * 1.5 + idf, gy * 4.0, 3) * 12.0)
            .sin()
            .abs();
        let g2 = noise2(ws + 2, gx * 3.0 + idf * 7.0, gy * 260.0) * 0.6
            + if ring < 0.25 { -0.6 } else { 0.0 };
        let check = smooth(
            0.85,
            0.97,
            noise2(ws + 3, gx * 0.6 + idf * 3.3, gy * 45.0).abs(),
        ) * smooth(-0.1, 0.3, noise2(ws + 4, gx * 2.2, idf));
        let k = base * (1.0 + g1 * 0.22 + g2 * 0.1) * (1.0 - check * 0.55);
        ([0.28 * k, 0.2 * k, 0.15 * k], check)
    };
    t.each(|t, u, v, x, y, idx| {
        let ph = -0.015 + npl.fbm(u, v, 0.4, 4) * 0.002 + npl.fbm(u, v, 0.03, 3) * 0.0004;
        let pk = 1.0 + npl.fbm(u, v, 0.5, 4) * 0.03
            - smooth(0.2, 0.9, nc.fbm_g(u, v, 0.15, 3, 0.5, 1.2)) * 0.07;
        let (mut r, mut g, mut b, mut rough, mut h) = (0.88 * pk, 0.86 * pk, 0.81 * pk, 0.9, ph);
        let edge_n = nw.fbm(u, v, 0.02, 3) * 0.004;
        let mut best = 1e9f32;
        let mut bh = 0.0f32;
        let mut hit: Option<(Rgb, f32)> = None;
        let mut consider = |dist: f32, half: f32, along: f32, across: f32, id: u32, z: f32| {
            let e = half - dist + edge_n;
            if e > 0.0 && z + (e / 0.012).min(1.0) * 0.004 > bh - 1e-6 {
                bh = z + smooth(0.0, 0.015, e) * 0.006;
                hit = Some(wood(along, across, id));
                best = e;
            }
        };
        let dpx = wrapd(x, 1.2).abs();
        let pi = ((x / 1.2).round() as i32).rem_euclid(2) as u32;
        consider(dpx, post_w / 2.0, y, wrapd(x, 1.2), 1 + pi, 0.003);
        consider(wrapd(y, w).abs(), rail / 2.0, x, wrapd(y, w), 3, 0.004);
        consider((y - 1.2).abs(), mid / 2.0, x + 7.0, y - 1.2, 4, 0.0035);
        for bm in &beams {
            let (dx, dy) = (bm.b.0 - bm.a.0, bm.b.1 - bm.a.1);
            let l = (dx * dx + dy * dy).sqrt();
            let (tx, ty) = (dx / l, dy / l);
            let (qx, qy) = (x - bm.a.0, y - bm.a.1);
            let along = qx * tx + qy * ty;
            if !(-0.05..=l + 0.05).contains(&along) {
                continue;
            }
            let across = -qx * ty + qy * tx;
            consider(across.abs(), bm.w / 2.0, along, across, bm.id, 0.002);
        }
        if let Some((wc, check)) = hit {
            let m = smooth(-0.001, 0.002, best);
            r = mix(r, wc[0], m);
            g = mix(g, wc[1], m);
            b = mix(b, wc[2], m);
            rough = mix(rough, 0.75, m);
            h = mix(
                h,
                bh + noise2(ws + 5, x * 40.0, y * 40.0) * 0.0004 - check * 0.003,
                m,
            );
            let gap = smooth(0.006, 0.0, best) * smooth(-0.003, 0.0, best);
            r *= 1.0 - gap * 0.4;
            g *= 1.0 - gap * 0.4;
            b *= 1.0 - gap * 0.4;
        }
        // Pegs where posts and rails cross.
        for px in [0.0, 1.2, 2.4] {
            for py in [0.06, 1.2, w - 0.06] {
                let dd = ((x - px).powi(2) + (y - py).powi(2)).sqrt() - 0.012;
                if (x - px).abs() < 0.05 && (y - py).abs() < 0.05 && dd < 0.0 {
                    r *= 0.75;
                    g *= 0.72;
                    b *= 0.7;
                    h += 0.001;
                }
            }
        }
        let dirt = (1.0 - smooth(0.0, 0.5, y)) * smooth(-0.2, 0.6, nc.fbm(u, v, 0.3, 3)) * 0.15;
        t.set(
            idx,
            r * (1.0 - dirt),
            g * (1.0 - dirt),
            b * (1.0 - dirt * 1.1),
            rough,
            h,
        );
    });
}

/// Clapboard lap siding, 0.15 m exposure, painted near-white (tintable).
pub(crate) fn wood_siding(t: &mut Tex) {
    let w = t.w;
    let n = ((w / 0.15).round() as i32).max(1);
    let bh = w / n as f32;
    let (ng, np, nd) = (
        TNoise::new(1401, w),
        TNoise::new(1402, w),
        TNoise::new(1403, w),
    );
    let mut r0 = rng(1404);
    let joints: Vec<f32> = (0..n).map(|_| r0.float() * w).collect();
    t.each(|t, u, v, x, y, idx| {
        let row = ((y / bh).floor() as i32).rem_euclid(n);
        let ly = (y - row as f32 * bh) / bh;
        let jx = wrapd(x - joints[row as usize], w);
        let bid = (row * 2 + if jx > 0.0 { 1 } else { 0 }) as u32;
        let mut h = 0.011 * (1.0 - ly) + 0.002;
        h -= (1.0 - smooth(0.0, 0.05, ly)) * 0.0015;
        h += (rnd1(1405, bid) - 0.5) * 0.0006;
        let grain = ng.fbm_g(u + rnd1(1406, bid), v, 0.6, 4, 0.55, 0.004);
        h += grain * 0.00025;
        let jg = 1.0 - smooth(0.0012, 0.0025, jx.abs());
        h -= jg * 0.004;
        let peel = smooth(
            0.62,
            0.66,
            np.fbm_g(u, v, 0.3, 5, 0.6, 0.3) + np.fbm(u, v, 0.03, 2) * 0.15,
        );
        let tone =
            1.0 + grain * 0.025 + (rnd1(1407, bid) - 0.5) * 0.03 + nd.fbm(u, v, 0.9, 3) * 0.02;
        let under = smooth(0.0, 0.18, ly);
        let dirt = (1.0 - under) * 0.06 + smooth(0.2, 0.9, nd.fbm_g(u, v, 0.1, 3, 0.5, 1.4)) * 0.06;
        let (mut r, mut g, mut b) = (
            0.87 * tone * (1.0 - dirt),
            0.86 * tone * (1.0 - dirt),
            0.83 * tone * (1.0 - dirt),
        );
        if peel > 0.0 {
            let wd = 0.55 + grain * 0.08;
            r = mix(r, wd * 0.82, peel);
            g = mix(g, wd * 0.72, peel);
            b = mix(b, wd * 0.6, peel);
            h -= peel * 0.0003;
        }
        let j = 1.0 - jg * 0.5;
        t.set(idx, r * j, g * j, b * j, 0.62 + peel * 0.25 + dirt, h);
    });
}
