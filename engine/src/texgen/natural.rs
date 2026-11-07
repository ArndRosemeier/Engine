//! Natural ground layers for open country: wild grass, forest floor, rock,
//! cliff strata, scree, snow, sand, mud, heather, clay, limestone pavement,
//! chalk turf, gravel and basalt. One tile is `W` metres square and wraps;
//! +v is up (on a cliff: up the face).

// Colour channels read clearer as indexed loops.
#![allow(clippy::needless_range_loop)]

use super::core::{mix, rgb, rnd1, smooth, worley, Rgb, TNoise, Tex};
use super::paint::{pebbles, rng, splat, stroke, Pebbles};
use std::f32::consts::{PI, TAU};

/// Long meadow grass: tufts of blades over thatch, dry and fresh patches, flowers.
pub(crate) fn wild_grass(t: &mut Tex) {
    let w = t.w;
    let (nd, nc, ns) = (
        TNoise::new(5101, w),
        TNoise::new(5102, w),
        TNoise::new(5103, w),
    );
    t.each(|t, u, v, _, _, idx| {
        // Old flattened grass under the living blades, so gaps read as grass.
        let k = 0.8 + ns.fbm(u, v, 0.08, 3) * 0.25 + (rnd1(5104, idx as u32) - 0.5) * 0.3;
        t.set(
            idx,
            0.25 * k,
            0.26 * k,
            0.12 * k,
            0.95,
            ns.fbm(u, v, 0.15, 3) * 0.004,
        );
    });
    let mut r = rng(5105);
    for _ in 0..((w * w) / 0.0004).round() as u32 {
        let (x, y, a) = (r.float() * w, r.float() * w, r.float() * TAU);
        let (len, vk, dry) = (r.range(0.015, 0.04), r.range(0.6, 1.1), r.float());
        stroke(t, x, y, a, len, |t, _, idx| {
            let z = 0.004 + r.float() * 0.002;
            if z <= t.h[idx] {
                return;
            }
            t.set(
                idx,
                mix(0.20, 0.40, dry) * vk,
                mix(0.27, 0.36, dry) * vk,
                mix(0.09, 0.18, dry) * vk,
                0.9,
                z,
            );
        });
    }
    // Tufts: blades fanning out from a root, so the grass reads as clumps.
    for _ in 0..((w * w) / 0.0012).round() as u32 {
        let (x, y) = (r.float() * w, r.float() * w);
        let (u, v) = (x / w, y / w);
        if r.float() > 0.8 + 0.4 * nc.fbm(u, v, 0.4, 3) {
            continue;
        }
        let dry = smooth(0.15, 0.7, nd.fbm(u, v, 1.5, 4) + nd.fbm(u, v, 0.2, 2) * 0.3) * 0.7;
        let (blades, lean) = (r.int(5, 11), r.float() * TAU);
        for _ in 0..blades {
            let a = lean + r.range(-1.1, 1.1);
            let (len, hue, vk) = (r.range(0.03, 0.09), r.float(), r.range(0.65, 1.25));
            let dk = dry * r.range(0.4, 1.0);
            let cr = mix(mix(0.16, 0.27, hue), 0.58, dk);
            let cg = mix(mix(0.30, 0.40, hue), 0.50, dk);
            let cb = mix(mix(0.07, 0.11, hue), 0.27, dk);
            stroke(t, x, y, a, len, |t, s, idx| {
                let z = 0.01 + s * len * 0.8;
                if z <= t.h[idx] {
                    return;
                }
                let sh = vk * (0.45 + 0.55 * s);
                t.set(idx, cr * sh, cg * sh, cb * sh, 0.72 + dk * 0.2, z);
            });
        }
    }
    // Flowers: small bright heads above the blades, in loose drifts.
    let colours = [
        rgb(238, 232, 210),
        rgb(236, 196, 60),
        rgb(150, 110, 200),
        rgb(205, 70, 60),
    ];
    for _ in 0..(w * w * 30.0).round() as u32 {
        let (x, y) = (r.float() * w, r.float() * w);
        if r.float() > smooth(0.1, 0.5, nc.fbm(x / w + 0.3, y / w, 0.7, 3)) {
            continue;
        }
        let c = colours[r.int(0, colours.len() as u32 - 1) as usize];
        let (rad, z) = (r.range(0.004, 0.008), r.range(0.06, 0.1));
        splat(t, x, y, rad, |t, dx, dy, idx| {
            let q = 1.0 - (dx * dx + dy * dy).sqrt() / rad;
            if z <= t.h[idx] {
                return;
            }
            let k = 0.8 + 0.2 * q;
            t.set(idx, c[0] * k, c[1] * k, c[2] * k, 0.6, z + q * 0.002);
        });
    }
}

/// Forest floor: dark soil, fallen leaves, needles, twigs and moss cushions.
pub(crate) fn forest_floor(t: &mut Tex) {
    let w = t.w;
    let (ns, nm) = (TNoise::new(5201, w), TNoise::new(5202, w));
    t.each(|t, u, v, _, _, idx| {
        let k = 0.8 + ns.fbm(u, v, 0.1, 3) * 0.25 + (rnd1(5203, idx as u32) - 0.5) * 0.2;
        t.set(
            idx,
            0.27 * k,
            0.20 * k,
            0.13 * k,
            0.9,
            ns.fbm(u, v, 0.4, 3) * 0.01,
        );
    });
    let mut r = rng(5204);
    let leaf_colours = [
        rgb(120, 70, 30),
        rgb(150, 95, 40),
        rgb(95, 60, 32),
        rgb(170, 120, 55),
        rgb(80, 55, 35),
    ];
    for _ in 0..((w * w) / 0.0012).round() as u32 {
        let (cx, cy) = (r.float() * w, r.float() * w);
        let a = r.range(0.025, 0.055);
        let b = a * r.range(0.4, 0.6);
        let th = r.float() * PI;
        let (ct, st) = (th.cos(), th.sin());
        let z0 = r.range(0.0, 0.012);
        let col = leaf_colours[r.int(0, leaf_colours.len() as u32 - 1) as usize];
        let vk = r.range(0.7, 1.15);
        splat(t, cx, cy, a, |t, dx, dy, idx| {
            let (px, py) = (dx * ct + dy * st, -dx * st + dy * ct);
            let q = (px / a).powi(2) + (py / b).powi(2);
            if q >= 1.0 {
                return;
            }
            let z = z0 + (1.0 - q) * 0.0015 + (py / b).abs() * 0.001;
            if z <= t.h[idx] {
                return;
            }
            let vein = if py.abs() < b * 0.08 { 0.8 } else { 1.0 };
            t.set(
                idx,
                col[0] * vk * vein,
                col[1] * vk * vein,
                col[2] * vk * vein,
                0.75,
                z,
            );
        });
    }
    // Twigs and needles.
    for _ in 0..(w * w * 260.0).round() as u32 {
        let (x, y, a) = (r.float() * w, r.float() * w, r.float() * TAU);
        let (len, z0, vk) = (
            r.range(0.02, 0.12),
            r.range(0.004, 0.016),
            r.range(0.6, 1.0),
        );
        stroke(t, x, y, a, len, |t, _, idx| {
            if z0 <= t.h[idx] {
                return;
            }
            t.set(idx, 0.24 * vk, 0.17 * vk, 0.11 * vk, 0.85, z0);
        });
    }
    // Moss cushions on top.
    t.each(|t, u, v, _, _, idx| {
        let m = smooth(0.1, 0.35, nm.fbm(u, v, 0.6, 4));
        if m <= 0.0 {
            return;
        }
        let k = 0.8 + nm.fbm(u, v, 0.01, 2) * 0.3;
        t.r[idx] = mix(t.r[idx], 0.24 * k, m);
        t.g[idx] = mix(t.g[idx], 0.36 * k, m);
        t.b[idx] = mix(t.b[idx], 0.10 * k, m);
        t.rough[idx] = mix(t.rough[idx], 0.95, m);
        t.h[idx] = t.h[idx].max(0.012 + m * 0.01 + nm.fbm(u, v, 0.008, 2) * 0.002);
    });
}

struct RockOpts {
    seed: u32,
    base: Rgb,
    warm: Rgb,
    lichen: f32,
    strata: f32,
}

/// Weathered rock in place: the face broken into slabs (beds), each a facet
/// of its own tilt, so neighbours step or fold. Fissures wander across it;
/// every facet is chipped and grainy; rain has run dark streaks down it,
/// iron stains it warm and lichen crusts the facets that face up.
fn rock(t: &mut Tex, o: RockOpts) {
    let w = t.w;
    let s = o.seed;
    let (n1, n2, n3, n4) = (
        TNoise::new(s + 1, w),
        TNoise::new(s + 2, w),
        TNoise::new(s + 3, w),
        TNoise::new(s + 7, w),
    );
    // Slabs about 1.6 m across and 0.7 m high (0.35 m when banded).
    let banded = o.strata > 0.0;
    let fx = ((w / if banded { 1.6 } else { 1.3 }).round() as i32).max(2);
    let fy = ((w / if banded { 0.35 } else { 0.85 }).round() as i32).max(2);
    let (fxf, fyf) = (fx as f32, fy as f32);
    let bands = (w / 0.8).round();
    t.each(|t, u, v, _, _, idx| {
        let wu = u + n1.fbm(u, v, 1.4, 3) * (0.35 / fxf);
        let wv = v + n1.fbm(u + 0.5, v, 1.4, 3) * (0.25 / fyf);
        let cell = worley(s + 4, wu * fxf, wv * fyf, fx, fy, 0.9, true);
        let id = cell.id;
        // The facet's own tilt and how far it stands out.
        let ax = (rnd1(s + 17, id) - 0.5) * 0.5;
        let ay = (rnd1(s + 18, id) - 0.35) * 0.4;
        let dx = (wu * fxf - cell.px) * (w / fxf);
        let dy = (wv * fyf - cell.py) * (w / fyf);
        let facet = ax * dx + ay * dy + (rnd1(s + 11, id) - 0.5) * 0.06;
        // A narrow worn bevel where facets meet.
        let edge_m = cell.edge * (w / fxf).min(w / fyf);
        let worn = smooth(0.0, 0.05, edge_m);
        // Fissures: the wavering zero lines of a noise, open only in stretches.
        let fis = n2.fbm(u, v, 2.2, 3);
        let fissure =
            (1.0 - smooth(0.003, 0.014, fis.abs())) * smooth(0.0, 0.35, n3.fbm(u + 0.2, v, 1.7, 2));
        let band = if banded {
            (v * TAU * bands + n3.fbm(u, v, 2.0, 2) * 2.0).sin() * 0.5 + 0.5
        } else {
            0.0
        };
        let chip = n2.ridged(u + rnd1(s + 12, id), v, 0.5, 4) * 0.04 + n3.fbm(u, v, 0.14, 3) * 0.01;
        let bulge = n4.fbm(u + 0.7, v, 2.5, 3) * 0.06;
        let grain = (rnd1(s + 5, idx as u32) - 0.5) * 0.004;
        let mut h = facet + bulge + chip + grain - fissure * 0.02 - (1.0 - worn) * 0.006
            + band * o.strata * 0.008;
        let streak = smooth(0.2, 0.65, n4.fbm_g(u, v, 0.3, 3, 0.5, 3.0)) * 0.2;
        let lit = ay * 0.5;
        let tone = (1.0
            + (rnd1(s + 13, id) - 0.5) * 0.14
            + lit
            + n1.fbm(u, v, 1.6, 3) * 0.16
            + (rnd1(s + 19, idx as u32) - 0.5) * 0.16
            + (band - 0.5) * o.strata * 0.22)
            * (1.0 - streak);
        let warm = smooth(0.05, 0.65, n3.fbm(u + 0.4, v, 1.8, 3)) * (0.6 + 0.4 * rnd1(s + 14, id));
        let dark = 1.0 - fissure * 0.4 - (1.0 - worn) * 0.06;
        let mut c = [0.0f32; 3];
        for k in 0..3 {
            c[k] = mix(o.base[k], o.warm[k], warm) * tone * dark;
        }
        // Lichen: grey-green and orange crusts on facets that face up.
        let li = smooth(0.3, 0.55, n2.fbm(u + 0.3, v, 0.45, 4))
            * o.lichen
            * smooth(-0.02, 0.12, ay)
            * (1.0 - fissure);
        if li > 0.0 {
            let lc = if rnd1(s + 6, id) < 0.3 {
                [0.62, 0.40, 0.15]
            } else {
                [0.55, 0.58, 0.45]
            };
            for k in 0..3 {
                c[k] = mix(c[k], lc[k], li);
            }
            h += li * 0.002;
        }
        t.set(idx, c[0], c[1], c[2], 0.86 - fissure * 0.08 - li * 0.05, h);
    });
}

pub(crate) fn rock_face(t: &mut Tex) {
    rock(
        t,
        RockOpts {
            seed: 5301,
            base: rgb(112, 108, 102),
            warm: rgb(138, 118, 96),
            lichen: 0.6,
            strata: 0.0,
        },
    );
}

pub(crate) fn cliff_strata(t: &mut Tex) {
    rock(
        t,
        RockOpts {
            seed: 5401,
            base: rgb(150, 128, 104),
            warm: rgb(176, 140, 102),
            lichen: 0.25,
            strata: 1.0,
        },
    );
}

/// Scree: angular stones of all sizes over grit.
pub(crate) fn scree(t: &mut Tex) {
    pebbles(
        t,
        5501,
        &Pebbles {
            count: 1.6,
            r_min: 0.02,
            r_max: 0.11,
            height: 0.45,
            palette: &[
                rgb(120, 116, 110),
                rgb(98, 95, 90),
                rgb(140, 132, 120),
                rgb(110, 100, 88),
            ],
            fines: rgb(92, 86, 78),
            dust: 0.25,
            ..Pebbles::default()
        },
    );
}

/// Snow: wind ripples, soft drifts, a cold blue in the hollows.
pub(crate) fn snow(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(5601, w), TNoise::new(5602, w));
    t.each(|t, u, v, _, _, idx| {
        let ripple = n1.ridged_g(u + n2.fbm(u, v, 1.5, 2) * 0.05, v, 0.9, 3, 0.5, 0.25);
        let h = n2.fbm(u, v, 1.6, 4) * 0.05 + ripple * 0.012 + n1.fbm(u, v, 0.02, 2) * 0.0008;
        let hollow = smooth(0.02, -0.03, h);
        let k = 0.94 + (rnd1(5603, idx as u32) - 0.5) * 0.04;
        let sparkle = if rnd1(5604, idx as u32) < 0.02 {
            -0.3
        } else {
            0.0
        };
        t.set(
            idx,
            mix(0.95, 0.86, hollow) * k,
            mix(0.96, 0.9, hollow) * k,
            mix(0.98, 0.97, hollow) * k,
            0.55 + sparkle,
            h,
        );
    });
}

/// Sand: wind ripples and fine grain, darker and smoother where damp.
pub(crate) fn sand(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(5701, w), TNoise::new(5702, w));
    let waves = (w / 0.09).round();
    t.each(|t, u, v, _, _, idx| {
        let warp = n2.fbm(u, v, 0.8, 3) * 0.25;
        let ripple = ((v + warp) * TAU * waves).sin() * 0.5 + 0.5;
        let damp = smooth(0.1, 0.5, n1.fbm(u, v, 1.4, 3)) * 0.5;
        let h = ripple * 0.004 * (1.0 - damp) + n1.fbm(u, v, 0.5, 3) * 0.01;
        let k = (1.0 + (rnd1(5703, idx as u32) - 0.5) * 0.18 + ripple * 0.04) * (1.0 - damp * 0.35);
        t.set(idx, 0.76 * k, 0.66 * k, 0.48 * k, 0.9 - damp * 0.3, h);
    });
}

/// Mud: wet dark earth and puddles.
pub(crate) fn mud(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(5801, w), TNoise::new(5802, w));
    t.each(|t, u, v, _, _, idx| {
        let h0 = n1.fbm(u, v, 0.5, 4) * 0.02
            + n2.fbm(u, v, 0.05, 3) * 0.004
            + n2.ridged(u, v, 0.3, 3) * 0.006;
        let wet = smooth(-0.005, -0.015, h0);
        let k = 1.0 + (rnd1(5803, idx as u32) - 0.5) * 0.1;
        let tone = 1.0 + n2.fbm(u, v, 0.4, 3) * 0.3;
        t.set(
            idx,
            mix(0.30, 0.13, wet) * k * tone,
            mix(0.23, 0.10, wet) * k * tone,
            mix(0.15, 0.08, wet) * k * tone,
            mix(0.75, 0.08, wet),
            h0.max(-0.012),
        );
    });
}

/// Short tufts: `n` per square metre, blades `len` long, standing `z0` over the bed.
fn tufts(
    t: &mut Tex,
    seed: u32,
    n: f32,
    len: [f32; 2],
    colours: &[Rgb],
    z0: f32,
    wh: &dyn Fn(f32, f32) -> f32,
) {
    let w = t.w;
    let mut r = rng(seed);
    for _ in 0..(w * w * n).round() as u32 {
        let (x, y) = (r.float() * w, r.float() * w);
        if r.float() > wh(x / w, y / w) {
            continue;
        }
        let c = colours[r.int(0, colours.len() as u32 - 1) as usize];
        let lean = r.float() * TAU;
        for _ in 0..r.int(3, 7) {
            let a = lean + r.range(-1.2, 1.2);
            let (l, vk) = (r.range(len[0], len[1]), r.range(0.7, 1.2));
            stroke(t, x, y, a, l, |t, s, idx| {
                let z = z0 + s * l * 0.6;
                if z <= t.h[idx] {
                    return;
                }
                let sh = vk * (0.55 + 0.45 * s);
                t.set(idx, c[0] * sh, c[1] * sh, c[2] * sh, 0.85, z);
            });
        }
    }
}

/// Moor: heather in low dark clumps over black peat, purple where it flowers.
pub(crate) fn heather(t: &mut Tex) {
    let w = t.w;
    let (np, nc) = (TNoise::new(5901, w), TNoise::new(5902, w));
    t.each(|t, u, v, _, _, idx| {
        let wet = smooth(0.15, 0.4, np.fbm(u, v, 0.9, 3));
        let k = 0.85 + np.fbm(u, v, 0.05, 3) * 0.3 + (rnd1(5903, idx as u32) - 0.5) * 0.2;
        let h = np.fbm(u, v, 0.3, 3) * 0.006 - wet * 0.006;
        t.set(
            idx,
            mix(0.16, 0.08, wet) * k,
            mix(0.12, 0.07, wet) * k,
            mix(0.08, 0.05, wet) * k,
            mix(0.92, 0.4, wet),
            h,
        );
    });
    tufts(
        t,
        5904,
        60.0,
        [0.02, 0.06],
        &[rgb(110, 112, 60), rgb(140, 128, 70), rgb(80, 96, 44)],
        0.002,
        &|_, _| 1.0,
    );
    let mut r = rng(5905);
    for _ in 0..(w * w * 40.0).round() as u32 {
        let (cx, cy) = (r.float() * w, r.float() * w);
        let (u, v) = (cx / w, cy / w);
        if r.float() > smooth(-0.3, 0.3, nc.fbm(u, v, 0.8, 3)) + 0.5 {
            continue;
        }
        let rad = r.range(0.1, 0.25);
        let bloom = smooth(0.0, 0.35, nc.fbm(u + 0.5, v, 1.2, 3)) * r.range(0.4, 1.0);
        for _ in 0..(rad * rad * 14000.0).round() as u32 {
            let a = r.float() * TAU;
            let d = r.float().sqrt() * rad;
            let (x, y) = (cx + a.cos() * d, cy + a.sin() * d);
            let dome = (1.0 - (d / rad).powi(2)).max(0.0).sqrt();
            let flower = r.float() < bloom * 0.7;
            let vk = r.range(0.6, 1.2);
            let c = if flower {
                [0.5 * vk, 0.24 * vk, 0.46 * vk]
            } else if r.float() < 0.6 {
                [0.22 * vk, 0.24 * vk, 0.12 * vk]
            } else {
                [0.3 * vk, 0.2 * vk, 0.13 * vk]
            };
            let (dir, len) = (r.float() * TAU, r.range(0.01, 0.025));
            stroke(t, x, y, dir, len, |t, _, idx| {
                let z = 0.01 + dome * rad * 0.5 + if flower { 0.003 } else { 0.0 };
                if z <= t.h[idx] {
                    return;
                }
                t.set(idx, c[0], c[1], c[2], 0.9, z);
            });
        }
    }
}

/// Badlands: bare clay crusted by the sun, cracked into polygons and crazed
/// into small lumps, a few stones washed out on top.
pub(crate) fn clay(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(6001, w),
        TNoise::new(6002, w),
        TNoise::new(6003, w),
    );
    let bf = ((w / 0.8).round() as i32).max(2);
    let sf = ((w / 0.07).round() as i32).max(2);
    let (bff, sff) = (bf as f32, sf as f32);
    t.each(|t, u, v, _, _, idx| {
        let wu = u + n1.fbm(u, v, 1.2, 3) * (0.3 / bff);
        let wv = v + n1.fbm(u + 0.5, v, 1.2, 3) * (0.3 / bff);
        let big = worley(6007, wu * bff, wv * bff, bf, bf, 0.9, false);
        let small = worley(6008, u * sff, v * sff, sf, sf, 0.9, false);
        let crack = 1.0 - smooth(0.0, 0.04, (big.f2 - big.f1) * (w / bff));
        let seam = 1.0 - smooth(0.0, 0.2, small.f2 - small.f1);
        let lump = (1.0 - small.f1 * small.f1 * 1.6).max(0.0).sqrt();
        let deep = smooth(-0.2, 0.4, n3.fbm(u, v, 0.8, 3));
        let h = lump * 0.006 - seam * 0.002 - crack * 0.02
            + n2.fbm(u, v, 0.15, 3) * 0.006
            + (rnd1(6004, idx as u32) - 0.5) * 0.001;
        let k = (0.92
            + n1.fbm(u, v, 0.6, 3) * 0.16
            + (rnd1(6009, big.id) - 0.5) * 0.1
            + (rnd1(6005, idx as u32) - 0.5) * 0.06)
            * (1.0 - crack * 0.45 - seam * 0.06)
            * (0.94 + 0.06 * deep);
        t.set(idx, 0.70 * k, 0.62 * k, 0.54 * k, 0.93, h);
    });
    pebbles(
        t,
        6006,
        &Pebbles {
            count: 0.06,
            r_min: 0.01,
            r_max: 0.04,
            height: 0.5,
            palette: &[rgb(150, 136, 118), rgb(110, 100, 90), rgb(170, 150, 120)],
            bed: false,
            ..Pebbles::default()
        },
    );
}

/// Karst: limestone pavement, grey blocks between deep joints with moss
/// and ferns down in them.
pub(crate) fn limestone(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(6101, w),
        TNoise::new(6102, w),
        TNoise::new(6103, w),
    );
    let (a, b) = ((w / 1.3).round(), (w / 0.9).round());
    t.each(|t, u, v, _, _, idx| {
        let j1 = ((u + n1.fbm(u, v, 0.6, 2) * 0.03) * PI * a).sin().abs();
        let j2 = ((v + u * 0.25 + n2.fbm(u, v, 0.6, 2) * 0.03) * PI * b)
            .sin()
            .abs();
        let open2 = smooth(-0.15, 0.15, n3.fbm(u, v, 0.9, 2));
        let grike = (1.0 - smooth(0.0, 0.07, j1)).max((1.0 - smooth(0.0, 0.06, j2)) * open2);
        let pit = n2.ridged(u, v, 0.08, 3) * 0.004 + n1.fbm(u, v, 0.35, 3) * 0.008;
        let tone = 0.92 + n3.fbm(u, v, 0.5, 3) * 0.14 + (rnd1(6104, idx as u32) - 0.5) * 0.1;
        let moss = grike * smooth(-0.1, 0.3, n1.fbm(u + 0.4, v, 0.7, 3));
        let weather = smooth(-0.1, 0.45, n1.fbm(u + 0.9, v, 1.1, 3)) * 0.22;
        let base = [0.43, 0.42, 0.39];
        let gc = [0.08, 0.08, 0.07];
        let mc = [0.16, 0.24, 0.08];
        let mut c = [0.0f32; 3];
        for k in 0..3 {
            c[k] = mix(
                mix(base[k] * tone * (1.0 - weather), gc[k], grike * 0.85),
                mc[k],
                moss * 0.6,
            );
        }
        t.set(
            idx,
            c[0],
            c[1],
            c[2],
            0.8 + grike * 0.15,
            pit - grike * 0.05,
        );
    });
    tufts(
        t,
        6105,
        40.0,
        [0.03, 0.08],
        &[rgb(80, 110, 50), rgb(110, 120, 60)],
        -0.03,
        &|u, _| smooth(0.4, 0.9, 1.0 - (u * PI * a).sin().abs()),
    );
}

/// Downs: turf grazed short and dense, white chalk showing where worn, flints.
pub(crate) fn chalk_turf(t: &mut Tex) {
    let w = t.w;
    let (n1, n2) = (TNoise::new(6201, w), TNoise::new(6202, w));
    t.each(|t, u, v, _, _, idx| {
        let k = 0.85 + n1.fbm(u, v, 0.1, 3) * 0.25 + (rnd1(6203, idx as u32) - 0.5) * 0.25;
        t.set(
            idx,
            0.18 * k,
            0.24 * k,
            0.09 * k,
            0.92,
            n1.fbm(u, v, 0.2, 3) * 0.002,
        );
    });
    let worn = |u: f32, v: f32| smooth(0.1, 0.36, n2.fbm(u, v, 1.6, 4));
    tufts(
        t,
        6204,
        700.0,
        [0.008, 0.022],
        &[
            rgb(70, 100, 36),
            rgb(92, 118, 44),
            rgb(110, 120, 56),
            rgb(60, 88, 34),
        ],
        0.002,
        &|u, v| 1.0 - worn(u, v) * 0.9,
    );
    t.each(|t, u, v, _, _, idx| {
        let c = worn(u, v);
        if c <= 0.0 {
            return;
        }
        let k = 0.92 + (rnd1(6205, idx as u32) - 0.5) * 0.12;
        t.r[idx] = mix(t.r[idx], 0.82 * k, c);
        t.g[idx] = mix(t.g[idx], 0.80 * k, c);
        t.b[idx] = mix(t.b[idx], 0.74 * k, c);
        t.h[idx] = mix(t.h[idx], -0.002, c);
    });
    let mask = |u: f32, v: f32| 0.2 + worn(u, v);
    pebbles(
        t,
        6206,
        &Pebbles {
            count: 0.04,
            r_min: 0.012,
            r_max: 0.035,
            height: 0.4,
            palette: &[rgb(70, 66, 64), rgb(120, 110, 100), rgb(210, 205, 196)],
            bed: false,
            mask: Some(&mask),
            ..Pebbles::default()
        },
    );
}

/// A lava field's crust: black-grey basalt clinker with dark cracks, gas
/// pits, rust and drifts of ash.
pub(crate) fn basalt(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(6401, w),
        TNoise::new(6402, w),
        TNoise::new(6403, w),
    );
    let bf = ((w / 0.3).round() as i32).max(2);
    let sf = ((w / 0.05).round() as i32).max(2);
    let hf = ((w / 1.1).round() as i32).max(2);
    let (bff, sff, hff) = (bf as f32, sf as f32, hf as f32);
    t.each(|t, u, v, _, _, idx| {
        let wu = u + n1.fbm(u, v, 1.2, 3) * (0.5 / bff);
        let wv = v + n1.fbm(u + 0.5, v, 1.2, 3) * (0.5 / bff);
        let big = worley(6407, wu * bff, wv * bff, bf, bf, 0.9, false);
        let small = worley(6408, u * sff, v * sff, sf, sf, 0.9, false);
        let huge = worley(6410, wu * hff, wv * hff, hf, hf, 0.9, false);
        let open = smooth(-0.25, 0.25, n3.fbm(u, v, 0.8, 3));
        let crack = ((1.0 - smooth(0.0, 0.05, (big.f2 - big.f1) * (w / bff))) * open)
            .max(1.0 - smooth(0.0, 0.035, (huge.f2 - huge.f1) * (w / hff)));
        let pit = smooth(0.0, 0.25, 0.55 - small.f1);
        let block = (1.0 - big.f1 * big.f1 * 0.9).max(0.0).sqrt();
        let rust = smooth(0.0, 0.5, n2.fbm(u, v, 0.9, 3)) * (0.4 + 0.6 * rnd1(6409, big.id));
        let ash = smooth(0.05, 0.55, n3.fbm(u + 0.3, v, 0.7, 3));
        let h = block * 0.03 * open - crack * 0.04 - pit * 0.004
            + n1.fbm(u, v, 0.2, 3) * 0.01
            + (rnd1(6404, idx as u32) - 0.5) * 0.002;
        let k = (0.95
            + n1.fbm(u, v, 0.5, 3) * 0.3
            + (rnd1(6405, big.id) - 0.5) * 0.5
            + (rnd1(6406, idx as u32) - 0.5) * 0.1)
            * (0.8 + 0.35 * block)
            * (1.0 - crack * 0.7 - pit * 0.25);
        let base = [0.2 * k, 0.195 * k, 0.2 * k];
        let rc = [0.42 * k, 0.22 * k, 0.14 * k];
        let ac = [0.4 * k, 0.39 * k, 0.38 * k];
        let mut c = [0.0f32; 3];
        for i in 0..3 {
            c[i] = mix(mix(base[i], rc[i], rust * 0.6), ac[i], ash * 0.7);
        }
        t.set(idx, c[0], c[1], c[2], 0.88 - block * 0.08, h);
    });
}

/// A river valley's floor: rounded pebbles and grit, grass in drifts between.
pub(crate) fn gravel(t: &mut Tex) {
    let w = t.w;
    let n1 = TNoise::new(6301, w);
    pebbles(
        t,
        6302,
        &Pebbles {
            count: 1.1,
            r_min: 0.015,
            r_max: 0.06,
            height: 0.5,
            palette: &[
                rgb(150, 144, 132),
                rgb(118, 112, 104),
                rgb(170, 160, 140),
                rgb(130, 118, 100),
                rgb(96, 92, 88),
            ],
            fines: rgb(120, 110, 94),
            dust: 0.15,
            ..Pebbles::default()
        },
    );
    tufts(
        t,
        6303,
        260.0,
        [0.03, 0.08],
        &[rgb(70, 98, 36), rgb(96, 112, 46), rgb(120, 118, 60)],
        0.02,
        &|u, v| smooth(-0.1, 0.35, n1.fbm(u, v, 0.8, 3)),
    );
}
