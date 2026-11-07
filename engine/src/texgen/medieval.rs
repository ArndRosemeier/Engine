//! Medieval layers for towns and cellars: rubble, oak, thatch, shingles,
//! leaded glass, cobbles, flagstones, old brick, floorboards, limewash.

use super::core::{clamp01, mix, rgb, rnd1, smooth, worley, wrapd, Rgb, TNoise, Tex};
use super::layout::{random_courses, running_bond, Hit};
use super::masonry::{brick_wall, stone_blocks, BrickOpts, Paint, StoneOpts};

/// Rubble walling: rough field stones in roughly level courses, deep lime joints.
pub(crate) fn rubble(t: &mut Tex) {
    stone_blocks(
        t,
        &StoneOpts {
            seed: 7101,
            layout: random_courses(
                t.w,
                &[0.16, 0.24, 0.14, 0.3, 0.2, 0.18, 0.26, 0.22, 0.12, 0.28],
                0.18,
                0.55,
                7102,
            ),
            joint: 0.026,
            depth: 0.014,
            bevel: 0.03,
            ragged: 0.018,
            bulge: 0.012,
            rough3d: 0.01,
            base: rgb(150, 142, 128),
            vary: 0.14,
            hue: 0.05,
            grain: 0.12,
            bedding: 1.5,
            mortar: rgb(196, 188, 170),
            joint_dirt: 0.6,
            soot: 0.2,
            streaks: 0.3,
            rough: 0.92,
            rough_joint: 0.97,
            erosion: 0.3,
            extra: None,
        },
    );
}

/// Old oak: adzed faces, long grain along u, open checks, grey sheen.
pub(crate) fn oak(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(7201, w),
        TNoise::new(7202, w),
        TNoise::new(7203, w),
    );
    t.each(|t, u, v, _, _, idx| {
        let grain = n1.fbm_g(u, v, 0.6, 4, 0.55, 0.012);
        let ring = (v * 220.0 + n2.fbm(u, v, 0.5, 3) * 9.0).sin().abs();
        let check = smooth(0.82, 0.96, n3.ridged_g(u, v, 0.9, 3, 0.5, 0.05))
            * smooth(-0.2, 0.4, n2.fbm(u, v, 1.2, 2));
        let adze = n2.fbm_g(u, v, 0.18, 2, 0.5, 0.1) * 0.0015;
        let k = (1.0 + grain * 0.25 - if ring < 0.2 { 0.1 } else { 0.0 }) * (1.0 - check * 0.6);
        let grey = smooth(-0.2, 0.5, n3.fbm(u, v, 0.8, 3)) * 0.35;
        t.set(
            idx,
            mix(0.36, 0.33, grey) * k,
            mix(0.27, 0.31, grey) * k,
            mix(0.2, 0.28, grey) * k,
            0.78 + grey * 0.1,
            adze + grain * 0.0008 - check * 0.004,
        );
    });
}

/// Thatch: straw in courses, combed down the slope, darker and mossy where old.
pub(crate) fn thatch(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(7301, w),
        TNoise::new(7302, w),
        TNoise::new(7303, w),
    );
    let course = 0.3;
    t.each(|t, u, v, _, y, idx| {
        let stalk = n1.fbm_g(u, v, 0.012, 3, 0.6, 0.25);
        let bunch = n2.fbm_g(u, v, 0.09, 3, 0.5, 0.4);
        let ly = (y / course).fract();
        let butt = smooth(0.0, 0.12, ly) * (1.0 - smooth(0.7, 1.0, ly) * 0.3);
        let h = stalk * 0.006 + bunch * 0.012 + butt * 0.02 + (1.0 - ly) * 0.01;
        let age = smooth(-0.3, 0.5, n3.fbm(u, v, 0.8, 4));
        let moss = smooth(0.35, 0.7, n3.fbm(u + 0.3, v, 0.5, 3)) * 0.5;
        let k = 1.0 + stalk * 0.35 + bunch * 0.15 - (1.0 - butt) * 0.25;
        t.set(
            idx,
            mix(mix(0.62, 0.42, age) * k, 0.26, moss),
            mix(mix(0.5, 0.36, age) * k, 0.3, moss),
            mix(mix(0.3, 0.24, age) * k, 0.16, moss),
            0.95,
            h,
        );
    });
}

/// Split wooden shingles in staggered courses, silvered by the weather.
pub(crate) fn shingle(t: &mut Tex) {
    let w = t.w;
    let nr = ((w / 0.16).round() as i32).max(1);
    let ev = w / nr as f32;
    let (n1, n2) = (TNoise::new(7401, w), TNoise::new(7402, w));
    t.each(|t, u, v, x, y, idx| {
        let row = ((y / ev).floor() as i32).rem_euclid(nr) as u32;
        let ly = (y / ev).fract();
        let nc = ((w / (0.12 + 0.08 * rnd1(7403, row))).round() as i32).max(1);
        let sw = w / nc as f32;
        let xx = (x + rnd1(7404, row) * w) % w;
        let col = ((xx / sw).floor() as i32).rem_euclid(nc) as u32;
        let lx = (xx / sw).fract();
        let id = row * 97 + col;
        let (ra, rb) = (rnd1(7405, id), rnd1(7406, id));
        let gap = smooth(0.0, 0.06, lx.min(1.0 - lx));
        let grain = n1.fbm_g(u, v, 0.4, 3, 0.5, 0.01);
        let h = (1.0 - ly) * 0.008 + grain * 0.0005 + (ra - 0.5) * 0.002;
        let silver = clamp01(0.5 + (rb - 0.5) * 0.6 + n2.fbm(u, v, 0.6, 3) * 0.3);
        let k = (0.85 + 0.3 * ra)
            * (1.0 + grain * 0.3)
            * (0.55 + 0.45 * gap)
            * (0.8 + 0.2 * smooth(0.0, 0.15, ly));
        t.set(
            idx,
            mix(0.42, 0.44, silver) * k,
            mix(0.3, 0.42, silver) * k,
            mix(0.2, 0.4, silver) * k,
            0.8,
            mix(-0.002, h, gap),
        );
    });
}

/// Leaded glass: diamond panes of greenish crown glass in lead cames.
pub(crate) fn leaded(t: &mut Tex) {
    let w = t.w;
    let n1 = TNoise::new(7501, w);
    let p = 0.11;
    t.each(|t, u, v, x, y, idx| {
        let (la, lb) = ((x + y) / p, (x - y) / p);
        let came = wrapd(la, 1.0).abs().min(wrapd(lb, 1.0).abs()) * p;
        let lead = smooth(0.006, 0.003, came);
        let cell = worley(7502, u * 40.0, v * 40.0, 40, 40, 1.0, false);
        let ripple = n1.fbm(u, v, 0.05, 2) * 0.3;
        let gk = 0.12 + ripple * 0.05 + (cell.id & 7) as f32 * 0.006;
        t.set(
            idx,
            mix(gk * 0.8, 0.16, lead),
            mix(gk, 0.17, lead),
            mix(gk * 0.85, 0.18, lead),
            mix(0.05, 0.55, lead),
            lead * 0.002 + ripple * 0.0004,
        );
    });
}

/// Cobbles: rounded field stones set close in sandy earth, worn smooth.
pub(crate) fn cobbles(t: &mut Tex) {
    let w = t.w;
    let n = ((w / 0.14).round() as i32).max(1);
    let m = w / n as f32;
    let (n1, n2) = (TNoise::new(7601, w), TNoise::new(7602, w));
    let stones = [
        rgb(126, 120, 112),
        rgb(108, 106, 104),
        rgb(94, 96, 100),
        rgb(134, 120, 106),
        rgb(142, 136, 124),
        rgb(116, 110, 100),
        rgb(100, 92, 84),
    ];
    t.each(|t, u, v, _, _, idx| {
        let wu = u + n1.fbm(u, v, 0.3, 2) * 0.004;
        let wv = v + n1.fbm(u + 0.4, v, 0.3, 2) * 0.004;
        let cell = worley(
            7603,
            wu * n as f32 + 0.5,
            wv * n as f32 + 0.5,
            n,
            n,
            0.7,
            true,
        );
        let (ra, rb) = (rnd1(7604, cell.id), rnd1(7605, cell.id));
        let e = cell.edge * m - 0.005 - ra * 0.004;
        let stone = smooth(-0.002, 0.005, e);
        let dome = clamp01(e / (0.045 + rb * 0.02)).powf(0.6) * 0.03;
        let grain = n2.fbm(u, v, 0.015, 3) * 0.002;
        let c = stones[((ra * stones.len() as f32) as usize) % stones.len()];
        let k = (0.85 + rb * 0.3) * (1.0 + grain * 40.0) * (0.75 + 0.25 * clamp01(e / 0.04).sqrt());
        let dirt = smooth(-0.3, 0.6, n2.fbm(u, v, 0.4, 3)) * 0.25;
        t.set(
            idx,
            mix(0.2, mix(c[0] * k, 0.3, dirt), stone),
            mix(0.18, mix(c[1] * k, 0.27, dirt), stone),
            mix(0.14, mix(c[2] * k, 0.2, dirt), stone),
            mix(0.95, 0.7 - dome * 6.0, stone),
            stone * (0.004 + dome) + grain,
        );
    });
}

/// Flagstones: worn slabs of many sizes, gritty earth in the wide joints.
pub(crate) fn flagstone(t: &mut Tex) {
    stone_blocks(
        t,
        &StoneOpts {
            seed: 7701,
            layout: random_courses(
                t.w,
                &[0.34, 0.58, 0.27, 0.49, 0.38, 0.62, 0.3],
                0.26,
                1.05,
                7702,
            ),
            joint: 0.014,
            depth: 0.008,
            bevel: 0.03,
            ragged: 0.03,
            bulge: 0.003,
            rough3d: 0.005,
            base: rgb(124, 120, 112),
            vary: 0.22,
            hue: 0.05,
            grain: 0.12,
            bedding: 1.2,
            mortar: rgb(104, 96, 84),
            joint_dirt: 0.45,
            soot: 0.25,
            streaks: 0.0,
            rough: 0.78,
            rough_joint: 0.98,
            erosion: 0.35,
            extra: None,
        },
    );
}

/// Old hand-made brick: dull red-brown, chipped, lime joints with salt bloom.
pub(crate) fn brick(t: &mut Tex) {
    let (c1, c2, c3, c4) = (
        rgb(132, 70, 50),
        rgb(104, 56, 44),
        rgb(78, 46, 40),
        rgb(150, 92, 66),
    );
    let lerp =
        |a: Rgb, b: Rgb, k: f32| [mix(a[0], b[0], k), mix(a[1], b[1], k), mix(a[2], b[2], k)];
    let palette = move |a: f32, b: f32, _k: u32| {
        let mut o = lerp(c1, c2, a);
        if b < 0.15 {
            o = lerp(o, c3, 0.7);
        } else if b > 0.88 {
            o = lerp(o, c4, 0.6);
        }
        let v = 0.86 + 0.26 * ((a * 6.13) % 1.0);
        [o[0] * v, o[1] * v, o[2] * v]
    };
    brick_wall(
        t,
        &BrickOpts {
            seed: 7801,
            layout: running_bond(t.w, 24, 8, 7802, 0.008, 0.5),
            joint: 0.013,
            depth: 0.008,
            bevel: 0.006,
            ragged: 0.0016,
            chips: 0.6,
            palette: &palette,
            mottle: 0.14,
            mortar: rgb(168, 160, 144),
            soot: 0.6,
            streaks: 0.45,
            rough: 0.9,
            paint: Some(Paint {
                color: rgb(214, 210, 198),
                coverage: 0.12,
            }),
        },
    );
}

/// Floorboards: long planks of uneven width along u, staggered butt joints, nails.
pub(crate) fn boards(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(7901, w),
        TNoise::new(7902, w),
        TNoise::new(7903, w),
    );
    let mut widths = Vec::new();
    let mut sum = 0.0;
    while sum < w - 0.12 {
        let pw = 0.17 + rnd1(7904, widths.len() as u32) * 0.09;
        widths.push(pw);
        sum += pw;
    }
    let mut edges = vec![0.0f32];
    for pw in &widths {
        edges.push(edges.last().copied().unwrap_or(0.0) + pw * w / sum);
    }
    t.each(|t, u, v, x, y, idx| {
        let mut row = 0;
        while row < widths.len() - 1 && y >= edges[row + 1] {
            row += 1;
        }
        let rw = edges[row + 1] - edges[row];
        let ly = (y - edges[row]) / rw;
        let len = w / (1.0 + (rnd1(7905, row as u32) * 2.0).floor());
        let xx = (x + rnd1(7906, row as u32) * w) % w;
        let piece = (xx / len).floor() as u32;
        let lx = (xx % len) / len;
        let id = row as u32 * 31 + piece;
        let (ra, rb) = (rnd1(7907, id), rnd1(7908, id));
        let end_d = (lx * len).min((1.0 - lx) * len);
        let gap_e = (ly * rw).min((1.0 - ly) * rw).min(end_d);
        let gap = smooth(0.002, 0.008, gap_e);
        let grain = n1.fbm_g(u + ra, v, 0.5, 4, 0.55, 0.01);
        let ring = (v * 260.0 + ra * 40.0 + n2.fbm(u, v, 0.4, 3) * 8.0)
            .sin()
            .abs();
        let kx = ((xx % len) - len * (0.3 + ra * 0.4)) * 0.6;
        let ky = (ly - 0.5) * rw;
        let knot = smooth(0.08, 0.0, (kx * kx + ky * ky).sqrt()) * if rb > 0.6 { 1.0 } else { 0.0 };
        let nail = if end_d < 0.05 && (ly - 0.5).abs() < 0.12 {
            smooth(0.007, 0.004, ((end_d - 0.03).powi(2) + ky * ky).sqrt())
        } else {
            0.0
        };
        let wear = smooth(-0.2, 0.5, n3.fbm(u, v, 0.9, 3));
        let dirt = (1.0 - smooth(0.004, 0.03, gap_e)) * 0.4;
        let k = (0.82 + ra * 0.3)
            * (1.0 + grain * 0.22 - if ring < 0.18 { 0.08 } else { 0.0 })
            * (1.0 - knot * 0.2)
            * (1.0 - dirt);
        let r = mix(mix(0.42, 0.37, wear) * k, 0.12, nail);
        let g = mix(mix(0.31, 0.3, wear) * k, 0.11, nail);
        let b = mix(mix(0.21, 0.24, wear) * k, 0.1, nail);
        let h = mix(
            -0.004,
            0.002 * (1.0 - wear * 0.5) + grain * 0.0006 - knot * 0.0005 + (rb - 0.5) * 0.0015,
            gap,
        );
        t.set(
            idx,
            mix(0.06, r, gap),
            mix(0.05, g, gap),
            mix(0.04, b, gap),
            mix(0.95, 0.72 + wear * 0.12, gap) - nail * 0.3,
            h,
        );
    });
}

/// Limewashed rubble: stones showing through whitewash that flakes off, damp and mould.
pub(crate) fn limewash(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(8001, w),
        TNoise::new(8002, w),
        TNoise::new(8003, w),
    );
    let extra = |u: f32, v: f32, mask: f32, _: &Hit, c: &mut [f32; 5]| {
        let flake = smooth(
            0.16,
            0.22,
            n1.fbm_g(u, v, 0.6, 5, 0.6, 0.6) * 0.75 + n2.fbm(u, v, 0.06, 3) * 0.25,
        );
        let coat = 1.0 - flake;
        let tone = 0.9 + n2.fbm(u, v, 0.25, 3) * 0.05 + n3.fbm(u, v, 0.03, 2) * 0.03;
        let mould = smooth(0.1, 0.45, n3.fbm(u + 0.37, v, 0.35, 4)) * 0.5;
        let r = mix(0.86 * tone, 0.34, mould * 0.6);
        let g = mix(0.84 * tone, 0.36, mould * 0.6);
        let b = mix(0.78 * tone, 0.28, mould * 0.6);
        c[0] = mix(c[0], r, coat);
        c[1] = mix(c[1], g, coat);
        c[2] = mix(c[2], b, coat);
        c[3] = mix(c[3], 0.95, coat);
        c[4] = c[4] * mix(1.0, 0.6, coat) + (1.0 - mask) * coat * 0.004 + coat * 0.0012;
    };
    stone_blocks(
        t,
        &StoneOpts {
            seed: 8004,
            layout: random_courses(
                w,
                &[0.14, 0.27, 0.12, 0.33, 0.19, 0.24, 0.3, 0.16, 0.22],
                0.14,
                0.6,
                8005,
            ),
            joint: 0.034,
            depth: 0.008,
            bevel: 0.05,
            ragged: 0.04,
            bulge: 0.014,
            rough3d: 0.006,
            base: rgb(140, 134, 122),
            vary: 0.14,
            hue: 0.05,
            grain: 0.1,
            bedding: 1.4,
            mortar: rgb(176, 168, 150),
            joint_dirt: 0.5,
            soot: 0.0,
            streaks: 0.0,
            rough: 0.92,
            rough_joint: 0.96,
            erosion: 0.2,
            extra: Some(&extra),
        },
    );
}
