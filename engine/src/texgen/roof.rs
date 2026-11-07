//! Roof layers. +v is up the slope: shingles, tiles and slates overlap
//! downward (each course's butt edge lies at the bottom of its row).

use super::core::{clamp01, mix, rgb, rnd, rnd1, smooth, worley, wrap, wrapd, Rgb, TNoise, Tex};
use super::paint::{pebbles, rng, splat, Pebbles};
use std::f32::consts::{PI, TAU};

/// Flat bitumen membrane: 1 m rolls with laps, patches, blisters, dried puddles.
pub(crate) fn tar(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(1601, w),
        TNoise::new(1602, w),
        TNoise::new(1603, w),
    );
    let mut r0 = rng(1604);
    let patches: Vec<[f32; 4]> = (0..3)
        .map(|_| {
            [
                r0.float() * w,
                r0.float() * w,
                r0.range(0.5, 1.2),
                r0.range(0.4, 0.9),
            ]
        })
        .collect();
    let rolls = (w.round() as i32).max(1);
    let ends: Vec<f32> = (0..rolls).map(|_| r0.float() * w).collect();
    let fb = (w / 0.25).round() as i32;
    t.each(|t, u, v, x, y, idx| {
        let strip = (x.floor() as i32).rem_euclid(rolls) as usize;
        let sx = x - x.floor();
        let lap = if sx < 0.08 { 1.0 } else { 0.0 };
        let edge = 1.0 - smooth(0.0, 0.006, (sx - 0.08).abs());
        let mut h = lap * 0.0025 + edge * 0.0012 * if sx < 0.085 { 1.0 } else { 0.0 };
        let ey = wrapd(y - ends[strip], w);
        let elap = if ey > 0.0 && ey < 0.1 { 1.0 } else { 0.0 };
        let eedge = 1.0 - smooth(0.0, 0.006, (ey - 0.1).abs());
        h += elap * 0.0025 + eedge * 0.001 * if ey < 0.105 && ey > 0.0 { 1.0 } else { 0.0 };
        h += n1.fbm(u, v, 0.4, 4) * 0.0015
            + n2.fbm(u, v, 0.01, 3) * 0.0003
            + n3.ridged_g(u, v, 0.12, 3, 0.5, 0.5) * 0.0008
            + (rnd1(1607, idx as u32) - 0.5) * 0.0003;
        let cell = worley(1605, u * fb as f32, v * fb as f32, fb, fb, 1.0, false);
        if (cell.id & 255) < 30 {
            h += smooth(0.35, 0.0, cell.f1) * 0.004;
        }
        let mut pm = 0.0f32;
        for p in &patches {
            let dx = wrapd(x - p[0], w).abs() - p[2] / 2.0;
            let dy = wrapd(y - p[1], w).abs() - p[3] / 2.0;
            let d = dx.max(dy) + n3.fbm(u, v, 0.08, 3) * 0.025;
            pm = pm.max(smooth(0.002, -0.002, d));
        }
        h += pm * 0.003;
        let pud = n3.fbm_g(u, v, 1.6, 5, 0.55, 1.6);
        let dust = smooth(0.0, 0.6, pud) * (0.7 + 0.3 * n2.fbm(u, v, 0.05, 3));
        let tone = 1.0
            + n1.fbm(u, v, 0.15, 3) * 0.06
            + n2.fbm(u, v, 1.5, 3) * 0.1
            + (rnd1(1606, idx as u32) - 0.5) * 0.12;
        let mut r = mix(0.235 * tone, 0.36, dust * 0.7);
        let mut g = mix(0.23 * tone, 0.35, dust * 0.7);
        let mut b = mix(0.225 * tone, 0.33, dust * 0.7);
        let mut rough = 0.82;
        if pm > 0.0 {
            r = mix(r, 0.19 * tone, pm);
            g = mix(g, 0.19 * tone, pm);
            b = mix(b, 0.195 * tone, pm);
            rough = mix(rough, 0.6, pm);
        }
        let e = 1.0 - edge * 0.25;
        t.set(idx, r * e, g * e, b * e, rough + dust * 0.1, h);
    });
}

/// Terracotta pantiles: 0.3 m wide S-profile, 0.3 m gauge.
pub(crate) fn clay_tile(t: &mut Tex) {
    let w = t.w;
    let nc = ((w / 0.3).round() as i32).max(1);
    let nr = ((w / 0.3).round() as i32).max(1);
    let (twd, e) = (w / nc as f32, w / nr as f32);
    let (n1, n2, nm) = (
        TNoise::new(1701, w),
        TNoise::new(1702, w),
        TNoise::new(1703, w),
    );
    let fl = (w / 0.012).round() as i32;
    let colours = [
        rgb(172, 90, 60),
        rgb(162, 82, 56),
        rgb(150, 76, 54),
        rgb(138, 78, 60),
        rgb(178, 100, 66),
    ];
    let row_off: Vec<f32> = (0..nr).map(|r| rnd1(1704, r as u32) * 0.03).collect();
    t.each(|t, u, v, x, y, idx| {
        let row = ((y / e).floor() as i32).rem_euclid(nr);
        let ly = y / e - (y / e).floor();
        let xx = x + row_off[row as usize];
        let col = (xx / twd).floor();
        let lx = xx / twd - col;
        let tid = (row * 64 + (col as i32).rem_euclid(nc)) as u32;
        let (ra, rb) = (rnd1(1705, tid), rnd1(1706, tid));
        let prof = if lx < 0.68 {
            -((lx / 0.68) * PI).sin() * 0.038
        } else {
            (((lx - 0.68) / 0.32) * PI).sin() * 0.034
        };
        let butt = ly + n1.fbm(u, v, 0.05, 2) * 0.01;
        let lift = 0.022 * (1.0 - ly) + (ra - 0.5) * 0.004;
        let mut h = prof + lift + n2.fbm(u, v, 0.01, 3) * 0.0003;
        h -= (1.0 - smooth(0.0, 0.06, butt)) * 0.004;
        let base = colours[((ra * 5.0) as usize) % 5];
        let vv = 0.93 + 0.14 * rb;
        let trough = if lx < 0.68 {
            ((lx / 0.68) * PI).sin()
        } else {
            0.0
        };
        let roll = if lx >= 0.68 {
            (((lx - 0.68) / 0.32) * PI).sin()
        } else {
            0.0
        };
        let mott = 1.0 + n1.fbm(u, v, 0.08, 4) * 0.08 + n2.fbm(u, v, 0.02, 2) * 0.04;
        let k = vv * mott * (1.0 - trough * 0.08 + roll * 0.06);
        let dirt = smooth(0.0, 0.7, n2.fbm(u, v, 0.7, 4)) * 0.3 * (0.5 + trough * 0.5);
        let mut r = mix(base[0] * k, 0.3, dirt * 0.5);
        let mut g = mix(base[1] * k, 0.27, dirt * 0.5);
        let mut b = mix(base[2] * k, 0.24, dirt * 0.5);
        let mut rough = 0.78;
        let moss_m = smooth(
            0.15,
            0.45,
            nm.fbm(u, v, 0.4, 4) + nm.fbm(u, v, 0.03, 3) * 0.4,
        ) * (1.0 - smooth(0.1, 0.45, ly))
            * (0.4 + trough * 0.6);
        let cell = worley(1707, u * fl as f32, v * fl as f32, fl, fl, 1.0, false);
        if (cell.id & 255) < 40 && cell.f1 < 0.5 && nm.fbm(u + 0.4, v, 0.35, 3) > 0.3 {
            r = mix(r, 0.6, 0.55);
            g = mix(g, 0.58, 0.55);
            b = mix(b, 0.46, 0.55);
            h += 0.0005;
        }
        if moss_m > 0.0 {
            let mk = 0.8 + nm.fbm(u, v, 0.006, 2) * 0.3;
            r = mix(r, 0.26 * mk, moss_m);
            g = mix(g, 0.32 * mk, moss_m);
            b = mix(b, 0.14 * mk, moss_m);
            rough = mix(rough, 0.95, moss_m);
            h += moss_m * 0.003 * (0.5 + nm.fbm(u, v, 0.006, 2));
        }
        t.set(idx, r, g, b, rough, h);
    });
}

/// Natural slate: 0.25 m wide, 0.2 m exposure, half-bond.
pub(crate) fn slate(t: &mut Tex) {
    let w = t.w;
    let nc = ((w / 0.25).round() as i32).max(1);
    let nr = ((w / 0.2).round() as i32).max(1);
    let (sw, e) = (w / nc as f32, w / nr as f32);
    let (n1, n2, n3) = (
        TNoise::new(1801, w),
        TNoise::new(1802, w),
        TNoise::new(1803, w),
    );
    let colours = [
        rgb(70, 76, 86),
        rgb(62, 66, 74),
        rgb(80, 74, 82),
        rgb(74, 80, 80),
        rgb(58, 62, 70),
    ];
    t.each(|t, u, v, x, y, idx| {
        let row = ((y / e).floor() as i32).rem_euclid(nr);
        let ly = y / e - (y / e).floor();
        let xx = x + (row % 2) as f32 * sw * 0.5 + (rnd1(1804, row as u32) - 0.5) * 0.02;
        let col = (xx / sw).floor();
        let lx = xx / sw - col;
        let sid = (row * 64 + (col as i32).rem_euclid(nc)) as u32;
        let (ra, rb, rc) = (rnd1(1805, sid), rnd1(1806, sid), rnd1(1807, sid));
        let gap_d = lx.min(1.0 - lx) * sw - 0.002 - rc * 0.002;
        let gap = smooth(0.0, 0.001, gap_d);
        let butt_n = n1.fbm(u, v, 0.03, 3) * 0.006 + n2.fbm(u, v, 0.012, 2).abs() * 0.004;
        let butt = ly * e - butt_n;
        let nose = smooth(0.0, 0.004, butt);
        let cleave = n3.ridged_g(u, v, 0.08, 4, 0.5, 0.25) * 0.0008;
        let hs = 0.008 * (1.0 - ly) + (lx - 0.5) * (rb - 0.5) * 0.003 + cleave + (ra - 0.5) * 0.002;
        let h = mix(0.0, hs, gap) - (1.0 - nose) * 0.002;
        let base = colours[((ra * 5.0) as usize) % 5];
        let k = (0.88 + 0.24 * rb) * (1.0 + n1.fbm(u, v, 0.1, 4) * 0.06 + cleave * 30.0);
        let gk = 0.5 + 0.5 * gap;
        let bloom = smooth(0.35, 0.7, n2.fbm(u, v, 0.4, 4)) * 0.25;
        let wear = 1.0 + (1.0 - smooth(0.0, 0.01, butt)) * 0.15;
        let c = |i: usize, s: f32| mix(base[i] * k * gk, s, bloom * 0.4) * wear;
        t.set(
            idx,
            c(0, 0.48),
            c(1, 0.48),
            c(2, 0.46),
            0.62 + (rb - 0.5) * 0.1 + bloom * 0.2,
            h,
        );
    });
}

/// Paris zinc standing seam: 0.5 m pans, rounded seams, cross welts.
pub(crate) fn zinc(t: &mut Tex) {
    let w = t.w;
    let p = 0.5f32;
    let np = ((w / p).round() as i32).max(1);
    let (n1, n2, n3) = (
        TNoise::new(1901, w),
        TNoise::new(1902, w),
        TNoise::new(1903, w),
    );
    t.each(|t, u, v, x, y, idx| {
        let d = wrapd(x, p).abs();
        let pan = ((x / p).round() as i32).rem_euclid(np) as u32;
        let seam = if d < 0.012 {
            (1.0 - (d / 0.012).powi(2)).sqrt()
        } else {
            0.0
        };
        let mut h = seam * 0.028
            + if (0.012..0.018).contains(&d) {
                (1.0 - (d - 0.012) / 0.006) * 0.003
            } else {
                0.0
            };
        let in_pan = smooth(0.015, 0.05, d);
        h += n1.fbm(u + pan as f32 * 0.13, v, 0.5, 3) * 0.0015 * in_pan;
        let wy = wrapd(y - rnd1(1904, pan) * w, w).abs();
        h += (1.0 - smooth(0.0, 0.008, wy)) * 0.005 * in_pan;
        let welt = (1.0 - smooth(0.006, 0.01, wy)) * (1.0 - smooth(0.0, 0.004, (wy - 0.008).abs()));
        let pat = n2.fbm_g(u, v, 0.4, 5, 0.55, 0.4);
        let white = smooth(0.2, 0.55, pat) * 0.18;
        let run = smooth(0.1, 0.9, n3.fbm_g(u, v, 0.08, 3, 0.5, 1.5)) * 0.14;
        let tone = 1.0 + (rnd1(1905, pan) - 0.5) * 0.06 + seam * 0.04;
        let r = mix(0.58 * tone, 0.72, white) * (1.0 - run - welt * 0.15);
        let g = mix(0.615 * tone, 0.73, white) * (1.0 - run - welt * 0.15);
        let b = mix(0.645 * tone, 0.73, white) * (1.0 - run * 0.9 - welt * 0.15);
        t.set(idx, r, g, b, 0.48 + white * 0.6 + run * 0.4, h);
    });
}

/// Three-tab asphalt shingles.
pub(crate) fn asphalt_shingle(t: &mut Tex) {
    let w = t.w;
    let nr = ((w / 0.1429).round() as i32).max(1);
    let e = w / nr as f32;
    let nt = ((w / 0.3333).round() as i32).max(1);
    let tw = w / nt as f32;
    let (n1, n2) = (TNoise::new(2001, w), TNoise::new(2002, w));
    let granules = [
        rgb(112, 110, 106),
        rgb(66, 66, 66),
        rgb(118, 98, 82),
        rgb(150, 146, 138),
        rgb(90, 84, 78),
    ];
    t.each(|t, u, v, x, y, idx| {
        let row = ((y / e).floor() as i32).rem_euclid(nr);
        let ly = y / e - (y / e).floor();
        let xx = x + (row % 2) as f32 * tw * 0.5 + (rnd1(2003, row as u32) - 0.5) * 0.01;
        let col = (xx / tw).floor();
        let lx = (xx / tw - col) * tw;
        let tid = (row * 64 + (col as i32).rem_euclid(nt)) as u32;
        let ra = rnd1(2004, tid);
        let in_slot = lx.min(tw - lx) - 0.003 < 0.0 && ly < 0.9;
        let mut h = 0.004 * (1.0 - ly) + 0.0005 * n1.fbm(u, v, 0.005, 2);
        h -= (1.0 - smooth(0.0, 0.08, ly)) * 0.001;
        let gi = (rnd1(2005, idx as u32) * 100.0) as u32;
        let gc = granules[match gi {
            0..=44 => 0,
            45..=69 => 1,
            70..=84 => 2,
            85..=92 => 3,
            _ => 4,
        }];
        let blend = 0.92 + 0.16 * ra + n2.fbm(u, v, 0.5, 3) * 0.08;
        let sh = 1.0 - (1.0 - smooth(0.0, 0.12, ly)) * 0.1;
        let mut k = blend * sh;
        if in_slot {
            h = 0.0005;
            k *= 0.55;
        }
        let wash = smooth(0.2, 0.8, n1.fbm(u, v, 0.6, 4)) * 0.12;
        h += (rnd1(2006, idx as u32) - 0.5) * 0.0004;
        t.set(
            idx,
            mix(gc[0] * k, 0.45, wash),
            mix(gc[1] * k, 0.44, wash),
            mix(gc[2] * k, 0.42, wash),
            0.92,
            h,
        );
    });
}

/// Corrugated galvanised steel (76 mm pitch), screws on purlin lines, rust.
pub(crate) fn corrugated(t: &mut Tex) {
    let w = t.w;
    let nc = ((w / 0.076).round() as i32).max(1);
    let p = w / nc as f32;
    let (n1, n2, n3) = (
        TNoise::new(2101, w),
        TNoise::new(2102, w),
        TNoise::new(2103, w),
    );
    let fs = (w / 0.04).round() as i32;
    t.each(|t, u, v, x, y, idx| {
        let mut h = ((x / p) * TAU).cos() * 0.009;
        let side_lap = wrapd(x, p * 13.0).abs() < 0.004;
        let ly = wrap(y, w);
        if ly < 0.15 {
            h += 0.0015;
        }
        let end_line = 1.0 - smooth(0.0, 0.004, (ly - 0.15).abs());
        let mut screw = 0.0f32;
        let mut rust_src = 0.0f32;
        for py in [0.075f32, 1.0] {
            let dy = wrapd(y - py, w);
            let k = (x / (p * 3.0)).round();
            let dx = x - k * p * 3.0;
            let d = (dx * dx + dy * dy).sqrt();
            if d < 0.009 {
                screw = screw.max((1.0 - (d / 0.009).powi(2)).sqrt());
            }
            let stream = if dy < 0.0 && dy > -0.6 {
                (1.0 + dy / 0.6)
                    * smooth(0.012, 0.002, dx.abs())
                    * if rnd(2104, k as i32 as u32, (py * 10.0) as u32) > 0.4 {
                        1.0
                    } else {
                        0.0
                    }
            } else {
                0.0
            };
            rust_src = rust_src.max(smooth(0.03, 0.0, d)).max(stream * 0.8);
        }
        h += screw * 0.006;
        let rust_n = n1.fbm_g(u, v, 0.3, 5, 0.6, 0.3);
        let rust = clamp01(
            smooth(0.25, 0.6, rust_n) * 0.8
                + rust_src * (0.5 + 0.5 * n2.fbm(u, v, 0.02, 2))
                + end_line * 0.3 * smooth(0.0, 0.5, rust_n),
        );
        let cell = worley(2105, u * fs as f32, v * fs as f32, fs, fs, 1.0, false);
        let sp = 1.0 + ((cell.id & 255) as f32 / 255.0 - 0.5) * 0.08;
        let dull = smooth(0.0, 0.6, n3.fbm(u, v, 0.7, 4)) * 0.15;
        let rk = 0.85 + n2.fbm(u, v, 0.01, 3) * 0.25;
        let mut r = mix(mix(0.6 * sp, 0.68, dull), 0.48 * rk, rust);
        let mut g = mix(mix(0.61 * sp, 0.68, dull), 0.26 * rk, rust);
        let mut b = mix(mix(0.6 * sp, 0.66, dull), 0.16 * rk, rust);
        let rough = mix(0.42 + dull, 0.88, rust);
        h += rust * n2.fbm(u, v, 0.008, 2) * 0.0004;
        let mut k = 1.0 - end_line * 0.3;
        if side_lap {
            k *= 0.7;
        }
        r *= k;
        g *= k;
        b *= k;
        if screw > 0.0 {
            r = mix(r, 0.5, screw);
            g = mix(g, 0.5, screw);
            b = mix(b, 0.52, screw);
        }
        t.set(idx, r, g, b, rough, h);
    });
}

/// Pebble ballast on a flat roof.
pub(crate) fn gravel_roof(t: &mut Tex) {
    pebbles(
        t,
        2201,
        &Pebbles {
            count: 2.6,
            r_min: 0.008,
            r_max: 0.016,
            height: 0.6,
            palette: &[
                rgb(150, 146, 138),
                rgb(170, 156, 132),
                rgb(96, 92, 88),
                rgb(196, 192, 182),
                rgb(136, 116, 96),
                rgb(120, 120, 124),
            ],
            fines: rgb(80, 76, 70),
            fines_h: -0.004,
            ..Pebbles::default()
        },
    );
}

/// Extensive green roof: sedum rosettes in species patches over substrate.
pub(crate) fn green_roof(t: &mut Tex) {
    let w = t.w;
    let (ns, nb) = (TNoise::new(2301, w), TNoise::new(2302, w));
    t.each(|t, u, v, _, _, idx| {
        let k = 1.0 + nb.fbm(u, v, 0.03, 3) * 0.2 + (rnd1(2303, idx as u32) - 0.5) * 0.2;
        t.set(
            idx,
            0.36 * k,
            0.29 * k,
            0.22 * k,
            0.95,
            -0.01 + nb.fbm(u, v, 0.1, 3) * 0.003,
        );
    });
    let mut r = rng(2304);
    let species: [(Rgb, Rgb); 4] = [
        (rgb(96, 128, 56), rgb(150, 160, 70)),
        (rgb(120, 136, 60), rgb(176, 160, 70)),
        (rgb(104, 110, 58), rgb(164, 84, 62)),
        (rgb(80, 112, 70), rgb(120, 150, 100)),
    ];
    let n = ((w * w) / (PI * 0.02 * 0.02) * 2.2).round() as u32;
    for _ in 0..n {
        let (cx, cy) = (r.float() * w, r.float() * w);
        let bare = nb.fbm(cx / w, cy / w, 0.8, 4);
        if bare < -0.45 && r.float() < 0.7 {
            continue;
        }
        let pick =
            clamp01(ns.fbm(cx / w, cy / w, 0.9, 3) * 0.9 + 0.5 + (r.float() - 0.5) * 0.5) * 3.999;
        let (c0, c1) = species[pick as usize];
        let (rad, np, rot) = (r.range(0.012, 0.03), r.int(5, 8) as f32, r.float() * TAU);
        let z0 = r.range(0.0, 0.01);
        let vk = r.range(0.85, 1.15);
        splat(t, cx, cy, rad, |t, dx, dy, idx| {
            let d = (dx * dx + dy * dy).sqrt() / rad;
            let a = dy.atan2(dx) * np + rot;
            let petal = 0.55 + 0.45 * a.cos().max(0.0);
            if d > petal {
                return;
            }
            let z = z0 + (1.0 - d * d) * rad * 0.8 + a.cos() * 0.001;
            if z <= t.h[idx] {
                return;
            }
            let f = (d / petal).powi(2);
            let sh = (0.75 + 0.25 * a.cos()) * vk;
            t.set(
                idx,
                mix(c0[0], c1[0], f) * sh,
                mix(c0[1], c1[1], f) * sh,
                mix(c0[2], c1[2], f) * sh,
                0.7,
                z,
            );
        });
    }
}
