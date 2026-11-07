//! Urban ground layers: asphalt, road paint, paving, curbs, plaza stone,
//! cobbles, gravel paths, dirt, lawn, promenade and quay walls.

#![allow(clippy::needless_range_loop)]

use super::core::{blur, clamp01, mix, rgb, rnd1, smooth, worley, wrap, Rgb, TNoise, Tex};
use super::layout::{block_dist, random_courses, running_bond, Hit};
use super::masonry::{stone_blocks, StoneOpts};
use super::paint::{pebbles, rng, splat, stroke, Pebbles};
use std::f32::consts::TAU;

struct Asphalt {
    binder: f32,
    /// Share of exposed aggregate.
    expose: f32,
    stone: f32,
    cracks: f32,
    sealed: f32,
    oil: f32,
    ravel: f32,
}

fn asphalt(t: &mut Tex, seed: u32, o: Asphalt) {
    let w = t.w;
    let (n1, n2, n3, n4) = (
        TNoise::new(seed + 1, w),
        TNoise::new(seed + 2, w),
        TNoise::new(seed + 3, w),
        TNoise::new(seed + 4, w),
    );
    let fa = (w / 0.012).round() as i32;
    let fb = (w / 0.03).round() as i32;
    let fc = (w / 0.9).round() as i32;
    let fs = (w / 1.7).round() as i32;
    t.each(|t, u, v, _, _, idx| {
        let c = worley(seed + 6, u * fa as f32, v * fa as f32, fa, fa, 1.0, false);
        let ex = o.expose * (1.0 + n3.fbm(u, v, 0.5, 3) * 0.6);
        let mut stone = if ((c.id & 255) as f32) < ex * 255.0 {
            smooth(0.55, 0.3, c.f1)
        } else {
            0.0
        };
        let mut sc = o.stone * (0.75 + 0.5 * ((c.id >> 8) & 255) as f32 / 255.0);
        let c2 = worley(seed + 7, u * fb as f32, v * fb as f32, fb, fb, 1.0, false);
        if ((c2.id & 255) as f32) < ex * 120.0 && c2.f1 < 0.4 {
            stone = stone.max(smooth(0.4, 0.25, c2.f1));
            sc = o.stone * (0.8 + 0.4 * ((c2.id >> 8) & 255) as f32 / 255.0);
        }
        let mut h = stone * 0.0015 + n1.fbm(u, v, 1.0, 3) * 0.003 + n1.fbm(u, v, 0.1, 3) * 0.0006;
        let bk = o.binder
            * (1.0
                + n2.fbm(u, v, 0.3, 4) * 0.1
                + n1.fbm(u + 0.4, v, 2.0, 3) * 0.12
                + (rnd1(seed, idx as u32) - 0.5) * 0.15);
        let mut r = mix(bk, sc, stone);
        let mut g = mix(bk, sc * 0.99, stone);
        let mut b = mix(bk * 1.02, sc * 0.97, stone);
        let mut rough = mix(0.9, 0.75, stone);
        // Ravelling: pits where the aggregate fell out.
        let rav = smooth(0.3, 0.6, n3.fbm(u, v, 0.4, 4))
            * o.ravel
            * if rnd1(seed + 8, idx as u32) < 0.35 {
                1.0
            } else {
                0.0
            };
        h -= rav * 0.003;
        r *= 1.0 - rav * 0.3;
        g *= 1.0 - rav * 0.3;
        b *= 1.0 - rav * 0.3;
        let cr_area = smooth(0.12, 0.4, n4.fbm(u, v, 1.5, 3)) * o.cracks;
        if cr_area > 0.0 {
            let cc = worley(
                seed + 9,
                u * fc as f32 + n4.fbm(u, v, 0.15, 3) * 0.3,
                v * fc as f32 + n4.fbm(u + 0.3, v, 0.15, 3) * 0.3,
                fc,
                fc,
                1.0,
                true,
            );
            let cw = cc.edge * (w / fc as f32);
            let cr = (1.0 - smooth(0.0015, 0.005, cw)) * cr_area;
            h -= cr * 0.006;
            r *= 1.0 - cr * 0.6;
            g *= 1.0 - cr * 0.6;
            b *= 1.0 - cr * 0.6;
        }
        let seal_area = if o.sealed > 0.0 {
            smooth(-0.1, 0.2, n1.fbm(u, v, 2.0, 2))
        } else {
            0.0
        };
        if seal_area > 0.0 {
            let cs = worley(
                seed + 10,
                u * fs as f32 + n2.fbm(u, v, 0.3, 3) * 0.25,
                v * fs as f32 + n2.fbm(u + 0.5, v, 0.3, 3) * 0.25,
                fs,
                fs,
                1.0,
                true,
            );
            let sw = cs.edge * (w / fs as f32);
            let sm = smooth(
                0.004,
                -0.004,
                n4.fbm(u + 0.7, v, 0.5, 2) * 0.006 + sw - 0.014,
            ) * o.sealed
                * seal_area;
            r = mix(r, 0.085, sm);
            g = mix(g, 0.085, sm);
            b = mix(b, 0.09, sm);
            rough = mix(rough, 0.45, sm);
            h = mix(h, 0.0012, sm);
        }
        let oil = smooth(0.25, 0.7, n2.fbm_g(u + 0.2, v, 0.5, 5, 0.6, 0.5)) * o.oil;
        r *= 1.0 - oil * 0.35;
        g *= 1.0 - oil * 0.35;
        b *= 1.0 - oil * 0.33;
        rough -= oil * 0.25 + smooth(0.0, 0.6, n3.fbm_g(u, v, 0.8, 3, 0.5, 0.15)) * 0.08;
        t.set(idx, r, g, b, rough, h);
    });
}

pub(crate) fn road_asphalt(t: &mut Tex) {
    asphalt(
        t,
        3001,
        Asphalt {
            binder: 0.2,
            expose: 0.4,
            stone: 0.37,
            cracks: 0.7,
            sealed: 0.8,
            oil: 0.4,
            ravel: 0.2,
        },
    );
}

pub(crate) fn light_asphalt(t: &mut Tex) {
    asphalt(
        t,
        3101,
        Asphalt {
            binder: 0.3,
            expose: 0.6,
            stone: 0.5,
            cracks: 1.0,
            sealed: 0.3,
            oil: 0.9,
            ravel: 0.7,
        },
    );
}

fn road_paint(t: &mut Tex, color: Rgb, seed: u32) {
    asphalt(
        t,
        seed,
        Asphalt {
            binder: 0.2,
            expose: 0.4,
            stone: 0.4,
            cracks: 0.0,
            sealed: 0.0,
            oil: 0.0,
            ravel: 0.1,
        },
    );
    let w = t.w;
    let (n1, n2) = (TNoise::new(seed + 20, w), TNoise::new(seed + 21, w));
    let fc = (w / 0.08).round() as i32;
    let hb = blur(&t.h, t.size, 3.0);
    t.each(|t, u, v, _, _, idx| {
        let hgt = t.h[idx] - hb[idx];
        // Paint wears off the high points and in worn patches.
        let wear =
            n1.fbm_g(u, v, 0.35, 5, 0.6, 0.35) * 0.3 + n2.fbm(u, v, 0.02, 3) * 0.3 + hgt * 300.0;
        let pm = smooth(0.42, 0.34, wear);
        let c = worley(
            seed + 22,
            u * fc as f32,
            v * fc as f32 + n2.fbm(u, v, 0.05, 2) * 0.3,
            fc,
            fc,
            1.0,
            true,
        );
        let crack =
            (1.0 - smooth(0.0, 0.025, c.edge)) * smooth(0.1, 0.4, n1.fbm(u + 0.5, v, 0.5, 3)) * 0.6;
        let dirt = 1.0
            - smooth(-0.2, 0.7, n1.fbm(u, v, 0.3, 4)) * 0.07
            - (rnd1(seed + 23, idx as u32) - 0.5) * 0.05;
        let k = dirt * (1.0 - crack * 0.35);
        t.r[idx] = mix(t.r[idx], color[0] * k, pm);
        t.g[idx] = mix(t.g[idx], color[1] * k, pm);
        t.b[idx] = mix(t.b[idx], color[2] * k, pm);
        t.rough[idx] = mix(t.rough[idx], 0.6, pm);
        t.h[idx] += pm * 0.0008 - crack * pm * 0.0004;
    });
}

pub(crate) fn road_paint_white(t: &mut Tex) {
    road_paint(t, rgb(222, 220, 212), 4101);
}

pub(crate) fn road_paint_yellow(t: &mut Tex) {
    road_paint(t, rgb(226, 170, 40), 4201);
}

/// Sidewalk concrete slabs 1.5 × 1.5 m: tooled margins, broom finish, gum, stains.
pub(crate) fn paving(t: &mut Tex) {
    let w = t.w;
    let lay = running_bond(w, 2, 2, 3201, 0.0, 0.0);
    let (n1, n2, n3) = (
        TNoise::new(3202, w),
        TNoise::new(3203, w),
        TNoise::new(3204, w),
    );
    let fg = (w / 0.12).round() as i32;
    t.each(|t, u, v, x, y, idx| {
        let hit = lay.at(x, y);
        let id = hit.id;
        let (ra, rb) = (rnd1(3205, id), rnd1(3206, id));
        let e = block_dist(&hit, 0.006, 0.006) + n1.fbm(u, v, 0.05, 2) * 0.0015;
        let mask = smooth(-0.002, 0.002, e);
        let margin = 1.0 - smooth(0.035, 0.045, e);
        let lx = hit.x - (hit.x0 + hit.x1) / 2.0;
        let ly = hit.y - (hit.y0 + hit.y1) / 2.0;
        let mut hs = 0.008 + (ra - 0.5) * 0.004 + lx * (rb - 0.5) * 0.004 + ly * (ra - 0.5) * 0.003;
        hs -= (1.0 - smooth(0.0, 0.008, e)) * 0.003;
        let broom = n2.fbm_g(u, v, 0.25, 3, 0.5, 0.004) * (1.0 - margin);
        hs += broom * 0.0003 + n3.fbm(u, v, 0.006, 2) * 0.0002;
        let hm = -0.004 + n1.fbm(u, v, 0.01, 2) * 0.001;
        let mut h = mix(hm, hs, mask);
        let tone = 1.0
            + (ra - 0.5) * 0.08
            + n1.fbm(u, v, 0.4, 4) * 0.05
            + broom * 0.03
            + margin * 0.03
            + (rnd1(3207, idx as u32) - 0.5) * 0.07
            + n3.fbm(u, v, 0.012, 2) * 0.04;
        let st = smooth(0.25, 0.65, n2.fbm_g(u + 0.3, v, 0.6, 5, 0.6, 0.6)) * 0.22;
        let edge_dirt = (1.0 - smooth(0.0, 0.03, e)) * 0.2;
        let mut r = 0.64 * tone * (1.0 - st) * (1.0 - edge_dirt);
        let mut g = 0.625 * tone * (1.0 - st) * (1.0 - edge_dirt);
        let mut b = 0.6 * tone * (1.0 - st * 0.95) * (1.0 - edge_dirt);
        let mut rough = mix(0.9, 0.8, margin);
        // Chewing gum.
        let c = worley(3208, u * fg as f32, v * fg as f32, fg, fg, 1.0, false);
        let gr = 0.06 + ((c.id >> 8) & 255) as f32 / 255.0 * 0.08;
        if (c.id & 255) < 22 && c.f1 < gr {
            let gk = smooth(gr, gr * 0.6, c.f1);
            let gc = 0.28 + ((c.id >> 16) & 255) as f32 / 255.0 * 0.12;
            r = mix(r, gc, gk);
            g = mix(g, gc, gk);
            b = mix(b, gc * 1.02, gk);
            rough = mix(rough, 0.55, gk);
            h += gk * 0.0006;
        }
        let jc = 0.22 + n3.fbm(u, v, 0.02, 2) * 0.04;
        t.set(
            idx,
            mix(jc * 1.05, r, mask),
            mix(jc, g, mask),
            mix(jc * 0.9, b, mask),
            mix(1.0, rough, mask),
            h,
        );
    });
}

/// Granite speckle over a stone colour (feldspar, quartz, mica).
fn granite_speckle(
    seed: u32,
    w: f32,
    lambda: f32,
    palette: Vec<Rgb>,
    weights: &[f32],
) -> impl Fn(f32, f32, f32, &mut [f32; 5]) {
    let f = (w / lambda).round() as i32;
    let mut cum = Vec::new();
    let mut acc = 0.0;
    for wt in weights {
        acc += wt;
        cum.push(acc);
    }
    move |u, v, mask, c| {
        let cell = worley(seed, u * f as f32, v * f as f32, f, f, 1.0, false);
        let m = (cell.id & 1023) as f32 / 1024.0 * acc;
        let mut p = 0;
        while p < cum.len() - 1 && m > cum[p] {
            p += 1;
        }
        let col = palette[p];
        let k = mask * (0.7 + 0.3 * smooth(0.6, 0.4, cell.f1));
        for i in 0..3 {
            c[i] = mix(c[i], col[i], k);
        }
    }
}

fn stone_base(seed: u32, layout: super::layout::Layout) -> StoneOpts<'static> {
    StoneOpts {
        seed,
        layout,
        joint: 0.006,
        depth: 0.003,
        bevel: 0.003,
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
        soot: 0.2,
        streaks: 0.0,
        rough: 0.7,
        rough_joint: 0.95,
        erosion: 0.0,
        extra: None,
    }
}

pub(crate) fn curb(t: &mut Tex) {
    let sp = granite_speckle(
        3301,
        t.w,
        0.008,
        vec![
            rgb(150, 148, 144),
            rgb(120, 118, 116),
            rgb(178, 176, 172),
            rgb(50, 50, 52),
        ],
        &[40.0, 30.0, 20.0, 10.0],
    );
    let extra =
        move |u: f32, v: f32, mask: f32, _: &Hit, c: &mut [f32; 5]| sp(u, v, mask * 0.55, c);
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.008,
            depth: 0.004,
            bevel: 0.006,
            ragged: 0.002,
            base: rgb(140, 138, 134),
            vary: 0.04,
            hue: 0.01,
            grain: 0.03,
            mortar: rgb(76, 74, 70),
            joint_dirt: 0.7,
            soot: 0.5,
            rough: 0.68,
            erosion: 0.2,
            extra: Some(&extra),
            ..stone_base(3302, random_courses(t.w, &[1.0, 1.0, 1.0], 0.8, 1.6, 3303))
        },
    );
}

pub(crate) fn plaza_stone(t: &mut Tex) {
    stone_blocks(
        t,
        &StoneOpts {
            bevel: 0.003,
            ragged: 0.0008,
            bulge: 0.0006,
            base: rgb(180, 172, 160),
            vary: 0.06,
            hue: 0.03,
            grain: 0.06,
            mortar: rgb(110, 104, 96),
            joint_dirt: 0.6,
            soot: 0.25,
            rough: 0.72,
            erosion: 0.1,
            ..stone_base(
                3401,
                random_courses(
                    t.w,
                    &[0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4],
                    0.5,
                    1.1,
                    3402,
                ),
            )
        },
    );
}

pub(crate) fn promenade(t: &mut Tex) {
    let n = TNoise::new(3701, t.w);
    let extra = |u: f32, v: f32, mask: f32, _: &Hit, c: &mut [f32; 5]| {
        let salt = smooth(0.1, 0.6, n.fbm(u, v, 0.8, 4)) * 0.05 * mask;
        c[0] += salt;
        c[1] += salt;
        c[2] += salt;
    };
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.005,
            depth: 0.002,
            bevel: 0.003,
            ragged: 0.0006,
            bulge: 0.0005,
            base: rgb(192, 186, 174),
            vary: 0.04,
            hue: 0.02,
            grain: 0.05,
            mortar: rgb(130, 124, 114),
            joint_dirt: 0.5,
            soot: 0.15,
            rough: 0.66,
            rough_joint: 0.9,
            erosion: 0.05,
            extra: Some(&extra),
            ..stone_base(3702, running_bond(t.w, 8, 4, 3703, 0.01, 0.5))
        },
    );
}

/// Granite setts laid in segmental arcs, domed tops, sand and dirt joints.
pub(crate) fn cobble(t: &mut Tex) {
    let w = t.w;
    let (rh, p, sag) = (0.1f32, 1.0f32, 0.1f32);
    let nr = ((w / rh).round() as i32).max(1);
    let (nw, n1, nm) = (
        TNoise::new(3501, w),
        TNoise::new(3502, w),
        TNoise::new(3503, w),
    );
    let mut r0 = rng(3504);
    let rows: Vec<Vec<f32>> = (0..nr)
        .map(|_| {
            let mut xs = Vec::new();
            let mut x = r0.float() * 0.1;
            while x < w {
                xs.push(x);
                x += r0.range(0.085, 0.14);
            }
            let k = w / x;
            xs.into_iter().map(|q| q * k).collect()
        })
        .collect();
    let stones = [
        rgb(128, 124, 120),
        rgb(112, 110, 110),
        rgb(98, 99, 102),
        rgb(134, 122, 112),
        rgb(140, 134, 122),
        rgb(120, 116, 110),
    ];
    t.each(|t, u, v, x, y, idx| {
        let wx = x + nw.fbm(u, v, 0.4, 3) * 0.02;
        let wy = y + nw.fbm(u + 0.5, v, 0.4, 3) * 0.02;
        let arc = (wx / p).floor();
        let xm = wx - (arc + 0.5) * p;
        let yy = wy + sag * (2.0 * xm / p).powi(2);
        let row_f = yy / rh;
        let row = row_f.floor();
        let ly = (row_f - row) * rh;
        let rr = (row as i32).rem_euclid(nr) as usize;
        let xs = &rows[rr];
        let xq = wrap(wx, w);
        let mut b = xs.len() - 1;
        for (k, v) in xs.iter().enumerate() {
            if *v <= xq {
                b = k;
            } else {
                break;
            }
        }
        let x0 = if xs[b] > xq { xs[b] - w } else { xs[b] };
        let x1 = if b + 1 < xs.len() {
            xs[b + 1]
        } else {
            xs[0] + w
        };
        let slope = (1.0 + (8.0 * sag * xm / (p * p)).powi(2)).sqrt();
        let dx = (xq - x0).min(x1 - xq);
        let dy = ly.min(rh - ly) / slope;
        let cusp = p / 2.0 - xm.abs();
        let arc_odd = (arc as i32).rem_euclid(2) as usize;
        let sid = (rr * 256 + b + if cusp < 0.15 { arc_odd * 65536 } else { 0 }) as u32;
        let (ra, rb, rc) = (rnd1(3505, sid), rnd1(3506, sid), rnd1(3507, sid));
        let hj = 0.005 + rc * 0.003;
        let (ex, ey, cr) = (dx.min(cusp) - hj, dy - hj, 0.022f32);
        let e0 = if ex < cr && ey < cr {
            cr - ((cr - ex).powi(2) + (cr - ey).powi(2)).sqrt()
        } else {
            ex.min(ey)
        };
        let e = e0 + n1.fbm(u, v, 0.03, 3) * 0.006;
        let mask = smooth(-0.001, 0.002, e);
        let dome = clamp01(e / 0.03).sqrt();
        let hs = 0.01 + dome * 0.012 + (ra - 0.5) * 0.006 + n1.fbm(u, v, 0.01, 3) * 0.0008;
        let hj0 = -0.004 + n1.fbm(u, v, 0.01, 2) * 0.002;
        let h = mix(hj0, hs, mask);
        let sc = stones[((rb * stones.len() as f32) as usize) % stones.len()];
        let tone = (0.85 + 0.3 * ra)
            * (1.0 + n1.fbm(u, v, 0.012, 2) * 0.08 + (rnd1(3508, idx as u32) - 0.5) * 0.12);
        let polish = smooth(0.0, 0.035, e);
        let dirt = (1.0 - smooth(0.0, 0.02, e)) * 0.25;
        let k = tone * (1.0 + polish * 0.05) * (1.0 - dirt);
        let moss = smooth(0.2, 0.5, nm.fbm(u, v, 0.6, 4));
        let jk = 1.0 + n1.fbm(u, v, 0.006, 2) * 0.2;
        let joint = [
            mix(0.3, 0.2, moss) * jk,
            mix(0.27, 0.26, moss) * jk,
            mix(0.23, 0.12, moss) * jk,
        ];
        let rough = mix(0.97, mix(0.82, 0.55, polish), mask);
        t.set(
            idx,
            mix(joint[0], sc[0] * k, mask),
            mix(joint[1], sc[1] * k, mask),
            mix(joint[2], sc[2] * k, mask),
            rough,
            h,
        );
    });
}

pub(crate) fn gravel_path(t: &mut Tex) {
    let w = t.w;
    let n1 = TNoise::new(3601, w);
    t.each(|t, u, v, _, _, idx| {
        let k = 1.0 + n1.fbm(u, v, 0.3, 4) * 0.08 + (rnd1(3602, idx as u32) - 0.5) * 0.3;
        t.set(
            idx,
            0.6 * k,
            0.55 * k,
            0.46 * k,
            0.95,
            n1.fbm(u, v, 0.15, 3) * 0.003 - 0.003,
        );
    });
    let mask = |u: f32, v: f32| 0.55 + 0.45 * n1.fbm(u, v, 0.4, 3);
    pebbles(
        t,
        3603,
        &Pebbles {
            count: 1.1,
            r_min: 0.004,
            r_max: 0.011,
            height: 0.45,
            bed: false,
            palette: &[
                rgb(170, 160, 140),
                rgb(150, 140, 125),
                rgb(120, 112, 104),
                rgb(190, 182, 166),
                rgb(140, 120, 100),
            ],
            fines: rgb(152, 140, 118),
            dust: 0.35,
            mask: Some(&mask),
            ..Pebbles::default()
        },
    );
}

pub(crate) fn dirt(t: &mut Tex) {
    let w = t.w;
    let (n1, n2, n3) = (
        TNoise::new(3801, w),
        TNoise::new(3802, w),
        TNoise::new(3803, w),
    );
    let fc = (w / 0.12).round() as i32;
    t.each(|t, u, v, _, _, idx| {
        let lump = n1.fbm_g(u, v, 0.06, 4, 0.55, 0.06);
        let mut h = n1.fbm(u, v, 0.8, 3) * 0.012 + lump * 0.003 + n2.fbm(u, v, 0.008, 2) * 0.0006;
        let damp = smooth(-0.2, 0.6, n2.fbm(u, v, 0.9, 4)) * 0.6;
        let c = worley(
            3804,
            u * fc as f32 + n3.fbm(u, v, 0.05, 2) * 0.3,
            v * fc as f32,
            fc,
            fc,
            1.0,
            true,
        );
        let cr = (1.0 - smooth(0.0, 0.04, c.edge))
            * (1.0 - damp)
            * smooth(0.0, 0.3, n3.fbm(u, v, 0.7, 3));
        h -= cr * 0.003;
        let k = (1.0 + lump * 0.1 + (rnd1(3805, idx as u32) - 0.5) * 0.12)
            * (1.0 - damp * 0.28)
            * (1.0 - cr * 0.3);
        t.set(
            idx,
            0.49 * k,
            0.4 * k * (1.0 - damp * 0.05),
            0.3 * k,
            0.95 - damp * 0.15,
            h,
        );
    });
    pebbles(
        t,
        3806,
        &Pebbles {
            count: 0.12,
            r_min: 0.003,
            r_max: 0.012,
            height: 0.4,
            bed: false,
            palette: &[rgb(140, 128, 112), rgb(110, 100, 90), rgb(160, 150, 132)],
            fines: rgb(125, 102, 76),
            dust: 0.4,
            ..Pebbles::default()
        },
    );
}

/// Lawn: z-buffered blades over thatch, clumps, dry patches and clover.
pub(crate) fn grass(t: &mut Tex) {
    let w = t.w;
    let (nd, nc, nt) = (
        TNoise::new(3901, w),
        TNoise::new(3902, w),
        TNoise::new(3903, w),
    );
    t.each(|t, u, v, _, _, idx| {
        let k = 0.8 + nt.fbm(u, v, 0.05, 3) * 0.3 + (rnd1(3904, idx as u32) - 0.5) * 0.3;
        t.set(idx, 0.2 * k, 0.19 * k, 0.09 * k, 0.95, 0.0);
    });
    let mut r = rng(3905);
    let n = (((w * w) / (0.03 * t.ps)) * 1.6).round() as u32;
    for _ in 0..n {
        let (x, y) = (r.float() * w, r.float() * w);
        let (u, v) = (x / w, y / w);
        if r.float() > 0.65 + 0.35 * nc.fbm(u, v, 0.12, 3) {
            continue;
        }
        let dry = smooth(
            0.15,
            0.55,
            nd.fbm(u, v, 1.2, 4) + nd.fbm(u, v, 0.15, 2) * 0.3,
        );
        let (len, a, z0, hue, vk) = (
            r.range(0.015, 0.045),
            r.float() * TAU,
            r.float() * 0.012,
            r.float(),
            r.range(0.7, 1.25),
        );
        let dk = dry * r.range(0.5, 1.0);
        let cr = mix(mix(0.2, 0.3, hue), 0.55, dk);
        let cg = mix(mix(0.33, 0.4, hue), 0.48, dk);
        let cb = mix(mix(0.09, 0.12, hue), 0.26, dk);
        stroke(t, x, y, a, len, |t, s, idx| {
            let z = z0 + s * len * 0.9;
            if z <= t.h[idx] {
                return;
            }
            let sh = vk * (0.5 + 0.5 * s);
            t.set(idx, cr * sh, cg * sh, cb * sh, 0.7 + dk * 0.2, z);
        });
    }
    for _ in 0..24 {
        let (px, py, pr) = (r.float() * w, r.float() * w, r.range(0.05, 0.15));
        for _ in 0..(pr * pr * 4000.0).round() as u32 {
            let (a, d) = (r.float() * TAU, r.float().sqrt() * pr);
            let (cx, cy) = (px + a.cos() * d, py + a.sin() * d);
            let (z0, rot, rad, vk) = (
                r.range(0.01, 0.035),
                r.float() * TAU,
                r.range(0.006, 0.01),
                r.range(0.8, 1.1),
            );
            splat(t, cx, cy, rad, |t, dx, dy, idx| {
                let an = dy.atan2(dx) * 3.0 + rot;
                let rr = (dx * dx + dy * dy).sqrt() / rad;
                if rr > 0.6 + 0.4 * (an * 0.5).cos().abs() {
                    return;
                }
                let z = z0 + (1.0 - rr) * 0.003;
                if z <= t.h[idx] {
                    return;
                }
                t.set(idx, 0.16 * vk, 0.3 * vk, 0.12 * vk, 0.6, z);
            });
        }
    }
}

/// Quay wall: big blocks, wet and algae-green toward the bottom (the waterline).
pub(crate) fn quay_wall(t: &mut Tex) {
    let w = t.w;
    let (n, n2) = (TNoise::new(4001, w), TNoise::new(4002, w));
    let extra = |u: f32, v: f32, mask: f32, _: &Hit, c: &mut [f32; 5]| {
        let wet0 = (1.0 - smooth(0.0, 0.42, v + n.fbm(u, v, 0.3, 3) * 0.06))
            .max(1.0 - smooth(0.0, 0.03, 1.0 - v));
        let wet = clamp01(wet0 * (0.75 + 0.25 * n2.fbm(u, v, 0.1, 3)));
        let algae = smooth(0.4, 0.85, wet) * (0.6 + 0.4 * n.fbm(u, v, 0.05, 3));
        let dk = 1.0 - wet * 0.45;
        c[0] = mix(c[0] * dk, 0.2, algae * 0.5);
        c[1] = mix(c[1] * dk, 0.24, algae * 0.5);
        c[2] = mix(c[2] * dk, 0.14, algae * 0.5);
        c[3] = mix(c[3], 0.45, wet * 0.6);
        let eff = (1.0 - mask) * smooth(0.3, 0.6, n2.fbm(u, v, 0.2, 3)) * (1.0 - wet) * 0.4;
        c[0] = mix(c[0], 0.72, eff);
        c[1] = mix(c[1], 0.72, eff);
        c[2] = mix(c[2], 0.7, eff);
    };
    stone_blocks(
        t,
        &StoneOpts {
            joint: 0.022,
            depth: 0.025,
            bevel: 0.02,
            ragged: 0.006,
            bulge: 0.02,
            rough3d: 0.025,
            base: rgb(138, 130, 118),
            vary: 0.08,
            hue: 0.03,
            grain: 0.06,
            bedding: 1.5,
            mortar: rgb(110, 106, 98),
            joint_dirt: 0.8,
            soot: 0.4,
            streaks: 0.6,
            rough: 0.85,
            rough_joint: 0.95,
            erosion: 0.4,
            extra: Some(&extra),
            ..stone_base(
                4003,
                random_courses(w, &[0.55, 0.7, 0.6, 0.65, 0.75, 0.75], 0.9, 1.7, 4004),
            )
        },
    );
}
