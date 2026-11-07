//! Decals: scorch marks, cracks, frost, impacts, paint and glowing power
//! marks projected onto any surface, terrain and characters included.
//!
//! A decal is a box: everything inside it facing the decal's direction takes
//! on the decal's albedo, normal, roughness and metalness (and glow, for
//! power marks), so a scorch mark is matte and dented while frost is glossy.
//! Decals stuck to a moving body ride along with it; on an animated
//! character they stick to the nearest bone.
//!
//! Decals live in a fixed pool ([`World::set_decal_budget`](crate::world::World::set_decal_budget),
//! 256 by default): a new decal beyond the budget replaces the oldest. Each
//! can fade in and out over its lifetime.
//!
//! ```no_run
//! # use engine::prelude::*;
//! # fn demo(world: &mut World, eye: Vec3, dir: Vec3) -> EngineResult<()> {
//! if let Some(hit) = world.cast_ray(eye, dir, 100.0) {
//!     world.spawn_decal(
//!         Decal::new(DecalKind::Scorch, hit.point, hit.normal, 1.5)
//!             .lifetime(20.0)
//!             .on(hit.entity),
//!     )?;
//! }
//! # Ok(()) }
//! ```

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::world::EntityId;
use glam::Vec3;

/// Most decals the pool can hold.
pub const MAX_DECALS: usize = 512;
/// Built-in kinds plus custom images.
pub const MAX_DECAL_KINDS: usize = 16;
/// Side of each decal image in texels.
pub const DECAL_TEXELS: usize = 256;

/// What a decal looks like.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecalKind {
    /// Soot-black burn with a ragged edge, matte.
    Scorch,
    /// Branching cracks pressed into the surface.
    Crack,
    /// Glossy pale-blue ice crystals.
    Frost,
    /// A bullet or fist hole: a dark pit with a cracked rim.
    Impact,
    /// A paint or blood splat with droplets; tint it.
    Splat,
    /// A glowing ring of energy; tint it and set its glow.
    Power,
    /// An image added with `World::add_decal_image`.
    Custom(u16),
}

impl DecalKind {
    pub const BUILT_IN: [DecalKind; 6] = [
        DecalKind::Scorch,
        DecalKind::Crack,
        DecalKind::Frost,
        DecalKind::Impact,
        DecalKind::Splat,
        DecalKind::Power,
    ];

    /// Index into the decal image array.
    pub(crate) fn layer(self) -> usize {
        match self {
            DecalKind::Scorch => 0,
            DecalKind::Crack => 1,
            DecalKind::Frost => 2,
            DecalKind::Impact => 3,
            DecalKind::Splat => 4,
            DecalKind::Power => 5,
            DecalKind::Custom(n) => Self::BUILT_IN.len() + n as usize,
        }
    }
}

/// A decal handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DecalId(pub u64);

/// One decal to spawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decal {
    pub kind: DecalKind,
    /// A point on the surface.
    pub at: Vec3,
    /// The surface normal there (the decal projects against it).
    pub normal: Vec3,
    /// Width and height on the surface, metres.
    pub size: [f32; 2],
    /// How far the box reaches in front of and behind the surface, metres
    /// (enough to cover bumps and curved bodies).
    pub depth: f32,
    /// Turn about the normal, radians.
    pub angle: f32,
    /// Multiplies the image's colour.
    pub tint: Color,
    pub opacity: f32,
    /// Glow strength (power marks); 0 none.
    pub emission: f32,
    /// Seconds until it is gone (None: until removed or recycled).
    pub lifetime: Option<f32>,
    /// Seconds to fade in, and to fade out at the end of its lifetime.
    pub fade_in: f32,
    pub fade_out: f32,
    /// Stick to this entity (static, moving or animated) instead of the world.
    pub attach: Option<EntityId>,
}

impl Decal {
    /// A square decal `size` metres across at `at`, facing `normal`.
    pub fn new(kind: DecalKind, at: impl Into<Vec3>, normal: impl Into<Vec3>, size: f32) -> Self {
        Self {
            kind,
            at: at.into(),
            normal: normal.into(),
            size: [size, size],
            depth: (size * 0.25).clamp(0.1, 1.0),
            angle: 0.0,
            tint: Color::WHITE,
            opacity: 1.0,
            emission: if kind == DecalKind::Power { 3.0 } else { 0.0 },
            lifetime: None,
            fade_in: 0.0,
            fade_out: 1.0,
            attach: None,
        }
    }

    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.size = [width, height];
        self
    }

    pub fn depth(mut self, metres: f32) -> Self {
        self.depth = metres;
        self
    }

    pub fn angle(mut self, radians: f32) -> Self {
        self.angle = radians;
        self
    }

    pub fn tint(mut self, tint: Color) -> Self {
        self.tint = tint;
        self
    }

    pub fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    pub fn emission(mut self, glow: f32) -> Self {
        self.emission = glow;
        self
    }

    /// Gone after `seconds` (fading out over the last `fade_out`).
    pub fn lifetime(mut self, seconds: f32) -> Self {
        self.lifetime = Some(seconds);
        self
    }

    pub fn fade(mut self, fade_in: f32, fade_out: f32) -> Self {
        self.fade_in = fade_in;
        self.fade_out = fade_out;
        self
    }

    /// Stick to `entity` (for example the one a ray hit); `None` keeps it in
    /// the world.
    pub fn on(mut self, entity: Option<EntityId>) -> Self {
        self.attach = entity;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("decal {what}")));
        if !self.at.is_finite() || !self.normal.is_finite() || self.normal.length_squared() < 1e-8 {
            return bad("position and normal must be finite, the normal non-zero");
        }
        if !self
            .size
            .iter()
            .all(|s| s.is_finite() && (0.01..=100.0).contains(s))
        {
            return bad("size must be within 0.01..=100 metres");
        }
        if !self.depth.is_finite() || !(0.005..=20.0).contains(&self.depth) {
            return bad("depth must be within 0.005..=20 metres");
        }
        if !self.angle.is_finite()
            || !self.opacity.is_finite()
            || !(0.0..=1.0).contains(&self.opacity)
            || !self.emission.is_finite()
            || self.emission < 0.0
        {
            return bad("angle must be finite, opacity within 0..=1, emission >= 0");
        }
        if self.lifetime.is_some_and(|l| !l.is_finite() || l <= 0.0)
            || !self.fade_in.is_finite()
            || !self.fade_out.is_finite()
            || self.fade_in < 0.0
            || self.fade_out < 0.0
        {
            return bad("lifetime must be > 0 and fades >= 0");
        }
        if let DecalKind::Custom(n) = self.kind {
            if Self::custom_limit() <= n as usize {
                return bad("custom kind out of range");
            }
        }
        Ok(())
    }

    pub(crate) fn custom_limit() -> usize {
        MAX_DECAL_KINDS - DecalKind::BUILT_IN.len()
    }

    /// Opacity at `age` seconds: fades in, holds, fades out; None once gone.
    pub(crate) fn opacity_at(&self, age: f32) -> Option<f32> {
        let mut a = self.opacity;
        if self.fade_in > 0.0 {
            a *= (age / self.fade_in).clamp(0.0, 1.0);
        }
        if let Some(life) = self.lifetime {
            if age >= life {
                return None;
            }
            if self.fade_out > 0.0 {
                a *= ((life - age) / self.fade_out).clamp(0.0, 1.0);
            }
        }
        Some(a)
    }
}

/// An image of one decal kind: albedo with alpha (sRGB) and material
/// (normal x, normal y, roughness, metalness; linear), both
/// `DECAL_TEXELS` square, RGBA8.
pub(crate) struct DecalImage {
    pub albedo: Vec<u8>,
    pub material: Vec<u8>,
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

/// Smooth value noise in 0..1.
fn noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let a = hash(xi, yi, seed);
    let b = hash(xi + 1, yi, seed);
    let c = hash(xi, yi + 1, seed);
    let d = hash(xi + 1, yi + 1, seed);
    (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy
}

fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut f = 1.0;
    for o in 0..5 {
        sum += noise(x * f, y * f, seed + o) * amp;
        amp *= 0.5;
        f *= 2.0;
    }
    sum
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Distance from `p` to the nearest of a set of random-walk crack lines.
fn crack_field(cracks: &[Vec<(f32, f32)>], x: f32, y: f32) -> f32 {
    let mut best = f32::MAX;
    for line in cracks {
        for w in line.windows(2) {
            let (a, b) = (w[0], w[1]);
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            let len2 = (dx * dx + dy * dy).max(1e-9);
            let t = (((x - a.0) * dx + (y - a.1) * dy) / len2).clamp(0.0, 1.0);
            let (px, py) = (a.0 + dx * t - x, a.1 + dy * t - y);
            best = best.min((px * px + py * py).sqrt());
        }
    }
    best
}

fn cracks(seed: u32, count: usize, start_r: f32, length: f32) -> Vec<Vec<(f32, f32)>> {
    let mut out = Vec::new();
    for i in 0..count {
        let mut angle =
            i as f32 / count as f32 * std::f32::consts::TAU + hash(i as i32, 7, seed) * 0.6;
        let mut p = (angle.cos() * start_r, angle.sin() * start_r);
        let mut line = vec![p];
        let steps = 14;
        for s in 0..steps {
            angle += (hash(i as i32, s, seed + 1) - 0.5) * 0.9;
            let step = length / steps as f32;
            p = (p.0 + angle.cos() * step, p.1 + angle.sin() * step);
            line.push(p);
            // Branch now and then.
            if s == 5 && hash(i as i32, 99, seed) > 0.4 {
                let mut q = p;
                let mut b = angle
                    + 0.7
                        * if hash(i as i32, 98, seed) > 0.5 {
                            1.0
                        } else {
                            -1.0
                        };
                let mut branch = vec![q];
                for k in 0..6 {
                    b += (hash(i as i32, 50 + k, seed) - 0.5) * 0.8;
                    q = (q.0 + b.cos() * step * 0.8, q.1 + b.sin() * step * 0.8);
                    branch.push(q);
                }
                out.push(branch);
            }
        }
        out.push(line);
    }
    out
}

/// The image of a built-in kind.
pub(crate) fn builtin_image(kind: DecalKind) -> DecalImage {
    let n = DECAL_TEXELS;
    let mut height = vec![0.0f32; n * n];
    let mut albedo = vec![0u8; n * n * 4];
    let mut rough = vec![0.9f32; n * n];
    let mut metal = vec![0.0f32; n * n];
    let crack_lines = match kind {
        DecalKind::Crack => cracks(11, 7, 0.02, 0.95),
        DecalKind::Impact => cracks(23, 9, 0.12, 0.6),
        _ => Vec::new(),
    };
    for y in 0..n {
        for x in 0..n {
            let i = y * n + x;
            let (u, v) = (
                (x as f32 + 0.5) / n as f32 * 2.0 - 1.0,
                (y as f32 + 0.5) / n as f32 * 2.0 - 1.0,
            );
            let r = (u * u + v * v).sqrt();
            let ang = v.atan2(u);
            let (rgb, a, h, ro, me): ([f32; 3], f32, f32, f32, f32) = match kind {
                DecalKind::Scorch => {
                    let edge = 0.62
                        + 0.28 * (fbm(ang.cos() * 2.0 + 3.0, ang.sin() * 2.0 + 3.0, 1) - 0.5) * 2.0;
                    let a = 1.0 - smoothstep(edge - 0.35, edge, r);
                    let soot = fbm(u * 6.0, v * 6.0, 2);
                    let core = 1.0 - smoothstep(0.0, edge * 0.8, r);
                    let c = 0.02 + 0.06 * soot + 0.05 * (1.0 - core);
                    (
                        [c * 1.1, c * 0.85, c * 0.7],
                        a * (0.75 + 0.25 * soot),
                        -soot * 0.3 * a,
                        0.97,
                        0.0,
                    )
                }
                DecalKind::Crack => {
                    let d = crack_field(&crack_lines, u, v);
                    let w = 0.012 + 0.01 * fbm(u * 8.0, v * 8.0, 3);
                    let line = 1.0 - smoothstep(w * 0.4, w * 2.2, d);
                    let fade = 1.0 - smoothstep(0.75, 1.0, r);
                    ([0.05, 0.045, 0.04], line * fade, -line * fade, 0.95, 0.0)
                }
                DecalKind::Frost => {
                    let f = fbm(u * 5.0 + 9.0, v * 5.0, 4);
                    let crystals = (fbm(u * 22.0, v * 22.0, 5) - 0.45).max(0.0) * 3.0;
                    let edge = 0.75 + 0.2 * (fbm(ang.cos() * 3.0, ang.sin() * 3.0, 6) - 0.5);
                    let a = (1.0 - smoothstep(edge - 0.3, edge, r)) * (0.55 + 0.45 * f);
                    let c = 0.75 + 0.25 * crystals.min(1.0);
                    (
                        [c * 0.85, c * 0.93, c],
                        a,
                        crystals.min(1.0) * 0.5,
                        0.18,
                        0.0,
                    )
                }
                DecalKind::Impact => {
                    let pit = 1.0 - smoothstep(0.08, 0.2, r);
                    let rim = (1.0 - ((r - 0.22) / 0.08).abs()).max(0.0);
                    let d = crack_field(&crack_lines, u, v);
                    let line =
                        (1.0 - smoothstep(0.004, 0.02, d)) * (1.0 - smoothstep(0.5, 0.75, r));
                    let dust = (1.0 - smoothstep(0.15, 0.55, r)) * 0.5;
                    let a = (pit + rim * 0.6 + line + dust).min(1.0);
                    let c = 0.03 + 0.25 * (1.0 - pit) * (1.0 - line);
                    (
                        [c, c * 0.95, c * 0.9],
                        a,
                        -pit * 1.6 + rim * 0.4 - line * 0.6,
                        0.9,
                        0.0,
                    )
                }
                DecalKind::Splat => {
                    let wob = fbm(ang.cos() * 2.5 + 1.0, ang.sin() * 2.5 + 1.0, 7);
                    let edge = 0.45 + 0.3 * wob;
                    let blob = 1.0 - smoothstep(edge - 0.04, edge, r);
                    // Droplets flung around.
                    let mut drops: f32 = 0.0;
                    for k in 0..14 {
                        let da = hash(k, 1, 8) * std::f32::consts::TAU;
                        let dr = 0.55 + 0.4 * hash(k, 2, 8);
                        let size = 0.02 + 0.05 * hash(k, 3, 8);
                        let (cx, cy) = (da.cos() * dr, da.sin() * dr);
                        let dd = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt();
                        drops = drops.max(1.0 - smoothstep(size * 0.7, size, dd));
                    }
                    let a = blob.max(drops);
                    ([0.92, 0.92, 0.92], a, a * 0.25, 0.25, 0.0)
                }
                DecalKind::Power => {
                    let ring = (1.0 - ((r - 0.62) / 0.07).abs()).max(0.0);
                    let inner = (1.0 - ((r - 0.42) / 0.025).abs()).max(0.0);
                    let rays = ((ang * 6.0).cos().abs().powf(12.0))
                        * (1.0 - smoothstep(0.3, 0.6, r))
                        * smoothstep(0.08, 0.2, r);
                    let glow = (1.0 - smoothstep(0.0, 0.9, r)) * 0.25;
                    let a = (ring + inner + rays + glow).min(1.0);
                    let c = (ring + inner + rays).min(1.0);
                    (
                        [0.3 + 0.7 * c, 0.3 + 0.7 * c, 0.3 + 0.7 * c],
                        a,
                        0.0,
                        0.4,
                        0.0,
                    )
                }
                DecalKind::Custom(_) => ([1.0, 1.0, 1.0], 0.0, 0.0, 0.9, 0.0),
            };
            height[i] = h;
            rough[i] = ro;
            metal[i] = me;
            let srgb = |c: f32| (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
            albedo[i * 4] = srgb(rgb[0]);
            albedo[i * 4 + 1] = srgb(rgb[1]);
            albedo[i * 4 + 2] = srgb(rgb[2]);
            albedo[i * 4 + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    DecalImage {
        material: material_from_height(&height, &rough, &metal, 6.0),
        albedo,
    }
}

/// Normals (in decal x/y, y up the image) from a height field, packed with
/// roughness and metalness.
fn material_from_height(height: &[f32], rough: &[f32], metal: &[f32], strength: f32) -> Vec<u8> {
    let n = DECAL_TEXELS;
    let at = |x: i32, y: i32| {
        height[(y.clamp(0, n as i32 - 1) as usize) * n + x.clamp(0, n as i32 - 1) as usize]
    };
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n as i32 {
        for x in 0..n as i32 {
            let dx = (at(x + 1, y) - at(x - 1, y)) * strength;
            // Image rows go down, decal y goes up.
            let dy = (at(x, y - 1) - at(x, y + 1)) * strength;
            let nrm = Vec3::new(-dx, -dy, 1.0).normalize();
            let i = (y as usize * n + x as usize) * 4;
            out[i] = ((nrm.x * 0.5 + 0.5) * 255.0).round() as u8;
            out[i + 1] = ((nrm.y * 0.5 + 0.5) * 255.0).round() as u8;
            out[i + 2] = (rough[i / 4].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[i + 3] = (metal[i / 4].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    out
}

/// A user image (RGBA8, sRGB, any size) resampled to a decal image, flat
/// with the given roughness.
pub(crate) fn custom_image(rgba: &[u8], width: usize, height: usize, roughness: f32) -> DecalImage {
    let n = DECAL_TEXELS;
    let mut albedo = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let sx = ((x as f32 + 0.5) / n as f32 * width as f32) as usize;
            let sy = ((y as f32 + 0.5) / n as f32 * height as f32) as usize;
            let s = (sy.min(height - 1) * width + sx.min(width - 1)) * 4;
            albedo[(y * n + x) * 4..(y * n + x) * 4 + 4].copy_from_slice(&rgba[s..s + 4]);
        }
    }
    let flat = vec![0.0; n * n];
    let rough = vec![roughness; n * n];
    DecalImage {
        material: material_from_height(&flat, &rough, &flat, 1.0),
        albedo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_images_have_a_shape_and_clear_corners() {
        for kind in DecalKind::BUILT_IN {
            let img = builtin_image(kind);
            let n = DECAL_TEXELS;
            let alpha = |x: usize, y: usize| img.albedo[(y * n + x) * 4 + 3];
            assert!(
                alpha(n / 2, n / 2) > 0 || kind == DecalKind::Power || kind == DecalKind::Crack,
                "{kind:?} centre"
            );
            assert!(alpha(0, 0) < 8, "{kind:?} corner should be clear");
            let covered = (0..n * n).filter(|i| img.albedo[i * 4 + 3] > 64).count();
            assert!(covered > n * n / 50, "{kind:?} covers {covered}");
        }
    }

    #[test]
    fn decals_fade_and_expire() {
        let d = Decal::new(DecalKind::Scorch, Vec3::ZERO, Vec3::Y, 1.0)
            .lifetime(10.0)
            .fade(1.0, 2.0);
        assert_eq!(d.opacity_at(0.0), Some(0.0));
        assert_eq!(d.opacity_at(5.0), Some(1.0));
        assert!((d.opacity_at(9.0).unwrap() - 0.5).abs() < 1e-5);
        assert_eq!(d.opacity_at(10.0), None);
        assert!(Decal::new(DecalKind::Custom(99), Vec3::ZERO, Vec3::Y, 1.0)
            .validate()
            .is_err());
        assert!(Decal::new(DecalKind::Scorch, Vec3::ZERO, Vec3::ZERO, 1.0)
            .validate()
            .is_err());
    }
}
