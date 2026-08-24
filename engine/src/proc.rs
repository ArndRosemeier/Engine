//! First-class procedural generation helpers.

use crate::color::Color;
use crate::error::EngineResult;
use crate::limits::EngineLimits;
use crate::mesh::Mesh;
use crate::volume::Volume;
use glam::Vec3;
use noise::{NoiseFn, Perlin};

/// Deterministic noise helper keyed by `seed`.
#[derive(Clone, Debug)]
pub struct Noise {
    perlin: Perlin,
    seed: u32,
}

impl Noise {
    pub fn new(seed: u32) -> Self {
        Self {
            perlin: Perlin::new(seed),
            seed,
        }
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Smooth value roughly in [-1, 1].
    pub fn sample2(&self, x: f32, y: f32) -> f32 {
        self.perlin.get([x as f64, y as f64]) as f32
    }

    /// Smooth value roughly in [-1, 1].
    pub fn sample3(&self, p: Vec3) -> f32 {
        self.perlin.get([p.x as f64, p.y as f64, p.z as f64]) as f32
    }

    /// Fractal Brownian motion in 2D.
    pub fn fbm2(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut amp = 1.0;
        let mut freq = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..octaves {
            sum += amp * self.sample2(x * freq, y * freq);
            norm += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }

    /// Lattice value in `[0, 1]` for 2D value noise.
    fn lattice2(&self, ix: i32, iy: i32) -> f32 {
        let mut h = self
            .seed
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((ix as u32).wrapping_mul(374_761_393))
            .wrapping_add((iy as u32).wrapping_mul(668_265_263));
        h ^= h >> 16;
        h = h.wrapping_mul(0x7FEB_352D);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846C_A68B);
        h ^= h >> 16;
        (h >> 8) as f32 * (1.0 / 16_777_215.0)
    }

    /// Value noise in `[-1, 1]` plus analytical derivatives in input space.
    ///
    /// This is Quilez `noised()` — value noise, not Perlin. The Elevated
    /// terrain trick needs those derivatives in the same units as the value.
    /// `https://iquilezles.org/articles/morenoise/`
    pub fn value2_grad(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let px = x.floor();
        let py = y.floor();
        let wx = x - px;
        let wy = y - py;
        let ux = wx * wx * wx * (wx * (wx * 6.0 - 15.0) + 10.0);
        let uy = wy * wy * wy * (wy * (wy * 6.0 - 15.0) + 10.0);
        let dux = 30.0 * wx * wx * (wx * (wx - 2.0) + 1.0);
        let duy = 30.0 * wy * wy * (wy * (wy - 2.0) + 1.0);

        let ix = px as i32;
        let iy = py as i32;
        let a = self.lattice2(ix, iy);
        let b = self.lattice2(ix + 1, iy);
        let c = self.lattice2(ix, iy + 1);
        let d = self.lattice2(ix + 1, iy + 1);

        let k0 = a;
        let k1 = b - a;
        let k2 = c - a;
        let k4 = a - b - c + d;
        let n = k0 + k1 * ux + k2 * uy + k4 * ux * uy;
        (
            -1.0 + 2.0 * n,
            2.0 * dux * (k1 + k4 * uy),
            2.0 * duy * (k2 + k4 * ux),
        )
    }

    /// Quilez gradient-scaled fbm: later octaves fade on steep accumulated slope.
    ///
    /// `https://iquilezles.org/articles/morenoise/` — this is the height
    /// function, not a garnish. Returns a sum on the order of `[-2, 2]`.
    pub fn iq_fbm2(&self, x: f32, y: f32, octaves: u32) -> f32 {
        let mut a = 0.0;
        let mut b = 1.0;
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut px = x;
        let mut py = y;
        for _ in 0..octaves {
            let (n, gx, gy) = self.value2_grad(px, py);
            dx += gx;
            dy += gy;
            a += b * n / (1.0 + dx * dx + dy * dy);
            b *= 0.5;
            let nx = 0.8 * px - 0.6 * py;
            let ny = 0.6 * px + 0.8 * py;
            px = nx * 2.0;
            py = ny * 2.0;
        }
        a
    }

    /// Ridged multifractal in 2D: `1 - |n|`, sharpened, remapped to roughly [-1, 1].
    pub fn ridged2(&self, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut amp = 1.0;
        let mut freq = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..octaves {
            let ridge = 1.0 - self.sample2(x * freq, y * freq).abs();
            let ridge = ridge * ridge;
            sum += amp * ridge;
            norm += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        if norm > 0.0 {
            (sum / norm) * 2.0 - 1.0
        } else {
            0.0
        }
    }

    /// Fractal Brownian motion.
    pub fn fbm3(&self, p: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut amp = 1.0;
        let mut freq = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..octaves {
            sum += amp * self.sample3(p * freq);
            norm += amp;
            amp *= gain;
            freq *= lacunarity;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }
}

/// Rules for generating hilly terrain that can contain caves.
#[derive(Clone, Debug)]
pub struct TerrainRules {
    pub seed: u32,
    pub base_height: f32,
    pub hill_height: f32,
    pub hill_scale: f32,
    pub cave_threshold: f32,
    pub cave_scale: f32,
    pub cave_max_y: f32,
    pub solid_color: Vec3,
}

impl Default for TerrainRules {
    fn default() -> Self {
        Self {
            seed: 42,
            base_height: 8.0,
            hill_height: 10.0,
            hill_scale: 0.03,
            cave_threshold: 0.62,
            cave_scale: 0.06,
            cave_max_y: 14.0,
            solid_color: Vec3::new(0.45, 0.62, 0.38),
        }
    }
}

/// Fill `volume` over `[min, max]` with procedural hills and caves.
pub fn terrain(
    volume: &mut Volume,
    min: Vec3,
    max: Vec3,
    rules: &TerrainRules,
    limits: &EngineLimits,
) -> EngineResult<()> {
    let noise = Noise::new(rules.seed);
    let cave_noise = Noise::new(rules.seed ^ 0xA5A5_5A5A);

    volume.paint_fn_limited(min, max, limits, |p| {
        let h = rules.base_height
            + rules.hill_height
                * noise.fbm3(
                    Vec3::new(p.x * rules.hill_scale, 0.0, p.z * rules.hill_scale),
                    4,
                    2.0,
                    0.5,
                );

        let mut d = h - p.y;

        if p.y < rules.cave_max_y && p.y > 1.0 {
            let c = cave_noise.fbm3(p * rules.cave_scale, 3, 2.0, 0.5);
            let cave = (c * 0.5 + 0.5).clamp(0.0, 1.0);
            if cave > rules.cave_threshold {
                d = d.min(-1.0);
            }
        }

        d
    })
}

/// Carve a cylindrical tunnel along X through the volume.
pub fn carve_tunnel_x(volume: &mut Volume, start: Vec3, length: f32, radius: f32) {
    let step = volume.voxel_size();
    let mut t = 0.0;
    while t <= length {
        let center = start + Vec3::new(t, 0.0, 0.0);
        volume.carve_sphere(center, radius);
        t += step * 0.75;
    }
}

/// Scatter deterministic instance positions on a height band using noise.
pub fn scatter_on_xz(
    seed: u32,
    min: Vec3,
    max: Vec3,
    spacing: f32,
    density: f32,
    y: f32,
) -> Vec<Vec3> {
    assert!((0.0..=1.0).contains(&density), "density must be in 0..=1");
    let noise = Noise::new(seed);
    let mut out = Vec::new();
    let mut z = min.z;
    while z <= max.z {
        let mut x = min.x;
        while x <= max.x {
            let n = noise.sample3(Vec3::new(x * 0.17, 19.0, z * 0.17)) * 0.5 + 0.5;
            if n < density {
                let jitter = noise.sample3(Vec3::new(x, 3.0, z));
                out.push(Vec3::new(
                    x + jitter * spacing * 0.3,
                    y,
                    z + jitter * spacing * 0.3,
                ));
            }
            x += spacing;
        }
        z += spacing;
    }
    out
}

/// Simple unit cube mesh factory.
pub fn unit_box(color: Color) -> Mesh {
    Mesh::box_at(Vec3::ZERO, Vec3::ONE, color).expect("unit box")
}

#[cfg(test)]
mod noise_tests {
    use super::Noise;

    #[test]
    fn sample2_and_fbm_are_finite() {
        let n = Noise::new(42);
        let s = n.sample2(1.25, -3.5);
        assert!(s.is_finite());
        assert!((-1.5..=1.5).contains(&s));
        let f = n.fbm2(0.1, 0.2, 4, 2.0, 0.5);
        assert!(f.is_finite());
        let r = n.ridged2(0.1, 0.2, 4, 2.0, 0.5);
        assert!(r.is_finite());
        assert!((-1.5..=1.5).contains(&r));
        let iq = n.iq_fbm2(0.1, 0.2, 6);
        assert!(iq.is_finite());
        assert_eq!(iq, n.iq_fbm2(0.1, 0.2, 6));
    }

    #[test]
    fn value2_grad_matches_finite_differences() {
        let n = Noise::new(42);
        let (v, gx, gy) = n.value2_grad(1.25, -3.5);
        let e = 1e-3;
        let vx = n.value2_grad(1.25 + e, -3.5).0;
        let vy = n.value2_grad(1.25, -3.5 + e).0;
        assert!(
            (gx - (vx - v) / e).abs() < 0.08,
            "gx {gx} vs fd {}",
            (vx - v) / e
        );
        assert!(
            (gy - (vy - v) / e).abs() < 0.08,
            "gy {gy} vs fd {}",
            (vy - v) / e
        );
    }
}
