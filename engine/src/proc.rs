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
    let step = volume.voxel_size;
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
    }
}
