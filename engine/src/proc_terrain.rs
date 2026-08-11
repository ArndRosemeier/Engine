//! GPU procedural terrain (clipmap) — formula evaluated on the GPU.
//!
//! CPU [`HeightField`] samples the same portable noise for gameplay (feet).
//! Visuals use a displaced clipmap; no CPU mesh bake required.

use crate::color::{rgb, rgba, Color};
use crate::terrain::TerrainRules;
use glam::Vec3;

/// Clipmap layout for GPU heightfield rendering.
#[derive(Clone, Debug)]
pub struct ClipmapConfig {
    /// Nested rings (1..=6). Ring 0 is finest.
    pub rings: u32,
    /// Vertices along one edge of each ring (e.g. 128 → 127 quads).
    pub resolution: u32,
    /// World size of one finest-ring cell.
    pub cell_size: f32,
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self {
            rings: 4,
            resolution: 128,
            cell_size: 0.5,
        }
    }
}

/// GPU procgen terrain request held by [`crate::world::World`].
#[derive(Clone, Debug)]
pub struct ProcTerrain {
    pub rules: TerrainRules,
    pub config: ClipmapConfig,
    /// Walker’s XZ focus; rings snap around this each frame.
    pub focus: Vec3,
}

impl ProcTerrain {
    pub fn gpu_clipmap(rules: TerrainRules, config: ClipmapConfig) -> Self {
        Self {
            rules,
            config,
            focus: Vec3::ZERO,
        }
    }

    pub fn with_focus(mut self, focus: impl Into<Vec3>) -> Self {
        self.focus = focus.into();
        self
    }
}

/// CPU-side sampler using the same portable noise as the GPU clipmap shader.
#[derive(Clone, Debug)]
pub struct HeightField {
    rules: TerrainRules,
}

impl HeightField {
    pub fn new(rules: TerrainRules) -> Self {
        Self { rules }
    }

    pub fn rules(&self) -> &TerrainRules {
        &self.rules
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z).height
    }

    pub fn sample(&self, x: f32, z: f32) -> FieldSample {
        sample_field(&self.rules, x, z)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FieldSample {
    pub height: f32,
    pub ground: f32,
    pub water: bool,
}

/// Pack rules into a GPU uniform block (must match WGSL `TerrainParams`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainParamsUniform {
    pub seed: u32,
    pub _pad0: u32,
    pub base_height: f32,
    pub hill_height: f32,
    pub hill_scale: f32,
    pub lake_scale: f32,
    pub lake_threshold: f32,
    pub water_level: f32,
    pub grass: [f32; 4],
    pub sand: [f32; 4],
    pub rock: [f32; 4],
    pub water: [f32; 4],
}

impl TerrainParamsUniform {
    pub fn from_rules(r: &TerrainRules) -> Self {
        Self {
            seed: r.seed,
            _pad0: 0,
            base_height: r.base_height,
            hill_height: r.hill_height,
            hill_scale: r.hill_scale,
            lake_scale: r.lake_scale,
            lake_threshold: r.lake_threshold,
            water_level: r.water_level,
            grass: color4(r.grass),
            sand: color4(r.sand),
            rock: color4(r.rock),
            water: color4(r.water),
        }
    }
}

fn color4(c: Color) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

// --- Portable noise (must stay in sync with WGSL in render/clipmap.rs) ---

fn hash21(p: glam::Vec2, seed: u32) -> f32 {
    let mut x = p.x.mul_add(127.1, p.y * 311.7) + seed as f32 * 0.017;
    x = x.sin() * 43758.5453;
    x.fract().abs()
}

fn value_noise(p: glam::Vec2, seed: u32) -> f32 {
    let i = p.floor();
    let f = p.fract();
    let u = f * f * (glam::Vec2::splat(3.0) - 2.0 * f);
    let a = hash21(i, seed);
    let b = hash21(i + glam::Vec2::X, seed);
    let c = hash21(i + glam::Vec2::Y, seed);
    let d = hash21(i + glam::Vec2::ONE, seed);
    // Remap [0,1] → [-1,1]
    let v = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    v * 2.0 - 1.0
}

fn fbm2(mut p: glam::Vec2, seed: u32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amp * value_noise(p, seed);
        norm += amp;
        amp *= gain;
        p *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

pub fn sample_field(r: &TerrainRules, x: f32, z: f32) -> FieldSample {
    let n = fbm2(
        glam::Vec2::new(x * r.hill_scale, z * r.hill_scale),
        r.seed,
        5,
        2.1,
        0.5,
    );
    let h_raw = r.base_height + r.hill_height * n;

    let lake = fbm2(
        glam::Vec2::new(x * r.lake_scale + 17.0, z * r.lake_scale + 9.0),
        r.seed ^ 0xC0FFEE,
        3,
        2.0,
        0.55,
    );
    let lake_t = lake * 0.5 + 0.5;
    let span = (1.0 - r.lake_threshold).max(1e-3);
    let basin = ((lake_t - r.lake_threshold) / span).clamp(0.0, 1.0);

    let floor = h_raw.max(r.water_level);
    let carved = floor - basin * 3.5;
    let near_shore = floor <= r.water_level + 1.5;
    let water = basin > 0.25 && near_shore && carved < r.water_level;
    let ground = if water { carved } else { floor };
    let height = if water {
        r.water_level
    } else if basin > 0.0 && near_shore {
        let t = (basin / 0.25).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        floor * (1.0 - t) + r.water_level * t
    } else {
        floor
    };

    FieldSample {
        height,
        ground,
        water,
    }
}

/// Defaults tuned for GPU clipmap demos.
pub fn demo_terrain_rules() -> TerrainRules {
    TerrainRules {
        seed: 19,
        chunk_cells: 32, // unused by GPU path
        cell_size: 1.0,  // unused by GPU path
        base_height: 7.0,
        hill_height: 12.0,
        hill_scale: 0.016,
        lake_scale: 0.011,
        lake_threshold: 0.58,
        water_level: 5.0,
        grass: rgb(92, 140, 70),
        sand: rgb(194, 178, 128),
        water: rgba(40, 120, 175, 90),
        rock: rgb(120, 118, 112),
    }
}
