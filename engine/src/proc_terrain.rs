//! GPU procedural terrain (clipmap) — formula evaluated on the GPU.
//!
//! CPU [`HeightField`] samples the same portable noise for gameplay (feet).
//! Visuals use a displaced clipmap; no CPU mesh bake required.

use crate::color::{rgb, rgba, Color};
use crate::surface::{SurfaceSample, WATER_CLEARANCE};
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
            rings: 5,
            resolution: 256,
            cell_size: 0.25,
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
        self.surface_sample(x, z).walk_height()
    }

    pub fn sample(&self, x: f32, z: f32) -> FieldSample {
        sample_field(&self.rules, x, z)
    }

    pub fn surface_sample(&self, x: f32, z: f32) -> SurfaceSample {
        let s = self.sample(x, z);
        SurfaceSample {
            ground: s.ground,
            water_top: s.water_top,
        }
    }

    /// Height matching the GPU clipmap *triangles* on the finest ring.
    ///
    /// Point-sampling the continuous formula sinks the walker through steep
    /// facets. GPU quads are split as (00,01,11) + (00,11,10); we use the same
    /// planar interp so feet sit on the drawn surface.
    pub fn walk_height_on_clipmap(
        &self,
        x: f32,
        z: f32,
        config: &ClipmapConfig,
        focus: Vec3,
    ) -> f32 {
        let cell = config.cell_size.max(1e-4);
        let res = config.resolution.max(2);
        let extent = cell * res as f32;
        let cx = (focus.x / cell).floor() * cell;
        let cz = (focus.z / cell).floor() * cell;
        let origin_x = cx - extent * 0.5;
        let origin_z = cz - extent * 0.5;

        let fx = ((x - origin_x) / cell).clamp(0.0, (res - 1) as f32 - 1e-4);
        let fz = ((z - origin_z) / cell).clamp(0.0, (res - 1) as f32 - 1e-4);
        let ix = fx.floor();
        let iz = fz.floor();
        let tx = fx - ix;
        let tz = fz - iz;

        let sample_g = |i: f32, j: f32| {
            sample_field(
                &self.rules,
                origin_x + i * cell,
                origin_z + j * cell,
            )
            .ground
        };
        let h00 = sample_g(ix, iz);
        let h10 = sample_g(ix + 1.0, iz);
        let h01 = sample_g(ix, iz + 1.0);
        let h11 = sample_g(ix + 1.0, iz + 1.0);
        // Match clipmap index winding: [i00,i01,i11, i00,i11,i10].
        let ground = if tz >= tx {
            h00 * (1.0 - tz) + h01 * (tz - tx) + h11 * tx
        } else {
            h00 * (1.0 - tx) + h10 * (tx - tz) + h11 * tz
        };

        let water_top = sample_field(&self.rules, x, z).water_top;
        let col = SurfaceSample { ground, water_top };
        col.walk_height()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FieldSample {
    /// Walkable height ([`SurfaceSample::walk_height`]).
    pub height: f32,
    pub ground: f32,
    pub water_top: f32,
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
// Integer hash — stable across CPU/GPU (avoid sin/fract precision drift).

fn hash21(ix: i32, iy: i32, seed: u32) -> f32 {
    let mut n = (ix as u32)
        .wrapping_mul(1597334677)
        .wrapping_add((iy as u32).wrapping_mul(3812015801))
        .wrapping_add(seed.wrapping_mul(2747636419));
    n ^= n >> 16;
    n = n.wrapping_mul(2246822519);
    n ^= n >> 13;
    (n >> 8) as f32 / 16777215.0
}

fn value_noise(p: glam::Vec2, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (glam::Vec2::splat(3.0) - 2.0 * f);
    let ix = i.x as i32;
    let iy = i.y as i32;
    let a = hash21(ix, iy, seed);
    let b = hash21(ix + 1, iy, seed);
    let c = hash21(ix, iy + 1, seed);
    let d = hash21(ix + 1, iy + 1, seed);
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
    // Candidate lake body: assign water_top, then wetness from clearance only.
    let in_basin = basin > 0.25 && near_shore;
    let (ground, water_top, water) = if in_basin {
        let water_top = r.water_level;
        let ground = carved.min(water_top - WATER_CLEARANCE - 1e-3);
        (ground, water_top, true)
    } else {
        (floor, f32::NEG_INFINITY, false)
    };
    let col = SurfaceSample { ground, water_top };
    // Soft apron on dry land approaching a basin (visual only; does not invent wetness).
    let height = if water {
        col.walk_height()
    } else if basin > 0.0 && near_shore {
        let t = (basin / 0.25).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        floor * (1.0 - t) + r.water_level * t
    } else {
        ground
    };

    FieldSample {
        height,
        ground,
        water_top,
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
