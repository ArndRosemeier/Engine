//! CPU-side textures and terrain materials.
//!
//! GPU upload happens in the renderer on sync. Sampling for terrain is
//! world-XZ tiling (no mesh UVs required).

use crate::error::{EngineError, EngineResult};
use crate::proc::Noise;
use std::fmt;
use std::path::Path;

/// Opaque texture handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(pub(crate) u64);

impl fmt::Display for TextureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Opaque terrain-material handle (grass / sand / rock blend).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub(crate) u64);

impl fmt::Display for MaterialId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CpuTexture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Description for a world-XZ terrain material.
#[derive(Clone, Debug)]
pub struct TerrainMaterialDesc {
    pub grass: TextureId,
    pub sand: TextureId,
    pub rock: TextureId,
    /// World metres covered by one texture tile.
    pub metres_per_tile: f32,
    /// Slope (`1 - n.y`) where rock blend begins / saturates.
    pub rock_slope_start: f32,
    pub rock_slope_end: f32,
    /// Height band above `sea_surface_z` that fades sand → grass.
    pub sand_height_band: f32,
    pub sea_surface_z: f32,
    /// How strongly vertex color tints the albedo (0 = ignore, 1 = full multiply).
    pub tint_strength: f32,
}

impl Default for TerrainMaterialDesc {
    fn default() -> Self {
        Self {
            grass: TextureId(0),
            sand: TextureId(0),
            rock: TextureId(0),
            metres_per_tile: 14.0,
            rock_slope_start: 0.38,
            rock_slope_end: 0.72,
            sand_height_band: 8.0,
            sea_surface_z: 0.0,
            tint_strength: 0.35,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TerrainMaterial {
    pub desc: TerrainMaterialDesc,
}

/// Decode a PNG file into RGBA8 (loud failure on bad paths / formats).
pub fn load_rgba8_png(path: impl AsRef<Path>) -> EngineResult<(u32, u32, Vec<u8>)> {
    let path = path.as_ref();
    let img = image::open(path).map_err(|e| {
        EngineError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("failed to open texture {}: {e}", path.display()),
        ))
    })?;
    let rgba = img.to_rgba8();
    let w = rgba.width();
    let h = rgba.height();
    if w == 0 || h == 0 {
        return Err(EngineError::InvalidValue(format!(
            "texture {} has zero size",
            path.display()
        )));
    }
    Ok((w, h, rgba.into_raw()))
}

/// Kind of built-in tileable albedo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainAlbedo {
    Grass,
    Sand,
    Rock,
}

/// Generate a tileable RGBA8 albedo (seamless via 4D domain wrap).
pub fn generate_terrain_albedo(kind: TerrainAlbedo, size: u32, seed: u32) -> (u32, u32, Vec<u8>) {
    let size = size.max(16);
    let n0 = Noise::new(seed ^ kind_seed(kind));
    let n1 = Noise::new(seed.wrapping_mul(0x9E37_79B9) ^ kind_seed(kind));
    let n2 = Noise::new(seed.wrapping_add(0xA5A5_A5A5) ^ kind_seed(kind));
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let inv = 1.0 / size as f32;
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 * inv;
            let v = y as f32 * inv;
            let a = tileable(&n0, u, v, 3.0);
            let b = tileable(&n1, u, v, 7.0);
            let c = tileable(&n2, u, v, 13.0);
            let (r, g, bcol) = match kind {
                TerrainAlbedo::Grass => {
                    let t = (0.45 + 0.35 * a + 0.25 * b + 0.15 * c).clamp(0.0, 1.0);
                    let blade = (0.5 + 0.5 * c).clamp(0.0, 1.0);
                    // Macro clumps so tiling reads at walk / screenshot scale.
                    let clump = (0.5 + 0.5 * tileable(&n0, u, v, 1.2)).clamp(0.0, 1.0);
                    let r = 0.12 + 0.18 * t + 0.08 * blade + 0.06 * clump;
                    let g = 0.28 + 0.40 * t + 0.12 * blade + 0.10 * clump;
                    let b = 0.08 + 0.10 * t + 0.04 * clump;
                    (r, g, b)
                }
                TerrainAlbedo::Sand => {
                    let t = (0.45 + 0.3 * a + 0.25 * b + 0.15 * c).clamp(0.0, 1.0);
                    let grain = (0.5 + 0.5 * tileable(&n1, u, v, 18.0)).clamp(0.0, 1.0);
                    let r = 0.55 + 0.28 * t + 0.08 * grain;
                    let g = 0.48 + 0.22 * t + 0.06 * grain;
                    let b = 0.28 + 0.16 * t + 0.04 * grain;
                    (r, g, b)
                }
                TerrainAlbedo::Rock => {
                    let t = (0.40 + 0.35 * a + 0.25 * b + 0.2 * c.abs()).clamp(0.0, 1.0);
                    let crack = (b * c).abs().clamp(0.0, 1.0);
                    let strata = (0.5 + 0.5 * tileable(&n2, u * 0.35, v, 2.5)).clamp(0.0, 1.0);
                    let r = 0.22 + 0.28 * t - 0.08 * crack + 0.08 * strata;
                    let g = 0.21 + 0.22 * t - 0.06 * crack + 0.06 * strata;
                    let b = 0.20 + 0.20 * t - 0.05 * crack + 0.05 * strata;
                    (r, g, b)
                }
            };
            let i = ((y * size + x) * 4) as usize;
            rgba[i] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 2] = (bcol.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 3] = 255;
        }
    }
    (size, size, rgba)
}

fn kind_seed(kind: TerrainAlbedo) -> u32 {
    match kind {
        TerrainAlbedo::Grass => 0x67A55,
        TerrainAlbedo::Sand => 0x5A11D,
        TerrainAlbedo::Rock => 0x20C6,
    }
}

/// Seamless sample in unit square `[0,1)²` at the given frequency.
fn tileable(n: &Noise, u: f32, v: f32, freq: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let x = u * freq;
    let y = v * freq;
    // Map torus → R⁴ then average two 3D-ish projections via sample2 pairs.
    let sx = (x * tau).sin();
    let cx = (x * tau).cos();
    let sy = (y * tau).sin();
    let cy = (y * tau).cos();
    let a = n.sample2(cx * 1.7 + sy * 0.3, sx * 1.7 + cy * 0.3);
    let b = n.sample2(cy * 1.9 + sx * 0.25, sy * 1.9 + cx * 0.25);
    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn albedo_size_and_opaque() {
        let (w, h, rgba) = generate_terrain_albedo(TerrainAlbedo::Grass, 64, 1);
        assert_eq!(w, 64);
        assert_eq!(h, 64);
        assert_eq!(rgba.len(), 64 * 64 * 4);
        assert!(rgba.chunks(4).all(|c| c[3] == 255));
    }

    #[test]
    fn albedo_is_tileable_edge_close() {
        let (w, h, rgba) = generate_terrain_albedo(TerrainAlbedo::Rock, 32, 7);
        let pix = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [rgba[i], rgba[i + 1], rgba[i + 2]]
        };
        // Left/right edges should be near-identical (seamless wrap).
        let mut max_d = 0i32;
        for y in 0..h {
            let a = pix(0, y);
            let b = pix(w - 1, y);
            for i in 0..3 {
                max_d = max_d.max((a[i] as i32 - b[i] as i32).abs());
            }
        }
        assert!(
            max_d < 40,
            "left/right seam too large (max channel delta {max_d})"
        );
    }
}
