//! CPU-side textures and terrain materials.
//!
//! GPU upload happens in the renderer on sync. Sampling for terrain is
//! world-XZ tiling (no mesh UVs required).

use crate::color::Color;
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

/// Opaque water-material handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaterMaterialId(pub(crate) u64);

impl fmt::Display for WaterMaterialId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An animated water sheet.
///
/// Waves are analytic, so no texture is needed; depth comes from the mesh's
/// vertex alpha, which the sheet builder fills with
/// `depth_metres / depth_scale_m`.
#[derive(Clone, Debug)]
pub struct WaterMaterialDesc {
    /// Colour of a hand-deep puddle.
    pub shallow: Color,
    /// Colour once the bed is `depth_scale_m` down.
    pub deep: Color,
    /// Depth that vertex alpha 1.0 stands for.
    pub depth_scale_m: f32,
    /// World metres of one wave period.
    pub wave_length_m: f32,
    /// Peak-to-trough steepness of the shading normal, 0 = glassy.
    pub wave_steepness: f32,
    /// Wave travel speed in metres per second.
    pub wave_speed_m_s: f32,
    /// Width of the foam band along the shoreline.
    pub foam_width_m: f32,
    /// Sun glint strength.
    pub glint: f32,
}

impl Default for WaterMaterialDesc {
    fn default() -> Self {
        Self {
            shallow: Color::rgb(108, 168, 178),
            deep: Color::rgb(14, 48, 78),
            depth_scale_m: 9.0,
            wave_length_m: 5.5,
            wave_steepness: 0.35,
            wave_speed_m_s: 0.9,
            foam_width_m: 1.6,
            glint: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WaterMaterial {
    pub desc: WaterMaterialDesc,
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
///
/// Each kind is built from a few named ingredients rather than one noise sum,
/// because ground only reads as ground when the eye can pick out features at
/// several scales at once: blades and pebbles up close, clumps and dirt at a
/// few metres, drifts across the whole tile.
pub fn generate_terrain_albedo(kind: TerrainAlbedo, size: u32, seed: u32) -> (u32, u32, Vec<u8>) {
    let size = size.max(16);
    let n0 = Noise::new(seed ^ kind_seed(kind));
    let n1 = Noise::new(seed.wrapping_mul(0x9E37_79B9) ^ kind_seed(kind));
    let n2 = Noise::new(seed.wrapping_add(0xA5A5_A5A5) ^ kind_seed(kind));
    let n3 = Noise::new(seed.wrapping_mul(0x85EB_CA6B) ^ kind_seed(kind));
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    let inv = 1.0 / size as f32;
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 * inv;
            let v = y as f32 * inv;
            let (r, g, b) = match kind {
                TerrainAlbedo::Grass => {
                    // Sward: fine blades, clumps of growth, and bare earth
                    // showing through where the cover thins.
                    let blades = unit(tileable(&n2, u, v, 26.0));
                    let fibre = unit(tileable(&n3, u * 0.25, v, 42.0));
                    let clump = unit(tileable(&n1, u, v, 5.5));
                    let drift = unit(tileable(&n0, u, v, 1.7));
                    let health = (0.35 * clump + 0.45 * drift + 0.20 * blades).clamp(0.0, 1.0);
                    let dry = smoothstep(0.30, 0.72, unit(tileable(&n0, u, v, 3.1)));
                    let bare = smoothstep(0.62, 0.88, 1.0 - health);

                    let lush = (0.19, 0.42, 0.15);
                    let straw = (0.48, 0.47, 0.23);
                    let earth = (0.30, 0.23, 0.15);
                    let (mut r, mut g, mut b) = mix3(lush, straw, dry * 0.75);
                    let e = mix3((r, g, b), earth, bare);
                    r = e.0;
                    g = e.1;
                    b = e.2;

                    let shade = 0.82 + 0.30 * health + 0.10 * fibre;
                    (r * shade, g * shade, b * shade)
                }
                TerrainAlbedo::Sand => {
                    // Beach: wind ripples across the tile, wet-dark patches,
                    // and a scatter of coarse grains.
                    let ripple = unit((tileable(&n0, u, v, 3.0) * 6.0).sin());
                    let grain = unit(tileable(&n1, u, v, 55.0));
                    let shell = smoothstep(0.86, 0.97, unit(tileable(&n3, u, v, 34.0)));
                    let damp = smoothstep(0.45, 0.85, unit(tileable(&n2, u, v, 2.2)));

                    let dry = (0.78, 0.70, 0.52);
                    let wet = (0.46, 0.40, 0.31);
                    let (r, g, b) = mix3(dry, wet, damp * 0.6);
                    let shade = 0.90 + 0.14 * ripple + 0.10 * grain;
                    (
                        r * shade + shell * 0.18,
                        g * shade + shell * 0.17,
                        b * shade + shell * 0.15,
                    )
                }
                TerrainAlbedo::Rock => {
                    // Cliff: bedding planes, fractures between them, and lichen
                    // in the shelter of the cracks.
                    let strata = unit((tileable(&n0, u * 0.3, v, 2.0) * 4.5).sin());
                    let grit = unit(tileable(&n1, u, v, 30.0));
                    let fracture =
                        smoothstep(0.55, 0.95, 1.0 - tileable(&n2, u, v, 7.0).abs() * 3.0);
                    let lichen = smoothstep(
                        0.70,
                        0.95,
                        unit(tileable(&n3, u, v, 9.0)) * 0.7 + fracture * 0.5,
                    );

                    let pale = (0.47, 0.45, 0.42);
                    let dark = (0.22, 0.21, 0.20);
                    let moss = (0.24, 0.31, 0.19);
                    let (r, g, b) = mix3(pale, dark, 0.35 + 0.45 * strata);
                    let (r, g, b) = mix3((r, g, b), dark, fracture * 0.7);
                    let (r, g, b) = mix3((r, g, b), moss, lichen * 0.45);
                    let shade = 0.92 + 0.14 * grit;
                    (r * shade, g * shade, b * shade)
                }
            };
            let i = ((y * size + x) * 4) as usize;
            rgba[i] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i + 3] = 255;
        }
    }
    (size, size, rgba)
}

/// Noise in `[-1, 1]` remapped to `[0, 1]`.
#[inline]
fn unit(n: f32) -> f32 {
    (n * 0.5 + 0.5).clamp(0.0, 1.0)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn mix3(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    let t = t.clamp(0.0, 1.0);
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
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
