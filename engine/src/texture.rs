//! CPU-side textures and terrain materials.
//!
//! GPU upload happens in the renderer on sync. Terrain albedos are sampled in
//! world XZ; mesh UVs, when present, are soil splat weights (dry, moor).
//!
//! Terrain materials are deliberately separate from [`crate::mesh::SurfaceMaterial`]:
//! use `TerrainMaterialDesc` for streamed heightfields and `SurfaceMaterial` for
//! ordinary meshes, cave meshes, props, and authored material profiles.

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

/// Opaque terrain-material handle.
///
/// A terrain material owns the complete eight-layer albedo set: lush grass,
/// dry grass, moor, mud, tundra, scree, sand, and rock.
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
///
/// All eight texture handles are required because the terrain shader can blend
/// between them continuously. The usual setup is to create each with
/// [`crate::world::World::create_terrain_albedo`] and construct this value with
/// `..TerrainMaterialDesc::default()` for the tuning fields. A missing texture
/// handle is rejected loudly by [`crate::world::World::create_terrain_material`].
#[derive(Clone, Debug)]
pub struct TerrainMaterialDesc {
    /// Lush lowland sward.
    pub grass: TextureId,
    /// Straw / steppe. Vertex UV.x blends this over [`Self::grass`].
    pub grass_dry: TextureId,
    /// Peat and duff. Vertex UV.y blends this over [`Self::grass`].
    pub grass_moor: TextureId,
    /// Peat and duff; vertex UV.y blends this over lush/dry grass.
    /// Wet clods and dark organic soil for banks and basins.
    pub mud: TextureId,
    /// Low mat vegetation and lichen for cold high ground.
    pub tundra: TextureId,
    /// Angular fragments and fines on exposed slopes.
    pub scree: TextureId,
    /// Warm, dry lowland substrate.
    pub sand: TextureId,
    /// Exposed bedrock and cliff faces.
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
    /// Height above sea where shaded, gentle ground can hold snow.
    ///
    /// Set `snow_full_m` at or below this to disable snow (the Engine default).
    pub snow_line_m: f32,
    /// Height where gentle ground is fully snowed, regardless of aspect.
    pub snow_full_m: f32,
    /// Slope (`1 - n.y`) where snow starts to shed / is gone.
    pub snow_slope_start: f32,
    pub snow_slope_end: f32,
}

impl TerrainMaterialDesc {
    /// Build a terrain descriptor from the complete generated albedo set.
    ///
    /// This constructor avoids a fragile eight-field literal. Use the builder
    /// fields on the returned value only for visual tuning; texture handles are
    /// kept explicit so accidentally missing terrain layers fail at setup time.
    pub fn from_albedos(
        grass: TextureId,
        grass_dry: TextureId,
        grass_moor: TextureId,
        mud: TextureId,
        tundra: TextureId,
        scree: TextureId,
        sand: TextureId,
        rock: TextureId,
    ) -> Self {
        Self {
            grass,
            grass_dry,
            grass_moor,
            mud,
            tundra,
            scree,
            sand,
            rock,
            ..Self::default()
        }
    }
}

impl Default for TerrainMaterialDesc {
    fn default() -> Self {
        Self {
            grass: TextureId(0),
            grass_dry: TextureId(0),
            grass_moor: TextureId(0),
            mud: TextureId(0),
            tundra: TextureId(0),
            scree: TextureId(0),
            sand: TextureId(0),
            rock: TextureId(0),
            metres_per_tile: 14.0,
            rock_slope_start: 0.38,
            rock_slope_end: 0.72,
            sand_height_band: 8.0,
            sea_surface_z: 0.0,
            tint_strength: 0.35,
            // Above any terrain the Engine demos build, so they stay snow-free
            // until a game asks for a snow line.
            snow_line_m: 8_000.0,
            snow_full_m: 9_000.0,
            snow_slope_start: 0.32,
            snow_slope_end: 0.68,
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

/// Write RGBA8 pixels out as a PNG (loud failure on a bad path or size).
///
/// The counterpart of [`load_rgba8_png`], for diagnostics that draw a picture of
/// something the renderer cannot show — a plan view of generated geometry, a
/// field plotted as an image — rather than for anything the game does at runtime.
pub fn save_rgba8_png(path: impl AsRef<Path>, w: u32, h: u32, rgba: &[u8]) -> EngineResult<()> {
    let path = path.as_ref();
    let want = (w as usize) * (h as usize) * 4;
    if w == 0 || h == 0 || rgba.len() != want {
        return Err(EngineError::InvalidValue(format!(
            "{}: {w}x{h} needs {want} bytes, got {}",
            path.display(),
            rgba.len()
        )));
    }
    image::save_buffer(path, rgba, w, h, image::ColorType::Rgba8).map_err(|e| {
        EngineError::Io(std::io::Error::other(format!(
            "failed to write {}: {e}",
            path.display()
        )))
    })
}

/// Kind of built-in tileable albedo.
///
/// Generated albedos are deterministic for a `(kind, size, seed)` tuple and are
/// tileable in both axes. They are intentionally colour/albedo only; terrain
/// slope, elevation, moisture, and snow response are controlled by
/// `TerrainMaterialDesc` and the terrain shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainAlbedo {
    Grass,
    /// Straw and earth: the dry flank of a meadow, not a second sand.
    GrassDry,
    /// Dark peat and leaf-mould: banks, wetlands, and the floor of a stand.
    GrassMoor,
    /// Wet clods and organic mud for saturated low ground.
    Mud,
    /// Cold low vegetation, lichen, and exposed frost soil.
    Tundra,
    /// Angular debris and fines on exposed alpine slopes.
    Scree,
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
                    // Sward: anisotropic blades, clumps of growth, and only a
                    // little earth where the cover really thins. The old mix
                    // was olive mud with straw noise — readable as "not sand",
                    // not as grass.
                    let blades = unit(tileable(&n2, u, v, 28.0));
                    let fibre = unit(tileable(&n3, u * 0.18, v, 52.0));
                    let clump = unit(tileable(&n1, u, v, 5.5));
                    let drift = unit(tileable(&n0, u, v, 1.7));
                    let health = (0.32 * clump + 0.40 * drift + 0.28 * blades).clamp(0.0, 1.0);
                    let dry = smoothstep(0.42, 0.82, unit(tileable(&n0, u, v, 3.1)));
                    let bare = smoothstep(0.74, 0.96, 1.0 - health);

                    let lush = (0.20, 0.46, 0.12);
                    let bright = (0.32, 0.54, 0.16);
                    let straw = (0.50, 0.48, 0.20);
                    let earth = (0.28, 0.22, 0.13);
                    let (mut r, mut g, mut b) = mix3(lush, bright, blades * 0.55);
                    let s = mix3((r, g, b), straw, dry * 0.40);
                    r = s.0;
                    g = s.1;
                    b = s.2;
                    let e = mix3((r, g, b), earth, bare * 0.65);
                    r = e.0;
                    g = e.1;
                    b = e.2;

                    let shade = 0.88 + 0.22 * health + 0.12 * fibre;
                    (r * shade, g * shade, b * shade)
                }
                TerrainAlbedo::GrassDry => {
                    // Steppe: the same sward grammar, but straw and earth win.
                    // A dry hill has to read as grass that thirsted, not as sand
                    // that climbed inland.
                    let blades = unit(tileable(&n2, u, v, 24.0));
                    let fibre = unit(tileable(&n3, u * 0.22, v, 40.0));
                    let clump = unit(tileable(&n1, u, v, 4.8));
                    let drift = unit(tileable(&n0, u, v, 1.4));
                    let health = (0.22 * clump + 0.38 * drift + 0.40 * blades).clamp(0.0, 1.0);
                    let straw_w = smoothstep(0.18, 0.62, unit(tileable(&n0, u, v, 2.6)));
                    let bare = smoothstep(0.52, 0.88, 1.0 - health);

                    let straw = (0.58, 0.50, 0.22);
                    let ochre = (0.48, 0.38, 0.16);
                    let earth = (0.34, 0.26, 0.14);
                    let leftover = (0.30, 0.42, 0.14);
                    let (r, g, b) = mix3(leftover, straw, 0.45 + 0.40 * straw_w);
                    let (r, g, b) = mix3((r, g, b), ochre, 0.22 * fibre);
                    let (r, g, b) = mix3((r, g, b), earth, bare * 0.70);
                    let shade = 0.90 + 0.16 * health + 0.08 * drift;
                    (r * shade, g * shade, b * shade)
                }
                TerrainAlbedo::GrassMoor => {
                    // Peat: dark, wet, a little russet. Duff under a wood is
                    // this, not a greener multiply of the meadow.
                    let clump = unit(tileable(&n1, u, v, 4.2));
                    let fibre = unit(tileable(&n3, u * 0.15, v, 36.0));
                    let wet = smoothstep(0.35, 0.82, unit(tileable(&n2, u, v, 2.0)));
                    let moss = smoothstep(0.62, 0.92, unit(tileable(&n0, u, v, 8.5)));

                    let peat = (0.16, 0.14, 0.10);
                    let russet = (0.32, 0.22, 0.12);
                    let sedge = (0.18, 0.26, 0.12);
                    let (r, g, b) = mix3(peat, russet, 0.28 + 0.35 * clump);
                    let (r, g, b) = mix3((r, g, b), sedge, moss * 0.40);
                    let shade = 0.78 + 0.14 * fibre - 0.12 * wet;
                    (r * shade, g * shade, b * shade)
                }
                TerrainAlbedo::Mud => {
                    let clods = unit(tileable(&n0, u, v, 7.0));
                    let wet = unit(tileable(&n2, u, v, 2.4));
                    let grit = unit(tileable(&n3, u, v, 64.0));
                    let cracks = smoothstep(0.62, 0.94, unit(tileable(&n1, u, v, 13.0)));
                    let dark = (0.16, 0.12, 0.08);
                    let wet_soil = (0.25, 0.18, 0.11);
                    let clay = (0.34, 0.25, 0.16);
                    let mut c = mix3(dark, wet_soil, wet * 0.7 + clods * 0.2);
                    c = mix3(c, clay, grit * 0.18);
                    c = mix3(c, dark, cracks * 0.28);
                    let shade = 0.88 + 0.12 * grit;
                    (c.0 * shade, c.1 * shade, c.2 * shade)
                }
                TerrainAlbedo::Tundra => {
                    let mat = unit(tileable(&n0, u, v, 4.5));
                    let lichen = unit(tileable(&n1, u, v, 19.0));
                    let frost = smoothstep(0.58, 0.92, unit(tileable(&n2, u, v, 3.0)));
                    let grit = unit(tileable(&n3, u, v, 72.0));
                    let moss = (0.25, 0.31, 0.18);
                    let heath = (0.39, 0.34, 0.20);
                    let soil = (0.28, 0.25, 0.19);
                    let frost_tone = (0.58, 0.61, 0.57);
                    let mut c = mix3(moss, heath, mat * 0.75);
                    c = mix3(c, soil, (1.0 - lichen) * 0.48);
                    c = mix3(c, frost_tone, frost * 0.20);
                    let shade = 0.90 + 0.12 * grit;
                    (c.0 * shade, c.1 * shade, c.2 * shade)
                }
                TerrainAlbedo::Scree => {
                    let stones = unit(tileable(&n0, u, v, 9.0));
                    let facets = unit(tileable(&n1, u, v, 26.0));
                    let fines = unit(tileable(&n2, u, v, 70.0));
                    let lichen = smoothstep(0.72, 0.96, unit(tileable(&n3, u, v, 11.0)));
                    let base = (0.32, 0.31, 0.29);
                    let pale = (0.50, 0.47, 0.41);
                    let dark = (0.18, 0.18, 0.17);
                    let moss = (0.24, 0.29, 0.18);
                    let mut c = mix3(dark, pale, stones * 0.72 + facets * 0.16);
                    c = mix3(c, base, fines * 0.34);
                    c = mix3(c, moss, lichen * 0.18);
                    (c.0, c.1, c.2)
                }
                TerrainAlbedo::Sand => {
                    // Beach: wind ripples, warm dry grains, cooler damp troughs.
                    let ripple = unit((tileable(&n0, u, v, 3.2) * 6.5).sin());
                    let grain = unit(tileable(&n1, u, v, 58.0));
                    let shell = smoothstep(0.88, 0.98, unit(tileable(&n3, u, v, 34.0)));
                    let damp = smoothstep(0.50, 0.88, unit(tileable(&n2, u, v, 2.2)));

                    let dry = (0.86, 0.74, 0.48);
                    let wet = (0.55, 0.42, 0.28);
                    let (r, g, b) = mix3(dry, wet, damp * 0.45);
                    let shade = 0.92 + 0.16 * ripple + 0.10 * grain;
                    (
                        r * shade + shell * 0.14,
                        g * shade + shell * 0.12,
                        b * shade + shell * 0.08,
                    )
                }
                TerrainAlbedo::Rock => {
                    // Cliff: bedding, fractures, and a warm/cool split so a
                    // range is stone, not a grey slab under the snow.
                    let strata = unit((tileable(&n0, u * 0.3, v, 2.0) * 4.5).sin());
                    let grit = unit(tileable(&n1, u, v, 30.0));
                    let fracture =
                        smoothstep(0.55, 0.95, 1.0 - tileable(&n2, u, v, 7.0).abs() * 3.0);
                    let lichen = smoothstep(
                        0.70,
                        0.95,
                        unit(tileable(&n3, u, v, 9.0)) * 0.7 + fracture * 0.5,
                    );
                    let warmth = unit(tileable(&n0, u, v, 1.0));

                    let cool = (0.46, 0.46, 0.48);
                    let warm = (0.52, 0.44, 0.38);
                    let dark = (0.20, 0.19, 0.18);
                    let moss = (0.22, 0.32, 0.16);
                    let pale = mix3(cool, warm, 0.35 + 0.40 * warmth);
                    let (r, g, b) = mix3(pale, dark, 0.28 + 0.50 * strata);
                    let (r, g, b) = mix3((r, g, b), dark, fracture * 0.7);
                    let (r, g, b) = mix3((r, g, b), moss, lichen * 0.35);
                    let shade = 0.94 + 0.12 * grit;
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
        TerrainAlbedo::GrassDry => 0xD8A11,
        TerrainAlbedo::GrassMoor => 0x3EA7,
        TerrainAlbedo::Mud => 0xBADC0,
        TerrainAlbedo::Tundra => 0x7A2D2,
        TerrainAlbedo::Scree => 0x5C0EE,
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

/// Generate an eight-cell atlas for generated limestone cave meshes.
/// Cells cover dry rock, warm/cool strata, damp rock, calcite, algae, iron
/// stain, and dark talus. The atlas is deterministic for a stable seed.
pub fn generate_cave_albedo(size: u32, seed: u32) -> (u32, u32, Vec<u8>) {
    let tile = size.max(32);
    let width = tile * 4;
    let height = tile * 2;
    let n0 = Noise::new(seed ^ 0xCAFE_51A7);
    let n1 = Noise::new(seed.wrapping_mul(0x9E37_79B9) ^ 0x51A7_3D21);
    let n2 = Noise::new(seed.wrapping_add(0xA5A5_A5A5));
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let kind = (y / tile) * 4 + x / tile;
            let u = (x % tile) as f32 / tile as f32;
            let v = (y % tile) as f32 / tile as f32;
            let grain = 0.5 + 0.5 * tileable(&n0, u, v, 18.0);
            let strata =
                (0.5 + 0.5 * (v * 11.0 + 1.8 * tileable(&n1, u, v, 3.0)).sin()).clamp(0.0, 1.0);
            let vein = smoothstep(0.68, 0.9, 0.5 + 0.5 * tileable(&n2, u, v, 7.0));
            let (mut r, mut g, mut b) = match kind {
                0 => (128.0, 116.0, 98.0),
                1 => (154.0, 132.0, 103.0),
                2 => (101.0, 113.0, 119.0),
                3 => (67.0, 81.0, 78.0),
                4 => (205.0, 198.0, 177.0),
                5 => (62.0, 112.0, 91.0),
                6 => (126.0, 73.0, 48.0),
                _ => (55.0, 50.0, 45.0),
            };
            let value = 0.78 + 0.28 * grain + 0.10 * strata;
            r *= value;
            g *= value;
            b *= value;
            if kind == 4 {
                r += 28.0 * vein;
                g += 26.0 * vein;
                b += 22.0 * vein;
            }
            if kind == 6 {
                r += 22.0 * vein;
                g += 7.0 * vein;
            }
            if kind == 5 {
                g += 20.0 * vein;
            }
            let i = ((y * width + x) * 4) as usize;
            rgba[i..i + 4].copy_from_slice(&[
                r.clamp(0.0, 255.0) as u8,
                g.clamp(0.0, 255.0) as u8,
                b.clamp(0.0, 255.0) as u8,
                255,
            ]);
        }
    }
    (width, height, rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cave_albedo_has_eight_deterministic_material_cells() {
        let (w, h, first) = generate_cave_albedo(32, 7);
        let (w2, h2, second) = generate_cave_albedo(32, 7);
        assert_eq!((w, h), (128, 64));
        assert_eq!((w, h, first.clone()), (w2, h2, second));
        let mut cells = Vec::new();
        for cy in 0..2 {
            for cx in 0..4 {
                let i = (((cy * 32 + 16) * w + cx * 32 + 16) * 4) as usize;
                cells.push([first[i], first[i + 1], first[i + 2]]);
            }
        }
        assert!(cells.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn albedo_size_and_opaque() {
        for kind in [
            TerrainAlbedo::Grass,
            TerrainAlbedo::GrassDry,
            TerrainAlbedo::GrassMoor,
            TerrainAlbedo::Mud,
            TerrainAlbedo::Tundra,
            TerrainAlbedo::Scree,
            TerrainAlbedo::Sand,
            TerrainAlbedo::Rock,
        ] {
            let (w, h, rgba) = generate_terrain_albedo(kind, 64, 1);
            assert_eq!(w, 64);
            assert_eq!(h, 64);
            assert_eq!(rgba.len(), 64 * 64 * 4);
            assert!(
                rgba.chunks(4).all(|c| c[3] == 255),
                "{kind:?} albedo is not opaque"
            );
        }
    }

    #[test]
    fn albedo_is_tileable_edge_close() {
        for kind in [
            TerrainAlbedo::Mud,
            TerrainAlbedo::Tundra,
            TerrainAlbedo::Scree,
        ] {
            let (w, h, a) = generate_terrain_albedo(kind, 32, 7);
            let (_, _, b) = generate_terrain_albedo(kind, 32, 7);
            assert_eq!(a, b, "terrain albedo must be deterministic: {kind:?}");
            assert_eq!(a.len(), (w * h * 4) as usize);
        }
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
            max_d < 48,
            "left/right seam too large (max channel delta {max_d})"
        );
    }
}
