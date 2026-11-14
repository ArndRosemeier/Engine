use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::place::ensure_finite3;
use glam::{Vec3, Vec4};
use std::collections::HashMap;

/// Opaque index into [`Mesh`] points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointId(u32);

impl PointId {
    pub fn index(self) -> u32 {
        self.0
    }

    /// Test-only constructor for out-of-range ids.
    #[cfg(test)]
    pub(crate) fn peek(id: u32) -> Self {
        Self(id)
    }
}

/// Baked RGBA8 albedo sampled in the fragment shader.
#[derive(Clone, Debug)]
pub struct AlbedoMap {
    pub width: u32,
    pub height: u32,
    pub rgba: MapBytes,
}

/// The texels of a map, shared by its copies: cloning a map (a garment's
/// texture on every character that wears it) shares the bytes instead of
/// copying megabytes, and writing to a shared copy copies it first (so a
/// change never shows through another map). Reads like a `[u8]`; make one
/// from a `Vec<u8>` with `.into()`.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct MapBytes(std::sync::Arc<Vec<u8>>);

impl MapBytes {
    /// The bytes as a `Vec`, copied only if another map shares them.
    pub fn into_vec(self) -> Vec<u8> {
        std::sync::Arc::try_unwrap(self.0).unwrap_or_else(|shared| (*shared).clone())
    }

    /// A fast 64-bit hash of the bytes (several GB/s), to find maps with
    /// the same texels; not for security.
    pub fn content_hash(&self) -> u64 {
        crate::world::content_hash(&self.0)
    }

    /// Does another map hold these bytes too (a clone of this one, or the
    /// one this was cloned from)?
    pub fn is_shared(&self) -> bool {
        std::sync::Arc::strong_count(&self.0) > 1
    }

    /// Are these the very same bytes as `other`'s (one is a clone of the
    /// other), not only equal ones?
    pub fn shares(&self, other: &MapBytes) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl From<Vec<u8>> for MapBytes {
    fn from(v: Vec<u8>) -> Self {
        Self(std::sync::Arc::new(v))
    }
}

impl FromIterator<u8> for MapBytes {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl std::ops::Deref for MapBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl std::ops::DerefMut for MapBytes {
    /// Copies the bytes first if another map shares them.
    fn deref_mut(&mut self) -> &mut [u8] {
        let bytes: &mut Vec<u8> = std::sync::Arc::make_mut(&mut self.0);
        bytes
    }
}

impl std::fmt::Debug for MapBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MapBytes({} bytes)", self.0.len())
    }
}

/// Texture maps that go with an albedo, in glTF's metallic-roughness model.
///
/// Every map is optional and sampled with the same UVs as the albedo. Missing
/// maps fall back to neutral texels (flat normal, full occlusion, the
/// per-vertex roughness and metallic, no emission).
#[derive(Clone, Debug)]
pub struct MaterialMaps {
    /// Tangent-space normal map, linear RGB, +Y up (glTF convention).
    pub normal: Option<AlbedoMap>,
    /// Packed occlusion (R), roughness (G) and metallic (B), linear. Roughness
    /// and metallic multiply the per-vertex [`SurfaceMaterial`] values.
    pub orm: Option<AlbedoMap>,
    /// Emitted colour, sRGB, multiplied by `emissive_factor`.
    pub emissive: Option<AlbedoMap>,
    pub normal_scale: f32,
    /// UV scale (xy) and offset (zw) for the normal map, from glTF's
    /// `KHR_texture_transform`: tiled detail normals (fabric weave, skin pores).
    pub normal_uv: [f32; 4],
    pub occlusion_strength: f32,
    /// Linear emitted colour; values above 1 glow brighter (emissive strength).
    pub emissive_factor: [f32; 3],
    /// Roughness and metallic for meshes that carry no per-vertex material
    /// (skinned characters). Static meshes use their [`SurfaceMaterial`].
    pub roughness: f32,
    pub metallic: f32,
    /// Metres per texture tile for triplanar mapping, 0 to use the mesh UVs.
    /// Triplanar maps project the textures along the mesh's own axes (scaled
    /// to world size), so procedural meshes without UVs (rocks, cliffs,
    /// shattered pieces) texture seamlessly and the texture moves with them.
    pub triplanar: f32,
}

impl Default for MaterialMaps {
    fn default() -> Self {
        Self {
            normal: None,
            orm: None,
            emissive: None,
            normal_scale: 1.0,
            normal_uv: [1.0, 1.0, 0.0, 0.0],
            occlusion_strength: 1.0,
            emissive_factor: [0.0; 3],
            roughness: 0.65,
            metallic: 0.0,
            triplanar: 0.0,
        }
    }
}

impl MaterialMaps {
    /// True when nothing here would change how the surface looks.
    pub fn is_neutral(&self) -> bool {
        self.normal.is_none()
            && self.orm.is_none()
            && self.emissive.is_none()
            && self.emissive_factor == [0.0; 3]
            && self.triplanar == 0.0
    }

    pub fn validate(&self) -> EngineResult<()> {
        for (name, map) in [
            ("normal", &self.normal),
            ("orm", &self.orm),
            ("emissive", &self.emissive),
        ] {
            if let Some(m) = map {
                if m.width == 0
                    || m.height == 0
                    || m.rgba.len() != m.width as usize * m.height as usize * 4
                {
                    return Err(EngineError::InvalidMesh(format!(
                        "{name} map is {}x{} with {} bytes",
                        m.width,
                        m.height,
                        m.rgba.len()
                    )));
                }
            }
        }
        let finite = [
            self.normal_scale,
            self.occlusion_strength,
            self.normal_uv[0],
            self.normal_uv[1],
            self.normal_uv[2],
            self.normal_uv[3],
            self.roughness,
            self.metallic,
            self.triplanar,
        ]
        .into_iter()
        .chain(self.emissive_factor)
        .all(f32::is_finite);
        if !finite {
            return Err(EngineError::InvalidMesh(
                "material map factors must be finite".into(),
            ));
        }
        Ok(())
    }
}

/// Compact material controls carried by generated mesh vertices.
///
/// The shared renderer uses these values for every surface type: authored
/// props, terrain-like meshes, and procedural cave geometry all get the same
/// roughness/metallic/emission/detail vocabulary without a cave-only shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceMaterial {
    pub roughness: f32,
    pub metallic: f32,
    pub emission: f32,
    pub detail_scale: f32,
    pub strata_strength: f32,
    pub seed: f32,
    pub warp_strength: f32,
    pub noise_gain: f32,
    /// Local material orientation. Directional materials use this as their
    /// primary axis; isotropic materials may ignore it.
    pub orientation: [f32; 3],
    /// Profile discriminator for shader layers; zero means the generic stone path.
    pub profile: f32,
    /// Direction for a deposited overlay such as snow, moss, or dust.
    pub coverage_direction: [f32; 3],
    pub coverage_level: f32,
    pub coverage_softness: f32,
    pub coverage_strength: f32,
    pub overlay_roughness: f32,
    pub overlay_metallic: f32,
    /// Glow channel of the emission (0 = always on, 1 to 15 = switchable):
    /// [`crate::world::World::set_glow_channel`] dims or brightens every
    /// surface on a channel at once, and can give it its own glow colour.
    /// Lit windows that come on at dusk, neon that flickers, a lamp bulb
    /// that goes dark with its lamp.
    pub glow_channel: u8,
}

impl Default for SurfaceMaterial {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl SurfaceMaterial {
    /// Configure a generic directional deposit such as snow, moss, or dust.
    pub const fn with_coverage(
        self,
        direction: [f32; 3],
        level: f32,
        softness: f32,
        strength: f32,
    ) -> Self {
        Self {
            coverage_direction: direction,
            coverage_level: level,
            coverage_softness: softness,
            coverage_strength: strength,
            ..self
        }
    }

    pub const fn with_overlay_response(self, roughness: f32, metallic: f32) -> Self {
        Self {
            overlay_roughness: roughness,
            overlay_metallic: metallic,
            ..self
        }
    }

    /// The same material glowing on a switchable channel (1 to 15; 0 is
    /// always on). See [`crate::world::World::set_glow_channel`].
    pub const fn on_glow_channel(self, channel: u8) -> Self {
        Self {
            glow_channel: channel,
            ..self
        }
    }

    /// Return the same material with a local directional basis axis.
    ///
    /// The vector is normalized by the shader. Use a zero vector only when the
    /// material is intentionally isotropic; the renderer supplies a safe Y-axis
    /// fallback for that case.
    pub const fn with_orientation(self, orientation: [f32; 3]) -> Self {
        Self {
            orientation,
            ..self
        }
    }

    /// Return the same material with a deterministic per-object noise phase.
    pub const fn with_seed(self, seed: f32) -> Self {
        Self { seed, ..self }
    }

    /// Return a deterministic variation without changing the material family.
    pub const fn with_variation(self, seed: f32, detail_scale: f32) -> Self {
        Self {
            seed,
            detail_scale,
            ..self
        }
    }

    pub const DEFAULT: Self = Self {
        roughness: 0.82,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 3.5,
        strata_strength: 0.0,
        seed: 0.0,
        warp_strength: 0.18,
        noise_gain: 0.5,
        orientation: [0.0, 1.0, 0.0],
        profile: 0.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    pub const STONE: Self = Self {
        roughness: 0.88,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 4.0,
        strata_strength: 0.35,
        seed: 17.0,
        warp_strength: 0.28,
        noise_gain: 0.52,
        orientation: [1.0, 0.0, 0.0],
        profile: 0.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    pub const WET_STONE: Self = Self {
        roughness: 0.28,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 5.5,
        strata_strength: 0.28,
        seed: 31.0,
        warp_strength: 0.32,
        noise_gain: 0.48,
        orientation: [0.0, 1.0, 0.0],
        profile: 0.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    pub const CALCITE: Self = Self {
        roughness: 0.42,
        metallic: 0.0,
        emission: 0.08,
        detail_scale: 7.0,
        strata_strength: 0.18,
        seed: 47.0,
        warp_strength: 0.22,
        noise_gain: 0.5,
        orientation: [0.0, 1.0, 0.0],
        profile: 0.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    pub const METAL: Self = Self {
        roughness: 0.2,
        metallic: 0.88,
        emission: 0.0,
        detail_scale: 9.0,
        strata_strength: 0.0,
        seed: 61.0,
        warp_strength: 0.12,
        noise_gain: 0.44,
        orientation: [1.0, 0.0, 0.0],
        profile: 0.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Mottled soil profile with clumps, organic pockets, and fine grit.
    pub const DIRT: Self = Self {
        roughness: 0.96,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 1.3,
        strata_strength: 0.72,
        seed: 131.0,
        warp_strength: 0.31,
        noise_gain: 0.54,
        orientation: [0.0, 1.0, 0.0],
        profile: 3.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Cluster-aware leaf-card profile. The shader breaks crossed cards into
    /// irregular lobes, veins, and small gaps; it is intended for foliage
    /// prototypes rather than terrain or solid tree trunks.
    pub const FOLIAGE: Self = Self {
        roughness: 0.92,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 4.0,
        strata_strength: 0.34,
        seed: 181.0,
        warp_strength: 0.26,
        noise_gain: 0.5,
        orientation: [0.0, 1.0, 0.0],
        profile: 6.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.94,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Needled foliage profile. Needle silhouettes are authored by the mesh;
    /// the shader applies only stable shading and low-frequency variation.
    pub const NEEDLED_FOLIAGE: Self = Self {
        roughness: 0.94,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 5.0,
        strata_strength: 0.18,
        seed: 271.0,
        warp_strength: 0.14,
        noise_gain: 0.42,
        orientation: [0.0, 1.0, 0.0],
        profile: 7.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.3,
        coverage_softness: 0.12,
        coverage_strength: 0.0,
        overlay_roughness: 0.96,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Grass-surface profile. This shades a surface; foliage geometry remains
    /// a separate system so coverage does not become a fragment-only hack.
    pub const GRASS: Self = Self {
        roughness: 0.88,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 2.0,
        strata_strength: 0.66,
        seed: 173.0,
        warp_strength: 0.28,
        noise_gain: 0.52,
        orientation: [0.0, 1.0, 0.0],
        profile: 4.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Snow surface with directional accumulation and cool granular breakup.
    pub const SNOW: Self = Self {
        roughness: 0.88,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 4.8,
        strata_strength: 0.42,
        seed: 211.0,
        warp_strength: 0.27,
        noise_gain: 0.52,
        orientation: [0.0, 1.0, 0.0],
        profile: 5.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.28,
        coverage_softness: 0.22,
        coverage_strength: 1.0,
        overlay_roughness: 0.92,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Layered dune-ripple profile. `orientation` is the surface normal.
    pub const SAND: Self = Self {
        roughness: 0.9,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 1.0,
        strata_strength: 0.58,
        seed: 89.0,
        warp_strength: 0.24,
        noise_gain: 0.5,
        orientation: [0.0, 1.0, 0.0],
        profile: 2.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    /// Directional wood grain profile. `orientation` is the grain axis.
    pub const WOOD: Self = Self {
        roughness: 0.62,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: 2.2,
        strata_strength: 0.72,
        seed: 113.0,
        warp_strength: 0.34,
        noise_gain: 0.56,
        orientation: [1.0, 0.0, 0.0],
        profile: 1.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };

    pub const GLOWING: Self = Self {
        roughness: 0.45,
        metallic: 0.0,
        emission: 0.7,
        detail_scale: 6.0,
        strata_strength: 0.0,
        seed: 79.0,
        warp_strength: 0.25,
        noise_gain: 0.5,
        orientation: [0.0, 1.0, 0.0],
        profile: 0.0,
        coverage_direction: [0.0, 1.0, 0.0],
        coverage_level: 0.35,
        coverage_softness: 0.15,
        coverage_strength: 0.0,
        overlay_roughness: 0.9,
        overlay_metallic: 0.0,
        glow_channel: 0,
    };
}

/// A CPU-side triangle mesh built from human-friendly points and faces.
///
/// Faces may be triangles or quads. Call [`Mesh::build`] (or let the world do it
/// on spawn) to triangulate and compute normals.
///
/// Face winding should be counter-clockwise when looking at the outside.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    points: Vec<Vec3>,
    colors: Vec<Vec4>,
    uvs: Vec<[f32; 2]>,
    faces: Vec<Vec<PointId>>,
    /// Authored per-point normals (zero = not set). Empty when none were set:
    /// such meshes keep flat facets.
    normals: Vec<Vec3>,
    albedo: Option<AlbedoMap>,
    maps: Option<MaterialMaps>,
    surface: SurfaceMaterial,
    surface_overrides: Vec<SurfaceMaterial>,
}

/// A mesh made ready to spawn ([`Mesh::prepare`]): triangulated, with
/// normals, and its maps' sharing key worked out. Making one is the slow
/// part of a spawn (tens of milliseconds for a big textured mesh), and it
/// runs on any thread; [`crate::World::spawn_prepared`] then takes it in at
/// once.
#[derive(Clone, Debug)]
pub struct PreparedMesh {
    pub(crate) built: BuiltMesh,
    pub(crate) albedo: Option<AlbedoMap>,
    pub(crate) maps: Option<MaterialMaps>,
    pub(crate) key: Option<u64>,
}

impl PreparedMesh {
    /// Triangles, as built.
    pub fn triangle_count(&self) -> usize {
        self.built.indices.len() / 3
    }
}

/// Friendly alias — same type as [`Mesh`].
pub type Shape = Mesh;

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    /// Axis-aligned box as a complete mesh.
    pub fn box_at(
        center: impl Into<Vec3>,
        size: impl Into<Vec3>,
        color: Color,
    ) -> EngineResult<Self> {
        let mut mesh = Self::new();
        mesh.add_box(center, size, color)?;
        Ok(mesh)
    }

    /// A smooth sphere (24 × 16 segments) as a complete mesh.
    pub fn ball(center: impl Into<Vec3>, radius: f32, color: Color) -> EngineResult<Self> {
        let center = center.into();
        ensure_finite3(center, "ball center")?;
        if !radius.is_finite() || radius <= 0.0 {
            return Err(EngineError::InvalidMesh("ball radius must be > 0".into()));
        }
        let (around, down) = (24u32, 16u32);
        let mut mesh = Self::new();
        let mut ids = Vec::with_capacity(((around + 1) * (down + 1)) as usize);
        for j in 0..=down {
            let v = j as f32 / down as f32 * std::f32::consts::PI;
            for i in 0..=around {
                let u = i as f32 / around as f32 * std::f32::consts::TAU;
                let n = Vec3::new(v.sin() * u.cos(), v.cos(), v.sin() * u.sin());
                let id = mesh.add_point(center + n * radius)?;
                mesh.set_point_color(id, color)?;
                mesh.set_point_normal(id, n)?;
                ids.push(id);
            }
        }
        let at = |i: u32, j: u32| ids[(j * (around + 1) + i) as usize];
        for j in 0..down {
            for i in 0..around {
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                if j == 0 {
                    mesh.add_face(&[a, c, d])?;
                } else if j == down - 1 {
                    mesh.add_face(&[a, b, c])?;
                } else {
                    mesh.add_face(&[a, b, c, d])?;
                }
            }
        }
        Ok(mesh)
    }

    /// A closed cylinder (wheels, posts, logs, barrels) around `axis` through
    /// `center`, `length` long, with `sides` facets (3 to 128).
    pub fn cylinder(
        center: impl Into<Vec3>,
        axis: impl Into<Vec3>,
        radius: f32,
        length: f32,
        sides: u32,
        color: Color,
    ) -> EngineResult<Self> {
        let (center, axis) = (center.into(), axis.into());
        ensure_finite3(center, "cylinder center")?;
        ensure_finite3(axis, "cylinder axis")?;
        if axis.length_squared() < 1e-12 {
            return Err(EngineError::InvalidMesh(
                "cylinder axis must not be zero".into(),
            ));
        }
        if !(radius.is_finite() && radius > 0.0 && length.is_finite() && length > 0.0) {
            return Err(EngineError::InvalidMesh(
                "cylinder radius and length must be > 0".into(),
            ));
        }
        if !(3..=128).contains(&sides) {
            return Err(EngineError::InvalidMesh(
                "cylinder sides must be 3..=128".into(),
            ));
        }
        let a = axis.normalize();
        let p = a.any_orthonormal_vector();
        let q = a.cross(p);
        let half = a * (length * 0.5);
        let ring = |i: u32| {
            let t = i as f32 / sides as f32 * std::f32::consts::TAU;
            p * t.cos() + q * t.sin()
        };
        let mut mesh = Self::new();
        let point = |mesh: &mut Self, at: Vec3, n: Vec3| -> EngineResult<PointId> {
            let id = mesh.add_point(at)?;
            mesh.set_point_color(id, color)?;
            mesh.set_point_normal(id, n)?;
            Ok(id)
        };
        // Sides: smooth normals.
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for i in 0..=sides {
            let r = ring(i);
            bottom.push(point(&mut mesh, center - half + r * radius, r)?);
            top.push(point(&mut mesh, center + half + r * radius, r)?);
        }
        for i in 0..sides as usize {
            mesh.add_face(&[bottom[i], bottom[i + 1], top[i + 1], top[i]])?;
        }
        // Caps: flat.
        let mut cap_top = Vec::new();
        let mut cap_bottom = Vec::new();
        for i in 0..sides {
            let r = ring(i) * radius;
            cap_top.push(point(&mut mesh, center + half + r, a)?);
            cap_bottom.push(point(&mut mesh, center - half + r, -a)?);
        }
        cap_bottom.reverse();
        for cap in [cap_top, cap_bottom] {
            for i in 1..cap.len() - 1 {
                mesh.add_face(&[cap[0], cap[i], cap[i + 1]])?;
            }
        }
        Ok(mesh)
    }

    /// A doorway, dungeon mouth, or teleport surface: a quad on the XY plane
    /// facing `+Z`. Pair two of these with [`crate::world::World::link`].
    pub fn opening(width: f32, height: f32) -> EngineResult<Self> {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(EngineError::InvalidMesh(
                "opening width and height must be finite and > 0".into(),
            ));
        }
        let mut mesh = Self::new();
        let hx = width * 0.5;
        let hy = height * 0.5;
        let a = mesh.add_point((-hx, -hy, 0.0))?;
        let b = mesh.add_point((hx, -hy, 0.0))?;
        let c = mesh.add_point((hx, hy, 0.0))?;
        let d = mesh.add_point((-hx, hy, 0.0))?;
        for id in [a, b, c, d] {
            mesh.set_point_color(id, Color::rgb(0, 0, 0))?;
        }
        mesh.add_quad(a, b, c, d)?;
        Ok(mesh)
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Add a point in local space.
    pub fn add_point(&mut self, position: impl Into<Vec3>) -> EngineResult<PointId> {
        let position = position.into();
        ensure_finite3(position, "point position")?;
        let id = PointId(self.points.len() as u32);
        self.points.push(position);
        self.colors.push(Color::rgb(191, 191, 191).to_vec4());
        self.uvs.push([0.0, 0.0]);
        self.surface_overrides.push(self.surface);
        if !self.normals.is_empty() {
            self.normals.push(Vec3::ZERO);
        }
        Ok(id)
    }

    /// Set the shading normal of an existing point (`NORMAL` from glTF).
    ///
    /// A mesh with authored normals is drawn smooth and indexed, one GPU
    /// vertex per point, like an imported model; points left without one get
    /// the average of their faces.
    pub fn set_point_normal(&mut self, id: PointId, normal: impl Into<Vec3>) -> EngineResult<()> {
        let normal = normal.into();
        ensure_finite3(normal, "point normal")?;
        let idx = id.0 as usize;
        if idx >= self.points.len() {
            return Err(EngineError::InvalidMesh(format!(
                "point {} out of range (have {})",
                id.0,
                self.points.len()
            )));
        }
        if self.normals.is_empty() {
            self.normals = vec![Vec3::ZERO; self.points.len()];
        }
        self.normals[idx] = normal.normalize_or_zero();
        Ok(())
    }

    /// Smallest box around every point, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let first = *self.points.first()?;
        Some(
            self.points
                .iter()
                .fold((first, first), |(lo, hi), &p| (lo.min(p), hi.max(p))),
        )
    }

    /// Whether this mesh carries authored normals (see [`Self::set_point_normal`]).
    pub fn has_normals(&self) -> bool {
        !self.normals.is_empty()
    }

    /// Set the display color of an existing point (alpha < 1 draws in the transparent pass).
    pub fn set_point_color(&mut self, id: PointId, color: Color) -> EngineResult<()> {
        let idx = id.0 as usize;
        if idx >= self.colors.len() {
            return Err(EngineError::InvalidMesh(format!(
                "point id {} is out of range",
                id.0
            )));
        }
        self.colors[idx] = color.to_vec4();
        Ok(())
    }

    /// Set the albedo UV of an existing point (`TEXCOORD_0` from glTF).
    pub fn set_point_uv(&mut self, id: PointId, uv: [f32; 2]) -> EngineResult<()> {
        let idx = id.0 as usize;
        if idx >= self.uvs.len() {
            return Err(EngineError::InvalidMesh(format!(
                "point id {} is out of range",
                id.0
            )));
        }
        self.uvs[idx] = uv;
        Ok(())
    }

    /// Attach a baked albedo. The renderer samples it with the point UVs.
    pub fn set_albedo_rgba(&mut self, width: u32, height: u32, rgba: Vec<u8>) -> EngineResult<()> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidMesh(
                "albedo size must be non-zero".into(),
            ));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| EngineError::InvalidMesh("albedo dimensions overflow".into()))?;
        if rgba.len() != expected {
            return Err(EngineError::InvalidMesh(format!(
                "albedo rgba len {} != {width}x{height}x4",
                rgba.len()
            )));
        }
        self.albedo = Some(AlbedoMap {
            width,
            height,
            rgba: rgba.into(),
        });
        Ok(())
    }

    /// Baked albedo, if this mesh carries one. Sampled with the point UVs.
    pub fn albedo(&self) -> Option<&AlbedoMap> {
        self.albedo.as_ref()
    }

    /// Attach normal, occlusion-roughness-metallic and emissive maps.
    /// Neutral maps are dropped, so plain meshes keep the cheap path.
    pub fn set_material_maps(&mut self, maps: MaterialMaps) -> EngineResult<()> {
        maps.validate()?;
        self.maps = (!maps.is_neutral()).then_some(maps);
        Ok(())
    }

    /// Texture maps that go with the albedo, if any.
    pub fn material_maps(&self) -> Option<&MaterialMaps> {
        self.maps.as_ref()
    }

    /// Apply deterministic UV projection for the eight-cell cave material atlas.
    /// The selected atlas cell follows authored vertex tint for formations and
    /// uses broad spatial patches for the generated shell.
    pub fn project_cave_uvs(&mut self, metres_per_tile: f32) {
        let scale = metres_per_tile.max(0.25);
        for (i, p) in self.points.iter().enumerate() {
            let c = self.colors[i];
            let warm = c.x > c.z * 1.35 && c.x > c.y * 1.25;
            let green = c.y > c.x * 1.35 && c.y > c.z * 1.25;
            let pale = c.x > 0.72 && c.y > 0.68 && c.z > 0.58;
            let dark = c.x < 0.40 && c.y < 0.40 && c.z < 0.40;
            let cell: u32 = if green {
                5
            } else if dark {
                7
            } else if pale {
                4
            } else if warm {
                6
            } else {
                let patch_x = (p.x / scale).floor() as i32;
                let patch_z = (p.z / scale).floor() as i32;
                ((patch_x.wrapping_mul(73) ^ patch_z.wrapping_mul(151)) as u32) % 4
            };
            let u = (p.x / scale).rem_euclid(1.0) * 0.92 + 0.04;
            let v = (p.z / scale).rem_euclid(1.0) * 0.92 + 0.04;
            self.uvs[i] = [
                (cell % 4) as f32 * 0.25 + u * 0.25,
                (cell / 4) as f32 * 0.5 + v * 0.5,
            ];
        }
    }

    /// Apply one reusable material descriptor to this mesh.
    pub fn set_surface_material(&mut self, material: SurfaceMaterial) {
        self.surface = material;
        self.surface_overrides.fill(material);
    }

    pub fn surface_material(&self) -> SurfaceMaterial {
        self.surface
    }

    #[cfg(test)]
    pub(crate) fn surface_overrides(&self) -> &[SurfaceMaterial] {
        &self.surface_overrides
    }

    #[cfg(test)]
    pub(crate) fn point_colors(&self) -> &[Vec4] {
        &self.colors
    }

    pub(crate) fn set_surface_overrides(&mut self, overrides: Vec<SurfaceMaterial>) {
        assert_eq!(
            overrides.len(),
            self.points.len(),
            "surface override count must match mesh points"
        );
        self.surface_overrides = overrides;
    }

    /// Paint every point one colour.
    ///
    /// Vertex colour still tints a sampled albedo (`color * texture`). Untextured
    /// meshes keep a white 1×1 albedo, so this is the whole look.
    pub fn paint_all(&mut self, color: Color) {
        let c = color.to_vec4();
        for slot in &mut self.colors {
            *slot = c;
        }
    }

    /// Connect 3 or 4 existing points into a face. List them
    /// counter-clockwise as seen from the side the face shows (its front);
    /// the back is not drawn.
    pub fn add_face(&mut self, points: &[PointId]) -> EngineResult<()> {
        if points.len() != 3 && points.len() != 4 {
            return Err(EngineError::InvalidMesh(format!(
                "faces must have 3 or 4 points, got {}",
                points.len()
            )));
        }
        for &p in points {
            if (p.0 as usize) >= self.points.len() {
                return Err(EngineError::InvalidMesh(format!(
                    "face references missing point {}",
                    p.0
                )));
            }
        }
        self.faces.push(points.to_vec());
        Ok(())
    }

    /// A triangle, counter-clockwise seen from its front (see [`Self::add_face`]).
    pub fn add_triangle(&mut self, a: PointId, b: PointId, c: PointId) -> EngineResult<()> {
        self.add_face(&[a, b, c])
    }

    /// A quad, counter-clockwise seen from its front (see [`Self::add_face`]).
    pub fn add_quad(&mut self, a: PointId, b: PointId, c: PointId, d: PointId) -> EngineResult<()> {
        self.add_face(&[a, b, c, d])
    }

    /// Axis-aligned box centered at `center` with full `size` extents.
    pub fn add_box(
        &mut self,
        center: impl Into<Vec3>,
        size: impl Into<Vec3>,
        color: Color,
    ) -> EngineResult<()> {
        let center = center.into();
        let size = size.into();
        ensure_finite3(center, "box center")?;
        ensure_finite3(size, "box size")?;
        if size.x <= 0.0 || size.y <= 0.0 || size.z <= 0.0 {
            return Err(EngineError::InvalidMesh(
                "box size components must be > 0".into(),
            ));
        }
        let half = size * 0.5;
        let corners = [
            center + Vec3::new(-half.x, -half.y, -half.z),
            center + Vec3::new(half.x, -half.y, -half.z),
            center + Vec3::new(half.x, half.y, -half.z),
            center + Vec3::new(-half.x, half.y, -half.z),
            center + Vec3::new(-half.x, -half.y, half.z),
            center + Vec3::new(half.x, -half.y, half.z),
            center + Vec3::new(half.x, half.y, half.z),
            center + Vec3::new(-half.x, half.y, half.z),
        ];
        let base = self.points.len() as u32;
        for c in corners {
            let id = self.add_point(c)?;
            self.set_point_color(id, color)?;
        }
        // Outward-facing quads (CCW from outside): -Z +Z -Y +Y -X +X
        let faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ];
        for face in faces {
            let ids = face.map(|i| PointId(base + i));
            self.add_face(&ids)?;
        }
        Ok(())
    }

    /// Merge every point and face of `other` into `self`, translating by
    /// `translation`. Colors, UVs, normals, per-point surfaces, faces and
    /// albedo carry over; point ids in `other` do not remain valid for the
    /// merged result.
    pub fn append_translated(&mut self, other: &Mesh, translation: Vec3) -> EngineResult<()> {
        let mut ids = Vec::with_capacity(other.points.len());
        for p in &other.points {
            ids.push(self.add_point(*p + translation)?);
        }
        let n = self.points.len() - ids.len();
        self.colors[n..].copy_from_slice(&other.colors);
        self.uvs[n..].copy_from_slice(&other.uvs);
        self.surface_overrides[n..].copy_from_slice(&other.surface_overrides);
        if other.has_normals() {
            if self.normals.is_empty() {
                self.normals = vec![Vec3::ZERO; self.points.len()];
            }
            self.normals[n..].copy_from_slice(&other.normals);
        }
        for face in &other.faces {
            let mapped: Vec<PointId> = face.iter().map(|p| ids[p.0 as usize]).collect();
            self.add_face(&mapped)?;
        }
        if let Some(albedo) = &other.albedo {
            if self.albedo.is_none() {
                self.albedo = Some(albedo.clone());
            }
        }
        Ok(())
    }

    /// Build GPU-ready geometry: triangulate faces with flat per-face normals.
    ///
    /// Opaque faces are packed first; [`BuiltMesh::opaque_index_count`] marks the split
    /// so the renderer can draw transparent triangles in a second pass.
    ///
    /// Meshes with authored normals (imported models) are built smooth and
    /// indexed instead, keeping those normals.
    /// A copy with about `fraction` of the triangles (0 to 1), for a far
    /// LOD level: meshoptimizer's edge collapse keeps the silhouette, and
    /// colours, UVs, normals, materials and texture maps carry over. Small
    /// meshes keep at least a few dozen triangles.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(part: Mesh) -> EngineResult<Vec<LodLevel>> {
    /// Ok(vec![
    ///     LodLevel::new(part.clone(), 30.0),
    ///     LodLevel::new(part.simplified(0.2), 120.0),
    ///     LodLevel::new(part.simplified(0.04), 600.0),
    /// ])
    /// # }
    /// ```
    pub fn simplified(&self, fraction: f32) -> Mesh {
        let built = self.build();
        let tris = built.indices.len() / 3;
        let fraction = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let target = ((tris as f32 * fraction) as usize).max(32);
        if target >= tris || built.positions.is_empty() {
            return self.clone();
        }
        let bytes: &[u8] = bytemuck::cast_slice(&built.positions);
        let Ok(adapter) = meshopt::VertexDataAdapter::new(bytes, 12, 0) else {
            return self.clone();
        };
        // Opaque and see-through triangles are simplified apart, so they
        // stay in their own ranges.
        let split = built.opaque_index_count.min(built.indices.len());
        let mut indices = Vec::with_capacity(target * 3);
        for range in [&built.indices[..split], &built.indices[split..]] {
            if range.is_empty() {
                continue;
            }
            let want = (range.len() as f32 * target as f32 / tris as f32) as usize / 3 * 3;
            let want = want.max(3);
            let mut part = meshopt::simplify(
                range,
                &adapter,
                want,
                0.02,
                meshopt::SimplifyOptions::None,
                None,
            );
            if part.len() > want * 2 {
                part = meshopt::simplify_sloppy(&part, &adapter, want, 0.02, None);
            }
            if part.is_empty() {
                part = range.to_vec();
            }
            indices.extend(part);
        }
        let mut remap = vec![u32::MAX; built.positions.len()];
        let mut out = Mesh {
            albedo: self.albedo.clone(),
            maps: self.maps.clone(),
            surface: self.surface,
            ..Mesh::default()
        };
        let overrides = built.surface_overrides.len() == built.positions.len();
        let mut face = Vec::with_capacity(3);
        for &i in &indices {
            let i = i as usize;
            if remap[i] == u32::MAX {
                remap[i] = out.points.len() as u32;
                out.points.push(built.positions[i]);
                out.colors.push(built.colors[i]);
                out.uvs.push(built.uvs.get(i).copied().unwrap_or([0.0; 2]));
                out.normals.push(built.normals[i]);
                out.surface_overrides.push(if overrides {
                    built.surface_overrides[i]
                } else {
                    built.surface
                });
            }
            face.push(PointId(remap[i]));
            if face.len() == 3 {
                out.faces.push(std::mem::take(&mut face));
            }
        }
        out
    }

    /// Does the slow part of a spawn now, on this thread (build it on a
    /// worker, spawn it with [`crate::World::spawn_prepared`]).
    pub fn prepare(mut self) -> PreparedMesh {
        let built = self.build();
        let (albedo, maps) = (self.albedo.take(), self.maps.take());
        let key = crate::world::texture_key(albedo.as_ref(), maps.as_ref(), false);
        PreparedMesh {
            built,
            albedo,
            maps,
            key,
        }
    }

    pub fn build(&self) -> BuiltMesh {
        if self.has_normals() {
            return self.build_indexed_smooth();
        }
        self.build_flat()
    }
    /// Like [`Self::build`], but averages face normals at shared authoring points
    /// so heightfields / ribbons shade as continuous surfaces instead of facets.
    ///
    /// GPU vertices are shared per authoring point (opaque and translucent
    /// ranges stay split), so a heightfield of N×N samples stays N² vertices
    /// rather than six per quad.
    pub fn build_smooth(&self) -> BuiltMesh {
        self.build_indexed_smooth()
    }

    fn build_flat(&self) -> BuiltMesh {
        let mut opaque_faces = Vec::new();
        let mut xlucent_faces = Vec::new();
        for face in &self.faces {
            let transparent = face.iter().any(|p| self.colors[p.0 as usize].w < 0.999);
            if transparent {
                xlucent_faces.push(face.as_slice());
            } else {
                opaque_faces.push(face.as_slice());
            }
        }

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        let mut surfaces = Vec::new();

        let mut emit = |faces: &[&[PointId]],
                        positions: &mut Vec<Vec3>,
                        normals: &mut Vec<Vec3>,
                        colors: &mut Vec<Vec4>,
                        uvs: &mut Vec<[f32; 2]>,
                        indices: &mut Vec<u32>| {
            for face in faces {
                let tris: [[PointId; 3]; 2] = match face.len() {
                    3 => [[face[0], face[1], face[2]], [face[0], face[0], face[0]]],
                    4 => [[face[0], face[1], face[2]], [face[0], face[2], face[3]]],
                    _ => unreachable!("add_face only allows 3 or 4 points"),
                };
                let tri_count = if face.len() == 3 { 1 } else { 2 };
                for tri in tris.iter().take(tri_count) {
                    let a = self.points[tri[0].0 as usize];
                    let b = self.points[tri[1].0 as usize];
                    let c = self.points[tri[2].0 as usize];
                    let face_n = {
                        let raw = (b - a).cross(c - a);
                        if raw.length_squared() > 0.0 {
                            raw.normalize()
                        } else {
                            Vec3::Y
                        }
                    };
                    let base = positions.len() as u32;
                    positions.push(a);
                    positions.push(b);
                    positions.push(c);
                    normals.extend([face_n, face_n, face_n]);
                    colors.push(self.colors[tri[0].0 as usize]);
                    colors.push(self.colors[tri[1].0 as usize]);
                    colors.push(self.colors[tri[2].0 as usize]);
                    uvs.push(self.uvs[tri[0].0 as usize]);
                    uvs.push(self.uvs[tri[1].0 as usize]);
                    uvs.push(self.uvs[tri[2].0 as usize]);
                    surfaces.extend([
                        self.surface_overrides[tri[0].0 as usize],
                        self.surface_overrides[tri[1].0 as usize],
                        self.surface_overrides[tri[2].0 as usize],
                    ]);
                    indices.extend([base, base + 1, base + 2]);
                }
            }
        };

        emit(
            &opaque_faces,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );
        let opaque_index_count = indices.len();
        emit(
            &xlucent_faces,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );

        BuiltMesh {
            elements: Vec::new(),
            positions,
            normals,
            colors,
            uvs,
            indices,
            opaque_index_count,
            surface: self.surface,
            surface_overrides: surfaces,
        }
    }

    /// Indexed smooth bake: one GPU vertex per authoring point per opacity pass.
    fn build_indexed_smooth(&self) -> BuiltMesh {
        let mut opaque_faces = Vec::new();
        let mut xlucent_faces = Vec::new();
        for face in &self.faces {
            let transparent = face.iter().any(|p| self.colors[p.0 as usize].w < 0.999);
            if transparent {
                xlucent_faces.push(face.as_slice());
            } else {
                opaque_faces.push(face.as_slice());
            }
        }

        let smooth_normals = self.averaged_vertex_normals();
        // Authoring point -> GPU vertex; reset between opacity passes so each
        // pass gets its own vertices (they may differ in alpha blending).
        let mut remap = vec![u32::MAX; self.points.len()];
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        let mut surfaces = Vec::new();

        let mut emit = |faces: &[&[PointId]],
                        remap: &mut [u32],
                        positions: &mut Vec<Vec3>,
                        normals: &mut Vec<Vec3>,
                        colors: &mut Vec<Vec4>,
                        uvs: &mut Vec<[f32; 2]>,
                        indices: &mut Vec<u32>| {
            remap.fill(u32::MAX);
            for face in faces {
                let tris: [[PointId; 3]; 2] = match face.len() {
                    3 => [[face[0], face[1], face[2]], [face[0], face[0], face[0]]],
                    4 => [[face[0], face[1], face[2]], [face[0], face[2], face[3]]],
                    _ => unreachable!("add_face only allows 3 or 4 points"),
                };
                let tri_count = if face.len() == 3 { 1 } else { 2 };
                for tri in tris.iter().take(tri_count) {
                    let a = self.points[tri[0].0 as usize];
                    let b = self.points[tri[1].0 as usize];
                    let c = self.points[tri[2].0 as usize];
                    let face_n = {
                        let raw = (b - a).cross(c - a);
                        if raw.length_squared() > 0.0 {
                            raw.normalize()
                        } else {
                            Vec3::Y
                        }
                    };
                    let mut gpu = [0u32; 3];
                    for k in 0..3 {
                        let pi = tri[k].0 as usize;
                        if remap[pi] == u32::MAX {
                            remap[pi] = positions.len() as u32;
                            positions.push(self.points[pi]);
                            let mut n = match self.normals.get(pi) {
                                Some(&a) if a != Vec3::ZERO => a,
                                _ => smooth_normals[pi],
                            };
                            if n.length_squared() < 1e-10 {
                                n = face_n;
                            }
                            normals.push(n);
                            colors.push(self.colors[pi]);
                            uvs.push(self.uvs[pi]);
                            surfaces.push(self.surface_overrides[pi]);
                        }
                        gpu[k] = remap[pi];
                    }
                    indices.extend(gpu);
                }
            }
        };

        emit(
            &opaque_faces,
            &mut remap,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );
        let opaque_index_count = indices.len();
        emit(
            &xlucent_faces,
            &mut remap,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );

        BuiltMesh {
            elements: Vec::new(),
            positions,
            normals,
            colors,
            uvs,
            indices,
            opaque_index_count,
            surface: self.surface,
            surface_overrides: surfaces,
        }
    }

    fn averaged_vertex_normals(&self) -> Vec<Vec3> {
        let mut accum = vec![Vec3::ZERO; self.points.len()];
        for face in &self.faces {
            let tris: [[PointId; 3]; 2] = match face.len() {
                3 => [[face[0], face[1], face[2]], [face[0], face[0], face[0]]],
                4 => [[face[0], face[1], face[2]], [face[0], face[2], face[3]]],
                _ => continue,
            };
            let tri_count = if face.len() == 3 { 1 } else { 2 };
            for tri in tris.iter().take(tri_count) {
                let a = self.points[tri[0].0 as usize];
                let b = self.points[tri[1].0 as usize];
                let c = self.points[tri[2].0 as usize];
                let raw = (b - a).cross(c - a);
                if raw.length_squared() <= 0.0 {
                    continue;
                }
                // Area-weighted (unnormalized cross).
                accum[tri[0].0 as usize] += raw;
                accum[tri[1].0 as usize] += raw;
                accum[tri[2].0 as usize] += raw;
            }
        }
        accum
            .into_iter()
            .map(|n| {
                if n.length_squared() > 0.0 {
                    n.normalize()
                } else {
                    Vec3::Y
                }
            })
            .collect()
    }
}

/// Triangulated mesh with normals, ready for GPU upload (advanced / internal).
#[derive(Clone, Debug, Default)]
pub struct BuiltMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub colors: Vec<Vec4>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// Indices `[0..opaque_index_count)` are opaque; the rest use alpha blending.
    pub opaque_index_count: usize,
    pub surface: SurfaceMaterial,
    /// Optional per-vertex materials for imported multi-material meshes.
    pub surface_overrides: Vec<SurfaceMaterial>,
    /// Optional per-vertex breakable element ids (see [`crate::building`]):
    /// 0 is none, otherwise the vertex disappears when its element breaks.
    pub elements: Vec<u32>,
}

impl BuiltMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub(crate) fn to_interleaved(&self) -> Vec<Vertex> {
        self.positions
            .iter()
            .zip(self.normals.iter())
            .enumerate()
            .map(|(i, (p, n))| Vertex {
                position: (*p).into(),
                normal: (*n).into(),
                color: self.colors[i].into(),
                uv: self.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                surface: {
                    let m = self
                        .surface_overrides
                        .get(i)
                        .copied()
                        .unwrap_or(self.surface);
                    [m.roughness, m.metallic, m.emission, m.detail_scale]
                },
                surface2: {
                    let m = self
                        .surface_overrides
                        .get(i)
                        .copied()
                        .unwrap_or(self.surface);
                    [m.strata_strength, m.seed, m.warp_strength, m.noise_gain]
                },
                surface3: {
                    let m = self
                        .surface_overrides
                        .get(i)
                        .copied()
                        .unwrap_or(self.surface);
                    [
                        m.orientation[0],
                        m.orientation[1],
                        m.orientation[2],
                        m.profile,
                    ]
                },
                surface4: {
                    let m = self
                        .surface_overrides
                        .get(i)
                        .copied()
                        .unwrap_or(self.surface);
                    [
                        m.coverage_direction[0],
                        m.coverage_direction[1],
                        m.coverage_direction[2],
                        m.coverage_level,
                    ]
                },
                surface5: {
                    let m = self
                        .surface_overrides
                        .get(i)
                        .copied()
                        .unwrap_or(self.surface);
                    [
                        m.coverage_softness,
                        m.coverage_strength,
                        m.overlay_roughness,
                        m.overlay_metallic,
                    ]
                },
                element: {
                    let m = self
                        .surface_overrides
                        .get(i)
                        .copied()
                        .unwrap_or(self.surface);
                    let channel =
                        (m.glow_channel.min(GLOW_CHANNELS as u8 - 1) as u32) << ELEMENT_BITS;
                    (self.elements.get(i).copied().unwrap_or(0) & ELEMENT_MASK) | channel
                },
            })
            .collect()
    }

    pub fn append_translated(&mut self, other: &BuiltMesh, translation: Vec3) {
        // Preserve opaque-then-transparent ordering across both meshes.
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        let mut surfaces = Vec::new();

        let mut push_range = |src: &BuiltMesh,
                              index_start: usize,
                              index_end: usize,
                              translation: Vec3,
                              positions: &mut Vec<Vec3>,
                              normals: &mut Vec<Vec3>,
                              colors: &mut Vec<Vec4>,
                              uvs: &mut Vec<[f32; 2]>,
                              indices: &mut Vec<u32>| {
            let mut remap = HashMap::new();
            for &old in &src.indices[index_start..index_end] {
                let new = *remap.entry(old).or_insert_with(|| {
                    let i = old as usize;
                    let id = positions.len() as u32;
                    positions.push(src.positions[i] + translation);
                    normals.push(src.normals[i]);
                    colors.push(src.colors[i]);
                    uvs.push(src.uvs.get(i).copied().unwrap_or([0.0, 0.0]));
                    surfaces.push(src.surface_overrides.get(i).copied().unwrap_or(src.surface));
                    id
                });
                indices.push(new);
            }
        };

        push_range(
            self,
            0,
            self.opaque_index_count,
            Vec3::ZERO,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );
        push_range(
            other,
            0,
            other.opaque_index_count,
            translation,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );
        let opaque_index_count = indices.len();
        push_range(
            self,
            self.opaque_index_count,
            self.indices.len(),
            Vec3::ZERO,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );
        push_range(
            other,
            other.opaque_index_count,
            other.indices.len(),
            translation,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut uvs,
            &mut indices,
        );

        self.positions = positions;
        self.normals = normals;
        self.colors = colors;
        self.uvs = uvs;
        self.surface_overrides = surfaces;
        self.indices = indices;
        self.opaque_index_count = opaque_index_count;
    }
}

/// Number of glow channels ([`SurfaceMaterial::glow_channel`]).
pub const GLOW_CHANNELS: usize = 16;
/// The vertex's `element` word: breakable element in the low bits, glow
/// channel in the top four.
pub(crate) const ELEMENT_BITS: u32 = 28;
pub(crate) const ELEMENT_MASK: u32 = (1 << ELEMENT_BITS) - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub surface: [f32; 4],
    pub surface2: [f32; 4],
    pub surface3: [f32; 4],
    pub surface4: [f32; 4],
    pub surface5: [f32; 4],
    /// Breakable element (1-based, 0 = none) in the low 28 bits: hidden
    /// once its bit is set. The top four bits are the glow channel.
    pub element: u32,
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x4,
            3 => Float32x2,
            4 => Float32x4,
            5 => Float32x4,
            11 => Float32x4,
            12 => Float32x4,
            13 => Float32x4,
            14 => Uint32,
        ],
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    /// Linear RGBA multiply (white = authored look).
    pub tint: [f32; 4],
    /// The mesh's slot in the renderer's material table (texture arrays),
    /// stamped when the instances are uploaded.
    pub material: u32,
    pub _pad: [u32; 3],
}

impl InstanceRaw {
    pub fn from_matrix(m: glam::Mat4) -> Self {
        Self::from_matrix_tint(m, crate::color::Color::WHITE)
    }

    pub fn from_matrix_tint(m: glam::Mat4, tint: crate::color::Color) -> Self {
        Self {
            model: m.to_cols_array_2d(),
            tint: [tint.r, tint.g, tint.b, tint.a],
            material: 0,
            _pad: [0; 3],
        }
    }

    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4,
            9 => Float32x4,
            10 => Float32x4,
            15 => Uint32,
        ],
    };
}

#[cfg(test)]
mod map_bytes_tests {
    use super::*;

    #[test]
    fn copies_of_a_map_share_bytes_until_one_is_written() {
        let a = AlbedoMap {
            width: 1,
            height: 2,
            rgba: vec![1, 2, 3, 4, 5, 6, 7, 8].into(),
        };
        assert!(!a.rgba.is_shared());
        let mut b = a.clone();
        assert!(b.rgba.shares(&a.rgba), "a clone shares the texels");
        assert!(a.rgba.is_shared() && b.rgba.is_shared());
        b.rgba[0] = 9;
        assert!(!b.rgba.shares(&a.rgba), "writing copies them first");
        assert_eq!(a.rgba[0], 1, "the original is unchanged");
        assert_ne!(a.rgba.content_hash(), b.rgba.content_hash());
        b.rgba[0] = 1;
        assert_eq!(a.rgba.content_hash(), b.rgba.content_hash());
        assert_eq!(&b.rgba[..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(b.rgba.clone().into_vec().len(), 8);
        assert_eq!(format!("{:?}", a.rgba), "MapBytes(8 bytes)");
    }
}
