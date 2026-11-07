//! Procedural PBR textures, ported from AssetGenerator's texgen package.
//!
//! Every layer tiles seamlessly and is deterministic. A [`TextureLayer`]
//! holds albedo, a tangent-space normal map and occlusion-roughness, ready
//! for a [`Mesh`]: [`TextureLayer::apply_triplanar`] for procedural meshes
//! without UVs (rocks, cliffs, shattered pieces), [`TextureLayer::apply_uv`]
//! for meshes whose UVs are in metres.
//!
//! ```no_run
//! use engine::texgen::{natural_layer, NaturalLayer};
//! # fn demo(mesh: &mut engine::Mesh) -> engine::EngineResult<()> {
//! natural_layer(NaturalLayer::RockFace).apply_triplanar(mesh)?;
//! # Ok(()) }
//! ```
//!
//! Layers are generated once per process and shared ([`natural_layer`]
//! caches them); [`NaturalLayer::generate`] makes one at any size.

// Some helpers serve only the building layers.
#[allow(dead_code)]
mod core;
mod natural;
mod paint;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::EngineResult;
use crate::mesh::{AlbedoMap, MaterialMaps, Mesh};
use core::{finish, Finish, Tex};

/// Default layer size in pixels.
pub const TEX_SIZE: u32 = 512;

/// One generated texture set.
#[derive(Clone, Debug)]
pub struct TextureLayer {
    pub size: u32,
    /// Metres one tile covers.
    pub tile_metres: f32,
    /// sRGB albedo, opaque, rows top-down (image convention).
    pub albedo: Vec<u8>,
    /// Tangent-space normal (rgb, +y up the image, glTF convention) and
    /// relief height (a, 0 deepest).
    pub normal: Vec<u8>,
    /// Occlusion (r) and roughness (g), linear; metallic (b) is 0.
    pub orm: Vec<u8>,
    /// How strongly the normal map is applied by default.
    pub normal_scale: f32,
}

impl TextureLayer {
    fn map(&self, rgba: &[u8]) -> AlbedoMap {
        AlbedoMap {
            width: self.size,
            height: self.size,
            rgba: rgba.to_vec(),
        }
    }

    /// The normal and occlusion-roughness maps, projected triplanar with
    /// one tile per [`Self::tile_metres`].
    pub fn material_maps(&self) -> MaterialMaps {
        MaterialMaps {
            normal: Some(self.map(&self.normal)),
            orm: Some(self.map(&self.orm)),
            normal_scale: self.normal_scale,
            roughness: 1.0,
            triplanar: self.tile_metres,
            ..MaterialMaps::default()
        }
    }

    /// Texture `mesh` by projecting the layer along its axes (no UVs
    /// needed). Vertex colours still tint it; roughness comes from the map
    /// times the mesh's [`crate::mesh::SurfaceMaterial`] roughness.
    pub fn apply_triplanar(&self, mesh: &mut Mesh) -> EngineResult<()> {
        self.apply_scaled(mesh, self.tile_metres)
    }

    /// Like [`Self::apply_triplanar`] with a different tile size (metres).
    pub fn apply_scaled(&self, mesh: &mut Mesh, tile_metres: f32) -> EngineResult<()> {
        mesh.set_albedo_rgba(self.size, self.size, self.albedo.clone())?;
        mesh.set_material_maps(MaterialMaps {
            triplanar: tile_metres.max(1e-3),
            ..self.material_maps()
        })
    }

    /// Texture `mesh` through its own UVs (one UV unit = one tile).
    pub fn apply_uv(&self, mesh: &mut Mesh) -> EngineResult<()> {
        mesh.set_albedo_rgba(self.size, self.size, self.albedo.clone())?;
        mesh.set_material_maps(MaterialMaps {
            triplanar: 0.0,
            ..self.material_maps()
        })
    }
}

type Gen = fn(&mut Tex);

macro_rules! natural_layers {
    ($($(#[$doc:meta])* $name:ident = ($label:literal, $gen:path, $w:expr, $normal:expr, $ao_k:expr, $ao_r:expr)),* $(,)?) => {
        /// Ground of open country.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum NaturalLayer {
            $($(#[$doc])* $name,)*
        }

        impl NaturalLayer {
            pub const ALL: &'static [NaturalLayer] = &[$(NaturalLayer::$name,)*];

            pub fn name(self) -> &'static str {
                match self { $(NaturalLayer::$name => $label,)* }
            }

            /// Metres one tile covers.
            pub fn tile_metres(self) -> f32 {
                match self { $(NaturalLayer::$name => $w,)* }
            }

            fn def(self) -> (Gen, Finish) {
                match self {
                    $(NaturalLayer::$name => ($gen as Gen, Finish { normal: $normal, ao_k: $ao_k, ao_r: $ao_r }),)*
                }
            }
        }
    };
}

natural_layers! {
    /// Long meadow grass with tufts, dry patches and flowers.
    WildGrass = ("wild grass", natural::wild_grass, 3.0, 0.4, 25.0, [0.01, 0.05]),
    /// Soil, fallen leaves, twigs and moss.
    ForestFloor = ("forest floor", natural::forest_floor, 3.0, 0.8, 40.0, [0.01, 0.05]),
    /// Weathered grey rock in facets, fissures and lichen.
    RockFace = ("rock face", natural::rock_face, 4.0, 1.0, 20.0, [0.02, 0.1]),
    /// Warm sandstone in thin banded beds.
    CliffStrata = ("cliff strata", natural::cliff_strata, 6.0, 1.0, 20.0, [0.02, 0.12]),
    /// Angular stones over grit.
    Scree = ("scree", natural::scree, 2.0, 1.0, 50.0, [0.01, 0.04]),
    /// Wind-rippled snow.
    Snow = ("snow", natural::snow, 4.0, 0.6, 10.0, [0.03, 0.12]),
    /// Rippled sand, damp in patches.
    Sand = ("sand", natural::sand, 3.0, 0.7, 20.0, [0.01, 0.05]),
    /// Wet earth with puddles.
    Mud = ("mud", natural::mud, 3.0, 0.8, 30.0, [0.01, 0.05]),
    /// Moor heather over peat.
    Heather = ("heather", natural::heather, 3.0, 0.8, 40.0, [0.01, 0.05]),
    /// Badland clay, cracked and crazed.
    Clay = ("clay", natural::clay, 3.0, 0.8, 20.0, [0.01, 0.05]),
    /// Limestone pavement with mossy joints.
    Limestone = ("limestone", natural::limestone, 4.0, 1.0, 30.0, [0.02, 0.08]),
    /// Short downland turf with chalk showing through.
    ChalkTurf = ("chalk turf", natural::chalk_turf, 2.0, 0.5, 25.0, [0.01, 0.04]),
    /// River gravel with grass in drifts.
    Gravel = ("gravel", natural::gravel, 2.0, 1.0, 50.0, [0.01, 0.04]),
    /// Black lava clinker, rust and ash.
    Basalt = ("basalt", natural::basalt, 3.0, 1.0, 40.0, [0.01, 0.06]),
}

impl NaturalLayer {
    /// Generate the layer at `size` × `size` pixels (at least 16).
    pub fn generate(self, size: u32) -> TextureLayer {
        let (gen, fin) = self.def();
        build(gen, fin, self.tile_metres(), size)
    }
}

/// The layer at [`TEX_SIZE`], generated on first use and shared after.
pub fn natural_layer(layer: NaturalLayer) -> Arc<TextureLayer> {
    static CACHE: OnceLock<Mutex<HashMap<NaturalLayer, Arc<TextureLayer>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(l) = cache.lock().expect("texgen cache").get(&layer) {
        return l.clone();
    }
    let made = Arc::new(layer.generate(TEX_SIZE));
    cache
        .lock()
        .expect("texgen cache")
        .entry(layer)
        .or_insert(made)
        .clone()
}

fn build(gen: Gen, fin: Finish, tile_metres: f32, size: u32) -> TextureLayer {
    let size = size.max(16);
    let s = size as usize;
    let mut t = Tex::new(s, tile_metres);
    gen(&mut t);
    let packed = finish(&t, fin);
    // Flip rows to the image convention (row 0 at the top).
    let mut albedo = vec![0u8; s * s * 4];
    let mut normal = vec![0u8; s * s * 4];
    let mut orm = vec![0u8; s * s * 4];
    for j in 0..s {
        let src = (s - 1 - j) * s;
        for i in 0..s {
            let (a, b) = ((src + i) * 4, (j * s + i) * 4);
            albedo[b..b + 3].copy_from_slice(&packed.albedo[a..a + 3]);
            albedo[b + 3] = 255;
            // Tilt at most 60 degrees: the steep sides of tiny bumps (a
            // flower over grass) otherwise catch the sun as fireflies.
            let mut nx = packed.normal[a] as f32 / 127.5 - 1.0;
            let mut ny = packed.normal[a + 1] as f32 / 127.5 - 1.0;
            let xy = (nx * nx + ny * ny).sqrt();
            const MAX_XY: f32 = 0.866;
            if xy > MAX_XY {
                nx *= MAX_XY / xy;
                ny *= MAX_XY / xy;
            }
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            let q = |c: f32| ((c * 0.5 + 0.5) * 255.0).round() as u8;
            normal[b] = q(nx);
            normal[b + 1] = q(ny);
            normal[b + 2] = q(nz);
            normal[b + 3] = packed.normal[a + 2];
            orm[b] = packed.normal[a + 3];
            orm[b + 1] = packed.albedo[a + 3];
            orm[b + 2] = 0;
            orm[b + 3] = 255;
        }
    }
    TextureLayer {
        size,
        tile_metres,
        albedo,
        normal,
        orm,
        normal_scale: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Albedo difference across the wrap seam and between interior neighbours.
    fn seam(a: &[u8], n: usize) -> (u32, u32) {
        let (mut wrap, mut inner) = (0u32, 0u32);
        let d = |x: u8, y: u8| (x as i32 - y as i32).unsigned_abs();
        for i in 0..n {
            for c in 0..3 {
                wrap += d(a[(i * n + n - 1) * 4 + c], a[(i * n) * 4 + c]);
                wrap += d(a[((n - 1) * n + i) * 4 + c], a[i * 4 + c]);
                inner += d(a[(i * n + n / 2) * 4 + c], a[(i * n + n / 2 - 1) * 4 + c]);
                inner += d(
                    a[((n / 2) * n + i) * 4 + c],
                    a[((n / 2 - 1) * n + i) * 4 + c],
                );
            }
        }
        (wrap, inner)
    }

    #[test]
    fn natural_layers_tile_vary_and_repeat() {
        const N: usize = 64;
        for layer in NaturalLayer::ALL {
            let a = layer.generate(N as u32);
            let b = layer.generate(N as u32);
            assert_eq!(a.albedo.len(), N * N * 4);
            assert_eq!(a.albedo, b.albedo, "{layer:?} deterministic");
            assert_eq!(a.normal, b.normal);
            let (wrap, inner) = seam(&a.albedo, N);
            assert!(
                wrap < inner * 5 / 2 + N as u32,
                "{layer:?} seam {wrap} vs {inner}"
            );
            let g = a.albedo.chunks(4).map(|p| p[1]);
            let (lo, hi) = g.fold((255, 0), |(lo, hi), v| (lo.min(v), hi.max(v)));
            assert!(hi - lo > 10, "{layer:?} has visible variation");
            assert!(a.albedo.chunks(4).all(|p| p[3] == 255), "opaque");
            assert!(layer.tile_metres() > 0.0);
        }
    }

    #[test]
    fn layers_become_triplanar_material_maps() {
        let layer = NaturalLayer::RockFace.generate(32);
        let mut mesh = Mesh::new();
        layer.apply_triplanar(&mut mesh).expect("applies");
        let maps = mesh.material_maps().expect("maps");
        assert_eq!(maps.triplanar, 4.0);
        assert!(maps.normal.is_some() && maps.orm.is_some());
        assert!(mesh.albedo().is_some());
    }
}
