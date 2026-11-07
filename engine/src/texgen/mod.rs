//! Procedural PBR textures, ported from AssetGenerator's texgen package.
//!
//! Five families of seamless, deterministic layers:
//! [`NaturalLayer`] (open country: grass, rock, snow, mud...),
//! [`GroundLayer`] (towns: asphalt, paving, cobbles, lawn...),
//! [`FacadeLayer`] (walls: brick, stone, plaster, concrete, glass, timber...),
//! [`RoofLayer`] (tar, tiles, slate, zinc, shingles...) and
//! [`MedievalLayer`] (rubble, oak, thatch, leaded glass, flagstones...).
//!
//! A [`TextureLayer`] holds albedo, a tangent-space normal map and
//! occlusion-roughness, ready for a [`Mesh`]:
//! [`TextureLayer::apply_triplanar`] for procedural meshes without UVs
//! (rocks, cliffs, shattered pieces), [`TextureLayer::apply_uv`] for meshes
//! whose UVs count tiles.
//!
//! ```no_run
//! use engine::texgen::{layer, FacadeLayer, NaturalLayer};
//! # fn demo(rock: &mut engine::Mesh, wall: &mut engine::Mesh) -> engine::EngineResult<()> {
//! layer(NaturalLayer::RockFace).apply_triplanar(rock)?;
//! layer(FacadeLayer::BrickRed).apply_triplanar(wall)?;
//! # Ok(()) }
//! ```
//!
//! [`layer`] generates a layer once per process (at [`TEX_SIZE`]) and shares
//! it; `generate(size)` on any layer makes one at another size. Walls and
//! roofs use +v as up (up the slope on a roof): triplanar projection keeps
//! that the right way round on a mesh's vertical faces.

#[allow(dead_code)]
mod core;
mod facade;
mod ground;
mod layout;
mod masonry;
mod medieval;
mod natural;
mod paint;
mod roof;

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
    /// needed). Vertex colours still tint it (tintable layers such as
    /// plaster are near white for that); roughness comes from the map
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

macro_rules! layers {
    ($(#[$edoc:meta])* $enum:ident {
        $($(#[$doc:meta])* $name:ident = ($label:literal, $gen:path, $w:expr, $normal:expr, $ao_k:expr, $ao_r:expr)),* $(,)?
    }) => {
        $(#[$edoc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $enum {
            $($(#[$doc])* $name,)*
        }

        impl $enum {
            pub const ALL: &'static [$enum] = &[$($enum::$name,)*];

            pub fn name(self) -> &'static str {
                match self { $($enum::$name => $label,)* }
            }

            /// Metres one tile covers.
            pub fn tile_metres(self) -> f32 {
                match self { $($enum::$name => $w,)* }
            }

            /// Generate the layer at `size` × `size` pixels (at least 16).
            pub fn generate(self, size: u32) -> TextureLayer {
                let (gen, fin): (Gen, Finish) = match self {
                    $($enum::$name => ($gen as Gen, Finish { normal: $normal, ao_k: $ao_k, ao_r: $ao_r }),)*
                };
                build(gen, fin, self.tile_metres(), size)
            }
        }

        impl From<$enum> for Layer {
            fn from(l: $enum) -> Self {
                Layer::$enum(l)
            }
        }
    };
}

layers! {
    /// Ground of open country.
    NaturalLayer {
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
}

layers! {
    /// Town ground: roads, pavements, squares, paths and quays.
    GroundLayer {
        /// Road asphalt with cracks and sealed tar snakes.
        RoadAsphalt = ("road asphalt", ground::road_asphalt, 4.0, 1.0, 60.0, [0.01, 0.04]),
        /// Concrete pavement slabs.
        Paving = ("paving", ground::paving, 3.0, 1.0, 50.0, [0.008, 0.03]),
        /// Granite curb stones.
        Curb = ("curb", ground::curb, 3.0, 1.0, 50.0, [0.008, 0.03]),
        /// Mown lawn with clover.
        Lawn = ("lawn", ground::grass, 3.0, 0.35, 25.0, [0.01, 0.04]),
        /// Pale stone slabs of a square.
        PlazaStone = ("plaza stone", ground::plaza_stone, 4.0, 1.0, 50.0, [0.008, 0.03]),
        /// Granite setts laid in arcs.
        Cobble = ("cobble", ground::cobble, 2.0, 1.0, 40.0, [0.006, 0.025]),
        /// Fine gravel path.
        GravelPath = ("gravel path", ground::gravel_path, 2.0, 0.8, 70.0, [0.005, 0.02]),
        /// Seaside promenade slabs.
        Promenade = ("promenade", ground::promenade, 4.0, 1.0, 50.0, [0.008, 0.03]),
        /// Worn, oily car-park asphalt.
        LightAsphalt = ("light asphalt", ground::light_asphalt, 4.0, 1.0, 60.0, [0.01, 0.04]),
        /// White road marking, worn.
        RoadPaintWhite = ("road paint white", ground::road_paint_white, 2.0, 1.0, 60.0, [0.008, 0.03]),
        /// Yellow road marking, worn.
        RoadPaintYellow = ("road paint yellow", ground::road_paint_yellow, 2.0, 1.0, 60.0, [0.008, 0.03]),
        /// Packed dirt, damp in patches.
        Dirt = ("dirt", ground::dirt, 3.0, 0.8, 50.0, [0.01, 0.04]),
        /// Quay wall blocks, wet and green at the waterline (the bottom).
        QuayWall = ("quay wall", ground::quay_wall, 4.0, 1.0, 25.0, [0.015, 0.06]),
    }
}

layers! {
    /// Building walls. +v is up the wall.
    FacadeLayer {
        /// Red brick, running bond.
        BrickRed = ("brick red", facade::brick_red, 2.25, 1.0, 60.0, [0.008, 0.03]),
        /// Purple-brown brick, Flemish bond.
        BrickBrown = ("brick brown", facade::brick_brown, 2.7, 1.0, 60.0, [0.008, 0.03]),
        /// Yellow London stock brick.
        BrickYellow = ("brick yellow", facade::brick_yellow, 2.25, 1.0, 60.0, [0.008, 0.03]),
        /// Brownstone ashlar.
        Brownstone = ("brownstone", facade::brownstone, 2.4, 1.0, 50.0, [0.01, 0.04]),
        /// Pale limestone ashlar.
        Limestone = ("limestone", facade::limestone, 3.0, 1.0, 50.0, [0.01, 0.04]),
        /// Eroded sandstone ashlar.
        Sandstone = ("sandstone", facade::sandstone, 2.4, 1.0, 50.0, [0.01, 0.04]),
        /// Trowelled plaster (near white: tint it with vertex colour).
        Plaster = ("plaster", facade::plaster, 3.0, 1.2, 80.0, [0.01, 0.05]),
        /// Knock-down stucco (near white: tint it).
        Stucco = ("stucco", facade::stucco, 2.0, 1.0, 120.0, [0.006, 0.02]),
        /// Board-formed concrete with tie holes.
        Concrete = ("concrete", facade::concrete, 2.4, 1.0, 50.0, [0.008, 0.03]),
        /// Precast exposed-aggregate panels.
        ConcretePanel = ("concrete panel", facade::concrete_panel, 3.0, 1.0, 40.0, [0.01, 0.04]),
        /// Glass curtain wall with mullions.
        GlassCurtain = ("glass curtain", facade::glass_curtain, 3.9, 0.6, 20.0, [0.02, 0.06]),
        /// Box-rib metal cladding.
        MetalPanel = ("metal panel", facade::metal_panel, 2.0, 1.0, 25.0, [0.01, 0.04]),
        /// Half-timbering with white infill.
        Timber = ("timber", facade::timber, 2.4, 1.0, 50.0, [0.01, 0.04]),
        /// Polished granite slabs.
        Granite = ("granite", facade::granite, 2.4, 1.0, 50.0, [0.006, 0.02]),
        /// Painted clapboard siding (near white: tint it).
        WoodSiding = ("wood siding", facade::wood_siding, 2.4, 1.0, 60.0, [0.01, 0.03]),
        /// White paint flaking off brick.
        BrickWhite = ("brick white", facade::brick_white, 2.25, 1.0, 60.0, [0.008, 0.03]),
    }
}

layers! {
    /// Roofs. +v is up the slope: with triplanar mapping, model a roof
    /// plane so its own +z runs up the slope (or use [`TextureLayer::apply_uv`]).
    RoofLayer {
        /// Flat bitumen membrane.
        Tar = ("tar", roof::tar, 4.0, 1.0, 40.0, [0.01, 0.05]),
        /// Terracotta pantiles.
        ClayTile = ("clay tile", roof::clay_tile, 2.1, 1.0, 40.0, [0.03, 0.1]),
        /// Natural slate.
        Slate = ("slate", roof::slate, 2.0, 1.0, 60.0, [0.008, 0.03]),
        /// Standing-seam zinc.
        Zinc = ("zinc", roof::zinc, 3.0, 1.0, 30.0, [0.01, 0.04]),
        /// Three-tab asphalt shingles.
        AsphaltShingle = ("asphalt shingle", roof::asphalt_shingle, 2.0, 1.0, 80.0, [0.006, 0.02]),
        /// Corrugated galvanised steel with rust.
        Corrugated = ("corrugated", roof::corrugated, 2.0, 1.0, 20.0, [0.01, 0.04]),
        /// Pebble ballast on a flat roof.
        GravelRoof = ("gravel roof", roof::gravel_roof, 2.0, 1.0, 60.0, [0.006, 0.02]),
        /// Sedum green roof.
        GreenRoof = ("green roof", roof::green_roof, 2.0, 0.7, 50.0, [0.008, 0.03]),
    }
}

layers! {
    /// Medieval towns and cellars.
    MedievalLayer {
        /// Rubble walling in rough courses.
        Rubble = ("rubble", medieval::rubble, 2.4, 1.0, 40.0, [0.01, 0.05]),
        /// Old adzed oak (grain along u).
        Oak = ("oak", medieval::oak, 1.6, 1.0, 30.0, [0.005, 0.02]),
        /// Straw thatch.
        Thatch = ("thatch", medieval::thatch, 2.4, 1.0, 60.0, [0.006, 0.03]),
        /// Split wooden shingles.
        Shingle = ("shingle", medieval::shingle, 2.0, 1.0, 60.0, [0.006, 0.02]),
        /// Leaded diamond-pane glass.
        Leaded = ("leaded", medieval::leaded, 0.88, 0.6, 10.0, [0.003, 0.01]),
        /// Rounded field-stone cobbles.
        Cobbles = ("cobbles", medieval::cobbles, 2.1, 1.0, 50.0, [0.006, 0.025]),
        /// Worn flagstones.
        Flagstone = ("flagstone", medieval::flagstone, 2.4, 1.0, 45.0, [0.008, 0.04]),
        /// Old hand-made brick with salt bloom.
        Brick = ("brick", medieval::brick, 1.8, 1.0, 40.0, [0.004, 0.02]),
        /// Floorboards.
        Boards = ("boards", medieval::boards, 2.0, 1.0, 30.0, [0.004, 0.02]),
        /// Limewashed rubble, flaking and damp.
        Limewash = ("limewash", medieval::limewash, 2.4, 1.0, 40.0, [0.01, 0.05]),
    }
}

/// Any texgen layer; every family converts into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layer {
    NaturalLayer(NaturalLayer),
    GroundLayer(GroundLayer),
    FacadeLayer(FacadeLayer),
    RoofLayer(RoofLayer),
    MedievalLayer(MedievalLayer),
}

impl Layer {
    pub fn generate(self, size: u32) -> TextureLayer {
        match self {
            Layer::NaturalLayer(l) => l.generate(size),
            Layer::GroundLayer(l) => l.generate(size),
            Layer::FacadeLayer(l) => l.generate(size),
            Layer::RoofLayer(l) => l.generate(size),
            Layer::MedievalLayer(l) => l.generate(size),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Layer::NaturalLayer(l) => l.name(),
            Layer::GroundLayer(l) => l.name(),
            Layer::FacadeLayer(l) => l.name(),
            Layer::RoofLayer(l) => l.name(),
            Layer::MedievalLayer(l) => l.name(),
        }
    }

    /// Every layer of every family.
    pub fn all() -> Vec<Layer> {
        let mut v: Vec<Layer> = Vec::new();
        v.extend(NaturalLayer::ALL.iter().map(|l| Layer::from(*l)));
        v.extend(GroundLayer::ALL.iter().map(|l| Layer::from(*l)));
        v.extend(FacadeLayer::ALL.iter().map(|l| Layer::from(*l)));
        v.extend(RoofLayer::ALL.iter().map(|l| Layer::from(*l)));
        v.extend(MedievalLayer::ALL.iter().map(|l| Layer::from(*l)));
        v
    }
}

/// The layer at [`TEX_SIZE`], generated on first use and shared after.
pub fn layer(layer: impl Into<Layer>) -> Arc<TextureLayer> {
    static CACHE: OnceLock<Mutex<HashMap<Layer, Arc<TextureLayer>>>> = OnceLock::new();
    let key = layer.into();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(l) = cache.lock().expect("texgen cache").get(&key) {
        return l.clone();
    }
    let made = Arc::new(key.generate(TEX_SIZE));
    cache
        .lock()
        .expect("texgen cache")
        .entry(key)
        .or_insert(made)
        .clone()
}

/// Generate `layers` on all cores now, so later [`layer`] calls return at
/// once (handy behind a loading screen).
pub fn prepare(layers: &[Layer]) {
    let todo: Vec<Layer> = layers.to_vec();
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(todo.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(l) = todo.get(i) else { break };
                layer(*l);
            });
        }
    });
}

/// The natural layer at [`TEX_SIZE`], shared (same as [`layer`]).
pub fn natural_layer(l: NaturalLayer) -> Arc<TextureLayer> {
    layer(l)
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
    fn every_layer_tiles_varies_and_repeats() {
        const N: usize = 64;
        let mut seams = Vec::new();
        for layer in Layer::all() {
            let a = layer.generate(N as u32);
            let b = layer.generate(N as u32);
            assert_eq!(a.albedo.len(), N * N * 4);
            assert_eq!(a.albedo, b.albedo, "{layer:?} deterministic");
            assert_eq!(a.normal, b.normal);
            let (wrap, inner) = seam(&a.albedo, N);
            // Mullions, panel joints and tile courses lie on the tile edge by design.
            let edge_lines = matches!(
                layer,
                Layer::FacadeLayer(FacadeLayer::GlassCurtain | FacadeLayer::ConcretePanel)
                    | Layer::RoofLayer(RoofLayer::ClayTile)
            );
            if !edge_lines && wrap >= inner * 5 / 2 + N as u32 {
                seams.push(format!("{layer:?} seam {wrap} vs {inner}"));
            }
            let g = a.albedo.chunks(4).map(|p| p[1]);
            let (lo, hi) = g.fold((255, 0), |(lo, hi), v| (lo.min(v), hi.max(v)));
            assert!(hi - lo > 10, "{layer:?} has visible variation");
            assert!(a.albedo.chunks(4).all(|p| p[3] == 255), "opaque");
        }
        assert!(seams.is_empty(), "{seams:#?}");
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
