use crate::anim::{AnimatedModel, AnimationAction, AnimationProfile, Animator, Locomotion};
use crate::anim_graph::{AnimEvent, AnimGraph};
use crate::camera::Camera;
use crate::collision::{ActorBody, ActorMove, CollisionWorld};
use crate::color::Color;
use crate::contact::ContactSnapshot;
use crate::error::{EngineError, EngineResult};
use crate::input::Input;
use crate::limits::EngineLimits;
use crate::lod::{LodId, LodLevel, LodSet};
use crate::mesh::{AlbedoMap, BuiltMesh, MaterialMaps, Mesh, PreparedMesh};
use crate::particles::ParticleWorld;
use crate::place::{GlobalPlace, MeshInstance, Place};
use crate::portal::{
    opening_extents, segment_crosses_opening, teleport_yaw, Portal, PortalId, PortalPlane,
    PortalSettings, SpaceId, VisiblePortal,
};
use crate::proc_terrain::{HeightField, ProcTerrain};
use crate::ribbons::RibbonWorld;
use crate::space::{ChunkId, GlobalPosition, GlobalXZ, RenderOrigin};
use crate::texture::{
    generate_terrain_albedo, load_rgba8_png, CpuTexture, MaterialId, TerrainAlbedo,
    TerrainMaterial, TerrainMaterialDesc, TextureId, WaterMaterial, WaterMaterialDesc,
    WaterMaterialId,
};
use crate::ui::UiFrame;
use glam::{IVec3, Mat4, Vec2, Vec3};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "save.rs"]
mod save;
pub use save::Loaded;

#[path = "nav.rs"]
mod nav;

#[path = "world_audio.rs"]
mod world_audio;
#[path = "world_buildings.rs"]
mod world_buildings;
#[path = "world_census.rs"]
mod world_census;
#[path = "world_cpu_mesh.rs"]
mod world_cpu_mesh;
#[path = "world_crowds.rs"]
mod world_crowds;
#[path = "world_decals.rs"]
mod world_decals;
#[path = "world_effects.rs"]
pub mod world_effects;
#[path = "world_grading.rs"]
mod world_grading;
#[path = "world_ground.rs"]
mod world_ground;
pub use world_cpu_mesh::CpuMeshes;
#[path = "world_components.rs"]
pub(crate) mod world_components;
#[path = "world_ragdolls.rs"]
mod world_ragdolls;
#[path = "world_rooms.rs"]
mod world_rooms;
// `impl World`, by topic.
#[path = "world_animation.rs"]
mod world_animation;
#[path = "world_custom_materials.rs"]
mod world_custom_materials;
#[path = "world_hierarchy.rs"]
mod world_hierarchy;
#[path = "world_labels.rs"]
mod world_labels;
#[path = "world_lamps.rs"]
mod world_lamps;
#[path = "world_lighting.rs"]
mod world_lighting;
#[path = "world_lod_quality.rs"]
mod world_lod_quality;
#[path = "world_physics.rs"]
mod world_physics;
#[path = "world_resources.rs"]
mod world_resources;
#[path = "world_sim.rs"]
mod world_sim;
#[path = "world_sky.rs"]
mod world_sky;
#[path = "world_spaces.rs"]
mod world_spaces;
#[path = "world_spawn.rs"]
mod world_spawn;
#[path = "world_terrain.rs"]
mod world_terrain;
#[path = "world_verify.rs"]
pub(crate) mod world_verify;
#[path = "world_zfight.rs"]
mod world_zfight;
pub use nav::{NavAgent, NavSettings, NavStats, NavStatus};
pub use world_zfight::{WorldScan, MAX_Z_FIGHTING_MARKS};

/// Opaque handle to a spawned entity. Saves can hold it; after loading,
/// map it to the new one with [`Loaded::entity`].
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct EntityId(u64);

#[cfg(test)]
impl EntityId {
    pub(crate) fn test(id: u64) -> Self {
        Self(id)
    }
}

impl EntityId {
    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug)]
pub struct Light {
    /// Direction toward the light (will be normalized).
    pub direction: Vec3,
    pub color: Vec3,
    pub ambient: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.4, 1.0, 0.25),
            color: Vec3::splat(1.0),
            ambient: 0.22,
        }
    }
}

/// A light carried by the viewer ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a lantern, a torch, headlamp.
///
/// Shaded as a point light at the camera eye with a smooth inverse-square-ish
/// falloff, so nearby surfaces brighten and the glow dies out over
/// [`Self::radius_m`]. The engine holds one; games toggle it per scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorchLight {
    pub color: Color,
    /// Metres at which the torch contribution has faded to nothing.
    pub radius_m: f32,
    /// Exponent of the smooth distance falloff. Lower values keep more light
    /// at range; higher values concentrate the torch near the player.
    pub curve: f32,
}

impl TorchLight {
    /// Warm handheld lantern: amber cast reaching a room-scale distance.
    pub fn lantern() -> Self {
        Self {
            color: Color::rgb(255, 214, 156),
            radius_m: 18.0,
            curve: 2.0,
        }
    }

    /// Cool cave headlamp: whiter, tighter beam reach.
    pub fn headlamp() -> Self {
        Self {
            color: Color::rgb(228, 236, 255),
            radius_m: 50.0,
            curve: 1.35,
        }
    }
}

/// How instanced meshes reach the GPU.
///
/// Both paths are always compiled. The selected one is the only one that runs;
/// there is no silent fallback from GPU to CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InstanceSubmit {
    /// Entity-sphere cull, then `draw_indexed` of every instance.
    CpuIndexed,
    /// Prototype-batched per-instance GPU compact, then `draw_indexed_indirect`.
    /// The default: each shadow cascade and the view draw only the instances
    /// inside them (lod_forest at 4K: shadows 1.44 -> 0.43 ms, +23% FPS).
    #[default]
    GpuIndirect,
}

impl InstanceSubmit {
    /// `ENGINE_INSTANCE_SUBMIT=cpu` starts on the CPU path (default gpu).
    fn from_env() -> Self {
        match std::env::var("ENGINE_INSTANCE_SUBMIT").as_deref() {
            Ok("cpu") => Self::CpuIndexed,
            _ => Self::GpuIndirect,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuIndexed => "cpu_indexed",
            Self::GpuIndirect => "gpu_indirect",
        }
    }
}

/// Fast 64-bit hash of texture bytes (several GB/s), to find identical maps.
/// Four independent lanes keep the CPU busy; not for security.
/// The 1x1 white albedo that maps without an albedo ride along with.
fn white_texel() -> AlbedoMap {
    AlbedoMap {
        width: 1,
        height: 1,
        rgba: vec![255; 4].into(),
    }
}

/// The key a texture is shared by: large maps (or all, with `share`) are
/// looked up by content, so generated assets (every tree, every facade)
/// that carry the same maps share one GPU texture. `None`: not shared.
pub(crate) fn texture_key(
    albedo: Option<&AlbedoMap>,
    maps: Option<&MaterialMaps>,
    share: bool,
) -> Option<u64> {
    let white;
    let map = match (albedo, maps) {
        (Some(map), _) => map,
        (None, Some(_)) => {
            white = white_texel();
            &white
        }
        (None, None) => return None,
    };
    (share || map.rgba.len() >= 1 << 20).then(|| {
        let mut h = content_hash(&map.rgba) ^ ((map.width as u64) << 32 | map.height as u64);
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        if let Some(m) = maps {
            for map in [&m.normal, &m.orm, &m.emissive] {
                mix(map.as_ref().map_or(1, |n| content_hash(&n.rgba)));
            }
            mix(m.normal_scale.to_bits() as u64);
            mix(m.roughness.to_bits() as u64);
            mix(m.metallic.to_bits() as u64);
            mix(m.triplanar.to_bits() as u64);
        }
        h
    })
}

pub(crate) fn content_hash(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut lanes = [K, K ^ 1, K ^ 2, K ^ 3];
    let mut blocks = bytes.chunks_exact(32);
    for block in &mut blocks {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let v = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
            *lane = (*lane ^ v).wrapping_mul(K).rotate_left(31);
        }
    }
    let mut h = bytes.len() as u64;
    for (i, &b) in blocks.remainder().iter().enumerate() {
        h ^= (b as u64) << ((i % 8) * 8);
        if i % 8 == 7 {
            h = h.wrapping_mul(K).rotate_left(27);
        }
    }
    for lane in lanes {
        h = (h ^ lane).wrapping_mul(K).rotate_left(27);
    }
    h ^ (h >> 32)
}

/// [`crate::world::World::set_debug_shading`] flags: what each bit does.
pub const DEBUG_SHADING_FLAGS: [(u32, &str); 17] = [
    (1, "keep LOD-faded pixels"),
    (2, "keep broken elements"),
    (4, "depth prepass keeps every pixel"),
    (8, "textured surfaces magenta"),
    (16, "untextured surfaces cyan"),
    (32, "every pixel opaque"),
    (64, "invalid colours green"),
    (128, "no reflections"),
    (256, "no lamp light"),
    (512, "no decals"),
    (1024, "no bounce light (GI)"),
    (2048, "textures without normal/ORM maps"),
    (4096, "colour by depth value"),
    (8192, "hide things farther than 60 m"),
    (16384, "world pass writes depth too"),
    (32768, "red where depth does not match position"),
    (65536, "world pass clears its own depth"),
];

/// The curve that turns the HDR scene into screen colours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneMap {
    /// AgX (default): bright, saturated light fades towards white like film,
    /// so sunsets, fire and spell effects keep their hue instead of turning
    /// into flat yellow or cyan.
    #[default]
    Agx,
    /// The classic ACES fit: more contrast, more saturated highlights.
    Aces,
}

impl ToneMap {
    /// `"agx"` or `"aces"` (any case), for settings files and env vars.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "agx" => Some(Self::Agx),
            "aces" => Some(Self::Aces),
            _ => None,
        }
    }
}

/// HDR bloom and tone-mapping controls for the rendered world.
///
/// Values are validated when constructed or changed. Disabling bloom leaves
/// HDR rendering and filmic tone mapping active, but contributes no blur.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomSettings {
    enabled: bool,
    threshold: f32,
    soft_knee: f32,
    intensity: f32,
    exposure: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 1.0,
            soft_knee: 0.5,
            intensity: 0.35,
            exposure: 1.0,
        }
    }
}

impl BloomSettings {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn enabled(self) -> bool {
        self.enabled
    }

    pub fn threshold(self) -> f32 {
        self.threshold
    }

    pub fn soft_knee(self) -> f32 {
        self.soft_knee
    }

    pub fn intensity(self) -> f32 {
        self.intensity
    }

    pub fn exposure(self) -> f32 {
        self.exposure
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> EngineResult<Self> {
        self.set_threshold(threshold)?;
        Ok(self)
    }

    pub fn with_soft_knee(mut self, soft_knee: f32) -> EngineResult<Self> {
        self.set_soft_knee(soft_knee)?;
        Ok(self)
    }

    pub fn with_intensity(mut self, intensity: f32) -> EngineResult<Self> {
        self.set_intensity(intensity)?;
        Ok(self)
    }

    pub fn with_exposure(mut self, exposure: f32) -> EngineResult<Self> {
        self.set_exposure(exposure)?;
        Ok(self)
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_threshold(&mut self, threshold: f32) -> EngineResult<()> {
        require_finite_non_negative("bloom threshold", threshold)?;
        self.threshold = threshold;
        Ok(())
    }

    pub fn set_soft_knee(&mut self, soft_knee: f32) -> EngineResult<()> {
        if !soft_knee.is_finite() || !(0.0..=1.0).contains(&soft_knee) {
            return Err(EngineError::InvalidValue(
                "bloom soft knee must be finite and in 0..=1".into(),
            ));
        }
        self.soft_knee = soft_knee;
        Ok(())
    }

    pub fn set_intensity(&mut self, intensity: f32) -> EngineResult<()> {
        require_finite_non_negative("bloom intensity", intensity)?;
        self.intensity = intensity;
        Ok(())
    }

    pub fn set_exposure(&mut self, exposure: f32) -> EngineResult<()> {
        if !exposure.is_finite() || exposure <= 0.0 {
            return Err(EngineError::InvalidValue(
                "tone-map exposure must be finite and > 0".into(),
            ));
        }
        self.exposure = exposure;
        Ok(())
    }
}

/// Screen-space ambient occlusion: contact shadows in creases, under props
/// and where walls meet the ground. It darkens sky light only; the sun has
/// its own shadows. On by default; `World::set_ambient_occlusion(None)`
/// switches it off and skips the depth prepass it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientOcclusion {
    radius_m: f32,
    intensity: f32,
}

impl Default for AmbientOcclusion {
    fn default() -> Self {
        Self {
            radius_m: 1.0,
            intensity: 1.0,
        }
    }
}

impl AmbientOcclusion {
    /// `radius_m`: how far, in metres, geometry still occludes a point.
    /// `intensity`: 0 none, 1 the default, higher darkens creases more.
    pub fn new(radius_m: f32, intensity: f32) -> EngineResult<Self> {
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(EngineError::InvalidValue(
                "ambient occlusion radius must be finite and > 0".into(),
            ));
        }
        require_finite_non_negative("ambient occlusion intensity", intensity)?;
        Ok(Self {
            radius_m,
            intensity,
        })
    }

    pub fn radius_m(self) -> f32 {
        self.radius_m
    }

    pub fn intensity(self) -> f32 {
        self.intensity
    }
}

/// Global illumination: sky light that knows about walls, roofs and caves,
/// plus sunlight bounced off nearby surfaces. Light probes around the camera
/// trace a voxel copy of the scene that is rebuilt whenever geometry changes,
/// so destruction relights itself. On by default;
/// `World::set_global_illumination(None)` switches it off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalIllumination {
    probe_spacing_m: f32,
}

impl Default for GlobalIllumination {
    fn default() -> Self {
        Self {
            probe_spacing_m: 2.0,
        }
    }
}

impl GlobalIllumination {
    /// `probe_spacing_m`: metres between light probes. The probe box spans
    /// 32 x 16 x 32 probes around the camera; voxels are half this size.
    /// Smaller spacing gives finer detail over a smaller area.
    pub fn new(probe_spacing_m: f32) -> EngineResult<Self> {
        if !probe_spacing_m.is_finite() || probe_spacing_m <= 0.0 {
            return Err(EngineError::InvalidValue(
                "GI probe spacing must be finite and > 0".into(),
            ));
        }
        Ok(Self { probe_spacing_m })
    }

    pub fn probe_spacing_m(self) -> f32 {
        self.probe_spacing_m
    }
}

fn require_finite_non_negative(name: &str, value: f32) -> EngineResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(EngineError::InvalidValue(format!(
            "{name} must be finite and >= 0"
        )));
    }
    Ok(())
}
impl fmt::Display for InstanceSubmit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Nearby mesh cascades plus optional height-field raymarch.
///
/// Shadows are on by default once a sun is set. Pass `None` to [`World::set_shadows`]
/// to turn them off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowSettings {
    /// Camera-distance splits for the three cascaded maps, in metres.
    pub cascade_end_m: [f32; 3],
    /// Edge length of each cascade depth map.
    pub map_size: u32,
    /// Raymarch a height source (clipmap formula or resident contact atlas).
    pub raymarch_height: bool,
    /// World-XZ extent of the contact height atlas, in metres.
    pub atlas_extent_m: f32,
    /// Edge length of the contact height atlas.
    pub atlas_size: u32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            cascade_end_m: [12.0, 40.0, 120.0],
            map_size: 1024,
            raymarch_height: true,
            atlas_extent_m: 1024.0,
            atlas_size: 1024,
        }
    }
}

/// A procedural sky dome: zenith, horizon, ground, and a sun.
///
/// Drawn as a fullscreen pass behind every surface. Pair the horizon colour
/// with [`Haze::color`] so distant ground dissolves into the same band the
/// sky is already showing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    pub zenith: Color,
    pub horizon: Color,
    pub ground: Color,
    pub sun_color: Color,
    /// Angular radius of the sun disc, in degrees.
    pub sun_size_degrees: f32,
    /// How far the sun's bloom reaches past the disc, in degrees.
    pub sun_bloom_degrees: f32,
    /// 0 is an even wash from zenith to horizon; 1 pins colour at the horizon.
    pub curve: f32,
}

impl Sky {
    /// Cool daylight: deeper blue above, a pale warm horizon, a low sun.
    ///
    /// Colours follow the title vista: slate zenith, gold near the sun, a
    /// long bloom rather than a hard disc.
    pub fn daylight() -> Self {
        Self {
            zenith: Color::rgb(74, 114, 168),
            horizon: Color::rgb(214, 208, 198),
            ground: Color::rgb(98, 108, 112),
            sun_color: Color::rgb(255, 244, 220),
            sun_size_degrees: 1.4,
            sun_bloom_degrees: 22.0,
            curve: 0.22,
        }
    }
}

/// The air between the eye and the world.
///
/// Distance haze is what stops a view distance from reading as an edge: with
/// it, ground far enough away is indistinguishable from sky, so the last chunk
/// dissolves instead of ending. Set the colour to the sky's horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Haze {
    pub color: Color,
    /// Distance at which a surface has all but dissolved into `color`.
    pub visibility_m: f32,
    /// Altitude the air starts to thin from.
    pub base_y: f32,
    /// Every this many metres above `base_y`, the air is `1/e` as thick, so
    /// mountains stand clear of the murk their valleys are buried in.
    pub height_m: f32,
}

impl Haze {
    /// Air of `visibility_m`, thinning over a kilometre of altitude.
    pub fn new(color: Color, visibility_m: f32) -> Self {
        Self {
            color,
            visibility_m,
            base_y: 0.0,
            height_m: 1_000.0,
        }
    }

    pub fn thinning_above(mut self, base_y: f32, height_m: f32) -> Self {
        self.base_y = base_y;
        self.height_m = height_m;
        self
    }

    /// Reciprocal metres for the shader, from the visibility distance.
    ///
    /// `1 - exp(-(dÃƒâ€šÃ‚Â·k)Ãƒâ€šÃ‚Â²)` reaches 0.98 at `d = visibility`, which is where a
    /// surface stops being distinguishable from the sky behind it.
    pub fn density(&self) -> f32 {
        1.978 / self.visibility_m.max(1.0)
    }
}

/// Which world-space material an entity is drawn with, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceMaterialRef {
    /// World-XZ grass/sand/rock ground.
    Terrain(MaterialId),
    /// Animated water sheet.
    Water(WaterMaterialId),
    /// A game's own shader code (`World::add_material`).
    Custom(crate::custom_material::CustomMaterialId),
}

#[derive(Clone, Debug)]
pub struct Entity {
    /// The mesh as built. Empty once [`Self::released`] is set.
    pub(crate) mesh: BuiltMesh,
    /// Set when the world let go of the CPU copy after the renderer
    /// uploaded it (see [`CpuMeshes`]): what is still known of it.
    pub(crate) released: Option<world_cpu_mesh::Released>,
    pub(crate) transform: Mat4,
    /// Per-instance transforms (+ tint) for GPU instancing.
    pub(crate) instances: Vec<MeshInstance>,
    /// Minimum GPU slot capacity requested by the owner. This reserves space
    /// without drawing placeholder instances.
    pub(crate) instance_reserve: usize,
    /// Whether `instances` is authoritative, even when it is empty.
    ///
    /// An instanced entity with nothing to place draws nothing; a plain entity
    /// with no instances draws once at `transform`.
    pub(crate) instanced: bool,
    pub(crate) material: Option<SurfaceMaterialRef>,
    /// Optional baked albedo sampled with mesh UVs. `None` uses a white 1ÃƒÆ’Ã¢â‚¬â€1.
    pub(crate) albedo: Option<TextureId>,
    /// Bumped when transform or instance list changes. The renderer skips
    /// rewriting the GPU instance buffer when this matches the last upload.
    pub(crate) xform_rev: u64,
    /// Another instanced entity whose GPU mesh this one shares.
    ///
    /// Scatter bins are many placements of one pine, not many pines. CPU submit
    /// gives each bin an instance buffer; GPU submit coalesces compatible bins
    /// into one prototype batch.
    pub(crate) instance_of: Option<EntityId>,
    /// Mesh casters only. Terrain and water never cast, even when this is set.
    pub(crate) casts_shadow: bool,
    pub(crate) space: SpaceId,
}

impl Entity {
    /// The mesh as built. Empty (no vertices) once the world has released
    /// its CPU copy (see [`CpuMeshes`] and [`Self::has_cpu_mesh`]).
    pub fn mesh(&self) -> &BuiltMesh {
        &self.mesh
    }

    /// Whether the world still holds the mesh in RAM. `false` once it was
    /// released after the upload (see [`CpuMeshes`]); the GPU copy draws on.
    pub fn has_cpu_mesh(&self) -> bool {
        self.released.is_none()
    }

    /// Vertices of the mesh, also after its CPU copy was released.
    pub fn vertex_count(&self) -> usize {
        self.released
            .as_ref()
            .map_or(self.mesh.vertex_count(), |r| r.vertex_count)
    }

    /// Indices of the mesh, also after its CPU copy was released.
    pub fn index_count(&self) -> usize {
        self.released
            .as_ref()
            .map_or(self.mesh.index_count(), |r| r.index_count)
    }

    pub fn transform(&self) -> Mat4 {
        self.transform
    }

    pub fn material(&self) -> Option<SurfaceMaterialRef> {
        self.material
    }

    /// Prototype this instance list is drawn with, if it does not own a mesh.
    pub fn instance_of(&self) -> Option<EntityId> {
        self.instance_of
    }

    pub fn casts_shadow(&self) -> bool {
        self.casts_shadow
    }

    pub fn space(&self) -> SpaceId {
        self.space
    }

    /// Whether the instance list is the draw list, including when that list is empty.
    pub fn instanced(&self) -> bool {
        self.instanced
    }

    fn bump_xform(&mut self) {
        self.xform_rev = self.xform_rev.wrapping_add(1);
    }
}

/// What a placed prefab made, so despawning its root removes all of it
/// (parts with dynamic bodies have let go of the root by then).
#[derive(Clone, Debug)]
pub(crate) struct PrefabInstance {
    pub prefab: String,
    pub parts: Vec<EntityId>,
    pub lamps: Vec<crate::lamps::LampId>,
}

#[derive(Clone, Copy, Debug)]
struct AnchoredChunk {
    entity: EntityId,
    anchor: GlobalPosition,
}

/// Skinned entity with clip playback.
#[derive(Clone, Debug)]
pub struct AnimatedEntity {
    pub(crate) animator: Animator,
    /// A state machine choosing the clips (`World::set_anim_graph`).
    pub(crate) graph: Option<Box<crate::anim_graph::AnimGraphRun>>,
    pub(crate) transform: Mat4,
    pub(crate) space: SpaceId,
}

impl AnimatedEntity {
    pub fn animator(&self) -> &Animator {
        &self.animator
    }

    pub fn transform(&self) -> Mat4 {
        self.transform
    }

    pub fn space(&self) -> SpaceId {
        self.space
    }
}

/// Scene state visible to user update callbacks.
#[derive(Debug)]
pub struct World {
    camera: Camera,
    light: Light,
    clear_color: Color,
    pub(crate) limits: EngineLimits,
    next_id: u64,
    entities: HashMap<EntityId, Entity>,
    order: Vec<EntityId>,
    /// Despawned ids still listed in `order` (dropped in bulk, so despawning
    /// many entities stays cheap).
    order_stale: usize,
    /// Changes whenever static render state could require a GPU resync.
    render_epoch: u64,
    animated: HashMap<EntityId, AnimatedEntity>,
    animated_order: Vec<EntityId>,
    /// Chunk key -> entity id for streamed volume meshes (advanced).
    pub(crate) chunk_entities: HashMap<glam::IVec3, EntityId>,
    /// Horizontal anchor that render space is measured from.
    render_origin: RenderOrigin,
    /// Entities whose transform is re-derived on rebase.
    anchored_entities: HashMap<EntityId, GlobalPlace>,
    /// Streamed chunks with global anchors and chunk-local vertices.
    anchored_chunks: HashMap<ChunkId, AnchoredChunk>,
    /// Optional GPU procgen terrain (clipmap). Drawn before entity meshes.
    pub(crate) proc_terrain: Option<ProcTerrain>,
    /// CPU sampler matching the GPU formula (for feet / gameplay).
    pub(crate) height_field: Option<HeightField>,
    pub(crate) textures: HashMap<TextureId, CpuTexture>,
    /// Normal / ORM / emissive maps that go with an albedo texture.
    pub(crate) surface_maps: HashMap<TextureId, std::sync::Arc<MaterialMaps>>,
    pub(crate) materials: HashMap<MaterialId, TerrainMaterial>,
    next_texture_id: u64,
    next_material_id: u64,
    pub(crate) water_materials: HashMap<WaterMaterialId, WaterMaterial>,
    /// Changes whenever textures or material resources are added.
    resource_epoch: u64,
    /// Bumped by [`Self::request_gpu_reupload`]: (meshes, textures).
    pub(crate) reupload: (u64, u64),
    /// Bumped by [`Self::request_gpu_check`].
    pub(crate) gpu_check: u64,
    gpu_info: Option<crate::gpu_info::GpuInfo>,
    /// Bumped by [`Self::request_renderer_rebuild`]: (surface, screen buffers).
    pub(crate) renderer_rebuild: (u64, u64),
    /// [`Self::set_debug_shading`] flags.
    pub(crate) debug_shading: u32,
    /// Changes when only instance lists changed (LOD swaps): the renderer
    /// re-uploads instances without rebuilding GI or shadow sources.
    instance_epoch: u64,
    lods: std::collections::BTreeMap<LodId, LodSet>,
    next_lod: u64,
    /// LOD sets that follow an entity ([`Self::attach_lod`]): the parent
    /// and where on it.
    lod_follow: std::collections::BTreeMap<LodId, (EntityId, Mat4)>,
    lod_bias: f32,
    lod_fade: f32,
    /// Applied to fully-opaque `set_chunk_built` uploads when set.
    pub(crate) default_terrain_material: Option<MaterialId>,
    /// Applied to fully-translucent chunk layers when set.
    pub(crate) default_water_material: Option<WaterMaterialId>,
    /// Whether the game wants the pointer pinned for mouse-look.
    pointer_lock: bool,
    /// Settings bind-listen: Esc cancels listen only (no close / unlock / quit).
    bind_listen: bool,
    /// Seconds since start, for animated materials.
    time: f32,
    /// How far the camera helpers put their far plane.
    view_distance: f32,
    /// The air, when the game wants any.
    haze: Option<Haze>,
    /// Procedural sky behind the scene, when the game wants one.
    sky: Option<Sky>,
    /// Time of day and weather driving sun, sky, haze and exposure.
    physical_sky: Option<crate::sky::PhysicalSky>,
    moons: Vec<crate::sky::MoonSpec>,
    wind: f32,
    season: f32,
    /// Glow channels: colour (r < 0: the surface's own) and strength.
    glow: [[f32; 4]; crate::mesh::GLOW_CHANNELS],
    /// Albedo textures by content, so meshes that share an atlas (every
    /// generated tree) share one GPU texture.
    texture_by_content: HashMap<u64, TextureId>,
    /// Textures the world made itself from a mesh's albedo map: freed once
    /// no entity draws with them (see [`Self::tick_textures`]). Textures
    /// the game creates (`create_texture_rgba`) are never in here.
    auto_textures: HashSet<TextureId>,
    /// The renderer's last count (see [`crate::census`]).
    gpu_census: crate::census::GpuCensus,
    /// An entity holding an auto texture went away: sweep next tick.
    texture_sweep: bool,
    /// Light carried by the viewer (lantern / headlamp), when lit.
    torch: Option<TorchLight>,
    lamps: std::collections::BTreeMap<crate::lamps::LampId, (crate::lamps::Lamp, SpaceId)>,
    next_lamp: u32,
    lamp_shadow_budget: usize,
    /// Parent/child links, pivots and lamps riding on entities.
    hierarchy: crate::hierarchy::Hierarchy,
    /// Prefabs by name, for spawning by name and for loading saves.
    prefabs: HashMap<String, Arc<crate::prefab::Prefab>>,
    /// Placed prefabs by root: which prefab and every entity and lamp made.
    prefab_instances: HashMap<EntityId, PrefabInstance>,
    /// Walkable areas, paths and agents.
    nav: nav::NavState,
    /// Nearby mesh CSM + height raymarch. `None` disables shadows.
    shadows: Option<ShadowSettings>,
    /// Every how many frames each sun cascade is drawn anew.
    shadow_cascade_refresh: [u32; 3],
    shadow_cache: bool,
    /// Instanced mesh submit path. Default stays CPU so Engine demos match today.
    instance_submit: InstanceSubmit,
    bloom: BloomSettings,
    ambient_occlusion: Option<AmbientOcclusion>,
    global_illumination: Option<GlobalIllumination>,
    /// Smooth destructible terrain, meshed by the renderer on the GPU.
    voxel_terrain: Option<crate::voxel_terrain::VoxelTerrain>,
    /// The world stream, if one is set (taken out while it updates).
    stream: Option<Box<crate::stream::WorldStream>>,
    /// Volume revision when the voxel terrain was set: chunks changed after
    /// it were dug or built on, and go into saves.
    voxel_baseline_rev: u64,
    water: Option<crate::water::Water>,
    /// Voxel terrain revision the water last read its columns from.
    water_terrain_rev: u64,
    /// Rigid bodies, including pieces of shattered meshes.
    physics: crate::physics::Physics,
    /// The fixed simulation tick (see [`crate::sim`]), when on.
    sim_clock: Option<crate::sim::SimClock>,
    /// Steps begun with `begin_sim_step`.
    sim_tick: u64,
    /// Where drawing is between the last two steps.
    sim_alpha: f32,
    sim_poses: HashMap<EntityId, crate::sim::SimPose>,
    /// Breakable buildings by their mesh entity.
    buildings: HashMap<EntityId, world_buildings::BuildingVisual>,
    /// Element bits the GPU reads to hide broken parts.
    element_states: world_buildings::ElementStates,
    falling: Vec<world_buildings::FallingPart>,
    rubble: world_buildings::RubbleSet,
    /// Characters knocked down, lying or getting up.
    ragdolls: world_ragdolls::Ragdolls,
    /// Indoor culling: rooms, openings, and what is in which room.
    rooms: crate::rooms::Rooms,
    reflection_probes: crate::reflections::ReflectionProbes,
    decals: world_decals::DecalPool,
    grading: crate::grading::Grading,
    crowds: Vec<Option<crate::crowd::Crowd>>,
    ssr: Option<crate::reflections::ScreenSpaceReflections>,
    entity_rooms: HashMap<EntityId, crate::rooms::RoomId>,
    /// Z-fighting markers and the `ENGINE_ZFIGHT` watcher.
    zfight: world_zfight::ZFightState,
    /// [`Self::check`] and [`Self::report`] results for verify runs.
    verify: world_verify::VerifyLog,
    /// Game data on entities: see [`World::insert`].
    components: world_components::Components,
    /// What the game started with ([`World::settings`]).
    settings: crate::settings::Settings,
    /// `ENGINE_*` variables set at startup ([`World::env_overrides`]).
    env_overrides: Vec<String>,
    /// What feet stand on: see [`World::ground_below`].
    ground: world_ground::GroundState,
    /// Which CPU copies of meshes go once uploaded: see [`CpuMeshes`].
    cpu_meshes: world_cpu_mesh::CpuMeshState,
    /// Debug names (see [`World::set_name`]).
    entity_names: HashMap<EntityId, String>,
    /// Surface presentation sync. Engine default stays on so demos never tear;
    /// the renderer applies it at every present.
    vsync: bool,
    whats_new_button: bool,
    /// Resident contact grids, pushed by the chunk streamer when they change.
    shadow_contact: ContactSnapshot,
    shadow_contact_epoch: u64,
    /// Named costs for the current frame. Printed only on a hitch, and only
    /// when [`Self::set_hitch_log`] has a path.
    hitch_spans: Vec<HitchSpan>,
    hitch_log: Option<PathBuf>,
    /// PNG paths the game asked to capture after the next drawn frame.
    screenshot_queue: Vec<PathBuf>,
    /// Close the window after this frame (and any queued shots) finish.
    exit_requested: bool,
    /// Set by the game when the scene is ready for a waiting screenshot harness.
    ready: bool,
    /// Static obstacles actors slide against. Independent of render entities.
    collision: CollisionWorld,
    space_by_name: HashMap<String, SpaceId>,
    next_space_id: u32,
    spawn_space: SpaceId,
    live_space: SpaceId,
    /// Named spaces that also draw sky and clipmap. [`SpaceId::DEFAULT`] always does.
    environment_spaces: HashSet<SpaceId>,
    portals: Vec<Portal>,
    next_portal_id: u32,
    /// Max portal-in-portal recursion depth for rendering.
    portal_recursion: u8,
    travel_prev: Option<(SpaceId, Vec3)>,
    /// Destination opening ignored for one frame after crossing (prevents ping-pong).
    travel_gate: Option<EntityId>,
    particles: ParticleWorld,
    /// Text rising from points (`float_text`).
    float_texts: Vec<crate::labels::FloatText>,
    /// Materials made of game shader code, with a revision per change.
    pub(crate) custom_materials: std::collections::BTreeMap<
        crate::custom_material::CustomMaterialId,
        (crate::custom_material::CustomMaterial, u64),
    >,
    next_custom_material: u32,
    /// Clip events since `take_anim_events` was last called.
    anim_events: Vec<crate::anim_graph::AnimEvent>,
    labels_on_screen: usize,
    /// Dust and flying chunks when things are dug, shattered or hit.
    destruction_effects: bool,
    audio: crate::audio::AudioScene,
    auto_audio: world_audio::AutoAudio,
    ribbons: RibbonWorld,
    quality: crate::quality::Quality,
    taa: bool,
    /// Share of the window the scene is drawn at (1 = full size).
    render_scale: f32,
    occlusion_mode: OcclusionMode,
    occlusion_stats: OcclusionStats,
    tone_map: ToneMap,
    grass: f32,
    auto_tune: Option<crate::quality::AutoTune>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            light: Light::default(),
            clear_color: Color::rgb(133, 184, 235),
            limits: EngineLimits::default(),
            next_id: 1,
            entities: HashMap::new(),
            order: Vec::new(),
            order_stale: 0,
            render_epoch: 1,
            animated: HashMap::new(),
            animated_order: Vec::new(),
            chunk_entities: HashMap::new(),
            render_origin: RenderOrigin::default(),
            anchored_entities: HashMap::new(),
            anchored_chunks: HashMap::new(),
            proc_terrain: None,
            height_field: None,
            textures: HashMap::new(),
            surface_maps: HashMap::new(),
            materials: HashMap::new(),
            next_texture_id: 1,
            next_material_id: 1,
            water_materials: HashMap::new(),
            resource_epoch: 1,
            reupload: (0, 0),
            gpu_check: 0,
            gpu_info: None,
            renderer_rebuild: (0, 0),
            debug_shading: 0,
            instance_epoch: 1,
            lods: std::collections::BTreeMap::new(),
            next_lod: 1,
            lod_follow: std::collections::BTreeMap::new(),
            lod_bias: 1.0,
            lod_fade: 0.1,
            default_terrain_material: None,
            default_water_material: None,
            pointer_lock: false,
            bind_listen: false,
            time: 0.0,
            view_distance: Camera::default().far(),
            haze: None,
            sky: None,
            physical_sky: None,
            moons: Vec::new(),
            wind: 0.5,
            season: 0.0,
            glow: [[-1.0, 0.0, 0.0, 1.0]; crate::mesh::GLOW_CHANNELS],
            texture_by_content: HashMap::new(),
            auto_textures: HashSet::new(),
            gpu_census: Default::default(),
            texture_sweep: false,
            torch: None,
            lamps: std::collections::BTreeMap::new(),
            next_lamp: 1,
            lamp_shadow_budget: 4,
            hierarchy: Default::default(),
            prefabs: HashMap::new(),
            prefab_instances: HashMap::new(),
            nav: Default::default(),
            shadows: Some(ShadowSettings::default()),
            shadow_cascade_refresh: DEFAULT_SHADOW_CASCADE_REFRESH,
            shadow_cache: true,
            instance_submit: InstanceSubmit::from_env(),
            bloom: BloomSettings::default(),
            ambient_occlusion: Some(AmbientOcclusion::default()),
            global_illumination: Some(GlobalIllumination::default()),
            voxel_terrain: None,
            voxel_baseline_rev: 0,
            water: None,
            water_terrain_rev: 0,
            stream: None,
            physics: crate::physics::Physics::new(),
            sim_clock: None,
            sim_tick: 0,
            sim_alpha: 1.0,
            sim_poses: HashMap::new(),
            buildings: HashMap::new(),
            element_states: Default::default(),
            falling: Vec::new(),
            rubble: Default::default(),
            ragdolls: Default::default(),
            rooms: crate::rooms::Rooms::new(),
            reflection_probes: Default::default(),
            decals: Default::default(),
            grading: Default::default(),
            crowds: Vec::new(),
            ssr: Some(Default::default()),
            entity_rooms: HashMap::new(),
            zfight: Default::default(),
            verify: Default::default(),
            components: Default::default(),
            settings: Default::default(),
            env_overrides: Vec::new(),
            ground: Default::default(),
            cpu_meshes: Default::default(),
            entity_names: HashMap::new(),
            vsync: true,
            whats_new_button: true,
            shadow_contact: ContactSnapshot::default(),
            shadow_contact_epoch: 0,
            hitch_spans: Vec::new(),
            hitch_log: None,
            screenshot_queue: Vec::new(),
            exit_requested: false,
            ready: false,
            collision: CollisionWorld::new(),
            space_by_name: HashMap::new(),
            next_space_id: 1,
            spawn_space: SpaceId::DEFAULT,
            live_space: SpaceId::DEFAULT,
            environment_spaces: HashSet::new(),
            portals: Vec::new(),
            next_portal_id: 1,
            portal_recursion: 3,
            travel_prev: None,
            travel_gate: None,
            particles: ParticleWorld::default(),
            float_texts: Vec::new(),
            custom_materials: Default::default(),
            next_custom_material: 0,
            anim_events: Vec::new(),
            labels_on_screen: 0,
            destruction_effects: true,
            audio: crate::audio::AudioScene::new(),
            auto_audio: Default::default(),
            ribbons: RibbonWorld::default(),
            quality: crate::quality::Quality::High,
            taa: true,
            render_scale: 1.0,
            occlusion_mode: OcclusionMode::Auto,
            occlusion_stats: OcclusionStats::default(),
            tone_map: ToneMap::default(),
            grass: 1.0,
            auto_tune: None,
        }
    }
}

/// Below this many instances to reassign, LOD assignment runs on the
/// calling thread instead of all cores.
const LOD_PARALLEL_PLACES: usize = 8192;
/// The near sun cascade every frame, the other two on alternate frames.
pub const DEFAULT_SHADOW_CASCADE_REFRESH: [u32; 3] = [1, 2, 2];
/// From this many animated characters their clocks tick on the frame's
/// threads (a clock alone is a few nanoseconds).
const ANIM_PARALLEL_CLOCKS: usize = 512;
/// From this many characters their poses are sampled on the frame's
/// threads (one pose is tens of microseconds; waking threads costs a few).
pub(crate) const ANIM_PARALLEL_POSES: usize = 4;

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limits(mut self, limits: EngineLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn ribbons_mut(&mut self) -> &mut RibbonWorld {
        &mut self.ribbons
    }
    pub(crate) fn ribbon_frame(&self) -> Vec<crate::ribbons::RibbonSnapshot> {
        self.ribbons.snapshots()
    }

    pub fn particles_mut(&mut self) -> &mut ParticleWorld {
        &mut self.particles
    }
    pub(crate) fn particles_frame(&mut self) -> (Vec<crate::particles::ParticleCommand>, f32, f32) {
        self.particles.drain()
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn light(&self) -> &Light {
        &self.light
    }

    pub fn clear_color(&self) -> Color {
        self.clear_color
    }

    pub fn set_camera_lens(&mut self, fov_y_degrees: f32, near: f32) {
        self.camera = self
            .camera
            .clone()
            .with_lens(fov_y_degrees, near, self.view_distance);
    }

    pub fn set_sun_color(&mut self, color: Vec3) {
        self.light.color = color;
    }

    pub fn limits(&self) -> &EngineLimits {
        &self.limits
    }

    fn bump_render_epoch(&mut self) {
        self.render_epoch = self
            .render_epoch
            .checked_add(1)
            .expect("world render epoch overflow");
    }

    fn bump_resource_epoch(&mut self) {
        self.resource_epoch = self
            .resource_epoch
            .checked_add(1)
            .expect("world resource epoch overflow");
    }

    pub(crate) fn render_epoch(&self) -> u64 {
        self.render_epoch
    }

    pub(crate) fn instance_epoch(&self) -> u64 {
        self.instance_epoch
    }
}

/// One timed slice of a frame, kept until the hitch log prints or discards it.
#[derive(Clone, Debug)]
pub struct HitchSpan {
    pub name: String,
    pub ms: f32,
    pub detail: String,
}

/// Per-frame timing and input passed to the update closure.
#[derive(Clone, Debug)]
pub struct Frame {
    pub dt: f32,
    pub time: f32,
    /// Smoothed frames per second (updated about twice a second).
    pub fps: f32,
    pub width: u32,
    pub height: u32,
    pub aspect: f32,
    /// True only on the first update after the window is ready.
    pub first: bool,
    /// Keys held this frame (cleared while UI wants keyboard/pointer).
    pub input: Input,
    /// Immediate-mode UI (modals, buttons, images).
    pub ui: UiFrame,
    /// Fixed simulation steps due this frame (see [`World::set_sim_rate`]).
    pub sim: crate::sim::SimSteps,
}

#[cfg(test)]
mod bloom_tests {
    use super::*;

    #[test]
    fn bloom_settings_reject_invalid_values() {
        assert!(BloomSettings::default().with_threshold(f32::NAN).is_err());
        assert!(BloomSettings::default().with_soft_knee(1.1).is_err());
        assert!(BloomSettings::default().with_intensity(-0.01).is_err());
        assert!(BloomSettings::default().with_exposure(0.0).is_err());
    }

    #[test]
    fn content_hash_tells_maps_apart() {
        let a = vec![7u8; 1 << 20];
        let mut b = a.clone();
        assert_eq!(content_hash(&a), content_hash(&b));
        b[123_457] = 8;
        assert_ne!(content_hash(&a), content_hash(&b));
        assert_ne!(content_hash(&a[..1000]), content_hash(&a[..1001]));
    }

    #[test]
    fn bloom_can_be_explicitly_disabled() {
        let settings = BloomSettings::disabled();
        assert!(!settings.enabled());
        assert_eq!(settings.intensity(), BloomSettings::default().intensity());
    }
}

#[cfg(test)]
mod vsync_tests {
    use super::*;

    #[test]
    fn vsync_defaults_on_and_tracks_the_preference() {
        let mut world = World::default();
        assert!(world.vsync(), "engine default must be synchronized");
        world.set_vsync(false);
        assert!(!world.vsync());
        world.set_vsync(true);
        assert!(world.vsync());
    }
}

/// Meshes the occlusion pass tested in a frame and how many were hidden.
/// The answer comes back from the GPU a frame or two late.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcclusionStats {
    pub tested: u32,
    pub hidden: u32,
}

/// When occlusion culling runs ([`World::set_occlusion_mode`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OcclusionMode {
    /// Never.
    Off,
    /// Every frame.
    On,
    /// Where it pays: tried whenever enough meshes are in view, kept while
    /// it hides a good share of them.
    #[default]
    Auto,
}

/// `ENGINE_OCCLUSION=0|1|auto` overrides [`World::set_occlusion_mode`].
fn occlusion_env() -> Option<OcclusionMode> {
    static ENV: std::sync::OnceLock<Option<OcclusionMode>> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| match std::env::var("ENGINE_OCCLUSION").as_deref() {
        Ok("0") => Some(OcclusionMode::Off),
        Ok("1") => Some(OcclusionMode::On),
        Ok("auto") => Some(OcclusionMode::Auto),
        Ok(other) => panic!("ENGINE_OCCLUSION must be 0, 1 or auto, got {other:?}"),
        Err(_) => None,
    })
}
