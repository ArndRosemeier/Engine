use crate::anim::{AnimatedModel, AnimationAction, AnimationProfile, Animator, Locomotion};
use crate::camera::Camera;
use crate::collision::{ActorBody, ActorMove, CollisionWorld};
use crate::color::Color;
use crate::contact::ContactSnapshot;
use crate::error::{EngineError, EngineResult};
use crate::input::Input;
use crate::limits::EngineLimits;
use crate::lod::{LodId, LodLevel, LodSet};
use crate::mesh::{AlbedoMap, BuiltMesh, MaterialMaps, Mesh};
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
use glam::{IVec3, Mat4, Vec3};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[path = "save.rs"]
mod save;
pub use save::Loaded;

#[path = "nav.rs"]
mod nav;

#[path = "world_buildings.rs"]
mod world_buildings;
pub use nav::{NavAgent, NavSettings, NavStats, NavStatus};

/// Opaque handle to a spawned entity. Saves can hold it; after loading,
/// map it to the new one with [`Loaded::entity`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
fn content_hash(bytes: &[u8]) -> u64 {
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
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub(crate) mesh: BuiltMesh,
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
    pub fn mesh(&self) -> &BuiltMesh {
        &self.mesh
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
    /// Changes when only instance lists changed (LOD swaps): the renderer
    /// re-uploads instances without rebuilding GI or shadow sources.
    instance_epoch: u64,
    lods: std::collections::BTreeMap<LodId, LodSet>,
    next_lod: u64,
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
    wind: f32,
    season: f32,
    /// Albedo textures by content, so meshes that share an atlas (every
    /// generated tree) share one GPU texture.
    texture_by_content: HashMap<u64, TextureId>,
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
    /// Instanced mesh submit path. Default stays CPU so Engine demos match today.
    instance_submit: InstanceSubmit,
    bloom: BloomSettings,
    ambient_occlusion: Option<AmbientOcclusion>,
    global_illumination: Option<GlobalIllumination>,
    /// Smooth destructible terrain, meshed by the renderer on the GPU.
    voxel_terrain: Option<crate::voxel_terrain::VoxelTerrain>,
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
    /// Surface presentation sync. Engine default stays on so demos never tear;
    /// the renderer applies it at every present.
    vsync: bool,
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
    ribbons: RibbonWorld,
    quality: crate::quality::Quality,
    taa: bool,
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
            instance_epoch: 1,
            lods: std::collections::BTreeMap::new(),
            next_lod: 1,
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
            wind: 0.5,
            season: 0.0,
            texture_by_content: HashMap::new(),
            torch: None,
            lamps: std::collections::BTreeMap::new(),
            next_lamp: 1,
            lamp_shadow_budget: 4,
            hierarchy: Default::default(),
            prefabs: HashMap::new(),
            prefab_instances: HashMap::new(),
            nav: Default::default(),
            shadows: Some(ShadowSettings::default()),
            instance_submit: InstanceSubmit::from_env(),
            bloom: BloomSettings::default(),
            ambient_occlusion: Some(AmbientOcclusion::default()),
            global_illumination: Some(GlobalIllumination::default()),
            voxel_terrain: None,
            voxel_baseline_rev: 0,
            water: None,
            water_terrain_rev: 0,
            physics: crate::physics::Physics::new(),
            sim_clock: None,
            sim_tick: 0,
            sim_alpha: 1.0,
            sim_poses: HashMap::new(),
            buildings: HashMap::new(),
            element_states: Default::default(),
            falling: Vec::new(),
            rubble: Default::default(),
            vsync: true,
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
            ribbons: RibbonWorld::default(),
            quality: crate::quality::Quality::High,
            taa: true,
            tone_map: ToneMap::default(),
            grass: 1.0,
            auto_tune: None,
        }
    }
}

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

    /// Spawn a LOD set: one placement list drawn with `levels` (detailed
    /// first), each instance at the level its camera distance falls in, with
    /// a dithered cross-fade between levels. See [`crate::lod`].
    pub fn spawn_lod(&mut self, levels: Vec<LodLevel>) -> EngineResult<LodId> {
        LodSet::validate_levels(&levels)?;
        // Building GPU-ready vertices is the slow part: do the levels at once.
        use rayon::prelude::*;
        let built: Vec<_> = levels
            .into_par_iter()
            .map(|level| {
                let albedo = (
                    level.mesh.albedo().cloned(),
                    level.mesh.material_maps().cloned(),
                );
                (level.mesh.build(), albedo, level.max_distance)
            })
            .collect();
        let mut ids = Vec::with_capacity(built.len());
        for (mesh, albedo, max_distance) in built {
            let id = self.spawn_built_instanced(mesh, Vec::new());
            self.attach_albedo(id, albedo)?;
            ids.push((id, max_distance));
        }
        let id = LodId(self.next_lod);
        self.next_lod += 1;
        self.lods.insert(
            id,
            LodSet {
                levels: ids,
                places: Vec::new(),
                assigned_for: None,
            },
        );
        Ok(id)
    }

    /// [`Self::spawn_lod`] and [`Self::set_lod_places`] in one call: the
    /// usual way to put a tree, rock or prop model into the world.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World) -> EngineResult<()> {
    /// let oak = tree_model(TreeSpecies::Oak, 0)?;
    /// let places = Scatter::new(1).rect(-50.0, -50.0, 50.0, 50.0).spacing(9.0).places()?;
    /// world.spawn_lod_at(oak.lod_levels(), &places)?;
    /// # Ok(()) }
    /// ```
    pub fn spawn_lod_at(&mut self, levels: Vec<LodLevel>, places: &[Place]) -> EngineResult<LodId> {
        let id = self.spawn_lod(levels)?;
        if let Err(e) = self.set_lod_places(id, places) {
            self.despawn_lod(id);
            return Err(e);
        }
        Ok(id)
    }

    /// Where a LOD set's instances stand.
    pub fn set_lod_places(&mut self, id: LodId, places: &[Place]) -> EngineResult<()> {
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "set_lod_places has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let set = self.lods.get_mut(&id).ok_or(EngineError::UnknownEntity)?;
        set.places = crate::lod::instances_from(places);
        set.assigned_for = None;
        Ok(())
    }

    /// The instanced entities drawing each level, detailed first (to change
    /// their shadow casting, say).
    pub fn lod_levels(&self, id: LodId) -> EngineResult<Vec<EntityId>> {
        let set = self.lods.get(&id).ok_or(EngineError::UnknownEntity)?;
        Ok(set.levels.iter().map(|(e, _)| *e).collect())
    }

    pub fn despawn_lod(&mut self, id: LodId) {
        if let Some(set) = self.lods.remove(&id) {
            for (entity, _) in set.levels {
                self.despawn(entity);
            }
        }
    }

    /// Scales every LOD distance: above 1 keeps detail further out, below 1
    /// trades it for speed. The quality knob for automatic tuning.
    pub fn set_lod_bias(&mut self, bias: f32) -> EngineResult<()> {
        if !bias.is_finite() || bias <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "LOD bias {bias} must be positive"
            )));
        }
        self.lod_bias = bias;
        Ok(())
    }

    pub fn lod_bias(&self) -> f32 {
        self.lod_bias
    }

    /// Apply a quality preset: shadows, ambient occlusion, GI and LOD bias.
    /// A new world starts at [`Quality::High`](crate::quality::Quality::High).
    /// Settings changed afterwards (for example `set_ambient_occlusion`) win
    /// until the next call.
    pub fn set_quality(&mut self, quality: crate::quality::Quality) {
        let s = quality.settings();
        self.quality = quality;
        self.shadows = s.shadows;
        self.ambient_occlusion = s.ambient_occlusion;
        self.global_illumination = s.global_illumination;
        self.lod_bias = s.lod_bias;
        self.grass = s.grass;
        if let Some(t) = &mut self.auto_tune {
            t.max_bias = s.lod_bias;
        }
    }

    /// How far grass reaches, as a factor on the terrain's
    /// [`crate::proc_terrain::Grass::distance`] (0 hides it). Quality presets
    /// set it; the LOD bias (and so the frame-rate tuner) scales it further.
    pub fn set_grass_scale(&mut self, scale: f32) -> EngineResult<()> {
        if !scale.is_finite() || !(0.0..=4.0).contains(&scale) {
            return Err(EngineError::InvalidValue(format!(
                "grass scale {scale} must be within 0..=4"
            )));
        }
        self.grass = scale;
        Ok(())
    }

    /// The grass distance factor actually used: the scale times the LOD bias.
    pub fn grass_scale(&self) -> f32 {
        self.grass * self.lod_bias
    }

    /// Temporal anti-aliasing (on by default): smooths edges, shimmer and
    /// the LOD cross-fade by blending each frame with the last ones.
    /// `ENGINE_TAA=0` also switches it off. TAA is the engine's main
    /// anti-aliasing (MSAA defaults to off while it runs): turning it off in
    /// code leaves edges jagged unless the app starts with `ENGINE_MSAA=4`.
    pub fn set_taa(&mut self, on: bool) {
        self.taa = on;
    }

    pub fn taa(&self) -> bool {
        self.taa
    }

    /// The last preset given to [`Self::set_quality`].
    pub fn quality(&self) -> crate::quality::Quality {
        self.quality
    }

    /// Keep the frame rate at or above `fps` by trading detail for speed,
    /// within the current [`Self::set_quality`] preset: LOD distances first,
    /// then GI, then AO. `None` stops tuning and leaves things as they are.
    /// See [`crate::quality`].
    pub fn set_target_fps(&mut self, fps: Option<f32>) -> EngineResult<()> {
        match fps {
            None => self.auto_tune = None,
            Some(f) if f.is_finite() && (1.0..=1000.0).contains(&f) => {
                let max_bias = self.quality.settings().lod_bias;
                self.auto_tune = Some(crate::quality::AutoTune::new(f, max_bias));
            }
            Some(f) => {
                return Err(EngineError::InvalidValue(format!(
                    "target fps {f} must be within 1..=1000"
                )))
            }
        }
        Ok(())
    }

    pub fn target_fps(&self) -> Option<f32> {
        self.auto_tune.as_ref().map(|t| t.target_fps)
    }

    /// Runs particles and ribbons by the frame time, unless the game drives
    /// them itself (see [`ParticleWorld::advance`]).
    pub(crate) fn tick_effects(&mut self, dt: f32) {
        self.particles.auto_advance(dt);
        self.ribbons.auto_advance(dt);
    }

    pub(crate) fn tick_auto_quality(&mut self, dt: f32) {
        use crate::quality::TuneStep;
        let (gi, ao) = (
            self.global_illumination.is_some(),
            self.ambient_occlusion.is_some(),
        );
        let Some(t) = &mut self.auto_tune else {
            return;
        };
        match t.tick(dt, self.lod_bias, gi, ao) {
            TuneStep::Hold => {}
            TuneStep::Bias(b) => self.lod_bias = b,
            TuneStep::DropGi => self.global_illumination = None,
            TuneStep::DropAo => self.ambient_occlusion = None,
        }
    }

    /// Width of the cross-fade band as a fraction of each level's end
    /// distance (default 0.1; 0 swaps without fading).
    pub fn set_lod_fade(&mut self, fraction: f32) -> EngineResult<()> {
        if !(0.0..=0.5).contains(&fraction) {
            return Err(EngineError::InvalidValue(format!(
                "LOD fade {fraction} must be within 0..=0.5"
            )));
        }
        self.lod_fade = fraction;
        for set in self.lods.values_mut() {
            set.assigned_for = None;
        }
        Ok(())
    }

    /// Reassign LOD levels for the current camera. Runs once per frame
    /// before rendering; cheap while the camera stands still.
    pub(crate) fn tick_lods(&mut self) {
        if self.lods.is_empty() {
            return;
        }
        let eye = self.camera.eye();
        let (bias, fade) = (self.lod_bias, self.lod_fade);
        let mut updates = Vec::new();
        for set in self.lods.values_mut() {
            let stale = match set.assigned_for {
                None => true,
                Some((at, b)) => b != bias || at.distance(eye) >= set.move_threshold(bias, fade),
            };
            if !stale {
                continue;
            }
            set.assigned_for = Some((eye, bias));
            let ends: Vec<f32> = set.levels.iter().map(|(_, d)| *d).collect();
            let lists = crate::lod::assign(&set.places, &ends, eye, bias, fade);
            for ((entity, _), list) in set.levels.iter().zip(lists) {
                updates.push((*entity, list));
            }
        }
        for (entity, list) in updates {
            self.replace_instances(entity, list);
        }
    }

    /// Swap an instanced entity's list. Only an empty/non-empty change needs
    /// a scene resync (draw and shadow source lists skip empty entities).
    fn replace_instances(&mut self, id: EntityId, instances: Vec<MeshInstance>) {
        let Some(e) = self.entities.get_mut(&id) else {
            return;
        };
        let was_empty = e.instances.is_empty();
        let now_empty = instances.is_empty();
        if was_empty && now_empty {
            return;
        }
        e.instances = instances;
        e.bump_xform();
        if was_empty != now_empty {
            self.bump_render_epoch();
        } else {
            self.instance_epoch = self
                .instance_epoch
                .checked_add(1)
                .expect("world instance epoch overflow");
        }
    }

    pub(crate) fn resource_epoch(&self) -> u64 {
        self.resource_epoch
    }

    /// Record a named cost on this frame. Printed only if the frame hitches.
    pub fn hitch_span(&mut self, name: impl Into<String>, ms: f32, detail: impl Into<String>) {
        if !ms.is_finite() || ms < 0.0 {
            panic!("hitch span ms must be finite and >= 0, got {ms}");
        }
        self.hitch_spans.push(HitchSpan {
            name: name.into(),
            ms,
            detail: detail.into(),
        });
    }

    pub(crate) fn take_hitch_spans(&mut self) -> Vec<HitchSpan> {
        std::mem::take(&mut self.hitch_spans)
    }

    /// Append hitch reports to `path`, or pass `None` to stop logging.
    pub fn set_hitch_log(&mut self, path: Option<PathBuf>) {
        if let Some(p) = &path {
            if p.as_os_str().is_empty() {
                panic!("hitch log path must not be empty");
            }
        }
        self.hitch_log = path;
    }

    pub fn hitch_log(&self) -> Option<&Path> {
        self.hitch_log.as_deref()
    }

    /// Static obstacles actors slide against.
    pub fn collision(&self) -> &CollisionWorld {
        &self.collision
    }

    pub fn collision_mut(&mut self) -> &mut CollisionWorld {
        &mut self.collision
    }

    /// Slide `from` by `(dx, dz)` using `body`. Collision-off bodies translate.
    /// Only colliders in [`Self::living_in`] that overlap the actor vertically
    /// participate.
    pub fn move_actor(&self, body: &ActorBody, feet: GlobalPosition, dx: f64, dz: f64) -> GlobalXZ {
        self.collision
            .move_in(self.live_space, body, feet.horizontal(), dx, dz, feet.y)
    }

    /// Sweep a vertical actor capsule through finite walls, floors, and ceilings.
    ///
    /// `from` is the actor's feet. The caller owns gravity and velocity.
    pub fn move_actor_3d(
        &self,
        body: &ActorBody,
        from: GlobalPosition,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> ActorMove {
        self.collision
            .move_3d_in(self.live_space, body, from, dx, dy, dz)
    }

    /// Intern a disconnected space. Does not change where new entities spawn;
    /// call [`Self::in_space`] or [`Self::spawn_in`] for that.
    pub fn space(&mut self, name: impl Into<String>) -> EngineResult<SpaceId> {
        let name = name.into();
        if name.is_empty() {
            return Err(EngineError::InvalidValue(
                "space name must not be empty".into(),
            ));
        }
        if let Some(&id) = self.space_by_name.get(&name) {
            return Ok(id);
        }
        let id = SpaceId::from_raw(self.next_space_id);
        self.next_space_id = self
            .next_space_id
            .checked_add(1)
            .expect("space id overflow");
        self.space_by_name.insert(name, id);
        Ok(id)
    }

    /// Subsequent [`Self::spawn`] / [`Self::place`] calls go into `space`.
    pub fn in_space(&mut self, space: SpaceId) -> EngineResult<()> {
        self.require_space(space)?;
        self.spawn_space = space;
        Ok(())
    }

    /// Which space the main camera is standing in.
    pub fn live_in(&mut self, space: SpaceId) -> EngineResult<()> {
        self.require_space(space)?;
        if self.live_space != space {
            self.live_space = space;
            self.travel_prev = None;
        }
        Ok(())
    }

    pub fn living_in(&self) -> SpaceId {
        self.live_space
    }

    pub fn spawning_in(&self) -> SpaceId {
        self.spawn_space
    }

    /// Sky and clipmap belong to the default outdoor space, and to any named
    /// space that opted in with [`Self::set_space_draws_environment`].
    pub fn space_draws_environment(&self, space: SpaceId) -> bool {
        space == SpaceId::DEFAULT || self.environment_spaces.contains(&space)
    }

    /// Let a named space draw the outdoor sky and clipmap, or take that back.
    ///
    /// The default space always draws the environment. Asking to disable it is
    /// a caller bug, not a silent no-op.
    pub fn set_space_draws_environment(&mut self, space: SpaceId, draws: bool) -> EngineResult<()> {
        self.require_space(space)?;
        if space == SpaceId::DEFAULT {
            if draws {
                return Ok(());
            }
            return Err(EngineError::InvalidValue(
                "the default space always draws the outdoor environment".into(),
            ));
        }
        if draws {
            self.environment_spaces.insert(space);
        } else {
            self.environment_spaces.remove(&space);
        }
        Ok(())
    }

    fn require_space(&self, space: SpaceId) -> EngineResult<()> {
        if space == SpaceId::DEFAULT || self.space_by_name.values().any(|&id| id == space) {
            return Ok(());
        }
        Err(EngineError::InvalidValue(format!(
            "unknown space {}",
            space.raw()
        )))
    }

    /// Spawn every part of a model (one entity per material) at `place`.
    /// Move them together by giving each the same place.
    pub fn spawn_model(
        &mut self,
        model: &crate::model::ModelScene,
        place: Place,
    ) -> EngineResult<Vec<EntityId>> {
        model
            .parts
            .iter()
            .map(|part| self.place(part.clone(), place))
            .collect()
    }

    /// Spawn a mesh at the origin.
    pub fn spawn(&mut self, mesh: Mesh) -> EntityId {
        self.place(mesh, Place::default())
            .expect("spawn at identity")
    }

    pub fn spawn_in(&mut self, space: SpaceId, mesh: Mesh) -> EngineResult<EntityId> {
        self.require_space(space)?;
        let prev = self.spawn_space;
        self.spawn_space = space;
        let id = self.spawn(mesh);
        self.spawn_space = prev;
        Ok(id)
    }

    pub fn place_in(&mut self, space: SpaceId, mesh: Mesh, place: Place) -> EngineResult<EntityId> {
        self.require_space(space)?;
        let prev = self.spawn_space;
        self.spawn_space = space;
        let id = self.place(mesh, place);
        self.spawn_space = prev;
        id
    }

    pub fn set_space(&mut self, id: EntityId, space: SpaceId) -> EngineResult<()> {
        self.require_space(space)?;
        if let Some(e) = self.entities.get_mut(&id) {
            if e.space != space {
                e.space = space;
                self.bump_render_epoch();
            }
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.space = space;
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
    }

    /// Two openings become the same portal. Both meshes should face their own
    /// space (`+Z` local). Looking through shows the other space; walking
    /// through teleports when [`PortalSettings::teleport`] is true.
    pub fn create_portal(
        &mut self,
        a: EntityId,
        b: EntityId,
        settings: PortalSettings,
    ) -> EngineResult<PortalId> {
        self.validate_portal_sides(a, b)?;
        let id = PortalId::from_raw(self.next_portal_id);
        self.next_portal_id = self
            .next_portal_id
            .checked_add(1)
            .expect("portal id overflow");
        self.portals.push(Portal {
            id,
            sides: [a, b],
            teleport: settings.teleport,
            enabled: true,
        });
        self.bump_render_epoch();
        Ok(id)
    }

    /// Toggle whether [`Self::travel`] crosses this portal.
    pub fn set_portal_teleport(&mut self, id: PortalId, teleport: bool) -> EngineResult<()> {
        let portal = self
            .portals
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| EngineError::InvalidValue(format!("unknown portal {id:?}")))?;
        portal.teleport = teleport;
        Ok(())
    }

    /// Remove a portal by id. The opening entities stay in the world.
    pub fn destroy_portal(&mut self, id: PortalId) -> EngineResult<()> {
        let before = self.portals.len();
        self.portals.retain(|p| p.id != id);
        if self.portals.len() == before {
            return Err(EngineError::InvalidValue(format!("unknown portal {id:?}")));
        }
        self.bump_render_epoch();
        Ok(())
    }

    pub fn set_portal_enabled(&mut self, id: PortalId, enabled: bool) -> EngineResult<()> {
        let portal = self
            .portals
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| EngineError::InvalidValue(format!("unknown portal {id:?}")))?;
        portal.set_enabled(enabled);
        self.bump_render_epoch();
        Ok(())
    }

    pub fn portal_recursion(&self) -> u8 {
        self.portal_recursion
    }

    pub fn set_portal_recursion(&mut self, depth: u8) {
        if depth == 0 {
            panic!("portal recursion depth must be at least 1");
        }
        self.portal_recursion = depth;
    }

    fn validate_portal_sides(&self, a: EntityId, b: EntityId) -> EngineResult<()> {
        if a == b {
            return Err(EngineError::InvalidValue(
                "a portal cannot link an opening to itself".into(),
            ));
        }
        let plane_a = self.portal_plane(a)?;
        let plane_b = self.portal_plane(b)?;
        if plane_a.half_width <= 0.0 || plane_a.half_height <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "entity {a} has no XY extent to use as an opening"
            )));
        }
        if plane_b.half_width <= 0.0 || plane_b.half_height <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "entity {b} has no XY extent to use as an opening"
            )));
        }
        if self.is_linked_opening(a) || self.is_linked_opening(b) {
            return Err(EngineError::InvalidValue(
                "opening is already part of a portal".into(),
            ));
        }
        Ok(())
    }

    pub fn portals(&self) -> &[Portal] {
        &self.portals
    }

    /// All openings in `live` the camera is facing, nearest first.
    pub(crate) fn visible_portals(
        &self,
        eye: Vec3,
        look: Vec3,
        live: SpaceId,
    ) -> EngineResult<Vec<VisiblePortal>> {
        let look = {
            if look.length_squared() <= 0.0 {
                panic!("visible_portals look direction is zero");
            }
            look.normalize()
        };
        let mut found = Vec::new();
        for portal in &self.portals {
            if !portal.is_enabled() {
                continue;
            }
            let (Some(a), Some(b)) = (
                self.entities.get(&portal.a()),
                self.entities.get(&portal.b()),
            ) else {
                return Err(EngineError::InvalidValue(format!(
                    "enabled portal links missing entity {} or {}",
                    portal.a(),
                    portal.b()
                )));
            };
            for (src, dst) in portal.directions(a.space, b.space, live) {
                let src_e = if src == portal.a() { a } else { b };
                let dst_e = if dst == portal.a() { a } else { b };
                let src_plane = self.portal_plane(src).map_err(|error| {
                    EngineError::InvalidValue(format!(
                        "visible portal source {src} has no valid plane: {error}"
                    ))
                })?;
                if src_plane.signed_distance(eye) <= -crate::portal::PORTAL_VISIBILITY_PLANE_EPS {
                    continue;
                }
                if look.dot(-src_plane.normal) <= 0.0 {
                    continue;
                }
                let dist = eye.distance_squared(src_plane.center);
                let dst_plane = self.portal_plane(dst).map_err(|error| {
                    EngineError::InvalidValue(format!(
                        "visible portal destination {dst} has no valid plane: {error}"
                    ))
                })?;
                found.push((
                    dist,
                    VisiblePortal {
                        src,
                        dst,
                        src_transform: src_e.transform,
                        dst_transform: dst_e.transform,
                        dst_center: dst_plane.center,
                        dst_normal: dst_plane.normal,
                        dest_space: dst_e.space,
                    },
                ));
            }
        }
        found.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("finite portal planes produce finite distances")
        });
        Ok(found.into_iter().map(|(_, p)| p).collect())
    }

    /// Closest opening in the live space that the camera is facing.
    #[allow(dead_code)]
    pub(crate) fn visible_portal(
        &self,
        eye: Vec3,
        look: Vec3,
    ) -> EngineResult<Option<VisiblePortal>> {
        self.visible_portals(eye, look, self.live_space)
            .map(|portals| portals.into_iter().next())
    }

    /// Whether this entity is portal stencil geometry, enabled or disabled.
    /// Disabled portals must remain excluded from ordinary mesh rendering.
    pub fn is_portal_surface(&self, id: EntityId) -> bool {
        self.portals.iter().any(|p| p.a() == id || p.b() == id)
    }

    pub(crate) fn is_linked_opening(&self, id: EntityId) -> bool {
        self.portals.iter().any(|p| p.a() == id || p.b() == id)
    }

    pub(crate) fn portal_plane(&self, id: EntityId) -> EngineResult<PortalPlane> {
        let e = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if e.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} is instanced; openings must be a single mesh"
            )));
        }
        let (half_w, half_h) = opening_extents(&e.mesh);
        Ok(PortalPlane::from_transform(e.transform, half_w, half_h))
    }

    /// If `position` crossed an opening this frame, move it to the other side
    /// and switch [`Self::living_in`]. Call before the camera helper.
    ///
    /// Returns the space you entered.
    pub fn travel(&mut self, position: &mut Vec3, yaw_degrees: &mut f32) -> Option<SpaceId> {
        if !position.is_finite() {
            panic!("travel position must be finite, got {position}");
        }
        if !yaw_degrees.is_finite() {
            panic!("travel yaw must be finite, got {yaw_degrees}");
        }
        let live = self.live_space;
        let prev = match self.travel_prev {
            Some((space, p)) if space == live => p,
            _ => {
                self.travel_prev = Some((live, *position));
                return None;
            }
        };
        let links: Vec<Portal> = self.portals.clone();
        for portal in links {
            if !portal.is_enabled() || !portal.teleport {
                continue;
            }
            let a_space = self.entities.get(&portal.a()).map(|e| e.space);
            let b_space = self.entities.get(&portal.b()).map(|e| e.space);
            let (Some(a_space), Some(b_space)) = (a_space, b_space) else {
                continue;
            };
            for (src, dst) in portal.directions(a_space, b_space, live) {
                if self.travel_gate == Some(dst) {
                    continue;
                }
                let src_plane = self
                    .portal_plane(src)
                    .expect("linked opening still has a plane");
                if !segment_crosses_opening(prev, *position, src_plane) {
                    continue;
                }
                let dst_e = self
                    .entities
                    .get(&dst)
                    .expect("linked opening still exists");
                let t_src = src_plane.transform;
                let t_dst = dst_e.transform;
                *position = crate::portal::portal_matrix(t_src, t_dst).transform_point3(*position);
                if dst_e.space != live {
                    let dst_plane = self
                        .portal_plane(dst)
                        .expect("linked opening still has a plane");
                    *position += dst_plane.normal * crate::portal::PORTAL_TELEPORT_SETBACK;
                }
                *yaw_degrees = teleport_yaw(*yaw_degrees, t_src, t_dst);
                self.live_space = dst_e.space;
                self.travel_prev = Some((self.live_space, *position));
                self.travel_gate = Some(dst);
                return Some(self.live_space);
            }
        }
        self.travel_gate = None;
        self.travel_prev = Some((live, *position));
        None
    }

    /// Spawn a mesh at a friendly [`Place`].
    pub fn place(&mut self, mesh: Mesh, place: Place) -> EngineResult<EntityId> {
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built(mesh.build(), place.to_matrix());
        self.attach_albedo(id, albedo)?;
        Ok(id)
    }

    /// Spawn one mesh at many places (GPU instancing).
    pub fn spawn_many(
        &mut self,
        mesh: Mesh,
        places: impl Into<Vec<Place>>,
    ) -> EngineResult<EntityId> {
        let places = places.into();
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "spawn_many has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let instances: Vec<MeshInstance> = places.into_iter().map(Place::to_instance).collect();
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built_instanced(mesh.build(), instances);
        self.attach_albedo(id, albedo)?;
        Ok(id)
    }

    /// Spawn an instanced entity that starts with nothing placed.
    ///
    /// The mesh is uploaded once; [`Self::set_instances`] then drives where it
    /// appears.
    pub fn spawn_instanced(&mut self, mesh: Mesh) -> EntityId {
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built_instanced(mesh.build(), Vec::new());
        self.attach_albedo(id, albedo)
            .expect("albedo upload at spawn_instanced");
        id
    }

    /// Another instanced entity that draws `prototype`'s mesh and albedo.
    ///
    /// The prototype must itself own a mesh (not be a like-entity). This one
    /// starts with no placements; [`Self::set_instances`] fills them.
    pub fn spawn_instanced_like(&mut self, prototype: EntityId) -> EngineResult<EntityId> {
        let src = self
            .entities
            .get(&prototype)
            .ok_or(EngineError::UnknownEntity)?;
        if !src.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {prototype} was not spawned instanced"
            )));
        }
        if src.instance_of.is_some() {
            return Err(EngineError::InvalidValue(format!(
                "entity {prototype} is already a like-entity; instance the prototype"
            )));
        }
        let albedo = src.albedo;
        let material = src.material;
        let casts_shadow = src.casts_shadow;
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh: BuiltMesh::default(),
                transform: Mat4::IDENTITY,
                instances: Vec::new(),
                instance_reserve: 0,
                instanced: true,
                material,
                albedo,
                xform_rev: 1,
                instance_of: Some(prototype),
                casts_shadow,
                space: self.spawn_space,
            },
        );
        self.order.push(id);
        self.bump_render_epoch();
        Ok(id)
    }

    pub(crate) fn spawn_built(&mut self, mesh: BuiltMesh, transform: Mat4) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh,
                transform,
                instances: Vec::new(),
                instance_reserve: 0,
                instanced: false,
                material: None,
                albedo: None,
                xform_rev: 1,
                instance_of: None,
                casts_shadow: true,
                space: self.spawn_space,
            },
        );
        self.order.push(id);
        self.bump_render_epoch();
        id
    }

    pub(crate) fn spawn_built_instanced(
        &mut self,
        mesh: BuiltMesh,
        instances: Vec<MeshInstance>,
    ) -> EntityId {
        let instance_reserve = instances.len();
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh,
                transform: Mat4::IDENTITY,
                instances,
                instance_reserve,
                instanced: true,
                material: None,
                albedo: None,
                xform_rev: 1,
                instance_of: None,
                casts_shadow: true,
                space: self.spawn_space,
            },
        );
        self.order.push(id);
        self.bump_render_epoch();
        id
    }

    /// Grass and far stand-ins should not enter the cascaded depth maps.
    pub fn set_casts_shadow(&mut self, id: EntityId, casts: bool) -> EngineResult<()> {
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if e.casts_shadow == casts {
            return Ok(());
        }
        e.casts_shadow = casts;
        self.bump_render_epoch();
        Ok(())
    }

    fn attach_albedo(
        &mut self,
        id: EntityId,
        (albedo, maps): (Option<AlbedoMap>, Option<MaterialMaps>),
    ) -> EngineResult<()> {
        let map = match (albedo, &maps) {
            (Some(map), _) => map,
            // Maps need an albedo to ride along with: a white texel.
            (None, Some(_)) => AlbedoMap {
                width: 1,
                height: 1,
                rgba: vec![255; 4],
            },
            (None, None) => return Ok(()),
        };
        // Large maps are looked up by content first: generated assets (every
        // tree) carry the same atlas and should share one GPU texture.
        let key = (map.rgba.len() >= 1 << 20).then(|| {
            let mut h = content_hash(&map.rgba) ^ ((map.width as u64) << 32 | map.height as u64);
            let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
            if let Some(m) = &maps {
                for map in [&m.normal, &m.orm, &m.emissive] {
                    mix(map.as_ref().map_or(1, |n| content_hash(&n.rgba)));
                }
                mix(m.normal_scale.to_bits() as u64);
                mix(m.roughness.to_bits() as u64);
                mix(m.metallic.to_bits() as u64);
                mix(m.triplanar.to_bits() as u64);
            }
            h
        });
        let shared = key
            .and_then(|k| self.texture_by_content.get(&k).copied())
            .filter(|tid| self.textures.contains_key(tid));
        let tid = match shared {
            Some(tid) => tid,
            None => {
                let tid = self.create_texture_rgba(map.width, map.height, map.rgba)?;
                if let Some(maps) = maps {
                    self.surface_maps.insert(tid, std::sync::Arc::new(maps));
                }
                if let Some(k) = key {
                    self.texture_by_content.insert(k, tid);
                }
                tid
            }
        };
        self.entities
            .get_mut(&id)
            .expect("entity was just spawned")
            .albedo = Some(tid);
        self.bump_render_epoch();
        Ok(())
    }

    /// Remove an entity, its children (and theirs) and the lamps riding on
    /// them. [`Self::detach`] a child first to keep it.
    pub fn despawn(&mut self, id: EntityId) {
        self.sim_poses.remove(&id);
        if self.buildings.contains_key(&id) || self.falling.iter().any(|p| p.building == id) {
            self.forget_building(id);
        }
        if let Some(made) = self.prefab_instances.remove(&id) {
            for lamp in made.lamps {
                self.remove_lamp(lamp);
            }
            for part in made.parts {
                self.despawn(part);
            }
        }
        let (children, lamps) = self.hierarchy.forget(id);
        for lamp in lamps {
            self.lamps.remove(&lamp);
        }
        for child in children {
            self.despawn(child);
        }
        self.physics.remove(id);
        let removed_static = self.entities.remove(&id).is_some();
        self.order.retain(|x| *x != id);
        self.animated.remove(&id);
        self.animated_order.retain(|x| *x != id);
        self.chunk_entities.retain(|_, eid| *eid != id);
        self.anchored_entities.remove(&id);
        self.anchored_chunks.retain(|_, c| c.entity != id);
        let linked = self.portals.iter().any(|p| p.a() == id || p.b() == id);
        self.portals.retain(|p| p.a() != id && p.b() != id);
        if removed_static || linked {
            self.bump_render_epoch();
        }
    }

    /// Horizontal anchor that render-space coordinates are measured from.
    pub fn render_origin(&self) -> RenderOrigin {
        self.render_origin
    }

    /// Rebase render space onto a new origin.
    ///
    /// Every anchored entity and chunk transform is recomputed from its
    /// immutable global anchor, so repeated rebases never accumulate drift.
    /// Unanchored content (plain [`Self::spawn`] / [`Self::set_chunk_built`])
    /// is left alone: it was authored directly in render space.
    pub fn set_render_origin(&mut self, origin: RenderOrigin) -> EngineResult<()> {
        if self.render_origin == origin {
            return Ok(());
        }
        self.render_origin = origin;
        let places: Vec<(EntityId, GlobalPlace)> = self
            .anchored_entities
            .iter()
            .map(|(id, place)| (*id, *place))
            .collect();
        for (id, place) in places {
            let local = place.to_place(origin)?;
            let e = self
                .entities
                .get_mut(&id)
                .ok_or(EngineError::UnknownEntity)?;
            e.transform = local.to_matrix();
            e.bump_xform();
        }
        let chunks: Vec<AnchoredChunk> = self.anchored_chunks.values().copied().collect();
        for chunk in chunks {
            let offset = chunk.anchor.to_render(origin)?.vec3();
            let e = self
                .entities
                .get_mut(&chunk.entity)
                .ok_or(EngineError::UnknownEntity)?;
            e.transform = Mat4::from_translation(offset);
            e.bump_xform();
        }
        self.bump_render_epoch();
        Ok(())
    }

    /// Render-space position of a global point under the active origin.
    pub fn to_render(&self, position: GlobalPosition) -> EngineResult<Vec3> {
        Ok(position.to_render(self.render_origin)?.vec3())
    }

    /// Global position of a render-space point under the active origin.
    pub fn to_global(&self, position: Vec3) -> EngineResult<GlobalPosition> {
        Ok(crate::space::RenderPosition::new(position)?.to_global(self.render_origin))
    }

    /// Spawn a mesh anchored in absolute world metres (survives rebasing).
    pub fn spawn_anchored(&mut self, mesh: Mesh, place: GlobalPlace) -> EngineResult<EntityId> {
        let local = place.to_place(self.render_origin)?;
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built(mesh.build(), local.to_matrix());
        self.attach_albedo(id, albedo)?;
        self.anchored_entities.insert(id, place);
        Ok(id)
    }

    /// Move an anchored entity to a new global place.
    pub fn set_anchored_place(&mut self, id: EntityId, place: GlobalPlace) -> EngineResult<()> {
        if !self.anchored_entities.contains_key(&id) {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} is not anchored; use set_place for render-space entities"
            )));
        }
        let local = place.to_place(self.render_origin)?;
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.transform = local.to_matrix();
        e.bump_xform();
        self.anchored_entities.insert(id, place);
        self.bump_render_epoch();
        Ok(())
    }

    /// Upload a streamed chunk whose vertices are relative to `anchor`.
    ///
    /// Chunk-local vertices keep `f32` mesh precision independent of how far the
    /// chunk sits from the world origin.
    pub fn set_anchored_chunk(
        &mut self,
        id: ChunkId,
        anchor: GlobalPosition,
        built: BuiltMesh,
    ) -> EngineResult<EntityId> {
        let offset = anchor.to_render(self.render_origin)?.vec3();
        if let Some(existing) = self.anchored_chunks.remove(&id) {
            // Remesh needs a new GPU upload ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â swap entity id so the renderer rebuilds.
            self.despawn(existing.entity);
        }
        let material = self.chunk_material(&built);
        let entity = self.spawn_built(built, Mat4::from_translation(offset));
        if material.is_some() {
            self.entities
                .get_mut(&entity)
                .expect("just spawned")
                .material = material;
        }
        self.anchored_chunks
            .insert(id, AnchoredChunk { entity, anchor });
        Ok(entity)
    }

    pub fn clear_anchored_chunk(&mut self, id: ChunkId) {
        if let Some(chunk) = self.anchored_chunks.remove(&id) {
            self.despawn(chunk.entity);
        }
    }

    pub fn has_anchored_chunk(&self, id: ChunkId) -> bool {
        self.anchored_chunks.contains_key(&id)
    }

    pub fn anchored_chunk_count(&self) -> usize {
        self.anchored_chunks.len()
    }

    /// Drop every anchored chunk (leaving a world / regenerating).
    pub fn clear_anchored_chunks(&mut self) {
        let ids: Vec<ChunkId> = self.anchored_chunks.keys().copied().collect();
        for id in ids {
            self.clear_anchored_chunk(id);
        }
    }

    /// Pin the pointer to the window and hide it, for mouse-look.
    ///
    /// While locked, look deltas arrive as [`crate::input::Input::mouse_delta`]
    /// and the UI no longer steals input on hover. Release it before showing a
    /// menu or map the player has to click.
    ///
    /// This is a request, not a guarantee: the window drops the grab when it
    /// loses focus and when Escape is pressed, and clears the flag with it, so
    /// a game that wants the pointer back has to ask again. Nothing should pin
    /// the cursor without the player having asked for it ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a grab taken on
    /// startup follows them out of the window.
    pub fn set_pointer_lock(&mut self, locked: bool) {
        self.pointer_lock = locked;
    }

    pub fn pointer_lock(&self) -> bool {
        self.pointer_lock
    }

    /// The ray under the mouse cursor, `(origin, unit direction)`, for
    /// picking, digging and placing. With the pointer locked or the cursor
    /// outside the window it is the ray through the screen centre.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World, frame: &Frame) {
    /// let (eye, dir) = world.cursor_ray(frame);
    /// if let Some(hit) = world.voxel_terrain().and_then(|t| t.raycast(eye, dir, 300.0)) {
    ///     world.voxel_terrain_mut().unwrap().dig_sphere(hit, 3.0);
    /// }
    /// # }
    /// ```
    pub fn cursor_ray(&self, frame: &Frame) -> (Vec3, Vec3) {
        let (w, h) = (frame.width as f32, frame.height as f32);
        let at = frame
            .input
            .cursor()
            .filter(|_| !self.pointer_lock)
            .unwrap_or(glam::Vec2::new(w * 0.5, h * 0.5));
        self.camera().ray_through(at.x, at.y, w, h)
    }

    /// While true, Escape cancels bind-listen only: no modal close, unlock, or quit.
    pub fn set_bind_listen(&mut self, listening: bool) {
        self.bind_listen = listening;
    }

    pub fn bind_listen(&self) -> bool {
        self.bind_listen
    }

    /// Capture `path` as a PNG after this frame has been drawn.
    pub fn queue_screenshot(&mut self, path: impl AsRef<Path>) {
        self.screenshot_queue.push(path.as_ref().to_path_buf());
    }

    /// Tell a waiting screenshot harness that the scene is ready.
    pub fn mark_ready(&mut self) {
        self.ready = true;
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    /// Close the window after this frame, once any queued shots have been written.
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    pub(crate) fn has_screenshot_queued(&self) -> bool {
        !self.screenshot_queue.is_empty()
    }

    pub(crate) fn take_screenshot_queue(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.screenshot_queue)
    }

    pub(crate) fn take_exit_requested(&mut self) -> bool {
        let wanted = self.exit_requested;
        self.exit_requested = false;
        wanted
    }

    /// First-person camera at an anchored eye position, expressed globally.
    pub fn look_first_person_global(
        &mut self,
        eye: GlobalPosition,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) -> EngineResult<()> {
        let local = self.to_render(eye)?;
        self.look_first_person(local, yaw_degrees, pitch_degrees);
        Ok(())
    }

    /// Look from one global point at another. A title shot, a cutscene, anything
    /// that is not a walker.
    pub fn look_at_global(
        &mut self,
        eye: GlobalPosition,
        target: GlobalPosition,
    ) -> EngineResult<()> {
        let e = self.to_render(eye)?;
        let t = self.to_render(target)?;
        let fov = self.camera.fov_y_degrees();
        let near = self.camera.near();
        self.camera = Camera::look_at(e, t).with_lens(fov, near, self.view_distance);
        Ok(())
    }

    /// Follow camera around an anchored target, expressed globally.
    pub fn look_follow_global(
        &mut self,
        target: GlobalPosition,
        yaw_degrees: f32,
        distance: f32,
        height: f32,
    ) -> EngineResult<()> {
        let local = self.to_render(target)?;
        self.look_follow(local, yaw_degrees, distance, height);
        Ok(())
    }

    /// Distance from the render origin to `p`, used to decide when to rebase.
    pub fn render_offset_m(&self, p: GlobalXZ) -> f64 {
        p.distance(self.render_origin.horizontal())
    }

    /// Move an entity. For a child (see [`Self::attach`]) the place is
    /// relative to its parent; its own children come along either way.
    pub fn set_place(&mut self, id: EntityId, place: Place) -> EngineResult<()> {
        self.set_matrix(id, place.to_matrix())
    }

    /// `set_place` with a ready matrix.
    fn set_matrix(&mut self, id: EntityId, m: Mat4) -> EngineResult<()> {
        if let Some(link) = self.hierarchy.links.get_mut(&id) {
            link.local = m;
            return self.update_subtree(id);
        }
        if self.hierarchy.pivots.contains_key(&id) || self.hierarchy.children.contains_key(&id) {
            self.write_matrix(id, m)?;
            return self.update_subtree(id);
        }
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform = m;
            e.bump_xform();
            self.bump_render_epoch();
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.transform = m;
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
    }

    /// Turn the fixed simulation tick on (`Some(hz)`) or off. While on, every
    /// [`Frame::sim`] says how many steps are due. See [`crate::sim`].
    pub fn set_sim_rate(&mut self, hz: Option<f32>) -> EngineResult<()> {
        self.sim_clock = match hz {
            Some(hz) => Some(crate::sim::SimClock::new(hz)?),
            None => None,
        };
        Ok(())
    }

    pub fn sim_rate(&self) -> Option<f32> {
        self.sim_clock.as_ref().map(|c| c.rate())
    }

    /// Advance the fixed tick by a frame of `dt` seconds (the engine's loop
    /// calls this before your update).
    pub(crate) fn advance_sim(&mut self, dt: f32) -> crate::sim::SimSteps {
        match self.sim_clock.as_mut() {
            Some(clock) => {
                let s = clock.advance(dt);
                self.sim_alpha = s.alpha;
                s
            }
            None => crate::sim::SimSteps {
                alpha: self.sim_alpha,
                ..Default::default()
            },
        }
    }

    /// Start a simulation step: poses set from now on belong to the new step,
    /// and the ones before become the step drawing starts from.
    pub fn begin_sim_step(&mut self) {
        self.sim_tick += 1;
    }

    /// Where an entity is after the current simulation step. It is drawn
    /// between this and its previous step's pose; an entity whose pose isn't
    /// set in a step rests where it was. The entity follows its simulation
    /// poses until [`Self::clear_sim_pose`].
    pub fn set_sim_place(&mut self, id: EntityId, place: Place) -> EngineResult<()> {
        self.set_sim_transform(id, place.to_matrix())
    }

    /// [`Self::set_sim_place`] with a matrix.
    pub fn set_sim_transform(&mut self, id: EntityId, m: Mat4) -> EngineResult<()> {
        crate::place::ensure_finite_mat(m)?;
        if !self.entities.contains_key(&id)
            && !self.animated.contains_key(&id)
            && !self.hierarchy.pivots.contains_key(&id)
        {
            return Err(EngineError::UnknownEntity);
        }
        let tick = self.sim_tick;
        let pose = self.sim_poses.entry(id).or_insert(crate::sim::SimPose {
            prev: m,
            cur: m,
            tick,
        });
        if pose.tick != tick {
            pose.prev = pose.cur;
            pose.tick = tick;
        }
        pose.cur = m;
        Ok(())
    }

    /// Both poses at once, for a client that keeps the previous and current
    /// simulation state itself: drawn between them by the frame's blend.
    pub fn set_sim_poses(
        &mut self,
        id: EntityId,
        previous: Mat4,
        current: Mat4,
    ) -> EngineResult<()> {
        crate::place::ensure_finite_mat(previous)?;
        self.set_sim_transform(id, current)?;
        if let Some(p) = self.sim_poses.get_mut(&id) {
            p.prev = previous;
        }
        Ok(())
    }

    /// Stop following simulation poses; the entity stays where it is drawn.
    pub fn clear_sim_pose(&mut self, id: EntityId) {
        self.sim_poses.remove(&id);
    }

    /// Where drawing is between the last two simulation steps (0..1).
    pub fn sim_alpha(&self) -> f32 {
        self.sim_alpha
    }

    /// Set the blend yourself, for a game that runs its own fixed clock
    /// instead of [`Self::set_sim_rate`].
    pub fn set_sim_alpha(&mut self, alpha: f32) {
        if alpha.is_finite() {
            self.sim_alpha = alpha.clamp(0.0, 1.0);
        }
    }

    /// Move entities with simulation poses to their blended pose. The engine
    /// calls this every frame.
    pub fn tick_interpolation(&mut self) {
        if self.sim_poses.is_empty() {
            return;
        }
        let (tick, alpha) = (self.sim_tick, self.sim_alpha);
        let poses: Vec<(EntityId, Mat4)> = self
            .sim_poses
            .iter()
            .map(|(id, p)| (*id, p.at(tick, alpha)))
            .collect();
        for (id, m) in poses {
            if self.set_matrix(id, m).is_err() {
                self.sim_poses.remove(&id);
            }
        }
    }

    /// Everything the windowed loop ticks that makes sense with no screen.
    pub(crate) fn tick_headless(&mut self, dt: f32) {
        self.tick_nav(dt);
        self.tick_animations(dt);
        self.tick_physics(dt);
        self.tick_buildings(dt);
        self.tick_interpolation();
        self.tick_hierarchy();
    }

    /// Element words changed since the last call (start word, words), for the renderer.
    pub(crate) fn take_element_upload(&mut self) -> Option<(usize, &[u32])> {
        let (a, b) = self.element_states.dirty.take()?;
        Some((a, &self.element_states.bits[a..b]))
    }

    /// Spawn a skinned model with clip playback.
    pub fn spawn_animated(&mut self, model: AnimatedModel, place: Place) -> EngineResult<EntityId> {
        self.spawn_animated_shared(Arc::new(model), place)
    }

    /// Spawn from a shared model so a herd does not clone the mesh per animal.
    pub fn spawn_animated_shared(
        &mut self,
        model: Arc<AnimatedModel>,
        place: Place,
    ) -> EngineResult<EntityId> {
        let animator = Animator::new(model)?;
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.animated.insert(
            id,
            AnimatedEntity {
                animator,
                transform: place.to_matrix(),
                space: self.spawn_space,
            },
        );
        self.animated_order.push(id);
        Ok(id)
    }

    pub fn configure_animation(
        &mut self,
        id: EntityId,
        profile: AnimationProfile,
    ) -> EngineResult<()> {
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .configure_profile(profile)
    }

    pub fn set_locomotion(&mut self, id: EntityId, state: Locomotion) -> EngineResult<()> {
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .set_locomotion(state)
    }

    pub fn play_animation_action(
        &mut self,
        id: EntityId,
        action: AnimationAction,
    ) -> EngineResult<()> {
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .play_action(action)
    }

    pub fn play_animation(&mut self, id: EntityId, clip_name: &str) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.play(clip_name)
    }

    /// Play once (`looping = false`) and hold the last frame when the clip ends.
    pub fn play_animation_once(&mut self, id: EntityId, clip_name: &str) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.play_once(clip_name)
    }

    pub fn set_animation_speed(&mut self, id: EntityId, speed: f32) -> EngineResult<()> {
        if !speed.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "animation speed must be finite, got {speed}"
            )));
        }
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.set_speed(speed);
        Ok(())
    }

    /// Advance all skinned clip clocks (call once per frame before render sync).
    pub fn tick_animations(&mut self, dt: f32) {
        for e in self.animated.values_mut() {
            e.animator.tick(dt);
        }
        self.tick_foot_ik(dt);
    }

    /// Cross-fade time between clips for one animated entity (default 0.2 s,
    /// 0 switches at once).
    pub fn set_animation_blend(&mut self, id: EntityId, seconds: f32) -> EngineResult<()> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "animation blend must be finite and >= 0, got {seconds}"
            )));
        }
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .set_blend_time(seconds);
        Ok(())
    }

    /// Let a character's feet find the ground (slopes, steps, rocks, other
    /// bodies), or `None` to switch it off. Finds the legs by bone name
    /// (UE, Mixamo and Blender style rigs); errors if it can't.
    pub fn set_foot_ik(
        &mut self,
        id: EntityId,
        ik: Option<crate::anim::FootIk>,
    ) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        let Some(settings) = ik else {
            e.animator.ik = None;
            return Ok(());
        };
        if !(settings.max_step.is_finite() && settings.max_step >= 0.0)
            || !(settings.response.is_finite() && settings.response > 0.0)
        {
            return Err(EngineError::InvalidValue(
                "foot IK max_step must be >= 0 and response > 0".into(),
            ));
        }
        let skeleton = &e.animator.model().skeleton;
        let legs = skeleton.find_legs().ok_or_else(|| {
            EngineError::InvalidValue(
                "no legs found (looked for foot_l/foot_r, LeftFoot/RightFoot, foot.L/foot.R)"
                    .into(),
            )
        })?;
        let hips = (skeleton.parents[legs[0].thigh] == skeleton.parents[legs[1].thigh])
            .then_some(skeleton.parents[legs[0].thigh])
            .flatten();
        let (feet, drop) = e
            .animator
            .ik
            .as_ref()
            .map_or(([Vec3::ZERO; 2], Vec3::ZERO), |old| (old.feet, old.drop));
        e.animator.ik = Some(crate::anim::IkState {
            settings,
            legs,
            hips,
            feet,
            drop,
        });
        Ok(())
    }

    fn tick_foot_ik(&mut self, dt: f32) {
        let mut updates = Vec::new();
        for (id, e) in &self.animated {
            let Some(ik) = e.animator.ik.as_ref() else {
                continue;
            };
            let t = e.transform;
            let to_model = t.inverse();
            let globals = e.animator.fk_globals();
            let root = t.w_axis.truncate();
            let ground = |x: f32, z: f32, from: f32| {
                self.cast_ray(Vec3::new(x, from + 1.0, z), Vec3::NEG_Y, 3.0)
                    .map(|h| h.point.y)
            };
            let base = ground(root.x, root.z, root.y);
            let max = ik.settings.max_step;
            let dy = ik.legs.map(|leg| {
                let foot = t.transform_point3(globals[leg.foot].w_axis.truncate());
                match (base, ground(foot.x, foot.z, root.y.max(foot.y))) {
                    (Some(b), Some(g)) => (g - b).clamp(-max, max),
                    _ => 0.0,
                }
            });
            let drop = if ik.settings.drop_hips {
                dy[0].min(dy[1]).min(0.0)
            } else {
                0.0
            };
            let k = 1.0 - (-ik.settings.response * dt.max(0.0)).exp();
            let feet = [0, 1].map(|i| {
                let want = to_model.transform_vector3(Vec3::Y * dy[i]);
                ik.feet[i] + (want - ik.feet[i]) * k
            });
            let want = to_model.transform_vector3(Vec3::Y * drop);
            updates.push((*id, feet, ik.drop + (want - ik.drop) * k));
        }
        for (id, feet, drop) in updates {
            if let Some(ik) = self
                .animated
                .get_mut(&id)
                .and_then(|e| e.animator.ik.as_mut())
            {
                ik.feet = feet;
                ik.drop = drop;
            }
        }
    }

    /// Look up one skinned/animated entity by its typed render identity.
    pub fn animated_entity(&self, id: EntityId) -> EngineResult<&AnimatedEntity> {
        self.animated.get(&id).ok_or(EngineError::UnknownEntity)
    }
    pub fn animated_entities(&self) -> impl Iterator<Item = (&EntityId, &AnimatedEntity)> {
        self.animated_order
            .iter()
            .filter_map(|id| self.animated.get(id).map(|e| (id, e)))
    }

    /// Orbit camera helper (yaw/pitch in degrees).
    pub fn look_orbit(
        &mut self,
        target: impl Into<Vec3>,
        distance: f32,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) {
        self.camera = Camera::orbit(target, distance, yaw_degrees, pitch_degrees);
        self.camera = self.camera.clone().with_lens(
            self.camera.fov_y_degrees(),
            self.camera.near(),
            self.view_distance,
        );
    }

    /// Set sun direction (need not be normalized) and ambient 0..1.
    pub fn set_sun(&mut self, direction: impl Into<Vec3>, ambient: f32) {
        self.light.direction = direction.into();
        self.light.ambient = ambient.clamp(0.0, 1.0);
    }

    /// Light the scene from the camera eye ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a lantern or headlamp ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â or take
    /// it out again with `None`.
    ///
    /// The torch follows the view every frame; there is no separate position.
    pub fn set_torch(&mut self, torch: Option<TorchLight>) {
        self.torch = torch;
    }

    pub fn torch(&self) -> Option<TorchLight> {
        self.torch
    }

    /// Place a lamp (point light or spotlight) in the space things currently
    /// spawn in. See [`crate::lamps`].
    pub fn add_lamp(&mut self, lamp: crate::lamps::Lamp) -> EngineResult<crate::lamps::LampId> {
        lamp.validate()?;
        let id = crate::lamps::LampId(self.next_lamp);
        self.next_lamp = self.next_lamp.wrapping_add(1).max(1);
        self.lamps.insert(id, (lamp, self.spawn_space));
        Ok(id)
    }

    /// Change everything about a lamp at once.
    pub fn set_lamp(
        &mut self,
        id: crate::lamps::LampId,
        lamp: crate::lamps::Lamp,
    ) -> EngineResult<()> {
        lamp.validate()?;
        let slot = self.lamps.get_mut(&id).ok_or(EngineError::UnknownEntity)?;
        slot.0 = lamp;
        Ok(())
    }

    /// Move a lamp (a lantern someone carries, a car's headlight).
    pub fn set_lamp_position(
        &mut self,
        id: crate::lamps::LampId,
        position: impl Into<Vec3>,
    ) -> EngineResult<()> {
        let p = position.into();
        crate::place::ensure_finite3(p, "lamp position")?;
        let slot = self.lamps.get_mut(&id).ok_or(EngineError::UnknownEntity)?;
        slot.0.position = p;
        Ok(())
    }

    pub fn lamp(&self, id: crate::lamps::LampId) -> Option<&crate::lamps::Lamp> {
        self.lamps.get(&id).map(|(l, _)| l)
    }

    pub fn remove_lamp(&mut self, id: crate::lamps::LampId) {
        self.lamps.remove(&id);
        self.hierarchy.lamps.remove(&id);
    }

    /// Every lamp, with the space it lives in.
    pub fn lamps(
        &self,
    ) -> impl Iterator<Item = (crate::lamps::LampId, &crate::lamps::Lamp, SpaceId)> {
        self.lamps.iter().map(|(id, (l, s))| (*id, l, *s))
    }

    /// How many shadow-casting lamps get their shadows drawn per frame (the
    /// brightest near the camera win; the rest light without shadows).
    /// Default 4; each costs about as much as drawing the scene around it six
    /// times at low resolution (once for a spotlight).
    pub fn set_lamp_shadow_budget(&mut self, lamps: usize) {
        self.lamp_shadow_budget = lamps.min(crate::lamps::MAX_SHADOW_LAMPS);
    }

    pub fn lamp_shadow_budget(&self) -> usize {
        self.lamp_shadow_budget
    }

    // ---- Prefabs (see `crate::prefab`) ----

    /// Place a prefab: makes its parts (hung on a new root pivot at `place`)
    /// and lamps. Move or despawn the root to move or remove all of it.
    pub fn spawn_prefab(
        &mut self,
        prefab: &crate::prefab::Prefab,
        place: Place,
    ) -> EngineResult<crate::prefab::Spawned> {
        use crate::prefab::PartKind;
        prefab.validate()?;
        let root = self.spawn_pivot(place);
        let mut ids: Vec<EntityId> = Vec::with_capacity(prefab.parts.len());
        let mut bodies = Vec::new();
        let result = (|| -> EngineResult<()> {
            for part in &prefab.parts {
                let id = match &part.kind {
                    PartKind::Pivot => self.spawn_pivot(Place::default()),
                    PartKind::Mesh(mesh) => {
                        let id = self.place((**mesh).clone(), Place::default())?;
                        if !part.casts_shadow {
                            self.set_casts_shadow(id, false)?;
                        }
                        id
                    }
                    PartKind::Model { model, clip } => {
                        let id = self.spawn_animated_shared(model.clone(), Place::default())?;
                        if let Some(clip) = clip {
                            self.play_animation(id, clip)?;
                        }
                        id
                    }
                };
                ids.push(id);
                let parent = part.parent.map(|i| ids[i]).unwrap_or(root);
                self.attach(id, parent, part.place)?;
                if let Some(body) = &part.body {
                    bodies.push((id, body.clone()));
                }
            }
            Ok(())
        })();
        let mut lamps = Vec::new();
        let result = result.and_then(|()| {
            for l in &prefab.lamps {
                let on = l.on.map(|i| ids[i]).unwrap_or(root);
                let lamp = self.add_lamp(l.lamp)?;
                lamps.push(lamp);
                match &l.bone {
                    Some(bone) => self.attach_lamp_to_bone(lamp, on, bone, l.lamp.position)?,
                    None => self.attach_lamp(lamp, on, l.lamp.position)?,
                }
            }
            // Bodies last, once every part sits where it belongs.
            for (id, body) in bodies {
                self.add_body(id, body)?;
            }
            Ok(())
        });
        self.prefab_instances.insert(
            root,
            PrefabInstance {
                prefab: prefab.name().to_string(),
                parts: ids.clone(),
                lamps: lamps.clone(),
            },
        );
        if let Err(e) = result {
            self.despawn(root);
            return Err(e);
        }
        Ok(crate::prefab::Spawned {
            root,
            parts: prefab
                .parts
                .iter()
                .zip(&ids)
                .map(|(p, id)| (p.name.clone(), *id))
                .collect(),
            lamps,
        })
    }

    /// Remember a prefab under its name, for [`Self::spawn_named`] and for
    /// loading saved worlds. Registering the same name again replaces it.
    pub fn register_prefab(&mut self, prefab: crate::prefab::Prefab) -> EngineResult<()> {
        prefab.validate()?;
        self.prefabs
            .insert(prefab.name().to_string(), Arc::new(prefab));
        Ok(())
    }

    pub fn prefab(&self, name: &str) -> Option<&crate::prefab::Prefab> {
        self.prefabs.get(name).map(|p| &**p)
    }

    /// Place a registered prefab by name.
    pub fn spawn_named(
        &mut self,
        name: &str,
        place: Place,
    ) -> EngineResult<crate::prefab::Spawned> {
        let prefab = self
            .prefabs
            .get(name)
            .cloned()
            .ok_or_else(|| EngineError::InvalidValue(format!("no prefab called {name:?}")))?;
        self.spawn_prefab(&prefab, place)
    }

    /// Which prefab a root entity was made from.
    pub fn prefab_of(&self, root: EntityId) -> Option<&str> {
        self.prefab_instances.get(&root).map(|p| p.prefab.as_str())
    }

    // ---- Parents and children (see `crate::hierarchy`) ----

    /// An invisible handle with a place: group things under it with
    /// [`Self::attach`] and move them all at once.
    pub fn spawn_pivot(&mut self, place: Place) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.hierarchy.pivots.insert(id, place.to_matrix());
        id
    }

    /// Hang `child` on `parent` at `local` (relative to the parent). From now
    /// on the child follows the parent, and [`Self::set_place`] on the child
    /// is relative to it.
    pub fn attach(&mut self, child: EntityId, parent: EntityId, local: Place) -> EngineResult<()> {
        self.link_child(child, parent, local.to_matrix(), None)
    }

    /// Hang `child` on a bone of an animated `parent` (`"hand_r"`, `"head"`),
    /// at `local` relative to the bone. It moves with the animation.
    pub fn attach_to_bone(
        &mut self,
        child: EntityId,
        parent: EntityId,
        bone: &str,
        local: Place,
    ) -> EngineResult<()> {
        let bone = self.bone_index(parent, bone)?;
        self.link_child(child, parent, local.to_matrix(), Some(bone))
    }

    /// Give `child` a new parent (or none) without moving it on screen.
    pub fn set_parent(&mut self, child: EntityId, parent: Option<EntityId>) -> EngineResult<()> {
        let world = self.world_transform(child)?;
        match parent {
            None => {
                if self.hierarchy.unlink(child).is_some() {
                    self.write_matrix(child, world)?;
                }
                Ok(())
            }
            Some(parent) => {
                let frame = self.world_transform(parent)?;
                self.link_child(child, parent, frame.inverse() * world, None)
            }
        }
    }

    /// Let go of the parent; the entity stays where it is.
    pub fn detach(&mut self, child: EntityId) -> EngineResult<()> {
        self.set_parent(child, None)
    }

    pub fn parent(&self, id: EntityId) -> Option<EntityId> {
        self.hierarchy.links.get(&id).map(|l| l.parent)
    }

    pub fn children(&self, id: EntityId) -> &[EntityId] {
        self.hierarchy.children_of(id)
    }

    /// Where an entity (mesh, animated model or pivot) really is, parents
    /// included.
    pub fn world_transform(&self, id: EntityId) -> EngineResult<Mat4> {
        self.matrix_of(id).ok_or(EngineError::UnknownEntity)
    }

    /// Where a bone of an animated entity is in the world this frame.
    pub fn bone_transform(&self, id: EntityId, bone: &str) -> EngineResult<Mat4> {
        let index = self.bone_index(id, bone)?;
        let e = self.animated.get(&id).ok_or(EngineError::UnknownEntity)?;
        Ok(e.transform * e.animator.joint_globals()[index])
    }

    /// Let a lamp ride on an entity: `offset` is relative to it, and a
    /// spotlight's direction is taken in the entity's frame (a headlight
    /// pointing along +Z keeps pointing forward as the car turns).
    pub fn attach_lamp(
        &mut self,
        lamp: crate::lamps::LampId,
        entity: EntityId,
        offset: impl Into<Vec3>,
    ) -> EngineResult<()> {
        self.link_lamp(lamp, entity, offset.into(), None)
    }

    /// Let a lamp ride on a bone: a lantern in a hand, a torch on a helmet.
    pub fn attach_lamp_to_bone(
        &mut self,
        lamp: crate::lamps::LampId,
        entity: EntityId,
        bone: &str,
        offset: impl Into<Vec3>,
    ) -> EngineResult<()> {
        let bone = self.bone_index(entity, bone)?;
        self.link_lamp(lamp, entity, offset.into(), Some(bone))
    }

    /// Stop a lamp riding on anything; it stays where it is.
    pub fn detach_lamp(&mut self, lamp: crate::lamps::LampId) {
        self.hierarchy.lamps.remove(&lamp);
    }

    fn link_lamp(
        &mut self,
        lamp: crate::lamps::LampId,
        entity: EntityId,
        offset: Vec3,
        bone: Option<usize>,
    ) -> EngineResult<()> {
        if !offset.is_finite() {
            return Err(EngineError::InvalidValue(
                "lamp offset must be finite".into(),
            ));
        }
        let (l, _) = self.lamps.get(&lamp).ok_or(EngineError::UnknownEntity)?;
        self.matrix_of(entity).ok_or(EngineError::UnknownEntity)?;
        let direction = match l.shape {
            crate::lamps::LampShape::Spot { direction, .. } => Some(direction),
            crate::lamps::LampShape::Point => None,
        };
        self.hierarchy.lamps.insert(
            lamp,
            crate::hierarchy::LampLink {
                entity,
                bone,
                offset,
                direction,
            },
        );
        self.update_lamps();
        Ok(())
    }

    fn bone_index(&self, id: EntityId, bone: &str) -> EngineResult<usize> {
        let e = self.animated.get(&id).ok_or_else(|| {
            EngineError::InvalidValue(format!("entity {id} has no skeleton (not animated)"))
        })?;
        e.animator
            .model()
            .skeleton
            .joint_index(bone)
            .ok_or_else(|| EngineError::InvalidValue(format!("no bone called {bone:?}")))
    }

    fn link_child(
        &mut self,
        child: EntityId,
        parent: EntityId,
        local: Mat4,
        bone: Option<usize>,
    ) -> EngineResult<()> {
        if !local.is_finite() {
            return Err(EngineError::InvalidValue(
                "local place must be finite".into(),
            ));
        }
        self.matrix_of(child).ok_or(EngineError::UnknownEntity)?;
        self.matrix_of(parent).ok_or(EngineError::UnknownEntity)?;
        if self.hierarchy.is_ancestor(child, parent) {
            return Err(EngineError::InvalidValue(format!(
                "can't attach {child} below itself"
            )));
        }
        if self
            .physics
            .body_kind(child)
            .is_some_and(|k| k == crate::physics::BodyKind::Dynamic)
        {
            return Err(EngineError::InvalidValue(format!(
                "{child} has a dynamic body; physics moves it, so it can't follow a parent"
            )));
        }
        self.hierarchy.link(
            child,
            crate::hierarchy::Link {
                parent,
                local,
                bone,
            },
        );
        self.update_subtree(child)
    }

    fn matrix_of(&self, id: EntityId) -> Option<Mat4> {
        if let Some(e) = self.entities.get(&id) {
            return Some(e.transform);
        }
        if let Some(e) = self.animated.get(&id) {
            return Some(e.transform);
        }
        self.hierarchy.pivots.get(&id).copied()
    }

    fn write_matrix(&mut self, id: EntityId, m: Mat4) -> EngineResult<()> {
        if let Some(e) = self.entities.get_mut(&id) {
            if !crate::hierarchy::same_up_to_rounding(e.transform, m) {
                e.transform = m;
                e.bump_xform();
                self.bump_render_epoch();
            }
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.transform = m;
            return Ok(());
        }
        if let Some(p) = self.hierarchy.pivots.get_mut(&id) {
            *p = m;
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
    }

    /// Recompute where everything below `id` is (and `id` itself when it is
    /// a child).
    fn update_subtree(&mut self, id: EntityId) -> EngineResult<()> {
        let mut poses: HashMap<EntityId, Vec<Mat4>> = HashMap::new();
        for e in self.hierarchy.subtree(id) {
            self.place_child(e, &mut poses)?;
        }
        self.update_lamps();
        Ok(())
    }

    fn place_child(
        &mut self,
        id: EntityId,
        poses: &mut HashMap<EntityId, Vec<Mat4>>,
    ) -> EngineResult<()> {
        let Some(link) = self.hierarchy.links.get(&id).copied() else {
            return Ok(());
        };
        let frame = self.link_frame(link.parent, link.bone, poses)?;
        self.write_matrix(id, frame * link.local)
    }

    /// World matrix of a parent, or of one of its bones.
    fn link_frame(
        &self,
        parent: EntityId,
        bone: Option<usize>,
        poses: &mut HashMap<EntityId, Vec<Mat4>>,
    ) -> EngineResult<Mat4> {
        let frame = self.matrix_of(parent).ok_or(EngineError::UnknownEntity)?;
        let Some(bone) = bone else {
            return Ok(frame);
        };
        let pose = match poses.get(&parent) {
            Some(p) => p,
            None => {
                let e = self
                    .animated
                    .get(&parent)
                    .ok_or(EngineError::UnknownEntity)?;
                poses.entry(parent).or_insert(e.animator.joint_globals())
            }
        };
        Ok(frame * pose.get(bone).copied().unwrap_or(Mat4::IDENTITY))
    }

    fn update_lamps(&mut self) {
        if self.hierarchy.lamps.is_empty() {
            return;
        }
        let mut poses = HashMap::new();
        let links: Vec<_> = self.hierarchy.lamps.iter().map(|(k, v)| (*k, *v)).collect();
        for (id, link) in links {
            let Ok(frame) = self.link_frame(link.entity, link.bone, &mut poses) else {
                continue;
            };
            if let Some((lamp, _)) = self.lamps.get_mut(&id) {
                crate::hierarchy::place_lamp(lamp, &link, frame);
            }
        }
    }

    /// Once a frame after animation and physics: children follow parents
    /// that moved by themselves (bodies, bones) and lamps follow their
    /// entities.
    pub(crate) fn tick_hierarchy(&mut self) {
        if self.hierarchy.links.is_empty() && self.hierarchy.lamps.is_empty() {
            return;
        }
        let mut poses = HashMap::new();
        for root in self.hierarchy.roots() {
            for id in self.hierarchy.subtree(root) {
                let _ = self.place_child(id, &mut poses);
            }
        }
        self.update_lamps();
    }

    /// Enable hybrid sun shadows, or pass `None` to turn them off.
    pub fn set_shadows(&mut self, shadows: Option<ShadowSettings>) {
        self.shadows = shadows;
    }

    pub fn shadows(&self) -> Option<ShadowSettings> {
        self.shadows
    }

    /// Choose the instanced submit path. Both are first-class; neither falls back.
    pub fn set_instance_submit(&mut self, submit: InstanceSubmit) {
        if self.instance_submit == submit {
            return;
        }
        self.instance_submit = submit;
        self.bump_render_epoch();
    }

    pub fn instance_submit(&self) -> InstanceSubmit {
        self.instance_submit
    }

    pub fn ambient_occlusion(&self) -> Option<AmbientOcclusion> {
        self.ambient_occlusion
    }

    pub fn set_ambient_occlusion(&mut self, ao: Option<AmbientOcclusion>) {
        self.ambient_occlusion = ao;
    }

    pub fn global_illumination(&self) -> Option<GlobalIllumination> {
        self.global_illumination
    }

    pub fn set_global_illumination(&mut self, gi: Option<GlobalIllumination>) {
        self.global_illumination = gi;
    }

    /// Smooth destructible terrain drawn in the default space.
    pub fn voxel_terrain(&self) -> Option<&crate::voxel_terrain::VoxelTerrain> {
        self.voxel_terrain.as_ref()
    }

    /// Edit the voxel terrain; the renderer re-meshes the changed chunks.
    pub fn voxel_terrain_mut(&mut self) -> Option<&mut crate::voxel_terrain::VoxelTerrain> {
        self.voxel_terrain.as_mut()
    }

    pub fn set_voxel_terrain(&mut self, terrain: Option<crate::voxel_terrain::VoxelTerrain>) {
        self.voxel_baseline_rev = terrain.as_ref().map_or(0, |t| t.volume().rev());
        self.voxel_terrain = terrain;
    }

    /// The water simulation, if one is set.
    pub fn water(&self) -> Option<&crate::water::Water> {
        self.water.as_ref()
    }

    /// Pour, add sources, change settings. Terrain edits made through
    /// [`World::voxel_terrain_mut`] reach the water by themselves.
    pub fn water_mut(&mut self) -> Option<&mut crate::water::Water> {
        self.water.as_mut()
    }

    /// Set (or remove) the water. The engine steps it every frame and, when
    /// the voxel terrain is dug or built on, re-reads the changed columns so
    /// water flows into new cavities and out of filled ones.
    pub fn set_water(&mut self, water: Option<crate::water::Water>) {
        self.water = water;
        self.water_terrain_rev = self.voxel_terrain.as_ref().map_or(0, |t| t.volume().rev());
    }

    /// Step the water. The engine calls this every frame.
    pub fn tick_water(&mut self, dt: f32) {
        let Some(water) = self.water.as_mut() else {
            return;
        };
        if let Some(terrain) = self.voxel_terrain.as_ref() {
            let volume = terrain.volume();
            if volume.rev() != self.water_terrain_rev {
                let (keys, _) = volume.changed_since(self.water_terrain_rev);
                self.water_terrain_rev = volume.rev();
                let size = volume.chunk_world_size();
                let mut xz: Vec<glam::IVec2> =
                    keys.iter().map(|k| glam::IVec2::new(k.x, k.z)).collect();
                xz.sort_by_key(|k| (k.x, k.y));
                xz.dedup();
                let region = *water.region();
                let mut changed = vec![false; region.column_count()];
                for k in xz {
                    let min = k.as_vec2() * size;
                    crate::water::mark_columns(
                        &region,
                        min,
                        min + glam::Vec2::splat(size),
                        &mut changed,
                    );
                }
                water.resync_columns(terrain, &changed);
            }
        }
        if dt.is_finite() && dt > 0.0 {
            water.update(dt);
        }
    }

    /// Break a mesh entity into solid pieces that fly away from `impact`
    /// (world space), fall, bounce off the ground and settle. Works on any
    /// ordinary mesh (props, rocks, walls, buildings); the original entity is
    /// removed. Returns the pieces' entity ids.
    pub fn shatter(
        &mut self,
        id: EntityId,
        impact: impl Into<Vec3>,
        settings: crate::debris::Shatter,
    ) -> EngineResult<Vec<EntityId>> {
        settings.validate()?;
        let impact = impact.into();
        crate::place::ensure_finite3(impact, "shatter impact")?;
        let entity = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if entity.instanced || entity.instance_of.is_some() {
            return Err(EngineError::InvalidValue(
                "shatter works on single (non-instanced) mesh entities".into(),
            ));
        }
        let fragments = crate::fracture::fracture(
            &entity.mesh,
            entity.transform,
            impact,
            settings.fracture_settings(),
        );
        let (material, albedo, casts_shadow, space) = (
            entity.material,
            entity.albedo,
            entity.casts_shadow,
            entity.space,
        );
        self.despawn(id);
        let prev_space = self.spawn_space;
        self.spawn_space = space;
        let mut rng = settings.seed.wrapping_mul(2_654_435_761) | 1;
        let mut rand = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let reach = fragments
            .iter()
            .map(|f| f.centroid.distance(impact))
            .fold(0.0_f32, f32::max)
            .max(0.5);
        let mut ids = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            let piece_id =
                self.spawn_built(fragment.mesh, Mat4::from_translation(fragment.centroid));
            if let Some(e) = self.entities.get_mut(&piece_id) {
                e.material = material;
                e.albedo = albedo;
                e.casts_shadow = casts_shadow;
            }
            let away = (fragment.centroid - impact)
                .try_normalize()
                .unwrap_or(Vec3::Y);
            let falloff = 1.0 - 0.6 * (fragment.centroid.distance(impact) / reach);
            let velocity = (away
                + Vec3::new(rand() * 0.3, 0.35 + rand().abs() * 0.3, rand() * 0.3))
                * settings.speed
                * falloff;
            let spin = Vec3::new(rand(), rand(), rand()) * settings.speed
                / fragment.radius.max(0.15)
                * 0.4;
            let body = crate::physics::Body::dynamic()
                .density(settings.density)
                .friction(0.7)
                .bounce(0.2)
                .velocity(velocity)
                .spin(spin)
                .ccd(fragment.radius < 0.3);
            let positions = &self.entities[&piece_id].mesh.positions;
            let added = self.physics.add(
                piece_id,
                positions,
                Mat4::from_translation(fragment.centroid),
                &body,
                Some(crate::physics::Temporary {
                    keep: settings.keep_seconds.unwrap_or(f32::INFINITY),
                }),
                Vec::new,
            );
            if added.is_err() {
                // Slivers too thin for a hull: drop them rather than leave
                // them hanging in the air.
                self.despawn(piece_id);
                continue;
            }
            ids.push(piece_id);
        }
        self.spawn_space = prev_space;
        Ok(ids)
    }

    /// Shattered pieces still in the world (flying, resting or sinking away).
    pub fn debris_count(&self) -> usize {
        self.physics.temporary_count()
    }

    /// Give an entity a rigid body: from now on physics moves it (or, for
    /// fixed bodies, other bodies bump into it). Moving the entity with
    /// [`Self::set_place`] still works and teleports the body. Adding a body
    /// to an entity that has one replaces it. See [`crate::physics`].
    ///
    /// A hull or bounding-box body also wraps the meshes hanging on the
    /// entity (attach the wheels first, then add the body).
    ///
    /// A dynamic body on a child (see [`Self::attach`]) lets go of its parent:
    /// physics moves it from then on. Fixed and kinematic children keep
    /// following their parent.
    pub fn add_body(&mut self, id: EntityId, body: crate::physics::Body) -> EngineResult<()> {
        if body.kind == crate::physics::BodyKind::Dynamic && self.hierarchy.links.contains_key(&id)
        {
            self.detach(id)?;
        }
        let e = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if e.instanced || e.instance_of.is_some() {
            return Err(EngineError::InvalidValue(
                "bodies work on single (non-instanced) mesh entities".into(),
            ));
        }
        let mesh = &e.mesh;
        // A hull or box also wraps the meshes hanging on this entity (a
        // cart's wheels), so the whole thing rests on the ground.
        let mut points = std::borrow::Cow::Borrowed(&mesh.positions[..]);
        if matches!(
            body.shape,
            crate::physics::BodyShape::Hull | crate::physics::BodyShape::BoundingBox
        ) && self.hierarchy.children.contains_key(&id)
        {
            let to_local = e.transform.inverse();
            let mut all = mesh.positions.clone();
            for child in self.hierarchy.subtree(id).into_iter().skip(1) {
                if let Some(c) = self.entities.get(&child) {
                    let m = to_local * c.transform;
                    all.extend(c.mesh.positions.iter().map(|p| m.transform_point3(*p)));
                }
            }
            points = std::borrow::Cow::Owned(all);
        }
        self.physics.add(id, &points, e.transform, &body, None, || {
            mesh.indices
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect()
        })
    }

    /// Take an entity's body away; it stays where it is and stops moving.
    pub fn remove_body(&mut self, id: EntityId) {
        self.physics.remove(id);
    }

    pub fn has_body(&self, id: EntityId) -> bool {
        self.physics.contains(id)
    }

    /// Velocity of a body in m/s.
    pub fn body_velocity(&self, id: EntityId) -> Option<Vec3> {
        self.physics.velocity(id)
    }

    pub fn set_body_velocity(
        &mut self,
        id: EntityId,
        velocity: impl Into<Vec3>,
    ) -> EngineResult<()> {
        self.physics.set_velocity(id, velocity.into())
    }

    /// Spin of a body in rad/s around each axis.
    pub fn body_spin(&self, id: EntityId) -> Option<Vec3> {
        self.physics.spin(id)
    }

    pub fn set_body_spin(&mut self, id: EntityId, spin: impl Into<Vec3>) -> EngineResult<()> {
        self.physics.set_spin(id, spin.into())
    }

    /// Mass of a body in kg (volume times density times 1000).
    pub fn body_mass(&self, id: EntityId) -> Option<f32> {
        self.physics.mass(id)
    }

    /// Give a body a kick through its centre. `impulse` is in kg·m/s, so
    /// `direction * speed * world.body_mass(id)` changes its speed by `speed`.
    pub fn push(&mut self, id: EntityId, impulse: impl Into<Vec3>) -> EngineResult<()> {
        self.physics.push(id, impulse.into(), None)
    }

    /// Kick a body at a world point, so it also starts to spin.
    pub fn push_at(
        &mut self,
        id: EntityId,
        impulse: impl Into<Vec3>,
        point: impl Into<Vec3>,
    ) -> EngineResult<()> {
        self.physics.push(id, impulse.into(), Some(point.into()))
    }

    /// Throw every body within `radius` of `center` outward: `speed` m/s at
    /// the centre, fading to nothing at the edge. Mass doesn't matter.
    pub fn blast(&mut self, center: impl Into<Vec3>, radius: f32, speed: f32) {
        let c = center.into();
        if c.is_finite() && radius.is_finite() && radius > 0.0 && speed.is_finite() {
            self.physics.blast(c, radius, speed);
        }
    }

    /// Hard hits since the last call (for impact sounds, damage or dust),
    /// hardest first.
    pub fn take_impacts(&mut self) -> Vec<crate::physics::Impact> {
        self.physics.take_impacts()
    }

    /// First body or piece of terrain along a ray, within `max_distance`.
    pub fn cast_ray(
        &self,
        origin: impl Into<Vec3>,
        direction: impl Into<Vec3>,
        max_distance: f32,
    ) -> Option<crate::physics::RayHit> {
        let (origin, dir) = (origin.into(), direction.into().try_normalize()?);
        if !origin.is_finite() || !max_distance.is_finite() || max_distance <= 0.0 {
            return None;
        }
        let body = self.physics.cast_ray(origin, dir, max_distance);
        let reach = body.map_or(max_distance, |h| h.distance);
        let ground = self.ground_ray(origin, dir, reach);
        match (body, ground) {
            (Some(b), Some(g)) if g.distance < b.distance => Some(g),
            (Some(b), _) => Some(b),
            (None, g) => g,
        }
    }

    fn ground_ray(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<crate::physics::RayHit> {
        use crate::physics::{PhysicsGround, RayHit};
        let hit = |p: Vec3, n: Vec3| RayHit {
            entity: None,
            point: p,
            normal: n,
            distance: (p - origin).length(),
        };
        let plane = |y: f32| {
            (dir.y.abs() > 1e-6)
                .then(|| (y - origin.y) / dir.y)
                .filter(|t| *t >= 0.0 && *t <= max)
                .map(|t| hit(origin + dir * t, Vec3::Y))
        };
        match self.physics.settings().ground {
            PhysicsGround::None => None,
            PhysicsGround::Plane(y) => plane(y),
            PhysicsGround::Auto => {
                if let Some(v) = self.voxel_terrain.as_ref() {
                    if v.volume().chunk_keys().next().is_some() {
                        return v.raycast(origin, dir, max).map(|p| hit(p, v.normal_at(p)));
                    }
                }
                if let Some(f) = self.height_field.as_ref() {
                    return crate::physics::height_ray(&|x, z| f.height_at(x, z), origin, dir, max)
                        .map(|(p, n)| hit(p, n));
                }
                plane(0.0)
            }
        }
    }

    pub fn physics_settings(&self) -> crate::physics::PhysicsSettings {
        self.physics.settings()
    }

    pub fn set_physics_settings(
        &mut self,
        settings: crate::physics::PhysicsSettings,
    ) -> EngineResult<()> {
        settings.validate()?;
        self.physics.set_settings(settings);
        Ok(())
    }

    /// Bodies in the world, and how many of them are moving (not asleep).
    pub fn body_count(&self) -> (usize, usize) {
        (self.physics.body_count(), self.physics.awake_count())
    }

    /// Step physics. The engine calls this every frame.
    pub fn tick_physics(&mut self, dt: f32) {
        let field = self.height_field.as_ref();
        let height = |x: f32, z: f32| field.map_or(0.0, |f| f.height_at(x, z));
        let around = crate::physics::Surroundings {
            voxels: self.voxel_terrain.as_ref(),
            heights: field.map(|f| (f as *const _ as usize, &height as crate::physics::HeightFn)),
            water: self.water.as_ref(),
        };
        let entities = &self.entities;
        let result = self
            .physics
            .tick(dt, &around, |id| entities.get(&id).map(|e| e.transform));
        if let Some(water) = self.water.as_mut() {
            for s in &result.splashes {
                water.disturb(s.at, 1.5, (s.speed * 0.3).min(4.0));
            }
        }
        let mut changed = false;
        for (id, m) in result.moved {
            if let Some(e) = self.entities.get_mut(&id) {
                e.transform = m;
                e.bump_xform();
                changed = true;
            }
        }
        for (id, dy) in result.sink {
            if let Some(e) = self.entities.get_mut(&id) {
                e.transform.w_axis.y -= dy;
                e.bump_xform();
                changed = true;
            }
        }
        if changed {
            self.bump_render_epoch();
        }
        for id in result.expired {
            self.despawn(id);
        }
    }

    pub fn bloom(&self) -> BloomSettings {
        self.bloom
    }

    pub fn set_bloom(&mut self, bloom: BloomSettings) {
        self.bloom = bloom;
    }

    /// Choose the tone-mapping curve (AgX by default). `ENGINE_TONEMAP=aces`
    /// or `agx` overrides it for quick comparisons.
    pub fn set_tone_map(&mut self, tone_map: ToneMap) {
        self.tone_map = tone_map;
    }

    pub fn tone_map(&self) -> ToneMap {
        self.tone_map
    }

    /// Request synchronized presentation (FIFO). Turning it off presents
    /// uncapped (`Immediate`, or `Mailbox` when the surface offers it); the
    /// renderer must refuse a surface that cannot present uncapped.
    pub fn set_vsync(&mut self, vsync: bool) {
        self.vsync = vsync;
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }
    pub(crate) fn note_shadow_contact(&mut self, snap: ContactSnapshot, epoch: u64) {
        if self.shadow_contact_epoch != epoch {
            self.shadow_contact_epoch = epoch;
            self.shadow_contact = snap;
        }
    }

    pub(crate) fn shadow_contact(&self) -> &ContactSnapshot {
        &self.shadow_contact
    }

    pub(crate) fn shadow_contact_epoch(&self) -> u64 {
        self.shadow_contact_epoch
    }

    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    /// Put air in the scene, or take it out again with `None`.
    pub fn set_haze(&mut self, haze: Option<Haze>) {
        self.haze = haze;
    }

    pub fn haze(&self) -> Option<Haze> {
        self.haze
    }

    /// Put a sky behind the scene, or fall back to [`Self::clear_color`].
    pub fn set_sky(&mut self, sky: Option<Sky>) {
        self.sky = sky;
    }

    pub fn sky(&self) -> Option<Sky> {
        self.sky
    }

    /// Drive the sun, moon, sky dome, ambient light, haze and exposure from
    /// a time of day and weather (see [`crate::sky`]). Change `hour` every
    /// frame for a running day. `None` hands sun, sky and haze back to the
    /// game, as they were last set.
    pub fn set_physical_sky(&mut self, sky: Option<crate::sky::PhysicalSky>) {
        self.physical_sky = sky;
        self.tick_sky();
    }

    pub fn physical_sky(&self) -> Option<crate::sky::PhysicalSky> {
        self.physical_sky
    }

    /// Wind for swaying vegetation (see [`crate::vegetation`]): 0 calm,
    /// 1 a strong breeze, 2 a storm. Default 0.5.
    pub fn set_wind(&mut self, strength: f32) -> EngineResult<()> {
        if !strength.is_finite() || !(0.0..=4.0).contains(&strength) {
            return Err(EngineError::InvalidValue(format!(
                "wind {strength} must be within 0..=4"
            )));
        }
        self.wind = strength;
        Ok(())
    }

    pub fn wind(&self) -> f32 {
        self.wind
    }

    /// Leaf colour season: 0 summer green, 0.5 early autumn, 1 full autumn
    /// (evergreens ignore it).
    pub fn set_season(&mut self, season: f32) -> EngineResult<()> {
        if !season.is_finite() || !(0.0..=1.0).contains(&season) {
            return Err(EngineError::InvalidValue(format!(
                "season {season} must be within 0..=1"
            )));
        }
        self.season = season;
        Ok(())
    }

    pub fn season(&self) -> f32 {
        self.season
    }

    /// Apply the physical sky's lighting to the world's light, sky, haze
    /// and exposure. Runs every frame before rendering.
    pub(crate) fn tick_sky(&mut self) {
        let Some(sky) = self.physical_sky else {
            return;
        };
        let l = sky.lighting();
        // AssetGenerator's sun is 3.2 at a clear low sun, ~4.5 at noon, under
        // an exposure of 0.55; the engine's daylight is 1.0 under exposure 1.
        self.light.direction = l.light_dir;
        self.light.color = l.light_color * (l.light_intensity / 3.2);
        self.light.ambient = (l.hemi_intensity * 0.3).clamp(0.0, 1.0);
        let c = |v: Vec3| Color {
            r: v.x,
            g: v.y,
            b: v.z,
            a: 1.0,
        };
        let sun_color = l.light_color / l.light_color.max_element().max(1e-3);
        self.sky = Some(Sky {
            zenith: c(l.sky_color),
            horizon: c(l.fog_color),
            ground: c(l.ground_color),
            sun_color: c(sun_color),
            sun_size_degrees: crate::sky::SUN_RADIUS.to_degrees(),
            sun_bloom_degrees: 20.0,
            curve: 0.22,
        });
        // exp(-d * density) reaches 2% at 3.9 / density.
        let visibility = (3.9 / l.fog_density).min(self.view_distance.max(50.0) * 1.5);
        self.haze = Some(Haze::new(c(l.fog_color), visibility));
        let _ = self.bloom.set_exposure((l.exposure / 0.55).clamp(0.1, 8.0));
    }

    pub fn entity(&self, id: EntityId) -> EngineResult<&Entity> {
        self.entities.get(&id).ok_or(EngineError::UnknownEntity)
    }

    pub(crate) fn entities(&self) -> impl Iterator<Item = (EntityId, &Entity)> {
        self.order
            .iter()
            .filter_map(|id| self.entities.get(id).map(|e| (*id, e)))
    }

    pub(crate) fn contains_entity(&self, id: EntityId) -> bool {
        self.entities.contains_key(&id)
    }

    pub(crate) fn contains_animated(&self, id: EntityId) -> bool {
        self.animated.contains_key(&id)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Replace or create a streamed chunk mesh (infinite terrain / LOD).
    pub fn set_chunk(&mut self, key: IVec3, mesh: Mesh) -> EntityId {
        self.set_chunk_built(key, mesh.build())
    }

    /// Like [`set_chunk`] but accepts a pre-built mesh (for background generation).
    pub fn set_chunk_built(&mut self, key: IVec3, built: BuiltMesh) -> EntityId {
        if let Some(&id) = self.chunk_entities.get(&key) {
            // Remesh requires a new GPU upload ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â swap entity id so renderer rebuilds.
            self.despawn(id);
            self.chunk_entities.remove(&key);
        }
        let material = self.chunk_material(&built);
        let id = self.spawn_built(built, Mat4::IDENTITY);
        if material.is_some() {
            self.entities.get_mut(&id).expect("just spawned").material = material;
        }
        self.chunk_entities.insert(key, id);
        id
    }

    /// Pick the default material for a streamed layer from its opacity.
    ///
    /// A fully opaque layer is ground; a fully translucent one is a water
    /// sheet. Mixed layers are left to the plain lit pipeline.
    fn chunk_material(&self, built: &BuiltMesh) -> Option<SurfaceMaterialRef> {
        if built.index_count() == 0 {
            return None;
        }
        if built.opaque_index_count == built.index_count() {
            return self
                .default_terrain_material
                .map(SurfaceMaterialRef::Terrain);
        }
        if built.opaque_index_count == 0 {
            return self.default_water_material.map(SurfaceMaterialRef::Water);
        }
        None
    }

    /// Create an RGBA8 texture from raw bytes (`width * height * 4`).
    pub fn create_texture_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> EngineResult<TextureId> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidValue(
                "texture size must be non-zero".into(),
            ));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| EngineError::InvalidValue("texture dimensions overflow".into()))?;
        if rgba.len() != expected {
            return Err(EngineError::InvalidValue(format!(
                "texture rgba len {} != {}x{}x4",
                rgba.len(),
                width,
                height
            )));
        }
        let id = TextureId(self.next_texture_id);
        self.next_texture_id += 1;
        self.textures.insert(
            id,
            CpuTexture {
                width,
                height,
                rgba,
            },
        );
        self.bump_resource_epoch();
        Ok(id)
    }

    /// Load a PNG from disk into an RGBA8 texture.
    pub fn load_texture_png(&mut self, path: impl AsRef<Path>) -> EngineResult<TextureId> {
        let (w, h, rgba) = load_rgba8_png(path)?;
        self.create_texture_rgba(w, h, rgba)
    }

    /// Create a built-in tileable terrain albedo.
    pub fn create_terrain_albedo(
        &mut self,
        kind: TerrainAlbedo,
        size: u32,
        seed: u32,
    ) -> EngineResult<TextureId> {
        let (w, h, rgba) = generate_terrain_albedo(kind, size, seed);
        self.create_texture_rgba(w, h, rgba)
    }

    /// Create a lush/dry/moor/sand/rock terrain material (world-XZ sampling).
    pub fn create_terrain_material(
        &mut self,
        desc: TerrainMaterialDesc,
    ) -> EngineResult<MaterialId> {
        for tid in [
            desc.grass,
            desc.grass_dry,
            desc.grass_moor,
            desc.mud,
            desc.tundra,
            desc.scree,
            desc.sand,
            desc.rock,
        ] {
            if !self.textures.contains_key(&tid) {
                return Err(EngineError::UnknownTexture);
            }
        }
        if desc.metres_per_tile <= 0.0 {
            return Err(EngineError::InvalidValue(
                "metres_per_tile must be > 0".into(),
            ));
        }
        let id = MaterialId(self.next_material_id);
        self.next_material_id += 1;
        self.materials.insert(id, TerrainMaterial { desc });
        self.bump_resource_epoch();
        Ok(id)
    }

    /// Fully-opaque chunk uploads (`set_chunk_built`) receive this material.
    pub fn set_default_terrain_material(&mut self, material: Option<MaterialId>) {
        self.default_terrain_material = material;
    }

    /// Create an animated water material.
    pub fn create_water_material(
        &mut self,
        desc: WaterMaterialDesc,
    ) -> EngineResult<WaterMaterialId> {
        if desc.depth_scale_m <= 0.0 || desc.wave_length_m <= 0.0 {
            return Err(EngineError::InvalidValue(
                "water depth scale and wave length must be > 0".into(),
            ));
        }
        let id = WaterMaterialId(self.next_material_id);
        self.next_material_id += 1;
        self.water_materials.insert(id, WaterMaterial { desc });
        self.bump_resource_epoch();
        Ok(id)
    }

    /// Fully-translucent chunk layers receive this material.
    pub fn set_default_water_material(&mut self, material: Option<WaterMaterialId>) {
        self.default_water_material = material;
    }

    pub fn set_entity_material(
        &mut self,
        id: EntityId,
        material: Option<SurfaceMaterialRef>,
    ) -> EngineResult<()> {
        match material {
            Some(SurfaceMaterialRef::Terrain(mat)) if !self.materials.contains_key(&mat) => {
                return Err(EngineError::UnknownMaterial)
            }
            Some(SurfaceMaterialRef::Water(mat)) if !self.water_materials.contains_key(&mat) => {
                return Err(EngineError::UnknownMaterial)
            }
            _ => {}
        }
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if e.material == material {
            return Ok(());
        }
        e.material = material;
        self.bump_render_epoch();
        Ok(())
    }

    /// Replace the placements of an instanced entity (see [`Self::spawn_many`]).
    ///
    /// Only the instance buffer is rewritten, so a scatter layer can follow the
    /// streamed world every frame without re-uploading its mesh.
    pub fn set_instances(&mut self, id: EntityId, places: &[Place]) -> EngineResult<()> {
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "set_instances has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if !e.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} was not spawned instanced; use set_place"
            )));
        }
        e.instances.clear();
        e.instances
            .extend(places.iter().copied().map(Place::to_instance));
        e.bump_xform();
        self.bump_render_epoch();
        Ok(())
    }

    /// Reserve room in an instanced entity's GPU batch slot without drawing
    /// placeholder instances.
    ///
    /// Streaming systems should reserve their known per-cell maximum before
    /// filling a reusable entity. Later instance-count changes within this
    /// capacity remain partial writes instead of rebuilding the whole batch.
    pub fn reserve_instances(&mut self, id: EntityId, minimum_capacity: usize) -> EngineResult<()> {
        if minimum_capacity as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "reserve_instances requested {minimum_capacity} instances (limit {})",
                self.limits.max_instances_per_spawn
            )));
        }
        let entity = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if !entity.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} was not spawned instanced; it cannot reserve instances"
            )));
        }
        if minimum_capacity <= entity.instance_reserve {
            return Ok(());
        }
        entity.instance_reserve = minimum_capacity;
        self.bump_render_epoch();
        Ok(())
    }

    /// Seconds since start, for materials that animate.
    pub fn set_time(&mut self, seconds: f32) {
        self.time = seconds;
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub(crate) fn textures(&self) -> &HashMap<TextureId, CpuTexture> {
        &self.textures
    }

    pub(crate) fn surface_maps(&self, id: TextureId) -> Option<&std::sync::Arc<MaterialMaps>> {
        self.surface_maps.get(&id)
    }

    pub(crate) fn materials(&self) -> &HashMap<MaterialId, TerrainMaterial> {
        &self.materials
    }

    pub(crate) fn water_materials(&self) -> &HashMap<WaterMaterialId, WaterMaterial> {
        &self.water_materials
    }

    /// Remove a streamed chunk if present.
    pub fn clear_chunk(&mut self, key: IVec3) {
        if let Some(id) = self.chunk_entities.remove(&key) {
            self.despawn(id);
        }
    }

    pub fn has_chunk(&self, key: IVec3) -> bool {
        self.chunk_entities.contains_key(&key)
    }

    /// First-person camera (yaw in degrees, 0 = +Z; pitch positive is up).
    pub fn look_first_person(
        &mut self,
        eye: impl Into<Vec3>,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) {
        self.camera = Camera::first_person(eye, yaw_degrees, pitch_degrees);
        self.camera = self.camera.clone().with_lens(
            self.camera.fov_y_degrees(),
            self.camera.near(),
            self.view_distance,
        );
    }

    /// Third-person follow camera (yaw in degrees, 0 = +Z).
    pub fn look_follow(
        &mut self,
        target: impl Into<Vec3>,
        yaw_degrees: f32,
        distance: f32,
        height: f32,
    ) {
        self.camera = Camera::follow(target, yaw_degrees, distance, height);
        self.camera = self.camera.clone().with_lens(
            self.camera.fov_y_degrees(),
            self.camera.near(),
            self.view_distance,
        );
    }

    /// How far the camera helpers may see, in metres.
    ///
    /// The helpers rebuild the camera every frame, so a game that reaches into
    /// `world.camera.far()` has it overwritten on the next one; this is the knob
    /// that survives. Depth is reversed, so a horizon-scale distance costs
    /// nothing in precision.
    pub fn set_view_distance(&mut self, metres: f32) -> EngineResult<()> {
        if !(metres.is_finite() && metres > self.camera.near()) {
            return Err(EngineError::InvalidValue(format!(
                "view distance must be finite and beyond the near plane, got {metres}"
            )));
        }
        self.view_distance = metres;
        self.camera =
            self.camera
                .clone()
                .with_lens(self.camera.fov_y_degrees(), self.camera.near(), metres);
        Ok(())
    }

    pub fn view_distance(&self) -> f32 {
        self.view_distance
    }

    /// Enable GPU procedural terrain (clipmap). Replaces any previous proc terrain.
    /// Its texture layers start generating in the background right away, so
    /// they are ready by the first frame.
    pub fn set_proc_terrain(&mut self, terrain: ProcTerrain) {
        if let Some(layers) = &terrain.layers {
            let list: Vec<crate::texgen::Layer> =
                layers.as_array().iter().map(|&l| l.into()).collect();
            crate::texgen::prepare_in_background(&list);
        }
        self.height_field = Some(HeightField::new(terrain.rules.clone()));
        self.proc_terrain = Some(terrain);
    }

    /// Clear GPU procedural terrain (mesh-only mode).
    pub fn clear_proc_terrain(&mut self) {
        self.proc_terrain = None;
        self.height_field = None;
    }

    /// Update clipmap focus (usually the walker position).
    pub fn set_proc_focus(&mut self, focus: impl Into<Vec3>) {
        let focus = focus.into();
        if let Some(t) = self.proc_terrain.as_mut() {
            t.focus = focus;
        }
    }

    /// Height from the GPU terrain formula (CPU portable noise). Falls back to 0.
    pub fn proc_height_at(&self, x: f32, z: f32) -> f32 {
        self.height_field
            .as_ref()
            .map(|f| f.height_at(x, z))
            .unwrap_or(0.0)
    }

    /// Walkable height matching the rendered clipmap surface (finest-ring triangles).
    pub fn proc_walk_height(&self, x: f32, z: f32) -> f32 {
        match (&self.height_field, &self.proc_terrain) {
            (Some(field), Some(proc)) => {
                field.walk_height_on_clipmap(x, z, &proc.config, proc.focus)
            }
            (Some(field), None) => field.height_at(x, z),
            _ => 0.0,
        }
    }

    pub(crate) fn proc_terrain(&self) -> Option<&ProcTerrain> {
        self.proc_terrain.as_ref()
    }

    #[allow(dead_code)]
    pub(crate) fn height_field(&self) -> Option<&HeightField> {
        self.height_field.as_ref()
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
