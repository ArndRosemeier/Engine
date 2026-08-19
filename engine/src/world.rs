use crate::anim::{AnimatedModel, Animator};
use crate::camera::Camera;
use crate::collision::{ActorBody, ActorMove, CollisionWorld};
use crate::color::Color;
use crate::contact::ContactSnapshot;
use crate::error::{EngineError, EngineResult};
use crate::input::Input;
use crate::limits::EngineLimits;
use crate::mesh::{AlbedoMap, BuiltMesh, Mesh};
use crate::place::{GlobalPlace, Place};
use crate::portal::{
    opening_extents, segment_crosses_opening, teleport_yaw, PortalLink, PortalPlane, PortalView,
    SpaceId,
};
use crate::proc_terrain::{HeightField, ProcTerrain};
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

/// Opaque handle to a spawned entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(u64);

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

/// How instanced meshes reach the GPU.
///
/// Both paths are always compiled. The selected one is the only one that runs;
/// there is no silent fallback from GPU to CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InstanceSubmit {
    /// Entity-sphere cull, then `draw_indexed` of every instance.
    #[default]
    CpuIndexed,
    /// Prototype-batched per-instance GPU compact, then `draw_indexed_indirect`.
    GpuIndirect,
}

impl InstanceSubmit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuIndexed => "cpu_indexed",
            Self::GpuIndirect => "gpu_indirect",
        }
    }
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
    /// `1 - exp(-(d·k)²)` reaches 0.98 at `d = visibility`, which is where a
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
    /// Per-instance transforms for GPU instancing.
    pub(crate) instances: Vec<Mat4>,
    /// Minimum GPU slot capacity requested by the owner. This reserves space
    /// without drawing placeholder instances.
    pub(crate) instance_reserve: usize,
    /// Whether `instances` is authoritative, even when it is empty.
    ///
    /// An instanced entity with nothing to place draws nothing; a plain entity
    /// with no instances draws once at `transform`.
    pub(crate) instanced: bool,
    pub(crate) material: Option<SurfaceMaterialRef>,
    /// Optional baked albedo sampled with mesh UVs. `None` uses a white 1×1.
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
    pub camera: Camera,
    pub light: Light,
    pub clear_color: Color,
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
    pub(crate) materials: HashMap<MaterialId, TerrainMaterial>,
    next_texture_id: u64,
    next_material_id: u64,
    pub(crate) water_materials: HashMap<WaterMaterialId, WaterMaterial>,
    /// Changes whenever textures or material resources are added.
    resource_epoch: u64,
    /// Applied to fully-opaque `set_chunk_built` uploads when set.
    pub(crate) default_terrain_material: Option<MaterialId>,
    /// Applied to fully-translucent chunk layers when set.
    pub(crate) default_water_material: Option<WaterMaterialId>,
    /// Whether the game wants the pointer pinned for mouse-look.
    pointer_lock: bool,
    /// Seconds since start, for animated materials.
    time: f32,
    /// How far the camera helpers put their far plane.
    view_distance: f32,
    /// The air, when the game wants any.
    haze: Option<Haze>,
    /// Procedural sky behind the scene, when the game wants one.
    sky: Option<Sky>,
    /// Nearby mesh CSM + height raymarch. `None` disables shadows.
    shadows: Option<ShadowSettings>,
    /// Instanced mesh submit path. Default stays CPU so Engine demos match today.
    instance_submit: InstanceSubmit,
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
    portals: Vec<PortalLink>,
    travel_prev: Option<(SpaceId, Vec3)>,
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
            materials: HashMap::new(),
            next_texture_id: 1,
            next_material_id: 1,
            water_materials: HashMap::new(),
            resource_epoch: 1,
            default_terrain_material: None,
            default_water_material: None,
            pointer_lock: false,
            time: 0.0,
            view_distance: Camera::default().far,
            haze: None,
            sky: None,
            shadows: Some(ShadowSettings::default()),
            instance_submit: InstanceSubmit::CpuIndexed,
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
            travel_prev: None,
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
    /// Only colliders in [`Self::living_in`] participate.
    pub fn move_actor(&self, body: &ActorBody, from: GlobalXZ, dx: f64, dz: f64) -> GlobalXZ {
        self.collision.move_in(self.live_space, body, from, dx, dz)
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

    /// Two openings become the same hole. Both meshes should face their own
    /// space (`+Z` local). Walking through teleports; looking through shows
    /// the other side as if the world continued.
    pub fn link(&mut self, a: EntityId, b: EntityId) -> EngineResult<()> {
        self.link_with_view(a, b, PortalView::Transformed)
    }

    /// Link two openings with a stable destination view at the threshold.
    ///
    /// `eye_offset_y` is measured above the opening centre. `setback` places
    /// the virtual eye just behind the destination plane.
    pub fn link_threshold_view(
        &mut self,
        a: EntityId,
        b: EntityId,
        eye_offset_y: f32,
        setback: f32,
    ) -> EngineResult<()> {
        if !eye_offset_y.is_finite() || !setback.is_finite() || setback <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "portal threshold view needs finite eye offset and positive setback, got ({eye_offset_y}, {setback})"
            )));
        }
        self.link_with_view(
            a,
            b,
            PortalView::Threshold {
                eye_offset_y,
                setback,
            },
        )
    }

    fn link_with_view(&mut self, a: EntityId, b: EntityId, view: PortalView) -> EngineResult<()> {
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
        if self.is_portal_surface(a) || self.is_portal_surface(b) {
            return Err(EngineError::InvalidValue(
                "opening is already linked".into(),
            ));
        }
        self.portals.push(PortalLink { a, b, view });
        self.bump_render_epoch();
        Ok(())
    }

    /// Drop a previously linked pair. The opening entities stay; walking
    /// through them no longer teleports until they are linked again.
    pub fn unlink(&mut self, a: EntityId, b: EntityId) -> EngineResult<()> {
        let before = self.portals.len();
        self.portals
            .retain(|p| !((p.a == a && p.b == b) || (p.a == b && p.b == a)));
        if self.portals.len() == before {
            return Err(EngineError::InvalidValue(
                "those openings are not linked".into(),
            ));
        }
        self.bump_render_epoch();
        Ok(())
    }

    pub fn portals(&self) -> &[PortalLink] {
        &self.portals
    }

    /// Closest opening in the live space that the camera is facing.
    pub(crate) fn visible_portal(
        &self,
        eye: Vec3,
        look: Vec3,
    ) -> Option<crate::portal::VisiblePortal> {
        let look = {
            if look.length_squared() <= 0.0 {
                panic!("visible_portal look direction is zero");
            }
            look.normalize()
        };
        let mut best: Option<(f32, crate::portal::VisiblePortal)> = None;
        for link in &self.portals {
            let (Some(a), Some(b)) = (self.entities.get(&link.a), self.entities.get(&link.b))
            else {
                continue;
            };
            for (src, dst) in link.directions(a.space, b.space, self.live_space) {
                let src_e = if src == link.a { a } else { b };
                let dst_e = if dst == link.a { a } else { b };
                let Ok(src_plane) = self.portal_plane(src) else {
                    continue;
                };
                if src_plane.signed_distance(eye) <= 0.05 {
                    continue;
                }
                if look.dot(-src_plane.normal) <= 0.0 {
                    continue;
                }
                let dist = eye.distance_squared(src_plane.center);
                let Ok(dst_plane) = self.portal_plane(dst) else {
                    continue;
                };
                let candidate = crate::portal::VisiblePortal {
                    src,
                    dst,
                    src_transform: src_e.transform,
                    dst_transform: dst_e.transform,
                    dst_center: dst_plane.center,
                    dst_normal: dst_plane.normal,
                    dest_space: dst_e.space,
                    view: link.view,
                };
                if best.as_ref().is_none_or(|(best_dist, _)| dist < *best_dist) {
                    best = Some((dist, candidate));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    pub fn is_portal_surface(&self, id: EntityId) -> bool {
        self.portals.iter().any(|p| p.a == id || p.b == id)
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
        let links: Vec<PortalLink> = self.portals.clone();
        for link in links {
            let a_space = self.entities.get(&link.a).map(|e| e.space);
            let b_space = self.entities.get(&link.b).map(|e| e.space);
            let (Some(a_space), Some(b_space)) = (a_space, b_space) else {
                continue;
            };
            for (src, dst) in link.directions(a_space, b_space, live) {
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
                *yaw_degrees = teleport_yaw(*yaw_degrees, t_src, t_dst);
                self.live_space = dst_e.space;
                self.travel_prev = Some((self.live_space, *position));
                return Some(self.live_space);
            }
        }
        self.travel_prev = Some((live, *position));
        None
    }

    /// Spawn a mesh at a friendly [`Place`].
    pub fn place(&mut self, mesh: Mesh, place: Place) -> EngineResult<EntityId> {
        let albedo = mesh.albedo().cloned();
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
        let instances: Vec<Mat4> = places.into_iter().map(Place::to_matrix).collect();
        let albedo = mesh.albedo().cloned();
        let id = self.spawn_built_instanced(mesh.build(), instances);
        self.attach_albedo(id, albedo)?;
        Ok(id)
    }

    /// Spawn an instanced entity that starts with nothing placed.
    ///
    /// The mesh is uploaded once; [`Self::set_instances`] then drives where it
    /// appears.
    pub fn spawn_instanced(&mut self, mesh: Mesh) -> EntityId {
        let albedo = mesh.albedo().cloned();
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
        instances: Vec<Mat4>,
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

    fn attach_albedo(&mut self, id: EntityId, albedo: Option<AlbedoMap>) -> EngineResult<()> {
        let Some(map) = albedo else {
            return Ok(());
        };
        let tid = self.create_texture_rgba(map.width, map.height, map.rgba)?;
        self.entities
            .get_mut(&id)
            .expect("entity was just spawned")
            .albedo = Some(tid);
        self.bump_render_epoch();
        Ok(())
    }

    pub fn despawn(&mut self, id: EntityId) {
        let removed_static = self.entities.remove(&id).is_some();
        self.order.retain(|x| *x != id);
        self.animated.remove(&id);
        self.animated_order.retain(|x| *x != id);
        self.chunk_entities.retain(|_, eid| *eid != id);
        self.anchored_entities.remove(&id);
        self.anchored_chunks.retain(|_, c| c.entity != id);
        let linked = self.portals.iter().any(|p| p.a == id || p.b == id);
        self.portals.retain(|p| p.a != id && p.b != id);
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
        let albedo = mesh.albedo().cloned();
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
            // Remesh needs a new GPU upload — swap entity id so the renderer rebuilds.
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
    /// the cursor without the player having asked for it — a grab taken on
    /// startup follows them out of the window.
    pub fn set_pointer_lock(&mut self, locked: bool) {
        self.pointer_lock = locked;
    }

    pub fn pointer_lock(&self) -> bool {
        self.pointer_lock
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
        let fov = self.camera.fov_y_degrees;
        let near = self.camera.near;
        self.camera = Camera::look_at(e, t);
        self.camera.fov_y_degrees = fov;
        self.camera.near = near;
        self.camera.far = self.view_distance;
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

    pub fn set_place(&mut self, id: EntityId, place: Place) -> EngineResult<()> {
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform = place.to_matrix();
            e.bump_xform();
            self.bump_render_epoch();
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.transform = place.to_matrix();
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
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

    pub fn play_animation(&mut self, id: EntityId, clip_name: &str) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.play(clip_name)
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
        e.animator.speed = speed;
        Ok(())
    }

    /// Advance all skinned clip clocks (call once per frame before render sync).
    pub fn tick_animations(&mut self, dt: f32) {
        for e in self.animated.values_mut() {
            e.animator.tick(dt);
        }
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
        self.camera.far = self.view_distance;
    }

    /// Set sun direction (need not be normalized) and ambient 0..1.
    pub fn set_sun(&mut self, direction: impl Into<Vec3>, ambient: f32) {
        self.light.direction = direction.into();
        self.light.ambient = ambient.clamp(0.0, 1.0);
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

    pub fn entity(&self, id: EntityId) -> EngineResult<&Entity> {
        self.entities.get(&id).ok_or(EngineError::UnknownEntity)
    }

    pub fn entity_mut(&mut self, id: EntityId) -> EngineResult<&mut Entity> {
        self.entities.get_mut(&id).ok_or(EngineError::UnknownEntity)
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
            // Remesh requires a new GPU upload — swap entity id so renderer rebuilds.
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
        e.instances.extend(places.iter().map(|p| p.to_matrix()));
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
        self.camera.far = self.view_distance;
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
        self.camera.far = self.view_distance;
    }

    /// How far the camera helpers may see, in metres.
    ///
    /// The helpers rebuild the camera every frame, so a game that reaches into
    /// `world.camera.far` has it overwritten on the next one; this is the knob
    /// that survives. Depth is reversed, so a horizon-scale distance costs
    /// nothing in precision.
    pub fn set_view_distance(&mut self, metres: f32) -> EngineResult<()> {
        if !(metres.is_finite() && metres > self.camera.near) {
            return Err(EngineError::InvalidValue(format!(
                "view distance must be finite and beyond the near plane, got {metres}"
            )));
        }
        self.view_distance = metres;
        self.camera.far = metres;
        Ok(())
    }

    pub fn view_distance(&self) -> f32 {
        self.view_distance
    }

    /// Enable GPU procedural terrain (clipmap). Replaces any previous proc terrain.
    pub fn set_proc_terrain(&mut self, terrain: ProcTerrain) {
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
}
