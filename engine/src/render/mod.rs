mod audit;
mod clipmap;
mod crowd;
mod decals;
pub(crate) mod frustum;
mod geometry;
mod gi;
mod gpu_mesh;
mod gpu_profile;
mod grass;
mod instance_cull;
mod lights;
mod material;
mod occlusion;
mod particles_gpu;
mod pipeline;
pub(crate) mod pipeline_cache;
mod portal;
mod postprocess;
mod reflect;
mod ribbons_gpu;
mod shadow;
mod skinned;
mod sky_pipeline;
mod ssao;
mod stencil;
pub(crate) mod taa;
mod terrain_pipeline;
mod terrain_shadow;
mod voxel_terrain;
mod water_gpu;
mod water_pipeline;

use crate::camera::Camera;
use crate::mesh::InstanceRaw;
use crate::portal::{
    oblique_view_projection, portal_destination_clip, portal_render_camera, portal_view_camera,
    SpaceId,
};
use crate::texture::{MaterialId, TextureId, WaterMaterialId};
use crate::world::{Entity, EntityId, InstanceSubmit, ShadowSettings, SurfaceMaterialRef, World};
use audit::audit_note;
use clipmap::ClipmapRenderer;
use frustum::{Bounds, Frustum, Visibility};
use gpu_mesh::GpuMesh;
use gpu_profile::{GpuProfiler, Mark, SceneMark};
use instance_cull::{
    ActiveInstanceRange, CullDraws, CullJob, InstanceCull, OPAQUE_INDIRECT_OFFSET,
    TRANSLUCENT_INDIRECT_OFFSET, WORKGROUP,
};
use particles_gpu::ParticleGpu;
use pipeline::{create_pipelines, Pipelines, Uniforms};
use portal::PortalGpu;
use postprocess::{Postprocess, HDR_FORMAT};
use ribbons_gpu::RibbonGpu;
use shadow::{material_casts_shadow, ShadowGpu};
use skinned::{
    create_skinned_pipelines, joint_bind_layout, GpuSkinnedEntity, GpuSkinnedMesh, SkinnedPipelines,
};
use sky_pipeline::{create_sky_pipelines, SkyPipelines, SkyUniforms};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use terrain_pipeline::{
    build_terrain_material, create_terrain_pipelines, upload_texture, GpuTerrainMaterial,
    GpuTexture, TerrainPipelines,
};
use water_pipeline::{
    build_water_material, create_water_pipelines, GpuWaterMaterial, WaterPipelines,
};
use winit::dpi::PhysicalSize;

/// Depth is reversed: near is 1, far is 0, so a horizon-scale far plane keeps
/// its precision (see [`crate::camera::Camera::projection_matrix`]). Every
/// pipeline compares this way and the pass clears to [`DEPTH_CLEAR`]; the
/// three must agree or the world draws inside out.
/// GPU work counted during sync + draw, for the hitch log.
#[derive(Clone, Debug, Default)]
pub struct GpuFrameStats {
    pub mesh_new: u32,
    pub mesh_rebuild: u32,
    pub instance_rewrites: u32,
    pub instance_batches: u32,
    pub batch_rewrites: u32,
    pub batch_partial_writes: u32,
    pub skinned_new: u32,
    pub skinned_model_uploads: u32,
    pub skinned_pose_writes: u32,
    pub shadow_atlas: bool,
    /// Terrain horizon-map rebuilds this frame (0 or 1).
    pub terrain_horizon_rebuilds: u32,
    pub entities: u32,
    pub animated: u32,
    pub instance_submit: InstanceSubmit,
    pub compute_culls: u32,
    pub indirect_draws: u32,
    pub direct_inside_batches: u32,
    pub boundary_batches: u32,
    pub cull_instances: u64,
    pub cull_workgroups: u64,
    /// Meshes the occlusion pass tested, and how many it found hidden
    /// (from the newest answer read back, a frame or two old).
    pub occlusion_tested: u32,
    pub occlusion_hidden: u32,
    /// Sun cascades drawn anew this frame (0 to 3).
    pub shadow_cascades_drawn: u32,
    /// Of those, how many reused their cached static casters.
    pub shadow_cache_hits: u32,
}

impl GpuFrameStats {
    pub fn sync_line(&self) -> String {
        format!(
            "mesh_new={} rebuild={} inst={} batches={} batch_rewrites={} batch_partial={} skinned_new={} skinned_model={} pose={} entities={} animated={}",
            self.mesh_new,
            self.mesh_rebuild,
            self.instance_rewrites,
            self.instance_batches,
            self.batch_rewrites,
            self.batch_partial_writes,
            self.skinned_new,
            self.skinned_model_uploads,
            self.skinned_pose_writes,
            self.entities,
            self.animated
        )
    }

    pub fn draw_line(&self) -> String {
        format!(
            "shadow_atlas={} cascades={} cache_hits={} horizon={} submit={} inside={} boundary={} tested={} groups={} indirect={} occlusion_tested={} occlusion_hidden={}",
            self.shadow_atlas,
            self.shadow_cascades_drawn,
            self.shadow_cache_hits,
            self.terrain_horizon_rebuilds,
            self.instance_submit,
            self.direct_inside_batches,
            self.boundary_batches,
            self.cull_instances,
            self.cull_workgroups,
            self.indirect_draws,
            self.occlusion_tested,
            self.occlusion_hidden,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshSubmit {
    Direct,
    Compacted,
}

#[derive(Clone, Copy, Debug)]
enum CullView {
    Main,
    Shadow,
}

/// Below this many opaque draws they are recorded straight into the pass.
const OPAQUE_BUNDLE_MIN_DRAWS: usize = 256;
/// Opaque draws per bundle recorded on one thread.
const OPAQUE_BUNDLE_CHUNK: usize = 128;

/// How one opaque draw is submitted.
#[derive(Clone, Copy)]
enum OpaqueSubmit<'r> {
    /// Straight or GPU-compacted instances (`submit_mesh_draw`).
    Mesh(MeshSubmit),
    /// Through its occlusion draw (`submit_occluded_draw`): args, slot.
    Occluded(&'r wgpu::Buffer, u32),
}

/// One opaque mesh draw, with everything it binds already looked up.
struct OpaqueDraw<'r> {
    albedo: &'r wgpu::BindGroup,
    gpu: &'r GpuMesh,
    submit: OpaqueSubmit<'r>,
}

impl<'r> OpaqueDraw<'r> {
    /// The buffers the draw reads: instances, and indirect args with their
    /// byte offset when the GPU decides the count.
    fn sources(&self) -> (&'r wgpu::Buffer, Option<(&'r wgpu::Buffer, u64)>) {
        let g = self.gpu;
        match self.submit {
            OpaqueSubmit::Mesh(MeshSubmit::Compacted) => (
                &g.compact_buf,
                Some((&g.indirect_buf, OPAQUE_INDIRECT_OFFSET)),
            ),
            OpaqueSubmit::Mesh(MeshSubmit::Direct) => (&g.instance_buf, None),
            OpaqueSubmit::Occluded(args, slot) => (
                &g.instance_buf,
                Some((
                    args,
                    occlusion::OcclusionFrame::offset(slot, OPAQUE_INDIRECT_OFFSET),
                )),
            ),
        }
    }

    /// Record the draw: the same commands `submit_mesh_draw` and
    /// `submit_occluded_draw` give a pass.
    fn record(&self, bundle: &mut wgpu::RenderBundleEncoder<'r>) {
        let g = self.gpu;
        let (instances, indirect) = self.sources();
        bundle.set_bind_group(1, self.albedo, &[]);
        bundle.set_vertex_buffer(0, g.vertex_buf.slice(..));
        match self.submit {
            OpaqueSubmit::Mesh(MeshSubmit::Compacted) => {
                bundle.set_vertex_buffer(1, instances.slice(..))
            }
            _ => bundle.set_vertex_buffer(1, g.instance_slice()),
        }
        bundle.set_index_buffer(g.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        match indirect {
            Some((args, offset)) => bundle.draw_indexed_indirect(args, offset),
            None => bundle.draw_indexed(
                g.opaque_indices(),
                g.base_vertex,
                0..g.instance_count as u32,
            ),
        }
    }
}

/// Everything one recorded opaque draw baked into its bundle. Equal keys
/// mean the bundle would be recorded the same, so it is used again.
#[derive(PartialEq)]
struct OpaqueKey {
    albedo: wgpu::BindGroup,
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    /// Where the mesh starts in the arena pages: first index, base vertex,
    /// first instance and instance room.
    offsets: (u32, i32, u32, usize),
    instances: wgpu::Buffer,
    /// Indirect args and offset, or the direct index and instance counts.
    indirect: Option<(wgpu::Buffer, u64)>,
    counts: (u32, u32),
}

impl OpaqueKey {
    fn matches(&self, d: &OpaqueDraw<'_>) -> bool {
        let g = d.gpu;
        let (instances, indirect) = d.sources();
        self.albedo == *d.albedo
            && self.vertex == g.vertex_buf
            && self.index == g.index_buf
            && self.offsets
                == (
                    g.first_index,
                    g.base_vertex,
                    g.first_instance,
                    g.instance_capacity,
                )
            && self.instances == *instances
            && match (&self.indirect, indirect) {
                (Some((a, o)), Some((b, p))) => a == b && *o == p,
                (None, None) => {
                    self.counts == (g.opaque_index_count as u32, g.instance_count as u32)
                }
                _ => false,
            }
    }

    fn of(d: &OpaqueDraw<'_>) -> Self {
        let g = d.gpu;
        let (instances, indirect) = d.sources();
        Self {
            albedo: d.albedo.clone(),
            vertex: g.vertex_buf.clone(),
            index: g.index_buf.clone(),
            offsets: (
                g.first_index,
                g.base_vertex,
                g.first_instance,
                g.instance_capacity,
            ),
            instances: instances.clone(),
            indirect: indirect.map(|(b, o)| (b.clone(), o)),
            counts: if indirect.is_some() {
                (0, 0)
            } else {
                (g.opaque_index_count as u32, g.instance_count as u32)
            },
        }
    }
}

/// One chunk of recorded opaque draws: its keys and its bundle.
struct OpaqueChunk {
    keys: Vec<OpaqueKey>,
    bundle: wgpu::RenderBundle,
}

/// Which opaque pass a set of bundles is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OpaqueTarget {
    /// The depth prepass (phase `All` or `Seen`).
    Prepass(PrepassPhase),
    /// The top-level world pass.
    World,
}

/// The top-level view's opaque mesh draws for one pass as bundles, kept
/// from frame to frame: a chunk whose draws bake in the same buffers and
/// bind groups is used again, the others are recorded anew on the frame's
/// threads.
#[derive(Default)]
struct OpaqueBundles {
    /// What every chunk shares (scene bind group, pipeline, samples); when
    /// it changes every chunk is recorded anew.
    shared: Option<(wgpu::BindGroup, wgpu::RenderPipeline, u32)>,
    chunks: Vec<OpaqueChunk>,
    /// This frame's use: what the chunks were checked for, and how many.
    frame: Option<OpaqueFrame>,
}

struct OpaqueFrame {
    target: OpaqueTarget,
    space: SpaceId,
    uniform_slot: usize,
    chunks: usize,
    indirect_draws: u32,
}

struct MainCullSelection {
    ids: Vec<EntityId>,
    compacted: HashSet<EntityId>,
    /// This frame's occlusion draws, when occlusion culling is on.
    occlusion: Option<occlusion::OcclusionFrame>,
}

impl MainCullSelection {
    fn none() -> Self {
        Self {
            ids: Vec::new(),
            compacted: HashSet::new(),
            occlusion: None,
        }
    }

    /// The occlusion slot `id` draws through in the main view, if any.
    fn occlusion_slot(&self, id: EntityId) -> Option<(&wgpu::Buffer, u32)> {
        let frame = self.occlusion.as_ref()?;
        frame.slots.get(&id).map(|&slot| (&frame.args, slot))
    }
}

/// Which part of the depth prepass to draw (see `occlusion.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrepassPhase {
    /// Everything (occlusion culling off).
    All,
    /// Everything but the occlusion candidates not seen lately.
    Seen,
    /// Only the candidates seen lately: without AO nothing else needs the
    /// prepass, and the world pass then keeps writing depth.
    SeenOccluders,
    /// Only those candidates, through the occlusion draws.
    Unseen,
}

/// Which space a portal level draws, how deep it recurses, and how the scene
/// is viewed from there.
struct PortalLevel {
    space: SpaceId,
    /// Portal recursion depth.
    depth: u8,
    view: PortalLevelView,
}

/// How a portal level's scene is viewed: clip plane, environment, culling.
struct PortalLevelView {
    clip_plane: Option<(glam::Vec3, glam::Vec3)>,
    draw_environment: bool,
    use_cull: bool,
}

/// Which pass of the scene is being drawn and with which uniform slot.
#[derive(Clone, Copy)]
struct ScenePass {
    /// Drawing seen through a stencil-marked portal opening.
    portal_mask: bool,
    /// Immutable scene/sky uniform slot for this encoded view.
    uniform_slot: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ShadowDraw {
    id: EntityId,
    /// Which group of shadow sources it is in (its source index /
    /// [`SHADOW_GROUP`]); a view's mesh bundles are recorded per group.
    group: u32,
    submit: MeshSubmit,
    /// Textured casters go through the alpha-tested shadow pipeline.
    albedo: Option<TextureId>,
}

#[derive(Clone)]
struct ShadowCullSelection {
    draws: Vec<ShadowDraw>,
    batch_draws: Vec<ShadowDraw>,
    compact_batches: Vec<EntityId>,
}

/// Shadow mesh sources per bundle group: one changed caster re-records
/// only its group's bundle, not the whole view's.
const SHADOW_GROUP: usize = 128;

/// Bundle slots of the sun cascades' static casters (cascade `c` at
/// `STATIC_SHADOW_SLOT + c`), after the live views' slots.
const STATIC_SHADOW_SLOT: usize = 3 + lights::SHADOW_LAYERS;
/// A caster counts as static once it has not changed for this many frames.
const STATIC_SHADOW_SETTLE: u64 = 30;
/// A cached static layer is drawn anew after this many frames regardless.
const STATIC_SHADOW_MAX_AGE: u64 = 900;

/// The bundle slot's light view: its cascade or lamp layer.
fn shadow_view_of(slot: usize) -> usize {
    if slot >= STATIC_SHADOW_SLOT {
        slot - STATIC_SHADOW_SLOT
    } else {
        slot
    }
}

/// How a sun cascade's map is drawn this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShadowCacheUse {
    /// Every caster straight into the map (no cache).
    Plain,
    /// The cached static layer is copied in; only moving casters are drawn.
    Hit,
    /// The static layer is drawn anew, copied in, then moving casters.
    Miss,
}

/// One shadow view's recorded casters: mesh bundles per source group, and
/// one bundle of instance batches (which change more often).
#[derive(Default)]
struct ShadowViewBundles {
    groups: Vec<ShadowGroupBundle>,
    batch_draws: Vec<ShadowDraw>,
    batches: Option<wgpu::RenderBundle>,
}

struct ShadowGroupBundle {
    group: u32,
    draws: Vec<ShadowDraw>,
    bundle: wgpu::RenderBundle,
}

struct GpuInstanceBatch {
    gpu: GpuMesh,
    slots: Vec<BatchSlot>,
    /// Unused instance ranges (filled with [`DEAD_INSTANCE`]), sorted.
    free: Vec<std::ops::Range<usize>>,
    /// End of the highest slot; everything after it is free.
    used_end: usize,
    active_ranges: Vec<ActiveInstanceRange>,
}

/// Fills unused instance entries: zero scale, far below the world, so the
/// GPU cull drops it and a direct draw makes no pixels.
const DEAD_INSTANCE: glam::Mat4 = glam::Mat4::from_cols(
    glam::Vec4::ZERO,
    glam::Vec4::ZERO,
    glam::Vec4::ZERO,
    glam::Vec4::new(0.0, -1.0e9, 0.0, 1.0),
);

impl GpuInstanceBatch {
    fn rebuild_active_ranges(&mut self) {
        self.active_ranges = active_instance_ranges(&self.slots);
    }
}

/// Take `len` entries from a sorted free list (first fit).
fn take_free(free: &mut Vec<std::ops::Range<usize>>, len: usize) -> Option<usize> {
    let i = free.iter().position(|r| r.end - r.start >= len)?;
    let start = free[i].start;
    free[i].start += len;
    if free[i].is_empty() {
        free.remove(i);
    }
    Some(start)
}

/// Return a range to a sorted free list, merging neighbours.
fn give_free(free: &mut Vec<std::ops::Range<usize>>, r: std::ops::Range<usize>) {
    if r.is_empty() {
        return;
    }
    let i = free.partition_point(|f| f.start < r.start);
    free.insert(i, r);
    if i + 1 < free.len() && free[i].end == free[i + 1].start {
        free[i].end = free[i + 1].end;
        free.remove(i + 1);
    }
    if i > 0 && free[i - 1].end == free[i].start {
        free[i - 1].end = free[i].end;
        free.remove(i);
    }
}

impl GpuInstanceBatch {
    fn has_holes(&self) -> bool {
        !self.free.is_empty()
            || self
                .slots
                .iter()
                .any(|slot| slot.active_count < slot.capacity)
    }
}

struct BatchSlot {
    id: EntityId,
    xform_rev: u64,
    start: usize,
    capacity: usize,
    active_count: usize,
    bounds: Option<Bounds>,
}

/// Instance ranges to cull or draw. Unused entries hold [`DEAD_INSTANCE`]
/// (dropped by the cull, invisible when drawn), so slots are merged into
/// one range unless that would add mostly dead entries: one dispatch per
/// batch instead of one per streamed tile.
fn active_instance_ranges(slots: &[BatchSlot]) -> Vec<ActiveInstanceRange> {
    let mut live: Vec<(usize, usize)> = slots
        .iter()
        .filter(|slot| slot.active_count > 0)
        .map(|slot| (slot.start, slot.start + slot.active_count))
        .collect();
    live.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in live {
        match merged.last_mut() {
            // Bridge gaps of dead entries up to the size of what is joined.
            Some(last) if a <= last.1 + (last.1 - last.0).max(b - a).max(256) => {
                last.1 = last.1.max(b)
            }
            _ => merged.push((a, b)),
        }
    }
    merged
        .into_iter()
        .map(|(a, b)| ActiveInstanceRange {
            start: u32::try_from(a).expect("instance batch start exceeds GPU u32 range"),
            count: u32::try_from(b - a).expect("instance batch active count exceeds GPU u32 range"),
        })
        .collect()
}

#[derive(Default)]
struct BatchBuild {
    sources: Vec<BatchBuildSource>,
}

#[derive(Clone, Copy)]
struct BatchBuildSource {
    id: EntityId,
    xform_rev: u64,
    instance_count: usize,
    instance_reserve: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchSourceState {
    id: EntityId,
    prototype: EntityId,
    xform_rev: u64,
    instance_count: usize,
    instance_reserve: usize,
    material: Option<SurfaceMaterialRef>,
    albedo: Option<TextureId>,
    casts_shadow: bool,
    prototype_material: Option<SurfaceMaterialRef>,
    prototype_albedo: Option<TextureId>,
    prototype_casts_shadow: bool,
}

/// Reversed-Z. `GreaterEqual` so the world pass accepts the depth the prepass
/// laid for the same surface.
pub(crate) const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::GreaterEqual;
pub(crate) const DEPTH_CLEAR: f32 = 0.0;
pub(crate) use shadow::ELEMENT_STATE_WORDS;

pub(crate) use stencil::depth_format;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SceneUniformTarget {
    /// The live camera's view: the one the depth prepass and SSAO see.
    Main,
    /// Views through portals, where the occlusion texture does not apply.
    Portal,
    /// Reflection probe faces: no occlusion texture, but the GI probes hold.
    Probe,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    config: wgpu::SurfaceConfiguration,
    /// `ENGINE_PRESENT_MODE` pins presentation for stress tests; when set it
    /// overrides the world's vsync preference.
    env_present_mode: Option<wgpu::PresentMode>,
    /// Last vsync preference applied to the surface, so capability queries
    /// and reconfiguration only run when the preference changes.
    synced_vsync: Option<bool>,
    /// When the last swapchain image was acquired, and the time since the
    /// one before (the display's frame rhythm).
    last_acquire: Option<std::time::Instant>,
    acquire_interval: Option<f32>,
    pipelines: Pipelines,
    terrain: TerrainPipelines,
    water: WaterPipelines,
    skinned: SkinnedPipelines,
    crowd: crowd::CrowdGpu,
    sky: SkyPipelines,
    shadow: ShadowGpu,
    shadow_vp: [glam::Mat4; 3],
    /// Which sun cascades are drawn anew this frame (the others keep their
    /// maps; see `World::set_shadow_cascade_refresh`).
    shadow_draw: [bool; 3],
    /// Frames synced so far.
    frame_no: u64,
    /// When each mesh entity or batch prototype last changed on the GPU
    /// (`frame_no`); casters changed lately are not cached as static.
    moved_at: HashMap<EntityId, u64>,
    /// Bumped by changes to static shadow content not tied to one caster
    /// (building elements, rebuilt bundles).
    static_shadow_epoch: u64,
    clipmap: Option<ClipmapRenderer>,
    /// Ground pipelines, built on a worker thread at startup.
    clipmap_pipelines: pipeline_cache::Prewarm<std::sync::Arc<clipmap::ClipmapPipelines>>,
    /// Compiled pipelines saved to disk (Vulkan); `None` where unsupported.
    pipeline_cache: Option<pipeline_cache::DiskCache>,
    /// Set once every startup worker finished and the cache was saved.
    pipeline_cache_settled: bool,
    portal: PortalGpu,
    depth_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    water_gpu: water_gpu::WaterGpu,
    /// Scene uniform slot of the main view this frame.
    main_uniform_slot: usize,
    /// Ambient occlusion from the depth prepass.
    ssao: ssao::Ssao,
    reflect: reflect::Reflect,
    /// Aspect for views other than the window's (reflection probe faces).
    view_aspect: Option<f32>,
    /// This frame ran the depth prepass and SSAO.
    ao_active: bool,
    /// Voxel scene and light probes for global illumination.
    gi: gi::GiVolume,
    voxel: voxel_terrain::VoxelTerrainGpu,
    gpu_meshes: HashMap<EntityId, GpuMesh>,
    /// Every mesh's vertices and indices, in shared pages.
    arenas: geometry::MeshArenas,
    /// GPU-mode instance buffers, one per prototype instead of one per spatial bin.
    gpu_instance_batches: HashMap<EntityId, GpuInstanceBatch>,
    gpu_batch_sources: Vec<BatchSourceState>,
    gpu_batch_source_scratch: Vec<BatchSourceState>,
    gpu_skinned: HashMap<EntityId, GpuSkinnedEntity>,
    /// Vertex/index buffers keyed by `Arc::as_ptr` of the shared [`AnimatedModel`].
    /// Per model: its uploaded meshes and morph table, shared by every entity of it.
    gpu_skinned_meshes: HashMap<usize, (Vec<GpuSkinnedMesh>, wgpu::Buffer)>,
    /// Characters' maps, one GPU copy per content (see [`material::MapCache`]).
    skinned_maps: material::MapCache,
    gpu_textures: HashMap<TextureId, GpuTexture>,
    gpu_mesh_albedo: HashMap<TextureId, wgpu::BindGroup>,
    audit_frame: u64,
    /// The world's re-upload requests already carried out.
    reupload_seen: (u64, u64),
    gpu_check_seen: u64,
    rebuild_seen: (u64, u64),
    gpu_materials: HashMap<MaterialId, GpuTerrainMaterial>,
    gpu_water_materials: HashMap<WaterMaterialId, GpuWaterMaterial>,
    synced_render_epoch: u64,
    synced_instance_epoch: u64,
    synced_resource_epoch: u64,
    /// Origin the terrain material phases were last written for.
    terrain_origin: crate::space::RenderOrigin,
    /// This frame's view volume, for skipping draws outside it.
    frustum: Frustum,
    size: PhysicalSize<u32>,
    /// Reused while packing instance rows for a dirty entity.
    instance_scratch: Vec<InstanceRaw>,
    draw_source_epoch: u64,
    source_opaque: Vec<EntityId>,
    source_terrain: Vec<EntityId>,
    source_transparent: Vec<EntityId>,
    source_water: Vec<EntityId>,
    batch_source_opaque: Vec<EntityId>,
    batch_source_terrain: Vec<EntityId>,
    batch_source_transparent: Vec<EntityId>,
    batch_source_water: Vec<EntityId>,
    terrain_bundle_ids: Vec<EntityId>,
    terrain_bundle: Option<wgpu::RenderBundle>,
    shadow_source_epoch: u64,
    shadow_mesh_sources: Vec<EntityId>,
    shadow_batch_sources: Vec<EntityId>,
    /// Sun cascades first, then lamp shadow layers.
    shadow_bundles: Vec<ShadowViewBundles>,
    /// The top-level view's opaque draws as bundles, kept from frame to
    /// frame: depth prepass, world pass.
    opaque_prepass: OpaqueBundles,
    opaque_world: OpaqueBundles,
    draw_opaque: Vec<EntityId>,
    draw_terrain: Vec<EntityId>,
    draw_transparent: Vec<EntityId>,
    draw_water: Vec<EntityId>,
    batch_opaque: Vec<EntityId>,
    batch_terrain: Vec<EntityId>,
    batch_transparent: Vec<EntityId>,
    batch_water: Vec<EntityId>,
    gpu_stats: GpuFrameStats,
    next_scene_uniform_slot: usize,
    gpu_profiler: Option<GpuProfiler>,
    instance_cull: InstanceCull,
    /// Skips meshes hidden behind others (see `occlusion.rs`).
    occlusion: occlusion::Occlusion,
    /// Picks the frames `OcclusionMode::Auto` culls in.
    occlusion_auto: occlusion::OcclusionAuto,
    /// Whether this frame culls by occlusion (decided as it starts).
    occlusion_active: bool,
    /// The main view's (jittered) view-projection this frame.
    occlusion_view_proj: glam::Mat4,
    /// This frame filled depth before the world pass (for AO or occlusion).
    prepass_done: bool,
    /// This frame's skinning matrices, one list per animated entity in
    /// `World::animated_entities` order (kept to reuse the memory).
    poses: Vec<Vec<glam::Mat4>>,
    /// Last frame after overlay passes (egui). For capture_png.
    last_color: Option<wgpu::Texture>,
    /// Keep a copy of the next presented frame for `capture_png`. Copying the
    /// whole swapchain image every frame costs bandwidth nobody uses.
    capture_next_frame: bool,
    particles: ParticleGpu,
    pub(crate) cpu_spans: crate::cpu_trace::Spans,
    ribbons: RibbonGpu,
    postprocess: Postprocess,
    taa: taa::Taa,
    /// Frames rendered with TAA on (drives jitter and dither).
    taa_frame: u32,
    /// This frame's camera jitter in NDC (zero with TAA off).
    taa_jitter: glam::Vec2,
    /// Share of the window's width and height the scene is drawn at
    /// (`World::set_render_scale`); below 1 the TAA pass upscales.
    render_scale: f32,
}

impl Renderer {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            // `WGPU_BACKEND=dx12|vulkan|metal` picks one (a driver test).
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            backend_options: dx12_options(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter");

        let gpu_profile_interval = GpuProfiler::requested_interval();
        let portal_depth = wgpu::Features::DEPTH32FLOAT_STENCIL8;
        if !adapter.features().contains(portal_depth) {
            panic!("GPU adapter does not support DEPTH32FLOAT_STENCIL8 (required for portals)");
        }
        let mut required_features = portal_depth | pipeline_cache::wanted_features(&adapter);
        if gpu_profile_interval.is_some() {
            let timestamp_features =
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            if !adapter.features().contains(timestamp_features) {
                panic!("ENGINE_GPU_PROFILE_EVERY requested unsupported GPU timestamp queries");
            }
            required_features |= timestamp_features;
            // Splitting the scene pass needs stamps inside it; optional.
            if adapter
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            {
                required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
            }
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine-device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        let hdr_features = adapter.get_texture_format_features(HDR_FORMAT);
        stencil::init_scene_samples(choose_scene_samples(&adapter));
        let hdr_usages =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        if !hdr_features.allowed_usages.contains(hdr_usages)
            || !hdr_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
            || !hdr_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)
        {
            panic!(
                "GPU adapter does not support filterable, blendable Rgba16Float render targets required for Engine HDR bloom"
            );
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // FIFO is the synchronized default: it prevents startup tearing and
        // keeps normal presentation stable across GPUs and window compositors.
        // Uncapped modes remain available explicitly for rendering stress tests.
        let env_present_mode = match std::env::var("ENGINE_PRESENT_MODE") {
            // Empty counts as unset.
            Ok(value) if value.trim().is_empty() => None,
            Ok(value) => Some(match value.trim() {
                "immediate" => wgpu::PresentMode::Immediate,
                "mailbox" => wgpu::PresentMode::Mailbox,
                "fifo-relaxed" => wgpu::PresentMode::FifoRelaxed,
                "fifo" => wgpu::PresentMode::Fifo,
                _ => panic!(
                    "ENGINE_PRESENT_MODE must be one of: fifo, fifo-relaxed, mailbox, immediate; got {value:?}"
                ),
            }),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("ENGINE_PRESENT_MODE must contain valid Unicode")
            }
        };
        let present_mode = env_present_mode.unwrap_or(wgpu::PresentMode::Fifo);
        if !caps.present_modes.contains(&present_mode) {
            panic!("requested present mode {present_mode:?} is unsupported by this surface");
        }

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            // One frame in flight: with two, a vsynced CPU ran several frames
            // ahead and then stalled, so frame times went 5, 5, 6, 50 ms and
            // movement jumped every fourth frame. One also cuts input lag.
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        // Every pipeline is built here, before the first frame, so no frame
        // stalls on a shader compile. The disk cache makes later launches skip
        // the driver compile too.
        let pipelines_started = std::time::Instant::now();
        let pipeline_cache = pipeline_cache::DiskCache::open(&adapter, &device);
        let joint_layout = joint_bind_layout(&device);
        let mut shadow = ShadowGpu::new(&device, &queue, ShadowSettings::default(), &joint_layout);
        // Parts only some apps use are built on worker threads meanwhile.
        let clipmap_pipelines = {
            let device = device.clone();
            let shadow_layout = shadow.resource_layout.clone();
            pipeline_cache::Prewarm::spawn("clipmap", move || {
                std::sync::Arc::new(clipmap::ClipmapPipelines::new(
                    &device,
                    HDR_FORMAT,
                    &shadow_layout,
                ))
            })
        };
        let (depth_texture, depth_view) = create_depth(&device, config.width, config.height);
        audit_note(&format!(
            "renderer starts at {}x{}, {:?}, {}x MSAA, depth {:?}, {:?}",
            config.width,
            config.height,
            config.present_mode,
            stencil::scene_samples(),
            depth_format(),
            adapter.get_info()
        ));
        // Independent parts compile on worker threads at once: startup
        // costs about the slowest part instead of the sum of all of them.
        let (ssao, reflect, gi, voxel, sky, instance_cull, postprocess, taa, occlusion) =
            std::thread::scope(|s| {
                let (device, queue, depth) = (&device, &queue, &depth_texture);
                let (w, h) = (config.width, config.height);
                let resource_layout = &shadow.resource_layout;
                let ssao = s.spawn(move || ssao::Ssao::new(device, depth, w, h));
                let reflect = s.spawn(move || reflect::Reflect::new(device, queue, depth, w, h));
                let gi = s.spawn(move || gi::GiVolume::new(device, resource_layout));
                let voxel = s.spawn(move || voxel_terrain::VoxelTerrainGpu::new(device));
                let sky = s.spawn(move || create_sky_pipelines(device, HDR_FORMAT));
                let cull = s.spawn(move || InstanceCull::new(device));
                let post = s.spawn(move || Postprocess::new(device, format, w, h));
                let occlusion = s.spawn(move || {
                    let mut o = occlusion::Occlusion::new(device, depth.sample_count());
                    o.resize(device, depth);
                    o
                });
                let taa = taa::Taa::new(device, depth, w, h);
                (
                    join_worker(ssao),
                    join_worker(reflect),
                    join_worker(gi),
                    join_worker(voxel),
                    join_worker(sky),
                    join_worker(cull),
                    join_worker(post),
                    taa,
                    join_worker(occlusion),
                )
            });
        shadow.set_ao_view(&device, ssao.view().clone());
        shadow.set_reflection_views(
            &device,
            reflect.ssr_view().clone(),
            reflect.probe_view().clone(),
        );
        shadow.set_gi_views(&device, gi.probe_views().clone());
        let pipelines = create_pipelines(&device, &queue, HDR_FORMAT, &shadow);
        let (terrain, water, skinned, crowd, portal, particles, ribbons, water_gpu) =
            std::thread::scope(|s| {
                let (device, depth) = (&device, &depth_texture);
                let (bind_layout, albedo_layout) =
                    (&pipelines.bind_layout, &pipelines.albedo_layout);
                let terrain =
                    s.spawn(move || create_terrain_pipelines(device, HDR_FORMAT, bind_layout));
                let water =
                    s.spawn(move || create_water_pipelines(device, HDR_FORMAT, bind_layout));
                let skinned = s.spawn(move || {
                    create_skinned_pipelines(
                        device,
                        HDR_FORMAT,
                        bind_layout,
                        joint_layout,
                        albedo_layout,
                    )
                });
                let crowd = {
                    let cascade_layout = &shadow.cascade_layout;
                    let depth_state = shadow.depth_state.clone();
                    s.spawn(move || {
                        crowd::CrowdGpu::new(
                            device,
                            HDR_FORMAT,
                            bind_layout,
                            albedo_layout,
                            cascade_layout,
                            depth_state,
                        )
                    })
                };
                let portal = s.spawn(move || PortalGpu::new(device, HDR_FORMAT, bind_layout));
                let particles =
                    s.spawn(move || ParticleGpu::new(device, HDR_FORMAT, bind_layout, depth));
                let ribbons = s.spawn(move || RibbonGpu::new(device, HDR_FORMAT, bind_layout));
                let water_gpu =
                    water_gpu::WaterGpu::new(device, bind_layout, depth.sample_count() > 1);
                (
                    join_worker(terrain),
                    join_worker(water),
                    join_worker(skinned),
                    join_worker(crowd),
                    join_worker(portal),
                    join_worker(particles),
                    join_worker(ribbons),
                    water_gpu,
                )
            });
        shadow.init_cutout(&device, &pipelines.albedo_layout);
        let gpu_profiler = gpu_profile_interval.map(|every| {
            GpuProfiler::new(
                &device,
                queue.get_timestamp_period(),
                every,
                device
                    .features()
                    .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            )
        });
        if pipeline_cache::log_enabled() {
            let info = adapter.get_info();
            eprintln!(
                "[engine] pipelines: built in {:.1} ms on {} ({:?}), disk cache {}",
                pipelines_started.elapsed().as_secs_f64() * 1000.0,
                info.name,
                info.backend,
                if pipeline_cache.is_some() {
                    "on"
                } else {
                    "unavailable"
                },
            );
        }
        Self {
            pipeline_cache,
            pipeline_cache_settled: false,
            clipmap_pipelines,
            surface,
            device,
            queue,
            adapter,
            config,
            env_present_mode,
            synced_vsync: None,
            last_acquire: None,
            acquire_interval: None,
            pipelines,
            terrain,
            water,
            skinned,
            crowd,
            sky,
            shadow,
            shadow_vp: [glam::Mat4::IDENTITY; 3],
            shadow_draw: [true; 3],
            frame_no: 0,
            moved_at: HashMap::new(),
            static_shadow_epoch: 0,
            clipmap: None,
            portal,
            depth_view,
            depth_texture,
            water_gpu,
            main_uniform_slot: 0,
            ssao,
            reflect,
            view_aspect: None,
            ao_active: false,
            gi,
            voxel,
            gpu_meshes: HashMap::new(),
            arenas: geometry::MeshArenas::default(),
            gpu_instance_batches: HashMap::new(),
            gpu_batch_sources: Vec::new(),
            gpu_batch_source_scratch: Vec::new(),
            gpu_skinned: HashMap::new(),
            gpu_skinned_meshes: HashMap::new(),
            skinned_maps: material::MapCache::default(),
            gpu_textures: HashMap::new(),
            gpu_mesh_albedo: HashMap::new(),
            audit_frame: 0,
            reupload_seen: (0, 0),
            gpu_check_seen: 0,
            rebuild_seen: (0, 0),
            gpu_materials: HashMap::new(),
            gpu_water_materials: HashMap::new(),
            synced_render_epoch: 0,
            synced_instance_epoch: 0,
            synced_resource_epoch: 0,
            terrain_origin: crate::space::RenderOrigin::default(),
            frustum: Frustum::default(),
            size,
            instance_scratch: Vec::new(),
            draw_source_epoch: 0,
            source_opaque: Vec::new(),
            source_terrain: Vec::new(),
            source_transparent: Vec::new(),
            source_water: Vec::new(),
            batch_source_opaque: Vec::new(),
            batch_source_terrain: Vec::new(),
            batch_source_transparent: Vec::new(),
            batch_source_water: Vec::new(),
            terrain_bundle_ids: Vec::new(),
            terrain_bundle: None,
            shadow_source_epoch: 0,
            shadow_mesh_sources: Vec::new(),
            shadow_batch_sources: Vec::new(),
            shadow_bundles: (0..STATIC_SHADOW_SLOT + 3)
                .map(|_| ShadowViewBundles::default())
                .collect(),
            opaque_prepass: OpaqueBundles::default(),
            opaque_world: OpaqueBundles::default(),
            draw_opaque: Vec::new(),
            draw_terrain: Vec::new(),
            draw_transparent: Vec::new(),
            draw_water: Vec::new(),
            batch_opaque: Vec::new(),
            batch_terrain: Vec::new(),
            batch_transparent: Vec::new(),
            batch_water: Vec::new(),
            gpu_stats: GpuFrameStats::default(),
            next_scene_uniform_slot: 0,
            gpu_profiler,
            instance_cull,
            occlusion,
            occlusion_auto: Default::default(),
            occlusion_active: false,
            occlusion_view_proj: glam::Mat4::IDENTITY,
            prepass_done: false,
            poses: Vec::new(),
            last_color: None,
            capture_next_frame: false,
            particles,
            cpu_spans: crate::cpu_trace::Spans::default(),
            ribbons,
            postprocess,
            taa,
            taa_frame: 0,
            taa_jitter: glam::Vec2::ZERO,
            render_scale: 1.0,
        }
    }

    /// Apply the world's presentation preference to the surface.
    ///
    /// `ENGINE_PRESENT_MODE` pins the mode for stress tests and overrides the
    /// preference. Without the pin, vsync requests FIFO and vsync off requires
    /// Immediate. A surface that cannot present immediately is a loud
    /// configuration error.
    fn apply_present_mode(&mut self, vsync: bool) {
        if self.synced_vsync == Some(vsync) {
            return;
        }
        if let Some(pinned) = self.env_present_mode {
            if self.config.present_mode != pinned {
                self.config.present_mode = pinned;
                self.surface.configure(&self.device, &self.config);
            }
            self.synced_vsync = Some(vsync);
            return;
        }
        let caps = self.surface.get_capabilities(&self.adapter);
        let requested = resolve_present_mode(vsync, &caps.present_modes);
        if requested != self.config.present_mode {
            self.config.present_mode = requested;
            self.surface.configure(&self.device, &self.config);
        }
        self.synced_vsync = Some(vsync);
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        audit_note(&format!(
            "resize to {}x{} (frame {})",
            new_size.width, new_size.height, self.audit_frame
        ));
        self.surface.configure(&self.device, &self.config);
        self.rebuild_screen_targets();
    }

    /// Debugging aid (see `World::request_renderer_rebuild`).
    pub(crate) fn rebuild_if_asked(&mut self, world: &World) {
        let asked = world.renderer_rebuild;
        if asked.0 != self.rebuild_seen.0 {
            audit_note(&format!(
                "surface reconfigured (frame {})",
                self.audit_frame
            ));
            self.surface.configure(&self.device, &self.config);
        }
        if asked.1 != self.rebuild_seen.1 {
            audit_note(&format!(
                "screen buffers rebuilt (frame {})",
                self.audit_frame
            ));
            self.rebuild_screen_targets();
        }
        self.rebuild_seen = asked;
    }

    /// The size the scene is drawn at: the window's, scaled by the render
    /// scale.
    fn scene_size(&self) -> (u32, u32) {
        taa::scaled_size(self.config.width, self.config.height, self.render_scale)
    }

    /// Follow the world's render scale; a new one rebuilds the scene targets.
    fn follow_render_scale(&mut self, world: &World) {
        let scale = render_scale_env()
            .unwrap_or(world.render_scale())
            .clamp(taa::MIN_RENDER_SCALE, 1.0);
        if (scale - self.render_scale).abs() > 1e-4 {
            self.render_scale = scale;
            self.rebuild_screen_targets();
        }
    }

    /// Everything sized to the window: depth, AO, reflections, post, TAA.
    /// The scene's targets are drawn at the render scale's share of it.
    fn rebuild_screen_targets(&mut self) {
        let (sw, sh) = self.scene_size();
        let (depth_texture, depth_view) = create_depth(&self.device, sw, sh);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.particles.set_depth(&self.device, &self.depth_texture);
        self.occlusion.resize(&self.device, &self.depth_texture);
        self.ssao.resize(&self.device, &self.depth_texture, sw, sh);
        self.shadow
            .set_ao_view(&self.device, self.ssao.view().clone());
        self.reflect.resize(&self.device, sw, sh);
        self.shadow.set_reflection_views(
            &self.device,
            self.reflect.ssr_view().clone(),
            self.reflect.probe_view().clone(),
        );
        self.pipelines
            .rebuild_scene_bind_groups(&self.device, &self.shadow);
        // The terrain bundle holds the old scene bind group.
        self.terrain_bundle_ids.clear();
        self.terrain_bundle = None;
        self.portal.resize();
        self.postprocess.resize(
            &self.device,
            (sw, sh),
            self.config.width,
            self.config.height,
        );
        self.taa.resize(
            &self.device,
            &self.depth_texture,
            self.config.width,
            self.config.height,
        );
        self.last_color = None;
        self.water_gpu.resize();
    }

    /// The adapter and driver this renderer runs on.
    pub fn gpu_info(&self) -> crate::gpu_info::GpuInfo {
        let i = self.adapter.get_info();
        crate::gpu_info::GpuInfo {
            name: i.name,
            vendor: i.vendor,
            driver: i.driver,
            driver_info: i.driver_info,
            backend: format!("{:?}", i.backend),
        }
    }

    pub fn take_gpu_stats(&mut self) -> GpuFrameStats {
        std::mem::take(&mut self.gpu_stats)
    }

    pub fn sync_world(&mut self, world: &mut World) {
        let mut t = std::time::Instant::now();
        self.reupload_if_asked(world);
        self.rebuild_if_asked(world);
        self.apply_present_mode(world.vsync());
        self.frame_no += 1;
        if self.frame_no % 512 == 0 {
            // Forget moves long settled (the entries only matter for a while).
            let frame = self.frame_no;
            self.moved_at
                .retain(|_, at| frame - *at <= STATIC_SHADOW_SETTLE);
        }
        if let Some((start, words)) = world.take_element_upload() {
            // Building elements fell: static shadows change.
            self.static_shadow_epoch += 1;
            let words = &words[..words.len().min(ELEMENT_STATE_WORDS.saturating_sub(start))];
            if !words.is_empty() {
                self.queue.write_buffer(
                    &self.shadow.element_states,
                    (start * 4) as u64,
                    bytemuck::cast_slice(words),
                );
            }
        }
        self.ribbons.sync(&self.queue, world.ribbon_frame());
        let (particle_commands, particle_dt, particle_time) = world.particles_frame();
        self.particles
            .sync(&self.queue, particle_commands, particle_dt, particle_time);
        self.cpu_spans.mark("sync.effects", &mut t);
        self.gpu_stats = GpuFrameStats::default();
        self.gpu_stats.entities =
            u32::try_from(world.entity_count()).expect("entity count exceeds u32");
        self.gpu_stats.animated = u32::try_from(world.animated_entities().count())
            .expect("animated entity count exceeds u32");
        self.gpu_stats.instance_submit = world.instance_submit();
        if self.synced_resource_epoch != world.resource_epoch()
            || self.terrain_origin != world.render_origin()
        {
            self.sync_textures_and_materials(world);
            self.synced_resource_epoch = world.resource_epoch();
        }

        if let Some(proc) = world.proc_terrain() {
            let format = HDR_FORMAT;
            match self.clipmap.as_mut() {
                Some(clip) => clip.ensure_config(&self.device, &self.queue, &proc.config),
                None => {
                    let pipes = match self.clipmap_pipelines.get() {
                        Some(pipes) => pipes.clone(),
                        None => std::sync::Arc::new(clipmap::ClipmapPipelines::new(
                            &self.device,
                            format,
                            &self.shadow.resource_layout,
                        )),
                    };
                    self.clipmap = Some(ClipmapRenderer::new(
                        &self.device,
                        &self.queue,
                        pipes,
                        proc.config.clone(),
                    ));
                }
            }
            if let Some(clip) = self.clipmap.as_mut() {
                clip.ensure_layers(&self.device, &self.queue, proc.layers);
                let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
                let vp = taa::jittered(world.camera().view_projection(aspect), self.taa_jitter);
                let light_dir = world.light().direction.normalize_or_zero();
                let torch_color = world.torch().map(|t| t.color.to_vec3()).unwrap_or_default();
                let torch_radius_m = world.torch().map(|t| t.radius_m.max(0.0)).unwrap_or(0.0);
                let torch_curve = world.torch().map(|t| t.curve.max(0.05)).unwrap_or(2.0);
                clip.prepare(
                    &self.queue,
                    vp,
                    light_dir,
                    world.light().ambient,
                    world.light().color,
                    world.camera().eye(),
                    torch_color,
                    torch_radius_m,
                    torch_curve,
                    SkyLight::from_world(world),
                    if world.ambient_occlusion().is_some() {
                        1.0
                    } else {
                        0.0
                    },
                    proc,
                    clipmap::GrassFrame {
                        scale: world.grass_scale(),
                        time: world.time(),
                        wind: world.wind(),
                    },
                );
            }
        } else {
            self.clipmap = None;
        }

        self.cpu_spans.mark("sync.terrain", &mut t);
        if self.synced_render_epoch != world.render_epoch()
            || self.synced_instance_epoch != world.instance_epoch()
        {
            self.gpu_meshes.retain(|id, _| world.contains_entity(*id));

            // Prototypes first so like-entities can share their GPU vertex buffers.
            for (id, entity) in world.entities() {
                if entity.instance_of.is_none() {
                    self.sync_entity_mesh(id, entity);
                }
            }
            match world.instance_submit() {
                InstanceSubmit::CpuIndexed => {
                    for (id, entity) in world.entities() {
                        if entity.instance_of.is_some() {
                            self.sync_entity_mesh(id, entity);
                        }
                    }
                }
                InstanceSubmit::GpuIndirect => {
                    self.cpu_spans.mark("sync.meshes", &mut t);
                    self.sync_gpu_instance_batches(world);
                    self.cpu_spans.mark("sync.batches", &mut t);
                }
            }
            self.synced_render_epoch = world.render_epoch();
            self.synced_instance_epoch = world.instance_epoch();
            self.cpu_spans.mark("sync.meshes", &mut t);
            // Meshes the world no longer needs in RAM go once they are here.
            let gpu = &self.gpu_meshes;
            world.release_uploaded(|id| gpu.contains_key(&id));
            self.cpu_spans.mark("sync.release", &mut t);
        } else {
            self.gpu_stats.instance_batches = u32::try_from(self.gpu_instance_batches.len())
                .expect("GPU instance batch count exceeds u32");
        }

        self.cpu_spans.mark("sync.meshes", &mut t);
        self.gpu_skinned
            .retain(|id, _| world.contains_animated(*id));
        let live_models: HashSet<usize> = world
            .animated_entities()
            .map(|(_, anim)| Arc::as_ptr(anim.animator.model()) as usize)
            .collect();
        self.gpu_skinned_meshes
            .retain(|k, _| live_models.contains(k));

        // Every pose first, on the frame's threads when there are several
        // (each reads only its own animator), then the uploads in order.
        let animated: Vec<(&EntityId, &crate::world::AnimatedEntity)> =
            world.animated_entities().collect();
        if self.poses.len() < animated.len() {
            self.poses.resize_with(animated.len(), Vec::new);
        }
        let poses = &mut self.poses[..animated.len()];
        let pose = |scratch: &mut crate::anim::PoseScratch,
                    (out, (_, anim)): (
            &mut Vec<glam::Mat4>,
            &(&EntityId, &crate::world::AnimatedEntity),
        )| {
            anim.animator.write_pose(scratch, out);
        };
        if animated.len() >= crate::world::ANIM_PARALLEL_POSES {
            use rayon::prelude::*;
            crate::parallel::frame_pool().install(|| {
                poses
                    .par_iter_mut()
                    .zip(animated.par_iter())
                    .for_each_init(crate::anim::PoseScratch::default, pose)
            });
        } else {
            let mut scratch = crate::anim::PoseScratch::default();
            for pair in poses.iter_mut().zip(animated.iter()) {
                pose(&mut scratch, pair);
            }
        }
        self.cpu_spans.mark("sync.poses", &mut t);
        for (i, (id, anim)) in animated.into_iter().enumerate() {
            let joints = std::mem::take(&mut self.poses[i]);
            match self.gpu_skinned.get_mut(id) {
                Some(gpu) => {
                    gpu.update(
                        &self.queue,
                        anim.transform,
                        &joints,
                        anim.animator.morph_weights(),
                    );
                    self.gpu_stats.skinned_pose_writes += 1;
                }
                None => {
                    let key = Arc::as_ptr(anim.animator.model()) as usize;
                    let (meshes, morphs) = if let Some(shared) = self.gpu_skinned_meshes.get(&key) {
                        shared.clone()
                    } else {
                        let model = anim.animator.model();
                        let (rows, table) = skinned::morph_table(model);
                        let uploaded: Vec<GpuSkinnedMesh> = model
                            .meshes
                            .iter()
                            .zip(&rows)
                            .map(|(m, rows)| {
                                GpuSkinnedMesh::upload(
                                    &self.device,
                                    &self.queue,
                                    &self.pipelines.albedo_layout,
                                    &self.pipelines.material_sampler,
                                    &self.pipelines.neutral_maps,
                                    &mut self.skinned_maps,
                                    m,
                                    m,
                                    Some(rows),
                                )
                            })
                            .collect();
                        let shared = (uploaded, skinned::upload_morph_table(&self.device, &table));
                        self.gpu_skinned_meshes.insert(key, shared.clone());
                        self.gpu_stats.skinned_model_uploads += 1;
                        shared
                    };
                    self.gpu_stats.skinned_new += 1;
                    self.gpu_skinned.insert(
                        *id,
                        GpuSkinnedEntity::from_shared_meshes(
                            &self.device,
                            &self.skinned.joint_bind_layout,
                            meshes,
                            &morphs,
                            anim.transform,
                            &joints,
                            anim.animator.morph_weights(),
                        ),
                    );
                }
            }
            self.poses[i] = joints;
        }
        self.cpu_spans.mark("sync.skinned", &mut t);
    }

    fn sync_entity_mesh(&mut self, id: EntityId, entity: &Entity) {
        if entity.instanced && entity.instances.is_empty() {
            if let Some(gpu) = self.gpu_meshes.get_mut(&id) {
                if gpu.xform_rev != entity.xform_rev {
                    gpu.clear_instances();
                    gpu.xform_rev = entity.xform_rev;
                }
            } else if entity.instance_of.is_none() {
                let mut gpu = GpuMesh::upload(
                    &self.device,
                    &self.queue,
                    &self.arenas,
                    entity.mesh(),
                    &[],
                    &self.instance_cull,
                );
                gpu.xform_rev = entity.xform_rev;
                self.gpu_meshes.insert(id, gpu);
            }
            return;
        }

        if let Some(gpu) = self.gpu_meshes.get(&id) {
            let same_mesh = entity.instance_of.is_some()
                || (gpu.vertex_count == entity.vertex_count()
                    && gpu.index_count == entity.index_count());
            if same_mesh && gpu.xform_rev == entity.xform_rev {
                return;
            }
        }

        self.instance_scratch.clear();
        if entity.instanced {
            self.instance_scratch
                .extend(entity.instances.iter().map(|inst| {
                    InstanceRaw::from_matrix_tint(entity.transform * inst.transform, inst.tint)
                }));
        } else {
            self.instance_scratch
                .push(InstanceRaw::from_matrix(entity.transform));
        }

        let shadow_buffers_changed = match self.gpu_meshes.get_mut(&id) {
            Some(gpu) => {
                let buffers_changed = if entity.instance_of.is_none()
                    && (gpu.vertex_count != entity.vertex_count()
                        || gpu.index_count != entity.index_count())
                {
                    *gpu = GpuMesh::upload(
                        &self.device,
                        &self.queue,
                        &self.arenas,
                        entity.mesh(),
                        &self.instance_scratch,
                        &self.instance_cull,
                    );
                    self.gpu_stats.mesh_rebuild += 1;
                    true
                } else {
                    let reallocated = gpu.update_instances(
                        &self.device,
                        &self.queue,
                        &self.arenas,
                        &self.instance_scratch,
                        &self.instance_cull,
                    );
                    self.gpu_stats.instance_rewrites += 1;
                    reallocated
                };
                gpu.xform_rev = entity.xform_rev;
                buffers_changed
            }
            None => {
                let mut gpu = if let Some(proto) = entity.instance_of {
                    let proto_gpu = self.gpu_meshes.get(&proto).unwrap_or_else(|| {
                        panic!("instanced-like entity {id} has no GPU mesh for prototype {proto}")
                    });
                    GpuMesh::share_vertices(
                        &self.device,
                        &self.queue,
                        &self.arenas,
                        proto_gpu,
                        &self.instance_scratch,
                        &self.instance_cull,
                    )
                } else {
                    GpuMesh::upload(
                        &self.device,
                        &self.queue,
                        &self.arenas,
                        entity.mesh(),
                        &self.instance_scratch,
                        &self.instance_cull,
                    )
                };
                gpu.xform_rev = entity.xform_rev;
                self.gpu_meshes.insert(id, gpu);
                self.gpu_stats.mesh_new += 1;
                false
            }
        };
        self.moved_at.insert(id, self.frame_no);
        if shadow_buffers_changed {
            self.invalidate_shadow_bundles();
        }
    }

    /// Mesh or instance buffers moved: record the bundles that bake them
    /// in anew (shadow views, and the terrain bundle, which is keyed by
    /// its entities only).
    fn invalidate_shadow_bundles(&mut self) {
        for view in &mut self.shadow_bundles {
            *view = ShadowViewBundles::default();
        }
        self.terrain_bundle_ids.clear();
        self.terrain_bundle = None;
        self.static_shadow_epoch += 1;
    }

    /// Bring a batch up to date without a new buffer: keep the slots that
    /// still fit, free the ones whose entity is gone, place new or grown
    /// entities in free space. False when it no longer fits (rebuild it).
    fn update_batch_in_place(
        &mut self,
        world: &World,
        batch: &mut GpuInstanceBatch,
        build: &BatchBuild,
    ) -> bool {
        let unchanged = batch.slots.len() == build.sources.len()
            && batch
                .slots
                .iter()
                .zip(&build.sources)
                .all(|(slot, source)| {
                    slot.id == source.id
                        && slot.xform_rev == source.xform_rev
                        && slot.active_count == source.instance_count
                });
        if unchanged {
            return true;
        }
        let wanted: HashMap<EntityId, &BatchBuildSource> =
            build.sources.iter().map(|s| (s.id, s)).collect();
        // Plan first, so a batch that cannot fit is left untouched.
        let mut free = batch.free.clone();
        let mut used_end = batch.used_end;
        let mut kept: Vec<BatchSlot> = Vec::with_capacity(build.sources.len());
        let mut freed: Vec<std::ops::Range<usize>> = Vec::new();
        for slot in batch.slots.drain(..) {
            match wanted.get(&slot.id) {
                Some(source)
                    if source.instance_count <= slot.capacity
                        && source.instance_reserve <= slot.capacity =>
                {
                    kept.push(slot)
                }
                _ => freed.push(slot.start..slot.start + slot.capacity),
            }
        }
        for r in &freed {
            give_free(&mut free, r.clone());
        }
        let kept_ids: HashSet<EntityId> = kept.iter().map(|s| s.id).collect();
        let mut placed: Vec<BatchSlot> = Vec::new();
        for source in &build.sources {
            if kept_ids.contains(&source.id) {
                continue;
            }
            let capacity = batch_slot_capacity(source.instance_count.max(source.instance_reserve));
            let Some(start) = take_free(&mut free, capacity) else {
                // Put the kept slots back so the caller can rebuild cleanly.
                batch.slots = kept;
                return false;
            };
            used_end = used_end.max(start + capacity);
            placed.push(BatchSlot {
                id: source.id,
                xform_rev: u64::MAX,
                start,
                capacity,
                active_count: 0,
                bounds: None,
            });
        }
        // Shrink the used end past trailing free space.
        if let Some(last) = free.last() {
            if last.end >= batch.gpu.instance_capacity && last.start < used_end {
                used_end = last.start;
            }
        }
        batch.free = free;
        batch.used_end = used_end;
        // Freed ranges go dead so the merged ranges never draw stale data.
        for r in freed {
            let dead = vec![InstanceRaw::from_matrix(DEAD_INSTANCE); r.end - r.start];
            batch.gpu.write_instances_at(&self.queue, r.start, &dead);
        }
        kept.extend(placed);
        // Keep the source order so the next frame's quick check matches.
        let order: HashMap<EntityId, usize> = build
            .sources
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();
        kept.sort_by_key(|slot| order[&slot.id]);
        for (slot, source) in kept.iter_mut().zip(&build.sources) {
            if slot.xform_rev == source.xform_rev && slot.active_count == source.instance_count {
                continue;
            }
            let entity = world
                .entity(source.id)
                .unwrap_or_else(|_| panic!("instance batch source {} disappeared", source.id));
            self.instance_scratch.clear();
            self.instance_scratch
                .extend(entity.instances.iter().map(|inst| {
                    InstanceRaw::from_matrix_tint(entity.transform * inst.transform, inst.tint)
                }));
            slot.bounds = batch.gpu.bounds_for_instances(&self.instance_scratch);
            // Pad to the old count with dead entries so a shrink leaves no
            // stale instances behind.
            let n = self.instance_scratch.len();
            if slot.active_count > n {
                self.instance_scratch
                    .resize(slot.active_count, InstanceRaw::from_matrix(DEAD_INSTANCE));
            }
            batch
                .gpu
                .write_instances_at(&self.queue, slot.start, &self.instance_scratch);
            slot.xform_rev = source.xform_rev;
            slot.active_count = source.instance_count;
            self.gpu_stats.batch_partial_writes += 1;
        }
        batch.slots = kept;
        batch.rebuild_active_ranges();
        batch.gpu.bounds = batch
            .slots
            .iter()
            .filter_map(|slot| slot.bounds)
            .reduce(|combined, bounds| combined.union(bounds));
        true
    }

    fn sync_gpu_instance_batches(&mut self, world: &World) {
        self.gpu_batch_source_scratch.clear();
        for (id, entity) in world.entities() {
            if !entity.instanced {
                continue;
            }
            let prototype = entity.instance_of.unwrap_or(id);
            let prototype_entity = world.entity(prototype).unwrap_or_else(|_| {
                panic!("instanced entity {id} references missing prototype {prototype}")
            });
            if entity.material != prototype_entity.material
                || entity.albedo != prototype_entity.albedo
                || entity.casts_shadow != prototype_entity.casts_shadow
            {
                panic!("instanced entity {id} is not render-compatible with prototype {prototype}");
            }
            self.gpu_batch_source_scratch.push(BatchSourceState {
                id,
                prototype,
                xform_rev: entity.xform_rev,
                instance_count: entity.instances.len(),
                instance_reserve: entity.instance_reserve,
                material: entity.material,
                albedo: entity.albedo,
                casts_shadow: entity.casts_shadow,
                prototype_material: prototype_entity.material,
                prototype_albedo: prototype_entity.albedo,
                prototype_casts_shadow: prototype_entity.casts_shadow,
            });
        }

        if self.gpu_batch_source_scratch == self.gpu_batch_sources {
            self.gpu_stats.instance_batches = u32::try_from(self.gpu_instance_batches.len())
                .expect("GPU instance batch count exceeds u32");
            return;
        }

        let mut builds: HashMap<EntityId, BatchBuild> = HashMap::new();
        for source in &self.gpu_batch_source_scratch {
            builds
                .entry(source.prototype)
                .or_default()
                .sources
                .push(BatchBuildSource {
                    id: source.id,
                    xform_rev: source.xform_rev,
                    instance_count: source.instance_count,
                    instance_reserve: source.instance_reserve,
                });
        }

        let batch_count_before = self.gpu_instance_batches.len();
        self.gpu_instance_batches
            .retain(|prototype, _| builds.contains_key(prototype));
        let mut shadow_buffers_changed = self.gpu_instance_batches.len() != batch_count_before;
        for (prototype, build) in builds {
            if let Some(mut batch) = self.gpu_instance_batches.remove(&prototype) {
                let (writes, slots) = (self.gpu_stats.batch_partial_writes, batch.slots.len());
                if self.update_batch_in_place(world, &mut batch, &build) {
                    if writes != self.gpu_stats.batch_partial_writes || slots != batch.slots.len() {
                        self.moved_at.insert(prototype, self.frame_no);
                    }
                    self.gpu_instance_batches.insert(prototype, batch);
                    continue;
                }
            }
            self.moved_at.insert(prototype, self.frame_no);

            let prototype_gpu = self.gpu_meshes.get(&prototype).unwrap_or_else(|| {
                panic!("instance batch has no GPU mesh for prototype {prototype}")
            });
            self.instance_scratch.clear();
            let mut slots = Vec::with_capacity(build.sources.len());
            for source in build.sources {
                let entity = world
                    .entity(source.id)
                    .unwrap_or_else(|_| panic!("instance batch source {} disappeared", source.id));
                let start = self.instance_scratch.len();
                self.instance_scratch
                    .extend(entity.instances.iter().map(|inst| {
                        InstanceRaw::from_matrix_tint(entity.transform * inst.transform, inst.tint)
                    }));
                let bounds = prototype_gpu.bounds_for_instances(&self.instance_scratch[start..]);
                let capacity =
                    batch_slot_capacity(source.instance_count.max(source.instance_reserve));
                self.instance_scratch
                    .resize(start + capacity, InstanceRaw::from_matrix(DEAD_INSTANCE));
                slots.push(BatchSlot {
                    id: source.id,
                    xform_rev: source.xform_rev,
                    start,
                    capacity,
                    active_count: source.instance_count,
                    bounds,
                });
            }
            // Room to grow, so streamed tiles coming and going reuse this
            // buffer instead of rebuilding it.
            let used = self.instance_scratch.len();
            let headroom = (used / 2).max(64);
            self.instance_scratch
                .resize(used + headroom, InstanceRaw::from_matrix(DEAD_INSTANCE));
            let mut gpu = GpuMesh::share_vertices(
                &self.device,
                &self.queue,
                &self.arenas,
                prototype_gpu,
                &self.instance_scratch,
                &self.instance_cull,
            );
            gpu.bounds = slots
                .iter()
                .filter_map(|slot| slot.bounds)
                .reduce(|combined, bounds| combined.union(bounds));
            self.gpu_instance_batches.insert(
                prototype,
                GpuInstanceBatch {
                    gpu,
                    slots,
                    free: vec![used..used + headroom],
                    used_end: used,
                    active_ranges: Vec::new(),
                },
            );
            self.gpu_instance_batches
                .get_mut(&prototype)
                .expect("instance batch was just inserted")
                .rebuild_active_ranges();
            self.gpu_stats.batch_rewrites += 1;
            shadow_buffers_changed = true;
        }
        if shadow_buffers_changed {
            self.invalidate_shadow_bundles();
        }
        std::mem::swap(
            &mut self.gpu_batch_sources,
            &mut self.gpu_batch_source_scratch,
        );
        self.gpu_stats.instance_batches = u32::try_from(self.gpu_instance_batches.len())
            .expect("GPU instance batch count exceeds u32");
    }

    /// 3D-only frame (no overlay pass).
    #[allow(dead_code)]
    pub fn render(&mut self, world: &World) -> Result<(), wgpu::SurfaceError> {
        self.render_with(world, |_, _, _, _| {})
    }

    /// Render the 3D world, then invoke `after` for overlay passes (egui).
    pub fn render_with(
        &mut self,
        world: &World,
        after: impl FnOnce(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Result<(), wgpu::SurfaceError> {
        self.follow_render_scale(world);
        self.occlusion_active = match world.occlusion_mode() {
            crate::world::OcclusionMode::Off => false,
            crate::world::OcclusionMode::On => true,
            crate::world::OcclusionMode::Auto => self
                .occlusion_auto
                .decide(self.draw_opaque.len(), self.occlusion.counts()),
        };
        let mut t = std::time::Instant::now();
        // The scene draws into Engine's own targets, and only the last
        // steps (post-processing, the UI) into the window's image. So the
        // scene is recorded and sent to the GPU first, and the window's
        // image is waited for afterwards: the GPU starts on the scene while
        // the display is still busy. The queue runs submissions in order,
        // so the post-processing always reads the finished scene.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        self.mark_gpu(&mut encoder, Mark::Start);
        if self.occlusion_active {
            self.occlusion.collect(&self.device);
        }
        let taa_on = world.taa() && taa_env_enabled();
        let (width, height) = (self.config.width.max(1), self.config.height.max(1));
        // Jitter moves by a fraction of a scene pixel (smaller than a window
        // pixel when the scene is drawn at a lower scale).
        let (scene_w, scene_h) = self.scene_size();
        self.taa_jitter = if taa_on {
            self.taa_frame = self.taa_frame.wrapping_add(1);
            let j = taa::jitter_pixels(self.taa_frame);
            glam::Vec2::new(j.x * 2.0 / scene_w as f32, j.y * 2.0 / scene_h as f32)
        } else {
            glam::Vec2::ZERO
        };
        let (scene_view, scene_resolve) = self.postprocess.scene_attachment();
        // Command buffers recorded on other threads, in submission order,
        // all before `encoder`'s own (submitted together, so every buffer
        // write of this frame lands before any of them runs).
        let mut recorded: Vec<wgpu::CommandBuffer> = Vec::new();
        self.encode_world(
            &mut encoder,
            &mut recorded,
            &scene_view,
            scene_resolve.as_ref(),
            world,
        );
        self.write_audit(world);
        self.gpu_check_if_asked(world);
        self.mark_gpu(&mut encoder, Mark::SceneDone);
        self.cpu_spans.mark("render.world", &mut t);
        if let Some(water) = world.water() {
            let camera = Self::main_scene_camera(world);
            let aspect = width as f32 / height as f32;
            let view_proj = taa::jittered(camera.view_projection(aspect), self.taa_jitter);
            let (scene_view, scene_texture) = self.postprocess.scene_target();
            self.water_gpu.encode(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.depth_texture,
                water,
                water_gpu::WaterFrame {
                    scene_view,
                    scene_texture,
                    scene_bind: self.pipelines.scene_bind_group(self.main_uniform_slot),
                    view_proj,
                    eye: camera.eye(),
                    time: world.time(),
                    wind: world.wind(),
                },
            );
        }
        self.mark_gpu(&mut encoder, Mark::WaterDone);
        self.cpu_spans.mark("render.water", &mut t);
        {
            let camera = Self::main_scene_camera(world);
            let aspect = width as f32 / height as f32;
            let view_proj = taa::jittered(camera.view_projection(aspect), self.taa_jitter);
            let (scene_view, _) = self.postprocess.scene_target();
            let scene_view = scene_view.clone();
            self.particles.encode_draw(
                &self.queue,
                &mut encoder,
                &scene_view,
                self.pipelines.scene_bind_group(self.main_uniform_slot),
                view_proj,
                camera.eye(),
            );
        }
        {
            let camera = Self::main_scene_camera(world);
            let aspect = width as f32 / height as f32;
            let projection = taa::jittered(camera.projection_matrix(aspect), self.taa_jitter);
            let view_proj = taa::jittered(camera.view_projection(aspect), self.taa_jitter);
            let (scene_view, _) = self.postprocess.scene_target();
            let scene_view = scene_view.clone();
            self.reflect.capture(
                &self.device,
                &self.queue,
                &mut encoder,
                &scene_view,
                &self.depth_texture,
                projection,
                view_proj,
                world.render_origin(),
            );
        }
        // TAA, and the upscale to the window's size when the scene is drawn
        // smaller (it runs with TAA off too, then without history).
        if taa_on || self.taa.upscaling() {
            if !taa_on {
                self.taa.reset();
            }
            let aspect = width as f32 / height as f32;
            let vp = Self::main_scene_camera(world).view_projection(aspect);
            let (view, _) = self.postprocess.scene_target();
            self.taa.encode(
                &self.device,
                &self.queue,
                &mut encoder,
                view,
                self.postprocess.resolve_target(),
                vp,
                self.taa_jitter,
                taa_on,
            );
        } else {
            self.taa.reset();
        }
        self.mark_gpu(&mut encoder, Mark::WorldDone);
        self.cpu_spans.mark("render.particles_fx", &mut t);
        self.queue.submit(
            recorded
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        self.cpu_spans.mark("render.submit_scene", &mut t);

        let frame = self.surface.get_current_texture()?;
        self.cpu_spans.mark("render.acquire", &mut t);
        // Acquiring waits for the display, so the time between acquires is
        // the even rhythm frames reach the screen at.
        let acquired = std::time::Instant::now();
        if let Some(last) = self.last_acquire.replace(acquired) {
            self.acquire_interval = Some((acquired - last).as_secs_f32());
        }
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present-encoder"),
            });
        self.postprocess.encode(
            &self.queue,
            &mut encoder,
            &view,
            world.bloom(),
            world.screen_grade(),
            tone_map_env().unwrap_or(world.tone_map()),
        );
        self.mark_gpu(&mut encoder, Mark::PostDone);
        after(&self.device, &self.queue, &mut encoder, &view);
        if std::mem::take(&mut self.capture_next_frame) {
            self.store_composited(&mut encoder, &frame.texture);
        }
        self.mark_gpu(&mut encoder, Mark::UiDone);
        self.cpu_spans.mark("render.post_ui", &mut t);
        if let Some(profiler) = self.gpu_profiler.as_ref() {
            profiler.resolve(&mut encoder);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.occlusion.after_submit(&self.device);
        self.cpu_spans.mark("render.submit", &mut t);
        if let Some(profiler) = self.gpu_profiler.as_mut() {
            profiler.read_and_report(&self.device);
        }
        frame.present();
        self.cpu_spans.mark("render.present", &mut t);
        self.settle_pipeline_cache();
        Ok(())
    }

    /// Once every startup worker has finished, save the pipeline cache so
    /// the next launch starts with all pipelines already compiled.
    fn settle_pipeline_cache(&mut self) {
        if self.pipeline_cache_settled
            || !self.clipmap_pipelines.is_ready()
            || !self.water_gpu.prewarm_ready()
        {
            return;
        }
        self.pipeline_cache_settled = true;
        if let Some(cache) = self.pipeline_cache.as_mut() {
            cache.save();
        }
    }

    /// Stamp a part of the main scene pass (only the top level, not portals).
    fn mark_scene(&self, pass: &mut wgpu::RenderPass<'_>, on: bool, mark: SceneMark) {
        if let (true, Some(profiler)) = (on, self.gpu_profiler.as_ref()) {
            profiler.scene_mark(pass, mark);
        }
    }

    fn mark_gpu(&self, encoder: &mut wgpu::CommandEncoder, mark: Mark) {
        if let Some(profiler) = self.gpu_profiler.as_ref() {
            profiler.timestamp(encoder, mark);
        }
    }

    /// Rough video memory held by meshes and textures (bytes): (mesh
    /// buffers, textures, mesh count, texture count). For diagnostics.
    pub fn mesh_and_texture_bytes(&self) -> (u64, u64, usize, usize) {
        // Vertices, indices and instances: the arenas' pages.
        let mesh = |g: &gpu_mesh::GpuMesh| g.compact_buf.size();
        let meshes: u64 = self.arenas.reserved_bytes()
            + self.gpu_meshes.values().map(mesh).sum::<u64>()
            + self
                .gpu_instance_batches
                .values()
                .map(|b| mesh(&b.gpu))
                .sum::<u64>();
        let textures: u64 = self
            .gpu_textures
            .values()
            .map(|t| t.width as u64 * t.height as u64 * 4)
            .sum();
        (
            meshes,
            textures,
            self.gpu_meshes.len() + self.gpu_instance_batches.len(),
            self.gpu_textures.len() + self.gpu_mesh_albedo.len(),
        )
    }

    /// Rough video memory held by characters (bytes): (vertex and index
    /// buffers of animated models and crowds, maps, map count). Maps are
    /// shared by content, so a garment worn by many counts once.
    pub fn character_bytes(&self) -> (u64, u64, usize) {
        let buffers: u64 = self
            .gpu_skinned_meshes
            .values()
            .flat_map(|(m, _)| m)
            .map(GpuSkinnedMesh::buffer_bytes)
            .sum::<u64>()
            + self.crowd.buffer_bytes();
        let (n, maps) = self.skinned_maps.usage();
        (buffers, maps, n)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Render the current world to a PNG (for demo QA / automation).
    /// Ask the next `render_with` to keep the finished frame for `capture_png`.
    pub fn request_capture(&mut self) {
        self.capture_next_frame = true;
    }

    pub fn capture_png(
        &mut self,
        _world: &World,
        path: impl AsRef<std::path::Path>,
    ) -> crate::EngineResult<()> {
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.config.format;

        let bpp = 4u32;
        let unpadded = width * bpp;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer_size = (padded * height) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture-buf"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture-encoder"),
            });
        let src = self.last_color.as_ref().expect(
            "capture_png needs request_capture before render_with so the HUD is in the shot",
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = output.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // Receiver loss is only possible if capture teardown abandons this operation.
            if tx.send(result).is_err() {
                return;
            }
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .expect("device poll failed");
        rx.recv().expect("map channel closed").expect("map failed");

        let data = slice.get_mapped_range();
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        let is_bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        for y in 0..height as usize {
            let src = &data[y * padded as usize..y * padded as usize + unpadded as usize];
            let dst = &mut rgba[y * unpadded as usize..(y + 1) * unpadded as usize];
            if is_bgra {
                for (i, px) in src.chunks_exact(4).enumerate() {
                    dst[i * 4] = px[2];
                    dst[i * 4 + 1] = px[1];
                    dst[i * 4 + 2] = px[0];
                    dst[i * 4 + 3] = px[3];
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        drop(data);
        output.unmap();

        image::save_buffer(path.as_ref(), &rgba, width, height, image::ColorType::Rgba8).map_err(
            |source| crate::EngineError::ScreenshotSave {
                path: path.as_ref().to_path_buf(),
                source,
            },
        )
    }

    fn store_composited(&mut self, encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture) {
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.config.format;
        let stale = self.last_color.as_ref().is_none_or(|t| {
            let s = t.size();
            s.width != width || s.height != height || t.format() != format
        });
        if stale {
            self.last_color = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("last-color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
        }
        let dst = self.last_color.as_ref().expect("last-color");
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn write_scene_uniforms(
        &mut self,
        world: &World,
        camera: &Camera,
        clip_plane: Option<(glam::Vec3, glam::Vec3)>,
        target: SceneUniformTarget,
    ) -> usize {
        let ao_strength = match target {
            SceneUniformTarget::Main if self.ao_active => 1.0,
            _ => 0.0,
        };
        let gi_strength = match target {
            SceneUniformTarget::Main | SceneUniformTarget::Probe
                if world.global_illumination().is_some() =>
            {
                1.0
            }
            _ => 0.0,
        };
        let slot = self.next_scene_uniform_slot;
        self.next_scene_uniform_slot = self
            .next_scene_uniform_slot
            .checked_add(1)
            .expect("scene uniform slot counter overflow");
        let aspect = self
            .view_aspect
            .unwrap_or(self.config.width as f32 / self.config.height.max(1) as f32);
        let vp = match clip_plane {
            Some((point, normal)) => oblique_view_projection(camera, aspect, point, normal),
            None => camera.view_projection(aspect),
        };
        let vp = taa::jittered(vp, self.taa_jitter);
        self.frustum = Frustum::from_view_projection(vp);
        if target == SceneUniformTarget::Main {
            self.occlusion_view_proj = vp;
        }
        let light_dir = world.light().direction.normalize_or_zero();
        let eye = camera.eye();
        let haze = world.screen_haze();
        let torch = world.torch();
        let forward = (camera.target() - camera.eye()).normalize_or_zero();
        let right = forward.cross(camera.up()).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let sky_light = SkyLight::from_world(world);
        let uniforms = Uniforms {
            view_proj: vp.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z],
            ambient: world.light().ambient,
            light_color: world.light().color.into(),
            wind: world.wind(),
            eye: [eye.x, eye.y, eye.z],
            time: world.time(),
            haze_color: haze
                .map(|h| h.color.to_vec3().into())
                .unwrap_or([1.0, 1.0, 1.0]),
            haze_density: haze.map(|h| h.density()).unwrap_or(0.0),
            haze_height_m: haze.map(|h| h.height_m.max(1.0)).unwrap_or(1.0),
            haze_base_y: haze.map(|h| h.base_y).unwrap_or(0.0),
            torch_radius_m: torch.map(|t| t.radius_m.max(0.0)).unwrap_or(0.0),
            torch_curve: torch.map(|t| t.curve.max(0.05)).unwrap_or(2.0),
            torch_color: torch
                .map(|t| {
                    let c = t.color.to_vec3();
                    [c.x, c.y, c.z, world.debug_shading as f32]
                })
                .unwrap_or([0.0, 0.0, 0.0, world.debug_shading as f32]),
            camera_right: right.into(),
            ao_strength,
            camera_up: up.into(),
            gi_strength,
            sky_up: sky_light.up,
            env_strength: 1.0,
            sky_horizon: sky_light.horizon,
            dither_frame: (self.taa_frame % 64) as f32,
            sky_down: sky_light.down,
            season: world.season(),
            glow: world.glow_table(),
        };
        // Destination and live views share one encoder. Queue writes land
        // before that encoder runs, so they cannot share a uniform buffer.
        let (uniform_buf, _) = self.pipelines.scene_uniforms.get(slot);
        self.queue
            .write_buffer(uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        if let Some(sky) = world.sky() {
            let mut sky_u =
                SkyUniforms::from_scene(&sky, camera, world.light(), aspect, world.time());
            if let Some(physical) = world.physical_sky() {
                sky_u = sky_u.with_physical(&physical, world.moons(), camera, world.time());
            }
            self.queue
                .write_buffer(&self.sky.uniform_bufs[slot], 0, bytemuck::bytes_of(&sky_u));
        }
        slot
    }

    fn write_shadows(&mut self, world: &World) {
        let eye = Self::main_scene_camera(world).eye();
        (self.shadow_vp, self.shadow_draw) =
            self.shadow.prepare(&self.queue, world, eye, &self.frustum);
        self.gpu_stats.shadow_atlas = self.shadow.atlas_wrote();
    }

    /// Seconds between the last two swapchain acquires: how often frames
    /// actually reach the display (None before two frames).
    pub fn acquire_interval(&self) -> Option<f32> {
        self.acquire_interval
    }

    fn main_scene_camera(world: &World) -> Camera {
        let camera = world.camera();
        let look = camera.target() - camera.eye();
        if look.length_squared() <= 0.0 {
            return camera.clone();
        }
        let live = world.living_in();
        let portals = world
            .visible_portals(camera.eye(), look, live)
            .expect("portal visibility invariants must hold while rendering");
        let Some(visible) = portals.first() else {
            return camera.clone();
        };
        let plane = world
            .portal_plane(visible.src)
            .expect("visible portal source must retain its validated plane");
        portal_render_camera(camera, plane)
    }

    fn encode_world(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        recorded: &mut Vec<wgpu::CommandBuffer>,
        color: &wgpu::TextureView,
        resolve: Option<&wgpu::TextureView>,
        world: &World,
    ) {
        let mut t = std::time::Instant::now();
        self.next_scene_uniform_slot = 0;
        let scene_camera = Self::main_scene_camera(world);
        let ao = world.ambient_occlusion();
        self.ao_active = ao.is_some();
        let main_uniform_slot =
            self.write_scene_uniforms(world, &scene_camera, None, SceneUniformTarget::Main);
        self.main_uniform_slot = main_uniform_slot;
        {
            let (device, queue, p) = (&self.device, &self.queue, &self.pipelines);
            let maps = &mut self.skinned_maps;
            self.crowd.prepare(
                device,
                queue,
                world,
                &self.frustum,
                scene_camera.eye(),
                |m, material| {
                    GpuSkinnedMesh::upload(
                        device,
                        queue,
                        &p.albedo_layout,
                        &p.material_sampler,
                        &p.neutral_maps,
                        maps,
                        m,
                        material,
                        None,
                    )
                },
            );
        }
        let gi_placement = world
            .global_illumination()
            .map(|gi| gi::GiPlacement::around(scene_camera.eye(), gi.probe_spacing_m()));
        self.shadow.set_gi(gi_placement);
        self.voxel.update(
            &self.device,
            &self.queue,
            encoder,
            world.voxel_terrain(),
            scene_camera.eye(),
        );
        self.mark_gpu(encoder, Mark::VoxelDone);
        self.cpu_spans.mark("world.crowd_voxel", &mut t);
        self.write_shadows(world);
        self.begin_instance_cull_frame(world);
        if self.shadow.encode_horizon(encoder) {
            self.gpu_stats.terrain_horizon_rebuilds += 1;
        }
        self.cpu_spans.mark("world.shadow_setup", &mut t);
        self.encode_shadow_pass(encoder, recorded, world);
        self.mark_gpu(encoder, Mark::ShadowDone);
        self.cpu_spans.mark("world.shadows", &mut t);
        if let Some(placement) = gi_placement {
            self.encode_gi(encoder, world, placement, &scene_camera);
        }
        self.mark_gpu(encoder, Mark::GiDone);
        self.cpu_spans.mark("world.gi", &mut t);
        self.capture_probes(encoder, world);
        let probes: Vec<_> = world
            .reflection_probe_slots()
            .map(|(i, p, _)| (i, *p))
            .collect();
        let reflect_u = self.reflect.uniforms(
            world.screen_space_reflections(),
            world.render_origin(),
            &probes,
        );
        self.queue.write_buffer(
            &self.shadow.reflect.uniform,
            0,
            bytemuck::bytes_of(&reflect_u),
        );
        self.particles
            .encode_compute(encoder, self.pipelines.scene_bind_group(main_uniform_slot));

        self.cpu_spans.mark("world.probes_fx", &mut t);
        let live = world.living_in();
        self.collect_draws(world, live, true);
        self.cpu_spans.mark("world.collect_draws", &mut t);
        self.prepare_terrain_bundle(world);
        let submit = world.instance_submit();
        let mut main_cull = if submit == InstanceSubmit::GpuIndirect {
            let frustum = self.frustum;
            let selection = self.select_main_culls(&frustum);
            self.dispatch_instance_cull(encoder, &frustum, &selection.ids, CullView::Main, None);
            selection
        } else {
            MainCullSelection::none()
        };
        let occlusion_ids = if self.occlusion_active {
            let ids = self.occlusion_candidates(world, live);
            main_cull.occlusion = Some(self.begin_occlusion(&ids));
            ids
        } else {
            Vec::new()
        };

        self.cpu_spans.mark("world.cull", &mut t);
        let depth = self.depth_view.clone();
        let occluding = main_cull.occlusion.is_some();
        // A full prepass (the world pass then only tests depth) for AO; for
        // occlusion alone just the meshes it needs, which is far cheaper
        // than drawing voxel ground and props twice.
        self.prepass_done = ao.is_some();
        let phase = match (self.prepass_done, occluding) {
            (true, true) => Some(PrepassPhase::Seen),
            (true, false) => Some(PrepassPhase::All),
            (false, true) => Some(PrepassPhase::SeenOccluders),
            (false, false) => None,
        };
        if let Some(phase) = phase {
            if phase != PrepassPhase::SeenOccluders {
                self.prepare_opaque_bundles(
                    OpaqueTarget::Prepass(phase),
                    world,
                    live,
                    &main_cull,
                    main_uniform_slot,
                );
                self.cpu_spans.mark("world.record_opaque", &mut t);
            }
            self.encode_depth_prepass(encoder, world, live, &main_cull, main_uniform_slot, phase);
        }
        if occluding {
            self.occlusion.encode(
                &self.device,
                &self.queue,
                encoder,
                self.occlusion_view_proj,
                &occlusion_ids,
            );
            self.encode_depth_prepass(
                encoder,
                world,
                live,
                &main_cull,
                main_uniform_slot,
                PrepassPhase::Unseen,
            );
            if submit == InstanceSubmit::GpuIndirect {
                // Instanced props: compact again, now also dropping the
                // instances hidden behind the pyramid. The prepass above
                // drew them all; the world pass draws only these.
                let mut seen = HashSet::new();
                let batches: Vec<EntityId> = self
                    .batch_opaque
                    .iter()
                    .chain(&self.batch_terrain)
                    .chain(&self.batch_transparent)
                    .copied()
                    .filter(|id| seen.insert(*id))
                    .collect();
                let (test, pyramid) = self.occlusion.test_resources();
                let hiz = self.instance_cull.hiz_bind(&self.device, test, pyramid);
                let frustum = self.frustum;
                self.dispatch_instance_cull(
                    encoder,
                    &frustum,
                    &batches,
                    CullView::Main,
                    Some(&hiz),
                );
                main_cull.compacted.extend(batches);
            }
        }
        if let Some(ao) = ao {
            let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
            self.ssao.prepare(
                &self.queue,
                ssao::AoFrame {
                    projection: scene_camera.projection_matrix(aspect),
                    fov_y_radians: scene_camera.fov_y_degrees().to_radians(),
                    radius_m: ao.radius_m(),
                    intensity: ao.intensity(),
                },
            );
            self.ssao.encode(encoder);
        }
        self.mark_gpu(encoder, Mark::AoDone);
        self.cpu_spans.mark("world.prepass_ao", &mut t);
        self.prepare_opaque_bundles(
            OpaqueTarget::World,
            world,
            live,
            &main_cull,
            main_uniform_slot,
        );
        self.cpu_spans.mark("world.record_opaque", &mut t);
        // With a prepass the world pass keeps its depth: hidden fragments
        // then fail the depth test before any shading runs.
        let own_depth = world.debug_shading & 65536 != 0;
        let (depth_load, stencil_load) = if (self.prepass_done || occluding) && !own_depth {
            (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
        } else {
            (wgpu::LoadOp::Clear(DEPTH_CLEAR), wgpu::LoadOp::Clear(0))
        };

        let clear = world.clear_color().to_vec3();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                resolve_target: resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear.x as f64,
                        g: clear.y as f64,
                        b: clear.z as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: stencil::has_stencil().then_some(wgpu::Operations {
                    load: stencil_load,
                    store: wgpu::StoreOp::Store,
                }),
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        self.render_portal_level(
            &mut pass,
            world,
            &scene_camera,
            PortalLevel {
                space: live,
                depth: 0,
                view: PortalLevelView {
                    clip_plane: None,
                    draw_environment: world.space_draws_environment(live),
                    use_cull: true,
                },
            },
            &main_cull,
            Some(main_uniform_slot),
        );
        drop(pass);
        self.opaque_prepass.frame = None;
        self.opaque_world.frame = None;
        self.cpu_spans.mark("world.main_pass", &mut t);
    }

    /// Draw this frame's reflection probe faces (see `reflect.rs`).
    fn capture_probes(&mut self, encoder: &mut wgpu::CommandEncoder, world: &World) {
        let probes: Vec<_> = world
            .reflection_probe_slots()
            .map(|(i, p, v)| (i, *p, v))
            .collect();
        let jobs = self.reflect.plan(&probes);
        if jobs.is_empty() {
            return;
        }
        let saved_frustum = self.frustum;
        let saved_jitter = std::mem::replace(&mut self.taa_jitter, glam::Vec2::ZERO);
        self.view_aspect = Some(1.0);
        // A probe sees every room, not just the ones the camera does.
        world.set_room_bypass(true);
        let live = world.living_in();
        let no_cull = MainCullSelection::none();
        let clear = world.clear_color().to_vec3();
        let far = world.camera().far().max(100.0);
        let (color, resolve, depth) = {
            let (c, r, d) = self.reflect.face_targets();
            (c.clone(), r.cloned(), d.clone())
        };
        for job in jobs {
            let (forward, up) = reflect::face_view(job.face);
            let camera = Camera::from_parts(job.center, job.center + forward, up, 90.0, 0.05, far);
            let slot = self.write_scene_uniforms(world, &camera, None, SceneUniformTarget::Probe);
            self.collect_draws(world, live, true);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("probe-face"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color,
                        resolve_target: resolve.as_ref(),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: clear.x as f64,
                                g: clear.y as f64,
                                b: clear.z as f64,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: stencil::has_stencil().then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Discard,
                        }),
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                if world.sky().is_some() {
                    pass.set_pipeline(&self.sky.pipeline);
                    pass.set_bind_group(0, self.sky.bind_group(slot), &[]);
                    pass.draw(0..3, 0..1);
                }
                self.draw_space_scene(
                    &mut pass,
                    world,
                    live,
                    &no_cull,
                    PortalLevelView {
                        clip_plane: None,
                        draw_environment: false,
                        use_cull: false,
                    },
                    ScenePass {
                        portal_mask: false,
                        uniform_slot: slot,
                    },
                );
            }
            self.reflect.store_face(&self.device, encoder, job);
        }
        world.set_room_bypass(false);
        self.view_aspect = None;
        self.taa_jitter = saved_jitter;
        self.frustum = saved_frustum;
    }

    fn render_portal_level<'a>(
        &mut self,
        pass: &mut wgpu::RenderPass<'a>,
        world: &World,
        camera: &Camera,
        level: PortalLevel,
        main_cull: &MainCullSelection,
        level_slot: Option<usize>,
    ) {
        let PortalLevel {
            space,
            depth,
            view:
                PortalLevelView {
                    clip_plane,
                    draw_environment,
                    use_cull,
                },
        } = level;
        let level = depth;
        let look = camera.target() - camera.eye();
        if look.length_squared() <= 0.0 {
            return;
        }
        let portals = world
            .visible_portals(camera.eye(), look, space)
            .expect("portal visibility invariants must hold while rendering");
        // The live view reuses the main uniform slot, which the depth prepass
        // and the terrain bundle were drawn with: their depths must match.
        let uniform_slot = match level_slot {
            Some(slot) => slot,
            None => {
                let scene_camera = if let Some(visible) = portals.first() {
                    let plane = world
                        .portal_plane(visible.src)
                        .expect("visible portal source must retain its validated plane");
                    portal_render_camera(camera, plane)
                } else {
                    camera.clone()
                };
                self.write_scene_uniforms(
                    world,
                    &scene_camera,
                    clip_plane,
                    SceneUniformTarget::Portal,
                )
            }
        };
        pass.set_stencil_reference(u32::from(level));

        self.draw_space_scene(
            pass,
            world,
            space,
            main_cull,
            PortalLevelView {
                clip_plane,
                draw_environment,
                use_cull,
            },
            ScenePass {
                portal_mask: level > 0,
                uniform_slot,
            },
        );

        if level >= world.portal_recursion() {
            return;
        }

        if portals.is_empty() {
            return;
        }

        for visible in portals {
            if !self.portal_in_frustum(world, visible.src, camera.eye()) {
                continue;
            }
            let src_plane = world
                .portal_plane(visible.src)
                .expect("visible portal has a plane");
            let render_cam = portal_render_camera(camera, src_plane);
            let stencil_uniform_slot = self.write_scene_uniforms(
                world,
                &render_cam,
                clip_plane,
                SceneUniformTarget::Portal,
            );
            self.draw_portal_stencil_incr(pass, world, visible.src, level, stencil_uniform_slot);
            pass.set_pipeline(&self.portal.depth_clear);
            pass.set_stencil_reference(u32::from(level + 1));
            pass.draw(0..3, 0..1);

            let virt = portal_view_camera(&render_cam, &visible, src_plane);
            let dest_clip = portal_destination_clip(&render_cam, &visible, src_plane);
            let dest_env = world.space_draws_environment(visible.dest_space);
            // Cull destination draws with the virtual camera's normal frustum.
            // Oblique clipping is applied when the pass actually draws; using it
            // here over-tightens culling and drops the whole yard at doorways.
            self.collect_draws(world, visible.dest_space, false);

            self.render_portal_level(
                pass,
                world,
                &virt,
                PortalLevel {
                    space: visible.dest_space,
                    depth: level + 1,
                    view: PortalLevelView {
                        clip_plane: dest_clip,
                        draw_environment: dest_env,
                        use_cull: false,
                    },
                },
                main_cull,
                None,
            );
            let restore_uniform_slot =
                self.write_scene_uniforms(world, camera, clip_plane, SceneUniformTarget::Portal);
            self.collect_draws(world, space, true);
            self.draw_portal_stencil_decr(
                pass,
                world,
                visible.src,
                level + 1,
                restore_uniform_slot,
            );
        }
    }

    fn portal_in_frustum(&self, world: &World, id: EntityId, eye: glam::Vec3) -> bool {
        let plane = world.portal_plane(id).expect("portal surface has a plane");
        if plane.signed_distance(eye) < crate::portal::PORTAL_CLOSE_VIEW_DIST {
            return true;
        }
        let radius =
            (plane.half_width * plane.half_width + plane.half_height * plane.half_height).sqrt();
        self.frustum.intersects(Bounds {
            centre: plane.center,
            radius,
        })
    }

    fn draw_portal_stencil_incr<'a>(
        &mut self,
        pass: &mut wgpu::RenderPass<'a>,
        _world: &World,
        id: EntityId,
        level: u8,
        uniform_slot: usize,
    ) {
        pass.set_pipeline(&self.portal.stencil_incr);
        pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
        pass.set_stencil_reference(u32::from(level));
        self.draw_portal_mesh(pass, id);
    }

    fn draw_portal_stencil_decr<'a>(
        &mut self,
        pass: &mut wgpu::RenderPass<'a>,
        _world: &World,
        id: EntityId,
        level: u8,
        uniform_slot: usize,
    ) {
        pass.set_pipeline(&self.portal.stencil_decr);
        pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
        pass.set_stencil_reference(u32::from(level));
        self.draw_portal_mesh(pass, id);
    }

    fn draw_portal_mesh(&mut self, pass: &mut wgpu::RenderPass<'_>, id: EntityId) {
        let gpu = self.gpu_meshes.get(&id).expect("portal mesh is synced");
        submit_mesh_draw(
            pass,
            gpu,
            MeshSubmit::Direct,
            0..gpu.index_count as u32,
            OPAQUE_INDIRECT_OFFSET,
            &mut self.gpu_stats.indirect_draws,
        );
    }

    /// Rebuild the GI voxels when needed and update the light probes.
    fn encode_gi(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        placement: gi::GiPlacement,
        camera: &Camera,
    ) {
        let light_color: [f32; 3] = world.light().color.into();
        let frame = gi::GiFrame {
            placement,
            eye: camera.eye(),
            light_dir: world.light().direction,
            light_color: glam::Vec3::from(light_color),
            ambient: world.light().ambient,
            sky: SkyLight::from_world(world),
        };
        let epoch = world
            .render_epoch()
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.voxel.epoch());
        if self.gi.prepare(&self.queue, &frame, epoch) {
            self.gi.encode_fill(encoder, &self.shadow.resource_bind);
            self.voxelize_scene(encoder, world, placement);
        }
        self.gi.encode_update(encoder, &self.shadow.resource_bind);
    }

    /// Rasterise every opaque mesh inside the GI box into its voxels, once
    /// along each axis.
    fn voxelize_scene(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        placement: gi::GiPlacement,
    ) {
        if self.draw_source_epoch != world.render_epoch() {
            self.rebuild_draw_sources(world);
            self.draw_source_epoch = world.render_epoch();
        }
        let space = world.living_in();
        let half = placement.dims.as_vec3() * placement.spacing * 0.5;
        let box_centre = placement.origin + half;
        let near_box = |bounds: Option<Bounds>| {
            bounds.is_none_or(|b| {
                let d = (b.centre - box_centre).abs() - half;
                d.max(glam::Vec3::ZERO).length() <= b.radius
            })
        };
        let meshes: Vec<EntityId> = self
            .source_opaque
            .iter()
            .chain(&self.source_terrain)
            .copied()
            .filter(|&id| {
                entity_in_space(world, id, space)
                    && self.gpu_meshes.get(&id).is_some_and(|g| near_box(g.bounds))
            })
            .collect();
        let batches: Vec<EntityId> = self
            .batch_source_opaque
            .iter()
            .chain(&self.batch_source_terrain)
            .copied()
            .filter(|&id| {
                entity_in_space(world, id, space)
                    && self
                        .gpu_instance_batches
                        .get(&id)
                        .is_some_and(|b| near_box(b.gpu.bounds))
            })
            .collect();
        let voxel_draws = if space == SpaceId::DEFAULT {
            self.voxel.visible(|b| near_box(Some(b)))
        } else {
            Vec::new()
        };
        let mut draws = 0;
        for axis in 0..3 {
            let mut pass = self.gi.begin_voxelize(encoder, axis);
            if !voxel_draws.is_empty() {
                self.gi.use_terrain_voxelizer(&mut pass);
                self.voxel.draw(&mut pass, &voxel_draws);
                self.gi.use_mesh_voxelizer(&mut pass);
            }
            for id in &meshes {
                let gpu = &self.gpu_meshes[id];
                submit_mesh_draw(
                    &mut pass,
                    gpu,
                    MeshSubmit::Direct,
                    0..gpu.opaque_index_count as u32,
                    OPAQUE_INDIRECT_OFFSET,
                    &mut draws,
                );
            }
            for id in &batches {
                let batch = &self.gpu_instance_batches[id];
                submit_batch_draw(
                    &mut pass,
                    batch,
                    None,
                    0..batch.gpu.opaque_index_count as u32,
                    OPAQUE_INDIRECT_OFFSET,
                    &mut draws,
                );
            }
        }
    }

    /// Get the top-level view's opaque mesh draws for one pass ready as
    /// render bundles, in chunks. The draws are exactly the ones that pass
    /// would make itself (same list, order, culling and occlusion choices).
    /// Chunks whose draws are unchanged are kept; the others are recorded on
    /// the frame's threads, reading only renderer state that nothing changes
    /// meanwhile. The chunks stay in draw order and are executed in that
    /// order, so the result matches drawing one by one. With too few draws
    /// for bundles to pay off there are none, and the pass draws directly.
    fn prepare_opaque_bundles(
        &mut self,
        target: OpaqueTarget,
        world: &World,
        space: SpaceId,
        main_cull: &MainCullSelection,
        uniform_slot: usize,
    ) {
        let albedo_of = |entity: &Entity| {
            entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo)
        };
        let mut draws: Vec<OpaqueDraw<'_>> = Vec::new();
        for &id in &self.draw_opaque {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let occlusion = main_cull.occlusion_slot(id);
            let submit = match target {
                // Candidates not seen lately wait for the occlusion test.
                OpaqueTarget::Prepass(PrepassPhase::Seen)
                    if occlusion.is_some()
                        && !main_cull
                            .occlusion
                            .as_ref()
                            .is_some_and(|f| f.first.contains(&id)) =>
                {
                    continue;
                }
                OpaqueTarget::World if occlusion.is_some() => {
                    let (args, slot) = occlusion.expect("checked");
                    OpaqueSubmit::Occluded(args, slot)
                }
                _ => OpaqueSubmit::Mesh(mesh_submit(&main_cull.compacted, id)),
            };
            let entity = world.entity(id).expect("draw list is live");
            draws.push(OpaqueDraw {
                albedo: albedo_of(entity),
                gpu: self.gpu_meshes.get(&id).expect("draw list is synced"),
                submit,
            });
        }
        for &prototype in &self.batch_opaque {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            draws.push(OpaqueDraw {
                albedo: albedo_of(entity),
                gpu: &batch.gpu,
                submit: OpaqueSubmit::Mesh(mesh_submit(&main_cull.compacted, prototype)),
            });
        }
        let cache = match target {
            OpaqueTarget::Prepass(_) => &self.opaque_prepass,
            OpaqueTarget::World => &self.opaque_world,
        };
        if draws.len() < OPAQUE_BUNDLE_MIN_DRAWS {
            let empty = OpaqueBundles::default();
            match target {
                OpaqueTarget::Prepass(_) => self.opaque_prepass = empty,
                OpaqueTarget::World => self.opaque_world = empty,
            }
            return;
        }
        let nowrite = self.prepass_done && world.debug_shading & (16384 | 65536) == 0;
        let scene_bind = self.pipelines.scene_bind_group(uniform_slot);
        let pipeline = match target {
            OpaqueTarget::Prepass(_) => &self.pipelines.depth_prepass,
            OpaqueTarget::World if nowrite => &self.pipelines.opaque_nowrite,
            OpaqueTarget::World => &self.pipelines.opaque,
        };
        let world_pass = target == OpaqueTarget::World;
        let samples = stencil::scene_samples();
        let shared_same = cache
            .shared
            .as_ref()
            .is_some_and(|(bind, pipe, n)| bind == scene_bind && pipe == pipeline && *n == samples);
        let chunks: Vec<&[OpaqueDraw<'_>]> = draws.chunks(OPAQUE_BUNDLE_CHUNK).collect();
        // A chunk is kept when every draw in it is the same as last time.
        let keep: Vec<bool> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                shared_same
                    && cache.chunks.get(i).is_some_and(|c| {
                        c.keys.len() == chunk.len()
                            && c.keys.iter().zip(chunk.iter()).all(|(k, d)| k.matches(d))
                    })
            })
            .collect();
        let device = &self.device;
        let record = |chunk: &[OpaqueDraw<'_>]| {
            let color = [Some(HDR_FORMAT)];
            let mut bundle =
                device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some("opaque-meshes"),
                    color_formats: if world_pass { &color } else { &[] },
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: depth_format(),
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: samples,
                    multiview: None,
                });
            bundle.set_bind_group(0, scene_bind, &[]);
            bundle.set_pipeline(pipeline);
            for d in chunk {
                d.record(&mut bundle);
            }
            bundle.finish(&wgpu::RenderBundleDescriptor {
                label: Some("opaque-meshes"),
            })
        };
        let jobs: Vec<usize> = (0..chunks.len()).filter(|&i| !keep[i]).collect();
        let recorded: Vec<wgpu::RenderBundle> = {
            use rayon::prelude::*;
            crate::parallel::frame_pool()
                .install(|| jobs.par_iter().map(|&i| record(chunks[i])).collect())
        };
        let indirect_draws = draws
            .iter()
            .filter(|d| matches!(d.submit, OpaqueSubmit::Mesh(MeshSubmit::Compacted)))
            .count() as u32;
        let mut fresh: Vec<Option<OpaqueChunk>> = (0..chunks.len()).map(|_| None).collect();
        for (i, bundle) in jobs.into_iter().zip(recorded) {
            fresh[i] = Some(OpaqueChunk {
                keys: chunks[i].iter().map(OpaqueKey::of).collect(),
                bundle,
            });
        }
        let shared = (scene_bind.clone(), pipeline.clone(), samples);
        let n = chunks.len();
        drop(chunks);
        drop(draws);
        let cache = match target {
            OpaqueTarget::Prepass(_) => &mut self.opaque_prepass,
            OpaqueTarget::World => &mut self.opaque_world,
        };
        let mut old = std::mem::take(&mut cache.chunks).into_iter();
        for (i, f) in fresh.into_iter().enumerate() {
            let kept = old.next();
            cache.chunks.push(f.unwrap_or_else(|| {
                kept.unwrap_or_else(|| panic!("opaque chunk {i} kept but not cached"))
            }));
        }
        cache.shared = Some(shared);
        cache.frame = Some(OpaqueFrame {
            target,
            space,
            uniform_slot,
            chunks: n,
            indirect_draws,
        });
    }

    /// Lay the live view's opaque depth before shading, for SSAO and so the
    /// world pass shades each pixel once. Foliage cutouts and see-through
    /// surfaces stay out; they still draw, and test, in the world pass.
    fn encode_depth_prepass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        space: SpaceId,
        main_cull: &MainCullSelection,
        slot: usize,
        phase: PrepassPhase,
    ) {
        let depth = &self.depth_view.clone();
        let unseen = phase == PrepassPhase::Unseen;
        let (depth_load, stencil_load) = if unseen {
            (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
        } else {
            (wgpu::LoadOp::Clear(DEPTH_CLEAR), wgpu::LoadOp::Clear(0))
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(if unseen {
                "depth-prepass-unseen"
            } else {
                "depth-prepass"
            }),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: stencil::has_stencil().then_some(wgpu::Operations {
                    load: stencil_load,
                    store: wgpu::StoreOp::Store,
                }),
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        // Prepass draws are not counted in the frame stats.
        let mut indirect_draws = 0;
        let scene_bind = self.pipelines.scene_bind_group(slot);
        pass.set_bind_group(0, scene_bind, &[]);
        if unseen {
            // Phase 2: candidates not drawn in phase 1, each through its
            // occlusion draw (zero instances when hidden).
            let frame = main_cull
                .occlusion
                .as_ref()
                .expect("the unseen prepass runs only with occlusion culling");
            pass.set_pipeline(&self.pipelines.depth_prepass);
            for &id in &self.draw_opaque {
                if frame.first.contains(&id) {
                    continue;
                }
                let Some(&cslot) = frame.slots.get(&id) else {
                    continue;
                };
                let entity = world.entity(id).expect("draw list is live");
                let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
                let albedo = entity
                    .albedo
                    .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                    .unwrap_or(&self.pipelines.white_albedo);
                pass.set_bind_group(1, albedo, &[]);
                submit_occluded_draw(&mut pass, gpu, &frame.args, cslot, OPAQUE_INDIRECT_OFFSET);
            }
            return;
        }
        if phase == PrepassPhase::SeenOccluders {
            let frame = main_cull
                .occlusion
                .as_ref()
                .expect("the occluder prepass runs only with occlusion culling");
            // Ground hides things too (hills, voxel cliffs): its depth goes
            // into the pyramid with the meshes seen lately.
            if world.space_draws_environment(space) {
                if let Some(clip) = self.clipmap.as_ref() {
                    clip.draw_land_depth(&mut pass, &self.shadow.resource_bind);
                    pass.set_bind_group(0, scene_bind, &[]);
                }
            }
            if space == SpaceId::DEFAULT && self.voxel.is_drawing() {
                let draws = self.voxel.visible_in(&self.frustum);
                pass.set_pipeline(&self.pipelines.voxel_depth_prepass);
                pass.set_bind_group(1, &self.pipelines.white_albedo, &[]);
                self.voxel.draw(&mut pass, &draws);
            }
            pass.set_pipeline(&self.pipelines.depth_prepass);
            for &id in &self.draw_opaque {
                if !frame.first.contains(&id) {
                    continue;
                }
                let entity = world.entity(id).expect("draw list is live");
                let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
                let albedo = entity
                    .albedo
                    .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                    .unwrap_or(&self.pipelines.white_albedo);
                pass.set_bind_group(1, albedo, &[]);
                submit_mesh_draw(
                    &mut pass,
                    gpu,
                    MeshSubmit::Direct,
                    0..gpu.opaque_index_count as u32,
                    OPAQUE_INDIRECT_OFFSET,
                    &mut indirect_draws,
                );
            }
            return;
        }
        if world.space_draws_environment(space) {
            if let Some(clip) = self.clipmap.as_ref() {
                clip.draw_land_depth(&mut pass, &self.shadow.resource_bind);
                pass.set_bind_group(0, scene_bind, &[]);
            }
        }

        if space == SpaceId::DEFAULT && self.voxel.is_drawing() {
            let draws = self.voxel.visible_in(&self.frustum);
            pass.set_pipeline(&self.pipelines.voxel_depth_prepass);
            pass.set_bind_group(1, &self.pipelines.white_albedo, &[]);
            self.voxel.draw(&mut pass, &draws);
        }

        let bundles = self.opaque_prepass.frame.as_ref().filter(|f| {
            f.target == OpaqueTarget::Prepass(phase) && f.space == space && f.uniform_slot == slot
        });
        if let Some(frame) = bundles {
            pass.execute_bundles(
                self.opaque_prepass.chunks[..frame.chunks]
                    .iter()
                    .map(|c| &c.bundle),
            );
            // Bundles leave the pass with nothing bound.
            pass.set_bind_group(0, scene_bind, &[]);
        }
        pass.set_pipeline(&self.pipelines.depth_prepass);
        for &id in self.draw_opaque.iter().filter(|_| bundles.is_none()) {
            if !entity_in_space(world, id, space) {
                continue;
            }
            if phase == PrepassPhase::Seen
                && main_cull.occlusion_slot(id).is_some()
                && !main_cull
                    .occlusion
                    .as_ref()
                    .is_some_and(|f| f.first.contains(&id))
            {
                // Waits for phase 2 and the occlusion test.
                continue;
            }
            let entity = world.entity(id).expect("draw list is live");
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, id),
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut indirect_draws,
            );
        }
        for &prototype in self.batch_opaque.iter().filter(|_| bundles.is_none()) {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            submit_batch_draw(
                &mut pass,
                batch,
                Some(mesh_submit(&main_cull.compacted, prototype)),
                0..batch.gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut indirect_draws,
            );
        }

        pass.set_pipeline(&self.terrain.depth_prepass);
        // The same set the terrain bundle draws.
        for &id in &self.draw_terrain {
            let entity = world.entity(id).expect("draw list is live");
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                panic!("terrain entity {id} lost its terrain material");
            };
            let mat = self
                .gpu_materials
                .get(&mid)
                .unwrap_or_else(|| panic!("terrain entity {id} material is not synced"));
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                MeshSubmit::Direct,
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut indirect_draws,
            );
        }
        for &prototype in &self.batch_terrain {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                panic!("terrain batch {prototype} lost its terrain material");
            };
            let mat = self
                .gpu_materials
                .get(&mid)
                .unwrap_or_else(|| panic!("terrain batch {prototype} material is not synced"));
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_batch_draw(
                &mut pass,
                batch,
                Some(mesh_submit(&main_cull.compacted, prototype)),
                0..batch.gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut indirect_draws,
            );
        }

        pass.set_pipeline(&self.skinned.depth_prepass);
        for (id, anim) in world.animated_entities() {
            if anim.space() != space || world.room_hidden(*id) {
                continue;
            }
            let Some(gpu) = self.gpu_skinned.get(id) else {
                continue;
            };
            pass.set_bind_group(1, &gpu.joint_bind, &[]);
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            for mesh in &gpu.meshes {
                pass.set_bind_group(2, &mesh.albedo_bind, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    fn draw_space_scene(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        world: &World,
        space: SpaceId,
        main_cull: &MainCullSelection,
        view: PortalLevelView,
        scene: ScenePass,
    ) {
        let ScenePass {
            portal_mask,
            uniform_slot,
        } = scene;
        let PortalLevelView {
            clip_plane: _,
            draw_environment,
            use_cull,
        } = view;
        // Profile only the top-level view (portal views reuse this code).
        let stamp = !portal_mask && use_cull;
        // After a depth prepass the opaque draws only test depth: with depth
        // writes on, shaders that can discard (LOD dither, leaves) lose early
        // depth testing and shade every hidden fragment (2.4 -> 1.2 ms for
        // lod_forest on an RTX 5070). Leaf edges the prepass left out still
        // draw, they just don't occlude each other.
        let nowrite = stamp && self.prepass_done && world.debug_shading & (16384 | 65536) == 0;
        self.mark_scene(pass, stamp, SceneMark::Begin);
        pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);

        if draw_environment && world.sky().is_some() {
            pass.set_pipeline(if portal_mask {
                &self.sky.pipeline_portal
            } else {
                &self.sky.pipeline
            });
            pass.set_bind_group(0, self.sky.bind_group(uniform_slot), &[]);
            pass.draw(0..3, 0..1);
            pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
        }

        self.mark_scene(pass, stamp, SceneMark::Sky);
        if draw_environment {
            if let Some(clip) = self.clipmap.as_ref() {
                clip.draw_land(pass, &self.shadow.resource_bind, portal_mask);
            }
        }
        // Every stamp is written each frame: the readback waits for all.
        self.mark_scene(pass, stamp, SceneMark::Land);
        if draw_environment {
            if let Some(clip) = self.clipmap.as_ref() {
                if !portal_mask {
                    clip.draw_grass(pass, &self.shadow.resource_bind);
                }
                // The clipmap owns group 0 while it draws; give it back.
                pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
            }
        }

        self.mark_scene(pass, stamp, SceneMark::Grass);
        if !portal_mask && space == SpaceId::DEFAULT && self.voxel.is_drawing() {
            let draws = self.voxel.visible_in(&self.frustum);
            pass.set_pipeline(if nowrite {
                &self.pipelines.voxel_nowrite
            } else {
                &self.pipelines.voxel_opaque
            });
            pass.set_bind_group(1, &self.pipelines.white_albedo, &[]);
            self.voxel.draw(pass, &draws);
        }

        self.mark_scene(pass, stamp, SceneMark::Voxel);
        // The top-level view's opaque meshes, when recorded on the frame's
        // threads for exactly this space, uniform slot and pipeline.
        let pipeline = if portal_mask {
            &self.pipelines.opaque_portal
        } else if nowrite {
            &self.pipelines.opaque_nowrite
        } else {
            &self.pipelines.opaque
        };
        let bundles = self
            .opaque_world
            .frame
            .as_ref()
            .filter(|f| {
                stamp
                    && f.target == OpaqueTarget::World
                    && f.space == space
                    && f.uniform_slot == uniform_slot
                    && self
                        .opaque_world
                        .shared
                        .as_ref()
                        .is_some_and(|(_, p, _)| p == pipeline)
            })
            .map(|f| (f.chunks, f.indirect_draws));
        if let Some((chunks, indirect_draws)) = bundles {
            pass.execute_bundles(self.opaque_world.chunks[..chunks].iter().map(|c| &c.bundle));
            self.gpu_stats.indirect_draws += indirect_draws;
            pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
        }
        let direct = bundles.is_none();
        pass.set_pipeline(if portal_mask {
            &self.pipelines.opaque_portal
        } else if nowrite {
            &self.pipelines.opaque_nowrite
        } else {
            &self.pipelines.opaque
        });
        for &id in self.draw_opaque.iter().filter(|_| direct) {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let entity = world.entity(id).expect("draw list is live");
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            if let Some((args, cslot)) = stamp.then(|| main_cull.occlusion_slot(id)).flatten() {
                submit_occluded_draw(pass, gpu, args, cslot, OPAQUE_INDIRECT_OFFSET);
                continue;
            }
            submit_mesh_draw(
                pass,
                gpu,
                if use_cull {
                    mesh_submit(&main_cull.compacted, id)
                } else {
                    MeshSubmit::Direct
                },
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }
        self.mark_scene(pass, stamp, SceneMark::Meshes);
        for &prototype in self.batch_opaque.iter().filter(|_| direct) {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            let gpu = &batch.gpu;
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            submit_batch_draw(
                pass,
                batch,
                use_cull.then(|| mesh_submit(&main_cull.compacted, prototype)),
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        self.mark_scene(pass, stamp, SceneMark::Batches);
        if use_cull {
            if let Some(bundle) = self.terrain_bundle.as_ref() {
                pass.execute_bundles(std::iter::once(bundle));
            }
        }
        pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
        pass.set_pipeline(if portal_mask {
            &self.terrain.opaque_portal
        } else {
            &self.terrain.opaque
        });
        if !use_cull {
            for &id in &self.draw_terrain {
                if !entity_in_space(world, id, space) {
                    continue;
                }
                let entity = world.entity(id).expect("draw list is live");
                let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
                let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                    panic!("terrain entity {id} lost its terrain material");
                };
                let mat = self
                    .gpu_materials
                    .get(&mid)
                    .unwrap_or_else(|| panic!("terrain entity {id} material is not synced"));
                pass.set_bind_group(1, &mat.bind_group, &[]);
                submit_mesh_draw(
                    pass,
                    gpu,
                    MeshSubmit::Direct,
                    0..gpu.opaque_index_count as u32,
                    OPAQUE_INDIRECT_OFFSET,
                    &mut self.gpu_stats.indirect_draws,
                );
            }
        }
        for &prototype in &self.batch_terrain {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            let gpu = &batch.gpu;
            let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                panic!("terrain batch {prototype} lost its terrain material");
            };
            let mat = self
                .gpu_materials
                .get(&mid)
                .unwrap_or_else(|| panic!("terrain batch {prototype} material is not synced"));
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_batch_draw(
                pass,
                batch,
                use_cull.then(|| mesh_submit(&main_cull.compacted, prototype)),
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        self.mark_scene(pass, stamp, SceneMark::Terrain);
        pass.set_pipeline(if portal_mask {
            &self.skinned.opaque_portal
        } else {
            &self.skinned.opaque
        });
        for (id, anim) in world.animated_entities() {
            if anim.space() != space || world.room_hidden(*id) {
                continue;
            }
            let Some(gpu) = self.gpu_skinned.get(id) else {
                continue;
            };
            pass.set_bind_group(1, &gpu.joint_bind, &[]);
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            for mesh in &gpu.meshes {
                pass.set_bind_group(2, &mesh.albedo_bind, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        if space == SpaceId::DEFAULT {
            self.crowd.draw(pass, portal_mask);
        }
        pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);

        self.mark_scene(pass, stamp, SceneMark::Skinned);
        pass.set_pipeline(if portal_mask {
            &self.pipelines.transparent_portal
        } else {
            &self.pipelines.transparent
        });
        for &id in &self.draw_transparent {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let entity = world.entity(id).expect("draw list is live");
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            if let Some((args, cslot)) = stamp.then(|| main_cull.occlusion_slot(id)).flatten() {
                submit_occluded_draw(pass, gpu, args, cslot, TRANSLUCENT_INDIRECT_OFFSET);
                continue;
            }
            submit_mesh_draw(
                pass,
                gpu,
                if use_cull {
                    mesh_submit(&main_cull.compacted, id)
                } else {
                    MeshSubmit::Direct
                },
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }
        for &prototype in &self.batch_transparent {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            let gpu = &batch.gpu;
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            submit_batch_draw(
                pass,
                batch,
                use_cull.then(|| mesh_submit(&main_cull.compacted, prototype)),
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        pass.set_pipeline(if portal_mask {
            &self.water.blend_portal
        } else {
            &self.water.blend
        });
        for &id in &self.draw_water {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let entity = world.entity(id).expect("draw list is live");
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            let Some(SurfaceMaterialRef::Water(mid)) = entity.material else {
                continue;
            };
            let Some(mat) = self.gpu_water_materials.get(&mid) else {
                continue;
            };
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_mesh_draw(
                pass,
                gpu,
                if use_cull {
                    mesh_submit(&main_cull.compacted, id)
                } else {
                    MeshSubmit::Direct
                },
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }
        for &prototype in &self.batch_water {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced");
            let gpu = &batch.gpu;
            let Some(SurfaceMaterialRef::Water(mid)) = entity.material else {
                panic!("water batch {prototype} lost its water material");
            };
            let mat = self
                .gpu_water_materials
                .get(&mid)
                .unwrap_or_else(|| panic!("water batch {prototype} material is not synced"));
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_batch_draw(
                pass,
                batch,
                use_cull.then(|| mesh_submit(&main_cull.compacted, prototype)),
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        self.mark_scene(pass, stamp, SceneMark::Transparent);
        self.ribbons
            .draw(pass, self.pipelines.scene_bind_group(uniform_slot));

        if draw_environment {
            if let Some(clip) = self.clipmap.as_ref() {
                clip.draw_water(pass, &self.shadow.resource_bind, portal_mask);
            }
        }
        self.mark_scene(pass, stamp, SceneMark::End);
    }

    /// Record the shadow bundles whose casters changed, all at once on the
    /// frame's threads. A view's meshes are bundled per source group and
    /// its instance batches in one more bundle; only bundles whose own
    /// casters changed are recorded. Each reads only shared, unchanging
    /// renderer state and goes to its own place; all are done before any
    /// pass uses them.
    fn prepare_shadow_bundles(&mut self, slots: &[usize], selections: &[ShadowCullSelection]) {
        // (slot, group, draws, batches); `group` is the place in the
        // view's new group list (unused for batches).
        let mut jobs: Vec<(usize, usize, &[ShadowDraw], bool)> = Vec::new();
        let mut kept: Vec<Vec<Option<ShadowGroupBundle>>> = Vec::new();
        for (&slot, selection) in slots.iter().zip(selections) {
            let view = &mut self.shadow_bundles[slot];
            let mut old: HashMap<u32, ShadowGroupBundle> = std::mem::take(&mut view.groups)
                .into_iter()
                .map(|g| (g.group, g))
                .collect();
            let mut list = Vec::new();
            for run in selection.draws.chunk_by(|a, b| a.group == b.group) {
                let reuse = old.remove(&run[0].group).filter(|g| g.draws == run);
                if reuse.is_none() {
                    jobs.push((slot, list.len(), run, false));
                }
                list.push(reuse);
            }
            kept.push(list);
            if view.batch_draws != selection.batch_draws {
                view.batch_draws.clone_from(&selection.batch_draws);
                view.batches = None;
                if !selection.batch_draws.is_empty() {
                    jobs.push((slot, 0, &selection.batch_draws, true));
                }
            }
        }
        let ctx = ShadowBundleCtx {
            device: &self.device,
            shadow: &self.shadow,
            gpu_meshes: &self.gpu_meshes,
            gpu_mesh_albedo: &self.gpu_mesh_albedo,
            gpu_instance_batches: &self.gpu_instance_batches,
        };
        let record = |&(slot, _, draws, batches): &(usize, usize, &[ShadowDraw], bool)| {
            ctx.record(slot, draws, batches)
        };
        let recorded: Vec<wgpu::RenderBundle> = if jobs.len() > 1 {
            use rayon::prelude::*;
            crate::parallel::frame_pool().install(|| jobs.par_iter().map(record).collect())
        } else {
            jobs.iter().map(record).collect()
        };
        let index: HashMap<usize, usize> = slots.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        for ((slot, place, draws, batches), bundle) in jobs.into_iter().zip(recorded) {
            if batches {
                self.shadow_bundles[slot].batches = Some(bundle);
            } else {
                kept[index[&slot]][place] = Some(ShadowGroupBundle {
                    group: draws[0].group,
                    draws: draws.to_vec(),
                    bundle,
                });
            }
        }
        for (&slot, list) in slots.iter().zip(kept) {
            self.shadow_bundles[slot].groups = list
                .into_iter()
                .map(|g| g.expect("every shadow group is kept or recorded"))
                .collect();
        }
    }

    /// The shadow views (sun cascades, lamp layers), each in its own
    /// command buffer, recorded side by side on the frame's threads.
    ///
    /// Order is kept by construction: the commands `encoder` holds so far
    /// are closed into `recorded` first, then each view's buffer follows in
    /// view order, and `encoder` starts afresh for what comes after. The
    /// queue runs them in that order. Each view's instance cull (which
    /// shares buffers between views) is put in its own view's buffer, ahead
    /// of its pass, here on this thread; the threads only add the passes,
    /// reading renderer state that nothing changes meanwhile.
    fn encode_shadow_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        recorded: &mut Vec<wgpu::CommandBuffer>,
        world: &World,
    ) {
        let sun = world.shadows().is_some();
        let lamp_views = self.shadow.lamps.frame_views.clone();
        if !sun && lamp_views.is_empty() {
            return;
        }
        let far = world.shadows().map(|s| s.cascade_end_m[2]).unwrap_or(120.0);
        let focus = world.camera().target();
        let submit = world.instance_submit();
        // One frustum per view: the sun's cascades (when on), then lamp layers.
        let mut frustums: Vec<Frustum> = Vec::new();
        let mut slots: Vec<usize> = Vec::new();
        if sun {
            for cascade in (0..3).filter(|&c| self.shadow_draw[c]) {
                frustums.push(Frustum::from_view_projection(self.shadow_vp[cascade]));
                slots.push(cascade);
            }
        }
        self.gpu_stats.shadow_cascades_drawn =
            self.shadow_draw.iter().filter(|d| **d).count() as u32 * u32::from(sun);
        for (i, vp) in lamp_views.iter().enumerate() {
            frustums.push(Frustum::from_view_projection(*vp));
            slots.push(3 + i);
        }
        if frustums.is_empty() {
            return;
        }
        let mut t = std::time::Instant::now();
        let mut selections = self.select_shadow_culls(world, &frustums);
        self.cpu_spans.mark("shadows.select", &mut t);
        let default_space = world.living_in() == SpaceId::DEFAULT;
        let voxel_on = default_space && self.voxel.is_drawing();
        // Sun cascades with the static cache: static casters (not changed
        // lately, not swaying) go to their own selection, drawn into the
        // cache layer only when it is out of date.
        let caching = sun && world.shadow_cache() && shadow_cache_env();
        let mut uses = vec![ShadowCacheUse::Plain; slots.len()];
        let mut statics: Vec<(usize, ShadowCullSelection)> = Vec::new();
        if caching {
            if self.shadow.static_cache.is_none() {
                let size = self.shadow.depth_texture.width();
                self.shadow.static_cache = Some(shadow::StaticShadowCache::new(&self.device, size));
            }
            for (i, &slot) in slots.iter().enumerate() {
                if slot >= 3 {
                    continue;
                }
                let (fixed, moving) = self.split_static_casters(&selections[i]);
                selections[i] = moving;
                let key = shadow::StaticShadowKey {
                    vp: self.shadow_vp[slot].to_cols_array().map(f32::to_bits),
                    epoch: self.static_shadow_epoch,
                    voxel_epoch: if voxel_on {
                        self.voxel.epoch()
                    } else {
                        u64::MAX
                    },
                    casters: caster_hash(&fixed, world.living_in()),
                };
                let cache = self
                    .shadow
                    .static_cache
                    .as_mut()
                    .expect("static shadow cache made above");
                let fresh = cache.keys[slot].is_some_and(|(k, at)| {
                    k == key && self.frame_no.saturating_sub(at) < STATIC_SHADOW_MAX_AGE
                });
                if fresh {
                    uses[i] = ShadowCacheUse::Hit;
                    self.gpu_stats.shadow_cache_hits += 1;
                } else {
                    uses[i] = ShadowCacheUse::Miss;
                    cache.keys[slot] = Some((key, self.frame_no));
                    statics.push((i, fixed));
                }
            }
        } else {
            self.shadow.static_cache = None;
        }
        // Bundles: every view's (moving) casters, and the static casters of
        // the cascades drawn into the cache this frame.
        let mut bundle_slots = slots.clone();
        let mut bundle_selections: Vec<ShadowCullSelection> = Vec::new();
        bundle_selections.append(&mut selections);
        let mut static_compact: HashMap<usize, Vec<EntityId>> = HashMap::new();
        for (i, fixed) in statics {
            bundle_slots.push(STATIC_SHADOW_SLOT + slots[i]);
            static_compact.insert(i, fixed.compact_batches.clone());
            self.gpu_stats.indirect_draws += fixed
                .batch_draws
                .iter()
                .filter(|draw| draw.submit == MeshSubmit::Compacted)
                .count() as u32;
            bundle_selections.push(fixed);
        }
        self.prepare_shadow_bundles(&bundle_slots, &bundle_selections);
        self.cpu_spans.mark("shadows.bundles", &mut t);
        let selections = &bundle_selections[..slots.len()];

        let new_encoder = |device: &wgpu::Device, label| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        };
        let before = std::mem::replace(encoder, new_encoder(&self.device, "after-shadows"));
        recorded.push(before.finish());
        let mut views = Vec::new();
        for (i, ((frustum, selection), &slot)) in
            frustums.iter().zip(selections).zip(&slots).enumerate()
        {
            let mut view_encoder = new_encoder(&self.device, "shadow-view");
            if submit == InstanceSubmit::GpuIndirect {
                // A cache miss also draws the static batches: cull them too.
                let mut compact = selection.compact_batches.clone();
                if let Some(more) = static_compact.get(&i) {
                    compact.extend_from_slice(more);
                }
                self.dispatch_instance_cull(
                    &mut view_encoder,
                    frustum,
                    &compact,
                    CullView::Shadow,
                    None,
                );
            }
            self.gpu_stats.indirect_draws += u32::try_from(
                selection
                    .batch_draws
                    .iter()
                    .filter(|draw| draw.submit == MeshSubmit::Compacted)
                    .count(),
            )
            .expect("shadow indirect draw count exceeds u32");
            let skinned = world
                .animated_entities()
                .filter(|(id, anim)| anim.space() == world.living_in() && !world.room_hidden(**id))
                .filter(|(_, anim)| {
                    let t = anim.transform().w_axis.truncate();
                    if slot < 3 {
                        t.distance(focus) <= far + 16.0
                    } else {
                        frustum.intersects_sphere(t + glam::Vec3::Y, 1.5)
                    }
                })
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            // Voxel ground is static: drawn with the static casters when the
            // cache is used, so not again on a hit.
            let voxel = (voxel_on && uses[i] != ShadowCacheUse::Hit)
                .then(|| self.voxel.visible_in(frustum));
            views.push((slot, uses[i], view_encoder, skinned, voxel));
        }
        let this = &*self;
        let views: Vec<ShadowViewWork<'_>> = views
            .into_iter()
            .map(|(slot, cache, encoder, skinned, voxel)| ShadowViewWork {
                slot,
                cache,
                encoder,
                skinned: skinned
                    .iter()
                    .filter_map(|id| this.gpu_skinned.get(id))
                    .collect(),
                voxel,
            })
            .collect();
        let record = |work: ShadowViewWork<'_>| -> wgpu::CommandBuffer {
            let ShadowViewWork {
                slot,
                cache,
                mut encoder,
                skinned,
                voxel,
            } = work;
            let (target, view_bind) = if slot < 3 {
                (
                    &this.shadow.layer_views[slot],
                    &this.shadow.cascade_binds[slot],
                )
            } else {
                (
                    &this.shadow.lamps.layer_views[slot - 3],
                    &this.shadow.lamps.layer_binds[slot - 3],
                )
            };
            let draw_voxel = |pass: &mut wgpu::RenderPass<'_>| {
                if let Some(draws) = voxel.as_ref() {
                    pass.set_pipeline(&this.shadow.voxel_pipeline);
                    pass.set_bind_group(0, view_bind, &[]);
                    this.voxel.draw(pass, draws);
                }
            };
            let depth_pass = |encoder: &'_ mut wgpu::CommandEncoder,
                              view: &wgpu::TextureView,
                              load: wgpu::LoadOp<f32>,
                              label: &'static str| {
                encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view,
                            depth_ops: Some(wgpu::Operations {
                                load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    })
                    .forget_lifetime()
            };
            let mut load = wgpu::LoadOp::Clear(1.0);
            if cache != ShadowCacheUse::Plain {
                let cached = this
                    .shadow
                    .static_cache
                    .as_ref()
                    .expect("static shadow cache in use");
                if cache == ShadowCacheUse::Miss {
                    // The static casters into the cache layer first.
                    let mut pass = depth_pass(
                        &mut encoder,
                        &cached.views[slot],
                        wgpu::LoadOp::Clear(1.0),
                        "shadow-csm-static",
                    );
                    let view = &this.shadow_bundles[STATIC_SHADOW_SLOT + slot];
                    pass.execute_bundles(
                        view.groups
                            .iter()
                            .map(|g| &g.bundle)
                            .chain(view.batches.as_ref()),
                    );
                    draw_voxel(&mut pass);
                    drop(pass);
                }
                shadow::copy_layer(
                    &mut encoder,
                    &cached.texture,
                    &this.shadow.depth_texture,
                    slot as u32,
                );
                load = wgpu::LoadOp::Load;
            }
            let mut pass = depth_pass(
                &mut encoder,
                target,
                load,
                if slot < 3 {
                    "shadow-csm"
                } else {
                    "shadow-lamp"
                },
            );
            let view = &this.shadow_bundles[slot];
            pass.execute_bundles(
                view.groups
                    .iter()
                    .map(|g| &g.bundle)
                    .chain(view.batches.as_ref()),
            );
            if cache == ShadowCacheUse::Plain {
                draw_voxel(&mut pass);
            }
            pass.set_pipeline(&this.shadow.skinned_pipeline);
            pass.set_bind_group(0, view_bind, &[]);
            for gpu in skinned {
                pass.set_bind_group(1, &gpu.joint_bind, &[]);
                pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
                for mesh in &gpu.meshes {
                    pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                    pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
            if slot < 3 && default_space {
                this.crowd.draw_shadow(&mut pass, view_bind);
            }
            drop(pass);
            encoder.finish()
        };
        let buffers: Vec<wgpu::CommandBuffer> = {
            use rayon::prelude::*;
            crate::parallel::frame_pool().install(|| views.into_par_iter().map(record).collect())
        };
        // `collect` keeps the views' order whatever thread finished first.
        recorded.extend(buffers);
    }

    /// Split a cascade's casters into static ones (unchanged for
    /// `STATIC_SHADOW_SETTLE` frames, not swaying in the wind) and moving
    /// ones. Order within each part is kept.
    fn split_static_casters(
        &self,
        selection: &ShadowCullSelection,
    ) -> (ShadowCullSelection, ShadowCullSelection) {
        let settled = |id: &EntityId| {
            self.moved_at
                .get(id)
                .is_none_or(|&at| self.frame_no.saturating_sub(at) > STATIC_SHADOW_SETTLE)
        };
        let mesh_static =
            |d: &ShadowDraw| settled(&d.id) && self.gpu_meshes.get(&d.id).is_some_and(|g| !g.sways);
        let batch_static = |id: &EntityId| {
            settled(id)
                && self
                    .gpu_instance_batches
                    .get(id)
                    .is_some_and(|b| !b.gpu.sways)
        };
        let empty = || ShadowCullSelection {
            draws: Vec::new(),
            batch_draws: Vec::new(),
            compact_batches: Vec::new(),
        };
        let (mut fixed, mut moving) = (empty(), empty());
        for d in &selection.draws {
            if mesh_static(d) {
                fixed.draws.push(*d);
            } else {
                moving.draws.push(*d);
            }
        }
        for d in &selection.batch_draws {
            if batch_static(&d.id) {
                fixed.batch_draws.push(*d);
            } else {
                moving.batch_draws.push(*d);
            }
        }
        for id in &selection.compact_batches {
            if batch_static(id) {
                fixed.compact_batches.push(*id);
            } else {
                moving.compact_batches.push(*id);
            }
        }
        (fixed, moving)
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    /// This frame's occlusion candidates: single meshes the main view draws.
    fn occlusion_candidates(&self, world: &World, space: SpaceId) -> Vec<EntityId> {
        self.draw_opaque
            .iter()
            .copied()
            .filter(|&id| {
                entity_in_space(world, id, space)
                    && self
                        .gpu_meshes
                        .get(&id)
                        .is_some_and(|gpu| gpu.world_box.is_some() && gpu.instance_count > 0)
            })
            .collect()
    }

    fn begin_occlusion(&mut self, ids: &[EntityId]) -> occlusion::OcclusionFrame {
        let rows: Vec<_> = ids
            .iter()
            .map(|id| {
                let gpu = self
                    .gpu_meshes
                    .get(id)
                    .expect("occlusion candidate is synced");
                occlusion::CandidateMesh {
                    id: *id,
                    aabb: gpu.world_box.expect("occlusion candidate has a box"),
                    opaque_index_count: u32::try_from(gpu.opaque_index_count)
                        .expect("index count exceeds u32"),
                    index_count: u32::try_from(gpu.index_count).expect("index count exceeds u32"),
                    instance_count: u32::try_from(gpu.instance_count)
                        .expect("instance count exceeds u32"),
                    first_index: gpu.first_index,
                    base_vertex: gpu.base_vertex,
                }
            })
            .collect();
        let counts = self.occlusion.counts();
        self.gpu_stats.occlusion_tested = counts.tested;
        self.gpu_stats.occlusion_hidden = counts.hidden;
        self.occlusion.begin(&self.device, &self.queue, &rows)
    }

    fn select_main_culls(&mut self, frustum: &Frustum) -> MainCullSelection {
        let mut seen = HashSet::new();
        let mut ids = Vec::new();
        let mut compacted = HashSet::new();
        for list in [
            &self.batch_opaque,
            &self.batch_terrain,
            &self.batch_transparent,
            &self.batch_water,
        ] {
            for &prototype in list {
                if !seen.insert(prototype) {
                    continue;
                }
                let batch = self
                    .gpu_instance_batches
                    .get(&prototype)
                    .expect("batch draw list is synced");
                let gpu = &batch.gpu;
                let bounds = gpu
                    .bounds
                    .unwrap_or_else(|| panic!("instance batch {prototype} has no bounds"));
                match frustum.classify(bounds) {
                    Visibility::Outside => {
                        panic!("draw list contains outside instance batch {prototype}")
                    }
                    Visibility::Intersecting => {
                        ids.push(prototype);
                        compacted.insert(prototype);
                    }
                    Visibility::Inside if batch.has_holes() => {
                        ids.push(prototype);
                        compacted.insert(prototype);
                    }
                    Visibility::Inside => self.gpu_stats.direct_inside_batches += 1,
                }
            }
        }
        MainCullSelection {
            ids,
            compacted,
            occlusion: None,
        }
    }

    fn select_shadow_culls(
        &mut self,
        world: &World,
        light_frustums: &[Frustum],
    ) -> Vec<ShadowCullSelection> {
        if self.shadow_source_epoch != world.render_epoch() {
            self.rebuild_shadow_sources(world);
            self.shadow_source_epoch = world.render_epoch();
        }
        let mut selections: Vec<ShadowCullSelection> = light_frustums
            .iter()
            .map(|_| ShadowCullSelection {
                draws: Vec::new(),
                batch_draws: Vec::new(),
                compact_batches: Vec::new(),
            })
            .collect();
        for (source, &id) in self.shadow_mesh_sources.iter().enumerate() {
            let group = (source / SHADOW_GROUP) as u32;
            if !entity_in_space(world, id, world.living_in()) {
                continue;
            }
            let gpu = self
                .gpu_meshes
                .get(&id)
                .expect("shadow mesh source is synced");
            let Some(bounds) = gpu.bounds else {
                continue;
            };
            let albedo = world.entity(id).ok().and_then(|e| e.albedo);
            for (selection, frustum) in selections.iter_mut().zip(light_frustums) {
                match frustum.classify(bounds) {
                    Visibility::Outside => {}
                    Visibility::Inside | Visibility::Intersecting => {
                        selection.draws.push(ShadowDraw {
                            id,
                            group,
                            submit: MeshSubmit::Direct,
                            albedo,
                        });
                    }
                }
            }
        }
        for &prototype in &self.shadow_batch_sources {
            if !entity_in_space(world, prototype, world.living_in()) {
                continue;
            }
            let batch = self
                .gpu_instance_batches
                .get(&prototype)
                .expect("shadow batch source is synced");
            let gpu = &batch.gpu;
            let Some(bounds) = gpu.bounds else {
                continue;
            };
            let albedo = world.entity(prototype).ok().and_then(|e| e.albedo);
            for (selection, frustum) in selections.iter_mut().zip(light_frustums) {
                match frustum.classify(bounds) {
                    Visibility::Outside => {}
                    Visibility::Inside if batch.has_holes() => {
                        selection.compact_batches.push(prototype);
                        selection.batch_draws.push(ShadowDraw {
                            id: prototype,
                            group: 0,
                            submit: MeshSubmit::Compacted,
                            albedo,
                        });
                    }
                    Visibility::Inside => {
                        self.gpu_stats.direct_inside_batches += 1;
                        selection.batch_draws.push(ShadowDraw {
                            id: prototype,
                            group: 0,
                            submit: MeshSubmit::Direct,
                            albedo,
                        });
                    }
                    Visibility::Intersecting => {
                        selection.compact_batches.push(prototype);
                        selection.batch_draws.push(ShadowDraw {
                            id: prototype,
                            group: 0,
                            submit: MeshSubmit::Compacted,
                            albedo,
                        });
                    }
                }
            }
        }
        selections
    }

    fn rebuild_shadow_sources(&mut self, world: &World) {
        self.shadow_mesh_sources.clear();
        self.shadow_batch_sources.clear();
        let gpu_batches = world.instance_submit() == InstanceSubmit::GpuIndirect;
        for (id, entity) in world.entities() {
            if world.is_portal_surface(id)
                || (gpu_batches && entity.instanced())
                || !entity.casts_shadow()
                || !material_casts_shadow(entity.material())
            {
                continue;
            }
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            if gpu.instance_count > 0 && gpu.opaque_index_count > 0 {
                self.shadow_mesh_sources.push(id);
            }
        }
        if gpu_batches {
            for (&prototype, batch) in &self.gpu_instance_batches {
                let entity = world
                    .entity(prototype)
                    .unwrap_or_else(|_| panic!("instance batch prototype {prototype} disappeared"));
                if entity.casts_shadow()
                    && material_casts_shadow(entity.material())
                    && batch.gpu.instance_count > 0
                    && batch.gpu.opaque_index_count > 0
                {
                    self.shadow_batch_sources.push(prototype);
                }
            }
        }
    }

    fn dispatch_instance_cull(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frustum: &Frustum,
        ids: &[EntityId],
        view: CullView,
        hiz: Option<&wgpu::BindGroup>,
    ) {
        let jobs: Vec<CullJob<'_>> = ids
            .iter()
            .map(|id| {
                let gpu = &self
                    .gpu_instance_batches
                    .get(id)
                    .expect("instance batch cull list is synced")
                    .gpu;
                if gpu.instance_count == 0 {
                    panic!("instance cull list included batch {id} with no instances");
                }
                let local = gpu.local_sphere().unwrap_or_else(|| {
                    panic!("instance batch {id} has instances but no local sphere")
                });
                let draws = match view {
                    CullView::Main if gpu.opaque_index_count < gpu.index_count => {
                        CullDraws::OpaqueAndTranslucent
                    }
                    CullView::Main | CullView::Shadow => CullDraws::OpaqueOnly,
                };
                let active_ranges = &self
                    .gpu_instance_batches
                    .get(id)
                    .expect("instance batch cull list is synced")
                    .active_ranges;
                CullJob {
                    gpu,
                    local,
                    draws,
                    active_ranges,
                }
            })
            .collect();
        self.instance_cull
            .dispatch(&self.queue, encoder, frustum, &jobs, hiz);
        let job_count = u32::try_from(jobs.len()).expect("cull dispatch count exceeds u32");
        self.gpu_stats.compute_culls += job_count;
        self.gpu_stats.boundary_batches += job_count;
        for job in jobs {
            for range in job.active_ranges {
                self.gpu_stats.cull_instances += u64::from(range.count);
                self.gpu_stats.cull_workgroups += u64::from(range.count).div_ceil(WORKGROUP as u64);
            }
        }
    }

    fn begin_instance_cull_frame(&mut self, world: &World) {
        if world.instance_submit() != InstanceSubmit::GpuIndirect {
            return;
        }
        let sun = if world.shadows().is_some() { 3 } else { 0 };
        // The main view culls twice with occlusion on: frustum, then depth.
        let main = if self.occlusion_active { 2 } else { 1 };
        let views = main + sun + self.shadow.lamps.frame_views.len();
        let active_ranges = self
            .gpu_instance_batches
            .values()
            .map(|batch| batch.active_ranges.len())
            .sum::<usize>();
        let max_jobs = active_ranges
            .checked_mul(views)
            .expect("instance-cull frame job count overflow");
        self.instance_cull.begin_frame(&self.device, max_jobs);
    }

    fn prepare_terrain_bundle(&mut self, world: &World) {
        if self.terrain_bundle_ids == self.draw_terrain {
            return;
        }
        self.terrain_bundle_ids.clone_from(&self.draw_terrain);
        if self.draw_terrain.is_empty() {
            self.terrain_bundle = None;
            return;
        }

        let mut bundle =
            self.device
                .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some("visible-terrain-bundle"),
                    color_formats: &[Some(HDR_FORMAT)],
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: depth_format(),
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: stencil::scene_samples(),
                    multiview: None,
                });
        bundle.set_bind_group(0, self.pipelines.scene_bind_group(0), &[]);
        bundle.set_pipeline(&self.terrain.opaque);
        for &id in &self.draw_terrain {
            let entity = world
                .entity(id)
                .unwrap_or_else(|_| panic!("terrain bundle entity {id} disappeared"));
            let Some(SurfaceMaterialRef::Terrain(material_id)) = entity.material else {
                panic!("terrain bundle entity {id} lost its terrain material");
            };
            let material = self
                .gpu_materials
                .get(&material_id)
                .unwrap_or_else(|| panic!("terrain bundle entity {id} material is not synced"));
            let gpu = self
                .gpu_meshes
                .get(&id)
                .unwrap_or_else(|| panic!("terrain bundle entity {id} is not synced"));
            let instance_count =
                u32::try_from(gpu.instance_count).expect("terrain instance count exceeds u32");
            let opaque_index_count = u32::try_from(gpu.opaque_index_count)
                .expect("terrain opaque index count exceeds u32");
            bundle.set_bind_group(1, &material.bind_group, &[]);
            bundle.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
            bundle.set_vertex_buffer(1, gpu.instance_slice());
            bundle.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            bundle.draw_indexed(
                gpu.indices(0..opaque_index_count),
                gpu.base_vertex,
                0..instance_count,
            );
        }
        self.terrain_bundle = Some(bundle.finish(&wgpu::RenderBundleDescriptor {
            label: Some("visible-terrain-bundle"),
        }));
    }

    fn collect_draws(&mut self, world: &World, space: SpaceId, frustum_cull: bool) {
        if self.draw_source_epoch != world.render_epoch() {
            self.rebuild_draw_sources(world);
            self.draw_source_epoch = world.render_epoch();
        }

        self.draw_opaque.clear();
        self.draw_terrain.clear();
        self.draw_transparent.clear();
        self.draw_water.clear();
        self.batch_opaque.clear();
        self.batch_terrain.clear();
        self.batch_transparent.clear();
        self.batch_water.clear();

        let frustum = self.frustum;
        let visible = |gpu: &GpuMesh| {
            gpu.instance_count > 0
                && (!frustum_cull || gpu.bounds.is_some_and(|b| frustum.intersects(b)))
        };
        for &id in &self.source_opaque {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let gpu = self.gpu_meshes.get(&id).expect("opaque source is synced");
            if visible(gpu) {
                self.draw_opaque.push(id);
            }
        }
        for &id in &self.source_terrain {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let gpu = self.gpu_meshes.get(&id).expect("terrain source is synced");
            if visible(gpu) {
                self.draw_terrain.push(id);
            }
        }
        for &id in &self.source_transparent {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let gpu = self
                .gpu_meshes
                .get(&id)
                .expect("transparent source is synced");
            if visible(gpu) {
                self.draw_transparent.push(id);
            }
        }
        for &id in &self.source_water {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let gpu = self.gpu_meshes.get(&id).expect("water source is synced");
            if visible(gpu) {
                self.draw_water.push(id);
            }
        }
        for &prototype in &self.batch_source_opaque {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("opaque batch source is synced")
                .gpu;
            if visible(gpu) {
                self.batch_opaque.push(prototype);
            }
        }
        for &prototype in &self.batch_source_terrain {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("terrain batch source is synced")
                .gpu;
            if visible(gpu) {
                self.batch_terrain.push(prototype);
            }
        }
        for &prototype in &self.batch_source_transparent {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("transparent batch source is synced")
                .gpu;
            if visible(gpu) {
                self.batch_transparent.push(prototype);
            }
        }
        for &prototype in &self.batch_source_water {
            if !entity_in_space(world, prototype, space) {
                continue;
            }
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("water batch source is synced")
                .gpu;
            if visible(gpu) {
                self.batch_water.push(prototype);
            }
        }
    }

    /// Rebuild material/pass membership only when static world state changes.
    fn rebuild_draw_sources(&mut self, world: &World) {
        self.source_opaque.clear();
        self.source_terrain.clear();
        self.source_transparent.clear();
        self.source_water.clear();
        self.batch_source_opaque.clear();
        self.batch_source_terrain.clear();
        self.batch_source_transparent.clear();
        self.batch_source_water.clear();

        let gpu_batches = world.instance_submit() == InstanceSubmit::GpuIndirect;
        for (id, entity) in world.entities() {
            if world.is_portal_surface(id) {
                continue;
            }
            if gpu_batches && entity.instanced() {
                continue;
            }
            let (has_opaque, has_xlucent) = {
                let Some(gpu) = self.gpu_meshes.get(&id) else {
                    continue;
                };
                (
                    gpu.opaque_index_count > 0,
                    gpu.opaque_index_count < gpu.index_count,
                )
            };
            match entity.material {
                None => {
                    if has_opaque {
                        self.source_opaque.push(id);
                    }
                    if has_xlucent {
                        self.source_transparent.push(id);
                    }
                }
                Some(SurfaceMaterialRef::Terrain(_)) => {
                    if has_opaque {
                        self.source_terrain.push(id);
                    }
                    if has_xlucent {
                        self.source_transparent.push(id);
                    }
                }
                Some(SurfaceMaterialRef::Water(_)) => {
                    if has_xlucent {
                        self.source_water.push(id);
                    }
                }
            }
        }
        if gpu_batches {
            let mut seen = HashSet::new();
            for (id, entity) in world.entities() {
                if !entity.instanced() {
                    continue;
                }
                let prototype = entity.instance_of.unwrap_or(id);
                if !seen.insert(prototype) {
                    continue;
                }
                let batch = self
                    .gpu_instance_batches
                    .get(&prototype)
                    .unwrap_or_else(|| panic!("instance batch {prototype} is not synced"));
                let gpu = &batch.gpu;
                let entity = world
                    .entity(prototype)
                    .unwrap_or_else(|_| panic!("instance batch prototype {prototype} disappeared"));
                let has_opaque = gpu.opaque_index_count > 0;
                let has_xlucent = gpu.opaque_index_count < gpu.index_count;
                match entity.material {
                    None => {
                        if has_opaque {
                            self.batch_source_opaque.push(prototype);
                        }
                        if has_xlucent {
                            self.batch_source_transparent.push(prototype);
                        }
                    }
                    Some(SurfaceMaterialRef::Terrain(_)) => {
                        if has_opaque {
                            self.batch_source_terrain.push(prototype);
                        }
                        if has_xlucent {
                            self.batch_source_transparent.push(prototype);
                        }
                    }
                    Some(SurfaceMaterialRef::Water(_)) => {
                        if has_xlucent {
                            self.batch_source_water.push(prototype);
                        }
                    }
                }
            }
        }
    }

    fn sync_textures_and_materials(&mut self, world: &World) {
        for (id, cpu) in world.textures() {
            if self.gpu_textures.contains_key(id) {
                continue;
            }
            let gpu = upload_texture(&self.device, &self.queue, cpu.width, cpu.height, &cpu.rgba);
            self.gpu_textures.insert(*id, gpu);
        }
        self.gpu_textures
            .retain(|id, _| world.textures().contains_key(id));

        for (_, entity) in world.entities() {
            let Some(tid) = entity.albedo else {
                continue;
            };
            if self.gpu_mesh_albedo.contains_key(&tid) {
                continue;
            }
            let p = &self.pipelines;
            let bind = if let Some(maps) = world.surface_maps(tid) {
                // Imported material sets: mipmapped, repeating, full PBR.
                let cpu = &world.textures()[&tid];
                let albedo = material::upload_map(
                    &self.device,
                    &self.queue,
                    &crate::mesh::AlbedoMap {
                        width: cpu.width,
                        height: cpu.height,
                        rgba: cpu.rgba.clone().into(),
                    },
                    material::MapKind::Color,
                )
                .create_view(&Default::default());
                material::GpuMaterial::new(
                    &self.device,
                    &self.queue,
                    &p.albedo_layout,
                    &p.material_sampler,
                    &p.neutral_maps,
                    Some(&albedo),
                    Some(maps),
                )
                .bind
            } else {
                let gpu = self
                    .gpu_textures
                    .get(&tid)
                    .expect("mesh albedo missing on GPU");
                material::GpuMaterial::new(
                    &self.device,
                    &self.queue,
                    &p.albedo_layout,
                    &p.albedo_sampler,
                    &p.neutral_maps,
                    Some(&gpu.view),
                    None,
                )
                .bind
            };
            self.gpu_mesh_albedo.insert(tid, bind);
        }
        self.gpu_mesh_albedo
            .retain(|id, _| world.textures().contains_key(id));

        for (id, mat) in world.materials() {
            if self.gpu_materials.contains_key(id) {
                continue;
            }
            let grass = self
                .gpu_textures
                .get(&mat.desc.grass)
                .expect("terrain grass texture missing on GPU");
            let grass_dry = self
                .gpu_textures
                .get(&mat.desc.grass_dry)
                .expect("terrain dry-grass texture missing on GPU");
            let grass_moor = self
                .gpu_textures
                .get(&mat.desc.grass_moor)
                .expect("terrain moor texture missing on GPU");
            let mud = self
                .gpu_textures
                .get(&mat.desc.mud)
                .expect("terrain mud texture missing on GPU");
            let tundra = self
                .gpu_textures
                .get(&mat.desc.tundra)
                .expect("terrain tundra texture missing on GPU");
            let scree = self
                .gpu_textures
                .get(&mat.desc.scree)
                .expect("terrain scree texture missing on GPU");
            let sand = self
                .gpu_textures
                .get(&mat.desc.sand)
                .expect("terrain sand texture missing on GPU");
            let rock = self
                .gpu_textures
                .get(&mat.desc.rock)
                .expect("terrain rock texture missing on GPU");
            let gpu = build_terrain_material(
                &self.device,
                &self.terrain.mat_bind_layout,
                &self.terrain.sampler,
                [grass, grass_dry, grass_moor, mud, tundra, scree, sand, rock],
                &mat.desc,
                world.render_origin(),
            );
            self.gpu_materials.insert(*id, gpu);
        }
        self.gpu_materials
            .retain(|id, _| world.materials().contains_key(id));

        for (id, mat) in world.water_materials() {
            if self.gpu_water_materials.contains_key(id) {
                continue;
            }
            let gpu = build_water_material(
                &self.device,
                &self.water.mat_bind_layout,
                &mat.desc,
                world.render_origin(),
            );
            self.gpu_water_materials.insert(*id, gpu);
        }
        self.gpu_water_materials
            .retain(|id, _| world.water_materials().contains_key(id));

        // Rebase moved render space under the terrain; re-phase the tiling so
        // the ground texture and the waves stay locked to world coordinates.
        if world.render_origin() != self.terrain_origin {
            self.terrain_origin = world.render_origin();
            for mat in self.gpu_materials.values() {
                mat.write_origin(&self.queue, self.terrain_origin);
            }
            for mat in self.gpu_water_materials.values() {
                mat.write_origin(&self.queue, self.terrain_origin);
            }
        }
    }
}

fn batch_slot_capacity(instance_count: usize) -> usize {
    instance_count
        .max(1)
        .checked_next_power_of_two()
        .expect("instance batch slot capacity overflow")
        .max(8)
}

/// One shadow view to record on a frame thread: its own encoder (holding
/// its instance cull already) and what it draws besides its bundles.
struct ShadowViewWork<'r> {
    slot: usize,
    cache: ShadowCacheUse,
    encoder: wgpu::CommandEncoder,
    skinned: Vec<&'r GpuSkinnedEntity>,
    voxel: Option<Vec<(std::ops::Range<u32>, i32, u32)>>,
}

/// What recording a shadow bundle reads: shared and unchanged while the
/// bundles are recorded, so several threads can read it at once.
struct ShadowBundleCtx<'r> {
    device: &'r wgpu::Device,
    shadow: &'r ShadowGpu,
    gpu_meshes: &'r HashMap<EntityId, GpuMesh>,
    gpu_mesh_albedo: &'r HashMap<TextureId, wgpu::BindGroup>,
    gpu_instance_batches: &'r HashMap<EntityId, GpuInstanceBatch>,
}

impl ShadowBundleCtx<'_> {
    /// One view's casters: `draws` are plain meshes, or instance batches
    /// when `batches`.
    fn record(&self, cascade: usize, draws: &[ShadowDraw], batches: bool) -> wgpu::RenderBundle {
        let mut bundle =
            self.device
                .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some("visible-shadow-casters"),
                    color_formats: &[],
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                });
        let view = shadow_view_of(cascade);
        let view_bind = if view < 3 {
            &self.shadow.cascade_binds[view]
        } else {
            &self.shadow.lamps.layer_binds[view - 3]
        };
        bundle.set_bind_group(0, view_bind, &[]);
        // Untextured casters first on the cheap depth-only pipeline, then the
        // textured ones through the alpha test.
        for cutout in [false, true] {
            let pipeline = if cutout {
                match self.shadow.cutout_pipeline.as_ref() {
                    Some(p) => p,
                    None => &self.shadow.mesh_pipeline,
                }
            } else {
                &self.shadow.mesh_pipeline
            };
            bundle.set_pipeline(pipeline);
            let albedo_of =
                |draw: &ShadowDraw| draw.albedo.and_then(|tid| self.gpu_mesh_albedo.get(&tid));
            for draw in draws {
                let albedo = albedo_of(draw);
                if albedo.is_some() != cutout {
                    continue;
                }
                if let Some(bind) = albedo {
                    bundle.set_bind_group(1, bind, &[]);
                }
                let gpu = if batches {
                    &self
                        .gpu_instance_batches
                        .get(&draw.id)
                        .expect("shadow bundle batch is synced")
                        .gpu
                } else {
                    self.gpu_meshes
                        .get(&draw.id)
                        .expect("shadow bundle caster is synced")
                };
                encode_shadow_bundle_draw(&mut bundle, gpu, draw.submit);
            }
        }
        bundle.finish(&wgpu::RenderBundleDescriptor {
            label: Some("visible-shadow-casters"),
        })
    }
}

fn encode_shadow_bundle_draw<'a>(
    bundle: &mut wgpu::RenderBundleEncoder<'a>,
    gpu: &'a GpuMesh,
    submit: MeshSubmit,
) {
    let opaque_index_count =
        u32::try_from(gpu.opaque_index_count).expect("shadow opaque index count exceeds u32");
    bundle.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
    bundle.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
    match submit {
        MeshSubmit::Compacted => {
            bundle.set_vertex_buffer(1, gpu.compact_buf.slice(..));
            bundle.draw_indexed_indirect(&gpu.indirect_buf, OPAQUE_INDIRECT_OFFSET);
        }
        MeshSubmit::Direct => {
            let instance_count =
                u32::try_from(gpu.instance_count).expect("shadow instance count exceeds u32");
            bundle.set_vertex_buffer(1, gpu.instance_slice());
            bundle.draw_indexed(
                gpu.indices(0..opaque_index_count),
                gpu.base_vertex,
                0..instance_count,
            );
        }
    }
}

fn submit_batch_draw(
    pass: &mut wgpu::RenderPass<'_>,
    batch: &GpuInstanceBatch,
    submit: Option<MeshSubmit>,
    indices: std::ops::Range<u32>,
    indirect_offset: u64,
    indirect_draws: &mut u32,
) {
    let gpu = &batch.gpu;
    if let Some(submit) = submit {
        submit_mesh_draw(pass, gpu, submit, indices, indirect_offset, indirect_draws);
        return;
    }
    pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
    pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
    for range in &batch.active_ranges {
        pass.set_vertex_buffer(1, gpu.instance_slice_from(range.start as usize));
        pass.draw_indexed(
            gpu.indices(indices.clone()),
            gpu.base_vertex,
            0..range.count,
        );
    }
}
fn submit_mesh_draw(
    pass: &mut wgpu::RenderPass<'_>,
    gpu: &GpuMesh,
    submit: MeshSubmit,
    indices: std::ops::Range<u32>,
    indirect_offset: u64,
    indirect_draws: &mut u32,
) {
    pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
    pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
    match submit {
        MeshSubmit::Compacted => {
            pass.set_vertex_buffer(1, gpu.compact_buf.slice(..));
            pass.draw_indexed_indirect(&gpu.indirect_buf, indirect_offset);
            *indirect_draws += 1;
        }
        MeshSubmit::Direct => {
            pass.set_vertex_buffer(1, gpu.instance_slice());
            pass.draw_indexed(
                gpu.indices(indices),
                gpu.base_vertex,
                0..gpu.instance_count as u32,
            );
        }
    }
}

/// One mesh drawn through its occlusion draw (see `occlusion.rs`).
fn submit_occluded_draw(
    pass: &mut wgpu::RenderPass<'_>,
    gpu: &GpuMesh,
    args: &wgpu::Buffer,
    slot: u32,
    range_offset: u64,
) {
    pass.set_vertex_buffer(0, gpu.vertex_buf.slice(..));
    pass.set_vertex_buffer(1, gpu.instance_slice());
    pass.set_index_buffer(gpu.index_buf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed_indirect(args, occlusion::OcclusionFrame::offset(slot, range_offset));
}

fn entity_in_space(world: &World, id: EntityId, space: SpaceId) -> bool {
    world.entity(id).expect("draw source is live").space() == space && !world.room_hidden(id)
}

fn mesh_submit(compacted: &HashSet<EntityId>, id: EntityId) -> MeshSubmit {
    if compacted.contains(&id) {
        MeshSubmit::Compacted
    } else {
        MeshSubmit::Direct
    }
}

/// `ENGINE_TAA=0` switches temporal anti-aliasing off for comparisons.
/// `ENGINE_TONEMAP=agx|aces` overrides the world's choice (for comparisons).
/// Finish a startup worker, passing its panic on unchanged.
pub(super) fn join_worker<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

fn tone_map_env() -> Option<crate::world::ToneMap> {
    static MODE: std::sync::OnceLock<Option<crate::world::ToneMap>> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("ENGINE_TONEMAP")
            .ok()
            .and_then(|v| crate::world::ToneMap::from_name(&v))
    })
}

/// `ENGINE_RENDER_SCALE=0.75` draws the scene at that share of the window,
/// whatever the game set.
fn render_scale_env() -> Option<f32> {
    static SCALE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("ENGINE_RENDER_SCALE")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite())
    })
}

/// `ENGINE_SHADOW_CACHE=0` draws every shadow caster every frame.
fn shadow_cache_env() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_SHADOW_CACHE").map_or(true, |v| v.trim() != "0"))
}

/// Which static casters a cache layer holds, and how they are drawn.
fn caster_hash(selection: &ShadowCullSelection, space: SpaceId) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    space.hash(&mut h);
    for d in selection.draws.iter().chain(&selection.batch_draws) {
        d.id.hash(&mut h);
        (d.submit == MeshSubmit::Compacted).hash(&mut h);
        d.albedo.hash(&mut h);
    }
    selection.draws.len().hash(&mut h);
    h.finish()
}

fn taa_env_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_TAA").map_or(true, |v| v.trim() != "0"))
}

/// DirectX 12 compiles shaders with FXC unless told otherwise, and FXC
/// cannot build every Engine shader. `ENGINE_DXC_DIR=<folder>` (holding
/// `dxcompiler.dll` and `dxil.dll`, e.g. the Windows SDK's bin\x64) or
/// `WGPU_DX12_COMPILER=dxc` selects the newer DXC.
fn dx12_options() -> wgpu::BackendOptions {
    let mut options = wgpu::BackendOptions::from_env_or_default();
    if let Ok(dir) = std::env::var("ENGINE_DXC_DIR") {
        let dir = std::path::Path::new(&dir);
        options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
            dxc_path: dir.join("dxcompiler.dll").to_string_lossy().into_owned(),
            dxil_path: dir.join("dxil.dll").to_string_lossy().into_owned(),
            max_shader_model: wgpu::DxcShaderModel::V6_5,
        };
    }
    options
}

fn create_depth(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: stencil::scene_samples(),
        dimension: wgpu::TextureDimension::D2,
        format: depth_format(),
        // Sampled by the SSAO pass after the depth prepass.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod active_instance_range_tests {
    use super::*;

    fn slot(start: usize, capacity: usize, active_count: usize) -> BatchSlot {
        BatchSlot {
            id: EntityId::test(1),
            xform_rev: 1,
            start,
            capacity,
            active_count,
            bounds: None,
        }
    }

    #[test]
    fn reserved_holes_are_not_part_of_gpu_cull_ranges() {
        let slots = [slot(0, 1024, 3), slot(1024, 1024, 0), slot(2048, 1024, 7)];
        assert_eq!(
            active_instance_ranges(&slots),
            vec![
                ActiveInstanceRange { start: 0, count: 3 },
                ActiveInstanceRange {
                    start: 2048,
                    count: 7
                },
            ]
        );
    }
}

/// Map the vsync preference to a present mode. FIFO synchronizes presentation;
/// Immediate is the explicit uncapped, tearing-allowed mode.
///
/// `supported` must come from `Surface::get_capabilities`. Vsync off on a
/// surface without Immediate support is a loud error rather than a refresh-
/// paced Mailbox/FIFO substitution that lies about the setting.
/// Sky-dome tints for ambient light, scaled so the sky's average is 1 and the
/// world's `ambient` alone sets how bright the sky light is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyLight {
    pub up: [f32; 3],
    pub horizon: [f32; 3],
    pub down: [f32; 3],
}

impl SkyLight {
    pub(crate) fn from_world(world: &World) -> Self {
        let luma = |c: glam::Vec3| c.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
        let (zenith, horizon, ground) = match world.sky() {
            Some(sky) => (
                sky.zenith.to_vec3(),
                sky.horizon.to_vec3(),
                sky.ground.to_vec3(),
            ),
            None => {
                // Without a sky, lean the dome toward the clear colour when it
                // reads as sky; a black or dark backdrop leaves neutral light.
                let clear = world.clear_color().to_vec3();
                let tint = if luma(clear) > 0.05 {
                    (clear / luma(clear)).lerp(glam::Vec3::ONE, 0.6)
                } else {
                    glam::Vec3::ONE
                };
                (tint, glam::Vec3::ONE, glam::Vec3::new(0.62, 0.58, 0.52))
            }
        };
        let sky_luma = luma((zenith + horizon) * 0.5).max(1e-3);
        let up = zenith / sky_luma;
        let horizon = horizon / sky_luma;
        // Bounce light: tinted by the ground but never darker than a third
        // of the sky, so undersides do not drop to black.
        let mut down = ground / sky_luma;
        let down_luma = luma(down);
        if down_luma < 0.35 {
            down = if down_luma > 1e-4 {
                down * (0.35 / down_luma)
            } else {
                glam::Vec3::splat(0.35)
            };
        }
        let down = down.min(glam::Vec3::splat(1.0));
        Self {
            up: up.into(),
            horizon: horizon.into(),
            down: down.into(),
        }
    }
}

/// World-pass MSAA: `ENGINE_MSAA=1` or `4`. WebGPU guarantees 4x for both
/// scene formats; other counts would need adapter-specific features.
///
/// Default 1 while TAA is on: TAA already smooths edges, and leaf cut-outs
/// dither their coverage for it, which looks the same as 4x MSAA in stills at
/// 10-15% more frame rate at 4K (RTX 5070). With `ENGINE_TAA=0` the default
/// is 4x so edges stay smooth.
fn choose_scene_samples(adapter: &wgpu::Adapter) -> u32 {
    let requested = match std::env::var("ENGINE_MSAA") {
        Ok(raw) => match raw.parse::<u32>() {
            Ok(n @ (1 | 4)) => n,
            _ => panic!("ENGINE_MSAA must be 1 or 4, got {raw:?}"),
        },
        Err(std::env::VarError::NotPresent) => {
            if taa_env_enabled() {
                1
            } else {
                4
            }
        }
        Err(std::env::VarError::NotUnicode(_)) => panic!("ENGINE_MSAA must be valid Unicode"),
    };
    let color = adapter.get_texture_format_features(HDR_FORMAT).flags;
    let depth = adapter.get_texture_format_features(depth_format()).flags;
    if requested == 4 && color.sample_count_supported(4) && depth.sample_count_supported(4) {
        4
    } else {
        1
    }
}

fn resolve_present_mode(vsync: bool, supported: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    let requested = if vsync {
        wgpu::PresentMode::Fifo
    } else {
        wgpu::PresentMode::Immediate
    };
    if !supported.contains(&requested) {
        panic!(
            "vsync {} requires {requested:?}, but the surface supports only {supported:?}",
            if vsync { "on" } else { "off" }
        );
    }
    requested
}

#[cfg(test)]
mod vsync_tests {
    use super::*;

    #[test]
    fn vsync_on_is_fifo_and_off_is_immediate() {
        let supported = [wgpu::PresentMode::Fifo, wgpu::PresentMode::Immediate];
        assert_eq!(
            resolve_present_mode(true, &supported),
            wgpu::PresentMode::Fifo
        );
        assert_eq!(
            resolve_present_mode(false, &supported),
            wgpu::PresentMode::Immediate
        );
    }

    #[test]
    #[should_panic(expected = "vsync off requires Immediate")]
    fn vsync_off_without_immediate_support_is_loud() {
        let refresh_paced_only = [wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox];
        resolve_present_mode(false, &refresh_paced_only);
    }
}
