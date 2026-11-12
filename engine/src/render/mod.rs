mod audit;
mod clipmap;
mod crowd;
mod decals;
pub(crate) mod frustum;
mod geometry;
mod gi;
mod gpu_mesh;
mod gpu_profile;
mod gpu_scene;
mod grass;
mod instance_cull;
mod lights;
mod material;
mod material_table;
mod mipgen;
mod occlusion;
mod particle_sort;
mod particles_gpu;
mod pipeline;
pub(crate) use pipeline::{lit_shader_source, CUSTOM_CODE_MARK};
pub(crate) mod pipeline_cache;
mod portal;
mod postprocess;
mod reflect;
mod ribbons_gpu;
mod shadow;
mod shadow_scene;
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
// `impl Renderer`, by topic.
mod renderer_frame;
mod renderer_gpu_sync;
mod renderer_opaque;
mod renderer_portals;
mod renderer_setup;
mod renderer_shadows;
mod renderer_space;
mod renderer_sync;

use crate::camera::Camera;
use crate::error::{EngineError, EngineResult};
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
    ActiveInstanceRange, CullDraws, CullJob, DrawIndexedArgs, InstanceCull, OPAQUE_INDIRECT_OFFSET,
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
    /// Instance-cull compute dispatches (one per active range per view).
    pub cull_dispatches: u32,
    /// Views whose instances were culled (main, cascades, lamp faces).
    pub cull_views: u32,
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
    /// Multi-draw calls the world pass sent its opaque meshes in (0 when
    /// it drew them one by one).
    pub multi_draws: u32,
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
            "shadow_atlas={} cascades={} cache_hits={} horizon={} submit={} inside={} boundary={} tested={} groups={} indirect={} occlusion_tested={} occlusion_hidden={} multi_draws={}",
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
            self.multi_draws,
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
    /// Its material-table slot, when its instances carry the current one:
    /// multi-draws then read the texture arrays instead of `albedo`.
    material: Option<u32>,
    gpu: &'r GpuMesh,
    submit: OpaqueSubmit<'r>,
    /// Its row in the main view's packed cull args, for a compacted batch.
    packed: Option<u32>,
}

impl<'r> OpaqueDraw<'r> {
    /// The buffers the draw reads: instances, and indirect args with their
    /// byte offset when the GPU decides the count.
    fn sources(&self) -> ((&'r wgpu::Buffer, u64), Option<(&'r wgpu::Buffer, u64)>) {
        let g = self.gpu;
        match self.submit {
            OpaqueSubmit::Mesh(MeshSubmit::Compacted) => {
                let target = cull_target(g);
                (
                    target.compact_at(),
                    Some(target.args(OPAQUE_INDIRECT_OFFSET)),
                )
            }
            OpaqueSubmit::Mesh(MeshSubmit::Direct) => ((&g.instance_buf, 0), None),
            OpaqueSubmit::Occluded(args, slot) => (
                (&g.instance_buf, 0),
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
        let (_, indirect) = self.sources();
        bundle.set_bind_group(1, self.albedo, &[]);
        bundle.set_vertex_buffer(0, g.vertex_buf.slice(..));
        match self.submit {
            OpaqueSubmit::Mesh(MeshSubmit::Compacted) => {
                bundle.set_vertex_buffer(1, cull_target(g).compact_slice())
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
    /// The instance buffer and where the draw's instances start in it.
    instances: (wgpu::Buffer, u64),
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
            && self.instances.0 == *instances.0
            && self.instances.1 == instances.1
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
            instances: (instances.0.clone(), instances.1),
            indirect: indirect.map(|(b, o)| (b.clone(), o)),
            counts: if indirect.is_some() {
                (0, 0)
            } else {
                (g.opaque_index_count as u32, g.instance_count as u32)
            },
        }
    }
}

/// Below this many opaque draws a pass draws them one by one, not with
/// multi-draws.
const OPAQUE_MULTI_MIN_DRAWS: usize = 64;

/// Draws that share a texture and buffers, sent as one
/// `multi_draw_indexed_indirect`.
struct MultiGroup {
    albedo: wgpu::BindGroup,
    /// Through the texture-array pipeline (`albedo` is the material table).
    bindless: bool,
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    instances: wgpu::Buffer,
    /// Where its arguments are.
    source: GroupArgs,
    offset: u64,
    /// At most this many draws; the GPU says how many (GPU scene).
    count: u32,
    /// Where in `MultiDraw::draw_counts` the GPU wrote the draw count.
    draw_count: Option<u64>,
}

/// Where a multi-draw group's arguments are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupArgs {
    /// The pass's own, written by the CPU (`MultiDraw::args`).
    Own,
    /// The occlusion pass's packed rows.
    Occlusion,
    /// The GPU instance cull's packed rows for the main view.
    Culled,
}

/// A pass's opaque meshes as a few multi-draws, as professional engines
/// draw them: meshes and their instances live in shared buffers (see
/// `geometry.rs`), so every mesh with the same texture in the same pages
/// goes out in one call whose per-mesh arguments sit in a buffer. Meshes
/// the occlusion test decides take theirs from its packed rows; GPU-culled
/// instanced props keep their own draws (in `rest`).
#[derive(Default)]
struct MultiDraw {
    pipeline: Option<wgpu::RenderPipeline>,
    /// For the groups that read the material table.
    bindless_pipeline: Option<wgpu::RenderPipeline>,
    groups: Vec<MultiGroup>,
    /// The direct draws' arguments, as last written.
    args: Option<wgpu::Buffer>,
    written: Vec<DrawIndexedArgs>,
    occlusion_args: Option<wgpu::Buffer>,
    /// The instance cull's packed rows.
    culled_args: Option<wgpu::Buffer>,
    /// The GPU scene's draw counts.
    draw_counts: Option<wgpu::Buffer>,
    rest: Option<wgpu::RenderBundle>,
}

impl MultiDraw {
    /// Draw it: the pipeline, then each group, then the rest; leaves the
    /// scene bind group bound.
    fn encode(&self, pass: &mut wgpu::RenderPass<'_>, scene_bind: &wgpu::BindGroup) -> u32 {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, scene_bind, &[]);
        let mut bound: Option<&MultiGroup> = None;
        for g in &self.groups {
            if g.bindless && bound.is_none_or(|b| !b.bindless) {
                let Some(bindless) = self.bindless_pipeline.as_ref() else {
                    continue;
                };
                pass.set_pipeline(bindless);
            } else if !g.bindless && bound.is_some_and(|b| b.bindless) {
                pass.set_pipeline(pipeline);
            }
            let args = match g.source {
                GroupArgs::Own => self.args.as_ref(),
                GroupArgs::Occlusion => self.occlusion_args.as_ref(),
                GroupArgs::Culled => self.culled_args.as_ref(),
            };
            let Some(args) = args else {
                continue;
            };
            if bound.is_none_or(|b| b.albedo != g.albedo) {
                pass.set_bind_group(1, &g.albedo, &[]);
            }
            if bound.is_none_or(|b| b.vertex != g.vertex) {
                pass.set_vertex_buffer(0, g.vertex.slice(..));
            }
            if bound.is_none_or(|b| b.instances != g.instances) {
                pass.set_vertex_buffer(1, g.instances.slice(..));
            }
            if bound.is_none_or(|b| b.index != g.index) {
                pass.set_index_buffer(g.index.slice(..), wgpu::IndexFormat::Uint32);
            }
            match (g.draw_count, self.draw_counts.as_ref()) {
                (Some(at), Some(counts)) => {
                    pass.multi_draw_indexed_indirect_count(args, g.offset, counts, at, g.count);
                }
                _ => pass.multi_draw_indexed_indirect(args, g.offset, g.count),
            }
            bound = Some(g);
        }
        if let Some(rest) = self.rest.as_ref() {
            pass.execute_bundles(std::iter::once(rest));
            pass.set_bind_group(0, scene_bind, &[]);
        } else if bound.is_some_and(|b| b.bindless) {
            pass.set_pipeline(pipeline);
        }
        self.groups.len() as u32
    }
}

/// What `plan_multi_draw` worked out for a pass.
struct MultiPlan {
    groups: Vec<MultiGroup>,
    args: Vec<DrawIndexedArgs>,
    occlusion_args: Option<wgpu::Buffer>,
    culled_args: Option<wgpu::Buffer>,
    draw_counts: Option<wgpu::Buffer>,
    rest: Option<wgpu::RenderBundle>,
    indirect_draws: u32,
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
    /// The same draws as multi-draws, where the GPU can.
    multi: MultiDraw,
    /// This frame's use: what the chunks were checked for, and how many.
    frame: Option<OpaqueFrame>,
}

struct OpaqueFrame {
    target: OpaqueTarget,
    space: SpaceId,
    uniform_slot: usize,
    chunks: usize,
    indirect_draws: u32,
    /// Drawn with `multi` instead of the chunks.
    multi: bool,
}

struct MainCullSelection {
    ids: Vec<EntityId>,
    compacted: HashSet<EntityId>,
    /// This frame's occlusion draws, when occlusion culling is on.
    occlusion: Option<occlusion::OcclusionFrame>,
    /// Where the cull packs the opaque batches' args for multi-draws.
    pack: Option<MainPack>,
}

/// The main view's packed batch args: a row per opaque batch, in the
/// order multi-draws group them (texture, then buffers), so a run of rows
/// is one call.
struct MainPack {
    args: wgpu::Buffer,
    rows: HashMap<EntityId, u32>,
}

impl MainCullSelection {
    fn none() -> Self {
        Self {
            ids: Vec::new(),
            compacted: HashSet::new(),
            occlusion: None,
            pack: None,
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
#[derive(Clone, Copy)]
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
/// Most `settling` entries kept before the shadow scene is built again instead.
const SETTLING_LIMIT: usize = 1 << 17;
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
    /// The mesh casters as multi-draws instead of `groups`, where the GPU
    /// can.
    multi: Option<ShadowMulti>,
}

/// Fewer mesh casters than this in a shadow view: bundles, not multi-draws.
const SHADOW_MULTI_MIN_DRAWS: usize = 32;

/// Casters that share a pipeline, texture and buffers: one multi-draw.
struct ShadowMultiGroup {
    /// Through the alpha-tested pipeline (textured casters).
    cutout: bool,
    albedo: Option<wgpu::BindGroup>,
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    instances: wgpu::Buffer,
    offset: u64,
    count: u32,
}

/// A shadow view's mesh casters as a few multi-draws (see `MultiDraw`).
/// Kept while its casters are the same; GPU-culled ones stay in `rest`.
struct ShadowMulti {
    draws: Vec<ShadowDraw>,
    groups: Vec<ShadowMultiGroup>,
    args: wgpu::Buffer,
    rest: Option<wgpu::RenderBundle>,
}

impl ShadowMulti {
    /// Draw the casters; leaves `view_bind` bound at group 0.
    fn encode(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        shadow: &ShadowGpu,
        view_bind: &wgpu::BindGroup,
    ) {
        pass.set_bind_group(0, view_bind, &[]);
        let mut bound: Option<&ShadowMultiGroup> = None;
        for g in &self.groups {
            if bound.is_none_or(|b| b.cutout != g.cutout) {
                pass.set_pipeline(match (g.cutout, shadow.cutout_pipeline.as_ref()) {
                    (true, Some(p)) => p,
                    _ => &shadow.mesh_pipeline,
                });
            }
            if let Some(albedo) = g.albedo.as_ref() {
                if bound.is_none_or(|b| b.albedo.as_ref() != Some(albedo)) {
                    pass.set_bind_group(1, albedo, &[]);
                }
            }
            if bound.is_none_or(|b| b.vertex != g.vertex) {
                pass.set_vertex_buffer(0, g.vertex.slice(..));
            }
            if bound.is_none_or(|b| b.instances != g.instances) {
                pass.set_vertex_buffer(1, g.instances.slice(..));
            }
            if bound.is_none_or(|b| b.index != g.index) {
                pass.set_index_buffer(g.index.slice(..), wgpu::IndexFormat::Uint32);
            }
            pass.multi_draw_indexed_indirect(&self.args, g.offset, g.count);
            bound = Some(g);
        }
        if let Some(rest) = self.rest.as_ref() {
            pass.execute_bundles(std::iter::once(rest));
            pass.set_bind_group(0, view_bind, &[]);
        }
    }
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
    /// `moved_at` entries in the order they were made, while the GPU
    /// shadow scene is on: when one settles, its caster may count as
    /// static again (only those rows are rechecked, not every caster).
    settling: std::collections::VecDeque<(u64, EntityId)>,
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
    /// The GPU draws many meshes per call (`MultiDraw`).
    multi_draw: bool,
    /// Opaque meshes kept on the GPU as rows the GPU culls and draws from
    /// (see `gpu_scene.rs`); with multi-draws unless `ENGINE_GPU_SCENE=0`.
    gpu_scene: Option<gpu_scene::GpuScene>,
    /// Shadow-casting meshes kept on the GPU, culled per shadow view there
    /// (see `shadow_scene.rs`); with GPU draw counts unless
    /// `ENGINE_GPU_SHADOWS=0`.
    shadow_scene: Option<shadow_scene::ShadowScene>,
    /// Per sun cascade: goes up when a static caster in its view changes
    /// (the GPU shadow scene's part of the static cache key).
    static_caster_gen: [u64; 3],
    /// Goes up whenever textures and materials are synced (their bind
    /// groups may be new).
    materials_synced: u64,
    /// Texture arrays: the material table and the opaque pipelines that
    /// read it, so multi-draws need not split by texture.
    bindless: Option<Bindless>,
    /// GPU-mode instance buffers, one per prototype instead of one per spatial bin.
    gpu_instance_batches: HashMap<EntityId, GpuInstanceBatch>,
    gpu_batch_sources: Vec<BatchSourceState>,
    gpu_batch_source_scratch: Vec<BatchSourceState>,
    gpu_skinned: HashMap<EntityId, GpuSkinnedEntity>,
    /// Vertex/index buffers keyed by `Arc::as_ptr` of the shared [`AnimatedModel`].
    /// Per model: its uploaded meshes and morph table, shared by every entity of it.
    gpu_skinned_meshes: HashMap<usize, (Vec<GpuSkinnedMesh>, wgpu::Buffer)>,
    /// Per model (same key): how far its skin reaches from its joints.
    skinned_reach: HashMap<usize, Arc<skinned::SkinReach>>,
    /// Characters' maps, one GPU copy per content (see [`material::MapCache`]).
    skinned_maps: material::MapCache,
    /// Keeps the GPU mip generator registered for this device.
    _mipgen: mipgen::Registration,
    gpu_textures: HashMap<TextureId, GpuTexture>,
    gpu_mesh_albedo: HashMap<TextureId, wgpu::BindGroup>,
    /// The mesh albedos with see-through texels (alpha below one half):
    /// only those cast shadows through the alpha test. Casters with an
    /// opaque texture take the plain depth pipeline, no texture bound.
    cutout_albedo: HashMap<TextureId, wgpu::BindGroup>,
    audit_frame: u64,
    /// The world's re-upload requests already carried out.
    reupload_seen: (u64, u64),
    gpu_check_seen: u64,
    rebuild_seen: (u64, u64),
    gpu_materials: HashMap<MaterialId, GpuTerrainMaterial>,
    gpu_water_materials: HashMap<WaterMaterialId, GpuWaterMaterial>,
    synced_render_epoch: u64,
    synced_instance_epoch: u64,
    /// Entities the world touched (moved, new instance list) that a sync
    /// brought up to date without a full resync, whose rows the GPU scene
    /// and the GPU shadow scene still have to patch (each drains its own
    /// list when it syncs; a rebuild of that scene covers them too).
    scene_touched: Vec<EntityId>,
    shadow_touched: Vec<EntityId>,
    /// Plain meshes a sync added or removed one by one (see
    /// `World::take_members`), whose rows the GPU scene and the GPU shadow
    /// scene still have to add or drop (like `scene_touched`).
    scene_added: Vec<EntityId>,
    scene_removed: Vec<EntityId>,
    shadow_added: Vec<EntityId>,
    shadow_removed: Vec<EntityId>,
    /// Full syncs so far: the draw and shadow source lists and the GPU
    /// scenes are built again when it moves.
    full_syncs: u64,
    /// Moves with every full sync and every mesh added or removed (GI
    /// voxelises the scene again).
    scene_gen: u64,
    /// `ENGINE_CPU_TRACE`: where in the world the full syncs came from.
    /// `ENGINE_CPU_TRACE`: instance-cull views, jobs, dispatches and
    /// instances summed over frames, printed every 120 frames.
    cull_trace: [u64; 5],
    full_sync_sites: HashMap<Option<&'static std::panic::Location<'static>>, u32>,
    synced_resource_epoch: u64,
    /// Origin the terrain material phases were last written for.
    terrain_origin: crate::space::RenderOrigin,
    /// This frame's view volume, for skipping draws outside it.
    frustum: Frustum,
    /// Where that view looks from, for drawing see-through things far to near.
    view_eye: glam::Vec3,
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
    /// Solid meshes and batches drawn with a game's custom material.
    source_custom: Vec<TransparentDraw>,
    draw_custom: Vec<TransparentDraw>,
    gpu_custom: HashMap<crate::custom_material::CustomMaterialId, pipeline::GpuCustomMaterial>,
    /// Custom materials whose pipelines compile on a worker.
    custom_compiling: HashMap<
        crate::custom_material::CustomMaterialId,
        std::thread::JoinHandle<pipeline::GpuCustomMaterial>,
    >,
    /// Fixed-step runs wait for worker results (see `set_wait_for_workers`).
    wait_for_workers: bool,
    custom_params_layout: Option<wgpu::BindGroupLayout>,
    /// `draw_transparent` and `batch_transparent` together, farthest first.
    transparent_order: Vec<TransparentDraw>,
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
    cutout_albedo: &'r HashMap<TextureId, wgpu::BindGroup>,
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
                |draw: &ShadowDraw| draw.albedo.and_then(|tid| self.cutout_albedo.get(&tid));
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
            let target = cull_target(gpu);
            let (args, offset) = target.args(OPAQUE_INDIRECT_OFFSET);
            bundle.set_vertex_buffer(1, target.compact_slice());
            bundle.draw_indexed_indirect(args, offset);
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

/// Where a batch drawn with [`MeshSubmit::Compacted`] finds its culled
/// instances and draw args (placed at the start of the frame's culls).
fn cull_target(gpu: &GpuMesh) -> &instance_cull::CullTarget {
    gpu.cull_target
        .as_ref()
        .expect("a compacted draw's batch was placed for the GPU cull")
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
            let target = cull_target(gpu);
            let (args, offset) = target.args(indirect_offset);
            pass.set_vertex_buffer(1, target.compact_slice());
            pass.draw_indexed_indirect(args, offset);
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

/// The GPU scene's groups as multi-draws over one of `frame`'s packed
/// regions (plain groups come first).
fn scene_groups<'a>(
    scene: &'a gpu_scene::GpuScene,
    frame: &'a occlusion::OcclusionFrame,
    region: occlusion::ArgsRegion,
) -> impl Iterator<Item = MultiGroup> + 'a {
    scene
        .groups
        .iter()
        .zip(0u32..)
        .map(move |(g, i)| MultiGroup {
            albedo: g.key.bind.clone(),
            bindless: g.key.bindless,
            vertex: g.key.vertex.clone(),
            index: g.key.index.clone(),
            instances: g.key.instances.clone(),
            source: GroupArgs::Occlusion,
            offset: frame.region_offset(region, g.start),
            count: g.len,
            draw_count: frame
                .draw_counts
                .is_some()
                .then(|| frame.count_offset(region, i)),
        })
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
/// `ENGINE_MULTI_DRAW=0` draws every opaque mesh on its own.
fn multi_draw_env() -> Option<bool> {
    std::env::var("ENGINE_MULTI_DRAW")
        .ok()
        .map(|v| v.trim() != "0")
}

fn shadow_cache_env() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_SHADOW_CACHE").map_or(true, |v| v.trim() != "0"))
}

/// Which static casters a cache layer holds, and how they are drawn.
/// A view's casters from the GPU shadow scene: its groups as draw-count
/// multi-draws. Leaves `view_bind` bound at group 0.
fn draw_shadow_scene(
    pass: &mut wgpu::RenderPass<'_>,
    scene: &shadow_scene::ShadowScene,
    view: usize,
    shadow: &ShadowGpu,
    view_bind: &wgpu::BindGroup,
) {
    pass.set_bind_group(0, view_bind, &[]);
    let mut bound: Option<&shadow_scene::ShadowKey> = None;
    for (i, g) in scene.groups.iter().enumerate() {
        if !scene.draws(view, g) {
            continue;
        }
        let k = &g.key;
        if bound.is_none_or(|b| b.cutout != k.cutout) {
            pass.set_pipeline(match (k.cutout, shadow.cutout_pipeline.as_ref()) {
                (true, Some(p)) => p,
                _ => &shadow.mesh_pipeline,
            });
        }
        if let Some(albedo) = k.albedo.as_ref() {
            if bound.is_none_or(|b| b.albedo.as_ref() != Some(albedo)) {
                pass.set_bind_group(1, albedo, &[]);
            }
        }
        if bound.is_none_or(|b| b.vertex != k.vertex) {
            pass.set_vertex_buffer(0, k.vertex.slice(..));
        }
        if bound.is_none_or(|b| b.instances != k.instances) {
            pass.set_vertex_buffer(1, k.instances.slice(..));
        }
        if bound.is_none_or(|b| b.index != k.index) {
            pass.set_index_buffer(k.index.slice(..), wgpu::IndexFormat::Uint32);
        }
        let (args, count) = scene.offsets(view, g, i);
        pass.multi_draw_indexed_indirect_count(&scene.args, args, &scene.counts, count, g.len);
        bound = Some(k);
    }
}

/// Whether an RGBA8 texture has texels the shadow alpha test can cut
/// (alpha below one half). Without any, filtering and mips stay at or above
/// one half too, so the test never discards and plain depth is the same.
fn has_cutout_texels(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).any(|px| px[3] < 128)
}

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

/// World-pass MSAA: the game's setting, or `ENGINE_MSAA=1` or `4` over it.
/// WebGPU guarantees 4x for both scene formats; other counts would need
/// adapter-specific features.
///
/// Default 1 while TAA is on: TAA already smooths edges, and leaf cut-outs
/// dither their coverage for it, which looks the same as 4x MSAA in stills at
/// 10-15% more frame rate at 4K (RTX 5070). With TAA off (`ENGINE_TAA=0` or
/// `Settings::taa(false)`) the default is 4x so edges stay smooth.
fn choose_scene_samples(adapter: &wgpu::Adapter, setting: Option<u32>) -> u32 {
    let requested = match std::env::var("ENGINE_MSAA") {
        Ok(raw) => match raw.parse::<u32>() {
            Ok(n @ (1 | 4)) => n,
            _ => {
                eprintln!("[engine] ENGINE_MSAA must be 1 or 4, got {raw:?}; ignoring it");
                setting.unwrap_or(if taa_env_enabled() { 1 } else { 4 })
            }
        },
        Err(_) => setting.unwrap_or(if taa_env_enabled() { 1 } else { 4 }),
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
    fn only_see_through_texels_need_the_shadow_alpha_test() {
        assert!(!has_cutout_texels(&[10, 20, 30, 255, 0, 0, 0, 128]));
        assert!(has_cutout_texels(&[10, 20, 30, 255, 0, 0, 0, 127]));
    }

    #[test]
    #[should_panic(expected = "vsync off requires Immediate")]
    fn vsync_off_without_immediate_support_is_loud() {
        let refresh_paced_only = [wgpu::PresentMode::Fifo, wgpu::PresentMode::Mailbox];
        resolve_present_mode(false, &refresh_paced_only);
    }
}

/// Texture arrays (see `material_table.rs`).
struct Bindless {
    table: material_table::MaterialTable,
    pipes: pipeline::BindlessPipelines,
}

/// One see-through draw: a single mesh or an instanced batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransparentDraw {
    Mesh(EntityId),
    Batch(EntityId),
}

/// Sort see-through draws far to near from `eye`, so nearer glass blends
/// over farther glass. Ties keep a fixed order (by entity) so frames repeat.
pub(crate) fn back_to_front(eye: glam::Vec3, draws: &mut [(glam::Vec3, TransparentDraw)]) {
    let key = |d: &TransparentDraw| match *d {
        TransparentDraw::Mesh(id) => (id.raw(), 0),
        TransparentDraw::Batch(id) => (id.raw(), 1),
    };
    draws.sort_by(|a, b| {
        b.0.distance_squared(eye)
            .total_cmp(&a.0.distance_squared(eye))
            .then_with(|| key(&a.1).cmp(&key(&b.1)))
    });
}

/// Parse and validate a whole WGSL source, as the GPU's shader check would.
#[cfg(test)]
pub(crate) fn assert_valid_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("WGSL does not parse: {}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("WGSL does not validate: {e:?}"));
}
