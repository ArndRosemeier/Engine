mod clipmap;
mod frustum;
mod gpu_mesh;
mod instance_cull;
mod pipeline;
mod shadow;
mod skinned;
mod sky_pipeline;
mod terrain_pipeline;
mod water_pipeline;

use crate::mesh::InstanceRaw;
use crate::texture::{MaterialId, TextureId, WaterMaterialId};
use crate::world::{Entity, EntityId, InstanceSubmit, ShadowSettings, SurfaceMaterialRef, World};
use clipmap::ClipmapRenderer;
use frustum::{Frustum, Visibility};
use gpu_mesh::GpuMesh;
use instance_cull::{
    CullDraws, CullJob, InstanceCull, OPAQUE_INDIRECT_OFFSET, TRANSLUCENT_INDIRECT_OFFSET,
    WORKGROUP,
};
use pipeline::{create_pipelines, Pipelines, Uniforms};
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
    pub skinned_new: u32,
    pub skinned_model_uploads: u32,
    pub skinned_pose_writes: u32,
    pub shadow_atlas: bool,
    pub entities: u32,
    pub animated: u32,
    pub instance_submit: InstanceSubmit,
    pub compute_culls: u32,
    pub indirect_draws: u32,
    pub direct_inside_batches: u32,
    pub boundary_batches: u32,
    pub cull_instances: u64,
    pub cull_workgroups: u64,
}

impl GpuFrameStats {
    pub fn sync_line(&self) -> String {
        format!(
            "mesh_new={} rebuild={} inst={} batches={} batch_rewrites={} skinned_new={} skinned_model={} pose={} entities={} animated={}",
            self.mesh_new,
            self.mesh_rebuild,
            self.instance_rewrites,
            self.instance_batches,
            self.batch_rewrites,
            self.skinned_new,
            self.skinned_model_uploads,
            self.skinned_pose_writes,
            self.entities,
            self.animated
        )
    }

    pub fn draw_line(&self) -> String {
        format!(
            "shadow_atlas={} submit={} inside={} boundary={} tested={} groups={} indirect={}",
            self.shadow_atlas,
            self.instance_submit,
            self.direct_inside_batches,
            self.boundary_batches,
            self.cull_instances,
            self.cull_workgroups,
            self.indirect_draws
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

struct MainCullSelection {
    ids: Vec<EntityId>,
    compacted: HashSet<EntityId>,
}

struct ShadowDraw {
    id: EntityId,
    submit: MeshSubmit,
}

struct ShadowCullSelection {
    draws: Vec<ShadowDraw>,
    batch_draws: Vec<ShadowDraw>,
    compact_batches: Vec<EntityId>,
}

struct GpuInstanceBatch {
    gpu: GpuMesh,
    /// Ordered source entities and the revisions packed into `gpu`.
    sources: Vec<(EntityId, u64)>,
}

#[derive(Default)]
struct BatchBuild {
    sources: Vec<(EntityId, u64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchSourceState {
    id: EntityId,
    prototype: EntityId,
    xform_rev: u64,
    instance_count: usize,
    material: Option<SurfaceMaterialRef>,
    albedo: Option<TextureId>,
    casts_shadow: bool,
    prototype_material: Option<SurfaceMaterialRef>,
    prototype_albedo: Option<TextureId>,
    prototype_casts_shadow: bool,
}

pub(crate) const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::Greater;
pub(crate) const DEPTH_CLEAR: f32 = 0.0;
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipelines: Pipelines,
    terrain: TerrainPipelines,
    water: WaterPipelines,
    skinned: SkinnedPipelines,
    sky: SkyPipelines,
    shadow: ShadowGpu,
    shadow_vp: [glam::Mat4; 3],
    clipmap: Option<ClipmapRenderer>,
    depth_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    gpu_meshes: HashMap<EntityId, GpuMesh>,
    /// GPU-mode instance buffers, one per prototype instead of one per spatial bin.
    gpu_instance_batches: HashMap<EntityId, GpuInstanceBatch>,
    gpu_batch_sources: Vec<BatchSourceState>,
    gpu_batch_source_scratch: Vec<BatchSourceState>,
    gpu_skinned: HashMap<EntityId, GpuSkinnedEntity>,
    /// Vertex/index buffers keyed by `Arc::as_ptr` of the shared [`AnimatedModel`].
    gpu_skinned_meshes: HashMap<usize, Vec<GpuSkinnedMesh>>,
    gpu_textures: HashMap<TextureId, GpuTexture>,
    gpu_mesh_albedo: HashMap<TextureId, wgpu::BindGroup>,
    gpu_materials: HashMap<MaterialId, GpuTerrainMaterial>,
    gpu_water_materials: HashMap<WaterMaterialId, GpuWaterMaterial>,
    /// Origin the terrain material phases were last written for.
    terrain_origin: crate::space::RenderOrigin,
    /// This frame's view volume, for skipping draws outside it.
    frustum: Frustum,
    size: PhysicalSize<u32>,
    /// Reused while packing instance rows for a dirty entity.
    instance_scratch: Vec<InstanceRaw>,
    draw_opaque: Vec<EntityId>,
    draw_terrain: Vec<EntityId>,
    draw_transparent: Vec<EntityId>,
    draw_water: Vec<EntityId>,
    batch_opaque: Vec<EntityId>,
    batch_terrain: Vec<EntityId>,
    batch_transparent: Vec<EntityId>,
    batch_water: Vec<EntityId>,
    gpu_stats: GpuFrameStats,
    instance_cull: InstanceCull,
    joint_locals: Vec<(glam::Vec3, glam::Quat, glam::Vec3)>,
    joint_global: Vec<glam::Mat4>,
    joint_out: Vec<glam::Mat4>,
}

impl Renderer {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        // Uncapped present for stress testing / real FPS (Immediate → Mailbox → …).
        let present_mode = [
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::FifoRelaxed,
            wgpu::PresentMode::Fifo,
        ]
        .into_iter()
        .find(|m| caps.present_modes.contains(m))
        .unwrap_or(wgpu::PresentMode::Fifo);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let joint_layout = joint_bind_layout(&device);
        let shadow = ShadowGpu::new(&device, &queue, ShadowSettings::default(), &joint_layout);
        let pipelines = create_pipelines(&device, &queue, format, &shadow);
        let terrain = create_terrain_pipelines(&device, format, &pipelines.bind_layout);
        let water = create_water_pipelines(&device, format, &pipelines.bind_layout);
        let skinned =
            create_skinned_pipelines(&device, format, &pipelines.bind_layout, joint_layout);
        let sky = create_sky_pipelines(&device, format);
        let (depth_texture, depth_view) = create_depth(&device, config.width, config.height);
        let instance_cull = InstanceCull::new(&device);

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            terrain,
            water,
            skinned,
            sky,
            shadow,
            shadow_vp: [glam::Mat4::IDENTITY; 3],
            clipmap: None,
            depth_view,
            depth_texture,
            gpu_meshes: HashMap::new(),
            gpu_instance_batches: HashMap::new(),
            gpu_batch_sources: Vec::new(),
            gpu_batch_source_scratch: Vec::new(),
            gpu_skinned: HashMap::new(),
            gpu_skinned_meshes: HashMap::new(),
            gpu_textures: HashMap::new(),
            gpu_mesh_albedo: HashMap::new(),
            gpu_materials: HashMap::new(),
            gpu_water_materials: HashMap::new(),
            terrain_origin: crate::space::RenderOrigin::default(),
            frustum: Frustum::default(),
            size,
            instance_scratch: Vec::new(),
            draw_opaque: Vec::new(),
            draw_terrain: Vec::new(),
            draw_transparent: Vec::new(),
            draw_water: Vec::new(),
            batch_opaque: Vec::new(),
            batch_terrain: Vec::new(),
            batch_transparent: Vec::new(),
            batch_water: Vec::new(),
            gpu_stats: GpuFrameStats::default(),
            instance_cull,
            joint_locals: Vec::new(),
            joint_global: Vec::new(),
            joint_out: Vec::new(),
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        let (depth_texture, depth_view) =
            create_depth(&self.device, self.config.width, self.config.height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    pub fn take_gpu_stats(&mut self) -> GpuFrameStats {
        std::mem::take(&mut self.gpu_stats)
    }

    pub fn sync_world(&mut self, world: &World) {
        self.gpu_stats = GpuFrameStats::default();
        self.gpu_stats.entities = world.entities().count() as u32;
        self.gpu_stats.animated = world.animated_entities().count() as u32;
        self.gpu_stats.instance_submit = world.instance_submit();
        self.sync_textures_and_materials(world);

        if let Some(proc) = world.proc_terrain() {
            let format = self.config.format;
            match self.clipmap.as_mut() {
                Some(clip) => clip.ensure_config(&self.device, format, &proc.config),
                None => {
                    self.clipmap = Some(ClipmapRenderer::new(
                        &self.device,
                        format,
                        proc.config.clone(),
                        &self.shadow.resource_layout,
                    ));
                }
            }
            if let Some(clip) = self.clipmap.as_mut() {
                let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
                let vp = world.camera.view_projection(aspect);
                let light_dir = world.light.direction.normalize_or_zero();
                clip.prepare(
                    &self.queue,
                    vp,
                    light_dir,
                    world.light.ambient,
                    world.light.color,
                    world.camera.eye,
                    proc,
                );
            }
        } else {
            self.clipmap = None;
        }

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
                self.sync_gpu_instance_batches(world);
            }
        }

        self.gpu_skinned
            .retain(|id, _| world.contains_animated(*id));
        let live_models: HashSet<usize> = world
            .animated_entities()
            .map(|(_, anim)| Arc::as_ptr(&anim.animator.model) as usize)
            .collect();
        self.gpu_skinned_meshes
            .retain(|k, _| live_models.contains(k));

        for (id, anim) in world.animated_entities() {
            crate::anim::write_joint_matrices(
                &anim.animator.model,
                anim.animator.clip_index,
                anim.animator.time,
                &mut self.joint_locals,
                &mut self.joint_global,
                &mut self.joint_out,
            );
            let joints = std::mem::take(&mut self.joint_out);
            match self.gpu_skinned.get_mut(id) {
                Some(gpu) => {
                    gpu.update(&self.queue, anim.transform, &joints);
                    self.gpu_stats.skinned_pose_writes += 1;
                }
                None => {
                    let key = Arc::as_ptr(&anim.animator.model) as usize;
                    let meshes = if let Some(shared) = self.gpu_skinned_meshes.get(&key) {
                        shared.clone()
                    } else {
                        let uploaded: Vec<GpuSkinnedMesh> = anim
                            .animator
                            .model
                            .meshes
                            .iter()
                            .map(|m| GpuSkinnedMesh::upload(&self.device, m))
                            .collect();
                        self.gpu_skinned_meshes.insert(key, uploaded.clone());
                        self.gpu_stats.skinned_model_uploads += 1;
                        uploaded
                    };
                    self.gpu_stats.skinned_new += 1;
                    self.gpu_skinned.insert(
                        *id,
                        GpuSkinnedEntity::from_shared_meshes(
                            &self.device,
                            &self.skinned.joint_bind_layout,
                            meshes,
                            anim.transform,
                            &joints,
                        ),
                    );
                }
            }
            self.joint_out = joints;
        }
    }

    fn sync_entity_mesh(&mut self, id: EntityId, entity: &Entity) {
        if entity.instanced && entity.instances.is_empty() {
            if let Some(gpu) = self.gpu_meshes.get_mut(&id) {
                if gpu.xform_rev != entity.xform_rev {
                    gpu.clear_instances();
                    gpu.xform_rev = entity.xform_rev;
                }
            } else if entity.instance_of.is_none() {
                let mut gpu =
                    GpuMesh::upload(&self.device, entity.mesh(), &[], &self.instance_cull);
                gpu.xform_rev = entity.xform_rev;
                self.gpu_meshes.insert(id, gpu);
            }
            return;
        }

        if let Some(gpu) = self.gpu_meshes.get(&id) {
            let same_mesh = entity.instance_of.is_some()
                || (gpu.vertex_count == entity.mesh().vertex_count()
                    && gpu.index_count == entity.mesh().index_count());
            if same_mesh && gpu.xform_rev == entity.xform_rev {
                return;
            }
        }

        self.instance_scratch.clear();
        if entity.instanced {
            self.instance_scratch.extend(
                entity
                    .instances
                    .iter()
                    .map(|m| InstanceRaw::from_matrix(entity.transform * *m)),
            );
        } else {
            self.instance_scratch
                .push(InstanceRaw::from_matrix(entity.transform));
        }

        match self.gpu_meshes.get_mut(&id) {
            Some(gpu) => {
                if entity.instance_of.is_none()
                    && (gpu.vertex_count != entity.mesh().vertex_count()
                        || gpu.index_count != entity.mesh().index_count())
                {
                    *gpu = GpuMesh::upload(
                        &self.device,
                        entity.mesh(),
                        &self.instance_scratch,
                        &self.instance_cull,
                    );
                    self.gpu_stats.mesh_rebuild += 1;
                } else {
                    gpu.update_instances(
                        &self.device,
                        &self.queue,
                        &self.instance_scratch,
                        &self.instance_cull,
                    );
                    self.gpu_stats.instance_rewrites += 1;
                }
                gpu.xform_rev = entity.xform_rev;
            }
            None => {
                let mut gpu = if let Some(proto) = entity.instance_of {
                    let proto_gpu = self.gpu_meshes.get(&proto).unwrap_or_else(|| {
                        panic!("instanced-like entity {id} has no GPU mesh for prototype {proto}")
                    });
                    GpuMesh::share_vertices(
                        &self.device,
                        proto_gpu,
                        &self.instance_scratch,
                        &self.instance_cull,
                    )
                } else {
                    GpuMesh::upload(
                        &self.device,
                        entity.mesh(),
                        &self.instance_scratch,
                        &self.instance_cull,
                    )
                };
                gpu.xform_rev = entity.xform_rev;
                self.gpu_meshes.insert(id, gpu);
                self.gpu_stats.mesh_new += 1;
            }
        }
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
                .push((source.id, source.xform_rev));
        }

        self.gpu_instance_batches
            .retain(|prototype, _| builds.contains_key(prototype));
        for (prototype, build) in builds {
            let unchanged = self
                .gpu_instance_batches
                .get(&prototype)
                .is_some_and(|batch| batch.sources == build.sources);
            if unchanged {
                continue;
            }

            self.instance_scratch.clear();
            for &(source, _) in &build.sources {
                let entity = world
                    .entity(source)
                    .unwrap_or_else(|_| panic!("instance batch source {source} disappeared"));
                self.instance_scratch.extend(
                    entity
                        .instances
                        .iter()
                        .map(|matrix| InstanceRaw::from_matrix(entity.transform * *matrix)),
                );
            }

            if let Some(batch) = self.gpu_instance_batches.get_mut(&prototype) {
                batch.gpu.update_instances(
                    &self.device,
                    &self.queue,
                    &self.instance_scratch,
                    &self.instance_cull,
                );
                batch.sources = build.sources;
            } else {
                let prototype_gpu = self.gpu_meshes.get(&prototype).unwrap_or_else(|| {
                    panic!("instance batch has no GPU mesh for prototype {prototype}")
                });
                self.gpu_instance_batches.insert(
                    prototype,
                    GpuInstanceBatch {
                        gpu: GpuMesh::share_vertices(
                            &self.device,
                            prototype_gpu,
                            &self.instance_scratch,
                            &self.instance_cull,
                        ),
                        sources: build.sources,
                    },
                );
            }
            self.gpu_stats.batch_rewrites += 1;
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
        self.write_uniforms(world);
        self.begin_instance_cull_frame(world);

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });
        self.encode_shadow_pass(&mut encoder, world);
        self.encode_pass(&mut encoder, &view, world);
        after(&self.device, &self.queue, &mut encoder, &view);
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
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
    pub fn capture_png(&mut self, world: &World, path: impl AsRef<std::path::Path>) {
        self.write_uniforms(world);
        self.begin_instance_cull_frame(world);

        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let format = self.config.format;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

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
        self.encode_shadow_pass(&mut encoder, world);
        self.encode_pass(&mut encoder, &view, world);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
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
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
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

        image::save_buffer(path.as_ref(), &rgba, width, height, image::ColorType::Rgba8)
            .unwrap_or_else(|e| panic!("failed to save screenshot: {e}"));
    }

    fn write_uniforms(&mut self, world: &World) {
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let vp = world.camera.view_projection(aspect);
        self.frustum = Frustum::from_view_projection(vp);
        let light_dir = world.light.direction.normalize_or_zero();
        let eye = world.camera.eye;
        let haze = world.haze();
        let uniforms = Uniforms {
            view_proj: vp.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z],
            ambient: world.light.ambient,
            light_color: world.light.color.into(),
            _pad: 0.0,
            eye: [eye.x, eye.y, eye.z],
            time: world.time(),
            haze_color: haze
                .map(|h| h.color.to_vec3().into())
                .unwrap_or([1.0, 1.0, 1.0]),
            haze_density: haze.map(|h| h.density()).unwrap_or(0.0),
            haze_height_m: haze.map(|h| h.height_m.max(1.0)).unwrap_or(1.0),
            haze_base_y: haze.map(|h| h.base_y).unwrap_or(0.0),
            _pad2: [0.0, 0.0],
        };
        self.queue.write_buffer(
            &self.pipelines.uniform_buf,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        if let Some(sky) = world.sky() {
            let sky_u =
                SkyUniforms::from_scene(&sky, &world.camera, &world.light, aspect, world.time());
            self.queue
                .write_buffer(&self.sky.uniform_buf, 0, bytemuck::bytes_of(&sky_u));
        }
        self.shadow_vp = self.shadow.prepare(&self.queue, world);
        self.gpu_stats.shadow_atlas = self.shadow.atlas_wrote();
    }

    fn encode_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        world: &World,
    ) {
        self.collect_draws(world);
        let submit = world.instance_submit();
        let main_cull = if submit == InstanceSubmit::GpuIndirect {
            let frustum = self.frustum;
            let selection = self.select_main_culls(&frustum);
            self.dispatch_instance_cull(encoder, &frustum, &selection.ids, CullView::Main);
            selection
        } else {
            MainCullSelection {
                ids: Vec::new(),
                compacted: HashSet::new(),
            }
        };

        let clear = world.clear_color.to_vec3();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
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
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        if world.sky().is_some() {
            pass.set_pipeline(&self.sky.pipeline);
            pass.set_bind_group(0, &self.sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // GPU procgen terrain (depth-writing land), then entity meshes.
        if let Some(clip) = self.clipmap.as_ref() {
            clip.draw_land(&mut pass, &self.shadow.resource_bind);
        }

        pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

        // Opaque untextured, then terrain-textured, then skinned / transparent.
        pass.set_pipeline(&self.pipelines.opaque);
        for &id in &self.draw_opaque {
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
                &mut self.gpu_stats.indirect_draws,
            );
        }
        for &prototype in &self.batch_opaque {
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced")
                .gpu;
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, prototype),
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        pass.set_pipeline(&self.terrain.opaque);
        for &id in &self.draw_terrain {
            let entity = world.entity(id).expect("draw list is live");
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                continue;
            };
            let Some(mat) = self.gpu_materials.get(&mid) else {
                continue;
            };
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, id),
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }
        for &prototype in &self.batch_terrain {
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced")
                .gpu;
            let Some(SurfaceMaterialRef::Terrain(mid)) = entity.material else {
                panic!("terrain batch {prototype} lost its terrain material");
            };
            let mat = self
                .gpu_materials
                .get(&mid)
                .unwrap_or_else(|| panic!("terrain batch {prototype} material is not synced"));
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, prototype),
                0..gpu.opaque_index_count as u32,
                OPAQUE_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        pass.set_pipeline(&self.skinned.opaque);
        for gpu in self.gpu_skinned.values() {
            pass.set_bind_group(1, &gpu.joint_bind, &[]);
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            for mesh in &gpu.meshes {
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        // Restore scene bind group for subsequent passes.
        pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

        pass.set_pipeline(&self.pipelines.transparent);
        for &id in &self.draw_transparent {
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
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }
        for &prototype in &self.batch_transparent {
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced")
                .gpu;
            let albedo = entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo);
            pass.set_bind_group(1, albedo, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, prototype),
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        // Water sheets last, so everything standing in them is already in the
        // colour buffer to blend against.
        pass.set_pipeline(&self.water.blend);
        for &id in &self.draw_water {
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
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, id),
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }
        for &prototype in &self.batch_water {
            let entity = world
                .entity(prototype)
                .expect("batch draw prototype is live");
            let gpu = &self
                .gpu_instance_batches
                .get(&prototype)
                .expect("batch draw is synced")
                .gpu;
            let Some(SurfaceMaterialRef::Water(mid)) = entity.material else {
                panic!("water batch {prototype} lost its water material");
            };
            let mat = self
                .gpu_water_materials
                .get(&mid)
                .unwrap_or_else(|| panic!("water batch {prototype} material is not synced"));
            pass.set_bind_group(1, &mat.bind_group, &[]);
            submit_mesh_draw(
                &mut pass,
                gpu,
                mesh_submit(&main_cull.compacted, prototype),
                gpu.opaque_index_count as u32..gpu.index_count as u32,
                TRANSLUCENT_INDIRECT_OFFSET,
                &mut self.gpu_stats.indirect_draws,
            );
        }

        // Translucent clipmap water after meshes so the walker can occlude shorelines.
        if let Some(clip) = self.clipmap.as_ref() {
            clip.draw_water(&mut pass, &self.shadow.resource_bind);
        }
    }

    fn encode_shadow_pass(&mut self, encoder: &mut wgpu::CommandEncoder, world: &World) {
        if world.shadows().is_none() {
            return;
        }
        let far = world.shadows().map(|s| s.cascade_end_m[2]).unwrap_or(120.0);
        let focus = world.camera.target;
        let submit = world.instance_submit();
        for i in 0..3 {
            let frustum = Frustum::from_view_projection(self.shadow_vp[i]);
            let gpu_selection = if submit == InstanceSubmit::GpuIndirect {
                let selection = self.select_shadow_culls(world, &frustum);
                self.dispatch_instance_cull(
                    encoder,
                    &frustum,
                    &selection.compact_batches,
                    CullView::Shadow,
                );
                Some(selection)
            } else {
                None
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-csm"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow.layer_views[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            match submit {
                InstanceSubmit::CpuIndexed => {
                    self.shadow
                        .draw_mesh_casters(&mut pass, i, world, &self.gpu_meshes, &frustum);
                }
                InstanceSubmit::GpuIndirect => {
                    pass.set_pipeline(&self.shadow.mesh_pipeline);
                    pass.set_bind_group(0, &self.shadow.cascade_binds[i], &[]);
                    for draw in &gpu_selection
                        .as_ref()
                        .expect("GPU selection exists in GPU mode")
                        .draws
                    {
                        let gpu = self.gpu_meshes.get(&draw.id).expect("caster is synced");
                        submit_mesh_draw(
                            &mut pass,
                            gpu,
                            draw.submit,
                            0..gpu.opaque_index_count as u32,
                            OPAQUE_INDIRECT_OFFSET,
                            &mut self.gpu_stats.indirect_draws,
                        );
                    }
                    for draw in &gpu_selection
                        .as_ref()
                        .expect("GPU selection exists in GPU mode")
                        .batch_draws
                    {
                        let gpu = &self
                            .gpu_instance_batches
                            .get(&draw.id)
                            .expect("shadow instance batch is synced")
                            .gpu;
                        submit_mesh_draw(
                            &mut pass,
                            gpu,
                            draw.submit,
                            0..gpu.opaque_index_count as u32,
                            OPAQUE_INDIRECT_OFFSET,
                            &mut self.gpu_stats.indirect_draws,
                        );
                    }
                }
            }
            pass.set_pipeline(&self.shadow.skinned_pipeline);
            pass.set_bind_group(0, &self.shadow.cascade_binds[i], &[]);
            for (id, anim) in world.animated_entities() {
                let Some(gpu) = self.gpu_skinned.get(id) else {
                    continue;
                };
                let t = anim.transform().w_axis.truncate();
                if t.distance(focus) > far + 16.0 {
                    continue;
                }
                pass.set_bind_group(1, &gpu.joint_bind, &[]);
                pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
                for mesh in &gpu.meshes {
                    pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                    pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
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
                let gpu = &self
                    .gpu_instance_batches
                    .get(&prototype)
                    .expect("batch draw list is synced")
                    .gpu;
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
                    Visibility::Inside => self.gpu_stats.direct_inside_batches += 1,
                }
            }
        }
        MainCullSelection { ids, compacted }
    }

    fn select_shadow_culls(
        &mut self,
        world: &World,
        light_frustum: &Frustum,
    ) -> ShadowCullSelection {
        let mut draws = Vec::new();
        let mut batch_draws = Vec::new();
        let mut compact_batches = Vec::new();
        for (id, entity) in world.entities() {
            if entity.instanced()
                || !entity.casts_shadow()
                || !material_casts_shadow(entity.material())
            {
                continue;
            }
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            if gpu.instance_count == 0 || gpu.opaque_index_count == 0 {
                continue;
            }
            let Some(bounds) = gpu.bounds else {
                continue;
            };
            match light_frustum.classify(bounds) {
                Visibility::Outside => {}
                Visibility::Inside | Visibility::Intersecting => draws.push(ShadowDraw {
                    id,
                    submit: MeshSubmit::Direct,
                }),
            }
        }
        for (&prototype, batch) in &self.gpu_instance_batches {
            let entity = world
                .entity(prototype)
                .unwrap_or_else(|_| panic!("instance batch prototype {prototype} disappeared"));
            if !entity.casts_shadow() || !material_casts_shadow(entity.material()) {
                continue;
            }
            let gpu = &batch.gpu;
            if gpu.instance_count == 0 || gpu.opaque_index_count == 0 {
                continue;
            }
            let bounds = gpu
                .bounds
                .unwrap_or_else(|| panic!("non-empty instance batch {prototype} has no bounds"));
            match light_frustum.classify(bounds) {
                Visibility::Outside => {}
                Visibility::Inside => {
                    self.gpu_stats.direct_inside_batches += 1;
                    batch_draws.push(ShadowDraw {
                        id: prototype,
                        submit: MeshSubmit::Direct,
                    });
                }
                Visibility::Intersecting => {
                    compact_batches.push(prototype);
                    batch_draws.push(ShadowDraw {
                        id: prototype,
                        submit: MeshSubmit::Compacted,
                    });
                }
            }
        }
        ShadowCullSelection {
            draws,
            batch_draws,
            compact_batches,
        }
    }

    fn dispatch_instance_cull(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frustum: &Frustum,
        ids: &[EntityId],
        view: CullView,
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
                CullJob { gpu, local, draws }
            })
            .collect();
        self.instance_cull
            .dispatch(&self.queue, encoder, frustum, &jobs);
        let job_count = u32::try_from(jobs.len()).expect("cull dispatch count exceeds u32");
        self.gpu_stats.compute_culls += job_count;
        self.gpu_stats.boundary_batches += job_count;
        for job in jobs {
            self.gpu_stats.cull_instances += job.gpu.instance_count as u64;
            self.gpu_stats.cull_workgroups +=
                (job.gpu.instance_count as u64).div_ceil(WORKGROUP as u64);
        }
    }

    fn begin_instance_cull_frame(&mut self, world: &World) {
        if world.instance_submit() != InstanceSubmit::GpuIndirect {
            return;
        }
        let views = if world.shadows().is_some() {
            4usize
        } else {
            1usize
        };
        let max_jobs = self
            .gpu_instance_batches
            .len()
            .checked_mul(views)
            .expect("instance-cull frame job count overflow");
        self.instance_cull.begin_frame(&self.device, max_jobs);
    }

    /// One walk of the live entities, bucketed for the four mesh passes.
    fn collect_draws(&mut self, world: &World) {
        self.draw_opaque.clear();
        self.draw_terrain.clear();
        self.draw_transparent.clear();
        self.draw_water.clear();
        self.batch_opaque.clear();
        self.batch_terrain.clear();
        self.batch_transparent.clear();
        self.batch_water.clear();
        let gpu_batches = world.instance_submit() == InstanceSubmit::GpuIndirect;
        for (id, entity) in world.entities() {
            if gpu_batches && entity.instanced() {
                continue;
            }
            let (has_opaque, has_xlucent) = {
                let Some(gpu) = self.gpu_meshes.get(&id) else {
                    continue;
                };
                if gpu.instance_count == 0 || self.hidden(gpu) {
                    continue;
                }
                (
                    gpu.opaque_index_count > 0,
                    gpu.opaque_index_count < gpu.index_count,
                )
            };
            match entity.material {
                None => {
                    if has_opaque {
                        self.draw_opaque.push(id);
                    }
                    if has_xlucent {
                        self.draw_transparent.push(id);
                    }
                }
                Some(SurfaceMaterialRef::Terrain(_)) => {
                    if has_opaque {
                        self.draw_terrain.push(id);
                    }
                    if has_xlucent {
                        self.draw_transparent.push(id);
                    }
                }
                Some(SurfaceMaterialRef::Water(_)) => {
                    if has_xlucent {
                        self.draw_water.push(id);
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
                if gpu.instance_count == 0 || self.hidden(gpu) {
                    continue;
                }
                let entity = world
                    .entity(prototype)
                    .unwrap_or_else(|_| panic!("instance batch prototype {prototype} disappeared"));
                let has_opaque = gpu.opaque_index_count > 0;
                let has_xlucent = gpu.opaque_index_count < gpu.index_count;
                match entity.material {
                    None => {
                        if has_opaque {
                            self.batch_opaque.push(prototype);
                        }
                        if has_xlucent {
                            self.batch_transparent.push(prototype);
                        }
                    }
                    Some(SurfaceMaterialRef::Terrain(_)) => {
                        if has_opaque {
                            self.batch_terrain.push(prototype);
                        }
                        if has_xlucent {
                            self.batch_transparent.push(prototype);
                        }
                    }
                    Some(SurfaceMaterialRef::Water(_)) => {
                        if has_xlucent {
                            self.batch_water.push(prototype);
                        }
                    }
                }
            }
        }
    }

    /// Whether this mesh is entirely outside the view volume.
    ///
    /// A mesh with no bounds has nothing to draw, and one whose sphere reaches
    /// into the frustum is kept: the test errs towards drawing, because the
    /// cost of a wrong answer is terrain blinking out at the edge of the screen.
    fn hidden(&self, gpu: &GpuMesh) -> bool {
        match gpu.bounds {
            Some(bounds) => !self.frustum.intersects(bounds),
            None => true,
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
            let gpu = self
                .gpu_textures
                .get(&tid)
                .expect("mesh albedo missing on GPU");
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mesh-albedo"),
                layout: &self.pipelines.albedo_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.pipelines.albedo_sampler),
                    },
                ],
            });
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
                [grass, grass_dry, grass_moor, sand, rock],
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
            pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
            pass.draw_indexed(indices, 0, 0..gpu.instance_count as u32);
        }
    }
}

fn mesh_submit(compacted: &HashSet<EntityId>, id: EntityId) -> MeshSubmit {
    if compacted.contains(&id) {
        MeshSubmit::Compacted
    } else {
        MeshSubmit::Direct
    }
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
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
