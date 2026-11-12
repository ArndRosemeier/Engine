//! Copying the world into GPU buffers each frame: entity meshes, instance batches, materials.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
    pub fn sync_world(&mut self, world: &mut World) {
        let mut t = std::time::Instant::now();
        self.reupload_if_asked(world);
        self.rebuild_if_asked(world);
        self.apply_present_mode(world.vsync());
        self.frame_no += 1;
        if self.frame_no.is_multiple_of(512) {
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
            self.materials_synced += 1;
        }

        self.sync_custom_materials(world);

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
        let mut touched = world.take_touched();
        touched.sort_unstable();
        touched.dedup();
        let (spawned, despawned) = world.take_members();
        let mut full = self.synced_render_epoch != world.render_epoch();
        if !full && !self.sync_members(world, &spawned, &despawned) {
            full = true;
            // What the world last changed in full is not it this time.
            self.note_full_sync(None);
        }
        if !full {
            // Only moves, new instance lists and plain meshes that came or
            // went: bring just those entities up to date (walking every
            // entity cost milliseconds a frame in a town where something
            // moves every frame and streams in every few).
            self.sync_touched(world, &touched);
            if self.gpu_scene.is_some() {
                self.scene_touched.extend_from_slice(&touched);
            }
            if self.shadow_scene.is_some() {
                self.shadow_touched.extend_from_slice(&touched);
            }
            self.synced_instance_epoch = world.instance_epoch();
            if !spawned.is_empty() {
                let gpu = &self.gpu_meshes;
                world.release_uploaded(|id| gpu.contains_key(&id));
            }
            if super::renderer_gpu_sync::scene_check() {
                self.check_sources(world);
            }
            self.gpu_stats.instance_batches = u32::try_from(self.gpu_instance_batches.len())
                .expect("GPU instance batch count exceeds u32");
            self.cpu_spans.mark("sync.touched", &mut t);
        } else {
            if self.synced_render_epoch != world.render_epoch() {
                self.note_full_sync(world.render_bump_site);
            }
            self.full_syncs += 1;
            self.scene_gen += 1;
            self.scene_added.clear();
            self.scene_removed.clear();
            self.shadow_added.clear();
            self.shadow_removed.clear();
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
            self.cpu_spans.mark("sync.full", &mut t);
            // Meshes the world no longer needs in RAM go once they are here.
            let gpu = &self.gpu_meshes;
            world.release_uploaded(|id| gpu.contains_key(&id));
            self.cpu_spans.mark("sync.release", &mut t);
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
        self.skinned_reach.retain(|k, _| live_models.contains(k));

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
            let key = Arc::as_ptr(anim.animator.model()) as usize;
            let bounds = self
                .skinned_reach
                .entry(key)
                .or_insert_with(|| Arc::new(skinned::SkinReach::of(anim.animator.model())))
                .bounds(&joints, anim.transform);
            match self.gpu_skinned.get_mut(id) {
                Some(gpu) => {
                    gpu.bounds = bounds;
                    gpu.update(
                        &self.queue,
                        anim.transform,
                        &joints,
                        anim.animator.morph_weights(),
                    );
                    self.gpu_stats.skinned_pose_writes += 1;
                }
                None => {
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
                    let mut gpu = GpuSkinnedEntity::from_shared_meshes(
                        &self.device,
                        &self.skinned.joint_bind_layout,
                        meshes,
                        &morphs,
                        anim.transform,
                        &joints,
                        anim.animator.morph_weights(),
                    );
                    gpu.bounds = bounds;
                    self.gpu_skinned.insert(*id, gpu);
                }
            }
            self.poses[i] = joints;
        }
        self.cpu_spans.mark("sync.skinned", &mut t);
    }

    /// The material-table slot a mesh with `albedo` draws with (texture
    /// arrays), as `albedo_of` picks its bind group; `None` without a table
    /// or when the table is full.
    pub(super) fn material_slot(&self, albedo: Option<TextureId>) -> Option<u32> {
        let table = &self.bindless.as_ref()?.table;
        match albedo {
            Some(tid) if self.gpu_mesh_albedo.contains_key(&tid) => table.slot(Some(tid)),
            _ => table.slot(None),
        }
    }

    /// Pipelines for new custom materials, params for changed ones.
    ///
    /// A new material's two pipelines compile on a worker thread (a driver
    /// compile can take tens of milliseconds); things wearing it are drawn
    /// from the frame it is ready. Fixed-step runs (verify, screenshots)
    /// wait for it, so their frames stay the same every run.
    fn sync_custom_materials(&mut self, world: &World) {
        self.gpu_custom
            .retain(|id, _| world.custom_materials.contains_key(id));
        self.custom_compiling
            .retain(|id, _| world.custom_materials.contains_key(id));
        if world.custom_materials.is_empty() {
            return;
        }
        let layout = self
            .custom_params_layout
            .get_or_insert_with(|| pipeline::custom_params_layout(&self.device))
            .clone();
        for (id, (material, _)) in &world.custom_materials {
            if self.gpu_custom.contains_key(id) || self.custom_compiling.contains_key(id) {
                continue;
            }
            let device = self.device.clone();
            let scene = self.pipelines.bind_layout.clone();
            let albedo = self.pipelines.albedo_layout.clone();
            let params = layout.clone();
            let material = material.clone();
            let job = std::thread::Builder::new()
                .name("engine-custom-material".into())
                .spawn(move || {
                    pipeline::create_custom_material(
                        &device,
                        HDR_FORMAT,
                        [&scene, &albedo],
                        &params,
                        &material,
                    )
                })
                .expect("spawn custom material compile thread");
            self.custom_compiling.insert(*id, job);
        }
        let ready: Vec<_> = self
            .custom_compiling
            .iter()
            .filter(|(_, job)| self.wait_for_workers || job.is_finished())
            .map(|(id, _)| *id)
            .collect();
        for id in ready {
            let job = self.custom_compiling.remove(&id).expect("listed above");
            let gpu = match job.join() {
                Ok(gpu) => gpu,
                // A WGSL mistake is caught by `add_material`; a panic here is a bug.
                Err(panic) => std::panic::resume_unwind(panic),
            };
            self.gpu_custom.insert(id, gpu);
        }
        for (id, (material, revision)) in &world.custom_materials {
            let Some(gpu) = self.gpu_custom.get_mut(id) else {
                continue;
            };
            if gpu.revision != *revision {
                self.queue
                    .write_buffer(&gpu.params, 0, bytemuck::cast_slice(&material.params));
                gpu.revision = *revision;
            }
        }
    }

    /// Wait for work on other threads (custom material compiles) instead of
    /// drawing without it: fixed-step runs want the same frames every time.
    pub fn set_wait_for_workers(&mut self, wait: bool) {
        self.wait_for_workers = wait;
    }

    /// Brings the `touched` entities (moved, or with a new instance list of
    /// the same emptiness) up to date: their instance buffers, and the
    /// instance batches they belong to. Nothing else can have changed, or
    /// the world's render epoch would have moved.
    /// Adds the plain meshes the world spawned and drops those it
    /// despawned (see `World::take_members`) without a full sync: their GPU
    /// meshes, their places in the draw and shadow source lists, and (on
    /// their next sync) their GPU scene rows. False, having changed
    /// nothing, when one of them needs the full sync (a portal surface, a
    /// custom or water material, an instance batch prototype).
    fn sync_members(
        &mut self,
        world: &World,
        spawned: &[EntityId],
        despawned: &[EntityId],
    ) -> bool {
        if spawned.is_empty() && despawned.is_empty() {
            return true;
        }
        let simple = spawned.iter().all(|&id| match world.entity(id) {
            Ok(e) => {
                !e.instanced
                    && e.instance_of.is_none()
                    && !world.is_portal_surface(id)
                    && matches!(e.material, None | Some(SurfaceMaterialRef::Terrain(_)))
            }
            // Spawned and despawned again since the last sync.
            Err(_) => true,
        }) && despawned
            .iter()
            .all(|id| !self.gpu_instance_batches.contains_key(id));
        if !simple {
            return false;
        }
        self.scene_gen += 1;
        let gone: HashSet<EntityId> = despawned
            .iter()
            .copied()
            .filter(|id| {
                self.moved_at.remove(id);
                self.gpu_meshes.remove(id).is_some()
            })
            .collect();
        if !gone.is_empty() {
            let keep = |id: &EntityId| !gone.contains(id);
            self.source_opaque.retain(keep);
            self.source_terrain.retain(keep);
            self.source_transparent.retain(keep);
            self.source_water.retain(keep);
            let casters = self.shadow_mesh_sources.len();
            self.shadow_mesh_sources.retain(keep);
            if self.shadow_scene.is_none() && casters != self.shadow_mesh_sources.len() {
                // The sun's cached static shadows may hold one of them.
                self.static_shadow_epoch += 1;
            }
            if self.gpu_scene.is_some() {
                self.scene_removed.extend(gone.iter().copied());
            }
            if self.shadow_scene.is_some() {
                self.shadow_removed.extend(gone.iter().copied());
            }
        }
        let gpu_batches = world.instance_submit() == InstanceSubmit::GpuIndirect;
        for &id in spawned {
            let Ok(entity) = world.entity(id) else {
                continue;
            };
            self.sync_entity_mesh(id, entity);
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                continue;
            };
            // As `rebuild_draw_sources` and `rebuild_shadow_sources` sort it.
            let has_opaque = gpu.opaque_index_count > 0;
            let has_xlucent = gpu.opaque_index_count < gpu.index_count;
            let casts = entity.casts_shadow()
                && material_casts_shadow(entity.material())
                && gpu.instance_count > 0
                && has_opaque
                && !(gpu_batches && entity.instanced());
            match entity.material {
                Some(SurfaceMaterialRef::Terrain(_)) if has_opaque => {
                    self.source_terrain.push(id);
                }
                None if has_opaque => {
                    self.source_opaque.push(id);
                    if self.gpu_scene.is_some() {
                        self.scene_added.push(id);
                    }
                }
                _ => {}
            }
            if has_xlucent {
                self.source_transparent.push(id);
            }
            if casts {
                self.shadow_mesh_sources.push(id);
                if self.shadow_scene.is_some() {
                    self.shadow_added.push(id);
                }
            }
        }
        true
    }

    /// `ENGINE_SCENE_CHECK=1`: the draw and shadow source lists, kept up to
    /// date one entity at a time, must be what building them again gives.
    fn check_sources(&mut self, world: &World) {
        if world.instance_submit() == InstanceSubmit::GpuIndirect {
            // Each batch holds its entities' current places.
            let mut want: HashMap<EntityId, Vec<(EntityId, u64, usize)>> = HashMap::new();
            for (id, e) in world.entities() {
                if e.instanced {
                    want.entry(e.instance_of.unwrap_or(id)).or_default().push((
                        id,
                        e.xform_rev,
                        e.instances.len(),
                    ));
                }
            }
            assert_eq!(
                want.len(),
                self.gpu_instance_batches.len(),
                "ENGINE_SCENE_CHECK: instance batches differ from the world's"
            );
            for (prototype, mut want) in want {
                let batch = self
                    .gpu_instance_batches
                    .get(&prototype)
                    .unwrap_or_else(|| {
                        panic!("ENGINE_SCENE_CHECK: no instance batch for {prototype}")
                    });
                let mut held: Vec<_> = batch
                    .slots
                    .iter()
                    .map(|s| (s.id, s.xform_rev, s.active_count))
                    .collect();
                want.sort_unstable();
                held.sort_unstable();
                assert!(
                    want == held,
                    "ENGINE_SCENE_CHECK: instance batch {prototype} is out of date"
                );
            }
        }
        if self.draw_source_epoch == self.full_syncs {
            let held = (
                self.source_opaque.clone(),
                self.source_terrain.clone(),
                self.source_transparent.clone(),
                self.source_water.clone(),
            );
            self.rebuild_draw_sources(world);
            let built = (
                &self.source_opaque,
                &self.source_terrain,
                &self.source_transparent,
                &self.source_water,
            );
            assert!(
                (&held.0, &held.1, &held.2, &held.3) == built,
                "ENGINE_SCENE_CHECK: the draw source lists differ from a rebuild"
            );
        }
        if self.shadow_source_epoch == self.full_syncs {
            let held = self.shadow_mesh_sources.clone();
            let mut held_batches = self.shadow_batch_sources.clone();
            self.rebuild_shadow_sources(world);
            held_batches.sort_unstable();
            let mut batches = self.shadow_batch_sources.clone();
            batches.sort_unstable();
            assert!(
                held == self.shadow_mesh_sources && held_batches == batches,
                "ENGINE_SCENE_CHECK: the shadow source lists differ from a rebuild"
            );
        }
    }

    /// `ENGINE_CPU_TRACE`: count a full sync by where the world asked for
    /// it (`None`: something the renderer could not add one by one), and
    /// list them every 120 full syncs.
    fn note_full_sync(&mut self, site: Option<&'static std::panic::Location<'static>>) {
        if !self.cpu_spans.on {
            return;
        }
        *self.full_sync_sites.entry(site).or_insert(0) += 1;
        let total: u32 = self.full_sync_sites.values().sum();
        if total >= 120 {
            let mut sites: Vec<_> = self.full_sync_sites.drain().collect();
            sites.sort_by_key(|s| std::cmp::Reverse(s.1));
            let list: Vec<String> = sites
                .iter()
                .map(|(at, n)| match at {
                    Some(at) => format!("{at} x{n}"),
                    None => format!("spawns the renderer cannot add alone x{n}"),
                })
                .collect();
            eprintln!(
                "full renderer syncs (frame {}): {}",
                self.frame_no,
                list.join(", ")
            );
        }
    }

    fn sync_touched(&mut self, world: &World, touched: &[EntityId]) {
        let gpu_batches = world.instance_submit() == InstanceSubmit::GpuIndirect;
        let mut prototypes = Vec::new();
        for &id in touched {
            let Ok(entity) = world.entity(id) else {
                continue;
            };
            if entity.instance_of.is_none() || !gpu_batches {
                self.sync_entity_mesh(id, entity);
            }
            if gpu_batches && entity.instanced {
                prototypes.push(entity.instance_of.unwrap_or(id));
            }
        }
        prototypes.sort_unstable();
        prototypes.dedup();
        if !prototypes.is_empty() {
            if !self.sync_touched_batches(world, &prototypes) {
                self.sync_gpu_instance_batches(world);
            }
            if self.shadow_source_epoch == self.full_syncs {
                // A batch may have emptied or filled.
                self.rebuild_shadow_batch_sources(world);
            }
        }
    }

    pub(super) fn sync_entity_mesh(&mut self, id: EntityId, entity: &Entity) {
        if entity.instanced && entity.instances.is_empty() {
            if let Some(gpu) = self.gpu_meshes.get_mut(&id) {
                if gpu.xform_rev != entity.xform_rev {
                    gpu.clear_instances();
                    gpu.xform_rev = entity.xform_rev;
                }
            } else if entity.instance_of.is_none() {
                let mut gpu =
                    GpuMesh::upload(&self.device, &self.queue, &self.arenas, entity.mesh(), &[]);
                gpu.xform_rev = entity.xform_rev;
                self.gpu_meshes.insert(id, gpu);
            }
            return;
        }

        let material = self.material_slot(entity.albedo);
        if let Some(gpu) = self.gpu_meshes.get(&id) {
            let same_mesh = entity.instance_of.is_some()
                || (gpu.vertex_count == entity.vertex_count()
                    && gpu.index_count == entity.index_count());
            if same_mesh && gpu.xform_rev == entity.xform_rev && gpu.material == material {
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
        if let Some(slot) = material {
            for raw in &mut self.instance_scratch {
                raw.material = slot;
            }
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
                    );
                    self.gpu_stats.mesh_rebuild += 1;
                    true
                } else {
                    let reallocated = gpu.update_instances(
                        &self.device,
                        &self.queue,
                        &self.arenas,
                        &self.instance_scratch,
                    );
                    self.gpu_stats.instance_rewrites += 1;
                    reallocated
                };
                gpu.xform_rev = entity.xform_rev;
                gpu.material = material;
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
                    )
                } else {
                    GpuMesh::upload(
                        &self.device,
                        &self.queue,
                        &self.arenas,
                        entity.mesh(),
                        &self.instance_scratch,
                    )
                };
                gpu.xform_rev = entity.xform_rev;
                gpu.material = material;
                self.gpu_meshes.insert(id, gpu);
                self.gpu_stats.mesh_new += 1;
                false
            }
        };
        self.mark_moved(id);
        if shadow_buffers_changed {
            self.invalidate_shadow_bundles();
        }
    }

    /// Entity `id` (a mesh or a batch prototype) changed on the GPU this
    /// frame: its shadow is not cached as static for a while.
    fn mark_moved(&mut self, id: EntityId) {
        self.moved_at.insert(id, self.frame_no);
        if self.shadow_scene.is_some() {
            if self.settling.len() >= SETTLING_LIMIT {
                // Not drained (the shadow scene was not synced for long):
                // build it again instead of keeping more.
                self.settling.clear();
                if let Some(scene) = self.shadow_scene.as_mut() {
                    scene.stamp = None;
                }
            }
            self.settling.push_back((self.frame_no, id));
        }
    }

    /// Mesh or instance buffers moved: record the bundles that bake them
    /// in anew (shadow views, and the terrain bundle, which is keyed by
    /// its entities only).
    pub(super) fn invalidate_shadow_bundles(&mut self) {
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
    pub(super) fn update_batch_in_place(
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

    pub(super) fn sync_gpu_instance_batches(&mut self, world: &World) {
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
        let removed = self.gpu_instance_batches.len() != batch_count_before;
        if self.build_batches(world, builds) || removed {
            self.invalidate_shadow_bundles();
        }
        std::mem::swap(
            &mut self.gpu_batch_sources,
            &mut self.gpu_batch_source_scratch,
        );
        self.gpu_stats.instance_batches = u32::try_from(self.gpu_instance_batches.len())
            .expect("GPU instance batch count exceeds u32");
    }

    /// Brings just the batches of `prototypes` up to date (their entities'
    /// places or instance lists changed, which entities they hold did not:
    /// that is a full sync). False when one has no batch yet: then
    /// [`Self::sync_gpu_instance_batches`] must run.
    fn sync_touched_batches(&mut self, world: &World, prototypes: &[EntityId]) -> bool {
        let mut builds: HashMap<EntityId, BatchBuild> = HashMap::new();
        for prototype in prototypes {
            let Some(batch) = self.gpu_instance_batches.get(prototype) else {
                return false;
            };
            let mut build = BatchBuild::default();
            for slot in &batch.slots {
                let Ok(entity) = world.entity(slot.id) else {
                    return false;
                };
                build.sources.push(BatchBuildSource {
                    id: slot.id,
                    xform_rev: entity.xform_rev,
                    instance_count: entity.instances.len(),
                    instance_reserve: entity.instance_reserve,
                });
            }
            builds.insert(*prototype, build);
        }
        if self.build_batches(world, builds) {
            self.invalidate_shadow_bundles();
        }
        // The list a full sync compares with is out of date now.
        self.gpu_batch_sources.clear();
        true
    }

    /// Updates (in place where they fit) or builds again the batches of
    /// `builds`; true when a batch got new buffers.
    fn build_batches(&mut self, world: &World, builds: HashMap<EntityId, BatchBuild>) -> bool {
        let mut shadow_buffers_changed = false;
        for (prototype, build) in builds {
            if let Some(mut batch) = self.gpu_instance_batches.remove(&prototype) {
                let (writes, slots) = (self.gpu_stats.batch_partial_writes, batch.slots.len());
                if self.update_batch_in_place(world, &mut batch, &build) {
                    if writes != self.gpu_stats.batch_partial_writes || slots != batch.slots.len() {
                        self.mark_moved(prototype);
                    }
                    self.gpu_instance_batches.insert(prototype, batch);
                    continue;
                }
            }
            self.mark_moved(prototype);

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
                    free: std::iter::once(used..used + headroom).collect(),
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
        shadow_buffers_changed
    }
}
