//! The GPU scene, culling and draw sources: rows, occlusion, instance culls, textures.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

/// `ENGINE_SCENE_CHECK=1`: after each sync, compare the GPU scene and the
/// GPU shadow scene with what a rebuild from scratch would hold, and stop
/// with the first difference (the rows patched for moved entities must
/// never drift from the truth). Slow; for tests and verify runs.
pub(super) fn scene_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_SCENE_CHECK").is_ok_and(|v| v.trim() == "1"))
}

impl Renderer {
    /// Bring the GPU scene's rows up to date with the live space; its row
    /// count, or `None` without a GPU scene.
    pub(super) fn sync_gpu_scene(&mut self, world: &World, space: SpaceId) -> Option<usize> {
        let mut scene = self.gpu_scene.take()?;
        let stamp = gpu_scene::SceneStamp {
            render: self.full_syncs,
            resource: self.materials_synced,
            table: self.bindless.as_ref().map_or(0, |b| b.table.version()),
            space,
        };
        // Rooms hide and show meshes from frame to frame.
        let mut rebuild = scene.stamp != Some(stamp) || world.rooms_culling();
        let removed = std::mem::take(&mut self.scene_removed);
        let added = std::mem::take(&mut self.scene_added);
        let mut written = 0;
        if !rebuild && (!removed.is_empty() || !added.is_empty()) {
            // Plain meshes that came or went: drop and add just their rows.
            let rows: Vec<gpu_scene::SceneRow> = added
                .iter()
                .filter(|id| !scene.slots.contains_key(id))
                .filter_map(|&id| self.scene_row(world, space, id, 0))
                .collect();
            if scene.worn() || !self.occlusion.fits(scene.ids.len() + rows.len()) {
                rebuild = true;
            } else {
                for id in removed {
                    if let Some(slot) = scene.remove(id) {
                        self.occlusion.write_rows(
                            &self.queue,
                            slot,
                            &scene.rows[slot..slot + 1],
                            false,
                        );
                        written += 1;
                    }
                }
                let start = scene.ids.len();
                for row in rows {
                    scene.add(row);
                }
                let end = scene.ids.len();
                if end > start {
                    self.occlusion.write_row_groups_at(
                        &self.queue,
                        start,
                        &scene.row_groups_from(start),
                    );
                    self.occlusion
                        .write_rows(&self.queue, start, &scene.rows[start..end], true);
                    written += end - start;
                }
            }
        }
        if !rebuild {
            // Only moves and new instance lists since the last sync: patch
            // those rows (rebuilding every row each frame cost milliseconds
            // in a town where something moves every frame).
            let mut touched = std::mem::take(&mut self.scene_touched);
            touched.sort_unstable();
            touched.dedup();
            for &id in &touched {
                let row = self.scene_row(world, space, id, 0);
                let patched = match &row {
                    Some(row) => scene.patch(row),
                    // It has a row but no longer makes one: build again.
                    None if scene.slots.contains_key(&id) => Err(()),
                    None => Ok(None),
                };
                match patched {
                    Ok(Some(slot)) => {
                        self.occlusion.write_rows(
                            &self.queue,
                            slot,
                            &scene.rows[slot..slot + 1],
                            false,
                        );
                        written += 1;
                    }
                    Ok(None) => {}
                    Err(()) => {
                        rebuild = true;
                        break;
                    }
                }
            }
            scene.written = written;
        }
        if rebuild {
            self.scene_touched.clear();
            let rows = self.scene_rows(world, space);
            let all = self.occlusion.reserve_rows(&self.device, rows.len());
            let writes = scene.apply(rows, all);
            if scene.regrouped {
                self.occlusion
                    .write_row_groups(&self.queue, &scene.row_groups());
            }
            for w in writes {
                self.occlusion.write_rows(
                    &self.queue,
                    w.start,
                    &scene.rows[w.start..w.start + w.len],
                    w.reset_seen,
                );
            }
            scene.stamp = Some(stamp);
        }
        if scene_check() {
            if let Err(e) = scene.check(&self.scene_rows(world, space)) {
                panic!("ENGINE_SCENE_CHECK: the GPU scene differs from a rebuild: {e}");
            }
        }
        let rows = scene.ids.len();
        self.gpu_scene = Some(scene);
        Some(rows)
    }

    /// Every opaque single mesh of `space` as a GPU scene row (a mesh a
    /// room hides draws no instances).
    pub(super) fn scene_rows(&self, world: &World, space: SpaceId) -> Vec<gpu_scene::SceneRow> {
        self.source_opaque
            .iter()
            .enumerate()
            .filter_map(|(order, &id)| {
                self.scene_row(
                    world,
                    space,
                    id,
                    u32::try_from(order).expect("scene row exceeds u32"),
                )
            })
            .collect()
    }

    /// Entity `id`'s GPU scene row (`order`: its place among the opaque
    /// meshes), or `None` if it has none.
    fn scene_row(
        &self,
        world: &World,
        space: SpaceId,
        id: EntityId,
        order: u32,
    ) -> Option<gpu_scene::SceneRow> {
        let table = self.bindless.as_ref().map(|b| &b.table);
        let entity = world.entity(id).ok()?;
        if entity.space() != space {
            return None;
        }
        let gpu = self.gpu_meshes.get(&id)?;
        let aabb = gpu.world_box?;
        if gpu.instance_count == 0 {
            return None;
        }
        let table_bind = gpu
            .material
            .filter(|&m| Some(m) == self.material_slot(entity.albedo))
            .and_then(|slot| table.and_then(|t| t.bind_for(slot)));
        let bind = table_bind.unwrap_or_else(|| {
            entity
                .albedo
                .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                .unwrap_or(&self.pipelines.white_albedo)
        });
        let instance_count = if world.room_hidden(id) {
            0
        } else {
            u32::try_from(gpu.instance_count).expect("instance count exceeds u32")
        };
        Some(gpu_scene::SceneRow {
            id,
            order,
            key: gpu_scene::GroupKey {
                bindless: table_bind.is_some(),
                bind: bind.clone(),
                vertex: gpu.vertex_buf.clone(),
                index: gpu.index_buf.clone(),
                instances: gpu.instance_buf.clone(),
            },
            mesh: occlusion::CandidateMesh {
                id,
                aabb,
                opaque_index_count: u32::try_from(gpu.opaque_index_count)
                    .expect("index count exceeds u32"),
                index_count: u32::try_from(gpu.index_count).expect("index count exceeds u32"),
                instance_count,
                first_index: gpu.first_index,
                base_vertex: gpu.base_vertex,
                first_instance: gpu.first_instance,
            },
        })
    }

    /// The GPU scene's groups as depth-prepass multi-draws over `region`.
    pub(super) fn scene_prepass(
        &self,
        main_cull: &MainCullSelection,
        region: occlusion::ArgsRegion,
    ) -> Option<MultiDraw> {
        let scene = self.gpu_scene.as_ref()?;
        let frame = main_cull.occlusion.as_ref()?;
        Some(MultiDraw {
            pipeline: Some(self.pipelines.depth_prepass.clone()),
            bindless_pipeline: self
                .bindless
                .as_ref()
                .map(|b| b.pipes.depth_prepass.clone()),
            groups: scene_groups(scene, frame, region).collect(),
            occlusion_args: Some(frame.args.clone()),
            draw_counts: frame.draw_counts.clone(),
            ..MultiDraw::default()
        })
    }

    /// This frame's occlusion candidates: single meshes the main view draws.
    /// The meshes the occlusion test runs for, ordered by what a
    /// multi-draw shares (texture, geometry page, instance page) so each
    /// such group's packed draws sit side by side.
    pub(super) fn occlusion_candidates(&self, world: &World, space: SpaceId) -> Vec<EntityId> {
        let mut ids: Vec<EntityId> = self
            .draw_opaque
            .iter()
            .copied()
            .filter(|&id| {
                entity_in_space(world, id, space)
                    && self
                        .gpu_meshes
                        .get(&id)
                        .is_some_and(|gpu| gpu.world_box.is_some() && gpu.instance_count > 0)
            })
            .collect();
        if self.multi_draw {
            ids.sort_by_cached_key(|id| {
                let gpu = &self.gpu_meshes[id];
                let albedo = world
                    .entity(*id)
                    .ok()
                    .and_then(|e| e.albedo)
                    .filter(|t| self.gpu_mesh_albedo.contains_key(t));
                // With texture arrays the multi-draw groups ignore the
                // texture (only the sampler kind splits them).
                let texture = match (self.bindless.as_ref(), self.material_slot(albedo)) {
                    (Some(b), Some(slot)) => (0, b.table.repeats(slot) as u64),
                    _ => (1, albedo.map_or(0, |t| t.0)),
                };
                (texture, gpu.geometry.page, gpu.instance_page())
            });
        }
        ids
    }

    pub(super) fn begin_occlusion(&mut self, ids: &[EntityId]) -> occlusion::OcclusionFrame {
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
                    first_instance: gpu.first_instance,
                }
            })
            .collect();
        let counts = self.occlusion.counts();
        self.gpu_stats.occlusion_tested = counts.tested;
        self.gpu_stats.occlusion_hidden = counts.hidden;
        self.occlusion.begin(&self.device, &self.queue, &rows)
    }

    pub(super) fn select_main_culls(&mut self, frustum: &Frustum) -> MainCullSelection {
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

    pub(super) fn select_shadow_culls(
        &mut self,
        world: &World,
        light_frustums: &[Frustum],
    ) -> Vec<ShadowCullSelection> {
        if self.shadow_source_epoch != self.full_syncs {
            self.rebuild_shadow_sources(world);
            self.shadow_source_epoch = self.full_syncs;
        }
        let mut selections: Vec<ShadowCullSelection> = light_frustums
            .iter()
            .map(|_| ShadowCullSelection {
                draws: Vec::new(),
                batch_draws: Vec::new(),
                compact_batches: Vec::new(),
            })
            .collect();
        // With the GPU shadow scene the GPU picks each view's mesh casters.
        let mesh_sources = if self.shadow_scene.is_some() {
            &[][..]
        } else {
            &self.shadow_mesh_sources[..]
        };
        for (source, &id) in mesh_sources.iter().enumerate() {
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

    /// Whether a caster has not changed lately (the static shadow cache).
    pub(super) fn caster_settled(&self, id: EntityId) -> bool {
        self.moved_at
            .get(&id)
            .is_none_or(|&at| self.frame_no.saturating_sub(at) > STATIC_SHADOW_SETTLE)
    }

    /// Bring the GPU shadow scene up to date; the bounds of static casters
    /// that came, went or changed.
    pub(super) fn sync_shadow_scene(&mut self, world: &World) -> Vec<Bounds> {
        let Some(mut scene) = self.shadow_scene.take() else {
            return Vec::new();
        };
        let space = world.living_in();
        let stamp = shadow_scene::ShadowStamp {
            render: self.full_syncs,
            resource: self.materials_synced,
            space,
        };
        let mut rebuild = scene.stamp != Some(stamp) || world.rooms_culling();
        let mut changed = Vec::new();
        let removed = std::mem::take(&mut self.shadow_removed);
        let added = std::mem::take(&mut self.shadow_added);
        if !rebuild && (!removed.is_empty() || !added.is_empty()) {
            // Casters that came or went: drop and add just their rows.
            let rows: Vec<shadow_scene::ShadowRow> = added
                .iter()
                .filter(|&&id| !scene.has(id))
                .filter_map(|&id| self.shadow_row(world, space, id, 0))
                .collect();
            if scene.worn() || !scene.fits(rows.len()) {
                rebuild = true;
            } else {
                for id in removed {
                    scene.remove(&self.queue, id, &mut changed);
                }
                for row in rows {
                    scene.add(&self.queue, row, &mut changed);
                }
            }
        }
        if !rebuild {
            // Only moves and new instance lists: patch those rows (see
            // `sync_gpu_scene`).
            let mut touched = std::mem::take(&mut self.shadow_touched);
            touched.sort_unstable();
            touched.dedup();
            for &id in &touched {
                let patched = match self.shadow_row(world, space, id, 0) {
                    Some(row) => scene.patch(&self.queue, &row, &mut changed),
                    None if scene.has(id) => Err(()),
                    None => Ok(()),
                };
                if patched.is_err() {
                    rebuild = true;
                    break;
                }
            }
        }
        if rebuild {
            self.shadow_touched.clear();
            let rows = self.shadow_rows(world, space);
            scene.stamp = Some(stamp);
            changed = scene.apply(&self.device, &self.queue, rows);
        } else {
            // Only casters whose last change has just settled can turn
            // static; one changed since is turned moving by its patch.
            let mut due = Vec::new();
            while let Some(&(at, id)) = self.settling.front() {
                if self.frame_no.saturating_sub(at) <= STATIC_SHADOW_SETTLE {
                    break;
                }
                self.settling.pop_front();
                if self.moved_at.get(&id) == Some(&at) {
                    due.push(id);
                }
            }
            changed.extend(scene.refresh_fixed(&self.queue, &due, |id| {
                self.caster_settled(id) && self.gpu_meshes.get(&id).is_some_and(|g| !g.sways)
            }));
        }
        if scene_check() {
            if let Err(e) = scene.check(&self.shadow_rows(world, space)) {
                panic!("ENGINE_SCENE_CHECK: the GPU shadow scene differs from a rebuild: {e}");
            }
        }
        self.shadow_scene = Some(scene);
        changed
    }

    /// Every shadow-casting mesh of `space` as a GPU shadow row.
    pub(super) fn shadow_rows(
        &self,
        world: &World,
        space: SpaceId,
    ) -> Vec<shadow_scene::ShadowRow> {
        self.shadow_mesh_sources
            .iter()
            .enumerate()
            .filter_map(|(order, &id)| {
                let order = u32::try_from(order).expect("shadow source exceeds u32");
                self.shadow_row(world, space, id, order)
            })
            .collect()
    }

    /// Caster `id`'s GPU shadow row (`order`: its place among the shadow
    /// sources), or `None` if it has none.
    fn shadow_row(
        &self,
        world: &World,
        space: SpaceId,
        id: EntityId,
        order: u32,
    ) -> Option<shadow_scene::ShadowRow> {
        let entity = world.entity(id).ok()?;
        if entity.space() != space {
            return None;
        }
        let gpu = self.gpu_meshes.get(&id)?;
        let bounds = gpu.bounds?;
        let albedo = entity.albedo.and_then(|tid| self.cutout_albedo.get(&tid));
        Some(shadow_scene::ShadowRow {
            id,
            order,
            key: shadow_scene::ShadowKey {
                cutout: albedo.is_some(),
                albedo: albedo.cloned(),
                vertex: gpu.vertex_buf.clone(),
                index: gpu.index_buf.clone(),
                instances: gpu.instance_buf.clone(),
            },
            bounds,
            args: DrawIndexedArgs {
                index_count: u32::try_from(gpu.opaque_index_count)
                    .expect("index count exceeds u32"),
                instance_count: u32::try_from(gpu.instance_count)
                    .expect("instance count exceeds u32"),
                first_index: gpu.first_index,
                base_vertex: gpu.base_vertex,
                first_instance: gpu.first_instance,
            },
            visible: !world.room_hidden(id),
            fixed: self.caster_settled(id) && !gpu.sways,
        })
    }

    pub(super) fn rebuild_shadow_sources(&mut self, world: &World) {
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
            self.rebuild_shadow_batch_sources(world);
        }
    }

    /// The instance batches that cast shadows (their instances change
    /// without a full sync: see `World::batched_instances`).
    pub(super) fn rebuild_shadow_batch_sources(&mut self, world: &World) {
        self.shadow_batch_sources.clear();
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

    pub(super) fn dispatch_instance_cull(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frustum: &Frustum,
        ids: &[EntityId],
        view: CullView,
        hiz: Option<&wgpu::BindGroup>,
    ) {
        // Each batch once: its args are reset once per view.
        let mut seen = HashSet::with_capacity(ids.len());
        let jobs: Vec<CullJob<'_>> = ids
            .iter()
            .filter(|id| seen.insert(**id))
            .map(|id| {
                let batch = self
                    .gpu_instance_batches
                    .get(id)
                    .expect("instance batch cull list is synced");
                let gpu = &batch.gpu;
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
                CullJob {
                    gpu,
                    local,
                    draws,
                    active_ranges: &batch.active_ranges,
                }
            })
            .collect();
        if jobs.is_empty() {
            return;
        }
        let dispatches =
            self.instance_cull
                .dispatch(&self.device, &self.queue, encoder, frustum, &jobs, hiz);
        let job_count = u32::try_from(jobs.len()).expect("cull dispatch count exceeds u32");
        self.gpu_stats.compute_culls += job_count;
        self.gpu_stats.cull_views += 1;
        self.gpu_stats.cull_dispatches += dispatches;
        self.gpu_stats.boundary_batches += job_count;
        for job in jobs {
            for range in job.active_ranges {
                self.gpu_stats.cull_instances += u64::from(range.count);
                self.gpu_stats.cull_workgroups += u64::from(range.count).div_ceil(WORKGROUP as u64);
            }
        }
    }

    /// Size this frame's cull tables and give every batch its place in the
    /// shared compact pages and draw args. Runs before any bundle that
    /// draws a culled batch is recorded.
    pub(super) fn begin_instance_cull_frame(&mut self, world: &World) {
        if world.instance_submit() != InstanceSubmit::GpuIndirect {
            return;
        }
        let sun = if world.shadows().is_some() { 3 } else { 0 };
        // The main view culls twice with occlusion on: frustum, then depth.
        let main = if self.occlusion_active { 2 } else { 1 };
        let views = main + sun + self.shadow.lamps.frame_views.len();
        let ranges = self
            .gpu_instance_batches
            .values()
            .map(|batch| batch.active_ranges.len())
            .sum::<usize>();
        let mut batches: Vec<(EntityId, &mut GpuMesh)> = self
            .gpu_instance_batches
            .iter_mut()
            .map(|(id, batch)| (*id, &mut batch.gpu))
            .collect();
        let count = batches.len();
        let moved = self.instance_cull.place(&self.device, &mut batches);
        self.instance_cull
            .begin_frame(&self.device, views, count, ranges);
        if moved {
            self.invalidate_shadow_bundles();
        }
    }

    pub(super) fn prepare_terrain_bundle(&mut self, world: &World) {
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

    pub(super) fn collect_draws(&mut self, world: &World, space: SpaceId, frustum_cull: bool) {
        if self.draw_source_epoch != self.full_syncs {
            self.rebuild_draw_sources(world);
            self.draw_source_epoch = self.full_syncs;
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
        self.draw_custom.clear();
        for &draw in &self.source_custom {
            let (id, gpu) = match draw {
                TransparentDraw::Mesh(id) => (id, self.gpu_meshes.get(&id)),
                TransparentDraw::Batch(id) => {
                    (id, self.gpu_instance_batches.get(&id).map(|b| &b.gpu))
                }
            };
            if entity_in_space(world, id, space) && gpu.is_some_and(visible) {
                self.draw_custom.push(draw);
            }
        }
        self.sort_transparent();
    }

    /// See-through draws far to near, so each blends over what is behind it.
    fn sort_transparent(&mut self) {
        use super::{back_to_front, TransparentDraw};
        let centre = |gpu: &GpuMesh| gpu.bounds.map_or(glam::Vec3::ZERO, |b| b.centre);
        let mut keyed: Vec<(glam::Vec3, TransparentDraw)> = self
            .draw_transparent
            .iter()
            .map(|&id| (centre(&self.gpu_meshes[&id]), TransparentDraw::Mesh(id)))
            .chain(self.batch_transparent.iter().map(|&id| {
                (
                    centre(&self.gpu_instance_batches[&id].gpu),
                    TransparentDraw::Batch(id),
                )
            }))
            .collect();
        back_to_front(self.view_eye, &mut keyed);
        self.transparent_order.clear();
        self.transparent_order
            .extend(keyed.into_iter().map(|(_, d)| d));
    }

    /// Rebuild material/pass membership only when static world state changes.
    pub(super) fn rebuild_draw_sources(&mut self, world: &World) {
        self.source_opaque.clear();
        self.source_custom.clear();
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
                Some(SurfaceMaterialRef::Custom(mid)) => {
                    if world.custom_material(mid).is_some_and(|m| m.see_through) {
                        self.source_transparent.push(id);
                    } else {
                        self.source_custom.push(TransparentDraw::Mesh(id));
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
                    Some(SurfaceMaterialRef::Custom(mid)) => {
                        if world.custom_material(mid).is_some_and(|m| m.see_through) {
                            self.batch_source_transparent.push(prototype);
                        } else {
                            self.source_custom.push(TransparentDraw::Batch(prototype));
                        }
                    }
                }
            }
        }
    }

    pub(super) fn sync_textures_and_materials(&mut self, world: &World) {
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
            let (gpu_material, repeat) = if let Some(maps) = world.surface_maps(tid) {
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
                let material = material::GpuMaterial::new(
                    &self.device,
                    &self.queue,
                    &p.albedo_layout,
                    &p.material_sampler,
                    &p.neutral_maps,
                    Some(&albedo),
                    Some(maps),
                );
                (material, true)
            } else {
                let gpu = self
                    .gpu_textures
                    .get(&tid)
                    .expect("mesh albedo missing on GPU");
                let material = material::GpuMaterial::new(
                    &self.device,
                    &self.queue,
                    &p.albedo_layout,
                    &p.albedo_sampler,
                    &p.neutral_maps,
                    Some(&gpu.view),
                    None,
                );
                (material, false)
            };
            if let Some(b) = self.bindless.as_mut() {
                b.table
                    .insert(&self.queue, Some(tid), &gpu_material, repeat);
            }
            let bind = gpu_material.bind;
            if has_cutout_texels(&world.textures()[&tid].rgba) {
                self.cutout_albedo.insert(tid, bind.clone());
            } else {
                self.cutout_albedo.remove(&tid);
            }
            self.gpu_mesh_albedo.insert(tid, bind);
        }
        self.gpu_mesh_albedo
            .retain(|id, _| world.textures().contains_key(id));
        self.cutout_albedo
            .retain(|id, _| self.gpu_mesh_albedo.contains_key(id));
        if let Some(b) = self.bindless.as_mut() {
            b.table.retain(|id| world.textures().contains_key(&id));
            let p = &self.pipelines;
            b.table.flush(
                &self.device,
                &p.neutral_maps,
                [&p.albedo_sampler, &p.material_sampler],
            );
        }

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
