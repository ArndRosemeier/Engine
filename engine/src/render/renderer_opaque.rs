//! Opaque mesh draws: bundles, multi-draw planning and the depth prepass.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
    /// Get the top-level view's opaque mesh draws for one pass ready as
    /// render bundles, in chunks. The draws are exactly the ones that pass
    /// would make itself (same list, order, culling and occlusion choices).
    /// Chunks whose draws are unchanged are kept; the others are recorded on
    /// the frame's threads, reading only renderer state that nothing changes
    /// meanwhile. The chunks stay in draw order and are executed in that
    /// order, so the result matches drawing one by one. With too few draws
    /// for bundles to pay off there are none, and the pass draws directly.
    pub(super) fn prepare_opaque_bundles(
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
        // The GPU scene's rows go out as whole groups of its draw regions.
        let scene_region = self
            .gpu_scene
            .as_ref()
            .is_some_and(|s| !s.groups.is_empty() && main_cull.occlusion.is_some())
            .then_some(target)
            .and_then(|target| match target {
                OpaqueTarget::Prepass(PrepassPhase::Seen) => Some(occlusion::ArgsRegion::Phase1),
                OpaqueTarget::Prepass(PrepassPhase::All) | OpaqueTarget::World => {
                    Some(occlusion::ArgsRegion::Shown)
                }
                OpaqueTarget::Prepass(_) => None,
            });
        let mut draws: Vec<OpaqueDraw<'_>> = Vec::new();
        for &id in &self.draw_opaque {
            if !entity_in_space(world, id, space) {
                continue;
            }
            let occlusion = main_cull.occlusion_slot(id);
            if scene_region.is_some() && occlusion.is_some() {
                continue;
            }
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
            let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
            draws.push(OpaqueDraw {
                albedo: albedo_of(entity),
                material: gpu
                    .material
                    .filter(|&m| Some(m) == self.material_slot(entity.albedo)),
                gpu,
                submit,
                packed: None,
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
                material: None,
                gpu: &batch.gpu,
                submit: OpaqueSubmit::Mesh(mesh_submit(&main_cull.compacted, prototype)),
                packed: main_cull
                    .pack
                    .as_ref()
                    .filter(|_| main_cull.compacted.contains(&prototype))
                    .and_then(|p| p.rows.get(&prototype).copied()),
            });
        }
        let cache = match target {
            OpaqueTarget::Prepass(_) => &self.opaque_prepass,
            OpaqueTarget::World => &self.opaque_world,
        };
        let use_multi =
            self.multi_draw && (draws.len() >= OPAQUE_MULTI_MIN_DRAWS || scene_region.is_some());
        if draws.len() < OPAQUE_BUNDLE_MIN_DRAWS && !use_multi {
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
        if use_multi {
            let plan = self.plan_multi_draw(
                &draws,
                main_cull,
                scene_region,
                scene_bind,
                pipeline,
                world_pass,
                samples,
            );
            let pipeline = pipeline.clone();
            drop(draws);
            let cache = match target {
                OpaqueTarget::Prepass(_) => &mut self.opaque_prepass,
                OpaqueTarget::World => &mut self.opaque_world,
            };
            // Bundled chunks are not kept alongside.
            cache.chunks.clear();
            cache.shared = None;
            let multi = &mut cache.multi;
            if multi.written != plan.args {
                let bytes =
                    (plan.args.len().max(1) * std::mem::size_of::<DrawIndexedArgs>()) as u64;
                if multi.args.as_ref().is_none_or(|b| b.size() < bytes) {
                    multi.args = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("opaque-multi-draw-args"),
                        size: bytes.next_power_of_two(),
                        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                }
                if !plan.args.is_empty() {
                    self.queue.write_buffer(
                        multi.args.as_ref().expect("made above"),
                        0,
                        bytemuck::cast_slice(&plan.args),
                    );
                }
                multi.written = plan.args;
            }
            multi.pipeline = Some(pipeline);
            multi.bindless_pipeline = self.bindless.as_ref().map(|b| {
                match target {
                    OpaqueTarget::Prepass(_) => &b.pipes.depth_prepass,
                    OpaqueTarget::World if nowrite => &b.pipes.opaque_nowrite,
                    OpaqueTarget::World => &b.pipes.opaque,
                }
                .clone()
            });
            multi.groups = plan.groups;
            multi.occlusion_args = plan.occlusion_args;
            multi.culled_args = plan.culled_args;
            multi.draw_counts = plan.draw_counts;
            multi.rest = plan.rest;
            cache.frame = Some(OpaqueFrame {
                target,
                space,
                uniform_slot,
                chunks: 0,
                indirect_draws: plan.indirect_draws,
                multi: true,
            });
            return;
        }
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
            multi: false,
        });
        cache.multi = MultiDraw::default();
    }

    /// Sort a pass's opaque draws into multi-draws: groups by texture and
    /// buffers in order of first appearance, each group's direct draws in
    /// one call and its occlusion-tested draws in runs of their packed rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_multi_draw(
        &self,
        draws: &[OpaqueDraw<'_>],
        main_cull: &MainCullSelection,
        scene_region: Option<occlusion::ArgsRegion>,
        scene_bind: &wgpu::BindGroup,
        pipeline: &wgpu::RenderPipeline,
        world_pass: bool,
        samples: u32,
    ) -> MultiPlan {
        type Key = (
            wgpu::BindGroup,
            bool,
            wgpu::Buffer,
            wgpu::Buffer,
            wgpu::Buffer,
        );
        let table = self.bindless.as_ref().map(|b| &b.table);
        // wgpu handles hash by their id, which never changes.
        #[allow(clippy::mutable_key_type)]
        let mut index: HashMap<Key, usize> = HashMap::new();
        let mut keys: Vec<Key> = Vec::new();
        let mut direct: Vec<Vec<DrawIndexedArgs>> = Vec::new();
        let mut occluded: Vec<Vec<u32>> = Vec::new();
        let mut rest: Vec<&OpaqueDraw<'_>> = Vec::new();
        // Compacted batches with a packed row: grouped like the rest, each
        // group's rows in runs (the cull packed them in this order).
        #[allow(clippy::mutable_key_type)]
        let mut culled: HashMap<Key, Vec<u32>> = HashMap::new();
        let mut culled_keys: Vec<Key> = Vec::new();
        for d in draws {
            if matches!(d.submit, OpaqueSubmit::Mesh(MeshSubmit::Compacted)) {
                match (d.packed, d.gpu.cull_target.as_ref()) {
                    (Some(row), Some(target)) if main_cull.pack.is_some() => {
                        let key: Key = (
                            d.albedo.clone(),
                            false,
                            d.gpu.vertex_buf.clone(),
                            d.gpu.index_buf.clone(),
                            target.compact_at().0.clone(),
                        );
                        culled
                            .entry(key.clone())
                            .or_insert_with(|| {
                                culled_keys.push(key);
                                Vec::new()
                            })
                            .push(row);
                    }
                    _ => rest.push(d),
                }
                continue;
            }
            let g = d.gpu;
            // Texture arrays: every material of a sampler kind in one group.
            let table_bind = d
                .material
                .and_then(|slot| table.and_then(|t| t.bind_for(slot)));
            let key: Key = (
                table_bind.unwrap_or(d.albedo).clone(),
                table_bind.is_some(),
                g.vertex_buf.clone(),
                g.index_buf.clone(),
                g.instance_buf.clone(),
            );
            let at = *index.entry(key.clone()).or_insert_with(|| {
                keys.push(key);
                direct.push(Vec::new());
                occluded.push(Vec::new());
                keys.len() - 1
            });
            match d.submit {
                OpaqueSubmit::Occluded(_, slot) => occluded[at].push(slot),
                _ => direct[at].push(DrawIndexedArgs {
                    index_count: g.opaque_index_count as u32,
                    instance_count: g.instance_count as u32,
                    first_index: g.first_index,
                    base_vertex: g.base_vertex,
                    first_instance: g.first_instance,
                }),
            }
        }
        let occlusion = main_cull.occlusion.as_ref();
        let stride = std::mem::size_of::<DrawIndexedArgs>() as u64;
        let mut args = Vec::new();
        let mut groups = Vec::new();
        for (((albedo, bindless, vertex, index, instances), own), mut slots) in
            keys.into_iter().zip(direct).zip(occluded)
        {
            let group = |source, offset, count| MultiGroup {
                albedo: albedo.clone(),
                bindless,
                vertex: vertex.clone(),
                index: index.clone(),
                instances: instances.clone(),
                source,
                offset,
                count,
                draw_count: None,
            };
            if !own.is_empty() {
                groups.push(group(
                    GroupArgs::Own,
                    args.len() as u64 * stride,
                    own.len() as u32,
                ));
                args.extend(own);
            }
            if let Some(frame) = occlusion.filter(|_| !slots.is_empty()) {
                slots.sort_unstable();
                for run in slots.chunk_by(|a, b| b == &(a + 1)) {
                    groups.push(group(
                        GroupArgs::Occlusion,
                        frame.packed_offset(run[0]),
                        run.len() as u32,
                    ));
                }
            }
        }
        let mut culled_draws = 0u32;
        let mut culled_groups = Vec::new();
        for (albedo, bindless, vertex, index, instances) in culled_keys {
            let mut rows = culled
                .remove(&(
                    albedo.clone(),
                    bindless,
                    vertex.clone(),
                    index.clone(),
                    instances.clone(),
                ))
                .expect("every culled key has rows");
            rows.sort_unstable();
            for run in rows.chunk_by(|a, b| *b == a + 1) {
                culled_draws += 1;
                culled_groups.push(MultiGroup {
                    albedo: albedo.clone(),
                    bindless,
                    vertex: vertex.clone(),
                    index: index.clone(),
                    instances: instances.clone(),
                    source: GroupArgs::Culled,
                    offset: u64::from(run[0]) * stride,
                    count: run.len() as u32,
                    draw_count: None,
                });
            }
        }
        if let (Some(region), Some(scene), Some(frame)) =
            (scene_region, self.gpu_scene.as_ref(), occlusion)
        {
            groups.extend(scene_groups(scene, frame, region));
        }
        // Plain groups first, then the texture-array ones: one pipeline
        // switch.
        groups.sort_by_key(|g| g.bindless);
        // GPU-culled batches last, where their own draws used to go.
        groups.extend(culled_groups);
        let indirect_draws = rest.len() as u32 + culled_draws;
        let rest = (!rest.is_empty()).then(|| {
            let color = [Some(HDR_FORMAT)];
            let mut bundle =
                self.device
                    .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                        label: Some("opaque-multi-rest"),
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
            for d in &rest {
                d.record(&mut bundle);
            }
            bundle.finish(&wgpu::RenderBundleDescriptor {
                label: Some("opaque-multi-rest"),
            })
        });
        MultiPlan {
            groups,
            args,
            occlusion_args: occlusion.map(|f| f.args.clone()),
            culled_args: main_cull.pack.as_ref().map(|p| p.args.clone()),
            draw_counts: occlusion
                .filter(|_| scene_region.is_some())
                .and_then(|f| f.draw_counts.clone()),
            rest,
            indirect_draws,
        }
    }

    /// Lay the live view's opaque depth before shading, for SSAO and so the
    /// world pass shades each pixel once. Foliage cutouts and see-through
    /// surfaces stay out; they still draw, and test, in the world pass.
    pub(super) fn encode_depth_prepass(
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
            if let Some(multi) = self.scene_prepass(main_cull, occlusion::ArgsRegion::Phase2) {
                multi.encode(&mut pass, scene_bind);
                return;
            }
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
            if let Some(multi) = self.scene_prepass(main_cull, occlusion::ArgsRegion::Phase1) {
                multi.encode(&mut pass, scene_bind);
                return;
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
            if frame.multi {
                self.opaque_prepass.multi.encode(&mut pass, scene_bind);
            } else {
                pass.execute_bundles(
                    self.opaque_prepass.chunks[..frame.chunks]
                        .iter()
                        .map(|c| &c.bundle),
                );
            }
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
}
