//! Drawing one space (the main world or a portal pocket) into the scene pass.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
    pub(super) fn draw_space_scene(
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
                    && if f.multi {
                        self.opaque_world.multi.pipeline.as_ref() == Some(pipeline)
                    } else {
                        self.opaque_world
                            .shared
                            .as_ref()
                            .is_some_and(|(_, p, _)| p == pipeline)
                    }
            })
            .map(|f| (f.chunks, f.indirect_draws, f.multi));
        if let Some((chunks, indirect_draws, multi)) = bundles {
            if multi {
                let scene_bind = self.pipelines.scene_bind_group(uniform_slot);
                self.gpu_stats.multi_draws += self.opaque_world.multi.encode(pass, scene_bind);
            } else {
                pass.execute_bundles(self.opaque_world.chunks[..chunks].iter().map(|c| &c.bundle));
            }
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
            // Off screen in the top-level view (portal views look elsewhere).
            if stamp && !self.frustum.intersects(gpu.bounds) {
                continue;
            }
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

        // Solid meshes with a game's custom material: each with its own
        // pipeline and params, the whole mesh (not in portal views yet).
        if !portal_mask {
            for i in 0..self.draw_custom.len() {
                let draw = self.draw_custom[i];
                let (TransparentDraw::Mesh(id) | TransparentDraw::Batch(id)) = draw;
                let entity = world.entity(id).expect("draw list is live");
                let Some(SurfaceMaterialRef::Custom(mid)) = entity.material else {
                    continue;
                };
                let Some(custom) = self.gpu_custom.get(&mid) else {
                    continue;
                };
                let albedo = entity
                    .albedo
                    .and_then(|tid| self.gpu_mesh_albedo.get(&tid))
                    .unwrap_or(&self.pipelines.white_albedo);
                pass.set_pipeline(&custom.opaque);
                pass.set_bind_group(1, albedo, &[]);
                pass.set_bind_group(2, &custom.bind, &[]);
                match draw {
                    TransparentDraw::Mesh(_) => {
                        let gpu = self.gpu_meshes.get(&id).expect("draw list is synced");
                        submit_mesh_draw(
                            pass,
                            gpu,
                            MeshSubmit::Direct,
                            0..gpu.index_count as u32,
                            0,
                            &mut self.gpu_stats.indirect_draws,
                        );
                    }
                    TransparentDraw::Batch(_) => {
                        let batch = self
                            .gpu_instance_batches
                            .get(&id)
                            .expect("batch draw is synced");
                        submit_batch_draw(
                            pass,
                            batch,
                            None,
                            0..batch.gpu.index_count as u32,
                            0,
                            &mut self.gpu_stats.indirect_draws,
                        );
                    }
                }
            }
        }
    }

    /// The see-through part of a space's scene: glass, see-through custom
    /// materials, water, ribbons. Drawn after the views through this
    /// level's portals, so a portal's rim or a pane in front of a doorway
    /// shows over the view instead of under it.
    pub(super) fn draw_space_see_through(
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
        let stamp = !portal_mask && use_cull;
        pass.set_bind_group(0, self.pipelines.scene_bind_group(uniform_slot), &[]);
        let transparent_pipeline = if portal_mask {
            &self.pipelines.transparent_portal
        } else {
            &self.pipelines.transparent
        };
        pass.set_pipeline(transparent_pipeline);
        // Far to near, single meshes and instanced batches together.
        for i in 0..self.transparent_order.len() {
            match self.transparent_order[i] {
                TransparentDraw::Mesh(id) => {
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
                    if let Some(SurfaceMaterialRef::Custom(mid)) = entity.material {
                        // A see-through custom material: its own pipeline, whole mesh.
                        if let (false, Some(custom)) = (portal_mask, self.gpu_custom.get(&mid)) {
                            pass.set_pipeline(&custom.blend);
                            pass.set_bind_group(2, &custom.bind, &[]);
                            submit_mesh_draw(
                                pass,
                                gpu,
                                MeshSubmit::Direct,
                                0..gpu.index_count as u32,
                                0,
                                &mut self.gpu_stats.indirect_draws,
                            );
                            pass.set_pipeline(transparent_pipeline);
                        }
                        continue;
                    }
                    if let Some((args, cslot)) =
                        stamp.then(|| main_cull.occlusion_slot(id)).flatten()
                    {
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
                TransparentDraw::Batch(prototype) => {
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
                    if let Some(SurfaceMaterialRef::Custom(mid)) = entity.material {
                        if let (false, Some(custom)) = (portal_mask, self.gpu_custom.get(&mid)) {
                            pass.set_pipeline(&custom.blend);
                            pass.set_bind_group(2, &custom.bind, &[]);
                            submit_batch_draw(
                                pass,
                                batch,
                                None,
                                0..gpu.index_count as u32,
                                0,
                                &mut self.gpu_stats.indirect_draws,
                            );
                            pass.set_pipeline(transparent_pipeline);
                        }
                        continue;
                    }
                    submit_batch_draw(
                        pass,
                        batch,
                        use_cull.then(|| mesh_submit(&main_cull.compacted, prototype)),
                        gpu.opaque_index_count as u32..gpu.index_count as u32,
                        TRANSLUCENT_INDIRECT_OFFSET,
                        &mut self.gpu_stats.indirect_draws,
                    );
                }
            }
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
}
