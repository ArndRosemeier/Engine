//! Reflection probes, portals (stencil and recursive views), GI voxelization.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
    /// Draw this frame's reflection probe faces (see `reflect.rs`).
    pub(super) fn capture_probes(&mut self, encoder: &mut wgpu::CommandEncoder, world: &World) {
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

    pub(super) fn render_portal_level<'a>(
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

        // Looking through portals needs the stencil to keep the far space
        // inside the doorway. Without one (a GPU without Depth32FloatStencil8,
        // or ENGINE_DEPTH_FORMAT=d32) doorways show nothing beyond them.
        if level >= world.portal_recursion() || !stencil::has_stencil() {
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

    pub(super) fn portal_in_frustum(&self, world: &World, id: EntityId, eye: glam::Vec3) -> bool {
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

    pub(super) fn draw_portal_stencil_incr<'a>(
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

    pub(super) fn draw_portal_stencil_decr<'a>(
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

    pub(super) fn draw_portal_mesh(&mut self, pass: &mut wgpu::RenderPass<'_>, id: EntityId) {
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
    pub(super) fn encode_gi(
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
    pub(super) fn voxelize_scene(
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
}
