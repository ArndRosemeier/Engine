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
        let saved_eye = self.view_eye;
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
                let view = PortalLevelView {
                    clip_plane: None,
                    draw_environment: false,
                    use_cull: false,
                };
                let scene = ScenePass {
                    portal_mask: false,
                    uniform_slot: slot,
                };
                self.draw_space_scene(&mut pass, world, live, &no_cull, view, scene);
                self.draw_space_see_through(&mut pass, world, live, &no_cull, view, scene);
            }
            self.reflect.store_face(&self.device, encoder, job);
        }
        world.set_room_bypass(false);
        self.view_aspect = None;
        self.taa_jitter = saved_jitter;
        self.frustum = saved_frustum;
        self.view_eye = saved_eye;
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

        let view = PortalLevelView {
            clip_plane,
            draw_environment,
            use_cull,
        };
        let scene = ScenePass {
            portal_mask: level > 0,
            uniform_slot,
        };
        self.draw_space_scene(pass, world, space, main_cull, view, scene);

        // Looking through portals needs the stencil to keep the far space
        // inside the doorway. Without one (a GPU without Depth32FloatStencil8,
        // or ENGINE_DEPTH_FORMAT=d32) doorways show nothing beyond them.
        let looks_through = level < world.portal_recursion() && stencil::has_stencil();
        for visible in portals.into_iter().filter(|_| looks_through) {
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
            // Cull destination draws to what can be seen through the opening:
            // the virtual camera's normal frustum (oblique clipping is applied
            // when the pass draws; culling with it dropped whole yards at
            // doorways), narrowed to the opening's rectangle on the screen.
            // Drawing the whole space unculled cost a big city 4x its frame.
            self.frustum = self.through_frustum(&render_cam, &virt, visible.src);
            self.collect_draws(world, visible.dest_space, true);

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

        // See-through things last, over the views (a portal's rim, glass in
        // front of a doorway): drawn into them they were painted over. Each
        // view above put this level's draw lists back when it was done.
        self.draw_space_see_through(pass, world, space, main_cull, view, scene);
    }

    /// The frustum of `virt` (the camera through a portal) narrowed to the
    /// rectangle the opening `src` covers on the screen of `render_cam`
    /// (both share the screen). The whole frustum when the opening is not
    /// all in front of the camera.
    fn through_frustum(&self, render_cam: &Camera, virt: &Camera, src: EntityId) -> Frustum {
        let aspect = self
            .view_aspect
            .unwrap_or(self.config.width as f32 / self.config.height.max(1) as f32);
        let vp_virt = virt.view_projection(aspect);
        let whole = Frustum::from_view_projection(vp_virt);
        let Some(b) = self.gpu_meshes.get(&src).and_then(|g| g.bounds) else {
            return whole;
        };
        let Some(rect) = screen_rect(render_cam.view_projection(aspect), b) else {
            return whole;
        };
        Frustum::from_view_projection(rect_matrix(rect) * vp_virt)
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
        let epoch = self
            .scene_gen
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
        if self.draw_source_epoch != self.full_syncs {
            self.rebuild_draw_sources(world);
            self.draw_source_epoch = self.full_syncs;
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

/// The rectangle (NDC: x0, y0, x1, y1) that `bounds` covers on the screen
/// of `vp`, inside the screen; None when part of it is behind the camera or
/// it covers nothing.
fn screen_rect(vp: glam::Mat4, b: Bounds) -> Option<[f32; 4]> {
    let (lo, hi) = (
        b.centre - glam::Vec3::splat(b.radius),
        b.centre + glam::Vec3::splat(b.radius),
    );
    let mut r = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for i in 0..8 {
        let p = glam::Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let c = vp * p.extend(1.0);
        if c.w <= 1e-4 {
            return None;
        }
        let (x, y) = (c.x / c.w, c.y / c.w);
        r = [r[0].min(x), r[1].min(y), r[2].max(x), r[3].max(y)];
    }
    let r = [r[0].max(-1.0), r[1].max(-1.0), r[2].min(1.0), r[3].min(1.0)];
    (r[2] > r[0] && r[3] > r[1]).then_some(r)
}

/// Maps the NDC rectangle `r` onto the whole screen (applied after a
/// view-projection: its frustum is the rectangle's).
fn rect_matrix(r: [f32; 4]) -> glam::Mat4 {
    let (cx, cy) = ((r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5);
    let (hx, hy) = ((r[2] - r[0]) * 0.5, (r[3] - r[1]) * 0.5);
    glam::Mat4::from_cols(
        glam::Vec4::new(1.0 / hx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 1.0 / hy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(-cx / hx, -cy / hy, 0.0, 1.0),
    )
}

#[cfg(test)]
mod through_tests {
    use super::*;

    #[test]
    fn a_doorway_view_is_culled_to_the_doorway() {
        let cam = Camera::look_at((0.0, 1.6, -10.0), (0.0, 1.6, 0.0));
        let vp = cam.view_projection(16.0 / 9.0);
        // A 2 x 3 m doorway 10 m ahead covers a small part of the screen.
        let door = Bounds {
            centre: glam::Vec3::new(0.0, 1.5, 0.0),
            radius: 1.8,
        };
        let r = screen_rect(vp, door).expect("in front");
        assert!(r[2] - r[0] < 0.6 && r[3] - r[1] < 0.9, "{r:?}");
        let f = Frustum::from_view_projection(rect_matrix(r) * vp);
        // Straight through it: seen. Off to the side, though in the whole
        // frustum: not.
        let ahead = Bounds {
            centre: glam::Vec3::new(0.0, 1.5, 30.0),
            radius: 0.5,
        };
        let aside = Bounds {
            centre: glam::Vec3::new(14.0, 1.5, 30.0),
            radius: 0.5,
        };
        assert!(Frustum::from_view_projection(vp).intersects(aside));
        assert!(f.intersects(ahead));
        assert!(!f.intersects(aside));
        // Behind the camera: no narrowing.
        let behind = Bounds {
            centre: glam::Vec3::new(0.0, 1.5, -12.0),
            radius: 1.0,
        };
        assert!(screen_rect(vp, behind).is_none());
    }
}
