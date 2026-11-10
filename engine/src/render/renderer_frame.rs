//! One frame: render_with, screenshots, scene uniforms and the world pass encoder.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
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
    pub(super) fn settle_pipeline_cache(&mut self) {
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
    pub(super) fn mark_scene(&self, pass: &mut wgpu::RenderPass<'_>, on: bool, mark: SceneMark) {
        if let (true, Some(profiler)) = (on, self.gpu_profiler.as_ref()) {
            profiler.scene_mark(pass, mark);
        }
    }

    pub(super) fn mark_gpu(&self, encoder: &mut wgpu::CommandEncoder, mark: Mark) {
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
        let Some(src) = self.last_color.as_ref() else {
            return Err(crate::EngineError::Gpu(
                if self.config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
                    "no frame to capture yet (request_capture comes before render_with)"
                } else {
                    "screenshots are off: this window surface cannot be copied"
                }
                .into(),
            ));
        };
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
            if tx.send(result).is_err() {}
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

    pub(super) fn store_composited(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
    ) {
        if !self.config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            return;
        }
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

    pub(super) fn write_scene_uniforms(
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

    pub(super) fn write_shadows(&mut self, world: &World) {
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

    pub(super) fn main_scene_camera(world: &World) -> Camera {
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

    pub(super) fn encode_world(
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
        let scene_rows = self.sync_gpu_scene(world, live);
        let occlusion_ids = if let Some(rows) = scene_rows {
            // The GPU scene: phase 1 draws what was seen last frame; without
            // occlusion culling the view test alone decides, right away.
            let slots = self.gpu_scene.as_ref().expect("synced above").slots.clone();
            main_cull.occlusion = Some(self.occlusion.scene_frame(slots));
            let vp = self.occlusion_view_proj;
            let (device, queue) = (&self.device, &self.queue);
            if self.occlusion_active {
                let counts = self.occlusion.counts();
                self.gpu_stats.occlusion_tested = counts.tested;
                self.gpu_stats.occlusion_hidden = counts.hidden;
                self.occlusion.encode_seed(device, queue, encoder, vp, rows);
            } else {
                self.occlusion
                    .encode_scene(device, queue, encoder, vp, rows, false);
            }
            Vec::new()
        } else if self.occlusion_active {
            let ids = self.occlusion_candidates(world, live);
            main_cull.occlusion = Some(self.begin_occlusion(&ids));
            ids
        } else {
            Vec::new()
        };

        self.cpu_spans.mark("world.cull", &mut t);
        // Finer spans inside world.prepass_ao (its own clock).
        let mut tp = std::time::Instant::now();
        let depth = self.depth_view.clone();
        let occluding = self.occlusion_active;
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
            self.cpu_spans.mark("prepass.first", &mut tp);
        }
        if occluding {
            match scene_rows {
                Some(rows) => self.occlusion.encode_scene(
                    &self.device,
                    &self.queue,
                    encoder,
                    self.occlusion_view_proj,
                    rows,
                    true,
                ),
                None => self.occlusion.encode(
                    &self.device,
                    &self.queue,
                    encoder,
                    self.occlusion_view_proj,
                    &occlusion_ids,
                ),
            }
            self.cpu_spans.mark("prepass.hiz_cull", &mut tp);
            self.encode_depth_prepass(
                encoder,
                world,
                live,
                &main_cull,
                main_uniform_slot,
                PrepassPhase::Unseen,
            );
            self.cpu_spans.mark("prepass.unseen", &mut tp);
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
                self.cpu_spans.mark("prepass.instances", &mut tp);
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
            self.cpu_spans.mark("prepass.ssao", &mut tp);
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
}
