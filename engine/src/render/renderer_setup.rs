//! Creating the renderer (GPU, device, pipelines), resizing and the present mode.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
    /// `msaa`: the game's [`crate::Settings::msaa`] (`ENGINE_MSAA` wins).
    pub async fn new(
        window: std::sync::Arc<winit::window::Window>,
        msaa: Option<u32>,
    ) -> EngineResult<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            // `WGPU_BACKEND=dx12|vulkan|metal` picks one (a driver test).
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            backend_options: dx12_options(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).map_err(|e| {
            EngineError::Window(format!(
                "could not create a drawing surface for the window ({e}). \
                 The engine needs a Vulkan, DirectX 12 or Metal driver: update \
                 the graphics driver. Without a GPU (CI, containers) install \
                 Mesa's software Vulkan driver (lavapipe, package \
                 mesa-vulkan-drivers); on Linux also check that a display (or \
                 Xvfb) is running"
            ))
        })?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                EngineError::Gpu(format!(
                    "no GPU that can draw to this window ({e}). The engine needs \
                     Vulkan, DirectX 12 or Metal: update the graphics driver. \
                     Without a GPU (CI, containers) install Mesa's software \
                     Vulkan driver (lavapipe, package mesa-vulkan-drivers)"
                ))
            })?;

        if adapter.get_info().backend == wgpu::Backend::Gl {
            return Err(EngineError::Gpu(
                "OpenGL is not supported (WGPU_BACKEND=gl?). The engine runs on Vulkan, \
                 DirectX 12 or Metal: unset WGPU_BACKEND or pick one of those"
                    .into(),
            ));
        }
        let mut gpu_profile_interval = GpuProfiler::requested_interval();
        let portal_depth = wgpu::Features::DEPTH32FLOAT_STENCIL8;
        // Portals and room culling need a stencil next to 32-bit depth. A GPU
        // without it still runs, with plain depth (portals draw without
        // their stencil mask).
        let stencil_ok = adapter.features().contains(portal_depth);
        stencil::choose_depth_format(stencil_ok);
        if !stencil_ok {
            eprintln!(
                "[engine] this GPU has no Depth32FloatStencil8; running with plain depth \
                 (portal masks off)"
            );
        }
        let base_features = if stencil::has_stencil() {
            portal_depth
        } else {
            wgpu::Features::empty()
        };
        let mut required_features = base_features | pipeline_cache::wanted_features(&adapter);
        // Many meshes per draw call (see `MultiDraw`), where the GPU can.
        let multi_draw_features =
            wgpu::Features::MULTI_DRAW_INDIRECT | wgpu::Features::INDIRECT_FIRST_INSTANCE;
        let mut multi_draw =
            adapter.features().contains(multi_draw_features) && multi_draw_env() != Some(false);
        if multi_draw {
            required_features |= multi_draw_features;
        }
        // The GPU scene's multi-draws read their draw counts from the GPU.
        let mut gpu_scene_on = multi_draw && !gpu_scene::disabled_by_env();
        let mut draw_count = gpu_scene_on
            && adapter
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
            && !gpu_scene::draw_count_disabled_by_env();
        if draw_count {
            required_features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        }
        let mut required_limits = wgpu::Limits::default();
        let mut texture_arrays = multi_draw
            && adapter
                .features()
                .contains(material_table::MATERIAL_TABLE_FEATURES)
            && adapter.limits().max_binding_array_elements_per_shader_stage
                >= material_table::MATERIAL_TABLE_ELEMENTS
            && !material_table::disabled_by_env();
        if texture_arrays {
            required_features |= material_table::MATERIAL_TABLE_FEATURES;
            required_limits.max_binding_array_elements_per_shader_stage =
                material_table::MATERIAL_TABLE_ELEMENTS;
        }
        // Bind only the table slots in use: wgpu checks every bound texture
        // each time a pass binds the table, so 16k fallback slots cost
        // about a millisecond of CPU per pass.
        let mut partial_table = texture_arrays
            && adapter
                .features()
                .contains(wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY)
            && !material_table::partial_disabled_by_env();
        if partial_table {
            required_features |= wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY;
        }
        if gpu_profile_interval.is_some() {
            let timestamp_features =
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            if !adapter.features().contains(timestamp_features) {
                eprintln!(
                    "[engine] ENGINE_GPU_PROFILE_EVERY: this GPU has no timestamp queries; \
                     GPU timings are off"
                );
                gpu_profile_interval = None;
            } else {
                required_features |= timestamp_features;
                // Splitting the scene pass needs stamps inside it; optional.
                if adapter
                    .features()
                    .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
                {
                    required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
                }
            }
        }
        let describe = |features: wgpu::Features, limits: &wgpu::Limits| wgpu::DeviceDescriptor {
            label: Some("engine-device"),
            required_features: features,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        };
        let (device, queue) = match adapter
            .request_device(&describe(required_features, &required_limits))
            .await
        {
            Ok(pair) => pair,
            Err(first) => {
                // The driver refused the extras: run with the plain feature
                // set (one draw call per mesh, textures bound per draw).
                eprintln!(
                    "[engine] the GPU refused the optional features ({first}); \
                     running without multi-draw, the GPU scene and texture arrays"
                );
                multi_draw = false;
                gpu_scene_on = false;
                draw_count = false;
                texture_arrays = false;
                partial_table = false;
                gpu_profile_interval = None;
                required_limits = wgpu::Limits::default();
                adapter
                    .request_device(&describe(base_features, &required_limits))
                    .await
                    .map_err(|e| {
                        EngineError::Gpu(format!(
                            "the GPU ({}) refused to start ({e}). Update the graphics driver",
                            adapter.get_info().name
                        ))
                    })?
            }
        };

        let hdr_features = adapter.get_texture_format_features(HDR_FORMAT);
        stencil::init_scene_samples(choose_scene_samples(&adapter, msaa));
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
            return Err(EngineError::Gpu(format!(
                "the GPU ({}) cannot draw to filterable, blendable Rgba16Float targets, \
                 which HDR lighting and bloom need. Update the graphics driver",
                adapter.get_info().name
            )));
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| {
                EngineError::Window(format!(
                    "the GPU ({}) cannot draw to this window (no surface formats)",
                    adapter.get_info().name
                ))
            })?;

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

        // Screenshots copy the finished frame out of the surface; a surface
        // that cannot be copied still runs, without screenshots.
        let surface_usage = if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            eprintln!("[engine] this window surface cannot be copied; screenshots are off");
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let config = wgpu::SurfaceConfiguration {
            usage: surface_usage,
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
        let shadow_scene = (draw_count && !shadow_scene::disabled_by_env())
            .then(|| shadow_scene::ShadowScene::new(&device));
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
        let bindless = texture_arrays.then(|| {
            let mut table = material_table::MaterialTable::new(&device, partial_table);
            let white = material::GpuMaterial::new(
                &device,
                &queue,
                &pipelines.albedo_layout,
                &pipelines.albedo_sampler,
                &pipelines.neutral_maps,
                None,
                None,
            );
            table.insert(&queue, None, &white, false);
            let pipes = pipeline::create_bindless_pipelines(
                &device,
                HDR_FORMAT,
                &pipelines.bind_layout,
                &table.layout,
            );
            Bindless { table, pipes }
        });
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
        Ok(Self {
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
            multi_draw,
            gpu_scene: gpu_scene_on.then(gpu_scene::GpuScene::default),
            shadow_scene,
            static_caster_gen: [0; 3],
            materials_synced: 0,
            bindless,
            gpu_instance_batches: HashMap::new(),
            gpu_batch_sources: Vec::new(),
            gpu_batch_source_scratch: Vec::new(),
            gpu_skinned: HashMap::new(),
            gpu_skinned_meshes: HashMap::new(),
            skinned_maps: material::MapCache::default(),
            gpu_textures: HashMap::new(),
            gpu_mesh_albedo: HashMap::new(),
            cutout_albedo: HashMap::new(),
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
            occlusion: {
                let mut o = occlusion;
                o.compact = draw_count;
                o
            },
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
        })
    }

    /// Apply the world's presentation preference to the surface.
    ///
    /// `ENGINE_PRESENT_MODE` pins the mode for stress tests and overrides the
    /// preference. Without the pin, vsync requests FIFO and vsync off requires
    /// Immediate. A surface that cannot present immediately is a loud
    /// configuration error.
    pub(super) fn apply_present_mode(&mut self, vsync: bool) {
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
    pub(super) fn scene_size(&self) -> (u32, u32) {
        taa::scaled_size(self.config.width, self.config.height, self.render_scale)
    }

    /// Follow the world's render scale; a new one rebuilds the scene targets.
    pub(super) fn follow_render_scale(&mut self, world: &World) {
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
    pub(super) fn rebuild_screen_targets(&mut self) {
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
}
