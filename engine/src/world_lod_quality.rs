//! LOD models, quality presets and render switches, GPU info, hitch logs.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Spawn a LOD set: one placement list drawn with `levels` (detailed
    /// first), each instance at the level its camera distance falls in, with
    /// a dithered cross-fade between levels. See [`crate::lod`].
    pub fn spawn_lod(&mut self, levels: Vec<LodLevel>) -> EngineResult<LodId> {
        LodSet::validate_levels(&levels)?;
        // Building GPU-ready vertices is the slow part: do the levels at once.
        use rayon::prelude::*;
        let built: Vec<_> = levels
            .into_par_iter()
            .map(|level| {
                let albedo = (
                    level.mesh.albedo().cloned(),
                    level.mesh.material_maps().cloned(),
                );
                (level.mesh.build(), albedo, level.max_distance)
            })
            .collect();
        let mut ids = Vec::with_capacity(built.len());
        for (mesh, albedo, max_distance) in built {
            let id = self.spawn_built_instanced(mesh, Vec::new());
            self.attach_albedo(id, albedo)?;
            ids.push((id, max_distance));
        }
        Ok(self.add_lod_set(ids))
    }

    /// [`Self::spawn_lod`] from meshes made ready on another thread with
    /// [`Mesh::prepare`]: `(mesh, max_distance)` per level, detailed first.
    /// For LOD sets a game makes while it runs (streamed tiles of a distant
    /// city, say), where building the vertices on the main thread would
    /// cost a frame.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn demo(world: &mut World) -> EngineResult<()> {
    /// // On a worker: the meshes, prepared.
    /// let near = Shape::box_at((0.0, 5.0, 0.0), (10.0, 10.0, 10.0), rgb(200, 190, 170))?.prepare();
    /// let far = Shape::box_at((0.0, 5.0, 0.0), (10.0, 10.0, 10.0), rgb(200, 190, 170))?.prepare();
    /// // On the main thread: only the hand-over.
    /// let tile = world.spawn_lod_prepared(vec![(near, 800.0), (far, 3000.0)])?;
    /// world.set_lod_places(tile, &[Place::new(0.0, 0.0, 0.0)])?;
    /// # Ok(()) }
    /// ```
    pub fn spawn_lod_prepared(&mut self, levels: Vec<(PreparedMesh, f32)>) -> EngineResult<LodId> {
        LodSet::validate_distances(levels.iter().map(|l| l.1))?;
        let mut ids: Vec<(EntityId, f32)> = Vec::with_capacity(levels.len());
        for (mesh, max_distance) in levels {
            let PreparedMesh {
                built,
                albedo,
                maps,
                key,
            } = mesh;
            let id = self.spawn_built_instanced(built, Vec::new());
            if let Err(e) = self.attach_albedo_keyed(id, (albedo, maps), key) {
                self.despawn(id);
                for (done, _) in ids {
                    self.despawn(done);
                }
                return Err(e);
            }
            ids.push((id, max_distance));
        }
        Ok(self.add_lod_set(ids))
    }

    /// Registers spawned level entities as a new, empty LOD set.
    fn add_lod_set(&mut self, ids: Vec<(EntityId, f32)>) -> LodId {
        let id = LodId(self.next_lod);
        self.next_lod += 1;
        self.lods.insert(
            id,
            LodSet {
                levels: ids,
                places: Vec::new(),
                assigned_for: None,
                bounds: None,
                assigned_zone: None,
            },
        );
        id
    }

    /// Another LOD set drawing the same model as `model` at `places`. Its
    /// levels share `model`'s meshes and GPU buffers and are drawn in the
    /// same batches, so many small sets (one per streamed tile, say) cost
    /// about as much as one big one. Despawn the like-sets before `model`.
    pub fn spawn_lod_like(&mut self, model: LodId, places: &[Place]) -> EngineResult<LodId> {
        let levels: Vec<(EntityId, f32)> = self
            .lods
            .get(&model)
            .ok_or(EngineError::UnknownEntity)?
            .levels
            .clone();
        let mut ids = Vec::with_capacity(levels.len());
        for (prototype, max_distance) in levels {
            // A like-set of a like-set draws the original prototype.
            let prototype = self
                .entities
                .get(&prototype)
                .and_then(|e| e.instance_of)
                .unwrap_or(prototype);
            match self.spawn_instanced_like(prototype) {
                Ok(id) => ids.push((id, max_distance)),
                Err(e) => {
                    for (id, _) in ids {
                        self.despawn(id);
                    }
                    return Err(e);
                }
            }
        }
        let id = LodId(self.next_lod);
        self.next_lod += 1;
        self.lods.insert(
            id,
            LodSet {
                levels: ids,
                places: Vec::new(),
                assigned_for: None,
                bounds: None,
                assigned_zone: None,
            },
        );
        if let Err(e) = self.set_lod_places(id, places) {
            self.despawn_lod(id);
            return Err(e);
        }
        Ok(id)
    }

    /// [`Self::spawn_lod`] and [`Self::set_lod_places`] in one call: the
    /// usual way to put a tree, rock or prop model into the world.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World) -> EngineResult<()> {
    /// let oak = tree_model(TreeSpecies::Oak, 0)?;
    /// let places = Scatter::new(1).rect(-50.0, -50.0, 50.0, 50.0).spacing(9.0).places()?;
    /// world.spawn_lod_at(oak.lod_levels(), &places)?;
    /// # Ok(()) }
    /// ```
    pub fn spawn_lod_at(&mut self, levels: Vec<LodLevel>, places: &[Place]) -> EngineResult<LodId> {
        let id = self.spawn_lod(levels)?;
        if let Err(e) = self.set_lod_places(id, places) {
            self.despawn_lod(id);
            return Err(e);
        }
        Ok(id)
    }

    /// Where a LOD set's instances stand.
    pub fn set_lod_places(&mut self, id: LodId, places: &[Place]) -> EngineResult<()> {
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "set_lod_places has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let set = self.lods.get_mut(&id).ok_or(EngineError::UnknownEntity)?;
        set.places = crate::lod::instances_from(places);
        set.assigned_for = None;
        set.update_bounds();
        Ok(())
    }

    /// The instanced entities drawing each level, detailed first (to change
    /// their shadow casting, say).
    pub fn lod_levels(&self, id: LodId) -> EngineResult<Vec<EntityId>> {
        let set = self.lods.get(&id).ok_or(EngineError::UnknownEntity)?;
        Ok(set.levels.iter().map(|(e, _)| *e).collect())
    }

    /// Hang a LOD set on an entity (a pivot, a part of a robot): it draws
    /// once, at `local` on `parent`, follows it as it moves, and picks its
    /// level by camera distance like any set. With [`Self::spawn_lod_like`]
    /// every copy of a model shares one set of meshes and maps, so a robot
    /// built of many moving parts costs one upload per part, however many
    /// robots walk about, and carries LOD levels.
    ///
    /// The set goes when its parent is despawned.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World, leg: Vec<LodLevel>) -> EngineResult<()> {
    /// let model = world.spawn_lod(leg)?; // once: the shared meshes
    /// let hip = world.spawn_pivot(Place::new(0.0, 4.0, 0.0));
    /// let thigh = world.spawn_lod_like(model, &[])?; // per robot
    /// world.attach_lod(thigh, hip, Place::default())?;
    /// world.set_place(hip, Place::new(0.0, 4.0, 0.0).with_pitch_deg(20.0))?; // it swings
    /// # Ok(()) }
    /// ```
    pub fn attach_lod(&mut self, id: LodId, parent: EntityId, local: Place) -> EngineResult<()> {
        if !self.lods.contains_key(&id) {
            return Err(EngineError::UnknownEntity);
        }
        let frame = self.world_transform(parent)?;
        let local = local.to_matrix();
        self.lod_follow.insert(id, (parent, local));
        self.place_lod_on(id, frame * local);
        Ok(())
    }

    /// Stop a LOD set following its parent; it stays where it is.
    pub fn detach_lod(&mut self, id: LodId) {
        self.lod_follow.remove(&id);
    }

    /// The entity a LOD set hangs on, if any.
    pub fn lod_parent(&self, id: LodId) -> Option<EntityId> {
        self.lod_follow.get(&id).map(|(p, _)| *p)
    }

    pub(super) fn place_lod_on(&mut self, id: LodId, m: Mat4) {
        let Some(set) = self.lods.get_mut(&id) else {
            return;
        };
        if set.places.len() == 1
            && crate::hierarchy::same_up_to_rounding(set.places[0].transform, m)
        {
            return;
        }
        set.places = vec![crate::place::MeshInstance::from_matrix(m)];
        set.assigned_for = None;
        set.assigned_zone = None;
        set.update_bounds();
    }

    /// Move attached LOD sets with their parents; drop those whose parent
    /// is gone.
    pub(super) fn follow_lods(&mut self) {
        if self.lod_follow.is_empty() {
            return;
        }
        let follow: Vec<(LodId, EntityId, Mat4)> = self
            .lod_follow
            .iter()
            .map(|(id, (p, l))| (*id, *p, *l))
            .collect();
        for (id, parent, local) in follow {
            match self.world_transform(parent) {
                Ok(frame) if self.lods.contains_key(&id) => self.place_lod_on(id, frame * local),
                _ => {
                    self.lod_follow.remove(&id);
                    self.despawn_lod(id);
                }
            }
        }
    }

    pub fn despawn_lod(&mut self, id: LodId) {
        self.lod_follow.remove(&id);
        if let Some(set) = self.lods.remove(&id) {
            for (entity, _) in set.levels {
                self.despawn(entity);
            }
        }
    }

    /// Scales every LOD distance: above 1 keeps detail further out, below 1
    /// trades it for speed. The quality knob for automatic tuning.
    pub fn set_lod_bias(&mut self, bias: f32) -> EngineResult<()> {
        if !bias.is_finite() || bias <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "LOD bias {bias} must be positive"
            )));
        }
        self.lod_bias = bias;
        Ok(())
    }

    pub fn lod_bias(&self) -> f32 {
        self.lod_bias
    }

    /// Take a game's startup [`Settings`](crate::Settings): quality first,
    /// then the single switches, so they win over the preset.
    pub(crate) fn apply_startup_settings(
        &mut self,
        settings: &crate::settings::Settings,
    ) -> EngineResult<()> {
        self.set_quality(settings.quality);
        self.set_taa(settings.taa);
        self.set_tone_map(settings.tone_map);
        self.set_render_scale(settings.render_scale);
        self.set_target_fps(settings.target_fps)?;
        if let Some(m) = settings.view_distance {
            self.set_view_distance(m)?;
        }
        self.set_vsync(settings.vsync);
        self.set_whats_new_button(settings.whats_new_button);
        self.settings = settings.clone();
        self.env_overrides = crate::settings::Settings::env_overrides();
        Ok(())
    }

    /// The settings the game started with (`Engine::run_with_settings`;
    /// defaults for `Engine::run`). Changes made later with the setters
    /// (`set_quality`, `set_taa`, ...) are not reflected here.
    pub fn settings(&self) -> &crate::settings::Settings {
        &self.settings
    }

    /// The `ENGINE_*` environment variables this run started with, as
    /// `NAME=value`. They override the game's settings (debug switches).
    pub fn env_overrides(&self) -> &[String] {
        &self.env_overrides
    }

    /// Apply a quality preset: shadows, ambient occlusion, GI and LOD bias.
    /// A new world starts at [`Quality::High`](crate::quality::Quality::High).
    /// Settings changed afterwards (for example `set_ambient_occlusion`) win
    /// until the next call.
    pub fn set_quality(&mut self, quality: crate::quality::Quality) {
        let s = quality.settings();
        self.quality = quality;
        self.shadows = s.shadows;
        self.ambient_occlusion = s.ambient_occlusion;
        self.global_illumination = s.global_illumination;
        self.lod_bias = s.lod_bias;
        self.grass = s.grass;
        self.ssr = s.screen_space_reflections;
        if let Some(t) = &mut self.auto_tune {
            t.max_bias = s.lod_bias;
        }
    }

    /// How far grass reaches, as a factor on the terrain's
    /// [`crate::proc_terrain::Grass::distance`] (0 hides it). Quality presets
    /// set it; the LOD bias (and so the frame-rate tuner) scales it further.
    pub fn set_grass_scale(&mut self, scale: f32) -> EngineResult<()> {
        if !scale.is_finite() || !(0.0..=4.0).contains(&scale) {
            return Err(EngineError::InvalidValue(format!(
                "grass scale {scale} must be within 0..=4"
            )));
        }
        self.grass = scale;
        Ok(())
    }

    /// The grass distance factor actually used: the scale times the LOD bias.
    pub fn grass_scale(&self) -> f32 {
        self.grass * self.lod_bias
    }

    /// Temporal anti-aliasing (on by default): smooths edges, shimmer and
    /// the LOD cross-fade by blending each frame with the last ones.
    /// `ENGINE_TAA=0` also switches it off. TAA is the engine's main
    /// anti-aliasing (MSAA defaults to off while it runs): turning it off in
    /// code leaves edges jagged unless the app starts with `ENGINE_MSAA=4`.
    pub fn set_taa(&mut self, on: bool) {
        self.taa = on;
    }

    pub fn taa(&self) -> bool {
        self.taa
    }

    /// Draw the scene at this share of the window's width and height, and
    /// let the TAA pass upscale it to the window (like FSR 2 or Unreal's
    /// TSR). 1.0 (the default) draws at full size; 0.75 draws 56% of the
    /// pixels, 0.5 a quarter. Everything sized to the screen (depth, AO,
    /// reflections, occlusion) shrinks with it, so GPU time spent per pixel
    /// drops about in step; bloom, tone mapping and the UI stay at full
    /// size. With TAA on, a still view gathers nearly full detail over a few
    /// frames; with it off the frame is only filtered up. Clamped to
    /// 0.5..=1.0. `ENGINE_RENDER_SCALE` overrides it.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = if scale.is_finite() {
            scale.clamp(crate::render::taa::MIN_RENDER_SCALE, 1.0)
        } else {
            1.0
        };
    }

    /// See [`World::set_render_scale`].
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    /// Occlusion culling: skip meshes hidden behind other things, such as
    /// the blocks behind a street wall. The GPU tests each mesh's box against
    /// this frame's depth, so nothing pops in; what it saves is the vertex
    /// and pixel work of everything out of sight. `true` runs it always,
    /// `false` never; see [`World::set_occlusion_mode`] for the default,
    /// [`OcclusionMode::Auto`], which runs it only where it pays.
    pub fn set_occlusion_culling(&mut self, on: bool) {
        self.occlusion_mode = if on {
            OcclusionMode::On
        } else {
            OcclusionMode::Off
        };
    }

    /// How occlusion culling is chosen. The default, [`OcclusionMode::Auto`],
    /// tries it for a moment whenever enough meshes are in view, keeps it
    /// while it hides a good share of them (cities, interiors, hills) and
    /// switches it off again where it doesn't (open fields), trying again
    /// later, less often each time it didn't pay. `ENGINE_OCCLUSION=0`, `=1`
    /// or `=auto` overrides this for comparisons.
    pub fn set_occlusion_mode(&mut self, mode: OcclusionMode) {
        self.occlusion_mode = mode;
    }

    /// The occlusion mode in effect (the setting, or `ENGINE_OCCLUSION`).
    pub fn occlusion_mode(&self) -> OcclusionMode {
        occlusion_env().unwrap_or(self.occlusion_mode)
    }

    /// What occlusion culling found in a recent frame (zeros while it is off).
    pub fn occlusion_stats(&self) -> OcclusionStats {
        self.occlusion_stats
    }

    pub(crate) fn set_occlusion_stats(&mut self, stats: OcclusionStats) {
        self.occlusion_stats = stats;
    }

    /// Whether occlusion culling may run: on, or chosen automatically (the
    /// setting, or `ENGINE_OCCLUSION`). [`World::occlusion_stats`] says
    /// whether it ran lately.
    pub fn occlusion_culling(&self) -> bool {
        self.occlusion_mode() != OcclusionMode::Off
    }

    /// The last preset given to [`Self::set_quality`].
    pub fn quality(&self) -> crate::quality::Quality {
        self.quality
    }

    /// Keep the frame rate at or above `fps` by trading detail for speed,
    /// within the current [`Self::set_quality`] preset: LOD distances first,
    /// then GI, then AO. `None` stops tuning and leaves things as they are.
    /// See [`crate::quality`].
    pub fn set_target_fps(&mut self, fps: Option<f32>) -> EngineResult<()> {
        match fps {
            None => self.auto_tune = None,
            Some(f) if f.is_finite() && (1.0..=1000.0).contains(&f) => {
                let max_bias = self.quality.settings().lod_bias;
                self.auto_tune = Some(crate::quality::AutoTune::new(f, max_bias));
            }
            Some(f) => {
                return Err(EngineError::InvalidValue(format!(
                    "target fps {f} must be within 1..=1000"
                )))
            }
        }
        Ok(())
    }

    pub fn target_fps(&self) -> Option<f32> {
        self.auto_tune.as_ref().map(|t| t.target_fps)
    }

    /// Runs particles and ribbons by the frame time, unless the game drives
    /// them itself (see [`ParticleWorld::advance`]).
    pub(crate) fn tick_effects(&mut self, dt: f32) {
        self.particles.auto_advance(dt);
        self.ribbons.auto_advance(dt);
    }

    pub(crate) fn tick_auto_quality(&mut self, dt: f32) {
        use crate::quality::TuneStep;
        let (gi, ao) = (
            self.global_illumination.is_some(),
            self.ambient_occlusion.is_some(),
        );
        let Some(t) = &mut self.auto_tune else {
            return;
        };
        match t.tick(dt, self.lod_bias, gi, ao) {
            TuneStep::Hold => {}
            TuneStep::Bias(b) => self.lod_bias = b,
            TuneStep::DropGi => self.global_illumination = None,
            TuneStep::DropAo => self.ambient_occlusion = None,
        }
    }

    /// Width of the cross-fade band as a fraction of each level's end
    /// distance (default 0.1; 0 swaps without fading).
    pub fn set_lod_fade(&mut self, fraction: f32) -> EngineResult<()> {
        if !(0.0..=0.5).contains(&fraction) {
            return Err(EngineError::InvalidValue(format!(
                "LOD fade {fraction} must be within 0..=0.5"
            )));
        }
        self.lod_fade = fraction;
        for set in self.lods.values_mut() {
            set.assigned_for = None;
            set.assigned_zone = None;
        }
        Ok(())
    }

    /// Reassign LOD levels for the current camera. Runs once per frame
    /// before rendering; cheap while the camera stands still.
    pub(crate) fn tick_lods(&mut self) {
        self.follow_lods();
        if self.lods.is_empty() {
            return;
        }
        let eye = self.camera.eye();
        let (bias, fade) = (self.lod_bias, self.lod_fade);
        let mut stale_sets: Vec<&mut LodSet> = Vec::new();
        for set in self.lods.values_mut() {
            // Far sets need a proportionally longer move to change visibly
            // (a metre at a kilometre moves a fade by a fraction of a percent).
            let near = set
                .bounds
                .map_or(0.0, |(c, r)| (c.distance(eye) - r).max(0.0));
            let threshold = set.move_threshold(bias, fade).max(near * 0.01);
            let stale = match set.assigned_for {
                None => true,
                Some((at, b)) => b != bias || at.distance(eye) >= threshold,
            };
            if !stale {
                continue;
            }
            // Wholly within the same pair of thresholds as last time: no
            // instance can have changed level.
            let zone = set.zone(eye, bias, fade);
            if zone.is_some()
                && zone == set.assigned_zone
                && set.assigned_for.is_some_and(|(_, b)| b == bias)
            {
                continue;
            }
            set.assigned_zone = zone;
            set.assigned_for = Some((eye, bias));
            stale_sets.push(set);
        }
        let assign_set = |set: &&mut LodSet| {
            let ends: Vec<f32> = set.levels.iter().map(|(_, d)| *d).collect();
            let lists = crate::lod::assign(&set.places, &ends, eye, bias, fade);
            set.levels
                .iter()
                .map(|(e, _)| *e)
                .zip(lists)
                .collect::<Vec<_>>()
        };
        // Sets are independent: many are assigned on the frame's threads. A
        // few are quicker here than waking them.
        let places: usize = stale_sets.iter().map(|s| s.places.len()).sum();
        let updates: Vec<(EntityId, Vec<MeshInstance>)> = if places < LOD_PARALLEL_PLACES {
            stale_sets.iter().flat_map(assign_set).collect()
        } else {
            use rayon::prelude::*;
            crate::parallel::frame_pool()
                .install(|| stale_sets.par_iter().flat_map_iter(assign_set).collect())
        };
        for (entity, list) in updates {
            self.replace_instances(entity, list);
        }
    }

    /// Swap an instanced entity's list. Only an empty/non-empty change needs
    /// a scene resync (draw and shadow source lists skip empty entities),
    /// and with GPU batches not even that (see `batched_instances`).
    #[track_caller]
    pub(super) fn replace_instances(&mut self, id: EntityId, instances: Vec<MeshInstance>) {
        let Some(e) = self.entities.get_mut(&id) else {
            return;
        };
        let was_empty = e.instances.is_empty();
        let now_empty = instances.is_empty();
        if was_empty && now_empty {
            return;
        }
        e.instances = instances;
        e.bump_xform();
        if was_empty != now_empty && !self.batched_instances() {
            self.bump_render_epoch();
        } else {
            self.instance_epoch = self
                .instance_epoch
                .checked_add(1)
                .expect("world instance epoch overflow");
            self.touch(id);
        }
    }

    /// Debugging aid: switches off the places where the mesh shader drops
    /// triangles or pixels, or paints what passes one of them (bit flags,
    /// see [`DEBUG_SHADING_FLAGS`]). 0 is normal drawing.
    pub fn set_debug_shading(&mut self, flags: u32) {
        self.debug_shading = flags;
    }

    pub fn debug_shading(&self) -> u32 {
        self.debug_shading
    }

    /// Debugging aid: the renderer throws away its GPU copies of every mesh
    /// and/or every texture and uploads them again from the world before the
    /// next frame. If something looks wrong and comes back right after this,
    /// the GPU copy was damaged, not the world's data.
    /// Debugging aid: before the next frame the renderer reads every mesh
    /// back from the GPU, compares it byte by byte with the world's mesh and
    /// checks the world's meshes themselves (indices in range, finite
    /// positions). Findings go to `ENGINE_AUDIT_LOG` (or `gpu_check.log`).
    /// Stalls for a moment.
    /// Debugging aid: before the next frame the renderer configures the
    /// window surface again (`surface`) and/or makes every window-sized
    /// buffer anew (`buffers`: depth, AO, reflections, post, TAA), as a
    /// window resize does.
    pub fn request_renderer_rebuild(&mut self, surface: bool, buffers: bool) {
        if surface {
            self.renderer_rebuild.0 += 1;
        }
        if buffers {
            self.renderer_rebuild.1 += 1;
        }
    }

    /// The graphics card and driver, once the window's renderer exists
    /// (None headless). [`crate::gpu_info::GpuInfo::driver_warning`] says
    /// when the driver is known to draw wrongly.
    pub fn gpu_info(&self) -> Option<&crate::gpu_info::GpuInfo> {
        self.gpu_info.as_ref()
    }

    pub(crate) fn set_gpu_info(&mut self, info: crate::gpu_info::GpuInfo) {
        self.gpu_info = Some(info);
    }

    pub fn request_gpu_check(&mut self) {
        self.gpu_check += 1;
    }

    pub fn request_gpu_reupload(&mut self, meshes: bool, textures: bool) {
        if meshes {
            self.reupload.0 += 1;
        }
        if textures {
            self.reupload.1 += 1;
        }
    }

    pub(crate) fn resource_epoch(&self) -> u64 {
        self.resource_epoch
    }

    /// Record a named cost on this frame. Printed only if the frame hitches.
    pub fn hitch_span(&mut self, name: impl Into<String>, ms: f32, detail: impl Into<String>) {
        if !ms.is_finite() || ms < 0.0 {
            panic!("hitch span ms must be finite and >= 0, got {ms}");
        }
        self.hitch_spans.push(HitchSpan {
            name: name.into(),
            ms,
            detail: detail.into(),
        });
    }

    pub(crate) fn take_hitch_spans(&mut self) -> Vec<HitchSpan> {
        std::mem::take(&mut self.hitch_spans)
    }

    /// Append hitch reports to `path`, or pass `None` to stop logging.
    pub fn set_hitch_log(&mut self, path: Option<PathBuf>) {
        if let Some(p) = &path {
            if p.as_os_str().is_empty() {
                panic!("hitch log path must not be empty");
            }
        }
        self.hitch_log = path;
    }

    pub fn hitch_log(&self) -> Option<&Path> {
        self.hitch_log.as_deref()
    }
}
