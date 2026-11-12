//! Sun shadow cascades: bundles, multi-draw planning and the shadow pass.
//!
//! Part of `impl Renderer`, split from `render/mod.rs` by topic.

use super::*;

impl Renderer {
    /// Record the shadow bundles whose casters changed, all at once on the
    /// frame's threads. A view's meshes are bundled per source group and
    /// its instance batches in one more bundle; only bundles whose own
    /// casters changed are recorded. Each reads only shared, unchanging
    /// renderer state and goes to its own place; all are done before any
    /// pass uses them.
    pub(super) fn prepare_shadow_bundles(
        &mut self,
        slots: &[usize],
        selections: &[ShadowCullSelection],
    ) {
        // (slot, group, draws, batches); `group` is the place in the
        // view's new group list (unused for batches).
        let mut jobs: Vec<(usize, usize, &[ShadowDraw], bool)> = Vec::new();
        let mut kept: Vec<Vec<Option<ShadowGroupBundle>>> = Vec::new();
        // Views whose mesh casters go out as multi-draws made anew.
        let mut multi_jobs: Vec<(usize, &[ShadowDraw])> = Vec::new();
        for (&slot, selection) in slots.iter().zip(selections) {
            let use_multi = self.multi_draw && selection.draws.len() >= SHADOW_MULTI_MIN_DRAWS;
            let view = &mut self.shadow_bundles[slot];
            let mut old: HashMap<u32, ShadowGroupBundle> = std::mem::take(&mut view.groups)
                .into_iter()
                .map(|g| (g.group, g))
                .collect();
            let mut list = Vec::new();
            if use_multi {
                if view
                    .multi
                    .as_ref()
                    .is_none_or(|m| m.draws != selection.draws)
                {
                    view.multi = None;
                    multi_jobs.push((slot, &selection.draws));
                }
            } else {
                view.multi = None;
                for run in selection.draws.chunk_by(|a, b| a.group == b.group) {
                    let reuse = old.remove(&run[0].group).filter(|g| g.draws == run);
                    if reuse.is_none() {
                        jobs.push((slot, list.len(), run, false));
                    }
                    list.push(reuse);
                }
            }
            kept.push(list);
            if view.batch_draws != selection.batch_draws {
                view.batch_draws.clone_from(&selection.batch_draws);
                view.batches = None;
                if !selection.batch_draws.is_empty() {
                    jobs.push((slot, 0, &selection.batch_draws, true));
                }
            }
        }
        let ctx = ShadowBundleCtx {
            device: &self.device,
            shadow: &self.shadow,
            gpu_meshes: &self.gpu_meshes,
            cutout_albedo: &self.cutout_albedo,
            gpu_instance_batches: &self.gpu_instance_batches,
        };
        let record = |&(slot, _, draws, batches): &(usize, usize, &[ShadowDraw], bool)| {
            ctx.record(slot, draws, batches)
        };
        let recorded: Vec<wgpu::RenderBundle> = if jobs.len() > 1 {
            use rayon::prelude::*;
            crate::parallel::frame_pool().install(|| jobs.par_iter().map(record).collect())
        } else {
            jobs.iter().map(record).collect()
        };
        let index: HashMap<usize, usize> = slots.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        for ((slot, place, draws, batches), bundle) in jobs.into_iter().zip(recorded) {
            if batches {
                self.shadow_bundles[slot].batches = Some(bundle);
            } else {
                kept[index[&slot]][place] = Some(ShadowGroupBundle {
                    group: draws[0].group,
                    draws: draws.to_vec(),
                    bundle,
                });
            }
        }
        for (&slot, list) in slots.iter().zip(kept) {
            self.shadow_bundles[slot].groups = list
                .into_iter()
                .map(|g| g.expect("every shadow group is kept or recorded"))
                .collect();
        }
        let made: Vec<(usize, ShadowMulti)> = multi_jobs
            .into_iter()
            .map(|(slot, draws)| (slot, self.plan_shadow_multi(slot, draws)))
            .collect();
        for (slot, multi) in made {
            self.shadow_bundles[slot].multi = Some(multi);
        }
    }

    /// A shadow view's mesh casters as multi-draws: untextured casters
    /// first, then the alpha-tested ones, each grouped by texture and
    /// buffers in order of first appearance.
    pub(super) fn plan_shadow_multi(&self, slot: usize, draws: &[ShadowDraw]) -> ShadowMulti {
        type Key = (
            bool,
            Option<wgpu::BindGroup>,
            wgpu::Buffer,
            wgpu::Buffer,
            wgpu::Buffer,
        );
        // wgpu handles hash by their id, which never changes.
        #[allow(clippy::mutable_key_type)]
        let mut index: HashMap<Key, usize> = HashMap::new();
        let mut keys: Vec<Key> = Vec::new();
        let mut lists: Vec<Vec<DrawIndexedArgs>> = Vec::new();
        let mut rest: Vec<ShadowDraw> = Vec::new();
        for cutout in [false, true] {
            for d in draws {
                let albedo = d.albedo.and_then(|tid| self.cutout_albedo.get(&tid));
                if albedo.is_some() != cutout {
                    continue;
                }
                if d.submit == MeshSubmit::Compacted {
                    rest.push(*d);
                    continue;
                }
                let g = self
                    .gpu_meshes
                    .get(&d.id)
                    .expect("shadow multi-draw caster is synced");
                let key: Key = (
                    cutout,
                    albedo.cloned(),
                    g.vertex_buf.clone(),
                    g.index_buf.clone(),
                    g.instance_buf.clone(),
                );
                let at = *index.entry(key.clone()).or_insert_with(|| {
                    keys.push(key);
                    lists.push(Vec::new());
                    keys.len() - 1
                });
                lists[at].push(DrawIndexedArgs {
                    index_count: g.opaque_index_count as u32,
                    instance_count: g.instance_count as u32,
                    first_index: g.first_index,
                    base_vertex: g.base_vertex,
                    first_instance: g.first_instance,
                });
            }
        }
        let stride = std::mem::size_of::<DrawIndexedArgs>() as u64;
        let mut args: Vec<DrawIndexedArgs> = Vec::new();
        let mut groups = Vec::new();
        for ((cutout, albedo, vertex, index, instances), list) in keys.into_iter().zip(lists) {
            groups.push(ShadowMultiGroup {
                cutout,
                albedo,
                vertex,
                index,
                instances,
                offset: args.len() as u64 * stride,
                count: list.len() as u32,
            });
            args.extend(list);
        }
        if args.is_empty() {
            args.push(DrawIndexedArgs::empty_range(0, 0, 0));
        }
        use wgpu::util::DeviceExt;
        let args = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shadow-multi-draw-args"),
                contents: bytemuck::cast_slice(&args),
                usage: wgpu::BufferUsages::INDIRECT,
            });
        let rest = (!rest.is_empty()).then(|| {
            ShadowBundleCtx {
                device: &self.device,
                shadow: &self.shadow,
                gpu_meshes: &self.gpu_meshes,
                cutout_albedo: &self.cutout_albedo,
                gpu_instance_batches: &self.gpu_instance_batches,
            }
            .record(slot, &rest, false)
        });
        ShadowMulti {
            draws: draws.to_vec(),
            groups,
            args,
            rest,
        }
    }

    /// The shadow views (sun cascades, lamp layers), each in its own
    /// command buffer, recorded side by side on the frame's threads.
    ///
    /// Order is kept by construction: the commands `encoder` holds so far
    /// are closed into `recorded` first, then each view's buffer follows in
    /// view order, and `encoder` starts afresh for what comes after. The
    /// queue runs them in that order. Each view's instance cull (which
    /// shares buffers between views) is put in its own view's buffer, ahead
    /// of its pass, here on this thread; the threads only add the passes,
    /// reading renderer state that nothing changes meanwhile.
    pub(super) fn encode_shadow_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        recorded: &mut Vec<wgpu::CommandBuffer>,
        world: &World,
    ) {
        let sun = world.shadows().is_some();
        let lamp_views = self.shadow.lamps.frame_views.clone();
        if !sun && lamp_views.is_empty() {
            return;
        }
        let far = world.shadows().map(|s| s.cascade_end_m[2]).unwrap_or(120.0);
        let focus = world.camera().target();
        let submit = world.instance_submit();
        // One frustum per view: the sun's cascades (when on), then lamp layers.
        let mut frustums: Vec<Frustum> = Vec::new();
        let mut slots: Vec<usize> = Vec::new();
        if sun {
            for cascade in (0..3).filter(|&c| self.shadow_draw[c]) {
                frustums.push(Frustum::from_view_projection(self.shadow_vp[cascade]));
                slots.push(cascade);
            }
        }
        self.gpu_stats.shadow_cascades_drawn =
            self.shadow_draw.iter().filter(|d| **d).count() as u32 * u32::from(sun);
        for (i, vp) in lamp_views.iter().enumerate() {
            frustums.push(Frustum::from_view_projection(*vp));
            slots.push(3 + i);
        }
        if frustums.is_empty() {
            return;
        }
        let mut t = std::time::Instant::now();
        let mut selections = self.select_shadow_culls(world, &frustums);
        let changed_statics = self.sync_shadow_scene(world);
        if !changed_statics.is_empty() {
            for (c, gen) in self.static_caster_gen.iter_mut().enumerate() {
                let frustum = Frustum::from_view_projection(self.shadow_vp[c]);
                if changed_statics.iter().any(|b| frustum.intersects(*b)) {
                    *gen += 1;
                }
            }
        }
        self.cpu_spans.mark("shadows.select", &mut t);
        let default_space = world.living_in() == SpaceId::DEFAULT;
        let voxel_on = default_space && self.voxel.is_drawing();
        // Sun cascades with the static cache: static casters (not changed
        // lately, not swaying) go to their own selection, drawn into the
        // cache layer only when it is out of date.
        let caching = sun && world.shadow_cache() && shadow_cache_env();
        let mut uses = vec![ShadowCacheUse::Plain; slots.len()];
        let mut statics: Vec<(usize, ShadowCullSelection)> = Vec::new();
        if caching {
            if self.shadow.static_cache.is_none() {
                let size = self.shadow.depth_texture.width();
                self.shadow.static_cache = Some(shadow::StaticShadowCache::new(&self.device, size));
            }
            for (i, &slot) in slots.iter().enumerate() {
                if slot >= 3 {
                    continue;
                }
                let (fixed, moving) = self.split_static_casters(&selections[i]);
                selections[i] = moving;
                let key = shadow::StaticShadowKey {
                    vp: self.shadow_vp[slot].to_cols_array().map(f32::to_bits),
                    epoch: self.static_shadow_epoch,
                    voxel_epoch: if voxel_on {
                        self.voxel.epoch()
                    } else {
                        u64::MAX
                    },
                    casters: caster_hash(&fixed, world.living_in())
                        ^ self.static_caster_gen[slot].wrapping_mul(0x9E37_79B9_7F4A_7C15),
                };
                let cache = self
                    .shadow
                    .static_cache
                    .as_mut()
                    .expect("static shadow cache made above");
                let fresh = cache.keys[slot].is_some_and(|(k, at)| {
                    k == key && self.frame_no.saturating_sub(at) < STATIC_SHADOW_MAX_AGE
                });
                if fresh {
                    uses[i] = ShadowCacheUse::Hit;
                    self.gpu_stats.shadow_cache_hits += 1;
                } else {
                    uses[i] = ShadowCacheUse::Miss;
                    cache.keys[slot] = Some((key, self.frame_no));
                    statics.push((i, fixed));
                }
            }
        } else {
            self.shadow.static_cache = None;
        }
        // Bundles: every view's (moving) casters, and the static casters of
        // the cascades drawn into the cache this frame.
        let mut bundle_slots = slots.clone();
        let mut bundle_selections: Vec<ShadowCullSelection> = Vec::new();
        bundle_selections.append(&mut selections);
        let mut static_compact: HashMap<usize, Vec<EntityId>> = HashMap::new();
        for (i, fixed) in statics {
            bundle_slots.push(STATIC_SHADOW_SLOT + slots[i]);
            static_compact.insert(i, fixed.compact_batches.clone());
            self.gpu_stats.indirect_draws += fixed
                .batch_draws
                .iter()
                .filter(|draw| draw.submit == MeshSubmit::Compacted)
                .count() as u32;
            bundle_selections.push(fixed);
        }
        self.prepare_shadow_bundles(&bundle_slots, &bundle_selections);
        self.cpu_spans.mark("shadows.bundles", &mut t);
        let selections = &bundle_selections[..slots.len()];
        if let Some(scene) = self.shadow_scene.as_mut() {
            let mut views: Vec<Option<(Frustum, shadow_scene::ViewMode)>> =
                vec![None; shadow_scene::MAX_VIEWS];
            for (i, &slot) in slots.iter().enumerate() {
                let mode = match uses[i] {
                    ShadowCacheUse::Plain => shadow_scene::ViewMode::All,
                    ShadowCacheUse::Hit | ShadowCacheUse::Miss => shadow_scene::ViewMode::Moving,
                };
                views[slot] = Some((frustums[i], mode));
                if uses[i] == ShadowCacheUse::Miss {
                    views[STATIC_SHADOW_SLOT + slot] =
                        Some((frustums[i], shadow_scene::ViewMode::Static));
                }
            }
            scene.encode(&self.device, &self.queue, encoder, &views);
        }

        let new_encoder = |device: &wgpu::Device, label| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        };
        let before = std::mem::replace(encoder, new_encoder(&self.device, "after-shadows"));
        recorded.push(before.finish());
        let mut views = Vec::new();
        for (i, ((frustum, selection), &slot)) in
            frustums.iter().zip(selections).zip(&slots).enumerate()
        {
            let mut view_encoder = new_encoder(&self.device, "shadow-view");
            if submit == InstanceSubmit::GpuIndirect {
                // A cache miss also draws the static batches: cull them too.
                let mut compact = selection.compact_batches.clone();
                if let Some(more) = static_compact.get(&i) {
                    compact.extend_from_slice(more);
                }
                self.dispatch_instance_cull(
                    &mut view_encoder,
                    frustum,
                    &compact,
                    CullView::Shadow,
                    None,
                    None,
                );
            }
            self.gpu_stats.indirect_draws += u32::try_from(
                selection
                    .batch_draws
                    .iter()
                    .filter(|draw| draw.submit == MeshSubmit::Compacted)
                    .count(),
            )
            .expect("shadow indirect draw count exceeds u32");
            let skinned = world
                .animated_entities()
                .filter(|(id, anim)| anim.space() == world.living_in() && !world.room_hidden(**id))
                .filter(|(id, anim)| {
                    let t = anim.transform().w_axis.truncate();
                    // Not posed yet: nothing to draw this frame anyway.
                    let Some(bounds) = self.gpu_skinned.get(*id).map(|g| g.bounds) else {
                        return false;
                    };
                    if !frustum.intersects(bounds) {
                        return false;
                    }
                    // A caster outside a cascade's box cannot shade
                    // anything inside it: the box runs along the light.
                    slot >= 3 || t.distance(focus) <= far + 16.0
                })
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            // Voxel ground is static: drawn with the static casters when the
            // cache is used, so not again on a hit.
            let voxel = (voxel_on && uses[i] != ShadowCacheUse::Hit)
                .then(|| self.voxel.visible_in(frustum));
            views.push((slot, uses[i], view_encoder, skinned, voxel));
        }
        let this = &*self;
        let views: Vec<ShadowViewWork<'_>> = views
            .into_iter()
            .map(|(slot, cache, encoder, skinned, voxel)| ShadowViewWork {
                slot,
                cache,
                encoder,
                skinned: skinned
                    .iter()
                    .filter_map(|id| this.gpu_skinned.get(id))
                    .collect(),
                voxel,
            })
            .collect();
        let record = |work: ShadowViewWork<'_>| -> wgpu::CommandBuffer {
            let ShadowViewWork {
                slot,
                cache,
                mut encoder,
                skinned,
                voxel,
            } = work;
            let (target, view_bind) = if slot < 3 {
                (
                    &this.shadow.layer_views[slot],
                    &this.shadow.cascade_binds[slot],
                )
            } else {
                (
                    &this.shadow.lamps.layer_views[slot - 3],
                    &this.shadow.lamps.layer_binds[slot - 3],
                )
            };
            let draw_voxel = |pass: &mut wgpu::RenderPass<'_>| {
                if let Some(draws) = voxel.as_ref() {
                    pass.set_pipeline(&this.shadow.voxel_pipeline);
                    pass.set_bind_group(0, view_bind, &[]);
                    this.voxel.draw(pass, draws);
                }
            };
            let depth_pass = |encoder: &'_ mut wgpu::CommandEncoder,
                              view: &wgpu::TextureView,
                              load: wgpu::LoadOp<f32>,
                              label: &'static str| {
                encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view,
                            depth_ops: Some(wgpu::Operations {
                                load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    })
                    .forget_lifetime()
            };
            let mut load = wgpu::LoadOp::Clear(1.0);
            if cache != ShadowCacheUse::Plain {
                let cached = this
                    .shadow
                    .static_cache
                    .as_ref()
                    .expect("static shadow cache in use");
                if cache == ShadowCacheUse::Miss {
                    // The static casters into the cache layer first.
                    let mut pass = depth_pass(
                        &mut encoder,
                        &cached.views[slot],
                        wgpu::LoadOp::Clear(1.0),
                        "shadow-csm-static",
                    );
                    if let Some(scene) = this.shadow_scene.as_ref() {
                        draw_shadow_scene(
                            &mut pass,
                            scene,
                            STATIC_SHADOW_SLOT + slot,
                            &this.shadow,
                            view_bind,
                        );
                    }
                    let view = &this.shadow_bundles[STATIC_SHADOW_SLOT + slot];
                    if let Some(multi) = view.multi.as_ref() {
                        multi.encode(&mut pass, &this.shadow, view_bind);
                    }
                    pass.execute_bundles(
                        view.groups
                            .iter()
                            .map(|g| &g.bundle)
                            .chain(view.batches.as_ref()),
                    );
                    draw_voxel(&mut pass);
                    drop(pass);
                }
                shadow::copy_layer(
                    &mut encoder,
                    &cached.texture,
                    &this.shadow.depth_texture,
                    slot as u32,
                );
                load = wgpu::LoadOp::Load;
            }
            let mut pass = depth_pass(
                &mut encoder,
                target,
                load,
                if slot < 3 {
                    "shadow-csm"
                } else {
                    "shadow-lamp"
                },
            );
            if let Some(scene) = this.shadow_scene.as_ref() {
                draw_shadow_scene(&mut pass, scene, slot, &this.shadow, view_bind);
            }
            let view = &this.shadow_bundles[slot];
            if let Some(multi) = view.multi.as_ref() {
                multi.encode(&mut pass, &this.shadow, view_bind);
            }
            pass.execute_bundles(
                view.groups
                    .iter()
                    .map(|g| &g.bundle)
                    .chain(view.batches.as_ref()),
            );
            if cache == ShadowCacheUse::Plain {
                draw_voxel(&mut pass);
            }
            pass.set_pipeline(&this.shadow.skinned_pipeline);
            pass.set_bind_group(0, view_bind, &[]);
            for gpu in skinned {
                pass.set_bind_group(1, &gpu.joint_bind, &[]);
                pass.set_vertex_buffer(1, gpu.instance_buf.slice(..));
                for mesh in &gpu.meshes {
                    pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                    pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
            if slot < 3 && default_space {
                this.crowd.draw_shadow(&mut pass, view_bind);
            }
            drop(pass);
            encoder.finish()
        };
        let buffers: Vec<wgpu::CommandBuffer> = {
            use rayon::prelude::*;
            crate::parallel::frame_pool().install(|| views.into_par_iter().map(record).collect())
        };
        // `collect` keeps the views' order whatever thread finished first.
        recorded.extend(buffers);
    }

    /// Split a cascade's casters into static ones (unchanged for
    /// `STATIC_SHADOW_SETTLE` frames, not swaying in the wind) and moving
    /// ones. Order within each part is kept.
    pub(super) fn split_static_casters(
        &self,
        selection: &ShadowCullSelection,
    ) -> (ShadowCullSelection, ShadowCullSelection) {
        let settled = |id: &EntityId| {
            self.moved_at
                .get(id)
                .is_none_or(|&at| self.frame_no.saturating_sub(at) > STATIC_SHADOW_SETTLE)
        };
        let mesh_static =
            |d: &ShadowDraw| settled(&d.id) && self.gpu_meshes.get(&d.id).is_some_and(|g| !g.sways);
        let batch_static = |id: &EntityId| {
            settled(id)
                && self
                    .gpu_instance_batches
                    .get(id)
                    .is_some_and(|b| !b.gpu.sways)
        };
        let empty = || ShadowCullSelection {
            draws: Vec::new(),
            batch_draws: Vec::new(),
            compact_batches: Vec::new(),
        };
        let (mut fixed, mut moving) = (empty(), empty());
        for d in &selection.draws {
            if mesh_static(d) {
                fixed.draws.push(*d);
            } else {
                moving.draws.push(*d);
            }
        }
        for d in &selection.batch_draws {
            if batch_static(&d.id) {
                fixed.batch_draws.push(*d);
            } else {
                moving.batch_draws.push(*d);
            }
        }
        for id in &selection.compact_batches {
            if batch_static(id) {
                fixed.compact_batches.push(*id);
            } else {
                moving.compact_batches.push(*id);
            }
        }
        (fixed, moving)
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }
}
