//! Spawning and despawning meshes, instancing, render origin, anchored chunks, pointer and screenshot requests.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Spawn a mesh at a friendly [`Place`].
    pub fn place(&mut self, mesh: Mesh, place: Place) -> EngineResult<EntityId> {
        self.place_prepared(mesh.prepare(), place)
    }

    /// Spawn a mesh made ready on another thread with [`Mesh::prepare`]:
    /// only the bookkeeping is left for this frame.
    pub fn spawn_prepared(&mut self, mesh: PreparedMesh) -> EntityId {
        self.place_prepared(mesh, Place::default())
            .expect("spawn at identity")
    }

    /// [`Self::spawn_prepared`] at `place`.
    pub fn place_prepared(&mut self, mesh: PreparedMesh, place: Place) -> EngineResult<EntityId> {
        let PreparedMesh {
            built,
            albedo,
            maps,
            key,
        } = mesh;
        let id = self.spawn_built(built, place.to_matrix());
        self.attach_albedo_keyed(id, (albedo, maps), key)?;
        Ok(id)
    }

    /// Spawn one mesh at many places (GPU instancing).
    pub fn spawn_many(
        &mut self,
        mesh: Mesh,
        places: impl Into<Vec<Place>>,
    ) -> EngineResult<EntityId> {
        let places = places.into();
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "spawn_many has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let instances: Vec<MeshInstance> = places.into_iter().map(Place::to_instance).collect();
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built_instanced(mesh.build(), instances);
        self.attach_albedo(id, albedo)?;
        Ok(id)
    }

    /// Spawn an instanced entity that starts with nothing placed.
    ///
    /// The mesh is uploaded once; [`Self::set_instances`] then drives where it
    /// appears.
    pub fn spawn_instanced(&mut self, mesh: Mesh) -> EntityId {
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built_instanced(mesh.build(), Vec::new());
        self.attach_albedo(id, albedo)
            .expect("albedo upload at spawn_instanced");
        id
    }

    /// Another instanced entity that draws `prototype`'s mesh and albedo.
    ///
    /// The prototype must itself own a mesh (not be a like-entity). This one
    /// starts with no placements; [`Self::set_instances`] fills them.
    pub fn spawn_instanced_like(&mut self, prototype: EntityId) -> EngineResult<EntityId> {
        let src = self
            .entities
            .get(&prototype)
            .ok_or(EngineError::UnknownEntity)?;
        if !src.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {prototype} was not spawned instanced"
            )));
        }
        if src.instance_of.is_some() {
            return Err(EngineError::InvalidValue(format!(
                "entity {prototype} is already a like-entity; instance the prototype"
            )));
        }
        let albedo = src.albedo;
        let material = src.material;
        let casts_shadow = src.casts_shadow;
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh: BuiltMesh::default(),
                released: None,
                transform: Mat4::IDENTITY,
                instances: Vec::new(),
                instance_reserve: 0,
                instanced: true,
                material,
                albedo,
                xform_rev: 1,
                instance_of: Some(prototype),
                casts_shadow,
                space: self.spawn_space,
            },
        );
        self.order.push(id);
        self.bump_render_epoch();
        Ok(id)
    }

    pub(crate) fn spawn_built(&mut self, mesh: BuiltMesh, transform: Mat4) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh,
                released: None,
                transform,
                instances: Vec::new(),
                instance_reserve: 0,
                instanced: false,
                material: None,
                albedo: None,
                xform_rev: 1,
                instance_of: None,
                casts_shadow: true,
                space: self.spawn_space,
            },
        );
        self.order.push(id);
        self.cpu_meshes.spawned(id);
        self.spawned_plain(id);
        id
    }

    pub(crate) fn spawn_built_instanced(
        &mut self,
        mesh: BuiltMesh,
        instances: Vec<MeshInstance>,
    ) -> EntityId {
        let instance_reserve = instances.len();
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh,
                released: None,
                transform: Mat4::IDENTITY,
                instances,
                instance_reserve,
                instanced: true,
                material: None,
                albedo: None,
                xform_rev: 1,
                instance_of: None,
                casts_shadow: true,
                space: self.spawn_space,
            },
        );
        self.order.push(id);
        self.cpu_meshes.spawned(id);
        self.bump_render_epoch();
        id
    }

    /// Grass and far stand-ins should not enter the cascaded depth maps.
    pub fn set_casts_shadow(&mut self, id: EntityId, casts: bool) -> EngineResult<()> {
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if e.casts_shadow == casts {
            return Ok(());
        }
        e.casts_shadow = casts;
        self.redraw(id);
        Ok(())
    }

    pub(super) fn attach_albedo(
        &mut self,
        id: EntityId,
        textures: (Option<AlbedoMap>, Option<MaterialMaps>),
    ) -> EngineResult<()> {
        self.attach_albedo_by_content(id, textures, false)
    }

    /// [`Self::attach_albedo`]; `share` looks every map up by content (not
    /// only large ones), for textures many entities reuse.
    pub(crate) fn attach_albedo_by_content(
        &mut self,
        id: EntityId,
        (albedo, maps): (Option<AlbedoMap>, Option<MaterialMaps>),
        share: bool,
    ) -> EngineResult<()> {
        let key = texture_key(albedo.as_ref(), maps.as_ref(), share);
        self.attach_albedo_keyed(id, (albedo, maps), key)
    }

    /// [`Self::attach_albedo_by_content`] with the content key worked out
    /// already (see [`texture_key`]).
    pub(super) fn attach_albedo_keyed(
        &mut self,
        id: EntityId,
        (albedo, maps): (Option<AlbedoMap>, Option<MaterialMaps>),
        key: Option<u64>,
    ) -> EngineResult<()> {
        let map = match (albedo, &maps) {
            (Some(map), _) => map,
            // Maps need an albedo to ride along with: a white texel.
            (None, Some(_)) => white_texel(),
            (None, None) => return Ok(()),
        };
        let shared = key
            .and_then(|k| self.texture_by_content.get(&k).copied())
            .filter(|tid| self.textures.contains_key(tid));
        let tid = match shared {
            Some(tid) => tid,
            None => {
                let tid = self.create_texture_rgba(map.width, map.height, map.rgba.into_vec())?;
                if let Some(maps) = maps {
                    self.surface_maps.insert(tid, std::sync::Arc::new(maps));
                }
                if let Some(k) = key {
                    self.texture_by_content.insert(k, tid);
                }
                self.auto_textures.insert(tid);
                tid
            }
        };
        let entity = self.entities.get_mut(&id).expect("entity was just spawned");
        if let Some(old) = entity.albedo.replace(tid) {
            if old != tid && self.auto_textures.contains(&old) {
                self.texture_sweep = true;
            }
        }
        self.redraw(id);
        Ok(())
    }

    /// Remove an entity, its children (and theirs) and the lamps riding on
    /// them. [`Self::detach`] a child first to keep it.
    pub fn despawn(&mut self, id: EntityId) {
        self.components.forget(id);
        self.forget_ground(id);
        self.cpu_meshes.forget(id);
        self.sim_poses.remove(&id);
        self.entity_names.remove(&id);
        self.forget_ragdoll(id);
        self.entity_rooms.remove(&id);
        if self.buildings.contains_key(&id) || self.falling.iter().any(|p| p.building == id) {
            self.forget_building(id);
        }
        if let Some(made) = self.prefab_instances.remove(&id) {
            for lamp in made.lamps {
                self.remove_lamp(lamp);
            }
            for part in made.parts {
                self.despawn(part);
            }
        }
        let (children, lamps) = self.hierarchy.forget(id);
        for lamp in lamps {
            self.lamps.remove(&lamp);
        }
        for child in children {
            self.despawn(child);
        }
        self.physics.remove(id);
        let removed = self.entities.remove(&id);
        let removed_static = removed.is_some();
        // A plain mesh the renderer can drop on its own (see `despawned`).
        let plain = removed
            .as_ref()
            .is_some_and(|e| !e.instanced && e.instance_of.is_none());
        if let Some(tid) = removed.and_then(|e| e.albedo) {
            if self.auto_textures.contains(&tid) {
                self.texture_sweep = true;
            }
        }
        if removed_static {
            // `entities()` skips ids that are gone; compact now and then.
            self.order_stale += 1;
            if self.order_stale > 256 && self.order_stale * 2 > self.order.len() {
                let entities = &self.entities;
                self.order.retain(|x| entities.contains_key(x));
                self.order_stale = 0;
            }
        }
        self.animated.remove(&id);
        self.animated_order.retain(|x| *x != id);
        self.chunk_entities.retain(|_, eid| *eid != id);
        self.anchored_entities.remove(&id);
        self.anchored_chunks.retain(|_, c| c.entity != id);
        let linked = self.portals.iter().any(|p| p.a() == id || p.b() == id);
        self.portals.retain(|p| p.a() != id && p.b() != id);
        if plain && !linked {
            self.despawned_plain(id);
        } else if removed_static || linked {
            self.bump_render_epoch();
        }
    }

    /// Horizontal anchor that render-space coordinates are measured from.
    pub fn render_origin(&self) -> RenderOrigin {
        self.render_origin
    }

    /// Rebase render space onto a new origin.
    ///
    /// Every anchored entity and chunk transform is recomputed from its
    /// immutable global anchor, so repeated rebases never accumulate drift.
    /// Unanchored content (plain [`Self::spawn`] / [`Self::set_chunk_built`])
    /// is left alone: it was authored directly in render space.
    pub fn set_render_origin(&mut self, origin: RenderOrigin) -> EngineResult<()> {
        if self.render_origin == origin {
            return Ok(());
        }
        self.render_origin = origin;
        let places: Vec<(EntityId, GlobalPlace)> = self
            .anchored_entities
            .iter()
            .map(|(id, place)| (*id, *place))
            .collect();
        for (id, place) in places {
            let local = place.to_place(origin)?;
            let e = self
                .entities
                .get_mut(&id)
                .ok_or(EngineError::UnknownEntity)?;
            e.transform = local.to_matrix();
            e.bump_xform();
        }
        let chunks: Vec<AnchoredChunk> = self.anchored_chunks.values().copied().collect();
        for chunk in chunks {
            let offset = chunk.anchor.to_render(origin)?.vec3();
            let e = self
                .entities
                .get_mut(&chunk.entity)
                .ok_or(EngineError::UnknownEntity)?;
            e.transform = Mat4::from_translation(offset);
            e.bump_xform();
        }
        self.bump_render_epoch();
        Ok(())
    }

    /// Render-space position of a global point under the active origin.
    pub fn to_render(&self, position: GlobalPosition) -> EngineResult<Vec3> {
        Ok(position.to_render(self.render_origin)?.vec3())
    }

    /// Global position of a render-space point under the active origin.
    pub fn to_global(&self, position: Vec3) -> EngineResult<GlobalPosition> {
        Ok(crate::space::RenderPosition::new(position)?.to_global(self.render_origin))
    }

    /// Spawn a mesh anchored in absolute world metres (survives rebasing).
    pub fn spawn_anchored(&mut self, mesh: Mesh, place: GlobalPlace) -> EngineResult<EntityId> {
        let local = place.to_place(self.render_origin)?;
        let albedo = (mesh.albedo().cloned(), mesh.material_maps().cloned());
        let id = self.spawn_built(mesh.build(), local.to_matrix());
        self.attach_albedo(id, albedo)?;
        self.anchored_entities.insert(id, place);
        Ok(id)
    }

    /// Move an anchored entity to a new global place.
    pub fn set_anchored_place(&mut self, id: EntityId, place: GlobalPlace) -> EngineResult<()> {
        if !self.anchored_entities.contains_key(&id) {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} is not anchored; use set_place for render-space entities"
            )));
        }
        let local = place.to_place(self.render_origin)?;
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.transform = local.to_matrix();
        e.bump_xform();
        self.anchored_entities.insert(id, place);
        self.touch(id);
        Ok(())
    }

    /// Upload a streamed chunk whose vertices are relative to `anchor`.
    ///
    /// Chunk-local vertices keep `f32` mesh precision independent of how far the
    /// chunk sits from the world origin.
    pub fn set_anchored_chunk(
        &mut self,
        id: ChunkId,
        anchor: GlobalPosition,
        built: BuiltMesh,
    ) -> EngineResult<EntityId> {
        let offset = anchor.to_render(self.render_origin)?.vec3();
        if let Some(existing) = self.anchored_chunks.remove(&id) {
            // Remesh needs a new GPU upload ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â swap entity id so the renderer rebuilds.
            self.despawn(existing.entity);
        }
        let material = self.chunk_material(&built);
        let entity = self.spawn_built(built, Mat4::from_translation(offset));
        if material.is_some() {
            self.entities
                .get_mut(&entity)
                .expect("just spawned")
                .material = material;
        }
        self.anchored_chunks
            .insert(id, AnchoredChunk { entity, anchor });
        Ok(entity)
    }

    pub fn clear_anchored_chunk(&mut self, id: ChunkId) {
        if let Some(chunk) = self.anchored_chunks.remove(&id) {
            self.despawn(chunk.entity);
        }
    }

    pub fn has_anchored_chunk(&self, id: ChunkId) -> bool {
        self.anchored_chunks.contains_key(&id)
    }

    pub fn anchored_chunk_count(&self) -> usize {
        self.anchored_chunks.len()
    }

    /// Drop every anchored chunk (leaving a world / regenerating).
    pub fn clear_anchored_chunks(&mut self) {
        let ids: Vec<ChunkId> = self.anchored_chunks.keys().copied().collect();
        for id in ids {
            self.clear_anchored_chunk(id);
        }
    }

    /// Pin the pointer to the window and hide it, for mouse-look.
    ///
    /// While locked, look deltas arrive as [`crate::input::Input::mouse_delta`]
    /// and the UI no longer steals input on hover. Release it before showing a
    /// menu or map the player has to click.
    ///
    /// This is a request, not a guarantee: the window drops the grab when it
    /// loses focus and when Escape is pressed, and clears the flag with it, so
    /// a game that wants the pointer back has to ask again. Nothing should pin
    /// the cursor without the player having asked for it ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a grab taken on
    /// startup follows them out of the window.
    pub fn set_pointer_lock(&mut self, locked: bool) {
        self.pointer_lock = locked;
    }

    pub fn pointer_lock(&self) -> bool {
        self.pointer_lock
    }

    /// The ray under the mouse cursor, `(origin, unit direction)`, for
    /// picking, digging and placing. With the pointer locked or the cursor
    /// outside the window it is the ray through the screen centre.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World, frame: &Frame) {
    /// let (eye, dir) = world.cursor_ray(frame);
    /// if let Some(hit) = world.voxel_terrain().and_then(|t| t.raycast(eye, dir, 300.0)) {
    ///     world.voxel_terrain_mut().unwrap().dig_sphere(hit, 3.0);
    /// }
    /// # }
    /// ```
    pub fn cursor_ray(&self, frame: &Frame) -> (Vec3, Vec3) {
        let (w, h) = (frame.width as f32, frame.height as f32);
        let at = frame
            .input
            .cursor()
            .filter(|_| !self.pointer_lock)
            .unwrap_or(glam::Vec2::new(w * 0.5, h * 0.5));
        self.camera().ray_through(at.x, at.y, w, h)
    }

    /// While true, Escape cancels bind-listen only: no modal close, unlock, or quit.
    pub fn set_bind_listen(&mut self, listening: bool) {
        self.bind_listen = listening;
    }

    pub fn bind_listen(&self) -> bool {
        self.bind_listen
    }

    /// Capture `path` as a PNG after this frame has been drawn.
    pub fn queue_screenshot(&mut self, path: impl AsRef<Path>) {
        self.screenshot_queue.push(path.as_ref().to_path_buf());
    }

    /// Tell a waiting screenshot harness that the scene is ready.
    pub fn mark_ready(&mut self) {
        self.ready = true;
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    /// Close the window after this frame, once any queued shots have been written.
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    pub(crate) fn has_screenshot_queued(&self) -> bool {
        !self.screenshot_queue.is_empty()
    }

    pub(crate) fn take_screenshot_queue(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.screenshot_queue)
    }

    pub(crate) fn take_exit_requested(&mut self) -> bool {
        let wanted = self.exit_requested;
        self.exit_requested = false;
        wanted
    }
}
