//! Entities, chunks, textures, materials, instances, camera helpers and procedural terrain.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    pub fn entity(&self, id: EntityId) -> EngineResult<&Entity> {
        self.entities.get(&id).ok_or(EngineError::UnknownEntity)
    }

    pub(crate) fn entities(&self) -> impl Iterator<Item = (EntityId, &Entity)> {
        self.order
            .iter()
            .filter_map(|id| self.entities.get(id).map(|e| (*id, e)))
    }

    pub(crate) fn contains_entity(&self, id: EntityId) -> bool {
        self.entities.contains_key(&id)
    }

    pub(crate) fn contains_animated(&self, id: EntityId) -> bool {
        self.animated.contains_key(&id)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Replace or create a streamed chunk mesh (infinite terrain / LOD).
    pub fn set_chunk(&mut self, key: IVec3, mesh: Mesh) -> EntityId {
        self.set_chunk_built(key, mesh.build())
    }

    /// Like [`set_chunk`] but accepts a pre-built mesh (for background generation).
    pub fn set_chunk_built(&mut self, key: IVec3, built: BuiltMesh) -> EntityId {
        if let Some(&id) = self.chunk_entities.get(&key) {
            // Remesh requires a new GPU upload ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â swap entity id so renderer rebuilds.
            self.despawn(id);
            self.chunk_entities.remove(&key);
        }
        let material = self.chunk_material(&built);
        let id = self.spawn_built(built, Mat4::IDENTITY);
        if material.is_some() {
            self.entities.get_mut(&id).expect("just spawned").material = material;
        }
        self.chunk_entities.insert(key, id);
        id
    }

    /// Pick the default material for a streamed layer from its opacity.
    ///
    /// A fully opaque layer is ground; a fully translucent one is a water
    /// sheet. Mixed layers are left to the plain lit pipeline.
    pub(super) fn chunk_material(&self, built: &BuiltMesh) -> Option<SurfaceMaterialRef> {
        if built.index_count() == 0 {
            return None;
        }
        if built.opaque_index_count == built.index_count() {
            return self
                .default_terrain_material
                .map(SurfaceMaterialRef::Terrain);
        }
        if built.opaque_index_count == 0 {
            return self.default_water_material.map(SurfaceMaterialRef::Water);
        }
        None
    }

    /// Create an RGBA8 texture from raw bytes (`width * height * 4`).
    pub fn create_texture_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> EngineResult<TextureId> {
        if width == 0 || height == 0 {
            return Err(EngineError::InvalidValue(
                "texture size must be non-zero".into(),
            ));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| EngineError::InvalidValue("texture dimensions overflow".into()))?;
        if rgba.len() != expected {
            return Err(EngineError::InvalidValue(format!(
                "texture rgba len {} != {}x{}x4",
                rgba.len(),
                width,
                height
            )));
        }
        let id = TextureId(self.next_texture_id);
        self.next_texture_id += 1;
        self.textures.insert(
            id,
            CpuTexture {
                width,
                height,
                rgba,
            },
        );
        self.bump_resource_epoch();
        Ok(id)
    }

    /// Load a PNG from disk into an RGBA8 texture.
    pub fn load_texture_png(&mut self, path: impl AsRef<Path>) -> EngineResult<TextureId> {
        let (w, h, rgba) = load_rgba8_png(path)?;
        self.create_texture_rgba(w, h, rgba)
    }

    /// Create a built-in tileable terrain albedo.
    pub fn create_terrain_albedo(
        &mut self,
        kind: TerrainAlbedo,
        size: u32,
        seed: u32,
    ) -> EngineResult<TextureId> {
        let (w, h, rgba) = generate_terrain_albedo(kind, size, seed);
        self.create_texture_rgba(w, h, rgba)
    }

    /// Create a lush/dry/moor/sand/rock terrain material (world-XZ sampling).
    pub fn create_terrain_material(
        &mut self,
        desc: TerrainMaterialDesc,
    ) -> EngineResult<MaterialId> {
        for tid in [
            desc.grass,
            desc.grass_dry,
            desc.grass_moor,
            desc.mud,
            desc.tundra,
            desc.scree,
            desc.sand,
            desc.rock,
        ] {
            if !self.textures.contains_key(&tid) {
                return Err(EngineError::UnknownTexture);
            }
        }
        if desc.metres_per_tile <= 0.0 {
            return Err(EngineError::InvalidValue(
                "metres_per_tile must be > 0".into(),
            ));
        }
        let id = MaterialId(self.next_material_id);
        self.next_material_id += 1;
        self.materials.insert(id, TerrainMaterial { desc });
        self.bump_resource_epoch();
        Ok(id)
    }

    /// Fully-opaque chunk uploads (`set_chunk_built`) receive this material.
    pub fn set_default_terrain_material(&mut self, material: Option<MaterialId>) {
        self.default_terrain_material = material;
    }

    /// Create an animated water material.
    pub fn create_water_material(
        &mut self,
        desc: WaterMaterialDesc,
    ) -> EngineResult<WaterMaterialId> {
        if desc.depth_scale_m <= 0.0 || desc.wave_length_m <= 0.0 {
            return Err(EngineError::InvalidValue(
                "water depth scale and wave length must be > 0".into(),
            ));
        }
        let id = WaterMaterialId(self.next_material_id);
        self.next_material_id += 1;
        self.water_materials.insert(id, WaterMaterial { desc });
        self.bump_resource_epoch();
        Ok(id)
    }

    /// Fully-translucent chunk layers receive this material.
    pub fn set_default_water_material(&mut self, material: Option<WaterMaterialId>) {
        self.default_water_material = material;
    }

    pub fn set_entity_material(
        &mut self,
        id: EntityId,
        material: Option<SurfaceMaterialRef>,
    ) -> EngineResult<()> {
        match material {
            Some(SurfaceMaterialRef::Terrain(mat)) if !self.materials.contains_key(&mat) => {
                return Err(EngineError::UnknownMaterial)
            }
            Some(SurfaceMaterialRef::Water(mat)) if !self.water_materials.contains_key(&mat) => {
                return Err(EngineError::UnknownMaterial)
            }
            Some(SurfaceMaterialRef::Custom(mat)) if !self.custom_materials.contains_key(&mat) => {
                return Err(EngineError::UnknownMaterial)
            }
            _ => {}
        }
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if e.material == material {
            return Ok(());
        }
        e.material = material;
        self.bump_render_epoch();
        Ok(())
    }

    /// Replace the placements of an instanced entity (see [`Self::spawn_many`]).
    ///
    /// Only the instance buffer is rewritten, so a scatter layer can follow the
    /// streamed world every frame without re-uploading its mesh.
    pub fn set_instances(&mut self, id: EntityId, places: &[Place]) -> EngineResult<()> {
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "set_instances has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if !e.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} was not spawned instanced; use set_place"
            )));
        }
        e.instances.clear();
        e.instances
            .extend(places.iter().copied().map(Place::to_instance));
        e.bump_xform();
        self.bump_render_epoch();
        Ok(())
    }

    /// Reserve room in an instanced entity's GPU batch slot without drawing
    /// placeholder instances.
    ///
    /// Streaming systems should reserve their known per-cell maximum before
    /// filling a reusable entity. Later instance-count changes within this
    /// capacity remain partial writes instead of rebuilding the whole batch.
    pub fn reserve_instances(&mut self, id: EntityId, minimum_capacity: usize) -> EngineResult<()> {
        if minimum_capacity as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "reserve_instances requested {minimum_capacity} instances (limit {})",
                self.limits.max_instances_per_spawn
            )));
        }
        let entity = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if !entity.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} was not spawned instanced; it cannot reserve instances"
            )));
        }
        if minimum_capacity <= entity.instance_reserve {
            return Ok(());
        }
        entity.instance_reserve = minimum_capacity;
        self.bump_render_epoch();
        Ok(())
    }

    /// Seconds since start, for materials that animate.
    pub fn set_time(&mut self, seconds: f32) {
        self.time = seconds;
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub(crate) fn textures(&self) -> &HashMap<TextureId, CpuTexture> {
        &self.textures
    }

    pub(crate) fn surface_maps(&self, id: TextureId) -> Option<&std::sync::Arc<MaterialMaps>> {
        self.surface_maps.get(&id)
    }

    pub(crate) fn materials(&self) -> &HashMap<MaterialId, TerrainMaterial> {
        &self.materials
    }

    pub(crate) fn water_materials(&self) -> &HashMap<WaterMaterialId, WaterMaterial> {
        &self.water_materials
    }

    /// Remove a streamed chunk if present.
    pub fn clear_chunk(&mut self, key: IVec3) {
        if let Some(id) = self.chunk_entities.remove(&key) {
            self.despawn(id);
        }
    }

    pub fn has_chunk(&self, key: IVec3) -> bool {
        self.chunk_entities.contains_key(&key)
    }

    /// First-person camera (yaw in degrees, 0 = +Z; pitch positive is up).
    pub fn look_first_person(
        &mut self,
        eye: impl Into<Vec3>,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) {
        self.camera = Camera::first_person(eye, yaw_degrees, pitch_degrees);
        self.camera = self.camera.clone().with_lens(
            self.camera.fov_y_degrees(),
            self.camera.near(),
            self.view_distance,
        );
    }

    /// Third-person follow camera (yaw in degrees, 0 = +Z).
    pub fn look_follow(
        &mut self,
        target: impl Into<Vec3>,
        yaw_degrees: f32,
        distance: f32,
        height: f32,
    ) {
        self.camera = Camera::follow(target, yaw_degrees, distance, height);
        self.camera = self.camera.clone().with_lens(
            self.camera.fov_y_degrees(),
            self.camera.near(),
            self.view_distance,
        );
    }

    /// How far the camera helpers may see, in metres.
    ///
    /// The helpers rebuild the camera every frame, so a game that reaches into
    /// `world.camera.far()` has it overwritten on the next one; this is the knob
    /// that survives. Depth is reversed, so a horizon-scale distance costs
    /// nothing in precision.
    pub fn set_view_distance(&mut self, metres: f32) -> EngineResult<()> {
        if !(metres.is_finite() && metres > self.camera.near()) {
            return Err(EngineError::InvalidValue(format!(
                "view distance must be finite and beyond the near plane, got {metres}"
            )));
        }
        self.view_distance = metres;
        self.camera =
            self.camera
                .clone()
                .with_lens(self.camera.fov_y_degrees(), self.camera.near(), metres);
        Ok(())
    }

    pub fn view_distance(&self) -> f32 {
        self.view_distance
    }

    /// Enable GPU procedural terrain (clipmap). Replaces any previous proc terrain.
    /// Its texture layers start generating in the background right away, so
    /// they are ready by the first frame.
    pub fn set_proc_terrain(&mut self, terrain: ProcTerrain) {
        if let Some(layers) = &terrain.layers {
            let list: Vec<crate::texgen::Layer> =
                layers.as_array().iter().map(|&l| l.into()).collect();
            crate::texgen::prepare_in_background(&list);
        }
        self.height_field = Some(HeightField::new(terrain.rules.clone()));
        self.proc_terrain = Some(terrain);
    }

    /// Clear GPU procedural terrain (mesh-only mode).
    pub fn clear_proc_terrain(&mut self) {
        self.proc_terrain = None;
        self.height_field = None;
    }

    /// Update clipmap focus (usually the walker position).
    pub fn set_proc_focus(&mut self, focus: impl Into<Vec3>) {
        let focus = focus.into();
        if let Some(t) = self.proc_terrain.as_mut() {
            t.focus = focus;
        }
    }

    /// Height from the GPU terrain formula (CPU portable noise). Falls back to 0.
    pub fn proc_height_at(&self, x: f32, z: f32) -> f32 {
        self.height_field
            .as_ref()
            .map(|f| f.height_at(x, z))
            .unwrap_or(0.0)
    }

    /// Walkable height matching the rendered clipmap surface (finest-ring triangles).
    pub fn proc_walk_height(&self, x: f32, z: f32) -> f32 {
        match (&self.height_field, &self.proc_terrain) {
            (Some(field), Some(proc)) => {
                field.walk_height_on_clipmap(x, z, &proc.config, proc.focus)
            }
            (Some(field), None) => field.height_at(x, z),
            _ => 0.0,
        }
    }

    pub(crate) fn proc_terrain(&self) -> Option<&ProcTerrain> {
        self.proc_terrain.as_ref()
    }

    #[allow(dead_code)]
    pub(crate) fn height_field(&self) -> Option<&HeightField> {
        self.height_field.as_ref()
    }
}
