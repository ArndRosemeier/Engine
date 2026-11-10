//! Voxel terrain, streaming, water and shattering meshes.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Smooth destructible terrain drawn in the default space.
    pub fn voxel_terrain(&self) -> Option<&crate::voxel_terrain::VoxelTerrain> {
        self.voxel_terrain.as_ref()
    }

    /// Edit the voxel terrain; the renderer re-meshes the changed chunks.
    pub fn voxel_terrain_mut(&mut self) -> Option<&mut crate::voxel_terrain::VoxelTerrain> {
        self.voxel_terrain.as_mut()
    }

    pub fn set_voxel_terrain(&mut self, terrain: Option<crate::voxel_terrain::VoxelTerrain>) {
        self.voxel_baseline_rev = terrain.as_ref().map_or(0, |t| t.volume().rev());
        self.voxel_terrain = terrain;
    }

    /// Stream a procedural world around the camera (see [`crate::stream`]).
    /// Replaces the voxel terrain with the stream's; `None` removes the
    /// stream and everything it loaded.
    pub fn set_stream(&mut self, stream: Option<crate::stream::WorldStream>) -> EngineResult<()> {
        if let Some(mut old) = self.stream.take() {
            old.detach(self);
        }
        if let Some(mut stream) = stream {
            stream.attach(self)?;
            self.stream = Some(Box::new(stream));
        }
        Ok(())
    }

    pub fn stream(&self) -> Option<&crate::stream::WorldStream> {
        self.stream.as_deref()
    }

    pub fn stream_mut(&mut self) -> Option<&mut crate::stream::WorldStream> {
        self.stream.as_deref_mut()
    }

    /// Load around the camera until the ground is there (or `timeout_s`).
    /// Call it once on the first frame so nothing starts in the void.
    pub fn prepare_stream(&mut self, timeout_s: f32) -> bool {
        let Some(mut stream) = self.stream.take() else {
            return true;
        };
        let ready = stream.prepare(self, timeout_s);
        self.stream = Some(stream);
        ready
    }

    /// Advance the world stream. The engine calls this every frame.
    pub fn tick_stream(&mut self) {
        if let Some(mut stream) = self.stream.take() {
            stream.update(self);
            self.stream = Some(stream);
        }
    }

    /// Re-read water columns from the voxel terrain (after a stream moved
    /// the water window).
    pub(crate) fn resync_water_columns(&mut self, changed: &[bool]) {
        if let (Some(water), Some(terrain)) = (self.water.as_mut(), self.voxel_terrain.as_ref()) {
            water.resync_columns(terrain, changed);
        }
    }

    /// Spawn a mesh that was built elsewhere (a worker thread).
    pub(crate) fn spawn_prebuilt(
        &mut self,
        built: BuiltMesh,
        albedo: (
            Option<crate::mesh::AlbedoMap>,
            Option<crate::mesh::MaterialMaps>,
        ),
        place: Place,
    ) -> EngineResult<EntityId> {
        let id = self.spawn_built(built, place.to_matrix());
        if let Err(e) = self.attach_albedo(id, albedo) {
            self.despawn(id);
            return Err(e);
        }
        Ok(id)
    }

    /// The water simulation, if one is set.
    pub fn water(&self) -> Option<&crate::water::Water> {
        self.water.as_ref()
    }

    /// Pour, add sources, change settings. Terrain edits made through
    /// [`World::voxel_terrain_mut`] reach the water by themselves.
    pub fn water_mut(&mut self) -> Option<&mut crate::water::Water> {
        self.water.as_mut()
    }

    /// Set (or remove) the water. The engine steps it every frame and, when
    /// the voxel terrain is dug or built on, re-reads the changed columns so
    /// water flows into new cavities and out of filled ones.
    pub fn set_water(&mut self, water: Option<crate::water::Water>) {
        self.water = water;
        self.water_terrain_rev = self.voxel_terrain.as_ref().map_or(0, |t| t.volume().rev());
    }

    /// Step the water. The engine calls this every frame.
    pub fn tick_water(&mut self, dt: f32) {
        let Some(water) = self.water.as_mut() else {
            return;
        };
        if let Some(terrain) = self.voxel_terrain.as_ref() {
            let volume = terrain.volume();
            if volume.rev() != self.water_terrain_rev {
                let (keys, _) = volume.changed_since(self.water_terrain_rev);
                self.water_terrain_rev = volume.rev();
                let size = volume.chunk_world_size();
                let mut xz: Vec<glam::IVec2> =
                    keys.iter().map(|k| glam::IVec2::new(k.x, k.z)).collect();
                xz.sort_by_key(|k| (k.x, k.y));
                xz.dedup();
                let region = *water.region();
                let mut changed = vec![false; region.column_count()];
                for k in xz {
                    let min = k.as_vec2() * size;
                    crate::water::mark_columns(
                        &region,
                        min,
                        min + glam::Vec2::splat(size),
                        &mut changed,
                    );
                }
                water.resync_columns(terrain, &changed);
            }
        }
        if dt.is_finite() && dt > 0.0 {
            water.update(dt);
        }
    }

    /// Break a mesh entity into solid pieces that fly away from `impact`
    /// (world space), fall, bounce off the ground and settle. Works on any
    /// ordinary mesh (props, rocks, walls, buildings); the original entity is
    /// removed. Returns the pieces' entity ids.
    pub fn shatter(
        &mut self,
        id: EntityId,
        impact: impl Into<Vec3>,
        settings: crate::debris::Shatter,
    ) -> EngineResult<Vec<EntityId>> {
        settings.validate()?;
        let impact = impact.into();
        crate::place::ensure_finite3(impact, "shatter impact")?;
        self.needs_cpu_mesh(id, "shatter")?;
        let entity = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if entity.instanced || entity.instance_of.is_some() {
            return Err(EngineError::InvalidValue(
                "shatter works on single (non-instanced) mesh entities".into(),
            ));
        }
        let fragments = crate::fracture::fracture(
            &entity.mesh,
            entity.transform,
            impact,
            settings.fracture_settings(),
        );
        let dust_color = world_effects::mesh_color(&entity.mesh.colors);
        let dust_size = {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for p in entity.mesh.positions.iter().step_by(7) {
                let w = entity.transform.transform_point3(*p);
                lo = lo.min(w);
                hi = hi.max(w);
            }
            ((hi - lo).max_element() * 0.35).clamp(0.2, 6.0)
        };
        let (material, albedo, casts_shadow, space) = (
            entity.material,
            entity.albedo,
            entity.casts_shadow,
            entity.space,
        );
        self.despawn(id);
        let prev_space = self.spawn_space;
        self.spawn_space = space;
        let mut rng = settings.seed.wrapping_mul(2_654_435_761) | 1;
        let mut rand = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let reach = fragments
            .iter()
            .map(|f| f.centroid.distance(impact))
            .fold(0.0_f32, f32::max)
            .max(0.5);
        let mut ids = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            let piece_id =
                self.spawn_built(fragment.mesh, Mat4::from_translation(fragment.centroid));
            if let Some(e) = self.entities.get_mut(&piece_id) {
                e.material = material;
                e.albedo = albedo;
                e.casts_shadow = casts_shadow;
            }
            let away = (fragment.centroid - impact)
                .try_normalize()
                .unwrap_or(Vec3::Y);
            let falloff = 1.0 - 0.6 * (fragment.centroid.distance(impact) / reach);
            let velocity = (away
                + Vec3::new(rand() * 0.3, 0.35 + rand().abs() * 0.3, rand() * 0.3))
                * settings.speed
                * falloff;
            let spin = Vec3::new(rand(), rand(), rand()) * settings.speed
                / fragment.radius.max(0.15)
                * 0.4;
            let body = crate::physics::Body::dynamic()
                .density(settings.density)
                .friction(0.7)
                .bounce(0.2)
                .velocity(velocity)
                .spin(spin)
                .ccd(fragment.radius < 0.3);
            let positions = &self.entities[&piece_id].mesh.positions;
            let added = self.physics.add(
                piece_id,
                positions,
                Mat4::from_translation(fragment.centroid),
                &body,
                Some(crate::physics::Temporary {
                    keep: settings.keep_seconds.unwrap_or(f32::INFINITY),
                }),
                Vec::new,
            );
            if added.is_err() {
                // Slivers too thin for a hull: drop them rather than leave
                // them hanging in the air.
                self.despawn(piece_id);
                continue;
            }
            ids.push(piece_id);
        }
        self.spawn_space = prev_space;
        self.destruction_burst(impact, dust_size, dust_color, Vec3::Y);
        Ok(ids)
    }

    /// Shattered pieces still in the world (flying, resting or sinking away).
    pub fn debris_count(&self) -> usize {
        self.physics.temporary_count()
    }
}
