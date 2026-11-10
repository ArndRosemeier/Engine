//! Rigid bodies, pushes, blasts, rays and the physics tick.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Give an entity a rigid body: from now on physics moves it (or, for
    /// fixed bodies, other bodies bump into it). Moving the entity with
    /// [`Self::set_place`] still works and teleports the body. Adding a body
    /// to an entity that has one replaces it. See [`crate::physics`].
    ///
    /// A hull or bounding-box body also wraps the meshes hanging on the
    /// entity (attach the wheels first, then add the body).
    ///
    /// A dynamic body on a child (see [`Self::attach`]) lets go of its parent:
    /// physics moves it from then on. Fixed and kinematic children keep
    /// following their parent.
    pub fn add_body(&mut self, id: EntityId, body: crate::physics::Body) -> EngineResult<()> {
        if body.kind == crate::physics::BodyKind::Dynamic && self.hierarchy.links.contains_key(&id)
        {
            self.detach(id)?;
        }
        self.needs_cpu_mesh(id, "add_body")?;
        let e = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if e.instanced || e.instance_of.is_some() {
            return Err(EngineError::InvalidValue(
                "bodies work on single (non-instanced) mesh entities".into(),
            ));
        }
        let mesh = &e.mesh;
        // A hull or box also wraps the meshes hanging on this entity (a
        // cart's wheels), so the whole thing rests on the ground.
        let mut points = std::borrow::Cow::Borrowed(&mesh.positions[..]);
        if matches!(
            body.shape,
            crate::physics::BodyShape::Hull | crate::physics::BodyShape::BoundingBox
        ) && self.hierarchy.children.contains_key(&id)
        {
            let to_local = e.transform.inverse();
            let mut all = mesh.positions.clone();
            for child in self.hierarchy.subtree(id).into_iter().skip(1) {
                if let Some(c) = self.entities.get(&child) {
                    let m = to_local * c.transform;
                    all.extend(c.mesh.positions.iter().map(|p| m.transform_point3(*p)));
                }
            }
            points = std::borrow::Cow::Owned(all);
        }
        self.physics.add(id, &points, e.transform, &body, None, || {
            mesh.indices
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect()
        })
    }

    /// Take an entity's body away; it stays where it is and stops moving.
    pub fn remove_body(&mut self, id: EntityId) {
        self.physics.remove(id);
    }

    pub fn has_body(&self, id: EntityId) -> bool {
        self.physics.contains(id)
    }

    /// Velocity of a body in m/s.
    pub fn body_velocity(&self, id: EntityId) -> Option<Vec3> {
        self.physics.velocity(id)
    }

    pub fn set_body_velocity(
        &mut self,
        id: EntityId,
        velocity: impl Into<Vec3>,
    ) -> EngineResult<()> {
        self.physics.set_velocity(id, velocity.into())
    }

    /// Spin of a body in rad/s around each axis.
    pub fn body_spin(&self, id: EntityId) -> Option<Vec3> {
        self.physics.spin(id)
    }

    pub fn set_body_spin(&mut self, id: EntityId, spin: impl Into<Vec3>) -> EngineResult<()> {
        self.physics.set_spin(id, spin.into())
    }

    /// Mass of a body in kg (volume times density times 1000).
    pub fn body_mass(&self, id: EntityId) -> Option<f32> {
        self.physics.mass(id)
    }

    /// Give a body a kick through its centre. `impulse` is in kg·m/s, so
    /// `direction * speed * world.body_mass(id)` changes its speed by `speed`.
    pub fn push(&mut self, id: EntityId, impulse: impl Into<Vec3>) -> EngineResult<()> {
        self.physics.push(id, impulse.into(), None)
    }

    /// Kick a body at a world point, so it also starts to spin.
    pub fn push_at(
        &mut self,
        id: EntityId,
        impulse: impl Into<Vec3>,
        point: impl Into<Vec3>,
    ) -> EngineResult<()> {
        self.physics.push(id, impulse.into(), Some(point.into()))
    }

    /// Throw every body within `radius` of `center` outward: `speed` m/s at
    /// the centre, fading to nothing at the edge. Mass doesn't matter.
    pub fn blast(&mut self, center: impl Into<Vec3>, radius: f32, speed: f32) {
        let c = center.into();
        if c.is_finite() && radius.is_finite() && radius > 0.0 && speed.is_finite() {
            self.physics.blast(c, radius, speed);
        }
    }

    /// Hard hits since the last call (for impact sounds, damage or dust),
    /// hardest first.
    pub fn take_impacts(&mut self) -> Vec<crate::physics::Impact> {
        self.physics.take_impacts()
    }

    /// First body or piece of terrain along a ray, within `max_distance`.
    pub fn cast_ray(
        &self,
        origin: impl Into<Vec3>,
        direction: impl Into<Vec3>,
        max_distance: f32,
    ) -> Option<crate::physics::RayHit> {
        let (origin, dir) = (origin.into(), direction.into().try_normalize()?);
        if !origin.is_finite() || !max_distance.is_finite() || max_distance <= 0.0 {
            return None;
        }
        let body = self.physics.cast_ray(origin, dir, max_distance);
        let reach = body.map_or(max_distance, |h| h.distance);
        let ground = self.ground_ray(origin, dir, reach);
        match (body, ground) {
            (Some(b), Some(g)) if g.distance < b.distance => Some(g),
            (Some(b), _) => Some(b),
            (None, g) => g,
        }
    }

    pub(super) fn ground_ray(
        &self,
        origin: Vec3,
        dir: Vec3,
        max: f32,
    ) -> Option<crate::physics::RayHit> {
        use crate::physics::{PhysicsGround, RayHit};
        let hit = |p: Vec3, n: Vec3| RayHit {
            entity: None,
            point: p,
            normal: n,
            distance: (p - origin).length(),
        };
        let plane = |y: f32| {
            (dir.y.abs() > 1e-6)
                .then(|| (y - origin.y) / dir.y)
                .filter(|t| *t >= 0.0 && *t <= max)
                .map(|t| hit(origin + dir * t, Vec3::Y))
        };
        match self.physics.settings().ground {
            PhysicsGround::None => None,
            PhysicsGround::Plane(y) => plane(y),
            PhysicsGround::Auto => {
                if let Some(v) = self.voxel_terrain.as_ref() {
                    if v.volume().chunk_keys().next().is_some() {
                        return v.raycast(origin, dir, max).map(|p| hit(p, v.normal_at(p)));
                    }
                }
                if let Some(f) = self.height_field.as_ref() {
                    return crate::physics::height_ray(&|x, z| f.height_at(x, z), origin, dir, max)
                        .map(|(p, n)| hit(p, n));
                }
                plane(0.0)
            }
        }
    }

    // ---- Triggers and shape queries ----

    /// Give an entity (a mesh, a pivot, anything with a place) an invisible
    /// trigger zone that follows it. Read who went in and out with
    /// [`Self::take_trigger_events`], or who is inside with
    /// [`Self::in_trigger`]. Replaces a trigger the entity already has; it
    /// goes away when the entity is despawned.
    ///
    /// ```ignore
    /// let pad = world.spawn_trigger(Place::new(0.0, 1.0, 5.0), Trigger::cuboid((1.0, 1.0, 1.0)))?;
    /// for e in world.take_trigger_events() {
    ///     if e.trigger == pad && e.entered { /* something stepped on the pad */ }
    /// }
    /// ```
    pub fn add_trigger(
        &mut self,
        id: EntityId,
        trigger: crate::physics::Trigger,
    ) -> EngineResult<()> {
        let m = self.world_transform(id)?;
        self.physics.add_trigger(id, &trigger, m)
    }

    /// A new invisible entity at `place` carrying a trigger zone (a pivot:
    /// move it with [`Self::set_place`], hang it on things with
    /// [`Self::attach`]).
    pub fn spawn_trigger(
        &mut self,
        place: Place,
        trigger: crate::physics::Trigger,
    ) -> EngineResult<EntityId> {
        trigger.validate()?;
        let id = self.spawn_pivot(place);
        self.add_trigger(id, trigger)?;
        Ok(id)
    }

    pub fn remove_trigger(&mut self, id: EntityId) {
        self.physics.remove_trigger(id);
    }

    pub fn has_trigger(&self, id: EntityId) -> bool {
        self.physics.has_trigger(id)
    }

    /// How many trigger zones there are.
    pub fn trigger_count(&self) -> usize {
        self.physics.trigger_count()
    }

    /// Bodies inside a trigger after the last physics step, in id order.
    pub fn in_trigger(&self, trigger: EntityId) -> Vec<EntityId> {
        self.physics.in_trigger(trigger)
    }

    /// Everything that went into or came out of a trigger since the last
    /// call, in the order it happened. Call it every frame.
    pub fn take_trigger_events(&mut self) -> Vec<crate::physics::TriggerEvent> {
        self.physics.take_trigger_events()
    }

    /// Entities with a body (or a ragdoll) touching a sphere, in id order.
    /// Triggers and the ground are left out. Sees bodies as of the last
    /// physics step.
    pub fn overlap_sphere(&self, center: impl Into<Vec3>, radius: f32) -> Vec<EntityId> {
        let c = center.into();
        if !c.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return Vec::new();
        }
        self.physics.overlap_ball(c, radius)
    }

    /// Entities with a body touching an axis-aligned box, in id order.
    pub fn overlap_box(
        &self,
        center: impl Into<Vec3>,
        half_extents: impl Into<Vec3>,
    ) -> Vec<EntityId> {
        let (c, h) = (center.into(), half_extents.into());
        if !c.is_finite() || !h.is_finite() || h.min_element() <= 0.0 {
            return Vec::new();
        }
        self.physics.overlap_box(c, h, glam::Quat::IDENTITY)
    }

    /// Sweep a ball of `radius` from `origin` along `direction`: the first
    /// body or ground it touches within `max_distance`. A fat ray, for
    /// thrown things, "can I fit through here" and aiming help.
    /// `distance` is how far the ball's centre got; `point` is where it
    /// touched. Height and voxel ground count as locally flat.
    pub fn cast_sphere(
        &self,
        origin: impl Into<Vec3>,
        direction: impl Into<Vec3>,
        radius: f32,
        max_distance: f32,
    ) -> Option<crate::physics::RayHit> {
        let (origin, dir) = (origin.into(), direction.into().try_normalize()?);
        if !origin.is_finite()
            || !max_distance.is_finite()
            || max_distance <= 0.0
            || !radius.is_finite()
            || radius <= 0.0
        {
            return None;
        }
        let body = self.physics.cast_ball(origin, dir, radius, max_distance);
        // The ground: a ray from the centre, pulled back by how far the
        // ball reaches towards the surface along the ray.
        let ground = self
            .ground_ray(origin, dir, max_distance + radius * 8.0)
            .and_then(|g| {
                let facing = (-dir.dot(g.normal)).max(0.125);
                let distance = (g.distance - radius / facing).max(0.0);
                (distance <= max_distance).then(|| crate::physics::RayHit {
                    point: origin + dir * distance - g.normal * radius,
                    distance,
                    ..g
                })
            });
        match (body, ground) {
            (Some(b), Some(g)) if g.distance < b.distance => Some(g),
            (Some(b), _) => Some(b),
            (None, g) => g,
        }
    }

    pub fn physics_settings(&self) -> crate::physics::PhysicsSettings {
        self.physics.settings()
    }

    pub fn set_physics_settings(
        &mut self,
        settings: crate::physics::PhysicsSettings,
    ) -> EngineResult<()> {
        settings.validate()?;
        self.physics.set_settings(settings);
        Ok(())
    }

    /// Bodies in the world, and how many of them are moving (not asleep).
    pub fn body_count(&self) -> (usize, usize) {
        (self.physics.body_count(), self.physics.awake_count())
    }

    /// Step physics. The engine calls this every frame.
    pub fn tick_physics(&mut self, dt: f32) {
        for id in self.physics.trigger_ids() {
            let m = self.world_transform(id).ok();
            self.physics.move_trigger(id, m);
        }
        let field = self.height_field.as_ref();
        let height = |x: f32, z: f32| field.map_or(0.0, |f| f.height_at(x, z));
        let around = crate::physics::Surroundings {
            voxels: self.voxel_terrain.as_ref(),
            heights: field.map(|f| (f as *const _ as usize, &height as crate::physics::HeightFn)),
            water: self.water.as_ref(),
        };
        let entities = &self.entities;
        let impacts_before = self.physics.pending_impacts().len();
        let result = self
            .physics
            .tick(dt, &around, |id| entities.get(&id).map(|e| e.transform));
        if let Some(water) = self.water.as_mut() {
            for s in &result.splashes {
                water.disturb(s.at, 1.5, (s.speed * 0.3).min(4.0));
            }
        }
        if self.audio.auto_sounds {
            let splashes: Vec<(Vec3, f32)> =
                result.splashes.iter().map(|s| (s.at, s.speed)).collect();
            self.sound_splashes(&splashes);
            let pending = self.physics.pending_impacts();
            let new = pending.get(impacts_before..).unwrap_or(pending).to_vec();
            self.sound_impacts(&new, dt);
        }
        let mut changed = false;
        for (id, m) in result.moved {
            if let Some(e) = self.entities.get_mut(&id) {
                e.transform = m;
                e.bump_xform();
                changed = true;
            }
        }
        for (id, dy) in result.sink {
            if let Some(e) = self.entities.get_mut(&id) {
                e.transform.w_axis.y -= dy;
                e.bump_xform();
                changed = true;
            }
        }
        if changed {
            self.bump_render_epoch();
        }
        for id in result.expired {
            self.despawn(id);
        }
    }
}
