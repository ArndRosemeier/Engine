//! Collision actors, spaces and portals.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Static obstacles actors slide against.
    pub fn collision(&self) -> &CollisionWorld {
        &self.collision
    }

    pub fn collision_mut(&mut self) -> &mut CollisionWorld {
        &mut self.collision
    }

    /// Slide `from` by `(dx, dz)` using `body`. Collision-off bodies translate.
    /// Only colliders in [`Self::living_in`] that overlap the actor vertically
    /// participate.
    pub fn move_actor(&self, body: &ActorBody, feet: GlobalPosition, dx: f64, dz: f64) -> GlobalXZ {
        self.collision
            .move_in(self.live_space, body, feet.horizontal(), dx, dz, feet.y)
    }

    /// Sweep a vertical actor capsule through finite walls, floors, and ceilings.
    ///
    /// `from` is the actor's feet. The caller owns gravity and velocity.
    pub fn move_actor_3d(
        &self,
        body: &ActorBody,
        from: GlobalPosition,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> ActorMove {
        self.collision
            .move_3d_in(self.live_space, body, from, dx, dy, dz)
    }

    /// Intern a disconnected space. Does not change where new entities spawn;
    /// call [`Self::in_space`] or [`Self::spawn_in`] for that.
    pub fn space(&mut self, name: impl Into<String>) -> EngineResult<SpaceId> {
        let name = name.into();
        if name.is_empty() {
            return Err(EngineError::InvalidValue(
                "space name must not be empty".into(),
            ));
        }
        if let Some(&id) = self.space_by_name.get(&name) {
            return Ok(id);
        }
        let id = SpaceId::from_raw(self.next_space_id);
        self.next_space_id = self
            .next_space_id
            .checked_add(1)
            .expect("space id overflow");
        self.space_by_name.insert(name, id);
        Ok(id)
    }

    /// Subsequent [`Self::spawn`] / [`Self::place`] calls go into `space`.
    pub fn in_space(&mut self, space: SpaceId) -> EngineResult<()> {
        self.require_space(space)?;
        self.spawn_space = space;
        Ok(())
    }

    /// Which space the main camera is standing in.
    pub fn live_in(&mut self, space: SpaceId) -> EngineResult<()> {
        self.require_space(space)?;
        if self.live_space != space {
            self.live_space = space;
            self.travel_prev = None;
        }
        Ok(())
    }

    pub fn living_in(&self) -> SpaceId {
        self.live_space
    }

    pub fn spawning_in(&self) -> SpaceId {
        self.spawn_space
    }

    /// Sky and clipmap belong to the default outdoor space, and to any named
    /// space that opted in with [`Self::set_space_draws_environment`].
    pub fn space_draws_environment(&self, space: SpaceId) -> bool {
        space == SpaceId::DEFAULT || self.environment_spaces.contains(&space)
    }

    /// Let a named space draw the outdoor sky and clipmap, or take that back.
    ///
    /// The default space always draws the environment. Asking to disable it is
    /// a caller bug, not a silent no-op.
    pub fn set_space_draws_environment(&mut self, space: SpaceId, draws: bool) -> EngineResult<()> {
        self.require_space(space)?;
        if space == SpaceId::DEFAULT {
            if draws {
                return Ok(());
            }
            return Err(EngineError::InvalidValue(
                "the default space always draws the outdoor environment".into(),
            ));
        }
        if draws {
            self.environment_spaces.insert(space);
        } else {
            self.environment_spaces.remove(&space);
        }
        Ok(())
    }

    pub(super) fn require_space(&self, space: SpaceId) -> EngineResult<()> {
        if space == SpaceId::DEFAULT || self.space_by_name.values().any(|&id| id == space) {
            return Ok(());
        }
        Err(EngineError::InvalidValue(format!(
            "unknown space {}",
            space.raw()
        )))
    }

    /// Spawn every part of a model (one entity per material) at `place`.
    /// Move them together by giving each the same place.
    pub fn spawn_model(
        &mut self,
        model: &crate::model::ModelScene,
        place: Place,
    ) -> EngineResult<Vec<EntityId>> {
        model
            .parts
            .iter()
            .map(|part| self.place(part.clone(), place))
            .collect()
    }

    /// Spawn a mesh at the origin.
    pub fn spawn(&mut self, mesh: Mesh) -> EntityId {
        self.place(mesh, Place::default())
            .expect("spawn at identity")
    }

    pub fn spawn_in(&mut self, space: SpaceId, mesh: Mesh) -> EngineResult<EntityId> {
        self.require_space(space)?;
        let prev = self.spawn_space;
        self.spawn_space = space;
        let id = self.spawn(mesh);
        self.spawn_space = prev;
        Ok(id)
    }

    pub fn place_in(&mut self, space: SpaceId, mesh: Mesh, place: Place) -> EngineResult<EntityId> {
        self.require_space(space)?;
        let prev = self.spawn_space;
        self.spawn_space = space;
        let id = self.place(mesh, place);
        self.spawn_space = prev;
        id
    }

    pub fn set_space(&mut self, id: EntityId, space: SpaceId) -> EngineResult<()> {
        self.require_space(space)?;
        if let Some(e) = self.entities.get_mut(&id) {
            if e.space != space {
                e.space = space;
                self.bump_render_epoch();
            }
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.space = space;
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
    }

    /// Two openings become the same portal. Both meshes should face their own
    /// space (`+Z` local). Looking through shows the other space; walking
    /// through teleports when [`PortalSettings::teleport`] is true.
    pub fn create_portal(
        &mut self,
        a: EntityId,
        b: EntityId,
        settings: PortalSettings,
    ) -> EngineResult<PortalId> {
        self.validate_portal_sides(a, b)?;
        let id = PortalId::from_raw(self.next_portal_id);
        self.next_portal_id = self
            .next_portal_id
            .checked_add(1)
            .expect("portal id overflow");
        self.portals.push(Portal {
            id,
            sides: [a, b],
            teleport: settings.teleport,
            enabled: true,
        });
        self.bump_render_epoch();
        Ok(id)
    }

    /// Toggle whether [`Self::travel`] crosses this portal.
    pub fn set_portal_teleport(&mut self, id: PortalId, teleport: bool) -> EngineResult<()> {
        let portal = self
            .portals
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| EngineError::InvalidValue(format!("unknown portal {id:?}")))?;
        portal.teleport = teleport;
        Ok(())
    }

    /// Remove a portal by id. The opening entities stay in the world.
    pub fn destroy_portal(&mut self, id: PortalId) -> EngineResult<()> {
        let before = self.portals.len();
        self.portals.retain(|p| p.id != id);
        if self.portals.len() == before {
            return Err(EngineError::InvalidValue(format!("unknown portal {id:?}")));
        }
        self.bump_render_epoch();
        Ok(())
    }

    pub fn set_portal_enabled(&mut self, id: PortalId, enabled: bool) -> EngineResult<()> {
        let portal = self
            .portals
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| EngineError::InvalidValue(format!("unknown portal {id:?}")))?;
        portal.set_enabled(enabled);
        self.bump_render_epoch();
        Ok(())
    }

    pub fn portal_recursion(&self) -> u8 {
        self.portal_recursion
    }

    pub fn set_portal_recursion(&mut self, depth: u8) {
        if depth == 0 {
            panic!("portal recursion depth must be at least 1");
        }
        self.portal_recursion = depth;
    }

    pub(super) fn validate_portal_sides(&self, a: EntityId, b: EntityId) -> EngineResult<()> {
        if a == b {
            return Err(EngineError::InvalidValue(
                "a portal cannot link an opening to itself".into(),
            ));
        }
        let plane_a = self.portal_plane(a)?;
        let plane_b = self.portal_plane(b)?;
        if plane_a.half_width <= 0.0 || plane_a.half_height <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "entity {a} has no XY extent to use as an opening"
            )));
        }
        if plane_b.half_width <= 0.0 || plane_b.half_height <= 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "entity {b} has no XY extent to use as an opening"
            )));
        }
        if self.is_linked_opening(a) || self.is_linked_opening(b) {
            return Err(EngineError::InvalidValue(
                "opening is already part of a portal".into(),
            ));
        }
        Ok(())
    }

    pub fn portals(&self) -> &[Portal] {
        &self.portals
    }

    /// All openings in `live` the camera is facing, nearest first.
    pub(crate) fn visible_portals(
        &self,
        eye: Vec3,
        look: Vec3,
        live: SpaceId,
    ) -> EngineResult<Vec<VisiblePortal>> {
        let look = {
            if look.length_squared() <= 0.0 {
                panic!("visible_portals look direction is zero");
            }
            look.normalize()
        };
        let mut found = Vec::new();
        for portal in &self.portals {
            if !portal.is_enabled() {
                continue;
            }
            let (Some(a), Some(b)) = (
                self.entities.get(&portal.a()),
                self.entities.get(&portal.b()),
            ) else {
                return Err(EngineError::InvalidValue(format!(
                    "enabled portal links missing entity {} or {}",
                    portal.a(),
                    portal.b()
                )));
            };
            for (src, dst) in portal.directions(a.space, b.space, live) {
                let src_e = if src == portal.a() { a } else { b };
                let dst_e = if dst == portal.a() { a } else { b };
                let src_plane = self.portal_plane(src).map_err(|error| {
                    EngineError::InvalidValue(format!(
                        "visible portal source {src} has no valid plane: {error}"
                    ))
                })?;
                if src_plane.signed_distance(eye) <= -crate::portal::PORTAL_VISIBILITY_PLANE_EPS {
                    continue;
                }
                if look.dot(-src_plane.normal) <= 0.0 {
                    continue;
                }
                let dist = eye.distance_squared(src_plane.center);
                let dst_plane = self.portal_plane(dst).map_err(|error| {
                    EngineError::InvalidValue(format!(
                        "visible portal destination {dst} has no valid plane: {error}"
                    ))
                })?;
                found.push((
                    dist,
                    VisiblePortal {
                        src,
                        dst,
                        src_transform: src_e.transform,
                        dst_transform: dst_e.transform,
                        dst_center: dst_plane.center,
                        dst_normal: dst_plane.normal,
                        dest_space: dst_e.space,
                    },
                ));
            }
        }
        found.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("finite portal planes produce finite distances")
        });
        Ok(found.into_iter().map(|(_, p)| p).collect())
    }

    /// Closest opening in the live space that the camera is facing.
    #[allow(dead_code)]
    pub(crate) fn visible_portal(
        &self,
        eye: Vec3,
        look: Vec3,
    ) -> EngineResult<Option<VisiblePortal>> {
        self.visible_portals(eye, look, self.live_space)
            .map(|portals| portals.into_iter().next())
    }

    /// Whether this entity is portal stencil geometry, enabled or disabled.
    /// Disabled portals must remain excluded from ordinary mesh rendering.
    pub fn is_portal_surface(&self, id: EntityId) -> bool {
        self.portals.iter().any(|p| p.a() == id || p.b() == id)
    }

    pub(crate) fn is_linked_opening(&self, id: EntityId) -> bool {
        self.portals.iter().any(|p| p.a() == id || p.b() == id)
    }

    pub(crate) fn portal_plane(&self, id: EntityId) -> EngineResult<PortalPlane> {
        let e = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if e.instanced {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} is instanced; openings must be a single mesh"
            )));
        }
        let (half_w, half_h) = match &e.released {
            Some(r) if r.lo.x <= r.hi.x => (
                r.lo.x.abs().max(r.hi.x.abs()),
                r.lo.y.abs().max(r.hi.y.abs()),
            ),
            Some(_) => (0.0, 0.0),
            None => opening_extents(&e.mesh),
        };
        Ok(PortalPlane::from_transform(e.transform, half_w, half_h))
    }

    /// If `position` crossed an opening this frame, move it to the other side
    /// and switch [`Self::living_in`]. Call before the camera helper.
    ///
    /// Returns the space you entered.
    pub fn travel(&mut self, position: &mut Vec3, yaw_degrees: &mut f32) -> Option<SpaceId> {
        if !position.is_finite() {
            panic!("travel position must be finite, got {position}");
        }
        if !yaw_degrees.is_finite() {
            panic!("travel yaw must be finite, got {yaw_degrees}");
        }
        let live = self.live_space;
        let prev = match self.travel_prev {
            Some((space, p)) if space == live => p,
            _ => {
                self.travel_prev = Some((live, *position));
                return None;
            }
        };
        let links: Vec<Portal> = self.portals.clone();
        for portal in links {
            if !portal.is_enabled() || !portal.teleport {
                continue;
            }
            let a_space = self.entities.get(&portal.a()).map(|e| e.space);
            let b_space = self.entities.get(&portal.b()).map(|e| e.space);
            let (Some(a_space), Some(b_space)) = (a_space, b_space) else {
                continue;
            };
            for (src, dst) in portal.directions(a_space, b_space, live) {
                if self.travel_gate == Some(dst) {
                    continue;
                }
                let src_plane = self
                    .portal_plane(src)
                    .expect("linked opening still has a plane");
                if !segment_crosses_opening(prev, *position, src_plane) {
                    continue;
                }
                let dst_e = self
                    .entities
                    .get(&dst)
                    .expect("linked opening still exists");
                let t_src = src_plane.transform;
                let t_dst = dst_e.transform;
                *position = crate::portal::portal_matrix(t_src, t_dst).transform_point3(*position);
                if dst_e.space != live {
                    let dst_plane = self
                        .portal_plane(dst)
                        .expect("linked opening still has a plane");
                    *position += dst_plane.normal * crate::portal::PORTAL_TELEPORT_SETBACK;
                }
                *yaw_degrees = teleport_yaw(*yaw_degrees, t_src, t_dst);
                self.live_space = dst_e.space;
                self.travel_prev = Some((self.live_space, *position));
                self.travel_gate = Some(dst);
                return Some(self.live_space);
            }
        }
        self.travel_gate = None;
        self.travel_prev = Some((live, *position));
        None
    }
}
