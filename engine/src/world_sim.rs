//! Global-coordinate cameras, placing entities, the fixed simulation tick and interpolation.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// First-person camera at an anchored eye position, expressed globally.
    pub fn look_first_person_global(
        &mut self,
        eye: GlobalPosition,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) -> EngineResult<()> {
        let local = self.to_render(eye)?;
        self.look_first_person(local, yaw_degrees, pitch_degrees);
        Ok(())
    }

    /// Look from one global point at another. A title shot, a cutscene, anything
    /// that is not a walker.
    pub fn look_at_global(
        &mut self,
        eye: GlobalPosition,
        target: GlobalPosition,
    ) -> EngineResult<()> {
        let e = self.to_render(eye)?;
        let t = self.to_render(target)?;
        let fov = self.camera.fov_y_degrees();
        let near = self.camera.near();
        self.camera = Camera::look_at(e, t).with_lens(fov, near, self.view_distance);
        Ok(())
    }

    /// Follow camera around an anchored target, expressed globally.
    pub fn look_follow_global(
        &mut self,
        target: GlobalPosition,
        yaw_degrees: f32,
        distance: f32,
        height: f32,
    ) -> EngineResult<()> {
        let local = self.to_render(target)?;
        self.look_follow(local, yaw_degrees, distance, height);
        Ok(())
    }

    /// Distance from the render origin to `p`, used to decide when to rebase.
    pub fn render_offset_m(&self, p: GlobalXZ) -> f64 {
        p.distance(self.render_origin.horizontal())
    }

    /// Move an entity. For a child (see [`Self::attach`]) the place is
    /// relative to its parent; its own children come along either way.
    pub fn set_place(&mut self, id: EntityId, place: Place) -> EngineResult<()> {
        self.set_matrix(id, place.to_matrix())
    }

    /// `set_place` with a ready matrix.
    pub(super) fn set_matrix(&mut self, id: EntityId, m: Mat4) -> EngineResult<()> {
        if let Some(link) = self.hierarchy.links.get_mut(&id) {
            link.local = m;
            return self.update_subtree(id);
        }
        if self.hierarchy.pivots.contains_key(&id) || self.hierarchy.children.contains_key(&id) {
            self.write_matrix(id, m)?;
            return self.update_subtree(id);
        }
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform = m;
            e.bump_xform();
            self.bump_render_epoch();
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.transform = m;
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
    }

    /// Turn the fixed simulation tick on (`Some(hz)`) or off. While on, every
    /// [`Frame::sim`] says how many steps are due. See [`crate::sim`].
    pub fn set_sim_rate(&mut self, hz: Option<f32>) -> EngineResult<()> {
        self.sim_clock = match hz {
            Some(hz) => Some(crate::sim::SimClock::new(hz)?),
            None => None,
        };
        Ok(())
    }

    pub fn sim_rate(&self) -> Option<f32> {
        self.sim_clock.as_ref().map(|c| c.rate())
    }

    /// Advance the fixed tick by a frame of `dt` seconds (the engine's loop
    /// calls this before your update).
    pub(crate) fn advance_sim(&mut self, dt: f32) -> crate::sim::SimSteps {
        match self.sim_clock.as_mut() {
            Some(clock) => {
                let s = clock.advance(dt);
                self.sim_alpha = s.alpha;
                s
            }
            None => crate::sim::SimSteps {
                alpha: self.sim_alpha,
                ..Default::default()
            },
        }
    }

    /// Start a simulation step: poses set from now on belong to the new step,
    /// and the ones before become the step drawing starts from.
    pub fn begin_sim_step(&mut self) {
        self.sim_tick += 1;
    }

    /// Where an entity is after the current simulation step. It is drawn
    /// between this and its previous step's pose; an entity whose pose isn't
    /// set in a step rests where it was. The entity follows its simulation
    /// poses until [`Self::clear_sim_pose`].
    pub fn set_sim_place(&mut self, id: EntityId, place: Place) -> EngineResult<()> {
        self.set_sim_transform(id, place.to_matrix())
    }

    /// [`Self::set_sim_place`] with a matrix.
    pub fn set_sim_transform(&mut self, id: EntityId, m: Mat4) -> EngineResult<()> {
        crate::place::ensure_finite_mat(m)?;
        if !self.entities.contains_key(&id)
            && !self.animated.contains_key(&id)
            && !self.hierarchy.pivots.contains_key(&id)
        {
            return Err(EngineError::UnknownEntity);
        }
        let tick = self.sim_tick;
        let pose = self.sim_poses.entry(id).or_insert(crate::sim::SimPose {
            prev: m,
            cur: m,
            tick,
        });
        if pose.tick != tick {
            pose.prev = pose.cur;
            pose.tick = tick;
        }
        pose.cur = m;
        Ok(())
    }

    /// Both poses at once, for a client that keeps the previous and current
    /// simulation state itself: drawn between them by the frame's blend.
    pub fn set_sim_poses(
        &mut self,
        id: EntityId,
        previous: Mat4,
        current: Mat4,
    ) -> EngineResult<()> {
        crate::place::ensure_finite_mat(previous)?;
        self.set_sim_transform(id, current)?;
        if let Some(p) = self.sim_poses.get_mut(&id) {
            p.prev = previous;
        }
        Ok(())
    }

    /// Stop following simulation poses; the entity stays where it is drawn.
    pub fn clear_sim_pose(&mut self, id: EntityId) {
        self.sim_poses.remove(&id);
    }

    /// Where drawing is between the last two simulation steps (0..1).
    pub fn sim_alpha(&self) -> f32 {
        self.sim_alpha
    }

    /// Set the blend yourself, for a game that runs its own fixed clock
    /// instead of [`Self::set_sim_rate`].
    pub fn set_sim_alpha(&mut self, alpha: f32) {
        if alpha.is_finite() {
            self.sim_alpha = alpha.clamp(0.0, 1.0);
        }
    }

    /// Move entities with simulation poses to their blended pose. The engine
    /// calls this every frame.
    pub fn tick_interpolation(&mut self) {
        if self.sim_poses.is_empty() {
            return;
        }
        let (tick, alpha) = (self.sim_tick, self.sim_alpha);
        let poses: Vec<(EntityId, Mat4)> = self
            .sim_poses
            .iter()
            .map(|(id, p)| (*id, p.at(tick, alpha)))
            .collect();
        for (id, m) in poses {
            if self.set_matrix(id, m).is_err() {
                self.sim_poses.remove(&id);
            }
        }
    }

    /// Everything the windowed loop ticks that makes sense with no screen.
    pub(crate) fn tick_headless(&mut self, dt: f32) {
        self.tick_nav(dt);
        self.tick_animations(dt);
        self.tick_physics(dt);
        self.tick_ragdolls(dt);
        self.tick_buildings(dt);
        self.tick_interpolation();
        self.tick_hierarchy();
    }

    /// Element words changed since the last call (start word, words), for the renderer.
    pub(crate) fn take_element_upload(&mut self) -> Option<(usize, &[u32])> {
        let (a, b) = self.element_states.dirty.take()?;
        Some((a, &self.element_states.bits[a..b]))
    }
}
