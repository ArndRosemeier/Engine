//! The orbit camera, sun, torch and lamps.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Orbit camera helper (yaw/pitch in degrees).
    pub fn look_orbit(
        &mut self,
        target: impl Into<Vec3>,
        distance: f32,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) {
        self.camera = Camera::orbit(target, distance, yaw_degrees, pitch_degrees);
        self.camera = self.camera.clone().with_lens(
            self.camera.fov_y_degrees(),
            self.camera.near(),
            self.view_distance,
        );
    }

    /// Set sun direction (need not be normalized) and ambient 0..1.
    pub fn set_sun(&mut self, direction: impl Into<Vec3>, ambient: f32) {
        self.light.direction = direction.into();
        self.light.ambient = ambient.clamp(0.0, 1.0);
    }

    /// Light the scene from the camera eye ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a lantern or headlamp ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â or take
    /// it out again with `None`.
    ///
    /// The torch follows the view every frame; there is no separate position.
    pub fn set_torch(&mut self, torch: Option<TorchLight>) {
        self.torch = torch;
    }

    pub fn torch(&self) -> Option<TorchLight> {
        self.torch
    }

    /// Place a lamp (point light or spotlight) in the space things currently
    /// spawn in. See [`crate::lamps`].
    pub fn add_lamp(&mut self, lamp: crate::lamps::Lamp) -> EngineResult<crate::lamps::LampId> {
        lamp.validate()?;
        let id = crate::lamps::LampId(self.next_lamp);
        self.next_lamp = self.next_lamp.wrapping_add(1).max(1);
        self.lamps.insert(id, (lamp, self.spawn_space));
        Ok(id)
    }

    /// Change everything about a lamp at once.
    pub fn set_lamp(
        &mut self,
        id: crate::lamps::LampId,
        lamp: crate::lamps::Lamp,
    ) -> EngineResult<()> {
        lamp.validate()?;
        let slot = self.lamps.get_mut(&id).ok_or(EngineError::UnknownEntity)?;
        slot.0 = lamp;
        Ok(())
    }

    /// Move a lamp (a lantern someone carries, a car's headlight).
    pub fn set_lamp_position(
        &mut self,
        id: crate::lamps::LampId,
        position: impl Into<Vec3>,
    ) -> EngineResult<()> {
        let p = position.into();
        crate::place::ensure_finite3(p, "lamp position")?;
        let slot = self.lamps.get_mut(&id).ok_or(EngineError::UnknownEntity)?;
        slot.0.position = p;
        Ok(())
    }

    pub fn lamp(&self, id: crate::lamps::LampId) -> Option<&crate::lamps::Lamp> {
        self.lamps.get(&id).map(|(l, _)| l)
    }

    pub fn remove_lamp(&mut self, id: crate::lamps::LampId) {
        self.lamps.remove(&id);
        self.hierarchy.lamps.remove(&id);
    }

    /// Every lamp, with the space it lives in.
    pub fn lamps(
        &self,
    ) -> impl Iterator<Item = (crate::lamps::LampId, &crate::lamps::Lamp, SpaceId)> {
        self.lamps.iter().map(|(id, (l, s))| (*id, l, *s))
    }

    /// How many shadow-casting lamps get their shadows drawn per frame (the
    /// brightest near the camera win; the rest light without shadows).
    /// Default 4; each costs about as much as drawing the scene around it six
    /// times at low resolution (once for a spotlight).
    pub fn set_lamp_shadow_budget(&mut self, lamps: usize) {
        self.lamp_shadow_budget = lamps.min(crate::lamps::MAX_SHADOW_LAMPS);
    }

    pub fn lamp_shadow_budget(&self) -> usize {
        self.lamp_shadow_budget
    }
}
