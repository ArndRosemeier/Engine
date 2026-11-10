//! Shadows, ambient occlusion, reflections, reflection probes and GI.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Enable hybrid sun shadows, or pass `None` to turn them off.
    pub fn set_shadows(&mut self, shadows: Option<ShadowSettings>) {
        self.shadows = shadows;
    }

    pub fn shadows(&self) -> Option<ShadowSettings> {
        self.shadows
    }

    /// Every how many frames each sun cascade (near, middle, far) is drawn
    /// anew. Between redraws a cascade keeps its last map and the light
    /// matrix it was drawn with, so its shadows stay right, only up to
    /// `n - 1` frames old. Redraws of different cascades fall on different
    /// frames, so each frame draws fewer of them. A cascade is drawn at
    /// once when it has no map yet, the sun turns, or the camera moves too
    /// far for the old map.
    ///
    /// The default `[1, 2, 2]` draws the near cascade every frame and the
    /// other two on alternate frames: two cascades a frame instead of
    /// three. `[1, 1, 1]` draws all of them every frame (the old way);
    /// `ENGINE_SHADOW_REFRESH=1` forces that for comparisons. Values are
    /// clamped to 1..=8.
    pub fn set_shadow_cascade_refresh(&mut self, every: [u32; 3]) {
        self.shadow_cascade_refresh = every.map(|n| n.clamp(1, 8));
    }

    /// Cache the sun shadows of things that don't move (on by default).
    /// Each cascade keeps a second map holding only its static casters:
    /// meshes and instance batches unchanged for half a second that don't
    /// sway in the wind, and voxel ground. While none of them changes and
    /// the cascade stays put, that map is copied in and only moving things
    /// (characters, crowds, vehicles, swaying plants) are drawn on top. A
    /// caster that moves leaves the cache at once and comes back once it has
    /// rested. `ENGINE_SHADOW_CACHE=0` turns it off for comparisons.
    pub fn set_shadow_cache(&mut self, on: bool) {
        self.shadow_cache = on;
    }

    /// See [`World::set_shadow_cache`].
    pub fn shadow_cache(&self) -> bool {
        self.shadow_cache
    }

    /// See [`World::set_shadow_cascade_refresh`].
    pub fn shadow_cascade_refresh(&self) -> [u32; 3] {
        self.shadow_cascade_refresh
    }

    /// Choose the instanced submit path. Both are first-class; neither falls back.
    pub fn set_instance_submit(&mut self, submit: InstanceSubmit) {
        if self.instance_submit == submit {
            return;
        }
        self.instance_submit = submit;
        self.bump_render_epoch();
    }

    pub fn instance_submit(&self) -> InstanceSubmit {
        self.instance_submit
    }

    pub fn ambient_occlusion(&self) -> Option<AmbientOcclusion> {
        self.ambient_occlusion
    }

    pub fn set_ambient_occlusion(&mut self, ao: Option<AmbientOcclusion>) {
        self.ambient_occlusion = ao;
    }

    /// Screen-space reflections (on by default; see [`crate::reflections`]).
    pub fn screen_space_reflections(&self) -> Option<crate::reflections::ScreenSpaceReflections> {
        self.ssr
    }

    pub fn set_screen_space_reflections(
        &mut self,
        ssr: Option<crate::reflections::ScreenSpaceReflections>,
    ) -> EngineResult<()> {
        if let Some(s) = &ssr {
            s.validate()?;
        }
        self.ssr = ssr;
        Ok(())
    }

    /// Add a reflection probe (at most
    /// [`crate::reflections::MAX_REFLECTION_PROBES`]); it is captured over
    /// the next frame.
    pub fn add_reflection_probe(
        &mut self,
        probe: crate::reflections::ReflectionProbe,
    ) -> EngineResult<crate::reflections::ReflectionProbeId> {
        self.reflection_probes.add(probe)
    }

    /// Move or reshape a probe (it is captured again).
    pub fn set_reflection_probe(
        &mut self,
        id: crate::reflections::ReflectionProbeId,
        probe: crate::reflections::ReflectionProbe,
    ) -> EngineResult<()> {
        self.reflection_probes.set(id, probe)
    }

    /// Capture a probe again, after what it sees changed.
    pub fn refresh_reflection_probe(&mut self, id: crate::reflections::ReflectionProbeId) {
        self.reflection_probes.refresh(id);
    }

    /// Capture every probe again (say after the time of day jumped).
    pub fn refresh_reflection_probes(&mut self) {
        self.reflection_probes.refresh_all();
    }

    pub fn remove_reflection_probe(
        &mut self,
        id: crate::reflections::ReflectionProbeId,
    ) -> Option<crate::reflections::ReflectionProbe> {
        self.reflection_probes.remove(id)
    }

    pub fn reflection_probe(
        &self,
        id: crate::reflections::ReflectionProbeId,
    ) -> Option<&crate::reflections::ReflectionProbe> {
        self.reflection_probes.get(id)
    }

    /// Live probes as (slot, probe, version); the version changes whenever
    /// the probe should be captured again.
    pub(crate) fn reflection_probe_slots(
        &self,
    ) -> impl Iterator<Item = (usize, &crate::reflections::ReflectionProbe, u64)> {
        self.reflection_probes.iter()
    }

    pub fn global_illumination(&self) -> Option<GlobalIllumination> {
        self.global_illumination
    }

    pub fn set_global_illumination(&mut self, gi: Option<GlobalIllumination>) {
        self.global_illumination = gi;
    }
}
