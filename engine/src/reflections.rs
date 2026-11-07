//! Reflections: screen-space reflections plus reflection probes.
//!
//! Glossy surfaces (chrome, glass, wet streets, polished floors) reflect
//! what is around them in three layers, each filling in where the one
//! before has nothing:
//!
//! 1. **Screen-space reflections** ([`ScreenSpaceReflections`], on by
//!    default): the reflected ray is traced through the last frame's image
//!    and depth. Exact and moving, but only for what is on screen.
//! 2. **Reflection probes** ([`ReflectionProbe`]): a cube map captured at a
//!    point and projected onto a box (a room, a street, a hall), so a mirror
//!    inside it shows the walls in the right place. Rough surfaces read a
//!    blurrier version. Captured once when added (or moved), or every few
//!    frames for `dynamic` probes.
//! 3. The sky, dimmed where the light probes say it is hidden.
//!
//! ```no_run
//! # use engine::prelude::*;
//! # fn demo(world: &mut World) -> EngineResult<()> {
//! // A probe filling a 20 x 6 x 12 m hall, captured at its middle.
//! let hall = world.add_reflection_probe(ReflectionProbe::new((0.0, 3.0, 0.0), (10.0, 3.0, 6.0)))?;
//! // Re-capture after the room changed (a wall broke, lights went out).
//! world.refresh_reflection_probe(hall);
//! # Ok(()) }
//! ```

use crate::error::{EngineError, EngineResult};
use glam::Vec3;

/// How many reflection probes can be live at once.
pub const MAX_REFLECTION_PROBES: usize = 8;

/// Screen-space reflections: tune or switch off with
/// `World::set_screen_space_reflections`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenSpaceReflections {
    /// Longest reflected ray, in metres.
    pub max_distance_m: f32,
    /// 0..1: how much of a hit replaces the probe or sky reflection.
    pub strength: f32,
    /// Steps along each ray; more finds thin objects, costs more.
    pub steps: u32,
}

impl Default for ScreenSpaceReflections {
    fn default() -> Self {
        Self {
            max_distance_m: 60.0,
            strength: 1.0,
            steps: 24,
        }
    }
}

impl ScreenSpaceReflections {
    pub fn validate(&self) -> EngineResult<()> {
        if !self.max_distance_m.is_finite() || !(0.5..=2000.0).contains(&self.max_distance_m) {
            return Err(EngineError::InvalidValue(
                "SSR max distance must be within 0.5..=2000 metres".into(),
            ));
        }
        if !self.strength.is_finite() || !(0.0..=1.0).contains(&self.strength) {
            return Err(EngineError::InvalidValue(
                "SSR strength must be within 0..=1".into(),
            ));
        }
        if !(4..=128).contains(&self.steps) {
            return Err(EngineError::InvalidValue(
                "SSR steps must be within 4..=128".into(),
            ));
        }
        Ok(())
    }
}

/// A reflection probe handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReflectionProbeId(pub u32);

/// A cube map of the surroundings, captured at `center` and projected onto
/// the box `center ± half_extents`. Surfaces inside the box reflect it;
/// near the box faces it fades out over `blend_m` metres into whatever lies
/// outside (another probe, or the sky). Where boxes overlap, the smaller
/// box wins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionProbe {
    pub center: Vec3,
    pub half_extents: Vec3,
    pub blend_m: f32,
    /// Re-capture continuously (one cube face per frame) instead of once.
    pub dynamic: bool,
}

impl ReflectionProbe {
    pub fn new(center: impl Into<Vec3>, half_extents: impl Into<Vec3>) -> Self {
        Self {
            center: center.into(),
            half_extents: half_extents.into(),
            blend_m: 1.0,
            dynamic: false,
        }
    }

    /// Re-capture continuously, for places where things move or lights
    /// change. Costs one extra scene draw at 128 x 128 per frame.
    pub fn dynamic(mut self) -> Self {
        self.dynamic = true;
        self
    }

    pub fn blend(mut self, metres: f32) -> Self {
        self.blend_m = metres;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| {
            Err(EngineError::InvalidValue(format!(
                "reflection probe {what}"
            )))
        };
        if !self.center.is_finite() {
            return bad("center must be finite");
        }
        if !self.half_extents.is_finite() || self.half_extents.min_element() <= 0.05 {
            return bad("half extents must be finite and > 0.05");
        }
        if !self.blend_m.is_finite() || self.blend_m < 0.0 {
            return bad("blend must be finite and >= 0");
        }
        Ok(())
    }

    pub fn contains(&self, p: Vec3) -> bool {
        ((p - self.center).abs() - self.half_extents).max_element() <= 0.0
    }

    pub(crate) fn volume(&self) -> f32 {
        self.half_extents.x * self.half_extents.y * self.half_extents.z
    }
}

/// The probes a world holds, with a version per slot that bumps on every
/// change so the renderer knows what to re-capture.
#[derive(Debug, Default)]
pub(crate) struct ReflectionProbes {
    slots: Vec<Option<(ReflectionProbe, u64)>>,
    next_version: u64,
}

impl ReflectionProbes {
    pub fn add(&mut self, probe: ReflectionProbe) -> EngineResult<ReflectionProbeId> {
        probe.validate()?;
        let live = self.slots.iter().filter(|s| s.is_some()).count();
        if live >= MAX_REFLECTION_PROBES {
            return Err(EngineError::InvalidValue(format!(
                "at most {MAX_REFLECTION_PROBES} reflection probes"
            )));
        }
        let v = self.bump();
        let slot = (probe, v);
        if let Some(i) = self.slots.iter().position(|s| s.is_none()) {
            self.slots[i] = Some(slot);
            return Ok(ReflectionProbeId(i as u32));
        }
        self.slots.push(Some(slot));
        Ok(ReflectionProbeId(self.slots.len() as u32 - 1))
    }

    fn bump(&mut self) -> u64 {
        self.next_version += 1;
        self.next_version
    }

    pub fn set(&mut self, id: ReflectionProbeId, probe: ReflectionProbe) -> EngineResult<()> {
        probe.validate()?;
        let v = self.bump();
        match self.slots.get_mut(id.0 as usize) {
            Some(Some(s)) => {
                *s = (probe, v);
                Ok(())
            }
            _ => Err(EngineError::InvalidValue(format!(
                "no reflection probe {}",
                id.0
            ))),
        }
    }

    pub fn refresh(&mut self, id: ReflectionProbeId) {
        let v = self.bump();
        if let Some(Some(s)) = self.slots.get_mut(id.0 as usize) {
            s.1 = v;
        }
    }

    pub fn refresh_all(&mut self) {
        for i in 0..self.slots.len() {
            self.refresh(ReflectionProbeId(i as u32));
        }
    }

    pub fn remove(&mut self, id: ReflectionProbeId) -> Option<ReflectionProbe> {
        self.slots.get_mut(id.0 as usize)?.take().map(|s| s.0)
    }

    pub fn get(&self, id: ReflectionProbeId) -> Option<&ReflectionProbe> {
        self.slots.get(id.0 as usize)?.as_ref().map(|s| &s.0)
    }

    /// Live probes with their slot index and version.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &ReflectionProbe, u64)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|(p, v)| (i, p, *v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_are_capped_reused_and_versioned() {
        let mut p = ReflectionProbes::default();
        let a = p.add(ReflectionProbe::new(Vec3::ZERO, Vec3::ONE)).unwrap();
        let v0 = p.iter().next().unwrap().2;
        p.refresh(a);
        assert!(p.iter().next().unwrap().2 > v0);
        for _ in 1..MAX_REFLECTION_PROBES {
            p.add(ReflectionProbe::new(Vec3::ZERO, Vec3::ONE)).unwrap();
        }
        assert!(p.add(ReflectionProbe::new(Vec3::ZERO, Vec3::ONE)).is_err());
        p.remove(a);
        assert_eq!(
            p.add(ReflectionProbe::new(Vec3::ZERO, Vec3::ONE)).unwrap(),
            a
        );
        assert!(ReflectionProbe::new(Vec3::ZERO, Vec3::ZERO)
            .validate()
            .is_err());
        assert!(ReflectionProbe::new(Vec3::ZERO, Vec3::ONE).contains(Vec3::splat(0.9)));
    }
}
