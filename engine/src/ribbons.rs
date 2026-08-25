//! Standalone bounded world-space ribbon trails.
use crate::{Color, EngineError, EngineResult};
use glam::Vec3;
use std::collections::VecDeque;

pub(crate) const MAX_RIBBONS: usize = 32;
pub(crate) const MAX_RIBBON_POINTS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RibbonId {
    slot: u32,
    generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RibbonProfile {
    Smooth,
    Turbulent,
    Jagged,
    Organic,
    Orbit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonStyle {
    width_m: f32,
    primary: Color,
    secondary: Color,
    lifetime_s: f32,
    emissive_intensity: f32,
    point_spacing_m: f32,
    profile: RibbonProfile,
    cross_ribbon: bool,
}
impl RibbonStyle {
    pub fn new(
        width_m: f32,
        primary: Color,
        secondary: Color,
        lifetime_s: f32,
    ) -> EngineResult<Self> {
        Self {
            width_m,
            primary,
            secondary,
            lifetime_s,
            emissive_intensity: 1.0,
            point_spacing_m: 0.04,
            profile: RibbonProfile::Smooth,
            cross_ribbon: false,
        }
        .validate()
    }
    pub fn with_emissive_intensity(mut self, v: f32) -> EngineResult<Self> {
        self.emissive_intensity = v;
        self.validate()
    }
    pub fn with_point_spacing(mut self, v: f32) -> EngineResult<Self> {
        self.point_spacing_m = v;
        self.validate()
    }
    pub fn with_profile(mut self, v: RibbonProfile) -> Self {
        self.profile = v;
        self
    }
    pub fn with_cross_ribbon(mut self, v: bool) -> Self {
        self.cross_ribbon = v;
        self
    }
    pub fn validate(self) -> EngineResult<Self> {
        let colors = [self.primary, self.secondary].into_iter().all(|c| {
            [c.r, c.g, c.b, c.a]
                .into_iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(&v))
        });
        if !colors
            || !self.width_m.is_finite()
            || self.width_m <= 0.0
            || !self.lifetime_s.is_finite()
            || self.lifetime_s <= 0.0
            || !self.emissive_intensity.is_finite()
            || self.emissive_intensity < 0.0
            || !self.point_spacing_m.is_finite()
            || self.point_spacing_m <= 0.0
        {
            return Err(EngineError::InvalidValue("invalid ribbon style".into()));
        }
        Ok(self)
    }
    pub(crate) fn width_m(self) -> f32 {
        self.width_m
    }
    pub(crate) fn primary(self) -> Color {
        self.primary
    }
    pub(crate) fn secondary(self) -> Color {
        self.secondary
    }
    pub(crate) fn lifetime_s(self) -> f32 {
        self.lifetime_s
    }
    pub(crate) fn emissive_intensity(self) -> f32 {
        self.emissive_intensity
    }
    pub(crate) fn profile(self) -> RibbonProfile {
        self.profile
    }
    pub(crate) fn cross_ribbon(self) -> bool {
        self.cross_ribbon
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RibbonPoint {
    pub position: Vec3,
    pub age_s: f32,
}
#[derive(Clone, Debug)]
pub(crate) struct RibbonSnapshot {
    pub style: RibbonStyle,
    pub points: Vec<RibbonPoint>,
}
#[derive(Clone, Debug)]
struct Ribbon {
    style: RibbonStyle,
    points: VecDeque<RibbonPoint>,
    recording: bool,
}
#[derive(Clone, Debug)]
struct Slot {
    generation: u32,
    ribbon: Option<Ribbon>,
}
impl Default for Slot {
    fn default() -> Self {
        Self {
            generation: 1,
            ribbon: None,
        }
    }
}
#[derive(Debug)]
pub struct RibbonWorld {
    slots: [Slot; MAX_RIBBONS],
    time_scale: f32,
}
impl Default for RibbonWorld {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Slot::default()),
            time_scale: 1.0,
        }
    }
}
impl RibbonWorld {
    pub fn start(&mut self, position: Vec3, style: RibbonStyle) -> EngineResult<RibbonId> {
        if !position.is_finite() {
            return Err(EngineError::InvalidValue(
                "ribbon position must be finite".into(),
            ));
        }
        let style = style.validate()?;
        let (index, slot) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, s)| s.ribbon.is_none())
            .ok_or_else(|| {
                EngineError::ResourceLimit(format!("ribbon capacity ({MAX_RIBBONS}) exhausted"))
            })?;
        let id = RibbonId {
            slot: u32::try_from(index).expect("ribbon capacity exceeds u32"),
            generation: slot.generation,
        };
        slot.ribbon = Some(Ribbon {
            style,
            points: VecDeque::from([RibbonPoint {
                position,
                age_s: 0.0,
            }]),
            recording: true,
        });
        Ok(id)
    }
    pub fn update_head(&mut self, id: RibbonId, position: Vec3) -> EngineResult<()> {
        if !position.is_finite() {
            return Err(EngineError::InvalidValue(
                "ribbon position must be finite".into(),
            ));
        }
        let ribbon = self.ribbon_mut(id)?;
        if ribbon.points.is_empty() {
            ribbon.points.push_back(RibbonPoint {
                position,
                age_s: 0.0,
            });
            return Ok(());
        }
        if !ribbon.recording {
            return Err(EngineError::InvalidValue(
                "cannot update a stopped ribbon".into(),
            ));
        }
        let last = ribbon.points.back().expect("active ribbon has a point");
        if last.position.distance_squared(position)
            >= ribbon.style.point_spacing_m * ribbon.style.point_spacing_m
        {
            if ribbon.points.len() == MAX_RIBBON_POINTS {
                ribbon.points.pop_front();
            }
            ribbon.points.push_back(RibbonPoint {
                position,
                age_s: 0.0,
            });
        } else {
            ribbon
                .points
                .back_mut()
                .expect("active ribbon has a point")
                .position = position;
        }
        Ok(())
    }
    pub fn stop(&mut self, id: RibbonId) -> EngineResult<()> {
        let ribbon = self.ribbon_mut(id)?;
        if !ribbon.recording {
            return Err(EngineError::InvalidValue("ribbon already stopped".into()));
        }
        ribbon.recording = false;
        Ok(())
    }
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            if slot.ribbon.take().is_some() {
                slot.generation = next_generation(slot.generation);
            }
        }
    }
    pub fn set_time_scale(&mut self, scale: f32) -> EngineResult<()> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(EngineError::InvalidValue(
                "ribbon time scale must be finite and >= 0".into(),
            ));
        }
        self.time_scale = scale;
        Ok(())
    }
    pub fn advance(&mut self, dt_s: f32) -> EngineResult<()> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(EngineError::InvalidValue(
                "ribbon dt must be finite and >= 0".into(),
            ));
        }
        let dt = dt_s * self.time_scale;
        for slot in &mut self.slots {
            let Some(r) = slot.ribbon.as_mut() else {
                continue;
            };
            for point in &mut r.points {
                point.age_s += dt;
            }
            while r
                .points
                .front()
                .is_some_and(|p| p.age_s >= r.style.lifetime_s)
            {
                r.points.pop_front();
            }
            if !r.recording && r.points.is_empty() {
                slot.ribbon = None;
                slot.generation = next_generation(slot.generation);
            }
        }
        Ok(())
    }
    pub(crate) fn snapshots(&self) -> Vec<RibbonSnapshot> {
        self.slots
            .iter()
            .filter_map(|s| {
                s.ribbon.as_ref().map(|r| RibbonSnapshot {
                    style: r.style,
                    points: r.points.iter().copied().collect(),
                })
            })
            .collect()
    }
    fn ribbon_mut(&mut self, id: RibbonId) -> EngineResult<&mut Ribbon> {
        let slot = self
            .slots
            .get_mut(id.slot as usize)
            .ok_or_else(|| EngineError::InvalidValue("ribbon handle has invalid slot".into()))?;
        if slot.generation != id.generation || slot.ribbon.is_none() {
            return Err(EngineError::InvalidValue("stale ribbon handle".into()));
        }
        Ok(slot.ribbon.as_mut().unwrap())
    }
}
fn next_generation(g: u32) -> u32 {
    g.checked_add(1).expect("ribbon generation exhausted")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn style() -> RibbonStyle {
        RibbonStyle::new(0.2, Color::WHITE, Color::BLACK, 0.5).unwrap()
    }
    #[test]
    fn validates_config() {
        assert!(RibbonStyle::new(0.0, Color::WHITE, Color::WHITE, 1.0).is_err());
        assert!(style().with_emissive_intensity(f32::NAN).is_err());
    }
    #[test]
    fn lifecycle_and_stale_handles() {
        let mut w = RibbonWorld::default();
        let id = w.start(Vec3::ZERO, style()).unwrap();
        w.update_head(id, Vec3::X).unwrap();
        w.stop(id).unwrap();
        assert!(w.update_head(id, Vec3::Y).is_err());
        w.advance(0.6).unwrap();
        assert!(w.stop(id).is_err());
    }
    #[test]
    fn stopped_ribbon_keeps_tail_until_expiry() {
        let mut world = RibbonWorld::default();
        let id = world.start(Vec3::ZERO, style()).unwrap();
        world.update_head(id, Vec3::X).unwrap();
        world.stop(id).unwrap();
        world.advance(0.2).unwrap();
        assert_eq!(world.snapshots()[0].points.len(), 2);
        world.advance(0.31).unwrap();
        assert!(world.snapshots().is_empty());
    }

    #[test]
    fn capacity_is_loud() {
        let mut w = RibbonWorld::default();
        for _ in 0..MAX_RIBBONS {
            w.start(Vec3::ZERO, style()).unwrap();
        }
        assert!(w.start(Vec3::ZERO, style()).is_err());
    }
    #[test]
    fn history_is_bounded_and_expires() {
        let mut w = RibbonWorld::default();
        let id = w
            .start(Vec3::ZERO, style().with_point_spacing(0.001).unwrap())
            .unwrap();
        for i in 1..100 {
            w.update_head(id, Vec3::X * i as f32).unwrap();
        }
        assert_eq!(w.snapshots()[0].points.len(), MAX_RIBBON_POINTS);
        w.advance(0.6).unwrap();
        assert!(w.snapshots()[0].points.is_empty());
    }
}
