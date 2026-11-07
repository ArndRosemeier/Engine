//! Colour grading and fog by zone.
//!
//! A [`ColorGrade`] is the look of the final image: exposure, contrast,
//! saturation, white balance, a tint for shadows and one for highlights, and
//! a vignette. The world has a base grade; [`FogVolume`]s are boxes that
//! bring their own grade and fog. As the camera moves, the look blends
//! across each box's edge (over `blend_m` metres) and then eases toward
//! that target over [`World::set_zone_transition`](crate::world::World::set_zone_transition)
//! seconds, so crossing into a smog district turns the air brown over a few
//! seconds instead of at a line.
//!
//! ```no_run
//! # use engine::prelude::*;
//! # fn demo(world: &mut World) -> EngineResult<()> {
//! world.set_color_grade(ColorGrade::warm());
//! world.add_fog_volume(
//!     FogVolume::new((200.0, 30.0, 0.0), (100.0, 60.0, 100.0))
//!         .grade(ColorGrade::smog())
//!         .fog(rgb(120, 104, 80), 90.0)
//!         .blend(25.0),
//! )?;
//! # Ok(()) }
//! ```

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::world::Haze;
use glam::Vec3;

/// The look of the final image. Every field blends linearly, so any two
/// grades mix smoothly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGrade {
    /// Brightness in stops (0 none, 1 twice as bright).
    pub exposure: f32,
    /// 1 none; above 1 punchier, below flatter.
    pub contrast: f32,
    /// 1 none, 0 grey, above 1 more vivid.
    pub saturation: f32,
    /// White balance: -1 cold blue .. 1 warm orange.
    pub temperature: f32,
    /// -1 magenta .. 1 green.
    pub tint: f32,
    /// Multiplies the dark tones (white: none).
    pub shadows: Vec3,
    /// Multiplies the bright tones (white: none).
    pub highlights: Vec3,
    /// Darkened corners, 0..1.
    pub vignette: f32,
}

impl Default for ColorGrade {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

impl ColorGrade {
    /// No change to the image.
    pub const NEUTRAL: Self = Self {
        exposure: 0.0,
        contrast: 1.0,
        saturation: 1.0,
        temperature: 0.0,
        tint: 0.0,
        shadows: Vec3::ONE,
        highlights: Vec3::ONE,
        vignette: 0.0,
    };

    /// Golden and lush: a hopeful, sunny district.
    pub fn warm() -> Self {
        Self {
            exposure: 0.1,
            contrast: 1.06,
            saturation: 1.18,
            temperature: 0.35,
            shadows: Vec3::new(1.0, 0.96, 0.92),
            highlights: Vec3::new(1.04, 1.0, 0.92),
            vignette: 0.15,
            ..Self::NEUTRAL
        }
    }

    /// Washed-out brown and sick green: a polluted district.
    pub fn smog() -> Self {
        Self {
            exposure: -0.25,
            contrast: 0.88,
            saturation: 0.55,
            temperature: 0.2,
            tint: 0.25,
            shadows: Vec3::new(0.95, 0.9, 0.78),
            highlights: Vec3::new(1.02, 0.98, 0.82),
            vignette: 0.35,
        }
    }

    /// Clean blue and crisp: cold, clinical, or high in the mountains.
    pub fn cool() -> Self {
        Self {
            contrast: 1.08,
            saturation: 0.9,
            temperature: -0.4,
            shadows: Vec3::new(0.9, 0.95, 1.05),
            vignette: 0.1,
            ..Self::NEUTRAL
        }
    }

    /// Nearly grey and hard: a bleak, oppressive place.
    pub fn bleak() -> Self {
        Self {
            exposure: -0.15,
            contrast: 1.2,
            saturation: 0.25,
            temperature: -0.1,
            vignette: 0.45,
            ..Self::NEUTRAL
        }
    }

    pub fn validate(&self) -> EngineResult<()> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && (lo..=hi).contains(&v);
        let tints_ok = |v: Vec3| v.is_finite() && v.min_element() >= 0.0 && v.max_element() <= 4.0;
        if !ok(self.exposure, -8.0, 8.0)
            || !ok(self.contrast, 0.0, 4.0)
            || !ok(self.saturation, 0.0, 4.0)
            || !ok(self.temperature, -1.0, 1.0)
            || !ok(self.tint, -1.0, 1.0)
            || !ok(self.vignette, 0.0, 1.0)
            || !tints_ok(self.shadows)
            || !tints_ok(self.highlights)
        {
            return Err(EngineError::InvalidValue(
                "colour grade out of range (exposure -8..8, contrast and saturation 0..4, \
                 temperature, tint -1..1, vignette 0..1, tints 0..4)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// `self` moved `t` (0..1) of the way to `other`.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let f = |a: f32, b: f32| a + (b - a) * t;
        Self {
            exposure: f(self.exposure, other.exposure),
            contrast: f(self.contrast, other.contrast),
            saturation: f(self.saturation, other.saturation),
            temperature: f(self.temperature, other.temperature),
            tint: f(self.tint, other.tint),
            shadows: self.shadows.lerp(other.shadows, t),
            highlights: self.highlights.lerp(other.highlights, t),
            vignette: f(self.vignette, other.vignette),
        }
    }
}

/// Fog inside a [`FogVolume`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeFog {
    pub color: Color,
    /// Distance at which things have all but vanished into the fog.
    pub visibility_m: f32,
}

/// A fog volume handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FogVolumeId(pub u32);

/// A box with its own look: a grade, fog, or both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogVolume {
    pub center: Vec3,
    pub half_extents: Vec3,
    /// Metres over which the look fades in inside the box's faces.
    pub blend_m: f32,
    pub grade: Option<ColorGrade>,
    pub fog: Option<VolumeFog>,
    /// Higher wins where volumes overlap.
    pub priority: i32,
}

impl FogVolume {
    pub fn new(center: impl Into<Vec3>, half_extents: impl Into<Vec3>) -> Self {
        Self {
            center: center.into(),
            half_extents: half_extents.into(),
            blend_m: 10.0,
            grade: None,
            fog: None,
            priority: 0,
        }
    }

    pub fn grade(mut self, grade: ColorGrade) -> Self {
        self.grade = Some(grade);
        self
    }

    pub fn fog(mut self, color: Color, visibility_m: f32) -> Self {
        self.fog = Some(VolumeFog {
            color,
            visibility_m,
        });
        self
    }

    pub fn blend(mut self, metres: f32) -> Self {
        self.blend_m = metres;
        self
    }

    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("fog volume {what}")));
        if !self.center.is_finite()
            || !self.half_extents.is_finite()
            || self.half_extents.min_element() <= 0.0
        {
            return bad("needs a finite center and positive half extents");
        }
        if !self.blend_m.is_finite() || self.blend_m < 0.0 {
            return bad("blend must be finite and >= 0");
        }
        if let Some(g) = &self.grade {
            g.validate()?;
        }
        if let Some(f) = &self.fog {
            if !f.visibility_m.is_finite() || f.visibility_m < 1.0 {
                return bad("fog visibility must be >= 1 metre");
            }
        }
        Ok(())
    }

    /// How much of the volume's look applies at `p`: 0 outside, 1 deeper
    /// than `blend_m` inside, smooth in between.
    pub fn weight(&self, p: Vec3) -> f32 {
        let inside = (self.half_extents - (p - self.center).abs()).min_element();
        if inside <= 0.0 {
            return 0.0;
        }
        if self.blend_m <= 0.0 {
            return 1.0;
        }
        let t = (inside / self.blend_m).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Fog as it blends: colour and density (1 / m), plus the base haze's
/// thinning with height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FogState {
    pub color: Vec3,
    pub density: f32,
    pub base_y: f32,
    pub height_m: f32,
}

impl FogState {
    fn from_haze(h: Option<Haze>) -> Self {
        match h {
            Some(h) => Self {
                color: h.color.to_vec3(),
                density: h.density(),
                base_y: h.base_y,
                height_m: h.height_m,
            },
            None => Self {
                color: Vec3::ONE,
                density: 0.0,
                base_y: 0.0,
                height_m: 1_000.0,
            },
        }
    }

    fn lerp(&self, o: &Self, t: f32) -> Self {
        // A clear sky has no colour to keep: take the other's.
        let color = if self.density <= 0.0 {
            o.color
        } else if o.density <= 0.0 {
            self.color
        } else {
            self.color.lerp(o.color, t)
        };
        Self {
            color,
            density: self.density + (o.density - self.density) * t,
            base_y: self.base_y + (o.base_y - self.base_y) * t,
            height_m: self.height_m + (o.height_m - self.height_m) * t,
        }
    }

    pub fn haze(&self) -> Option<Haze> {
        (self.density > 1e-6).then(|| {
            let c = self.color;
            Haze::new(
                Color::rgb01(c.x, c.y, c.z).unwrap_or(Color::WHITE),
                1.978 / self.density,
            )
            .thinning_above(self.base_y, self.height_m)
        })
    }
}

/// The world's grading state: base grade, volumes and how far the camera is
/// into each one right now.
#[derive(Debug)]
pub(crate) struct Grading {
    pub base: ColorGrade,
    pub volumes: Vec<Option<FogVolume>>,
    /// Eased weight of each volume (same index as `volumes`).
    pub weights: Vec<f32>,
    pub transition_s: f32,
    /// False until the first tick, which snaps to the camera's place.
    started: bool,
}

impl Default for Grading {
    fn default() -> Self {
        Self {
            base: ColorGrade::NEUTRAL,
            volumes: Vec::new(),
            weights: Vec::new(),
            transition_s: 2.5,
            started: false,
        }
    }
}

impl Grading {
    pub fn add(&mut self, v: FogVolume) -> EngineResult<FogVolumeId> {
        v.validate()?;
        if let Some(i) = self.volumes.iter().position(|s| s.is_none()) {
            self.volumes[i] = Some(v);
            self.weights[i] = 0.0;
            return Ok(FogVolumeId(i as u32));
        }
        self.volumes.push(Some(v));
        self.weights.push(0.0);
        Ok(FogVolumeId(self.volumes.len() as u32 - 1))
    }

    pub fn get(&self, id: FogVolumeId) -> Option<&FogVolume> {
        self.volumes.get(id.0 as usize).and_then(|v| v.as_ref())
    }

    pub fn set(&mut self, id: FogVolumeId, v: FogVolume) -> EngineResult<()> {
        v.validate()?;
        match self.volumes.get_mut(id.0 as usize) {
            Some(slot @ Some(_)) => {
                *slot = Some(v);
                Ok(())
            }
            _ => Err(EngineError::InvalidValue(format!("no fog volume {}", id.0))),
        }
    }

    pub fn remove(&mut self, id: FogVolumeId) -> bool {
        match self.volumes.get_mut(id.0 as usize) {
            Some(slot @ Some(_)) => {
                *slot = None;
                self.weights[id.0 as usize] = 0.0;
                true
            }
            _ => false,
        }
    }

    pub fn count(&self) -> usize {
        self.volumes.iter().flatten().count()
    }

    /// Ease each volume's weight toward how deep `eye` is inside it.
    pub fn tick(&mut self, eye: Vec3, dt: f32) {
        // About 95% of the way there after `transition_s`.
        let k = if !self.started || self.transition_s <= 0.0 {
            1.0
        } else {
            1.0 - (-3.0 * dt.max(0.0) / self.transition_s).exp()
        };
        self.started = true;
        for (v, w) in self.volumes.iter().zip(self.weights.iter_mut()) {
            let target = v.as_ref().map_or(0.0, |v| v.weight(eye));
            *w += (target - *w) * k;
        }
    }

    /// The look now: the base grade and haze with each volume blended in by
    /// its eased weight, lowest priority first.
    pub fn compose(&self, base_haze: Option<Haze>) -> (ColorGrade, FogState) {
        let mut grade = self.base;
        let mut fog = FogState::from_haze(base_haze);
        let mut order: Vec<(usize, &FogVolume)> = self
            .volumes
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|v| (i, v)))
            .collect();
        order.sort_by_key(|(i, v)| (v.priority, *i));
        for (i, v) in order {
            let w = self.weights[i];
            if w <= 1e-4 {
                continue;
            }
            if let Some(g) = &v.grade {
                grade = grade.lerp(g, w);
            }
            if let Some(f) = &v.fog {
                let vol = FogState {
                    color: f.color.to_vec3(),
                    density: 1.978 / f.visibility_m,
                    ..fog
                };
                fog = fog.lerp(&vol, w);
            }
        }
        (grade, fog)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_into_a_volume_blends_without_a_pop() {
        let mut g = Grading::default();
        g.add(
            FogVolume::new((100.0, 0.0, 0.0), (50.0, 50.0, 50.0))
                .grade(ColorGrade::smog())
                .fog(Color::rgb(120, 100, 80), 80.0)
                .blend(20.0),
        )
        .unwrap();
        let haze = Some(Haze::new(Color::rgb(200, 220, 255), 2000.0));
        let dt = 1.0 / 60.0;
        // Fly from x = 0 into the volume at 20 m/s; the look must never jump.
        let mut last: Option<(f32, f32)> = None;
        let mut max_step = 0.0f32;
        let mut frames_to_settle = None;
        for i in 0..(8 * 60) {
            let x = (i as f32 * 20.0 * dt).min(100.0);
            g.tick(Vec3::new(x, 0.0, 0.0), dt);
            let (grade, fog) = g.compose(haze);
            if let Some((s, d)) = last {
                max_step = max_step
                    .max((grade.saturation - s).abs())
                    .max((fog.density - d).abs() / 0.025);
            }
            last = Some((grade.saturation, fog.density));
            if frames_to_settle.is_none() && g.weights[0] > 0.5 {
                frames_to_settle = Some(i);
            }
        }
        assert!(max_step < 0.02, "largest change in a frame: {max_step}");
        // Not instant: half way takes over a second from the edge (x = 50).
        assert!(frames_to_settle.unwrap() > 150 + 60, "{frames_to_settle:?}");
        let (grade, fog) = g.compose(haze);
        assert!((grade.saturation - ColorGrade::smog().saturation).abs() < 0.02);
        assert!((fog.density - 1.978 / 80.0).abs() < 0.002);
        // Outside again it eases back.
        for _ in 0..(5 * 60) {
            g.tick(Vec3::new(-50.0, 0.0, 0.0), dt);
        }
        assert!((g.compose(haze).0.saturation - 1.0).abs() < 0.01);
    }

    #[test]
    fn grades_validate_and_volumes_weigh() {
        assert!(ColorGrade {
            contrast: -1.0,
            ..ColorGrade::NEUTRAL
        }
        .validate()
        .is_err());
        for g in [
            ColorGrade::warm(),
            ColorGrade::smog(),
            ColorGrade::cool(),
            ColorGrade::bleak(),
        ] {
            g.validate().unwrap();
        }
        let v = FogVolume::new(Vec3::ZERO, Vec3::splat(10.0)).blend(5.0);
        assert_eq!(v.weight(Vec3::new(11.0, 0.0, 0.0)), 0.0);
        assert_eq!(v.weight(Vec3::ZERO), 1.0);
        assert!((v.weight(Vec3::new(7.5, 0.0, 0.0)) - 0.5).abs() < 1e-5);
    }
}
