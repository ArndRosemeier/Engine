//! Quality presets and a frame-rate auto-tuner.
//!
//! [`World::set_quality`](crate::world::World::set_quality) sets shadows,
//! ambient occlusion, global illumination and the LOD distance bias in one
//! call. [`World::set_target_fps`](crate::world::World::set_target_fps) then
//! keeps the frame rate up by itself: it pulls LOD distances in when frames
//! run slow and lets them back out when there is room, and only drops a
//! whole feature (GI first, then AO) when that is not enough.
//!
//! The tuner measures wall-clock frame time, so it reads best with vsync off;
//! with vsync on it still steps down when frames miss the target.

use crate::world::{AmbientOcclusion, GlobalIllumination, ShadowSettings};

/// A rendering quality preset. `High` is what a new `World` starts with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Quality {
    /// Short shadows, no AO or GI, near LOD swaps. For weak GPUs.
    Low,
    /// Full shadows and AO, no GI.
    Medium,
    /// Everything on (the default).
    #[default]
    High,
    /// Everything on, LOD detail kept half again as far.
    Ultra,
}

impl Quality {
    pub const ALL: [Quality; 4] = [Quality::Low, Quality::Medium, Quality::High, Quality::Ultra];

    /// Parse `"low"`, `"medium"`, `"high"` or `"ultra"` (any case).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|q| q.name().eq_ignore_ascii_case(name.trim()))
    }

    pub fn name(self) -> &'static str {
        match self {
            Quality::Low => "low",
            Quality::Medium => "medium",
            Quality::High => "high",
            Quality::Ultra => "ultra",
        }
    }

    /// The settings this preset stands for.
    pub fn settings(self) -> QualitySettings {
        let shadows = match self {
            Quality::Low => ShadowSettings {
                cascade_end_m: [8.0, 25.0, 70.0],
                ..ShadowSettings::default()
            },
            _ => ShadowSettings::default(),
        };
        QualitySettings {
            shadows: Some(shadows),
            ambient_occlusion: (self >= Quality::Medium).then(AmbientOcclusion::default),
            global_illumination: (self >= Quality::High).then(GlobalIllumination::default),
            lod_bias: match self {
                Quality::Low => 0.6,
                Quality::Medium => 0.85,
                Quality::High => 1.0,
                Quality::Ultra => 1.5,
            },
        }
    }
}

/// What a [`Quality`] preset sets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualitySettings {
    pub shadows: Option<ShadowSettings>,
    pub ambient_occlusion: Option<AmbientOcclusion>,
    pub global_illumination: Option<GlobalIllumination>,
    pub lod_bias: f32,
}

/// Lowest LOD bias the tuner goes to before it drops features.
pub(crate) const MIN_TUNED_BIAS: f32 = 0.35;

/// What the tuner wants changed this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TuneStep {
    Hold,
    Bias(f32),
    DropGi,
    DropAo,
}

/// State of [`World::set_target_fps`](crate::world::World::set_target_fps).
#[derive(Clone, Debug)]
pub(crate) struct AutoTune {
    pub target_fps: f32,
    /// Bias of the chosen preset: the tuner never goes above it.
    pub max_bias: f32,
    window_s: f32,
    window_frames: u32,
    slow_windows: u32,
}

impl AutoTune {
    pub fn new(target_fps: f32, max_bias: f32) -> Self {
        Self {
            target_fps,
            max_bias,
            window_s: 0.0,
            window_frames: 0,
            slow_windows: 0,
        }
    }

    /// Feed one frame; every half second decide on a step.
    pub fn tick(&mut self, dt: f32, bias: f32, has_gi: bool, has_ao: bool) -> TuneStep {
        // Ignore hitches (loading, window drags): they are not the steady cost.
        if !(dt.is_finite() && dt > 0.0 && dt < 0.5) {
            return TuneStep::Hold;
        }
        self.window_s += dt;
        self.window_frames += 1;
        if self.window_s < 0.5 {
            return TuneStep::Hold;
        }
        let fps = self.window_frames as f32 / self.window_s;
        self.window_s = 0.0;
        self.window_frames = 0;
        let target = self.target_fps;
        if fps < target * 0.95 {
            if bias > MIN_TUNED_BIAS + 1e-3 {
                self.slow_windows = 0;
                // Cut harder the further behind we are.
                let k = (fps / target).clamp(0.6, 0.92);
                return TuneStep::Bias((bias * k).max(MIN_TUNED_BIAS));
            }
            // Bias is spent: after two more slow seconds drop a feature.
            self.slow_windows += 1;
            if self.slow_windows >= 4 {
                self.slow_windows = 0;
                if has_gi {
                    return TuneStep::DropGi;
                }
                if has_ao {
                    return TuneStep::DropAo;
                }
            }
            return TuneStep::Hold;
        }
        self.slow_windows = 0;
        // Room to spare (or pinned at the vsync cap): ease detail back out.
        if fps >= target * 0.99 && bias < self.max_bias - 1e-3 {
            return TuneStep::Bias((bias * 1.04).min(self.max_bias));
        }
        TuneStep::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(t: &mut AutoTune, fps: f32, seconds: f32, bias: &mut f32, gi: &mut bool, ao: &mut bool) {
        let frames = (fps * seconds) as u32;
        for _ in 0..frames {
            match t.tick(1.0 / fps, *bias, *gi, *ao) {
                TuneStep::Bias(b) => *bias = b,
                TuneStep::DropGi => *gi = false,
                TuneStep::DropAo => *ao = false,
                TuneStep::Hold => {}
            }
        }
    }

    #[test]
    fn presets_are_ordered() {
        let biases: Vec<f32> = Quality::ALL.iter().map(|q| q.settings().lod_bias).collect();
        assert!(biases.windows(2).all(|w| w[0] < w[1]));
        assert!(Quality::Low.settings().global_illumination.is_none());
        assert!(Quality::High.settings().global_illumination.is_some());
        assert_eq!(Quality::from_name(" Ultra"), Some(Quality::Ultra));
        assert_eq!(Quality::from_name("max"), None);
    }

    #[test]
    fn tuner_steps_down_then_drops_features_then_recovers() {
        let mut t = AutoTune::new(60.0, 1.0);
        let (mut bias, mut gi, mut ao) = (1.0, true, true);
        run(&mut t, 40.0, 2.0, &mut bias, &mut gi, &mut ao);
        assert!(bias < 0.6, "bias fell to {bias}");
        assert!(gi, "bias goes first");
        run(&mut t, 40.0, 10.0, &mut bias, &mut gi, &mut ao);
        assert!((bias - MIN_TUNED_BIAS).abs() < 1e-3);
        assert!(!gi && !ao, "features dropped once bias was spent");
        run(&mut t, 75.0, 30.0, &mut bias, &mut gi, &mut ao);
        assert!((bias - 1.0).abs() < 1e-3, "bias recovered to {bias}");
    }

    #[test]
    fn tuner_holds_at_target() {
        let mut t = AutoTune::new(60.0, 1.0);
        let (mut bias, mut gi, mut ao) = (1.0, true, true);
        run(&mut t, 60.0, 20.0, &mut bias, &mut gi, &mut ao);
        assert_eq!((bias, gi, ao), (1.0, true, true));
    }
}
