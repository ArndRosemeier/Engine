//! Frame timing: a steady `dt` for movement, and optional frame-time stats.
//!
//! The time between two redraws wobbles by a millisecond or more even when
//! the display shows frames at a perfectly even rate (the OS wakes the loop a
//! little early or late). Moving the camera by that raw time makes motion
//! shake. The clock follows the average frame time instead, and repays the
//! difference gradually so the game's total time never drifts from the real
//! one. Real hitches (a frame far longer than usual) pass through at once.

/// How far a frame may stray from the average and still count as jitter.
const JITTER: f32 = 0.5;
/// Longest step handed to the game, so one long stall does not fling the
/// camera across the map.
const MAX_DT: f32 = 0.1;

pub(crate) struct FrameClock {
    smooth: Option<f32>,
    /// Real time not yet handed out (positive) or handed out early.
    debt: f32,
    /// Frames in a row that were far from the average.
    outliers: u32,
    enabled: bool,
}

impl FrameClock {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            smooth: None,
            debt: 0.0,
            outliers: 0,
            enabled,
        }
    }

    /// From the environment: `ENGINE_SMOOTH_DT=0` hands out raw frame times.
    pub(crate) fn from_env() -> Self {
        Self::new(std::env::var("ENGINE_SMOOTH_DT").map_or(true, |v| v.trim() != "0"))
    }

    /// The step for a frame that really took `raw` seconds.
    pub(crate) fn tick(&mut self, raw: f32) -> f32 {
        let raw = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
        if !self.enabled {
            return raw.min(MAX_DT);
        }
        let Some(prev) = self.smooth else {
            self.smooth = Some(raw);
            return raw.min(MAX_DT);
        };
        if (raw - prev).abs() > prev * JITTER {
            // A hitch passes straight through; several odd frames in a row
            // mean the frame rate itself changed, so follow it.
            self.outliers += 1;
            if self.outliers >= 4 {
                self.smooth = Some(raw);
                self.debt = 0.0;
                self.outliers = 0;
            }
            return raw.min(MAX_DT);
        }
        self.outliers = 0;
        let avg = prev + (raw - prev) * 0.08;
        self.smooth = Some(avg);
        self.debt += raw - avg;
        // Repay a tenth of the difference per frame, never more than a few
        // percent of a frame, so the step stays even.
        let repay = (self.debt * 0.1).clamp(-avg * 0.05, avg * 0.05);
        self.debt -= repay;
        (avg + repay).min(MAX_DT)
    }
}

/// `ENGINE_FRAME_STATS=1`: every two seconds, print frame-time statistics
/// (to judge smoothness: a high max or spread means visible stutter).
pub(crate) struct FrameStats {
    samples: Vec<f32>,
    since: f32,
}

impl FrameStats {
    pub(crate) fn from_env() -> Option<Self> {
        std::env::var("ENGINE_FRAME_STATS")
            .ok()
            .filter(|v| v.trim() != "0")
            .map(|_| Self {
                samples: Vec::with_capacity(512),
                since: 0.0,
            })
    }

    pub(crate) fn add(&mut self, raw: f32) {
        self.samples.push(raw);
        self.since += raw;
        if self.since < 2.0 {
            return;
        }
        let n = self.samples.len() as f32;
        let mean = self.samples.iter().sum::<f32>() / n;
        let var = self.samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
        let max = self.samples.iter().cloned().fold(0.0, f32::max);
        let min = self.samples.iter().cloned().fold(f32::MAX, f32::min);
        let long = self.samples.iter().filter(|&&s| s > mean * 1.5).count();
        eprintln!(
            "frame stats: {:.0} FPS, dt {:.2} ms avg, {:.2} min, {:.2} max, {:.2} spread, {} long frames of {}",
            1.0 / mean.max(1e-6),
            mean * 1000.0,
            min * 1000.0,
            max * 1000.0,
            var.sqrt() * 1000.0,
            long,
            self.samples.len()
        );
        self.samples.clear();
        self.since = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_is_smoothed_and_total_time_kept() {
        let mut clock = FrameClock::new(true);
        let mut raw_sum = 0.0;
        let mut used_sum = 0.0;
        let mut steps = Vec::new();
        for i in 0..600 {
            // 120 Hz with +-1.5 ms of wake-up noise.
            let raw = 1.0 / 120.0 + if i % 2 == 0 { 0.0015 } else { -0.0015 };
            let used = clock.tick(raw);
            raw_sum += raw;
            used_sum += used;
            if i > 100 {
                steps.push(used);
            }
        }
        let max = steps.iter().cloned().fold(0.0, f32::max);
        let min = steps.iter().cloned().fold(1.0, f32::min);
        assert!(max - min < 0.0006, "steps still wobble: {min}..{max}");
        assert!((raw_sum - used_sum).abs() < 0.01, "{raw_sum} vs {used_sum}");
    }

    #[test]
    fn hitches_pass_through_and_are_capped() {
        let mut clock = FrameClock::new(true);
        for _ in 0..50 {
            clock.tick(1.0 / 60.0);
        }
        assert!((clock.tick(0.05) - 0.05).abs() < 1e-6);
        assert_eq!(clock.tick(2.0), MAX_DT);
        assert_eq!(FrameClock::new(false).tick(0.0123), 0.0123);
        assert_eq!(clock.tick(f32::NAN), 0.0);
    }
}
