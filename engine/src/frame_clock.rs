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
    /// Slow average of the raw frame time (about the last second).
    long: Option<f32>,
    enabled: bool,
}

impl FrameClock {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            smooth: None,
            debt: 0.0,
            outliers: 0,
            long: None,
            enabled,
        }
    }

    /// From the environment: `ENGINE_SMOOTH_DT=0` hands out raw frame times.
    pub(crate) fn from_env() -> Self {
        Self::new(std::env::var("ENGINE_SMOOTH_DT").map_or(true, |v| v.trim() != "0"))
    }

    /// The step for a frame that really took `raw` seconds.
    /// `refresh_hz`: the display rate when frames are vsynced (None
    /// otherwise). While the frame rate matches it, every step is one
    /// refresh interval: the display shows frames evenly even when the
    /// moments the CPU gets through bunch up (held keys on Windows did that).
    pub(crate) fn tick(&mut self, raw: f32, refresh_hz: Option<f32>) -> f32 {
        let raw = if raw.is_finite() { raw.max(0.0) } else { 0.0 };
        if !self.enabled {
            return raw.min(MAX_DT);
        }
        let long = self.long.map_or(raw, |l| l + (raw - l) * 0.03);
        self.long = Some(long);
        if let Some(hz) = refresh_hz.filter(|hz| *hz > 20.0) {
            let interval = 1.0 / hz;
            if (long - interval).abs() < interval * 0.08 && raw < interval * 4.0 {
                self.debt += raw - interval;
                self.debt = self.debt.clamp(-interval * 4.0, interval * 4.0);
                let repay = (self.debt * 0.05).clamp(-interval * 0.05, interval * 0.05);
                self.debt -= repay;
                self.smooth = Some(interval);
                self.outliers = 0;
                return (interval + repay).min(MAX_DT);
            }
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

    /// Returns whether it printed a line just now.
    pub(crate) fn add(&mut self, raw: f32) -> bool {
        self.samples.push(raw);
        self.since += raw;
        if self.since < 2.0 {
            return false;
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
        true
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
            let used = clock.tick(raw, None);
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
    fn vsynced_bursts_become_even_refresh_steps() {
        // Acquire times bunched by input: 16.5, 26.5, 7 ms (average 1/60 s).
        let mut clock = FrameClock::new(true);
        let pattern = [0.0165, 0.0265, 0.0070];
        let mut steps = Vec::new();
        let (mut raw_sum, mut used_sum) = (0.0, 0.0);
        for i in 0..600 {
            let raw = pattern[i % 3];
            let used = clock.tick(raw, Some(60.0));
            raw_sum += raw;
            used_sum += used;
            if i > 120 {
                steps.push(used);
            }
        }
        for s in &steps {
            assert!((s - 1.0 / 60.0).abs() < 0.001, "uneven step {s}");
        }
        assert!((raw_sum - used_sum).abs() < 0.1);
    }

    #[test]
    fn hitches_pass_through_and_are_capped() {
        let mut clock = FrameClock::new(true);
        for _ in 0..50 {
            clock.tick(1.0 / 60.0, None);
        }
        assert!((clock.tick(0.05, None) - 0.05).abs() < 1e-6);
        assert_eq!(clock.tick(2.0, None), MAX_DT);
        assert_eq!(FrameClock::new(false).tick(0.0123, None), 0.0123);
        assert_eq!(clock.tick(f32::NAN, None), 0.0);
    }
}

/// Paces frames to the display's refresh rate when vsync was asked for but
/// presentation does not wait for it (seen with a window on a monitor driven
/// by another GPU). Rendering far faster than the display shows frames makes
/// each shown frame sample the motion at an uneven moment: judder.
pub(crate) struct FramePacer {
    next: Option<std::time::Instant>,
    active: bool,
}

impl FramePacer {
    pub(crate) fn new() -> Self {
        Self {
            next: None,
            active: false,
        }
    }

    /// Turn pacing on once frames clearly outrun the display (`fps` well
    /// above `refresh_hz`); off when vsync is not wanted.
    pub(crate) fn observe(&mut self, vsync: bool, fps: f32, refresh_hz: Option<f32>) {
        match refresh_hz {
            Some(hz) if vsync && hz > 20.0 => {
                if fps > hz * 1.15 {
                    self.active = true;
                }
            }
            _ => {
                self.active = false;
                self.next = None;
            }
        }
    }

    /// When the next frame may start (`None`: now). Keeps a fixed schedule
    /// so the frames line up with the display, and starts over after a
    /// stall instead of rushing to catch up.
    pub(crate) fn schedule(
        &mut self,
        now: std::time::Instant,
        refresh_hz: f32,
    ) -> Option<std::time::Instant> {
        if !self.active || refresh_hz <= 0.0 {
            return None;
        }
        let interval = std::time::Duration::from_secs_f64(1.0 / refresh_hz as f64);
        let due = match self.next {
            Some(due) if now <= due + interval => due,
            _ => now,
        };
        self.next = Some(due + interval);
        (due > now).then_some(due)
    }

    #[cfg(test)]
    pub(crate) fn active(&self) -> bool {
        self.active
    }
}

/// Wait until `until`: sleep most of the way, then spin the last stretch
/// (sleep alone can overshoot by a millisecond or more).
pub(crate) fn wait_until(until: std::time::Instant) {
    let spin = std::time::Duration::from_micros(1500);
    loop {
        let now = std::time::Instant::now();
        if now >= until {
            return;
        }
        let left = until - now;
        if left > spin {
            std::thread::sleep(left - spin);
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod pacer_tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn paces_only_when_frames_outrun_the_display() {
        let mut p = FramePacer::new();
        p.observe(true, 62.0, Some(60.0));
        assert!(!p.active());
        p.observe(true, 200.0, Some(60.0));
        assert!(p.active());
        p.observe(false, 200.0, Some(60.0));
        assert!(!p.active());
    }

    #[test]
    fn keeps_a_fixed_schedule_and_resyncs_after_a_stall() {
        let mut p = FramePacer::new();
        p.observe(true, 200.0, Some(50.0));
        let t0 = Instant::now();
        assert_eq!(p.schedule(t0, 50.0), None);
        // A fast frame waits for the next slot, 20 ms after the first.
        let due = p.schedule(t0 + Duration::from_millis(5), 50.0).unwrap();
        assert_eq!(due - t0, Duration::from_millis(20));
        // Long stall: start over from now.
        let late = t0 + Duration::from_millis(500);
        assert_eq!(p.schedule(late, 50.0), None);
        let due = p.schedule(late + Duration::from_millis(1), 50.0).unwrap();
        assert_eq!(due - late, Duration::from_millis(20));
    }
}
