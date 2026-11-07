//! `ENGINE_CPU_TRACE=1`: every two seconds, print where the CPU time of a
//! frame went (game update, each world tick, renderer sync and drawing),
//! as average and worst milliseconds per frame, biggest first.

use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct CpuTrace {
    spans: HashMap<&'static str, (f64, f32)>,
    frames: u32,
    frame_ms: f64,
    frame_max: f32,
    since: f32,
}

impl CpuTrace {
    pub(crate) fn from_env() -> Option<Self> {
        std::env::var("ENGINE_CPU_TRACE")
            .ok()
            .filter(|v| !v.trim().is_empty() && v.trim() != "0")
            .map(|_| Self::default())
    }

    pub(crate) fn add(&mut self, name: &'static str, ms: f32) {
        let e = self.spans.entry(name).or_insert((0.0, 0.0));
        e.0 += f64::from(ms);
        e.1 = e.1.max(ms);
    }

    /// End of a frame that took `frame_ms` of wall time.
    pub(crate) fn end_frame(&mut self, frame_ms: f32) {
        self.frames += 1;
        self.frame_ms += f64::from(frame_ms);
        self.frame_max = self.frame_max.max(frame_ms);
        self.since += frame_ms / 1000.0;
        if self.since < 2.0 {
            return;
        }
        let n = f64::from(self.frames.max(1));
        let mut rows: Vec<_> = self
            .spans
            .iter()
            .map(|(name, (sum, max))| (*name, sum / n, *max))
            .filter(|(_, avg, max)| *avg >= 0.05 || *max >= 1.0)
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut text = format!(
            "cpu trace: {} frames, frame {:.2} ms avg, {:.1} max (ms per frame: avg / max)\n",
            self.frames,
            self.frame_ms / n,
            self.frame_max
        );
        for (name, avg, max) in rows {
            text.push_str(&format!("  {name:<22} {avg:>6.2} / {max:>6.1}\n"));
        }
        eprint!("{text}");
        *self = Self::default();
    }
}

/// Spans a part of the engine collects for the trace, taken by the app.
#[derive(Default)]
pub(crate) struct Spans {
    pub(crate) on: bool,
    list: Vec<(&'static str, f32)>,
}

impl Spans {
    pub(crate) fn mark(&mut self, name: &'static str, since: &mut std::time::Instant) {
        if self.on {
            let now = std::time::Instant::now();
            self.list
                .push((name, (now - *since).as_secs_f32() * 1000.0));
            *since = now;
        }
    }

    pub(crate) fn take(&mut self) -> Vec<(&'static str, f32)> {
        std::mem::take(&mut self.list)
    }
}
