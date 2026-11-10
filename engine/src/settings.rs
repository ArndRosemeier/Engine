//! Everything a game sets once at startup, in one typed struct.
//!
//! ```ignore
//! let settings = Settings::default()
//!     .window(1280, 720)
//!     .vsync(false)
//!     .quality(Quality::Medium);
//! Engine::run_with_settings("my game", settings, |world, frame| Ok(()))
//! ```
//!
//! Each field has a `World` setter too (`set_quality`, `set_taa`, ...) for
//! changes while the game runs; the startup-only ones (window size, MSAA,
//! audio, fixed step, limits) live only here. `ENGINE_*` environment
//! variables still override the game's settings, for debugging and A/B
//! comparisons; [`crate::World::settings`] says which ones were set, and
//! verify reports list them.

use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::quality::Quality;
use crate::world::ToneMap;
use serde::Serialize;

/// Startup settings. `Settings::default()` is what `Engine::run` uses.
#[derive(Clone, Debug, Serialize)]
pub struct Settings {
    /// Window size in logical pixels (default 1920 x 1080, at least 320 x 180).
    pub width: u32,
    pub height: u32,
    /// Wait for the display's refresh (default on). Off shows the real frame
    /// rate; the examples turn it off.
    pub vsync: bool,
    /// Quality preset: shadows, AO, GI, LOD distance, grass (default High).
    #[serde(serialize_with = "quality_name")]
    pub quality: Quality,
    /// Temporal anti-aliasing (default on).
    pub taa: bool,
    /// MSAA samples for the 3D scene, 1 or 4. `None` (default) picks 1
    /// while TAA is on and 4 without it.
    pub msaa: Option<u32>,
    /// Tone-mapping curve (default AgX).
    #[serde(serialize_with = "tone_map_name")]
    pub tone_map: ToneMap,
    /// Draw the scene at this share of the window and upscale, 0.5 to 1.0
    /// (default 1.0).
    pub render_scale: f32,
    /// Keep this frame rate by trading detail (default `None`: never trade).
    pub target_fps: Option<f32>,
    /// How far the camera sees, in metres (default `None`: the engine's own).
    pub view_distance: Option<f32>,
    /// Open a sound device (default on). Off runs silent.
    pub audio: bool,
    /// Step every frame by exactly this many seconds instead of the real
    /// frame time (default `None`). For repeatable runs and recordings.
    pub fixed_dt: Option<f32>,
    /// The "Engine x.yyy · What's new" button in the corner (default on).
    pub whats_new_button: bool,
    /// Caps on model sizes, paints and instances (protects against
    /// accidental huge inputs).
    #[serde(skip)]
    pub limits: EngineLimits,
}

fn quality_name<S: serde::Serializer>(q: &Quality, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(q.name())
}

fn tone_map_name<S: serde::Serializer>(t: &ToneMap, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(match t {
        ToneMap::Agx => "agx",
        ToneMap::Aces => "aces",
    })
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            vsync: true,
            quality: Quality::High,
            taa: true,
            msaa: None,
            tone_map: ToneMap::Agx,
            render_scale: 1.0,
            target_fps: None,
            view_distance: None,
            audio: true,
            fixed_dt: None,
            whats_new_button: true,
            limits: EngineLimits::default(),
        }
    }
}

impl Settings {
    /// Window size in logical pixels.
    pub fn window(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    pub fn vsync(mut self, on: bool) -> Self {
        self.vsync = on;
        self
    }
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }
    pub fn taa(mut self, on: bool) -> Self {
        self.taa = on;
        self
    }
    /// 1 or 4 samples.
    pub fn msaa(mut self, samples: u32) -> Self {
        self.msaa = Some(samples);
        self
    }
    pub fn tone_map(mut self, tone_map: ToneMap) -> Self {
        self.tone_map = tone_map;
        self
    }
    pub fn render_scale(mut self, scale: f32) -> Self {
        self.render_scale = scale;
        self
    }
    pub fn target_fps(mut self, fps: f32) -> Self {
        self.target_fps = Some(fps);
        self
    }
    pub fn view_distance(mut self, metres: f32) -> Self {
        self.view_distance = Some(metres);
        self
    }
    pub fn audio(mut self, on: bool) -> Self {
        self.audio = on;
        self
    }
    pub fn fixed_dt(mut self, seconds: f32) -> Self {
        self.fixed_dt = Some(seconds);
        self
    }
    pub fn whats_new_button(mut self, show: bool) -> Self {
        self.whats_new_button = show;
        self
    }
    pub fn limits(mut self, limits: EngineLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Check every value; the message says which one is wrong and why.
    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: String| Err(EngineError::InvalidValue(format!("settings: {what}")));
        if self.width < 320 || self.height < 180 {
            return bad(format!(
                "window {}x{} is too small (at least 320x180)",
                self.width, self.height
            ));
        }
        if let Some(n) = self.msaa {
            if n != 1 && n != 4 {
                return bad(format!("msaa must be 1 or 4, got {n}"));
            }
        }
        if !self.render_scale.is_finite() || !(0.5..=1.0).contains(&self.render_scale) {
            return bad(format!(
                "render_scale must be within 0.5..=1.0, got {}",
                self.render_scale
            ));
        }
        if let Some(fps) = self.target_fps {
            if !fps.is_finite() || fps <= 0.0 {
                return bad(format!("target_fps must be above 0, got {fps}"));
            }
        }
        if let Some(m) = self.view_distance {
            if !m.is_finite() || m <= 0.0 {
                return bad(format!("view_distance must be above 0, got {m}"));
            }
        }
        if let Some(dt) = self.fixed_dt {
            if !dt.is_finite() || dt <= 0.0 || dt > 1.0 {
                return bad(format!("fixed_dt must be within (0, 1] seconds, got {dt}"));
            }
        }
        Ok(())
    }

    /// The `ENGINE_*` variables set in this process's environment, as
    /// `NAME=value`, sorted. They override the game's settings.
    pub fn env_overrides() -> Vec<String> {
        let mut out: Vec<String> = std::env::vars()
            .filter(|(k, _)| k.starts_with("ENGINE_"))
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_builders_set_fields() {
        Settings::default().validate().unwrap();
        let s = Settings::default()
            .window(1280, 720)
            .vsync(false)
            .quality(Quality::Low)
            .msaa(4)
            .render_scale(0.75)
            .fixed_dt(1.0 / 30.0);
        s.validate().unwrap();
        assert_eq!(
            (s.width, s.height, s.vsync, s.msaa),
            (1280, 720, false, Some(4))
        );
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(json["quality"], "low");
        assert_eq!(json["tone_map"], "agx");
    }

    #[test]
    fn bad_values_name_the_setting() {
        for (s, word) in [
            (Settings::default().window(100, 100), "window"),
            (Settings::default().msaa(2), "msaa"),
            (Settings::default().render_scale(0.2), "render_scale"),
            (Settings::default().target_fps(0.0), "target_fps"),
            (Settings::default().fixed_dt(f32::NAN), "fixed_dt"),
        ] {
            let err = s.validate().unwrap_err().to_string();
            assert!(err.contains(word), "{err}");
        }
    }
}
