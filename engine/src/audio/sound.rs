//! [`Sound`]: a clip decoded into memory, ready to start without any I/O.

use crate::error::{EngineError, EngineResult};
use std::path::Path;
use std::sync::Arc;

/// Decoded samples, shared by every voice that plays them.
#[derive(Debug)]
pub(crate) struct SoundData {
    /// Interleaved samples in -1..1.
    pub samples: Vec<f32>,
    /// 1 (mono) or 2 (stereo).
    pub channels: u16,
    pub rate: u32,
}

impl SoundData {
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// One frame as (left, right); mono is copied to both.
    #[inline]
    pub fn frame(&self, i: usize) -> (f32, f32) {
        if self.channels == 1 {
            let s = self.samples[i];
            (s, s)
        } else {
            (self.samples[2 * i], self.samples[2 * i + 1])
        }
    }
}

/// A sound clip in memory. Cheap to clone (the samples are shared).
///
/// Load one from a file (`.wav`, `.ogg`, `.mp3`, `.flac`) with
/// [`Sound::load`], or make one in code: the `Sound::explosion`,
/// `Sound::fire_loop` and similar constructors synthesise ready-made
/// effects, so a procedural game can have sound without any audio files.
#[derive(Clone, Debug)]
pub struct Sound(pub(crate) Arc<SoundData>);

impl PartialEq for Sound {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Sound {
    /// Decode a sound file (`.wav`, `.ogg`, `.mp3` or `.flac`).
    pub fn load(path: impl AsRef<Path>) -> EngineResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes).map_err(|e| {
            EngineError::Audio(format!("{} could not be decoded: {e}", path.display()))
        })
    }

    /// Decode a sound file already in memory.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> EngineResult<Self> {
        use rodio::Source;
        let decoder = rodio::Decoder::new(std::io::Cursor::new(bytes.into()))
            .map_err(|e| EngineError::Audio(format!("not a playable sound: {e}")))?;
        let channels = decoder.channels();
        let rate = decoder.sample_rate();
        let raw: Vec<f32> = decoder.collect();
        if channels == 0 || rate == 0 {
            return Err(EngineError::Audio("sound has no channels".into()));
        }
        // Keep at most two channels: games mix in stereo.
        let samples = match channels {
            1 | 2 => raw,
            n => raw.chunks(n as usize).flat_map(|f| [f[0], f[1]]).collect(),
        };
        Ok(Self(Arc::new(SoundData {
            samples,
            channels: channels.min(2),
            rate,
        })))
    }

    /// A mono sound from samples in -1..1 at `rate` samples per second.
    pub fn from_mono(samples: Vec<f32>, rate: u32) -> EngineResult<Self> {
        Self::from_samples(samples, 1, rate)
    }

    /// A sound from interleaved samples (`channels` 1 or 2).
    pub fn from_samples(samples: Vec<f32>, channels: u16, rate: u32) -> EngineResult<Self> {
        if !(1..=2).contains(&channels) {
            return Err(EngineError::InvalidValue(format!(
                "a sound has 1 or 2 channels, got {channels}"
            )));
        }
        if rate == 0 || !samples.len().is_multiple_of(channels as usize) {
            return Err(EngineError::InvalidValue(
                "sound samples must fill whole frames at a rate above 0".into(),
            ));
        }
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(EngineError::InvalidValue(
                "sound samples must be finite".into(),
            ));
        }
        Ok(Self(Arc::new(SoundData {
            samples,
            channels,
            rate,
        })))
    }

    /// Length in seconds.
    pub fn duration(&self) -> f32 {
        self.0.frames() as f32 / self.0.rate as f32
    }

    pub fn channels(&self) -> u16 {
        self.0.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.0.rate
    }
}

#[cfg(test)]
mod tests {
    use super::Sound;

    #[test]
    fn samples_must_fill_whole_frames() {
        assert!(Sound::from_samples(vec![0.0; 3], 2, 48_000).is_err());
        assert!(Sound::from_samples(vec![0.0; 4], 2, 48_000).is_ok());
        assert!(Sound::from_samples(vec![0.0; 4], 3, 48_000).is_err());
        assert!(Sound::from_mono(vec![f32::NAN], 48_000).is_err());
    }

    #[test]
    fn duration_follows_rate() {
        let s = Sound::from_mono(vec![0.0; 24_000], 48_000).unwrap();
        assert!((s.duration() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn garbage_bytes_are_rejected() {
        assert!(Sound::from_bytes(b"not a sound".to_vec()).is_err());
    }
}
