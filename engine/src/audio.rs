//! Clip playback through the default output device.
//!
//! The game owns policy (when to start a village loop, how loud a river is).
//! This module loads WAV bytes, keeps the output stream alive, and exposes
//! voices the game can fade, stop, or poll.

use crate::error::{EngineError, EngineResult};
use rodio::{Decoder, OutputStream, OutputStreamBuilder, Sink, Source};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

/// A decoded-ready WAV sitting in memory so a voice can start without I/O.
#[derive(Clone)]
struct Clip {
    bytes: Arc<[u8]>,
}

/// Handle to a clip loaded into an [`Audio`] mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClipId(usize);

/// Handle to a playing (or fading) voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoiceId(usize);

/// How a clip should start.
#[derive(Clone, Copy, Debug)]
pub struct Play {
    pub looped: bool,
    pub volume: f32,
}

struct Voice {
    sink: Sink,
}

/// Default-device mixer. Dropping this stops every voice.
pub struct Audio {
    stream: OutputStream,
    clips: Vec<Clip>,
    voices: Vec<Option<Voice>>,
}

impl Audio {
    /// Open the default output device. Fails out loud when the OS has none.
    pub fn open() -> EngineResult<Self> {
        let stream = OutputStreamBuilder::open_default_stream().map_err(|err| {
            EngineError::Audio(format!("could not open the default output device: {err}"))
        })?;
        Ok(Self {
            stream,
            clips: Vec::new(),
            voices: Vec::new(),
        })
    }

    /// Read a WAV from disk and keep the bytes for later voices.
    pub fn load_wav(&mut self, path: &Path) -> EngineResult<ClipId> {
        let bytes = std::fs::read(path)?;
        decode_wav(&bytes).map_err(|err| {
            EngineError::Audio(format!("{} is not a playable WAV: {err}", path.display()))
        })?;
        self.clips.push(Clip {
            bytes: Arc::from(bytes),
        });
        Ok(ClipId(self.clips.len() - 1))
    }

    /// Start a voice. Volume must be finite and non-negative.
    pub fn play(&mut self, clip: ClipId, play: Play) -> EngineResult<VoiceId> {
        if !play.volume.is_finite() || play.volume < 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "voice volume must be a finite value >= 0, got {}",
                play.volume
            )));
        }
        let clip = self.clips.get(clip.0).ok_or(EngineError::InvalidValue(
            "play was given a clip this mixer never loaded".into(),
        ))?;
        let source = decode_wav(clip.bytes.as_ref()).expect("clip already decoded at load");
        let sink = Sink::connect_new(self.stream.mixer());
        sink.set_volume(play.volume);
        if play.looped {
            sink.append(source.repeat_infinite());
        } else {
            sink.append(source);
        }
        let id = self.alloc_voice(Voice { sink });
        Ok(id)
    }

    pub fn set_volume(&self, voice: VoiceId, volume: f32) -> EngineResult<()> {
        if !volume.is_finite() || volume < 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "voice volume must be a finite value >= 0, got {volume}"
            )));
        }
        let voice = self.voice(voice)?;
        voice.sink.set_volume(volume);
        Ok(())
    }

    pub fn stop(&mut self, voice: VoiceId) -> EngineResult<()> {
        let slot = self
            .voices
            .get_mut(voice.0)
            .ok_or(EngineError::InvalidValue(
                "stop was given a voice this mixer never started".into(),
            ))?;
        if let Some(voice) = slot.take() {
            voice.sink.stop();
        }
        Ok(())
    }

    /// True while the sink still has samples to mix.
    pub fn is_playing(&self, voice: VoiceId) -> EngineResult<bool> {
        match self.voices.get(voice.0) {
            Some(Some(voice)) => Ok(!voice.sink.empty()),
            Some(None) => Ok(false),
            None => Err(EngineError::InvalidValue(
                "is_playing was given a voice this mixer never started".into(),
            )),
        }
    }

    fn alloc_voice(&mut self, voice: Voice) -> VoiceId {
        if let Some(index) = self.voices.iter().position(Option::is_none) {
            self.voices[index] = Some(voice);
            VoiceId(index)
        } else {
            self.voices.push(Some(voice));
            VoiceId(self.voices.len() - 1)
        }
    }

    fn voice(&self, id: VoiceId) -> EngineResult<&Voice> {
        self.voices
            .get(id.0)
            .and_then(|slot| slot.as_ref())
            .ok_or(EngineError::InvalidValue(
                "that voice has already been stopped".into(),
            ))
    }
}

fn decode_wav(bytes: &[u8]) -> Result<Decoder<Cursor<Vec<u8>>>, rodio::decoder::DecoderError> {
    Decoder::new(Cursor::new(bytes.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::decode_wav;
    use rodio::Source;

    fn pcm16_wav(samples: &[i16]) -> Vec<u8> {
        let data_bytes = (samples.len() * 2) as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&44_100u32.to_le_bytes());
        wav.extend_from_slice(&(44_100u32 * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }

    #[test]
    fn a_hand_written_wav_decodes() {
        let bytes = pcm16_wav(&[0, 1, -1, 0]);
        let source = decode_wav(&bytes).expect("valid WAV");
        assert!(source.channels() >= 1);
        assert!(source.sample_rate() > 0);
    }

    #[test]
    fn garbage_bytes_are_rejected() {
        assert!(
            decode_wav(b"not a wav").is_err(),
            "non-WAV bytes must not decode"
        );
    }
}
