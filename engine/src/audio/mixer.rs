//! The real-time mixer. It runs on the audio device's thread and knows
//! nothing about the world: the game side ([`super::scene`]) works out each
//! voice's gains, filter and pitch once per frame, and the mixer glides to
//! them sample by sample so nothing clicks.

use super::reverb::{Reverb, ReverbUnit};
use super::sound::SoundData;
use std::sync::Arc;

/// Mixer groups with their own volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bus {
    /// Sounds in the world: explosions, knocks, footsteps (default).
    Effects,
    /// Long loops that set the scene: wind, water, fire.
    Ambience,
    /// Music tracks.
    Music,
    /// Interface sounds: clicks, beeps. Never reverberated or muffled.
    Ui,
}

impl Bus {
    pub(crate) const COUNT: usize = 4;
    pub(crate) fn index(self) -> usize {
        match self {
            Bus::Effects => 0,
            Bus::Ambience => 1,
            Bus::Music => 2,
            Bus::Ui => 3,
        }
    }
}

/// Where a voice should be heading, worked out by the game side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VoiceTarget {
    pub gain_l: f32,
    pub gain_r: f32,
    /// Reverb send level.
    pub send: f32,
    /// Low-pass cutoff (distance, occlusion, behind the listener).
    pub lowpass_hz: f32,
    /// Playback speed: pitch times Doppler.
    pub speed: f32,
}

impl VoiceTarget {
    pub const SILENT: Self = Self {
        gain_l: 0.0,
        gain_r: 0.0,
        send: 0.0,
        lowpass_hz: 20_000.0,
        speed: 1.0,
    };
}

struct MixVoice {
    id: u64,
    data: Arc<SoundData>,
    pos: f64,
    looped: bool,
    /// Downmix to mono (placed sounds); otherwise stereo stays stereo.
    spatial: bool,
    bus: usize,
    cur: VoiceTarget,
    target: VoiceTarget,
    lp: [f32; 2],
    /// Fade multiplier 0..1 and its change per sample.
    fade: f32,
    fade_step: f32,
    finished: bool,
}

struct MusicLayerMix {
    data: Arc<SoundData>,
    gain: f32,
    target: f32,
}

struct TrackMix {
    id: u64,
    layers: Vec<MusicLayerMix>,
    /// Position in output frames.
    pos: f64,
    fade: f32,
    fade_step: f32,
}

pub(crate) struct Mixer {
    rate: u32,
    voices: Vec<MixVoice>,
    tracks: Vec<TrackMix>,
    reverb: ReverbUnit,
    pub master: f32,
    pub buses: [f32; Bus::COUNT],
    master_cur: f32,
    bus_cur: [f32; Bus::COUNT],
    /// Whole-world muffling (under water); UI is not muffled.
    muffle_target: f32,
    muffle_cur: f32,
    muffle_lp: [f32; 2],
    send_buf: Vec<f32>,
    world_buf: Vec<f32>,
    /// Ids of voices that ended since the game side last asked.
    ended: Vec<u64>,
    /// Frames rendered so far (for tests and the game's clock).
    pub rendered: u64,
}

fn lp_coeff(hz: f32, rate: f32) -> f32 {
    1.0 - (-2.0 * std::f32::consts::PI * hz.clamp(20.0, 20_000.0) / rate).exp()
}

impl Mixer {
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            voices: Vec::new(),
            tracks: Vec::new(),
            reverb: ReverbUnit::new(rate),
            master: 1.0,
            buses: [1.0; Bus::COUNT],
            master_cur: 1.0,
            bus_cur: [1.0; Bus::COUNT],
            muffle_target: 20_000.0,
            muffle_cur: 20_000.0,
            muffle_lp: [0.0; 2],
            send_buf: Vec::new(),
            world_buf: Vec::new(),
            ended: Vec::new(),
            rendered: 0,
        }
    }

    /// The output device runs at another rate: rebuild what depends on it.
    pub fn set_rate(&mut self, rate: u32) {
        if rate != self.rate && rate > 0 {
            self.rate = rate;
            let target = Reverb::OUTDOORS;
            self.reverb = ReverbUnit::new(rate);
            self.reverb.set_target(target);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &mut self,
        id: u64,
        data: Arc<SoundData>,
        looped: bool,
        spatial: bool,
        bus: Bus,
        target: VoiceTarget,
        fade_in: f32,
    ) {
        let fade_step = if fade_in > 0.0 {
            1.0 / (fade_in * self.rate as f32)
        } else {
            0.0
        };
        self.voices.push(MixVoice {
            id,
            data,
            pos: 0.0,
            looped,
            spatial,
            bus: bus.index(),
            // Start right at the target: a new sound should not swoop in.
            cur: target,
            target,
            lp: [0.0; 2],
            fade: if fade_in > 0.0 { 0.0 } else { 1.0 },
            fade_step,
            finished: false,
        });
    }

    pub fn set_target(&mut self, id: u64, target: VoiceTarget) {
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            v.target = target;
        }
    }

    /// Fade a voice out over `seconds` (0 = at once), then drop it.
    pub fn stop(&mut self, id: u64, seconds: f32) {
        let rate = self.rate as f32;
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == id) {
            if seconds <= 0.0 {
                v.finished = true;
            } else {
                v.fade_step = -1.0 / (seconds * rate);
            }
        }
    }

    #[cfg(test)]
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    pub fn take_ended(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.ended)
    }

    pub fn set_reverb(&mut self, reverb: Reverb) {
        self.reverb.set_target(reverb);
    }

    pub fn set_muffle(&mut self, hz: f32) {
        self.muffle_target = hz.clamp(100.0, 20_000.0);
    }

    pub fn start_track(&mut self, id: u64, layers: Vec<(Arc<SoundData>, f32)>, fade_in: f32) {
        let rate = self.rate as f32;
        self.tracks.push(TrackMix {
            id,
            layers: layers
                .into_iter()
                .map(|(data, gain)| MusicLayerMix {
                    data,
                    gain,
                    target: gain,
                })
                .collect(),
            pos: 0.0,
            fade: if fade_in > 0.0 { 0.0 } else { 1.0 },
            fade_step: if fade_in > 0.0 {
                1.0 / (fade_in * rate)
            } else {
                0.0
            },
        });
    }

    pub fn set_layer_gains(&mut self, id: u64, gains: &[f32]) {
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            for (layer, g) in t.layers.iter_mut().zip(gains) {
                layer.target = *g;
            }
        }
    }

    pub fn stop_track(&mut self, id: u64, seconds: f32) {
        let rate = self.rate as f32;
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.fade_step = -1.0 / (seconds.max(0.01) * rate);
        }
    }

    /// Mix `out.len() / 2` stereo frames into `out` (overwriting it).
    pub fn render(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        out.fill(0.0);
        self.send_buf.clear();
        self.send_buf.resize(frames, 0.0);
        self.world_buf.clear();
        self.world_buf.resize(frames * 2, 0.0);
        let rate = self.rate as f32;
        let inv = 1.0 / frames.max(1) as f32;

        // Bus and master volumes glide once per block.
        let bus_from = self.bus_cur;
        let master_from = self.master_cur;
        for (cur, target) in self.bus_cur.iter_mut().zip(self.buses) {
            *cur = target.clamp(0.0, 4.0);
        }
        self.master_cur = self.master.clamp(0.0, 4.0);
        let master_step = (self.master_cur - master_from) * inv;

        for v in self.voices.iter_mut() {
            if v.finished {
                continue;
            }
            let ratio = v.data.rate as f64 / rate as f64;
            let n_frames = v.data.frames();
            if n_frames == 0 {
                v.finished = true;
                continue;
            }
            let from = v.cur;
            let to = v.target;
            let lp_a = lp_coeff(to.lowpass_hz, rate);
            let lp_from = lp_coeff(from.lowpass_hz, rate);
            let bus_g = bus_from[v.bus];
            let bus_step = (self.bus_cur[v.bus] - bus_g) * inv;
            // UI and music skip the world's reverb and muffling.
            let dest_world = v.bus != Bus::Ui.index() && v.bus != Bus::Music.index();
            for i in 0..frames {
                let t = i as f32 * inv;
                let gl = from.gain_l + (to.gain_l - from.gain_l) * t;
                let gr = from.gain_r + (to.gain_r - from.gain_r) * t;
                let send = from.send + (to.send - from.send) * t;
                let speed = from.speed + (to.speed - from.speed) * t;
                let a = lp_from + (lp_a - lp_from) * t;
                // Linear interpolation between source frames.
                let p = v.pos;
                let i0 = p as usize;
                let frac = (p - i0 as f64) as f32;
                let i1 = if i0 + 1 < n_frames {
                    i0 + 1
                } else if v.looped {
                    0
                } else {
                    i0
                };
                let (l0, r0) = v.data.frame(i0);
                let (l1, r1) = v.data.frame(i1);
                let (mut l, mut r) = (l0 + (l1 - l0) * frac, r0 + (r1 - r0) * frac);
                if v.spatial {
                    let m = (l + r) * 0.5;
                    l = m;
                    r = m;
                }
                v.lp[0] += a * (l - v.lp[0]);
                v.lp[1] += a * (r - v.lp[1]);
                let g = v.fade * (bus_g + bus_step * i as f32);
                let (ol, or) = (v.lp[0] * gl * g, v.lp[1] * gr * g);
                if dest_world {
                    self.world_buf[2 * i] += ol;
                    self.world_buf[2 * i + 1] += or;
                    self.send_buf[i] += (v.lp[0] + v.lp[1]) * 0.5 * send * g;
                } else {
                    out[2 * i] += ol;
                    out[2 * i + 1] += or;
                }

                if v.fade_step != 0.0 {
                    v.fade += v.fade_step;
                    if v.fade >= 1.0 {
                        v.fade = 1.0;
                        v.fade_step = 0.0;
                    } else if v.fade <= 0.0 {
                        v.finished = true;
                        break;
                    }
                }
                v.pos += speed.max(0.01) as f64 * ratio;
                if v.pos >= n_frames as f64 {
                    if v.looped {
                        v.pos -= n_frames as f64;
                    } else {
                        v.finished = true;
                        break;
                    }
                }
            }
            v.cur = to;
        }

        // Music.
        let music_g = bus_from[Bus::Music.index()];
        let music_step = (self.bus_cur[Bus::Music.index()] - music_g) * inv;
        for track in self.tracks.iter_mut() {
            for layer in track.layers.iter_mut() {
                let from = layer.gain;
                let to = layer.target;
                let ratio = layer.data.rate as f64 / rate as f64;
                let n = layer.data.frames();
                if n == 0 {
                    continue;
                }
                let mut fade = track.fade;
                for i in 0..frames {
                    let g = (from + (to - from) * i as f32 * inv)
                        * fade
                        * (music_g + music_step * i as f32);
                    let p = (track.pos + i as f64) * ratio;
                    let idx = (p as usize) % n;
                    let (l, r) = layer.data.frame(idx);
                    out[2 * i] += l * g;
                    out[2 * i + 1] += r * g;
                    fade = (fade + track.fade_step).clamp(0.0, 1.0);
                }
                layer.gain = to;
            }
            track.fade = (track.fade + track.fade_step * frames as f32).clamp(0.0, 1.0);
            if track.fade_step > 0.0 && track.fade >= 1.0 {
                track.fade_step = 0.0;
            }
            track.pos += frames as f64;
        }
        self.tracks
            .retain(|t| !(t.fade_step < 0.0 && t.fade <= 0.0));

        // World sounds: reverb, then muffling (under water).
        self.reverb.process(&self.send_buf, &mut self.world_buf);
        let m_from = lp_coeff(self.muffle_cur, rate);
        self.muffle_cur += (self.muffle_target - self.muffle_cur) * 0.2;
        let m_to = lp_coeff(self.muffle_cur, rate);
        let muffled = self.muffle_cur < 19_000.0 || self.muffle_target < 19_000.0;
        for i in 0..frames {
            let (mut l, mut r) = (self.world_buf[2 * i], self.world_buf[2 * i + 1]);
            if muffled {
                let a = m_from + (m_to - m_from) * i as f32 * inv;
                self.muffle_lp[0] += a * (l - self.muffle_lp[0]);
                self.muffle_lp[1] += a * (r - self.muffle_lp[1]);
                l = self.muffle_lp[0];
                r = self.muffle_lp[1];
            } else {
                self.muffle_lp = [l, r];
            }
            out[2 * i] += l;
            out[2 * i + 1] += r;
        }

        // Master volume and a soft limiter, so many explosions at once
        // squash instead of clipping harshly.
        for i in 0..frames {
            let g = master_from + master_step * i as f32;
            for c in 0..2 {
                let x = out[2 * i + c] * g;
                out[2 * i + c] = soft_clip(x);
            }
        }

        for v in &self.voices {
            if v.finished {
                self.ended.push(v.id);
            }
        }
        self.voices.retain(|v| !v.finished);
        self.rendered += frames as u64;
    }
}

/// Transparent below 0.8, then rounds smoothly toward 1.
#[inline]
fn soft_clip(x: f32) -> f32 {
    let a = x.abs();
    if a <= 0.8 {
        x
    } else {
        let over = a - 0.8;
        (0.8 + 0.2 * (over / 0.2).tanh()).copysign(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, seconds: f32) -> Arc<SoundData> {
        let n = (rate as f32 * seconds) as usize;
        Arc::new(SoundData {
            samples: (0..n)
                .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5)
                .collect(),
            channels: 1,
            rate,
        })
    }

    fn energy(buf: &[f32], ch: usize) -> f32 {
        buf.iter().skip(ch).step_by(2).map(|v| v * v).sum()
    }

    fn target(gl: f32, gr: f32) -> VoiceTarget {
        VoiceTarget {
            gain_l: gl,
            gain_r: gr,
            send: 0.0,
            lowpass_hz: 20_000.0,
            speed: 1.0,
        }
    }

    #[test]
    fn a_one_shot_plays_then_reports_it_ended() {
        let mut m = Mixer::new(48_000);
        m.start(
            7,
            tone(48_000, 0.05),
            false,
            true,
            Bus::Effects,
            target(1.0, 1.0),
            0.0,
        );
        let mut out = vec![0.0; 2 * 4800];
        m.render(&mut out);
        assert!(energy(&out, 0) > 1.0);
        assert_eq!(m.voice_count(), 0);
        assert_eq!(m.take_ended(), vec![7]);
    }

    #[test]
    fn pan_puts_the_sound_on_one_side() {
        let mut m = Mixer::new(48_000);
        m.start(
            1,
            tone(48_000, 1.0),
            true,
            true,
            Bus::Effects,
            target(1.0, 0.1),
            0.0,
        );
        let mut out = vec![0.0; 2 * 2048];
        m.render(&mut out);
        assert!(energy(&out, 0) > 50.0 * energy(&out, 1));
    }

    #[test]
    fn a_low_pass_darkens_and_quietens_a_high_tone() {
        let run = |hz: f32| {
            let mut m = Mixer::new(48_000);
            let mut t = target(1.0, 1.0);
            t.lowpass_hz = hz;
            m.start(1, tone(48_000, 1.0), true, true, Bus::Effects, t, 0.0);
            let mut out = vec![0.0; 2 * 4096];
            m.render(&mut out);
            energy(&out, 0)
        };
        assert!(run(150.0) < run(20_000.0) * 0.2);
    }

    #[test]
    fn a_stopped_loop_fades_out_and_ends() {
        let mut m = Mixer::new(48_000);
        m.start(
            3,
            tone(48_000, 0.5),
            true,
            false,
            Bus::Ambience,
            target(1.0, 1.0),
            0.0,
        );
        m.stop(3, 0.01);
        let mut out = vec![0.0; 2 * 2048];
        m.render(&mut out);
        assert_eq!(m.voice_count(), 0);
        assert_eq!(m.take_ended(), vec![3]);
    }

    #[test]
    fn output_never_clips() {
        let mut m = Mixer::new(48_000);
        for id in 0..40 {
            m.start(
                id,
                tone(48_000, 1.0),
                true,
                true,
                Bus::Effects,
                target(3.0, 3.0),
                0.0,
            );
        }
        let mut out = vec![0.0; 2 * 2048];
        m.render(&mut out);
        assert!(out.iter().all(|v| v.abs() <= 1.0));
    }

    #[test]
    fn music_layers_follow_their_gains() {
        let mut m = Mixer::new(48_000);
        m.start_track(
            1,
            vec![(tone(48_000, 0.5), 1.0), (tone(48_000, 0.5), 0.0)],
            0.0,
        );
        let mut out = vec![0.0; 2 * 2048];
        m.render(&mut out);
        let one = energy(&out, 0);
        m.set_layer_gains(1, &[1.0, 1.0]);
        m.render(&mut out); // glide
        m.render(&mut out);
        assert!(
            energy(&out, 0) > one * 3.0,
            "both layers in phase: 4x energy"
        );
    }
}
