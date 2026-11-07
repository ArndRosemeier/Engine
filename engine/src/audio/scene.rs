//! The game side of audio: which sounds play where, the listener, the
//! acoustics of the space around it, and music. Once per frame it turns all
//! of that into gains, filters and pitch for the real-time [`Mixer`].

use super::mixer::{Bus, Mixer, VoiceTarget};
use super::reverb::Reverb;
use super::sound::Sound;
use crate::world::EntityId;
use glam::Vec3;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Speed of sound, m/s (for the Doppler shift).
const SPEED_OF_SOUND: f32 = 343.0;
/// Most voices mixed at once; the quietest one-shot gives way beyond it.
const MAX_VOICES: usize = 96;

/// A playing sound, from [`crate::World::play_sound`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SoundId(pub(crate) u64);

/// How to play a sound: where, how loud, looped or not.
///
/// ```no_run
/// # use engine::prelude::*;
/// # fn f(world: &mut World, boom: &Sound, door: EntityId) {
/// world.play_sound(boom, PlaySound::at([10.0, 0.0, 5.0]).volume(2.0));
/// world.play_sound(&Sound::fire_loop(1), PlaySound::following(door).looped());
/// world.play_sound(&Sound::tone(880.0, 0.1), PlaySound::ui());
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PlaySound {
    pub(crate) at: Option<Vec3>,
    pub(crate) follow: Option<EntityId>,
    pub(crate) volume: f32,
    pub(crate) pitch: f32,
    pub(crate) looped: bool,
    pub(crate) bus: Bus,
    /// Distance (m) at which the sound is at full volume.
    pub(crate) near: f32,
    /// Distance (m) beyond which it cannot be heard.
    pub(crate) range: f32,
    pub(crate) reverb: f32,
    pub(crate) doppler: f32,
    pub(crate) fade_in: f32,
    pub(crate) pan: f32,
}

impl PlaySound {
    fn base(at: Option<Vec3>, bus: Bus) -> Self {
        Self {
            at,
            follow: None,
            volume: 1.0,
            pitch: 1.0,
            looped: false,
            bus,
            near: 2.0,
            range: 120.0,
            reverb: 1.0,
            doppler: 1.0,
            fade_in: 0.0,
            pan: 0.0,
        }
    }

    /// A sound at a point in the world: it gets quieter with distance,
    /// comes from its direction, echoes in rooms and caves, and is muffled
    /// behind walls.
    pub fn at(position: impl Into<Vec3>) -> Self {
        Self::base(Some(position.into()), Bus::Effects)
    }

    /// A sound that moves with an entity (a car engine, a torch).
    pub fn following(entity: EntityId) -> Self {
        let mut p = Self::base(Some(Vec3::ZERO), Bus::Effects);
        p.follow = Some(entity);
        p
    }

    /// Not placed in the world: plays straight to the speakers (menus,
    /// beeps). Never reverberated or muffled.
    pub fn ui() -> Self {
        let mut p = Self::base(None, Bus::Ui);
        p.reverb = 0.0;
        p
    }

    /// Not placed, but part of the world's sound (a wind bed, a distant
    /// city): muffled under water and slightly reverberated.
    pub fn ambient() -> Self {
        let mut p = Self::base(None, Bus::Ambience);
        p.reverb = 0.3;
        p
    }

    /// Loudness, 1 = as recorded.
    pub fn volume(mut self, volume: f32) -> Self {
        self.volume = volume;
        self
    }

    /// Playback speed: 2 = an octave higher and twice as fast.
    pub fn pitch(mut self, pitch: f32) -> Self {
        self.pitch = pitch;
        self
    }

    /// Repeat until stopped.
    pub fn looped(mut self) -> Self {
        self.looped = true;
        self
    }

    /// Full volume within `near` metres, silent beyond `range` metres
    /// (defaults 2 and 120). Loud things (explosions) want a long range.
    pub fn range(mut self, near: f32, range: f32) -> Self {
        self.near = near;
        self.range = range;
        self
    }

    /// How much of this sound goes into the room or cave reverb, 0..2
    /// (default 1).
    pub fn reverb(mut self, amount: f32) -> Self {
        self.reverb = amount;
        self
    }

    /// Strength of the Doppler pitch shift for moving sounds (default 1,
    /// 0 turns it off).
    pub fn doppler(mut self, amount: f32) -> Self {
        self.doppler = amount;
        self
    }

    /// Fade in over `seconds` instead of starting at full volume.
    pub fn fade_in(mut self, seconds: f32) -> Self {
        self.fade_in = seconds;
        self
    }

    /// Put it on another mixer group (see [`Bus`]).
    pub fn bus(mut self, bus: Bus) -> Self {
        self.bus = bus;
        self
    }

    /// Left (-1) to right (1), for sounds not placed in the world.
    pub fn pan(mut self, pan: f32) -> Self {
        self.pan = pan;
        self
    }

    fn sanitized(mut self) -> Self {
        let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        self.volume = f(self.volume, 0.0, 16.0, 1.0);
        self.pitch = f(self.pitch, 0.05, 8.0, 1.0);
        self.near = f(self.near, 0.01, 10_000.0, 2.0);
        self.range = f(self.range, self.near + 0.01, 100_000.0, 120.0);
        self.reverb = f(self.reverb, 0.0, 2.0, 1.0);
        self.doppler = f(self.doppler, 0.0, 4.0, 1.0);
        self.fade_in = f(self.fade_in, 0.0, 60.0, 0.0);
        self.pan = f(self.pan, -1.0, 1.0, 0.0);
        if self.at.is_some_and(|p| !p.is_finite()) {
            self.at = Some(Vec3::ZERO);
        }
        self
    }
}

/// What the space around the listener does to sound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Acoustics {
    pub reverb: Reverb,
    /// Low-pass on everything in the world (20000 = clear; about 500
    /// sounds like being under water).
    pub muffle_hz: f32,
}

impl Acoustics {
    pub const OUTDOORS: Self = Self {
        reverb: Reverb::OUTDOORS,
        muffle_hz: 20_000.0,
    };
    pub const UNDERWATER: Self = Self {
        reverb: Reverb::UNDERWATER,
        muffle_hz: 450.0,
    };

    pub fn with_reverb(reverb: Reverb) -> Self {
        Self {
            reverb,
            muffle_hz: 20_000.0,
        }
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        // Muffling glides on a log scale so it sounds even.
        let (a, b) = (
            self.muffle_hz.max(20.0).ln(),
            other.muffle_hz.max(20.0).ln(),
        );
        Self {
            reverb: self.reverb.lerp(other.reverb, t),
            muffle_hz: (a + (b - a) * t.clamp(0.0, 1.0)).exp(),
        }
    }
}

impl Default for Acoustics {
    fn default() -> Self {
        Self::OUTDOORS
    }
}

/// One stem of a [`MusicTrack`], audible over a range of intensity.
#[derive(Clone, Debug, PartialEq)]
pub struct MusicLayer {
    pub sound: Sound,
    /// Silent at or below this intensity...
    pub from: f32,
    /// ...and at full volume from this intensity up.
    pub to: f32,
    pub volume: f32,
}

/// Music made of layers that loop in step and fade in and out with the
/// game's intensity ([`crate::World::set_music_intensity`]): calm
/// exploring plays the pad alone, a fight brings in drums.
///
/// Give every layer the same length so they stay in time.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MusicTrack {
    pub layers: Vec<MusicLayer>,
}

impl MusicTrack {
    pub fn new() -> Self {
        Self::default()
    }

    /// A layer that always plays.
    pub fn layer(mut self, sound: Sound) -> Self {
        self.layers.push(MusicLayer {
            sound,
            from: -1.0,
            to: 0.0,
            volume: 1.0,
        });
        self
    }

    /// A layer that fades in as intensity rises from `from` to `to` (0..1).
    pub fn layer_between(mut self, sound: Sound, from: f32, to: f32) -> Self {
        self.layers.push(MusicLayer {
            sound,
            from,
            to,
            volume: 1.0,
        });
        self
    }

    /// Music made in code (8 bars at `bpm`, key and chords picked by
    /// `seed`): a pad that always plays, bass from intensity 0.2, an
    /// arpeggio from 0.45, drums from 0.7.
    pub fn procedural(seed: u64, bpm: f32) -> Self {
        let s = super::synth::music::stems(seed, bpm);
        let rate = super::synth::music::STEM_RATE;
        let stem = |v: Vec<f32>| Sound::from_mono(v, rate).expect("finite stems");
        Self::new()
            .layer(stem(s.pad))
            .layer_between(stem(s.bass), 0.15, 0.35)
            .layer_between(stem(s.arp), 0.4, 0.6)
            .layer_between(stem(s.drums), 0.65, 0.85)
    }

    fn gains(&self, intensity: f32) -> Vec<f32> {
        self.layers
            .iter()
            .map(|l| {
                let t = if l.to > l.from {
                    ((intensity - l.from) / (l.to - l.from)).clamp(0.0, 1.0)
                } else if intensity >= l.to {
                    1.0
                } else {
                    0.0
                };
                t * t * (3.0 - 2.0 * t) * l.volume.max(0.0)
            })
            .collect()
    }
}

struct VoiceState {
    play: PlaySound,
    /// Current position (placed sounds).
    pos: Vec3,
    last_pos: Option<Vec3>,
    velocity: Vec3,
    occlusion: f32,
    occlusion_target: f32,
    /// When a one-shot will have finished (game clock), for bookkeeping
    /// when no device pulls the mixer.
    ends_at: Option<f64>,
    /// Last loudness sent, to pick a voice to steal.
    loudness: f32,
}

/// Where the ears are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ListenerPose {
    pub pos: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
}

/// All of a world's sound. Reach it with [`crate::World::audio`] and
/// [`crate::World::audio_mut`]; the common calls are also on `World`.
pub struct AudioScene {
    mixer: Arc<Mutex<Mixer>>,
    connected: Arc<AtomicBool>,
    voices: HashMap<u64, VoiceState>,
    /// Starts and stops waiting for the next mixer update.
    pending_start: Vec<(u64, Sound)>,
    pending_stop: Vec<(u64, f32)>,
    next_id: u64,
    listener: ListenerPose,
    listener_last: Option<Vec3>,
    listener_velocity: Vec3,
    acoustics: Acoustics,
    acoustics_target: Acoustics,
    acoustics_override: Option<Acoustics>,
    music: Option<(u64, MusicTrack)>,
    intensity: f32,
    master: f32,
    buses: [f32; Bus::COUNT],
    muted: bool,
    clock: f64,
    pub(crate) auto_sounds: bool,
}

impl std::fmt::Debug for AudioScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioScene")
            .field("voices", &self.voices.len())
            .field("music", &self.music.is_some())
            .field("acoustics", &self.acoustics)
            .field("output", &self.has_output())
            .finish()
    }
}

impl Default for AudioScene {
    fn default() -> Self {
        Self::new()
    }
}

fn mixer_lock(m: &Mutex<Mixer>) -> std::sync::MutexGuard<'_, Mixer> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl AudioScene {
    pub fn new() -> Self {
        Self {
            mixer: Arc::new(Mutex::new(Mixer::new(48_000))),
            connected: Arc::new(AtomicBool::new(false)),
            voices: HashMap::new(),
            pending_start: Vec::new(),
            pending_stop: Vec::new(),
            next_id: 1,
            listener: ListenerPose {
                pos: Vec3::ZERO,
                forward: Vec3::NEG_Z,
                up: Vec3::Y,
            },
            listener_last: None,
            listener_velocity: Vec3::ZERO,
            acoustics: Acoustics::OUTDOORS,
            acoustics_target: Acoustics::OUTDOORS,
            acoustics_override: None,
            music: None,
            intensity: 0.0,
            master: 1.0,
            buses: [1.0; Bus::COUNT],
            muted: false,
            clock: 0.0,
            auto_sounds: true,
        }
    }

    /// Start a sound. See [`PlaySound`] for where and how.
    pub fn play(&mut self, sound: &Sound, play: PlaySound) -> SoundId {
        let play = play.sanitized();
        let id = self.next_id;
        self.next_id += 1;
        if self.voices.len() >= MAX_VOICES {
            self.steal_voice();
        }
        let ends_at = (!play.looped).then(|| {
            self.clock + (sound.duration() / play.pitch) as f64 + play.fade_in as f64 + 0.1
        });
        let pos = play.at.unwrap_or(Vec3::ZERO);
        self.voices.insert(
            id,
            VoiceState {
                pos,
                last_pos: None,
                velocity: Vec3::ZERO,
                occlusion: 0.0,
                occlusion_target: 0.0,
                ends_at,
                loudness: play.volume,
                play,
            },
        );
        self.pending_start.push((id, sound.clone()));
        SoundId(id)
    }

    fn steal_voice(&mut self) {
        let quietest = self
            .voices
            .iter()
            .filter(|(_, v)| !v.play.looped)
            .min_by(|a, b| a.1.loudness.total_cmp(&b.1.loudness))
            .map(|(id, _)| *id);
        if let Some(id) = quietest {
            self.stop(SoundId(id), 0.03);
        }
    }

    /// Stop a sound, fading out over `fade` seconds (0 = at once).
    pub fn stop(&mut self, id: SoundId, fade: f32) {
        if self.voices.remove(&id.0).is_some() {
            self.pending_start.retain(|(v, _)| *v != id.0);
            self.pending_stop.push((id.0, fade.max(0.0)));
        }
    }

    /// Stop every placed and ambient sound (not the music).
    pub fn stop_all(&mut self, fade: f32) {
        let ids: Vec<u64> = self.voices.keys().copied().collect();
        for id in ids {
            self.stop(SoundId(id), fade);
        }
    }

    /// Whether the sound is still playing (one-shots end by themselves).
    pub fn is_playing(&self, id: SoundId) -> bool {
        self.voices.contains_key(&id.0)
    }

    /// Move a placed sound.
    pub fn set_position(&mut self, id: SoundId, position: impl Into<Vec3>) {
        let p = position.into();
        if let Some(v) = self.voices.get_mut(&id.0) {
            if p.is_finite() {
                v.pos = p;
                v.play.at = Some(p);
            }
        }
    }

    pub fn set_volume(&mut self, id: SoundId, volume: f32) {
        if let Some(v) = self.voices.get_mut(&id.0) {
            v.play.volume = if volume.is_finite() {
                volume.clamp(0.0, 16.0)
            } else {
                0.0
            };
        }
    }

    pub fn set_pitch(&mut self, id: SoundId, pitch: f32) {
        if let Some(v) = self.voices.get_mut(&id.0) {
            if pitch.is_finite() {
                v.play.pitch = pitch.clamp(0.05, 8.0);
            }
        }
    }

    /// Sounds playing now.
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Overall volume (default 1).
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master = if volume.is_finite() {
            volume.clamp(0.0, 4.0)
        } else {
            1.0
        };
    }

    pub fn master_volume(&self) -> f32 {
        self.master
    }

    /// Volume of one mixer group (default 1 each).
    pub fn set_bus_volume(&mut self, bus: Bus, volume: f32) {
        self.buses[bus.index()] = if volume.is_finite() {
            volume.clamp(0.0, 4.0)
        } else {
            1.0
        };
    }

    pub fn bus_volume(&self, bus: Bus) -> f32 {
        self.buses[bus.index()]
    }

    /// Silence everything without losing what plays.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }

    pub fn muted(&self) -> bool {
        self.muted
    }

    /// Play `track`, cross-fading from the current music over `fade`
    /// seconds.
    pub fn play_music(&mut self, track: MusicTrack, fade: f32) {
        let fade = fade.max(0.0);
        let id = self.next_id;
        self.next_id += 1;
        let gains = track.gains(self.intensity);
        let layers = track
            .layers
            .iter()
            .zip(&gains)
            .map(|(l, g)| (l.sound.0.clone(), *g))
            .collect();
        let mut m = mixer_lock(&self.mixer);
        if let Some((old, _)) = self.music.take() {
            m.stop_track(old, fade.max(0.05));
        }
        m.start_track(id, layers, fade);
        drop(m);
        self.music = Some((id, track));
    }

    pub fn stop_music(&mut self, fade: f32) {
        if let Some((id, _)) = self.music.take() {
            mixer_lock(&self.mixer).stop_track(id, fade.max(0.05));
        }
    }

    /// 0 (calm) to 1 (fierce): which music layers play.
    pub fn set_music_intensity(&mut self, intensity: f32) {
        if intensity.is_finite() {
            self.intensity = intensity.clamp(0.0, 1.0);
        }
    }

    pub fn music_intensity(&self) -> f32 {
        self.intensity
    }

    pub fn music_playing(&self) -> bool {
        self.music.is_some()
    }

    /// The acoustics heard right now (gliding toward the space the
    /// listener is in).
    pub fn acoustics(&self) -> Acoustics {
        self.acoustics
    }

    /// Force these acoustics instead of working them out from rooms, caves
    /// and water (`None` goes back to automatic).
    pub fn set_acoustics_override(&mut self, acoustics: Option<Acoustics>) {
        self.acoustics_override = acoustics;
    }

    /// Whether sounds reach a speaker (an output device opened).
    pub fn has_output(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Mix the next `frames` stereo frames yourself, at the mixer's rate
    /// (48 kHz without a device): for recording, tests and tools. Not for
    /// use while an output device is playing the same scene.
    pub fn render_offline(&mut self, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; frames * 2];
        mixer_lock(&self.mixer).render(&mut out);
        out
    }

    pub(crate) fn mixer_handle(&self) -> (Arc<Mutex<Mixer>>, Arc<AtomicBool>) {
        (self.mixer.clone(), self.connected.clone())
    }

    /// Ids and positions of placed voices, with the entity they follow.
    pub(crate) fn placed_voices(&self) -> Vec<(u64, Vec3, Option<EntityId>)> {
        self.voices
            .iter()
            .filter(|(_, v)| v.play.at.is_some() && v.play.bus != Bus::Ui)
            .map(|(id, v)| (*id, v.pos, v.play.follow))
            .collect()
    }

    pub(crate) fn set_voice_pos(&mut self, id: u64, pos: Vec3) {
        if let Some(v) = self.voices.get_mut(&id) {
            if pos.is_finite() {
                v.pos = pos;
            }
        }
    }

    /// 0 (clear line) to 1 (behind a wall), from the world's ray casts.
    pub(crate) fn set_voice_occlusion(&mut self, id: u64, occlusion: f32) {
        if let Some(v) = self.voices.get_mut(&id) {
            v.occlusion_target = occlusion.clamp(0.0, 1.0);
        }
    }

    /// Advance by `dt` with the ears at `listener` in a space that sounds
    /// like `acoustics`, and hand the mixer its new targets.
    pub(crate) fn update(&mut self, dt: f32, listener: ListenerPose, acoustics: Acoustics) {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        self.clock += dt as f64;
        let mut l = listener;
        l.forward = l.forward.try_normalize().unwrap_or(Vec3::NEG_Z);
        l.up = l.up.try_normalize().unwrap_or(Vec3::Y);
        let right = l.forward.cross(l.up).try_normalize().unwrap_or(Vec3::X);
        if dt > 0.0 {
            if let Some(last) = self.listener_last {
                let v = (l.pos - last) / dt;
                // Teleports are not motion.
                let v = if v.length() > 200.0 { Vec3::ZERO } else { v };
                self.listener_velocity = self.listener_velocity.lerp(v, 0.3);
            }
        }
        self.listener_last = Some(l.pos);
        self.listener = l;

        self.acoustics_target = self.acoustics_override.unwrap_or(acoustics);
        // About half a second to settle into a new space.
        let k = 1.0 - (-dt * 5.0).exp();
        self.acoustics = self.acoustics.lerp(self.acoustics_target, k);

        let occ_k = 1.0 - (-dt * 8.0).exp();
        let mut targets = Vec::with_capacity(self.voices.len());
        for (id, v) in self.voices.iter_mut() {
            if dt > 0.0 {
                if let Some(last) = v.last_pos {
                    let vel = (v.pos - last) / dt;
                    let vel = if vel.length() > 300.0 {
                        Vec3::ZERO
                    } else {
                        vel
                    };
                    v.velocity = v.velocity.lerp(vel, 0.3);
                }
            }
            v.last_pos = Some(v.pos);
            v.occlusion += (v.occlusion_target - v.occlusion) * occ_k;
            let t = voice_target(
                &v.play,
                v.pos,
                v.velocity,
                v.occlusion,
                &l,
                right,
                self.listener_velocity,
            );
            v.loudness = t.gain_l.max(t.gain_r);
            targets.push((*id, t));
        }

        // Without a device nothing pulls the mixer: retire one-shots by
        // the clock instead.
        let connected = self.connected.load(Ordering::Relaxed);
        if !connected {
            let clock = self.clock;
            let done: Vec<u64> = self
                .voices
                .iter()
                .filter(|(_, v)| v.ends_at.is_some_and(|e| clock > e))
                .map(|(id, _)| *id)
                .collect();
            for id in done {
                self.voices.remove(&id);
                self.pending_stop.push((id, 0.0));
            }
        }

        let master = if self.muted { 0.0 } else { self.master };
        let music_gains = self
            .music
            .as_ref()
            .map(|(id, t)| (*id, t.gains(self.intensity)));
        let mut m = mixer_lock(&self.mixer);
        for (id, sound) in self.pending_start.drain(..) {
            if let Some(v) = self.voices.get(&id) {
                let target = targets
                    .iter()
                    .find(|(t, _)| *t == id)
                    .map(|(_, t)| *t)
                    .unwrap_or(VoiceTarget::SILENT);
                m.start(
                    id,
                    sound.0.clone(),
                    v.play.looped,
                    v.play.at.is_some(),
                    v.play.bus,
                    target,
                    v.play.fade_in,
                );
            }
        }
        for (id, fade) in self.pending_stop.drain(..) {
            m.stop(id, fade);
        }
        for (id, t) in &targets {
            m.set_target(*id, *t);
        }
        if let Some((id, gains)) = music_gains {
            m.set_layer_gains(id, &gains);
        }
        m.master = master;
        m.buses = self.buses;
        m.set_reverb(self.acoustics.reverb);
        m.set_muffle(self.acoustics.muffle_hz);
        let ended = m.take_ended();
        drop(m);
        for id in ended {
            self.voices.remove(&id);
        }
    }
}

/// Gains, filter and pitch for one voice heard from `l`.
fn voice_target(
    p: &PlaySound,
    pos: Vec3,
    velocity: Vec3,
    occlusion: f32,
    l: &ListenerPose,
    right: Vec3,
    listener_velocity: Vec3,
) -> VoiceTarget {
    let Some(_) = p.at else {
        // Not placed: equal-power pan only.
        let a = (p.pan + 1.0) * std::f32::consts::FRAC_PI_4;
        let s = std::f32::consts::SQRT_2;
        return VoiceTarget {
            gain_l: p.volume * a.cos() * s,
            gain_r: p.volume * a.sin() * s,
            send: p.reverb * 0.3,
            lowpass_hz: 20_000.0,
            speed: p.pitch,
        };
    };
    let d = pos - l.pos;
    let dist = d.length();
    let dir = if dist > 1e-3 { d / dist } else { l.forward };
    // Inverse-distance roll-off past `near`, faded to silence at `range`.
    let mut gain = p.near / (p.near + (dist - p.near).max(0.0));
    let edge = ((p.range - dist) / (p.range * 0.25)).clamp(0.0, 1.0);
    gain *= edge * edge * (3.0 - 2.0 * edge);
    // Direction: equal-power pan, a little blended to the middle so close
    // sounds are not stuck in one ear.
    let closeness = (dist / p.near.max(0.5)).min(1.0);
    let x = dir.dot(right) * 0.85 * closeness;
    let a = (x + 1.0) * std::f32::consts::FRAC_PI_4;
    let s = std::f32::consts::SQRT_2;
    // Air absorbs highs over distance; the head shadows sounds from behind.
    let mut lowpass = 20_000.0 / (1.0 + dist / 150.0);
    let front = dir.dot(l.forward);
    if front < 0.0 {
        lowpass *= 1.0 - 0.4 * (-front) * closeness;
    }
    // Walls in between: duller and quieter.
    lowpass = lowpass * (1.0 - occlusion) + 650.0 * occlusion;
    gain *= 1.0 - 0.6 * occlusion;
    // Doppler.
    let mut speed = p.pitch;
    if p.doppler > 0.0 {
        let toward = listener_velocity.dot(dir); // listener moving toward the source
        let away = velocity.dot(dir); // source moving away from the listener
        let shift = ((SPEED_OF_SOUND + toward) / (SPEED_OF_SOUND + away).max(1.0)).clamp(0.5, 2.0);
        speed *= 1.0 + (shift - 1.0) * p.doppler;
    }
    // Far sounds are relatively more reverberant.
    let send = p.reverb * (0.4 + 0.6 * (dist / 30.0).min(1.0)) * gain * p.volume;
    VoiceTarget {
        gain_l: p.volume * gain * a.cos() * s,
        gain_r: p.volume * gain * a.sin() * s,
        send,
        lowpass_hz: lowpass.max(200.0),
        speed,
    }
}

/// The audio device, kept on its own thread (some platforms do not let
/// the stream move between threads). Dropping it stops playback.
pub(crate) struct AudioOutput {
    stop: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Pulls blocks from the shared mixer for the device.
struct MixSource {
    mixer: Arc<Mutex<Mixer>>,
    buf: Vec<f32>,
    at: usize,
    rate: u32,
}

impl Iterator for MixSource {
    type Item = f32;
    #[inline]
    fn next(&mut self) -> Option<f32> {
        if self.at >= self.buf.len() {
            // 256 frames (about 5 ms): small enough for quick reactions,
            // big enough that the lock is rare.
            self.buf.resize(512, 0.0);
            mixer_lock(&self.mixer).render(&mut self.buf);
            self.at = 0;
        }
        let s = self.buf[self.at];
        self.at += 1;
        Some(s)
    }
}

impl rodio::Source for MixSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> rodio::ChannelCount {
        2
    }
    fn sample_rate(&self) -> rodio::SampleRate {
        self.rate
    }
    fn total_duration(&self) -> Option<std::time::Duration> {
        None
    }
}

impl AudioOutput {
    /// Open the default output device for `scene`. Returns `None` (and the
    /// game stays silent) when there is no device or `ENGINE_AUDIO=0`.
    pub(crate) fn start(scene: &AudioScene) -> Option<Self> {
        if std::env::var("ENGINE_AUDIO").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false")) {
            return None;
        }
        let (mixer, connected) = scene.mixer_handle();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("engine-audio".into())
            .spawn(move || {
                let mut stream = match rodio::OutputStreamBuilder::open_default_stream() {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[engine] no audio output ({e}); the game runs silent");
                        return;
                    }
                };
                stream.log_on_drop(false);
                let rate = stream.config().sample_rate();
                mixer_lock(&mixer).set_rate(rate);
                stream.mixer().add(MixSource {
                    mixer,
                    buf: Vec::new(),
                    at: 0,
                    rate,
                });
                connected.store(true, Ordering::Relaxed);
                // Keep the stream alive until the game ends.
                let _ = stop_rx.recv();
                connected.store(false, Ordering::Relaxed);
                drop(stream);
            })
            .ok()?;
        Some(Self {
            stop: Some(stop_tx),
            thread: Some(thread),
        })
    }
}

impl Drop for AudioOutput {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose_at(pos: Vec3) -> ListenerPose {
        ListenerPose {
            pos,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
        }
    }

    fn energy(buf: &[f32], ch: usize) -> f32 {
        buf.iter().skip(ch).step_by(2).map(|v| v * v).sum()
    }

    fn hear(play: PlaySound, listener: Vec3) -> (f32, f32) {
        let mut scene = AudioScene::new();
        scene.play(&Sound::tone(300.0, 2.0), play.looped());
        scene.update(0.016, pose_at(listener), Acoustics::OUTDOORS);
        let out = scene.render_offline(4096);
        (energy(&out, 0), energy(&out, 1))
    }

    #[test]
    fn a_sound_on_the_right_is_louder_in_the_right_ear() {
        let (l, r) = hear(PlaySound::at([5.0, 0.0, 0.0]), Vec3::ZERO);
        assert!(r > l * 4.0, "left {l} right {r}");
    }

    #[test]
    fn distance_makes_sounds_quieter_and_range_silences_them() {
        let (near, _) = hear(PlaySound::at([0.0, 0.0, -3.0]), Vec3::ZERO);
        let (far, _) = hear(PlaySound::at([0.0, 0.0, -60.0]), Vec3::ZERO);
        let (gone, _) = hear(PlaySound::at([0.0, 0.0, -500.0]), Vec3::ZERO);
        assert!(near > far * 10.0);
        assert!(gone < 1e-6);
    }

    #[test]
    fn music_intensity_brings_in_layers() {
        let track = MusicTrack::new()
            .layer(Sound::tone(220.0, 1.0))
            .layer_between(Sound::tone(330.0, 1.0), 0.5, 0.7);
        let g = track.gains(0.0);
        assert_eq!(g[0], 1.0);
        assert_eq!(g[1], 0.0);
        assert_eq!(track.gains(1.0)[1], 1.0);
    }

    #[test]
    fn one_shots_retire_without_a_device() {
        let mut scene = AudioScene::new();
        let id = scene.play(&Sound::tone(440.0, 0.1), PlaySound::ui());
        assert!(scene.is_playing(id));
        for _ in 0..30 {
            scene.update(0.016, pose_at(Vec3::ZERO), Acoustics::OUTDOORS);
        }
        assert!(!scene.is_playing(id));
        assert_eq!(scene.voice_count(), 0);
    }

    #[test]
    fn too_many_voices_steal_the_quietest() {
        let mut scene = AudioScene::new();
        for _ in 0..(MAX_VOICES + 10) {
            scene.play(&Sound::tone(440.0, 5.0), PlaySound::at([0.0, 0.0, -10.0]));
        }
        assert!(scene.voice_count() <= MAX_VOICES);
    }

    #[test]
    fn walls_muffle_a_sound() {
        let mut clear = AudioScene::new();
        let mut walled = AudioScene::new();
        for (scene, occ) in [(&mut clear, 0.0), (&mut walled, 1.0)] {
            let id = scene.play(
                &Sound::tone(3000.0, 2.0),
                PlaySound::at([0.0, 0.0, -5.0]).looped(),
            );
            scene.set_voice_occlusion(id.0, occ);
            for _ in 0..20 {
                scene.update(0.05, pose_at(Vec3::ZERO), Acoustics::OUTDOORS);
            }
        }
        // The first block glides to the target; measure the second.
        let _ = (clear.render_offline(1024), walled.render_offline(1024));
        let a = energy(&clear.render_offline(4096), 0);
        let b = energy(&walled.render_offline(4096), 0);
        assert!(b < a * 0.1, "clear {a} walled {b}");
    }
}
