//! Ready-made sounds made in code, so a procedural game has sound without
//! any audio files: explosions, crumbling, digging, knocks, splashes, and
//! seamless loops for fire, streams, surf and wind.
//!
//! Every constructor takes a `seed`, so a world can give each explosion or
//! knock its own slightly different voice.

use super::sound::Sound;

/// Rate the synthesised sounds are made at.
const RATE: u32 = 48_000;

/// Small fast random generator (xorshift), so sounds are repeatable.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    /// Uniform in -1..1.
    fn white(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
    /// Uniform in 0..1.
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

/// One-pole low-pass, `hz` cutoff.
struct LowPass {
    a: f32,
    y: f32,
}

impl LowPass {
    fn new(hz: f32) -> Self {
        Self {
            a: coeff(hz),
            y: 0.0,
        }
    }
    fn set(&mut self, hz: f32) {
        self.a = coeff(hz);
    }
    fn run(&mut self, x: f32) -> f32 {
        self.y += self.a * (x - self.y);
        self.y
    }
}

fn coeff(hz: f32) -> f32 {
    1.0 - (-2.0 * std::f32::consts::PI * hz.max(1.0) / RATE as f32).exp()
}

fn frames(seconds: f32) -> usize {
    (seconds * RATE as f32) as usize
}

/// Scale to a peak of `peak`.
fn normalize(mut s: Vec<f32>, peak: f32) -> Vec<f32> {
    let max = s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max > 1e-6 {
        let k = peak / max;
        s.iter_mut().for_each(|v| *v *= k);
    }
    s
}

/// Make a loop seamless: cross-fade the last `fade` seconds into the start.
fn seamless(mut s: Vec<f32>, fade: f32) -> Vec<f32> {
    let n = frames(fade).min(s.len() / 2);
    let len = s.len() - n;
    for i in 0..n {
        let t = i as f32 / n as f32;
        // Equal-power cross-fade.
        let (a, b) = (
            (t * std::f32::consts::FRAC_PI_2).sin(),
            (t * std::f32::consts::FRAC_PI_2).cos(),
        );
        s[i] = s[i] * a + s[len + i] * b;
    }
    s.truncate(len);
    s
}

fn mono(samples: Vec<f32>) -> Sound {
    Sound::from_mono(samples, RATE).expect("synthesised samples are finite")
}

impl Sound {
    /// A deep explosion: a sub-bass thump, a roar of noise that darkens as
    /// it fades, and a rumbling tail. About 3 seconds.
    pub fn explosion(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(3.2);
        let mut out = Vec::with_capacity(n);
        let mut roar = LowPass::new(3000.0);
        let mut rumble = LowPass::new(120.0);
        let mut rumble2 = LowPass::new(120.0);
        let pitch = 0.85 + 0.3 * rng.unit();
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            // Thump: falling sine.
            let f = 30.0 + 90.0 * (-t * 18.0).exp();
            let thump = (2.0 * std::f32::consts::PI * f * pitch * t).sin() * (-t * 6.0).exp();
            roar.set(200.0 + 3500.0 * (-t * 3.0).exp());
            let noise = rng.white();
            let crack = roar.run(noise) * (-t * 2.2).exp() * (1.0 - (-t * 200.0).exp());
            let rum = rumble2.run(rumble.run(noise)) * 4.0 * (-t * 0.9).exp();
            out.push(thump * 0.9 + crack * 0.8 + rum * 0.6);
        }
        mono(normalize(out, 0.95))
    }

    /// Stone and earth crumbling: many small grains and knocks over a soft
    /// rumble. About 2 seconds.
    pub fn crumble(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(2.2);
        let mut out = vec![0.0f32; n];
        let mut low = LowPass::new(300.0);
        for (i, o) in out.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            *o = low.run(rng.white()) * 1.5 * (-t * 1.6).exp();
        }
        // Grains: short decaying clicks, denser early on.
        let grains = 120 + (rng.unit() * 60.0) as usize;
        for _ in 0..grains {
            let start = (rng.unit().powf(2.0) * (n as f32 - 4000.0)) as usize;
            let len = 300 + (rng.unit() * 2500.0) as usize;
            let f = 400.0 + rng.unit() * 2600.0;
            let amp = 0.2 + 0.8 * rng.unit();
            let mut lp = LowPass::new(f);
            for k in 0..len.min(n - start) {
                let env = (-(k as f32) / (len as f32 * 0.25)).exp();
                out[start + k] += lp.run(rng.white()) * amp * env;
            }
        }
        mono(normalize(out, 0.9))
    }

    /// A spade or blast biting into soil: a dull thud with crunchy grit.
    pub fn dig(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(0.7);
        let mut out = Vec::with_capacity(n);
        let mut grit = LowPass::new(1800.0);
        let mut body = LowPass::new(160.0);
        let f0 = 70.0 + rng.unit() * 30.0;
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            let thud = (2.0 * std::f32::consts::PI * f0 * t).sin() * (-t * 14.0).exp();
            let noise = rng.white();
            let crunch = grit.run(noise) * (-t * 7.0).exp() * (0.6 + 0.4 * rng.unit());
            out.push(thud + body.run(noise) * 2.0 * (-t * 10.0).exp() + crunch * 0.5);
        }
        mono(normalize(out, 0.9))
    }

    /// A short knock, for things hitting each other. `hardness` 0 (wood,
    /// earth: dull) to 1 (stone, metal: bright and ringing).
    pub fn knock(seed: u64, hardness: f32) -> Self {
        let mut rng = Rng::new(seed);
        let h = hardness.clamp(0.0, 1.0);
        let n = frames(0.25 + 0.5 * h);
        let mut out = Vec::with_capacity(n);
        let mut click = LowPass::new(800.0 + 6000.0 * h);
        // A few inharmonic modes ring like a struck object.
        let base = 180.0 + 500.0 * h + rng.unit() * 80.0;
        let modes = [1.0, 2.32 + 0.1 * rng.unit(), 4.25, 6.63];
        let decay = 40.0 - 32.0 * h;
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            let mut ring = 0.0;
            for (k, m) in modes.iter().enumerate() {
                ring += (2.0 * std::f32::consts::PI * base * m * t).sin()
                    * (-t * decay * (1.0 + k as f32 * 0.6)).exp()
                    / (1.0 + k as f32);
            }
            let hit = click.run(rng.white()) * (-t * 300.0).exp();
            out.push(ring * 0.6 + hit);
        }
        mono(normalize(out, 0.85))
    }

    /// Something falling into water: a plop and a spray of droplets.
    pub fn splash(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(1.1);
        let mut out = vec![0.0f32; n];
        let mut spray = LowPass::new(4000.0);
        for (i, o) in out.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            // Plop: a quick upward chirp (the bubble cavity resonating).
            let f = 250.0 + 900.0 * t.min(0.08) / 0.08;
            let plop = (2.0 * std::f32::consts::PI * f * t).sin() * (-t * 25.0).exp();
            let wash = spray.run(rng.white()) * (-t * 3.5).exp() * (1.0 - (-t * 60.0).exp());
            *o = plop * 0.8 + wash * 0.7;
        }
        add_droplets(&mut rng, &mut out, 40, 0.05, 0.9, 0.35);
        mono(normalize(out, 0.9))
    }

    /// A crackling fire loop (4 s, seamless): a soft roar with pops.
    pub fn fire_loop(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(4.5);
        let mut out = Vec::with_capacity(n);
        let mut roar = LowPass::new(500.0);
        let mut slow = LowPass::new(2.0);
        for _ in 0..n {
            let flutter = 0.6 + 2.0 * slow.run(rng.white()).abs();
            out.push(roar.run(rng.white()) * 1.2 * flutter);
        }
        // Crackles: sharp clicks of varying brightness.
        for _ in 0..90 {
            let start = (rng.unit() * (n as f32 - 2000.0)) as usize;
            let len = 80 + (rng.unit() * 900.0) as usize;
            let amp = 0.15 + rng.unit().powf(3.0) * 1.2;
            let mut lp = LowPass::new(1500.0 + rng.unit() * 6000.0);
            for k in 0..len {
                out[start + k] +=
                    lp.run(rng.white()) * amp * (-(k as f32) / (len as f32 * 0.2)).exp();
            }
        }
        mono(normalize(seamless(out, 0.5), 0.8))
    }

    /// A babbling stream or river loop (5 s, seamless): rushing water with
    /// gurgles. Louder and darker for big rivers at higher `flow` (0..1).
    pub fn stream_loop(seed: u64, flow: f32) -> Self {
        let mut rng = Rng::new(seed);
        let flow = flow.clamp(0.0, 1.0);
        let n = frames(5.6);
        let mut out = vec![0.0f32; n];
        let mut rush = LowPass::new(1400.0 - 700.0 * flow);
        let mut hiss = LowPass::new(5000.0);
        let mut slow = LowPass::new(0.7);
        for o in out.iter_mut() {
            let swell = 0.8 + 1.5 * slow.run(rng.white()).abs();
            *o = rush.run(rng.white()) * swell * (1.0 + flow) + hiss.run(rng.white()) * 0.12;
        }
        // Gurgles: bubbles ringing up in pitch.
        add_droplets(
            &mut rng,
            &mut out,
            (180.0 * (1.0 - 0.5 * flow)) as usize,
            0.02,
            0.0,
            0.25,
        );
        mono(normalize(seamless(out, 0.6), 0.8))
    }

    /// Waves on a beach (8 s, seamless): slow swells that break and wash.
    pub fn surf_loop(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(8.8);
        let mut out = Vec::with_capacity(n);
        let mut wash = LowPass::new(900.0);
        let mut body = LowPass::new(250.0);
        let period = 4.4 + rng.unit() * 0.6;
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            let phase = (t / period).fract();
            // Each wave: rising swell, a break, then a long hiss.
            let env = if phase < 0.25 {
                (phase / 0.25).powi(2) * 0.5
            } else {
                0.5 + 0.5 * (-(phase - 0.25) * 4.0).exp()
            } * (1.0 - (phase - 0.97).max(0.0) * 30.0).max(0.0);
            wash.set(400.0 + 2600.0 * env);
            let noise = rng.white();
            out.push(wash.run(noise) * env * 1.4 + body.run(noise) * 1.8 * env);
        }
        mono(normalize(seamless(out, 0.8), 0.8))
    }

    /// Wind loop (6 s, seamless): gusting, whistling air.
    pub fn wind_loop(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = frames(6.6);
        let mut out = Vec::with_capacity(n);
        let mut band = LowPass::new(400.0);
        let mut band2 = LowPass::new(400.0);
        let mut gust = LowPass::new(0.4);
        for _ in 0..n {
            let g = (0.4 + 3.0 * gust.run(rng.white()).abs()).min(1.5);
            band.set(250.0 + 700.0 * g);
            band2.set(150.0 + 400.0 * g);
            let x = rng.white();
            out.push((band.run(x) - band2.run(x)) * g * 3.0);
        }
        mono(normalize(seamless(out, 0.7), 0.7))
    }

    /// A clean sine tone, for UI beeps and tests.
    pub fn tone(hz: f32, seconds: f32) -> Self {
        let n = frames(seconds.max(0.01));
        let fade = frames(0.005).max(1);
        let out = (0..n)
            .map(|i| {
                let env = (i.min(n - 1 - i) as f32 / fade as f32).min(1.0);
                (2.0 * std::f32::consts::PI * hz * i as f32 / RATE as f32).sin() * 0.5 * env
            })
            .collect();
        mono(out)
    }
}

/// Add `count` water droplets: short chirps rising in pitch. `from`/`to`
/// (0..1) is where they fall in the clip, weighted toward the start by
/// `early` (0 = spread evenly).
fn add_droplets(rng: &mut Rng, out: &mut [f32], count: usize, amp: f32, early: f32, spread: f32) {
    let n = out.len();
    for _ in 0..count {
        let u = rng.unit();
        let pos = if early > 0.0 {
            u.powf(1.0 + 2.0 * early)
        } else {
            u
        };
        let start = (pos * (n as f32 - 3000.0)) as usize;
        let f0 = 600.0 + rng.unit() * 2000.0;
        let len = 600 + (rng.unit() * 2000.0) as usize;
        let a = amp * (0.3 + rng.unit()) / spread.max(0.05) * 0.1;
        let mut phase = 0.0f32;
        for k in 0..len {
            let t = k as f32 / RATE as f32;
            let f = f0 * (1.0 + t * 30.0);
            phase += 2.0 * std::f32::consts::PI * f / RATE as f32;
            out[start + k] += phase.sin() * a * (-(k as f32) / (len as f32 * 0.3)).exp();
        }
    }
}

/// Synthesised music stems for [`super::MusicTrack::procedural`]: all the
/// same length, so they loop in step.
pub(crate) mod music {
    use super::{frames, normalize, LowPass, Rng, RATE};
    use std::f32::consts::PI;

    pub struct Stems {
        pub pad: Vec<f32>,
        pub bass: Vec<f32>,
        pub arp: Vec<f32>,
        pub drums: Vec<f32>,
    }

    fn midi_hz(note: f32) -> f32 {
        440.0 * 2f32.powf((note - 69.0) / 12.0)
    }

    /// Eight bars of a minor-key progression at `bpm`, as stereo-ready mono
    /// stems.
    pub fn stems(seed: u64, bpm: f32) -> Stems {
        let mut rng = Rng::new(seed);
        let beat = 60.0 / bpm.clamp(50.0, 180.0);
        let bars = 8;
        let n = frames(beat * 4.0 * bars as f32);
        // Root notes per bar (A minor-ish, picked by seed).
        let progressions: [[i32; 4]; 4] = [
            [0, -4, -7, -2],
            [0, 3, -2, -4],
            [0, -5, -2, -7],
            [0, 5, 3, -2],
        ];
        let prog = progressions[(rng.next_u32() % 4) as usize];
        let key = 45.0 + (rng.next_u32() % 5) as f32; // A2..C#3
        let root_at = |bar: usize| key + prog[bar % 4] as f32;
        let bar_len = beat * 4.0;

        let mut pad = vec![0.0f32; n];
        let mut bass = vec![0.0f32; n];
        let mut arp = vec![0.0f32; n];
        let mut drums = vec![0.0f32; n];

        // Pad: soft detuned saw chord, low-passed, slow swell per bar.
        let mut lp = LowPass::new(900.0);
        let mut phases = [0.0f32; 6];
        for (i, s) in pad.iter_mut().enumerate() {
            let t = i as f32 / RATE as f32;
            let bar = (t / bar_len) as usize;
            let in_bar = (t / bar_len).fract();
            let root = root_at(bar) + 12.0;
            let chord = [root, root + 3.0, root + 7.0];
            let mut v = 0.0;
            for (k, note) in chord.iter().enumerate() {
                for d in 0..2 {
                    let f = midi_hz(*note) * (1.0 + (d as f32 - 0.5) * 0.006);
                    let p = &mut phases[k * 2 + d];
                    *p = (*p + f / RATE as f32).fract();
                    v += *p * 2.0 - 1.0;
                }
            }
            let swell = 0.55 + 0.45 * (in_bar * PI).sin();
            *s = lp.run(v) * swell;
        }

        // Bass: plucked root on beats 1 and 3, fifth on the "and" of 4.
        for bar in 0..bars {
            let root = root_at(bar) - 12.0;
            for (b, note) in [(0.0, root), (2.0, root), (3.5, root + 7.0)] {
                let start = frames((bar as f32 * 4.0 + b) * beat);
                let len = frames(beat * 1.4);
                let f = midi_hz(note);
                for k in 0..len.min(n.saturating_sub(start)) {
                    let t = k as f32 / RATE as f32;
                    let env = (-t * 3.5).exp() * (t * 400.0).min(1.0);
                    let v = (2.0 * PI * f * t).sin() + 0.3 * (4.0 * PI * f * t).sin();
                    bass[start + k] += v * env;
                }
            }
        }

        // Arpeggio: eighth notes up the chord, bell-like.
        let pattern = [0.0, 3.0, 7.0, 12.0, 7.0, 3.0, 15.0, 12.0];
        for bar in 0..bars {
            let root = root_at(bar) + 24.0;
            for (step, iv) in pattern.iter().enumerate() {
                let start = frames((bar as f32 * 4.0 + step as f32 * 0.5) * beat);
                let len = frames(beat * 0.9);
                let f = midi_hz(root + iv);
                for k in 0..len.min(n.saturating_sub(start)) {
                    let t = k as f32 / RATE as f32;
                    let env = (-t * 6.0).exp() * (t * 800.0).min(1.0);
                    let v = (2.0 * PI * f * t).sin() + 0.25 * (2.0 * PI * f * 3.01 * t).sin();
                    arp[start + k] += v * env;
                }
            }
        }

        // Drums: kick on 1 and 3, snare on 2 and 4, soft hats on eighths.
        let mut hat_lp = LowPass::new(7000.0);
        let mut snare_lp = LowPass::new(3000.0);
        for bar in 0..bars {
            for eighth in 0..8 {
                let start = frames((bar as f32 * 4.0 + eighth as f32 * 0.5) * beat);
                let len = frames(0.35);
                for k in 0..len.min(n.saturating_sub(start)) {
                    let t = k as f32 / RATE as f32;
                    let mut v = 0.0;
                    if eighth % 4 == 0 {
                        let f = 50.0 + 100.0 * (-t * 30.0).exp();
                        v += (2.0 * PI * f * t).sin() * (-t * 9.0).exp() * 1.2;
                    }
                    if eighth % 4 == 2 {
                        v += snare_lp.run(rng.white()) * (-t * 18.0).exp() * 0.8
                            + (2.0 * PI * 190.0 * t).sin() * (-t * 25.0).exp() * 0.4;
                    }
                    let x = rng.white();
                    v += (x - hat_lp.run(x)) * (-t * 60.0).exp() * 0.25;
                    drums[start + k] += v;
                }
            }
        }

        Stems {
            pad: normalize(pad, 0.5),
            bass: normalize(bass, 0.6),
            arp: normalize(arp, 0.35),
            drums: normalize(drums, 0.6),
        }
    }

    pub const STEM_RATE: u32 = RATE;
}

#[cfg(test)]
mod tests {
    use super::Sound;

    #[test]
    fn synthesised_sounds_are_audible_and_in_range() {
        for s in [
            Sound::explosion(1),
            Sound::crumble(2),
            Sound::dig(3),
            Sound::knock(4, 0.2),
            Sound::knock(5, 0.9),
            Sound::splash(6),
            Sound::fire_loop(7),
            Sound::stream_loop(8, 0.5),
            Sound::surf_loop(9),
            Sound::wind_loop(10),
            Sound::tone(440.0, 0.2),
        ] {
            let peak = s.0.samples.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(peak > 0.1 && peak <= 1.0, "peak {peak}");
            assert!(s.duration() > 0.05);
        }
    }

    #[test]
    fn same_seed_same_sound() {
        assert_eq!(Sound::crumble(42).0.samples, Sound::crumble(42).0.samples);
        assert_ne!(Sound::crumble(42).0.samples, Sound::crumble(43).0.samples);
    }

    #[test]
    fn music_stems_share_one_length() {
        let s = super::music::stems(1, 100.0);
        assert_eq!(s.pad.len(), s.bass.len());
        assert_eq!(s.pad.len(), s.arp.len());
        assert_eq!(s.pad.len(), s.drums.len());
    }
}
