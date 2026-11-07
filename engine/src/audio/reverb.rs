//! A Freeverb-style stereo reverb (8 damped combs and 4 all-passes per
//! side) with a pre-delay. Its settings glide, so walking from a field into
//! a cave changes the sound smoothly.

/// How a space sounds. Mix two with [`Reverb::lerp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reverb {
    /// 0 (tiny, short tail) to 1 (huge, very long tail).
    pub size: f32,
    /// 0 (bright, hard stone) to 1 (dull, soft furnishings or earth).
    pub damping: f32,
    /// How loud the reverb is next to the dry sound, 0..1.
    pub wet: f32,
    /// Delay before the first reflections, in milliseconds (0..120): larger
    /// spaces have longer gaps.
    pub predelay_ms: f32,
}

impl Reverb {
    /// Open air: almost dry, a faint short echo off the ground.
    pub const OUTDOORS: Self = Self {
        size: 0.2,
        damping: 0.7,
        wet: 0.06,
        predelay_ms: 5.0,
    };
    /// A furnished room.
    pub const ROOM: Self = Self {
        size: 0.45,
        damping: 0.6,
        wet: 0.22,
        predelay_ms: 6.0,
    };
    /// A big hall, church or warehouse.
    pub const HALL: Self = Self {
        size: 0.82,
        damping: 0.35,
        wet: 0.35,
        predelay_ms: 30.0,
    };
    /// A rock cave: long, ringing tail.
    pub const CAVE: Self = Self {
        size: 0.9,
        damping: 0.2,
        wet: 0.45,
        predelay_ms: 40.0,
    };
    /// Under water: close and muffled (the muffling itself comes from
    /// [`super::Acoustics::muffle_hz`]).
    pub const UNDERWATER: Self = Self {
        size: 0.6,
        damping: 0.9,
        wet: 0.4,
        predelay_ms: 2.0,
    };

    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let m = |a: f32, b: f32| a + (b - a) * t;
        Self {
            size: m(self.size, other.size),
            damping: m(self.damping, other.damping),
            wet: m(self.wet, other.wet),
            predelay_ms: m(self.predelay_ms, other.predelay_ms),
        }
    }

    pub(crate) fn sanitized(self) -> Self {
        let c = |v: f32, hi: f32| if v.is_finite() { v.clamp(0.0, hi) } else { 0.0 };
        Self {
            size: c(self.size, 1.0),
            damping: c(self.damping, 1.0),
            wet: c(self.wet, 1.0),
            predelay_ms: c(self.predelay_ms, 120.0),
        }
    }
}

impl Default for Reverb {
    fn default() -> Self {
        Self::OUTDOORS
    }
}

const COMBS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASSES: [usize; 4] = [556, 441, 341, 225];
const SPREAD: usize = 23;

struct Comb {
    buf: Vec<f32>,
    i: usize,
    store: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            i: 0,
            store: 0.0,
        }
    }
    #[inline]
    fn run(&mut self, x: f32, feedback: f32, damp: f32) -> f32 {
        let y = self.buf[self.i];
        self.store = y * (1.0 - damp) + self.store * damp;
        self.buf[self.i] = x + self.store * feedback;
        self.i = (self.i + 1) % self.buf.len();
        y
    }
}

struct AllPass {
    buf: Vec<f32>,
    i: usize,
}

impl AllPass {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            i: 0,
        }
    }
    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let b = self.buf[self.i];
        self.buf[self.i] = x + b * 0.5;
        self.i = (self.i + 1) % self.buf.len();
        b - x
    }
}

pub(crate) struct ReverbUnit {
    combs: [Vec<Comb>; 2],
    allpasses: [Vec<AllPass>; 2],
    predelay: Vec<f32>,
    pre_i: usize,
    rate: f32,
    current: Reverb,
    target: Reverb,
}

impl ReverbUnit {
    pub fn new(rate: u32) -> Self {
        let scale = rate as f32 / 44_100.0;
        let len = |n: usize, side: usize| ((n + side * SPREAD) as f32 * scale) as usize;
        let combs = [0, 1].map(|s| COMBS.iter().map(|&n| Comb::new(len(n, s))).collect());
        let allpasses =
            [0, 1].map(|s| ALLPASSES.iter().map(|&n| AllPass::new(len(n, s))).collect());
        Self {
            combs,
            allpasses,
            predelay: vec![0.0; (rate as f32 * 0.125) as usize + 1],
            pre_i: 0,
            rate: rate as f32,
            current: Reverb::OUTDOORS,
            target: Reverb::OUTDOORS,
        }
    }

    pub fn set_target(&mut self, reverb: Reverb) {
        self.target = reverb.sanitized();
    }

    /// Mix the mono `send` into `out` (stereo interleaved).
    pub fn process(&mut self, send: &[f32], out: &mut [f32]) {
        // Glide toward the target once per block (blocks are ~5 ms).
        self.current = self.current.lerp(self.target, 0.05);
        let r = self.current;
        let feedback = 0.7 + 0.28 * r.size;
        let damp = 0.4 * r.damping;
        let wet = r.wet * 0.25;
        let delay = ((r.predelay_ms / 1000.0) * self.rate) as usize;
        let delay = delay.min(self.predelay.len() - 1);
        let plen = self.predelay.len();
        for (i, &x) in send.iter().enumerate() {
            self.predelay[self.pre_i] = x;
            let input = self.predelay[(self.pre_i + plen - delay) % plen] * 0.015;
            self.pre_i = (self.pre_i + 1) % plen;
            for side in 0..2 {
                let mut acc = 0.0;
                for c in self.combs[side].iter_mut() {
                    acc += c.run(input, feedback, damp);
                }
                for a in self.allpasses[side].iter_mut() {
                    acc = a.run(acc);
                }
                out[2 * i + side] += acc * wet;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Reverb, ReverbUnit};

    #[test]
    fn a_cave_rings_longer_than_open_air() {
        let tail = |r: Reverb| {
            let mut unit = ReverbUnit::new(48_000);
            unit.current = r;
            unit.target = r;
            let mut send = vec![0.0f32; 48_000];
            send[0] = 1.0;
            let mut out = vec![0.0f32; 96_000];
            unit.process(&send, &mut out);
            // Energy in the second half second.
            out[48_000..].iter().map(|v| v * v).sum::<f32>()
        };
        assert!(tail(Reverb::CAVE) > tail(Reverb::OUTDOORS) * 20.0);
    }
}
