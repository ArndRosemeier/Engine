//! Small deterministic random numbers and noise for the vegetation generator.

use glam::Vec3;

/// 32-bit integer hash (lowbias32).
pub(crate) fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

pub(crate) fn hash2i(seed: u32, x: i32, y: i32) -> u32 {
    hash_u32(seed ^ hash_u32((x as u32).wrapping_mul(0x9e37_79b9) ^ hash_u32(y as u32)))
}

pub(crate) fn hash_to_float(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

fn hash_str(seed: u64, s: &str) -> u64 {
    let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// SplitMix64 stream with labelled forks, so adding draws in one part of the
/// generator leaves the others unchanged.
#[derive(Clone, Debug)]
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub fn from(parts: &[&str], n: u32) -> Self {
        let mut h = 0x1234_5678_9abc_def0u64;
        for p in parts {
            h = hash_str(h, p);
        }
        Self {
            state: h ^ (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        }
    }

    pub fn fork(&self, label: &str) -> Self {
        Self {
            state: hash_str(self.state, label),
        }
    }

    pub fn fork_n(&self, label: &str, n: u32) -> Self {
        Self {
            state: hash_str(self.state, label) ^ (n as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn float(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.float()
    }

    /// Integer in [a, b] inclusive.
    pub fn int(&mut self, a: u32, b: u32) -> u32 {
        a + (self.float() * (b - a + 1) as f32) as u32
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.float() < p
    }

    pub fn sign(&mut self) -> f32 {
        if self.float() < 0.5 {
            -1.0
        } else {
            1.0
        }
    }

    pub fn on_sphere(&mut self) -> Vec3 {
        let z = self.range(-1.0, 1.0);
        let a = self.range(0.0, std::f32::consts::TAU);
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * a.cos(), r * a.sin(), z)
    }
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Smooth value noise in [-1, 1].
pub(crate) fn noise2(seed: u32, x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (fade(x - xi), fade(y - yi));
    let (xi, yi) = (xi as i32, yi as i32);
    let h = |i: i32, j: i32| hash_to_float(hash2i(seed, i, j));
    let (a, b, c, d) = (h(xi, yi), h(xi + 1, yi), h(xi, yi + 1), h(xi + 1, yi + 1));
    (a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy) * 2.0 - 1.0
}

/// Smooth value noise in [-1, 1].
pub(crate) fn noise3(seed: u32, p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let (fx, fy, fz) = (fade(f.x), fade(f.y), fade(f.z));
    let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
    let h = |a: i32, b: i32, c: i32| {
        hash_to_float(hash_u32(
            hash2i(seed, a, b) ^ (c as u32).wrapping_mul(0x27d4_eb2d),
        ))
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(h(x, y, z), h(x + 1, y, z), fx);
    let x10 = lerp(h(x, y + 1, z), h(x + 1, y + 1, z), fx);
    let x01 = lerp(h(x, y, z + 1), h(x + 1, y, z + 1), fx);
    let x11 = lerp(h(x, y + 1, z + 1), h(x + 1, y + 1, z + 1), fx);
    lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz) * 2.0 - 1.0
}

pub(crate) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_deterministic_and_forks_differ() {
        let a = Rng::from(&["tree", "oak"], 3);
        let (mut x, mut y) = (a.fork("leaves"), a.fork("leaves"));
        assert_eq!(x.float(), y.float());
        let mut z = a.fork("skeleton");
        assert_ne!(x.float(), z.float());
        for _ in 0..1000 {
            let v = z.int(2, 5);
            assert!((2..=5).contains(&v));
        }
    }

    #[test]
    fn noise_stays_in_range() {
        for i in 0..500 {
            let p = Vec3::new(i as f32 * 0.37, i as f32 * 0.11, i as f32 * 0.73);
            assert!(noise3(7, p).abs() <= 1.0);
            assert!(noise2(7, p.x, p.z).abs() <= 1.0);
        }
    }
}
