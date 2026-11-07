//! A physical sky with a day, a moon and weather (port of AssetGenerator's
//! `@futuremagic/sky`, itself from Scale's SkySystem).
//!
//! [`PhysicalSky`] says what time it is and what the weather does; the world
//! turns that into the sun or moon light, the sky-dome ambient, haze and
//! exposure each frame, and the sky pass draws a Preetham sky with a sun
//! disc and glow, twilight after sunset, a deep-blue night with stars, a
//! phased moon, and a cloud layer. This module is the numbers half: where
//! the sun and moon stand and what light they give, no GPU.

use glam::Vec3;

/// Days from one new moon to the next.
pub const SYNODIC_DAYS: f32 = 29.53;

/// Angular radius of the drawn sun and moon discs (about three times the
/// real ones, so they read on screen).
pub const SUN_RADIUS: f32 = 0.026;
pub const MOON_RADIUS: f32 = 0.024;

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Unit direction to the sun at `hour` (0..24) on a path like ~45°N around
/// the equinox: up at 6, highest at noon in the south (+z), down at 18. The
/// moon runs the same path, shifted by its phase.
pub fn sky_path(hour: f32) -> Vec3 {
    let t = (hour - 6.0) / 24.0 * std::f32::consts::TAU;
    let elev = t.sin() * 0.8;
    let az = t + 0.4;
    Vec3::new(
        az.cos() * elev.cos(),
        elev.sin(),
        az.sin() * elev.cos() * 0.6 + 0.35,
    )
    .normalize()
}

/// Lunar phase 0..1 (0 new, 0.5 full) `hours` into the game.
pub fn moon_phase(hours: f32, offset: f32) -> f32 {
    (hours / 24.0 / SYNODIC_DAYS + offset).rem_euclid(1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moon {
    pub dir: Vec3,
    /// Direction sunlight falls on the moon from (sets the phase shown).
    pub sun_side: Vec3,
    /// Lit fraction of the disc, 0 new .. 1 full.
    pub lit: f32,
}

/// The moon at `hour` of a day `hours` into the game: it runs the sun's
/// path, lagging by its phase.
pub fn moon_at(hour: f32, hours: f32, sun: Vec3) -> Moon {
    MoonSpec::default().at(hour, hours, sun).moon
}

/// Most moons in a sky.
pub const MAX_MOONS: usize = 4;

/// One moon of a world with several (see
/// [`crate::world::World::set_moons`]): how big it looks, how fast it goes
/// through its phases, where in its cycle it starts and how its path is
/// tilted against the sun's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoonSpec {
    /// Disc size, 1 = the usual moon (0.1..8).
    pub size: f32,
    /// Days from one new moon to the next (Earth's: 29.53).
    pub period_days: f32,
    /// Where in its cycle it is on day 0 (0 new, 0.5 full).
    pub phase: f32,
    /// Its path turned this many degrees about the east-west line from the
    /// sun's (it rides higher or lower in the sky).
    pub tilt_deg: f32,
    /// Light it reflects, 1 = the usual moon.
    pub brightness: f32,
    /// Colour of its surface (and its light).
    pub color: Vec3,
}

impl Default for MoonSpec {
    fn default() -> Self {
        Self {
            size: 1.0,
            period_days: SYNODIC_DAYS,
            phase: 0.3,
            tilt_deg: 0.0,
            brightness: 1.0,
            color: Vec3::ONE,
        }
    }
}

impl MoonSpec {
    pub fn new(size: f32, period_days: f32, phase: f32) -> Self {
        Self {
            size,
            period_days,
            phase,
            ..Self::default()
        }
    }

    pub fn tilt(mut self, degrees: f32) -> Self {
        self.tilt_deg = degrees;
        self
    }

    pub fn brightness(mut self, b: f32) -> Self {
        self.brightness = b;
        self
    }

    /// Surface colour as linear RGB (white: none).
    pub fn color(mut self, color: crate::color::Color) -> Self {
        self.color = color.to_vec3();
        self
    }

    pub fn validate(&self) -> crate::error::EngineResult<()> {
        let ok = |v: f32, lo: f32, hi: f32| v.is_finite() && (lo..=hi).contains(&v);
        if !ok(self.size, 0.1, 8.0)
            || !ok(self.period_days, 0.1, 10_000.0)
            || !self.phase.is_finite()
            || !ok(self.tilt_deg, -80.0, 80.0)
            || !ok(self.brightness, 0.0, 10.0)
            || !self.color.is_finite()
            || self.color.min_element() < 0.0
        {
            return Err(crate::error::EngineError::InvalidValue(
                "moon out of range (size 0.1..8, period 0.1..10000 days, tilt -80..80, \
                 brightness 0..10)"
                    .into(),
            ));
        }
        Ok(())
    }

    /// This moon at `hour` of a day `hours` into the game.
    pub fn at(&self, hour: f32, hours: f32, sun: Vec3) -> MoonView {
        let phase = (hours / 24.0 / self.period_days + self.phase).rem_euclid(1.0);
        let dir =
            glam::Quat::from_rotation_x(self.tilt_deg.to_radians()) * sky_path(hour - phase * 24.0);
        let pa = phase * std::f32::consts::TAU;
        let mut perp = sun - dir * sun.dot(dir);
        if perp.length_squared() < 1e-6 {
            perp = Vec3::new(-dir.x * dir.y, 1.0 - dir.y * dir.y, -dir.z * dir.y);
        }
        let perp = perp.normalize_or_zero();
        let s = pa.sin() * if pa > std::f32::consts::PI { -1.0 } else { 1.0 };
        MoonView {
            moon: Moon {
                dir,
                sun_side: dir * pa.cos() + perp * s,
                lit: (1.0 - pa.cos()) * 0.5,
            },
            radius: MOON_RADIUS * self.size,
            light: self.brightness * self.size * self.size,
            color: self.color,
        }
    }
}

/// A moon as it stands in the sky right now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoonView {
    pub moon: Moon,
    /// Angular radius of its disc.
    pub radius: f32,
    /// Light it gives when full and high, 1 = the usual moon.
    pub light: f32,
    pub color: Vec3,
}

/// Weather as the sky sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyWeather {
    /// Cloud cover 0..1.
    pub cover: f32,
    /// Cloud opacity.
    pub density: f32,
    /// Storm darkness 0..1.
    pub dark: f32,
    /// Direct sun left, 0..1.
    pub sun: f32,
    /// Fog 0..1.
    pub fog: f32,
    /// Rain 0..1.
    pub rain: f32,
}

impl SkyWeather {
    pub const CLEAR: Self = Self {
        cover: 0.35,
        density: 0.45,
        dark: 0.0,
        sun: 1.0,
        fog: 0.0,
        rain: 0.0,
    };
    pub const OVERCAST: Self = Self {
        cover: 0.95,
        density: 0.8,
        dark: 0.25,
        sun: 0.25,
        fog: 0.1,
        rain: 0.0,
    };
    pub const STORM: Self = Self {
        cover: 1.0,
        density: 1.0,
        dark: 0.8,
        sun: 0.08,
        fog: 0.3,
        rain: 1.0,
    };
}

impl Default for SkyWeather {
    fn default() -> Self {
        Self::CLEAR
    }
}

/// What a game sets: the time and the weather. Hand it to
/// [`crate::world::World::set_physical_sky`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalSky {
    /// Hour of the day, 0..24.
    pub hour: f32,
    /// Whole days since the game began (the moon's phase follows them).
    pub day: f32,
    pub weather: SkyWeather,
    /// Metres up to the cloud layer: clouds pass overhead as the camera
    /// moves. 0 keeps them fixed to the view.
    pub cloud_height_m: f32,
    /// How fast clouds drift (1 = default).
    pub cloud_speed: f32,
}

impl PhysicalSky {
    pub fn at_hour(hour: f32) -> Self {
        Self {
            hour,
            day: 0.0,
            weather: SkyWeather::CLEAR,
            cloud_height_m: 1500.0,
            cloud_speed: 1.0,
        }
    }

    pub fn with_weather(mut self, weather: SkyWeather) -> Self {
        self.weather = weather;
        self
    }

    pub fn sun(&self) -> Vec3 {
        sky_path(self.hour)
    }

    pub fn moon(&self) -> Moon {
        moon_at(self.hour, self.day * 24.0 + self.hour, self.sun())
    }

    pub fn lighting(&self) -> SkyLighting {
        sky_lighting(self.sun(), self.moon(), self.weather)
    }

    /// Each of `moons` as it stands now (the usual single moon if empty).
    pub fn moon_views(&self, moons: &[MoonSpec]) -> Vec<MoonView> {
        let hours = self.day * 24.0 + self.hour;
        if moons.is_empty() {
            return vec![MoonSpec::default().at(self.hour, hours, self.sun())];
        }
        moons
            .iter()
            .map(|m| m.at(self.hour, hours, self.sun()))
            .collect()
    }

    /// The lighting with several moons: their light adds up, coming from
    /// where the brightest stand, tinted by their colours.
    pub fn lighting_with(&self, moons: &[MoonView]) -> SkyLighting {
        sky_lighting_moons(self.sun(), moons, self.weather)
    }
}

/// How much of the sun's light gets through the air to the ground, per
/// channel, relative to a sun straight overhead: the sky shader's own
/// extinction (Rayleigh + Mie, Preetham's air mass), so the light's colour
/// matches the sky's. White at noon, orange near the horizon, red at sunset.
pub fn sun_transmittance(sun_up: f32) -> Vec3 {
    let (turbidity, rayleigh, mie) = (4.0_f32, 1.4_f32, 0.004_f32);
    const R: [f32; 3] = [5.804_543e-6, 1.356_291_2e-5, 3.026_590_2e-5];
    const M: [f32; 3] = [1.839_991_9e14, 2.779_802_4e14, 4.079_048e14];
    let c = 0.2 * turbidity * 10e-18;
    let depth = |up: f32, i: usize| {
        let z = up.clamp(0.0, 1.0).acos();
        let inv = 1.0 / (z.cos() + 0.15 * (93.885 - z.to_degrees()).powf(-1.253));
        R[i] * rayleigh * 8.4e3 * inv + 0.434 * c * M[i] * mie * 1.25e3 * inv
    };
    Vec3::from_array(std::array::from_fn(|i| {
        (depth(1.0, i) - depth(sun_up.max(0.0), i)).exp()
    }))
}

/// The lighting a sky gives at one moment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyLighting {
    /// 0 night .. 1 day; golden hour and night 0..1.
    pub day: f32,
    pub golden: f32,
    pub night: f32,
    /// Direction the key light comes from (sun by day, moon by night).
    pub light_dir: Vec3,
    pub light_color: Vec3,
    /// In AssetGenerator's units: 3.2 for a low clear sun, about 4.5 at noon.
    pub light_intensity: f32,
    pub sky_color: Vec3,
    pub ground_color: Vec3,
    pub hemi_intensity: f32,
    /// Horizon colour and exponential fog density per metre.
    pub fog_color: Vec3,
    pub fog_density: f32,
    /// Tone-mapping exposure (AssetGenerator's scale: 0.55 by day).
    pub exposure: f32,
    pub overcast: f32,
    pub sky_fog: f32,
    pub stars: f32,
    pub moon_bright: f32,
}

pub fn sky_lighting(sun: Vec3, moon: Moon, wx: SkyWeather) -> SkyLighting {
    let view = MoonView {
        moon,
        radius: MOON_RADIUS,
        light: 1.0,
        color: Vec3::ONE,
    };
    sky_lighting_moons(sun, &[view], wx)
}

pub fn sky_lighting_moons(sun: Vec3, moons: &[MoonView], wx: SkyWeather) -> SkyLighting {
    let sun_up = sun.y;
    let day = smoothstep(-0.12, 0.15, sun_up);
    let night = 1.0 - smoothstep(-0.2, 0.02, sun_up);
    let golden = smoothstep(0.35, 0.02, sun_up) * day;
    let overcast = smoothstep(0.55, 1.0, wx.cover) * 0.9;
    let sky_fog = (wx.fog * 1.1 + wx.rain * 0.25).clamp(0.0, 1.0);
    // Moonlight: each moon by how high and how bright it is; the light
    // comes from where the moons are (weighted), tinted by their colours.
    let mut toward = Vec3::ZERO;
    let mut reach = 0.0;
    let mut lit = 0.0;
    let mut tint = Vec3::ZERO;
    for m in moons {
        let up = smoothstep(-0.03, 0.2, m.moon.dir.y);
        toward += m.moon.dir * up * m.light;
        reach += up * m.light;
        lit += m.moon.lit * up * m.light;
        tint += m.color * (m.moon.lit * up * m.light + 1e-4);
    }
    let tint = tint / tint.max_element().max(1e-4);
    // A high sun in a clear sky gives most of the light, so shadows read at
    // midday; a low or clouded sun leaves the sky to light the shade.
    let high = smoothstep(0.2, 0.7, sun_up) * wx.sun;
    let (light_dir, light_color, light_intensity) = if sun_up > -0.05 {
        let t = sun_transmittance(sun_up);
        (
            sun,
            Vec3::new(
                t.x.powf(0.8) + 0.03,
                t.y.powf(0.8) + 0.02,
                t.z.powf(0.8) + 0.02,
            ),
            3.2 * smoothstep(-0.05, 0.12, sun_up) * wx.sun * (1.0 + 0.4 * high),
        )
    } else {
        // Night: the moons while they are up, else a faint glow from overhead.
        let to = toward.normalize_or(Vec3::Y);
        let n = Vec3::new(to.x * 0.5, 1.0, to.z * 0.5).normalize();
        let d = n.lerp(to, reach.min(1.0)).normalize_or_zero();
        (
            d,
            Vec3::new(0.55, 0.65, 0.9) * tint,
            0.25 * night * wx.sun * (0.45 + 0.75 * lit.min(4.0)),
        )
    };
    let dull = 1.0 - wx.sun;
    let hemi_intensity = lerp(0.06, 0.75, day) * (1.0 + 0.35 * dull) * (1.0 - 0.4 * wx.dark);
    let mut sky = Vec3::new(
        lerp(0.25, 0.75, day),
        lerp(0.3, 0.84, day),
        lerp(0.5, 1.0, day),
    );
    let grey = (sky.x + sky.y + sky.z) / 3.0;
    sky = sky.lerp(Vec3::splat(grey), dull * 0.75);
    let ground = Vec3::new(
        lerp(0.06, 0.38, day),
        lerp(0.06, 0.34, day),
        lerp(0.08, 0.28, day),
    );
    let mut fog = Vec3::new(
        lerp(0.03, 0.72, day) + golden * 0.15,
        lerp(0.04, 0.8, day) + golden * 0.05,
        lerp(0.07, 0.9, day) - golden * 0.1,
    );
    let g = lerp(0.035, 0.66 * (1.0 - 0.6 * wx.dark), day);
    let gw = overcast.max((wx.fog * 1.3).min(1.0));
    fog = fog.lerp(Vec3::new(g * 0.96, g, g * 1.03), gw);
    SkyLighting {
        day,
        golden,
        night,
        light_dir,
        light_color,
        light_intensity,
        sky_color: sky,
        ground_color: ground,
        hemi_intensity,
        fog_color: fog,
        fog_density: 0.00012 + 0.0062 * wx.fog.powf(2.2) + 0.0011 * wx.rain,
        exposure: lerp(1.6, 0.55, day) * (1.0 + 0.45 * dull * day) * (1.0 - 0.45 * wx.dark * day),
        overcast,
        sky_fog,
        stars: night * (1.0 - smoothstep(0.4, 0.85, wx.cover)) * (1.0 - sky_fog),
        moon_bright: lerp(1.5, 0.3, day),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_rises_at_six_peaks_at_noon_sets_at_eighteen() {
        assert!(sky_path(6.0).y.abs() < 1e-5);
        assert!(sky_path(18.0).y.abs() < 1e-5);
        assert!(
            sky_path(12.0).y > 0.6 && sky_path(12.0).z > 0.0,
            "high, in the south"
        );
        assert!(sky_path(0.0).y < -0.6);
    }

    #[test]
    fn noon_light_is_white_and_sunset_red() {
        let noon = sun_transmittance(0.7);
        assert!(noon.x > 0.85 && noon.z > 0.7, "{noon}");
        let low = sun_transmittance(0.03);
        assert!(low.x > low.y && low.y > low.z, "reddened: {low}");
    }

    #[test]
    fn day_and_night_lighting() {
        let noon = PhysicalSky::at_hour(12.0).lighting();
        assert!(noon.day > 0.99 && noon.light_intensity > 4.0);
        assert!(noon.stars == 0.0);
        let midnight = PhysicalSky::at_hour(0.0).lighting();
        assert!(midnight.night > 0.99 && midnight.light_intensity < 0.3);
        assert!(midnight.exposure > noon.exposure);
        let storm = PhysicalSky::at_hour(12.0)
            .with_weather(SkyWeather::STORM)
            .lighting();
        assert!(storm.light_intensity < noon.light_intensity * 0.2);
        assert!(storm.fog_density > noon.fog_density * 5.0);
    }

    #[test]
    fn several_moons_add_their_light() {
        let sky = PhysicalSky::at_hour(0.0);
        let one = sky.lighting_with(&sky.moon_views(&[]));
        assert_eq!(one, sky.lighting(), "no moons listed: the usual one");
        // Two full moons high at midnight light more than one.
        let full = |size: f32, tilt: f32| MoonSpec::new(size, 10.0, 0.5).tilt(tilt);
        let a = sky.moon_views(&[full(1.0, 0.0)]);
        let b = sky.moon_views(&[full(1.0, 0.0), full(2.0, -20.0)]);
        assert!(a[0].moon.lit > 0.99 && a[0].moon.dir.y > 0.3);
        let la = sky.lighting_with(&a);
        let lb = sky.lighting_with(&b);
        assert!(
            lb.light_intensity > la.light_intensity * 1.5,
            "{} {}",
            la.light_intensity,
            lb.light_intensity
        );
        // A red moon tints the night light.
        let red = sky.moon_views(&[full(1.5, 0.0).color(crate::color::Color::rgb(255, 120, 90))]);
        let lr = sky.lighting_with(&red);
        assert!(lr.light_color.x / lr.light_color.z > la.light_color.x / la.light_color.z);
        assert!(MoonSpec::new(20.0, 1.0, 0.0).validate().is_err());
    }

    #[test]
    fn moon_phase_cycles() {
        let new = moon_at(0.0, (1.0 - 0.3) * SYNODIC_DAYS * 24.0, sky_path(0.0));
        assert!(new.lit < 0.01, "{}", new.lit);
        let full = moon_at(0.0, (1.5 - 0.3) * SYNODIC_DAYS * 24.0, sky_path(0.0));
        assert!(full.lit > 0.99);
    }
}
