//! Fullscreen procedural sky, drawn behind every surface.
//!
//! A gradient from zenith through the horizon, a warm sun disc, and a little
//! drifting cloud, all from the view ray. Games pick the colours; this pass
//! only paints them. Drawn first in the colour pass so water and ground sit
//! in front without a depth fight.

use crate::camera::Camera;
use crate::world::{Light, Sky};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniforms {
    pub right: [f32; 3],
    pub tan_half_fov: f32,
    pub up: [f32; 3],
    pub aspect: f32,
    pub forward: [f32; 3],
    pub time: f32,
    pub zenith: [f32; 3],
    pub curve: f32,
    pub horizon: [f32; 3],
    pub sun_cos: f32,
    pub ground: [f32; 3],
    pub sun_bloom: f32,
    pub sun_dir: [f32; 3],
    /// 1 draws the physical sky (crate::sky), 0 the colour gradient.
    pub physical: f32,
    pub sun_color: [f32; 3],
    pub overcast: f32,
    pub moon_dir: [f32; 3],
    pub moon_bright: f32,
    pub moon_sun: [f32; 3],
    pub stars: f32,
    pub fog_col: [f32; 3],
    pub sky_fog: f32,
    pub cloud_origin: [f32; 3],
    pub cloud_cover: f32,
    /// Cloud density, storm darkness, layer height (m), drift time.
    pub cloud: [f32; 4],
    /// Per moon, smallest first: (dir, radius), (sun side, brightness),
    /// (colour, surface seed).
    pub moons: [[f32; 4]; 3 * crate::sky::MAX_MOONS],
    /// Moon count, then unused.
    pub moon_info: [f32; 4],
}

impl SkyUniforms {
    pub fn from_scene(sky: &Sky, camera: &Camera, light: &Light, aspect: f32, time: f32) -> Self {
        let forward = (camera.target() - camera.eye()).normalize_or_zero();
        let mut right = forward.cross(camera.up());
        if right.length_squared() < 1e-8 {
            right = glam::Vec3::X;
        }
        let right = right.normalize();
        let up = right.cross(forward).normalize();
        let tan_half_fov = (camera.fov_y_degrees().to_radians() * 0.5).tan();
        let sun_size = sky.sun_size_degrees.to_radians().max(0.001);
        let bloom = (sky.sun_size_degrees + sky.sun_bloom_degrees)
            .to_radians()
            .max(sun_size + 0.001);
        let sun_dir = light.direction.normalize_or_zero();
        Self {
            right: right.into(),
            tan_half_fov,
            up: up.into(),
            aspect: aspect.max(0.001),
            forward: forward.into(),
            time,
            zenith: sky.zenith.to_vec3().into(),
            curve: sky.curve.clamp(0.0, 1.0),
            horizon: sky.horizon.to_vec3().into(),
            sun_cos: sun_size.cos(),
            ground: sky.ground.to_vec3().into(),
            sun_bloom: bloom.cos(),
            sun_dir: sun_dir.into(),
            physical: 0.0,
            sun_color: sky.sun_color.to_vec3().into(),
            overcast: 0.0,
            moon_dir: [0.0, -1.0, 0.0],
            moon_bright: 0.0,
            moon_sun: [0.0, 1.0, 0.0],
            stars: 0.0,
            fog_col: [0.0; 3],
            sky_fog: 0.0,
            cloud_origin: [0.0; 3],
            cloud_cover: 0.0,
            cloud: [0.0; 4],
            moons: [[0.0; 4]; 3 * crate::sky::MAX_MOONS],
            moon_info: [0.0; 4],
        }
    }

    /// The physical sky's parameters on top of the camera basis.
    pub fn with_physical(
        mut self,
        sky: &crate::sky::PhysicalSky,
        moons: &[crate::sky::MoonSpec],
        camera: &Camera,
        time: f32,
    ) -> Self {
        let mut views = sky.moon_views(moons);
        let light = sky.lighting_with(&views);
        let moon = views[0].moon;
        // Seeds follow the list, so each moon keeps its face when sorted.
        let mut seeded: Vec<(crate::sky::MoonView, f32)> = views
            .drain(..)
            .enumerate()
            .map(|(i, v)| (v, i as f32))
            .collect();
        // Small ones first: the bigger (nearer) ones are drawn over them.
        seeded.sort_by(|a, b| a.0.radius.total_cmp(&b.0.radius));
        for (i, (v, seed)) in seeded.iter().take(crate::sky::MAX_MOONS).enumerate() {
            let d = v.moon.dir;
            let ss = v.moon.sun_side;
            self.moons[i * 3] = [d.x, d.y, d.z, v.radius];
            self.moons[i * 3 + 1] = [
                ss.x,
                ss.y,
                ss.z,
                v.light / (v.radius / crate::sky::MOON_RADIUS).powi(2),
            ];
            self.moons[i * 3 + 2] = [v.color.x, v.color.y, v.color.z, *seed];
        }
        self.moon_info = [
            seeded.len().min(crate::sky::MAX_MOONS) as f32,
            0.0,
            0.0,
            0.0,
        ];
        self.physical = 1.0;
        self.sun_dir = sky.sun().into();
        self.sun_cos = crate::sky::SUN_RADIUS.cos();
        self.overcast = light.overcast;
        self.moon_dir = moon.dir.into();
        self.moon_bright = light.moon_bright;
        self.moon_sun = moon.sun_side.into();
        self.stars = light.stars;
        self.fog_col = light.fog_color.into();
        self.sky_fog = light.sky_fog;
        self.cloud_origin = camera.eye().into();
        self.cloud_cover = sky.weather.cover;
        self.cloud = [
            sky.weather.density,
            sky.weather.dark,
            sky.cloud_height_m,
            time * sky.cloud_speed,
        ];
        self
    }
}

pub struct SkyPipelines {
    pub pipeline: wgpu::RenderPipeline,
    pub pipeline_portal: wgpu::RenderPipeline,
    pub uniform_bufs: [wgpu::Buffer; super::pipeline::SCENE_UNIFORM_SLOTS],
    pub bind_groups: [wgpu::BindGroup; super::pipeline::SCENE_UNIFORM_SLOTS],
}

impl SkyPipelines {
    pub fn bind_group(&self, slot: usize) -> &wgpu::BindGroup {
        self.bind_groups.get(slot).unwrap_or_else(|| {
            panic!(
                "sky uniform slot {slot} exceeds capacity {}",
                super::pipeline::SCENE_UNIFORM_SLOTS
            )
        })
    }
}

const SHADER: &str = r#"
struct SkyUniforms {
    right: vec3<f32>,
    tan_half_fov: f32,
    up: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    time: f32,
    zenith: vec3<f32>,
    curve: f32,
    horizon: vec3<f32>,
    sun_cos: f32,
    ground: vec3<f32>,
    sun_bloom: f32,
    sun_dir: vec3<f32>,
    physical: f32,
    sun_color: vec3<f32>,
    overcast: f32,
    moon_dir: vec3<f32>,
    moon_bright: f32,
    moon_sun: vec3<f32>,
    stars: f32,
    fog_col: vec3<f32>,
    sky_fog: f32,
    cloud_origin: vec3<f32>,
    cloud_cover: f32,
    cloud: vec4<f32>,
    moons: array<vec4<f32>, 12>,
    moon_info: vec4<f32>,
};

@group(0) @binding(0) var<uniform> s: SkyUniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VsOut;
    // Reversed-Z: 0 is the far plane, so the sky sits behind every surface.
    out.clip = vec4<f32>(p[i], 0.0, 1.0);
    out.ndc = p[i];
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx) * 2.0 - 1.0;
}

fn grad(i: vec3<f32>, f: vec3<f32>, c: vec3<f32>) -> f32 {
    var g = hash33(i + c);
    let len = length(g);
    // WGSL `select` may evaluate both values. Dividing by zero here poisoned
    // directional sky-noise pixels with NaNs, which some GPUs present as black tiles.
    let normalized = g / max(len, 1e-5);
    g = select(vec3<f32>(1.0, 0.0, 0.0), normalized, len > 1e-5);
    return dot(g, f - c);
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(
        mix(
            mix(grad(i, f, vec3<f32>(0.0, 0.0, 0.0)), grad(i, f, vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(grad(i, f, vec3<f32>(0.0, 1.0, 0.0)), grad(i, f, vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(grad(i, f, vec3<f32>(0.0, 0.0, 1.0)), grad(i, f, vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(grad(i, f, vec3<f32>(0.0, 1.0, 1.0)), grad(i, f, vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

fn fbm3(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var n = 0; n < 5; n++) {
        v += a * noise3(q);
        q = q.yzx * 2.02 + vec3<f32>(0.13, 0.07, 0.19);
        a *= 0.5;
    }
    return v * 0.5 + 0.5;
}

const PHYSICAL_SKY_SCALE: f32 = 1.0;
// ---- Physical sky (port of AssetGenerator's @futuremagic/sky SkyMesh) ----

fn pgrad(i: vec2<f32>) -> vec2<f32> {
    var p = fract(i.xyx * vec3<f32>(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.xx + p.yz) * p.zy) * 2.0 - 1.0;
}

fn pnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(pgrad(i), f);
    let b = dot(pgrad(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
    let c = dot(pgrad(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
    let d = dot(pgrad(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 1.6;
}

fn pfbm(position: vec2<f32>, drift: f32) -> f32 {
    var p = position;
    var result = 0.0;
    var amplitude = 1.0;
    for (var i = 0; i < 4; i++) {
        result += amplitude * pnoise(p);
        amplitude *= 0.5;
        p = p * 2.0 + drift;
    }
    return result;
}

// The glow while the sun is just below the horizon, where Preetham is black.
fn twilight(direction: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let sy = sun.y;
    let glow = smoothstep(-0.18, -0.01, sy) * ((1.0 - smoothstep(-0.01, 0.06, sy)) * 0.85 + 0.15)
        * (1.0 - smoothstep(0.03, 0.1, sy));
    let h = max(direction.y, 0.0);
    let flat_d = vec2<f32>(direction.x, direction.z);
    let flat_s = vec2<f32>(sun.x, sun.z);
    let toward = clamp(dot(flat_d / max(length(flat_d), 1e-4), flat_s / max(length(flat_s), 1e-4)) * 0.5 + 0.5, 0.0, 1.0);
    let low = pow(1.0 - h, 6.0);
    let band = vec3<f32>(0.9, 0.38, 0.12) * low * (pow(toward, 3.0) * 0.85 + 0.15) * smoothstep(-0.18, -0.02, sy);
    let blue = mix(vec3<f32>(0.05, 0.08, 0.16), vec3<f32>(0.012, 0.03, 0.085), pow(h, 0.6)) * (smoothstep(-0.2, -0.04, sy) + 0.3);
    let belt = vec3<f32>(0.12, 0.06, 0.09) * pow(1.0 - h, 3.0) * pow(1.0 - toward, 2.0);
    return (band + blue + belt) * glow * 0.5 * smoothstep(-0.12, 0.0, direction.y);
}

// The night once twilight has gone: deep blue, lighter at the horizon and under the moons.
fn night_sky(direction: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    let dark = 1.0 - smoothstep(-0.2, -0.06, sun.y);
    let h = max(direction.y, 0.0);
    let base = mix(vec3<f32>(0.022, 0.03, 0.055), vec3<f32>(0.006, 0.011, 0.03), pow(h, 0.5));
    var moon_up = 0.0;
    var halo = vec3<f32>(0.0);
    for (var i = 0; i < i32(s.moon_info.x); i++) {
        let m = s.moons[i * 3];
        let size = m.w / 0.024;
        // Light it gives: brightness times disc area.
        let light = s.moons[i * 3 + 1].w * size * size;
        let up = smoothstep(-0.05, 0.25, m.y) * min(light, 1.5);
        moon_up += up;
        // A bigger moon has a somewhat wider halo.
        let spread = 12.0 / sqrt(max(size, 0.2));
        halo += vec3<f32>(0.05, 0.06, 0.08) * s.moons[i * 3 + 2].rgb
            * pow(max(dot(direction, m.xyz), 0.0), spread) * up;
    }
    moon_up = min(moon_up, 1.4);
    return (base * (moon_up * 1.2 + 0.6) + halo) * dark * smoothstep(-0.15, 0.0, direction.y);
}

// Every moon over `color`: spheres lit from the sun's side (their phases),
// maria and craters (different on each), reddened low down; nearer (bigger)
// ones cover those behind.
fn draw_moons(color_in: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    var layer = vec3<f32>(0.0);
    var cover = 0.0;
    for (var i = 0; i < i32(s.moon_info.x); i++) {
        let m = s.moons[i * 3];
        let moon_r = m.w;
        let dir = m.xyz;
        let cos_m = dot(direction, dir);
        if cos_m <= cos(moon_r * 1.05) {
            continue;
        }
        let sun_side = s.moons[i * 3 + 1].xyz;
        let bright = s.moons[i * 3 + 1].w;
        let tint = s.moons[i * 3 + 2].rgb;
        let seed = s.moons[i * 3 + 2].w * 13.7;
        let m_right = normalize(cross(dir, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1e-5, 0.0, 0.0));
        let m_up = cross(m_right, dir);
        let mp = vec2<f32>(dot(direction, m_right), dot(direction, m_up)) / moon_r;
        let mr2 = dot(mp, mp);
        let m_disc = 1.0 - smoothstep(0.92, 1.0, mr2);
        let m_n = mp.x * m_right + mp.y * m_up - sqrt(max(0.0, 1.0 - mr2)) * dir;
        let m_lit = smoothstep(-0.04, 0.12, dot(m_n, sun_side));
        let maria = smoothstep(0.05, 0.45, pnoise(mp * 1.7 + vec2<f32>(3.1 + seed, 7.4)) * 0.7 + pnoise(mp * 3.9 + vec2<f32>(1.3, 2.9 + seed)) * 0.3);
        let craters = smoothstep(0.35, 0.7, pnoise(mp * 6.0 + vec2<f32>(5.7 + seed, 0.4))) * 0.14;
        let albedo = (1.0 - maria * 0.38) * (1.0 - craters);
        let m_col = vec3<f32>(0.92, 0.9, 0.85) * tint * albedo * (m_lit + 0.006) * bright
            * mix(vec3<f32>(1.0, 0.62, 0.38), vec3<f32>(1.0), smoothstep(0.0, 0.22, direction.y));
        layer = mix(layer, m_col, m_disc);
        cover = max(cover, m_disc);
    }
    let shown = s.moon_bright * (1.0 - s.overcast) * smoothstep(-0.01, 0.02, direction.y);
    // The dark side of a moon hides the stars and sky glow behind it a little.
    return color_in * (1.0 - cover * 0.5 * shown) + layer * shown;
}

// Inside any moon's disc (stars stay hidden there).
fn behind_moon(direction: vec3<f32>) -> bool {
    for (var i = 0; i < i32(s.moon_info.x); i++) {
        let m = s.moons[i * 3];
        if dot(direction, m.xyz) > cos(m.w) {
            return true;
        }
    }
    return false;
}

// Stars: one per cell of a cube-mapped grid, twinkling a little.
fn star_field(direction: vec3<f32>) -> vec3<f32> {
    let a = abs(direction);
    var uv: vec2<f32>;
    var face: f32;
    if a.x >= a.y && a.x >= a.z {
        uv = direction.yz / a.x;
        face = sign(direction.x);
    } else if a.y >= a.z {
        uv = direction.xz / a.y;
        face = 2.0 + sign(direction.y);
    } else {
        uv = direction.xy / a.z;
        face = 4.0 + sign(direction.z);
    }
    let cells = 220.0;
    let g = uv * cells;
    let cell = floor(g);
    let r = hash21(cell + vec2<f32>(face * 17.0, face * 31.0));
    if r < 0.93 {
        return vec3<f32>(0.0);
    }
    let pos = vec2<f32>(hash21(cell + 3.7), hash21(cell + 9.1)) * 0.7 + 0.15;
    let d = length(fract(g) - pos);
    let b = (r - 0.93) / 0.07;
    let tw = 0.75 + 0.25 * sin(s.time * (1.5 + 3.0 * b) + r * 60.0);
    let core = smoothstep(0.09, 0.0, d) * (0.3 + 1.7 * b * b) * tw;
    return vec3<f32>(1.0, 0.95 + 0.05 * b, 0.88 + 0.2 * b) * core;
}

fn physical_sky(direction: vec3<f32>) -> vec3<f32> {
    let pi = 3.141592653589793;
    let sun = normalize(s.sun_dir);
    // Preetham: per-frame constants, cheap enough per pixel.
    let total_rayleigh = vec3<f32>(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5);
    let mie_const = vec3<f32>(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14);
    let turbidity = 4.0;
    let rayleigh = 1.4;
    let mie_coefficient = 0.004;
    let mie_g = 0.8;
    let zenith_cos = clamp(sun.y, -1.0, 1.0);
    let sun_e = 1000.0 * max(0.0, 1.0 - exp(-((1.6110731556870734 - acos(zenith_cos)) / 1.5)));
    let beta_r = total_rayleigh * rayleigh;
    let beta_m = 0.434 * (0.2 * turbidity * 10e-18) * mie_const * mie_coefficient;

    let zenith_angle = acos(max(0.0, direction.y));
    let inv = 1.0 / (cos(zenith_angle) + 0.15 * pow(max(93.885 - zenith_angle * 180.0 / pi, 1e-3), -1.253));
    let fex = exp(-(beta_r * 8.4e3 * inv + beta_m * 1.25e3 * inv));
    let cos_theta = dot(direction, sun);
    let cc = cos_theta * 0.5 + 0.5;
    let r_phase = 0.05968310365946075 * (1.0 + cc * cc);
    let g2 = mie_g * mie_g;
    let m_phase = 0.07957747154594767 * (1.0 - g2) / pow(max(1.0 - 2.0 * mie_g * cos_theta + g2, 1e-6), 1.5);
    let scatter = (beta_r * r_phase + beta_m * m_phase) / (beta_r + beta_m);
    var lin = pow(sun_e * scatter * (1.0 - fex), vec3<f32>(1.5));
    lin *= mix(vec3<f32>(1.0), pow(sun_e * scatter * fex, vec3<f32>(0.5)), clamp(pow(1.0 - sun.y, 5.0), 0.0, 1.0));
    let l0 = vec3<f32>(0.1) * fex;

    // Sun disc, bright but keeping its colour.
    let disc = smoothstep(s.sun_cos - 0.00002, s.sun_cos + 0.00002, cos_theta);
    let disc_color = fex / max(max(fex.r, fex.g), max(fex.b, 1e-4)) * disc * (1.0 - s.overcast)
        * mix(9.0, 30.0, smoothstep(0.0, 0.35, sun.y)) * smoothstep(0.0, 0.05, sun_e / 1000.0);

    // Glare round a low sun, compressed above a soft knee.
    var sky_c = (lin + l0) * 0.04;
    let sky_l = max(max(sky_c.r, sky_c.g), sky_c.b);
    if sky_l > 1.0 {
        sky_c = sky_c * (1.0 + log(sky_l)) / sky_l;
    }
    var color = sky_c + disc_color + vec3<f32>(0.0, 0.0003, 0.00075);
    let cz = max(cos_theta, 0.0);
    color += min(sun_e * fex, vec3<f32>(80.0)) * (0.05 * pow(cz, 6000.0) + 0.003 * pow(cz, 400.0))
        * (1.0 - s.overcast) * smoothstep(-0.05, 0.02, direction.y);

    let twi = twilight(direction, sun);
    color += twi * (1.0 - s.overcast * 0.6);
    let night = night_sky(direction, sun);
    color += night * (1.0 - s.overcast * 0.5);

    // Stars, hidden behind the moons' discs.
    if s.stars > 0.0 && !behind_moon(direction) {
        color += star_field(direction) * s.stars * 0.6 * smoothstep(0.0, 0.08, direction.y);
    }
    if s.moon_bright > 0.0 {
        color = draw_moons(color, direction);
    }

    // Clouds: a layer `cloud.z` metres up, drifting.
    let cloud_scale = 0.0002;
    let cloud_speed = 0.00002;
    let cloud_elevation = 0.5;
    if direction.y > 0.0 && s.cloud_cover > 0.0 {
        let origin = select(vec3<f32>(0.0), s.cloud_origin, s.cloud.z > 0.0);
        let elevation = mix(1.0, 0.1, cloud_elevation);
        let flat_uv = direction.xz / max(direction.y, 0.02) + origin.xz / max(s.cloud.z, 1.0);
        let cloud_uv = flat_uv / elevation * cloud_scale + s.cloud.w * cloud_speed;
        let evolve = s.cloud.w * cloud_speed * 300.0;
        let cloud_noise = clamp(pfbm(cloud_uv * 1000.0, evolve) * 0.7 + 0.5, 0.0, 1.0);
        let region = pnoise(cloud_uv * 300.0) * 0.37 + 0.5;
        let cov = clamp(s.cloud_cover + (region - 0.5) * 0.6, 0.0, 1.0);
        let threshold = 1.0 - cov;
        let horizon_fade = smoothstep(0.0, 0.03 + 0.06 * cloud_elevation, direction.y);
        let mask = smoothstep(threshold, threshold + 0.3, cloud_noise) * horizon_fade;
        let day_factor = smoothstep(-0.08, 0.3, sun.y);
        let sun_color = sun_e * fex * 0.22 * 0.04;
        let sky_ambient = lin * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);
        let depth = max(0.0, cloud_noise - threshold);
        let beer = exp(-depth * 4.0);
        let powder = 1.0 - beer * beer;
        let shade = mix(0.45, 1.0, clamp(beer * powder * 2.6, 0.0, 1.0));
        let silver = clamp(0.51 / pow(max(1.49 - cos_theta * 1.4, 1e-3), 1.5), 0.0, 3.0);
        let edge = mask * (1.0 - mask) * 4.0;
        var cloud_color = sky_ambient + sun_color * shade;
        cloud_color += sun_color * silver * edge * 0.6;
        cloud_color *= max(day_factor, 0.03);
        cloud_color += twi * 0.6 + twilight(normalize(vec3<f32>(direction.x, 0.0, direction.z) + vec3<f32>(1e-5, 0.0, 0.0)), sun) * 0.5;
        cloud_color += night * 0.8;
        cloud_color *= 1.0 - s.cloud.y * 0.72;
        let alpha = (1.0 - exp(-depth * s.cloud.x * 12.0)) * horizon_fade;
        color -= l0 * 0.04 * alpha;
        let aerial = mix(color, cloud_color, fex);
        color = mix(color, aerial, alpha);
    }

    // Overcast greys the sky; fog veils the horizon (everything in thick fog).
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = mix(color, vec3<f32>(lum) * (0.72 - s.cloud.y * 0.4), s.overcast);
    let hz = 1.0 - smoothstep(-0.02, 0.4, direction.y);
    color = mix(color, s.fog_col, clamp(s.sky_fog * mix(hz, 1.0, s.sky_fog * s.sky_fog), 0.0, 1.0));
    // Below the horizon: the horizon colour, darkening, never black (no ground drawn there).
    let below = smoothstep(0.0, -0.3, direction.y);
    color = mix(color, s.fog_col * 0.6, below);
    return max(color, vec3<f32>(0.0)) * PHYSICAL_SKY_SCALE;
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    let view = vec3<f32>(
        v.ndc.x * s.tan_half_fov * s.aspect,
        v.ndc.y * s.tan_half_fov,
        1.0,
    );
    let dir = normalize(s.right * view.x + s.up * view.y + s.forward * view.z);
    if s.physical > 0.5 {
        let dither = (hash21(v.ndc * vec2<f32>(137.0, 251.0)) - 0.5) * 0.004;
        return vec4<f32>(physical_sky(dir) + dither, 1.0);
    }
    let elev = dir.y;
    let toward = max(dot(dir, s.sun_dir), 0.0);

    let flat = vec3<f32>(dir.x, 0.0, dir.z);
    let sun_flat = vec3<f32>(s.sun_dir.x, 0.0, s.sun_dir.z);
    // A normalized-vector dot can round just below -1 at the anti-sun
    // azimuth. Fractional pow of a negative value is NaN.
    let az = clamp(0.5 + 0.5 * dot(
        normalize(flat + vec3<f32>(1e-5, 0.0, 0.0)),
        normalize(sun_flat + vec3<f32>(1e-5, 0.0, 0.0)),
    ), 0.0, 1.0);
    let band = pow(clamp(1.0 - abs(elev), 0.0, 1.0), 2.2);
    let cool_h = mix(s.horizon, s.zenith, 0.22);
    let warm_h = mix(s.horizon, s.sun_color, 0.62);
    let hz = mix(cool_h, warm_h, pow(az, 1.35) * band);

    let zenith_t = pow(clamp(elev, 0.0, 1.0), 0.28 + s.curve * 2.4);
    var color = mix(hz, s.zenith, zenith_t);
    let ground_t = pow(clamp(-elev, 0.0, 1.0), 0.55);
    color = mix(color, s.ground, ground_t);

    let disc = smoothstep(s.sun_bloom, s.sun_cos, toward);
    let glow = pow(toward, 8.0) * 0.55;
    let horizon_glow = pow(toward, 4.0) * pow(1.0 - abs(elev), 2.2) * 0.42;
    color += s.sun_color * (disc * 1.8 + glow + horizon_glow);

    let drift = vec3<f32>(s.time * 0.003, 0.0, s.time * 0.001);
    let chunky = fbm3(dir * 2.2 + drift);
    let wispy = fbm3(dir * 6.4 + vec3<f32>(3.1, 8.2, 1.7) - drift * 0.35);
    let density = smoothstep(0.38, 0.74, chunky) * (0.45 + 0.55 * wispy);
    let cloud_mask = smoothstep(0.04, 0.18, elev) * (1.0 - smoothstep(0.62, 0.98, elev));
    let shade = mix(
        vec3<f32>(0.52, 0.56, 0.64),
        mix(vec3<f32>(0.97, 0.98, 0.99), s.sun_color, toward * 0.55),
        0.28 + 0.72 * toward,
    );
    color = mix(color, shade, density * cloud_mask * 0.48);

    let dither = (hash21(v.ndc * vec2<f32>(137.0, 251.0)) - 0.5) * 0.004;
    color += dither;
    return vec4<f32>(color, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::SHADER;

    #[test]
    fn gradient_normalization_never_divides_by_zero() {
        assert!(SHADER.contains("g / max(len, 1e-5)"));
        assert!(!SHADER.contains("g / len"));
    }

    #[test]
    fn anti_sun_azimuth_is_clamped_before_fractional_power() {
        assert!(SHADER.contains("let az = clamp(0.5 + 0.5 * dot("));
        assert!(SHADER.contains("), 0.0, 1.0);"));
        assert!(SHADER.contains("pow(az, 1.35)"));
    }
}
pub fn create_sky_pipelines(device: &wgpu::Device, format: wgpu::TextureFormat) -> SkyPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let empty = SkyUniforms {
        right: [1.0, 0.0, 0.0],
        tan_half_fov: 0.5,
        up: [0.0, 1.0, 0.0],
        aspect: 1.0,
        forward: [0.0, 0.0, 1.0],
        time: 0.0,
        zenith: [0.2, 0.4, 0.8],
        curve: 0.2,
        horizon: [0.7, 0.75, 0.8],
        sun_cos: 0.999,
        ground: [0.4, 0.42, 0.44],
        sun_bloom: 0.97,
        sun_dir: [0.4, 0.8, 0.3],
        physical: 0.0,
        sun_color: [1.0, 0.95, 0.85],
        overcast: 0.0,
        moon_dir: [0.0, -1.0, 0.0],
        moon_bright: 0.0,
        moon_sun: [0.0, 1.0, 0.0],
        stars: 0.0,
        fog_col: [0.0; 3],
        sky_fog: 0.0,
        cloud_origin: [0.0; 3],
        cloud_cover: 0.0,
        cloud: [0.0; 4],
        moons: [[0.0; 4]; 3 * crate::sky::MAX_MOONS],
        moon_info: [0.0; 4],
    };

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sky-bind-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let uniform_bufs = std::array::from_fn(|slot| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("sky-uniforms-{slot}")),
            contents: bytemuck::bytes_of(&empty),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    });
    let bind_groups = std::array::from_fn(|slot| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("sky-bind-{slot}")),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_bufs[slot].as_entire_binding(),
            }],
        })
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky-pipeline-layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let make = |label: &str, depth_stencil: wgpu::DepthStencilState| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: super::stencil::scene_multisample(false),
            multiview: None,
            cache: None,
        })
    };

    let pipeline = make("sky-pipeline", super::stencil::sky_depth_stencil_unmasked());
    let pipeline_portal = make(
        "sky-pipeline-portal",
        super::stencil::sky_depth_stencil_masked(),
    );

    SkyPipelines {
        pipeline,
        pipeline_portal,
        uniform_bufs,
        bind_groups,
    }
}
