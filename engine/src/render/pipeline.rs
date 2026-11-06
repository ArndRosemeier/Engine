use crate::mesh::{InstanceRaw, Vertex};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub ambient: f32,
    pub light_color: [f32; 3],
    pub _pad: f32,
    pub eye: [f32; 3],
    /// Seconds since start, for materials that animate.
    pub time: f32,
    pub haze_color: [f32; 3],
    /// Reciprocal metres; zero switches the haze off.
    pub haze_density: f32,
    /// Scale height of the air: every this many metres it thins by `1/e`.
    pub haze_height_m: f32,
    /// Altitude the air starts thinning from.
    pub haze_base_y: f32,
    /// Metres at which the torch contribution reaches zero.
    pub torch_radius_m: f32,
    pub torch_curve: f32,
    /// Viewer-carried point light (lantern / headlamp); rgb, alpha unused.
    /// Radius 0 switches it off.
    pub torch_color: [f32; 4],
    pub camera_right: [f32; 3],
    /// Strength of screen-space AO for this view; 0 where it does not apply.
    pub ao_strength: f32,
    pub camera_up: [f32; 3],
    /// 1 where the GI probes describe this view, 0 for views through portals.
    pub gi_strength: f32,
    /// Sky light from straight above, as a tint whose sky average is 1
    /// (the brightness itself is `ambient`).
    pub sky_up: [f32; 3],
    /// Scales sky reflections on glossy surfaces; 1 is physically balanced.
    pub env_strength: f32,
    pub sky_horizon: [f32; 3],
    pub _horizon_pad: f32,
    /// Light bounced up from the ground, same scale as `sky_up`.
    pub sky_down: [f32; 3],
    pub _down_pad: f32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 240);

impl Uniforms {
    pub fn empty() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_dir: [0.0, 1.0, 0.0],
            ambient: 0.2,
            light_color: [1.0, 1.0, 1.0],
            _pad: 0.0,
            eye: [0.0, 0.0, 0.0],
            time: 0.0,
            haze_color: [1.0, 1.0, 1.0],
            haze_density: 0.0,
            haze_height_m: 1.0,
            haze_base_y: 0.0,
            torch_radius_m: 0.0,
            torch_curve: 2.0,
            torch_color: [0.0, 0.0, 0.0, 0.0],
            camera_right: [1.0, 0.0, 0.0],
            ao_strength: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            gi_strength: 0.0,
            sky_up: [1.0, 1.0, 1.0],
            env_strength: 1.0,
            sky_horizon: [1.0, 1.0, 1.0],
            _horizon_pad: 0.0,
            sky_down: [1.0, 1.0, 1.0],
            _down_pad: 0.0,
        }
    }
}

/// Declarations every surface shader shares: the frame uniforms and the air
/// between the eye and what it is looking at.
///
/// One copy, because four shaders drifting apart on the layout of a single
/// uniform buffer is a class of bug that only shows up as garbage on screen.
pub const SCENE_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _pad: f32,
    eye: vec3<f32>,
    time: f32,
    haze_color: vec3<f32>,
    haze_density: f32,
    haze_height_m: f32,
    haze_base_y: f32,
    torch_radius_m: f32,
    torch_curve: f32,
    torch_color: vec4<f32>,
    camera_right: vec3<f32>,
    ao_strength: f32,
    camera_up: vec3<f32>,
    gi_strength: f32,
    sky_up: vec3<f32>,
    env_strength: f32,
    sky_horizon: vec3<f32>,
    _horizon_pad: f32,
    sky_down: vec3<f32>,
    _down_pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Sun plus sky lighting for a surface, with the frame's light and sky.
// `sun_diffuse` is the caller's diffuse shape for the sun (each surface keeps
// its own wrap), `vis` the sun shadow term.
fn shade_surface(
    base: vec3<f32>,
    n: vec3<f32>,
    world_p: vec3<f32>,
    roughness: f32,
    metallic: f32,
    vis: f32,
    sun_diffuse: f32,
) -> vec3<f32> {
    return shade_surface_occluded(base, n, world_p, roughness, metallic, vis, sun_diffuse, 1.0);
}

// `shade_surface` with baked ambient occlusion (an occlusion map), which
// dims sky and probe light but not the sun.
fn shade_surface_occluded(
    base: vec3<f32>,
    n: vec3<f32>,
    world_p: vec3<f32>,
    roughness: f32,
    metallic: f32,
    vis: f32,
    sun_diffuse: f32,
    occlusion: f32,
) -> vec3<f32> {
    var light: SceneLight;
    light.dir = normalize(u.light_dir);
    light.color = u.light_color;
    light.ambient = u.ambient;
    light.sky_up = u.sky_up;
    light.sky_horizon = u.sky_horizon;
    light.sky_down = u.sky_down;
    light.env_strength = u.env_strength;
    light.ao = screen_ao(u.view_proj * vec4<f32>(world_p, 1.0), u.ao_strength) * occlusion;
    light.gi = gi_irradiance(world_p, n);
    light.gi.a *= u.gi_strength;
    let v = normalize(u.eye - world_p);
    return pbr_shade(base, n, v, light, roughness, metallic, vis, sun_diffuse);
}

// Light from the viewer's own lantern. A point light at the eye with a
// distance falloff that stays gentle near the holder and dies at `reach`:
//   1 - (d/reach)^2  clamped ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â quadratic near, linear tail, zero past reach.
// The 1/(1+d*d) term keeps a hand's-breadth wall from blowing out.
fn torch_light(world_p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if u.torch_radius_m <= 0.0 {
        return vec3<f32>(0.0);
    }
    let to_p = world_p - u.eye;
    let d = length(to_p);
    let reach = max(u.torch_radius_m, 0.5);
    let fall = pow(clamp(1.0 - d / reach, 0.0, 1.0), max(u.torch_curve, 0.05));
    // Surfaces facing away get only the wrap spill, never a hard black edge.
    let ndl = max(dot(n, -to_p / max(d, 1e-4)), 0.0);
    let wrap = ndl * 0.7 + 0.3;
    return u.torch_color.rgb * (2.2 * fall * wrap);
}

// Fade a surface into the sky by how much air the view ray crossed.
//
// Without this the ground simply ends at the last chunk. Air thins with height,
// so the amount crossed is the integral of exp(-(y - base) / H) along the ray,
// which has a closed form: a summit looks out over tens of kilometres while the
// valley below it is milk within five. Taking the density at the midpoint
// instead ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â the obvious shortcut ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â all but switches the haze off as soon as the
// eye climbs a mountain, and the view from up there is the one that matters.
fn haze(color: vec3<f32>, world_p: vec3<f32>) -> vec3<f32> {
    if u.haze_density <= 0.0 {
        return color;
    }
    let d = distance(u.eye, world_p);
    let h = max(u.haze_height_m, 1.0);
    let a0 = exp(-max(u.eye.y - u.haze_base_y, 0.0) / h);
    let a1 = exp(-max(world_p.y - u.haze_base_y, 0.0) / h);
    let rise = world_p.y - u.eye.y;
    var air = d * a0;
    if abs(rise) > 1.0 {
        air = d * h * (a0 - a1) / rise;
    }
    let optical = air * u.haze_density;
    let f = 1.0 - exp(-optical * optical);
    return mix(color, u.haze_color, clamp(f, 0.0, 1.0));
}
"#;

/// Physically based sun and sky lighting, independent of any uniform layout.
///
/// Sun: GGX specular with Smith visibility and Schlick Fresnel, next to the
/// caller's diffuse shape. Sky: a three-colour dome (zenith, horizon, ground
/// bounce) for diffuse light and, sharpened by roughness, for reflections, so
/// shaded sides pick up blue from above and warm bounce from below and glossy
/// or metal surfaces reflect the sky instead of going black.
pub const LIGHTING_WGSL: &str = r#"
const PI: f32 = 3.14159265;

struct SceneLight {
    dir: vec3<f32>,
    color: vec3<f32>,
    ambient: f32,
    sky_up: vec3<f32>,
    sky_horizon: vec3<f32>,
    sky_down: vec3<f32>,
    env_strength: f32,
    // Screen-space ambient occlusion, 1 = open sky. Sky light only.
    ao: f32,
    // Probe light (rgb, same units as sky_irradiance * ambient) and how much
    // of it to use (a); see gi_irradiance.
    gi: vec4<f32>,
};

// Cosine-weighted sky light arriving at a surface facing `n`.
fn sky_irradiance(n: vec3<f32>, light: SceneLight) -> vec3<f32> {
    let t = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let s = 1.0 - t;
    return light.sky_down * (s * s) + light.sky_horizon * (2.0 * s * t) + light.sky_up * (t * t);
}

// Sky seen along `dir`, blurred toward the irradiance as roughness grows.
fn sky_radiance(dir: vec3<f32>, roughness: f32, light: SceneLight) -> vec3<f32> {
    let y = clamp(dir.y, -1.0, 1.0);
    var sharp = mix(light.sky_horizon, light.sky_up, sqrt(max(y, 0.0)));
    if y < 0.0 {
        sharp = mix(light.sky_horizon, light.sky_down, sqrt(-y));
    }
    return mix(sharp, sky_irradiance(dir, light), roughness * roughness);
}

fn pbr_shade(
    base: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    light: SceneLight,
    roughness_in: f32,
    metallic_in: f32,
    vis: f32,
    sun_diffuse: f32,
) -> vec3<f32> {
    let roughness = clamp(roughness_in, 0.045, 1.0);
    let metallic = clamp(metallic_in, 0.0, 1.0);
    let diffuse_color = base * (1.0 - metallic);
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let nv = max(dot(n, v), 1e-4);
    let sun = light.color * vis * (1.0 - light.ambient);

    // Sun specular. Radiance from the sun's irradiance E = PI * light.color,
    // matching the unnormalised diffuse term below.
    var spec = vec3<f32>(0.0);
    let nl = dot(n, light.dir);
    if nl > 0.0 {
        let h = normalize(light.dir + v);
        let nh = max(dot(n, h), 0.0);
        let lh = max(dot(light.dir, h), 0.0);
        let a = roughness * roughness;
        let a2 = a * a;
        let d_den = nh * nh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * d_den * d_den);
        let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        let g = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
        let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - lh, 5.0);
        spec = d * g * f / (4.0 * nv) * PI;
    }
    let direct = diffuse_color * sun_diffuse * sun + spec * sun;

    // Sky light. Pre-integrated GGX reflectance (Karis' analytic fit of the
    // split-sum lookup), so rough surfaces do not pick up a grazing sheen.
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let rr = roughness * c0 + c1;
    let a004 = min(rr.x * rr.x, exp2(-9.28 * nv)) * rr.x + rr.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + rr.zw;
    let fr = f0 * ab.x + vec3<f32>(ab.y);
    let r = reflect(-v, n);
    // Open-sky light, replaced by the probes' where they cover the point:
    // they know about walls, roofs and caves, and carry light bounced off
    // nearby surfaces. Reflections dim by the same amount the sky is hidden.
    let open_sky = sky_irradiance(n, light) * light.ambient;
    var sky_e = open_sky;
    var reflect_scale = light.ambient;
    if light.gi.a > 0.0 {
        sky_e = mix(open_sky, light.gi.rgb, light.gi.a);
        let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
        let seen = clamp(dot(sky_e, lum) / max(dot(open_sky, lum), 1e-4), 0.0, 1.0);
        reflect_scale = light.ambient * seen;
    }
    let env_spec = fr * sky_radiance(r, roughness, light) * light.env_strength * reflect_scale;
    let env_diffuse = diffuse_color * sky_e * (vec3<f32>(1.0) - fr);
    return direct + (env_diffuse + env_spec) * light.ao;
}
"#;

pub fn scene_shader_prefix() -> String {
    format!(
        "{}{}{}{}{}",
        LIGHTING_WGSL,
        SCENE_WGSL,
        super::shadow::SHADOW_UNIFORMS_WGSL,
        super::shadow::SCENE_SHADOW_WGSL,
        super::shadow::SHADOW_EVAL_WGSL
    )
}

const SHADER: &str = r#"
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
    @location(10) tint: vec4<f32>,
    @location(4) surface: vec4<f32>,
    @location(5) surface2: vec4<f32>,
    @location(11) surface3: vec4<f32>,
    @location(12) surface4: vec4<f32>,
    @location(13) surface5: vec4<f32>,
};

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) surface: vec4<f32>,
    @location(5) surface2: vec4<f32>,
    @location(11) surface3: vec4<f32>,
    @location(12) surface4: vec4<f32>,
    @location(13) surface5: vec4<f32>,
};

fn hash31(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.1031);
    let r = q + dot(q, q.yzx + 33.33);
    return fract((r.x + r.y) * r.z);
}

fn value3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

fn fbm3(p: vec3<f32>, gain: f32) -> f32 {
    var q = p;
    var amplitude = 0.5;
    var total = 0.0;
    for (var octave = 0; octave < 4; octave++) {
        total += value3(q) * amplitude;
        q = q * 2.03 + vec3<f32>(17.1, 9.4, 3.7);
        amplitude *= gain;
    }
    return total / 0.5;
}

fn ridged3(p: vec3<f32>, gain: f32) -> f32 {
    let ridge = 1.0 - abs(fbm3(p, gain) * 2.0 - 1.0);
    return ridge * ridge;
}

// Signed, energy-weighted detail in the spirit of Musgrave fBm.
fn musgrave3(p: vec3<f32>, gain: f32, lacunarity: f32) -> f32 {
    var q = p;
    var amplitude = 1.0;
    var total = 0.0;
    var weight = 0.0;
    for (var octave = 0; octave < 5; octave++) {
        total += (value3(q) * 2.0 - 1.0) * amplitude;
        weight += amplitude;
        amplitude *= gain;
        q = q * lacunarity + vec3<f32>(7.1, 13.7, 3.9);
    }
    return total / max(weight, 0.001);
}

fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * p.x + s * p.y, -s * p.x + c * p.y);
}

fn sand_gradient(x: f32, offset: f32) -> f32 {
    let repeated = abs(fract(x / 6.2831855 + offset - 0.25) - 0.5) * 2.0;
    let peaked = clamp(repeated * repeated * (-1.0 + 2.0 * repeated), 0.0, 1.0);
    let smoothed = repeated * repeated * (3.0 - 2.0 * repeated);
    return mix(smoothed, peaked, 0.15);
}

fn sand_layer(p: vec2<f32>, seed: f32) -> f32 {
    var q = rotate2(p, 3.1415927 / 18.0);
    q.y += (fbm3(vec3<f32>(q * 18.0, seed), 0.5) - 0.5) * 0.05;
    let first = sand_gradient(q.y * 80.0, 0.0);
    q = rotate2(p, -3.1415927 / 20.0);
    q.y += (fbm3(vec3<f32>(q * 12.0, seed + 7.0), 0.5) - 0.5) * 0.05;
    let second = sand_gradient(q.y * 80.0, 0.5);
    q = rotate2(p, 3.1415927 / 4.0);
    let blend = dot(sin(q * 12.0 - cos(q.yx * 12.0)), vec2<f32>(0.25)) + 0.5;
    return 1.0 - (1.0 - first * (1.0 - blend)) * (1.0 - second * blend);
}


fn dirt_profile(p: vec3<f32>, seed: f32, gain: f32) -> vec3<f32> {
    let broad = fbm3(p * 0.24 + vec3<f32>(seed, 3.0, 11.0), gain);
    let clumps = ridged3(p * 0.9 + vec3<f32>(5.0, seed, 17.0), gain);
    let grit = musgrave3(p * 5.5 + vec3<f32>(seed * 0.7, 23.0, 2.0), 0.54, 2.35);
    let organic = smoothstep(0.48, 0.73, broad + clumps * 0.22);
    return vec3<f32>(organic, clumps, grit);
}

fn grass_profile(p: vec3<f32>, seed: f32, gain: f32) -> vec3<f32> {
    let cell = floor(p.xz * 1.8);
    let jitter = vec2<f32>(
        hash31(vec3<f32>(cell, seed)),
        hash31(vec3<f32>(cell + vec2<f32>(31.0, 17.0), seed + 7.0)));
    let local = fract(p.xz * 1.8) - 0.5 - (jitter - vec2<f32>(0.5)) * 0.22;
    let blade_axis = normalize(vec2<f32>(
        hash31(vec3<f32>(cell + 11.0, seed + 13.0)),
        hash31(vec3<f32>(cell + 23.0, seed + 29.0))) * 2.0 - 1.0);
    let along = dot(local, blade_axis);
    let across = abs(dot(local, vec2<f32>(-blade_axis.y, blade_axis.x)));
    let blade = smoothstep(0.34, 0.0, across) * smoothstep(0.46, 0.0, abs(along));
    let patch_noise = fbm3(p * 0.32 + vec3<f32>(seed, 41.0, 9.0), gain);
    let fine = musgrave3(p * 6.0 + vec3<f32>(seed * 0.4, 7.0, 19.0), 0.52, 2.25);
    return vec3<f32>(blade, patch_noise, fine);
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.position, 1.0);
    var out: VsOut;
    out.clip = u.view_proj * world;
    // Assume uniform scale for normals (friendly default).
    out.world_n = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    out.color = v.color * v.tint;
    out.world_p = world.xyz;
    out.uv = v.uv;
    out.surface = v.surface;
    out.surface2 = v.surface2;
    out.surface3 = v.surface3;
    out.surface4 = v.surface4;
    out.surface5 = v.surface5;
    return out;
}

// Depth prepass: only texels the shaded pass will cover on every sample.
@fragment
fn fs_depth(in: VsOut) {
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let foliage = in.surface3.w > 5.5 && in.surface3.w < 7.5;
    if foliage || in.color.a * texel.a < 0.999 {
        discard;
    }
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Most world props use the default material. Keep them on the cheap,
    // conventional path; authored wood/dirt/grass/sand/snow/cave profiles
    // continue through the full procedural material evaluation below.
    if in.surface3.w < 0.001 {
        let geo_n = normalize(in.world_n);
        // Normal, occlusion-roughness-metallic and emissive maps; neutral
        // 1x1 texels for meshes without them.
        let n = mapped_normal(geo_n, in.world_p, in.uv);
        let orm = material_orm(in.uv);
        let l = normalize(u.light_dir);
        let ndl = max(dot(n, l), 0.0);
        let wrap = ndl * 0.65 + 0.35;
        let base = in.color * textureSample(albedo_tex, albedo_sampler, in.uv);
        let vis = sun_visibility(in.world_p, geo_n, u.eye);
        let lit = shade_surface_occluded(base.rgb, n, in.world_p, in.surface.x * orm.y,
                in.surface.y * orm.z, vis, wrap * wrap, orm.x)
            + base.rgb * torch_light(in.world_p, n)
            + base.rgb * in.surface.z * 3.0
            + material_emission(in.uv);
        return vec4<f32>(haze(lit, in.world_p), base.a);
    }

    let seed = in.surface2.y;
    let leaf_tex = textureSample(albedo_tex, albedo_sampler, in.uv);    // Imported cutouts own the visible silhouette for SpeedTree-style foliage.
    if leaf_tex.a < 0.08 {
        discard;
    }
    // Foliage cards stay on the opaque path. Each card is a compact proxy for
    // many leaves: one deterministic leaf is evaluated per UV cell, so the
    // mesh can remain cheap without presenting large rectangular panels.
    let is_broadleaf = in.surface3.w > 5.5 && in.surface3.w < 6.5;
    let is_needled = in.surface3.w > 6.5 && in.surface3.w < 7.5;
    if is_broadleaf || is_needled {
        // A deliberately overlapping field: cells are only a sampling lattice,
        // not visible tiles. Warped coordinates and generous silhouettes hide
        // the lattice while keeping the fragment cost constant.
        let grid = select(vec2<f32>(7.0, 6.0), vec2<f32>(6.0, 9.0), is_broadleaf);
        let leaf_uv = in.uv * grid;
        let cell = floor(leaf_uv + vec2<f32>(0.37, 0.19));
        let cell_hash = hash31(vec3<f32>(cell, seed));
        let cell_hash2 = hash31(vec3<f32>(cell + vec2<f32>(19.0, 7.0), seed + 13.0));
        let local = fract(leaf_uv + vec2<f32>(0.37, 0.19)) - vec2<f32>(0.5);
        let offset = (vec2<f32>(cell_hash, cell_hash2) - vec2<f32>(0.5)) * 0.34;
        let angle = (cell_hash * 2.0 - 1.0) * 1.7;
        let leaf_local = rotate2(local - offset, angle);
        var silhouette = 0.0;
        if is_broadleaf {
            let length = 0.48 + cell_hash2 * 0.14;
            let width = 0.25 + cell_hash * 0.10;
            let taper = max(1.0 - abs(leaf_local.y) / max(length, 0.01), 0.0);
            silhouette = taper * taper - abs(leaf_local.x) / max(width, 0.01);
        } else {
            let length = 0.52 + cell_hash2 * 0.16;
            let width = 0.085 + cell_hash * 0.045;
            silhouette = 1.0 - abs(leaf_local.x) / width - abs(leaf_local.y) / length;
        }
        // Smooth the analytic cutout over a pixel footprint. Hard, high-frequency
        // discard edges were the source of the close-range shimmer/noise.
        let edge = max(fwidth(silhouette), 0.015);
        if silhouette <= -edge {
            discard;
        }
        let vein = select(0.0, exp(-abs(leaf_local.x) * 24.0) * 0.10, is_broadleaf);
        let base_leaf = in.color * leaf_tex;
        let coverage = smoothstep(-edge, edge, silhouette);
        let variation = 0.90 + cell_hash * 0.15 + cell_hash2 * 0.07;
        let leaf_tone = select(vec3<f32>(0.22, 0.38, 0.15), vec3<f32>(0.40, 0.58, 0.20), is_broadleaf);
        let leaf_base = base_leaf.rgb * leaf_tone * variation + base_leaf.rgb * vein;
        let n = normalize(in.world_n);
        let l = normalize(u.light_dir);
        let ndl = dot(n, l);
        let front = max(ndl, 0.0);
        let transmission = max(-ndl, 0.0) * select(0.10, 0.24, is_broadleaf);
        let wrap = front * 0.55 + 0.45;
        let vis = sun_visibility(in.world_p, n, u.eye);
        var sky: SceneLight;
        sky.sky_up = u.sky_up;
        sky.sky_horizon = u.sky_horizon;
        sky.sky_down = u.sky_down;
        // Cards stand in for leaves facing every way: light them with the
        // sky's average rather than the card's own normal.
        let sky_fill = (sky_irradiance(n, sky) + sky_irradiance(vec3<f32>(0.0, 1.0, 0.0), sky)) * 0.5;
        let light = u.ambient * sky_fill + wrap * wrap * (1.0 - u.ambient) * u.light_color * vis + transmission;
        return vec4<f32>(haze(leaf_base * light, in.world_p), coverage);
    }

    // Default-material meshes have no authored orientation and carry a zero
    // vector. Never normalize that sentinel: NaNs here contaminate every
    // procedural sample and turn otherwise valid houses/props into black.
    let orientation_raw = in.surface3.xyz;
    let has_orientation = dot(orientation_raw, orientation_raw) > 0.001;
    let axis = normalize(select(vec3<f32>(0.0, 1.0, 0.0), orientation_raw, has_orientation));
    let orientation_use = select(1.0, abs(axis.y) + abs(axis.x) * 0.5 + abs(axis.z) * 0.5, has_orientation);
    let oriented_p = in.world_p + axis * dot(in.world_p, axis) * (orientation_use - 1.0);
    let warped = oriented_p * in.surface.w + vec3<f32>(seed * 1.37, seed * 0.71, seed * 2.11);
    let warp = vec3<f32>(
        fbm3(warped * 0.17 + vec3<f32>(11.0, 3.0, 7.0), in.surface2.w),
        fbm3(warped * 0.17 + vec3<f32>(2.0, 17.0, 5.0), in.surface2.w),
        fbm3(warped * 0.17 + vec3<f32>(7.0, 5.0, 19.0), in.surface2.w));
    let sample_p = warped + (warp - vec3<f32>(0.5)) * in.surface2.z * 3.0;
    let ridge = ridged3(sample_p * 0.65, in.surface2.w);
    let material_dx = dpdx(oriented_p * in.surface.w);
    let material_dy = dpdy(oriented_p * in.surface.w);
    let footprint = max(length(material_dx), length(material_dy));
    let lod = log2(max(footprint, 0.0001));
    let fine_visibility = 1.0 - smoothstep(-4.0, -0.35, lod);
    let medium_visibility = 1.0 - smoothstep(-1.4, 1.8, lod);

    // Evaluate only the profile this surface uses. Profiles are per material,
    // so neighbouring pixels take the same branch and the GPU skips the rest.
    // 1 wood, 2 sand, 3 dirt, 4 grass, 5 snow (grass base under a snow cover).
    let profile = i32(round(in.surface3.w));
    var material_grain = 0.0;
    var material_strata = 0.0;
    var fine_detail = 0.0;
    var wood_tone = 1.0;
    var fine_color = 1.0;
    var wood_fine = 0.0;
    switch profile {
        case 1: {
            // A separate, much finer directional fiber field. Narrow valleys
            // darken the wood and also perturb the normal, so grain remains
            // readable at distance.
            let axis_coord = dot(in.world_p, axis);
            let cross_coord = in.world_p - axis * axis_coord;
            let wood_bands = 0.5 + 0.5 * sin(axis_coord * in.surface.w * 2.6
                + fbm3(cross_coord * 0.75 + vec3<f32>(seed, 4.0, 9.0), in.surface2.w) * 3.8);
            let wood_grain = pow(clamp(wood_bands, 0.0, 1.0), 1.7);
            let wood_fine_signed = musgrave3(
                axis * axis_coord * in.surface.w * 18.0
                    + cross_coord * vec3<f32>(7.0, 3.0, 11.0)
                    + vec3<f32>(seed * 2.3, seed * 0.37, seed * 1.71),
                0.52,
                2.5);
            wood_fine = smoothstep(-0.18, 0.42, wood_fine_signed);
            material_grain = wood_grain * 0.72 + wood_fine * 0.28;
            material_strata = wood_grain - 0.5;
            fine_detail = (wood_fine - 0.5) * fine_visibility;
            wood_tone = 0.56 + material_grain * 0.78;
            fine_color = 0.82 + wood_fine * 0.34;
        }
        case 2: {
            let sand_plane_a = select(in.world_p.yz, in.world_p.xy, abs(axis.z) > 0.5);
            let sand_plane = select(sand_plane_a, in.world_p.xz, abs(axis.y) > 0.5);
            let sand_base = sand_plane * in.surface.w * 0.22 + vec2<f32>(seed * 0.031, seed * 0.071);
            let sand_a = sand_layer(sand_base, seed);
            let sand_b = sand_layer(rotate2(sand_base, 0.19) * 1.27, seed + 19.0);
            let sand_pattern = mix(sand_a, sand_b, smoothstep(0.18, 0.82,
                fbm3(vec3<f32>(sand_base * 1.8, seed + 41.0), 0.5)));
            var sand_grit = 0.0;
            if fine_visibility > 0.0 {
                sand_grit = musgrave3(vec3<f32>(sand_plane * in.surface.w * 7.5, seed + 67.0), 0.52, 2.35);
            }
            material_grain = sand_pattern;
            material_strata = sand_pattern - 0.5;
            fine_detail = sand_grit * fine_visibility;
            wood_tone = 0.72 + material_grain * 0.55;
        }
        case 3: {
            let dirt = dirt_profile(oriented_p * 0.75, seed, in.surface2.w);
            material_grain = dirt.x * 0.62 + dirt.y * 0.38;
            material_strata = dirt.y - 0.5;
            fine_detail = dirt.z * fine_visibility;
            wood_tone = 0.72 + material_grain * 0.45;
            fine_color = 0.82 + fine_detail * 0.25;
        }
        default: {
            let grass = grass_profile(oriented_p * 0.62, seed, in.surface2.w);
            material_grain = grass.x * 0.52 + grass.y * 0.30 + grass.z * 0.18;
            material_strata = grass.y - 0.5;
            fine_detail = (grass.z - 0.5) * fine_visibility;
            wood_tone = 0.74 + material_grain * 0.48;
            fine_color = 0.78 + fine_detail * 0.38;
        }
    }
    let detail = (in.surface.x * 0.13 + 0.045) * fine_detail;
    let n = normalize(in.world_n + vec3<f32>(detail, detail * 0.35, -detail));
    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let layer = 1.0 + (material_strata * in.surface2.x * medium_visibility + (ridge - 0.5) * 0.55) * 0.32;
    // Tone and layering shade the colour only. Scaling alpha with them made
    // opaque surfaces partly transparent once alpha-to-coverage was enabled.
    let base = vec4<f32>(in.color.rgb * texel.rgb * layer * wood_tone * fine_color, in.color.a * texel.a);
    // Soft wrap lighting ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â enough contrast for smooth heightfields to read as
    // hills. Sky and sun share one budget, so raising ambient fills the shadows
    // instead of blowing out everything the sun already reaches.
    let wrap = ndl * 0.65 + 0.35;
    let vis = sun_visibility(in.world_p, n, u.eye);
    var lit = shade_surface(base.rgb, n, in.world_p, in.surface.x, in.surface.y, vis, wrap * wrap);
    lit += base.rgb * torch_light(in.world_p, n);
    let glint = smoothstep(0.92, 0.99, material_grain) * in.surface.z
        + smoothstep(0.72, 0.98, wood_fine) * select(0.0, 0.025, profile == 1);
    lit += vec3<f32>(1.0, 0.86, 0.62) * glint * 0.08;
    // Cave formations carry restrained mineral emission: pale calcite and
    // green/blue bioluminescent polish remain readable beyond the headlamp's
    // hot spot without turning the whole cave into a self-lit scene.
    let calcite = smoothstep(0.62, 0.86, min(in.color.r, min(in.color.g, in.color.b)));
    let bio = smoothstep(1.15, 1.75, in.color.g - in.color.r + in.color.b * 0.25);
    lit += vec3<f32>(1.0, 0.78, 0.52) * calcite * 0.055;
    lit += vec3<f32>(0.18, 0.95, 0.72) * bio * 0.12;
    // Soft fresnel rim for translucent surfaces (keep grazing alpha modest so
    // water stays see-through from typical third-person angles).
    var alpha = base.a;
    if alpha < 0.999 {
        let view = normalize(u.eye - in.world_p);
        let fresnel = pow(1.0 - max(dot(n, view), 0.0), 2.0);
        alpha = mix(alpha, min(alpha + 0.18, 0.55), fresnel * 0.65);
    } else {
        alpha = 1.0;
    }
    let coverage_dir = normalize(in.surface4.xyz);
    let facing = dot(n, coverage_dir);
    let directional = smoothstep(in.surface4.w - in.surface5.x,
        in.surface4.w + in.surface5.x, facing);
    // Normal-weighted triplanar breakup keeps deposits natural on arbitrary
    // rocks and avoids privileging the XZ plane on walls or overhangs. Skipped
    // entirely where no cover can land (most surfaces, and every downward face).
    var overlay_mix = 0.0;
    var snow_fine = 0.0;
    if in.surface5.y > 0.0 && directional > 0.0 {
        let tri_weights = abs(n) / max(dot(abs(n), vec3<f32>(1.0)), 0.001);
        let tri_noise = fbm3(in.world_p.yzx * 0.55 + vec3<f32>(seed, 37.0, 11.0), 0.5) * tri_weights.x
            + fbm3(in.world_p.zxy * 0.55 + vec3<f32>(seed + 17.0, 7.0, 19.0), 0.5) * tri_weights.y
            + fbm3(in.world_p.xyz * 0.55 + vec3<f32>(seed + 31.0, 13.0, 5.0), 0.5) * tri_weights.z;
        let breakup = smoothstep(0.30, 0.72, tri_noise);
        overlay_mix = clamp(directional * breakup * in.surface5.y, 0.0, 1.0);
        if overlay_mix > 0.0 {
            snow_fine = musgrave3(in.world_p * in.surface.w * 4.5 + vec3<f32>(seed, 61.0, 23.0), 0.52, 2.25);
        }
    }
    let overlay_color = vec3<f32>(0.78 + snow_fine * 0.12, 0.84 + snow_fine * 0.10, 0.94 + snow_fine * 0.06);
    let overlay_base = mix(base.rgb, overlay_color, overlay_mix);
    let overlay_lit = shade_surface(overlay_base, n, in.world_p, in.surface5.z, in.surface5.w, vis, wrap * wrap);
    let snow_bump = snow_fine * overlay_mix * select(0.0, 0.055, in.surface3.w > 4.5);
    let snow_sparkle = snow_bump * smoothstep(0.55, 0.95, dot(n, normalize(u.light_dir)))
        * vec3<f32>(0.32, 0.48, 0.72);
    let overlay_specular = mix(0.0, 0.06 * (1.0 - in.surface5.z), overlay_mix)
        + snow_bump * 0.22;
    let emission = base.rgb * in.surface.z * 3.0;
    return vec4<f32>(haze(mix(lit, overlay_lit, overlay_mix) + overlay_specular + snow_sparkle + emission, in.world_p), alpha);
}

// Voxel terrain: grass on flat ground, earth on moderate slopes, layered rock
// on steep faces, overhangs and cave walls. Everything is sampled in world
// space, so crater sides and tunnels never show stretched texture.
@fragment
fn fs_voxel(in: VsOut) -> @location(0) vec4<f32> {
    let n0 = normalize(in.world_n);
    let p = in.world_p;
    let seed = in.surface2.y;
    let broad = fbm3(p * 0.11 + vec3<f32>(seed, 5.0, 9.0), 0.5);
    let fine = fbm3(p * 1.7 + vec3<f32>(3.0, seed, 1.0), 0.5);
    let strata = fbm3(vec3<f32>(p.x * 0.04, p.y * 0.85, p.z * 0.04) + vec3<f32>(seed * 0.3), 0.55);
    let n = normalize(n0 + (vec3<f32>(fine, fbm3(p * 1.7 + 11.0, 0.5), fbm3(p * 1.7 + 23.0, 0.5)) - 0.5) * 0.18);

    let rock = in.color.rgb * (0.62 + 0.5 * strata) * (0.86 + 0.28 * fine);
    let earth = vec3<f32>(0.33, 0.25, 0.17) * (0.8 + 0.35 * broad) * (0.9 + 0.2 * fine);
    let grass = mix(vec3<f32>(0.20, 0.33, 0.10), vec3<f32>(0.36, 0.42, 0.16), broad)
        * (0.85 + 0.3 * fine);
    let up = n0.y + (broad - 0.5) * 0.3;
    let grass_w = smoothstep(0.72, 0.84, up);
    let earth_w = smoothstep(0.42, 0.62, up);
    // Surfaces opened up by digging (or built) show bare earth and rock.
    let bare = mix(rock, earth * 0.85, smoothstep(0.35, 0.8, up));
    let natural = mix(mix(rock, earth, earth_w), grass, grass_w);
    let base = mix(natural, bare, clamp(in.uv.y, 0.0, 1.0));
    let roughness = mix(in.surface.x, 0.95, grass_w * (1.0 - in.uv.y));

    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.65 + 0.35;
    let vis = sun_visibility(p, n0, u.eye);
    let lit = shade_surface(base, n, p, roughness, in.surface.y, vis, wrap * wrap)
        + base * torch_light(p, n);
    return vec4<f32>(haze(lit, p), 1.0);
}
"#;

pub const SCENE_UNIFORM_SLOTS: usize = 64;

pub struct SceneUniformSlots {
    pub buffers: [wgpu::Buffer; SCENE_UNIFORM_SLOTS],
    pub bind_groups: [wgpu::BindGroup; SCENE_UNIFORM_SLOTS],
}

impl SceneUniformSlots {
    pub fn get(&self, slot: usize) -> (&wgpu::Buffer, &wgpu::BindGroup) {
        (
            self.buffers.get(slot).unwrap_or_else(|| {
                panic!("scene uniform slot {slot} exceeds capacity {SCENE_UNIFORM_SLOTS}")
            }),
            self.bind_groups.get(slot).unwrap_or_else(|| {
                panic!("scene uniform slot {slot} exceeds capacity {SCENE_UNIFORM_SLOTS}")
            }),
        )
    }
}

const MESH_BUFFERS: [wgpu::VertexBufferLayout<'static>; 2] = [Vertex::LAYOUT, InstanceRaw::LAYOUT];

pub struct Pipelines {
    pub opaque: wgpu::RenderPipeline,
    /// Voxel terrain through the same lit shader (compact vertex layout).
    pub voxel_opaque: wgpu::RenderPipeline,
    pub voxel_depth_prepass: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub opaque_portal: wgpu::RenderPipeline,
    pub transparent_portal: wgpu::RenderPipeline,
    /// Depth-only pass over fully opaque texels (foliage cutouts stay out:
    /// their soft alpha-to-coverage edges must not hide what is behind them).
    pub depth_prepass: wgpu::RenderPipeline,
    pub scene_uniforms: SceneUniformSlots,
    pub bind_layout: wgpu::BindGroupLayout,
    /// Mesh material group: albedo, sampler, normal, ORM, emissive, factors.
    pub albedo_layout: wgpu::BindGroupLayout,
    /// Clamped sampler for plain baked albedos.
    pub albedo_sampler: wgpu::Sampler,
    /// Repeating anisotropic sampler for imported material sets.
    pub material_sampler: wgpu::Sampler,
    pub neutral_maps: super::material::NeutralMaps,
    pub white_albedo: wgpu::BindGroup,
}

impl Pipelines {
    pub fn scene_bind_group(&self, slot: usize) -> &wgpu::BindGroup {
        self.scene_uniforms.get(slot).1
    }

    /// Rebuild the scene bind groups after a shadow resource (the AO
    /// target) changed.
    pub fn rebuild_scene_bind_groups(
        &mut self,
        device: &wgpu::Device,
        shadow: &super::shadow::ShadowGpu,
    ) {
        self.scene_uniforms.bind_groups = std::array::from_fn(|slot| {
            create_scene_bind_group(
                device,
                &self.bind_layout,
                shadow,
                &self.scene_uniforms.buffers[slot],
                slot,
            )
        });
    }
}

fn create_scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shadow: &super::shadow::ShadowGpu,
    buffer: &wgpu::Buffer,
    slot: usize,
) -> wgpu::BindGroup {
    let entries = shadow.scene_bind_entries(buffer.as_entire_binding());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("uniform-bind-{slot}")),
        layout,
        entries: &entries,
    })
}

pub fn create_pipelines(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    shadow: &super::shadow::ShadowGpu,
) -> Pipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lit-shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "{}{SHADER}{}",
                scene_shader_prefix(),
                super::material::material_wgsl(1)
            )
            .into(),
        ),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform-layout"),
        entries: &super::shadow::ShadowGpu::scene_layout_entries(),
    });

    let scene_uniforms = {
        let buffers = std::array::from_fn(|slot| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("uniforms-{slot}")),
                contents: bytemuck::bytes_of(&Uniforms::empty()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let bind_groups = std::array::from_fn(|slot| {
            create_scene_bind_group(device, &bind_layout, shadow, &buffers[slot], slot)
        });
        SceneUniformSlots {
            buffers,
            bind_groups,
        }
    };

    let albedo_layout = super::material::material_layout(device);
    let albedo_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mesh-albedo-sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let material_sampler = super::material::material_sampler(device);
    let neutral_maps = super::material::NeutralMaps::new(device, queue);
    let white_albedo = super::material::GpuMaterial::new(
        device,
        queue,
        &albedo_layout,
        &albedo_sampler,
        &neutral_maps,
        None,
        None,
    )
    .bind;

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout"),
        bind_group_layouts: &[&bind_layout, &albedo_layout],
        push_constant_ranges: &[],
    });

    let make = |label: &str,
                blend: wgpu::BlendState,
                depth_write: bool,
                cull: Option<wgpu::Face>,
                depth_stencil: wgpu::DepthStencilState,
                buffers: &[wgpu::VertexBufferLayout<'static>],
                fs_entry: &str| {
        // Opaque draws (the only ones writing depth) turn foliage alpha into coverage.
        let alpha_to_coverage = depth_write;
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                // See-through surfaces are not in the depth the AO came from.
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: if depth_write {
                        &[]
                    } else {
                        &[("SCENE_AO", 0.0)]
                    },
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: super::stencil::scene_multisample(alpha_to_coverage),
            multiview: None,
            cache: None,
        })
    };

    let opaque = make(
        "opaque-pipeline",
        wgpu::BlendState::REPLACE,
        true,
        Some(wgpu::Face::Back),
        super::stencil::scene_depth_stencil_unmasked_write(true),
        &MESH_BUFFERS,
        "fs_main",
    );
    let transparent = make(
        "transparent-pipeline",
        wgpu::BlendState::ALPHA_BLENDING,
        false,
        None, // water/glass readable from both sides
        super::stencil::scene_depth_stencil_unmasked_write(false),
        &MESH_BUFFERS,
        "fs_main",
    );
    let opaque_portal = make(
        "opaque-portal-pipeline",
        wgpu::BlendState::REPLACE,
        true,
        Some(wgpu::Face::Back),
        super::stencil::scene_depth_stencil_masked_write(true),
        &MESH_BUFFERS,
        "fs_main",
    );
    let transparent_portal = make(
        "transparent-portal-pipeline",
        wgpu::BlendState::ALPHA_BLENDING,
        false,
        None,
        super::stencil::scene_depth_stencil_masked_write(false),
        &MESH_BUFFERS,
        "fs_main",
    );

    let voxel_opaque = make(
        "voxel-opaque-pipeline",
        wgpu::BlendState::REPLACE,
        true,
        Some(wgpu::Face::Back),
        super::stencil::scene_depth_stencil_unmasked_write(true),
        &super::voxel_terrain::VOXEL_BUFFERS,
        "fs_voxel",
    );
    let depth_prepass_for = |label: &str, buffers: &[wgpu::VertexBufferLayout<'static>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_depth"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(super::stencil::scene_depth_stencil_unmasked_write(true)),
            multisample: super::stencil::scene_multisample(false),
            multiview: None,
            cache: None,
        })
    };
    let depth_prepass = depth_prepass_for("opaque-depth-prepass", &MESH_BUFFERS);
    let voxel_depth_prepass =
        depth_prepass_for("voxel-depth-prepass", &super::voxel_terrain::VOXEL_BUFFERS);

    Pipelines {
        opaque,
        voxel_opaque,
        voxel_depth_prepass,
        transparent,
        opaque_portal,
        transparent_portal,
        depth_prepass,
        scene_uniforms,
        bind_layout,
        albedo_layout,
        albedo_sampler,
        material_sampler,
        neutral_maps,
        white_albedo,
    }
}
