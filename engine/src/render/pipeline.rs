use crate::mesh::{InstanceRaw, Vertex};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub ambient: f32,
    pub light_color: [f32; 3],
    /// Wind strength for swaying vegetation: 0 calm, 1 breeze, 2 storm.
    pub wind: f32,
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
    /// Frame counter (mod 64) that shifts the LOD dither each frame, so
    /// temporal anti-aliasing averages it into a smooth fade.
    pub dither_frame: f32,
    /// Light bounced up from the ground, same scale as `sky_up`.
    pub sky_down: [f32; 3],
    /// 0 summer green, 0.5 early autumn, 1 full autumn colours.
    pub season: f32,
    /// Glow channels ([`crate::mesh::SurfaceMaterial::glow_channel`]):
    /// rgb is the channel's glow colour (r below 0: the surface's own
    /// colour), a its strength.
    pub glow: [[f32; 4]; crate::mesh::GLOW_CHANNELS],
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 240 + 16 * crate::mesh::GLOW_CHANNELS);

impl Uniforms {
    pub fn empty() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_dir: [0.0, 1.0, 0.0],
            ambient: 0.2,
            light_color: [1.0, 1.0, 1.0],
            wind: 0.0,
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
            dither_frame: 0.0,
            sky_down: [1.0, 1.0, 1.0],
            season: 0.0,
            glow: [[-1.0, 0.0, 0.0, 1.0]; crate::mesh::GLOW_CHANNELS],
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
    wind: f32,
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
    dither_frame: f32,
    sky_down: vec3<f32>,
    season: f32,
    glow: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Debug shading (World::set_debug_shading): flags ride in torch_color.w.
fn dbg(bit: u32) -> bool {
    return (u32(u.torch_color.w + 0.5) & bit) != 0u;
}

fn not_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u;
}

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
    base_in: vec3<f32>,
    n_in: vec3<f32>,
    world_p: vec3<f32>,
    roughness_in: f32,
    metallic_in: f32,
    vis: f32,
    sun_diffuse: f32,
    occlusion: f32,
) -> vec3<f32> {
    var ds = apply_decals(world_p, n_in, base_in, roughness_in, metallic_in, u.eye);
    if dbg(512u) {
        ds.base = base_in;
        ds.n = n_in;
        ds.roughness = roughness_in;
        ds.metallic = metallic_in;
        ds.emit = vec3<f32>(0.0);
    }
    let base = ds.base;
    let n = ds.n;
    let roughness = ds.roughness;
    let metallic = ds.metallic;
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
    if dbg(1024u) {
        light.gi = vec4<f32>(0.0);
    }
    let v = normalize(u.eye - world_p);
    let clip = u.view_proj * vec4<f32>(world_p, 1.0);
    let frag = clip.xy / max(clip.w, 1e-4) * vec2<f32>(960.0, 540.0);
    light.reflection = scene_reflection(world_p, n, v, roughness, base, metallic, frag);
    if dbg(128u) {
        light.reflection = vec4<f32>(0.0);
    }
    var lamps = lamp_light(base, n, v, world_p, roughness, metallic);
    if dbg(256u) {
        lamps = vec3<f32>(0.0);
    }
    return pbr_shade(base, n, v, light, roughness, metallic, vis, sun_diffuse)
        + lamps + ds.emit;
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
// The half vector between a light and the eye. When they point exactly
// opposite ways the sum is zero and normalize() gives NaN; then this is zero
// (no highlight) instead.
fn safe_half(l: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let s = l + v;
    return s * inverseSqrt(max(dot(s, s), 1e-12));
}
// A unit normal from one that may be zero (a model vertex without a normal,
// or two that cancel between vertices): normalize() of zero is NaN, which
// would spread from that pixel. Zero becomes straight up.
fn safe_normal(n: vec3<f32>) -> vec3<f32> {
    let d = dot(n, n);
    if !(d > 1e-12 && d < 1e30) { return vec3<f32>(0.0, 1.0, 0.0); }
    return n * inverseSqrt(d);
}
const PI: f32 = 3.14159265;
// Brightest a sun highlight gets (times the sun's colour); see pbr_shade.
const SUN_SPEC_MAX: f32 = 120.0;
// Brightest colour the half-float scene image is given.
const HDR_MAX: f32 = 60000.0;

// A colour fit to store in the scene image: NaN or infinite (tested on the
// bits) becomes black, and nothing is brighter than HDR_MAX, which a half
// float would store as infinity. Every pass that writes lit colour ends
// with this: reflections and probes read the image back, so one broken
// pixel would otherwise spread.
fn hdr_safe(c: vec3<f32>) -> vec3<f32> {
    let e = bitcast<vec3<u32>>(c) & vec3<u32>(0x7f800000u);
    if any(e == vec3<u32>(0x7f800000u)) {
        return vec3<f32>(0.0);
    }
    return clamp(c, vec3<f32>(0.0), vec3<f32>(HDR_MAX));
}

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
    // Reflected light from screen-space reflections and reflection probes
    // (rgb) and how much it replaces the sky's (a); see scene_reflection.
    reflection: vec4<f32>,
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
        let h = safe_half(light.dir, v);
        let nh = max(dot(n, h), 0.0);
        let lh = max(dot(light.dir, h), 0.0);
        let a = roughness * roughness;
        let a2 = a * a;
        let d_den = nh * nh * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * d_den * d_den);
        let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        let g = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
        let f = f0 + (vec3<f32>(1.0) - f0) * pow(max(1.0 - lh, 0.0), 5.0);
        // The sun is a disc, not a point, so its mirror image has a finite
        // brightness. Without the cap, chrome seen from above (normal, sun
        // and eye lined up) blew past half-float range: white glare
        // speckled with NaN black.
        spec = min(d * g * f / (4.0 * nv) * PI, vec3<f32>(SUN_SPEC_MAX));
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
    var env_rad = sky_radiance(r, roughness, light) * reflect_scale;
    if light.reflection.a > 0.0 {
        env_rad = mix(env_rad, light.reflection.rgb, light.reflection.a);
    }
    let env_spec = fr * env_rad * light.env_strength;
    let env_diffuse = diffuse_color * sky_e * (vec3<f32>(1.0) - fr);
    return direct + (env_diffuse + env_spec) * light.ao;
}
"#;

/// Vegetation wind (shared by the scene and shadow shaders): sways a vertex
/// in its instance's own space. `w` is the sway weight (0 root, 1 twig tip),
/// `flutter` 1 for leaves, `seed` a per-tuft random for the flutter phase.
/// Port of AssetGenerator's `vegWind`.
pub(crate) const VEG_WIND_WGSL: &str = r#"
fn veg_wind(p: vec3<f32>, n: vec3<f32>, w: f32, flutter: f32, seed: f32,
        model: mat4x4<f32>, t: f32, strength: f32) -> vec3<f32> {
    let org = model[3].xyz;
    let ph = dot(org.xz, vec2<f32>(0.137, 0.213));
    let dir_w = vec3<f32>(0.9216, 0.0, 0.3871);
    let sc = max(1e-3, length(model[0].xyz));
    // The world wind direction in the instance's space (any yaw, uniform scale).
    let dir = vec3<f32>(dot(model[0].xyz, dir_w), dot(model[1].xyz, dir_w), dot(model[2].xyz, dir_w)) / sc;
    let gust = 0.55 + 0.45 * sin(t * 0.37 + ph * 0.5) * sin(t * 0.23 + ph * 0.9 + 1.3);
    let s = strength * (0.35 + gust);
    let w2 = w * w;
    let sway = 0.55 + 0.45 * sin(t * 1.05 + ph);
    var off = dir * (s * w2 * 0.42 * sway);
    let bp = ph * 3.0 + dot(p, vec3<f32>(0.45, 0.31, 0.38));
    let osc = vec3<f32>(sin(t * 2.1 + bp), 0.35 * sin(t * 2.7 + bp * 1.3), sin(t * 1.8 + bp * 0.8 + 1.7));
    off += osc * (s * w2 * w * 0.13);
    off += n * (sin(t * 9.0 + seed * 40.0 + bp * 2.0) * flutter * s * w * 0.05);
    off.y -= dot(off.xz, off.xz) * 0.12;
    return p + off;
}
"#;

pub fn scene_shader_prefix() -> String {
    format!(
        "{}{}{}{}{}{}{}{}{}{}{}",
        super::shadow::element_dead_wgsl(0, 15),
        VEG_WIND_WGSL,
        crate::lod::LOD_DITHER_WGSL,
        LIGHTING_WGSL,
        SCENE_WGSL,
        super::shadow::SHADOW_UNIFORMS_WGSL,
        super::shadow::SCENE_SHADOW_WGSL,
        super::shadow::SHADOW_EVAL_WGSL,
        super::lights::LAMP_EVAL_WGSL,
        super::reflect::reflect_wgsl(),
        super::decals::decal_wgsl(0, 20, 17, false),
    )
}

/// What a custom material function gets and gives back (see
/// `crate::custom_material`). The part between the markers is replaced by
/// the game's own code in a custom material's shader.
pub(crate) const MATERIAL_HOOK_WGSL: &str = r#"
// Set to 1 in a custom material's pipelines.
override CUSTOM_MATERIAL: f32 = 0.0;

struct MaterialIn {
    world_pos: vec3<f32>,
    object_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    uv: vec2<f32>,
    time: f32,
    params: array<vec4<f32>, 4>,
};

struct MaterialOut {
    color: vec4<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
};
"#;

/// The stand-in when no custom material is set: changes nothing.
pub(crate) const MATERIAL_STUB_WGSL: &str = r#"
fn custom_params() -> array<vec4<f32>, 4> {
    return array<vec4<f32>, 4>();
}
fn custom_material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
    return base;
}
"#;

/// The lit mesh shader, with `custom` as the material code (the stub when
/// `None`). Custom code defines `fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut`.
pub(crate) fn lit_shader_source(custom: Option<&str>) -> String {
    let hook = match custom {
        None => MATERIAL_STUB_WGSL.to_owned(),
        Some(code) => format!(
            "struct CustomParams {{ p: array<vec4<f32>, 4> }}\n\
             @group(2) @binding(0) var<uniform> custom_material_params: CustomParams;\n\
             fn custom_params() -> array<vec4<f32>, 4> {{ return custom_material_params.p; }}\n\
             fn custom_material(m: MaterialIn, base: MaterialOut) -> MaterialOut {{ return material(m, base); }}\n\
             {CUSTOM_CODE_MARK}\n{code}\n"
        ),
    };
    format!(
        "{}{MATERIAL_HOOK_WGSL}{hook}{SHADER}{}",
        scene_shader_prefix(),
        super::material::material_wgsl(1)
    )
}

/// The line before a game's material code in [`lit_shader_source`]: error
/// lines are counted from here.
pub(crate) const CUSTOM_CODE_MARK: &str = "// ---- custom material code ----";

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
    @location(14) element: u32,
};

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) surface: vec4<f32>,
    @location(5) surface2: vec4<f32>,
    // Object-space position scaled to world size, and the object's x and y
    // axes in world space: triplanar texturing that moves with the mesh.
    @location(6) obj_p: vec3<f32>,
    @location(7) @interpolate(flat) obj_x: vec3<f32>,
    @location(8) @interpolate(flat) obj_y: vec3<f32>,
    @location(11) surface3: vec4<f32>,
    @location(12) surface4: vec4<f32>,
    @location(13) surface5: vec4<f32>,
    // Instance tint alpha: below 0 it is a LOD cross-fade code (crate::lod).
    @location(14) @interpolate(flat) lod_fade: f32,
    // The vertex's glow channel: colour (r < 0: the surface's own) and strength.
    @location(15) @interpolate(flat) glow: vec4<f32>,
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

// Three independent value noises at once (one hash per corner, not three).
fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

fn value3v(p: vec3<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash33(i);
    let n100 = hash33(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash33(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash33(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash33(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash33(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash33(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash33(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
        mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// fbm3-like noise that stops at octaves finer than the pixel (`foot`: pixel
// size in noise units). Skipped octaves add their mean, so the brightness
// matches; distant surfaces then cost one or two octaves instead of four.
fn fbm3_lod(p: vec3<f32>, gain: f32, foot: f32) -> f32 {
    var q = p;
    var amplitude = 0.5;
    var total = 0.0;
    var freq = 1.0;
    for (var octave = 0; octave < 4; octave++) {
        let keep = 1.0 - smoothstep(0.25, 0.5, freq * foot);
        if keep <= 0.0 {
            total += 0.5 * amplitude * (1.0 - pow(gain, f32(4 - octave))) / (1.0 - gain);
            break;
        }
        total += mix(0.5, value3(q), keep) * amplitude;
        q = q * 2.03 + vec3<f32>(17.1, 9.4, 3.7);
        amplitude *= gain;
        freq *= 2.03;
    }
    return total / 0.5;
}

fn fbm3v_lod(p: vec3<f32>, gain: f32, foot: f32) -> vec3<f32> {
    var q = p;
    var amplitude = 0.5;
    var total = vec3<f32>(0.0);
    var freq = 1.0;
    for (var octave = 0; octave < 4; octave++) {
        let keep = 1.0 - smoothstep(0.25, 0.5, freq * foot);
        if keep <= 0.0 {
            total += vec3<f32>(0.5 * amplitude * (1.0 - pow(gain, f32(4 - octave))) / (1.0 - gain));
            break;
        }
        total += mix(vec3<f32>(0.5), value3v(q), keep) * amplitude;
        q = q * 2.03 + vec3<f32>(17.1, 9.4, 3.7);
        amplitude *= gain;
        freq *= 2.03;
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
    var out: VsOut;
    if element_dead(v.element) && !dbg(2u) {
        // A broken element: every corner outside the view, so nothing draws.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    var local = v.position;
    if v.surface3.w > 7.5 && v.surface3.w < 10.5 {
        let flutter = select(0.0, 1.0, v.surface3.w < 8.5);
        local = veg_wind(local, v.normal, v.surface2.x, flutter, v.surface2.y, model, u.time, u.wind);
    }
    let world = model * vec4<f32>(local, 1.0);
    out.clip = u.view_proj * world;
    // Assume uniform scale for normals (friendly default).
    out.world_n = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    out.color = v.color * vec4<f32>(v.tint.rgb, select(v.tint.a, 1.0, v.tint.a < 0.0));
    out.lod_fade = v.tint.a;
    out.world_p = world.xyz;
    out.uv = v.uv;
    out.obj_p = local * vec3<f32>(length(v.m0.xyz), length(v.m1.xyz), length(v.m2.xyz));
    out.obj_x = v.m0.xyz;
    out.obj_y = v.m1.xyz;
    out.surface = v.surface;
    out.surface2 = v.surface2;
    out.surface3 = v.surface3;
    out.surface4 = v.surface4;
    out.surface5 = v.surface5;
    out.glow = u.glow[v.element >> 28u];
    return out;
}

// ---- Vegetation (crate::vegetation): leaf cards, bark, far canopy hulls ----
// surface2 = (wind weight, per-tuft random, crown occlusion, far part kind),
// surface3.xyz = linear autumn colour, surface4.w = turns in autumn,
// surface.w = leaf translucency.
const VEG_ATLAS: vec2<f32> = vec2<f32>(1024.0, 1536.0);
const VEG_LEAF_V: f32 = 0.6666667;
const VEG_ALPHA_TEST: f32 = 0.45;

fn veg_hash(p0: vec3<f32>) -> f32 {
    let p = fract(p0 * 0.3183099 + 0.1) * 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

fn veg_noise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f0 = fract(x);
    let f = f0 * f0 * (3.0 - 2.0 * f0);
    return mix(
        mix(mix(veg_hash(i), veg_hash(i + vec3<f32>(1.0, 0.0, 0.0)), f.x),
            mix(veg_hash(i + vec3<f32>(0.0, 1.0, 0.0)), veg_hash(i + vec3<f32>(1.0, 1.0, 0.0)), f.x), f.y),
        mix(mix(veg_hash(i + vec3<f32>(0.0, 0.0, 1.0)), veg_hash(i + vec3<f32>(1.0, 0.0, 1.0)), f.x),
            mix(veg_hash(i + vec3<f32>(0.0, 1.0, 1.0)), veg_hash(i + vec3<f32>(1.0, 1.0, 1.0)), f.x), f.y),
        f.z);
}

// Leaf coverage 0..1 for alpha-to-coverage: the atlas alpha (boosted in
// small mips so thin leaves don't vanish), faded where a card is seen
// edge-on, cut at the alpha test with a one-pixel soft edge.
fn veg_leaf_coverage(uv: vec2<f32>, world_p: vec3<f32>) -> f32 {
    let c = textureSample(albedo_tex, albedo_sampler, uv).a;
    let dx = dpdx(uv * VEG_ATLAS);
    let dy = dpdy(uv * VEG_ATLAS);
    let lod = max(0.0, 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-8)));
    let a = clamp(c * (1.0 + lod * 0.32), 0.0, 1.0);
    let face_n = normalize(cross(dpdx(world_p), dpdy(world_p)));
    let edge_on = abs(dot(face_n, normalize(world_p - u.eye)));
    let alpha = a * (smoothstep(0.04, 0.32, edge_on) * 0.55 + 0.45);
    return clamp((alpha - VEG_ALPHA_TEST) / max(fwidth(alpha), 1e-4) + 0.5, 0.0, 1.0);
}

fn veg_far_noise(world_p: vec3<f32>) -> f32 {
    return veg_noise(world_p * 2.2) * 0.6 + veg_noise(world_p * 5.1) * 0.4;
}

// Far canopy hulls dissolve towards their silhouette, so the ~150-triangle
// ball reads as a ragged crown.
fn veg_far_cut(in: VsOut) -> bool {
    if in.surface2.w < 0.75 {
        return false;
    }
    let edge = 1.0 - abs(dot(normalize(in.world_n), normalize(u.eye - in.world_p)));
    return edge * edge * 1.25 > veg_far_noise(in.world_p) + 0.12;
}

fn veg_sun_through_leaves(base: vec3<f32>, n: vec3<f32>, world_p: vec3<f32>, vis: f32, trans: f32, occ: f32) -> vec3<f32> {
    let l = normalize(u.light_dir);
    let to_eye = normalize(u.eye - world_p);
    let back = pow(clamp(dot(to_eye, -l), 0.0, 1.0), 6.0);
    let wrap = clamp(0.5 - 0.5 * dot(n, l), 0.0, 1.0);
    return u.light_color * base * (back * 0.8 + wrap * 0.18) * trans * (1.0 - occ * 0.6) * vis;
}

fn vegetation_shade(in: VsOut) -> vec4<f32> {
    let occ = in.surface2.z;
    let geo_n = normalize(in.world_n);
    let l = normalize(u.light_dir);
    if in.surface3.w < 8.5 {
        // Leaf card: crown normal on both faces (not flipped on the back).
        let coverage = veg_leaf_coverage(in.uv, in.world_p);
        if coverage <= 0.0 {
            discard;
        }
        var c = textureSample(albedo_tex, albedo_sampler, in.uv).rgb * in.color.rgb;
        let tv = in.surface2.y - 0.5;
        c *= 1.0 + tv * 0.4;
        c = mix(c, c * vec3<f32>(1.12, 1.06, 0.7), max(tv, 0.0) * 0.5);
        let lum = dot(c, vec3<f32>(0.3, 0.59, 0.11));
        let turn = clamp(u.season * 1.6 - in.surface2.y * 0.6, 0.0, 1.0) * in.surface4.w;
        c = mix(c, in.surface3.xyz * (lum * 3.2 + 0.015), turn);
        c *= 1.0 - occ * 0.62;
        let vis = sun_visibility(in.world_p, geo_n, u.eye);
        let ndl = max(dot(geo_n, l), 0.0);
        let lit = shade_surface_occluded(c, geo_n, in.world_p, in.surface.x, 0.0, vis, ndl, 1.0 - occ * 0.45)
            + veg_sun_through_leaves(c, geo_n, in.world_p, vis, in.surface.w, occ)
            + c * torch_light(in.world_p, geo_n);
        return vec4<f32>(haze(lit, in.world_p), coverage);
    }
    if in.surface3.w < 9.5 {
        // Bark: uv.x = strip * 16 + u (around), uv.y along; the strip's
        // column of the bark atlas under the leaves, sampled with the
        // gradients of the unwrapped uv so the wrap seams stay clean.
        let strip = floor(in.uv.x / 16.0);
        let bu = vec2<f32>(in.uv.x - strip * 16.0, in.uv.y);
        let span = vec2<f32>(0.976 / 8.0, (1.0 - VEG_LEAF_V) * 0.996);
        let at = vec2<f32>((strip + 0.012 + fract(bu.x) * 0.976) / 8.0,
            VEG_LEAF_V + (0.002 + fract(bu.y) * 0.996) * (1.0 - VEG_LEAF_V));
        let gx = dpdx(bu) * span;
        let gy = dpdy(bu) * span;
        let tex = textureSampleGrad(albedo_tex, albedo_sampler, at, gx, gy).rgb;
        var t = textureSampleGrad(normal_tex, albedo_sampler, at, gx, gy).xyz * 2.0 - 1.0;
        t = vec3<f32>(t.xy * 1.1, t.z);
        // Cotangent frame from the continuous bark uv (see mapped_normal).
        let dp1 = dpdx(in.world_p);
        let dp2 = dpdy(in.world_p);
        let duv1 = dpdx(bu);
        let duv2 = dpdy(bu);
        let dp2perp = cross(dp2, geo_n);
        let dp1perp = cross(geo_n, dp1);
        let tangent = -(dp2perp * duv1.x + dp1perp * duv2.x);
        let bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
        let scale = max(dot(tangent, tangent), dot(bitangent, bitangent));
        var n = geo_n;
        if scale > 1e-20 {
            let inv = inverseSqrt(scale);
            n = normalize(tangent * (t.x * inv) + bitangent * (t.y * inv) + geo_n * t.z);
        }
        let c = tex * in.color.rgb * (1.0 - occ * 0.6);
        let vis = sun_visibility(in.world_p, geo_n, u.eye);
        let ndl = max(dot(n, l), 0.0);
        let lit = shade_surface_occluded(c, n, in.world_p, in.surface.x, 0.0, vis, ndl, 1.0 - occ * 0.5)
            + c * torch_light(in.world_p, n);
        return vec4<f32>(haze(lit, in.world_p), 1.0);
    }
    // Far hull: bark prisms (kind 0), palm frond ribbons (0.5), canopy (1).
    if veg_far_cut(in) {
        discard;
    }
    let kind = in.surface2.w;
    let nz = veg_far_noise(in.world_p);
    var c = in.color.rgb;
    var n = geo_n;
    let dist = distance(u.eye, in.world_p);
    if kind < 0.25 {
        // Far trunks stand in the wood's shade.
        c *= mix(1.0, 0.25, smoothstep(55.0, 130.0, dist));
    } else {
        // Clumps: the hull normal bent by noise, so the crown catches the
        // light in lumps instead of shading as one smooth ball.
        let q = in.world_p;
        let lump1 = vec3<f32>(veg_noise(q * 0.9), veg_noise(q * 0.9 + 17.3), veg_noise(q * 0.9 + 41.7)) - 0.5;
        let lump2 = vec3<f32>(veg_noise(q * 2.4), veg_noise(q * 2.4 + 17.3), veg_noise(q * 2.4 + 41.7)) - 0.5;
        n = normalize(geo_n + lump1 * 1.5 + lump2 * 0.7);
        let tint = veg_noise(floor(q * 0.08) + 0.5);
        c = c * (0.62 + nz * 0.5) * vec3<f32>(0.84 + tint * 0.3, 0.9 + tint * 0.18, 0.92);
        c = mix(c, in.surface3.xyz * (0.78 + nz * 0.45), clamp(u.season * 1.4 - nz * 0.4, 0.0, 1.0) * in.surface4.w);
    }
    let vis = sun_visibility(in.world_p, geo_n, u.eye);
    let ndl = max(dot(n, l), 0.0);
    let lit = shade_surface(c, n, in.world_p, in.surface.x, 0.0, vis, ndl * 0.8 + 0.2 * ndl * ndl);
    return vec4<f32>(haze(lit, in.world_p), 1.0);
}

// Depth prepass: only texels the shaded pass will cover on every sample.
@fragment
fn fs_depth(in: VsOut) {
    if dbg(8192u) && distance(in.world_p, u.eye) > 60.0 {
        discard;
    }
    if dbg(4u) {
        return;
    }
    if !lod_dither_keep(in.lod_fade, in.clip.xy, u.dither_frame) && !dbg(1u) {
        discard;
    }
    if in.surface3.w > 7.5 {
        if in.surface3.w < 8.5 {
            if veg_leaf_coverage(in.uv, in.world_p) < 0.999 {
                discard;
            }
        } else if in.surface3.w > 9.5 && veg_far_cut(in) {
            discard;
        }
        return;
    }
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let foliage = in.surface3.w > 5.5 && in.surface3.w < 7.5;
    if foliage || in.color.a * texel.a < 0.999 {
        discard;
    }
}

// ---- Triplanar texturing (MaterialMaps::triplanar) ----
struct TriSurface {
    albedo: vec4<f32>,
    n: vec3<f32>,
    orm: vec3<f32>,
};

// Albedo, normal and occlusion-roughness-metallic projected along the
// object's three axes and blended by the normal. Normals use the whiteout
// blend (after Ben Golus); maps follow the glTF convention, so the sample
// v is negated to make +v run up the image like the projected axis.
fn triplanar_surface(in: VsOut, geo_n: vec3<f32>) -> TriSurface {
    let ax = normalize(in.obj_x);
    let ay = normalize(in.obj_y);
    let az = cross(ax, ay);
    let on = vec3<f32>(dot(geo_n, ax), dot(geo_n, ay), dot(geo_n, az));
    var w = pow(abs(on), vec3<f32>(4.0));
    w = w / max(w.x + w.y + w.z, 1e-5);
    let sg = select(vec3<f32>(-1.0), vec3<f32>(1.0), on >= vec3<f32>(0.0));
    let p = in.obj_p * mat.triplanar.x;
    let uvx = vec2<f32>(p.z * sg.x, -p.y);
    let uvy = vec2<f32>(p.x * sg.y, -p.z);
    let uvz = vec2<f32>(-p.x * sg.z, -p.y);
    var out: TriSurface;
    out.albedo = textureSample(albedo_tex, albedo_sampler, uvx) * w.x
        + textureSample(albedo_tex, albedo_sampler, uvy) * w.y
        + textureSample(albedo_tex, albedo_sampler, uvz) * w.z;
    out.orm = material_orm(uvx) * w.x + material_orm(uvy) * w.y + material_orm(uvz) * w.z;
    if mat.has_normal < 0.5 {
        out.n = geo_n;
        return out;
    }
    let k = vec2<f32>(mat.normal_scale);
    var tx = textureSample(normal_tex, albedo_sampler, uvx).xyz * 2.0 - 1.0;
    var ty = textureSample(normal_tex, albedo_sampler, uvy).xyz * 2.0 - 1.0;
    var tz = textureSample(normal_tex, albedo_sampler, uvz).xyz * 2.0 - 1.0;
    tx = vec3<f32>(tx.xy * k, tx.z);
    ty = vec3<f32>(ty.xy * k, ty.z);
    tz = vec3<f32>(tz.xy * k, tz.z);
    tx.x *= sg.x;
    ty.x *= sg.y;
    tz.x *= -sg.z;
    tx = vec3<f32>(tx.xy + on.zy, abs(tx.z) * on.x);
    ty = vec3<f32>(ty.xy + on.xz, abs(ty.z) * on.y);
    tz = vec3<f32>(tz.xy + on.xy, abs(tz.z) * on.z);
    let n_obj = normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
    out.n = normalize(ax * n_obj.x + ay * n_obj.y + az * n_obj.z);
    return out;
}

// Without MSAA there is no alpha-to-coverage: opaque pipelines then turn
// partial coverage (leaf edges, cut-out texels) into a per-pixel dither that
// changes every frame, and TAA blends it back into a soft edge.
override SCENE_ALPHA_DITHER: f32 = 0.0;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var c = shade_main(in);
    if not_finite(c.r) || not_finite(c.g) || not_finite(c.b) || not_finite(c.a) {
        if dbg(64u) {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0);
        }
        // A broken shading input (NaN or infinite) must not reach the
        // image: reflections read last frame's image and would carry it
        // to every glossy surface, frame after frame.
        c = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // The image holds half floats (largest 65504): anything brighter would
    // be stored as infinity, which probes and reflections turn into NaN.
    c = vec4<f32>(min(c.rgb, vec3<f32>(HDR_MAX)), c.a);
    if dbg(32u) {
        c.a = 1.0;
    }
    if SCENE_ALPHA_DITHER > 0.5 {
        let q = floor(in.clip.xy) + vec2<f32>(5.588238 * u.dither_frame);
        let h = fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
        if c.a <= h * 0.98 + 0.01 {
            discard;
        }
        c.a = 1.0;
    }
    return c;
}

fn shade_main(in: VsOut) -> vec4<f32> {
    if dbg(8192u) && distance(in.world_p, u.eye) > 60.0 {
        discard;
    }
    if !lod_dither_keep(in.lod_fade, in.clip.xy, u.dither_frame) && !dbg(1u) {
        discard;
    }
    if dbg(32768u) {
        // The depth this pixel was given, against the depth its own position
        // has under the frame's camera: green where they agree, red where not.
        let c = u.view_proj * vec4<f32>(in.world_p, 1.0);
        let want = c.z / c.w;
        let off = abs(want - in.clip.z) / max(abs(want), 1e-7);
        if off > 0.02 || c.w <= 0.0 {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        return vec4<f32>(0.1, 0.6 + 0.4 * fract(log2(max(in.clip.z, 1e-12)) * 0.5), 0.1, 1.0);
    }
    if dbg(4096u) {
        // Depth bands: each doubling of the stored depth value is one band.
        let z = max(in.clip.z, 1e-12);
        let band = fract(log2(z) * 0.5);
        return vec4<f32>(band, fract(band * 3.0), 1.0 - band, 1.0);
    }
    if dbg(8u) && in.surface3.w < 0.001 && mat.triplanar.x > 0.0 {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }
    if dbg(16u) && in.surface3.w < 0.001 && mat.triplanar.x <= 0.0 {
        return vec4<f32>(0.0, 1.0, 1.0, 1.0);
    }
    if in.surface3.w > 7.5 {
        return vegetation_shade(in);
    }
    // Most world props use the default material. Keep them on the cheap,
    // conventional path; authored wood/dirt/grass/sand/snow/cave profiles
    // continue through the full procedural material evaluation below.
    if in.surface3.w < 0.001 || CUSTOM_MATERIAL > 0.5 {
        let geo_n = normalize(in.world_n);
        // Normal, occlusion-roughness-metallic and emissive maps; neutral
        // 1x1 texels for meshes without them.
        var n: vec3<f32>;
        var orm: vec3<f32>;
        var texel: vec4<f32>;
        if mat.triplanar.x > 0.0 {
            let t = triplanar_surface(in, geo_n);
            n = t.n;
            orm = t.orm;
            texel = t.albedo;
            if dbg(2048u) {
                n = geo_n;
                orm = vec3<f32>(1.0);
            }
        } else {
            n = mapped_normal(geo_n, in.world_p, in.uv);
            orm = material_orm(in.uv);
            texel = textureSample(albedo_tex, albedo_sampler, in.uv);
        }
        var surf: MaterialOut;
        surf.color = in.color * texel;
        surf.normal = n;
        surf.roughness = in.surface.x * orm.y;
        surf.metallic = in.surface.y * orm.z;
        surf.emission = vec3<f32>(0.0);
        if CUSTOM_MATERIAL > 0.5 {
            // A game's own material function (World::add_material).
            var m: MaterialIn;
            m.world_pos = in.world_p;
            m.object_pos = in.obj_p;
            m.normal = n;
            m.view_dir = normalize(u.eye - in.world_p);
            m.uv = in.uv;
            m.time = u.time;
            m.params = custom_params();
            surf = custom_material(m, surf);
            surf.normal = normalize(surf.normal);
        }
        let l = normalize(u.light_dir);
        let ndl = max(dot(surf.normal, l), 0.0);
        let wrap = ndl * 0.65 + 0.35;
        let base = surf.color;
        let vis = sun_visibility(in.world_p, geo_n, u.eye);
        let lit = shade_surface_occluded(base.rgb, surf.normal, in.world_p, surf.roughness,
                surf.metallic, vis, wrap * wrap, orm.x)
            + base.rgb * torch_light(in.world_p, surf.normal)
            + select(in.glow.rgb, base.rgb, in.glow.r < 0.0) * (in.surface.z * 3.0 * in.glow.a)
            + material_emission(in.uv)
            + surf.emission;
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
        let fresnel = pow(max(1.0 - max(dot(n, view), 0.0), 0.0), 2.0);
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
    // Pixel size in metres: noise octaves finer than that are skipped.
    let foot = max(length(dpdx(p)), length(dpdy(p)));
    let broad = fbm3_lod(p * 0.11 + vec3<f32>(seed, 5.0, 9.0), 0.5, foot * 0.11);
    // Colour grain and the bumped normal from one three-channel noise.
    let grain = fbm3v_lod(p * 1.7 + vec3<f32>(3.0, seed, 1.0), 0.5, foot * 1.7);
    let fine = grain.x;
    let strata = fbm3_lod(vec3<f32>(p.x * 0.04, p.y * 0.85, p.z * 0.04) + vec3<f32>(seed * 0.3), 0.55, foot * 0.85);
    let n = normalize(n0 + (grain - 0.5) * 0.18);

    var rock = in.color.rgb * (0.62 + 0.5 * strata) * (0.86 + 0.28 * fine);
    var earth = vec3<f32>(0.33, 0.25, 0.17) * (0.8 + 0.35 * broad) * (0.9 + 0.2 * fine);
    var grass = mix(vec3<f32>(0.20, 0.33, 0.10), vec3<f32>(0.36, 0.42, 0.16), broad)
        * (0.85 + 0.3 * fine);
    // Sand material: sand (the terrain colour) where grass would grow,
    // weathered grey-brown rock on the steep faces.
    if i32(round(in.surface3.w)) == 2 {
        let sand = in.color.rgb * (0.9 + 0.2 * broad) * (0.9 + 0.2 * fine);
        grass = sand;
        earth = sand * 0.88;
        rock = vec3<f32>(0.40, 0.37, 0.33) * (0.62 + 0.5 * strata) * (0.86 + 0.28 * fine);
    }
    if in.surface5.y > 0.5 {
        // Beach: sand on the low ground, its upper edge wandering with the
        // noise; steep faces keep their rock.
        let top = in.surface5.x + (broad - 0.5) * 2.4;
        let s = 1.0 - smoothstep(top - 0.5, top + 0.5, p.y);
        let sand = vec3<f32>(0.66, 0.55, 0.36) * (0.9 + 0.2 * broad) * (0.9 + 0.2 * fine);
        grass = mix(grass, sand, s);
        earth = mix(earth, sand * 0.88, s);
    }
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
    return vec4<f32>(hdr_safe(haze(lit, p)), 1.0);
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
    /// After a depth prepass: same shading, depth test only.
    pub opaque_nowrite: wgpu::RenderPipeline,
    pub voxel_nowrite: wgpu::RenderPipeline,
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
        source: wgpu::ShaderSource::Wgsl(lit_shader_source(None).into()),
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
                _depth_write: bool,
                cull: Option<wgpu::Face>,
                depth_stencil: wgpu::DepthStencilState,
                buffers: &[wgpu::VertexBufferLayout<'static>],
                fs_entry: &str| {
        // Opaque draws turn foliage alpha into coverage.
        // Voxel ground is always solid: no coverage from alpha.
        let alpha_to_coverage = blend == wgpu::BlendState::REPLACE && fs_entry != "fs_voxel";
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
                    constants: if blend == wgpu::BlendState::REPLACE {
                        if super::stencil::scene_samples() == 1 && fs_entry == "fs_main" {
                            &[("SCENE_ALPHA_DITHER", 1.0)]
                        } else {
                            &[]
                        }
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
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        })
    };

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
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        })
    };

    // The scene shader is large and each variant compiles it again, so
    // the variants compile on worker threads at once.
    let (
        opaque,
        transparent,
        opaque_portal,
        transparent_portal,
        voxel_opaque,
        opaque_nowrite,
        voxel_nowrite,
        depth_prepass,
        voxel_depth_prepass,
    ) = std::thread::scope(|s| {
        let opaque = s.spawn(|| {
            make(
                "opaque-pipeline",
                wgpu::BlendState::REPLACE,
                true,
                Some(wgpu::Face::Back),
                super::stencil::scene_depth_stencil_unmasked_write(true),
                &MESH_BUFFERS,
                "fs_main",
            )
        });
        let transparent = s.spawn(|| {
            make(
                "transparent-pipeline",
                wgpu::BlendState::ALPHA_BLENDING,
                false,
                None, // water/glass readable from both sides
                super::stencil::scene_depth_stencil_unmasked_write(false),
                &MESH_BUFFERS,
                "fs_main",
            )
        });
        let opaque_portal = s.spawn(|| {
            make(
                "opaque-portal-pipeline",
                wgpu::BlendState::REPLACE,
                true,
                Some(wgpu::Face::Back),
                super::stencil::scene_depth_stencil_masked_write(true),
                &MESH_BUFFERS,
                "fs_main",
            )
        });
        let transparent_portal = s.spawn(|| {
            make(
                "transparent-portal-pipeline",
                wgpu::BlendState::ALPHA_BLENDING,
                false,
                None,
                super::stencil::scene_depth_stencil_masked_write(false),
                &MESH_BUFFERS,
                "fs_main",
            )
        });

        let voxel_opaque = s.spawn(|| {
            make(
                "voxel-opaque-pipeline",
                wgpu::BlendState::REPLACE,
                true,
                Some(wgpu::Face::Back),
                super::stencil::scene_depth_stencil_unmasked_write(true),
                &super::voxel_terrain::VOXEL_BUFFERS,
                "fs_voxel",
            )
        });
        let opaque_nowrite = s.spawn(|| {
            make(
                "opaque-nowrite-pipeline",
                wgpu::BlendState::REPLACE,
                false,
                Some(wgpu::Face::Back),
                super::stencil::scene_depth_stencil_unmasked_write(false),
                &MESH_BUFFERS,
                "fs_main",
            )
        });
        let voxel_nowrite = s.spawn(|| {
            make(
                "voxel-nowrite-pipeline",
                wgpu::BlendState::REPLACE,
                false,
                Some(wgpu::Face::Back),
                super::stencil::scene_depth_stencil_unmasked_write(false),
                &super::voxel_terrain::VOXEL_BUFFERS,
                "fs_voxel",
            )
        });
        let depth_prepass = s.spawn(|| depth_prepass_for("opaque-depth-prepass", &MESH_BUFFERS));
        let voxel_depth_prepass = s.spawn(|| {
            depth_prepass_for("voxel-depth-prepass", &super::voxel_terrain::VOXEL_BUFFERS)
        });
        (
            super::join_worker(opaque),
            super::join_worker(transparent),
            super::join_worker(opaque_portal),
            super::join_worker(transparent_portal),
            super::join_worker(voxel_opaque),
            super::join_worker(opaque_nowrite),
            super::join_worker(voxel_nowrite),
            super::join_worker(depth_prepass),
            super::join_worker(voxel_depth_prepass),
        )
    });

    Pipelines {
        opaque,
        voxel_opaque,
        opaque_nowrite,
        voxel_nowrite,
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

/// The opaque scene pipelines reading materials through the material table
/// (texture arrays): what multi-draws use, so one call can cover meshes
/// with different textures.
pub struct BindlessPipelines {
    pub opaque: wgpu::RenderPipeline,
    pub opaque_nowrite: wgpu::RenderPipeline,
    pub depth_prepass: wgpu::RenderPipeline,
}

pub fn create_bindless_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    bind_layout: &wgpu::BindGroupLayout,
    table_layout: &wgpu::BindGroupLayout,
) -> BindlessPipelines {
    let source = super::material_table::bindless_scene_source(&lit_shader_source(None));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lit-shader-bindless"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout-bindless"),
        bind_group_layouts: &[bind_layout, table_layout],
        push_constant_ranges: &[],
    });
    let make = |label: &str, depth_write: bool, prepass: bool| {
        let targets = [Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let dither: &[(&str, f64)] = if super::stencil::scene_samples() == 1 {
            &[("SCENE_ALPHA_DITHER", 1.0)]
        } else {
            &[]
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &MESH_BUFFERS,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(if prepass { "fs_depth" } else { "fs_main" }),
                targets: if prepass { &[] } else { &targets },
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: if prepass { &[] } else { dither },
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(super::stencil::scene_depth_stencil_unmasked_write(
                depth_write,
            )),
            multisample: super::stencil::scene_multisample(!prepass),
            multiview: None,
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        })
    };
    let (opaque, opaque_nowrite, depth_prepass) = std::thread::scope(|s| {
        let opaque = s.spawn(|| make("opaque-pipeline-bindless", true, false));
        let nowrite = s.spawn(|| make("opaque-nowrite-pipeline-bindless", false, false));
        let prepass = s.spawn(|| make("opaque-depth-prepass-bindless", true, true));
        (
            super::join_worker(opaque),
            super::join_worker(nowrite),
            super::join_worker(prepass),
        )
    });
    BindlessPipelines {
        opaque,
        opaque_nowrite,
        depth_prepass,
    }
}

/// A custom material's GPU side: its two pipelines (solid and blended) and
/// the uniform buffer holding its four params.
pub struct GpuCustomMaterial {
    pub revision: u64,
    pub opaque: wgpu::RenderPipeline,
    pub blend: wgpu::RenderPipeline,
    pub params: wgpu::Buffer,
    pub bind: wgpu::BindGroup,
}

pub fn custom_params_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("custom-material-params-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Build the pipelines for one custom material (code already checked by
/// `CustomMaterial::check`).
pub fn create_custom_material(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    scene_layouts: [&wgpu::BindGroupLayout; 2],
    params_layout: &wgpu::BindGroupLayout,
    material: &crate::custom_material::CustomMaterial,
) -> GpuCustomMaterial {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("custom-material-{}", material.name)),
        source: wgpu::ShaderSource::Wgsl(lit_shader_source(Some(&material.code)).into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("custom-material-layout"),
        bind_group_layouts: &[scene_layouts[0], scene_layouts[1], params_layout],
        push_constant_ranges: &[],
    });
    let make = |blend: wgpu::BlendState, solid: bool| {
        let constants: &[(&str, f64)] = if solid {
            &[("CUSTOM_MATERIAL", 1.0)]
        } else {
            &[("CUSTOM_MATERIAL", 1.0), ("SCENE_AO", 0.0)]
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!(
                "custom-material-{}-{}",
                material.name,
                if solid { "solid" } else { "blend" }
            )),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &MESH_BUFFERS,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: if solid { Some(wgpu::Face::Back) } else { None },
                ..Default::default()
            },
            depth_stencil: Some(super::stencil::scene_depth_stencil_unmasked_write(solid)),
            multisample: super::stencil::scene_multisample(false),
            multiview: None,
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        })
    };
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom-material-params"),
        contents: bytemuck::cast_slice(&material.params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("custom-material-params-bind"),
        layout: params_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }],
    });
    GpuCustomMaterial {
        revision: 0,
        opaque: make(wgpu::BlendState::REPLACE, true),
        blend: make(wgpu::BlendState::ALPHA_BLENDING, false),
        params,
        bind,
    }
}
