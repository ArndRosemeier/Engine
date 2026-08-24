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
}

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
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Light from the viewer's own lantern. A point light at the eye with a
// distance falloff that stays gentle near the holder and dies at `reach`:
//   1 - (d/reach)^2  clamped â€” quadratic near, linear tail, zero past reach.
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
// instead â€” the obvious shortcut â€” all but switches the haze off as soon as the
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

pub fn scene_shader_prefix() -> String {
    format!(
        "{}{}{}{}",
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
    @builtin(position) clip: vec4<f32>,
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Most world props use the default material. Keep them on the cheap,
    // conventional path; authored wood/dirt/grass/sand/snow/cave profiles
    // continue through the full procedural material evaluation below.
    if in.surface3.w < 0.001 {
        let n = normalize(in.world_n);
        let l = normalize(u.light_dir);
        let ndl = max(dot(n, l), 0.0);
        let wrap = ndl * 0.65 + 0.35;
        let base = in.color * textureSample(albedo_tex, albedo_sampler, in.uv);
        let vis = sun_visibility(in.world_p, n, u.eye);
        let lit = base.rgb * (u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color * vis)
            + base.rgb * torch_light(in.world_p, n);
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
        let light = u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color * vis + transmission;
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
    let grain = fbm3(sample_p * 1.7, in.surface2.w);
    let ridge = ridged3(sample_p * 0.65, in.surface2.w);
    let strata = sin(in.world_p.y * 1.35 + ridge * 4.2 + seed);
    let axis_coord = dot(in.world_p, axis);
    let cross_coord = in.world_p - axis * axis_coord;
    let wood_bands = 0.5 + 0.5 * sin(axis_coord * in.surface.w * 2.6
        + fbm3(cross_coord * 0.75 + vec3<f32>(seed, 4.0, 9.0), in.surface2.w) * 3.8);
    let wood_grain = pow(clamp(wood_bands, 0.0, 1.0), 1.7);
    let sand_plane_a = select(in.world_p.yz, in.world_p.xy, abs(axis.z) > 0.5);
    let sand_plane = select(sand_plane_a, in.world_p.xz, abs(axis.y) > 0.5);
    let sand_base = sand_plane * in.surface.w * 0.22 + vec2<f32>(seed * 0.031, seed * 0.071);
    let sand_a = sand_layer(sand_base, seed);
    let sand_b = sand_layer(rotate2(sand_base, 0.19) * 1.27, seed + 19.0);
    let sand_pattern = mix(sand_a, sand_b, smoothstep(0.18, 0.82,
        fbm3(vec3<f32>(sand_base * 1.8, seed + 41.0), 0.5)));
    let sand_grit = musgrave3(vec3<f32>(sand_plane * in.surface.w * 7.5, seed + 67.0), 0.52, 2.35);
    let dirt = dirt_profile(oriented_p * 0.75, seed, in.surface2.w);
    let grass = grass_profile(oriented_p * 0.62, seed, in.surface2.w);
    let material_dx = dpdx(oriented_p * in.surface.w);
    let material_dy = dpdy(oriented_p * in.surface.w);
    let footprint = max(length(material_dx), length(material_dy));
    let lod = log2(max(footprint, 0.0001));
    let fine_visibility = 1.0 - smoothstep(-4.0, -0.35, lod);
    let medium_visibility = 1.0 - smoothstep(-1.4, 1.8, lod);
    let is_dirt = in.surface3.w > 2.5 && in.surface3.w < 3.5;
    let is_grass = in.surface3.w > 3.5;
    // A separate, much finer directional fiber field. Narrow valleys darken the
    // wood and also perturb the normal, so grain remains readable at distance.
    let wood_fine_signed = musgrave3(
        axis * axis_coord * in.surface.w * 18.0
            + cross_coord * vec3<f32>(7.0, 3.0, 11.0)
            + vec3<f32>(seed * 2.3, seed * 0.37, seed * 1.71),
        0.52,
        2.5);
    let wood_fine = smoothstep(-0.18, 0.42, wood_fine_signed);
    let base_grain = select(grain, sand_pattern, in.surface3.w > 1.5);
    var material_grain = select(base_grain, dirt.x * 0.62 + dirt.y * 0.38, is_dirt);
    material_grain = select(material_grain,
        grass.x * 0.52 + grass.y * 0.30 + grass.z * 0.18, is_grass);
    material_grain = select(material_grain,
        wood_grain * 0.72 + wood_fine * 0.28,
        in.surface3.w > 0.5 && in.surface3.w < 1.5);
    let base_strata = select(strata, sand_pattern - 0.5, in.surface3.w > 1.5);
    var material_strata = select(base_strata, dirt.y - 0.5, is_dirt);
    material_strata = select(material_strata, grass.y - 0.5, is_grass);
    material_strata = select(material_strata, wood_grain - 0.5,
        in.surface3.w > 0.5 && in.surface3.w < 1.5);
    let base_detail = select(grain - 0.5, sand_grit, in.surface3.w > 1.5);
    var fine_detail = select(base_detail, dirt.z, is_dirt);
    fine_detail = select(fine_detail, grass.z - 0.5, is_grass);
    fine_detail = select(fine_detail, wood_fine - 0.5,
        in.surface3.w > 0.5 && in.surface3.w < 1.5);
    fine_detail *= fine_visibility;
    let detail = (in.surface.x * 0.13 + select(0.0, 0.045, in.surface3.w > 0.5)) * fine_detail;
    let n = normalize(in.world_n + vec3<f32>(detail, detail * 0.35, -detail));
    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let layer = 1.0 + (material_strata * in.surface2.x * medium_visibility + (ridge - 0.5) * 0.55) * 0.32;
    let base_tone = select(1.0, 0.72 + material_grain * 0.55, in.surface3.w > 1.5);
    var wood_tone = select(base_tone, 0.72 + material_grain * 0.45, is_dirt);
    wood_tone = select(wood_tone, 0.74 + material_grain * 0.48, is_grass);
    wood_tone = select(wood_tone, 0.56 + material_grain * 0.78,
        in.surface3.w > 0.5 && in.surface3.w < 1.5);
    let base_fine_color = select(0.92 + fine_detail * 0.18, 1.0, in.surface3.w > 1.5);
    var fine_color = select(base_fine_color, 0.82 + fine_detail * 0.25, is_dirt);
    fine_color = select(fine_color, 0.78 + fine_detail * 0.38, is_grass);
    fine_color = select(fine_color, 0.82 + wood_fine * 0.34,
        in.surface3.w > 0.5 && in.surface3.w < 1.5);
    let base = in.color * texel * layer * wood_tone * fine_color;
    // Soft wrap lighting â€” enough contrast for smooth heightfields to read as
    // hills. Sky and sun share one budget, so raising ambient fills the shadows
    // instead of blowing out everything the sun already reaches.
    let wrap = ndl * 0.65 + 0.35;
    let vis = sun_visibility(in.world_p, n, u.eye);
    var lit = base.rgb * (u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color * vis);
    lit += base.rgb * torch_light(in.world_p, n);
    let glint = smoothstep(0.92, 0.99, material_grain) * in.surface.z
        + smoothstep(0.72, 0.98, wood_fine) * select(0.0, 0.025, in.surface3.w > 0.5);
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
    // rocks and avoids privileging the XZ plane on walls or overhangs.
    let tri_weights = abs(n) / max(dot(abs(n), vec3<f32>(1.0)), 0.001);
    let tri_noise = fbm3(in.world_p.yzx * 0.55 + vec3<f32>(seed, 37.0, 11.0), 0.5) * tri_weights.x
        + fbm3(in.world_p.zxy * 0.55 + vec3<f32>(seed + 17.0, 7.0, 19.0), 0.5) * tri_weights.y
        + fbm3(in.world_p.xyz * 0.55 + vec3<f32>(seed + 31.0, 13.0, 5.0), 0.5) * tri_weights.z;
    let breakup = smoothstep(0.30, 0.72, tri_noise);
    let snow_fine = musgrave3(in.world_p * in.surface.w * 4.5 + vec3<f32>(seed, 61.0, 23.0), 0.52, 2.25);
    let overlay = directional * breakup * in.surface5.y;
    let overlay_mix = clamp(overlay, 0.0, 1.0);
    let overlay_color = vec3<f32>(0.78 + snow_fine * 0.12, 0.84 + snow_fine * 0.10, 0.94 + snow_fine * 0.06);
    let overlay_base = mix(base.rgb, overlay_color, overlay_mix);
    let overlay_lit = overlay_base * (u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color * vis);
    let snow_bump = snow_fine * overlay_mix * select(0.0, 0.055, in.surface3.w > 4.5);
    let snow_sparkle = snow_bump * smoothstep(0.55, 0.95, dot(n, normalize(u.light_dir)))
        * vec3<f32>(0.32, 0.48, 0.72);
    let overlay_specular = mix(0.0, 0.06 * (1.0 - in.surface5.z), overlay_mix)
        + snow_bump * 0.22;
    let emission = vec3<f32>(1.0, 0.78, 0.48) * in.surface.z;
    return vec4<f32>(haze(mix(lit, overlay_lit, overlay_mix) + overlay_specular + snow_sparkle + emission, in.world_p), alpha);
}
"#;

pub const SCENE_UNIFORM_SLOTS: usize = 8;

pub struct SceneUniformSlots {
    pub buffers: [wgpu::Buffer; SCENE_UNIFORM_SLOTS],
    pub bind_groups: [wgpu::BindGroup; SCENE_UNIFORM_SLOTS],
}

impl SceneUniformSlots {
    pub fn get(&self, level: usize) -> (&wgpu::Buffer, &wgpu::BindGroup) {
        let level = level.min(SCENE_UNIFORM_SLOTS - 1);
        (&self.buffers[level], &self.bind_groups[level])
    }
}

pub struct Pipelines {
    pub opaque: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub opaque_portal: wgpu::RenderPipeline,
    pub transparent_portal: wgpu::RenderPipeline,
    pub scene_uniforms: SceneUniformSlots,
    pub bind_layout: wgpu::BindGroupLayout,
    pub albedo_layout: wgpu::BindGroupLayout,
    pub albedo_sampler: wgpu::Sampler,
    /// Keeps the 1Ã—1 white texel alive for `white_albedo`.
    #[allow(dead_code)]
    pub white_texture: wgpu::Texture,
    pub white_albedo: wgpu::BindGroup,
}

impl Pipelines {
    pub fn scene_bind_group(&self, level: usize) -> &wgpu::BindGroup {
        &self.scene_uniforms.bind_groups[level.min(SCENE_UNIFORM_SLOTS - 1)]
    }
}

pub fn create_pipelines(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    shadow: &super::shadow::ShadowGpu,
) -> Pipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lit-shader"),
        source: wgpu::ShaderSource::Wgsl(format!("{}{SHADER}", scene_shader_prefix()).into()),
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
            let entries = shadow.scene_bind_entries(buffers[slot].as_entire_binding());
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("uniform-bind-{slot}")),
                layout: &bind_layout,
                entries: &entries,
            })
        });
        SceneUniformSlots {
            buffers,
            bind_groups,
        }
    };

    let albedo_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-albedo-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
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
    let white_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh-albedo-white-tex"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &white_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[255, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let white_view = white_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let white_albedo = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-albedo-white"),
        layout: &albedo_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&white_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&albedo_sampler),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout"),
        bind_group_layouts: &[&bind_layout, &albedo_layout],
        push_constant_ranges: &[],
    });

    let make = |label: &str,
                blend: wgpu::BlendState,
                depth_write: bool,
                cull: Option<wgpu::Face>,
                depth_stencil: wgpu::DepthStencilState| {
        let _ = depth_write;
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
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
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
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
    );
    let transparent = make(
        "transparent-pipeline",
        wgpu::BlendState::ALPHA_BLENDING,
        false,
        None, // water/glass readable from both sides
        super::stencil::scene_depth_stencil_unmasked_write(false),
    );
    let opaque_portal = make(
        "opaque-portal-pipeline",
        wgpu::BlendState::REPLACE,
        true,
        Some(wgpu::Face::Back),
        super::stencil::scene_depth_stencil_masked_write(true),
    );
    let transparent_portal = make(
        "transparent-portal-pipeline",
        wgpu::BlendState::ALPHA_BLENDING,
        false,
        None,
        super::stencil::scene_depth_stencil_masked_write(false),
    );

    Pipelines {
        opaque,
        transparent,
        opaque_portal,
        transparent_portal,
        scene_uniforms,
        bind_layout,
        albedo_layout,
        albedo_sampler,
        white_texture,
        white_albedo,
    }
}
