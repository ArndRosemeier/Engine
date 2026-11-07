//! GPU clipmap: static grids displaced by a WGSL height formula.

use super::grass::{GrassUniform, BLADE_VERTICES};
use crate::proc_terrain::{ClipmapConfig, ProcTerrain, TerrainLayers, TerrainParamsUniform};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Per-frame grass inputs: distance scale (quality times LOD bias, 0 = off),
/// animation time and wind strength.
#[derive(Clone, Copy, Debug, Default)]
pub struct GrassFrame {
    pub scale: f32,
    pub time: f32,
    pub wind: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ClipVertex {
    /// Local XZ in [-0.5, 0.5]; y unused (displaced in VS).
    pos: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RingUniform {
    /// World center of this ring (snapped).
    center: [f32; 2],
    /// World extent (full width) of this ring.
    extent: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    light_dir: [f32; 3],
    ambient: f32,
    light_color: [f32; 3],
    _pad: f32,
    eye: [f32; 3],
    /// Metres at which the torch contribution reaches zero.
    torch_radius_m: f32,
    torch_curve: f32,
    /// WGSL aligns the following vec4 to 16 bytes.
    _pad_torch: [f32; 3],
    /// Viewer-carried point light (rgb + unused); radius 0 switches it off.
    torch_color: [f32; 4],
    sky_up: [f32; 4],
    sky_horizon: [f32; 4],
    sky_down: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<FrameUniform>() == 192);

const SHADER: &str = r#"
struct TerrainParams {
    seed: u32,
    _pad0: u32,
    base_height: f32,
    hill_height: f32,
    hill_scale: f32,
    lake_scale: f32,
    lake_threshold: f32,
    water_level: f32,
    grass: vec4<f32>,
    sand: vec4<f32>,
    rock: vec4<f32>,
    water: vec4<f32>,
};

struct FrameUniform {
    view_proj: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _pad: f32,
    eye: vec3<f32>,
    torch_radius_m: f32,
    torch_curve: f32,
    _pad_t0: f32,
    _pad_t1: f32,
    _pad_t2: f32,
    torch_color: vec4<f32>,
    sky_up: vec4<f32>,
    sky_horizon: vec4<f32>,
    sky_down: vec4<f32>,
};

fn frame_light() -> SceneLight {
    var light: SceneLight;
    light.dir = normalize(frame.light_dir);
    light.color = frame.light_color;
    light.ambient = frame.ambient;
    light.sky_up = frame.sky_up.rgb;
    light.sky_horizon = frame.sky_horizon.rgb;
    light.sky_down = frame.sky_down.rgb;
    light.env_strength = 1.0;
    light.ao = 1.0;
    light.gi = vec4<f32>(0.0);
    return light;
}

struct RingUniform {
    center: vec2<f32>,
    extent: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniform;
@group(0) @binding(1) var<uniform> terrain: TerrainParams;
@group(0) @binding(2) var<uniform> ring: RingUniform;

// Ground texture layers, in order: grass, meadow, shore, rock, lake bed.
struct LayerParams {
    // x..w: 1 / tile metres of grass, meadow, shore, rock.
    inv_tile_a: vec4<f32>,
    // x: lake bed 1 / tile metres; y: 1 when textured, 0 for flat colours.
    inv_tile_b: vec4<f32>,
    normal_scale_a: vec4<f32>,
    normal_scale_b: vec4<f32>,
};
@group(2) @binding(0) var layer_albedo: texture_2d_array<f32>;
@group(2) @binding(1) var layer_normal: texture_2d_array<f32>;
@group(2) @binding(2) var layer_orm: texture_2d_array<f32>;
@group(2) @binding(3) var layer_sampler: sampler;
@group(2) @binding(4) var<uniform> layers: LayerParams;

// Light from the viewer's own lantern — same shape as the scene pipelines.
fn torch_light(world_p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if frame.torch_radius_m <= 0.0 {
        return vec3<f32>(0.0);
    }
    let to_p = world_p - frame.eye;
    let d = length(to_p);
    let reach = max(frame.torch_radius_m, 0.5);
    let fall = pow(clamp(1.0 - d / reach, 0.0, 1.0), max(frame.torch_curve, 0.05));
    let ndl = max(dot(n, -to_p / max(d, 1e-4)), 0.0);
    let wrap = ndl * 0.7 + 0.3;
    return frame.torch_color.rgb * (2.2 * fall * wrap);
}

struct VsIn {
    @location(0) local_xz: vec2<f32>,
};

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world_p: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) wet: f32,
    @location(4) ground: f32,
};

fn hash21(ix: i32, iy: i32, seed: u32) -> f32 {
    var n = u32(ix) * 1597334677u + u32(iy) * 3812015801u + seed * 2747636419u;
    n = n ^ (n >> 16u);
    n = n * 2246822519u;
    n = n ^ (n >> 13u);
    return f32(n >> 8u) / 16777215.0;
}

fn value_noise(p: vec2<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let ix = i32(i.x);
    let iy = i32(i.y);
    let a = hash21(ix, iy, seed);
    let b = hash21(ix + 1, iy, seed);
    let c = hash21(ix, iy + 1, seed);
    let d = hash21(ix + 1, iy + 1, seed);
    let v = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    return v * 2.0 - 1.0;
}

fn fbm2(p0: vec2<f32>, seed: u32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    var p = p0;
    var amp = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0u; o < octaves; o++) {
        sum += amp * value_noise(p, seed);
        norm += amp;
        amp *= gain;
        p *= lacunarity;
    }
    if norm > 0.0 {
        return sum / norm;
    }
    return 0.0;
}

const WATER_CLEARANCE: f32 = 0.02;

struct HeightOut {
    height: f32,
    ground: f32,
    water_top: f32,
    water: f32,
};

fn sample_field(xz: vec2<f32>) -> HeightOut {
    let n = fbm2(xz * terrain.hill_scale, terrain.seed, 5u, 2.1, 0.5);
    let h_raw = terrain.base_height + terrain.hill_height * n;
    let lake = fbm2(
        xz * terrain.lake_scale + vec2(17.0, 9.0),
        terrain.seed ^ 0xC0FFEEu,
        3u,
        2.0,
        0.55,
    );
    let lake_t = lake * 0.5 + 0.5;
    let span = max(1.0 - terrain.lake_threshold, 1e-3);
    let basin = clamp((lake_t - terrain.lake_threshold) / span, 0.0, 1.0);
    let floor_h = max(h_raw, terrain.water_level);
    let carved = floor_h - basin * 3.5;
    let near_shore = floor_h <= terrain.water_level + 1.5;
    let in_basin = basin > 0.25 && near_shore;
    var ground = floor_h;
    var water_top = -1e30;
    var water = false;
    if in_basin {
        water_top = terrain.water_level;
        ground = min(carved, water_top - WATER_CLEARANCE - 0.001);
        water = true;
    }
    var height = ground;
    if water {
        height = water_top;
    } else if basin > 0.0 && near_shore {
        var t = clamp(basin / 0.25, 0.0, 1.0);
        t = t * t * (3.0 - 2.0 * t);
        height = mix(floor_h, terrain.water_level, t);
    }
    var out: HeightOut;
    out.height = height;
    out.ground = ground;
    out.water_top = water_top;
    out.water = select(0.0, 1.0, water);
    return out;
}

fn terrain_color(ground: f32, water: f32) -> vec4<f32> {
    if water > 0.5 {
        return terrain.water;
    }
    if ground < terrain.water_level - 0.15 {
        return vec4(110.0/255.0, 125.0/255.0, 95.0/255.0, 1.0);
    }
    if ground < terrain.water_level + 1.0 {
        return terrain.sand;
    }
    if ground > terrain.base_height + terrain.hill_height * 0.55 {
        return terrain.rock;
    }
    return terrain.grass;
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let xz = ring.center + v.local_xz * ring.extent;
    let h = sample_field(xz);
    // Land uses ground; water surface drawn as second pass via wet flag / height.
    let y = h.ground;
    let world = vec3(xz.x, y, xz.y);

    // Finite-difference normal on the ground surface.
    let e = max(ring.extent / 128.0, 0.35);
    let hx = sample_field(xz + vec2(e, 0.0)).ground;
    let hz = sample_field(xz + vec2(0.0, e)).ground;
    let n = normalize(vec3(h.ground - hx, e, h.ground - hz));

    var out: VsOut;
    out.clip = frame.view_proj * vec4(world, 1.0);
    out.world_p = world;
    out.world_n = n;
    out.color = terrain_color(h.ground, h.water);
    out.wet = h.water;
    out.ground = h.ground;
    return out;
}

struct Grads {
    x: vec3<f32>,
    y: vec3<f32>,
};

struct LayerSample {
    albedo: vec3<f32>,
    n: vec3<f32>,
    // x: occlusion, y: roughness, z: relief height (0 deepest).
    orh: vec3<f32>,
};

// Seen from above: u along +x, v along -z (image rows run top-down), so the
// map's +y points along +z. `far` fades in a second, larger and turned
// sample that hides the tiling in the distance.
fn sample_flat(i: i32, xz: vec2<f32>, inv_tile: f32, k: f32, geo: vec3<f32>, far: f32, d: Grads) -> LayerSample {
    let uv = vec2<f32>(xz.x, -xz.y) * inv_tile;
    let gx = vec2<f32>(d.x.x, -d.x.z) * inv_tile;
    let gy = vec2<f32>(d.y.x, -d.y.z) * inv_tile;
    var out: LayerSample;
    var albedo = textureSampleGrad(layer_albedo, layer_sampler, uv, i, gx, gy).rgb;
    let r = mat2x2<f32>(0.8 * 0.27, 0.6 * 0.27, -0.6 * 0.27, 0.8 * 0.27);
    let uv2 = r * uv + vec2<f32>(0.37, 0.71);
    albedo = mix(albedo, textureSampleGrad(layer_albedo, layer_sampler, uv2, i, r * gx, r * gy).rgb, far * 0.5);
    out.albedo = albedo;
    let nm = textureSampleGrad(layer_normal, layer_sampler, uv, i, gx, gy);
    var t = nm.xyz * 2.0 - 1.0;
    t = vec3<f32>(t.xy * k, t.z);
    // Whiteout blend onto the ground normal.
    out.n = normalize(vec3<f32>(t.x + geo.x, abs(t.z) * geo.y, t.y + geo.z));
    let orm = textureSampleGrad(layer_orm, layer_sampler, uv, i, gx, gy);
    out.orh = vec3<f32>(orm.r, orm.g, nm.a);
    return out;
}

// World-space triplanar for cliffs (the same projection as mesh triplanar).
fn sample_tri(i: i32, wp: vec3<f32>, inv_tile: f32, k: f32, geo: vec3<f32>, d: Grads) -> LayerSample {
    var w = pow(abs(geo), vec3<f32>(4.0));
    w = w / max(w.x + w.y + w.z, 1e-5);
    let sg = select(vec3<f32>(-1.0), vec3<f32>(1.0), geo >= vec3<f32>(0.0));
    let p = wp * inv_tile;
    let uvx = vec2<f32>(p.z * sg.x, -p.y);
    let uvy = vec2<f32>(p.x * sg.y, -p.z);
    let uvz = vec2<f32>(-p.x * sg.z, -p.y);
    let ax = d.x * inv_tile;
    let ay = d.y * inv_tile;
    let xx = vec2<f32>(ax.z * sg.x, -ax.y);
    let xy = vec2<f32>(ay.z * sg.x, -ay.y);
    let yx = vec2<f32>(ax.x * sg.y, -ax.z);
    let yy = vec2<f32>(ay.x * sg.y, -ay.z);
    let zx = vec2<f32>(-ax.x * sg.z, -ax.y);
    let zy = vec2<f32>(-ay.x * sg.z, -ay.y);
    var out: LayerSample;
    out.albedo = textureSampleGrad(layer_albedo, layer_sampler, uvx, i, xx, xy).rgb * w.x
        + textureSampleGrad(layer_albedo, layer_sampler, uvy, i, yx, yy).rgb * w.y
        + textureSampleGrad(layer_albedo, layer_sampler, uvz, i, zx, zy).rgb * w.z;
    let ox = textureSampleGrad(layer_orm, layer_sampler, uvx, i, xx, xy);
    let oy = textureSampleGrad(layer_orm, layer_sampler, uvy, i, yx, yy);
    let oz = textureSampleGrad(layer_orm, layer_sampler, uvz, i, zx, zy);
    let nx = textureSampleGrad(layer_normal, layer_sampler, uvx, i, xx, xy);
    let ny = textureSampleGrad(layer_normal, layer_sampler, uvy, i, yx, yy);
    let nz = textureSampleGrad(layer_normal, layer_sampler, uvz, i, zx, zy);
    var tx = nx.xyz * 2.0 - 1.0;
    var ty = ny.xyz * 2.0 - 1.0;
    var tz = nz.xyz * 2.0 - 1.0;
    tx = vec3<f32>(tx.xy * k, tx.z);
    ty = vec3<f32>(ty.xy * k, ty.z);
    tz = vec3<f32>(tz.xy * k, tz.z);
    tx.x *= sg.x;
    ty.x *= sg.y;
    tz.x *= -sg.z;
    tx = vec3<f32>(tx.xy + geo.zy, abs(tx.z) * geo.x);
    ty = vec3<f32>(ty.xy + geo.xz, abs(ty.z) * geo.y);
    tz = vec3<f32>(tz.xy + geo.xy, abs(tz.z) * geo.z);
    out.n = normalize(tx.zyx * w.x + ty.xzy * w.y + tz.xyz * w.z);
    let o = ox * w.x + oy * w.y + oz * w.z;
    let h = nx.a * w.x + ny.a * w.y + nz.a * w.z;
    out.orh = vec3<f32>(o.r, o.g, h);
    return out;
}

fn layer_tile(i: i32) -> f32 {
    switch i {
        case 0: { return layers.inv_tile_a.x; }
        case 1: { return layers.inv_tile_a.y; }
        case 2: { return layers.inv_tile_a.z; }
        case 3: { return layers.inv_tile_a.w; }
        default: { return layers.inv_tile_b.x; }
    }
}

fn layer_normal_scale(i: i32) -> f32 {
    switch i {
        case 0: { return layers.normal_scale_a.x; }
        case 1: { return layers.normal_scale_a.y; }
        case 2: { return layers.normal_scale_a.z; }
        case 3: { return layers.normal_scale_a.w; }
        default: { return layers.normal_scale_b.x; }
    }
}

fn sample_layer(i: i32, in: VsOut, geo: vec3<f32>, far: f32, d: Grads) -> LayerSample {
    if i == 3 {
        return sample_tri(i, in.world_p, layer_tile(i), layer_normal_scale(i), geo, d);
    }
    return sample_flat(i, in.world_p.xz, layer_tile(i), layer_normal_scale(i), geo, far, d);
}

// Blend the ground layers by water level, slope, height and large patches,
// sharpened by each layer's relief so stones poke through sand and grass.
struct GroundMix {
    // Weights of grass, meadow, shore, rock and lake bed.
    w: array<f32, 5>,
    patches: f32,
};

// Which ground layer covers a spot: by water level, slope, height and large
// patches. Shared by the ground and the grass blades.
fn ground_mix(xz: vec2<f32>, g: f32, up: f32) -> GroundMix {
    let wl = terrain.water_level;
    let patches = fbm2(xz * 0.012, terrain.seed ^ 0x5151u, 3u, 2.0, 0.5);
    let fine = value_noise(xz * 0.13, terrain.seed ^ 0x7777u);
    let meadow = smoothstep(0.32, 0.6, patches);
    // Beaches only round the lakes (the same lake noise as the height field),
    // not on every low flat.
    let lake = fbm2(
        xz * terrain.lake_scale + vec2(17.0, 9.0),
        terrain.seed ^ 0xC0FFEEu,
        3u,
        2.0,
        0.55,
    ) * 0.5 + 0.5;
    let near_lake = smoothstep(terrain.lake_threshold - 0.07, terrain.lake_threshold + 0.01, lake + fine * 0.02);
    let shore = near_lake * (1.0 - smoothstep(wl + 0.6, wl + 1.8, g + fine * 0.35));
    let bed = 1.0 - smoothstep(wl - 0.5, wl - 0.05, g);
    let slope = 1.0 - up;
    let high = smoothstep(
        terrain.base_height + terrain.hill_height * 0.5,
        terrain.base_height + terrain.hill_height * 0.68,
        g + fine * 1.2 + patches * 1.5,
    );
    let rock = max(smoothstep(0.2, 0.34, slope + fine * 0.05), high);
    let rest = (1.0 - bed) * (1.0 - shore);
    let grassy = rest * (1.0 - rock);
    var out: GroundMix;
    out.w = array<f32, 5>(
        grassy * (1.0 - meadow),
        grassy * meadow,
        shore * (1.0 - bed),
        rest * rock,
        bed,
    );
    out.patches = patches;
    return out;
}

fn ground_surface(in: VsOut, geo: vec3<f32>, d: Grads) -> LayerSample {
    let mix5 = ground_mix(in.world_p.xz, in.ground, geo.y);
    var w = mix5.w;
    let patches = mix5.patches;
    let far = smoothstep(25.0, 90.0, distance(frame.eye, in.world_p));
    var s: array<LayerSample, 5>;
    var best = 0.0;
    for (var i = 0; i < 5; i++) {
        if w[i] > 0.002 {
            s[i] = sample_layer(i, in, geo, far, d);
            best = max(best, w[i] + s[i].orh.z * 0.5);
        }
    }
    var out: LayerSample;
    out.albedo = vec3<f32>(0.0);
    out.n = vec3<f32>(0.0);
    out.orh = vec3<f32>(0.0);
    var total = 0.0;
    for (var i = 0; i < 5; i++) {
        if w[i] > 0.002 {
            let b = max(w[i] + s[i].orh.z * 0.5 - best + 0.18, 0.0);
            out.albedo += s[i].albedo * b;
            out.n += s[i].n * b;
            out.orh += s[i].orh * b;
            total += b;
        }
    }
    let inv = 1.0 / max(total, 1e-5);
    out.albedo *= inv;
    out.n = normalize(out.n + geo * 1e-4);
    out.orh *= inv;
    // Broad brightness drift so repeats do not line up in the distance.
    out.albedo *= 0.88 + 0.24 * (patches * 0.5 + 0.5);
    return out;
}

@fragment
fn fs_land(in: VsOut) -> @location(0) vec4<f32> {
    // Texture gradients, taken while control flow is still uniform.
    var grads: Grads;
    grads.x = dpdx(in.world_p);
    grads.y = dpdy(in.world_p);
    let n = normalize(in.world_n);
    let l = normalize(frame.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.5 + 0.5;
    let vis = sun_visibility(in.world_p, n, frame.eye);
    let v = normalize(frame.eye - in.world_p);
    // Full-strength sun as before; the sky dome now tints the shade.
    let sun_diffuse = wrap * wrap / max(1.0 - frame.ambient, 0.05);
    var light = frame_light();
    // sky_up.w carries this view's AO strength.
    light.ao = screen_ao(frame.view_proj * vec4(in.world_p, 1.0), frame.sky_up.w);
    light.gi = gi_irradiance(in.world_p, n);
    if layers.inv_tile_b.y > 0.5 && in.wet < 0.5 {
        let surf = ground_surface(in, n, grads);
        light.ao *= surf.orh.x;
        let lit = pbr_shade(surf.albedo, surf.n, v, light, surf.orh.y, 0.0, vis, sun_diffuse)
            + surf.albedo * torch_light(in.world_p, surf.n);
        return vec4(lit, 1.0);
    }
    let lit = pbr_shade(in.color.rgb, n, v, light, 0.92, 0.0, vis, sun_diffuse)
        + in.color.rgb * torch_light(in.world_p, n);
    return vec4(lit, 1.0);
}

struct WaterVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_p: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) wet: f32,
};

@vertex
fn vs_water(v: VsIn) -> WaterVsOut {
    let xz = ring.center + v.local_xz * ring.extent;
    let h = sample_field(xz);
    // Sheet sits on water_top from the same sample that decided wetness (rim marriage).
    let y = h.water_top + WATER_CLEARANCE * 0.5;
    let world = vec3(xz.x, y, xz.y);
    var out: WaterVsOut;
    out.clip = frame.view_proj * vec4(world, 1.0);
    out.world_p = world;
    out.color = terrain.water;
    out.wet = h.water;
    return out;
}

@fragment
fn fs_water(in: WaterVsOut) -> @location(0) vec4<f32> {
    if in.wet < 0.5 {
        discard;
    }
    let n = vec3(0.0, 1.0, 0.0);
    let l = normalize(frame.light_dir);
    let view = normalize(frame.eye - in.world_p);
    let fresnel = pow(1.0 - max(dot(n, view), 0.0), 2.0);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.5 + 0.5;
    let vis = sun_visibility(in.world_p, n, frame.eye);
    let lit = in.color.rgb * (frame.ambient + wrap * wrap * vis * frame.light_color)
        + in.color.rgb * torch_light(in.world_p, n);
    var alpha = in.color.a;
    alpha = mix(alpha, min(alpha + 0.18, 0.55), fresnel * 0.65);
    return vec4(lit, alpha);
}
"#;

struct RingGpu {
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// World cell size for this ring (coarser rings double each level).
    #[allow(dead_code)]
    cell: f32,
    extent: f32,
    /// Index range into the shared index buffer (full or annulus).
    index_start: u32,
    index_count: u32,
}

pub struct ClipmapRenderer {
    land_pipeline: wgpu::RenderPipeline,
    land_pipeline_portal: wgpu::RenderPipeline,
    land_depth_prepass: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    water_pipeline_portal: wgpu::RenderPipeline,
    grass_pipeline: wgpu::RenderPipeline,
    grass_buf: wgpu::Buffer,
    /// Blades to draw this frame (0: no grass).
    grass_blades: u32,
    last_grass: Option<GrassUniform>,
    frame_buf: wgpu::Buffer,
    terrain_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    rings: Vec<RingGpu>,
    config: ClipmapConfig,
    last_frame: Option<FrameUniform>,
    last_params: Option<TerrainParamsUniform>,
    last_ring_center: Option<[f32; 2]>,
    shadow_layout: wgpu::BindGroupLayout,
    layer_layout: wgpu::BindGroupLayout,
    layer_bind: wgpu::BindGroup,
    /// Which layers `layer_bind` holds (`None` until the first `ensure_layers`).
    layers_for: Option<Option<TerrainLayers>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LayerParamsUniform {
    inv_tile_a: [f32; 4],
    inv_tile_b: [f32; 4],
    normal_scale_a: [f32; 4],
    normal_scale_b: [f32; 4],
}

fn layer_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let tex = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clipmap-layer-layout"),
        entries: &[
            tex(0),
            tex(1),
            tex(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// One texture array per map kind (albedo, normal + height, ORM), a layer
/// per ground layer, with full mip chains. `None` binds 1x1 placeholders and
/// switches texturing off in the shader.
fn build_layer_bind(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    layers: Option<TerrainLayers>,
) -> wgpu::BindGroup {
    use super::material::{mip_chain, MapKind};
    use crate::mesh::AlbedoMap;
    let sets: Vec<std::sync::Arc<crate::texgen::TextureLayer>> = match layers {
        Some(l) => {
            let all = l.as_array();
            let list: Vec<crate::texgen::Layer> = all.iter().map(|&n| n.into()).collect();
            crate::texgen::prepare(&list);
            all.iter().map(|&n| crate::texgen::layer(n)).collect()
        }
        None => Vec::new(),
    };
    let size = sets.first().map_or(1, |t| t.size);
    let count = sets.len().max(1) as u32;
    let make =
        |kind: MapKind, pick: &dyn Fn(&crate::texgen::TextureLayer) -> Vec<u8>, flat: [u8; 4]| {
            let maps: Vec<AlbedoMap> = if sets.is_empty() {
                vec![AlbedoMap {
                    width: 1,
                    height: 1,
                    rgba: flat.to_vec(),
                }]
            } else {
                sets.iter()
                    .map(|t| AlbedoMap {
                        width: size,
                        height: size,
                        rgba: pick(t),
                    })
                    .collect()
            };
            let chains: Vec<_> = maps.iter().map(|m| mip_chain(m, kind)).collect();
            let format = if kind == MapKind::Color {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("clipmap-ground-layers"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: count,
                },
                mip_level_count: chains[0].len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for (layer, chain) in chains.iter().enumerate() {
                for (level, (w, h, data)) in chain.iter().enumerate() {
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: level as u32,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: 0,
                                z: layer as u32,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        data,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(w * 4),
                            rows_per_image: Some(*h),
                        },
                        wgpu::Extent3d {
                            width: *w,
                            height: *h,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
    let albedo = make(MapKind::Color, &|t| t.albedo.clone(), [255, 255, 255, 255]);
    let normal = make(MapKind::Normal, &|t| t.normal.clone(), [128, 128, 255, 255]);
    let orm = make(MapKind::Data, &|t| t.orm.clone(), [255, 235, 0, 255]);
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("clipmap-layer-sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    });
    let mut p = LayerParamsUniform {
        inv_tile_a: [1.0; 4],
        inv_tile_b: [1.0, 0.0, 0.0, 0.0],
        normal_scale_a: [1.0; 4],
        normal_scale_b: [1.0, 0.0, 0.0, 0.0],
    };
    if !sets.is_empty() {
        for (i, t) in sets.iter().enumerate() {
            let inv = 1.0 / t.tile_metres.max(0.01);
            if i < 4 {
                p.inv_tile_a[i] = inv;
                p.normal_scale_a[i] = t.normal_scale;
            } else {
                p.inv_tile_b[0] = inv;
                p.normal_scale_b[0] = t.normal_scale;
            }
        }
        p.inv_tile_b[1] = 1.0;
    }
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("clipmap-layer-params"),
        contents: bytemuck::bytes_of(&p),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clipmap-layer-bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&albedo),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&normal),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&orm),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

impl ClipmapRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        config: ClipmapConfig,
        shadow_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clipmap-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}{SHADER}{}{}{}{}",
                    super::pipeline::LIGHTING_WGSL,
                    super::shadow::SHADOW_UNIFORMS_WGSL,
                    super::shadow::CLIPMAP_SHADOW_WGSL,
                    super::shadow::SHADOW_EVAL_WGSL,
                    super::grass::GRASS_WGSL
                )
                .into(),
            ),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clipmap-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layer_layout = layer_bind_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clipmap-pipeline-layout"),
            bind_group_layouts: &[&bind_layout, shadow_layout, &layer_layout],
            push_constant_ranges: &[],
        });

        let vertex_attrs = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }];

        let make_land = |label: &str, depth_stencil: wgpu::DepthStencilState, shaded: bool| {
            let color_targets = [Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ClipVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: shaded.then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_land"),
                    targets: &color_targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(depth_stencil),
                multisample: super::stencil::scene_multisample(false),
                multiview: None,
                cache: None,
            })
        };

        let land_pipeline = make_land(
            "clipmap-land",
            super::stencil::scene_depth_stencil_unmasked(),
            true,
        );
        let land_pipeline_portal = make_land(
            "clipmap-land-portal",
            super::stencil::scene_depth_stencil_masked(),
            true,
        );
        let land_depth_prepass = make_land(
            "clipmap-land-depth-prepass",
            super::stencil::scene_depth_stencil_unmasked(),
            false,
        );

        let make_water = |label: &str, depth_stencil: wgpu::DepthStencilState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_water"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ClipVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_water"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let water_pipeline = make_water(
            "clipmap-water",
            super::stencil::scene_depth_stencil_unmasked_write(false),
        );
        let water_pipeline_portal = make_water(
            "clipmap-water-portal",
            super::stencil::scene_depth_stencil_masked_write(false),
        );

        let grass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clipmap-grass"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_grass"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_grass"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(super::stencil::scene_depth_stencil_unmasked()),
            multisample: super::stencil::scene_multisample(false),
            multiview: None,
            cache: None,
        });
        let grass_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clipmap-grass"),
            size: std::mem::size_of::<GrassUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clipmap-frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clipmap-terrain"),
            size: std::mem::size_of::<TerrainParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (vertex_buf, index_buf, full_range, annulus_range) =
            build_grid(device, config.resolution.max(8));

        let rings = build_rings(
            device,
            &bind_layout,
            &frame_buf,
            &terrain_buf,
            &grass_buf,
            &config,
            full_range,
            annulus_range,
        );

        Self {
            land_pipeline,
            land_pipeline_portal,
            land_depth_prepass,
            water_pipeline,
            water_pipeline_portal,
            grass_pipeline,
            grass_buf,
            grass_blades: 0,
            last_grass: None,
            frame_buf,
            terrain_buf,
            vertex_buf,
            index_buf,
            rings,
            config,
            last_frame: None,
            last_params: None,
            last_ring_center: None,
            shadow_layout: shadow_layout.clone(),
            layer_bind: build_layer_bind(device, queue, &layer_layout, None),
            layer_layout,
            layers_for: None,
        }
    }

    /// Upload the ground texture layers when they change.
    pub fn ensure_layers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layers: Option<TerrainLayers>,
    ) {
        if self.layers_for == Some(layers) {
            return;
        }
        self.layer_bind = build_layer_bind(device, queue, &self.layer_layout, layers);
        self.layers_for = Some(layers);
    }

    pub fn ensure_config(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        config: &ClipmapConfig,
    ) {
        if config.rings == self.config.rings
            && config.resolution == self.config.resolution
            && (config.cell_size - self.config.cell_size).abs() < 1e-6
        {
            return;
        }
        *self = Self::new(
            device,
            queue,
            format,
            config.clone(),
            &self.shadow_layout.clone(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        light_dir: Vec3,
        ambient: f32,
        light_color: Vec3,
        eye: Vec3,
        torch_color: Vec3,
        torch_radius_m: f32,
        torch_curve: f32,
        sky: super::SkyLight,
        ao_strength: f32,
        proc: &ProcTerrain,
        grass: GrassFrame,
    ) {
        let (grass_u, blades) = match (&proc.grass, grass.scale > 0.0) {
            (Some(g), true) => GrassUniform::around(eye, g, grass.scale, grass.time, grass.wind),
            _ => (GrassUniform::default(), 0),
        };
        self.grass_blades = blades;
        if blades > 0 && self.last_grass != Some(grass_u) {
            queue.write_buffer(&self.grass_buf, 0, bytemuck::bytes_of(&grass_u));
            self.last_grass = Some(grass_u);
        }
        let pad = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
        let frame = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z],
            ambient,
            light_color: [light_color.x, light_color.y, light_color.z],
            _pad: 0.0,
            eye: [eye.x, eye.y, eye.z],
            torch_radius_m,
            torch_curve,
            _pad_torch: [0.0; 3],
            torch_color: [torch_color.x, torch_color.y, torch_color.z, 0.0],
            sky_up: [sky.up[0], sky.up[1], sky.up[2], ao_strength],
            sky_horizon: pad(sky.horizon),
            sky_down: pad(sky.down),
        };
        if self.last_frame.as_ref().map(bytemuck::bytes_of) != Some(bytemuck::bytes_of(&frame)) {
            queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
            self.last_frame = Some(frame);
        }

        let params = TerrainParamsUniform::from_rules(&proc.rules);
        if self.last_params.as_ref().map(bytemuck::bytes_of) != Some(bytemuck::bytes_of(&params)) {
            queue.write_buffer(&self.terrain_buf, 0, bytemuck::bytes_of(&params));
            self.last_params = Some(params);
        }

        // Snap every ring to the finest cell so nested holes stay aligned.
        // Per-ring cell snaps drift apart and let coarser (higher) facets win
        // depth over the fine ring — the walker then appears to sink through.
        let fine_cell = self.config.cell_size.max(1e-4);
        let snapped_x = (proc.focus.x / fine_cell).floor() * fine_cell;
        let snapped_z = (proc.focus.z / fine_cell).floor() * fine_cell;
        let center = [snapped_x, snapped_z];
        if self.last_ring_center != Some(center) {
            self.last_ring_center = Some(center);
            for ring in &self.rings {
                let u = RingUniform {
                    center,
                    extent: ring.extent,
                    _pad: 0.0,
                };
                queue.write_buffer(&ring.uniform_buf, 0, bytemuck::bytes_of(&u));
            }
        }
    }

    pub fn draw_land(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        shadow_bind: &wgpu::BindGroup,
        portal_mask: bool,
    ) {
        self.draw_land_with(
            pass,
            shadow_bind,
            if portal_mask {
                &self.land_pipeline_portal
            } else {
                &self.land_pipeline
            },
        );
    }

    /// Lay the land's depth for the depth prepass.
    pub fn draw_land_depth(&self, pass: &mut wgpu::RenderPass<'_>, shadow_bind: &wgpu::BindGroup) {
        self.draw_land_with(pass, shadow_bind, &self.land_depth_prepass);
    }

    fn draw_land_with(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        shadow_bind: &wgpu::BindGroup,
        pipeline: &wgpu::RenderPipeline,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, shadow_bind, &[]);
        pass.set_bind_group(2, &self.layer_bind, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        // Coarse → fine so finer wins depth.
        for ring in self.rings.iter().rev() {
            pass.set_bind_group(0, &ring.bind_group, &[]);
            let end = ring.index_start + ring.index_count;
            pass.draw_indexed(ring.index_start..end, 0, 0..1);
        }
    }

    /// Grass blades on the land (main pass only, after the land).
    pub fn draw_grass(&self, pass: &mut wgpu::RenderPass<'_>, shadow_bind: &wgpu::BindGroup) {
        let Some(ring) = self.rings.first() else {
            return;
        };
        if self.grass_blades == 0 {
            return;
        }
        pass.set_pipeline(&self.grass_pipeline);
        pass.set_bind_group(0, &ring.bind_group, &[]);
        pass.set_bind_group(1, shadow_bind, &[]);
        pass.set_bind_group(2, &self.layer_bind, &[]);
        pass.draw(0..BLADE_VERTICES, 0..self.grass_blades);
    }

    pub fn draw_water(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        shadow_bind: &wgpu::BindGroup,
        portal_mask: bool,
    ) {
        pass.set_pipeline(if portal_mask {
            &self.water_pipeline_portal
        } else {
            &self.water_pipeline
        });
        pass.set_bind_group(1, shadow_bind, &[]);
        pass.set_bind_group(2, &self.layer_bind, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        for ring in self.rings.iter().rev() {
            pass.set_bind_group(0, &ring.bind_group, &[]);
            let end = ring.index_start + ring.index_count;
            pass.draw_indexed(ring.index_start..end, 0, 0..1);
        }
    }
}

type IndexRange = (u32, u32); // start, count

fn build_grid(
    device: &wgpu::Device,
    resolution: u32,
) -> (wgpu::Buffer, wgpu::Buffer, IndexRange, IndexRange) {
    let res = resolution.max(8);
    let mut verts = Vec::with_capacity(((res + 1) * (res + 1)) as usize);
    for iz in 0..=res {
        for ix in 0..=res {
            let u = ix as f32 / res as f32 - 0.5;
            let v = iz as f32 / res as f32 - 0.5;
            verts.push(ClipVertex { pos: [u, v] });
        }
    }

    let mut full = Vec::new();
    let mut annulus = Vec::new();
    let stride = res + 1;
    let inner0 = res / 4;
    let inner1 = res - inner0;
    for iz in 0..res {
        for ix in 0..res {
            let i00 = iz * stride + ix;
            let i10 = i00 + 1;
            let i01 = i00 + stride;
            let i11 = i01 + 1;
            let tri = [i00, i01, i11, i00, i11, i10];
            full.extend(tri);
            // Outer rings skip the central quarter (covered by the finer ring).
            let in_hole = ix >= inner0 && ix < inner1 && iz >= inner0 && iz < inner1;
            if !in_hole {
                annulus.extend(tri);
            }
        }
    }

    let full_range = (0u32, full.len() as u32);
    let annulus_start = full.len() as u32;
    let mut indices = full;
    let annulus_count = annulus.len() as u32;
    indices.extend(annulus);

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("clipmap-verts"),
        contents: bytemuck::cast_slice(&verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("clipmap-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (
        vertex_buf,
        index_buf,
        full_range,
        (annulus_start, annulus_count),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_rings(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    frame_buf: &wgpu::Buffer,
    terrain_buf: &wgpu::Buffer,
    grass_buf: &wgpu::Buffer,
    config: &ClipmapConfig,
    full_range: IndexRange,
    annulus_range: IndexRange,
) -> Vec<RingGpu> {
    let rings_n = config.rings.clamp(1, 6);
    let res = config.resolution.max(8) as f32;
    let mut out = Vec::with_capacity(rings_n as usize);
    for i in 0..rings_n {
        let cell = config.cell_size * 2f32.powi(i as i32);
        let extent = cell * res; // full width
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clipmap-ring"),
            size: std::mem::size_of::<RingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clipmap-ring-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: terrain_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grass_buf.as_entire_binding(),
                },
            ],
        });
        let (index_start, index_count) = if i == 0 { full_range } else { annulus_range };
        out.push(RingGpu {
            uniform_buf,
            bind_group,
            cell,
            extent,
            index_start,
            index_count,
        });
    }
    out
}
