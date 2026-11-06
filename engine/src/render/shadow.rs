//! Hybrid sun shadows: height-field raymarch for terrain, cascaded depth for meshes.
//!
//! Land never enters the depth map. Nearby props and characters do. Shadow depth
//! is conventional 0–1 (`Less`, clear to 1), separate from the scene's reversed-Z.

use crate::anim::MAX_JOINTS;
use crate::contact::ContactSnapshot;
use crate::mesh::{InstanceRaw, Vertex};
use crate::space::{RenderOrigin, RenderPosition};
use crate::terrain::TerrainRules;
use crate::world::{ShadowSettings, SurfaceMaterialRef, World};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use super::skinned::SkinnedVertex;
use super::terrain_shadow::{GroundSource, HorizonMap};

const _: () = assert!(MAX_JOINTS == 128);

pub const CASCADE_COUNT: usize = 3;

const FLAG_CSM: u32 = 1;
const FLAG_ATLAS: u32 = 2;
const FLAG_FORMULA: u32 = 4;
/// The horizon-height map (`terrain_shadow`) covers `atlas_origin_xz`/`atlas_extent`.
const FLAG_HORIZON: u32 = 8;
/// Light probes from `gi.rs` cover `gi_origin` .. `gi_origin + gi_dims * gi_spacing`.
const FLAG_GI: u32 = 16;
const MISSING_HEIGHT: f32 = -10_000.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub light_vp0: [[f32; 4]; 4],
    pub light_vp1: [[f32; 4]; 4],
    pub light_vp2: [[f32; 4]; 4],
    pub cascade_end: [f32; 4],
    pub atlas_origin_xz: [f32; 2],
    pub atlas_extent: f32,
    pub flags: u32,
    pub light_dir: [f32; 3],
    pub map_texel: f32,
    pub seed: u32,
    pub _pad0: u32,
    pub base_height: f32,
    pub hill_height: f32,
    pub hill_scale: f32,
    pub lake_scale: f32,
    pub lake_threshold: f32,
    pub water_level: f32,
    pub gi_origin: [f32; 3],
    pub gi_spacing: f32,
    pub gi_dims: [u32; 3],
    pub _pad1: u32,
}

/// Texel-snapped ortho looking from the sun toward `focus`.
///
/// `focus` is snapped onto the light-space texel grid first so a walk that stays
/// inside one texel does not move the cascade.
pub fn cascade_view_proj(focus: Vec3, light_dir: Vec3, radius: f32, map_size: u32) -> Mat4 {
    let mut light = light_dir.normalize_or_zero();
    if light.length_squared() < 1e-12 {
        light = Vec3::Y;
    }
    let r = radius.max(0.5);
    let back = r * 2.0;
    let up = if light.y.abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view_rot = Mat4::look_at_rh(light * back, Vec3::ZERO, up);
    let ls = view_rot.transform_point3(focus);
    let texel = (2.0 * r) / map_size.max(1) as f32;
    let snapped_ls = Vec3::new(
        (ls.x / texel).round() * texel,
        (ls.y / texel).round() * texel,
        ls.z,
    );
    let snapped_focus = view_rot.inverse().transform_point3(snapped_ls);
    let view = Mat4::look_at_rh(snapped_focus + light * back, snapped_focus, up);
    let proj = Mat4::orthographic_rh(-r, r, -r, r, 0.5, back + r * 2.0);
    proj * view
}

/// Rasterise resident contact heights into a world-XZ atlas (render space).
///
/// Missing coverage is [`MISSING_HEIGHT`] so the shader march never hits empty sky.
pub fn fill_height_atlas(
    pixels: &mut [f32],
    size: u32,
    origin_xz: [f32; 2],
    extent: f32,
    render_origin: RenderOrigin,
    snapshot: &ContactSnapshot,
) {
    let size_n = size as usize;
    if pixels.len() != size_n * size_n {
        panic!("height atlas pixels len {} != {size}×{size}", pixels.len());
    }
    if size == 0 {
        panic!("height atlas size must be non-zero");
    }
    let inv = 1.0 / size as f32;
    for z in 0..size_n {
        for x in 0..size_n {
            let wx = origin_xz[0] + (x as f32 + 0.5) * inv * extent;
            let wz = origin_xz[1] + (z as f32 + 0.5) * inv * extent;
            let global = RenderPosition::at(wx, 0.0, wz)
                .to_global(render_origin)
                .horizontal();
            pixels[z * size_n + x] = snapshot.height_at(global).unwrap_or(MISSING_HEIGHT);
        }
    }
}

/// Terrain and water are height-marched, not depth-map casters.
pub fn material_casts_shadow(material: Option<SurfaceMaterialRef>) -> bool {
    !matches!(
        material,
        Some(SurfaceMaterialRef::Terrain(_) | SurfaceMaterialRef::Water(_))
    )
}

macro_rules! shadow_uniforms_wgsl {
    () => {
        r#"
struct ShadowUniforms {
    light_vp0: mat4x4<f32>,
    light_vp1: mat4x4<f32>,
    light_vp2: mat4x4<f32>,
    cascade_end: vec4<f32>,
    atlas_origin_xz: vec2<f32>,
    atlas_extent: f32,
    flags: u32,
    light_dir: vec3<f32>,
    map_texel: f32,
    seed: u32,
    _pad0: u32,
    base_height: f32,
    hill_height: f32,
    hill_scale: f32,
    lake_scale: f32,
    lake_threshold: f32,
    water_level: f32,
    gi_origin: vec3<f32>,
    gi_spacing: f32,
    gi_dims: vec3<u32>,
    _pad1: u32,
};
"#
    };
}
pub(crate) use shadow_uniforms_wgsl;

pub const SHADOW_UNIFORMS_WGSL: &str = shadow_uniforms_wgsl!();

macro_rules! terrain_formula_wgsl {
    () => {
        r#"
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

fn formula_ground(xz: vec2<f32>) -> f32 {
    let n = fbm2(xz * shadow.hill_scale, shadow.seed, 5u, 2.1, 0.5);
    let h_raw = shadow.base_height + shadow.hill_height * n;
    let lake = fbm2(
        xz * shadow.lake_scale + vec2(17.0, 9.0),
        shadow.seed ^ 0xC0FFEEu,
        3u,
        2.0,
        0.55,
    );
    let lake_t = lake * 0.5 + 0.5;
    let span = max(1.0 - shadow.lake_threshold, 1e-3);
    let basin = clamp((lake_t - shadow.lake_threshold) / span, 0.0, 1.0);
    let floor_h = max(h_raw, shadow.water_level);
    let carved = floor_h - basin * 3.5;
    let near_shore = floor_h <= shadow.water_level + 1.5;
    let in_basin = basin > 0.25 && near_shore;
    var ground = floor_h;
    if in_basin {
        let water_top = shadow.water_level;
        ground = min(carved, water_top - WATER_CLEARANCE - 0.001);
    }
    return ground;
}

"#
    };
}
pub(crate) use terrain_formula_wgsl;

pub const SCENE_SHADOW_WGSL: &str = concat!(
    r#"
@group(0) @binding(1) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(3) var shadow_samp: sampler_comparison;
@group(0) @binding(4) var height_atlas: texture_2d<f32>;
@group(0) @binding(5) var height_samp: sampler;
@group(0) @binding(6) var ao_tex: texture_2d<f32>;
@group(0) @binding(7) var ao_samp: sampler;
@group(0) @binding(8) var gi_probe_r: texture_3d<f32>;
@group(0) @binding(9) var gi_probe_g: texture_3d<f32>;
@group(0) @binding(10) var gi_probe_b: texture_3d<f32>;

"#,
    terrain_formula_wgsl!(),
    r#"
fn atlas_ground(xz: vec2<f32>) -> f32 {
    if shadow.atlas_extent <= 0.0 {
        return -10000.0;
    }
    let uv = (xz - shadow.atlas_origin_xz) / shadow.atlas_extent;
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
        return -10000.0;
    }
    return textureSampleLevel(height_atlas, height_samp, uv, 0.0).r;
}

fn shadow_ground(xz: vec2<f32>) -> f32 {
    if (shadow.flags & 4u) != 0u {
        return formula_ground(xz);
    }
    if (shadow.flags & 2u) != 0u {
        return atlas_ground(xz);
    }
    return -10000.0;
}
"#
);

pub const SHADOW_EVAL_WGSL: &str = r#"
// Pipelines drawing over surfaces they do not occlude (glass, water) set
// this false: the occlusion texture describes what lies behind them.
override SCENE_AO: bool = true;

// Sky and bounce light reaching a surface at `p` facing `n`, from the light
// probes, in the units of `sky_irradiance * ambient`. Alpha is how much to
// trust it: 0 outside the probe box or with no live probe around, fading to
// 0 toward the box edge.
fn gi_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    if (shadow.flags & 16u) == 0u {
        return vec4<f32>(0.0);
    }
    let dims = vec3<i32>(shadow.gi_dims);
    let size = vec3<f32>(dims);
    // Half a probe off the surface, so the probes behind it do not count.
    let g = (p + n * shadow.gi_spacing * 0.5 - shadow.gi_origin) / shadow.gi_spacing - vec3<f32>(0.5);
    if any(g < vec3<f32>(-0.5)) || any(g > size - vec3<f32>(0.5)) {
        return vec4<f32>(0.0);
    }
    let edge = min(min(g + vec3<f32>(0.5), size - vec3<f32>(0.5) - g), vec3<f32>(2.0));
    let fade = smoothstep(0.0, 2.0, min(edge.x, min(edge.y, edge.z)));
    let base = vec3<i32>(floor(g));
    let f = g - floor(g);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0; i < 8; i++) {
        let o = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let c = clamp(base + o, vec3<i32>(0), dims - vec3<i32>(1));
        let r = textureLoad(gi_probe_r, c, 0);
        if r.x < 0.0 {
            continue;
        }
        let gr = textureLoad(gi_probe_g, c, 0);
        let b = textureLoad(gi_probe_b, c, 0);
        let corner = vec3<f32>(o);
        let tri = mix(vec3<f32>(1.0) - f, f, corner);
        let probe_p = shadow.gi_origin + (vec3<f32>(c) + vec3<f32>(0.5)) * shadow.gi_spacing;
        let to_probe = normalize(probe_p - p + n * 1e-3);
        // Favour probes in front of the surface.
        let facing = (dot(to_probe, n) + 1.0) * 0.5;
        let w = tri.x * tri.y * tri.z * (facing * facing + 0.05);
        // Cosine-convolved linear SH: E/pi = mean + 2/3 lin . n.
        let e = vec3<f32>(
            r.x + dot(r.yzw, n) * (2.0 / 3.0),
            gr.x + dot(gr.yzw, n) * (2.0 / 3.0),
            b.x + dot(b.yzw, n) * (2.0 / 3.0),
        );
        sum += max(e, vec3<f32>(0.0)) * w;
        weight += w;
    }
    if weight < 1e-5 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum / weight, fade);
}

// Ambient occlusion where `clip` lands on screen; `strength` 0 switches it
// off for views the occlusion was not computed from.
fn screen_ao(clip: vec4<f32>, strength: f32) -> f32 {
    if !SCENE_AO || strength <= 0.0 || clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let ao = textureSampleLevel(ao_tex, ao_samp, uv, 0.0).r;
    return mix(1.0, ao, strength);
}

fn horizon_at(texel: vec2<i32>, dims: vec2<i32>) -> f32 {
    return textureLoad(height_atlas, clamp(texel, vec2<i32>(0), dims - vec2<i32>(1)), 0).r;
}

// Terrain shadow from the precomputed horizon-height map: lit above H(xz).
// Bilinear by hand because R32Float is not filterable everywhere.
fn height_visibility(world_p: vec3<f32>) -> f32 {
    if (shadow.flags & 8u) == 0u || shadow.atlas_extent <= 0.0 {
        return 1.0;
    }
    let uv = (world_p.xz - shadow.atlas_origin_xz) / shadow.atlas_extent;
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5));
    if edge >= 0.5 {
        return 1.0;
    }
    let dims = vec2<i32>(textureDimensions(height_atlas));
    let p = uv * vec2<f32>(dims) - vec2<f32>(0.5);
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let h00 = horizon_at(i, dims);
    let h10 = horizon_at(i + vec2<i32>(1, 0), dims);
    let h01 = horizon_at(i + vec2<i32>(0, 1), dims);
    let h11 = horizon_at(i + vec2<i32>(1, 1), dims);
    let h = mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    let lit = smoothstep(-0.25, 0.25, world_p.y + 0.12 - h);
    // Terrain self-shadow keeps some sun, as the old march did; fade out at
    // the region edge so the boundary never shows as a line.
    let vis = mix(0.42, 1.0, lit);
    return mix(vis, 1.0, smoothstep(0.45, 0.5, edge));
}

fn cascade_vp(layer: i32) -> mat4x4<f32> {
    if layer == 0 {
        return shadow.light_vp0;
    }
    if layer == 1 {
        return shadow.light_vp1;
    }
    return shadow.light_vp2;
}

// Sun visibility from one cascade: a 3x3 grid of bilinear depth compares
// spread over about six texels, so penumbrae are soft and free of the
// stair-steps and noise that fewer taps or random rotations give.
fn cascade_visibility(layer: i32, world_p: vec3<f32>, n: vec3<f32>) -> f32 {
    let bias_p = world_p + n * max(shadow.map_texel * 4.0, 0.04) * f32(layer + 1);
    let clip = cascade_vp(layer) * vec4(bias_p, 1.0);
    let ndc = clip.xyz / max(clip.w, 1e-6);
    let uv = vec2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let step = shadow.map_texel * 1.5;
    var s = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * step;
            s += textureSampleCompareLevel(shadow_map, shadow_samp, uv + offset, layer, ndc.z);
        }
    }
    return s / 9.0;
}

// Cascaded shadow lookup. The last tenth of each cascade blends into the
// next, and the last cascade fades out, so no boundary shows as a line.
fn csm_visibility(world_p: vec3<f32>, n: vec3<f32>, eye: vec3<f32>) -> f32 {
    let d = distance(world_p, eye);
    let ends = shadow.cascade_end.xyz;
    if ends.z > 0.0 && d >= ends.z {
        return 1.0;
    }
    var layer = 2;
    if d < ends.x {
        layer = 0;
    } else if d < ends.y {
        layer = 1;
    }
    let end = ends[layer];
    let start = select(ends[max(layer - 1, 0)], 0.0, layer == 0);
    let blend_from = end - (end - start) * 0.1;
    var vis = cascade_visibility(layer, world_p, n);
    if d > blend_from {
        let t = smoothstep(blend_from, end, d);
        var next = 1.0;
        if layer < 2 {
            next = cascade_visibility(layer + 1, world_p, n);
        }
        vis = mix(vis, next, t);
    }
    return vis;
}

fn sun_visibility(world_p: vec3<f32>, n: vec3<f32>, eye: vec3<f32>) -> f32 {
    var vis = 1.0;
    if (shadow.flags & 1u) != 0u {
        vis = min(vis, csm_visibility(world_p, n, eye));
    }
    if (shadow.flags & 8u) != 0u {
        vis = min(vis, height_visibility(world_p));
    }
    return vis;
}
"#;

pub const CLIPMAP_SHADOW_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> shadow: ShadowUniforms;
@group(1) @binding(1) var shadow_map: texture_depth_2d_array;
@group(1) @binding(2) var shadow_samp: sampler_comparison;
@group(1) @binding(3) var height_atlas: texture_2d<f32>;
@group(1) @binding(4) var height_samp: sampler;
@group(1) @binding(5) var ao_tex: texture_2d<f32>;
@group(1) @binding(6) var ao_samp: sampler;
@group(1) @binding(7) var gi_probe_r: texture_3d<f32>;
@group(1) @binding(8) var gi_probe_g: texture_3d<f32>;
@group(1) @binding(9) var gi_probe_b: texture_3d<f32>;

fn atlas_ground(xz: vec2<f32>) -> f32 {
    if shadow.atlas_extent <= 0.0 {
        return -10000.0;
    }
    let uv = (xz - shadow.atlas_origin_xz) / shadow.atlas_extent;
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 {
        return -10000.0;
    }
    return textureSampleLevel(height_atlas, height_samp, uv, 0.0).r;
}

fn shadow_ground(xz: vec2<f32>) -> f32 {
    if (shadow.flags & 4u) != 0u {
        return sample_field(xz).ground;
    }
    if (shadow.flags & 2u) != 0u {
        return atlas_ground(xz);
    }
    return -10000.0;
}
"#;

const MESH_DEPTH_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> cascade: mat4x4<f32>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    return cascade * (model * vec4<f32>(v.position, 1.0));
}
"#;

const SKINNED_DEPTH_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> cascade: mat4x4<f32>;

struct Joints {
    m: array<mat4x4<f32>, 128>,
};

@group(1) @binding(0) var<uniform> bones: Joints;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    var skin = mat4x4<f32>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
    );
    skin += bones.m[v.joints.x] * v.weights.x;
    skin += bones.m[v.joints.y] * v.weights.y;
    skin += bones.m[v.joints.z] * v.weights.z;
    skin += bones.m[v.joints.w] * v.weights.w;
    let world = model * (skin * vec4<f32>(v.position, 1.0));
    return cascade * world;
}
"#;

fn resource_layout_entries(base: u32) -> [wgpu::BindGroupLayoutEntry; 10] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: base,
            visibility: wgpu::ShaderStages::VERTEX
                | wgpu::ShaderStages::FRAGMENT
                | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 1,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 2,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 3,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 4,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
        // Screen-space ambient occlusion (see `ssao.rs`).
        wgpu::BindGroupLayoutEntry {
            binding: base + 5,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 6,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        // Light probes, red/green/blue linear SH (see `gi.rs`).
        gi_probe_entry(base + 7),
        gi_probe_entry(base + 8),
        gi_probe_entry(base + 9),
    ]
}

fn gi_probe_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

fn upload_atlas(queue: &wgpu::Queue, texture: &wgpu::Texture, pixels: &[f32], size: u32) {
    let row_bytes = size * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = row_bytes.div_ceil(align) * align;
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(padded_row),
        rows_per_image: Some(size),
    };
    let extent = wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    };
    let dest = wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    };
    if padded_row == row_bytes {
        queue.write_texture(dest, bytemuck::cast_slice(pixels), layout, extent);
        return;
    }
    let mut padded = vec![0u8; (padded_row * size) as usize];
    for y in 0..size as usize {
        let src = &pixels[y * size as usize..(y + 1) * size as usize];
        let dst = (y as u32 * padded_row) as usize;
        padded[dst..dst + row_bytes as usize].copy_from_slice(bytemuck::cast_slice(src));
    }
    queue.write_texture(dest, &padded, layout, extent);
}

/// Identity of the terrain formula, so the horizon map rebuilds when it changes.
fn formula_epoch(u: &ShadowUniforms) -> u64 {
    let words = [
        u.seed,
        u.base_height.to_bits(),
        u.hill_height.to_bits(),
        u.hill_scale.to_bits(),
        u.lake_scale.to_bits(),
        u.lake_threshold.to_bits(),
        u.water_level.to_bits(),
    ];
    words.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, w| {
        (h ^ u64::from(*w)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn apply_terrain_rules(u: &mut ShadowUniforms, rules: &TerrainRules) {
    u.seed = rules.seed;
    u.base_height = rules.base_height;
    u.hill_height = rules.hill_height;
    u.hill_scale = rules.hill_scale;
    u.lake_scale = rules.lake_scale;
    u.lake_threshold = rules.lake_threshold;
    u.water_level = rules.water_level;
}

struct ResourceViews<'a> {
    gi: &'a [wgpu::TextureView; 3],
    uniform: &'a wgpu::Buffer,
    map: &'a wgpu::TextureView,
    comparison: &'a wgpu::Sampler,
    atlas: &'a wgpu::TextureView,
    atlas_sampler: &'a wgpu::Sampler,
    ao: &'a wgpu::TextureView,
    ao_sampler: &'a wgpu::Sampler,
}

fn create_resource_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    v: ResourceViews<'_>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow-resource-bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: v.uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(v.map),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(v.comparison),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(v.atlas),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(v.atlas_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(v.ao),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(v.ao_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&v.gi[0]),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&v.gi[1]),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&v.gi[2]),
            },
        ],
    })
}

pub struct ShadowGpu {
    pub uniform_buf: wgpu::Buffer,
    pub resource_layout: wgpu::BindGroupLayout,
    pub resource_bind: wgpu::BindGroup,
    #[allow(dead_code)]
    pub cascade_layout: wgpu::BindGroupLayout,
    pub cascade_binds: [wgpu::BindGroup; CASCADE_COUNT],
    pub mesh_pipeline: wgpu::RenderPipeline,
    pub skinned_pipeline: wgpu::RenderPipeline,
    pub layer_views: [wgpu::TextureView; CASCADE_COUNT],
    #[allow(dead_code)]
    depth_texture: wgpu::Texture,
    #[allow(dead_code)]
    cascade_bufs: [wgpu::Buffer; CASCADE_COUNT],
    map_array_view: wgpu::TextureView,
    horizon: HorizonMap,
    comparison_sampler: wgpu::Sampler,
    atlas_sampler: wgpu::Sampler,
    ao_view: wgpu::TextureView,
    ao_sampler: wgpu::Sampler,
    gi_views: [wgpu::TextureView; 3],
    /// Probe box for this frame, from the renderer; None switches GI off.
    gi: Option<super::gi::GiPlacement>,
    map_size: u32,
    atlas_size: u32,
    atlas_pixels: Vec<f32>,
    atlas_origin_xz: [f32; 2],
    atlas_extent: f32,
    last_contact_epoch: u64,
    last_origin: RenderOrigin,
    last_focus_xz: [f32; 2],
    atlas_wrote: bool,
    /// Bumped on every atlas refill so the horizon map knows to rebuild.
    atlas_generation: u64,
    atlas_max_ground: f32,
}

impl ShadowGpu {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: ShadowSettings,
        joint_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let map_size = settings.map_size.max(1);
        let atlas_size = settings.atlas_size.max(1);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-depth-array"),
            size: wgpu::Extent3d {
                width: map_size,
                height: map_size,
                depth_or_array_layers: CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let map_array_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-map-array"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(CASCADE_COUNT as u32),
            usage: None,
        });
        let layer_views = std::array::from_fn(|i| {
            depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("shadow-cascade-layer"),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                usage: None,
            })
        });

        let atlas_pixels = vec![MISSING_HEIGHT; atlas_size as usize * atlas_size as usize];

        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-comparison"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::Less),
            ..Default::default()
        });
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-atlas-nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        let uniforms = ShadowUniforms::zeroed();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let horizon = HorizonMap::new(device, &uniform_buf, atlas_size);
        upload_atlas(queue, &horizon.ground_texture, &atlas_pixels, atlas_size);
        let atlas_view = &horizon.view;

        // No occlusion until the renderer hands over its SSAO target.
        let ao_view = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("ssao-none"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            )
            .create_view(&wgpu::TextureViewDescriptor::default());
        let ao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // No probes until the renderer hands over the GI volume.
        let gi_views: [wgpu::TextureView; 3] = std::array::from_fn(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("gi-none"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let resource_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-resource-layout"),
            entries: &resource_layout_entries(0),
        });
        let resource_bind = create_resource_bind(
            device,
            &resource_layout,
            ResourceViews {
                uniform: &uniform_buf,
                map: &map_array_view,
                comparison: &comparison_sampler,
                atlas: atlas_view,
                atlas_sampler: &atlas_sampler,
                ao: &ao_view,
                ao_sampler: &ao_sampler,
                gi: &gi_views,
            },
        );

        let cascade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-cascade-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let ident = Mat4::IDENTITY.to_cols_array_2d();
        let cascade_bufs = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(if i == 0 {
                    "shadow-cascade-0"
                } else if i == 1 {
                    "shadow-cascade-1"
                } else {
                    "shadow-cascade-2"
                }),
                contents: bytemuck::bytes_of(&ident),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let cascade_binds = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow-cascade-bind"),
                layout: &cascade_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cascade_bufs[i].as_entire_binding(),
                }],
            })
        });

        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-mesh-depth"),
            source: wgpu::ShaderSource::Wgsl(MESH_DEPTH_WGSL.into()),
        });
        let skinned_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-skinned-depth"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_DEPTH_WGSL.into()),
        });

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-mesh-depth-layout"),
            bind_group_layouts: &[&cascade_layout],
            push_constant_ranges: &[],
        });
        let skinned_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow-skinned-depth-layout"),
                bind_group_layouts: &[&cascade_layout, joint_layout],
                push_constant_ranges: &[],
            });

        let depth_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        };

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow-mesh-depth"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let skinned_instance = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                6 => Float32x4,
                7 => Float32x4,
                8 => Float32x4,
                9 => Float32x4,
            ],
        };
        let skinned_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow-skinned-depth"),
            layout: Some(&skinned_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skinned_shader,
                entry_point: Some("vs_main"),
                buffers: &[SkinnedVertex::LAYOUT, skinned_instance],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            uniform_buf,
            resource_layout,
            resource_bind,
            cascade_layout,
            cascade_binds,
            mesh_pipeline,
            skinned_pipeline,
            layer_views,
            depth_texture,
            cascade_bufs,
            map_array_view,
            horizon,
            comparison_sampler,
            atlas_sampler,
            ao_view,
            ao_sampler,
            gi_views,
            gi: None,
            map_size,
            atlas_size,
            atlas_pixels,
            atlas_origin_xz: [0.0, 0.0],
            atlas_extent: settings.atlas_extent_m,
            last_contact_epoch: u64::MAX,
            last_origin: RenderOrigin::default(),
            last_focus_xz: [f32::MAX, f32::MAX],
            atlas_wrote: false,
            atlas_generation: 0,
            atlas_max_ground: MISSING_HEIGHT,
        }
    }

    /// Rebuild the terrain horizon map if the sun, region or terrain changed.
    /// Returns whether a rebuild was encoded.
    pub fn encode_horizon(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        self.horizon.encode(encoder)
    }

    pub fn atlas_wrote(&self) -> bool {
        self.atlas_wrote
    }

    pub fn scene_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend(resource_layout_entries(1));
        entries
    }

    /// Point the lit shaders at a new occlusion texture. Bind groups made from
    /// [`Self::scene_bind_entries`] must be rebuilt afterwards.
    pub fn set_ao_view(&mut self, device: &wgpu::Device, view: wgpu::TextureView) {
        self.ao_view = view;
        self.rebuild_resource_bind(device);
    }

    /// Point the lit shaders at the GI probe textures. Bind groups made from
    /// [`Self::scene_bind_entries`] must be rebuilt afterwards.
    pub fn set_gi_views(&mut self, device: &wgpu::Device, views: [wgpu::TextureView; 3]) {
        self.gi_views = views;
        self.rebuild_resource_bind(device);
    }

    /// This frame's probe box, or None when GI is off.
    pub fn set_gi(&mut self, placement: Option<super::gi::GiPlacement>) {
        self.gi = placement;
    }

    fn rebuild_resource_bind(&mut self, device: &wgpu::Device) {
        self.resource_bind = create_resource_bind(
            device,
            &self.resource_layout,
            ResourceViews {
                uniform: &self.uniform_buf,
                map: &self.map_array_view,
                comparison: &self.comparison_sampler,
                atlas: &self.horizon.view,
                atlas_sampler: &self.atlas_sampler,
                ao: &self.ao_view,
                ao_sampler: &self.ao_sampler,
                gi: &self.gi_views,
            },
        );
    }

    pub fn scene_bind_entries<'a>(
        &'a self,
        scene_uniforms: wgpu::BindingResource<'a>,
    ) -> [wgpu::BindGroupEntry<'a>; 11] {
        [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniforms,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&self.map_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&self.comparison_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&self.horizon.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&self.ao_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&self.ao_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&self.gi_views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&self.gi_views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(&self.gi_views[2]),
            },
        ]
    }

    pub fn prepare(&mut self, queue: &wgpu::Queue, world: &World) -> [Mat4; CASCADE_COUNT] {
        self.atlas_wrote = false;
        let mut uniforms = ShadowUniforms::zeroed();
        let light = world.light().direction.normalize_or_zero();
        uniforms.light_dir = [light.x, light.y, light.z];
        uniforms.map_texel = 1.0 / self.map_size as f32;
        if let Some(gi) = self.gi {
            uniforms.flags |= FLAG_GI;
            uniforms.gi_origin = gi.origin.into();
            uniforms.gi_spacing = gi.spacing;
            uniforms.gi_dims = gi.dims.into();
        }

        let Some(settings) = world.shadows() else {
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
            let ident = Mat4::IDENTITY.to_cols_array_2d();
            for buf in &self.cascade_bufs {
                queue.write_buffer(buf, 0, bytemuck::bytes_of(&ident));
            }
            return [Mat4::IDENTITY; CASCADE_COUNT];
        };

        uniforms.flags |= FLAG_CSM;
        let focus = world.camera().target();
        let ends = settings.cascade_end_m;
        uniforms.cascade_end = [ends[0], ends[1], ends[2], 0.0];
        let mats = [
            cascade_view_proj(focus, light, ends[0], self.map_size),
            cascade_view_proj(focus, light, ends[1], self.map_size),
            cascade_view_proj(focus, light, ends[2], self.map_size),
        ];
        uniforms.light_vp0 = mats[0].to_cols_array_2d();
        uniforms.light_vp1 = mats[1].to_cols_array_2d();
        uniforms.light_vp2 = mats[2].to_cols_array_2d();
        for (buf, m) in self.cascade_bufs.iter().zip(mats) {
            queue.write_buffer(buf, 0, bytemuck::bytes_of(&m.to_cols_array_2d()));
        }

        if settings.raymarch_height {
            if let Some(rules) = world
                .proc_terrain()
                .map(|p| &p.rules)
                .or_else(|| world.height_field().map(|h| h.rules()))
            {
                uniforms.flags |= FLAG_FORMULA | FLAG_HORIZON;
                apply_terrain_rules(&mut uniforms, rules);
                let max_ground =
                    (rules.base_height + rules.hill_height.abs()).max(rules.water_level);
                self.horizon.prepare(
                    queue,
                    focus,
                    settings.atlas_extent_m,
                    None,
                    light,
                    GroundSource::Formula,
                    formula_epoch(&uniforms),
                    max_ground,
                );
            } else if !world.shadow_contact().is_empty() {
                uniforms.flags |= FLAG_ATLAS | FLAG_HORIZON;
                let extent = settings.atlas_extent_m.max(1.0);
                let focus_xz = [focus.x, focus.z];
                let origin = world.render_origin();
                let epoch = world.shadow_contact_epoch();
                let moved = (focus_xz[0] - self.last_focus_xz[0])
                    .hypot(focus_xz[1] - self.last_focus_xz[1]);
                let refresh = epoch != self.last_contact_epoch
                    || origin != self.last_origin
                    || moved > extent / 16.0
                    || (self.atlas_extent - extent).abs() > 1e-3;
                if refresh {
                    self.atlas_wrote = true;
                    self.atlas_generation += 1;
                    let origin_xz = [focus.x - extent * 0.5, focus.z - extent * 0.5];
                    fill_height_atlas(
                        &mut self.atlas_pixels,
                        self.atlas_size,
                        origin_xz,
                        extent,
                        origin,
                        world.shadow_contact(),
                    );
                    upload_atlas(
                        queue,
                        &self.horizon.ground_texture,
                        &self.atlas_pixels,
                        self.atlas_size,
                    );
                    self.atlas_max_ground = self
                        .atlas_pixels
                        .iter()
                        .copied()
                        .fold(MISSING_HEIGHT, f32::max);
                    self.atlas_origin_xz = origin_xz;
                    self.atlas_extent = extent;
                    self.last_contact_epoch = epoch;
                    self.last_origin = origin;
                    self.last_focus_xz = focus_xz;
                }
                self.horizon.prepare(
                    queue,
                    focus,
                    self.atlas_extent,
                    Some(self.atlas_origin_xz),
                    light,
                    GroundSource::Atlas,
                    self.atlas_generation,
                    self.atlas_max_ground,
                );
            }
            let (origin_xz, extent) = self.horizon.region();
            uniforms.atlas_origin_xz = origin_xz;
            uniforms.atlas_extent = extent;
        }

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        mats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::ContactGrid;
    use crate::space::{ChunkCoord, ChunkSpan, GlobalXZ};
    use crate::texture::{MaterialId, WaterMaterialId};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn texel_snap_is_stable_inside_one_texel() {
        let light = Vec3::Y;
        let radius = 40.0;
        let map_size = 1024;
        let texel = (2.0 * radius) / map_size as f32;
        let a = cascade_view_proj(Vec3::ZERO, light, radius, map_size);
        let b = cascade_view_proj(Vec3::new(texel * 0.25, 0.0, 0.0), light, radius, map_size);
        assert_eq!(a, b);
        let c = cascade_view_proj(Vec3::new(texel * 3.0, 0.0, 0.0), light, radius, map_size);
        assert_ne!(a, c);
    }

    #[test]
    fn terrain_and_water_materials_do_not_cast() {
        assert!(!material_casts_shadow(Some(SurfaceMaterialRef::Terrain(
            MaterialId(1)
        ))));
        assert!(!material_casts_shadow(Some(SurfaceMaterialRef::Water(
            WaterMaterialId(1)
        ))));
        assert!(material_casts_shadow(None));
    }

    #[test]
    fn height_atlas_samples_a_known_contact_point() {
        let verts = 3;
        let step = 10.0;
        let mut heights = vec![0.0; verts * verts];
        heights[verts + 1] = 42.0;
        let grid = ContactGrid::new(GlobalXZ::ORIGIN, step, verts, heights).unwrap();
        let span = ChunkSpan::new(20.0).unwrap();
        let mut grids = HashMap::new();
        grids.insert(ChunkCoord::new(0, 0), Arc::new(grid));
        let snapshot = ContactSnapshot::new(span, grids);

        let mut pixels = vec![0.0; 1];
        fill_height_atlas(
            &mut pixels,
            1,
            [9.5, 9.5],
            1.0,
            RenderOrigin::default(),
            &snapshot,
        );
        assert!(
            (pixels[0] - 42.0).abs() < 1e-3,
            "expected 42.0 at the centre vertex, got {}",
            pixels[0]
        );
    }
}
