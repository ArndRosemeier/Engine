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

const _: () = assert!(MAX_JOINTS == 128);

pub const CASCADE_COUNT: usize = 3;

const FLAG_CSM: u32 = 1;
const FLAG_ATLAS: u32 = 2;
const FLAG_FORMULA: u32 = 4;
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

pub const SHADOW_UNIFORMS_WGSL: &str = r#"
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
};
"#;

pub const SCENE_SHADOW_WGSL: &str = r#"
@group(0) @binding(1) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(3) var shadow_samp: sampler_comparison;
@group(0) @binding(4) var height_atlas: texture_2d<f32>;
@group(0) @binding(5) var height_samp: sampler;

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
"#;

pub const SHADOW_EVAL_WGSL: &str = r#"
fn height_visibility(world_p: vec3<f32>) -> f32 {
    let l = normalize(shadow.light_dir);
    let xz_len = length(l.xz);
    if xz_len < 1e-4 {
        return 1.0;
    }
    let dir_xz = l.xz / xz_len;
    let rise = l.y / xz_len;
    let max_dist = max(shadow.cascade_end.z, 56.0);
    let fade_m = 12.0;
    let base = max(shadow.atlas_extent * shadow.map_texel, 0.45);
    var t = base;
    var hit_t = -1.0;
    for (var i = 0u; i < 10u; i++) {
        // Slope-scaled: a low sun (small rise) still covers distance in 10 taps.
        let step = base * (1.0 + 0.35 * f32(i)) / max(abs(rise), 0.18);
        t += step;
        let h = world_p.y + rise * t;
        let g = shadow_ground(world_p.xz + dir_xz * t);
        if g > -9999.0 && h + 0.12 < g {
            hit_t = t;
            break;
        }
    }
    if hit_t < 0.0 {
        return 1.0;
    }
    let fade = clamp((max_dist - hit_t) / fade_m, 0.0, 1.0);
    return mix(1.0, 0.42, fade);
}

fn csm_visibility(world_p: vec3<f32>, n: vec3<f32>, eye: vec3<f32>) -> f32 {
    let d = distance(world_p, eye);
    var layer = 0i;
    var vp = shadow.light_vp0;
    if d >= shadow.cascade_end.z && shadow.cascade_end.z > 0.0 {
        return 1.0;
    } else if d >= shadow.cascade_end.y {
        layer = 2i;
        vp = shadow.light_vp2;
    } else if d >= shadow.cascade_end.x {
        layer = 1i;
        vp = shadow.light_vp1;
    }
    let bias_p = world_p + n * max(shadow.map_texel * 4.0, 0.04);
    let clip = vp * vec4(bias_p, 1.0);
    let ndc = clip.xyz / max(clip.w, 1e-6);
    let uv = vec2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let texel = shadow.map_texel;
    var s = 0.0;
    s += textureSampleCompare(shadow_map, shadow_samp, uv + vec2(-texel, -texel), layer, ndc.z);
    s += textureSampleCompare(shadow_map, shadow_samp, uv + vec2( texel, -texel), layer, ndc.z);
    s += textureSampleCompare(shadow_map, shadow_samp, uv + vec2(-texel,  texel), layer, ndc.z);
    s += textureSampleCompare(shadow_map, shadow_samp, uv + vec2( texel,  texel), layer, ndc.z);
    return s * 0.25;
}

fn sun_visibility(world_p: vec3<f32>, n: vec3<f32>, eye: vec3<f32>) -> f32 {
    var vis = 1.0;
    if (shadow.flags & 1u) != 0u {
        vis = min(vis, csm_visibility(world_p, n, eye));
    }
    if (shadow.flags & 6u) != 0u {
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
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
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
    @location(3) joints: vec4<u32>,
    @location(4) weights: vec4<f32>,
    @location(5) m0: vec4<f32>,
    @location(6) m1: vec4<f32>,
    @location(7) m2: vec4<f32>,
    @location(8) m3: vec4<f32>,
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

fn resource_layout_entries(base: u32) -> [wgpu::BindGroupLayoutEntry; 5] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: base,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        },
    ]
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

fn apply_terrain_rules(u: &mut ShadowUniforms, rules: &TerrainRules) {
    u.seed = rules.seed;
    u.base_height = rules.base_height;
    u.hill_height = rules.hill_height;
    u.hill_scale = rules.hill_scale;
    u.lake_scale = rules.lake_scale;
    u.lake_threshold = rules.lake_threshold;
    u.water_level = rules.water_level;
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
    atlas_texture: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    comparison_sampler: wgpu::Sampler,
    atlas_sampler: wgpu::Sampler,
    map_size: u32,
    atlas_size: u32,
    atlas_pixels: Vec<f32>,
    atlas_origin_xz: [f32; 2],
    atlas_extent: f32,
    last_contact_epoch: u64,
    last_origin: RenderOrigin,
    last_focus_xz: [f32; 2],
    atlas_wrote: bool,
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

        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-height-atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-height-atlas-view"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        });
        let atlas_pixels = vec![MISSING_HEIGHT; atlas_size as usize * atlas_size as usize];
        upload_atlas(queue, &atlas_texture, &atlas_pixels, atlas_size);

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

        let resource_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-resource-layout"),
            entries: &resource_layout_entries(0),
        });
        let resource_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-resource-bind"),
            layout: &resource_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&map_array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&comparison_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

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
                5 => Float32x4,
                6 => Float32x4,
                7 => Float32x4,
                8 => Float32x4,
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
            atlas_texture,
            atlas_view,
            comparison_sampler,
            atlas_sampler,
            map_size,
            atlas_size,
            atlas_pixels,
            atlas_origin_xz: [0.0, 0.0],
            atlas_extent: settings.atlas_extent_m,
            last_contact_epoch: u64::MAX,
            last_origin: RenderOrigin::default(),
            last_focus_xz: [f32::MAX, f32::MAX],
            atlas_wrote: false,
        }
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

    pub fn scene_bind_entries<'a>(
        &'a self,
        scene_uniforms: wgpu::BindingResource<'a>,
    ) -> [wgpu::BindGroupEntry<'a>; 6] {
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
                resource: wgpu::BindingResource::TextureView(&self.atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
            },
        ]
    }

    pub fn prepare(&mut self, queue: &wgpu::Queue, world: &World) -> [Mat4; CASCADE_COUNT] {
        self.atlas_wrote = false;
        let mut uniforms = ShadowUniforms::zeroed();
        let light = world.light.direction.normalize_or_zero();
        uniforms.light_dir = [light.x, light.y, light.z];
        uniforms.map_texel = 1.0 / self.map_size as f32;

        let Some(settings) = world.shadows() else {
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
            let ident = Mat4::IDENTITY.to_cols_array_2d();
            for buf in &self.cascade_bufs {
                queue.write_buffer(buf, 0, bytemuck::bytes_of(&ident));
            }
            return [Mat4::IDENTITY; CASCADE_COUNT];
        };

        uniforms.flags = FLAG_CSM;
        let focus = world.camera.target;
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
                uniforms.flags |= FLAG_FORMULA;
                apply_terrain_rules(&mut uniforms, rules);
            } else if !world.shadow_contact().is_empty() {
                uniforms.flags |= FLAG_ATLAS;
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
                        &self.atlas_texture,
                        &self.atlas_pixels,
                        self.atlas_size,
                    );
                    self.atlas_origin_xz = origin_xz;
                    self.atlas_extent = extent;
                    self.last_contact_epoch = epoch;
                    self.last_origin = origin;
                    self.last_focus_xz = focus_xz;
                }
                uniforms.atlas_origin_xz = self.atlas_origin_xz;
                uniforms.atlas_extent = self.atlas_extent;
            }
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
        heights[1 * verts + 1] = 42.0;
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
