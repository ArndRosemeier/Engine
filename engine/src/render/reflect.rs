//! Reflections on the GPU (see `crate::reflections` for the user side).
//!
//! **Screen-space reflections** are traced inside the lit shaders, against
//! last frame's image: after the world (and water) pass, `capture` copies the
//! scene colour at half resolution with the linear depth in alpha. Next
//! frame every glossy pixel marches its reflected ray through that image,
//! projected with last frame's view-projection, so colour and depth always
//! match. A pixel's own surface is in the image too, so reflections bounce
//! more than once over frames.
//!
//! **Reflection probes** are cube maps in one cube array (six layers per
//! probe, mips blurred for rough surfaces). A probe is captured by drawing
//! the scene six times at 128 x 128 into a scratch target; each face is then
//! copied (mirrored, since the cube map's faces are seen from inside) into
//! the array, and the mips are filtered from the one above.

use super::postprocess::HDR_FORMAT;
use crate::reflections::MAX_REFLECTION_PROBES;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

pub const PROBE_SIZE: u32 = 128;
pub const PROBE_MIPS: u32 = 6;
const SSR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// What the lit shaders read; matches `ReflectUniforms` in [`REFLECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReflectUniforms {
    /// Last frame's (jittered) view-projection, which the SSR image was drawn with.
    prev_vp: [[f32; 4]; 4],
    /// x strength (0 off), y max distance, z steps, w unused.
    ssr: [f32; 4],
    /// SSR image size and its reciprocal.
    ssr_size: [f32; 4],
    /// x probe count, y highest mip.
    probe_info: [f32; 4],
    /// Per probe: (center, blend), (half extents, array index).
    probes: [[f32; 4]; MAX_REFLECTION_PROBES * 2],
}

const _: () = assert!(std::mem::size_of::<ReflectUniforms>() == 112 + 32 * MAX_REFLECTION_PROBES);

/// Declarations and functions the scene shaders use (scene group 0).
pub const REFLECT_WGSL: &str = r#"
struct ReflectUniforms {
    prev_vp: mat4x4<f32>,
    ssr: vec4<f32>,
    ssr_size: vec4<f32>,
    probe_info: vec4<f32>,
    probes: array<vec4<f32>, PROBE_VECS>,
};

@group(0) @binding(16) var ssr_tex: texture_2d<f32>;
@group(0) @binding(17) var reflect_samp: sampler;
@group(0) @binding(18) var probe_tex: texture_cube_array<f32>;
@group(0) @binding(19) var<uniform> refl: ReflectUniforms;

// Box-projected probe light along `r` from `p`; alpha is coverage.
fn probe_reflection(p: vec3<f32>, r: vec3<f32>, roughness: f32) -> vec4<f32> {
    let count = i32(refl.probe_info.x);
    var sum = vec3<f32>(0.0);
    var w = 0.0;
    let lod = roughness * refl.probe_info.y;
    for (var i = 0; i < count; i++) {
        let a = refl.probes[i * 2];
        let b = refl.probes[i * 2 + 1];
        let local = p - a.xyz;
        let room = b.xyz - abs(local);
        if any(room < vec3<f32>(0.0)) {
            continue;
        }
        let edge = min(room.x, min(room.y, room.z));
        let wt = select(1.0, clamp(edge / a.w, 0.0, 1.0), a.w > 0.0);
        // Where the ray leaves the box, seen from the capture point.
        let rr = select(r, vec3<f32>(1e-5), abs(r) < vec3<f32>(1e-5));
        let t1 = (b.xyz - local) / rr;
        let t2 = (-b.xyz - local) / rr;
        let tf = max(t1, t2);
        let dist = min(tf.x, min(tf.y, tf.z));
        let dir = local + r * max(dist, 0.0);
        let c = textureSampleLevel(probe_tex, reflect_samp, dir, i32(b.w), lod).rgb;
        let take = wt * (1.0 - w);
        sum += c * take;
        w += take;
        if w > 0.999 {
            break;
        }
    }
    return vec4<f32>(sum, w);
}

// Last frame's image is fed back here, so a NaN or infinite pixel in it
// (tested on the bits) reads as black instead of spreading over every
// glossy surface frame after frame.
fn reflect_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn reflect_clean(c: vec3<f32>) -> vec3<f32> {
    if reflect_finite(c.r) && reflect_finite(c.g) && reflect_finite(c.b) {
        return c;
    }
    return vec3<f32>(0.0);
}

// The reflected ray from `p` along `r`, traced through last frame's image.
// Alpha is how far to trust the hit (0 for a miss).
fn ssr_trace(p: vec3<f32>, r: vec3<f32>, roughness: f32, noise: f32) -> vec4<f32> {
    let strength = refl.ssr.x * (1.0 - smoothstep(0.2, 0.45, roughness));
    if strength <= 0.0 {
        return vec4<f32>(0.0);
    }
    let c0 = refl.prev_vp * vec4<f32>(p, 1.0);
    if c0.w <= 0.05 {
        return vec4<f32>(0.0);
    }
    var len = refl.ssr.y;
    var c1 = refl.prev_vp * vec4<f32>(p + r * len, 1.0);
    // Stop the ray in front of the eye (w grows linearly along it).
    if c1.w < 0.05 {
        len = len * (c0.w - 0.05) / (c0.w - c1.w);
        c1 = refl.prev_vp * vec4<f32>(p + r * len, 1.0);
    }
    let k0 = 1.0 / c0.w;
    let k1 = 1.0 / c1.w;
    let s0 = c0.xy * k0;
    let s1 = c1.xy * k1;
    let size = refl.ssr_size.xy;
    let span = length((s1 - s0) * 0.5 * size);
    if span < 2.0 {
        return vec4<f32>(0.0);
    }
    let steps = i32(refl.ssr.z);
    // Skip the first two pixels, where the ray still sits on its own surface.
    let t_min = 2.0 / span;
    var prev_t = t_min;
    var hit = -1.0;
    for (var i = 0; i < steps; i++) {
        let f = (f32(i) + noise) / f32(steps);
        let t = t_min + (1.0 - t_min) * f * f;
        let s = mix(s0, s1, t);
        let uv = vec2<f32>(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
            break;
        }
        let ray_depth = 1.0 / mix(k0, k1, t);
        let scene_depth = textureLoad(ssr_tex, vec2<i32>(uv * size), 0).a;
        let step_depth = abs(ray_depth - 1.0 / mix(k0, k1, prev_t));
        let thickness = max(step_depth * 2.0, 0.15 + ray_depth * 0.02);
        let diff = ray_depth - scene_depth;
        if diff > 0.0 && diff < thickness {
            hit = t;
            break;
        }
        prev_t = t;
    }
    if hit < 0.0 {
        return vec4<f32>(0.0);
    }
    // Close in on the surface between the last miss and the hit.
    var lo = prev_t;
    var hi = hit;
    for (var j = 0; j < 5; j++) {
        let mid = (lo + hi) * 0.5;
        let s = mix(s0, s1, mid);
        let uv = vec2<f32>(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
        let ray_depth = 1.0 / mix(k0, k1, mid);
        let scene_depth = textureLoad(ssr_tex, vec2<i32>(uv * size), 0).a;
        if ray_depth > scene_depth {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let s = mix(s0, s1, hi);
    let uv = vec2<f32>(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
    let color = reflect_clean(textureSampleLevel(ssr_tex, reflect_samp, uv, 0.0).rgb);
    // Fade near the screen edges and the end of the ray, where the image
    // runs out, so a reflection never ends in a line.
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let fade = smoothstep(0.0, 0.08, edge) * (1.0 - smoothstep(0.75, 1.0, hi));
    return vec4<f32>(color, strength * fade);
}

// Everything a glossy surface reflects, except the sky: SSR over probes.
// Alpha 0 leaves the sky reflection alone.
fn scene_reflection(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, roughness: f32,
        base: vec3<f32>, metallic: f32, frag: vec2<f32>) -> vec4<f32> {
    // Faint reflections (rough or plain dielectric seen head-on) skip the
    // work: they would not show.
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let nv = max(dot(n, v), 0.0);
    let fres = max(f0.x, max(f0.y, f0.z)) + pow(max(1.0 - nv, 0.0), 5.0) * (1.0 - roughness);
    let r = reflect(-v, n);
    var probe = vec4<f32>(0.0);
    if refl.probe_info.x > 0.0 {
        probe = probe_reflection(p, r, roughness);
    }
    if fres < 0.06 || refl.ssr.x <= 0.0 || roughness >= 0.45 {
        return probe;
    }
    let noise = fract(52.9829189 * fract(dot(frag, vec2<f32>(0.06711056, 0.00583715)) + u.dither_frame * 0.618034));
    let hit = ssr_trace(p, r, roughness, noise);
    let a = hit.a + (1.0 - hit.a) * probe.a;
    if a <= 0.0 {
        return vec4<f32>(0.0);
    }
    let rgb = (hit.rgb * hit.a + probe.rgb * probe.a * (1.0 - hit.a)) / a;
    return vec4<f32>(rgb, a);
}
"#;

/// The reflection WGSL with its array size filled in.
pub fn reflect_wgsl() -> String {
    REFLECT_WGSL.replace("PROBE_VECS", &(MAX_REFLECTION_PROBES * 2).to_string())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CaptureParams {
    inv_proj: [[f32; 4]; 4],
    full_size: [f32; 2],
    out_size: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FilterParams {
    face: u32,
    /// Blur cone half-angle tangent for this mip.
    spread: f32,
    size: f32,
    _pad: f32,
}

const CAPTURE_WGSL: &str = r#"
struct CaptureParams {
    inv_proj: mat4x4<f32>,
    full_size: vec2<f32>,
    out_size: vec2<f32>,
};
@group(0) @binding(0) var<uniform> p: CaptureParams;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depth_tex: DEPTH_TYPE;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_capture(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / p.out_size;
    let raw = textureSampleLevel(scene, samp, uv, 0.0).rgb;
    let raw_e = bitcast<vec3<u32>>(raw) & vec3<u32>(0x7f800000u);
    // Infinity (half-float overflow) or NaN reads as black: the cap below
    // would turn infinity into NaN.
    let color = select(max(raw, vec3<f32>(0.0)), vec3<f32>(0.0), any(raw_e == vec3<u32>(0x7f800000u)));
    let px = clamp(vec2<i32>(uv * p.full_size), vec2<i32>(0), vec2<i32>(p.full_size) - vec2<i32>(1));
    let d = textureLoad(depth_tex, px, 0);
    // Reversed-Z: the sky is at 0, beyond everything.
    var depth = 1e9;
    if d > 0.0 {
        let v = p.inv_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
        depth = -v.z / v.w;
    }
    // Keep fireflies (sun glints) from smearing over reflections.
    let lum = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let c = color * min(1.0, 64.0 / max(lum, 1e-4));
    return vec4<f32>(c, depth);
}
"#;

const FILTER_WGSL: &str = r#"
struct FilterParams {
    face: u32,
    spread: f32,
    size: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> f: FilterParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src_face: texture_2d<f32>;
@group(0) @binding(3) var src_cube: texture_cube<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// The direction of texel `uv` (rows downward) on cube face `face`.
fn face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let sc = uv.x * 2.0 - 1.0;
    let tc = uv.y * 2.0 - 1.0;
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -tc, -sc)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -tc, sc)); }
        case 2u: { return normalize(vec3<f32>(sc, 1.0, tc)); }
        case 3u: { return normalize(vec3<f32>(sc, -1.0, -tc)); }
        case 4u: { return normalize(vec3<f32>(sc, -tc, 1.0)); }
        default: { return normalize(vec3<f32>(-sc, -tc, -1.0)); }
    }
}

// Non-finite colours (a half-float overflow stored as infinity, or NaN)
// read as black, and one bright texel may not outweigh the rest: a probe
// keeps what it captured and every surface near it reads the result.
fn probe_clean(c: vec3<f32>) -> vec3<f32> {
    let e = bitcast<vec3<u32>>(c) & vec3<u32>(0x7f800000u);
    if any(e == vec3<u32>(0x7f800000u)) {
        return vec3<f32>(0.0);
    }
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return max(c, vec3<f32>(0.0)) * min(1.0, 64.0 / max(lum, 1e-4));
}

// A captured face, mirrored left to right: the camera saw it from inside.
@fragment
fn fs_copy(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let size = i32(f.size);
    let px = vec2<i32>(frag.xy);
    let c = textureLoad(src_face, vec2<i32>(size - 1 - px.x, px.y), 0);
    return vec4<f32>(probe_clean(c.rgb), 1.0);
}

// One mip: the mip above, blurred over a cone.
@fragment
fn fs_filter(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let d = face_dir(f.face, frag.xy / f.size);
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.y) > 0.9);
    let t = normalize(cross(up, d));
    let b = cross(d, t);
    var sum = textureSampleLevel(src_cube, samp, d, 0.0).rgb;
    var w = 1.0;
    for (var ring = 1; ring <= 2; ring++) {
        let rr = f.spread * f32(ring) * 0.5;
        let wt = exp(-f32(ring * ring) * 0.5);
        for (var k = 0; k < 8; k++) {
            let a = (f32(k) + 0.5 * f32(ring)) * 0.785398;
            let dir = normalize(d + (t * cos(a) + b * sin(a)) * rr);
            sum += textureSampleLevel(src_cube, samp, dir, 0.0).rgb * wt;
            w += wt;
        }
    }
    return vec4<f32>(sum / w, 1.0);
}
"#;

/// A probe slot's capture progress.
#[derive(Clone, Copy, Debug, Default)]
struct Slot {
    /// Version the cube map shows (0: never captured).
    shown: u64,
    /// Version being captured and the next face to draw.
    capturing: Option<(u64, u32)>,
    live: bool,
}

/// One cube face to draw this frame.
#[derive(Clone, Copy, Debug)]
pub struct FaceJob {
    pub slot: usize,
    pub face: u32,
    pub center: Vec3,
    /// Filter the mips after this face (the cube is complete).
    pub finish: bool,
}

/// Camera for cube face `face` at `center`: the forward and up that, once
/// the image is mirrored, give the cube map's face layout.
pub fn face_view(face: u32) -> (Vec3, Vec3) {
    match face {
        0 => (Vec3::X, Vec3::Y),
        1 => (Vec3::NEG_X, Vec3::Y),
        2 => (Vec3::Y, Vec3::NEG_Z),
        3 => (Vec3::NEG_Y, Vec3::Z),
        4 => (Vec3::Z, Vec3::Y),
        _ => (Vec3::NEG_Z, Vec3::Y),
    }
}

pub struct Reflect {
    capture_pipeline: wgpu::RenderPipeline,
    capture_layout: wgpu::BindGroupLayout,
    capture_params: wgpu::Buffer,
    ssr_view: wgpu::TextureView,
    ssr_size: [u32; 2],
    full_size: [u32; 2],
    linear: wgpu::Sampler,
    probe_texture: wgpu::Texture,
    probe_view: wgpu::TextureView,
    face_color: wgpu::TextureView,
    face_resolve: Option<wgpu::TextureView>,
    face_depth: wgpu::TextureView,
    copy_pipeline: wgpu::RenderPipeline,
    filter_pipeline: wgpu::RenderPipeline,
    filter_layout: wgpu::BindGroupLayout,
    /// Stand-ins for the filter group's unused texture.
    blank_face: wgpu::TextureView,
    blank_cube: wgpu::TextureView,
    slots: [Slot; MAX_REFLECTION_PROBES],
    /// View-projection the SSR image was captured with, and the render
    /// origin it was in.
    captured_vp: Option<(Mat4, crate::space::RenderOrigin)>,
}

impl Reflect {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        depth: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        let multisampled = depth.sample_count() > 1;
        let depth_type = if multisampled {
            "texture_depth_multisampled_2d"
        } else {
            "texture_depth_2d"
        };
        let capture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssr-capture-shader"),
            source: wgpu::ShaderSource::Wgsl(CAPTURE_WGSL.replace("DEPTH_TYPE", depth_type).into()),
        });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture = |binding, dim| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        };
        let sampler = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let capture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr-capture-layout"),
            entries: &[
                uniform(0),
                texture(1, wgpu::TextureViewDimension::D2),
                sampler(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: frag,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
            ],
        });
        let filter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe-filter-layout"),
            entries: &[
                uniform(0),
                sampler(1),
                texture(2, wgpu::TextureViewDimension::D2),
                texture(3, wgpu::TextureViewDimension::Cube),
            ],
        });
        let make = |label: &str,
                    module: &wgpu::ShaderModule,
                    layout: &wgpu::BindGroupLayout,
                    entry: &str,
                    format| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_full"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            })
        };
        let capture_pipeline = make(
            "ssr-capture",
            &capture_shader,
            &capture_layout,
            "fs_capture",
            SSR_FORMAT,
        );
        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe-filter-shader"),
            source: wgpu::ShaderSource::Wgsl(FILTER_WGSL.into()),
        });
        let copy_pipeline = make(
            "probe-copy",
            &filter_shader,
            &filter_layout,
            "fs_copy",
            PROBE_FORMAT,
        );
        let filter_pipeline = make(
            "probe-filter",
            &filter_shader,
            &filter_layout,
            "fs_filter",
            PROBE_FORMAT,
        );
        let capture_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ssr-capture-params"),
            contents: bytemuck::bytes_of(&CaptureParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let linear = linear_sampler(device);
        let probe_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection-probes"),
            size: wgpu::Extent3d {
                width: PROBE_SIZE,
                height: PROBE_SIZE,
                depth_or_array_layers: 6 * MAX_REFLECTION_PROBES as u32,
            },
            mip_level_count: PROBE_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PROBE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let probe_view = probe_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("reflection-probes-view"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let samples = super::stencil::scene_samples();
        let face = |label: &str, format, samples| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: PROBE_SIZE,
                        height: PROBE_SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let face_color = face("probe-face", HDR_FORMAT, samples);
        let face_resolve = (samples > 1).then(|| face("probe-face-resolve", HDR_FORMAT, 1));
        let face_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("probe-face-depth"),
                size: wgpu::Extent3d {
                    width: PROBE_SIZE,
                    height: PROBE_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: super::stencil::depth_format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let blank_face = face("probe-blank", HDR_FORMAT, 1);
        let blank_cube = blank_cube_view(device, queue, wgpu::TextureViewDimension::Cube);
        let mut r = Self {
            capture_pipeline,
            capture_layout,
            capture_params,
            ssr_view: blank_face.clone(),
            ssr_size: [1, 1],
            full_size: [1, 1],
            linear,
            probe_texture,
            probe_view,
            face_color,
            face_resolve,
            face_depth,
            copy_pipeline,
            filter_pipeline,
            filter_layout,
            blank_face,
            blank_cube,
            slots: [Slot::default(); MAX_REFLECTION_PROBES],
            captured_vp: None,
        };
        r.resize(device, width, height);
        r
    }

    /// New SSR image size (the scene bind groups must then be rebuilt with
    /// [`Self::ssr_view`]).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.div_ceil(2).max(1);
        let h = height.div_ceil(2).max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssr-image"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.ssr_view = texture.create_view(&Default::default());
        self.ssr_size = [w, h];
        self.full_size = [width.max(1), height.max(1)];
        self.captured_vp = None;
    }

    pub fn ssr_view(&self) -> &wgpu::TextureView {
        &self.ssr_view
    }

    pub fn probe_view(&self) -> &wgpu::TextureView {
        &self.probe_view
    }

    /// Uniforms for this frame's lit shaders.
    pub fn uniforms(
        &self,
        ssr: Option<crate::reflections::ScreenSpaceReflections>,
        origin: crate::space::RenderOrigin,
        probes: &[(usize, crate::reflections::ReflectionProbe)],
    ) -> ReflectUniforms {
        let mut u = ReflectUniforms::zeroed();
        if let (Some(s), Some((vp, at))) = (ssr, self.captured_vp) {
            if at == origin {
                u.prev_vp = vp.to_cols_array_2d();
                u.ssr = [s.strength, s.max_distance_m, s.steps as f32, 0.0];
            }
        }
        let [w, h] = self.ssr_size;
        u.ssr_size = [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32];
        // Smallest box first: it wins where boxes overlap.
        let mut live: Vec<_> = probes
            .iter()
            .filter(|(slot, _)| self.slots.get(*slot).is_some_and(|s| s.shown > 0))
            .collect();
        live.sort_by(|a, b| a.1.volume().total_cmp(&b.1.volume()));
        for (i, (slot, p)) in live.iter().enumerate() {
            u.probes[i * 2] = [p.center.x, p.center.y, p.center.z, p.blend_m];
            u.probes[i * 2 + 1] = [
                p.half_extents.x,
                p.half_extents.y,
                p.half_extents.z,
                *slot as f32,
            ];
        }
        u.probe_info = [live.len() as f32, (PROBE_MIPS - 1) as f32, 0.0, 0.0];
        u
    }

    /// The cube faces to draw this frame: a new probe gets all six at once
    /// (the first one found), changed and dynamic probes one face each.
    pub fn plan(
        &mut self,
        probes: &[(usize, crate::reflections::ReflectionProbe, u64)],
    ) -> Vec<FaceJob> {
        plan_slots(&mut self.slots, probes)
    }

    pub fn face_targets(
        &self,
    ) -> (
        &wgpu::TextureView,
        Option<&wgpu::TextureView>,
        &wgpu::TextureView,
    ) {
        (
            &self.face_color,
            self.face_resolve.as_ref(),
            &self.face_depth,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn filter_pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        params: FilterParams,
        src_face: &wgpu::TextureView,
        src_cube: &wgpu::TextureView,
        target: &wgpu::TextureView,
    ) {
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe-filter-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe-filter-bind"),
            layout: &self.filter_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(src_face),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(src_cube),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("probe-filter"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }

    fn layer_view(&self, layer: u32, mip: u32) -> wgpu::TextureView {
        self.probe_texture
            .create_view(&wgpu::TextureViewDescriptor {
                label: Some("probe-layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
    }

    /// Copy the face just drawn into its probe, and filter the mips once
    /// the cube is whole.
    pub fn store_face(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        job: FaceJob,
    ) {
        let src = self.face_resolve.as_ref().unwrap_or(&self.face_color);
        let base = job.slot as u32 * 6;
        let params = FilterParams {
            face: job.face,
            spread: 0.0,
            size: PROBE_SIZE as f32,
            _pad: 0.0,
        };
        let target = self.layer_view(base + job.face, 0);
        self.filter_pass(
            device,
            encoder,
            &self.copy_pipeline,
            params,
            src,
            &self.blank_cube,
            &target,
        );
        if !job.finish {
            return;
        }
        let mut prev_spread = 0.0f32;
        for mip in 1..PROBE_MIPS {
            let roughness = mip as f32 / (PROBE_MIPS - 1) as f32;
            // Widen the blur so mip m matches roughness m / (mips - 1):
            // each pass adds what the mip above lacks.
            let want = (roughness * roughness * 1.4).min(1.2);
            let spread = (want * want - prev_spread * prev_spread).max(0.0).sqrt();
            prev_spread = want;
            let src_cube = self
                .probe_texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: Some("probe-src-cube"),
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    base_mip_level: mip - 1,
                    mip_level_count: Some(1),
                    base_array_layer: base,
                    array_layer_count: Some(6),
                    ..Default::default()
                });
            for face in 0..6 {
                let params = FilterParams {
                    face,
                    spread,
                    size: (PROBE_SIZE >> mip).max(1) as f32,
                    _pad: 0.0,
                };
                let target = self.layer_view(base + face, mip);
                self.filter_pass(
                    device,
                    encoder,
                    &self.filter_pipeline,
                    params,
                    &self.blank_face,
                    &src_cube,
                    &target,
                );
            }
        }
    }

    /// Copy this frame's scene colour and depth into the SSR image, for the
    /// next frame to trace against.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &wgpu::TextureView,
        depth: &wgpu::Texture,
        projection: Mat4,
        view_proj: Mat4,
        origin: crate::space::RenderOrigin,
    ) {
        let params = CaptureParams {
            inv_proj: projection.inverse().to_cols_array_2d(),
            full_size: [self.full_size[0] as f32, self.full_size[1] as f32],
            out_size: [self.ssr_size[0] as f32, self.ssr_size[1] as f32],
        };
        queue.write_buffer(&self.capture_params, 0, bytemuck::bytes_of(&params));
        // The scene targets can change behind our back (resize, MSAA), so
        // the bind group is cheap enough to make each frame.
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ssr-depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr-capture-bind"),
            layout: &self.capture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.capture_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssr-capture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.capture_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.captured_vp = Some((view_proj, origin));
    }
}

/// See [`Reflect::plan`].
fn plan_slots(
    slots: &mut [Slot; MAX_REFLECTION_PROBES],
    probes: &[(usize, crate::reflections::ReflectionProbe, u64)],
) -> Vec<FaceJob> {
    for (i, s) in slots.iter_mut().enumerate() {
        s.live = probes.iter().any(|(slot, _, _)| *slot == i);
        if !s.live {
            *s = Slot::default();
        }
    }
    let mut jobs = Vec::new();
    let mut whole_done = false;
    for &(slot, probe, version) in probes {
        let Some(s) = slots.get_mut(slot) else {
            continue;
        };
        if let Some((v, _)) = s.capturing {
            if v != version {
                s.capturing = None;
            }
        }
        let stale = s.shown != version && s.capturing.is_none();
        // A probe never captured gets its six faces in one frame, so it
        // never shows black. One that only changed (moved on with the
        // camera, say) is redrawn a face a frame like a dynamic one: six
        // scene passes in one frame are a hitch of 10 to 20 ms, and the old
        // faces stand in for the five frames it takes.
        if stale && !whole_done && s.shown == 0 {
            whole_done = true;
            for face in 0..6 {
                jobs.push(FaceJob {
                    slot,
                    face,
                    center: probe.center,
                    finish: face == 5,
                });
            }
            s.shown = version;
            continue;
        }
        if stale && s.shown == 0 {
            // Its turn for a whole capture comes in a later frame.
            continue;
        }
        if probe.dynamic || stale || s.capturing.is_some() {
            let (v, face) = s.capturing.unwrap_or((version, 0));
            jobs.push(FaceJob {
                slot,
                face,
                center: probe.center,
                finish: face == 5,
            });
            s.capturing = if face == 5 { None } else { Some((v, face + 1)) };
            if face == 5 {
                s.shown = v;
            }
        }
    }
    jobs
}

fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("reflect-linear"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// A 1 x 1 black cube (or cube array of one cube).
fn blank_cube_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("reflect-blank-cube"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[0u8; 8 * 6],
    );
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(dimension),
        ..Default::default()
    })
}

impl super::shadow::ReflectResources {
    /// Nothing to reflect yet: a blank image and no probes.
    pub fn placeholder(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let ssr = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("ssr-none"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SSR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0u8; 8],
        );
        Self {
            ssr_view: ssr.create_view(&Default::default()),
            sampler: linear_sampler(device),
            probe_view: blank_cube_view(device, queue, wgpu::TextureViewDimension::CubeArray),
            uniform: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reflect-uniforms"),
                contents: bytemuck::bytes_of(&ReflectUniforms::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_capture_and_filter_shaders_are_valid_wgsl() {
        for depth in ["texture_depth_2d", "texture_depth_multisampled_2d"] {
            crate::render::assert_valid_wgsl(&super::CAPTURE_WGSL.replace("DEPTH_TYPE", depth));
        }
        crate::render::assert_valid_wgsl(super::FILTER_WGSL);
    }
    use super::*;

    /// The capture cameras, mirrored, must give the cube map's layout: the
    /// direction of a face texel (as the filter shader computes it) must be
    /// where that texel's mirrored pixel looks.
    #[test]
    fn face_cameras_match_the_cube_layout() {
        let dir = |face: u32, u: f32, v: f32| -> Vec3 {
            let (sc, tc) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
            match face {
                0 => Vec3::new(1.0, -tc, -sc),
                1 => Vec3::new(-1.0, -tc, sc),
                2 => Vec3::new(sc, 1.0, tc),
                3 => Vec3::new(sc, -1.0, -tc),
                4 => Vec3::new(sc, -tc, 1.0),
                _ => Vec3::new(-sc, -tc, -1.0),
            }
            .normalize()
        };
        for face in 0..6 {
            let (fwd, up) = face_view(face);
            let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, fwd, up);
            let proj = glam::camera::rh::proj::directx::perspective(
                std::f32::consts::FRAC_PI_2,
                1.0,
                0.1,
                10.0,
            );
            for (u, v) in [(0.2, 0.3), (0.8, 0.1), (0.5, 0.9)] {
                let d = dir(face, u, v);
                let c = proj * view * d.extend(1.0);
                let ndc = c.truncate() / c.w;
                // Pixel the camera drew it at, then mirrored left-right.
                let px = (ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                let mirrored = (1.0 - px.0, px.1);
                assert!(
                    (mirrored.0 - u).abs() < 1e-4 && (mirrored.1 - v).abs() < 1e-4,
                    "face {face} uv {u},{v} drew at {mirrored:?}"
                );
            }
        }
    }

    #[test]
    fn plan_captures_new_probes_whole_and_dynamic_ones_a_face_a_frame() {
        let mut slots = [Slot::default(); MAX_REFLECTION_PROBES];
        let probe = crate::reflections::ReflectionProbe::new(Vec3::ZERO, Vec3::ONE);
        let mut plan = |probes: &[(usize, crate::reflections::ReflectionProbe, u64)]| {
            let mut r = PlanOnly { slots };
            let jobs = r.plan(probes);
            slots = r.slots;
            jobs
        };
        let jobs = plan(&[(0, probe, 1), (1, probe, 2)]);
        assert_eq!(jobs.len(), 6, "one whole probe per frame");
        assert!(jobs.iter().all(|j| j.slot == 0) && jobs[5].finish);
        let jobs = plan(&[(0, probe, 1), (1, probe, 2)]);
        assert!(jobs.len() == 6 && jobs[0].slot == 1);
        assert!(
            plan(&[(0, probe, 1), (1, probe, 2)]).is_empty(),
            "nothing changed"
        );
        // A probe that changed is redrawn a face a frame, then rests.
        let mut faces = Vec::new();
        for _ in 0..8 {
            let jobs = plan(&[(0, probe, 5), (1, probe, 2)]);
            assert!(jobs.len() <= 1 && jobs.iter().all(|j| j.slot == 0));
            faces.extend(jobs.iter().map(|j| (j.face, j.finish)));
        }
        let want: Vec<(u32, bool)> = (0..6).map(|f| (f, f == 5)).collect();
        assert_eq!(faces, want);
        let dynamic = probe.dynamic();
        let mut faces = Vec::new();
        for _ in 0..7 {
            faces.extend(plan(&[(0, dynamic, 1)]).iter().map(|j| j.face));
        }
        assert_eq!(faces, vec![0, 1, 2, 3, 4, 5, 0]);
    }

    /// `Reflect::plan` without a device.
    struct PlanOnly {
        slots: [Slot; MAX_REFLECTION_PROBES],
    }

    impl PlanOnly {
        fn plan(
            &mut self,
            probes: &[(usize, crate::reflections::ReflectionProbe, u64)],
        ) -> Vec<FaceJob> {
            plan_slots(&mut self.slots, probes)
        }
    }
}
