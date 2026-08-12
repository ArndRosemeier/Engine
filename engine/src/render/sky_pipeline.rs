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
    pub _pad: f32,
    pub sun_color: [f32; 3],
    pub _pad2: f32,
}

impl SkyUniforms {
    pub fn from_scene(sky: &Sky, camera: &Camera, light: &Light, aspect: f32, time: f32) -> Self {
        let forward = (camera.target - camera.eye).normalize_or_zero();
        let mut right = forward.cross(camera.up);
        if right.length_squared() < 1e-8 {
            right = glam::Vec3::X;
        }
        let right = right.normalize();
        let up = right.cross(forward).normalize();
        let tan_half_fov = (camera.fov_y_degrees.to_radians() * 0.5).tan();
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
            _pad: 0.0,
            sun_color: sky.sun_color.to_vec3().into(),
            _pad2: 0.0,
        }
    }
}

pub struct SkyPipelines {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buf: wgpu::Buffer,
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
    _pad: f32,
    sun_color: vec3<f32>,
    _pad2: f32,
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
    g = select(vec3<f32>(1.0, 0.0, 0.0), g / len, len > 1e-5);
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

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    let view = vec3<f32>(
        v.ndc.x * s.tan_half_fov * s.aspect,
        v.ndc.y * s.tan_half_fov,
        1.0,
    );
    let dir = normalize(s.right * view.x + s.up * view.y + s.forward * view.z);
    let elev = dir.y;
    let toward = max(dot(dir, s.sun_dir), 0.0);

    let flat = vec3<f32>(dir.x, 0.0, dir.z);
    let sun_flat = vec3<f32>(s.sun_dir.x, 0.0, s.sun_dir.z);
    let az = 0.5 + 0.5 * dot(
        normalize(flat + vec3<f32>(1e-5, 0.0, 0.0)),
        normalize(sun_flat + vec3<f32>(1e-5, 0.0, 0.0)),
    );
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

pub fn create_sky_pipelines(device: &wgpu::Device, format: wgpu::TextureFormat) -> SkyPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sky-uniforms"),
        contents: bytemuck::bytes_of(&SkyUniforms {
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
            _pad: 0.0,
            sun_color: [1.0, 0.95, 0.85],
            _pad2: 0.0,
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky-bind"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky-pipeline-layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky-pipeline"),
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: super::DEPTH_FORMAT,
            depth_write_enabled: false,
            // First in the pass: paint every pixel, then ground covers it.
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    SkyPipelines {
        pipeline,
        bind_group,
        uniform_buf,
    }
}
