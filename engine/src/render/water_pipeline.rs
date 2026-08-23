//! Animated water pipeline — analytic waves over a world-XZ sheet.
//!
//! Water is drawn from the same flat sheet geometry the game builds for its
//! shoreline contour; everything that makes it read as water (travelling
//! ripples, sun glint, depth colour, shore foam) happens per pixel here. The
//! sheet carries its depth in vertex alpha, so a hand-deep margin stays clear
//! and a channel goes dark without a second geometry pass.

use crate::mesh::{InstanceRaw, Vertex};
use crate::space::RenderOrigin;
use crate::texture::WaterMaterialDesc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterParams {
    pub shallow: [f32; 3],
    pub depth_scale_m: f32,
    pub deep: [f32; 3],
    pub wave_length_m: f32,
    pub wave_steepness: f32,
    pub wave_speed_m_s: f32,
    pub foam_width_m: f32,
    pub glint: f32,
    /// Render origin in world metres, so waves stay put across a rebase.
    pub world_offset_x: f32,
    pub world_offset_z: f32,
    pub _pad: [f32; 2],
}

impl WaterParams {
    pub fn from_desc(d: &WaterMaterialDesc, origin: RenderOrigin) -> Self {
        let shallow = d.shallow.to_vec3();
        let deep = d.deep.to_vec3();
        // Waves repeat over their own wavelength, so phasing on that period
        // keeps the pattern identical before and after a rebase.
        let phase = origin.texture_phase(d.wave_length_m.max(0.5));
        Self {
            shallow: shallow.into(),
            depth_scale_m: d.depth_scale_m.max(0.05),
            deep: deep.into(),
            wave_length_m: d.wave_length_m.max(0.5),
            wave_steepness: d.wave_steepness.max(0.0),
            wave_speed_m_s: d.wave_speed_m_s,
            foam_width_m: d.foam_width_m.max(0.0),
            glint: d.glint.max(0.0),
            world_offset_x: phase[0],
            world_offset_z: phase[1],
            _pad: [0.0; 2],
        }
    }
}

const SHADER: &str = r#"
struct WaterParams {
    shallow: vec3<f32>,
    depth_scale_m: f32,
    deep: vec3<f32>,
    wave_length_m: f32,
    wave_steepness: f32,
    wave_speed_m_s: f32,
    foam_width_m: f32,
    glint: f32,
    world_offset_x: f32,
    world_offset_z: f32,
};

@group(1) @binding(0) var<uniform> wp: WaterParams;

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

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_p: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.position, 1.0);
    var out: VsOut;
    out.clip = u.view_proj * world;
    out.color = v.color;
    out.world_p = world.xyz;
    return out;
}

/// One travelling wave's contribution to the surface gradient.
fn wave_gradient(p: vec2<f32>, dir: vec2<f32>, wavelength: f32, amp: f32, speed: f32) -> vec2<f32> {
    let k = 6.28318530718 / wavelength;
    let phase = dot(dir, p) * k + u.time * speed * k;
    return dir * (amp * k * cos(phase));
}

/// Sum a small spectrum of crossing swells into a shading normal.
fn water_normal(p: vec2<f32>, fade: f32) -> vec3<f32> {
    let base = wp.wave_length_m;
    let amp = wp.wave_steepness * base * 0.045 * fade;
    var g = vec2<f32>(0.0, 0.0);
    g += wave_gradient(p, normalize(vec2<f32>( 0.86,  0.51)), base,        amp,        wp.wave_speed_m_s);
    g += wave_gradient(p, normalize(vec2<f32>(-0.42,  0.91)), base * 0.63, amp * 0.70, wp.wave_speed_m_s * 1.27);
    g += wave_gradient(p, normalize(vec2<f32>( 0.31, -0.95)), base * 0.37, amp * 0.45, wp.wave_speed_m_s * 1.63);
    g += wave_gradient(p, normalize(vec2<f32>(-0.97, -0.24)), base * 0.19, amp * 0.24, wp.wave_speed_m_s * 2.20);
    return normalize(vec3<f32>(-g.x, 1.0, -g.y));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let world_xz = in.world_p.xz + vec2<f32>(wp.world_offset_x, wp.world_offset_z);
    let view_v = u.eye - in.world_p;
    let dist = length(view_v);
    let view = view_v / max(dist, 0.001);

    // Ripples that are finer than a pixel only alias, so let them settle with
    // distance instead of sparkling.
    let fade = clamp(1.0 - dist / 420.0, 0.12, 1.0);
    let n = water_normal(world_xz, fade);

    let depth_m = clamp(in.color.a, 0.0, 1.0) * wp.depth_scale_m;
    let deepness = smoothstep(0.0, wp.depth_scale_m * 0.55, depth_m);
    var body = mix(wp.shallow, wp.deep, deepness);
    // The sheet's own tint (river / lake / ocean authoring) nudges the hue.
    body = mix(body, body * (0.55 + in.color.rgb), 0.35);

    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    // Sky and sun share one budget, as on land, or water lit from both ends up
    // brighter than the sand beside it.
    let vis = sun_visibility(in.world_p, n, u.eye);
    var diffuse = u.ambient + (0.35 + 0.65 * ndl) * (1.0 - u.ambient) * u.light_color * vis;
    // The lantern reaches water too: wading a stream at night should show it.
    diffuse += torch_light(in.world_p, n);

    // Sky reflection at grazing angles is what makes flat water read as water.
    // Capped well below a mirror: with no reflection buffer, a full fresnel
    // turns the far half of a lake into flat sky.
    let fresnel = pow(1.0 - max(dot(n, view), 0.0), 4.0);
    let sky = vec3<f32>(0.26, 0.44, 0.72);
    var color = mix(body * diffuse, sky, clamp(fresnel * 0.45, 0.0, 0.38));

    let half_v = normalize(l + view);
    let glint = pow(max(dot(n, half_v), 0.0), 220.0) * wp.glint;
    color += u.light_color * glint * vis;

    // Foam where the bed comes up to meet the sheet, torn along the wave tilt
    // so the shoreline is a moving edge rather than a painted band.
    let shore = 1.0 - smoothstep(0.0, max(wp.foam_width_m, 0.001), depth_m);
    let tilt = smoothstep(0.02, 0.16, 1.0 - n.y);
    let foam = clamp(shore * shore * (0.35 + 0.75 * tilt), 0.0, 1.0);
    color = mix(color, vec3<f32>(0.90, 0.94, 0.96), foam);

    // Even a hand's depth of water carries its own colour; fully clear shallows
    // leave nothing but wet sand with a sheen on it.
    var alpha = mix(0.55, 0.96, deepness);
    alpha = max(alpha, foam * 0.85);
    alpha = clamp(alpha + fresnel * 0.20, 0.0, 1.0);
    // Distant sea has to close up as well as pale out, or the haze shows the
    // land through a sheet that should already be sky.
    let air = haze(color, in.world_p);
    alpha = mix(alpha, 1.0, clamp(length(air - color) * 2.0, 0.0, 1.0));
    return vec4<f32>(air, alpha);
}
"#;

pub struct GpuWaterMaterial {
    pub bind_group: wgpu::BindGroup,
    pub params_buf: wgpu::Buffer,
    pub desc: WaterMaterialDesc,
}

impl GpuWaterMaterial {
    pub fn write_origin(&self, queue: &wgpu::Queue, origin: RenderOrigin) {
        let params = WaterParams::from_desc(&self.desc, origin);
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }
}

pub struct WaterPipelines {
    pub blend: wgpu::RenderPipeline,
    pub blend_portal: wgpu::RenderPipeline,
    pub mat_bind_layout: wgpu::BindGroupLayout,
}

pub fn create_water_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    scene_bind_layout: &wgpu::BindGroupLayout,
) -> WaterPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("water-shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}{SHADER}", super::pipeline::scene_shader_prefix()).into(),
        ),
    });

    let mat_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("water-mat-layout"),
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
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("water-pipeline-layout"),
        bind_group_layouts: &[scene_bind_layout, &mat_bind_layout],
        push_constant_ranges: &[],
    });

    let make = |label: &str, depth_stencil: wgpu::DepthStencilState| {
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Readable from below, e.g. standing in a river.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    };

    let blend = make(
        "water-blend",
        super::stencil::scene_depth_stencil_unmasked_write(false),
    );
    let blend_portal = make(
        "water-blend-portal",
        super::stencil::scene_depth_stencil_masked_write(false),
    );

    WaterPipelines {
        blend,
        blend_portal,
        mat_bind_layout,
    }
}

pub fn build_water_material(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    desc: &WaterMaterialDesc,
    origin: RenderOrigin,
) -> GpuWaterMaterial {
    let params = WaterParams::from_desc(desc, origin);
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("water-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water-mat-bind"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buf.as_entire_binding(),
        }],
    });
    GpuWaterMaterial {
        bind_group,
        params_buf,
        desc: desc.clone(),
    }
}
