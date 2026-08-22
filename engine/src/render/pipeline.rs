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
    pub _pad2: [f32; 2],
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
            _pad2: [0.0, 0.0],
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
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Fade a surface into the sky by how much air the view ray crossed.
//
// Without this the ground simply ends at the last chunk. Air thins with height,
// so the amount crossed is the integral of exp(-(y - base) / H) along the ray,
// which has a closed form: a summit looks out over tens of kilometres while the
// valley below it is milk within five. Taking the density at the midpoint
// instead — the obvious shortcut — all but switches the haze off as soon as the
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
    @location(4) m0: vec4<f32>,
    @location(5) m1: vec4<f32>,
    @location(6) m2: vec4<f32>,
    @location(7) m3: vec4<f32>,
    @location(8) tint: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

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
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_n);
    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let base = in.color * texel;
    // Soft wrap lighting — enough contrast for smooth heightfields to read as
    // hills. Sky and sun share one budget, so raising ambient fills the shadows
    // instead of blowing out everything the sun already reaches.
    let wrap = ndl * 0.65 + 0.35;
    let vis = sun_visibility(in.world_p, n, u.eye);
    let lit = base.rgb * (u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color * vis);
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
    return vec4<f32>(haze(lit, in.world_p), alpha);
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
    /// Keeps the 1×1 white texel alive for `white_albedo`.
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
        SceneUniformSlots { buffers, bind_groups }
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
