//! Terrain albedo pipeline — world-XZ grass/sand/rock blend.

use crate::mesh::{InstanceRaw, Vertex};
use crate::space::RenderOrigin;
use crate::texture::TerrainMaterialDesc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainParams {
    pub metres_per_tile: f32,
    pub rock_slope_start: f32,
    pub rock_slope_end: f32,
    pub sand_height_band: f32,
    pub sea_surface_z: f32,
    pub tint_strength: f32,
    /// Render origin wrapped into this material's tile period, so texture phase
    /// is continuous across a floating-origin rebase.
    pub world_offset_x: f32,
    pub world_offset_z: f32,
}

impl TerrainParams {
    pub fn from_desc(d: &TerrainMaterialDesc, origin: RenderOrigin) -> Self {
        let metres_per_tile = d.metres_per_tile.max(0.5);
        let phase = origin.texture_phase(metres_per_tile);
        Self {
            metres_per_tile,
            rock_slope_start: d.rock_slope_start,
            rock_slope_end: d.rock_slope_end.max(d.rock_slope_start + 0.01),
            sand_height_band: d.sand_height_band.max(0.5),
            sea_surface_z: d.sea_surface_z,
            tint_strength: d.tint_strength.clamp(0.0, 1.0),
            world_offset_x: phase[0],
            world_offset_z: phase[1],
        }
    }
}

const SHADER: &str = r#"
struct TerrainParams {
    metres_per_tile: f32,
    rock_slope_start: f32,
    rock_slope_end: f32,
    sand_height_band: f32,
    sea_surface_z: f32,
    tint_strength: f32,
    world_offset_x: f32,
    world_offset_z: f32,
};

@group(1) @binding(0) var grass_tex: texture_2d<f32>;
@group(1) @binding(1) var sand_tex: texture_2d<f32>;
@group(1) @binding(2) var rock_tex: texture_2d<f32>;
@group(1) @binding(3) var tex_sampler: sampler;
@group(1) @binding(4) var<uniform> tp: TerrainParams;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = model * vec4<f32>(v.position, 1.0);
    var out: VsOut;
    out.clip = u.view_proj * world;
    out.world_n = normalize((model * vec4<f32>(v.normal, 0.0)).xyz);
    out.color = v.color;
    out.world_p = world.xyz;
    return out;
}

/// Three octaves of the same tile, plus macro variation.
///
/// One tiling texture repeats visibly within a few metres. Adding a far
/// coarser sample of the same image both fills in large-scale shape and, used
/// as a brightness modulator, breaks the grid the eye would otherwise lock on.
fn sample_albedo(tex: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let base = textureSample(tex, tex_sampler, uv).rgb;
    let fine = textureSample(tex, tex_sampler, uv * 3.7 + vec2<f32>(0.17, 0.31)).rgb;
    let wide = textureSample(tex, tex_sampler, uv * 0.137 + vec2<f32>(0.61, 0.44)).rgb;
    let c = base * 0.64 + fine * 0.24 + wide * 0.12;
    let macro_luma = dot(wide, vec3<f32>(0.299, 0.587, 0.114));
    return c * (0.86 + 0.26 * macro_luma);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_n);
    // Add the wrapped render origin so tiling follows world space, not the
    // camera-local frame: rebasing must not slide the ground texture.
    let world_xz = in.world_p.xz + vec2<f32>(tp.world_offset_x, tp.world_offset_z);
    let uv = world_xz / tp.metres_per_tile;
    let grass = sample_albedo(grass_tex, uv);
    let sand = sample_albedo(sand_tex, uv * 1.15);
    let rock = sample_albedo(rock_tex, uv * 0.85);

    let slope = 1.0 - clamp(n.y, 0.0, 1.0);
    // Wobble the slope threshold with the ground itself, so the rock line is a
    // ragged edge instead of a contour drawn around the hill.
    let edge = (dot(rock, vec3<f32>(0.333, 0.333, 0.333)) - 0.35) * 0.22;
    let rock_w = smoothstep(tp.rock_slope_start, tp.rock_slope_end, slope + edge);
    let h = in.world_p.y - tp.sea_surface_z;
    let sand_w = (1.0 - smoothstep(0.0, tp.sand_height_band, h)) * (1.0 - rock_w);
    let grass_w = max(1.0 - rock_w - sand_w, 0.0);

    var albedo = grass * grass_w + sand * sand_w + rock * rock_w;
    // River / lake bed: authored as dark brown (low green). Must win over grass
    // or translucent water shows sliding grass parallax in the channel.
    let bed_w = (1.0 - smoothstep(0.14, 0.30, in.color.g))
        * (1.0 - smoothstep(0.28, 0.48, in.color.r));
    let mud = sand * vec3<f32>(0.42, 0.34, 0.26) + rock * vec3<f32>(0.18, 0.14, 0.12);
    albedo = mix(albedo, mud, clamp(bed_w, 0.0, 1.0));
    // Mild vertex-color tint (biome / slope authoring) — skip on beds.
    albedo = mix(albedo, albedo * in.color.rgb, tp.tint_strength * (1.0 - bed_w));

    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.65 + 0.35;
    let lit = albedo * (u.ambient + wrap * wrap * (1.0 - u.ambient) * u.light_color);
    return vec4<f32>(haze(lit, in.world_p), 1.0);
}
"#;

pub struct GpuTexture {
    pub view: wgpu::TextureView,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
}

pub struct GpuTerrainMaterial {
    pub bind_group: wgpu::BindGroup,
    /// Kept alive for the bind-group uniform binding, and rewritten on rebase.
    pub params_buf: wgpu::Buffer,
    pub desc: TerrainMaterialDesc,
}

impl GpuTerrainMaterial {
    /// Re-derive the texture phase after the render origin moved.
    pub fn write_origin(&self, queue: &wgpu::Queue, origin: RenderOrigin) {
        let params = TerrainParams::from_desc(&self.desc, origin);
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }
}

pub struct TerrainPipelines {
    pub opaque: wgpu::RenderPipeline,
    pub mat_bind_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
}

pub fn create_terrain_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    scene_bind_layout: &wgpu::BindGroupLayout,
) -> TerrainPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain-lit-shader"),
        source: wgpu::ShaderSource::Wgsl(format!("{}{SHADER}", super::pipeline::SCENE_WGSL).into()),
    });

    let mat_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain-mat-layout"),
        entries: &[
            tex_entry(0),
            tex_entry(1),
            tex_entry(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain-pipeline-layout"),
        bind_group_layouts: &[scene_bind_layout, &mat_bind_layout],
        push_constant_ranges: &[],
    });

    let opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("terrain-opaque"),
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
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: super::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: super::DEPTH_COMPARE,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain-sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    TerrainPipelines {
        opaque,
        mat_bind_layout,
        sampler,
    }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> GpuTexture {
    assert_eq!(rgba.len(), (width * height * 4) as usize);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain-albedo"),
        size: wgpu::Extent3d {
            width,
            height,
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
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        view,
        width,
        height,
    }
}

pub fn build_terrain_material(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    // Grass, sand, rock albedo in that order.
    layers: [&GpuTexture; 3],
    desc: &TerrainMaterialDesc,
    origin: RenderOrigin,
) -> GpuTerrainMaterial {
    let [grass, sand, rock] = layers;
    let params = TerrainParams::from_desc(desc, origin);
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain-mat-bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&grass.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&sand.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&rock.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    GpuTerrainMaterial {
        bind_group,
        params_buf,
        desc: desc.clone(),
    }
}
