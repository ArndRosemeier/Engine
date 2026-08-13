//! GPU clipmap: static grids displaced by a WGSL height formula.

use crate::proc_terrain::{ClipmapConfig, ProcTerrain, TerrainParamsUniform};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

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
    _pad2: f32,
}

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
    _pad2: f32,
};

struct RingUniform {
    center: vec2<f32>,
    extent: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> frame: FrameUniform;
@group(0) @binding(1) var<uniform> terrain: TerrainParams;
@group(0) @binding(2) var<uniform> ring: RingUniform;

struct VsIn {
    @location(0) local_xz: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_p: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) wet: f32,
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
    return out;
}

@fragment
fn fs_land(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_n);
    let l = normalize(frame.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.5 + 0.5;
    let vis = sun_visibility(in.world_p, n, frame.eye);
    let lit = in.color.rgb * (frame.ambient + wrap * wrap * vis * frame.light_color);
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
    let lit = in.color.rgb * (frame.ambient + wrap * wrap * vis * frame.light_color);
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
    water_pipeline: wgpu::RenderPipeline,
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
}

impl ClipmapRenderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        config: ClipmapConfig,
        shadow_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clipmap-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{SHADER}{}{}{}",
                    super::shadow::SHADOW_UNIFORMS_WGSL,
                    super::shadow::CLIPMAP_SHADOW_WGSL,
                    super::shadow::SHADOW_EVAL_WGSL
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clipmap-pipeline-layout"),
            bind_group_layouts: &[&bind_layout, shadow_layout],
            push_constant_ranges: &[],
        });

        let vertex_attrs = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        }];

        let land_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clipmap-land"),
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
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_land"),
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

        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clipmap-water"),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: super::DEPTH_COMPARE,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
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
            &config,
            full_range,
            annulus_range,
        );

        Self {
            land_pipeline,
            water_pipeline,
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
        }
    }

    pub fn ensure_config(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        config: &ClipmapConfig,
    ) {
        if config.rings == self.config.rings
            && config.resolution == self.config.resolution
            && (config.cell_size - self.config.cell_size).abs() < 1e-6
        {
            return;
        }
        *self = Self::new(device, format, config.clone(), &self.shadow_layout.clone());
    }

    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        light_dir: Vec3,
        ambient: f32,
        light_color: Vec3,
        eye: Vec3,
        proc: &ProcTerrain,
    ) {
        let frame = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: [light_dir.x, light_dir.y, light_dir.z],
            ambient,
            light_color: [light_color.x, light_color.y, light_color.z],
            _pad: 0.0,
            eye: [eye.x, eye.y, eye.z],
            _pad2: 0.0,
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

    pub fn draw_land<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        shadow_bind: &'a wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.land_pipeline);
        pass.set_bind_group(1, shadow_bind, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        // Coarse → fine so finer wins depth.
        for ring in self.rings.iter().rev() {
            pass.set_bind_group(0, &ring.bind_group, &[]);
            let end = ring.index_start + ring.index_count;
            pass.draw_indexed(ring.index_start..end, 0, 0..1);
        }
    }

    pub fn draw_water<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        shadow_bind: &'a wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.water_pipeline);
        pass.set_bind_group(1, shadow_bind, &[]);
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

fn build_rings(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    frame_buf: &wgpu::Buffer,
    terrain_buf: &wgpu::Buffer,
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
