//! Draws [`crate::water::Water`]: the simulated surface, the water body seen
//! through it, and the view from under water.
//!
//! Runs after the world pass, on the resolved HDR scene:
//! 1. The scene colour is copied (refraction reads it) and the scene depth is
//!    resolved into a single-sample depth buffer plus a readable copy.
//! 2. Every wet cell (and each dry cell beside one) is an instanced quad. Its
//!    corners take the mean surface of the cells that meet there and are
//!    joined by water, so a body of water is one continuous sheet and dry
//!    quads carry it into the bank: the shoreline is exactly where the water
//!    plane cuts the ground. Shading is physical: Beer-Lambert absorption and
//!    single scattering along the real path length through the water,
//!    Fresnel between screen-space reflections (sky as fallback) and the
//!    refracted scene, GGX sun glint, flow-advected ripples, foam from
//!    breaking fronts, fast flow and shores, and caustics on lit beds.
//! 3. With the eye under water, a full-screen pass fogs everything by the
//!    distance travelled through water, per pixel, so a camera half in and
//!    half out shows both sides of the waterline.

use super::postprocess::HDR_FORMAT;
use crate::water::Water;
use glam::{Mat4, Vec3};

const WATER_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const DEPTH_COPY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_view_proj: [[f32; 4]; 4],
    /// min x, min z, cell size, cells along x.
    region: [f32; 4],
    /// absorption rgb, roughness.
    absorption: [f32; 4],
    /// scatter rgb, ripples.
    scatter: [f32; 4],
    /// ripple length, refraction, foam, caustics.
    look: [f32; 4],
    /// width, height, 1/width, 1/height.
    screen: [f32; 4],
    /// surface height above the eye, eye under water (0/1), time, wind.
    eye_water: [f32; 4],
    /// Open sea: level, sea time, mean depth, 1 if there is a sea.
    sea: [f32; 4],
    /// Region max x, max z, simulation time, unused.
    region_max: [f32; 4],
    /// Swell trains: height (ramped), period, direction degrees, 0.
    swell: [[f32; 4]; SWELLS],
}

/// Swell trains the far sea draws.
const SWELLS: usize = 4;
/// The far sea: rings around the region, segments per ring side.
const SEA_RINGS: u32 = 100;

/// What a frame of water drawing needs from the renderer.
pub struct WaterFrame<'a> {
    pub scene_view: &'a wgpu::TextureView,
    pub scene_texture: &'a wgpu::Texture,
    pub scene_bind: &'a wgpu::BindGroup,
    pub view_proj: Mat4,
    pub eye: Vec3,
    pub time: f32,
    pub wind: f32,
}

struct Targets {
    size: [u32; 2],
    refract: wgpu::Texture,
    refract_view: wgpu::TextureView,
    depth_copy_view: wgpu::TextureView,
    water_depth_view: wgpu::TextureView,
    resolve_bind: wgpu::BindGroup,
    under_bind: wgpu::BindGroup,
}

struct Buffers {
    state: wgpu::Buffer,
    corners: wgpu::Buffer,
    columns: wgpu::Buffer,
    draw: wgpu::Buffer,
    marks: wgpu::Buffer,
    /// GPU simulation only: wet ground cells, and the two indirect draws.
    ground: wgpu::Buffer,
    indirect: wgpu::Buffer,
    /// Per cell along the region's edge, the cell whose surface the far
    /// sea starts from (see `Water::render_edge_cells`).
    edge: wgpu::Buffer,
    cell_capacity: usize,
    draw_capacity: usize,
}

pub struct WaterGpu {
    surface_pipeline: wgpu::RenderPipeline,
    sea_pipeline: wgpu::RenderPipeline,
    ground_pipeline: wgpu::RenderPipeline,
    resolve_pipeline: wgpu::RenderPipeline,
    under_pipeline: wgpu::RenderPipeline,
    surface_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    under_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    targets: Option<Targets>,
    buffers: Option<Buffers>,
    surface_bind: Option<wgpu::BindGroup>,
    /// GPU simulation only: the surface bind group reading the ground list.
    ground_bind: Option<wgpu::BindGroup>,
    /// What the bind groups were made for: buffers rev, GPU sim buffers rev.
    bind_key: (u64, u64),
    /// Bumped when `buffers` are replaced.
    buffers_rev: u64,
    /// Sea edge revision whose edge cells are on the GPU.
    synced_boundary: u64,
    /// Topology revision whose corner table is on the GPU.
    synced_topology: u64,
    state: Vec<[f32; 4]>,
    corners: Vec<[u32; 16]>,
    draw: Vec<u32>,
    depth_multisampled: bool,
    /// The simulation, when the water runs on the GPU.
    sim: Option<crate::water::gpu_sim::GpuWaterSim>,
}

impl WaterGpu {
    pub fn new(
        device: &wgpu::Device,
        scene_layout: &wgpu::BindGroupLayout,
        depth_multisampled: bool,
    ) -> Self {
        let prefix = super::pipeline::scene_shader_prefix();
        let surface_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-surface-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{prefix}{COMMON_WGSL}{SURFACE_WGSL}").into()),
        });
        let under_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-under-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{prefix}{COMMON_WGSL}{UNDER_WGSL}").into()),
        });
        let depth_type = if depth_multisampled {
            "texture_depth_multisampled_2d"
        } else {
            "texture_depth_2d"
        };
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water-depth-resolve-shader"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_WGSL.replace("DEPTH_TYPE", depth_type).into()),
        });

        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture = |binding: u32, sample_type: wgpu::TextureSampleType, ms: bool| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: ms,
                },
                count: None,
            }
        };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let unfilterable = wgpu::TextureSampleType::Float { filterable: false };

        let surface_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-surface-layout"),
            entries: &[
                uniform(0),
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                texture(5, float, false),
                texture(6, unfilterable, false),
                sampler_entry(7),
                storage(8),
                storage(9),
            ],
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-resolve-layout"),
            entries: &[texture(
                0,
                wgpu::TextureSampleType::Depth,
                depth_multisampled,
            )],
        });
        let under_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water-under-layout"),
            entries: &[
                uniform(0),
                texture(1, float, false),
                texture(2, wgpu::TextureSampleType::Depth, false),
                sampler_entry(3),
            ],
        });

        let color_target = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let surface_like = |label: &str, vs: &str, depth_write: bool| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("water-surface-pipeline-layout"),
                bind_group_layouts: &[scene_layout, &surface_layout],
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &surface_shader,
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &surface_shader,
                    entry_point: Some("fs_surface"),
                    targets: &[color_target(HDR_FORMAT)],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: WATER_DEPTH_FORMAT,
                    depth_write_enabled: depth_write,
                    depth_compare: super::DEPTH_COMPARE,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let surface_pipeline = surface_like("water-surface", "vs_surface", true);
        let sea_pipeline = surface_like("water-sea", "vs_sea", true);
        // Wet ground only darkens what is there; it must not hide water.
        let ground_pipeline = surface_like("water-wet-ground", "vs_surface", false);
        let fullscreen = |label: &str,
                          shader: &wgpu::ShaderModule,
                          layouts: &[&wgpu::BindGroupLayout],
                          targets: &[Option<wgpu::ColorTargetState>],
                          depth: Option<wgpu::DepthStencilState>| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_full"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_full"),
                    targets,
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: depth,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let resolve_pipeline = fullscreen(
            "water-depth-resolve",
            &resolve_shader,
            &[&resolve_layout],
            &[color_target(DEPTH_COPY_FORMAT)],
            Some(wgpu::DepthStencilState {
                format: WATER_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );
        let under_pipeline = fullscreen(
            "water-under",
            &under_shader,
            &[scene_layout, &under_layout],
            &[color_target(HDR_FORMAT)],
            None,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            sim: None,
            ground_bind: None,
            bind_key: (u64::MAX, u64::MAX),
            synced_boundary: u64::MAX,
            buffers_rev: 0,
            surface_pipeline,
            sea_pipeline,
            ground_pipeline,
            resolve_pipeline,
            under_pipeline,
            surface_layout,
            resolve_layout,
            under_layout,
            sampler,
            params,
            targets: None,
            buffers: None,
            surface_bind: None,
            synced_topology: 0,
            state: Vec::new(),
            corners: Vec::new(),
            draw: Vec::new(),
            depth_multisampled,
        }
    }

    /// Drop screen-sized targets after a resize (rebuilt on the next frame).
    pub fn resize(&mut self) {
        self.targets = None;
        self.surface_bind = None;
    }

    fn ensure_targets(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
        main_depth: &wgpu::Texture,
    ) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        self.surface_bind = None;
        let make = |label: &str, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let refract = make(
            "water-refract",
            HDR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let depth_copy = make(
            "water-depth-copy",
            DEPTH_COPY_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let water_depth = make(
            "water-depth",
            WATER_DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let refract_view = refract.create_view(&Default::default());
        let depth_copy_view = depth_copy.create_view(&Default::default());
        let water_depth_view = water_depth.create_view(&Default::default());
        let main_depth_view = main_depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("water-main-depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let resolve_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-resolve-bind"),
            layout: &self.resolve_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&main_depth_view),
            }],
        });
        let under_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water-under-bind"),
            layout: &self.under_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&refract_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&water_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.targets = Some(Targets {
            size,
            refract,
            refract_view,
            depth_copy_view,
            water_depth_view,
            resolve_bind,
            under_bind,
        });
    }

    fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        cells: usize,
        draws: usize,
        perimeter: usize,
    ) {
        let fits = self
            .buffers
            .as_ref()
            .is_some_and(|b| b.cell_capacity >= cells.max(perimeter) && b.draw_capacity >= draws);
        if fits {
            return;
        }
        let cell_capacity = cells.max(perimeter).max(1).next_power_of_two();
        let draw_capacity = draws.max(1).next_power_of_two();
        let make_with = |label: &str, size: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let make = |label: &str, size: usize| make_with(label, size, wgpu::BufferUsages::empty());
        self.buffers = Some(Buffers {
            state: make("water-state", cell_capacity * 16),
            corners: make("water-corners", cell_capacity * 64),
            columns: make("water-columns", cell_capacity * 4),
            draw: make("water-draw", draw_capacity * 4),
            marks: make("water-marks", cell_capacity * 16),
            ground: make("water-ground", cell_capacity * 4),
            indirect: make_with("water-indirect", 32, wgpu::BufferUsages::INDIRECT),
            edge: make("water-edge", cell_capacity * 4),
            cell_capacity,
            draw_capacity,
        });
        self.surface_bind = None;
        self.buffers_rev += 1;
        // New buffers start empty: upload the topology again.
        self.synced_topology = 0;
    }

    fn upload_topology(&mut self, queue: &wgpu::Queue, water: &Water) {
        let buffers = self.buffers.as_ref().expect("buffers");
        queue.write_buffer(
            &buffers.edge,
            0,
            bytemuck::cast_slice(&water.render_edge_cells()),
        );
        self.synced_boundary = water.boundary_rev;
        if !self.corners.is_empty() {
            queue.write_buffer(&buffers.corners, 0, bytemuck::cast_slice(&self.corners));
            queue.write_buffer(
                &buffers.columns,
                0,
                bytemuck::cast_slice(water.render_cell_columns()),
            );
        }
        self.synced_topology = water.topology_rev;
    }

    /// Draw the water onto the resolved scene.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        main_depth: &wgpu::Texture,
        water: &Water,
        frame: WaterFrame<'_>,
    ) {
        debug_assert_eq!(main_depth.sample_count() > 1, self.depth_multisampled);
        let ext = frame.scene_texture.size();
        let size = [ext.width.max(1), ext.height.max(1)];
        self.ensure_targets(device, size, main_depth);
        if water.on_gpu() {
            self.sim
                .get_or_insert_with(|| crate::water::gpu_sim::GpuWaterSim::new(device))
                .encode_frame(device, queue, encoder, water);
        }

        // Simulation state for this frame.
        let topology = water.topology_rev;
        if topology != self.synced_topology || self.corners.len() != water.cell_count() {
            self.corners = water.render_corners();
        }
        let on_gpu = self.sim.is_some() && water.on_gpu();
        let region = water.region();
        let perimeter = 2 * (region.cells_x + region.cells_z) as usize;
        let water_draws = if on_gpu {
            // Straight from the GPU state: no copies through the CPU.
            let cells = water.cell_count().max(1);
            self.ensure_buffers(device, cells, cells, perimeter);
            if topology != self.synced_topology || self.synced_boundary != water.boundary_rev {
                self.upload_topology(queue, water);
            }
            let buffers = self.buffers.as_ref().expect("buffers");
            queue.write_buffer(
                &buffers.indirect,
                0,
                bytemuck::cast_slice(&[6u32, 0, 0, 0, 6, 0, 0, 0]),
            );
            let sim = self.sim.as_mut().expect("sim");
            sim.encode_draw(
                device,
                encoder,
                crate::water::gpu_sim::DrawTargets {
                    state: &buffers.state,
                    corners: &buffers.corners,
                    water_list: &buffers.draw,
                    ground_list: &buffers.ground,
                    indirect: &buffers.indirect,
                    rev: self.buffers_rev,
                },
            );
            0
        } else {
            water.render_state(&mut self.state);
            let mut draw = std::mem::take(&mut self.draw);
            let water_draws = water.render_draw_list(&self.state, &self.corners, &mut draw) as u32;
            self.draw = draw;
            self.ensure_buffers(device, self.state.len(), self.draw.len(), perimeter);
            if topology != self.synced_topology || self.synced_boundary != water.boundary_rev {
                self.upload_topology(queue, water);
            }
            let buffers = self.buffers.as_ref().expect("buffers");
            if !self.state.is_empty() {
                queue.write_buffer(&buffers.state, 0, bytemuck::cast_slice(&self.state));
                queue.write_buffer(&buffers.marks, 0, bytemuck::cast_slice(&water.marks));
            }
            if !self.draw.is_empty() {
                queue.write_buffer(&buffers.draw, 0, bytemuck::cast_slice(&self.draw));
            }
            water_draws
        };

        // Is the eye in water, and how high is the surface above it?
        let sample = water.sample(frame.eye);
        let region = water.region();
        let outside = {
            let (lo, hi) = (region.min_xz, region.max_xz());
            frame.eye.x < lo.x || frame.eye.z < lo.y || frame.eye.x > hi.x || frame.eye.z > hi.y
        };
        let open_sea = water.sea().filter(|_| outside).map(|sea| {
            let xz = glam::Vec2::new(frame.eye.x, frame.eye.z);
            sea.level + sea.incoming(xz, water.sea_time(), water.sea_depth())
        });
        let (eye_surface, eye_under) = match (sample, open_sea) {
            (_, Some(y)) => (y, if frame.eye.y < y { 1.0 } else { 0.0 }),
            (Some(s), _) if s.submerged() => (s.surface_y, 1.0),
            (Some(s), _) => (s.surface_y, 0.0),
            (None, None) => (frame.eye.y - 1e4, 0.0),
        };
        let in_water = sample.is_some() || open_sea.is_some();
        let look = water.look();
        let params = Params {
            inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
            region: [
                region.min_xz.x,
                region.min_xz.y,
                region.cell_size,
                region.cells_x as f32,
            ],
            absorption: [
                look.absorption[0],
                look.absorption[1],
                look.absorption[2],
                look.roughness,
            ],
            scatter: [
                look.scatter[0],
                look.scatter[1],
                look.scatter[2],
                look.ripples,
            ],
            look: [
                look.ripple_length_m,
                look.refraction,
                look.foam,
                look.caustics,
            ],
            screen: [
                size[0] as f32,
                size[1] as f32,
                1.0 / size[0] as f32,
                1.0 / size[1] as f32,
            ],
            eye_water: [eye_surface, eye_under, frame.time, frame.wind],
            sea: match water.sea() {
                Some(sea) => [sea.level, water.sea_time(), water.sea_depth(), 1.0],
                None => [0.0; 4],
            },
            region_max: [
                region.max_xz().x,
                region.max_xz().y,
                water.time() as f32,
                0.0,
            ],
            swell: {
                let mut out = [[0.0; 4]; SWELLS];
                if let Some(sea) = water.sea() {
                    let ramp = sea.ramp(water.sea_time());
                    for (o, s) in out.iter_mut().zip(&sea.swell) {
                        *o = [s.height * ramp, s.period, s.direction_deg, 0.0];
                    }
                }
                out
            },
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let targets = self.targets.as_ref().expect("targets");
        let buffers = self.buffers.as_ref().expect("buffers");
        let sim_marks = self
            .sim
            .as_ref()
            .filter(|_| on_gpu)
            .and_then(|s| s.marks_buffer());
        let key = (
            self.buffers_rev,
            sim_marks.map_or(u64::MAX, |_| {
                self.sim.as_ref().map_or(0, |s| s.buffers_rev())
            }),
        );
        if self.surface_bind.is_none() || self.bind_key != key {
            let make = |draw: &wgpu::Buffer, marks: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("water-surface-bind"),
                    layout: &self.surface_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.state.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffers.corners.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buffers.columns.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: draw.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&targets.refract_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&targets.depth_copy_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: marks.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: buffers.edge.as_entire_binding(),
                        },
                    ],
                })
            };
            let marks = sim_marks.unwrap_or(&buffers.marks);
            self.surface_bind = Some(make(&buffers.draw, marks));
            self.ground_bind = sim_marks.map(|m| make(&buffers.ground, m));
            self.bind_key = key;
        }
        let surface_bind = self.surface_bind.as_ref().expect("surface bind");

        let copy_scene = |encoder: &mut wgpu::CommandEncoder| {
            encoder.copy_texture_to_texture(
                frame.scene_texture.as_image_copy(),
                targets.refract.as_image_copy(),
                wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
            );
        };

        // 1. Scene colour for refraction; scene depth, resolved.
        copy_scene(encoder);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water-depth-resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.depth_copy_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.water_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(super::DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &targets.resolve_bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. The surface, then the open sea around it.
        let has_sea = water.sea().is_some();
        if on_gpu || !self.draw.is_empty() || has_sea {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water-surface"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.water_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(0, frame.scene_bind, &[]);
            pass.set_bind_group(1, surface_bind, &[]);
            // Wet ground first, so water drawn over it stays on top.
            if let (true, Some(ground_bind)) = (on_gpu, self.ground_bind.as_ref()) {
                pass.set_bind_group(1, ground_bind, &[]);
                pass.set_pipeline(&self.ground_pipeline);
                pass.draw_indirect(&buffers.indirect, 16);
                pass.set_bind_group(1, surface_bind, &[]);
                pass.set_pipeline(&self.surface_pipeline);
                pass.draw_indirect(&buffers.indirect, 0);
            } else if (self.draw.len() as u32) > water_draws {
                pass.set_pipeline(&self.ground_pipeline);
                pass.draw(0..6, water_draws..self.draw.len() as u32);
            }
            if water_draws > 0 {
                pass.set_pipeline(&self.surface_pipeline);
                pass.draw(0..6, 0..water_draws);
            }
            if has_sea {
                pass.set_pipeline(&self.sea_pipeline);
                pass.draw(0..6, 0..SEA_RINGS * perimeter as u32);
            }
        }

        // 3. Under water: fog by distance through water.
        if eye_surface > frame.eye.y - 2.0 && in_water {
            copy_scene(encoder);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water-under"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.under_pipeline);
            pass.set_bind_group(0, frame.scene_bind, &[]);
            pass.set_bind_group(1, &targets.under_bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// Fullscreen triangle plus helpers shared by the surface and underwater passes.
const COMMON_WGSL: &str = r#"
struct WaterParams {
    inv_view_proj: mat4x4<f32>,
    region: vec4<f32>,
    absorption: vec4<f32>,
    scatter: vec4<f32>,
    look: vec4<f32>,
    screen: vec4<f32>,
    eye_water: vec4<f32>,
    sea: vec4<f32>,
    region_max: vec4<f32>,
    swell: array<vec4<f32>, 4>,
};

// Incoming swell above sea level at `p` (matches water::Sea::incoming);
// xyz = height, d/dx, d/dz. Waves shorter than `fade_len` fade out.
fn swell_at(p: vec2<f32>, fade_len: f32) -> vec3<f32> {
    let t = wp.sea.y;
    let c = sqrt(9.81 * max(wp.sea.z, 0.5));
    var out = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let s = wp.swell[i];
        if s.x == 0.0 {
            continue;
        }
        let omega = 6.2831853 / max(s.y, 0.5);
        let a = radians(s.z);
        let k = vec2<f32>(sin(a), cos(a)) * (omega / c);
        let wavelength = 6.2831853 / length(k);
        let amp = 0.5 * s.x * smoothstep(0.5 * wavelength, 0.25 * wavelength, fade_len);
        let phase = omega * t - dot(k, p) + f32(i) * 1.3;
        out += vec3<f32>(amp * cos(phase), amp * sin(phase) * k);
    }
    return out;
}

// World position of a pixel (uv 0..1, y down) at a raw depth.
fn world_at(uv: vec2<f32>, raw_depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, raw_depth, 1.0);
    let w = wp.inv_view_proj * ndc;
    return w.xyz / w.w;
}

fn sigma_t() -> vec3<f32> {
    return max(wp.absorption.rgb + wp.scatter.rgb, vec3<f32>(1e-4));
}

// Light reaching water near the surface: sun through the surface plus sky.
fn water_light(vis: f32) -> vec3<f32> {
    let l = normalize(u.light_dir);
    let sun = u.light_color * max(l.y, 0.0) * (1.0 - u.ambient) * vis;
    return sun + u.sky_up * u.ambient;
}

// Colour scattered toward the eye by `dist` metres of water lit by `light`
// at `below` metres under the surface.
fn inscatter(dist: f32, below: f32, light: vec3<f32>) -> vec3<f32> {
    let st = sigma_t();
    let reach = exp(-st * max(below, 0.0));
    return wp.scatter.rgb / st * (vec3<f32>(1.0) - exp(-st * dist)) * light * reach * 0.6;
}

fn water_hash(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn water_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = water_hash(i).x;
    let b = water_hash(i + vec2<f32>(1.0, 0.0)).x;
    let c = water_hash(i + vec2<f32>(0.0, 1.0)).x;
    let d = water_hash(i + vec2<f32>(1.0, 1.0)).x;
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

// Bright filaments where light bunches up under a wavy surface: edges of an
// animated Voronoi pattern, two layers drifting apart.
fn caustic_layer(p: vec2<f32>, t: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2<f32>(f32(x), f32(y));
            let h = water_hash(i + o);
            let pt = o + 0.5 + 0.42 * sin(t * (0.6 + h * 0.5) + h * 6.2831);
            let d = distance(f, pt);
            if d < d1 {
                d2 = d1;
                d1 = d;
            } else if d < d2 {
                d2 = d;
            }
        }
    }
    return pow(clamp(1.0 - (d2 - d1) * 2.2, 0.0, 1.0), 5.0);
}

fn caustics(xz: vec2<f32>, below: f32) -> f32 {
    if below <= 0.02 || wp.look.w <= 0.0 {
        return 0.0;
    }
    let t = wp.eye_water.z;
    let c = caustic_layer(xz * 0.55, t) + caustic_layer(xz * 0.55 * 1.37 + vec2<f32>(7.1, 3.4), t * 1.21);
    let fade = smoothstep(0.02, 0.5, below) * exp(-below * 0.12);
    let l = normalize(u.light_dir);
    return c * fade * max(l.y, 0.0) * wp.look.w * 0.9;
}
"#;

const SURFACE_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> wp: WaterParams;
@group(1) @binding(1) var<storage, read> cells: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> corners: array<u32>;
@group(1) @binding(3) var<storage, read> cell_col: array<u32>;
@group(1) @binding(4) var<storage, read> draw_list: array<u32>;
@group(1) @binding(5) var refract_tex: texture_2d<f32>;
@group(1) @binding(6) var depth_tex: texture_2d<f32>;
@group(1) @binding(7) var lin: sampler;
// Per cell: foam, when it was set, when the cell was last wet.
@group(1) @binding(8) var<storage, read> marks: array<vec4<f32>>;
@group(1) @binding(9) var<storage, read> edge_cells: array<u32>;

const NO_CELL: u32 = 0xffffffffu;
const SAND_BIT: u32 = 0x80000000u;
const FOAM_SECONDS: f32 = 5.0;
const DRYING_SECONDS: f32 = 30.0;

fn foam_of(i: u32) -> f32 {
    let m = marks[i];
    return m.x * exp(-(wp.region_max.z - m.y) / FOAM_SECONDS);
}

fn wetness_of(i: u32) -> f32 {
    return exp(-max(wp.region_max.z - marks[i].z, 0.0) / DRYING_SECONDS);
}
const DRAW_DEPTH: f32 = 0.004;

fn is_wet(i: u32) -> bool {
    if i == NO_CELL {
        return false;
    }
    return cells[i].y > DRAW_DEPTH;
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_p: vec3<f32>,
    @location(1) grad: vec2<f32>,
    @location(2) flow: vec2<f32>,
    @location(3) depth: f32,
    // Foam (water) or wetness (wet ground, where depth < 0).
    @location(4) foam: f32,
};

@vertex
fn vs_surface(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let entry = draw_list[ii];
    let cell = entry & ~SAND_BIT;
    var order = array<u32, 6>(0u, 1u, 3u, 0u, 3u, 2u);
    let k = order[vi];
    let cells_x = u32(wp.region.w);
    let col = cell_col[cell];
    let cx = col % cells_x;
    let cz = col / cells_x;
    let xz = wp.region.xy + vec2<f32>(f32(cx + (k & 1u)), f32(cz + (k >> 1u))) * wp.region.z;

    // Wet ground: a quad a little above the ground that darkens what is
    // behind it (the fragment shader checks it is the ground).
    if (entry & SAND_BIT) != 0u {
        var ground = 0.0;
        var wet = 0.0;
        var n = 0.0;
        for (var j = 0u; j < 4u; j++) {
            let c = corners[cell * 16u + k * 4u + j];
            if c != NO_CELL && cells[c].y >= 0.0 && cells[c].y <= DRAW_DEPTH {
                ground += cells[c].x - cells[c].y;
                wet += wetness_of(c);
                n += 1.0;
            }
        }
        var sand: VsOut;
        sand.world_p = vec3<f32>(xz.x, ground / max(n, 1.0) + 0.25, xz.y);
        sand.clip = u.view_proj * vec4<f32>(sand.world_p, 1.0);
        sand.grad = vec2<f32>(0.0);
        sand.flow = vec2<f32>(0.0);
        sand.depth = -1.0;
        sand.foam = wet / max(n, 1.0);
        return sand;
    }

    // A dry quad beside water sits at its neighbour's level and runs into
    // the bank, where the depth test trims it to the true shoreline.
    var fallback = -1e9;
    if is_wet(cell) {
        fallback = cells[cell].x;
    } else {
        for (var j = 0u; j < 16u; j++) {
            let n = corners[cell * 16u + j];
            if is_wet(n) {
                fallback = max(fallback, cells[n].x);
            }
        }
    }
    // Water level at this corner: mean of the wet cells meeting here.
    var level = 0.0;
    var wet_count = 0.0;
    var flow = vec2<f32>(0.0);
    var depth = 0.0;
    var foam = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let n = corners[cell * 16u + k * 4u + j];
        if is_wet(n) {
            let c = cells[n];
            level += c.x;
            wet_count += 1.0;
            flow += c.zw;
            depth += c.y;
            foam += foam_of(n);
        }
    }
    if wet_count > 0.0 {
        level /= wet_count;
        flow /= wet_count;
        depth /= wet_count;
        foam /= wet_count;
    } else {
        level = fallback;
    }
    // Each cell around the corner votes for a height: wet ones their
    // surface; dry ones the water level where their ground is higher (a
    // bank: the flat plane runs into it and the depth test draws the
    // shoreline), else their ground (a sheet thinning out downhill lies on
    // the slope instead of floating off it).
    var h: array<f32, 4>;
    var sum = 0.0;
    var count = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let n = corners[cell * 16u + k * 4u + j];
        if n == NO_CELL {
            h[j] = level;
            continue;
        }
        let c = cells[n];
        var v = c.x;
        if !is_wet(n) {
            let ground = select(c.x - c.y, c.x, c.y <= 0.0);
            v = min(ground - 0.03, level);
        }
        h[j] = v;
        sum += v;
        count += 1.0;
    }
    let height = select(level, sum / count, count > 0.0);
    let inv = 0.5 / wp.region.z;
    let grad = vec2<f32>((h[1] + h[3]) - (h[0] + h[2]), (h[2] + h[3]) - (h[0] + h[1])) * inv;

    var out: VsOut;
    out.world_p = vec3<f32>(xz.x, height, xz.y);
    out.clip = u.view_proj * vec4<f32>(out.world_p, 1.0);
    out.grad = grad;
    out.flow = flow;
    out.depth = depth;
    out.foam = foam;
    return out;
}

// Distance of far-sea ring `r` outside the region: 3 m first, then
// growing by 7 % per ring (100 rings reach about 40 km).
fn sea_ring(r: f32) -> f32 {
    return 3.0 * (pow(1.07, r) - 1.0) / 0.07;
}

fn cells_z() -> u32 {
    return u32(round((wp.region_max.y - wp.region.y) / wp.region.z));
}

// The far sea has one segment per edge cell on every ring.
fn perimeter() -> u32 {
    return 2u * (u32(wp.region.w) + cells_z());
}

// Perimeter step `s` as (side, position along it, side length), walking
// +x along z = min, +z along x = max, -x along z = max, -z along x = min.
fn sea_side(s: u32) -> vec3<u32> {
    let nx = u32(wp.region.w);
    let nz = cells_z();
    let p = s % perimeter();
    if p < nx {
        return vec3<u32>(0u, p, nx);
    }
    if p < nx + nz {
        return vec3<u32>(1u, p - nx, nz);
    }
    if p < 2u * nx + nz {
        return vec3<u32>(2u, p - nx - nz, nx);
    }
    return vec3<u32>(3u, p - 2u * nx - nz, nz);
}

// A point on ring `r` at perimeter step `s`.
fn sea_point(s: u32, r: u32) -> vec2<f32> {
    let side = sea_side(s);
    let t = f32(side.y) / f32(side.z);
    let e = sea_ring(f32(r));
    let lo = wp.region.xy - vec2<f32>(e);
    let hi = wp.region_max.xy + vec2<f32>(e);
    switch side.x {
        case 0u: { return vec2<f32>(mix(lo.x, hi.x, t), lo.y); }
        case 1u: { return vec2<f32>(hi.x, mix(lo.y, hi.y, t)); }
        case 2u: { return vec2<f32>(mix(hi.x, lo.x, t), hi.y); }
        default: { return vec2<f32>(lo.x, mix(hi.y, lo.y, t)); }
    }
}

// The simulated water where perimeter step `s` meets the edge, exactly as
// the edge of the surface mesh has it: surface height, depth, current, and
// how many of the two edge cells there are wet.
struct Edge {
    height: f32,
    depth: f32,
    flow: vec2<f32>,
    wet: f32,
};

fn edge_at(s: u32) -> Edge {
    let n = perimeter();
    var e: Edge;
    e.height = 0.0;
    e.depth = 0.0;
    e.flow = vec2<f32>(0.0);
    e.wet = 0.0;
    for (var j = 0u; j < 2u; j++) {
        let c = edge_cells[(s + n - 1u + j) % n];
        if is_wet(c) {
            e.height += cells[c].x;
            e.depth += cells[c].y;
            e.flow += cells[c].zw;
            e.wet += 1.0;
        }
    }
    let k = 1.0 / max(e.wet, 1.0);
    e.height *= k;
    e.depth *= k;
    e.flow *= k;
    return e;
}

// Along the side and outwards, for perimeter step `s`.
fn sea_axes(s: u32) -> array<vec2<f32>, 2> {
    switch sea_side(s).x {
        case 0u: { return array<vec2<f32>, 2>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, -1.0)); }
        case 1u: { return array<vec2<f32>, 2>(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0)); }
        case 2u: { return array<vec2<f32>, 2>(vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 1.0)); }
        default: { return array<vec2<f32>, 2>(vec2<f32>(0.0, -1.0), vec2<f32>(-1.0, 0.0)); }
    }
}

// Width (m) of the band outside the region over which the simulated edge
// (its waves, currents and depth) turns into the open sea.
const SEA_BLEND: f32 = 60.0;

// The open sea beyond the simulated region, with the incoming swell. It
// starts as the simulated edge and turns into the open-sea swell over
// SEA_BLEND metres, so neither the level nor the slope steps at the edge.
@vertex
fn vs_sea(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var order = array<u32, 6>(0u, 1u, 3u, 0u, 3u, 2u);
    let k = order[vi];
    let per_ring = perimeter();
    let s = ii % per_ring + (k & 1u);
    let r = ii / per_ring + (k >> 1u);
    let xz = sea_point(s, r);
    // Spacing of the mesh here: waves much shorter only alias.
    let spacing = max(sea_ring(f32(r) + 1.0) - sea_ring(f32(r)), distance(sea_point(s + 1u, r), xz));
    let w = swell_at(xz, spacing);
    let open_y = wp.sea.x + w.x;
    var y = open_y;
    var grad = w.yz;
    var flow = vec2<f32>(0.0);
    var depth = max(wp.sea.z, 2.0);
    let e = edge_at(s);
    let x = sea_ring(f32(r)) / SEA_BLEND;
    if e.wet > 0.0 && x < 1.0 {
        let f = x * x * (3.0 - 2.0 * x);
        let df = 6.0 * x * (1.0 - x) / SEA_BLEND;
        let axes = sea_axes(s);
        // Slope of the simulated edge along the side.
        let e0 = edge_at(s + per_ring - 1u);
        let e1 = edge_at(s + 1u);
        var along = 0.0;
        if e0.wet > 0.0 && e1.wet > 0.0 {
            along = (e1.height - e0.height) * 0.5 / wp.region.z;
        }
        y = mix(e.height, open_y, f);
        grad = mix(axes[0] * along, w.yz, f) + axes[1] * (open_y - e.height) * df;
        flow = e.flow * (1.0 - f);
        depth = mix(max(e.depth, 0.05), depth, f);
    }
    var out: VsOut;
    out.world_p = vec3<f32>(xz.x, y, xz.y);
    out.clip = u.view_proj * vec4<f32>(out.world_p, 1.0);
    out.grad = grad;
    out.flow = flow;
    out.depth = depth;
    out.foam = 0.0;
    return out;
}

// Slope of a handful of travelling deep-water waves around the wind.
// `footprint` is the size of a pixel on the surface: waves much shorter
// than that only alias, so they fade out.
fn ripple_grad(p: vec2<f32>, t: f32, len: f32, footprint: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    var l = len;
    var angle = 0.4;
    for (var i = 0; i < 6; i++) {
        let d = vec2<f32>(cos(angle), sin(angle));
        let k = 6.2831853 / l;
        let w = sqrt(9.81 * k);
        let phase = k * dot(d, p) - w * t + f32(i) * 1.7;
        let amp = l * 0.011 * smoothstep(2.0 * footprint, 6.0 * footprint, l);
        g += d * (amp * k * cos(phase));
        l *= 0.62;
        angle += 2.39996;
    }
    return g;
}

// Ripples carried along by the current: two copies advected on staggered
// cycles and cross-faded, so they flow without stretching.
fn flowing_ripples(p: vec2<f32>, flow: vec2<f32>, t: f32, len: f32, footprint: f32) -> vec2<f32> {
    let cycle = 3.0;
    let ph0 = fract(t / cycle);
    let ph1 = fract(t / cycle + 0.5);
    let g0 = ripple_grad(p - flow * (ph0 * cycle), t, len, footprint);
    let g1 = ripple_grad(p - flow * (ph1 * cycle) + vec2<f32>(17.3, 9.1), t, len, footprint);
    let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
    return g0 * w0 + g1 * (1.0 - w0);
}

fn flowing_noise(p: vec2<f32>, flow: vec2<f32>, t: f32) -> f32 {
    let cycle = 2.0;
    let ph0 = fract(t / cycle);
    let ph1 = fract(t / cycle + 0.5);
    let a = water_noise(p - flow * (ph0 * cycle)) * 0.65 + water_noise((p - flow * (ph0 * cycle)) * 2.7) * 0.35;
    let b = water_noise(p - flow * (ph1 * cycle) + 5.3) * 0.65 + water_noise((p - flow * (ph1 * cycle)) * 2.7 + 3.1) * 0.35;
    let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
    return a * w0 + b * (1.0 - w0);
}

fn scene_depth(pix: vec2<f32>) -> f32 {
    let size = vec2<i32>(wp.screen.xy);
    let c = clamp(vec2<i32>(pix), vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(depth_tex, c, 0).r;
}

// The sky as drawn on screen in direction `r`: the pixel where that
// direction lies (clamped to the screen edge), if it shows sky. Matches the
// drawn sky far better than evaluating a sky model. w = 0 if not found.
fn screen_sky(r: vec3<f32>) -> vec4<f32> {
    let c = u.view_proj * vec4<f32>(r, 0.0);
    if c.w <= 1e-4 {
        return vec4<f32>(0.0);
    }
    let ndc = c.xy / c.w;
    let uv = clamp(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5), vec2<f32>(0.002), vec2<f32>(0.998));
    // Step toward the top of the screen until the sky shows.
    var q = uv;
    for (var i = 0; i < 6; i++) {
        if scene_depth(q * wp.screen.xy) <= 0.0 {
            return vec4<f32>(textureSampleLevel(refract_tex, lin, q, 0.0).rgb, 1.0);
        }
        q.y = max(q.y - 0.08, 0.002);
    }
    return vec4<f32>(0.0);
}

// Screen-space reflection: march the reflected ray against the scene depth.
// Returns colour and confidence (0 = fall back to the sky).
fn reflect_ray(p: vec3<f32>, r: vec3<f32>) -> vec4<f32> {
    var t = 0.4;
    for (var i = 0; i < 28; i++) {
        let q = p + r * t;
        let c = u.view_proj * vec4<f32>(q, 1.0);
        if c.w <= 0.0 {
            break;
        }
        let ndc = c.xyz / c.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
            break;
        }
        let d = scene_depth(uv * wp.screen.xy);
        if d > 0.0 {
            let s = world_at(uv, d);
            let ray_dist = distance(u.eye, q);
            let scene_dist = distance(u.eye, s);
            if ray_dist > scene_dist && ray_dist - scene_dist < t * 0.35 + 0.3 {
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                let conf = smoothstep(0.0, 0.08, edge);
                let hit = textureSampleLevel(refract_tex, lin, uv, 0.0).rgb;
                let sky = screen_sky(r);
                if sky.w > 0.0 {
                    return vec4<f32>(mix(sky.rgb, hit, conf), 1.0);
                }
                return vec4<f32>(hit, conf);
            }
        } else if t > 30.0 {
            // Reached open sky: nothing nearer is reflected. The sky is at
            // infinity, so it is the sky in the ray's direction, not the
            // sky behind this point on screen (that would be the haze at
            // the horizon).
            break;
        }
        t *= 1.22;
    }
    return screen_sky(r);
}

fn scene_light() -> SceneLight {
    var light: SceneLight;
    light.dir = normalize(u.light_dir);
    light.color = u.light_color;
    light.ambient = u.ambient;
    light.sky_up = u.sky_up;
    light.sky_horizon = u.sky_horizon;
    light.sky_down = u.sky_down;
    light.env_strength = u.env_strength;
    light.ao = 1.0;
    light.gi = vec4<f32>(0.0);
    return light;
}

// Ground that water has just left: darker and a little glossy, drying out.
fn wet_ground(in: VsOut, uv: vec2<f32>) -> vec4<f32> {
    let d = scene_depth(in.clip.xy);
    if d <= 0.0 || in.foam < 0.02 {
        discard;
    }
    let ground = world_at(uv, d);
    // Only the ground itself, not things standing on it.
    if abs(ground.y - (in.world_p.y - 0.25)) > 0.3 {
        discard;
    }
    let scene = textureSampleLevel(refract_tex, lin, uv, 0.0).rgb;
    let wet = clamp(in.foam, 0.0, 1.0);
    // Water in the pores: darker, more saturated; a sheen while very wet.
    let grey = dot(scene, vec3<f32>(0.333));
    var color = mix(scene, mix(vec3<f32>(grey), scene, 1.25) * 0.55, wet);
    let v = normalize(u.eye - ground);
    let r = reflect(-v, vec3<f32>(0.0, 1.0, 0.0));
    let sky = screen_sky(r);
    let fresnel = 0.02 + 0.98 * pow(1.0 - max(v.y, 0.0), 5.0);
    if sky.w > 0.0 {
        color = mix(color, sky.rgb, fresnel * smoothstep(0.7, 1.0, wet) * 0.15);
    }
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_surface(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_p;
    let pix = in.clip.xy;
    let uv = pix * wp.screen.zw;
    if in.depth < 0.0 {
        return wet_ground(in, uv);
    }
    let to_eye = u.eye - p;
    let dist = length(to_eye);
    let v = to_eye / max(dist, 1e-4);
    let t = wp.eye_water.z;

    let ng = normalize(vec3<f32>(-in.grad.x, 1.0, -in.grad.y));
    let above = dot(to_eye, ng) > 0.0;

    // What lies behind the surface along this pixel.
    let d0 = scene_depth(pix);
    var behind = p + (p - u.eye) / max(dist, 1e-4) * 1e4;
    if d0 > 0.0 {
        behind = world_at(uv, d0);
    }
    let thickness = max(distance(p, behind) , 0.0);
    // Vertical water depth under this pixel (for shores and foam).
    let pix_depth = clamp(p.y - behind.y, 0.0, 1e3);

    // Ripples: wind, flow turbulence and depth, fading out with distance
    // before they alias.
    let speed = length(in.flow);
    let calm_shallow = smoothstep(0.0, 0.25, min(in.depth, pix_depth));
    let strength = wp.scatter.w * (0.45 + 0.55 * u.wind + 0.5 * min(speed, 3.0))
        * calm_shallow * clamp(1.0 - dist / 600.0, 0.15, 1.0);
    // Pixel size on the surface, stretched at grazing angles.
    let footprint = dist * 1.2 * wp.screen.w / max(abs(v.y), 0.08);
    let g = flowing_ripples(p.xz, in.flow, t, max(wp.look.x, 0.3), footprint) * strength;
    var n = normalize(vec3<f32>(-(in.grad.x + g.x), 1.0, -(in.grad.y + g.y)));
    let light = scene_light();
    let l = light.dir;
    let vis = sun_visibility(p, ng, u.eye);
    let st = sigma_t();
    var color: vec3<f32>;

    if above {
        // Refraction: bend the view by the surface tilt, more for thicker water.
        let bend = wp.look.y * 0.035 * clamp(thickness * 0.5, 0.0, 1.0) / (1.0 + dist * 0.03);
        var uv_r = uv + n.xz * bend;
        var d_r = scene_depth(uv_r * wp.screen.xy);
        var behind_r = behind;
        if d_r > 0.0 {
            behind_r = world_at(uv_r, d_r);
        }
        // Something above the water at the bent spot: do not pull it in.
        if distance(u.eye, behind_r) < dist {
            uv_r = uv;
            behind_r = behind;
        }
        var refr = textureSampleLevel(refract_tex, lin, uv_r, 0.0).rgb;
        let path = distance(p, behind_r);
        let bed_below = max(p.y - behind_r.y, 0.0);
        // Caustics brighten lit ground under the water.
        refr *= 1.0 + caustics(behind_r.xz, bed_below) * vis;
        // Light reaching the bed was dimmed on the way down as well.
        let sun_path = bed_below / max(l.y, 0.25);
        let trans = exp(-st * (path + sun_path));
        let body = refr * trans + inscatter(path, 0.0, water_light(vis));

        // Reflection: screen-space where possible, sky elsewhere.
        var r = reflect(-v, n);
        r.y = abs(r.y);
        let rough = wp.absorption.w;
        let sky = sky_radiance(r, rough, light) * u.ambient * u.env_strength;
        let ssr = reflect_ray(p + n * 0.05, r);
        let refl = mix(sky, ssr.rgb, ssr.a);
        let nv = max(dot(n, v), 0.0);
        // A film over grass and stones is broken up, not a mirror.
        let sheet = smoothstep(0.02, 0.3, min(in.depth, pix_depth));
        let fresnel = (0.02 + 0.98 * pow(1.0 - nv, 5.0)) * mix(0.25, 1.0, sheet);

        // Sun glint (GGX).
        var spec = vec3<f32>(0.0);
        let nl = dot(n, l);
        if nl > 0.0 {
            let hv = normalize(l + v);
            let nh = max(dot(n, hv), 0.0);
            let a = max(rough * rough, 0.002);
            let a2 = a * a;
            let dd = nh * nh * (a2 - 1.0) + 1.0;
            let ggx = a2 / (3.14159 * dd * dd);
            let kk = (rough + 1.0) * (rough + 1.0) / 8.0;
            let gv = (nv / (nv * (1.0 - kk) + kk)) * (nl / (nl * (1.0 - kk) + kk));
            spec = u.light_color * (1.0 - u.ambient) * vis * ggx * gv * fresnel / (4.0 * max(nv, 1e-3)) * 3.14159;
        }
        color = mix(body, refl, fresnel) + spec;

        // A film of water at the very edge shows only what is under it.
        let film = smoothstep(0.0, 0.04, pix_depth);
        let plain = textureSampleLevel(refract_tex, lin, uv, 0.0).rgb;
        color = mix(plain, color, film);
        // Foam: breaking fronts, fast water, shorelines.
        // Supercritical flow (Froude number above 1) is where water breaks:
        // rapids, hydraulic jumps, bores and fronts running up a beach.
        let froude = speed / sqrt(9.81 * max(in.depth, 0.01));
        let turbulence = smoothstep(0.7, 1.8, froude) * smoothstep(0.02, 0.15, in.depth)
            + smoothstep(2.5, 6.0, speed);
        let shore = 1.0 - smoothstep(0.0, 0.25, pix_depth);
        let amount = clamp((turbulence * 0.7 + shore * 0.25 + in.foam) * wp.look.z, 0.0, 1.0);
        // Lacy whitewater: broad patches broken into bubbly strands, which
        // close up into solid white where the foam is thick.
        let broad = flowing_noise(p.xz * 0.9, in.flow * 0.9, t);
        let fine = flowing_noise(p.xz * 4.1 + 13.7, in.flow * 4.1, t * 1.3);
        let lace = broad * 0.55 + (1.0 - abs(fine * 2.0 - 1.0)) * 0.45;
        let edge = 0.86 - amount * 0.42;
        let foam = smoothstep(edge, edge + 0.1, lace) * min(amount * 1.5, 0.95);
        let foam_light = (u.light_color * max(nl, 0.0) * (1.0 - u.ambient) * vis + sky_irradiance(vec3<f32>(0.0, 1.0, 0.0), light) * u.ambient) * 0.95;
        // Foam rides even the thinnest swash.
        color = mix(color, foam_light, clamp(foam, 0.0, 1.0) * smoothstep(0.0, 0.01, pix_depth));

        color = haze(color, p);
    } else {
        // Seen from below: Snell's window to the air, mirror outside it.
        let nb = -n;
        let ri = refract(-v, nb, 1.33);
        let deep = inscatter(1e3, max(wp.eye_water.x - u.eye.y, 0.0), water_light(1.0));
        if dot(ri, ri) < 1e-6 {
            color = deep;
        } else {
            let nv = max(dot(nb, v), 0.0);
            let fresnel = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
            let uv_r = uv + n.xz * 0.04 * wp.look.y;
            let air = textureSampleLevel(refract_tex, lin, uv_r, 0.0).rgb;
            color = mix(air, deep, clamp(fresnel, 0.0, 1.0));
        }
    }
    return vec4<f32>(color, 1.0);
}
"#;

const UNDER_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> wp: WaterParams;
@group(1) @binding(1) var scene_tex: texture_2d<f32>;
@group(1) @binding(2) var depth_tex: texture_depth_2d;
@group(1) @binding(3) var lin: sampler;

struct FsIn {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> FsIn {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: FsIn;
    out.clip = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

@fragment
fn fs_full(in: FsIn) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.clip.xy);
    let uv = in.clip.xy * wp.screen.zw;
    let scene = textureLoad(scene_tex, pix, 0).rgb;
    let surface_y = wp.eye_water.x;
    // Which side of the waterline this pixel's view starts on.
    let near = world_at(uv, 1.0);
    if wp.eye_water.y < 0.5 && near.y > surface_y {
        return vec4<f32>(scene, 1.0);
    }
    if near.y > surface_y {
        return vec4<f32>(scene, 1.0);
    }
    let raw = textureLoad(depth_tex, pix, 0);
    var p = u.eye + normalize(world_at(uv, 0.5) - u.eye) * 1e3;
    if raw > 0.0 {
        p = world_at(uv, raw);
    }
    let d = distance(u.eye, p);
    let below_eye = max(surface_y - u.eye.y, 0.0);
    var color = scene;
    let bed_below = surface_y - p.y;
    color *= 1.0 + caustics(p.xz, bed_below) * 0.8;
    let st = sigma_t();
    let trans = exp(-st * (d + max(bed_below, 0.0) * 0.5));
    color = color * trans + inscatter(d, below_eye, water_light(1.0));
    return vec4<f32>(color, 1.0);
}
"#;

const RESOLVE_WGSL: &str = r#"
@group(0) @binding(0) var depth_tex: DEPTH_TYPE;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

struct Out {
    @location(0) raw: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_full(@builtin(position) pos: vec4<f32>) -> Out {
    let d = textureLoad(depth_tex, vec2<i32>(pos.xy), 0);
    var out: Out;
    out.raw = vec4<f32>(d, 0.0, 0.0, 0.0);
    out.depth = d;
    return out;
}
"#;
