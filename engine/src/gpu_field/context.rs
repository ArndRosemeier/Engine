//! Headless wgpu context and compute pipelines for scalar fields.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::UVec3;

use wgpu::util::DeviceExt;

use crate::error::{EngineError, EngineResult};
use crate::marching_cubes::{EDGE_TABLE, TRI_TABLE};
use crate::mesh::BuiltMesh;
use crate::color::Color;

use super::custom::CustomFieldKernel;
use super::grid::FieldGrid;
use super::kernel::FieldKernel;

/// Ceiling for a custom field kernel's uniform block (WGSL + CPU mirror).
/// 384 bytes = 24 vec4 slots: 16 for geometry, 8 for composition knobs.
pub const MAX_FIELD_UNIFORM_BYTES: u32 = 384;

pub const PAINT_WORKGROUP: u32 = 8;
pub const EXTRACT_WORKGROUP: u32 = 4;
pub const MAX_TRIS_PER_CELL: u32 = 5;
const VERT_FLOATS: u32 = 6; // position xyz + normal xyz

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PaintParams {
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    corner_dims: [u32; 4],
    sphere_center: [f32; 4],
    sphere_radius: f32,
    voxel_size: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ExtractParams {
    bounds_min: [f32; 4],
    voxel_size: f32,
    _pad_after_voxel: [f32; 3],
    _pad_to_corner: [f32; 4],
    corner_dims: [u32; 4],
    /// x = cell stride for LOD extracts (1 = full detail); y..w unused.
    lod_stride: [u32; 4],
}

const _EXTRACT_PARAMS_SIZE: usize = std::mem::size_of::<ExtractParams>();
const _ASSERT_EXTRACT_PARAMS_80: () = assert!(_EXTRACT_PARAMS_SIZE == 80);

const PAINT_WGSL: &str = r#"
struct PaintParams {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    corner_dims: vec4<u32>,
    sphere_center: vec4<f32>,
    sphere_radius: f32,
    voxel_size: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: PaintParams;
@group(0) @binding(1) var<storage, read_write> density: array<f32>;

fn demo_density(p: vec3<f32>) -> f32 {
    let bmin = params.bounds_min.xyz;
    let bmax = params.bounds_max.xyz;
    if (any(p < bmin) || any(p > bmax)) {
        return -1.0;
    }
    var d = 1.0;
    let sd = length(p - params.sphere_center.xyz) - params.sphere_radius;
    d = min(d, sd);
    return d;
}

@compute @workgroup_size(8, 4, 4)
fn paint(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = params.corner_dims.xyz;
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }
    let p = params.bounds_min.xyz + vec3<f32>(gid) * params.voxel_size;
    let idx = gid.x + dims.x * (gid.y + gid.z * dims.y);
    density[idx] = demo_density(p);
}
"#;

const EXTRACT_WGSL: &str = r#"
struct ExtractParams {
    bounds_min: vec4<f32>,
    voxel_size: f32,
    _pad0: vec3<f32>,
    corner_dims: vec4<u32>,
    // x = LOD cell stride; y..w unused. Kept as one vec4 so the struct
    // stays exactly 80 bytes like the Rust mirror.
    lod_stride: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: ExtractParams;
@group(0) @binding(1) var<storage, read> density: array<f32>;
@group(0) @binding(2) var<storage, read> edge_table: array<u32, 256>;
@group(0) @binding(3) var<storage, read> tri_table: array<i32, 4096>;
@group(0) @binding(4) var<storage, read_write> tri_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> vert_data: array<f32>;

const MAX_TRIS: u32 = 5u;
const VERT_FLOATS: u32 = 6u;

fn corner_density(ix: u32, iy: u32, iz: u32) -> f32 {
    let dims = params.corner_dims.xyz;
    let idx = ix + dims.x * (iy + iz * dims.y);
    return density[idx];
}

fn interp(iso: f32, p1: vec3<f32>, p2: vec3<f32>, v1: f32, v2: f32) -> vec3<f32> {
    if (abs(iso - v1) < 0.00001) { return p1; }
    if (abs(iso - v2) < 0.00001) { return p2; }
    if (abs(v1 - v2) < 0.00001) { return p1; }
    let t = (iso - v1) / (v2 - v1);
    return mix(p1, p2, t);
}

fn write_vert(cell_base: u32, tri: u32, vert: u32, pos: vec3<f32>, nrm: vec3<f32>) {
    let base = cell_base + (tri * 3u + vert) * VERT_FLOATS;
    vert_data[base + 0u] = pos.x;
    vert_data[base + 1u] = pos.y;
    vert_data[base + 2u] = pos.z;
    vert_data[base + 3u] = nrm.x;
    vert_data[base + 4u] = nrm.y;
    vert_data[base + 5u] = nrm.z;
}

@compute @workgroup_size(4, 4, 4)
fn extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let stride = max(params.lod_stride.x, 1u);
    // Strided cells: sample every `stride`-th cell so corner indices stay
    // aligned across LOD levels (cell at base + gid * stride).
    // Cell i spans corners [i*s, (i+1)*s]; require the top corner to exist:
    // count = (corner_dims - 1) / stride.
    let cells = max((params.corner_dims.xyz - vec3<u32>(1u)) / stride, vec3<u32>(1u));
    if (gid.x >= cells.x || gid.y >= cells.y || gid.z >= cells.z) {
        return;
    }
    let cell0 = gid * stride;

    let v0 = corner_density(cell0.x, cell0.y, cell0.z);
    let v1 = corner_density(cell0.x + stride, cell0.y, cell0.z);
    let v2 = corner_density(cell0.x + stride, cell0.y, cell0.z + stride);
    let v3 = corner_density(cell0.x, cell0.y, cell0.z + stride);
    let v4 = corner_density(cell0.x, cell0.y + stride, cell0.z);
    let v5 = corner_density(cell0.x + stride, cell0.y + stride, cell0.z);
    let v6 = corner_density(cell0.x + stride, cell0.y + stride, cell0.z + stride);
    let v7 = corner_density(cell0.x, cell0.y + stride, cell0.z + stride);
    let vals = array<f32, 8>(v0, v1, v2, v3, v4, v5, v6, v7);

    var cube_index = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (vals[i] < 0.0) {
            cube_index = cube_index | (1u << i);
        }
    }
    if (cube_index == 0u || cube_index == 255u) {
        tri_counts[gid.x + cells.x * (gid.y + gid.z * cells.y)] = 0u;
        return;
    }

    let edges = edge_table[cube_index];
    let h = params.voxel_size * f32(stride);
    let origin = params.bounds_min.xyz + vec3<f32>(cell0) * params.voxel_size;

    var vert_list: array<vec3<f32>, 12>;
    if ((edges & 1u) != 0u) { vert_list[0] = interp(0.0, origin, origin + vec3(h,0,0), v0, v1); }
    if ((edges & 2u) != 0u) { vert_list[1] = interp(0.0, origin + vec3(h,0,0), origin + vec3(h,0,h), v1, v2); }
    if ((edges & 4u) != 0u) { vert_list[2] = interp(0.0, origin + vec3(h,0,h), origin + vec3(0,0,h), v2, v3); }
    if ((edges & 8u) != 0u) { vert_list[3] = interp(0.0, origin + vec3(0,0,h), origin, v3, v0); }
    if ((edges & 16u) != 0u) { vert_list[4] = interp(0.0, origin + vec3(0,h,0), origin + vec3(h,h,0), v4, v5); }
    if ((edges & 32u) != 0u) { vert_list[5] = interp(0.0, origin + vec3(h,h,0), origin + vec3(h,h,h), v5, v6); }
    if ((edges & 64u) != 0u) { vert_list[6] = interp(0.0, origin + vec3(h,h,h), origin + vec3(0,h,h), v6, v7); }
    if ((edges & 128u) != 0u) { vert_list[7] = interp(0.0, origin + vec3(0,h,h), origin + vec3(0,h,0), v7, v4); }
    if ((edges & 256u) != 0u) { vert_list[8] = interp(0.0, origin, origin + vec3(0,h,0), v0, v4); }
    if ((edges & 512u) != 0u) { vert_list[9] = interp(0.0, origin + vec3(h,0,0), origin + vec3(h,h,0), v1, v5); }
    if ((edges & 1024u) != 0u) { vert_list[10] = interp(0.0, origin + vec3(h,0,h), origin + vec3(h,h,h), v2, v6); }
    if ((edges & 2048u) != 0u) { vert_list[11] = interp(0.0, origin + vec3(0,0,h), origin + vec3(0,h,h), v3, v7); }

    let cell_index = gid.x + cells.x * (gid.y + gid.z * cells.y);
    let cell_base = cell_index * MAX_TRIS * 3u * VERT_FLOATS;
    var tri_count = 0u;
    let row_base = cube_index * 16u;
    var i = 0u;
    loop {
        if (i >= 16u || tri_table[row_base + i] < 0) { break; }
        let e0 = u32(tri_table[row_base + i]);
        let e1 = u32(tri_table[row_base + i + 1u]);
        let e2 = u32(tri_table[row_base + i + 2u]);
        let a = vert_list[e0];
        let b = vert_list[e1];
        let c = vert_list[e2];
        var n = cross(b - a, c - a);
        if (length(n) > 0.0001) {
            n = normalize(n);
        } else {
            n = vec3(0.0, 1.0, 0.0);
        }
        write_vert(cell_base, tri_count, 0u, a, n);
        write_vert(cell_base, tri_count, 1u, c, n);
        write_vert(cell_base, tri_count, 2u, b, n);
        tri_count = tri_count + 1u;
        i = i + 3u;
    }
    tri_counts[cell_index] = tri_count;
}
"#;

/// Headless GPU device for field paint and isosurface extraction.
pub struct FieldGpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    paint_pipeline: wgpu::ComputePipeline,
    paint_layout: wgpu::BindGroupLayout,
    custom_pipelines: RefCell<HashMap<&'static str, wgpu::ComputePipeline>>,
    extract_pipeline: wgpu::ComputePipeline,
    extract_layout: wgpu::BindGroupLayout,
    edge_table_buf: wgpu::Buffer,
    tri_table_buf: wgpu::Buffer,
    /// Last uncaptured GPU validation error (cleared before risky dispatches).
    last_error: Arc<Mutex<Option<String>>>,
}

impl FieldGpuContext {
    pub fn try_new() -> EngineResult<Self> {
        pollster::block_on(Self::try_new_async())
    }

    pub async fn try_new_async() -> EngineResult<Self> {
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| EngineError::InvalidValue(format!("gpu_field adapter: {e}")))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpu-field-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    // Large-chamber marching cubes needs scratch buffers well above
                    // the conservative defaults (256 MiB buffer / 128 MiB binding).
                    max_buffer_size: 1 << 30,
                    max_storage_buffer_binding_size: 1 << 30,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| EngineError::InvalidValue(format!("gpu_field device: {e}")))?;
        let handler = last_error.clone();
        device.on_uncaptured_error(Box::new(move |err| {
            *handler.lock().expect("gpu_field error handler") = Some(err.to_string());
        }));

        let paint_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field-paint"),
            source: wgpu::ShaderSource::Wgsl(PAINT_WGSL.into()),
        });
        let paint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field-paint-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(MAX_FIELD_UNIFORM_BYTES as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let paint_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("field-paint-pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("field-paint-pipeline-layout"),
                bind_group_layouts: &[&paint_layout],
                push_constant_ranges: &[],
            })),
            module: &paint_shader,
            entry_point: Some("paint"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let extract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field-extract"),
            source: wgpu::ShaderSource::Wgsl(EXTRACT_WGSL.into()),
        });
        *last_error.lock().expect("gpu_field error handler") = None;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let extract_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field-extract-layout"),
            entries: &[
                uniform_entry(
                    0,
                    wgpu::BufferSize::new(std::mem::size_of::<ExtractParams>() as u64)
                        .expect("ExtractParams uniform size > 0"),
                ),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("field-extract-pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("field-extract-pipeline-layout"),
                bind_group_layouts: &[&extract_layout],
                push_constant_ranges: &[],
            })),
            module: &extract_shader,
            entry_point: Some("extract"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scope_err: Option<wgpu::Error> = pollster::block_on(device.pop_error_scope());
        if let Some(err) = scope_err {
            return Err(EngineError::InvalidValue(format!(
                "gpu_field extract pipeline: {err}"
            )));
        }
        if let Some(err) = last_error.lock().expect("gpu_field error handler").take() {
            return Err(EngineError::InvalidValue(format!(
                "gpu_field extract pipeline: {err}"
            )));
        }

        let edge_u32: Vec<u32> = EDGE_TABLE.iter().map(|&e| e as u32).collect();
        let edge_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc-edge-table"),
            contents: bytemuck::cast_slice(&edge_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tri_flat: Vec<i32> = TRI_TABLE
            .iter()
            .flat_map(|row| row.iter().map(|&v| i32::from(v)))
            .collect();
        let tri_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc-tri-table"),
            contents: bytemuck::cast_slice(&tri_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Ok(Self {
            device,
            queue,
            paint_pipeline,
            paint_layout,
            custom_pipelines: RefCell::new(HashMap::new()),
            extract_pipeline,
            extract_layout,
            edge_table_buf,
            tri_table_buf,
            last_error,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn paint_density(
        &self,
        grid: &FieldGrid,
        kernel: &FieldKernel,
    ) -> EngineResult<Vec<f32>> {
        match kernel {
            FieldKernel::DemoSphereVoid {
                sphere_center,
                sphere_radius,
            } => self.paint_demo_sphere(grid, *sphere_center, *sphere_radius),
            FieldKernel::Custom(custom) => self.paint_custom(grid, custom),
        }
    }

    fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        assert!(
            data.len() <= MAX_FIELD_UNIFORM_BYTES as usize,
            "uniform data {} bytes exceeds MAX_FIELD_UNIFORM_BYTES",
            data.len()
        );
        let mut padded = vec![0u8; MAX_FIELD_UNIFORM_BYTES as usize];
        padded[..data.len()].copy_from_slice(data);
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &padded,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn paint_demo_sphere(
        &self,
        grid: &FieldGrid,
        sphere_center: glam::Vec3,
        sphere_radius: f32,
    ) -> EngineResult<Vec<f32>> {
        let corner_count = grid.corner_count();
        let params = PaintParams {
            bounds_min: [
                grid.bounds.min.x,
                grid.bounds.min.y,
                grid.bounds.min.z,
                0.0,
            ],
            bounds_max: [
                grid.bounds.max.x,
                grid.bounds.max.y,
                grid.bounds.max.z,
                0.0,
            ],
            corner_dims: [
                grid.corners[0],
                grid.corners[1],
                grid.corners[2],
                0,
            ],
            sphere_center: [
                sphere_center.x,
                sphere_center.y,
                sphere_center.z,
                0.0,
            ],
            sphere_radius,
            voxel_size: grid.bounds.voxel_size,
            _pad: [0.0; 2],
        };
        let params_buf =
            self.create_uniform_buffer("field-paint-params", bytemuck::bytes_of(&params));
        self.dispatch_paint(grid, &self.paint_pipeline, &params_buf, corner_count)
    }

    fn paint_custom(
        &self,
        grid: &FieldGrid,
        custom: &CustomFieldKernel,
    ) -> EngineResult<Vec<f32>> {
        let pipeline = self.custom_pipeline(custom)?;
        let corner_count = grid.corner_count();
        let mut uniform = vec![0u8; custom.uniform_size as usize];
        let bytes = custom.params.as_bytes();
        uniform[..bytes.len()].copy_from_slice(bytes);
        let params_buf = self.create_uniform_buffer("field-custom-paint-params", &uniform);
        self.dispatch_paint(grid, &pipeline, &params_buf, corner_count)
    }

    fn custom_pipeline(
        &self,
        custom: &CustomFieldKernel,
    ) -> EngineResult<wgpu::ComputePipeline> {
        let mut cache = self.custom_pipelines.borrow_mut();
        if let Some(pipeline) = cache.get(custom.shader_key) {
            return Ok(pipeline.clone());
        }
        *self
            .last_error
            .lock()
            .expect("gpu_field error handler") = None;
        self.device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(custom.shader_key),
            source: wgpu::ShaderSource::Wgsl(custom.wgsl.into()),
        });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field-custom-paint-layout"),
            bind_group_layouts: &[&self.paint_layout],
            push_constant_ranges: &[],
        });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(custom.shader_key),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("paint"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let scope_err: Option<wgpu::Error> =
            pollster::block_on(self.device.pop_error_scope());
        if let Some(err) = scope_err {
            return Err(EngineError::InvalidValue(format!(
                "gpu_field custom pipeline '{}' failed to compile: {err}",
                custom.shader_key
            )));
        }
        if let Some(err) = self.last_error.lock().expect("gpu_field error handler").take() {
            return Err(EngineError::InvalidValue(format!(
                "gpu_field custom pipeline '{}': {err}",
                custom.shader_key
            )));
        }
        cache.insert(custom.shader_key, pipeline.clone());
        Ok(pipeline)
    }

    fn dispatch_paint(
        &self,
        grid: &FieldGrid,
        pipeline: &wgpu::ComputePipeline,
        params_buf: &wgpu::Buffer,
        corner_count: usize,
    ) -> EngineResult<Vec<f32>> {
        *self
            .last_error
            .lock()
            .expect("gpu_field error handler") = None;
        let density_bytes = (corner_count * std::mem::size_of::<f32>()) as u64;
        let density_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-density"),
            size: density_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let paint_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field-paint-bind"),
            layout: &self.paint_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density_buf.as_entire_binding(),
                },
            ],
        });

        let corners = UVec3::new(grid.corners[0], grid.corners[1], grid.corners[2]);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field-paint-encoder"),
            });
        self.device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field-paint"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &paint_bind, &[]);
            pass.dispatch_workgroups(
                corners.x.div_ceil(PAINT_WORKGROUP),
                corners.y.div_ceil(4),
                corners.z.div_ceil(4),
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let scope_err: Option<wgpu::Error> = pollster::block_on(self.device.pop_error_scope());
        if let Some(err) = scope_err {
            return Err(EngineError::InvalidValue(format!(
                "gpu_field paint dispatch: {err}"
            )));
        }
        let density = read_buffer_f32(&self.device, &self.queue, &density_buf, corner_count)?;
        if let Some(err) = self.last_error.lock().expect("gpu_field error handler").take() {
            return Err(EngineError::InvalidValue(format!(
                "gpu_field paint dispatch: {err}"
            )));
        }
        Ok(density)
    }

    pub fn extract_mesh(&self, grid: &FieldGrid, density: &[f32], color: Color) -> EngineResult<BuiltMesh> {
        self.extract_mesh_lod(grid, density, color, 1)
    }

    /// Extract a decimated mesh by sampling every `lod_stride`-th cell.
    ///
    /// Stride 1 equals [`Self::extract_mesh`]. Corner indices stay aligned
    /// across levels (cell corners at multiples of the stride), so LOD meshes
    /// occupy identical world-space shells — only coarser.
    pub fn extract_mesh_lod(
        &self,
        grid: &FieldGrid,
        density: &[f32],
        color: Color,
        lod_stride: u32,
    ) -> EngineResult<BuiltMesh> {
        let stride = lod_stride.max(1);
        let corner_count = grid.corner_count();
        if density.len() != corner_count {
            return Err(EngineError::InvalidValue(format!(
                "density len {} != corner count {corner_count}",
                density.len()
            )));
        }
        // Strided cell grid: cell i spans corners [i*s, (i+1)*s], so the count
        // is (corners − 1) / stride (top partial cell is dropped, keeping
        // LOD shells corner-aligned with the full-detail extract).
        let lod_cells = [
            (grid.corners[0] - 1) / stride,
            (grid.corners[1] - 1) / stride,
            (grid.corners[2] - 1) / stride,
        ];
        let lod_cells = [
            lod_cells[0].max(1),
            lod_cells[1].max(1),
            lod_cells[2].max(1),
        ];
        let cell_count =
            lod_cells[0] as usize * lod_cells[1] as usize * lod_cells[2] as usize;

        let density_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("field-density-read"),
            contents: bytemuck::cast_slice(density),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let tri_counts_bytes = (cell_count * std::mem::size_of::<u32>()) as u64;
        let tri_counts_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-tri-counts"),
            size: tri_counts_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let floats_per_cell = MAX_TRIS_PER_CELL * 3 * VERT_FLOATS;
        let vert_bytes = (cell_count as u64) * (floats_per_cell as u64) * 4;
        {
            let limits = self.device.limits();
            assert!(
                vert_bytes <= limits.max_buffer_size,
                "gpu_field extract: vertex scratch buffer needs {vert_bytes} bytes for {cell_count} cells, \
                 exceeding device max_buffer_size ({}) — reduce field size or raise device limits",
                limits.max_buffer_size
            );
        }
        let vert_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field-vert-data"),
            size: vert_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = ExtractParams {
            bounds_min: [
                grid.bounds.min.x,
                grid.bounds.min.y,
                grid.bounds.min.z,
                0.0,
            ],
            voxel_size: grid.bounds.voxel_size,
            _pad_after_voxel: [0.0; 3],
            _pad_to_corner: [0.0; 4],
            corner_dims: [
                grid.corners[0],
                grid.corners[1],
                grid.corners[2],
                0,
            ],
            lod_stride: [stride, 0, 0, 0],
        };
        let params_buf =
            self.create_uniform_buffer("field-extract-params", bytemuck::bytes_of(&params));

        let extract_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field-extract-bind"),
            layout: &self.extract_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: density_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.edge_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tri_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tri_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vert_buf.as_entire_binding(),
                },
            ],
        });

        let cells = UVec3::new(lod_cells[0], lod_cells[1], lod_cells[2]);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field-extract-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field-extract"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, &extract_bind, &[]);
            pass.dispatch_workgroups(
                cells.x.div_ceil(EXTRACT_WORKGROUP),
                cells.y.div_ceil(EXTRACT_WORKGROUP),
                cells.z.div_ceil(EXTRACT_WORKGROUP),
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let tri_counts = read_buffer_u32(&self.device, &self.queue, &tri_counts_buf, cell_count)?;
        let vert_floats = read_buffer_f32(
            &self.device,
            &self.queue,
            &vert_buf,
            cell_count * floats_per_cell as usize,
        )?;

        pack_extracted_mesh_lod(grid, &lod_cells, stride, &tri_counts, &vert_floats, color)
    }
}

fn uniform_entry(binding: u32, min_binding_size: wgpu::BufferSize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(min_binding_size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn read_buffer_bytes(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, size: u64) -> EngineResult<Vec<u8>> {
    let limits = device.limits();
    assert!(
        size <= limits.max_buffer_size,
        "field readback of {size} bytes exceeds device max_buffer_size ({})",
        limits.max_buffer_size
    );
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("field-readback"),
        size: size.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("field-readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size.max(4));
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("field readback channel");
    });
    device
        .poll(wgpu::PollType::Wait)
        .expect("gpu_field device poll failed");
    rx.recv()
        .expect("field readback result")
        .map_err(|e| EngineError::InvalidValue(format!("field buffer map: {e:?}")))?;

    let data = slice.get_mapped_range();
    Ok(data.to_vec())
}

fn read_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> EngineResult<Vec<f32>> {
    let bytes = read_buffer_bytes(device, queue, src, (count * 4) as u64)?;
    Ok(bytemuck::cast_slice(&bytes[..count * 4]).to_vec())
}

fn read_buffer_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> EngineResult<Vec<u32>> {
    let bytes = read_buffer_bytes(device, queue, src, (count * 4) as u64)?;
    Ok(bytemuck::cast_slice(&bytes[..count * 4]).to_vec())
}

/// Pack GPU cell outputs into a mesh. `lod_cells`/`stride` document the strided
/// dispatch the outputs came from; vertices are already world-space.
fn pack_extracted_mesh_lod(
    _grid: &FieldGrid,
    _lod_cells: &[u32; 3],
    _stride: u32,
    tri_counts: &[u32],
    vert_floats: &[f32],
    color: Color,
) -> EngineResult<BuiltMesh> {
    let floats_per_cell = (MAX_TRIS_PER_CELL * 3 * VERT_FLOATS) as usize;
    let rgba = color.to_vec4();
    let mut mesh = BuiltMesh::default();

    for (cell_index, &tri_count) in tri_counts.iter().enumerate() {
        if tri_count == 0 {
            continue;
        }

        let base = cell_index * floats_per_cell;
        for tri in 0..tri_count as usize {
            for vert in 0..3 {
                let f = base + (tri * 3 + vert) * VERT_FLOATS as usize;
                if f + 5 >= vert_floats.len() {
                    return Err(EngineError::InvalidMesh(
                        "gpu_field extract vertex buffer overrun".into(),
                    ));
                }
                let pos = glam::Vec3::new(vert_floats[f], vert_floats[f + 1], vert_floats[f + 2]);
                let nrm = glam::Vec3::new(vert_floats[f + 3], vert_floats[f + 4], vert_floats[f + 5]);
                let idx = mesh.positions.len() as u32;
                mesh.positions.push(pos);
                mesh.normals.push(nrm);
                mesh.colors.push(rgba);
                mesh.uvs.push([0.0, 0.0]);
                mesh.indices.push(idx);
            }
        }
    }

    if mesh.triangle_count() == 0 {
        return Err(EngineError::InvalidMesh(
            "gpu_field extract produced zero triangles".into(),
        ));
    }

    mesh.opaque_index_count = mesh.indices.len();
    Ok(mesh)
}
