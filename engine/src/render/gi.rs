//! Global illumination from a voxel copy of the scene.
//!
//! A box of voxels follows the camera. It is rebuilt from the scene's meshes
//! (rasterised once along each axis) and the terrain height whenever geometry
//! or the box position changes, so a blasted hole or a new wall changes the
//! lighting within a few frames. No baking.
//!
//! A coarser grid of light probes sits in the same box. Every frame each
//! probe traces a handful of rays through the voxels: a ray that escapes sees
//! the sky, a ray that hits a voxel picks up the sun light falling on that
//! face plus the probes' own light there from the last frame. That last part
//! makes light bounce more than once over a few frames. Each probe keeps its
//! result as linear (L1) spherical harmonics, blended over time.
//!
//! Lit shaders read the probes through `gi_irradiance` (see `shadow.rs`) in
//! place of the plain sky dome.

use crate::mesh::{InstanceRaw, Vertex};
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};
use wgpu::util::DeviceExt;

/// Probes along x, y, z. Voxels are twice as fine.
pub const PROBE_DIMS: [u32; 3] = [32, 16, 32];
const VOXEL_DIMS: [u32; 3] = [PROBE_DIMS[0] * 2, PROBE_DIMS[1] * 2, PROBE_DIMS[2] * 2];
/// Rays per probe per frame.
const RAYS: u32 = 16;
/// Voxel steps a ray may take before it counts as escaped.
const MAX_STEPS: u32 = 48;
/// Weight of this frame's rays in the probes; history keeps the rest.
const BLEND: f32 = 0.12;
/// Frames a changed scene waits for new voxels after the last rebuild.
const SCENE_CHANGE_WAIT: u32 = 8;
/// Frames after which the voxels are rebuilt anyway, for props that moved.
const REFRESH_FRAMES: u32 = 30;

/// Whether the voxels must be rebuilt this frame: `built` is the box and
/// scene they were last built for, `frames` ago.
///
/// A moved box needs its voxels at once. A changed scene waits a few
/// frames: a streamed world spawns and drops something somewhere on most
/// frames, nearly always far from the box, and each rebuild draws every
/// mesh near it three times. The probes blend over many frames anyway, so
/// the wait does not show.
fn voxels_are_stale(
    built: Option<(GiPlacement, u64)>,
    placement: GiPlacement,
    scene_epoch: u64,
    frames: u32,
) -> bool {
    built.is_none_or(|(p, epoch)| {
        p != placement
            || (epoch != scene_epoch && frames >= SCENE_CHANGE_WAIT)
            || frames >= REFRESH_FRAMES
    })
}
const RASTER_SIZE: u32 = 128;
const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GiParams {
    origin: [f32; 3],
    voxel: f32,
    vox_dims: [u32; 3],
    frame: u32,
    probe_dims: [u32; 3],
    blend: f32,
    eye: [f32; 3],
    ambient: f32,
    light_color: [f32; 3],
    _p0: f32,
    sky_up: [f32; 3],
    _p1: f32,
    sky_horizon: [f32; 3],
    _p2: f32,
    sky_down: [f32; 3],
    _p3: f32,
    light_dir: [f32; 3],
    _p4: f32,
}

/// Where the probe box sits this frame; also written to the shadow uniforms
/// so every lit shader can find the probes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiPlacement {
    pub origin: Vec3,
    pub spacing: f32,
    pub dims: UVec3,
}

impl GiPlacement {
    /// Box of `PROBE_DIMS` probes around `focus`, snapped to whole probes so
    /// it only moves in steps and the voxels stay put in between.
    pub fn around(focus: Vec3, spacing: f32) -> Self {
        let dims = UVec3::from(PROBE_DIMS);
        let half = dims.as_vec3() * spacing * 0.5;
        // Keep more of the box above the ground the camera stands on.
        let centre = focus + Vec3::new(0.0, half.y * 0.4, 0.0);
        let origin = ((centre - half) / spacing).floor() * spacing;
        Self {
            origin,
            spacing,
            dims,
        }
    }
}

/// Frame inputs for the probe update.
pub struct GiFrame {
    pub placement: GiPlacement,
    pub eye: Vec3,
    pub light_dir: Vec3,
    pub light_color: Vec3,
    pub ambient: f32,
    pub sky: super::SkyLight,
}

pub struct GiVolume {
    params: wgpu::Buffer,
    /// Probe history the lit shaders sample (red, green, blue SH).
    history: [wgpu::Texture; 3],
    history_views: [wgpu::TextureView; 3],
    /// This frame's probes, copied over `history` after the update.
    next: [wgpu::Texture; 3],
    fill_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    voxel_pipeline: wgpu::RenderPipeline,
    terrain_voxel_pipeline: wgpu::RenderPipeline,
    fill_bind: wgpu::BindGroup,
    update_bind: wgpu::BindGroup,
    voxel_binds: [wgpu::BindGroup; 3],
    raster_view: wgpu::TextureView,
    frame: u32,
    /// Placement and scene epoch the voxels were last built for.
    built: Option<(GiPlacement, u64)>,
    frames_since_build: u32,
}

fn texture_3d(
    device: &wgpu::Device,
    label: &str,
    dims: [u32; 3],
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: dims[0],
            height: dims[1],
            depth_or_array_layers: dims[2],
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage,
        view_formats: &[],
    })
}

fn storage_entry(
    binding: u32,
    format: wgpu::TextureFormat,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The scene's shadow and terrain declarations, moved from the scene bind
/// group (0, after its uniform at binding 0) to the shadow resource group 1.
fn shadow_resources_as_group1() -> String {
    let mut wgsl = super::shadow::SCENE_SHADOW_WGSL.to_string();
    for binding in (1..=14).rev() {
        wgsl = wgsl.replace(
            &format!("@group(0) @binding({binding})"),
            &format!("@group(1) @binding({})", binding - 1),
        );
    }
    wgsl
}

impl GiVolume {
    pub fn new(device: &wgpu::Device, shadow_layout: &wgpu::BindGroupLayout) -> Self {
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gi-params"),
            contents: bytemuck::bytes_of(&GiParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let albedo = texture_3d(
            device,
            "gi-voxels",
            VOXEL_DIMS,
            ALBEDO_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let albedo_view = albedo.create_view(&Default::default());
        let history: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            texture_3d(
                device,
                &format!("gi-probes-{i}"),
                PROBE_DIMS,
                PROBE_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            )
        });
        let next: [wgpu::Texture; 3] = std::array::from_fn(|i| {
            texture_3d(
                device,
                &format!("gi-probes-next-{i}"),
                PROBE_DIMS,
                PROBE_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            )
        });
        let history_views = std::array::from_fn(|i| history[i].create_view(&Default::default()));
        let next_views: [wgpu::TextureView; 3] =
            std::array::from_fn(|i| next[i].create_view(&Default::default()));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi-shader"),
            source: wgpu::ShaderSource::Wgsl(gi_source().into()),
        });
        let voxel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi-voxelize-shader"),
            source: wgpu::ShaderSource::Wgsl(format!("{PARAMS_WGSL}{VOXEL_WGSL}").into()),
        });

        let cs = wgpu::ShaderStages::COMPUTE;
        let fill_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi-fill-layout"),
            entries: &[uniform_entry(0, cs), storage_entry(1, ALBEDO_FORMAT, cs)],
        });
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi-update-layout"),
            entries: &[
                uniform_entry(0, cs),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: cs,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(2, PROBE_FORMAT, cs),
                storage_entry(3, PROBE_FORMAT, cs),
                storage_entry(4, PROBE_FORMAT, cs),
            ],
        });
        let fs = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let voxel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi-voxelize-layout"),
            entries: &[
                uniform_entry(0, fs),
                uniform_entry(1, fs),
                storage_entry(2, ALBEDO_FORMAT, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let compute = |label: &str, layouts: &[&wgpu::BindGroupLayout], entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            })
        };
        let fill_pipeline = compute("gi-fill", &[&fill_layout, shadow_layout], "cs_fill");
        let update_pipeline = compute("gi-update", &[&update_layout, shadow_layout], "cs_update");

        let voxel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gi-voxelize"),
                bind_group_layouts: &[&voxel_layout],
                push_constant_ranges: &[],
            });
        let voxel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gi-voxelize"),
            layout: Some(&voxel_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &voxel_shader,
                entry_point: Some("vs_voxel"),
                buffers: &[Vertex::LAYOUT, InstanceRaw::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &voxel_shader,
                entry_point: Some("fs_voxel"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        });
        let terrain_voxel_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gi-voxelize-terrain"),
                layout: Some(&voxel_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &voxel_shader,
                    entry_point: Some("vs_voxel"),
                    buffers: &super::voxel_terrain::VOXEL_BUFFERS,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &voxel_shader,
                    entry_point: Some("fs_voxel"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            });
        let raster_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("gi-voxelize-target"),
                size: wgpu::Extent3d {
                    width: RASTER_SIZE,
                    height: RASTER_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());

        let fill_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-fill"),
            layout: &fill_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
            ],
        });
        let update_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-update"),
            layout: &update_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&next_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&next_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&next_views[2]),
                },
            ],
        });
        let voxel_binds = std::array::from_fn(|axis| {
            let axis_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gi-voxelize-axis"),
                contents: bytemuck::bytes_of(&[axis as u32, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gi-voxelize"),
                layout: &voxel_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: axis_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&albedo_view),
                    },
                ],
            })
        });

        Self {
            params,
            history,
            history_views,
            next,
            fill_pipeline,
            update_pipeline,
            voxel_pipeline,
            terrain_voxel_pipeline,
            fill_bind,
            update_bind,
            voxel_binds,
            raster_view,
            frame: 0,
            built: None,
            frames_since_build: 0,
        }
    }

    /// The probe textures the lit shaders sample.
    pub fn probe_views(&self) -> &[wgpu::TextureView; 3] {
        &self.history_views
    }

    /// Write this frame's inputs. Returns whether the voxels must be rebuilt
    /// (box moved, scene changed, or a periodic refresh for moving props).
    pub fn prepare(&mut self, queue: &wgpu::Queue, frame: &GiFrame, scene_epoch: u64) -> bool {
        let placement = frame.placement;
        let rebuild = voxels_are_stale(self.built, placement, scene_epoch, self.frames_since_build);
        let moved = self.built.is_none_or(|(p, _)| p != placement);
        if rebuild {
            self.built = Some((placement, scene_epoch));
            self.frames_since_build = 0;
        } else {
            self.frames_since_build += 1;
        }
        self.frame = self.frame.wrapping_add(1);
        let params = GiParams {
            origin: placement.origin.into(),
            voxel: placement.spacing * 0.5,
            vox_dims: VOXEL_DIMS,
            frame: self.frame,
            probe_dims: PROBE_DIMS,
            // A moved box has no history at the new probe positions.
            blend: if moved { 1.0 } else { BLEND },
            eye: frame.eye.into(),
            ambient: frame.ambient,
            light_color: frame.light_color.into(),
            _p0: 0.0,
            sky_up: frame.sky.up,
            _p1: 0.0,
            sky_horizon: frame.sky.horizon,
            _p2: 0.0,
            sky_down: frame.sky.down,
            _p3: 0.0,
            light_dir: frame.light_dir.normalize_or_zero().into(),
            _p4: 0.0,
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        rebuild
    }

    /// Clear the voxels and fill everything under the terrain surface.
    pub fn encode_fill(&self, encoder: &mut wgpu::CommandEncoder, shadow_bind: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gi-fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fill_pipeline);
        pass.set_bind_group(0, &self.fill_bind, &[]);
        pass.set_bind_group(1, shadow_bind, &[]);
        pass.dispatch_workgroups(VOXEL_DIMS[0] / 8, VOXEL_DIMS[2] / 8, 1);
    }

    /// A render pass that writes every fragment of the meshes drawn in it
    /// into the voxels, projected along `axis` (0 x, 1 y, 2 z). Draw the
    /// scene's opaque meshes with `Vertex` and `InstanceRaw` buffers.
    pub fn begin_voxelize<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        axis: usize,
    ) -> wgpu::RenderPass<'e> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gi-voxelize"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.raster_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.voxel_pipeline);
        pass.set_bind_group(0, &self.voxel_binds[axis], &[]);
        pass
    }

    /// Switch a voxelize pass over to voxel-terrain geometry.
    pub fn use_terrain_voxelizer(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.terrain_voxel_pipeline);
    }

    /// Back to regular meshes after [`Self::use_terrain_voxelizer`].
    pub fn use_mesh_voxelizer(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.voxel_pipeline);
    }

    /// Trace this frame's probe rays and fold them into the history.
    pub fn encode_update(&self, encoder: &mut wgpu::CommandEncoder, shadow_bind: &wgpu::BindGroup) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi-update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.update_bind, &[]);
            pass.set_bind_group(1, shadow_bind, &[]);
            pass.dispatch_workgroups(PROBE_DIMS[0] / 4, PROBE_DIMS[1] / 4, PROBE_DIMS[2] / 4);
        }
        let size = wgpu::Extent3d {
            width: PROBE_DIMS[0],
            height: PROBE_DIMS[1],
            depth_or_array_layers: PROBE_DIMS[2],
        };
        for (next, history) in self.next.iter().zip(&self.history) {
            encoder.copy_texture_to_texture(next.as_image_copy(), history.as_image_copy(), size);
        }
    }
}

/// The probe update shader with everything it includes.
fn gi_source() -> String {
    format!(
        "{}{}{}{}{}{PARAMS_WGSL}{GI_WGSL}",
        super::pipeline::LIGHTING_WGSL,
        super::shadow::SHADOW_UNIFORMS_WGSL,
        shadow_resources_as_group1(),
        super::shadow::SHADOW_EVAL_WGSL,
        super::lights::LAMP_EVAL_WGSL,
    )
    .replace("RAYS", &RAYS.to_string())
    .replace("MAX_STEPS", &MAX_STEPS.to_string())
}

const PARAMS_WGSL: &str = r#"
struct GiParams {
    origin: vec3<f32>,
    voxel: f32,
    vox_dims: vec3<u32>,
    frame: u32,
    probe_dims: vec3<u32>,
    blend: f32,
    eye: vec3<f32>,
    ambient: f32,
    light_color: vec3<f32>,
    _p0: f32,
    sky_up: vec3<f32>,
    _p1: f32,
    sky_horizon: vec3<f32>,
    _p2: f32,
    sky_down: vec3<f32>,
    _p3: f32,
    light_dir: vec3<f32>,
    _p4: f32,
};
@group(0) @binding(0) var<uniform> gi: GiParams;
"#;

const GI_WGSL: &str = r#"
@group(0) @binding(1) var voxels_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(5) var voxels: texture_3d<f32>;
@group(0) @binding(2) var probe_r_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var probe_g_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var probe_b_out: texture_storage_3d<rgba16float, write>;

const GROUND_ALBEDO: vec3<f32> = vec3<f32>(0.32, 0.30, 0.22);

// One thread per voxel column: empty above the terrain, ground below it.
// Meshes are rasterised into the volume afterwards.
@compute @workgroup_size(8, 8, 1)
fn cs_fill(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= gi.vox_dims.x || id.y >= gi.vox_dims.z {
        return;
    }
    let xz = gi.origin.xz + (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) * gi.voxel;
    let ground = shadow_ground(xz);
    for (var y = 0u; y < gi.vox_dims.y; y++) {
        let centre_y = gi.origin.y + (f32(y) + 0.5) * gi.voxel;
        var v = vec4<f32>(0.0);
        if centre_y < ground {
            v = vec4<f32>(GROUND_ALBEDO, 1.0);
        }
        textureStore(voxels_out, vec3<u32>(id.x, y, id.y), v);
    }
}

fn voxel_at(c: vec3<i32>) -> vec4<f32> {
    return textureLoad(voxels, c, 0);
}

fn inside(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(gi.vox_dims));
}

fn gi_sky(dir: vec3<f32>) -> vec3<f32> {
    var light: SceneLight;
    light.sky_up = gi.sky_up;
    light.sky_horizon = gi.sky_horizon;
    light.sky_down = gi.sky_down;
    return sky_radiance(dir, 0.0, light) * gi.ambient;
}

// Light leaving a voxel face toward the probe: the sun on that face plus last
// frame's probe light there (which carries all earlier bounces).
fn hit_radiance(albedo: vec3<f32>, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let l = gi.light_dir;
    let ndl = max(dot(n, l), 0.0);
    var sun = vec3<f32>(0.0);
    if ndl > 0.0 {
        sun = gi.light_color * (1.0 - gi.ambient) * ndl * sun_visibility(p, n, gi.eye);
    }
    let bounce = gi_irradiance(p, n);
    return albedo * (sun + bounce.rgb * bounce.a);
}

// Points spread evenly over the sphere, turned a little every frame so the
// history averages many more directions than one frame traces.
fn ray_dir(i: u32, frame: u32) -> vec3<f32> {
    let golden = 2.39996323;
    let fi = f32(i) + 0.5;
    let z = 1.0 - 2.0 * fi / f32(RAYS);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let spin = fract(f32(frame) * 0.618034) * 6.2831853;
    let phi = fi * golden + spin;
    let tilt = fract(f32(frame) * 0.754878) * 2.0 - 1.0;
    let d = vec3<f32>(cos(phi) * r, z, sin(phi) * r);
    // Rotate about x by a per-frame angle so the poles move too.
    let a = tilt * 1.5;
    return vec3<f32>(d.x, d.y * cos(a) - d.z * sin(a), d.y * sin(a) + d.z * cos(a));
}

@compute @workgroup_size(4, 4, 4)
fn cs_update(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= gi.probe_dims) {
        return;
    }
    let probe_p = gi.origin + (vec3<f32>(id) + vec3<f32>(0.5)) * gi.voxel * 2.0;
    let start = (probe_p - gi.origin) / gi.voxel;
    let start_cell = vec3<i32>(floor(start));
    // A probe buried in geometry sees nothing useful; mark it so shading
    // skips it rather than leaking darkness.
    if voxel_at(start_cell).a > 0.5 {
        let dead = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        textureStore(probe_r_out, id, dead);
        textureStore(probe_g_out, id, dead);
        textureStore(probe_b_out, id, dead);
        return;
    }

    var mean = vec3<f32>(0.0);
    var lin_r = vec3<f32>(0.0);
    var lin_g = vec3<f32>(0.0);
    var lin_b = vec3<f32>(0.0);
    for (var i = 0u; i < RAYS; i++) {
        let dir = ray_dir(i, gi.frame);
        // Grid traversal (Amanatides-Woo) in voxel units.
        var cell = start_cell;
        let step = vec3<i32>(sign(dir));
        let inv = 1.0 / max(abs(dir), vec3<f32>(1e-6));
        let frac = start - floor(start);
        var t_max = select(frac, vec3<f32>(1.0) - frac, dir > vec3<f32>(0.0)) * inv;
        var radiance = gi_sky(dir);
        for (var s = 0u; s < MAX_STEPS; s++) {
            // The axis to step along, as a 0/1 mask: DirectX's FXC compiler
            // cannot write vector components picked by a runtime index.
            var mask = vec3<f32>(1.0, 0.0, 0.0);
            if t_max.y < t_max.x && t_max.y <= t_max.z {
                mask = vec3<f32>(0.0, 1.0, 0.0);
            } else if t_max.z < t_max.x && t_max.z < t_max.y {
                mask = vec3<f32>(0.0, 0.0, 1.0);
            }
            let t = dot(t_max, mask);
            cell += step * vec3<i32>(mask);
            t_max += inv * mask;
            if !inside(cell) {
                break;
            }
            let v = voxel_at(cell);
            if v.a > 0.5 {
                let n = select(vec3<f32>(0.0), -vec3<f32>(step), mask > vec3<f32>(0.5));
                let hit = gi.origin + (start + dir * t) * gi.voxel;
                radiance = hit_radiance(v.rgb, hit + n * 0.05, n);
                break;
            }
        }
        mean += radiance;
        lin_r += radiance.r * dir;
        lin_g += radiance.g * dir;
        lin_b += radiance.b * dir;
    }
    let k = 1.0 / f32(RAYS);
    // L(w) ~ mean + lin . w; see gi_irradiance for the cosine convolution.
    let fresh_r = vec4<f32>(mean.r * k, lin_r * 3.0 * k);
    let fresh_g = vec4<f32>(mean.g * k, lin_g * 3.0 * k);
    let fresh_b = vec4<f32>(mean.b * k, lin_b * 3.0 * k);
    let c = vec3<i32>(id);
    var old_r = textureLoad(gi_probe_r, c, 0);
    var old_g = textureLoad(gi_probe_g, c, 0);
    var old_b = textureLoad(gi_probe_b, c, 0);
    // A NaN or infinite value must never stay in the history: it would
    // spread to every probe and every lit pixel. A broken fresh value is
    // dropped this frame; a broken old one is replaced. `mix` with a blend
    // of 1 still multiplies the old value by 0, which keeps a NaN, so the
    // fresh value is picked outright then.
    if !(gi_finite4(fresh_r) && gi_finite4(fresh_g) && gi_finite4(fresh_b)) {
        return;
    }
    var blend = gi.blend;
    if old_r.x < 0.0 || !(gi_finite4(old_r) && gi_finite4(old_g) && gi_finite4(old_b)) {
        blend = 1.0;
    }
    let fresh_only = blend >= 1.0;
    textureStore(probe_r_out, id, select(mix(old_r, fresh_r, blend), fresh_r, fresh_only));
    textureStore(probe_g_out, id, select(mix(old_g, fresh_g, blend), fresh_g, fresh_only));
    textureStore(probe_b_out, id, select(mix(old_b, fresh_b, blend), fresh_b, fresh_only));
}

// Tested on the bits: a compiler may assume floats are finite.
fn gi_finite4(v: vec4<f32>) -> bool {
    let e = bitcast<vec4<u32>>(v) & vec4<u32>(0x7f800000u);
    return all(e != vec4<u32>(0x7f800000u));
}
"#;

const VOXEL_WGSL: &str = r#"
struct Axis {
    axis: u32,
};
@group(0) @binding(1) var<uniform> proj: Axis;
@group(0) @binding(2) var voxels_out: texture_storage_3d<rgba8unorm, write>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
    @location(10) tint: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_p: vec3<f32>,
    @location(1) color: vec3<f32>,
};

// Orthographic view of the voxel box along one axis; each axis gets its own
// pass so surfaces facing any way leave fragments in the volume.
@vertex
fn vs_voxel(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let world = (model * vec4<f32>(v.position, 1.0)).xyz;
    let local = (world - gi.origin) / (vec3<f32>(gi.vox_dims) * gi.voxel);
    var flat = vec3<f32>(local.x, local.y, local.z);
    if proj.axis == 0u {
        flat = vec3<f32>(local.z, local.y, local.x);
    } else if proj.axis == 1u {
        flat = vec3<f32>(local.x, local.z, local.y);
    }
    var out: VsOut;
    out.clip = vec4<f32>(flat.xy * 2.0 - vec2<f32>(1.0), flat.z, 1.0);
    out.world_p = world;
    out.color = (v.color * v.tint).rgb;
    return out;
}

fn gi_voxel_finite(v: vec3<f32>) -> bool {
    let e = bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u);
    return all(e != vec3<u32>(0x7f800000u));
}

@fragment
fn fs_voxel(in: VsOut) -> @location(0) vec4<f32> {
    let c = vec3<i32>(floor((in.world_p - gi.origin) / gi.voxel));
    if all(c >= vec3<i32>(0)) && all(c < vec3<i32>(gi.vox_dims)) {
        // A broken (NaN or infinite) colour would reach every probe.
        var colour = in.color;
        if !gi_voxel_finite(colour) {
            colour = vec3<f32>(0.0);
        }
        textureStore(voxels_out, c, vec4<f32>(colour, 1.0));
    }
    return vec4<f32>(0.0);
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn the_probe_shader_is_valid_wgsl() {
        crate::render::assert_valid_wgsl(&super::gi_source());
        crate::render::assert_valid_wgsl(&format!("{}{}", super::PARAMS_WGSL, super::VOXEL_WGSL));
    }

    use super::*;

    #[test]
    fn placement_snaps_to_whole_probes_and_contains_focus() {
        let spacing = 2.0;
        let focus = Vec3::new(13.3, 4.7, -21.9);
        let p = GiPlacement::around(focus, spacing);
        let cells = p.origin / spacing;
        assert!((cells - cells.round()).abs().max_element() < 1e-4);
        let far = p.origin + p.dims.as_vec3() * spacing;
        assert!(focus.cmpge(p.origin).all() && focus.cmplt(far).all());
        // Small moves inside one probe cell keep the box (and its voxels).
        let nudged = GiPlacement::around(focus + Vec3::splat(0.01), spacing);
        assert_eq!(p, nudged);
    }

    #[test]
    fn a_changed_scene_waits_for_voxels_and_a_moved_box_does_not() {
        let here = GiPlacement::around(Vec3::ZERO, 2.0);
        let there = GiPlacement::around(Vec3::splat(40.0), 2.0);
        assert!(voxels_are_stale(None, here, 1, 0));
        assert!(!voxels_are_stale(Some((here, 1)), here, 1, 5));
        assert!(voxels_are_stale(Some((here, 1)), there, 1, 0));
        // Something spawned: not on the very next frames, but soon.
        assert!(!voxels_are_stale(
            Some((here, 1)),
            here,
            2,
            SCENE_CHANGE_WAIT - 1
        ));
        assert!(voxels_are_stale(
            Some((here, 1)),
            here,
            2,
            SCENE_CHANGE_WAIT
        ));
        // Nothing changed: still refreshed for things that moved.
        assert!(voxels_are_stale(Some((here, 1)), here, 1, REFRESH_FRAMES));
    }

    #[test]
    fn shadow_bindings_move_to_group_one() {
        let wgsl = shadow_resources_as_group1();
        assert!(!wgsl.contains("@group(0)"));
        assert!(wgsl.contains("@group(1) @binding(0) var<uniform> shadow"));
        assert!(wgsl.contains("@group(1) @binding(9) var gi_probe_b"));
    }
}
