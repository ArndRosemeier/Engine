//! Terrain sun shadows from a precomputed horizon-height map.
//!
//! For every texel of a square region around the camera, a compute pass marches
//! toward the sun once and stores the lowest height at which that column sees
//! the sun:
//!
//! `H(xz) = max over t of ( ground(xz + sun_xz * t) - rise * t )`
//!
//! A surface point is lit when `y >= H(xz)`. Shading then costs four texel
//! loads instead of a raymarch per pixel, works for anything standing on the
//! terrain (props, characters), and the march can reach hundreds of metres, so
//! hills cast long evening shadows. The map is rebuilt only when the sun turns,
//! the camera leaves the middle of the region, or the terrain changes.

use super::shadow::{shadow_uniforms_wgsl, terrain_formula_wgsl};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Where the ground heights come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroundSource {
    /// The analytic terrain formula in the shadow uniforms.
    Formula,
    /// The CPU-filled contact height atlas covering the same region.
    Atlas,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct HorizonParams {
    origin_xz: [f32; 2],
    extent: f32,
    size: u32,
    sun_xz: [f32; 2],
    rise: f32,
    max_ground: f32,
    source: u32,
    _pad: [u32; 3],
}

const WORKGROUP: u32 = 8;

const HORIZON_WGSL_BODY: &str = r#"
struct HorizonParams {
    origin_xz: vec2<f32>,
    extent: f32,
    size: u32,
    sun_xz: vec2<f32>,
    rise: f32,
    max_ground: f32,
    source: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(1) var<uniform> hp: HorizonParams;
@group(0) @binding(2) var ground_tex: texture_2d<f32>;
@group(0) @binding(3) var horizon_out: texture_storage_2d<r32float, write>;

fn ground_at(xz: vec2<f32>) -> f32 {
    if hp.source == 0u {
        return formula_ground(xz);
    }
    let uv = (xz - hp.origin_xz) / hp.extent;
    if uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0 {
        return -10000.0;
    }
    let dims = vec2<f32>(textureDimensions(ground_tex));
    return textureLoad(ground_tex, vec2<i32>(uv * dims), 0).r;
}

@compute @workgroup_size(8, 8)
fn cs_horizon(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= hp.size || id.y >= hp.size {
        return;
    }
    let texel = hp.extent / f32(hp.size);
    let xz = hp.origin_xz + (vec2<f32>(id.xy) + vec2<f32>(0.5)) * texel;
    let g0 = ground_at(xz);
    // Anything at or above the ground counts as lit until an occluder says otherwise.
    var h = g0 - 1.0;
    if hp.rise < 50.0 {
        var t = 0.0;
        var step = texel * 0.75;
        for (var i = 0u; i < 320u; i++) {
            t += step;
            step *= 1.025;
            // No farther occluder can rise above the best height found so far.
            if hp.max_ground - hp.rise * t <= h {
                break;
            }
            let g = ground_at(xz + hp.sun_xz * t);
            h = max(h, g - hp.rise * t);
        }
    }
    textureStore(horizon_out, vec2<i32>(id.xy), vec4<f32>(h, 0.0, 0.0, 0.0));
}
"#;

pub struct HorizonMap {
    size: u32,
    #[cfg_attr(not(test), allow(dead_code))]
    texture: wgpu::Texture,
    pub ground_texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    origin_xz: [f32; 2],
    extent: f32,
    key: Option<HorizonKey>,
    dirty: bool,
}

/// Everything the map depends on; a change means a rebuild.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HorizonKey {
    origin_xz: [f32; 2],
    extent: f32,
    sun: [f32; 3],
    source: GroundSource,
    terrain_epoch: u64,
}

impl HorizonMap {
    pub fn new(device: &wgpu::Device, shadow_uniforms: &wgpu::Buffer, size: u32) -> Self {
        let size = size.max(1);
        let extent3 = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let horizon_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-horizon"),
            size: extent3,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = horizon_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let ground_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain-horizon-ground"),
            size: extent3,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ground_view = ground_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-horizon-params"),
            contents: bytemuck::bytes_of(&HorizonParams::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-horizon-layout"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-horizon-bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ground_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain-horizon-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}{}{HORIZON_WGSL_BODY}",
                    shadow_uniforms_wgsl!(),
                    terrain_formula_wgsl!()
                )
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-horizon-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain-horizon"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_horizon"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            size,
            texture: horizon_texture,
            ground_texture,
            view,
            params_buf,
            bind_group,
            pipeline,
            origin_xz: [0.0, 0.0],
            extent: 0.0,
            key: None,
            dirty: false,
        }
    }

    /// Region the map currently covers: `(origin_xz, extent)`.
    pub fn region(&self) -> ([f32; 2], f32) {
        (self.origin_xz, self.extent)
    }

    /// Pick the region and decide whether the map must be rebuilt this frame.
    ///
    /// `region` fixes the region (the atlas already chose one); otherwise the
    /// map follows `focus`, recentring only after the camera has crossed an
    /// eighth of the region, on a coarse grid so it never creeps.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        focus: Vec3,
        extent: f32,
        region: Option<[f32; 2]>,
        sun: Vec3,
        source: GroundSource,
        terrain_epoch: u64,
        max_ground: f32,
    ) {
        let extent = extent.max(1.0);
        let origin_xz = match region {
            Some(origin) => origin,
            None => {
                let grid = extent / 8.0;
                let keep = self.key.is_some_and(|k| {
                    k.extent == extent
                        && k.source == source
                        && (focus.x - (k.origin_xz[0] + extent * 0.5)).abs() < grid
                        && (focus.z - (k.origin_xz[1] + extent * 0.5)).abs() < grid
                });
                if keep {
                    self.origin_xz
                } else {
                    [
                        (focus.x / grid).round() * grid - extent * 0.5,
                        (focus.z / grid).round() * grid - extent * 0.5,
                    ]
                }
            }
        };
        let sun = sun.normalize_or_zero();
        let key = HorizonKey {
            origin_xz,
            extent,
            sun: [sun.x, sun.y, sun.z],
            source,
            terrain_epoch,
        };
        if self.key == Some(key) {
            return;
        }
        self.key = Some(key);
        self.origin_xz = origin_xz;
        self.extent = extent;
        self.dirty = true;

        let xz_len = Vec3::new(sun.x, 0.0, sun.z).length();
        let (sun_xz, rise) = if xz_len < 1e-4 {
            ([0.0, 0.0], 1e9)
        } else {
            ([sun.x / xz_len, sun.z / xz_len], sun.y / xz_len)
        };
        let params = HorizonParams {
            origin_xz,
            extent,
            size: self.size,
            sun_xz,
            rise,
            max_ground,
            source: match source {
                GroundSource::Formula => 0,
                GroundSource::Atlas => 1,
            },
            _pad: [0; 3],
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Rebuild the map if `prepare` asked for it. Returns whether it ran.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if !std::mem::take(&mut self.dirty) {
            return false;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain-horizon"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let groups = self.size.div_ceil(WORKGROUP);
        pass.dispatch_workgroups(groups, groups, 1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_field::verify::gpu_tests_enabled;
    use crate::proc_terrain::{demo_terrain_rules, sample_field};
    use crate::render::shadow::{apply_terrain_rules, ShadowUniforms};

    /// CPU twin of `cs_horizon` for one texel.
    fn cpu_horizon(
        xz: glam::Vec2,
        texel: f32,
        sun_xz: glam::Vec2,
        rise: f32,
        max_ground: f32,
    ) -> f32 {
        let rules = demo_terrain_rules();
        let g0 = sample_field(&rules, xz.x, xz.y).ground;
        let mut h = g0 - 1.0;
        let mut t = 0.0;
        let mut step = texel * 0.75;
        for _ in 0..320 {
            t += step;
            step *= 1.025;
            if max_ground - rise * t <= h {
                break;
            }
            let p = xz + sun_xz * t;
            h = h.max(sample_field(&rules, p.x, p.y).ground - rise * t);
        }
        h
    }

    #[test]
    fn gpu_horizon_matches_cpu_march() {
        if !gpu_tests_enabled() {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");

        let rules = demo_terrain_rules();
        let mut uniforms = ShadowUniforms::zeroed();
        apply_terrain_rules(&mut uniforms, &rules);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let size = 64;
        let mut map = HorizonMap::new(&device, &uniform_buf, size);
        let sun = Vec3::new(0.9, 0.18, 0.3);
        let max_ground = rules.base_height + rules.hill_height;
        map.prepare(
            &queue,
            Vec3::ZERO,
            256.0,
            None,
            sun,
            GroundSource::Formula,
            1,
            max_ground,
        );

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(size * size * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(map.encode(&mut encoder));
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &map.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::Wait).expect("poll");
        let data = readback.slice(..).get_mapped_range();
        let gpu: &[f32] = bytemuck::cast_slice(&data);

        let (origin, extent) = map.region();
        let texel = extent / size as f32;
        let sun_n = sun.normalize();
        let xz_len = glam::Vec2::new(sun_n.x, sun_n.z).length();
        let sun_xz = glam::Vec2::new(sun_n.x, sun_n.z) / xz_len;
        let rise = sun_n.y / xz_len;
        let mut shadowed = 0;
        let mut worst: f32 = 0.0;
        for y in (0..size).step_by(7) {
            for x in (0..size).step_by(7) {
                let xz = glam::Vec2::new(origin[0], origin[1])
                    + (glam::Vec2::new(x as f32, y as f32) + 0.5) * texel;
                let cpu = cpu_horizon(xz, texel, sun_xz, rise, max_ground);
                let g = gpu[(y * size + x) as usize];
                worst = worst.max((cpu - g).abs());
                if cpu > sample_field(&rules, xz.x, xz.y).ground {
                    shadowed += 1;
                }
            }
        }
        assert!(
            worst < 0.05,
            "GPU horizon differs from CPU march by {worst} m"
        );
        assert!(
            shadowed > 0,
            "a low sun over these hills must shadow something"
        );
    }
}
