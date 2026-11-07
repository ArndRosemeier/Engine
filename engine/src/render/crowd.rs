//! Crowds on the GPU: each crowd's idle and walk clips baked into a texture
//! of joint matrices (30 frames a second, blended in the vertex shader), and
//! every far agent one instance: one draw per mesh for the whole crowd.

use super::frustum::Frustum;
use super::skinned::{GpuSkinnedMesh, SkinnedVertex};
use crate::anim::{sample_joint_matrices, AnimatedModel, SkinMesh};
use crate::crowd::{Crowd, CrowdInstance, CROWD_BAKE_FPS};
use crate::world::World;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CrowdInstanceRaw {
    model: [[f32; 4]; 4],
    /// Current clip: first row, frame count, frame position, weight of `fade`.
    clip: [f32; 4],
    /// The clip fading out: first row, frame count, frame position.
    fade: [f32; 4],
}

const INSTANCE_ATTRS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    6 => Float32x4,
    7 => Float32x4,
    8 => Float32x4,
    9 => Float32x4,
    10 => Float32x4,
    11 => Float32x4,
];

fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<CrowdInstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_ATTRS,
    }
}

/// Posing from the baked texture, shared by the shaded and shadow passes.
const CROWD_POSE_WGSL: &str = r#"
@group(1) @binding(0) var crowd_anim: texture_2d<f32>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
    @location(6) m0: vec4<f32>,
    @location(7) m1: vec4<f32>,
    @location(8) m2: vec4<f32>,
    @location(9) m3: vec4<f32>,
    @location(10) clip: vec4<f32>,
    @location(11) fade: vec4<f32>,
};

// The top three rows of a skinning matrix.
struct Rows { r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32> };

fn joint_rows(row: i32, j: u32) -> Rows {
    let x = i32(j) * 3;
    return Rows(textureLoad(crowd_anim, vec2<i32>(x, row), 0),
                textureLoad(crowd_anim, vec2<i32>(x + 1, row), 0),
                textureLoad(crowd_anim, vec2<i32>(x + 2, row), 0));
}

fn add_rows(a: Rows, b: Rows, w: f32) -> Rows {
    return Rows(a.r0 + b.r0 * w, a.r1 + b.r1 * w, a.r2 + b.r2 * w);
}

// One clip's skin matrix for this vertex, blended between baked frames.
fn pose_rows(c: vec4<f32>, joints: vec4<u32>, weights: vec4<f32>) -> Rows {
    let frames = max(i32(c.y), 1);
    let f0 = i32(floor(c.z)) % frames;
    let f1 = (f0 + 1) % frames;
    let t = fract(c.z);
    let row0 = i32(c.x) + f0;
    let row1 = i32(c.x) + f1;
    var acc = Rows(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var k = 0u; k < 4u; k++) {
        let w = weights[k];
        if w > 0.0 {
            acc = add_rows(acc, joint_rows(row0, joints[k]), w * (1.0 - t));
            acc = add_rows(acc, joint_rows(row1, joints[k]), w * t);
        }
    }
    return acc;
}

fn crowd_skin(v: VsIn) -> Rows {
    var r = pose_rows(v.clip, v.joints, v.weights);
    if v.clip.w > 0.001 {
        let o = pose_rows(v.fade, v.joints, v.weights);
        let w = v.clip.w;
        r = Rows(mix(r.r0, o.r0, w), mix(r.r1, o.r1, w), mix(r.r2, o.r2, w));
    }
    return r;
}

fn apply_rows(r: Rows, p: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r.r0, p), dot(r.r1, p), dot(r.r2, p));
}
"#;

const CROWD_SHADER: &str = r#"
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world_n: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_p: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let skin = crowd_skin(v);
    let world = model * vec4<f32>(apply_rows(skin, vec4<f32>(v.position, 1.0)), 1.0);
    let n = apply_rows(skin, vec4<f32>(v.normal, 0.0));
    var out: VsOut;
    out.clip = u.view_proj * world;
    out.world_n = normalize((model * vec4<f32>(n, 0.0)).xyz);
    out.color = v.color;
    out.world_p = world.xyz;
    out.uv = v.uv;
    return out;
}

// The same shading as full characters, so a swap doesn't change the look.
@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var geo_n = normalize(in.world_n);
    if (!front) {
        geo_n = -geo_n;
    }
    let n = mapped_normal(geo_n, in.world_p, in.uv);
    let orm = material_orm(in.uv);
    let l = normalize(u.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.5 + 0.5;
    let vis = sun_visibility(in.world_p, geo_n, u.eye);
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    let base = in.color * texel;
    if base.a < SKIN_ALPHA_CUT {
        discard;
    }
    let sun_diffuse = wrap * wrap / max(1.0 - u.ambient, 0.05);
    let lit = shade_surface_occluded(base.rgb, n, in.world_p, mat.roughness * orm.y,
            mat.metallic * orm.z, vis, sun_diffuse, orm.x)
        + base.rgb * torch_light(in.world_p, n)
        + material_emission(in.uv);
    return vec4<f32>(haze(lit, in.world_p), base.a);
}
"#;

const CROWD_SHADOW_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> cascade: mat4x4<f32>;

@vertex
fn vs_shadow(v: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    let skin = crowd_skin(v);
    return cascade * (model * vec4<f32>(apply_rows(skin, vec4<f32>(v.position, 1.0)), 1.0));
}
"#;

/// The baked clips of one model: rows of `joints * 3` texels per frame.
pub(crate) struct Bake {
    pub width: u32,
    pub rows: Vec<[f32; 4]>,
    /// First row and frame count of idle and walk.
    pub clips: [(u32, u32); 2],
}

pub(crate) fn bake(model: &AnimatedModel, clips: [usize; 2]) -> Bake {
    let joints = model.skeleton.joint_names.len().max(1);
    let width = (joints * 3) as u32;
    let mut rows = Vec::new();
    let mut out = [(0u32, 0u32); 2];
    for (k, &clip) in clips.iter().enumerate() {
        let duration = model.clips[clip].duration.max(1e-3);
        let frames = ((duration * CROWD_BAKE_FPS).ceil() as u32).max(2);
        out[k] = ((rows.len() / width as usize) as u32, frames);
        for f in 0..frames {
            // Frames span the loop exactly, so the last blends into the first.
            let t = f as f32 * duration / frames as f32;
            let mats = sample_joint_matrices(model, clip, t);
            for j in 0..joints {
                let m = mats.get(j).copied().unwrap_or(glam::Mat4::IDENTITY);
                for r in 0..3 {
                    rows.push(m.row(r).to_array());
                }
            }
        }
    }
    Bake {
        width,
        rows,
        clips: out,
    }
}

impl Bake {
    fn raw(&self, c: &Crowd, i: &CrowdInstance) -> CrowdInstanceRaw {
        let pos = |clip: usize, time: f32| {
            let (start, frames) = self.clips[clip];
            let d = c.duration(clip);
            [start as f32, frames as f32, time / d * frames as f32]
        };
        let [a0, a1, a2] = pos(i.clip, i.time);
        let (fade, w) = match i.fade {
            Some((clip, time, w)) => {
                let [b0, b1, b2] = pos(clip, time);
                ([b0, b1, b2, 0.0], w)
            }
            None => ([0.0; 4], 0.0),
        };
        CrowdInstanceRaw {
            model: i.place.to_matrix().to_cols_array_2d(),
            clip: [a0, a1, a2, w],
            fade,
        }
    }
}

struct CrowdDraw {
    key: (usize, usize),
    meshes: Vec<GpuSkinnedMesh>,
    bake: Bake,
    _anim: wgpu::Texture,
    anim_bind: wgpu::BindGroup,
    instances: wgpu::Buffer,
    capacity: usize,
    /// Instances seen by the camera, then those casting shadows.
    visible: u32,
    shadow: u32,
}

pub(crate) struct CrowdGpu {
    anim_layout: wgpu::BindGroupLayout,
    opaque: wgpu::RenderPipeline,
    opaque_portal: wgpu::RenderPipeline,
    shadow: wgpu::RenderPipeline,
    draws: HashMap<u32, CrowdDraw>,
    scratch: Vec<CrowdInstanceRaw>,
}

impl CrowdGpu {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        scene_layout: &wgpu::BindGroupLayout,
        albedo_layout: &wgpu::BindGroupLayout,
        cascade_layout: &wgpu::BindGroupLayout,
        shadow_depth: wgpu::DepthStencilState,
    ) -> Self {
        let anim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("crowd-anim-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("crowd-shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "const SKIN_ALPHA_CUT: f32 = {};\n{}{CROWD_POSE_WGSL}{CROWD_SHADER}{}",
                    if super::stencil::scene_samples() > 1 {
                        "0.0"
                    } else {
                        "0.5"
                    },
                    super::pipeline::scene_shader_prefix(),
                    super::material::material_wgsl(2)
                )
                .into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("crowd-layout"),
            bind_group_layouts: &[scene_layout, &anim_layout, albedo_layout],
            push_constant_ranges: &[],
        });
        let make = |label: &str, depth: wgpu::DepthStencilState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[SkinnedVertex::LAYOUT, instance_layout()],
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
                depth_stencil: Some(depth),
                multisample: super::stencil::scene_multisample(true),
                multiview: None,
                cache: None,
            })
        };
        let opaque = make("crowd-opaque", super::stencil::scene_depth_stencil_unmasked());
        let opaque_portal = make(
            "crowd-opaque-portal",
            super::stencil::scene_depth_stencil_masked(),
        );
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("crowd-shadow"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{CROWD_POSE_WGSL}{CROWD_SHADOW_WGSL}").into(),
            ),
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("crowd-shadow-layout"),
            bind_group_layouts: &[cascade_layout, &anim_layout],
            push_constant_ranges: &[],
        });
        let shadow = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("crowd-shadow"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_shadow"),
                buffers: &[SkinnedVertex::LAYOUT, instance_layout()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(shadow_depth),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            anim_layout,
            opaque,
            opaque_portal,
            shadow,
            draws: HashMap::new(),
            scratch: Vec::new(),
        }
    }

    /// Bake new crowds and write this frame's instances: those the camera
    /// sees, then those within shadow range.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &World,
        frustum: &Frustum,
        eye: Vec3,
        mut upload: impl FnMut(&SkinMesh) -> GpuSkinnedMesh,
    ) {
        let live: Vec<u32> = world.crowds().map(|(id, _)| id.0).collect();
        self.draws.retain(|id, _| live.contains(id));
        for (id, crowd) in world.crowds() {
            let drawn: &Arc<AnimatedModel> = crowd.far.as_ref().unwrap_or(&crowd.model);
            let key = (
                Arc::as_ptr(&crowd.model) as usize,
                Arc::as_ptr(drawn) as usize,
            );
            if self.draws.get(&id.0).is_none_or(|d| d.key != key) {
                let bake = bake(&crowd.model, crowd.clips);
                let height = (bake.rows.len() / bake.width as usize).max(1) as u32;
                let anim = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("crowd-anim"),
                        size: wgpu::Extent3d {
                            width: bake.width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(&bake.rows),
                );
                let anim_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("crowd-anim"),
                    layout: &self.anim_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &anim.create_view(&Default::default()),
                        ),
                    }],
                });
                let capacity = 256;
                self.draws.insert(
                    id.0,
                    CrowdDraw {
                        key,
                        meshes: drawn.meshes.iter().map(&mut upload).collect(),
                        bake,
                        _anim: anim,
                        anim_bind,
                        instances: instance_buffer(device, capacity),
                        capacity,
                        visible: 0,
                        shadow: 0,
                    },
                );
            }
            let draw = self.draws.get_mut(&id.0).expect("just made");
            self.scratch.clear();
            let p = &crowd.profile;
            let mut shadow = Vec::new();
            for a in crowd.agents.iter().flatten().filter(|a| a.near.is_none()) {
                let at = a.place.position;
                let d = at.distance(eye);
                if d > p.far_m {
                    continue;
                }
                let raw = || draw.bake.raw(crowd, &CrowdInstance::of(crowd, a));
                let s = a.place.scale.max(0.01);
                if frustum.intersects_sphere(at + Vec3::Y * 0.9 * s, 1.3 * s) {
                    self.scratch.push(raw());
                }
                if d <= p.shadow_m {
                    shadow.push(raw());
                }
            }
            draw.visible = self.scratch.len() as u32;
            draw.shadow = shadow.len() as u32;
            self.scratch.extend(shadow);
            if self.scratch.len() > draw.capacity {
                draw.capacity = self.scratch.len().next_power_of_two();
                draw.instances = instance_buffer(device, draw.capacity);
            }
            if !self.scratch.is_empty() {
                queue.write_buffer(&draw.instances, 0, bytemuck::cast_slice(&self.scratch));
            }
        }
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, portal_mask: bool) {
        if self.draws.values().all(|d| d.visible == 0) {
            return;
        }
        pass.set_pipeline(if portal_mask {
            &self.opaque_portal
        } else {
            &self.opaque
        });
        for d in self.draws.values().filter(|d| d.visible > 0) {
            pass.set_bind_group(1, &d.anim_bind, &[]);
            pass.set_vertex_buffer(1, d.instances.slice(..));
            for mesh in &d.meshes {
                pass.set_bind_group(2, &mesh.albedo_bind, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..d.visible);
            }
        }
    }

    /// Into a sun cascade (group 0 already holds its view).
    pub fn draw_shadow(&self, pass: &mut wgpu::RenderPass<'_>, view_bind: &wgpu::BindGroup) {
        if self.draws.values().all(|d| d.shadow == 0) {
            return;
        }
        pass.set_pipeline(&self.shadow);
        pass.set_bind_group(0, view_bind, &[]);
        for d in self.draws.values().filter(|d| d.shadow > 0) {
            pass.set_bind_group(1, &d.anim_bind, &[]);
            pass.set_vertex_buffer(1, d.instances.slice(..));
            let range = d.visible..d.visible + d.shadow;
            for mesh in &d.meshes {
                pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, range.clone());
            }
        }
    }
}

fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("crowd-instances"),
        size: (capacity * std::mem::size_of::<CrowdInstanceRaw>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crowd::{IDLE, WALK};

    #[test]
    fn baked_frames_match_the_animator() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/model_viewer/assets/character.glb"
        );
        let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
        let model =
            AnimatedModel::load_with(path, examples, &crate::limits::EngineLimits::default())
                .unwrap();
        let walk = model.clips.iter().position(|c| c.name == "walk").unwrap();
        let idle = model.clips.iter().position(|c| c.name == "idle").unwrap();
        let b = bake(&model, [idle, walk]);
        let joints = model.skeleton.joint_names.len();
        let (start, frames) = b.clips[WALK];
        let d = model.clips[walk].duration;
        let f = 5u32;
        let mats = sample_joint_matrices(&model, walk, f as f32 * d / frames as f32);
        let row = (start + f) as usize * joints * 3;
        for j in 0..joints {
            for r in 0..3 {
                let got = glam::Vec4::from_array(b.rows[row + j * 3 + r]);
                assert!((got - mats[j].row(r)).length() < 1e-5);
            }
        }
        assert_eq!(b.clips[IDLE].0, 0);
        assert_eq!(b.rows.len(), (b.clips[0].1 + b.clips[1].1) as usize * joints * 3);
        assert_eq!(std::mem::size_of::<CrowdInstanceRaw>(), 96);
    }
}
