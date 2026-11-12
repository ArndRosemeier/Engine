//! GPU skinning pipeline and mesh upload.

use crate::anim::{AnimatedModel, SkinMesh, MAX_JOINTS, MAX_MORPH_TARGETS};
use crate::mesh::InstanceRaw;
use crate::render::material::{GpuMaterial, MapCache, NeutralMaps, SharedMap};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
    /// Row of this vertex in the model's morph table, or [`NO_MORPH`].
    pub morph: u32,
}

/// `SkinnedVertex::morph` of a vertex no morph target moves.
pub const NO_MORPH: u32 = u32::MAX;

impl SkinnedVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SkinnedVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x4,
            3 => Float32x2,
            4 => Uint16x4,
            5 => Float32x4,
            15 => Uint32,
        ],
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct JointPalette {
    pub joints: [[[f32; 4]; 4]; MAX_JOINTS],
    /// Morph target weights, four per entry.
    pub morph_weights: [[f32; 4]; MAX_MORPH_TARGETS / 4],
    /// x: the model's morph target count (the morph table's row width).
    pub morph_info: [u32; 4],
}

impl JointPalette {
    /// Joint matrices plus the weights of the model's morph targets (one
    /// per target, in model order).
    pub fn with_morphs(mats: &[Mat4], morph_weights: &[f32]) -> Self {
        let mut joints = [[[0.0; 4]; 4]; MAX_JOINTS];
        for (i, m) in mats.iter().take(MAX_JOINTS).enumerate() {
            joints[i] = m.to_cols_array_2d();
        }
        // Unused slots identity so stray weights don't explode.
        for slot in joints.iter_mut().skip(mats.len().min(MAX_JOINTS)) {
            *slot = Mat4::IDENTITY.to_cols_array_2d();
        }
        let mut weights = [[0.0; 4]; MAX_MORPH_TARGETS / 4];
        let count = morph_weights.len().min(MAX_MORPH_TARGETS);
        for (i, w) in morph_weights.iter().take(count).enumerate() {
            weights[i / 4][i % 4] = *w;
        }
        Self {
            joints,
            morph_weights: weights,
            morph_info: [count as u32, 0, 0, 0],
        }
    }
}

/// WGSL shared by the skinned shaders: the joint palette with morph weights
/// (group 1) and the morph table, and `morph_vertex`, which adds the morph
/// offsets of a vertex to its bind-pose position and normal.
pub const SKIN_WGSL: &str = r#"
struct Joints {
    m: array<mat4x4<f32>, 128>,
    morph_w: array<vec4<f32>, 16>,
    morph_info: vec4<u32>,
};

@group(1) @binding(0) var<uniform> bones: Joints;
// Per morph table row and target: position offset (x, y | z, -) and normal
// offset (x, y | z, -) as half floats.
@group(1) @binding(1) var<storage, read> morph_table: array<vec4<u32>>;

struct Morphed {
    p: vec3<f32>,
    n: vec3<f32>,
};

fn morph_vertex(row: u32, p: vec3<f32>, n: vec3<f32>) -> Morphed {
    var out = Morphed(p, n);
    let count = bones.morph_info.x;
    if row == 0xffffffffu || count == 0u {
        return out;
    }
    for (var t = 0u; t < count; t++) {
        let w = bones.morph_w[t / 4u][t % 4u];
        if w == 0.0 {
            continue;
        }
        let d = morph_table[row * count + t];
        out.p += w * vec3<f32>(unpack2x16float(d.x), unpack2x16float(d.y).x);
        out.n += w * vec3<f32>(unpack2x16float(d.z), unpack2x16float(d.w).x);
    }
    return out;
}
"#;

/// f32 to IEEE half float bits (round to nearest; out of range saturates).
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if !v.is_finite() {
        return sign;
    }
    let a = v.abs();
    if a < 6.0e-8 {
        return sign;
    }
    if a >= 65504.0 {
        return sign | 0x7bff;
    }
    if a < 6.103_515_6e-5 {
        // Subnormal: units of 2^-24.
        return sign | (a * 16_777_216.0).round() as u16;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    // Round to nearest, ties to even.
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h.min(0x7bff) as u16
}

fn pack_half2(a: f32, b: f32) -> u32 {
    f16_bits(a) as u32 | (f16_bits(b) as u32) << 16
}

/// A model's morph table: the rows of each mesh's vertices (one row per
/// vertex some target moves, [`NO_MORPH`] for the rest) and the table, one
/// `[u32; 4]` per row and target (see [`SKIN_WGSL`]).
pub fn morph_table(model: &AnimatedModel) -> (Vec<Vec<u32>>, Vec<[u32; 4]>) {
    let targets = model.morph_targets.len();
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(model.meshes.len());
    let mut table: Vec<[u32; 4]> = Vec::new();
    let mut next = 0u32;
    for mesh in &model.meshes {
        let mut mine = vec![NO_MORPH; mesh.positions.len()];
        if targets == 0 || mesh.morphs.is_empty() {
            rows.push(mine);
            continue;
        }
        for m in &mesh.morphs {
            for &v in &m.vertices {
                if let Some(r) = mine.get_mut(v as usize).filter(|r| **r == NO_MORPH) {
                    *r = next;
                    next += 1;
                }
            }
        }
        let mut offsets =
            vec![(glam::Vec3::ZERO, glam::Vec3::ZERO); next as usize * targets - table.len()];
        let base = table.len();
        for m in &mesh.morphs {
            for (k, &v) in m.vertices.iter().enumerate() {
                let Some(&row) = mine.get(v as usize) else {
                    continue;
                };
                let i = row as usize * targets + m.target - base;
                offsets[i].0 += m.positions[k];
                offsets[i].1 += m.normals.get(k).copied().unwrap_or_default();
            }
        }
        table.extend(offsets.iter().map(|(p, n)| {
            [
                pack_half2(p.x, p.y),
                pack_half2(p.z, 0.0),
                pack_half2(n.x, n.y),
                pack_half2(n.z, 0.0),
            ]
        }));
        rows.push(mine);
    }
    (rows, table)
}

/// The morph table on the GPU (a 1-row zero table when the model has none, so
/// the bind group is always complete).
pub fn upload_morph_table(device: &wgpu::Device, table: &[[u32; 4]]) -> wgpu::Buffer {
    let zero = [[0u32; 4]];
    let data = if table.is_empty() { &zero[..] } else { table };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("skinned-morph-table"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

const SKINNED_SHADER: &str = r#"
@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;

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
    @location(15) morph: u32,
};

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
    var skin = mat4x4<f32>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
    );
    skin += bones.m[v.joints.x] * v.weights.x;
    skin += bones.m[v.joints.y] * v.weights.y;
    skin += bones.m[v.joints.z] * v.weights.z;
    skin += bones.m[v.joints.w] * v.weights.w;

    let bind = morph_vertex(v.morph, v.position, v.normal);
    let skinned_pos = skin * vec4<f32>(bind.p, 1.0);
    let skinned_n = skin * vec4<f32>(bind.n, 0.0);
    let world = model * skinned_pos;
    var out: VsOut;
    out.clip = u.view_proj * world;
    out.world_n = normalize((model * skinned_n).xyz);
    out.color = v.color;
    out.world_p = world.xyz;
    out.uv = v.uv;
    return out;
}

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
    // Under MSAA alpha becomes coverage (soft lashes, hair cards); without it, cut at half.
    if base.a < SKIN_ALPHA_CUT {
        discard;
    }
    // Characters keep their full-strength sun (no ambient budget split).
    let sun_diffuse = wrap * wrap / max(1.0 - u.ambient, 0.05);
    let lit = shade_surface_occluded(base.rgb, n, in.world_p, mat.roughness * orm.y,
            mat.metallic * orm.z, vis, sun_diffuse, orm.x)
        + base.rgb * torch_light(in.world_p, n)
        + material_emission(in.uv);
    return vec4<f32>(hdr_safe(haze(lit, in.world_p)), base.a);
}

// Depth prepass: only texels the shaded pass covers on every sample, so
// see-through parts (lash cards, hair strands) never hide what is behind.
@fragment
fn fs_depth(in: VsOut) {
    let texel = textureSample(albedo_tex, albedo_sampler, in.uv);
    if in.color.a * texel.a < 0.999 {
        discard;
    }
}
"#;

pub struct SkinnedPipelines {
    pub opaque: wgpu::RenderPipeline,
    pub opaque_portal: wgpu::RenderPipeline,
    /// Depth-only twin of `opaque` for the depth prepass.
    pub depth_prepass: wgpu::RenderPipeline,
    pub joint_bind_layout: wgpu::BindGroupLayout,
}

pub fn joint_bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("joint-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // The model's morph table.
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn create_skinned_pipelines(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    scene_bind_layout: &wgpu::BindGroupLayout,
    joint_bind_layout: wgpu::BindGroupLayout,
    albedo_layout: &wgpu::BindGroupLayout,
) -> SkinnedPipelines {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinned-shader"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                "const SKIN_ALPHA_CUT: f32 = {};\n{}{SKIN_WGSL}{SKINNED_SHADER}{}",
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("skinned-pipeline-layout"),
        bind_group_layouts: &[scene_bind_layout, &joint_bind_layout, albedo_layout],
        push_constant_ranges: &[],
    });

    let instance_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            6 => Float32x4,
            7 => Float32x4,
            8 => Float32x4,
            9 => Float32x4,
        ],
    };

    let make = |label: &str, depth_stencil: wgpu::DepthStencilState, shaded: bool| {
        let color_targets = [Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[SkinnedVertex::LAYOUT, instance_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(if shaded { "fs_main" } else { "fs_depth" }),
                targets: if shaded { &color_targets } else { &[] },
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil),
            // The shaded passes turn alpha into coverage.
            multisample: super::stencil::scene_multisample(shaded),
            multiview: None,
            cache: crate::render::pipeline_cache::get(device).as_ref(),
        })
    };

    let opaque = make(
        "skinned-opaque",
        super::stencil::scene_depth_stencil_unmasked(),
        true,
    );
    let opaque_portal = make(
        "skinned-opaque-portal",
        super::stencil::scene_depth_stencil_masked(),
        true,
    );
    let depth_prepass = make(
        "skinned-depth-prepass",
        super::stencil::scene_depth_stencil_unmasked(),
        false,
    );

    SkinnedPipelines {
        opaque,
        opaque_portal,
        depth_prepass,
        joint_bind_layout,
    }
}

#[derive(Clone)]
pub struct GpuSkinnedMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
    pub albedo_bind: wgpu::BindGroup,
    /// The shared maps the bind group uses, kept alive with it.
    _maps: Vec<Arc<SharedMap>>,
}

impl GpuSkinnedMesh {
    /// Uploads `mesh`'s vertices and indices, textured with `material`'s
    /// maps (the mesh itself, or for a crowd's simplified far mesh the
    /// mesh it was made from). Maps come from `cache`, so characters that
    /// share a texture share one GPU copy.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        albedo_layout: &wgpu::BindGroupLayout,
        material_sampler: &wgpu::Sampler,
        neutral: &NeutralMaps,
        cache: &mut MapCache,
        mesh: &SkinMesh,
        material: &SkinMesh,
        morph_rows: Option<&[u32]>,
    ) -> Self {
        let mut vertices = Vec::with_capacity(mesh.positions.len());
        for i in 0..mesh.positions.len() {
            vertices.push(SkinnedVertex {
                position: mesh.positions[i].into(),
                normal: mesh.normals[i].into(),
                color: mesh.colors[i].into(),
                uv: mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                joints: mesh.joints[i],
                weights: mesh.weights[i],
                morph: morph_rows
                    .and_then(|r| r.get(i))
                    .copied()
                    .unwrap_or(NO_MORPH),
            });
        }
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned-vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned-indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let (material, maps) = GpuMaterial::shared(
            device,
            queue,
            albedo_layout,
            material_sampler,
            neutral,
            cache,
            material.albedo.as_ref(),
            &material.maps,
        );
        Self {
            vertex_buf,
            index_buf,
            index_count: mesh.indices.len() as u32,
            albedo_bind: material.bind,
            _maps: maps,
        }
    }

    /// Video memory of the vertex and index buffers (bytes).
    pub fn buffer_bytes(&self) -> u64 {
        self.vertex_buf.size() + self.index_buf.size()
    }
}

pub struct GpuSkinnedEntity {
    pub meshes: Vec<GpuSkinnedMesh>,
    pub instance_buf: wgpu::Buffer,
    pub joint_buf: wgpu::Buffer,
    pub joint_bind: wgpu::BindGroup,
}

impl GpuSkinnedEntity {
    /// Same GPU vertices as another animal of this model; only pose buffers are new.
    pub fn from_shared_meshes(
        device: &wgpu::Device,
        joint_layout: &wgpu::BindGroupLayout,
        meshes: Vec<GpuSkinnedMesh>,
        morph_table: &wgpu::Buffer,
        transform: Mat4,
        joints: &[Mat4],
        morph_weights: &[f32],
    ) -> Self {
        let instance = InstanceRaw::from_matrix(transform);
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned-instance"),
            contents: bytemuck::bytes_of(&instance),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let palette = JointPalette::with_morphs(joints, morph_weights);
        let joint_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("joint-palette"),
            contents: bytemuck::bytes_of(&palette),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let joint_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint-bind"),
            layout: joint_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: joint_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: morph_table.as_entire_binding(),
                },
            ],
        });
        Self {
            meshes,
            instance_buf,
            joint_buf,
            joint_bind,
        }
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        transform: Mat4,
        joints: &[Mat4],
        morph_weights: &[f32],
    ) {
        let instance = InstanceRaw::from_matrix(transform);
        queue.write_buffer(&self.instance_buf, 0, bytemuck::bytes_of(&instance));
        let palette = JointPalette::with_morphs(joints, morph_weights);
        queue.write_buffer(&self.joint_buf, 0, bytemuck::bytes_of(&palette));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::MeshMorph;
    use glam::Vec3;

    fn half(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1f) as i32;
        let mant = (bits & 0x3ff) as f32;
        sign * if exp == 0 {
            mant * 2f32.powi(-24)
        } else {
            (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
        }
    }

    #[test]
    fn half_floats_round_trip() {
        for v in [0.0, 1.0, -2.5, 0.0123, -0.000_4, 3.0e-6, 1000.0] {
            let back = half(f16_bits(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {back}");
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(1.0e9), 0x7bff);
    }

    #[test]
    fn morph_table_rows_cover_moved_vertices() {
        let m = |target, v: u32, y: f32| MeshMorph {
            target,
            vertices: vec![v],
            positions: vec![Vec3::new(0.0, y, 0.0)],
            normals: vec![Vec3::X],
        };
        let model = crate::anim::tests::morph_model(
            vec![m(0, 2, 0.5), m(1, 2, -0.25), m(1, 0, 1.0)],
            &["a", "b"],
        );
        let (rows, table) = morph_table(&model);
        // Vertex 2 gets row 0, vertex 0 row 1; vertex 1 moves with nothing.
        assert_eq!(rows, vec![vec![1, NO_MORPH, 0]]);
        assert_eq!(table.len(), 2 * 2);
        let y = |row: usize, t: usize| half((table[row * 2 + t][0] >> 16) as u16);
        assert_eq!((y(0, 0), y(0, 1), y(1, 0), y(1, 1)), (0.5, -0.25, 0.0, 1.0));
        assert_eq!(half(table[0][2] as u16), 1.0, "normal offset x");
        // No morph targets: every vertex is NO_MORPH and the table is empty.
        let (rows, table) = morph_table(&crate::anim::tests::morph_model(Vec::new(), &[]));
        assert!(rows[0].iter().all(|&r| r == NO_MORPH) && table.is_empty());
    }
}
