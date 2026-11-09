//! Surface texture sets for meshes: albedo plus glTF-style normal,
//! occlusion-roughness-metallic and emissive maps, with a small uniform of
//! factors. One bind group per set, shared by the static and skinned shaders.

use crate::mesh::{AlbedoMap, MaterialMaps};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialParams {
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    /// Used by meshes without per-vertex material (skinned characters).
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: [f32; 3],
    /// 1 when a normal map is bound.
    pub has_normal: f32,
    /// Normal map UV scale (xy) and offset (zw).
    pub normal_uv: [f32; 4],
    /// x: tiles per metre for triplanar mapping (0: mesh UVs).
    pub triplanar: [f32; 4],
}

impl MaterialParams {
    pub fn neutral() -> Self {
        Self::from_maps(&MaterialMaps::default())
    }

    pub fn from_maps(maps: &MaterialMaps) -> Self {
        Self {
            normal_scale: maps.normal_scale,
            occlusion_strength: maps.occlusion_strength,
            roughness: maps.roughness,
            metallic: maps.metallic,
            emissive: maps.emissive_factor,
            has_normal: if maps.normal.is_some() { 1.0 } else { 0.0 },
            normal_uv: maps.normal_uv,
            triplanar: [
                if maps.triplanar > 0.0 {
                    1.0 / maps.triplanar
                } else {
                    0.0
                },
                0.0,
                0.0,
                0.0,
            ],
        }
    }
}

/// Bindings 2..=5 of the mesh material group, and the helpers that read them.
/// Binding 0 is the albedo and 1 the sampler, declared by each shader.
pub fn material_wgsl(group: u32) -> String {
    format!(
        r#"
struct MaterialParams {{
    normal_scale: f32,
    occlusion_strength: f32,
    roughness: f32,
    metallic: f32,
    emissive: vec3<f32>,
    has_normal: f32,
    normal_uv: vec4<f32>,
    triplanar: vec4<f32>,
}};
@group({group}) @binding(2) var normal_tex: texture_2d<f32>;
@group({group}) @binding(3) var orm_tex: texture_2d<f32>;
@group({group}) @binding(4) var emissive_tex: texture_2d<f32>;
@group({group}) @binding(5) var<uniform> mat: MaterialParams;

// Shading normal from the tangent-space normal map. The tangent frame comes
// from screen-space derivatives of position and UV (a cotangent frame), so
// meshes need no stored tangents and mirrored UVs just work.
fn mapped_normal(n: vec3<f32>, p: vec3<f32>, mesh_uv: vec2<f32>) -> vec3<f32> {{
    if mat.has_normal < 0.5 {{
        return n;
    }}
    let uv = mesh_uv * mat.normal_uv.xy + mat.normal_uv.zw;
    var t = textureSample(normal_tex, albedo_sampler, uv).xyz * 2.0 - 1.0;
    t = vec3<f32>(t.xy * mat.normal_scale, t.z);
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    // Screen y runs down in wgpu, which turns the textbook frame round; and
    // glTF UVs run down the image, so the bitangent points against +v. Both
    // together leave the tangent negated and the bitangent as is. Checked
    // against Khronos' NormalTangentMirrorTest.
    let tangent = -(dp2perp * duv1.x + dp1perp * duv2.x);
    let bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
    let scale = max(dot(tangent, tangent), dot(bitangent, bitangent));
    if scale < 1e-20 {{
        return n;
    }}
    let inv = inverseSqrt(scale);
    let mapped = tangent * (t.x * inv) + bitangent * (t.y * inv) + n * t.z;
    let len2 = dot(mapped, mapped);
    if len2 < 1e-12 {{
        return n;
    }}
    return mapped * inverseSqrt(len2);
}}

// Occlusion (x), roughness and metallic multipliers (y, z) at `uv`.
fn material_orm(uv: vec2<f32>) -> vec3<f32> {{
    let s = textureSample(orm_tex, albedo_sampler, uv).rgb;
    return vec3<f32>(mix(1.0, s.r, mat.occlusion_strength), s.g, s.b);
}}

fn material_emission(uv: vec2<f32>) -> vec3<f32> {{
    return textureSample(emissive_tex, albedo_sampler, uv).rgb * mat.emissive;
}}
"#
    )
}

pub fn material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-material-layout"),
        entries: &[
            texture(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            texture(2),
            texture(3),
            texture(4),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Repeating, anisotropic sampler for imported material textures (glTF's
/// default wrap is repeat).
pub fn material_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mesh-material-sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapKind {
    /// sRGB colour (albedo, emissive).
    Color,
    /// Linear data (occlusion-roughness-metallic).
    Data,
    /// Tangent-space normals: linear, renormalised per mip.
    Normal,
}

/// Upload an RGBA8 map with a full CPU-built mip chain.
pub fn upload_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    map: &AlbedoMap,
    kind: MapKind,
) -> wgpu::Texture {
    let levels = mip_chain(map, kind);
    let format = match kind {
        MapKind::Color => wgpu::TextureFormat::Rgba8UnormSrgb,
        MapKind::Data | MapKind::Normal => wgpu::TextureFormat::Rgba8Unorm,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh-material-map"),
        size: wgpu::Extent3d {
            width: map.width,
            height: map.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (w, h, data)) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(*h),
            },
            wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Box-filtered mip levels, level 0 first. Colour is averaged in linear
/// light, normals are renormalised so distant bumps flatten correctly.
pub(crate) fn mip_chain(map: &AlbedoMap, kind: MapKind) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = vec![(map.width, map.height, map.rgba.to_vec())];
    let to_lin: Vec<f32> = (0..=255u8).map(srgb_to_linear).collect();
    loop {
        let (w, h, src) = levels.last().expect("level 0");
        if *w == 1 && *h == 1 {
            break;
        }
        let (w, h) = (*w as usize, *h as usize);
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut out = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                let mut count = 0.0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let i = (sy * w + sx) * 4;
                    for c in 0..4 {
                        let v = src[i + c];
                        acc[c] += if kind == MapKind::Color && c < 3 {
                            to_lin[v as usize]
                        } else {
                            v as f32 / 255.0
                        };
                    }
                    count += 1.0;
                }
                let o = (y * nw + x) * 4;
                match kind {
                    MapKind::Color => {
                        for c in 0..3 {
                            out[o + c] = linear_to_srgb(acc[c] / count);
                        }
                    }
                    MapKind::Data => {
                        for c in 0..3 {
                            out[o + c] = (acc[c] / count * 255.0 + 0.5) as u8;
                        }
                    }
                    MapKind::Normal => {
                        let n =
                            glam::Vec3::new(acc[0], acc[1], acc[2]) / count * 2.0 - glam::Vec3::ONE;
                        let n = n.try_normalize().unwrap_or(glam::Vec3::Z);
                        for c in 0..3 {
                            out[o + c] = ((n[c] * 0.5 + 0.5) * 255.0 + 0.5) as u8;
                        }
                    }
                }
                out[o + 3] = (acc[3] / count * 255.0 + 0.5) as u8;
            }
        }
        levels.push((nw as u32, nh as u32, out));
    }
    levels
}

/// 1x1 fallbacks for maps a material does not have.
pub struct NeutralMaps {
    pub white_srgb: wgpu::TextureView,
    pub white_linear: wgpu::TextureView,
    pub flat_normal: wgpu::TextureView,
}

impl NeutralMaps {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let texel = |rgba: [u8; 4], kind| {
            let map = AlbedoMap {
                width: 1,
                height: 1,
                rgba: rgba.to_vec().into(),
            };
            upload_map(device, queue, &map, kind).create_view(&Default::default())
        };
        Self {
            white_srgb: texel([255; 4], MapKind::Color),
            white_linear: texel([255; 4], MapKind::Data),
            flat_normal: texel([128, 128, 255, 255], MapKind::Normal),
        }
    }
}

/// GPU copy of one texture set. The bind group holds the textures alive.
pub struct GpuMaterial {
    pub bind: wgpu::BindGroup,
}

impl GpuMaterial {
    /// `albedo` is an already uploaded view (or `None` for white).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        neutral: &NeutralMaps,
        albedo: Option<&wgpu::TextureView>,
        maps: Option<&MaterialMaps>,
    ) -> Self {
        let up = |map: &Option<AlbedoMap>, kind| {
            map.as_ref()
                .map(|m| upload_map(device, queue, m, kind).create_view(&Default::default()))
        };
        let normal = maps.and_then(|m| up(&m.normal, MapKind::Normal));
        let orm = maps.and_then(|m| up(&m.orm, MapKind::Data));
        let emissive = maps.and_then(|m| up(&m.emissive, MapKind::Color));
        Self::from_views(
            device,
            layout,
            sampler,
            neutral,
            [albedo, normal.as_ref(), orm.as_ref(), emissive.as_ref()],
            maps,
        )
    }

    /// A texture set whose maps come from `cache`: maps with the same
    /// texels (the same garment on many characters, the near and far
    /// models of a crowd) are one GPU texture. The returned handles keep
    /// the cache's entries alive as long as the material is used.
    #[allow(clippy::too_many_arguments)]
    pub fn shared(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        neutral: &NeutralMaps,
        cache: &mut MapCache,
        albedo: Option<&AlbedoMap>,
        maps: &MaterialMaps,
    ) -> (Self, Vec<Arc<SharedMap>>) {
        let mut get =
            |map: Option<&AlbedoMap>, kind| map.map(|m| cache.get(device, queue, m, kind));
        let held: [Option<Arc<SharedMap>>; 4] = [
            get(albedo, MapKind::Color),
            get(maps.normal.as_ref(), MapKind::Normal),
            get(maps.orm.as_ref(), MapKind::Data),
            get(maps.emissive.as_ref(), MapKind::Color),
        ];
        let views = [0, 1, 2, 3].map(|i| held[i].as_ref().map(|m| &m.view));
        let material = Self::from_views(device, layout, sampler, neutral, views, Some(maps));
        (material, held.into_iter().flatten().collect())
    }

    /// The bind group over uploaded views: albedo, normal, ORM, emissive
    /// (`None`: the neutral texel).
    fn from_views(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        neutral: &NeutralMaps,
        [albedo, normal, orm, emissive]: [Option<&wgpu::TextureView>; 4],
        maps: Option<&MaterialMaps>,
    ) -> Self {
        let params = maps.map_or_else(MaterialParams::neutral, MaterialParams::from_maps);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh-material-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        fn view<'a>(
            v: Option<&'a wgpu::TextureView>,
            fallback: &'a wgpu::TextureView,
        ) -> wgpu::BindingResource<'a> {
            wgpu::BindingResource::TextureView(v.unwrap_or(fallback))
        }
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh-material"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view(albedo, &neutral.white_srgb),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: view(normal, &neutral.flat_normal),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view(orm, &neutral.white_linear),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: view(emissive, &neutral.white_srgb),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        Self { bind }
    }
}

/// One uploaded map that several materials share (see [`MapCache`]).
pub struct SharedMap {
    pub view: wgpu::TextureView,
    /// Video memory of the texture with its mips (bytes).
    pub bytes: u64,
}

/// Maps uploaded once per content: characters built from the same pack
/// textures (a garment, eyes, teeth, a hair card) carry byte-identical maps,
/// and each would otherwise be its own GPU texture. Entries live while a
/// material holds them.
#[derive(Default)]
pub struct MapCache {
    maps: HashMap<(u64, u32, u32, MapKind), Weak<SharedMap>>,
    inserts: usize,
}

impl MapCache {
    /// The GPU copy of `map`, uploaded now unless one with the same texels
    /// is still in use.
    pub fn get(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        map: &AlbedoMap,
        kind: MapKind,
    ) -> Arc<SharedMap> {
        let key = (
            crate::world::content_hash(&map.rgba),
            map.width,
            map.height,
            kind,
        );
        if let Some(m) = self.maps.get(&key).and_then(Weak::upgrade) {
            return m;
        }
        let bytes = mip_chain_bytes(map.width, map.height);
        let view = upload_map(device, queue, map, kind).create_view(&Default::default());
        let m = Arc::new(SharedMap { view, bytes });
        self.inserts += 1;
        if self.inserts.is_multiple_of(64) {
            self.maps.retain(|_, w| w.strong_count() > 0);
        }
        self.maps.insert(key, Arc::downgrade(&m));
        m
    }

    /// Maps in use and their video memory (bytes).
    pub fn usage(&self) -> (usize, u64) {
        self.maps
            .values()
            .filter_map(Weak::upgrade)
            .fold((0, 0), |(n, b), m| (n + 1, b + m.bytes))
    }
}

/// Bytes of an RGBA8 map of this size with its full mip chain.
fn mip_chain_bytes(width: u32, height: u32) -> u64 {
    let (mut w, mut h, mut sum) = (width as u64, height as u64, 0);
    loop {
        sum += w * h * 4;
        if w == 1 && h == 1 {
            return sum;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_reaches_one_texel_and_keeps_normals_unit() {
        let mut rgba = Vec::new();
        for i in 0..(6 * 3) {
            // Alternating tilted normals average to straight up.
            let x = if i % 2 == 0 { 200 } else { 56 };
            rgba.extend_from_slice(&[x, 128, 230, 255]);
        }
        let map = AlbedoMap {
            width: 6,
            height: 3,
            rgba: rgba.into(),
        };
        let levels = mip_chain(&map, MapKind::Normal);
        let sizes: Vec<(u32, u32)> = levels.iter().map(|l| (l.0, l.1)).collect();
        assert_eq!(sizes, vec![(6, 3), (3, 1), (1, 1)]);
        let last = &levels.last().unwrap().2;
        let n = glam::Vec3::new(last[0] as f32, last[1] as f32, last[2] as f32) / 255.0 * 2.0
            - glam::Vec3::ONE;
        assert!((n.length() - 1.0).abs() < 0.02, "unit length: {n:?}");
        assert!(n.z > 0.95, "flattened toward +z: {n:?}");
    }

    #[test]
    fn colour_mips_average_in_linear_light() {
        // Black and white average to linear 0.5, which is sRGB ~188, not 128.
        let map = AlbedoMap {
            width: 2,
            height: 1,
            rgba: vec![0, 0, 0, 255, 255, 255, 255, 255].into(),
        };
        let levels = mip_chain(&map, MapKind::Color);
        let px = &levels[1].2;
        assert!((185..=190).contains(&px[0]), "got {}", px[0]);
    }
}
