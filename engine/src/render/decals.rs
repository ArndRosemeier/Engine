//! Decals on the GPU (see `crate::decals`): every lit shader blends the
//! decals covering its pixel into the surface's albedo, normal, roughness
//! and metalness before lighting.
//!
//! Decals sit in a uniform array (oldest first, so newer ones paint over
//! older). A world-space grid of 4 m cells around the camera lists the
//! decals touching each cell, so a pixel only looks at a handful. The grid
//! lives in an R32Uint texture: texels 0..3 hold its origin and the decal
//! count, then one `offset << 8 | count` header per cell, then the indices.
//! Decal images are one texture array, two layers per kind: albedo with
//! coverage, then (normal x, normal y, roughness, metalness).

use crate::decals::{
    builtin_image, DecalImage, DecalKind, DECAL_TEXELS, MAX_DECALS, MAX_DECAL_KINDS,
};
use crate::world::World;
use glam::{IVec3, Vec3};
use wgpu::util::DeviceExt;

const GRID: IVec3 = IVec3::new(32, 16, 32);
const CELLS: usize = (GRID.x * GRID.y * GRID.z) as usize;
const CELL_M: f32 = 4.0;
const MAX_INDICES: usize = 1 << 16;
const ROW: usize = 4096;
const ROWS: usize = (4 + CELLS + MAX_INDICES).div_ceil(ROW);
const MIPS: u32 = 9;
/// Decals farther than this from the eye are not drawn (the grid's reach).
pub const DECAL_REACH_M: f32 = 60.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDecal {
    /// World to unit box, as three affine rows.
    rows: [[f32; 4]; 3],
    /// x image layer (albedo; material is the next), y opacity, z glow,
    /// w texels per metre over the box width.
    params: [f32; 4],
    tint: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuDecal>() == 80);

/// WGSL for the decal bindings in `group` from `first` (uniform, grid,
/// images); `sampler` is the binding of a linear clamping sampler, declared
/// here when `declare_sampler`.
pub fn decal_wgsl(group: u32, first: u32, sampler: u32, declare_sampler: bool) -> String {
    let samp = if declare_sampler {
        format!("@group({group}) @binding({sampler}) var reflect_samp: sampler;\n")
    } else {
        String::new()
    };
    format!(
        "{samp}@group({group}) @binding({}) var<uniform> decal_list: DecalList;
@group({group}) @binding({}) var decal_grid: texture_2d<u32>;
@group({group}) @binding({}) var decal_images: texture_2d_array<f32>;
{}",
        first,
        first + 1,
        first + 2,
        DECAL_WGSL.replace("MAX_DECALS", &MAX_DECALS.to_string())
    )
}

const DECAL_WGSL: &str = r#"
struct GpuDecal {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    params: vec4<f32>,
    tint: vec4<f32>,
};
struct DecalList {
    d: array<GpuDecal, MAX_DECALS>,
};

struct DecalSurface {
    base: vec3<f32>,
    n: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emit: vec3<f32>,
};

fn decal_texel(i: u32) -> u32 {
    return textureLoad(decal_grid, vec2<i32>(i32(i % 4096u), i32(i / 4096u)), 0).r;
}

// The surface at `p` with every decal over it painted on.
fn apply_decals(p: vec3<f32>, n: vec3<f32>, base: vec3<f32>, roughness: f32,
        metallic: f32, eye: vec3<f32>) -> DecalSurface {
    var s = DecalSurface(base, n, roughness, metallic, vec3<f32>(0.0));
    if decal_texel(3u) == 0u {
        return s;
    }
    let origin = vec3<f32>(bitcast<f32>(decal_texel(0u)), bitcast<f32>(decal_texel(1u)),
        bitcast<f32>(decal_texel(2u)));
    let g = vec3<i32>(floor((p - origin) / 4.0));
    let dims = vec3<i32>(32, 16, 32);
    if any(g < vec3<i32>(0)) || any(g >= dims) {
        return s;
    }
    let head = decal_texel(4u + u32((g.y * dims.z + g.z) * dims.x + g.x));
    let first = head >> 8u;
    let count = head & 255u;
    let dist = distance(p, eye);
    for (var k = 0u; k < count; k++) {
        let d = decal_list.d[decal_texel(first + k)];
        let q = vec4<f32>(p, 1.0);
        let local = vec3<f32>(dot(d.r0, q), dot(d.r1, q), dot(d.r2, q));
        if any(abs(local) > vec3<f32>(1.0)) {
            continue;
        }
        let nd = normalize(d.r2.xyz);
        let facing = dot(n, nd);
        if facing < 0.1 {
            continue;
        }
        let uv = vec2<f32>(local.x * 0.5 + 0.5, 0.5 - local.y * 0.5);
        // About one texel per pixel: a pixel spans ~dist * 0.0015 m.
        let lod = clamp(log2(max(dist * 0.0015 * d.params.w, 1e-4)), 0.0, 8.0);
        let layer = i32(d.params.x) * 2;
        let a = textureSampleLevel(decal_images, reflect_samp, uv, layer, lod);
        let depth_fade = 1.0 - pow(abs(local.z), 6.0);
        let alpha = a.a * d.params.y * smoothstep(0.1, 0.35, facing) * depth_fade;
        if alpha <= 0.003 {
            continue;
        }
        let m = textureSampleLevel(decal_images, reflect_samp, uv, layer + 1, lod);
        let albedo = pow(a.rgb, vec3<f32>(2.2)) * d.tint.rgb;
        s.base = mix(s.base, albedo, alpha);
        // Tangent frame from the box's own axes.
        let tx = normalize(d.r0.xyz - n * dot(d.r0.xyz, n));
        let ty = normalize(d.r1.xyz - n * dot(d.r1.xyz, n));
        let nxy = m.rg * 2.0 - vec2<f32>(1.0);
        let nz = sqrt(max(1.0 - dot(nxy, nxy), 0.0));
        let nn = normalize(tx * nxy.x + ty * nxy.y + s.n * nz);
        s.n = normalize(mix(s.n, nn, alpha));
        s.roughness = mix(s.roughness, m.b, alpha);
        s.metallic = mix(s.metallic, m.a, alpha);
        s.emit += albedo * d.params.z * alpha;
    }
    return s;
}
"#;

pub struct DecalGpu {
    pub uniform: wgpu::Buffer,
    grid: wgpu::Texture,
    pub grid_view: wgpu::TextureView,
    images: wgpu::Texture,
    pub images_view: wgpu::TextureView,
    /// Custom images uploaded so far.
    uploaded_custom: usize,
    /// Whether the last upload had decals (to clear the count once).
    had_decals: bool,
}

impl DecalGpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decal-list"),
            size: (MAX_DECALS * std::mem::size_of::<GpuDecal>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let grid = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("decal-grid"),
                size: wgpu::Extent3d {
                    width: ROW as u32,
                    height: ROWS as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &vec![0u8; ROW * ROWS * 4],
        );
        let grid_view = grid.create_view(&Default::default());
        let images = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("decal-images"),
            size: wgpu::Extent3d {
                width: DECAL_TEXELS as u32,
                height: DECAL_TEXELS as u32,
                depth_or_array_layers: 2 * MAX_DECAL_KINDS as u32,
            },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let images_view = images.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let gpu = Self {
            uniform,
            grid,
            grid_view,
            images,
            images_view,
            uploaded_custom: 0,
            had_decals: false,
        };
        for kind in DecalKind::BUILT_IN {
            gpu.upload_image(queue, kind.layer(), &builtin_image(kind));
        }
        gpu
    }

    fn upload_image(&self, queue: &wgpu::Queue, layer: usize, image: &DecalImage) {
        for (k, pixels) in [&image.albedo, &image.material].into_iter().enumerate() {
            // Alpha-weighted box filter for albedo so edges do not darken.
            let mut level = pixels.clone();
            let mut size = DECAL_TEXELS;
            for mip in 0..MIPS {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.images,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: (layer * 2 + k) as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &level,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size as u32 * 4),
                        rows_per_image: Some(size as u32),
                    },
                    wgpu::Extent3d {
                        width: size as u32,
                        height: size as u32,
                        depth_or_array_layers: 1,
                    },
                );
                if size == 1 {
                    break;
                }
                level = downsample(&level, size, k == 0);
                size /= 2;
            }
        }
    }

    /// Upload this frame's decals and their grid.
    pub fn upload(&mut self, queue: &wgpu::Queue, world: &World, eye: Vec3) {
        let images = world.decal_images();
        while self.uploaded_custom < images.len() {
            let layer = DecalKind::Custom(self.uploaded_custom as u16).layer();
            self.upload_image(queue, layer, &images[self.uploaded_custom]);
            self.uploaded_custom += 1;
        }
        let draws = world.decal_draws(eye, DECAL_REACH_M);
        if draws.is_empty() {
            if self.had_decals {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.grid,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &[0u8; 16],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(16),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d {
                        width: 4,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                self.had_decals = false;
            }
            return;
        }
        self.had_decals = true;
        let mut gpu = Vec::with_capacity(draws.len());
        let mut spheres = Vec::with_capacity(draws.len());
        for d in draws.iter().take(MAX_DECALS) {
            let inv = d.box_world.inverse();
            let row = |i: usize| {
                let r = inv.row(i);
                [r.x, r.y, r.z, r.w]
            };
            gpu.push(GpuDecal {
                rows: [row(0), row(1), row(2)],
                params: [
                    d.layer as f32,
                    d.opacity,
                    d.emission,
                    DECAL_TEXELS as f32 / d.width.max(0.01),
                ],
                tint: [d.tint.x, d.tint.y, d.tint.z, 0.0],
            });
            let m = d.box_world;
            let radius =
                (m.x_axis.truncate().abs() + m.y_axis.truncate().abs() + m.z_axis.truncate().abs())
                    .length();
            spheres.push((m.w_axis.truncate(), radius));
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&gpu));
        let texels = build_grid(eye, &spheres);
        let rows = texels.len().div_ceil(ROW);
        let mut padded = texels;
        padded.resize(rows * ROW, 0);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.grid,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&padded),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW as u32 * 4),
                rows_per_image: Some(rows as u32),
            },
            wgpu::Extent3d {
                width: ROW as u32,
                height: rows as u32,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Halve an RGBA8 image; `weighted` averages colour by alpha.
fn downsample(src: &[u8], size: usize, weighted: bool) -> Vec<u8> {
    let half = size / 2;
    let mut out = vec![0u8; half * half * 4];
    for y in 0..half {
        for x in 0..half {
            let mut sum = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let i = ((y * 2 + dy) * size + x * 2 + dx) * 4;
                let w = if weighted {
                    src[i + 3] as f32 / 255.0 + 1e-3
                } else {
                    1.0
                };
                for c in 0..3 {
                    sum[c] += src[i + c] as f32 * w;
                }
                sum[3] += src[i + 3] as f32;
                wsum += w;
            }
            let o = (y * half + x) * 4;
            for c in 0..3 {
                out[o + c] = (sum[c] / wsum).round() as u8;
            }
            out[o + 3] = (sum[3] / 4.0).round() as u8;
        }
    }
    out
}

/// The grid texels: header, cell lists, indices.
fn build_grid(eye: Vec3, spheres: &[(Vec3, f32)]) -> Vec<u32> {
    let origin = ((eye / CELL_M).floor() - GRID.as_vec3() * 0.5) * CELL_M;
    let range = |c: Vec3, r: f32| {
        let lo = ((c - Vec3::splat(r) - origin) / CELL_M)
            .floor()
            .as_ivec3()
            .max(IVec3::ZERO);
        let hi = ((c + Vec3::splat(r) - origin) / CELL_M)
            .floor()
            .as_ivec3()
            .min(GRID - IVec3::ONE);
        (lo, hi)
    };
    let cell = |g: IVec3| ((g.y * GRID.z + g.z) * GRID.x + g.x) as usize;
    let mut counts = vec![0u32; CELLS];
    for &(c, r) in spheres {
        let (lo, hi) = range(c, r);
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    let i = cell(IVec3::new(x, y, z));
                    counts[i] = (counts[i] + 1).min(255);
                }
            }
        }
    }
    let mut out = vec![0u32; 4 + CELLS];
    out[0] = origin.x.to_bits();
    out[1] = origin.y.to_bits();
    out[2] = origin.z.to_bits();
    out[3] = spheres.len() as u32;
    let limit = (4 + CELLS + MAX_INDICES) as u32;
    let mut offset = (4 + CELLS) as u32;
    for (i, n) in counts.iter().enumerate() {
        let n = (*n).min(limit.saturating_sub(offset));
        out[4 + i] = (offset << 8) | n;
        offset += n;
    }
    out.resize(offset as usize, 0);
    let mut fill = vec![0u32; CELLS];
    // Oldest first in every cell, so newer decals paint over older.
    for (k, &(c, r)) in spheres.iter().enumerate() {
        let (lo, hi) = range(c, r);
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    let i = cell(IVec3::new(x, y, z));
                    let head = out[4 + i];
                    if fill[i] < (head & 255) {
                        out[((head >> 8) + fill[i]) as usize] = k as u32;
                        fill[i] += 1;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_lists_decals_in_their_cells_oldest_first() {
        let eye = Vec3::new(3.0, 1.0, -2.0);
        let spheres = [
            (Vec3::new(5.0, 0.5, 0.0), 1.0),
            (Vec3::new(5.5, 0.5, 0.2), 0.5),
            (Vec3::new(40.0, 0.0, 0.0), 0.5),
        ];
        let g = build_grid(eye, &spheres);
        let origin = Vec3::new(
            f32::from_bits(g[0]),
            f32::from_bits(g[1]),
            f32::from_bits(g[2]),
        );
        assert_eq!(g[3], 3);
        let lookup = |p: Vec3| -> Vec<u32> {
            let c = ((p - origin) / CELL_M).floor().as_ivec3();
            let head = g[4 + ((c.y * GRID.z + c.z) * GRID.x + c.x) as usize];
            (0..head & 255)
                .map(|k| g[((head >> 8) + k) as usize])
                .collect()
        };
        assert_eq!(lookup(Vec3::new(5.2, 0.5, 0.1)), vec![0, 1]);
        assert_eq!(lookup(Vec3::new(40.0, 0.0, 0.0)), vec![2]);
        assert!(lookup(Vec3::new(20.0, 0.0, 20.0)).is_empty());
    }
}
