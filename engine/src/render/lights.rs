//! Lamps on the GPU: the lamp list, a world-space grid that tells each pixel
//! which lamps reach it, and the shadow maps of the lamps that cast shadows.
//!
//! The grid is two nested boxes of 32 x 16 x 32 cells around the camera: 8 m
//! cells out to 128 m, then 32 m cells out to 512 m. Each cell lists the
//! lamps whose reach touches it, so a pixel loops over a handful of lamps,
//! not hundreds. Being in world space, it works the same for every view
//! (portals, the GI probes) without any per-view rebuild.
//!
//! Shadowed lamps get layers in one depth-texture array: six (a cube) for a
//! point lamp, one for a spotlight.

use crate::lamps::{Lamp, LampShape, MAX_SHADOW_LAMPS};
use glam::{IVec3, Mat4, Vec3};

pub const MAX_LAMPS: usize = 256;
pub const GRID: IVec3 = IVec3::new(32, 16, 32);
pub const CELLS: usize = (GRID.x * GRID.y * GRID.z) as usize;
pub const CELL0_M: f32 = 8.0;
pub const CELL1_M: f32 = 32.0;
pub const MAX_INDICES: usize = 1 << 18;
pub const SHADOW_LAYERS: usize = 6 * MAX_SHADOW_LAMPS;
pub const SHADOW_SIZE: u32 = 512;
/// Point-lamp cube faces are a little wider than 90 degrees so the shadow
/// filter never reads past a face's edge.
const FACE_FOV_DEG: f32 = 96.0;
const SHADOW_NEAR: f32 = 0.05;

/// One lamp as the shaders read it (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLamp {
    /// xyz position, w range.
    pub pos_range: [f32; 4],
    /// rgb colour times intensity, w first shadow layer (-1: no shadow).
    pub color: [f32; 4],
    /// xyz spot direction, w cosine of the outer cone (-2 for point lamps).
    pub dir_cos: [f32; 4],
    /// x cosine of the inner cone, y 1 for a spotlight, zw unused.
    pub extra: [f32; 4],
}

/// Everything the renderer uploads for lamps this frame.
#[derive(Debug, Default)]
pub struct LampFrame {
    pub lamps: Vec<GpuLamp>,
    /// `2 * CELLS` packed cell headers (offset << 8 | count), then indices.
    pub cells: Vec<u32>,
    pub origin0: Vec3,
    pub origin1: Vec3,
    /// View-projection of each shadow layer, in layer order.
    pub shadow_views: Vec<Mat4>,
}

pub fn grid_origin(eye: Vec3, cell: f32) -> Vec3 {
    let half = GRID.as_vec3() * 0.5;
    ((eye / cell).floor() - half) * cell
}

fn cell_index(g: IVec3) -> usize {
    ((g.y * GRID.z + g.z) * GRID.x + g.x) as usize
}

/// The six cube-face view-projections of a point lamp, in the order the
/// shader picks them: +X, -X, +Y, -Y, +Z, -Z.
pub fn cube_views(p: Vec3, range: f32) -> [Mat4; 6] {
    let proj = glam::camera::rh::proj::directx::perspective(
        FACE_FOV_DEG.to_radians(),
        1.0,
        SHADOW_NEAR,
        range.max(SHADOW_NEAR * 2.0),
    );
    let faces = [
        (Vec3::X, Vec3::Y),
        (-Vec3::X, Vec3::Y),
        (Vec3::Y, Vec3::Z),
        (-Vec3::Y, Vec3::Z),
        (Vec3::Z, Vec3::Y),
        (-Vec3::Z, Vec3::Y),
    ];
    faces.map(|(dir, up)| proj * glam::camera::rh::view::look_at_mat4(p, p + dir, up))
}

pub fn spot_view(p: Vec3, dir: Vec3, angle_deg: f32, range: f32) -> Mat4 {
    let dir = dir.normalize();
    let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let fov = (angle_deg + 8.0).min(170.0).to_radians();
    glam::camera::rh::proj::directx::perspective(
        fov,
        1.0,
        SHADOW_NEAR,
        range.max(SHADOW_NEAR * 2.0),
    ) * glam::camera::rh::view::look_at_mat4(p, p + dir, up)
}

/// Choose this frame's lamps, give the most important ones shadow layers and
/// build the cell grid. `visible` says whether a sphere touches the view.
pub fn build_frame(
    lamps: impl Iterator<Item = (u32, Lamp)>,
    eye: Vec3,
    time: f32,
    shadow_budget: usize,
    visible: impl Fn(Vec3, f32) -> bool,
) -> LampFrame {
    let origin0 = grid_origin(eye, CELL0_M);
    let origin1 = grid_origin(eye, CELL1_M);
    let reach = GRID.x as f32 * 0.5 * CELL1_M * 0.94;

    // Candidates that can light something on screen, most important first.
    let mut chosen: Vec<(f32, u32, Lamp, f32)> = lamps
        .filter_map(|(seed, lamp)| {
            let intensity = lamp.intensity_at(time, seed);
            if intensity <= 0.0 {
                return None;
            }
            let gap = (lamp.position.distance(eye) - lamp.range).max(0.0);
            // Fade out before the grid ends instead of popping.
            let fade = 1.0 - ((gap - reach * 0.8) / (reach * 0.2)).clamp(0.0, 1.0);
            if fade <= 0.0 || !visible(lamp.position, lamp.range) {
                return None;
            }
            let score = intensity * lamp.range / (1.0 + gap * gap);
            Some((score, seed, lamp, intensity * fade))
        })
        .collect();
    chosen.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    chosen.truncate(MAX_LAMPS);

    let mut frame = LampFrame {
        origin0,
        origin1,
        ..Default::default()
    };
    let mut shadowed = 0;
    for (_, _, lamp, intensity) in &chosen {
        let mut layer = -1.0;
        if lamp.shadows && shadowed < shadow_budget {
            let need = match lamp.shape {
                LampShape::Point => 6,
                LampShape::Spot { .. } => 1,
            };
            if frame.shadow_views.len() + need <= SHADOW_LAYERS {
                layer = frame.shadow_views.len() as f32;
                shadowed += 1;
                match lamp.shape {
                    LampShape::Point => {
                        frame
                            .shadow_views
                            .extend(cube_views(lamp.position, lamp.range));
                    }
                    LampShape::Spot {
                        direction,
                        angle_deg,
                        ..
                    } => frame.shadow_views.push(spot_view(
                        lamp.position,
                        direction,
                        angle_deg,
                        lamp.range,
                    )),
                }
            }
        }
        let c = lamp.color.to_vec3() * *intensity;
        let (dir_cos, extra) = match lamp.shape {
            LampShape::Point => ([0.0, -1.0, 0.0, -2.0], [-2.0, 0.0, 0.0, 0.0]),
            LampShape::Spot {
                direction,
                angle_deg,
                softness,
            } => {
                let d = direction.normalize();
                let outer = (angle_deg * 0.5).to_radians();
                let inner = outer * (1.0 - softness.clamp(0.0, 1.0));
                (
                    [d.x, d.y, d.z, outer.cos()],
                    [inner.cos().max(outer.cos() + 1e-4), 1.0, 0.0, 0.0],
                )
            }
        };
        frame.lamps.push(GpuLamp {
            pos_range: [
                lamp.position.x,
                lamp.position.y,
                lamp.position.z,
                lamp.range,
            ],
            color: [c.x, c.y, c.z, layer],
            dir_cos,
            extra,
        });
    }

    // Cell lists: count, prefix-sum, fill.
    let level_cells = |origin: Vec3, cell: f32, lamp: &GpuLamp| {
        let p = Vec3::from_slice(&lamp.pos_range[..3]);
        let r = lamp.pos_range[3];
        let lo = ((p - Vec3::splat(r) - origin) / cell)
            .floor()
            .as_ivec3()
            .max(IVec3::ZERO);
        let hi = ((p + Vec3::splat(r) - origin) / cell)
            .floor()
            .as_ivec3()
            .min(GRID - IVec3::ONE);
        (lo, hi)
    };
    // Level-1 cells wholly inside level 0 are never read.
    let l0_lo = ((origin0 - origin1) / CELL1_M).ceil().as_ivec3();
    let l0_hi = ((origin0 + GRID.as_vec3() * CELL0_M - origin1) / CELL1_M)
        .floor()
        .as_ivec3()
        - IVec3::ONE;
    let covered = |g: IVec3| g.cmpge(l0_lo).all() && g.cmple(l0_hi).all();
    let mut counts = vec![0u32; 2 * CELLS];
    let each = |f: &mut dyn FnMut(usize, usize)| {
        for (i, lamp) in frame.lamps.iter().enumerate() {
            for (level, origin, cell) in [(0, origin0, CELL0_M), (1, origin1, CELL1_M)] {
                let (lo, hi) = level_cells(origin, cell, lamp);
                for y in lo.y..=hi.y {
                    for z in lo.z..=hi.z {
                        for x in lo.x..=hi.x {
                            let g = IVec3::new(x, y, z);
                            if level == 1 && covered(g) {
                                continue;
                            }
                            f(level * CELLS + cell_index(g), i);
                        }
                    }
                }
            }
        }
    };
    each(&mut |c, _| counts[c] = (counts[c] + 1).min(255));
    let mut cells = vec![0u32; 2 * CELLS];
    let mut offset = (2 * CELLS) as u32;
    let limit = (2 * CELLS + MAX_INDICES) as u32;
    for (c, n) in counts.iter().enumerate() {
        let n = (*n).min(limit.saturating_sub(offset));
        cells[c] = (offset << 8) | n;
        offset += n;
    }
    cells.resize(offset as usize, 0);
    let mut fill = vec![0u32; 2 * CELLS];
    each(&mut |c, i| {
        let n = cells[c] & 255;
        if fill[c] < n {
            let at = (cells[c] >> 8) + fill[c];
            cells[at as usize] = i as u32;
            fill[c] += 1;
        }
    });
    frame.cells = cells;
    frame
}

/// WGSL: lamp lighting, after the bindings `lamps`, `lamp_cells`,
/// `lamp_views`, `lamp_shadow_map` and `shadow_samp` and the shadow uniform
/// fields `lamp_origin0` (w = lamp count) and `lamp_origin1`.
pub const LAMP_EVAL_WGSL: &str = r#"
struct GpuLamp {
    pos_range: vec4<f32>,
    color: vec4<f32>,
    dir_cos: vec4<f32>,
    extra: vec4<f32>,
};

const LAMP_GRID: vec3<i32> = vec3<i32>(32, 16, 32);
const LAMP_CELLS: u32 = 16384u;
const LAMP_SHADOW_TEXEL: f32 = 1.0 / 512.0;

fn lamp_cell(p: vec3<f32>) -> u32 {
    let g0 = vec3<i32>(floor((p - shadow.lamp_origin0.xyz) / 8.0));
    if all(g0 >= vec3<i32>(0)) && all(g0 < LAMP_GRID) {
        return u32((g0.y * LAMP_GRID.z + g0.z) * LAMP_GRID.x + g0.x);
    }
    let g1 = vec3<i32>(floor((p - shadow.lamp_origin1.xyz) / 32.0));
    if all(g1 >= vec3<i32>(0)) && all(g1 < LAMP_GRID) {
        return LAMP_CELLS + u32((g1.y * LAMP_GRID.z + g1.z) * LAMP_GRID.x + g1.x);
    }
    return 0xffffffffu;
}

// How much of a lamp reaches `p`: 1 lit, 0 in shadow.
fn lamp_visibility(first: i32, spot: bool, lamp_p: vec3<f32>, p: vec3<f32>, n: vec3<f32>) -> f32 {
    let d = p - lamp_p;
    var layer = first;
    if !spot {
        let a = abs(d);
        if a.x >= a.y && a.x >= a.z {
            layer += select(1, 0, d.x > 0.0);
        } else if a.y >= a.z {
            layer += select(3, 2, d.y > 0.0);
        } else {
            layer += select(5, 4, d.z > 0.0);
        }
    }
    let dist = length(d);
    // Texels grow with distance, and so must the offset off the surface.
    let bias_p = p + n * (0.015 + dist * 0.012);
    let clip = lamp_views[layer] * vec4<f32>(bias_p, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 {
        return 1.0;
    }
    let z = ndc.z - 0.00002;
    var s = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2<f32>(f32(x), f32(y)) * LAMP_SHADOW_TEXEL * 1.25;
            s += textureSampleCompareLevel(lamp_shadow_map, shadow_samp, uv + o, layer, z);
        }
    }
    return s / 9.0;
}

// Light from every lamp reaching `p`, shaded like the sun (GGX specular).
fn lamp_light(base: vec3<f32>, n: vec3<f32>, v: vec3<f32>, p: vec3<f32>,
        roughness_in: f32, metallic_in: f32) -> vec3<f32> {
    if shadow.lamp_origin0.w < 0.5 {
        return vec3<f32>(0.0);
    }
    let cell = lamp_cell(p);
    if cell == 0xffffffffu {
        return vec3<f32>(0.0);
    }
    let head = lamp_cells[cell];
    let count = head & 255u;
    let offset = head >> 8u;
    let roughness = clamp(roughness_in, 0.045, 1.0);
    let metallic = clamp(metallic_in, 0.0, 1.0);
    let diffuse_color = base * (1.0 - metallic);
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let nv = max(dot(n, v), 1e-4);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < count; i++) {
        let lamp = lamps[lamp_cells[offset + i]];
        let to = lamp.pos_range.xyz - p;
        let d2 = dot(to, to);
        let r = lamp.pos_range.w;
        if d2 >= r * r {
            continue;
        }
        let d = sqrt(d2);
        let l = to / max(d, 1e-4);
        let nl = dot(n, l);
        if nl <= 0.0 {
            continue;
        }
        // Inverse square, softened near the bulb, windowed to zero at range.
        let x = d2 / (r * r);
        let window = (1.0 - x * x) * (1.0 - x * x);
        var a = window / (d2 + 0.25);
        let spot = lamp.extra.y > 0.5;
        if spot {
            a *= smoothstep(lamp.dir_cos.w, lamp.extra.x, dot(-l, lamp.dir_cos.xyz));
        }
        if a <= 1e-4 {
            continue;
        }
        if lamp.color.w >= 0.0 {
            a *= lamp_visibility(i32(lamp.color.w), spot, lamp.pos_range.xyz, p, n);
            if a <= 1e-4 {
                continue;
            }
        }
        let h = normalize(l + v);
        let nh = max(dot(n, h), 0.0);
        let lh = max(dot(l, h), 0.0);
        let ar = roughness * roughness;
        let a2 = ar * ar;
        let dd = nh * nh * (a2 - 1.0) + 1.0;
        let dist = a2 / (PI * dd * dd);
        let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        let g = (nv / (nv * (1.0 - k) + k)) * (nl / (nl * (1.0 - k) + k));
        let f = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - lh, 5.0);
        let spec = dist * g * f / (4.0 * nv) * PI;
        sum += (diffuse_color * nl + spec) * lamp.color.rgb * a;
    }
    return sum;
}
"#;

/// GPU side: lamp list, cell grid, shadow views and the shadow texture.
pub struct LampResources {
    pub lamps: wgpu::Buffer,
    pub cells: wgpu::Buffer,
    pub views: wgpu::Buffer,
    #[allow(dead_code)]
    texture: wgpu::Texture,
    pub array_view: wgpu::TextureView,
    pub layer_views: Vec<wgpu::TextureView>,
    /// One uniform buffer and bind group per shadow layer, in the cascade
    /// layout, so the sun's caster pipelines draw lamp shadows too.
    pub layer_bufs: Vec<wgpu::Buffer>,
    pub layer_binds: Vec<wgpu::BindGroup>,
    /// This frame's shadow views (one per used layer).
    pub frame_views: Vec<Mat4>,
}

impl LampResources {
    pub fn new(
        device: &wgpu::Device,
        cascade_layout: &wgpu::BindGroupLayout,
        element_states: &wgpu::Buffer,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let storage = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let lamps = storage("lamps", MAX_LAMPS * std::mem::size_of::<GpuLamp>());
        let cells = storage("lamp-cells", (2 * CELLS + MAX_INDICES) * 4);
        let views = storage("lamp-shadow-views", SHADOW_LAYERS * 64);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lamp-shadow-maps"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: SHADOW_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("lamp-shadow-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let layer_views = (0..SHADOW_LAYERS as u32)
            .map(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("lamp-shadow-layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let layer_bufs: Vec<wgpu::Buffer> = (0..SHADOW_LAYERS)
            .map(|_| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("lamp-shadow-layer-uniform"),
                    contents: &[0u8; 80],
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            })
            .collect();
        let layer_binds = layer_bufs
            .iter()
            .map(|buf| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("lamp-shadow-layer-bind"),
                    layout: cascade_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: element_states.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        Self {
            lamps,
            cells,
            views,
            texture,
            array_view,
            layer_views,
            layer_bufs,
            layer_binds,
            frame_views: Vec::new(),
        }
    }

    /// Upload a frame; returns the lamp count.
    pub fn upload(&mut self, queue: &wgpu::Queue, frame: &LampFrame, wind: [f32; 4]) -> usize {
        if !frame.lamps.is_empty() {
            queue.write_buffer(&self.lamps, 0, bytemuck::cast_slice(&frame.lamps));
            queue.write_buffer(&self.cells, 0, bytemuck::cast_slice(&frame.cells));
        }
        let mats: Vec<[[f32; 4]; 4]> = frame
            .shadow_views
            .iter()
            .map(|m| m.to_cols_array_2d())
            .collect();
        if !mats.is_empty() {
            queue.write_buffer(&self.views, 0, bytemuck::cast_slice(&mats));
        }
        for (buf, m) in self.layer_bufs.iter().zip(&frame.shadow_views) {
            let mut bytes = Vec::with_capacity(80);
            bytes.extend_from_slice(bytemuck::bytes_of(&m.to_cols_array_2d()));
            bytes.extend_from_slice(bytemuck::bytes_of(&wind));
            queue.write_buffer(buf, 0, &bytes);
        }
        self.frame_views.clone_from(&frame.shadow_views);
        frame.lamps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;

    fn read_cell(frame: &LampFrame, p: Vec3) -> Vec<u32> {
        let g0 = ((p - frame.origin0) / CELL0_M).floor().as_ivec3();
        let c = if g0.cmpge(IVec3::ZERO).all() && g0.cmplt(GRID).all() {
            cell_index(g0)
        } else {
            let g1 = ((p - frame.origin1) / CELL1_M).floor().as_ivec3();
            CELLS + cell_index(g1)
        };
        let head = frame.cells[c];
        let (off, n) = ((head >> 8) as usize, (head & 255) as usize);
        frame.cells[off..off + n].to_vec()
    }

    #[test]
    fn each_cell_lists_the_lamps_that_reach_it() {
        let lamps = vec![
            (
                1,
                Lamp::point(Vec3::new(10.0, 1.0, 0.0), Color::WHITE, 5.0, 6.0),
            ),
            (
                2,
                Lamp::point(Vec3::new(-30.0, 1.0, 0.0), Color::WHITE, 5.0, 6.0),
            ),
            (
                3,
                Lamp::point(Vec3::new(300.0, 1.0, 0.0), Color::WHITE, 50.0, 40.0),
            ),
        ];
        let f = build_frame(lamps.into_iter(), Vec3::ZERO, 0.0, 4, |_, _| true);
        assert_eq!(f.lamps.len(), 3);
        let near_a = read_cell(&f, Vec3::new(11.0, 1.0, 0.0));
        let near_far = read_cell(&f, Vec3::new(300.0, 1.0, 0.0));
        let a = f.lamps.iter().position(|l| l.pos_range[0] == 10.0).unwrap() as u32;
        let far = f
            .lamps
            .iter()
            .position(|l| l.pos_range[0] == 300.0)
            .unwrap() as u32;
        assert!(near_a.contains(&a) && !near_a.contains(&far));
        assert!(near_far.contains(&far));
        assert!(read_cell(&f, Vec3::new(60.0, 1.0, 60.0)).is_empty());
    }

    #[test]
    fn shadows_go_to_the_most_important_lamps_within_budget() {
        let lamps = (0..6).map(|i| {
            (
                i,
                Lamp::point(Vec3::new(i as f32 * 20.0, 1.0, 0.0), Color::WHITE, 5.0, 8.0)
                    .shadows(true),
            )
        });
        let f = build_frame(lamps, Vec3::ZERO, 0.0, 2, |_, _| true);
        assert_eq!(f.shadow_views.len(), 12);
        let shadowed: Vec<f32> = f
            .lamps
            .iter()
            .filter(|l| l.color[3] >= 0.0)
            .map(|l| l.pos_range[0])
            .collect();
        assert_eq!(shadowed, vec![0.0, 20.0]);
    }

    #[test]
    fn hidden_and_dark_lamps_are_skipped() {
        let lamps = vec![
            (
                1,
                Lamp::point(Vec3::new(5.0, 0.0, 0.0), Color::WHITE, 0.0, 6.0),
            ),
            (
                2,
                Lamp::point(Vec3::new(-5.0, 0.0, 0.0), Color::WHITE, 3.0, 6.0),
            ),
        ];
        let f = build_frame(lamps.into_iter(), Vec3::ZERO, 0.0, 4, |p, _| p.x > 0.0);
        assert!(f.lamps.is_empty());
    }

    #[test]
    fn cube_faces_look_along_their_axes() {
        let views = cube_views(Vec3::ZERO, 10.0);
        for (i, dir) in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
            .into_iter()
            .enumerate()
        {
            let c = views[i] * (dir * 5.0).extend(1.0);
            let ndc = c.truncate() / c.w;
            assert!(ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4 && ndc.z > 0.0 && ndc.z < 1.0);
        }
    }
}
