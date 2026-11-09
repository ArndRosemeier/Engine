//! Occlusion culling: skip meshes hidden behind other things.
//!
//! The frustum test keeps everything inside the camera's view, so a street
//! wall still lets every block behind it through, a kilometre deep. This pass
//! drops those draws on the GPU, in two phases per frame:
//!
//! 1. The depth prepass draws the meshes that were visible a few frames ago
//!    (and everything that is never tested: land, terrain, characters).
//! 2. A depth pyramid ("Hi-Z", each level keeping the farthest depth of four
//!    texels below) is built from that depth. A compute pass projects every
//!    candidate's box onto the screen and compares its nearest point with the
//!    farthest depth under it. It writes one indirect draw per candidate:
//!    instance count `n` when any part may show, 0 when it is hidden.
//! 3. The prepass then draws the candidates that were not in phase 1 through
//!    those draws, and the main pass draws every candidate through them.
//!
//! Why nothing pops: the pyramid holds only depths of things really drawn
//! this frame with this frame's camera, and a mesh is dropped only when its
//! whole box lies behind them. Last frame's visibility only chooses what goes
//! into phase 1, so a stale answer costs speed, never a missing mesh. That
//! answer comes back through a small ring of read buffers that are mapped
//! without waiting; when none is free that frame simply reads nothing back.

use super::frustum::Aabb;
use super::instance_cull::{DrawIndexedArgs, INDIRECT_STRIDE};
use crate::world::EntityId;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

const GROUP: u32 = 64;
const PYRAMID_GROUP: u32 = 8;
/// Two draws per candidate: its opaque range, then its translucent range.
pub const ARGS_PER_CANDIDATE: u64 = 2;
/// A box must lie this far (relative depth) behind the pyramid to be dropped,
/// so rounding in the depth buffer can never hide a mesh standing right at a wall.
const DEPTH_MARGIN: f32 = 1.0 - 1.0 / 2048.0;
const READBACK_RING: usize = 3;

const IDLE: u8 = 0;
const MAPPING: u8 = 1;
const MAPPED: u8 = 2;

/// One mesh to test, as the cull shader reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Candidate {
    lo: [f32; 3],
    opaque_index_count: u32,
    hi: [f32; 3],
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

/// One mesh to test: its box, index counts, instances and where its
/// indices and vertices start in the shared geometry arena.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CandidateMesh {
    pub id: EntityId,
    pub aabb: Aabb,
    pub opaque_index_count: u32,
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    /// Its first slot in the instance arena page, for the packed draws.
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CullParams {
    view_proj: [[f32; 4]; 4],
    screen: [f32; 2],
    levels: u32,
    count: u32,
    margin: f32,
    _pad: [f32; 3],
}

const PYRAMID_FIRST: &str = r#"
@group(0) @binding(0) var depth_tex: DEPTH_TYPE;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let last = vec2<i32>(textureDimensions(depth_tex)) - vec2<i32>(1);
    // Depth is reversed (near 1, far 0): the farthest is the smallest.
    var far = 1.0;
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let p = min(vec2<i32>(id.xy) * 2 + vec2<i32>(dx, dy), last);
            SAMPLE_LOOP
        }
    }
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(far, 0.0, 0.0, 0.0));
}
"#;

const PYRAMID_DOWN: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let src_size = textureDimensions(src);
    let last = vec2<i32>(src_size) - vec2<i32>(1);
    // Levels halve rounding down: the last texel of an odd row or column
    // also covers the one left over.
    let reach = vec2<i32>(
        select(1, 2, id.x + 1u == size.x && (src_size.x & 1u) == 1u),
        select(1, 2, id.y + 1u == size.y && (src_size.y & 1u) == 1u),
    );
    var far = 1.0;
    for (var dy = 0; dy <= reach.y; dy++) {
        for (var dx = 0; dx <= reach.x; dx++) {
            let p = min(vec2<i32>(id.xy) * 2 + vec2<i32>(dx, dy), last);
            far = min(far, textureLoad(src, p, 0).r);
        }
    }
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(far, 0.0, 0.0, 0.0));
}
"#;

/// The box test, shared with the per-instance cull (`instance_cull.rs`).
/// The including shader binds `params: Params` and `pyramid`.
pub(crate) const BOX_TEST_WGSL: &str = r#"
struct Params {
    view_proj: mat4x4<f32>,
    screen: vec2<f32>,
    levels: u32,
    count: u32,
    margin: f32,
};

// Same test as `box_may_show` in occlusion.rs (the CPU copy the tests run).
fn may_show(lo: vec3<f32>, hi: vec3<f32>) -> bool {
    var mn = vec2<f32>(1e30);
    var mx = vec2<f32>(-1e30);
    var nearest = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3<f32>(
            select(lo.x, hi.x, (i & 1u) != 0u),
            select(lo.y, hi.y, (i & 2u) != 0u),
            select(lo.z, hi.z, (i & 4u) != 0u),
        );
        let clip = occ.view_proj * vec4<f32>(corner, 1.0);
        // A corner at or behind the eye: the box reaches the camera.
        if clip.w <= 1e-5 {
            return true;
        }
        let ndc = clip.xyz / clip.w;
        mn = min(mn, ndc.xy);
        mx = max(mx, ndc.xy);
        nearest = max(nearest, ndc.z);
    }
    if nearest >= 1.0 {
        return true;
    }
    // Wholly beyond the far plane or beside the screen: nothing can show.
    if nearest <= 0.0 || mx.x < -1.0 || mn.x > 1.0 || mx.y < -1.0 || mn.y > 1.0 {
        return false;
    }
    // Pixel rectangle (y down), one pixel wider all round.
    let a = clamp(vec2<f32>(mn.x, -mx.y) * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let b = clamp(vec2<f32>(mx.x, -mn.y) * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let last_px = vec2<i32>(occ.screen) - vec2<i32>(1);
    let p0 = clamp(vec2<i32>(floor(a * occ.screen)) - vec2<i32>(1), vec2<i32>(0), last_px);
    let p1 = clamp(vec2<i32>(floor(b * occ.screen)) + vec2<i32>(1), vec2<i32>(0), last_px);
    // The level where the rectangle spans at most 4 x 4 texels. Level l
    // texels cover 2^(l+1) pixels (level 0 is half the screen).
    var level = 0u;
    loop {
        let t0 = p0 >> vec2<u32>(level + 1u);
        let t1 = p1 >> vec2<u32>(level + 1u);
        if (t1.x - t0.x <= 3 && t1.y - t0.y <= 3) || level + 1u >= occ.levels {
            break;
        }
        level += 1u;
    }
    // Level sizes from level 0 (half the screen), halved rounding down. Not
    // textureDimensions: lavapipe answers it for the next level down.
    let size0 = vec2<u32>(occ.screen) / 2u;
    let top = vec2<i32>(max(size0 >> vec2<u32>(level), vec2<u32>(1u))) - vec2<i32>(1);
    let t0 = min(p0 >> vec2<u32>(level + 1u), top);
    let t1 = min(p1 >> vec2<u32>(level + 1u), top);
    var far = 1.0;
    for (var y = t0.y; y <= t1.y; y++) {
        for (var x = t0.x; x <= t1.x; x++) {
            far = min(far, textureLoad(pyramid, vec2<i32>(x, y), i32(level)).r);
        }
    }
    return nearest >= far * occ.margin;
}

"#;

const CULL: &str = r#"

struct Candidate {
    lo: vec3<f32>,
    opaque_index_count: u32,
    hi: vec3<f32>,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> occ: Params;
@group(0) @binding(1) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(2) var<storage, read_write> draws: array<DrawArgs>;
@group(0) @binding(3) var<storage, read_write> visible: array<u32>;
@group(0) @binding(4) var pyramid: texture_2d<f32>;

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= occ.count {
        return;
    }
    let c = candidates[i];
    let show = may_show(c.lo, c.hi);
    let n = select(0u, c.instance_count, show);
    draws[2u * i] = DrawArgs(c.opaque_index_count, n, c.first_index, c.base_vertex, 0u);
    // The packed copy after both per-candidate draws, for multi-draws that
    // bind the whole instance page.
    let packed = arrayLength(&draws) / 3u * 2u;
    draws[packed + i] = DrawArgs(c.opaque_index_count, n, c.first_index, c.base_vertex, c.first_instance);
    draws[2u * i + 1u] = DrawArgs(
        c.index_count - c.opaque_index_count,
        n,
        c.first_index + c.opaque_index_count,
        c.base_vertex,
        0u,
    );
    visible[i] = select(0u, 1u, show);
}
"#;

/// The depth pyramid for one screen size.
struct Pyramid {
    texture: wgpu::Texture,
    full_view: wgpu::TextureView,
    sizes: Vec<(u32, u32)>,
    binds: Vec<wgpu::BindGroup>,
}

struct Readback {
    buffer: wgpu::Buffer,
    capacity: usize,
    ids: Vec<EntityId>,
    /// Frame number the answer belongs to: the newest mapped one wins.
    frame: u64,
    state: Arc<AtomicU8>,
}

/// What this frame's occlusion pass set up, for the draws that follow.
pub struct OcclusionFrame {
    /// Draw arguments, [`ARGS_PER_CANDIDATE`] per candidate.
    pub args: wgpu::Buffer,
    /// Candidate -> its index in `args`.
    pub slots: HashMap<EntityId, u32>,
    /// Candidates the prepass draws in phase 1 (seen lately).
    pub first: HashSet<EntityId>,
    /// Candidates `args` has room for.
    pub capacity: u32,
}

impl OcclusionFrame {
    /// Byte offset of a candidate's opaque (0) or translucent (`INDIRECT_STRIDE`) draw.
    pub fn offset(slot: u32, range_offset: u64) -> u64 {
        u64::from(slot) * ARGS_PER_CANDIDATE * INDIRECT_STRIDE + range_offset
    }

    /// Byte offset of a candidate's opaque draw in the packed rows: one
    /// draw per candidate, tightly packed in slot order, with the instances
    /// at their place in the arena page (for `multi_draw_indexed_indirect`).
    pub fn packed_offset(&self, slot: u32) -> u64 {
        (u64::from(self.capacity) * ARGS_PER_CANDIDATE + u64::from(slot)) * INDIRECT_STRIDE
    }
}

/// Fewer opaque meshes in view than this: nothing worth testing.
const AUTO_MIN_CANDIDATES: usize = 32;
/// Frames a try lasts (and between re-checks while on).
const AUTO_WINDOW: u32 = 30;
/// The first frames of a try are skipped: their answers come back late.
const AUTO_SETTLE: u32 = 4;
/// Kept on while at least this share of the tested meshes is hidden.
const AUTO_KEEP_SHARE: f32 = 0.15;
/// Frames before the first retry after a try that didn't pay; doubles up to
/// `AUTO_MAX_WAIT` with every try that doesn't.
const AUTO_FIRST_WAIT: u32 = 120;
const AUTO_MAX_WAIT: u32 = 1920;

/// Chooses, frame by frame, whether `OcclusionMode::Auto` culls: it tries,
/// measures the share of tested meshes found hidden, and keeps culling only
/// while that share is worth the test.
#[derive(Debug, Clone)]
pub(crate) struct OcclusionAuto {
    active: bool,
    /// Frames in the current state.
    frames: u32,
    /// Frames to wait (off) before the next try.
    wait: u32,
    tested: u64,
    hidden: u64,
}

impl Default for OcclusionAuto {
    fn default() -> Self {
        Self {
            active: false,
            frames: 0,
            wait: 0,
            tested: 0,
            hidden: 0,
        }
    }
}

impl OcclusionAuto {
    /// Whether to cull this frame. `candidates` is how many opaque meshes
    /// the last frame drew; `counts` the newest answer read back.
    pub(crate) fn decide(&mut self, candidates: usize, counts: OcclusionCounts) -> bool {
        if candidates < AUTO_MIN_CANDIDATES {
            // Nothing to gain; try as soon as there is.
            *self = Self {
                wait: self.wait,
                ..Self::default()
            };
            return false;
        }
        self.frames += 1;
        if !self.active {
            if self.frames > self.wait {
                self.active = true;
                self.frames = 0;
                self.tested = 0;
                self.hidden = 0;
            }
            return self.active;
        }
        if self.frames > AUTO_SETTLE {
            self.tested += u64::from(counts.tested);
            self.hidden += u64::from(counts.hidden);
        }
        if self.frames >= AUTO_WINDOW {
            let share = if self.tested == 0 {
                0.0
            } else {
                self.hidden as f32 / self.tested as f32
            };
            self.frames = 0;
            self.tested = 0;
            self.hidden = 0;
            if share < AUTO_KEEP_SHARE {
                self.active = false;
                self.wait = if self.wait == 0 {
                    AUTO_FIRST_WAIT
                } else {
                    (self.wait * 2).min(AUTO_MAX_WAIT)
                };
            } else {
                self.wait = 0;
            }
        }
        self.active
    }
}

/// Numbers from the newest read-back frame, for frame stats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OcclusionCounts {
    pub tested: u32,
    pub hidden: u32,
}

pub struct Occlusion {
    first_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    cull_layout: wgpu::BindGroupLayout,
    first_pipeline: wgpu::ComputePipeline,
    down_pipeline: wgpu::ComputePipeline,
    cull_pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    candidates: wgpu::Buffer,
    args: wgpu::Buffer,
    visible: wgpu::Buffer,
    capacity: usize,
    cull_bind: Option<wgpu::BindGroup>,
    pyramid: Option<Pyramid>,
    ring: Vec<Readback>,
    /// Candidates seen in the newest read-back frame.
    seen: HashSet<EntityId>,
    /// The ring slot this frame copied its answer into, mapped after submit.
    pending: Option<usize>,
    counts: OcclusionCounts,
    scratch: Vec<Candidate>,
    frame: u64,
    seen_frame: u64,
    /// `ENGINE_OCCLUSION_CHECK=1`: what to compare after this frame.
    check: Option<Check>,
}

/// One frame's GPU answer, read back in full and compared with the CPU copy
/// of the test (a self-check for drivers, like the renderer audit).
struct Check {
    view_proj: Mat4,
    boxes: Vec<Aabb>,
    levels: Vec<(u32, u32, u32, wgpu::Buffer)>,
    visible: wgpu::Buffer,
}

fn check_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_OCCLUSION_CHECK").is_ok_and(|v| v == "1"))
}

impl Occlusion {
    pub fn new(device: &wgpu::Device, samples: u32) -> Self {
        let multisampled = samples > 1;
        let storage_out = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let first_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion-pyramid-first"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                },
                storage_out,
            ],
        });
        let down_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion-pyramid-down"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_out,
            ],
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("occlusion-cull"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CullParams>() as u64
                        ),
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let (depth_type, sample_loop) = if multisampled {
            (
                "texture_depth_multisampled_2d",
                "for (var s = 0; s < i32(textureNumSamples(depth_tex)); s++) { far = min(far, textureLoad(depth_tex, p, s)); }",
            )
        } else {
            (
                "texture_depth_2d",
                "far = min(far, textureLoad(depth_tex, p, 0));",
            )
        };
        let first_src = PYRAMID_FIRST
            .replace("DEPTH_TYPE", depth_type)
            .replace("SAMPLE_LOOP", sample_loop);
        let pipeline = |label: &str, layout: &wgpu::BindGroupLayout, src: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            })
        };
        let first_pipeline = pipeline("occlusion-pyramid-first", &first_layout, &first_src, "main");
        let down_pipeline = pipeline("occlusion-pyramid-down", &down_layout, PYRAMID_DOWN, "main");
        let cull_src = format!("{BOX_TEST_WGSL}{CULL}");
        let cull_pipeline = pipeline("occlusion-cull", &cull_layout, &cull_src, "cull");
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occlusion-params"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (candidates, args, visible) = candidate_buffers(device, 1);
        Self {
            first_layout,
            down_layout,
            cull_layout,
            first_pipeline,
            down_pipeline,
            cull_pipeline,
            params,
            candidates,
            args,
            visible,
            capacity: 1,
            cull_bind: None,
            pyramid: None,
            ring: Vec::new(),
            seen: HashSet::new(),
            pending: None,
            counts: OcclusionCounts::default(),
            scratch: Vec::new(),
            frame: 0,
            seen_frame: 0,
            check: None,
        }
    }

    /// Follow the scene depth texture (new size, new texture).
    pub fn resize(&mut self, device: &wgpu::Device, depth: &wgpu::Texture) {
        let (width, height) = (depth.width().max(1), depth.height().max(1));
        let sizes = pyramid_sizes(width, height);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("occlusion-pyramid"),
            size: wgpu::Extent3d {
                width: sizes[0].0,
                height: sizes[0].1,
                depth_or_array_layers: 1,
            },
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let level = |l: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("occlusion-pyramid-level"),
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("occlusion-depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let mut binds = Vec::with_capacity(sizes.len());
        binds.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("occlusion-pyramid-first"),
            layout: &self.first_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&level(0)),
                },
            ],
        }));
        for l in 1..sizes.len() as u32 {
            binds.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("occlusion-pyramid-down"),
                layout: &self.down_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&level(l - 1)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&level(l)),
                    },
                ],
            }));
        }
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.pyramid = Some(Pyramid {
            texture,
            full_view,
            sizes,
            binds,
        });
        self.cull_bind = None;
    }

    /// Take in the newest finished read-back (never waits for the GPU).
    pub fn collect(&mut self, device: &wgpu::Device) {
        device
            .poll(wgpu::PollType::Poll)
            .expect("occlusion read-back poll failed");
        for entry in &mut self.ring {
            if entry.state.load(Ordering::Acquire) != MAPPED {
                continue;
            }
            if entry.frame > self.seen_frame {
                self.seen_frame = entry.frame;
                let mapped = entry.buffer.slice(..).get_mapped_range();
                let flags: &[u32] = bytemuck::cast_slice(&mapped);
                self.seen.clear();
                let mut hidden = 0u32;
                for (id, flag) in entry.ids.iter().zip(flags) {
                    if *flag != 0 {
                        self.seen.insert(*id);
                    } else {
                        hidden += 1;
                    }
                }
                self.counts = OcclusionCounts {
                    tested: u32::try_from(entry.ids.len()).expect("candidate count exceeds u32"),
                    hidden,
                };
            }
            entry.buffer.unmap();
            entry.state.store(IDLE, Ordering::Release);
        }
    }

    /// The test's uniform (written by [`Self::encode`]) and the pyramid, for
    /// the per-instance cull.
    pub fn test_resources(&self) -> (&wgpu::Buffer, &wgpu::TextureView) {
        let pyramid = self
            .pyramid
            .as_ref()
            .expect("occlusion pyramid follows the depth texture from the start");
        (&self.params, &pyramid.full_view)
    }

    pub fn counts(&self) -> OcclusionCounts {
        self.counts
    }

    /// Upload this frame's candidates (id, box, opaque and total index
    /// counts, instance count) and return the frame's draw set-up.
    pub fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidates: &[CandidateMesh],
    ) -> OcclusionFrame {
        if candidates.len() > self.capacity {
            self.capacity = candidates
                .len()
                .checked_next_power_of_two()
                .expect("occlusion candidate capacity overflow");
            (self.candidates, self.args, self.visible) = candidate_buffers(device, self.capacity);
            self.cull_bind = None;
        }
        self.scratch.clear();
        let mut slots = HashMap::with_capacity(candidates.len());
        let mut first = HashSet::new();
        for (i, c) in candidates.iter().enumerate() {
            let id = c.id;
            self.scratch.push(Candidate {
                lo: c.aabb.min.into(),
                opaque_index_count: c.opaque_index_count,
                hi: c.aabb.max.into(),
                index_count: c.index_count,
                instance_count: c.instance_count,
                first_index: c.first_index,
                base_vertex: c.base_vertex,
                first_instance: c.first_instance,
            });
            slots.insert(id, u32::try_from(i).expect("candidate index exceeds u32"));
            if self.seen.contains(&id) {
                first.insert(id);
            }
        }
        if !self.scratch.is_empty() {
            queue.write_buffer(&self.candidates, 0, bytemuck::cast_slice(&self.scratch));
        }
        OcclusionFrame {
            args: self.args.clone(),
            slots,
            first,
            capacity: u32::try_from(self.capacity).expect("candidate capacity exceeds u32"),
        }
    }

    /// Build the pyramid from the phase-1 depth and test every candidate.
    /// `ids` is the candidate order given to [`Self::begin`].
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_proj: Mat4,
        ids: &[EntityId],
    ) {
        let pyramid = self
            .pyramid
            .as_ref()
            .expect("occlusion pyramid follows the depth texture from the start");
        for (l, bind) in pyramid.binds.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("occlusion-pyramid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(if l == 0 {
                &self.first_pipeline
            } else {
                &self.down_pipeline
            });
            pass.set_bind_group(0, bind, &[]);
            let (w, h) = pyramid.sizes[l];
            pass.dispatch_workgroups(w.div_ceil(PYRAMID_GROUP), h.div_ceil(PYRAMID_GROUP), 1);
        }
        let count = u32::try_from(ids.len()).expect("candidate count exceeds u32");
        let params = CullParams {
            view_proj: view_proj.to_cols_array_2d(),
            screen: [
                (pyramid.sizes[0].0 * 2) as f32,
                (pyramid.sizes[0].1 * 2) as f32,
            ],
            levels: pyramid.sizes.len() as u32,
            count,
            margin: DEPTH_MARGIN,
            _pad: [0.0; 3],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        // Written even with no candidates: the instance cull reads it too.
        if ids.is_empty() {
            return;
        }
        if self.cull_bind.is_none() {
            self.cull_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("occlusion-cull"),
                layout: &self.cull_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.candidates.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.args.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.visible.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&pyramid.full_view),
                    },
                ],
            }));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("occlusion-cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.cull_pipeline);
        pass.set_bind_group(0, self.cull_bind.as_ref(), &[]);
        pass.dispatch_workgroups(count.div_ceil(GROUP), 1, 1);
        drop(pass);
        if check_enabled() {
            self.copy_check(device, encoder, view_proj, ids.len());
        }
        self.copy_answer(device, encoder, ids);
    }

    fn copy_check(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view_proj: Mat4,
        count: usize,
    ) {
        let pyramid = self.pyramid.as_ref().expect("pyramid exists");
        let mut levels = Vec::new();
        for (l, &(w, h)) in pyramid.sizes.iter().enumerate() {
            let row = (w * 4).div_ceil(256) * 256;
            let buffer = read_buffer(device, (row * h / 4) as usize);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &pyramid.texture,
                    mip_level: l as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            levels.push((w, h, row, buffer));
        }
        let visible = read_buffer(device, count);
        encoder.copy_buffer_to_buffer(&self.visible, 0, &visible, 0, flag_bytes(count));
        let boxes = self
            .scratch
            .iter()
            .map(|c| Aabb {
                min: c.lo.into(),
                max: c.hi.into(),
            })
            .collect();
        self.check = Some(Check {
            view_proj,
            boxes,
            levels,
            visible,
        });
    }

    /// Compare the GPU's answers with the CPU copy (waits for the GPU).
    fn run_check(&mut self, device: &wgpu::Device) {
        let Some(check) = self.check.take() else {
            return;
        };
        let read = |buffer: &wgpu::Buffer| -> Vec<u8> {
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.expect("check map failed"));
            device
                .poll(wgpu::PollType::Wait)
                .expect("check poll failed");
            let bytes = slice.get_mapped_range().to_vec();
            buffer.unmap();
            bytes
        };
        let levels = check
            .levels
            .iter()
            .map(|(w, h, row, buffer)| {
                let bytes = read(buffer);
                let mut texels = Vec::with_capacity((w * h) as usize);
                for y in 0..*h {
                    let start = (y * row) as usize;
                    let line: &[f32] =
                        bytemuck::cast_slice(&bytes[start..start + (*w * 4) as usize]);
                    texels.extend_from_slice(line);
                }
                (*w, *h, texels)
            })
            .collect();
        let cpu = CpuPyramid { levels };
        let flags: Vec<u32> = bytemuck::cast_slice(&read(&check.visible)).to_vec();
        let mut wrong = 0;
        for (i, aabb) in check.boxes.iter().enumerate() {
            let want = cpu.box_may_show(check.view_proj, *aabb);
            if want != (flags[i] != 0) {
                wrong += 1;
                if wrong <= 5 {
                    eprintln!(
                        "[engine] occlusion check: candidate {i} box {aabb:?} gpu={} cpu={want}",
                        flags[i]
                    );
                }
            }
        }
        let hidden = flags
            .iter()
            .take(check.boxes.len())
            .filter(|f| **f == 0)
            .count();
        eprintln!(
            "[engine] occlusion check: {} candidates, {hidden} hidden, {wrong} differ from the CPU test",
            check.boxes.len()
        );
    }

    fn copy_answer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        ids: &[EntityId],
    ) {
        let slot = match self
            .ring
            .iter()
            .position(|e| e.state.load(Ordering::Acquire) == IDLE)
        {
            Some(slot) => slot,
            None if self.ring.len() < READBACK_RING => {
                self.ring.push(Readback {
                    buffer: read_buffer(device, ids.len()),
                    capacity: ids.len(),
                    ids: Vec::new(),
                    frame: 0,
                    state: Arc::new(AtomicU8::new(IDLE)),
                });
                self.ring.len() - 1
            }
            // Every read buffer is still on its way back: skip this frame's.
            None => return,
        };
        let entry = &mut self.ring[slot];
        if entry.capacity < ids.len() {
            let capacity = ids
                .len()
                .checked_next_power_of_two()
                .expect("occlusion read-back capacity overflow");
            entry.buffer = read_buffer(device, capacity);
            entry.capacity = capacity;
        }
        self.frame += 1;
        entry.frame = self.frame;
        entry.ids.clear();
        entry.ids.extend_from_slice(ids);
        encoder.copy_buffer_to_buffer(&self.visible, 0, &entry.buffer, 0, flag_bytes(ids.len()));
        self.pending = Some(slot);
    }

    /// Start mapping this frame's answer. Call after the frame is submitted.
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        self.run_check(device);
        let Some(slot) = self.pending.take() else {
            return;
        };
        let entry = &self.ring[slot];
        entry.state.store(MAPPING, Ordering::Release);
        let state = entry.state.clone();
        let bytes = flag_bytes(entry.ids.len());
        entry
            .buffer
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                result.expect("occlusion read-back map failed");
                state.store(MAPPED, Ordering::Release);
            });
    }
}

fn flag_bytes(count: usize) -> u64 {
    (count.max(1) * std::mem::size_of::<u32>()) as u64
}

fn read_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("occlusion-read-back"),
        size: flag_bytes(capacity),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn candidate_buffers(
    device: &wgpu::Device,
    capacity: usize,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let candidates = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("occlusion-candidates"),
        size: (capacity * std::mem::size_of::<Candidate>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let args = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("occlusion-draws"),
        // Two draws per candidate, then the packed opaque rows.
        size: capacity as u64
            * (ARGS_PER_CANDIDATE + 1)
            * std::mem::size_of::<DrawIndexedArgs>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });
    let visible = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("occlusion-visible"),
        size: flag_bytes(capacity),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (candidates, args, visible)
}

/// Pyramid level sizes: level 0 is half the screen (rounded up, so every
/// pixel has a texel), then the texture's own mip sizes (halved, rounded
/// down) to 1 x 1.
fn pyramid_sizes(width: u32, height: u32) -> Vec<(u32, u32)> {
    let (w0, h0) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
    let levels = 32 - w0.max(h0).leading_zeros();
    (0..levels)
        .map(|l| ((w0 >> l).max(1), (h0 >> l).max(1)))
        .collect()
}

/// CPU copy of the shader's pyramid and test, for the tests and for
/// `ENGINE_OCCLUSION_CHECK=1`.
pub(crate) struct CpuPyramid {
    levels: Vec<(u32, u32, Vec<f32>)>,
}

impl CpuPyramid {
    #[cfg(test)]
    pub(crate) fn from_depth(width: u32, height: u32, depth: &[f32]) -> Self {
        let sizes = pyramid_sizes(width, height);
        let mut levels: Vec<(u32, u32, Vec<f32>)> = Vec::new();
        for (l, &(w, h)) in sizes.iter().enumerate() {
            let (sw, sh, src): (u32, u32, &[f32]) = if l == 0 {
                (width, height, depth)
            } else {
                let (sw, sh, ref v) = levels[l - 1];
                (sw, sh, v.as_slice())
            };
            let mut out = vec![1.0f32; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    let odd_x = l > 0 && x + 1 == w && sw % 2 == 1;
                    let odd_y = l > 0 && y + 1 == h && sh % 2 == 1;
                    let mut far = 1.0f32;
                    for dy in 0..=(1 + u32::from(odd_y)) {
                        for dx in 0..=(1 + u32::from(odd_x)) {
                            let px = (x * 2 + dx).min(sw - 1);
                            let py = (y * 2 + dy).min(sh - 1);
                            far = far.min(src[(py * sw + px) as usize]);
                        }
                    }
                    out[(y * w + x) as usize] = far;
                }
            }
            levels.push((w, h, out));
        }
        Self { levels }
    }

    /// Same steps as `may_show` in the cull shader.
    pub(crate) fn box_may_show(&self, view_proj: Mat4, aabb: Aabb) -> bool {
        let (lo, hi) = (aabb.min, aabb.max);
        let (mut mn, mut mx, mut nearest) = (Vec2::splat(1e30), Vec2::splat(-1e30), 0.0f32);
        for i in 0..8u32 {
            let corner = Vec3::new(
                if i & 1 != 0 { hi.x } else { lo.x },
                if i & 2 != 0 { hi.y } else { lo.y },
                if i & 4 != 0 { hi.z } else { lo.z },
            );
            let clip = view_proj * corner.extend(1.0);
            if clip.w <= 1e-5 {
                return true;
            }
            let ndc = clip.truncate() / clip.w;
            mn = mn.min(ndc.truncate());
            mx = mx.max(ndc.truncate());
            nearest = nearest.max(ndc.z);
        }
        if nearest >= 1.0 {
            return true;
        }
        if nearest <= 0.0 || mx.x < -1.0 || mn.x > 1.0 || mx.y < -1.0 || mn.y > 1.0 {
            return false;
        }
        let (w0, h0, _) = self.levels[0];
        let screen = Vec2::new((w0 * 2) as f32, (h0 * 2) as f32);
        let a = (Vec2::new(mn.x, -mx.y) * 0.5 + 0.5).clamp(Vec2::ZERO, Vec2::ONE);
        let b = (Vec2::new(mx.x, -mn.y) * 0.5 + 0.5).clamp(Vec2::ZERO, Vec2::ONE);
        let last = (screen.x as i32 - 1, screen.y as i32 - 1);
        let p0 = (
            ((a.x * screen.x).floor() as i32 - 1).clamp(0, last.0),
            ((a.y * screen.y).floor() as i32 - 1).clamp(0, last.1),
        );
        let p1 = (
            ((b.x * screen.x).floor() as i32 + 1).clamp(0, last.0),
            ((b.y * screen.y).floor() as i32 + 1).clamp(0, last.1),
        );
        let mut level = 0usize;
        loop {
            let s = level + 1;
            let span = ((p1.0 >> s) - (p0.0 >> s), (p1.1 >> s) - (p0.1 >> s));
            if (span.0 <= 3 && span.1 <= 3) || level + 1 >= self.levels.len() {
                break;
            }
            level += 1;
        }
        let s = level + 1;
        let (w, h, ref texels) = self.levels[level];
        let (tx, ty) = (w as i32 - 1, h as i32 - 1);
        let mut far = 1.0f32;
        for y in (p0.1 >> s).min(ty)..=(p1.1 >> s).min(ty) {
            for x in (p0.0 >> s).min(tx)..=(p1.0 >> s).min(tx) {
                far = far.min(texels[(y * w as i32 + x) as usize]);
            }
        }
        nearest >= far * DEPTH_MARGIN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;

    const W: u32 = 160;
    const H: u32 = 90;

    fn street_camera() -> (Camera, Mat4) {
        let camera = Camera::from_parts(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, -1.0),
            Vec3::Y,
            60.0,
            0.1,
            2_000.0,
        );
        let vp = camera.view_projection(W as f32 / H as f32);
        (camera, vp)
    }

    /// Reversed depth of the nearest box face along each pixel's ray, by
    /// marching: enough to stand in for the rasterised prepass.
    fn render_depth(vp: Mat4, boxes: &[Aabb]) -> Vec<f32> {
        let inv = vp.inverse();
        let mut depth = vec![0.0f32; (W * H) as usize];
        for y in 0..H {
            for x in 0..W {
                let ndc = Vec2::new(
                    (x as f32 + 0.5) / W as f32 * 2.0 - 1.0,
                    1.0 - (y as f32 + 0.5) / H as f32 * 2.0,
                );
                let near = inv.project_point3(ndc.extend(1.0));
                let far = inv.project_point3(ndc.extend(0.0001));
                let dir = (far - near).normalize();
                let mut best: Option<f32> = None;
                for b in boxes {
                    if let Some(t) = ray_box(near, dir, *b) {
                        best = Some(best.map_or(t, |bt: f32| bt.min(t)));
                    }
                }
                if let Some(t) = best {
                    let p = near + dir * t;
                    let clip = vp * p.extend(1.0);
                    depth[(y * W + x) as usize] = clip.z / clip.w;
                }
            }
        }
        depth
    }

    fn ray_box(o: Vec3, d: Vec3, b: Aabb) -> Option<f32> {
        let inv = d.recip();
        let t0 = (b.min - o) * inv;
        let t1 = (b.max - o) * inv;
        let enter = t0.min(t1).max_element();
        let exit = t0.max(t1).min_element();
        (exit >= enter.max(0.0)).then_some(enter.max(0.0))
    }

    fn cube(centre: Vec3, half: Vec3) -> Aabb {
        Aabb {
            min: centre - half,
            max: centre + half,
        }
    }

    #[test]
    fn auto_keeps_culling_where_it_hides_and_backs_off_where_it_does_not() {
        let mut auto = OcclusionAuto::default();
        let counts = |tested, hidden| OcclusionCounts { tested, hidden };
        // Too few meshes: never.
        assert!(!auto.decide(10, counts(0, 0)));
        // A city: tried at once and kept.
        let mut on = 0;
        for _ in 0..300 {
            on += usize::from(auto.decide(500, counts(500, 400)));
        }
        assert!(on >= 299, "kept on in a city: {on}");
        // An open field: switched off after one window, retried later,
        // less often each time.
        let mut runs = Vec::new();
        let mut was = true;
        for frame in 0..4000 {
            let now = auto.decide(500, counts(500, 10));
            if now && !was {
                runs.push(frame);
            }
            was = now;
        }
        assert!(runs.len() >= 2 && runs.len() < 8, "retries: {runs:?}");
        let gaps: Vec<_> = runs.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.windows(2).all(|g| g[1] >= g[0]), "backs off: {gaps:?}");
        // Back in the city: kept on again after the next try.
        let mut on_late = 0;
        for _ in 0..2500 {
            on_late += usize::from(auto.decide(500, counts(500, 400)));
        }
        assert!(on_late > 0);
        assert!(auto.decide(500, counts(500, 400)));
    }

    #[test]
    fn pyramid_sizes_halve_down_to_one() {
        assert_eq!(
            pyramid_sizes(5, 3),
            vec![(3, 2), (1, 1)],
            "level 0 rounds up so every pixel has a texel, then mips round down"
        );
        assert_eq!(pyramid_sizes(1, 1), vec![(1, 1)]);
        let full_hd = pyramid_sizes(1920, 1080);
        assert_eq!(full_hd[0], (960, 540));
        assert_eq!(full_hd.len(), 10, "as many levels as the texture allows");
        assert_eq!(*full_hd.last().unwrap(), (1, 1));
    }

    #[test]
    fn a_block_behind_a_wall_is_hidden_and_one_beside_it_is_not() {
        let (_, vp) = street_camera();
        // 12 m high, 40 m ahead: its top edge is 14 degrees over the eye.
        let wall = cube(Vec3::new(0.0, 6.0, -40.0), Vec3::new(200.0, 6.0, 1.0));
        let depth = render_depth(vp, &[wall]);
        let pyramid = CpuPyramid::from_depth(W, H, &depth);
        // Its top, 170 m off, is 6 degrees up.
        let behind = cube(Vec3::new(0.0, 10.0, -200.0), Vec3::new(30.0, 10.0, 30.0));
        assert!(!pyramid.box_may_show(vp, behind), "hidden behind the wall");
        // Taller than the wall's top edge as seen from the street.
        let tower = cube(Vec3::new(0.0, 150.0, -200.0), Vec3::new(10.0, 150.0, 10.0));
        assert!(
            pyramid.box_may_show(vp, tower),
            "a tower over the wall shows"
        );
        // The wall itself was drawn: never hidden by its own depth.
        assert!(pyramid.box_may_show(vp, wall));
        // Just in front of the wall.
        let car = cube(Vec3::new(0.0, 1.0, -38.0), Vec3::new(1.0, 1.0, 0.5));
        assert!(pyramid.box_may_show(vp, car));
    }

    #[test]
    fn a_gap_in_the_wall_lets_things_through() {
        let (_, vp) = street_camera();
        let left = cube(Vec3::new(-31.0, 15.0, -20.0), Vec3::new(30.0, 15.0, 1.0));
        let right = cube(Vec3::new(31.0, 15.0, -20.0), Vec3::new(30.0, 15.0, 1.0));
        let depth = render_depth(vp, &[left, right]);
        let pyramid = CpuPyramid::from_depth(W, H, &depth);
        // Seen through the 2 m gap only.
        let far = cube(Vec3::new(0.0, 2.0, -300.0), Vec3::new(3.0, 2.0, 3.0));
        assert!(pyramid.box_may_show(vp, far));
        // Off to the side, behind the wall.
        let hidden = cube(Vec3::new(-60.0, 2.0, -300.0), Vec3::new(3.0, 2.0, 3.0));
        assert!(!pyramid.box_may_show(vp, hidden));
    }

    #[test]
    fn boxes_at_the_camera_or_off_screen() {
        let (_, vp) = street_camera();
        let wall = cube(Vec3::new(0.0, 2.0, -10.0), Vec3::new(60.0, 2.0, 1.0));
        let pyramid = CpuPyramid::from_depth(W, H, &render_depth(vp, &[wall]));
        // Around the eye: always kept.
        let around = cube(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(3.0));
        assert!(pyramid.box_may_show(vp, around));
        // Behind the eye the test keeps a box (the frustum test drops it).
        let behind = cube(Vec3::new(0.0, 2.0, 50.0), Vec3::splat(3.0));
        assert!(pyramid.box_may_show(vp, behind));
        // Over the 4 m wall: a balloon 30 m up shows.
        let balloon = cube(Vec3::new(0.0, 30.0, -100.0), Vec3::splat(3.0));
        assert!(pyramid.box_may_show(vp, balloon));
    }

    #[test]
    fn nothing_drawn_hides_nothing() {
        let (_, vp) = street_camera();
        let pyramid = CpuPyramid::from_depth(W, H, &vec![0.0; (W * H) as usize]);
        for z in [-10.0, -100.0, -1_000.0, -1_900.0] {
            let b = cube(Vec3::new(0.0, 2.0, z), Vec3::splat(1.0));
            assert!(pyramid.box_may_show(vp, b), "z={z}");
        }
    }

    #[test]
    fn box_transform_keeps_every_corner() {
        let local = cube(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 1.0, 2.0));
        let m = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 1.5),
            glam::Quat::from_rotation_y(0.7) * glam::Quat::from_rotation_x(0.3),
            Vec3::new(10.0, -3.0, 7.0),
        );
        let world = local.transformed(m);
        for i in 0..8u32 {
            let c = Vec3::new(
                if i & 1 != 0 { local.max.x } else { local.min.x },
                if i & 2 != 0 { local.max.y } else { local.min.y },
                if i & 4 != 0 { local.max.z } else { local.min.z },
            );
            let p = m.transform_point3(c);
            assert!(p.cmpge(world.min - 1e-4).all() && p.cmple(world.max + 1e-4).all());
        }
    }
}
