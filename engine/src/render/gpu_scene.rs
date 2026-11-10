//! The GPU scene: every opaque single mesh of the live space as a row that
//! stays on the GPU from frame to frame, as professional engines keep their
//! scene. The CPU writes a row only when its mesh changes; each frame the
//! GPU tests every row (view, and the depth pyramid when occlusion culling
//! is on) and writes the draw arguments itself (see `occlusion.rs`).
//!
//! Rows are sorted by what a multi-draw shares (texture or material-table
//! group, geometry and instance buffers), so each such group is one range
//! of rows and goes out as one `multi_draw_indexed_indirect` per pass.

use super::occlusion::CandidateMesh;
use crate::portal::SpaceId;
use crate::world::EntityId;
use std::collections::HashMap;
use std::sync::Arc;

/// Whether the GPU scene is switched off (`ENGINE_GPU_SCENE=0`).
pub fn disabled_by_env() -> bool {
    std::env::var("ENGINE_GPU_SCENE").is_ok_and(|v| v.trim() == "0")
}

/// Whether multi-draws take their count from the GPU is switched off
/// (`ENGINE_DRAW_COUNT=0`: every row is drawn, hidden ones with no
/// instances).
pub fn draw_count_disabled_by_env() -> bool {
    std::env::var("ENGINE_DRAW_COUNT").is_ok_and(|v| v.trim() == "0")
}

/// Whether multi-draws may take their count from the GPU on this adapter.
/// Not on Mesa's software driver (lavapipe, the verify runs' and CI's
/// GPU): it carries a draw-count multi-draw's count over to the plain
/// indirect draws after it, so after an empty group (nothing of it in view)
/// the GPU-culled instanced props drew nothing.
pub fn draw_counts_trusted(info: &wgpu::AdapterInfo) -> bool {
    !(info.backend == wgpu::Backend::Vulkan && info.driver.eq_ignore_ascii_case("llvmpipe"))
}

/// What one multi-draw shares.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GroupKey {
    /// Drawn through the texture-array pipeline (`bind` is the table).
    pub bindless: bool,
    pub bind: wgpu::BindGroup,
    pub vertex: wgpu::Buffer,
    pub index: wgpu::Buffer,
    pub instances: wgpu::Buffer,
}

/// A range of rows that share a [`GroupKey`].
#[derive(Clone, Debug)]
pub struct SceneGroup {
    pub key: GroupKey,
    pub start: u32,
    pub len: u32,
}

/// What, when unchanged, means no row needs looking at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SceneStamp {
    pub render: u64,
    pub instance: u64,
    pub resource: u64,
    pub table: u64,
    pub space: SpaceId,
}

/// One row as the CPU worked it out.
pub struct SceneRow {
    pub id: EntityId,
    /// Its place among the opaque meshes: groups keep the order of their
    /// first mesh and their meshes this order, as the CPU-built draws do
    /// (cutout foliage and the meshes behind it come out the same).
    pub order: u32,
    pub key: GroupKey,
    pub mesh: CandidateMesh,
}

/// Rows to write after a sync: `start..start + len`, and whether their
/// "seen last frame" flags must be set again (a different mesh now).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowWrite {
    pub start: usize,
    pub len: usize,
    pub reset_seen: bool,
}

#[derive(Default)]
pub struct GpuScene {
    pub stamp: Option<SceneStamp>,
    pub ids: Vec<EntityId>,
    pub slots: Arc<HashMap<EntityId, u32>>,
    pub rows: Vec<CandidateMesh>,
    keys: Vec<GroupKey>,
    pub groups: Vec<SceneGroup>,
    /// Rows written last sync (for the frame stats).
    pub written: usize,
    /// The last sync changed which rows form which group.
    pub regrouped: bool,
}

impl GpuScene {
    /// Take this frame's rows (any order); returns the ranges to write.
    /// `all` asks for every row to be written again (new GPU buffers).
    pub fn apply(&mut self, rows: Vec<SceneRow>, all: bool) -> Vec<RowWrite> {
        // wgpu handles hash by their id, which never changes.
        #[allow(clippy::mutable_key_type)]
        let mut first: HashMap<&GroupKey, u32> = HashMap::new();
        for r in &rows {
            let at = first.entry(&r.key).or_insert(r.order);
            *at = (*at).min(r.order);
        }
        let mut sort: Vec<(bool, u32, u32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.key.bindless, first[&r.key], r.order, i))
            .collect();
        drop(first);
        sort.sort_unstable();
        let mut taken: Vec<Option<SceneRow>> = rows.into_iter().map(Some).collect();
        let rows: Vec<SceneRow> = sort
            .into_iter()
            .map(|(.., i)| taken[i].take().expect("each row once"))
            .collect();
        let same_order = !all
            && rows.len() == self.ids.len()
            && rows
                .iter()
                .zip(&self.ids)
                .zip(&self.keys)
                .all(|((r, id), k)| r.id == *id && r.key == *k);
        let mut dirty: Vec<(usize, bool)> = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let same_id = !all && self.ids.get(i) == Some(&r.id);
            let same_mesh = same_id && self.rows.get(i) == Some(&r.mesh);
            if !same_mesh {
                dirty.push((i, !same_id));
            }
        }
        let writes = runs(&dirty);
        self.written = dirty.len();
        self.regrouped = !same_order;
        if !same_order {
            self.ids = rows.iter().map(|r| r.id).collect();
            self.slots = Arc::new(
                self.ids
                    .iter()
                    .enumerate()
                    .map(|(i, id)| (*id, u32::try_from(i).expect("scene row exceeds u32")))
                    .collect(),
            );
            self.keys = rows.iter().map(|r| r.key.clone()).collect();
            self.groups.clear();
            for (i, key) in self.keys.iter().enumerate() {
                match self.groups.last_mut() {
                    Some(g) if g.key == *key => g.len += 1,
                    _ => self.groups.push(SceneGroup {
                        key: key.clone(),
                        start: i as u32,
                        len: 1,
                    }),
                }
            }
        }
        self.rows = rows.into_iter().map(|r| r.mesh).collect();
        writes
    }

    /// Each row's group index and that group's first row (for the GPU's
    /// per-group compaction).
    pub fn row_groups(&self) -> Vec<[u32; 2]> {
        let mut out = Vec::with_capacity(self.ids.len());
        for (i, g) in self.groups.iter().enumerate() {
            let at = [u32::try_from(i).expect("group count exceeds u32"), g.start];
            out.extend(std::iter::repeat_n(at, g.len as usize));
        }
        out
    }
}

/// Consecutive dirty rows with the same flag, as ranges.
fn runs(dirty: &[(usize, bool)]) -> Vec<RowWrite> {
    let mut out: Vec<RowWrite> = Vec::new();
    for &(i, reset_seen) in dirty {
        match out.last_mut() {
            Some(w) if w.start + w.len == i && w.reset_seen == reset_seen => w.len += 1,
            _ => out.push(RowWrite {
                start: i,
                len: 1,
                reset_seen,
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_split_on_gaps_and_flags() {
        let w = runs(&[(0, false), (1, false), (2, true), (5, true), (6, true)]);
        assert_eq!(
            w,
            vec![
                RowWrite {
                    start: 0,
                    len: 2,
                    reset_seen: false
                },
                RowWrite {
                    start: 2,
                    len: 1,
                    reset_seen: true
                },
                RowWrite {
                    start: 5,
                    len: 2,
                    reset_seen: true
                },
            ]
        );
    }

    /// Draw a full-screen triangle with a plain indirect draw, right after
    /// a draw-count multi-draw whose count is zero when `after_count`; true
    /// when it shows.
    fn plain_draw_shows(device: &wgpu::Device, queue: &wgpu::Queue, after_count: bool) -> bool {
        use wgpu::util::DeviceExt;
        const SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4<f32>(p, 0.5, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview: None,
            cache: None,
        });
        let indirect = wgpu::BufferUsages::INDIRECT;
        let buffer = |data: &[u32], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            })
        };
        let index = buffer(&[0, 1, 2], wgpu::BufferUsages::INDEX);
        // One triangle, one instance.
        let args = buffer(&[3, 1, 0, 0, 0], indirect);
        let counts = buffer(&[0], indirect);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
            if after_count {
                pass.multi_draw_indexed_indirect_count(&args, 0, &counts, 0, 1);
            }
            pass.draw_indexed_indirect(&args, 0);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(4),
                },
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));
        read.slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::Wait).expect("poll");
        let px = read.slice(..).get_mapped_range()[256 + 4];
        px == 255
    }

    /// Some drivers (Mesa's lavapipe) carry a draw-count multi-draw's count
    /// over to the plain indirect draws after it, so those draw nothing
    /// when it was zero: the GPU scene's empty groups then hid the instanced
    /// props drawn after them. Draw counts are used only on drivers where
    /// plain draws after them still show.
    #[test]
    fn plain_draws_after_draw_counts_show_where_draw_counts_are_used() {
        if !crate::gpu_field::verify::gpu_tests_enabled() {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("adapter");
        let wanted =
            wgpu::Features::MULTI_DRAW_INDIRECT | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        if !adapter.features().contains(wanted) {
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wanted,
            ..Default::default()
        }))
        .expect("device");
        assert!(
            plain_draw_shows(&device, &queue, false),
            "the test draw shows"
        );
        let info = adapter.get_info();
        if draw_counts_trusted(&info) {
            assert!(
                plain_draw_shows(&device, &queue, true),
                "{info:?} uses draw counts, yet a plain draw after an empty one draws nothing"
            );
        }
    }
}
