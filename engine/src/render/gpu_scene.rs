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
}
