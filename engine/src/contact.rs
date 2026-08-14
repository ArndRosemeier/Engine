//! CPU terrain contact for streamed heightfield chunks.
//!
//! The drawn triangles *are* the walkable surface. A [`ContactGrid`] is produced
//! by the same chunk bake that produced the land mesh, from the same samples and
//! the same diagonal split, so feet can never sink into or float above the
//! geometry on screen.
//!
//! Horizontal obstacles (trees, walls) live in [`crate::collision`], not here.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{EngineError, EngineResult};
use crate::space::{ChunkCoord, ChunkSpan, GlobalXZ};

/// Vertex heights of one chunk's land mesh, in absolute metres.
#[derive(Clone, Debug)]
pub struct ContactGrid {
    origin: GlobalXZ,
    step_m: f64,
    verts: usize,
    heights: Vec<f32>,
}

impl ContactGrid {
    /// `heights` is row-major (`z` major, `x` minor) with `verts * verts` entries.
    pub fn new(
        origin: GlobalXZ,
        step_m: f64,
        verts: usize,
        heights: Vec<f32>,
    ) -> EngineResult<Self> {
        if verts < 2 {
            return Err(EngineError::InvalidValue(format!(
                "contact grid needs at least 2 vertices per axis, got {verts}"
            )));
        }
        if !(step_m.is_finite() && step_m > 0.0) {
            return Err(EngineError::InvalidValue(format!(
                "contact grid step must be finite and > 0, got {step_m}"
            )));
        }
        if heights.len() != verts * verts {
            return Err(EngineError::InvalidValue(format!(
                "contact grid expected {} heights, got {}",
                verts * verts,
                heights.len()
            )));
        }
        if let Some(bad) = heights.iter().find(|h| !h.is_finite()) {
            return Err(EngineError::InvalidValue(format!(
                "contact grid height must be finite, got {bad}"
            )));
        }
        Ok(Self {
            origin,
            step_m,
            verts,
            heights,
        })
    }

    pub fn origin(&self) -> GlobalXZ {
        self.origin
    }

    pub fn step_m(&self) -> f64 {
        self.step_m
    }

    pub fn verts(&self) -> usize {
        self.verts
    }

    /// Extent covered by this grid, in metres.
    pub fn span_m(&self) -> f64 {
        (self.verts - 1) as f64 * self.step_m
    }

    pub fn contains(&self, p: GlobalXZ) -> bool {
        let span = self.span_m();
        let dx = p.x - self.origin.x;
        let dz = p.z - self.origin.z;
        (0.0..=span).contains(&dx) && (0.0..=span).contains(&dz)
    }

    #[inline]
    fn height_at_vertex(&self, ix: usize, iz: usize) -> f32 {
        self.heights[iz * self.verts + ix]
    }

    /// Exact height on the drawn triangle under `p`, or `None` outside the grid.
    ///
    /// Uses the same quad diagonal as the mesh builder
    /// (`[i00, i01, i11]` + `[i00, i11, i10]`), so this is the rendered surface,
    /// not a re-evaluation of the generator.
    pub fn height_at(&self, p: GlobalXZ) -> Option<f32> {
        if !self.contains(p) {
            return None;
        }
        let last = (self.verts - 1) as f64;
        let fx = ((p.x - self.origin.x) / self.step_m).clamp(0.0, last - f64::EPSILON);
        let fz = ((p.z - self.origin.z) / self.step_m).clamp(0.0, last - f64::EPSILON);
        let ix = fx.floor() as usize;
        let iz = fz.floor() as usize;
        let u = (fx - ix as f64) as f32;
        let v = (fz - iz as f64) as f32;

        let h00 = self.height_at_vertex(ix, iz);
        let h10 = self.height_at_vertex(ix + 1, iz);
        let h01 = self.height_at_vertex(ix, iz + 1);
        let h11 = self.height_at_vertex(ix + 1, iz + 1);

        Some(if v >= u {
            h00 * (1.0 - v) + h01 * (v - u) + h11 * u
        } else {
            h00 * (1.0 - u) + h10 * (u - v) + h11 * v
        })
    }
}

/// The drawn ground as it stands right now, readable from another thread.
///
/// Contact grids belong to the streamer, which is main-thread state: a worker
/// that wants to place things on the ground — cover, props, anything sown on the
/// surface the player actually walks on — cannot borrow it while the streamer
/// keeps loading. Taking a snapshot costs one clone of a map of handles, and the
/// grids themselves are shared, not copied. It is a snapshot in the strict
/// sense: chunks that arrive afterwards are not in it.
#[derive(Clone, Debug, Default)]
pub struct ContactSnapshot {
    span: Option<ChunkSpan>,
    grids: Arc<HashMap<ChunkCoord, Arc<ContactGrid>>>,
}

impl ContactSnapshot {
    pub fn new(span: ChunkSpan, grids: HashMap<ChunkCoord, Arc<ContactGrid>>) -> Self {
        Self {
            span: Some(span),
            grids: Arc::new(grids),
        }
    }

    /// Height of the drawn ground under `p`, or `None` where nothing is resident.
    pub fn height_at(&self, p: GlobalXZ) -> Option<f32> {
        let span = self.span?;
        self.grids
            .get(&ChunkCoord::containing(p, span))
            .and_then(|g| g.height_at(p))
    }

    pub fn len(&self) -> usize {
        self.grids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_grid() -> ContactGrid {
        // Height = x metres, so interpolation is exactly predictable.
        let verts = 3;
        let step = 10.0;
        let mut heights = Vec::new();
        for _z in 0..verts {
            for x in 0..verts {
                heights.push(x as f32 * step as f32);
            }
        }
        ContactGrid::new(GlobalXZ::at(1000.0, -500.0), step, verts, heights).unwrap()
    }

    #[test]
    fn interpolates_planar_ramp_exactly() {
        let g = ramp_grid();
        let h = g.height_at(GlobalXZ::at(1005.0, -495.0)).unwrap();
        assert!((h - 5.0).abs() < 1e-4, "expected 5.0, got {h}");
    }

    #[test]
    fn corners_match_vertex_heights() {
        let g = ramp_grid();
        assert_eq!(g.height_at(GlobalXZ::at(1000.0, -500.0)).unwrap(), 0.0);
        assert!((g.height_at(GlobalXZ::at(1020.0, -480.0)).unwrap() - 20.0).abs() < 1e-3);
    }

    #[test]
    fn outside_the_grid_reports_no_contact() {
        let g = ramp_grid();
        assert!(g.height_at(GlobalXZ::at(1021.0, -500.0)).is_none());
        assert!(g.height_at(GlobalXZ::at(999.0, -500.0)).is_none());
    }

    #[test]
    fn malformed_grids_are_rejected() {
        assert!(ContactGrid::new(GlobalXZ::ORIGIN, 1.0, 2, vec![0.0; 3]).is_err());
        assert!(ContactGrid::new(GlobalXZ::ORIGIN, 0.0, 2, vec![0.0; 4]).is_err());
        assert!(ContactGrid::new(GlobalXZ::ORIGIN, 1.0, 2, vec![f32::NAN; 4]).is_err());
    }
}
