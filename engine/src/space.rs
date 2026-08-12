//! Typed world spaces for large worlds.
//!
//! Two spaces exist and never mix implicitly:
//!
//! * **Global** ([`GlobalPosition`], [`GlobalXZ`]) — absolute `f64` metres.
//!   Generation, saves, and gameplay logic live here. Values are invariant:
//!   rebasing the render origin never changes them.
//! * **Render** ([`RenderPosition`]) — `f32` metres relative to the current
//!   [`RenderOrigin`]. Mesh vertices, entity transforms, and the camera live
//!   here so `f32` precision stays local even 1000 km from the world origin.
//!
//! Rebasing is horizontal only. Vertical extent is bounded (hundreds of metres,
//! not thousands of kilometres), so `y` stays absolute in both spaces. That
//! keeps height-dependent shading (sea level bands, snow lines) correct without
//! threading a vertical offset through every shader.

use crate::error::{EngineError, EngineResult};
use glam::{DVec2, Vec3};

/// Absolute horizontal position in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalXZ {
    pub x: f64,
    pub z: f64,
}

impl GlobalXZ {
    pub const ORIGIN: Self = Self { x: 0.0, z: 0.0 };

    /// Checked constructor — global coordinates must be finite.
    pub fn new(x: f64, z: f64) -> EngineResult<Self> {
        if !x.is_finite() || !z.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "global xz must be finite, got ({x}, {z})"
            )));
        }
        Ok(Self { x, z })
    }

    pub fn at(x: f64, z: f64) -> Self {
        Self::new(x, z).expect("GlobalXZ::at requires finite coordinates")
    }

    pub fn with_height(self, y: f64) -> EngineResult<GlobalPosition> {
        GlobalPosition::new(self.x, y, self.z)
    }

    pub fn offset(self, dx: f64, dz: f64) -> EngineResult<Self> {
        Self::new(self.x + dx, self.z + dz)
    }

    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dz = self.z - other.z;
        (dx * dx + dz * dz).sqrt()
    }

    pub fn to_dvec2(self) -> DVec2 {
        DVec2::new(self.x, self.z)
    }
}

/// Absolute position in metres (`y` up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl GlobalPosition {
    pub const ORIGIN: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> EngineResult<Self> {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "global position must be finite, got ({x}, {y}, {z})"
            )));
        }
        Ok(Self { x, y, z })
    }

    pub fn at(x: f64, y: f64, z: f64) -> Self {
        Self::new(x, y, z).expect("GlobalPosition::at requires finite coordinates")
    }

    pub fn horizontal(self) -> GlobalXZ {
        GlobalXZ {
            x: self.x,
            z: self.z,
        }
    }

    pub fn with_height(self, y: f64) -> EngineResult<Self> {
        Self::new(self.x, y, self.z)
    }

    /// Convert to render space. Fails if the result exceeds `f32` range.
    pub fn to_render(self, origin: RenderOrigin) -> EngineResult<RenderPosition> {
        RenderPosition::new(Vec3::new(
            (self.x - origin.horizontal.x) as f32,
            self.y as f32,
            (self.z - origin.horizontal.z) as f32,
        ))
    }
}

/// Position in render space: metres relative to the active [`RenderOrigin`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPosition(Vec3);

impl RenderPosition {
    pub const ZERO: Self = Self(Vec3::ZERO);

    pub fn new(v: Vec3) -> EngineResult<Self> {
        if !v.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "render position must be finite, got {v}"
            )));
        }
        Ok(Self(v))
    }

    pub fn at(x: f32, y: f32, z: f32) -> Self {
        Self::new(Vec3::new(x, y, z)).expect("RenderPosition::at requires finite coordinates")
    }

    pub fn vec3(self) -> Vec3 {
        self.0
    }

    pub fn to_global(self, origin: RenderOrigin) -> GlobalPosition {
        GlobalPosition {
            x: origin.horizontal.x + self.0.x as f64,
            y: self.0.y as f64,
            z: origin.horizontal.z + self.0.z as f64,
        }
    }
}

/// Horizontal anchor that render space is expressed relative to.
///
/// Held by [`crate::world::World`]. Changing it re-derives every anchored
/// transform from its immutable global anchor — global data is never shifted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOrigin {
    horizontal: GlobalXZ,
}

impl Default for RenderOrigin {
    fn default() -> Self {
        Self {
            horizontal: GlobalXZ::ORIGIN,
        }
    }
}

impl RenderOrigin {
    pub fn new(horizontal: GlobalXZ) -> Self {
        Self { horizontal }
    }

    pub fn horizontal(self) -> GlobalXZ {
        self.horizontal
    }

    /// Snap to a multiple of `step_m` so repeated rebases land on a stable grid.
    pub fn snapped(horizontal: GlobalXZ, step_m: f64) -> EngineResult<Self> {
        if !(step_m.is_finite() && step_m > 0.0) {
            return Err(EngineError::InvalidValue(format!(
                "rebase step must be finite and > 0, got {step_m}"
            )));
        }
        Ok(Self {
            horizontal: GlobalXZ {
                x: (horizontal.x / step_m).floor() * step_m,
                z: (horizontal.z / step_m).floor() * step_m,
            },
        })
    }

    /// Offset applied to render XZ to recover world-space texture phase.
    ///
    /// Wrapped into `period_m` so the value stays small in `f32` while keeping
    /// tiling continuous across rebases.
    pub fn texture_phase(self, period_m: f32) -> [f32; 2] {
        if !(period_m.is_finite() && period_m > 0.0) {
            return [0.0, 0.0];
        }
        let p = period_m as f64;
        [
            self.horizontal.x.rem_euclid(p) as f32,
            self.horizontal.z.rem_euclid(p) as f32,
        ]
    }
}

/// Edge length of a square streaming chunk, in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkSpan(f64);

impl ChunkSpan {
    pub fn new(metres: f64) -> EngineResult<Self> {
        if !(metres.is_finite() && metres > 0.0) {
            return Err(EngineError::InvalidValue(format!(
                "chunk span must be finite and > 0, got {metres}"
            )));
        }
        Ok(Self(metres))
    }

    pub fn metres(self) -> f64 {
        self.0
    }
}

/// Integer address of a streaming chunk on the horizontal plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// Chunk containing `p`.
    pub fn containing(p: GlobalXZ, span: ChunkSpan) -> Self {
        Self {
            x: (p.x / span.metres()).floor() as i32,
            z: (p.z / span.metres()).floor() as i32,
        }
    }

    /// Minimum-corner position of this chunk.
    pub fn origin(self, span: ChunkSpan) -> GlobalXZ {
        GlobalXZ {
            x: self.x as f64 * span.metres(),
            z: self.z as f64 * span.metres(),
        }
    }

    pub fn centre(self, span: ChunkSpan) -> GlobalXZ {
        GlobalXZ {
            x: (self.x as f64 + 0.5) * span.metres(),
            z: (self.z as f64 + 0.5) * span.metres(),
        }
    }

    /// Chebyshev distance in chunks.
    pub fn ring_distance(self, other: Self) -> i32 {
        (self.x - other.x).abs().max((self.z - other.z).abs())
    }

    /// Manhattan distance in chunks (load ordering).
    pub fn walk_distance(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.z - other.z).abs()
    }

    pub fn offset(self, dx: i32, dz: i32) -> Self {
        Self {
            x: self.x + dx,
            z: self.z + dz,
        }
    }
}

/// Draw layer of a chunk mesh. Layers of one chunk share an anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChunkLayer {
    /// Opaque ground.
    Land,
    /// Translucent water sheets.
    Water,
}

impl ChunkLayer {
    pub const ALL: [ChunkLayer; 2] = [ChunkLayer::Land, ChunkLayer::Water];
}

/// Identity of one uploaded chunk mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId {
    pub coord: ChunkCoord,
    pub layer: ChunkLayer,
}

impl ChunkId {
    pub fn new(coord: ChunkCoord, layer: ChunkLayer) -> Self {
        Self { coord, layer }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_round_trip_preserves_global_position() {
        let origin = RenderOrigin::new(GlobalXZ::at(1_234_000.0, -98_000.0));
        let g = GlobalPosition::at(1_234_050.5, 42.25, -97_980.75);
        let r = g.to_render(origin).unwrap();
        let back = r.to_global(origin);
        assert!((back.x - g.x).abs() < 1e-3);
        assert!((back.y - g.y).abs() < 1e-3);
        assert!((back.z - g.z).abs() < 1e-3);
    }

    #[test]
    fn far_positions_keep_metre_precision_after_rebase() {
        let far = GlobalXZ::at(4_000_000.0, 4_000_000.0);
        let origin = RenderOrigin::snapped(far, 1_000.0).unwrap();
        let a = GlobalPosition::at(far.x + 0.05, 10.0, far.z);
        let b = GlobalPosition::at(far.x + 0.10, 10.0, far.z);
        let ra = a.to_render(origin).unwrap().vec3();
        let rb = b.to_render(origin).unwrap().vec3();
        assert!(
            (rb.x - ra.x) > 0.04,
            "5 cm apart must stay distinguishable after rebase: {} vs {}",
            ra.x,
            rb.x
        );
    }

    #[test]
    fn chunk_lookup_is_exact_on_boundaries() {
        let span = ChunkSpan::new(200.0).unwrap();
        assert_eq!(
            ChunkCoord::containing(GlobalXZ::at(0.0, 0.0), span),
            ChunkCoord::new(0, 0)
        );
        assert_eq!(
            ChunkCoord::containing(GlobalXZ::at(199.999, 200.0), span),
            ChunkCoord::new(0, 1)
        );
        assert_eq!(
            ChunkCoord::containing(GlobalXZ::at(-0.001, -200.0), span),
            ChunkCoord::new(-1, -1)
        );
        let c = ChunkCoord::new(-3, 7);
        assert_eq!(c.origin(span).x, -600.0);
        assert_eq!(c.origin(span).z, 1400.0);
    }

    #[test]
    fn texture_phase_wraps_into_tile_period() {
        let origin = RenderOrigin::new(GlobalXZ::at(1_000_003.5, -7.0));
        let phase = origin.texture_phase(7.0);
        assert!(phase[0] >= 0.0 && phase[0] < 7.0);
        assert!(phase[1] >= 0.0 && phase[1] < 7.0);
        // Same world point must land on the same phase from a different origin.
        let shifted = RenderOrigin::new(GlobalXZ::at(1_000_003.5 + 70.0, -7.0));
        assert!((shifted.texture_phase(7.0)[0] - phase[0]).abs() < 1e-3);
    }

    #[test]
    fn non_finite_coordinates_are_rejected() {
        assert!(GlobalXZ::new(f64::NAN, 0.0).is_err());
        assert!(GlobalPosition::new(0.0, f64::INFINITY, 0.0).is_err());
        assert!(RenderPosition::new(Vec3::new(0.0, f32::NAN, 0.0)).is_err());
        assert!(ChunkSpan::new(0.0).is_err());
    }
}
