//! Where water can be: open spans of each column, read from the solid world.

use crate::volume::Volume;
use glam::Vec3;

/// One open (non-solid) vertical span of a column, bottom to top, in metres.
/// `ceil` is `f32::INFINITY` when the column is open to the sky.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub floor: f32,
    pub ceil: f32,
}

impl Span {
    pub fn height(&self) -> f32 {
        self.ceil - self.floor
    }
}

/// The solid world as water sees it.
///
/// Implement [`WaterTerrain::density`] (positive inside rock, negative in
/// air, like [`Volume`]) and the default [`WaterTerrain::column`] finds the
/// open spans by scanning it. Height maps override `column` with the exact
/// answer.
pub trait WaterTerrain: Sync {
    /// Signed density at a world position: > 0 solid, < 0 open.
    fn density(&self, p: Vec3) -> f32;

    /// Vertical sampling step for [`WaterTerrain::column`], in metres.
    fn scan_step(&self) -> f32 {
        0.5
    }

    /// Append the open spans of the column through `(x, z)` between `y_min`
    /// and `y_max`, bottom to top. The region bottom is a closed floor; a
    /// span that reaches `y_max` is open to the sky (`ceil = INFINITY`).
    fn column(&self, x: f32, z: f32, y_min: f32, y_max: f32, out: &mut Vec<Span>) {
        scan_column(
            |y| self.density(Vec3::new(x, y, z)),
            y_min,
            y_max,
            self.scan_step(),
            out,
        );
    }
}

/// Find open spans along a vertical line by sampling `density` every `step`
/// metres and placing each surface at the linear zero crossing.
pub fn scan_column(
    density: impl Fn(f32) -> f32,
    y_min: f32,
    y_max: f32,
    step: f32,
    out: &mut Vec<Span>,
) {
    let step = step.max(1e-3);
    let n = (((y_max - y_min) / step).ceil() as usize).max(1);
    let mut prev_y = y_min;
    let mut prev_d = density(y_min);
    let mut open_from: Option<f32> = (prev_d <= 0.0).then_some(y_min);
    for i in 1..=n {
        let y = (y_min + i as f32 * step).min(y_max);
        let d = density(y);
        let was_open = prev_d <= 0.0;
        let is_open = d <= 0.0;
        if was_open != is_open {
            // Linear zero crossing between the two samples.
            let t = if (prev_d - d).abs() > 1e-12 {
                (prev_d / (prev_d - d)).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let yc = prev_y + (y - prev_y) * t;
            if is_open {
                open_from = Some(yc);
            } else if let Some(floor) = open_from.take() {
                out.push(Span { floor, ceil: yc });
            }
        }
        prev_y = y;
        prev_d = d;
    }
    if let Some(floor) = open_from {
        out.push(Span {
            floor,
            ceil: f32::INFINITY,
        });
    }
}

impl WaterTerrain for Volume {
    fn density(&self, p: Vec3) -> f32 {
        self.sample(p)
    }

    fn scan_step(&self) -> f32 {
        // Density is trilinear between corners, so a quarter voxel finds
        // every crossing the mesher shows.
        self.voxel_size() * 0.25
    }
}

impl WaterTerrain for crate::voxel_terrain::VoxelTerrain {
    fn density(&self, p: Vec3) -> f32 {
        self.volume().sample(p)
    }

    fn scan_step(&self) -> f32 {
        self.volume().voxel_size() * 0.25
    }
}

/// A height map as water terrain: solid below `height(x, z)`, open above.
pub struct HeightMapTerrain<F: Fn(f32, f32) -> f32 + Sync>(pub F);

impl<F: Fn(f32, f32) -> f32 + Sync> WaterTerrain for HeightMapTerrain<F> {
    fn density(&self, p: Vec3) -> f32 {
        (self.0)(p.x, p.z) - p.y
    }

    fn column(&self, x: f32, z: f32, y_min: f32, _y_max: f32, out: &mut Vec<Span>) {
        let h = (self.0)(x, z);
        out.push(Span {
            floor: if h.is_finite() { h.max(y_min) } else { y_min },
            ceil: f32::INFINITY,
        });
    }
}
