//! Corner grid indexing for a painted scalar field.

use glam::Vec3;

use super::field::FieldBounds;

/// Cell and corner counts for a bounds-aligned regular grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldGrid {
    pub bounds: FieldBounds,
    pub cells: [u32; 3],
    pub corners: [u32; 3],
}

impl FieldGrid {
    pub fn from_bounds(bounds: FieldBounds) -> Self {
        let e = bounds.extent();
        let cells = [
            (e.x / bounds.voxel_size()).ceil().max(1.0) as u32,
            (e.y / bounds.voxel_size()).ceil().max(1.0) as u32,
            (e.z / bounds.voxel_size()).ceil().max(1.0) as u32,
        ];
        let corners = [cells[0] + 1, cells[1] + 1, cells[2] + 1];
        Self {
            bounds,
            cells,
            corners,
        }
    }

    pub fn corner_count(&self) -> usize {
        self.corners[0] as usize * self.corners[1] as usize * self.corners[2] as usize
    }

    pub fn cell_count(&self) -> usize {
        self.cells[0] as usize * self.cells[1] as usize * self.cells[2] as usize
    }

    pub fn corner_world(&self, ix: u32, iy: u32, iz: u32) -> Vec3 {
        self.bounds.min + Vec3::new(ix as f32, iy as f32, iz as f32) * self.bounds.voxel_size()
    }

    pub fn cell_origin(&self, ix: u32, iy: u32, iz: u32) -> Vec3 {
        self.corner_world(ix, iy, iz)
    }

    pub fn corner_index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        (ix + self.corners[0] * (iy + iz * self.corners[1])) as usize
    }

    pub fn corner_density(&self, density: &[f32], ix: u32, iy: u32, iz: u32) -> f32 {
        density[self.corner_index(ix, iy, iz)]
    }

    /// Trilinear sample of the corner grid at a world position.
    pub fn sample_density(&self, density: &[f32], world: Vec3) -> f32 {
        let local = (world - self.bounds.min) / self.bounds.voxel_size();
        let max_corner = self.corners.map(|count| {
            count
                .checked_sub(1)
                .expect("FieldGrid invariant: every axis has at least one corner")
        });
        let max = Vec3::new(
            max_corner[0] as f32,
            max_corner[1] as f32,
            max_corner[2] as f32,
        );
        let clamped = local.clamp(Vec3::ZERO, max);
        let base = clamped.floor();
        let frac = clamped - base;
        let x0 = base.x as u32;
        let y0 = base.y as u32;
        let z0 = base.z as u32;
        let x1 = x0
            .checked_add(1)
            .expect("sample x corner overflow")
            .min(max_corner[0]);
        let y1 = y0
            .checked_add(1)
            .expect("sample y corner overflow")
            .min(max_corner[1]);
        let z1 = z0
            .checked_add(1)
            .expect("sample z corner overflow")
            .min(max_corner[2]);

        let c000 = self.corner_density(density, x0, y0, z0);
        let c100 = self.corner_density(density, x1, y0, z0);
        let c010 = self.corner_density(density, x0, y1, z0);
        let c110 = self.corner_density(density, x1, y1, z0);
        let c001 = self.corner_density(density, x0, y0, z1);
        let c101 = self.corner_density(density, x1, y0, z1);
        let c011 = self.corner_density(density, x0, y1, z1);
        let c111 = self.corner_density(density, x1, y1, z1);

        let c00 = c000 * (1.0 - frac.x) + c100 * frac.x;
        let c10 = c010 * (1.0 - frac.x) + c110 * frac.x;
        let c01 = c001 * (1.0 - frac.x) + c101 * frac.x;
        let c11 = c011 * (1.0 - frac.x) + c111 * frac.x;
        let c0 = c00 * (1.0 - frac.y) + c10 * frac.y;
        let c1 = c01 * (1.0 - frac.y) + c11 * frac.y;
        c0 * (1.0 - frac.z) + c1 * frac.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_count_matches_cells_plus_one() {
        let bounds = FieldBounds::try_new(Vec3::ZERO, Vec3::splat(8.0), 1.0).unwrap();
        let grid = FieldGrid::from_bounds(bounds);
        assert_eq!(grid.cells, [8, 8, 8]);
        assert_eq!(grid.corners, [9, 9, 9]);
        assert_eq!(grid.corner_count(), 9 * 9 * 9);
    }
}
