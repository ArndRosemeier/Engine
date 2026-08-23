//! CPU isosurface extraction from a corner density grid (verification reference).

use glam::Vec3;

use crate::color::Color;
use crate::marching_cubes::{triangulate_cell, CornerValues};
use crate::mesh::BuiltMesh;

use super::grid::FieldGrid;

/// Marching cubes on CPU — reference path for GPU extract verification.
pub fn extract_mesh_cpu(grid: &FieldGrid, density: &[f32], color: Color) -> BuiltMesh {
    let rgba = color.to_vec4();
    let mut mesh = BuiltMesh::default();
    let voxel = grid.bounds.voxel_size;

    for z in 0..grid.cells[2] {
        for y in 0..grid.cells[1] {
            for x in 0..grid.cells[0] {
                let corners = CornerValues([
                    grid.corner_density(density, x, y, z),
                    grid.corner_density(density, x + 1, y, z),
                    grid.corner_density(density, x + 1, y, z + 1),
                    grid.corner_density(density, x, y, z + 1),
                    grid.corner_density(density, x, y + 1, z),
                    grid.corner_density(density, x + 1, y + 1, z),
                    grid.corner_density(density, x + 1, y + 1, z + 1),
                    grid.corner_density(density, x, y + 1, z + 1),
                ]);
                let origin = grid.cell_origin(x, y, z);
                for tri in triangulate_cell(corners, origin, voxel) {
                    for &p in &tri {
                        let idx = mesh.positions.len() as u32;
                        mesh.positions.push(p);
                        mesh.normals.push(Vec3::Y);
                        mesh.colors.push(rgba);
                        mesh.uvs.push([0.0, 0.0]);
                        mesh.indices.push(idx);
                    }
                }
            }
        }
    }
    mesh.opaque_index_count = mesh.indices.len();
    mesh
}
