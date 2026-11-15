//! The water's skin: particles are splatted onto the grid's corners as a
//! density, softened once, and marching cubes draws where the density
//! crosses half of a full cell.

use super::sim::Sim;
use crate::color::Color;
use crate::error::EngineResult;
use crate::marching_cubes::{triangulate_cell, CornerValues};
use crate::mesh::{Mesh, PointId};
use glam::{IVec3, UVec3, Vec3};
use std::collections::HashMap;

/// Density of a full cell (particles per cell); the surface sits at half.
const FULL: f32 = 8.0;
const LEVEL: f32 = 0.45 * FULL;

struct Grid {
    n: UVec3,
    data: Vec<f32>,
}

impl Grid {
    fn at(&self, x: i32, y: i32, z: i32) -> f32 {
        if x < 0
            || y < 0
            || z < 0
            || x >= self.n.x as i32
            || y >= self.n.y as i32
            || z >= self.n.z as i32
        {
            return 0.0;
        }
        self.data[(x as u32 + self.n.x * (y as u32 + self.n.y * z as u32)) as usize]
    }

    /// Trilinear sample in node units.
    fn sample(&self, g: Vec3) -> f32 {
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        let mut s = 0.0;
        for (dz, wz) in [(0, 1.0 - f.z), (1, f.z)] {
            for (dy, wy) in [(0, 1.0 - f.y), (1, f.y)] {
                for (dx, wx) in [(0, 1.0 - f.x), (1, f.x)] {
                    s += self.at(i.x + dx, i.y + dy, i.z + dz) * wx * wy * wz;
                }
            }
        }
        s
    }
}

fn density(sim: &Sim) -> Grid {
    // Nodes sit on cell corners: (n + 1) per side.
    let n = sim.n + 1;
    let mut data = vec![0.0f32; (n.x * n.y * n.z) as usize];
    let at = |c: IVec3| (c.x as u32 + n.x * (c.y as u32 + n.y * c.z as u32)) as usize;
    for p in &sim.pos {
        let g = *p / sim.h;
        let i = g.floor().as_ivec3();
        let f = g - i.as_vec3();
        for (dz, wz) in [(0, 1.0 - f.z), (1, f.z)] {
            for (dy, wy) in [(0, 1.0 - f.y), (1, f.y)] {
                for (dx, wx) in [(0, 1.0 - f.x), (1, f.x)] {
                    let c = i + IVec3::new(dx, dy, dz);
                    if c.cmplt(IVec3::ZERO).any() || c.cmpge(n.as_ivec3()).any() {
                        continue;
                    }
                    // A node inside full water gathers about FULL.
                    data[at(c)] += wx * wy * wz;
                }
            }
        }
    }
    let raw = Grid { n, data };
    // One soft pass (6-neighbour blur) evens out particle noise.
    let mut soft = vec![0.0f32; raw.data.len()];
    for z in 0..n.z as i32 {
        for y in 0..n.y as i32 {
            for x in 0..n.x as i32 {
                let mut s = raw.at(x, y, z) * 2.0;
                for (dx, dy, dz) in [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ] {
                    s += raw.at(x + dx, y + dy, z + dz);
                }
                soft[at(IVec3::new(x, y, z))] = s / 8.0;
            }
        }
    }
    // The outermost nodes stay empty so the skin closes at the box walls.
    for z in 0..n.z as i32 {
        for y in 0..n.y as i32 {
            for x in 0..n.x as i32 {
                let edge = x == 0
                    || y == 0
                    || z == 0
                    || x == n.x as i32 - 1
                    || y == n.y as i32 - 1
                    || z == n.z as i32 - 1;
                if edge {
                    soft[at(IVec3::new(x, y, z))] = 0.0;
                }
            }
        }
    }
    Grid { n, data: soft }
}

pub(super) fn build(sim: &Sim, color: Color) -> EngineResult<Mesh> {
    let mut mesh = Mesh::new();
    let mut surface = mesh.surface_material();
    surface.roughness = 0.06;
    surface.metallic = 0.0;
    mesh.set_surface_material(surface);
    if sim.pos.is_empty() {
        return Ok(mesh);
    }
    let grid = density(sim);
    let h = sim.h;
    // Cells worth looking at: any with a particle in or next to it.
    let n = sim.n;
    let mut busy = vec![false; (n.x * n.y * n.z) as usize];
    for p in &sim.pos {
        let c = sim.cell_of(*p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(i) = sim.cell_index(c + IVec3::new(dx, dy, dz)) {
                        busy[i] = true;
                    }
                }
            }
        }
    }
    let mut points: HashMap<IVec3, PointId> = HashMap::new();
    let key = |p: Vec3| (p * (1024.0 / h)).round().as_ivec3();
    for z in 0..n.z as i32 {
        for y in 0..n.y as i32 {
            for x in 0..n.x as i32 {
                let i = (x as u32 + n.x * (y as u32 + n.y * z as u32)) as usize;
                if !busy[i] {
                    continue;
                }
                let corner = |dx: i32, dy: i32, dz: i32| grid.at(x + dx, y + dy, z + dz) - LEVEL;
                let values = CornerValues([
                    corner(0, 0, 0),
                    corner(1, 0, 0),
                    corner(1, 0, 1),
                    corner(0, 0, 1),
                    corner(0, 1, 0),
                    corner(1, 1, 0),
                    corner(1, 1, 1),
                    corner(0, 1, 1),
                ]);
                let origin = Vec3::new(x as f32, y as f32, z as f32) * h;
                for tri in triangulate_cell(values, origin, h) {
                    let mut ids = Vec::with_capacity(3);
                    for p in tri.iter() {
                        let id = match points.get(&key(*p)) {
                            Some(id) => *id,
                            None => {
                                let id = mesh.add_point(*p)?;
                                let g = *p / h;
                                let e = 0.5;
                                let grad = Vec3::new(
                                    grid.sample(g + Vec3::X * e) - grid.sample(g - Vec3::X * e),
                                    grid.sample(g + Vec3::Y * e) - grid.sample(g - Vec3::Y * e),
                                    grid.sample(g + Vec3::Z * e) - grid.sample(g - Vec3::Z * e),
                                );
                                let normal = (-grad).try_normalize().unwrap_or(Vec3::Y);
                                mesh.set_point_normal(id, normal)?;
                                mesh.set_point_color(id, color)?;
                                points.insert(key(*p), id);
                                id
                            }
                        };
                        ids.push(id);
                    }
                    if ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2] {
                        mesh.add_triangle(ids[0], ids[1], ids[2])?;
                    }
                }
            }
        }
    }
    Ok(mesh)
}
