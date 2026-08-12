use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::place::ensure_finite3;
use glam::{Vec3, Vec4};
use std::collections::HashMap;

/// Opaque index into [`Mesh`] points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointId(u32);

impl PointId {
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A CPU-side triangle mesh built from human-friendly points and faces.
///
/// Faces may be triangles or quads. Call [`Mesh::build`] (or let the world do it
/// on spawn) to triangulate and compute normals.
///
/// Face winding should be counter-clockwise when looking at the outside.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    points: Vec<Vec3>,
    colors: Vec<Vec4>,
    faces: Vec<Vec<PointId>>,
}

/// Friendly alias — same type as [`Mesh`].
pub type Shape = Mesh;

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    /// Axis-aligned box as a complete mesh.
    pub fn box_at(
        center: impl Into<Vec3>,
        size: impl Into<Vec3>,
        color: Color,
    ) -> EngineResult<Self> {
        let mut mesh = Self::new();
        mesh.add_box(center, size, color)?;
        Ok(mesh)
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Add a point in local space.
    pub fn add_point(&mut self, position: impl Into<Vec3>) -> EngineResult<PointId> {
        let position = position.into();
        ensure_finite3(position, "point position")?;
        let id = PointId(self.points.len() as u32);
        self.points.push(position);
        self.colors.push(Color::rgb(191, 191, 191).to_vec4());
        Ok(id)
    }

    /// Set the display color of an existing point (alpha < 1 draws in the transparent pass).
    pub fn set_point_color(&mut self, id: PointId, color: Color) -> EngineResult<()> {
        let idx = id.0 as usize;
        if idx >= self.colors.len() {
            return Err(EngineError::InvalidMesh(format!(
                "point id {} is out of range",
                id.0
            )));
        }
        self.colors[idx] = color.to_vec4();
        Ok(())
    }

    /// Connect 3 or 4 existing points into a face.
    pub fn add_face(&mut self, points: &[PointId]) -> EngineResult<()> {
        if points.len() != 3 && points.len() != 4 {
            return Err(EngineError::InvalidMesh(format!(
                "faces must have 3 or 4 points, got {}",
                points.len()
            )));
        }
        for &p in points {
            if (p.0 as usize) >= self.points.len() {
                return Err(EngineError::InvalidMesh(format!(
                    "face references missing point {}",
                    p.0
                )));
            }
        }
        self.faces.push(points.to_vec());
        Ok(())
    }

    pub fn add_triangle(&mut self, a: PointId, b: PointId, c: PointId) -> EngineResult<()> {
        self.add_face(&[a, b, c])
    }

    pub fn add_quad(&mut self, a: PointId, b: PointId, c: PointId, d: PointId) -> EngineResult<()> {
        self.add_face(&[a, b, c, d])
    }

    /// Axis-aligned box centered at `center` with full `size` extents.
    pub fn add_box(
        &mut self,
        center: impl Into<Vec3>,
        size: impl Into<Vec3>,
        color: Color,
    ) -> EngineResult<()> {
        let center = center.into();
        let size = size.into();
        ensure_finite3(center, "box center")?;
        ensure_finite3(size, "box size")?;
        if size.x <= 0.0 || size.y <= 0.0 || size.z <= 0.0 {
            return Err(EngineError::InvalidMesh(
                "box size components must be > 0".into(),
            ));
        }
        let half = size * 0.5;
        let corners = [
            center + Vec3::new(-half.x, -half.y, -half.z),
            center + Vec3::new(half.x, -half.y, -half.z),
            center + Vec3::new(half.x, half.y, -half.z),
            center + Vec3::new(-half.x, half.y, -half.z),
            center + Vec3::new(-half.x, -half.y, half.z),
            center + Vec3::new(half.x, -half.y, half.z),
            center + Vec3::new(half.x, half.y, half.z),
            center + Vec3::new(-half.x, half.y, half.z),
        ];
        let base = self.points.len() as u32;
        for c in corners {
            let id = self.add_point(c)?;
            self.set_point_color(id, color)?;
        }
        // Outward-facing quads (CCW from outside): -Z +Z -Y +Y -X +X
        let faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [0, 4, 7, 3],
            [1, 2, 6, 5],
        ];
        for face in faces {
            let ids = face.map(|i| PointId(base + i));
            self.add_face(&ids)?;
        }
        Ok(())
    }

    /// Build GPU-ready geometry: triangulate faces with flat per-face normals.
    ///
    /// Opaque faces are packed first; [`BuiltMesh::opaque_index_count`] marks the split
    /// so the renderer can draw transparent triangles in a second pass.
    pub fn build(&self) -> BuiltMesh {
        self.build_with_normals(false)
    }

    /// Like [`Self::build`], but averages face normals at shared authoring points
    /// so heightfields / ribbons shade as continuous surfaces instead of facets.
    pub fn build_smooth(&self) -> BuiltMesh {
        self.build_with_normals(true)
    }

    fn build_with_normals(&self, smooth: bool) -> BuiltMesh {
        let mut opaque_faces = Vec::new();
        let mut xlucent_faces = Vec::new();
        for face in &self.faces {
            let transparent = face.iter().any(|p| self.colors[p.0 as usize].w < 0.999);
            if transparent {
                xlucent_faces.push(face.as_slice());
            } else {
                opaque_faces.push(face.as_slice());
            }
        }

        let smooth_normals = if smooth {
            Some(self.averaged_vertex_normals())
        } else {
            None
        };

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut indices = Vec::new();

        let emit = |faces: &[&[PointId]],
                    positions: &mut Vec<Vec3>,
                    normals: &mut Vec<Vec3>,
                    colors: &mut Vec<Vec4>,
                    indices: &mut Vec<u32>| {
            for face in faces {
                let tris: [[PointId; 3]; 2] = match face.len() {
                    3 => [[face[0], face[1], face[2]], [face[0], face[0], face[0]]],
                    4 => [[face[0], face[1], face[2]], [face[0], face[2], face[3]]],
                    _ => unreachable!("add_face only allows 3 or 4 points"),
                };
                let tri_count = if face.len() == 3 { 1 } else { 2 };
                for tri in tris.iter().take(tri_count) {
                    let a = self.points[tri[0].0 as usize];
                    let b = self.points[tri[1].0 as usize];
                    let c = self.points[tri[2].0 as usize];
                    let face_n = {
                        let raw = (b - a).cross(c - a);
                        if raw.length_squared() > 0.0 {
                            raw.normalize()
                        } else {
                            Vec3::Y
                        }
                    };
                    let base = positions.len() as u32;
                    positions.push(a);
                    positions.push(b);
                    positions.push(c);
                    if let Some(sn) = smooth_normals.as_ref() {
                        let mut n0 = sn[tri[0].0 as usize];
                        let mut n1 = sn[tri[1].0 as usize];
                        let mut n2 = sn[tri[2].0 as usize];
                        if n0.length_squared() < 1e-10 {
                            n0 = face_n;
                        }
                        if n1.length_squared() < 1e-10 {
                            n1 = face_n;
                        }
                        if n2.length_squared() < 1e-10 {
                            n2 = face_n;
                        }
                        normals.extend([n0, n1, n2]);
                    } else {
                        normals.extend([face_n, face_n, face_n]);
                    }
                    colors.push(self.colors[tri[0].0 as usize]);
                    colors.push(self.colors[tri[1].0 as usize]);
                    colors.push(self.colors[tri[2].0 as usize]);
                    indices.extend([base, base + 1, base + 2]);
                }
            }
        };

        emit(
            &opaque_faces,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );
        let opaque_index_count = indices.len();
        emit(
            &xlucent_faces,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );

        BuiltMesh {
            positions,
            normals,
            colors,
            indices,
            opaque_index_count,
        }
    }

    fn averaged_vertex_normals(&self) -> Vec<Vec3> {
        let mut accum = vec![Vec3::ZERO; self.points.len()];
        for face in &self.faces {
            let tris: [[PointId; 3]; 2] = match face.len() {
                3 => [[face[0], face[1], face[2]], [face[0], face[0], face[0]]],
                4 => [[face[0], face[1], face[2]], [face[0], face[2], face[3]]],
                _ => continue,
            };
            let tri_count = if face.len() == 3 { 1 } else { 2 };
            for tri in tris.iter().take(tri_count) {
                let a = self.points[tri[0].0 as usize];
                let b = self.points[tri[1].0 as usize];
                let c = self.points[tri[2].0 as usize];
                let raw = (b - a).cross(c - a);
                if raw.length_squared() <= 0.0 {
                    continue;
                }
                // Area-weighted (unnormalized cross).
                accum[tri[0].0 as usize] += raw;
                accum[tri[1].0 as usize] += raw;
                accum[tri[2].0 as usize] += raw;
            }
        }
        accum
            .into_iter()
            .map(|n| {
                if n.length_squared() > 0.0 {
                    n.normalize()
                } else {
                    Vec3::Y
                }
            })
            .collect()
    }
}

/// Triangulated mesh with normals, ready for GPU upload (advanced / internal).
#[derive(Clone, Debug)]
pub struct BuiltMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub colors: Vec<Vec4>,
    pub indices: Vec<u32>,
    /// Indices `[0..opaque_index_count)` are opaque; the rest use alpha blending.
    pub opaque_index_count: usize,
}

impl BuiltMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub(crate) fn to_interleaved(&self) -> Vec<Vertex> {
        self.positions
            .iter()
            .zip(self.normals.iter())
            .zip(self.colors.iter())
            .map(|((p, n), c)| Vertex {
                position: (*p).into(),
                normal: (*n).into(),
                color: (*c).into(),
            })
            .collect()
    }

    pub fn append_translated(&mut self, other: &BuiltMesh, translation: Vec3) {
        // Preserve opaque-then-transparent ordering across both meshes.
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut indices = Vec::new();

        let push_range = |src: &BuiltMesh,
                          index_start: usize,
                          index_end: usize,
                          translation: Vec3,
                          positions: &mut Vec<Vec3>,
                          normals: &mut Vec<Vec3>,
                          colors: &mut Vec<Vec4>,
                          indices: &mut Vec<u32>| {
            let mut remap = HashMap::new();
            for &old in &src.indices[index_start..index_end] {
                let new = *remap.entry(old).or_insert_with(|| {
                    let i = old as usize;
                    let id = positions.len() as u32;
                    positions.push(src.positions[i] + translation);
                    normals.push(src.normals[i]);
                    colors.push(src.colors[i]);
                    id
                });
                indices.push(new);
            }
        };

        push_range(
            self,
            0,
            self.opaque_index_count,
            Vec3::ZERO,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );
        push_range(
            other,
            0,
            other.opaque_index_count,
            translation,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );
        let opaque_index_count = indices.len();
        push_range(
            self,
            self.opaque_index_count,
            self.indices.len(),
            Vec3::ZERO,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );
        push_range(
            other,
            other.opaque_index_count,
            other.indices.len(),
            translation,
            &mut positions,
            &mut normals,
            &mut colors,
            &mut indices,
        );

        self.positions = positions;
        self.normals = normals;
        self.colors = colors;
        self.indices = indices;
        self.opaque_index_count = opaque_index_count;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x4,
        ],
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct InstanceRaw {
    pub model: [[f32; 4]; 4],
}

impl InstanceRaw {
    pub fn from_matrix(m: glam::Mat4) -> Self {
        Self {
            model: m.to_cols_array_2d(),
        }
    }

    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            3 => Float32x4,
            4 => Float32x4,
            5 => Float32x4,
            6 => Float32x4,
        ],
    };
}
