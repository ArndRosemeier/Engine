use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::place::ensure_finite3;
use glam::Vec3;

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
    colors: Vec<Vec3>,
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
        self.colors.push(Color::rgb(191, 191, 191).to_vec3());
        Ok(id)
    }

    /// Set the display color of an existing point.
    pub fn set_point_color(&mut self, id: PointId, color: Color) -> EngineResult<()> {
        let idx = id.0 as usize;
        if idx >= self.colors.len() {
            return Err(EngineError::InvalidMesh(format!(
                "point id {} is out of range",
                id.0
            )));
        }
        self.colors[idx] = color.to_vec3();
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

    pub fn add_quad(
        &mut self,
        a: PointId,
        b: PointId,
        c: PointId,
        d: PointId,
    ) -> EngineResult<()> {
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
    pub fn build(&self) -> BuiltMesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut colors = Vec::new();
        let mut indices = Vec::new();

        for face in &self.faces {
            let tris: [[PointId; 3]; 2] = match face.len() {
                3 => [[face[0], face[1], face[2]], [face[0], face[0], face[0]]],
                4 => [
                    [face[0], face[1], face[2]],
                    [face[0], face[2], face[3]],
                ],
                _ => unreachable!("add_face only allows 3 or 4 points"),
            };
            let tri_count = if face.len() == 3 { 1 } else { 2 };

            for tri in tris.iter().take(tri_count) {
                let a = self.points[tri[0].0 as usize];
                let b = self.points[tri[1].0 as usize];
                let c = self.points[tri[2].0 as usize];
                let n = {
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
                normals.extend([n, n, n]);
                colors.push(self.colors[tri[0].0 as usize]);
                colors.push(self.colors[tri[1].0 as usize]);
                colors.push(self.colors[tri[2].0 as usize]);
                indices.extend([base, base + 1, base + 2]);
            }
        }

        BuiltMesh {
            positions,
            normals,
            colors,
            indices,
        }
    }
}

/// Triangulated mesh with normals, ready for GPU upload (advanced / internal).
#[derive(Clone, Debug)]
pub struct BuiltMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub colors: Vec<Vec3>,
    pub indices: Vec<u32>,
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
        let base = self.positions.len() as u32;
        self.positions
            .extend(other.positions.iter().map(|p| *p + translation));
        self.normals.extend_from_slice(&other.normals);
        self.colors.extend_from_slice(&other.colors);
        self.indices
            .extend(other.indices.iter().map(|i| i + base));
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x3,
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
