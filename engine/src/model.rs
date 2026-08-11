//! glTF import (Quaternius-style packs and similar).

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::{BuiltMesh, Mesh};
use crate::place::Place;
use glam::{Mat4, Vec3};
use std::path::{Path, PathBuf};

/// Friendly model loader with path + size limits.
pub struct Model;

impl Model {
    /// Load a glTF/GLB file into a [`Mesh`].
    ///
    /// The path must resolve under `base_dir` (default: current directory).
    pub fn load(path: impl AsRef<Path>) -> EngineResult<Mesh> {
        Self::load_with(path, PathBuf::from("."), &EngineLimits::default())
    }

    pub fn load_with(
        path: impl AsRef<Path>,
        base_dir: impl AsRef<Path>,
        limits: &EngineLimits,
    ) -> EngineResult<Mesh> {
        let path = resolve_allowed_path(path.as_ref(), base_dir.as_ref())?;
        let meta = std::fs::metadata(&path)?;
        if meta.len() > limits.max_gltf_buffer_bytes {
            return Err(EngineError::ResourceLimit(format!(
                "glTF file is {} bytes (limit {})",
                meta.len(),
                limits.max_gltf_buffer_bytes
            )));
        }

        let (document, buffers, _images) =
            gltf::import(&path).map_err(|e| EngineError::Model(e.to_string()))?;

        let total_buf: u64 = buffers.iter().map(|b| b.0.len() as u64).sum();
        if total_buf > limits.max_gltf_buffer_bytes {
            return Err(EngineError::ResourceLimit(format!(
                "glTF buffers are {total_buf} bytes (limit {})",
                limits.max_gltf_buffer_bytes
            )));
        }

        let built = load_gltf_document(&document, &buffers, limits)?;
        built_to_mesh(&built)
    }
}

fn resolve_allowed_path(path: &Path, base_dir: &Path) -> EngineResult<PathBuf> {
    let base = std::fs::canonicalize(base_dir).unwrap_or_else(|_| base_dir.to_path_buf());
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let canonical = std::fs::canonicalize(&candidate).map_err(|e| {
        EngineError::Io(std::io::Error::new(
            e.kind(),
            format!("{} ({})", e, candidate.display()),
        ))
    })?;
    if !canonical.starts_with(&base) {
        return Err(EngineError::PathNotAllowed(format!(
            "{} is outside allowed root {}",
            canonical.display(),
            base.display()
        )));
    }
    Ok(canonical)
}

fn load_gltf_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    limits: &EngineLimits,
) -> EngineResult<BuiltMesh> {
    let mut combined = BuiltMesh {
        positions: Vec::new(),
        normals: Vec::new(),
        colors: Vec::new(),
        indices: Vec::new(),
    };

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| EngineError::Model("glTF primitive missing POSITION".into()))?
                .collect();
            if positions.is_empty() {
                continue;
            }

            let normals: Vec<Vec3> = if let Some(iter) = reader.read_normals() {
                iter.map(Vec3::from).collect()
            } else {
                vec![Vec3::Y; positions.len()]
            };

            let base_color = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();
            let base = Vec3::new(base_color[0], base_color[1], base_color[2]);

            let colors: Vec<Vec3> = if let Some(iter) = reader.read_colors(0) {
                match iter {
                    gltf::mesh::util::ReadColors::RgbU8(i) => i
                        .map(|c| {
                            Vec3::new(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0)
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbU16(i) => i
                        .map(|c| {
                            Vec3::new(
                                c[0] as f32 / 65535.0,
                                c[1] as f32 / 65535.0,
                                c[2] as f32 / 65535.0,
                            )
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbF32(i) => {
                        i.map(|c| Vec3::new(c[0], c[1], c[2])).collect()
                    }
                    gltf::mesh::util::ReadColors::RgbaU8(i) => i
                        .map(|c| {
                            Vec3::new(c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0)
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaU16(i) => i
                        .map(|c| {
                            Vec3::new(
                                c[0] as f32 / 65535.0,
                                c[1] as f32 / 65535.0,
                                c[2] as f32 / 65535.0,
                            )
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaF32(i) => {
                        i.map(|c| Vec3::new(c[0], c[1], c[2])).collect()
                    }
                }
            } else {
                vec![base; positions.len()]
            };

            let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
                iter.into_u32().collect()
            } else {
                (0..positions.len() as u32).collect()
            };

            let part = BuiltMesh {
                positions: positions.into_iter().map(Vec3::from).collect(),
                normals,
                colors,
                indices,
            };
            combined.append_translated(&part, Vec3::ZERO);

            if combined.triangle_count() as u64 > limits.max_model_triangles {
                return Err(EngineError::ResourceLimit(format!(
                    "model exceeds {} triangles",
                    limits.max_model_triangles
                )));
            }
        }
    }

    if combined.indices.is_empty() {
        return Err(EngineError::Model(
            "glTF has no triangle mesh primitives".into(),
        ));
    }

    recompute_normals_if_needed(&mut combined);
    Ok(combined)
}

fn recompute_normals_if_needed(mesh: &mut BuiltMesh) {
    let mostly_default = mesh.normals.iter().all(|n| n.distance(Vec3::Y) < 1e-5);
    if !mostly_default {
        return;
    }
    mesh.normals.fill(Vec3::ZERO);
    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize];
        let b = mesh.positions[tri[1] as usize];
        let c = mesh.positions[tri[2] as usize];
        let n = (b - a).cross(c - a);
        if n.length_squared() > 0.0 {
            let n = n.normalize();
            mesh.normals[tri[0] as usize] += n;
            mesh.normals[tri[1] as usize] += n;
            mesh.normals[tri[2] as usize] += n;
        }
    }
    for n in &mut mesh.normals {
        if n.length_squared() > 0.0 {
            *n = n.normalize();
        } else {
            *n = Vec3::Y;
        }
    }
}

fn built_to_mesh(built: &BuiltMesh) -> EngineResult<Mesh> {
    let mut mesh = Mesh::new();
    let mut ids = Vec::with_capacity(built.positions.len());
    for (i, p) in built.positions.iter().enumerate() {
        let id = mesh.add_point(*p)?;
        let c = built.colors[i];
        mesh.set_point_color(id, Color::rgb01_unchecked(c.x, c.y, c.z))?;
        ids.push(id);
    }
    for tri in built.indices.chunks_exact(3) {
        mesh.add_face(&[
            ids[tri[0] as usize],
            ids[tri[1] as usize],
            ids[tri[2] as usize],
        ])?;
    }
    Ok(mesh)
}

/// Build [`Place`]s for scattering a model through the world.
pub fn scatter_places(
    positions: &[Vec3],
    scale: f32,
    y_rotation_degrees: impl Fn(usize) -> f32,
) -> Vec<Place> {
    positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            Place {
                position: *p,
                yaw_degrees: y_rotation_degrees(i),
                scale,
            }
        })
        .collect()
}

/// Legacy helper returning matrices (advanced).
pub fn scatter_transforms(
    positions: &[Vec3],
    scale: f32,
    y_rotation_degrees: impl Fn(usize) -> f32,
) -> Vec<Mat4> {
    scatter_places(positions, scale, y_rotation_degrees)
        .into_iter()
        .map(Place::to_matrix)
        .collect()
}
