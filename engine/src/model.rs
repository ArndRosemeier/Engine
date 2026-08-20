//! glTF import (Quaternius-style packs and similar).

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::{AlbedoMap, BuiltMesh, Mesh};
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

        let (document, buffers, images) =
            gltf::import(&path).map_err(|e| EngineError::Model(e.to_string()))?;

        let total_buf: u64 = buffers.iter().map(|b| b.0.len() as u64).sum();
        if total_buf > limits.max_gltf_buffer_bytes {
            return Err(EngineError::ResourceLimit(format!(
                "glTF buffers are {total_buf} bytes (limit {})",
                limits.max_gltf_buffer_bytes
            )));
        }

        let built = load_gltf_document(&document, &buffers, &images, limits)?;
        let mut mesh = built_to_mesh(&built)?;
        if let Some(map) = first_base_color_albedo(&document, &images)? {
            mesh.set_albedo_rgba(map.width, map.height, map.rgba)?;
        }
        Ok(mesh)
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
    images: &[gltf::image::Data],
    limits: &EngineLimits,
) -> EngineResult<BuiltMesh> {
    let mut combined = BuiltMesh {
        positions: Vec::new(),
        normals: Vec::new(),
        colors: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        opaque_index_count: 0,
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

            let colors: Vec<glam::Vec4> = if let Some(iter) = reader.read_colors(0) {
                match iter {
                    gltf::mesh::util::ReadColors::RgbU8(i) => i
                        .map(|c| {
                            glam::Vec4::new(
                                c[0] as f32 / 255.0,
                                c[1] as f32 / 255.0,
                                c[2] as f32 / 255.0,
                                1.0,
                            )
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbU16(i) => i
                        .map(|c| {
                            glam::Vec4::new(
                                c[0] as f32 / 65535.0,
                                c[1] as f32 / 65535.0,
                                c[2] as f32 / 65535.0,
                                1.0,
                            )
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbF32(i) => {
                        i.map(|c| glam::Vec4::new(c[0], c[1], c[2], 1.0)).collect()
                    }
                    gltf::mesh::util::ReadColors::RgbaU8(i) => i
                        .map(|c| {
                            glam::Vec4::new(
                                c[0] as f32 / 255.0,
                                c[1] as f32 / 255.0,
                                c[2] as f32 / 255.0,
                                c[3] as f32 / 255.0,
                            )
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaU16(i) => i
                        .map(|c| {
                            glam::Vec4::new(
                                c[0] as f32 / 65535.0,
                                c[1] as f32 / 65535.0,
                                c[2] as f32 / 65535.0,
                                c[3] as f32 / 65535.0,
                            )
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaF32(i) => {
                        i.map(|c| glam::Vec4::new(c[0], c[1], c[2], c[3])).collect()
                    }
                }
            } else {
                vec![glam::Vec4::new(base.x, base.y, base.z, 1.0); positions.len()]
            };
            let albedo = primitive_base_color_albedo(&primitive, images)?;
            reject_alpha_mode_without_real_alpha(primitive.material().alpha_mode(), &albedo)?;

            let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
                iter.into_u32().collect()
            } else {
                (0..positions.len() as u32).collect()
            };

            let uvs: Vec<[f32; 2]> = if let Some(iter) = reader.read_tex_coords(0) {
                match iter {
                    gltf::mesh::util::ReadTexCoords::U8(i) => i
                        .map(|t| [t[0] as f32 / 255.0, t[1] as f32 / 255.0])
                        .collect(),
                    gltf::mesh::util::ReadTexCoords::U16(i) => i
                        .map(|t| [t[0] as f32 / 65535.0, t[1] as f32 / 65535.0])
                        .collect(),
                    gltf::mesh::util::ReadTexCoords::F32(i) => i.collect(),
                }
            } else {
                vec![[0.0, 0.0]; positions.len()]
            };

            let part = BuiltMesh {
                positions: positions.into_iter().map(Vec3::from).collect(),
                normals,
                colors,
                uvs,
                opaque_index_count: indices.len(),
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
        mesh.set_point_color(id, Color::rgba01(c.x, c.y, c.z, c.w).expect("color"))?;
        if let Some(uv) = built.uvs.get(i) {
            mesh.set_point_uv(id, *uv)?;
        }
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

fn first_base_color_albedo(
    document: &gltf::Document,
    images: &[gltf::image::Data],
) -> EngineResult<Option<AlbedoMap>> {
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(map) = primitive_base_color_albedo(&primitive, images)? {
                return Ok(Some(map));
            }
        }
    }
    Ok(None)
}

fn primitive_base_color_albedo(
    primitive: &gltf::Primitive<'_>,
    images: &[gltf::image::Data],
) -> EngineResult<Option<AlbedoMap>> {
    let Some(info) = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
    else {
        return Ok(None);
    };
    let index = info.texture().source().index();
    let Some(image) = images.get(index) else {
        return Err(EngineError::Model(format!(
            "glTF baseColorTexture index {index} is missing"
        )));
    };
    Ok(Some(image_to_albedo(image)?))
}

pub(crate) fn image_to_albedo(image: &gltf::image::Data) -> EngineResult<AlbedoMap> {
    let rgba = match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity(image.pixels.len() / 3 * 4);
            for chunk in image.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            rgba
        }
        gltf::image::Format::R8 => image
            .pixels
            .iter()
            .flat_map(|v| [*v, *v, *v, 255])
            .collect(),
        other => {
            return Err(EngineError::Model(format!(
                "unsupported glTF image format {other:?}"
            )))
        }
    };
    Ok(AlbedoMap {
        width: image.width,
        height: image.height,
        rgba,
    })
}

/// True when some albedo texel has A < 255.
/// RGB images padded to A=255, and a missing texture, count as no real alpha.
pub(crate) fn albedo_has_real_alpha(albedo: &Option<AlbedoMap>) -> bool {
    albedo
        .as_ref()
        .is_some_and(|map| map.rgba.chunks_exact(4).any(|px| px[3] < 255))
}

pub(crate) fn reject_alpha_mode_without_real_alpha(
    alpha_mode: gltf::material::AlphaMode,
    albedo: &Option<AlbedoMap>,
) -> EngineResult<()> {
    match alpha_mode {
        gltf::material::AlphaMode::Opaque => Ok(()),
        gltf::material::AlphaMode::Blend | gltf::material::AlphaMode::Mask => {
            if albedo_has_real_alpha(albedo) {
                return Ok(());
            }
            let mode = match alpha_mode {
                gltf::material::AlphaMode::Blend => "BLEND",
                gltf::material::AlphaMode::Mask => "MASK",
                gltf::material::AlphaMode::Opaque => unreachable!(),
            };
            let why = if albedo.is_none() {
                "no alpha channel on the baseColor texture"
            } else {
                "every albedo alpha byte is 255"
            };
            Err(EngineError::Model(format!(
                "alphaMode {mode} without real alpha ({why})"
            )))
        }
    }
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
        .map(|(i, p)| Place {
            position: *p,
            yaw_degrees: y_rotation_degrees(i),
            pitch_degrees: 0.0,
            scale,
            stretch: Vec3::ONE,
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

#[cfg(test)]
pub(crate) fn test_glb_with_bin_space_fourcc() -> Vec<u8> {
    let json = br#"{"asset":{"version":"2.0"}}"#;
    let json_len = (json.len() + 3) & !3;
    let json_pad = json_len - json.len();
    let bin = [1_u8, 2, 3, 4];
    let total = 12 + 8 + json_len + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_len as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json);
    out.extend(std::iter::repeat(b' ').take(json_pad));
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN ");
    out.extend_from_slice(&bin);
    out
}

#[cfg(test)]
fn write_rgb_png(path: &Path) {
    let img = image::RgbImage::from_pixel(1, 1, image::Rgb([180, 90, 40]));
    img.save(path).expect("write RGB PNG");
}

#[cfg(test)]
fn write_triangle_bin(path: &Path, skinned: bool) {
    let mut bin = Vec::new();
    for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for c in p {
            bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    for i in [0u16, 1, 2] {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    if skinned {
        bin.extend_from_slice(&[0, 0]); // pad to 44
        for _ in 0..3 {
            bin.extend_from_slice(&[0u8, 0, 0, 0]);
        }
        for _ in 0..3 {
            for w in [1.0f32, 0.0, 0.0, 0.0] {
                bin.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    std::fs::write(path, bin).expect("write mesh.bin");
}

#[cfg(test)]
pub(crate) fn write_minimal_static_gltf(dir: &Path, alpha_mode: &str, with_rgb_texture: bool) {
    std::fs::create_dir_all(dir).expect("temp dir");
    write_triangle_bin(&dir.join("mesh.bin"), false);
    let material = if with_rgb_texture {
        write_rgb_png(&dir.join("albedo.png"));
        format!(
            r#"{{"alphaMode":"{alpha_mode}","pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}"#
        )
    } else {
        format!(r#"{{"alphaMode":"{alpha_mode}"}}"#)
    };
    let images_textures = if with_rgb_texture {
        r#","textures":[{"source":0}],"images":[{"uri":"albedo.png"}]"#
    } else {
        ""
    };
    let json = format!(
        r#"{{
  "asset":{{"version":"2.0"}},
  "scene":0,
  "scenes":[{{"nodes":[0]}}],
  "nodes":[{{"mesh":0}}],
  "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],
  "materials":[{material}]{images_textures},
  "accessors":[
    {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","max":[1,1,0],"min":[0,0,0]}},
    {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}
  ],
  "bufferViews":[
    {{"buffer":0,"byteOffset":0,"byteLength":36}},
    {{"buffer":0,"byteOffset":36,"byteLength":6}}
  ],
  "buffers":[{{"byteLength":42,"uri":"mesh.bin"}}]
}}"#
    );
    std::fs::write(dir.join("model.gltf"), json).expect("write gltf");
}

#[cfg(test)]
pub(crate) fn write_minimal_skinned_gltf(dir: &Path, alpha_mode: &str, with_rgb_texture: bool) {
    std::fs::create_dir_all(dir).expect("temp dir");
    write_triangle_bin(&dir.join("mesh.bin"), true);
    let material = if with_rgb_texture {
        write_rgb_png(&dir.join("albedo.png"));
        format!(
            r#"{{"alphaMode":"{alpha_mode}","pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}"#
        )
    } else {
        format!(r#"{{"alphaMode":"{alpha_mode}"}}"#)
    };
    let images_textures = if with_rgb_texture {
        r#","textures":[{"source":0}],"images":[{"uri":"albedo.png"}]"#
    } else {
        ""
    };
    let json = format!(
        r#"{{
  "asset":{{"version":"2.0"}},
  "scene":0,
  "scenes":[{{"nodes":[0]}}],
  "nodes":[{{"mesh":0,"skin":0,"children":[1]}},{{"name":"joint"}}],
  "skins":[{{"joints":[1]}}],
  "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"JOINTS_0":2,"WEIGHTS_0":3}},"indices":1,"material":0}}]}}],
  "materials":[{material}]{images_textures},
  "accessors":[
    {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","max":[1,1,0],"min":[0,0,0]}},
    {{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}},
    {{"bufferView":2,"componentType":5121,"count":3,"type":"VEC4"}},
    {{"bufferView":3,"componentType":5126,"count":3,"type":"VEC4"}}
  ],
  "bufferViews":[
    {{"buffer":0,"byteOffset":0,"byteLength":36}},
    {{"buffer":0,"byteOffset":36,"byteLength":6}},
    {{"buffer":0,"byteOffset":44,"byteLength":12}},
    {{"buffer":0,"byteOffset":56,"byteLength":48}}
  ],
  "buffers":[{{"byteLength":104,"uri":"mesh.bin"}}]
}}"#
    );
    std::fs::write(dir.join("model.gltf"), json).expect("write gltf");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::AlbedoMap;

    #[test]
    fn albedo_has_real_alpha_treats_padded_rgb_as_none() {
        assert!(!albedo_has_real_alpha(&None));
        assert!(!albedo_has_real_alpha(&Some(AlbedoMap {
            width: 1,
            height: 1,
            rgba: vec![10, 20, 30, 255],
        })));
        assert!(albedo_has_real_alpha(&Some(AlbedoMap {
            width: 1,
            height: 1,
            rgba: vec![10, 20, 30, 200],
        })));
    }
}
