//! glTF import (Quaternius-style packs and similar).

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::{AlbedoMap, BuiltMesh, Mesh};
use crate::place::Place;
use glam::{Mat4, Vec3, Vec4};
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

        validate_glb_path(&path)?;
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
            let mut colors = colors;
            let mut albedo = primitive_base_color_albedo(&primitive, images)?;
            if alpha_mode_needs_opaque_fallback(primitive.material().alpha_mode(), albedo.as_ref())
            {
                promote_blend_without_alpha(&mut colors, albedo.as_mut());
            }

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
            let Some(mut map) = primitive_base_color_albedo(&primitive, images)? else {
                continue;
            };
            if alpha_mode_needs_opaque_fallback(primitive.material().alpha_mode(), Some(&map)) {
                promote_blend_without_alpha(&mut [], Some(&mut map));
            }
            return Ok(Some(map));
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

/// Reject unknown or truncated GLB chunks before `gltf::import`.
/// JSON `.gltf` files (no `glTF` magic) are left for the glTF parser.
pub(crate) fn validate_glb_path(path: &Path) -> EngineResult<()> {
    let bytes = std::fs::read(path)?;
    validate_glb_bytes(&bytes)
}

pub(crate) fn validate_glb_bytes(bytes: &[u8]) -> EngineResult<()> {
    if bytes.len() < 4 || &bytes[..4] != b"glTF" {
        return Ok(());
    }
    if bytes.len() < 12 {
        return Err(EngineError::Model("truncated GLB header".into()));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 2 {
        return Err(EngineError::Model(format!(
            "unsupported GLB version {version} (expected 2)"
        )));
    }
    let declared_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if declared_len < 12 || declared_len > bytes.len() {
        return Err(EngineError::Model(format!(
            "truncated GLB: declared length {declared_len} (file is {} bytes)",
            bytes.len()
        )));
    }
    let mut offset = 12;
    while offset < declared_len {
        if declared_len - offset < 8 {
            return Err(EngineError::Model("truncated GLB chunk header".into()));
        }
        let chunk_len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let chunk_type = [
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ];
        if chunk_type != *b"JSON" && chunk_type != *b"BIN\0" {
            return Err(EngineError::Model(unknown_glb_chunk_message(chunk_type)));
        }
        if chunk_len % 4 != 0 {
            return Err(EngineError::Model(format!(
                "malformed GLB chunk {}: length {chunk_len} is not a multiple of 4",
                fourcc_label(chunk_type)
            )));
        }
        let payload_end = offset
            .checked_add(8)
            .and_then(|h| h.checked_add(chunk_len))
            .ok_or_else(|| EngineError::Model("malformed GLB chunk length".into()))?;
        if payload_end > declared_len {
            return Err(EngineError::Model(format!(
                "truncated GLB chunk {}: length {chunk_len} overruns file",
                fourcc_label(chunk_type)
            )));
        }
        offset = payload_end;
    }
    Ok(())
}

pub(crate) fn unknown_glb_chunk_message(ty: [u8; 4]) -> String {
    format!(
        "unknown GLB chunk type {} (fourcc 0x{:08X})",
        fourcc_label(ty),
        u32::from_le_bytes(ty)
    )
}

fn fourcc_label(ty: [u8; 4]) -> String {
    match std::str::from_utf8(&ty) {
        Ok(s) => format!("{s:?}"),
        Err(_) => format!("0x{:02X}{:02X}{:02X}{:02X}", ty[0], ty[1], ty[2], ty[3]),
    }
}

/// BLEND/MASK clothes whose baseColor has no real alpha become OPAQUE.
pub(crate) fn alpha_mode_needs_opaque_fallback(
    alpha_mode: gltf::material::AlphaMode,
    albedo: Option<&AlbedoMap>,
) -> bool {
    matches!(
        alpha_mode,
        gltf::material::AlphaMode::Blend | gltf::material::AlphaMode::Mask
    ) && !albedo_has_real_alpha(albedo)
}

pub(crate) fn albedo_has_real_alpha(albedo: Option<&AlbedoMap>) -> bool {
    let Some(map) = albedo else {
        return false;
    };
    map.rgba.chunks_exact(4).any(|px| px[3] < 255)
}

pub(crate) fn promote_blend_without_alpha(colors: &mut [Vec4], albedo: Option<&mut AlbedoMap>) {
    for c in colors.iter_mut() {
        c.w = 1.0;
    }
    if let Some(map) = albedo {
        for px in map.rgba.chunks_exact_mut(4) {
            px[3] = 255;
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
mod tests {
    use super::*;
    use crate::mesh::AlbedoMap;
    use glam::Vec4;

    #[test]
    fn unknown_glb_chunk_bytes_include_readable_fourcc() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"glTF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        let total = 12 + 8 + 4;
        bytes.extend_from_slice(&(total as u32).to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(b"BIN ");
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        let err = validate_glb_bytes(&bytes).expect_err("BIN<space> must fail");
        let msg = err.to_string();
        assert!(msg.contains("BIN "), "got {msg}");
        assert!(
            msg.contains("0x204E4942") || msg.contains("BIN "),
            "got {msg}"
        );
        match err {
            EngineError::Model(inner) => {
                assert_eq!(inner, unknown_glb_chunk_message(*b"BIN "));
            }
            other => panic!("expected Model, got {other:?}"),
        }
    }

    #[test]
    fn truncated_glb_is_a_hard_error() {
        let err = validate_glb_bytes(b"glTF\x02").expect_err("truncated header");
        assert!(matches!(err, EngineError::Model(_)));
        let mut bytes = Vec::from(&b"glTF"[..]);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(b"JSON");
        // declared 32 bytes but file stops after the chunk header
        let err = validate_glb_bytes(&bytes).expect_err("truncated payload");
        assert!(matches!(err, EngineError::Model(_)));
    }

    #[test]
    fn json_gltf_magic_is_skipped() {
        validate_glb_bytes(br#"{ "asset": { "version": "2.0" } }"#).unwrap();
    }

    #[test]
    fn blend_without_alpha_promotes_to_opaque() {
        let mut colors = vec![Vec4::new(1.0, 0.0, 0.0, 0.3)];
        let mut albedo = AlbedoMap {
            width: 1,
            height: 1,
            rgba: vec![10, 20, 30, 255],
        };
        assert!(alpha_mode_needs_opaque_fallback(
            gltf::material::AlphaMode::Blend,
            Some(&albedo)
        ));
        promote_blend_without_alpha(&mut colors, Some(&mut albedo));
        assert_eq!(colors[0].w, 1.0);
        assert_eq!(albedo.rgba[3], 255);
    }

    #[test]
    fn mask_without_texture_promotes_to_opaque() {
        assert!(alpha_mode_needs_opaque_fallback(
            gltf::material::AlphaMode::Mask,
            None
        ));
        let mut colors = vec![Vec4::new(0.2, 0.3, 0.4, 0.5)];
        promote_blend_without_alpha(&mut colors, None);
        assert_eq!(colors[0].w, 1.0);
    }

    #[test]
    fn blend_with_real_alpha_stays() {
        let albedo = AlbedoMap {
            width: 1,
            height: 1,
            rgba: vec![10, 20, 30, 128],
        };
        assert!(!alpha_mode_needs_opaque_fallback(
            gltf::material::AlphaMode::Blend,
            Some(&albedo)
        ));
    }

    #[test]
    fn opaque_mode_is_not_forced() {
        let albedo = AlbedoMap {
            width: 1,
            height: 1,
            rgba: vec![10, 20, 30, 255],
        };
        assert!(!alpha_mode_needs_opaque_fallback(
            gltf::material::AlphaMode::Opaque,
            Some(&albedo)
        ));
    }
}
