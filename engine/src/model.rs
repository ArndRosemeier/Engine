//! glTF import (Quaternius-style packs and similar).

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::{AlbedoMap, BuiltMesh, MaterialMaps, Mesh, SurfaceMaterial};
use crate::place::Place;
use glam::{Mat4, Vec3};
use std::path::{Path, PathBuf};

/// Friendly model loader with path + size limits.
pub struct Model;

/// A static glTF model split by material: one [`Mesh`] per glTF material,
/// each with its own albedo and [`MaterialMaps`], in the model's own space
/// (node transforms applied). Spawn every part with the same place.
#[derive(Clone, Debug, Default)]
pub struct ModelScene {
    pub parts: Vec<Mesh>,
}

impl ModelScene {
    pub fn triangle_count(&self) -> usize {
        self.parts.iter().map(|m| m.build().triangle_count()).sum()
    }

    /// Box around every part, or `None` for an empty model.
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        self.parts
            .iter()
            .filter_map(Mesh::bounds)
            .reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))
    }
}

impl Model {
    /// Load a glTF/GLB file into a [`Mesh`].
    ///
    /// The path must resolve under `base_dir` (default: current directory).
    pub fn load(path: impl AsRef<Path>) -> EngineResult<Mesh> {
        Self::load_with(path, PathBuf::from("."), &EngineLimits::default())
    }

    /// Load a glTF/GLB file as one mesh per material, with full PBR maps
    /// (base colour, normal, metallic-roughness, occlusion, emissive).
    pub fn load_scene(path: impl AsRef<Path>) -> EngineResult<ModelScene> {
        Self::load_scene_with(path, PathBuf::from("."), &EngineLimits::default())
    }

    pub fn load_scene_with(
        path: impl AsRef<Path>,
        base_dir: impl AsRef<Path>,
        limits: &EngineLimits,
    ) -> EngineResult<ModelScene> {
        let (document, buffers, images) = import_limited(path.as_ref(), base_dir.as_ref(), limits)?;
        load_scene_document(&document, &buffers, &images, limits)
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
        mesh.set_surface_overrides(built.surface_overrides);
        Ok(mesh)
    }
}

fn resolve_allowed_path(path: &Path, base_dir: &Path) -> EngineResult<PathBuf> {
    let base = std::fs::canonicalize(base_dir).map_err(|error| {
        EngineError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to canonicalize model sandbox root {}: {error}",
                base_dir.display()
            ),
        ))
    })?;
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

fn imported_surface_material(name: Option<&str>) -> SurfaceMaterial {
    let name = name.unwrap_or_default().to_ascii_lowercase();
    if name.contains("foliage") || name.contains("leaf") || name.contains("crown") {
        if name.contains("needle") || name.contains("pine") || name.contains("spruce") {
            return SurfaceMaterial::NEEDLED_FOLIAGE;
        }
        return SurfaceMaterial::FOLIAGE;
    }
    if name.contains("bark") || name.contains("wood") || name.contains("trunk") {
        return SurfaceMaterial::WOOD;
    }
    SurfaceMaterial::DEFAULT
}

fn load_gltf_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    limits: &EngineLimits,
) -> EngineResult<BuiltMesh> {
    let mut combined = BuiltMesh {
        elements: Vec::new(),
        positions: Vec::new(),
        normals: Vec::new(),
        colors: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
        opaque_index_count: 0,
        surface: Default::default(),
        surface_overrides: Vec::new(),
    };

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let mut positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| EngineError::Model("glTF primitive missing POSITION".into()))?
                .collect();
            if positions.is_empty() {
                continue;
            }

            let mut normals: Vec<Vec3> = if let Some(iter) = reader.read_normals() {
                iter.map(Vec3::from).collect()
            } else {
                vec![Vec3::Y; positions.len()]
            };
            if let Some((dp, dn)) = morph_deltas(&mesh, &reader, positions.len()) {
                for (p, d) in positions.iter_mut().zip(&dp) {
                    *p = (Vec3::from(*p) + *d).to_array();
                }
                for (v, d) in normals.iter_mut().zip(&dn) {
                    *v = (*v + *d).normalize_or_zero();
                }
            }

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
            let mut albedo = primitive_base_color_albedo(&primitive, images)?;
            apply_alpha_mode(&primitive.material(), &mut albedo);

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

            let material = imported_surface_material(primitive.material().name());
            let vertex_count = positions.len();
            let part = BuiltMesh {
                elements: Vec::new(),
                positions: positions.into_iter().map(Vec3::from).collect(),
                // Primitive vertex count is retained before positions is moved.
                normals,
                colors,
                uvs,
                opaque_index_count: indices.len(),
                indices,
                surface: material,
                surface_overrides: vec![material; vertex_count],
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
        if let Some(n) = built.normals.get(i) {
            mesh.set_point_normal(id, *n)?;
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

/// The mesh's default morph target weights (glTF `mesh.weights`) applied to
/// one primitive: position and normal offsets per vertex, or `None` when no
/// target has weight. Exporters store a shaped character (body sliders, a
/// resting expression) this way; animated weights are not supported.
pub(crate) fn morph_deltas<'a, 's, F>(
    mesh: &gltf::Mesh<'_>,
    reader: &gltf::mesh::Reader<'a, 's, F>,
    vertex_count: usize,
) -> Option<(Vec<Vec3>, Vec<Vec3>)>
where
    F: Clone + Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    let weights = mesh.weights()?;
    if weights.iter().all(|w| *w == 0.0 || !w.is_finite()) {
        return None;
    }
    let mut dp = vec![Vec3::ZERO; vertex_count];
    let mut dn = vec![Vec3::ZERO; vertex_count];
    for (w, (pos, nrm, _)) in weights.iter().zip(reader.read_morph_targets()) {
        if *w == 0.0 || !w.is_finite() {
            continue;
        }
        if let Some(pos) = pos {
            for (d, p) in dp.iter_mut().zip(pos) {
                *d += Vec3::from(p) * *w;
            }
        }
        if let Some(nrm) = nrm {
            for (d, n) in dn.iter_mut().zip(nrm) {
                *d += Vec3::from(n) * *w;
            }
        }
    }
    Some((dp, dn))
}

/// Makes the albedo alpha follow the material's `alphaMode`, as glTF says:
/// OPAQUE ignores alpha (a texture with an alpha channel never cuts holes in
/// a body), MASK is fully opaque at or above `alphaCutoff` and fully clear
/// below it, BLEND keeps the alpha. A BLEND or MASK material without real
/// alpha (exporters mark many uniform materials BLEND) simply draws opaque.
pub(crate) fn apply_alpha_mode(material: &gltf::Material<'_>, albedo: &mut Option<AlbedoMap>) {
    let Some(map) = albedo.as_mut() else {
        return;
    };
    match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => {
            for px in map.rgba.chunks_exact_mut(4) {
                px[3] = 255;
            }
        }
        gltf::material::AlphaMode::Mask => {
            let cutoff = material.alpha_cutoff().unwrap_or(0.5).clamp(0.0, 1.0) * 255.0;
            for px in map.rgba.chunks_exact_mut(4) {
                px[3] = if px[3] as f32 >= cutoff { 255 } else { 0 };
            }
        }
        gltf::material::AlphaMode::Blend => {}
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
            roll_degrees: 0.0,
            scale,
            stretch: Vec3::ONE,
            tint: crate::color::Color::WHITE,
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
    out.extend(std::iter::repeat_n(b' ', json_pad));
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

/// `gltf::import` inside the sandbox root and the size limits.
pub(crate) fn import_limited(
    path: &Path,
    base_dir: &Path,
    limits: &EngineLimits,
) -> EngineResult<(
    gltf::Document,
    Vec<gltf::buffer::Data>,
    Vec<gltf::image::Data>,
)> {
    let path = resolve_allowed_path(path, base_dir)?;
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
    Ok((document, buffers, images))
}

/// One material's geometry, gathered from every primitive that uses it.
#[derive(Default)]
struct Part {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    colors: Vec<glam::Vec4>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

fn load_scene_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    limits: &EngineLimits,
) -> EngineResult<ModelScene> {
    // Walk the default scene (or every root) with world transforms.
    let roots: Vec<gltf::Node<'_>> = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => scene.nodes().collect(),
        None => document.nodes().collect(),
    };
    // Depth first in document order, so parts come out in a stable order.
    let mut stack: Vec<(gltf::Node<'_>, Mat4)> = roots
        .into_iter()
        .rev()
        .map(|n| (n, Mat4::IDENTITY))
        .collect();
    // Keyed by material index; `None` is glTF's default material.
    let mut parts: Vec<(Option<usize>, gltf::Material<'_>, Part)> = Vec::new();
    let mut triangles = 0u64;
    while let Some((node, parent)) = stack.pop() {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        let children: Vec<_> = node.children().collect();
        stack.extend(children.into_iter().rev().map(|c| (c, world)));
        let Some(mesh) = node.mesh() else { continue };
        let normal_matrix = glam::Mat3::from_mat4(world).inverse().transpose();
        // A mirroring transform flips the winding; turn triangles back round.
        let mirrored = world.determinant() < 0.0;
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
            let Some(positions) = reader.read_positions() else {
                return Err(EngineError::Model("glTF primitive missing POSITION".into()));
            };
            let mut local: Vec<Vec3> = positions.map(Vec3::from).collect();
            if local.is_empty() {
                continue;
            }
            let n = local.len();
            let mut local_normals: Option<Vec<Vec3>> =
                reader.read_normals().map(|i| i.map(Vec3::from).collect());
            if let Some((dp, dn)) = morph_deltas(&mesh, &reader, n) {
                local.iter_mut().zip(&dp).for_each(|(p, d)| *p += *d);
                if let Some(ns) = &mut local_normals {
                    ns.iter_mut().zip(&dn).for_each(|(v, d)| *v += *d);
                }
            }
            let positions: Vec<Vec3> = local.iter().map(|p| world.transform_point3(*p)).collect();
            let normals: Vec<Vec3> = match local_normals {
                Some(ns) => ns
                    .into_iter()
                    .map(|v| (normal_matrix * v).normalize_or_zero())
                    .collect(),
                None => vec![Vec3::ZERO; n],
            };
            let material = primitive.material();
            let factor = glam::Vec4::from(material.pbr_metallic_roughness().base_color_factor());
            let colors: Vec<glam::Vec4> = match reader.read_colors(0) {
                Some(iter) => iter
                    .into_rgba_f32()
                    .map(|c| glam::Vec4::from(c) * factor)
                    .collect(),
                None => vec![factor; n],
            };
            let set = material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map_or(0, |t| t.tex_coord());
            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(set) {
                Some(iter) => iter.into_f32().collect(),
                None => vec![[0.0, 0.0]; n],
            };
            let mut indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..n as u32).collect(),
            };
            if normals.len() != n || colors.len() != n || uvs.len() != n {
                return Err(EngineError::Model(
                    "glTF attribute lengths differ from POSITION".into(),
                ));
            }
            if let Some(bad) = indices.iter().find(|&&i| i as usize >= n) {
                return Err(EngineError::Model(format!(
                    "glTF index {bad} out of range ({n} vertices)"
                )));
            }
            indices.truncate(indices.len() / 3 * 3);
            if mirrored {
                for tri in indices.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
            triangles += indices.len() as u64 / 3;
            if triangles > limits.max_model_triangles {
                return Err(EngineError::ResourceLimit(format!(
                    "model exceeds {} triangles",
                    limits.max_model_triangles
                )));
            }
            let key = material.index();
            let part = match parts.iter().position(|(k, _, _)| *k == key) {
                Some(i) => &mut parts[i].2,
                None => {
                    parts.push((key, material, Part::default()));
                    &mut parts.last_mut().expect("just pushed").2
                }
            };
            let base = part.positions.len() as u32;
            part.positions.extend(positions);
            part.normals.extend(normals);
            part.colors.extend(colors);
            part.uvs.extend(uvs);
            part.indices.extend(indices.into_iter().map(|i| i + base));
        }
    }
    if parts.is_empty() {
        return Err(EngineError::Model(
            "glTF has no triangle mesh primitives".into(),
        ));
    }

    let mut out = Vec::with_capacity(parts.len());
    for (_, material, part) in parts {
        out.push(part_to_mesh(part, &material, images)?);
    }
    Ok(ModelScene { parts: out })
}

fn part_to_mesh(
    part: Part,
    material: &gltf::Material<'_>,
    images: &[gltf::image::Data],
) -> EngineResult<Mesh> {
    let mut albedo = match material.pbr_metallic_roughness().base_color_texture() {
        Some(info) => Some(image_to_albedo(texture_image(images, info.texture())?)?),
        None => None,
    };
    apply_alpha_mode(material, &mut albedo);

    let mut surface = imported_surface_material(material.name());
    if surface == SurfaceMaterial::DEFAULT {
        let pbr = material.pbr_metallic_roughness();
        surface.roughness = pbr.roughness_factor();
        surface.metallic = pbr.metallic_factor();
    }

    let mut mesh = Mesh::new();
    let mut ids = Vec::with_capacity(part.positions.len());
    let smooth = part.normals.iter().all(|n| *n != Vec3::ZERO);
    for i in 0..part.positions.len() {
        let id = mesh.add_point(part.positions[i])?;
        let c = part.colors[i].clamp(glam::Vec4::ZERO, glam::Vec4::ONE);
        mesh.set_point_color(id, Color::rgba01(c.x, c.y, c.z, c.w)?)?;
        mesh.set_point_uv(id, part.uvs[i])?;
        if smooth {
            mesh.set_point_normal(id, part.normals[i])?;
        }
        ids.push(id);
    }
    for tri in part.indices.chunks_exact(3) {
        let (a, b, c) = (
            ids[tri[0] as usize],
            ids[tri[1] as usize],
            ids[tri[2] as usize],
        );
        if a == b || b == c || a == c {
            continue;
        }
        mesh.add_triangle(a, b, c)?;
    }
    mesh.set_surface_material(surface);
    if let Some(map) = albedo {
        mesh.set_albedo_rgba(map.width, map.height, map.rgba)?;
    }
    // Neutral sets are dropped, so plain models stay on the cheap path.
    mesh.set_material_maps(material_maps(material, images)?)?;
    Ok(mesh)
}

fn texture_image<'a>(
    images: &'a [gltf::image::Data],
    texture: gltf::Texture<'_>,
) -> EngineResult<&'a gltf::image::Data> {
    let index = texture.source().index();
    images
        .get(index)
        .ok_or_else(|| EngineError::Model(format!("glTF image {index} is missing")))
}

/// Normal, occlusion-roughness-metallic and emissive maps of a glTF material.
///
/// Occlusion and metallic-roughness are often one texture (ORM); when they
/// are separate the occlusion is resampled into the red channel of the
/// metallic-roughness map, so the shader always reads one packed texture.
/// Scale and offset of `KHR_texture_transform` on a normal texture (rotation
/// is not supported and left out).
fn normal_uv_transform(info: &gltf::material::NormalTexture<'_>) -> [f32; 4] {
    let mut uv = [1.0, 1.0, 0.0, 0.0];
    let Some(t) = info.extension_value("KHR_texture_transform") else {
        return uv;
    };
    let pair = |key: &str| -> Option<[f32; 2]> {
        let a = t.get(key)?.as_array()?;
        Some([a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32])
    };
    if let Some(s) = pair("scale").filter(|s| s.iter().all(|v| v.is_finite())) {
        uv[0] = s[0];
        uv[1] = s[1];
    }
    if let Some(o) = pair("offset").filter(|o| o.iter().all(|v| v.is_finite())) {
        uv[2] = o[0];
        uv[3] = o[1];
    }
    uv
}

pub(crate) fn material_maps(
    material: &gltf::Material<'_>,
    images: &[gltf::image::Data],
) -> EngineResult<MaterialMaps> {
    let pbr = material.pbr_metallic_roughness();
    let mut maps = MaterialMaps {
        roughness: pbr.roughness_factor(),
        metallic: pbr.metallic_factor(),
        ..MaterialMaps::default()
    };
    if let Some(info) = material.normal_texture() {
        maps.normal = Some(image_to_albedo(texture_image(images, info.texture())?)?);
        maps.normal_scale = info.scale();
        maps.normal_uv = normal_uv_transform(&info);
    }
    let mr = pbr.metallic_roughness_texture();
    let occlusion = material.occlusion_texture();
    if let Some(occ) = &occlusion {
        maps.occlusion_strength = occ.strength();
    }
    match (mr, occlusion) {
        (Some(mr), occ) => {
            let mut orm = image_to_albedo(texture_image(images, mr.texture())?)?;
            let same = occ
                .as_ref()
                .is_some_and(|o| o.texture().source().index() == mr.texture().source().index());
            if !same {
                let occ_map = match &occ {
                    Some(o) => Some(image_to_albedo(texture_image(images, o.texture())?)?),
                    None => None,
                };
                put_red(&mut orm, occ_map.as_ref());
            }
            maps.orm = Some(orm);
        }
        (None, Some(occ)) => {
            let occ_map = image_to_albedo(texture_image(images, occ.texture())?)?;
            let mut orm = AlbedoMap {
                width: occ_map.width,
                height: occ_map.height,
                rgba: vec![255; occ_map.rgba.len()],
            };
            put_red(&mut orm, Some(&occ_map));
            maps.orm = Some(orm);
        }
        (None, None) => {}
    }
    if let Some(info) = material.emissive_texture() {
        maps.emissive = Some(image_to_albedo(texture_image(images, info.texture())?)?);
    }
    let strength = material.emissive_strength().unwrap_or(1.0);
    maps.emissive_factor = material.emissive_factor().map(|c| c * strength);
    Ok(maps)
}

/// Write `source`'s red channel (resampled to fit) into `target`'s red; no
/// source means fully unoccluded.
fn put_red(target: &mut AlbedoMap, source: Option<&AlbedoMap>) {
    let (w, h) = (target.width as usize, target.height as usize);
    for y in 0..h {
        for x in 0..w {
            let r = source.map_or(255, |s| {
                let sx = x * s.width as usize / w;
                let sy = y * s.height as usize / h;
                s.rgba[(sy * s.width as usize + sx) * 4]
            });
            target.rgba[(y * w + x) * 4] = r;
        }
    }
}

#[cfg(test)]
mod scene_tests {
    use super::*;

    fn write_png(path: &Path, w: u32, h: u32, rgb: [u8; 3]) {
        image::RgbImage::from_pixel(w, h, image::Rgb(rgb))
            .save(path)
            .expect("write PNG");
    }

    /// Two materials on two nodes, one moved, one mirrored.
    fn write_two_material_gltf(dir: &Path) {
        std::fs::create_dir_all(dir).expect("temp dir");
        let mut bin = Vec::new();
        let f = |bin: &mut Vec<u8>, v: &[f32]| {
            for c in v {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        };
        f(&mut bin, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // 36 positions
        f(&mut bin, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]); // 36 normals
        f(&mut bin, &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]); // 24 uvs
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin.extend_from_slice(&[0, 0]); // align the morph target
        f(&mut bin, &[0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0]); // 36 morph offsets
        std::fs::write(dir.join("mesh.bin"), &bin).expect("bin");
        write_png(&dir.join("normal.png"), 4, 4, [128, 128, 255]);
        write_png(&dir.join("mr.png"), 4, 4, [9, 200, 50]);
        write_png(&dir.join("occ.png"), 2, 2, [77, 0, 0]);
        let json = r#"{
  "asset":{"version":"2.0"},
  "extensionsUsed":["KHR_materials_emissive_strength"],
  "scene":0,
  "scenes":[{"nodes":[0,1]}],
  "nodes":[
    {"mesh":0,"translation":[10,0,0]},
    {"mesh":1,"scale":[-1,1,1]}
  ],
  "meshes":[
    {"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":0}]},
    {"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3,"material":1,
      "targets":[{"POSITION":4}]}],"weights":[0.5]}
  ],
  "materials":[
    {"normalTexture":{"index":0,"scale":0.5,
       "extensions":{"KHR_texture_transform":{"scale":[48,24],"offset":[0.25,0]}}},
     "occlusionTexture":{"index":2,"strength":0.8},
     "pbrMetallicRoughness":{"metallicRoughnessTexture":{"index":1}},
     "emissiveFactor":[1,0.5,0],
     "extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":4}}},
    {"pbrMetallicRoughness":{"metallicFactor":0,"roughnessFactor":0.3}}
  ],
  "textures":[{"source":0},{"source":1},{"source":2}],
  "images":[{"uri":"normal.png"},{"uri":"mr.png"},{"uri":"occ.png"}],
  "accessors":[
    {"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","max":[1,1,0],"min":[0,0,0]},
    {"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"},
    {"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"},
    {"bufferView":3,"componentType":5123,"count":3,"type":"SCALAR"},
    {"bufferView":4,"componentType":5126,"count":3,"type":"VEC3","max":[0,0,2],"min":[0,0,2]}
  ],
  "bufferViews":[
    {"buffer":0,"byteOffset":0,"byteLength":36},
    {"buffer":0,"byteOffset":36,"byteLength":36},
    {"buffer":0,"byteOffset":72,"byteLength":24},
    {"buffer":0,"byteOffset":96,"byteLength":6},
    {"buffer":0,"byteOffset":104,"byteLength":36}
  ],
  "buffers":[{"byteLength":140,"uri":"mesh.bin"}]
}"#;
        std::fs::write(dir.join("model.gltf"), json).expect("gltf");
    }

    #[test]
    fn scene_splits_by_material_with_transforms_and_maps() {
        let dir = std::env::temp_dir().join(format!("engine-scene-{}", std::process::id()));
        write_two_material_gltf(&dir);
        let scene = Model::load_scene_with(dir.join("model.gltf"), &dir, &EngineLimits::default())
            .expect("scene loads");
        assert_eq!(scene.parts.len(), 2);

        // Part 0: moved by its node, full map set.
        let moved = &scene.parts[0];
        let (lo, hi) = moved.bounds().unwrap();
        assert!((lo.x - 10.0).abs() < 1e-5 && (hi.x - 11.0).abs() < 1e-5);
        assert!(moved.has_normals());
        let maps = moved.material_maps().expect("maps kept");
        assert!(maps.normal.is_some());
        assert_eq!(maps.normal_scale, 0.5);
        assert_eq!(
            maps.normal_uv,
            [48.0, 24.0, 0.25, 0.0],
            "KHR_texture_transform"
        );
        assert_eq!(maps.occlusion_strength, 0.8);
        let orm = maps.orm.as_ref().expect("orm packed");
        assert_eq!(
            &orm.rgba[..3],
            &[77, 200, 50],
            "occlusion goes into red, roughness and metallic stay"
        );
        assert_eq!(maps.emissive_factor, [4.0, 2.0, 0.0]);

        // Part 1: plain factors, mirrored, still facing +z.
        let plain = &scene.parts[1];
        assert!(plain.material_maps().is_none(), "neutral maps dropped");
        let (lo, hi) = plain.bounds().unwrap();
        assert!(
            (lo.z - 1.0).abs() < 1e-5 && (hi.z - 1.0).abs() < 1e-5,
            "default morph weight 0.5 of a +2 z offset applied: {lo} {hi}"
        );
        let surface = plain.surface_material();
        assert_eq!((surface.roughness, surface.metallic), (0.3, 0.0));
        let built = plain.build();
        let [a, b, c] = [0, 1, 2].map(|i| built.positions[built.indices[i] as usize]);
        assert!(
            (b - a).cross(c - a).z > 0.0,
            "winding turned back after mirroring"
        );
        assert!(
            built.normals.iter().all(|n| n.z > 0.99),
            "authored normals kept"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
