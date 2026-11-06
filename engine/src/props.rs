//! AssetGenerator prop packs (FMPK v1 and v2).
//!
//! Props are modelled in Blender from AssetGenerator specs and packed by
//! `ag.py pack`: quantised positions, normals, vertex colours with ambient
//! occlusion baked in, and (v2, `--slots`) the material slot of every vertex
//! so wood, iron and cloth can each get their own surface. The format is
//! described in AssetGenerator's `blender/entrypoints/pack.py`.
//!
//! Frame: metres, y up, origin on the ground under the footprint centre,
//! front +z.

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::{Mesh, SurfaceMaterial};
use glam::Vec3;
use std::path::Path;

/// One prop from a pack.
#[derive(Clone, Debug)]
pub struct PropModel {
    pub name: String,
    pub positions: Vec<Vec3>,
    /// Unit normals.
    pub normals: Vec<Vec3>,
    /// Linear RGB with the baked occlusion.
    pub colors: Vec<Vec3>,
    pub indices: Vec<u32>,
    /// The spec's material names (v2; empty in a v1 pack).
    pub slot_names: Vec<String>,
    /// Index into `slot_names` per vertex (v2; all 0 in a v1 pack).
    pub slots: Vec<u8>,
    pub min: Vec3,
    pub max: Vec3,
}

/// Every prop of a pack, in pack order.
#[derive(Clone, Debug, Default)]
pub struct PropPack {
    pub models: Vec<PropModel>,
}

impl PropPack {
    /// Read a `.fmpk` file. The path is used as given (no sandbox root).
    pub fn load(path: impl AsRef<Path>) -> EngineResult<Self> {
        let bytes = std::fs::read(path)?;
        Self::decode_with(&bytes, &EngineLimits::default())
    }

    pub fn decode(bytes: &[u8]) -> EngineResult<Self> {
        Self::decode_with(bytes, &EngineLimits::default())
    }

    pub fn decode_with(bytes: &[u8], limits: &EngineLimits) -> EngineResult<Self> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != b"FMPK" {
            return Err(bad("not an FMPK pack"));
        }
        let version = r.u32()?;
        if version != 1 && version != 2 {
            return Err(bad(&format!("FMPK version {version} is not known")));
        }
        let count = r.u32()?;
        let mut models = Vec::new();
        let mut triangles = 0u64;
        for _ in 0..count {
            let name = r.str()?;
            r.pad();
            let mut slot_names = Vec::new();
            if version == 2 {
                for _ in 0..r.u16()? {
                    slot_names.push(r.str()?);
                }
                r.pad();
            }
            let scale = r.f32()?;
            let min = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
            let max = Vec3::new(r.f32()?, r.f32()?, r.f32()?);
            let n = r.u32()? as usize;
            let ni = r.u32()? as usize;
            triangles += ni as u64 / 3;
            if triangles > limits.max_model_triangles {
                return Err(EngineError::ResourceLimit(format!(
                    "prop pack exceeds {} triangles",
                    limits.max_model_triangles
                )));
            }
            if !scale.is_finite() {
                return Err(bad("prop position scale is not finite"));
            }
            let raw = r.take(6 * n)?;
            let positions = raw
                .chunks_exact(6)
                .map(|c| {
                    let q = |k: usize| i16::from_le_bytes([c[k], c[k + 1]]) as f32 * scale;
                    Vec3::new(q(0), q(2), q(4))
                })
                .collect();
            r.pad();
            let normals = r
                .take(3 * n)?
                .chunks_exact(3)
                .map(|c| Vec3::new(c[0] as i8 as f32, c[1] as i8 as f32, c[2] as i8 as f32) / 127.0)
                .collect();
            let colors = r
                .take(3 * n)?
                .chunks_exact(3)
                .map(|c| {
                    let lin = |v: u8| (v as f32 / 255.0).powi(2);
                    Vec3::new(lin(c[0]), lin(c[1]), lin(c[2]))
                })
                .collect();
            let slots = if version == 2 {
                r.take(n)?.to_vec()
            } else {
                vec![0; n]
            };
            r.pad();
            let wide = n > 65535;
            let indices: Vec<u32> = if wide {
                r.take(4 * ni)?
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                r.take(2 * ni)?
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                    .collect()
            };
            r.pad();
            if indices.iter().any(|&i| i as usize >= n) {
                return Err(bad(&format!(
                    "prop '{name}' has an index past its vertices"
                )));
            }
            if version == 2 && slots.iter().any(|&s| s as usize >= slot_names.len()) {
                return Err(bad(&format!(
                    "prop '{name}' has a slot past its slot names"
                )));
            }
            models.push(PropModel {
                name,
                positions,
                normals,
                colors,
                indices,
                slot_names,
                slots,
                min,
                max,
            });
        }
        Ok(Self { models })
    }

    pub fn get(&self, name: &str) -> Option<&PropModel> {
        self.models.iter().find(|m| m.name == name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.models.iter().map(|m| m.name.as_str())
    }
}

impl PropModel {
    /// A mesh with each slot's surface guessed from its material name
    /// (see [`default_slot_material`]).
    pub fn to_mesh(&self) -> EngineResult<Mesh> {
        self.to_mesh_with(default_slot_material)
    }

    /// A mesh whose vertices take the surface `slot_material` gives their
    /// slot's name (`""` for v1 packs, which have no slots).
    pub fn to_mesh_with(
        &self,
        slot_material: impl Fn(&str) -> SurfaceMaterial,
    ) -> EngineResult<Mesh> {
        let materials: Vec<SurfaceMaterial> = if self.slot_names.is_empty() {
            vec![slot_material("")]
        } else {
            self.slot_names.iter().map(|s| slot_material(s)).collect()
        };
        let mut mesh = Mesh::new();
        mesh.set_surface_material(materials[0]);
        let mut ids = Vec::with_capacity(self.positions.len());
        for i in 0..self.positions.len() {
            let id = mesh.add_point(self.positions[i])?;
            let c = self.colors[i];
            mesh.set_point_color(id, Color::rgba01(c.x, c.y, c.z, 1.0)?)?;
            mesh.set_point_normal(id, self.normals[i])?;
            ids.push(id);
        }
        for tri in self.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| ids[tri[k] as usize]);
            if a != b && b != c && a != c {
                mesh.add_triangle(a, b, c)?;
            }
        }
        if materials.len() > 1 {
            let overrides = self.slots.iter().map(|&s| materials[s as usize]).collect();
            mesh.set_surface_overrides(overrides);
        }
        Ok(mesh)
    }
}

/// A surface for an AssetGenerator material name: metals shine, wood gets
/// its grain, stone its strata, cloth and leather go matte.
pub fn default_slot_material(name: &str) -> SurfaceMaterial {
    let name = name.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
    if has(&[
        "iron", "steel", "metal", "gold", "silver", "bronze", "copper", "brass",
    ]) {
        return SurfaceMaterial {
            roughness: 0.4,
            metallic: 1.0,
            ..SurfaceMaterial::DEFAULT
        };
    }
    if has(&["wood", "plank", "bark", "oak", "pine"]) {
        return SurfaceMaterial::WOOD;
    }
    if has(&["stone", "rock", "brick", "marble", "granite"]) {
        return SurfaceMaterial::STONE;
    }
    if has(&[
        "cloth", "fabric", "leather", "rope", "wool", "linen", "straw",
    ]) {
        return SurfaceMaterial {
            roughness: 0.9,
            metallic: 0.0,
            ..SurfaceMaterial::DEFAULT
        };
    }
    if has(&["glass", "crystal", "gem"]) {
        return SurfaceMaterial {
            roughness: 0.1,
            metallic: 0.0,
            ..SurfaceMaterial::DEFAULT
        };
    }
    SurfaceMaterial::DEFAULT
}

fn bad(msg: &str) -> EngineError {
    EngineError::Model(msg.to_string())
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> EngineResult<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| bad("FMPK pack is cut short"))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }
    fn pad(&mut self) {
        self.at = (self.at + 3) & !3;
    }
    fn u16(&mut self) -> EngineResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> EngineResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn f32(&mut self) -> EngineResult<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn str(&mut self) -> EngineResult<String> {
        let len = self.u16()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| bad("FMPK name is not UTF-8"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) struct TestModel<'a> {
        pub name: &'a str,
        pub slots: &'a [&'a str],
        pub pos: &'a [[f32; 3]],
        pub col: &'a [[u8; 3]],
        pub slot: &'a [u8],
        pub idx: &'a [u16],
    }

    /// A pack written the way AssetGenerator's pack.py writes it.
    pub(crate) fn encode(version: u32, models: &[TestModel<'_>]) -> Vec<u8> {
        let mut out: Vec<u8> = b"FMPK".to_vec();
        let pad = |out: &mut Vec<u8>| {
            while out.len() % 4 != 0 {
                out.push(0);
            }
        };
        let str = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u16).to_le_bytes());
            out.extend(s.as_bytes());
        };
        out.extend(version.to_le_bytes());
        out.extend((models.len() as u32).to_le_bytes());
        for m in models {
            str(&mut out, m.name);
            pad(&mut out);
            if version == 2 {
                out.extend((m.slots.len() as u16).to_le_bytes());
                for s in m.slots {
                    str(&mut out, s);
                }
                pad(&mut out);
            }
            let scale = m.pos.iter().flatten().fold(0.0f32, |a, v| a.max(v.abs())) / 32767.0;
            out.extend(scale.to_le_bytes());
            for k in 0..3 {
                let v = m.pos.iter().map(|p| p[k]).fold(f32::MAX, f32::min);
                out.extend(v.to_le_bytes());
            }
            for k in 0..3 {
                let v = m.pos.iter().map(|p| p[k]).fold(f32::MIN, f32::max);
                out.extend(v.to_le_bytes());
            }
            out.extend((m.pos.len() as u32).to_le_bytes());
            out.extend((m.idx.len() as u32).to_le_bytes());
            for v in m.pos.iter().flatten() {
                out.extend(((v / scale).round() as i16).to_le_bytes());
            }
            pad(&mut out);
            for _ in m.pos {
                out.extend([0u8, 127, 0]);
            }
            for c in m.col {
                out.extend(c);
            }
            if version == 2 {
                out.extend(m.slot);
            }
            pad(&mut out);
            for i in m.idx {
                out.extend(i.to_le_bytes());
            }
            pad(&mut out);
        }
        out
    }

    const POS: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, -0.5]];
    const COL: [[u8; 3]; 3] = [[255, 0, 0], [0, 255, 0], [0, 0, 128]];

    #[test]
    fn decodes_v1_and_v2_packs() {
        for version in [1, 2] {
            let bytes = encode(
                version,
                &[
                    TestModel {
                        name: "one",
                        slots: &["oak wood", "iron"],
                        pos: &POS,
                        col: &COL,
                        slot: &[0, 1, 1],
                        idx: &[0, 1, 2],
                    },
                    TestModel {
                        name: "second_model",
                        slots: &["bone"],
                        pos: &POS,
                        col: &COL,
                        slot: &[0, 0, 0],
                        idx: &[0, 1, 2],
                    },
                ],
            );
            let pack = PropPack::decode(&bytes).expect("decodes");
            assert_eq!(pack.names().collect::<Vec<_>>(), ["one", "second_model"]);
            let m = pack.get("one").unwrap();
            assert!((m.positions[2] - Vec3::new(0.0, 2.0, -0.5)).length() < 1e-3);
            assert_eq!(m.normals[0], Vec3::Y);
            assert_eq!(m.colors[0].x, 1.0);
            assert!((m.colors[2].z - (128.0f32 / 255.0).powi(2)).abs() < 1e-6);
            assert_eq!(m.indices, [0, 1, 2]);
            assert_eq!(m.max, Vec3::new(1.0, 2.0, 0.0));
            if version == 2 {
                assert_eq!(m.slot_names, ["oak wood", "iron"]);
                assert_eq!(m.slots, [0, 1, 1]);
            } else {
                assert!(m.slot_names.is_empty());
                assert_eq!(m.slots, [0, 0, 0]);
            }
            let mesh = m.to_mesh().expect("mesh");
            assert_eq!(mesh.face_count(), 1);
            assert!(mesh.has_normals());
        }
    }

    #[test]
    fn slots_pick_surfaces_and_bad_packs_are_errors() {
        let bytes = encode(
            2,
            &[TestModel {
                name: "crate",
                slots: &["Wood", "Iron bands"],
                pos: &POS,
                col: &COL,
                slot: &[0, 1, 1],
                idx: &[0, 1, 2],
            }],
        );
        let pack = PropPack::decode(&bytes).unwrap();
        let built = pack.models[0].to_mesh().unwrap().build();
        assert!(built.surface_overrides.iter().any(|s| s.metallic == 1.0));
        assert!(built.surface_overrides.contains(&SurfaceMaterial::WOOD));

        assert!(PropPack::decode(b"NOPE").is_err());
        assert!(
            PropPack::decode(&bytes[..bytes.len() - 6]).is_err(),
            "cut short"
        );
        let mut bad_version = bytes.clone();
        bad_version[4] = 9;
        assert!(PropPack::decode(&bad_version).is_err());
    }
}
