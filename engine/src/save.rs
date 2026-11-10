//! Saving and loading a world: everything that changes while people play.
//!
//! ```ignore
//! #[derive(Serialize, Deserialize)]
//! struct Game { gold: u32, hero: EntityId, camera: [f32; 3] }
//!
//! world.save("slot1.save", &game)?;
//! // ... later, after building the world as at startup (terrain, scenery,
//! // register_prefab for every prefab):
//! let loaded = world.load("slot1.save")?;
//! let mut game: Game = loaded.game()?;
//! game.hero = loaded.entity(game.hero).unwrap();
//! ```
//!
//! A save holds:
//! - every placed prefab (where it is, where each part is, bodies with
//!   their speed and spin, which clip a character plays and how far in),
//! - lamps that stand alone or ride on prefab things,
//! - the voxel terrain wherever it was dug or built on,
//! - the time of day,
//! - your own game data (anything serde can write).
//!
//! What your code builds at startup (terrain, scenery, plain meshes) is not
//! saved: build it as usual, register your prefabs, then load. Entity ids
//! change on load; [`Loaded::entity`] maps old ids to new ones.

use super::{EntityId, World};
use crate::error::{EngineError, EngineResult};
use crate::hierarchy::LampLink;
use crate::lamps::{Lamp, LampId, LampShape};
use glam::{IVec3, Mat4, Vec3};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"ENGSAVE\0";
const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SaveFile {
    version: u32,
    sky: Option<(f32, f32)>,
    things: Vec<SavedThing>,
    lamps: Vec<SavedLamp>,
    chunks: Vec<SavedChunk>,
    game: serde_json::Value,
    /// Components registered with `World::save_component`, by saved name:
    /// (old entity id, value).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    components: BTreeMap<String, Vec<(u64, serde_json::Value)>>,
}

#[derive(Serialize, Deserialize)]
struct SavedThing {
    root: u64,
    prefab: String,
    place: [f32; 16],
    /// The entity the root hangs on, if that is a saved thing too.
    parent: Option<u64>,
    parts: Vec<SavedPart>,
    /// Lamp values as they are now (colour, flicker...), in prefab order.
    lamps: Vec<(u32, SavedLampLook)>,
}

#[derive(Serialize, Deserialize)]
struct SavedPart {
    /// Which part of the prefab (parts destroyed in play are left out).
    index: u32,
    id: u64,
    /// Relative to the parent while still attached, otherwise in the world.
    place: [f32; 16],
    attached: bool,
    velocity: Option<[f32; 3]>,
    spin: Option<[f32; 3]>,
    clip: Option<(String, f32, bool)>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
struct SavedLampLook {
    color: [f32; 4],
    intensity: f32,
    range: f32,
    shadows: bool,
    flicker: f32,
    /// `None` for a point light; full angle and softness for a spot.
    spot: Option<(f32, f32)>,
}

#[derive(Serialize, Deserialize)]
struct SavedLamp {
    id: u32,
    look: SavedLampLook,
    position: [f32; 3],
    direction: [f32; 3],
    /// Riding on a saved entity: (entity, bone name, offset, local direction).
    on: Option<SavedLampMount>,
}

/// A saved lamp's mount: (entity, bone name, offset, local direction).
type SavedLampMount = (u64, Option<String>, [f32; 3], Option<[f32; 3]>);

#[derive(Serialize, Deserialize)]
struct SavedChunk {
    key: [i32; 3],
    /// Byte range of the f32 samples in the blob section.
    samples: (u64, u64),
    materials: Option<(u64, u64)>,
}

/// What [`World::load`] gives back: your game data, and new ids for old.
#[derive(Debug)]
pub struct Loaded {
    game: serde_json::Value,
    entities: HashMap<u64, EntityId>,
    lamps: HashMap<u32, LampId>,
}

impl Loaded {
    /// Your game data, as saved.
    pub fn game<T: DeserializeOwned>(&self) -> EngineResult<T> {
        T::deserialize(&self.game)
            .map_err(|e| EngineError::InvalidValue(format!("save: game data doesn't fit: {e}")))
    }

    /// The new id of an entity that was saved (prefab roots and parts).
    pub fn entity(&self, old: EntityId) -> Option<EntityId> {
        self.entities.get(&old.raw()).copied()
    }

    /// The new id of a saved lamp.
    pub fn lamp(&self, old: LampId) -> Option<LampId> {
        self.lamps.get(&old.0).copied()
    }
}

fn mat(m: Mat4) -> [f32; 16] {
    m.to_cols_array()
}

fn bad(what: impl std::fmt::Display) -> EngineError {
    EngineError::InvalidValue(format!("save file: {what}"))
}

fn look_of(l: &Lamp) -> SavedLampLook {
    SavedLampLook {
        color: [l.color.r, l.color.g, l.color.b, l.color.a],
        intensity: l.intensity,
        range: l.range,
        shadows: l.shadows,
        flicker: l.flicker,
        spot: match l.shape {
            LampShape::Point => None,
            LampShape::Spot {
                angle_deg,
                softness,
                ..
            } => Some((angle_deg, softness)),
        },
    }
}

fn apply_look(l: &mut Lamp, look: &SavedLampLook, direction: Vec3) {
    l.color = crate::color::Color {
        r: look.color[0],
        g: look.color[1],
        b: look.color[2],
        a: look.color[3],
    };
    l.intensity = look.intensity;
    l.range = look.range;
    l.shadows = look.shadows;
    l.flicker = look.flicker;
    l.shape = match look.spot {
        None => LampShape::Point,
        Some((angle_deg, softness)) => LampShape::Spot {
            direction,
            angle_deg,
            softness,
        },
    };
}

fn lamp_direction(l: &Lamp) -> Vec3 {
    match l.shape {
        LampShape::Spot { direction, .. } => direction,
        LampShape::Point => Vec3::NEG_Y,
    }
}

impl World {
    /// Write everything that changed during play, plus `game`, to a file.
    pub fn save(&self, path: impl AsRef<Path>, game: &impl Serialize) -> EngineResult<()> {
        let bytes = self.save_to_bytes(game)?;
        let path = path.as_ref();
        // Write next to it first, so a crash never leaves half a save.
        let tmp = path.with_extension("saving");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Like [`Self::save`], into memory (for network sync or your own files).
    pub fn save_to_bytes(&self, game: &impl Serialize) -> EngineResult<Vec<u8>> {
        let game = serde_json::to_value(game)
            .map_err(|e| EngineError::InvalidValue(format!("save: game data: {e}")))?;
        let mut blob: Vec<u8> = Vec::new();
        let mut put = |bytes: &[u8]| {
            let at = blob.len() as u64;
            blob.extend_from_slice(bytes);
            (at, bytes.len() as u64)
        };

        // Prefab things.
        let mut roots: Vec<EntityId> = self.prefab_instances.keys().copied().collect();
        roots.sort_by_key(|id| id.raw());
        let saved_entities: HashSet<u64> = roots
            .iter()
            .flat_map(|r| {
                std::iter::once(r.raw())
                    .chain(self.prefab_instances[r].parts.iter().map(|p| p.raw()))
            })
            .collect();
        let mut prefab_lamps: HashSet<LampId> = HashSet::new();
        let mut things = Vec::new();
        for root in roots {
            let made = &self.prefab_instances[&root];
            let Some(place) = self.matrix_of(root) else {
                continue;
            };
            let parent = self
                .parent(root)
                .filter(|p| saved_entities.contains(&p.raw()))
                .map(|p| p.raw());
            let place = match self.hierarchy.links.get(&root) {
                Some(link) if parent.is_some() => link.local,
                _ => place,
            };
            let mut parts = Vec::new();
            for (index, id) in made.parts.iter().enumerate() {
                let Some(world) = self.matrix_of(*id) else {
                    continue;
                };
                let link = self.hierarchy.links.get(id);
                let clip = self.animated.get(id).map(|e| {
                    let a = &e.animator;
                    (a.clip_name().to_string(), a.time(), a.is_looping())
                });
                parts.push(SavedPart {
                    index: index as u32,
                    id: id.raw(),
                    place: mat(link.map_or(world, |l| l.local)),
                    attached: link.is_some(),
                    velocity: self.body_velocity(*id).map(|v| v.to_array()),
                    spin: self.body_spin(*id).map(|v| v.to_array()),
                    clip,
                });
            }
            let lamps = made
                .lamps
                .iter()
                .enumerate()
                .filter_map(|(i, id)| {
                    prefab_lamps.insert(*id);
                    self.lamps.get(id).map(|(l, _)| (i as u32, look_of(l)))
                })
                .collect();
            things.push(SavedThing {
                root: root.raw(),
                prefab: made.prefab.clone(),
                place: mat(place),
                parent,
                parts,
                lamps,
            });
        }

        // Lamps of their own, or riding on saved things.
        let mut lamps = Vec::new();
        for (id, (l, _)) in &self.lamps {
            if prefab_lamps.contains(id) {
                continue;
            }
            let on = match self.hierarchy.lamps.get(id) {
                None => None,
                Some(link) if saved_entities.contains(&link.entity.raw()) => {
                    let bone = link.bone.and_then(|b| {
                        let e = self.animated.get(&link.entity)?;
                        e.animator.model().skeleton.joint_names.get(b).cloned()
                    });
                    Some((
                        link.entity.raw(),
                        bone,
                        link.offset.to_array(),
                        link.direction.map(|d| d.to_array()),
                    ))
                }
                // Riding on scenery your code builds: it builds the lamp too.
                Some(_) => continue,
            };
            lamps.push(SavedLamp {
                id: id.0,
                look: look_of(l),
                position: l.position.to_array(),
                direction: lamp_direction(l).to_array(),
                on,
            });
        }

        // Dug and built voxel terrain.
        let mut chunks = Vec::new();
        if let Some(t) = self.voxel_terrain.as_ref() {
            let (mut keys, _) = t.volume().changed_since(self.voxel_baseline_rev);
            keys.sort_by_key(|k| (k.x, k.y, k.z));
            for key in keys {
                let Some((samples, materials)) = t.volume().chunk_raw(key) else {
                    continue;
                };
                let samples = put(bytemuck::cast_slice(samples));
                let materials = materials.map(&mut put);
                chunks.push(SavedChunk {
                    key: key.to_array(),
                    samples,
                    materials,
                });
            }
        }
        // Edits a world stream keeps while their ground is unloaded.
        if let Some(stream) = self.stream() {
            let mut stored: Vec<&crate::volume::EditedChunk> = stream.stored_edits().collect();
            stored.sort_by_key(|e| (e.key.x, e.key.y, e.key.z));
            for e in stored {
                let samples = put(bytemuck::cast_slice(&e.samples));
                let materials = e.materials.as_deref().map(&mut put);
                chunks.push(SavedChunk {
                    key: e.key.to_array(),
                    samples,
                    materials,
                });
            }
        }

        let components = self
            .components
            .save(&|id| saved_entities.contains(&id.raw()))?;
        let file = SaveFile {
            version: VERSION,
            components,
            sky: self.physical_sky.map(|s| (s.hour, s.day)),
            things,
            lamps,
            chunks,
            game,
        };
        let json = serde_json::to_vec(&file).map_err(bad)?;
        let mut out = Vec::with_capacity(json.len() / 4 + blob.len() / 4 + 16);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        let mut z = flate2::write::ZlibEncoder::new(out, flate2::Compression::fast());
        z.write_all(&(json.len() as u64).to_le_bytes())?;
        z.write_all(&json)?;
        z.write_all(&blob)?;
        Ok(z.finish()?)
    }

    /// Read a save written by [`Self::save`]: removes the prefab things and
    /// free-standing lamps there are now and brings back the saved ones,
    /// restores dug terrain and the time of day. Build the world as at
    /// startup and register your prefabs first.
    pub fn load(&mut self, path: impl AsRef<Path>) -> EngineResult<Loaded> {
        let bytes = std::fs::read(path)?;
        self.load_from_bytes(&bytes)
    }

    pub fn load_from_bytes(&mut self, bytes: &[u8]) -> EngineResult<Loaded> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err(bad("not a save file"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
        if version > VERSION {
            return Err(bad(format!(
                "written by a newer engine (version {version})"
            )));
        }
        let mut payload = Vec::new();
        flate2::read::ZlibDecoder::new(&bytes[12..])
            .take(4 << 30)
            .read_to_end(&mut payload)
            .map_err(|e| bad(format!("damaged ({e})")))?;
        if payload.len() < 8 {
            return Err(bad("damaged (too short)"));
        }
        let json_len = u64::from_le_bytes(payload[..8].try_into().expect("8 bytes")) as usize;
        let json = payload
            .get(8..8usize.saturating_add(json_len))
            .ok_or_else(|| bad("damaged (header)"))?;
        let blob = &payload[8 + json_len..];
        let file: SaveFile = serde_json::from_slice(json).map_err(bad)?;
        let slice = |(at, len): (u64, u64)| -> EngineResult<&[u8]> {
            blob.get(at as usize..(at + len) as usize)
                .ok_or_else(|| bad("damaged (data out of range)"))
        };

        // Check everything we need is here before touching the world.
        let unknown = self.components.unknown_names(file.components.keys());
        if !unknown.is_empty() {
            return Err(bad(format!(
                "has components {unknown:?}; call world.save_component::<YourType>(name) for each before loading"
            )));
        }
        for t in &file.things {
            if !self.prefabs.contains_key(&t.prefab) {
                return Err(bad(format!(
                    "needs prefab {:?}; register_prefab it before loading",
                    t.prefab
                )));
            }
        }

        // Clear what the save replaces.
        let roots: Vec<EntityId> = self.prefab_instances.keys().copied().collect();
        for root in roots {
            self.despawn(root);
        }
        let free: Vec<LampId> = self
            .lamps
            .keys()
            .filter(|id| !self.hierarchy.lamps.contains_key(id))
            .copied()
            .collect();
        for id in free {
            self.remove_lamp(id);
        }

        let mut entities: HashMap<u64, EntityId> = HashMap::new();
        let mut lamp_ids: HashMap<u32, LampId> = HashMap::new();

        // Prefab things, parents before children.
        let mut order: Vec<usize> = (0..file.things.len()).collect();
        let index: HashMap<u64, usize> = file
            .things
            .iter()
            .enumerate()
            .flat_map(|(i, t)| {
                std::iter::once((t.root, i)).chain(t.parts.iter().map(move |p| (p.id, i)))
            })
            .collect();
        let depth = |mut i: usize| {
            let mut d = 0;
            while let Some(p) = file.things[i].parent.and_then(|p| index.get(&p)) {
                d += 1;
                i = *p;
                if d > file.things.len() {
                    break;
                }
            }
            d
        };
        order.sort_by_key(|i| depth(*i));
        for i in order {
            let t = &file.things[i];
            let prefab = self.prefabs[&t.prefab].clone();
            let spawned = self.spawn_prefab(&prefab, crate::place::Place::default())?;
            entities.insert(t.root, spawned.root);
            let place = Mat4::from_cols_array(&t.place);
            match t.parent.and_then(|p| entities.get(&p).copied()) {
                Some(parent) => {
                    self.link_child(spawned.root, parent, place, None)?;
                }
                None => {
                    self.write_matrix(spawned.root, place)?;
                    self.update_subtree(spawned.root)?;
                }
            }
            let made: Vec<EntityId> = spawned.parts().map(|(_, id)| id).collect();
            let part = |saved: &SavedPart| made.get(saved.index as usize).copied();
            // Parts destroyed before the save stay destroyed.
            for (i, id) in made.iter().enumerate() {
                if !t.parts.iter().any(|p| p.index as usize == i) {
                    self.despawn(*id);
                }
            }
            for saved in &t.parts {
                let Some(id) = part(saved) else {
                    return Err(bad(format!(
                        "prefab {:?} has fewer parts than saved",
                        t.prefab
                    )));
                };
                entities.insert(saved.id, id);
                let m = Mat4::from_cols_array(&saved.place);
                if !m.is_finite() {
                    return Err(bad("a part's place is not finite"));
                }
                match (saved.attached, self.hierarchy.links.get_mut(&id)) {
                    (true, Some(link)) => link.local = m,
                    (false, Some(_)) => {
                        self.hierarchy.unlink(id);
                        self.write_matrix(id, m)?;
                    }
                    (_, None) => self.write_matrix(id, m)?,
                }
                if let Some((clip, time, looping)) = &saved.clip {
                    if let Some(e) = self.animated.get_mut(&id) {
                        let played = if *looping {
                            e.animator.play(clip)
                        } else {
                            e.animator.play_once(clip)
                        };
                        if played.is_ok() {
                            e.animator.seek(*time);
                        }
                    }
                }
            }
            self.update_subtree(spawned.root)?;
            for saved in &t.parts {
                if let Some(id) = part(saved) {
                    self.update_subtree(id)?;
                }
            }
            for saved in &t.parts {
                let Some(id) = part(saved) else { continue };
                if let Some(v) = saved.velocity {
                    let _ = self.set_body_velocity(id, Vec3::from_array(v));
                }
                if let Some(s) = saved.spin {
                    let _ = self.set_body_spin(id, Vec3::from_array(s));
                }
            }
            for (k, look) in &t.lamps {
                if let Some(id) = spawned.lamps.get(*k as usize) {
                    if let Some((l, _)) = self.lamps.get_mut(id) {
                        let dir = lamp_direction(l);
                        apply_look(l, look, dir);
                    }
                }
            }
        }

        // Free-standing lamps and lamps riding on things.
        for s in &file.lamps {
            let direction = Vec3::from_array(s.direction);
            let mut lamp = Lamp::point(
                Vec3::from_array(s.position),
                crate::color::Color::WHITE,
                1.0,
                1.0,
            );
            apply_look(&mut lamp, &s.look, direction);
            let id = self.add_lamp(lamp)?;
            lamp_ids.insert(s.id, id);
            if let Some((entity, bone, offset, local)) = &s.on {
                let Some(entity) = entities.get(entity).copied() else {
                    continue;
                };
                let bone = match bone {
                    Some(name) => match self.bone_index(entity, name) {
                        Ok(b) => Some(b),
                        Err(_) => continue,
                    },
                    None => None,
                };
                self.hierarchy.lamps.insert(
                    id,
                    LampLink {
                        entity,
                        bone,
                        offset: Vec3::from_array(*offset),
                        direction: local.map(Vec3::from_array),
                    },
                );
            }
        }
        self.update_lamps();

        // Dug terrain.
        if !file.chunks.is_empty() {
            let terrain = self.voxel_terrain.as_mut().ok_or_else(|| {
                bad("has dug terrain but this world has no voxel terrain; set it before loading")
            })?;
            for c in &file.chunks {
                let raw = slice(c.samples)?;
                if raw.len() % 4 != 0 {
                    return Err(bad("damaged (samples)"));
                }
                let samples: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                    .collect();
                let materials = c.materials.map(slice).transpose()?.map(<[u8]>::to_vec);
                if !terrain
                    .volume_mut()
                    .put_chunk_raw(IVec3::from_array(c.key), samples, materials)
                {
                    return Err(bad("damaged (chunk size)"));
                }
            }
        }

        if let Some((hour, day)) = file.sky {
            let mut sky = self
                .physical_sky
                .unwrap_or_else(|| crate::sky::PhysicalSky::at_hour(hour));
            sky.hour = hour;
            sky.day = day;
            self.set_physical_sky(Some(sky));
        }

        self.components
            .load(file.components, &|old| entities.get(&old).copied())?;

        Ok(Loaded {
            game: file.game,
            entities,
            lamps: lamp_ids,
        })
    }
}
