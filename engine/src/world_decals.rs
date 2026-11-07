//! `World`'s decal pool (see `crate::decals`).

use super::{EntityId, World};
use crate::decals::{custom_image, Decal, DecalId, DecalImage, DecalKind, MAX_DECALS};
use crate::error::{EngineError, EngineResult};
use glam::{Mat3, Mat4, Quat, Vec3};
use std::collections::HashMap;

/// What a decal box hangs on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Anchor {
    World,
    Entity(EntityId),
    /// A joint of an animated entity.
    Bone(EntityId, usize),
}

#[derive(Clone, Debug)]
pub(crate) struct DecalEntry {
    pub id: DecalId,
    pub decal: Decal,
    pub born: f32,
    pub anchor: Anchor,
    /// The unit box (-1..1 on each axis) to the anchor's space.
    pub local: Mat4,
}

/// One decal as the renderer draws it this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecalDraw {
    /// The unit box to world space.
    pub box_world: Mat4,
    pub layer: usize,
    pub opacity: f32,
    pub emission: f32,
    pub tint: Vec3,
    /// Metres across (for picking a texture mip).
    pub width: f32,
}

#[derive(Debug)]
pub(crate) struct DecalPool {
    pub entries: Vec<DecalEntry>,
    next_id: u64,
    pub budget: usize,
    /// Custom images, by custom index.
    pub images: Vec<std::sync::Arc<DecalImage>>,
}

impl std::fmt::Debug for DecalImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DecalImage")
    }
}

impl Default for DecalPool {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            budget: 256,
            images: Vec::new(),
        }
    }
}

/// The unit box of a decal in world space.
fn box_matrix(d: &Decal) -> Mat4 {
    let n = d.normal.normalize();
    let reference = if n.y.abs() > 0.9 { Vec3::Z } else { Vec3::Y };
    // Image up follows the world's up (or +z on floors), turned by `angle`.
    let up0 = (reference - n * reference.dot(n)).normalize();
    let right0 = up0.cross(n);
    let rot = Quat::from_axis_angle(n, d.angle);
    let right = rot * right0;
    let up = rot * up0;
    let r = Quat::from_mat3(&Mat3::from_cols(right, up, n));
    Mat4::from_scale_rotation_translation(
        Vec3::new(d.size[0] * 0.5, d.size[1] * 0.5, d.depth),
        r,
        d.at,
    )
}

impl World {
    /// Where `anchor` is now (None once its entity is gone).
    fn anchor_matrix(
        &self,
        anchor: Anchor,
        poses: &mut HashMap<EntityId, Vec<Mat4>>,
    ) -> Option<Mat4> {
        match anchor {
            Anchor::World => Some(Mat4::IDENTITY),
            Anchor::Entity(id) => {
                if let Some(a) = self.animated.get(&id) {
                    return Some(a.transform);
                }
                self.entity(id).ok().map(|e| e.transform())
            }
            Anchor::Bone(id, joint) => {
                let a = self.animated.get(&id)?;
                let globals = poses.entry(id).or_insert_with(|| a.animator.pose_now().1);
                Some(a.transform * *globals.get(joint)?)
            }
        }
    }

    /// Put a decal in the world (see [`crate::decals`]). Beyond the budget
    /// the oldest decal makes room.
    pub fn spawn_decal(&mut self, decal: Decal) -> EngineResult<DecalId> {
        decal.validate()?;
        if let DecalKind::Custom(n) = decal.kind {
            if n as usize >= self.decals.images.len() {
                return Err(EngineError::InvalidValue(format!(
                    "no custom decal image {n}"
                )));
            }
        }
        self.purge_decals();
        let world_box = box_matrix(&decal);
        let mut poses = HashMap::new();
        let anchor = match decal.attach {
            None => Anchor::World,
            Some(id) => match self.animated.get(&id) {
                Some(a) => {
                    // The nearest joint, so the decal rides on that limb.
                    let globals = a.animator.pose_now().1;
                    let best = globals
                        .iter()
                        .enumerate()
                        .map(|(j, g)| {
                            (
                                j,
                                a.transform
                                    .transform_point3(g.w_axis.truncate())
                                    .distance(decal.at),
                            )
                        })
                        .min_by(|x, y| x.1.total_cmp(&y.1));
                    poses.insert(id, globals);
                    match best {
                        Some((j, _)) => Anchor::Bone(id, j),
                        None => Anchor::Entity(id),
                    }
                }
                None => {
                    self.entity(id)?;
                    Anchor::Entity(id)
                }
            },
        };
        let m = self
            .anchor_matrix(anchor, &mut poses)
            .ok_or(EngineError::UnknownEntity)?;
        let local = m.inverse() * world_box;
        let id = DecalId(self.decals.next_id);
        self.decals.next_id += 1;
        while self.decals.entries.len() >= self.decals.budget {
            self.decals.entries.remove(0);
        }
        self.decals.entries.push(DecalEntry {
            id,
            decal,
            born: self.time(),
            anchor,
            local,
        });
        Ok(id)
    }

    /// Drop decals whose time ran out or whose entity is gone.
    fn purge_decals(&mut self) {
        let now = self.time();
        let mut entries = std::mem::take(&mut self.decals.entries);
        entries.retain(|e| {
            let alive = match e.anchor {
                Anchor::World => true,
                Anchor::Entity(id) => self.animated.contains_key(&id) || self.entity(id).is_ok(),
                Anchor::Bone(id, _) => self.animated.contains_key(&id),
            };
            alive && e.decal.opacity_at(now - e.born).is_some()
        });
        self.decals.entries = entries;
    }

    pub fn remove_decal(&mut self, id: DecalId) -> bool {
        let before = self.decals.entries.len();
        self.decals.entries.retain(|e| e.id != id);
        self.decals.entries.len() != before
    }

    pub fn clear_decals(&mut self) {
        self.decals.entries.clear();
    }

    /// Live decals (including ones fading out).
    pub fn decal_count(&self) -> usize {
        let now = self.time();
        self.decals
            .entries
            .iter()
            .filter(|e| e.decal.opacity_at(now - e.born).is_some())
            .count()
    }

    /// Most decals at once (1..=[`MAX_DECALS`], 256 by default); the oldest
    /// go first.
    pub fn set_decal_budget(&mut self, decals: usize) -> EngineResult<()> {
        if !(1..=MAX_DECALS).contains(&decals) {
            return Err(EngineError::InvalidValue(format!(
                "decal budget must be within 1..={MAX_DECALS}"
            )));
        }
        self.decals.budget = decals;
        let extra = self.decals.entries.len().saturating_sub(decals);
        self.decals.entries.drain(..extra);
        Ok(())
    }

    pub fn decal_budget(&self) -> usize {
        self.decals.budget
    }

    /// A decal kind from your own image (RGBA8, sRGB, alpha = coverage),
    /// for graffiti, logos or posters; flat, with `roughness`.
    pub fn add_decal_image(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        roughness: f32,
    ) -> EngineResult<DecalKind> {
        if width == 0 || height == 0 || rgba.len() != width * height * 4 {
            return Err(EngineError::InvalidValue(
                "decal image must be width * height RGBA8 pixels".into(),
            ));
        }
        if self.decals.images.len() >= Decal::custom_limit() {
            return Err(EngineError::InvalidValue("too many decal images".into()));
        }
        let n = self.decals.images.len() as u16;
        self.decals.images.push(std::sync::Arc::new(custom_image(
            rgba,
            width,
            height,
            roughness.clamp(0.0, 1.0),
        )));
        Ok(DecalKind::Custom(n))
    }

    pub(crate) fn decal_images(&self) -> &[std::sync::Arc<DecalImage>] {
        &self.decals.images
    }

    /// This frame's decals, oldest first, within `reach` of `eye`.
    pub(crate) fn decal_draws(&self, eye: Vec3, reach: f32) -> Vec<DecalDraw> {
        let now = self.time();
        let mut poses = HashMap::new();
        let mut out = Vec::new();
        for e in &self.decals.entries {
            let Some(opacity) = e.decal.opacity_at(now - e.born) else {
                continue;
            };
            if opacity <= 0.0 {
                continue;
            }
            let Some(m) = self.anchor_matrix(e.anchor, &mut poses) else {
                continue;
            };
            let box_world = m * e.local;
            if box_world.w_axis.truncate().distance(eye) > reach {
                continue;
            }
            out.push(DecalDraw {
                box_world,
                layer: e.decal.kind.layer(),
                opacity,
                emission: e.decal.emission,
                tint: e.decal.tint.to_vec3(),
                width: e.decal.size[0].max(e.decal.size[1]),
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use glam::Vec3;

    #[test]
    fn decals_are_pooled_and_follow_moving_entities() {
        let mut world = World::new();
        world.set_decal_budget(3).unwrap();
        let ids: Vec<_> = (0..5)
            .map(|i| {
                world
                    .spawn_decal(Decal::new(
                        DecalKind::Scorch,
                        (i as f32, 0.0, 0.0),
                        Vec3::Y,
                        1.0,
                    ))
                    .unwrap()
            })
            .collect();
        assert_eq!(world.decal_count(), 3);
        assert!(!world.remove_decal(ids[0]), "the oldest made room");
        assert!(world.remove_decal(ids[4]));

        // A decal on a box keeps its place on the box when the box moves.
        world.clear_decals();
        let cube = world
            .spawn(Shape::box_at((0.0, 1.0, 0.0), (2.0, 2.0, 2.0), rgb(200, 200, 200)).unwrap());
        world
            .spawn_decal(Decal::new(DecalKind::Crack, (0.0, 1.0, 1.0), Vec3::Z, 1.0).on(Some(cube)))
            .unwrap();
        let before = world.decal_draws(Vec3::ZERO, 100.0)[0]
            .box_world
            .w_axis
            .truncate();
        world.set_place(cube, Place::new(5.0, 0.0, 0.0)).unwrap();
        let after = world.decal_draws(Vec3::ZERO, 100.0)[0]
            .box_world
            .w_axis
            .truncate();
        assert!(
            (after - before - Vec3::new(5.0, 0.0, 0.0)).length() < 0.2,
            "{before} -> {after}"
        );

        // Gone with its entity.
        world.despawn(cube);
        assert!(world.decal_draws(Vec3::ZERO, 100.0).is_empty());
    }

    #[test]
    fn decals_expire_and_custom_images_are_checked() {
        let mut world = World::new();
        assert!(world
            .spawn_decal(Decal::new(DecalKind::Custom(0), Vec3::ZERO, Vec3::Y, 1.0))
            .is_err());
        let kind = world.add_decal_image(&[255; 4 * 4 * 4], 4, 4, 0.5).unwrap();
        world
            .spawn_decal(Decal::new(kind, Vec3::ZERO, Vec3::Y, 1.0).lifetime(0.5))
            .unwrap();
        assert_eq!(world.decal_count(), 1);
        world.set_time(world.time() + 1.0);
        assert_eq!(world.decal_count(), 0);
    }
}
