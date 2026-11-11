//! Prefabs, and parents and children (pivots, bones, attached lamps).
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    // ---- Prefabs (see `crate::prefab`) ----

    /// Place a prefab: makes its parts (hung on a new root pivot at `place`)
    /// and lamps. Move or despawn the root to move or remove all of it.
    pub fn spawn_prefab(
        &mut self,
        prefab: &crate::prefab::Prefab,
        place: Place,
    ) -> EngineResult<crate::prefab::Spawned> {
        use crate::prefab::PartKind;
        prefab.validate()?;
        let root = self.spawn_pivot(place);
        let mut ids: Vec<EntityId> = Vec::with_capacity(prefab.parts.len());
        let mut bodies = Vec::new();
        let result = (|| -> EngineResult<()> {
            for part in &prefab.parts {
                let id = match &part.kind {
                    PartKind::Pivot => self.spawn_pivot(Place::default()),
                    PartKind::Mesh(mesh) => {
                        let id = self.place((**mesh).clone(), Place::default())?;
                        if !part.casts_shadow {
                            self.set_casts_shadow(id, false)?;
                        }
                        id
                    }
                    PartKind::Model { model, clip } => {
                        let id = self.spawn_animated_shared(model.clone(), Place::default())?;
                        if let Some(clip) = clip {
                            self.play_animation(id, clip)?;
                        }
                        id
                    }
                };
                ids.push(id);
                let parent = part.parent.map(|i| ids[i]).unwrap_or(root);
                self.attach(id, parent, part.place)?;
                if let Some(body) = &part.body {
                    bodies.push((id, body.clone()));
                }
            }
            Ok(())
        })();
        let mut lamps = Vec::new();
        let result = result.and_then(|()| {
            for l in &prefab.lamps {
                let on = l.on.map(|i| ids[i]).unwrap_or(root);
                let lamp = self.add_lamp(l.lamp)?;
                lamps.push(lamp);
                match &l.bone {
                    Some(bone) => self.attach_lamp_to_bone(lamp, on, bone, l.lamp.position)?,
                    None => self.attach_lamp(lamp, on, l.lamp.position)?,
                }
            }
            // Bodies last, once every part sits where it belongs.
            for (id, body) in bodies {
                self.add_body(id, body)?;
            }
            Ok(())
        });
        self.prefab_instances.insert(
            root,
            PrefabInstance {
                prefab: prefab.name().to_string(),
                parts: ids.clone(),
                lamps: lamps.clone(),
            },
        );
        if let Err(e) = result {
            self.despawn(root);
            return Err(e);
        }
        Ok(crate::prefab::Spawned {
            root,
            parts: prefab
                .parts
                .iter()
                .zip(&ids)
                .map(|(p, id)| (p.name.clone(), *id))
                .collect(),
            lamps,
        })
    }

    /// Remember a prefab under its name, for [`Self::spawn_named`] and for
    /// loading saved worlds. Registering the same name again replaces it.
    pub fn register_prefab(&mut self, prefab: crate::prefab::Prefab) -> EngineResult<()> {
        prefab.validate()?;
        self.prefabs
            .insert(prefab.name().to_string(), Arc::new(prefab));
        Ok(())
    }

    pub fn prefab(&self, name: &str) -> Option<&crate::prefab::Prefab> {
        self.prefabs.get(name).map(|p| &**p)
    }

    /// Place a registered prefab by name.
    pub fn spawn_named(
        &mut self,
        name: &str,
        place: Place,
    ) -> EngineResult<crate::prefab::Spawned> {
        let prefab = self
            .prefabs
            .get(name)
            .cloned()
            .ok_or_else(|| EngineError::InvalidValue(format!("no prefab called {name:?}")))?;
        self.spawn_prefab(&prefab, place)
    }

    /// Which prefab a root entity was made from.
    pub fn prefab_of(&self, root: EntityId) -> Option<&str> {
        self.prefab_instances.get(&root).map(|p| p.prefab.as_str())
    }

    // ---- Parents and children (see `crate::hierarchy`) ----

    /// An invisible handle with a place: group things under it with
    /// [`Self::attach`] and move them all at once.
    pub fn spawn_pivot(&mut self, place: Place) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.hierarchy.pivots.insert(id, place.to_matrix());
        id
    }

    /// Hang `child` on `parent` at `local` (relative to the parent). From now
    /// on the child follows the parent, and [`Self::set_place`] on the child
    /// is relative to it.
    pub fn attach(&mut self, child: EntityId, parent: EntityId, local: Place) -> EngineResult<()> {
        self.link_child(child, parent, local.to_matrix(), None)
    }

    /// Hang `child` on a bone of an animated `parent` (`"hand_r"`, `"head"`),
    /// at `local` relative to the bone. It moves with the animation.
    pub fn attach_to_bone(
        &mut self,
        child: EntityId,
        parent: EntityId,
        bone: &str,
        local: Place,
    ) -> EngineResult<()> {
        let bone = self.bone_index(parent, bone)?;
        self.link_child(child, parent, local.to_matrix(), Some(bone))
    }

    /// Give `child` a new parent (or none) without moving it on screen.
    pub fn set_parent(&mut self, child: EntityId, parent: Option<EntityId>) -> EngineResult<()> {
        let world = self.world_transform(child)?;
        match parent {
            None => {
                if self.hierarchy.unlink(child).is_some() {
                    self.write_matrix(child, world)?;
                }
                Ok(())
            }
            Some(parent) => {
                let frame = self.world_transform(parent)?;
                self.link_child(child, parent, frame.inverse() * world, None)
            }
        }
    }

    /// Let go of the parent; the entity stays where it is.
    pub fn detach(&mut self, child: EntityId) -> EngineResult<()> {
        self.set_parent(child, None)
    }

    pub fn parent(&self, id: EntityId) -> Option<EntityId> {
        self.hierarchy.links.get(&id).map(|l| l.parent)
    }

    pub fn children(&self, id: EntityId) -> &[EntityId] {
        self.hierarchy.children_of(id)
    }

    /// Where an entity (mesh, animated model or pivot) really is, parents
    /// included.
    pub fn world_transform(&self, id: EntityId) -> EngineResult<Mat4> {
        self.matrix_of(id).ok_or(EngineError::UnknownEntity)
    }

    /// Where a bone of an animated entity is in the world this frame.
    pub fn bone_transform(&self, id: EntityId, bone: &str) -> EngineResult<Mat4> {
        let index = self.bone_index(id, bone)?;
        let e = self.animated.get(&id).ok_or(EngineError::UnknownEntity)?;
        Ok(e.transform * e.animator.joint_globals()[index])
    }

    /// Let a lamp ride on an entity: `offset` is relative to it, and a
    /// spotlight's direction is taken in the entity's frame (a headlight
    /// pointing along +Z keeps pointing forward as the car turns).
    pub fn attach_lamp(
        &mut self,
        lamp: crate::lamps::LampId,
        entity: EntityId,
        offset: impl Into<Vec3>,
    ) -> EngineResult<()> {
        self.link_lamp(lamp, entity, offset.into(), None)
    }

    /// Let a lamp ride on a bone: a lantern in a hand, a torch on a helmet.
    pub fn attach_lamp_to_bone(
        &mut self,
        lamp: crate::lamps::LampId,
        entity: EntityId,
        bone: &str,
        offset: impl Into<Vec3>,
    ) -> EngineResult<()> {
        let bone = self.bone_index(entity, bone)?;
        self.link_lamp(lamp, entity, offset.into(), Some(bone))
    }

    /// Stop a lamp riding on anything; it stays where it is.
    pub fn detach_lamp(&mut self, lamp: crate::lamps::LampId) {
        self.hierarchy.lamps.remove(&lamp);
    }

    pub(super) fn link_lamp(
        &mut self,
        lamp: crate::lamps::LampId,
        entity: EntityId,
        offset: Vec3,
        bone: Option<usize>,
    ) -> EngineResult<()> {
        if !offset.is_finite() {
            return Err(EngineError::InvalidValue(
                "lamp offset must be finite".into(),
            ));
        }
        let (l, _) = self.lamps.get(&lamp).ok_or(EngineError::UnknownEntity)?;
        self.matrix_of(entity).ok_or(EngineError::UnknownEntity)?;
        let direction = match l.shape {
            crate::lamps::LampShape::Spot { direction, .. } => Some(direction),
            crate::lamps::LampShape::Point => None,
        };
        self.hierarchy.lamps.insert(
            lamp,
            crate::hierarchy::LampLink {
                entity,
                bone,
                offset,
                direction,
            },
        );
        self.update_lamps();
        Ok(())
    }

    pub(super) fn bone_index(&self, id: EntityId, bone: &str) -> EngineResult<usize> {
        let e = self.animated.get(&id).ok_or_else(|| {
            EngineError::InvalidValue(format!("entity {id} has no skeleton (not animated)"))
        })?;
        e.animator
            .model()
            .skeleton
            .joint_index(bone)
            .ok_or_else(|| EngineError::InvalidValue(format!("no bone called {bone:?}")))
    }

    pub(super) fn link_child(
        &mut self,
        child: EntityId,
        parent: EntityId,
        local: Mat4,
        bone: Option<usize>,
    ) -> EngineResult<()> {
        if !local.is_finite() {
            return Err(EngineError::InvalidValue(
                "local place must be finite".into(),
            ));
        }
        self.matrix_of(child).ok_or(EngineError::UnknownEntity)?;
        self.matrix_of(parent).ok_or(EngineError::UnknownEntity)?;
        if self.hierarchy.is_ancestor(child, parent) {
            return Err(EngineError::InvalidValue(format!(
                "can't attach {child} below itself"
            )));
        }
        if self
            .physics
            .body_kind(child)
            .is_some_and(|k| k == crate::physics::BodyKind::Dynamic)
        {
            return Err(EngineError::InvalidValue(format!(
                "{child} has a dynamic body; physics moves it, so it can't follow a parent"
            )));
        }
        self.hierarchy.link(
            child,
            crate::hierarchy::Link {
                parent,
                local,
                bone,
            },
        );
        self.update_subtree(child)
    }

    pub(super) fn matrix_of(&self, id: EntityId) -> Option<Mat4> {
        if let Some(e) = self.entities.get(&id) {
            return Some(e.transform);
        }
        if let Some(e) = self.animated.get(&id) {
            return Some(e.transform);
        }
        self.hierarchy.pivots.get(&id).copied()
    }

    pub(super) fn write_matrix(&mut self, id: EntityId, m: Mat4) -> EngineResult<()> {
        if let Some(e) = self.entities.get_mut(&id) {
            if !crate::hierarchy::same_up_to_rounding(e.transform, m) {
                e.transform = m;
                e.bump_xform();
                self.touch(id);
            }
            return Ok(());
        }
        if let Some(e) = self.animated.get_mut(&id) {
            e.transform = m;
            return Ok(());
        }
        if let Some(p) = self.hierarchy.pivots.get_mut(&id) {
            *p = m;
            return Ok(());
        }
        Err(EngineError::UnknownEntity)
    }

    /// Recompute where everything below `id` is (and `id` itself when it is
    /// a child).
    pub(super) fn update_subtree(&mut self, id: EntityId) -> EngineResult<()> {
        let mut poses: HashMap<EntityId, Vec<Mat4>> = HashMap::new();
        for e in self.hierarchy.subtree(id) {
            self.place_child(e, &mut poses)?;
        }
        self.update_lamps();
        Ok(())
    }

    pub(super) fn place_child(
        &mut self,
        id: EntityId,
        poses: &mut HashMap<EntityId, Vec<Mat4>>,
    ) -> EngineResult<()> {
        let Some(link) = self.hierarchy.links.get(&id).copied() else {
            return Ok(());
        };
        let frame = self.link_frame(link.parent, link.bone, poses)?;
        self.write_matrix(id, frame * link.local)
    }

    /// World matrix of a parent, or of one of its bones.
    pub(super) fn link_frame(
        &self,
        parent: EntityId,
        bone: Option<usize>,
        poses: &mut HashMap<EntityId, Vec<Mat4>>,
    ) -> EngineResult<Mat4> {
        let frame = self.matrix_of(parent).ok_or(EngineError::UnknownEntity)?;
        let Some(bone) = bone else {
            return Ok(frame);
        };
        let pose = match poses.get(&parent) {
            Some(p) => p,
            None => {
                let e = self
                    .animated
                    .get(&parent)
                    .ok_or(EngineError::UnknownEntity)?;
                poses.entry(parent).or_insert(e.animator.joint_globals())
            }
        };
        Ok(frame * pose.get(bone).copied().unwrap_or(Mat4::IDENTITY))
    }

    pub(super) fn update_lamps(&mut self) {
        if self.hierarchy.lamps.is_empty() {
            return;
        }
        let mut poses = HashMap::new();
        let links: Vec<_> = self.hierarchy.lamps.iter().map(|(k, v)| (*k, *v)).collect();
        for (id, link) in links {
            let Ok(frame) = self.link_frame(link.entity, link.bone, &mut poses) else {
                continue;
            };
            if let Some((lamp, _)) = self.lamps.get_mut(&id) {
                crate::hierarchy::place_lamp(lamp, &link, frame);
            }
        }
    }

    /// Once a frame after animation and physics: children follow parents
    /// that moved by themselves (bodies, bones) and lamps follow their
    /// entities.
    pub(crate) fn tick_hierarchy(&mut self) {
        if self.hierarchy.links.is_empty() && self.hierarchy.lamps.is_empty() {
            return;
        }
        let mut poses = HashMap::new();
        for root in self.hierarchy.roots() {
            for id in self.hierarchy.subtree(root) {
                let _ = self.place_child(id, &mut poses);
            }
        }
        self.update_lamps();
    }
}
