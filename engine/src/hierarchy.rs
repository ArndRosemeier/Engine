//! Parent and child objects: wheels that ride on a cart, a lantern in a
//! hand, a turret on a tank. A child keeps a place relative to its parent and
//! follows it wherever it goes (moved by you, by physics or by an animation).
//!
//! ```ignore
//! let cart = world.spawn(cart_mesh);
//! let wheel = world.spawn(wheel_mesh);
//! world.attach(wheel, cart, Place::new(0.8, 0.3, 1.0))?;
//! world.set_place(cart, Place::new(10.0, 0.0, 4.0))?; // the wheel comes along
//! world.set_place(wheel, Place::new(0.8, 0.3, 1.0).with_pitch_deg(spin))?; // relative to the cart
//! ```
//!
//! [`crate::world::World::spawn_pivot`] makes an invisible handle to group
//! things under, and [`crate::world::World::attach_to_bone`] hangs things on
//! a character's skeleton.

use crate::lamps::{Lamp, LampId, LampShape};
use crate::world::EntityId;
use glam::{Mat4, Vec3};
use std::collections::{BTreeMap, HashMap};

/// How a child hangs on its parent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Link {
    pub parent: EntityId,
    /// Place relative to the parent (or to the parent's bone).
    pub local: Mat4,
    /// Joint index on an animated parent.
    pub bone: Option<usize>,
}

/// A lamp riding on an entity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LampLink {
    pub entity: EntityId,
    pub bone: Option<usize>,
    /// Position relative to the entity.
    pub offset: Vec3,
    /// Spot direction relative to the entity.
    pub direction: Option<Vec3>,
}

#[derive(Default, Debug)]
pub(crate) struct Hierarchy {
    pub links: HashMap<EntityId, Link>,
    pub children: HashMap<EntityId, Vec<EntityId>>,
    /// Invisible handles: just a place.
    pub pivots: HashMap<EntityId, Mat4>,
    pub lamps: BTreeMap<LampId, LampLink>,
}

impl Hierarchy {
    pub fn link(&mut self, child: EntityId, link: Link) {
        self.unlink(child);
        self.children.entry(link.parent).or_default().push(child);
        self.links.insert(child, link);
    }

    pub fn unlink(&mut self, child: EntityId) -> Option<Link> {
        let link = self.links.remove(&child)?;
        if let Some(list) = self.children.get_mut(&link.parent) {
            list.retain(|c| *c != child);
            if list.is_empty() {
                self.children.remove(&link.parent);
            }
        }
        Some(link)
    }

    /// Whether `ancestor` is `id` or above it.
    pub fn is_ancestor(&self, ancestor: EntityId, mut id: EntityId) -> bool {
        loop {
            if id == ancestor {
                return true;
            }
            match self.links.get(&id) {
                Some(link) => id = link.parent,
                None => return false,
            }
        }
    }

    pub fn children_of(&self, id: EntityId) -> &[EntityId] {
        self.children.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    /// `id` and everything below it, parents before children.
    pub fn subtree(&self, id: EntityId) -> Vec<EntityId> {
        let mut out = vec![id];
        let mut i = 0;
        while i < out.len() {
            out.extend_from_slice(self.children_of(out[i]));
            i += 1;
        }
        out
    }

    /// Entities with children but no parent: where updates start.
    pub fn roots(&self) -> Vec<EntityId> {
        let mut roots: Vec<EntityId> = self
            .children
            .keys()
            .filter(|id| !self.links.contains_key(id))
            .copied()
            .collect();
        roots.sort_by_key(|id| id.raw());
        roots
    }

    /// Forget everything about `id` (it was despawned). Returns its children
    /// and the lamps riding on it.
    pub fn forget(&mut self, id: EntityId) -> (Vec<EntityId>, Vec<LampId>) {
        self.unlink(id);
        self.pivots.remove(&id);
        let children = self.children.remove(&id).unwrap_or_default();
        for c in &children {
            self.links.remove(c);
        }
        let lamps: Vec<LampId> = self
            .lamps
            .iter()
            .filter(|(_, l)| l.entity == id)
            .map(|(k, _)| *k)
            .collect();
        for l in &lamps {
            self.lamps.remove(l);
        }
        (children, lamps)
    }
}

/// Where an attached lamp is, given the world matrix of what it rides on.
pub(crate) fn place_lamp(lamp: &mut Lamp, link: &LampLink, frame: Mat4) {
    lamp.position = frame.transform_point3(link.offset);
    if let (LampShape::Spot { direction, .. }, Some(local)) = (&mut lamp.shape, link.direction) {
        let d = frame.transform_vector3(local);
        if d.length_squared() > 1e-12 {
            *direction = d.normalize();
        }
    }
}

/// Whether two matrices are the same apart from float noise (so a child
/// that didn't move isn't re-uploaded).
pub(crate) fn same_up_to_rounding(a: Mat4, b: Mat4) -> bool {
    a.abs_diff_eq(b, 1e-6)
}
