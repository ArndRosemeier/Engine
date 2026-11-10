//! Gameplay physics: collision layers, trigger zones, and overlap and
//! shape-cast queries.
//!
//! Part of `crate::physics`, split out by topic.

use super::*;
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::parry::shape::{Ball, Cuboid, Shape};
use rapier3d::prelude::{ActiveCollisionTypes, Group, InteractionGroups, InteractionTestMode};
use std::collections::BTreeSet;

/// The highest collision layer a body can be on (layers are 0 to 27).
pub const MAX_LAYER: u8 = 27;
/// Interaction-group bits the engine keeps for itself: the ground, trigger
/// zones and ragdoll parts.
const GROUND_BIT: u32 = 1 << 31;
const TRIGGER_BIT: u32 = 1 << 30;
pub(super) const RAGDOLL_BIT: u32 = 1 << 29;
pub(super) const RAGDOLL_ARM_BIT: u32 = 1 << 28;
const RAGDOLL_BITS: u32 = RAGDOLL_BIT | RAGDOLL_ARM_BIT;
const LAYER_BITS: u32 = (1 << 28) - 1;

fn groups(memberships: u32, filter: u32) -> InteractionGroups {
    InteractionGroups::new(
        Group::from_bits_retain(memberships),
        Group::from_bits_retain(filter),
        InteractionTestMode::And,
    )
}

/// A body on `layer` that collides with the layers in `collides_with`, and
/// always with the ground, triggers and ragdolls.
pub(super) fn body_groups(layer: u8, collides_with: u32) -> InteractionGroups {
    groups(
        1 << layer.min(MAX_LAYER),
        (collides_with & LAYER_BITS) | GROUND_BIT | TRIGGER_BIT | RAGDOLL_BITS,
    )
}

/// Ground, terrain tiles and building boxes: every body lands on them.
pub(super) fn ground_groups() -> InteractionGroups {
    groups(GROUND_BIT, LAYER_BITS | RAGDOLL_BITS)
}

/// The shape of a trigger zone, in metres around the entity that carries it
/// (the entity's scale does not change it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TriggerShape {
    /// A sphere with this radius.
    Ball(f32),
    /// A box with these half extents, turned with the entity.
    Box(Vec3),
    /// An upright capsule: half the straight part's length, and the radius.
    Capsule { half_height: f32, radius: f32 },
}

/// An invisible zone that reports what goes in and out: a door that opens,
/// a coin to pick up, a finish line, lava. See [`crate::World::add_trigger`].
///
/// Triggers notice entities with a body ([`Body`], any kind), characters
/// and ragdolls;
/// they never push anything, and rays and overlap queries pass through them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trigger {
    pub shape: TriggerShape,
    /// Layers it notices, one bit per layer (default: all).
    pub layers: u32,
}

impl Trigger {
    /// A sphere zone.
    pub fn ball(radius: f32) -> Self {
        Self {
            shape: TriggerShape::Ball(radius),
            layers: LAYER_BITS,
        }
    }

    /// A box zone with these half extents (a 4 x 2 x 4 m room is
    /// `Trigger::cuboid((2.0, 1.0, 2.0))`).
    pub fn cuboid(half_extents: impl Into<Vec3>) -> Self {
        Self {
            shape: TriggerShape::Box(half_extents.into()),
            layers: LAYER_BITS,
        }
    }

    /// An upright capsule zone.
    pub fn capsule(half_height: f32, radius: f32) -> Self {
        Self {
            shape: TriggerShape::Capsule {
                half_height,
                radius,
            },
            layers: LAYER_BITS,
        }
    }

    /// Notice only bodies on these layers.
    pub fn layers(mut self, layers: &[u8]) -> Self {
        self.layers = layers
            .iter()
            .filter(|l| **l <= MAX_LAYER)
            .fold(0, |m, l| m | 1 << l);
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let ok = match self.shape {
            TriggerShape::Ball(r) => r.is_finite() && r > 0.0,
            TriggerShape::Box(h) => h.is_finite() && h.min_element() > 0.0,
            TriggerShape::Capsule {
                half_height,
                radius,
            } => {
                half_height.is_finite() && half_height >= 0.0 && radius.is_finite() && radius > 0.0
            }
        };
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidValue(format!(
                "trigger shape needs finite sizes above 0, got {:?}",
                self.shape
            )))
        }
    }

    fn collider(&self) -> ColliderBuilder {
        match self.shape {
            TriggerShape::Ball(r) => ColliderBuilder::ball(r),
            TriggerShape::Box(h) => ColliderBuilder::cuboid(h.x, h.y, h.z),
            TriggerShape::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(half_height, radius),
        }
    }
}

/// Something went into or came out of a trigger. A body that is despawned
/// while inside comes out (`entered == false`) on the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerEvent {
    /// The entity carrying the trigger.
    pub trigger: EntityId,
    /// The entity that went in or came out.
    pub other: EntityId,
    /// `true` when it went in, `false` when it came out.
    pub entered: bool,
}

#[derive(Debug)]
pub(super) struct TriggerTracked {
    handle: RigidBodyHandle,
    collider: ColliderHandle,
    written: Mat4,
    inside: BTreeSet<EntityId>,
}

impl Physics {
    /// Give `id` a trigger zone at `transform` (scale is ignored). Replaces
    /// a trigger it already has.
    pub fn add_trigger(
        &mut self,
        id: EntityId,
        trigger: &Trigger,
        transform: Mat4,
    ) -> EngineResult<()> {
        trigger.validate()?;
        if !transform.is_finite() {
            return Err(EngineError::InvalidValue(
                "trigger transform must be finite".into(),
            ));
        }
        self.remove_trigger(id);
        let (_, r, p) = split(transform);
        let handle = self.world.insert_body(
            RigidBodyBuilder::kinematic_position_based()
                .pose(pose(p, r))
                .user_data(tag(id))
                .build(),
        );
        let collider = self.world.insert_collider(
            trigger
                .collider()
                .sensor(true)
                .active_collision_types(ActiveCollisionTypes::all())
                .collision_groups(groups(
                    TRIGGER_BIT,
                    (trigger.layers & LAYER_BITS) | RAGDOLL_BITS,
                ))
                .user_data(tag(id)),
            Some(handle),
        );
        self.triggers.insert(
            id,
            TriggerTracked {
                handle,
                collider,
                written: transform,
                inside: BTreeSet::new(),
            },
        );
        Ok(())
    }

    pub fn remove_trigger(&mut self, id: EntityId) {
        if let Some(t) = self.triggers.remove(&id) {
            self.world.remove_body_with_colliders(t.handle, true);
        }
    }

    pub fn has_trigger(&self, id: EntityId) -> bool {
        self.triggers.contains_key(&id)
    }

    pub fn trigger_count(&self) -> usize {
        self.triggers.len()
    }

    /// Entities carrying triggers, in id order.
    pub fn trigger_ids(&self) -> Vec<EntityId> {
        let mut ids: Vec<EntityId> = self.triggers.keys().copied().collect();
        ids.sort_by_key(|id| id.raw());
        ids
    }

    /// Follow the entity carrying a trigger; `None` means it is gone.
    pub fn move_trigger(&mut self, id: EntityId, transform: Option<Mat4>) {
        let Some(m) = transform.filter(|m| m.is_finite()) else {
            self.remove_trigger(id);
            return;
        };
        let Some(t) = self.triggers.get_mut(&id) else {
            return;
        };
        if t.written == m {
            return;
        }
        t.written = m;
        let (_, r, p) = split(m);
        if let Some(b) = self.world.bodies.get_mut(t.handle) {
            b.set_position(pose(p, r), true);
        }
    }

    /// The entity a collider belongs to, if it is a body or ragdoll part
    /// (not ground, not a trigger).
    fn thing_of(&self, c: &rapier3d::prelude::Collider) -> Option<EntityId> {
        if c.user_data == GROUND_TAG || c.is_sensor() {
            return None;
        }
        let id = EntityId::from_raw(c.user_data as u64);
        (self.bodies.contains_key(&id)
            || self.ragdolls.contains_key(&id)
            || self.characters.contains_key(&id))
        .then_some(id)
    }

    /// Compare what is in each trigger now with the last step, and record
    /// who went in and who came out.
    pub(super) fn update_triggers(&mut self) {
        for id in self.trigger_ids() {
            let t = &self.triggers[&id];
            let mut now = BTreeSet::new();
            for (h1, c1, _, c2, hit) in self.world.intersection_pairs_with(t.collider) {
                if !hit {
                    continue;
                }
                let other = if h1 == t.collider { c2 } else { c1 };
                if let Some(e) = self.thing_of(other).filter(|e| *e != id) {
                    now.insert(e);
                }
            }
            let t = self.triggers.get_mut(&id).expect("trigger");
            for &e in now.difference(&t.inside) {
                self.trigger_events.push(TriggerEvent {
                    trigger: id,
                    other: e,
                    entered: true,
                });
            }
            for &e in t.inside.difference(&now) {
                self.trigger_events.push(TriggerEvent {
                    trigger: id,
                    other: e,
                    entered: false,
                });
            }
            t.inside = now;
        }
        if self.trigger_events.len() > 4096 {
            // Nobody is taking them; keep the newest.
            let cut = self.trigger_events.len() - 2048;
            self.trigger_events.drain(..cut);
        }
    }

    /// Entities inside a trigger after the last step, in id order.
    pub fn in_trigger(&self, id: EntityId) -> Vec<EntityId> {
        self.triggers
            .get(&id)
            .map_or_else(Vec::new, |t| t.inside.iter().copied().collect())
    }

    /// Everything that went in or out since the last call, in order.
    pub fn take_trigger_events(&mut self) -> Vec<TriggerEvent> {
        std::mem::take(&mut self.trigger_events)
    }

    fn overlap(&self, shape: &dyn Shape, at: rapier3d::math::Pose) -> Vec<EntityId> {
        let filter = QueryFilter::default().exclude_sensors();
        let mut out: Vec<EntityId> = self
            .world
            .intersect_shape(at, shape, filter)
            .filter_map(|(_, c)| self.thing_of(c))
            .collect();
        out.sort_by_key(|id| id.raw());
        out.dedup();
        out
    }

    /// Bodies (and ragdolls) touching a sphere, in id order.
    pub fn overlap_ball(&self, center: Vec3, radius: f32) -> Vec<EntityId> {
        self.overlap(&Ball::new(radius), pose(center, Quat::IDENTITY))
    }

    /// Bodies (and ragdolls) touching a box, in id order.
    pub fn overlap_box(&self, center: Vec3, half: Vec3, rotation: Quat) -> Vec<EntityId> {
        self.overlap(&Cuboid::new(half), pose(center, rotation))
    }

    /// Sweep a sphere from `origin` along unit `dir`; the first body or
    /// built ground collider it touches within `max`. `distance` is how far
    /// the sphere's centre travelled.
    pub fn cast_ball(
        &self,
        origin: Vec3,
        dir: Vec3,
        radius: f32,
        max: f32,
        ignore: Option<EntityId>,
    ) -> Option<RayHit> {
        let dir = dir.try_normalize()?;
        let ignore_raw = ignore.map(|id| id.raw());
        let skip_ground = |_: ColliderHandle, c: &rapier3d::prelude::Collider| {
            c.user_data != GROUND_TAG && Some(c.user_data as u64) != ignore_raw
        };
        let filter = QueryFilter::default()
            .exclude_sensors()
            .predicate(&skip_ground);
        let (handle, hit) = self.world.cast_shape(
            &pose(origin, Quat::IDENTITY),
            dir,
            &Ball::new(radius),
            ShapeCastOptions::with_max_time_of_impact(max),
            filter,
        )?;
        let c = self.world.colliders.get(handle)?;
        let normal = hit.normal1.try_normalize().unwrap_or(-dir);
        Some(RayHit {
            entity: self.thing_of(c),
            point: hit.witness1,
            normal,
            distance: hit.time_of_impact,
        })
    }
}
