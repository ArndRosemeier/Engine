//! Ragdoll bodies inside [`Physics`]: a set of jointed rigid bodies owned by
//! one character entity. The pose logic lives in `world_ragdolls.rs`; this
//! file only makes, steps and reads the bodies.

use super::{pose, tag, Physics};
use crate::world::EntityId;
use glam::{Quat, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, Group, InteractionGroups, InteractionTestMode, JointAxis,
    RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle, SphericalJointBuilder,
};

/// Collision group of ragdoll torsos, heads and legs.
const RAGDOLL: Group = Group::GROUP_30;
/// Collision group of ragdoll arms: they start out against the torso, so
/// they skip every ragdoll part.
const RAGDOLL_ARM: Group = Group::GROUP_31;

/// The collider of one part, in its body's frame.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PartShape {
    /// A capsule between two points.
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: f32,
    },
    /// A rounded box.
    Box {
        center: Vec3,
        half: Vec3,
        round: f32,
    },
    Ball {
        center: Vec3,
        radius: f32,
    },
}

impl PartShape {
    pub fn center(&self) -> Vec3 {
        match *self {
            PartShape::Capsule { a, b, .. } => (a + b) * 0.5,
            PartShape::Box { center, .. } | PartShape::Ball { center, .. } => center,
        }
    }
}

/// One body of a ragdoll.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PartDesc {
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub shape: PartShape,
    pub mass: f32,
    pub arm: bool,
    pub linear_damping: f32,
    pub angular_damping: f32,
}

/// How a part hangs on its parent.
#[derive(Clone, Copy, Debug)]
pub(crate) enum JointDesc {
    /// A ball joint with angle limits about the parent's X, Y and Z axes, and
    /// the angles a faint muscle tone pulls toward.
    Ball { limits: [f32; 6], relax: [f32; 3] },
    /// A hinge about `axis` (the same in both bodies' frames).
    Hinge {
        axis: Vec3,
        limits: [f32; 2],
        relax: f32,
    },
}

/// A joint between part `parent` and part `child`. The anchor is the child's
/// body origin, given in the parent's frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LinkDesc {
    pub parent: usize,
    pub child: usize,
    pub anchor: Vec3,
    pub joint: JointDesc,
    /// Muscle tone (stiffness toward `relax`) and joint friction (damping).
    pub tone: f32,
    pub damping: f32,
}

#[derive(Debug)]
pub(crate) struct RagPart {
    handle: RigidBodyHandle,
    prev: (Vec3, Quat),
    cur: (Vec3, Quat),
}

/// The bodies of one ragdoll.
#[derive(Debug, Default)]
pub(crate) struct RagBodies {
    parts: Vec<RagPart>,
}

impl RagBodies {
    pub fn prev_cur(&mut self) {
        for p in &mut self.parts {
            p.prev = p.cur;
        }
    }
}

impl Physics {
    /// Make the bodies of `owner`'s ragdoll (replacing any it had).
    pub(crate) fn add_ragdoll(&mut self, owner: EntityId, parts: &[PartDesc], links: &[LinkDesc]) {
        self.remove_ragdoll(owner);
        let mut bodies = RagBodies::default();
        for p in parts {
            let rb = RigidBodyBuilder::dynamic()
                .pose(pose(p.position, p.rotation))
                .linvel(p.velocity)
                .linear_damping(p.linear_damping)
                .angular_damping(p.angular_damping)
                .ccd_enabled(true)
                .additional_solver_iterations(2)
                .user_data(tag(owner))
                .build();
            let handle = self.world.insert_body(rb);
            let builder = match p.shape {
                PartShape::Capsule { a, b, radius } => {
                    let d = b - a;
                    let len = d.length().max(1e-4);
                    let rot = Quat::from_rotation_arc(Vec3::Y, d / len);
                    ColliderBuilder::capsule_y(len * 0.5, radius).position(pose((a + b) * 0.5, rot))
                }
                PartShape::Box {
                    center,
                    half,
                    round,
                } => {
                    let r = round.min(half.min_element() * 0.9).max(0.0);
                    let h = (half - Vec3::splat(r)).max(Vec3::splat(1e-3));
                    ColliderBuilder::round_cuboid(h.x, h.y, h.z, r).translation(center)
                }
                PartShape::Ball { center, radius } => {
                    ColliderBuilder::ball(radius).translation(center)
                }
            };
            let groups = if p.arm {
                InteractionGroups::new(
                    RAGDOLL_ARM,
                    Group::ALL ^ (RAGDOLL | RAGDOLL_ARM),
                    InteractionTestMode::And,
                )
            } else {
                InteractionGroups::new(RAGDOLL, Group::ALL ^ RAGDOLL_ARM, InteractionTestMode::And)
            };
            self.world.insert_collider(
                builder
                    .mass(p.mass.max(0.05))
                    .friction(0.8)
                    .restitution(0.05)
                    .collision_groups(groups)
                    .user_data(tag(owner)),
                Some(handle),
            );
            bodies.parts.push(RagPart {
                handle,
                prev: (p.position, p.rotation),
                cur: (p.position, p.rotation),
            });
        }
        for l in links {
            let (Some(a), Some(b)) = (bodies.parts.get(l.parent), bodies.parts.get(l.child)) else {
                continue;
            };
            let (a, b) = (a.handle, b.handle);
            match l.joint {
                JointDesc::Ball { limits, relax } => {
                    let mut j = SphericalJointBuilder::new()
                        .local_anchor1(l.anchor)
                        .local_anchor2(Vec3::ZERO)
                        .contacts_enabled(false);
                    for (k, axis) in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ]
                        .into_iter()
                        .enumerate()
                    {
                        j = j
                            .limits(axis, [limits[k * 2], limits[k * 2 + 1]])
                            .motor(axis, relax[k], 0.0, l.tone, l.damping);
                    }
                    self.world.insert_impulse_joint(a, b, j);
                }
                JointDesc::Hinge {
                    axis,
                    limits,
                    relax,
                } => {
                    let j = RevoluteJointBuilder::new(axis.try_normalize().unwrap_or(Vec3::X))
                        .local_anchor1(l.anchor)
                        .local_anchor2(Vec3::ZERO)
                        .limits(limits)
                        .motor(relax, 0.0, l.tone * 4.0, l.damping)
                        .contacts_enabled(false);
                    self.world.insert_impulse_joint(a, b, j);
                }
            }
        }
        self.ragdolls.insert(owner, bodies);
    }

    /// Remove `owner`'s ragdoll bodies (and their joints).
    pub(crate) fn remove_ragdoll(&mut self, owner: EntityId) {
        if let Some(r) = self.ragdolls.remove(&owner) {
            for p in r.parts {
                self.world.remove_body_with_colliders(p.handle, true);
            }
        }
    }

    pub(crate) fn has_ragdoll(&self, owner: EntityId) -> bool {
        self.ragdolls.contains_key(&owner)
    }

    #[cfg(test)]
    pub(crate) fn ragdoll_count(&self) -> usize {
        self.ragdolls.len()
    }

    /// Every part's pose, drawn between the last two steps.
    pub(crate) fn ragdoll_poses(&self, owner: EntityId) -> Option<Vec<(Vec3, Quat)>> {
        let r = self.ragdolls.get(&owner)?;
        let a = self.alpha;
        Some(
            r.parts
                .iter()
                .map(|p| (p.prev.0.lerp(p.cur.0, a), p.prev.1.slerp(p.cur.1, a)))
                .collect(),
        )
    }

    /// Largest part speed and spin, and the mean velocity.
    pub(crate) fn ragdoll_motion(&self, owner: EntityId) -> Option<(f32, f32, Vec3)> {
        let r = self.ragdolls.get(&owner)?;
        let (mut v, mut w, mut sum) = (0.0f32, 0.0f32, Vec3::ZERO);
        for p in &r.parts {
            if let Some(b) = self.world.bodies.get(p.handle) {
                v = v.max(b.linvel().length());
                w = w.max(b.angvel().length());
                sum += b.linvel();
            }
        }
        Some((v, w, sum / r.parts.len().max(1) as f32))
    }

    /// Change every part's velocity by `dv` times its weight.
    pub(crate) fn ragdoll_add_velocity(&mut self, owner: EntityId, dv: Vec3, weights: &[f32]) {
        let Some(r) = self.ragdolls.get(&owner) else {
            return;
        };
        for (i, p) in r.parts.iter().enumerate() {
            if let Some(b) = self.world.bodies.get_mut(p.handle) {
                let w = weights.get(i).copied().unwrap_or(1.0);
                b.set_linvel(b.linvel() + dv * w, true);
            }
        }
    }

    /// An impulse at a world point on part `part`.
    pub(crate) fn ragdoll_impulse(
        &mut self,
        owner: EntityId,
        part: usize,
        impulse: Vec3,
        at: Vec3,
    ) {
        let Some(p) = self.ragdolls.get(&owner).and_then(|r| r.parts.get(part)) else {
            return;
        };
        if let Some(b) = self.world.bodies.get_mut(p.handle) {
            b.apply_impulse_at_point(impulse, at, true);
        }
    }

    pub(crate) fn ragdolls_mut(&mut self) -> impl Iterator<Item = &mut RagBodies> {
        self.ragdolls.values_mut()
    }

    /// Read every ragdoll part's pose after a step.
    pub(crate) fn read_ragdolls(&mut self) {
        for r in self.ragdolls.values_mut() {
            for p in &mut r.parts {
                if let Some(b) = self.world.bodies.get(p.handle) {
                    p.cur = (b.translation(), *b.rotation());
                }
            }
        }
    }

    /// Ragdoll part handles (for ground tiles and blasts).
    pub(crate) fn ragdoll_handles(&self) -> Vec<RigidBodyHandle> {
        let mut out: Vec<RigidBodyHandle> = self
            .ragdolls
            .values()
            .flat_map(|r| r.parts.iter().map(|p| p.handle))
            .collect();
        out.sort_by_key(|h| h.into_raw_parts());
        out
    }
}
