//! Joints: hinges, ball joints, sliders, ropes and welds between bodies.
//!
//! Part of `crate::physics`, split out by topic.

use super::*;
use rapier3d::prelude::{
    GenericJointBuilder, ImpulseJointHandle, JointAxesMask, JointAxis, RopeJointBuilder,
};

/// A joint between two bodies, or between a body and a fixed point in the
/// world. Start from [`Joint::hinge`], [`Joint::ball`], [`Joint::slider`],
/// [`Joint::rope`] or [`Joint::weld`], then say where with [`Joint::at`].
///
/// ```ignore
/// // A door hinged on its left edge, swinging up to 100 degrees.
/// world.pin(door, Joint::hinge(Vec3::Y).at((-0.5, 1.0, 0.0)).limits(0.0, 100.0))?;
/// // A wheel on an axle, driven at 2 turns a second.
/// world.add_joint(cart, wheel, Joint::hinge(Vec3::X).at(axle).motor(720.0, 500.0))?;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Joint {
    pub kind: JointKind,
    /// The world point the joint sits on when it is made. `None` (default)
    /// uses the second body's place (the first's for a rope), or the
    /// body's own place for [`crate::World::pin`]. Limits and motors move
    /// the second body relative to the first (a pinned body relative to
    /// the world).
    pub anchor: Option<Vec3>,
    /// Let the two joined bodies collide with each other (default off).
    pub collide: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    /// Glued together.
    Weld,
    /// Turns around `axis` only (doors, wheels, levers). `limits` in
    /// degrees from the starting angle; `motor` is a target speed in
    /// degrees per second and the most force it may use.
    Hinge {
        axis: Vec3,
        limits: Option<(f32, f32)>,
        motor: Option<(f32, f32)>,
    },
    /// Turns any way around one point (a chain link, a pendulum, a hanging lamp).
    Ball,
    /// Slides along `axis` only (drawers, pistons, lifts). `limits` in metres
    /// from the start; `motor` is a target speed in m/s and the most force.
    Slider {
        axis: Vec3,
        limits: Option<(f32, f32)>,
        motor: Option<(f32, f32)>,
    },
    /// Keeps the second body's centre (the pinned body's, for a pin) at
    /// most `length` metres from the anchor on the first (or in the world).
    Rope { length: f32 },
}

impl Joint {
    fn of(kind: JointKind) -> Self {
        Self {
            kind,
            anchor: None,
            collide: false,
        }
    }

    pub fn weld() -> Self {
        Self::of(JointKind::Weld)
    }

    /// Turns around `axis` (in world space).
    pub fn hinge(axis: impl Into<Vec3>) -> Self {
        Self::of(JointKind::Hinge {
            axis: axis.into(),
            limits: None,
            motor: None,
        })
    }

    pub fn ball() -> Self {
        Self::of(JointKind::Ball)
    }

    /// Slides along `axis` (in world space).
    pub fn slider(axis: impl Into<Vec3>) -> Self {
        Self::of(JointKind::Slider {
            axis: axis.into(),
            limits: None,
            motor: None,
        })
    }

    pub fn rope(length: f32) -> Self {
        Self::of(JointKind::Rope { length })
    }

    /// Where the joint sits, in world space.
    pub fn at(mut self, point: impl Into<Vec3>) -> Self {
        self.anchor = Some(point.into());
        self
    }

    /// Hinge: degrees either side of the start. Slider: metres.
    pub fn limits(mut self, min: f32, max: f32) -> Self {
        match &mut self.kind {
            JointKind::Hinge { limits, .. } | JointKind::Slider { limits, .. } => {
                *limits = Some((min, max))
            }
            _ => {}
        }
        self
    }

    /// Hinge: drive at `speed` degrees per second. Slider: m/s. `max_force`
    /// caps how hard it pushes (N, or N·m for a hinge).
    pub fn motor(mut self, speed: f32, max_force: f32) -> Self {
        match &mut self.kind {
            JointKind::Hinge { motor, .. } | JointKind::Slider { motor, .. } => {
                *motor = Some((speed, max_force))
            }
            _ => {}
        }
        self
    }

    /// Let the joined bodies collide with each other.
    pub fn collide(mut self, on: bool) -> Self {
        self.collide = on;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("joint {what}")));
        if self.anchor.is_some_and(|a| !a.is_finite()) {
            return bad("anchor must be finite");
        }
        let pair_ok = |p: Option<(f32, f32)>| p.is_none_or(|(a, b)| a.is_finite() && b.is_finite());
        match self.kind {
            JointKind::Hinge {
                axis,
                limits,
                motor,
            }
            | JointKind::Slider {
                axis,
                limits,
                motor,
            } => {
                if !axis.is_finite() || axis.length_squared() < 1e-8 {
                    return bad("axis must be a finite, non-zero direction");
                }
                if !pair_ok(limits) || limits.is_some_and(|(a, b)| a > b) {
                    return bad("limits must be finite with min <= max");
                }
                if !pair_ok(motor) || motor.is_some_and(|(_, f)| f < 0.0) {
                    return bad("motor needs a finite speed and a force >= 0");
                }
            }
            JointKind::Rope { length } if !(length.is_finite() && length > 0.0) => {
                return bad("rope length must be > 0");
            }
            _ => {}
        }
        Ok(())
    }
}

/// How hard a motor holds its target speed (rapier's motor damping): high,
/// so it is the force cap that limits it, not a soft spring.
const MOTOR_GRIP: f32 = 1000.0;

/// A joint you made; pass it to [`crate::World::remove_joint`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JointId(pub(crate) u64);

#[derive(Debug)]
pub(super) struct JointTracked {
    handle: ImpulseJointHandle,
    /// The fixed body a pinned joint hangs on.
    anchor_body: Option<RigidBodyHandle>,
}

impl Physics {
    fn handle_of(&self, id: EntityId) -> EngineResult<RigidBodyHandle> {
        self.bodies.get(&id).map(|t| t.handle).ok_or_else(|| {
            EngineError::InvalidValue(format!("joint: entity {id} has no body (add_body first)"))
        })
    }

    /// Join `a` to `b` (or to a fixed point in the world when `b` is `None`).
    pub fn add_joint(
        &mut self,
        a: EntityId,
        b: Option<EntityId>,
        joint: &Joint,
    ) -> EngineResult<JointId> {
        joint.validate()?;
        if b == Some(a) {
            return Err(EngineError::InvalidValue(
                "joint: a body cannot be joined to itself".into(),
            ));
        }
        let ha = self.handle_of(a)?;
        let hb = b.map(|b| self.handle_of(b)).transpose()?;
        let place = |h: RigidBodyHandle| self.world.bodies[h].translation();
        let anchor = joint.anchor.unwrap_or_else(|| match (hb, joint.kind) {
            (Some(hb), JointKind::Rope { .. }) => {
                let _ = hb;
                place(ha)
            }
            (Some(hb), _) => place(hb),
            (None, _) => place(ha),
        });
        // Rapier moves body 2 relative to body 1 (limits, motors): `b`
        // relative to `a`, or a pinned body relative to the world, where a
        // fixed body at the anchor stands in for the world.
        let (h1, h2, anchor_body) = match hb {
            Some(hb) => (ha, hb, None),
            None => {
                let fixed = self
                    .world
                    .insert_body(RigidBodyBuilder::fixed().translation(anchor).build());
                (fixed, ha, Some(fixed))
            }
        };
        let pos1 = *self.world.bodies[h1].position();
        let pos2 = *self.world.bodies[h2].position();
        let axis = match joint.kind {
            JointKind::Hinge { axis, .. } | JointKind::Slider { axis, .. } => axis.normalize(),
            _ => Vec3::X,
        };
        // The joint's frame in the world: at the anchor, its x axis along
        // the hinge or slider axis.
        let frame = pose(anchor, Quat::from_rotation_arc(Vec3::X, axis));
        let local1 = pos1.inverse() * frame;
        let local2 = pos2.inverse() * frame;
        let generic = |mask: JointAxesMask| {
            GenericJointBuilder::new(mask)
                .local_frame1(local1)
                .local_frame2(local2)
                .contacts_enabled(joint.collide)
        };
        let built = match joint.kind {
            JointKind::Weld => generic(JointAxesMask::LOCKED_FIXED_AXES).build(),
            JointKind::Ball => generic(JointAxesMask::LOCKED_SPHERICAL_AXES).build(),
            JointKind::Hinge { limits, motor, .. } => {
                let mut g = generic(JointAxesMask::LOCKED_REVOLUTE_AXES);
                if let Some((lo, hi)) = limits {
                    g = g.limits(JointAxis::AngX, [lo.to_radians(), hi.to_radians()]);
                }
                if let Some((speed, force)) = motor {
                    g = g
                        .motor_velocity(JointAxis::AngX, speed.to_radians(), MOTOR_GRIP)
                        .motor_max_force(JointAxis::AngX, force);
                }
                g.build()
            }
            JointKind::Slider { limits, motor, .. } => {
                let mut g = generic(JointAxesMask::LOCKED_PRISMATIC_AXES);
                if let Some((lo, hi)) = limits {
                    g = g.limits(JointAxis::LinX, [lo, hi]);
                }
                if let Some((speed, force)) = motor {
                    g = g
                        .motor_velocity(JointAxis::LinX, speed, MOTOR_GRIP)
                        .motor_max_force(JointAxis::LinX, force);
                }
                g.build()
            }
            // The rope runs from the anchor to the moving body's centre.
            JointKind::Rope { length } => RopeJointBuilder::new(length)
                .local_anchor1(pos1.inverse() * anchor)
                .local_anchor2(Vec3::ZERO)
                .contacts_enabled(joint.collide)
                .build()
                .into(),
        };
        let handle = self.world.insert_impulse_joint(h1, h2, built);
        self.world.wake_up(h1, true);
        self.world.wake_up(h2, true);
        self.next_joint += 1;
        let id = JointId(self.next_joint);
        self.joints.insert(
            id,
            JointTracked {
                handle,
                anchor_body,
            },
        );
        Ok(id)
    }

    pub fn remove_joint(&mut self, id: JointId) {
        if let Some(j) = self.joints.remove(&id) {
            if let Some((b1, b2)) = self
                .world
                .impulse_joints
                .get(j.handle)
                .map(|g| (g.body1(), g.body2()))
            {
                self.world.remove_impulse_joint(j.handle);
                self.world.wake_up(b1, true);
                self.world.wake_up(b2, true);
            }
            if let Some(b) = j.anchor_body {
                self.world.remove_body(b);
            }
        }
    }

    /// Joints still holding (removing a body removes its joints).
    pub fn joint_count(&self) -> usize {
        self.joints
            .values()
            .filter(|j| self.world.impulse_joints.get(j.handle).is_some())
            .count()
    }

    pub fn has_joint(&self, id: JointId) -> bool {
        self.joints
            .get(&id)
            .is_some_and(|j| self.world.impulse_joints.get(j.handle).is_some())
    }

    /// Forget joints whose bodies are gone, with their world anchors.
    pub(super) fn drop_broken_joints(&mut self) {
        let mut gone: Vec<JointId> = self
            .joints
            .iter()
            .filter(|(_, j)| self.world.impulse_joints.get(j.handle).is_none())
            .map(|(id, _)| *id)
            .collect();
        gone.sort();
        for id in gone {
            self.remove_joint(id);
        }
    }
}
