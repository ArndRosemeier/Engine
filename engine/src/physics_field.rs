//! Gravity from a field: each body takes its own acceleration from the game.
//!
//! By default every body falls with [`PhysicsSettings::gravity`], one vector
//! for the whole world. A game where "down" differs from place to place
//! (planets, a ship that spins, attraction stones) sets a [`GravityField`]
//! instead: every fixed step the field is asked, for each moving dynamic
//! body, ragdoll part and character, which acceleration acts on it there.
//! The answer may be zero (bodies drift) and may differ by mass or by body.
//!
//! A field is a value the game replaces whenever its sources change: build
//! a cheap snapshot (a list of sources), wrap it in an `Arc` and hand it to
//! [`crate::world::World::set_gravity_field`]. Physics only reads it, so the
//! same snapshot gives the same steps on every machine.
//!
//! Part of `crate::physics`, split out by topic.

use super::*;
use std::sync::Arc;

/// Which kind of thing a [`FieldProbe`] asks about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldBody {
    /// A dynamic body (`World::add_body`, rubble, `PhysicsWorld::add`).
    Body,
    /// One part of a ragdoll; `id` is the character that owns it.
    RagdollPart,
    /// A walking character (`World::add_character`).
    Character,
}

/// One question to a [`GravityField`]: a body, where it is and how heavy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldProbe {
    /// The body's entity as a raw id: [`FieldProbe::entity`] in a `World`,
    /// [`FieldProbe::body`] in a [`PhysicsWorld`].
    pub id: u64,
    pub kind: FieldBody,
    /// Centre of mass, in world space.
    pub position: Vec3,
    /// Mass in kg.
    pub mass: f32,
}

impl FieldProbe {
    /// The entity this probe is about (in a `World`).
    pub fn entity(&self) -> EntityId {
        EntityId::from_raw(self.id)
    }

    /// The body this probe is about (in a [`PhysicsWorld`]).
    pub fn body(&self) -> BodyId {
        BodyId::from_raw(self.id)
    }
}

/// Acceleration in m/s² for a body at a point. Any
/// `Fn(&FieldProbe) -> Vec3 + Send + Sync` is a field.
///
/// ```
/// use engine::physics::{FieldProbe, GravityField};
/// use engine::glam::Vec3;
///
/// // Everything falls towards the origin at 9.81 m/s², nothing within 0.1 m.
/// let planet = |p: &FieldProbe| {
///     let to = -p.position;
///     if to.length() < 0.1 { Vec3::ZERO } else { to.normalize() * 9.81 }
/// };
/// let probe = FieldProbe {
///     id: 1,
///     kind: engine::physics::FieldBody::Body,
///     position: Vec3::new(0.0, 0.0, 5.0),
///     mass: 10.0,
/// };
/// assert!((planet.acceleration(&probe) - Vec3::new(0.0, 0.0, -9.81)).length() < 1e-5);
/// ```
pub trait GravityField: Send + Sync {
    fn acceleration(&self, probe: &FieldProbe) -> Vec3;
}

impl<F: Fn(&FieldProbe) -> Vec3 + Send + Sync> GravityField for F {
    fn acceleration(&self, probe: &FieldProbe) -> Vec3 {
        self(probe)
    }
}

/// A shared field, as physics keeps it.
pub type SharedField = Arc<dyn GravityField>;

/// A sleeping body wakes when its acceleration changes by more than this
/// (m/s²), so a stone opened next to a pile sets it moving.
const WAKE_CHANGE: f32 = 0.05;

impl Physics {
    /// Use `field` for every moving body from the next step on; `None` goes
    /// back to [`PhysicsSettings::gravity`].
    pub fn set_gravity_field(&mut self, field: Option<SharedField>) {
        if field.is_some() != self.field.is_some() {
            // Switching between world gravity and a field changes what acts
            // on sleeping bodies too. (A new field wakes bodies one by one,
            // where its answer changed.)
            self.world.wake_up_all(true);
        }
        self.field = field;
        self.world.gravity = if self.field.is_some() {
            Vec3::ZERO
        } else {
            self.settings.gravity
        };
    }

    pub fn has_gravity_field(&self) -> bool {
        self.field.is_some()
    }

    /// The acceleration acting on a body at `position`: the field's answer,
    /// or the world gravity without one. Not finite answers count as zero.
    pub fn gravity_for(&self, id: u64, kind: FieldBody, position: Vec3, mass: f32) -> Vec3 {
        match &self.field {
            None => self.settings.gravity,
            Some(f) => {
                let a = f.acceleration(&FieldProbe {
                    id,
                    kind,
                    position,
                    mass,
                });
                if a.is_finite() {
                    a
                } else {
                    Vec3::ZERO
                }
            }
        }
    }

    /// One fixed step: push every dynamic body and ragdoll part by the
    /// field. Runs after forces were reset for the step. Sleeping bodies
    /// stay asleep unless their acceleration changed since they last moved.
    pub(super) fn apply_field(&mut self) {
        let Some(field) = self.field.clone() else {
            return;
        };
        let ask = |id: u64, kind: FieldBody, position: Vec3, mass: f32| {
            let a = field.acceleration(&FieldProbe {
                id,
                kind,
                position,
                mass,
            });
            if a.is_finite() {
                a
            } else {
                Vec3::ZERO
            }
        };
        for (id, t) in self.bodies.iter_mut() {
            if t.kind != BodyKind::Dynamic {
                continue;
            }
            let Some(b) = self.world.bodies.get_mut(t.handle) else {
                continue;
            };
            let mass = b.mass();
            let a = ask(id.raw(), FieldBody::Body, b.center_of_mass(), mass);
            if b.is_sleeping() {
                if a.distance(t.field_accel) <= WAKE_CHANGE {
                    continue;
                }
                b.wake_up(true);
            }
            t.field_accel = a;
            b.add_force(a * mass, false);
        }
        for (owner, r) in self.ragdolls.iter_mut() {
            for p in r.parts_mut() {
                let Some(b) = self.world.bodies.get_mut(p.handle()) else {
                    continue;
                };
                let mass = b.mass();
                let a = ask(
                    owner.raw(),
                    FieldBody::RagdollPart,
                    b.center_of_mass(),
                    mass,
                );
                if b.is_sleeping() {
                    if a.distance(p.field_accel) <= WAKE_CHANGE {
                        continue;
                    }
                    b.wake_up(true);
                }
                p.field_accel = a;
                b.reset_forces(false);
                b.add_force(a * mass, false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> PhysicsWorld {
        PhysicsWorld::with_settings(PhysicsSettings {
            ground: PhysicsGround::None,
            ..PhysicsSettings::default()
        })
        .unwrap()
    }

    fn ball(w: &mut PhysicsWorld, at: Vec3, density: f32) -> BodyId {
        w.add(
            &Body::dynamic()
                .shape(BodyShape::Ball(0.25))
                .density(density)
                .damping(0.0),
            Mat4::from_translation(at),
        )
        .unwrap()
    }

    fn run(w: &mut PhysicsWorld, seconds: f32) {
        for _ in 0..(seconds * 60.0).round() as usize {
            w.step(1.0 / 60.0);
        }
    }

    #[test]
    fn bodies_fall_towards_a_point_attractor() {
        let mut w = space();
        let b = ball(&mut w, Vec3::new(5.0, 0.0, 0.0), 1.0);
        w.set_gravity_field(Some(Arc::new(|p: &FieldProbe| {
            -p.position.normalize_or_zero() * 4.0
        })));
        run(&mut w, 1.0);
        let p = w.position(b).unwrap();
        // x = 5 - a t² / 2 = 3; no world gravity in y.
        assert!((p.x - 3.0).abs() < 0.1, "x {}", p.x);
        assert!(p.y.abs() < 1e-3 && p.z.abs() < 1e-3, "{p}");
    }

    #[test]
    fn zero_field_lets_bodies_drift() {
        let mut w = space();
        let b = ball(&mut w, Vec3::ZERO, 1.0);
        w.set_velocity(b, Vec3::new(0.0, 1.0, 0.0)).unwrap();
        w.set_gravity_field(Some(Arc::new(|_: &FieldProbe| Vec3::ZERO)));
        run(&mut w, 2.0);
        let v = w.velocity(b).unwrap();
        assert!((v - Vec3::Y).length() < 1e-3, "{v}");
        assert!((w.position(b).unwrap().y - 2.0).abs() < 0.05);
    }

    #[test]
    fn a_field_can_pick_bodies_by_mass() {
        let mut w = space();
        let light = ball(&mut w, Vec3::new(0.0, 0.0, 0.0), 0.2);
        let heavy = ball(&mut w, Vec3::new(3.0, 0.0, 0.0), 8.0);
        let cut = (w.mass(light).unwrap() + w.mass(heavy).unwrap()) * 0.5;
        w.set_gravity_field(Some(Arc::new(move |p: &FieldProbe| {
            if p.mass < cut {
                Vec3::Z * 9.81
            } else {
                Vec3::ZERO
            }
        })));
        run(&mut w, 1.0);
        assert!(w.position(light).unwrap().z > 4.0);
        assert!(w.position(heavy).unwrap().z.abs() < 1e-3);
    }

    #[test]
    fn a_resting_body_wakes_when_the_field_turns() {
        let mut w = space();
        w.add(
            &Body::fixed().shape(BodyShape::Box(Vec3::new(5.0, 0.5, 5.0))),
            Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)),
        )
        .unwrap();
        let b = w
            .add(
                &Body::dynamic().shape(BodyShape::Box(Vec3::splat(0.25))),
                Mat4::from_translation(Vec3::new(0.0, 0.3, 0.0)),
            )
            .unwrap();
        w.set_gravity_field(Some(Arc::new(|_: &FieldProbe| Vec3::NEG_Y * 9.81)));
        run(&mut w, 4.0);
        assert!(w.inner.is_sleeping(b.key()), "box should rest and sleep");
        let rest = w.position(b).unwrap();
        // Down turns to +x: the box slides off the side.
        w.set_gravity_field(Some(Arc::new(|_: &FieldProbe| Vec3::X * 9.81)));
        run(&mut w, 1.0);
        let p = w.position(b).unwrap();
        assert!(p.x > rest.x + 2.0, "{rest} -> {p}");
    }

    #[test]
    fn a_bad_answer_counts_as_zero_and_none_restores_gravity() {
        let mut w = space();
        let b = ball(&mut w, Vec3::ZERO, 1.0);
        w.set_gravity_field(Some(Arc::new(|_: &FieldProbe| Vec3::splat(f32::NAN))));
        run(&mut w, 0.5);
        assert!(w.position(b).unwrap().length() < 1e-3);
        w.set_gravity_field(None);
        run(&mut w, 1.0);
        assert!(w.position(b).unwrap().y < -4.0);
    }

    #[test]
    fn characters_follow_the_fields_vertical_part() {
        let mut p = Physics::new();
        p.set_settings(PhysicsSettings {
            ground: PhysicsGround::Plane(0.0),
            ..PhysicsSettings::default()
        });
        let id = EntityId::from_raw(7);
        p.add_character(id, &Character::default(), Vec3::new(0.0, 10.0, 0.0))
            .unwrap();
        // A light district: 0.3 g down.
        p.set_gravity_field(Some(Arc::new(|_: &FieldProbe| Vec3::NEG_Y * 0.3 * 9.81)));
        let around = Surroundings {
            voxels: None,
            heights: None,
            water: None,
        };
        let mut feet = Vec3::new(0.0, 10.0, 0.0);
        for _ in 0..60 {
            let r = p.tick(1.0 / 60.0, &around, |_| Some(Mat4::from_translation(feet)));
            if let Some((_, f)) = r.characters.into_iter().find(|(c, _)| *c == id) {
                feet = f;
            }
        }
        // Fallen 0.3 g * 1 s² / 2 = 1.47 m, not 4.9 m.
        let fallen = 10.0 - feet.y;
        assert!((fallen - 1.47).abs() < 0.2, "fell {fallen}");
    }
}
