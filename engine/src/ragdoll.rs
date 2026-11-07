//! Ragdolls: knock a character flying, let physics tumble it, and stand it
//! back up.
//!
//! ```ignore
//! // A punch from the left at chest height.
//! world.knock_down(hero, Knock::new(Vec3::new(400.0, 150.0, 0.0)).at(chest))?;
//! // Later: is it down, and which way up?
//! if let Some(RagdollState::Lying { face_up }) = world.ragdoll_state(hero) { /* ... */ }
//! ```
//!
//! Any skinned character with a humanlike skeleton works (Unreal, Mixamo,
//! MakeHuman, Blender/Rigify bone names, found by name and by shape). Its
//! ragdoll is 14 bodies (pelvis, belly, chest, head, upper and lower arms,
//! thighs, shins, feet) sized from the skeleton, joined with limited ball
//! joints and hinges (elbows, knees), with a faint muscle tone so a limp
//! body still folds naturally.
//!
//! The life of a ragdoll:
//! 1. **Knocked**: the bodies start from the pose the character is drawn in,
//!    so there is no pop, and take the hit (most of it moves the body, a
//!    share at the hit point spins it).
//! 2. **Tumbling**: physics drives the skeleton; the character's transform
//!    follows the pelvis.
//! 3. **Lying**: once it has been still for a moment the bodies are removed
//!    (no physics cost) and the final pose is kept.
//! 4. **Getting up** after [`RagdollSettings::get_up_after`] seconds, from
//!    the back or face down, through key poses, ending where the body lay
//!    and blending back into the animation. Or it stays down
//!    ([`Knock::stay_down`]) until [`World::get_up`] is called.
//!
//! At most [`RagdollSettings::max_active`] ragdolls tumble at once; knocking
//! down another settles the calmest one.
//!
//! [`World::get_up`]: crate::world::World::get_up

use crate::anim::Skeleton;
use crate::error::{EngineError, EngineResult};
use crate::physics::{JointDesc, PartShape};
use glam::{Mat3, Mat4, Quat, Vec3};

/// A hit that knocks a character down. Impulses are in newton-seconds (a
/// person weighs about 72 kg: an impulse of 360 throws them at 5 m/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knock {
    pub impulse: Vec3,
    /// Where the hit lands (world space). `None` pushes the upper body more
    /// than the feet, so the character topples.
    pub at: Option<Vec3>,
    /// Velocity the whole body already has (it was running, or riding).
    pub velocity: Vec3,
    /// Stay down after settling until [`crate::world::World::get_up`].
    pub stay_down: bool,
}

impl Knock {
    pub fn new(impulse: impl Into<Vec3>) -> Self {
        Self {
            impulse: impulse.into(),
            at: None,
            velocity: Vec3::ZERO,
            stay_down: false,
        }
    }

    pub fn at(mut self, point: impl Into<Vec3>) -> Self {
        self.at = Some(point.into());
        self
    }

    pub fn velocity(mut self, velocity: impl Into<Vec3>) -> Self {
        self.velocity = velocity.into();
        self
    }

    pub fn stay_down(mut self, stay_down: bool) -> Self {
        self.stay_down = stay_down;
        self
    }

    pub(crate) fn validate(&self) -> EngineResult<()> {
        let ok = self.impulse.is_finite()
            && self.velocity.is_finite()
            && self.at.is_none_or(|p| p.is_finite());
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidValue(
                "knock impulse, point and velocity must be finite".into(),
            ))
        }
    }
}

/// Where a character is in its ragdoll life.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RagdollState {
    /// Physics drives the body.
    Tumbling,
    /// Still, on the ground; no physics cost.
    Lying { face_up: bool },
    /// Standing up; `progress` runs from 0 to 1.
    GettingUp { face_up: bool, progress: f32 },
}

/// Something that happened to a ragdoll, from
/// [`crate::world::World::take_ragdoll_events`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RagdollEvent {
    /// Came to rest; `at` is the pelvis.
    Settled {
        id: crate::world::EntityId,
        face_up: bool,
        at: Vec3,
    },
    /// Standing again, back on its animation.
    GotUp { id: crate::world::EntityId },
}

/// Limits and timing for all ragdolls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RagdollSettings {
    /// Most ragdolls simulated at once (each is 14 bodies). Default 8.
    pub max_active: usize,
    /// Seconds lying still before getting up (unless it stays down). Default 1.5.
    pub get_up_after: f32,
    /// Longest a ragdoll tumbles before it is settled anyway. Default 9.
    pub max_tumble_seconds: f32,
    /// Speed of getting up (1 = normal, 2 = twice as fast). Default 1.
    pub get_up_speed: f32,
}

impl Default for RagdollSettings {
    fn default() -> Self {
        Self {
            max_active: 8,
            get_up_after: 1.5,
            max_tumble_seconds: 9.0,
            get_up_speed: 1.0,
        }
    }
}

impl RagdollSettings {
    pub(crate) fn validate(&self) -> EngineResult<()> {
        let ok = self.max_active > 0
            && self.get_up_after.is_finite()
            && self.get_up_after >= 0.0
            && self.max_tumble_seconds.is_finite()
            && self.max_tumble_seconds > 0.0
            && self.get_up_speed.is_finite()
            && self.get_up_speed > 0.0;
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidValue(
                "ragdoll settings: max_active > 0, times >= 0 and speed > 0".into(),
            ))
        }
    }
}

// ---------------------------------------------------------------- body plan

pub(crate) const PARTS: usize = 14;
pub(crate) const PELVIS: usize = 0;
pub(crate) const ABDOMEN: usize = 1;
pub(crate) const CHEST: usize = 2;
pub(crate) const HEAD: usize = 3;
pub(crate) const UARM_L: usize = 4;
pub(crate) const LARM_L: usize = 5;
pub(crate) const UARM_R: usize = 6;
pub(crate) const LARM_R: usize = 7;
pub(crate) const THIGH_L: usize = 8;
pub(crate) const SHIN_L: usize = 9;
pub(crate) const FOOT_L: usize = 10;
pub(crate) const THIGH_R: usize = 11;
pub(crate) const SHIN_R: usize = 12;
pub(crate) const FOOT_R: usize = 13;

/// Parent part of each part (the pelvis has none).
pub(crate) const PARENT: [usize; PARTS] = [0, 0, 1, 2, 2, 4, 2, 6, 0, 8, 9, 0, 11, 12];

/// Share of the body mass per part.
const MASS: [f32; PARTS] = [
    0.15, 0.11, 0.2, 0.08, 0.028, 0.022, 0.028, 0.022, 0.1, 0.047, 0.014, 0.1, 0.047, 0.014,
];
/// Body mass of a 1.76 m person, kg.
const BASE_MASS: f32 = 72.0;
/// Hip-to-ankle length of a 1.76 m person, m.
const BASE_LEG: f32 = 0.871;

/// How much of an impulse without a hit point each part takes: the upper
/// body more than the feet, so people topple instead of sliding.
pub(crate) const SOFT_WEIGHTS: [f32; PARTS] = [
    0.8, 0.9, 1.0, 1.0, 0.95, 0.95, 0.95, 0.95, 0.7, 0.55, 0.45, 0.7, 0.55, 0.45,
];

// Joint limits about the parent's axes (X right: + swings a hanging limb
// forward; Y up: twist; Z back: sideways), as [xmin, xmax, ymin, ymax,
// zmin, zmax], for the left side. The right side mirrors Y and Z.
const SPINE: [f32; 6] = [-0.6, 0.35, -0.4, 0.4, -0.35, 0.35];
const NECK: [f32; 6] = [-0.75, 0.6, -1.0, 1.0, -0.5, 0.5];
const SHOULDER_L: [f32; 6] = [-0.9, 2.3, -1.0, 1.0, -1.7, 0.75];
const HIP_L: [f32; 6] = [-0.55, 1.95, -0.5, 0.5, -0.85, 0.25];
const ANKLE: [f32; 6] = [-0.7, 0.6, -0.2, 0.2, -0.3, 0.3];

fn mirror(l: [f32; 6]) -> [f32; 6] {
    [l[0], l[1], -l[3], -l[2], -l[5], -l[4]]
}

/// Direction of a relaxed hanging left upper arm in the character frame
/// (the reference the shoulder limits are measured from).
const ARM_DOWN_L: Vec3 = Vec3::new(-0.662, -0.75, 0.0);

/// A character's ragdoll plan, worked out once per skeleton.
///
/// The *character frame* has X to the character's right, Y up and Z to its
/// back. Every part's body frame equals it in the *reference pose*
/// (standing, arms hanging, elbows and knees straight), so joint limits are
/// plain angles about the character's axes.
#[derive(Clone, Debug)]
pub(crate) struct Rig {
    /// The joint whose head is each part's origin and which the part turns.
    pub drivers: [usize; PARTS],
    /// Part driven by each joint.
    pub part_of: Vec<Option<usize>>,
    /// Joint at the far end of each part (for its shape).
    pub ends: [Option<usize>; PARTS],
    pub hip_joints: [usize; 2],
    /// Character frame to model space.
    pub frame: Quat,
    /// Each part's body orientation in the bind pose (character frame).
    pub bind_body: [Quat; PARTS],
    /// Joint rotation from body rotation: `driver = body · to_joint`
    /// (model space, bind pose).
    pub to_joint: [Quat; PARTS],
    /// Hinge axes in the parent's frame (forearms and shins).
    pub hinge: [Vec3; PARTS],
    /// Part origins in the bind pose, character frame (model units).
    pub bind_pos: [Vec3; PARTS],
    /// Hip height in the bind pose (model units).
    pub hip_height: f32,
}

impl Rig {
    pub fn new(skeleton: &Skeleton, bind_globals: &[Mat4]) -> EngineResult<Self> {
        let bad = |what: &str| {
            EngineError::InvalidValue(format!("ragdoll needs a humanlike skeleton ({what})"))
        };
        let n = skeleton.joint_names.len();
        let legs = skeleton.find_legs().ok_or_else(|| bad("no legs found"))?;
        let ancestors = |mut j: usize| {
            let mut out = vec![j];
            while let Some(p) = skeleton.parents[j] {
                out.push(p);
                j = p;
            }
            out
        };
        let common = |a: usize, b: usize| {
            let up = ancestors(a);
            ancestors(b).into_iter().find(|j| up.contains(j))
        };
        let pelvis = common(legs[0].thigh, legs[1].thigh).ok_or_else(|| bad("legs not joined"))?;
        let find = |names: &[&str]| names.iter().find_map(|n| skeleton.joint_index(n));
        let hand_l = find(&[
            "hand_l",
            "hand.l",
            "lefthand",
            "mixamorig:lefthand",
            "l_hand",
            "hand_left",
            "lhand",
            "wrist.l",
            "wrist_l",
            "leftwrist",
        ])
        .ok_or_else(|| bad("no left hand"))?;
        let hand_r = find(&[
            "hand_r",
            "hand.r",
            "righthand",
            "mixamorig:righthand",
            "r_hand",
            "hand_right",
            "rhand",
            "wrist.r",
            "wrist_r",
            "rightwrist",
        ])
        .ok_or_else(|| bad("no right hand"))?;
        let arm = |hand: usize| -> EngineResult<(usize, usize)> {
            let lower = skeleton.parents[hand].ok_or_else(|| bad("hand without arm"))?;
            let upper = skeleton.parents[lower].ok_or_else(|| bad("hand without arm"))?;
            Ok((upper, lower))
        };
        let (uarm_l, larm_l) = arm(hand_l)?;
        let (uarm_r, larm_r) = arm(hand_r)?;
        let top = common(uarm_l, uarm_r).ok_or_else(|| bad("arms not joined"))?;
        // The spine from the pelvis (excluded) up to where the arms meet.
        let mut spine: Vec<usize> = ancestors(top)
            .into_iter()
            .take_while(|j| *j != pelvis)
            .collect();
        if !ancestors(top).contains(&pelvis) {
            return Err(bad("arms not above the pelvis"));
        }
        spine.reverse();
        if spine.len() < 2 {
            return Err(bad("fewer than two spine joints"));
        }
        let abdomen = spine[0];
        let chest = spine[spine.len() / 2];
        let head = find(&["head", "mixamorig:head", "head_jnt", "bip01_head"])
            .filter(|h| ancestors(*h).contains(&top))
            .ok_or_else(|| bad("no head"))?;
        // The head part turns at the neck: the first joint above the top of
        // the spine on the way to the head.
        let neck = ancestors(head)
            .into_iter()
            .take_while(|j| *j != top)
            .last()
            .unwrap_or(head);
        let first_child = |j: usize| (0..n).find(|c| skeleton.parents[*c] == Some(j));

        let drivers = [
            pelvis,
            abdomen,
            chest,
            neck,
            uarm_l,
            larm_l,
            uarm_r,
            larm_r,
            legs[0].thigh,
            legs[0].calf,
            legs[0].foot,
            legs[1].thigh,
            legs[1].calf,
            legs[1].foot,
        ];
        let mut seen = drivers.to_vec();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != PARTS {
            return Err(bad("parts share joints"));
        }
        let ends = [
            None,
            Some(chest),
            Some(neck),
            Some(head).filter(|h| *h != neck),
            Some(larm_l),
            Some(hand_l),
            Some(larm_r),
            Some(hand_r),
            Some(legs[0].calf),
            Some(legs[0].foot),
            first_child(legs[0].foot),
            Some(legs[1].calf),
            Some(legs[1].foot),
            first_child(legs[1].foot),
        ];
        let mut part_of = vec![None; n];
        for (k, d) in drivers.iter().enumerate() {
            part_of[*d] = Some(k);
        }

        let at = |j: usize| bind_globals[j].w_axis.truncate();
        let rot = |j: usize| {
            bind_globals[j]
                .to_scale_rotation_translation()
                .1
                .normalize()
        };
        // The character frame from the shape of the skeleton.
        let right = (at(legs[1].thigh) - at(legs[0].thigh))
            .try_normalize()
            .ok_or_else(|| bad("hips on top of each other"))?;
        if at(head).y <= at(pelvis).y {
            return Err(bad("head below the pelvis in the bind pose"));
        }
        // Up is the model's up (glTF is Y-up), so poses keyed to the
        // character frame stand level even when the rig leans.
        let up0 = Vec3::Y;
        let up = (up0 - right * up0.dot(right))
            .try_normalize()
            .ok_or_else(|| bad("legs and spine in a line"))?;
        let back = right.cross(up);
        let frame = Quat::from_mat3(&Mat3::from_cols(right, up, back)).normalize();
        let to_char = frame.inverse();
        let char_at = |j: usize| to_char * at(j);
        let dir = |a: usize, b: usize| (char_at(b) - char_at(a)).try_normalize();

        let mut bind_body = [Quat::IDENTITY; PARTS];
        let mut hinge = [Vec3::X; PARTS];
        let arc = |from: Vec3, to: Option<Vec3>| {
            to.map_or(Quat::IDENTITY, |t| Quat::from_rotation_arc(from, t))
        };
        // Signed angle from `u` to `w` about `axis`.
        let angle = |u: Vec3, w: Vec3, axis: Vec3| u.cross(w).dot(axis).atan2(u.dot(w));
        for (thigh, shin, calf, foot) in [
            (THIGH_L, SHIN_L, legs[0].calf, legs[0].foot),
            (THIGH_R, SHIN_R, legs[1].calf, legs[1].foot),
        ] {
            let d_thigh = dir(drivers[thigh], calf);
            bind_body[thigh] = arc(Vec3::NEG_Y, d_thigh);
            let inv = bind_body[thigh].inverse();
            let axis = Vec3::X;
            let u = inv * d_thigh.unwrap_or(Vec3::NEG_Y);
            let w = inv * dir(calf, foot).unwrap_or(Vec3::NEG_Y);
            let w = (w - axis * w.dot(axis)).try_normalize().unwrap_or(u);
            bind_body[shin] = bind_body[thigh] * Quat::from_axis_angle(axis, angle(u, w, axis));
            hinge[shin] = axis;
        }
        for (side, upper, lower, elbow, hand) in [
            (-1.0f32, UARM_L, LARM_L, larm_l, hand_l),
            (1.0, UARM_R, LARM_R, larm_r, hand_r),
        ] {
            let down = Vec3::new(ARM_DOWN_L.x * -side, ARM_DOWN_L.y, 0.0).normalize();
            let d_upper = dir(drivers[upper], elbow);
            let d_lower = dir(elbow, hand);
            bind_body[upper] = arc(down, d_upper);
            let (u, w) = (d_upper.unwrap_or(down), d_lower.unwrap_or(down));
            // Elbows bend about the axis across the arm's plane; a straight
            // arm bends forward.
            let c = u.cross(w);
            let axis_char = if c.length() > 0.05 {
                c.normalize()
            } else {
                u.cross(Vec3::NEG_Z)
                    .try_normalize()
                    .unwrap_or(Vec3::X * -side)
            };
            let inv = bind_body[upper].inverse();
            let axis = (inv * axis_char).normalize();
            let (ul, wl) = (inv * u, inv * w);
            bind_body[lower] = bind_body[upper] * Quat::from_axis_angle(axis, angle(ul, wl, axis));
            hinge[lower] = axis;
        }
        let mut to_joint = [Quat::IDENTITY; PARTS];
        let mut bind_pos = [Vec3::ZERO; PARTS];
        for k in 0..PARTS {
            to_joint[k] = ((frame * bind_body[k]).inverse() * rot(drivers[k])).normalize();
            bind_pos[k] = char_at(drivers[k]);
        }
        let hip_height = (char_at(legs[0].thigh).y + char_at(legs[1].thigh).y) * 0.5;
        Ok(Self {
            drivers,
            part_of,
            ends,
            hip_joints: [legs[0].thigh, legs[1].thigh],
            frame,
            bind_body,
            to_joint,
            hinge,
            bind_pos,
            hip_height: hip_height.max(1e-3),
        })
    }

    /// The joint between part `k` and its parent, with its limits and the
    /// relaxed angles a faint muscle tone pulls toward.
    pub fn joint(&self, k: usize) -> JointDesc {
        let ball = |limits: [f32; 6], relax: [f32; 3]| JointDesc::Ball { limits, relax };
        match k {
            ABDOMEN | CHEST => ball(SPINE, [-0.08, 0.0, 0.0]),
            HEAD => ball(NECK, [-0.15, 0.0, 0.0]),
            UARM_L => ball(SHOULDER_L, [0.15, 0.0, 0.35]),
            UARM_R => ball(mirror(SHOULDER_L), [0.15, 0.0, -0.35]),
            THIGH_L => ball(HIP_L, [0.25, 0.0, 0.0]),
            THIGH_R => ball(mirror(HIP_L), [0.25, 0.0, 0.0]),
            FOOT_L => ball(ANKLE, [0.2, 0.0, 0.0]),
            FOOT_R => ball(mirror(ANKLE), [0.2, 0.0, 0.0]),
            LARM_L | LARM_R => JointDesc::Hinge {
                axis: self.hinge[k],
                limits: [-0.05, 2.5],
                relax: 0.9,
            },
            _ => JointDesc::Hinge {
                axis: self.hinge[k],
                limits: [-2.45, 0.03],
                relax: -0.6,
            },
        }
    }
}

/// Mass and size of one character's ragdoll, from its legs in the world.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Size {
    /// Size factor against a 1.76 m person.
    pub f: f32,
    pub mass: f32,
}

impl Size {
    pub fn from_leg(leg_m: f32) -> Self {
        let f = (leg_m / BASE_LEG).clamp(0.02, 200.0);
        Self {
            f,
            mass: BASE_MASS * f * f * f,
        }
    }

    /// Size against a 1.8 m person (for time scales).
    pub fn k(&self) -> f32 {
        self.f * 1.76 / 1.8
    }

    pub fn part_mass(&self, k: usize) -> f32 {
        self.mass * MASS[k] / MASS.iter().sum::<f32>()
    }
}

/// A capsule from `from` to `to` (body frame), trimmed at both ends by a
/// share of the radius so neighbours overlap less.
fn capsule(from: Vec3, to: Vec3, r: f32, trim_a: f32, trim_b: f32) -> PartShape {
    let d = to - from;
    let l = d.length().max(1e-4);
    let ta = (r * trim_a / l).min(0.3);
    let tb = (r * trim_b / l).min(0.3);
    PartShape::Capsule {
        a: from + d * ta,
        b: from + d * (1.0 - tb),
        radius: r,
    }
}

/// Each part's collider, in its body frame. `local(j)` is joint `j`'s head
/// in part `k`'s body frame.
pub(crate) fn part_shape(rig: &Rig, k: usize, f: f32, local: impl Fn(usize) -> Vec3) -> PartShape {
    let end = rig.ends[k].map(&local);
    match k {
        PELVIS => {
            let (l, r) = (local(rig.hip_joints[0]), local(rig.hip_joints[1]));
            let across = (r - l).try_normalize().unwrap_or(Vec3::X);
            let half = ((r - l).length() * 0.5 - 0.035 * f).max(0.02 * f);
            let mid = (l + r) * 0.5 - Vec3::Y * 0.01 * f;
            PartShape::Capsule {
                a: mid - across * half,
                b: mid + across * half,
                radius: 0.11 * f,
            }
        }
        ABDOMEN | CHEST => {
            let top = end.unwrap_or(Vec3::Y * 0.15 * f);
            let len = top.length();
            let (w, d, round, extra) = if k == ABDOMEN {
                (0.125, 0.085, 0.03, 0.015)
            } else {
                (0.135, 0.095, 0.04, -0.01)
            };
            PartShape::Box {
                center: top * 0.5 + Vec3::Z * -0.005 * f,
                half: Vec3::new(w * f, (len * 0.5 + extra * f).max(0.03 * f), d * f),
                round: round * f,
            }
        }
        HEAD => {
            let h = end.unwrap_or(Vec3::Y * 0.08 * f);
            PartShape::Ball {
                center: h + Vec3::new(0.0, 0.07 * f, 0.015 * f),
                radius: 0.1 * f,
            }
        }
        UARM_L | UARM_R => capsule(
            Vec3::ZERO,
            end.unwrap_or(Vec3::NEG_Y * 0.25 * f),
            0.047 * f,
            0.8,
            0.4,
        ),
        LARM_L | LARM_R => {
            let hand = end.unwrap_or(Vec3::NEG_Y * 0.25 * f);
            let tip = hand + hand.try_normalize().unwrap_or(Vec3::NEG_Y) * 0.08 * f;
            capsule(Vec3::ZERO, tip, 0.04 * f, 0.4, 0.0)
        }
        THIGH_L | THIGH_R => capsule(
            Vec3::ZERO,
            end.unwrap_or(Vec3::NEG_Y * 0.43 * f),
            0.07 * f,
            1.2,
            0.3,
        ),
        SHIN_L | SHIN_R => capsule(
            Vec3::ZERO,
            end.unwrap_or(Vec3::NEG_Y * 0.44 * f),
            0.05 * f,
            0.3,
            0.6,
        ),
        _ => {
            // Foot: a box from behind the ankle to past the toes, sole below.
            let ankle_h = 0.07 * f;
            let toe = end.unwrap_or(Vec3::new(0.0, -ankle_h + 0.02 * f, -0.13 * f));
            let heel = 0.06 * f;
            let tip = toe.z - 0.07 * f;
            let top = (ankle_h + 0.01 * f).max(0.05 * f);
            PartShape::Box {
                center: Vec3::new(toe.x * 0.5, -ankle_h + top * 0.5, (heel + tip) * 0.5),
                half: Vec3::new(0.045 * f, top * 0.5, ((heel - tip) * 0.5).max(0.03 * f)),
                round: 0.012 * f,
            }
        }
    }
}

/// Joint locals that put each part's driver joint at `parts` (model-space
/// position and body rotation); the joints in between keep `frozen`.
pub(crate) fn locals_from_parts(
    skeleton: &Skeleton,
    rig: &Rig,
    frozen: &[(Vec3, Quat, Vec3)],
    scales: &[Vec3],
    parts: &[(Vec3, Quat)],
) -> Vec<(Vec3, Quat, Vec3)> {
    let n = skeleton.joint_names.len();
    let mut global = vec![Mat4::IDENTITY; n];
    let mut out = frozen.to_vec();
    for j in 0..n {
        let parent = match skeleton.parents[j] {
            Some(p) => global[p],
            None => skeleton
                .root_transforms
                .get(j)
                .copied()
                .unwrap_or(Mat4::IDENTITY),
        };
        if let Some(k) = rig.part_of[j] {
            let (p, r) = parts[k];
            let want = Mat4::from_scale_rotation_translation(
                scales.get(j).copied().unwrap_or(Vec3::ONE),
                (r * rig.to_joint[k]).normalize(),
                p,
            );
            let local = parent.inverse() * want;
            let (s, r, t) = local.to_scale_rotation_translation();
            out[j] = (t, r.normalize(), s);
            global[j] = want;
        } else {
            let (t, r, s) = frozen[j];
            global[j] = parent * Mat4::from_scale_rotation_translation(s, r, t);
        }
    }
    out
}

// ---------------------------------------------------------------- getting up

/// A get-up key pose: Euler angles per part relative to the reference pose
/// (limbs X·Z·Y, spine, neck and pelvis Y·X·Z; hinges use X), and the
/// pelvis offset from where it stands (character frame, × hip height / 0.94).
#[derive(Clone, Copy, Debug, Default)]
struct Key {
    rot: [Vec3; PARTS],
    root: Vec3,
}

impl Key {
    fn root(mut self, x: f32, y: f32, z: f32) -> Self {
        self.rot[PELVIS].x += x;
        self.root = Vec3::new(0.0, y, z);
        self
    }

    fn spine(mut self, x: f32) -> Self {
        self.rot[ABDOMEN].x += x * 0.45;
        self.rot[CHEST].x += x * 0.55;
        self
    }

    fn neck(mut self, x: f32) -> Self {
        self.rot[HEAD].x += x;
        self
    }

    /// Arm from hanging: forward swing, out to the side, elbow bend.
    fn arm(mut self, left: bool, fwd: f32, out: f32, elbow: f32) -> Self {
        let (u, l, m) = if left {
            (UARM_L, LARM_L, 1.0)
        } else {
            (UARM_R, LARM_R, -1.0)
        };
        self.rot[u] += Vec3::new(fwd, 0.0, -out * m);
        self.rot[l].x += elbow + REST_ELBOW;
        self
    }

    /// Leg: hip flex, out to the side, knee bend, ankle.
    fn leg(mut self, left: bool, flex: f32, abd: f32, knee: f32, ankle: f32) -> Self {
        let (t, s, f, m) = if left {
            (THIGH_L, SHIN_L, FOOT_L, 1.0)
        } else {
            (THIGH_R, SHIN_R, FOOT_R, -1.0)
        };
        self.rot[t] += Vec3::new(flex, 0.0, -abd * m);
        self.rot[s].x -= knee;
        self.rot[f].x -= ankle;
        self
    }
}

/// Relaxed elbow bend the key poses' elbow angles add to.
const REST_ELBOW: f32 = 0.6;
const HIP: f32 = 0.94;
const PI2: f32 = std::f32::consts::FRAC_PI_2;
/// How far forward (character −Z, × hip height / 0.94) the body ends up.
pub(crate) const END_Z_UP: f32 = -0.42;
pub(crate) const END_Z_DOWN: f32 = -0.38;

fn up_keys() -> [(f32, Key); 6] {
    let k = Key::default;
    [
        // Lying on the back.
        (
            0.0,
            k().root(PI2, -(HIP - 0.1), 0.0)
                .arm(true, 0.1, 0.55, 0.35)
                .arm(false, 0.1, 0.55, 0.35)
                .leg(true, 0.0, 0.08, 0.1, 0.3)
                .leg(false, 0.0, 0.06, 0.05, 0.3),
        ),
        // Propped on the elbows, chin to the chest.
        (
            0.2,
            k().root(1.0, -(HIP - 0.11), 0.0)
                .spine(-0.35)
                .neck(-0.5)
                .arm(true, -0.95, 0.35, 1.45)
                .arm(false, -0.95, 0.35, 1.45)
                .leg(true, 0.6, 0.08, 0.5, 0.2)
                .leg(false, 0.55, 0.06, 0.1, 0.2),
        ),
        // Sitting, knees up, a hand behind on the ground.
        (
            0.42,
            k().root(0.35, -(HIP - 0.13), 0.0)
                .spine(-0.5)
                .neck(-0.25)
                .arm(false, -0.75, 0.4, 0.15)
                .arm(true, 0.95, 0.2, 0.7)
                .leg(true, 1.95, 0.1, 2.05, -0.1)
                .leg(false, 1.85, 0.12, 1.75, -0.1),
        ),
        // Deep squat, weight over the feet, arms out front.
        (
            0.66,
            k().root(-0.45, -(HIP - 0.47), -0.13)
                .spine(-0.45)
                .neck(0.45)
                .arm(true, 1.0, 0.25, 0.5)
                .arm(false, 0.9, 0.25, 0.6)
                .leg(true, 2.0, 0.12, 1.87, -0.3)
                .leg(false, 2.0, 0.12, 1.87, -0.3),
        ),
        // Rising.
        (
            0.84,
            k().root(-0.25, -0.19, -0.21)
                .spine(-0.25)
                .neck(0.25)
                .arm(true, 0.45, 0.2, 0.4)
                .arm(false, 0.4, 0.2, 0.45)
                .leg(true, 1.15, 0.06, 1.15, -0.25)
                .leg(false, 1.15, 0.06, 1.15, -0.25),
        ),
        (1.0, k().root(0.0, 0.0, END_Z_UP)),
    ]
}

fn down_keys() -> [(f32, Key); 6] {
    let k = Key::default;
    [
        // Face down, hands under the shoulders.
        (
            0.0,
            k().root(-PI2, -(HIP - 0.1), 0.0)
                .neck(0.2)
                .arm(true, 0.6, 0.75, 1.6)
                .arm(false, 0.6, 0.75, 1.6)
                .leg(true, 0.0, 0.08, 0.1, -0.6)
                .leg(false, 0.0, 0.06, 0.05, -0.6),
        ),
        // Push-up, chest off the ground.
        (
            0.24,
            k().root(-1.15, -(HIP - 0.15), 0.0)
                .spine(0.35)
                .neck(0.3)
                .arm(true, 1.2, 0.35, 0.35)
                .arm(false, 1.2, 0.35, 0.35)
                .leg(true, -0.2, 0.08, 0.35, -0.5)
                .leg(false, -0.25, 0.06, 0.3, -0.5),
        ),
        // All fours.
        (
            0.46,
            k().root(-1.42, -(HIP - 0.49), 0.0)
                .spine(0.1)
                .neck(0.6)
                .arm(true, 1.42, 0.12, 0.1)
                .arm(false, 1.42, 0.12, 0.1)
                .leg(true, 1.42, 0.1, 1.57, 0.0)
                .leg(false, 1.42, 0.1, 1.57, 0.0),
        ),
        // Kneeling on the right knee, left foot planted.
        (
            0.68,
            k().root(-0.3, -(HIP - 0.49), 0.0)
                .spine(-0.1)
                .neck(0.1)
                .arm(true, 0.8, 0.2, 0.6)
                .arm(false, 0.2, 0.2, 0.3)
                .leg(true, 1.87, 0.08, 1.57, 0.0)
                .leg(false, 0.3, 0.06, 1.57, -0.57),
        ),
        // Rising over the left foot.
        (
            0.85,
            k().root(-0.2, -0.1, -0.27)
                .spine(-0.05)
                .arm(true, 0.45, 0.2, 0.5)
                .arm(false, 0.25, 0.2, 0.3)
                .leg(true, 0.8, 0.06, 0.8, -0.2)
                .leg(false, 0.3, 0.06, 1.0, -0.3),
        ),
        (1.0, k().root(0.0, 0.0, END_Z_DOWN)),
    ]
}

pub(crate) fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn euler_limb(e: Vec3) -> Quat {
    Quat::from_rotation_x(e.x) * Quat::from_rotation_z(e.z) * Quat::from_rotation_y(e.y)
}

fn euler_spine(e: Vec3) -> Quat {
    Quat::from_rotation_y(e.y) * Quat::from_rotation_x(e.x) * Quat::from_rotation_z(e.z)
}

/// The get-up pose at progress `u` (0 lying … 1 standing) as part poses in
/// model space (model units). `shift` moves the pelvis along character Z
/// (the forward offset handed over to the transform at the end).
pub(crate) fn get_up_parts(rig: &Rig, face_up: bool, u: f32, shift: f32) -> Vec<(Vec3, Quat)> {
    let keys = if face_up { up_keys() } else { down_keys() };
    let mut i = 1;
    while i < keys.len() - 1 && u > keys[i].0 {
        i += 1;
    }
    let (t0, a) = keys[i - 1];
    let (t1, b) = keys[i];
    let w = smooth(t0, t1, u);
    let q = rig.hip_height / HIP;
    let mut rot = [Quat::IDENTITY; PARTS];
    let mut pos = [Vec3::ZERO; PARTS];
    // Where each part is standing in the reference pose.
    let mut rest = [rig.bind_pos[PELVIS]; PARTS];
    let root = a.root.lerp(b.root, w) * q + Vec3::Z * shift;
    for k in 0..PARTS {
        let e = a.rot[k].lerp(b.rot[k], w);
        let local = match k {
            PELVIS | ABDOMEN | CHEST | HEAD => euler_spine(e),
            LARM_L | LARM_R | SHIN_L | SHIN_R => Quat::from_axis_angle(rig.hinge[k], e.x),
            _ => euler_limb(e),
        };
        if k == PELVIS {
            rot[k] = local;
            pos[k] = rig.bind_pos[PELVIS] + root;
        } else {
            let p = PARENT[k];
            rot[k] = (rot[p] * local).normalize();
            let off = rig.bind_body[p].inverse() * (rig.bind_pos[k] - rig.bind_pos[p]);
            pos[k] = pos[p] + rot[p] * off;
            rest[k] = rest[p] + off;
        }
    }
    // Between keys a limb can dip into the ground: lift the body instead.
    let clearance = |k: usize| {
        let nominal = match k {
            FOOT_L | FOOT_R => 0.06,
            _ => 0.03,
        } * q;
        nominal.min(rest[k].y) - 0.005 * q
    };
    let dip = (0..PARTS)
        .map(|k| pos[k].y - clearance(k))
        .fold(0.0f32, f32::min);
    (0..PARTS)
        .map(|k| {
            let p = pos[k] - Vec3::Y * dip;
            (rig.frame * p, (rig.frame * rot[k]).normalize())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirrored_limits_swap_sides() {
        let m = mirror(SHOULDER_L);
        assert_eq!(m[0], SHOULDER_L[0]);
        assert_eq!(m[2], -SHOULDER_L[3]);
        assert_eq!(m[5], -SHOULDER_L[4]);
    }

    #[test]
    fn heavier_people_are_bigger() {
        let a = Size::from_leg(BASE_LEG);
        assert!((a.mass - BASE_MASS).abs() < 1e-3);
        let b = Size::from_leg(BASE_LEG * 2.0);
        assert!((b.mass / a.mass - 8.0).abs() < 1e-3);
        let total: f32 = (0..PARTS).map(|k| a.part_mass(k)).sum();
        assert!((total - a.mass).abs() < 1e-2);
    }

    #[test]
    fn smoothstep_ends() {
        assert_eq!(smooth(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smooth(0.0, 1.0, 2.0), 1.0);
        assert!((smooth(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
    }
}
