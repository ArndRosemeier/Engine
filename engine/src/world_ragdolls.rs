//! Ragdolls on animated characters: the World side of [`crate::ragdoll`].

use super::{AnimatedEntity, EntityId, World};
use crate::anim::PoseOverride;
use crate::error::{EngineError, EngineResult};
use crate::physics::{LinkDesc, PartDesc};
use crate::ragdoll::{
    get_up_parts, locals_from_parts, part_shape, smooth, Knock, RagdollEvent, RagdollSettings,
    RagdollState, Rig, Size, CHEST, END_Z_DOWN, END_Z_UP, PARENT, PARTS, PELVIS, SOFT_WEIGHTS,
};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Seconds the ragdoll's pose takes to blend back into the animation.
const HAND_BACK: f32 = 0.35;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Tumbling,
    Lying,
    GettingUp,
}

#[derive(Clone, Copy, Debug)]
struct GetUp {
    face_up: bool,
    t: f32,
    hold: f32,
    duration: f32,
    blend: f32,
    /// Forward offset of the standing pose (model units), handed to the
    /// transform at the end.
    end_z: f32,
    handed: bool,
}

#[derive(Debug)]
pub(crate) struct Entry {
    rig: Arc<Rig>,
    size: Size,
    phase: Phase,
    t: f32,
    calm: f32,
    stay_down: bool,
    /// Every part's world pose (the last one drawn).
    parts: Vec<(Vec3, Quat)>,
    /// The pose at the knock: joints inside parts keep these locals.
    frozen: Vec<(Vec3, Quat, Vec3)>,
    scales: Vec<Vec3>,
    /// The pelvis in model space at the knock (the transform follows it).
    pelvis_model: Vec3,
    up: Option<GetUp>,
    serial: u64,
}

/// All ragdolls of a world.
#[derive(Debug, Default)]
pub(crate) struct Ragdolls {
    entries: HashMap<EntityId, Entry>,
    /// Plans by model (the model is kept so its address stays unique).
    rigs: HashMap<usize, (Arc<crate::anim::AnimatedModel>, Arc<Rig>)>,
    settings: RagdollSettings,
    events: Vec<RagdollEvent>,
    serial: u64,
}

fn split(m: Mat4) -> (Vec3, Quat, Vec3) {
    let (s, r, t) = m.to_scale_rotation_translation();
    (s, r.normalize(), t)
}

impl Entry {
    fn face_up(&self) -> bool {
        let (_, chest) = self.parts[CHEST];
        (chest * Vec3::NEG_Z).y > 0.0
    }

    fn state(&self) -> RagdollState {
        match self.phase {
            Phase::Tumbling => RagdollState::Tumbling,
            Phase::Lying => RagdollState::Lying {
                face_up: self.face_up(),
            },
            Phase::GettingUp => {
                let up = self.up.expect("getting up");
                RagdollState::GettingUp {
                    face_up: up.face_up,
                    progress: ((up.t - up.hold) / up.duration).clamp(0.0, 1.0),
                }
            }
        }
    }

    /// Part poses in model space, from world poses and the transform.
    fn model_parts(&self, transform: Mat4) -> Vec<(Vec3, Quat)> {
        let to_model = transform.inverse();
        let (_, rot, _) = split(transform);
        let inv = rot.inverse();
        self.parts
            .iter()
            .map(|(p, r)| (to_model.transform_point3(*p), (inv * *r).normalize()))
            .collect()
    }

    /// The transform that keeps the pelvis where the body's is.
    fn follow(&self, transform: Mat4) -> Mat4 {
        let (s, r, _) = split(transform);
        let pelvis = self.parts[PELVIS].0;
        let t = pelvis - r * (s * self.pelvis_model);
        Mat4::from_scale_rotation_translation(s, r, t)
    }
}

impl World {
    fn rig_of(&mut self, id: EntityId) -> EngineResult<Arc<Rig>> {
        let e = self.animated.get(&id).ok_or(EngineError::UnknownEntity)?;
        let model = e.animator.model().clone();
        let key = Arc::as_ptr(&model) as usize;
        if let Some((_, r)) = self.ragdolls.rigs.get(&key) {
            return Ok(r.clone());
        }
        let skeleton = &model.skeleton;
        let mut bind = Vec::new();
        crate::anim::globals_from_locals(skeleton, &model.bind_local, &mut bind);
        let rig = Arc::new(Rig::new(skeleton, &bind)?);
        self.ragdolls.rigs.insert(key, (model.clone(), rig.clone()));
        Ok(rig)
    }

    /// Knock a skinned character down: it goes limp from the pose it is drawn
    /// in and takes the hit. Knocking one that is already down throws it again.
    pub fn knock_down(&mut self, id: EntityId, knock: Knock) -> EngineResult<()> {
        knock.validate()?;
        let rig = self.rig_of(id)?;
        let tumbling = self
            .ragdolls
            .entries
            .get(&id)
            .is_some_and(|e| e.phase == Phase::Tumbling && self.physics.has_ragdoll(id));
        if !tumbling {
            self.make_room_for_ragdoll(id);
            self.start_ragdoll(id, rig, knock.velocity)?;
        } else if knock.velocity != Vec3::ZERO {
            self.physics
                .ragdoll_add_velocity(id, knock.velocity, &[1.0; PARTS]);
        }
        let e = self.ragdolls.entries.get_mut(&id).expect("started");
        e.stay_down = knock.stay_down;
        e.calm = 0.0;
        e.t = 0.0;
        let mass = e.size.mass;
        let j = knock.impulse;
        if j != Vec3::ZERO {
            // Most of it moves the body; a share at the hit point spins it.
            self.physics
                .ragdoll_add_velocity(id, j * 0.6 / mass, &[1.0; PARTS]);
            match knock.at {
                Some(at) => {
                    let (part, at) = self.nearest_part(id, at);
                    self.physics.ragdoll_impulse(id, part, j * 0.4, at);
                }
                None => self
                    .physics
                    .ragdoll_add_velocity(id, j * 0.4 / mass, &SOFT_WEIGHTS),
            }
        }
        Ok(())
    }

    /// The part closest to `at`, and the point to push it at (its centre when
    /// `at` is far from the body).
    fn nearest_part(&self, id: EntityId, at: Vec3) -> (usize, Vec3) {
        let Some(e) = self.ragdolls.entries.get(&id) else {
            return (PELVIS, at);
        };
        let f = e.size.f;
        let centre = |k: usize| {
            let (p, r) = e.parts[k];
            let shape = part_shape(&e.rig, k, f, |_| Vec3::ZERO);
            p + r * shape.center()
        };
        let (best, d) = (0..PARTS)
            .map(|k| (k, centre(k).distance(at)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("parts");
        let reach = 0.45 * f;
        (best, if d > reach { centre(best) } else { at })
    }

    /// Settle the calmest tumbling ragdoll when the budget is full.
    fn make_room_for_ragdoll(&mut self, coming: EntityId) {
        let max = self.ragdolls.settings.max_active;
        loop {
            let mut live: Vec<(EntityId, f32, u64)> = self
                .ragdolls
                .entries
                .iter()
                .filter(|(id, e)| **id != coming && e.phase == Phase::Tumbling)
                .map(|(id, e)| {
                    let v = self.physics.ragdoll_motion(*id).map_or(0.0, |m| m.0);
                    (*id, v, e.serial)
                })
                .collect();
            if live.len() < max {
                return;
            }
            live.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)));
            self.settle_ragdoll(live[0].0);
        }
    }

    fn start_ragdoll(&mut self, id: EntityId, rig: Arc<Rig>, velocity: Vec3) -> EngineResult<()> {
        let e = self.animated.get(&id).ok_or(EngineError::UnknownEntity)?;
        let (frozen, globals) = e.animator.pose_now();
        let transform = e.transform;
        let skeleton = &e.animator.model().skeleton;
        if frozen.len() != skeleton.joint_names.len() {
            return Err(EngineError::InvalidValue("character has no pose".into()));
        }
        let (_, e_rot, _) = split(transform);
        let world_at = |j: usize| transform.transform_point3(globals[j].w_axis.truncate());
        let scales: Vec<Vec3> = globals
            .iter()
            .map(|g| g.to_scale_rotation_translation().0)
            .collect();
        let mut parts = Vec::with_capacity(PARTS);
        for k in 0..PARTS {
            let j = rig.drivers[k];
            let g_rot = split(globals[j]).1;
            let r = (e_rot * g_rot * rig.to_joint[k].inverse()).normalize();
            parts.push((world_at(j), r));
        }
        let leg = |thigh: usize, shin: usize, foot: usize| {
            parts[thigh].0.distance(parts[shin].0) + parts[shin].0.distance(parts[foot].0)
        };
        let size = Size::from_leg(
            (leg(
                crate::ragdoll::THIGH_L,
                crate::ragdoll::SHIN_L,
                crate::ragdoll::FOOT_L,
            ) + leg(
                crate::ragdoll::THIGH_R,
                crate::ragdoll::SHIN_R,
                crate::ragdoll::FOOT_R,
            )) * 0.5,
        );
        let (f, k) = (size.f, size.k());
        let sk = k.sqrt();
        let descs: Vec<PartDesc> = (0..PARTS)
            .map(|p| {
                let (pos, r) = parts[p];
                let inv = r.inverse();
                let shape = part_shape(&rig, p, f, |j| inv * (world_at(j) - pos));
                PartDesc {
                    position: pos,
                    rotation: r,
                    velocity,
                    shape,
                    mass: size.part_mass(p),
                    arm: (crate::ragdoll::UARM_L..=crate::ragdoll::LARM_R).contains(&p),
                    linear_damping: 0.05 / sk,
                    angular_damping: 1.1 / sk,
                }
            })
            .collect();
        let links: Vec<LinkDesc> = (1..PARTS)
            .map(|c| {
                let p = PARENT[c];
                LinkDesc {
                    parent: p,
                    child: c,
                    anchor: parts[p].1.inverse() * (parts[c].0 - parts[p].0),
                    joint: rig.joint(c),
                    tone: 3.0 / k,
                    damping: 1.4 / sk,
                }
            })
            .collect();
        self.physics.add_ragdoll(id, &descs, &links);
        let pelvis_model = globals[rig.drivers[PELVIS]].w_axis.truncate();
        self.ragdolls.serial += 1;
        let serial = self.ragdolls.serial;
        let stay_down = self.ragdolls.entries.get(&id).is_some_and(|e| e.stay_down);
        self.ragdolls.entries.insert(
            id,
            Entry {
                rig,
                size,
                phase: Phase::Tumbling,
                t: 0.0,
                calm: 0.0,
                stay_down,
                parts,
                frozen,
                scales,
                pelvis_model,
                up: None,
                serial,
            },
        );
        self.write_ragdoll_pose(id);
        Ok(())
    }

    /// Stop simulating: keep the pose, drop the bodies.
    fn settle_ragdoll(&mut self, id: EntityId) {
        self.physics.remove_ragdoll(id);
        let Some(e) = self.ragdolls.entries.get_mut(&id) else {
            return;
        };
        if e.phase != Phase::Tumbling {
            return;
        }
        e.phase = Phase::Lying;
        e.t = 0.0;
        let event = RagdollEvent::Settled {
            id,
            face_up: e.face_up(),
            at: e.parts[PELVIS].0,
        };
        self.ragdolls.events.push(event);
    }

    /// Stand a lying character up now (one knocked down to stay down, say).
    /// Does nothing to a character that is not lying.
    pub fn get_up(&mut self, id: EntityId) -> EngineResult<()> {
        if !self.animated.contains_key(&id) {
            return Err(EngineError::UnknownEntity);
        }
        if self
            .ragdolls
            .entries
            .get(&id)
            .is_some_and(|e| e.phase == Phase::Lying)
        {
            self.start_get_up(id);
        }
        Ok(())
    }

    /// Keep a downed character down (or let it get up by itself again).
    pub fn set_stay_down(&mut self, id: EntityId, stay_down: bool) -> EngineResult<()> {
        let e = self
            .ragdolls
            .entries
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.stay_down = stay_down;
        if !stay_down && e.phase == Phase::Lying {
            e.t = e.t.min(self.ragdolls.settings.get_up_after);
        }
        Ok(())
    }

    /// Where a character is in its ragdoll life; `None` when it is not one.
    pub fn ragdoll_state(&self, id: EntityId) -> Option<RagdollState> {
        self.ragdolls.entries.get(&id).map(Entry::state)
    }

    /// Ragdolls tumbling (simulated) and in total (tumbling, lying, getting up).
    pub fn ragdoll_count(&self) -> (usize, usize) {
        let live = self
            .ragdolls
            .entries
            .values()
            .filter(|e| e.phase == Phase::Tumbling)
            .count();
        (live, self.ragdolls.entries.len())
    }

    pub fn ragdoll_settings(&self) -> RagdollSettings {
        self.ragdolls.settings
    }

    pub fn set_ragdoll_settings(&mut self, settings: RagdollSettings) -> EngineResult<()> {
        settings.validate()?;
        self.ragdolls.settings = settings;
        Ok(())
    }

    /// Ragdolls that settled or got up since the last call.
    pub fn take_ragdoll_events(&mut self) -> Vec<RagdollEvent> {
        std::mem::take(&mut self.ragdolls.events)
    }

    /// Knock down every animated character within `radius` of `center`,
    /// thrown outward and up at up to `speed` m/s (less further out).
    /// Returns how many went down.
    pub fn knock_down_in(&mut self, center: Vec3, radius: f32, speed: f32) -> usize {
        if !center.is_finite() || !radius.is_finite() || radius <= 0.0 || !speed.is_finite() {
            return 0;
        }
        let mut hit: Vec<(EntityId, Vec3)> = self
            .animated
            .iter()
            .filter_map(|(id, e)| {
                let p = e.transform.w_axis.truncate() + Vec3::Y * 0.9;
                (p.distance(center) < radius).then_some((*id, p))
            })
            .collect();
        hit.sort_by_key(|(id, _)| id.raw());
        let mut n = 0;
        for (id, p) in hit {
            let to = p - center;
            let fall = 1.0 - to.length() / radius;
            let dir = (to.normalize_or_zero() + Vec3::Y * 0.5).normalize_or(Vec3::Y);
            if self
                .knock_down(id, Knock::new(Vec3::ZERO).velocity(dir * speed * fall))
                .is_ok()
            {
                n += 1;
            }
        }
        n
    }

    /// The animated character a ray hits first (its bones as capsules), and
    /// where. For clicking on people.
    pub fn pick_character(
        &self,
        origin: impl Into<Vec3>,
        direction: impl Into<Vec3>,
        max_distance: f32,
    ) -> Option<(EntityId, Vec3)> {
        let origin = origin.into();
        let dir = direction.into().try_normalize()?;
        let mut best: Option<(f32, EntityId)> = None;
        let mut ids: Vec<&EntityId> = self.animated.keys().collect();
        ids.sort_by_key(|id| id.raw());
        for id in ids {
            let e = &self.animated[id];
            let (s, _, at) = split(e.transform);
            // Skip what is far from the ray before posing the skeleton.
            let reach = 2.5 * s.max_element().max(1e-3);
            let centre = self
                .ragdolls
                .entries
                .get(id)
                .map_or(at, |r| r.parts[PELVIS].0);
            let along = (centre - origin).dot(dir).clamp(0.0, max_distance);
            if (origin + dir * along).distance(centre) > reach {
                continue;
            }
            if let Some(t) = ray_skeleton(e, origin, dir, max_distance) {
                if best.is_none_or(|b| t < b.0) {
                    best = Some((t, *id));
                }
            }
        }
        best.map(|(t, id)| (id, origin + dir * t))
    }

    fn start_get_up(&mut self, id: EntityId) {
        let Some(transform) = self.animated.get(&id).map(|e| e.transform) else {
            return;
        };
        let speed = self.ragdolls.settings.get_up_speed;
        let Some(e) = self.ragdolls.entries.get(&id) else {
            return;
        };
        let face_up = e.face_up();
        let rig = e.rig.clone();
        let pelvis = e.parts[PELVIS].0;
        let mut along = e.parts[CHEST].0 - pelvis;
        along.y = 0.0;
        let along = along.try_normalize().unwrap_or_else(|| {
            (e.parts[PELVIS].1 * Vec3::Y)
                .with_y(0.0)
                .normalize_or(Vec3::Z)
        });
        // Face up: stand facing the feet; face down: facing where the head was.
        let forward = if face_up { -along } else { along };
        let (s, _, _) = split(transform);
        // The character's own forward in model space, turned to face `forward`.
        let model_fwd = (rig.frame * Vec3::NEG_Z)
            .with_y(0.0)
            .normalize_or(Vec3::NEG_Z);
        let yaw = forward.x.atan2(forward.z) - model_fwd.x.atan2(model_fwd.z);
        let rot = Quat::from_rotation_y(yaw);
        // Ground: the lowest part, or the terrain under the pelvis near it.
        let f = e.size.f;
        let lowest = e.parts.iter().map(|p| p.0.y).fold(f32::MAX, f32::min) - 0.07 * f;
        let ground = self
            .ground_ray(pelvis + Vec3::Y, Vec3::NEG_Y, 4.0)
            .map(|h| h.point.y)
            .filter(|g| (g - lowest).abs() < 0.3 * f)
            .unwrap_or(lowest);
        // Lying pose: pelvis over the model origin plus its rest offset.
        let rest = rig.frame * rig.bind_pos[PELVIS];
        let rest_xz = Vec3::new(rest.x, 0.0, rest.z);
        let origin = Vec3::new(pelvis.x, ground, pelvis.z) - rot * (s * rest_xz);
        let k = e.size.k();
        let q = rig.hip_height / 0.94;
        let up = GetUp {
            face_up,
            t: 0.0,
            hold: 0.25,
            duration: (if face_up { 2.3 } else { 2.5 }) * k.sqrt() / speed,
            blend: 0.5,
            end_z: (if face_up { END_Z_UP } else { END_Z_DOWN }) * q,
            handed: false,
        };
        let new = Mat4::from_scale_rotation_translation(s, rot, origin);
        if let Some(a) = self.animated.get_mut(&id) {
            a.transform = new;
        }
        let e = self.ragdolls.entries.get_mut(&id).expect("entry");
        e.up = Some(up);
        e.phase = Phase::GettingUp;
        e.t = 0.0;
    }

    /// Once a frame after physics: bodies drive skeletons, lying ones wait,
    /// standing ones blend back into their animation.
    pub(crate) fn tick_ragdolls(&mut self, dt: f32) {
        if self.ragdolls.entries.is_empty() {
            return;
        }
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 0.25)
        } else {
            0.0
        };
        let mut ids: Vec<EntityId> = self.ragdolls.entries.keys().copied().collect();
        ids.sort_by_key(|id| id.raw());
        let settings = self.ragdolls.settings;
        for id in ids {
            if !self.animated.contains_key(&id) {
                self.forget_ragdoll(id);
                continue;
            }
            let phase = self.ragdolls.entries[&id].phase;
            match phase {
                Phase::Tumbling => {
                    let poses = self.physics.ragdoll_poses(id);
                    let motion = self.physics.ragdoll_motion(id);
                    let e = self.ragdolls.entries.get_mut(&id).expect("entry");
                    e.t += dt;
                    let (Some(poses), Some((v, w, _))) = (poses, motion) else {
                        self.settle_ragdoll(id);
                        continue;
                    };
                    e.parts = poses;
                    let sk = e.size.k().sqrt();
                    if v < 0.3 * sk && w < 1.6 / sk {
                        e.calm += dt;
                    } else {
                        e.calm = 0.0;
                    }
                    if e.calm > 0.45 * sk || e.t > settings.max_tumble_seconds * sk {
                        self.settle_ragdoll(id);
                    }
                    if let Some(a) = self.animated.get(&id).map(|a| a.transform) {
                        let m = self.ragdolls.entries[&id].follow(a);
                        self.animated.get_mut(&id).expect("animated").transform = m;
                    }
                }
                Phase::Lying => {
                    let e = self.ragdolls.entries.get_mut(&id).expect("entry");
                    e.t += dt;
                    if !e.stay_down && e.t >= settings.get_up_after {
                        self.start_get_up(id);
                    }
                }
                Phase::GettingUp => {
                    let e = self.ragdolls.entries.get_mut(&id).expect("entry");
                    let up = e.up.as_mut().expect("getting up");
                    up.t += dt;
                    let end = up.hold + up.duration;
                    if !up.handed && up.t >= end {
                        // Standing: move the transform over the feet.
                        up.handed = true;
                        let shift = up.end_z;
                        if let Some(a) = self.animated.get_mut(&id) {
                            let (s, r, t) = split(a.transform);
                            let moved = t + r * (s * (e.rig.frame * Vec3::Z * shift));
                            a.transform = Mat4::from_scale_rotation_translation(s, r, moved);
                        }
                    }
                    if up.handed && up.t >= end + HAND_BACK {
                        self.forget_ragdoll(id);
                        self.ragdolls.events.push(RagdollEvent::GotUp { id });
                        continue;
                    }
                }
            }
            self.write_ragdoll_pose(id);
        }
    }

    /// Put the ragdoll's pose on the character's animator.
    fn write_ragdoll_pose(&mut self, id: EntityId) {
        let Some(e) = self.ragdolls.entries.get(&id) else {
            return;
        };
        let Some(a) = self.animated.get(&id) else {
            return;
        };
        let transform = a.transform;
        let skeleton = &a.animator.model().skeleton;
        let (parts, weight) = match (e.phase, e.up) {
            (Phase::GettingUp, Some(up)) => {
                let u = ((up.t - up.hold) / up.duration).clamp(0.0, 1.0);
                let shift = if up.handed { -up.end_z } else { 0.0 };
                let key = get_up_parts(&e.rig, up.face_up, u, shift);
                let lying = e.model_parts(transform);
                let w = smooth(0.0, up.blend, up.t);
                let parts: Vec<(Vec3, Quat)> = lying
                    .iter()
                    .zip(&key)
                    .map(|(a, b)| (a.0.lerp(b.0, w), a.1.slerp(b.1, w).normalize()))
                    .collect();
                let end = up.hold + up.duration;
                let weight = if up.handed {
                    1.0 - smooth(end, end + HAND_BACK, up.t)
                } else {
                    1.0
                };
                (parts, weight)
            }
            _ => (e.model_parts(transform), 1.0),
        };
        let locals = locals_from_parts(skeleton, &e.rig, &e.frozen, &e.scales, &parts);
        let a = self.animated.get_mut(&id).expect("animated");
        a.animator.pose_override = Some(PoseOverride { locals, weight });
    }

    /// Drop a character's ragdoll and hand it back to its animation.
    pub(crate) fn forget_ragdoll(&mut self, id: EntityId) {
        self.physics.remove_ragdoll(id);
        if self.ragdolls.entries.remove(&id).is_some() {
            if let Some(a) = self.animated.get_mut(&id) {
                a.animator.pose_override = None;
            }
        }
    }

    /// Is this character a ragdoll right now (tumbling, lying or getting up)?
    pub fn is_ragdoll(&self, id: EntityId) -> bool {
        self.ragdolls.entries.contains_key(&id)
    }
}

/// Nearest hit of a ray on a character's bones (capsules between joints).
fn ray_skeleton(e: &AnimatedEntity, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> {
    let globals = e.animator.joint_globals();
    let skeleton = &e.animator.model().skeleton;
    let (s, _, _) = split(e.transform);
    let at = |j: usize| e.transform.transform_point3(globals[j].w_axis.truncate());
    // Bone thickness from the character's size.
    let height = globals
        .iter()
        .map(|g| g.w_axis.y)
        .fold(f32::MIN, f32::max)
        .max(0.1)
        * s.y.abs();
    let r = height * 0.06;
    let mut best: Option<f32> = None;
    for (j, p) in skeleton.parents.iter().enumerate() {
        let Some(p) = *p else { continue };
        if let Some(t) = ray_capsule(origin, dir, at(p), at(j), r).filter(|t| *t <= max) {
            if best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        }
    }
    best
}

/// Distance along a ray to a capsule from `a` to `b`, if it hits.
fn ray_capsule(o: Vec3, d: Vec3, a: Vec3, b: Vec3, r: f32) -> Option<f32> {
    // Sample the segment: closest approach of the ray to each point.
    let steps = 6;
    let mut best: Option<f32> = None;
    for i in 0..=steps {
        let c = a.lerp(b, i as f32 / steps as f32);
        let t = (c - o).dot(d);
        if t < 0.0 {
            continue;
        }
        let miss = (o + d * t).distance(c);
        if miss < r {
            let back = (r * r - miss * miss).sqrt();
            let hit = (t - back).max(0.0);
            if best.is_none_or(|bt| hit < bt) {
                best = Some(hit);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::AnimatedModel;
    use crate::place::Place;

    fn character() -> Option<Arc<AnimatedModel>> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/model_viewer/assets/character.glb"
        );
        let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
        let limits = crate::limits::EngineLimits::default();
        match AnimatedModel::load_with(path, examples, &limits) {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => {
                eprintln!("skipped: no character at {path}: {e}");
                None
            }
        }
    }

    fn run(world: &mut World, seconds: f32) {
        let dt = 1.0 / 60.0;
        for _ in 0..(seconds / dt) as usize {
            world.tick_headless(dt);
        }
    }

    fn pelvis(world: &mut World, id: EntityId) -> Vec3 {
        let j = world.rig_of(id).unwrap().drivers[PELVIS];
        let e = &world.animated[&id];
        e.transform
            .transform_point3(e.animator.joint_globals()[j].w_axis.truncate())
    }

    #[test]
    fn ray_hits_capsule_side() {
        let t = ray_capsule(
            Vec3::new(0.0, 1.0, -5.0),
            Vec3::Z,
            Vec3::ZERO,
            Vec3::Y * 2.0,
            0.3,
        )
        .expect("hit");
        assert!((t - 4.7).abs() < 0.05, "{t}");
        assert!(
            ray_capsule(Vec3::new(2.0, 1.0, -5.0), Vec3::Z, Vec3::ZERO, Vec3::Y, 0.3).is_none()
        );
    }

    #[test]
    fn rig_is_found_and_the_first_pose_does_not_pop() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        let id = world
            .spawn_animated_shared(model, Place::new(1.0, 0.0, 2.0).with_yaw_deg(30.0))
            .unwrap();
        world.play_animation(id, "walk").unwrap();
        world.tick_animations(0.3);
        let before = world.animated[&id].animator.joint_globals();
        let rig = world.rig_of(id).unwrap();
        let names = &world.animated[&id].animator.model().skeleton.joint_names;
        assert_eq!(names[rig.drivers[PELVIS]].to_lowercase(), "pelvis");
        let right = rig.frame * Vec3::X;
        let up = rig.frame * Vec3::Y;
        assert!(up.y > 0.9, "up {up}");
        assert!(right.y.abs() < 0.2, "right {right}");
        world.knock_down(id, Knock::new(Vec3::ZERO)).unwrap();
        let after = world.animated[&id].animator.joint_globals();
        let worst = before
            .iter()
            .zip(&after)
            .map(|(a, b)| a.w_axis.truncate().distance(b.w_axis.truncate()))
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-3,
            "pose moved {worst} m when the ragdoll took over"
        );
    }

    #[test]
    fn knocked_down_tumbles_settles_and_gets_up() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        let id = world
            .spawn_animated_shared(model, Place::new(0.0, 0.0, 0.0))
            .unwrap();
        world.tick_headless(1.0 / 60.0);
        let chest = pelvis(&mut world, id) + Vec3::Y * 0.4;
        world
            .knock_down(id, Knock::new(Vec3::new(500.0, 120.0, 0.0)).at(chest))
            .unwrap();
        assert_eq!(world.ragdoll_state(id), Some(RagdollState::Tumbling));
        assert_eq!(world.ragdoll_count(), (1, 1));
        let mut settled = None;
        for _ in 0..600 {
            world.tick_headless(1.0 / 60.0);
            for ev in world.take_ragdoll_events() {
                if let RagdollEvent::Settled { at, .. } = ev {
                    settled = Some(at);
                }
            }
            if settled.is_some() {
                break;
            }
        }
        let at = settled.expect("settled within 10 s");
        assert!(at.x > 0.8, "thrown along the push: {at}");
        assert!(at.y < 0.45, "lying down: {at}");
        assert!(matches!(
            world.ragdoll_state(id),
            Some(RagdollState::Lying { .. })
        ));
        assert_eq!(world.ragdoll_count().0, 0, "no physics once lying");
        // Every part rests on or above the ground.
        let lowest = world.ragdolls.entries[&id]
            .parts
            .iter()
            .map(|p| p.0.y)
            .fold(f32::MAX, f32::min);
        assert!(lowest > -0.05, "through the floor: {lowest}");

        run(&mut world, 2.0);
        assert!(matches!(
            world.ragdoll_state(id),
            Some(RagdollState::GettingUp { .. })
        ));
        let mut got_up = false;
        for _ in 0..600 {
            world.tick_headless(1.0 / 60.0);
            got_up |= world
                .take_ragdoll_events()
                .iter()
                .any(|e| matches!(e, RagdollEvent::GotUp { .. }));
            if got_up {
                break;
            }
        }
        assert!(got_up, "stood up");
        assert!(!world.is_ragdoll(id));
        assert!(world.animated[&id].animator.pose_override.is_none());
        let t = world.animated[&id].transform;
        let (_, r, p) = t.to_scale_rotation_translation();
        assert!((r * Vec3::Y).y > 0.999, "upright");
        assert!(p.y.abs() < 0.1, "on the ground: {p}");
        assert!(
            p.distance(Vec3::new(at.x, 0.0, at.z)) < 1.2,
            "where it lay: {p} vs {at}"
        );
        let head = pelvis(&mut world, id);
        assert!(head.y > 0.7, "pelvis back at hip height: {head}");
    }

    #[test]
    fn stay_down_waits_for_get_up() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        let id = world
            .spawn_animated_shared(model, Place::default())
            .unwrap();
        world
            .knock_down(id, Knock::new(Vec3::new(0.0, 0.0, -300.0)).stay_down(true))
            .unwrap();
        run(&mut world, 9.0);
        assert!(matches!(
            world.ragdoll_state(id),
            Some(RagdollState::Lying { .. })
        ));
        world.get_up(id).unwrap();
        assert!(matches!(
            world.ragdoll_state(id),
            Some(RagdollState::GettingUp { .. })
        ));
        run(&mut world, 4.0);
        assert!(!world.is_ragdoll(id));
    }

    #[test]
    fn only_so_many_tumble_at_once() {
        let Some(model) = character() else { return };
        let mut world = World::default();
        world
            .set_ragdoll_settings(RagdollSettings {
                max_active: 2,
                ..Default::default()
            })
            .unwrap();
        let ids: Vec<EntityId> = (0..4)
            .map(|i| {
                world
                    .spawn_animated_shared(model.clone(), Place::new(i as f32 * 3.0, 0.0, 0.0))
                    .unwrap()
            })
            .collect();
        for id in &ids {
            world
                .knock_down(*id, Knock::new(Vec3::new(0.0, 200.0, 300.0)))
                .unwrap();
            assert!(world.ragdoll_count().0 <= 2);
        }
        assert_eq!(world.ragdoll_count(), (2, 4));
        assert_eq!(world.physics.ragdoll_count(), 2);
        world.despawn(ids[3]);
        assert_eq!(world.ragdoll_count().1, 3);
        assert_eq!(world.physics.ragdoll_count(), 1);
    }

    #[test]
    fn same_knock_same_fall() {
        let Some(model) = character() else { return };
        let fall = || {
            let mut world = World::default();
            let id = world
                .spawn_animated_shared(model.clone(), Place::new(0.0, 0.0, 0.0))
                .unwrap();
            world
                .knock_down(id, Knock::new(Vec3::new(200.0, 300.0, -150.0)))
                .unwrap();
            run(&mut world, 2.0);
            world.ragdolls.entries[&id].parts.clone()
        };
        let (a, b) = (fall(), fall());
        for (x, y) in a.iter().zip(&b) {
            assert!(x.0.distance(y.0) < 1e-4);
        }
    }

    #[test]
    fn get_up_poses_stay_above_the_ground() {
        let Some(model) = character() else { return };
        let mut bind = Vec::new();
        crate::anim::globals_from_locals(&model.skeleton, &model.bind_local, &mut bind);
        let rig = Rig::new(&model.skeleton, &bind).unwrap();
        let q = rig.hip_height / 0.94;
        for face_up in [true, false] {
            for i in 0..=20 {
                let u = i as f32 / 20.0;
                let parts = get_up_parts(&rig, face_up, u, 0.0);
                let lowest = parts.iter().map(|p| p.0.y).fold(f32::MAX, f32::min);
                assert!(lowest > 0.0, "face_up {face_up} u {u}: {lowest}");
            }
            // Standing at the end: the pelvis back at hip height.
            let end = get_up_parts(&rig, face_up, 1.0, 0.0);
            assert!(
                (end[PELVIS].0.y - rig.frame.mul_vec3(rig.bind_pos[PELVIS]).y).abs() < 0.02 * q
            );
        }
    }
}
