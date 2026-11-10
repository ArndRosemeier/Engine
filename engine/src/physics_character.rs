//! A walking character: a capsule that slides along walls, climbs steps
//! and slopes, sticks to the ground going downhill, falls, jumps and
//! shoves crates.
//!
//! Part of `crate::physics`, split out by topic.

use super::physics_gameplay::body_groups;
use super::*;
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::parry::shape::Capsule;

/// The gap the character controller keeps to everything around it.
const SKIN: f32 = 0.02;

/// How a character is built. `Character::default()` is a person: 1.8 m
/// tall, 0.35 m wide, climbs 0.35 m steps and 45 degree slopes.
///
/// ```ignore
/// world.add_character(player, Character::default())?;
/// // Every frame: walk where the keys say, jump on Space.
/// world.walk(player, frame.input.move_dir_xz(yaw) * 5.0)?;
/// if frame.input.pressed(Key::Space) { world.jump(player, 5.0)?; }
/// let grounded = world.character(player).is_some_and(|c| c.grounded);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Character {
    /// Total height in metres; the entity's place is the feet.
    pub height: f32,
    pub radius: f32,
    /// Ledges up to this high are stepped onto.
    pub step_height: f32,
    /// Steepest slope walked up, in degrees.
    pub max_slope_deg: f32,
    /// Going down slopes and steps up to this far, the feet stay on the
    /// ground instead of flying off.
    pub snap_to_ground: f32,
    /// Kilograms, for shoving dynamic bodies.
    pub mass: f32,
    /// 1 is normal gravity.
    pub gravity_scale: f32,
    /// Collision layer, as for [`Body::layer`].
    pub layer: u8,
    pub collides_with: u32,
}

impl Default for Character {
    fn default() -> Self {
        Self {
            height: 1.8,
            radius: 0.35,
            step_height: 0.35,
            max_slope_deg: 45.0,
            snap_to_ground: 0.3,
            mass: 80.0,
            gravity_scale: 1.0,
            layer: 0,
            collides_with: u32::MAX,
        }
    }
}

impl Character {
    pub fn size(mut self, height: f32, radius: f32) -> Self {
        self.height = height;
        self.radius = radius;
        self
    }
    pub fn step_height(mut self, metres: f32) -> Self {
        self.step_height = metres;
        self
    }
    pub fn max_slope(mut self, degrees: f32) -> Self {
        self.max_slope_deg = degrees;
        self
    }
    pub fn mass(mut self, kg: f32) -> Self {
        self.mass = kg;
        self
    }
    pub fn gravity_scale(mut self, scale: f32) -> Self {
        self.gravity_scale = scale;
        self
    }
    pub fn layer(mut self, layer: u8) -> Self {
        self.layer = layer;
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("character {what}")));
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return bad("radius must be > 0");
        }
        if !(self.height.is_finite() && self.height >= self.radius * 2.0) {
            return bad("height must be at least twice the radius");
        }
        if !(self.step_height.is_finite() && self.step_height >= 0.0) {
            return bad("step_height must be >= 0");
        }
        if !(self.max_slope_deg.is_finite() && (0.0..90.0).contains(&self.max_slope_deg)) {
            return bad("max_slope_deg must be in 0..90");
        }
        if !(self.snap_to_ground.is_finite() && self.snap_to_ground >= 0.0) {
            return bad("snap_to_ground must be >= 0");
        }
        if !(self.mass.is_finite() && self.mass > 0.0) || !self.gravity_scale.is_finite() {
            return bad("mass must be > 0 and gravity_scale finite");
        }
        if self.layer > MAX_LAYER {
            return bad("layer is too high");
        }
        Ok(())
    }

    /// The capsule's centre for feet at `feet`. The capsule keeps a small
    /// gap (`SKIN`) to what it stands on, so it hangs that much lower and
    /// the feet touch the ground.
    fn center(&self, feet: Vec3) -> Vec3 {
        feet + Vec3::Y * (self.height * 0.5 - SKIN)
    }

    fn half_height(&self) -> f32 {
        (self.height * 0.5 - self.radius).max(0.0)
    }

    fn controller(&self) -> KinematicCharacterController {
        KinematicCharacterController {
            offset: CharacterLength::Absolute(SKIN),
            autostep: (self.step_height > 0.0).then_some(CharacterAutostep {
                max_height: CharacterLength::Absolute(self.step_height),
                min_width: CharacterLength::Absolute(self.radius * 0.5),
                include_dynamic_bodies: false,
            }),
            max_slope_climb_angle: self.max_slope_deg.to_radians(),
            min_slope_slide_angle: self.max_slope_deg.to_radians(),
            snap_to_ground: (self.snap_to_ground > 0.0)
                .then_some(CharacterLength::Absolute(self.snap_to_ground)),
            ..KinematicCharacterController::default()
        }
    }
}

/// What a character did in the last physics step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CharacterState {
    /// Standing on something.
    pub grounded: bool,
    /// How fast it actually moved, m/s (after walls and slopes).
    pub velocity: Vec3,
    /// Where its feet are.
    pub feet: Vec3,
}

#[derive(Debug)]
pub(super) struct CharacterTracked {
    handle: RigidBodyHandle,
    collider: ColliderHandle,
    spec: Character,
    /// What the game asked for: horizontal speed, and a jump to start.
    walk: Vec3,
    jump: Option<f32>,
    vertical: f32,
    state: CharacterState,
    prev: Vec3,
    cur: Vec3,
    /// The feet position physics last gave the entity.
    written: Vec3,
    /// A piece of the flat floor under it, for its shape casts only.
    floor: Option<(ColliderHandle, Vec3)>,
}

/// Half the side of a character's floor piece, and how far it walks
/// before the piece is moved under it again.
const FLOOR_HALF: f32 = 32.0;
const FLOOR_RECENTER: f32 = 16.0;

impl Physics {
    /// Make the entity at `feet` a character. Replaces one it already has.
    pub fn add_character(
        &mut self,
        id: EntityId,
        spec: &Character,
        feet: Vec3,
    ) -> EngineResult<()> {
        spec.validate()?;
        crate::place::ensure_finite3(feet, "character place")?;
        if self.bodies.contains_key(&id) {
            return Err(EngineError::InvalidValue(format!(
                "character: entity {id} already has a body; remove_body first"
            )));
        }
        self.remove_character(id);
        let center = spec.center(feet);
        let handle = self.world.insert_body(
            RigidBodyBuilder::kinematic_position_based()
                .translation(center)
                .user_data(tag(id))
                .build(),
        );
        let collider = self.world.insert_collider(
            ColliderBuilder::capsule_y(spec.half_height(), spec.radius)
                .user_data(tag(id))
                .collision_groups(body_groups(spec.layer, spec.collides_with)),
            Some(handle),
        );
        self.characters.insert(
            id,
            CharacterTracked {
                handle,
                collider,
                spec: *spec,
                walk: Vec3::ZERO,
                jump: None,
                vertical: 0.0,
                state: CharacterState {
                    feet,
                    ..CharacterState::default()
                },
                prev: feet,
                cur: feet,
                written: feet,
                floor: None,
            },
        );
        Ok(())
    }

    pub fn remove_character(&mut self, id: EntityId) {
        if let Some(c) = self.characters.remove(&id) {
            self.world.remove_body_with_colliders(c.handle, true);
            if let Some((floor, _)) = c.floor {
                self.world.remove_collider(floor);
            }
        }
    }

    /// Keep a piece of the flat floor (when the ground is a plane) under a
    /// character. It touches nothing (no interaction groups) and is ground
    /// for every query, so only the character's casts see it.
    fn place_floor(&mut self, id: EntityId) {
        let want = match self.source {
            GroundSource::Plane(y) => Some(y),
            _ => None,
        };
        let c = &self.characters[&id];
        let feet = c.cur;
        let keep = match (c.floor, want) {
            (Some((_, at)), Some(y)) => {
                at.y == y && Vec2::new(at.x - feet.x, at.z - feet.z).length() < FLOOR_RECENTER
            }
            (None, None) => true,
            _ => false,
        };
        if keep {
            return;
        }
        if let Some((old, _)) = self.characters.get_mut(&id).and_then(|c| c.floor.take()) {
            self.world.remove_collider(old);
        }
        if let Some(y) = want {
            let at = Vec3::new(feet.x, y, feet.z);
            let h = self.world.insert_collider(
                ColliderBuilder::cuboid(FLOOR_HALF, 5.0, FLOOR_HALF)
                    .translation(at - Vec3::Y * 5.0)
                    .user_data(GROUND_TAG)
                    .collision_groups(super::physics_gameplay::ground_groups()),
                None,
            );
            if let Some(c) = self.characters.get_mut(&id) {
                c.floor = Some((h, at));
            }
        }
    }

    pub fn is_character(&self, id: EntityId) -> bool {
        self.characters.contains_key(&id)
    }

    pub fn character_ids(&self) -> Vec<EntityId> {
        let mut ids: Vec<EntityId> = self.characters.keys().copied().collect();
        ids.sort_by_key(|id| id.raw());
        ids
    }

    pub fn character_state(&self, id: EntityId) -> Option<CharacterState> {
        self.characters.get(&id).map(|c| c.state)
    }

    pub fn walk(&mut self, id: EntityId, velocity: Vec3) -> EngineResult<()> {
        crate::place::ensure_finite3(velocity, "walk velocity")?;
        let c = self.characters.get_mut(&id).ok_or_else(not_a_character)?;
        c.walk = Vec3::new(velocity.x, 0.0, velocity.z);
        Ok(())
    }

    /// Start a jump at `speed` m/s upwards. Only from the ground; returns
    /// whether it jumped.
    pub fn jump(&mut self, id: EntityId, speed: f32) -> EngineResult<bool> {
        if !(speed.is_finite() && speed > 0.0) {
            return Err(EngineError::InvalidValue(format!(
                "jump speed must be > 0, got {speed}"
            )));
        }
        let c = self.characters.get_mut(&id).ok_or_else(not_a_character)?;
        if c.state.grounded {
            c.jump = Some(speed);
        }
        Ok(c.state.grounded)
    }

    /// Follow the entity's place; `None` means it is gone. A place the
    /// game set (not the one physics wrote) teleports the character.
    pub fn sync_character(&mut self, id: EntityId, feet: Option<Vec3>) {
        let Some(feet) = feet.filter(|f| f.is_finite()) else {
            self.remove_character(id);
            return;
        };
        let Some(c) = self.characters.get_mut(&id) else {
            return;
        };
        if feet.distance_squared(c.written) < 1e-10 {
            return;
        }
        c.written = feet;
        c.prev = feet;
        c.cur = feet;
        c.vertical = 0.0;
        c.state.feet = feet;
        let center = c.spec.center(feet);
        if let Some(b) = self.world.bodies.get_mut(c.handle) {
            b.set_position(pose(center, Quat::IDENTITY), true);
        }
    }

    /// One fixed step of every character, before the world steps.
    pub(super) fn step_characters(&mut self, h: f32) {
        let g = self.settings.gravity.y;
        for id in self.character_ids() {
            let c = &self.characters[&id];
            let spec = c.spec;
            let controller = spec.controller();
            let shape = Capsule::new_y(spec.half_height(), spec.radius);
            let center = spec.center(c.cur);
            let mut vertical = c.vertical;
            if let Some(speed) = c.jump {
                vertical = speed;
            }
            // Standing: no push into the ground (snapping keeps the feet on
            // it going downhill). Otherwise gravity.
            let standing = c.state.grounded && c.jump.is_none() && vertical <= 0.0;
            vertical = if standing {
                0.0
            } else {
                vertical + g * spec.gravity_scale * h
            };
            let desired = c.walk * h + Vec3::Y * vertical * h;
            let (own, jumping) = (c.collider, c.jump.is_some());
            // The flat floor is a box too big for exact shape casts: the
            // character walks on a smaller copy of it instead.
            self.place_floor(id);
            let c = &self.characters[&id];
            let skip_floor = self.plane;
            let not_floor =
                |h: ColliderHandle, _: &rapier3d::prelude::Collider| Some(h) != skip_floor;
            let mut hits = Vec::new();
            let moved = {
                let filter = QueryFilter::default()
                    .exclude_sensors()
                    .exclude_collider(own)
                    .predicate(&not_floor);
                let queries = self.world.query_pipeline_with_filter(filter);
                let mut ctl = controller;
                if jumping {
                    // Leave the ground instead of snapping back to it.
                    ctl.snap_to_ground = None;
                }
                ctl.move_shape(
                    h,
                    &queries,
                    &shape,
                    &pose(center, Quat::IDENTITY),
                    desired,
                    |hit| hits.push(hit),
                )
            };
            // Shove the dynamic bodies it walked into.
            if !hits.is_empty() {
                let filter = QueryFilter::default()
                    .exclude_sensors()
                    .exclude_collider(own)
                    .predicate(&not_floor);
                let w = &mut self.world;
                let dispatcher = w.narrow_phase.query_dispatcher();
                let mut queries = w.broad_phase.as_query_pipeline_mut(
                    dispatcher,
                    &mut w.bodies,
                    &mut w.colliders,
                    filter,
                );
                controller.solve_character_collision_impulses(
                    h,
                    &mut queries,
                    &shape,
                    spec.mass,
                    &hits,
                );
            }
            let hit_ceiling = vertical > 0.0
                && hits.iter().any(|hit| hit.hit.normal1.y < -0.5)
                && moved.translation.y < desired.y * 0.5;
            let feet = c.cur + moved.translation;
            let c = self.characters.get_mut(&id).expect("character");
            c.jump = None;
            c.vertical = if (moved.grounded && vertical <= 0.0) || hit_ceiling {
                0.0
            } else {
                vertical
            };
            c.prev = c.cur;
            c.cur = feet;
            c.state = CharacterState {
                grounded: moved.grounded,
                velocity: moved.translation / h,
                feet,
            };
            if let Some(b) = self.world.bodies.get_mut(c.handle) {
                b.set_next_kinematic_position(pose(spec.center(feet), Quat::IDENTITY));
            }
        }
    }

    /// Feet positions to draw, between the last two steps.
    pub(super) fn character_places(&mut self, alpha: f32) -> Vec<(EntityId, Vec3)> {
        let mut out = Vec::new();
        for id in self.character_ids() {
            let c = self.characters.get_mut(&id).expect("character");
            let feet = c.prev.lerp(c.cur, alpha);
            if feet != c.written {
                c.written = feet;
                out.push((id, feet));
            }
        }
        out
    }

    /// Rigid-body handles of characters, for building ground around them.
    pub(super) fn character_handles(&self) -> Vec<RigidBodyHandle> {
        let mut ids: Vec<(u64, RigidBodyHandle)> = self
            .characters
            .iter()
            .map(|(id, c)| (id.raw(), c.handle))
            .collect();
        ids.sort_by_key(|(id, _)| *id);
        ids.into_iter().map(|(_, h)| h).collect()
    }
}

fn not_a_character() -> EngineError {
    EngineError::InvalidValue("not a character: call add_character first".into())
}
