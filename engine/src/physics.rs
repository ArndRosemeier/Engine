//! Rigid-body physics: boxes that stack, crates that tumble down hills,
//! rubble that piles up, things that float.
//!
//! Give any mesh entity a [`Body`] with [`World::add_body`] and the engine
//! moves it every frame. The ground comes for free: voxel terrain and height
//! terrain become colliders around each moving body as it needs them (and
//! are rebuilt after digging), or a flat floor at y = 0 when the world has no
//! terrain. Bodies in water float or sink by their density.
//!
//! ```ignore
//! let crate_id = world.spawn(Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(150, 100, 50))?);
//! world.set_place(crate_id, Place::new(0.0, 5.0, 0.0))?;
//! world.add_body(crate_id, Body::dynamic().density(0.6))?;
//! world.push(crate_id, Vec3::new(0.0, 0.0, 300.0))?;
//! ```
//!
//! Under the hood this is [Rapier](https://rapier.rs), stepped at a fixed
//! rate and interpolated for drawing, so motion is smooth at any frame rate.
//!
//! [`PhysicsWorld`] is the same physics without a `World`, a window or a GPU,
//! for a game server or tests; it owns its bodies by [`BodyId`].
//!
//! [`World::add_body`]: crate::world::World::add_body

use crate::error::{EngineError, EngineResult};
use crate::world::EntityId;
use glam::{IVec3, Mat4, Quat, Vec2, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, PhysicsWorld as RapierWorld, QueryFilter, Ray,
    RigidBodyBuilder, RigidBodyHandle, TriMeshFlags,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Water: 1000 kg per cubic metre. Densities in this module are relative to it.
const WATER_KG_M3: f32 = 1000.0;
/// `user_data` tag for ground colliders.
const GROUND_TAG: u128 = u128::MAX;
/// Side of one height-terrain collider tile in metres, and its samples per side.
const HEIGHT_TILE_M: f32 = 16.0;
const HEIGHT_TILE_SAMPLES: usize = 33;
/// Seconds an unused ground tile is kept before it is dropped.
const TILE_KEEP_S: f32 = 3.0;
/// Rubble sinks into the ground for this long before it vanishes.
const SINK_SECONDS: f32 = 2.0;

#[path = "physics_gameplay.rs"]
mod physics_gameplay;
#[path = "physics_ragdoll.rs"]
mod physics_ragdoll;
use physics_gameplay::{body_groups, ground_groups, TriggerTracked};
#[path = "physics_joints.rs"]
mod physics_joints;
use physics_joints::JointTracked;
pub use physics_joints::{Joint, JointId, JointKind};
#[path = "physics_character.rs"]
mod physics_character;
use physics_character::CharacterTracked;
pub use physics_character::{Character, CharacterState};
pub use physics_gameplay::{Trigger, TriggerEvent, TriggerShape, MAX_LAYER};
pub(crate) use physics_ragdoll::{JointDesc, LinkDesc, PartDesc, PartShape};

/// How a body moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BodyKind {
    /// Falls, collides, gets pushed.
    #[default]
    Dynamic,
    /// Never moves (walls, floors, static props). Other bodies collide with it.
    /// Moving the entity with `set_place` teleports it.
    Fixed,
    /// Moved by you with `set_place`; pushes dynamic bodies out of the way
    /// (lifts, doors, a swinging hammer). Not affected by anything.
    Kinematic,
}

/// The collision shape of a body, in the entity's own (unscaled) space.
/// The entity's scale is applied on top.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum BodyShape {
    /// The convex hull of the entity's mesh. Good for props, rocks, rubble.
    #[default]
    Hull,
    /// The exact triangle mesh. Only for [`BodyKind::Fixed`] and
    /// [`BodyKind::Kinematic`] bodies (walls, buildings, terrain pieces).
    Mesh,
    /// The mesh's bounding box.
    BoundingBox,
    /// A box with these half extents.
    Box(Vec3),
    /// A sphere.
    Ball(f32),
    /// An upright capsule: half the straight part's length, and the radius.
    Capsule { half_height: f32, radius: f32 },
}

/// Everything about a physics body. Start from [`Body::dynamic`],
/// [`Body::fixed`] or [`Body::kinematic`] and change what you need.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub kind: BodyKind,
    pub shape: BodyShape,
    /// Relative to water: wood 0.6 floats, stone 2.4 sinks.
    pub density: f32,
    /// 0 is ice, 1 is rubber on concrete.
    pub friction: f32,
    /// 0 is a sandbag, 1 bounces back to where it fell from.
    pub bounce: f32,
    /// Starting velocity in m/s.
    pub velocity: Vec3,
    /// Starting spin in rad/s around each axis.
    pub spin: Vec3,
    /// Continuous collision detection: stops small fast things (bullets,
    /// pebbles) from passing through thin walls. Costs a little.
    pub ccd: bool,
    /// How quickly motion and spin die down in air (0 = never).
    pub damping: f32,
    /// Collision layer, 0 to [`MAX_LAYER`] (default 0). See [`Body::layer`].
    pub layer: u8,
    /// Which layers this body collides with, one bit per layer (default:
    /// all). The ground and triggers are always included.
    pub collides_with: u32,
}

impl Default for Body {
    fn default() -> Self {
        Self::dynamic()
    }
}

impl Body {
    /// A body that falls and collides, with the mesh's convex hull as shape.
    pub fn dynamic() -> Self {
        Self {
            kind: BodyKind::Dynamic,
            shape: BodyShape::Hull,
            density: 1.0,
            friction: 0.6,
            bounce: 0.1,
            velocity: Vec3::ZERO,
            spin: Vec3::ZERO,
            ccd: false,
            damping: 0.05,
            layer: 0,
            collides_with: u32::MAX,
        }
    }

    /// A body that never moves, with the exact mesh as shape.
    pub fn fixed() -> Self {
        Self {
            kind: BodyKind::Fixed,
            shape: BodyShape::Mesh,
            ..Self::dynamic()
        }
    }

    /// A body you move by hand that shoves dynamic bodies aside.
    pub fn kinematic() -> Self {
        Self {
            kind: BodyKind::Kinematic,
            shape: BodyShape::Hull,
            ..Self::dynamic()
        }
    }

    pub fn shape(mut self, shape: BodyShape) -> Self {
        self.shape = shape;
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    pub fn friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    pub fn bounce(mut self, bounce: f32) -> Self {
        self.bounce = bounce;
        self
    }

    pub fn velocity(mut self, velocity: impl Into<Vec3>) -> Self {
        self.velocity = velocity.into();
        self
    }

    pub fn spin(mut self, spin: impl Into<Vec3>) -> Self {
        self.spin = spin.into();
        self
    }

    pub fn ccd(mut self, ccd: bool) -> Self {
        self.ccd = ccd;
        self
    }

    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Put the body on a collision layer (0 to [`MAX_LAYER`]). Two bodies
    /// collide only when each one's `collides_with` has the other's layer:
    /// `Body::dynamic().layer(2).collides_with(&[0, 2])` lets debris (layer 2)
    /// hit the world (0) and other debris but pass through the player (1).
    pub fn layer(mut self, layer: u8) -> Self {
        self.layer = layer;
        self
    }

    /// Collide only with bodies on these layers (and always with the ground).
    pub fn collides_with(mut self, layers: &[u8]) -> Self {
        self.collides_with = layers
            .iter()
            .filter(|l| **l <= MAX_LAYER)
            .fold(0, |m, l| m | 1 << l);
        self
    }

    /// Pass through bodies on these layers.
    pub fn ignores(mut self, layers: &[u8]) -> Self {
        for l in layers.iter().filter(|l| **l <= MAX_LAYER) {
            self.collides_with &= !(1 << l);
        }
        self
    }

    pub fn validate(&self) -> EngineResult<()> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("body {what}")));
        if !self.density.is_finite() || self.density <= 0.0 {
            return bad("density must be finite and > 0");
        }
        if !self.friction.is_finite() || self.friction < 0.0 {
            return bad("friction must be finite and >= 0");
        }
        if !self.bounce.is_finite() || !(0.0..=1.0).contains(&self.bounce) {
            return bad("bounce must be in 0..=1");
        }
        if !self.damping.is_finite() || self.damping < 0.0 {
            return bad("damping must be finite and >= 0");
        }
        if !self.velocity.is_finite() || !self.spin.is_finite() {
            return bad("velocity and spin must be finite");
        }
        if self.layer > MAX_LAYER {
            return Err(EngineError::InvalidValue(format!(
                "body layer must be 0..={MAX_LAYER}, got {}",
                self.layer
            )));
        }
        match self.shape {
            BodyShape::Mesh if self.kind == BodyKind::Dynamic => {
                return bad("shape Mesh only works for fixed and kinematic bodies; use Hull");
            }
            BodyShape::Box(h) if !(h.is_finite() && h.min_element() > 0.0) => {
                return bad("box half extents must be > 0");
            }
            BodyShape::Ball(r) if !(r.is_finite() && r > 0.0) => {
                return bad("ball radius must be > 0");
            }
            BodyShape::Capsule {
                half_height,
                radius,
            } if !(half_height.is_finite()
                && half_height >= 0.0
                && radius.is_finite()
                && radius > 0.0) =>
            {
                return bad("capsule needs half_height >= 0 and radius > 0");
            }
            _ => {}
        }
        Ok(())
    }
}

/// What dynamic bodies land on when nothing else is in the way.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum PhysicsGround {
    /// The voxel or height terrain; a flat floor at y = 0 without one.
    #[default]
    Auto,
    /// A flat floor at this height, ignoring any terrain.
    Plane(f32),
    /// Nothing: bodies fall forever unless a fixed body catches them.
    None,
}

/// World-wide physics settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicsSettings {
    /// m/s². Earth by default.
    pub gravity: Vec3,
    pub ground: PhysicsGround,
    /// Simulation steps per second. Drawing is interpolated between steps,
    /// so 60 looks smooth even at 200 FPS.
    pub steps_per_second: f32,
    /// Most steps taken in one frame; a long hitch slows physics down
    /// instead of freezing the game catching up.
    pub max_steps_per_frame: u32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            ground: PhysicsGround::Auto,
            steps_per_second: 60.0,
            max_steps_per_frame: 4,
        }
    }
}

impl PhysicsSettings {
    pub fn validate(&self) -> EngineResult<()> {
        if !self.gravity.is_finite() {
            return Err(EngineError::InvalidValue("gravity must be finite".into()));
        }
        if !self.steps_per_second.is_finite() || !(10.0..=1000.0).contains(&self.steps_per_second) {
            return Err(EngineError::InvalidValue(
                "steps_per_second must be in 10..=1000".into(),
            ));
        }
        if self.max_steps_per_frame == 0 {
            return Err(EngineError::InvalidValue(
                "max_steps_per_frame must be >= 1".into(),
            ));
        }
        if let PhysicsGround::Plane(y) = self.ground {
            if !y.is_finite() {
                return Err(EngineError::InvalidValue(
                    "ground plane must be finite".into(),
                ));
            }
        }
        Ok(())
    }
}

/// What a ray hit: a body, or the ground (`entity` is `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub entity: Option<EntityId>,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

/// Two things hitting each other hard enough to matter (for sounds, damage,
/// dust). `b` is `None` when `a` hit the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Impact {
    pub a: EntityId,
    pub b: Option<EntityId>,
    pub point: Vec3,
    /// Closing speed along the contact normal, m/s.
    pub speed: f32,
}

/// Where the ground comes from this frame; tiles are rebuilt when it changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GroundSource {
    None,
    Plane(f32),
    Voxels { id: usize, voxel_size: f32 },
    Heights { id: usize },
}

/// Ground height at (x, z), shared by tiles built on worker threads.
pub(crate) type HeightFn<'a> = &'a (dyn Fn(f32, f32) -> f32 + Sync);

/// What the world lends physics for one tick.
pub(crate) struct Surroundings<'a> {
    pub voxels: Option<&'a crate::voxel_terrain::VoxelTerrain>,
    /// A height function and an id that changes only when the terrain does.
    pub heights: Option<(usize, HeightFn<'a>)>,
    pub water: Option<&'a crate::water::Water>,
}

impl Surroundings<'_> {
    fn source(&self, ground: PhysicsGround) -> GroundSource {
        match ground {
            PhysicsGround::None => GroundSource::None,
            PhysicsGround::Plane(y) => GroundSource::Plane(y),
            PhysicsGround::Auto => {
                if let Some(v) = self.voxels {
                    if v.volume().chunk_keys().next().is_some() {
                        return GroundSource::Voxels {
                            id: v as *const _ as usize,
                            voxel_size: v.volume().voxel_size(),
                        };
                    }
                }
                if let Some((id, _)) = self.heights {
                    return GroundSource::Heights { id };
                }
                GroundSource::Plane(0.0)
            }
        }
    }

    /// Water surface height and current at `p`, if there is water nearby.
    pub fn water_at(&self, p: Vec3) -> Option<(f32, Vec2)> {
        let w = self.water?;
        let surface = w.surface_at(p + Vec3::Y * 2.0)?;
        (surface > p.y - 3.0).then(|| {
            (
                surface,
                w.sample(p.min(Vec3::new(p.x, surface - 0.05, p.z)))
                    .map_or(Vec2::ZERO, |s| s.current),
            )
        })
    }
}

/// A splash for the water to show.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Splash {
    pub at: Vec3,
    pub speed: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Temporary {
    pub keep: f32,
}

#[derive(Clone, Debug)]
struct Tracked {
    handle: RigidBodyHandle,
    kind: BodyKind,
    scale: Vec3,
    /// Pose at the previous and current step, for drawing in between.
    prev: (Vec3, Quat),
    cur: (Vec3, Quat),
    /// The transform physics last gave the entity; anything else means the
    /// game moved it.
    written: Mat4,
    /// Local points sampled for buoyancy.
    probes: Vec<Vec3>,
    volume: f32,
    density: f32,
    wet: bool,
    age: f32,
    temporary: Option<Temporary>,
}

/// Rubble whose time is up, sinking into the ground.
#[derive(Clone, Copy, Debug)]
struct Sinking {
    id: EntityId,
    left: f32,
}

#[derive(Debug)]
struct Tile {
    collider: Option<ColliderHandle>,
    last_needed: f32,
}

/// The physics world inside [`crate::world::World`].
pub(crate) struct Physics {
    world: RapierWorld,
    settings: PhysicsSettings,
    bodies: HashMap<EntityId, Tracked>,
    sinking: Vec<Sinking>,
    tiles: HashMap<IVec3, Tile>,
    source: GroundSource,
    plane: Option<ColliderHandle>,
    voxel_rev: u64,
    clock: f32,
    accumulator: f32,
    impacts: Vec<Impact>,
    /// Fixed boxes owned by something else (building elements), by owner and part.
    statics: HashMap<(u64, u32), ColliderHandle>,
    /// Ragdoll bodies by the character that owns them.
    ragdolls: HashMap<EntityId, physics_ragdoll::RagBodies>,
    /// How far between the last two steps things are drawn.
    alpha: f32,
    /// Trigger zones by the entity that carries them.
    triggers: HashMap<EntityId, TriggerTracked>,
    trigger_events: Vec<TriggerEvent>,
    joints: HashMap<JointId, JointTracked>,
    next_joint: u64,
    /// Walking characters by their entity.
    characters: HashMap<EntityId, CharacterTracked>,
}

impl std::fmt::Debug for Physics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Physics")
            .field("bodies", &self.bodies.len())
            .field("tiles", &self.tiles.len())
            .finish()
    }
}

impl Default for Physics {
    fn default() -> Self {
        Self::new()
    }
}

fn tag(id: EntityId) -> u128 {
    id.raw() as u128
}

fn split(m: Mat4) -> (Vec3, Quat, Vec3) {
    let (s, r, t) = m.to_scale_rotation_translation();
    (s, r.normalize(), t)
}

fn pose(t: Vec3, r: Quat) -> rapier3d::math::Pose {
    rapier3d::math::Pose::from_parts(t, r)
}

/// Up to a dozen points spread over a point cloud: the extreme point along
/// a fixed set of directions. Used to sample how much of a body is under water.
pub(crate) fn probe_points(positions: &[Vec3]) -> Vec<Vec3> {
    let dirs = [
        Vec3::X,
        -Vec3::X,
        Vec3::Y,
        -Vec3::Y,
        Vec3::Z,
        -Vec3::Z,
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
    ];
    let mut out: Vec<Vec3> = Vec::with_capacity(dirs.len());
    for d in dirs {
        if let Some(&p) = positions
            .iter()
            .max_by(|a, b| a.dot(d).total_cmp(&b.dot(d)))
        {
            if !out.iter().any(|q| q.distance_squared(p) < 1e-8) {
                out.push(p);
            }
        }
    }
    out
}

fn box_points(min: Vec3, max: Vec3) -> Vec<Vec3> {
    (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { min.x } else { max.x },
                if i & 2 == 0 { min.y } else { max.y },
                if i & 4 == 0 { min.z } else { max.z },
            )
        })
        .collect()
}

impl Physics {
    pub fn new() -> Self {
        let mut world = RapierWorld::new();
        let settings = PhysicsSettings::default();
        world.gravity = settings.gravity;
        world.integration_parameters.dt = 1.0 / settings.steps_per_second;
        Self {
            world,
            settings,
            bodies: HashMap::new(),
            sinking: Vec::new(),
            tiles: HashMap::new(),
            source: GroundSource::None,
            plane: None,
            voxel_rev: 0,
            clock: 0.0,
            accumulator: 0.0,
            impacts: Vec::new(),
            statics: HashMap::new(),
            ragdolls: HashMap::new(),
            alpha: 1.0,
            triggers: HashMap::new(),
            trigger_events: Vec::new(),
            joints: HashMap::new(),
            next_joint: 0,
            characters: HashMap::new(),
        }
    }

    pub fn settings(&self) -> PhysicsSettings {
        self.settings
    }

    pub fn set_settings(&mut self, settings: PhysicsSettings) {
        self.settings = settings;
        self.world.gravity = settings.gravity;
        self.world.integration_parameters.dt = 1.0 / settings.steps_per_second;
        self.world.wake_up_all(true);
    }

    pub fn contains(&self, id: EntityId) -> bool {
        self.bodies.contains_key(&id)
    }

    pub fn body_kind(&self, id: EntityId) -> Option<BodyKind> {
        self.bodies.get(&id).map(|t| t.kind)
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn awake_count(&self) -> usize {
        self.bodies
            .values()
            .filter(|t| {
                t.kind == BodyKind::Dynamic
                    && self
                        .world
                        .bodies
                        .get(t.handle)
                        .is_some_and(|b| !b.is_sleeping())
            })
            .count()
    }

    pub fn temporary_count(&self) -> usize {
        self.bodies
            .values()
            .filter(|t| t.temporary.is_some())
            .count()
            + self.sinking.len()
    }

    /// Add a body for an entity whose mesh positions are `positions`
    /// (local space) and whose transform is `transform`.
    pub fn add(
        &mut self,
        id: EntityId,
        positions: &[Vec3],
        transform: Mat4,
        body: &Body,
        temporary: Option<Temporary>,
        indices: impl FnOnce() -> Vec<[u32; 3]>,
    ) -> EngineResult<()> {
        body.validate()?;
        if self.characters.contains_key(&id) {
            return Err(EngineError::InvalidValue(format!(
                "entity {id} is a character; remove_character before add_body"
            )));
        }
        self.remove(id);
        let (scale, rotation, translation) = split(transform);
        if !scale.is_finite() || scale.min_element() <= 1e-6 {
            return Err(EngineError::InvalidValue(
                "body entity needs a finite, non-zero scale".into(),
            ));
        }
        let scaled: Vec<Vec3> = positions.iter().map(|p| *p * scale).collect();
        let (min, max) = if scaled.is_empty() {
            (Vec3::splat(-0.5), Vec3::splat(0.5))
        } else {
            scaled.iter().fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| {
                (lo.min(*p), hi.max(*p))
            })
        };
        let no_mesh = || EngineError::InvalidValue("body shape needs a mesh with points".into());
        let (builder, probes) = match &body.shape {
            BodyShape::Hull => {
                if scaled.len() < 4 {
                    return Err(no_mesh());
                }
                let b = ColliderBuilder::convex_hull(&scaled)
                    .or_else(|| {
                        // Flat or degenerate meshes: fall back to the box.
                        let h = ((max - min) * 0.5).max(Vec3::splat(0.01));
                        Some(ColliderBuilder::cuboid(h.x, h.y, h.z).translation((min + max) * 0.5))
                    })
                    .expect("fallback box");
                (b, probe_points(&scaled))
            }
            BodyShape::Mesh => {
                let tris = indices();
                if scaled.is_empty() || tris.is_empty() {
                    return Err(no_mesh());
                }
                let b = ColliderBuilder::trimesh_with_flags(
                    scaled.clone(),
                    tris.clone(),
                    TriMeshFlags::MERGE_DUPLICATE_VERTICES | TriMeshFlags::FIX_INTERNAL_EDGES,
                )
                .or_else(|_| ColliderBuilder::trimesh(scaled.clone(), tris))
                .map_err(|e| EngineError::InvalidValue(format!("body mesh: {e}")))?;
                (b, probe_points(&scaled))
            }
            BodyShape::BoundingBox => {
                let h = ((max - min) * 0.5).max(Vec3::splat(0.01));
                (
                    ColliderBuilder::cuboid(h.x, h.y, h.z).translation((min + max) * 0.5),
                    box_points(min, max),
                )
            }
            BodyShape::Box(h) => {
                let h = *h * scale;
                (ColliderBuilder::cuboid(h.x, h.y, h.z), box_points(-h, h))
            }
            BodyShape::Ball(r) => {
                let r = r * scale.max_element();
                let mut pts = box_points(Vec3::splat(-r * 0.7), Vec3::splat(r * 0.7));
                pts.extend([Vec3::Y * r, -Vec3::Y * r]);
                (ColliderBuilder::ball(r), pts)
            }
            BodyShape::Capsule {
                half_height,
                radius,
            } => {
                let r = radius * scale.x.max(scale.z);
                let hh = half_height * scale.y;
                let mut pts = box_points(Vec3::new(-r, -hh, -r), Vec3::new(r, hh, r));
                pts.extend([Vec3::Y * (hh + r), -Vec3::Y * (hh + r)]);
                (ColliderBuilder::capsule_y(hh, r), pts)
            }
        };
        let rb = match body.kind {
            BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            BodyKind::Fixed => RigidBodyBuilder::fixed(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        }
        .pose(pose(translation, rotation))
        .linvel(body.velocity)
        .angvel(body.spin)
        .linear_damping(body.damping)
        .angular_damping(body.damping * 2.0 + 0.05)
        .ccd_enabled(body.ccd)
        .user_data(tag(id))
        .build();
        let handle = self.world.insert_body(rb);
        let collider = self.world.insert_collider(
            builder
                .density(body.density * WATER_KG_M3)
                .friction(body.friction)
                .restitution(body.bounce)
                .user_data(tag(id))
                .collision_groups(body_groups(body.layer, body.collides_with))
                .active_events(rapier3d::prelude::ActiveEvents::COLLISION_EVENTS),
            Some(handle),
        );
        let volume = self.world.colliders[collider].volume();
        self.bodies.insert(
            id,
            Tracked {
                handle,
                kind: body.kind,
                scale,
                prev: (translation, rotation),
                cur: (translation, rotation),
                written: transform,
                probes,
                volume,
                density: body.density,
                wet: false,
                age: 0.0,
                temporary,
            },
        );
        Ok(())
    }

    pub fn remove(&mut self, id: EntityId) {
        if let Some(t) = self.bodies.remove(&id) {
            self.world.remove_body_with_colliders(t.handle, true);
        }
        self.sinking.retain(|s| s.id != id);
    }

    fn body_mut(&mut self, id: EntityId) -> EngineResult<&mut rapier3d::prelude::RigidBody> {
        let t = self.bodies.get(&id).ok_or(EngineError::UnknownEntity)?;
        self.world
            .bodies
            .get_mut(t.handle)
            .ok_or(EngineError::UnknownEntity)
    }

    pub fn velocity(&self, id: EntityId) -> Option<Vec3> {
        let t = self.bodies.get(&id)?;
        self.world.bodies.get(t.handle).map(|b| b.linvel())
    }

    pub fn spin(&self, id: EntityId) -> Option<Vec3> {
        let t = self.bodies.get(&id)?;
        self.world.bodies.get(t.handle).map(|b| b.angvel())
    }

    pub fn mass(&self, id: EntityId) -> Option<f32> {
        let t = self.bodies.get(&id)?;
        self.world.bodies.get(t.handle).map(|b| b.mass())
    }

    pub fn set_velocity(&mut self, id: EntityId, v: Vec3) -> EngineResult<()> {
        crate::place::ensure_finite3(v, "velocity")?;
        self.body_mut(id)?.set_linvel(v, true);
        Ok(())
    }

    pub fn set_spin(&mut self, id: EntityId, w: Vec3) -> EngineResult<()> {
        crate::place::ensure_finite3(w, "spin")?;
        self.body_mut(id)?.set_angvel(w, true);
        Ok(())
    }

    pub fn push(&mut self, id: EntityId, impulse: Vec3, at: Option<Vec3>) -> EngineResult<()> {
        crate::place::ensure_finite3(impulse, "impulse")?;
        let b = self.body_mut(id)?;
        match at {
            Some(p) => {
                crate::place::ensure_finite3(p, "impulse point")?;
                b.apply_impulse_at_point(impulse, p, true);
            }
            None => b.apply_impulse(impulse, true),
        }
        Ok(())
    }

    /// Push every dynamic body near `center` away from it, strongest at the
    /// centre and fading to nothing at `radius`. `strength` is the speed in
    /// m/s given to a body right at the centre.
    pub fn blast(&mut self, center: Vec3, radius: f32, strength: f32) {
        let ragdoll_parts = self.ragdoll_handles();
        let handles: Vec<RigidBodyHandle> = self
            .bodies
            .values()
            .filter(|t| t.kind == BodyKind::Dynamic)
            .map(|t| t.handle)
            .chain(ragdoll_parts)
            .collect();
        for handle in handles {
            let Some(b) = self.world.bodies.get_mut(handle) else {
                continue;
            };
            let to = b.translation() - center;
            let d = to.length();
            if d >= radius {
                continue;
            }
            let dir = to.try_normalize().unwrap_or(Vec3::Y);
            let falloff = 1.0 - d / radius;
            b.apply_impulse(dir * strength * falloff * b.mass(), true);
        }
    }

    /// Bodies hit since the last call, hardest first.
    /// Impacts not yet taken, without taking them.
    pub(crate) fn pending_impacts(&self) -> &[Impact] {
        &self.impacts
    }

    pub fn take_impacts(&mut self) -> Vec<Impact> {
        let mut out = std::mem::take(&mut self.impacts);
        out.sort_by(|a, b| b.speed.total_cmp(&a.speed));
        out
    }

    /// First body or ground along a ray. Ground means the colliders physics
    /// has built so far; the world combines this with terrain rays.
    pub fn cast_ray(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<RayHit> {
        let dir = dir.try_normalize()?;
        let ray = Ray::new(origin, dir);
        let skip_ground =
            |_: ColliderHandle, c: &rapier3d::prelude::Collider| c.user_data != GROUND_TAG;
        let filter = QueryFilter::default()
            .exclude_sensors()
            .predicate(&skip_ground);
        let (handle, hit) = self
            .world
            .cast_ray_and_get_normal(&ray, max, true, filter)?;
        let c = self.world.colliders.get(handle)?;
        let entity = Some(EntityId::from_raw(c.user_data as u64))
            .filter(|id| self.bodies.contains_key(id) || self.ragdolls.contains_key(id));
        Some(RayHit {
            entity,
            point: origin + dir * hit.time_of_impact,
            normal: hit.normal,
            distance: hit.time_of_impact,
        })
    }

    /// Bring ray queries up to date with bodies added or moved since the
    /// last step (navigation reads them before the first step).
    pub(crate) fn refresh_queries(&mut self) {
        self.world.detect_collisions(&(), &());
    }

    /// When `p` is inside fixed bodies: the lowest underside straight below
    /// it (`lo - 1` when one reaches below `lo`). `None` when `p` is free.
    pub fn fixed_underside(&self, p: Vec3, lo: f32) -> Option<f32> {
        let fixed_only = |_: ColliderHandle, c: &rapier3d::prelude::Collider| {
            c.user_data != GROUND_TAG
                && self
                    .bodies
                    .get(&EntityId::from_raw(c.user_data as u64))
                    .is_some_and(|t| t.kind == BodyKind::Fixed)
        };
        let filter = QueryFilter::default()
            .exclude_sensors()
            .predicate(&fixed_only);
        let start = lo - 1.0;
        let ray = Ray::new(Vec3::new(p.x, start, p.z), Vec3::Y);
        self.world
            .intersect_point(p, filter)
            .map(|(_, c)| {
                c.shape()
                    .cast_ray(c.position(), &ray, p.y - start, true)
                    .map_or(start, |t| start + t)
            })
            .reduce(f32::min)
    }

    /// A ray against fixed bodies only (walls, floors, buildings): what
    /// navigation treats as ground. `solid = false` lets a ray that starts
    /// inside a shape find its far side.
    pub fn cast_ray_fixed(&self, origin: Vec3, dir: Vec3, max: f32, solid: bool) -> Option<RayHit> {
        let dir = dir.try_normalize()?;
        let ray = Ray::new(origin, dir);
        let fixed_only = |_: ColliderHandle, c: &rapier3d::prelude::Collider| {
            c.user_data != GROUND_TAG
                && self
                    .bodies
                    .get(&EntityId::from_raw(c.user_data as u64))
                    .is_some_and(|t| t.kind == BodyKind::Fixed)
        };
        let filter = QueryFilter::default()
            .exclude_sensors()
            .predicate(&fixed_only);
        let (handle, hit) = self
            .world
            .cast_ray_and_get_normal(&ray, max, solid, filter)?;
        let c = self.world.colliders.get(handle)?;
        // A triangle mesh reports every face as facing the ray, and marks a
        // hit on a face's back as `Face(1)`. Turn those round, so the inside
        // of a closed mesh (a wall's underside seen from within) is not
        // taken for a floor.
        let mut normal = hit.normal;
        if c.shape().as_trimesh().is_some()
            && hit.feature == rapier3d::parry::shape::FeatureId::Face(1)
        {
            normal = -normal;
        }
        Some(RayHit {
            entity: Some(EntityId::from_raw(c.user_data as u64)),
            point: origin + dir * hit.time_of_impact,
            normal,
            distance: hit.time_of_impact,
        })
    }

    /// Advance by `dt`. `transform_of` reads an entity's current transform
    /// (to notice when the game moved it). Returns new transforms to write,
    /// rubble to despawn, and splashes.
    pub fn tick(
        &mut self,
        dt: f32,
        around: &Surroundings,
        mut transform_of: impl FnMut(EntityId) -> Option<Mat4>,
    ) -> TickResult {
        let mut out = TickResult::default();
        if !dt.is_finite() || dt <= 0.0 {
            return out;
        }
        // Forget bodies whose entities are gone; follow ones the game moved.
        let mut ids: Vec<EntityId> = self.bodies.keys().copied().collect();
        ids.sort_by_key(|id| id.raw());
        for id in ids {
            let Some(m) = transform_of(id) else {
                self.remove(id);
                continue;
            };
            let t = &self.bodies[&id];
            if m == t.written {
                continue;
            }
            let (_, r, p) = split(m);
            let (handle, kind) = (t.handle, t.kind);
            if let Some(b) = self.world.bodies.get_mut(handle) {
                match kind {
                    BodyKind::Kinematic => b.set_next_kinematic_position(pose(p, r)),
                    _ => b.set_position(pose(p, r), true),
                }
            }
            let t = self.bodies.get_mut(&id).expect("tracked");
            t.written = m;
            if kind != BodyKind::Kinematic {
                t.prev = (p, r);
                t.cur = (p, r);
            }
        }
        self.sinking.retain_mut(|s| {
            s.left -= dt;
            if s.left <= 0.0 {
                out.expired.push(s.id);
                return false;
            }
            out.sink.push((s.id, dt * 0.6));
            true
        });
        if self.bodies.is_empty() && self.ragdolls.is_empty() && self.characters.is_empty() {
            self.accumulator = 0.0;
            // Nothing can be inside a trigger.
            self.update_triggers();
            return out;
        }

        let h = 1.0 / self.settings.steps_per_second;
        self.accumulator += dt.min(0.25);
        let mut steps = 0;
        while self.accumulator >= h && steps < self.settings.max_steps_per_frame {
            self.accumulator -= h;
            steps += 1;
            self.clock += h;
            self.update_ground(around);
            self.apply_water(around, &mut out.splashes);
            self.step_characters(h);
            for t in self.bodies.values_mut() {
                t.prev = t.cur;
            }
            for r in self.ragdolls_mut() {
                r.prev_cur();
            }
            let before: HashMap<RigidBodyHandle, Vec3> = self
                .world
                .active_bodies()
                .map(|(h, b)| (h, b.linvel()))
                .collect();
            let events = StartEvents::default();
            self.world.step_with_events(&(), &events);
            self.collect_impacts(events, &before);
            self.update_triggers();
            for t in self.bodies.values_mut() {
                if let Some(b) = self.world.bodies.get(t.handle) {
                    t.cur = (b.translation(), *b.rotation());
                }
            }
            self.read_ragdolls();
        }
        if steps == self.settings.max_steps_per_frame {
            // Fell behind: drop the backlog rather than spiral.
            self.accumulator = self.accumulator.min(h);
        }

        self.drop_broken_joints();

        // Rubble whose time is up starts sinking.
        let mut done = Vec::new();
        for (id, t) in self.bodies.iter_mut() {
            t.age += dt;
            if let Some(tmp) = t.temporary {
                if t.age > tmp.keep {
                    done.push(*id);
                }
            }
        }
        done.sort_by_key(|id| id.raw());
        for id in done {
            self.remove(id);
            self.sinking.push(Sinking {
                id,
                left: SINK_SECONDS,
            });
        }

        // Draw between the last two steps.
        let alpha = (self.accumulator / h).clamp(0.0, 1.0);
        self.alpha = alpha;
        out.characters = self.character_places(alpha);
        for (id, t) in self.bodies.iter_mut() {
            if t.kind == BodyKind::Fixed {
                continue;
            }
            let p = t.prev.0.lerp(t.cur.0, alpha);
            let r = t.prev.1.slerp(t.cur.1, alpha);
            let m = Mat4::from_scale_rotation_translation(t.scale, r, p);
            if m != t.written {
                t.written = m;
                out.moved.push((*id, m));
            }
        }
        out
    }

    /// Draw every moving body at its last step (no interpolation).
    fn show_current(&mut self) -> Vec<(EntityId, Mat4)> {
        let mut out = Vec::new();
        for (id, t) in self.bodies.iter_mut() {
            if t.kind == BodyKind::Fixed {
                continue;
            }
            let m = Mat4::from_scale_rotation_translation(t.scale, t.cur.1, t.cur.0);
            t.written = m;
            out.push((*id, m));
        }
        out
    }

    fn collect_impacts(&mut self, events: StartEvents, before: &HashMap<RigidBodyHandle, Vec3>) {
        // Contacts that began this step, with the speed things had just
        // before they met. Resting contacts don't start again, so piles stay quiet.
        let pairs = events.pairs.into_inner().unwrap_or_default();
        for (h1, h2) in pairs {
            let (Some(c1), Some(c2)) = (self.world.colliders.get(h1), self.world.colliders.get(h2))
            else {
                continue;
            };
            let id_of = |c: &rapier3d::prelude::Collider| {
                (c.user_data != GROUND_TAG)
                    .then(|| EntityId::from_raw(c.user_data as u64))
                    .filter(|id| self.bodies.contains_key(id))
            };
            let (a, b) = match (id_of(c1), id_of(c2)) {
                (Some(a), b) => (a, b),
                (None, Some(b)) => (b, None),
                _ => continue,
            };
            let Some(pair) = self.world.contact_pair(h1, h2) else {
                continue;
            };
            let Some((manifold, point)) = pair.find_deepest_contact() else {
                continue;
            };
            let normal = manifold.data.normal;
            let vel = |c: &rapier3d::prelude::Collider| {
                c.parent()
                    .and_then(|h| before.get(&h).copied())
                    .unwrap_or(Vec3::ZERO)
            };
            let at = c1.position() * point.local_p1;
            let speed = (vel(c1) - vel(c2)).dot(normal).abs();
            if speed > 1.0 {
                self.impacts.push(Impact {
                    a,
                    b,
                    point: at,
                    speed,
                });
            }
        }
        if self.impacts.len() > 256 {
            self.impacts.sort_by(|a, b| b.speed.total_cmp(&a.speed));
            self.impacts.truncate(128);
        }
    }

    /// Buoyancy, drag and current for bodies in water.
    fn apply_water(&mut self, around: &Surroundings, splashes: &mut Vec<Splash>) {
        let g = -self.settings.gravity.y;
        for t in self.bodies.values_mut() {
            if t.kind != BodyKind::Dynamic {
                continue;
            }
            let Some(b) = self.world.bodies.get_mut(t.handle) else {
                continue;
            };
            b.reset_forces(false);
            b.reset_torques(false);
            if around.water.is_none() || b.is_sleeping() && !t.wet {
                continue;
            }
            let pos = b.translation();
            let Some((surface, current)) = around.water_at(pos) else {
                t.wet = false;
                continue;
            };
            let rot = *b.rotation();
            // Share of the body's height under the surface, pushing up at the
            // middle of the submerged part (so a tilted body rights itself).
            let points: Vec<Vec3> = t.probes.iter().map(|&l| pos + rot * l).collect();
            let (lo, hi) = points
                .iter()
                .fold((pos.y, pos.y), |(lo, hi), p| (lo.min(p.y), hi.max(p.y)));
            let frac = ((surface - lo) / (hi - lo).max(1e-3)).clamp(0.0, 1.0);
            if frac <= 0.0 {
                t.wet = false;
                continue;
            }
            let under: Vec<Vec3> = points.iter().copied().filter(|p| p.y < surface).collect();
            let mut centre = if under.is_empty() {
                pos
            } else {
                (under.iter().copied().sum::<Vec3>() + pos) / (under.len() + 1) as f32
            };
            centre.y = (lo + surface.min(hi)) * 0.5;
            b.add_force_at_point(Vec3::Y * WATER_KG_M3 * g * t.volume * frac, centre, true);
            let v = b.linvel();
            if !t.wet && v.y < -1.5 {
                splashes.push(Splash {
                    at: pos,
                    speed: -v.y,
                });
            }
            t.wet = true;
            let rel = v - Vec3::new(current.x, 0.0, current.y);
            let mass = b.mass();
            b.add_force(-rel * mass * 3.0 * frac, true);
            let w = b.angvel();
            let inertia = mass * t.volume.cbrt().powi(2) * 0.2;
            b.add_torque(-w * inertia * 2.0 * frac, true);
            let _ = t.density;
        }
    }

    /// Make sure every dynamic body has ground colliders around it.
    fn update_ground(&mut self, around: &Surroundings) {
        let source = around.source(self.settings.ground);
        if source != self.source {
            self.clear_ground();
            self.source = source;
            if let GroundSource::Plane(y) = source {
                // A thick slab rather than a half-space: same contacts, and
                // fast things can't tunnel through it.
                let c = ColliderBuilder::cuboid(1.0e4, 5.0, 1.0e4)
                    .translation(Vec3::new(0.0, y - 5.0, 0.0))
                    .friction(0.7)
                    .user_data(GROUND_TAG)
                    .collision_groups(ground_groups());
                self.plane = Some(self.world.insert_collider(c, None));
            }
            if let (GroundSource::Voxels { .. }, Some(v)) = (source, around.voxels) {
                self.voxel_rev = v.volume().rev();
            }
            self.world.wake_up_all(false);
        }
        match source {
            GroundSource::Voxels { voxel_size, .. } => {
                let v = around.voxels.expect("voxels").volume();
                // Rebuild tiles whose voxels changed and wake what rests there.
                let (changed, removed) = if v.rev() == self.voxel_rev {
                    (Vec::new(), false)
                } else {
                    v.changed_since(self.voxel_rev)
                };
                if !changed.is_empty() || removed {
                    self.voxel_rev = v.rev();
                    let chunk_m = v.chunk_world_size();
                    // Tiles reach one cell into their neighbours, so an edit
                    // can change the tiles around a changed chunk too.
                    let mut stale: Vec<IVec3> = changed
                        .into_iter()
                        .flat_map(|k| {
                            (-1..=1).flat_map(move |x| {
                                (-1..=1).flat_map(move |y| {
                                    (-1..=1).map(move |z| k + IVec3::new(x, y, z))
                                })
                            })
                        })
                        .filter(|k| self.tiles.contains_key(k))
                        .collect();
                    stale.sort_by_key(|k| (k.x, k.y, k.z));
                    stale.dedup();
                    if removed {
                        stale.extend(self.tiles.keys().filter(|k| !v.has_chunk(**k)));
                    }
                    for key in &stale {
                        if let Some(tile) = self.tiles.remove(key) {
                            if let Some(c) = tile.collider {
                                self.world.remove_collider(c);
                            }
                        }
                        let lo = key.as_vec3() * chunk_m - Vec3::splat(1.0);
                        let hi = lo + Vec3::splat(chunk_m + 2.0);
                        self.wake_in(lo, hi);
                    }
                }
                let chunk_m = v.chunk_world_size();
                let needed = self.needed_tiles(chunk_m, true);
                let missing = missing_sorted(&needed, &self.tiles);
                let built: Vec<(IVec3, Option<ColliderBuilder>)> = missing
                    .par_iter()
                    .map(|k| {
                        let (verts, tris) = v.collision_triangles(*k);
                        (*k, ground_mesh(verts, tris))
                    })
                    .collect();
                let _ = voxel_size;
                for (key, builder) in built {
                    let collider = builder.map(|b| self.world.insert_collider(b, None));
                    self.tiles.insert(
                        key,
                        Tile {
                            collider,
                            last_needed: self.clock,
                        },
                    );
                }
                self.touch_and_drop(&needed);
            }
            GroundSource::Heights { .. } => {
                let (_, field) = around.heights.expect("heights");
                let needed = self.needed_tiles(HEIGHT_TILE_M, false);
                let missing = missing_sorted(&needed, &self.tiles);
                let built: Vec<(IVec3, Option<ColliderBuilder>)> = missing
                    .par_iter()
                    .map(|k| {
                        let (verts, tris) = height_tile(field, *k);
                        (*k, ground_mesh(verts, tris))
                    })
                    .collect();
                for (key, builder) in built {
                    let collider = builder.map(|b| self.world.insert_collider(b, None));
                    self.tiles.insert(
                        key,
                        Tile {
                            collider,
                            last_needed: self.clock,
                        },
                    );
                }
                self.touch_and_drop(&needed);
            }
            GroundSource::Plane(_) | GroundSource::None => {}
        }
    }

    fn touch_and_drop(&mut self, needed: &HashSet<IVec3>) {
        let now = self.clock;
        let mut drop = Vec::new();
        for (k, tile) in self.tiles.iter_mut() {
            if needed.contains(k) {
                tile.last_needed = now;
            } else if now - tile.last_needed > TILE_KEEP_S {
                drop.push(*k);
            }
        }
        for k in drop {
            if let Some(tile) = self.tiles.remove(&k) {
                if let Some(c) = tile.collider {
                    self.world.remove_collider(c);
                }
            }
        }
    }

    /// Tile keys overlapping each dynamic body's reach for the next step.
    fn needed_tiles(&self, tile_m: f32, three_d: bool) -> HashSet<IVec3> {
        let h = 1.0 / self.settings.steps_per_second;
        let mut out = HashSet::new();
        let ragdoll_parts = self.ragdoll_handles();
        let characters = self.character_handles();
        let handles = self
            .bodies
            .values()
            .filter(|t| t.kind == BodyKind::Dynamic)
            .map(|t| t.handle)
            .chain(ragdoll_parts)
            .chain(characters);
        for handle in handles {
            let Some(b) = self.world.bodies.get(handle) else {
                continue;
            };
            let Some(c) = b
                .colliders()
                .first()
                .and_then(|c| self.world.colliders.get(*c))
            else {
                continue;
            };
            let aabb = c.compute_aabb();
            let reach = Vec3::splat(0.5) + b.linvel().abs() * h * 2.0;
            let lo = ((aabb.mins - reach) / tile_m).floor().as_ivec3();
            let hi = ((aabb.maxs + reach) / tile_m).floor().as_ivec3();
            if (hi - lo).max_element() > 8 {
                continue; // a giant body; don't build half the world
            }
            for x in lo.x..=hi.x {
                for z in lo.z..=hi.z {
                    if three_d {
                        for y in lo.y..=hi.y {
                            out.insert(IVec3::new(x, y, z));
                        }
                    } else {
                        out.insert(IVec3::new(x, 0, z));
                    }
                }
            }
        }
        out
    }

    /// A fixed box collider (a building element): `center`, `rotation` and
    /// `half` extents in world space. Bodies collide with it like ground.
    pub fn add_static_box(
        &mut self,
        owner: u64,
        part: u32,
        center: Vec3,
        rotation: Quat,
        half: Vec3,
    ) {
        self.remove_static(owner, part);
        let c = ColliderBuilder::cuboid(half.x.max(0.01), half.y.max(0.01), half.z.max(0.01))
            .position(pose(center, rotation))
            .friction(0.7)
            .user_data(GROUND_TAG)
            .collision_groups(ground_groups());
        let h = self.world.insert_collider(c, None);
        self.statics.insert((owner, part), h);
    }

    /// Take a fixed box away and wake whatever rested on it.
    pub fn remove_static(&mut self, owner: u64, part: u32) {
        if let Some(h) = self.statics.remove(&(owner, part)) {
            if let Some(c) = self.world.colliders.get(h) {
                let aabb = c.compute_aabb();
                let (lo, hi) = (aabb.mins - Vec3::splat(0.5), aabb.maxs + Vec3::splat(0.5));
                self.world.remove_collider(h);
                self.wake_in(lo, hi);
            }
        }
    }

    /// Take every fixed box of `owner` away.
    pub fn remove_statics_of(&mut self, owner: u64) {
        let mut parts: Vec<u32> = self
            .statics
            .keys()
            .filter(|k| k.0 == owner)
            .map(|k| k.1)
            .collect();
        parts.sort_unstable();
        for p in parts {
            self.remove_static(owner, p);
        }
    }

    /// Is this dynamic body resting (asleep)?
    pub fn is_sleeping(&self, id: EntityId) -> bool {
        self.bodies
            .get(&id)
            .and_then(|t| self.world.bodies.get(t.handle))
            .is_some_and(|b| b.is_sleeping())
    }

    fn wake_in(&mut self, lo: Vec3, hi: Vec3) {
        for t in self.bodies.values() {
            if let Some(b) = self.world.bodies.get_mut(t.handle) {
                let p = b.translation();
                if p.cmpge(lo).all() && p.cmple(hi).all() {
                    b.wake_up(true);
                }
            }
        }
    }

    fn clear_ground(&mut self) {
        for (_, tile) in self.tiles.drain() {
            if let Some(c) = tile.collider {
                self.world.remove_collider(c);
            }
        }
        if let Some(c) = self.plane.take() {
            self.world.remove_collider(c);
        }
    }
}

/// Collects which collider pairs started touching during a step.
#[derive(Default)]
struct StartEvents {
    pairs: std::sync::Mutex<Vec<(ColliderHandle, ColliderHandle)>>,
}

impl rapier3d::prelude::EventHandler for StartEvents {
    fn handle_collision_event(
        &self,
        _: &rapier3d::prelude::RigidBodySet,
        _: &rapier3d::prelude::ColliderSet,
        event: rapier3d::prelude::CollisionEvent,
        _: Option<&rapier3d::prelude::ContactPair>,
    ) {
        if let rapier3d::prelude::CollisionEvent::Started(a, b, _) = event {
            if let Ok(mut v) = self.pairs.lock() {
                v.push((a, b));
            }
        }
    }

    fn handle_contact_force_event(
        &self,
        _: f32,
        _: &rapier3d::prelude::RigidBodySet,
        _: &rapier3d::prelude::ColliderSet,
        _: &rapier3d::prelude::ContactPair,
        _: f32,
    ) {
    }

    fn handle_soft_body_tear_event(
        &self,
        _: &rapier3d::prelude::SoftBodySet,
        _: &rapier3d::prelude::SoftBodyTearEvent,
    ) {
    }
}

/// Per-tick output for the world to apply.
#[derive(Default, Debug)]
pub(crate) struct TickResult {
    pub moved: Vec<(EntityId, Mat4)>,
    pub sink: Vec<(EntityId, f32)>,
    pub expired: Vec<EntityId>,
    pub splashes: Vec<Splash>,
    /// Characters' feet positions to write.
    pub characters: Vec<(EntityId, Vec3)>,
}

/// A ground collider from triangles; `None` when there are none.
fn ground_mesh(verts: Vec<Vec3>, tris: Vec<[u32; 3]>) -> Option<ColliderBuilder> {
    if tris.is_empty() {
        return None;
    }
    let b = ColliderBuilder::trimesh_with_flags(
        verts.clone(),
        tris.clone(),
        TriMeshFlags::FIX_INTERNAL_EDGES,
    )
    .or_else(|_| ColliderBuilder::trimesh(verts, tris))
    .ok()?;
    Some(
        b.friction(0.7)
            .user_data(GROUND_TAG)
            .collision_groups(ground_groups()),
    )
}

/// Tiles still to build, in a fixed order: colliders inserted in the same
/// order give the same simulation on every machine and in every run.
fn missing_sorted(needed: &HashSet<IVec3>, tiles: &HashMap<IVec3, Tile>) -> Vec<IVec3> {
    let mut out: Vec<IVec3> = needed
        .iter()
        .filter(|k| !tiles.contains_key(k))
        .copied()
        .collect();
    out.sort_by_key(|k| (k.x, k.y, k.z));
    out
}

/// Triangles of one height-terrain tile.
fn height_tile(field: HeightFn, key: IVec3) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    // One extra step on every side: neighbouring tiles overlap, so nothing
    // rests on a tile's open border edge (see `Volume::collision_triangles`).
    let n = HEIGHT_TILE_SAMPLES + 2;
    let step = HEIGHT_TILE_M / (HEIGHT_TILE_SAMPLES - 1) as f32;
    let x0 = key.x as f32 * HEIGHT_TILE_M - step;
    let z0 = key.z as f32 * HEIGHT_TILE_M - step;
    let mut verts = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let x = x0 + i as f32 * step;
            let z = z0 + j as f32 * step;
            verts.push(Vec3::new(x, field(x, z), z));
        }
    }
    let mut tris = Vec::with_capacity((n - 1) * (n - 1) * 2);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            tris.push([a, c, b]);
            tris.push([b, c, d]);
        }
    }
    (verts, tris)
}

/// A body in a [`PhysicsWorld`]. The game owns it: store it, send it over the
/// network as [`BodyId::raw`], turn it back with [`BodyId::from_raw`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId(u64);

impl BodyId {
    pub fn raw(self) -> u64 {
        self.0
    }

    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    fn key(self) -> EntityId {
        EntityId::from_raw(self.0)
    }
}

/// The colliders of a building in a [`PhysicsWorld`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BuildingColliders(u64);

/// What a ray hit in a [`PhysicsWorld`]: a body, or the ground (`body` is `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHit {
    pub body: Option<BodyId>,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

/// Two bodies hitting each other in a [`PhysicsWorld`]; `b` is `None` for the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyImpact {
    pub a: BodyId,
    pub b: Option<BodyId>,
    pub point: Vec3,
    /// Closing speed along the contact normal, m/s.
    pub speed: f32,
}

type SharedHeight = std::sync::Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>;

/// Physics on its own: no window, no GPU, no [`World`](crate::world::World).
/// For a game server, tools and tests. Bodies, shapes, ground and stepping are
/// the same code `World` uses, so a server and its clients agree on every shape.
///
/// ```
/// use engine::physics::{Body, BodyShape, PhysicsWorld};
/// use engine::glam::{Mat4, Vec3};
///
/// let mut physics = PhysicsWorld::new();
/// let ball = physics
///     .add(&Body::dynamic().shape(BodyShape::Ball(0.5)), Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0)))
///     .unwrap();
/// for _ in 0..120 {
///     physics.step(1.0 / 60.0);
/// }
/// assert!((physics.position(ball).unwrap().y - 0.5).abs() < 0.05);
/// ```
pub struct PhysicsWorld {
    inner: Physics,
    /// The transform of every body as the game last saw it (interpolated).
    transforms: HashMap<EntityId, Mat4>,
    next: u64,
    heights: Option<SharedHeight>,
    voxels: Option<crate::voxel_terrain::VoxelTerrain>,
    expired: Vec<BodyId>,
}

impl std::fmt::Debug for PhysicsWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsWorld")
            .field("bodies", &self.inner.body_count())
            .field("height_terrain", &self.heights.is_some())
            .field("voxel_terrain", &self.voxels.is_some())
            .finish()
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    /// Earth gravity, 60 steps a second, a flat floor at y = 0.
    pub fn new() -> Self {
        Self {
            inner: Physics::new(),
            transforms: HashMap::new(),
            next: 1,
            heights: None,
            voxels: None,
            expired: Vec::new(),
        }
    }

    pub fn with_settings(settings: PhysicsSettings) -> EngineResult<Self> {
        let mut w = Self::new();
        w.set_settings(settings)?;
        Ok(w)
    }

    pub fn settings(&self) -> PhysicsSettings {
        self.inner.settings()
    }

    pub fn set_settings(&mut self, settings: PhysicsSettings) -> EngineResult<()> {
        settings.validate()?;
        self.inner.set_settings(settings);
        Ok(())
    }

    /// Height terrain as the ground (with [`PhysicsGround::Auto`]): the same
    /// [`HeightField`](crate::proc_terrain::HeightField) the client draws.
    pub fn set_height_terrain(&mut self, field: crate::proc_terrain::HeightField) {
        self.set_height_fn(move |x, z| field.height_at(x, z));
    }

    /// Any height function as the ground: `height(x, z)` in metres.
    pub fn set_height_fn(&mut self, height: impl Fn(f32, f32) -> f32 + Send + Sync + 'static) {
        self.heights = Some(std::sync::Arc::new(height));
    }

    /// Voxel terrain as the ground; it wins over height terrain. Dig it with
    /// [`Self::voxel_terrain_mut`] and colliders near bodies are rebuilt.
    pub fn set_voxel_terrain(&mut self, terrain: crate::voxel_terrain::VoxelTerrain) {
        self.voxels = Some(terrain);
    }

    pub fn voxel_terrain(&self) -> Option<&crate::voxel_terrain::VoxelTerrain> {
        self.voxels.as_ref()
    }

    pub fn voxel_terrain_mut(&mut self) -> Option<&mut crate::voxel_terrain::VoxelTerrain> {
        self.voxels.as_mut()
    }

    /// Drop both terrains: the ground is a floor at y = 0 again.
    pub fn clear_terrain(&mut self) {
        self.heights = None;
        self.voxels = None;
    }

    fn new_id(&mut self) -> BodyId {
        let id = BodyId(self.next);
        self.next += 1;
        id
    }

    /// A body with a shape that needs no mesh: [`BodyShape::Box`],
    /// [`BodyShape::Ball`] or [`BodyShape::Capsule`]. `transform` places it
    /// (its scale scales the shape).
    pub fn add(&mut self, body: &Body, transform: Mat4) -> EngineResult<BodyId> {
        if !matches!(
            body.shape,
            BodyShape::Box(_) | BodyShape::Ball(_) | BodyShape::Capsule { .. }
        ) {
            return Err(EngineError::InvalidValue(
                "this shape needs a mesh: use add_mesh, or a Box, Ball or Capsule shape".into(),
            ));
        }
        self.add_points(body, transform, &[], &[])
    }

    /// A body shaped by a mesh (its hull, triangles or bounds, as
    /// `body.shape` says), exactly as `World::add_body` builds it.
    pub fn add_mesh(
        &mut self,
        body: &Body,
        transform: Mat4,
        mesh: &crate::mesh::Mesh,
    ) -> EngineResult<BodyId> {
        let built = mesh.build();
        self.add_points(body, transform, &built.positions, &built.indices)
    }

    /// A body from raw local points (and triangle indices for
    /// [`BodyShape::Mesh`]), for shapes that come from somewhere else.
    pub fn add_points(
        &mut self,
        body: &Body,
        transform: Mat4,
        points: &[Vec3],
        indices: &[u32],
    ) -> EngineResult<BodyId> {
        let id = self.new_id();
        self.insert(id, body, transform, points, indices, None)?;
        Ok(id)
    }

    /// Like [`Self::add_mesh`] for short-lived rubble: after `keep_seconds`
    /// the body sinks into the ground, is removed and shows up in
    /// [`Self::take_expired`].
    pub fn add_temporary(
        &mut self,
        body: &Body,
        transform: Mat4,
        mesh: &crate::mesh::Mesh,
        keep_seconds: f32,
    ) -> EngineResult<BodyId> {
        let built = mesh.build();
        let id = self.new_id();
        self.insert(
            id,
            body,
            transform,
            &built.positions,
            &built.indices,
            Some(Temporary { keep: keep_seconds }),
        )?;
        Ok(id)
    }

    fn insert(
        &mut self,
        id: BodyId,
        body: &Body,
        transform: Mat4,
        points: &[Vec3],
        indices: &[u32],
        temporary: Option<Temporary>,
    ) -> EngineResult<()> {
        if !transform.is_finite() {
            return Err(EngineError::InvalidValue(
                "body transform must be finite".into(),
            ));
        }
        self.inner
            .add(id.key(), points, transform, body, temporary, || {
                indices
                    .chunks_exact(3)
                    .map(|t| [t[0], t[1], t[2]])
                    .collect()
            })?;
        self.transforms.insert(id.key(), transform);
        Ok(())
    }

    pub fn remove(&mut self, id: BodyId) {
        self.inner.remove(id.key());
        self.transforms.remove(&id.key());
    }

    pub fn contains(&self, id: BodyId) -> bool {
        self.inner.contains(id.key())
    }

    /// Bodies, and how many of them are moving (not asleep).
    pub fn body_count(&self) -> (usize, usize) {
        (self.inner.body_count(), self.inner.awake_count())
    }

    /// Where a body is drawn: between its last two steps, so it moves
    /// smoothly when read every frame. Includes the body's scale.
    pub fn transform(&self, id: BodyId) -> Option<Mat4> {
        self.transforms.get(&id.key()).copied()
    }

    pub fn position(&self, id: BodyId) -> Option<Vec3> {
        self.transform(id).map(|m| m.w_axis.truncate())
    }

    /// Position and rotation after the last whole step (no interpolation):
    /// the authoritative state to send to clients.
    pub fn pose(&self, id: BodyId) -> Option<(Vec3, Quat)> {
        self.inner.bodies.get(&id.key()).map(|t| t.cur)
    }

    /// Teleport a body (and its scale) to `transform`. Kinematic bodies move
    /// there over the next step, pushing what's in the way.
    pub fn set_transform(&mut self, id: BodyId, transform: Mat4) -> EngineResult<()> {
        if !self.contains(id) {
            return Err(EngineError::UnknownEntity);
        }
        if !transform.is_finite() {
            return Err(EngineError::InvalidValue(
                "body transform must be finite".into(),
            ));
        }
        self.transforms.insert(id.key(), transform);
        Ok(())
    }

    pub fn velocity(&self, id: BodyId) -> Option<Vec3> {
        self.inner.velocity(id.key())
    }

    pub fn set_velocity(&mut self, id: BodyId, velocity: Vec3) -> EngineResult<()> {
        self.inner.set_velocity(id.key(), velocity)
    }

    pub fn spin(&self, id: BodyId) -> Option<Vec3> {
        self.inner.spin(id.key())
    }

    pub fn set_spin(&mut self, id: BodyId, spin: Vec3) -> EngineResult<()> {
        self.inner.set_spin(id.key(), spin)
    }

    pub fn mass(&self, id: BodyId) -> Option<f32> {
        self.inner.mass(id.key())
    }

    /// A kick through the centre, in kg·m/s.
    pub fn push(&mut self, id: BodyId, impulse: Vec3) -> EngineResult<()> {
        self.inner.push(id.key(), impulse, None)
    }

    /// A kick at a world point, so the body also spins.
    pub fn push_at(&mut self, id: BodyId, impulse: Vec3, point: Vec3) -> EngineResult<()> {
        self.inner.push(id.key(), impulse, Some(point))
    }

    /// Throw every dynamic body within `radius` outward: `speed` m/s at the
    /// centre, fading to nothing at the edge.
    pub fn blast(&mut self, center: Vec3, radius: f32, speed: f32) {
        if center.is_finite() && radius.is_finite() && radius > 0.0 && speed.is_finite() {
            self.inner.blast(center, radius, speed);
        }
    }

    /// Hard hits since the last call, hardest first.
    pub fn take_impacts(&mut self) -> Vec<BodyImpact> {
        self.inner
            .take_impacts()
            .into_iter()
            .map(|i| BodyImpact {
                a: BodyId(i.a.raw()),
                b: i.b.map(|b| BodyId(b.raw())),
                point: i.point,
                speed: i.speed,
            })
            .collect()
    }

    /// Temporary bodies whose time ran out since the last call.
    pub fn take_expired(&mut self) -> Vec<BodyId> {
        std::mem::take(&mut self.expired)
    }

    /// First body or ground along a ray within `max_distance`.
    pub fn cast_ray(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<BodyHit> {
        let dir = direction.try_normalize()?;
        if !origin.is_finite() || !max_distance.is_finite() || max_distance <= 0.0 {
            return None;
        }
        let body = self
            .inner
            .cast_ray(origin, dir, max_distance)
            .map(|h| BodyHit {
                body: h.entity.map(|e| BodyId(e.raw())),
                point: h.point,
                normal: h.normal,
                distance: h.distance,
            });
        let reach = body.map_or(max_distance, |h| h.distance);
        let ground = self.ground_ray(origin, dir, reach);
        match (body, ground) {
            (Some(b), Some(g)) if g.distance < b.distance => Some(g),
            (Some(b), _) => Some(b),
            (None, g) => g,
        }
    }

    fn ground_ray(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<BodyHit> {
        let hit = |p: Vec3, n: Vec3| BodyHit {
            body: None,
            point: p,
            normal: n,
            distance: (p - origin).length(),
        };
        let plane = |y: f32| {
            (dir.y.abs() > 1e-6)
                .then(|| (y - origin.y) / dir.y)
                .filter(|t| *t >= 0.0 && *t <= max)
                .map(|t| hit(origin + dir * t, Vec3::Y))
        };
        match self.inner.settings().ground {
            PhysicsGround::None => None,
            PhysicsGround::Plane(y) => plane(y),
            PhysicsGround::Auto => {
                if let Some(v) = self.voxels.as_ref() {
                    if v.volume().chunk_keys().next().is_some() {
                        return v.raycast(origin, dir, max).map(|p| hit(p, v.normal_at(p)));
                    }
                }
                if let Some(f) = self.heights.as_ref() {
                    return height_ray(&**f, origin, dir, max).map(|(p, n)| hit(p, n));
                }
                plane(0.0)
            }
        }
    }

    /// Advance by `dt` seconds of real time: as many fixed steps as are due
    /// (see [`PhysicsSettings::steps_per_second`]). Returns how many ran.
    pub fn step(&mut self, dt: f32) -> u32 {
        let heights = self.heights.clone();
        let height_ref = heights.as_deref().map(|h| h as HeightFn);
        let around = Surroundings {
            voxels: self.voxels.as_ref(),
            heights: height_ref.map(|h| {
                (
                    heights
                        .as_ref()
                        .map_or(0, |a| std::sync::Arc::as_ptr(a) as *const u8 as usize),
                    h,
                )
            }),
            water: None,
        };
        let transforms = &self.transforms;
        let before = self.inner.clock;
        let result = self
            .inner
            .tick(dt, &around, |id| transforms.get(&id).copied());
        for (id, m) in result.moved {
            self.transforms.insert(id, m);
        }
        for (id, dy) in result.sink {
            if let Some(m) = self.transforms.get_mut(&id) {
                m.w_axis.y -= dy;
            }
        }
        for id in result.expired {
            self.transforms.remove(&id);
            self.expired.push(BodyId(id.raw()));
        }
        ((self.inner.clock - before) * self.inner.settings.steps_per_second).round() as u32
    }

    /// A building's standing elements as fixed box colliders, placed by
    /// `transform`. Keep the key and call [`Self::update_building`] after
    /// the building changes.
    pub fn add_building(
        &mut self,
        building: &crate::building::Building,
        transform: Mat4,
    ) -> BuildingColliders {
        let key = BuildingColliders(self.next | (1 << 62));
        self.next += 1;
        let (_, rot, pos) = transform.to_scale_rotation_translation();
        for (i, e) in building.plan().elements().iter().enumerate() {
            if building.is_standing(crate::building::ElementId(i as u32)) {
                self.inner
                    .add_static_box(key.0, i as u32, pos + rot * e.center, rot, e.half);
            }
        }
        key
    }

    /// Drop the colliders of elements that broke or fell (and wake what
    /// rested on them).
    pub fn update_building(
        &mut self,
        key: BuildingColliders,
        building: &crate::building::Building,
    ) {
        let gone = building.gone();
        for i in gone.iter_set() {
            self.inner.remove_static(key.0, i as u32);
        }
    }

    pub fn remove_building(&mut self, key: BuildingColliders) {
        self.inner.remove_statics_of(key.0);
    }

    /// Exactly one fixed step, whatever the clock says. For servers that
    /// keep their own tick.
    pub fn step_once(&mut self) {
        let h = 1.0 / self.inner.settings.steps_per_second;
        // Top the accumulator up to one step; tick then runs exactly one.
        self.inner.accumulator = 0.0;
        self.step(h * 1.000_01);
        self.inner.accumulator = 0.0;
        for (id, m) in self.inner.show_current() {
            self.transforms.insert(id, m);
        }
    }
}

/// March a height function along a ray, then refine by bisection.
pub(crate) fn height_ray(
    f: &dyn Fn(f32, f32) -> f32,
    origin: Vec3,
    dir: Vec3,
    max: f32,
) -> Option<(Vec3, Vec3)> {
    let step = 0.5;
    let mut t0 = 0.0;
    let mut t = 0.0;
    while t <= max {
        let p = origin + dir * t;
        if p.y <= f(p.x, p.z) {
            let (mut lo, mut hi) = (t0, t);
            for _ in 0..16 {
                let mid = (lo + hi) * 0.5;
                let q = origin + dir * mid;
                if q.y <= f(q.x, q.z) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let p = origin + dir * hi;
            let e = 0.25;
            let n = Vec3::new(
                f(p.x - e, p.z) - f(p.x + e, p.z),
                2.0 * e,
                f(p.x, p.z - e) - f(p.x, p.z + e),
            )
            .normalize();
            return Some((p, n));
        }
        t0 = t;
        t += step;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> Vec<Vec3> {
        box_points(Vec3::splat(-0.5), Vec3::splat(0.5))
    }

    fn none() -> Surroundings<'static> {
        Surroundings {
            voxels: None,
            heights: None,
            water: None,
        }
    }

    fn run(p: &mut Physics, seconds: f32, transforms: &mut HashMap<EntityId, Mat4>) {
        let frames = (seconds * 120.0) as usize;
        for _ in 0..frames {
            let r = p.tick(1.0 / 120.0, &none(), |id| transforms.get(&id).copied());
            for (id, m) in r.moved {
                transforms.insert(id, m);
            }
        }
    }

    #[test]
    fn a_dropped_box_lands_on_the_floor_and_sleeps() {
        let mut p = Physics::new();
        let id = EntityId::test(1);
        let m = Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0));
        p.add(id, &cube(), m, &Body::dynamic(), None, Vec::new)
            .unwrap();
        let mut tf = HashMap::from([(id, m)]);
        run(&mut p, 4.0, &mut tf);
        let y = tf[&id].w_axis.y;
        assert!((y - 0.5).abs() < 0.03, "rest height {y}");
        assert_eq!(p.awake_count(), 0, "box should be asleep");
    }

    #[test]
    fn boxes_stack_instead_of_passing_through_each_other() {
        let mut p = Physics::new();
        let mut tf = HashMap::new();
        for i in 0..3 {
            let id = EntityId::test(10 + i);
            let m = Mat4::from_translation(Vec3::new(0.0, 0.6 + i as f32 * 1.1, 0.0));
            p.add(id, &cube(), m, &Body::dynamic(), None, Vec::new)
                .unwrap();
            tf.insert(id, m);
        }
        run(&mut p, 4.0, &mut tf);
        let top = tf[&EntityId::test(12)].w_axis.y;
        assert!((top - 2.5).abs() < 0.08, "top of the stack at {top}");
    }

    #[test]
    fn moving_the_entity_teleports_the_body() {
        let mut p = Physics::new();
        let id = EntityId::test(2);
        let m = Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0));
        p.add(id, &cube(), m, &Body::dynamic(), None, Vec::new)
            .unwrap();
        let mut tf = HashMap::from([(id, m)]);
        run(&mut p, 0.5, &mut tf);
        tf.insert(id, Mat4::from_translation(Vec3::new(10.0, 3.0, 0.0)));
        run(&mut p, 2.0, &mut tf);
        let at = tf[&id].w_axis;
        assert!(
            (at.x - 10.0).abs() < 0.05 && (at.y - 0.5).abs() < 0.05,
            "{at:?}"
        );
    }

    #[test]
    fn a_push_slides_a_box_and_friction_stops_it() {
        let mut p = Physics::new();
        let id = EntityId::test(3);
        let m = Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0));
        p.add(id, &cube(), m, &Body::dynamic(), None, Vec::new)
            .unwrap();
        let mut tf = HashMap::from([(id, m)]);
        run(&mut p, 0.3, &mut tf);
        let mass = p.mass(id).unwrap();
        assert!(
            (mass - 1000.0).abs() < 1.0,
            "1 m³ of water-density box: {mass}"
        );
        p.push(id, Vec3::new(3.0 * mass, 0.0, 0.0), None).unwrap();
        run(&mut p, 3.0, &mut tf);
        let x = tf[&id].w_axis.x;
        assert!(x > 0.3 && x < 2.0, "slid to {x}");
        assert!(p.velocity(id).unwrap().length() < 0.05);
    }

    #[test]
    fn rubble_expires_after_its_time() {
        let mut p = Physics::new();
        let id = EntityId::test(4);
        let m = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0));
        p.add(
            id,
            &cube(),
            m,
            &Body::dynamic(),
            Some(Temporary { keep: 0.5 }),
            Vec::new,
        )
        .unwrap();
        let mut gone = false;
        let mut tf = HashMap::from([(id, m)]);
        for _ in 0..400 {
            let r = p.tick(1.0 / 120.0, &none(), |id| tf.get(&id).copied());
            for (id, m) in r.moved {
                tf.insert(id, m);
            }
            gone |= r.expired.contains(&id);
        }
        assert!(gone);
        assert_eq!(p.temporary_count(), 0);
    }

    #[test]
    fn rays_hit_bodies_but_not_the_floor() {
        let mut p = Physics::new();
        let id = EntityId::test(5);
        let m = Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0));
        p.add(id, &cube(), m, &Body::dynamic(), None, Vec::new)
            .unwrap();
        let mut tf = HashMap::from([(id, m)]);
        run(&mut p, 0.1, &mut tf);
        let hit = p
            .cast_ray(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, 10.0)
            .unwrap();
        assert_eq!(hit.entity, Some(id));
        assert!((hit.point.y - 1.0).abs() < 0.02);
        assert!(p
            .cast_ray(Vec3::new(5.0, 5.0, 0.0), -Vec3::Y, 10.0)
            .is_none());
    }

    #[test]
    fn a_dynamic_triangle_mesh_is_refused() {
        let mut p = Physics::new();
        let err = p.add(
            EntityId::test(6),
            &cube(),
            Mat4::IDENTITY,
            &Body::dynamic().shape(BodyShape::Mesh),
            None,
            Vec::new,
        );
        assert!(err.is_err());
    }
}
