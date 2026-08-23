//! Static obstacle collision for actors moving through finite 3D spaces.
//!
//! Terrain contact — feet on the drawn ground — stays in [`crate::contact`].
//! This module is trunks, walls, floors, and ceilings: cylinders and oriented
//! boxes that a vertical capsule slides against. There is no rigid-body solver;
//! callers integrate gravity and pass the requested displacement to
//! [`CollisionWorld::move_3d`].
//!
//! [`ActorBody::player`] collides by default. Every other actor starts with
//! collision off and opts in with [`ActorBody::with_collides`].

use std::collections::{HashMap, HashSet};

use crate::error::{EngineError, EngineResult};
use crate::portal::SpaceId;
use crate::space::{GlobalPosition, GlobalXZ};

/// Horizontal radius and standing height of a moving character.
///
/// [`CollisionWorld::move_xz`] uses only `radius`; [`CollisionWorld::move_3d`]
/// uses the full vertical capsule and `step_height`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActorBody {
    pub radius: f32,
    pub height: f32,
    /// Maximum ledge climbed during horizontal movement.
    pub step_height: f32,
    /// When false, actor moves are plain translations.
    pub collides: bool,
}

impl ActorBody {
    /// Horizontal radius of the default player capsule, in metres.
    pub const PLAYER_RADIUS: f32 = 0.35;
    /// Standing height of the default player capsule, in metres.
    pub const PLAYER_HEIGHT: f32 = 1.8;

    /// Player capsule. Collision is on.
    pub fn player() -> Self {
        Self {
            radius: Self::PLAYER_RADIUS,
            height: Self::PLAYER_HEIGHT,
            step_height: 0.45,
            collides: true,
        }
    }

    /// Generic actor. Collision is off until the game turns it on.
    pub fn new(radius: f32, height: f32) -> Self {
        Self::at(radius, height, false)
    }

    fn at(radius: f32, height: f32, collides: bool) -> Self {
        if !radius.is_finite() || radius <= 0.0 {
            panic!("actor radius must be finite and > 0, got {radius}");
        }
        if !height.is_finite() || height <= 0.0 {
            panic!("actor height must be finite and > 0, got {height}");
        }
        Self {
            radius,
            height,
            step_height: 0.0,
            collides,
        }
    }

    pub fn with_step_height(mut self, step_height: f32) -> Self {
        if !step_height.is_finite() || step_height < 0.0 {
            panic!("actor step height must be finite and >= 0, got {step_height}");
        }
        self.step_height = step_height;
        self
    }

    pub fn with_collides(mut self, collides: bool) -> Self {
        self.collides = collides;
        self
    }
}

/// Result of one swept 3D actor move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActorMove {
    /// Feet position after collision resolution.
    pub position: GlobalPosition,
    /// Downward movement touched the top of a collider.
    pub grounded: bool,
    /// Upward movement touched the underside of a collider.
    pub hit_ceiling: bool,
}

/// Opaque handle to one static collider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderId(u64);

/// Game-defined layer so streamed content can be replaced in bulk.
pub type ColliderLayer = u32;

/// Horizontal footprint of an obstacle. [`StaticCollider`] supplies its
/// finite or infinite vertical span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColliderShape {
    /// Vertical cylinder. Typical for a tree trunk.
    Cylinder { radius: f32 },
    /// Oriented box in XZ. `yaw` on the collider rotates local +X/+Z.
    Box { half_x: f32, half_z: f32 },
}

/// One static obstacle in absolute world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCollider {
    pub at: GlobalXZ,
    /// Radians. Zero keeps the box aligned with world XZ.
    pub yaw: f32,
    pub shape: ColliderShape,
    /// Vertical obstacle span. The legacy constructor is infinite.
    pub min_y: f64,
    pub max_y: f64,
    /// Only actors whose live space matches this are blocked.
    pub space: SpaceId,
}

impl StaticCollider {
    pub fn new(at: GlobalXZ, yaw: f32, shape: ColliderShape) -> Self {
        Self {
            at,
            yaw,
            shape,
            min_y: f64::NEG_INFINITY,
            max_y: f64::INFINITY,
            space: SpaceId::DEFAULT,
        }
    }

    /// Give this collider a finite vertical span.
    ///
    /// Floors, walls, and ceilings must use this. Infinite spans remain useful
    /// for legacy XZ-only obstacles such as tree trunks.
    pub fn with_y_span(mut self, min_y: f64, max_y: f64) -> Self {
        if !min_y.is_finite() || !max_y.is_finite() || min_y >= max_y {
            panic!("collider y span must be finite and min < max, got {min_y}..{max_y}");
        }
        self.min_y = min_y;
        self.max_y = max_y;
        self
    }

    pub fn in_space(mut self, space: SpaceId) -> Self {
        self.space = space;
        self
    }
}

/// Spatial hash cell, in metres. Small enough that a walker hits a handful.
const HASH_CELL_M: f64 = 8.0;
const DEPENETRATE_ITERS: usize = 6;
const INSIDE_EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
struct Entry {
    layer: ColliderLayer,
    collider: StaticCollider,
}

/// Static obstacles an [`ActorBody`] can slide against.
#[derive(Clone, Debug, Default)]
pub struct CollisionWorld {
    next_id: u64,
    by_id: HashMap<ColliderId, Entry>,
    cells: HashMap<(i32, i32), Vec<ColliderId>>,
}

impl CollisionWorld {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Insert one obstacle. `layer` is the game's bulk-replace key.
    pub fn insert(
        &mut self,
        layer: ColliderLayer,
        collider: StaticCollider,
    ) -> EngineResult<ColliderId> {
        validate(&collider)?;
        let id = ColliderId(self.next_id);
        self.next_id += 1;
        self.place(id, layer, collider);
        Ok(id)
    }

    /// Drop every collider previously inserted on `layer`, then insert `colliders`.
    pub fn replace_layer(
        &mut self,
        layer: ColliderLayer,
        colliders: impl IntoIterator<Item = StaticCollider>,
    ) -> EngineResult<()> {
        self.clear_layer(layer);
        for collider in colliders {
            self.insert(layer, collider)?;
        }
        Ok(())
    }

    pub fn clear_layer(&mut self, layer: ColliderLayer) {
        let ids: Vec<ColliderId> = self
            .by_id
            .iter()
            .filter(|(_, entry)| entry.layer == layer)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.remove(id);
        }
    }

    pub fn remove(&mut self, id: ColliderId) {
        let Some(entry) = self.by_id.remove(&id) else {
            return;
        };
        for cell in aabb_cells(aabb(&entry.collider)) {
            if let Some(list) = self.cells.get_mut(&cell) {
                list.retain(|kept| *kept != id);
                if list.is_empty() {
                    self.cells.remove(&cell);
                }
            }
        }
    }

    /// Slide `from` by `(dx, dz)` against default-space colliders.
    pub fn move_xz(&self, body: &ActorBody, from: GlobalXZ, dx: f64, dz: f64) -> GlobalXZ {
        self.move_in(SpaceId::DEFAULT, body, from, dx, dz, 0.0)
    }

    /// Sweep a vertical actor capsule through finite 3D colliders.
    ///
    /// `from` is the actor's feet. Long moves are subdivided so falling and
    /// sprinting cannot tunnel through thin floors or walls.
    pub fn move_3d(
        &self,
        body: &ActorBody,
        from: GlobalPosition,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> ActorMove {
        self.move_3d_in(SpaceId::DEFAULT, body, from, dx, dy, dz)
    }

    /// [`Self::move_3d`] restricted to one disconnected world space.
    pub fn move_3d_in(
        &self,
        space: SpaceId,
        body: &ActorBody,
        from: GlobalPosition,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> ActorMove {
        if !dx.is_finite() || !dy.is_finite() || !dz.is_finite() {
            panic!("actor 3D move must be finite, got ({dx}, {dy}, {dz})");
        }
        if !body.collides {
            return ActorMove {
                position: GlobalPosition::at(from.x + dx, from.y + dy, from.z + dz),
                grounded: false,
                hit_ceiling: false,
            };
        }

        let radius = f64::from(body.radius);
        let height = f64::from(body.height);
        let distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let steps = ((distance / radius).ceil() as i32).max(1);
        let inv = 1.0 / f64::from(steps);
        let mut xz = from.horizontal();
        let mut y = from.y;
        let mut grounded = false;
        let mut hit_ceiling = false;

        for _ in 0..steps {
            let desired_xz = displace(xz, dx * inv, dz * inv);
            let blocked_xz = self.depenetrate_at(space, desired_xz, radius, y, height);
            let blocked_distance = blocked_xz.distance(desired_xz);
            if blocked_distance > INSIDE_EPS && body.step_height > 0.0 {
                if let Some(step_y) = self.step_top(
                    space,
                    desired_xz,
                    radius,
                    y,
                    height,
                    f64::from(body.step_height),
                ) {
                    let stepped_xz = self.depenetrate_at(space, desired_xz, radius, step_y, height);
                    if stepped_xz.distance(desired_xz) + INSIDE_EPS < blocked_distance {
                        xz = stepped_xz;
                        y = step_y;
                        grounded = true;
                    } else {
                        xz = blocked_xz;
                    }
                } else {
                    xz = blocked_xz;
                }
            } else {
                xz = blocked_xz;
            }
            let vertical = self.resolve_vertical(space, xz, radius, y, y + dy * inv, height);
            y = vertical.0;
            grounded |= vertical.1;
            hit_ceiling |= vertical.2;
        }

        ActorMove {
            position: GlobalPosition::at(xz.x, y, xz.z),
            grounded,
            hit_ceiling,
        }
    }

    /// Slide `from` by `(dx, dz)` against colliders that live in `space`.
    ///
    /// Long steps are split so a sprint cannot tunnel through a thin wall.
    /// Collision-off bodies are translated with no query. Only colliders whose
    /// vertical span overlaps the actor at `feet_y` participate.
    pub fn move_in(
        &self,
        space: SpaceId,
        body: &ActorBody,
        from: GlobalXZ,
        dx: f64,
        dz: f64,
        feet_y: f64,
    ) -> GlobalXZ {
        if !dx.is_finite() || !dz.is_finite() {
            panic!("actor move must be finite, got ({dx}, {dz})");
        }
        if !feet_y.is_finite() {
            panic!("actor feet_y must be finite, got {feet_y}");
        }
        if !body.collides {
            return displace(from, dx, dz);
        }
        let radius = f64::from(body.radius);
        let height = f64::from(body.height);
        let dist = (dx * dx + dz * dz).sqrt();
        if dist <= INSIDE_EPS {
            return self.depenetrate_at(space, from, radius, feet_y, height);
        }
        let step_m = radius;
        let steps = ((dist / step_m).ceil() as i32).max(1);
        let inv = 1.0 / f64::from(steps);
        let mut pos = from;
        for _ in 0..steps {
            pos = displace(pos, dx * inv, dz * inv);
            pos = self.depenetrate_at(space, pos, radius, feet_y, height);
        }
        pos
    }

    fn depenetrate_at(
        &self,
        space: SpaceId,
        mut pos: GlobalXZ,
        radius: f64,
        feet_y: f64,
        height: f64,
    ) -> GlobalXZ {
        for _ in 0..DEPENETRATE_ITERS {
            let mut pushed = false;
            for collider in self.nearby(space, pos, radius) {
                if !vertical_overlap(feet_y, feet_y + height, &collider) {
                    continue;
                }
                if let Some(mtv) = overlap_mtv(pos, radius, &collider) {
                    pos = displace(pos, mtv.0, mtv.1);
                    pushed = true;
                }
            }
            if !pushed {
                break;
            }
        }
        pos
    }

    fn resolve_vertical(
        &self,
        space: SpaceId,
        pos: GlobalXZ,
        radius: f64,
        from_y: f64,
        target_y: f64,
        height: f64,
    ) -> (f64, bool, bool) {
        const CONTACT_EPS: f64 = 1e-6;
        if (target_y - from_y).abs() <= CONTACT_EPS {
            return (target_y, false, false);
        }

        let mut resolved = target_y;
        let mut grounded = false;
        let mut hit_ceiling = false;
        for collider in self.nearby(space, pos, radius) {
            if !collider.min_y.is_finite() || !collider.max_y.is_finite() {
                continue;
            }
            if overlap_mtv(pos, radius, &collider).is_none() {
                continue;
            }
            if target_y < from_y {
                let top = collider.max_y;
                if top <= from_y + CONTACT_EPS && top >= resolved - CONTACT_EPS {
                    resolved = resolved.max(top);
                    grounded = true;
                }
            } else {
                let old_head = from_y + height;
                let target_head = resolved + height;
                let bottom = collider.min_y;
                if bottom >= old_head - CONTACT_EPS && bottom <= target_head + CONTACT_EPS {
                    resolved = resolved.min(bottom - height);
                    hit_ceiling = true;
                }
            }
        }
        (resolved, grounded, hit_ceiling)
    }

    fn step_top(
        &self,
        space: SpaceId,
        pos: GlobalXZ,
        radius: f64,
        feet_y: f64,
        height: f64,
        max_step: f64,
    ) -> Option<f64> {
        const CONTACT_EPS: f64 = 1e-6;
        self.nearby(space, pos, radius)
            .into_iter()
            .filter(|collider| collider.max_y.is_finite())
            .filter(|collider| overlap_mtv(pos, radius, collider).is_some())
            .map(|collider| collider.max_y)
            .filter(|top| *top > feet_y + CONTACT_EPS && *top <= feet_y + max_step + CONTACT_EPS)
            .filter(|top| {
                !self.nearby(space, pos, radius).into_iter().any(|collider| {
                    overlap_mtv(pos, radius, &collider).is_some()
                        && vertical_overlap(*top, *top + height, &collider)
                })
            })
            .max_by(f64::total_cmp)
    }

    fn nearby(&self, space: SpaceId, pos: GlobalXZ, radius: f64) -> Vec<StaticCollider> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for cell in aabb_cells((
            pos.x - radius,
            pos.z - radius,
            pos.x + radius,
            pos.z + radius,
        )) {
            let Some(ids) = self.cells.get(&cell) else {
                continue;
            };
            for id in ids {
                if !seen.insert(*id) {
                    continue;
                }
                if let Some(entry) = self.by_id.get(id) {
                    if entry.collider.space == space {
                        out.push(entry.collider);
                    }
                }
            }
        }
        out
    }

    fn place(&mut self, id: ColliderId, layer: ColliderLayer, collider: StaticCollider) {
        for cell in aabb_cells(aabb(&collider)) {
            self.cells.entry(cell).or_default().push(id);
        }
        self.by_id.insert(id, Entry { layer, collider });
    }
}

fn validate(collider: &StaticCollider) -> EngineResult<()> {
    if !collider.yaw.is_finite() {
        return Err(EngineError::InvalidValue(format!(
            "collider yaw must be finite, got {}",
            collider.yaw
        )));
    }
    let legacy_infinite = collider.min_y == f64::NEG_INFINITY && collider.max_y == f64::INFINITY;
    let finite_span =
        collider.min_y.is_finite() && collider.max_y.is_finite() && collider.min_y < collider.max_y;
    if !legacy_infinite && !finite_span {
        return Err(EngineError::InvalidValue(format!(
            "collider y span must be finite and min < max, or legacy infinite; got {}..{}",
            collider.min_y, collider.max_y
        )));
    }
    match collider.shape {
        ColliderShape::Cylinder { radius } => {
            if !radius.is_finite() || radius <= 0.0 {
                return Err(EngineError::InvalidValue(format!(
                    "cylinder radius must be finite and > 0, got {radius}"
                )));
            }
        }
        ColliderShape::Box { half_x, half_z } => {
            if !half_x.is_finite() || half_x <= 0.0 || !half_z.is_finite() || half_z <= 0.0 {
                return Err(EngineError::InvalidValue(format!(
                    "box half-extents must be finite and > 0, got ({half_x}, {half_z})"
                )));
            }
        }
    }
    Ok(())
}

fn vertical_overlap(actor_min: f64, actor_max: f64, collider: &StaticCollider) -> bool {
    const CONTACT_EPS: f64 = 1e-6;
    actor_min < collider.max_y - CONTACT_EPS && actor_max > collider.min_y + CONTACT_EPS
}

fn displace(p: GlobalXZ, dx: f64, dz: f64) -> GlobalXZ {
    GlobalXZ::at(p.x + dx, p.z + dz)
}

fn hash_cell(m: f64) -> i32 {
    (m / HASH_CELL_M).floor() as i32
}

fn aabb_cells(
    (min_x, min_z, max_x, max_z): (f64, f64, f64, f64),
) -> impl Iterator<Item = (i32, i32)> {
    let x0 = hash_cell(min_x);
    let x1 = hash_cell(max_x);
    let z0 = hash_cell(min_z);
    let z1 = hash_cell(max_z);
    (z0..=z1).flat_map(move |z| (x0..=x1).map(move |x| (x, z)))
}

fn aabb(collider: &StaticCollider) -> (f64, f64, f64, f64) {
    match collider.shape {
        ColliderShape::Cylinder { radius } => {
            let r = f64::from(radius);
            (
                collider.at.x - r,
                collider.at.z - r,
                collider.at.x + r,
                collider.at.z + r,
            )
        }
        ColliderShape::Box { half_x, half_z } => {
            let (sin, cos) = (collider.yaw.sin().abs(), collider.yaw.cos().abs());
            let ex = f64::from(half_x * cos + half_z * sin);
            let ez = f64::from(half_x * sin + half_z * cos);
            (
                collider.at.x - ex,
                collider.at.z - ez,
                collider.at.x + ex,
                collider.at.z + ez,
            )
        }
    }
}

/// Vector that pushes a circle of `radius` out of `collider`, or `None`.
fn overlap_mtv(pos: GlobalXZ, radius: f64, collider: &StaticCollider) -> Option<(f64, f64)> {
    match collider.shape {
        ColliderShape::Cylinder { radius: other } => {
            circle_circle_mtv(pos, radius, collider.at, f64::from(other))
        }
        ColliderShape::Box { half_x, half_z } => circle_box_mtv(
            pos,
            radius,
            collider.at,
            collider.yaw,
            f64::from(half_x),
            f64::from(half_z),
        ),
    }
}

fn circle_circle_mtv(pos: GlobalXZ, radius: f64, at: GlobalXZ, other: f64) -> Option<(f64, f64)> {
    let dx = pos.x - at.x;
    let dz = pos.z - at.z;
    let dist_sq = dx * dx + dz * dz;
    let min_sep = radius + other;
    if dist_sq >= min_sep * min_sep {
        return None;
    }
    if dist_sq <= INSIDE_EPS {
        return Some((min_sep, 0.0));
    }
    let dist = dist_sq.sqrt();
    let scale = (min_sep - dist) / dist;
    Some((dx * scale, dz * scale))
}

fn circle_box_mtv(
    pos: GlobalXZ,
    radius: f64,
    at: GlobalXZ,
    yaw: f32,
    half_x: f64,
    half_z: f64,
) -> Option<(f64, f64)> {
    let (sin, cos) = (f64::from(yaw.sin()), f64::from(yaw.cos()));
    let dx = pos.x - at.x;
    let dz = pos.z - at.z;
    let lx = dx * cos - dz * sin;
    let lz = dx * sin + dz * cos;
    let closest_x = lx.clamp(-half_x, half_x);
    let closest_z = lz.clamp(-half_z, half_z);
    let ox = lx - closest_x;
    let oz = lz - closest_z;
    let dist_sq = ox * ox + oz * oz;
    let inside = lx.abs() <= half_x && lz.abs() <= half_z;
    let (mlx, mlz) = if inside {
        let px = half_x - lx.abs();
        let pz = half_z - lz.abs();
        if px < pz {
            let sx = if lx >= 0.0 { 1.0 } else { -1.0 };
            (sx * (px + radius), 0.0)
        } else {
            let sz = if lz >= 0.0 { 1.0 } else { -1.0 };
            (0.0, sz * (pz + radius))
        }
    } else if dist_sq < radius * radius {
        if dist_sq <= INSIDE_EPS {
            (radius, 0.0)
        } else {
            let dist = dist_sq.sqrt();
            let scale = (radius - dist) / dist;
            (ox * scale, oz * scale)
        }
    } else {
        return None;
    };
    Some((mlx * cos + mlz * sin, -mlx * sin + mlz * cos))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cylinder(x: f64, z: f64, radius: f32) -> StaticCollider {
        StaticCollider::new(GlobalXZ::at(x, z), 0.0, ColliderShape::Cylinder { radius })
    }

    fn wall(x: f64, z: f64, half_x: f32, half_z: f32) -> StaticCollider {
        StaticCollider::new(
            GlobalXZ::at(x, z),
            0.0,
            ColliderShape::Box { half_x, half_z },
        )
    }

    fn finite_box(
        x: f64,
        z: f64,
        half_x: f32,
        half_z: f32,
        min_y: f64,
        max_y: f64,
    ) -> StaticCollider {
        wall(x, z, half_x, half_z).with_y_span(min_y, max_y)
    }

    #[test]
    fn player_body_collides_by_default() {
        assert!(ActorBody::player().collides);
        assert!(!ActorBody::new(0.4, 1.2).collides);
        assert!(ActorBody::new(0.4, 1.2).with_collides(true).collides);
    }

    #[test]
    fn cylinder_stops_a_capsule() {
        let mut world = CollisionWorld::new();
        world.insert(1, cylinder(2.0, 0.0, 0.5)).expect("insert");
        let body = ActorBody::player();
        let from = GlobalXZ::at(0.0, 0.0);
        let to = world.move_xz(&body, from, 4.0, 0.0);
        assert!(
            to.x < 1.2,
            "player should stop before the trunk, got x={}",
            to.x
        );
        assert!(to.x > 0.9, "player should reach the trunk, got x={}", to.x);
        assert!(to.z.abs() < 1e-6, "no sideways drift, got z={}", to.z);
    }

    #[test]
    fn disabled_body_walks_through() {
        let mut world = CollisionWorld::new();
        world.insert(1, cylinder(2.0, 0.0, 0.5)).expect("insert");
        let body = ActorBody::new(0.35, 1.8);
        let to = world.move_xz(&body, GlobalXZ::at(0.0, 0.0), 4.0, 0.0);
        assert!((to.x - 4.0).abs() < 1e-9);
        assert!(to.z.abs() < 1e-9);
    }

    #[test]
    fn box_stops_a_capsule() {
        let mut world = CollisionWorld::new();
        world.insert(1, wall(2.0, 0.0, 0.25, 4.0)).expect("insert");
        let body = ActorBody::player();
        let to = world.move_xz(&body, GlobalXZ::at(0.0, 0.0), 4.0, 0.0);
        assert!(to.x < 1.5, "should not pass the wall, got x={}", to.x);
        assert!(to.x > 1.0, "should reach the wall, got x={}", to.x);
    }

    #[test]
    fn slide_along_a_wall() {
        let mut world = CollisionWorld::new();
        world.insert(1, wall(2.0, 0.0, 0.25, 8.0)).expect("insert");
        let body = ActorBody::player();
        let to = world.move_xz(&body, GlobalXZ::at(0.0, 0.0), 4.0, 3.0);
        assert!(to.x < 1.5, "still blocked in X, got x={}", to.x);
        assert!(
            to.z > 2.0,
            "tangential travel should survive, got z={}",
            to.z
        );
    }

    #[test]
    fn replace_layer_drops_old_colliders() {
        let mut world = CollisionWorld::new();
        world.insert(1, cylinder(0.0, 0.0, 1.0)).expect("insert");
        world
            .replace_layer(1, [cylinder(10.0, 0.0, 1.0)])
            .expect("replace");
        assert_eq!(world.len(), 1);
        let body = ActorBody::player();
        let through = world.move_xz(&body, GlobalXZ::at(-2.0, 0.0), 4.0, 0.0);
        assert!(
            (through.x - 2.0).abs() < 0.05,
            "old trunk should be gone, got x={}",
            through.x
        );
    }

    #[test]
    fn sprint_does_not_tunnel_a_thin_wall() {
        let mut world = CollisionWorld::new();
        world.insert(1, wall(1.0, 0.0, 0.2, 4.0)).expect("insert");
        let body = ActorBody::player();
        let to = world.move_xz(&body, GlobalXZ::at(0.0, 0.0), 8.0, 0.0);
        assert!(to.x < 1.0, "must not tunnel, got x={}", to.x);
    }

    #[test]
    fn other_space_walls_are_ignored() {
        let mut world = CollisionWorld::new();
        let house = SpaceId::from_raw(3);
        world
            .insert(1, wall(2.0, 0.0, 0.25, 4.0).in_space(house))
            .expect("insert");
        let body = ActorBody::player();
        let through = world.move_xz(&body, GlobalXZ::at(0.0, 0.0), 4.0, 0.0);
        assert!(
            (through.x - 4.0).abs() < 1e-6,
            "default-space walk should ignore a house wall, got x={}",
            through.x
        );
        let blocked = world.move_in(house, &body, GlobalXZ::at(0.0, 0.0), 4.0, 0.0, 0.0);
        assert!(
            blocked.x < 1.5,
            "house walk should hit the wall, got x={}",
            blocked.x
        );
    }

    #[test]
    fn actor_falls_onto_a_finite_floor() {
        let mut world = CollisionWorld::new();
        world
            .insert(1, finite_box(0.0, 0.0, 2.0, 2.0, -0.2, 0.0))
            .expect("floor");
        let moved = world.move_3d(
            &ActorBody::player(),
            GlobalPosition::at(0.0, 3.0, 0.0),
            0.0,
            -6.0,
            0.0,
        );
        assert!(moved.grounded);
        assert!(!moved.hit_ceiling);
        assert!((moved.position.y - 0.0).abs() < 1e-6, "{moved:?}");
    }

    #[test]
    fn actor_hits_a_finite_ceiling() {
        let mut world = CollisionWorld::new();
        world
            .insert(1, finite_box(0.0, 0.0, 2.0, 2.0, 2.0, 2.2))
            .expect("ceiling");
        let moved = world.move_3d(
            &ActorBody::player(),
            GlobalPosition::at(0.0, 0.0, 0.0),
            0.0,
            2.0,
            0.0,
        );
        assert!(!moved.grounded);
        assert!(moved.hit_ceiling);
        assert!((moved.position.y - 0.2).abs() < 1e-5, "{moved:?}");
    }

    #[test]
    fn xz_move_respects_collider_y_span() {
        let mut world = CollisionWorld::new();
        world
            .insert(1, finite_box(2.0, 0.0, 0.25, 4.0, 2.7, 5.4))
            .expect("upper wall");
        let body = ActorBody::player();
        let through = world.move_in(
            SpaceId::DEFAULT,
            &body,
            GlobalXZ::at(0.0, 0.0),
            4.0,
            0.0,
            0.0,
        );
        assert!(
            (through.x - 4.0).abs() < 1e-6,
            "ground walker should pass under an upper-storey wall, got x={}",
            through.x
        );
        let blocked = world.move_in(
            SpaceId::DEFAULT,
            &body,
            GlobalXZ::at(0.0, 0.0),
            4.0,
            0.0,
            3.0,
        );
        assert!(
            blocked.x < 1.5,
            "loft walker should hit the upper-storey wall, got x={}",
            blocked.x
        );
    }

    #[test]
    fn finite_wall_only_blocks_its_storey() {
        let mut world = CollisionWorld::new();
        world
            .insert(1, finite_box(2.0, 0.0, 0.25, 4.0, 0.0, 2.7))
            .expect("wall");
        let body = ActorBody::player();
        let blocked = world.move_3d(&body, GlobalPosition::at(0.0, 0.0, 0.0), 4.0, 0.0, 0.0);
        assert!(blocked.position.x < 1.5, "{blocked:?}");
        let above = world.move_3d(&body, GlobalPosition::at(0.0, 3.0, 0.0), 4.0, 0.0, 0.0);
        assert!((above.position.x - 4.0).abs() < 1e-6, "{above:?}");
    }

    #[test]
    fn falling_sprint_does_not_tunnel_floor_or_wall() {
        let mut world = CollisionWorld::new();
        world
            .insert(1, finite_box(0.0, 0.0, 4.0, 4.0, -0.1, 0.0))
            .expect("floor");
        world
            .insert(1, finite_box(1.5, 0.0, 0.1, 4.0, 0.0, 2.7))
            .expect("wall");
        let moved = world.move_3d(
            &ActorBody::player(),
            GlobalPosition::at(0.0, 2.0, 0.0),
            8.0,
            -8.0,
            0.0,
        );
        assert!(moved.position.x < 1.5, "{moved:?}");
        assert!(moved.position.y >= -1e-6, "{moved:?}");
    }

    #[test]
    fn player_steps_up_a_low_ledge_but_not_a_tall_one() {
        let body = ActorBody::player();
        let mut low = CollisionWorld::new();
        low.insert(1, finite_box(0.8, 0.0, 0.3, 2.0, 0.2, 0.3))
            .unwrap();
        let stepped = low.move_3d(&body, GlobalPosition::at(0.0, 0.0, 0.0), 1.0, 0.0, 0.0);
        assert!(stepped.position.x > 0.7, "{stepped:?}");
        assert!((stepped.position.y - 0.3).abs() < 1e-6, "{stepped:?}");

        let mut tall = CollisionWorld::new();
        tall.insert(1, finite_box(0.8, 0.0, 0.3, 2.0, 0.7, 0.8))
            .unwrap();
        let blocked = tall.move_3d(&body, GlobalPosition::at(0.0, 0.0, 0.0), 1.0, 0.0, 0.0);
        assert!(blocked.position.x < 0.5, "{blocked:?}");
        assert!(blocked.position.y.abs() < 1e-6, "{blocked:?}");
    }
}
