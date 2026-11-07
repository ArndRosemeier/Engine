//! Breakable buildings in the world: drawing, collapse animation, debris and
//! rubble on top of [`crate::building`].

use super::{EntityId, World};
use crate::building::{
    Building, BuildingChange, BuildingPlan, CollapseMode, ElementBits, ElementId,
};
use crate::error::{EngineError, EngineResult};
use crate::mesh::BuiltMesh;
use crate::place::Place;
use glam::{Mat4, Quat, Vec3};
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

/// Most debris bodies tumbling at once; older ones freeze into static rubble.
const MAX_FLYING: usize = 300;
/// Most pieces of rubble lying around; the oldest go first.
const MAX_RUBBLE: usize = 2500;
/// New debris bodies per frame, and the token bucket behind the per-second rate.
const MAX_NEW_PER_FRAME: f32 = 24.0;
const TOKENS: f32 = 60.0;
const TOKENS_PER_SECOND: f32 = 40.0;
/// Seconds a debris body may fly before it freezes anyway.
const FLY_SECONDS: f32 = 8.0;

/// Element bits the GPU reads, shared by every breakable mesh in the world.
#[derive(Debug, Default)]
pub(crate) struct ElementStates {
    /// One bit per global element id (bit set = gone). Id 0 is never used.
    pub bits: Vec<u32>,
    next: u32,
    free: Vec<(u32, u32)>,
    /// Words changed since the renderer last uploaded them.
    pub dirty: Option<(usize, usize)>,
}

impl ElementStates {
    /// A run of `n` ids, all alive.
    fn alloc(&mut self, n: u32) -> EngineResult<u32> {
        if let Some(k) = self.free.iter().position(|r| r.1 >= n) {
            let (start, len) = self.free[k];
            if len == n {
                self.free.remove(k);
            } else {
                self.free[k] = (start + n, len - n);
            }
            for i in start..start + n {
                self.set(i, false);
            }
            return Ok(start);
        }
        let start = self.next.max(1);
        let end = start as usize + n as usize;
        if end > crate::render::ELEMENT_STATE_WORDS * 32 {
            return Err(EngineError::InvalidValue(format!(
                "too many breakable elements in the world (limit {})",
                crate::render::ELEMENT_STATE_WORDS * 32
            )));
        }
        self.next = end as u32;
        if self.bits.len() < end.div_ceil(32) {
            self.bits.resize(end.div_ceil(32), 0);
        }
        Ok(start)
    }

    fn release(&mut self, start: u32, n: u32) {
        if n > 0 {
            self.free.push((start, n));
        }
    }

    pub fn set(&mut self, id: u32, dead: bool) {
        let w = id as usize / 32;
        if w >= self.bits.len() {
            self.bits.resize(w + 1, 0);
        }
        let bit = 1u32 << (id % 32);
        let before = self.bits[w];
        if dead {
            self.bits[w] |= bit;
        } else {
            self.bits[w] &= !bit;
        }
        if self.bits[w] != before {
            self.dirty = Some(match self.dirty {
                Some((a, b)) => (a.min(w), b.max(w + 1)),
                None => (w, w + 1),
            });
        }
    }

    pub fn get(&self, id: u32) -> bool {
        self.bits
            .get(id as usize / 32)
            .is_some_and(|w| w & (1 << (id % 32)) != 0)
    }
}

/// A building's drawing state.
#[derive(Debug)]
pub(crate) struct BuildingVisual {
    pub state: Building,
    /// Global element id of local element 0.
    pub base: u32,
    pub mounds: Vec<EntityId>,
    /// Physics owner key of the element colliders.
    pub owner: u64,
    /// The plan's rooms in the world, and the openings elements leave.
    pub rooms: Vec<crate::rooms::RoomId>,
    pub openings: Vec<(u32, crate::rooms::OpeningId)>,
}

/// The upper part of a building on its way down.
#[derive(Debug)]
pub(crate) struct FallingPart {
    pub building: EntityId,
    entity: EntityId,
    plan: Arc<BuildingPlan>,
    /// Global id of local element 0 in the falling mesh.
    base: u32,
    transform: Mat4,
    mode: CollapseMode,
    from_floor: usize,
    base_y: f32,
    height: f32,
    radius: f32,
    eaten: usize,
    drop: f32,
    vel: f32,
    angle: f32,
    spin: f32,
    shed: f32,
    elements: Vec<u32>,
    seed: u64,
}

/// Debris and rubble bookkeeping.
#[derive(Debug)]
pub(crate) struct RubbleSet {
    /// Pieces with a body, oldest first, with their age.
    flying: VecDeque<(EntityId, f32)>,
    /// Pieces frozen in place, oldest first.
    resting: VecDeque<EntityId>,
    tokens: f32,
    frame_budget: f32,
    seed: u64,
}

impl Default for RubbleSet {
    fn default() -> Self {
        Self {
            flying: VecDeque::new(),
            resting: VecDeque::new(),
            tokens: TOKENS,
            frame_budget: MAX_NEW_PER_FRAME,
            seed: 0x51ed_270b,
        }
    }
}

/// Pre-fractured unit boxes the debris is made from.
fn templates() -> &'static Vec<Vec<crate::fracture::Fragment>> {
    static T: OnceLock<Vec<Vec<crate::fracture::Fragment>>> = OnceLock::new();
    T.get_or_init(|| {
        let cube = crate::mesh::Mesh::box_at(Vec3::ZERO, Vec3::ONE, crate::color::Color::WHITE)
            .expect("unit box")
            .build();
        (0..4)
            .map(|seed| {
                crate::fracture::fracture(
                    &cube,
                    Mat4::IDENTITY,
                    Vec3::new(0.3, 0.2, -0.1) * (seed as f32 - 1.5),
                    crate::fracture::FractureSettings {
                        pieces: 5 + seed,
                        seed: 11 + seed as u32 * 7,
                        interior_shade: 0.7,
                    },
                )
            })
            .collect()
    })
}

impl World {
    /// Put a breakable building into the world (see [`crate::building`]).
    /// Its elements also become fixed colliders, so bodies rest on floors
    /// and bounce off walls until those break.
    pub fn spawn_building(
        &mut self,
        plan: Arc<BuildingPlan>,
        place: Place,
    ) -> EngineResult<EntityId> {
        if plan.is_empty() {
            return Err(EngineError::InvalidValue(
                "building plan has no elements".into(),
            ));
        }
        let base = self.element_states.alloc(plan.len() as u32)?;
        let mut mesh = plan.mesh().clone();
        for e in mesh.elements.iter_mut() {
            *e = base + *e - 1;
        }
        let transform = place.to_matrix();
        let id = self.spawn_built(mesh, transform);
        let owner = id.raw() | (1 << 62);
        let (_, rot, pos) = transform.to_scale_rotation_translation();
        for (i, e) in plan.elements().iter().enumerate() {
            self.physics
                .add_static_box(owner, i as u32, pos + rot * e.center, rot, e.half);
        }
        let (rooms, openings) = self.add_building_rooms(&plan, transform, owner)?;
        self.buildings.insert(
            id,
            BuildingVisual {
                state: Building::new(plan),
                base,
                mounds: Vec::new(),
                owner,
                rooms,
                openings,
            },
        );
        self.sync_building_openings(id);
        Ok(id)
    }

    /// The plan's rooms and openings, placed in the world.
    #[allow(clippy::type_complexity)]
    fn add_building_rooms(
        &mut self,
        plan: &BuildingPlan,
        transform: Mat4,
        owner: u64,
    ) -> EngineResult<(
        Vec<crate::rooms::RoomId>,
        Vec<(u32, crate::rooms::OpeningId)>,
    )> {
        use crate::rooms::{Opening, Room};
        let (scale, rot, _) = transform.to_scale_rotation_translation();
        let rooms: Vec<_> = plan
            .rooms()
            .iter()
            .map(|(c, h)| {
                let room = Room::new(transform.transform_point3(*c), *h * scale).rotated(rot);
                self.rooms.add_room(room, Some(owner))
            })
            .collect();
        let corners = |q: [Vec3; 4]| q.map(|c| transform.transform_point3(c));
        for gap in plan.gaps() {
            let o = Opening::door(
                rooms[gap.a as usize],
                gap.b.map(|b| rooms[b as usize]),
                corners(gap.corners),
            );
            self.rooms.add_opening(o, true, Some(owner))?;
        }
        let mut openings = Vec::new();
        for i in 0..plan.len() {
            if let Some((a, b, q)) = plan.element_opening(i) {
                let o = Opening::door(rooms[a as usize], b.map(|b| rooms[b as usize]), corners(q));
                openings.push((i as u32, self.rooms.add_opening(o, false, Some(owner))?));
            }
        }
        Ok((rooms, openings))
    }

    /// Windows, broken elements and anything after a collapse are open.
    fn sync_building_openings(&mut self, id: EntityId) {
        let Some(b) = self.buildings.get(&id) else {
            return;
        };
        let gone = b.state.gone();
        let collapsed = b.state.collapse().is_some();
        let plan = b.state.plan();
        for (i, o) in &b.openings {
            let window = plan.elements()[*i as usize].kind == crate::building::ElementKind::Window;
            let open = window || collapsed || gone.get(*i as usize);
            let _ = self.rooms.set_open(*o, open);
        }
    }

    /// A building's state (what is broken, what fell, the rubble).
    pub fn building(&self, id: EntityId) -> Option<&Building> {
        self.buildings.get(&id).map(|b| &b.state)
    }

    /// Every building in the world.
    pub fn buildings(&self) -> impl Iterator<Item = (EntityId, &Building)> {
        self.buildings.iter().map(|(id, b)| (*id, &b.state))
    }

    /// Hit a building at a world point: elements within `radius` weaker than
    /// `power` (fading to the edge) break, debris flies, and an unsupported
    /// upper part collapses.
    pub fn hit_building(
        &mut self,
        id: EntityId,
        point: impl Into<Vec3>,
        radius: f32,
        power: f32,
    ) -> EngineResult<BuildingChange> {
        let point = point.into();
        crate::place::ensure_finite3(point, "hit point")?;
        let local = self.building_local(id)?.transform_point3(point);
        let change = self
            .buildings
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .state
            .hit(local, radius, power);
        self.show_building_change(id, &change, Some(point), true);
        Ok(change)
    }

    /// Hit every building near `point` (see [`Self::hit_building`]).
    pub fn hit_buildings(
        &mut self,
        point: impl Into<Vec3>,
        radius: f32,
        power: f32,
    ) -> Vec<(EntityId, BuildingChange)> {
        let point = point.into();
        let mut ids: Vec<EntityId> = self.buildings.keys().copied().collect();
        ids.sort_by_key(|id| id.raw());
        let mut out = Vec::new();
        for id in ids {
            if let Ok(change) = self.hit_building(id, point, radius, power) {
                if !change.is_empty() {
                    out.push((id, change));
                }
            }
        }
        out
    }

    /// Break chosen elements of a building.
    pub fn break_building_elements(
        &mut self,
        id: EntityId,
        elements: &[ElementId],
    ) -> EngineResult<BuildingChange> {
        let change = self
            .buildings
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .state
            .break_elements(elements);
        self.show_building_change(id, &change, None, true);
        Ok(change)
    }

    /// Take over broken bits from the server or a save. `animate`: play the
    /// breaking (debris, a falling part); otherwise jump straight to the
    /// result (a player joining late).
    pub fn apply_building_state(
        &mut self,
        id: EntityId,
        broken: &ElementBits,
        animate: bool,
    ) -> EngineResult<BuildingChange> {
        let b = self
            .buildings
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        if broken.len() != b.state.plan().len() {
            return Err(EngineError::InvalidValue(format!(
                "building has {} elements, the bits {}",
                b.state.plan().len(),
                broken.len()
            )));
        }
        // Elements broken before but whole in the new bits come back.
        let restored: Vec<u32> = b.state.gone().iter_set().map(|i| i as u32).collect();
        let change = b.state.apply(broken);
        let gone = b.state.gone();
        let (base, owner) = (b.base, b.owner);
        let plan = b.state.plan().clone();
        let transform = self
            .entities
            .get(&id)
            .map_or(Mat4::IDENTITY, |e| e.transform);
        let (_, rot, pos) = transform.to_scale_rotation_translation();
        for i in restored {
            if !gone.get(i as usize) {
                self.element_states.set(base + i, false);
                let e = &plan.elements()[i as usize];
                self.physics
                    .add_static_box(owner, i, pos + rot * e.center, rot, e.half);
            }
        }
        self.show_building_change(id, &change, None, animate);
        self.sync_building_openings(id);
        if !animate || change.collapse.is_none() {
            self.place_mounds(id);
        }
        Ok(change)
    }

    /// First standing building element along a ray: the building, the
    /// element and the world point.
    pub fn cast_ray_buildings(
        &self,
        origin: impl Into<Vec3>,
        direction: impl Into<Vec3>,
        max_distance: f32,
    ) -> Option<(EntityId, ElementId, Vec3)> {
        let (origin, dir) = (origin.into(), direction.into().try_normalize()?);
        let mut best: Option<(EntityId, ElementId, f32)> = None;
        for (id, b) in &self.buildings {
            let Some(e) = self.entities.get(id) else {
                continue;
            };
            let inv = e.transform.inverse();
            let o = inv.transform_point3(origin);
            let d = inv.transform_vector3(dir);
            let scale = d.length();
            if let Some((el, t)) = b.state.raycast(o, d, max_distance * scale) {
                let t = t / scale.max(1e-6);
                if best.is_none_or(|x| t < x.2) {
                    best = Some((*id, el, t));
                }
            }
        }
        best.map(|(id, el, t)| (id, el, origin + dir * t))
    }

    /// Debris pieces flying and lying around.
    pub fn rubble_count(&self) -> (usize, usize) {
        (self.rubble.flying.len(), self.rubble.resting.len())
    }

    fn building_local(&self, id: EntityId) -> EngineResult<Mat4> {
        if !self.buildings.contains_key(&id) {
            return Err(EngineError::UnknownEntity);
        }
        Ok(self
            .entities
            .get(&id)
            .ok_or(EngineError::UnknownEntity)?
            .transform
            .inverse())
    }

    /// Hide what broke, drop colliders, throw debris, start a collapse.
    fn show_building_change(
        &mut self,
        id: EntityId,
        change: &BuildingChange,
        impact: Option<Vec3>,
        animate: bool,
    ) {
        let Some(b) = self.buildings.get(&id) else {
            return;
        };
        let (base, owner) = (b.base, b.owner);
        let plan = b.state.plan().clone();
        let collapse = b.state.collapse();
        let transform = self
            .entities
            .get(&id)
            .map_or(Mat4::IDENTITY, |e| e.transform);
        let all = change
            .broken
            .iter()
            .chain(&change.dropped)
            .chain(&change.collapsed);
        for el in all {
            self.element_states.set(base + el.0, true);
            self.physics.remove_static(owner, el.0);
        }
        self.sync_building_openings(id);
        if !animate {
            let _ = collapse;
            return;
        }
        for (k, el) in change.broken.iter().chain(&change.dropped).enumerate() {
            let push = impact.map_or(Vec3::new(0.0, -1.0, 0.0), |p| {
                let c = transform.transform_point3(plan.elements()[el.0 as usize].center);
                (c - p).normalize_or_zero() * 4.0
            });
            self.spawn_element_debris(&plan, transform, el.0, push, k);
        }
        if let (Some(c), false) = (change.collapse, change.collapsed.is_empty()) {
            self.start_collapse(id, &plan, transform, c, &change.collapsed);
        }
        let _ = collapse;
    }

    fn start_collapse(
        &mut self,
        id: EntityId,
        plan: &Arc<BuildingPlan>,
        transform: Mat4,
        c: crate::building::Collapse,
        elements: &[ElementId],
    ) {
        let Ok(base) = self.element_states.alloc(plan.len() as u32) else {
            self.place_mounds(id);
            return;
        };
        // Everything not falling stays hidden in the falling mesh.
        let set: std::collections::HashSet<u32> = elements.iter().map(|e| e.0).collect();
        for i in 0..plan.len() as u32 {
            if !set.contains(&i) {
                self.element_states.set(base + i, true);
            }
        }
        let mesh: BuiltMesh = plan.extract(|local| {
            if set.contains(&local) {
                base + local
            } else {
                0
            }
        });
        let entity = self.spawn_built(mesh, transform);
        let (lo, hi) = plan.footprint();
        let s = (hi - lo).max(glam::Vec2::splat(0.1));
        self.falling.push(FallingPart {
            building: id,
            entity,
            plan: plan.clone(),
            base,
            transform,
            mode: c.mode,
            from_floor: c.from_floor as usize,
            base_y: c.base_y,
            height: c.height,
            radius: (s.x * s.y / std::f32::consts::PI).sqrt(),
            eaten: c.from_floor as usize,
            drop: 0.0,
            vel: 0.0,
            angle: 0.03,
            spin: 0.18,
            shed: 0.0,
            elements: elements.iter().map(|e| e.0).collect(),
            seed: plan.seed() ^ id.raw(),
        });
    }

    /// Advance falling parts, freeze settled debris. The engine calls this every frame.
    pub fn tick_buildings(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.rubble.tokens = (self.rubble.tokens + dt * TOKENS_PER_SECOND).min(TOKENS);
        self.rubble.frame_budget = MAX_NEW_PER_FRAME;
        let mut parts = std::mem::take(&mut self.falling);
        parts.retain_mut(|p| self.step_falling(p, dt));
        parts.append(&mut self.falling);
        self.falling = parts;
        // Freeze debris that came to rest (or flew long enough).
        let mut keep = VecDeque::with_capacity(self.rubble.flying.len());
        while let Some((id, age)) = self.rubble.flying.pop_front() {
            if !self.entities.contains_key(&id) {
                continue;
            }
            let age = age + dt;
            if age > FLY_SECONDS || (age > 0.5 && self.physics.is_sleeping(id)) {
                self.physics.remove(id);
                self.rubble.resting.push_back(id);
            } else {
                keep.push_back((id, age));
            }
        }
        self.rubble.flying = keep;
        while self.rubble.resting.len() > MAX_RUBBLE {
            if let Some(id) = self.rubble.resting.pop_front() {
                self.despawn(id);
            }
        }
    }

    /// One frame of a falling part; false when it is done.
    fn step_falling(&mut self, p: &mut FallingPart, dt: f32) -> bool {
        let g = 9.81;
        let floors = p.plan.floors().to_vec();
        match p.mode {
            CollapseMode::Pancake => {
                // Resisted fall: each storey crushed takes energy.
                p.vel += g * 0.62 * dt;
                p.drop += p.vel * dt;
                let m = p.transform * Mat4::from_translation(Vec3::new(0.0, -p.drop, 0.0));
                self.write_falling(p.entity, m);
                while p.eaten < floors.len() && floors[p.eaten].y0 - p.drop <= p.base_y + 0.3 {
                    self.eat_floor(p, p.eaten);
                    p.eaten += 1;
                }
                if p.eaten >= floors.len() {
                    self.finish_falling(p);
                    return false;
                }
                true
            }
            CollapseMode::Topple { hinge, toward } => {
                let r = p.height * 0.5;
                let inertia = p.height * p.height / 3.0 + p.radius * p.radius / 4.0;
                p.spin += g * r * (p.angle + 0.05).sin() / inertia * dt;
                p.angle += p.spin * dt;
                let axis = Vec3::Y.cross(toward).normalize_or_zero();
                let q = Quat::from_axis_angle(axis, p.angle);
                let m = p.transform
                    * Mat4::from_translation(hinge)
                    * Mat4::from_quat(q)
                    * Mat4::from_translation(-hinge);
                self.write_falling(p.entity, m);
                p.shed += dt;
                if p.shed > 0.12 {
                    // Crushed storeys at the hinge shed a little.
                    p.shed = 0.0;
                    if p.eaten < floors.len() {
                        self.eat_floor(p, p.eaten);
                        p.eaten = (p.eaten + 1).min(p.from_floor + 1);
                    }
                }
                if p.angle > 1.3 {
                    self.land_toppled(p, m, hinge, toward);
                    self.finish_falling(p);
                    return false;
                }
                true
            }
        }
    }

    fn write_falling(&mut self, id: EntityId, m: Mat4) {
        if let Some(e) = self.entities.get_mut(&id) {
            e.transform = m;
            e.bump_xform();
            self.bump_render_epoch();
        }
    }

    /// Crush one storey of a falling part into debris.
    fn eat_floor(&mut self, p: &mut FallingPart, floor: usize) {
        let plan = p.plan.clone();
        let entity_m = self
            .entities
            .get(&p.entity)
            .map_or(p.transform, |e| e.transform);
        let mut picked = 0;
        let mut h = crate::building::Hash(p.seed ^ floor as u64);
        for &i in &p.elements {
            let e = &plan.elements()[i as usize];
            if e.floor as usize != floor {
                continue;
            }
            self.element_states.set(p.base + i, true);
            if picked < 6 && h.unit() < 0.35 {
                picked += 1;
                let out = e.outward * 3.0 + Vec3::Y;
                self.spawn_element_debris(&plan, entity_m, i, out, picked);
            }
        }
        // Crush what is around the base of the fall.
        let c = plan.centroid();
        let at = p.transform.transform_point3(Vec3::new(c.x, p.base_y, c.y));
        self.blast(at, p.radius + 4.0, 4.0);
    }

    fn land_toppled(&mut self, p: &mut FallingPart, m: Mat4, hinge: Vec3, toward: Vec3) {
        let plan = p.plan.clone();
        let mut h = crate::building::Hash(p.seed ^ 0x70);
        let mut n = 0;
        for &i in &p.elements {
            if p.base + i == 0 || self.element_states.get(p.base + i) {
                continue;
            }
            if h.unit() < 0.12 && n < 40 {
                n += 1;
                let e = &plan.elements()[i as usize];
                let along = (e.center - hinge).dot(Vec3::Y).max(0.0);
                let out = toward * (p.spin * along * 0.3) - Vec3::Y * 2.0;
                self.spawn_element_debris(&plan, m, i, out, n);
            }
        }
        let tip = p
            .transform
            .transform_point3(hinge + toward * p.height * 0.5);
        self.blast(tip, p.radius + 3.0, 5.0);
    }

    fn finish_falling(&mut self, p: &mut FallingPart) {
        self.despawn(p.entity);
        self.element_states.release(p.base, p.plan.len() as u32);
        self.place_mounds(p.building);
    }

    /// Show the building's rubble mounds (fixed bodies too, so things rest on them).
    fn place_mounds(&mut self, id: EntityId) {
        let Some(b) = self.buildings.get_mut(&id) else {
            return;
        };
        let old = std::mem::take(&mut b.mounds);
        let mounds = b.state.mounds().to_vec();
        for m in old {
            self.despawn(m);
        }
        let transform = self
            .entities
            .get(&id)
            .map_or(Mat4::IDENTITY, |e| e.transform);
        let mut made = Vec::new();
        for m in mounds {
            let Ok(mesh) = m.mesh(crate::color::Color::rgb(138, 130, 121)) else {
                continue;
            };
            let at = transform * Mat4::from_translation(m.center) * Mat4::from_rotation_y(m.yaw);
            let e = self.spawn_built(mesh.build(), at);
            let _ = self.add_body(e, crate::physics::Body::fixed());
            made.push(e);
        }
        if let Some(b) = self.buildings.get_mut(&id) {
            b.mounds = made;
        }
    }

    /// A broken element flies apart into a few pre-fractured pieces.
    fn spawn_element_debris(
        &mut self,
        plan: &BuildingPlan,
        transform: Mat4,
        element: u32,
        push: Vec3,
        salt: usize,
    ) {
        let e = &plan.elements()[element as usize];
        let size = e.half * 2.0;
        if size.min_element() < 0.02 {
            return;
        }
        let color = element_color(plan, element);
        let mut h = crate::building::Hash(self.rubble.seed ^ (element as u64) << 8 ^ salt as u64);
        self.rubble.seed = self.rubble.seed.wrapping_add(0x9e37_79b9);
        let set = &templates()[(h.next() % templates().len() as u64) as usize];
        // Big panels break into more, smaller lumps.
        let lumps = ((size.max_element() / 1.6).ceil() as usize).clamp(1, 3);
        for lump in 0..lumps {
            for frag in set {
                if self.rubble.tokens < 1.0 || self.rubble.frame_budget < 1.0 {
                    return;
                }
                self.rubble.tokens -= 1.0;
                self.rubble.frame_budget -= 1.0;
                let lump_size = size / Vec3::new(lumps as f32, 1.0, 1.0).max(Vec3::ONE)
                    * Vec3::new(1.0, 1.0 / lumps as f32, 1.0);
                let offset = Vec3::new(0.0, (lump as f32 + 0.5) / lumps as f32 - 0.5, 0.0) * size;
                let mut mesh = frag.mesh.clone();
                for p in mesh.positions.iter_mut() {
                    *p *= lump_size;
                }
                for c in mesh.colors.iter_mut() {
                    *c = glam::Vec4::new(c.x * color.x, c.y * color.y, c.z * color.z, 1.0);
                }
                mesh.elements.clear();
                let local = e.center + offset + frag.centroid * lump_size;
                let at = transform.transform_point3(local);
                let (_, rot, _) = transform.to_scale_rotation_translation();
                let id = self.spawn_built(mesh, Mat4::from_rotation_translation(rot, at));
                let spin = Vec3::new(h.unit() - 0.5, h.unit() - 0.5, h.unit() - 0.5) * 6.0;
                let jitter = Vec3::new(h.unit() - 0.5, h.unit() * 0.5, h.unit() - 0.5) * 2.0;
                let body = crate::physics::Body::dynamic()
                    .density(2.0)
                    .friction(0.8)
                    .bounce(0.1)
                    .velocity(push + jitter)
                    .spin(spin);
                if self.add_body(id, body).is_err() {
                    self.despawn(id);
                    continue;
                }
                self.rubble.flying.push_back((id, 0.0));
                if self.rubble.flying.len() > MAX_FLYING {
                    if let Some((old, _)) = self.rubble.flying.pop_front() {
                        self.physics.remove(old);
                        self.rubble.resting.push_back(old);
                    }
                }
            }
        }
    }

    /// Forget a building's extra state when its entity goes.
    pub(crate) fn forget_building(&mut self, id: EntityId) {
        if let Some(b) = self.buildings.remove(&id) {
            for r in &b.rooms {
                self.rooms.remove_room(*r);
            }
            self.entity_rooms.retain(|_, r| !b.rooms.contains(r));
            self.element_states
                .release(b.base, b.state.plan().len() as u32);
            self.physics.remove_statics_of(b.owner);
            for m in b.mounds {
                self.despawn(m);
            }
        }
        let falling: Vec<EntityId> = self
            .falling
            .iter()
            .filter(|p| p.building == id)
            .map(|p| p.entity)
            .collect();
        self.falling.retain(|p| p.building != id);
        for f in falling {
            self.despawn(f);
        }
    }
}

/// Average colour of an element's vertices.
fn element_color(plan: &BuildingPlan, element: u32) -> glam::Vec4 {
    let m = plan.mesh();
    let mut sum = glam::Vec4::ZERO;
    let mut n = 0.0;
    for (v, &e) in m.elements.iter().enumerate() {
        if e == element + 1 {
            sum += m.colors[v];
            n += 1.0;
            if n >= 24.0 {
                break;
            }
        }
    }
    if n > 0.0 {
        sum / n
    } else {
        glam::Vec4::new(0.6, 0.58, 0.55, 1.0)
    }
}

/// Element-state ranges per building, for tests.
#[cfg(test)]
pub(crate) fn element_base(world: &World, id: EntityId) -> u32 {
    world.buildings[&id].base
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::TowerSpec;

    fn bits_of(world: &World, id: EntityId) -> Vec<bool> {
        let base = element_base(world, id);
        let n = world.building(id).unwrap().plan().len() as u32;
        (0..n).map(|i| world.element_states.get(base + i)).collect()
    }

    fn mound_places(world: &World, id: EntityId) -> Vec<Mat4> {
        world.buildings[&id]
            .mounds
            .iter()
            .map(|m| world.entity(*m).unwrap().transform())
            .collect()
    }

    #[test]
    fn a_blown_out_front_brings_the_tower_down_into_rubble() {
        let plan = Arc::new(BuildingPlan::tower(&TowerSpec::default()).unwrap());
        let mut world = World::new();
        let id = world
            .spawn_building(plan.clone(), Place::new(10.0, 0.0, 5.0))
            .unwrap();
        let mut total = BuildingChange::default();
        for x in [-6.0f32, -3.0, 0.0, 3.0, 6.0] {
            let c = world
                .hit_building(id, Vec3::new(10.0 + x, 1.7, -1.0), 2.5, 1.0e6)
                .unwrap();
            total.broken.extend(c.broken);
            total.collapse = total.collapse.or(c.collapse);
        }
        assert!(
            total.collapse.is_some(),
            "the front is gone: it must come down"
        );
        assert_eq!(world.falling.len(), 1);
        // The GPU bits hide exactly what's gone.
        let gone = world.building(id).unwrap().gone();
        let bits = bits_of(&world, id);
        for (i, b) in bits.iter().enumerate() {
            assert_eq!(*b, gone.get(i), "element {i}");
        }
        assert!(world.rubble_count().0 > 0, "debris flies");
        for _ in 0..(8.0 * 60.0) as usize {
            world.tick_physics(1.0 / 60.0);
            world.tick_buildings(1.0 / 60.0);
        }
        assert!(world.falling.is_empty(), "the fall is over");
        assert!(
            !world.buildings[&id].mounds.is_empty(),
            "rubble mounds lie there"
        );
        let (flying, resting) = world.rubble_count();
        assert!(flying + resting > 0);
        assert!(flying <= MAX_FLYING);

        // A second world takes over the bits without animation: same
        // building, same mounds.
        let mut other = World::new();
        let id2 = other
            .spawn_building(plan, Place::new(10.0, 0.0, 5.0))
            .unwrap();
        other
            .apply_building_state(id2, world.building(id).unwrap().broken(), false)
            .unwrap();
        assert_eq!(bits_of(&other, id2), bits_of(&world, id));
        assert_eq!(mound_places(&other, id2), mound_places(&world, id));
        assert!(other.falling.is_empty());
        assert_eq!(other.rubble_count(), (0, 0));

        // Despawning frees everything.
        world.despawn(id);
        assert!(world.building(id).is_none());
    }

    #[test]
    fn element_ranges_are_reused() {
        let mut s = ElementStates::default();
        let a = s.alloc(10).unwrap();
        let b = s.alloc(5).unwrap();
        assert!(a >= 1 && b == a + 10);
        s.set(a + 3, true);
        s.release(a, 10);
        let c = s.alloc(8).unwrap();
        assert_eq!(c, a);
        assert!(!s.get(a + 3), "reused ids start alive");
    }
}
