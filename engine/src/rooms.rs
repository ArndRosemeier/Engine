//! Rooms and openings: indoor visibility culling.
//!
//! Interiors are expensive and mostly hidden. Give the world **rooms** (boxes)
//! and the **openings** between them (doors, windows, holes) and put what is
//! inside a room into it. Each frame the engine walks from the room the
//! camera is in (or from outside) through the openings it can see, narrowing
//! the view to each opening's outline on screen, and draws only the rooms it
//! reaches. From the street that is the rooms seen through open windows and
//! doors; inside, the room you are in and what its doorways show.
//!
//! ```ignore
//! let hall = world.add_room(Room::new(Vec3::new(0.0, 1.5, 0.0), Vec3::new(4.0, 1.5, 3.0)))?;
//! let kitchen = world.add_room(Room::new(Vec3::new(7.0, 1.5, 0.0), Vec3::new(3.0, 1.5, 3.0)))?;
//! world.add_opening(Opening::door(hall, Some(kitchen), quad), true)?; // a doorway
//! world.add_opening(Opening::door(hall, None, window), true)?;       // a window to the street
//! world.set_room(table, Some(hall))?;
//! // A wall blown open later:
//! world.set_opening_open(hole, true)?;
//! ```
//!
//! Breakable buildings ([`crate::building`]) make their rooms and openings
//! themselves: windows are open, and every wall panel or floor tile that
//! breaks opens up the rooms on either side.
//!
//! Anything without a room is drawn as usual. Rooms cull meshes and animated
//! characters (and their shadows); instanced meshes ignore rooms (merge a
//! room's furniture into one mesh instead: it is also faster to draw).

use crate::error::{EngineError, EngineResult};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

/// A room's handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RoomId(pub(crate) u32);

impl RoomId {
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// An opening's handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpeningId(pub(crate) u32);

impl OpeningId {
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// A room: a box, optionally turned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    pub center: Vec3,
    pub half: Vec3,
    pub rotation: Quat,
}

impl Room {
    pub fn new(center: impl Into<Vec3>, half: impl Into<Vec3>) -> Self {
        Self {
            center: center.into(),
            half: half.into(),
            rotation: Quat::IDENTITY,
        }
    }

    pub fn rotated(mut self, rotation: Quat) -> Self {
        self.rotation = rotation.normalize();
        self
    }

    /// Is `p` inside the box?
    pub fn contains(&self, p: Vec3) -> bool {
        let l = self.rotation.inverse() * (p - self.center);
        l.abs().cmple(self.half).all()
    }

    pub fn volume(&self) -> f32 {
        self.half.x * self.half.y * self.half.z * 8.0
    }

    pub(crate) fn validate(&self) -> EngineResult<()> {
        if self.center.is_finite()
            && self.half.is_finite()
            && self.half.min_element() > 0.0
            && self.rotation.is_finite()
        {
            Ok(())
        } else {
            Err(EngineError::InvalidValue(
                "room needs a finite centre and positive half sizes".into(),
            ))
        }
    }
}

/// A gap between room `a` and room `b` (`None`: the outside), outlined by a
/// quad in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opening {
    pub a: RoomId,
    pub b: Option<RoomId>,
    pub corners: [Vec3; 4],
}

impl Opening {
    pub fn door(a: RoomId, b: Option<RoomId>, corners: [Vec3; 4]) -> Self {
        Self { a, b, corners }
    }

    /// A rectangle `width` × `height` standing on `bottom_center`, facing
    /// `normal` (horizontal).
    pub fn rect(
        a: RoomId,
        b: Option<RoomId>,
        bottom_center: Vec3,
        normal: Vec3,
        width: f32,
        height: f32,
    ) -> Self {
        let n = Vec3::new(normal.x, 0.0, normal.z).normalize_or(Vec3::Z);
        let side = Vec3::Y.cross(n) * (width * 0.5);
        let up = Vec3::Y * height;
        let c = bottom_center;
        Self {
            a,
            b,
            corners: [c - side, c + side, c + side + up, c - side + up],
        }
    }

    pub(crate) fn validate(&self) -> EngineResult<()> {
        if self.corners.iter().all(|c| c.is_finite()) && Some(self.a) != self.b {
            Ok(())
        } else {
            Err(EngineError::InvalidValue(
                "opening needs finite corners between two different rooms".into(),
            ))
        }
    }
}

/// A screen rectangle in NDC.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rect {
    lo: Vec2,
    hi: Vec2,
}

impl Rect {
    pub const FULL: Rect = Rect {
        lo: Vec2::splat(-1.0),
        hi: Vec2::splat(1.0),
    };

    fn intersect(self, o: Rect) -> Option<Rect> {
        let lo = self.lo.max(o.lo);
        let hi = self.hi.min(o.hi);
        (lo.x < hi.x && lo.y < hi.y).then_some(Rect { lo, hi })
    }

    fn contains(self, o: Rect) -> bool {
        self.lo.cmple(o.lo).all() && self.hi.cmpge(o.hi).all()
    }

    fn union(self, o: Rect) -> Rect {
        Rect {
            lo: self.lo.min(o.lo),
            hi: self.hi.max(o.hi),
        }
    }
}

/// The opening's outline on screen, or `None` when it is behind the eye.
/// An outline crossing the eye's plane covers the whole screen.
fn project(vp: Mat4, corners: &[Vec3; 4]) -> Option<Rect> {
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    let mut behind = 0;
    for c in corners {
        let p = vp * Vec4::new(c.x, c.y, c.z, 1.0);
        if p.w <= 1e-4 {
            behind += 1;
            continue;
        }
        let ndc = Vec2::new(p.x, p.y) / p.w;
        lo = lo.min(ndc);
        hi = hi.max(ndc);
    }
    match behind {
        4 => None,
        0 => Some(Rect { lo, hi }),
        _ => Some(Rect::FULL),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OpeningSlot {
    pub opening: Opening,
    pub open: bool,
    /// The building that made it (for debugging and future bulk edits).
    #[allow(dead_code)]
    pub owner: Option<u64>,
}

/// All rooms and openings, and what the last frame could see.
#[derive(Debug, Default)]
pub(crate) struct Rooms {
    pub rooms: Vec<Option<(Room, Option<u64>)>>,
    pub openings: Vec<Option<OpeningSlot>>,
    /// Openings touching each room (by room index), and the outside.
    pub links: Vec<Vec<u32>>,
    pub outside: Vec<u32>,
    pub free_rooms: Vec<u32>,
    pub free_openings: Vec<u32>,
    /// Rooms seen last frame (by index).
    pub visible: Vec<bool>,
    pub enabled: bool,
    pub computed: bool,
    /// Set by the renderer while it draws views other than the camera's
    /// (reflection probes), which must see every room.
    pub bypass: std::sync::atomic::AtomicBool,
}

impl Rooms {
    pub fn new() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    pub fn add_room(&mut self, room: Room, owner: Option<u64>) -> RoomId {
        let i = match self.free_rooms.pop() {
            Some(i) => {
                self.rooms[i as usize] = Some((room, owner));
                self.links[i as usize].clear();
                i
            }
            None => {
                self.rooms.push(Some((room, owner)));
                self.links.push(Vec::new());
                self.visible.push(true);
                (self.rooms.len() - 1) as u32
            }
        };
        self.visible[i as usize] = true;
        RoomId(i)
    }

    pub fn room(&self, id: RoomId) -> Option<&Room> {
        self.rooms.get(id.0 as usize)?.as_ref().map(|r| &r.0)
    }

    pub fn remove_room(&mut self, id: RoomId) {
        if self.room(id).is_none() {
            return;
        }
        let touching: Vec<u32> = self.links[id.0 as usize].clone();
        for o in touching {
            self.remove_opening(OpeningId(o));
        }
        self.rooms[id.0 as usize] = None;
        self.visible[id.0 as usize] = true;
        self.free_rooms.push(id.0);
    }

    pub fn add_opening(
        &mut self,
        opening: Opening,
        open: bool,
        owner: Option<u64>,
    ) -> EngineResult<OpeningId> {
        opening.validate()?;
        if self.room(opening.a).is_none() || opening.b.is_some_and(|b| self.room(b).is_none()) {
            return Err(EngineError::InvalidValue(
                "opening to a room that does not exist".into(),
            ));
        }
        let slot = OpeningSlot {
            opening,
            open,
            owner,
        };
        let i = match self.free_openings.pop() {
            Some(i) => {
                self.openings[i as usize] = Some(slot);
                i
            }
            None => {
                self.openings.push(Some(slot));
                (self.openings.len() - 1) as u32
            }
        };
        self.links[opening.a.0 as usize].push(i);
        match opening.b {
            Some(b) => self.links[b.0 as usize].push(i),
            None => self.outside.push(i),
        }
        Ok(OpeningId(i))
    }

    pub fn remove_opening(&mut self, id: OpeningId) {
        let Some(Some(slot)) = self.openings.get(id.0 as usize).copied() else {
            return;
        };
        let o = slot.opening;
        self.links[o.a.0 as usize].retain(|x| *x != id.0);
        match o.b {
            Some(b) => self.links[b.0 as usize].retain(|x| *x != id.0),
            None => self.outside.retain(|x| *x != id.0),
        }
        self.openings[id.0 as usize] = None;
        self.free_openings.push(id.0);
    }

    pub fn set_open(&mut self, id: OpeningId, open: bool) -> EngineResult<()> {
        let slot = self
            .openings
            .get_mut(id.0 as usize)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| EngineError::InvalidValue("no such opening".into()))?;
        slot.open = open;
        Ok(())
    }

    /// The smallest room containing `p`.
    pub fn room_at(&self, p: Vec3) -> Option<RoomId> {
        self.rooms
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().map(|r| (i, r.0)))
            .filter(|(_, r)| r.contains(p))
            .min_by(|a, b| a.1.volume().total_cmp(&b.1.volume()))
            .map(|(i, _)| RoomId(i as u32))
    }

    pub fn count(&self) -> (usize, usize) {
        (
            self.rooms.iter().filter(|r| r.is_some()).count(),
            self.openings.iter().filter(|o| o.is_some()).count(),
        )
    }

    fn owner_of(&self, room: usize) -> u64 {
        self.rooms[room]
            .as_ref()
            .and_then(|r| r.1)
            .unwrap_or(u64::MAX - room as u64)
    }

    /// Which rooms an eye at `eye` with view-projection `vp` sees.
    ///
    /// From the room the eye is in (or outside), through every open opening
    /// whose outline is on screen, faces the eye and is not hidden behind
    /// another building's rooms, narrowing the view to the outline each time.
    pub fn visibility(&self, eye: Vec3, vp: Mat4) -> Vec<bool> {
        let n = self.rooms.len();
        let mut seen = vec![false; n];
        let occluders = Occluders::build(self, vp);
        // Screen area already explored behind each room (and outside, last).
        let mut explored: Vec<Option<Rect>> = vec![None; n + 1];
        let start = self.room_at(eye).map(|r| r.0 as usize);
        let node_of = |r: Option<usize>| r.unwrap_or(n);
        if let Some(r) = start {
            seen[r] = true;
        }
        let mut stack = vec![(start, Rect::FULL, 0u32)];
        explored[node_of(start)] = Some(Rect::FULL);
        while let Some((node, rect, depth)) = stack.pop() {
            let links = match node {
                Some(r) => &self.links[r],
                None => &self.outside,
            };
            for &o in links {
                let Some(slot) = self.openings[o as usize].as_ref() else {
                    continue;
                };
                if !slot.open {
                    continue;
                }
                let op = slot.opening;
                let other = if Some(op.a.0 as usize) == node {
                    op.b.map(|b| b.0 as usize)
                } else {
                    Some(op.a.0 as usize)
                };
                if other == node {
                    continue;
                }
                // Rooms are convex: looking from `node` into `other`, the
                // eye is on `node`'s side of the opening.
                let c = (op.corners[0] + op.corners[1] + op.corners[2] + op.corners[3]) * 0.25;
                let normal = (op.corners[1] - op.corners[0])
                    .cross(op.corners[3] - op.corners[0])
                    .normalize_or_zero();
                let far_side = match other {
                    Some(i) => self.rooms[i].as_ref().map(|r| r.0.center),
                    None => node
                        .and_then(|i| self.rooms[i].as_ref())
                        .map(|r| 2.0 * c - r.0.center),
                };
                if let Some(far) = far_side {
                    let into = normal.dot(far - c);
                    let eye_side = normal.dot(eye - c);
                    if into * eye_side > 0.0 && eye_side.abs() > 0.05 {
                        continue;
                    }
                }
                let Some(r) = project(vp, &op.corners).and_then(|r| r.intersect(rect)) else {
                    continue;
                };
                let owner = self.owner_of(op.a.0 as usize);
                if !occluders.shows(vp, &op.corners, r, owner) {
                    continue;
                }
                if let Some(i) = other {
                    seen[i] = true;
                }
                let k = node_of(other);
                if explored[k].is_some_and(|e| e.contains(r)) || depth >= 24 {
                    continue;
                }
                explored[k] = Some(explored[k].map_or(r, |e| e.union(r)));
                stack.push((other, r, depth + 1));
            }
        }
        seen
    }
}

/// A small software depth buffer of every room box, so a window behind
/// another building is known to be hidden. Each pixel keeps the nearest
/// depth and its owner (building), and the nearest depth of any other owner:
/// a building never hides its own openings.
struct Occluders {
    near: Vec<(f32, u64)>,
    other: Vec<f32>,
}

const OCC_W: usize = 128;
const OCC_H: usize = 72;

fn to_pixels(r: Rect) -> (usize, usize, usize, usize) {
    let px = |x: f32, size: usize| ((x * 0.5 + 0.5) * size as f32).clamp(0.0, size as f32);
    let (x0, x1) = (
        px(r.lo.x, OCC_W).floor() as usize,
        px(r.hi.x, OCC_W).ceil() as usize,
    );
    let (y0, y1) = (
        px(r.lo.y, OCC_H).floor() as usize,
        px(r.hi.y, OCC_H).ceil() as usize,
    );
    (x0, x1.max(x0 + 1).min(OCC_W), y0, y1.max(y0 + 1).min(OCC_H))
}

impl Occluders {
    fn build(rooms: &Rooms, vp: Mat4) -> Self {
        let mut me = Self {
            near: vec![(0.0, 0); OCC_W * OCC_H],
            other: vec![0.0; OCC_W * OCC_H],
        };
        const FACES: [[usize; 4]; 6] = [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 3, 7, 5],
        ];
        for (i, r) in rooms.rooms.iter().enumerate() {
            let Some((room, _)) = r else { continue };
            let owner = rooms.owner_of(i);
            let mut pts = [(Vec2::ZERO, 0.0f32); 8];
            let mut ok = true;
            for (k, p) in pts.iter_mut().enumerate() {
                let s = Vec3::new(
                    if k & 1 == 0 { -1.0 } else { 1.0 },
                    if k & 2 == 0 { -1.0 } else { 1.0 },
                    if k & 4 == 0 { -1.0 } else { 1.0 },
                );
                let w = room.center + room.rotation * (room.half * s);
                let c = vp * Vec4::new(w.x, w.y, w.z, 1.0);
                if c.w <= 1e-3 {
                    ok = false;
                    break;
                }
                let ndc = Vec2::new(c.x, c.y) / c.w;
                *p = (
                    Vec2::new(
                        (ndc.x * 0.5 + 0.5) * OCC_W as f32,
                        (ndc.y * 0.5 + 0.5) * OCC_H as f32,
                    ),
                    1.0 / c.w,
                );
            }
            if !ok {
                continue;
            }
            for f in FACES {
                me.triangle(pts[f[0]], pts[f[1]], pts[f[2]], owner);
                me.triangle(pts[f[0]], pts[f[2]], pts[f[3]], owner);
            }
        }
        me
    }

    /// Rasterise one triangle (screen position, 1/w) at pixel centres.
    fn triangle(&mut self, a: (Vec2, f32), b: (Vec2, f32), c: (Vec2, f32), owner: u64) {
        let area = (b.0 - a.0).perp_dot(c.0 - a.0);
        if area.abs() < 1e-6 {
            return;
        }
        let lo = a.0.min(b.0).min(c.0).max(Vec2::ZERO);
        let hi =
            a.0.max(b.0)
                .max(c.0)
                .min(Vec2::new(OCC_W as f32, OCC_H as f32));
        if lo.x >= hi.x || lo.y >= hi.y {
            return;
        }
        for y in (lo.y.floor() as usize)..(hi.y.ceil() as usize).min(OCC_H) {
            for x in (lo.x.floor() as usize)..(hi.x.ceil() as usize).min(OCC_W) {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let wa = (b.0 - p).perp_dot(c.0 - p) / area;
                let wb = (c.0 - p).perp_dot(a.0 - p) / area;
                let wc = 1.0 - wa - wb;
                if wa < 0.0 || wb < 0.0 || wc < 0.0 {
                    continue;
                }
                let d = wa * a.1 + wb * b.1 + wc * c.1;
                let i = y * OCC_W + x;
                let (d1, o1) = self.near[i];
                if o1 == owner {
                    self.near[i].0 = d1.max(d);
                } else if d > d1 {
                    self.near[i] = (d, owner);
                    self.other[i] = d1;
                } else {
                    self.other[i] = self.other[i].max(d);
                }
            }
        }
    }

    /// Could an opening of `owner` with these corners show anywhere in `r`?
    fn shows(&self, vp: Mat4, corners: &[Vec3; 4], r: Rect, owner: u64) -> bool {
        let mut nearest = 0.0f32;
        for c in corners {
            let p = vp * Vec4::new(c.x, c.y, c.z, 1.0);
            if p.w <= 1e-3 {
                return true;
            }
            nearest = nearest.max(1.0 / p.w);
        }
        let (x0, x1, y0, y1) = to_pixels(r);
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * OCC_W + x;
                let (d1, o1) = self.near[i];
                let occ = if o1 == owner { self.other[i] } else { d1 };
                if occ <= nearest * 1.0001 {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;

    /// Two rooms side by side along x with a door between, the first with a
    /// window to the street at z = -3.
    fn house() -> (Rooms, RoomId, RoomId, OpeningId, OpeningId) {
        let mut r = Rooms::new();
        let a = r.add_room(Room::new((0.0, 1.5, 0.0), (3.0, 1.5, 3.0)), None);
        let b = r.add_room(Room::new((6.0, 1.5, 0.0), (3.0, 1.5, 3.0)), None);
        let door = r
            .add_opening(
                Opening::rect(a, Some(b), Vec3::new(3.0, 0.0, 0.0), Vec3::X, 1.0, 2.1),
                true,
                None,
            )
            .unwrap();
        let window = r
            .add_opening(
                Opening::rect(a, None, Vec3::new(0.0, 1.0, -3.0), Vec3::Z, 1.5, 1.2),
                true,
                None,
            )
            .unwrap();
        (r, a, b, door, window)
    }

    fn vp(cam: &Camera) -> Mat4 {
        cam.view_projection(16.0 / 9.0)
    }

    #[test]
    fn from_the_street_through_a_window() {
        let (mut r, a, b, door, window) = house();
        let cam = Camera::look_at((0.0, 1.6, -12.0), (0.0, 1.6, 0.0));
        let seen = r.visibility(cam.eye(), vp(&cam));
        assert!(seen[a.0 as usize], "the room behind the window");
        // The door is at the room's far right: from straight in front it is
        // outside the window's outline.
        assert!(!seen[b.0 as usize], "the next room is hidden");
        r.set_open(window, false).unwrap();
        let seen = r.visibility(cam.eye(), vp(&cam));
        assert!(!seen[a.0 as usize] && !seen[b.0 as usize]);
        // Looking away from the window: nothing inside.
        r.set_open(window, true).unwrap();
        let away = Camera::look_at((0.0, 1.6, -12.0), (0.0, 1.6, -30.0));
        assert!(!r.visibility(away.eye(), vp(&away)).iter().any(|v| *v));
        let _ = door;
    }

    #[test]
    fn inside_sees_through_the_door() {
        let (mut r, a, b, door, _) = house();
        let cam = Camera::look_at((-2.0, 1.6, 0.0), (6.0, 1.6, 0.0));
        let seen = r.visibility(cam.eye(), vp(&cam));
        assert!(seen[a.0 as usize] && seen[b.0 as usize]);
        r.set_open(door, false).unwrap();
        let seen = r.visibility(cam.eye(), vp(&cam));
        assert!(seen[a.0 as usize] && !seen[b.0 as usize]);
        // Facing the other way, the door is behind.
        r.set_open(door, true).unwrap();
        let back = Camera::look_at((-2.0, 1.6, 0.0), (-10.0, 1.6, 0.0));
        let seen = r.visibility(back.eye(), vp(&back));
        assert!(seen[a.0 as usize] && !seen[b.0 as usize]);
    }

    #[test]
    fn through_window_room_and_door_to_the_next() {
        let (r, _, b, _, _) = house();
        // From the street at an angle, the door shows through the window.
        let cam = Camera::look_at((-3.0, 1.6, -6.0), (3.0, 1.2, 0.0));
        assert!(r.visibility(cam.eye(), vp(&cam))[b.0 as usize]);
    }

    #[test]
    fn a_building_in_front_hides_the_one_behind() {
        let mut r = Rooms::new();
        let near = r.add_room(Room::new((0.0, 1.5, 0.0), (4.0, 1.5, 3.0)), Some(1));
        let far = r.add_room(Room::new((0.0, 1.5, 12.0), (4.0, 1.5, 3.0)), Some(2));
        // Both have a window toward the camera (at -z), and the far one a
        // window at its back too.
        r.add_opening(
            Opening::rect(far, None, Vec3::new(0.0, 1.0, 9.0), Vec3::Z, 1.5, 1.2),
            true,
            None,
        )
        .unwrap();
        let back = r
            .add_opening(
                Opening::rect(far, None, Vec3::new(2.0, 1.0, 15.0), Vec3::Z, 1.5, 1.2),
                true,
                None,
            )
            .unwrap();
        let cam = Camera::look_at((0.0, 1.6, -12.0), (0.0, 1.6, 12.0));
        let seen = r.visibility(cam.eye(), vp(&cam));
        assert!(!seen[far.0 as usize], "behind the near building");
        assert!(!seen[near.0 as usize], "no opening on the near one");
        // Without the near building the far room shows through its front
        // window, but its back window faces away.
        r.remove_room(near);
        assert!(r.visibility(cam.eye(), vp(&cam))[far.0 as usize]);
        r.set_open(OpeningId(0), false).unwrap();
        assert!(
            !r.visibility(cam.eye(), vp(&cam))[far.0 as usize],
            "a back window faces away"
        );
        let _ = back;
    }

    #[test]
    fn rooms_and_openings_are_reused() {
        let (mut r, a, _, _, _) = house();
        assert_eq!(r.count(), (2, 2));
        r.remove_room(a);
        assert_eq!(r.count(), (1, 0), "openings of a removed room go too");
        let c = r.add_room(Room::new(Vec3::ZERO, Vec3::ONE), None);
        assert_eq!(c, a);
        assert_eq!(r.room_at(Vec3::new(0.5, 0.0, 0.0)), Some(c));
        assert!(Opening::door(c, Some(c), [Vec3::ZERO; 4])
            .validate()
            .is_err());
    }
}
