//! Buildings that break element by element and collapse.
//!
//! A building is a list of **elements**: wall panels (one per window bay per
//! floor), slab tiles, columns, roof pieces and interior walls, each a box in
//! the building's own space with a strength. Breaking one is a state change,
//! a bit in [`ElementBits`]; the renderer hides dead elements in the vertex
//! shader, so nothing is re-uploaded.
//!
//! After damage the structure is checked floor by floor, following Scale's
//! rules: a storey that loses half its load-bearing perimeter, or a long run
//! of it on one side, fails, and everything above comes down. Even damage
//! pancakes the floors above one after another; damage on one side topples
//! the upper part around a hinge on that side. Slab tiles stand while a wall
//! or column holds them up, or a held tile is at most two tiles away.
//!
//! Everything here is plain data and runs without a window: [`Building`] is
//! what a game server keeps. The **broken bits** are the only state; the fallen
//! elements, the collapse and the rubble mounds follow from them, so
//! [`Building::apply`] on a fresh building gives the same building and the
//! same rubble on every machine. The falling chunks, debris and dust are
//! cosmetic and drawn by [`World`](crate::world::World) (see
//! [`World::spawn_building`](crate::world::World::spawn_building)).
//!
//! ```
//! use engine::building::{Building, BuildingPlan, TowerSpec};
//! use engine::glam::Vec3;
//! use std::sync::Arc;
//!
//! let plan = Arc::new(BuildingPlan::tower(&TowerSpec::default()).unwrap());
//! let mut server = Building::new(plan.clone());
//! // A wrecking ball through the ground floor's front.
//! for x in [-6.0, -3.0, 0.0, 3.0, 6.0] {
//!     server.hit(Vec3::new(x, 1.7, -6.0), 2.5, 1.0e6);
//! }
//! let mut client = Building::new(plan);
//! client.apply(server.broken());
//! assert_eq!(client.fallen(), server.fallen());
//! assert_eq!(client.mounds(), server.mounds());
//! ```

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::mesh::{BuiltMesh, Mesh};
use glam::{Vec2, Vec3};
use std::sync::Arc;

/// What an element is; decides how it carries load and its default strength.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ElementKind {
    /// An outer wall panel: carries the floors above.
    Wall,
    /// An outer panel that is mostly glass: carries a little.
    Window,
    /// An inner column: carries the floors above and holds slabs up.
    Column,
    /// A piece of floor slab, standing on walls and columns below.
    Slab,
    /// A piece of roof, like a slab on top.
    Roof,
    /// An inner wall: carries nothing, breaks easily.
    Partition,
}

impl ElementKind {
    /// Impact power (roughly kJ) that breaks it at the centre of a hit.
    pub fn default_strength(self) -> f32 {
        match self {
            Self::Wall => 400.0,
            Self::Window => 120.0,
            Self::Column => 900.0,
            Self::Slab => 600.0,
            Self::Roof => 400.0,
            Self::Partition => 60.0,
        }
    }

    /// How much of the storey's load this carries per metre of width.
    fn carries(self) -> f32 {
        match self {
            Self::Wall => 1.0,
            Self::Window => 0.6,
            Self::Column => 1.0,
            _ => 0.0,
        }
    }

    fn holds_slabs(self) -> bool {
        matches!(self, Self::Wall | Self::Window | Self::Column)
    }

    fn is_slab(self) -> bool {
        matches!(self, Self::Slab | Self::Roof)
    }
}

/// Index of an element in its [`BuildingPlan`].
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ElementId(pub u32);

/// One breakable piece: a box in the building's space.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    pub kind: ElementKind,
    /// The storey it belongs to (slabs: the storey they are the floor of).
    pub floor: u16,
    pub center: Vec3,
    pub half: Vec3,
    /// Impact power that breaks it (see [`ElementKind::default_strength`]).
    pub strength: f32,
    /// Outer panels: the facade's outward direction. Zero inside.
    pub outward: Vec3,
}

impl Element {
    pub fn new(kind: ElementKind, floor: u16, center: Vec3, half: Vec3) -> Self {
        Self {
            kind,
            floor,
            center,
            half,
            strength: kind.default_strength(),
            outward: Vec3::ZERO,
        }
    }

    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    pub fn with_outward(mut self, outward: Vec3) -> Self {
        self.outward = outward.normalize_or_zero();
        self
    }

    fn distance_to(&self, p: Vec3) -> f32 {
        let d = (p - self.center).abs() - self.half;
        d.max(Vec3::ZERO).length()
    }

    /// Horizontal width (the longer horizontal side).
    fn width(&self) -> f32 {
        2.0 * self.half.x.max(self.half.z)
    }
}

/// A storey: its floor and ceiling height in the building's space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor {
    pub y0: f32,
    pub y1: f32,
}

/// A generic tower: a box of storeys with a panel per bay, slab tiles,
/// columns, a roof and a few inner walls. Good for tests and for a city
/// that has no buildings of its own yet.
#[derive(Clone, Debug, PartialEq)]
pub struct TowerSpec {
    pub width: f32,
    pub depth: f32,
    pub floors: u16,
    pub floor_height: f32,
    /// Width of one window bay (one wall panel).
    pub bay: f32,
    /// Side of a slab tile.
    pub tile: f32,
    /// Share of panels that are glass windows (0..1).
    pub glazing: f32,
    pub wall_color: Color,
    pub glass_color: Color,
    pub slab_color: Color,
    pub seed: u64,
}

impl Default for TowerSpec {
    fn default() -> Self {
        Self {
            width: 15.0,
            depth: 12.0,
            floors: 6,
            floor_height: 3.4,
            bay: 3.0,
            tile: 3.0,
            glazing: 0.0,
            wall_color: Color::rgb(196, 186, 170),
            glass_color: Color::rgb(70, 96, 120),
            slab_color: Color::rgb(150, 146, 140),
            seed: 1,
        }
    }
}

/// The parts of a building and what they look like. Make one with
/// [`BuildingPlan::tower`] or element by element with [`BuildingPlan::new`]
/// and [`BuildingPlan::add`], then share it (`Arc`) between every building
/// that looks the same.
#[derive(Clone, Debug)]
pub struct BuildingPlan {
    elements: Vec<Element>,
    floors: Vec<Floor>,
    /// Geometry of all elements; `mesh.elements` holds local index + 1.
    mesh: BuiltMesh,
    seed: u64,
    /// Outer panels of each storey in order around the footprint.
    rings: Vec<Vec<u32>>,
    footprint_min: Vec2,
    footprint_max: Vec2,
    /// Slab tiles of each storey.
    slabs: Vec<Vec<u32>>,
    tile: f32,
    /// Rooms inside (boxes in the building's space), see [`crate::rooms`].
    rooms: Vec<(Vec3, Vec3)>,
    /// Gaps that are always open (doorways): rooms on each side (`None`
    /// outside) and the outline.
    gaps: Vec<PlanGap>,
}

/// A doorway in a plan: always open.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanGap {
    pub a: u16,
    pub b: Option<u16>,
    pub corners: [Vec3; 4],
}

impl BuildingPlan {
    /// An empty plan with these storeys (bottom first).
    pub fn new(floors: Vec<Floor>, seed: u64) -> EngineResult<Self> {
        if floors.is_empty() {
            return Err(EngineError::InvalidValue("a building needs a floor".into()));
        }
        for w in floors.windows(2) {
            if !(w[0].y1 <= w[1].y0 + 1e-3) {
                return Err(EngineError::InvalidValue(
                    "floors must go up without overlapping".into(),
                ));
            }
        }
        if floors.iter().any(|f| !(f.y0.is_finite() && f.y1 > f.y0)) {
            return Err(EngineError::InvalidValue(
                "each floor needs y1 above y0".into(),
            ));
        }
        Ok(Self {
            elements: Vec::new(),
            floors,
            mesh: BuiltMesh::default(),
            seed,
            rings: Vec::new(),
            footprint_min: Vec2::splat(f32::MAX),
            footprint_max: Vec2::splat(f32::MIN),
            slabs: Vec::new(),
            tile: 3.0,
            rooms: Vec::new(),
            gaps: Vec::new(),
        })
    }

    /// Add an element. `mesh` is its look in the building's space (None: a
    /// box of `color` filling the element). Triangles of the mesh vanish
    /// when the element breaks.
    pub fn add(
        &mut self,
        element: Element,
        mesh: Option<&Mesh>,
        color: Color,
    ) -> EngineResult<ElementId> {
        if !(element.center.is_finite() && element.half.is_finite())
            || element.half.min_element() <= 0.0
        {
            return Err(EngineError::InvalidValue(
                "element needs a finite box with positive size".into(),
            ));
        }
        if element.floor as usize >= self.floors.len() + 1 {
            return Err(EngineError::InvalidValue(format!(
                "element floor {} but the plan has {} floors",
                element.floor,
                self.floors.len()
            )));
        }
        if !(element.strength.is_finite() && element.strength > 0.0) {
            return Err(EngineError::InvalidValue(
                "element strength must be > 0".into(),
            ));
        }
        if self.elements.len() >= (1 << 20) {
            return Err(EngineError::InvalidValue(
                "a building has at most 1M elements".into(),
            ));
        }
        let id = self.elements.len() as u32;
        let built = match mesh {
            Some(m) => m.build(),
            None => Mesh::box_at(element.center, element.half * 2.0, color)?.build(),
        };
        append(&mut self.mesh, &built, id + 1);
        let lo = Vec2::new(
            element.center.x - element.half.x,
            element.center.z - element.half.z,
        );
        let hi = Vec2::new(
            element.center.x + element.half.x,
            element.center.z + element.half.z,
        );
        self.footprint_min = self.footprint_min.min(lo);
        self.footprint_max = self.footprint_max.max(hi);
        self.elements.push(element);
        self.index();
        Ok(ElementId(id))
    }

    /// Rebuild the per-storey lists after an element was added.
    fn index(&mut self) {
        let n = self.floors.len() + 1;
        let c = self.centroid();
        let mut rings: Vec<Vec<(f32, u32)>> = vec![Vec::new(); n];
        let mut slabs: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (i, e) in self.elements.iter().enumerate() {
            let f = (e.floor as usize).min(n - 1);
            if e.kind.is_slab() {
                slabs[f].push(i as u32);
            } else if matches!(e.kind, ElementKind::Wall | ElementKind::Window)
                && e.outward != Vec3::ZERO
            {
                let a = (e.center.z - c.y).atan2(e.center.x - c.x);
                rings[f].push((a, i as u32));
            }
        }
        self.rings = rings
            .into_iter()
            .map(|mut r| {
                r.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                r.into_iter().map(|(_, i)| i).collect()
            })
            .collect();
        self.slabs = slabs;
    }

    /// The generic tower (see [`TowerSpec`]), centred on x/z = 0 with its base at y = 0.
    pub fn tower(spec: &TowerSpec) -> EngineResult<Self> {
        let ok = |v: f32| v.is_finite() && v > 0.0;
        if !(ok(spec.width) && ok(spec.depth) && ok(spec.floor_height) && ok(spec.bay))
            || !ok(spec.tile)
            || spec.floors == 0
            || spec.floors > 200
        {
            return Err(EngineError::InvalidValue(
                "tower needs positive sizes and 1..=200 floors".into(),
            ));
        }
        let fh = spec.floor_height;
        let floors = (0..spec.floors)
            .map(|f| Floor {
                y0: f as f32 * fh,
                y1: (f + 1) as f32 * fh,
            })
            .collect();
        let mut plan = Self::new(floors, spec.seed)?;
        plan.tile = spec.tile;
        let (hw, hd) = (spec.width * 0.5, spec.depth * 0.5);
        let wall_t = 0.25;
        let slab_t = 0.3;
        let mut rng = Hash(spec.seed);
        for f in 0..spec.floors {
            let y0 = f as f32 * fh;
            let wall_y = y0 + if f == 0 { 0.0 } else { slab_t };
            let wall_h = fh - if f == 0 { 0.0 } else { slab_t };
            // Outer panels, side by side around the footprint.
            let sides = [
                (Vec3::new(-hw, 0.0, -hd), Vec3::X, spec.width, -Vec3::Z),
                (Vec3::new(hw, 0.0, -hd), Vec3::Z, spec.depth, Vec3::X),
                (Vec3::new(hw, 0.0, hd), -Vec3::X, spec.width, Vec3::Z),
                (Vec3::new(-hw, 0.0, hd), -Vec3::Z, spec.depth, -Vec3::X),
            ];
            for (start, along, len, out) in sides {
                let bays = (len / spec.bay).round().max(1.0) as usize;
                let bw = len / bays as f32;
                for b in 0..bays {
                    let mid = start + along * (bw * (b as f32 + 0.5)) + out * (-wall_t * 0.5);
                    let center = Vec3::new(mid.x, wall_y + wall_h * 0.5, mid.z);
                    let half_along = (bw * 0.5 - 0.005).max(0.05);
                    let half = if along.x.abs() > 0.5 {
                        Vec3::new(half_along, wall_h * 0.5, wall_t * 0.5)
                    } else {
                        Vec3::new(wall_t * 0.5, wall_h * 0.5, half_along)
                    };
                    let glass = rng.unit() < spec.glazing;
                    let kind = if glass {
                        ElementKind::Window
                    } else {
                        ElementKind::Wall
                    };
                    let e = Element::new(kind, f, center, half).with_outward(out);
                    let mesh = panel_mesh(&e, along, out, glass, spec)?;
                    plan.add(e, Some(&mesh), spec.wall_color)?;
                }
            }
            // Slab tiles (the ground floor stands on the ground).
            let nx = (spec.width / spec.tile).round().max(1.0) as usize;
            let nz = (spec.depth / spec.tile).round().max(1.0) as usize;
            let (tx, tz) = (spec.width / nx as f32, spec.depth / nz as f32);
            if f > 0 {
                for i in 0..nx {
                    for j in 0..nz {
                        let c = Vec3::new(
                            -hw + tx * (i as f32 + 0.5),
                            y0 + slab_t * 0.5,
                            -hd + tz * (j as f32 + 0.5),
                        );
                        let e = Element::new(
                            ElementKind::Slab,
                            f,
                            c,
                            Vec3::new(tx * 0.5 - 0.004, slab_t * 0.5, tz * 0.5 - 0.004),
                        );
                        plan.add(e, None, spec.slab_color)?;
                    }
                }
            }
            // Columns at every other inner tile corner.
            for i in (2..nx).step_by(2) {
                for j in (1..nz).step_by(2) {
                    let c = Vec3::new(
                        -hw + tx * i as f32,
                        wall_y + wall_h * 0.5,
                        -hd + tz * j as f32,
                    );
                    let e = Element::new(
                        ElementKind::Column,
                        f,
                        c,
                        Vec3::new(0.25, wall_h * 0.5, 0.25),
                    );
                    plan.add(e, None, dim(spec.slab_color, 0.9))?;
                }
            }
            // Rooms: the front and back halves (or one room when shallow).
            let y1 = y0 + fh;
            let (ry, rh) = ((y0 + y1) * 0.5, fh * 0.5);
            if spec.depth >= 6.0 {
                let front =
                    plan.add_room(Vec3::new(0.0, ry, -hd * 0.5), Vec3::new(hw, rh, hd * 0.5))?;
                let back =
                    plan.add_room(Vec3::new(0.0, ry, hd * 0.5), Vec3::new(hw, rh, hd * 0.5))?;
                let segs = nx.max(2);
                let w = spec.width / segs as f32;
                let x = -hw + w * (segs / 2) as f32;
                let (yb, yt) = (wall_y, wall_y + wall_h);
                plan.add_gap(PlanGap {
                    a: front,
                    b: Some(back),
                    corners: [
                        Vec3::new(x, yb, 0.0),
                        Vec3::new(x + w, yb, 0.0),
                        Vec3::new(x + w, yt, 0.0),
                        Vec3::new(x, yt, 0.0),
                    ],
                })?;
            } else {
                plan.add_room(Vec3::new(0.0, ry, 0.0), Vec3::new(hw, rh, hd))?;
            }
            // An inner wall across the middle, with a door gap.
            if spec.depth >= 6.0 {
                let segs = nx.max(2);
                for i in 0..segs {
                    if i == segs / 2 {
                        continue;
                    }
                    let w = spec.width / segs as f32;
                    let c = Vec3::new(-hw + w * (i as f32 + 0.5), wall_y + wall_h * 0.5, 0.0);
                    let e = Element::new(
                        ElementKind::Partition,
                        f,
                        c,
                        Vec3::new(w * 0.5 - 0.01, wall_h * 0.5 - 0.01, 0.06),
                    );
                    plan.add(e, None, Color::rgb(220, 214, 200))?;
                }
            }
        }
        // Roof tiles.
        let top = spec.floors as f32 * fh;
        let nx = (spec.width / spec.tile).round().max(1.0) as usize;
        let nz = (spec.depth / spec.tile).round().max(1.0) as usize;
        let (tx, tz) = (spec.width / nx as f32, spec.depth / nz as f32);
        for i in 0..nx {
            for j in 0..nz {
                let c = Vec3::new(
                    -hw + tx * (i as f32 + 0.5),
                    top + 0.2,
                    -hd + tz * (j as f32 + 0.5),
                );
                let e = Element::new(
                    ElementKind::Roof,
                    spec.floors,
                    c,
                    Vec3::new(tx * 0.5 - 0.004, 0.2, tz * 0.5 - 0.004),
                );
                plan.add(e, None, dim(spec.slab_color, 0.8))?;
            }
        }
        Ok(plan)
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    /// Add a room (a box in the building's space). Elements between two
    /// rooms, or a room and the outside, become openings when they break
    /// (windows are open from the start), so what is inside is drawn only
    /// when it can be seen. Returns the room's index in the plan.
    pub fn add_room(&mut self, center: Vec3, half: Vec3) -> EngineResult<u16> {
        if !(center.is_finite() && half.is_finite() && half.min_element() > 0.0) {
            return Err(EngineError::InvalidValue(
                "room needs a box with positive size".into(),
            ));
        }
        if self.rooms.len() >= u16::MAX as usize {
            return Err(EngineError::InvalidValue("too many rooms".into()));
        }
        self.rooms.push((center, half));
        Ok((self.rooms.len() - 1) as u16)
    }

    /// A doorway that is always open between room `a` and room `b` (`None`: outside).
    pub fn add_gap(&mut self, gap: PlanGap) -> EngineResult<()> {
        let ok = (gap.a as usize) < self.rooms.len()
            && gap
                .b
                .is_none_or(|b| (b as usize) < self.rooms.len() && b != gap.a)
            && gap.corners.iter().all(|c| c.is_finite());
        if !ok {
            return Err(EngineError::InvalidValue(
                "gap between rooms that are not in the plan".into(),
            ));
        }
        self.gaps.push(gap);
        Ok(())
    }

    /// Rooms as (centre, half size) in the building's space.
    pub fn rooms(&self) -> &[(Vec3, Vec3)] {
        &self.rooms
    }

    pub fn gaps(&self) -> &[PlanGap] {
        &self.gaps
    }

    fn room_at(&self, p: Vec3) -> Option<u16> {
        self.rooms
            .iter()
            .enumerate()
            .filter(|(_, (c, h))| (p - *c).abs().cmple(*h).all())
            .min_by(|a, b| {
                let v = |h: Vec3| h.x * h.y * h.z;
                v(a.1 .1).total_cmp(&v(b.1 .1))
            })
            .map(|(i, _)| i as u16)
    }

    /// The rooms on either side of element `i` (across its thinnest side)
    /// and its outline there, when the sides differ: what opens up when it
    /// breaks.
    pub fn element_opening(&self, i: usize) -> Option<(u16, Option<u16>, [Vec3; 4])> {
        let e = self.elements.get(i)?;
        if self.rooms.is_empty() || e.kind == ElementKind::Column {
            return None;
        }
        let h = e.half;
        let axis = if h.x <= h.y && h.x <= h.z {
            0
        } else if h.y <= h.z {
            1
        } else {
            2
        };
        let n = Vec3::AXES[axis];
        let reach = h[axis] + 0.3;
        let a = self.room_at(e.center + n * reach);
        let b = self.room_at(e.center - n * reach);
        let (a, b) = match (a, b) {
            (Some(a), b) if Some(a) != b => (a, b),
            (None, Some(b)) => (b, None),
            _ => return None,
        };
        let (u, v) = (Vec3::AXES[(axis + 1) % 3], Vec3::AXES[(axis + 2) % 3]);
        let (du, dv) = (u * h[(axis + 1) % 3], v * h[(axis + 2) % 3]);
        let c = e.center;
        Some((a, b, [c - du - dv, c + du - dv, c + du + dv, c - du + dv]))
    }

    pub fn element(&self, id: ElementId) -> Option<&Element> {
        self.elements.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn floors(&self) -> &[Floor] {
        &self.floors
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The geometry: `mesh().elements[v]` is the element (index + 1) of vertex `v`.
    pub fn mesh(&self) -> &BuiltMesh {
        &self.mesh
    }

    /// Middle of the footprint (x, z).
    pub fn centroid(&self) -> Vec2 {
        if self.elements.is_empty() {
            return Vec2::ZERO;
        }
        (self.footprint_min + self.footprint_max) * 0.5
    }

    /// Footprint corners (x, z).
    pub fn footprint(&self) -> (Vec2, Vec2) {
        (self.footprint_min, self.footprint_max)
    }

    /// Radius of a circle with the footprint's area.
    fn radius(&self) -> f32 {
        let s = (self.footprint_max - self.footprint_min).max(Vec2::splat(0.1));
        (s.x * s.y / std::f32::consts::PI).sqrt()
    }

    /// The geometry of a set of elements only (for a falling part), with
    /// element ids renumbered by `id_of(local) -> u32` (0 drops the triangle).
    pub(crate) fn extract(&self, mut id_of: impl FnMut(u32) -> u32) -> BuiltMesh {
        let m = &self.mesh;
        let mut out = BuiltMesh {
            surface: m.surface,
            ..Default::default()
        };
        let mut remap = vec![u32::MAX; m.positions.len()];
        let mut keep = |out: &mut BuiltMesh, v: usize, id: u32| -> u32 {
            if remap[v] == u32::MAX {
                remap[v] = out.positions.len() as u32;
                out.positions.push(m.positions[v]);
                out.normals.push(m.normals[v]);
                out.colors.push(m.colors[v]);
                out.uvs.push(m.uvs.get(v).copied().unwrap_or([0.0; 2]));
                if !m.surface_overrides.is_empty() {
                    out.surface_overrides.push(m.surface_overrides[v]);
                }
                out.elements.push(id);
            }
            remap[v]
        };
        for (n, tri) in m.indices.chunks_exact(3).enumerate() {
            let local = m.elements[tri[0] as usize];
            if local == 0 {
                continue;
            }
            let id = id_of(local - 1);
            if id == 0 {
                continue;
            }
            for &v in tri {
                let i = keep(&mut out, v as usize, id);
                out.indices.push(i);
            }
            if n * 3 < m.opaque_index_count {
                out.opaque_index_count = out.indices.len();
            }
        }
        out
    }
}

/// Append `src` to `dst`, tagging its vertices with `element`.
fn append(dst: &mut BuiltMesh, src: &BuiltMesh, element: u32) {
    let base = dst.positions.len() as u32;
    // Opaque triangles first, then transparent ones, as the renderer expects.
    let opaque_src = &src.indices[..src.opaque_index_count.min(src.indices.len())];
    let clear_src = &src.indices[src.opaque_index_count.min(src.indices.len())..];
    let (opaque_dst, clear_dst) = dst
        .indices
        .split_at(dst.opaque_index_count.min(dst.indices.len()));
    let mut indices: Vec<u32> = opaque_dst.to_vec();
    indices.extend(opaque_src.iter().map(|i| i + base));
    let opaque = indices.len();
    indices.extend_from_slice(clear_dst);
    indices.extend(clear_src.iter().map(|i| i + base));
    dst.indices = indices;
    dst.opaque_index_count = opaque;
    let had_overrides = !dst.surface_overrides.is_empty();
    if !src.surface_overrides.is_empty() && !had_overrides {
        dst.surface_overrides = vec![dst.surface; dst.positions.len()];
    }
    dst.positions.extend_from_slice(&src.positions);
    dst.normals.extend_from_slice(&src.normals);
    dst.colors.extend_from_slice(&src.colors);
    let uvs = src.uvs.len();
    dst.uvs.extend_from_slice(&src.uvs);
    dst.uvs
        .extend(std::iter::repeat_n([0.0; 2], src.positions.len() - uvs));
    if !dst.surface_overrides.is_empty() {
        if src.surface_overrides.is_empty() {
            dst.surface_overrides
                .extend(std::iter::repeat_n(src.surface, src.positions.len()));
        } else {
            dst.surface_overrides
                .extend_from_slice(&src.surface_overrides);
        }
    }
    dst.elements
        .extend(std::iter::repeat_n(element, src.positions.len()));
}

fn dim(c: Color, k: f32) -> Color {
    Color::rgb01_unchecked(c.r * k, c.g * k, c.b * k)
}

/// A wall panel: the wall with a window opening's glass and sill, or a
/// glass panel in a thin frame.
fn panel_mesh(
    e: &Element,
    along: Vec3,
    out: Vec3,
    glass: bool,
    spec: &TowerSpec,
) -> EngineResult<Mesh> {
    let size = e.half * 2.0;
    let mut m = Mesh::box_at(
        e.center,
        size,
        if glass {
            dim(spec.wall_color, 0.7)
        } else {
            spec.wall_color
        },
    )?;
    let w = 2.0 * e.half.dot(along.abs());
    let h = size.y;
    let (ww, wh) = if glass {
        (w * 0.9, h * 0.86)
    } else {
        (w * 0.45, h * 0.5)
    };
    let pane = e.center
        + out * (e.half.dot(out.abs()) + 0.01)
        + Vec3::Y * (if glass { 0.0 } else { h * 0.05 });
    let pane_size = along.abs() * ww + Vec3::Y * wh + out.abs() * 0.02;
    let mut glass_mesh = Mesh::box_at(pane, pane_size, spec.glass_color)?;
    glass_mesh.set_surface_material(crate::mesh::SurfaceMaterial {
        roughness: 0.08,
        metallic: 0.6,
        ..crate::mesh::SurfaceMaterial::DEFAULT
    });
    m.append_translated(&glass_mesh, Vec3::ZERO)?;
    Ok(m)
}

/// Element bits: one per element in a plan. The broken bits are a
/// building's whole state; store them, send them, apply them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ElementBits {
    words: Vec<u64>,
    len: u32,
}

impl ElementBits {
    pub fn new(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(64)],
            len: len as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, i: usize) -> bool {
        i < self.len() && self.words[i / 64] & (1 << (i % 64)) != 0
    }

    pub fn set(&mut self, i: usize, on: bool) {
        if i >= self.len() {
            return;
        }
        let bit = 1u64 << (i % 64);
        if on {
            self.words[i / 64] |= bit;
        } else {
            self.words[i / 64] &= !bit;
        }
    }

    /// Bits that are on.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(move |&i| self.get(i))
    }

    /// Compact bytes for the network: little-endian length, then the bits.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.len.to_le_bytes().to_vec();
        let bytes = self.len().div_ceil(8);
        for i in 0..bytes {
            out.push((self.words[i / 8] >> ((i % 8) * 8)) as u8);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> EngineResult<Self> {
        let bad = || EngineError::InvalidValue("element bits: bad byte length".into());
        let len = u32::from_le_bytes(
            bytes
                .get(..4)
                .ok_or_else(bad)?
                .try_into()
                .map_err(|_| bad())?,
        );
        let mut b = Self::new(len as usize);
        let body = &bytes[4..];
        if body.len() != (len as usize).div_ceil(8) {
            return Err(bad());
        }
        for (i, byte) in body.iter().enumerate() {
            b.words[i / 8] |= (*byte as u64) << ((i % 8) * 8);
        }
        Ok(b)
    }

    fn or(&mut self, other: &ElementBits) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }
}

/// How the upper part of a building comes down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CollapseMode {
    /// Straight down, storey after storey.
    Pancake,
    /// Tipping over around a hinge on the damaged side.
    Topple {
        /// Point on the hinge line (building space, at the failing floor).
        hinge: Vec3,
        /// Horizontal direction the top falls towards.
        toward: Vec3,
    },
}

/// The structural failure of a building, from the broken bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Collapse {
    /// Lowest storey that failed; it and everything above came down.
    pub from_floor: u16,
    pub mode: CollapseMode,
    /// Height of the falling part in metres.
    pub height: f32,
    /// Bottom of the falling part (building space).
    pub base_y: f32,
}

/// A heap of rubble left by a collapse, in the building's space.
/// Deterministic: the same broken bits give the same mounds everywhere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RubbleMound {
    pub center: Vec3,
    pub radius: f32,
    pub height: f32,
    pub yaw: f32,
}

impl RubbleMound {
    /// A lumpy dome mesh for this mound (base at y = 0, local to its centre).
    pub fn mesh(&self, color: Color) -> EngineResult<Mesh> {
        let rings = 8usize;
        let seg = 28usize;
        let mut m = Mesh::new();
        let k = self.yaw * 3.7;
        let lump = |x: f32, z: f32| {
            1.0 + 0.09 * (x * 5.1 + z * 3.3 + k).sin() * (z * 4.7 - x * 2.1 + k).cos()
                + 0.04 * (x * 13.0 + z * 11.0 + k).sin()
        };
        let (r, h) = (self.radius, self.height);
        let mut ids = Vec::new();
        for ring in 0..=rings {
            let phi = ring as f32 / rings as f32 * std::f32::consts::FRAC_PI_2;
            let mut row = Vec::new();
            for s in 0..seg {
                let a = s as f32 / seg as f32 * std::f32::consts::TAU;
                let (x, z) = (phi.cos() * a.cos(), phi.cos() * a.sin());
                let n = if ring == rings { 1.0 } else { lump(x, z) };
                // A low, spread-out heap: flatter near the top.
                let y = phi.sin().powf(0.8) * n - 0.04;
                let p = Vec3::new(x * n * r, y * h, z * n * r);
                let id = m.add_point(p)?;
                m.set_point_color(id, color)?;
                let normal = Vec3::new(p.x / (r * r), (p.y + 0.04 * h) / (h * h), p.z / (r * r));
                m.set_point_normal(id, normal.normalize_or(Vec3::Y))?;
                row.push(id);
            }
            ids.push(row);
        }
        for ring in 0..rings {
            for s in 0..seg {
                let t = (s + 1) % seg;
                let (a, b, c, d) = (
                    ids[ring][s],
                    ids[ring][t],
                    ids[ring + 1][t],
                    ids[ring + 1][s],
                );
                if ring + 1 == rings {
                    m.add_triangle(a, c, b)?;
                } else {
                    m.add_quad(a, d, c, b)?;
                }
            }
        }
        m.set_surface_material(crate::mesh::SurfaceMaterial {
            roughness: 0.95,
            ..crate::mesh::SurfaceMaterial::DEFAULT
        });
        Ok(m)
    }
}

/// What changed after a hit or [`Building::apply`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildingChange {
    /// Elements broken directly.
    pub broken: Vec<ElementId>,
    /// Elements that fell because nothing holds them any more (not
    /// counting a collapse's falling part).
    pub dropped: Vec<ElementId>,
    /// Elements of a new collapse's falling part.
    pub collapsed: Vec<ElementId>,
    /// A collapse that started with this change.
    pub collapse: Option<Collapse>,
}

impl BuildingChange {
    pub fn is_empty(&self) -> bool {
        self.broken.is_empty() && self.dropped.is_empty() && self.collapsed.is_empty()
    }
}

/// A building's state: what is broken, what fell, and the rubble. No
/// rendering, so a server keeps one per building.
#[derive(Clone, Debug)]
pub struct Building {
    plan: Arc<BuildingPlan>,
    broken: ElementBits,
    fallen: ElementBits,
    collapse: Option<Collapse>,
    mounds: Vec<RubbleMound>,
}

impl Building {
    pub fn new(plan: Arc<BuildingPlan>) -> Self {
        let n = plan.len();
        Self {
            plan,
            broken: ElementBits::new(n),
            fallen: ElementBits::new(n),
            collapse: None,
            mounds: Vec::new(),
        }
    }

    pub fn plan(&self) -> &Arc<BuildingPlan> {
        &self.plan
    }

    /// Broken directly: the building's whole state.
    pub fn broken(&self) -> &ElementBits {
        &self.broken
    }

    /// Came down because they lost support (follows from [`Self::broken`]).
    pub fn fallen(&self) -> &ElementBits {
        &self.fallen
    }

    pub fn is_standing(&self, id: ElementId) -> bool {
        let i = id.0 as usize;
        i < self.plan.len() && !self.broken.get(i) && !self.fallen.get(i)
    }

    /// Elements gone for whatever reason.
    pub fn gone(&self) -> ElementBits {
        let mut g = self.broken.clone();
        g.or(&self.fallen);
        g
    }

    pub fn collapse(&self) -> Option<Collapse> {
        self.collapse
    }

    pub fn mounds(&self) -> &[RubbleMound] {
        &self.mounds
    }

    /// Hit at `point` (building space): every standing element within
    /// `radius` whose strength is below `power` (fading to zero at the edge)
    /// breaks. Then the structure is checked.
    pub fn hit(&mut self, point: Vec3, radius: f32, power: f32) -> BuildingChange {
        if !(point.is_finite() && radius.is_finite() && radius > 0.0 && power.is_finite()) {
            return BuildingChange::default();
        }
        let ids: Vec<ElementId> = self
            .plan
            .elements
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                if self.broken.get(*i) || self.fallen.get(*i) {
                    return false;
                }
                let d = e.distance_to(point);
                d < radius && power * (1.0 - d / radius) >= e.strength
            })
            .map(|(i, _)| ElementId(i as u32))
            .collect();
        self.break_elements(&ids)
    }

    /// Break these elements and check the structure.
    pub fn break_elements(&mut self, ids: &[ElementId]) -> BuildingChange {
        let mut broken = Vec::new();
        for id in ids {
            let i = id.0 as usize;
            if i < self.plan.len() && !self.broken.get(i) && !self.fallen.get(i) {
                self.broken.set(i, true);
                broken.push(*id);
            }
        }
        if broken.is_empty() {
            return BuildingChange::default();
        }
        let mut change = self.settle();
        change.broken = broken;
        change
    }

    /// Take over another building's broken bits (from a save or the
    /// server): the result is identical to the building they came from.
    pub fn apply(&mut self, bits: &ElementBits) -> BuildingChange {
        if bits.len() != self.plan.len() {
            return BuildingChange::default();
        }
        let before = self.broken.clone();
        self.broken = bits.clone();
        let mut change = self.settle();
        change.broken = bits
            .iter_set()
            .filter(|&i| !before.get(i))
            .map(|i| ElementId(i as u32))
            .collect();
        change
    }

    /// Recompute what fell, the collapse and the mounds from the broken bits.
    fn settle(&mut self) -> BuildingChange {
        let (fallen, collapse, collapsed) = analyze(&self.plan, &self.broken);
        let mut change = BuildingChange::default();
        for i in fallen.iter_set() {
            if !self.fallen.get(i) {
                let id = ElementId(i as u32);
                if collapsed.get(i) {
                    change.collapsed.push(id);
                } else {
                    change.dropped.push(id);
                }
            }
        }
        if collapse != self.collapse {
            change.collapse = collapse.filter(|_| !change.collapsed.is_empty());
        }
        self.fallen = fallen;
        self.collapse = collapse;
        self.mounds = mounds(&self.plan, collapse);
        change
    }

    /// First standing element along a ray (building space).
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<(ElementId, f32)> {
        let dir = dir.try_normalize()?;
        let inv = dir.recip();
        let mut best: Option<(ElementId, f32)> = None;
        for (i, e) in self.plan.elements.iter().enumerate() {
            if self.broken.get(i) || self.fallen.get(i) {
                continue;
            }
            let t0 = (e.center - e.half - origin) * inv;
            let t1 = (e.center + e.half - origin) * inv;
            let near = t0.min(t1).max_element().max(0.0);
            let far = t0.max(t1).min_element();
            if near <= far && near <= max && best.is_none_or(|b| near < b.1) {
                best = Some((ElementId(i as u32), near));
            }
        }
        best
    }
}

/// Fallen elements, the collapse and the collapse's own elements.
fn analyze(
    plan: &BuildingPlan,
    broken: &ElementBits,
) -> (ElementBits, Option<Collapse>, ElementBits) {
    let n = plan.len();
    let mut fallen = ElementBits::new(n);
    let mut collapsed = ElementBits::new(n);
    let gone = |i: usize, fallen: &ElementBits| broken.get(i) || fallen.get(i);
    let centroid = plan.centroid();
    let rad = plan.radius();
    let top = plan.floors.last().map_or(0.0, |f| f.y1);
    let mut collapse = None;
    for (f, floor) in plan.floors.iter().enumerate() {
        // Load-bearing perimeter of this storey.
        let ring = plan.rings.get(f).map_or(&[][..], |r| &r[..]);
        if ring.is_empty() {
            continue;
        }
        let (mut alive, mut total) = (0.0f32, 0.0f32);
        let (mut run, mut worst, mut first) = (0.0f32, 0.0f32, None::<f32>);
        let mut asym = Vec2::ZERO;
        // Width standing and in all, per facade direction.
        let mut sides: Vec<(Vec3, f32, f32)> = Vec::new();
        for &i in ring {
            let e = &plan.elements[i as usize];
            let w = e.width() * e.kind.carries().max(0.3);
            total += w;
            let dead = gone(i as usize, &fallen);
            match sides.iter_mut().find(|s| s.0.dot(e.outward) > 0.95) {
                Some(s) => {
                    s.2 += w;
                    if !dead {
                        s.1 += w;
                    }
                }
                None => sides.push((e.outward, if dead { 0.0 } else { w }, w)),
            }
            if dead {
                run += w;
                asym += (Vec2::new(e.center.x, e.center.z) - centroid) * w;
            } else {
                alive += w;
                if first.is_none() {
                    first = Some(run);
                }
                worst = worst.max(run);
                run = 0.0;
            }
        }
        worst = worst.max(run + first.unwrap_or(0.0));
        // Inner columns carry too.
        let (mut col_alive, mut col_total) = (0.0f32, 0.0f32);
        for (i, e) in plan.elements.iter().enumerate() {
            if e.kind == ElementKind::Column && e.floor as usize == f {
                col_total += 1.0;
                if !gone(i, &fallen) {
                    col_alive += 1.0;
                }
            }
        }
        let column_share = if col_total > 0.0 {
            col_alive / col_total
        } else {
            1.0
        };
        let ratio = alive / total.max(1e-6);
        let side_gone = sides.iter().any(|s| s.1 < s.2 * 0.15);
        let fails = ratio < 0.5
            || side_gone
            || worst > total * 0.42
            || (col_total > 0.0 && column_share < 0.25 && ratio < 0.8);
        if fails {
            let height = top - floor.y0;
            let mag = asym.length();
            let mode = if mag > rad * 2.5 && height > rad * 0.8 {
                let toward = Vec3::new(asym.x, 0.0, asym.y) / mag;
                let (lo, hi) = plan.footprint();
                let half = (hi - lo) * 0.5;
                let reach = toward.x.abs() * half.x + toward.z.abs() * half.y;
                CollapseMode::Topple {
                    hinge: Vec3::new(centroid.x, floor.y0, centroid.y) + toward * reach,
                    toward,
                }
            } else {
                CollapseMode::Pancake
            };
            collapse = Some(Collapse {
                from_floor: f as u16,
                mode,
                height,
                base_y: floor.y0,
            });
            for (i, e) in plan.elements.iter().enumerate() {
                if e.floor as usize >= f && !broken.get(i) {
                    fallen.set(i, true);
                    collapsed.set(i, true);
                }
            }
            break;
        }
    }
    // Slab tiles, bottom up: held by a standing wall or column of the
    // storey below, or within two tiles of a held tile.
    let reach = plan.tile * 0.5 + 0.6;
    for f in 1..plan.slabs.len() {
        let tiles: Vec<u32> = plan.slabs[f]
            .iter()
            .copied()
            .filter(|&i| !gone(i as usize, &fallen))
            .collect();
        if tiles.is_empty() {
            continue;
        }
        let holders: Vec<&Element> = plan
            .elements
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                e.kind.holds_slabs() && e.floor as usize + 1 == f && !gone(*i, &fallen)
            })
            .map(|(_, e)| e)
            .collect();
        let mut dist = vec![u8::MAX; tiles.len()];
        let mut queue = std::collections::VecDeque::new();
        for (k, &i) in tiles.iter().enumerate() {
            let t = &plan.elements[i as usize];
            let held = holders.iter().any(|h| {
                let d = (Vec2::new(h.center.x - t.center.x, h.center.z - t.center.z)).abs()
                    - Vec2::new(h.half.x + t.half.x, h.half.z + t.half.z);
                d.max(Vec2::ZERO).length() < reach - plan.tile * 0.5
            });
            if held {
                dist[k] = 0;
                queue.push_back(k);
            }
        }
        while let Some(k) = queue.pop_front() {
            if dist[k] >= 2 {
                continue;
            }
            let a = &plan.elements[tiles[k] as usize];
            for (m, &j) in tiles.iter().enumerate() {
                if dist[m] != u8::MAX {
                    continue;
                }
                let b = &plan.elements[j as usize];
                let gap = (Vec2::new(a.center.x - b.center.x, a.center.z - b.center.z)).abs()
                    - Vec2::new(a.half.x + b.half.x, a.half.z + b.half.z);
                if gap.max_element() < 0.05 && gap.min_element() < 0.05 && gap.x.max(gap.y) < 0.05 {
                    dist[m] = dist[k] + 1;
                    queue.push_back(m);
                }
            }
        }
        for (k, &i) in tiles.iter().enumerate() {
            if dist[k] == u8::MAX {
                fallen.set(i as usize, true);
            }
        }
    }
    (fallen, collapse, collapsed)
}

/// The rubble a collapse leaves, from the collapse alone.
fn mounds(plan: &BuildingPlan, collapse: Option<Collapse>) -> Vec<RubbleMound> {
    let Some(c) = collapse else {
        return Vec::new();
    };
    let centroid = plan.centroid();
    let rad = plan.radius();
    let mut h = Hash(plan.seed ^ 0x9e37_79b9_7f4a_7c15 ^ c.from_floor as u64);
    let ground = plan.floors.first().map_or(0.0, |f| f.y0);
    match c.mode {
        CollapseMode::Pancake => {
            let vol = std::f32::consts::PI * rad * rad * c.height;
            let height = (c.height * 0.3)
                .min((vol * 0.18 / std::f32::consts::PI).cbrt() * 0.9)
                .max(1.0);
            vec![RubbleMound {
                center: Vec3::new(centroid.x, c.base_y, centroid.y),
                radius: rad * 1.35,
                height,
                yaw: h.unit() * std::f32::consts::TAU,
            }]
        }
        CollapseMode::Topple { hinge, toward } => {
            let mut out = vec![RubbleMound {
                center: Vec3::new(hinge.x, c.base_y, hinge.z) - toward * rad * 0.6,
                radius: rad * 0.9,
                height: (rad * 0.4).clamp(1.0, 5.0),
                yaw: h.unit() * std::f32::consts::TAU,
            }];
            let segs = (c.height / 6.0).ceil().max(1.0) as usize;
            for k in 0..segs {
                let s = k as f32 * 6.0;
                if s >= rad && k % 2 == 0 {
                    out.push(RubbleMound {
                        center: Vec3::new(hinge.x, ground, hinge.z) + toward * (s + 3.0),
                        radius: rad * 0.9,
                        height: (rad * 0.35).clamp(0.8, 4.0),
                        yaw: h.unit() * std::f32::consts::TAU,
                    });
                }
            }
            out
        }
    }
}

/// A tiny seeded generator, the same on every machine.
pub(crate) struct Hash(pub u64);

impl Hash {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// 0..1
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tower() -> Arc<BuildingPlan> {
        Arc::new(BuildingPlan::tower(&TowerSpec::default()).unwrap())
    }

    fn panels(plan: &BuildingPlan, floor: u16, outward: Vec3) -> Vec<ElementId> {
        plan.elements()
            .iter()
            .enumerate()
            .filter(|(_, e)| e.floor == floor && e.outward.dot(outward) > 0.9)
            .map(|(i, _)| ElementId(i as u32))
            .collect()
    }

    #[test]
    fn a_tower_has_every_kind_of_element_and_tagged_geometry() {
        let plan = tower();
        for kind in [
            ElementKind::Wall,
            ElementKind::Slab,
            ElementKind::Column,
            ElementKind::Roof,
            ElementKind::Partition,
        ] {
            assert!(plan.elements().iter().any(|e| e.kind == kind), "{kind:?}");
        }
        let m = plan.mesh();
        assert_eq!(m.elements.len(), m.positions.len());
        assert!(m
            .elements
            .iter()
            .all(|&e| e >= 1 && e as usize <= plan.len()));
        assert_eq!(plan.floors().len(), 6);
    }

    #[test]
    fn one_broken_panel_does_not_bring_the_building_down() {
        let mut b = Building::new(tower());
        let id = panels(b.plan(), 0, -Vec3::Z)[1];
        let change = b.break_elements(&[id]);
        assert_eq!(change.broken, vec![id]);
        assert!(change.collapse.is_none());
        assert_eq!(b.fallen().count(), 0);
        assert!(!b.is_standing(id));
    }

    #[test]
    fn losing_a_whole_side_topples_the_upper_part_that_way() {
        let mut b = Building::new(tower());
        let front = panels(b.plan(), 1, -Vec3::Z);
        let change = b.break_elements(&front);
        let c = change.collapse.expect("collapse");
        assert_eq!(c.from_floor, 1);
        match c.mode {
            CollapseMode::Topple { toward, hinge } => {
                assert!(toward.z < -0.9, "falls forward: {toward}");
                assert!((hinge.y - 3.4).abs() < 1e-3);
            }
            m => panic!("expected a topple, got {m:?}"),
        }
        // Everything from floor 1 up is gone, the ground floor stands.
        for (i, e) in b.plan().elements().iter().enumerate() {
            let gone = !b.is_standing(ElementId(i as u32));
            if e.floor >= 1 {
                assert!(gone, "element {i} on floor {} should be down", e.floor);
            } else if e.kind != ElementKind::Partition {
                assert!(!gone, "ground floor element {i} should stand");
            }
        }
        assert!(b.mounds().len() >= 2);
    }

    #[test]
    fn even_damage_all_round_pancakes() {
        let mut b = Building::new(tower());
        let plan = b.plan().clone();
        // Every other panel of the ground floor, all the way round, and the columns.
        let ring = plan.rings[0].clone();
        let mut ids: Vec<ElementId> = ring
            .iter()
            .enumerate()
            .filter(|(k, _)| k % 3 != 0)
            .map(|(_, &i)| ElementId(i))
            .collect();
        ids.extend(
            plan.elements()
                .iter()
                .enumerate()
                .filter(|(_, e)| e.kind == ElementKind::Column && e.floor == 0)
                .map(|(i, _)| ElementId(i as u32)),
        );
        let change = b.break_elements(&ids);
        let c = change.collapse.expect("collapse");
        assert_eq!(c.from_floor, 0);
        assert_eq!(c.mode, CollapseMode::Pancake);
        assert_eq!(b.mounds().len(), 1);
        assert!(b.mounds()[0].height >= 1.0);
    }

    #[test]
    fn a_slab_tile_far_from_any_support_drops() {
        // Columns gone on floor 2 and the walls next to some tiles: the
        // middle tiles of floor 3 lose their holders.
        let mut b = Building::new(tower());
        let plan = b.plan().clone();
        let cols: Vec<ElementId> = plan
            .elements()
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kind == ElementKind::Column && e.floor == 2)
            .map(|(i, _)| ElementId(i as u32))
            .collect();
        let change = b.break_elements(&cols);
        // The tower is 5x4 tiles: every tile is within two of an edge, so all stay.
        assert!(change.dropped.is_empty());
        // Break a block of tiles around one tile: it has no neighbours left.
        let tiles: Vec<u32> = plan.slabs[3].clone();
        let keep = tiles[6];
        let around: Vec<ElementId> = tiles
            .iter()
            .filter(|&&t| t != keep)
            .filter(|&&t| {
                let a = &plan.elements()[t as usize];
                let k = &plan.elements()[keep as usize];
                (a.center - k.center).length() < 4.5
            })
            .map(|&t| ElementId(t))
            .collect();
        let change = b.break_elements(&around);
        assert!(change.collapse.is_none());
        let k = &plan.elements()[keep as usize];
        let near_wall = (k.center.x.abs() > 4.0) || (k.center.z.abs() > 3.0);
        if !near_wall {
            assert!(
                change.dropped.contains(&ElementId(keep)),
                "lone tile should drop"
            );
        }
    }

    #[test]
    fn hits_break_weak_things_first() {
        let mut b = Building::new(tower());
        // A small knock at a ground-floor partition breaks it, not the walls.
        let part = b
            .plan()
            .elements()
            .iter()
            .position(|e| e.kind == ElementKind::Partition && e.floor == 0)
            .unwrap();
        let p = b.plan().elements()[part].center;
        let change = b.hit(p, 0.8, 100.0);
        assert!(change.broken.contains(&ElementId(part as u32)));
        assert!(change
            .broken
            .iter()
            .all(|id| b.plan().elements()[id.0 as usize].kind == ElementKind::Partition));
    }

    #[test]
    fn the_same_bits_give_the_same_building_and_rubble() {
        let plan = tower();
        let mut a = Building::new(plan.clone());
        for (i, x) in [-6.0f32, -3.0, 0.0, 3.0, 6.0].into_iter().enumerate() {
            a.hit(Vec3::new(x, 1.7 + i as f32 * 0.1, -6.0), 2.5, 1.0e6);
        }
        assert!(a.collapse().is_some());
        let bytes = a.broken().to_bytes();
        let mut b = Building::new(plan);
        let change = b.apply(&ElementBits::from_bytes(&bytes).unwrap());
        assert_eq!(b.broken(), a.broken());
        assert_eq!(b.fallen(), a.fallen());
        assert_eq!(b.collapse(), a.collapse());
        assert_eq!(b.mounds(), a.mounds());
        assert!(change.collapse.is_some());
    }

    #[test]
    fn rays_hit_the_facade_and_pass_through_holes() {
        let mut b = Building::new(tower());
        let (id, t) = b
            .raycast(Vec3::new(0.0, 1.7, -20.0), Vec3::Z, 50.0)
            .unwrap();
        assert_eq!(b.plan().elements()[id.0 as usize].kind, ElementKind::Wall);
        assert!((t - 14.0).abs() < 0.3, "{t}");
        b.break_elements(&[id]);
        let (next, t2) = b
            .raycast(Vec3::new(0.0, 1.7, -20.0), Vec3::Z, 50.0)
            .unwrap();
        assert_ne!(next, id);
        assert!(t2 > t);
    }

    #[test]
    fn bits_round_trip_through_bytes() {
        let mut bits = ElementBits::new(77);
        for i in [0, 5, 63, 64, 76] {
            bits.set(i, true);
        }
        let back = ElementBits::from_bytes(&bits.to_bytes()).unwrap();
        assert_eq!(back, bits);
        assert_eq!(back.count(), 5);
        assert!(ElementBits::from_bytes(&[1, 2]).is_err());
    }

    #[test]
    fn a_server_building_holds_bodies_until_its_slab_breaks() {
        use crate::physics::{Body, BodyShape, PhysicsWorld};
        use glam::Mat4;
        let plan = tower();
        let mut b = Building::new(plan.clone());
        let mut p = PhysicsWorld::new();
        let key = p.add_building(&b, Mat4::IDENTITY);
        // A crate dropped onto the second floor rests on its slab.
        let crate_at = Vec3::new(1.5, 5.0, 1.5);
        let id = p
            .add(
                &Body::dynamic().shape(BodyShape::Box(Vec3::splat(0.4))),
                Mat4::from_translation(crate_at),
            )
            .unwrap();
        for _ in 0..120 {
            p.step(1.0 / 60.0);
        }
        let rest = p.position(id).unwrap().y;
        assert!(
            (rest - (3.4 + 0.3 + 0.4)).abs() < 0.1,
            "rests on the slab: {rest}"
        );
        // Break the slab tiles under it: it falls to the ground floor.
        b.hit(Vec3::new(1.5, 3.55, 1.5), 3.5, 1.0e5);
        p.update_building(key, &b);
        for _ in 0..120 {
            p.step(1.0 / 60.0);
        }
        let low = p.position(id).unwrap().y;
        assert!(low < 1.0, "fell through: {low}");
    }
}
