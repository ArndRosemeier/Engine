//! Water that flows in three dimensions: a FLIP fluid in a box.
//!
//! [`crate::water`] is a height field: a level surface in y, cheap and
//! endless, right for seas and rivers under ordinary gravity. A [`Fluid`]
//! is real volume in a box of cells. Particles carry the water and a grid
//! keeps it from compressing, so it pours, splashes, fills basins and,
//! with a gravity field ([`crate::physics::GravityField`]), pools on walls,
//! hangs under a ceiling that pulls, or drifts as a blob where nothing
//! pulls at all.
//!
//! The simulation runs on the CPU and is deterministic: the same box, the
//! same water and the same field give the same particles every run. Keep
//! boxes to the active zone around the player (tens of metres at 0.25 to
//! 0.5 m cells); far water can be put to sleep with
//! [`World::set_fluid_awake`](crate::World::set_fluid_awake).
//!
//! ```no_run
//! use engine::prelude::*;
//!
//! fn setup(world: &mut World) -> EngineResult<FluidId> {
//!     // A 12 x 6 x 12 m box of 0.4 m cells with its corner at (-6, 0, -6).
//!     let mut lake = Fluid::new(FluidSettings::new(Vec3::new(-6.0, 0.0, -6.0), Vec3::new(12.0, 6.0, 12.0), 0.4))?;
//!     // Rock inside the box: a basin. (The box walls hold water too.)
//!     lake.mark_solid(|p| p.y < 0.6 + 0.05 * (p.x * p.x + p.z * p.z).sqrt());
//!     // A block of water that will slump into the basin.
//!     lake.fill_box(Vec3::new(-2.0, 2.0, -2.0), Vec3::new(2.0, 4.0, 2.0));
//!     world.add_fluid(lake)
//! }
//! ```

mod sim;
mod surface;

use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::mesh::Mesh;
use glam::{IVec3, UVec3, Vec3};
use sim::{Cell, Sim};

/// Most cells along one side of a fluid box.
pub const MAX_FLUID_CELLS_PER_SIDE: u32 = 160;
/// Most cells in one fluid box.
pub const MAX_FLUID_CELLS: u32 = 1_500_000;
/// Particles seeded per cell side by `fill_*` (2 means 8 per cell).
const SEED_PER_SIDE: u32 = 2;

/// How a [`Fluid`] box is laid out and how its water behaves.
#[derive(Clone, Debug)]
pub struct FluidSettings {
    /// The box's minimum corner, in world space.
    pub origin: Vec3,
    /// Cells along x, y and z.
    pub cells: UVec3,
    /// Edge of one cell, metres. Water thinner than about two cells does
    /// not show; smaller cells cost more (the cost grows with the cube).
    pub cell_size: f32,
    /// Surface colour; its alpha is how much of what lies behind shows.
    pub color: Color,
    /// Share of FLIP (lively, splashy) against PIC (calm, viscous), 0..1.
    pub flip: f32,
    /// Velocity lost per second inside the water (a little keeps it calm).
    pub drag: f32,
    /// Most solver steps per frame; fast water slows down past it.
    pub max_substeps: u32,
    /// Most pressure iterations per step.
    pub pressure_iterations: usize,
    /// Rebuild the surface mesh every this many frames (1 = every frame).
    pub mesh_every: u32,
}

impl FluidSettings {
    /// A box from `origin` (minimum corner) of `size` metres, cut into
    /// cells of `cell_size`.
    pub fn new(origin: Vec3, size: Vec3, cell_size: f32) -> Self {
        let cells = if cell_size > 0.0 && size.is_finite() {
            (size / cell_size).ceil().max(Vec3::ONE).as_uvec3()
        } else {
            UVec3::ONE
        };
        Self {
            origin,
            cells,
            cell_size,
            color: Color::rgba(45, 110, 150, 190),
            flip: 0.95,
            drag: 0.05,
            max_substeps: 4,
            pressure_iterations: 150,
            mesh_every: 1,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn flip(mut self, flip: f32) -> Self {
        self.flip = flip;
        self
    }

    pub fn drag(mut self, drag: f32) -> Self {
        self.drag = drag;
        self
    }

    pub fn max_substeps(mut self, steps: u32) -> Self {
        self.max_substeps = steps;
        self
    }

    pub fn pressure_iterations(mut self, iterations: usize) -> Self {
        self.pressure_iterations = iterations;
        self
    }

    pub fn mesh_every(mut self, frames: u32) -> Self {
        self.mesh_every = frames;
        self
    }

    /// The box's size in metres.
    pub fn size(&self) -> Vec3 {
        self.cells.as_vec3() * self.cell_size
    }
}

/// A box of three-dimensional water. Build it, mark rock and fill it, then
/// hand it to [`World::add_fluid`](crate::World::add_fluid), which steps it
/// with the world's gravity (or gravity field) and draws its surface. It
/// also works on its own: call [`Fluid::step`] with any acceleration.
#[derive(Clone, Debug)]
pub struct Fluid {
    settings: FluidSettings,
    sim: Sim,
    seed: u32,
    awake: bool,
}

/// Deterministic jitter in 0..1.
fn hash01(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

impl Fluid {
    /// An empty box. Errors name what is wrong with the settings.
    pub fn new(settings: FluidSettings) -> EngineResult<Self> {
        let bad = |what: &str| Err(EngineError::InvalidValue(format!("fluid: {what}")));
        if !settings.origin.is_finite() {
            return bad("origin must be finite");
        }
        if !(settings.cell_size.is_finite() && settings.cell_size > 0.0) {
            return bad("cell_size must be a positive number");
        }
        let n = settings.cells;
        if n.min_element() < 2 {
            return bad("the box needs at least 2 cells along each side");
        }
        if n.max_element() > MAX_FLUID_CELLS_PER_SIDE {
            return bad(&format!(
                "at most {MAX_FLUID_CELLS_PER_SIDE} cells per side (asked {n}); use bigger cells"
            ));
        }
        if n.x * n.y * n.z > MAX_FLUID_CELLS {
            return bad(&format!(
                "at most {MAX_FLUID_CELLS} cells (asked {n}); use bigger cells"
            ));
        }
        if !(0.0..=1.0).contains(&settings.flip) {
            return bad("flip must be between 0 and 1");
        }
        if !(settings.drag.is_finite() && settings.drag >= 0.0) {
            return bad("drag must be zero or more");
        }
        let mut sim = Sim::new(n, settings.cell_size);
        sim.flip = settings.flip;
        sim.drag = settings.drag;
        Ok(Self {
            settings,
            sim,
            seed: 0x9E37_79B9,
            awake: true,
        })
    }

    pub fn settings(&self) -> &FluidSettings {
        &self.settings
    }

    /// The box's minimum corner in world space.
    pub fn origin(&self) -> Vec3 {
        self.settings.origin
    }

    /// The box as (minimum, maximum) corners in world space.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let o = self.settings.origin;
        (o, o + self.settings.size())
    }

    fn local(&self, p: Vec3) -> Vec3 {
        p - self.settings.origin
    }

    /// Mark rock: every cell whose centre `solid` says is inside. Water
    /// does not enter rock; the box's own walls are always closed.
    pub fn mark_solid(&mut self, solid: impl Fn(Vec3) -> bool) {
        let n = self.sim.n;
        for z in 0..n.z as i32 {
            for y in 0..n.y as i32 {
                for x in 0..n.x as i32 {
                    let c = IVec3::new(x, y, z);
                    let i = self.sim.cell_index(c).expect("inside");
                    let p = self.sim.cell_center(c) + self.settings.origin;
                    self.sim.solid[i] = solid(p);
                }
            }
        }
        self.drop_particles_in_rock();
    }

    /// Mark or clear one cell as rock (for digging and building).
    pub fn set_solid_at(&mut self, p: Vec3, solid: bool) {
        let c = self.sim.cell_of(self.local(p));
        if let Some(i) = self.sim.cell_index(c) {
            self.sim.solid[i] = solid;
        }
        if solid {
            self.drop_particles_in_rock();
        }
    }

    /// Whether the cell at `p` is rock (outside the box counts as rock).
    pub fn is_solid_at(&self, p: Vec3) -> bool {
        self.sim.is_solid(self.sim.cell_of(self.local(p)))
    }

    fn drop_particles_in_rock(&mut self) {
        let mut k = 0;
        while k < self.sim.pos.len() {
            if self.sim.is_solid(self.sim.cell_of(self.sim.pos[k])) {
                self.sim.pos.swap_remove(k);
                self.sim.vel.swap_remove(k);
            } else {
                k += 1;
            }
        }
    }

    /// Fill every open cell whose centre `inside` accepts with water, at
    /// rest. Returns how many particles were added.
    pub fn fill(&mut self, inside: impl Fn(Vec3) -> bool) -> usize {
        let n = self.sim.n;
        let h = self.sim.h;
        let before = self.sim.pos.len();
        let step = h / SEED_PER_SIDE as f32;
        for z in 0..n.z as i32 {
            for y in 0..n.y as i32 {
                for x in 0..n.x as i32 {
                    let c = IVec3::new(x, y, z);
                    if self.sim.is_solid(c)
                        || !inside(self.sim.cell_center(c) + self.settings.origin)
                    {
                        continue;
                    }
                    let corner = c.as_vec3() * h;
                    for k in 0..SEED_PER_SIDE.pow(3) {
                        let sub = UVec3::new(
                            k % SEED_PER_SIDE,
                            (k / SEED_PER_SIDE) % SEED_PER_SIDE,
                            k / (SEED_PER_SIDE * SEED_PER_SIDE),
                        );
                        let j = Vec3::new(
                            hash01(&mut self.seed),
                            hash01(&mut self.seed),
                            hash01(&mut self.seed),
                        );
                        let p = corner + (sub.as_vec3() + 0.2 + 0.6 * j) * step;
                        self.sim.pos.push(p);
                        self.sim.vel.push(Vec3::ZERO);
                    }
                }
            }
        }
        self.sim.pos.len() - before
    }

    /// Fill the part of the box between two world corners.
    pub fn fill_box(&mut self, min: Vec3, max: Vec3) -> usize {
        self.fill(|p| p.cmpge(min).all() && p.cmple(max).all())
    }

    /// Fill a ball of water.
    pub fn fill_ball(&mut self, center: Vec3, radius: f32) -> usize {
        self.fill(|p| p.distance(center) <= radius)
    }

    /// Pour: add `count` particles in a ball of `radius` at `at`, all moving
    /// at `velocity` (a spring, a burst pipe, a bucket). Eight particles
    /// make one cell of water. Particles in rock or outside are skipped.
    pub fn pour(&mut self, at: Vec3, radius: f32, velocity: Vec3, count: usize) -> usize {
        if !at.is_finite() || !velocity.is_finite() || !radius.is_finite() {
            return 0;
        }
        let before = self.sim.pos.len();
        let size = self.settings.size();
        for _ in 0..count {
            // Rejection sample a unit ball.
            let mut d;
            loop {
                d = Vec3::new(
                    hash01(&mut self.seed),
                    hash01(&mut self.seed),
                    hash01(&mut self.seed),
                ) * 2.0
                    - 1.0;
                if d.length_squared() <= 1.0 {
                    break;
                }
            }
            let p = self.local(at) + d * radius;
            if p.cmplt(Vec3::ZERO).any()
                || p.cmpge(size).any()
                || self.sim.is_solid(self.sim.cell_of(p))
            {
                continue;
            }
            self.sim.pos.push(p);
            self.sim.vel.push(velocity);
        }
        self.sim.pos.len() - before
    }

    /// Take water out: remove particles within `radius` of `at`. Returns
    /// how many were removed.
    pub fn drain(&mut self, at: Vec3, radius: f32) -> usize {
        let c = self.local(at);
        let before = self.sim.pos.len();
        let mut k = 0;
        while k < self.sim.pos.len() {
            if self.sim.pos[k].distance(c) <= radius {
                self.sim.pos.swap_remove(k);
                self.sim.vel.swap_remove(k);
            } else {
                k += 1;
            }
        }
        before - self.sim.pos.len()
    }

    /// Particles in the box.
    pub fn particle_count(&self) -> usize {
        self.sim.pos.len()
    }

    /// Water in the box, cubic metres (each particle stands for an eighth
    /// of a cell).
    pub fn volume(&self) -> f32 {
        let per = self.sim.h.powi(3) / SEED_PER_SIDE.pow(3) as f32;
        self.sim.pos.len() as f32 * per
    }

    /// Particles that left the box (over its edge, or flung out) since the
    /// last call. They are gone; a game can re-add them elsewhere.
    pub fn take_lost(&mut self) -> usize {
        std::mem::take(&mut self.sim.lost)
    }

    /// The middle of the water, world space (None when empty).
    pub fn center(&self) -> Option<Vec3> {
        if self.sim.pos.is_empty() {
            return None;
        }
        let sum: Vec3 = self.sim.pos.iter().copied().sum();
        Some(sum / self.sim.pos.len() as f32 + self.settings.origin)
    }

    /// The fastest particle, m/s.
    pub fn max_speed(&self) -> f32 {
        self.sim.max_speed()
    }

    /// Particle positions in world space (for spray, debugging, sounds).
    pub fn particles(&self) -> impl Iterator<Item = Vec3> + '_ {
        let o = self.settings.origin;
        self.sim.pos.iter().map(move |p| *p + o)
    }

    /// Whether `p` is in water (the cell holds water after the last step).
    pub fn is_water_at(&self, p: Vec3) -> bool {
        let c = self.sim.cell_of(self.local(p));
        self.sim
            .cell_index(c)
            .is_some_and(|i| self.sim.cells[i] == Cell::Water)
    }

    /// Water velocity at `p` (zero outside the box).
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        let l = self.local(p);
        if l.cmplt(Vec3::ZERO).any() || l.cmpgt(self.settings.size()).any() {
            return Vec3::ZERO;
        }
        self.sim.velocity_at(l)
    }

    /// Whether the fluid is being simulated (see `World::set_fluid_awake`).
    pub fn is_awake(&self) -> bool {
        self.awake
    }

    pub fn set_awake(&mut self, awake: bool) {
        self.awake = awake;
    }

    /// Pressure iterations the last solver step needed.
    pub fn last_pressure_iterations(&self) -> usize {
        self.sim.last_iterations
    }

    /// Advance by `dt` seconds. `accel(p)` is the acceleration at world
    /// point `p` (gravity, a field); it is read once per frame at cells
    /// holding water and next to them. Splits into solver steps that keep
    /// particles under one cell per step, at most `max_substeps`.
    pub fn step(&mut self, dt: f32, accel: impl Fn(Vec3) -> Vec3) {
        if !self.awake || self.sim.pos.is_empty() || !(dt.is_finite() && dt > 0.0) {
            return;
        }
        self.refresh_accel(&accel);
        let h = self.sim.h;
        let a_max = self
            .sim
            .accel
            .iter()
            .map(|a| a.length())
            .fold(0.0, f32::max);
        let max_steps = self.settings.max_substeps.max(1);
        let mut left = dt;
        let mut steps = 0;
        while left > 1e-6 && steps < max_steps {
            let v = self.sim.max_speed() + a_max * left;
            let cfl = if v > 1e-6 { h / v } else { left };
            let remaining_steps = (max_steps - steps) as f32;
            let sub = cfl.max(left / remaining_steps).min(left);
            self.sim.step(sub, self.settings.pressure_iterations);
            left -= sub;
            steps += 1;
        }
    }

    fn refresh_accel(&mut self, accel: &impl Fn(Vec3) -> Vec3) {
        let n = self.sim.n;
        let mut near = vec![false; self.sim.accel.len()];
        for p in &self.sim.pos {
            let c = self.sim.cell_of(*p);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(i) = self.sim.cell_index(c + IVec3::new(dx, dy, dz)) {
                            near[i] = true;
                        }
                    }
                }
            }
        }
        for z in 0..n.z as i32 {
            for y in 0..n.y as i32 {
                for x in 0..n.x as i32 {
                    let c = IVec3::new(x, y, z);
                    let i = self.sim.cell_index(c).expect("inside");
                    self.sim.accel[i] = if near[i] && !self.sim.solid[i] {
                        let a = accel(self.sim.cell_center(c) + self.settings.origin);
                        if a.is_finite() {
                            a
                        } else {
                            Vec3::ZERO
                        }
                    } else {
                        Vec3::ZERO
                    };
                }
            }
        }
    }

    /// The water's surface as a mesh in the box's own space (place it at
    /// [`Fluid::origin`]). `World` builds and draws this for you.
    pub fn surface_mesh(&self) -> EngineResult<Mesh> {
        surface::build(&self.sim, self.settings.color)
    }
}

#[cfg(test)]
mod tests;
