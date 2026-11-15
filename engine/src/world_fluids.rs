//! Three-dimensional water boxes in the world: stepped with the world's
//! gravity or gravity field, drawn as a see-through surface, and pushing on
//! the bodies that fall in.

use super::{EntityId, World};
use crate::error::{EngineError, EngineResult};
use crate::fluid::Fluid;
use crate::physics::FieldBody;
use crate::place::Place;

/// A water box added with [`World::add_fluid`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FluidId(u32);

impl FluidId {
    /// The number that [`FieldBody::Water`] probes carry as their `id`.
    pub fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug)]
struct Slot {
    id: FluidId,
    entity: Option<EntityId>,
    frames: u32,
}

/// The world's fluids. `list` and `slots` stay in step.
#[derive(Debug, Default)]
pub(crate) struct Fluids {
    pub(crate) list: Vec<Fluid>,
    slots: Vec<Slot>,
    next: u32,
}

impl Fluids {
    fn index(&self, id: FluidId) -> Option<usize> {
        self.slots.iter().position(|s| s.id == id)
    }
}

impl World {
    /// Add a water box. From now on the world steps it every frame with
    /// the gravity that bodies feel (the gravity field when one is set:
    /// [`FieldBody::Water`] probes, one per cell holding water), draws its
    /// surface, and lets it carry and slow the dynamic bodies inside it.
    pub fn add_fluid(&mut self, fluid: Fluid) -> EngineResult<FluidId> {
        if self.fluids.list.len() >= 64 {
            return Err(EngineError::ResourceLimit("at most 64 fluid boxes".into()));
        }
        self.fluids.next += 1;
        let id = FluidId(self.fluids.next);
        self.fluids.list.push(fluid);
        self.fluids.slots.push(Slot {
            id,
            entity: None,
            frames: 0,
        });
        self.remesh_fluid(self.fluids.list.len() - 1)?;
        Ok(id)
    }

    /// Remove a water box and its surface.
    pub fn remove_fluid(&mut self, id: FluidId) -> Option<Fluid> {
        let i = self.fluids.index(id)?;
        let slot = self.fluids.slots.remove(i);
        if let Some(e) = slot.entity {
            self.despawn(e);
        }
        Some(self.fluids.list.remove(i))
    }

    pub fn fluid(&self, id: FluidId) -> Option<&Fluid> {
        self.fluids.index(id).map(|i| &self.fluids.list[i])
    }

    /// Change a fluid: pour, drain, dig rock out of its box.
    pub fn fluid_mut(&mut self, id: FluidId) -> Option<&mut Fluid> {
        let i = self.fluids.index(id)?;
        Some(&mut self.fluids.list[i])
    }

    pub fn fluid_ids(&self) -> Vec<FluidId> {
        self.fluids.slots.iter().map(|s| s.id).collect()
    }

    /// Pause a fluid (far from the player) or wake it. A sleeping fluid
    /// keeps its water and surface but costs nothing.
    pub fn set_fluid_awake(&mut self, id: FluidId, awake: bool) -> EngineResult<()> {
        self.fluid_mut(id)
            .ok_or_else(|| EngineError::InvalidValue(format!("unknown fluid {id:?}")))?
            .set_awake(awake);
        Ok(())
    }

    /// The entity that draws a fluid's surface (for a custom material);
    /// it changes when the surface is rebuilt.
    pub fn fluid_entity(&self, id: FluidId) -> Option<EntityId> {
        self.fluids
            .index(id)
            .and_then(|i| self.fluids.slots[i].entity)
    }

    fn remesh_fluid(&mut self, i: usize) -> EngineResult<()> {
        let fluid = &self.fluids.list[i];
        let mesh = fluid.surface_mesh()?;
        let origin = fluid.origin();
        let old = self.fluids.slots[i].entity.take();
        let material = old.and_then(|e| self.entities.get(&e).and_then(|e| e.material));
        if let Some(e) = old {
            self.despawn(e);
        }
        if mesh.face_count() == 0 {
            return Ok(());
        }
        let e = self.spawn(mesh);
        self.set_place(e, Place::new(origin.x, origin.y, origin.z))?;
        self.set_casts_shadow(e, false)?;
        if let Some(m) = material {
            if let Some(ent) = self.entities.get_mut(&e) {
                ent.material = Some(m);
            }
        }
        self.fluids.slots[i].entity = Some(e);
        Ok(())
    }

    /// Step every awake fluid by `dt` and rebuild surfaces that are due.
    /// Runs after the physics step.
    pub(crate) fn tick_fluids(&mut self, dt: f32) {
        for i in 0..self.fluids.list.len() {
            if !self.fluids.list[i].is_awake() {
                continue;
            }
            let raw = self.fluids.slots[i].id.raw() as u64;
            let physics = &self.physics;
            self.fluids.list[i].step(dt, |p| physics.gravity_for(raw, FieldBody::Water, p, 1.0));
            let slot = &mut self.fluids.slots[i];
            slot.frames += 1;
            let every = self.fluids.list[i].settings().mesh_every.max(1);
            if slot.frames >= every {
                slot.frames = 0;
                if let Err(e) = self.remesh_fluid(i) {
                    eprintln!("fluid surface: {e}");
                }
            }
        }
    }
}
