//! Rooms and openings on the World: see [`crate::rooms`].

use super::{EntityId, World};
use crate::error::{EngineError, EngineResult};
use crate::rooms::{Opening, OpeningId, Room, RoomId};

impl World {
    /// Add a room (a box). Put things in it with [`World::set_room`].
    pub fn add_room(&mut self, room: Room) -> EngineResult<RoomId> {
        room.validate()?;
        Ok(self.rooms.add_room(room, None))
    }

    /// Remove a room and its openings. What was in it is drawn as usual.
    pub fn remove_room(&mut self, id: RoomId) {
        self.rooms.remove_room(id);
        self.entity_rooms.retain(|_, r| *r != id);
    }

    pub fn room(&self, id: RoomId) -> Option<&Room> {
        self.rooms.room(id)
    }

    /// The smallest room containing `point`.
    pub fn room_at(&self, point: impl Into<glam::Vec3>) -> Option<RoomId> {
        self.rooms.room_at(point.into())
    }

    /// Add an opening (a door, window or hole), open or closed.
    pub fn add_opening(&mut self, opening: Opening, open: bool) -> EngineResult<OpeningId> {
        self.rooms.add_opening(opening, open, None)
    }

    /// Open or close an opening (a door swinging, a wall blown out).
    pub fn set_opening_open(&mut self, id: OpeningId, open: bool) -> EngineResult<()> {
        self.rooms.set_open(id, open)
    }

    pub fn remove_opening(&mut self, id: OpeningId) {
        self.rooms.remove_opening(id);
    }

    /// Put an entity (mesh or animated character) in a room, or take it out
    /// (`None`): it is drawn only while the room can be seen.
    pub fn set_room(&mut self, id: EntityId, room: Option<RoomId>) -> EngineResult<()> {
        if !self.entities.contains_key(&id) && !self.animated.contains_key(&id) {
            return Err(EngineError::UnknownEntity);
        }
        match room {
            Some(r) => {
                if self.rooms.room(r).is_none() {
                    return Err(EngineError::InvalidValue("no such room".into()));
                }
                self.entity_rooms.insert(id, r);
            }
            None => {
                self.entity_rooms.remove(&id);
            }
        }
        Ok(())
    }

    /// Put an entity in the room its origin is in (or none).
    pub fn set_room_by_position(&mut self, id: EntityId) -> EngineResult<Option<RoomId>> {
        let m = self.matrix_of(id).ok_or(EngineError::UnknownEntity)?;
        let room = self.rooms.room_at(m.w_axis.truncate());
        self.set_room(id, room)?;
        Ok(room)
    }

    pub fn room_of(&self, id: EntityId) -> Option<RoomId> {
        self.entity_rooms.get(&id).copied()
    }

    /// Turn indoor culling off (everything in rooms is drawn) or on (default).
    pub fn set_room_culling(&mut self, on: bool) {
        self.rooms.enabled = on;
    }

    pub fn room_culling(&self) -> bool {
        self.rooms.enabled
    }

    /// Rooms the camera could see last frame.
    pub fn visible_rooms(&self) -> Vec<RoomId> {
        self.rooms
            .visible
            .iter()
            .enumerate()
            .filter(|(i, v)| **v && self.rooms.rooms[*i].is_some())
            .map(|(i, _)| RoomId(i as u32))
            .collect()
    }

    /// Rooms, openings, and rooms seen last frame.
    pub fn room_count(&self) -> (usize, usize, usize) {
        let (r, o) = self.rooms.count();
        (r, o, self.visible_rooms().len())
    }

    /// Work out which rooms the camera sees (the engine does this before
    /// every frame; call it yourself in a headless world).
    pub fn update_room_visibility(&mut self, aspect: f32) {
        if self.rooms.rooms.is_empty() {
            return;
        }
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            1.0
        };
        let vp = self.camera().view_projection(aspect);
        let eye = self.camera().eye();
        self.rooms.visible = self.rooms.visibility(eye, vp);
        self.rooms.computed = true;
    }

    /// Is this entity in a room the camera cannot see?
    pub(crate) fn room_hidden(&self, id: EntityId) -> bool {
        if !self.rooms.enabled || !self.rooms.computed {
            return false;
        }
        self.entity_rooms
            .get(&id)
            .is_some_and(|r| self.rooms.visible.get(r.0 as usize) == Some(&false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::{BuildingPlan, ElementId, ElementKind, TowerSpec};
    use crate::camera::Camera;
    use crate::color::rgb;
    use crate::mesh::Mesh;
    use crate::place::Place;
    use glam::Vec3;
    use std::sync::Arc;

    fn look(world: &mut World, eye: Vec3, at: Vec3) {
        world.camera = Camera::look_at(eye, at);
        world.update_room_visibility(16.0 / 9.0);
    }

    #[test]
    fn a_tower_hides_its_rooms_until_a_wall_breaks() {
        let plan = Arc::new(BuildingPlan::tower(&TowerSpec::default()).unwrap());
        assert_eq!(plan.rooms().len(), 12, "front and back on six floors");
        let mut world = World::default();
        let tower = world
            .spawn_building(plan.clone(), Place::new(0.0, 0.0, 20.0))
            .unwrap();
        let (rooms, openings, _) = world.room_count();
        assert_eq!(rooms, 12);
        assert!(openings > 100, "walls, slabs and doorways: {openings}");
        // A sofa in the ground-floor front room.
        let sofa = world.spawn(Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(200, 0, 0)).unwrap());
        world.set_place(sofa, Place::new(0.0, 0.5, 16.0)).unwrap();
        let room = world
            .set_room_by_position(sofa)
            .unwrap()
            .expect("in a room");

        // From the street, solid walls hide everything.
        look(
            &mut world,
            Vec3::new(0.0, 1.7, -10.0),
            Vec3::new(0.0, 1.7, 20.0),
        );
        assert_eq!(world.room_count().2, 0);
        assert!(world.room_hidden(sofa));

        // Break the wall panel in front of the sofa: that room shows.
        let front: Vec<ElementId> = plan
            .elements()
            .iter()
            .enumerate()
            .filter(|(_, e)| e.floor == 0 && e.outward.z < -0.9 && e.center.x.abs() < 2.0)
            .map(|(i, _)| ElementId(i as u32))
            .collect();
        world.break_building_elements(tower, &front).unwrap();
        look(
            &mut world,
            Vec3::new(0.0, 1.7, -10.0),
            Vec3::new(0.0, 1.7, 20.0),
        );
        let seen = world.visible_rooms();
        assert!(seen.contains(&room), "{seen:?}");
        assert!(!world.room_hidden(sofa));
        assert!(
            seen.len() <= 2,
            "only the room behind the hole, and maybe through its doorway: {seen:?}"
        );

        // Culling off draws everything.
        world.set_room_culling(false);
        look(
            &mut world,
            Vec3::new(0.0, 1.7, -10.0),
            Vec3::new(0.0, -10.0, -40.0),
        );
        assert!(!world.room_hidden(sofa));
        world.set_room_culling(true);
        assert!(world.room_hidden(sofa), "looking away");

        // The building goes: its rooms go, the sofa is drawn again.
        world.despawn(tower);
        assert_eq!(world.room_count().0, 0);
        assert_eq!(world.room_of(sofa), None);
    }

    #[test]
    fn windows_are_open_and_inside_sees_its_room() {
        let spec = TowerSpec {
            glazing: 1.0,
            ..Default::default()
        };
        let plan = Arc::new(BuildingPlan::tower(&spec).unwrap());
        assert!(plan
            .elements()
            .iter()
            .any(|e| e.kind == ElementKind::Window));
        let mut world = World::default();
        world
            .spawn_building(plan, Place::new(0.0, 0.0, 20.0))
            .unwrap();
        look(
            &mut world,
            Vec3::new(0.0, 1.7, -10.0),
            Vec3::new(0.0, 1.7, 20.0),
        );
        let from_street = world.visible_rooms().len();
        assert!(from_street >= 2, "rooms behind the glass: {from_street}");
        // Standing in the third-floor back room looking at its back wall.
        let eye = Vec3::new(0.0, 2.0 * 3.4 + 1.7, 23.0);
        let here = world.room_at(eye).expect("inside");
        look(&mut world, eye, eye + Vec3::Z);
        let seen = world.visible_rooms();
        assert!(seen.contains(&here));
        assert!(seen.len() < 6, "not the whole tower: {seen:?}");
    }
}
