use crate::camera::Camera;
use crate::color::Color;
use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::{BuiltMesh, Mesh};
use crate::place::Place;
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::fmt;

/// Opaque handle to a spawned entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId(u64);

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug)]
pub struct Light {
    /// Direction toward the light (will be normalized).
    pub direction: Vec3,
    pub color: Vec3,
    pub ambient: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.4, 1.0, 0.25),
            color: Vec3::splat(1.0),
            ambient: 0.22,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub(crate) mesh: BuiltMesh,
    pub(crate) transform: Mat4,
    /// If non-empty, drawn with GPU instancing.
    pub(crate) instances: Vec<Mat4>,
}

impl Entity {
    pub fn mesh(&self) -> &BuiltMesh {
        &self.mesh
    }

    pub fn transform(&self) -> Mat4 {
        self.transform
    }
}

/// Scene state visible to user update callbacks.
#[derive(Debug)]
pub struct World {
    pub camera: Camera,
    pub light: Light,
    pub clear_color: Color,
    pub(crate) limits: EngineLimits,
    next_id: u64,
    entities: HashMap<EntityId, Entity>,
    order: Vec<EntityId>,
    /// Chunk key -> entity id for streamed volume meshes (advanced).
    pub(crate) chunk_entities: HashMap<glam::IVec3, EntityId>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            light: Light::default(),
            clear_color: Color::rgb(133, 184, 235),
            limits: EngineLimits::default(),
            next_id: 1,
            entities: HashMap::new(),
            order: Vec::new(),
            chunk_entities: HashMap::new(),
        }
    }
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limits(mut self, limits: EngineLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn limits(&self) -> &EngineLimits {
        &self.limits
    }

    /// Spawn a mesh at the origin.
    pub fn spawn(&mut self, mesh: Mesh) -> EntityId {
        self.place(mesh, Place::default())
            .expect("spawn at identity")
    }

    /// Spawn a mesh at a friendly [`Place`].
    pub fn place(&mut self, mesh: Mesh, place: Place) -> EngineResult<EntityId> {
        Ok(self.spawn_built(mesh.build(), place.to_matrix()))
    }

    /// Spawn one mesh at many places (GPU instancing).
    pub fn spawn_many(&mut self, mesh: Mesh, places: impl Into<Vec<Place>>) -> EngineResult<EntityId> {
        let places = places.into();
        if places.len() as u64 > self.limits.max_instances_per_spawn {
            return Err(EngineError::ResourceLimit(format!(
                "spawn_many has {} instances (limit {})",
                places.len(),
                self.limits.max_instances_per_spawn
            )));
        }
        let instances: Vec<Mat4> = places.into_iter().map(Place::to_matrix).collect();
        Ok(self.spawn_built_instanced(mesh.build(), instances))
    }

    pub(crate) fn spawn_built(&mut self, mesh: BuiltMesh, transform: Mat4) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh,
                transform,
                instances: Vec::new(),
            },
        );
        self.order.push(id);
        id
    }

    pub(crate) fn spawn_built_instanced(
        &mut self,
        mesh: BuiltMesh,
        instances: Vec<Mat4>,
    ) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(
            id,
            Entity {
                mesh,
                transform: Mat4::IDENTITY,
                instances,
            },
        );
        self.order.push(id);
        id
    }

    pub fn despawn(&mut self, id: EntityId) {
        self.entities.remove(&id);
        self.order.retain(|x| *x != id);
        self.chunk_entities.retain(|_, eid| *eid != id);
    }

    pub fn set_place(&mut self, id: EntityId, place: Place) -> EngineResult<()> {
        let e = self
            .entities
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.transform = place.to_matrix();
        Ok(())
    }

    /// Orbit camera helper (yaw/pitch in degrees).
    pub fn look_orbit(
        &mut self,
        target: impl Into<Vec3>,
        distance: f32,
        yaw_degrees: f32,
        pitch_degrees: f32,
    ) {
        self.camera = Camera::orbit(target, distance, yaw_degrees, pitch_degrees);
    }

    /// Set sun direction (need not be normalized) and ambient 0..1.
    pub fn set_sun(&mut self, direction: impl Into<Vec3>, ambient: f32) {
        self.light.direction = direction.into();
        self.light.ambient = ambient.clamp(0.0, 1.0);
    }

    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    pub fn entity(&self, id: EntityId) -> EngineResult<&Entity> {
        self.entities.get(&id).ok_or(EngineError::UnknownEntity)
    }

    pub fn entity_mut(&mut self, id: EntityId) -> EngineResult<&mut Entity> {
        self.entities.get_mut(&id).ok_or(EngineError::UnknownEntity)
    }

    pub(crate) fn entities(&self) -> impl Iterator<Item = (EntityId, &Entity)> {
        self.order
            .iter()
            .filter_map(|id| self.entities.get(id).map(|e| (*id, e)))
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Advanced: replace/create a chunk mesh entity (streaming).
    #[allow(dead_code)]
    pub(crate) fn upsert_chunk_mesh(&mut self, key: glam::IVec3, mesh: BuiltMesh) -> EntityId {
        if let Some(&id) = self.chunk_entities.get(&key) {
            let e = self.entities.get_mut(&id).expect("chunk entity");
            e.mesh = mesh;
            e.transform = Mat4::IDENTITY;
            e.instances.clear();
            id
        } else {
            let id = self.spawn_built(mesh, Mat4::IDENTITY);
            self.chunk_entities.insert(key, id);
            id
        }
    }

    /// Advanced: remove a streamed chunk entity.
    #[allow(dead_code)]
    pub(crate) fn remove_chunk(&mut self, key: glam::IVec3) {
        if let Some(id) = self.chunk_entities.remove(&key) {
            self.despawn(id);
        }
    }
}

/// Per-frame timing passed to the update closure.
#[derive(Clone, Debug)]
pub struct Frame {
    pub dt: f32,
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub aspect: f32,
    /// True only on the first update after the window is ready.
    pub first: bool,
}
