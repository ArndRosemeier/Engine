//! Letting go of a mesh's CPU copy once the renderer has it.
//!
//! A world keeps every mesh as built ([`crate::mesh::BuiltMesh`]) so that
//! [`World::shatter`], [`World::add_body`], openings and the debugging tools
//! can read it. Most meshes never need that: a town's blocks are drawn and
//! stood on, nothing more. With [`CpuMeshes::Release`] the world drops the
//! copy right after the upload and keeps only what feet stand on (the
//! upward faces, see [`World::ground_below`]), the counts and the bounds.

use super::{EntityId, SurfaceMaterialRef, World};
use crate::error::{EngineError, EngineResult};
use crate::ground_probe::GroundMesh;
use crate::mesh::BuiltMesh;
use glam::Vec3;
use std::collections::BTreeSet;
use std::sync::Arc;

/// What a world keeps of a mesh in RAM once the renderer has uploaded it.
/// Set with [`World::set_cpu_meshes`]; it applies to entities spawned from
/// then on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CpuMeshes {
    /// Keep every mesh (the default): everything works on every entity.
    #[default]
    Keep,
    /// Drop the copy after the upload. The entity draws as before and feet
    /// still stand on it, but [`World::shatter`] and [`World::add_body`]
    /// refuse it, and the z-fighting finder and GPU byte check skip it
    /// (copies are kept while the `ENGINE_ZFIGHT` watcher runs). Keep the ones you will need with [`World::keep_cpu_mesh`]
    /// (or shatter them before the next frame is drawn). Saves about as
    /// much RAM as the meshes take on the graphics card.
    Release,
}

/// What is still known of a mesh whose CPU copy is gone.
#[derive(Clone, Debug)]
pub(crate) struct Released {
    pub vertex_count: usize,
    pub index_count: usize,
    /// Local bounds of the positions (`lo > hi` for an empty mesh).
    pub lo: Vec3,
    pub hi: Vec3,
}

#[derive(Debug, Default)]
pub(crate) struct CpuMeshState {
    policy: CpuMeshes,
    /// Entities whose copy goes once the renderer has it (sorted, so the
    /// work is the same every run).
    pending: BTreeSet<EntityId>,
    released: usize,
}

impl CpuMeshState {
    pub(crate) fn spawned(&mut self, id: EntityId) {
        if self.policy == CpuMeshes::Release {
            self.pending.insert(id);
        }
    }

    pub(crate) fn forget(&mut self, id: EntityId) {
        self.pending.remove(&id);
    }
}

fn bounds(mesh: &BuiltMesh) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &mesh.positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    (lo, hi)
}

impl World {
    /// Whether meshes spawned from now on keep their CPU copy after the
    /// upload. See [`CpuMeshes`].
    pub fn set_cpu_meshes(&mut self, policy: CpuMeshes) {
        self.cpu_meshes.policy = policy;
    }

    /// The policy set with [`Self::set_cpu_meshes`].
    pub fn cpu_meshes(&self) -> CpuMeshes {
        self.cpu_meshes.policy
    }

    /// Keep (or let go of) one entity's CPU copy, whatever the policy.
    /// Asking to keep a copy that is already gone is an error.
    pub fn keep_cpu_mesh(&mut self, id: EntityId, keep: bool) -> EngineResult<()> {
        let e = self.entities.get(&id).ok_or(EngineError::UnknownEntity)?;
        if keep {
            if e.released.is_some() {
                return Err(EngineError::InvalidValue(format!(
                    "entity {id}: its CPU mesh was released already"
                )));
            }
            self.cpu_meshes.pending.remove(&id);
        } else if e.released.is_none() && e.instance_of.is_none() {
            self.cpu_meshes.pending.insert(id);
        }
        Ok(())
    }

    /// How many entities have let go of their CPU mesh so far, and how many
    /// wait for their upload to do so.
    pub fn cpu_mesh_counts(&self) -> (usize, usize) {
        (self.cpu_meshes.released, self.cpu_meshes.pending.len())
    }

    /// Bytes the world holds in CPU copies of entity meshes (positions,
    /// normals, colours, UVs, indices and per-vertex extras).
    pub fn cpu_mesh_bytes(&self) -> usize {
        self.entities
            .values()
            .map(|e| {
                let m = &e.mesh;
                m.positions.capacity() * std::mem::size_of::<Vec3>()
                    + m.normals.capacity() * std::mem::size_of::<Vec3>()
                    + m.colors.capacity() * std::mem::size_of::<glam::Vec4>()
                    + m.uvs.capacity() * 8
                    + m.indices.capacity() * 4
                    + m.elements.capacity() * 4
                    + m.surface_overrides.capacity()
                        * std::mem::size_of::<crate::mesh::SurfaceMaterial>()
            })
            .sum()
    }

    /// Error for tools that need the mesh of an entity that let it go.
    pub(crate) fn needs_cpu_mesh(&self, id: EntityId, what: &str) -> EngineResult<()> {
        match self.entities.get(&id) {
            Some(e) if e.released.is_some() => Err(EngineError::InvalidValue(format!(
                "{what}: entity {id} released its CPU mesh (CpuMeshes::Release); \
                 keep it with World::keep_cpu_mesh"
            ))),
            Some(_) => Ok(()),
            None => Err(EngineError::UnknownEntity),
        }
    }

    /// Called by the renderer after a sync: drop the CPU copies of pending
    /// entities that `uploaded` says are on the GPU. What feet stand on is
    /// worked out first (on worker threads), so foot IK keeps working.
    pub(crate) fn release_uploaded(&mut self, uploaded: impl Fn(EntityId) -> bool) {
        if self.cpu_meshes.pending.is_empty() || self.zfight_watching() {
            return;
        }
        let ready: Vec<EntityId> = self
            .cpu_meshes
            .pending
            .iter()
            .copied()
            .filter(|id| uploaded(*id))
            .collect();
        if ready.is_empty() {
            return;
        }
        let mut taken: Vec<(EntityId, BuiltMesh, bool)> = Vec::with_capacity(ready.len());
        for id in ready {
            self.cpu_meshes.pending.remove(&id);
            let walkable = !self.ground.is_ignored(id);
            let Some(e) = self.entities.get_mut(&id) else {
                continue;
            };
            if e.released.is_some() || e.instance_of.is_some() {
                continue;
            }
            let water = matches!(e.material, Some(SurfaceMaterialRef::Water(_)));
            let mesh = std::mem::take(&mut e.mesh);
            let (lo, hi) = bounds(&mesh);
            e.released = Some(Released {
                vertex_count: mesh.vertex_count(),
                index_count: mesh.index_count(),
                lo,
                hi,
            });
            self.cpu_meshes.released += 1;
            taken.push((id, mesh, walkable && !water));
        }
        // Ground meshes off the main thread; the copies are freed there too.
        use rayon::prelude::*;
        let ground: Vec<_> = taken
            .into_par_iter()
            .map(|(id, mesh, walkable)| {
                let made = walkable.then(|| {
                    (
                        super::world_ground::sphere(&mesh),
                        GroundMesh::build(&mesh).map(Arc::new),
                    )
                });
                drop(mesh);
                (id, made)
            })
            .collect();
        for (id, made) in ground {
            if let Some((sphere, mesh)) = made {
                self.ground.remember(id, sphere, mesh);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::mesh::Mesh;

    fn slab() -> Mesh {
        Mesh::box_at(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(4.0, 1.0, 4.0),
            Color::WHITE,
        )
        .unwrap()
    }

    #[test]
    fn released_meshes_keep_counts_bounds_and_ground() {
        let mut world = World::new();
        world.set_cpu_meshes(CpuMeshes::Release);
        let id = world.spawn(slab());
        let kept = world.spawn(slab());
        world.keep_cpu_mesh(kept, true).unwrap();
        let (v, i) = {
            let e = world.entity(id).unwrap();
            (e.vertex_count(), e.index_count())
        };
        assert!(v > 0 && i > 0);
        assert_eq!(world.cpu_mesh_counts(), (0, 1));
        let before = world.cpu_mesh_bytes();
        // Not uploaded yet: nothing goes.
        world.release_uploaded(|_| false);
        assert!(world.entity(id).unwrap().has_cpu_mesh());
        world.release_uploaded(|_| true);
        let e = world.entity(id).unwrap();
        assert!(!e.has_cpu_mesh());
        assert_eq!(e.mesh().vertex_count(), 0);
        assert_eq!((e.vertex_count(), e.index_count()), (v, i));
        assert!(world.entity(kept).unwrap().has_cpu_mesh());
        assert_eq!(world.cpu_mesh_counts(), (1, 0));
        assert!(world.cpu_mesh_bytes() < before);
        // Feet still stand on it.
        let y = world.ground_below(Vec3::new(0.5, 1.0, 0.5), 3.0).unwrap();
        assert!(y.abs() < 1e-4, "ground at {y}");
        // Tools that need the mesh say why they refuse.
        let err = world
            .shatter(id, Vec3::ZERO, crate::debris::Shatter::default())
            .unwrap_err();
        assert!(format!("{err}").contains("keep_cpu_mesh"), "{err}");
        assert!(world.keep_cpu_mesh(id, true).is_err());
        assert!(world
            .shatter(kept, Vec3::ZERO, crate::debris::Shatter::default())
            .is_ok());
    }

    #[test]
    fn keep_is_the_default() {
        let mut world = World::new();
        let id = world.spawn(slab());
        world.release_uploaded(|_| true);
        assert!(world.entity(id).unwrap().has_cpu_mesh());
        assert_eq!(world.cpu_mesh_counts(), (0, 0));
    }

    #[test]
    fn despawned_entities_leave_the_queue() {
        let mut world = World::new();
        world.set_cpu_meshes(CpuMeshes::Release);
        let id = world.spawn(slab());
        world.despawn(id);
        assert_eq!(world.cpu_mesh_counts(), (0, 0));
    }
}
