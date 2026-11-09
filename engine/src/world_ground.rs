//! Standing on what is drawn: [`World::ground_below`] and the meshes foot
//! IK reads (see [`crate::ground_probe`]).

use super::{EntityId, SurfaceMaterialRef, World};
use crate::ground_probe::{GroundMesh, GroundPatch, Placed};
use crate::mesh::BuiltMesh;
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Ground meshes made so far, by the entity owning the mesh, and which
/// entities feet ignore.
#[derive(Debug, Default)]
pub(crate) struct GroundState {
    /// `None`: the mesh has no upward faces.
    meshes: HashMap<EntityId, Option<Arc<GroundMesh>>>,
    /// Bounding sphere of each mesh in its own space.
    spheres: HashMap<EntityId, (Vec3, f32)>,
    /// Entities that are not ground (grass blades, glass, effects).
    ignored: HashSet<EntityId>,
}

pub(crate) fn sphere(mesh: &BuiltMesh) -> (Vec3, f32) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &mesh.positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    if lo.x > hi.x {
        return (Vec3::ZERO, -1.0);
    }
    ((lo + hi) * 0.5, (hi - lo).length() * 0.5)
}

fn max_scale(m: &Mat4) -> f32 {
    m.x_axis
        .truncate()
        .length()
        .max(m.y_axis.truncate().length())
        .max(m.z_axis.truncate().length())
}

impl GroundState {
    /// Bytes held in ground meshes.
    pub(crate) fn bytes(&self) -> usize {
        self.meshes.values().flatten().map(|m| m.bytes()).sum()
    }

    pub(crate) fn is_ignored(&self, id: EntityId) -> bool {
        self.ignored.contains(&id)
    }

    /// What feet stand on for an entity whose mesh is about to go.
    pub(crate) fn remember(
        &mut self,
        id: EntityId,
        sphere: (Vec3, f32),
        mesh: Option<Arc<GroundMesh>>,
    ) {
        self.spheres.insert(id, sphere);
        self.meshes.insert(id, mesh);
    }
}

impl World {
    /// Whether feet may stand on entity `id` (default yes). Turn it off
    /// for things drawn over the ground that are not ground: grass blades,
    /// decals, glass, effects.
    pub fn set_walkable(&mut self, id: EntityId, walkable: bool) {
        if walkable {
            self.ground.ignored.remove(&id);
        } else {
            self.ground.ignored.insert(id);
        }
    }

    pub(crate) fn forget_ground(&mut self, id: EntityId) {
        self.ground.meshes.remove(&id);
        self.ground.spheres.remove(&id);
        self.ground.ignored.remove(&id);
    }

    /// Height of what is under `p`: the highest drawn surface (any mesh's
    /// upward faces, the terrain, physics bodies) no higher than `p.y` and
    /// no lower than `p.y - depth`.
    pub fn ground_below(&mut self, p: impl Into<Vec3>, depth: f32) -> Option<f32> {
        let p = p.into();
        if !p.is_finite() || depth.is_nan() || depth <= 0.0 {
            return None;
        }
        let patch = self.ground_patches(&[(p, 0.5)]).pop().unwrap_or_default();
        self.ground_below_in(&patch, p, depth)
    }

    /// [`Self::ground_below`] with the meshes gathered already.
    pub(crate) fn ground_below_in(&self, patch: &GroundPatch, p: Vec3, depth: f32) -> Option<f32> {
        let drawn = patch.height_below(p, depth);
        let other = self
            .cast_ray(p, Vec3::NEG_Y, depth)
            .map(|h| h.point.y)
            .filter(|y| *y <= p.y + 1e-4);
        match (drawn, other) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }

    /// For each disc (centre, radius) the ground meshes reaching into it,
    /// in one pass over the world.
    pub(crate) fn ground_patches(&mut self, discs: &[(Vec3, f32)]) -> Vec<GroundPatch> {
        let mut out = vec![GroundPatch::default(); discs.len()];
        if discs.is_empty() {
            return out;
        }
        // LOD sets: the detailed level at every place (as zfight does).
        let mut lod_level: HashMap<EntityId, (crate::lod::LodId, usize)> = HashMap::new();
        for (id, set) in &self.lods {
            for (k, (e, _)) in set.levels.iter().enumerate() {
                lod_level.insert(*e, (*id, k));
            }
        }
        // The discs by grid square, so each placement asks only the squares
        // it covers (a town holds tens of thousands of props).
        const SQUARE: f32 = 16.0;
        let key = |x: f32, z: f32| ((x / SQUARE).floor() as i32, (z / SQUARE).floor() as i32);
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        let mut widest = 0.0f32;
        for (k, (at, reach)) in discs.iter().enumerate() {
            grid.entry(key(at.x, at.z)).or_default().push(k);
            widest = widest.max(*reach);
        }
        // All discs inside one box: most of a town's placements are far
        // outside it and are turned away here, before any grid lookup.
        let (lo, hi) = discs.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), (at, reach)| {
                (
                    lo.min(*at - Vec3::splat(*reach)),
                    hi.max(*at + Vec3::splat(*reach)),
                )
            },
        );
        let near = |c: Vec3, r: f32, hits: &mut Vec<usize>| {
            hits.clear();
            if c.x + r < lo.x
                || c.x - r > hi.x
                || c.z + r < lo.z
                || c.z - r > hi.z
                || c.y - r > hi.y + 2.0
            {
                return;
            }
            let test = |k: usize| {
                let (at, reach) = discs[k];
                let d = Vec3::new(c.x - at.x, 0.0, c.z - at.z).length();
                d <= r + reach && c.y - r <= at.y + 2.0
            };
            if r > 4.0 * SQUARE {
                hits.extend((0..discs.len()).filter(|&k| test(k)));
                return;
            }
            let span = r + widest;
            let (a, b) = (key(c.x - span, c.z - span), key(c.x + span, c.z + span));
            for x in a.0..=b.0 {
                for z in a.1..=b.1 {
                    if let Some(ks) = grid.get(&(x, z)) {
                        hits.extend(ks.iter().copied().filter(|&k| test(k)));
                    }
                }
            }
        };
        // What to look at, decided without borrowing the caches. Every
        // entity and placement is looked at, so the scan runs on the
        // frame's threads; the finds come back in entity order. (A patch
        // keeps the highest surface, so order would not change it anyway.)
        let scanned: Vec<Scanned> = {
            use rayon::prelude::*;
            let world = &*self;
            crate::parallel::frame_pool().install(|| {
                world
                    .order
                    .par_iter()
                    .map(|id| world.ground_scan(*id, &lod_level, &near))
                    .collect()
            })
        };
        let mut wanted: Vec<(usize, EntityId, Mat4)> = Vec::new();
        let mut unmeasured: Vec<(EntityId, EntityId)> = Vec::new();
        for found in scanned {
            match found {
                Scanned::Found(list) => wanted.extend(list),
                Scanned::Unmeasured { id, owner } => unmeasured.push((id, owner)),
                Scanned::Nothing => {}
            }
        }
        // Meshes never measured: measure and keep them here, then look.
        for (id, owner) in unmeasured {
            if !self.ground.spheres.contains_key(&owner) {
                let Some(o) = self.entities.get(&owner) else {
                    continue;
                };
                let s = sphere(&o.mesh);
                self.ground.spheres.insert(owner, s);
            }
            if let Scanned::Found(list) = self.ground_scan(id, &lod_level, &near) {
                wanted.extend(list);
            }
        }
        for (k, owner, m) in wanted {
            let mesh = match self.ground.meshes.get(&owner) {
                Some(g) => g.clone(),
                None => {
                    let g = self
                        .entities
                        .get(&owner)
                        .and_then(|o| GroundMesh::build(&o.mesh))
                        .map(Arc::new);
                    self.ground.meshes.insert(owner, g.clone());
                    g
                }
            };
            let Some(mesh) = mesh else { continue };
            out[k].placed.push(Placed {
                mesh,
                to_local: m.inverse(),
                to_world: m,
            });
        }
        out
    }
}

/// What one entity offers the ground under some discs.
enum Scanned {
    Nothing,
    /// (disc, owner, placement) for every placement reaching a disc.
    Found(Vec<(usize, EntityId, Mat4)>),
    /// Its mesh's bounding sphere is not known yet.
    Unmeasured {
        id: EntityId,
        owner: EntityId,
    },
}

impl World {
    /// The placements of entity `id` that reach a disc (`near` lists the
    /// discs a sphere reaches). Reads only, so many run at once.
    fn ground_scan(
        &self,
        id: EntityId,
        lod_level: &HashMap<EntityId, (crate::lod::LodId, usize)>,
        near: &(impl Fn(Vec3, f32, &mut Vec<usize>) + Sync),
    ) -> Scanned {
        if self.ground.ignored.contains(&id) {
            return Scanned::Nothing;
        }
        let Some(e) = self.entities.get(&id) else {
            return Scanned::Nothing;
        };
        if matches!(e.material, Some(SurfaceMaterialRef::Water(_))) {
            return Scanned::Nothing;
        }
        let owner = e.instance_of.unwrap_or(id);
        let places: Box<dyn Iterator<Item = &Mat4>> = match lod_level.get(&id) {
            Some((_, level)) if *level > 0 => return Scanned::Nothing,
            Some((lod, _)) => Box::new(self.lods[lod].places.iter().map(|i| &i.transform)),
            None if e.instanced => Box::new(e.instances.iter().map(|i| &i.transform)),
            None => Box::new(std::iter::once(&e.transform)),
        };
        let mut places = places.peekable();
        if places.peek().is_none() {
            return Scanned::Nothing;
        }
        let Some(&sphere) = self.ground.spheres.get(&owner) else {
            return Scanned::Unmeasured { id, owner };
        };
        if sphere.1 < 0.0 {
            return Scanned::Nothing;
        }
        let mut hits = Vec::new();
        let mut found = Vec::new();
        for m in places {
            near(
                m.transform_point3(sphere.0),
                sphere.1 * max_scale(m),
                &mut hits,
            );
            found.extend(hits.iter().map(|&k| (k, owner, *m)));
        }
        if found.is_empty() {
            Scanned::Nothing
        } else {
            Scanned::Found(found)
        }
    }
}
