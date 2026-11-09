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

fn sphere(mesh: &BuiltMesh) -> (Vec3, f32) {
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
        let near = |c: Vec3, r: f32, hits: &mut Vec<usize>| {
            hits.clear();
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
        // What to look at, decided without borrowing the caches.
        let mut wanted: Vec<(usize, EntityId, Mat4)> = Vec::new();
        let mut hits = Vec::new();
        let mut places: Vec<Mat4> = Vec::new();
        for id in &self.order {
            if self.ground.ignored.contains(id) {
                continue;
            }
            let Some(e) = self.entities.get(id) else {
                continue;
            };
            if matches!(e.material, Some(SurfaceMaterialRef::Water(_))) {
                continue;
            }
            let owner = e.instance_of.unwrap_or(*id);
            places.clear();
            match lod_level.get(id) {
                Some((_, level)) if *level > 0 => continue,
                Some((lod, _)) => places.extend(self.lods[lod].places.iter().map(|i| i.transform)),
                None if e.instanced => places.extend(e.instances.iter().map(|i| i.transform)),
                None => places.push(e.transform),
            }
            if places.is_empty() {
                continue;
            }
            let sphere = match self.ground.spheres.get(&owner) {
                Some(s) => *s,
                None => {
                    let mesh = match self.entities.get(&owner) {
                        Some(o) => &o.mesh,
                        None => continue,
                    };
                    let s = sphere(mesh);
                    self.ground.spheres.insert(owner, s);
                    s
                }
            };
            if sphere.1 < 0.0 {
                continue;
            }
            for m in &places {
                near(
                    m.transform_point3(sphere.0),
                    sphere.1 * max_scale(m),
                    &mut hits,
                );
                for &k in &hits {
                    wanted.push((k, owner, *m));
                }
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
