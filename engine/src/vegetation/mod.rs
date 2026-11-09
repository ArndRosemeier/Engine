//! Procedural trees and shrubs with wind, ported from AssetGenerator's
//! vegetation package (which came from the "scale" game).
//!
//! [`tree_model`] grows one of ten species (and [`shrub_model`] a shrub),
//! deterministic per `(species, variant)`. Each model has three meshes for
//! [`crate::world::World::spawn_lod`]: full detail (bark tubes and
//! alpha-tested leaf cards carrying whole twig clusters), a mid level
//! (main limbs, half the leaf tufts, larger) and a far canopy hull of about
//! 150 triangles whose rim dissolves in the shader. All three sway in the
//! wind ([`crate::world::World::set_wind`]), shadows included, and turn with
//! the season ([`crate::world::World::set_season`]).
//!
//! The leaves and bark share one procedural texture atlas, so one entity
//! draws a whole tree.

mod atlas;
mod grow;
pub(crate) mod rng;
mod species;

pub use species::TreeSpecies;

use crate::color::Color;
use crate::error::EngineResult;
use crate::lod::LodLevel;
use crate::mesh::{AlbedoMap, MaterialMaps, Mesh, SurfaceMaterial};
use grow::{Part, VegVertex};
use species::Kind;

/// Shader profile ids of the vegetation surfaces (`SurfaceMaterial::profile`).
pub(crate) const PROFILE_LEAF: f32 = 8.0;
pub(crate) const PROFILE_BARK: f32 = 9.0;
pub(crate) const PROFILE_FAR: f32 = 10.0;

/// A grown tree or shrub: three detail levels and its size.
#[derive(Clone, Debug)]
pub struct TreeModel {
    /// Bark and every leaf card.
    pub near: Mesh,
    /// Trunk and main limbs, every other leaf tuft (grown to keep the crown full).
    pub mid: Mesh,
    /// Canopy hull and a few limbs, for the distance.
    pub far: Mesh,
    /// Top of the crown above the root (m).
    pub height: f32,
    pub trunk_radius: f32,
    /// Crown radius (m), handy for spacing.
    pub crown_radius: f32,
}

impl TreeModel {
    /// LOD levels for [`crate::world::World::spawn_lod`], with distances
    /// scaled to the tree's height (a 15 m tree: 45 m, 120 m, 900 m; shrubs
    /// keep their detail to at least 18 m).
    pub fn lod_levels(&self) -> Vec<LodLevel> {
        let k = (self.height / 15.0).clamp(0.4, 1.5);
        self.lod_levels_at(45.0 * k, 120.0 * k, 900.0 * k.max(0.5))
    }

    /// LOD levels ending at the given distances (metres).
    pub fn lod_levels_at(&self, near_m: f32, mid_m: f32, far_m: f32) -> Vec<LodLevel> {
        vec![
            LodLevel::new(self.near.clone(), near_m),
            LodLevel::new(self.mid.clone(), mid_m),
            LodLevel::new(self.far.clone(), far_m),
        ]
    }
}

/// Grow `species`; `variant` picks one of 64 individuals.
pub fn tree_model(species: TreeSpecies, variant: u32) -> EngineResult<TreeModel> {
    model(Kind::Tree(species), variant % 64, species.is_conifer())
}

/// Grow a shrub (odd variants are evergreen boxwood).
pub fn shrub_model(variant: u32) -> EngineResult<TreeModel> {
    model(Kind::Shrub, variant % 64, false)
}

fn model(kind: Kind, variant: u32, conifer: bool) -> EngineResult<TreeModel> {
    let g = grow::grow(kind, variant);
    let leaf_cell = match kind {
        Kind::Tree(TreeSpecies::Palm) => 12,
        _ => species::species_cfg(kind, variant).leaf_cell,
    };
    let translucency = if conifer { 0.35 } else { 0.85 };
    let near = textured_mesh(&g.near.wood, &g.near.leaves, leaf_cell, translucency)?;
    let mid = textured_mesh(&g.mid_wood, &g.mid_leaves, leaf_cell, translucency)?;
    let far = far_mesh(&g.near.far)?;
    Ok(TreeModel {
        near,
        mid,
        far,
        height: g.near.height,
        trunk_radius: g.near.trunk_radius,
        crown_radius: g.near.crown_radius,
    })
}

fn linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn surface(
    profile: f32,
    v: &VegVertex,
    roughness: f32,
    extra: f32,
    autumn: [f32; 3],
    turns: f32,
) -> SurfaceMaterial {
    SurfaceMaterial {
        roughness,
        metallic: 0.0,
        emission: 0.0,
        detail_scale: extra,
        strata_strength: v.wind,
        seed: v.tint,
        warp_strength: v.occ,
        noise_gain: v.kind,
        orientation: autumn,
        profile,
        coverage_direction: [0.0; 3],
        coverage_level: turns,
        coverage_softness: 0.0,
        coverage_strength: 0.0,
        overlay_roughness: 0.0,
        overlay_metallic: 0.0,
        glow_channel: 0,
    }
}

fn push_part(
    mesh: &mut Mesh,
    surfaces: &mut Vec<SurfaceMaterial>,
    part: &Part,
    uv_of: impl Fn(&VegVertex) -> [f32; 2],
    surf_of: impl Fn(&VegVertex) -> SurfaceMaterial,
) -> EngineResult<()> {
    let mut ids = Vec::with_capacity(part.verts.len());
    for v in &part.verts {
        let id = mesh.add_point(v.p)?;
        ids.push(id);
        mesh.set_point_normal(id, v.n)?;
        mesh.set_point_uv(id, uv_of(v))?;
        mesh.set_point_color(
            id,
            Color {
                r: v.color[0],
                g: v.color[1],
                b: v.color[2],
                a: 1.0,
            },
        )?;
        surfaces.push(surf_of(v));
    }
    for t in &part.tris {
        mesh.add_triangle(ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize])?;
    }
    Ok(())
}

fn textured_mesh(
    wood: &Part,
    leaves: &Part,
    leaf_cell: u32,
    translucency: f32,
) -> EngineResult<Mesh> {
    let mut mesh = Mesh::new();
    let mut surfaces = Vec::new();
    push_part(
        &mut mesh,
        &mut surfaces,
        wood,
        |v| v.uv,
        |v| surface(PROFILE_BARK, v, 0.92, 0.0, [0.0; 3], 0.0),
    )?;
    let aut = atlas::AUTUMN[leaf_cell as usize % 16];
    let autumn = [linear(aut[0]), linear(aut[1]), linear(aut[2])];
    push_part(
        &mut mesh,
        &mut surfaces,
        leaves,
        |v| [v.uv[0], v.uv[1] * atlas::LEAF_V_SCALE],
        |v| surface(PROFILE_LEAF, v, 0.78, translucency, autumn, aut[3]),
    )?;
    mesh.set_surface_overrides(surfaces);
    let a = atlas::atlas();
    mesh.set_albedo_rgba(atlas::ATLAS_W, atlas::ATLAS_H, a.color.clone())?;
    mesh.set_material_maps(MaterialMaps {
        normal: Some(AlbedoMap {
            width: atlas::ATLAS_W,
            height: atlas::ATLAS_H,
            rgba: a.normal.clone().into(),
        }),
        normal_scale: 1.1,
        roughness: 0.9,
        ..MaterialMaps::default()
    })?;
    Ok(mesh)
}

fn far_mesh(far: &Part) -> EngineResult<Mesh> {
    let mut mesh = Mesh::new();
    let mut surfaces = Vec::new();
    push_part(
        &mut mesh,
        &mut surfaces,
        far,
        |_| [0.0, 0.0],
        |v| surface(PROFILE_FAR, v, 0.9, 0.0, v.autumn, v.deciduous),
    )?;
    mesh.set_surface_overrides(surfaces);
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_species_grows_three_levels() {
        for s in TreeSpecies::ALL {
            let m = tree_model(s, 1).expect("tree grows");
            let (near, mid, far) = (m.near.face_count(), m.mid.face_count(), m.far.face_count());
            assert!(near > mid && mid > far, "{s:?}: {near} > {mid} > {far}");
            assert!(far < 400, "{s:?} far hull has {far} faces");
            assert!((6.0..25.0).contains(&m.height), "{s:?} height {}", m.height);
            assert!(m.near.albedo().is_some() && m.far.albedo().is_none());
            let (lo, hi) = m.near.bounds().expect("bounds");
            assert!(lo.y < 0.0 && hi.y > 5.0, "{s:?} spans the ground up");
        }
        let shrub = shrub_model(0).expect("shrub grows");
        assert!(shrub.height < 3.0);
    }

    #[test]
    fn growth_is_deterministic_and_variants_differ() {
        let a = tree_model(TreeSpecies::Oak, 3).expect("oak");
        let b = tree_model(TreeSpecies::Oak, 3).expect("oak");
        let c = tree_model(TreeSpecies::Oak, 4).expect("oak");
        assert_eq!(a.near.bounds(), b.near.bounds());
        assert_eq!(a.near.face_count(), b.near.face_count());
        assert_ne!(a.near.bounds(), c.near.bounds());
    }

    #[test]
    fn lod_distances_scale_with_height() {
        let tree = tree_model(TreeSpecies::Linden, 0).expect("linden");
        let shrub = shrub_model(0).expect("shrub");
        let (t, s) = (tree.lod_levels(), shrub.lod_levels());
        assert_eq!(t.len(), 3);
        assert!(s[0].max_distance < t[0].max_distance);
        assert!(crate::lod::LodSet::validate_levels(&t).is_ok());
    }
}
