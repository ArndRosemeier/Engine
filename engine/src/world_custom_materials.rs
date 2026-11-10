//! Materials made of a game's own shader code.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;
use crate::custom_material::{CustomMaterial, CustomMaterialId};

/// Most custom materials at once: each one is its own GPU pipeline.
const MAX_CUSTOM_MATERIALS: usize = 64;

impl World {
    /// Add a material made of WGSL code (see [`CustomMaterial`]). The code
    /// is checked now: a mistake is an error naming the line and the
    /// problem, not a crash later on the GPU.
    pub fn add_material(&mut self, material: CustomMaterial) -> EngineResult<CustomMaterialId> {
        if self.custom_materials.len() >= MAX_CUSTOM_MATERIALS {
            return Err(EngineError::ResourceLimit(format!(
                "at most {MAX_CUSTOM_MATERIALS} custom materials (each is its own pipeline): \
                 share one and vary it with set_material_param"
            )));
        }
        if material.params.iter().flatten().any(|v| !v.is_finite()) {
            return Err(EngineError::InvalidValue(format!(
                "custom material {:?}: params must be finite",
                material.name
            )));
        }
        material.check()?;
        let id = CustomMaterialId(self.next_custom_material);
        self.next_custom_material += 1;
        self.custom_materials.insert(id, (material, 0));
        Ok(id)
    }

    /// Draw `entity` with custom material `material` (the whole mesh;
    /// shadows as usual). `World::clear_custom_material` takes it off.
    pub fn set_custom_material(
        &mut self,
        entity: EntityId,
        material: CustomMaterialId,
    ) -> EngineResult<()> {
        if !self.custom_materials.contains_key(&material) {
            return Err(EngineError::InvalidValue(format!(
                "custom material {material:?} was not made by add_material in this world"
            )));
        }
        self.set_entity_material(entity, Some(SurfaceMaterialRef::Custom(material)))
    }

    /// Back to the mesh's own look.
    pub fn clear_custom_material(&mut self, entity: EntityId) -> EngineResult<()> {
        match self.entity(entity)?.material {
            Some(SurfaceMaterialRef::Custom(_)) => self.set_entity_material(entity, None),
            _ => Ok(()),
        }
    }

    /// Set `m.params[index]` (0 to 3) of a material: every entity using it
    /// changes at once. Cheap; call it every frame if you like.
    pub fn set_material_param(
        &mut self,
        material: CustomMaterialId,
        index: usize,
        value: [f32; 4],
    ) -> EngineResult<()> {
        let (m, rev) = self.custom_materials.get_mut(&material).ok_or_else(|| {
            EngineError::InvalidValue(format!("unknown custom material {material:?}"))
        })?;
        if index >= 4 || value.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::InvalidValue(format!(
                "custom material {:?}: param index must be 0 to 3 and values finite (got {index}, {value:?})",
                m.name
            )));
        }
        if m.params[index] != value {
            m.params[index] = value;
            *rev += 1;
        }
        Ok(())
    }

    pub fn custom_material(&self, material: CustomMaterialId) -> Option<&CustomMaterial> {
        self.custom_materials.get(&material).map(|(m, _)| m)
    }

    pub fn custom_material_count(&self) -> usize {
        self.custom_materials.len()
    }
}
