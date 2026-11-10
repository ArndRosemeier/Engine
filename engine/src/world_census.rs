//! The resource census (see [`crate::census`]).
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;
use crate::census::{GpuCensus, ResourceCensus};

impl World {
    /// Counts of what the world and the renderer hold now (see
    /// [`crate::census`]).
    pub fn resource_census(&self) -> ResourceCensus {
        ResourceCensus {
            entities: self.entity_count(),
            animated: self.animated.len(),
            lod_sets: self.lods.len(),
            textures: self.texture_count(),
            lamps: self.lamps.len(),
            decals: self.decals.entries.len(),
            custom_materials: self.custom_material_count(),
            crowd_agents: self
                .crowds
                .iter()
                .flatten()
                .map(|c| c.agents.iter().flatten().count())
                .sum(),
            bodies: self.physics.body_count(),
            sound_voices: self.audio.voice_count(),
            gpu: self.gpu_census,
        }
    }

    pub(crate) fn set_gpu_census(&mut self, gpu: GpuCensus) {
        self.gpu_census = gpu;
    }
}
