//! A count of everything the world and the renderer hold, to catch
//! resources that pile up over a long session (a frame rate that falls the
//! longer one plays is almost always something that is made and never
//! freed).
//!
//! [`crate::world::World::resource_census`] gives the counts; the GPU side is filled in
//! by the renderer about twice a second (zero before the first frame and in
//! headless worlds). Log it now and then and compare: in a game that
//! streams its world, every number should level off, not keep climbing.

use std::fmt;

/// What the renderer holds on the GPU. See [`ResourceCensus`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuCensus {
    /// Frame the renderer counted at.
    pub frame: u64,
    /// Meshes with their own GPU copy (single entities).
    pub meshes: usize,
    /// Instanced and LOD batches.
    pub batches: usize,
    /// Uploaded textures.
    pub textures: usize,
    /// Mesh materials (a texture with its bind group).
    pub materials: usize,
    /// Materials in the texture-array table (beyond its 4096 slots meshes
    /// fall back to their own bind group and draw one by one).
    pub table_slots: usize,
    /// Animated (skinned) entities and their uploaded models.
    pub skinned: usize,
    pub skinned_models: usize,
    /// Character maps (shared by content).
    pub character_maps: usize,
    /// Custom materials compiled.
    pub custom_materials: usize,
    /// Pages of the shared vertex/index buffers.
    pub geometry_pages: usize,
    /// Video memory, MiB: meshes (pages, batches), textures, characters
    /// (buffers and maps), and what the allocator holds in all (0 where
    /// the backend does not say).
    pub mesh_mib: u64,
    pub texture_mib: u64,
    pub character_mib: u64,
    pub allocated_mib: u64,
}

/// See the module docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceCensus {
    pub entities: usize,
    pub animated: usize,
    pub lod_sets: usize,
    /// Textures (the game's own and those made from mesh albedo maps).
    pub textures: usize,
    pub lamps: usize,
    pub decals: usize,
    pub custom_materials: usize,
    pub crowd_agents: usize,
    pub bodies: usize,
    pub sound_voices: usize,
    pub gpu: GpuCensus,
}

impl ResourceCensus {
    /// The named counts, for logs and checks.
    pub fn fields(&self) -> Vec<(&'static str, u64)> {
        let g = &self.gpu;
        vec![
            ("entities", self.entities as u64),
            ("animated", self.animated as u64),
            ("lod_sets", self.lod_sets as u64),
            ("textures", self.textures as u64),
            ("lamps", self.lamps as u64),
            ("decals", self.decals as u64),
            ("custom_materials", self.custom_materials as u64),
            ("crowd_agents", self.crowd_agents as u64),
            ("bodies", self.bodies as u64),
            ("sound_voices", self.sound_voices as u64),
            ("gpu_meshes", g.meshes as u64),
            ("gpu_batches", g.batches as u64),
            ("gpu_textures", g.textures as u64),
            ("gpu_materials", g.materials as u64),
            ("gpu_table_slots", g.table_slots as u64),
            ("gpu_skinned", g.skinned as u64),
            ("gpu_skinned_models", g.skinned_models as u64),
            ("gpu_character_maps", g.character_maps as u64),
            ("gpu_custom_materials", g.custom_materials as u64),
            ("gpu_geometry_pages", g.geometry_pages as u64),
            ("gpu_mesh_mib", g.mesh_mib),
            ("gpu_texture_mib", g.texture_mib),
            ("gpu_character_mib", g.character_mib),
            ("gpu_allocated_mib", g.allocated_mib),
        ]
    }
}

impl fmt::Display for ResourceCensus {
    /// One line: `entities=7454 animated=12 ... gpu_allocated_mib=2210`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (k, v)) in self.fields().into_iter().enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{k}={v}")?;
        }
        Ok(())
    }
}
