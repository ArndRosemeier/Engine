//! Everyday imports for the friendly API.

pub use crate::anim::AnimatedModel;
pub use crate::audio::{Audio, ClipId, Play, VoiceId};
pub use crate::camera::Camera;
pub use crate::chunk_stream::{ChunkBuilder, ChunkPayload, ChunkStream};
pub use crate::collision::{
    ActorBody, ActorMove, ColliderId, ColliderLayer, ColliderShape, CollisionWorld, StaticCollider,
};
pub use crate::color::{rgb, rgba, Color};
pub use crate::contact::{ContactGrid, ContactSnapshot};
pub use crate::debris::Shatter;
pub use crate::error::{EngineError, EngineResult};
pub use crate::fly::FlyCamera;
pub use crate::input::{Input, Key, MouseButton};
pub use crate::landscape::Landscape;
pub use crate::limits::EngineLimits;
pub use crate::lod::{LodId, LodLevel};
pub use crate::mesh::{Mesh, PointId, Shape, SurfaceMaterial};
pub use crate::model::{scatter_places, Model, ModelScene};
pub use crate::parallel::build_all;
pub use crate::particles::{
    EmitterId, ParticleBlend, ParticleEmitter, ParticleForce, ParticleMode, ParticleShape,
    ParticleSilhouette, ParticleWorld, SizeOverLife,
};
pub use crate::place::{GlobalPlace, MeshInstance, Place};
pub use crate::portal::{Portal, PortalId, PortalSettings, SpaceId};
pub use crate::proc::{scatter_on_xz, Noise};
pub use crate::proc_terrain::{
    demo_terrain_rules, ClipmapConfig, Grass, HeightField, ProcTerrain, TerrainLayers,
};
pub use crate::props::{PropModel, PropPack};
pub use crate::quality::Quality;
pub use crate::ribbons::{RibbonId, RibbonProfile, RibbonStyle, RibbonWorld};
pub use crate::rocks::{rock_model, rock_model_with, RockModel, RockShape};
pub use crate::scatter::Scatter;
pub use crate::sky::{PhysicalSky, SkyWeather};
pub use crate::space::{
    ChunkCoord, ChunkId, ChunkLayer, ChunkLevel, ChunkSpan, GlobalPosition, GlobalXZ, RenderOrigin,
    RenderPosition,
};
pub use crate::surface::{SurfaceSample, SurfaceSource, WaterSurface, WATER_CLEARANCE};
pub use crate::surface_terrain::{SurfaceMeshStyle, SurfaceStream, SurfaceTerrain};
pub use crate::terrain::{HeightTerrain, TerrainRules, TerrainSample, TerrainStream};
pub use crate::texgen::{
    layer as texture_layer, natural_layer, FacadeLayer, GroundLayer, Layer, MedievalLayer,
    NaturalLayer, RoofLayer, TextureLayer,
};
pub use crate::texture::{
    generate_cave_albedo, generate_terrain_albedo, load_rgba8_png, MaterialId, TerrainAlbedo,
    TerrainMaterialDesc, TextureId,
};
pub use crate::ui::{UiFrame, UiPanel};
pub use crate::vegetation::{shrub_model, tree_model, TreeModel, TreeSpecies};
pub use crate::vfx::{Delivery, EffectHandle, EffectSpec, VfxPalette, VfxSystem, VisualKind};
pub use crate::voxel_terrain::VoxelTerrain;
pub use crate::water::{Water, WaterRegion, WaterSample, WaterSource};
pub use crate::water_mesh::{band_mesh, polygon_fill_mesh, rect_fill_mesh, ribbon_mesh};
pub use crate::world::AnimatedEntity;
pub use crate::world::{
    AmbientOcclusion, BloomSettings, EntityId, Frame, GlobalIllumination, Haze, InstanceSubmit,
    Light, ShadowSettings, Sky, ToneMap, TorchLight, World,
};
pub use crate::Engine;
pub use glam::Vec3;
