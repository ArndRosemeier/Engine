//! A procedural 3D engine built on wgpu.
//!
//! You write one closure that runs every frame; [`Engine::run`] owns the
//! window, the GPU and the loop. Import [`prelude`] for everyday work; power
//! users can open [`advanced`].
//!
//! ```no_run
//! use engine::prelude::*;
//!
//! fn main() -> EngineResult<()> {
//!     let mut camera = FlyCamera::new((0.0, 3.0, -12.0), 0.0, -8.0);
//!     Engine::run("hello", move |world, frame| {
//!         if frame.first {
//!             world.spawn(Shape::box_at((0.0, -0.05, 0.0), (60.0, 0.1, 60.0), rgb(90, 120, 70))?);
//!             world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
//!             let places = Scatter::new(1).rect(-25.0, 5.0, 25.0, 30.0).spacing(8.0).places()?;
//!             world.spawn_lod_at(tree_model(TreeSpecies::Birch, 0)?.lod_levels(), &places)?;
//!         }
//!         camera.update(world, frame);
//!         Ok(())
//!     })
//! }
//! ```
//!
//! Where to look next:
//!
//! - [`world::World`]: the scene and every setting (camera, sky, lights, LOD,
//!   quality, terrain, portals).
//! - [`mesh::Mesh`] and [`mesh::Shape`]: build geometry; [`model::Model`] loads glTF.
//! - Generators: [`vegetation`], [`rocks`], [`texgen`], [`scatter`],
//!   [`proc_terrain`], [`landscape`], [`voxel_terrain`].
//! - [`quality`]: presets and the frame-rate auto-tuner.
//! - [`fly::FlyCamera`], [`input::Input`], [`ui`], [`audio`], [`collision`].
//!
//! Coordinates are metres with Y up; yaw is in degrees, 0 facing +Z.

pub mod advanced;
pub mod anim;
pub mod audio;
pub mod camera;
pub mod chunk_stream;
pub mod collision;
pub mod color;
pub mod contact;
pub mod debris;
pub mod error;
pub mod fly;
pub mod fracture;
mod frame_clock;
pub mod gpu_field;
pub mod input;
pub mod landscape;
pub mod limits;
pub mod lod;
pub mod mesh;
pub mod model;
pub mod parallel;
pub mod particles;
pub mod place;
pub mod portal;
pub mod prelude;
pub mod proc;
pub mod proc_terrain;
pub mod props;
pub mod quality;
pub mod ribbons;
pub mod rocks;
pub mod scatter;
pub mod sky;
pub mod space;
pub mod surface;
pub mod surface_terrain;
pub mod terrain;
pub mod texgen;
pub mod texture;

pub mod tree;
pub mod ui;
pub mod vegetation;
pub mod vfx;
pub mod voxel_terrain;
pub mod water;
pub mod water_mesh;
pub mod world;

pub(crate) mod app;
pub(crate) mod marching_cubes;
pub(crate) mod render;
pub(crate) mod ui_backend;
pub(crate) mod volume;

#[cfg(test)]
mod tint_proof;

pub use anim::{
    AnimatedModel, AnimationAction, AnimationClip, AnimationProfile, Animator, Locomotion,
};
pub use audio::{Audio, ClipId, Play, VoiceId};
pub use camera::Camera;
pub use chunk_stream::{ChunkBuilder, ChunkPayload, ChunkStream};
pub use collision::{
    ActorBody, ActorMove, ColliderId, ColliderLayer, ColliderShape, CollisionWorld, StaticCollider,
};
pub use color::{rgb, rgba, Color};
pub use contact::{ContactGrid, ContactSnapshot};
pub use debris::Shatter;
pub use error::{EngineError, EngineResult};
pub use gpu_field::{
    demo_sphere_density, extract_mesh_cpu, gpu_tests_enabled, CustomFieldKernel, DensityMirror,
    DensityReferenceFn, FieldBounds, FieldChunkKey, FieldChunkMesh, FieldGpuContext, FieldGrid,
    FieldKernel, FieldMeshBounds, FieldParamsBlob, GpuField, PaintedField, ParamsMirror,
    FIELD_CHUNK_SIZE, MAX_FIELD_UNIFORM_BYTES,
};
pub use input::{Input, Key, MouseButton};
pub use landscape::Landscape;
pub use limits::EngineLimits;
pub use mesh::{AlbedoMap, BuiltMesh, MaterialMaps, Mesh, PointId, Shape, SurfaceMaterial};
pub use model::{scatter_places, Model, ModelScene};
pub use particles::{
    EmitterId, ParticleBlend, ParticleEmitter, ParticleForce, ParticleMode, ParticleShape,
    ParticleSilhouette, ParticleWorld, SizeOverLife,
};
pub use place::{GlobalPlace, MeshInstance, Place};
pub use portal::{Portal, PortalId, PortalSettings, SpaceId};
pub use proc_terrain::{
    demo_terrain_rules, ClipmapConfig, Grass, HeightField, ProcTerrain, TerrainLayers,
};
pub use props::{PropModel, PropPack};
pub use ribbons::{RibbonId, RibbonProfile, RibbonStyle, RibbonWorld};
pub use space::{
    ChunkCoord, ChunkId, ChunkLayer, ChunkLevel, ChunkSpan, GlobalPosition, GlobalXZ, RenderOrigin,
    RenderPosition,
};
pub use surface::{SurfaceSample, SurfaceSource, WaterSurface, WATER_CLEARANCE};
pub use surface_terrain::{SurfaceMeshStyle, SurfaceStream, SurfaceTerrain};
pub use terrain::{HeightTerrain, TerrainRules, TerrainSample, TerrainStream};
pub use texture::{
    generate_cave_albedo, generate_terrain_albedo, load_rgba8_png, MaterialId, TerrainAlbedo,
    TerrainMaterialDesc, TextureId,
};
pub use tree::{BranchNode, LeafKind, TreeAsset, TreeKind, TreeLod, TreeProfile, TreeSettings};
pub use ui::{egui, UiFrame, UiPanel};
pub use vfx::{Delivery, EffectHandle, EffectSpec, VfxPalette, VfxSystem, VisualKind};
pub use voxel_terrain::{VoxelTerrain, VOXEL_MAX_LOD};

pub use water_mesh::{band_mesh, polygon_fill_mesh, rect_fill_mesh, ribbon_mesh};
pub use world::{
    AmbientOcclusion, AnimatedEntity, BloomSettings, EntityId, Frame, GlobalIllumination, Haze,
    HitchSpan, InstanceSubmit, Light, ShadowSettings, Sky, ToneMap, TorchLight, World,
};

/// Entry point matching the planned `Engine::run` shape.
pub struct Engine;

impl Engine {
    pub fn run(
        title: impl Into<String>,
        update: impl FnMut(&mut World, &Frame) -> EngineResult<()> + 'static,
    ) -> EngineResult<()> {
        app::run(title, update)
    }

    pub fn run_with(
        title: impl Into<String>,
        limits: EngineLimits,
        update: impl FnMut(&mut World, &Frame) -> EngineResult<()> + 'static,
    ) -> EngineResult<()> {
        app::run_with(title, limits, update)
    }
}

#[cfg(test)]
mod probe_teleport;
#[cfg(test)]
mod tests;
