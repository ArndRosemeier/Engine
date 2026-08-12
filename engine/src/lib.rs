//! Minimal procedural 3D engine built on wgpu.
//!
//! Prefer [`prelude`] for everyday use. Power users can open [`advanced`].

pub mod advanced;
pub mod anim;
pub mod camera;
pub mod chunk_stream;
pub mod color;
pub mod contact;
pub mod error;
pub mod input;
pub mod landscape;
pub mod limits;
pub mod mesh;
pub mod model;
pub mod place;
pub mod prelude;
pub mod proc;
pub mod proc_terrain;
pub mod space;
pub mod surface;
pub mod surface_terrain;
pub mod terrain;
pub mod texture;
pub mod ui;
pub mod water_mesh;
pub mod world;

pub(crate) mod app;
pub(crate) mod marching_cubes;
pub(crate) mod render;
pub(crate) mod ui_backend;
pub(crate) mod volume;

pub use anim::{AnimatedModel, AnimationClip, Animator};
pub use camera::Camera;
pub use chunk_stream::{ChunkBuilder, ChunkPayload, ChunkStream};
pub use color::{rgb, rgba, Color};
pub use contact::ContactGrid;
pub use error::{EngineError, EngineResult};
pub use input::{Input, Key};
pub use landscape::Landscape;
pub use limits::EngineLimits;
pub use mesh::{Mesh, PointId, Shape};
pub use model::{scatter_places, Model};
pub use place::{GlobalPlace, Place};
pub use proc_terrain::{demo_terrain_rules, ClipmapConfig, HeightField, ProcTerrain};
pub use space::{
    ChunkCoord, ChunkId, ChunkLayer, ChunkSpan, GlobalPosition, GlobalXZ, RenderOrigin,
    RenderPosition,
};
pub use surface::{SurfaceSample, SurfaceSource, WaterSurface, WATER_CLEARANCE};
pub use surface_terrain::{SurfaceMeshStyle, SurfaceStream, SurfaceTerrain};
pub use terrain::{HeightTerrain, TerrainRules, TerrainSample, TerrainStream};
pub use texture::{
    generate_terrain_albedo, load_rgba8_png, MaterialId, TerrainAlbedo, TerrainMaterialDesc,
    TextureId,
};
pub use ui::{egui, UiFrame, UiPanel};
pub use water_mesh::{band_mesh, polygon_fill_mesh, rect_fill_mesh, ribbon_mesh};
pub use world::{AnimatedEntity, EntityId, Frame, Light, World};

/// Entry point matching the planned `Engine::run` shape.
pub struct Engine;

impl Engine {
    pub fn run(title: impl Into<String>, update: impl FnMut(&mut World, &Frame) + 'static) {
        app::run(title, update);
    }

    pub fn run_with(
        title: impl Into<String>,
        limits: EngineLimits,
        update: impl FnMut(&mut World, &Frame) + 'static,
    ) {
        app::run_with(title, limits, update);
    }
}

#[cfg(test)]
mod tests;
