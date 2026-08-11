//! Everyday imports for the friendly API.

pub use crate::camera::Camera;
pub use crate::color::{rgb, rgba, Color};
pub use crate::error::{EngineError, EngineResult};
pub use crate::input::{Input, Key};
pub use crate::landscape::Landscape;
pub use crate::limits::EngineLimits;
pub use crate::mesh::{Mesh, PointId, Shape};
pub use crate::model::{scatter_places, Model};
pub use crate::place::Place;
pub use crate::proc::{scatter_on_xz, Noise};
pub use crate::proc_terrain::{
    demo_terrain_rules, ClipmapConfig, HeightField, ProcTerrain,
};
pub use crate::terrain::{HeightTerrain, TerrainRules, TerrainSample, TerrainStream};
pub use crate::ui::{UiFrame, UiPanel};
pub use crate::world::{EntityId, Frame, World};
pub use crate::Engine;
pub use glam::Vec3;
