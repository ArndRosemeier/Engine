//! Everyday imports for the friendly API.

pub use crate::camera::Camera;
pub use crate::color::{rgb, Color};
pub use crate::error::{EngineError, EngineResult};
pub use crate::landscape::Landscape;
pub use crate::limits::EngineLimits;
pub use crate::mesh::{Mesh, PointId, Shape};
pub use crate::model::{scatter_places, Model};
pub use crate::place::Place;
pub use crate::proc::scatter_on_xz;
pub use crate::world::{EntityId, Frame, World};
pub use crate::Engine;
pub use glam::Vec3;
