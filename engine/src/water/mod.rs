//! Water: one system for lakes, seas, rivers, caves and floods.
//!
//! [`Water`] simulates water over a region of the world as a grid of columns.
//! Each column has one cell per open span between rock, so a flooded cave
//! under a lake is its own cell, linked to whatever it touches. Water moves by
//! the shallow water equations along those links:
//!
//! - It flows into anything that opens up. Dig a tunnel from a lake into a
//!   cave and call [`Water::resync_area`] (the [`crate::world::World`] does this
//!   for you when the water and the voxel terrain are both set), and the cave
//!   floods while the lake drops by exactly the volume it lost.
//! - Waves are physical: they travel at √(g·h), slow down and steepen in the
//!   shallows, run up beaches and pull back, so big water changes its extent.
//! - Edges are guaranteed: water exists only in open space, only moves
//!   through openings, never goes negative, and the total volume only changes
//!   through sources, drains, rain, [`Water::pour`] and open sea edges.
//! - Coasts: [`Water::set_sea`] opens chosen edges onto an ocean. Swell
//!   ([`Swell`]) comes in through them and runs up the beaches; waves going
//!   out pass through instead of bouncing back. The renderer draws the sea on
//!   past the region to the horizon.
//! - Flooded tunnels are pressurised, so water rises in the far shaft of a
//!   U-bend and springs up where a cave opens below a lake's level.
//!
//! [`Water::sample`] answers what gameplay needs: is this point under water,
//! how deep, where is the surface, which way is the current, can I surface
//! here or is the cave full.
//!
//! ```
//! use engine::water::{HeightMapTerrain, Water, WaterRegion};
//! use glam::{Vec2, Vec3};
//!
//! // A bowl 64 m across, filled to 2 m above its lowest point.
//! let bowl = HeightMapTerrain(|x: f32, z: f32| ((x - 32.0).powi(2) + (z - 32.0).powi(2)) * 0.004);
//! let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(64.0), 1.0, -10.0, 40.0);
//! let mut water = Water::new(region, &bowl);
//! water.flood(Vec3::new(32.0, 0.0, 32.0), 2.0);
//! water.update(1.0 / 60.0);
//! assert!(water.is_submerged(Vec3::new(32.0, 1.0, 32.0)));
//! ```

mod active;
pub(crate) mod backend;
pub(crate) mod gpu_sim;
mod look;
mod marks;
mod render_data;
mod sea;
mod sim;
mod swimmer;
mod terrain;

#[cfg(test)]
mod tests;

pub use backend::WaterBackend;
pub use look::WaterLook;
pub use sea::{Sea, SeaEdges, Swell};
pub(crate) use sim::mark_columns;
pub use sim::{SourceId, Water, WaterRegion, WaterSample, WaterSettings, WaterSource, WaterStats};
pub use swimmer::{SwimEvents, SwimInput, SwimState, Swimmer, SwimmerSettings};
pub use terrain::{scan_column, HeightMapTerrain, Span, WaterTerrain};
