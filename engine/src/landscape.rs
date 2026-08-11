//! Friendly procedural landscape recipe — hides volume / extract details.

use crate::color::Color;
use crate::error::EngineResult;
use crate::limits::EngineLimits;
use crate::mesh::Mesh;
use crate::proc::{carve_tunnel_x, terrain, TerrainRules};
use crate::volume::Volume;
use glam::Vec3;

/// Builder for a hills-and-caves landscape mesh.
#[derive(Clone, Debug)]
pub struct Landscape {
    seed: u32,
    width: f32,
    depth: f32,
    height: f32,
    voxel_size: f32,
    caves: bool,
    tunnel: bool,
    color: Color,
    limits: EngineLimits,
}

impl Landscape {
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            width: 48.0,
            depth: 48.0,
            height: 20.0,
            voxel_size: 0.5,
            caves: true,
            tunnel: true,
            color: Color::rgb(122, 148, 92),
            limits: EngineLimits::default(),
        }
    }

    pub fn area(mut self, width: f32, depth: f32) -> Self {
        self.width = width;
        self.depth = depth;
        self
    }

    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    pub fn voxel_size(mut self, size: f32) -> Self {
        self.voxel_size = size;
        self
    }

    pub fn caves(mut self, enabled: bool) -> Self {
        self.caves = enabled;
        self
    }

    pub fn tunnel(mut self, enabled: bool) -> Self {
        self.tunnel = enabled;
        self
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn limits(mut self, limits: EngineLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Paint + extract into a spawnable [`Mesh`].
    pub fn build(self) -> EngineResult<Mesh> {
        let mut volume = Volume::try_new(self.voxel_size)?;
        let half_w = self.width * 0.5;
        let half_d = self.depth * 0.5;
        let min = Vec3::new(-half_w, 0.0, -half_d);
        let max = Vec3::new(half_w, self.height, half_d);

        let mut rules = TerrainRules {
            seed: self.seed,
            solid_color: self.color.to_vec3(),
            ..TerrainRules::default()
        };
        if !self.caves {
            rules.cave_threshold = 2.0; // unreachable
        }

        terrain(&mut volume, min, max, &rules, &self.limits)?;

        if self.tunnel {
            carve_tunnel_x(
                &mut volume,
                Vec3::new(-half_w * 0.7, 5.0, 2.0),
                self.width * 0.7,
                2.8,
            );
        }

        Ok(volume.extract_mesh(self.color))
    }
}
