//! Ready-made effects on top of the particles: fire, explosions, and the
//! dust and chunks that fly when the world is dug, shattered or hit.

use super::World;
use crate::color::Color;
use crate::error::EngineResult;
use crate::lamps::{Lamp, LampId};
use crate::particles::{EmitterId, ParticleEmitter};
use glam::Vec3;

/// A burning fire from [`World::start_fire`]: flames, smoke, embers and a
/// flickering light. Put it out with [`World::stop_fire`].
#[derive(Clone, Debug, PartialEq)]
pub struct Fire {
    emitters: Vec<EmitterId>,
    lamp: Option<LampId>,
}

/// Loose earth, for dug ground.
const EARTH: Color = Color {
    r: 0.30,
    g: 0.22,
    b: 0.15,
    a: 1.0,
};
/// Plaster and stone, for buildings.
const MASONRY: Color = Color {
    r: 0.55,
    g: 0.5,
    b: 0.44,
    a: 1.0,
};

impl World {
    /// Dust and flying chunks when things are dug, shattered or hit
    /// (on by default).
    pub fn set_destruction_effects(&mut self, on: bool) {
        self.destruction_effects = on;
    }

    pub fn destruction_effects(&self) -> bool {
        self.destruction_effects
    }

    /// Dig a smooth hole in the voxel terrain, with a puff of dust and
    /// clods of earth thrown out of it. Returns how many chunks changed
    /// (0 without voxel terrain).
    pub fn dig(&mut self, center: impl Into<Vec3>, radius: f32) -> usize {
        let center = center.into();
        let Some(terrain) = self.voxel_terrain_mut() else {
            return 0;
        };
        if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return 0;
        }
        let up = terrain.normal_at(center).normalize_or(Vec3::Y);
        let changed = terrain.dig_sphere(center, radius);
        if changed > 0 {
            self.destruction_burst(center + up * radius * 0.3, radius, EARTH, up);
        }
        changed
    }

    /// Add a smooth lump of ground to the voxel terrain. Returns how many
    /// chunks changed.
    pub fn add_ground(&mut self, center: impl Into<Vec3>, radius: f32) -> usize {
        let center = center.into();
        let Some(terrain) = self.voxel_terrain_mut() else {
            return 0;
        };
        if !center.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return 0;
        }
        let changed = terrain.add_sphere(center, radius);
        if changed > 0 && self.destruction_effects {
            let dust = ParticleEmitter::dust(center, radius * 0.8, EARTH);
            let _ = self.particles_mut().burst(dust);
        }
        changed
    }

    /// Light a fire `size` metres across at `at` (on the ground under it).
    pub fn start_fire(&mut self, at: impl Into<Vec3>, size: f32) -> EngineResult<Fire> {
        let at = at.into();
        let s = size.max(0.1);
        let emitters = self.particles_mut().start_group([
            ParticleEmitter::flames(at, s),
            ParticleEmitter::smoke(at, s),
            ParticleEmitter::embers(at, s),
        ])?;
        let lamp = match self.add_lamp(
            Lamp::fire(at + Vec3::Y * 0.5 * s)
                .intensity(5.0 * s.sqrt())
                .flicker(0.35),
        ) {
            Ok(id) => Some(id),
            Err(e) => {
                self.particles_mut().stop_all(&emitters);
                return Err(e);
            }
        };
        Ok(Fire { emitters, lamp })
    }

    /// Put a fire out: no new flames; the smoke drifts away on its own.
    pub fn stop_fire(&mut self, fire: Fire) {
        self.particles_mut().stop_all(&fire.emitters);
        if let Some(lamp) = fire.lamp {
            self.remove_lamp(lamp);
        }
    }

    /// The look of an explosion `radius` metres across: a fireball, sparks,
    /// a dark dust cloud and debris, and bodies nearby are thrown outward.
    /// Damage is up to you: pair it with [`World::dig`],
    /// [`World::hit_buildings`] or [`World::shatter`].
    pub fn explosion(&mut self, at: impl Into<Vec3>, radius: f32) -> EngineResult<()> {
        let at = at.into();
        let r = radius.clamp(0.2, 50.0);
        let soot = Color {
            r: 0.12,
            g: 0.11,
            b: 0.1,
            a: 1.0,
        };
        let p = self.particles_mut();
        p.burst(ParticleEmitter::fireball(at, r))?;
        p.burst(
            ParticleEmitter::sparks(at, Vec3::Y * 4.0 * r.sqrt(), (60.0 * r).min(600.0) as u32)
                .with_velocity_jitter(Vec3::splat(9.0 * r.sqrt())),
        )?;
        p.burst(
            ParticleEmitter::dust(at + Vec3::Y * 0.3 * r, r * 1.2, soot)
                .with_lifetime(7.0)
                .with_acceleration(Vec3::new(0.0, 0.5, 0.0)),
        )?;
        p.burst(ParticleEmitter::debris(
            at,
            EARTH,
            (20.0 * r).clamp(10.0, 300.0) as u32,
            7.0 * r.sqrt(),
        ))?;
        self.blast(at, r * 2.0, 6.0 * r.sqrt());
        Ok(())
    }

    /// Dust and chunks of `color` thrown out along `normal` from damage of
    /// size `radius`. Used by digging, shattering and building hits; call it
    /// yourself for custom destruction.
    pub fn destruction_burst(&mut self, at: Vec3, radius: f32, color: Color, normal: Vec3) {
        if !self.destruction_effects || !at.is_finite() || !radius.is_finite() {
            return;
        }
        let r = radius.clamp(0.1, 20.0);
        let normal = normal.normalize_or(Vec3::Y);
        let p = self.particles_mut();
        let _ = p.burst(ParticleEmitter::dust(at, r, color).with_velocity(normal * 0.6 * r.sqrt()));
        let speed = 3.0 + 2.0 * r.sqrt();
        let count = (14.0 * r * r).clamp(8.0, 250.0) as u32;
        let _ = p.burst(
            ParticleEmitter::debris(at, color, count, speed)
                .with_velocity(normal * speed * 0.7)
                .with_size(0.06 + 0.04 * r.min(3.0))
                .with_seed((at.x * 13.0 + at.z * 7.0) as i32 as u32),
        );
    }

    /// Dust for a broken building (see [`World::hit_building`]).
    pub(crate) fn building_dust(&mut self, at: Vec3, radius: f32) {
        self.destruction_burst(at, radius.max(0.5), MASONRY, Vec3::Y);
    }
}

/// Mean vertex colour of a mesh: the colour its dust and chunks get.
pub(crate) fn mesh_color(colors: &[glam::Vec4]) -> Color {
    if colors.is_empty() {
        return MASONRY;
    }
    let step = (colors.len() / 256).max(1);
    let (mut sum, mut n) = (glam::Vec3::ZERO, 0.0);
    for c in colors.iter().step_by(step) {
        sum += c.truncate();
        n += 1.0;
    }
    let m = (sum / n).clamp(Vec3::ZERO, Vec3::ONE);
    Color {
        r: m.x,
        g: m.y,
        b: m.z,
        a: 1.0,
    }
}
