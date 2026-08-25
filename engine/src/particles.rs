use crate::{Color, EngineError, EngineResult};
use glam::Vec3;

pub(crate) const MAX_PARTICLE_EMITTERS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmitterId {
    slot: u32,
    generation: u32,
}
impl EmitterId {
    pub(crate) fn slot(self) -> u32 {
        self.slot
    }
    pub(crate) fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMode {
    Continuous,
    Burst,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleShape {
    Point,
    Sphere,
    Cone,
    Ring,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleSilhouette {
    SoftOrb,
    Flame,
    SparkStreak,
    SmokeCloud,
    Shard,
    RuneMote,
    Heart,
    Bubble,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleBlend {
    Additive,
    Alpha,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeOverLife {
    Constant,
    FadeInOut,
    Shrink,
    GrowThenFade,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParticleForce {
    None,
    Vortex { axis: Vec3, strength: f32 },
    Radial { center: Vec3, strength: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleEmitter {
    position: Vec3,
    velocity: Vec3,
    spread: Vec3,
    color: Color,
    secondary_color: Color,
    size: f32,
    lifetime_s: f32,
    rate: f32,
    seed: u32,
    mode: ParticleMode,
    shape: ParticleShape,
    silhouette: ParticleSilhouette,
    blend: ParticleBlend,
    size_over_life: SizeOverLife,
    turbulence: f32,
    drag: f32,
    burst_count: u32,
    acceleration: Vec3,
    velocity_stretch: f32,
    force: ParticleForce,
    emissive_intensity: f32,
}
impl ParticleEmitter {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            spread: Vec3::splat(1.0),
            color: Color::WHITE,
            secondary_color: Color::WHITE,
            size: 0.1,
            lifetime_s: 1.0,
            rate: 20.0,
            seed: 0,
            mode: ParticleMode::Continuous,
            shape: ParticleShape::Point,
            silhouette: ParticleSilhouette::SoftOrb,
            blend: ParticleBlend::Additive,
            size_over_life: SizeOverLife::FadeInOut,
            turbulence: 1.0,
            drag: 0.0,
            burst_count: 64,
            acceleration: Vec3::ZERO,
            velocity_stretch: 0.0,
            force: ParticleForce::None,
            emissive_intensity: 1.0,
        }
    }
    pub fn with_velocity(mut self, v: Vec3) -> Self {
        self.velocity = v;
        self
    }
    pub fn with_spread(mut self, v: Vec3) -> Self {
        self.spread = v;
        self
    }
    pub fn with_color(mut self, v: Color) -> Self {
        self.color = v;
        self
    }
    pub fn with_secondary_color(mut self, v: Color) -> Self {
        self.secondary_color = v;
        self
    }
    pub fn with_size(mut self, v: f32) -> Self {
        self.size = v;
        self
    }
    pub fn with_lifetime(mut self, v: f32) -> Self {
        self.lifetime_s = v;
        self
    }
    pub fn with_rate(mut self, v: f32) -> Self {
        self.rate = v;
        self
    }
    pub fn with_seed(mut self, v: u32) -> Self {
        self.seed = v;
        self
    }
    pub fn with_mode(mut self, v: ParticleMode) -> Self {
        self.mode = v;
        self
    }
    pub fn with_shape(mut self, v: ParticleShape) -> Self {
        self.shape = v;
        self
    }
    pub fn with_silhouette(mut self, v: ParticleSilhouette) -> Self {
        self.silhouette = v;
        self
    }
    pub fn with_blend(mut self, v: ParticleBlend) -> Self {
        self.blend = v;
        self
    }
    pub fn with_size_over_life(mut self, v: SizeOverLife) -> Self {
        self.size_over_life = v;
        self
    }
    pub fn with_turbulence(mut self, v: f32) -> Self {
        self.turbulence = v;
        self
    }
    pub fn with_drag(mut self, v: f32) -> Self {
        self.drag = v;
        self
    }
    pub fn with_burst_count(mut self, v: u32) -> Self {
        self.burst_count = v;
        self
    }
    pub fn with_acceleration(mut self, v: Vec3) -> Self {
        self.acceleration = v;
        self
    }
    pub fn with_velocity_stretch(mut self, v: f32) -> Self {
        self.velocity_stretch = v;
        self
    }
    pub fn with_emissive_intensity(mut self, intensity: f32) -> Self {
        self.emissive_intensity = intensity;
        self
    }
    pub fn with_force(mut self, v: ParticleForce) -> Self {
        self.force = v;
        self
    }
    pub fn validate(self) -> EngineResult<Self> {
        let colors_valid = [self.color, self.secondary_color].into_iter().all(|c| {
            [c.r, c.g, c.b, c.a]
                .into_iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(&x))
        });
        let force_valid = match self.force {
            ParticleForce::None => true,
            ParticleForce::Vortex { axis, strength } => {
                axis.is_finite() && axis.length_squared() > 0.0 && strength.is_finite()
            }
            ParticleForce::Radial { center, strength } => {
                center.is_finite() && strength.is_finite()
            }
        };
        if !self.position.is_finite()
            || !self.velocity.is_finite()
            || !self.spread.is_finite()
            || !self.acceleration.is_finite()
            || !colors_valid
            || !force_valid
            || !self.size.is_finite()
            || self.size <= 0.0
            || !self.lifetime_s.is_finite()
            || self.lifetime_s <= 0.0
            || !self.rate.is_finite()
            || self.rate < 0.0
            || !self.turbulence.is_finite()
            || self.turbulence < 0.0
            || !self.drag.is_finite()
            || !(0.0..=1.0).contains(&self.drag)
            || !self.velocity_stretch.is_finite()
            || self.velocity_stretch < 0.0
            || !self.emissive_intensity.is_finite()
            || self.emissive_intensity < 0.0
            || self.burst_count == 0
        {
            return Err(EngineError::InvalidValue(
                "invalid particle emitter profile".into(),
            ));
        }
        Ok(self)
    }
    pub(crate) fn position(self) -> Vec3 {
        self.position
    }
    pub(crate) fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }
    pub(crate) fn velocity(self) -> Vec3 {
        self.velocity
    }
    pub(crate) fn spread(self) -> Vec3 {
        self.spread
    }
    pub(crate) fn color(self) -> Color {
        self.color
    }
    pub(crate) fn secondary_color(self) -> Color {
        self.secondary_color
    }
    pub(crate) fn size(self) -> f32 {
        self.size
    }
    pub(crate) fn lifetime_s(self) -> f32 {
        self.lifetime_s
    }
    pub(crate) fn rate(self) -> f32 {
        self.rate
    }
    pub(crate) fn seed(self) -> u32 {
        self.seed
    }
    pub(crate) fn mode(self) -> ParticleMode {
        self.mode
    }
    pub(crate) fn shape(self) -> ParticleShape {
        self.shape
    }
    pub(crate) fn silhouette(self) -> ParticleSilhouette {
        self.silhouette
    }
    pub(crate) fn blend(self) -> ParticleBlend {
        self.blend
    }
    pub(crate) fn size_over_life(self) -> SizeOverLife {
        self.size_over_life
    }
    pub(crate) fn turbulence(self) -> f32 {
        self.turbulence
    }
    pub(crate) fn drag(self) -> f32 {
        self.drag
    }
    pub(crate) fn burst_count(self) -> u32 {
        self.burst_count
    }
    pub(crate) fn acceleration(self) -> Vec3 {
        self.acceleration
    }
    pub(crate) fn velocity_stretch(self) -> f32 {
        self.velocity_stretch
    }
    pub fn emissive_intensity(self) -> f32 {
        self.emissive_intensity
    }
    pub(crate) fn force(self) -> ParticleForce {
        self.force
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ParticleCommand {
    Start(EmitterId, ParticleEmitter),
    UpdatePosition(EmitterId, Vec3),
    Stop(EmitterId),
    Clear,
}
#[derive(Clone, Copy, Debug)]
struct EmitterSlot {
    generation: u32,
    allocated: bool,
}
impl Default for EmitterSlot {
    fn default() -> Self {
        Self {
            generation: 1,
            allocated: false,
        }
    }
}

#[derive(Debug)]
pub struct ParticleWorld {
    slots: [EmitterSlot; MAX_PARTICLE_EMITTERS],
    commands: Vec<ParticleCommand>,
    simulation_dt_s: f32,
    simulation_time_s: f32,
    time_scale: f32,
}
impl Default for ParticleWorld {
    fn default() -> Self {
        Self {
            slots: [EmitterSlot::default(); MAX_PARTICLE_EMITTERS],
            commands: Vec::new(),
            simulation_dt_s: 0.0,
            simulation_time_s: 0.0,
            time_scale: 1.0,
        }
    }
}
impl ParticleWorld {
    pub(crate) fn drain(&mut self) -> (Vec<ParticleCommand>, f32, f32) {
        let dt = std::mem::take(&mut self.simulation_dt_s);
        (
            std::mem::take(&mut self.commands),
            dt,
            self.simulation_time_s,
        )
    }
    pub fn set_time_scale(&mut self, scale: f32) -> EngineResult<()> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(EngineError::InvalidValue(
                "particle time scale must be finite and >= 0".into(),
            ));
        }
        self.time_scale = scale;
        Ok(())
    }
    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }
    pub fn advance(&mut self, dt_s: f32) -> EngineResult<()> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(EngineError::InvalidValue(
                "particle dt must be finite and >= 0".into(),
            ));
        }
        let scaled = dt_s * self.time_scale;
        self.simulation_dt_s += scaled;
        self.simulation_time_s += scaled;
        if !self.simulation_dt_s.is_finite() || !self.simulation_time_s.is_finite() {
            return Err(EngineError::InvalidValue(
                "particle simulation time overflow".into(),
            ));
        }
        Ok(())
    }
    pub fn start(&mut self, emitter: ParticleEmitter) -> EngineResult<EmitterId> {
        let emitter = emitter.validate()?;
        let (i, slot) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, s)| !s.allocated)
            .ok_or_else(|| {
                EngineError::ResourceLimit(format!(
                    "particle emitter capacity ({MAX_PARTICLE_EMITTERS}) exhausted"
                ))
            })?;
        slot.allocated = true;
        let id = EmitterId {
            slot: u32::try_from(i).expect("emitter capacity exceeds u32"),
            generation: slot.generation,
        };
        self.commands.push(ParticleCommand::Start(id, emitter));
        Ok(id)
    }
    pub fn set_emitter_position(&mut self, id: EmitterId, position: Vec3) -> EngineResult<()> {
        if !position.is_finite() {
            return Err(EngineError::InvalidValue(
                "particle emitter position must be finite".into(),
            ));
        }
        let slot = self.slots.get(id.slot as usize).ok_or_else(|| {
            EngineError::InvalidValue("particle emitter handle has an invalid slot".into())
        })?;
        if !slot.allocated || slot.generation != id.generation {
            return Err(EngineError::InvalidValue(
                "stale particle emitter handle".into(),
            ));
        }
        self.commands
            .push(ParticleCommand::UpdatePosition(id, position));
        Ok(())
    }
    pub fn stop(&mut self, id: EmitterId) -> EngineResult<()> {
        let slot = self.slots.get_mut(id.slot as usize).ok_or_else(|| {
            EngineError::InvalidValue("particle emitter handle has an invalid slot".into())
        })?;
        if !slot.allocated || slot.generation != id.generation {
            return Err(EngineError::InvalidValue(
                "stale particle emitter handle".into(),
            ));
        }
        slot.allocated = false;
        slot.generation = next_generation(slot.generation);
        self.commands.push(ParticleCommand::Stop(id));
        Ok(())
    }
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            if slot.allocated {
                slot.allocated = false;
                slot.generation = next_generation(slot.generation);
            }
        }
        self.commands.push(ParticleCommand::Clear)
    }
}
fn next_generation(g: u32) -> u32 {
    g.checked_add(1)
        .expect("particle emitter generation exhausted")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_profiles() {
        assert!(ParticleEmitter::new(Vec3::ZERO)
            .with_velocity_stretch(-1.0)
            .validate()
            .is_err());
        assert!(ParticleEmitter::new(Vec3::ZERO)
            .with_force(ParticleForce::Vortex {
                axis: Vec3::ZERO,
                strength: 1.0
            })
            .validate()
            .is_err());
        assert!(ParticleEmitter::new(Vec3::ZERO)
            .with_silhouette(ParticleSilhouette::Flame)
            .with_acceleration(Vec3::Y)
            .validate()
            .is_ok());
    }
    #[test]
    fn allocation_and_stale_handles_are_loud() {
        let mut w = ParticleWorld::default();
        let first = w.start(ParticleEmitter::new(Vec3::ZERO)).unwrap();
        for _ in 1..MAX_PARTICLE_EMITTERS {
            w.start(ParticleEmitter::new(Vec3::ZERO)).unwrap();
        }
        assert!(w.start(ParticleEmitter::new(Vec3::ZERO)).is_err());
        w.stop(first).unwrap();
        let replacement = w.start(ParticleEmitter::new(Vec3::ZERO)).unwrap();
        assert_eq!(replacement.slot(), first.slot());
        assert_ne!(replacement.generation(), first.generation());
        assert!(w.stop(first).is_err());
    }
    #[test]
    fn emitter_position_updates_are_generation_safe() {
        let mut world = ParticleWorld::default();
        let id = world.start(ParticleEmitter::new(Vec3::ZERO)).unwrap();
        world
            .set_emitter_position(id, Vec3::new(1.0, 2.0, 3.0))
            .unwrap();
        assert!(world
            .set_emitter_position(id, Vec3::splat(f32::NAN))
            .is_err());
        world.stop(id).unwrap();
        assert!(world.set_emitter_position(id, Vec3::ZERO).is_err());
        let (commands, _, _) = world.drain();
        assert!(
            matches!(commands[1], ParticleCommand::UpdatePosition(got, p) if got == id && p == Vec3::new(1.0, 2.0, 3.0))
        );
    }
    #[test]
    fn explicit_time_scale_pauses_simulation() {
        let mut w = ParticleWorld::default();
        w.advance(0.5).unwrap();
        w.set_time_scale(0.0).unwrap();
        w.advance(4.0).unwrap();
        let (_, dt, time) = w.drain();
        assert_eq!(dt, 0.5);
        assert_eq!(time, 0.5);
    }
}
