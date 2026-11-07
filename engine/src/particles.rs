use crate::{Color, EngineError, EngineResult};
use glam::Vec3;

/// Emitters that can run at once (fire-and-forget bursts from
/// [`ParticleWorld::burst`] don't count: they free themselves).
pub const MAX_PARTICLE_EMITTERS: usize = 1024;
/// Particles alive at once, across all emitters.
pub const MAX_PARTICLES: u32 = 1 << 18;

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
    /// A small solid piece (stone, wood, glass) that tumbles, bounces off the
    /// ground and comes to rest: debris from destruction. Lit like the scene.
    Chunk,
    /// A soft, noisy puff: smoke, dust, steam, mist. Lit by sun and sky.
    Puff,
    /// A soft tongue of flame (use additive blend).
    Fire,
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
    /// Grows to 2.5x its size over its life (smoke and dust spreading).
    Expand,
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
    velocity_jitter: Option<Vec3>,
    size_variation: f32,
    collision: Option<ParticleCollision>,
    spin: f32,
    lighting: Option<f32>,
}

/// How particles hit the ground and other scene surfaces (anything drawn
/// on screen). Off-screen particles fly through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleCollision {
    /// Bounciness: 0 sticks, 1 bounces back at full speed.
    pub bounce: f32,
    /// Speed lost along the surface on each hit, 0..1.
    pub friction: f32,
}
impl Default for ParticleCollision {
    fn default() -> Self {
        Self {
            bounce: 0.3,
            friction: 0.4,
        }
    }
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
            velocity_jitter: None,
            size_variation: 0.0,
            collision: None,
            spin: 0.0,
            lighting: None,
        }
    }
    /// Random extra start speed per axis (m/s), on top of the velocity.
    /// Without it the jitter is a quarter of the spread.
    pub fn with_velocity_jitter(mut self, v: Vec3) -> Self {
        self.velocity_jitter = Some(v);
        self
    }
    /// Random size per particle: 0 all the same, 0.5 from half to 1.5x.
    pub fn with_size_variation(mut self, v: f32) -> Self {
        self.size_variation = v;
        self
    }
    /// Let particles bounce off scene surfaces and settle on the ground.
    pub fn with_collision(mut self, v: ParticleCollision) -> Self {
        self.collision = Some(v);
        self
    }
    /// Tumbling speed of chunks and rotating sprites, radians per second.
    pub fn with_spin(mut self, v: f32) -> Self {
        self.spin = v;
        self
    }
    /// How much sun and sky light the particles: 0 glows by itself, 1 fully
    /// lit. Defaults to 1 for alpha-blended and 0 for additive particles.
    pub fn with_lighting(mut self, v: f32) -> Self {
        self.lighting = Some(v);
        self
    }
    /// Moves the emitter before it starts (handy for presets).
    pub fn at(mut self, position: Vec3) -> Self {
        self.position = position;
        self
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
            || !self.size_variation.is_finite()
            || !(0.0..=1.0).contains(&self.size_variation)
            || !self.spin.is_finite()
            || self
                .velocity_jitter
                .is_some_and(|v| !v.is_finite() || v.min_element() < 0.0)
            || self.lighting.is_some_and(|l| !(0.0..=1.0).contains(&l))
            || self.collision.is_some_and(|c| {
                !(0.0..=1.0).contains(&c.bounce) || !(0.0..=1.0).contains(&c.friction)
            })
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
    pub(crate) fn velocity_jitter(self) -> Vec3 {
        self.velocity_jitter.unwrap_or(self.spread * 0.25)
    }
    pub(crate) fn size_variation(self) -> f32 {
        self.size_variation
    }
    pub(crate) fn collision(self) -> Option<ParticleCollision> {
        self.collision
    }
    pub(crate) fn spin(self) -> f32 {
        self.spin
    }
    pub(crate) fn lighting(self) -> f32 {
        self.lighting.unwrap_or(match self.blend {
            ParticleBlend::Alpha => 1.0,
            ParticleBlend::Additive => 0.0,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ParticleCommand {
    Start(EmitterId, ParticleEmitter),
    UpdatePosition(EmitterId, Vec3),
    Stop(EmitterId),
    /// A burst that needs no handle: spawned once, then forgotten.
    Burst(ParticleEmitter),
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
    /// Set when the game advanced the effects itself this frame.
    advanced: bool,
}
impl Default for ParticleWorld {
    fn default() -> Self {
        Self {
            slots: [EmitterSlot::default(); MAX_PARTICLE_EMITTERS],
            commands: Vec::new(),
            simulation_dt_s: 0.0,
            simulation_time_s: 0.0,
            time_scale: 1.0,
            advanced: false,
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
    /// Called by the engine once a frame: runs the particles by the frame
    /// time unless the game already called [`Self::advance`] itself (for
    /// slow motion or fixed steps).
    pub(crate) fn auto_advance(&mut self, dt_s: f32) {
        if !std::mem::take(&mut self.advanced) && dt_s.is_finite() && dt_s >= 0.0 {
            let _ = self.advance(dt_s);
            self.advanced = false;
        }
    }
    pub fn advance(&mut self, dt_s: f32) -> EngineResult<()> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(EngineError::InvalidValue(
                "particle dt must be finite and >= 0".into(),
            ));
        }
        self.advanced = true;
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
    /// Fire a one-off burst of `emitter.burst_count` particles (whatever the
    /// emitter's mode). Needs no handle and frees itself, so bursts don't
    /// use up emitter slots: use it for impacts, explosions and debris.
    pub fn burst(&mut self, emitter: ParticleEmitter) -> EngineResult<()> {
        let emitter = emitter.with_mode(ParticleMode::Burst).validate()?;
        self.commands.push(ParticleCommand::Burst(emitter));
        Ok(())
    }
    /// Start several emitters together (a preset like fire is flames, smoke
    /// and embers). Stop them with [`Self::stop_all`].
    pub fn start_group(
        &mut self,
        emitters: impl IntoIterator<Item = ParticleEmitter>,
    ) -> EngineResult<Vec<EmitterId>> {
        let mut ids = Vec::new();
        for e in emitters {
            match self.start(e) {
                Ok(id) => ids.push(id),
                Err(err) => {
                    self.stop_all(&ids);
                    return Err(err);
                }
            }
        }
        Ok(ids)
    }
    /// Stop every emitter in `ids` that is still running.
    pub fn stop_all(&mut self, ids: &[EmitterId]) {
        for id in ids {
            let _ = self.stop(*id);
        }
    }
    /// Emitters running now.
    pub fn running(&self) -> usize {
        self.slots.iter().filter(|s| s.allocated).count()
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

/// A linear colour (values 0..1).
const fn col(r: f32, g: f32, b: f32, a: f32) -> Color {
    Color { r, g, b, a }
}

const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Ready-made emitters. Each is an ordinary [`ParticleEmitter`], so any
/// setting can still be changed with the `with_*` methods.
impl ParticleEmitter {
    /// Flames of a fire about `size` metres across (keeps burning).
    pub fn flames(at: Vec3, size: f32) -> Self {
        let s = size.max(0.05);
        Self::new(at)
            .with_rate(70.0 * s.max(0.3))
            .with_silhouette(ParticleSilhouette::Fire)
            .with_blend(ParticleBlend::Additive)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::new(0.32, 0.08, 0.32) * s)
            .with_velocity(Vec3::new(0.0, 0.9 * s.sqrt(), 0.0))
            .with_velocity_jitter(Vec3::new(0.2, 0.35, 0.2) * s.sqrt())
            .with_acceleration(Vec3::new(0.0, 1.6 * s.sqrt(), 0.0))
            .with_lifetime(0.75)
            .with_size(0.42 * s)
            .with_size_variation(0.45)
            .with_size_over_life(SizeOverLife::Shrink)
            .with_color(col(1.0, 0.7, 0.32, 0.65))
            .with_secondary_color(col(0.85, 0.15, 0.03, 0.35))
            .with_emissive_intensity(2.6)
            .with_turbulence(1.4)
            .with_drag(0.3)
            .with_spin(1.5)
    }

    /// Smoke rising from a fire `size` metres across.
    pub fn smoke(at: Vec3, size: f32) -> Self {
        let s = size.max(0.05);
        Self::new(at + Vec3::Y * 0.6 * s)
            .with_rate(10.0 * s.max(0.4))
            .with_silhouette(ParticleSilhouette::Puff)
            .with_blend(ParticleBlend::Alpha)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::splat(0.25 * s))
            .with_velocity(Vec3::new(0.0, 1.3 * s.sqrt(), 0.0))
            .with_velocity_jitter(Vec3::new(0.25, 0.2, 0.25) * s.sqrt())
            .with_acceleration(Vec3::new(0.25, 0.35, 0.1))
            .with_lifetime(6.0)
            .with_size(0.55 * s)
            .with_size_variation(0.4)
            .with_size_over_life(SizeOverLife::Expand)
            .with_color(col(0.16, 0.15, 0.14, 0.55))
            .with_secondary_color(col(0.42, 0.42, 0.43, 0.25))
            .with_turbulence(0.5)
            .with_drag(0.25)
            .with_spin(0.4)
    }

    /// Glowing embers drifting up from a fire.
    pub fn embers(at: Vec3, size: f32) -> Self {
        let s = size.max(0.05);
        Self::new(at)
            .with_rate(9.0 * s.max(0.3))
            .with_silhouette(ParticleSilhouette::SoftOrb)
            .with_blend(ParticleBlend::Additive)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::new(0.3, 0.1, 0.3) * s)
            .with_velocity(Vec3::new(0.0, 2.2 * s.sqrt(), 0.0))
            .with_velocity_jitter(Vec3::new(0.7, 0.8, 0.7))
            .with_acceleration(Vec3::new(0.0, 0.4, 0.0))
            .with_lifetime(2.2)
            .with_size(0.022)
            .with_size_variation(0.5)
            .with_size_over_life(SizeOverLife::Shrink)
            .with_color(col(1.0, 0.65, 0.2, 1.0))
            .with_secondary_color(col(1.0, 0.2, 0.02, 1.0))
            .with_emissive_intensity(5.0)
            .with_turbulence(2.5)
            .with_drag(0.4)
    }

    /// A shower of sparks flying along `direction` (length = speed in m/s)
    /// that bounce off the ground.
    pub fn sparks(at: Vec3, direction: Vec3, count: u32) -> Self {
        let speed = direction.length().max(0.5);
        Self::new(at)
            .with_mode(ParticleMode::Burst)
            .with_burst_count(count.max(1))
            .with_silhouette(ParticleSilhouette::SparkStreak)
            .with_blend(ParticleBlend::Additive)
            .with_shape(ParticleShape::Point)
            .with_velocity(direction)
            .with_velocity_jitter(Vec3::splat(speed * 0.6))
            .with_acceleration(GRAVITY)
            .with_lifetime(0.9)
            .with_size(0.035)
            .with_size_variation(0.5)
            .with_size_over_life(SizeOverLife::Shrink)
            .with_velocity_stretch(0.06)
            .with_color(col(1.0, 0.92, 0.65, 1.0))
            .with_secondary_color(col(1.0, 0.35, 0.05, 1.0))
            .with_emissive_intensity(6.0)
            .with_turbulence(0.0)
            .with_drag(0.15)
            .with_collision(ParticleCollision {
                bounce: 0.45,
                friction: 0.25,
            })
    }

    /// A cloud of dust `radius` metres across, the colour of what broke.
    pub fn dust(at: Vec3, radius: f32, color: Color) -> Self {
        let r = radius.max(0.1);
        let count = (24.0 * r * r).clamp(6.0, 600.0) as u32;
        Self::new(at)
            .with_mode(ParticleMode::Burst)
            .with_burst_count(count)
            .with_silhouette(ParticleSilhouette::Puff)
            .with_blend(ParticleBlend::Alpha)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::new(0.7, 0.4, 0.7) * r)
            .with_velocity(Vec3::new(0.0, 0.5 * r.sqrt(), 0.0))
            .with_velocity_jitter(Vec3::new(1.4, 0.7, 1.4) * r.sqrt())
            .with_acceleration(Vec3::new(0.0, -0.25, 0.0))
            .with_lifetime(4.0)
            .with_size(0.55 * r.sqrt())
            .with_size_variation(0.45)
            .with_size_over_life(SizeOverLife::Expand)
            .with_color(col(color.r, color.g, color.b, 0.6))
            .with_secondary_color(col(
                (color.r * 1.15).min(1.0),
                (color.g * 1.15).min(1.0),
                (color.b * 1.15).min(1.0),
                0.3,
            ))
            .with_turbulence(0.35)
            .with_drag(0.8)
            .with_spin(0.5)
    }

    /// `count` solid pieces flung out at about `speed` m/s; they tumble,
    /// bounce, come to rest on the ground and sink away after a few seconds.
    pub fn debris(at: Vec3, color: Color, count: u32, speed: f32) -> Self {
        let v = speed.max(0.1);
        let shade = |k: f32| col(color.r * k, color.g * k, color.b * k, 1.0);
        Self::new(at)
            .with_mode(ParticleMode::Burst)
            .with_burst_count(count.max(1))
            .with_silhouette(ParticleSilhouette::Chunk)
            .with_blend(ParticleBlend::Alpha)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::splat(0.3))
            .with_velocity(Vec3::new(0.0, 0.55 * v, 0.0))
            .with_velocity_jitter(Vec3::new(v, 0.6 * v, v))
            .with_acceleration(GRAVITY)
            .with_lifetime(7.0)
            .with_size(0.09)
            .with_size_variation(0.7)
            .with_size_over_life(SizeOverLife::Constant)
            .with_color(shade(1.0))
            .with_secondary_color(shade(0.65))
            .with_turbulence(0.0)
            .with_drag(0.03)
            .with_spin(9.0)
            .with_collision(ParticleCollision {
                bounce: 0.25,
                friction: 0.55,
            })
    }

    /// The flash of fire at the heart of an explosion `radius` metres across.
    pub fn fireball(at: Vec3, radius: f32) -> Self {
        let r = radius.max(0.2);
        Self::new(at)
            .with_mode(ParticleMode::Burst)
            .with_burst_count((12.0 * r).clamp(10.0, 120.0) as u32)
            .with_silhouette(ParticleSilhouette::Fire)
            .with_blend(ParticleBlend::Additive)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::splat(0.35 * r))
            .with_velocity(Vec3::new(0.0, 0.9 * r, 0.0))
            .with_velocity_jitter(Vec3::splat(1.6 * r))
            .with_acceleration(Vec3::new(0.0, 2.0, 0.0))
            .with_lifetime(0.7)
            .with_size(0.55 * r)
            .with_size_variation(0.5)
            .with_size_over_life(SizeOverLife::GrowThenFade)
            .with_color(col(1.0, 0.62, 0.25, 0.75))
            .with_secondary_color(col(0.55, 0.08, 0.02, 0.3))
            .with_emissive_intensity(2.2)
            .with_turbulence(1.0)
            .with_drag(1.0)
            .with_spin(2.0)
    }
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
