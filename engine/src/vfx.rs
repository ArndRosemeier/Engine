//! Gameplay-agnostic, lifecycle-owned procedural visual effects.
use crate::{
    Color, EmitterId, EngineError, EngineResult, EntityId, Mesh, ParticleBlend, ParticleEmitter,
    ParticleForce, ParticleMode, ParticleShape, ParticleSilhouette, Place, RibbonId, RibbonProfile,
    RibbonStyle, SizeOverLife, SurfaceMaterial, World,
};
use glam::Vec3;
use std::collections::HashMap;

const MAX_ACTIVE_EFFECTS: usize = 16;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectHandle(u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    SingleTarget,
    Aoe,
    Pbaoe,
    Cone,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualKind {
    Fire,
    Frost,
    Lightning,
    Poison,
    Root,
    Hold,
    Snare,
    Charm,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VfxPalette {
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
}
impl VfxPalette {
    pub fn fire() -> Self {
        Self {
            primary: Color::rgb(255, 55, 8),
            secondary: Color::rgb(255, 170, 20),
            accent: Color::rgb(255, 245, 150),
        }
    }
    pub fn frost() -> Self {
        Self {
            primary: Color::rgb(40, 150, 255),
            secondary: Color::rgb(150, 235, 255),
            accent: Color::WHITE,
        }
    }
    pub fn lightning() -> Self {
        Self {
            primary: Color::rgb(80, 120, 255),
            secondary: Color::rgb(190, 210, 255),
            accent: Color::WHITE,
        }
    }
    pub fn poison() -> Self {
        Self {
            primary: Color::rgb(35, 180, 45),
            secondary: Color::rgb(130, 240, 40),
            accent: Color::rgb(220, 255, 80),
        }
    }
    pub fn root() -> Self {
        Self {
            primary: Color::rgb(80, 125, 35),
            secondary: Color::rgb(165, 85, 25),
            accent: Color::rgb(225, 175, 55),
        }
    }
    pub fn hold() -> Self {
        Self {
            primary: Color::rgb(110, 60, 220),
            secondary: Color::rgb(195, 130, 255),
            accent: Color::WHITE,
        }
    }
    pub fn snare() -> Self {
        Self {
            primary: Color::rgb(100, 70, 160),
            secondary: Color::rgb(190, 150, 235),
            accent: Color::WHITE,
        }
    }
    pub fn charm() -> Self {
        Self {
            primary: Color::rgb(220, 55, 180),
            secondary: Color::rgb(255, 135, 220),
            accent: Color::rgb(255, 235, 120),
        }
    }
}
pub fn palette(k: VisualKind) -> VfxPalette {
    match k {
        VisualKind::Fire => VfxPalette::fire(),
        VisualKind::Frost => VfxPalette::frost(),
        VisualKind::Lightning => VfxPalette::lightning(),
        VisualKind::Poison => VfxPalette::poison(),
        VisualKind::Root => VfxPalette::root(),
        VisualKind::Hold => VfxPalette::hold(),
        VisualKind::Snare => VfxPalette::snare(),
        VisualKind::Charm => VfxPalette::charm(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectSpec {
    pub kind: VisualKind,
    pub delivery: Delivery,
    pub origin: Vec3,
    pub target: Vec3,
    pub range_m: f32,
    pub radius_m: f32,
    pub angle_deg: f32,
    pub duration_s: f32,
    pub scale: f32,
    pub intensity: f32,
    pub seed: u32,
}
impl EffectSpec {
    pub fn validate(self) -> EngineResult<Self> {
        let values = [
            self.range_m,
            self.radius_m,
            self.angle_deg,
            self.duration_s,
            self.scale,
            self.intensity,
        ];
        if values.iter().any(|v| !v.is_finite())
            || !self.origin.is_finite()
            || !self.target.is_finite()
        {
            return Err(EngineError::InvalidValue(
                "VFX parameters must be finite".into(),
            ));
        }
        if self.range_m <= 0.0
            || self.radius_m <= 0.0
            || self.duration_s <= 0.0
            || self.scale <= 0.0
            || self.intensity < 0.0
        {
            return Err(EngineError::InvalidValue(
                "VFX range, radius, duration, scale must be > 0 and intensity >= 0".into(),
            ));
        }
        if self.delivery == Delivery::Cone
            && (!(0.0..=180.0).contains(&self.angle_deg) || self.angle_deg == 0.0)
        {
            return Err(EngineError::InvalidValue(
                "cone angle must be in (0, 180] degrees".into(),
            ));
        }
        if self.delivery == Delivery::SingleTarget
            && self.origin.distance_squared(self.target) < 0.0001
        {
            return Err(EngineError::InvalidValue(
                "single-target VFX origin and target must differ".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy)]
enum Motion {
    Static { center: Vec3 },
    Travel { from: Vec3, to: Vec3, end: f32 },
    Pulse { center: Vec3, phase: f32 },
    Rise { center: Vec3, height: f32 },
}
struct MeshLayer {
    id: EntityId,
    motion: Motion,
    base_scale: f32,
    yaw: f32,
    start: f32,
    end: f32,
}
#[derive(Clone, Copy)]
enum EmitterMotion {
    Static,
    Travel {
        from: Vec3,
        to: Vec3,
        start_s: f32,
        end_s: f32,
    },
}
#[derive(Clone, Copy)]
enum RibbonMotion {
    Curve {
        from: Vec3,
        to: Vec3,
        bend: Vec3,
        start_s: f32,
        end_s: f32,
    },
}
struct RibbonLayer {
    id: RibbonId,
    motion: RibbonMotion,
    recording: bool,
}

struct EmitterLayer {
    id: EmitterId,
    motion: EmitterMotion,
    stop_s: Option<f32>,
}
#[derive(Clone, Copy)]
struct PendingEmitter {
    start_s: f32,
    stop_s: Option<f32>,
    motion: EmitterMotion,
    emitter: ParticleEmitter,
}
struct ActiveEffect {
    meshes: Vec<MeshLayer>,
    emitters: Vec<EmitterLayer>,
    ribbons: Vec<RibbonLayer>,
    pending_emitters: Vec<PendingEmitter>,
    age_s: f32,
    duration_s: f32,
}
#[derive(Default)]
pub struct VfxSystem {
    next_handle: u64,
    active: HashMap<EffectHandle, ActiveEffect>,
    paused: bool,
}
impl VfxSystem {
    pub fn new() -> Self {
        Self {
            next_handle: 1,
            active: HashMap::new(),
            paused: false,
        }
    }
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn is_active(&self, h: EffectHandle) -> bool {
        self.active.contains_key(&h)
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn set_paused(&mut self, world: &mut World, paused: bool) -> EngineResult<()> {
        self.paused = paused;
        world
            .particles_mut()
            .set_time_scale(if paused { 0.0 } else { 1.0 })?;
        world
            .ribbons_mut()
            .set_time_scale(if paused { 0.0 } else { 1.0 })
    }
    pub fn spawn(&mut self, world: &mut World, spec: EffectSpec) -> EngineResult<EffectHandle> {
        let spec = spec.validate()?;
        if self.active.len() >= MAX_ACTIVE_EFFECTS {
            return Err(EngineError::ResourceLimit(format!(
                "active VFX capacity ({MAX_ACTIVE_EFFECTS}) exhausted"
            )));
        }
        let mut effect = ActiveEffect {
            meshes: Vec::new(),
            emitters: Vec::new(),
            ribbons: Vec::new(),
            pending_emitters: Vec::new(),
            age_s: 0.0,
            duration_s: spec.duration_s,
        };
        if let Err(error) = build_recipe(world, spec, &mut effect) {
            cleanup(world, effect).expect("partially built VFX cleanup failed");
            return Err(error);
        }
        let handle = EffectHandle(self.next_handle);
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or_else(|| EngineError::ResourceLimit("VFX handle space exhausted".into()))?;
        self.active.insert(handle, effect);
        Ok(handle)
    }
    pub fn interrupt(&mut self, world: &mut World, h: EffectHandle) -> EngineResult<()> {
        let e = self
            .active
            .remove(&h)
            .ok_or_else(|| EngineError::InvalidValue("stale or inactive VFX handle".into()))?;
        cleanup(world, e)
    }
    pub fn reset(&mut self, world: &mut World) -> EngineResult<()> {
        let effects = std::mem::take(&mut self.active);
        for (_, e) in effects {
            cleanup(world, e)?;
        }
        Ok(())
    }
    pub fn update(&mut self, world: &mut World, dt_s: f32) -> EngineResult<()> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err(EngineError::InvalidValue(
                "VFX dt must be finite and >= 0".into(),
            ));
        }
        let dt = if self.paused { 0.0 } else { dt_s };
        world.particles_mut().advance(dt_s)?;
        world.ribbons_mut().advance(dt_s)?;
        let mut expired = Vec::new();
        for (h, e) in &mut self.active {
            e.age_s += dt;
            let t = (e.age_s / e.duration_s).clamp(0.0, 1.0);
            let mut due = Vec::new();
            e.pending_emitters.retain(|pending| {
                if e.age_s >= pending.start_s {
                    due.push(*pending);
                    false
                } else {
                    true
                }
            });
            for pending in due {
                let id = world.particles_mut().start(pending.emitter)?;
                e.emitters.push(EmitterLayer {
                    id,
                    motion: pending.motion,
                    stop_s: pending.stop_s,
                });
            }
            let mut stopped = Vec::new();
            for (index, layer) in e.emitters.iter().enumerate() {
                if layer.stop_s.is_some_and(|stop_s| e.age_s >= stop_s) {
                    world.particles_mut().stop(layer.id)?;
                    stopped.push(index);
                    continue;
                }
                if let EmitterMotion::Travel {
                    from,
                    to,
                    start_s,
                    end_s,
                } = layer.motion
                {
                    let travel_t = smooth((e.age_s - start_s) / (end_s - start_s));
                    world
                        .particles_mut()
                        .set_emitter_position(layer.id, from.lerp(to, travel_t))?;
                }
            }
            for index in stopped.into_iter().rev() {
                e.emitters.swap_remove(index);
            }
            for layer in &mut e.ribbons {
                if !layer.recording {
                    continue;
                }
                let RibbonMotion::Curve {
                    from,
                    to,
                    bend,
                    start_s,
                    end_s,
                } = layer.motion;
                if e.age_s < start_s {
                    continue;
                }
                let q = smooth((e.age_s - start_s) / (end_s - start_s));
                let position = from.lerp(to, q) + bend * (q * (1.0 - q) * 4.0);
                world.ribbons_mut().update_head(layer.id, position)?;
                if e.age_s >= end_s {
                    world.ribbons_mut().stop(layer.id)?;
                    layer.recording = false;
                }
            }
            for l in &e.meshes {
                let local_t = ((t - l.start) / (l.end - l.start)).clamp(0.0, 1.0);
                let (p, mut s, y) = animate(l, local_t);
                if t < l.start || t > l.end {
                    s = 0.0001;
                }
                world.set_place(
                    l.id,
                    Place::new(p.x, p.y, p.z).with_yaw_deg(y).with_scale(s),
                )?;
            }
            if e.age_s >= e.duration_s {
                expired.push(*h);
            }
        }
        for h in expired {
            self.interrupt(world, h)?;
        }
        Ok(())
    }
}
fn cleanup(world: &mut World, e: ActiveEffect) -> EngineResult<()> {
    for layer in e.ribbons {
        if layer.recording {
            world.ribbons_mut().stop(layer.id)?;
        }
    }
    for layer in e.emitters {
        world.particles_mut().stop(layer.id)?;
    }
    for m in e.meshes {
        world.despawn(m.id);
    }
    Ok(())
}
fn animate(l: &MeshLayer, t: f32) -> (Vec3, f32, f32) {
    let fade = (1.0 - smooth((t - 0.78) / 0.22)).max(0.03);
    match l.motion {
        Motion::Static { center } => (center, l.base_scale * fade, l.yaw),
        Motion::Travel { from, to, end } => (
            from.lerp(to, smooth(t / end)),
            l.base_scale * (0.7 + 0.3 * (t / end).min(1.0)) * fade,
            l.yaw,
        ),
        Motion::Pulse { center, phase } => (
            center,
            l.base_scale
                * (0.72 + 0.28 * (t * std::f32::consts::TAU * 2.0 + phase).sin().abs())
                * fade,
            l.yaw + t * 90.0,
        ),
        Motion::Rise { center, height } => (
            center + Vec3::Y * (smooth(t) * height),
            l.base_scale * fade,
            l.yaw + t * 45.0,
        ),
    }
}
fn smooth(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn add_mesh(
    world: &mut World,
    e: &mut ActiveEffect,
    mesh: Mesh,
    position: Vec3,
    scale: f32,
    yaw: f32,
    motion: Motion,
) -> EngineResult<()> {
    let id = world.place(
        mesh,
        Place::new(position.x, position.y, position.z)
            .with_yaw_deg(yaw)
            .with_scale(scale),
    )?;
    world.set_casts_shadow(id, false)?;
    e.meshes.push(MeshLayer {
        id,
        motion,
        base_scale: scale,
        yaw,
        start: 0.0,
        end: 1.0,
    });
    Ok(())
}
fn delay_last_mesh(e: &mut ActiveEffect, start: f32, end: f32) {
    let layer = e.meshes.last_mut().expect("mesh delay requires a mesh");
    assert!(
        start >= 0.0 && start < end && end <= 1.0,
        "invalid VFX mesh interval"
    );
    layer.start = start;
    layer.end = end;
}
fn add_emitter(
    world: &mut World,
    e: &mut ActiveEffect,
    emitter: ParticleEmitter,
) -> EngineResult<()> {
    let id = world.particles_mut().start(emitter)?;
    e.emitters.push(EmitterLayer {
        id,
        motion: EmitterMotion::Static,
        stop_s: None,
    });
    Ok(())
}
fn add_timed_emitter(
    e: &mut ActiveEffect,
    start_s: f32,
    stop_s: Option<f32>,
    motion: EmitterMotion,
    emitter: ParticleEmitter,
) {
    assert!(
        start_s >= 0.0 && start_s < e.duration_s,
        "delayed emitter outside effect lifetime"
    );
    if let Some(stop_s) = stop_s {
        assert!(
            stop_s > start_s && stop_s <= e.duration_s,
            "invalid emitter stop time"
        );
    }
    e.pending_emitters.push(PendingEmitter {
        start_s,
        stop_s,
        motion,
        emitter,
    });
}
fn build_recipe(world: &mut World, s: EffectSpec, e: &mut ActiveEffect) -> EngineResult<()> {
    let p = palette(s.kind);
    let direction = (s.target - s.origin).normalize_or_zero();
    let center = if s.delivery == Delivery::Pbaoe {
        s.origin
    } else {
        s.target
    };
    match s.delivery {
        Delivery::SingleTarget => {
            add_mesh(
                world,
                e,
                ring(p.secondary.with_alpha(0.28), 32),
                s.origin,
                s.scale * 0.58,
                0.0,
                Motion::Pulse {
                    center: s.origin,
                    phase: 0.0,
                },
            )?;
            delay_last_mesh(e, 0.0, 0.18);
            if matches!(
                s.kind,
                VisualKind::Fire | VisualKind::Frost | VisualKind::Poison | VisualKind::Root
            ) {
                add_mesh(
                    world,
                    e,
                    identity_core(s.kind, p),
                    s.origin,
                    s.scale * 1.18,
                    0.0,
                    Motion::Travel {
                        from: s.origin,
                        to: s.target,
                        end: 1.0,
                    },
                )?;
                delay_last_mesh(e, 0.12, 0.46);
            }
            add_mesh(
                world,
                e,
                ring(p.secondary.with_alpha(0.34), 40),
                s.target,
                s.scale * 1.35,
                0.0,
                Motion::Pulse {
                    center: s.target,
                    phase: 0.0,
                },
            )?;
            delay_last_mesh(e, 0.44, 0.66);
        }
        Delivery::Aoe | Delivery::Pbaoe => {
            add_mesh(
                world,
                e,
                ring(p.primary.with_alpha(0.24), 48),
                center,
                s.radius_m * 0.62,
                0.0,
                Motion::Pulse { center, phase: 0.0 },
            )?;
            delay_last_mesh(e, 0.0, 0.25);
            if s.kind != VisualKind::Poison {
                add_mesh(
                    world,
                    e,
                    radial_rays(p.primary.with_alpha(0.30), 10),
                    center,
                    s.radius_m * 0.9,
                    0.0,
                    Motion::Pulse { center, phase: 1.2 },
                )?;
                delay_last_mesh(e, 0.22, 0.62);
                add_mesh(
                    world,
                    e,
                    ring(p.secondary.with_alpha(0.32), 48),
                    center,
                    s.radius_m * 0.76,
                    0.0,
                    Motion::Pulse { center, phase: 2.0 },
                )?;
                delay_last_mesh(e, 0.25, 0.74);
            } else {
                add_mesh(
                    world,
                    e,
                    ring(p.secondary.with_alpha(0.07), 48),
                    center,
                    s.radius_m * 0.34,
                    0.0,
                    Motion::Pulse { center, phase: 0.0 },
                )?;
                delay_last_mesh(e, 0.12, 0.78);
            }
        }
        Delivery::Cone => {
            let yaw = direction.x.atan2(direction.z).to_degrees();
            add_mesh(
                world,
                e,
                arc(
                    p.primary.with_alpha(0.28),
                    s.range_m * 0.45,
                    s.angle_deg,
                    24,
                ),
                s.origin,
                s.scale,
                yaw,
                Motion::Pulse {
                    center: s.origin,
                    phase: 0.4,
                },
            )?;
            delay_last_mesh(e, 0.0, 0.2);
            add_mesh(
                world,
                e,
                arc(p.secondary.with_alpha(0.38), s.range_m, s.angle_deg, 32),
                s.origin,
                s.scale,
                yaw,
                Motion::Pulse {
                    center: s.origin,
                    phase: 1.2,
                },
            )?;
            delay_last_mesh(e, 0.18, 0.62);
        }
    }
    let readable_start = e.meshes.len();
    add_readable_geometry(world, e, s, center, p)?;
    if s.kind == VisualKind::Fire {
        for layer in &mut e.meshes[readable_start..] {
            layer.start = 0.43;
            layer.end = 0.96;
        }
    }
    add_recipe_ribbons(world, e, s, center, p)?;
    identity_layers(world, e, s, center, direction, p)
}
fn add_static_ribbon(
    world: &mut World,
    e: &mut ActiveEffect,
    points: &[Vec3],
    style: RibbonStyle,
) -> EngineResult<()> {
    assert!(
        points.len() >= 2,
        "static ribbon requires at least two points"
    );
    let id = world.ribbons_mut().start(points[0], style)?;
    for point in &points[1..] {
        world.ribbons_mut().update_head(id, *point)?;
    }
    world.ribbons_mut().stop(id)?;
    e.ribbons.push(RibbonLayer {
        id,
        motion: RibbonMotion::Curve {
            from: points[0],
            to: *points.last().expect("non-empty static ribbon"),
            bend: Vec3::ZERO,
            start_s: 0.0,
            end_s: e.duration_s,
        },
        recording: false,
    });
    Ok(())
}

fn curved_points(from: Vec3, to: Vec3, bend: Vec3, segments: u32) -> Vec<Vec3> {
    (0..=segments)
        .map(|step| {
            let t = step as f32 / segments as f32;
            from.lerp(to, t) + bend * (t * (1.0 - t) * 4.0)
        })
        .collect()
}

fn ring_points(center: Vec3, radius: f32, y: f32, segments: u32) -> Vec<Vec3> {
    (0..=segments)
        .map(|step| {
            let a = step as f32 / segments as f32 * std::f32::consts::TAU;
            center + Vec3::new(a.cos() * radius, y, a.sin() * radius)
        })
        .collect()
}
fn add_readable_geometry(
    world: &mut World,
    e: &mut ActiveEffect,
    s: EffectSpec,
    center: Vec3,
    p: VfxPalette,
) -> EngineResult<()> {
    let mut add = |mesh: Mesh| {
        add_mesh(
            world,
            e,
            mesh,
            Vec3::ZERO,
            1.0,
            0.0,
            Motion::Static { center: Vec3::ZERO },
        )
    };
    match s.kind {
        VisualKind::Fire => {
            for i in 0..6 {
                let a = i as f32 * std::f32::consts::TAU / 6.0;
                let from = center + Vec3::new(a.cos() * 0.52, 0.05, a.sin() * 0.52);
                let points = curved_points(
                    from,
                    center + Vec3::new(a.cos() * 0.18, 2.15 + (i % 3) as f32 * 0.3, a.sin() * 0.18),
                    Vec3::new(-a.sin() * 0.38, 0.22, a.cos() * 0.38),
                    8,
                );
                add(polyline(
                    &points,
                    0.105,
                    if i % 3 == 0 {
                        p.accent
                    } else if i % 2 == 0 {
                        p.secondary
                    } else {
                        p.primary
                    },
                ))?;
            }
        }
        VisualKind::Lightning => {
            let from = s.origin + Vec3::Y * 0.15;
            let to = s.target + Vec3::Y * 1.05;
            let mut main = Vec::new();
            for i in 0..=14 {
                let t = i as f32 / 14.0;
                let offset = if i == 0 || i == 14 {
                    Vec3::ZERO
                } else {
                    Vec3::new(
                        0.0,
                        ((i * 5) as f32).sin() * 0.18,
                        ((i * 9) as f32).cos() * 0.24,
                    )
                };
                main.push(from.lerp(to, t) + offset);
            }
            add(polyline(&main, 0.085, Color::WHITE))?;
            for branch in 0..4 {
                let root = main[3 + branch * 3];
                let a = branch as f32 * 1.4 + 0.5;
                let points = curved_points(
                    root,
                    root + Vec3::new(a.cos() * 1.1, 0.2, a.sin() * 1.1),
                    Vec3::Y * 0.15,
                    4,
                );
                add(polyline(&points, 0.045, p.secondary))?;
            }
        }
        VisualKind::Hold => {
            let r = 1.15;
            for i in 0..6 {
                let a = i as f32 * std::f32::consts::TAU / 6.0;
                let b = a + if i % 2 == 0 { 0.14 } else { -0.14 };
                add(polyline(
                    &[
                        center + Vec3::new(a.cos() * r, 0.04, a.sin() * r),
                        center + Vec3::new(b.cos() * r, 2.75, b.sin() * r),
                    ],
                    0.055,
                    if i % 2 == 0 { p.secondary } else { p.primary },
                ))?;
            }
            for y in [0.42, 1.45, 2.68] {
                add(polyline(&ring_points(center, r, y, 28), 0.04, p.secondary))?;
            }
        }
        VisualKind::Snare => {
            let anchors: Vec<Vec3> = (0..8)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 8.0;
                    center + Vec3::new(a.cos() * 2.1, 0.05, a.sin() * 2.1)
                })
                .collect();
            let waist = center + Vec3::Y * 1.1;
            for anchor in &anchors {
                add(polyline(&[*anchor, waist], 0.038, p.secondary))?;
            }
            let mut outer = anchors.clone();
            outer.push(anchors[0]);
            add(polyline(&outer, 0.035, p.primary))?;
            for i in 0..3 {
                let a = i as f32 * std::f32::consts::TAU / 3.0;
                add(polyline(
                    &[
                        center + Vec3::new(a.cos() * 0.9, 0.05, a.sin() * 0.9),
                        center + Vec3::new((-a).cos() * 0.45, 2.25, (-a).sin() * 0.45),
                    ],
                    0.05,
                    p.secondary,
                ))?;
            }
        }
        VisualKind::Charm => {
            drop(add);
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::TAU / 8.0;
                let pos = center
                    + Vec3::new(
                        a.cos() * 1.25,
                        1.25 + (i % 3) as f32 * 0.42,
                        a.sin() * 0.55 - 0.5,
                    );
                add_mesh(
                    world,
                    e,
                    heart(if i % 2 == 0 { p.secondary } else { p.primary }),
                    pos,
                    0.46 + (i % 3) as f32 * 0.08,
                    0.0,
                    Motion::Static { center: pos },
                )?;
            }
            add_mesh(
                world,
                e,
                polyline(&ring_points(center, 1.45, 1.3, 36), 0.045, p.secondary),
                Vec3::ZERO,
                1.0,
                0.0,
                Motion::Static { center: Vec3::ZERO },
            )?;
            add_mesh(
                world,
                e,
                polyline(&ring_points(center, 1.18, 1.85, 36), 0.035, p.primary),
                Vec3::ZERO,
                1.0,
                0.0,
                Motion::Static { center: Vec3::ZERO },
            )?;
        }
        _ => {}
    }
    Ok(())
}
fn add_recipe_ribbons(
    world: &mut World,
    e: &mut ActiveEffect,
    s: EffectSpec,
    center: Vec3,
    p: VfxPalette,
) -> EngineResult<()> {
    let style = |width: f32, profile: RibbonProfile, cross: bool, lifetime: f32| {
        let (core, edge, emissive) = match s.kind {
            VisualKind::Fire => (p.accent, p.primary.with_alpha(0.92), 2.1),
            VisualKind::Lightning => (Color::WHITE, p.primary.with_alpha(0.92), 2.5),
            VisualKind::Charm => (p.secondary, p.primary.with_alpha(0.82), 1.8),
            _ => (p.secondary, p.primary.with_alpha(0.82), 1.4),
        };
        RibbonStyle::new(width * s.scale, core, edge, lifetime)?
            .with_emissive_intensity(emissive * s.intensity)?
            .with_point_spacing(0.045)
            .map(|value| value.with_profile(profile).with_cross_ribbon(cross))
    };

    if s.delivery == Delivery::SingleTarget && s.kind == VisualKind::Fire {
        let id = world
            .ribbons_mut()
            .start(s.origin, style(0.28, RibbonProfile::Turbulent, true, 2.2)?)?;
        e.ribbons.push(RibbonLayer {
            id,
            recording: true,
            motion: RibbonMotion::Curve {
                from: s.origin,
                to: s.target,
                bend: Vec3::new(0.0, 0.32, 0.34),
                start_s: s.duration_s * 0.10,
                end_s: s.duration_s * 0.43,
            },
        });
    }

    match s.kind {
        VisualKind::Fire => {
            for tongue in 0..6 {
                let a = tongue as f32 * std::f32::consts::TAU / 6.0;
                let from = center + Vec3::new(a.cos() * 0.54, 0.04, a.sin() * 0.54);
                let height = 1.75 + (tongue % 3) as f32 * 0.35;
                let to = center
                    + Vec3::new(
                        a.cos() * (0.18 + (tongue % 2) as f32 * 0.12),
                        height,
                        a.sin() * (0.18 + (tongue % 2) as f32 * 0.12),
                    );
                let bend = Vec3::new(-a.sin() * 0.42, 0.26, a.cos() * 0.42);
                let points = curved_points(from, to, bend, 18);
                add_static_ribbon(
                    world,
                    e,
                    &points,
                    style(
                        if tongue % 3 == 0 { 0.24 } else { 0.17 },
                        RibbonProfile::Turbulent,
                        true,
                        s.duration_s,
                    )?,
                )?;
            }
        }
        VisualKind::Lightning => {
            let from = s.origin + Vec3::Y * 0.25;
            let to = s.target + Vec3::Y * 1.05;
            let mut main = Vec::new();
            for step in 0..=20 {
                let t = step as f32 / 20.0;
                let offset = if step == 0 || step == 20 {
                    Vec3::ZERO
                } else {
                    Vec3::new(
                        ((step * 17) as f32).sin() * 0.22,
                        ((step * 11) as f32).cos() * 0.16,
                        ((step * 7) as f32).sin() * 0.26,
                    )
                };
                main.push(from.lerp(to, t) + offset);
            }
            add_static_ribbon(
                world,
                e,
                &main,
                style(0.19, RibbonProfile::Jagged, true, 8.0)?,
            )?;
            for branch in 0..4 {
                let root_index = 6 + branch * 3;
                let root = main[root_index];
                let phase = branch as f32 * 1.61 + 0.35;
                let end = root
                    + Vec3::new(
                        phase.cos() * 1.15,
                        0.35 + branch as f32 * 0.12,
                        phase.sin() * 1.15,
                    );
                let points = curved_points(
                    root,
                    end,
                    Vec3::new(phase.sin() * 0.28, 0.18, phase.cos() * -0.28),
                    8,
                );
                add_static_ribbon(
                    world,
                    e,
                    &points,
                    style(0.085, RibbonProfile::Jagged, true, 8.0)?,
                )?;
            }
        }
        VisualKind::Hold => {
            let radius = 1.18 * s.scale;
            for bar in 0..6 {
                let a = bar as f32 * std::f32::consts::TAU / 6.0;
                let bottom = center + Vec3::new(a.cos() * radius, 0.04, a.sin() * radius);
                let top_angle = a + if bar % 2 == 0 { 0.16 } else { -0.16 };
                let top =
                    center + Vec3::new(top_angle.cos() * radius, 2.75, top_angle.sin() * radius);
                add_static_ribbon(
                    world,
                    e,
                    &[bottom, bottom.lerp(top, 0.5), top],
                    style(0.075, RibbonProfile::Smooth, true, 8.0)?,
                )?;
            }
            for (ring, y) in [0.48_f32, 1.42, 2.62].into_iter().enumerate() {
                add_static_ribbon(
                    world,
                    e,
                    &ring_points(center, radius * (1.0 - ring as f32 * 0.05), y, 32),
                    style(0.055, RibbonProfile::Orbit, false, 8.0)?,
                )?;
            }
        }
        VisualKind::Snare => {
            let anchors: Vec<Vec3> = (0..8)
                .map(|index| {
                    let a = index as f32 * std::f32::consts::TAU / 8.0;
                    center
                        + Vec3::new(
                            a.cos() * s.radius_m * 0.82,
                            0.05,
                            a.sin() * s.radius_m * 0.82,
                        )
                })
                .collect();
            let waist = center + Vec3::Y * 1.05;
            for anchor in &anchors {
                add_static_ribbon(
                    world,
                    e,
                    &[*anchor, anchor.lerp(waist, 0.5) + Vec3::Y * 0.12, waist],
                    style(0.052, RibbonProfile::Organic, false, 8.0)?,
                )?;
            }
            let mut outer = anchors.clone();
            outer.push(anchors[0]);
            add_static_ribbon(
                world,
                e,
                &outer,
                style(0.045, RibbonProfile::Organic, false, 8.0)?,
            )?;
            for tether in 0..3 {
                let a = tether as f32 * std::f32::consts::TAU / 3.0 + 0.35;
                let from = center + Vec3::new(a.cos() * 0.86, 0.05, a.sin() * 0.86);
                let to = center + Vec3::new((-a).cos() * 0.42, 2.25, (-a).sin() * 0.42);
                add_static_ribbon(
                    world,
                    e,
                    &curved_points(from, to, Vec3::new(a.sin() * 0.35, 0.2, a.cos() * 0.35), 10),
                    style(0.065, RibbonProfile::Organic, true, 8.0)?,
                )?;
            }
        }
        VisualKind::Charm => {
            for orbit in 0..2 {
                let y = 1.25 + orbit as f32 * 0.48;
                let mut points = ring_points(center, 1.45 + orbit as f32 * 0.22, y, 40);
                for (index, point) in points.iter_mut().enumerate() {
                    point.y += (index as f32 * 0.31 + orbit as f32).sin() * 0.24;
                }
                add_static_ribbon(
                    world,
                    e,
                    &points,
                    style(0.085, RibbonProfile::Orbit, false, 8.0)?,
                )?;
            }
        }
        VisualKind::Root => {
            for branch in 0..6 {
                let a = branch as f32 * std::f32::consts::TAU / 6.0;
                let from = center
                    + Vec3::new(
                        a.cos() * s.radius_m * 0.82,
                        0.04,
                        a.sin() * s.radius_m * 0.82,
                    );
                let to = center + Vec3::new(a.cos() * 0.42, 2.15, a.sin() * 0.42);
                add_static_ribbon(
                    world,
                    e,
                    &curved_points(
                        from,
                        to,
                        Vec3::new(-a.sin() * 0.72, 0.75, a.cos() * 0.72),
                        14,
                    ),
                    style(0.15, RibbonProfile::Organic, true, s.duration_s)?,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn identity_layers(
    world: &mut World,
    e: &mut ActiveEffect,
    s: EffectSpec,
    c: Vec3,
    d: Vec3,
    p: VfxPalette,
) -> EngineResult<()> {
    if s.kind == VisualKind::Fire {
        return fire_layers(e, s, c, d, p);
    }
    if s.kind == VisualKind::Charm {
        return Ok(());
    }
    let radial = ParticleForce::Radial {
        center: c,
        strength: if s.kind == VisualKind::Frost {
            -2.0
        } else if s.kind == VisualKind::Poison {
            0.7
        } else {
            1.5
        },
    };
    let (sil, acc, force) = match s.kind {
        VisualKind::Fire => unreachable!("Fire has a dedicated recipe"),
        VisualKind::Frost => (ParticleSilhouette::Shard, Vec3::Y * -0.5, radial),
        VisualKind::Lightning => (ParticleSilhouette::SparkStreak, Vec3::ZERO, radial),
        VisualKind::Poison => (
            ParticleSilhouette::SmokeCloud,
            Vec3::Y * 0.45,
            ParticleForce::Vortex {
                axis: Vec3::Y,
                strength: 2.0,
            },
        ),
        VisualKind::Root => (ParticleSilhouette::Shard, Vec3::Y * 0.3, radial),
        VisualKind::Hold => (
            ParticleSilhouette::RuneMote,
            Vec3::Y * 0.2,
            ParticleForce::Vortex {
                axis: Vec3::Y,
                strength: 3.0,
            },
        ),
        VisualKind::Snare => (ParticleSilhouette::RuneMote, Vec3::ZERO, radial),
        VisualKind::Charm => (
            ParticleSilhouette::Heart,
            Vec3::Y * 0.8,
            ParticleForce::Vortex {
                axis: Vec3::Y,
                strength: 2.5,
            },
        ),
    };
    for layer in 0..2 {
        add_emitter(
            world,
            e,
            ParticleEmitter::new(c)
                .with_color(if layer == 0 { p.primary } else { p.secondary })
                .with_secondary_color(p.accent)
                .with_silhouette(if s.kind == VisualKind::Poison && layer == 1 {
                    ParticleSilhouette::Bubble
                } else {
                    sil
                })
                .with_blend(if s.kind == VisualKind::Poison {
                    ParticleBlend::Alpha
                } else {
                    ParticleBlend::Additive
                })
                .with_size_over_life(if layer == 0 {
                    SizeOverLife::GrowThenFade
                } else {
                    SizeOverLife::Shrink
                })
                .with_shape(ParticleShape::Sphere)
                .with_spread(Vec3::new(s.radius_m * 0.42, 0.9, s.radius_m * 0.42))
                .with_velocity(if s.delivery == Delivery::SingleTarget {
                    d * 5.0
                } else {
                    Vec3::Y * (0.4 + layer as f32)
                })
                .with_acceleration(acc)
                .with_force(force)
                .with_velocity_stretch(if sil == ParticleSilhouette::SparkStreak {
                    0.045
                } else {
                    0.03
                })
                .with_size(if s.kind == VisualKind::Charm {
                    0.46 + layer as f32 * 0.10
                } else {
                    0.13 + s.scale * 0.06
                })
                .with_rate(match s.kind {
                    VisualKind::Lightning => 5.0,
                    VisualKind::Root => 6.0,
                    VisualKind::Hold | VisualKind::Snare => 3.0,
                    VisualKind::Charm => 2.0,
                    _ => 14.0 + s.intensity * 6.0,
                })
                .with_lifetime(0.8 + s.duration_s * 0.25)
                .with_seed(s.seed.wrapping_add(layer))
                .with_emissive_intensity(
                    match s.kind {
                        VisualKind::Lightning => 1.6,
                        VisualKind::Charm => 1.45,
                        VisualKind::Poison => 0.25,
                        VisualKind::Root => 0.55,
                        _ => 0.95,
                    } * s.intensity,
                )
                .validate()?,
        )?;
    }
    match s.kind {
        VisualKind::Fire => unreachable!(),
        VisualKind::Frost => {
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::TAU / 8.0;
                add_mesh(
                    world,
                    e,
                    crystal(if i % 3 == 0 { p.accent } else { p.primary }),
                    c + Vec3::new(a.cos() * 1.5, 0.0, a.sin() * 1.5),
                    s.scale * (0.58 + (i % 3) as f32 * 0.14),
                    -a.to_degrees(),
                    Motion::Rise {
                        center: c + Vec3::new(a.cos() * 1.5, 0.0, a.sin() * 1.5),
                        height: 1.0,
                    },
                )?;
            }
        }
        VisualKind::Lightning => {}
        VisualKind::Poison => {
            add_mesh(
                world,
                e,
                bubbles(p.secondary.with_alpha(0.62)),
                c,
                s.scale * 1.15,
                0.0,
                Motion::Rise {
                    center: c,
                    height: 1.35,
                },
            )?;
        }
        VisualKind::Root => {
            for i in 0..3 {
                let a = i as f32 * std::f32::consts::TAU / 3.0;
                add_mesh(
                    world,
                    e,
                    tendril(p.secondary.with_alpha(0.58)),
                    c,
                    s.scale,
                    a.to_degrees(),
                    Motion::Pulse {
                        center: c,
                        phase: a,
                    },
                )?;
            }
        }
        VisualKind::Hold | VisualKind::Snare | VisualKind::Charm => {}
    }
    Ok(())
}
fn fire_layers(
    e: &mut ActiveEffect,
    s: EffectSpec,
    c: Vec3,
    d: Vec3,
    p: VfxPalette,
) -> EngineResult<()> {
    if s.delivery == Delivery::SingleTarget {
        let anticipation = ParticleEmitter::new(s.origin)
            .with_color(p.secondary)
            .with_secondary_color(p.accent)
            .with_silhouette(ParticleSilhouette::Flame)
            .with_size_over_life(SizeOverLife::GrowThenFade)
            .with_mode(ParticleMode::Burst)
            .with_burst_count(20)
            .with_shape(ParticleShape::Ring)
            .with_spread(Vec3::new(0.58, 0.12, 0.58))
            .with_velocity(Vec3::Y * 0.75)
            .with_acceleration(Vec3::Y * 0.8)
            .with_turbulence(0.18)
            .with_size(0.25 * s.scale)
            .with_lifetime(0.48)
            .with_seed(s.seed.wrapping_add(10))
            .with_emissive_intensity(1.35 * s.intensity)
            .validate()?;
        add_timed_emitter(e, 0.0, Some(0.62), EmitterMotion::Static, anticipation);
        let start_s = s.duration_s * 0.10;
        let end_s = s.duration_s * 0.43;
        let motion = EmitterMotion::Travel {
            from: s.origin,
            to: s.target,
            start_s,
            end_s,
        };
        for (offset, color, secondary, size, rate, silhouette, stretch) in [
            (
                0_u32,
                p.accent,
                Color::WHITE,
                0.22,
                42.0,
                ParticleSilhouette::Flame,
                0.02,
            ),
            (
                1,
                p.secondary,
                p.accent,
                0.34,
                58.0,
                ParticleSilhouette::Flame,
                0.03,
            ),
            (
                2,
                p.accent,
                p.primary,
                0.08,
                28.0,
                ParticleSilhouette::SparkStreak,
                0.32,
            ),
        ] {
            let emitter = ParticleEmitter::new(s.origin)
                .with_color(color)
                .with_secondary_color(secondary)
                .with_silhouette(silhouette)
                .with_size_over_life(SizeOverLife::Shrink)
                .with_shape(ParticleShape::Sphere)
                .with_spread(Vec3::splat(0.12 + offset as f32 * 0.05))
                .with_velocity(if silhouette == ParticleSilhouette::SparkStreak {
                    -d * 1.8
                } else {
                    Vec3::Y * (0.45 + offset as f32 * 0.2)
                })
                .with_acceleration(if silhouette == ParticleSilhouette::SparkStreak {
                    Vec3::Y * -0.8
                } else {
                    Vec3::Y * 1.4
                })
                .with_turbulence(0.22)
                .with_drag(0.08)
                .with_velocity_stretch(stretch)
                .with_size(size * s.scale)
                .with_rate(rate)
                .with_lifetime(0.34 + offset as f32 * 0.1)
                .with_seed(s.seed.wrapping_add(20 + offset))
                .with_emissive_intensity(
                    if silhouette == ParticleSilhouette::SparkStreak {
                        1.9
                    } else {
                        1.65
                    } * s.intensity,
                )
                .validate()?;
            add_timed_emitter(e, start_s, Some(end_s + 0.05), motion, emitter);
        }
        let trail_smoke = ParticleEmitter::new(s.origin)
            .with_color(Color::rgba(48, 38, 35, 85))
            .with_secondary_color(Color::rgba(90, 65, 45, 5))
            .with_silhouette(ParticleSilhouette::SmokeCloud)
            .with_blend(ParticleBlend::Alpha)
            .with_size_over_life(SizeOverLife::GrowThenFade)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::splat(0.12))
            .with_velocity(Vec3::Y * 0.28)
            .with_acceleration(Vec3::Y * 0.2)
            .with_turbulence(0.18)
            .with_drag(0.18)
            .with_size(0.26 * s.scale)
            .with_rate(32.0)
            .with_lifetime(0.75)
            .with_seed(s.seed.wrapping_add(30))
            .with_emissive_intensity(0.18)
            .validate()?;
        add_timed_emitter(e, start_s + 0.04, Some(end_s), motion, trail_smoke);
    }
    let impact_s = match s.delivery {
        Delivery::SingleTarget => 0.43,
        Delivery::Aoe | Delivery::Pbaoe => 0.22,
        Delivery::Cone => 0.16,
    } * s.duration_s;
    for (layer, color, secondary, size, count) in [
        (0_u32, p.accent, Color::WHITE, 0.22, 5),
        (1, p.secondary, p.accent, 0.34, 7),
        (2, p.primary, p.secondary, 0.42, 6),
    ] {
        let burst = ParticleEmitter::new(c + Vec3::Y * 0.18)
            .with_color(color)
            .with_secondary_color(secondary)
            .with_silhouette(if layer == 0 {
                ParticleSilhouette::Flame
            } else {
                ParticleSilhouette::Flame
            })
            .with_size_over_life(SizeOverLife::GrowThenFade)
            .with_mode(ParticleMode::Burst)
            .with_burst_count(count)
            .with_shape(ParticleShape::Sphere)
            .with_spread(Vec3::new(0.42 * s.scale, 0.52, 0.42 * s.scale))
            .with_velocity(Vec3::Y * (2.4 + layer as f32 * 0.55))
            .with_acceleration(Vec3::Y * 2.7)
            .with_turbulence(0.32)
            .with_drag(0.08)
            .with_size(size * s.scale)
            .with_lifetime(0.65 + layer as f32 * 0.16)
            .with_seed(s.seed.wrapping_add(100 + layer))
            .with_emissive_intensity((2.4 - layer as f32 * 0.58) * s.intensity)
            .validate()?;
        add_timed_emitter(
            e,
            impact_s,
            Some((impact_s + 0.10).min(e.duration_s)),
            EmitterMotion::Static,
            burst,
        );
    }
    let sparks = ParticleEmitter::new(c + Vec3::Y * 0.4)
        .with_color(p.accent)
        .with_secondary_color(p.primary)
        .with_silhouette(ParticleSilhouette::SparkStreak)
        .with_size_over_life(SizeOverLife::Shrink)
        .with_mode(ParticleMode::Burst)
        .with_burst_count(12)
        .with_shape(ParticleShape::Sphere)
        .with_spread(Vec3::new(1.5, 1.4, 1.5))
        .with_velocity(Vec3::Y * 2.2)
        .with_acceleration(Vec3::Y * -4.0)
        .with_force(ParticleForce::Radial {
            center: c,
            strength: 2.8,
        })
        .with_velocity_stretch(0.16)
        .with_size(0.12 * s.scale)
        .with_lifetime(0.78)
        .with_seed(s.seed.wrapping_add(160))
        .with_emissive_intensity(1.75 * s.intensity)
        .validate()?;
    add_timed_emitter(
        e,
        impact_s,
        Some((impact_s + 0.95).min(e.duration_s)),
        EmitterMotion::Static,
        sparks,
    );
    let smoke = ParticleEmitter::new(c + Vec3::Y * 0.65)
        .with_color(Color::rgba(58, 54, 58, 95))
        .with_secondary_color(Color::rgba(34, 32, 36, 4))
        .with_silhouette(ParticleSilhouette::SmokeCloud)
        .with_blend(ParticleBlend::Alpha)
        .with_size_over_life(SizeOverLife::GrowThenFade)
        .with_mode(ParticleMode::Burst)
        .with_burst_count(24)
        .with_shape(ParticleShape::Sphere)
        .with_spread(Vec3::new(0.85, 0.42, 0.85))
        .with_velocity(Vec3::Y * 0.85)
        .with_acceleration(Vec3::Y * 0.32)
        .with_turbulence(0.3)
        .with_drag(0.18)
        .with_size(0.62 * s.scale)
        .with_lifetime(1.45)
        .with_seed(s.seed.wrapping_add(170))
        .with_emissive_intensity(0.12)
        .validate()?;
    add_timed_emitter(
        e,
        impact_s + 0.12,
        Some((impact_s + 1.65).min(e.duration_s)),
        EmitterMotion::Static,
        smoke,
    );
    Ok(())
}
fn glow(mut m: Mesh) -> Mesh {
    m.set_surface_material(SurfaceMaterial::GLOWING);
    m
}
fn identity_core(k: VisualKind, p: VfxPalette) -> Mesh {
    match k {
        VisualKind::Fire => fire_head(p),
        VisualKind::Frost => crystal(p.accent),
        VisualKind::Lightning => bolt(p.accent, 4),
        VisualKind::Poison => bubbles(p.primary),
        VisualKind::Root => tendril(p.secondary),
        VisualKind::Hold => rune_ring(p.accent, 6),
        VisualKind::Snare => web(p.secondary),
        VisualKind::Charm => heart(p.primary),
    }
}
fn ring(c: Color, n: u32) -> Mesh {
    let mut m = Mesh::new();
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let b = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
        let ids = [
            m.add_point((a.cos(), 0.03, a.sin())).unwrap(),
            m.add_point((b.cos(), 0.03, b.sin())).unwrap(),
            m.add_point((b.cos() * 0.93, 0.03, b.sin() * 0.93)).unwrap(),
            m.add_point((a.cos() * 0.93, 0.03, a.sin() * 0.93)).unwrap(),
        ];
        for id in ids {
            m.set_point_color(id, c).unwrap();
        }
        m.add_quad(ids[0], ids[1], ids[2], ids[3]).unwrap();
    }
    glow(m)
}
fn ribbon_segment(m: &mut Mesh, a: Vec3, b: Vec3, width: f32, color: Color) {
    let direction = (b - a).normalize_or_zero();
    let side = direction.cross(Vec3::Y).normalize_or_zero() * width;
    let side = if side.length_squared() < 0.001 {
        Vec3::X * width
    } else {
        side
    };
    let ids = [
        m.add_point(a - side).unwrap(),
        m.add_point(a + side).unwrap(),
        m.add_point(b + side * 0.55).unwrap(),
        m.add_point(b - side * 0.55).unwrap(),
    ];
    for id in ids {
        m.set_point_color(id, color).unwrap();
    }
    m.add_quad(ids[0], ids[1], ids[2], ids[3]).unwrap();
}
fn polyline(points: &[Vec3], width: f32, color: Color) -> Mesh {
    assert!(points.len() >= 2, "polyline requires at least two points");
    let mut mesh = Mesh::new();
    for pair in points.windows(2) {
        ribbon_segment(&mut mesh, pair[0], pair[1], width, color);
    }
    glow(mesh)
}
fn radial_rays(c: Color, n: u32) -> Mesh {
    let mut m = Mesh::new();
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        ribbon_segment(
            &mut m,
            Vec3::new(a.cos() * 0.22, 0.045, a.sin() * 0.22),
            Vec3::new(a.cos(), 0.045, a.sin()),
            0.022,
            c,
        );
    }
    glow(m)
}
fn arc(c: Color, r: f32, deg: f32, n: u32) -> Mesh {
    let mut m = Mesh::new();
    let half = deg.to_radians() * 0.5;
    for i in 0..n {
        let a = -half + i as f32 / n as f32 * deg.to_radians();
        let b = -half + (i + 1) as f32 / n as f32 * deg.to_radians();
        ribbon_segment(
            &mut m,
            Vec3::new(a.sin() * r, 0.06, a.cos() * r),
            Vec3::new(b.sin() * r, 0.06, b.cos() * r),
            0.045,
            c,
        );
    }
    glow(m)
}
fn fire_head(p: VfxPalette) -> Mesh {
    let mut m = Mesh::new();
    for (z, color, scale) in [
        (-0.10, p.primary.with_alpha(0.72), 1.0),
        (0.10, p.secondary, 0.74),
    ] {
        let points = [
            Vec3::new(-0.42 * scale, 0.0, z),
            Vec3::new(0.42 * scale, 0.0, z),
            Vec3::new(0.0, 0.58 * scale, z),
            Vec3::new(0.0, -0.58 * scale, z),
            Vec3::new(-0.70 * scale, 0.0, z),
        ];
        for triangle in [[0, 2, 1], [0, 1, 3], [4, 2, 0], [4, 0, 3]] {
            let ids = triangle.map(|index| m.add_point(points[index]).unwrap());
            for id in ids {
                m.set_point_color(id, color).unwrap();
            }
            m.add_triangle(ids[0], ids[1], ids[2]).unwrap();
        }
    }
    glow(m)
}

fn crystal(c: Color) -> Mesh {
    let mut m = Mesh::new();
    for z in [-0.08, 0.08] {
        let a = m.add_point((-0.18, 0.0, z)).unwrap();
        let b = m.add_point((0.18, 0.0, z)).unwrap();
        let t = m.add_point((0.0, 1.85, z)).unwrap();
        for id in [a, b, t] {
            m.set_point_color(id, c).unwrap();
        }
        m.add_triangle(a, b, t).unwrap();
    }
    glow(m)
}
fn bolt(c: Color, steps: u32) -> Mesh {
    let mut m = Mesh::new();
    let mut a = Vec3::ZERO;
    for i in 0..steps {
        let b = Vec3::new(
            if i % 2 == 0 { -0.28 } else { 0.28 },
            i as f32 * 0.48 + 0.48,
            0.0,
        );
        ribbon_segment(&mut m, a, b, 0.055, c);
        a = b;
    }
    glow(m)
}
fn bubbles(c: Color) -> Mesh {
    let mut m = Mesh::new();
    for i in 0..5 {
        let a = i as f32 * 1.7;
        let center = Vec3::new(a.sin() * 0.7, 0.35 + i as f32 * 0.31, a.cos() * 0.35);
        for j in 0..12 {
            let q = j as f32 / 12.0 * std::f32::consts::TAU;
            let r = (j + 1) as f32 / 12.0 * std::f32::consts::TAU;
            ribbon_segment(
                &mut m,
                center + Vec3::new(q.cos() * 0.16, q.sin() * 0.16, 0.0),
                center + Vec3::new(r.cos() * 0.16, r.sin() * 0.16, 0.0),
                0.018,
                c,
            );
        }
    }
    glow(m)
}
fn tendril(c: Color) -> Mesh {
    let mut m = Mesh::new();
    let mut a = Vec3::new(-1.0, 0.08, 0.0);
    for i in 1..9 {
        let x = i as f32 / 8.0 * 2.0 - 1.0;
        let b = Vec3::new(x, 0.12 + (x * 4.0).sin().abs() * 0.52, 0.0);
        ribbon_segment(&mut m, a, b, 0.055, c);
        a = b;
    }
    glow(m)
}
fn rune_ring(c: Color, teeth: u32) -> Mesh {
    let mut m = ring(c, 32);
    for i in 0..teeth {
        let a = i as f32 / teeth as f32 * std::f32::consts::TAU;
        ribbon_segment(
            &mut m,
            Vec3::new(a.cos() * 0.93, 0.07, a.sin() * 0.93),
            Vec3::new(a.cos() * 1.16, 0.07, a.sin() * 1.16),
            0.035,
            c,
        );
    }
    glow(m)
}
fn web(c: Color) -> Mesh {
    let mut m = ring(c, 36);
    for i in 0..8 {
        let a = i as f32 / 8.0 * std::f32::consts::TAU;
        ribbon_segment(
            &mut m,
            Vec3::ZERO,
            Vec3::new(a.cos(), 0.055, a.sin()),
            0.025,
            c,
        );
    }
    glow(m)
}
fn heart(c: Color) -> Mesh {
    let mut mesh = Mesh::new();
    let center = mesh.add_point((0.0, 0.02, 0.0)).unwrap();
    mesh.set_point_color(center, c).unwrap();
    let outline = [
        Vec3::new(0.0, -0.72, 0.0),
        Vec3::new(-0.62, -0.08, 0.0),
        Vec3::new(-0.66, 0.30, 0.0),
        Vec3::new(-0.42, 0.58, 0.0),
        Vec3::new(-0.12, 0.56, 0.0),
        Vec3::new(0.0, 0.34, 0.0),
        Vec3::new(0.12, 0.56, 0.0),
        Vec3::new(0.42, 0.58, 0.0),
        Vec3::new(0.66, 0.30, 0.0),
        Vec3::new(0.62, -0.08, 0.0),
    ];
    let ids: Vec<_> = outline
        .into_iter()
        .map(|point| {
            let id = mesh.add_point(point).unwrap();
            mesh.set_point_color(id, c).unwrap();
            id
        })
        .collect();
    for index in 0..ids.len() {
        mesh.add_triangle(center, ids[index], ids[(index + 1) % ids.len()])
            .unwrap();
    }
    glow(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(k: VisualKind, d: Delivery) -> EffectSpec {
        EffectSpec {
            kind: k,
            delivery: d,
            origin: Vec3::ZERO,
            target: Vec3::Z * 4.0,
            range_m: 5.0,
            radius_m: 3.0,
            angle_deg: 70.0,
            duration_s: 1.0,
            scale: 1.0,
            intensity: 1.0,
            seed: 7,
        }
    }
    #[test]
    fn rejects_invalid_cone() {
        let mut s = spec(VisualKind::Fire, Delivery::Cone);
        s.angle_deg = 0.0;
        assert!(s.validate().is_err());
    }
    #[test]
    fn lifecycle_cleanup_and_stale_handles() {
        let mut w = World::new();
        let base = w.entity_count();
        let mut v = VfxSystem::new();
        let h = v
            .spawn(&mut w, spec(VisualKind::Fire, Delivery::SingleTarget))
            .unwrap();
        assert!(w.entity_count() > base);
        v.update(&mut w, 1.1).unwrap();
        assert_eq!(v.active_count(), 0);
        assert_eq!(w.entity_count(), base);
        assert!(v.interrupt(&mut w, h).is_err());
    }
    #[test]
    fn every_recipe_is_layered_and_deterministic() {
        for k in [
            VisualKind::Fire,
            VisualKind::Frost,
            VisualKind::Lightning,
            VisualKind::Poison,
            VisualKind::Root,
            VisualKind::Hold,
            VisualKind::Snare,
            VisualKind::Charm,
        ] {
            let mut a = World::new();
            let mut b = World::new();
            let mut va = VfxSystem::new();
            let mut vb = VfxSystem::new();
            va.spawn(&mut a, spec(k, Delivery::Aoe)).unwrap();
            vb.spawn(&mut b, spec(k, Delivery::Aoe)).unwrap();
            assert_eq!(a.entity_count(), b.entity_count());
            assert!(a.entity_count() >= 3);
        }
    }
}
