//! Flying pieces from [`crate::world::World::shatter`].
//!
//! A deliberately small rigid-body model: each piece falls, spins, and
//! bounces off the ground (voxel terrain where there is some, otherwise the
//! height terrain), then slides to rest and goes to sleep. Pieces do not
//! collide with each other; for scattered rubble that reads fine and keeps
//! the cost to a handful of samples per piece per frame.

use crate::error::{EngineError, EngineResult};
use crate::fracture::FractureSettings;
use crate::world::EntityId;
use glam::{Mat4, Quat, Vec3};

/// How an object breaks and how hard its pieces fly.
#[derive(Clone, Copy, Debug)]
pub struct Shatter {
    pub pieces: usize,
    /// Outward speed of the pieces nearest the impact, in m/s.
    pub speed: f32,
    /// Seconds the rubble stays before it sinks away; `None` keeps it.
    pub keep_seconds: Option<f32>,
    pub seed: u32,
    /// Colour multiplier for the broken inside faces.
    pub interior_shade: f32,
    /// Density relative to water: stone (2.4) sinks, wood (0.6) floats.
    pub density: f32,
}

impl Default for Shatter {
    fn default() -> Self {
        Self {
            pieces: 14,
            speed: 7.0,
            keep_seconds: Some(25.0),
            seed: 1,
            interior_shade: 0.6,
            density: 2.4,
        }
    }
}

impl Shatter {
    pub fn validate(&self) -> EngineResult<()> {
        if self.pieces == 0 || self.pieces > 64 {
            return Err(EngineError::InvalidValue(
                "shatter pieces must be 1..=64".into(),
            ));
        }
        if !self.speed.is_finite() || self.speed < 0.0 {
            return Err(EngineError::InvalidValue(
                "shatter speed must be finite and >= 0".into(),
            ));
        }
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(EngineError::InvalidValue(
                "shatter density must be finite and > 0".into(),
            ));
        }
        if self.keep_seconds.is_some_and(|s| !s.is_finite() || s < 0.0) {
            return Err(EngineError::InvalidValue(
                "shatter keep_seconds must be finite and >= 0".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn fracture_settings(&self) -> FractureSettings {
        FractureSettings {
            pieces: self.pieces,
            seed: self.seed,
            interior_shade: self.interior_shade,
        }
    }
}

const GRAVITY: f32 = 9.81;
const RESTITUTION: f32 = 0.25;
const FRICTION: f32 = 0.45;
const SLEEP_SPEED: f32 = 0.25;
const SINK_SECONDS: f32 = 2.0;

#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub id: EntityId,
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub spin: Vec3,
    /// Local points tested against the ground (hull corners of the piece).
    pub support: Vec<Vec3>,
    pub age: f32,
    pub keep: Option<f32>,
    pub still_for: f32,
    pub asleep: bool,
    /// Relative to water (see [`Shatter::density`]).
    pub density: f32,
    /// Was in water last step (for splashes).
    pub wet: bool,
}

impl Piece {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }
}

/// Up to `n` local points spread over a piece's surface: the extreme vertex
/// along each of a fixed set of directions.
pub(crate) fn support_points(positions: &[Vec3]) -> Vec<Vec3> {
    let dirs = [
        Vec3::X,
        -Vec3::X,
        Vec3::Y,
        -Vec3::Y,
        Vec3::Z,
        -Vec3::Z,
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
    ];
    let mut out: Vec<Vec3> = Vec::with_capacity(dirs.len());
    for d in dirs {
        if let Some(&p) = positions
            .iter()
            .max_by(|a, b| a.dot(d).total_cmp(&b.dot(d)))
        {
            if !out.iter().any(|q| q.distance_squared(p) < 1e-8) {
                out.push(p);
            }
        }
    }
    out
}

/// Ground contact: how deep `p` is below the ground and the ground's normal.
pub(crate) trait Ground {
    fn depth(&self, p: Vec3) -> Option<(f32, Vec3)>;

    /// Water surface height and current (X, Z) at `p`, if there is water.
    fn water(&self, _p: Vec3) -> Option<(f32, glam::Vec2)> {
        None
    }
}

/// A piece hitting the water: where, and how fast it was going down.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Splash {
    pub at: Vec3,
    pub speed: f32,
}

/// Advance every awake piece by `dt`. Returns whether any moved, the pieces
/// whose time is up, and splashes into water.
pub(crate) fn step(
    pieces: &mut [Piece],
    dt: f32,
    ground: &impl Ground,
) -> (bool, Vec<EntityId>, Vec<Splash>) {
    let mut moved = false;
    let mut expired = Vec::new();
    let mut splashes = Vec::new();
    let steps = (dt / (1.0 / 120.0)).ceil().clamp(1.0, 8.0) as usize;
    let h = dt.min(0.1) / steps as f32;
    for piece in pieces.iter_mut() {
        piece.age += dt;
        if let Some(keep) = piece.keep {
            if piece.age > keep + SINK_SECONDS {
                expired.push(piece.id);
                continue;
            }
            if piece.age > keep {
                // Sink into the ground before vanishing.
                piece.asleep = true;
                piece.position.y -= dt * 0.6;
                moved = true;
                continue;
            }
        }
        if piece.asleep {
            continue;
        }
        moved = true;
        let mut floating = false;
        for _ in 0..steps {
            piece.velocity.y -= GRAVITY * h;
            // In water: buoyancy by the share of the piece under the
            // surface, drag toward the current, damped spin.
            if let Some((surface, current)) = ground.water(piece.position) {
                let n = piece.support.len().max(1) as f32;
                let under = piece
                    .support
                    .iter()
                    .filter(|&&l| (piece.position + piece.rotation * l).y < surface)
                    .count() as f32;
                let centre = f32::from(piece.position.y < surface);
                let frac = (under + centre * 2.0) / (n + 2.0);
                if frac > 0.0 {
                    if !piece.wet && piece.velocity.y < -1.5 {
                        splashes.push(Splash {
                            at: piece.position,
                            speed: -piece.velocity.y,
                        });
                    }
                    piece.wet = true;
                    piece.velocity.y += GRAVITY * frac / piece.density * h;
                    let rel = piece.velocity - Vec3::new(current.x, 0.0, current.y);
                    piece.velocity -= rel * (1.0 - (-3.0 * frac * h).exp());
                    piece.spin *= (-2.0 * frac * h).exp();
                    floating |= piece.density < 1.0;
                } else {
                    piece.wet = false;
                }
            }
            piece.position += piece.velocity * h;
            let angle = piece.spin.length() * h;
            if angle > 0.0 {
                piece.rotation = (Quat::from_axis_angle(piece.spin.normalize(), angle)
                    * piece.rotation)
                    .normalize();
            }
            // Deepest support point below ground pushes the piece out.
            let mut contact: Option<(f32, Vec3, Vec3)> = None;
            for &local in &piece.support {
                let p = piece.position + piece.rotation * local;
                if let Some((depth, normal)) = ground.depth(p) {
                    if depth > 0.0 && contact.is_none_or(|c| depth > c.0) {
                        contact = Some((depth, normal, p));
                    }
                }
            }
            if let Some((depth, normal, at)) = contact {
                piece.position += normal * depth;
                let vn = piece.velocity.dot(normal);
                if vn < 0.0 {
                    piece.velocity -= normal * vn * (1.0 + RESTITUTION);
                    // Friction and a spin kick from hitting off-centre.
                    let tangential = piece.velocity - normal * piece.velocity.dot(normal);
                    piece.velocity -= tangential * FRICTION;
                    let lever = at - piece.position;
                    piece.spin = piece.spin * 0.6 + lever.cross(normal * -vn) * 0.25;
                }
                piece.spin *= 0.97;
            }
        }
        // Floating pieces keep drifting with the water.
        let resting =
            !floating && piece.velocity.length() < SLEEP_SPEED && piece.spin.length() < 0.6;
        piece.still_for = if resting { piece.still_for + dt } else { 0.0 };
        if piece.still_for > 0.4 || piece.position.y < -1000.0 {
            piece.asleep = true;
        }
    }
    (moved, expired, splashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flat;
    impl Ground for Flat {
        fn depth(&self, p: Vec3) -> Option<(f32, Vec3)> {
            Some((-p.y, Vec3::Y))
        }
    }

    #[test]
    fn pieces_fall_bounce_and_settle_on_the_ground() {
        let cube: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32) - 0.5)
            .collect();
        let mut pieces = vec![Piece {
            id: EntityId::test(1),
            position: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            velocity: Vec3::new(3.0, 2.0, 0.0),
            spin: Vec3::new(0.0, 0.0, 4.0),
            support: support_points(&cube),
            age: 0.0,
            keep: None,
            still_for: 0.0,
            asleep: false,
            density: 2.4,
            wet: false,
        }];
        for _ in 0..600 {
            step(&mut pieces, 1.0 / 60.0, &Flat);
        }
        let p = &pieces[0];
        assert!(p.asleep, "piece never came to rest: v={:?}", p.velocity);
        let lowest = p
            .support
            .iter()
            .map(|&l| (p.position + p.rotation * l).y)
            .fold(f32::MAX, f32::min);
        assert!(lowest > -0.05 && lowest < 0.05, "resting height {lowest}");
        assert!(
            p.position.x > 1.0,
            "slid forward a little: {}",
            p.position.x
        );
    }
}
