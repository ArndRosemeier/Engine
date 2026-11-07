//! A first-person body that walks, wades, swims and dives.
//!
//! ```no_run
//! use engine::prelude::*;
//! use engine::water::Swimmer;
//!
//! fn main() -> EngineResult<()> {
//!     let mut body = Swimmer::new((10.0, 20.0, 10.0), 0.0);
//!     Engine::run("swim", move |world, frame| {
//!         // ... set up a voxel terrain and water on the first frame ...
//!         body.update(world, frame);
//!         Ok(())
//!     })
//! }
//! ```
//!
//! On land it walks and jumps. Wading slows it by how deep the water is. Once
//! the water is chest deep it floats: it treads water with the eyes just
//! above the surface, swims where the camera looks, dives when looking down
//! (or with Ctrl / Q), and rises with Space / E. Currents carry it. Under the
//! surface, and in flooded caves with no air, it runs out of breath.

use super::sim::Water;
use super::terrain::WaterTerrain;
use crate::camera::Camera;
use crate::input::{Input, Key, MouseButton};
use crate::world::{Frame, World};
use glam::{Vec2, Vec3};

const GRAVITY: f32 = 9.81;

/// What the body is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwimState {
    OnGround,
    InAir,
    /// Standing in water up to the waist.
    Wading,
    /// Floating at the surface.
    Swimming,
    /// Head under water.
    Diving,
}

/// Tunables of a [`Swimmer`]. Metres, seconds, metres per second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwimmerSettings {
    pub height: f32,
    pub radius: f32,
    pub eye_height: f32,
    pub walk_speed: f32,
    pub sprint: f32,
    pub jump_speed: f32,
    pub swim_speed: f32,
    /// Seconds of air.
    pub breath: f32,
    /// Highest ledge the body steps or climbs onto (out of water too).
    pub step_height: f32,
}

impl Default for SwimmerSettings {
    fn default() -> Self {
        Self {
            height: 1.8,
            radius: 0.35,
            eye_height: 1.65,
            walk_speed: 4.5,
            sprint: 1.8,
            jump_speed: 5.0,
            swim_speed: 2.2,
            breath: 45.0,
            step_height: 0.55,
        }
    }
}

/// Movement wishes for one step, in the body's own terms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwimInput {
    /// -1 back .. 1 forward.
    pub forward: f32,
    /// -1 left .. 1 right.
    pub strafe: f32,
    /// -1 dive .. 1 rise (in water); jump on land when positive.
    pub up: f32,
    pub sprint: bool,
}

impl SwimInput {
    /// WASD / arrows, Space or E up, Ctrl or Q down, Shift sprint.
    pub fn from_keys(input: &Input) -> Self {
        let mv = input.move_xz();
        Self {
            forward: mv.y,
            strafe: mv.x,
            up: f32::from(input.down(Key::Space) || input.down(Key::E))
                - f32::from(input.down(Key::Ctrl) || input.down(Key::Q)),
            sprint: input.down(Key::Shift),
        }
    }
}

/// Things that happened during a step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwimEvents {
    /// Hit the water at this downward speed (m/s): make a splash.
    pub entered_water: Option<f32>,
    /// The head went under this step.
    pub submerged: bool,
    /// The head came up this step.
    pub surfaced: bool,
}

/// A first-person body for walking, swimming and diving. See the module docs.
#[derive(Clone, Debug)]
pub struct Swimmer {
    /// Bottom of the feet.
    pub feet: Vec3,
    pub velocity: Vec3,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub look_sensitivity: f32,
    pub settings: SwimmerSettings,
    breath_left: f32,
    state: SwimState,
    head_under: bool,
    in_water: bool,
}

impl Swimmer {
    /// A body standing at `feet`, facing `yaw_deg`.
    pub fn new(feet: impl Into<Vec3>, yaw_deg: f32) -> Self {
        let settings = SwimmerSettings::default();
        Self {
            feet: feet.into(),
            velocity: Vec3::ZERO,
            yaw_deg,
            pitch_deg: 0.0,
            look_sensitivity: 0.15,
            breath_left: settings.breath,
            settings,
            state: SwimState::InAir,
            head_under: false,
            in_water: false,
        }
    }

    pub fn state(&self) -> SwimState {
        self.state
    }

    /// Seconds of air left.
    pub fn breath(&self) -> f32 {
        self.breath_left
    }

    /// Air left as a fraction, 1 = full lungs.
    pub fn breath_fraction(&self) -> f32 {
        self.breath_left / self.settings.breath.max(1e-3)
    }

    pub fn out_of_breath(&self) -> bool {
        self.breath_left <= 0.0
    }

    pub fn eye(&self) -> Vec3 {
        self.feet + Vec3::Y * self.settings.eye_height
    }

    /// Look with the mouse, move with the keys against the world's voxel
    /// terrain (or height terrain) and water, splash into the water, and put
    /// the camera at the eyes.
    pub fn update(&mut self, world: &mut World, frame: &Frame) -> SwimEvents {
        let input = &frame.input;
        if input.mouse_down(MouseButton::Right) || world.pointer_lock() {
            let d = input.mouse_delta();
            self.yaw_deg = (self.yaw_deg + d.x * self.look_sensitivity).rem_euclid(360.0);
            self.pitch_deg = (self.pitch_deg - d.y * self.look_sensitivity).clamp(-89.0, 89.0);
        }
        let wish = SwimInput::from_keys(input);
        let events = {
            let ground = WorldGround(world);
            self.step(&ground, world.water(), wish, frame.dt)
        };
        if let (Some(speed), Some(water)) = (events.entered_water, world.water_mut()) {
            // A body landing in water pushes it aside: a splash ring.
            water.disturb(self.feet + Vec3::Y * 0.3, 1.2, (speed * 0.35).min(4.0));
        }
        world.look_first_person(self.eye(), self.yaw_deg, self.pitch_deg);
        events
    }

    /// Advance by `dt` seconds against `ground` and `water`.
    pub fn step(
        &mut self,
        ground: &dyn WaterTerrain,
        water: Option<&Water>,
        input: SwimInput,
        dt: f32,
    ) -> SwimEvents {
        let dt = dt.clamp(0.0, 0.05);
        let mut events = SwimEvents::default();
        if dt <= 0.0 {
            return events;
        }
        let s = self.settings;
        let centre = self.feet + Vec3::Y * (s.height * 0.5);
        let sample = water.and_then(|w| {
            w.sample(centre)
                .or_else(|| w.sample(self.feet + Vec3::Y * 0.1))
        });
        // How much of the body is under water, 0..1.
        let (surface, capped, current) = match sample {
            Some(w) => (w.surface_y, w.capped, w.current),
            None => (f32::NEG_INFINITY, false, Vec2::ZERO),
        };
        let submersion = if capped {
            1.0
        } else {
            ((surface - self.feet.y) / s.height).clamp(0.0, 1.0)
        };
        let eye = self.eye();
        let head_under = capped || eye.y < surface;
        events.submerged = head_under && !self.head_under;
        events.surfaced = !head_under && self.head_under;
        self.head_under = head_under;
        let was_in_water = self.in_water;
        self.in_water = submersion > 0.05;
        if self.in_water && !was_in_water && self.velocity.y < -2.0 {
            events.entered_water = Some(-self.velocity.y);
        }

        // Breath: lungs empty under water and refill quickly in air.
        if head_under {
            self.breath_left = (self.breath_left - dt).max(0.0);
        } else {
            self.breath_left = (self.breath_left + dt * 6.0).min(s.breath);
        }

        let grounded = self.on_ground(ground);
        let forward_flat = Camera::direction(self.yaw_deg, 0.0);
        let right = Camera::right_xz(self.yaw_deg);
        let floating = submersion > 0.6 && !(grounded && submersion < 0.75);

        if floating {
            // Swim where the camera looks; at the surface only dive when
            // looking well down, so a glance at the shore does not sink you.
            let look = Camera::direction(self.yaw_deg, self.pitch_deg);
            let mut forward = if head_under || self.pitch_deg < -30.0 {
                look
            } else {
                forward_flat
            };
            if !head_under && forward.y > 0.0 {
                forward.y = 0.0;
            }
            let speed = s.swim_speed * if input.sprint { 1.6 } else { 1.0 };
            let mut wish =
                (forward * input.forward + right * input.strafe).clamp_length_max(1.0) * speed;
            wish.y += input.up * speed * 0.8;
            // Tread water: float with the eyes a little above the surface.
            let float_feet = surface - s.eye_height + 0.12;
            let treading =
                !head_under || (input.up >= 0.0 && wish.y.abs() < 0.1 && surface - eye.y < 0.6);
            if treading && wish.y.abs() < 0.3 && !capped {
                wish.y += ((float_feet - self.feet.y) * 2.5).clamp(-1.0, 1.0);
            } else if head_under && input.up == 0.0 && input.forward == 0.0 {
                // Slightly buoyant when idle under water.
                wish.y += 0.15;
            }
            // Water drag pulls velocity toward the wish, carried by the current.
            let target = wish + Vec3::new(current.x, 0.0, current.y);
            let k = 1.0 - (-3.5 * dt).exp();
            self.velocity += (target - self.velocity) * k;
            self.state = if head_under {
                SwimState::Diving
            } else {
                SwimState::Swimming
            };
        } else {
            // On foot. Water slows walking by how deep it is.
            let drag = 1.0 - submersion * 0.7;
            let speed = s.walk_speed * drag * if input.sprint { s.sprint } else { 1.0 };
            let wish =
                (forward_flat * input.forward + right * input.strafe).clamp_length_max(1.0) * speed;
            let control = if grounded { 12.0 } else { 2.0 };
            let k = 1.0 - (-control * dt).exp();
            let horizontal = Vec3::new(self.velocity.x, 0.0, self.velocity.z);
            let new_h = horizontal
                + (wish - horizontal) * k
                + Vec3::new(current.x, 0.0, current.y) * submersion * k;
            self.velocity.x = new_h.x;
            self.velocity.z = new_h.z;
            // Gravity, eased by buoyancy in water.
            self.velocity.y -= GRAVITY * (1.0 - submersion * 0.9) * dt;
            if submersion > 0.0 {
                self.velocity.y *= (-2.0 * submersion * dt).exp();
            }
            if grounded && input.up > 0.0 && self.velocity.y <= 0.5 {
                self.velocity.y = s.jump_speed * (1.0 - submersion * 0.5);
            }
            self.state = if submersion > 0.15 {
                SwimState::Wading
            } else if grounded {
                SwimState::OnGround
            } else {
                SwimState::InAir
            };
        }

        self.move_and_collide(ground, self.velocity * dt, floating);
        events
    }

    /// Sphere centres of the body, feet to head.
    fn spheres(&self) -> [Vec3; 3] {
        let r = self.settings.radius;
        let h = self.settings.height;
        [
            self.feet + Vec3::Y * r,
            self.feet + Vec3::Y * (h * 0.5),
            self.feet + Vec3::Y * (h - r),
        ]
    }

    fn on_ground(&self, ground: &dyn WaterTerrain) -> bool {
        let probe = self.feet + Vec3::Y * self.settings.radius;
        signed_distance(ground, probe - Vec3::Y * 0.08) < self.settings.radius
    }

    /// Move by `delta` in small pieces, pushing the body spheres out of the
    /// ground and sliding along it. Small ledges are stepped onto.
    fn move_and_collide(&mut self, ground: &dyn WaterTerrain, delta: Vec3, floating: bool) {
        let r = self.settings.radius;
        let pieces = ((delta.length() / (r * 0.5)).ceil() as usize).clamp(1, 16);
        let step = delta / pieces as f32;
        for _ in 0..pieces {
            self.feet += step;
            for _ in 0..3 {
                let mut pushed = false;
                for (i, c) in self.spheres().into_iter().enumerate() {
                    let d = signed_distance(ground, c);
                    if d < r {
                        let n = ground_normal(ground, c);
                        let depth = r - d;
                        // A low obstacle under the feet is a step: lift
                        // straight up instead of sliding back.
                        let push = if i == 0 && n.y < 0.7 && depth < self.settings.step_height {
                            Vec3::Y * depth
                        } else {
                            n * depth
                        };
                        self.feet += push;
                        let into = self.velocity.dot(n);
                        if into < 0.0 {
                            self.velocity -= n * into;
                        }
                        pushed = true;
                    }
                }
                if !pushed {
                    break;
                }
            }
        }
        // Swimming into a bank: climb out when the ledge is low enough.
        if floating {
            let ahead = Vec3::new(self.velocity.x, 0.0, self.velocity.z).normalize_or_zero();
            if ahead != Vec3::ZERO {
                let knee = self.feet + Vec3::Y * r + ahead * (r + 0.15);
                let lip =
                    self.feet + Vec3::Y * (self.settings.step_height + r) + ahead * (r + 0.15);
                if signed_distance(ground, knee) < r && signed_distance(ground, lip) > r {
                    self.velocity.y = self.velocity.y.max(2.0);
                }
            }
        }
    }
}

/// Approximate distance to rock (negative inside) from the terrain density.
fn signed_distance(ground: &dyn WaterTerrain, p: Vec3) -> f32 {
    -ground.density(p)
}

fn ground_normal(ground: &dyn WaterTerrain, p: Vec3) -> Vec3 {
    let e = 0.25;
    let g = Vec3::new(
        ground.density(p - Vec3::X * e) - ground.density(p + Vec3::X * e),
        ground.density(p - Vec3::Y * e) - ground.density(p + Vec3::Y * e),
        ground.density(p - Vec3::Z * e) - ground.density(p + Vec3::Z * e),
    );
    g.try_normalize().unwrap_or(Vec3::Y)
}

/// The world's solid ground: voxel terrain where set, else the height terrain.
struct WorldGround<'a>(&'a World);

impl WaterTerrain for WorldGround<'_> {
    fn density(&self, p: Vec3) -> f32 {
        if let Some(t) = self.0.voxel_terrain() {
            return t.volume().sample(p) * t.volume().voxel_size();
        }
        match self.0.height_field.as_ref() {
            Some(f) => f.height_at(p.x, p.z) - p.y,
            None => -p.y,
        }
    }
}
