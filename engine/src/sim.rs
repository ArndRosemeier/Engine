//! A fixed simulation tick, separate from the render frame, and running the
//! game with no window.
//!
//! Games that need determinism (a server that owns the truth, clients that
//! predict) step their simulation at a fixed rate, say 30 times a second,
//! while the screen draws at 144. Turn the tick on with
//! [`World::set_sim_rate`]; every [`Frame`] then says how many steps are due
//! ([`SimSteps::steps`]). Run your simulation that many times, and after each
//! step tell the world where things are now. The world draws them between the
//! last two steps, so motion is smooth at any frame rate:
//!
//! ```no_run
//! use engine::prelude::*;
//!
//! # fn main() -> EngineResult<()> {
//! let mut x = 0.0f32;
//! let mut ball = None;
//! Engine::run("tick", move |world, frame| {
//!     if frame.first {
//!         world.set_sim_rate(Some(30.0))?;
//!         ball = Some(world.spawn(Mesh::ball((0.0, 1.0, 0.0), 0.5, rgb(200, 60, 40))?));
//!     }
//!     for _ in 0..frame.sim.steps {
//!         world.begin_sim_step();
//!         x += 3.0 * frame.sim.dt; // your simulation
//!         world.set_sim_place(ball.unwrap(), Place::new(x.sin() * 4.0, 1.0, 0.0))?;
//!     }
//!     Ok(())
//! })
//! # }
//! ```
//!
//! A client that mirrors state it already has twice (previous and current
//! tick) can skip `begin_sim_step` and hand both over with
//! [`World::set_sim_poses`].
//!
//! [`Headless`] runs the same update closure with no window and no GPU, for a
//! dedicated server or a test.
//!
//! [`World::set_sim_rate`]: crate::world::World::set_sim_rate
//! [`World::set_sim_poses`]: crate::world::World::set_sim_poses
//! [`Frame`]: crate::world::Frame

use crate::error::{EngineError, EngineResult};
use crate::world::{Frame, World};
use glam::{Mat4, Quat};
use std::time::{Duration, Instant};

/// Simulation steps due this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimSteps {
    /// Steps to run now (0 on most frames when drawing faster than the tick).
    pub steps: u32,
    /// Length of one step in seconds (1 / rate). 0 when the tick is off.
    pub dt: f32,
    /// How far drawing is between the last two steps, 0..1.
    pub alpha: f32,
    /// Number of the first step to run this frame (counting from 0); the
    /// steps are `tick .. tick + steps`.
    pub tick: u64,
}

impl Default for SimSteps {
    fn default() -> Self {
        Self {
            steps: 0,
            dt: 0.0,
            alpha: 1.0,
            tick: 0,
        }
    }
}

/// A fixed-rate clock: feed it real time, it says how many steps are due.
/// [`World`] keeps one when [`World::set_sim_rate`] is on; a server loop can
/// use its own.
#[derive(Clone, Debug)]
pub struct SimClock {
    step: f32,
    max_steps: u32,
    accumulator: f32,
    tick: u64,
}

impl SimClock {
    /// `hz` steps per second (1..=1000). At most 8 steps run in one frame; a
    /// longer stall slows the simulation down rather than freezing the game
    /// catching up.
    pub fn new(hz: f32) -> EngineResult<Self> {
        if !hz.is_finite() || !(1.0..=1000.0).contains(&hz) {
            return Err(EngineError::InvalidValue(format!(
                "simulation rate must be in 1..=1000 Hz, got {hz}"
            )));
        }
        Ok(Self {
            step: 1.0 / hz,
            max_steps: 8,
            accumulator: 0.0,
            tick: 0,
        })
    }

    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    pub fn rate(&self) -> f32 {
        1.0 / self.step
    }

    pub fn step_seconds(&self) -> f32 {
        self.step
    }

    /// Steps run so far.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Add `dt` seconds of real time.
    pub fn advance(&mut self, dt: f32) -> SimSteps {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.accumulator += dt;
        // A hair of slack so 1/30 s frames at 30 Hz don't alternate 0 and 2.
        let due = ((self.accumulator + self.step * 1e-4) / self.step).floor() as u32;
        let steps = due.min(self.max_steps);
        self.accumulator = (self.accumulator - steps as f32 * self.step).max(0.0);
        if due > self.max_steps {
            self.accumulator = self.accumulator.min(self.step);
        }
        let first = self.tick;
        self.tick += steps as u64;
        SimSteps {
            steps,
            dt: self.step,
            alpha: (self.accumulator / self.step).clamp(0.0, 1.0),
            tick: first,
        }
    }
}

/// An entity's pose at the last two simulation steps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimPose {
    pub prev: Mat4,
    pub cur: Mat4,
    /// The world's step counter when `cur` was set.
    pub tick: u64,
}

impl SimPose {
    /// The pose to draw: between the two steps when `cur` belongs to the
    /// latest step, otherwise resting at `cur`.
    pub fn at(&self, tick: u64, alpha: f32) -> Mat4 {
        if self.tick != tick || alpha >= 1.0 {
            return self.cur;
        }
        blend(self.prev, self.cur, alpha)
    }
}

/// Position and scale lerp, rotation slerp.
pub(crate) fn blend(a: Mat4, b: Mat4, t: f32) -> Mat4 {
    let (sa, ra, ta) = a.to_scale_rotation_translation();
    let (sb, rb, tb) = b.to_scale_rotation_translation();
    let r: Quat = ra.normalize().slerp(rb.normalize(), t);
    Mat4::from_scale_rotation_translation(sa.lerp(sb, t), r, ta.lerp(tb, t))
}

/// Settings for [`Headless::run`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadlessRun {
    /// Update calls per second.
    pub frames_per_second: f32,
    /// Wait for real time between frames (a server). Off runs as fast as
    /// possible with a steady `dt` (tests, tools).
    pub real_time: bool,
    /// Stop after this many frames (None: until `world.request_exit()`).
    pub max_frames: Option<u64>,
}

impl Default for HeadlessRun {
    fn default() -> Self {
        Self {
            frames_per_second: 60.0,
            real_time: true,
            max_frames: None,
        }
    }
}

/// The game loop with no window and no GPU: a dedicated server, a bot, a
/// test. It calls the same update closure as [`Engine::run`] and ticks
/// animation, physics, the hierarchy and the fixed simulation step; nothing
/// is drawn and [`Frame::input`] stays empty.
///
/// ```
/// use engine::sim::Headless;
///
/// let mut server = Headless::new();
/// server.world().set_sim_rate(Some(30.0)).unwrap();
/// let mut steps = 0;
/// for _ in 0..60 {
///     server.frame(1.0 / 60.0, |_world, frame| {
///         steps += frame.sim.steps;
///         Ok(())
///     }).unwrap();
/// }
/// assert_eq!(steps, 30);
/// ```
///
/// [`Engine::run`]: crate::Engine::run
pub struct Headless {
    world: World,
    time: f32,
    frames: u64,
}

impl Default for Headless {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Headless {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Headless")
            .field("time", &self.time)
            .field("frames", &self.frames)
            .finish()
    }
}

impl Headless {
    pub fn new() -> Self {
        Self::with_world(World::new())
    }

    pub fn with_world(world: World) -> Self {
        Self {
            world,
            time: 0.0,
            frames: 0,
        }
    }

    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn into_world(self) -> World {
        self.world
    }

    /// Seconds simulated so far.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// One frame of `dt` seconds: the update closure, then the world's ticks.
    pub fn frame(
        &mut self,
        dt: f32,
        mut update: impl FnMut(&mut World, &Frame) -> EngineResult<()>,
    ) -> EngineResult<()> {
        let dt = if dt.is_finite() {
            dt.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let sim = self.world.advance_sim(dt);
        let frame = Frame {
            dt,
            time: self.time,
            fps: if dt > 0.0 { 1.0 / dt } else { 0.0 },
            width: 0,
            height: 0,
            aspect: 1.0,
            first: self.frames == 0,
            input: crate::input::Input::new(),
            ui: crate::ui::UiFrame::default(),
            sim,
        };
        update(&mut self.world, &frame)?;
        self.time += dt;
        self.frames += 1;
        self.world.set_time(self.time);
        self.world.tick_headless(dt);
        Ok(())
    }

    /// Run frames until `world.request_exit()` or `max_frames`.
    pub fn run(
        &mut self,
        settings: HeadlessRun,
        mut update: impl FnMut(&mut World, &Frame) -> EngineResult<()>,
    ) -> EngineResult<()> {
        let fps = settings.frames_per_second;
        if !fps.is_finite() || !(1.0..=10_000.0).contains(&fps) {
            return Err(EngineError::InvalidValue(format!(
                "headless frames_per_second must be in 1..=10000, got {fps}"
            )));
        }
        let interval = Duration::from_secs_f64(1.0 / fps as f64);
        let mut next = Instant::now();
        let mut last = Instant::now();
        let mut done = 0u64;
        loop {
            if settings.max_frames.is_some_and(|m| done >= m) {
                return Ok(());
            }
            let dt = if settings.real_time {
                let now = Instant::now();
                let dt = (now - last).as_secs_f32();
                last = now;
                if done == 0 {
                    1.0 / fps
                } else {
                    dt
                }
            } else {
                1.0 / fps
            };
            self.frame(dt, &mut update)?;
            done += 1;
            if self.world.take_exit_requested() {
                return Ok(());
            }
            if settings.real_time {
                next += interval;
                let now = Instant::now();
                if next > now {
                    std::thread::sleep(next - now);
                } else {
                    next = now;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn a_30_hz_clock_runs_30_steps_a_second_at_any_frame_rate() {
        for fps in [24.0f32, 30.0, 60.0, 144.0, 240.0] {
            let mut c = SimClock::new(30.0).unwrap();
            let mut steps = 0;
            let frames = fps as usize * 10;
            for _ in 0..frames {
                steps += c.advance(1.0 / fps).steps;
            }
            assert!((299..=301).contains(&steps), "{fps} fps: {steps} steps");
            assert_eq!(c.tick(), steps as u64);
        }
    }

    #[test]
    fn alpha_climbs_between_steps() {
        let mut c = SimClock::new(30.0).unwrap();
        let mut last = 0.0;
        let mut climbs = 0;
        for _ in 0..48 {
            let s = c.advance(1.0 / 144.0);
            assert!((0.0..=1.0).contains(&s.alpha));
            if s.steps == 0 {
                assert!(s.alpha >= last - 1e-6);
                climbs += 1;
            }
            last = s.alpha;
        }
        assert!(climbs > 30);
    }

    #[test]
    fn a_long_stall_is_capped() {
        let mut c = SimClock::new(60.0).unwrap();
        assert_eq!(c.advance(0.5).steps, 8);
        assert!(c.advance(0.0).steps <= 1);
    }

    #[test]
    fn bad_rates_are_refused() {
        assert!(SimClock::new(0.0).is_err());
        assert!(SimClock::new(f32::NAN).is_err());
    }

    #[test]
    fn blending_rotates_the_short_way() {
        let a = Mat4::from_rotation_y(0.0);
        let b = Mat4::from_rotation_y(1.0);
        let m = blend(a, b, 0.5);
        let (_, r, _) = m.to_scale_rotation_translation();
        let (axis, angle) = r.to_axis_angle();
        assert!((angle - 0.5).abs() < 1e-4 && axis.y > 0.99);
        let p = blend(
            Mat4::from_translation(Vec3::ZERO),
            Mat4::from_translation(Vec3::X * 2.0),
            0.25,
        );
        assert!((p.w_axis.x - 0.5).abs() < 1e-5);
    }

    /// A 30 Hz simulation moving a box at constant speed, drawn at 144 Hz
    /// with no window: the drawn position advances evenly every frame.
    #[test]
    fn a_30_hz_simulation_draws_smoothly_at_144_hz_headless() {
        use crate::color::rgb;
        use crate::mesh::Mesh;
        use crate::place::Place;
        let mut run = Headless::new();
        run.world().set_sim_rate(Some(30.0)).unwrap();
        let id = run
            .world()
            .spawn(Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(200, 50, 50)).unwrap());
        let mut x = 0.0f32;
        let mut drawn = Vec::new();
        let mut total_steps = 0;
        for _ in 0..288 {
            run.frame(1.0 / 144.0, |world, frame| {
                for _ in 0..frame.sim.steps {
                    world.begin_sim_step();
                    x += 3.0 * frame.sim.dt;
                    world.set_sim_place(id, Place::new(x, 0.0, 0.0))?;
                }
                total_steps += frame.sim.steps;
                Ok(())
            })
            .unwrap();
            drawn.push(run.world().entity(id).unwrap().transform().w_axis.x);
        }
        assert!((59..=61).contains(&total_steps), "{total_steps}");
        // After the first steps, every frame moves about 3 m/s / 144.
        let per_frame = 3.0 / 144.0;
        for w in drawn[20..].windows(2) {
            let d = w[1] - w[0];
            assert!(
                (d - per_frame).abs() < per_frame * 0.35,
                "uneven frame step {d} (want {per_frame})"
            );
        }
    }

    #[test]
    fn an_entity_not_moved_in_a_step_rests_and_explicit_poses_blend() {
        use crate::color::rgb;
        use crate::mesh::Mesh;
        use crate::place::Place;
        let mut world = World::new();
        let id = world.spawn(Mesh::box_at(Vec3::ZERO, Vec3::ONE, rgb(200, 50, 50)).unwrap());
        world.begin_sim_step();
        world.set_sim_place(id, Place::new(1.0, 0.0, 0.0)).unwrap();
        world.begin_sim_step();
        world.set_sim_place(id, Place::new(2.0, 0.0, 0.0)).unwrap();
        world.set_sim_alpha(0.5);
        world.tick_interpolation();
        assert!((world.entity(id).unwrap().transform().w_axis.x - 1.5).abs() < 1e-5);
        world.begin_sim_step(); // not moved this step
        world.tick_interpolation();
        assert!((world.entity(id).unwrap().transform().w_axis.x - 2.0).abs() < 1e-5);
        world
            .set_sim_poses(
                id,
                Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0)),
                Mat4::from_translation(Vec3::new(8.0, 0.0, 0.0)),
            )
            .unwrap();
        world.set_sim_alpha(0.25);
        world.tick_interpolation();
        assert!((world.entity(id).unwrap().transform().w_axis.x - 5.0).abs() < 1e-5);
        world.despawn(id);
        world.tick_interpolation();
    }

    #[test]
    fn headless_run_stops_on_request_exit() {
        let mut run = Headless::new();
        let mut frames = 0;
        run.run(
            HeadlessRun {
                frames_per_second: 120.0,
                real_time: false,
                max_frames: Some(1000),
            },
            |world, _| {
                frames += 1;
                if frames == 10 {
                    world.request_exit();
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(frames, 10);
    }
}
