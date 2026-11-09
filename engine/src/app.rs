use crate::error::{EngineError, EngineResult};
use crate::input::Input;
use crate::limits::EngineLimits;
use crate::render::GpuFrameStats;
use crate::render::Renderer;
use crate::ui_backend::{UiBackend, UiPaintTarget, UiViewport};
use crate::world::{Frame, HitchSpan, World};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};

type UpdateFn = Box<dyn FnMut(&mut World, &Frame) -> EngineResult<()>>;

/// Run `$e`, adding its time to the CPU trace (when on) under `$name`.
macro_rules! timed {
    ($app:ident, $name:expr, $e:expr) => {{
        let t = Instant::now();
        $e;
        if let Some(trace) = $app.cpu_trace.as_mut() {
            trace.add($name, elapsed_ms(t));
        }
    }};
}

struct App {
    title: String,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    ui_backend: Option<UiBackend>,
    world: World,
    update: UpdateFn,
    input: Input,
    start: Instant,
    last: Instant,
    frame_index: u32,
    first_update_done: bool,
    screenshot_path: Option<PathBuf>,
    screenshot_frame: u32,
    screenshot_wait: bool,
    fps: f32,
    fps_accum_s: f32,
    fps_frames: u32,
    pointer_locked: bool,
    hitch_ms: f32,
    cpu_trace: Option<crate::cpu_trace::CpuTrace>,
    clock: crate::frame_clock::FrameClock,
    frame_stats: Option<crate::frame_clock::FrameStats>,
    pacer: crate::frame_clock::FramePacer,
    refresh_hz: Option<f32>,
    /// ENGINE_PRESENT_MODE or a screenshot run: never pace.
    no_pacing: bool,
    fatal_error: Option<String>,
    /// The sound device; `None` when there is none (the game runs silent).
    _audio_out: Option<crate::audio::scene::AudioOutput>,
}

impl App {
    fn new(title: String, limits: EngineLimits, update: UpdateFn) -> EngineResult<Self> {
        let now = Instant::now();
        let screenshot_path = optional_path_env("ENGINE_SCREENSHOT")?;
        let screenshot_frame = optional_parsed_env("ENGINE_SCREENSHOT_FRAME")?.unwrap_or(3);
        let screenshot_wait = optional_bool_env("ENGINE_SCREENSHOT_WAIT")?.unwrap_or(false);
        Ok(Self {
            title,
            window: None,
            renderer: None,
            ui_backend: None,
            world: {
                let mut world = World::new().with_limits(limits);
                world.watch_z_fighting_from_env();
                world
            },
            update,
            input: Input::new(),
            start: now,
            last: now,
            frame_index: 0,
            first_update_done: false,
            screenshot_path,
            screenshot_frame,
            screenshot_wait,
            fps: 0.0,
            fps_accum_s: 0.0,
            fps_frames: 0,
            pointer_locked: false,
            hitch_ms: hitch_threshold_ms()?,
            cpu_trace: crate::cpu_trace::CpuTrace::from_env(),
            clock: crate::frame_clock::FrameClock::from_env(),
            frame_stats: crate::frame_clock::FrameStats::from_env(),
            pacer: crate::frame_clock::FramePacer::new(),
            refresh_hz: None,
            no_pacing: std::env::var_os("ENGINE_PRESENT_MODE").is_some()
                || std::env::var_os("ENGINE_SCREENSHOT").is_some()
                || std::env::var("ENGINE_PACE").is_ok_and(|v| v.trim() == "0"),
            fatal_error: None,
            _audio_out: None,
        })
    }

    /// Match the window's pointer grab to what the game asked for.
    ///
    /// Windows only supports [`CursorGrabMode::Confined`], the other desktops
    /// prefer `Locked`; either is enough because look deltas come from raw
    /// device motion rather than the cursor position.
    fn apply_pointer_lock(&mut self, window: &Window, wanted: bool) {
        if wanted == self.pointer_locked {
            return;
        }
        if wanted {
            let grabbed = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            grabbed.expect("this platform supports neither locked nor confined pointer grab");
        } else {
            window
                .set_cursor_grab(CursorGrabMode::None)
                .expect("release pointer grab");
        }
        window.set_cursor_visible(!wanted);
        self.pointer_locked = wanted;
    }

    fn window_inner_size() -> EngineResult<winit::dpi::LogicalSize<u32>> {
        let w: u32 = optional_parsed_env("ENGINE_WIDTH")?.unwrap_or(1920);
        let h: u32 = optional_parsed_env("ENGINE_HEIGHT")?.unwrap_or(1080);
        if w < 320 || h < 180 {
            return Err(EngineError::InvalidValue(format!(
                "ENGINE_WIDTH and ENGINE_HEIGHT must be at least 320x180, got {w}x{h}"
            )));
        }
        Ok(winit::dpi::LogicalSize::new(w, h))
    }
}

impl App {
    /// Wait for the display's next refresh slot when vsync does not.
    fn pace_frame(&mut self) {
        if self.no_pacing {
            return;
        }
        let hz = self.refresh_hz.unwrap_or(0.0);
        if let Some(until) = self.pacer.schedule(Instant::now(), hz) {
            crate::frame_clock::wait_until(until);
        }
    }

    /// One whole frame: input, the game's update, simulation, rendering.
    fn run_frame(&mut self, event_loop: &ActiveEventLoop) {
        self.pace_frame();
        let now = Instant::now();
        let cpu_dt = (now - self.last).as_secs_f32();
        self.last = now;
        // Time frames by when the display took them: the CPU's own clock
        // bunches up when presentation makes it wait.
        let raw_dt = self
            .renderer
            .as_ref()
            .and_then(Renderer::acquire_interval)
            .filter(|d| d.is_finite() && *d > 0.0)
            .unwrap_or(cpu_dt);
        let time = (now - self.start).as_secs_f32();
        // Movement uses a steady step; the raw one only counts FPS.
        let vsynced = self.world.vsync() && !self.no_pacing;
        let dt = self.clock.tick(raw_dt, self.refresh_hz.filter(|_| vsynced));
        if let Some(stats) = self.frame_stats.as_mut() {
            stats.add(raw_dt);
        }

        self.fps_accum_s += raw_dt.max(0.0);
        self.fps_frames += 1;
        if self.fps_accum_s >= 0.5 {
            self.fps = self.fps_frames as f32 / self.fps_accum_s;
            self.fps_accum_s = 0.0;
            self.fps_frames = 0;
            self.refresh_hz = self
                .window
                .as_ref()
                .and_then(|w| w.current_monitor())
                .and_then(|m| m.refresh_rate_millihertz())
                .map(|mhz| mhz as f32 / 1000.0);
            self.pacer
                .observe(self.world.vsync(), self.fps, self.refresh_hz);
            if let Some(window) = &self.window {
                window.set_title(&format!(
                    "{} - Engine {} - {:.0} FPS",
                    self.title,
                    crate::version::version(),
                    self.fps
                ));
            }
        }

        let size = self
            .renderer
            .as_ref()
            .expect("renderer must exist before redraw")
            .size();
        let first = !self.first_update_done;
        self.first_update_done = true;

        let window = self
            .window
            .clone()
            .expect("window must exist before redraw");
        let ui_backend = self
            .ui_backend
            .as_mut()
            .expect("UI backend must exist before redraw");

        // A locked pointer means the game owns the mouse, so egui's
        // hover state must not swallow look and movement input.
        let mut input = self.input.clone();
        if !self.world.bind_listen()
            && !self.pointer_locked
            && (ui_backend.wants_keyboard_input() || ui_backend.wants_pointer_input())
        {
            input = Input::new();
        }
        if input_log() {
            eprintln!(
                "frame {:>9.3}s move {:?} ui_wants_kb={} ui_wants_pointer={}",
                time,
                input.move_xz(),
                ui_backend.wants_keyboard_input(),
                ui_backend.wants_pointer_input()
            );
        }

        let sim = if self.fatal_error.is_none() {
            self.world.advance_sim(dt)
        } else {
            Default::default()
        };
        let update = &mut self.update;
        let world = &mut self.world;
        let fps = self.fps;
        // Screenshot runs hide the button unless ENGINE_WHATS_NEW asks for it.
        let whats_new_env = std::env::var("ENGINE_WHATS_NEW").ok();
        let screenshot_run = self.screenshot_path.is_some() && whats_new_env.is_none();
        let update_t = Instant::now();
        let (modal_was_open, full_output) = {
            let (ui_result, full_output) = ui_backend.run_ui(&window, |ui| {
                ui.set_bind_listen(world.bind_listen());
                let frame = Frame {
                    dt,
                    time,
                    fps,
                    width: size.width,
                    height: size.height,
                    aspect: size.width as f32 / size.height.max(1) as f32,
                    first,
                    input: input.clone(),
                    ui: ui.clone(),
                    sim,
                };
                if let Some(message) = self.fatal_error.as_ref() {
                    egui::CentralPanel::default().show(ui.ctx(), |panel| {
                        panel.vertical_centered(|panel| {
                            panel.add_space(48.0);
                            panel.heading("Fatal application error");
                            panel.add_space(12.0);
                            panel.label(message);
                            panel.add_space(16.0);
                            panel.label("Gameplay has stopped and will not resume.");
                            panel.label("Press Escape, close the window, or choose Exit.");
                            if panel.button("Exit").clicked() {
                                world.request_exit();
                            }
                        });
                    });
                } else if !run_callback_until_fatal(&mut self.fatal_error, || update(world, &frame))
                {
                    world.set_pointer_lock(false);
                }
                if whats_new_env.as_deref() == Some("open") && first {
                    crate::version::set_whats_new_open(ui.ctx(), true);
                }
                if world.whats_new_button() && !screenshot_run && !world.pointer_lock() {
                    crate::version::whats_new_ui(ui.ctx());
                }
                ui.modal_was_open()
            });
            (ui_result, full_output)
        };
        let update_ms = elapsed_ms(update_t);
        self.input.end_frame();

        // Escape gives the mouse back before it closes the window: a
        // player whose cursor is pinned reaches for Escape to get it
        // out, not to quit.
        if ui_backend.take_escape_pressed() {
            if self.world.bind_listen() {
                self.world.set_bind_listen(false);
            } else if !modal_was_open {
                if self.world.pointer_lock() {
                    self.world.set_pointer_lock(false);
                } else {
                    event_loop.exit();
                    return;
                }
            }
        }

        let wants_lock = self.world.pointer_lock();
        self.apply_pointer_lock(&window, wants_lock);

        let anim_t = Instant::now();
        if self.fatal_error.is_none() {
            self.world.set_time(time);
            timed!(self, "tick.stream", self.world.tick_stream());
            timed!(self, "tick.nav", self.world.tick_nav(dt));
            timed!(self, "tick.crowds", self.world.tick_crowds(dt));
            timed!(self, "tick.animations", self.world.tick_animations(dt));
            timed!(self, "tick.physics", self.world.tick_physics(dt));
            timed!(self, "tick.ragdolls", self.world.tick_ragdolls(dt));
            timed!(self, "tick.buildings", self.world.tick_buildings(dt));
            timed!(self, "tick.interpolation", self.world.tick_interpolation());
            timed!(self, "tick.hierarchy", self.world.tick_hierarchy());
            timed!(self, "tick.effects", self.world.tick_effects(dt));
            timed!(self, "tick.water", self.world.tick_water(dt));
            timed!(self, "tick.auto_quality", self.world.tick_auto_quality(dt));
            timed!(self, "tick.lods", self.world.tick_lods());
            timed!(self, "tick.sky", self.world.tick_sky());
            timed!(self, "tick.grading", self.world.tick_grading(dt));
            timed!(self, "tick.zfight_watch", self.world.tick_zfight_watch(dt));
            timed!(self, "tick.audio", self.world.tick_audio(dt));
        }
        let anim_ms = elapsed_ms(anim_t);

        {
            let renderer = self
                .renderer
                .as_mut()
                .expect("renderer must exist before redraw");
            renderer.cpu_spans.on = self.cpu_trace.is_some();
            let sync_t = Instant::now();
            let size = renderer.size();
            self.world
                .update_room_visibility(size.width as f32 / size.height.max(1) as f32);
            renderer.sync_world(&mut self.world);
            let sync_ms = elapsed_ms(sync_t);
            let ui_backend = self.ui_backend.as_mut().expect("ui backend");
            let fixed_shot_due = !self.screenshot_wait
                && self.screenshot_path.is_some()
                && self.frame_index + 1 >= self.screenshot_frame;
            if fixed_shot_due || self.world.has_screenshot_queued() {
                renderer.request_capture();
            }
            let render_t = Instant::now();
            match renderer.render_with(&self.world, |device, queue, encoder, view| {
                ui_backend.paint(
                    &window,
                    device,
                    queue,
                    UiPaintTarget {
                        view,
                        viewport: UiViewport {
                            width: size.width,
                            height: size.height,
                        },
                        encoder,
                    },
                    full_output,
                );
            }) {
                Ok(()) => {}
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    renderer.resize(renderer.size());
                }
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    panic!("GPU out of memory");
                }
                // Timeout is transient presentation backpressure. Skip this frame, preserve
                // world state, and request the next redraw; repeated timeouts remain visible
                // through frame/hitch telemetry rather than corrupting renderer state.
                Err(wgpu::SurfaceError::Timeout) => {
                    eprintln!("surface frame timed out; retrying on next redraw");
                }
                Err(other) => panic!("surface error: {other}"),
            }
            let render_ms = elapsed_ms(render_t);
            if let Some(trace) = self.cpu_trace.as_mut() {
                trace.add("game update", update_ms);
                for (name, ms) in renderer.cpu_spans.take() {
                    trace.add(name, ms);
                }
                trace.end_frame(raw_dt * 1000.0);
            }
            let gpu = renderer.take_gpu_stats();
            let notes = self.world.take_hitch_spans();
            let phases = HitchPhases {
                update_ms,
                anim_ms,
                sync_ms,
                render_ms,
            };
            if self.frame_index < 5 && crate::render::pipeline_cache::log_enabled() {
                // The first frames are where a lazily built shader would stall.
                eprintln!(
                    "[engine] frame {}: {:.1} ms of work (sync {:.1}, render {:.1})",
                    self.frame_index,
                    phases.work_ms(),
                    phases.sync_ms,
                    phases.render_ms
                );
            }
            if phases.work_ms() >= self.hitch_ms {
                if let Some(path) = self.world.hitch_log() {
                    emit_hitch(
                        path,
                        self.frame_index,
                        fps,
                        raw_dt * 1000.0,
                        phases,
                        &notes,
                        &gpu,
                    );
                }
            }
            self.frame_index += 1;
            let queued = self.world.take_screenshot_queue();
            for path in queued {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).unwrap_or_else(|error| {
                        panic!(
                            "failed to create screenshot directory {} for {}: {error}",
                            parent.display(),
                            path.display()
                        )
                    });
                }
                renderer
                    .capture_png(&self.world, &path)
                    .unwrap_or_else(|error| panic!("{error}"));
                eprintln!("wrote screenshot {}", path.display());
            }
            if self.world.take_exit_requested() {
                event_loop.exit();
                return;
            }
            if !self.screenshot_wait {
                if let Some(path) = self.screenshot_path.clone() {
                    if self.frame_index >= self.screenshot_frame {
                        renderer
                            .capture_png(&self.world, &path)
                            .unwrap_or_else(|error| panic!("{error}"));
                        eprintln!("wrote screenshot {}", path.display());
                        event_loop.exit();
                    }
                }
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!(
                "{} - Engine {}",
                self.title,
                crate::version::version()
            ))
            .with_inner_size(Self::window_inner_size().expect("validated window size"))
            // Hidden while the renderer compiles its pipelines: this callback
            // holds the event loop, and a shown window that cannot answer
            // stands still and cannot be dragged. Shown once it can.
            .with_visible(false);
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        // Sound starts while the pipelines build: the device opens and the
        // engine's own sounds are synthesised on worker threads.
        if self.screenshot_path.is_none() {
            crate::audio::bank::warm();
            self._audio_out = crate::audio::scene::AudioOutput::start(self.world.audio());
        }
        let renderer = pollster::block_on(Renderer::new(window.clone()));
        let ui_backend = UiBackend::new(&window, renderer.device(), renderer.surface_format());
        window.set_visible(true);
        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.ui_backend = Some(ui_backend);
        self.start = Instant::now();
        self.last = self.start;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(window), Some(ui)) = (self.window.as_ref(), self.ui_backend.as_mut()) {
            let _consumed = ui.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Quit is decided after egui runs (Escape closes modals first).
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if input_log() {
                    eprintln!(
                        "input {:>9.3}s key {:?} {} repeat={}",
                        self.start.elapsed().as_secs_f64(),
                        event.physical_key,
                        if pressed { "down" } else { "up" },
                        event.repeat
                    );
                }
                if !event.repeat {
                    self.input.set_key(event.physical_key, pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input
                    .set_cursor(Some((position.x as f32, position.y as f32)));
            }
            WindowEvent::CursorLeft { .. } => self.input.set_cursor(None),
            WindowEvent::MouseInput { state, button, .. } => {
                self.input
                    .set_mouse_button(button, state == ElementState::Pressed);
            }
            // Whatever the game wanted, a window that is not in front has no
            // business holding the desktop's pointer.
            WindowEvent::Focused(false) => {
                self.world.set_pointer_lock(false);
                if let Some(window) = self.window.clone() {
                    self.apply_pointer_lock(&window, false);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size);
                }
            }
            // Frames run from `about_to_wait`: on Windows a burst of input
            // (a held key's auto-repeat) starves redraw requests, and the
            // view would only move at the key-repeat rate.
            WindowEvent::RedrawRequested if !self.first_update_done => {
                self.run_frame(event_loop);
            }
            _ => {}
        }
    }

    /// Raw pointer motion, which keeps working once the cursor is pinned.
    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.input.add_mouse_delta(dx as f32, dy as f32);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() && self.renderer.is_some() && !event_loop.exiting() {
            self.run_frame(event_loop);
        }
    }
}

fn optional_string_env(name: &'static str) -> EngineResult<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(EngineError::InvalidValue(format!(
            "{name} must contain valid Unicode"
        ))),
    }
}

fn optional_path_env(name: &'static str) -> EngineResult<Option<PathBuf>> {
    match std::env::var_os(name) {
        Some(value) if value.is_empty() => Err(EngineError::InvalidValue(format!(
            "{name} must not be empty"
        ))),
        Some(value) => Ok(Some(PathBuf::from(value))),
        None => Ok(None),
    }
}

fn optional_parsed_env<T>(name: &'static str) -> EngineResult<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    optional_string_env(name)?
        .map(|raw| {
            raw.parse::<T>().map_err(|error| {
                EngineError::InvalidValue(format!("{name} has invalid value {raw:?}: {error}"))
            })
        })
        .transpose()
}

fn parse_bool_env_value(name: &'static str, raw: &str) -> EngineResult<bool> {
    match raw {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(EngineError::InvalidValue(format!(
            "{name} must be 0 or 1, got {raw:?}"
        ))),
    }
}

fn optional_bool_env(name: &'static str) -> EngineResult<Option<bool>> {
    optional_string_env(name)?
        .map(|raw| parse_bool_env_value(name, &raw))
        .transpose()
}
/// `ENGINE_INPUT_LOG=1`: print every key event and each frame's movement
/// input, to debug input that arrives wrongly.
fn input_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_INPUT_LOG").is_ok_and(|v| v.trim() != "0"))
}

fn hitch_threshold_ms() -> EngineResult<f32> {
    match std::env::var("ENGINE_HITCH_MS") {
        Ok(raw) => {
            let ms = raw.parse::<f32>().map_err(|error| {
                EngineError::InvalidValue(format!(
                    "ENGINE_HITCH_MS must be a finite millisecond budget, got {raw:?}: {error}"
                ))
            })?;
            if !ms.is_finite() || ms <= 0.0 {
                return Err(EngineError::InvalidValue(format!(
                    "ENGINE_HITCH_MS must be > 0, got {ms}"
                )));
            }
            Ok(ms)
        }
        Err(std::env::VarError::NotPresent) => Ok(33.0),
        Err(std::env::VarError::NotUnicode(_)) => Err(EngineError::InvalidValue(
            "ENGINE_HITCH_MS must contain valid Unicode".into(),
        )),
    }
}

fn elapsed_ms(start: Instant) -> f32 {
    start.elapsed().as_secs_f32() * 1000.0
}

/// Per-phase frame timings the hitch log reports.
struct HitchPhases {
    update_ms: f32,
    anim_ms: f32,
    sync_ms: f32,
    render_ms: f32,
}

impl HitchPhases {
    fn work_ms(&self) -> f32 {
        self.update_ms + self.anim_ms + self.sync_ms + self.render_ms
    }
}

fn emit_hitch(
    path: &std::path::Path,
    frame_index: u32,
    fps: f32,
    wall_ms: f32,
    phases: HitchPhases,
    notes: &[HitchSpan],
    gpu: &GpuFrameStats,
) {
    use std::io::Write;
    let work_ms = phases.work_ms();
    let mut text = format!(
        "HITCH work={work_ms:.1}ms wall={wall_ms:.1}ms frame={frame_index} fps={fps:.0}  phases update={:.1} anim={:.1} sync={:.1} render={:.1}\n",
        phases.update_ms, phases.anim_ms, phases.sync_ms, phases.render_ms
    );
    let mut notes = notes.to_vec();
    notes.sort_by(|a, b| b.ms.total_cmp(&a.ms));
    for note in &notes {
        if note.detail.is_empty() {
            text.push_str(&format!("  {:<12} {:>5.1}ms\n", note.name, note.ms));
        } else {
            text.push_str(&format!(
                "  {:<12} {:>5.1}ms  {}\n",
                note.name, note.ms, note.detail
            ));
        }
    }
    text.push_str(&format!(
        "  {:<12} {:>5.1}ms  {}\n",
        "gpu_sync",
        phases.sync_ms,
        gpu.sync_line()
    ));
    text.push_str(&format!(
        "  {:<12} {:>5.1}ms  {}\n",
        "gpu_draw",
        phases.render_ms,
        gpu.draw_line()
    ));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("hitch log {} could not be opened: {e}", path.display()));
    file.write_all(text.as_bytes())
        .unwrap_or_else(|e| panic!("hitch log {} could not be written: {e}", path.display()));
}

/// Run the engine: opens a window and calls `update` every frame.
pub fn run(
    title: impl Into<String>,
    update: impl FnMut(&mut World, &Frame) -> EngineResult<()> + 'static,
) -> EngineResult<()> {
    run_with(title, EngineLimits::default(), update)
}

/// Run with custom resource limits.
pub fn run_with(
    title: impl Into<String>,
    limits: EngineLimits,
    update: impl FnMut(&mut World, &Frame) -> EngineResult<()> + 'static,
) -> EngineResult<()> {
    let event_loop = EventLoop::new().map_err(EngineError::EventLoopCreation)?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(title.into(), limits, Box::new(update))?;
    event_loop
        .run_app(&mut app)
        .map_err(EngineError::EventLoopRun)
}

fn run_callback_until_fatal(
    fatal_error: &mut Option<String>,
    callback: impl FnOnce() -> EngineResult<()>,
) -> bool {
    if fatal_error.is_some() {
        return false;
    }
    match callback() {
        Ok(()) => true,
        Err(error) => {
            *fatal_error = Some(format_error_chain(&error));
            false
        }
    }
}

fn format_error_chain(error: &EngineError) -> String {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(error) = source {
        message.push_str("\nCaused by: ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

#[cfg(test)]
mod tests {
    use super::{parse_bool_env_value, run_callback_until_fatal};

    #[test]
    fn boolean_environment_values_are_strict() {
        assert!(!parse_bool_env_value("FLAG", "0").unwrap());
        assert!(parse_bool_env_value("FLAG", "1").unwrap());
        assert!(parse_bool_env_value("FLAG", "true").is_err());
        assert!(parse_bool_env_value("FLAG", "").is_err());
    }
    use crate::error::{EngineError, EngineResult};

    #[test]
    fn callback_error_is_persistent_and_callback_never_runs_again() {
        let mut fatal_error = None;
        let mut calls = 0_u32;

        assert!(!run_callback_until_fatal(&mut fatal_error, || {
            calls += 1;
            Err(EngineError::application(std::io::Error::other(
                "session failed",
            )))
        }));
        assert!(!run_callback_until_fatal(&mut fatal_error, || {
            calls += 1;
            EngineResult::Ok(())
        }));

        assert_eq!(calls, 1);
        assert_eq!(
            fatal_error.as_deref(),
            Some("application callback failed: session failed\nCaused by: session failed")
        );
    }
}
