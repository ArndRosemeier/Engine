use crate::input::Input;
use crate::limits::EngineLimits;
use crate::render::GpuFrameStats;
use crate::render::Renderer;
use crate::ui_backend::UiBackend;
use crate::world::{Frame, HitchSpan, World};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};

type UpdateFn = Box<dyn FnMut(&mut World, &Frame)>;

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
}

impl App {
    fn new(title: String, limits: EngineLimits, update: UpdateFn) -> Self {
        let now = Instant::now();
        let screenshot_path = std::env::var_os("ENGINE_SCREENSHOT").map(PathBuf::from);
        let screenshot_frame = std::env::var("ENGINE_SCREENSHOT_FRAME")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let screenshot_wait = std::env::var("ENGINE_SCREENSHOT_WAIT")
            .ok()
            .is_some_and(|s| s == "1");
        Self {
            title,
            window: None,
            renderer: None,
            ui_backend: None,
            world: World::new().with_limits(limits),
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
            hitch_ms: hitch_threshold_ms(),
        }
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

    fn window_inner_size() -> winit::dpi::LogicalSize<u32> {
        let w = std::env::var("ENGINE_WIDTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1920);
        let h = std::env::var("ENGINE_HEIGHT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1080);
        winit::dpi::LogicalSize::new(w.max(320), h.max(180))
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(Self::window_inner_size());
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(window.clone()));
        let ui_backend = UiBackend::new(&window, renderer.device(), renderer.surface_format());
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
                if !event.repeat {
                    self.input.set_key(event.physical_key, pressed);
                }
            }
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
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last).as_secs_f32();
                self.last = now;
                let time = (now - self.start).as_secs_f32();

                self.fps_accum_s += dt.max(0.0);
                self.fps_frames += 1;
                if self.fps_accum_s >= 0.5 {
                    self.fps = self.fps_frames as f32 / self.fps_accum_s;
                    self.fps_accum_s = 0.0;
                    self.fps_frames = 0;
                    if let Some(window) = &self.window {
                        window.set_title(&format!("{} — {:.0} FPS", self.title, self.fps));
                    }
                }

                let size = self.renderer.as_ref().map(|r| r.size()).unwrap_or_default();
                let first = !self.first_update_done;
                self.first_update_done = true;

                let Some(window) = self.window.clone() else {
                    return;
                };
                let Some(ui_backend) = self.ui_backend.as_mut() else {
                    return;
                };

                // A locked pointer means the game owns the mouse, so egui's
                // hover state must not swallow look and movement input.
                let mut input = self.input.clone();
                if !self.world.bind_listen()
                    && !self.pointer_locked
                    && (ui_backend.wants_keyboard_input() || ui_backend.wants_pointer_input())
                {
                    input = Input::new();
                }

                let update = &mut self.update;
                let world = &mut self.world;
                let fps = self.fps;
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
                        };
                        update(world, &frame);
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

                self.world.set_time(time);
                let anim_t = Instant::now();
                self.world.tick_animations(dt);
                let anim_ms = elapsed_ms(anim_t);

                if let Some(renderer) = self.renderer.as_mut() {
                    let sync_t = Instant::now();
                    renderer.sync_world(&self.world);
                    let sync_ms = elapsed_ms(sync_t);
                    let ui_backend = self.ui_backend.as_mut().expect("ui backend");
                    let render_t = Instant::now();
                    match renderer.render_with(&self.world, |device, queue, encoder, view| {
                        ui_backend.paint(
                            &window,
                            device,
                            queue,
                            encoder,
                            view,
                            size.width,
                            size.height,
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
                        Err(wgpu::SurfaceError::Timeout) => {}
                        Err(other) => panic!("surface error: {other}"),
                    }
                    let render_ms = elapsed_ms(render_t);
                    let gpu = renderer.take_gpu_stats();
                    let notes = self.world.take_hitch_spans();
                    let work_ms = update_ms + anim_ms + sync_ms + render_ms;
                    if work_ms >= self.hitch_ms {
                        if let Some(path) = self.world.hitch_log() {
                            emit_hitch(
                                path,
                                self.frame_index,
                                fps,
                                dt * 1000.0,
                                update_ms,
                                anim_ms,
                                sync_ms,
                                render_ms,
                                &notes,
                                &gpu,
                            );
                        }
                    }
                    self.frame_index += 1;
                    let queued = self.world.take_screenshot_queue();
                    for path in queued {
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        renderer.capture_png(&self.world, &path);
                        eprintln!("wrote screenshot {}", path.display());
                    }
                    if self.world.take_exit_requested() {
                        event_loop.exit();
                        return;
                    }
                    if !self.screenshot_wait {
                        if let Some(path) = self.screenshot_path.clone() {
                            if self.frame_index >= self.screenshot_frame {
                                renderer.capture_png(&self.world, &path);
                                eprintln!("wrote screenshot {}", path.display());
                                event_loop.exit();
                                return;
                            }
                        }
                    }
                } else {
                    let _ = self.world.take_hitch_spans();
                }

                window.request_redraw();
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn hitch_threshold_ms() -> f32 {
    match std::env::var("ENGINE_HITCH_MS") {
        Ok(raw) => {
            let ms = raw.parse::<f32>().unwrap_or_else(|e| {
                panic!("ENGINE_HITCH_MS must be a finite millisecond budget, got {raw:?}: {e}")
            });
            if !ms.is_finite() || ms <= 0.0 {
                panic!("ENGINE_HITCH_MS must be > 0, got {ms}");
            }
            ms
        }
        Err(_) => 33.0,
    }
}

fn elapsed_ms(start: Instant) -> f32 {
    start.elapsed().as_secs_f32() * 1000.0
}

fn emit_hitch(
    path: &std::path::Path,
    frame_index: u32,
    fps: f32,
    wall_ms: f32,
    update_ms: f32,
    anim_ms: f32,
    sync_ms: f32,
    render_ms: f32,
    notes: &[HitchSpan],
    gpu: &GpuFrameStats,
) {
    use std::io::Write;
    let work_ms = update_ms + anim_ms + sync_ms + render_ms;
    let mut text = format!(
        "HITCH work={work_ms:.1}ms wall={wall_ms:.1}ms frame={frame_index} fps={fps:.0}  phases update={update_ms:.1} anim={anim_ms:.1} sync={sync_ms:.1} render={render_ms:.1}\n"
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
        sync_ms,
        gpu.sync_line()
    ));
    text.push_str(&format!(
        "  {:<12} {:>5.1}ms  {}\n",
        "gpu_draw",
        render_ms,
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
pub fn run(title: impl Into<String>, update: impl FnMut(&mut World, &Frame) + 'static) {
    run_with(title, EngineLimits::default(), update);
}

/// Run with custom resource limits.
pub fn run_with(
    title: impl Into<String>,
    limits: EngineLimits,
    update: impl FnMut(&mut World, &Frame) + 'static,
) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(title.into(), limits, Box::new(update));
    event_loop.run_app(&mut app).expect("event loop error");
}
