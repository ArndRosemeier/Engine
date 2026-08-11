use crate::input::Input;
use crate::limits::EngineLimits;
use crate::render::Renderer;
use crate::ui_backend::UiBackend;
use crate::world::{Frame, World};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

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
    fps: f32,
    fps_accum_s: f32,
    fps_frames: u32,
}

impl App {
    fn new(title: String, limits: EngineLimits, update: UpdateFn) -> Self {
        let now = Instant::now();
        let screenshot_path = std::env::var_os("ENGINE_SCREENSHOT").map(PathBuf::from);
        let screenshot_frame = std::env::var("ENGINE_SCREENSHOT_FRAME")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
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
            fps: 0.0,
            fps_accum_s: 0.0,
            fps_frames: 0,
        }
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
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

                let size = self
                    .renderer
                    .as_ref()
                    .map(|r| r.size())
                    .unwrap_or_default();
                let first = !self.first_update_done;
                self.first_update_done = true;

                let Some(window) = self.window.clone() else {
                    return;
                };
                let Some(ui_backend) = self.ui_backend.as_mut() else {
                    return;
                };

                let mut input = self.input.clone();
                if ui_backend.wants_keyboard_input() || ui_backend.wants_pointer_input() {
                    input = Input::new();
                }

                let update = &mut self.update;
                let world = &mut self.world;
                let fps = self.fps;
                let (modal_was_open, full_output) = {
                    let (ui_result, full_output) = ui_backend.run_ui(&window, |ui| {
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

                if ui_backend.take_escape_pressed() && !modal_was_open {
                    event_loop.exit();
                    return;
                }

                self.world.tick_animations(dt);

                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.sync_world(&self.world);
                    let ui_backend = self.ui_backend.as_mut().expect("ui backend");
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

                    self.frame_index += 1;
                    if let Some(path) = self.screenshot_path.clone() {
                        if self.frame_index >= self.screenshot_frame {
                            renderer.capture_png(&self.world, &path);
                            eprintln!("wrote screenshot {}", path.display());
                            event_loop.exit();
                            return;
                        }
                    }
                }

                window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
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
