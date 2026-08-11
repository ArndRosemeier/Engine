use crate::input::Input;
use crate::limits::EngineLimits;
use crate::render::Renderer;
use crate::world::{Frame, World};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

type UpdateFn = Box<dyn FnMut(&mut World, &Frame)>;

struct App {
    title: String,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    world: World,
    update: UpdateFn,
    input: Input,
    start: Instant,
    last: Instant,
    frame_index: u32,
    first_update_done: bool,
    screenshot_path: Option<PathBuf>,
    screenshot_frame: u32,
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
            world: World::new().with_limits(limits),
            update,
            input: Input::new(),
            start: now,
            last: now,
            frame_index: 0,
            first_update_done: false,
            screenshot_path,
            screenshot_frame,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(window.clone()));
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.start = Instant::now();
        self.last = self.start;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                // Ignore key repeat so we don't thrash; held state is enough.
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

                let size = self
                    .renderer
                    .as_ref()
                    .map(|r| r.size())
                    .unwrap_or_default();
                let first = !self.first_update_done;
                let frame = Frame {
                    dt,
                    time,
                    width: size.width,
                    height: size.height,
                    aspect: size.width as f32 / size.height.max(1) as f32,
                    first,
                    input: self.input.clone(),
                };
                self.first_update_done = true;

                (self.update)(&mut self.world, &frame);

                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.sync_world(&self.world);
                    match renderer.render(&self.world) {
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

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
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
