//! egui-winit + egui-wgpu glue for the engine app loop.

use crate::ui::UiFrame;
use egui::{Context, FullOutput, TextureHandle, ViewportId};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct UiBackend {
    ctx: Context,
    state: EguiWinitState,
    renderer: EguiRenderer,
    textures: Rc<RefCell<HashMap<String, TextureHandle>>>,
    escape_pressed: bool,
}

impl UiBackend {
    pub fn new(
        window: &Window,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let ctx = Context::default();
        let state = EguiWinitState::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer = EguiRenderer::new(device, surface_format, None, 1, false);
        Self {
            ctx,
            state,
            renderer,
            textures: Rc::new(RefCell::new(HashMap::new())),
            escape_pressed: false,
        }
    }

    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        if let WindowEvent::KeyboardInput {
            event:
                winit::event::KeyEvent {
                    logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                    state: winit::event::ElementState::Pressed,
                    repeat: false,
                    ..
                },
            ..
        } = event
        {
            self.escape_pressed = true;
        }
        self.state.on_window_event(window, event).consumed
    }

    pub fn take_escape_pressed(&mut self) -> bool {
        let v = self.escape_pressed;
        self.escape_pressed = false;
        v
    }

    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }

    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }

    /// Run egui for one frame while the caller builds UI via [`UiFrame`].
    pub fn run_ui<R>(
        &mut self,
        window: &Window,
        build: impl FnOnce(&UiFrame) -> R,
    ) -> (R, FullOutput) {
        let raw_input = self.state.take_egui_input(window);
        let textures = Rc::clone(&self.textures);
        let ctx = self.ctx.clone();
        let mut build = Some(build);
        let mut out = None;
        let full_output = self.ctx.run(raw_input, |_| {
            let ui = UiFrame::new(ctx.clone(), Rc::clone(&textures));
            if let Some(build) = build.take() {
                out = Some(build(&ui));
            }
        });
        (out.expect("egui run always invokes closure"), full_output)
    }

    pub fn paint(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        full_output: FullOutput,
    ) {
        let pixels_per_point = full_output.pixels_per_point;
        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self.ctx.tessellate(full_output.shapes, pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        let screen = ScreenDescriptor {
            size_in_pixels: [width.max(1), height.max(1)],
            pixels_per_point,
        };

        let _callbacks = self
            .renderer
            .update_buffers(device, queue, encoder, &tris, &screen);

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.renderer
                .render(&mut rpass.forget_lifetime(), &tris, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
