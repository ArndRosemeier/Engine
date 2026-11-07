//! GPU particles: emission, simulation and drawing all stay on the GPU.
//!
//! Every frame:
//! 1. `cs_emit` takes free particles off the dead list for this frame's
//!    spawns (the CPU only says "emitter E, n particles").
//! 2. `cs_prepare` sizes the simulation dispatch from the alive count.
//! 3. `cs_sim` moves the alive particles (forces, smooth turbulence, drag,
//!    collision against last frame's depth buffer), returns dead ones to the
//!    dead list and sorts survivors into three draw lists: chunks, alpha
//!    sprites and additive sprites.
//! 4. `cs_finish` writes the indirect draw counts.
//!
//! Drawing happens after the world (and water) in a pass of its own that
//! reads the scene depth: sprites fade where they touch geometry (soft
//! particles) and chunks test depth by hand.

use crate::particles::{
    EmitterId, ParticleBlend, ParticleCommand, ParticleEmitter, ParticleForce, ParticleMode,
    ParticleShape, ParticleSilhouette, SizeOverLife, MAX_PARTICLES, MAX_PARTICLE_EMITTERS,
};
use glam::{Mat4, Vec3};
use std::mem::size_of;
use wgpu::util::DeviceExt;

/// One-off bursts per frame (they use slots after the running emitters).
const TRANSIENT: usize = 512;
const EMITTER_SLOTS: usize = MAX_PARTICLE_EMITTERS + TRANSIENT;
const PARTICLE_BYTES: u64 = 128;
/// Lists in the list buffer, each `MAX_PARTICLES` long: dead, alive (two,
/// ping-pong), chunk draws, alpha draws, additive draws.
const LISTS: u64 = 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitterRaw {
    position: [f32; 3],
    size_variation: f32,
    velocity: [f32; 3],
    lifetime: f32,
    spread: [f32; 3],
    size: f32,
    jitter: [f32; 3],
    emissive: f32,
    color: [f32; 4],
    secondary: [f32; 4],
    acceleration: [f32; 3],
    drag: f32,
    force_vector: [f32; 3],
    force_strength: f32,
    turbulence: f32,
    seed: u32,
    flags: u32,
    shape: u32,
    bounce: f32,
    friction: f32,
    lighting: f32,
    spin: f32,
    stretch: f32,
    generation: u32,
    _pad: [u32; 2],
}
const _: () = assert!(size_of::<EmitterRaw>() == 176);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct SpawnRaw {
    emitter: u32,
    serial: u32,
    count: u32,
    start: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimRaw {
    prev_vp: [[f32; 4]; 4],
    inv_prev_vp: [[f32; 4]; 4],
    dt: f32,
    time: f32,
    width: f32,
    height: f32,
    depth_valid: u32,
    cur: u32,
    total: u32,
    spawns: u32,
    prev_eye: [f32; 3],
    _pad: f32,
}
const _: () = assert!(size_of::<SimRaw>() == 176);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawRaw {
    inv_vp: [[f32; 4]; 4],
    eye: [f32; 3],
    _pad: f32,
}

#[derive(Clone, Copy, Debug)]
struct EmitterRuntime {
    emitter: ParticleEmitter,
    generation: u32,
    emission_remainder: f64,
    serial: u32,
}

#[derive(Clone, Copy)]
struct View {
    vp: Mat4,
    eye: Vec3,
    size: (u32, u32),
}

pub struct ParticleGpu {
    particles: wgpu::Buffer,
    emitters: wgpu::Buffer,
    spawns: wgpu::Buffer,
    lists: wgpu::Buffer,
    counters: wgpu::Buffer,
    args: wgpu::Buffer,
    sim: wgpu::Buffer,
    draw_uniform: wgpu::Buffer,
    compute_layout: wgpu::BindGroupLayout,
    render_layout: wgpu::BindGroupLayout,
    compute_bind: wgpu::BindGroup,
    args_bind: wgpu::BindGroup,
    render_bind: wgpu::BindGroup,
    emit: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    simulate: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    chunk_render: wgpu::RenderPipeline,
    alpha_render: wgpu::RenderPipeline,
    additive_render: wgpu::RenderPipeline,
    runtimes: Vec<Option<EmitterRuntime>>,
    /// Parity of the alive list the next simulation reads.
    cur: u32,
    /// Simulation time until which particles may still be alive.
    busy_until: f32,
    /// Extra frames to simulate after `busy_until`, so the last particles
    /// make it back onto the dead list.
    tail_frames: u32,
    sim_time: f32,
    total_this_frame: u32,
    active: bool,
    last_view: Option<View>,
    depth_size: (u32, u32),
    reset_lists: bool,
    one_shot_serial: u32,
}

impl ParticleGpu {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        scene: &wgpu::BindGroupLayout,
        depth: &wgpu::Texture,
    ) -> Self {
        let n = MAX_PARTICLES as u64;
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: n * PARTICLE_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let emitters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-emitters"),
            size: (EMITTER_SLOTS * size_of::<EmitterRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spawns = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-spawns"),
            size: (EMITTER_SLOTS * size_of::<SpawnRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lists = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-lists"),
            contents: bytemuck::cast_slice(&initial_lists()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let counters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-counters"),
            contents: bytemuck::cast_slice(&initial_counters()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-indirect-args"),
            contents: &[0u8; 64],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });
        let sim = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-sim"),
            size: size_of::<SimRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-draw-view"),
            size: size_of::<DrawRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let multisampled = depth.sample_count() > 1;
        let depth_entry =
            |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled,
                },
                count: None,
            };
        let cs = wgpu::ShaderStages::COMPUTE;
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-compute-layout"),
            entries: &[
                storage_entry(0, cs, false),
                storage_entry(1, cs, true),
                storage_entry(2, cs, true),
                storage_entry(3, cs, false),
                storage_entry(4, cs, false),
                uniform_entry(5, cs),
                depth_entry(6, cs),
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-args-layout"),
            entries: &[storage_entry(0, cs, false)],
        });
        let vs = wgpu::ShaderStages::VERTEX;
        let fs = wgpu::ShaderStages::FRAGMENT;
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-render-layout"),
            entries: &[
                storage_entry(0, vs, true),
                storage_entry(1, vs, true),
                uniform_entry(2, vs | fs),
                depth_entry(3, fs),
            ],
        });
        let args_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle-args-bind"),
            layout: &args_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: args.as_entire_binding(),
            }],
        });
        let depth_type = if multisampled {
            "texture_depth_multisampled_2d"
        } else {
            "texture_depth_2d"
        };
        let common = COMMON_WGSL.replace("CAPACITY", &MAX_PARTICLES.to_string());
        let compute_src = format!("{common}{COMPUTE_WGSL}").replace("DEPTH_TYPE", depth_type);
        let render_src = format!("{common}{RENDER_WGSL}").replace("DEPTH_TYPE", depth_type);
        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle-compute"),
            source: wgpu::ShaderSource::Wgsl(compute_src.into()),
        });
        let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle-render"),
            source: wgpu::ShaderSource::Wgsl(render_src.into()),
        });
        let main_only = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle-compute-pipeline-layout"),
            bind_group_layouts: &[&compute_layout],
            push_constant_ranges: &[],
        });
        let with_args = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle-args-pipeline-layout"),
            bind_group_layouts: &[&compute_layout, &args_layout],
            push_constant_ranges: &[],
        });
        let compute = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &compute_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let emit = compute("cs_emit", &main_only);
        let prepare = compute("cs_prepare", &with_args);
        let simulate = compute("cs_sim", &main_only);
        let finish = compute("cs_finish", &with_args);
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-render-pipeline-layout"),
                bind_group_layouts: &[scene, &render_layout],
                push_constant_ranges: &[],
            });
        let render = |label: &str, vs: &str, fs: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: Some(vs),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: Some(fs),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let chunk_render = render("particle-chunks", "vs_chunk", "fs_chunk", None);
        let alpha_render = render(
            "particle-alpha-sprites",
            "vs_alpha",
            "fs_alpha",
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        );
        let additive_render = render(
            "particle-additive-sprites",
            "vs_additive",
            "fs_additive",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            }),
        );
        let (compute_bind, render_bind) = make_binds(
            device,
            &compute_layout,
            &render_layout,
            [&particles, &emitters, &spawns, &lists, &counters, &sim],
            &draw_uniform,
            depth,
        );
        Self {
            particles,
            emitters,
            spawns,
            lists,
            counters,
            args,
            sim,
            draw_uniform,
            compute_layout,
            render_layout,
            compute_bind,
            args_bind,
            render_bind,
            emit,
            prepare,
            simulate,
            finish,
            chunk_render,
            alpha_render,
            additive_render,
            runtimes: vec![None; MAX_PARTICLE_EMITTERS],
            cur: 0,
            busy_until: 0.0,
            tail_frames: 0,
            sim_time: 0.0,
            total_this_frame: 0,
            active: false,
            last_view: None,
            depth_size: (depth.width(), depth.height()),
            reset_lists: false,
            one_shot_serial: 0,
        }
    }

    /// The depth buffer was recreated (window resize).
    pub fn set_depth(&mut self, device: &wgpu::Device, depth: &wgpu::Texture) {
        let (c, r) = make_binds(
            device,
            &self.compute_layout,
            &self.render_layout,
            [
                &self.particles,
                &self.emitters,
                &self.spawns,
                &self.lists,
                &self.counters,
                &self.sim,
            ],
            &self.draw_uniform,
            depth,
        );
        self.compute_bind = c;
        self.render_bind = r;
        self.depth_size = (depth.width(), depth.height());
        self.last_view = None;
    }

    /// Take this frame's emitter changes and work out what to spawn.
    pub fn sync(
        &mut self,
        queue: &wgpu::Queue,
        commands: Vec<ParticleCommand>,
        dt: f32,
        time: f32,
    ) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "particle delta time must be finite and non-negative"
        );
        assert!(time.is_finite(), "particle time must be finite");
        self.sim_time = time;
        let mut spawns: Vec<SpawnRaw> = Vec::new();
        let mut total = 0u32;
        let mut longest = 0.0f32;
        let mut transient = 0usize;
        let push = |spawns: &mut Vec<SpawnRaw>,
                    total: &mut u32,
                    emitter: usize,
                    serial: u32,
                    count: u32| {
            let count = count.min(MAX_PARTICLES.saturating_sub(*total));
            if count > 0 {
                spawns.push(SpawnRaw {
                    emitter: emitter as u32,
                    serial,
                    count,
                    start: *total,
                });
                *total += count;
            }
        };
        for command in commands {
            match command {
                ParticleCommand::Start(id, emitter) => {
                    let slot = id.slot() as usize;
                    assert!(
                        self.runtimes[slot].is_none(),
                        "particle GPU emitter slot started while occupied"
                    );
                    queue.write_buffer(
                        &self.emitters,
                        (slot * size_of::<EmitterRaw>()) as u64,
                        bytemuck::bytes_of(&to_raw(emitter, id.generation())),
                    );
                    let mut runtime = EmitterRuntime {
                        emitter,
                        generation: id.generation(),
                        emission_remainder: 0.0,
                        serial: 0,
                    };
                    if emitter.mode() == ParticleMode::Burst {
                        push(&mut spawns, &mut total, slot, 0, emitter.burst_count());
                        runtime.serial = emitter.burst_count();
                        longest = longest.max(emitter.lifetime_s());
                    }
                    self.runtimes[slot] = Some(runtime);
                }
                ParticleCommand::UpdatePosition(id, position) => {
                    self.update_position(queue, id, position)
                }
                ParticleCommand::Stop(id) => {
                    let slot = id.slot() as usize;
                    let runtime = self.runtimes[slot]
                        .take()
                        .expect("particle GPU received stop for unoccupied slot");
                    assert_eq!(
                        runtime.generation,
                        id.generation(),
                        "particle GPU received stale stop command"
                    );
                }
                ParticleCommand::Burst(emitter) => {
                    if transient >= TRANSIENT {
                        continue;
                    }
                    let slot = MAX_PARTICLE_EMITTERS + transient;
                    transient += 1;
                    self.one_shot_serial = self.one_shot_serial.wrapping_add(1);
                    queue.write_buffer(
                        &self.emitters,
                        (slot * size_of::<EmitterRaw>()) as u64,
                        bytemuck::bytes_of(&to_raw(emitter, self.one_shot_serial)),
                    );
                    push(&mut spawns, &mut total, slot, 0, emitter.burst_count());
                    longest = longest.max(emitter.lifetime_s());
                }
                ParticleCommand::Clear => {
                    self.runtimes.fill(None);
                    spawns.clear();
                    total = 0;
                    self.reset_lists = true;
                }
            }
        }
        for (slot, runtime) in self.runtimes.iter_mut().enumerate() {
            let Some(runtime) = runtime else { continue };
            if runtime.emitter.mode() != ParticleMode::Continuous {
                continue;
            }
            let n =
                schedule_continuous(&mut runtime.emission_remainder, runtime.emitter.rate(), dt);
            // Running emitters keep the system awake even between spawns.
            longest = longest.max(runtime.emitter.lifetime_s());
            if n > 0 {
                push(&mut spawns, &mut total, slot, runtime.serial, n);
                runtime.serial = runtime.serial.wrapping_add(n);
            }
        }
        if self.reset_lists {
            queue.write_buffer(&self.lists, 0, bytemuck::cast_slice(&initial_lists()));
            queue.write_buffer(&self.counters, 0, bytemuck::cast_slice(&initial_counters()));
            self.cur = 0;
            self.reset_lists = false;
        }
        if longest > 0.0 {
            // Lifetimes vary by up to 15% per particle.
            self.busy_until = self.busy_until.max(time + longest * 1.2 + 0.05);
            self.tail_frames = 3;
        }
        if !spawns.is_empty() {
            queue.write_buffer(&self.spawns, 0, bytemuck::cast_slice(&spawns));
        }
        self.total_this_frame = total;
        self.active = time < self.busy_until || self.tail_frames > 0;
        let (width, height) = self.depth_size;
        let (prev_vp, prev_eye, valid) = match self.last_view {
            Some(v) if v.size == self.depth_size => (v.vp, v.eye, 1),
            _ => (Mat4::IDENTITY, Vec3::ZERO, 0),
        };
        queue.write_buffer(
            &self.sim,
            0,
            bytemuck::bytes_of(&SimRaw {
                prev_vp: prev_vp.to_cols_array_2d(),
                inv_prev_vp: prev_vp.inverse().to_cols_array_2d(),
                dt,
                time,
                width: width as f32,
                height: height as f32,
                depth_valid: valid,
                cur: self.cur,
                total,
                spawns: spawns.len() as u32,
                prev_eye: prev_eye.into(),
                _pad: 0.0,
            }),
        );
    }

    fn update_position(&mut self, queue: &wgpu::Queue, id: EmitterId, position: Vec3) {
        let slot = id.slot() as usize;
        let runtime = self
            .runtimes
            .get_mut(slot)
            .and_then(Option::as_mut)
            .expect("particle GPU received position update for unoccupied slot");
        assert_eq!(
            runtime.generation,
            id.generation(),
            "particle GPU received stale position update"
        );
        runtime.emitter.set_position(position);
        queue.write_buffer(
            &self.emitters,
            (slot * size_of::<EmitterRaw>()) as u64,
            bytemuck::bytes_of(&<[f32; 3]>::from(position)),
        );
    }

    /// Emit and simulate this frame's particles. Runs before the world is
    /// drawn, so the depth buffer still holds last frame's scene, which the
    /// collision test reads.
    pub fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder, _scene: &wgpu::BindGroup) {
        if !self.active {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle-emit"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.compute_bind, &[]);
            if self.total_this_frame > 0 {
                pass.set_pipeline(&self.emit);
                pass.dispatch_workgroups(self.total_this_frame.div_ceil(64), 1, 1);
            }
            pass.set_bind_group(1, &self.args_bind, &[]);
            pass.set_pipeline(&self.prepare);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle-simulate"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.compute_bind, &[]);
            pass.set_pipeline(&self.simulate);
            pass.dispatch_workgroups_indirect(&self.args, 0);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle-finish"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.compute_bind, &[]);
            pass.set_bind_group(1, &self.args_bind, &[]);
            pass.set_pipeline(&self.finish);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.cur ^= 1;
        if self.sim_time >= self.busy_until {
            self.tail_frames = self.tail_frames.saturating_sub(1);
        }
    }

    /// Draw the particles over the finished scene (`target` is the
    /// single-sample scene colour, `depth` sized). `view_proj` and `eye` are
    /// this frame's camera; they also feed next frame's collisions.
    pub fn encode_draw(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        scene: &wgpu::BindGroup,
        view_proj: Mat4,
        eye: Vec3,
    ) {
        self.last_view = Some(View {
            vp: view_proj,
            eye,
            size: self.depth_size,
        });
        if !self.active {
            return;
        }
        queue.write_buffer(
            &self.draw_uniform,
            0,
            bytemuck::bytes_of(&DrawRaw {
                inv_vp: view_proj.inverse().to_cols_array_2d(),
                eye: eye.into(),
                _pad: 0.0,
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_bind_group(0, scene, &[]);
        pass.set_bind_group(1, &self.render_bind, &[]);
        pass.set_pipeline(&self.chunk_render);
        pass.draw_indirect(&self.args, 16);
        pass.set_pipeline(&self.alpha_render);
        pass.draw_indirect(&self.args, 32);
        pass.set_pipeline(&self.additive_render);
        pass.draw_indirect(&self.args, 48);
    }
}

fn make_binds(
    device: &wgpu::Device,
    compute_layout: &wgpu::BindGroupLayout,
    render_layout: &wgpu::BindGroupLayout,
    [particles, emitters, spawns, lists, counters, sim]: [&wgpu::Buffer; 6],
    draw_uniform: &wgpu::Buffer,
    depth: &wgpu::Texture,
) -> (wgpu::BindGroup, wgpu::BindGroup) {
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
        label: Some("particle-depth-view"),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry { binding, resource }
    }
    let compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("particle-compute-bind"),
        layout: compute_layout,
        entries: &[
            entry(0, particles.as_entire_binding()),
            entry(1, emitters.as_entire_binding()),
            entry(2, spawns.as_entire_binding()),
            entry(3, lists.as_entire_binding()),
            entry(4, counters.as_entire_binding()),
            entry(5, sim.as_entire_binding()),
            entry(6, wgpu::BindingResource::TextureView(&depth_view)),
        ],
    });
    let render = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("particle-render-bind"),
        layout: render_layout,
        entries: &[
            entry(0, particles.as_entire_binding()),
            entry(1, lists.as_entire_binding()),
            entry(2, draw_uniform.as_entire_binding()),
            entry(3, wgpu::BindingResource::TextureView(&depth_view)),
        ],
    });
    (compute, render)
}

/// Dead list full (every particle free), everything else empty.
fn initial_lists() -> Vec<u32> {
    let n = MAX_PARTICLES as usize;
    let mut v = vec![0u32; n * LISTS as usize];
    for (i, x) in v[..n].iter_mut().enumerate() {
        *x = i as u32;
    }
    v
}

fn initial_counters() -> [u32; 8] {
    [MAX_PARTICLES, 0, 0, 0, 0, 0, 0, 0]
}

fn schedule_continuous(remainder: &mut f64, rate: f32, dt: f32) -> u32 {
    *remainder += f64::from(dt) * f64::from(rate);
    let count = remainder.floor();
    *remainder -= count;
    assert!(
        count <= f64::from(u32::MAX),
        "particle emission count exceeds u32"
    );
    count as u32
}

fn storage_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn silhouette_code(s: ParticleSilhouette) -> u32 {
    match s {
        ParticleSilhouette::SoftOrb => 0,
        ParticleSilhouette::Flame => 1,
        ParticleSilhouette::SparkStreak => 2,
        ParticleSilhouette::SmokeCloud => 3,
        ParticleSilhouette::Shard => 4,
        ParticleSilhouette::RuneMote => 5,
        ParticleSilhouette::Heart => 6,
        ParticleSilhouette::Bubble => 7,
        ParticleSilhouette::Chunk => 8,
        ParticleSilhouette::Puff => 9,
        ParticleSilhouette::Fire => 10,
    }
}

fn to_raw(e: ParticleEmitter, generation: u32) -> EmitterRaw {
    let c = e.color();
    let s = e.secondary_color();
    let size_over_life = match e.size_over_life() {
        SizeOverLife::Constant => 0,
        SizeOverLife::FadeInOut => 1,
        SizeOverLife::Shrink => 2,
        SizeOverLife::GrowThenFade => 3,
        SizeOverLife::Expand => 4,
    };
    let (force_kind, force_vector, force_strength) = match e.force() {
        ParticleForce::None => (0, Vec3::ZERO, 0.0),
        ParticleForce::Vortex { axis, strength } => (1, axis.normalize(), strength),
        ParticleForce::Radial { center, strength } => (2, center, strength),
    };
    let blend = match e.blend() {
        ParticleBlend::Additive => 0,
        ParticleBlend::Alpha => 1,
    };
    let collision = e.collision();
    EmitterRaw {
        position: e.position().into(),
        size_variation: e.size_variation(),
        velocity: e.velocity().into(),
        lifetime: e.lifetime_s(),
        spread: e.spread().into(),
        size: e.size(),
        jitter: e.velocity_jitter().into(),
        emissive: e.emissive_intensity(),
        color: [c.r, c.g, c.b, c.a],
        secondary: [s.r, s.g, s.b, s.a],
        acceleration: e.acceleration().into(),
        drag: e.drag(),
        force_vector: force_vector.into(),
        force_strength,
        turbulence: e.turbulence(),
        seed: e.seed(),
        flags: silhouette_code(e.silhouette())
            | (size_over_life << 8)
            | (force_kind << 12)
            | (blend << 16),
        shape: match e.shape() {
            ParticleShape::Point => 0,
            ParticleShape::Sphere => 1,
            ParticleShape::Cone => 2,
            ParticleShape::Ring => 3,
        },
        bounce: collision.map_or(-1.0, |c| c.bounce),
        friction: collision.map_or(0.0, |c| c.friction),
        lighting: e.lighting(),
        spin: e.spin(),
        stretch: e.velocity_stretch(),
        generation,
        _pad: [0; 2],
    }
}

const COMMON_WGSL: &str = r#"
const N: u32 = CAPACITYu;
const REST: u32 = 0x80000000u;
// p0 position, age | p1 velocity, lifetime | p2 acceleration, drag
// p3 force vector, strength | p4 origin, turbulence
// p5 colour, end colour (rgba8), emissive, size | p6 seed, flags, stretch, bounce
// p7 friction, lighting, spin, angle
struct Particle { p0:vec4<f32>, p1:vec4<f32>, p2:vec4<f32>, p3:vec4<f32>, p4:vec4<f32>, p5:vec4<u32>, p6:vec4<u32>, p7:vec4<f32> };
fn hash_u32(v:u32)->u32 { var x=v; x^=x>>16u; x*=0x7feb352du; x^=x>>15u; x*=0x846ca68bu; x^=x>>16u; return x; }
fn random(seed:u32,salt:u32)->f32 { return f32(hash_u32(seed^salt))/4294967296.0; }
fn random_dir(seed:u32)->vec3<f32> {
  let z=random(seed,0x91e10da5u)*2.0-1.0; let a=random(seed,0x6c8e9cf5u)*6.283185307;
  let r=sqrt(max(0.0,1.0-z*z)); return vec3<f32>(r*cos(a),z,r*sin(a));
}
"#;

const COMPUTE_WGSL: &str = r#"
struct Emitter {
  position:vec3<f32>, size_variation:f32, velocity:vec3<f32>, lifetime:f32,
  spread:vec3<f32>, size:f32, jitter:vec3<f32>, emissive:f32,
  color:vec4<f32>, secondary:vec4<f32>, acceleration:vec3<f32>, drag:f32,
  force_vector:vec3<f32>, force_strength:f32,
  turbulence:f32, seed:u32, flags:u32, shape:u32,
  bounce:f32, friction:f32, lighting:f32, spin:f32,
  stretch:f32, generation:u32, pad0:u32, pad1:u32,
};
struct Spawn { emitter:u32, serial:u32, count:u32, start:u32 };
struct Counters { dead:atomic<i32>, alive:array<atomic<u32>,2>, draw:array<atomic<u32>,3>, pad0:u32, pad1:u32 };
struct Args { dispatch:vec4<u32>, draws:array<vec4<u32>,3> };
struct Sim {
  prev_vp:mat4x4<f32>, inv_prev_vp:mat4x4<f32>,
  dt:f32, time:f32, width:f32, height:f32,
  depth_valid:u32, cur:u32, total:u32, spawns:u32,
  prev_eye:vec3<f32>, pad:f32,
};
@group(0) @binding(0) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(1) var<storage,read> emitters:array<Emitter>;
@group(0) @binding(2) var<storage,read> spawns:array<Spawn>;
@group(0) @binding(3) var<storage,read_write> lists:array<u32>;
@group(0) @binding(4) var<storage,read_write> counters:Counters;
@group(0) @binding(5) var<uniform> sim:Sim;
@group(0) @binding(6) var depth_tex:DEPTH_TYPE;
@group(1) @binding(0) var<storage,read_write> args:Args;

fn spawn_offset(shape:u32,spread:vec3<f32>,seed:u32)->vec3<f32>{
  if(shape==0u){return vec3<f32>(0.0);}
  if(shape==1u){return random_dir(seed)*pow(random(seed,0x27d4eb2du),1.0/3.0)*spread;}
  let a=random(seed,0x165667b1u)*6.283185307;
  if(shape==3u){return vec3<f32>(cos(a)*spread.x,0.0,sin(a)*spread.z);}
  let h=pow(random(seed,0xd3a2646cu),1.0/3.0); let r=sqrt(random(seed,0xfd7046c5u))*h;
  return vec3<f32>(cos(a)*r*spread.x,h*spread.y,sin(a)*r*spread.z);
}

@compute @workgroup_size(64) fn cs_emit(@builtin(global_invocation_id) id:vec3<u32>) {
  let t=id.x; if(t>=sim.total){return;}
  var lo=0u; var hi=sim.spawns-1u;
  loop { if(lo>=hi){break;} let mid=(lo+hi+1u)/2u; if(spawns[mid].start<=t){lo=mid;}else{hi=mid-1u;} }
  let sp=spawns[lo];
  let k=atomicSub(&counters.dead,1)-1;
  if(k<0){ atomicAdd(&counters.dead,1); return; }
  let pid=lists[u32(k)];
  let e=emitters[sp.emitter];
  let serial=sp.serial+(t-sp.start);
  let seed=hash_u32(e.seed^hash_u32(serial*0x9e3779b9u+e.generation*0x85ebca6bu+sp.emitter));
  var p:Particle;
  let pos=e.position+spawn_offset(e.shape,e.spread,seed);
  let vel=e.velocity+random_dir(seed^0xa511e9b3u)*e.jitter*(0.3+0.7*random(seed,0x2545f491u));
  let size=e.size*max(0.05,1.0+(random(seed,0x68e31da4u)*2.0-1.0)*e.size_variation);
  let life=e.lifetime*(0.85+0.3*random(seed,0xb5297a4du));
  let spin_sign=select(-1.0,1.0,random(seed,0x1b56c4e9u)<0.5);
  p.p0=vec4<f32>(pos,0.0);
  p.p1=vec4<f32>(vel,life);
  p.p2=vec4<f32>(e.acceleration,e.drag);
  p.p3=vec4<f32>(e.force_vector,e.force_strength);
  p.p4=vec4<f32>(e.position,e.turbulence);
  p.p5=vec4<u32>(pack4x8unorm(e.color),pack4x8unorm(e.secondary),bitcast<u32>(e.emissive),bitcast<u32>(size));
  p.p6=vec4<u32>(seed,e.flags,bitcast<u32>(e.stretch),bitcast<u32>(e.bounce));
  p.p7=vec4<f32>(e.friction,e.lighting,e.spin*spin_sign*(0.5+random(seed,0x7f4a7c15u)),random(seed,0x3c6ef372u)*6.283185307);
  particles[pid]=p;
  let slot=atomicAdd(&counters.alive[sim.cur],1u);
  lists[N*(1u+sim.cur)+slot]=pid;
}

@compute @workgroup_size(1) fn cs_prepare() {
  let n=atomicLoad(&counters.alive[sim.cur]);
  args.dispatch=vec4<u32>((n+63u)/64u,1u,1u,0u);
  atomicStore(&counters.alive[1u-sim.cur],0u);
  atomicStore(&counters.draw[0],0u);
  atomicStore(&counters.draw[1],0u);
  atomicStore(&counters.draw[2],0u);
}

@compute @workgroup_size(1) fn cs_finish() {
  args.draws[0]=vec4<u32>(60u,atomicLoad(&counters.draw[0]),0u,0u);
  args.draws[1]=vec4<u32>(6u,atomicLoad(&counters.draw[1]),0u,0u);
  args.draws[2]=vec4<u32>(6u,atomicLoad(&counters.draw[2]),0u,0u);
}

// Smooth swirling air: a cheap sum of sines, continuous in space and time.
fn flow(p:vec3<f32>,t:f32,seed:u32)->vec3<f32>{
  let q=p*0.55+vec3<f32>(t*0.31,t*0.23,t*0.27)+vec3<f32>(f32(seed&63u)*0.05);
  return vec3<f32>(sin(q.y*1.7+cos(q.z*1.3)),sin(q.z*1.5+cos(q.x*1.1))*0.7,sin(q.x*1.3+cos(q.y*1.9)));
}

// World position of the scene seen at pixel `px` last frame; w = 0 for sky.
fn scene_point(px:vec2<i32>)->vec4<f32>{
  let size=vec2<i32>(i32(sim.width),i32(sim.height));
  let q=clamp(px,vec2<i32>(0),size-vec2<i32>(1));
  let d=textureLoad(depth_tex,q,0);
  if(d<=0.0){return vec4<f32>(0.0);}
  let ndc=vec2<f32>((f32(q.x)+0.5)/sim.width*2.0-1.0,1.0-(f32(q.y)+0.5)/sim.height*2.0);
  let h=sim.inv_prev_vp*vec4<f32>(ndc,d,1.0);
  return vec4<f32>(h.xyz/h.w,1.0);
}

@compute @workgroup_size(64) fn cs_sim(@builtin(global_invocation_id) id:vec3<u32>) {
  let n=atomicLoad(&counters.alive[sim.cur]);
  if(id.x>=n){return;}
  let pid=lists[N*(1u+sim.cur)+id.x];
  var p=particles[pid];
  let dt=sim.dt;
  p.p0.w+=dt;
  if(p.p0.w>=p.p1.w){
    let d=atomicAdd(&counters.dead,1);
    lists[u32(d)]=pid;
    return;
  }
  var flags=p.p6.y;
  if((flags&REST)==0u && dt>0.0){
    var force=p.p2.xyz;
    let kind=(flags>>12u)&15u;
    if(kind==1u){ force+=cross(p.p3.xyz,p.p0.xyz-p.p4.xyz)*p.p3.w; }
    else if(kind==2u){ let r=p.p0.xyz-p.p3.xyz; force+=select(vec3<f32>(0.0,1.0,0.0),normalize(r),length(r)>0.0001)*p.p3.w; }
    if(p.p4.w>0.0){ force+=flow(p.p0.xyz,sim.time,p.p6.x)*p.p4.w*1.5; }
    var v=(p.p1.xyz+force*dt)*max(0.0,1.0-p.p2.w*dt);
    var pos=p.p0.xyz+v*dt;
    let bounce=bitcast<f32>(p.p6.w);
    if(bounce>=0.0 && sim.depth_valid!=0u){
      let clip=sim.prev_vp*vec4<f32>(pos,1.0);
      if(clip.w>0.05){
        let ndc=clip.xy/clip.w;
        if(abs(ndc.x)<0.999 && abs(ndc.y)<0.999){
          let px=vec2<i32>(i32((ndc.x*0.5+0.5)*sim.width),i32((0.5-ndc.y*0.5)*sim.height));
          let s=scene_point(px);
          if(s.w>0.0){
            let behind=clip.w-(sim.prev_vp*vec4<f32>(s.xyz,1.0)).w;
            let radius=bitcast<f32>(p.p5.w)*0.5;
            let thick=radius+max(0.3,length(v)*dt*2.0);
            if(behind>-radius*0.5 && behind<thick){
              var nrm=vec3<f32>(0.0,1.0,0.0);
              let sx=scene_point(px+vec2<i32>(2,0));
              let sy=scene_point(px+vec2<i32>(0,2));
              if(sx.w>0.0 && sy.w>0.0){
                let c=cross(sx.xyz-s.xyz,sy.xyz-s.xyz);
                if(dot(c,c)>1e-12){ nrm=normalize(c); }
              }
              if(dot(nrm,sim.prev_eye-s.xyz)<0.0){ nrm=-nrm; }
              let vn=dot(v,nrm);
              if(vn<0.0){
                let vt=v-vn*nrm;
                v=vt*(1.0-p.p7.x)-vn*nrm*bounce;
              }
              pos+=nrm*max(dot(s.xyz-pos,nrm)+radius,0.0);
              if(length(v)<0.6 && nrm.y>0.5){ v=vec3<f32>(0.0); flags|=REST; }
            }
          }
        }
      }
    }
    p.p0=vec4<f32>(pos,p.p0.w);
    p.p1=vec4<f32>(v,p.p1.w);
    p.p7.w+=p.p7.z*dt;
    p.p6.y=flags;
  }
  particles[pid]=p;
  let slot=atomicAdd(&counters.alive[1u-sim.cur],1u);
  lists[N*(2u-sim.cur)+slot]=pid;
  var which=2u;
  if((flags&255u)==8u){ which=0u; } else if(((flags>>16u)&15u)==1u){ which=1u; }
  let ds=atomicAdd(&counters.draw[which],1u);
  lists[N*(3u+which)+ds]=pid;
}
"#;

const RENDER_WGSL: &str = r#"
struct Uniforms {
  view_proj:mat4x4<f32>, light_dir:vec3<f32>, ambient:f32, light_color:vec3<f32>, wind:f32,
  eye:vec3<f32>, time:f32, haze_color:vec3<f32>, haze_density:f32, haze_height_m:f32, haze_base_y:f32,
  torch_radius_m:f32, torch_curve:f32, torch_color:vec4<f32>, camera_right:vec3<f32>, ao_strength:f32,
  camera_up:vec3<f32>, gi_strength:f32, sky_up:vec3<f32>, env_strength:f32, sky_horizon:vec3<f32>,
  dither_frame:f32, sky_down:vec3<f32>, season:f32,
};
struct DrawView { inv_vp:mat4x4<f32>, eye:vec3<f32>, pad:f32 };
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(1) @binding(0) var<storage,read> particles:array<Particle>;
@group(1) @binding(1) var<storage,read> lists:array<u32>;
@group(1) @binding(2) var<uniform> view:DrawView;
@group(1) @binding(3) var depth_tex:DEPTH_TYPE;

fn haze(color:vec3<f32>,world_p:vec3<f32>)->f32 {
  if(u.haze_density<=0.0){return 0.0;}
  let d=distance(u.eye,world_p); let h=max(u.haze_height_m,1.0);
  let a0=exp(-max(u.eye.y-u.haze_base_y,0.0)/h); let a1=exp(-max(world_p.y-u.haze_base_y,0.0)/h);
  let rise=world_p.y-u.eye.y; var air=d*a0;
  if(abs(rise)>1.0){air=d*h*(a0-a1)/rise;}
  let optical=air*u.haze_density;
  return clamp(1.0-exp(-optical*optical),0.0,1.0);
}
// Straight ahead, from the screen centre (independent of handedness).
fn forward()->vec3<f32>{
  let h=view.inv_vp*vec4<f32>(0.0,0.0,0.5,1.0);
  return normalize(h.xyz/h.w-u.eye);
}
// Distance along the view of the scene behind this pixel (huge for sky).
fn scene_depth(frag:vec4<f32>)->f32 {
  let d=textureLoad(depth_tex,vec2<i32>(frag.xy),0);
  if(d<=0.0){return 1e9;}
  let dims=vec2<f32>(textureDimensions(depth_tex));
  let ndc=vec2<f32>(frag.x/dims.x*2.0-1.0,1.0-frag.y/dims.y*2.0);
  let h=view.inv_vp*vec4<f32>(ndc,d,1.0);
  return dot(h.xyz/h.w-u.eye,forward());
}
fn sky_light(n:vec3<f32>)->vec3<f32>{
  return u.ambient*mix(u.sky_down,u.sky_up,clamp(n.y*0.5+0.5,0.0,1.0));
}

struct SpriteOut {
  @builtin(position) clip:vec4<f32>,
  @location(0) color:vec4<f32>,
  @location(1) uv:vec2<f32>,
  @location(2) @interpolate(flat) silhouette:u32,
  @location(3) age:f32,
  @location(4) @interpolate(flat) seed:u32,
  @location(5) center:vec3<f32>,
  @location(6) radius:f32,
  @location(7) lighting:f32,
};
fn quad_vertex(i:u32)->vec2<f32>{return array<vec2<f32>,6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(-1.0,1.0),vec2(-1.0,1.0),vec2(1.0,-1.0),vec2(1.0,1.0))[i];}
fn sprite(vi:u32,pid:u32)->SpriteOut {
  let p=particles[pid]; let corner=quad_vertex(vi);
  let age=clamp(p.p0.w/max(p.p1.w,0.001),0.0,1.0);
  let flags=p.p6.y; let profile=(flags>>8u)&15u; var life_scale=1.0;
  if(profile==1u){life_scale=sin(age*3.14159265);}
  else if(profile==2u){life_scale=1.0-age;}
  else if(profile==3u){life_scale=(0.35+age*1.4)*(1.0-smoothstep(0.65,1.0,age));}
  else if(profile==4u){life_scale=1.0+1.5*sqrt(age);}
  let silhouette=flags&255u; let vel=p.p1.xyz; let speed=length(vel); var angle=p.p7.w;
  if(silhouette==2u){let vx=dot(vel,u.camera_right);let vy=dot(vel,u.camera_up);angle=atan2(-vx,vy);}
  else if(silhouette==1u||silhouette==10u){angle=sin(p.p7.w)*0.25;}
  let cs=cos(angle); let sn=sin(angle); var aspect=vec2(1.0);
  let stretch=bitcast<f32>(p.p6.z);
  if(silhouette==1u){aspect=vec2(0.52,1.55);}
  else if(silhouette==2u){aspect=vec2(0.22,1.0+speed*stretch*30.0);}
  else if(silhouette==3u){aspect=vec2(1.18,0.92);}
  else if(silhouette==4u){aspect=vec2(0.45,1.25);}
  else if(silhouette==6u){aspect=vec2(1.0,0.92);}
  else if(silhouette==10u){aspect=vec2(0.8,1.25);}
  let size=bitcast<f32>(p.p5.w)*life_scale;
  let local=corner*aspect; let rotated=vec2(local.x*cs-local.y*sn,local.x*sn+local.y*cs)*size;
  var out:SpriteOut;
  out.clip=u.view_proj*vec4(p.p0.xyz+u.camera_right*rotated.x+u.camera_up*rotated.y,1.0);
  let emissive=bitcast<f32>(p.p5.z);
  var c=mix(unpack4x8unorm(p.p5.x),unpack4x8unorm(p.p5.y),age);
  c=vec4(c.rgb*emissive,c.a*smoothstep(0.0,0.08,age)*(1.0-smoothstep(0.7,1.0,age)));
  out.color=c; out.uv=corner; out.silhouette=silhouette; out.age=age; out.seed=p.p6.x;
  out.center=p.p0.xyz; out.radius=size; out.lighting=p.p7.y;
  return out;
}
@vertex fn vs_alpha(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32)->SpriteOut { return sprite(vi,lists[N*4u+ii]); }
@vertex fn vs_additive(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32)->SpriteOut { return sprite(vi,lists[N*5u+ii]); }

fn sd_circle(p:vec2<f32>,c:vec2<f32>,r:f32)->f32{return length(p-c)-r;}
fn aa(d:f32)->f32{return 1.0-smoothstep(-fwidth(d),fwidth(d),d);}
fn hash2(p:vec2<f32>)->f32{return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
fn vnoise(p:vec2<f32>)->f32{
  let i=floor(p); let f=fract(p); let w=f*f*(3.0-2.0*f);
  return mix(mix(hash2(i),hash2(i+vec2(1.0,0.0)),w.x),mix(hash2(i+vec2(0.0,1.0)),hash2(i+vec2(1.0,1.0)),w.x),w.y);
}
fn fbm(p:vec2<f32>)->f32{ return vnoise(p)*0.55+vnoise(p*2.03+vec2(5.2,1.3))*0.3+vnoise(p*4.1+vec2(1.7,9.2))*0.15; }
// (coverage, brightness) of a sprite's shape at its uv.
fn silhouette_value(i:SpriteOut)->vec2<f32>{
  var p=i.uv; var d=length(p)-0.82; var value=0.65;
  let s=i.silhouette;
  if(s==9u){
    let o=vec2(f32(i.seed&255u),f32((i.seed>>8u)&255u))*0.37;
    let n=fbm(p*1.6+o+vec2(0.0,i.age*0.8));
    let r=length(p);
    let cover=(1.0-smoothstep(0.15,1.0,r))*smoothstep(0.18,0.75,n+0.35*(1.0-r));
    return vec2(cover,0.85+0.3*n);
  }
  if(s==10u){
    let o=vec2(f32(i.seed&255u),f32((i.seed>>8u)&255u))*0.41;
    let n=fbm(vec2(p.x*1.8,p.y*1.2-i.age*3.0)+o);
    let r=length(vec2(p.x,p.y*0.85+0.1));
    let cover=(1.0-smoothstep(0.1,0.95,r))*smoothstep(0.2,0.7,n+0.4*(1.0-r));
    return vec2(cover,0.7+0.9*(1.0-r));
  }
  if(s==1u){let sway=(hash2(vec2(f32(i.seed),floor(i.age*8.0)))-0.5)*0.22;p.x+=sway*(p.y+1.0);let y=(p.y+1.0)*0.5;let width=mix(0.62,0.05,pow(y,0.82))*(0.9+0.1*sin(p.y*10.0+f32(i.seed%17u)));d=abs(p.x)-width;d=max(d,-p.y-0.92);d=max(d,p.y-0.94);value=mix(1.45,0.62,y)+0.18*(1.0-abs(p.x)/max(width,0.05));}
  else if(s==2u){d=max(abs(p.x)-0.13,abs(p.y)-0.88);value=1.25-0.35*abs(p.x);}
  else if(s==3u){let d0=sd_circle(p,vec2(-0.34,-0.02),0.52);let d1=sd_circle(p,vec2(0.30,-0.10),0.58);let d2=sd_circle(p,vec2(-0.05,0.30),0.56);let d3=sd_circle(p,vec2(0.18,0.38),0.40);d=min(min(d0,d1),min(d2,d3));let nz=(hash2(floor((p+vec2(i.age*0.12,0.0))*7.0))-0.5)*0.10;d+=nz;value=0.38+0.22*(1.0-length(p)*0.55);}
  else if(s==4u){d=abs(p.x)+abs(p.y)*0.55-0.62;value=0.72+0.45*step(0.0,p.x*p.y);}
  else if(s==5u){let bars=min(max(abs(p.x)-0.09,abs(p.y)-0.72),max(abs(p.y+0.34*p.x)-0.09,abs(p.x)-0.62));d=bars;value=1.05;}
  else if(s==6u){p.y-=0.04;p.x*=1.06;let q=p*vec2(1.0,1.08);let top=min(sd_circle(q,vec2(-0.34,0.30),0.48),sd_circle(q,vec2(0.34,0.30),0.48));let tip=max(abs(q.x)*0.92+q.y*0.72-0.62,-q.y-0.92);d=min(top,tip);d=max(d,q.y-0.72);value=1.18+0.28*(0.72-q.y);}
  else if(s==7u){let outer=length(p)-0.78;let inner=length(p)-0.58;d=max(outer,-inner);let hl=1.0-smoothstep(0.0,0.22,length(p-vec2(-0.32,0.34)));value=0.55+hl*1.2;}
  else if(s==0u){let r=length(p);return vec2(1.0-smoothstep(0.0,1.0,r),1.0+0.6*(1.0-r));}
  return vec2(aa(d),value);
}
fn shade(i:SpriteOut)->vec4<f32>{
  let fwd=forward();
  let lin=dot(i.center-u.eye,fwd);
  let soft=clamp((scene_depth(i.clip)-lin)/max(i.radius*0.7,0.03),0.0,1.0);
  let near=clamp((lin-0.15)/(i.radius+0.2),0.0,1.0);
  let mv=silhouette_value(i);
  var rgb=i.color.rgb*mv.y;
  if(i.lighting>0.0){
    let uv=i.uv*0.8;
    let n=normalize(u.camera_right*uv.x+u.camera_up*uv.y-fwd*sqrt(max(0.0,1.0-dot(uv,uv))));
    let wrap=clamp(dot(n,normalize(u.light_dir))*0.5+0.5,0.0,1.0);
    let light=sky_light(n)+wrap*wrap*(1.0-u.ambient)*u.light_color;
    rgb=mix(rgb,rgb*light,i.lighting);
  }
  let alpha=i.color.a*mv.x*soft*near;
  return vec4(rgb,alpha);
}
@fragment fn fs_alpha(i:SpriteOut)->@location(0) vec4<f32>{
  let c=shade(i); if(c.a<0.003){discard;}
  let rgb=mix(c.rgb,u.haze_color,haze(c.rgb,i.center));
  return vec4(rgb*c.a,c.a);
}
@fragment fn fs_additive(i:SpriteOut)->@location(0) vec4<f32>{
  let c=shade(i); if(c.a<0.003){discard;}
  return vec4(c.rgb*c.a*(1.0-haze(c.rgb,i.center)),0.0);
}

// Chunks: a jittered icosahedron per particle, tumbling about a random axis.
const ICO_V=array<vec3<f32>,12>(
  vec3(-0.5257,0.8507,0.0),vec3(0.5257,0.8507,0.0),vec3(-0.5257,-0.8507,0.0),vec3(0.5257,-0.8507,0.0),
  vec3(0.0,-0.5257,0.8507),vec3(0.0,0.5257,0.8507),vec3(0.0,-0.5257,-0.8507),vec3(0.0,0.5257,-0.8507),
  vec3(0.8507,0.0,-0.5257),vec3(0.8507,0.0,0.5257),vec3(-0.8507,0.0,-0.5257),vec3(-0.8507,0.0,0.5257));
const ICO_F=array<u32,60>(0u,11u,5u, 0u,5u,1u, 0u,1u,7u, 0u,7u,10u, 0u,10u,11u, 1u,5u,9u, 5u,11u,4u, 11u,10u,2u, 10u,7u,6u, 7u,1u,8u,
  3u,9u,4u, 3u,4u,2u, 3u,2u,6u, 3u,6u,8u, 3u,8u,9u, 4u,9u,5u, 2u,4u,11u, 6u,2u,10u, 8u,6u,7u, 9u,8u,1u);
struct ChunkOut {
  @builtin(position) clip:vec4<f32>,
  @location(0) color:vec3<f32>,
  @location(1) @interpolate(flat) normal:vec3<f32>,
  @location(2) world:vec3<f32>,
};
fn rotate(v:vec3<f32>,axis:vec3<f32>,a:f32)->vec3<f32>{
  let c=cos(a); let s=sin(a);
  return v*c+cross(axis,v)*s+axis*dot(axis,v)*(1.0-c);
}
fn chunk_corner(k:u32,seed:u32,scale:vec3<f32>,axis:vec3<f32>,angle:f32)->vec3<f32>{
  let r=0.7+0.6*random(seed,k*0x9e3779b9u+17u);
  return rotate(ICO_V[k]*r*scale,axis,angle);
}
@vertex fn vs_chunk(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32)->ChunkOut {
  let p=particles[lists[N*3u+ii]];
  let seed=p.p6.x;
  let age=clamp(p.p0.w/max(p.p1.w,0.001),0.0,1.0);
  // Settle into the ground at the end instead of popping away.
  let size=bitcast<f32>(p.p5.w)*0.5*(1.0-smoothstep(0.85,1.0,age));
  let scale=vec3(1.0,0.45+0.4*random(seed,0x51u),0.65+0.5*random(seed,0x52u))*size;
  let axis=random_dir(seed^0x5bd1e995u);
  let face=vi/3u;
  let a=chunk_corner(ICO_F[face*3u],seed,scale,axis,p.p7.w);
  let b=chunk_corner(ICO_F[face*3u+1u],seed,scale,axis,p.p7.w);
  let c=chunk_corner(ICO_F[face*3u+2u],seed,scale,axis,p.p7.w);
  var n=normalize(cross(b-a,c-a));
  if(dot(n,a+b+c)<0.0){n=-n;}
  let local=array<vec3<f32>,3>(a,b,c)[vi%3u];
  var out:ChunkOut;
  out.world=p.p0.xyz+local;
  out.clip=u.view_proj*vec4(out.world,1.0);
  if(size<=0.0){out.clip=vec4(2.0,2.0,2.0,1.0);}
  out.color=mix(unpack4x8unorm(p.p5.x),unpack4x8unorm(p.p5.y),random(seed,0x77u)).rgb;
  out.normal=n;
  return out;
}
@fragment fn fs_chunk(i:ChunkOut)->@location(0) vec4<f32>{
  if(dot(i.normal,u.eye-i.world)<0.0){discard;}
  let d=textureLoad(depth_tex,vec2<i32>(i.clip.xy),0);
  if(d>i.clip.z){discard;}
  let ndl=max(dot(i.normal,normalize(u.light_dir)),0.0);
  let light=sky_light(i.normal)+ndl*(1.0-u.ambient)*u.light_color;
  let rgb=i.color*light;
  return vec4(mix(rgb,u.haze_color,haze(rgb,i.world)),1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_scheduler_preserves_fractional_emission() {
        let mut remainder = 0.0;
        let mut total = 0;
        for _ in 0..60 {
            total += schedule_continuous(&mut remainder, 125.0, 1.0 / 60.0);
        }
        assert_eq!(total, 125);
        assert!(remainder.abs() < 0.000_01);
    }

    #[test]
    fn raw_emitter_carries_collision_and_flags() {
        let e = ParticleEmitter::debris(Vec3::new(1.0, 2.0, 3.0), crate::Color::WHITE, 30, 5.0);
        let raw = to_raw(e, 41);
        assert_eq!(raw.position, [1.0, 2.0, 3.0]);
        assert_eq!(raw.flags & 255, 8);
        assert!(raw.bounce >= 0.0);
        assert_eq!(raw.generation, 41);
        let smoke = to_raw(ParticleEmitter::smoke(Vec3::ZERO, 1.0), 1);
        assert_eq!(smoke.bounce, -1.0);
        assert_eq!((smoke.flags >> 16) & 15, 1);
        assert_eq!(smoke.lighting, 1.0);
    }

    #[test]
    fn lists_start_with_every_particle_free() {
        let lists = initial_lists();
        assert_eq!(lists.len(), MAX_PARTICLES as usize * LISTS as usize);
        assert_eq!(lists[0], 0);
        assert_eq!(lists[MAX_PARTICLES as usize - 1], MAX_PARTICLES - 1);
        assert_eq!(initial_counters()[0], MAX_PARTICLES);
    }
}
