use crate::particles::{
    EmitterId, ParticleBlend, ParticleCommand, ParticleEmitter, ParticleForce, ParticleMode,
    ParticleShape, ParticleSilhouette, SizeOverLife, MAX_PARTICLE_EMITTERS,
};
use std::mem::{offset_of, size_of};
use wgpu::util::DeviceExt;

const MAX_PARTICLES: u32 = 4096;
const PARTICLE_BYTES: u64 = (MAX_PARTICLES as u64) * 144;
const EMITTER_BYTES: u64 = size_of::<EmitterRaw>() as u64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitterRaw {
    position: [f32; 3],
    rate: f32,
    velocity: [f32; 3],
    lifetime: f32,
    spread: [f32; 3],
    size: f32,
    color: [f32; 4],
    secondary_color: [f32; 4],
    turbulence: f32,
    drag: f32,
    seed: u32,
    enabled: u32,
    mode: u32,
    shape: u32,
    burst_count: u32,
    scheduled: u32,
    claimed: u32,
    generation: u32,
    silhouette: u32,
    size_over_life: u32,
    emissive_intensity: f32,
    _alignment: [u32; 3],
    acceleration: [f32; 3],
    velocity_stretch: f32,
    force_vector: [f32; 3],
    force_strength: f32,
    force_kind: u32,
    _pad: [u32; 3],
}

const _: () = {
    assert!(size_of::<EmitterRaw>() == 192);
    assert!(offset_of!(EmitterRaw, position) == 0);
    assert!(offset_of!(EmitterRaw, enabled) == 92);
    assert!(offset_of!(EmitterRaw, scheduled) == 108);
    assert!(offset_of!(EmitterRaw, claimed) == 112);
    assert!(offset_of!(EmitterRaw, generation) == 116);
    assert!(offset_of!(EmitterRaw, emissive_intensity) == 128);
    assert!(offset_of!(EmitterRaw, acceleration) == 144);
};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimRaw {
    dt: f32,
    time: f32,
    _pad: [f32; 2],
}
#[derive(Clone, Copy, Debug)]
struct EmitterRuntime {
    emitter: ParticleEmitter,
    generation: u32,
    emission_remainder: f64,
    scheduled: u32,
}

pub struct ParticleGpu {
    particles: wgpu::Buffer,
    render_particles: wgpu::Buffer,
    emitters: wgpu::Buffer,
    sim: wgpu::Buffer,
    compute_bind: wgpu::BindGroup,
    render_bind: wgpu::BindGroup,
    compute: wgpu::ComputePipeline,
    additive_render: wgpu::RenderPipeline,
    alpha_render: wgpu::RenderPipeline,
    runtimes: [Option<EmitterRuntime>; MAX_PARTICLE_EMITTERS],
    render_lifetime_s: f32,
}
impl ParticleGpu {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        scene: &wgpu::BindGroupLayout,
    ) -> Self {
        let particles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-compute-storage"),
            contents: &vec![0u8; PARTICLE_BYTES as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let render_particles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-render-storage"),
            contents: &vec![0u8; PARTICLE_BYTES as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let emitters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-emitters"),
            contents: &vec![0u8; MAX_PARTICLE_EMITTERS * EMITTER_BYTES as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let sim = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-sim"),
            contents: bytemuck::bytes_of(&SimRaw {
                dt: 0.0,
                time: 0.0,
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-compute-bind-layout"),
            entries: &[
                buffer_entry(0, wgpu::ShaderStages::COMPUTE, false),
                buffer_entry(1, wgpu::ShaderStages::COMPUTE, false),
                uniform_entry(2),
            ],
        });
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-render-bind-layout"),
            entries: &[buffer_entry(3, wgpu::ShaderStages::VERTEX, true)],
        });
        let compute_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle-compute-bind"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: emitters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim.as_entire_binding(),
                },
            ],
        });
        let render_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle-render-bind"),
            layout: &render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 3,
                resource: render_particles.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle-shader"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-compute-pipeline-layout"),
                bind_group_layouts: &[scene, &compute_layout],
                push_constant_ranges: &[],
            });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle-compute"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("particle-render-pipeline-layout"),
                bind_group_layouts: &[scene, &render_layout],
                push_constant_ranges: &[],
            });
        let additive_render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-additive-billboards"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_additive"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: super::DEPTH_COMPARE,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let alpha_render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-alpha-billboards"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_alpha"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: super::DEPTH_COMPARE,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self {
            particles,
            render_particles,
            emitters,
            sim,
            compute_bind,
            render_bind,
            compute,
            additive_render,
            alpha_render,
            runtimes: [None; MAX_PARTICLE_EMITTERS],
            render_lifetime_s: 0.0,
        }
    }
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
        queue.write_buffer(
            &self.sim,
            0,
            bytemuck::bytes_of(&SimRaw {
                dt,
                time,
                _pad: [0.0; 2],
            }),
        );
        for command in commands {
            match command {
                ParticleCommand::Start(id, emitter) => self.start(queue, id, emitter),
                ParticleCommand::UpdatePosition(id, position) => {
                    self.update_position(queue, id, position)
                }
                ParticleCommand::Stop(id) => self.stop(queue, id),
                ParticleCommand::Clear => self.clear(queue),
            }
        }
        self.render_lifetime_s = (self.render_lifetime_s - dt).max(0.0);
        for (slot, runtime) in self.runtimes.iter_mut().enumerate() {
            let Some(runtime) = runtime else { continue };
            let newly_scheduled = match runtime.emitter.mode() {
                ParticleMode::Continuous => {
                    schedule_continuous(&mut runtime.emission_remainder, runtime.emitter.rate(), dt)
                }
                ParticleMode::Burst => 0,
            };
            if newly_scheduled > 0 {
                runtime.scheduled = runtime
                    .scheduled
                    .checked_add(newly_scheduled)
                    .expect("particle scheduled counter exhausted");
                queue.write_buffer(
                    &self.emitters,
                    slot as u64 * EMITTER_BYTES + offset_of!(EmitterRaw, scheduled) as u64,
                    bytemuck::bytes_of(&runtime.scheduled),
                );
                self.render_lifetime_s = self.render_lifetime_s.max(runtime.emitter.lifetime_s());
            }
        }
    }
    fn start(&mut self, queue: &wgpu::Queue, id: EmitterId, emitter: ParticleEmitter) {
        let slot = id.slot() as usize;
        assert!(
            slot < MAX_PARTICLE_EMITTERS,
            "particle emitter slot out of range"
        );
        assert!(
            self.runtimes[slot].is_none(),
            "particle GPU emitter slot started while occupied"
        );
        let scheduled = if emitter.mode() == ParticleMode::Burst {
            emitter.burst_count()
        } else {
            0
        };
        self.runtimes[slot] = Some(EmitterRuntime {
            emitter,
            generation: id.generation(),
            emission_remainder: 0.0,
            scheduled,
        });
        queue.write_buffer(
            &self.emitters,
            slot as u64 * EMITTER_BYTES,
            bytemuck::bytes_of(&to_raw(emitter, id.generation(), scheduled)),
        );
        if scheduled > 0 {
            self.render_lifetime_s = self.render_lifetime_s.max(emitter.lifetime_s());
        }
    }
    fn update_position(&mut self, queue: &wgpu::Queue, id: EmitterId, position: glam::Vec3) {
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
            slot as u64 * EMITTER_BYTES,
            bytemuck::bytes_of(&<[f32; 3]>::from(position)),
        );
    }
    fn stop(&mut self, queue: &wgpu::Queue, id: EmitterId) {
        let slot = id.slot() as usize;
        let runtime = self
            .runtimes
            .get_mut(slot)
            .and_then(Option::take)
            .expect("particle GPU received stop for unoccupied slot");
        assert_eq!(
            runtime.generation,
            id.generation(),
            "particle GPU received stale stop command"
        );
        queue.write_buffer(
            &self.emitters,
            slot as u64 * EMITTER_BYTES + offset_of!(EmitterRaw, enabled) as u64,
            bytemuck::bytes_of(&0u32),
        );
    }
    fn clear(&mut self, queue: &wgpu::Queue) {
        self.runtimes.fill(None);
        queue.write_buffer(
            &self.emitters,
            0,
            &vec![0u8; MAX_PARTICLE_EMITTERS * EMITTER_BYTES as usize],
        );
        self.render_lifetime_s = 0.0;
    }
    pub fn encode_compute(&self, encoder: &mut wgpu::CommandEncoder, scene: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle-simulation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute);
        pass.set_bind_group(0, scene, &[]);
        pass.set_bind_group(1, &self.compute_bind, &[]);
        pass.dispatch_workgroups(MAX_PARTICLES.div_ceil(64), 1, 1);
        drop(pass);
        encoder.copy_buffer_to_buffer(
            &self.particles,
            0,
            &self.render_particles,
            0,
            PARTICLE_BYTES,
        );
    }
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, scene: &wgpu::BindGroup) {
        if self.render_lifetime_s <= 0.0 {
            return;
        }
        pass.set_bind_group(0, scene, &[]);
        pass.set_bind_group(1, &self.render_bind, &[]);
        pass.set_pipeline(&self.alpha_render);
        pass.draw(0..6, 0..MAX_PARTICLES);
        pass.set_pipeline(&self.additive_render);
        pass.draw(0..6, 0..MAX_PARTICLES);
    }
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

fn buffer_entry(
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
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn to_raw(e: ParticleEmitter, generation: u32, scheduled: u32) -> EmitterRaw {
    let c = e.color();
    let s = e.secondary_color();
    EmitterRaw {
        position: e.position().into(),
        rate: e.rate(),
        velocity: e.velocity().into(),
        lifetime: e.lifetime_s(),
        spread: e.spread().into(),
        size: e.size(),
        color: [c.r, c.g, c.b, c.a],
        secondary_color: [s.r, s.g, s.b, s.a],
        turbulence: e.turbulence(),
        drag: e.drag(),
        seed: e.seed(),
        enabled: 1,
        mode: match e.mode() {
            ParticleMode::Continuous => 0,
            ParticleMode::Burst => 1,
        },
        shape: match e.shape() {
            ParticleShape::Point => 0,
            ParticleShape::Sphere => 1,
            ParticleShape::Cone => 2,
            ParticleShape::Ring => 3,
        },
        burst_count: e.burst_count(),
        scheduled,
        claimed: 0,
        generation,
        silhouette: match e.silhouette() {
            ParticleSilhouette::SoftOrb => 0,
            ParticleSilhouette::Flame => 1,
            ParticleSilhouette::SparkStreak => 2,
            ParticleSilhouette::SmokeCloud => 3,
            ParticleSilhouette::Shard => 4,
            ParticleSilhouette::RuneMote => 5,
            ParticleSilhouette::Heart => 6,
            ParticleSilhouette::Bubble => 7,
        },
        size_over_life: match e.size_over_life() {
            SizeOverLife::Constant => 0,
            SizeOverLife::FadeInOut => 1,
            SizeOverLife::Shrink => 2,
            SizeOverLife::GrowThenFade => 3,
        },
        emissive_intensity: e.emissive_intensity(),
        _alignment: [0; 3],
        acceleration: e.acceleration().into(),
        velocity_stretch: e.velocity_stretch(),
        force_vector: match e.force() {
            ParticleForce::None => [0.0; 3],
            ParticleForce::Vortex { axis, .. } => axis.normalize().into(),
            ParticleForce::Radial { center, .. } => center.into(),
        },
        force_strength: match e.force() {
            ParticleForce::None => 0.0,
            ParticleForce::Vortex { strength, .. } | ParticleForce::Radial { strength, .. } => {
                strength
            }
        },
        force_kind: match e.force() {
            ParticleForce::None => 0,
            ParticleForce::Vortex { .. } => 1,
            ParticleForce::Radial { .. } => 2,
        },
        _pad: [
            match e.blend() {
                ParticleBlend::Additive => 0,
                ParticleBlend::Alpha => 1,
            },
            0,
            0,
        ],
    }
}

const PARTICLE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> scene: array<vec4<f32>,14>;
struct Particle { pos:vec4<f32>, vel:vec4<f32>, state:vec4<f32>, color:vec4<f32>, secondary:vec4<f32>, owner:vec4<u32>, acceleration:vec4<f32>, force:vec4<f32>, origin:vec4<f32> };
struct Emitter { position:vec3<f32>,rate:f32, velocity:vec3<f32>,lifetime:f32, spread:vec3<f32>,size:f32, color:vec4<f32>,secondary_color:vec4<f32>, turbulence:f32,drag:f32,seed:u32,enabled:u32, mode:u32,shape:u32,burst_count:u32,scheduled:u32,claimed:atomic<u32>, generation:u32,silhouette:u32,size_over_life:u32,emissive_intensity:f32,_alignment:array<u32,3>, acceleration:vec3<f32>,velocity_stretch:f32, force_vector:vec3<f32>,force_strength:f32, force_kind:u32,_pad:array<u32,3> };
struct Sim {dt:f32,time:f32,_pad:vec2<f32>};
@group(1) @binding(0) var<storage,read_write> particles_compute:array<Particle>;
@group(1) @binding(1) var<storage,read_write> emitters:array<Emitter>;
@group(1) @binding(2) var<uniform> sim:Sim;
@group(1) @binding(3) var<storage,read> particles_render:array<Particle>;
fn hash_u32(v:u32)->u32 { var x=v; x^=x>>16u; x*=0x7feb352du; x^=x>>15u; x*=0x846ca68bu; x^=x>>16u; return x; }
fn random(seed:u32,salt:u32)->f32{return f32(hash_u32(seed^salt))/4294967296.0;}
fn random_dir(seed:u32)->vec3<f32>{let z=random(seed,0x91e10da5u)*2.0-1.0;let a=random(seed,0x6c8e9cf5u)*6.283185307;let r=sqrt(max(0.0,1.0-z*z));return vec3<f32>(r*cos(a),z,r*sin(a));}
fn spawn_offset(shape:u32,spread:vec3<f32>,seed:u32)->vec3<f32>{
 if(shape==0u){return vec3<f32>(0.0);}
 if(shape==1u){return random_dir(seed)*pow(random(seed,0x27d4eb2du),1.0/3.0)*spread;}
 let a=random(seed,0x165667b1u)*6.283185307;
 if(shape==3u){return vec3<f32>(cos(a)*spread.x,0.0,sin(a)*spread.z);}
 let h=pow(random(seed,0xd3a2646cu),1.0/3.0);let r=sqrt(random(seed,0xfd7046c5u))*h;
 return vec3<f32>(cos(a)*r*spread.x,h*spread.y,sin(a)*r*spread.z);
}
fn claim_emitter()->vec2<u32>{for(var slot=0u;slot<arrayLength(&emitters);slot++){let e=&emitters[slot];if((*e).enabled==0u){continue;}var claimed=atomicLoad(&(*e).claimed);loop{if(claimed>=(*e).scheduled){break;}let result=atomicCompareExchangeWeak(&(*e).claimed,claimed,claimed+1u);if(result.exchanged){return vec2<u32>(slot,claimed);}claimed=result.old_value;}}return vec2<u32>(0xffffffffu,0u);}
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=arrayLength(&particles_compute)){return;}var p=particles_compute[i];
 if(p.state.y>0.0){p.state.x+=sim.dt;p.state.y-=sim.dt;let n=random_dir(hash_u32(p.owner.z^bitcast<u32>(p.state.x)));var force=p.acceleration.xyz;if(((p.owner.w>>16u)&255u)==1u){force+=cross(normalize(p.force.xyz),p.pos.xyz-p.origin.xyz)*p.force.w;}else if(((p.owner.w>>16u)&255u)==2u){let radial=p.pos.xyz-p.force.xyz;force+=normalize(select(vec3<f32>(0.0,1.0,0.0),radial,length(radial)>0.0001))*p.force.w;}p.vel=vec4<f32>(p.vel.xyz+(n*bitcast<f32>(p.owner.y)+force)*sim.dt,0.0);p.vel=vec4<f32>(p.vel.xyz*max(0.0,1.0-bitcast<f32>(p.owner.x)*sim.dt),0.0);p.pos+=vec4<f32>(p.vel.xyz*sim.dt,0.0);}
 if(p.state.y<=0.0){let claim=claim_emitter();let slot=claim.x;if(slot!=0xffffffffu){let e=emitters[slot];let serial=claim.y;let seed=hash_u32(e.seed^serial^e.generation);let offset=spawn_offset(e.shape,e.spread,seed);p.pos=vec4<f32>(e.position+offset,1.0);p.vel=vec4<f32>(e.velocity+random_dir(seed^0xa511e9b3u)*e.spread*0.25,0.0);p.state=vec4<f32>(0.0,e.lifetime,e.size,e.velocity_stretch);p.color=vec4<f32>(e.color.rgb*e.emissive_intensity,e.color.a);p.secondary=vec4<f32>(e.secondary_color.rgb*e.emissive_intensity,e.secondary_color.a);p.owner=vec4<u32>(bitcast<u32>(e.drag),bitcast<u32>(e.turbulence),seed,e.silhouette|(e.size_over_life<<8u)|(e.force_kind<<16u)|(e._pad[0]<<24u));p.acceleration=vec4<f32>(e.acceleration,0.0);p.force=vec4<f32>(e.force_vector,e.force_strength);p.origin=vec4<f32>(e.position,1.0);}}
 particles_compute[i]=p;}
struct VOut {
 @builtin(position) clip: vec4<f32>,
 @location(0) color: vec4<f32>,
 @location(1) uv: vec2<f32>,
 @location(2) @interpolate(flat) silhouette: u32,
 @location(3) age: f32,
 @location(4) @interpolate(flat) seed: u32,
 @location(5) @interpolate(flat) blend: u32,
};
fn quad_vertex(i:u32)->vec2<f32>{return array<vec2<f32>,6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(-1.0,1.0),vec2(-1.0,1.0),vec2(1.0,-1.0),vec2(1.0,1.0))[i];}
@vertex fn vs_main(@builtin(vertex_index) vi:u32,@builtin(instance_index) ii:u32)->VOut {
 let particle=particles_render[ii]; let corner=quad_vertex(vi);
 let age=clamp(particle.state.x/max(particle.state.x+particle.state.y,0.001),0.0,1.0);
 let profile=(particle.owner.w>>8u)&255u; var life_scale=1.0;
 if(profile==1u){life_scale=sin(age*3.14159265);}else if(profile==2u){life_scale=1.0-age;}else if(profile==3u){life_scale=(0.35+age*1.4)*(1.0-smoothstep(0.65,1.0,age));}
 let silhouette=particle.owner.w&255u; let speed=length(particle.vel.xyz); var angle=0.0;
 if(silhouette==2u){let vx=dot(particle.vel.xyz,scene[10].xyz);let vy=dot(particle.vel.xyz,scene[11].xyz);angle=atan2(-vx,vy);}else if(silhouette==4u||silhouette==5u){angle=random(particle.owner.z,0x63d83595u)*6.283185307+age*1.2;}
 let cs=cos(angle);let sn=sin(angle); var aspect=vec2(1.0);
 if(silhouette==1u){aspect=vec2(0.52,1.55);}else if(silhouette==2u){aspect=vec2(0.22,1.0+speed*particle.state.w);}else if(silhouette==3u){aspect=vec2(1.18,0.92);}else if(silhouette==4u){aspect=vec2(0.45,1.25);}else if(silhouette==6u){aspect=vec2(1.0,0.92);}
 let local=corner*aspect;let rotated=vec2(local.x*cs-local.y*sn,local.x*sn+local.y*cs)*particle.state.z*life_scale;
 var out:VOut;out.clip=mat4x4<f32>(scene[0],scene[1],scene[2],scene[3])*vec4(particle.pos.xyz+scene[10].xyz*rotated.x+scene[11].xyz*rotated.y,1.0);
 if(particle.state.y<=0.0){out.clip=vec4(2.0,2.0,2.0,1.0);}out.color=mix(particle.color,particle.secondary,age);out.color.a*=smoothstep(0.0,0.12,age)*(1.0-smoothstep(0.72,1.0,age));out.uv=corner;out.silhouette=silhouette;out.age=age;out.seed=particle.owner.z;out.blend=(particle.owner.w>>24u)&255u;return out;
}
fn sd_ellipse(p:vec2<f32>,r:vec2<f32>)->f32{return (length(p/r)-1.0)*min(r.x,r.y);}
fn circle(p:vec2<f32>,c:vec2<f32>,r:f32)->f32{return length(p-c)-r;}
fn aa(d:f32)->f32{return 1.0-smoothstep(-fwidth(d),fwidth(d),d);}
fn hash2(p:vec2<f32>)->f32{return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
fn silhouette_value(i:VOut)->vec2<f32>{
 var p=i.uv;var d=length(p)-0.82;var value=0.65;
 if(i.silhouette==1u){let sway=(hash2(vec2(f32(i.seed),floor(i.age*8.0)))-0.5)*0.22;p.x+=sway*(p.y+1.0);let y=(p.y+1.0)*0.5;let width=mix(0.62,0.05,pow(y,0.82))*(0.9+0.1*sin(p.y*10.0+f32(i.seed%17u)));d=abs(p.x)-width;d=max(d,-p.y-0.92);d=max(d,p.y-0.94);value=mix(1.45,0.62,y)+0.18*(1.0-abs(p.x)/max(width,0.05));
 }else if(i.silhouette==2u){d=max(abs(p.x)-0.13,abs(p.y)-0.88);value=1.25-0.35*abs(p.x);
 }else if(i.silhouette==3u){let d0=circle(p,vec2(-0.34,-0.02),0.52);let d1=circle(p,vec2(0.30,-0.10),0.58);let d2=circle(p,vec2(-0.05,0.30),0.56);let d3=circle(p,vec2(0.18,0.38),0.40);d=min(min(d0,d1),min(d2,d3));let noise=(hash2(floor((p+vec2(i.age*0.12,0.0))*7.0))-0.5)*0.10;d+=noise;value=0.38+0.22*(1.0-length(p)*0.55);
 }else if(i.silhouette==4u){d=abs(p.x)+abs(p.y)*0.55-0.62;value=0.72+0.45*step(0.0,p.x*p.y);
 }else if(i.silhouette==5u){let bars=min(max(abs(p.x)-0.09,abs(p.y)-0.72),max(abs(p.y+0.34*p.x)-0.09,abs(p.x)-0.62));d=bars;value=1.05;
 }else if(i.silhouette==6u){p.y-=0.04;p.x*=1.06;let q=p*vec2(1.0,1.08);let top=min(circle(q,vec2(-0.34,0.30),0.48),circle(q,vec2(0.34,0.30),0.48));let tip=max(abs(q.x)*0.92+q.y*0.72-0.62,-q.y-0.92);d=min(top,tip);d=max(d,q.y-0.72);value=1.18+0.28*(0.72-q.y);
 }else if(i.silhouette==7u){let outer=length(p)-0.78;let inner=length(p)-0.58;d=max(outer,-inner);let highlight=1.0-smoothstep(0.0,0.22,length(p-vec2(-0.32,0.34)));value=0.55+highlight*1.2;
 }return vec2(aa(d),value);
}
fn shade(i:VOut)->vec4<f32>{let mv=silhouette_value(i);let alpha=i.color.a*mv.x;return vec4(i.color.rgb*mv.y,alpha);}
@fragment fn fs_additive(i:VOut)->@location(0) vec4<f32>{if(i.blend!=0u){discard;}let c=shade(i);if(c.a<0.005){discard;}return c;}
@fragment fn fs_alpha(i:VOut)->@location(0) vec4<f32>{if(i.blend!=1u){discard;}let c=shade(i);if(c.a<0.005){discard;}return vec4(c.rgb*c.a,c.a);}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_layout_matches_wgsl_storage_layout() {
        assert_eq!(size_of::<EmitterRaw>(), 192);
        assert_eq!(offset_of!(EmitterRaw, position), 0);
        assert_eq!(offset_of!(EmitterRaw, enabled), 92);
        assert_eq!(offset_of!(EmitterRaw, scheduled), 108);
        assert_eq!(offset_of!(EmitterRaw, claimed), 112);
        assert_eq!(offset_of!(EmitterRaw, generation), 116);
        assert_eq!(offset_of!(EmitterRaw, emissive_intensity), 128);
        assert_eq!(offset_of!(EmitterRaw, acceleration), 144);
    }

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
    fn burst_raw_schedules_exact_requested_count() {
        let emitter = ParticleEmitter::new(glam::Vec3::new(1.0, 2.0, 3.0))
            .with_mode(ParticleMode::Burst)
            .with_burst_count(317)
            .with_emissive_intensity(2.75);
        let raw = to_raw(emitter, 41, emitter.burst_count());
        assert_eq!(raw.position, [1.0, 2.0, 3.0]);
        assert_eq!(raw.enabled, 1);
        assert_eq!(raw.scheduled, 317);
        assert_eq!(raw.claimed, 0);
        assert_eq!(raw.generation, 41);
        assert_eq!(raw.emissive_intensity, 2.75);
    }
}
