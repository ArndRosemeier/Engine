//! Far-to-near order for alpha-blended particle sprites.
//!
//! The simulation appends survivors to the alpha draw list with atomics, so
//! the list comes out in a different order every frame. Alpha blending is
//! order dependent: big overlapping sprites (a cloud of puffs, thick smoke)
//! flickered as their order changed from frame to frame. This sorts the list
//! on the GPU every frame, far to near from the camera, ties broken by
//! particle id, so the same scene always blends the same way.
//!
//! A bitonic sort over (key, particle id) pairs, padded to a power of two
//! (at least one block of 512):
//! 1. `cs_sort_args` sizes every dispatch from the alpha count (a stage
//!    whose width is past the padded count gets zero workgroups).
//! 2. `cs_sort_local` builds the keys and sorts each block of 512 in
//!    workgroup memory.
//! 3. For each wider stage `k`: `cs_sort_global` passes for the strides of
//!    512 and up, then `cs_sort_local` merges the strides below 512.
//!
//! The vertex shader reads the particle ids from [`ParticleSort::sorted`].

/// Elements a workgroup sorts in its own memory.
const BLOCK: u32 = 512;
/// Bytes of one indirect dispatch entry (three counts and padding).
const ARGS_STRIDE: u64 = 16;

#[derive(Clone, Copy)]
enum Kind {
    Build,
    Global,
    Merge,
}

struct Pass {
    kind: Kind,
    /// Byte offset of this pass's (k, j) in the params buffer.
    params: u32,
    /// Byte offset of its dispatch size in the args buffer.
    args: u64,
}

pub struct ParticleSort {
    sorted: wgpu::Buffer,
    sort_args: wgpu::Buffer,
    params: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    args_bind: wgpu::BindGroup,
    bind: Option<wgpu::BindGroup>,
    size_args: wgpu::ComputePipeline,
    build: wgpu::ComputePipeline,
    global: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    passes: Vec<Pass>,
}

impl ParticleSort {
    /// `capacity` is the length of each particle list (a power of two, at
    /// least 512). The alpha list is list 4 of the list buffer, and its count
    /// is word 9 of the indirect args (the alpha draw's instance count).
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        assert!(
            capacity.is_power_of_two() && capacity >= BLOCK,
            "particle sort capacity must be a power of two of at least {BLOCK}"
        );
        let stages = capacity.trailing_zeros() - BLOCK.trailing_zeros();
        let sorted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-sorted-alpha"),
            size: u64::from(capacity) * 8,
            // COPY_SRC lets the tests read it back.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sort_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-sort-args"),
            size: u64::from(stages + 1) * ARGS_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });

        // Every pass with its (k, j): the build, then per wider stage the
        // global strides down to BLOCK and one merge below it.
        let mut steps: Vec<(Kind, u32, u32, u64)> = vec![(Kind::Build, BLOCK, 0, 0)];
        for s in 1..=stages {
            let k = BLOCK << s;
            let mut j = k / 2;
            while j >= BLOCK {
                steps.push((Kind::Global, k, j, u64::from(s) * ARGS_STRIDE));
                j /= 2;
            }
            steps.push((Kind::Merge, k, BLOCK / 2, u64::from(s) * ARGS_STRIDE));
        }
        let stride = device.limits().min_uniform_buffer_offset_alignment.max(16);
        let mut bytes = vec![0u8; steps.len() * stride as usize];
        let mut passes = Vec::with_capacity(steps.len());
        for (i, &(kind, k, j, args)) in steps.iter().enumerate() {
            let at = i * stride as usize;
            bytes[at..at + 4].copy_from_slice(&k.to_le_bytes());
            bytes[at + 4..at + 8].copy_from_slice(&j.to_le_bytes());
            passes.push(Pass {
                kind,
                params: (i as u32) * stride,
                args,
            });
        }
        let params = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("particle-sort-params"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );

        let cs = wgpu::ShaderStages::COMPUTE;
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32, dynamic: bool, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-sort-layout"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, false),
                uniform(4, false, 80),
                uniform(5, true, 16),
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-sort-args-layout"),
            entries: &[storage(0, false)],
        });
        let args_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle-sort-args-bind"),
            layout: &args_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sort_args.as_entire_binding(),
            }],
        });
        let src = SORT_WGSL
            .replace("CAPACITY", &capacity.to_string())
            .replace("STAGE_COUNT", &stages.to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle-sort"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let main_only = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle-sort-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let with_args = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle-sort-args-pipeline-layout"),
            bind_group_layouts: &[&layout, &args_layout],
            push_constant_ranges: &[],
        });
        let compute = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: crate::render::pipeline_cache::get(device).as_ref(),
            })
        };
        Self {
            size_args: compute("cs_sort_args", &with_args),
            build: compute("cs_sort_build", &main_only),
            global: compute("cs_sort_global", &main_only),
            merge: compute("cs_sort_merge", &main_only),
            sorted,
            sort_args,
            params,
            layout,
            args_bind,
            bind: None,
            passes,
        }
    }

    /// The sorted (key, particle id) pairs, far to near; the first `count`
    /// are the alpha sprites.
    pub fn sorted(&self) -> &wgpu::Buffer {
        &self.sorted
    }

    /// Point the sort at the particle buffers. `view` is a uniform whose
    /// bytes 64..76 hold the camera position.
    pub fn bind(
        &mut self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        lists: &wgpu::Buffer,
        draw_args: &wgpu::Buffer,
        view: &wgpu::Buffer,
    ) {
        fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource }
        }
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle-sort-bind"),
            layout: &self.layout,
            entries: &[
                entry(0, particles.as_entire_binding()),
                entry(1, lists.as_entire_binding()),
                entry(2, draw_args.as_entire_binding()),
                entry(3, self.sorted.as_entire_binding()),
                entry(4, view.as_entire_binding()),
                entry(
                    5,
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.params,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                ),
            ],
        }));
    }

    /// Sort this frame's alpha list (after the simulation, before drawing).
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let bind = self.bind.as_ref().expect("particle sort used before bind");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle-sort-size"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, bind, &[0]);
            pass.set_bind_group(1, &self.args_bind, &[]);
            pass.set_pipeline(&self.size_args);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle-sort"),
            timestamp_writes: None,
        });
        for p in &self.passes {
            pass.set_pipeline(match p.kind {
                Kind::Build => &self.build,
                Kind::Global => &self.global,
                Kind::Merge => &self.merge,
            });
            pass.set_bind_group(0, bind, &[p.params]);
            pass.dispatch_workgroups_indirect(&self.sort_args, p.args);
        }
    }

    /// Passes recorded per frame (for tests).
    #[cfg(test)]
    fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

const SORT_WGSL: &str = r#"
const N:u32=CAPACITYu;
const STAGES:u32=STAGE_COUNTu;
struct DrawView { inv_vp:mat4x4<f32>, eye:vec3<f32>, pad:f32 };
struct SortPass { k:u32, j:u32, pad0:u32, pad1:u32 };
@group(0) @binding(0) var<storage,read> particles:array<array<vec4<f32>,8>>;
@group(0) @binding(1) var<storage,read> lists:array<u32>;
@group(0) @binding(2) var<storage,read> draw_args:array<u32>;
@group(0) @binding(3) var<storage,read_write> sorted:array<vec2<u32>>;
@group(0) @binding(4) var<uniform> view:DrawView;
@group(0) @binding(5) var<uniform> sp:SortPass;
@group(1) @binding(0) var<storage,read_write> sort_args:array<vec4<u32>>;

var<workgroup> sh:array<vec2<u32>,512>;

fn alpha_count()->u32 { return min(draw_args[9],N); }
fn padded(c:u32)->u32 {
  if(c<=512u){ return 512u; }
  return 1u<<(32u-countLeadingZeros(c-1u));
}
// a goes before b: farther first (smaller key), then lower particle id.
fn before(a:vec2<u32>,b:vec2<u32>)->bool { return a.x<b.x || (a.x==b.x && a.y<b.y); }
// Padding sorts last; a sprite's key is its squared distance, inverted.
fn entry(i:u32)->vec2<u32> {
  if(i>=alpha_count()){ return vec2<u32>(0xffffffffu,0xffffffffu); }
  let pid=lists[N*4u+i];
  let d=particles[pid][0].xyz-view.eye;
  return vec2<u32>(~bitcast<u32>(dot(d,d)),pid);
}

@compute @workgroup_size(1) fn cs_sort_args() {
  let c=alpha_count();
  let p=padded(c);
  let groups=select(p/512u,0u,c==0u);
  for(var s=0u;s<=STAGES;s++){
    sort_args[s]=vec4<u32>(select(0u,groups,(512u<<s)<=p),1u,1u,0u);
  }
}

fn swap_in_block(t:u32,base:u32,k:u32,j:u32){
  let i=2u*t-(t&(j-1u));
  let l=i+j;
  let a=sh[i]; let b=sh[l];
  let up=((base+i)&k)==0u;
  if(select(before(a,b),before(b,a),up)){ sh[i]=b; sh[l]=a; }
}

@compute @workgroup_size(256) fn cs_sort_build(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) t:u32) {
  let base=w.x*512u;
  sh[t]=entry(base+t);
  sh[t+256u]=entry(base+t+256u);
  workgroupBarrier();
  for(var k=2u;k<=512u;k=k<<1u){
    for(var j=k>>1u;j>0u;j=j>>1u){
      swap_in_block(t,base,k,j);
      workgroupBarrier();
    }
  }
  sorted[base+t]=sh[t];
  sorted[base+t+256u]=sh[t+256u];
}

@compute @workgroup_size(256) fn cs_sort_merge(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) t:u32) {
  let base=w.x*512u;
  sh[t]=sorted[base+t];
  sh[t+256u]=sorted[base+t+256u];
  workgroupBarrier();
  for(var j=256u;j>0u;j=j>>1u){
    swap_in_block(t,base,sp.k,j);
    workgroupBarrier();
  }
  sorted[base+t]=sh[t];
  sorted[base+t+256u]=sh[t+256u];
}

@compute @workgroup_size(256) fn cs_sort_global(@builtin(global_invocation_id) g:vec3<u32>) {
  let t=g.x; let j=sp.j; let k=sp.k;
  let i=2u*t-(t&(j-1u));
  let l=i+j;
  let a=sorted[i]; let b=sorted[l];
  if(select(before(a,b),before(b,a),(i&k)==0u)){ sorted[i]=b; sorted[l]=a; }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    fn gpu_tests_enabled() -> bool {
        std::env::var("ENGINE_GPU_TESTS").is_ok_and(|v| v == "1")
    }

    /// Sorts a scrambled alpha list on the GPU and reads it back: far to
    /// near, ties by id, and the same result however the list was ordered.
    #[test]
    fn alpha_list_sorts_far_to_near_on_the_gpu() {
        if !gpu_tests_enabled() {
            return;
        }
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&Default::default())).expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");
        let capacity = 4096u32;
        let mut sort = ParticleSort::new(&device, capacity);
        // One build, then for k = 1024, 2048, 4096 the global strides from
        // k/2 down to 512 and a merge.
        assert_eq!(sort.pass_count(), 1 + (1 + 2 + 3) + 3);

        let mut run = |count: u32, order: &[u32], positions: &[[f32; 3]]| -> Vec<[u32; 2]> {
            let mut parts = vec![[0f32; 32]; capacity as usize];
            for (i, p) in positions.iter().enumerate() {
                parts[i][..3].copy_from_slice(p);
            }
            let particles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&parts),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let mut lists = vec![0u32; capacity as usize * 6];
            for (i, &pid) in order.iter().enumerate() {
                lists[capacity as usize * 4 + i] = pid;
            }
            let lists = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&lists),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let mut args = [0u32; 16];
            args[9] = count;
            let args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&args),
                usage: wgpu::BufferUsages::STORAGE,
            });
            // The camera at the origin.
            let view = [0f32; 20];
            let view = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&view),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            sort.bind(&device, &particles, &lists, &args, &view);
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(capacity) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            sort.encode(&mut encoder);
            encoder.copy_buffer_to_buffer(sort.sorted(), 0, &read, 0, u64::from(capacity) * 8);
            queue.submit([encoder.finish()]);
            read.slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
            device.poll(wgpu::PollType::Wait).expect("poll");
            let out: Vec<[u32; 2]> =
                bytemuck::cast_slice(&read.slice(..).get_mapped_range()).to_vec();
            out[..count as usize].to_vec()
        };

        for count in [1u32, 7, 511, 512, 513, 1500, 4096] {
            // Distances with plenty of ties (every 3 particles share one).
            let positions: Vec<[f32; 3]> = (0..count)
                .map(|i| [((i * 7919) % 97 / 3) as f32, 1.0, 0.0])
                .collect();
            let forward: Vec<u32> = (0..count).collect();
            let mut scrambled = forward.clone();
            let mut x = 0x9e37_79b9u32;
            for i in (1..scrambled.len()).rev() {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                scrambled.swap(i, x as usize % (i + 1));
            }
            let a = run(count, &forward, &positions);
            let b = run(count, &scrambled, &positions);
            assert_eq!(a, b, "{count}: order must not depend on the list order");
            let dist = |pid: u32| {
                let p = positions[pid as usize];
                p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
            };
            let mut ids: Vec<u32> = a.iter().map(|e| e[1]).collect();
            for w in a.windows(2) {
                let (d0, d1) = (dist(w[0][1]), dist(w[1][1]));
                assert!(
                    d0 > d1 || (d0 == d1 && w[0][1] < w[1][1]),
                    "{count}: {w:?} out of order"
                );
            }
            ids.sort_unstable();
            assert_eq!(ids, forward, "{count}: every sprite drawn once");
        }
    }
}
