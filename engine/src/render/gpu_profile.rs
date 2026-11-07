//! Per-phase GPU timings from timestamp queries.
//!
//! Enabled with `ENGINE_GPU_PROFILE_EVERY=N`. While enabled every frame is
//! timed (the readback waits for the GPU, so frame pacing is not
//! representative), and the average of the last `N` frames is printed:
//!
//! `gpu_profile frames=N voxel=… shadow=… gi=… ao=… scene=… water=… taa=… post=… ui=… total=…`
//!
//! Where the GPU can stamp inside render passes, a second line splits the
//! scene pass: sky, land (clipmap), grass, voxel, meshes, batches (instanced
//! and LOD), terrain, skinned, transparent, effects.
//!
//! `voxel` is voxel-terrain meshing, `gi` the voxel rebuild and probe update, `ao` the depth prepass plus
//! the SSAO passes.

/// Timestamp slots, written in this order during a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Start = 0,
    VoxelDone = 1,
    ShadowDone = 2,
    GiDone = 3,
    AoDone = 4,
    SceneDone = 5,
    WaterDone = 6,
    WorldDone = 7,
    PostDone = 8,
    UiDone = 9,
}

const QUERY_COUNT: u32 = 10;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * wgpu::QUERY_SIZE as u64;
const PHASES: [&str; 9] = [
    "voxel", "shadow", "gi", "ao", "scene", "water", "taa", "post", "ui",
];

/// Stamps inside the main scene pass (needs `TIMESTAMP_QUERY_INSIDE_PASSES`),
/// each closing the part named in [`SCENE_PARTS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneMark {
    Begin = 0,
    Sky = 1,
    Land = 2,
    Grass = 3,
    Voxel = 4,
    Meshes = 5,
    Batches = 6,
    Terrain = 7,
    Skinned = 8,
    Transparent = 9,
    End = 10,
}

const SCENE_COUNT: u32 = 11;
const SCENE_BYTES: u64 = SCENE_COUNT as u64 * wgpu::QUERY_SIZE as u64;
const SCENE_PARTS: [&str; 10] = [
    "sky", "land", "grass", "voxel", "meshes", "batches", "terrain", "skinned", "transparent",
    "effects",
];

struct SceneQueries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    sums_ms: [f64; 10],
}

pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    read_buffer: wgpu::Buffer,
    timestamp_period_ns: f32,
    every_frames: u32,
    sums_ms: [f64; 9],
    frames: u32,
    scene: Option<SceneQueries>,
}

impl GpuProfiler {
    pub fn requested_interval() -> Option<u32> {
        let value = match std::env::var("ENGINE_GPU_PROFILE_EVERY") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => return None,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("ENGINE_GPU_PROFILE_EVERY is not valid Unicode")
            }
        };
        let interval = value.parse::<u32>().unwrap_or_else(|error| {
            panic!("ENGINE_GPU_PROFILE_EVERY must be a positive u32: {error}")
        });
        if interval == 0 {
            panic!("ENGINE_GPU_PROFILE_EVERY must be greater than zero");
        }
        Some(interval)
    }

    pub fn new(
        device: &wgpu::Device,
        timestamp_period_ns: f32,
        every_frames: u32,
        inside_passes: bool,
    ) -> Self {
        let scene = inside_passes.then(|| SceneQueries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("scene-gpu-profile"),
                ty: wgpu::QueryType::Timestamp,
                count: SCENE_COUNT,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene-gpu-profile-resolve"),
                size: SCENE_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scene-gpu-profile-read"),
                size: SCENE_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            sums_ms: [0.0; 10],
        });
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame-gpu-profile"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-gpu-profile-resolve"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-gpu-profile-read"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            read_buffer,
            timestamp_period_ns,
            every_frames,
            sums_ms: [0.0; 9],
            frames: 0,
            scene,
        }
    }

    pub fn timestamp(&self, encoder: &mut wgpu::CommandEncoder, mark: Mark) {
        encoder.write_timestamp(&self.query_set, mark as u32);
    }

    /// Stamp inside the main scene pass (no-op without in-pass support).
    pub fn scene_mark(&self, pass: &mut wgpu::RenderPass<'_>, mark: SceneMark) {
        if let Some(scene) = &self.scene {
            pass.write_timestamp(&scene.set, mark as u32);
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.read_buffer, 0, QUERY_BYTES);
        if let Some(scene) = &self.scene {
            encoder.resolve_query_set(&scene.set, 0..SCENE_COUNT, &scene.resolve, 0);
            encoder.copy_buffer_to_buffer(&scene.resolve, 0, &scene.read, 0, SCENE_BYTES);
        }
    }

    /// Read this frame's timestamps and print the running average every
    /// `every_frames` frames. Waits for the GPU.
    pub fn read_and_report(&mut self, device: &wgpu::Device) {
        let slice = self.read_buffer.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).expect("GPU profile map receiver exists");
        });
        device
            .poll(wgpu::PollType::Wait)
            .expect("GPU profile device poll failed");
        receive
            .recv()
            .expect("GPU profile map callback disappeared")
            .expect("GPU profile buffer map failed");

        let mapped = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&mapped);
        if timestamps.len() != QUERY_COUNT as usize {
            panic!(
                "GPU profile returned {} timestamps, expected {QUERY_COUNT}",
                timestamps.len()
            );
        }
        for (phase, sum) in self.sums_ms.iter_mut().enumerate() {
            // Some drivers report equal or (rarely) reordered stamps for empty
            // phases; count those as zero instead of failing the frame.
            let ticks = timestamps[phase + 1].saturating_sub(timestamps[phase]);
            *sum += ticks as f64 * f64::from(self.timestamp_period_ns) / 1_000_000.0;
        }
        drop(mapped);
        self.read_buffer.unmap();
        if let Some(scene) = self.scene.as_mut() {
            let stamps = read_stamps(device, &scene.read, SCENE_COUNT);
            for (part, sum) in scene.sums_ms.iter_mut().enumerate() {
                let ticks = stamps[part + 1].saturating_sub(stamps[part]);
                *sum += ticks as f64 * f64::from(self.timestamp_period_ns) / 1_000_000.0;
            }
        }

        self.frames += 1;
        if self.frames >= self.every_frames {
            let n = f64::from(self.frames);
            let mut line = format!("gpu_profile frames={}", self.frames);
            let mut total = 0.0;
            for (name, sum) in PHASES.iter().zip(self.sums_ms) {
                line.push_str(&format!(" {name}={:.3}ms", sum / n));
                total += sum / n;
            }
            line.push_str(&format!(" total={total:.3}ms"));
            eprintln!("{line}");
            if let Some(scene) = self.scene.as_mut() {
                let mut line = String::from("gpu_profile scene:");
                for (name, sum) in SCENE_PARTS.iter().zip(scene.sums_ms) {
                    line.push_str(&format!(" {name}={:.3}ms", sum / n));
                }
                eprintln!("{line}");
                scene.sums_ms = [0.0; 10];
            }
            self.sums_ms = [0.0; 9];
            self.frames = 0;
        }
    }
}

fn read_stamps(device: &wgpu::Device, buffer: &wgpu::Buffer, count: u32) -> Vec<u64> {
    let slice = buffer.slice(..);
    let (send, receive) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        send.send(result).expect("GPU profile map receiver exists");
    });
    device
        .poll(wgpu::PollType::Wait)
        .expect("GPU profile device poll failed");
    receive
        .recv()
        .expect("GPU profile map callback disappeared")
        .expect("GPU profile buffer map failed");
    let stamps: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    buffer.unmap();
    assert_eq!(stamps.len(), count as usize, "GPU profile stamp count");
    stamps
}
