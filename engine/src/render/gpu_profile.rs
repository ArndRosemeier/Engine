const QUERY_COUNT: u32 = 4;
const QUERY_BYTES: u64 = QUERY_COUNT as u64 * wgpu::QUERY_SIZE as u64;

pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    read_buffer: wgpu::Buffer,
    timestamp_period_ns: f32,
    every_frames: u32,
    frame: u64,
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

    pub fn new(device: &wgpu::Device, timestamp_period_ns: f32, every_frames: u32) -> Self {
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
            frame: 0,
        }
    }

    pub fn begin_frame(&mut self) -> bool {
        self.frame = self
            .frame
            .checked_add(1)
            .expect("GPU profile frame overflow");
        self.frame.is_multiple_of(u64::from(self.every_frames))
    }

    pub fn timestamp(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        if index >= QUERY_COUNT {
            panic!("GPU profile query index {index} exceeds {QUERY_COUNT}");
        }
        encoder.write_timestamp(&self.query_set, index);
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.read_buffer, 0, QUERY_BYTES);
    }

    pub fn read_and_report(&self, device: &wgpu::Device) {
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
        let milliseconds = |start: usize, end: usize| -> f64 {
            let ticks = timestamps[end]
                .checked_sub(timestamps[start])
                .expect("GPU profile timestamps are not monotonic");
            ticks as f64 * f64::from(self.timestamp_period_ns) / 1_000_000.0
        };
        eprintln!(
            "gpu_profile frame={} shadow={:.3}ms main={:.3}ms ui={:.3}ms total={:.3}ms",
            self.frame,
            milliseconds(0, 1),
            milliseconds(1, 2),
            milliseconds(2, 3),
            milliseconds(0, 3),
        );
        drop(mapped);
        self.read_buffer.unmap();
    }
}
