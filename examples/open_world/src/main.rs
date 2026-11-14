//! An endless generated world, streamed around the camera.
//!
//! Continents and seas, hills, ridged mountains, beaches, woods, rocks and
//! the odd hut with a camp fire. Nothing is made up front: a `WorldGenerator`
//! (about a hundred lines below) describes the ground and what stands on each
//! tile, and `WorldStream` builds it on worker threads as you fly, from
//! diggable voxels at your feet to height tiles on the horizon.
//!
//! Controls: right mouse + move to look, WASD to fly, Space/E up, Ctrl/Q
//! down, Shift fast, F very fast (supersonic, to watch the streaming keep
//! up), left click digs, B builds, Esc quits.
//!
//! `ENGINE_AUTOFLY=1` flies on its own; `ENGINE_OPEN_WORLD_SPEED=300` sets
//! the speed (m/s) for that; `ENGINE_SKY_HOUR=19` sets the time of day;
//! `ENGINE_OPEN_WORLD_SEED=7` another world; `ENGINE_OPEN_WORLD_COLOUR=1`
//! grows wildflowers and paints the rock. `ENGINE_OPEN_WORLD_BENCH=30`
//! records frame times for 30 seconds (best with `ENGINE_AUTOFLY=1`), prints
//! them and quits.
use engine::prelude::*;
use engine::stream::{StreamSettings, Tile, WorldGenerator, WorldStream};

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The models tiles place: templates spawned once with no places.
#[derive(Clone)]
struct Models {
    leafy: Vec<LodId>,
    pines: Vec<LodId>,
    palms: Vec<LodId>,
    shrubs: Vec<LodId>,
    rocks: Vec<LodId>,
}

struct Continents {
    shape: Noise,
    hills: Noise,
    peaks: Noise,
    models: Models,
}

impl Continents {
    /// Land (above 0) or sea (below), smoothly.
    fn continent(&self, x: f32, z: f32) -> f32 {
        self.shape.fbm2(x * 0.00022, z * 0.00022, 4, 2.0, 0.5) + 0.12
    }

    fn slope(&self, x: f32, z: f32) -> f32 {
        let dx = self.height(x + 1.0, z) - self.height(x - 1.0, z);
        let dz = self.height(x, z + 1.0) - self.height(x, z - 1.0);
        (dx * dx + dz * dz).sqrt() * 0.5
    }
}

impl WorldGenerator for Continents {
    fn height(&self, x: f32, z: f32) -> f32 {
        let c = self.continent(x, z);
        let land = smoothstep(-0.05, 0.25, c);
        // Sea floor, coastal plains, rolling hills, then mountains inland.
        let base = c * 140.0;
        let hills = self.hills.iq_fbm2(x * 0.0035, z * 0.0035, 6) * 28.0 * land;
        let ridge = self.peaks.ridged2(x * 0.0011, z * 0.0011, 5, 2.0, 0.5);
        let peaks = ridge * ridge * 300.0 * smoothstep(0.18, 0.55, c);
        base + hills + peaks
    }

    /// Dry plains in broad patches, and snow on the mountains.
    fn ground_cover(&self, x: f32, z: f32) -> Option<GroundCover> {
        let rain = self
            .shape
            .fbm2(x * 0.0005 + 40.0, z * 0.0005 - 17.0, 3, 2.0, 0.5);
        Some(GroundCover {
            green: smoothstep(-0.25, 0.05, rain),
            snow_line: 210.0,
        })
    }

    fn populate(&self, tile: &mut Tile) -> EngineResult<()> {
        let (min, max) = (tile.min(), tile.max());
        let seed = tile.seed();
        let ground = |x: f32, z: f32| self.height(x, z);
        // Trees on gentle dry ground; palms by the sea, pines up high.
        let trees = Scatter::new(seed)
            .rect(min.x, min.y, max.x, max.y)
            .spacing(10.0)
            .density(0.55)
            .patch_size(90.0)
            .scale_range(0.8, 1.2)
            .on_ground(|x, z| ground(x, z) - 0.2)
            .only_where(|x, z| {
                let h = ground(x, z);
                h > 2.5 && h < 190.0 && self.slope(x, z) < 0.55
            })
            .places()?;
        for (i, p) in trees.into_iter().enumerate() {
            let h = p.position.y;
            let set = if h < 7.0 {
                &self.models.palms
            } else if h > 95.0 {
                &self.models.pines
            } else {
                &self.models.leafy
            };
            tile.place(set[i % set.len()], p);
        }
        let shrubs = Scatter::new(seed ^ 0x51)
            .rect(min.x, min.y, max.x, max.y)
            .spacing(7.0)
            .density(0.4)
            .scale_range(0.7, 1.3)
            .on_ground(|x, z| ground(x, z) - 0.1)
            .only_where(|x, z| ground(x, z) > 2.0 && self.slope(x, z) < 0.7)
            .places()?;
        for (i, p) in shrubs.into_iter().enumerate() {
            tile.place(self.models.shrubs[i % self.models.shrubs.len()], p);
        }
        // Rocks everywhere on land, more on steep ground.
        let rocks = Scatter::new(seed ^ 0xA7)
            .rect(min.x, min.y, max.x, max.y)
            .spacing(18.0)
            .density(0.35)
            .scale_range(0.6, 2.4)
            .on_ground(|x, z| ground(x, z) - 0.4)
            .only_where(|x, z| ground(x, z) > 1.0)
            .places()?;
        for (i, p) in rocks.into_iter().enumerate() {
            tile.place(self.models.rocks[i % self.models.rocks.len()], p);
        }
        // Now and then a hut with a camp fire on flat ground.
        if seed.is_multiple_of(7) {
            let c = tile.centre();
            let h = ground(c.x, c.y);
            if h > 3.0 && h < 120.0 && self.slope(c.x, c.y) < 0.15 {
                tile.spawn_solid(
                    hut()?,
                    Place::new(c.x, h - 0.3, c.y).with_yaw_deg((seed % 360) as f32),
                );
                let fire_at = Vec3::new(c.x + 4.0, 0.0, c.y + 3.0);
                tile.add_fire(fire_at.with_y(ground(fire_at.x, fire_at.z)), 0.8);
            }
        }
        Ok(())
    }

    fn sea(&self) -> Option<engine::water::Sea> {
        Some(
            engine::water::Sea::calm(0.0).with_swell(engine::water::Swell {
                height: 0.6,
                period: 7.0,
                direction_deg: 250.0,
            }),
        )
    }
}

/// A small timber hut.
fn hut() -> EngineResult<Mesh> {
    let wall = rgb(120, 86, 58);
    let roof = rgb(92, 60, 44);
    let mut m = Mesh::new();
    m.add_box((0.0, 1.4, 0.0), (4.0, 2.8, 3.2), wall)?;
    m.add_box((0.0, 3.0, 0.0), (4.6, 0.5, 3.8), roof)?;
    m.add_box((0.0, 3.5, 0.0), (3.4, 0.5, 2.6), roof)?;
    m.add_box((0.0, 1.0, 1.62), (0.9, 2.0, 0.05), rgb(60, 42, 30))?;
    Ok(m)
}

fn models(world: &mut World) -> EngineResult<Models> {
    let mut template = |levels: Vec<LodLevel>| world.spawn_lod(levels);
    let leafy_kinds = [
        TreeSpecies::Oak,
        TreeSpecies::Birch,
        TreeSpecies::Linden,
        TreeSpecies::Maple,
    ];
    let leafy = build_all(&[0usize, 1, 2, 3, 4, 5], |&k| {
        tree_model(leafy_kinds[k % 4], k as u32)
    })?;
    let pines = build_all(&[0u32, 1, 2], |&k| tree_model(TreeSpecies::Pine, k))?;
    let palms = build_all(&[0u32, 1], |&k| tree_model(TreeSpecies::Palm, k))?;
    let shrubs = build_all(&[0u32, 1], |&k| shrub_model(k))?;
    let rocks = build_all(&[0usize, 1, 2, 3], |&k| {
        rock_model(RockShape::ALL[k], k as u32)
    })?;
    Ok(Models {
        leafy: leafy
            .iter()
            .map(|t| template(t.lod_levels()))
            .collect::<EngineResult<_>>()?,
        pines: pines
            .iter()
            .map(|t| template(t.lod_levels()))
            .collect::<EngineResult<_>>()?,
        palms: palms
            .iter()
            .map(|t| template(t.lod_levels()))
            .collect::<EngineResult<_>>()?,
        shrubs: shrubs
            .iter()
            .map(|t| template(t.lod_levels()))
            .collect::<EngineResult<_>>()?,
        rocks: rocks
            .iter()
            .map(|r| template(r.lod_levels_scaled(1.8)))
            .collect::<EngineResult<_>>()?,
    })
}

/// Frame times and streaming cost while flying (`ENGINE_OPEN_WORLD_BENCH`).
#[derive(Default)]
struct Bench {
    time: f32,
    frames: Vec<f32>,
    stream_ms: Vec<f32>,
    /// Path length flown (the auto flight circles back to its start).
    path: f32,
    prev: Option<Vec3>,
    /// Wall clock between frames: `frame.dt` is smoothed and capped.
    clock: Option<std::time::Instant>,
    last: engine::stream::StreamStats,
}

impl Bench {
    fn add(&mut self, stats: engine::stream::StreamStats, at: Vec3) {
        let now = std::time::Instant::now();
        if let Some(before) = self.clock.replace(now) {
            let dt = (now - before).as_secs_f32();
            self.time += dt;
            self.frames.push(dt * 1000.0);
            self.stream_ms.push(stats.install_ms);
        }
        if let Some(p) = self.prev.replace(at) {
            self.path += (at - p).length();
        }
        self.last = stats;
    }

    fn report(&mut self) {
        if self.frames.is_empty() {
            println!("open_world bench: no frames recorded");
            return;
        }
        let pct = |v: &mut Vec<f32>, p: f32| {
            v.sort_by(f32::total_cmp);
            v[((v.len() - 1) as f32 * p).round() as usize]
        };
        let n = self.frames.len().max(1);
        let avg = self.frames.iter().sum::<f32>() / n as f32;
        let over = |ms: f32| self.frames.iter().filter(|&&f| f > ms).count();
        let (h25, h50) = (over(25.0), over(50.0));
        let mut f = self.frames.clone();
        let mut s = self.stream_ms.clone();
        let dist = self.path;
        println!(
            "open_world bench: {n} frames in {:.1} s, flew {dist:.0} m\n  \
             frame ms: avg {avg:.2} ({:.0} FPS)  p50 {:.2}  p95 {:.2}  p99 {:.2}  max {:.2}\n  \
             frames over 25 ms: {h25}, over 50 ms: {h50}\n  \
             stream main-thread ms: p50 {:.3}  p99 {:.3}  max {:.3}\n  \
             at the end: {:?}",
            self.time,
            1000.0 / avg,
            pct(&mut f, 0.5),
            pct(&mut f, 0.95),
            pct(&mut f, 0.99),
            pct(&mut f, 1.0),
            pct(&mut s, 0.5),
            pct(&mut s, 0.99),
            pct(&mut s, 1.0),
            self.last,
        );
    }
}

fn main() -> EngineResult<()> {
    let seed = setting("ENGINE_OPEN_WORLD_SEED")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(3);
    // Wildflower fields on the grass and rock banded red, ochre and cream.
    let colourful = setting("ENGINE_OPEN_WORLD_COLOUR").is_some();
    let hour = setting("ENGINE_SKY_HOUR")
        .and_then(|h| h.parse::<f32>().ok())
        .unwrap_or(15.5);
    let auto_speed = setting("ENGINE_OPEN_WORLD_SPEED").and_then(|s| s.parse::<f32>().ok());
    let bench_seconds = setting("ENGINE_OPEN_WORLD_BENCH").and_then(|s| s.parse::<f32>().ok());
    let mut bench = Bench::default();
    // Height function for the camera before the stream exists.
    let probe = Continents {
        shape: Noise::new(seed),
        hills: Noise::new(seed + 1),
        peaks: Noise::new(seed + 2),
        models: Models {
            leafy: vec![],
            pines: vec![],
            palms: vec![],
            shrubs: vec![],
            rocks: vec![],
        },
    };
    // Start on a coast: walk along +X until the ground rises out of the sea.
    let mut start = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..4000 {
        let x = i as f32 * 25.0;
        let h = probe.height(x, 0.0);
        if (4.0..30.0).contains(&h) {
            start = Vec3::new(x, h, 0.0);
            break;
        }
    }
    let mut camera =
        FlyCamera::new(start + Vec3::new(0.0, 25.0, 0.0), 250.0, -12.0).with_speed(15.0);
    if let Some(speed) = auto_speed {
        camera = camera.with_speed(speed);
    }

    Engine::run("open_world", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            world.set_view_distance(6000.0)?;
            let generator = Continents {
                shape: Noise::new(seed),
                hills: Noise::new(seed + 1),
                peaks: Noise::new(seed + 2),
                models: models(world)?,
            };
            let stream =
                WorldStream::new(generator, StreamSettings::default()).with_terrain(move |t| {
                    let t = t.with_beach(2.5);
                    if colourful {
                        t.with_flowers(1.0).with_painted_rock(1.0)
                    } else {
                        t
                    }
                });
            world.set_stream(Some(stream))?;
            world.look_first_person(camera.position, camera.yaw_deg, camera.pitch_deg);
            let t = std::time::Instant::now();
            world.prepare_stream(20.0);
            eprintln!(
                "open_world: ground ready in {:.0} ms",
                t.elapsed().as_secs_f32() * 1000.0
            );
        }

        // F: very fast, to see the streaming keep up.
        let boost = if frame.input.down(Key::F) { 20.0 } else { 1.0 };
        let base = camera.speed;
        camera.speed = base * boost;
        let ground = |x: f32, z: f32| world.stream().map_or(0.0, |s| s.height_at(x, z)).max(0.0);
        let floor = (ground)(camera.position.x, camera.position.z);
        camera.update_above(world, frame, |_, _| floor);
        camera.speed = base;

        // Dig where the mouse points; B builds.
        let dig = frame.input.mouse_clicked(MouseButton::Left);
        let build = frame.input.pressed(Key::B);
        if dig || build {
            let (origin, dir) = world.cursor_ray(frame);
            let hit = world
                .voxel_terrain()
                .and_then(|t| t.raycast(origin, dir, 400.0));
            if let Some(p) = hit {
                // Both throw dust (and digging clods of earth) by themselves.
                if dig {
                    world.dig(p, 4.0);
                } else {
                    world.add_ground(p + dir * -1.0, 3.0);
                }
            }
        }

        if let (Some(seconds), Some(stream)) = (bench_seconds, world.stream()) {
            // Skip the first second (shader warm-up), then record.
            if frame.time > 1.0 {
                bench.add(stream.stats(), camera.position);
            }
            if bench.time >= seconds {
                bench.report();
                world.request_exit();
            }
        }

        // The generator's ground cover is what the terrain draws with.
        let cover = world
            .voxel_terrain()
            .and_then(|t| t.ground_cover(camera.position.x, camera.position.z));
        world.check(
            "the ground has the generator's cover",
            frame.first || cover.is_some_and(|c| (0.0..=1.0).contains(&c.green)),
        );

        if let Some(stream) = world.stream() {
            let s = stream.stats();
            engine::egui::Window::new("open world")
                .anchor(engine::egui::Align2::LEFT_TOP, [12.0, 12.0])
                .resizable(false)
                .show(frame.ui.ctx(), |ui| {
                    ui.label(format!(
                        "{:.0} FPS   speed {:.0} m/s",
                        frame.fps,
                        camera.speed
                            * boost
                            * if frame.input.down(Key::Shift) {
                                camera.fast
                            } else {
                                1.0
                            }
                    ));
                    ui.label(format!(
                        "position {:.0}, {:.0}, {:.0}",
                        camera.position.x, camera.position.y, camera.position.z
                    ));
                    ui.label(format!(
                        "voxels: {} blocks, {} chunks ({} edits kept)",
                        s.voxel_blocks, s.voxel_chunks, s.stored_edits
                    ));
                    ui.label(format!("far ground: {} tiles", s.far_shown));
                    ui.label(format!("tiles: {} with {} things", s.tiles, s.tile_things));
                    ui.label(format!(
                        "jobs {}  waiting {}  stream {:.2} ms",
                        s.jobs, s.waiting, s.install_ms
                    ));
                    ui.label("left click digs, B builds, F very fast");
                });
        }
        Ok(())
    })
}
