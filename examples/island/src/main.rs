//! Island: the engine's parts in one scene. A wooded island in an open sea
//! whose swell runs up the beaches, a character walking the shore and one
//! waiting at the landing, a physical sky, and ground you can dig into
//! (the sea floods what you dig).
//!
//! Right mouse + move to look, WASD to move, Shift faster. G switches between
//! flying (Space/E up, Ctrl/Q down) and a body that walks, swims and dives.
//! Left click digs under the cursor, F builds rock, 1/2/3 sets a calm,
//! moderate or rough sea.
//!
//! `ENGINE_ISLAND_VIEW=landing|hill|sea|above` picks a starting view,
//! `ENGINE_SKY_HOUR=17.5` the time of day.
use engine::advanced::Volume;
use engine::prelude::*;
use engine::water::{Sea, Swell, Swimmer, WaterLook};
use glam::{Vec2, Vec3};

/// Simulated sea: 256 m square, the island in the middle.
const SIZE: f32 = 256.0;
const CENTER: Vec2 = Vec2::new(128.0, 128.0);
const SEA_LEVEL: f32 = 0.0;
/// Sand reaches up to here.
const BEACH_TOP: f32 = 2.2;
/// Angle (around the island centre) of the landing where the camera starts.
const LANDING: f32 = 2.4;
const CHARACTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model_viewer/assets/character.glb"
);

/// The island's shape, shared by the terrain, the scatter and the walkers.
struct Island {
    noise: Noise,
}

impl Island {
    /// Distance from the centre to the shoreline in direction `angle`.
    fn shore_radius(&self, angle: f32) -> f32 {
        let (s, c) = angle.sin_cos();
        78.0 + self.noise.fbm2(c * 1.3 + 5.0, s * 1.3 + 2.0, 3, 2.0, 0.5) * 22.0
    }

    /// Metres inland from the shore (negative out at sea).
    fn inland(&self, x: f32, z: f32) -> f32 {
        let d = Vec2::new(x, z) - CENTER;
        self.shore_radius(d.y.atan2(d.x)) - d.length()
    }

    fn ground(&self, x: f32, z: f32) -> f32 {
        let s = self.inland(x, z);
        let mut h = if s < 0.0 {
            // The sea floor shelves away to 14 m.
            (s * 0.08).max(-14.0)
        } else {
            // A gentle beach, then rolling hills rising to a rounded top.
            let rise = smoothstep(14.0, 80.0, s);
            let rolling = self.noise.fbm2(x * 0.025, z * 0.025, 3, 2.0, 0.5) * 0.5 + 0.5;
            s.min(14.0) * 0.13 + rise * (10.0 + rolling * 14.0)
        };
        h += self.noise.fbm2(x * 0.08, z * 0.08, 2, 2.0, 0.5) * 0.3;
        // A rocky bluff on the east side, dropping straight into the sea.
        let east = ((x - CENTER.x - 40.0) / 25.0).clamp(0.0, 1.0)
            * (1.0 - ((z - CENTER.y - 10.0).abs() / 30.0).min(1.0));
        if east > 0.0 && s > -12.0 {
            let top = 11.0 + self.noise.fbm2(x * 0.1, z * 0.1, 3, 2.0, 0.5) * 3.0;
            h = h.max(east * top * (s + 12.0).min(6.0) / 6.0 - 3.0 * (1.0 - east));
        }
        h
    }

    fn density(&self, p: Vec3) -> f32 {
        (self.ground(p.x, p.z) - p.y).clamp(-6.0, 6.0)
    }

    fn slope(&self, x: f32, z: f32) -> f32 {
        (self.ground(x + 1.0, z) - self.ground(x - 1.0, z)).abs()
            + (self.ground(x, z + 1.0) - self.ground(x, z - 1.0)).abs()
    }

    /// A point on a ring `inland` metres from the shore.
    fn ring_point(&self, angle: f32, inland: f32) -> Vec3 {
        let r = self.shore_radius(angle) - inland;
        let xz = CENTER + Vec2::new(angle.cos(), angle.sin()) * r;
        Vec3::new(xz.x, self.ground(xz.x, xz.y), xz.y)
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sea(kind: &str) -> Sea {
    let base = Sea::calm(SEA_LEVEL);
    let swell = |height, period, direction_deg| Swell {
        height,
        period,
        direction_deg,
    };
    match kind {
        "calm" => base.with_swell(swell(0.3, 7.0, 80.0)),
        "rough" => base
            .with_swell(swell(1.6, 11.0, 80.0))
            .with_swell(swell(0.6, 6.5, 120.0)),
        _ => base
            .with_swell(swell(0.8, 9.0, 80.0))
            .with_swell(swell(0.3, 6.0, 120.0)),
    }
}

/// Trees, shrubs and rocks. Palms along the beach, a mixed wood inland.
fn populate(world: &mut World, island: &Island) -> EngineResult<()> {
    let ground = |x: f32, z: f32| island.ground(x, z) - 0.15;
    let landing = island.ring_point(LANDING, 6.0);
    let open = |x: f32, z: f32| Vec2::new(x - landing.x, z - landing.z).length() > 14.0;

    let wood = Scatter::new(21)
        .disc(CENTER.x, CENTER.y, 110.0)
        .spacing(7.0)
        .density(0.95)
        .patch_size(40.0)
        .scale_range(0.75, 1.1)
        .on_ground(ground)
        .only_where(|x, z| island.inland(x, z) > 20.0 && island.slope(x, z) < 1.8 && open(x, z))
        .places()?;
    let kinds = [
        (TreeSpecies::Oak, 1),
        (TreeSpecies::Pine, 2),
        (TreeSpecies::Linden, 3),
        (TreeSpecies::Cypress, 4),
        (TreeSpecies::Pine, 5),
        (TreeSpecies::Chestnut, 6),
    ];
    let palms = Scatter::new(22)
        .disc(CENTER.x, CENTER.y, 110.0)
        .spacing(9.0)
        .density(0.45)
        .scale_range(0.8, 1.15)
        .on_ground(ground)
        .only_where(|x, z| {
            let s = island.inland(x, z);
            (6.0..20.0).contains(&s) && island.slope(x, z) < 1.2 && open(x, z)
        })
        .places()?;
    let mut all: Vec<(TreeSpecies, u32)> = kinds.to_vec();
    all.extend([(TreeSpecies::Palm, 7), (TreeSpecies::Palm, 8)]);
    let models = build_all(&all, |&(species, variant)| tree_model(species, variant))?;
    for (tree, places) in models[..kinds.len()]
        .iter()
        .zip(Scatter::split(&wood, kinds.len(), 1))
    {
        world.spawn_lod_at(tree.lod_levels(), &places)?;
    }
    for (tree, places) in models[kinds.len()..]
        .iter()
        .zip(Scatter::split(&palms, 2, 2))
    {
        world.spawn_lod_at(tree.lod_levels(), &places)?;
    }

    let shrubs = Scatter::new(23)
        .disc(CENTER.x, CENTER.y, 110.0)
        .spacing(5.0)
        .density(0.4)
        .scale_range(0.7, 1.3)
        .on_ground(ground)
        .only_where(|x, z| island.inland(x, z) > 19.0 && island.slope(x, z) < 1.8 && open(x, z))
        .places()?;
    let models = build_all(&[0, 1], |&v| shrub_model(v))?;
    for (shrub, places) in models.iter().zip(Scatter::split(&shrubs, 2, 3)) {
        world.spawn_lod_at(shrub.lod_levels(), &places)?;
    }

    let rocks = Scatter::new(24)
        .disc(CENTER.x, CENTER.y, 110.0)
        .spacing(13.0)
        .density(0.4)
        .scale_range(0.4, 1.8)
        .on_ground(|x, z| island.ground(x, z) - 0.3)
        .only_where(|x, z| island.inland(x, z) > 2.0 && open(x, z))
        .places()?;
    let models = build_all(&[0, 1, 2, 3], |&k: &usize| {
        rock_model(RockShape::ALL[k], k as u32)
    })?;
    for (rock, places) in models.iter().zip(Scatter::split(&rocks, 4, 4)) {
        world.spawn_lod_at(rock.lod_levels_scaled(1.4), &places)?;
    }
    eprintln!(
        "island: {} trees, {} palms, {} shrubs, {} rocks",
        wood.len(),
        palms.len(),
        shrubs.len(),
        rocks.len()
    );
    Ok(())
}

/// Starting views: eye, yaw, pitch.
fn view(island: &Island, name: &str) -> (Vec3, f32, f32) {
    match name {
        "hill" => {
            // Above the wood on the hill, looking down to the landing.
            let eye = Vec3::new(CENTER.x, island.ground(CENTER.x, CENTER.y) + 28.0, CENTER.y);
            let to = island.ring_point(LANDING, 0.0) - eye;
            (eye, to.x.atan2(to.z).to_degrees(), -14.0)
        }
        "sea" => (Vec3::new(-40.0, 30.0, 20.0), 50.0, -12.0),
        // High above the sea, where the simulated square meets the open sea.
        "above" => (Vec3::new(CENTER.x, 220.0, -150.0), 0.0, -45.0),
        _ => {
            // At the landing, looking along the beach and out to sea.
            let eye = island.ring_point(LANDING, 5.0) + Vec3::Y * 1.8;
            let to = island.ring_point(LANDING - 0.45, -14.0) - eye;
            (eye, to.x.atan2(to.z).to_degrees(), -7.0)
        }
    }
}

/// The walker's path: up and down the beach past the landing, turning at
/// either end.
struct Walk {
    angle: f32,
    dir: f32,
}

impl Walk {
    const SPEED: f32 = 1.35;
    const INLAND: f32 = 9.0;
    const REACH: f32 = 0.8;

    fn place(&mut self, island: &Island, dt: f32) -> EngineResult<Place> {
        let radius = island.shore_radius(self.angle) - Self::INLAND;
        self.angle += self.dir * Self::SPEED * dt / radius.max(1.0);
        if (self.angle - LANDING).abs() > Self::REACH {
            self.dir = -self.dir;
            self.angle = LANDING + (self.angle - LANDING).clamp(-Self::REACH, Self::REACH);
        }
        let here = island.ring_point(self.angle, Self::INLAND);
        let ahead = island.ring_point(self.angle + self.dir * 0.02, Self::INLAND);
        let dir = ahead - here;
        Place::at(here.x, here.y, here.z)?.yaw_deg(dir.x.atan2(dir.z).to_degrees())
    }
}

fn main() -> EngineResult<()> {
    let island = Island {
        noise: Noise::new(7),
    };
    let view_name = std::env::var("ENGINE_ISLAND_VIEW").unwrap_or_default();
    let hour = std::env::var("ENGINE_SKY_HOUR")
        .ok()
        .and_then(|h| h.parse::<f32>().ok())
        .unwrap_or(17.0);
    let (eye, yaw, pitch) = view(&island, &view_name);
    let mut camera = FlyCamera::new(eye, yaw, pitch).with_speed(10.0);
    let mut body: Option<Swimmer> = None;
    let character = AnimatedModel::load(CHARACTER);
    if let Err(e) = &character {
        eprintln!("island: no character ({e}), the beach stays empty");
    }
    let mut character = character.ok();
    let mut walk = Walk {
        angle: LANDING - 0.4,
        dir: 1.0,
    };
    let mut walker = None;

    Engine::run("island", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            let started = std::time::Instant::now();
            let mut terrain = VoxelTerrain::new(Volume::new(1.0))
                .with_beach(BEACH_TOP)
                .with_color(rgb(128, 120, 110));
            terrain.generate(
                // The sea floor goes on past the simulated area.
                Vec3::new(-128.0, -16.0, -128.0),
                Vec3::new(SIZE + 128.0, 48.0, SIZE + 128.0),
                |p| island.density(p),
            );
            let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(SIZE), 1.0, -16.0, 48.0);
            let mut water = Water::new(region, &terrain).with_look(WaterLook::ocean());
            water.flood_sea(SEA_LEVEL);
            water.set_sea(Some(sea("moderate")));
            // Let the swell reach the beaches before the first frame.
            for _ in 0..60 * 20 {
                water.update(1.0 / 60.0);
            }
            world.set_voxel_terrain(Some(terrain));
            world.set_water(Some(water));
            world.set_physical_sky(Some(PhysicalSky::at_hour(hour)));
            world.set_view_distance(4000.0)?;
            populate(world, &island)?;
            if let Some(model) = character.take() {
                // One waits at the landing, facing the camera.
                let at = island.ring_point(LANDING - 0.1, 6.0);
                let eye = island.ring_point(LANDING, 5.0);
                let face = (eye - at).x.atan2((eye - at).z).to_degrees();
                let waiting = Place::at(at.x, at.y, at.z)?.yaw_deg(face)?;
                let id = world.spawn_animated(model.clone(), waiting)?;
                world.play_animation(id, "talk")?;
                let id = world.spawn_animated(model, walk.place(&island, 0.0)?)?;
                world.play_animation(id, "walk")?;
                walker = Some(id);
            }
            eprintln!("island: ready in {:.1} s", started.elapsed().as_secs_f32());
        }
        if let Some(id) = walker {
            world.set_place(id, walk.place(&island, frame.dt)?)?;
        }

        for (key, kind) in [
            (Key::Digit1, "calm"),
            (Key::Digit2, "moderate"),
            (Key::Digit3, "rough"),
        ] {
            if frame.input.pressed(key) {
                if let Some(water) = world.water_mut() {
                    water.set_sea(Some(sea(kind)));
                }
            }
        }
        if frame.input.pressed(Key::G) {
            body = match body.take() {
                Some(b) => {
                    camera.position = b.eye();
                    camera.yaw_deg = b.yaw_deg;
                    camera.pitch_deg = b.pitch_deg;
                    None
                }
                None => {
                    let mut b = Swimmer::new(camera.position - Vec3::Y * 1.65, camera.yaw_deg);
                    b.pitch_deg = camera.pitch_deg;
                    Some(b)
                }
            };
        }
        match body.as_mut() {
            Some(b) => {
                b.update(world, frame);
            }
            None => camera.update(world, frame),
        }
        // Dig and build under the mouse cursor; the sea flows into the holes.
        let (eye, aim) = world.cursor_ray(frame);
        if let Some(hit) = world
            .voxel_terrain()
            .and_then(|t| t.raycast(eye, aim, 300.0))
        {
            let terrain = world.voxel_terrain_mut().expect("terrain");
            if frame.input.mouse_clicked(MouseButton::Left) {
                terrain.dig_sphere(hit, 3.0);
            }
            if frame.input.pressed(Key::F) {
                terrain.add_sphere(hit, 2.5);
            }
        }
        Ok(())
    })
}
