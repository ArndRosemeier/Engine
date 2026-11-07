use super::swimmer::*;
use super::*;
use crate::volume::Volume;
use glam::{Vec2, Vec3};

/// Terrain from a density closure (> 0 solid).
struct FnTerrain<F: Fn(Vec3) -> f32 + Sync>(F);

impl<F: Fn(Vec3) -> f32 + Sync> WaterTerrain for FnTerrain<F> {
    fn density(&self, p: Vec3) -> f32 {
        (self.0)(p)
    }
    fn scan_step(&self) -> f32 {
        0.25
    }
}

fn run(water: &mut Water, seconds: f32) {
    let steps = (seconds * 60.0).round() as usize;
    for _ in 0..steps {
        water.update(1.0 / 60.0);
    }
}

fn assert_sane(water: &Water) {
    for (i, &d) in water.depth.iter().enumerate() {
        assert!(d.is_finite() && d >= 0.0, "cell {i} depth {d}");
    }
    for &q in &water.q {
        assert!(q.is_finite(), "discharge {q}");
    }
}

fn rel_change(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(1e-9)
}

/// Heads of all wet cells whose floor lies below `y`.
fn wet_heads(water: &Water) -> Vec<f32> {
    (0..water.cell_count())
        .filter(|&i| water.depth[i] > 1e-3)
        .map(|i| water.cell_head(i))
        .collect()
}

#[test]
fn a_still_lake_stays_still() {
    let bowl = HeightMapTerrain(|x: f32, z: f32| ((x - 32.0).powi(2) + (z - 32.0).powi(2)) * 0.01);
    let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(64.0), 1.0, -5.0, 60.0);
    let mut water = Water::new(region, &bowl);
    let added = water.flood(Vec3::new(32.0, 0.0, 32.0), 3.0);
    assert!(added > 100.0, "flood should add water, added {added}");
    let v0 = water.total_volume();
    run(&mut water, 10.0);
    assert_sane(&water);
    for h in wet_heads(&water) {
        assert!((h - 3.0).abs() < 2e-3, "still water moved: head {h}");
    }
    assert!(rel_change(v0, water.total_volume()) < 1e-5);
    assert!(water.is_submerged(Vec3::new(32.0, 2.0, 32.0)));
    assert!(!water.is_submerged(Vec3::new(32.0, 3.5, 32.0)));
    assert!(water.sample(Vec3::new(2.0, 50.0, 2.0)).is_none());
}

#[test]
fn a_dam_break_settles_flat_and_keeps_its_volume() {
    // A flat closed tank 40 x 8 m; the left half starts 2 m deep.
    let flat = HeightMapTerrain(|_, _| 0.0);
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(40.0, 8.0), 1.0, -1.0, 20.0);
    let mut water = Water::new(region, &flat);
    for z in 0..8 {
        for x in 0..20 {
            water.pour(Vec3::new(x as f32 + 0.5, 0.5, z as f32 + 0.5), 2.0);
        }
    }
    let v0 = water.total_volume();
    assert!((v0 - 320.0).abs() < 1e-3);
    let mut early: f32 = 0.0;
    let mut late: f32 = 0.0;
    for second in 0..240 {
        run(&mut water, 1.0);
        assert_sane(&water);
        assert!(
            rel_change(v0, water.total_volume()) < 1e-4,
            "volume drifted at {second}s: {} vs {v0}",
            water.total_volume()
        );
        let heads = wet_heads(&water);
        let rms =
            (heads.iter().map(|h| (h - 1.0).powi(2)).sum::<f32>() / heads.len() as f32).sqrt();
        if second < 20 {
            early = early.max(rms);
        } else if second >= 200 {
            late = late.max(rms);
        }
    }
    // Sloshing (a seiche) dies down but is real physics; the surface must
    // be nearly flat after a few minutes.
    assert!(
        late < 0.03,
        "tank still sloshing: rms {late} m (start {early} m)"
    );
}

#[test]
fn water_never_crosses_a_wall_and_the_region_edge_is_closed() {
    // Two basins split by a wall higher than the water.
    let terrain = HeightMapTerrain(|x: f32, _| if (19.0..21.0).contains(&x) { 5.0 } else { 0.0 });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(40.0, 6.0), 1.0, -1.0, 20.0);
    let mut water = Water::new(region, &terrain);
    water.flood(Vec3::new(5.0, 0.5, 3.0), 3.0);
    let v0 = water.total_volume();
    run(&mut water, 20.0);
    assert_sane(&water);
    assert!(rel_change(v0, water.total_volume()) < 1e-5);
    assert!(
        water.sample(Vec3::new(30.0, 0.5, 3.0)).is_none(),
        "water leaked through the wall"
    );
}

/// Ground at y = 10 with a lake basin, and a sealed cave chamber beside it.
fn lake_and_cave() -> Volume {
    let mut volume = Volume::new(1.0);
    volume.generate_chunks(
        &[
            glam::IVec3::new(0, 0, 0),
            glam::IVec3::new(1, 0, 0),
            glam::IVec3::new(0, -1, 0),
            glam::IVec3::new(1, -1, 0),
        ],
        |p| {
            let ground = 10.0 - p.y;
            // Lake: a box basin 6 m deep at x 4..20.
            let lake = sd_box(p, Vec3::new(12.0, 7.0, 16.0), Vec3::new(8.0, 3.0, 8.0));
            // Cave chamber at x 34..50, y 2..8, sealed by rock.
            let cave = sd_box(p, Vec3::new(42.0, 5.0, 16.0), Vec3::new(8.0, 3.0, 6.0));
            ground.min(lake).min(cave).clamp(-4.0, 4.0)
        },
    );
    volume
}

fn sd_box(p: Vec3, centre: Vec3, half: Vec3) -> f32 {
    let d = (p - centre).abs() - half;
    d.max(Vec3::ZERO).length() + d.max_element().min(0.0)
}

#[test]
fn a_sealed_cave_stays_dry_until_a_tunnel_opens_it() {
    let mut volume = lake_and_cave();
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(60.0, 32.0), 1.0, -8.0, 30.0);
    let mut water = Water::new(region, &volume);
    water.flood(Vec3::new(12.0, 6.0, 16.0), 9.0);
    let cave_point = Vec3::new(42.0, 3.0, 16.0);
    assert!(
        water.cell_at(cave_point).is_some(),
        "the cave should be a cell"
    );
    run(&mut water, 5.0);
    assert!(water.sample(cave_point).is_none(), "sealed cave got water");
    let v0 = water.total_volume();
    let lake_before = water.surface_at(Vec3::new(12.0, 8.0, 16.0)).unwrap();

    // Dig a tunnel from the lake floor to the cave.
    let mut x = 18.0;
    while x <= 36.0 {
        volume.dig_sphere(Vec3::new(x, 4.5, 16.0), 1.6);
        x += 1.0;
    }
    water.resync_area(&volume, Vec2::new(16.0, 13.0), Vec2::new(38.0, 19.0));
    assert!(
        rel_change(v0, water.total_volume()) < 1e-5,
        "resync lost water"
    );
    run(&mut water, 120.0);
    assert_sane(&water);
    assert!(rel_change(v0, water.total_volume()) < 1e-4);
    let cave = water.sample(cave_point).expect("the cave should flood");
    assert!(
        cave.water_depth > 1.0,
        "cave only {} m deep",
        cave.water_depth
    );
    let lake_after = water.surface_at(Vec3::new(12.0, 8.0, 16.0)).unwrap();
    assert!(
        lake_after < lake_before - 0.3,
        "lake should drop as the cave fills: {lake_before} -> {lake_after}"
    );
}

#[test]
fn a_flooded_u_bend_levels_both_shafts() {
    // Shafts at x 2..4 and 16..18, joined by a tunnel y 0..1. All 2 m wide.
    let terrain = FnTerrain(|p: Vec3| {
        let in_z = p.z > 0.0 && p.z < 2.0;
        let shaft = (p.x > 2.0 && p.x < 4.0) || (p.x > 16.0 && p.x < 18.0);
        let tunnel = p.x > 2.0 && p.x < 18.0 && p.y < 1.0;
        if in_z && p.y > 0.0 && (shaft || tunnel) {
            -1.0
        } else {
            1.0
        }
    });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(20.0, 2.0), 0.5, -1.0, 20.0);
    let mut water = Water::new(region, &terrain);
    // Fill the left shaft and the tunnel.
    water.flood(Vec3::new(3.0, 0.5, 1.0), 1.0);
    for z in 0..4 {
        for x in 0..4 {
            water.pour(
                Vec3::new(2.25 + x as f32 * 0.5, 2.0, 0.25 + z as f32 * 0.5),
                0.25 * 6.0,
            );
        }
    }
    let v0 = water.total_volume();
    run(&mut water, 90.0);
    assert_sane(&water);
    let left = water.surface_at(Vec3::new(3.0, 15.0, 1.0)).unwrap();
    let right = water.surface_at(Vec3::new(17.0, 15.0, 1.0)).unwrap();
    assert!(
        (left - right).abs() < 0.15,
        "shafts did not level: {left} vs {right}"
    );
    assert!(right > 2.5, "the right shaft should rise, got {right}");
    assert!(rel_change(v0, water.total_volume()) < 1e-4);
}

#[test]
fn a_waterfall_drains_a_shelf_into_the_basin_below() {
    // A shelf at y 6 for x < 10 dropping to a basin at y 0.
    let terrain = HeightMapTerrain(|x: f32, _| if x < 10.0 { 6.0 } else { 0.0 });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(30.0, 4.0), 1.0, -1.0, 20.0);
    let mut water = Water::new(region, &terrain);
    let id = water.add_source(WaterSource {
        position: Vec3::new(1.5, 7.0, 2.0),
        rate: 2.0,
    });
    run(&mut water, 20.0);
    water.remove_source(id);
    run(&mut water, 30.0);
    assert_sane(&water);
    assert!(
        (water.total_volume() - 40.0).abs() < 0.5,
        "volume {}",
        water.total_volume()
    );
    let shelf: f64 = (0..10)
        .flat_map(|x| (0..4).map(move |z| (x, z)))
        .filter_map(|(x, z)| water.sample(Vec3::new(x as f32 + 0.5, 6.0, z as f32 + 0.5)))
        .map(|s| s.water_depth as f64)
        .sum();
    assert!(
        shelf < 1.0,
        "the shelf should drain, still holds {shelf} m of depth"
    );
    let basin = water.surface_at(Vec3::new(20.0, 3.0, 2.0)).unwrap();
    assert!(
        (basin - 0.5).abs() < 0.05,
        "basin level {basin}, expected 0.5"
    );
}

#[test]
fn a_wave_runs_up_the_beach_and_pulls_back() {
    // Channel 160 m long; flat 4 m deep sea floor, beach rising from x = 100.
    let beach = HeightMapTerrain(|x: f32, _| {
        if x < 100.0 {
            -4.0
        } else {
            -4.0 + (x - 100.0) * 0.1
        }
    });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(160.0, 2.0), 1.0, -6.0, 20.0);
    let mut water = Water::new(region, &beach);
    water.flood_sea(0.0);
    let shoreline = |w: &Water| -> f32 {
        (0..160)
            .rev()
            .find(|&x| {
                w.sample(Vec3::new(
                    x as f32 + 0.5,
                    -4.0 + ((x as f32 + 0.5 - 100.0).max(0.0)) * 0.1 + 0.01,
                    1.0,
                ))
                .is_some_and(|s| s.water_depth > 0.01)
            })
            .map_or(0.0, |x| x as f32 + 0.5)
    };
    let still = shoreline(&water);
    assert!((still - 139.5).abs() < 1.5, "still shoreline at {still}");
    // A 1 m swell hump out at sea.
    for x in 20..40 {
        let t = (x as f32 - 30.0) / 10.0;
        let h = 1.5 * (1.0 - t * t).max(0.0);
        for z in 0..2 {
            water.pour(Vec3::new(x as f32 + 0.5, -3.0, z as f32 + 0.5), h);
        }
    }
    let v0 = water.total_volume();
    let mut furthest = still;
    for _ in 0..60 * 30 {
        water.update(1.0 / 60.0);
        furthest = furthest.max(shoreline(&water));
    }
    assert_sane(&water);
    assert!(
        furthest > still + 4.0,
        "wave should run up the beach: {still} -> {furthest}"
    );
    run(&mut water, 120.0);
    let settled = shoreline(&water);
    assert!(
        settled < furthest - 2.0,
        "water should pull back: {settled} (max {furthest})"
    );
    assert!(rel_change(v0, water.total_volume()) < 1e-4);
}

#[test]
fn rock_added_into_a_lake_raises_it_without_losing_water() {
    let mut volume = lake_and_cave();
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(60.0, 32.0), 1.0, -8.0, 30.0);
    let mut water = Water::new(region, &volume);
    water.flood(Vec3::new(12.0, 6.0, 16.0), 9.0);
    let v0 = water.total_volume();
    volume.add_sphere(Vec3::new(12.0, 5.0, 16.0), 3.0);
    water.resync_area(&volume, Vec2::new(8.0, 12.0), Vec2::new(16.0, 20.0));
    assert!(rel_change(v0, water.total_volume()) < 1e-5);
    run(&mut water, 30.0);
    assert_sane(&water);
    let level = water.surface_at(Vec3::new(6.0, 9.5, 16.0)).unwrap();
    assert!(level > 9.05, "lake should rise, level {level}");
    assert!(rel_change(v0, water.total_volume()) < 1e-4);
}

#[test]
fn abuse_never_produces_nan_or_negative_water() {
    let noise = crate::proc::Noise::new(3);
    let terrain = FnTerrain(move |p: Vec3| {
        let h = noise.fbm2(p.x * 0.05, p.z * 0.05, 4, 2.0, 0.5) * 12.0;
        let cave = noise.sample3(p * 0.08).abs() - 0.15;
        (h - p.y).min(cave * 10.0)
    });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(48.0), 1.0, -16.0, 24.0);
    let mut water = Water::new(region, &terrain);
    let mut v = 0.0;
    for i in 0..40 {
        let p = Vec3::new((i * 7 % 48) as f32 + 0.5, 20.0, (i * 13 % 48) as f32 + 0.5);
        v += water.pour(p, 400.0) as f64;
        for _ in 0..5 {
            water.update(0.1);
        }
        assert_sane(&water);
    }
    assert!(
        rel_change(v, water.total_volume()) < 2e-4,
        "{v} vs {}",
        water.total_volume()
    );
}

#[test]
#[ignore]
fn bench_water_384() {
    let noise = crate::proc::Noise::new(11);
    let terrain =
        HeightMapTerrain(move |x: f32, z: f32| noise.fbm2(x * 0.01, z * 0.01, 5, 2.0, 0.5) * 20.0);
    let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(384.0), 1.0, -40.0, 60.0);
    let started = std::time::Instant::now();
    let mut water = Water::new(region, &terrain);
    eprintln!(
        "build {:.1} ms, {} cells",
        started.elapsed().as_secs_f32() * 1e3,
        water.cell_count()
    );
    water.flood_sea(0.0);
    let started = std::time::Instant::now();
    for _ in 0..120 {
        water.update(1.0 / 60.0);
    }
    let s = water.stats();
    eprintln!(
        "update {:.2} ms/frame, {} substeps, {} wet cells, tiles {:?}",
        started.elapsed().as_secs_f32() * 1e3 / 120.0,
        s.substeps,
        s.wet_cells,
        water.active_tiles()
    );
    // Everything stirred up: the worst case.
    water.rain(0.05);
    let started = std::time::Instant::now();
    for _ in 0..60 {
        water.update(1.0 / 60.0);
    }
    eprintln!(
        "after rain {:.2} ms/frame, tiles {:?}",
        started.elapsed().as_secs_f32() * 1e3 / 60.0,
        water.active_tiles()
    );
}

fn pool() -> (HeightMapTerrain<impl Fn(f32, f32) -> f32 + Sync>, Water) {
    // 40 x 40 m, 6 m deep in the middle, a beach rising toward +x from x = 25.
    let ground = HeightMapTerrain(|x: f32, _z: f32| {
        if x < 25.0 {
            -6.0
        } else {
            -6.0 + (x - 25.0) * 0.6
        }
    });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(40.0), 1.0, -8.0, 20.0);
    let mut water = Water::new(region, &ground);
    water.flood(Vec3::new(10.0, -5.0, 20.0), 0.0);
    (ground, water)
}

fn swim(
    body: &mut Swimmer,
    ground: &dyn WaterTerrain,
    water: &Water,
    input: SwimInput,
    seconds: f32,
) {
    for _ in 0..(seconds * 60.0) as usize {
        body.step(ground, Some(water), input, 1.0 / 60.0);
    }
}

#[test]
fn a_swimmer_treads_water_with_the_eyes_above_the_surface() {
    let (ground, water) = pool();
    let mut body = Swimmer::new((10.0, -1.0, 20.0), 0.0);
    swim(&mut body, &ground, &water, SwimInput::default(), 8.0);
    let eye = body.eye().y;
    assert!(eye > 0.0 && eye < 0.4, "eye at {eye}");
    assert_eq!(body.state(), SwimState::Swimming);
    assert!(body.breath_fraction() > 0.99);
}

#[test]
fn a_swimmer_dives_looking_down_loses_breath_and_comes_back_up() {
    let (ground, water) = pool();
    let mut body = Swimmer::new((10.0, -1.0, 20.0), 0.0);
    body.pitch_deg = -80.0;
    let down = SwimInput {
        forward: 1.0,
        ..SwimInput::default()
    };
    swim(&mut body, &ground, &water, down, 3.0);
    assert_eq!(body.state(), SwimState::Diving);
    assert!(
        body.eye().y < -2.0,
        "should be deep, eye at {}",
        body.eye().y
    );
    assert!(body.breath_fraction() < 1.0);
    assert!(
        body.feet.y > -6.05,
        "went through the floor: {}",
        body.feet.y
    );
    body.pitch_deg = 0.0;
    let up = SwimInput {
        up: 1.0,
        ..SwimInput::default()
    };
    swim(&mut body, &ground, &water, up, 6.0);
    swim(&mut body, &ground, &water, SwimInput::default(), 2.0);
    assert_eq!(body.state(), SwimState::Swimming);
}

#[test]
fn a_swimmer_climbs_out_onto_the_beach_and_walks() {
    let (ground, water) = pool();
    let mut body = Swimmer::new((18.0, -1.5, 20.0), 90.0);
    let ahead = SwimInput {
        forward: 1.0,
        ..SwimInput::default()
    };
    swim(&mut body, &ground, &water, ahead, 15.0);
    let floor = -6.0 + (body.feet.x - 25.0) * 0.6;
    assert!(body.feet.x > 36.0, "stuck at x = {}", body.feet.x);
    assert!(
        matches!(body.state(), SwimState::OnGround | SwimState::Wading),
        "{:?}",
        body.state()
    );
    assert!(
        (body.feet.y - floor).abs() < 0.3,
        "feet {} vs ground {floor}",
        body.feet.y
    );
}

#[test]
fn a_current_carries_a_swimmer_downstream() {
    // A long channel fed at one end and drained at the other.
    let ground = HeightMapTerrain(|x: f32, _| -3.0 - x * 0.01);
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(80.0, 6.0), 1.0, -6.0, 10.0);
    let mut water = Water::new(region, &ground);
    water.flood(Vec3::new(40.0, -3.0, 3.0), -0.5);
    water.add_source(WaterSource {
        position: Vec3::new(0.5, 0.0, 3.0),
        rate: 12.0,
    });
    water.add_source(WaterSource {
        position: Vec3::new(79.5, -3.5, 3.0),
        rate: -12.0,
    });
    run(&mut water, 40.0);
    let mut body = Swimmer::new((30.0, -2.3, 3.0), 0.0);
    let start = body.feet.x;
    for _ in 0..240 {
        body.step(&ground, Some(&water), SwimInput::default(), 1.0 / 60.0);
        water.update(1.0 / 60.0);
    }
    assert!(
        body.feet.x > start + 0.5,
        "not carried: {} -> {}",
        start,
        body.feet.x
    );
}

#[test]
fn a_disturbance_makes_waves_without_changing_the_volume() {
    let (_ground, mut water) = pool();
    let v0 = water.total_volume();
    water.disturb(Vec3::new(12.0, 0.0, 20.0), 2.5, 3.0);
    run(&mut water, 1.0);
    assert_sane(&water);
    assert!(rel_change(v0, water.total_volume()) < 1e-5);
    let dev = wet_heads(&water)
        .iter()
        .map(|h| h.abs())
        .fold(0.0, f32::max);
    assert!(dev > 0.02, "no waves: {dev}");
}

fn beach_shoreline(w: &Water, ground: impl Fn(f32) -> f32) -> f32 {
    (0..w.region().cells_x)
        .rev()
        .find(|&x| {
            let px = x as f32 + 0.5;
            w.sample(Vec3::new(px, ground(px) + 0.01, 1.0))
                .is_some_and(|s| s.water_depth > 0.02)
        })
        .map_or(0.0, |x| x as f32 + 0.5)
}

fn beach_ground(x: f32) -> f32 {
    if x < 60.0 {
        -5.0
    } else {
        -5.0 + (x - 60.0) * 0.08
    }
}

#[test]
fn a_calm_open_sea_stays_still() {
    let beach = HeightMapTerrain(|x: f32, _| beach_ground(x));
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(160.0, 2.0), 1.0, -8.0, 20.0);
    let mut water = Water::new(region, &beach);
    water.flood_sea(0.0);
    water.set_sea(Some(Sea::calm(0.0)));
    let v0 = water.total_volume();
    run(&mut water, 60.0);
    assert_sane(&water);
    assert!(rel_change(v0, water.total_volume()) < 1e-3);
    for h in wet_heads(&water) {
        assert!(h.abs() < 0.01, "head {h}");
    }
}

#[test]
fn swell_from_the_open_sea_runs_up_the_beach_again_and_again() {
    let beach = HeightMapTerrain(|x: f32, _| beach_ground(x));
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(160.0, 2.0), 1.0, -8.0, 20.0);
    let mut water = Water::new(region, &beach);
    water.flood_sea(0.0);
    let still = beach_shoreline(&water, beach_ground);
    water.set_sea(Some(
        Sea::calm(0.0)
            .with_edges(SeaEdges {
                min_x: true,
                ..Default::default()
            })
            .with_swell(Swell {
                height: 0.8,
                period: 10.0,
                direction_deg: 90.0,
            }),
    ));
    let v0 = water.total_volume();
    run(&mut water, 40.0);
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    let (mut vlo, mut vhi) = (f64::MAX, f64::MIN);
    for _ in 0..60 * 60 {
        water.update(1.0 / 60.0);
        let s = beach_shoreline(&water, beach_ground);
        lo = lo.min(s);
        hi = hi.max(s);
        let v = water.total_volume();
        vlo = vlo.min(v);
        vhi = vhi.max(v);
    }
    assert_sane(&water);
    assert!(
        hi - lo > 2.5 && hi > still + 2.0,
        "shoreline should swing: {lo}..{hi} (still {still})"
    );
    // The swell brings water in and takes it out; it does not fill or drain.
    let rel = |v: f64| (v - v0).abs() / v0;
    assert!(
        rel(vlo) < 0.05 && rel(vhi) < 0.05,
        "volume {vlo}..{vhi} from {v0}"
    );
}

#[test]
fn waves_leave_through_an_open_edge_instead_of_bouncing_back() {
    let flat = HeightMapTerrain(|_, _| -4.0);
    let region = WaterRegion::new(Vec2::ZERO, Vec2::new(120.0, 2.0), 1.0, -6.0, 20.0);
    let mut water = Water::new(region, &flat);
    water.flood_sea(0.0);
    water.set_sea(Some(Sea::calm(0.0)));
    for x in 50..70 {
        let t = (x as f32 - 60.0) / 10.0;
        let h = 0.6 * (1.0 - t * t).max(0.0);
        for z in 0..2 {
            water.pour(Vec3::new(x as f32 + 0.5, -3.0, z as f32 + 0.5), h);
        }
    }
    // Waves cross 60 m at √(g·4) ≈ 6.3 m/s in ~10 s.
    run(&mut water, 30.0);
    assert_sane(&water);
    let worst = wet_heads(&water)
        .into_iter()
        .fold(0.0f32, |m, h| m.max(h.abs()));
    assert!(worst < 0.05, "waves should be gone, still {worst} m");
}

#[test]
#[ignore]
fn bench_water_1024_mostly_dry() {
    let noise = crate::proc::Noise::new(11);
    let terrain = HeightMapTerrain(move |x: f32, z: f32| {
        noise.fbm2(x * 0.004, z * 0.004, 5, 2.0, 0.5) * 40.0 + 10.0
    });
    let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(1024.0), 1.0, -40.0, 80.0);
    let started = std::time::Instant::now();
    let mut water = Water::new(region, &terrain);
    eprintln!(
        "build {:.1} ms, {} cells",
        started.elapsed().as_secs_f32() * 1e3,
        water.cell_count()
    );
    // A few lakes and a river source; most of the land stays dry.
    water.flood_sea(0.0);
    water.add_source(WaterSource {
        position: Vec3::new(512.0, 70.0, 512.0),
        rate: 5.0,
    });
    for _ in 0..60 {
        water.update(1.0 / 60.0);
    }
    let started = std::time::Instant::now();
    for _ in 0..120 {
        water.update(1.0 / 60.0);
    }
    let s = water.stats();
    eprintln!(
        "update {:.2} ms/frame, {} substeps, {} wet cells, tiles {:?}",
        started.elapsed().as_secs_f32() * 1e3 / 120.0,
        s.substeps,
        s.wet_cells,
        water.active_tiles()
    );
    // Everything stirred up: the worst case.
    water.rain(0.05);
    let started = std::time::Instant::now();
    for _ in 0..60 {
        water.update(1.0 / 60.0);
    }
    eprintln!(
        "after rain {:.2} ms/frame, tiles {:?}",
        started.elapsed().as_secs_f32() * 1e3 / 60.0,
        water.active_tiles()
    );
}

#[test]
fn a_still_lake_sleeps_and_wakes_when_disturbed() {
    let bowl = HeightMapTerrain(|x: f32, z: f32| ((x - 64.0).powi(2) + (z - 64.0).powi(2)) * 0.002);
    let region = WaterRegion::new(Vec2::ZERO, Vec2::splat(128.0), 1.0, -5.0, 40.0);
    let far = Vec3::new(90.0, 2.0, 64.0);
    // The same splash with sleeping tiles and with everything running.
    let trace = |never_sleep: bool| {
        let mut water = Water::new(region, &bowl);
        water.activity.never_sleep = never_sleep;
        water.flood(Vec3::new(64.0, 0.1, 64.0), 3.0);
        run(&mut water, 20.0);
        let (running, total) = water.active_tiles();
        if !never_sleep {
            assert!(
                running < total / 4,
                "a still lake should mostly sleep: {running}/{total}"
            );
        }
        let v0 = water.total_volume();
        water.disturb(Vec3::new(40.0, 3.0, 64.0), 6.0, 3.0);
        let still = water.surface_at(far).expect("far side wet");
        let mut series = Vec::new();
        for _ in 0..60 * 15 {
            water.update(1.0 / 60.0);
            series.push(water.surface_at(far).unwrap() - still);
        }
        assert!(rel_change(v0, water.total_volume()) < 1e-5);
        run(&mut water, 90.0);
        if !never_sleep {
            let (running, _) = water.active_tiles();
            assert!(
                running < total / 4,
                "and go back to sleep: {running}/{total}"
            );
        }
        series
    };
    let asleep = trace(false);
    let awake = trace(true);
    let peak = awake.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak > 2e-4, "the wave should cross the lake, peak {peak}");
    let worst = asleep
        .iter()
        .zip(&awake)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst < peak * 0.1,
        "sleeping changed the wave: {worst} vs peak {peak}"
    );
}
