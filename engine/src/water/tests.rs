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
        "update {:.2} ms/frame, {} substeps, {} wet cells",
        started.elapsed().as_secs_f32() * 1e3 / 120.0,
        s.substeps,
        s.wet_cells
    );
}
