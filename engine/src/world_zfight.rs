//! Z-fighting checks on the World: see [`crate::zfight`].

use super::{EntityId, SurfaceMaterialRef, World};
use crate::color::rgb;
use crate::mesh::{BuiltMesh, Mesh, SurfaceMaterial};
use crate::portal::SpaceId;
use crate::zfight::{self, Options, ZFightReport};
use glam::{Mat4, Vec3, Vec4};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// What [`World::find_z_fighting_with`] looks at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldScan {
    pub options: Options,
    /// Only meshes (or instances) reaching within this many metres of
    /// `centre`. `None` scans the whole world.
    pub radius: Option<f32>,
    /// Where `radius` is measured from; `None` is the camera.
    pub centre: Option<Vec3>,
    /// Stop collecting after this many triangles (the scan says so in its
    /// notes). Keeps a huge forest from stalling the frame.
    pub max_triangles: usize,
    /// Instanced meshes and LOD sets (forests, scattered props) too.
    pub instanced: bool,
    /// Leave out clashes buried under the terrain height field (the bottoms
    /// of rocks and trees sunk into the ground), which nobody can see.
    pub skip_under_terrain: bool,
}

impl Default for WorldScan {
    fn default() -> Self {
        Self {
            options: Options::default(),
            radius: None,
            centre: None,
            max_triangles: 4_000_000,
            instanced: true,
            skip_under_terrain: true,
        }
    }
}

impl WorldScan {
    /// Everything within `radius` metres of the camera.
    pub fn around_camera(radius: f32) -> Self {
        Self {
            radius: Some(radius),
            ..Self::default()
        }
    }
}

/// Most posts [`World::show_z_fighting`] puts up (the worst spots first).
pub const MAX_Z_FIGHTING_MARKS: usize = 500;

/// How long the watcher waits before its first scan, so an app can put
/// things where they belong first (s).
const FIRST_SCAN_DELAY: f32 = 0.5;

/// Z-fighting markers and the `ENGINE_ZFIGHT` watcher.
#[derive(Debug, Default)]
pub(crate) struct ZFightState {
    marks: Option<EntityId>,
    watch: Option<Watch>,
}

#[derive(Debug)]
struct Watch {
    scan: WorldScan,
    /// Render epoch after our own markers went in.
    seen_epoch: Option<u64>,
    seen_eye: Vec3,
    cooldown: f32,
    last: Option<(usize, usize)>,
}

impl World {
    /// Give an entity a name for debugging tools: z-fighting reports use
    /// it instead of "entity 12". Despawning forgets it.
    pub fn set_name(&mut self, id: EntityId, name: impl Into<String>) {
        self.entity_names.insert(id, name.into());
    }

    /// The name given with [`Self::set_name`].
    pub fn name(&self, id: EntityId) -> Option<&str> {
        self.entity_names.get(&id).map(String::as_str)
    }

    /// Every z-fighting clash in the scene: entities (with their instances
    /// and the detailed level of LOD sets), each space on its own. Water,
    /// terrain clipmaps, skinned characters and the markers themselves are
    /// left out. See [`crate::zfight`].
    pub fn find_z_fighting(&self) -> ZFightReport {
        self.find_z_fighting_with(&WorldScan::default())
    }

    /// [`Self::find_z_fighting`] with a radius, triangle budget or other
    /// options.
    pub fn find_z_fighting_with(&self, scan: &WorldScan) -> ZFightReport {
        let t = Instant::now();
        let centre = scan.centre.unwrap_or_else(|| self.camera.eye());
        // LOD sets: the detailed level at every place, never the coarser
        // levels (they stand in the same spot on purpose).
        let mut lod_place: HashMap<EntityId, (crate::lod::LodId, usize)> = HashMap::new();
        for (id, set) in &self.lods {
            for (k, (e, _)) in set.levels.iter().enumerate() {
                lod_place.insert(*e, (*id, k));
            }
        }
        let mut prefab_of: HashMap<EntityId, &str> = HashMap::new();
        for (root, made) in &self.prefab_instances {
            prefab_of.insert(*root, made.prefab.as_str());
            for part in &made.parts {
                prefab_of.insert(*part, made.prefab.as_str());
            }
        }

        // First decide what goes in (cheap), then make the triangles in
        // parallel.
        struct Job<'a> {
            mesh: &'a BuiltMesh,
            transform: Mat4,
            tint: Vec4,
            look: u64,
            space: u32,
            source: u32,
        }
        let mut jobs: Vec<Job> = Vec::new();
        let mut per_space: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        let mut collected = 0usize;
        let mut notes = Vec::new();
        let mut left_out = 0usize;
        for id in &self.order {
            if Some(*id) == self.zfight.marks {
                continue;
            }
            let Some(e) = self.entities.get(id) else {
                continue;
            };
            if matches!(e.material, Some(SurfaceMaterialRef::Water(_))) {
                continue;
            }
            let mesh: &BuiltMesh = match e.instance_of {
                Some(proto) => match self.entities.get(&proto) {
                    Some(p) => &p.mesh,
                    None => continue,
                },
                None => &e.mesh,
            };
            if mesh.indices.is_empty() {
                continue;
            }
            let lod = lod_place.get(id).copied();
            if (e.instanced || lod.is_some()) && !scan.instanced {
                continue;
            }
            let (instances, label): (Vec<(Mat4, Vec4)>, String) = match lod {
                Some((_, level)) if level > 0 => continue,
                Some((lod_id, _)) => (
                    self.lods[&lod_id]
                        .places
                        .iter()
                        .map(|i| (i.transform, tint(i.tint)))
                        .collect(),
                    format!("LOD set {} (entity {id})", lod_id.0),
                ),
                None if e.instanced => (
                    e.instances
                        .iter()
                        .map(|i| (i.transform, tint(i.tint)))
                        .collect(),
                    format!("entity {id} (instanced)"),
                ),
                None => (vec![(e.transform, Vec4::ONE)], format!("entity {id}")),
            };
            let (local_centre, local_radius) = bounds(mesh);
            let label = match (self.entity_names.get(id), prefab_of.get(id)) {
                (Some(name), _) => name.clone(),
                (None, Some(p)) => format!("{label}, prefab {p:?}"),
                (None, None) if instances.len() == 1 => {
                    let c = instances[0].0.transform_point3(local_centre);
                    format!("{label} around ({:.1}, {:.1}, {:.1})", c.x, c.y, c.z)
                }
                (None, None) => label,
            };
            let look = look_key(mesh, e);
            let space = e.space.raw();
            let sources = per_space.entry(space).or_default();
            let source = sources.len() as u32;
            let mut used = false;
            let n = mesh.indices.len() / 3;
            for (k, (m, tint)) in instances.iter().enumerate() {
                if let Some(r) = scan.radius {
                    let c = m.transform_point3(local_centre);
                    let scale = m
                        .x_axis
                        .truncate()
                        .length()
                        .max(m.y_axis.truncate().length())
                        .max(m.z_axis.truncate().length());
                    if c.distance(centre) - local_radius * scale > r {
                        continue;
                    }
                }
                if collected + n > scan.max_triangles {
                    left_out += instances.len() - k;
                    break;
                }
                collected += n;
                jobs.push(Job {
                    mesh,
                    transform: *m,
                    tint: *tint,
                    look,
                    space,
                    source,
                });
                used = true;
            }
            if used {
                sources.push(label);
            }
        }
        if left_out > 0 {
            notes.push(format!(
                "stopped at {} triangles, {left_out} meshes or instances not checked: scan a radius or raise max_triangles",
                scan.max_triangles
            ));
        }

        use rayon::prelude::*;
        let mut report = ZFightReport::default();
        for (space, sources) in per_space {
            let tris: Vec<zfight::Tri> = jobs
                .par_iter()
                .filter(|j| j.space == space)
                .flat_map_iter(|j| {
                    zfight::triangles_tinted(j.mesh, j.transform, j.tint, j.look, j.source)
                })
                .collect();
            let mut clashes = zfight::find(&tris, &scan.options);
            if scan.skip_under_terrain && space == SpaceId::DEFAULT.raw() {
                if let Some(field) = &self.height_field {
                    let before = clashes.len();
                    // Under the surface by more than a centimetre: hidden.
                    clashes.retain(|c| {
                        let p = c.at.as_vec3();
                        p.y > field.height_at(p.x, p.z) - 0.01
                    });
                    let buried = before - clashes.len();
                    if buried > 0 {
                        notes.push(format!(
                            "{buried} clashes under the terrain left out (nobody sees them)"
                        ));
                    }
                }
            }
            report.merge(ZFightReport {
                triangles: tris.len(),
                clashes,
                sources,
                notes: Vec::new(),
                elapsed: Default::default(),
            });
        }
        report.notes = notes;
        report.elapsed = t.elapsed();
        report
    }

    /// Marks every flickering spot of `report` with a glowing magenta post
    /// (replacing earlier marks) and returns how many spots there are. Only
    /// the worst [`MAX_Z_FIGHTING_MARKS`] get a post, so a badly broken
    /// scene stays drawable. The posts are left out of later scans.
    pub fn show_z_fighting(&mut self, report: &ZFightReport) -> usize {
        self.clear_z_fighting_marks();
        let spots = report.spots(2.0);
        if spots.is_empty() {
            return 0;
        }
        let mut mesh = Mesh::new();
        let magenta = rgb(255, 0, 255);
        for s in spots.iter().take(MAX_Z_FIGHTING_MARKS) {
            let p = s.at.as_vec3();
            let n = s.normal.as_vec3();
            // A post standing up from the spot, and a knob sitting on the
            // fighting surface so it shows on walls and ceilings too.
            let _ = mesh.add_box(p + Vec3::Y * 1.5, (0.12, 3.0, 0.12), magenta);
            let _ = mesh.add_box(p + n * 0.1, (0.35, 0.35, 0.35), magenta);
        }
        mesh.set_surface_material(SurfaceMaterial {
            emission: 2.0,
            ..SurfaceMaterial::DEFAULT
        });
        let space = self.spawn_space;
        self.spawn_space = SpaceId::DEFAULT;
        let id = self.spawn(mesh);
        self.spawn_space = space;
        let _ = self.set_casts_shadow(id, false);
        self.zfight.marks = Some(id);
        spots.len()
    }

    /// Removes the posts placed by [`Self::show_z_fighting`].
    pub fn clear_z_fighting_marks(&mut self) {
        if let Some(id) = self.zfight.marks.take() {
            self.despawn(id);
        }
    }

    /// Finds z-fighting around the camera (250 m), prints the report and
    /// marks the spots. The one-line debugging call.
    pub fn check_z_fighting(&mut self) -> ZFightReport {
        let report = self.find_z_fighting_with(&WorldScan::around_camera(250.0));
        println!("{report}");
        self.show_z_fighting(&report);
        report
    }

    /// Keep checking while the app runs: scan with `scan` (its radius
    /// follows the camera) whenever the scene changes or the camera moves a
    /// quarter of the radius, print what changed and keep the marks up.
    /// `None` stops and removes the marks. `ENGINE_ZFIGHT=1` turns this on
    /// for any app.
    pub fn watch_z_fighting(&mut self, scan: Option<WorldScan>) {
        self.zfight.watch = scan.map(|scan| Watch {
            scan: WorldScan {
                centre: None,
                ..scan
            },
            seen_epoch: None,
            seen_eye: Vec3::ZERO,
            cooldown: FIRST_SCAN_DELAY,
            last: None,
        });
        if self.zfight.watch.is_none() {
            self.clear_z_fighting_marks();
        }
    }

    /// Whether the z-fighting watcher runs (it reads every mesh, so CPU
    /// copies are kept meanwhile).
    pub(crate) fn zfight_watching(&self) -> bool {
        self.zfight.watch.is_some()
    }

    /// `ENGINE_ZFIGHT`: unset or `0` off, `1` 250 m around the camera, a
    /// number that many metres, `all` the whole world.
    pub(crate) fn watch_z_fighting_from_env(&mut self) {
        let Ok(value) = std::env::var("ENGINE_ZFIGHT") else {
            return;
        };
        let value = value.trim().to_ascii_lowercase();
        let radius = match value.as_str() {
            "" | "0" | "off" | "false" => return,
            "1" | "on" | "true" => Some(250.0),
            "all" => None,
            v => match v.parse::<f32>() {
                Ok(r) if r > 1.0 && r.is_finite() => Some(r),
                _ => {
                    eprintln!(
                        "ENGINE_ZFIGHT={v:?} not understood: use 1, a radius in metres, or all"
                    );
                    return;
                }
            },
        };
        println!(
            "z-fighting watch on ({}): spots are marked with magenta posts",
            radius.map_or("whole world".to_string(), |r| format!(
                "{r} m around the camera"
            ))
        );
        self.watch_z_fighting(Some(WorldScan {
            radius,
            ..WorldScan::default()
        }));
    }

    pub(crate) fn tick_zfight_watch(&mut self, dt: f32) {
        let Some(w) = self.zfight.watch.as_mut() else {
            return;
        };
        w.cooldown -= dt;
        if w.cooldown > 0.0 {
            return;
        }
        let eye = self.camera.eye();
        let moved = w
            .scan
            .radius
            .is_some_and(|r| eye.distance(w.seen_eye) > r * 0.25);
        if w.seen_epoch == Some(self.scene_epoch) && !moved {
            return;
        }
        let scan = w.scan;
        let report = self.find_z_fighting_with(&scan);
        let spots = self.show_z_fighting(&report);
        let doubled = report.doubled().count();
        let epoch = self.scene_epoch;
        let w = self.zfight.watch.as_mut().expect("watch still on");
        if w.last != Some((spots, doubled)) {
            println!("{}", report.summary(15));
            w.last = Some((spots, doubled));
        }
        w.seen_epoch = Some(epoch);
        w.seen_eye = eye;
        // A big scan costs frames: wait longer after a slow one.
        w.cooldown = (report.elapsed.as_secs_f32() * 10.0).clamp(1.0, 10.0);
    }
}

fn tint(c: crate::color::Color) -> Vec4 {
    Vec4::new(c.r, c.g, c.b, c.a)
}

/// Local bounding sphere (centre of the box, half its diagonal).
fn bounds(mesh: &BuiltMesh) -> (Vec3, f32) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in &mesh.positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    if lo.x > hi.x {
        return (Vec3::ZERO, 0.0);
    }
    ((lo + hi) * 0.5, (hi - lo).length() * 0.5)
}

/// What besides vertex colour sets the look: texture, material and surface.
fn look_key(mesh: &BuiltMesh, e: &super::Entity) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}|{:?}|{:?}", e.albedo, e.material, mesh.surface).hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod::LodLevel;
    use crate::mesh::Shape;
    use crate::place::Place;

    fn ground() -> Mesh {
        Shape::box_at((0.0, -0.05, 0.0), (40.0, 0.1, 40.0), rgb(90, 120, 70)).unwrap()
    }

    fn rug(x: f32) -> Mesh {
        Shape::box_at((x, 0.0, 0.0), (2.0, 0.0001, 1.0), rgb(160, 30, 30)).unwrap()
    }

    #[test]
    fn finds_a_rug_on_the_ground_and_marks_it() {
        let mut world = World::new();
        world.spawn(ground());
        world.place(rug(0.0), Place::new(5.0, 0.0, 5.0)).unwrap();
        let report = world.find_z_fighting();
        assert!(!report.is_clean(), "{report}");
        let spot = report.spots(10.0)[0];
        assert!((spot.at.x - 5.0).abs() < 1.0, "{spot:?}");
        assert!((spot.area - 2.0).abs() < 1e-3, "{}", spot.area);

        assert_eq!(world.show_z_fighting(&report), 1);
        // The posts themselves are never reported, and replacing them keeps one.
        let again = world.find_z_fighting();
        assert_eq!(again.spots(10.0).len(), 1);
        world.show_z_fighting(&again);
        assert_eq!(world.entity_count(), 3);
        world.clear_z_fighting_marks();
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn instances_lods_and_radius() {
        let mut world = World::new();
        world.spawn(ground());
        // Two instances of one rug, overlapping each other and the ground.
        let id = world.spawn_instanced(rug(0.0));
        world
            .set_instances(
                id,
                &[Place::new(-10.0, 0.0, 0.0), Place::new(-9.0, 0.0, 0.0)],
            )
            .unwrap();
        // A LOD set whose levels stand in the same place: not a clash.
        let block = |c| Shape::box_at((0.0, 0.5, 0.0), (1.0, 1.0, 1.0), c).unwrap();
        world
            .spawn_lod_at(
                vec![
                    LodLevel::new(block(rgb(200, 200, 200)), 20.0),
                    LodLevel::new(block(rgb(100, 100, 100)), 80.0),
                ],
                &[Place::new(10.0, 0.0, 10.0)],
            )
            .unwrap();
        let report = world.find_z_fighting_with(&WorldScan {
            options: Options::ground_at(0.0),
            ..WorldScan::default()
        });
        let spots = report.spots(10.0);
        assert!(spots.iter().all(|s| s.at.x < 0.0), "{report}");
        assert!(!spots.is_empty());

        // Only 5 m around (10, 0, 10): the rugs are out of reach.
        let near = world.find_z_fighting_with(&WorldScan {
            options: Options::ground_at(0.0),
            radius: Some(5.0),
            centre: Some(Vec3::new(10.0, 0.0, 10.0)),
            ..WorldScan::default()
        });
        assert!(near.is_clean(), "{near}");
    }

    #[test]
    fn a_tiny_budget_says_what_it_skipped() {
        let mut world = World::new();
        world.spawn(ground());
        world.spawn(rug(0.0));
        let report = world.find_z_fighting_with(&WorldScan {
            max_triangles: 12,
            ..WorldScan::default()
        });
        assert_eq!(report.triangles, 12);
        assert!(
            report.notes.iter().any(|n| n.contains("stopped")),
            "{report}"
        );
    }

    #[test]
    fn the_watch_rescans_after_changes() {
        let mut world = World::new();
        world.spawn(ground());
        world.watch_z_fighting(Some(WorldScan::default()));
        world.tick_zfight_watch(0.1);
        assert!(world.zfight.marks.is_none());
        let rug = world.spawn(rug(0.0));
        world.tick_zfight_watch(2.0);
        assert!(world.zfight.marks.is_some());
        world.despawn(rug);
        world.tick_zfight_watch(2.0);
        assert!(world.zfight.marks.is_none());
        world.watch_z_fighting(None);
    }

    #[test]
    fn a_badly_broken_scene_gets_a_limited_number_of_posts() {
        let mut world = World::new();
        world.spawn(
            Shape::box_at((0.0, -0.05, 0.0), (400.0, 0.1, 400.0), rgb(90, 120, 70)).unwrap(),
        );
        for i in 0..30 {
            for j in 0..30 {
                world
                    .place(
                        rug(0.0),
                        Place::new(i as f32 * 6.0 - 90.0, 0.0, j as f32 * 6.0 - 90.0),
                    )
                    .unwrap();
            }
        }
        let report = world.find_z_fighting();
        let spots = world.show_z_fighting(&report);
        assert!(spots > MAX_Z_FIGHTING_MARKS, "{spots}");
        let marks = world.zfight.marks.expect("posts");
        let all = world.entities[&marks].mesh.positions.len();
        // One spot's post, for scale.
        let mut one = report.clone();
        one.clashes.truncate(1);
        world.show_z_fighting(&one);
        let marks = world.zfight.marks.expect("post");
        let per_spot = world.entities[&marks].mesh.positions.len();
        assert_eq!(all, per_spot * MAX_Z_FIGHTING_MARKS);
    }

    #[test]
    fn clashes_buried_under_the_terrain_are_left_out() {
        let rules = crate::proc_terrain::demo_terrain_rules();
        let mut world = World::new();
        world.set_proc_terrain(crate::proc_terrain::ProcTerrain::gpu_clipmap(
            rules,
            Default::default(),
        ));
        let h = world.proc_height_at(5.0, 5.0);
        // Two rocks sunk into the ground with only their bottoms in one
        // plane, and two rugs of different colours on top of each other well
        // above it.
        world
            .place(
                Shape::box_at((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), rgb(120, 120, 120)).unwrap(),
                Place::new(5.0, h - 0.4, 5.0),
            )
            .unwrap();
        world
            .place(
                Shape::box_at((0.0, 0.1, 0.0), (1.0, 1.2, 1.0), rgb(140, 130, 120)).unwrap(),
                Place::new(5.4, h - 0.4, 5.3),
            )
            .unwrap();
        world
            .place(rug(0.0), Place::new(5.0, h + 3.0, 5.0))
            .unwrap();
        world
            .place(
                Shape::box_at((0.5, 0.0, 0.0), (2.0, 0.0001, 1.0), rgb(30, 30, 160)).unwrap(),
                Place::new(5.0, h + 3.0, 5.0),
            )
            .unwrap();
        let report = world.find_z_fighting();
        assert!(
            report.fighting().all(|c| c.at.y > h as f64 + 2.0),
            "{report}"
        );
        assert!(!report.is_clean(), "{report}");
        assert!(
            report.notes.iter().any(|n| n.contains("under the terrain")),
            "{report}"
        );
        let all = world.find_z_fighting_with(&WorldScan {
            skip_under_terrain: false,
            ..WorldScan::default()
        });
        assert!(all.fighting().count() > report.fighting().count(), "{all}");
    }
}
