//! Verify runs: let a program check itself and say so in a file.
//!
//! `ENGINE_VERIFY=<folder>` runs the game for a fixed number of frames with a
//! fixed time step, then writes into the folder:
//!
//! - `screenshot.png`: the last frame,
//! - `report.json`: passed or failed and why, the frames run, the game's own
//!   [`World::check`] results and [`World::report`] values, counts of
//!   entities, bodies and lamps, the camera, and the first entities' places,
//!
//! and exits. The run fails (the program returns an error, so the exit code is
//! not 0) when the game returned an error, a check was false, or the picture
//! differs from a reference too much. Made for agents and CI: run it, read
//! `report.json`, look at the screenshot.
//!
//! | Variable | Meaning |
//! |---|---|
//! | `ENGINE_VERIFY=<folder>` | Turn it on; where the files go |
//! | `ENGINE_VERIFY_FRAMES=120` | Frames to run (fixed step `ENGINE_FIXED_DT`, default 1/60 s) |
//! | `ENGINE_VERIFY_SHOTS=30,60` | Extra screenshots `shot-<frame>.png` |
//! | `ENGINE_VERIFY_REFERENCE=<png>` | Compare the last frame with this picture |
//! | `ENGINE_VERIFY_TOLERANCE=0.002` | Share of pixels that may differ by more than 8/255 |
//!
//! `tools/verify.py <example>` builds an example and runs it this way.

use crate::error::{EngineError, EngineResult};
use crate::world::World;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Frames a verify run lasts unless `ENGINE_VERIFY_FRAMES` says otherwise.
const DEFAULT_FRAMES: u32 = 120;
/// A pixel differs when a channel is off by more than this.
const PIXEL_THRESHOLD: u8 = 8;
const DEFAULT_TOLERANCE: f64 = 0.002;
/// Entities listed by place in the report: enough to see where things are,
/// small enough that an agent reading the report does not drown in it. Games
/// put what matters into `world.report`.
const LISTED_ENTITIES: usize = 20;

pub(crate) struct Verify {
    pub dir: PathBuf,
    pub frames: u32,
    pub shots: Vec<u32>,
    reference: Option<PathBuf>,
    tolerance: f64,
    title: String,
    started: Instant,
}

impl Verify {
    /// `None` unless `ENGINE_VERIFY` is set.
    pub fn from_env(title: &str) -> EngineResult<Option<Self>> {
        let Some(dir) = std::env::var_os("ENGINE_VERIFY").filter(|v| !v.is_empty()) else {
            return Ok(None);
        };
        let number = |name: &str| -> EngineResult<Option<String>> {
            Ok(std::env::var(name).ok().filter(|v| !v.trim().is_empty()))
        };
        let frames = match number("ENGINE_VERIFY_FRAMES")? {
            Some(v) => v
                .trim()
                .parse::<u32>()
                .ok()
                .filter(|f| *f > 0)
                .ok_or_else(|| {
                    EngineError::InvalidValue(format!(
                        "ENGINE_VERIFY_FRAMES must be a whole number above 0, not {v:?}"
                    ))
                })?,
            None => DEFAULT_FRAMES,
        };
        let shots = match number("ENGINE_VERIFY_SHOTS")? {
            Some(v) => v
                .split(',')
                .map(|s| {
                    s.trim().parse::<u32>().map_err(|_| {
                        EngineError::InvalidValue(format!(
                            "ENGINE_VERIFY_SHOTS must be frame numbers like 30,60, not {v:?}"
                        ))
                    })
                })
                .collect::<EngineResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        let tolerance = match number("ENGINE_VERIFY_TOLERANCE")? {
            Some(v) => v
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|t| (0.0..=1.0).contains(t))
                .ok_or_else(|| {
                    EngineError::InvalidValue(format!(
                        "ENGINE_VERIFY_TOLERANCE must be a share between 0 and 1, not {v:?}"
                    ))
                })?,
            None => DEFAULT_TOLERANCE,
        };
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        Ok(Some(Self {
            dir,
            frames,
            shots,
            reference: std::env::var_os("ENGINE_VERIFY_REFERENCE")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            tolerance,
            title: title.to_string(),
            started: Instant::now(),
        }))
    }

    pub fn screenshot_path(&self) -> PathBuf {
        self.dir.join("screenshot.png")
    }

    pub fn shot_path(&self, frame: u32) -> PathBuf {
        self.dir.join(format!("shot-{frame}.png"))
    }

    fn report_path(&self) -> PathBuf {
        self.dir.join("report.json")
    }

    /// Write a panic into `report.json` before the default hook prints it,
    /// so a crashed run still leaves a report behind.
    pub fn install_panic_hook(&self) {
        let path = self.report_path();
        let title = self.title.clone();
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload()
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| info.payload().downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".into());
            let report = serde_json::json!({
                "passed": false,
                "status": "panic",
                "program": title,
                "engine": crate::version::version(),
                "problems": [format!("panic: {message}")],
                "panic_location": info.location().map(|l| l.to_string()),
            });
            if let Ok(text) = serde_json::to_string_pretty(&report) {
                let _ = std::fs::write(&path, text);
            }
            previous(info);
        }));
    }

    /// Write `report.json` and say how it went on stderr. `Err` when the
    /// run failed, with the problems as the message.
    pub fn finish(
        &self,
        world: &World,
        frames_run: u32,
        fatal_error: Option<&str>,
        screenshot: Option<&Path>,
    ) -> EngineResult<()> {
        let mut problems = Vec::new();
        if let Some(error) = fatal_error {
            problems.push(format!("the game returned an error: {error}"));
        }
        let log = world.verify_log();
        for (name, check) in &log.checks {
            if check.failures > 0 {
                problems.push(format!(
                    "check {name:?} failed {} of {} times (first in frame {})",
                    check.failures,
                    check.runs,
                    check.first_failure_frame.unwrap_or(0)
                ));
            }
        }
        let image = match (&self.reference, screenshot) {
            (Some(reference), Some(shot)) => {
                let diff = compare_images(reference, shot)?;
                if let Some(problem) = diff.problem(self.tolerance) {
                    problems.push(problem);
                }
                Some(diff)
            }
            (Some(_), None) => {
                problems.push("no screenshot to compare with the reference".into());
                None
            }
            _ => None,
        };
        let wall_s = self.started.elapsed().as_secs_f64();
        let camera = world.camera();
        let report = Report {
            passed: problems.is_empty(),
            status: if fatal_error.is_some() {
                "error"
            } else if problems.is_empty() {
                "passed"
            } else {
                "failed"
            },
            problems: problems.clone(),
            program: &self.title,
            engine: crate::version::version(),
            gpu: world
                .gpu_info()
                .map(|g| format!("{} ({})", g.name, g.backend)),
            frames: frames_run,
            sim_seconds: world.time(),
            wall_seconds: wall_s,
            screenshot: screenshot.map(|p| p.display().to_string()),
            image,
            checks: &log.checks,
            values: &log.values,
            world: WorldSummary {
                entities: world.entity_count(),
                animated: world.animated_entities().count(),
                lamps: world.lamps().count(),
                bodies: world.body_count().0,
                bodies_awake: world.body_count().1,
                camera_eye: camera.eye().to_array(),
                camera_target: camera.target().to_array(),
                first_entities: world
                    .entities()
                    .take(LISTED_ENTITIES)
                    .map(|(id, e)| EntitySummary {
                        id: id.raw(),
                        position: e.transform().w_axis.truncate().to_array(),
                        instances: e.instances.len(),
                    })
                    .collect(),
            },
        };
        let text = serde_json::to_string_pretty(&report)
            .map_err(|e| EngineError::InvalidValue(format!("verify report: {e}")))?;
        std::fs::write(self.report_path(), text)?;
        let checks = log.checks.len();
        if problems.is_empty() {
            eprintln!(
                "verify: PASSED after {frames_run} frames ({checks} checks); report {}",
                self.report_path().display()
            );
            Ok(())
        } else {
            eprintln!(
                "verify: FAILED after {frames_run} frames; report {}",
                self.report_path().display()
            );
            for p in &problems {
                eprintln!("  - {p}");
            }
            Err(EngineError::Verify(problems.join("; ")))
        }
    }
}

#[derive(Serialize)]
struct Report<'a> {
    passed: bool,
    status: &'static str,
    problems: Vec<String>,
    program: &'a str,
    engine: &'static str,
    gpu: Option<String>,
    frames: u32,
    sim_seconds: f32,
    wall_seconds: f64,
    screenshot: Option<String>,
    image: Option<ImageDiff>,
    checks: &'a std::collections::BTreeMap<String, crate::world::world_verify::CheckState>,
    values: &'a std::collections::BTreeMap<String, serde_json::Value>,
    world: WorldSummary,
}

#[derive(Serialize)]
struct WorldSummary {
    entities: usize,
    animated: usize,
    lamps: usize,
    bodies: usize,
    bodies_awake: usize,
    camera_eye: [f32; 3],
    camera_target: [f32; 3],
    first_entities: Vec<EntitySummary>,
}

#[derive(Serialize)]
struct EntitySummary {
    id: u64,
    position: [f32; 3],
    instances: usize,
}

/// How a screenshot compares with a reference picture.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ImageDiff {
    pub reference: String,
    /// Pixels where a colour channel differs by more than 8/255.
    pub differing_pixels: u64,
    pub total_pixels: u64,
    pub differing_share: f64,
    /// The largest channel difference anywhere (0..255).
    pub max_difference: u8,
    /// Set when the pictures could not be compared (size mismatch).
    pub mismatch: Option<String>,
}

impl ImageDiff {
    fn problem(&self, tolerance: f64) -> Option<String> {
        if let Some(m) = &self.mismatch {
            return Some(format!("picture differs from the reference: {m}"));
        }
        (self.differing_share > tolerance).then(|| {
            format!(
                "picture differs from the reference in {:.3}% of pixels (allowed {:.3}%), max {} of 255",
                self.differing_share * 100.0,
                tolerance * 100.0,
                self.max_difference
            )
        })
    }
}

pub(crate) fn compare_images(reference: &Path, shot: &Path) -> EngineResult<ImageDiff> {
    let open = |p: &Path| {
        image::open(p)
            .map(|i| i.to_rgb8())
            .map_err(|e| EngineError::InvalidValue(format!("cannot read {}: {e}", p.display())))
    };
    let (a, b) = (open(reference)?, open(shot)?);
    let mut diff = ImageDiff {
        reference: reference.display().to_string(),
        differing_pixels: 0,
        total_pixels: u64::from(b.width()) * u64::from(b.height()),
        differing_share: 0.0,
        max_difference: 0,
        mismatch: None,
    };
    if a.dimensions() != b.dimensions() {
        diff.mismatch = Some(format!(
            "size {}x{} vs reference {}x{} (set ENGINE_WIDTH / ENGINE_HEIGHT to match)",
            b.width(),
            b.height(),
            a.width(),
            a.height()
        ));
        return Ok(diff);
    }
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let d =
            pa.0.iter()
                .zip(pb.0.iter())
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap_or(0);
        diff.max_difference = diff.max_difference.max(d);
        if d > PIXEL_THRESHOLD {
            diff.differing_pixels += 1;
        }
    }
    diff.differing_share = diff.differing_pixels as f64 / diff.total_pixels.max(1) as f64;
    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, f: impl Fn(u32, u32) -> [u8; 3]) {
        let img = image::RgbImage::from_fn(10, 10, |x, y| image::Rgb(f(x, y)));
        img.save(path).unwrap();
    }

    #[test]
    fn image_compare_counts_pixels_over_the_threshold() {
        let dir = std::env::temp_dir().join(format!("engine-verify-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.png"), dir.join("b.png"));
        write(&a, |_, _| [100, 100, 100]);
        write(&b, |x, y| {
            if x == 0 && y < 3 {
                [150, 100, 100]
            } else {
                [104, 100, 100]
            }
        });
        let diff = compare_images(&a, &b).unwrap();
        assert_eq!(diff.differing_pixels, 3);
        assert_eq!(diff.max_difference, 50);
        assert!(diff.problem(0.05).is_none());
        assert!(diff.problem(0.01).is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checks_fail_the_run_and_values_are_kept() {
        let mut world = World::new();
        assert!(world.check("fine", true));
        assert!(!world.check("broken", false));
        world.report("score", 42);
        assert_eq!(world.failed_checks(), vec!["broken".to_string()]);
        let v = Verify {
            dir: std::env::temp_dir().join(format!("engine-verify-r-{}", std::process::id())),
            frames: 1,
            shots: Vec::new(),
            reference: None,
            tolerance: DEFAULT_TOLERANCE,
            title: "test".into(),
            started: Instant::now(),
        };
        std::fs::create_dir_all(&v.dir).unwrap();
        let result = v.finish(&world, 1, None, None);
        assert!(matches!(result, Err(EngineError::Verify(_))));
        let text = std::fs::read_to_string(v.report_path()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["passed"], false);
        assert_eq!(json["values"]["score"], 42);
        assert_eq!(json["checks"]["broken"]["failures"], 1);
        std::fs::remove_dir_all(&v.dir).ok();
    }
}
