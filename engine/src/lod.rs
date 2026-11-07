//! Per-instance level of detail with a dithered cross-fade.
//!
//! A LOD set is one placement list drawn with several meshes, from detailed
//! to coarse. Each instance picks its level from its distance to the camera
//! and fades into the next one over a band at the end of its range: both
//! levels draw it there, and a screen-space dither splits the pixels between
//! them, so there is no pop and no blending. Past the last level instances
//! fade out the same way.
//!
//! The fade travels in the instance tint's alpha (see [`fade_code`]); the
//! mesh shader decodes it. Levels are ordinary instanced entities whose
//! instance lists are rewritten when the camera has moved, without touching
//! the scene epoch (so GI and shadow sources aren't rebuilt for a LOD swap).

use crate::error::{EngineError, EngineResult};
use crate::mesh::Mesh;
use crate::place::{MeshInstance, Place};
use crate::world::EntityId;
use glam::Vec3;

/// One level of a [`LodSet`]: its mesh and the camera distance (metres) up
/// to which it draws. The last level's distance is where instances fade out
/// completely; use `f32::INFINITY` to keep them forever.
pub struct LodLevel {
    pub mesh: Mesh,
    pub max_distance: f32,
}

impl LodLevel {
    pub fn new(mesh: Mesh, max_distance: f32) -> Self {
        Self { mesh, max_distance }
    }
}

/// Handle to a LOD set spawned with [`crate::world::World::spawn_lod`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LodId(pub(crate) u64);

/// Tint alpha for an instance in a cross-fade at progress `t` (0 = only the
/// nearer level, 1 = only the farther one). The nearer (outgoing) level keeps
/// pixels whose dither value is at least `t`, the farther (incoming) one the
/// rest, so every pixel is drawn by exactly one of them. Normal alphas are
/// 0..=1, so negative codes never clash with real translucency.
pub(crate) fn fade_code(t: f32, incoming: bool) -> f32 {
    let t = t.clamp(0.0, 1.0);
    if incoming {
        -3.0 - t
    } else {
        -1.0 - t
    }
}

/// CPU twin of the shader's `lod_dither_keep`: whether a pixel with dither
/// value `h` (0..1) draws for an instance whose tint alpha is `code`.
#[cfg(test)]
pub(crate) fn dither_keep(code: f32, h: f32) -> bool {
    if code >= 0.0 {
        return true;
    }
    if code >= -2.0 {
        return h >= -code - 1.0;
    }
    h < -code - 3.0
}

/// Shared WGSL: decodes [`fade_code`] against an interleaved-gradient dither
/// of the pixel position.
pub(crate) const LOD_DITHER_WGSL: &str = r#"
fn lod_dither_keep(code: f32, frag: vec2<f32>, frame: f32) -> bool {
    if code >= 0.0 {
        return true;
    }
    // Interleaved gradient noise, stepped every frame (Jimenez 2014).
    let q = floor(frag) + vec2<f32>(5.588238 * frame);
    let h = fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
    if code >= -2.0 {
        return h >= -code - 1.0;
    }
    return h < -code - 3.0;
}
"#;

#[derive(Debug)]
pub(crate) struct LodSet {
    pub levels: Vec<(EntityId, f32)>,
    pub places: Vec<MeshInstance>,
    /// Camera position and bias the levels were last assigned for.
    pub assigned_for: Option<(Vec3, f32)>,
}

impl LodSet {
    pub fn validate_levels(levels: &[LodLevel]) -> EngineResult<()> {
        if levels.is_empty() {
            return Err(EngineError::InvalidValue(
                "a LOD set needs at least one level".into(),
            ));
        }
        let mut last = 0.0;
        for (i, level) in levels.iter().enumerate() {
            let d = level.max_distance;
            if d.is_nan() || d <= last {
                return Err(EngineError::InvalidValue(format!(
                    "LOD level {i} ends at {d} m; distances must be positive and increasing"
                )));
            }
            last = d;
        }
        Ok(())
    }

    /// Smallest camera move that can change an assignment noticeably.
    pub fn move_threshold(&self, bias: f32, fade: f32) -> f32 {
        let shortest = self
            .levels
            .iter()
            .map(|(_, d)| *d)
            .filter(|d| d.is_finite())
            .fold(f32::INFINITY, f32::min);
        if shortest.is_finite() {
            (shortest * bias * fade * 0.05).clamp(0.02, 0.5)
        } else {
            f32::INFINITY
        }
    }
}

/// Which level(s) draw each instance, as per-level instance lists.
///
/// `fade` is the width of the cross-fade band as a fraction of each level's
/// range end; `bias` scales every distance (above 1 keeps detail further out).
pub(crate) fn assign(
    places: &[MeshInstance],
    ends: &[f32],
    eye: Vec3,
    bias: f32,
    fade: f32,
) -> Vec<Vec<MeshInstance>> {
    let mut out: Vec<Vec<MeshInstance>> = vec![Vec::new(); ends.len()];
    let last = ends.len() - 1;
    for inst in places {
        let d = inst.transform.w_axis.truncate().distance(eye);
        let Some(i) = ends.iter().position(|e| d < e * bias) else {
            continue;
        };
        let end = ends[i] * bias;
        let start = if i == 0 { 0.0 } else { ends[i - 1] * bias };
        // The band never reaches back past the level's own start.
        let band = (end * fade).min((end - start) * 0.5);
        let band_start = end - band;
        if !end.is_finite() || d < band_start || band <= 0.0 {
            out[i].push(*inst);
            continue;
        }
        let t = (d - band_start) / band;
        let mut outgoing = *inst;
        outgoing.tint.a = fade_code(t, false);
        out[i].push(outgoing);
        if i < last {
            let mut incoming = *inst;
            incoming.tint.a = fade_code(t, true);
            out[i + 1].push(incoming);
        }
    }
    out
}

pub(crate) fn instances_from(places: &[Place]) -> Vec<MeshInstance> {
    places.iter().copied().map(Place::to_instance).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn at(x: f32) -> MeshInstance {
        MeshInstance::from_matrix(Mat4::from_translation(Vec3::new(x, 0.0, 0.0)))
    }

    #[test]
    fn levels_split_by_distance() {
        let places = [at(5.0), at(30.0), at(80.0), at(500.0)];
        let lists = assign(&places, &[20.0, 100.0], Vec3::ZERO, 1.0, 0.1);
        assert_eq!(lists[0].len(), 1);
        assert_eq!(lists[1].len(), 2);
        assert!(lists.iter().flatten().all(|i| i.tint.a == 1.0));
    }

    #[test]
    fn fade_band_draws_both_levels_with_complementary_pixels() {
        // Level 0 ends at 20 m with a 2 m band: 19 m is half way.
        let lists = assign(&[at(19.0)], &[20.0, 100.0], Vec3::ZERO, 1.0, 0.1);
        let (a, b) = (lists[0][0].tint.a, lists[1][0].tint.a);
        assert!(a < 0.0 && b < 0.0);
        for k in 0..100 {
            let h = k as f32 / 100.0;
            assert!(
                dither_keep(a, h) != dither_keep(b, h),
                "pixel {h} drawn once"
            );
        }
        let share = (0..100)
            .filter(|k| dither_keep(a, *k as f32 / 100.0))
            .count();
        assert_eq!(share, 50);
    }

    #[test]
    fn last_level_fades_out_and_bias_scales() {
        let lists = assign(&[at(97.0)], &[20.0, 100.0], Vec3::ZERO, 1.0, 0.1);
        assert!(lists[1][0].tint.a < 0.0, "fading out");
        let biased = assign(&[at(97.0)], &[20.0, 100.0], Vec3::ZERO, 2.0, 0.1);
        assert_eq!(biased[1][0].tint.a, 1.0, "bias 2 keeps it fully in");
        let biased = assign(&[at(30.0)], &[20.0, 100.0], Vec3::ZERO, 2.0, 0.1);
        assert_eq!((biased[0].len(), biased[1].len()), (1, 0));
    }

    #[test]
    fn rejects_bad_levels() {
        let lvl = |d| LodLevel::new(Mesh::new(), d);
        assert!(LodSet::validate_levels(&[]).is_err());
        assert!(LodSet::validate_levels(&[lvl(20.0), lvl(10.0)]).is_err());
        assert!(LodSet::validate_levels(&[lvl(20.0), lvl(f32::INFINITY)]).is_ok());
    }
}
