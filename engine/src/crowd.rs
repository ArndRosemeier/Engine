//! Crowds: hundreds or thousands of animated people for little cost.
//!
//! A crowd is one [`AnimatedModel`] and a list of agents. Far agents are
//! drawn instanced in one draw per mesh, posed on the GPU from the model's
//! idle and walk clips baked into a texture. The agents nearest the camera
//! (within [`CrowdProfile::near`]) become full animated characters, picked
//! up at exactly the pose and step they had, so nothing jumps when they
//! swap; walking away, they drop back into the crowd the same way.
//!
//! You move the agents (from your own simulation or navigation); the crowd
//! plays walk or idle to match each one's speed.
//!
//! ```no_run
//! # use engine::prelude::*;
//! # use std::sync::Arc;
//! # fn demo(world: &mut World, model: Arc<AnimatedModel>) -> EngineResult<()> {
//! let crowd = world.add_crowd(model, CrowdProfile::new("idle", "walk").walk_speed(1.3))?;
//! let agent = world.add_crowd_agent(crowd, Place::new(10.0, 0.0, 4.0))?;
//! // Every frame:
//! world.set_crowd_agent(crowd, agent, Place::new(10.5, 0.0, 4.0).with_yaw_deg(90.0), 1.3)?;
//! # Ok(()) }
//! ```

use crate::anim::{AnimatedModel, SkinMesh, MAX_JOINTS};
use crate::error::{EngineError, EngineResult};
use crate::mesh::MaterialMaps;
use crate::place::Place;
use crate::world::EntityId;
use glam::Vec3;
use std::sync::Arc;

/// Frames per second the clips are baked at (the GPU blends between them).
pub const CROWD_BAKE_FPS: f32 = 30.0;

/// How a crowd's agents are animated and when they become full characters.
#[derive(Clone, Debug)]
pub struct CrowdProfile {
    pub idle: String,
    pub walk: String,
    /// The speed the walk clip was authored at (plays faster or slower to
    /// match each agent, 0.5x..2x).
    pub walk_mps: f32,
    /// Agents closer than this become full characters.
    pub near_m: f32,
    /// At most this many full characters at once (the nearest win).
    pub max_near: usize,
    /// Agents beyond this are not drawn.
    pub far_m: f32,
    /// Agents within this distance cast shadows.
    pub shadow_m: f32,
    /// Seconds to cross-fade between idle and walk.
    pub blend_s: f32,
    /// Share of the triangles kept for far agents (1: the full mesh).
    pub far_detail: f32,
    /// A ready-made far model (Actors' "Crowd" level exported, say) with the
    /// same skeleton; replaces the automatic simplification.
    pub far_model: Option<Arc<AnimatedModel>>,
}

impl CrowdProfile {
    pub fn new(idle: impl Into<String>, walk: impl Into<String>) -> Self {
        Self {
            idle: idle.into(),
            walk: walk.into(),
            walk_mps: 1.4,
            near_m: 18.0,
            max_near: 24,
            far_m: 500.0,
            shadow_m: 60.0,
            blend_s: 0.2,
            far_detail: 0.05,
            far_model: None,
        }
    }

    /// Keep this share of the triangles far away (0.01..=1, 0.05 default).
    pub fn far_detail(mut self, fraction: f32) -> Self {
        self.far_detail = fraction;
        self
    }

    /// Draw far agents with this model (same skeleton) instead of a
    /// simplified copy.
    pub fn far_model(mut self, model: Arc<AnimatedModel>) -> Self {
        self.far_model = Some(model);
        self
    }

    pub fn walk_speed(mut self, mps: f32) -> Self {
        self.walk_mps = mps;
        self
    }

    /// Full characters within `metres`, at most `max` of them (0: never).
    pub fn near(mut self, metres: f32, max: usize) -> Self {
        self.near_m = metres;
        self.max_near = max;
        self
    }

    pub fn far(mut self, metres: f32) -> Self {
        self.far_m = metres;
        self
    }

    pub fn shadows_within(mut self, metres: f32) -> Self {
        self.shadow_m = metres;
        self
    }

    pub fn validate(&self, model: &AnimatedModel) -> EngineResult<[usize; 2]> {
        let bad = |m: String| Err(EngineError::InvalidValue(m));
        let ok = |v: f32| v.is_finite() && v >= 0.0;
        if !ok(self.near_m) || !ok(self.far_m) || !ok(self.shadow_m) || !ok(self.blend_s) {
            return bad("crowd distances and blend must be finite and >= 0".into());
        }
        if !self.walk_mps.is_finite() || self.walk_mps <= 0.0 {
            return bad("crowd walk speed must be > 0".into());
        }
        if !self.far_detail.is_finite() || !(0.01..=1.0).contains(&self.far_detail) {
            return bad("crowd far detail must be within 0.01..=1".into());
        }
        if let Some(far) = &self.far_model {
            if far.skeleton.joint_names != model.skeleton.joint_names {
                return bad("crowd far model needs the same skeleton as the model".into());
            }
        }
        if model.skeleton.joint_names.len() > MAX_JOINTS {
            return bad(format!("crowd model has more than {MAX_JOINTS} joints"));
        }
        let find = |name: &str| model.clips.iter().position(|c| c.name == name);
        match (find(&self.idle), find(&self.walk)) {
            (Some(i), Some(w)) => Ok([i, w]),
            _ => bad(format!(
                "crowd model needs clips {:?} and {:?} (it has {:?})",
                self.idle,
                self.walk,
                model.clip_names().collect::<Vec<_>>()
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CrowdId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CrowdAgentId(pub u32);

/// Which of the crowd's clips: 0 idle, 1 walk.
pub(crate) const IDLE: usize = 0;
pub(crate) const WALK: usize = 1;

/// The clip being faded out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AgentFade {
    pub clip: usize,
    pub time: f32,
    pub rate: f32,
    pub elapsed: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct CrowdAgent {
    pub place: Place,
    pub speed: f32,
    pub clip: usize,
    pub time: f32,
    pub fade: Option<AgentFade>,
    /// The full character standing in while near.
    pub near: Option<EntityId>,
}

impl CrowdAgent {
    pub fn rate(&self, walk_mps: f32) -> f32 {
        if self.clip == WALK {
            (self.speed / walk_mps).clamp(0.5, 2.0)
        } else {
            1.0
        }
    }
}

#[derive(Debug)]
pub(crate) struct Crowd {
    pub model: Arc<AnimatedModel>,
    pub profile: CrowdProfile,
    /// Model clip index for idle and walk.
    pub clips: [usize; 2],
    /// The simplified (or given) model drawn far away.
    pub far: Option<Arc<AnimatedModel>>,
    /// For each mesh of a simplified `far` model, the mesh of `model` whose
    /// maps it is drawn with (empty: a given far model carries its own).
    pub far_source: Vec<usize>,
    pub agents: Vec<Option<CrowdAgent>>,
}

impl Crowd {
    pub fn duration(&self, clip: usize) -> f32 {
        self.model.clips[self.clips[clip]].duration.max(1e-3)
    }

    /// Advance a far agent's clips by `dt`.
    #[cfg(test)]
    pub fn advance(&self, a: &mut CrowdAgent, dt: f32) {
        advance(&self.profile, &self.model, self.clips, a, dt);
    }
}

/// Advance a far agent: walk or idle by its speed, cross-fading on a change.
pub(crate) fn advance(
    p: &CrowdProfile,
    model: &AnimatedModel,
    clips: [usize; 2],
    a: &mut CrowdAgent,
    dt: f32,
) {
    let duration = |clip: usize| model.clips[clips[clip]].duration.max(1e-3);
    let want = if a.speed > 0.05 { WALK } else { IDLE };
    if want != a.clip {
        a.fade = (p.blend_s > 0.0).then_some(AgentFade {
            clip: a.clip,
            time: a.time,
            rate: a.rate(p.walk_mps),
            elapsed: 0.0,
        });
        a.clip = want;
        // Start the new clip from the top, like a character's animator.
        a.time = 0.0;
    }
    if let Some(f) = a.fade.as_mut() {
        f.elapsed += dt;
        f.time = (f.time + dt * f.rate).rem_euclid(duration(f.clip));
        if f.elapsed >= p.blend_s {
            a.fade = None;
        }
    }
    a.time = (a.time + dt * a.rate(p.walk_mps)).rem_euclid(duration(a.clip));
}

/// One far agent as the GPU draws it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CrowdInstance {
    pub place: Place,
    /// (clip, time) now and the clip fading out with its weight.
    pub clip: usize,
    pub time: f32,
    pub fade: Option<(usize, f32, f32)>,
}

impl CrowdInstance {
    pub fn of(c: &Crowd, a: &CrowdAgent) -> Self {
        let fade = a.fade.map(|f| {
            let t = (f.elapsed / c.profile.blend_s.max(1e-6)).clamp(0.0, 1.0);
            // The animator's smoothstep: the old clip's weight.
            (f.clip, f.time, 1.0 - t * t * (3.0 - 2.0 * t))
        });
        Self {
            place: a.place,
            clip: a.clip,
            time: a.time,
            fade,
        }
    }
}

/// A start phase for agent `i`, so a crowd does not step in lockstep.
pub(crate) fn spread(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    (x & 0xFFFF) as f32 / 65_536.0
}

/// `mesh` with about `fraction` of its triangles (at least a few), for
/// drawing far away: meshoptimizer's edge collapse, then unused vertices
/// dropped. Tiny parts (eyes, lashes, teeth) vanish at crowd distance, so
/// meshes under `min_size_m` across are left out (None).
pub(crate) fn simplify_skin_mesh(
    mesh: &SkinMesh,
    fraction: f32,
    min_size_m: f32,
) -> Option<SkinMesh> {
    let (lo, hi) = mesh.positions.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    if mesh.indices.len() < 3 || (hi - lo).max_element() < min_size_m {
        return None;
    }
    let target = ((mesh.indices.len() / 3) as f32 * fraction).max(16.0) as usize * 3;
    let bytes: &[u8] = bytemuck::cast_slice(&mesh.positions);
    let adapter = meshopt::VertexDataAdapter::new(bytes, 12, 0).ok()?;
    let mut indices = meshopt::simplify(
        &mesh.indices,
        &adapter,
        target,
        0.012,
        meshopt::SimplifyOptions::None,
        None,
    );
    if indices.len() > target * 2 {
        // Garments with many open edges resist; the sloppy pass doesn't.
        indices = meshopt::simplify_sloppy(&indices, &adapter, target, 0.012, None);
    }
    if indices.is_empty() {
        return None;
    }
    // Old vertex -> new, in first-use order (u32::MAX: unused).
    let mut remap = vec![u32::MAX; mesh.positions.len()];
    let mut order: Vec<usize> = Vec::new();
    for &i in &indices {
        if remap[i as usize] == u32::MAX {
            remap[i as usize] = order.len() as u32;
            order.push(i as usize);
        }
    }
    fn take<T: Copy>(order: &[usize], v: &[T]) -> Vec<T> {
        order.iter().map(|&i| v[i]).collect()
    }
    Some(SkinMesh {
        positions: take(&order, &mesh.positions),
        normals: take(&order, &mesh.normals),
        colors: take(&order, &mesh.colors),
        uvs: if mesh.uvs.len() == mesh.positions.len() {
            take(&order, &mesh.uvs)
        } else {
            Vec::new()
        },
        joints: take(&order, &mesh.joints),
        weights: take(&order, &mesh.weights),
        indices: indices.iter().map(|i| remap[*i as usize]).collect(),
        double_sided: mesh.double_sided,
        // No copy of the maps: the far mesh is drawn with its source mesh's
        // (see `Crowd::far_source`), on the GPU too.
        albedo: None,
        maps: MaterialMaps {
            normal: None,
            orm: None,
            emissive: None,
            normal_scale: mesh.maps.normal_scale,
            normal_uv: mesh.maps.normal_uv,
            occlusion_strength: mesh.maps.occlusion_strength,
            emissive_factor: mesh.maps.emissive_factor,
            roughness: mesh.maps.roughness,
            metallic: mesh.maps.metallic,
            triplanar: mesh.maps.triplanar,
        },
        // Too far to see a face move.
        morphs: Vec::new(),
    })
}

/// A copy of `model` simplified for crowd distance (see
/// [`simplify_skin_mesh`]), and for each of its meshes the index of the
/// mesh of `model` it was made from (whose maps it is drawn with).
pub(crate) fn far_model(model: &AnimatedModel, fraction: f32) -> (AnimatedModel, Vec<usize>) {
    // Parts smaller than 6 cm (eyes, teeth, lashes) can't be seen from afar.
    let (meshes, source): (Vec<SkinMesh>, Vec<usize>) = model
        .meshes
        .iter()
        .enumerate()
        .filter_map(|(i, m)| simplify_skin_mesh(m, fraction, 0.06).map(|f| (f, i)))
        .unzip();
    let far = AnimatedModel {
        skeleton: model.skeleton.clone(),
        meshes,
        clips: Vec::new(),
        bind_local: model.bind_local.clone(),
        morph_targets: Vec::new(),
    };
    (far, source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn character() -> AnimatedModel {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/model_viewer/assets/character.glb"
        );
        let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
        AnimatedModel::load_with(path, examples, &crate::limits::EngineLimits::default()).unwrap()
    }

    #[test]
    fn far_model_is_a_small_fraction_of_the_character() {
        let model = character();
        let verts = |m: &AnimatedModel| m.meshes.iter().map(|m| m.positions.len()).sum::<usize>();
        let tris = |m: &AnimatedModel| m.meshes.iter().map(|m| m.indices.len() / 3).sum::<usize>();
        let (far, source) = far_model(&model, 0.05);
        println!(
            "full {} verts {} tris {} meshes; far {} verts {} tris {} meshes",
            verts(&model),
            tris(&model),
            model.meshes.len(),
            verts(&far),
            tris(&far),
            far.meshes.len()
        );
        assert!(verts(&far) * 8 < verts(&model));
        assert!(far.meshes.len() >= 3, "body, clothes and hair stay");
        for m in &far.meshes {
            assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
        }
        // The far meshes hold no maps of their own: each is drawn with the
        // maps of the mesh it was made from.
        assert_eq!(source.len(), far.meshes.len());
        for (m, &s) in far.meshes.iter().zip(&source) {
            assert!(m.albedo.is_none() && m.maps.normal.is_none() && m.maps.orm.is_none());
            assert!(s < model.meshes.len());
        }
        assert!(
            source.iter().any(|&s| model.meshes[s].albedo.is_some()),
            "the test character is textured"
        );
    }

    #[test]
    fn far_agents_switch_clips_with_a_fade() {
        let model = Arc::new(character());
        let profile = CrowdProfile::new("idle", "walk");
        let clips = profile.validate(&model).unwrap();
        let crowd = Crowd {
            model,
            profile,
            clips,
            far: None,
            far_source: Vec::new(),
            agents: Vec::new(),
        };
        let mut a = CrowdAgent {
            place: Place::default(),
            speed: 0.0,
            clip: IDLE,
            time: 0.3,
            fade: None,
            near: None,
        };
        crowd.advance(&mut a, 0.1);
        assert_eq!(a.clip, IDLE);
        a.speed = 1.4;
        crowd.advance(&mut a, 0.05);
        assert_eq!(a.clip, WALK);
        let i = CrowdInstance::of(&crowd, &a);
        let (_, _, w) = i.fade.unwrap();
        assert!(w > 0.5 && w < 1.0, "old clip still mostly there: {w}");
        crowd.advance(&mut a, 0.3);
        assert!(a.fade.is_none());
    }
}
