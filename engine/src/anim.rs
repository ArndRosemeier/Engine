//! Skinned glTF models and clip playback.

use crate::error::{EngineError, EngineResult};
use crate::limits::EngineLimits;
use crate::mesh::AlbedoMap;
use crate::model::{image_to_albedo, reject_alpha_mode_without_real_alpha};
use glam::{Mat4, Quat, Vec3, Vec4};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Maximum joints uploaded to the GPU joint palette.
pub const MAX_JOINTS: usize = 128;

#[derive(Clone, Debug)]
pub struct Skeleton {
    pub joint_names: Vec<String>,
    /// Parent joint index, or `None` for roots.
    pub parents: Vec<Option<usize>>,
    pub inverse_bind: Vec<Mat4>,
    /// glTF node index for each joint (for animation channel targeting).
    pub joint_node_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct SkinMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub colors: Vec<Vec4>,
    pub uvs: Vec<[f32; 2]>,
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    /// From glTF material `doubleSided`.
    pub double_sided: bool,
    /// Per-primitive `baseColorTexture`, if the glTF has one.
    pub albedo: Option<AlbedoMap>,
}

#[derive(Clone, Debug)]
pub struct Vec3Track {
    pub times: Vec<f32>,
    pub values: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct QuatTrack {
    pub times: Vec<f32>,
    pub values: Vec<Quat>,
}

#[derive(Clone, Debug, Default)]
pub struct JointChannels {
    pub translation: Option<Vec3Track>,
    pub rotation: Option<QuatTrack>,
    pub scale: Option<Vec3Track>,
}

#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    /// Per skeleton joint index.
    pub channels: Vec<JointChannels>,
}

#[derive(Clone, Debug)]
pub struct AnimatedModel {
    pub skeleton: Skeleton,
    pub meshes: Vec<SkinMesh>,
    pub clips: Vec<AnimationClip>,
    /// Bind-pose local TRS per skeleton joint (from the glTF rest pose).
    pub bind_local: Vec<(Vec3, Quat, Vec3)>,
}

impl AnimatedModel {
    pub fn load(path: impl AsRef<Path>) -> EngineResult<Self> {
        Self::load_with(path, PathBuf::from("."), &EngineLimits::default())
    }

    pub fn load_with(
        path: impl AsRef<Path>,
        base_dir: impl AsRef<Path>,
        limits: &EngineLimits,
    ) -> EngineResult<Self> {
        let path = resolve_allowed_path(path.as_ref(), base_dir.as_ref())?;
        let meta = std::fs::metadata(&path)?;
        if meta.len() > limits.max_gltf_buffer_bytes {
            return Err(EngineError::ResourceLimit(format!(
                "glTF file is {} bytes (limit {})",
                meta.len(),
                limits.max_gltf_buffer_bytes
            )));
        }

        let (document, buffers, images) =
            gltf::import(&path).map_err(|e| EngineError::Model(e.to_string()))?;

        let total_buf: u64 = buffers.iter().map(|b| b.0.len() as u64).sum();
        if total_buf > limits.max_gltf_buffer_bytes {
            return Err(EngineError::ResourceLimit(format!(
                "glTF buffers are {total_buf} bytes (limit {})",
                limits.max_gltf_buffer_bytes
            )));
        }

        load_animated_document(&document, &buffers, &images, limits)
    }

    pub fn clip_names(&self) -> impl Iterator<Item = &str> {
        self.clips.iter().map(|c| c.name.as_str())
    }

    pub fn find_clip(&self, name: &str) -> Option<&AnimationClip> {
        self.clips.iter().find(|c| c.name == name)
    }
}

/// Persistent movement intent reported by a consumer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Locomotion {
    Idle,
    Moving { speed_mps: f32 },
}

/// Semantic one-shot action clips. The model profile maps these to authored names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationAction {
    Attack,
    Cast,
    Hit,
    Death,
}

/// Validated semantic animation mapping for one model.
#[derive(Clone, Debug, Default)]
pub struct AnimationProfile {
    idle: Option<String>,
    walk: Option<String>,
    run: Option<String>,
    attack: Option<String>,
    cast: Option<String>,
    hit: Option<String>,
    death: Option<String>,
}

impl AnimationProfile {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn idle(mut self, clip: impl Into<String>) -> Self {
        self.idle = Some(clip.into());
        self
    }
    pub fn walk(mut self, clip: impl Into<String>) -> Self {
        self.walk = Some(clip.into());
        self
    }
    pub fn run(mut self, clip: impl Into<String>) -> Self {
        self.run = Some(clip.into());
        self
    }
    pub fn attack(mut self, clip: impl Into<String>) -> Self {
        self.attack = Some(clip.into());
        self
    }
    pub fn cast(mut self, clip: impl Into<String>) -> Self {
        self.cast = Some(clip.into());
        self
    }
    pub fn hit(mut self, clip: impl Into<String>) -> Self {
        self.hit = Some(clip.into());
        self
    }
    pub fn death(mut self, clip: impl Into<String>) -> Self {
        self.death = Some(clip.into());
        self
    }
}

/// Playback state for one [`AnimatedModel`]. Consumers use semantic transitions;
/// clip indices, clocks, looping, and action resumption remain engine-owned.
#[derive(Clone, Debug)]
pub struct Animator {
    model: Arc<AnimatedModel>,
    clip_index: usize,
    time: f32,
    speed: f32,
    looping: bool,
    profile: Option<AnimationProfile>,
    locomotion: Locomotion,
    action_resume: Option<Locomotion>,
}

impl Animator {
    pub fn new(model: Arc<AnimatedModel>) -> EngineResult<Self> {
        Ok(Self {
            model,
            clip_index: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
            profile: None,
            locomotion: Locomotion::Idle,
            action_resume: None,
        })
    }

    pub fn play(&mut self, clip_name: &str) -> EngineResult<()> {
        self.play_clip(clip_name, true)
    }

    pub fn configure_profile(&mut self, profile: AnimationProfile) -> EngineResult<()> {
        let idle = profile
            .idle
            .as_deref()
            .ok_or_else(|| EngineError::Model("animation profile requires idle clip".into()))?;
        self.require_clip(idle)?;
        for clip in [
            &profile.walk,
            &profile.run,
            &profile.attack,
            &profile.cast,
            &profile.hit,
            &profile.death,
        ] {
            if let Some(name) = clip.as_deref() {
                self.require_clip(name)?;
            }
        }
        self.profile = Some(profile);
        self.set_locomotion(Locomotion::Idle)
    }

    pub fn set_locomotion(&mut self, state: Locomotion) -> EngineResult<()> {
        if let Locomotion::Moving { speed_mps } = state {
            if !speed_mps.is_finite() || speed_mps < 0.0 {
                return Err(EngineError::InvalidValue(format!(
                    "locomotion speed must be finite and non-negative, got {speed_mps}"
                )));
            }
        }
        self.locomotion = state;
        if !self.looping {
            // Preserve the latest intent while a one-shot action finishes.
            // Death has no resume state, so it remains terminal.
            if self.action_resume.is_some() {
                self.action_resume = Some(state);
            }
            return Ok(());
        }
        let clip = {
            let profile = self
                .profile
                .as_ref()
                .ok_or_else(|| EngineError::Model("animation profile is not configured".into()))?;
            match state {
                Locomotion::Idle => profile
                    .idle
                    .as_deref()
                    .expect("validated idle clip")
                    .to_owned(),
                Locomotion::Moving { speed_mps } if speed_mps.is_finite() && speed_mps >= 1.5 => {
                    profile
                        .run
                        .as_deref()
                        .or(profile.walk.as_deref())
                        .expect("validated locomotion clip")
                        .to_owned()
                }
                Locomotion::Moving { speed_mps } if speed_mps.is_finite() && speed_mps >= 0.0 => {
                    profile
                        .walk
                        .as_deref()
                        .or(profile.run.as_deref())
                        .expect("validated locomotion clip")
                        .to_owned()
                }
                Locomotion::Moving { speed_mps } => {
                    return Err(EngineError::InvalidValue(format!(
                        "locomotion speed must be finite and non-negative, got {speed_mps}"
                    )))
                }
            }
        };
        self.action_resume = None;
        if self.clip_name() == clip {
            return Ok(());
        }
        self.play_clip(&clip, true)
    }

    pub fn play_action(&mut self, action: AnimationAction) -> EngineResult<()> {
        let profile = self
            .profile
            .as_ref()
            .ok_or_else(|| EngineError::Model("animation profile is not configured".into()))?;
        let clip = match action {
            AnimationAction::Attack => profile.attack.as_deref(),
            AnimationAction::Cast => profile.cast.as_deref(),
            AnimationAction::Hit => profile.hit.as_deref(),
            AnimationAction::Death => profile.death.as_deref(),
        }
        .ok_or_else(|| {
            EngineError::Model(format!("animation action {action:?} is not configured"))
        })?
        .to_owned();
        self.action_resume = (action != AnimationAction::Death).then_some(self.locomotion);
        self.play_clip(&clip, false)
    }

    pub(crate) fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    pub(crate) fn model(&self) -> &Arc<AnimatedModel> {
        &self.model
    }
    pub(crate) fn clip_index(&self) -> usize {
        self.clip_index
    }
    pub fn time(&self) -> f32 {
        self.time
    }

    fn require_clip(&self, name: &str) -> EngineResult<()> {
        if self.model.find_clip(name).is_some() {
            Ok(())
        } else {
            Err(EngineError::Model(format!(
                "animation profile references unknown clip '{name}'"
            )))
        }
    }

    /// Play a clip once and hold the last frame when it ends.
    pub fn play_once(&mut self, clip_name: &str) -> EngineResult<()> {
        self.play_clip(clip_name, false)
    }

    fn play_clip(&mut self, clip_name: &str, looping: bool) -> EngineResult<()> {
        let idx = self
            .model
            .clips
            .iter()
            .position(|c| c.name == clip_name)
            .ok_or_else(|| EngineError::Model(format!("unknown animation clip '{clip_name}'")))?;
        self.clip_index = idx;
        self.time = 0.0;
        self.looping = looping;
        Ok(())
    }

    pub fn is_looping(&self) -> bool {
        self.looping
    }

    pub fn clip_duration(&self) -> Option<f32> {
        self.model
            .clips
            .get(self.clip_index)
            .map(|clip| clip.duration)
    }

    pub fn clip_name(&self) -> &str {
        self.model
            .clips
            .get(self.clip_index)
            .map(|c| c.name.as_str())
            .unwrap_or("")
    }

    pub fn tick(&mut self, dt: f32) {
        let was_action = !self.looping;
        let Some(clip) = self.model.clips.get(self.clip_index) else {
            return;
        };
        if clip.duration <= 0.0 {
            return;
        }
        self.time += dt * self.speed;
        if self.looping {
            self.time = self.time.rem_euclid(clip.duration);
        } else {
            self.time = self.time.clamp(0.0, clip.duration);
        }
        if was_action && !self.looping && self.time >= clip.duration {
            if let Some(state) = self.action_resume.take() {
                self.looping = true;
                self.set_locomotion(state)
                    .expect("validated animation resume state");
            }
        }
    }

    /// Skinning matrices: `global * inverse_bind` per joint.
    pub fn joint_matrices(&self) -> Vec<Mat4> {
        let mut out = Vec::new();
        let mut locals = Vec::new();
        let mut global = Vec::new();
        write_joint_matrices(
            &self.model,
            self.clip_index,
            self.time,
            &mut locals,
            &mut global,
            &mut out,
        );
        out
    }
}

pub fn sample_joint_matrices(model: &AnimatedModel, clip_index: usize, time: f32) -> Vec<Mat4> {
    let mut out = Vec::new();
    let mut locals = Vec::new();
    let mut global = Vec::new();
    write_joint_matrices(model, clip_index, time, &mut locals, &mut global, &mut out);
    out
}

/// Fill `out` without allocating when the scratches already have capacity.
pub fn write_joint_matrices(
    model: &AnimatedModel,
    clip_index: usize,
    time: f32,
    locals: &mut Vec<(Vec3, Quat, Vec3)>,
    global: &mut Vec<Mat4>,
    out: &mut Vec<Mat4>,
) {
    let skeleton = &model.skeleton;
    let n = skeleton.joint_names.len();
    locals.clear();
    locals.extend_from_slice(&model.bind_local);

    if let Some(clip) = model.clips.get(clip_index) {
        for (ji, channels) in clip.channels.iter().enumerate() {
            if ji >= n {
                break;
            }
            let (t, r, s) = &mut locals[ji];
            if let Some(track) = &channels.translation {
                *t = sample_vec3(track, time);
            }
            if let Some(track) = &channels.rotation {
                *r = sample_quat(track, time);
            }
            if let Some(track) = &channels.scale {
                *s = sample_vec3(track, time);
            }
        }
    }

    global.clear();
    global.resize(n, Mat4::IDENTITY);
    for i in 0..n {
        let (t, r, s) = locals[i];
        let local = Mat4::from_scale_rotation_translation(s, r, t);
        global[i] = match skeleton.parents[i] {
            Some(p) => global[p] * local,
            None => local,
        };
    }

    out.clear();
    out.extend((0..n).map(|i| global[i] * skeleton.inverse_bind[i]));
}

fn sample_vec3(track: &Vec3Track, time: f32) -> Vec3 {
    if track.times.is_empty() {
        return Vec3::ZERO;
    }
    if track.times.len() == 1 || time <= track.times[0] {
        return track.values[0];
    }
    let last = track.times.len() - 1;
    if time >= track.times[last] {
        return track.values[last];
    }
    let mut i = 0;
    while i + 1 < track.times.len() && track.times[i + 1] < time {
        i += 1;
    }
    let t0 = track.times[i];
    let t1 = track.times[i + 1];
    let u = if t1 > t0 {
        ((time - t0) / (t1 - t0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    track.values[i].lerp(track.values[i + 1], u)
}

fn sample_quat(track: &QuatTrack, time: f32) -> Quat {
    if track.times.is_empty() {
        return Quat::IDENTITY;
    }
    if track.times.len() == 1 || time <= track.times[0] {
        return track.values[0].normalize();
    }
    let last = track.times.len() - 1;
    if time >= track.times[last] {
        return track.values[last].normalize();
    }
    let mut i = 0;
    while i + 1 < track.times.len() && track.times[i + 1] < time {
        i += 1;
    }
    let t0 = track.times[i];
    let t1 = track.times[i + 1];
    let u = if t1 > t0 {
        ((time - t0) / (t1 - t0)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    track.values[i].slerp(track.values[i + 1], u).normalize()
}

fn resolve_allowed_path(path: &Path, base_dir: &Path) -> EngineResult<PathBuf> {
    let base = std::fs::canonicalize(base_dir).map_err(|error| {
        EngineError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "failed to canonicalize animation sandbox root {}: {error}",
                base_dir.display()
            ),
        ))
    })?;
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    let canonical = std::fs::canonicalize(&candidate).map_err(|e| {
        EngineError::Io(std::io::Error::new(
            e.kind(),
            format!("{} ({})", e, candidate.display()),
        ))
    })?;
    if !canonical.starts_with(&base) {
        return Err(EngineError::PathNotAllowed(format!(
            "{} is outside allowed root {}",
            canonical.display(),
            base.display()
        )));
    }
    Ok(canonical)
}

fn nearest_joint(
    mut node: usize,
    node_parent: &HashMap<usize, usize>,
    node_to_joint: &HashMap<usize, usize>,
) -> Option<usize> {
    loop {
        if let Some(&ji) = node_to_joint.get(&node) {
            return Some(ji);
        }
        node = *node_parent.get(&node)?;
    }
}

fn node_global_matrix(
    document: &gltf::Document,
    node_index: usize,
    node_parent: &HashMap<usize, usize>,
) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur = Some(node_index);
    while let Some(i) = cur {
        chain.push(i);
        cur = node_parent.get(&i).copied();
    }
    chain.reverse();
    let mut m = Mat4::IDENTITY;
    for i in chain {
        let node = document
            .nodes()
            .nth(i)
            .expect("node index from the same glTF parent map");
        let (t, r, s) = node_local_trs(&node);
        m *= Mat4::from_scale_rotation_translation(s, r, t);
    }
    m
}

fn node_local_trs(node: &gltf::Node<'_>) -> (Vec3, Quat, Vec3) {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => {
            let m = Mat4::from_cols_array_2d(&matrix);
            let (s, r, t) = m.to_scale_rotation_translation();
            (t, r, s)
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => (
            Vec3::from(translation),
            Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
            Vec3::from(scale),
        ),
    }
}

fn load_animated_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    limits: &EngineLimits,
) -> EngineResult<AnimatedModel> {
    let skin = document
        .skins()
        .next()
        .ok_or_else(|| EngineError::Model("glTF has no skins (not a skinned model)".into()))?;

    let joint_nodes: Vec<gltf::Node<'_>> = skin.joints().collect();
    if joint_nodes.is_empty() {
        return Err(EngineError::Model("skin has no joints".into()));
    }
    if joint_nodes.len() > limits.max_joints as usize {
        return Err(EngineError::ResourceLimit(format!(
            "skin has {} joints (limit {})",
            joint_nodes.len(),
            limits.max_joints
        )));
    }
    if joint_nodes.len() > MAX_JOINTS {
        return Err(EngineError::ResourceLimit(format!(
            "skin has {} joints (GPU palette max {})",
            joint_nodes.len(),
            MAX_JOINTS
        )));
    }

    let joint_node_indices: Vec<usize> = joint_nodes.iter().map(|n| n.index()).collect();
    let node_to_joint: HashMap<usize, usize> = joint_node_indices
        .iter()
        .enumerate()
        .map(|(ji, &ni)| (ni, ji))
        .collect();

    let mut node_parent: HashMap<usize, usize> = HashMap::new();
    for node in document.nodes() {
        for child in node.children() {
            node_parent.insert(child.index(), node.index());
        }
    }
    let parents: Vec<Option<usize>> = joint_node_indices
        .iter()
        .map(|&ni| {
            let mut cur = node_parent.get(&ni).copied();
            while let Some(p) = cur {
                if let Some(&ji) = node_to_joint.get(&p) {
                    return Some(ji);
                }
                cur = node_parent.get(&p).copied();
            }
            None
        })
        .collect();

    let reader = skin.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
    let inverse_bind: Vec<Mat4> = if let Some(iter) = reader.read_inverse_bind_matrices() {
        iter.map(|m| Mat4::from_cols_array_2d(&m)).collect()
    } else {
        vec![Mat4::IDENTITY; joint_nodes.len()]
    };
    if inverse_bind.len() != joint_nodes.len() {
        return Err(EngineError::Model(format!(
            "inverse bind count {} != joint count {}",
            inverse_bind.len(),
            joint_nodes.len()
        )));
    }

    let joint_names: Vec<String> = joint_nodes
        .iter()
        .map(|n| n.name().unwrap_or("joint").to_string())
        .collect();
    let bind_local: Vec<(Vec3, Quat, Vec3)> =
        joint_nodes.iter().map(|n| node_local_trs(n)).collect();

    let skeleton = Skeleton {
        joint_names,
        parents,
        inverse_bind,
        joint_node_indices: joint_node_indices.clone(),
    };

    let mut meshes = Vec::new();
    let mut total_tris = 0u64;

    for node in document.nodes() {
        let Some(mesh) = node.mesh() else {
            continue;
        };

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
            let mut positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| EngineError::Model("skinned mesh missing POSITION".into()))?
                .collect();
            if positions.is_empty() {
                continue;
            }

            let (joints, extra_weights) = match reader.read_joints(0) {
                Some(joints_iter) => {
                    let joints: Vec<[u16; 4]> = match joints_iter {
                        gltf::mesh::util::ReadJoints::U8(i) => i
                            .map(|j| [j[0] as u16, j[1] as u16, j[2] as u16, j[3] as u16])
                            .collect(),
                        gltf::mesh::util::ReadJoints::U16(i) => i.collect(),
                    };
                    (joints, None)
                }
                None => {
                    let joint_i = nearest_joint(node.index(), &node_parent, &node_to_joint)
                        .ok_or_else(|| {
                            EngineError::Model(format!(
                                "mesh '{}' has no JOINTS_0 and is not parented to a joint",
                                node.name().unwrap_or("unnamed")
                            ))
                        })?;
                    let n = positions.len();
                    (
                        vec![[joint_i as u16, 0, 0, 0]; n],
                        Some(vec![[1.0, 0.0, 0.0, 0.0]; n]),
                    )
                }
            };
            let bake_to_joint = extra_weights.is_some();
            let mut weights: Vec<[f32; 4]> = if let Some(w) = extra_weights {
                let xform = node_global_matrix(document, node.index(), &node_parent);
                for p in &mut positions {
                    *p = xform.transform_point3(Vec3::from(*p)).to_array();
                }
                w
            } else {
                let weights_iter = reader
                    .read_weights(0)
                    .ok_or_else(|| EngineError::Model("skinned mesh missing WEIGHTS_0".into()))?;
                match weights_iter {
                    gltf::mesh::util::ReadWeights::U8(i) => i
                        .map(|w| {
                            [
                                w[0] as f32 / 255.0,
                                w[1] as f32 / 255.0,
                                w[2] as f32 / 255.0,
                                w[3] as f32 / 255.0,
                            ]
                        })
                        .collect(),
                    gltf::mesh::util::ReadWeights::U16(i) => i
                        .map(|w| {
                            [
                                w[0] as f32 / 65535.0,
                                w[1] as f32 / 65535.0,
                                w[2] as f32 / 65535.0,
                                w[3] as f32 / 65535.0,
                            ]
                        })
                        .collect(),
                    gltf::mesh::util::ReadWeights::F32(i) => i.collect(),
                }
            };

            if joints.len() != positions.len() || weights.len() != positions.len() {
                return Err(EngineError::Model(
                    "JOINTS/WEIGHTS length mismatch with POSITION".into(),
                ));
            }
            for w in &mut weights {
                let sum = w[0] + w[1] + w[2] + w[3];
                if sum > 1e-6 {
                    w[0] /= sum;
                    w[1] /= sum;
                    w[2] /= sum;
                    w[3] /= sum;
                } else {
                    *w = [1.0, 0.0, 0.0, 0.0];
                }
            }

            let mut normals: Vec<[f32; 3]> = if let Some(iter) = reader.read_normals() {
                iter.collect()
            } else {
                vec![[0.0, 1.0, 0.0]; positions.len()]
            };
            if bake_to_joint {
                let xform = node_global_matrix(document, node.index(), &node_parent);
                for n in &mut normals {
                    *n = xform
                        .transform_vector3(Vec3::from(*n))
                        .normalize_or_zero()
                        .to_array();
                }
            }

            // glTF: final color = COLOR_0 * baseColorFactor (* texture if present).
            // Quaternius animals ship white COLOR_0 and put the look in material factors.
            let base_color = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_factor();
            let mut colors: Vec<[f32; 4]> = if let Some(iter) = reader.read_colors(0) {
                match iter {
                    gltf::mesh::util::ReadColors::RgbU8(i) => i
                        .map(|c| {
                            [
                                c[0] as f32 / 255.0,
                                c[1] as f32 / 255.0,
                                c[2] as f32 / 255.0,
                                1.0,
                            ]
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaU8(i) => i
                        .map(|c| {
                            [
                                c[0] as f32 / 255.0,
                                c[1] as f32 / 255.0,
                                c[2] as f32 / 255.0,
                                c[3] as f32 / 255.0,
                            ]
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbU16(i) => i
                        .map(|c| {
                            [
                                c[0] as f32 / 65535.0,
                                c[1] as f32 / 65535.0,
                                c[2] as f32 / 65535.0,
                                1.0,
                            ]
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbaU16(i) => i
                        .map(|c| {
                            [
                                c[0] as f32 / 65535.0,
                                c[1] as f32 / 65535.0,
                                c[2] as f32 / 65535.0,
                                c[3] as f32 / 65535.0,
                            ]
                        })
                        .collect(),
                    gltf::mesh::util::ReadColors::RgbF32(i) => {
                        i.map(|c| [c[0], c[1], c[2], 1.0]).collect()
                    }
                    gltf::mesh::util::ReadColors::RgbaF32(i) => i.collect(),
                }
            } else {
                vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
            };
            for c in &mut colors {
                c[0] *= base_color[0];
                c[1] *= base_color[1];
                c[2] *= base_color[2];
                c[3] *= base_color[3];
            }

            let indices: Vec<u32> = if let Some(iter) = reader.read_indices() {
                iter.into_u32().collect()
            } else {
                (0..positions.len() as u32).collect()
            };
            total_tris += (indices.len() / 3) as u64;
            if total_tris > limits.max_model_triangles {
                return Err(EngineError::ResourceLimit(format!(
                    "model exceeds {} triangles",
                    limits.max_model_triangles
                )));
            }

            let out_pos: Vec<Vec3> = positions.into_iter().map(Vec3::from).collect();
            let out_nrm: Vec<Vec3> = normals
                .into_iter()
                .map(|n| {
                    let v = Vec3::from(n);
                    if v.length_squared() > 0.0 {
                        v.normalize()
                    } else {
                        Vec3::Y
                    }
                })
                .collect();
            let out_col: Vec<Vec4> = colors.into_iter().map(Vec4::from).collect();
            let tex_set = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.tex_coord())
                .unwrap_or(0);
            let uvs: Vec<[f32; 2]> = if let Some(iter) = reader.read_tex_coords(tex_set) {
                match iter {
                    gltf::mesh::util::ReadTexCoords::U8(i) => i
                        .map(|t| [t[0] as f32 / 255.0, t[1] as f32 / 255.0])
                        .collect(),
                    gltf::mesh::util::ReadTexCoords::U16(i) => i
                        .map(|t| [t[0] as f32 / 65535.0, t[1] as f32 / 65535.0])
                        .collect(),
                    gltf::mesh::util::ReadTexCoords::F32(i) => i.collect(),
                }
            } else {
                vec![[0.0, 0.0]; out_pos.len()]
            };
            if uvs.len() != out_pos.len() {
                return Err(EngineError::Model(
                    "TEXCOORD length mismatch with POSITION".into(),
                ));
            }
            let albedo = if let Some(info) = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
            {
                let index = info.texture().source().index();
                let Some(image) = images.get(index) else {
                    return Err(EngineError::Model(format!(
                        "glTF baseColorTexture index {index} is missing"
                    )));
                };
                Some(image_to_albedo(image)?)
            } else {
                None
            };
            reject_alpha_mode_without_real_alpha(primitive.material().alpha_mode(), &albedo)?;

            meshes.push(SkinMesh {
                positions: out_pos,
                normals: out_nrm,
                colors: out_col,
                uvs,
                joints,
                weights,
                indices,
                double_sided: primitive.material().double_sided(),
                albedo,
            });
        }
    }

    if meshes.is_empty() {
        return Err(EngineError::Model(
            "skinned glTF contained no triangle mesh primitives".into(),
        ));
    }

    let mut clips = Vec::new();
    for animation in document.animations() {
        let name = animation
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("clip_{}", clips.len()));
        let mut channels = vec![JointChannels::default(); skeleton.joint_names.len()];
        let mut duration = 0.0_f32;

        for channel in animation.channels() {
            let node_index = channel.target().node().index();
            let Some(&joint_i) = node_to_joint.get(&node_index) else {
                continue;
            };
            let reader = channel.reader(|buffer| Some(buffers[buffer.index()].0.as_slice()));
            let inputs: Vec<f32> = reader
                .read_inputs()
                .ok_or_else(|| EngineError::Model(format!("clip '{name}' missing sampler inputs")))?
                .collect();
            if let Some(&t) = inputs.last() {
                duration = duration.max(t);
            }

            match channel.target().property() {
                gltf::animation::Property::Translation => {
                    let outputs: Vec<Vec3> = match reader.read_outputs().ok_or_else(|| {
                        EngineError::Model(format!("clip '{name}' missing translation outputs"))
                    })? {
                        gltf::animation::util::ReadOutputs::Translations(i) => {
                            i.map(Vec3::from).collect()
                        }
                        _ => {
                            return Err(EngineError::Model(format!(
                                "clip '{name}' translation channel has wrong output type"
                            )));
                        }
                    };
                    if outputs.len() != inputs.len() {
                        return Err(EngineError::Model(format!(
                            "clip '{name}' translation key count mismatch"
                        )));
                    }
                    channels[joint_i].translation = Some(Vec3Track {
                        times: inputs,
                        values: outputs,
                    });
                }
                gltf::animation::Property::Rotation => {
                    let outputs: Vec<Quat> = match reader.read_outputs().ok_or_else(|| {
                        EngineError::Model(format!("clip '{name}' missing rotation outputs"))
                    })? {
                        gltf::animation::util::ReadOutputs::Rotations(i) => match i {
                            gltf::animation::util::Rotations::F32(iter) => iter
                                .map(|r| Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize())
                                .collect(),
                            gltf::animation::util::Rotations::I8(iter) => iter
                                .map(|r| {
                                    Quat::from_xyzw(
                                        r[0] as f32 / 127.0,
                                        r[1] as f32 / 127.0,
                                        r[2] as f32 / 127.0,
                                        r[3] as f32 / 127.0,
                                    )
                                    .normalize()
                                })
                                .collect(),
                            gltf::animation::util::Rotations::U8(iter) => iter
                                .map(|r| {
                                    Quat::from_xyzw(
                                        r[0] as f32 / 255.0,
                                        r[1] as f32 / 255.0,
                                        r[2] as f32 / 255.0,
                                        r[3] as f32 / 255.0,
                                    )
                                    .normalize()
                                })
                                .collect(),
                            gltf::animation::util::Rotations::I16(iter) => iter
                                .map(|r| {
                                    Quat::from_xyzw(
                                        r[0] as f32 / 32767.0,
                                        r[1] as f32 / 32767.0,
                                        r[2] as f32 / 32767.0,
                                        r[3] as f32 / 32767.0,
                                    )
                                    .normalize()
                                })
                                .collect(),
                            gltf::animation::util::Rotations::U16(iter) => iter
                                .map(|r| {
                                    Quat::from_xyzw(
                                        r[0] as f32 / 65535.0,
                                        r[1] as f32 / 65535.0,
                                        r[2] as f32 / 65535.0,
                                        r[3] as f32 / 65535.0,
                                    )
                                    .normalize()
                                })
                                .collect(),
                        },
                        _ => {
                            return Err(EngineError::Model(format!(
                                "clip '{name}' rotation channel has wrong output type"
                            )));
                        }
                    };
                    if outputs.len() != inputs.len() {
                        return Err(EngineError::Model(format!(
                            "clip '{name}' rotation key count mismatch"
                        )));
                    }
                    channels[joint_i].rotation = Some(QuatTrack {
                        times: inputs,
                        values: outputs,
                    });
                }
                gltf::animation::Property::Scale => {
                    let outputs: Vec<Vec3> = match reader.read_outputs().ok_or_else(|| {
                        EngineError::Model(format!("clip '{name}' missing scale outputs"))
                    })? {
                        gltf::animation::util::ReadOutputs::Scales(i) => {
                            i.map(Vec3::from).collect()
                        }
                        _ => {
                            return Err(EngineError::Model(format!(
                                "clip '{name}' scale channel has wrong output type"
                            )));
                        }
                    };
                    if outputs.len() != inputs.len() {
                        return Err(EngineError::Model(format!(
                            "clip '{name}' scale key count mismatch"
                        )));
                    }
                    channels[joint_i].scale = Some(Vec3Track {
                        times: inputs,
                        values: outputs,
                    });
                }
                gltf::animation::Property::MorphTargetWeights => {}
            }
        }

        clips.push(AnimationClip {
            name,
            duration: duration.max(0.001),
            channels,
        });
    }

    Ok(AnimatedModel {
        skeleton,
        meshes,
        clips,
        bind_local,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn deer_asset_loads_with_clips() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Demo asset lives next to examples; fall back if missing during crate-only checks.
        let path = root.join("../examples/animated_animal/assets/deer.gltf");
        if !path.exists() {
            return;
        }
        let model = AnimatedModel::load_with(&path, root.join(".."), &EngineLimits::default())
            .expect("load deer");
        assert!(!model.meshes.is_empty());
        assert!(!model.skeleton.joint_names.is_empty());
        assert!(model.skeleton.joint_names.len() <= MAX_JOINTS);
        assert!(model.find_clip("Idle").is_some());
        assert!(model.find_clip("Walk").is_some());
        let mats = sample_joint_matrices(&model, 0, 0.0);
        assert_eq!(mats.len(), model.skeleton.joint_names.len());
        assert!(mats.iter().all(|m| m.is_finite()));
        // Quaternius colors live in baseColorFactor; must not stay clay-white.
        let has_tinted = model.meshes.iter().any(|m| {
            m.colors
                .iter()
                .any(|c| c.x < 0.95 || c.y < 0.95 || c.z < 0.95)
        });
        assert!(
            has_tinted,
            "expected material baseColorFactor applied to vertices"
        );
        assert!(
            model
                .meshes
                .iter()
                .all(|m| m.uvs.len() == m.positions.len()),
            "skinned mesh UV length must match POSITION"
        );
    }

    #[test]
    fn semantic_action_resumes_latest_locomotion_intent() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("../examples/animated_animal/assets/deer.gltf");
        if !path.exists() {
            return;
        }
        let model = AnimatedModel::load_with(&path, root.join(".."), &EngineLimits::default())
            .expect("load deer");
        let mut animator = Animator::new(std::sync::Arc::new(model)).expect("animator");
        animator
            .configure_profile(
                AnimationProfile::new()
                    .idle("Idle")
                    .walk("Walk")
                    .attack("Walk"),
            )
            .expect("configure profile");
        animator
            .set_locomotion(Locomotion::Moving { speed_mps: 1.0 })
            .expect("set moving");
        animator
            .play_action(AnimationAction::Attack)
            .expect("attack");
        animator
            .set_locomotion(Locomotion::Idle)
            .expect("queue idle");
        let duration = animator.clip_duration().expect("action duration");
        animator.tick(duration + 0.01);
        assert_eq!(animator.clip_name(), "Idle");
        assert!(animator.is_looping());
    }

    #[test]
    fn invalid_locomotion_speed_is_rejected() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("../examples/animated_animal/assets/deer.gltf");
        if !path.exists() {
            return;
        }
        let model = AnimatedModel::load_with(&path, root.join(".."), &EngineLimits::default())
            .expect("load deer");
        let mut animator = Animator::new(std::sync::Arc::new(model)).expect("animator");
        animator
            .configure_profile(AnimationProfile::new().idle("Idle").walk("Walk"))
            .expect("configure profile");
        let err = animator
            .set_locomotion(Locomotion::Moving { speed_mps: -1.0 })
            .expect_err("negative speed must fail");
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn play_once_holds_last_frame() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = root.join("../examples/animated_animal/assets/deer.gltf");
        if !path.exists() {
            return;
        }
        let model = AnimatedModel::load_with(&path, root.join(".."), &EngineLimits::default())
            .expect("load deer");
        let mut animator = Animator::new(std::sync::Arc::new(model)).expect("animator");
        animator.play_once("Idle").expect("play once");
        assert!(!animator.looping, "play_once must set looping=false");
        let duration = animator.model.clips[animator.clip_index].duration;
        animator.tick(duration + 1.0);
        assert!(
            (animator.time - duration).abs() < 1e-5,
            "play-once must hold last frame, time={} duration={}",
            animator.time,
            duration
        );
        animator.play("Idle").expect("loop play");
        assert!(animator.looping, "play must restore looping=true");
        animator.tick(duration + 0.05);
        assert!(
            animator.time < duration,
            "looping play must wrap, time={} duration={}",
            animator.time,
            duration
        );
    }

    #[test]
    fn bin_space_fourcc_is_unknown_chunk_type() {
        let dir =
            std::env::temp_dir().join(format!("engine-anim-bin-space-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bin_space.glb");
        std::fs::write(&path, crate::model::test_glb_with_bin_space_fourcc()).unwrap();
        let err = AnimatedModel::load_with(&path, &dir, &EngineLimits::default()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.to_ascii_lowercase().contains("unknown chunk type"),
            "expected gltf crate unknown chunk type, got {msg}"
        );
    }

    #[test]
    fn blend_rgb_without_real_alpha_is_load_error() {
        let dir =
            std::env::temp_dir().join(format!("engine-anim-blend-rgb-{}", std::process::id()));
        crate::model::write_minimal_skinned_gltf(&dir, "BLEND", true);
        let err = AnimatedModel::load_with(dir.join("model.gltf"), &dir, &EngineLimits::default())
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("BLEND") && msg.contains("alphaMode"),
            "expected BLEND hole-body gate, got {msg}"
        );
    }

    #[test]
    fn mask_rgb_without_real_alpha_is_load_error() {
        let dir = std::env::temp_dir().join(format!("engine-anim-mask-rgb-{}", std::process::id()));
        crate::model::write_minimal_skinned_gltf(&dir, "MASK", true);
        let err = AnimatedModel::load_with(dir.join("model.gltf"), &dir, &EngineLimits::default())
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MASK") && msg.contains("alphaMode"),
            "expected MASK hole-body gate, got {msg}"
        );
    }
}
