//! Animated characters: clips, blending, morph targets, foot IK.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    /// Spawn a skinned model with clip playback.
    pub fn spawn_animated(&mut self, model: AnimatedModel, place: Place) -> EngineResult<EntityId> {
        self.spawn_animated_shared(Arc::new(model), place)
    }

    /// Spawn from a shared model so a herd does not clone the mesh per animal.
    pub fn spawn_animated_shared(
        &mut self,
        model: Arc<AnimatedModel>,
        place: Place,
    ) -> EngineResult<EntityId> {
        let animator = Animator::new(model)?;
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.animated.insert(
            id,
            AnimatedEntity {
                animator,
                graph: None,
                transform: place.to_matrix(),
                space: self.spawn_space,
            },
        );
        self.animated_order.push(id);
        Ok(id)
    }

    pub fn configure_animation(
        &mut self,
        id: EntityId,
        profile: AnimationProfile,
    ) -> EngineResult<()> {
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .configure_profile(profile)
    }

    pub fn set_locomotion(&mut self, id: EntityId, state: Locomotion) -> EngineResult<()> {
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .set_locomotion(state)
    }

    pub fn play_animation_action(
        &mut self,
        id: EntityId,
        action: AnimationAction,
    ) -> EngineResult<()> {
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .play_action(action)
    }

    pub fn play_animation(&mut self, id: EntityId, clip_name: &str) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.play(clip_name)
    }

    /// Play once (`looping = false`) and hold the last frame when the clip ends.
    pub fn play_animation_once(&mut self, id: EntityId, clip_name: &str) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.play_once(clip_name)
    }

    pub fn set_animation_speed(&mut self, id: EntityId, speed: f32) -> EngineResult<()> {
        if !speed.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "animation speed must be finite, got {speed}"
            )));
        }
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        e.animator.set_speed(speed);
        Ok(())
    }

    /// Sets the weight of an animated entity's morph target by name (see
    /// [`crate::anim::AnimatedModel::morph_targets`]): a face expression, a
    /// blink. Weights are usually 0..1 and stay until changed.
    pub fn set_morph_weight(
        &mut self,
        id: EntityId,
        target: &str,
        weight: f32,
    ) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        let index = e
            .animator
            .model()
            .morph_targets
            .iter()
            .position(|n| n == target)
            .ok_or_else(|| {
                EngineError::InvalidValue(format!("no morph target called {target:?}"))
            })?;
        e.animator.set_morph_weight(index, weight)
    }

    /// Sets the weights of an animated entity's morph targets in model order
    /// (one per [`crate::anim::AnimatedModel::morph_targets`]; a shorter slice
    /// sets the first ones). Cheaper than naming each one every frame.
    pub fn set_morph_weights(&mut self, id: EntityId, weights: &[f32]) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        let n = e.animator.model().morph_targets.len();
        if weights.len() > n {
            return Err(EngineError::InvalidValue(format!(
                "{} morph weights for {n} morph targets",
                weights.len()
            )));
        }
        for (i, &w) in weights.iter().enumerate() {
            e.animator.set_morph_weight(i, w)?;
        }
        Ok(())
    }

    /// Advance all skinned clip clocks (call once per frame before render sync).
    pub fn tick_animations(&mut self, dt: f32) {
        use rayon::prelude::*;
        // State machines pick this frame's clip and speed first.
        for e in self.animated.values_mut() {
            if let Some(g) = e.graph.as_mut() {
                g.step(&mut e.animator);
            }
        }
        // Each clock is its own: on the frame's threads when there are many.
        let mut all: Vec<&mut AnimatedEntity> = self.animated.values_mut().collect();
        if all.len() >= ANIM_PARALLEL_CLOCKS {
            crate::parallel::frame_pool()
                .install(|| all.par_iter_mut().for_each(|e| e.animator.tick(dt)));
        } else {
            for e in all {
                e.animator.tick(dt);
            }
        }
        // Clip events, in spawn order so runs repeat exactly.
        const MAX_PENDING_EVENTS: usize = 4096;
        for id in &self.animated_order {
            if let Some(e) = self.animated.get_mut(id) {
                if let Some(g) = e.graph.as_mut() {
                    g.events(*id, &e.animator, &mut self.anim_events);
                }
            }
        }
        if self.anim_events.len() > MAX_PENDING_EVENTS {
            let extra = self.anim_events.len() - MAX_PENDING_EVENTS;
            self.anim_events.drain(..extra);
        }
        self.tick_foot_ik(dt);
    }

    /// Let a state machine choose `id`'s clips (see [`AnimGraph`]): it
    /// starts in the graph's start state. Errors name any state, clip or
    /// value the graph gets wrong. Replaces a graph set before; while one
    /// runs, don't also call `play_animation` or `set_locomotion`.
    pub fn set_anim_graph(&mut self, id: EntityId, graph: AnimGraph) -> EngineResult<()> {
        let e = self.animated.get_mut(&id).ok_or_else(|| {
            EngineError::InvalidValue(format!(
                "set_anim_graph: entity {id} is not animated (spawn_animated)"
            ))
        })?;
        e.graph = Some(Box::new(crate::anim_graph::AnimGraphRun::start(
            graph,
            &mut e.animator,
        )?));
        Ok(())
    }

    /// Stop the state machine; the clip playing keeps playing.
    pub fn remove_anim_graph(&mut self, id: EntityId) {
        if let Some(e) = self.animated.get_mut(&id) {
            e.graph = None;
            e.animator.set_speed(1.0);
        }
    }

    fn graph_of(&mut self, id: EntityId) -> EngineResult<&mut crate::anim_graph::AnimGraphRun> {
        self.animated
            .get_mut(&id)
            .and_then(|e| e.graph.as_deref_mut())
            .ok_or_else(|| {
                EngineError::InvalidValue(format!(
                    "entity {id} has no anim graph (set_anim_graph first)"
                ))
            })
    }

    /// Set a number the graph's conditions read ("speed").
    pub fn set_anim_value(&mut self, id: EntityId, name: &str, value: f32) -> EngineResult<()> {
        self.graph_of(id)?.set_value(name, value)
    }

    /// Set a yes/no the graph's conditions read ("grounded").
    pub fn set_anim_flag(&mut self, id: EntityId, name: &str, on: bool) -> EngineResult<()> {
        self.graph_of(id)?.set_flag(name, on)
    }

    /// Fire a one-off trigger ("attack"). It is used by the next frame's
    /// transitions and dropped if none of them takes it.
    pub fn anim_trigger(&mut self, id: EntityId, name: &str) -> EngineResult<()> {
        self.graph_of(id)?.trigger(name)
    }

    /// The state `id`'s graph is in.
    pub fn anim_state(&self, id: EntityId) -> Option<&str> {
        self.animated
            .get(&id)?
            .graph
            .as_deref()
            .map(|g| g.state_name())
    }

    /// Clip events ("footstep") since the last call, oldest first.
    pub fn take_anim_events(&mut self) -> Vec<AnimEvent> {
        std::mem::take(&mut self.anim_events)
    }

    /// Cross-fade time between clips for one animated entity (default 0.2 s,
    /// 0 switches at once).
    pub fn set_animation_blend(&mut self, id: EntityId, seconds: f32) -> EngineResult<()> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "animation blend must be finite and >= 0, got {seconds}"
            )));
        }
        self.animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?
            .animator
            .set_blend_time(seconds);
        Ok(())
    }

    /// Let a character's feet find the ground (slopes, steps, rocks, other
    /// bodies), or `None` to switch it off. Finds the legs by bone name
    /// (UE, Mixamo and Blender style rigs); errors if it can't.
    pub fn set_foot_ik(
        &mut self,
        id: EntityId,
        ik: Option<crate::anim::FootIk>,
    ) -> EngineResult<()> {
        let e = self
            .animated
            .get_mut(&id)
            .ok_or(EngineError::UnknownEntity)?;
        let Some(settings) = ik else {
            e.animator.ik = None;
            return Ok(());
        };
        if !(settings.max_step.is_finite() && settings.max_step >= 0.0)
            || !(settings.response.is_finite() && settings.response > 0.0)
        {
            return Err(EngineError::InvalidValue(
                "foot IK max_step must be >= 0 and response > 0".into(),
            ));
        }
        let skeleton = &e.animator.model().skeleton;
        let legs = skeleton.find_legs().ok_or_else(|| {
            EngineError::InvalidValue(
                "no legs found (looked for foot_l/foot_r, LeftFoot/RightFoot, foot.L/foot.R)"
                    .into(),
            )
        })?;
        let hips = (skeleton.parents[legs[0].thigh] == skeleton.parents[legs[1].thigh])
            .then_some(skeleton.parents[legs[0].thigh])
            .flatten();
        let (feet, drop, ground, ground_drop) = e
            .animator
            .ik
            .as_ref()
            .map_or(([Vec3::ZERO; 2], Vec3::ZERO, [0.0; 2], 0.0), |old| {
                (old.feet, old.drop, old.ground, old.ground_drop)
            });
        let model = e.animator.model();
        let soles = legs.map(|l| model.sole_points(l.foot));
        e.animator.ik = Some(crate::anim::IkState {
            settings,
            legs,
            hips,
            feet,
            drop,
            soles,
            ground,
            ground_drop,
        });
        Ok(())
    }

    /// Feet on what is drawn: each sole is set on the highest surface
    /// under it (a mesh's upward faces, the terrain, a physics body) as
    /// high above it as the clip lifts it above the character's floor, so
    /// a planted foot stands on the ground even where the character's
    /// origin is a little off (a server's rougher ground, a step). The hips
    /// follow the lower foot, down or up.
    pub(super) fn tick_foot_ik(&mut self, dt: f32) {
        // Below this lift a foot counts as planted, and is eased onto the
        // ground (a clip's sole a centimetre into its floor stands on it).
        const PLANT: f32 = 0.03;
        // Sampling the clips is the slow part: every character at once on
        // the frame's threads (each reads only its own animator).
        let with_ik: Vec<(EntityId, &AnimatedEntity)> = self
            .animated
            .iter()
            .filter(|(_, e)| e.animator.ik.is_some())
            .map(|(id, e)| (*id, e))
            .collect();
        let leg_of = |&(id, e): &(EntityId, &AnimatedEntity)| {
            let ik = e.animator.ik.as_ref().expect("filtered on foot IK");
            let t = e.transform;
            let globals = e.animator.fk_globals();
            let root = t.w_axis.truncate();
            let soles = [0, 1].map(|i| {
                let g = t * globals[ik.legs[i].foot];
                let [heel, toe] = ik.soles[i].map(|p| g.transform_point3(p));
                if heel.y <= toe.y {
                    heel
                } else {
                    toe
                }
            });
            (id, root, soles, ik.settings, t.inverse())
        };
        let legs: Vec<_> = if with_ik.len() >= ANIM_PARALLEL_POSES {
            use rayon::prelude::*;
            crate::parallel::frame_pool().install(|| with_ik.par_iter().map(leg_of).collect())
        } else {
            with_ik.iter().map(leg_of).collect()
        };
        if legs.is_empty() {
            return;
        }
        let discs: Vec<(Vec3, f32)> = legs.iter().map(|l| (l.1, 1.5)).collect();
        let patches = self.ground_patches(&discs);
        // Probing the ground under every foot reads the world only: on the
        // frame's threads, results in the characters' order.
        let probe = |((id, root, soles, settings, to_model), patch): (
            (EntityId, Vec3, [Vec3; 2], crate::anim::FootIk, Mat4),
            crate::ground_probe::GroundPatch,
        )| {
            let max = settings.max_step;
            // Per foot: the ground's height over the origin (eased: steps
            // and slopes), and what the clip's own sole needs to stand
            // on its floor (at once: it changes every frame of a walk).
            let parts = soles.map(|sole| {
                let from = Vec3::new(sole.x, root.y.max(sole.y) + max, sole.z);
                let lift = sole.y - root.y;
                let want = if lift <= 0.0 {
                    0.0
                } else if lift < PLANT {
                    lift * lift / PLANT
                } else {
                    lift
                };
                self.ground_below_in(&patch, from, 2.0 * max + 0.5)
                    .map(|g| ((g - root.y).clamp(-max, max), want - lift))
            });
            (id, parts, settings, to_model)
        };
        let updates: Vec<_> = if legs.len() >= ANIM_PARALLEL_POSES {
            use rayon::prelude::*;
            crate::parallel::frame_pool()
                .install(|| legs.into_par_iter().zip(patches).map(probe).collect())
        } else {
            legs.into_iter().zip(patches).map(probe).collect()
        };
        for (id, parts, settings, to_model) in updates {
            let Some(ik) = self
                .animated
                .get_mut(&id)
                .and_then(|e| e.animator.ik.as_mut())
            else {
                continue;
            };
            let k = 1.0 - (-settings.response * dt.max(0.0)).exp();
            let max = settings.max_step;
            let fix = parts.map(|p| p.map_or(0.0, |(_, f)| f));
            // Up at once (a foot never stays inside a kerb or a step it
            // walked onto), down eased (no snapping off ledges).
            let follow = |now: &mut f32, to: f32| {
                *now = if to > *now {
                    to
                } else {
                    *now + (to - *now) * k
                };
            };
            for (now, p) in ik.ground.iter_mut().zip(parts) {
                follow(now, p.map_or(0.0, |(g, _)| g));
            }
            let lower = match parts {
                [Some(a), Some(b)] => a.0.min(b.0),
                [Some(a), None] | [None, Some(a)] => a.0.min(0.0),
                [None, None] => 0.0,
            };
            follow(&mut ik.ground_drop, lower);
            for ((foot, g), f) in ik.feet.iter_mut().zip(ik.ground).zip(fix) {
                *foot = to_model.transform_vector3(Vec3::Y * (g + f).clamp(-max, max));
            }
            let drop = if settings.drop_hips {
                (ik.ground_drop + fix[0].min(fix[1])).clamp(-max, max)
            } else {
                0.0
            };
            ik.drop = to_model.transform_vector3(Vec3::Y * drop);
        }
    }

    /// Each foot's lowest sole point in the world as drawn now (after foot
    /// IK), for checks that feet stand on the ground. `None` without legs.
    pub fn sole_points(&self, id: EntityId) -> Option<[Vec3; 2]> {
        let e = self.animated.get(&id)?;
        let model = e.animator.model();
        let (legs, soles) = match e.animator.ik.as_ref() {
            Some(ik) => (ik.legs, ik.soles),
            None => {
                let legs = model.skeleton.find_legs()?;
                (legs, legs.map(|l| model.sole_points(l.foot)))
            }
        };
        let globals = e.animator.joint_globals();
        Some([0, 1].map(|i| {
            let g = e.transform * globals[legs[i].foot];
            let [heel, toe] = soles[i].map(|p| g.transform_point3(p));
            if heel.y <= toe.y {
                heel
            } else {
                toe
            }
        }))
    }

    /// Look up one skinned/animated entity by its typed render identity.
    pub fn animated_entity(&self, id: EntityId) -> EngineResult<&AnimatedEntity> {
        self.animated.get(&id).ok_or(EngineError::UnknownEntity)
    }
    pub fn animated_entities(&self) -> impl Iterator<Item = (&EntityId, &AnimatedEntity)> {
        self.animated_order
            .iter()
            .filter_map(|id| self.animated.get(id).map(|e| (id, e)))
    }
}
