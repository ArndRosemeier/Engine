//! `World`'s crowds (see `crate::crowd`).

use super::{EntityId, World};
use crate::anim::{AnimatedModel, AnimationProfile, Locomotion};
use crate::crowd::{
    far_model, spread, Crowd, CrowdAgent, CrowdAgentId, CrowdId, CrowdProfile, IDLE, WALK,
};
use crate::error::{EngineError, EngineResult};
use crate::place::Place;
use std::sync::Arc;

fn no_crowd(id: CrowdId) -> EngineError {
    EngineError::InvalidValue(format!("no crowd {}", id.0))
}

fn no_agent(id: CrowdAgentId) -> EngineError {
    EngineError::InvalidValue(format!("no crowd agent {}", id.0))
}

impl World {
    /// A crowd of `model` (which needs the profile's idle and walk clips).
    pub fn add_crowd(
        &mut self,
        model: Arc<AnimatedModel>,
        profile: CrowdProfile,
    ) -> EngineResult<CrowdId> {
        let clips = profile.validate(&model)?;
        let far = match &profile.far_model {
            Some(m) => Some(m.clone()),
            None if profile.far_detail < 1.0 => {
                Some(Arc::new(far_model(&model, profile.far_detail)))
            }
            None => None,
        };
        let crowd = Crowd {
            model,
            profile,
            clips,
            far,
            agents: Vec::new(),
        };
        if let Some(i) = self.crowds.iter().position(|c| c.is_none()) {
            self.crowds[i] = Some(crowd);
            return Ok(CrowdId(i as u32));
        }
        self.crowds.push(Some(crowd));
        Ok(CrowdId(self.crowds.len() as u32 - 1))
    }

    /// Remove a crowd and the full characters standing in for its agents.
    pub fn remove_crowd(&mut self, id: CrowdId) -> bool {
        let Some(crowd) = self.crowds.get_mut(id.0 as usize).and_then(|c| c.take()) else {
            return false;
        };
        for a in crowd.agents.iter().flatten() {
            if let Some(e) = a.near {
                self.despawn(e);
            }
        }
        true
    }

    fn crowd_mut(&mut self, id: CrowdId) -> EngineResult<&mut Crowd> {
        self.crowds
            .get_mut(id.0 as usize)
            .and_then(|c| c.as_mut())
            .ok_or(no_crowd(id))
    }

    fn crowd_ref(&self, id: CrowdId) -> EngineResult<&Crowd> {
        self.crowds
            .get(id.0 as usize)
            .and_then(|c| c.as_ref())
            .ok_or(no_crowd(id))
    }

    /// A person in the crowd, standing at `place`.
    pub fn add_crowd_agent(&mut self, crowd: CrowdId, place: Place) -> EngineResult<CrowdAgentId> {
        let c = self.crowd_mut(crowd)?;
        let slot = c.agents.iter().position(|a| a.is_none()).unwrap_or(c.agents.len());
        let agent = CrowdAgent {
            place,
            speed: 0.0,
            clip: IDLE,
            time: spread(slot as u32) * c.duration(IDLE),
            fade: None,
            near: None,
        };
        if slot == c.agents.len() {
            c.agents.push(Some(agent));
        } else {
            c.agents[slot] = Some(agent);
        }
        Ok(CrowdAgentId(slot as u32))
    }

    /// Move an agent; it walks (at a rate matching `speed_mps`) or idles.
    pub fn set_crowd_agent(
        &mut self,
        crowd: CrowdId,
        agent: CrowdAgentId,
        place: Place,
        speed_mps: f32,
    ) -> EngineResult<()> {
        if !speed_mps.is_finite() || speed_mps < 0.0 {
            return Err(EngineError::InvalidValue(
                "crowd agent speed must be finite and >= 0".into(),
            ));
        }
        let a = self
            .crowd_mut(crowd)?
            .agents
            .get_mut(agent.0 as usize)
            .and_then(|a| a.as_mut())
            .ok_or(no_agent(agent))?;
        a.place = place;
        a.speed = speed_mps;
        Ok(())
    }

    pub fn crowd_agent_place(&self, crowd: CrowdId, agent: CrowdAgentId) -> EngineResult<Place> {
        self.crowd_ref(crowd)?
            .agents
            .get(agent.0 as usize)
            .and_then(|a| a.as_ref())
            .map(|a| a.place)
            .ok_or(no_agent(agent))
    }

    pub fn remove_crowd_agent(&mut self, crowd: CrowdId, agent: CrowdAgentId) -> bool {
        let Ok(c) = self.crowd_mut(crowd) else {
            return false;
        };
        let Some(a) = c.agents.get_mut(agent.0 as usize).and_then(|a| a.take()) else {
            return false;
        };
        if let Some(e) = a.near {
            self.despawn(e);
        }
        true
    }

    pub fn crowd_agent_count(&self, crowd: CrowdId) -> usize {
        self.crowd_ref(crowd)
            .map_or(0, |c| c.agents.iter().flatten().count())
    }

    /// How many agents are full characters right now.
    pub fn crowd_near_count(&self, crowd: CrowdId) -> usize {
        self.crowd_ref(crowd).map_or(0, |c| {
            c.agents.iter().flatten().filter(|a| a.near.is_some()).count()
        })
    }

    /// The full character standing in for an agent while it is near (to
    /// hit it, put a decal on it, ragdoll it).
    pub fn crowd_agent_character(&self, crowd: CrowdId, agent: CrowdAgentId) -> Option<EntityId> {
        self.crowd_ref(crowd)
            .ok()?
            .agents
            .get(agent.0 as usize)?
            .as_ref()?
            .near
    }

    pub(crate) fn crowds(&self) -> impl Iterator<Item = (CrowdId, &Crowd)> {
        self.crowds
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|c| (CrowdId(i as u32), c)))
    }

    /// Animate the far agents and swap the nearest to full characters and
    /// back. Runs every frame before the animations (call it yourself only
    /// when driving the world by hand).
    pub fn tick_crowds(&mut self, dt: f32) {
        let eye = self.camera().eye();
        for ci in 0..self.crowds.len() {
            let Some(mut crowd) = self.crowds[ci].take() else {
                continue;
            };
            self.tick_crowd(&mut crowd, eye, dt);
            self.crowds[ci] = Some(crowd);
        }
    }

    fn tick_crowd(&mut self, crowd: &mut Crowd, eye: glam::Vec3, dt: f32) {
        let p = crowd.profile.clone();
        // The nearest within reach, a little stickier for those already near.
        let mut want: Vec<(f32, usize)> = crowd
            .agents
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                let a = a.as_ref()?;
                let d = a.place.position.distance(eye) / if a.near.is_some() { 1.1 } else { 1.0 };
                (d < p.near_m).then_some((d, i))
            })
            .collect();
        want.sort_by(|x, y| x.0.total_cmp(&y.0));
        want.truncate(p.max_near);
        let mut wanted = vec![false; crowd.agents.len()];
        for (_, i) in &want {
            wanted[*i] = true;
        }
        let state = |a: &CrowdAgent| {
            if a.clip == WALK {
                Locomotion::Moving {
                    speed_mps: a.speed.max(0.06),
                }
            } else {
                Locomotion::Idle
            }
        };
        for (i, slot) in crowd.agents.iter_mut().enumerate() {
            let Some(a) = slot.as_mut() else { continue };
            if let Some(e) = a.near {
                let alive = self.animated.contains_key(&e);
                if alive && wanted[i] {
                    continue;
                }
                // Back into the crowd, at the step the character was on.
                if let Some(anim) = self.animated.get(&e) {
                    let walking = anim.animator.clip_name() == p.walk;
                    a.clip = if walking { WALK } else { IDLE };
                    a.time = anim.animator.time();
                    a.fade = None;
                }
                a.near = None;
                if alive {
                    self.despawn(e);
                }
            } else if wanted[i] {
                // Out of the crowd as a full character with the same pose.
                let want_clip = if a.speed > 0.05 { WALK } else { IDLE };
                if want_clip != a.clip {
                    a.clip = want_clip;
                    a.time = 0.0;
                }
                let Ok(e) = self.spawn_animated_shared(crowd.model.clone(), a.place) else {
                    continue;
                };
                let setup = AnimationProfile::new()
                    .idle(p.idle.clone())
                    .walk(p.walk.clone())
                    .walk_speed(p.walk_mps);
                let ok = self.configure_animation(e, setup).is_ok()
                    && self.set_animation_blend(e, p.blend_s).is_ok()
                    && self.set_locomotion(e, state(a)).is_ok();
                if !ok {
                    self.despawn(e);
                    continue;
                }
                if let Some(anim) = self.animated.get_mut(&e) {
                    anim.animator.snap_to(a.time);
                }
                a.fade = None;
                a.near = Some(e);
            }
        }
        let model = crowd.model.clone();
        let clips = crowd.clips;
        for a in crowd.agents.iter_mut().flatten() {
            match a.near {
                Some(e) => {
                    a.clip = if a.speed > 0.05 { WALK } else { IDLE };
                    if let Some(anim) = self.animated.get_mut(&e) {
                        anim.transform = a.place.to_matrix();
                        let _ = anim.animator.set_locomotion(state(a));
                    }
                }
                None => crate::crowd::advance(&p, &model, clips, a, dt),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use std::sync::Arc;

    fn character() -> Arc<AnimatedModel> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/model_viewer/assets/character.glb"
        );
        let examples = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
        Arc::new(
            AnimatedModel::load_with(path, examples, &crate::limits::EngineLimits::default())
                .unwrap(),
        )
    }

    #[test]
    fn near_agents_become_characters_at_the_same_step() {
        let mut world = World::new();
        let crowd = world
            .add_crowd(
                character(),
                CrowdProfile::new("idle", "walk").near(16.0, 2).walk_speed(1.3),
            )
            .unwrap();
        let agents: Vec<_> = (0..5)
            .map(|i| {
                world
                    .add_crowd_agent(crowd, Place::new(i as f32 * 10.0, 0.0, 0.0))
                    .unwrap()
            })
            .collect();
        world.look_first_person((-5.0, 1.7, 0.0), 90.0, 0.0);
        for a in &agents {
            let p = world.crowd_agent_place(crowd, *a).unwrap();
            world.set_crowd_agent(crowd, *a, p, 1.3).unwrap();
        }
        let dt = 1.0 / 60.0;
        for _ in 0..30 {
            world.tick_crowds(dt);
            world.tick_animations(dt);
        }
        // Two within 16 m, at most two at once.
        assert_eq!(world.crowd_near_count(crowd), 2);
        let e = world.crowd_agent_character(crowd, agents[0]).unwrap();
        assert!(world.crowd_agent_character(crowd, agents[3]).is_none());

        // Walking away hands the step back without a jump in phase.
        let step = world.animated_entity(e).unwrap().animator().time();
        world.look_first_person((-500.0, 1.7, 0.0), 90.0, 0.0);
        world.tick_crowds(dt);
        assert_eq!(world.crowd_near_count(crowd), 0);
        assert!(!world.contains_animated(e));
        let c = world.crowds().next().unwrap().1;
        let back = c.agents[0].as_ref().unwrap();
        assert_eq!(back.clip, crate::crowd::WALK);
        assert!((back.time - step - dt * 1.0).abs() < 0.02, "{step} -> {}", back.time);
        assert_eq!(world.crowd_agent_count(crowd), 5);
        assert!(world.remove_crowd(crowd));
    }
}
