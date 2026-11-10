//! Animation state machines: named states that play clips, and rules for
//! when to go from one to the next, driven by values the game sets.
//!
//! ```ignore
//! let graph = AnimGraph::new("idle")
//!     .state(AnimState::looped("idle", "idle"))
//!     .state(AnimState::looped("walk", "walk").match_speed("speed", 1.4))
//!     .state(AnimState::once("attack", "sword attack").then("idle"))
//!     .transition("idle", "walk", When::above("speed", 0.2))
//!     .transition("walk", "idle", When::below("speed", 0.1))
//!     .transition_any("attack", When::trigger("attack"))
//!     .event("walk", 0.3, "footstep");
//! world.set_anim_graph(hero, graph)?;
//! // every frame:
//! world.set_anim_value(hero, "speed", speed)?;
//! if attack_pressed { world.anim_trigger(hero, "attack")?; }
//! for e in world.take_anim_events() { /* play a footstep sound at e.entity */ }
//! ```

use crate::anim::{AnimatedModel, Animator};
use crate::error::{EngineError, EngineResult};
use crate::world::EntityId;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

/// One state: which clip plays, looping or once, and how fast.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimState {
    pub name: String,
    pub clip: String,
    pub looping: bool,
    /// Playback speed (1 = as authored).
    pub speed: f32,
    /// Play at `value / authored_mps` (clamped to 0.5..2) so feet match
    /// the ground: `(value name, how fast the clip moves in m/s)`.
    pub match_speed: Option<(String, f32)>,
    /// For a clip played once: the state to go to when it ends. Without
    /// it, the last frame holds until a transition leaves.
    pub then: Option<String>,
}

impl AnimState {
    /// A state that loops `clip` (idle, walk, run).
    pub fn looped(name: impl Into<String>, clip: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            clip: clip.into(),
            looping: true,
            speed: 1.0,
            match_speed: None,
            then: None,
        }
    }

    /// A state that plays `clip` once (attack, jump, hit).
    pub fn once(name: impl Into<String>, clip: impl Into<String>) -> Self {
        Self {
            looping: false,
            ..Self::looped(name, clip)
        }
    }

    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Play faster or slower with the value `value` (say "speed" in m/s),
    /// for a clip that moves `authored_mps` when played as authored.
    pub fn match_speed(mut self, value: impl Into<String>, authored_mps: f32) -> Self {
        self.match_speed = Some((value.into(), authored_mps));
        self
    }

    /// When this (once) clip ends, go to `state`.
    pub fn then(mut self, state: impl Into<String>) -> Self {
        self.then = Some(state.into());
        self
    }
}

/// A condition on the values the game sets with `World::set_anim_value`,
/// `set_anim_flag` and `anim_trigger`.
#[derive(Clone, Debug, PartialEq)]
pub enum When {
    /// The value is above the threshold.
    Above(String, f32),
    /// The value is below the threshold.
    Below(String, f32),
    /// The flag is set to this.
    Flag(String, bool),
    /// The trigger was fired (it is used up when the transition is taken).
    Trigger(String),
    /// The state's clip played to its end (for clips played once).
    Finished,
    /// Every one of these holds.
    All(Vec<When>),
}

impl When {
    pub fn above(value: impl Into<String>, threshold: f32) -> Self {
        Self::Above(value.into(), threshold)
    }
    pub fn below(value: impl Into<String>, threshold: f32) -> Self {
        Self::Below(value.into(), threshold)
    }
    pub fn flag(name: impl Into<String>, on: bool) -> Self {
        Self::Flag(name.into(), on)
    }
    pub fn trigger(name: impl Into<String>) -> Self {
        Self::Trigger(name.into())
    }
    pub fn all(conditions: impl IntoIterator<Item = When>) -> Self {
        Self::All(conditions.into_iter().collect())
    }

    fn names(
        &self,
        values: &mut BTreeSet<String>,
        flags: &mut BTreeSet<String>,
        triggers: &mut BTreeSet<String>,
    ) {
        match self {
            When::Above(n, _) | When::Below(n, _) => {
                values.insert(n.clone());
            }
            When::Flag(n, _) => {
                flags.insert(n.clone());
            }
            When::Trigger(n) => {
                triggers.insert(n.clone());
            }
            When::Finished => {}
            When::All(all) => all.iter().for_each(|w| w.names(values, flags, triggers)),
        }
    }

    fn holds(&self, run: &AnimGraphRun, finished: bool) -> bool {
        match self {
            When::Above(n, t) => run.values.get(n).is_some_and(|v| v > t),
            When::Below(n, t) => run.values.get(n).is_none_or(|v| v < t),
            When::Flag(n, on) => run.flags.get(n).copied().unwrap_or(false) == *on,
            When::Trigger(n) => run.triggers.contains(n),
            When::Finished => finished,
            When::All(all) => all.iter().all(|w| w.holds(run, finished)),
        }
    }

    fn uses_trigger(&self, out: &mut Vec<String>) {
        match self {
            When::Trigger(n) => out.push(n.clone()),
            When::All(all) => all.iter().for_each(|w| w.uses_trigger(out)),
            _ => {}
        }
    }
}

/// A rule: go from `from` (any state when `None`) to `to` when `when` holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    pub from: Option<String>,
    pub to: String,
    pub when: When,
    /// Cross-fade seconds for this transition (the entity's blend time,
    /// 0.2 s by default, when `None`).
    pub blend: Option<f32>,
}

/// A named moment in a clip ("footstep" at 0.3 s), reported by
/// `World::take_anim_events` each time playback passes it.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipEvent {
    pub clip: String,
    pub at: f32,
    pub name: String,
}

/// A clip event that happened: `name` on `entity`, in `state`.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimEvent {
    pub entity: EntityId,
    pub name: String,
    pub state: String,
}

/// An animation state machine. Build it with [`AnimGraph::new`] and the
/// builder methods, then give it to an animated entity with
/// `World::set_anim_graph`. Transitions are checked in the order they were
/// added; the first that holds is taken (one per frame).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimGraph {
    pub start: String,
    pub states: Vec<AnimState>,
    pub transitions: Vec<Transition>,
    pub events: Vec<ClipEvent>,
}

impl AnimGraph {
    /// A graph that starts in state `start`.
    pub fn new(start: impl Into<String>) -> Self {
        Self {
            start: start.into(),
            ..Self::default()
        }
    }

    pub fn state(mut self, state: AnimState) -> Self {
        self.states.push(state);
        self
    }

    /// Go from `from` to `to` when `when` holds.
    pub fn transition(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        when: When,
    ) -> Self {
        self.transitions.push(Transition {
            from: Some(from.into()),
            to: to.into(),
            when,
            blend: None,
        });
        self
    }

    /// Go to `to` from whatever state is playing (but not from `to` itself)
    /// when `when` holds: hit, death, attack.
    pub fn transition_any(mut self, to: impl Into<String>, when: When) -> Self {
        self.transitions.push(Transition {
            from: None,
            to: to.into(),
            when,
            blend: None,
        });
        self
    }

    /// Cross-fade time for the transition added last.
    pub fn blend(mut self, seconds: f32) -> Self {
        if let Some(t) = self.transitions.last_mut() {
            t.blend = Some(seconds);
        }
        self
    }

    /// Report `name` each time `clip` plays past `at` seconds.
    pub fn event(mut self, clip: impl Into<String>, at: f32, name: impl Into<String>) -> Self {
        self.events.push(ClipEvent {
            clip: clip.into(),
            at,
            name: name.into(),
        });
        self
    }

    fn state_index(&self, name: &str) -> Option<usize> {
        self.states.iter().position(|s| s.name == name)
    }

    /// Checks every name against the graph and the model's clips.
    pub fn validate(&self, model: &AnimatedModel) -> EngineResult<()> {
        let bad = |what: String| Err(EngineError::InvalidValue(format!("anim graph: {what}")));
        let clips: Vec<&str> = model.clip_names().collect();
        let state_names: Vec<&str> = self.states.iter().map(|s| s.name.as_str()).collect();
        if self.states.is_empty() {
            return bad("no states (add one with .state(AnimState::looped(..)))".into());
        }
        for (i, s) in self.states.iter().enumerate() {
            if state_names[..i].contains(&s.name.as_str()) {
                return bad(format!("two states are called {:?}", s.name));
            }
            if !clips.contains(&s.clip.as_str()) {
                return bad(format!(
                    "state {:?} plays clip {:?}, which the model does not have (clips: {clips:?})",
                    s.name, s.clip
                ));
            }
            if !(s.speed.is_finite() && s.speed >= 0.0) {
                return bad(format!("state {:?} speed must be finite and >= 0", s.name));
            }
            if s.match_speed
                .as_ref()
                .is_some_and(|(_, m)| !(m.is_finite() && *m > 0.0))
            {
                return bad(format!("state {:?} match_speed needs a speed > 0", s.name));
            }
            if let Some(next) = &s.then {
                if self.state_index(next).is_none() {
                    return bad(format!(
                        "state {:?} goes on to unknown state {next:?} (states: {state_names:?})",
                        s.name
                    ));
                }
            }
        }
        if self.state_index(&self.start).is_none() {
            return bad(format!(
                "start state {:?} is not one of {state_names:?}",
                self.start
            ));
        }
        for t in &self.transitions {
            for end in t.from.iter().chain(std::iter::once(&t.to)) {
                if self.state_index(end).is_none() {
                    return bad(format!(
                        "transition names unknown state {end:?} (states: {state_names:?})"
                    ));
                }
            }
            if t.blend.is_some_and(|b| !(b.is_finite() && b >= 0.0)) {
                return bad(format!(
                    "transition to {:?}: blend must be finite and >= 0",
                    t.to
                ));
            }
        }
        for e in &self.events {
            if !clips.contains(&e.clip.as_str()) {
                return bad(format!(
                    "event {:?} is on unknown clip {:?} (clips: {clips:?})",
                    e.name, e.clip
                ));
            }
            if !(e.at.is_finite() && e.at >= 0.0) {
                return bad(format!("event {:?} time must be finite and >= 0", e.name));
            }
        }
        Ok(())
    }

    /// Value, flag and trigger names the graph reads.
    fn names(&self) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        let (mut values, mut flags, mut triggers) = Default::default();
        for t in &self.transitions {
            t.when.names(&mut values, &mut flags, &mut triggers);
        }
        for s in &self.states {
            if let Some((v, _)) = &s.match_speed {
                values.insert(v.clone());
            }
        }
        (values, flags, triggers)
    }
}

/// A graph running on one entity.
#[derive(Clone, Debug)]
pub(crate) struct AnimGraphRun {
    graph: Arc<AnimGraph>,
    state: usize,
    values: HashMap<String, f32>,
    flags: HashMap<String, bool>,
    triggers: BTreeSet<String>,
    known: (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>),
    /// Where the clip was before this frame's tick, for clip events.
    last_time: f32,
    last_clip: String,
}

impl AnimGraphRun {
    pub fn start(graph: AnimGraph, animator: &mut Animator) -> EngineResult<Self> {
        graph.validate(animator.model())?;
        let known = graph.names();
        let state = graph.state_index(&graph.start).expect("validated start");
        let mut run = Self {
            graph: Arc::new(graph),
            state,
            values: HashMap::new(),
            flags: HashMap::new(),
            triggers: BTreeSet::new(),
            known,
            last_time: 0.0,
            last_clip: String::new(),
        };
        run.enter(state, None, animator)?;
        Ok(run)
    }

    pub fn state_name(&self) -> &str {
        &self.graph.states[self.state].name
    }

    fn unknown(kind: &str, name: &str, known: &BTreeSet<String>) -> EngineError {
        EngineError::InvalidValue(format!(
            "anim graph has no {kind} called {name:?} (its {kind}s: {known:?})"
        ))
    }

    pub fn set_value(&mut self, name: &str, value: f32) -> EngineResult<()> {
        if !self.known.0.contains(name) {
            return Err(Self::unknown("value", name, &self.known.0));
        }
        if !value.is_finite() {
            return Err(EngineError::InvalidValue(format!(
                "anim value {name:?} must be finite, got {value}"
            )));
        }
        self.values.insert(name.to_owned(), value);
        Ok(())
    }

    pub fn set_flag(&mut self, name: &str, on: bool) -> EngineResult<()> {
        if !self.known.1.contains(name) {
            return Err(Self::unknown("flag", name, &self.known.1));
        }
        self.flags.insert(name.to_owned(), on);
        Ok(())
    }

    pub fn trigger(&mut self, name: &str) -> EngineResult<()> {
        if !self.known.2.contains(name) {
            return Err(Self::unknown("trigger", name, &self.known.2));
        }
        self.triggers.insert(name.to_owned());
        Ok(())
    }

    fn enter(
        &mut self,
        state: usize,
        blend: Option<f32>,
        animator: &mut Animator,
    ) -> EngineResult<()> {
        let s = &self.graph.states[state];
        let normal = animator.blend_time();
        if let Some(b) = blend {
            animator.set_blend_time(b);
        }
        let played = if s.looping {
            animator.play(&s.clip)
        } else {
            animator.play_once(&s.clip)
        };
        animator.set_blend_time(normal);
        played?;
        self.state = state;
        // Just before 0, so an event at 0 s fires on entry.
        self.last_time = -1e-6;
        self.last_clip = s.clip.clone();
        Ok(())
    }

    /// Before the clocks move: take at most one transition and set the speed.
    pub fn step(&mut self, animator: &mut Animator) {
        let finished = !animator.is_looping()
            && animator
                .clip_duration()
                .is_some_and(|d| animator.time() >= d - 1e-4);
        let graph = Arc::clone(&self.graph);
        let current = &graph.states[self.state];
        let mut next = None;
        for t in &graph.transitions {
            let from_ok = match &t.from {
                Some(f) => *f == current.name,
                None => t.to != current.name,
            };
            if from_ok && t.when.holds(self, finished) {
                next = Some((t.to.as_str(), t.blend, Some(&t.when)));
                break;
            }
        }
        if next.is_none() && finished {
            if let Some(then) = &current.then {
                next = Some((then.as_str(), None, None));
            }
        }
        if let Some((to, blend, when)) = next {
            if let Some(when) = when {
                let mut used = Vec::new();
                when.uses_trigger(&mut used);
                for u in used {
                    self.triggers.remove(&u);
                }
            }
            let index = graph.state_index(to).expect("validated transition");
            // Clips were checked when the graph was set.
            let _ = self.enter(index, blend, animator);
        }
        // A trigger nothing took this frame is dropped, not saved for later.
        self.triggers.clear();
        let s = &graph.states[self.state];
        let matched = s.match_speed.as_ref().map_or(1.0, |(value, authored)| {
            (self.values.get(value).copied().unwrap_or(0.0) / authored).clamp(0.5, 2.0)
        });
        animator.set_speed(s.speed * matched);
    }

    /// After the clocks moved: the clip events playback passed.
    pub fn events(&mut self, entity: EntityId, animator: &Animator, out: &mut Vec<AnimEvent>) {
        let (now, clip) = (animator.time(), animator.clip_name());
        if clip != self.last_clip {
            self.last_clip = clip.to_owned();
            self.last_time = now;
            return;
        }
        let before = self.last_time;
        self.last_time = now;
        let wrapped = animator.is_looping() && now < before;
        let duration = animator.clip_duration().unwrap_or(0.0);
        for e in self.graph.events.iter().filter(|e| e.clip == clip) {
            let passed = if wrapped {
                (e.at > before && e.at <= duration) || e.at <= now
            } else {
                e.at > before && e.at <= now
            };
            if passed {
                out.push(AnimEvent {
                    entity,
                    name: e.name.clone(),
                    state: self.state_name().to_owned(),
                });
            }
        }
    }
}
