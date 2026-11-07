//! Prefabs: a thing made of parts (meshes, characters, lamps, bodies) that
//! you build once and place as often as you like.
//!
//! ```ignore
//! let mut cart = Prefab::new("cart");
//! let body = cart.mesh("body", cart_mesh, Place::new(0.0, 0.6, 0.0));
//! for (name, x, z) in [("wheel_fl", -0.7, 1.0), ("wheel_fr", 0.7, 1.0)] {
//!     cart.mesh_on(body, name, wheel_mesh.clone(), Place::new(x, -0.25, z));
//! }
//! cart.lamp_on(body, Lamp::point((0.0, 1.2, 1.4), rgb(255, 200, 120), 3.0, 10.0));
//! cart.body(body, Body::dynamic().density(0.7));
//!
//! let a = world.spawn_prefab(&cart, Place::new(5.0, 0.0, 2.0))?;
//! world.set_place(a.part("wheel_fl").unwrap(), spun_wheel)?;
//! ```
//!
//! Registered prefabs ([`crate::world::World::register_prefab`]) can be
//! spawned by name, which is what saved worlds use to rebuild things.

use crate::anim::AnimatedModel;
use crate::error::{EngineError, EngineResult};
use crate::lamps::{Lamp, LampId};
use crate::mesh::Mesh;
use crate::physics::Body;
use crate::place::Place;
use crate::world::EntityId;
use std::sync::Arc;

/// One part of a [`Prefab`], returned when you add it so you can hang more
/// parts on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PartRef(usize);

#[derive(Clone, Debug)]
pub(crate) enum PartKind {
    Pivot,
    Mesh(Arc<Mesh>),
    Model {
        model: Arc<AnimatedModel>,
        clip: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Part {
    pub name: String,
    pub kind: PartKind,
    pub place: Place,
    pub parent: Option<usize>,
    pub body: Option<Body>,
    pub casts_shadow: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PrefabLamp {
    pub on: Option<usize>,
    pub bone: Option<String>,
    pub lamp: Lamp,
}

/// A blueprint for a thing made of parts. Cheap to clone (meshes and models
/// are shared).
#[derive(Clone, Debug)]
pub struct Prefab {
    name: String,
    pub(crate) parts: Vec<Part>,
    pub(crate) lamps: Vec<PrefabLamp>,
}

impl Prefab {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parts: Vec::new(),
            lamps: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    fn push(
        &mut self,
        parent: Option<PartRef>,
        name: &str,
        kind: PartKind,
        place: Place,
    ) -> PartRef {
        self.parts.push(Part {
            name: name.to_string(),
            kind,
            place,
            parent: parent.map(|p| p.0),
            body: None,
            casts_shadow: true,
        });
        PartRef(self.parts.len() - 1)
    }

    /// A mesh at the top level, `place` relative to where the prefab goes.
    pub fn mesh(&mut self, name: &str, mesh: Mesh, place: Place) -> PartRef {
        self.push(None, name, PartKind::Mesh(Arc::new(mesh)), place)
    }

    /// A mesh hanging on another part, `place` relative to it.
    pub fn mesh_on(&mut self, parent: PartRef, name: &str, mesh: Mesh, place: Place) -> PartRef {
        self.push(Some(parent), name, PartKind::Mesh(Arc::new(mesh)), place)
    }

    /// An invisible handle to group parts under (a turret's hinge, say).
    pub fn pivot(&mut self, name: &str, place: Place) -> PartRef {
        self.push(None, name, PartKind::Pivot, place)
    }

    pub fn pivot_on(&mut self, parent: PartRef, name: &str, place: Place) -> PartRef {
        self.push(Some(parent), name, PartKind::Pivot, place)
    }

    /// An animated character or animal, optionally playing a looping clip.
    pub fn model(
        &mut self,
        name: &str,
        model: Arc<AnimatedModel>,
        place: Place,
        clip: Option<&str>,
    ) -> PartRef {
        let kind = PartKind::Model {
            model,
            clip: clip.map(str::to_string),
        };
        self.push(None, name, kind, place)
    }

    pub fn model_on(
        &mut self,
        parent: PartRef,
        name: &str,
        model: Arc<AnimatedModel>,
        place: Place,
        clip: Option<&str>,
    ) -> PartRef {
        let kind = PartKind::Model {
            model,
            clip: clip.map(str::to_string),
        };
        self.push(Some(parent), name, kind, place)
    }

    /// A lamp; its position (and a spotlight's direction) are relative to
    /// where the prefab goes.
    pub fn lamp(&mut self, lamp: Lamp) -> &mut Self {
        self.lamps.push(PrefabLamp {
            on: None,
            bone: None,
            lamp,
        });
        self
    }

    /// A lamp riding on a part; position and direction relative to it.
    pub fn lamp_on(&mut self, part: PartRef, lamp: Lamp) -> &mut Self {
        self.lamps.push(PrefabLamp {
            on: Some(part.0),
            bone: None,
            lamp,
        });
        self
    }

    /// A lamp in a model part's hand (or on any other bone).
    pub fn lamp_on_bone(&mut self, part: PartRef, bone: &str, lamp: Lamp) -> &mut Self {
        self.lamps.push(PrefabLamp {
            on: Some(part.0),
            bone: Some(bone.to_string()),
            lamp,
        });
        self
    }

    /// Give a mesh part a physics body. A dynamic body lets go of its parent
    /// when spawned (physics moves it; its own children still follow it).
    pub fn body(&mut self, part: PartRef, body: Body) -> &mut Self {
        if let Some(p) = self.parts.get_mut(part.0) {
            p.body = Some(body);
        }
        self
    }

    /// Keep a part out of shadow maps (glowing bulbs, flames).
    pub fn no_shadow(&mut self, part: PartRef) -> &mut Self {
        if let Some(p) = self.parts.get_mut(part.0) {
            p.casts_shadow = false;
        }
        self
    }

    /// The part called `name`.
    pub fn find(&self, name: &str) -> Option<PartRef> {
        self.parts.iter().position(|p| p.name == name).map(PartRef)
    }

    pub(crate) fn validate(&self) -> EngineResult<()> {
        if self.parts.is_empty() && self.lamps.is_empty() {
            return Err(EngineError::InvalidValue(format!(
                "prefab {:?} has no parts",
                self.name
            )));
        }
        for (i, p) in self.parts.iter().enumerate() {
            if p.parent.is_some_and(|q| q >= i) {
                return Err(EngineError::InvalidValue(format!(
                    "prefab part {:?} hangs on a part added after it",
                    p.name
                )));
            }
            if p.body.is_some() && !matches!(p.kind, PartKind::Mesh(_)) {
                return Err(EngineError::InvalidValue(format!(
                    "prefab part {:?}: only mesh parts can have bodies",
                    p.name
                )));
            }
            if let Some(b) = &p.body {
                b.validate()?;
            }
        }
        for l in &self.lamps {
            l.lamp.validate()?;
            if l.bone.is_some()
                && !l
                    .on
                    .and_then(|i| self.parts.get(i))
                    .is_some_and(|p| matches!(p.kind, PartKind::Model { .. }))
            {
                return Err(EngineError::InvalidValue(
                    "a lamp on a bone needs a model part".into(),
                ));
            }
        }
        Ok(())
    }
}

/// A placed prefab: its root entity, its parts by name and its lamps.
#[derive(Clone, Debug)]
pub struct Spawned {
    /// Move this to move the whole thing; despawn it to remove everything.
    pub root: EntityId,
    pub(crate) parts: Vec<(String, EntityId)>,
    pub lamps: Vec<LampId>,
}

impl Spawned {
    /// The entity made for the part called `name`.
    pub fn part(&self, name: &str) -> Option<EntityId> {
        self.parts
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
    }

    /// Every part, in the order they were added to the prefab.
    pub fn parts(&self) -> impl Iterator<Item = (&str, EntityId)> {
        self.parts.iter().map(|(n, id)| (n.as_str(), *id))
    }
}
