//! Your own data on entities: `world.insert(orc, Health(100))`.
//!
//! Any `Send + Sync + 'static` type can be a component; the engine keeps one
//! value of each type per entity, removes them when the entity is despawned,
//! and lists them in queries in entity order (oldest first), so runs are
//! repeatable. Types registered with [`World::save_component`] are written by
//! [`World::save`] for saved (prefab) entities and restored by
//! [`World::load`].
//!
//! ```ignore
//! struct Health(i32);
//! struct Burning;
//!
//! let orc = world.spawn(orc_mesh);
//! world.insert(orc, Health(100))?;
//! world.insert(orc, Burning)?;
//!
//! for (_, health, _) in world.query2_mut::<Health, Burning>() {
//!     health.0 -= 1;
//! }
//! for id in world.ids_with::<Health>() {
//!     if world.get::<Health>(id).is_some_and(|h| h.0 <= 0) {
//!         world.despawn(id); // its components go with it
//!     }
//! }
//! ```

use super::{EntityId, World};
use crate::error::{EngineError, EngineResult};
use serde::{de::DeserializeOwned, Serialize};
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};

type ToJson<T> = fn(&T) -> Result<serde_json::Value, serde_json::Error>;
type FromJson<T> = fn(serde_json::Value) -> Result<T, serde_json::Error>;

struct Store<T> {
    values: BTreeMap<EntityId, T>,
    /// Set by [`World::save_component`].
    saved_as: Option<(String, ToJson<T>, FromJson<T>)>,
}

trait AnyStore: Send + Sync {
    fn remove(&mut self, id: EntityId);
    fn contains(&self, id: EntityId) -> bool;
    fn len(&self) -> usize;
    fn type_name(&self) -> &'static str;
    fn saved_name(&self) -> Option<&str>;
    fn save(&self, keep: &dyn Fn(EntityId) -> bool) -> EngineResult<Vec<(u64, serde_json::Value)>>;
    fn load(
        &mut self,
        entries: Vec<(u64, serde_json::Value)>,
        map: &dyn Fn(u64) -> Option<EntityId>,
    ) -> EngineResult<()>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Send + Sync + 'static> AnyStore for Store<T> {
    fn remove(&mut self, id: EntityId) {
        self.values.remove(&id);
    }
    fn contains(&self, id: EntityId) -> bool {
        self.values.contains_key(&id)
    }
    fn len(&self) -> usize {
        self.values.len()
    }
    fn type_name(&self) -> &'static str {
        short_type_name::<T>()
    }
    fn saved_name(&self) -> Option<&str> {
        self.saved_as.as_ref().map(|(name, _, _)| name.as_str())
    }
    fn save(&self, keep: &dyn Fn(EntityId) -> bool) -> EngineResult<Vec<(u64, serde_json::Value)>> {
        let Some((name, to_json, _)) = &self.saved_as else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (id, value) in &self.values {
            if keep(*id) {
                let json = to_json(value).map_err(|e| {
                    EngineError::InvalidValue(format!("save: component {name:?}: {e}"))
                })?;
                out.push((id.raw(), json));
            }
        }
        Ok(out)
    }
    fn load(
        &mut self,
        entries: Vec<(u64, serde_json::Value)>,
        map: &dyn Fn(u64) -> Option<EntityId>,
    ) -> EngineResult<()> {
        let Some((name, _, from_json)) = &self.saved_as else {
            return Ok(());
        };
        for (old, json) in entries {
            let Some(id) = map(old) else { continue };
            let value = from_json(json).map_err(|e| {
                EngineError::InvalidValue(format!(
                    "save: component {name:?} doesn't fit its type: {e}"
                ))
            })?;
            self.values.insert(id, value);
        }
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// `game::Health` -> `Health`, `Vec<game::Item>` -> `Vec<Item>`.
fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    match full.find('<') {
        Some(_) => full,
        None => full.rsplit("::").next().unwrap_or(full),
    }
}

/// Every component type's values. Lives in [`World`].
#[derive(Default)]
pub(crate) struct Components {
    stores: HashMap<TypeId, Box<dyn AnyStore>>,
}

impl std::fmt::Debug for Components {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.stores.values().map(|s| (s.type_name(), s.len())))
            .finish()
    }
}

impl Components {
    fn store<T: Send + Sync + 'static>(&self) -> Option<&Store<T>> {
        self.stores
            .get(&TypeId::of::<T>())
            .and_then(|s| s.as_any().downcast_ref::<Store<T>>())
    }

    fn store_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut Store<T>> {
        self.stores
            .get_mut(&TypeId::of::<T>())
            .and_then(|s| s.as_any_mut().downcast_mut::<Store<T>>())
    }

    fn store_or_new<T: Send + Sync + 'static>(&mut self) -> &mut Store<T> {
        self.stores
            .entry(TypeId::of::<T>())
            .or_insert_with(|| {
                Box::new(Store::<T> {
                    values: BTreeMap::new(),
                    saved_as: None,
                })
            })
            .as_any_mut()
            .downcast_mut::<Store<T>>()
            .expect("store type matches its TypeId")
    }

    /// Drop everything an entity carries (on despawn).
    pub(crate) fn forget(&mut self, id: EntityId) {
        for store in self.stores.values_mut() {
            store.remove(id);
        }
    }

    /// How many entities carry each type, by type name (for verify reports).
    pub(crate) fn counts(&self) -> BTreeMap<&'static str, usize> {
        let mut out = BTreeMap::new();
        for store in self.stores.values() {
            if store.len() > 0 {
                *out.entry(store.type_name()).or_insert(0) += store.len();
            }
        }
        out
    }

    /// Saved component values of the entities `keep` accepts, by saved name.
    pub(crate) fn save(
        &self,
        keep: &dyn Fn(EntityId) -> bool,
    ) -> EngineResult<BTreeMap<String, Vec<(u64, serde_json::Value)>>> {
        let mut out = BTreeMap::new();
        for store in self.stores.values() {
            if let Some(name) = store.saved_name() {
                let entries = store.save(keep)?;
                if !entries.is_empty() {
                    out.insert(name.to_string(), entries);
                }
            }
        }
        Ok(out)
    }

    /// Names in a save that no registered type takes.
    pub(crate) fn unknown_names<'a>(&self, names: impl Iterator<Item = &'a String>) -> Vec<String> {
        names
            .filter(|n| {
                !self
                    .stores
                    .values()
                    .any(|s| s.saved_name() == Some(n.as_str()))
            })
            .cloned()
            .collect()
    }

    pub(crate) fn load(
        &mut self,
        saved: BTreeMap<String, Vec<(u64, serde_json::Value)>>,
        map: &dyn Fn(u64) -> Option<EntityId>,
    ) -> EngineResult<()> {
        for (name, entries) in saved {
            if let Some(store) = self
                .stores
                .values_mut()
                .find(|s| s.saved_name() == Some(name.as_str()))
            {
                store.load(entries, map)?;
            }
        }
        Ok(())
    }
}

impl World {
    /// Put your own data on an entity: `world.insert(orc, Health(100))?`.
    ///
    /// One value per type per entity; inserting again replaces it. Errors
    /// with [`EngineError::UnknownEntity`] when the entity does not exist
    /// (never spawned, or despawned).
    pub fn insert<T: Send + Sync + 'static>(&mut self, id: EntityId, value: T) -> EngineResult<()> {
        if !self.component_owner_exists(id) {
            return Err(EngineError::UnknownEntity);
        }
        self.components.store_or_new::<T>().values.insert(id, value);
        Ok(())
    }

    /// The entity's `T`, if it has one: `world.get::<Health>(orc)`.
    pub fn get<T: Send + Sync + 'static>(&self, id: EntityId) -> Option<&T> {
        self.components.store::<T>()?.values.get(&id)
    }

    /// Change the entity's `T` in place:
    /// `if let Some(h) = world.get_mut::<Health>(orc) { h.0 -= 10; }`.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self, id: EntityId) -> Option<&mut T> {
        self.components.store_mut::<T>()?.values.get_mut(&id)
    }

    /// Whether the entity carries a `T`.
    pub fn has<T: Send + Sync + 'static>(&self, id: EntityId) -> bool {
        self.components
            .store::<T>()
            .is_some_and(|s| s.values.contains_key(&id))
    }

    /// Take the entity's `T` off it (the entity stays).
    pub fn remove<T: Send + Sync + 'static>(&mut self, id: EntityId) -> Option<T> {
        self.components.store_mut::<T>()?.values.remove(&id)
    }

    /// Every entity with a `T`, oldest first.
    pub fn query<T: Send + Sync + 'static>(&self) -> impl Iterator<Item = (EntityId, &T)> + '_ {
        self.components
            .store::<T>()
            .into_iter()
            .flat_map(|s| s.values.iter().map(|(id, v)| (*id, v)))
    }

    /// Every entity with a `T`, to change: `for (_, h) in world.query_mut::<Health>() { h.0 += 1; }`.
    pub fn query_mut<T: Send + Sync + 'static>(
        &mut self,
    ) -> impl Iterator<Item = (EntityId, &mut T)> + '_ {
        self.components
            .store_mut::<T>()
            .into_iter()
            .flat_map(|s| s.values.iter_mut().map(|(id, v)| (*id, v)))
    }

    /// Every entity with both an `A` and a `B`.
    pub fn query2<A: Send + Sync + 'static, B: Send + Sync + 'static>(
        &self,
    ) -> impl Iterator<Item = (EntityId, &A, &B)> + '_ {
        let b = self.components.store::<B>();
        self.query::<A>()
            .filter_map(move |(id, a)| b.and_then(|b| b.values.get(&id)).map(|b| (id, a, b)))
    }

    /// Every entity with both an `A` (to change) and a `B` (to read).
    /// `A` and `B` must be different types.
    pub fn query2_mut<A: Send + Sync + 'static, B: Send + Sync + 'static>(
        &mut self,
    ) -> impl Iterator<Item = (EntityId, &mut A, &B)> + '_ {
        assert_ne!(
            TypeId::of::<A>(),
            TypeId::of::<B>(),
            "query2_mut needs two different component types"
        );
        let (a_store, b_store) = two_stores::<A, B>(&mut self.components.stores);
        a_store.into_iter().flat_map(move |a| {
            a.values.iter_mut().filter_map(move |(id, a)| {
                b_store.and_then(|b| b.values.get(id)).map(|b| (*id, a, b))
            })
        })
    }

    /// The ids of every entity with a `T`, as a list you can loop over while
    /// changing the world (despawning, spawning, inserting).
    pub fn ids_with<T: Send + Sync + 'static>(&self) -> Vec<EntityId> {
        self.query::<T>().map(|(id, _)| id).collect()
    }

    /// Names of the component types an entity carries (for debugging).
    pub fn component_names(&self, id: EntityId) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self
            .components
            .stores
            .values()
            .filter(|s| s.contains(id))
            .map(|s| s.type_name())
            .collect();
        names.sort_unstable();
        names
    }

    /// Let [`World::save`] write every `T` on saved entities (prefab things)
    /// under `name`, and [`World::load`] read them back. Call it at startup,
    /// before loading, like `register_prefab`. The name is what the save
    /// file stores, so keep it when you rename the type.
    pub fn save_component<T: Serialize + DeserializeOwned + Send + Sync + 'static>(
        &mut self,
        name: &str,
    ) {
        let to_json: ToJson<T> = |v| serde_json::to_value(v);
        let from_json: FromJson<T> = |v| serde_json::from_value(v);
        self.components.store_or_new::<T>().saved_as = Some((name.to_string(), to_json, from_json));
    }

    pub(crate) fn component_counts(&self) -> BTreeMap<&'static str, usize> {
        self.components.counts()
    }

    fn component_owner_exists(&self, id: EntityId) -> bool {
        self.contains_entity(id)
            || self.contains_animated(id)
            || self.hierarchy.pivots.contains_key(&id)
    }
}

/// `A`'s store mutably and `B`'s store shared, from the same map.
fn two_stores<A: Send + Sync + 'static, B: Send + Sync + 'static>(
    stores: &mut HashMap<TypeId, Box<dyn AnyStore>>,
) -> (Option<&mut Store<A>>, Option<&Store<B>>) {
    let [a, b] = stores.get_disjoint_mut([&TypeId::of::<A>(), &TypeId::of::<B>()]);
    (
        a.and_then(|s| s.as_any_mut().downcast_mut::<Store<A>>()),
        b.map(|s| &*s)
            .and_then(|s| s.as_any().downcast_ref::<Store<B>>()),
    )
}
