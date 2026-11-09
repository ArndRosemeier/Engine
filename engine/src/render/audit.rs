//! A self-check of the renderer's mirror of the world (`ENGINE_AUDIT_LOG`).
//!
//! Compares what the world says should be drawn with what the GPU side
//! holds: a mesh for every entity, the same vertex and index counts, the
//! latest transforms, a place in the draw lists, its texture. Each mismatch
//! is one line; an empty list means the mirror is whole.

use super::*;

impl Renderer {
    pub(crate) fn audit(&self, world: &World) -> Vec<String> {
        let mut out = Vec::new();
        let opaque: HashSet<EntityId> = self.source_opaque.iter().copied().collect();
        let terrain: HashSet<EntityId> = self.source_terrain.iter().copied().collect();
        let batched: HashSet<EntityId> = self
            .batch_source_opaque
            .iter()
            .chain(&self.batch_source_terrain)
            .chain(&self.batch_source_transparent)
            .chain(&self.batch_source_water)
            .copied()
            .collect();
        let gpu_batches = world.instance_submit() == InstanceSubmit::GpuIndirect;
        let synced = self.synced_render_epoch == world.render_epoch()
            && self.synced_instance_epoch == world.instance_epoch()
            && self.draw_source_epoch == world.render_epoch();
        if !synced {
            out.push(format!(
                "epochs: render {} (synced {}, sources {}), instance {} (synced {})",
                world.render_epoch(),
                self.synced_render_epoch,
                self.draw_source_epoch,
                world.instance_epoch(),
                self.synced_instance_epoch
            ));
        }
        let name = |id: EntityId| {
            world
                .name(id)
                .map_or_else(|| format!("{id}"), |n| format!("{id} ({n})"))
        };
        for (id, e) in world.entities() {
            if world.is_portal_surface(id) {
                continue;
            }
            if let Some(tid) = e.albedo {
                if !self.gpu_mesh_albedo.contains_key(&tid) {
                    out.push(format!("{}: texture {tid:?} not on the GPU", name(id)));
                }
            }
            if gpu_batches && e.instanced() {
                let proto = e.instance_of.unwrap_or(id);
                if !e.instances.is_empty() && !batched.contains(&proto) {
                    out.push(format!("{}: instanced, batch {proto} not drawn", name(id)));
                }
                continue;
            }
            if e.instance_of.is_some() {
                continue;
            }
            let Some(gpu) = self.gpu_meshes.get(&id) else {
                if e.mesh.index_count() > 0 && !(e.instanced && e.instances.is_empty()) {
                    out.push(format!("{}: no GPU mesh", name(id)));
                }
                continue;
            };
            if gpu.vertex_count != e.mesh.vertex_count() || gpu.index_count != e.mesh.index_count()
            {
                out.push(format!(
                    "{}: GPU mesh {}v/{}i, world {}v/{}i",
                    name(id),
                    gpu.vertex_count,
                    gpu.index_count,
                    e.mesh.vertex_count(),
                    e.mesh.index_count()
                ));
            }
            if gpu.xform_rev != e.xform_rev {
                out.push(format!(
                    "{}: transform rev {} on the GPU, {} in the world",
                    name(id),
                    gpu.xform_rev,
                    e.xform_rev
                ));
            }
            let want = if e.instanced { e.instances.len() } else { 1 };
            if gpu.instance_count != want {
                out.push(format!(
                    "{}: {} instances on the GPU, {want} in the world",
                    name(id),
                    gpu.instance_count
                ));
            }
            if gpu.opaque_index_count > 0 && want > 0 {
                let listed = match e.material {
                    None => opaque.contains(&id),
                    Some(SurfaceMaterialRef::Terrain(_)) => terrain.contains(&id),
                    Some(SurfaceMaterialRef::Water(_)) => true,
                };
                if !listed {
                    out.push(format!("{}: missing from the draw list", name(id)));
                }
            }
            if !e.instanced {
                if let (Some(b), Some(local)) = (gpu.bounds, Bounds::around(&e.mesh.positions)) {
                    let c = e.transform.transform_point3(local.centre);
                    if c.distance(b.centre) > 0.01 + b.radius * 1e-3 {
                        out.push(format!(
                            "{}: GPU bounds at {:?}, mesh at {c:?}",
                            name(id),
                            b.centre
                        ));
                    }
                }
            }
        }
        out
    }
}

impl Renderer {
    /// With `ENGINE_AUDIT_LOG=<file>`, audits about once a second and appends
    /// what it finds (at most 40 lines a time) to the file.
    pub(crate) fn write_audit(&mut self, world: &World) {
        static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let Some(path) = PATH.get_or_init(|| std::env::var("ENGINE_AUDIT_LOG").ok()) else {
            return;
        };
        self.audit_frame += 1;
        if self.audit_frame % 60 != 1 {
            return;
        }
        let found = self.audit(world);
        use std::io::Write;
        let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        else {
            return;
        };
        let entities = world.entities().count();
        let _ = writeln!(
            f,
            "frame {}: {} entities, {} GPU meshes, {} drawn opaque, {} problems",
            self.audit_frame,
            entities,
            self.gpu_meshes.len(),
            self.draw_opaque.len(),
            found.len()
        );
        for line in found.iter().take(40) {
            let _ = writeln!(f, "  {line}");
        }
    }
}
