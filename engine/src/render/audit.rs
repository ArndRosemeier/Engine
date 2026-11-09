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

impl Renderer {
    /// Carries out [`World::request_gpu_reupload`]: drops the GPU copies so
    /// the next sync uploads everything again.
    pub(crate) fn reupload_if_asked(&mut self, world: &World) {
        let asked = world.reupload;
        if asked.0 != self.reupload_seen.0 {
            self.gpu_meshes.clear();
            self.gpu_instance_batches.clear();
            self.synced_render_epoch = u64::MAX;
            self.draw_source_epoch = u64::MAX;
            self.terrain_bundle_ids.clear();
            self.terrain_bundle = None;
            self.invalidate_shadow_bundles();
        }
        if asked.1 != self.reupload_seen.1 {
            self.gpu_textures.clear();
            self.gpu_mesh_albedo.clear();
            self.gpu_materials.clear();
            self.synced_resource_epoch = u64::MAX;
            self.terrain_bundle_ids.clear();
            self.terrain_bundle = None;
            self.invalidate_shadow_bundles();
        }
        self.reupload_seen = asked;
    }
}

/// Reads `buffers` (GPU buffer, byte length) back from the GPU.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &[(&wgpu::Buffer, u64)],
) -> Vec<Vec<u8>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-check-copy"),
    });
    let staging: Vec<wgpu::Buffer> = buffers
        .iter()
        .map(|&(src, len)| {
            let len = len.max(4).div_ceil(4) * 4;
            let dst = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-check-readback"),
                size: len,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(src, 0, &dst, 0, len.min(src.size()));
            dst
        })
        .collect();
    queue.submit(std::iter::once(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, b) in staging.iter().enumerate() {
        let tx = tx.clone();
        b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send((i, r.is_ok()));
        });
    }
    drop(tx);
    device
        .poll(wgpu::PollType::Wait)
        .expect("device poll failed");
    let mut ok = vec![false; staging.len()];
    for (i, r) in rx.iter() {
        ok[i] = r;
    }
    staging
        .iter()
        .zip(buffers)
        .zip(ok)
        .map(|((b, &(_, len)), ok)| {
            if !ok {
                return Vec::new();
            }
            let data = b.slice(..).get_mapped_range()[..len as usize].to_vec();
            b.unmap();
            data
        })
        .collect()
}

/// First differing byte and how many differ.
fn diff(gpu: &[u8], cpu: &[u8]) -> Option<(usize, usize)> {
    let mut first = None;
    let mut count = 0;
    for (i, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        if a != b {
            first.get_or_insert(i);
            count += 1;
        }
    }
    if gpu.len() != cpu.len() {
        first.get_or_insert(gpu.len().min(cpu.len()));
        count += gpu.len().abs_diff(cpu.len());
    }
    first.map(|f| (f, count))
}

impl Renderer {
    /// Carries out [`World::request_gpu_check`].
    pub(crate) fn gpu_check_if_asked(&mut self, world: &World) {
        if world.gpu_check == self.gpu_check_seen {
            return;
        }
        self.gpu_check_seen = world.gpu_check;
        let started = std::time::Instant::now();
        let mut out = Vec::new();
        let name = |id: EntityId| {
            world
                .name(id)
                .map_or_else(|| format!("{id}"), |n| format!("{id} ({n})"))
        };
        // The world's own meshes first: what the GPU is given.
        let mut meshes = 0usize;
        for (id, e) in world.entities() {
            let m = &e.mesh;
            if m.index_count() == 0 {
                continue;
            }
            meshes += 1;
            let n = m.vertex_count();
            let lens = [("normals", m.normals.len()), ("colours", m.colors.len())];
            for (what, len) in lens {
                if len != n {
                    out.push(format!("{}: {n} positions but {len} {what}", name(id)));
                }
            }
            if let Some(&big) = m.indices.iter().max() {
                if big as usize >= n {
                    let bad = m.indices.iter().filter(|&&i| i as usize >= n).count();
                    out.push(format!(
                        "{}: {bad} indices point past the {n} vertices (largest {big})",
                        name(id)
                    ));
                }
            }
            if m.indices.len() % 3 != 0 {
                out.push(format!(
                    "{}: {} indices, not whole triangles",
                    name(id),
                    m.indices.len()
                ));
            }
            if let Some(p) = m.positions.iter().find(|p| !p.is_finite()) {
                out.push(format!("{}: vertex position {p:?} is not finite", name(id)));
            }
        }
        // Then the GPU copies, compared byte by byte, a few at a time.
        let ids: Vec<EntityId> = self
            .gpu_meshes
            .keys()
            .copied()
            .filter(|id| world.entity(*id).is_ok_and(|e| e.instance_of.is_none()))
            .collect();
        let (mut compared, mut bytes) = (0usize, 0u64);
        for chunk in ids.chunks(64) {
            let mut bufs = Vec::new();
            for id in chunk {
                let gpu = &self.gpu_meshes[id];
                let e = world.entity(*id).expect("filtered above");
                let vbytes = (e.mesh.to_interleaved().len()
                    * std::mem::size_of::<crate::mesh::Vertex>())
                    as u64;
                let ibytes = (e.mesh.indices.len() * 4) as u64;
                bufs.push((&gpu.vertex_buf, vbytes.min(gpu.vertex_buf.size())));
                bufs.push((&gpu.index_buf, ibytes.min(gpu.index_buf.size())));
            }
            let data = read_back(&self.device, &self.queue, &bufs);
            for (k, id) in chunk.iter().enumerate() {
                let e = world.entity(*id).expect("filtered above");
                let verts = e.mesh.to_interleaved();
                let vcpu: &[u8] = bytemuck::cast_slice(&verts);
                let icpu: &[u8] = bytemuck::cast_slice(&e.mesh.indices);
                let (vg, ig) = (&data[2 * k], &data[2 * k + 1]);
                bytes += (vg.len() + ig.len()) as u64;
                compared += 1;
                if vg.is_empty() || ig.is_empty() {
                    out.push(format!("{}: could not read back", name(*id)));
                    continue;
                }
                let stride = std::mem::size_of::<crate::mesh::Vertex>();
                if let Some((at, count)) = diff(vg, vcpu) {
                    out.push(format!(
                        "{}: VERTICES differ on the GPU: {count} bytes, first at vertex {} byte {} (of {} vertices)",
                        name(*id),
                        at / stride,
                        at % stride,
                        verts.len()
                    ));
                }
                if let Some((at, count)) = diff(ig, icpu) {
                    out.push(format!(
                        "{}: INDICES differ on the GPU: {count} bytes, first at index {} (of {})",
                        name(*id),
                        at / 4,
                        e.mesh.indices.len()
                    ));
                }
            }
        }
        use std::io::Write;
        let path = std::env::var("ENGINE_AUDIT_LOG").unwrap_or_else(|_| "gpu_check.log".into());
        let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        else {
            return;
        };
        let _ = writeln!(
            f,
            "GPU CHECK frame {}: {meshes} world meshes checked, {compared} GPU meshes compared ({:.1} MB) in {:.1} s, {} problems",
            self.audit_frame,
            bytes as f64 / 1e6,
            started.elapsed().as_secs_f32(),
            out.len()
        );
        for line in out.iter().take(200) {
            let _ = writeln!(f, "  {line}");
        }
    }
}
