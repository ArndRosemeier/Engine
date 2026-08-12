//! Land heightfield chunks from a [`SurfaceSource`].
//!
//! Water is **not** meshed here — games that need water layers implement
//! [`crate::chunk_stream::ChunkBuilder`] directly and emit a
//! [`crate::space::ChunkLayer::Water`] layer alongside the land one.
//!
//! Queueing, budgets, hysteresis, and rebasing live in
//! [`crate::chunk_stream::ChunkStream`]; this module only decides what a land
//! chunk looks like.

use crate::chunk_stream::{ChunkBuilder, ChunkPayload, ChunkStream};
use crate::color::{rgb, Color};
use crate::contact::ContactGrid;
use crate::error::EngineResult;
use crate::mesh::{BuiltMesh, Mesh};
use crate::space::{ChunkCoord, ChunkLayer, ChunkSpan, GlobalXZ};
use crate::surface::{SharedSurface, SurfaceSample, SurfaceSource};
use crate::world::World;
use glam::Vec3;
use rayon::prelude::*;
use std::sync::Arc;

/// Visual knobs for land [`SurfaceTerrain`] meshes.
#[derive(Clone, Debug)]
pub struct SurfaceMeshStyle {
    pub chunk_cells: i32,
    pub cell_size: f32,
    pub grass: Color,
    pub sand: Color,
    pub rock: Color,
    pub bed: Color,
    /// Heights above this (approx) tint as rock.
    pub rock_height: f32,
}

impl Default for SurfaceMeshStyle {
    fn default() -> Self {
        Self {
            chunk_cells: 48,
            cell_size: 4.0,
            grass: rgb(92, 140, 70),
            sand: rgb(194, 178, 128),
            rock: rgb(120, 118, 112),
            bed: rgb(110, 125, 95),
            rock_height: 400.0,
        }
    }
}

/// Builds land chunk meshes from a pluggable surface sample.
#[derive(Clone)]
pub struct SurfaceTerrain {
    source: SharedSurface,
    style: SurfaceMeshStyle,
}

impl SurfaceTerrain {
    pub fn new(source: SharedSurface, style: SurfaceMeshStyle) -> Self {
        Self { source, style }
    }

    pub fn source(&self) -> &dyn SurfaceSource {
        self.source.as_ref()
    }

    pub fn style(&self) -> &SurfaceMeshStyle {
        &self.style
    }

    pub fn sample(&self, x: f32, z: f32) -> SurfaceSample {
        self.source.sample(x, z)
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.sample(x, z).walk_height()
    }

    pub fn chunk_span(&self) -> ChunkSpan {
        ChunkSpan::new(self.style.chunk_cells.max(1) as f64 * self.style.cell_size as f64)
            .expect("surface chunk span")
    }

    pub fn chunk_coord_for(&self, x: f32, z: f32) -> ChunkCoord {
        ChunkCoord::containing(GlobalXZ::at(x as f64, z as f64), self.chunk_span())
    }

    /// Chunk mesh with vertices relative to the chunk's minimum corner.
    pub fn build_chunk(&self, coord: ChunkCoord) -> (Mesh, Vec<f32>) {
        let s = &self.style;
        let cells = s.chunk_cells.max(1);
        let cs = s.cell_size;
        let origin = coord.origin(self.chunk_span());
        let origin_x = origin.x as f32;
        let origin_z = origin.z as f32;
        let verts = (cells + 1) as usize;

        let samples: Vec<SurfaceSample> = (0..verts * verts)
            .into_par_iter()
            .map(|i| {
                let ix = (i % verts) as i32;
                let iz = (i / verts) as i32;
                self.source
                    .sample(origin_x + ix as f32 * cs, origin_z + iz as f32 * cs)
            })
            .collect();

        let mut mesh = Mesh::new();
        let mut ids = Vec::with_capacity(verts * verts);
        let mut heights = Vec::with_capacity(verts * verts);

        for (i, sample) in samples.iter().enumerate() {
            let ix = (i % verts) as i32;
            let iz = (i / verts) as i32;
            // Local coordinates keep f32 mesh precision independent of distance
            // from the world origin.
            let id = mesh
                .add_point(Vec3::new(ix as f32 * cs, sample.ground(), iz as f32 * cs))
                .expect("terrain point");
            let color = match sample.water_top() {
                Some(_) => s.bed,
                None if sample.ground() > s.rock_height => s.rock,
                None => s.grass,
            };
            mesh.set_point_color(id, color).expect("terrain color");
            ids.push(id);
            heights.push(sample.ground());
        }

        for iz in 0..cells {
            for ix in 0..cells {
                let i00 = (iz * (cells + 1) + ix) as usize;
                let i10 = i00 + 1;
                let i01 = i00 + (cells as usize + 1);
                let i11 = i01 + 1;
                mesh.add_quad(ids[i00], ids[i01], ids[i11], ids[i10])
                    .expect("terrain quad");
            }
        }

        (mesh, heights)
    }

    pub fn build_chunk_built(&self, coord: ChunkCoord) -> BuiltMesh {
        self.build_chunk(coord).0.build()
    }
}

impl ChunkBuilder for SurfaceTerrain {
    fn span(&self) -> ChunkSpan {
        self.chunk_span()
    }

    fn build(&self, coord: ChunkCoord) -> EngineResult<Option<ChunkPayload>> {
        let span = self.chunk_span();
        let origin = coord.origin(span);
        let (mesh, heights) = self.build_chunk(coord);
        let contact = ContactGrid::new(
            origin,
            self.style.cell_size as f64,
            (self.style.chunk_cells.max(1) + 1) as usize,
            heights,
        )?;
        Ok(Some(
            ChunkPayload::new(origin.with_height(0.0)?)
                .with_layer(ChunkLayer::Land, mesh.build())?
                .with_contact(contact),
        ))
    }
}

/// Streams [`SurfaceTerrain`] land chunks around a focus.
///
/// A thin adapter over [`ChunkStream`] kept for demos that sample in render
/// space around the world origin.
pub struct SurfaceStream {
    terrain: Arc<SurfaceTerrain>,
    stream: ChunkStream,
}

impl SurfaceStream {
    pub fn new(terrain: SurfaceTerrain, radius: i32) -> Self {
        let terrain = Arc::new(terrain);
        let stream = ChunkStream::new(Arc::clone(&terrain) as Arc<dyn ChunkBuilder>, radius)
            .with_required_radius(0)
            .with_keep_margin(2);
        Self { terrain, stream }
    }

    pub fn with_budgets(mut self, jobs: usize, uploads: usize) -> Self {
        self.stream = self.stream.with_budgets(jobs, uploads);
        self
    }

    pub fn with_keep_margin(mut self, margin: i32) -> Self {
        self.stream = self.stream.with_keep_margin(margin);
        self
    }

    /// Ring that must be resident before the walker may move.
    pub fn with_required_radius(mut self, radius: i32) -> Self {
        self.stream = self.stream.with_required_radius(radius);
        self
    }

    pub fn terrain(&self) -> &SurfaceTerrain {
        &self.terrain
    }

    pub fn stream(&self) -> &ChunkStream {
        &self.stream
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.terrain.height_at(x, z)
    }

    /// Height of the drawn triangle under the walker, when that chunk is resident.
    pub fn contact_height(&self, x: f32, z: f32) -> Option<f32> {
        self.stream.contact_height(GlobalXZ::at(x as f64, z as f64))
    }

    pub fn sample(&self, x: f32, z: f32) -> SurfaceSample {
        self.terrain.sample(x, z)
    }

    pub fn loaded_count(&self) -> usize {
        self.stream.resident_count()
    }

    pub fn pending_count(&self) -> usize {
        self.stream.pending_count()
    }

    pub fn is_loaded(&self, cx: i32, cz: i32) -> bool {
        self.stream.is_resident(ChunkCoord::new(cx, cz))
    }

    pub fn chunk_span(&self) -> f32 {
        self.terrain.chunk_span().metres() as f32
    }

    pub fn sync(&mut self, world: &mut World, focus: Vec3) -> EngineResult<()> {
        self.sync_ex(world, focus, None)
    }

    /// `priority` biases the queue towards where the walker is heading.
    pub fn sync_ex(
        &mut self,
        world: &mut World,
        focus: Vec3,
        priority: Option<Vec3>,
    ) -> EngineResult<()> {
        let focus_xz = GlobalXZ::at(focus.x as f64, focus.z as f64);
        let lead = priority.map(|p| GlobalXZ::at(p.x as f64, p.z as f64));
        self.stream.sync(world, focus_xz, lead)
    }

    /// Build the required ring inline (tests / screenshots / first entry).
    pub fn sync_blocking(&mut self, world: &mut World, focus: Vec3) -> EngineResult<()> {
        let focus_xz = GlobalXZ::at(focus.x as f64, focus.z as f64);
        self.stream.ensure_required_blocking(world, focus_xz)?;
        self.stream.sync(world, focus_xz, None)
    }
}

impl std::fmt::Debug for SurfaceStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceStream")
            .field("stream", &self.stream)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{ChunkId, GlobalXZ};
    use crate::surface::SurfaceSample;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct TestSurface;

    impl SurfaceSource for TestSurface {
        fn sample(&self, x: f32, z: f32) -> SurfaceSample {
            let ground = 20.0 + (x * 0.01).sin() * 4.0 + (z * 0.013).cos() * 3.0;
            if x > 300.0 {
                SurfaceSample::wet(ground - 4.0, ground)
            } else {
                SurfaceSample::dry(ground)
            }
        }
    }

    struct CountingSurface(Arc<AtomicUsize>);

    impl SurfaceSource for CountingSurface {
        fn sample(&self, _x: f32, _z: f32) -> SurfaceSample {
            self.0.fetch_add(1, Ordering::Relaxed);
            SurfaceSample::dry(0.0)
        }
    }

    fn tiny_style() -> SurfaceMeshStyle {
        SurfaceMeshStyle {
            chunk_cells: 4,
            cell_size: 8.0,
            ..SurfaceMeshStyle::default()
        }
    }

    fn make_stream(radius: i32) -> SurfaceStream {
        let terrain = SurfaceTerrain::new(Arc::new(TestSurface), tiny_style());
        SurfaceStream::new(terrain, radius)
    }

    fn pump(stream: &mut SurfaceStream, world: &mut World, focus: Vec3, frames: usize) {
        for _ in 0..frames {
            stream.sync(world, focus).expect("sync");
            if stream.pending_count() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn chunk_vertices_are_local_to_the_chunk_anchor() {
        let terrain = SurfaceTerrain::new(Arc::new(TestSurface), tiny_style());
        let built = terrain.build_chunk_built(ChunkCoord::new(1000, -250));
        let span = terrain.chunk_span().metres() as f32;
        for p in &built.positions {
            assert!(
                p.x >= -0.001 && p.x <= span + 0.001 && p.z >= -0.001 && p.z <= span + 0.001,
                "vertex {p} escaped chunk-local bounds (span {span})"
            );
        }
    }

    #[test]
    fn focus_chunk_is_resident_after_blocking_sync() {
        let mut world = World::new();
        let mut stream = make_stream(1).with_required_radius(1);
        let focus = Vec3::new(70.0, 0.0, 70.0);
        stream.sync_blocking(&mut world, focus).expect("sync");
        let coord = stream.terrain().chunk_coord_for(focus.x, focus.z);
        assert!(stream.is_loaded(coord.x, coord.z));
        assert!(world.has_anchored_chunk(ChunkId::new(coord, ChunkLayer::Land)));
    }

    #[test]
    fn contact_height_matches_the_drawn_surface() {
        let mut world = World::new();
        let mut stream = make_stream(1).with_required_radius(0);
        let focus = Vec3::new(10.0, 0.0, 10.0);
        stream.sync_blocking(&mut world, focus).expect("sync");
        // Exactly on a sample point the triangle and the formula must agree.
        let contact = stream.contact_height(8.0, 8.0).expect("resident contact");
        let direct = stream.sample(8.0, 8.0).ground();
        assert!(
            (contact - direct).abs() < 1e-3,
            "contact {contact} vs surface {direct}"
        );
    }

    #[test]
    fn async_ring_fills_and_unloads_behind_the_walker() {
        let mut world = World::new();
        let mut stream = make_stream(1).with_keep_margin(0).with_budgets(9, 9);
        let near = Vec3::new(16.0, 0.0, 16.0);
        pump(&mut stream, &mut world, near, 400);
        assert!(stream.is_loaded(0, 0));

        let far = Vec3::new(16.0 + 32.0 * 6.0, 0.0, 16.0);
        pump(&mut stream, &mut world, far, 400);
        assert!(
            !stream.is_loaded(0, 0),
            "chunks far behind the walker must unload"
        );
        let coord = stream.terrain().chunk_coord_for(far.x, far.z);
        assert!(stream.is_loaded(coord.x, coord.z));
    }

    #[test]
    fn wet_columns_use_the_bed_tint() {
        let terrain = SurfaceTerrain::new(Arc::new(TestSurface), tiny_style());
        let dry = terrain.sample(0.0, 0.0);
        let wet = terrain.sample(400.0, 0.0);
        assert!(!dry.is_wet());
        assert!(wet.is_wet());
        assert!(wet.water_top().expect("sheet") > wet.ground());
    }

    #[test]
    fn background_jobs_do_the_sampling_work() {
        let samples_bg = Arc::new(AtomicUsize::new(0));
        let terrain = SurfaceTerrain::new(
            Arc::new(CountingSurface(Arc::clone(&samples_bg))),
            tiny_style(),
        );
        let mut world = World::new();
        let mut stream = SurfaceStream::new(terrain, 3).with_budgets(16, 16);
        let focus = Vec3::new(0.0, 0.0, 0.0);
        stream.sync(&mut world, focus).expect("sync");
        for _ in 0..400 {
            stream.sync(&mut world, focus).expect("sync");
            if stream.pending_count() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(samples_bg.load(Ordering::Relaxed) > 0);
        assert_eq!(stream.loaded_count(), 49);
    }

    #[test]
    fn rebasing_keeps_chunks_over_the_same_ground() {
        let mut world = World::new();
        let mut stream = make_stream(1).with_required_radius(1);
        let focus = Vec3::new(10.0, 0.0, 10.0);
        stream.sync_blocking(&mut world, focus).expect("sync");
        let before = world.anchored_chunk_count();
        world
            .set_render_origin(crate::space::RenderOrigin::new(GlobalXZ::at(
                1_000_000.0,
                -500_000.0,
            )))
            .expect("rebase");
        assert_eq!(world.anchored_chunk_count(), before);
        // The chunk under the walker keeps its global identity across a rebase.
        let coord = stream.terrain().chunk_coord_for(focus.x, focus.z);
        assert!(world.has_anchored_chunk(ChunkId::new(coord, ChunkLayer::Land)));
    }
}
