# Engine

Minimal procedural 3D engine (Rust + `wgpu`) with a **friendly, bounded** public API.

Everyday code should only need:

```rust
use engine::prelude::*;

fn main() {
    Engine::run("demo", |world, frame| {
        if frame.first {
            world.spawn(
                Shape::box_at(Vec3::ZERO, Vec3::ONE, rgb(220, 160, 100)).unwrap(),
            );
            world.spawn(Landscape::new(7).area(48.0, 48.0).build().unwrap());
        }
        world.look_orbit(Vec3::ZERO, 20.0, frame.time * 20.0, 25.0);
        world.set_sun(Vec3::new(0.4, 1.0, 0.2), 0.22);
    });
}
```

Power users: `engine::advanced` (`Volume`, `ChunkStreamer`, …).

## Safety

- Bad mesh / color / NaN inputs → `EngineError` (loud, not silent)
- Volume paints, glTF size, and instance counts are capped (`EngineLimits`)
- `Model::load` rejects paths outside an allowed root

## Examples

```bash
cargo run -p hello_mesh
cargo run -p procedural_caves
cargo run -p populate_world
cargo run -p infinite_walker   # WASD move, Q/E turn, Shift sprint
cargo run -p ui_modal          # egui modal + RGBA image overlay
cargo run -p animated_animal   # skinned glTF deer (Idle/Walk/Gallop)
```


## Materials

There are two separate material APIs:

- `SurfaceMaterial` is attached to ordinary meshes, props, cave meshes, and
  authored profiles. Start with `SurfaceMaterial::STONE`, `WOOD`, `DIRT`,
  `GRASS`, `SAND`, or `SNOW`, then use `with_seed`, `with_orientation`, and
  `with_coverage` only when the asset needs variation. `Mesh::new()` already
  carries the valid `SurfaceMaterial::DEFAULT`.
- `TerrainMaterialDesc` is for streamed world-XZ terrain. Create the eight
  generated albedos with `World::create_terrain_albedo`, pass them to
  `TerrainMaterialDesc::from_albedos`, tune the returned descriptor, then call
  `World::create_terrain_material` and `set_default_terrain_material`.

Terrain material texture handles are intentionally all required. A missing or
unknown handle is an error at material creation, rather than a silent fallback.
Terrain UVs are world-space and remain continuous across floating-origin
rebases; mesh UVs on terrain are only the existing dry/moor cover weights.

For directional materials, `with_orientation([x, y, z])` supplies the local
axis (wood grain is the typical use). Leave the default Y axis for isotropic
materials. A zero vector is reserved for an intentionally unspecified axis and
is handled safely by the renderer.
