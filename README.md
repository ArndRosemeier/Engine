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
