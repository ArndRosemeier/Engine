# Engine

A procedural 3D engine in Rust on [`wgpu`](https://wgpu.rs): large generated
worlds, destructible terrain and props, and a small, friendly API. You write
one closure that runs every frame; the engine owns the window, the GPU and the
render loop.

![The starter example](docs/starter.jpg)

**What you get**

- **Rendering:** HDR with ACES tone mapping and bloom, 4x MSAA, PBR materials
  (glTF metallic-roughness, normal, occlusion and emissive maps), three shadow
  cascades with soft filtering, SSAO, voxel-probe global illumination, a
  physical sky with day/night, clouds and weather, distance haze.
- **Big worlds:** per-instance LOD with dithered cross-fades, GPU frustum
  culling, an endless GPU clipmap terrain, chunk streaming, a floating origin
  for kilometre-scale coordinates, portals between separate spaces.
- **Generated assets:** ten tree species and shrubs that sway in the wind and
  turn with the seasons, four rock shapes, 61 tileable PBR texture layers
  (ground, cliffs, facades, roofs, paving, medieval), procedural surface
  materials, caves and landscapes.
- **Destruction:** smooth voxel terrain you can dig into (meshed on the GPU),
  and meshes that shatter into solid pieces.
- **Characters and models:** glTF/GLB loading with full materials, skinned
  animation with blending, Actors characters, AssetGenerator `.fmpk` prop packs.
- **Everything else a game needs:** keyboard and mouse input, a fly camera,
  collision and character movement, particles, ribbons and VFX, audio, and
  immediate-mode UI (egui).
- **Self-tuning:** one call picks a quality preset, another keeps a frame-rate
  target by trading detail for speed.
- **Safe by default:** bad input (NaN positions, broken meshes, oversized
  files) is an `EngineError`, never a silent glitch or a crash.

## Getting started

Run the starter scene, a landscape with a wood you can fly through:

```bash
cargo run --release -p starter
```

Right mouse + move looks around, WASD flies, Space/Ctrl go up and down, Shift
is fast, Esc quits.

### Your own project

Add the engine as a dependency. Optimising dependencies keeps debug builds
playable (the generators and the renderer are slow without optimisation):

```toml
[dependencies]
engine = { git = "https://github.com/ArndRosemeier/Engine" }

[profile.dev.package."*"]
opt-level = 3
```

The smallest program spawns a box and looks at it:

```rust
use engine::prelude::*;

fn main() -> EngineResult<()> {
    Engine::run("hello", |world, frame| {
        if frame.first {
            world.spawn(Shape::box_at((0.0, 0.5, 0.0), (1.0, 1.0, 1.0), rgb(220, 160, 100))?);
            world.spawn(Shape::box_at((0.0, -0.05, 0.0), (20.0, 0.1, 20.0), rgb(90, 120, 70))?);
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
        }
        world.look_orbit((0.0, 0.5, 0.0), 6.0, frame.time * 20.0, 25.0);
        Ok(())
    })
}
```

The quickest real start is to copy [`examples/starter`](examples/starter) into
your project: it is about a hundred lines and shows terrain, scattering, LOD,
the sky, quality settings and the fly camera together.

## How it fits together

| Idea | What it is |
|---|---|
| `Engine::run(title, update)` | Opens the window and calls `update(world, frame)` once per frame. Return an error to stop. |
| `World` | Everything in the scene: entities, camera, lights, sky, terrain, settings. |
| `Frame` | This frame's `dt`, `time`, `fps`, window size, `first` (true once, for setup), `input` and `ui`. |
| `Mesh` | Geometry you build (`add_box`, `add_point`, `add_triangle`, ...) or get from a generator or `Model::load`. |
| `EntityId` | What `world.spawn(mesh)` returns: move it with `set_place`, remove it with `despawn`. |
| `Place` | Position, yaw, pitch, scale and tint of one entity or instance. |
| `LodLevel` / `LodId` | Detail levels of one model, drawn per instance by camera distance (`spawn_lod_at`). |
| `SpaceId` / `Portal` | Separate pockets of the world (a house interior) joined by portals. |
| `EngineResult` | Every fallible call returns one; use `?` inside the closure. |

Coordinates are metres, Y is up. Yaw is in degrees, 0 faces +Z and 90 faces +X.
Colours are sRGB (`rgb(220, 160, 100)`).

Import `engine::prelude::*` for everyday work. `engine::advanced` has the power
tools (`Volume`, `ChunkStreamer`, custom GPU fields).

### Common recipes

**Put a forest on the ground:** generate a model, scatter places, spawn with LOD.

```rust
let ground = HeightField::new(demo_terrain_rules());
let places = Scatter::new(1)
    .disc(0.0, 0.0, 300.0)       // where
    .spacing(10.0)               // how dense
    .density(0.6)                // clumps and clearings
    .scale_range(0.8, 1.2)
    .on_ground(|x, z| ground.height_at(x, z))
    .only_where(|x, z| !ground.sample(x, z).water)
    .places()?;
for (i, mine) in Scatter::split(&places, 4, 1).iter().enumerate() {
    let tree = tree_model(TreeSpecies::Oak, i as u32)?;
    world.spawn_lod_at(tree.lod_levels(), mine)?;
}
```

**Load a model:** `world.spawn_model(&Model::load_scene("assets/house.glb")?, Place::new(0.0, 0.0, 0.0))?`
spawns a glTF/GLB with all its materials (`Model::load` merges it into one
`Mesh` instead). Paths must lie under the working directory; `load_scene_with`
takes another root. Animated characters: `AnimatedModel::load`, then
`world.spawn_animated(model, place)` and `play_animation(id, "Walk")`.

**Texture anything:** `texture_layer(NaturalLayer::RockFace).apply_triplanar(&mut mesh)?`
wraps a mesh in a generated PBR layer without UVs.

**Light and atmosphere:** `world.set_physical_sky(Some(PhysicalSky::at_hour(18.5)))`
drives sun, ambient light, fog and exposure. `set_wind(0..2)` and
`set_season(0..1)` move and colour the vegetation.

**Destruction:** `world.set_voxel_terrain(Some(VoxelTerrain::new(volume)))`
gives diggable ground (see `examples/voxel_destruction`);
`world.shatter(entity, impact_point, Shatter::default())?` breaks a mesh into
solid pieces.

**Input:** `frame.input.down(Key::W)`, `pressed(Key::Space)` (this frame only),
`mouse_down(MouseButton::Left)`, `move_xz()` for WASD, `mouse_delta()`.
`FlyCamera` does all of it for a free camera.

**UI:** `frame.ui` is an egui frame: panels, buttons, images, modals.

### Quality and performance

```rust
world.set_quality(Quality::Medium);   // Low, Medium, High (default), Ultra
world.set_target_fps(Some(60.0))?;    // keep 60 FPS: pull LOD in, then drop GI, then AO
world.set_vsync(false);               // uncapped frame rate (title bar shows FPS)
```

Finer knobs: `set_shadows`, `set_ambient_occlusion`, `set_global_illumination`,
`set_bloom`, `set_haze`, `set_lod_bias`, `set_view_distance`.

## Generators

| Module | Makes |
|---|---|
| `vegetation` | `tree_model(species, variant)`, `shrub_model(variant)`: three LOD meshes, wind, seasons |
| `rocks` | `rock_model(shape, seed)`: boulders, blocks, slabs, spires with LOD |
| `texgen` | `texture_layer(layer)`: 61 tileable albedo/normal/roughness layers |
| `scatter` | `Scatter`: deterministic places with clumping, ground snapping, filters |
| `proc_terrain` | `ProcTerrain::gpu_clipmap`: endless terrain with lakes, painted with texgen layers (`with_layers(Some(TerrainLayers { .. }))`), `HeightField` to query it |
| `landscape` | `Landscape::new(seed)`: hills and caves as a mesh |
| `voxel_terrain` | Smooth, diggable voxel ground meshed on the GPU |
| `fracture` / `debris` | Shatter meshes into solid chunks |
| `proc` | `Noise`, carving helpers |
| `props` | AssetGenerator `.fmpk` prop packs |

## Examples

```bash
cargo run --release -p <name>
```

| Example | Shows |
|---|---|
| `starter` | The template: terrain, wood, rocks, sky, quality presets, fly camera |
| `hello_mesh` | Building a mesh by hand (a little house) |
| `vegetation` | All tree species, shrubs and rocks in the wind |
| `lod_forest` | 12 000 trees with per-instance LOD |
| `sky_day` | A day passing under the physical sky |
| `texgen` | The texture layers (`ENGINE_TEXGEN_FAMILY=natural/ground/facade/roof/medieval`) |
| `voxel_destruction` | Digging terrain and shattering towers |
| `model_viewer` | Any glTF/GLB or `.fmpk` file: `-- path/to/model.glb` |
| `animated_animal` | Skinned glTF animation |
| `infinite_walker` | Endless clipmap terrain with a walker |
| `portals`, `portal_pair` | Spaces joined by portals |
| `material_showcase` | Procedural surface materials |
| `vfx_showcase` | Particles, ribbons and spell effects |
| `procedural_caves`, `populate_world`, `gpu_field_demo`, `ui_modal` | Caves, prop scattering, GPU density fields, UI |

`run_examples.py` (or `run_examples.bat` on Windows) opens a small launcher.

## Environment variables

Every program built on the engine understands these:

| Variable | Effect |
|---|---|
| `ENGINE_WIDTH`, `ENGINE_HEIGHT` | Window size (default 1920 x 1080) |
| `ENGINE_SCREENSHOT=out.png` | Save frame `ENGINE_SCREENSHOT_FRAME` (default 3) to a PNG and exit |
| `ENGINE_SCREENSHOT_WAIT=1` | Wait for `world.mark_ready()` before counting screenshot frames |
| `ENGINE_MSAA=1` | Turn MSAA off (default 4) |
| `ENGINE_PRESENT_MODE` | `fifo` (default), `fifo-relaxed`, `mailbox` or `immediate` |
| `ENGINE_GPU_PROFILE_EVERY=N` | Print GPU pass timings every N frames |
| `ENGINE_HITCH_MS` | Log frames slower than this (default 33) |

The examples add their own (listed at the top of each `main.rs`).

## Testing

```bash
cargo test -p engine                        # unit tests, no GPU needed
ENGINE_GPU_TESTS=1 cargo test -p engine     # also the GPU checks
cargo clippy --workspace --all-targets
```

Warnings are errors in this workspace. Screenshots work headless (for example
under Xvfb with a software Vulkan driver), which is how renders are checked
without a GPU.

## Materials in depth

There are two material systems:

- `SurfaceMaterial` is attached to ordinary meshes, props, cave meshes and
  authored profiles. Start with `SurfaceMaterial::STONE`, `WOOD`, `DIRT`,
  `GRASS`, `SAND` or `SNOW`, then use `with_seed`, `with_orientation` and
  `with_coverage` when the asset needs variation. `Mesh::new()` already
  carries `SurfaceMaterial::DEFAULT`. Texture maps (`MaterialMaps`) add albedo,
  normal, occlusion-roughness-metallic and emissive maps, by UV or triplanar.
- `TerrainMaterialDesc` is for streamed world-XZ terrain. Create the eight
  generated albedos with `World::create_terrain_albedo`, pass them to
  `TerrainMaterialDesc::from_albedos`, tune the descriptor, then call
  `World::create_terrain_material` and `set_default_terrain_material`. All
  handles are required: a missing one is an error, not a silent fallback.

For directional materials, `with_orientation([x, y, z])` gives the local axis
(wood grain is the usual case).

## Safety

- Bad mesh, colour or NaN input returns `EngineError`.
- Volume paints, glTF size and instance counts are capped (`EngineLimits`,
  adjustable with `Engine::run_with`).
- `Model::load` rejects paths outside an allowed root.
