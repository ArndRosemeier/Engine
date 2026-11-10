# Engine guide for coding agents

Read this before writing a game on the engine or changing the engine itself.
It is short on purpose: the [README](README.md) has every feature with a
recipe, and `examples/` has a runnable program for each. The code blocks
below are compiled by `cargo test -p engine`, so they match the real API.

## The model in one minute

- A program is **one closure that runs every frame**:
  `Engine::run(title, |world, frame| { ...; Ok(()) })`. The engine owns the
  window, the GPU and the loop. There is no scene file, no editor and no
  system scheduler: game state lives in your own variables captured by the
  closure (`move |world, frame|`).
- `world: &mut World` is the whole scene: entities, camera, sky, terrain,
  physics, sound, lamps. `frame: &Frame` is this frame's `dt`, `time`,
  `first` (true once, for setup), `input`, `ui` (egui) and window size.
- `world.spawn(mesh)` returns an `EntityId`. Keep the ids you need in your
  own state; move things with `world.set_place(id, Place::new(x, y, z))`,
  remove them with `world.despawn(id)`.
- Game data that belongs to one entity (health, loot, "locked") goes on
  the entity as a component: `world.insert(orc, Health(100))?`,
  `world.get_mut::<Health>(orc)`, `world.query::<Health>()`. It is removed
  when the entity is despawned. No side `HashMap<EntityId, _>` needed.
- Startup choices (window size, vsync, quality, TAA, MSAA, audio, a fixed
  time step) go in one `Settings` value passed to
  `Engine::run_with_settings`; `World` setters change them while running.
  `ENGINE_*` environment variables are debug overrides on top: don't make a
  game depend on them. `world.env_overrides()` and verify reports show
  which were set.
- Every call that can fail returns `EngineResult`; use `?` inside the closure.
  Returning an error ends the program and prints it.
- Units: metres, Y up, yaw in degrees (0 faces +Z, 90 faces +X), colours sRGB
  (`rgb(220, 160, 100)`). `use engine::prelude::*;` brings everything
  everyday into scope, including `glam` maths types (`Vec3`).

## A game skeleton

```rust,no_run
use engine::prelude::*;

/// A component: data that lives on an entity.
struct Coin {
    value: u32,
}

/// Everything the game remembers between frames.
#[derive(Default)]
struct Game {
    player: Option<EntityId>,
    position: Vec3,
    score: u32,
}

fn main() -> EngineResult<()> {
    let mut game = Game::default();
    // Startup settings in one place (Engine::run uses the defaults).
    let settings = Settings::default().window(1280, 720).quality(Quality::High);
    Engine::run_with_settings("my game", settings, move |world, frame| {
        if frame.first {
            // Setup runs once. Spawning here, not every frame.
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.0)));
            world.spawn(Shape::box_at((0.0, -0.05, 0.0), (40.0, 0.1, 40.0), rgb(90, 120, 70))?);
            let player = world.spawn(Shape::box_at((0.0, 0.5, 0.0), (1.0, 1.0, 1.0), rgb(220, 160, 100))?);
            game.player = Some(player);
            // Game data on an entity: any struct of yours.
            // Build the mesh around the origin and place the entity, so the
            // entity's position is where the coin is.
            let coin = world.spawn(Shape::box_at((0.0, 0.0, 0.0), (0.4, 0.4, 0.4), rgb(240, 200, 40))?);
            world.set_place(coin, Place::new(3.0, 0.3, 3.0))?;
            world.insert(coin, Coin { value: 10 })?;
        }

        // Input and movement, scaled by frame.dt so speed does not depend on FPS.
        let mut step = Vec3::ZERO;
        if frame.input.down(Key::W) { step.z += 1.0; }
        if frame.input.down(Key::S) { step.z -= 1.0; }
        if frame.input.down(Key::A) { step.x -= 1.0; }
        if frame.input.down(Key::D) { step.x += 1.0; }
        game.position += step.normalize_or_zero() * 5.0 * frame.dt;
        if let Some(player) = game.player {
            world.set_place(player, Place::new(game.position.x, 0.0, game.position.z))?;
        }
        // Pick up coins within a metre; despawning removes their Coin too.
        for coin in world.ids_with::<Coin>() {
            let at = world.entity(coin)?.transform().w_axis.truncate();
            if at.distance(Vec3::new(game.position.x, 0.3, game.position.z)) < 1.0 {
                game.score += world.get::<Coin>(coin).map_or(0, |c| c.value);
                world.despawn(coin);
            }
        }

        // Camera: orbit, follow or first person, set every frame.
        world.look_follow(game.position, 0.0, 8.0, 4.0);

        // A little UI (egui).
        engine::egui::Window::new("score").show(frame.ui.ctx(), |ui| {
            ui.label(format!("Score: {}", game.score));
        });

        // Self-checks: a verify run fails if one is ever false (see below).
        world.check("player stays on the field", game.position.length() < 20.0);
        world.report("score", game.score);
        Ok(())
    })
}
```

## Check your work: verify runs

Compiling is not enough; run the game and read what happened.

```bash
python tools/verify.py <package>                      # 120 frames, release build
python tools/verify.py <package> --frames 600 --shots 60,300
python tools/verify.py <package> --save-reference refs/<package>.png
python tools/verify.py <package> --reference refs/<package>.png
```

- Exit code 0 means passed. Results are in `target/verify/<package>/`:
  `report.json` (start here), `screenshot.png` (the last frame, look at it)
  and `shot-<N>.png` for `--shots`.
- `report.json` has `passed`, `problems` (why it failed), the game's `checks`
  and `values`, entity, body and lamp counts, the camera, and panics or
  engine errors with their message.
- The run uses a fixed 1/60 s step, so the same program gives the same
  frames every time. Input is empty: drive the game for a check with a demo
  mode (see `ENGINE_PHYSICS_DEMO` in `examples/physics`) or by time.
- Put `world.check(name, ok)` on whatever must hold (the player is above
  the ground, the score never drops, the door opens by second 5) and
  `world.report(key, value)` on what you want to read back. Both are cheap
  in normal runs.
- No display (Linux CI, containers): `tools/verify.py` uses `xvfb-run`; a
  software Vulkan driver (Mesa lavapipe) is enough.

## Rules that save you a debugging session

1. **Spawn in `if frame.first`, not every frame.** Spawning in the frame
   body adds a new entity 60+ times a second. Keep ids and move them.
2. **Many copies of one thing: one spawn, many places.** Use
   `world.spawn_lod_at(model.lod_levels(), &places)` or
   `spawn_instanced` + `set_instances`, not thousands of `spawn` calls.
3. **Generate models once.** Tree, rock and texture generators are slow;
   build them in setup, with `build_all` to use every core.
4. **Use `frame.dt` for motion.** Never assume 60 FPS; examples run with
   vsync off.
5. **Run release builds** (`cargo run --release -p <name>`). Debug builds of
   the renderer and generators are very slow; a game crate should set
   `[profile.dev.package."*"] opt-level = 3`.
6. **Paths are relative to the working directory**, and asset loads refuse
   paths outside it. `tools/verify.py` runs examples from their own folder.
7. **Physics owns a dynamic body's place.** Set the place, then
   `world.add_body(id, Body::dynamic())?`. After that the simulation
   overwrites `set_place`: move it with `world.push(id, impulse)` or
   `set_body_velocity`, or use `Body::kinematic()` for things you move by
   hand (doors, platforms, a sweeper arm).
8. **Read the error.** `EngineError` messages say what was wrong (bad
   number, unknown entity, limit hit). Don't wrap them in `unwrap`.
9. **Loop over a list of ids when the loop changes the world.**
   `for id in world.ids_with::<Health>() { ... world.despawn(id); }`, not
   `for (id, h) in world.query::<Health>()` (that borrows the world).
10. **Build meshes around the origin and place the entity.** A mesh made
    at `(3, 0, 3)` (`Shape::box_at((3.0, 0.0, 3.0), ...)`) still has its
    entity at `(0, 0, 0)`: `transform()` and distance checks see the
    origin. Make it at `(0, 0, 0)`, then `set_place` it.
11. **Ground height comes from the terrain you made.** Use your height
   function (`HeightField::height_at`, your own noise) or
   `world.cast_ray(...)` to place things on the ground.

## Where to find things

| You want | Look at |
|---|---|
| A starting point for a game | `examples/starter` (copy the folder) |
| Terrain you can dig | README "Destruction", `examples/voxel_destruction` |
| An endless streamed world | README "Huge streamed worlds", `examples/open_world` |
| Physics, throwing, explosions | README "Physics", `examples/physics` |
| Trigger zones, layers, overlap queries | README "Triggers, layers and shape queries", `examples/gameplay` |
| A player that walks, jumps, climbs stairs | README "Characters you walk", `examples/gameplay` |
| Doors, ropes, hinges, sliders | README "Joints", `examples/gameplay` |
| Characters, animation, foot IK | README "Character animation", `examples/characters` |
| AI walking around obstacles | README "Navigation", `examples/navigation` |
| Many people | README "Crowds", `examples/crowds` |
| Lights at night | README "Lamps", `examples/lamps` |
| Particles, fire, sparks | README "Particles", `examples/particles` |
| Sound and music | README "Sound", `examples/audio` |
| Save and load | README "Save and load" |
| Models from files (glTF, GLB) | README "Load a model", `examples/model_viewer` |
| Breakable buildings | README "Breakable buildings", `examples/buildings` |
| Speed and quality settings | README "Quality and performance" |
| Debug switches (`ENGINE_*`) | README "Environment variables" |

The API docs are in the code: `cargo doc -p engine --open`, or read the
`///` comments in `engine/src/`. Search for a method with
`grep -rn "pub fn <name>" engine/src`.

## Changing the engine itself

- Work on a branch and open a pull request into `main`. CI must be green:
  `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test -p engine` on Windows and Linux, and the verify runs.
- Raise the version by exactly 0.001 per merge:
  `python tools/bump_version.py "What this brings, in plain words"`.
- A new runnable example goes into `run_examples.py` (`VARIANTS`) and the
  Examples table in the README. Examples call `world.set_vsync(false)`.
- Fix bugs with a test that fails before the fix
  (`engine/src/tests.rs` has physics, world and save tests without a GPU).
  Add a `world.check` to an example when a bug shows only in a running game.
- Rendering changes: compare screenshots before and after
  (`--save-reference` on `main`, `--reference` on your branch) and look at
  them.
- `World`'s methods live in `engine/src/world_<topic>.rs` (spawn,
  physics, hierarchy, animation, lighting, sky, terrain, ...) and the
  renderer's in `engine/src/render/renderer_<topic>.rs` (setup, sync,
  frame, opaque, shadows, ...); `world.rs` and `render/mod.rs` hold the
  types. Search (`grep -rn "pub fn <name>" engine/src`) before reading
  whole files.
