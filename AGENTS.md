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
  Returning an error stops the game and prints `error: ...` to stderr at
  once. The window then shows the error until it is closed (Escape), and
  the program exits with code 1. A verify run stops right away.
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
    yaw: f32,
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
            // The ground: a mesh plus a fixed body, so the player stands on it.
            let ground = world.spawn(Shape::box_at((0.0, 0.0, 0.0), (40.0, 0.2, 40.0), rgb(90, 120, 70))?);
            world.set_place(ground, Place::new(0.0, -0.1, 0.0))?;
            world.add_body(ground, Body::fixed().shape(BodyShape::Box(Vec3::new(20.0, 0.1, 20.0))))?;
            // The player: a character controller walks it, slides along walls,
            // climbs steps and falls. Its place is its feet.
            let player = world.spawn(Shape::box_at((0.0, 0.9, 0.0), (0.6, 1.8, 0.6), rgb(220, 160, 100))?);
            world.add_character(player, Character::default())?;
            game.player = Some(player);
            // A coin: mesh made around the origin, then placed. A trigger on
            // it reports when something walks in; the Coin component is game data.
            let coin = world.spawn(Shape::box_at((0.0, 0.0, 0.0), (0.4, 0.4, 0.4), rgb(240, 200, 40))?);
            world.set_place(coin, Place::new(3.0, 0.5, 3.0))?;
            world.add_trigger(coin, Trigger::ball(0.6))?;
            world.insert(coin, Coin { value: 10 })?;
        }
        let Some(player) = game.player else { return Ok(()) };

        // Input: A/D turn, W/S walk (move_dir_xz matches the follow camera),
        // Space jumps. walk takes a speed in m/s; the engine applies frame.dt.
        game.yaw += frame.input.yaw_sign() * 90.0 * frame.dt;
        world.walk(player, frame.input.move_dir_xz(game.yaw) * 5.0)?;
        if frame.input.pressed(Key::Space) {
            world.jump(player, 5.0)?;
        }

        // Coins: picked up when the player enters their trigger.
        for e in world.take_trigger_events() {
            if e.entered && e.other == player {
                if let Some(coin) = world.get::<Coin>(e.trigger) {
                    game.score += coin.value;
                    world.despawn(e.trigger); // removes its Coin and trigger too
                }
            }
        }

        // Camera: orbit, follow or first person, set every frame.
        let feet = world.world_transform(player)?.w_axis.truncate();
        world.look_follow(feet, game.yaw, 6.0, 3.0);

        // HUD: text, bars and menus in corners (README "Game UI"; full egui: frame.ui.ctx()).
        frame.ui.text(Anchor::TopLeft, format!("Score: {}", game.score));

        // Self-checks: a verify run fails if one is ever false (see below).
        world.check("player stays above the ground", feet.y > -1.0);
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

## Recipes for common game pieces

Each function is compiled with the guide, so the calls are right. Call the
`setup_*` ones inside `if frame.first`, the others every frame.

```rust,no_run
use engine::prelude::*;

/// HUD: lines stack in each corner; bars, a crosshair, a pause menu.
fn hud(frame: &Frame, health: f32, paused: &mut bool, world: &mut World) {
    frame.ui.text(Anchor::TopLeft, "Find the key");
    frame.ui.bar(Anchor::TopLeft, "Health", health, 100.0, rgb(200, 60, 60));
    frame.ui.text_styled(Anchor::Bottom, "WASD walk, Space jump", 16.0, rgb(230, 230, 230));
    frame.ui.crosshair();
    if frame.input.pressed(Key::P) {
        *paused = !*paused;
    }
    if *paused {
        match frame.ui.menu("Paused", &["Resume", "Quit"]) {
            Some(0) => *paused = false,
            Some(1) => world.request_exit(),
            _ => {}
        }
    }
}

/// Text in the world: a name that follows an entity, a number that rises and fades.
fn setup_names(world: &mut World, orc: EntityId) -> EngineResult<()> {
    world.set_label(orc, Label::new("Orc").size(16.0).color(rgb(255, 120, 90)))
}
fn show_damage(world: &mut World, at: Vec3, damage: u32) {
    world.float_text(at + Vec3::Y * 2.0, format!("-{damage}"), rgb(255, 80, 60));
}

/// A zone: an invisible box that reports who walks in and out.
fn setup_zone(world: &mut World) -> EngineResult<EntityId> {
    world.spawn_trigger(Place::new(0.0, 1.0, 10.0), Trigger::cuboid((2.0, 1.0, 2.0)))
}
fn zone_events(world: &mut World, zone: EntityId, player: EntityId) -> bool {
    let mut entered = false;
    for e in world.take_trigger_events() {
        entered |= e.trigger == zone && e.other == player && e.entered;
    }
    entered
}

/// A door on a hinge that a motor swings open, and a crate that physics owns.
fn setup_door_and_crate(world: &mut World) -> EngineResult<()> {
    let door = world.spawn(Shape::box_at((0.0, 0.0, 0.0), (1.0, 2.0, 0.1), rgb(120, 80, 50))?);
    world.set_place(door, Place::new(5.0, 1.0, 0.0))?;
    world.add_body(door, Body::dynamic().shape(BodyShape::Box(Vec3::new(0.5, 1.0, 0.05))))?;
    world.pin(door, Joint::hinge(Vec3::Y).at((4.5, 1.0, 0.0)).limits(0.0, 100.0).motor(60.0, 50.0))?;

    let crate_box = world.spawn(Shape::box_at((0.0, 0.0, 0.0), (0.8, 0.8, 0.8), rgb(150, 110, 70))?);
    world.set_place(crate_box, Place::new(0.0, 3.0, 2.0))?; // place first, then the body
    world.add_body(crate_box, Body::dynamic())?;
    world.push(crate_box, Vec3::new(0.0, 2.0, 1.0))?; // from now on, move it by pushing
    Ok(())
}

/// A sound where something happened.
fn bang(world: &mut World, at: Vec3) {
    world.play_sound(&Sound::stock(Stock::Explosion, 0), PlaySound::at(at));
}
# fn main() {}
```

For animated characters (clips chosen by rules on speed, flags and
triggers, with footstep events) see `examples/anim_states`; for custom
shader looks see the next section.

## A custom material

A surface's look can be a small WGSL function. It runs inside the engine's
lit shader, so the result is still lit, shadowed and fogged. The code is
checked by `add_material`: a WGSL mistake is an error naming the line.

```rust,no_run
use engine::prelude::*;

const GLOW: &str = r#"
fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
    var o = base; // what the mesh looks like without the material
    let pulse = 0.5 + 0.5 * sin(m.time * 3.0 + m.world_pos.y * 4.0);
    o.emission = vec3<f32>(0.2, 0.8, 1.0) * pulse * m.params[0].x;
    return o;
}
"#;

fn setup(world: &mut World) -> EngineResult<CustomMaterialId> {
    let glow = world.add_material(CustomMaterial::new("glow", GLOW).param(0, [1.0, 0.0, 0.0, 0.0]))?;
    let pillar = world.spawn(Shape::box_at((0.0, 1.0, 0.0), (0.5, 2.0, 0.5), rgb(40, 40, 50))?);
    world.set_custom_material(pillar, glow)?;
    Ok(glow)
}

fn every_frame(world: &mut World, glow: CustomMaterialId, power: f32) -> EngineResult<()> {
    // params[0..4] are four vec4s the game may change any frame.
    world.set_material_param(glow, 0, [power, 0.0, 0.0, 0.0])
}
# fn main() {}
```

`m` gives `world_pos`, `object_pos`, `normal`, `view_dir`, `uv`, `time`
and `params`; return `color` (alpha used with `.see_through()`), `normal`,
`roughness`, `metallic` and `emission`. The full list is on
`engine::custom_material::MATERIAL_API`.

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
| HUD text, health bars, menus, crosshair | README "Game UI (HUD)", `examples/gameplay` |
| Names over things, damage numbers | README "Text in the world", `examples/gameplay` |
| Animation states (idle/walk/attack rules, footstep events) | README "Animation state machines", `examples/anim_states` |
| Glass and other see-through things | README "See-through things", `examples/glass` |
| Lava, holograms, force fields: your own shader code | README "Custom materials", `examples/materials`, and below |
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
- A new runnable example goes into the workspace `Cargo.toml`, the README
  Examples table, the "Where to find things" table above and, when it can
  check itself, a verify step in `.github/workflows/ci.yml`; variants with
  environment settings go into `run_examples.py` (`VARIANTS`). Examples call
  `world.set_vsync(false)`.
- A new feature gets a README section and, if agents will reach for it, a
  compiled snippet here (this file is a doctest).
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
