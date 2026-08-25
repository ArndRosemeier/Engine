# Engine VFX showcase

Gameplay-agnostic procedural VFX demonstration built on the shared Engine API. The showcase combines standalone GPU particles, HDR bloom, world-space ribbons, and lifecycle-owned effect recipes.

## Controls

- `1`-`8`: select Fire, Frost, Lightning, Poison, Root, Hold, Snare, or Charm
- `Q` / `W` / `E` / `R`: select single target, target-centered AOE, caster-centered PBAOE, or cone delivery
- `Space`: spawn the selected effect (up to 12 active effects)

The selected effect is spawned once at startup. Interactive controls are disabled when `VFX_SCREENSHOT_TIME` is set so deterministic captures cannot be altered by input.

The showcase uses a caster and a target dummy so delivery centers, actor size, and cone direction can be inspected without Orrun gameplay or GameData dependencies.

## Deterministic captures

The capture environment variables are:

- `VFX_KIND`: effect kind (`fire`, `frost`, `lightning`, `poison`, `root`, `hold`, `snare`, or `charm`); defaults to `fire`.
- `VFX_DELIVERY`: delivery (`single`/`singletarget`, `aoe`, `pbaoe`, or `cone`); defaults to `single`.
- `VFX_SCREENSHOT_TIME`: simulation time in seconds. When set, the showcase advances at a fixed 60 Hz and captures after reaching this time.
- `ENGINE_SCREENSHOT`: output PNG path. A deterministic showcase capture requires this together with `VFX_SCREENSHOT_TIME`.
- `ENGINE_SCREENSHOT_WAIT=1`: disables the engine's normal frame-number capture so the showcase can queue the screenshot at the requested simulation time.

For the maintained QA matrix, run from the Engine repository root:

```powershell
./examples/vfx_showcase/capture_matrix.ps1
```

The helper sets all five variables for each effect/delivery/time sample, runs `cargo run -q -p vfx_showcase`, and writes PNGs to `target-vfx-qa/polished` by default. Override the destination with `-OutputDirectory`:

```powershell
./examples/vfx_showcase/capture_matrix.ps1 -OutputDirectory target-vfx-qa/custom
```

