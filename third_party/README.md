# Patched dependencies

## wgpu-25.0.2

wgpu 25.0.2 from crates.io with one fix: `RenderBundle` now releases its
bundle when dropped (`src/backend/wgpu_core.rs`, look for "Engine patch").
Unpatched, every render bundle the engine records stays in memory for the
life of the device, so scenes that re-record bundles slow down and grow by
hundreds of megabytes a minute. Newer wgpu versions hold bundles by `Arc`
and do not need this. Remove the `[patch.crates-io]` entry in the root
`Cargo.toml` and this folder when the engine moves to such a version.

## wgpu-core-25.0.2

wgpu-core 25.0.2 from crates.io with one change: binding a bind group (or
executing a render bundle) no longer walks every texture it holds to check
that each is initialized, once all of them were found initialized and no
sampled or storage texture was discarded since. Unpatched, each bind of the
material table checks thousands of textures and their init-state locks, about
1.4 ms a frame of the main thread in HeroWorld's city on an RTX 5070. Look
for "Engine patch" in `src/command/memory_init.rs` (the cache and the discard
epoch), `src/command/render.rs`, `src/command/compute.rs`,
`src/command/bundle.rs`, `src/binding_model.rs` and `src/device/resource.rs`.
Games that patch wgpu to Engine's copy must patch `wgpu-core` the same way.
