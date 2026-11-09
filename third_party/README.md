# Patched dependencies

## wgpu-25.0.2

wgpu 25.0.2 from crates.io with one fix: `RenderBundle` now releases its
bundle when dropped (`src/backend/wgpu_core.rs`, look for "Engine patch").
Unpatched, every render bundle the engine records stays in memory for the
life of the device, so scenes that re-record bundles slow down and grow by
hundreds of megabytes a minute. Newer wgpu versions hold bundles by `Arc`
and do not need this. Remove the `[patch.crates-io]` entry in the root
`Cargo.toml` and this folder when the engine moves to such a version.
