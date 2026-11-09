# Engine changelog

Newest version first. Every merge into `main` adds exactly one entry and
raises the version by 0.001 (see "Versioning" in the README). The engine
reads this file at build time: the top entry is `engine::version()`, shown
in every window title and in the "What's new" button.

Add an entry with `python tools/bump_version.py "What this merge brings"`.

## 0.006 (2026-11-09)

- Fixed: egui textures are freed only after the frame that drew them is submitted, and ENGINE_FRAME_STATS shows video memory.
- Drawing checks for hunting render bugs: World::set_debug_shading paints or switches off parts of the shader (DEBUG_SHADING_FLAGS: colour by depth, red where depth does not match position, hide far things, prepass and depth-write switches, no decals/GI/reflections/lamps, and more).
- Renderer self-checks: ENGINE_AUDIT_LOG compares the world with its GPU copy once a second and logs the adapter, driver and every resize; World::request_gpu_check reads every mesh back from the GPU and compares it byte by byte; request_gpu_reupload and request_renderer_rebuild make GPU copies and window-sized buffers anew.
- Old drivers: World::gpu_info names the graphics card and driver, and GpuInfo::driver_warning flags NVIDIA drivers below 550 (528.76 drew far things over near ones at random on an RTX 4060 laptop; 621.14 is fine). Test switches: ENGINE_DEPTH_FORMAT=d32, WGPU_BACKEND=dx12 with ENGINE_DXC_DIR or WGPU_DX12_COMPILER=dxc.

## 0.005 (2026-11-09)

- Lights that switch with the hour: emissive surfaces sit on one of 15 glow channels (SurfaceMaterial::on_glow_channel) and World::set_glow_channel dims, brightens or recolours a whole channel at once, for windows that light up at dusk without rebuilding a mesh.
- Breakable buildings keep their textures: BuildingPlan::add_texture and add_parts give elements textured meshes (one draw call per texture, GPU textures shared between buildings), and add_all plans big towers in linear time.
- Machines of moving parts with LODs: World::attach_lod hangs a LOD set on a pivot so it follows the part, picks its level by distance and shares meshes and maps with every copy (spawn_lod_like). Mesh::simplified makes the lighter levels.
- Fog volumes can lay their grade on top of a changing base grade (FogVolume::grade_on_top). A collapse that starts above the ground floor keeps its rubble on the roof, with a skirt poured down the sides. The sun's mirror highlight on chrome is capped, so shiny parts seen from above no longer flare white with black specks.

## 0.004 (2026-11-08)

- Animated morph targets on skinned characters: faces can smile, blink and talk. Give AnimatedModel morph_targets and each SkinMesh its morphs, then set weights every frame with World::set_morph_weight or set_morph_weights (up to 64 per model, blended on the GPU before skinning, shadows included).

## 0.003 (2026-11-07)

- Sound: sounds placed in the world come from their direction, fade and dull with distance, are muffled behind walls and shift pitch when they move (World::play_sound, PlaySound::at / following / ui / ambient).
- Reverb follows the space the camera is in: open air, rooms and halls (from Rooms), caves, and muffled under water. Music plays in layers that fade in with World::set_music_intensity.
- No audio files needed: recorded explosions, crumbling, digging, knocks, splashes and loops for fire, stream, river, surf and wind are built in (Sound::stock), more are made in code (Sound::explosion, fire_loop, ...), and MusicTrack::procedural writes music. Explosions, digging, shattering, fires, splashes, impacts and flowing water make their own sounds (World::set_auto_sounds). New example: audio.
- The example launcher (run_examples) is a searchable, scrolling list now: pick an example to read what it does and start it or one of its variants.

## 0.002 (2026-11-07)

- Shaders never compile mid-game: every pipeline is built before the first frame, on all CPU cores at once (startup pipeline building about 2.7x faster on 4 cores). The clipmap ground and the GPU water simulation build on worker threads at launch instead of on the frame they first appear.
- On Vulkan the compiled pipelines are saved to disk, so later launches skip the driver compile. ENGINE_LOG_PIPELINES=1 shows the timings; ENGINE_PIPELINE_CACHE=0 turns the cache off.

## 0.001 (2026-11-07)

- The engine has a version number now. It starts here at 0.001 and goes up
  by 0.001 with every merge into main.
- Every window title shows the version, and a "What's new" button in the
  bottom right corner opens this list (`world.set_whats_new_button(false)`
  hides it).
- Everything merged before this point (pull requests #1 to #64) counts as
  0.001: renderer stages 1 to 3, glTF and Actors characters, LOD, TAA, AgX
  and bloom, grass and vegetation, water, Rapier physics, lamps, crowds,
  decals, fog volumes, moons, interiors, reflections, buildings, navigation,
  the z-fighting finder, world streaming, GPU particles and destruction
  effects (pull request #64 included).
