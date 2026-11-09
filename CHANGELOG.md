# Engine changelog

Newest version first. Every merge into `main` adds exactly one entry and
raises the version by 0.001 (see "Versioning" in the README). The engine
reads this file at build time: the top entry is `engine::version()`, shown
in every window title and in the "What's new" button.

Add an entry with `python tools/bump_version.py "What this merge brings"`.

## 0.013 (2026-11-09)

- Engine's frame work runs on all cores: animation (clip sampling, foot IK and its ground search), recording the shadow maps (one command buffer per light view) and recording the world's opaque meshes (render bundles in chunks of 128, recorded side by side) run on a frame-only thread pool that background jobs never use, so frame work never waits behind them. Each step finishes completely before the next starts, and command buffers are submitted in a fixed order, so the picture is the same as before.
- Bundles that would come out the same are kept from frame to frame: a shadow view re-records only the group of 128 casters that changed, and the world pass only the chunks whose meshes, buffers or textures changed.
- The scene goes to the GPU before Engine waits for the window's next image, so the GPU starts on it while the display is still busy.
- Foot IK no longer copies every placement of the town each frame, and turns far placements away with one box test (its ground search went from about 3 ms to 1.4 ms for nine characters in HeroWorld's town).
- ENGINE_CPU_TRACE shows the new steps: sync.poses, shadows.select, shadows.bundles, world.record_opaque.


## 0.012 (2026-11-09)

- Characters share their textures. On the GPU, maps with the same texels (a garment, eyes, teeth or hair worn by many characters) are one texture, and a crowd's far model is drawn with its near model's maps instead of holding its own copy, on the CPU and the GPU.
- AlbedoMap's texels are now MapBytes: cloning a map shares its bytes, and writing to a shared copy copies it first. It reads like a `[u8]`; make one from a `Vec<u8>` with `.into()`, and get the `Vec` back with `into_vec()`. `is_shared` and `content_hash` help a game share maps it makes from shared ones. This is an API change for code that builds maps.
- ENGINE_FRAME_STATS' gpu memory line shows characters' buffers and maps.
- Measured on HeroWorld's 41 start-up characters: their maps took 492 MB, of which 239 MB were distinct. The crowds' far models held another ~330 MB of copies.

## 0.011 (2026-11-09)

- Occlusion culling: world.set_occlusion_culling(true) stops drawing meshes and instanced props hidden behind others, such as every block behind a street wall. The GPU tests each mesh's box against this frame's own depth, so nothing pops in; in the new occlusion_city example 93% of 2 000 towers are skipped at street level and the GPU frame drops by about 30%. world.occlusion_stats() tells how many were tested and hidden; ENGINE_OCCLUSION=0/1 overrides the setting, ENGINE_OCCLUSION_CHECK=1 checks the GPU's answers against a CPU copy. Shadow maps still draw every caster.

## 0.010 (2026-11-09)

- Less RAM for big worlds: World::set_cpu_meshes(CpuMeshes::Release) drops each mesh's CPU copy once the renderer has uploaded it (a town's blocks were held twice, in RAM and on the graphics card). Feet still stand on released meshes (their upward faces are kept), openings still work; World::keep_cpu_mesh keeps the ones you will shatter or give a body. The default stays CpuMeshes::Keep, so nothing changes until a game asks. World::cpu_mesh_bytes, ground_mesh_bytes and cpu_mesh_counts show what is held.
- What feet stand on takes far less memory: a mesh's ground grid now has about one cell per two upward faces instead of up to 256 x 256 cells (a town's worth went from 870 MB to 110 MB).

## 0.009 (2026-11-09)

- Feet stand on what is drawn: foot IK now sets each sole on the highest surface under it, any mesh's upward faces included (not only the terrain and physics bodies), so characters no longer sink into roads, floors or stairs that have no collider. It also corrects a character whose origin is a few centimetres off the ground (a game server's simpler floor) or whose clip has the soles under its own floor; the hips follow the lower foot up or down, and feet keep their tilt while knees bend, so toes and heels stay out of the ground.
- World::ground_below(p, depth) gives the highest drawn surface under a point; World::set_walkable(id, false) leaves grass blades, glass or effects out; World::sole_points(id) gives both soles for checks. Crowd characters close to the camera get foot IK.

## 0.008 (2026-11-09)

- The window stays hidden while the renderer compiles its pipelines at start and appears once it can answer, so it no longer shows up frozen (not responding, cannot be dragged) for the first seconds.

## 0.007 (2026-11-09)

- Spawning big textured meshes no longer stalls a frame: Mesh::prepare does the slow part of a spawn (triangulating, normals, the texture's sharing key) on any thread, and World::spawn_prepared / place_prepared take the result in at once. A town block of 40 facade-textured meshes went from about 60 ms on the main thread to almost nothing.
- Hitch logs say where Engine's own time went: a hitch block now lists the engine steps over a millisecond (stream, LODs, crowds, physics, mesh sync, shadows, world, water, post, present...), slowest first, so a game's hitch log shows whether its stutter was its own update, an upload or drawing.

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
