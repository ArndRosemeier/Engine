# Engine changelog

Newest version first. Every merge into `main` adds exactly one entry and
raises the version by 0.001 (see "Versioning" in the README). The engine
reads this file at build time: the top entry is `engine::version()`, shown
in every window title and in the "What's new" button.

Add an entry with `python tools/bump_version.py "What this merge brings"`.

## 0.070 (2026-11-14)

- Water that follows the camera is drawn in the right place: a river no longer shows as patches over a dry bed after the window moved. Worlds with costly ground heights build faster.

## 0.069 (2026-11-14)

- Streamed worlds can have dry ground and snow: a generator's ground_cover says how green each place is and where the snow line lies

## 0.068 (2026-11-14)

- World::set_fullscreen / fullscreen: a game can toggle borderless fullscreen (e.g. on Alt+Enter)

## 0.067 (2026-11-12)

- Switching city size no longer crashes: instance batches of despawned things are dropped (new batch_churn example checks it)

## 0.066 (2026-11-12)

- Saved static shadows survive a streamed mesh: a mesh getting new buffers no longer redraws all three cascades' static layers

## 0.065 (2026-11-12)

- A job beside the frame (engine::parallel::beside_frame) and background_thread for a game's own workers; bounced light no longer rebuilds its voxels on every spawn

## 0.064 (2026-11-12)

- Fewer slow frames: new ground is worked out on background threads, and a reflection probe that moved is redrawn a face a frame instead of all at once

## 0.063 (2026-11-12)

- Meshes beside or behind the camera are no longer drawn (the GPU view test kept every box with a corner behind the eye), and crowd people cast shadows only into the cascades they can reach

## 0.062 (2026-11-12)

- Characters off screen or outside a shadow cascade are no longer drawn there, and the sky is shaded only where it shows

## 0.061 (2026-11-12)

- Character models with a missing normal no longer make broken pixels: the shaders take a zero normal safely and the normals are repaired when a model is uploaded

## 0.060 (2026-11-12)

- Very bright lamp glints no longer turn the picture dark: lamp highlights are capped like the sun's, stored brightness stays inside the image's range, and reflection probes drop broken values

## 0.059 (2026-11-11)

- Portals: glass and see-through materials in front of a doorway show over its view, and a portal's view draws only what fits through the opening (a big city drew everything again, unculled)

## 0.058 (2026-11-11)

- Portals: no ghost of the view flickering over a doorway while the camera moves (the opening's depth is put back after its view)

## 0.057 (2026-11-11)

- See-through particles (smoke, clouds of puffs) are drawn far to near every frame, so overlapping puffs no longer flicker

## 0.056 (2026-11-11)

- LOD sets from meshes prepared on worker threads (spawn_lod_prepared), for streamed far tiles

## 0.055 (2026-11-11)

- Lamp and sun highlights can no longer turn into broken pixels when a light lines up with the camera

## 0.054 (2026-11-11)

- A few broken pixels can no longer spread over the whole picture: shading, reflections, bounce light, anti-aliasing and glow drop invalid values; static shadow casters are no longer missed by the shadow cache

## 0.053 (2026-11-11)

- Less CPU time per frame in scenes with many materials: binding the material table no longer rechecks thousands of textures each time (a fix in Engine's copy of the graphics library)

## 0.052 (2026-11-11)

- Instanced things cut down to what is in view are drawn in a few combined draw calls instead of one per kind of thing, which saves CPU time in busy scenes

## 0.051 (2026-11-11)

- Instanced things (trees, rocks, props) are culled on the GPU in a few passes per view instead of one per kind of thing, which saves CPU time in busy scenes, and they no longer keep spare buffers each

## 0.050 (2026-11-11)

- Faster towns that stream: only what moved, came or went is updated each frame, instead of every entity and GPU scene row

## 0.049 (2026-11-10)

- Sun shadows keep their maps while the camera moves a little, so buildings are not drawn into them again every frame

## 0.048 (2026-11-10)

- Long sessions stay fast: textures made for meshes are freed with the last mesh that uses them (a streamed world used to pile them up until the frame rate fell), and world.resource_census() counts what the world and GPU hold, to catch anything else that grows

## 0.047 (2026-11-10)

- Instanced things no longer vanish up close on the software GPU used by verify runs and CI (it mishandled GPU draw counts, so they are off there)

## 0.046 (2026-11-10)

- Smoother loading: texture mip levels are made on the GPU (no CPU stall when a model or material first appears), custom material shaders compile on a worker thread, and the hitch log shows time lost between frames

## 0.045 (2026-11-10)

- Fixes from a fresh-agent test: look_follow_entity keeps the follow camera out of walls and the player, cast_ray/cast_sphere_ignoring, an API-at-a-glance page and a door recipe in AGENTS.md, and tools/new_example.py to start an example listed everywhere

## 0.044 (2026-11-10)

- Tab and the arrow keys belong to the game unless a UI text field has focus; a pointer resting on a panel no longer swallows the game's keys

## 0.043 (2026-11-10)

- Agent guide: the game skeleton walks a character, picks up coins with a trigger and shows a HUD; compiled recipes for HUD, world text, zones, doors, crates and sounds

## 0.042 (2026-11-10)

- Custom materials: a game writes a small WGSL function for a surface (lava, holograms, force fields), with time and params it sets each frame; it is lit and shadowed like any mesh, and mistakes name the line

## 0.041 (2026-11-10)

- Glass and other see-through things are drawn far to near, so they look right from every side; new glass example

## 0.040 (2026-11-10)

- Animation state machines: states, rules on speed, flags and triggers, and clip events like footsteps; new anim_states example

## 0.039 (2026-11-10)

- Game UI in one call each: corner text, bars, a pause menu and a crosshair; names hanging over things and floating text like "+1"

## 0.038 (2026-11-10)

- Character controller (walk, jump, steps, slopes, pushes crates) and joints (hinge, ball, slider, rope, weld), with a new gameplay example.

## 0.037 (2026-11-10)

- Gameplay physics: trigger zones that report what goes in and out, collision layers, and overlap and sphere-cast queries.

## 0.036 (2026-11-10)

- Engine source split into topic files: World methods in world_<topic>.rs, renderer methods in render/renderer_<topic>.rs, so agents and people find code faster. No behaviour change.

## 0.035 (2026-11-10)

- Helpful errors instead of panics at startup: no GPU, no window, no surface format, missing HDR support and running out of GPU memory now end the run with an error that says what to try (driver update, Mesa lavapipe without a GPU, lower quality). GPUs without Depth32FloatStencil8, timestamp queries or the optional draw features fall back instead of stopping. Errors print as readable messages.

## 0.034 (2026-11-10)

- Settings: everything a game sets at startup in one typed struct, Engine::run_with_settings("my game", Settings::default().window(1280, 720).vsync(false).quality(Quality::Medium), update). Bad values are named before the window opens; ENGINE_* variables stay as debug overrides, and world.env_overrides() and verify reports list them. AGENTS.md gains a rule on building meshes around the origin (its skeleton had that bug; a verify run caught it).

## 0.033 (2026-11-10)

- Game data on entities (typed components): world.insert(orc, Health(100)), get, get_mut, remove, has, query, query_mut, query2, query2_mut and ids_with. Components go away with their entity, queries list entities oldest first, verify reports count them, and save_component makes saves keep a type.

## 0.032 (2026-11-10)

- AGENTS.md: a short guide for coding agents (the model in one minute, a game skeleton, rules that avoid common mistakes, verify runs, where to find things, how to change the engine). CLAUDE.md points to it. Its code is compiled by the tests, so it stays right.

## 0.031 (2026-11-10)

- Verify runs for agents and CI: `python tools/verify.py <example>` plays a program for a fixed number of frames and writes a screenshot and report.json; games state what must hold with `world.check` and `world.report`. Fixes boulders and balls sinking through voxel and height terrain when they rest exactly on a seam between collision tiles.

## 0.030 (2026-11-10)

- CI on every pull request: formatting, clippy and the tests on Windows and Linux. Clippy is clean again (22 small lint fixes, no behaviour change).

## 0.029 (2026-11-10)

- Texture arrays bind only the material slots in use: wgpu checked all 16,384 fallback textures every time a pass bound the table. Finer CPU spans inside world.prepass_ao (prepass.first, hiz_cull, unseen, instances, ssao).

## 0.028 (2026-11-10)

- GPU shadows: shadow-casting meshes stay on the GPU; one compute pass culls them for every shadow view and writes draw-count multi-draws. ENGINE_GPU_SHADOWS=0 turns it off. Meshes whose texture has no see-through texels now cast shadows without the alpha test (one draw group instead of one per texture).

## 0.027 (2026-11-09)

- GPU draw counts: the GPU packs each multi-draw group's visible meshes and counts them (multi_draw_indexed_indirect_count), so hidden meshes cost no draw. ENGINE_DRAW_COUNT=0 turns it off.

## 0.026 (2026-11-09)

- GPU scene: opaque meshes stay on the GPU as rows; the GPU culls them (view and depth pyramid) and writes their draws, the CPU writes only changed rows. ENGINE_GPU_SCENE=0 turns it off.

## 0.025 (2026-11-09)

- Texture arrays: every mesh texture set sits in one material table, so meshes with different textures share one multi-draw call. occlusion_city with 16 facade textures (OCCLUSION_CITY_TEXTURES=16) draws its towers in 5 calls instead of 1,696, with the same picture. ENGINE_TEXTURE_ARRAYS=0 turns it off.

## 0.024 (2026-11-09)

- Fixed a memory leak that slowed scenes down the longer they ran: wgpu 25 never freed render bundles, so every re-recorded batch of draws stayed in memory (occlusion_city on an RTX 5070 fell from 244 to 74 FPS in 25 s with multi-draw and occlusion off). The engine now uses a patched copy of wgpu (third_party/).

## 0.023 (2026-11-09)

- Shadow maps draw many casters per call too: sun cascades, their static cache layers and lamp shadows send their mesh casters as one multi-draw per pipeline, texture and buffer page (kept while the casters stay the same). occlusion_city: the CPU's shadow setup went from 1.7 to 0.3 ms and the whole world from 9.9 to 7.1 ms per frame (lavapipe), picture identical. ENGINE_MULTI_DRAW=0 turns it off with the rest.

## 0.022 (2026-11-09)

- Many meshes per draw call: where the GPU supports multi-draw (every Vulkan and DX12 card), the world pass and the depth prepass send their opaque meshes as one call per texture and buffer page, with the per-mesh arguments in a GPU buffer; meshes the occlusion test decides take theirs straight from its output. occlusion_city: 1,697 towers in 5 calls instead of 1,697, about 2 ms less CPU per frame (lavapipe), picture identical. ENGINE_MULTI_DRAW=0 turns it off. ENGINE_FIXED_DT=0.016 steps every frame by the same time, so two runs can be compared picture for picture.

## 0.021 (2026-11-09)

- Instance transforms share GPU buffers too: every mesh's placements live in a few shared pages instead of a buffer per mesh (the next step towards drawing many meshes per draw call). Pictures identical (occlusion_city pixel for pixel, GPU byte check clean). Fixed: the terrain bundle is recorded anew when a terrain mesh's buffers change.

## 0.020 (2026-11-09)

- Meshes share GPU buffers: every mesh's vertices and indices now live in a few large pages (about 4 to 66 MB each) instead of two buffers per mesh, and each draw names its place in the page. Pictures are identical (occlusion_city and buildings checked pixel for pixel, a GPU byte check of every mesh is clean). It is the groundwork for drawing many meshes per draw call (next). ENGINE_FRAME_STATS counts the pages' size in the mesh memory.

## 0.019 (2026-11-09)

- Shadows of things that don't move are cached: each sun cascade keeps a map of its static casters (unchanged for half a second, not swaying in the wind, voxel ground), copies it in and draws only moving things on top (shadow pass 8.1 to 1.9 ms in occlusion_city on lavapipe, pictures identical). World::set_shadow_cache(false) or ENGINE_SHADOW_CACHE=0 turns it off.
- Fixed: lamp shadows were skipped on frames where no sun cascade was drawn (since 0.016).

## 0.018 (2026-11-09)

- Occlusion culling turns itself on where it pays: OcclusionMode::Auto (the new default) tries it whenever 32 or more meshes are in view, keeps it while it hides at least 15% of them and switches it off elsewhere, retrying less often each time. set_occlusion_culling(true/false) still forces it; set_occlusion_mode picks any mode; ENGINE_OCCLUSION=auto too. Without AO, hills and voxel ground now go into the depth pyramid, so they hide things as well.

## 0.017 (2026-11-09)

- Adjustable render resolution: world.set_render_scale(0.5 to 1.0) draws the scene smaller than the window and the TAA pass upscales it to full size (like FSR 2 or Unreal's TSR). Off by default (1.0). ENGINE_RENDER_SCALE overrides it. At 0.5 the scene pass costs about a third and AO about 40% as much (lavapipe, occlusion_city).

## 0.016 (2026-11-09)

- Far sun shadows are drawn on alternate frames: the near cascade every frame, the middle and far ones taking turns, so each frame draws two shadow maps instead of three (shadow pass about 40% cheaper, pictures unchanged). World::set_shadow_cascade_refresh sets the intervals; [1, 1, 1] or ENGINE_SHADOW_REFRESH=1 draws all three every frame.

## 0.015 (2026-11-09)

- Occlusion culling without ambient occlusion draws only the meshes it needs before the test, not a full depth prepass (example: OCCLUSION_CITY_AO=0)

## 0.014 (2026-11-09)

- Crowd poses: a crowd's profile can name extra clips (.poses(["drive", "ride"])) and world.set_crowd_agent_pose(crowd, agent, Some("drive")) holds an agent in one while it moves with its place, far (baked with idle and walk, nothing extra per agent) and near (its full character plays the pose, with foot IK off). For people sitting in cars or on benches without a second crowd, and a second copy of the model, per look.

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
