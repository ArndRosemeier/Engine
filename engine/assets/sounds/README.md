# Engine default sounds

The engine's own sounds for automatic effects, embedded in the engine and
reachable with `Sound::stock`. Generated with ByteDance Seed Audio through
OpenRouter (2026-11-07). Crumbles were shortened to 6 s, digs to 2.5 s,
splashes to 3.5 s and knock_hard_2 to 2 s, each with a 0.8 s fade-out.

# Engine default sound effects

Generated 2026-11-07 with OpenRouter model `bytedance-seed/seed-audio-1-0` (ByteDance Seed Audio, `POST /audio/speech`, mp3 output). No seed parameter is exposed, so regenerating gives a new take.

## Processing

- Mono, 48 kHz, OGG Vorbis 96 kbit/s.
- One-shots: leading silence trimmed (first point above -40 dB of the clip's peak, 5 ms pre-roll), trailing silence trimmed with a 150 ms fade-out.
- Loudness target -16 LUFS integrated, true peak at or below -1 dBTP. Gain is capped at 8 dB of limiting (lookahead brickwall limiter), so very transient sounds (knocks, digs, the fire crackle) stay below -16 LUFS instead of being squashed. Measured values are in the table.
- Loops: the steadiest window of the clip (most even loudness, after the first second) is cut, and its last 1.0 s is equal-power crossfaded with the audio just before its start, so the end runs straight into the beginning. The limiter runs on three tiled copies so the seam stays clean. Exact sample lengths (5 s = 240000 samples).
- Scripts: `tools/orc.py` (OpenRouter client), `tools/proc.py` (processing), `tools/prompts.json`.

## Files

| File | Length | LUFS | True peak | Prompt |
| --- | --- | --- | --- | --- |
| `explosion_1.ogg` | 9.23 s | -16.0 | -1.5 dB | The sound of a big deep explosion, a powerful low boom followed by a long rumbling tail that slowly fades |
| `explosion_2.ogg` | 9.41 s | -17.5 | -1.3 dB | The sound of a massive distant blast, heavy concussive thump, deep bass rumble rolling away over several seconds |
| `explosion_3.ogg` | 8.61 s | -17.2 | -1.3 dB | The sound of a huge explosion close by, sharp crack then a deep roaring boom and a long low rumble with faint debris |
| `crumble_1.ogg` | 14.07 s | -16.6 | -1.2 dB | The sound of stone and earth collapsing, heavy rocks tumbling and breaking apart, gravel and debris trickling down |
| `crumble_2.ogg` | 11.96 s | -16.9 | -1.2 dB | The sound of a rock wall crumbling, chunks of stone grinding and falling, dust and small pebbles scattering afterwards |
| `crumble_3.ogg` | 13.35 s | -16.3 | -1.2 dB | The sound of a cliff of dirt and boulders caving in, deep rumbling slide of earth with clattering stone debris |
| `dig_1.ogg` | 5.48 s | -24.2 | -1.2 dB | The sound of a shovel digging into soil, a dull thud followed by gritty scraping of dirt |
| `dig_2.ogg` | 5.07 s | -21.5 | -1.0 dB | The sound of a spade plunging into damp earth, soft heavy thump and loose grit sliding off |
| `dig_3.ogg` | 6.41 s | -16.5 | -1.6 dB | The sound of a small blast into soil, a muffled dull thud and dirt and gravel raining down |
| `knock_soft_1.ogg` | 0.45 s | -20.5 | -1.5 dB | The sound of a single dull knock of a wooden log hitting packed earth, short soft thud |
| `knock_soft_2.ogg` | 1.09 s | -22.6 | -1.3 dB | The sound of a wooden crate dropped onto soft ground, one short muffled wooden thud |
| `knock_soft_3.ogg` | 0.9 s | -22.7 | -1.6 dB | The sound of a single soft thump of wood on dirt, short and dull |
| `knock_hard_1.ogg` | 0.23 s | -24.7 | -1.5 dB | The sound of a single sharp knock of a stone hitting another stone, short hard clack |
| `knock_hard_2.ogg` | 3.23 s | -22.6 | -1.7 dB | The sound of a heavy rock dropped onto a stone floor, one hard sharp impact |
| `knock_hard_3.ogg` | 2.07 s | -22.3 | -1.2 dB | The sound of two rocks struck together once, a crisp hard stone click |
| `splash_1.ogg` | 4.84 s | -16.4 | -1.6 dB | The sound of a heavy object falling into water, a deep plunging splash and water settling |
| `splash_2.ogg` | 3.02 s | -18.3 | -1.6 dB | The sound of a rock dropped into a lake, a sharp splash and bubbling plop |
| `splash_3.ogg` | 5.02 s | -17.4 | -1.4 dB | The sound of a large log falling into a river, a big heavy splash with water spray falling back |
| `fire_loop.ogg` | 5.0 s | -27.2 | -1.4 dB | The sound of a campfire crackling steadily, wood popping and snapping, soft flame roar, continuous, no other sounds |
| `stream_loop.ogg` | 6.0 s | -16.5 | -1.6 dB | The sound of a small babbling brook, gentle trickling water over pebbles, continuous and steady |
| `river_loop.ogg` | 6.0 s | -16.2 | -2.7 dB | The sound of a big rushing river, powerful continuous roar of fast flowing water, steady |
| `surf_loop.ogg` | 8.0 s | -16.4 | -1.1 dB | The sound of ocean waves breaking on a sandy beach, waves rolling in and washing back, continuous and steady |
| `wind_loop.ogg` | 6.0 s | -16.4 | -1.3 dB | The sound of gusty wind blowing outdoors, rising and falling gusts, continuous, no other sounds |

Model id for every file: `bytedance-seed/seed-audio-1-0`.
