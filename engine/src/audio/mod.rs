//! Sound for games: clips placed in the world, reverb that follows rooms
//! and caves, muffling under water and behind walls, and music in layers
//! that follow the game's intensity.
//!
//! Most games only touch `World`:
//!
//! ```no_run
//! # use engine::prelude::*;
//! # fn f(world: &mut World) -> EngineResult<()> {
//! let boom = Sound::load("assets/boom.ogg")?;          // or Sound::explosion(seed)
//! world.play_sound(&boom, PlaySound::at([0.0, 1.0, -20.0]).range(5.0, 600.0));
//! world.play_music(MusicTrack::procedural(7, 96.0), 2.0);
//! world.set_music_intensity(0.8);                       // bring in the drums
//! # Ok(()) }
//! ```
//!
//! The engine opens the default output device when the app starts, moves
//! the listener with the camera, and works out the acoustics (open air,
//! room, hall, cave, under water) every frame. Destruction, fire,
//! explosions, splashes, hard impacts and flowing water make their own
//! sounds unless `world.set_auto_sounds(false)`.

pub(crate) mod bank;
mod mixer;
mod player;
mod reverb;
pub(crate) mod scene;
mod sound;
mod synth;

pub use mixer::Bus;
pub use player::{Audio, ClipId, Play, VoiceId};
pub use reverb::Reverb;
pub use scene::{Acoustics, AudioScene, MusicLayer, MusicTrack, PlaySound, SoundId};
pub use sound::Sound;
