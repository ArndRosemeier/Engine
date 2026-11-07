//! The engine's own sounds for automatic effects (destruction, fire,
//! water, impacts), synthesised once on a worker thread so no frame waits.

use super::sound::Sound;
use std::sync::OnceLock;

pub(crate) struct Bank {
    pub explosion: Vec<Sound>,
    pub crumble: Vec<Sound>,
    pub dig: Vec<Sound>,
    pub knock_soft: Vec<Sound>,
    pub knock_hard: Vec<Sound>,
    pub splash: Vec<Sound>,
    pub fire: Sound,
    pub stream: Sound,
    pub river: Sound,
    pub surf: Sound,
}

static BANK: OnceLock<Bank> = OnceLock::new();
static STARTED: std::sync::Once = std::sync::Once::new();

fn build() -> Bank {
    let v = |f: fn(u64) -> Sound, base: u64| (0..3).map(|i| f(base + i)).collect::<Vec<_>>();
    Bank {
        explosion: v(Sound::explosion, 11),
        crumble: v(Sound::crumble, 21),
        dig: v(Sound::dig, 31),
        knock_soft: (0..3).map(|i| Sound::knock(41 + i, 0.25)).collect(),
        knock_hard: (0..3).map(|i| Sound::knock(51 + i, 0.8)).collect(),
        splash: v(Sound::splash, 61),
        fire: Sound::fire_loop(71),
        stream: Sound::stream_loop(81, 0.2),
        river: Sound::stream_loop(82, 0.9),
        surf: Sound::surf_loop(91),
    }
}

/// Start building the bank in the background (once per process).
pub(crate) fn warm() {
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("engine-sound-bank".into())
            .spawn(|| {
                let _ = BANK.get_or_init(build);
            });
        if spawned.is_err() {
            let _ = BANK.get_or_init(build);
        }
    });
}

/// The bank if it is ready; `None` for the first moments after start-up.
pub(crate) fn get() -> Option<&'static Bank> {
    warm();
    BANK.get()
}

/// The bank, building it now if needed (tests and tools).
#[cfg(test)]
pub(crate) fn wait() -> &'static Bank {
    BANK.get_or_init(build)
}

/// Pick one of `variants` from a position, so repeats differ.
pub(crate) fn pick(variants: &[Sound], key: f32) -> &Sound {
    let i = (key.abs() * 7.31).fract() * variants.len() as f32;
    &variants[(i as usize).min(variants.len() - 1)]
}
