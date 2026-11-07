//! The engine's own sounds for automatic effects (destruction, fire,
//! water, impacts, wind). Recorded effects are embedded in the engine
//! (`engine/assets/sounds`) and decoded once on a worker thread at start-up,
//! so no frame waits for them.

use super::sound::Sound;
use std::sync::OnceLock;

/// One of the engine's built-in sounds, for [`Sound::stock`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stock {
    /// A big explosion with a rumbling tail (3 variants).
    Explosion,
    /// Stone and earth collapsing (3 variants).
    Crumble,
    /// A blast or spade into soil (3 variants).
    Dig,
    /// Wood or earth thud (3 variants).
    KnockSoft,
    /// Stone on stone (3 variants).
    KnockHard,
    /// Something falling into water (3 variants).
    Splash,
    /// A camp fire crackling (seamless loop).
    FireLoop,
    /// A small babbling stream (seamless loop).
    StreamLoop,
    /// A big rushing river (seamless loop).
    RiverLoop,
    /// Waves on a beach (seamless loop).
    SurfLoop,
    /// Gusty wind (seamless loop).
    WindLoop,
}

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
    pub wind: Sound,
}

impl Bank {
    fn get(&self, stock: Stock, variant: usize) -> &Sound {
        match stock {
            Stock::Explosion => return_ref(&self.explosion, variant),
            Stock::Crumble => return_ref(&self.crumble, variant),
            Stock::Dig => return_ref(&self.dig, variant),
            Stock::KnockSoft => return_ref(&self.knock_soft, variant),
            Stock::KnockHard => return_ref(&self.knock_hard, variant),
            Stock::Splash => return_ref(&self.splash, variant),
            Stock::FireLoop => &self.fire,
            Stock::StreamLoop => &self.stream,
            Stock::RiverLoop => &self.river,
            Stock::SurfLoop => &self.surf,
            Stock::WindLoop => &self.wind,
        }
    }
}

fn return_ref(list: &[Sound], variant: usize) -> &Sound {
    &list[variant % list.len()]
}

static BANK: OnceLock<Bank> = OnceLock::new();
static STARTED: std::sync::Once = std::sync::Once::new();

macro_rules! clip {
    ($name:literal) => {
        decode(
            include_bytes!(concat!("../../assets/sounds/", $name, ".ogg")),
            $name,
        )
    };
}

fn decode(bytes: &'static [u8], name: &str) -> Sound {
    Sound::from_bytes(bytes.to_vec())
        .unwrap_or_else(|e| panic!("built-in sound {name} does not decode: {e}"))
}

fn build() -> Bank {
    Bank {
        explosion: vec![
            clip!("explosion_1"),
            clip!("explosion_2"),
            clip!("explosion_3"),
        ],
        crumble: vec![clip!("crumble_1"), clip!("crumble_2"), clip!("crumble_3")],
        dig: vec![clip!("dig_1"), clip!("dig_2"), clip!("dig_3")],
        knock_soft: vec![
            clip!("knock_soft_1"),
            clip!("knock_soft_2"),
            clip!("knock_soft_3"),
        ],
        knock_hard: vec![
            clip!("knock_hard_1"),
            clip!("knock_hard_2"),
            clip!("knock_hard_3"),
        ],
        splash: vec![clip!("splash_1"), clip!("splash_2"), clip!("splash_3")],
        fire: clip!("fire_loop"),
        stream: clip!("stream_loop"),
        river: clip!("river_loop"),
        surf: clip!("surf_loop"),
        wind: clip!("wind_loop"),
    }
}

/// Start decoding the bank in the background (once per process).
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

/// The bank, decoding it now if needed.
pub(crate) fn wait() -> &'static Bank {
    BANK.get_or_init(build)
}

/// Pick one of `variants` from a position, so repeats differ.
pub(crate) fn pick(variants: &[Sound], key: f32) -> &Sound {
    let i = (key.abs() * 7.31).fract() * variants.len() as f32;
    &variants[(i as usize).min(variants.len() - 1)]
}

impl Sound {
    /// One of the engine's built-in recorded sounds (see [`Stock`]);
    /// `variant` picks among the takes where there are several. Cheap after
    /// the first call (the samples are shared).
    pub fn stock(stock: Stock, variant: usize) -> Sound {
        wait().get(stock, variant).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{Sound, Stock};

    #[test]
    fn every_stock_sound_decodes_and_is_audible() {
        for stock in [
            Stock::Explosion,
            Stock::Crumble,
            Stock::Dig,
            Stock::KnockSoft,
            Stock::KnockHard,
            Stock::Splash,
            Stock::FireLoop,
            Stock::StreamLoop,
            Stock::RiverLoop,
            Stock::SurfLoop,
            Stock::WindLoop,
        ] {
            for v in 0..3 {
                let s = Sound::stock(stock, v);
                let peak = s.0.samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                assert!(peak > 0.02, "{stock:?} {v} peak {peak}");
                assert_eq!(s.sample_rate(), 48_000);
            }
        }
    }
}
