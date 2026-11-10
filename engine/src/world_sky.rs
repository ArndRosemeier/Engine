//! Bloom, tone mapping, vsync, haze, sky, moons, wind, seasons and glow channels.
//!
//! Part of `impl World`, split from `world.rs` by topic.

use super::*;

impl World {
    pub fn bloom(&self) -> BloomSettings {
        self.bloom
    }

    pub fn set_bloom(&mut self, bloom: BloomSettings) {
        self.bloom = bloom;
    }

    /// Choose the tone-mapping curve (AgX by default). `ENGINE_TONEMAP=aces`
    /// or `agx` overrides it for quick comparisons.
    pub fn set_tone_map(&mut self, tone_map: ToneMap) {
        self.tone_map = tone_map;
    }

    pub fn tone_map(&self) -> ToneMap {
        self.tone_map
    }

    /// Request synchronized presentation (FIFO). Turning it off presents
    /// uncapped (`Immediate`, or `Mailbox` when the surface offers it); the
    /// renderer must refuse a surface that cannot present uncapped.
    pub fn set_vsync(&mut self, vsync: bool) {
        self.vsync = vsync;
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    /// Show the engine's "Engine 0.001 · What's new" button in the bottom
    /// right corner (default on; hidden in screenshot runs). Turn it off for
    /// a shipped game, or call [`crate::whats_new_ui`] where you want it.
    pub fn set_whats_new_button(&mut self, show: bool) {
        self.whats_new_button = show;
    }

    pub fn whats_new_button(&self) -> bool {
        self.whats_new_button
    }
    pub(crate) fn note_shadow_contact(&mut self, snap: ContactSnapshot, epoch: u64) {
        if self.shadow_contact_epoch != epoch {
            self.shadow_contact_epoch = epoch;
            self.shadow_contact = snap;
        }
    }

    pub(crate) fn shadow_contact(&self) -> &ContactSnapshot {
        &self.shadow_contact
    }

    pub(crate) fn shadow_contact_epoch(&self) -> u64 {
        self.shadow_contact_epoch
    }

    pub fn set_clear_color(&mut self, color: Color) {
        self.clear_color = color;
    }

    /// Put air in the scene, or take it out again with `None`.
    pub fn set_haze(&mut self, haze: Option<Haze>) {
        self.haze = haze;
    }

    pub fn haze(&self) -> Option<Haze> {
        self.haze
    }

    /// Put a sky behind the scene, or fall back to [`Self::clear_color`].
    pub fn set_sky(&mut self, sky: Option<Sky>) {
        self.sky = sky;
    }

    pub fn sky(&self) -> Option<Sky> {
        self.sky
    }

    /// Drive the sun, moon, sky dome, ambient light, haze and exposure from
    /// a time of day and weather (see [`crate::sky`]). Change `hour` every
    /// frame for a running day. `None` hands sun, sky and haze back to the
    /// game, as they were last set.
    pub fn set_physical_sky(&mut self, sky: Option<crate::sky::PhysicalSky>) {
        self.physical_sky = sky;
        self.tick_sky();
    }

    pub fn physical_sky(&self) -> Option<crate::sky::PhysicalSky> {
        self.physical_sky
    }

    /// The physical sky's moons (up to [`crate::sky::MAX_MOONS`]), each with
    /// its own size, cycle and path; their light adds up at night. Empty
    /// (the default) keeps the one usual moon.
    pub fn set_moons(&mut self, moons: Vec<crate::sky::MoonSpec>) -> EngineResult<()> {
        if moons.len() > crate::sky::MAX_MOONS {
            return Err(EngineError::InvalidValue(format!(
                "at most {} moons",
                crate::sky::MAX_MOONS
            )));
        }
        for m in &moons {
            m.validate()?;
        }
        self.moons = moons;
        Ok(())
    }

    pub fn moons(&self) -> &[crate::sky::MoonSpec] {
        &self.moons
    }

    /// Wind for swaying vegetation (see [`crate::vegetation`]): 0 calm,
    /// 1 a strong breeze, 2 a storm. Default 0.5.
    pub fn set_wind(&mut self, strength: f32) -> EngineResult<()> {
        if !strength.is_finite() || !(0.0..=4.0).contains(&strength) {
            return Err(EngineError::InvalidValue(format!(
                "wind {strength} must be within 0..=4"
            )));
        }
        self.wind = strength;
        Ok(())
    }

    pub fn wind(&self) -> f32 {
        self.wind
    }

    /// Leaf colour season: 0 summer green, 0.5 early autumn, 1 full autumn
    /// (evergreens ignore it).
    pub fn set_season(&mut self, season: f32) -> EngineResult<()> {
        if !season.is_finite() || !(0.0..=1.0).contains(&season) {
            return Err(EngineError::InvalidValue(format!(
                "season {season} must be within 0..=1"
            )));
        }
        self.season = season;
        Ok(())
    }

    pub fn season(&self) -> f32 {
        self.season
    }

    /// Switch a glow channel (see [`crate::mesh::SurfaceMaterial::glow_channel`]):
    /// every surface on `channel` (1 to 15) glows `strength` times its
    /// emission (0 dark, 1 as authored, more brighter), in `color` or, with
    /// `None`, in its own colour. Changing it costs nothing: the meshes stay
    /// as they are, so windows can come on at dusk one channel at a time
    /// and neon can flicker. Channel 0 is always on as authored.
    ///
    /// ```no_run
    /// # use engine::prelude::*;
    /// # fn f(world: &mut World, mut mesh: Mesh) -> EngineResult<()> {
    /// // Panes that glow warm at night and look like glass by day.
    /// mesh.set_surface_material(SurfaceMaterial { emission: 1.2, ..SurfaceMaterial::DEFAULT }
    ///     .on_glow_channel(1));
    /// world.set_glow_channel(1, 0.0, Some(Color::rgb(255, 210, 150)))?; // day: off
    /// world.set_glow_channel(1, 1.0, Some(Color::rgb(255, 210, 150)))?; // night: on
    /// # Ok(()) }
    /// ```
    pub fn set_glow_channel(
        &mut self,
        channel: u8,
        strength: f32,
        color: Option<crate::color::Color>,
    ) -> EngineResult<()> {
        if channel == 0 || channel as usize >= crate::mesh::GLOW_CHANNELS {
            return Err(EngineError::InvalidValue(format!(
                "glow channel {channel} must be within 1..={}",
                crate::mesh::GLOW_CHANNELS - 1
            )));
        }
        if !strength.is_finite() || strength < 0.0 {
            return Err(EngineError::InvalidValue(format!(
                "glow strength {strength} must be finite and not negative"
            )));
        }
        if color.is_some_and(|c| !(c.r.is_finite() && c.g.is_finite() && c.b.is_finite())) {
            return Err(EngineError::InvalidColor(
                "glow colour must be finite".into(),
            ));
        }
        let rgb = color.map_or([-1.0, 0.0, 0.0], |c| {
            [c.r.max(0.0), c.g.max(0.0), c.b.max(0.0)]
        });
        self.glow[channel as usize] = [rgb[0], rgb[1], rgb[2], strength];
        Ok(())
    }

    /// A glow channel's strength and colour (`None`: the surfaces' own).
    pub fn glow_channel(&self, channel: u8) -> Option<(f32, Option<crate::color::Color>)> {
        let g = self.glow.get(channel as usize)?;
        let color = (g[0] >= 0.0).then_some(crate::color::Color {
            r: g[0],
            g: g[1],
            b: g[2],
            a: 1.0,
        });
        Some((g[3], color))
    }

    pub(crate) fn glow_table(&self) -> [[f32; 4]; crate::mesh::GLOW_CHANNELS] {
        self.glow
    }

    /// Apply the physical sky's lighting to the world's light, sky, haze
    /// and exposure. Runs every frame before rendering.
    pub(crate) fn tick_sky(&mut self) {
        let Some(sky) = self.physical_sky else {
            return;
        };
        let l = sky.lighting_with(&sky.moon_views(&self.moons));
        // AssetGenerator's sun is 3.2 at a clear low sun, ~4.5 at noon, under
        // an exposure of 0.55; the engine's daylight is 1.0 under exposure 1.
        self.light.direction = l.light_dir;
        self.light.color = l.light_color * (l.light_intensity / 3.2);
        self.light.ambient = (l.hemi_intensity * 0.3).clamp(0.0, 1.0);
        let c = |v: Vec3| Color {
            r: v.x,
            g: v.y,
            b: v.z,
            a: 1.0,
        };
        let sun_color = l.light_color / l.light_color.max_element().max(1e-3);
        self.sky = Some(Sky {
            zenith: c(l.sky_color),
            horizon: c(l.fog_color),
            ground: c(l.ground_color),
            sun_color: c(sun_color),
            sun_size_degrees: crate::sky::SUN_RADIUS.to_degrees(),
            sun_bloom_degrees: 20.0,
            curve: 0.22,
        });
        // exp(-d * density) reaches 2% at 3.9 / density.
        let visibility = (3.9 / l.fog_density).min(self.view_distance.max(50.0) * 1.5);
        self.haze = Some(Haze::new(c(l.fog_color), visibility));
        let _ = self.bloom.set_exposure((l.exposure / 0.55).clamp(0.1, 8.0));
    }
}
