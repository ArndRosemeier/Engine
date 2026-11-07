//! Sound in the world: `World`'s audio calls, the listener riding on the
//! camera, acoustics worked out from rooms, caves and water, sounds muffled
//! behind walls, and automatic sounds for destruction, fire, splashes,
//! impacts and flowing water.

use super::World;
use crate::audio::bank;
use crate::audio::scene::ListenerPose;
use crate::audio::{Acoustics, AudioScene, MusicTrack, PlaySound, Reverb, Sound, SoundId};
use crate::rooms::RoomId;
use glam::Vec3;
use std::collections::HashMap;

/// Bookkeeping for automatic sounds and acoustics.
#[derive(Debug, Default)]
pub(crate) struct AutoAudio {
    room_reverb: HashMap<RoomId, Reverb>,
    /// Acoustics last worked out (recomputed a few times a second).
    acoustics: Option<Acoustics>,
    acoustics_timer: f32,
    occlusion_cursor: usize,
    water_timer: f32,
    river: Option<SoundId>,
    surf: Option<SoundId>,
    /// Entities that just made an impact sound (seconds of quiet left).
    impact_quiet: HashMap<crate::world::EntityId, f32>,
}

impl World {
    /// All of the world's sound: volumes, music, acoustics.
    pub fn audio(&self) -> &AudioScene {
        &self.audio
    }

    pub fn audio_mut(&mut self) -> &mut AudioScene {
        &mut self.audio
    }

    /// Play a sound. See [`PlaySound`] for placing it, looping it and
    /// more; [`World::play_sound_at`] is the short form for a one-shot.
    pub fn play_sound(&mut self, sound: &Sound, play: PlaySound) -> SoundId {
        self.audio.play(sound, play)
    }

    /// Play a sound once at a point in the world.
    pub fn play_sound_at(&mut self, sound: &Sound, at: impl Into<Vec3>) -> SoundId {
        self.audio.play(sound, PlaySound::at(at))
    }

    /// Stop a sound, fading out over `fade` seconds.
    pub fn stop_sound(&mut self, id: SoundId, fade: f32) {
        self.audio.stop(id, fade);
    }

    /// Play music, cross-fading from what plays now over `fade` seconds.
    pub fn play_music(&mut self, track: MusicTrack, fade: f32) {
        self.audio.play_music(track, fade);
    }

    pub fn stop_music(&mut self, fade: f32) {
        self.audio.stop_music(fade);
    }

    /// 0 (calm) to 1 (fierce): which layers of the music play.
    pub fn set_music_intensity(&mut self, intensity: f32) {
        self.audio.set_music_intensity(intensity);
    }

    /// Sounds the engine makes by itself: explosions, digging, crumbling,
    /// fire, splashes, hard impacts and flowing water (on by default).
    pub fn set_auto_sounds(&mut self, on: bool) {
        self.audio.auto_sounds = on;
        if !on {
            for id in [self.auto_audio.river.take(), self.auto_audio.surf.take()]
                .into_iter()
                .flatten()
            {
                self.audio.stop(id, 1.0);
            }
        }
    }

    pub fn auto_sounds(&self) -> bool {
        self.audio.auto_sounds
    }

    /// How a room sounds, instead of the reverb guessed from its size.
    pub fn set_room_reverb(&mut self, room: RoomId, reverb: Option<Reverb>) {
        match reverb {
            Some(r) => {
                self.auto_audio.room_reverb.insert(room, r);
            }
            None => {
                self.auto_audio.room_reverb.remove(&room);
            }
        }
        self.auto_audio.acoustics = None;
    }

    /// Play one of the engine's own sounds, if automatic sounds are on and
    /// the sounds are ready.
    fn auto_play(
        &mut self,
        pick: impl FnOnce(&'static bank::Bank) -> &'static Sound,
        play: PlaySound,
    ) {
        if !self.audio.auto_sounds {
            return;
        }
        if let Some(b) = bank::get() {
            self.audio.play(pick(b), play);
        }
    }

    pub(crate) fn sound_explosion(&mut self, at: Vec3, radius: f32) {
        let r = radius.clamp(0.2, 50.0);
        self.auto_play(
            |b| bank::pick(&b.explosion, at.x + at.z),
            PlaySound::at(at)
                .volume((r / 3.0).sqrt().clamp(0.4, 3.0))
                .pitch((1.15 - 0.12 * r.ln()).clamp(0.6, 1.3))
                .range(r * 3.0, 300.0 + 60.0 * r)
                .reverb(1.3),
        );
    }

    pub(crate) fn sound_dig(&mut self, at: Vec3, radius: f32) {
        self.auto_play(
            |b| bank::pick(&b.dig, at.x - at.z),
            PlaySound::at(at)
                .volume(1.5 * (radius / 1.5).sqrt().clamp(0.3, 2.0))
                .pitch((1.2 - 0.1 * radius).clamp(0.7, 1.3))
                .range(3.0, 80.0 + 10.0 * radius),
        );
    }

    pub(crate) fn sound_crumble(&mut self, at: Vec3, radius: f32) {
        self.auto_play(
            |b| bank::pick(&b.crumble, at.y + at.x),
            PlaySound::at(at)
                .volume((radius / 2.0).sqrt().clamp(0.3, 2.0))
                .pitch((1.25 - 0.08 * radius).clamp(0.7, 1.3))
                .range(3.0 + radius, 100.0 + 15.0 * radius),
        );
    }

    /// A crackling loop at a fire, `size` metres across.
    pub(crate) fn sound_fire(&mut self, at: Vec3, size: f32) -> Option<SoundId> {
        if !self.audio.auto_sounds {
            return None;
        }
        let b = bank::get()?;
        Some(
            self.audio.play(
                &b.fire,
                PlaySound::at(at)
                    .looped()
                    .volume((0.9 * size.sqrt()).clamp(0.3, 3.0))
                    .range(1.0 + size, 30.0 + 10.0 * size)
                    .bus(crate::audio::Bus::Ambience)
                    .fade_in(0.8),
            ),
        )
    }

    pub(crate) fn sound_splashes(&mut self, splashes: &[(Vec3, f32)]) {
        for &(at, speed) in splashes.iter().take(3) {
            if speed < 1.0 {
                continue;
            }
            self.auto_play(
                |b| bank::pick(&b.splash, at.x * 3.0 + at.z),
                PlaySound::at(at)
                    .volume((speed / 5.0).clamp(0.15, 1.5))
                    .pitch((1.3 - speed * 0.04).clamp(0.7, 1.3))
                    .range(2.0, 60.0),
            );
        }
    }

    pub(crate) fn sound_impacts(&mut self, impacts: &[crate::physics::Impact], dt: f32) {
        self.auto_audio.impact_quiet.retain(|_, t| {
            *t -= dt;
            *t > 0.0
        });
        let mut made = 0;
        for i in impacts {
            if made >= 4 {
                break;
            }
            if i.speed < 2.0 || self.auto_audio.impact_quiet.contains_key(&i.a) {
                continue;
            }
            self.auto_audio.impact_quiet.insert(i.a, 0.15);
            made += 1;
            // Hitting the ground is dull, body on body is harder.
            let hard = i.b.is_some();
            let key = i.point.x * 5.0 + i.point.z * 3.0;
            self.auto_play(
                |b| bank::pick(if hard { &b.knock_hard } else { &b.knock_soft }, key),
                PlaySound::at(i.point)
                    .volume((i.speed / 4.0).clamp(0.2, 3.0))
                    .pitch(0.9 + (key.fract().abs()) * 0.25)
                    .range(1.5, 60.0),
            );
        }
    }

    /// Move the ears with the camera, work out the acoustics, muffle
    /// sounds behind walls, keep water ambience going, and hand everything
    /// to the mixer. The engine calls this every frame.
    pub fn tick_audio(&mut self, dt: f32) {
        let cam = self.camera();
        let listener = ListenerPose {
            pos: cam.eye(),
            forward: cam.target() - cam.eye(),
            up: cam.up(),
        };

        // Sounds that follow entities.
        let placed = self.audio.placed_voices();
        for (id, _, follow) in &placed {
            if let Some(e) = follow {
                if let Some(t) = self.entities.get(e) {
                    self.audio.set_voice_pos(*id, t.transform.w_axis.truncate());
                }
            }
        }

        // Occlusion: a few voices a frame get a ray from the ears.
        if !placed.is_empty() {
            let listener_room = self.room_at(listener.pos);
            let n = placed.len().min(6);
            for k in 0..n {
                let idx = (self.auto_audio.occlusion_cursor + k) % placed.len();
                let (id, pos, follow) = placed[idx];
                let pos = follow
                    .and_then(|e| self.entities.get(&e))
                    .map_or(pos, |t| t.transform.w_axis.truncate());
                let d = pos - listener.pos;
                let dist = d.length();
                let mut occ = 0.0;
                if dist > 1.0 && dist < 400.0 {
                    if let Some(hit) = self.cast_ray(listener.pos, d, dist - 0.5) {
                        // Ignore the body that makes the sound.
                        if follow.is_none() || hit.entity != follow {
                            occ = 1.0;
                        }
                    }
                    let room = self.room_at(pos);
                    if room != listener_room && (room.is_some() || listener_room.is_some()) {
                        occ = f32::max(occ, 0.55);
                    }
                }
                self.audio.set_voice_occlusion(id, occ);
            }
            self.auto_audio.occlusion_cursor =
                (self.auto_audio.occlusion_cursor + n) % placed.len().max(1);
        }

        // Acoustics, a few times a second.
        self.auto_audio.acoustics_timer -= dt;
        if self.auto_audio.acoustics.is_none() || self.auto_audio.acoustics_timer <= 0.0 {
            self.auto_audio.acoustics_timer = 0.2;
            self.auto_audio.acoustics = Some(self.acoustics_at(listener.pos));
        }

        self.auto_audio.water_timer -= dt;
        if self.auto_audio.water_timer <= 0.0 {
            self.auto_audio.water_timer = 0.3;
            self.update_water_ambience(listener.pos);
        }

        let acoustics = self.auto_audio.acoustics.unwrap_or_default();
        self.audio.update(dt, listener, acoustics);
    }

    /// What the space at `p` sounds like: under water, in a room, in a
    /// cave or in the open.
    pub fn acoustics_at(&self, p: Vec3) -> Acoustics {
        if self.water().is_some_and(|w| w.is_submerged(p)) {
            return Acoustics::UNDERWATER;
        }
        if let Some(room) = self.room_at(p) {
            if let Some(r) = self.auto_audio.room_reverb.get(&room) {
                return Acoustics::with_reverb(*r);
            }
            let volume = self.room(room).map_or(100.0, |r| r.volume());
            // Small rooms sound like rooms, big ones like halls.
            let t = ((volume.max(1.0).ln() - 150f32.ln()) / (4000f32.ln() - 150f32.ln()))
                .clamp(0.0, 1.0);
            return Acoustics::with_reverb(Reverb::ROOM.lerp(Reverb::HALL, t));
        }
        // Enclosed by ground on most sides: a cave (or a narrow gorge).
        let dirs = [
            Vec3::Y,
            Vec3::new(1.0, 0.3, 0.0),
            Vec3::new(-1.0, 0.3, 0.0),
            Vec3::new(0.0, 0.3, 1.0),
            Vec3::new(0.0, 0.3, -1.0),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        let reach = 40.0;
        let mut hits = 0;
        let mut up_hit = false;
        let mut total = 0.0;
        for (i, d) in dirs.iter().enumerate() {
            if let Some(h) = self.ground_ray(p, d.normalize(), reach) {
                hits += 1;
                total += h.distance;
                if i == 0 {
                    up_hit = true;
                }
            }
        }
        if up_hit && hits >= 4 {
            let mean = total / hits as f32;
            let size = (mean / 25.0).clamp(0.0, 1.0);
            let small_cave = Reverb {
                size: 0.65,
                damping: 0.3,
                wet: 0.35,
                predelay_ms: 10.0,
            };
            return Acoustics::with_reverb(small_cave.lerp(Reverb::CAVE, size));
        }
        if hits >= 3 {
            return Acoustics::with_reverb(Reverb::OUTDOORS.lerp(Reverb::ROOM, 0.4));
        }
        Acoustics::OUTDOORS
    }

    /// Keep a stream/river loop and a surf loop where the water near the
    /// listener flows or breaks on a shore.
    fn update_water_ambience(&mut self, eye: Vec3) {
        let wanted = self.audio.auto_sounds && bank::get().is_some();
        let (mut river, mut surf) = ((0.0f32, Vec3::ZERO, 0.0f32), (0.0f32, Vec3::ZERO, 0.0f32));
        if let (true, Some(water)) = (wanted, self.water()) {
            let has_sea = water.sea().is_some();
            let (step, half) = (6.0f32, 5i32);
            for iz in -half..=half {
                for ix in -half..=half {
                    let x = eye.x + ix as f32 * step;
                    let z = eye.z + iz as f32 * step;
                    let Some(surface) = water.surface_at(Vec3::new(x, eye.y + 20.0, z)) else {
                        continue;
                    };
                    let at = Vec3::new(x, surface, z);
                    let Some(s) = water.sample(at - Vec3::Y * 0.05) else {
                        continue;
                    };
                    let d2 = (at - eye).length_squared();
                    let w = 1.0 / (1.0 + d2 / 60.0);
                    let flow = s.current.length() * s.water_depth.min(1.5);
                    if flow > 0.05 {
                        river.0 += flow * w;
                        river.1 += at * flow * w;
                        river.2 += flow * w * s.current.length();
                    }
                    if has_sea && s.water_depth < 1.2 {
                        surf.0 += w;
                        surf.1 += at * w;
                    }
                }
            }
        }
        let river_loud = (river.0 / 4.0).clamp(0.0, 1.0);
        let fast = if river.0 > 0.0 {
            river.2 / river.0
        } else {
            0.0
        };
        let river_at = if river.0 > 0.0 {
            river.1 / river.0
        } else {
            eye
        };
        let surf_loud = (surf.0 / 3.0).clamp(0.0, 1.0);
        let surf_at = if surf.0 > 0.0 { surf.1 / surf.0 } else { eye };
        let big = fast > 1.5;
        self.auto_audio.river = self.keep_loop(
            self.auto_audio.river,
            river_loud,
            river_at,
            |b| if big { &b.river } else { &b.stream },
            1.3,
        );
        self.auto_audio.surf =
            self.keep_loop(self.auto_audio.surf, surf_loud, surf_at, |b| &b.surf, 1.6);
    }

    fn keep_loop(
        &mut self,
        id: Option<SoundId>,
        loudness: f32,
        at: Vec3,
        pick: impl FnOnce(&'static bank::Bank) -> &'static Sound,
        gain: f32,
    ) -> Option<SoundId> {
        if loudness < 0.02 {
            if let Some(id) = id {
                self.audio.stop(id, 1.5);
            }
            return None;
        }
        let id = match id.filter(|id| self.audio.is_playing(*id)) {
            Some(id) => id,
            None => {
                let b = bank::get()?;
                self.audio.play(
                    pick(b),
                    PlaySound::at(at)
                        .looped()
                        .volume(0.0)
                        .range(4.0, 90.0)
                        .doppler(0.0)
                        .bus(crate::audio::Bus::Ambience),
                )
            }
        };
        self.audio.set_position(id, at);
        self.audio.set_volume(id, loudness * gain);
        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use crate::audio::{PlaySound, Sound};
    use crate::World;

    #[test]
    fn explosions_make_a_sound_by_themselves() {
        let _ = crate::audio::bank::wait();
        let mut world = World::new();
        world.explosion([0.0, 0.0, -10.0], 3.0).unwrap();
        assert!(world.audio().voice_count() >= 1);
        world.set_auto_sounds(false);
        let before = world.audio().voice_count();
        world.explosion([0.0, 0.0, -10.0], 3.0).unwrap();
        assert_eq!(world.audio().voice_count(), before);
    }

    #[test]
    fn a_fire_crackles_until_put_out() {
        let _ = crate::audio::bank::wait();
        let mut world = World::new();
        let fire = world.start_fire([0.0, 0.0, -5.0], 1.0).unwrap();
        world.tick_audio(0.016);
        assert_eq!(world.audio().voice_count(), 1);
        world.stop_fire(fire);
        world.tick_audio(0.016);
        assert_eq!(world.audio().voice_count(), 0);
    }

    #[test]
    fn open_ground_sounds_like_outdoors() {
        let world = World::new();
        assert_eq!(
            world.acoustics_at([0.0, 2.0, 0.0].into()),
            crate::audio::Acoustics::OUTDOORS
        );
    }

    #[test]
    fn a_sound_following_an_entity_moves_with_it() {
        let mut world = World::new();
        let mut m = crate::mesh::Mesh::new();
        let a = m.add_point((0.0, 0.0, 0.0)).unwrap();
        let b = m.add_point((1.0, 0.0, 0.0)).unwrap();
        let c = m.add_point((1.0, 0.0, 1.0)).unwrap();
        m.add_face(&[a, b, c]).unwrap();
        let id = world.spawn(m);
        let s = world.play_sound(&Sound::tone(200.0, 1.0), PlaySound::following(id).looped());
        world.tick_audio(0.016);
        assert!(world.audio().is_playing(s));
    }
}

#[cfg(test)]
mod cave_tests {
    use crate::advanced::Volume;
    use crate::VoxelTerrain;
    use crate::World;
    use glam::Vec3;

    #[test]
    fn a_hollow_in_the_ground_sounds_like_a_cave() {
        let mut terrain = VoxelTerrain::new(Volume::new(1.0));
        terrain.generate(Vec3::splat(-40.0), Vec3::splat(40.0), |p| {
            // Solid below y = 30, with a hollow ball of 10 m at the origin.
            let ground = (30.0 - p.y).clamp(-6.0, 6.0);
            ground.min((p.length() - 10.0).clamp(-6.0, 6.0))
        });
        let mut world = World::new();
        world.set_voxel_terrain(Some(terrain));
        let a = world.acoustics_at(Vec3::ZERO);
        assert!(a.reverb.size > 0.6, "{a:?}");
        let open = world.acoustics_at(Vec3::new(0.0, 33.0, 0.0));
        assert!(open.reverb.size < 0.3, "{open:?}");
    }
}
