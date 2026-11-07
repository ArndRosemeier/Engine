//! `World`'s colour grading and fog volumes (see `crate::grading`).

use super::{Haze, World};
use crate::error::{EngineError, EngineResult};
use crate::grading::{ColorGrade, FogVolume, FogVolumeId};

impl World {
    /// The look outside every fog volume (neutral by default).
    pub fn set_color_grade(&mut self, grade: ColorGrade) -> EngineResult<()> {
        grade.validate()?;
        self.grading.base = grade;
        Ok(())
    }

    pub fn color_grade(&self) -> ColorGrade {
        self.grading.base
    }

    /// A box with its own grade and/or fog; the camera blends into it.
    pub fn add_fog_volume(&mut self, volume: FogVolume) -> EngineResult<FogVolumeId> {
        self.grading.add(volume)
    }

    /// Change a volume (it keeps how far the camera has blended into it).
    pub fn set_fog_volume(&mut self, id: FogVolumeId, volume: FogVolume) -> EngineResult<()> {
        self.grading.set(id, volume)
    }

    pub fn fog_volume(&self, id: FogVolumeId) -> Option<&FogVolume> {
        self.grading.get(id)
    }

    pub fn remove_fog_volume(&mut self, id: FogVolumeId) -> bool {
        self.grading.remove(id)
    }

    pub fn fog_volume_count(&self) -> usize {
        self.grading.count()
    }

    /// Seconds a zone change takes to settle (about 95% there); 2.5 by
    /// default, 0 for instant.
    pub fn set_zone_transition(&mut self, seconds: f32) -> EngineResult<()> {
        if !seconds.is_finite() || !(0.0..=60.0).contains(&seconds) {
            return Err(EngineError::InvalidValue(
                "zone transition must be within 0..=60 seconds".into(),
            ));
        }
        self.grading.transition_s = seconds;
        Ok(())
    }

    pub fn zone_transition(&self) -> f32 {
        self.grading.transition_s
    }

    /// How far the camera has blended into a volume right now (0..1).
    pub fn fog_volume_blend(&self, id: FogVolumeId) -> f32 {
        match self.grading.get(id) {
            Some(_) => self.grading.weights[id.0 as usize],
            None => 0.0,
        }
    }

    /// The grade on screen now: the base with the volumes blended in.
    pub fn screen_grade(&self) -> ColorGrade {
        self.grading.compose(self.haze()).0
    }

    /// The haze on screen now: [`Self::haze`] with the volumes' fog blended in.
    pub fn screen_haze(&self) -> Option<Haze> {
        self.grading.compose(self.haze()).1.haze()
    }

    /// Ease the volumes toward the camera's place. Runs every frame after
    /// the sky (call it yourself only when driving the world by hand).
    pub fn tick_grading(&mut self, dt: f32) {
        let eye = self.camera().eye();
        self.grading.tick(eye, dt);
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn the_screen_look_follows_the_camera_into_a_volume() {
        let mut world = World::new();
        world.set_haze(Some(Haze::new(rgb(200, 210, 230), 3000.0)));
        world.set_zone_transition(1.0).unwrap();
        let id = world
            .add_fog_volume(
                FogVolume::new((0.0, 0.0, 0.0), (20.0, 20.0, 20.0))
                    .grade(ColorGrade::smog())
                    .fog(rgb(110, 95, 70), 60.0)
                    .blend(5.0),
            )
            .unwrap();
        world.look_first_person((100.0, 5.0, 0.0), 0.0, 0.0);
        world.tick_grading(1.0 / 60.0);
        assert_eq!(world.fog_volume_blend(id), 0.0);
        assert_eq!(world.screen_grade(), ColorGrade::NEUTRAL);
        world.look_first_person((0.0, 5.0, 0.0), 0.0, 0.0);
        world.tick_grading(1.0 / 60.0);
        let early = world.fog_volume_blend(id);
        assert!(early > 0.0 && early < 0.1, "{early}");
        for _ in 0..120 {
            world.tick_grading(1.0 / 60.0);
        }
        assert!(world.fog_volume_blend(id) > 0.99);
        let haze = world.screen_haze().unwrap();
        assert!(
            (haze.visibility_m - 60.0).abs() < 2.0,
            "{}",
            haze.visibility_m
        );
        assert!(world.remove_fog_volume(id));
        assert_eq!(world.screen_grade(), ColorGrade::NEUTRAL);
    }
}
