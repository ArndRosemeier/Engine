//! Reflections: a rain-wet street between glass towers at dusk, chrome
//! spheres, and a polished hall with a reflection probe.
//!
//! The street mirrors the towers and lamps (screen-space reflections); the
//! hall's floor and its chrome ball show the hall itself, walls included,
//! from a probe captured once when the scene starts.
//!
//! Controls: right mouse + move to look, WASD to fly, Shift fast.
//! - R: screen-space reflections on/off.
//! - P: the hall's reflection probe on/off.
//!
//! `ENGINE_REFLECTIONS_VIEW=hall` starts inside the hall,
//! `ENGINE_REFLECTIONS_SSR=0` without screen-space reflections.
use engine::prelude::*;
use glam::Vec3;

fn setting(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// A clean surface with the given gloss (no stone detail on top).
fn finish(roughness: f32, metallic: f32) -> SurfaceMaterial {
    SurfaceMaterial {
        roughness,
        metallic,
        strata_strength: 0.0,
        warp_strength: 0.0,
        noise_gain: 0.0,
        ..SurfaceMaterial::METAL
    }
}

fn slab(center: Vec3, size: Vec3, color: Color, material: SurfaceMaterial) -> EngineResult<Mesh> {
    let mut m = Mesh::new();
    m.add_box(center, size, color)?;
    m.set_surface_material(material);
    Ok(m)
}

fn ball(center: Vec3, radius: f32, color: Color, material: SurfaceMaterial) -> EngineResult<Mesh> {
    let mut m = Mesh::ball(center, radius, color)?;
    m.set_surface_material(material);
    Ok(m)
}

/// The hall: a polished floor, walls in three colours, an open front.
const HALL: Vec3 = Vec3::new(0.0, 0.0, 70.0);
const HALL_HALF: Vec3 = Vec3::new(9.0, 3.5, 7.0);

fn build(world: &mut World) -> EngineResult<ReflectionProbeId> {
    // The street: dark wet asphalt, rougher pavements either side.
    world.place(
        slab(
            Vec3::new(0.0, -0.05, 20.0),
            Vec3::new(14.0, 0.1, 160.0),
            rgb(32, 33, 36),
            finish(0.06, 0.0),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    for side in [-1.0f32, 1.0] {
        world.place(
            slab(
                // The pavements end where the hall begins.
                Vec3::new(side * 10.0, 0.05, 1.0),
                Vec3::new(6.0, 0.3, 122.0),
                rgb(120, 118, 112),
                finish(0.8, 0.0),
            )?,
            Place::new(0.0, 0.0, 0.0),
        )?;
    }
    // Towers: dark glass curtain walls and pale concrete.
    let mut z = -40.0;
    let mut k = 0u32;
    while z < 50.0 {
        for side in [-1.0f32, 1.0] {
            k += 1;
            let h = 18.0 + ((k * 37) % 5) as f32 * 9.0;
            let w = 10.0 + ((k * 13) % 3) as f32 * 2.0;
            let glass = k % 3 != 0;
            let (color, material) = if glass {
                (rgb(40, 62, 80), finish(0.04, 0.55))
            } else {
                (rgb(196, 188, 172), finish(0.7, 0.0))
            };
            world.place(
                slab(
                    Vec3::new(side * (13.0 + w * 0.5), h * 0.5, z),
                    Vec3::new(w, h, 14.0),
                    color,
                    material,
                )?,
                Place::new(0.0, 0.0, 0.0),
            )?;
            // Lit floors on the concrete towers.
            if !glass {
                for f in 0..(h / 3.5) as i32 {
                    let y = 2.0 + f as f32 * 3.5;
                    world.place(
                        slab(
                            Vec3::new(side * 12.95, y, z),
                            Vec3::new(0.12, 1.2, 12.0),
                            rgb(255, 214, 150),
                            SurfaceMaterial {
                                emission: 2.5,
                                ..finish(0.5, 0.0)
                            },
                        )?,
                        Place::new(0.0, 0.0, 0.0),
                    )?;
                }
            }
        }
        z += 16.0;
    }
    // Street lamps down both sides.
    let mut z = -36.0;
    while z < 56.0 {
        for side in [-1.0f32, 1.0] {
            let x = side * 7.6;
            world.place(
                slab(
                    Vec3::new(x, 2.6, z),
                    Vec3::new(0.16, 5.2, 0.16),
                    rgb(50, 52, 56),
                    finish(0.4, 0.9),
                )?,
                Place::new(0.0, 0.0, 0.0),
            )?;
            world.place(
                ball(
                    Vec3::new(x, 5.3, z),
                    0.3,
                    rgb(255, 220, 170),
                    SurfaceMaterial {
                        emission: 6.0,
                        ..finish(0.5, 0.0)
                    },
                )?,
                Place::new(0.0, 0.0, 0.0),
            )?;
            world.add_lamp(Lamp::street_lamp(Vec3::new(x, 5.0, z)))?;
        }
        z += 12.0;
    }
    // Chrome in the street: spheres and a block like a parked car.
    for (i, (x, z)) in [(-3.0f32, 6.0f32), (2.5, 14.0), (-1.0, 26.0)]
        .into_iter()
        .enumerate()
    {
        let r = 0.7 + i as f32 * 0.25;
        world.place(
            ball(Vec3::new(x, r, z), r, rgb(235, 236, 240), finish(0.03, 1.0))?,
            Place::new(0.0, 0.0, 0.0),
        )?;
    }
    world.place(
        slab(
            Vec3::new(4.0, 0.75, 32.0),
            Vec3::new(1.9, 1.1, 4.4),
            rgb(150, 20, 24),
            finish(0.12, 0.7),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;

    // The hall at the end of the street: open toward the street (-z).
    let c = HALL;
    let hh = HALL_HALF;
    let floor = finish(0.05, 0.0);
    world.place(
        slab(
            // A little deeper than it shows, so its underside sits below the
            // walls' and never fights with them.
            c + Vec3::new(0.0, 0.04, 0.0),
            Vec3::new(hh.x * 2.0, 0.12, hh.z * 2.0),
            rgb(200, 196, 188),
            floor,
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    let wall =
        |center: Vec3, size: Vec3, color: Color| slab(center, size, color, finish(0.85, 0.0));
    world.place(
        wall(
            // The back wall stops at the side walls instead of overlapping them.
            c + Vec3::new(0.0, hh.y, hh.z),
            Vec3::new(hh.x * 2.0 - 0.3, hh.y * 2.0, 0.3),
            rgb(170, 40, 36),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    world.place(
        wall(
            c + Vec3::new(-hh.x, hh.y, 0.05),
            Vec3::new(0.3, hh.y * 2.0, hh.z * 2.0 + 0.2),
            rgb(40, 110, 170),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    world.place(
        wall(
            c + Vec3::new(hh.x, hh.y, 0.05),
            Vec3::new(0.3, hh.y * 2.0, hh.z * 2.0 + 0.2),
            rgb(220, 180, 60),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    world.place(
        wall(
            c + Vec3::new(0.0, hh.y * 2.0, 0.0),
            Vec3::new(hh.x * 2.0, 0.3, hh.z * 2.0),
            rgb(230, 228, 222),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    world.place(
        ball(
            c + Vec3::new(0.0, 1.2, 1.5),
            1.2,
            rgb(235, 236, 240),
            finish(0.03, 1.0),
        )?,
        Place::new(0.0, 0.0, 0.0),
    )?;
    for x in [-5.0f32, 5.0] {
        world.place(
            slab(
                c + Vec3::new(x, 1.0, 3.0),
                Vec3::new(1.4, 2.0, 1.4),
                rgb(80, 160, 90),
                finish(0.6, 0.0),
            )?,
            Place::new(0.0, 0.0, 0.0),
        )?;
    }
    for x in [-4.0f32, 0.0, 4.0] {
        world.add_lamp(Lamp::point(
            c + Vec3::new(x, hh.y * 2.0 - 0.5, 0.0),
            rgb(255, 238, 210),
            6.0,
            10.0,
        ))?;
    }
    world.add_reflection_probe(ReflectionProbe::new(c + Vec3::new(0.0, hh.y, 0.0), hh).blend(0.5))
}

fn main() -> EngineResult<()> {
    let hall = setting("ENGINE_REFLECTIONS_VIEW").as_deref() == Some("hall");
    let mut ssr = setting("ENGINE_REFLECTIONS_SSR").as_deref() != Some("0");
    let mut camera = if hall {
        FlyCamera::new((5.0, 1.8, 65.0), -38.0, -10.0)
    } else {
        FlyCamera::new((1.5, 1.6, -8.0), 0.0, -4.0)
    }
    .with_speed(10.0);
    let mut probe: Option<ReflectionProbeId> = None;

    Engine::run("reflections", move |world, frame| {
        // Uncapped so the title bar shows the real frame rate.
        world.set_vsync(false);
        if frame.first {
            world.set_physical_sky(Some(PhysicalSky::at_hour(18.6)));
            probe = Some(build(world)?);
            if !ssr {
                world.set_screen_space_reflections(None)?;
            }
        }
        camera.update(world, frame);

        if frame.input.pressed(Key::R) {
            ssr = !ssr;
            world.set_screen_space_reflections(ssr.then(ScreenSpaceReflections::default))?;
        }
        if frame.input.pressed(Key::P) {
            match probe.take() {
                Some(id) => {
                    world.remove_reflection_probe(id);
                }
                None => {
                    probe = Some(
                        world.add_reflection_probe(
                            ReflectionProbe::new(
                                HALL + Vec3::new(0.0, HALL_HALF.y, 0.0),
                                HALL_HALF,
                            )
                            .blend(0.5),
                        )?,
                    );
                }
            }
        }

        engine::egui::Window::new("reflections")
            .default_pos((12.0, 12.0))
            .resizable(false)
            .show(frame.ui.ctx(), |ui| {
                ui.label(format!("{:.0} FPS", frame.fps));
                ui.label(format!(
                    "R: screen-space reflections {}   P: hall probe {}",
                    if ssr { "on" } else { "off" },
                    if probe.is_some() { "on" } else { "off" },
                ));
            });
        Ok(())
    })
}
