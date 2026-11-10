//! Custom materials: surfaces whose look is a small WGSL function the game
//! writes. A lava pool that flows and glows, a hologram that flickers with
//! scan lines, and a force-field dome that ripples brighter while it is hit.
//! They are lit, shadowed and fogged like every other mesh.
//!
//! The force field's strength is a material param the game sets each frame.
use engine::prelude::*;

/// Flowing, glowing lava. `params[0].x` is how hot it glows.
const LAVA: &str = r#"
fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
    var o = base;
    let p = m.world_pos.xz * 0.8;
    let flow = sin(p.x * 3.0 + m.time * 0.7) * sin(p.y * 2.3 - m.time * 0.5)
        + 0.5 * sin((p.x + p.y) * 5.0 + m.time * 1.3);
    let hot = smoothstep(-0.2, 1.2, flow);
    o.color = vec4<f32>(mix(vec3<f32>(0.05, 0.02, 0.02), vec3<f32>(0.3, 0.06, 0.0), hot), 1.0);
    o.roughness = mix(0.9, 0.4, hot);
    o.metallic = 0.0;
    o.emission = mix(vec3<f32>(0.8, 0.12, 0.0), vec3<f32>(1.0, 0.65, 0.1), hot) * hot * m.params[0].x;
    return o;
}
"#;

/// A blue hologram: bright rims, scan lines moving up, a little flicker.
const HOLOGRAM: &str = r#"
fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
    var o = base;
    let rim = pow(1.0 - abs(dot(m.normal, m.view_dir)), 2.0);
    let scan = 0.5 + 0.5 * sin(m.world_pos.y * 18.0 - m.time * 6.0);
    let flicker = 0.85 + 0.15 * sin(m.time * 23.0);
    let glow = (0.25 + rim * 1.5 + scan * 0.35) * flicker;
    o.color = vec4<f32>(0.0, 0.0, 0.0, clamp(0.15 + rim * 0.7 + scan * 0.2, 0.0, 1.0));
    o.emission = vec3<f32>(0.2, 0.7, 1.0) * glow;
    o.roughness = 1.0;
    return o;
}
"#;

/// A force-field dome: see-through in the middle, bright at the edges,
/// rings running out from where it was hit. `params[0].x` is the hit
/// strength (0..1), `params[1].xyz` where it was hit.
const FORCE_FIELD: &str = r#"
fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
    var o = base;
    let rim = pow(1.0 - abs(dot(m.normal, m.view_dir)), 3.0);
    let d = distance(m.world_pos, m.params[1].xyz);
    let rings = max(sin(d * 9.0 - m.time * 8.0), 0.0) * exp(-d * 0.6) * m.params[0].x;
    let hex = abs(sin(m.object_pos.x * 6.0) * sin(m.object_pos.y * 6.0) * sin(m.object_pos.z * 6.0));
    let edge = smoothstep(0.85, 1.0, hex) * 0.3;
    let a = clamp(0.06 + rim * 0.6 + rings * 0.8 + edge, 0.0, 1.0);
    o.color = vec4<f32>(0.0, 0.0, 0.0, a);
    o.emission = vec3<f32>(0.4, 1.0, 0.6) * (rim * 1.2 + rings * 2.0 + edge);
    return o;
}
"#;

#[derive(Default)]
struct Game {
    field: Option<CustomMaterialId>,
    time: f32,
}

fn main() -> EngineResult<()> {
    let mut game = Game::default();
    Engine::run("materials", move |world, frame| {
        if frame.first {
            world.set_vsync(false);
            world.set_physical_sky(Some(PhysicalSky::at_hour(17.8)));
            world.spawn(Mesh::box_at(
                (0.0, -0.05, 0.0),
                (30.0, 0.1, 30.0),
                rgb(80, 78, 74),
            )?);

            // Each material is checked here: a WGSL mistake is an error
            // naming the line, before anything is drawn.
            let lava = world
                .add_material(CustomMaterial::new("lava", LAVA).param(0, [3.0, 0.0, 0.0, 0.0]))?;
            let hologram =
                world.add_material(CustomMaterial::new("hologram", HOLOGRAM).see_through())?;
            let field = world
                .add_material(CustomMaterial::new("force field", FORCE_FIELD).see_through())?;
            game.field = Some(field);

            let pool = world.spawn(Mesh::box_at(
                Vec3::ZERO,
                (5.0, 0.12, 5.0),
                rgb(255, 255, 255),
            )?);
            world.set_place(pool, Place::new(-4.0, 0.0, 0.0))?;
            world.set_custom_material(pool, lava)?;

            let figure = world.spawn(Mesh::cylinder(
                (0.0, 0.9, 0.0),
                Vec3::Y,
                0.45,
                1.8,
                32,
                rgb(255, 255, 255),
            )?);
            world.set_place(figure, Place::new(0.5, 0.0, 0.0))?;
            world.set_custom_material(figure, hologram)?;
            let head = world.spawn(Mesh::ball(Vec3::ZERO, 0.3, rgb(255, 255, 255))?);
            world.set_place(head, Place::new(0.5, 2.1, 0.0))?;
            world.set_custom_material(head, hologram)?;

            // A crate inside a dome.
            world.spawn(Shape::box_at(
                (4.5, 0.5, 0.0),
                (1.0, 1.0, 1.0),
                rgb(150, 110, 70),
            )?);
            let dome = world.spawn(Mesh::ball(Vec3::ZERO, 2.0, rgb(255, 255, 255))?);
            world.set_place(dome, Place::new(4.5, 0.0, 0.0))?;
            world.set_custom_material(dome, field)?;
        }
        game.time += frame.dt;

        // The dome is "hit" every two seconds: the game drives the ripple.
        if let Some(field) = game.field {
            let since_hit = game.time % 2.0;
            let strength = (1.0 - since_hit / 1.5).max(0.0);
            world.set_material_param(field, 0, [strength, 0.0, 0.0, 0.0])?;
            world.set_material_param(field, 1, [3.2, 1.2, 1.2, 0.0])?;
        }

        world.look_orbit(Vec3::new(0.3, 0.9, 0.0), 11.0, 75.0 + game.time * 6.0, 16.0);
        world.check(
            "three custom materials were added",
            world.custom_material_count() == 3,
        );
        world.report("custom materials", world.custom_material_count());
        Ok(())
    })
}
