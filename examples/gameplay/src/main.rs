//! Gameplay physics in one small park: a walking character, a hinged door
//! it shoves open, stairs it climbs, coins picked up by trigger zones, and a
//! lamp swinging on a rope. On top: a HUD (coin count, a bar, a pause
//! menu), names hanging over things and "+1" floating up from coins.
//!
//! Controls: WASD walks, Space jumps, Q/E turn the camera, P pauses, Esc quits.
//!
//! `ENGINE_GAMEPLAY_DEMO=1` (on in verify runs) walks the route by itself:
//! through the door, up the stairs, picking up every coin.
use engine::prelude::*;

/// A coin: picked up when the player walks into its trigger.
struct Coin;

/// The demo route: points the player walks to, one after the other.
const ROUTE: [(f32, f32); 4] = [(0.0, 6.0), (1.5, 7.0), (7.0, 7.0), (7.0, 9.5)];

fn solid(world: &mut World, center: Vec3, size: Vec3, color: Color) -> EngineResult<EntityId> {
    let id = world.spawn(Mesh::box_at(Vec3::ZERO, size, color)?);
    world.set_place(id, Place::new(center.x, center.y, center.z))?;
    world.add_body(id, Body::fixed().shape(BodyShape::Box(size * 0.5)))?;
    Ok(id)
}

#[derive(Default)]
struct Game {
    player: Option<EntityId>,
    door: Option<EntityId>,
    lamp: Option<EntityId>,
    yaw: f32,
    coins: u32,
    widest_door: f32,
    highest: f32,
    lowest: f32,
    leg: usize,
    jumps: u32,
    paused: bool,
    most_labels: usize,
    floated: bool,
}

fn main() -> EngineResult<()> {
    let demo = std::env::var("ENGINE_GAMEPLAY_DEMO")
        .map(|v| v != "0")
        .unwrap_or_else(|_| std::env::var_os("ENGINE_VERIFY").is_some());
    let mut game = Game {
        lowest: f32::MAX,
        ..Game::default()
    };
    let mut frame_no = 0u32;

    Engine::run("gameplay", move |world, frame| {
        if frame.first {
            world.set_vsync(false);
            world.set_physical_sky(Some(PhysicalSky::at_hour(15.5)));
            // The floor you see; physics has a flat floor at y = 0 already.
            world.spawn(Mesh::box_at(
                (0.0, -0.05, 0.0),
                (40.0, 0.1, 40.0),
                rgb(96, 124, 76),
            )?);

            // A wall with a doorway at z = 2, and a door hinged on its left edge.
            let stone = rgb(170, 160, 146);
            solid(
                world,
                Vec3::new(-3.4, 1.5, 2.0),
                Vec3::new(5.2, 3.0, 0.3),
                stone,
            )?;
            solid(
                world,
                Vec3::new(3.4, 1.5, 2.0),
                Vec3::new(5.2, 3.0, 0.3),
                stone,
            )?;
            solid(
                world,
                Vec3::new(0.0, 2.75, 2.0),
                Vec3::new(1.6, 0.5, 0.3),
                stone,
            )?;
            let door = world.spawn(Mesh::box_at(
                Vec3::ZERO,
                (1.56, 2.4, 0.08),
                rgb(130, 86, 50),
            )?);
            world.set_place(door, Place::new(0.0, 1.22, 2.0))?;
            world.add_body(
                door,
                Body::dynamic()
                    .density(0.4)
                    .shape(BodyShape::Box(Vec3::new(0.78, 1.2, 0.04))),
            )?;
            world.pin(
                door,
                Joint::hinge(Vec3::Y)
                    .at((-0.8, 1.22, 2.0))
                    .limits(-100.0, 100.0),
            )?;
            world.set_label(door, Label::new("Door").above(1.6))?;
            game.door = Some(door);

            // Stairs up to a terrace: steps of 0.25 m a character climbs.
            let wood = rgb(176, 134, 90);
            for i in 0..6 {
                let h = (i + 1) as f32 * 0.25;
                solid(
                    world,
                    Vec3::new(2.25 + i as f32 * 0.5, h * 0.5, 7.0),
                    Vec3::new(0.5, h, 2.0),
                    wood,
                )?;
            }
            let terrace = solid(
                world,
                Vec3::new(7.0, 0.75, 8.0),
                Vec3::new(4.0, 1.5, 4.0),
                wood,
            )?;
            world.set_label(
                terrace,
                Label::new("Terrace").color(rgb(255, 225, 150)).above(1.6),
            )?;

            // Coins along the route, each with a trigger zone.
            for at in [
                (0.0, 0.8, -3.0),
                (0.0, 0.8, 4.5),
                (7.0, 2.3, 7.0),
                (7.0, 2.3, 9.5),
            ] {
                let coin = world.spawn(Mesh::cylinder(
                    Vec3::ZERO,
                    Vec3::Z,
                    0.3,
                    0.06,
                    24,
                    rgb(240, 200, 40),
                )?);
                world.set_place(coin, Place::new(at.0, at.1, at.2))?;
                world.add_trigger(coin, Trigger::ball(0.6))?;
                world.insert(coin, Coin)?;
            }

            // A lamp on a 2 m rope, let go sideways so it swings.
            let lamp = world.spawn(Mesh::ball(Vec3::ZERO, 0.25, rgb(250, 230, 160))?);
            world.set_place(lamp, Place::new(-2.0, 4.5, -2.0))?;
            world.add_body(lamp, Body::dynamic().shape(BodyShape::Ball(0.25)))?;
            world.pin(lamp, Joint::rope(2.0).at((-4.0, 4.5, -2.0)))?;
            game.lamp = Some(lamp);

            // The player: a walking character whose feet are its place.
            let player = world.spawn(Mesh::box_at(
                (0.0, 0.9, 0.0),
                (0.6, 1.8, 0.6),
                rgb(60, 110, 200),
            )?);
            world.set_place(player, Place::new(0.0, 0.0, -8.0))?;
            world.add_character(player, Character::default())?;
            world.set_label(player, Label::new("You").size(16.0))?;
            game.player = Some(player);
        }
        frame_no += 1;
        let Some(player) = game.player else {
            return Ok(());
        };
        let feet = world.world_transform(player)?.w_axis.truncate();

        // P opens and closes the pause menu; while it is open nothing moves.
        if frame.input.pressed(Key::P) {
            game.paused = !game.paused;
        }
        if game.paused {
            match frame.ui.menu("Paused", &["Resume", "Quit"]) {
                Some(0) => game.paused = false,
                Some(1) => world.request_exit(),
                _ => {}
            }
            world.walk(player, Vec3::ZERO)?;
            return Ok(());
        }

        // Walk where the keys (or the demo route) say.
        game.yaw += frame.input.yaw_sign() * 90.0 * frame.dt;
        let mut walk = frame.input.move_dir_xz(game.yaw) * 4.0;
        let mut jump = frame.input.pressed(Key::Space);
        if demo {
            walk = Vec3::ZERO;
            if let Some(&(x, z)) = ROUTE.get(game.leg) {
                let to = Vec3::new(x - feet.x, 0.0, z - feet.z);
                if to.length() < 0.3 {
                    game.leg += 1;
                } else {
                    walk = to.normalize() * 4.0;
                }
            } else if game.jumps == 0 {
                jump = true;
            }
        }
        world.walk(player, walk)?;
        if jump && world.jump(player, 5.0)? {
            game.jumps += 1;
        }

        // Coins: picked up when the player enters their trigger.
        for e in world.take_trigger_events() {
            if e.entered && e.other == player && world.has::<Coin>(e.trigger) {
                let at = world.world_transform(e.trigger)?.w_axis.truncate();
                world.float_text(at, "+1", rgb(255, 215, 60));
                world.despawn(e.trigger);
                game.coins += 1;
            }
        }

        // Follow camera, pulled in when a wall is between it and the player.
        world.look_follow_entity(player, game.yaw, 6.0, 3.0)?;

        // The HUD: text in the corners, a bar, a hint at the bottom.
        frame
            .ui
            .text(Anchor::TopLeft, format!("Coins: {} / 4", game.coins));
        frame.ui.bar(
            Anchor::TopLeft,
            "Route",
            game.leg as f32,
            ROUTE.len() as f32,
            rgb(90, 170, 230),
        );
        frame.ui.text_styled(
            Anchor::Bottom,
            "WASD walk, Space jump, Q/E turn, P pause",
            16.0,
            rgb(230, 230, 230),
        );

        // Self-checks for verify runs (`python tools/verify.py gameplay --frames 600`).
        game.highest = game.highest.max(feet.y);
        game.lowest = game.lowest.min(feet.y);
        if let Some(door) = game.door {
            let edge = world
                .entity(door)?
                .transform()
                .transform_point3(Vec3::X * 0.78);
            let angle = (edge.z - 2.0).atan2(edge.x + 0.8).to_degrees().abs();
            game.widest_door = game.widest_door.max(angle);
        }
        world.check("the player never sinks into the floor", feet.y > -0.05);
        // Labels drawn last frame, and floating text after a pickup.
        game.most_labels = game.most_labels.max(world.labels_on_screen());
        game.floated |= world.float_text_count() > 0;
        if let Some(lamp) = game.lamp {
            let p = world.entity(lamp)?.transform().w_axis.truncate();
            world.check(
                "the lamp's rope never stretches",
                p.distance(Vec3::new(-4.0, 4.5, -2.0)) < 2.1,
            );
        }
        if demo && frame_no == 600 {
            world.check("the demo picks up every coin", game.coins == 4);
            world.check("the player climbs the stairs", game.highest > 1.45);
            world.check(
                "walking through pushes the door open",
                game.widest_door > 30.0,
            );
            world.check("the player jumped", game.jumps == 1);
            world.check("labels show over the scene", game.most_labels >= 2);
            world.check("a picked-up coin floats +1", game.floated);
        }
        world.report("coins", game.coins);
        world.report("highest feet (m)", game.highest);
        world.report("lowest feet (m)", game.lowest);
        world.report("door opened (degrees)", game.widest_door);
        world.report("labels on screen", world.labels_on_screen());
        world.report(
            "on the ground",
            world.character(player).is_some_and(|c| c.grounded),
        );
        Ok(())
    })
}
