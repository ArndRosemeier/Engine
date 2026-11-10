//! Gem hunt: walk a walled arena, pick up five glowing gems, keep away from
//! the patrolling enemy (a touch costs 25 health). When every gem is taken a
//! door in the north wall swings open and "You win" shows.
//!
//! Controls: WASD walks, A/D (or Q/E) turn the camera, Space jumps, Esc quits.
//!
//! `ENGINE_GEM_HUNT_DEMO=1` (on in verify runs) plays by itself: it picks up
//! the gems, walks into the enemy once, jumps and leaves through the door.
use engine::prelude::*;

/// A gem: picked up when the player walks into its trigger.
struct Gem {
    /// Where it hovers (it bobs and spins around this point).
    home: Vec3,
}

/// The gems' look: the lit base colour plus a pulsing emission.
/// params[0].x = glow strength, params[1].xyz = glow colour.
const GLOW: &str = r#"
fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
    var o = base;
    let pulse = 0.6 + 0.4 * sin(m.time * 4.0 + m.world_pos.x + m.world_pos.z);
    let rim = pow(1.0 - abs(dot(normalize(m.normal), normalize(m.view_dir))), 2.0);
    o.emission = m.params[1].xyz * (pulse + rim * 1.5) * m.params[0].x;
    o.roughness = 0.15;
    o.metallic = 0.3;
    return o;
}
"#;

const ARENA: f32 = 10.0; // half size of the arena, metres
const WALL_H: f32 = 2.5;
const DOOR_W: f32 = 2.0;
const GEMS: [(f32, f32); 5] = [
    (-6.0, -6.0),
    (6.0, -6.0),
    (6.0, 4.0),
    (-6.0, 4.0),
    (0.0, 7.0),
];
const PATROL_X: f32 = 7.0; // the enemy walks x = -7 .. 7 at z = 0
const ENEMY_SPEED: f32 = 3.0;
const PLAYER_SPEED: f32 = 6.0;
const DOOR_OPEN_DEG: f32 = -100.0; // swings outwards (towards +Z)

/// One step of the demo route.
#[derive(Clone, Copy)]
enum Step {
    Go(f32, f32),
    /// Walk into the enemy until it hits.
    Chase,
    Jump,
    /// Wait until the door stands open.
    WaitDoor,
    /// Turn around to look back at the arena.
    LookBack,
}

const ROUTE: [Step; 12] = [
    Step::Go(-6.0, -6.0),
    Step::Jump,
    Step::Go(6.0, -6.0),
    Step::Chase,
    Step::Go(6.0, 4.0),
    Step::Go(-6.0, 4.0),
    Step::Go(0.0, 7.0),
    Step::WaitDoor,
    Step::Go(0.0, 9.0),
    Step::Go(0.0, 13.0),
    Step::LookBack,
    Step::Go(0.0, 13.0),
];

fn wall(world: &mut World, center: Vec3, size: Vec3, color: Color) -> EngineResult<EntityId> {
    let id = world.spawn(Mesh::box_at(Vec3::ZERO, size, color)?);
    world.set_place(id, Place::new(center.x, center.y, center.z))?;
    world.add_body(id, Body::fixed().shape(BodyShape::Box(size * 0.5)))?;
    Ok(id)
}

#[derive(Default)]
struct Game {
    player: Option<EntityId>,
    enemy: Option<EntityId>,
    door: Option<EntityId>,
    glow: Option<CustomMaterialId>,
    yaw: f32,
    health: f32,
    gems: u32,
    hits: u32,
    enemy_x: f32,
    enemy_dir: f32,
    door_angle: f32,
    step: usize,
    jumped: bool,
    highest: f32,
    won_at: Option<f32>,
    lowest_health: f32,
}

fn main() -> EngineResult<()> {
    let demo = std::env::var("ENGINE_GEM_HUNT_DEMO")
        .map(|v| v != "0")
        .unwrap_or_else(|_| std::env::var_os("ENGINE_VERIFY").is_some());
    let mut game = Game {
        health: 100.0,
        lowest_health: 100.0,
        enemy_x: -PATROL_X,
        enemy_dir: 1.0,
        ..Game::default()
    };
    let mut frame_no = 0u32;

    Engine::run("gem hunt", move |world, frame| {
        if frame.first {
            world.set_vsync(false);
            world.set_physical_sky(Some(PhysicalSky::at_hour(16.0)));
            // Floor (physics already has a flat floor at y = 0).
            world.spawn(Mesh::box_at(
                (0.0, -0.05, 0.0),
                (60.0, 0.1, 60.0),
                rgb(92, 120, 74),
            )?);
            world.spawn(Mesh::box_at(
                (0.0, 0.005, 0.0),
                (ARENA * 2.0, 0.01, ARENA * 2.0),
                rgb(150, 140, 120),
            )?);

            // Walls; the north one has a doorway DOOR_W wide at x = 0.
            let stone = rgb(165, 155, 140);
            let t = 0.4;
            let h = WALL_H;
            wall(
                world,
                Vec3::new(0.0, h / 2.0, -ARENA),
                Vec3::new(ARENA * 2.0 + t, h, t),
                stone,
            )?;
            wall(
                world,
                Vec3::new(-ARENA, h / 2.0, 0.0),
                Vec3::new(t, h, ARENA * 2.0),
                stone,
            )?;
            wall(
                world,
                Vec3::new(ARENA, h / 2.0, 0.0),
                Vec3::new(t, h, ARENA * 2.0),
                stone,
            )?;
            let side = ARENA - DOOR_W / 2.0;
            for sx in [-1.0, 1.0] {
                wall(
                    world,
                    Vec3::new(sx * (DOOR_W / 2.0 + side / 2.0), h / 2.0, ARENA),
                    Vec3::new(side + t / 2.0, h, t),
                    stone,
                )?;
            }

            // The door: mesh built from its hinge edge, so the entity's
            // origin is the hinge and turning its yaw swings it.
            let door = world.spawn(Mesh::box_at(
                (DOOR_W / 2.0, h / 2.0, 0.0),
                (DOOR_W - 0.04, h - 0.02, 0.15),
                rgb(120, 78, 44),
            )?);
            world.set_place(door, Place::new(-DOOR_W / 2.0, 0.0, ARENA))?;
            world.add_body(door, Body::kinematic())?;
            game.door = Some(door);

            // Gems: one glowing material shared by all five.
            let glow = world.add_material(
                CustomMaterial::new("gem_glow", GLOW)
                    .param(0, [1.0, 0.0, 0.0, 0.0])
                    .param(1, [0.3, 1.0, 0.8, 0.0]),
            )?;
            game.glow = Some(glow);
            for (x, z) in GEMS {
                let gem = world.spawn(Mesh::ball(Vec3::ZERO, 0.3, rgb(60, 200, 170))?);
                let home = Vec3::new(x, 0.9, z);
                world.set_place(gem, Place::new(home.x, home.y, home.z))?;
                world.set_custom_material(gem, glow)?;
                world.add_trigger(gem, Trigger::ball(0.7))?;
                world.insert(gem, Gem { home })?;
            }

            // The enemy: no body (it walks through you), a trigger that
            // follows it reports the touch.
            let enemy = world.spawn(Mesh::box_at(
                (0.0, 0.7, 0.0),
                (1.0, 1.4, 1.0),
                rgb(200, 40, 40),
            )?);
            world.set_place(enemy, Place::new(game.enemy_x, 0.0, 0.0))?;
            world.add_trigger(enemy, Trigger::ball(0.9))?;
            world.set_label(
                enemy,
                Label::new("Enemy").color(rgb(255, 120, 100)).above(1.8),
            )?;
            game.enemy = Some(enemy);

            // The player.
            let player = world.spawn(Mesh::box_at(
                (0.0, 0.9, 0.0),
                (0.6, 1.8, 0.6),
                rgb(60, 110, 200),
            )?);
            world.set_place(player, Place::new(0.0, 0.0, -6.0))?;
            world.add_character(player, Character::default())?;
            game.player = Some(player);
        }
        frame_no += 1;
        let (Some(player), Some(enemy), Some(door)) = (game.player, game.enemy, game.door) else {
            return Ok(());
        };
        let feet = world.world_transform(player)?.w_axis.truncate();
        let alive = game.health > 0.0;
        let all_gems = game.gems as usize == GEMS.len();

        // Enemy patrol: back and forth along x at z = 0.
        game.enemy_x += game.enemy_dir * ENEMY_SPEED * frame.dt;
        if game.enemy_x.abs() > PATROL_X {
            game.enemy_x = game.enemy_x.clamp(-PATROL_X, PATROL_X);
            game.enemy_dir = -game.enemy_dir;
        }
        let enemy_yaw = if game.enemy_dir > 0.0 { 90.0 } else { -90.0 };
        world.set_place(
            enemy,
            Place::new(game.enemy_x, 0.0, 0.0).with_yaw_deg(enemy_yaw),
        )?;

        // Input, or the demo route.
        game.yaw += frame.input.yaw_sign() * 90.0 * frame.dt;
        let mut walk = frame.input.move_dir_xz(game.yaw) * PLAYER_SPEED;
        let mut jump = frame.input.pressed(Key::Space);
        if demo {
            walk = Vec3::ZERO;
            jump = false;
            let goal = match ROUTE.get(game.step) {
                Some(Step::Go(x, z)) => Some(Vec3::new(*x, 0.0, *z)),
                Some(Step::Chase) => {
                    if game.hits > 0 {
                        game.step += 1;
                        None
                    } else {
                        Some(Vec3::new(game.enemy_x, 0.0, 0.0))
                    }
                }
                Some(Step::Jump) => {
                    jump = true;
                    None
                }
                Some(Step::WaitDoor) => {
                    if game.door_angle <= DOOR_OPEN_DEG + 1.0 {
                        game.step += 1;
                    }
                    None
                }
                Some(Step::LookBack) => {
                    game.step += 1;
                    None
                }
                None => None,
            };
            if let Some(goal) = goal {
                let to = Vec3::new(goal.x - feet.x, 0.0, goal.z - feet.z);
                if to.length() < 0.3 {
                    if game.step + 1 < ROUTE.len() {
                        game.step += 1;
                    }
                } else {
                    walk = to.normalize() * PLAYER_SPEED;
                }
            }
            // The camera turns towards where the player walks, or looks back
            // at the arena at the end.
            let looking_back = game.step + 1 >= ROUTE.len();
            let want = if looking_back {
                180.0
            } else if walk.length() > 0.1 {
                walk.x.atan2(walk.z).to_degrees()
            } else {
                game.yaw
            };
            let diff = (want - game.yaw + 540.0).rem_euclid(360.0) - 180.0;
            game.yaw += diff * (3.0 * frame.dt).min(1.0);
        }
        if !alive {
            walk = Vec3::ZERO;
            jump = false;
        }
        world.walk(player, walk)?;
        if jump && world.jump(player, 5.0)? {
            game.jumped = true;
            if demo {
                game.step += 1;
            }
        }

        // Pickups and enemy touches.
        for e in world.take_trigger_events() {
            if !e.entered || e.other != player {
                continue;
            }
            if world.has::<Gem>(e.trigger) {
                let at = world.world_transform(e.trigger)?.w_axis.truncate();
                world.float_text(at, "+1 gem", rgb(120, 255, 220));
                world.despawn(e.trigger);
                game.gems += 1;
            } else if e.trigger == enemy && alive {
                game.health = (game.health - 25.0).max(0.0);
                game.hits += 1;
                world.float_text(feet + Vec3::Y * 2.0, "-25", rgb(255, 80, 60));
            }
        }

        // Gems spin and bob; the glow breathes a little.
        for gem in world.ids_with::<Gem>() {
            if let Some(home) = world.get::<Gem>(gem).map(|g| g.home) {
                let bob = (frame.time * 2.0 + home.x).sin() * 0.12;
                world.set_place(
                    gem,
                    Place::new(home.x, home.y + bob, home.z)
                        .with_yaw_deg(frame.time * 90.0)
                        .with_pitch_deg(30.0),
                )?;
            }
        }
        if let Some(glow) = game.glow {
            world.set_material_param(
                glow,
                0,
                [4.0 + 1.5 * (frame.time * 2.0).sin(), 0.0, 0.0, 0.0],
            )?;
        }

        // The door swings open once every gem is taken.
        if all_gems {
            game.door_angle = (game.door_angle - 70.0 * frame.dt).max(DOOR_OPEN_DEG);
            if game.won_at.is_none() {
                game.won_at = Some(frame.time);
            }
        }
        world.set_place(
            door,
            Place::new(-DOOR_W / 2.0, 0.0, ARENA).with_yaw_deg(game.door_angle),
        )?;

        // Follow camera, pulled in when a wall is between it and the player.
        // The fat ray starts above the head: started inside the player's
        // own capsule it would hit the player at distance 0.
        let (dist, height) = (7.0, 3.5);
        let top = feet + Vec3::Y * 2.2;
        let back = Vec3::new(
            -game.yaw.to_radians().sin(),
            0.0,
            -game.yaw.to_radians().cos(),
        );
        let eye = feet + back * dist + Vec3::Y * height;
        let want = eye - top;
        let room = world
            .cast_sphere(top, want, 0.3, want.length())
            .map_or(1.0, |hit| hit.distance / want.length());
        world.look_follow(feet, game.yaw, (dist * room).max(1.5), height);
        world.report("camera room", room);

        // HUD.
        frame.ui.text(
            Anchor::TopLeft,
            format!("Gems: {} / {}", game.gems, GEMS.len()),
        );
        frame.ui.bar(
            Anchor::TopLeft,
            "Health",
            game.health,
            100.0,
            rgb(200, 60, 60),
        );
        frame.ui.text_styled(
            Anchor::Bottom,
            "WASD walk, A/D turn, Space jump",
            16.0,
            rgb(230, 230, 230),
        );
        if !alive {
            frame
                .ui
                .text_styled(Anchor::Center, "You lost", 64.0, rgb(255, 90, 70));
        } else if all_gems {
            frame
                .ui
                .text_styled(Anchor::Center, "You win!", 64.0, rgb(255, 220, 90));
            frame
                .ui
                .text_styled(Anchor::Center, "The door is open", 24.0, rgb(240, 240, 240));
        }

        // Self-checks.
        game.highest = game.highest.max(feet.y);
        game.lowest_health = game.lowest_health.min(game.health);
        world.check("the player never sinks into the floor", feet.y > -0.05);
        world.check(
            "health stays within 0..100",
            (0.0..=100.0).contains(&game.health),
        );
        world.check(
            "each enemy touch costs 25 health",
            game.health == 100.0 - 25.0 * game.hits.min(4) as f32,
        );
        world.check(
            "the door stays shut until every gem is taken",
            all_gems || game.door_angle == 0.0,
        );
        world.check(
            "the player stays in the arena while the door is shut",
            game.door_angle < -30.0 || (feet.x.abs() < ARENA && feet.z.abs() < ARENA),
        );
        if demo && frame_no == 880 {
            world.check("the demo picks up every gem", all_gems);
            world.check("the enemy hit the player exactly once", game.hits == 1);
            world.check("the health bar shows 75", game.health == 75.0);
            world.check("the player jumped", game.jumped && game.highest > 0.5);
            world.check(
                "the door swung open",
                game.door_angle <= DOOR_OPEN_DEG + 1.0,
            );
            world.check(
                "the player walked out through the door",
                feet.z > ARENA + 1.0,
            );
            world.check("'You win' is showing", game.won_at.is_some() && alive);
        }
        world.report("gems", game.gems);
        world.report("health", game.health);
        world.report("enemy hits", game.hits);
        world.report("door angle (degrees)", game.door_angle);
        world.report("highest feet (m)", game.highest);
        world.report("demo step", game.step);
        world.report("player z", feet.z);
        if let Some(t) = game.won_at {
            world.report("won at (s)", t);
        }
        Ok(())
    })
}
