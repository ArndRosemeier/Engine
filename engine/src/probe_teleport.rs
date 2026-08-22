#[test]
fn portals_example_teleport_position() {
    use glam::{Mat4, Vec3};
    fn door(position: Vec3, yaw_degrees: f32) -> Mat4 {
        Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(yaw_degrees.to_radians()),
            position,
        )
    }
    let src = door(Vec3::new(0.0, 1.1, -3.0), 180.0);
    let dst = door(Vec3::new(0.0, 1.1, -4.0), 0.0);
    let t = dst * Mat4::from_rotation_y(std::f32::consts::PI) * src.inverse();
    for z in [-3.05f32, -2.95, -2.85] {
        let p = Vec3::new(0.0, 1.6, z);
        let m = t.transform_point3(p);
        println!("src z={z} -> mapped {:?}", m);
    }
}
