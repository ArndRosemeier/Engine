//! Disconnected spaces joined by a pair of openings.
//!
//! A portal is two entities that are the same hole. Looking through one shows
//! the other space from the matching pose; walking through teleports you there.
//! Both openings face their own space (`+Z` in local mesh space). Crossing
//! continues forward because the link includes a 180° yaw.

use crate::camera::Camera;
use crate::mesh::BuiltMesh;
use crate::world::EntityId;
use glam::{Mat4, Vec3, Vec4};

/// Below this distance to the source opening, a transformed portal view blends
/// from a stable threshold camera into the fully teleported eye.
pub const PORTAL_CLOSE_VIEW_DIST: f32 = 0.35;
/// Push the walker past the close-render band after a cross-space teleport so
/// the first frame outside the doorway is stable.
pub const PORTAL_TELEPORT_SETBACK: f32 = 0.42;
/// Eye still grazes the plane within this band; keep drawing the portal.
pub const PORTAL_VISIBILITY_PLANE_EPS: f32 = 0.04;
const PORTAL_THRESHOLD_SETBACK: f32 = 0.12;

/// A named pocket of the world. Entities in different spaces never draw
/// together, so a house interior can sit on top of the yard in coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpaceId(u32);

impl SpaceId {
    /// The space [`crate::world::World::spawn`] uses until you say otherwise.
    /// Sky and clipmap terrain draw here, and in any named space that opts in.
    pub const DEFAULT: Self = Self(0);

    pub(crate) fn from_raw(id: u32) -> Self {
        Self(id)
    }

    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// The opening the camera is looking through, and where that view lands.
#[derive(Clone, Copy, Debug)]
pub struct VisiblePortal {
    pub src: EntityId,
    pub dst: EntityId,
    pub src_transform: Mat4,
    pub dst_transform: Mat4,
    pub dst_center: Vec3,
    pub dst_normal: Vec3,
    pub dest_space: SpaceId,
}

/// Behaviour configured when creating a portal pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortalSettings {
    /// When true, [`crate::world::World::travel`] crosses this opening and moves
    /// the walker into the linked space. When false, the portal is view-only.
    pub teleport: bool,
}

impl Default for PortalSettings {
    fn default() -> Self {
        Self::TELEPORTING
    }
}

impl PortalSettings {
    /// Look through the opening and walk through to the other space.
    pub const TELEPORTING: Self = Self { teleport: true };
    /// Recursive view only; the walker does not cross.
    pub const VIEW_ONLY: Self = Self { teleport: false };
}

/// Identifier for a linked portal pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortalId(u32);

impl PortalId {
    pub(crate) fn from_raw(id: u32) -> Self {
        Self(id)
    }

    #[allow(dead_code)]
    pub(crate) fn raw(self) -> u32 {
        self.0
    }
}

/// A portal pair: two openings, one transform, optional teleport.
#[derive(Clone, Copy, Debug)]
pub struct Portal {
    pub(crate) id: PortalId,
    pub sides: [EntityId; 2],
    pub(crate) teleport: bool,
    pub enabled: bool,
}

impl Portal {
    pub fn id(self) -> PortalId {
        self.id
    }

    pub fn a(self) -> EntityId {
        self.sides[0]
    }

    pub fn b(self) -> EntityId {
        self.sides[1]
    }

    /// Whether [`crate::world::World::travel`] crosses this portal.
    pub fn teleports(self) -> bool {
        self.teleport
    }

    pub fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Each opening that lives in `live`, paired with where it leads.
    pub fn directions(
        self,
        a_space: SpaceId,
        b_space: SpaceId,
        live: SpaceId,
    ) -> impl Iterator<Item = (EntityId, EntityId)> {
        let a_to_b = (self.enabled && a_space == live).then_some((self.sides[0], self.sides[1]));
        let b_to_a = (self.enabled && b_space == live).then_some((self.sides[1], self.sides[0]));
        [a_to_b, b_to_a].into_iter().flatten()
    }
}

/// World-space rectangle of an opening: centre, outward normal, and half-size.
#[derive(Clone, Copy, Debug)]
pub struct PortalPlane {
    pub center: Vec3,
    pub normal: Vec3,
    pub half_width: f32,
    pub half_height: f32,
    pub transform: Mat4,
}

impl PortalPlane {
    pub fn from_transform(transform: Mat4, half_width: f32, half_height: f32) -> Self {
        let normal = transform.transform_vector3(Vec3::Z);
        if normal.length_squared() <= 0.0 {
            panic!("portal transform has a zero +Z axis");
        }
        Self {
            center: transform.transform_point3(Vec3::ZERO),
            normal: normal.normalize(),
            half_width,
            half_height,
            transform,
        }
    }

    /// Signed distance to the opening plane. Positive is the front (the side
    /// you look at to see the other space).
    pub fn signed_distance(self, point: Vec3) -> f32 {
        (point - self.center).dot(self.normal)
    }
}

/// Half-extents of an opening mesh in local X and Y.
///
/// [`crate::mesh::Mesh::opening`] authors a quad on the XY plane; any mesh
/// whose points span X/Y works the same way.
pub fn opening_extents(mesh: &BuiltMesh) -> (f32, f32) {
    let mut half_w = 0.0_f32;
    let mut half_h = 0.0_f32;
    for p in &mesh.positions {
        half_w = half_w.max(p.x.abs());
        half_h = half_h.max(p.y.abs());
    }
    (half_w, half_h)
}

/// Maps a point in front of `src` onto the matching point in front of `dst`.
///
/// Both openings face their own space. The 180° yaw is what makes walking
/// forward through one come out walking forward through the other.
pub fn portal_matrix(src: Mat4, dst: Mat4) -> Mat4 {
    dst * Mat4::from_rotation_y(std::f32::consts::PI) * src.inverse()
}

/// Camera on the far side of a portal, same look as standing at the linked pose.
pub fn teleport_camera(camera: &Camera, src: Mat4, dst: Mat4) -> Camera {
    let t = portal_matrix(src, dst);
    let up = t.transform_vector3(camera.up);
    if up.length_squared() <= 0.0 {
        panic!("portal transform collapsed the camera up axis");
    }
    Camera {
        eye: t.transform_point3(camera.eye),
        target: t.transform_point3(camera.target),
        up: up.normalize(),
        fov_y_degrees: camera.fov_y_degrees,
        near: camera.near,
        far: camera.far,
    }
}

/// Destination view from a stable eye position just behind the opening.
///
/// This avoids exposing unrelated geometry when the source and destination
/// rooms have different depths. Look direction, pitch, FOV, and clipping still
/// come from the character camera.
pub fn threshold_camera(
    camera: &Camera,
    src: Mat4,
    dst: Mat4,
    dst_center: Vec3,
    dst_normal: Vec3,
    eye_offset_y: f32,
    setback: f32,
) -> Camera {
    if !eye_offset_y.is_finite() || !setback.is_finite() || setback <= 0.0 {
        panic!(
            "portal threshold view needs finite eye offset and positive setback, got ({eye_offset_y}, {setback})"
        );
    }
    let transformed = teleport_camera(camera, src, dst);
    let look = transformed.target - transformed.eye;
    if look.length_squared() <= 0.0 {
        panic!("portal threshold view has zero look direction");
    }
    let eye = dst_center - dst_normal * setback + Vec3::Y * eye_offset_y;
    Camera {
        eye,
        target: eye + look.normalize(),
        up: transformed.up,
        fov_y_degrees: transformed.fov_y_degrees,
        near: transformed.near,
        far: transformed.far,
    }
}

/// Minimum distance from the source opening used for portal *rendering* when the
/// real eye is pressed against the frame. Stops stencil/depth failing in the
/// last few centimetres before the plane.
pub fn portal_render_camera(camera: &Camera, src_plane: PortalPlane) -> Camera {
    let dist = src_plane.signed_distance(camera.eye);
    let min_dist = camera.near + 0.02;
    let target_dist = PORTAL_THRESHOLD_SETBACK.max(min_dist);
    if dist >= target_dist {
        return camera.clone();
    }
    let push = target_dist - dist.max(0.0);
    let offset = src_plane.normal * push;
    Camera {
        eye: camera.eye + offset,
        target: camera.target + offset,
        ..camera.clone()
    }
}

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn blend_cameras(a: Camera, b: Camera, t: f32) -> Camera {
    let up = a.up.lerp(b.up, t);
    if up.length_squared() <= 0.0 {
        panic!("portal camera blend collapsed the up axis");
    }
    Camera {
        eye: a.eye.lerp(b.eye, t),
        target: a.target.lerp(b.target, t),
        up: up.normalize(),
        fov_y_degrees: a.fov_y_degrees + (b.fov_y_degrees - a.fov_y_degrees) * t,
        near: a.near + (b.near - a.near) * t,
        far: a.far + (b.far - a.far) * t,
    }
}

fn transformed_portal_view(
    camera: &Camera,
    visible: &VisiblePortal,
    src_plane: PortalPlane,
) -> Camera {
    let camera = portal_render_camera(camera, src_plane);
    let dist = src_plane.signed_distance(camera.eye);
    let tele = teleport_camera(&camera, visible.src_transform, visible.dst_transform);
    if dist >= PORTAL_CLOSE_VIEW_DIST {
        return tele;
    }
    let eye_offset_y = camera.eye.y - visible.dst_center.y;
    let thresh = threshold_camera(
        &camera,
        visible.src_transform,
        visible.dst_transform,
        visible.dst_center,
        visible.dst_normal,
        eye_offset_y,
        PORTAL_THRESHOLD_SETBACK,
    );
    if dist <= PORTAL_THRESHOLD_SETBACK {
        return thresh;
    }
    let span = PORTAL_CLOSE_VIEW_DIST - PORTAL_THRESHOLD_SETBACK;
    let t = smoothstep01((dist - PORTAL_THRESHOLD_SETBACK) / span);
    blend_cameras(thresh, tele, t)
}

/// Virtual camera for drawing through `visible`, stable when pressed against the frame.
pub fn portal_view_camera(camera: &Camera, visible: &VisiblePortal, src_plane: PortalPlane) -> Camera {
    transformed_portal_view(camera, visible, src_plane)
}

/// True when oblique clipping should be skipped for this portal view.
pub fn portal_view_is_close(camera: &Camera, _visible: &VisiblePortal, src_plane: PortalPlane) -> bool {
    src_plane.signed_distance(camera.eye) < PORTAL_CLOSE_VIEW_DIST
}

/// Oblique near-plane clip for recursive portal draws.
///
/// When the viewer is pressed against the frame we already use a threshold
/// camera; adding oblique clipping there clips away the floor and walls that
/// should fill the opening (blue clear-color gaps at doorways).
pub fn portal_destination_clip(
    camera: &Camera,
    visible: &VisiblePortal,
    src_plane: PortalPlane,
) -> Option<(Vec3, Vec3)> {
    if portal_view_is_close(camera, visible, src_plane) {
        return None;
    }
    Some((
        visible.dst_center + visible.dst_normal * 0.02,
        visible.dst_normal,
    ))
}

/// Yaw (degrees, 0 = +Z) after walking through `src` into `dst`.
///
/// Doorways keep walking forward. Floor hatches cross vertically, so the
/// compass heading is left alone — applying the door 180° here spins the walker.
pub fn teleport_yaw(yaw_degrees: f32, src: Mat4, dst: Mat4) -> f32 {
    let src_normal = src.transform_vector3(Vec3::Z);
    let dst_normal = dst.transform_vector3(Vec3::Z);
    if src_normal.length_squared() > 0.0
        && dst_normal.length_squared() > 0.0
        && src_normal.normalize().y.abs() > 0.7
        && dst_normal.normalize().y.abs() > 0.7
    {
        return yaw_degrees;
    }
    let t = portal_matrix(src, dst);
    let facing = t.transform_vector3(Camera::facing_xz(yaw_degrees));
    facing.x.atan2(facing.z).to_degrees()
}

/// True when `prev` → `curr` goes through the front of the opening and hits
/// the rectangle, not just the infinite plane.
pub fn segment_crosses_opening(prev: Vec3, curr: Vec3, plane: PortalPlane) -> bool {
    let d0 = plane.signed_distance(prev);
    let d1 = plane.signed_distance(curr);
    if d0 <= 0.0 || d1 >= 0.0 {
        return false;
    }
    let denom = d0 - d1;
    if denom == 0.0 {
        return false;
    }
    let hit = prev + (curr - prev) * (d0 / denom);
    let local = plane.transform.inverse().transform_point3(hit);
    local.x.abs() <= plane.half_width && local.y.abs() <= plane.half_height
}

/// View-projection that uses `point`+`normal` as the near plane so geometry
/// behind the destination opening is clipped (Lengyel, reversed-Z).
pub fn oblique_view_projection(camera: &Camera, aspect: f32, point: Vec3, normal: Vec3) -> Mat4 {
    if normal.length_squared() <= 0.0 {
        panic!("oblique clip plane has a zero normal");
    }
    let n = normal.normalize();
    let view = camera.view_matrix();
    let mut proj = camera.projection_matrix(aspect);
    let world_plane = Vec4::new(n.x, n.y, n.z, -n.dot(point));
    let clip_view = view.inverse().transpose() * world_plane;
    let q = proj.inverse()
        * Vec4::new(
            sign_nonzero(clip_view.x),
            sign_nonzero(clip_view.y),
            1.0,
            1.0,
        );
    let denom = clip_view.dot(q);
    if denom.abs() < 1e-8 {
        return proj * view;
    }
    let c = clip_view * (1.0 / denom);
    // Reversed-Z near is NDC z = 1, i.e. clip.z = clip.w. The third row
    // plus the fourth row must equal the view-space clip plane.
    proj.x_axis.z = c.x + proj.x_axis.w;
    proj.y_axis.z = c.y + proj.y_axis.w;
    proj.z_axis.z = c.z + proj.z_axis.w;
    proj.w_axis.z = c.w + proj.w_axis.w;
    proj * view
}

fn sign_nonzero(v: f32) -> f32 {
    if v >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// wgpu clip volume after perspective divide: x,y in [-1,1], z in [0,1].
pub fn clip_contains(vp: Mat4, point: Vec3) -> bool {
    let h = vp * Vec4::from((point, 1.0));
    if h.w.abs() < 1e-8 {
        return false;
    }
    let x = h.x / h.w;
    let y = h.y / h.w;
    let z = h.z / h.w;
    (-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y) && (0.0..=1.0).contains(&z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn door(position: Vec3, yaw_degrees: f32) -> Mat4 {
        Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(yaw_degrees.to_radians()),
            position,
        )
    }

    #[test]
    fn front_of_a_maps_behind_b_looking_into_the_room() {
        // Yard door faces −Z (toward a player standing south). House door
        // faces +Z (into the room). A viewer still in the yard maps behind
        // the house door — that is the virtual camera. Crossing (tested on
        // World::travel) comes out in front, inside the room.
        let src = door(Vec3::new(0.0, 1.0, -1.2), 180.0);
        let dst = door(Vec3::new(0.0, 1.0, -3.0), 0.0);
        let t = portal_matrix(src, dst);
        let in_yard = Vec3::new(0.0, 1.6, -6.0);
        let virt = t.transform_point3(in_yard);
        assert!(
            virt.z < -3.0,
            "virtual camera sits behind the house door, got {virt}"
        );
        let yaw = teleport_yaw(0.0, src, dst);
        let facing = Camera::facing_xz(yaw);
        assert!(
            facing.z > 0.8,
            "forward should still be +Z, yaw={yaw} facing={facing}"
        );
    }

    fn hatch(position: Vec3, pitch_degrees: f32) -> Mat4 {
        crate::place::Place::new(position.x, position.y, position.z)
            .with_pitch_deg(pitch_degrees)
            .to_matrix()
    }

    #[test]
    fn falling_through_a_hatch_maps_below_the_mouth() {
        let src = hatch(Vec3::new(0.0, 10.0, 0.0), -90.0);
        let dst = hatch(Vec3::new(0.0, 4.0, 0.0), 90.0);
        let t = portal_matrix(src, dst);
        let crossed = Vec3::new(0.0, 9.0, 0.0);
        let mapped = t.transform_point3(crossed);
        assert!(
            mapped.y < 4.0,
            "a point that has fallen through the world hatch maps below the mouth, got {mapped}"
        );
        let yaw = teleport_yaw(0.0, src, dst);
        assert!(
            yaw.abs() < 5.0,
            "hatch travel must keep +Z facing, yaw={yaw}"
        );
    }

    #[test]
    fn hatch_crossing_hits_the_floor_rectangle() {
        let plane = PortalPlane::from_transform(hatch(Vec3::new(0.0, 10.0, 0.0), -90.0), 1.0, 1.0);
        assert!(segment_crosses_opening(
            Vec3::new(0.0, 11.0, 0.0),
            Vec3::new(0.0, 9.0, 0.0),
            plane
        ));
        assert!(!segment_crosses_opening(
            Vec3::new(3.0, 11.0, 0.0),
            Vec3::new(3.0, 9.0, 0.0),
            plane
        ));
        assert!(!segment_crosses_opening(
            Vec3::new(0.0, 9.0, 0.0),
            Vec3::new(0.0, 11.0, 0.0),
            plane
        ));
    }

    #[test]
    fn crossing_hits_the_rectangle_not_the_infinite_plane() {
        let plane = PortalPlane::from_transform(door(Vec3::new(0.0, 1.0, 0.0), 0.0), 0.6, 1.0);
        assert!(segment_crosses_opening(
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, -1.0),
            plane
        ));
        assert!(!segment_crosses_opening(
            Vec3::new(3.0, 1.0, 1.0),
            Vec3::new(3.0, 1.0, -1.0),
            plane
        ));
        assert!(!segment_crosses_opening(
            Vec3::new(0.0, 1.0, -1.0),
            Vec3::new(0.0, 1.0, 1.0),
            plane
        ));
    }

    #[test]
    fn teleport_camera_preserves_look_through_the_hole() {
        let src = door(Vec3::new(0.0, 1.0, 0.0), 0.0);
        let dst = door(Vec3::new(20.0, 1.0, 0.0), 180.0);
        let cam = Camera::look_at(Vec3::new(0.0, 1.6, 3.0), Vec3::new(0.0, 1.6, 0.0));
        let virt = teleport_camera(&cam, src, dst);
        let look = (virt.target - virt.eye).normalize();
        assert!(
            look.z < -0.8,
            "virtual camera should look through dest along −Z, look={look}"
        );
    }

    #[test]
    fn close_transformed_portal_uses_threshold_view() {
        let src = door(Vec3::new(0.0, 1.0, -1.2), 180.0);
        let dst = door(Vec3::new(0.0, 1.0, -3.0), 0.0);
        let src_plane = PortalPlane::from_transform(src, 0.6, 1.0);
        let dst_plane = PortalPlane::from_transform(dst, 0.6, 1.0);
        let cam = Camera::look_at(Vec3::new(0.0, 1.6, -1.28), Vec3::new(0.0, 1.6, -3.0));
        assert!(
            src_plane.signed_distance(cam.eye) < PORTAL_THRESHOLD_SETBACK,
            "test eye should be in the pure threshold band"
        );
        let virt = threshold_camera(
            &cam,
            src,
            dst,
            dst_plane.center,
            dst_plane.normal,
            cam.eye.y - dst_plane.center.y,
            PORTAL_THRESHOLD_SETBACK,
        );
        let teleported = teleport_camera(&cam, src, dst);
        assert!(
            virt.eye.distance(teleported.eye) > 0.01,
            "close view should not use the fully teleported eye"
        );
    }

    #[test]
    fn transformed_portal_view_blends_at_the_doorway() {
        let src = door(Vec3::new(0.0, 1.0, -1.2), 180.0);
        let dst = door(Vec3::new(0.0, 1.0, -3.0), 0.0);
        let src_plane = PortalPlane::from_transform(src, 0.6, 1.0);
        let dst_plane = PortalPlane::from_transform(dst, 0.6, 1.0);
        let visible = VisiblePortal {
            src: EntityId::test(1),
            dst: EntityId::test(2),
            src_transform: src,
            dst_transform: dst,
            dst_center: dst_plane.center,
            dst_normal: dst_plane.normal,
            dest_space: SpaceId::DEFAULT,
        };
        let cam_close = Camera::look_at(Vec3::new(0.0, 1.6, -1.28), Vec3::new(0.0, 1.6, -3.0));
        let cam_mid = Camera::look_at(Vec3::new(0.0, 1.6, -1.40), Vec3::new(0.0, 1.6, -3.0));
        let cam_far = Camera::look_at(Vec3::new(0.0, 1.6, -1.56), Vec3::new(0.0, 1.6, -3.0));
        let close = portal_view_camera(&cam_close, &visible, src_plane);
        let mid = portal_view_camera(&cam_mid, &visible, src_plane);
        let far = portal_view_camera(&cam_far, &visible, src_plane);
        let tele = teleport_camera(&cam_far, src, dst);
        assert!(
            mid.eye.distance(close.eye) > 0.01 && mid.eye.distance(far.eye) > 0.01,
            "blend band should sit between threshold and teleported eyes"
        );
        assert!(
            far.eye.distance(tele.eye) < 1e-4,
            "far doorway view should match the teleported camera"
        );
    }

    #[test]
    fn teleport_setback_clears_close_render_band() {
        assert!(
            PORTAL_TELEPORT_SETBACK > PORTAL_CLOSE_VIEW_DIST,
            "post-teleport pose must leave the doorway close band"
        );
    }

    #[test]
    fn inside_house_threshold_view_sees_yard_floor() {
        let door_out = door(Vec3::new(0.0, 1.1, -3.0), 180.0);
        let door_in = door(Vec3::new(0.0, 1.1, -4.0), 0.0);
        let src_plane = PortalPlane::from_transform(door_in, 0.6, 1.1);
        let dst_plane = PortalPlane::from_transform(door_out, 0.6, 1.1);
        let cam = Camera::first_person(Vec3::new(0.0, 1.6, -3.85), 180.0, 0.0);
        assert!(
            src_plane.signed_distance(cam.eye) < PORTAL_CLOSE_VIEW_DIST,
            "doorway pose should use the close threshold path"
        );
        let virt = threshold_camera(
            &cam,
            door_in,
            door_out,
            dst_plane.center,
            dst_plane.normal,
            cam.eye.y - dst_plane.center.y,
            PORTAL_THRESHOLD_SETBACK,
        );
        let vp = virt.view_projection(16.0 / 9.0);
        let yard_floor = Vec3::new(0.0, 0.0, -6.0);
        assert!(
            clip_contains(vp, yard_floor),
            "threshold view from inside should see the yard, eye={:?} look={:?}",
            virt.eye,
            virt.target - virt.eye
        );
    }

    #[test]
    fn threshold_camera_ignores_source_room_depth() {
        let src = door(Vec3::new(0.0, 1.0, 0.0), 0.0);
        let dst = door(Vec3::new(20.0, 1.0, 0.0), 180.0);
        let cam = Camera::look_at(Vec3::new(0.0, 1.6, 30.0), Vec3::new(0.0, 1.6, 0.0));
        let dst_plane = PortalPlane::from_transform(dst, 0.6, 1.0);
        let virt = threshold_camera(&cam, src, dst, dst_plane.center, dst_plane.normal, 0.6, 0.5);
        let expected = dst_plane.center - dst_plane.normal * 0.5 + Vec3::Y * 0.6;
        assert!(
            virt.eye.distance(expected) < 1e-4,
            "threshold eye drifted with source depth: {:?} != {:?}",
            virt.eye,
            expected
        );
        assert!((virt.eye.y - 1.6).abs() < 1e-4);
        let clip_point = dst_plane.center + dst_plane.normal * 0.02;
        let vp = oblique_view_projection(&virt, 16.0 / 9.0, clip_point, dst_plane.normal);
        let beyond_door = dst_plane.center + dst_plane.normal * 5.0 + Vec3::Y * 0.6;
        assert!(
            clip_contains(vp, beyond_door),
            "threshold view clipped everything beyond the destination door"
        );

        let reverse_cam = Camera::look_at(
            dst_plane.center + dst_plane.normal * 30.0 + Vec3::Y * 0.6,
            dst_plane.center + Vec3::Y * 0.6,
        );
        let src_plane = PortalPlane::from_transform(src, 0.6, 1.0);
        let reverse = threshold_camera(
            &reverse_cam,
            dst,
            src,
            src_plane.center,
            src_plane.normal,
            0.6,
            0.5,
        );
        let reverse_clip = src_plane.center + src_plane.normal * 0.02;
        let reverse_vp =
            oblique_view_projection(&reverse, 16.0 / 9.0, reverse_clip, src_plane.normal);
        let reverse_beyond = src_plane.center + src_plane.normal * 5.0 + Vec3::Y * 0.6;
        assert!(
            clip_contains(reverse_vp, reverse_beyond),
            "reverse threshold view clipped everything beyond the destination door"
        );
    }

    #[test]
    fn oblique_near_clips_behind_the_destination_door() {
        let camera = Camera::look_at(Vec3::new(0.0, 1.6, -3.0), Vec3::new(0.0, 1.6, 0.0));
        let plane_point = Vec3::new(0.0, 1.0, 0.0);
        let plane_normal = Vec3::Z;
        let vp = oblique_view_projection(&camera, 1.0, plane_point, plane_normal);
        let in_room = Vec3::new(0.0, 1.0, 4.0);
        let behind_door = Vec3::new(0.0, 1.0, -1.0);
        assert!(
            clip_contains(vp, in_room),
            "room in front of the door must stay visible"
        );
        assert!(
            !clip_contains(vp, behind_door),
            "geometry behind the destination door must clip"
        );
    }

    #[test]
    fn portal_pair_virtual_view_sees_the_room() {
        let src = door(Vec3::new(0.0, 1.18, -10.0), 0.0);
        let dst = door(Vec3::new(0.0, 1.18, 10.0), 180.0);
        let cam = Camera::first_person(Vec3::new(0.0, 1.6, 0.0), 180.0, 0.0);
        let virt = teleport_camera(&cam, src, dst);
        let dst_center = Vec3::new(0.0, 1.18, 10.0);
        let dst_normal = -Vec3::Z;
        let clip_point = dst_center + dst_normal * 0.02;
        let vp = oblique_view_projection(&virt, 16.0 / 9.0, clip_point, dst_normal);
        let room = Vec3::new(0.0, 1.0, 0.0);
        let behind = Vec3::new(0.0, 1.18, 12.0);
        assert!(
            virt.eye.z > 10.0,
            "virtual camera sits behind door B, got {:?}",
            virt.eye
        );
        assert!(
            clip_contains(vp, room),
            "room centre must stay visible through the pair"
        );
        assert!(
            !clip_contains(vp, behind),
            "geometry behind the destination door must clip"
        );
    }
}
