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
    pub view: PortalView,
}

/// How the destination camera is positioned while looking through a portal.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum PortalView {
    /// Preserve the viewer's full offset from the source opening.
    #[default]
    Transformed,
    /// Fix the camera at the destination threshold while preserving look direction.
    Threshold { eye_offset_y: f32, setback: f32 },
}

/// Two openings that are the same doorway, dungeon mouth, or teleport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortalLink {
    pub a: EntityId,
    pub b: EntityId,
    pub(crate) view: PortalView,
}

impl PortalLink {
    /// Each opening that lives in `live`, paired with where it leads.
    ///
    /// A same-space pair (Portal A ↔ B in one room) yields both directions.
    pub fn directions(
        self,
        a_space: SpaceId,
        b_space: SpaceId,
        live: SpaceId,
    ) -> impl Iterator<Item = (EntityId, EntityId)> {
        let a_to_b = (a_space == live).then_some((self.a, self.b));
        let b_to_a = (b_space == live).then_some((self.b, self.a));
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

/// Yaw (degrees, 0 = +Z) after walking through `src` into `dst`.
pub fn teleport_yaw(yaw_degrees: f32, src: Mat4, dst: Mat4) -> f32 {
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
