//! View frustum test, for skipping draws the camera cannot see.
//!
//! With a horizon-scale view distance a scene is hundreds of chunk meshes, and
//! most of them are behind the player. Submitting them all is CPU work that
//! buys nothing.

use glam::{Mat4, Vec3, Vec4};

/// World-space sphere enclosing everything a mesh draws, instances included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub centre: Vec3,
    pub radius: f32,
}

impl Bounds {
    /// A sphere around `points`, or `None` when there are none.
    pub fn around(points: &[Vec3]) -> Option<Self> {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in points {
            min = min.min(*p);
            max = max.max(*p);
        }
        if points.is_empty() {
            return None;
        }
        let centre = 0.5 * (min + max);
        Some(Self {
            centre,
            radius: (max - centre).length(),
        })
    }

    /// This sphere moved and scaled by `transform`.
    pub fn transformed(self, transform: Mat4) -> Self {
        // Any of the basis vectors may be the longest, and a sphere has to
        // survive the worst of them or a rotated mesh clips its own corner off.
        let scale = transform
            .x_axis
            .truncate()
            .length()
            .max(transform.y_axis.truncate().length())
            .max(transform.z_axis.truncate().length());
        Self {
            centre: transform.transform_point3(self.centre),
            radius: self.radius * scale,
        }
    }

    /// The smallest sphere containing both.
    pub fn union(self, other: Self) -> Self {
        let between = other.centre - self.centre;
        let distance = between.length();
        if distance + other.radius <= self.radius {
            return self;
        }
        if distance + self.radius <= other.radius {
            return other;
        }
        let radius = 0.5 * (distance + self.radius + other.radius);
        let towards = if distance > 1e-6 {
            between / distance
        } else {
            Vec3::ZERO
        };
        Self {
            centre: self.centre + towards * (radius - self.radius),
            radius,
        }
    }
}

/// The six clip planes of a view-projection, pointing inward.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the planes from `vp`.
    ///
    /// Works for reversed depth as well as conventional: the two depth planes
    /// come out of `0 <= z <= w`, which holds either way round.
    pub fn from_view_projection(vp: Mat4) -> Self {
        let row = |i: usize| Vec4::new(vp.x_axis[i], vp.y_axis[i], vp.z_axis[i], vp.w_axis[i]);
        let (x, y, z, w) = (row(0), row(1), row(2), row(3));
        let planes = [w + x, w - x, w + y, w - y, z, w - z].map(normalize_plane);
        Self { planes }
    }

    /// Nothing outside is drawn; something merely close to the edge still is.
    pub fn intersects_sphere(&self, centre: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|p| p.truncate().dot(centre) + p.w >= -radius)
    }

    pub fn intersects(&self, bounds: Bounds) -> bool {
        self.intersects_sphere(bounds.centre, bounds.radius)
    }
}

impl Default for Frustum {
    /// Sees everything, for a renderer that has not drawn a frame yet.
    fn default() -> Self {
        Self {
            planes: [Vec4::new(0.0, 0.0, 0.0, f32::MAX); 6],
        }
    }
}

fn normalize_plane(p: Vec4) -> Vec4 {
    let length = p.truncate().length();
    if length > 1e-9 {
        p / length
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;

    fn looking_down_negative_z() -> Frustum {
        let camera = Camera {
            eye: Vec3::ZERO,
            target: Vec3::new(0.0, 0.0, -1.0),
            far: 1_000.0,
            ..Camera::default()
        };
        Frustum::from_view_projection(camera.view_projection(16.0 / 9.0))
    }

    #[test]
    fn what_is_ahead_is_kept_and_what_is_behind_is_not() {
        let frustum = looking_down_negative_z();
        assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, -100.0), 1.0));
        assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, 100.0), 1.0));
        assert!(!frustum.intersects_sphere(Vec3::new(400.0, 0.0, -100.0), 1.0));
        // Beyond the far plane, and far off to the side but big enough to reach in.
        assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, -5_000.0), 10.0));
        assert!(frustum.intersects_sphere(Vec3::new(400.0, 0.0, -100.0), 350.0));
    }

    #[test]
    fn a_chunk_straddling_the_edge_is_never_dropped() {
        // The failure this guards against looks like terrain blinking out at the
        // side of the screen as the player turns.
        let frustum = looking_down_negative_z();
        let corner = Bounds::around(&[Vec3::new(-100.0, 0.0, -100.0), Vec3::new(0.0, 20.0, 0.0)])
            .expect("bounds");
        assert!(frustum.intersects(corner));
    }

    #[test]
    fn instances_widen_the_bounds_they_are_drawn_with() {
        let local = Bounds::around(&[Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)]).expect("bounds");
        let near = local.transformed(Mat4::from_translation(Vec3::ZERO));
        let far = local.transformed(Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)));
        let both = near.union(far);
        assert!(both.radius >= 50.0);
        assert!(both.centre.x > 40.0 && both.centre.x < 60.0);
        // Containment, the property culling actually depends on.
        for probe in [near, far] {
            assert!((probe.centre - both.centre).length() + probe.radius <= both.radius + 1e-3);
        }
    }
}
