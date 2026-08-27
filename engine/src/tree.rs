//! Deterministic, connected procedural tree geometry.
use crate::{Color, Mesh, PointId};
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TreeKind {
    Oak,
    Beech,
    Birch,
    Pine,
    Spruce,
    Willow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TreeLod {
    Hero,
    Mid,
    Far,
}

impl TreeLod {
    pub const fn level(self) -> u8 {
        match self {
            Self::Hero => 0,
            Self::Mid => 1,
            Self::Far => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    Broad,
    Pointed,
    Needle,
    Willow,
}

impl TreeKind {
    pub const fn height_range(self) -> (f32, f32) {
        let profile = self.profile();
        (profile.min_height, profile.max_height)
    }

    pub const fn leaf_kind(self) -> LeafKind {
        match self {
            Self::Oak | Self::Beech => LeafKind::Broad,
            Self::Birch => LeafKind::Pointed,
            Self::Pine | Self::Spruce => LeafKind::Needle,
            Self::Willow => LeafKind::Willow,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeProfile {
    pub height: f32,
    pub min_height: f32,
    pub max_height: f32,
    pub trunk_radius: f32,
    pub primary_branches: u32,
    pub branching_depth: u32,
    pub branch_upward_bias: f32,
    pub branch_spread: f32,
    pub lower_branch_start: f32,
    pub foliage_size: f32,
    pub foliage_per_cluster: u32,
    pub droop: f32,
}
impl TreeKind {
    pub const fn profile(self) -> TreeProfile {
        match self {
            Self::Oak => TreeProfile {
                height: 24.0,
                min_height: 5.0,
                max_height: 42.0,
                trunk_radius: 0.56,
                primary_branches: 7,
                branching_depth: 3,
                branch_upward_bias: 0.32,
                branch_spread: 0.92,
                lower_branch_start: 0.22,
                foliage_size: 0.60,
                foliage_per_cluster: 7,
                droop: 0.0,
            },
            Self::Beech => TreeProfile {
                height: 30.0,
                min_height: 6.0,
                max_height: 52.0,
                trunk_radius: 0.50,
                primary_branches: 8,
                branching_depth: 4,
                branch_upward_bias: 0.46,
                branch_spread: 0.78,
                lower_branch_start: 0.28,
                foliage_size: 0.54,
                foliage_per_cluster: 7,
                droop: 0.0,
            },
            Self::Birch => TreeProfile {
                height: 22.0,
                min_height: 5.0,
                max_height: 34.0,
                trunk_radius: 0.34,
                primary_branches: 6,
                branching_depth: 4,
                branch_upward_bias: 0.72,
                branch_spread: 0.64,
                lower_branch_start: 0.38,
                foliage_size: 0.46,
                foliage_per_cluster: 6,
                droop: 0.0,
            },
            Self::Pine => TreeProfile {
                height: 36.0,
                min_height: 7.0,
                max_height: 60.0,
                trunk_radius: 0.44,
                primary_branches: 6,
                branching_depth: 3,
                branch_upward_bias: 0.64,
                branch_spread: 0.68,
                lower_branch_start: 0.18,
                foliage_size: 0.48,
                foliage_per_cluster: 8,
                droop: 0.0,
            },
            Self::Spruce => TreeProfile {
                height: 38.0,
                min_height: 6.0,
                max_height: 60.0,
                trunk_radius: 0.48,
                primary_branches: 7,
                branching_depth: 3,
                branch_upward_bias: 0.82,
                branch_spread: 0.54,
                lower_branch_start: 0.12,
                foliage_size: 0.44,
                foliage_per_cluster: 9,
                droop: 0.0,
            },
            Self::Willow => TreeProfile {
                height: 18.0,
                min_height: 5.0,
                max_height: 30.0,
                trunk_radius: 0.52,
                primary_branches: 8,
                branching_depth: 3,
                branch_upward_bias: 0.18,
                branch_spread: 0.94,
                lower_branch_start: 0.20,
                foliage_size: 0.58,
                foliage_per_cluster: 8,
                droop: 0.42,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TreeSettings {
    pub kind: TreeKind,
    pub seed: u32,
    pub lod: TreeLod,
}
impl TreeSettings {
    pub const fn new(kind: TreeKind, seed: u32) -> Self {
        Self {
            kind,
            seed,
            lod: TreeLod::Hero,
        }
    }
    pub const fn with_lod(mut self, lod: TreeLod) -> Self {
        self.lod = lod;
        self
    }
    pub const fn broadleaf(seed: u32) -> Self {
        Self::new(TreeKind::Oak, seed)
    }
    pub const fn conifer(seed: u32) -> Self {
        Self::new(TreeKind::Pine, seed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BranchNode {
    start: Vec3,
    end: Vec3,
    start_radius: f32,
    end_radius: f32,
    parent: Option<usize>,
    depth: u32,
    wind_phase: f32,
}
impl BranchNode {
    pub fn start(self) -> Vec3 {
        self.start
    }
    pub fn end(self) -> Vec3 {
        self.end
    }
    pub fn start_radius(self) -> f32 {
        self.start_radius
    }
    pub fn end_radius(self) -> f32 {
        self.end_radius
    }
    pub fn parent(self) -> Option<usize> {
        self.parent
    }
    pub fn depth(self) -> u32 {
        self.depth
    }
    pub fn wind_phase(self) -> f32 {
        self.wind_phase
    }
}

#[derive(Debug)]
pub struct TreeAsset {
    bark: Mesh,
    foliage: Mesh,
    branches: Vec<BranchNode>,
    foliage_clusters: usize,
    kind: TreeKind,
    lod: TreeLod,
}
impl TreeAsset {
    pub fn generate(origin: Vec3, settings: TreeSettings) -> Self {
        let profile = settings.kind.profile();
        assert!(
            profile.branching_depth >= 1,
            "tree profile branching_depth must be at least 1"
        );
        // Branch geometry was authored around an 8 m prototype. Keep branch reach
        // and leaf size proportional when species heights use mature-world metres;
        // otherwise taller trees collapse into bare sticks.
        let structure_scale = profile.height / 8.0;
        let (branch_depth, primary_branches, foliage_per_cluster) = match settings.lod {
            TreeLod::Hero => (
                profile.branching_depth,
                profile.primary_branches,
                profile.foliage_per_cluster,
            ),
            TreeLod::Mid => (
                profile
                    .branching_depth
                    .checked_sub(1)
                    .expect("validated branching depth")
                    .max(1),
                (profile.primary_branches * 3 / 4).max(3),
                (profile.foliage_per_cluster * 3 / 4).max(2),
            ),
            TreeLod::Far => (1, (profile.primary_branches / 2).max(3), 2),
        };
        let mut asset = Self {
            bark: Mesh::new(),
            foliage: Mesh::new(),
            branches: Vec::new(),
            foliage_clusters: 0,
            kind: settings.kind,
            lod: settings.lod,
        };
        let trunk_end = origin + Vec3::Y * profile.height;
        let trunk = asset.add_branch(
            origin,
            trunk_end,
            profile.trunk_radius,
            profile.trunk_radius * 0.18,
            None,
            0,
            0.0,
        );
        let mut frontier = Vec::new();
        for primary in 0..primary_branches {
            let t = profile.lower_branch_start
                + primary as f32 / primary_branches as f32 * (0.74 - profile.lower_branch_start);
            let anchor = origin.lerp(trunk_end, t);
            let phase = hash(settings.seed, primary) as f32 * 0.013;
            let angle = phase + primary as f32 * std::f32::consts::TAU / primary_branches as f32;
            let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
            let taper = 1.0 - t * 0.45;
            let direction = (radial * profile.branch_spread
                + Vec3::Y * (profile.branch_upward_bias + t * 0.24 - profile.droop))
                .normalize();
            let end = anchor + direction * (1.65 * taper + 0.45) * structure_scale;
            let branch = asset.add_branch(
                anchor,
                end,
                profile.trunk_radius * 0.28 * taper,
                profile.trunk_radius * 0.11 * taper,
                Some(trunk),
                1,
                phase,
            );
            frontier.push((
                branch,
                end,
                profile.trunk_radius * 0.11 * taper,
                direction,
                1_u32,
            ));
        }
        for depth in 1..branch_depth {
            let mut next = Vec::new();
            for (parent, anchor, radius, parent_direction, _) in frontier {
                let count = 2 + (hash(settings.seed, parent as u32 + depth) % 2) as usize;
                for child in 0..count {
                    let phase = hash(
                        settings.seed.wrapping_add(depth),
                        (parent * 23 + child) as u32,
                    ) as f32
                        * 0.013;
                    let angle = phase + child as f32 * std::f32::consts::TAU / count as f32;
                    let radial = Vec3::new(angle.cos(), 0.0, angle.sin());
                    let direction = (parent_direction * 0.35
                        + radial * profile.branch_spread * 0.72
                        + Vec3::Y * (profile.branch_upward_bias + 0.12 - profile.droop))
                        .normalize();
                    let end = anchor
                        + direction * (1.18 - depth as f32 * 0.18).max(0.52) * structure_scale;
                    let end_radius = (radius * 0.58).max(0.016);
                    let index = asset.add_branch(
                        anchor,
                        end,
                        radius * 0.76,
                        end_radius,
                        Some(parent),
                        depth + 1,
                        phase,
                    );
                    next.push((index, end, end_radius, direction, depth + 1));
                }
            }
            frontier = next;
        }
        let terminals: Vec<(usize, BranchNode)> = asset
            .branches
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, branch)| {
                branch.depth >= branch_depth
                    || !asset
                        .branches
                        .iter()
                        .any(|child| child.parent == Some(*index))
            })
            .collect();
        for (index, branch) in terminals {
            asset.add_cluster(
                &branch,
                index,
                profile,
                foliage_per_cluster,
                settings.seed,
                settings.kind,
            );
        }
        // Every species gets a guaranteed crown anchor at the trunk top. Terminal
        // branches still provide the irregular lower and side foliage, but this
        // prevents a valid branch layout from reading as a dead, crownless tree.
        let crown = BranchNode {
            start: trunk_end - Vec3::Y,
            end: trunk_end,
            start_radius: profile.trunk_radius * 0.18,
            end_radius: profile.trunk_radius * 0.08,
            parent: Some(trunk),
            depth: branch_depth,
            wind_phase: 0.0,
        };
        asset.add_cluster(
            &crown,
            asset.branches.len(),
            profile,
            foliage_per_cluster,
            settings.seed ^ 0xC70A,
            settings.kind,
        );
        asset
    }
    pub fn generate_lod(origin: Vec3, settings: TreeSettings, lod: TreeLod) -> Self {
        Self::generate(origin, settings.with_lod(lod))
    }
    pub fn lod(&self) -> TreeLod {
        // The generated structure is intentionally represented by its public settings API;
        // assets retain the requested level for prototype introspection.
        self.lod
    }
    pub fn kind(&self) -> TreeKind {
        self.kind
    }
    pub fn profile(&self) -> TreeProfile {
        self.kind.profile()
    }
    pub fn bark(&self) -> &Mesh {
        &self.bark
    }
    pub fn foliage(&self) -> &Mesh {
        &self.foliage
    }

    /// Return one renderable mesh containing paired bark and foliage.
    pub fn mesh(&self) -> crate::error::EngineResult<Mesh> {
        let mut mesh = self.bark.clone();
        mesh.append_translated(&self.foliage, Vec3::ZERO)?;
        Ok(mesh)
    }
    pub fn branches(&self) -> &[BranchNode] {
        &self.branches
    }
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }
    pub fn foliage_cluster_count(&self) -> usize {
        self.foliage_clusters
    }
    fn add_branch(
        &mut self,
        start: Vec3,
        end: Vec3,
        start_radius: f32,
        end_radius: f32,
        parent: Option<usize>,
        depth: u32,
        wind_phase: f32,
    ) -> usize {
        let index = self.branches.len();
        self.branches.push(BranchNode {
            start,
            end,
            start_radius,
            end_radius,
            parent,
            depth,
            wind_phase,
        });
        add_tube(&mut self.bark, start, end, start_radius, end_radius);
        index
    }
    fn add_cluster(
        &mut self,
        branch: &BranchNode,
        index: usize,
        profile: TreeProfile,
        foliage_per_cluster: u32,
        seed: u32,
        kind: TreeKind,
    ) {
        let structure_scale = profile.height / 8.0;
        let direction = (branch.end - branch.start).normalize();
        let side = direction.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        let up = side.cross(direction).normalize();
        let centre = branch.end + direction * profile.foliage_size * structure_scale * 0.16;
        for leaf in 0..foliage_per_cluster {
            let phase = hash(seed, index as u32 * 31 + leaf) as f32 * 0.021;
            let tangent = (side * phase.cos() + up * phase.sin()).normalize();
            add_leaf(
                &mut self.foliage,
                centre + tangent * ((leaf as f32 - foliage_per_cluster as f32 * 0.5) * 0.11),
                direction,
                tangent,
                profile.foliage_size * structure_scale,
                phase,
                kind.leaf_kind(),
            );
        }
        self.foliage_clusters += 1;
    }
}

fn add_tube(mesh: &mut Mesh, start: Vec3, end: Vec3, r0: f32, r1: f32) {
    let axis = (end - start).normalize();
    let side = axis.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let up = side.cross(axis).normalize();
    let n = 8_usize;
    let mut rings: [Vec<PointId>; 2] = [Vec::with_capacity(n), Vec::with_capacity(n)];
    for (ring, (centre, radius)) in [(start, r0), (end, r1)].into_iter().enumerate() {
        for i in 0..n {
            let a = i as f32 * std::f32::consts::TAU / n as f32;
            let id = mesh
                .add_point(centre + (side * a.cos() + up * a.sin()) * radius)
                .expect("tree tube point");
            mesh.set_point_color(id, Color::rgb(90, 58, 34))
                .expect("tree bark color");
            rings[ring].push(id);
        }
    }
    for i in 0..n {
        let j = (i + 1) % n;
        mesh.add_quad(rings[0][i], rings[0][j], rings[1][j], rings[1][i])
            .expect("tree tube quad");
    }
}
fn add_leaf(
    mesh: &mut Mesh,
    centre: Vec3,
    direction: Vec3,
    side: Vec3,
    size: f32,
    phase: f32,
    kind: LeafKind,
) {
    let axis = (direction + side * phase.sin() * 0.24).normalize() * size;
    let width_scale = match kind {
        LeafKind::Broad => 0.72,
        LeafKind::Pointed => 0.60,
        LeafKind::Needle => 0.34,
        LeafKind::Willow => 0.46,
    };
    let length_scale = match kind {
        LeafKind::Broad => 1.0,
        LeafKind::Pointed => 1.18,
        LeafKind::Needle => 1.34,
        LeafKind::Willow => 1.42,
    };
    let axis = axis * length_scale;
    let width = side * size * width_scale;
    let points = match kind {
        LeafKind::Broad => [
            centre - axis * 0.82 - width * 0.18,
            centre - axis * 0.20 + width,
            centre + axis * 0.88 + width * 0.10,
            centre + axis * 0.72 - width * 0.78,
            centre - axis * 0.05 - width,
        ],
        LeafKind::Pointed => [
            centre - axis * 0.94 - width * 0.10,
            centre - axis * 0.10 + width,
            centre + axis * 1.10,
            centre + axis * 0.48 - width * 0.64,
            centre - axis * 0.16 - width * 0.74,
        ],
        LeafKind::Needle => [
            centre - axis * 1.05 - width * 0.12,
            centre - axis * 0.25 + width,
            centre + axis * 1.15 + width * 0.08,
            centre + axis * 0.90 - width * 0.72,
            centre - axis * 0.05 - width,
        ],
        LeafKind::Willow => [
            centre - axis * 1.02 - width * 0.10,
            centre - axis * 0.24 + width,
            centre + axis * 1.22,
            centre + axis * 0.88 - width * 0.68,
            centre - axis * 0.10 - width,
        ],
    };
    let ids = points.map(|point| mesh.add_point(point).expect("tree leaf point"));
    for (id, uv) in ids.into_iter().zip([
        [0.05, 0.45],
        [0.35, 0.05],
        [0.95, 0.42],
        [0.72, 0.95],
        [0.05, 0.75],
    ]) {
        mesh.set_point_uv(id, uv).expect("tree leaf uv");
        let leaf_color = match kind {
            LeafKind::Broad => Color::rgb(58, 124, 48),
            LeafKind::Pointed => Color::rgb(76, 142, 58),
            LeafKind::Needle => Color::rgb(36, 102, 42),
            LeafKind::Willow => Color::rgb(88, 148, 54),
        };
        mesh.set_point_color(id, leaf_color)
            .expect("tree leaf color");
    }
    for tri in [
        [ids[0], ids[1], ids[2]],
        [ids[0], ids[2], ids[3]],
        [ids[0], ids[3], ids[4]],
        [ids[4], ids[3], ids[2]],
        [ids[4], ids[2], ids[1]],
        [ids[4], ids[1], ids[0]],
    ] {
        mesh.add_triangle(tri[0], tri[1], tri[2])
            .expect("tree leaf triangle");
    }
}
fn hash(a: u32, b: u32) -> u32 {
    let mut x = a.wrapping_mul(0x9E3779B9).wrapping_add(b);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EBCA6B);
    x ^ (x >> 13)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn branches_are_connected_and_tapered() {
        let tree = TreeAsset::generate(Vec3::ZERO, TreeSettings::new(TreeKind::Oak, 7));
        assert!(tree.branch_count() > 20);
        for branch in tree.branches() {
            assert!(branch.start_radius > branch.end_radius && branch.end_radius > 0.0);
            if let Some(parent) = branch.parent {
                let p = tree.branches[parent];
                let on_parent = (branch.start - p.start).cross(p.end - p.start).length() < 0.001;
                assert!(on_parent, "child branch must attach to parent segment");
            }
        }
    }
    #[test]
    fn lods_reduce_mesh_complexity_and_keep_a_crown() {
        let settings = TreeSettings::new(TreeKind::Oak, 19);
        let hero = TreeAsset::generate_lod(Vec3::ZERO, settings, TreeLod::Hero);
        let mid = TreeAsset::generate_lod(Vec3::ZERO, settings, TreeLod::Mid);
        let far = TreeAsset::generate_lod(Vec3::ZERO, settings, TreeLod::Far);
        assert!(
            hero.mesh().expect("hero mesh").face_count()
                > mid.mesh().expect("mid mesh").face_count()
        );
        assert!(
            mid.mesh().expect("mid mesh").face_count() > far.mesh().expect("far mesh").face_count()
        );
        assert!(hero.foliage_cluster_count() > mid.foliage_cluster_count());
        assert!(mid.foliage_cluster_count() > 0);
        assert!(far.foliage_cluster_count() > 0);
        assert_eq!(
            far.branches()
                .iter()
                .map(|branch| branch.end().y)
                .fold(0.0, f32::max),
            TreeKind::Oak.profile().height
        );
    }

    #[test]
    fn lod_generation_is_deterministic() {
        for lod in [TreeLod::Hero, TreeLod::Mid, TreeLod::Far] {
            let a =
                TreeAsset::generate_lod(Vec3::ZERO, TreeSettings::new(TreeKind::Spruce, 11), lod);
            let b =
                TreeAsset::generate_lod(Vec3::ZERO, TreeSettings::new(TreeKind::Spruce, 11), lod);
            assert_eq!(a.lod(), lod);
            assert_eq!(
                a.mesh().expect("tree mesh").face_count(),
                b.mesh().expect("tree mesh").face_count()
            );
            assert_eq!(a.branches(), b.branches());
        }
    }

    #[test]
    fn mature_trees_keep_proportional_crowns() {
        for kind in [
            TreeKind::Oak,
            TreeKind::Beech,
            TreeKind::Birch,
            TreeKind::Pine,
            TreeKind::Spruce,
            TreeKind::Willow,
        ] {
            let tree = TreeAsset::generate(Vec3::ZERO, TreeSettings::new(kind, 23));
            let profile = kind.profile();
            let crown_width = tree
                .branches()
                .iter()
                .map(|branch| branch.end().x.abs().max(branch.end().z.abs()))
                .fold(0.0_f32, f32::max);
            assert!(
                crown_width > profile.height * 0.05,
                "{kind:?} crown is too narrow for {} m tree: {crown_width}",
                profile.height
            );
        }
    }

    #[test]
    fn generated_foliage_is_green() {
        for kind in [
            TreeKind::Oak,
            TreeKind::Beech,
            TreeKind::Birch,
            TreeKind::Pine,
            TreeKind::Spruce,
            TreeKind::Willow,
        ] {
            let mesh = TreeAsset::generate(Vec3::ZERO, TreeSettings::new(kind, 19))
                .mesh()
                .expect("tree mesh");
            let built = mesh.build();
            assert!(
                built
                    .colors
                    .iter()
                    .any(|color| color.y > color.x && color.y > color.z),
                "{kind:?} has no green foliage"
            );
        }
    }
    #[test]
    fn generation_is_deterministic() {
        let a = TreeAsset::generate(Vec3::ZERO, TreeSettings::new(TreeKind::Spruce, 11));
        let b = TreeAsset::generate(Vec3::ZERO, TreeSettings::new(TreeKind::Spruce, 11));
        assert_eq!(a.branch_count(), b.branch_count());
        assert_eq!(a.foliage_cluster_count(), b.foliage_cluster_count());
        assert_eq!(a.branches[4].end, b.branches[4].end);
    }
    #[test]
    fn all_species_have_lower_greenery() {
        for kind in [
            TreeKind::Oak,
            TreeKind::Beech,
            TreeKind::Birch,
            TreeKind::Pine,
            TreeKind::Spruce,
            TreeKind::Willow,
        ] {
            let tree = TreeAsset::generate(Vec3::ZERO, TreeSettings::new(kind, 3));
            let lowest = tree
                .branches()
                .iter()
                .skip(1)
                .map(|b| b.end.y)
                .fold(f32::INFINITY, f32::min);
            assert!(lowest < kind.profile().height * 0.75);
        }
    }
}
