//! Per-species growth parameters, ported from AssetGenerator's vegetation package.

use std::f32::consts::PI;

/// The tree species the generator knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TreeSpecies {
    Plane,
    Linden,
    Maple,
    Oak,
    Birch,
    Pine,
    Palm,
    Cypress,
    Chestnut,
    Ginkgo,
}

impl TreeSpecies {
    pub const ALL: [TreeSpecies; 10] = [
        TreeSpecies::Plane,
        TreeSpecies::Linden,
        TreeSpecies::Maple,
        TreeSpecies::Oak,
        TreeSpecies::Birch,
        TreeSpecies::Pine,
        TreeSpecies::Palm,
        TreeSpecies::Cypress,
        TreeSpecies::Chestnut,
        TreeSpecies::Ginkgo,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TreeSpecies::Plane => "plane",
            TreeSpecies::Linden => "linden",
            TreeSpecies::Maple => "maple",
            TreeSpecies::Oak => "oak",
            TreeSpecies::Birch => "birch",
            TreeSpecies::Pine => "pine",
            TreeSpecies::Palm => "palm",
            TreeSpecies::Cypress => "cypress",
            TreeSpecies::Chestnut => "chestnut",
            TreeSpecies::Ginkgo => "ginkgo",
        }
    }

    /// Needle and scale-leaf trees let less light through their crowns.
    pub fn is_conifer(self) -> bool {
        matches!(self, TreeSpecies::Pine | TreeSpecies::Cypress)
    }
}

/// What the generator builds: a tree species or a shrub.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Tree(TreeSpecies),
    Shrub,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Tree(s) => s.name(),
            Kind::Shrub => "shrub",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LenRef {
    Width,
    Parent,
}

/// One branching level: how many children, where on the parent, at what angle.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Level {
    pub n: (u32, u32),
    pub start: f32,
    pub end: f32,
    pub angle: f32,
    pub angle_var: f32,
    /// Multiple of the crown radius (`Width`) or of the parent's length.
    pub len: f32,
    pub len_ref: LenRef,
    /// Child base radius as a fraction of the parent's radius there.
    pub rr: f32,
    pub segs: u32,
    pub radial: u32,
    pub up: f32,
    pub curve: f32,
    pub phyllo: f32,
    pub whorl: Option<u32>,
}

const fn lv(n: (u32, u32)) -> Level {
    Level {
        n,
        start: 0.2,
        end: 1.0,
        angle: 45.0,
        angle_var: 12.0,
        len: 0.5,
        len_ref: LenRef::Parent,
        rr: 0.55,
        segs: 4,
        radial: 5,
        up: 0.05,
        curve: 0.25,
        phyllo: 137.5,
        whorl: None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CardMode {
    Cluster,
    Flat,
    Up,
    Tuft,
}

#[derive(Clone, Debug)]
pub(crate) struct SpeciesCfg {
    pub height: (f32, f32),
    pub trunk_r: f32,
    /// Bark atlas strip.
    pub bark: u32,
    /// Leaf atlas cell.
    pub leaf_cell: u32,
    pub leaf_size: f32,
    pub density: f32,
    pub leaf_start: f32,
    pub card_mode: CardMode,
    /// Crown radius (m).
    pub width: f32,
    /// Crown base as a fraction of the height.
    pub crown_base: f32,
    /// Crown radius profile over its height (0 base, 1 top).
    pub prof: fn(f32) -> f32,
    pub trunk_top: f32,
    pub trunk_up: f32,
    pub trunk_curve: f32,
    pub lean: f32,
    pub flare: f32,
    pub lobes: f32,
    pub levels: Vec<Level>,
    /// Levels whose branches carry leaves (default: the last level).
    pub leaf_levels: Option<Vec<usize>>,
    /// Cards on the trunk itself per metre (columnar conifers).
    pub trunk_leaves: Option<f32>,
    /// Far canopy colour and autumn colour (sRGB).
    pub far_color: [f32; 3],
    pub autumn: [f32; 3],
    pub evergreen: bool,
    /// Crossed cards per tuft.
    pub cross: u32,
    /// Target number of leaf tufts (overrides density).
    pub cards: Option<u32>,
}

fn sin_prof(t: f32, a: f32, b: f32) -> f32 {
    (PI * t.powf(a)).sin().max(0.0).powf(b)
}

pub(crate) fn species_cfg(kind: Kind, variant: u32) -> SpeciesCfg {
    let base = SpeciesCfg {
        height: (13.0, 16.0),
        trunk_r: 0.3,
        bark: 0,
        leaf_cell: 1,
        leaf_size: 1.15,
        density: 5.5,
        leaf_start: 0.2,
        card_mode: CardMode::Cluster,
        width: 4.6,
        crown_base: 0.2,
        prof: |t| sin_prof(t, 0.85, 0.7),
        trunk_top: 0.93,
        trunk_up: 0.02,
        trunk_curve: 0.05,
        lean: 0.02,
        flare: 0.3,
        lobes: 0.03,
        levels: Vec::new(),
        leaf_levels: None,
        trunk_leaves: None,
        far_color: [0.3, 0.42, 0.18],
        autumn: [0.82, 0.68, 0.22],
        evergreen: false,
        cross: 1,
        cards: None,
    };
    match kind {
        Kind::Tree(TreeSpecies::Plane) => SpeciesCfg {
            height: (14.0, 18.0),
            trunk_r: 0.36,
            bark: 3,
            leaf_cell: 0,
            cards: Some(1700),
            leaf_size: 1.35,
            density: 4.2,
            leaf_start: 0.25,
            width: 7.2,
            crown_base: 0.24,
            prof: |t| sin_prof(t, 0.8, 0.55),
            trunk_top: 0.3,
            trunk_up: 0.0,
            trunk_curve: 0.08,
            lean: 0.04,
            flare: 0.35,
            lobes: 0.06,
            levels: vec![
                Level {
                    start: 1.0,
                    end: 1.0,
                    whorl: Some(4),
                    angle: 32.0,
                    angle_var: 10.0,
                    len: 1.25,
                    len_ref: LenRef::Width,
                    rr: 0.72,
                    segs: 8,
                    radial: 8,
                    up: 0.06,
                    curve: 0.22,
                    ..lv((3, 4))
                },
                Level {
                    start: 0.18,
                    end: 0.98,
                    angle: 52.0,
                    angle_var: 15.0,
                    len: 0.55,
                    rr: 0.5,
                    segs: 4,
                    radial: 5,
                    up: 0.04,
                    curve: 0.3,
                    ..lv((8, 10))
                },
                Level {
                    start: 0.3,
                    end: 1.0,
                    angle: 48.0,
                    angle_var: 15.0,
                    len: 0.45,
                    rr: 0.55,
                    segs: 2,
                    radial: 3,
                    up: 0.0,
                    curve: 0.3,
                    ..lv((3, 4))
                },
            ],
            far_color: [0.36, 0.45, 0.22],
            autumn: [0.72, 0.55, 0.25],
            ..base
        },
        Kind::Tree(TreeSpecies::Linden) => SpeciesCfg {
            cards: Some(1600),
            levels: vec![
                Level {
                    start: 0.2,
                    end: 0.92,
                    angle: 58.0,
                    angle_var: 12.0,
                    len: 1.5,
                    len_ref: LenRef::Width,
                    rr: 0.42,
                    segs: 5,
                    radial: 6,
                    up: 0.1,
                    curve: 0.2,
                    ..lv((15, 19))
                },
                Level {
                    start: 0.2,
                    end: 0.95,
                    angle: 48.0,
                    angle_var: 14.0,
                    len: 0.5,
                    rr: 0.55,
                    segs: 3,
                    radial: 4,
                    up: -0.04,
                    curve: 0.3,
                    ..lv((5, 7))
                },
            ],
            leaf_levels: Some(vec![1, 2]),
            ..base
        },
        Kind::Tree(TreeSpecies::Maple) => SpeciesCfg {
            height: (11.0, 14.0),
            leaf_cell: 2,
            cards: Some(1500),
            leaf_size: 1.2,
            density: 4.6,
            leaf_start: 0.25,
            width: 5.2,
            crown_base: 0.26,
            prof: |t| sin_prof(t, 0.9, 0.5),
            trunk_top: 0.3,
            trunk_up: 0.0,
            trunk_curve: 0.06,
            lean: 0.03,
            lobes: 0.04,
            levels: vec![
                Level {
                    start: 1.0,
                    end: 1.0,
                    whorl: Some(5),
                    angle: 30.0,
                    angle_var: 10.0,
                    len: 1.3,
                    len_ref: LenRef::Width,
                    rr: 0.65,
                    segs: 7,
                    radial: 7,
                    up: 0.08,
                    curve: 0.2,
                    ..lv((4, 5))
                },
                Level {
                    start: 0.2,
                    end: 0.98,
                    angle: 50.0,
                    angle_var: 14.0,
                    len: 0.55,
                    rr: 0.5,
                    segs: 3,
                    radial: 4,
                    up: 0.05,
                    curve: 0.3,
                    ..lv((7, 9))
                },
                Level {
                    start: 0.35,
                    end: 1.0,
                    angle: 45.0,
                    angle_var: 12.0,
                    len: 0.45,
                    rr: 0.55,
                    segs: 2,
                    radial: 3,
                    up: 0.02,
                    curve: 0.3,
                    ..lv((3, 3))
                },
            ],
            far_color: [0.27, 0.4, 0.16],
            autumn: [0.85, 0.3, 0.08],
            ..base
        },
        Kind::Tree(TreeSpecies::Oak) => SpeciesCfg {
            height: (12.0, 15.0),
            trunk_r: 0.46,
            bark: 1,
            leaf_cell: 3,
            cards: Some(1700),
            density: 4.6,
            leaf_start: 0.25,
            width: 8.0,
            prof: |t| sin_prof(t, 0.72, 0.45),
            trunk_top: 0.26,
            trunk_up: 0.0,
            trunk_curve: 0.14,
            lean: 0.06,
            flare: 0.45,
            lobes: 0.1,
            levels: vec![
                Level {
                    start: 0.7,
                    end: 1.0,
                    whorl: Some(1),
                    phyllo: 140.0,
                    angle: 66.0,
                    angle_var: 12.0,
                    len: 1.3,
                    len_ref: LenRef::Width,
                    rr: 0.62,
                    segs: 9,
                    radial: 8,
                    up: 0.0,
                    curve: 0.95,
                    ..lv((4, 5))
                },
                Level {
                    start: 0.15,
                    end: 0.98,
                    angle: 60.0,
                    angle_var: 18.0,
                    len: 0.55,
                    rr: 0.5,
                    segs: 4,
                    radial: 4,
                    up: 0.02,
                    curve: 0.85,
                    ..lv((7, 9))
                },
                Level {
                    start: 0.3,
                    end: 1.0,
                    angle: 50.0,
                    angle_var: 18.0,
                    len: 0.45,
                    rr: 0.55,
                    segs: 2,
                    radial: 3,
                    up: 0.02,
                    curve: 0.5,
                    ..lv((3, 4))
                },
            ],
            far_color: [0.26, 0.36, 0.15],
            autumn: [0.55, 0.36, 0.12],
            ..base
        },
        Kind::Tree(TreeSpecies::Birch) => SpeciesCfg {
            trunk_r: 0.17,
            bark: 2,
            leaf_cell: 4,
            cards: Some(1150),
            leaf_size: 0.95,
            density: 6.5,
            leaf_start: 0.15,
            width: 3.0,
            crown_base: 0.3,
            prof: |t| sin_prof(t, 0.75, 0.9) * (1.0 - 0.25 * t),
            trunk_top: 0.96,
            lean: 0.05,
            trunk_curve: 0.07,
            flare: 0.15,
            lobes: 0.0,
            levels: vec![
                Level {
                    start: 0.28,
                    end: 0.95,
                    angle: 48.0,
                    angle_var: 12.0,
                    len: 1.6,
                    len_ref: LenRef::Width,
                    rr: 0.36,
                    segs: 5,
                    radial: 4,
                    up: 0.02,
                    curve: 0.25,
                    ..lv((20, 26))
                },
                Level {
                    start: 0.25,
                    end: 0.95,
                    angle: 40.0,
                    angle_var: 14.0,
                    len: 0.6,
                    rr: 0.55,
                    segs: 3,
                    radial: 3,
                    up: -0.22,
                    curve: 0.2,
                    ..lv((4, 5))
                },
            ],
            leaf_levels: Some(vec![1, 2]),
            far_color: [0.38, 0.5, 0.22],
            autumn: [0.9, 0.75, 0.25],
            ..base
        },
        Kind::Tree(TreeSpecies::Pine) if variant & 1 == 0 => SpeciesCfg {
            // Stone / umbrella pine: tall bare trunk, flat spreading crown.
            height: (12.0, 15.0),
            bark: 4,
            leaf_cell: 8,
            cards: Some(950),
            leaf_size: 1.05,
            density: 5.0,
            leaf_start: 0.35,
            card_mode: CardMode::Tuft,
            width: 6.0,
            crown_base: 0.58,
            prof: |t| {
                if t < 0.45 {
                    (t / 0.45).powf(0.6)
                } else {
                    (1.0 - ((t - 0.45) / 0.55).powf(2.4)).max(0.0).sqrt()
                }
            },
            trunk_top: 0.6,
            trunk_up: 0.03,
            trunk_curve: 0.12,
            lean: 0.1,
            flare: 0.25,
            levels: vec![
                Level {
                    start: 1.0,
                    end: 1.0,
                    whorl: Some(5),
                    angle: 42.0,
                    angle_var: 10.0,
                    len: 1.3,
                    len_ref: LenRef::Width,
                    rr: 0.65,
                    segs: 6,
                    radial: 6,
                    up: 0.1,
                    curve: 0.3,
                    ..lv((4, 5))
                },
                Level {
                    start: 0.35,
                    end: 0.98,
                    angle: 45.0,
                    angle_var: 14.0,
                    len: 0.5,
                    rr: 0.5,
                    segs: 3,
                    radial: 4,
                    up: 0.08,
                    curve: 0.3,
                    ..lv((5, 7))
                },
                Level {
                    start: 0.4,
                    end: 1.0,
                    angle: 40.0,
                    angle_var: 14.0,
                    len: 0.4,
                    rr: 0.55,
                    segs: 2,
                    radial: 3,
                    up: 0.1,
                    curve: 0.3,
                    ..lv((3, 4))
                },
            ],
            far_color: [0.17, 0.3, 0.14],
            autumn: [0.17, 0.3, 0.14],
            evergreen: true,
            cross: 2,
            ..base
        },
        Kind::Tree(TreeSpecies::Pine) => SpeciesCfg {
            // Conical, whorled pine / spruce form.
            height: (12.0, 16.0),
            trunk_r: 0.27,
            bark: 4,
            leaf_cell: 10,
            cards: Some(1800),
            leaf_size: 1.1,
            density: 3.4,
            leaf_start: 0.1,
            card_mode: CardMode::Flat,
            width: 3.4,
            crown_base: 0.1,
            prof: |t| (1.0 - t).max(0.0).powf(0.95) * (0.55 + t * 4.0).min(1.0),
            trunk_top: 1.0,
            trunk_up: 0.03,
            trunk_curve: 0.03,
            lean: 0.015,
            flare: 0.25,
            lobes: 0.02,
            levels: vec![Level {
                whorl: Some(4),
                phyllo: 41.0,
                start: 0.1,
                end: 0.97,
                angle: 82.0,
                angle_var: 8.0,
                len: 1.4,
                len_ref: LenRef::Width,
                rr: 0.32,
                segs: 4,
                radial: 4,
                up: -0.02,
                curve: 0.12,
                ..lv((22, 26))
            }],
            leaf_levels: Some(vec![1]),
            trunk_leaves: Some(3.0),
            far_color: [0.13, 0.25, 0.14],
            autumn: [0.13, 0.25, 0.14],
            evergreen: true,
            ..base
        },
        Kind::Tree(TreeSpecies::Cypress) => SpeciesCfg {
            height: (12.0, 16.0),
            trunk_r: 0.24,
            bark: 6,
            leaf_cell: 9,
            cards: Some(1500),
            leaf_size: 0.72,
            density: 9.0,
            leaf_start: 0.0,
            card_mode: CardMode::Up,
            width: 1.0,
            crown_base: 0.03,
            prof: |t| (t * 5.0 + 0.25).min(1.0).powf(0.5) * (1.0 - t).max(0.0).powf(0.62),
            trunk_top: 0.98,
            trunk_up: 0.05,
            trunk_curve: 0.02,
            lean: 0.01,
            flare: 0.2,
            lobes: 0.04,
            levels: vec![Level {
                start: 0.04,
                end: 0.96,
                angle: 22.0,
                angle_var: 8.0,
                len: 2.2,
                len_ref: LenRef::Width,
                rr: 0.32,
                segs: 3,
                radial: 3,
                up: 0.25,
                curve: 0.15,
                ..lv((44, 56))
            }],
            leaf_levels: Some(vec![1]),
            trunk_leaves: Some(9.0),
            far_color: [0.11, 0.22, 0.1],
            autumn: [0.11, 0.22, 0.1],
            evergreen: true,
            ..base
        },
        Kind::Tree(TreeSpecies::Chestnut) => SpeciesCfg {
            height: (14.0, 17.0),
            trunk_r: 0.42,
            bark: 1,
            leaf_cell: 5,
            cards: Some(1700),
            leaf_size: 1.35,
            density: 4.4,
            leaf_start: 0.25,
            width: 6.2,
            crown_base: 0.24,
            prof: |t| sin_prof(t, 0.8, 0.55),
            trunk_top: 0.32,
            trunk_up: 0.0,
            trunk_curve: 0.08,
            lean: 0.03,
            flare: 0.35,
            lobes: 0.06,
            levels: vec![
                Level {
                    start: 1.0,
                    end: 1.0,
                    whorl: Some(4),
                    angle: 30.0,
                    angle_var: 9.0,
                    len: 1.3,
                    len_ref: LenRef::Width,
                    rr: 0.7,
                    segs: 8,
                    radial: 8,
                    up: 0.07,
                    curve: 0.2,
                    ..lv((3, 4))
                },
                Level {
                    start: 0.15,
                    end: 0.98,
                    angle: 55.0,
                    angle_var: 14.0,
                    len: 0.55,
                    rr: 0.5,
                    segs: 4,
                    radial: 4,
                    up: -0.03,
                    curve: 0.25,
                    ..lv((8, 10))
                },
                Level {
                    start: 0.3,
                    end: 1.0,
                    angle: 45.0,
                    angle_var: 12.0,
                    len: 0.45,
                    rr: 0.55,
                    segs: 2,
                    radial: 3,
                    up: 0.06,
                    curve: 0.25,
                    ..lv((3, 4))
                },
            ],
            far_color: [0.24, 0.36, 0.14],
            autumn: [0.62, 0.42, 0.16],
            ..base
        },
        Kind::Tree(TreeSpecies::Ginkgo) => SpeciesCfg {
            height: (12.0, 15.0),
            trunk_r: 0.27,
            bark: 7,
            leaf_cell: 6,
            cards: Some(1150),
            leaf_size: 1.0,
            density: 4.6,
            width: 3.6,
            prof: |t| sin_prof(t, 0.62, 0.75) * (1.0 - 0.3 * t),
            trunk_top: 0.94,
            lean: 0.03,
            flare: 0.2,
            levels: vec![
                Level {
                    start: 0.2,
                    end: 0.9,
                    angle: 48.0,
                    angle_var: 20.0,
                    len: 1.5,
                    len_ref: LenRef::Width,
                    rr: 0.42,
                    segs: 5,
                    radial: 5,
                    up: 0.16,
                    curve: 0.3,
                    ..lv((10, 13))
                },
                Level {
                    start: 0.2,
                    end: 0.95,
                    angle: 48.0,
                    angle_var: 16.0,
                    len: 0.45,
                    rr: 0.55,
                    segs: 3,
                    radial: 3,
                    up: 0.1,
                    curve: 0.3,
                    ..lv((4, 5))
                },
            ],
            leaf_levels: Some(vec![1, 2]),
            far_color: [0.36, 0.48, 0.2],
            autumn: [0.98, 0.78, 0.12],
            ..base
        },
        Kind::Shrub => {
            let boxwood = variant & 1 == 1;
            SpeciesCfg {
                height: (1.2, 1.8),
                trunk_r: 0.035,
                bark: 7,
                leaf_cell: if boxwood { 11 } else { 7 },
                cards: Some(320),
                leaf_size: 0.5,
                density: 15.0,
                leaf_start: 0.1,
                width: 0.85,
                crown_base: 0.04,
                prof: |t| sin_prof(t, 0.75, 0.55),
                trunk_top: 0.02,
                trunk_up: 0.0,
                trunk_curve: 0.0,
                lean: 0.0,
                flare: 0.0,
                lobes: 0.0,
                levels: vec![
                    Level {
                        start: 1.0,
                        end: 1.0,
                        whorl: Some(7),
                        angle: 38.0,
                        angle_var: 16.0,
                        len: 1.6,
                        len_ref: LenRef::Width,
                        rr: 1.0,
                        segs: 4,
                        radial: 4,
                        up: 0.2,
                        curve: 0.3,
                        ..lv((6, 7))
                    },
                    Level {
                        start: 0.25,
                        end: 0.95,
                        angle: 45.0,
                        angle_var: 14.0,
                        len: 0.5,
                        rr: 0.6,
                        segs: 2,
                        radial: 3,
                        up: 0.1,
                        curve: 0.3,
                        ..lv((3, 4))
                    },
                ],
                leaf_levels: Some(vec![1, 2]),
                far_color: if boxwood {
                    [0.16, 0.28, 0.12]
                } else {
                    [0.27, 0.4, 0.17]
                },
                autumn: if boxwood {
                    [0.16, 0.28, 0.12]
                } else {
                    [0.7, 0.35, 0.12]
                },
                evergreen: boxwood,
                ..base
            }
        }
        // The palm has its own builder; this is only a fallback.
        Kind::Tree(TreeSpecies::Palm) => species_cfg(Kind::Tree(TreeSpecies::Linden), variant),
    }
}
