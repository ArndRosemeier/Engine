//! Depth-stencil format and stencil-mask states for recursive portal rendering.

use wgpu::{CompareFunction, DepthStencilState, StencilFaceState, StencilOperation, StencilState};

/// The world's depth format: depth with a stencil for portal masks.
/// `ENGINE_DEPTH_FORMAT=d32` picks plain 32-bit depth (portals then draw
/// without their stencil mask); a test of driver trouble with the
/// combined format.
static DEPTH_FORMAT: std::sync::OnceLock<wgpu::TextureFormat> = std::sync::OnceLock::new();

pub(crate) fn depth_format() -> wgpu::TextureFormat {
    *DEPTH_FORMAT.get_or_init(|| pick_depth_format(true))
}

/// Pick the depth format once at startup: `ENGINE_DEPTH_FORMAT=d32` or a
/// GPU without `Depth32FloatStencil8` gets plain 32-bit depth.
pub(crate) fn choose_depth_format(stencil_supported: bool) {
    let _ = DEPTH_FORMAT.set(pick_depth_format(stencil_supported));
}

fn pick_depth_format(stencil_supported: bool) -> wgpu::TextureFormat {
    match std::env::var("ENGINE_DEPTH_FORMAT").as_deref() {
        Ok("d32") => wgpu::TextureFormat::Depth32Float,
        _ if !stencil_supported => wgpu::TextureFormat::Depth32Float,
        _ => wgpu::TextureFormat::Depth32FloatStencil8,
    }
}

/// Whether [`depth_format`] has a stencil.
pub(crate) fn has_stencil() -> bool {
    depth_format().has_stencil_aspect()
}

/// A stencil state, or none when the depth format has no stencil.
fn gate(state: StencilState) -> StencilState {
    if has_stencil() {
        state
    } else {
        StencilState::default()
    }
}

/// MSAA sample count of the world pass, fixed once when the renderer starts.
static SCENE_SAMPLES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

/// Fix the world-pass sample count. Must run before any scene pipeline exists.
pub(crate) fn init_scene_samples(samples: u32) {
    let stored = *SCENE_SAMPLES.get_or_init(|| samples);
    if stored != samples {
        panic!("scene MSAA already fixed at {stored}x, cannot switch to {samples}x");
    }
}

pub(crate) fn scene_samples() -> u32 {
    *SCENE_SAMPLES
        .get()
        .expect("scene MSAA sample count is set when the renderer starts")
}

/// Multisample state for pipelines drawing into the world pass.
///
/// `alpha_to_coverage` turns fragment alpha into MSAA coverage, which gives
/// cut-out foliage soft, stable edges instead of a stair-stepped discard.
pub(crate) fn scene_multisample(alpha_to_coverage: bool) -> wgpu::MultisampleState {
    let count = scene_samples();
    wgpu::MultisampleState {
        count,
        mask: !0,
        alpha_to_coverage_enabled: alpha_to_coverage && count > 1,
    }
}

fn face_always_keep() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    }
}

fn face_equal_keep() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Equal,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    }
}

/// Level-0 scene draws ignore the portal stencil mask.
pub(crate) fn scene_depth_stencil_unmasked_write(depth_write: bool) -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: depth_write,
        depth_compare: super::DEPTH_COMPARE,
        stencil: gate(StencilState {
            front: face_always_keep(),
            back: face_always_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        }),
        bias: Default::default(),
    }
}

pub(crate) fn scene_depth_stencil_unmasked() -> DepthStencilState {
    scene_depth_stencil_unmasked_write(true)
}

/// Recursive portal views only draw where stencil == the pass reference.
pub(crate) fn scene_depth_stencil_masked_write(depth_write: bool) -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: depth_write,
        depth_compare: super::DEPTH_COMPARE,
        stencil: gate(StencilState {
            front: face_equal_keep(),
            back: face_equal_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        }),
        bias: Default::default(),
    }
}

pub(crate) fn scene_depth_stencil_masked() -> DepthStencilState {
    scene_depth_stencil_masked_write(true)
}

pub(crate) fn sky_depth_stencil_unmasked() -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: gate(StencilState {
            front: face_always_keep(),
            back: face_always_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        }),
        bias: Default::default(),
    }
}

/// Sky pass inside a recursive portal view.
pub(crate) fn sky_depth_stencil_masked() -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: gate(StencilState {
            front: face_equal_keep(),
            back: face_equal_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        }),
        bias: Default::default(),
    }
}

/// Write portal opening depth; colour writes are disabled on the pipeline.
pub(crate) fn portal_depth_write_depth_stencil() -> DepthStencilState {
    scene_depth_stencil_unmasked()
}

fn face_incr_on_pass() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Equal,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::IncrementClamp,
    }
}

/// Bias so a doorway opening wins over coplanar floor geometry (reversed-Z).
const PORTAL_STENCIL_DEPTH_BIAS: i32 = 4;

/// Mark the portal opening in stencil. Must respect scene depth so horizontal
/// occluders (e.g. an upper-storey floor seen from above) do not show the
/// linked view through solid geometry. A small depth bias plus the vertex
/// nudge in the portal mesh shader keeps coplanar doorway floors working.
pub(crate) fn stencil_incr_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: true,
        depth_compare: super::DEPTH_COMPARE,
        stencil: gate(StencilState {
            front: face_incr_on_pass(),
            back: face_incr_on_pass(),
            read_mask: 0xff,
            write_mask: 0xff,
        }),
        bias: wgpu::DepthBiasState {
            constant: PORTAL_STENCIL_DEPTH_BIAS,
            slope_scale: 1.0,
            clamp: 0.0,
        },
    }
}

fn face_decr_on_pass() -> StencilFaceState {
    StencilFaceState {
        compare: CompareFunction::Equal,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::DecrementClamp,
    }
}

/// Restore stencil after a recursive portal sub-pass, and the opening's
/// own depth over the view drawn through it. The view was drawn with the
/// linked (oblique) camera, so its depths mean nothing to this level's
/// camera: left there, temporal AA reprojected the doorway's history from
/// the wrong places (a ghost of the view flickering over the door while
/// the camera moved), and later see-through things tested against them.
/// The stencil limits the write to the doorway's own pixels.
pub(crate) fn stencil_decr_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: true,
        depth_compare: CompareFunction::Always,
        stencil: gate(StencilState {
            front: face_decr_on_pass(),
            back: face_decr_on_pass(),
            read_mask: 0xff,
            write_mask: 0xff,
        }),
        bias: Default::default(),
    }
}

/// Clear depth inside a portal mask (reversed-Z far plane).
pub(crate) fn depth_clear_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: depth_format(),
        depth_write_enabled: true,
        depth_compare: CompareFunction::Always,
        stencil: gate(StencilState {
            front: face_equal_keep(),
            back: face_equal_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        }),
        bias: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaving_a_portal_view_puts_back_the_openings_depth() {
        let d = stencil_decr_depth_stencil();
        assert!(d.depth_write_enabled, "the doorway gets its own depth back");
        assert_eq!(d.depth_compare, CompareFunction::Always);
    }

    #[test]
    fn stencil_incr_respects_scene_depth() {
        let state = stencil_incr_depth_stencil();
        assert_eq!(state.depth_compare, super::super::DEPTH_COMPARE);
        assert!(state.depth_write_enabled);
        assert_ne!(state.bias.constant, 0);
    }
}
