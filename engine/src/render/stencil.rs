//! Depth-stencil format and stencil-mask states for recursive portal rendering.

use wgpu::{CompareFunction, DepthStencilState, StencilFaceState, StencilOperation, StencilState};

pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32FloatStencil8;

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
        format: DEPTH_FORMAT,
        depth_write_enabled: depth_write,
        depth_compare: super::DEPTH_COMPARE,
        stencil: StencilState {
            front: face_always_keep(),
            back: face_always_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        },
        bias: Default::default(),
    }
}

pub(crate) fn scene_depth_stencil_unmasked() -> DepthStencilState {
    scene_depth_stencil_unmasked_write(true)
}

/// Recursive portal views only draw where stencil == the pass reference.
pub(crate) fn scene_depth_stencil_masked_write(depth_write: bool) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: depth_write,
        depth_compare: super::DEPTH_COMPARE,
        stencil: StencilState {
            front: face_equal_keep(),
            back: face_equal_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        },
        bias: Default::default(),
    }
}

pub(crate) fn scene_depth_stencil_masked() -> DepthStencilState {
    scene_depth_stencil_masked_write(true)
}

pub(crate) fn sky_depth_stencil_unmasked() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: StencilState {
            front: face_always_keep(),
            back: face_always_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        },
        bias: Default::default(),
    }
}

/// Sky pass inside a recursive portal view.
pub(crate) fn sky_depth_stencil_masked() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: StencilState {
            front: face_equal_keep(),
            back: face_equal_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        },
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

/// Mark the portal opening in stencil. Scene geometry in the doorway must not
/// block this pass — floor boxes often share the opening footprint.
pub(crate) fn stencil_incr_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Always,
        stencil: StencilState {
            front: face_incr_on_pass(),
            back: face_incr_on_pass(),
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: Default::default(),
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

/// Restore stencil after a recursive portal sub-pass.
pub(crate) fn stencil_decr_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: StencilState {
            front: face_decr_on_pass(),
            back: face_decr_on_pass(),
            read_mask: 0xff,
            write_mask: 0xff,
        },
        bias: Default::default(),
    }
}

/// Clear depth inside a portal mask (reversed-Z far plane).
pub(crate) fn depth_clear_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Always,
        stencil: StencilState {
            front: face_equal_keep(),
            back: face_equal_keep(),
            read_mask: 0xff,
            write_mask: 0x00,
        },
        bias: Default::default(),
    }
}
