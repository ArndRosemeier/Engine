//! Texture arrays: every mesh material in one bind group, so meshes with
//! different textures can share one multi-draw call.
//!
//! The table holds each material's four maps in binding arrays and its
//! factors in a storage buffer. Each instance carries its mesh's slot
//! (`InstanceRaw::material`), and the bindless scene shader variant reads
//! its textures through that slot. Two bind groups differ only in the
//! sampler: clamped for plain baked albedos, repeating and anisotropic for
//! imported material sets, exactly as the per-texture groups have them.

use super::material::{GpuMaterial, MaterialParams, NeutralMaps};
use crate::texture::TextureId;
use std::collections::HashMap;
use std::num::NonZeroU32;

/// Materials the table holds; ones past it keep their own bind group.
pub const MATERIAL_TABLE_SLOTS: u32 = 4096;

/// Binding-array elements the table needs per shader stage.
pub const MATERIAL_TABLE_ELEMENTS: u32 = MATERIAL_TABLE_SLOTS * 4;

/// The GPU features texture arrays need.
pub const MATERIAL_TABLE_FEATURES: wgpu::Features = wgpu::Features::TEXTURE_BINDING_ARRAY
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

/// Whether binding only the used slots is switched off
/// (`ENGINE_PARTIAL_TABLE=0`: every slot bound, unused ones to fallbacks).
pub fn partial_disabled_by_env() -> bool {
    std::env::var("ENGINE_PARTIAL_TABLE").is_ok_and(|v| v.trim() == "0")
}

/// Whether texture arrays are switched off (`ENGINE_TEXTURE_ARRAYS=0`).
pub fn disabled_by_env() -> bool {
    std::env::var("ENGINE_TEXTURE_ARRAYS").is_ok_and(|v| v.trim() == "0")
}

struct Entry {
    views: [wgpu::TextureView; 4],
    repeat: bool,
}

pub struct MaterialTable {
    pub layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    slots: HashMap<Option<TextureId>, u32>,
    entries: Vec<Option<Entry>>,
    free: Vec<u32>,
    /// Clamped-sampler and repeating-sampler groups, rebuilt on change.
    binds: Option<[wgpu::BindGroup; 2]>,
    /// Goes up whenever the bind groups are made anew.
    version: u64,
    /// Bind only the slots up to the highest one in use
    /// (`PARTIALLY_BOUND_BINDING_ARRAY`); shaders read only those.
    partial: bool,
}

pub fn table_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let array = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: NonZeroU32::new(MATERIAL_TABLE_SLOTS),
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material-table-layout"),
        entries: &[
            array(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            array(2),
            array(3),
            array(4),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

impl MaterialTable {
    pub fn new(device: &wgpu::Device, partial: bool) -> Self {
        Self {
            layout: table_layout(device),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material-table-params"),
                size: MATERIAL_TABLE_SLOTS as u64 * std::mem::size_of::<MaterialParams>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            slots: HashMap::new(),
            entries: Vec::new(),
            free: Vec::new(),
            binds: None,
            version: 0,
            partial,
        }
    }

    /// Add a material under `key` (`None`: the white fallback). `repeat`
    /// picks the imported-material sampler. False when the table is full.
    pub fn insert(
        &mut self,
        queue: &wgpu::Queue,
        key: Option<TextureId>,
        material: &GpuMaterial,
        repeat: bool,
    ) -> bool {
        self.remove(key);
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None if (self.entries.len() as u32) < MATERIAL_TABLE_SLOTS => {
                self.entries.push(None);
                self.entries.len() as u32 - 1
            }
            None => return false,
        };
        queue.write_buffer(
            &self.params,
            slot as u64 * std::mem::size_of::<MaterialParams>() as u64,
            bytemuck::bytes_of(&material.params),
        );
        self.entries[slot as usize] = Some(Entry {
            views: material.views.clone(),
            repeat,
        });
        self.slots.insert(key, slot);
        self.binds = None;
        true
    }

    pub fn remove(&mut self, key: Option<TextureId>) {
        if let Some(slot) = self.slots.remove(&key) {
            self.entries[slot as usize] = None;
            self.free.push(slot);
            self.binds = None;
        }
    }

    /// Keep only the textured materials `keep` says yes to.
    pub fn retain(&mut self, mut keep: impl FnMut(TextureId) -> bool) {
        let gone: Vec<TextureId> = self
            .slots
            .keys()
            .flatten()
            .copied()
            .filter(|&t| !keep(t))
            .collect();
        for t in gone {
            self.remove(Some(t));
        }
    }

    pub fn slot(&self, key: Option<TextureId>) -> Option<u32> {
        self.slots.get(&key).copied()
    }

    /// Whether `slot` samples with the repeating sampler.
    pub fn repeats(&self, slot: u32) -> bool {
        self.entries
            .get(slot as usize)
            .and_then(Option::as_ref)
            .is_some_and(|e| e.repeat)
    }

    /// Rebuild the bind groups if a material came or went.
    pub fn flush(
        &mut self,
        device: &wgpu::Device,
        neutral: &NeutralMaps,
        samplers: [&wgpu::Sampler; 2],
    ) {
        if self.binds.is_some() {
            return;
        }
        let fallback = [
            &neutral.white_srgb,
            &neutral.flat_normal,
            &neutral.white_linear,
            &neutral.white_srgb,
        ];
        let bound = if self.partial {
            let used = self
                .entries
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |i| i + 1);
            used.max(1)
        } else {
            MATERIAL_TABLE_SLOTS as usize
        };
        let maps: [Vec<&wgpu::TextureView>; 4] = std::array::from_fn(|map| {
            (0..bound)
                .map(|slot| match self.entries.get(slot) {
                    Some(Some(e)) => &e.views[map],
                    _ => fallback[map],
                })
                .collect()
        });
        let bind = |sampler: &wgpu::Sampler| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material-table"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureViewArray(&maps[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureViewArray(&maps[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureViewArray(&maps[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureViewArray(&maps[3]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            })
        };
        self.binds = Some([bind(samplers[0]), bind(samplers[1])]);
        self.version += 1;
    }

    /// Changes whenever the groups [`Self::bind_for`] gives change.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The group for `slot`'s sampler (after [`Self::flush`]).
    pub fn bind_for(&self, slot: u32) -> Option<&wgpu::BindGroup> {
        let binds = self.binds.as_ref()?;
        Some(&binds[self.repeats(slot) as usize])
    }
}

/// The scene shader reading its material through the table: the per-mesh
/// texture and factor bindings become arrays indexed by the instance's slot.
pub fn bindless_scene_source(source: &str) -> String {
    let n = MATERIAL_TABLE_SLOTS;
    let mut s = source.to_string();
    let mut swap = |from: &str, to: &str| {
        assert!(
            s.contains(from),
            "bindless scene shader: `{from}` not found"
        );
        s = s.replace(from, to);
    };
    swap(
        "@group(1) @binding(0) var albedo_tex: texture_2d<f32>;",
        &format!(
            "@group(1) @binding(0) var albedo_texs: binding_array<texture_2d<f32>, {n}>;\n\
             var<private> mat_id: u32;"
        ),
    );
    for (one, many) in [
        ("normal_tex", "normal_texs"),
        ("orm_tex", "orm_texs"),
        ("emissive_tex", "emissive_texs"),
    ] {
        swap(
            &format!("var {one}: texture_2d<f32>;"),
            &format!("var {many}: binding_array<texture_2d<f32>, {n}>;"),
        );
        swap(&format!("({one},"), &format!("({many}[mat_id],"));
    }
    swap("(albedo_tex,", "(albedo_texs[mat_id],");
    swap(
        "@group(1) @binding(5) var<uniform> mat: MaterialParams;",
        "@group(1) @binding(5) var<storage, read> mats: array<MaterialParams>;",
    );
    swap(
        "    @location(14) element: u32,\n};",
        "    @location(14) element: u32,\n    @location(15) material: u32,\n};",
    );
    swap(
        "    @location(15) @interpolate(flat) glow: vec4<f32>,\n};",
        "    @location(15) @interpolate(flat) glow: vec4<f32>,\n    @location(9) @interpolate(flat) material: u32,\n};",
    );
    swap(
        "    out.uv = v.uv;\n",
        "    out.uv = v.uv;\n    out.material = v.material;\n",
    );
    swap(
        "fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n",
        "fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n    mat_id = in.material;\n",
    );
    swap(
        "fn fs_depth(in: VsOut) {\n",
        "fn fs_depth(in: VsOut) {\n    mat_id = in.material;\n",
    );
    let s = replace_word(&s, "mat.", "mats[mat_id].");
    assert!(
        !s.contains("(albedo_tex,") && !s.contains("(normal_tex,"),
        "bindless scene shader: a texture read was missed"
    );
    s
}

/// Replace `from` where it is not the tail of a longer name.
fn replace_word(s: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find(from) {
        let before = rest[..at].chars().next_back();
        let part_of_name = before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.');
        out.push_str(&rest[..at]);
        out.push_str(if part_of_name { from } else { to });
        rest = &rest[at + from.len()..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_word_skips_longer_names() {
        assert_eq!(
            replace_word("mat.x + format.y + a.mat.z + (mat.w)", "mat.", "M."),
            "M.x + format.y + a.mat.z + (M.w)"
        );
    }
}
