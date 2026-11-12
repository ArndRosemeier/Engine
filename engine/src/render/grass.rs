//! Grass blades on the clipmap terrain.
//!
//! Nothing is stored: every blade is made in the vertex shader from its
//! instance number. Two grids follow the camera, a fine one near by and one
//! half as dense further out; both snap to their own spacing so blades stay
//! put in the world as the camera moves. Each blade asks the terrain shader
//! for its height and ground layer, so grass grows only where the ground is
//! painted grass or meadow, takes its colour from that texture, sways with
//! the world wind and thins out with distance.

use crate::proc_terrain::Grass;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Most blades drawn per frame (both grids), whatever the settings ask.
pub(crate) const MAX_BLADES: u32 = 600_000;

/// The far grid's spacing over the near one's (it has 1/9 of the blades).
pub(crate) const FAR_STEP: f32 = 3.0;

/// Vertices per blade: three quads tapering to a tip (a triangle strip).
pub(crate) const BLADE_VERTICES: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct GrassUniform {
    /// World XZ of cell (0, 0) of the near and the far grid.
    origin0: [f32; 2],
    origin1: [f32; 2],
    /// Cells per side of each grid.
    n0: u32,
    n1: u32,
    /// Near grid spacing (the far grid is twice as wide).
    spacing: f32,
    /// Half-width of the near grid (m).
    inner: f32,
    /// Where grass ends (m).
    outer: f32,
    height: f32,
    time: f32,
    wind: f32,
}

const _: () = assert!(std::mem::size_of::<GrassUniform>() == 48);

impl GrassUniform {
    /// Grids around `eye` for these settings; `scale` stretches the distance
    /// (quality and LOD bias). Returns the uniform and the blade count.
    pub(crate) fn around(
        eye: Vec3,
        grass: &Grass,
        scale: f32,
        time: f32,
        wind: f32,
    ) -> (Self, u32) {
        let outer = (grass.distance * scale).clamp(0.0, 400.0);
        let spacing = 0.2 / grass.density.clamp(0.05, 4.0).sqrt();
        if outer < 2.0 || grass.height <= 0.0 {
            return (Self::default(), 0);
        }
        let inner = (outer * 0.35).min(16.0);
        let grid = |half: f32, step: f32| {
            let n = ((2.0 * half / step).ceil() as u32 + 1).max(1);
            let x = ((eye.x - half) / step).floor() * step;
            let z = ((eye.z - half) / step).floor() * step;
            (n, [x, z])
        };
        let (mut n0, origin0) = grid(inner, spacing);
        let (mut n1, origin1) = grid(outer, spacing * FAR_STEP);
        // Over budget: drop the far grid first, then shrink the near one.
        if n0 * n0 + n1 * n1 > MAX_BLADES {
            n1 = ((MAX_BLADES.saturating_sub(n0 * n0)) as f32).sqrt() as u32;
            if n1 < 2 {
                n1 = 0;
                n0 = n0.min((MAX_BLADES as f32).sqrt() as u32);
            }
        }
        let u = Self {
            origin0,
            origin1,
            n0,
            n1,
            spacing,
            inner,
            outer,
            height: grass.height,
            time,
            wind,
        };
        (u, n0 * n0 + n1 * n1)
    }
}

pub(crate) const GRASS_WGSL: &str = r#"
struct GrassParams {
    origin0: vec2<f32>,
    origin1: vec2<f32>,
    n0: u32,
    n1: u32,
    spacing: f32,
    inner: f32,
    outer: f32,
    height: f32,
    time: f32,
    wind: f32,
};
@group(0) @binding(3) var<uniform> grass: GrassParams;

struct GrassOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_p: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) color: vec3<f32>,
    // 0 at the root, 1 at the tip.
    @location(3) t: f32,
};

fn grass_none() -> GrassOut {
    var out: GrassOut;
    // Behind the far plane in every API: the triangle is clipped away.
    out.clip = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.world_p = vec3<f32>(0.0);
    out.n = vec3<f32>(0.0, 1.0, 0.0);
    out.color = vec3<f32>(0.0);
    out.t = 0.0;
    return out;
}

@vertex
fn vs_grass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GrassOut {
    // Which grid and cell this blade belongs to.
    var tier = 0u;
    var idx = ii;
    var sp = grass.spacing;
    var origin = grass.origin0;
    var n = grass.n0;
    if ii >= grass.n0 * grass.n0 {
        tier = 1u;
        idx = ii - grass.n0 * grass.n0;
        sp = grass.spacing * 3.0;
        origin = grass.origin1;
        n = grass.n1;
    }
    let cell = vec2<i32>(
        i32(round(origin.x / sp)) + i32(idx % n),
        i32(round(origin.y / sp)) + i32(idx / n),
    );
    let k = terrain.seed ^ (tier * 0x9E3779B9u);
    let r0 = hash21(cell.x, cell.y, k ^ 0x1111u);
    let r1 = hash21(cell.x, cell.y, k ^ 0x2222u);
    let r2 = hash21(cell.x, cell.y, k ^ 0x3333u);
    let r3 = hash21(cell.x, cell.y, k ^ 0x4444u);
    let xz = (vec2<f32>(cell) + vec2<f32>(r0, r1)) * sp;

    // The near grid covers the inner square, the far grid the rest. The
    // near one thins to the far one's density towards its edge, and blades
    // widen to keep the cover.
    let rel = xz - frame.eye.xz;
    let cheb = max(abs(rel.x), abs(rel.y));
    var keep = 1.0;
    var widen = 3.0;
    if tier == 0u {
        if cheb > grass.inner {
            return grass_none();
        }
        keep = mix(1.0, 1.0 / 9.0, smoothstep(grass.inner * 0.5, grass.inner, cheb));
        widen = inverseSqrt(keep);
    } else if cheb <= grass.inner {
        return grass_none();
    }
    let dist = length(rel);
    let fade = 1.0 - smoothstep(grass.outer * 0.65, grass.outer, dist);
    if r2 > keep || fade <= 0.0 {
        return grass_none();
    }

    // Ground: height, slope and which layer is here.
    let hf = sample_field(xz);
    if hf.water > 0.5 {
        return grass_none();
    }
    let e = 0.6;
    let gx = sample_field(xz + vec2<f32>(e, 0.0)).ground;
    let gz = sample_field(xz + vec2<f32>(0.0, e)).ground;
    let ground_n = normalize(vec3<f32>(hf.ground - gx, e, hf.ground - gz));
    let m = ground_mix(xz, hf.ground, ground_n.y);
    let cover = m.w[0] + m.w[1] * 0.55;
    if r3 > cover * 1.15 {
        return grass_none();
    }
    // Meadow (heather) grows shorter and darker than open grass.
    let heathy = m.w[1] / max(m.w[0] + m.w[1], 1e-4);
    let tall = 0.7 + 0.6 * (value_noise(xz * 0.07, terrain.seed ^ 0x6161u) * 0.5 + 0.5);
    let h = grass.height * (0.55 + 0.9 * r0 * r1 + 0.25 * r3) * tall
        * mix(1.0, 0.55, heathy) * fade * clamp(cover * 1.4, 0.35, 1.0);
    if h < 0.025 {
        return grass_none();
    }

    // Blade frame: a random facing, turned partly towards the camera so
    // blades never vanish edge-on.
    let yaw = r1 * 6.2831853;
    var side = vec3<f32>(cos(yaw), 0.0, sin(yaw));
    let to_eye = normalize(vec3<f32>(-rel.x, 0.0, -rel.y) + vec3<f32>(1e-4, 0.0, 0.0));
    let facing_cam = vec3<f32>(-to_eye.z, 0.0, to_eye.x);
    side = normalize(mix(side, facing_cam * sign(dot(side, facing_cam) + 1e-4), 0.45));
    let fwd = vec3<f32>(-side.z, 0.0, side.x);

    // Vertex along the blade: pairs at t = 0, 1/3, 2/3, then the tip.
    let seg = min(vi / 2u, 3u);
    let t = f32(seg) / 3.0;
    var across = select(-0.5, 0.5, (vi & 1u) == 1u);
    if vi >= 6u {
        across = 0.0;
    }
    let width = (0.034 + 0.02 * r3) * widen * mix(1.0, 1.6, heathy) * sqrt(max(h / grass.height, 0.2));
    let taper = pow(max(1.0 - t, 0.0), 0.7);

    // Lean and wind: gusts roll across the field along the wind direction.
    let wind_dir = vec3<f32>(0.9216, 0.0, 0.3871);
    let wave = sin(dot(xz, wind_dir.xz) * 0.18 - grass.time * 1.7 + value_noise(xz * 0.05, terrain.seed) * 2.0);
    let flutter = sin(grass.time * 4.3 + r0 * 30.0 + dot(xz, vec2<f32>(0.9, 0.4)));
    let gust = grass.wind * (0.35 + 0.4 * (wave * 0.5 + 0.5)) + grass.wind * 0.06 * flutter;
    let lean = (0.15 + 0.25 * r2) * h;
    var bend = fwd * lean + wind_dir * gust * h * 0.75;
    // Keep the blade's length: a strongly bent blade gets lower.
    let bend_len = length(bend);
    let up_h = h * inverseSqrt(1.0 + (bend_len * bend_len) / max(h * h, 1e-6));
    let p = vec3<f32>(xz.x, hf.ground, xz.y)
        + side * (across * width * taper)
        + vec3<f32>(0.0, up_h * t, 0.0)
        + bend * (t * t);

    // Normal: the blade's face, bent towards the sky at the tip so a field
    // reads as one soft surface.
    let tangent = normalize(vec3<f32>(0.0, up_h, 0.0) + bend * (2.0 * t) + vec3<f32>(0.0, 1e-4, 0.0));
    let face = normalize(cross(side, tangent));
    let nrm = normalize(mix(face, ground_n, 0.7 + 0.2 * t));

    // Colour: the ground layer's own texture under the blade (a coarse mip),
    // varied per blade, lighter towards the tip.
    let layer = select(0, 1, heathy > 0.5);
    let uv = vec2<f32>(xz.x, -xz.y) * layer_tile(layer);
    var base = textureSampleLevel(layer_albedo, layer_sampler, uv, layer, 5.0).rgb;
    if layers.inv_tile_b.y < 0.5 {
        base = terrain.grass.rgb;
    }
    base *= 0.78 + 0.44 * r2;
    base *= 0.88 + 0.24 * (m.patches * 0.5 + 0.5);
    // Clumps of drier, yellower grass, and the odd straw-coloured blade.
    let dry = vec3<f32>(0.62, 0.56, 0.3) * dot(base, vec3<f32>(0.33)) * 2.4;
    let clump = smoothstep(0.55, 0.9, value_noise(xz * 0.11, terrain.seed ^ 0x9191u) * 0.5 + 0.5);
    base = mix(base, dry, clamp(0.45 * clump + 0.3 * step(0.93, r3), 0.0, 0.8));

    var out: GrassOut;
    out.clip = frame.view_proj * vec4<f32>(p, 1.0);
    out.world_p = p;
    out.n = nrm;
    out.color = base;
    out.t = t;
    return out;
}

@fragment
fn fs_grass(in: GrassOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let v = normalize(frame.eye - in.world_p);
    var n = normalize(in.n);
    if dot(n, v) < 0.0 {
        n = normalize(n - v * (2.0 * dot(n, v)));
    }
    let l = normalize(frame.light_dir);
    let ndl = max(dot(n, l), 0.0);
    let wrap = ndl * 0.6 + 0.4;
    let vis = sun_visibility(in.world_p, n, frame.eye);
    let sun_diffuse = wrap * wrap / max(1.0 - frame.ambient, 0.05);
    var light = frame_light();
    light.ao = screen_ao(frame.view_proj * vec4(in.world_p, 1.0), frame.sky_up.w)
        * mix(0.5, 1.0, in.t);
    light.gi = gi_irradiance(in.world_p, n);
    // Darker in the dense lower part of the sward.
    let albedo = in.color * mix(0.6, 1.35, in.t * in.t);
    var lit = pbr_shade(albedo, n, v, light, 0.75, 0.0, vis, sun_diffuse);
    // Sun shining through the blade when it is behind it.
    let through = pow(max(dot(-v, l), 0.0), 3.0) * vis * in.t;
    lit += albedo * frame.light_color * (1.0 - frame.ambient) * through * 0.45;
    lit += albedo * torch_light(in.world_p, n);
    // Lamps light the blades from either side.
    lit += lamp_light(albedo, n, v, in.world_p, 0.75, 0.0)
        + lamp_light(albedo, -n, v, in.world_p, 0.75, 0.0) * 0.6;
    return vec4<f32>(hdr_safe(lit), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_cover_the_distance_within_budget() {
        let grass = Grass::default();
        let (u, count) = GrassUniform::around(Vec3::new(10.0, 0.0, -5.0), &grass, 1.0, 0.0, 0.5);
        assert!(count > 10_000 && count <= MAX_BLADES, "{count}");
        assert!(u.inner < u.outer);
        // The far grid reaches the end of the grass.
        assert!(u.n1 as f32 * u.spacing * FAR_STEP >= 2.0 * u.outer);
        // Grids snap to their spacing: a small move keeps the origin.
        let (v, _) = GrassUniform::around(Vec3::new(10.01, 0.0, -5.0), &grass, 1.0, 0.0, 0.5);
        assert_eq!(u.origin1, v.origin1);
    }

    #[test]
    fn nothing_when_off_and_capped_when_huge() {
        let grass = Grass::default();
        assert_eq!(GrassUniform::around(Vec3::ZERO, &grass, 0.0, 0.0, 0.0).1, 0);
        let dense = Grass {
            density: 4.0,
            distance: 400.0,
            ..Grass::default()
        };
        let (_, count) = GrassUniform::around(Vec3::ZERO, &dense, 1.0, 0.0, 0.0);
        assert!(count <= MAX_BLADES);
    }
}
