//! Custom materials: a game's own WGSL function that decides a surface's
//! colour, normal, roughness, metal and glow, run inside the engine's lit
//! mesh shader (so it gets sun, shadows, lamps, fog and AO like any mesh).
//!
//! ```ignore
//! let lava = world.add_material(CustomMaterial::new("lava", r#"
//!     fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut {
//!         var o = base;
//!         let flow = sin(m.world_pos.x * 2.0 + m.time) * 0.5 + 0.5;
//!         o.color = vec4<f32>(0.1, 0.02, 0.0, 1.0);
//!         o.emission = mix(vec3<f32>(1.0, 0.2, 0.0), vec3<f32>(1.0, 0.8, 0.1), flow) * m.params[0].x;
//!         return o;
//!     }
//! "#).param(0, [2.0, 0.0, 0.0, 0.0]))?;
//! world.set_custom_material(rock, lava)?;
//! world.set_material_param(lava, 0, [heat, 0.0, 0.0, 0.0])?; // any frame
//! ```

use crate::error::{EngineError, EngineResult};

/// What the material function gets, and gives back, in WGSL:
///
/// ```wgsl
/// struct MaterialIn {
///     world_pos: vec3<f32>,   // where this pixel is, in the world (m)
///     object_pos: vec3<f32>,  // the same, relative to the entity (moves with it)
///     normal: vec3<f32>,      // the surface's normal, world space, unit length
///     view_dir: vec3<f32>,    // from the surface to the camera, unit length
///     uv: vec2<f32>,          // the mesh's texture coordinates
///     time: f32,              // seconds since start, for animation
///     params: array<vec4<f32>, 4>, // the material's numbers (set_material_param)
/// };
/// struct MaterialOut {
///     color: vec4<f32>,       // base colour (linear) and alpha
///     normal: vec3<f32>,      // the normal to light with (re-normalised)
///     roughness: f32,         // 0 mirror .. 1 matte
///     metallic: f32,          // 0 .. 1
///     emission: vec3<f32>,    // light of its own, added after lighting
/// };
/// ```
///
/// `base` arrives filled with what the mesh would look like without the
/// material (its vertex colour and textures), so a material can change only
/// what it needs.
pub const MATERIAL_API: &str = "fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut";

/// A material made of WGSL code. Add it with `World::add_material`.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomMaterial {
    /// Shown in errors.
    pub name: String,
    /// WGSL defining `fn material(m: MaterialIn, base: MaterialOut) -> MaterialOut`
    /// (and any helper functions or constants it needs).
    pub code: String,
    /// Drawn blended over what is behind it (with the far-to-near sort),
    /// using the alpha of `color`. Off: solid.
    pub see_through: bool,
    /// Four numbers of four each, read as `m.params[0..4]`.
    pub params: [[f32; 4]; 4],
}

impl CustomMaterial {
    pub fn new(name: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            code: code.into(),
            see_through: false,
            params: [[0.0; 4]; 4],
        }
    }

    /// Blend it over what is behind it (force fields, holograms, ghosts).
    pub fn see_through(mut self) -> Self {
        self.see_through = true;
        self
    }

    /// Start value of `m.params[index]` (index 0 to 3).
    pub fn param(mut self, index: usize, value: [f32; 4]) -> Self {
        if let Some(p) = self.params.get_mut(index) {
            *p = value;
        }
        self
    }

    /// Checks the code compiles as part of the engine's shader. Errors give
    /// the line in `code` and what is wrong.
    pub fn check(&self) -> EngineResult<()> {
        check_code(&self.name, &self.code)
    }
}

/// A material made with `World::add_material`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CustomMaterialId(pub(crate) u32);

fn shader_error(name: &str, line: Option<usize>, message: &str, detail: String) -> EngineError {
    let at = line.map_or(String::new(), |l| format!(" line {l}"));
    EngineError::InvalidValue(format!(
        "custom material {name:?}{at}: {message}\n{detail}\nThe code must define `{MATERIAL_API}` (see engine::custom_material)."
    ))
}

pub(crate) fn check_code(name: &str, code: &str) -> EngineResult<()> {
    if !code.contains("fn material") {
        return Err(shader_error(
            name,
            None,
            "no `fn material` in the code",
            String::new(),
        ));
    }
    let source = crate::render::lit_shader_source(Some(code));
    let mark = crate::render::CUSTOM_CODE_MARK;
    let first = source.lines().position(|l| l == mark).map_or(0, |i| i + 2);
    let code_lines = code.lines().count();
    // A line in the game's code, or None when the problem is elsewhere.
    let in_code = |line: u32| {
        let line = line as usize;
        (line >= first && line < first + code_lines).then(|| line - first + 1)
    };
    // Show the game's own line, or naga's report when it is elsewhere.
    let detail = |line: Option<usize>, report: String| match line {
        Some(l) => format!(
            "  {l} | {}",
            code.lines().nth(l - 1).unwrap_or("").trim_end()
        ),
        None => report.lines().take(24).collect::<Vec<_>>().join("\n"),
    };
    let module = naga::front::wgsl::parse_str(&source).map_err(|e| {
        let line = e.location(&source).and_then(|l| in_code(l.line_number));
        shader_error(
            name,
            line,
            e.message(),
            detail(line, e.emit_to_string(&source)),
        )
    })?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|e| {
        // The narrowest span inside the game's code says where; the error
        // chain ("function is invalid" -> "expression is invalid" -> why)
        // says what.
        let line = e
            .spans()
            .filter_map(|(span, _)| in_code(span.location(&source).line_number))
            .last()
            .or_else(|| e.location(&source).and_then(|l| in_code(l.line_number)));
        let mut message = e.as_inner().to_string();
        let mut cause = std::error::Error::source(e.as_inner());
        while let Some(c) = cause {
            message.push_str(": ");
            message.push_str(&c.to_string());
            cause = c.source();
        }
        let report = e.emit_to_string(&source);
        shader_error(name, line, &message, detail(line, report))
    })?;
    Ok(())
}
