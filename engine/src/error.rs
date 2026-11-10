use thiserror::Error;

/// User-facing errors. Prefer these over panics for bad input / resources.
///
/// `Debug` prints the same readable message as `Display` (plus causes), so
/// `fn main() -> EngineResult<()>` ends with `Error: GPU: ...` rather than a
/// struct dump.
#[derive(Error)]
pub enum EngineError {
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),

    #[error("invalid color: {0}")]
    InvalidColor(String),

    #[error("invalid value: {0}")]
    InvalidValue(String),

    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("audio: {0}")]
    Audio(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("path not allowed: {0}")]
    PathNotAllowed(String),

    #[error("unknown entity")]
    UnknownEntity,

    #[error("unknown texture")]
    UnknownTexture,

    #[error("unknown material")]
    UnknownMaterial,

    #[error("application callback failed: {source}")]
    Application {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// No usable GPU, or it lacks something the engine needs. The message
    /// says what and what to try.
    #[error("GPU: {0}")]
    Gpu(String),

    /// The window or its drawing surface could not be made.
    #[error("window: {0}")]
    Window(String),

    #[error("event loop creation failed: {0}")]
    EventLoopCreation(#[source] winit::error::EventLoopError),

    #[error("event loop failed: {0}")]
    EventLoopRun(#[source] winit::error::EventLoopError),

    /// A verify run (`ENGINE_VERIFY`) found problems; see its `report.json`.
    #[error("verify run failed: {0}")]
    Verify(String),

    #[error("failed to save screenshot {path}: {source}")]
    ScreenshotSave {
        path: std::path::PathBuf,
        #[source]
        source: image::ImageError,
    },
}

impl EngineError {
    pub fn application(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Application {
            source: Box::new(error),
        }
    }
}

impl std::fmt::Debug for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")?;
        let mut source = std::error::Error::source(self);
        while let Some(cause) = source {
            write!(f, "\n  caused by: {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

pub type EngineResult<T> = Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_reads_like_the_message_with_its_causes() {
        let plain = EngineError::Gpu("no GPU".into());
        assert_eq!(format!("{plain:?}"), "GPU: no GPU");
        let io = EngineError::Io(std::io::Error::other("disk full"));
        assert_eq!(
            format!("{io:?}"),
            "I/O error: disk full\n  caused by: disk full"
        );
    }
}
