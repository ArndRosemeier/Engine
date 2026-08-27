use thiserror::Error;

/// User-facing errors. Prefer these over panics for bad input / resources.
#[derive(Debug, Error)]
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

    #[error("event loop creation failed: {0}")]
    EventLoopCreation(#[source] winit::error::EventLoopError),

    #[error("event loop failed: {0}")]
    EventLoopRun(#[source] winit::error::EventLoopError),

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

pub type EngineResult<T> = Result<T, EngineError>;
