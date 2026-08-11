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

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("path not allowed: {0}")]
    PathNotAllowed(String),

    #[error("unknown entity")]
    UnknownEntity,
}

pub type EngineResult<T> = Result<T, EngineError>;
