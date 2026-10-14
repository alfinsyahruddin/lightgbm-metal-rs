/// Errors returned by input validation, training, prediction, and persistence.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid dataset shape or feature/label value.
    #[error("invalid data: {0}")]
    InvalidData(String),
    /// Invalid training configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The requested backend is unavailable; no CPU fallback was performed.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    /// GPU compilation, allocation, or execution failed.
    #[error("backend execution failed: {0}")]
    BackendExecution(String),
    /// A histogram allocation would exceed the configured workspace.
    #[error("insufficient histogram workspace: need {required} bytes, have {available}")]
    Workspace {
        /// Minimum bytes required for the operation.
        required: usize,
        /// Bytes allowed by the configured limit.
        available: usize,
    },
    /// A computation overflowed or produced a nonfinite statistic.
    #[error("numerical error: {0}")]
    Numerical(String),
    /// A native model is malformed or has an unsupported format version.
    #[error("invalid model: {0}")]
    InvalidModel(String),
    /// A prediction method does not match the model objective.
    #[error("prediction method does not match model objective")]
    ObjectiveMismatch,
    /// Native model encoding or decoding failed.
    #[error("model JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// A model file could not be read or written.
    #[error("model I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// A fallible library operation.
pub type Result<T> = std::result::Result<T, Error>;
