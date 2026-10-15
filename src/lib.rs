//! Histogram gradient boosting with a Rust CPU implementation and optional Metal execution.
//!
//! The shipped library does not link the upstream C++ LightGBM engine.
#![deny(missing_docs)]

mod binning;
mod config;
mod dataset;
mod error;
mod histogram;
mod model;
mod objective;
mod training;

pub use config::{Backend, Objective, StageProfile, TrainingConfig, TrainingReport};
pub use dataset::DenseMatrix;
pub use error::{Error, Result};
pub use model::Model;
pub use training::{train, TrainingResult};

/// Check whether the requested execution backend is available.
///
/// Metal requires the `metal` feature and an Apple Silicon macOS device.
pub fn backend_available(backend: Backend) -> Result<()> {
    histogram::check_backend(backend)
}
