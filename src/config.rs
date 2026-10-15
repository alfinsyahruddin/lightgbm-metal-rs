use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Supported training and prediction objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Squared-error regression.
    RegressionL2,
    /// Binary logloss with labels exactly zero or one and sigmoid scale one.
    BinaryLogloss,
}

/// Explicit histogram execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Portable Rust CPU histograms.
    Cpu,
    /// Apple Silicon Metal histograms; tree orchestration remains on CPU.
    Metal,
}

/// Configuration for the supported GBDT subset. Defaults use full data and features.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// Training objective (default L2 regression).
    pub objective: Objective,
    /// Histogram backend (default CPU).
    pub backend: Backend,
    /// Maximum boosting iterations (default 100; must be positive).
    pub iterations: usize,
    /// Tree shrinkage (default 0.1; finite and positive).
    pub learning_rate: f64,
    /// Maximum leaves per tree (default 31; at least two).
    pub num_leaves: usize,
    /// Optional maximum depth; root depth is zero (default unlimited).
    pub max_depth: Option<usize>,
    /// Maximum feature bins including missing values (default 255; 2 through 255).
    pub max_bin: usize,
    /// Minimum rows in an eligible child (default 20; positive).
    pub min_data_in_leaf: usize,
    /// Minimum Hessian sum in an eligible child (default 0.001).
    pub min_sum_hessian_in_leaf: f64,
    /// Minimum gain required to split a leaf (default zero).
    pub min_gain_to_split: f64,
    /// L1 penalty on leaf outputs (default zero).
    pub lambda_l1: f64,
    /// L2 penalty on leaf outputs (default zero).
    pub lambda_l2: f64,
    /// Limit for all live histogram/cache/reduction buffers; excludes dataset/model storage.
    pub histogram_workspace_bytes: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            objective: Objective::RegressionL2,
            backend: Backend::Cpu,
            iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: None,
            max_bin: 255,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 0.001,
            min_gain_to_split: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            histogram_workspace_bytes: 256 * 1024 * 1024,
        }
    }
}

impl TrainingConfig {
    /// Validate scalar domains without initializing an execution backend.
    pub fn validate(&self) -> Result<()> {
        for (invalid, name) in [
            (self.iterations == 0, "iterations must be positive"),
            (self.num_leaves < 2, "num_leaves must be at least two"),
            (self.max_depth == Some(0), "max_depth must be positive"),
            (
                !(2..=255).contains(&self.max_bin),
                "max_bin must be 2 through 255",
            ),
            (
                self.min_data_in_leaf == 0,
                "min_data_in_leaf must be positive",
            ),
            (
                self.histogram_workspace_bytes == 0,
                "histogram workspace must be positive",
            ),
        ] {
            if invalid {
                return Err(Error::InvalidConfig(name.into()));
            }
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(Error::InvalidConfig(
                "learning_rate must be finite and positive".into(),
            ));
        }
        for (name, value) in [
            ("min_sum_hessian_in_leaf", self.min_sum_hessian_in_leaf),
            ("min_gain_to_split", self.min_gain_to_split),
            ("lambda_l1", self.lambda_l1),
            ("lambda_l2", self.lambda_l2),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "{name} must be finite and nonnegative"
                )));
            }
        }
        Ok(())
    }
}

/// Observable execution and timing information for a completed training call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    /// Backend actually selected; Metal errors never switch this to CPU.
    pub backend: Backend,
    /// Successfully completed GPU histogram kernel dispatches.
    pub gpu_histogram_dispatches: u64,
    /// Peak tracked histogram/cache/reduction allocation bytes.
    pub peak_histogram_workspace_bytes: usize,
    /// Number of trees stored in the resulting model.
    pub trees: usize,
    /// Time spent validating and preparing bins, in seconds.
    pub preprocessing_seconds: f64,
    /// Time spent initializing the backend, in seconds.
    pub backend_setup_seconds: f64,
    /// Time spent fitting trees, in seconds.
    pub training_seconds: f64,
    /// Elapsed histogram construction time, including GPU waits and host copies.
    pub histogram_wall_seconds: f64,
    /// GPU command execution time for partial and reduction kernels; zero for CPU.
    pub histogram_gpu_seconds: f64,
    /// Number of directly constructed histograms; excludes derived siblings.
    pub histogram_calls: u64,
    /// Direct histogram calls for leaves with at most 4,096 rows.
    pub small_leaf_histogram_calls: u64,
    /// Detailed stage-by-stage timing breakdown for profiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_profile: Option<StageProfile>,
}

/// Stage-by-stage execution profiling breakdown for training.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct StageProfile {
    /// Time spent computing loss derivatives, in seconds.
    pub derivatives_seconds: f64,
    /// Time spent computing leaf totals, in seconds.
    pub leaf_totals_seconds: f64,
    /// Time spent searching for best splits, in seconds.
    pub split_search_seconds: f64,
    /// Time spent partitioning rows into child leaves, in seconds.
    pub partitioning_seconds: f64,
    /// Time spent updating training scores, in seconds.
    pub score_update_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_supported_reference_subset() {
        let c = TrainingConfig::default();
        assert!(c.validate().is_ok());
        assert_eq!((c.iterations, c.num_leaves, c.max_bin), (100, 31, 255));
    }

    #[test]
    fn invalid_numeric_domains_fail() {
        for c in [
            TrainingConfig {
                iterations: 0,
                ..Default::default()
            },
            TrainingConfig {
                max_bin: 256,
                ..Default::default()
            },
            TrainingConfig {
                lambda_l1: -1.0,
                ..Default::default()
            },
            TrainingConfig {
                learning_rate: f64::NAN,
                ..Default::default()
            },
            TrainingConfig {
                max_depth: Some(0),
                ..Default::default()
            },
        ] {
            assert!(matches!(c.validate(), Err(Error::InvalidConfig(_))));
        }
    }

    #[test]
    fn json_preserves_difficult_f64_values_exactly() {
        for value in [
            0.1,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::from_bits(1),
            -0.0,
            1.2345678901234567,
        ] {
            let encoded = serde_json::to_string(&value).unwrap();
            let decoded: f64 = serde_json::from_str(&encoded).unwrap();
            assert_eq!(value.to_bits(), decoded.to_bits());
        }
    }
}
