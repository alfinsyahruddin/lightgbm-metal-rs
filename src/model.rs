use crate::{objective::sigmoid, DenseMatrix, Error, Objective, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Node {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        default_left: bool,
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub fn predict(&self, matrix: DenseMatrix<'_>, row: usize) -> f64 {
        let mut index = 0;
        loop {
            match self.nodes[index] {
                Node::Leaf { value } => return value,
                Node::Split {
                    feature,
                    threshold,
                    default_left,
                    left,
                    right,
                } => {
                    let value = matrix.value(row, feature);
                    let go_left = if value.is_nan() {
                        default_left
                    } else {
                        value <= threshold
                    };
                    index = if go_left { left } else { right };
                }
            }
        }
    }

    fn validate(&self, features: usize) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(Error::InvalidModel("tree has no root".into()));
        }
        let mut visited = vec![false; self.nodes.len()];
        let mut pending = vec![0];
        while let Some(index) = pending.pop() {
            if index >= self.nodes.len() || visited[index] {
                return Err(Error::InvalidModel(
                    "tree contains an invalid child, cycle, or shared node".into(),
                ));
            }
            visited[index] = true;
            match self.nodes[index] {
                Node::Leaf { value } => {
                    if !value.is_finite() {
                        return Err(Error::InvalidModel("nonfinite leaf output".into()));
                    }
                }
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    ..
                } => {
                    if feature >= features || !threshold.is_finite() {
                        return Err(Error::InvalidModel(
                            "invalid split feature or threshold".into(),
                        ));
                    }
                    pending.extend([right, left]);
                }
            }
        }
        if visited.iter().any(|&seen| !seen) {
            return Err(Error::InvalidModel(
                "tree contains unreachable nodes".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    format_version: u32,
    objective: Objective,
    feature_count: usize,
    initial_score: f64,
    trees: Vec<Tree>,
}

/// Validated, backend-independent model using the native versioned JSON format.
///
/// Upstream LightGBM text/binary model files are not accepted by this format.
#[derive(Debug, Clone)]
pub struct Model {
    data: Envelope,
}

impl Model {
    pub(crate) fn new(
        objective: Objective,
        feature_count: usize,
        initial_score: f64,
        trees: Vec<Tree>,
    ) -> Result<Self> {
        Self::validated(Envelope {
            format_version: 1,
            objective,
            feature_count,
            initial_score,
            trees,
        })
    }

    fn validated(data: Envelope) -> Result<Self> {
        if data.format_version != 1 {
            return Err(Error::InvalidModel(
                "unsupported native format version".into(),
            ));
        }
        if data.feature_count == 0
            || data.feature_count > u32::MAX as usize
            || !data.initial_score.is_finite()
        {
            return Err(Error::InvalidModel(
                "invalid feature count or initial score".into(),
            ));
        }
        for tree in &data.trees {
            tree.validate(data.feature_count)?;
        }
        Ok(Self { data })
    }

    /// Objective used during training.
    pub fn objective(&self) -> Objective {
        self.data.objective
    }

    /// Number of features required for prediction.
    pub fn feature_count(&self) -> usize {
        self.data.feature_count
    }

    /// Number of fitted trees (zero for a constant model).
    pub fn tree_count(&self) -> usize {
        self.data.trees.len()
    }

    /// Predict untransformed scores using CPU traversal and learned missing branches.
    pub fn predict_raw(&self, matrix: DenseMatrix<'_>) -> Result<Vec<f64>> {
        if matrix.cols() != self.data.feature_count {
            return Err(Error::InvalidData(
                "prediction feature count does not match model".into(),
            ));
        }
        let mut predictions = Vec::with_capacity(matrix.rows());
        for row in 0..matrix.rows() {
            let mut score = self.data.initial_score;
            for tree in &self.data.trees {
                score += tree.predict(matrix, row);
            }
            if !score.is_finite() {
                return Err(Error::Numerical("prediction score overflowed".into()));
            }
            predictions.push(score);
        }
        Ok(predictions)
    }

    /// Predict regression values, or binary probabilities according to the model objective.
    pub fn predict(&self, matrix: DenseMatrix<'_>) -> Result<Vec<f64>> {
        match self.data.objective {
            Objective::RegressionL2 => self.predict_regression(matrix),
            Objective::BinaryLogloss => self.predict_proba(matrix),
        }
    }

    /// Predict regression values; fails on a binary model.
    pub fn predict_regression(&self, matrix: DenseMatrix<'_>) -> Result<Vec<f64>> {
        if self.data.objective != Objective::RegressionL2 {
            return Err(Error::ObjectiveMismatch);
        }
        self.predict_raw(matrix)
    }

    /// Predict finite binary probabilities with a stable sigmoid; fails on a regression model.
    pub fn predict_proba(&self, matrix: DenseMatrix<'_>) -> Result<Vec<f64>> {
        if self.data.objective != Objective::BinaryLogloss {
            return Err(Error::ObjectiveMismatch);
        }
        Ok(self.predict_raw(matrix)?.into_iter().map(sigmoid).collect())
    }

    /// Serialize the native versioned format with round-trip-safe floating-point values.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.data)?)
    }

    /// Decode and validate the native format, rejecting invalid structures before prediction.
    pub fn from_json(json: &str) -> Result<Self> {
        Self::validated(serde_json::from_str(json)?)
    }

    /// Save a native JSON model to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_json()?)?;
        Ok(())
    }

    /// Load and validate a native JSON model file without initializing Metal.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Model {
        Model::new(
            Objective::RegressionL2,
            1,
            0.1,
            vec![Tree {
                nodes: vec![
                    Node::Split {
                        feature: 0,
                        threshold: 0.5,
                        default_left: false,
                        left: 1,
                        right: 2,
                    },
                    Node::Leaf {
                        value: -1.2345678901234567,
                    },
                    Node::Leaf { value: 2.0 },
                ],
            }],
        )
        .unwrap()
    }

    #[test]
    fn raw_predictions_missing_routing_and_roundtrip_are_exact() {
        let model = example();
        let values = [0.0, 0.5, 1.0, f64::NAN];
        let matrix = DenseMatrix::new(&values, 4, 1).unwrap();
        let predictions = model.predict_raw(matrix).unwrap();
        assert_eq!(
            predictions,
            vec![0.1 - 1.2345678901234567, 0.1 - 1.2345678901234567, 2.1, 2.1]
        );
        assert_eq!(
            Model::from_json(&model.to_json().unwrap())
                .unwrap()
                .predict_raw(matrix)
                .unwrap(),
            predictions
        );
        assert!(model.predict_proba(matrix).is_err());
        assert_eq!(
            model.predict(DenseMatrix::new(&[], 0, 1).unwrap()).unwrap(),
            Vec::<f64>::new()
        );
        assert!(model
            .predict(DenseMatrix::new(&[0.0, 1.0], 1, 2).unwrap())
            .is_err());
    }

    #[test]
    fn malformed_models_are_rejected_before_prediction() {
        let original: serde_json::Value =
            serde_json::from_str(&example().to_json().unwrap()).unwrap();
        for (path, value) in [
            ("/format_version", serde_json::json!(2)),
            ("/trees/0/nodes/0/left", serde_json::json!(0)),
            ("/trees/0/nodes/0/right", serde_json::json!(99)),
            ("/trees/0/nodes/0/feature", serde_json::json!(1)),
        ] {
            let mut invalid = original.clone();
            *invalid.pointer_mut(path).unwrap() = value;
            assert!(Model::from_json(&invalid.to_string()).is_err(), "{path}");
        }
        let mut invalid = original;
        invalid["trees"][0]["nodes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"type":"leaf","value":1.0}));
        assert!(Model::from_json(&invalid.to_string()).is_err());
        assert!(Model::new(
            Objective::RegressionL2,
            1,
            0.0,
            vec![Tree {
                nodes: vec![Node::Leaf { value: f64::NAN }]
            }]
        )
        .is_err());
    }

    #[test]
    fn binary_probabilities_are_finite_for_extreme_scores() {
        for (score, probability) in [(-1000.0, 0.0), (0.0, 0.5), (1000.0, 1.0)] {
            let model = Model::new(Objective::BinaryLogloss, 1, score, vec![]).unwrap();
            assert_eq!(
                model
                    .predict_proba(DenseMatrix::new(&[0.0], 1, 1).unwrap())
                    .unwrap(),
                vec![probability]
            );
        }
    }
}
