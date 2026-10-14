use crate::{Error, Objective, Result};

pub(crate) const EPSILON: f64 = 1e-15_f32 as f64;

pub(crate) fn sigmoid(score: f64) -> f64 {
    if score >= 0.0 {
        1.0 / (1.0 + (-score).exp())
    } else {
        let exp = score.exp();
        exp / (1.0 + exp)
    }
}

pub(crate) fn initial_score(labels: &[f64], objective: Objective) -> Result<f64> {
    let mean = labels.iter().sum::<f64>() / labels.len() as f64;
    let score = match objective {
        Objective::RegressionL2 => mean,
        Objective::BinaryLogloss => {
            let probability = mean.clamp(EPSILON, 1.0 - EPSILON);
            (probability / (1.0 - probability)).ln()
        }
    };
    if score.is_finite() {
        Ok(score)
    } else {
        Err(Error::Numerical("initial score is nonfinite".into()))
    }
}

pub(crate) fn derivatives(
    scores: &[f64],
    labels: &[f64],
    objective: Objective,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut gradients = Vec::with_capacity(labels.len());
    let mut hessians = Vec::with_capacity(labels.len());
    for (&score, &label) in scores.iter().zip(labels) {
        let (gradient, hessian) = match objective {
            Objective::RegressionL2 => (score - label, 1.0),
            Objective::BinaryLogloss => {
                // Match upstream's label-specific response without subtracting nearly equal values.
                let gradient = if label == 1.0 {
                    -sigmoid(-score)
                } else {
                    sigmoid(score)
                };
                let magnitude = gradient.abs();
                (gradient, magnitude * (1.0 - magnitude))
            }
        };
        let (gradient, hessian) = (gradient as f32, hessian as f32);
        if !score.is_finite() || !gradient.is_finite() || !hessian.is_finite() {
            return Err(Error::Numerical(
                "objective derivatives are nonfinite".into(),
            ));
        }
        gradients.push(gradient);
        hessians.push(hessian);
    }
    Ok((gradients, hessians))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_scores_and_derivatives_match_frozen_reference() {
        for name in ["regression", "binary", "single_class"] {
            let fixture: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(format!(
                    "{}/validation/fixtures/{name}.json",
                    env!("CARGO_MANIFEST_DIR")
                ))
                .unwrap(),
            )
            .unwrap();
            let labels: Vec<_> = fixture["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let objective = if name == "regression" {
                Objective::RegressionL2
            } else {
                Objective::BinaryLogloss
            };
            let initial = initial_score(&labels, objective).unwrap();
            assert!(
                (initial - fixture["initial_score"].as_f64().unwrap()).abs() < 1e-12,
                "{name}"
            );
            let (g, h) = derivatives(&vec![initial; labels.len()], &labels, objective).unwrap();
            for (actual, expected) in g.iter().zip(fixture["gradients"].as_array().unwrap()) {
                assert!(
                    (*actual as f64 - expected.as_f64().unwrap()).abs() <= 1e-7,
                    "{name}"
                );
            }
            for (actual, expected) in h.iter().zip(fixture["hessians"].as_array().unwrap()) {
                assert!(
                    (*actual as f64 - expected.as_f64().unwrap()).abs() <= 1e-7,
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn extreme_scores_are_stable_and_gradient_overflow_fails() {
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
        assert!(derivatives(&[0.0], &[f64::MAX], Objective::RegressionL2).is_err());
        let (g, _) = derivatives(&[40.0], &[1.0], Objective::BinaryLogloss).unwrap();
        assert!(g[0] < 0.0);
    }
}
