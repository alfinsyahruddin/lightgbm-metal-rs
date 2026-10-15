use lightgbm_metal_rs::{train, DenseMatrix, Model, Objective, TrainingConfig};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(format!(
            "{}/validation/fixtures/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn config(fixture: &Value) -> TrainingConfig {
    let mut base = serde_json::to_value(TrainingConfig::default()).unwrap();
    for (name, value) in fixture["config"].as_object().unwrap() {
        base[name] = value.clone();
    }
    serde_json::from_value(base).unwrap()
}

#[test]
fn small_cpu_models_match_frozen_upstream_raw_predictions() {
    for name in [
        "regression",
        "binary",
        "regularized",
        "depth",
        "leaf_limit",
        "min_rows",
        "min_hessian",
        "min_gain",
        "two_bins",
        "constant",
        "all_missing",
        "single_class",
        "unseen_nan",
    ] {
        let fixture = fixture(name);
        let x = numbers(&fixture["features"]);
        let labels = numbers(&fixture["labels"]);
        let predict_x = numbers(&fixture["prediction_features"]);
        let cols = fixture["cols"].as_u64().unwrap() as usize;
        let trained = train(
            DenseMatrix::new(&x, labels.len(), cols).unwrap(),
            &labels,
            &config(&fixture),
        )
        .unwrap();
        let matrix = DenseMatrix::new(&predict_x, predict_x.len() / cols, cols).unwrap();
        let actual = trained.model.predict_raw(matrix).unwrap();
        let expected = numbers(&fixture["raw_predictions"]);
        for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-8 + 1e-6 * expected.abs(),
                "{name} row {row}: {actual} vs {expected}"
            );
        }
        let repeated = train(
            DenseMatrix::new(&x, labels.len(), cols).unwrap(),
            &labels,
            &config(&fixture),
        )
        .unwrap();
        assert_eq!(
            trained.model.to_json().unwrap(),
            repeated.model.to_json().unwrap(),
            "{name}"
        );
        assert_eq!(
            Model::from_json(&trained.model.to_json().unwrap())
                .unwrap()
                .predict_raw(matrix)
                .unwrap(),
            actual
        );
    }
}

#[test]
fn cpu_quality_meets_frozen_regression_and_binary_acceptance() {
    for name in ["quality_regression_l2", "quality_binary_logloss"] {
        let fixture = fixture(name);
        let x = numbers(&fixture["features"]);
        let labels = numbers(&fixture["labels"]);
        let rows = fixture["train_rows"].as_u64().unwrap() as usize;
        let cols = fixture["cols"].as_u64().unwrap() as usize;
        let config = config(&fixture);
        let trained = train(
            DenseMatrix::new(&x[..rows * cols], rows, cols).unwrap(),
            &labels[..rows],
            &config,
        )
        .unwrap();
        let raw = trained
            .model
            .predict_raw(DenseMatrix::new(&x[rows * cols..], labels.len() - rows, cols).unwrap())
            .unwrap();
        match config.objective {
            Objective::RegressionL2 => {
                let rmse = (raw
                    .iter()
                    .zip(&labels[rows..])
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f64>()
                    / raw.len() as f64)
                    .sqrt();
                let baseline = fixture["quality"]["rmse"].as_f64().unwrap();
                eprintln!("CPU RMSE {rmse}, upstream {baseline}");
                assert!(rmse <= 1.02 * baseline + 1e-6);
            }
            Objective::BinaryLogloss => {
                let logloss = raw
                    .iter()
                    .zip(&labels[rows..])
                    .map(|(p, y)| p.max(0.0) - p * y + (-p.abs()).exp().ln_1p())
                    .sum::<f64>()
                    / raw.len() as f64;
                let baseline = fixture["quality"]["logloss"].as_f64().unwrap();
                eprintln!("CPU logloss {logloss}, upstream {baseline}");
                assert!(logloss <= baseline + 0.01);
            }
        }
    }
}

#[test]
fn workspace_failure_is_explicit_before_oversized_histogram_allocation() {
    let x: Vec<_> = (0..60).map(|v| v as f64).collect();
    let config = TrainingConfig {
        histogram_workspace_bytes: 1,
        min_data_in_leaf: 1,
        ..Default::default()
    };
    assert!(matches!(
        train(DenseMatrix::new(&x, 60, 1).unwrap(), &x, &config),
        Err(lightgbm_metal_rs::Error::Workspace { .. })
    ));
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
fn metal_workspace_failure_is_explicit_before_oversized_histogram_allocation() {
    let x: Vec<_> = (0..60).map(|v| v as f64).collect();
    let config = TrainingConfig {
        backend: lightgbm_metal_rs::Backend::Metal,
        histogram_workspace_bytes: 1,
        min_data_in_leaf: 1,
        ..Default::default()
    };
    assert!(matches!(
        train(DenseMatrix::new(&x, 60, 1).unwrap(), &x, &config),
        Err(lightgbm_metal_rs::Error::Workspace { .. })
    ));
}

#[test]
fn bounded_cache_eviction_matches_unbounded_cpu_training() {
    let fixture = fixture("regression");
    let x = numbers(&fixture["features"]);
    let labels = numbers(&fixture["labels"]);
    let matrix = DenseMatrix::new(&x, labels.len(), 1).unwrap();
    let mut config = config(&fixture);
    let regular = train(matrix, &labels, &config).unwrap();
    config.histogram_workspace_bytes = config.max_bin * 24;
    let bounded = train(matrix, &labels, &config).unwrap();
    assert_eq!(
        regular.model.to_json().unwrap(),
        bounded.model.to_json().unwrap()
    );
    assert!(bounded.report.peak_histogram_workspace_bytes <= config.histogram_workspace_bytes);
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
fn metal_small_models_and_bounded_cache_match_reference() {
    for name in [
        "regression",
        "binary",
        "regularized",
        "depth",
        "leaf_limit",
        "min_rows",
        "min_hessian",
        "min_gain",
        "two_bins",
        "constant",
        "all_missing",
        "single_class",
        "unseen_nan",
    ] {
        let fixture = fixture(name);
        let x = numbers(&fixture["features"]);
        let labels = numbers(&fixture["labels"]);
        let predict_x = numbers(&fixture["prediction_features"]);
        let cols = fixture["cols"].as_u64().unwrap() as usize;
        let mut config = config(&fixture);
        config.backend = lightgbm_metal_rs::Backend::Metal;
        let matrix = DenseMatrix::new(&x, labels.len(), cols).unwrap();
        let prediction_matrix = DenseMatrix::new(&predict_x, predict_x.len() / cols, cols).unwrap();
        for budget in [256 * 1024 * 1024, config.max_bin * cols * 24 * 2] {
            config.histogram_workspace_bytes = budget;
            let trained = train(matrix, &labels, &config).unwrap();
            let actual = trained.model.predict_raw(prediction_matrix).unwrap();
            for (row, (&actual, expected)) in actual
                .iter()
                .zip(numbers(&fixture["raw_predictions"]))
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-5 + 1e-4 * expected.abs(),
                    "{name} row {row}: {actual} vs {expected}"
                );
            }
            assert!(trained.report.peak_histogram_workspace_bytes <= budget);
            if trained.model.tree_count() > 0 {
                assert!(trained.report.gpu_histogram_dispatches > 0);
            }
        }
    }
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[ignore = "requires real Apple Silicon Metal access; run the hardware validation command"]
fn metal_quality_and_portable_model_evidence() {
    for name in ["quality_regression_l2", "quality_binary_logloss"] {
        let fixture = fixture(name);
        let x = numbers(&fixture["features"]);
        let labels = numbers(&fixture["labels"]);
        let rows = fixture["train_rows"].as_u64().unwrap() as usize;
        let cols = fixture["cols"].as_u64().unwrap() as usize;
        let mut config = config(&fixture);
        let matrix = DenseMatrix::new(&x[..rows * cols], rows, cols).unwrap();
        let validation = DenseMatrix::new(&x[rows * cols..], labels.len() - rows, cols).unwrap();
        let cpu = train(matrix, &labels[..rows], &config)
            .unwrap()
            .model
            .predict_raw(validation)
            .unwrap();
        config.backend = lightgbm_metal_rs::Backend::Metal;
        let trained = train(matrix, &labels[..rows], &config).unwrap();
        let raw = trained.model.predict_raw(validation).unwrap();
        let metric = |values: &[f64]| match config.objective {
            Objective::RegressionL2 => (values
                .iter()
                .zip(&labels[rows..])
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f64>()
                / values.len() as f64)
                .sqrt(),
            Objective::BinaryLogloss => {
                values
                    .iter()
                    .zip(&labels[rows..])
                    .map(|(p, y)| p.max(0.0) - p * y + (-p.abs()).exp().ln_1p())
                    .sum::<f64>()
                    / values.len() as f64
            }
        };
        let actual = metric(&raw);
        let cpu_quality = metric(&cpu);
        let baseline = fixture["quality"][if config.objective == Objective::RegressionL2 {
            "rmse"
        } else {
            "logloss"
        }]
        .as_f64()
        .unwrap();
        let limit = |reference: f64| {
            if config.objective == Objective::RegressionL2 {
                reference * 1.02 + 1e-6
            } else {
                reference + 0.01
            }
        };
        assert!(
            actual <= limit(baseline) && actual <= limit(cpu_quality),
            "{name}: Metal {actual}, CPU {cpu_quality}, upstream {baseline}"
        );
        println!("{name}: Metal {actual}, CPU {cpu_quality}, upstream {baseline}, dispatches {}, peak workspace {}", trained.report.gpu_histogram_dispatches, trained.report.peak_histogram_workspace_bytes);
        assert_eq!(
            Model::from_json(&trained.model.to_json().unwrap())
                .unwrap()
                .predict_raw(validation)
                .unwrap(),
            raw
        );
        if config.objective == Objective::RegressionL2 {
            trained.model.save("target/metal-model.json").unwrap();
            std::fs::write(
                "target/metal-predictions.json",
                serde_json::to_string(&raw).unwrap(),
            )
            .unwrap();
        }
    }
}
