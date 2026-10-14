# Native model and prediction

`Model::predict_raw` returns the initial score plus the sum of tree leaf outputs. `predict_regression` returns that score for L2 models. `predict_proba` uses a stable sigmoid for binary models; `predict` chooses the appropriate method automatically. Objective-incompatible methods return `ObjectiveMismatch`.

Inputs are validated row-major `DenseMatrix` values with the trained feature count. NaNs follow the learned default branch. If a feature had no training NaNs, its default branch matches the reference's treatment of an unseen NaN as zero. Infinite features and inconsistent shapes fail. Correctly shaped empty prediction batches produce empty outputs.

```sh
cargo run --locked --no-default-features --example train
```

The example trains, saves to `target/demo-model.json`, loads, and checks exact prediction equality. This JSON format is independent of upstream LightGBM text and binary files. Version 1 contains `format_version`, `objective`, `feature_count`, `initial_score`, and flat `trees[].nodes` arrays. Splits contain a feature index, finite threshold, missing branch, and child indices. Leaves store final outputs with learning-rate shrinkage already applied.

The decoder checks version, feature count, finite values, valid indices, reachability, and acyclic tree structure before exposing a usable model. All inference is CPU Rust and has no dependency on the training backend. Full-precision floats round-trip exactly under the pinned JSON configuration. Nonfinite raw-score accumulation returns a numerical error.
