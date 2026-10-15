# lightgbm-metal-rs

An unofficial, community-maintained Rust implementation of a focused subset of [LightGBM](https://github.com/lightgbm-org/LightGBM) training, with optional Apple Silicon Metal acceleration.

This open-source project is maintained independently. It is not affiliated with or endorsed by the upstream LightGBM project. The training engine and prediction code are Rust; GPU histogram kernels use Metal Shading Language. The library has no runtime dependency on Python or the C++ LightGBM engine.

## Project status and supported features

The initial implementation supports CPU and Metal training, CPU prediction, and native model persistence. It implements the following subset rather than the complete upstream API:

| Area | Supported behavior |
| --- | --- |
| Objectives | L2 regression and binary logloss with labels exactly zero or one |
| Input | Dense row-major `f64` features and labels; NaNs represent missing features; zeros remain numeric |
| Trees | Standard histogram GBDT with leaf-wise growth, leaf/depth limits, minimum rows/Hessians/gain, and L1/L2 penalties |
| Binning | Training-only numeric boundaries, 2–255 bins, fixed `min_data_in_bin = 3`, no feature prefilter |
| Execution | Portable CPU histograms or explicitly selected Apple Silicon Metal histograms |
| Prediction | CPU raw scores, regression values, and stable binary probabilities |
| Persistence | Versioned native JSON with structural validation and exact same-build prediction round trips |

Sparse/categorical inputs, multiclass, ranking, DART/GOSS, sampling, weights, custom objectives, constraints, early stopping, distributed training, automatic backend routing, and GPU inference are not implemented. Upstream LightGBM model files cannot be loaded by the native JSON format. Identical trees across implementations are not guaranteed.

## Quick start

Use the source checkout:

```sh
git clone https://github.com/alfinsyahruddin/lightgbm-metal-rs.git
cd lightgbm-metal-rs
cargo run --locked --no-default-features --example train
```

The example trains an L2 model, predicts on CPU, saves `target/demo-model.json`, and verifies an exact save/load round trip.

Rust 1.74 is the declared minimum; validation currently uses Rust 1.97.1, and the minimum compiler has not yet been tested. CPU builds do not require Metal, Xcode, Python, or a C++ toolchain. Direct dependencies are pinned in [Cargo.toml](Cargo.toml), with transitive versions in [Cargo.lock](Cargo.lock).

To use the checkout from another Rust package, add a local dependency with the appropriate relative path:

```toml
[dependencies]
lightgbm-metal-rs = { path = "../lightgbm-metal-rs" }
```

```rust
use lightgbm_metal_rs::{train, DenseMatrix, Model, TrainingConfig};

fn main() -> lightgbm_metal_rs::Result<()> {
    let values = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let labels = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    let data = DenseMatrix::new(&values, 6, 1)?;
    let config = TrainingConfig {
        iterations: 10,
        min_data_in_leaf: 1,
        ..Default::default()
    };
    let trained = train(data, &labels, &config)?;
    println!("{:?}", trained.model.predict(data)?);
    trained.model.save("model.json")?;
    let loaded = Model::load("model.json")?;
    assert_eq!(trained.model.predict(data)?, loaded.predict(data)?);
    Ok(())
}
```

For binary classification, set `objective: Objective::BinaryLogloss` and use zero/one labels. `Model::predict` returns probabilities for binary models; `predict_raw` returns untransformed scores.

## Metal training

Metal training targets Apple Silicon macOS 11 or newer with a usable unified-memory GPU. Enable the optional `metal` feature and select the backend explicitly:

```sh
cargo run --locked --features metal --example train -- --metal
```

For a dependent package, add `features = ["metal"]` to its dependency declaration and set `backend: Backend::Metal` in `TrainingConfig`. An unavailable or failed Metal backend returns an error; there is no implicit CPU fallback. Histograms, split finding, row partitioning, leaf score updates, and loss derivatives execute residently on Metal GPU; tree topology orchestration and candidate threshold certification remain on CPU. Saved models can be loaded for prediction by a CPU-only build.

The default histogram workspace cap is 256 MiB. It covers live host histogram/cache arrays and GPU partial/output buffers. Dataset, model, score, derivative, row-membership, and framework storage are additional memory. See [Metal execution](docs/metal.md) for prerequisites, allocation limits, and hardware validation.

## Measured performance

The three benchmark baselines are:

- **Rust CPU:** this repository's trainer using CPU histograms.
- **Rust Metal:** this repository's trainer using resident GPU histograms, split scans, partitioning, and score updates.
- **Upstream CPU:** the pinned bundled LightGBM C++ trainer, invoked through its C API by the benchmark tooling.

On an Apple M4 Pro (20 GPU cores), the million-row, 64-feature workloads show substantial GPU speedups over single-threaded CPU baselines:
- **L2 regression, 63 bins:** **6.70 s** on Rust Metal vs **17.48 s** on Rust CPU (2.61× faster) and **10.78 s** on Upstream CPU (1.61× faster).
- **L2 regression, 255 bins:** **11.57 s** on Rust Metal vs **22.89 s** on Rust CPU (1.98× faster), matching Upstream CPU (**11.55 s**).
- **Binary logloss, 63 bins:** **7.48 s** on Rust Metal vs **20.29 s** on Rust CPU (2.71× faster) and **12.76 s** on Upstream CPU (1.71× faster).
- **Binary logloss, 255 bins:** **11.89 s** on Rust Metal vs **24.34 s** on Rust CPU (2.05× faster) and **12.34 s** on Upstream CPU (1.04× faster).

These are complete-training medians of five warm runs, including preprocessing, with one CPU thread. Small and medium workloads pay submission and synchronization overhead relative to arithmetic and are slower on Metal than CPU. Both faster and slower cases are retained; no universal GPU speedup is claimed.

See the [benchmark report](docs/benchmarks.md) for all 12 cases, cold timings, quality, memory interpretation, and reproduction commands.

## Development and validation

```sh
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --locked --no-default-features
cargo doc --locked --all-features --no-deps
```

On Apple Silicon with GPU access, run the complete portable and hardware suite:

```sh
python3 scripts/validate.py --require-metal
```

Hardware checks cannot be replaced by a CPU pass or a skipped GPU test. Some sandboxes restrict GPU discovery; the validator records skips explicitly and fails if required hardware is unavailable.

Frozen comparisons use the bundled `lightgbm-py/` source (version 4.7.0.99), pinned by source-tree identity. Python and C++ are reference tooling only. Reference rebuilding requires additional tools; benchmarks require NumPy. See [reference validation](validation/README.md) for dependency pins, fixture regeneration, and numerical/quality thresholds.

## Documentation

- [Native models and prediction](docs/native-model.md)
- [Metal execution and memory](docs/metal.md)
- [Complete-training benchmarks](docs/benchmarks.md)
- [Reference evidence](validation/README.md)
- [Guidance for coding agents](AGENTS.md)
- [Metal performance task plan](TASKS.md)

Build local API documentation with `cargo doc --locked --no-deps --open`.

## Contributing

Community contributions are welcome: bug reports, documentation, tests, performance investigations, and implementation improvements. Use this repository's [issues](https://github.com/alfinsyahruddin/lightgbm-metal-rs/issues) and [pull requests](https://github.com/alfinsyahruddin/lightgbm-metal-rs/pulls) for this project's work; use upstream's tracker for upstream LightGBM issues.

Discuss substantial API or algorithm changes in an issue before starting. Keep pull requests focused, explain the resulting behavior, update affected docs, and report the checks you actually ran. For backend or numerical changes, include relevant frozen-fixture and real-hardware evidence. Preserve reference data and acceptance thresholds; do not regenerate fixtures merely to hide a failure. Performance claims should include complete-training timings, quality, configuration, and slower cases.

## License and attribution

Licensed under the [MIT License](LICENSE). Copyright for this project's contributions is attributed to the lightgbm-metal-rs contributors. Translated algorithms retain the Microsoft Corporation and LightGBM developers' notices. The bundled upstream source retains its [original license](lightgbm-py/LICENSE); third-party components retain their own licenses. See the [official LightGBM repository](https://github.com/lightgbm-org/LightGBM) for the upstream project.
