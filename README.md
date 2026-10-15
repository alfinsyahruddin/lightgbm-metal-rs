# lightgbm-metal-rs

[![Crates.io](https://img.shields.io/crates/v/lightgbm-metal-rs.svg)](https://crates.io/crates/lightgbm-metal-rs) [![docs.rs](https://docs.rs/lightgbm-metal-rs/badge.svg)](https://docs.rs/lightgbm-metal-rs) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An unofficial, community-maintained Rust implementation of histogram gradient boosted decision trees (GBDT) focused on a core LightGBM training subset, with optional Apple Silicon Metal GPU acceleration.

> [!NOTE]
> `lightgbm-metal-rs` is an independent open-source project and is not affiliated with upstream LightGBM. The training and prediction engines are written in Rust; GPU histogram kernels use Metal Shading Language.

---

## Key Features

- 🦀 **Pure Rust Core**: Zero external C++ or Python runtime dependencies. Builds with standard `cargo`.
- ⚡ **Apple Silicon Metal Acceleration**: Optional resident GPU execution for histogram building, split evaluation, row partitioning, and score updates on Apple M-series chips (`--features metal`). Up to 2.7× faster on large workloads.
- 💻 **Cross-Platform CPU Execution**: Highly optimized portable CPU training and inference that runs everywhere Rust runs (macOS, Linux, Windows).
- 📦 **Self-Contained Model Persistence**: Human-readable, versioned native JSON model format with structural validation and exact round-trip predictions. Models trained on Metal can be deployed for CPU-only inference without Metal or macOS dependencies.
- 🎯 **Predictable Missing Value Handling**: Learned missing branch routing for `NaN` values (treats `0.0` as regular numeric data).

---

## Installation

Add `lightgbm-metal-rs` to your `Cargo.toml`:

### CPU-only (portable across macOS, Linux, Windows)

```sh
cargo add lightgbm-metal-rs
```

Or manually in `Cargo.toml`:

```toml
[dependencies]
lightgbm-metal-rs = "0.1"
```

### With Apple Silicon Metal GPU Acceleration (macOS `aarch64`)

```sh
cargo add lightgbm-metal-rs --features metal
```

Or manually in `Cargo.toml`:

```toml
[dependencies]
lightgbm-metal-rs = { version = "0.1", features = ["metal"] }
```

---

## Quick Start

### 1. Regression (L2 Squared Error)

```rust
use lightgbm_metal_rs::{train, DenseMatrix, Model, TrainingConfig};

fn main() -> lightgbm_metal_rs::Result<()> {
    // 6 samples with 1 feature each
    let features = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let targets = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0];

    // Create a zero-copy borrowed row-major dense matrix
    let data = DenseMatrix::new(&features, 6, 1)?;

    let config = TrainingConfig {
        iterations: 10,
        min_data_in_leaf: 1,
        ..Default::default()
    };

    // Train the model
    let trained = train(data, &targets, &config)?;

    // Predict regression scores
    let predictions = trained.model.predict(data)?;
    println!("Predictions: {:?}", predictions);

    // Save model to native JSON
    trained.model.save("model.json")?;

    // Load and run inference anywhere (CPU-only, no Metal required)
    let loaded = Model::load("model.json")?;
    assert_eq!(predictions, loaded.predict(data)?);

    Ok(())
}
```

### 2. Binary Classification (Logloss)

```rust
use lightgbm_metal_rs::{train, DenseMatrix, Objective, TrainingConfig};

fn main() -> lightgbm_metal_rs::Result<()> {
    // 6 samples with 2 features each (row-major)
    let features = [
        0.1, 1.2,
        0.4, 0.9,
        0.2, 1.5,
        2.5, 3.1,
        3.0, 2.8,
        2.8, 3.5,
    ];
    let labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // Binary labels must be strictly 0.0 or 1.0
    let data = DenseMatrix::new(&features, 6, 2)?;

    let config = TrainingConfig {
        objective: Objective::BinaryLogloss,
        iterations: 10,
        min_data_in_leaf: 1,
        ..Default::default()
    };

    let trained = train(data, &labels, &config)?;

    // `predict()` returns calibrated probabilities in [0.0, 1.0] for binary models
    let probabilities = trained.model.predict(data)?;
    println!("Probabilities: {:?}", probabilities);

    // Or get raw uncalibrated margin / log-odds scores
    let raw_scores = trained.model.predict_raw(data)?;
    println!("Raw log-odds: {:?}", raw_scores);

    Ok(())
}
```

### 3. Apple Silicon Metal GPU Acceleration

To train with resident GPU acceleration on Apple Silicon, enable the `metal` feature and set `backend: Backend::Metal`:

```rust
use lightgbm_metal_rs::{backend_available, train, Backend, DenseMatrix, TrainingConfig};

fn main() -> lightgbm_metal_rs::Result<()> {
    // Check if Apple Silicon Metal GPU is available at runtime
    let backend = if backend_available(Backend::Metal).is_ok() {
        Backend::Metal
    } else {
        eprintln!("Metal GPU not available on this machine; falling back to CPU.");
        Backend::Cpu
    };

    let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data = DenseMatrix::new(&values, 6, 1)?;

    let config = TrainingConfig {
        backend,
        iterations: 20,
        min_data_in_leaf: 1,
        ..Default::default()
    };

    let trained = train(data, &labels, &config)?;
    println!(
        "Trained {} trees using {:?} backend in {:.4}s (GPU histograms: {:.4}s)",
        trained.report.trees,
        trained.report.backend,
        trained.report.training_seconds,
        trained.report.histogram_gpu_seconds
    );

    Ok(())
}
```

---

## Configuration

`TrainingConfig` configures the tree growth and execution parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `objective` | `Objective` | `Objective::RegressionL2` | Training objective: `RegressionL2` or `BinaryLogloss`. |
| `backend` | `Backend` | `Backend::Cpu` | Execution backend: `Backend::Cpu` or `Backend::Metal`. |
| `iterations` | `usize` | `100` | Maximum number of boosting trees to fit (must be > 0). |
| `learning_rate` | `f64` | `0.1` | Step size shrinkage used in leaf updates. |
| `num_leaves` | `usize` | `31` | Maximum leaves per tree (must be $\ge$ 2). |
| `max_depth` | `Option<usize>` | `None` | Optional limit on tree depth (root depth is 0). `None` = unlimited. |
| `max_bin` | `usize` | `255` | Maximum histogram bins per feature (range 2–255). |
| `min_data_in_leaf` | `usize` | `20` | Minimum number of samples required in a leaf node. |
| `min_sum_hessian_in_leaf` | `f64` | `0.001` | Minimum sum of Hessians required in a leaf node. |
| `min_gain_to_split` | `f64` | `0.0` | Minimum gain improvement required to split a leaf. |
| `lambda_l1` | `f64` | `0.0` | L1 regularization penalty on leaf values. |
| `lambda_l2` | `f64` | `0.0` | L2 regularization penalty on leaf values. |
| `histogram_workspace_bytes` | `usize` | `256 MiB` | Buffer cap for live histogram and reduction buffers (excludes dataset). |

---

## Data Handling (`DenseMatrix`)

`DenseMatrix` represents a borrowed, row-major slice of `f64` features:

```rust
use lightgbm_metal_rs::DenseMatrix;

let data = [
    1.0, f64::NAN, // Row 0: feature 0 = 1.0, feature 1 is missing
    2.5, 0.0,      // Row 1: feature 0 = 2.5, feature 1 = 0.0 (numeric zero)
];
let matrix = DenseMatrix::new(&data, 2, 2)?;
```

- **Missing Values**: Pass `f64::NAN` to represent missing features. LightGBM's learned missing branch routing automatically determines whether missing values go to the left or right branch during training.
- **Numeric Zeros**: `0.0` is treated as a regular numeric value, not as missing.
- **Labels**:
  - Regression (`RegressionL2`): Any finite `f64` targets.
  - Classification (`BinaryLogloss`): Binary targets must be strictly `0.0` or `1.0`.

---

## Model Persistence & Deployment

Models are serialized to a versioned JSON format with structural integrity validation:

- **Save to disk**: `model.save("path/to/model.json")?`
- **Load from disk**: `let model = Model::load("path/to/model.json")?`
- **In-memory JSON**: `model.to_json()?` and `Model::from_json(&json_string)?`
- **Cross-platform inference**: Models trained on macOS using Metal can be transferred to Linux or Windows and loaded for CPU inference without Metal frameworks or GPU dependencies.

---

## Supported Features & Scope

`lightgbm-metal-rs` focuses on high performance and rock-solid reliability for the core GBDT subset:

| Area | Status | Details |
| --- | --- | --- |
| **Objectives** | Supported | L2 regression (`RegressionL2`) and binary logloss (`BinaryLogloss`). |
| **Tree Growth** | Supported | Leaf-wise (best-first) growth, max leaves, max depth, min data in leaf, min Hessian, min gain, L1/L2 leaf penalties. |
| **Missing Values** | Supported | Native `NaN` missing value routing learned during tree building. |
| **Histogram Binning** | Supported | Uniform/quantile numeric binning with 2–255 bins. |
| **Backends** | Supported | Portable Rust CPU and resident Apple Silicon Metal GPU (`--features metal`). |
| **Inference** | Supported | Fast CPU inference: `predict`, `predict_proba`, `predict_raw`. |
| **Persistence** | Supported | Versioned native JSON with schema validation and exact round trips. |
| *Multiclass / Ranking* | Not supported | Only regression and binary classification are currently implemented. |
| *Sparse / Categorical* | Not supported | Inputs must be dense row-major numeric slices. |
| *GOSS / DART / Subsampling* | Not supported | Standard histogram GBDT only. |
| *Upstream Model Format* | Not supported | Upstream LightGBM text/binary models cannot be loaded; use native JSON. |
| *GPU Inference* | Not supported | Inference is CPU-only. |

---

## Performance Benchmarks

Benchmarks run on an **Apple M4 Pro (20-core GPU)** on a synthetic 1,000,000 row × 64 feature dataset (single CPU thread):

| Task | Configuration | Rust Metal GPU | Rust CPU | Upstream C++ CPU | Speedup (Metal vs Rust CPU) | Speedup (Metal vs Upstream) |
| --- | --- | --- | --- | --- | --- | --- |
| **L2 Regression** | 63 bins | **6.70 s** | 17.48 s | 10.78 s | **2.61×** | **1.61×** |
| **L2 Regression** | 255 bins | **11.57 s** | 22.89 s | 11.55 s | **1.98×** | **1.00×** |
| **Binary Logloss** | 63 bins | **7.48 s** | 20.29 s | 12.76 s | **2.71×** | **1.71×** |
| **Binary Logloss** | 255 bins | **11.89 s** | 24.34 s | 12.34 s | **2.05×** | **1.04×** |

*Timings are complete training medians over 5 warm runs, including binning and preprocessing.*

> [!TIP]
> **When to use Metal:** GPU acceleration delivers significant speedups on large datasets ($\ge 100\text{k}$ rows with tens of features). For very small datasets ($< 10\text{k}$ rows), CPU training is often faster due to GPU buffer allocation and kernel dispatch overhead.

See the [complete benchmark report](docs/benchmarks.md) for full benchmarks across dataset sizes, cold timings, and reproduction steps.

---

## Running Examples

Clone the repository to explore the built-in examples:

```sh
git clone https://github.com/alfinsyahruddin/lightgbm-metal-rs.git
cd lightgbm-metal-rs

# Run CPU training example
cargo run --locked --no-default-features --example train

# Run Metal GPU training example (Apple Silicon Mac)
cargo run --locked --features metal --example train -- --metal
```

---

## Development & Verification

Run the test suite and linters:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --locked --no-default-features
cargo doc --locked --all-features --no-deps
```

On Apple Silicon with Metal hardware access, run the GPU tests:

```sh
cargo test --locked --features metal -- --ignored --nocapture --test-threads=1
```

---

## Documentation

- [docs.rs Documentation](https://docs.rs/lightgbm-metal-rs)
- [Native Model Architecture & JSON Format](docs/native-model.md)
- [Apple Silicon Metal Execution & Memory Management](docs/metal.md)
- [Complete-Training Benchmark Report](docs/benchmarks.md)
- [Reference Validation & Fixtures](validation/README.md)
- [Guidance for Coding Agents](AGENTS.md)

---

## Contributing

Contributions are welcome: bug reports, documentation, tests, performance investigations, and implementation improvements. Please use the [GitHub issues](https://github.com/alfinsyahruddin/lightgbm-metal-rs/issues) and [pull requests](https://github.com/alfinsyahruddin/lightgbm-metal-rs/pulls) for this repository.

---

## License & Attribution

Licensed under the [MIT License](LICENSE).

Copyright for this project's contributions is attributed to the `lightgbm-metal-rs` contributors. Translated algorithms retain original notices from Microsoft Corporation and LightGBM contributors. See the [official LightGBM repository](https://github.com/lightgbm-org/LightGBM) for the upstream project.
