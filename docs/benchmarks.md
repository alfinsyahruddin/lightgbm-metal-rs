# Complete-training benchmarks

Measured on the Apple M4 Pro (20 GPU cores), using the pinned bundled LightGBM source and the Rust package in this repository. CPU baselines use one thread. These are measurements for these workloads and settings; backend selection remains explicit.

Captured at `2026-10-14T08:28:44.521270+00:00`. Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`. NumPy: `2.3.5`.

Each case contains one first-call cold run, one warm-up, and five measured warm runs in a fresh process per baseline. The tables report medians of the five warm runs. OS shader caches may already be warm. Raw-data generation and file I/O are excluded; matrix validation, binning/Dataset construction, backend setup, and fitting are included. Twenty percent of rows are held out for validation. All models use 100 rounds, 31 leaves, learning rate 0.1, no sampling, and the matched supported parameter subset.

The raw report contains every run, phase timings, input hashes, generator details, resolved configuration, source/dependency identities, binary/kernel/lockfile hashes, hardware/software provenance, tracked histogram memory, and process peak RSS.

## Warm end-to-end time

| Total rows × features | Objective | Bins | Upstream CPU (s) | Rust CPU (s) | Rust Metal (s) | Rust CPU / Metal | Upstream CPU / Metal |
|---|---|---:|---:|---:|---:|---:|---:|
| 1,000 × 16 | L2 | 63 | 0.0150 | 0.0214 | 0.3109 | 0.07× | 0.05× |
| 1,000 × 16 | L2 | 255 | 0.0353 | 0.0536 | 0.3957 | 0.14× | 0.09× |
| 1,000 × 16 | Binary | 63 | 0.0167 | 0.0211 | 0.3255 | 0.06× | 0.05× |
| 1,000 × 16 | Binary | 255 | 0.0374 | 0.0541 | 0.3765 | 0.14× | 0.10× |
| 100,000 × 32 | L2 | 63 | 0.5576 | 1.2028 | 1.2758 | 0.94× | 0.44× |
| 100,000 × 32 | L2 | 255 | 0.5832 | 1.4132 | 1.8410 | 0.77× | 0.32× |
| 100,000 × 32 | Binary | 63 | 0.5670 | 1.2159 | 1.2809 | 0.95× | 0.44× |
| 100,000 × 32 | Binary | 255 | 0.6164 | 1.5287 | 1.8849 | 0.81× | 0.33× |
| 1,000,000 × 64 | L2 | 63 | 10.8287 | 20.0245 | 8.8647 | 2.26× | 1.22× |
| 1,000,000 × 64 | L2 | 255 | 11.5723 | 25.8519 | 19.9655 | 1.29× | 0.58× |
| 1,000,000 × 64 | Binary | 63 | 12.5149 | 23.4036 | 10.7465 | 2.18× | 1.16× |
| 1,000,000 × 64 | Binary | 255 | 12.4483 | 28.0358 | 21.3302 | 1.31× | 0.58× |

Metal is faster than Rust CPU in 4/12 cases. Ratios below 1 mean Metal is slower. Both faster and slower cases are retained; no universal GPU speedup is claimed.

## Cold first-call time

| Case | Upstream CPU (s) | Rust CPU (s) | Rust Metal (s) |
|---|---:|---:|---:|
| small_regression_l2_63 | 0.0187 | 0.0248 | 0.4113 |
| small_regression_l2_255 | 0.0357 | 0.0549 | 0.4430 |
| small_binary_logloss_63 | 0.0172 | 0.0213 | 0.4077 |
| small_binary_logloss_255 | 0.0379 | 0.0547 | 0.4335 |
| medium_regression_l2_63 | 0.5571 | 1.1982 | 1.3346 |
| medium_regression_l2_255 | 0.5921 | 1.4025 | 1.8020 |
| medium_binary_logloss_63 | 0.5711 | 1.2312 | 1.4665 |
| medium_binary_logloss_255 | 0.6193 | 1.5288 | 1.9540 |
| large_regression_l2_63 | 10.9624 | 20.7169 | 8.8962 |
| large_regression_l2_255 | 11.6410 | 26.1319 | 19.6923 |
| large_binary_logloss_63 | 12.8111 | 23.2625 | 10.8484 |
| large_binary_logloss_255 | 12.6925 | 27.9541 | 21.5761 |

## Validation quality

| Case | Metric | Upstream CPU | Rust CPU | Rust Metal |
|---|---|---:|---:|---:|
| small_regression_l2_63 | RMSE | 0.21080914 | 0.21080914 | 0.21080914 |
| small_regression_l2_255 | RMSE | 0.20919406 | 0.20919406 | 0.20919406 |
| small_binary_logloss_63 | Logloss | 0.14815505 | 0.17178168 | 0.17231501 |
| small_binary_logloss_255 | Logloss | 0.20118925 | 0.18685366 | 0.18685295 |
| medium_regression_l2_63 | RMSE | 0.12617527 | 0.12617527 | 0.12617527 |
| medium_regression_l2_255 | RMSE | 0.15688513 | 0.15688513 | 0.15688513 |
| medium_binary_logloss_63 | Logloss | 0.10766802 | 0.10560894 | 0.10560894 |
| medium_binary_logloss_255 | Logloss | 0.10711566 | 0.10371139 | 0.10371139 |
| large_regression_l2_63 | RMSE | 0.21385728 | 0.21385728 | 0.21385728 |
| large_regression_l2_255 | RMSE | 0.11841503 | 0.11841503 | 0.11841503 |
| large_binary_logloss_63 | Logloss | 0.09816231 | 0.09714242 | 0.09714242 |
| large_binary_logloss_255 | Logloss | 0.09624347 | 0.09841464 | 0.09841464 |

Binary trees can differ from upstream because this implementation enforces minimum rows using exact counts, while upstream can estimate counts from Hessians. The frozen acceptance datasets are checked separately by the validation suite. Benchmark quality is reported for every baseline rather than assuming equal learned trees.

## Metal histogram profile

| Case | Histogram wall (s) | Partial + reduction GPU (s) | Calls on leaves ≤ 4,096 rows | Peak tracked histogram (MiB) |
|---|---:|---:|---:|---:|
| small_regression_l2_63 | 0.2896 | 0.0222 | 100.0% | 0.37 |
| small_regression_l2_255 | 0.3423 | 0.0262 | 100.0% | 1.59 |
| small_binary_logloss_63 | 0.3034 | 0.0256 | 100.0% | 0.32 |
| small_binary_logloss_255 | 0.3208 | 0.0246 | 100.0% | 1.31 |
| medium_regression_l2_63 | 0.8132 | 0.2056 | 71.1% | 1.91 |
| medium_regression_l2_255 | 1.2831 | 0.6084 | 69.6% | 7.75 |
| medium_binary_logloss_63 | 0.7717 | 0.1926 | 73.0% | 1.91 |
| medium_binary_logloss_255 | 1.2316 | 0.5610 | 69.8% | 7.75 |
| large_regression_l2_63 | 3.7007 | 2.9392 | 33.1% | 11.95 |
| large_regression_l2_255 | 13.7833 | 12.7692 | 22.9% | 48.37 |
| large_binary_logloss_63 | 4.2586 | 3.4103 | 22.1% | 11.95 |
| large_binary_logloss_255 | 14.5496 | 13.4289 | 20.5% | 48.37 |

Histogram wall time includes command encoding, queueing, synchronization, copies, and GPU execution. GPU timestamps cover the partial/reduction command together. Small leaves pay submission and synchronization overhead despite little arithmetic. Larger workloads provide enough work for the GPU to amortize that cost. The partial histogram buffer scales with row chunks, features, and bins; row tiling and bounded caches keep tracked allocations within the configured 256 MiB cap.

The final measured implementation drains temporary command/encoder references after each completed histogram to bound their lifetimes. No isolated speedup is attributed to that change. Profiling counters distinguish GPU work from wall time and identify small-leaf calls. GPU timestamps cover both kernels together, so they do not isolate partial accumulation from reduction. Further kernel restructuring and automatic CPU/GPU routing are deferred: the recorded profiles show submission overhead dominates small workloads, while the full matrix documents the current performance boundary. Initial small/medium measurements are retained in `validation/results/benchmarks-small-initial.json` and `benchmarks-medium-profile.json`.

## Memory interpretation

Peak tracked histogram bytes include live host histogram/cache arrays and GPU partial/output buffers. Raw numeric inputs, bins, gradients/Hessians, row membership, scores, model nodes, and Metal framework overhead are additional memory. Process peak RSS is recorded separately in the raw report; it is not the histogram workspace cap and is not a full accounting of GPU/framework memory.

## Reproduce

Use Python 3.11+ with NumPy 2.3.5 for benchmarks; reference fixture tools need only Python's standard library. A usable Apple Silicon Metal device and the reference build prerequisites are required.

```sh
python3 scripts/build_reference.py --fetch
python3 scripts/validate.py --require-metal
python3 scripts/benchmark.py --sizes small,medium,large --runs 5 --output benchmarks.json
python3 scripts/report_benchmarks.py
```

The benchmark freezes inputs before measuring, checks their hashes between baselines, and writes atomic report checkpoints after each case. An incomplete report has `complete: false`. Data files are recreated under ignored `.build/bench/`; reproducible hashes and generator settings are preserved in the report.
