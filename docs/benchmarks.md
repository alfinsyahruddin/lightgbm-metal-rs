# Complete-training benchmarks

Measured on the Apple M4 Pro (20 GPU cores), using the pinned bundled LightGBM source and the Rust package in this repository. CPU baselines use one thread. These are measurements for these workloads and settings; backend selection remains explicit.

Captured at `2026-10-15T01:48:17.076822+00:00`. Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`. NumPy: `2.5.4`.

Each case contains one first-call cold run, one warm-up, and five measured warm runs in a fresh process per baseline. The tables report medians of the five warm runs. OS shader caches may already be warm. Raw-data generation and file I/O are excluded; matrix validation, binning/Dataset construction, backend setup, and fitting are included. Twenty percent of rows are held out for validation. All models use 100 rounds, 31 leaves, learning rate 0.1, no sampling, and the matched supported parameter subset.

The raw report contains every run, phase timings, input hashes, generator details, resolved configuration, source/dependency identities, binary/kernel/lockfile hashes, hardware/software provenance, tracked histogram memory, and process peak RSS.

## Warm end-to-end time

| Total rows × features | Objective | Bins | Upstream CPU (s) | Rust CPU (s) | Rust Metal (s) | Rust CPU / Metal | Upstream CPU / Metal |
|---|---|---:|---:|---:|---:|---:|---:|
| 1,000 × 16 | L2 | 63 | 0.0155 | 0.0179 | 0.6578 | 0.03× | 0.02× |
| 1,000 × 16 | L2 | 255 | 0.0351 | 0.0507 | 0.8085 | 0.06× | 0.04× |
| 1,000 × 16 | Binary | 63 | 0.0166 | 0.0177 | 0.6561 | 0.03× | 0.03× |
| 1,000 × 16 | Binary | 255 | 0.0375 | 0.0509 | 0.8043 | 0.06× | 0.05× |
| 100,000 × 32 | L2 | 63 | 0.5357 | 0.9512 | 1.2983 | 0.73× | 0.41× |
| 100,000 × 32 | L2 | 255 | 0.5764 | 1.1707 | 2.1393 | 0.55× | 0.27× |
| 100,000 × 32 | Binary | 63 | 0.5585 | 0.9495 | 1.3109 | 0.72× | 0.43× |
| 100,000 × 32 | Binary | 255 | 0.6092 | 1.2346 | 2.1032 | 0.59× | 0.29× |
| 1,000,000 × 64 | L2 | 63 | 10.7766 | 17.4801 | 6.6964 | 2.61× | 1.61× |
| 1,000,000 × 64 | L2 | 255 | 11.5475 | 22.8942 | 11.5698 | 1.98× | 1.00× |
| 1,000,000 × 64 | Binary | 63 | 12.7606 | 20.2875 | 7.4840 | 2.71× | 1.71× |
| 1,000,000 × 64 | Binary | 255 | 12.3381 | 24.3414 | 11.8899 | 2.05× | 1.04× |

Metal is faster than Rust CPU in 4/12 cases. Ratios below 1 mean Metal is slower. Both faster and slower cases are retained; no universal GPU speedup is claimed.

## Cold first-call time

| Case | Upstream CPU (s) | Rust CPU (s) | Rust Metal (s) |
|---|---:|---:|---:|
| small_regression_l2_63 | 0.0173 | 0.0373 | 0.7403 |
| small_regression_l2_255 | 0.0371 | 0.0524 | 0.8823 |
| small_binary_logloss_63 | 0.0200 | 0.0184 | 0.7803 |
| small_binary_logloss_255 | 0.0390 | 0.0507 | 0.8547 |
| medium_regression_l2_63 | 0.5515 | 0.9267 | 1.3389 |
| medium_regression_l2_255 | 0.5801 | 1.1724 | 2.1834 |
| medium_binary_logloss_63 | 0.5663 | 0.9519 | 1.3604 |
| medium_binary_logloss_255 | 0.6108 | 1.2526 | 2.1809 |
| large_regression_l2_63 | 11.0630 | 17.6055 | 7.2692 |
| large_regression_l2_255 | 11.8563 | 23.0431 | 11.5992 |
| large_binary_logloss_63 | 13.0394 | 20.5487 | 7.5594 |
| large_binary_logloss_255 | 12.6102 | 24.8937 | 12.0771 |

## Validation quality

| Case | Metric | Upstream CPU | Rust CPU | Rust Metal |
|---|---|---:|---:|---:|
| small_regression_l2_63 | RMSE | 0.21080914 | 0.21080914 | 0.21080914 |
| small_regression_l2_255 | RMSE | 0.20919406 | 0.20919406 | 0.20919405 |
| small_binary_logloss_63 | Logloss | 0.14815505 | 0.17178168 | 0.15779579 |
| small_binary_logloss_255 | Logloss | 0.20118925 | 0.18685366 | 0.18479603 |
| medium_regression_l2_63 | RMSE | 0.12617527 | 0.12617527 | 0.12617527 |
| medium_regression_l2_255 | RMSE | 0.15688513 | 0.15688513 | 0.15688514 |
| medium_binary_logloss_63 | Logloss | 0.10766802 | 0.10560894 | 0.10560892 |
| medium_binary_logloss_255 | Logloss | 0.10711566 | 0.10371139 | 0.10371140 |
| large_regression_l2_63 | RMSE | 0.21385728 | 0.21385728 | 0.21385728 |
| large_regression_l2_255 | RMSE | 0.11841503 | 0.11841503 | 0.13228901 |
| large_binary_logloss_63 | Logloss | 0.09816231 | 0.09714242 | 0.09714242 |
| large_binary_logloss_255 | Logloss | 0.09624347 | 0.09841464 | 0.09841464 |

Binary trees can differ from upstream because this implementation enforces minimum rows using exact counts, while upstream can estimate counts from Hessians. The frozen acceptance datasets are checked separately by the validation suite. Benchmark quality is reported for every baseline rather than assuming equal learned trees.

## Metal histogram profile

| Case | Histogram wall (s) | Partial + reduction GPU (s) | Calls on leaves ≤ 4,096 rows | Peak tracked histogram (MiB) |
|---|---:|---:|---:|---:|
| small_regression_l2_63 | 0.4977 | 0.0883 | 70.4% | 0.36 |
| small_regression_l2_255 | 0.6505 | 0.2342 | 71.4% | 1.54 |
| small_binary_logloss_63 | 0.4996 | 0.0890 | 71.5% | 0.29 |
| small_binary_logloss_255 | 0.6484 | 0.2334 | 72.1% | 1.17 |
| medium_regression_l2_63 | 1.1782 | 0.3648 | 38.7% | 1.91 |
| medium_regression_l2_255 | 2.0082 | 0.8543 | 40.3% | 7.75 |
| medium_binary_logloss_63 | 1.1970 | 0.3560 | 40.1% | 1.91 |
| medium_binary_logloss_255 | 1.9749 | 0.8600 | 40.4% | 7.75 |
| large_regression_l2_63 | 5.3298 | 4.0231 | 17.8% | 11.95 |
| large_regression_l2_255 | 10.1216 | 8.6485 | 14.9% | 48.37 |
| large_binary_logloss_63 | 6.0761 | 4.6546 | 12.1% | 11.95 |
| large_binary_logloss_255 | 10.4231 | 8.9112 | 12.3% | 48.37 |

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
