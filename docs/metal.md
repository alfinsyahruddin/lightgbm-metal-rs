# Metal execution and validation

The optional `metal` feature targets Apple Silicon macOS 11 or newer with a usable unified-memory device. Kernels compile at runtime with Metal Shading Language 2.2 and fast math disabled. Xcode's compiler is useful for development; the library uses the system Metal framework's runtime compilation path.

```sh
cargo run --locked --features metal --example backend_probe -- --metal
cargo run --locked --features metal --example train -- --metal
python3 scripts/validate.py --require-metal
```

Some sandboxes prevent GPU discovery. Run hardware validation in a process with GPU access. The validator records portable results, hardware provenance, and logs under `validation/results/`. Unavailable hardware is explicitly skipped unless `--require-metal` is set, which fails rather than treating skipped checks as complete acceptance.

The resident byte-binned dataset, gradient/Hessian arrays, ping-pong row-index buffers (`rows_a`, `rows_b`), and score buffers (`scores`, `score_errs`) remain owned on GPU throughout boosting. Shared storage is synchronized by successful command completion before CPU reads or reuse. Histograms, split candidate evaluations, row partitioning, leaf score updates, and loss derivatives execute on the Metal GPU. Tree topology orchestration and candidate threshold certification remain on CPU using native `f64` boundaries. There is no implicit CPU fallback after a Metal failure.

### GPU Kernels and Pipelines

1. **Histogram Accumulation**: For small leaves (≤ 4,096 rows) or datasets with max bins ≤ 64, a single-pass kernel accumulates 64-bin histograms directly in lane-private SIMD registers with warp shuffle reduction. For larger leaves, chunked row-tiled partial accumulation is combined by a reduction kernel. Features identified as unsplittable are skipped directly in GPU threadgroup dispatch.
2. **Split Finding**: The `find_splits` kernel scans all numeric bins and both eligible missing directions in parallel on GPU, enforcing `min_data_in_leaf`, `min_sum_hessian_in_leaf`, L1/L2 penalties, and gain thresholds. Split finding is fused directly with histogram reduction (`construct_and_find_split`) and histogram subtraction (`subtract_and_find_split`) into single command buffers. Only compact best candidates (one per feature) are read back to the host.
3. **Resident Partitioning**: Leaf row index partitioning executes on GPU using ping-pong buffers (`rows_a`, `rows_b`). A single-block kernel handles leaves with ≤ 1,024 rows; a 3-pass multi-block kernel (block count, SIMD prefix scan, scatter) partitions larger leaves. Row index uploads between host and GPU are completely eliminated during tree fitting.
4. **Resident Objectives and Boosting Scores**: Initial scores and gradients/Hessians are computed on GPU (`init_scores_and_derivatives`). At tree completion, leaf scores are updated directly in resident row buffers using Kahan compensated summation (`scores` + `score_errs`) and next-round derivatives are calculated on GPU in a single fused command buffer (`update_scores_and_derivatives`), eliminating CPU tree traversal.

### Memory Accounting and Workspace Budget

`histogram_workspace_bytes` limits all live host histogram/cache buffers and GPU partial/output buffers. The default is 256 MiB. A direct Metal histogram needs at least two times `feature_count * max_bin * 24` bytes. Resident datasets, ping-pong row buffers, score/error buffers, candidate structures, model nodes, and Metal framework overhead are accounted separately.

A bounded histogram buffer pool (`HistHandle`) enables parent/child sibling subtractions and reuses allocated device buffers across tree levels. Subtractions detect negative bin counts or non-finite values and cancel cleanly before model corruption occurs.

### Hardware Validation and Portability

Hardware checks cover irregular row membership, missing/empty bins, cancelling gradients, 2/63/255 bins, buffer reuse, tight budgets, compilation failures, small model predictions, frozen quality, isolated concurrent sessions across threads, and native model round trips. Allocation and failed-command handling use injected boundary errors; real commands check completion status and device errors before reading.

After GPU training completes, models are exported into portable native Rust structures. The `load_metal_model` example and validation suite verify that CPU-only builds without Metal frameworks load Metal-trained models and compute bit-for-bit identical predictions.
