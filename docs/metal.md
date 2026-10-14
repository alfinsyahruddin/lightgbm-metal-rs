# Metal execution and validation

The optional `metal` feature targets Apple Silicon macOS 11 or newer with a usable unified-memory device. Kernels compile at runtime with Metal Shading Language 2.2 and fast math disabled. Xcode's compiler is useful for development; the library uses the system Metal framework's runtime compilation path.

```sh
cargo run --locked --features metal --example backend_probe -- --metal
cargo run --locked --features metal --example train -- --metal
python3 scripts/validate.py --require-metal
```

Some sandboxes prevent GPU discovery. Run hardware validation in a process with GPU access. The validator records portable results, hardware provenance, and logs under `validation/results/`. Unavailable hardware is explicitly skipped unless `--require-metal` is set, which fails rather than treating skipped checks as complete acceptance.

The resident byte-binned dataset, gradient/Hessian arrays, and row-index buffers stay owned until their commands complete. Gradients/Hessians refresh once per boosting iteration. Shared storage is synchronized by successful command completion before CPU reads or reuse. GPU histograms contain gradient sums, Hessian sums, and exact bin counts; split scans, leaf-wise selection, partitioning, and score updates remain on CPU. There is no implicit CPU histogram fallback after a Metal failure.

Two kernels produce lane-private partial histograms for 16-bin tiles, then combine row chunks in a fixed order. An allocation budget determines row-chunk sizing; partial buffers are reused when capacity fits the budget. A bounded host histogram cache enables smaller-child construction and sibling subtraction. Cache eviction or cancellation can require direct recomputation. Compilation, allocation, unsupported dimensions, numerical overflow, and command errors are returned before any model is reported as successful.

`histogram_workspace_bytes` limits all live host histogram/cache buffers and GPU partial/output buffers. The default is 256 MiB. A direct Metal histogram needs at least two times `feature_count * max_bin * 24` bytes (host doubles/counts plus GPU partial and output). Input features, bins, scores, derivatives, row membership, model nodes, and framework overhead are additional storage. Reports provide peak tracked histogram bytes and successful GPU kernel dispatches; a constant model can have zero dispatches.

Hardware checks cover irregular row membership, missing/empty bins, cancelling gradients, 2/63/255 bins, buffer reuse, tight budgets, compilation failures, small model predictions, frozen quality, and native model round trips. Allocation and failed-command handling use injected boundary errors; real commands also check their completion/error status before reading. After GPU tests produce a model, `load_metal_model` builds without Metal and verifies exact CPU-only prediction equality.

Histogram reports also record wall time, GPU execution time for the partial/reduction command, total histogram calls, and calls for leaves with at most 4,096 rows. GPU timestamp time is measured after successful completion; CPU histogram GPU time is zero. Temporary command and encoder references are drained through an autorelease pool after each completed histogram to bound framework object lifetimes during repeated runs.
