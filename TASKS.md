# Metal training performance tasks

## Goal and scope

Maximize complete-training speed for the existing dense numeric L2 regression and binary logloss subset by moving more work onto Metal, keeping training state resident, improving kernel efficiency, and reducing CPU/GPU synchronization. Optimize measured time to a usable model, including input validation, preprocessing, setup, fitting, and model export. GPU utilization and dispatch count are diagnostic metrics, not the success criterion.

This is a future implementation checklist; unchecked tasks are not implemented or validated. Work in dependency order, prioritize measured bottlenecks, and record evidence before marking a task complete. Preserve the public CPU backend, portable native models, pinned upstream reference, supported configuration domains, and current numerical/quality gates. Substantial API or compatibility changes need discussion before implementation.

## Starting evidence

Baseline implementation: commit `3342485`. Measurements below come from the [existing benchmark report](docs/benchmarks.md) and [raw results](validation/results/benchmarks.json), recorded on an Apple M4 Pro with 20 GPU cores. Times are medians of five warm runs; CPU baselines use one thread. They are historical measurements, so collect a same-machine baseline again before comparing a new implementation.

| Case | Rust CPU | Current Metal | Upstream CPU | Metal histogram wall / GPU time |
| --- | ---: | ---: | ---: | ---: |
| 1,000 × 16, L2, 63 bins | 0.0214 s | 0.3109 s | 0.0150 s | 0.2896 / 0.0222 s |
| 100,000 × 32, L2, 255 bins | 1.4132 s | 1.8410 s | 0.5832 s | 1.2831 / 0.6084 s |
| 1,000,000 × 64, L2, 63 bins | 20.0245 s | 8.8647 s | 10.8287 s | 3.7007 / 2.9392 s |
| 1,000,000 × 64, L2, 255 bins | 25.8519 s | 19.9655 s | 11.5723 s | 13.7833 / 12.7692 s |

The small-case wall/GPU gap includes encoding, queueing, waiting, copying, and other host overhead; it is not a separate measurement of synchronization alone. Large 255-bin cases require substantial kernel improvement as well as less host work. Existing GPU timestamps combine partial accumulation and reduction.

The current [histogram kernel](kernels/histogram.metal) rereads a leaf's rows for each 16-bin tile: four tile scans at 63 bins and sixteen at 255 bins. [MetalHistograms](src/histogram/metal.rs) uploads row membership, waits for each histogram command, and returns a full host histogram. [Training](src/training.rs) performs leaf totals, split scans, leaf selection, sibling subtraction, row partitioning, derivatives, and score updates on CPU.

## Intended execution boundary

| Stage | Current execution | Target execution |
| --- | --- | --- |
| Input/configuration validation and model validation | Rust CPU | Rust CPU |
| Numeric bin learning and initial score | Rust CPU | Profile first; preserve exact bin semantics |
| Scores, labels, derivatives, row membership | CPU arrays plus GPU copies | Resident GPU training state |
| Histograms and sibling subtraction | GPU construction; CPU subtraction/cache | GPU construction, cache, and subtraction |
| Leaf totals, split scans, candidate reduction | Rust CPU | GPU, with precision safeguards |
| Leaf choice and row partitioning | Rust CPU | GPU metadata and partition kernels |
| Score updates and next-round derivatives | Rust CPU | GPU, preferably fused where justified |
| Training control and model export | Rust CPU | Minimal host coordination; bounded GPU work |
| Saved-model inference and persistence | Rust CPU | Preserve portable Rust CPU behavior |

More GPU work is useful when it improves end-to-end results. GPU inference, new objectives/data types, distributed training, and a new automatic backend API are separate efforts. Kernel selection within the explicit Metal backend may adapt to workload/device capabilities; backend failures must still return errors without silent CPU fallback.

## 0. Freeze comparisons and identify costs — first priority

- [ ] **0.1 Reproduce the baseline.** Preserve the existing reports and inputs. Collect the full 12-case matrix for the baseline and candidate on the same machine, release profile, data hashes, parameters, and timing boundary. Record commit, binary/kernel hashes, hardware, OS, Rust, power mode, warm-up, and competing workload conditions.
- [ ] **0.2 Measure each training stage.** Separate validation/binning, initial score, allocation/pipeline setup, derivatives, leaf totals, histogram partials/reduction, split search, leaf selection, partitioning, score updates, and model export. Record command-buffer submissions, host waits, bytes written/read, allocation counts, leaf-size distributions, workspace, and resident/process memory. Keep expensive profiling optional and verify its timing overhead.
- [ ] **0.3 Capture CPU and GPU traces.** Use Instruments Metal System Trace and CPU sampling on representative small, medium, large, and 255-bin cases. Attribute idle gaps and distinguish memory bandwidth, repeated scans, occupancy/register pressure, command encoding, and synchronization. Use instrumented runs for diagnosis and uninstrumented runs for performance acceptance.
- [ ] **0.4 Lock performance gates before tuning.** Prioritize lower median complete-training time across the eight medium/large cases, especially the 255-bin cases, while reducing small-case latency. Estimate run-to-run noise; use paired/interleaved comparisons and extra repetitions near that noise floor. Select a minimum meaningful improvement and a per-case regression budget before evaluating candidates. Report per-case ratios and geometric means separately by size; retain all slower cases. Upstream CPU remains a comparison, not a guaranteed speedup target.
- [ ] **0.5 Freeze additional stress workloads.** Add missing/constant-feature, skewed-leaf, deeper-tree, long-boosting, tight-workspace, and feature-count cases before tuning on them. Keep at least one workload set out of tuning. Retain the original 12 cases for continuity, and add a separately labeled upstream CPU comparison using available CPU cores so single-thread results do not stand in for production performance.
- [ ] **0.6 Measure compiler-profile changes separately.** Compare the newly configured release settings with the baseline profile before attributing gains to GPU work. Record optimization level, LTO, codegen units, panic strategy, symbol stripping, and relevant target flags. Measure Rust CPU and Metal complete-training results, binary size, setup costs, and compile-time tradeoffs; preserve symbols in a separate profiling build when needed. Hold these settings fixed during kernel comparisons and discuss panic/compatibility implications before adopting a changed release policy.

**Exit evidence:** reproducible baseline, stage profiles, declared comparison thresholds, frozen additional inputs, and a ranked bottleneck list.

## 1. Enable resident training state and settle precision

Dependencies: stage 0. This stage enables GPU split search, partitioning, and boosting without forcing full histogram readbacks through the existing interface.

- [ ] **1.1 Design an internal training execution interface.** Retain the CPU learner as an independently usable correctness reference. Define backend operations for resident histograms, candidate evaluation, row partitioning, score/derivative updates, and finalization rather than requiring every operation to return `Vec<Stat>` to CPU. Document which steps still need host results and why. Keep public training/model APIs compatible unless a concrete change is discussed.
- [ ] **1.2 Allocate a reusable training session.** Keep bins, labels, scores, derivatives, row-index storage, leaf descriptors, row-to-leaf mappings, and error/status buffers resident for a fit. Use explicit ownership, checked dimensions, host/device layout checks, and safe lifetimes. Record resident bytes separately from histogram workspace, including buffers retained between operations.
- [ ] **1.3 Reuse immutable setup safely.** Measure pipeline compilation and allocation costs across repeated fits. Cache compatible pipelines and reuse bounded scratch storage where beneficial, without sharing mutable training state between independent/concurrent fits. Test reset, failure cleanup, and repeated-fit isolation; keep cold and warm setup costs visible in reports.
- [ ] **1.4 Specify and validate GPU numerical representations.** Current scores, gain calculations, leaf outputs, thresholds, and configuration scalars use host `f64`; derivatives and GPU histogram sums use `f32`. Verify precision support on the declared Metal/MSL/device targets. Do not assume native `f64` or silently cast the entire learner to `f32`. Evaluate compensated representations, controlled reductions, and compact CPU verification of ambiguous candidates where needed. Define handling of extreme valid parameters, near-tied gains, tiny Hessians, cancellation, overflow, and nonfinite values; demonstrate the existing gates before adopting a representation.
- [ ] **1.5 Define command dependencies and failure propagation.** Track buffer reads/writes, required barriers, and command completion. Keep CPU reads and buffer reuse after successful completion. GPU error flags must stop training and become typed errors before any model is returned. Test injected failures and real dependent dispatches, including cancellation/recomputation paths.
- [ ] **1.6 Enforce bounded memory across stages.** Count simultaneously live GPU histogram/cache/reduction buffers, host verification copies, and new scan/scatter scratch. Preserve `histogram_workspace_bytes` as the histogram cap; classify other workspace separately and document it. Prevent unbounded scratch/cache growth, check device allocation limits, and test minimum-budget behavior and cleanup after errors.

**Exit evidence:** reviewed execution/precision design, reusable isolated sessions, safe synchronization, bounded allocations, and unchanged CPU behavior.

## 2. Improve histogram kernels and keep caches on GPU — early payoff

Dependencies: stage 0 for standalone kernel experiments; stage 1 for resident caches and later fusion.

- [ ] **2.1 Remove avoidable repeated bin-tile scans.** Benchmark wider bin coverage per loaded row, threadgroup-local accumulation, register/SIMD variants, and feature tiling against the current 16-bin kernel. Measure register pressure, occupancy, contention, traffic, and actual training time, especially at 255 bins. Feature-gate atomics or other capabilities; retain a supported variant for existing targets.
- [ ] **2.2 Tune reduction and chunk sizing.** Measure partial accumulation and reduction separately. Compare chunk sizes, reduction trees, threadgroup widths, and one-pass small-leaf kernels. Use measured variants for 2/63/255 bins and different leaf sizes; preserve exact counts and numerical bounds under irregular membership and cancellation.
- [ ] **2.3 Skip unused feature/bin work.** Exclude nonsplittable columns and evaluate compact feature/bin layouts instead of allocating/processing every `max_bin` slot. Preserve original feature indices, missing bins, and model thresholds. Account for layout preparation cost in complete-training timing.
- [ ] **2.4 Store bounded histograms and sibling subtraction on GPU.** Construct the smaller child, derive its sibling, and keep cached histograms resident through candidate evaluation. Carry exact counts, detect cancellation/invalid subtraction, and directly recompute when required. Compare derived versus direct histograms under cache eviction, low budgets, and near-zero Hessians.
- [ ] **2.5 Remove redundant staging and allocation.** Avoid uploading row membership already produced on GPU, recreating histogram copies, and allocating root rows or per-leaf vectors unnecessarily. Return only required compact results to Rust. Record bytes and allocations saved and rerun affected correctness and performance cases.

**Exit evidence:** accepted kernel variants, direct/subtracted histogram agreement, bounded resident cache, and measured kernel/end-to-end gains with slower cases retained.

## 3. Move split search and leaf totals to Metal

Dependencies: stages 1 and 2. Keep a staged integration path with compact host verification before removing host decisions.

- [ ] **3.1 Compute leaf statistics on GPU.** Produce gradient/Hessian totals and exact counts from the current row membership. Reuse partition statistics when valid and avoid rescanning all rows on CPU. Verify totals independently of feature histograms and propagate numerical errors.
- [ ] **3.2 Implement GPU threshold scans.** Scan numeric bins and both eligible missing directions; enforce minimum rows/Hessians/gain, L1/L2 penalties, and depth/leaf eligibility. Match constant/all-missing, missing-only splits, terminal bounds, and the supported scan semantics. Apply the precision decision from 1.4.
- [ ] **3.3 Reduce candidates with explicit tie ordering.** Reduce feature/threshold candidates to per-leaf winners, then select the highest-gain eligible leaf. Define deterministic feature/threshold/missing-direction/leaf ordering for equal represented gains. Test exact ties, near ties, no-split leaves, depth boundaries, and repeated GPU runs; cross-backend tree identity remains a separate concern from numerical agreement.
- [ ] **3.4 Replace full histogram readbacks with compact candidates.** Return candidate indices, counts, sums, and status only when the host needs them. Recover native `f64` thresholds from Rust bin metadata rather than rounding them through GPU floats. If precision certification needs host work, bound and measure that work; do not make approximate ranking silently skip a materially better candidate.
- [ ] **3.5 Integrate scans with histogram reduction/subtraction.** Chain dependent kernels without a host wait between them. Fuse operations only when profiling shows lower complete-training time and resource use. Compare staged versus fused implementations and verify all intermediate calculations and frozen model-quality gates on hardware.

**Exit evidence:** correct GPU totals/scans/selection, compact readbacks, and a measured reduction in host work and synchronization.

## 4. Move row partitioning and leaf metadata to Metal

Dependencies: stage 3.

- [ ] **4.1 Implement predicate, prefix scan, and scatter.** Partition leaf row indices by binned thresholds and learned missing direction using bounded reusable scratch. Maintain deterministic/stable membership ordering where needed for controlled reduction order; avoid a global atomic bottleneck. Test empty, singleton, skewed, irregular, and missing-heavy leaves.
- [ ] **4.2 Keep membership and leaf descriptors resident.** Maintain leaf offsets/counts/depth, histogram/cache ownership, candidate validity, and row-to-leaf mapping on GPU. Update only the split leaf and new child; invalidate stale candidates/cache entries and refresh only the affected histograms.
- [ ] **4.3 Verify partition invariants.** Every parent row must occur in exactly one child, no row may be lost/duplicated, child counts must match exact histogram counts, and all indices must be in bounds. Exercise depth/leaf limits, final splits, cache eviction, and repeated fits on real hardware.
- [ ] **4.4 Chain partition, child histograms, subtraction, and candidate scans.** Submit these dependent passes together where supported. Read back only compact decisions still required by the host, and measure waits/submissions per split as well as complete-training time.

**Exit evidence:** GPU membership generation without per-leaf uploads, exact partition invariants, and successful end-to-end tree training.

## 5. Move boosting scores and objectives to Metal

Dependencies: stage 1 precision policy and stage 4 row-to-leaf mapping.

- [ ] **5.1 Implement resident L2 derivatives.** Calculate score-minus-label gradients and unit Hessians using the selected representation. Preserve label validation, initial-score semantics, finite-result checks, and conversion behavior at the CPU/GPU boundary. Verify intermediate values against CPU and frozen evidence.
- [ ] **5.2 Implement stable binary derivatives.** Compute sigmoid-based gradients/Hessians with stable positive/negative branches. Verify extreme logits, label-specific cancellation behavior, clipped initial scores, balanced/imbalanced targets, and single-class constant models. Meet existing numerical and logloss gates without loosening tolerances.
- [ ] **5.3 Update training scores from leaf assignments.** Apply shrunken leaf outputs using resident row-to-leaf indices instead of traversing the entire tree for every training row. Preserve native model output values and check accumulated score error over long boosting runs; constant/no-gain stopping must remain correct.
- [ ] **5.4 Fuse score update and next-round derivatives where useful.** Keep the new derivatives available for the next histogram pass without host refresh/copy. Measure saved memory passes and submissions; retain separate kernels when they are faster or numerically safer.
- [ ] **5.5 Preserve portable final models.** Export compact tree nodes and original thresholds, construct and validate the native Rust model, and verify save/load equality and CPU-only deployment from a Metal-trained model. Compare GPU training scores to CPU model predictions within the established numerical policy; exact serialization round trips remain a separate gate.

**Exit evidence:** resident scores/derivatives, correct L2/binary boosting, long-run numerical evidence, and portable validated models.

## 6. Reduce host coordination and optimize remaining costs

Dependencies: stages 3–5; select experiments from updated profiles.

- [ ] **6.1 Reduce tree-growth synchronization.** Compare minimal per-split compact readback with bounded pre-encoded/device-controlled growth using leaf/status buffers. Verify device/API support, dependent-pass barriers, early termination, maximum work bounds, allocation limits, and command failure reporting. Avoid unbounded persistent kernels or assuming unsupported dynamic dispatch.
- [ ] **6.2 Batch successive boosting passes safely.** Chain tree finalization, score update, derivatives, and next-root work where dependencies permit. Keep no-gain stopping, error/cancellation checkpoints, and model visibility correct. Measure end-to-end latency and host waits; overlapping buffers must not introduce stale state.
- [ ] **6.3 Reprofile preprocessing and export.** Once GPU fitting is faster, measure the remaining input validation, binning, initial-score, and final model-construction costs. Consider parallel CPU preprocessing or exact GPU bin assignment only when material. Preserve learned boundaries, zero/NaN behavior, feature dimensions, and timing inclusion; rerun bin fixtures before accepting changes.
- [ ] **6.4 Evaluate newer Metal APIs as optional variants.** Investigate indirect dispatch, argument buffers, reusable command encoding, and newer command APIs only where available and beneficial. Record device/MSL/macOS capabilities and compatibility costs; preserve existing supported targets or explicitly discuss and document any proposed minimum-version change.
- [ ] **6.5 Validate workload-dependent kernel selection.** Choose GPU variants from measured leaf size, bin/feature count, and memory limits. Test selection boundaries and all reachable variants. Do not silently select CPU histograms when Metal was requested; a separate automatic backend contract would require its own design and validation.

**Exit evidence:** lower coordination overhead, honest profiling of remaining costs, capability-checked variants, and no hidden compatibility regression.

## 7. Final acceptance and documentation

Dependencies: completed candidate stages and the frozen performance gates from 0.4.

- [ ] **7.1 Pass intermediate GPU checks.** Verify histogram sums/counts, direct/subtracted siblings, leaf totals, split candidates, partition membership, derivatives, score updates, limits, reuse, and failure handling independently. Add meaningful regressions for newly introduced kernels and synchronization boundaries.
- [ ] **7.2 Pass existing model gates unchanged.** Keep CPU small-fixture tolerance `1e-8 + 1e-6 * abs(reference)` and Metal `1e-5 + 1e-4 * abs(reference)`. Histogram counts remain exact; each sum error is <= `1e-5 + 1e-4 * sum(abs(contributing values))`. Frozen regression RMSE remains <= `1.02 * baseline + 1e-6`, and binary logloss <= `baseline + 0.01`, against upstream and, for Metal, Rust CPU. Preserve CPU repeatability and exact native round trips. Do not regenerate fixtures to repair failures.
- [ ] **7.3 Verify memory, compatibility, and concurrency.** Exercise tight/default budgets, resident footprint limits, repeated fits, isolated concurrent sessions, unsupported devices, and cleanup after allocation/command errors. CPU-only builds and inference must retain independence from Metal and reference tooling.
- [ ] **7.4 Run portable and real-GPU validation.** Run formatting, Clippy, CPU tests, API docs, package audit, and `python3 scripts/validate.py --require-metal` as required by [AGENTS.md](AGENTS.md). Verify new GPU tests are invoked by the validator. Record hardware/software details and exact commands; unavailable hardware is unverified, not passed.
- [ ] **7.5 Publish reproducible before/after performance evidence.** Recollect the original matrix with one cold call, one warm-up, and at least five measured warm runs per baseline/case, plus frozen extra/holdout and multi-thread upstream comparisons. Include medians, dispersion, quality, preprocessing/setup/fit/export times, submissions/waits, transfer bytes, and tracked/resident/process memory. Preserve previous reports under distinct names and retain all slower cases. Reject or revise changes that miss the predeclared performance/quality gates.
- [ ] **7.6 Update user and maintainer documentation.** Update README, API reports, Metal execution docs, benchmark definitions/scripts, and memory accounting to describe the implemented boundary and actual results. Distinguish planned work, measured improvements, and remaining limits. Mark tasks complete only with linked verification evidence.

## Source references

Use the bundled CUDA implementations as algorithm references while preserving their MIT notices: [tree learner](lightgbm-py/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp), [split finder](lightgbm-py/src/treelearner/cuda/cuda_best_split_finder.cu), [partitioning](lightgbm-py/src/treelearner/cuda/cuda_data_partition.cu), [binary objective](lightgbm-py/src/objective/cuda/cuda_binary_objective.cu), and [score updater](lightgbm-py/src/boosting/cuda/cuda_score_updater.cu). CUDA mechanisms must be adapted to Metal capabilities rather than translated mechanically.

Apple's [command-buffer guidance](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html) explains the tradeoff between submission frequency, GPU starvation, and CPU/GPU stalls. Apply it to training dependencies using measurements, rather than copying frame-rendering policies. Consult the [Metal resources and language specification](https://developer.apple.com/metal/resources/) and [feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) when choosing precision, atomics, or device-controlled dispatch features.
