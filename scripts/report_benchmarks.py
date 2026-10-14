#!/usr/bin/env python3
"""Render the completed benchmark evidence as reviewable Markdown."""
import json
from pathlib import Path
import statistics

from build_reference import ROOT


def median_field(baseline, field):
    return statistics.median(run[field] for run in baseline["runs"] if run["phase"] == "measured_warm")


def report():
    source = ROOT / "validation/results/benchmarks.json"
    data = json.loads(source.read_text())
    if not data["complete"] or len(data["cases"]) != data["expected_cases"]:
        raise RuntimeError("benchmark matrix is incomplete")
    lines = ["# Complete-training benchmarks", "",
        "Measured on the Apple M4 Pro (20 GPU cores), using the pinned bundled LightGBM source and the Rust package in this repository. CPU baselines use one thread. These are measurements for these workloads and settings; backend selection remains explicit.", "",
        f"Captured at `{data['timestamp_utc']}`. Rust: `{data['rust']}`. NumPy: `{data['numpy']}`.", "",
        "Each case contains one first-call cold run, one warm-up, and five measured warm runs in a fresh process per baseline. The tables report medians of the five warm runs. OS shader caches may already be warm. Raw-data generation and file I/O are excluded; matrix validation, binning/Dataset construction, backend setup, and fitting are included. Twenty percent of rows are held out for validation. All models use 100 rounds, 31 leaves, learning rate 0.1, no sampling, and the matched supported parameter subset.", "",
        "The raw report contains every run, phase timings, input hashes, generator details, resolved configuration, source/dependency identities, binary/kernel/lockfile hashes, hardware/software provenance, tracked histogram memory, and process peak RSS.", "",
        "## Warm end-to-end time", "",
        "| Total rows × features | Objective | Bins | Upstream CPU (s) | Rust CPU (s) | Rust Metal (s) | Rust CPU / Metal | Upstream CPU / Metal |",
        "|---|---|---:|---:|---:|---:|---:|---:|" ]
    ratios=[]
    for case in data["cases"]:
        inp=case["input"]
        upstream,cpu,metal=case["baselines"]
        u,c,m=[b["median_warm"]["end_to_end_seconds"] for b in (upstream,cpu,metal)]
        ratios.append(c/m)
        objective="L2" if inp["objective"]=="regression_l2" else "Binary"
        lines.append(f"| {inp['rows']:,} × {inp['cols']} | {objective} | {inp['max_bin']} | {u:.4f} | {c:.4f} | {m:.4f} | {c/m:.2f}× | {u/m:.2f}× |")
    faster=sum(r>1 for r in ratios)
    lines += ["",f"Metal is faster than Rust CPU in {faster}/{len(ratios)} cases. Ratios below 1 mean Metal is slower. Both faster and slower cases are retained; no universal GPU speedup is claimed.","",
        "## Cold first-call time", "", "| Case | Upstream CPU (s) | Rust CPU (s) | Rust Metal (s) |", "|---|---:|---:|---:|"]
    for case in data["cases"]:
        cold=[b["runs"][0]["end_to_end_seconds"] for b in case["baselines"]]
        lines.append(f"| {case['input']['name']} | {cold[0]:.4f} | {cold[1]:.4f} | {cold[2]:.4f} |")
    lines += ["", "## Validation quality", "", "| Case | Metric | Upstream CPU | Rust CPU | Rust Metal |", "|---|---|---:|---:|---:|"]
    for case in data["cases"]:
        quality=[b["median_warm"]["quality"] for b in case["baselines"]]
        metric="RMSE" if case["input"]["objective"]=="regression_l2" else "Logloss"
        lines.append(f"| {case['input']['name']} | {metric} | {quality[0]:.8f} | {quality[1]:.8f} | {quality[2]:.8f} |")
    lines += ["", "Binary trees can differ from upstream because this implementation enforces minimum rows using exact counts, while upstream can estimate counts from Hessians. The frozen acceptance datasets are checked separately by the validation suite. Benchmark quality is reported for every baseline rather than assuming equal learned trees.", "",
        "## Metal histogram profile", "", "| Case | Histogram wall (s) | Partial + reduction GPU (s) | Calls on leaves ≤ 4,096 rows | Peak tracked histogram (MiB) |",
        "|---|---:|---:|---:|---:|"]
    for case in data["cases"]:
        metal=case["baselines"][2]
        wall=median_field(metal,"histogram_wall_seconds")
        gpu=median_field(metal,"histogram_gpu_seconds")
        small=median_field(metal,"small_leaf_histogram_calls")/median_field(metal,"histogram_calls")
        peak=max(r["peak_histogram_bytes"] for r in metal["runs"])/(1024*1024)
        lines.append(f"| {case['input']['name']} | {wall:.4f} | {gpu:.4f} | {small:.1%} | {peak:.2f} |")
    lines += ["", "Histogram wall time includes command encoding, queueing, synchronization, copies, and GPU execution. GPU timestamps cover the partial/reduction command together. Small leaves pay submission and synchronization overhead despite little arithmetic. Larger workloads provide enough work for the GPU to amortize that cost. The partial histogram buffer scales with row chunks, features, and bins; row tiling and bounded caches keep tracked allocations within the configured 256 MiB cap.", "",
        "The final measured implementation drains temporary command/encoder references after each completed histogram to bound their lifetimes. No isolated speedup is attributed to that change. Profiling counters distinguish GPU work from wall time and identify small-leaf calls. GPU timestamps cover both kernels together, so they do not isolate partial accumulation from reduction. Further kernel restructuring and automatic CPU/GPU routing are deferred: the recorded profiles show submission overhead dominates small workloads, while the full matrix documents the current performance boundary. Initial small/medium measurements are retained in `validation/results/benchmarks-small-initial.json` and `benchmarks-medium-profile.json`.", "",
        "## Memory interpretation", "", "Peak tracked histogram bytes include live host histogram/cache arrays and GPU partial/output buffers. Raw numeric inputs, bins, gradients/Hessians, row membership, scores, model nodes, and Metal framework overhead are additional memory. Process peak RSS is recorded separately in the raw report; it is not the histogram workspace cap and is not a full accounting of GPU/framework memory.", "",
        "## Reproduce", "", "Use Python 3.11+ with NumPy 2.3.5 for benchmarks; reference fixture tools need only Python's standard library. A usable Apple Silicon Metal device and the reference build prerequisites are required.", "",
        "```sh", "python3 scripts/build_reference.py --fetch", "python3 scripts/validate.py --require-metal",
        "python3 scripts/benchmark.py --sizes small,medium,large --runs 5 --output benchmarks.json",
        "python3 scripts/report_benchmarks.py", "```", "",
        "The benchmark freezes inputs before measuring, checks their hashes between baselines, and writes atomic report checkpoints after each case. An incomplete report has `complete: false`. Data files are recreated under ignored `.build/bench/`; reproducible hashes and generator settings are preserved in the report.", ""]
    output=ROOT/"docs/benchmarks.md"
    output.write_text("\n".join(lines))
    print(f"Created {output}")


if __name__=="__main__":
    report()
