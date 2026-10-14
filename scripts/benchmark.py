#!/usr/bin/env python3
"""Freeze benchmark inputs and compare complete training on three baselines."""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

from build_reference import ROOT
from reference import Reference

WORK = ROOT / ".build/bench"
RESULTS = ROOT / "validation/results"


def file_hash(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(size, rows, cols, objective, bins):
    directory = WORK / f"{size}_{objective}"
    directory.mkdir(parents=True, exist_ok=True)
    x_path, y_path = directory / "features.f64", directory / "labels.f64"
    if not x_path.exists() or not y_path.exists():
        random = np.random.Generator(np.random.PCG64(20261014))
        x = random.uniform(-1, 1, (rows, cols))
        noise = random.uniform(-1, 1, rows)
        score = 2*x[:,0]-x[:,1]+0.7*x[:,2]*x[:,3]+np.where(x[:,4]>0,0.5,-0.5)
        y = (score+0.15*noise).astype(np.float32).astype(np.float64) if objective == "regression_l2" else (score+0.3*noise>0).astype(np.float64)
        x.astype("<f8", copy=False).tofile(x_path)
        y.astype("<f8", copy=False).tofile(y_path)
    case = {"name":f"{size}_{objective}_{bins}", "rows":rows, "cols":cols, "train_rows":rows*4//5,
        "features":str(x_path.relative_to(ROOT)), "labels":str(y_path.relative_to(ROOT)),
        "objective":objective,"max_bin":bins,"data_sha256":{"features":file_hash(x_path),"labels":file_hash(y_path)},
        "generator":{"seed":20261014,"algorithm":"PCG64","numpy":np.__version__,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"dtype":"float64-le"}
    path = directory / f"case_{bins}.json"
    path.write_text(json.dumps(case, indent=2)+"\n")
    return path, case


def upstream_run(reference, case, x, y):
    lib = reference.lib
    train_rows, cols = case["train_rows"], case["cols"]
    params = {"objective":"regression" if case["objective"]=="regression_l2" else "binary",
        "learning_rate":0.1,"num_leaves":31,"max_depth":-1,"max_bin":case["max_bin"],
        "min_data_in_leaf":20,"min_sum_hessian_in_leaf":0.001,"min_gain_to_split":0,
        "lambda_l1":0,"lambda_l2":0,"bin_construct_sample_cnt":train_rows,"min_data_in_bin":3,
        "feature_pre_filter":"false","enable_bundle":"false","enable_sparse":"false","use_missing":"true",
        "zero_as_missing":"false","device_type":"cpu","deterministic":"true","force_col_wise":"true",
        "num_threads":1,"bagging_fraction":1,"bagging_freq":0,"feature_fraction":1,
        "use_quantized_grad":"false","sigmoid":1,"verbosity":-1}
    encoded = " ".join(f"{k}={v}" for k,v in params.items()).encode()
    dataset, booster = C.c_void_p(), C.c_void_p()
    started = time.perf_counter()
    target = y[:train_rows].astype(np.float32)
    try:
        reference.check(lib.LGBM_DatasetCreateFromMat(C.c_void_p(x.ctypes.data),1,train_rows,cols,1,encoded,None,C.byref(dataset)))
        reference.check(lib.LGBM_DatasetSetField(dataset,b"label",C.c_void_p(target.ctypes.data),train_rows,0))
        prepared = time.perf_counter()
        reference.check(lib.LGBM_BoosterCreate(dataset,encoded,C.byref(booster)))
        initialized = time.perf_counter()
        for _ in range(100):
            finished=C.c_int()
            reference.check(lib.LGBM_BoosterUpdateOneIter(booster,C.byref(finished)))
            if finished.value: break
        fitted = time.perf_counter()
        validation = x[train_rows:]
        raw = np.empty(len(validation), dtype=np.float64)
        length=C.c_int64()
        reference.check(lib.LGBM_BoosterPredictForMat(booster,C.c_void_p(validation.ctypes.data),1,len(validation),cols,1,1,0,-1,b"num_threads=1",C.byref(length),raw.ctypes.data_as(C.POINTER(C.c_double))))
        quality = float(np.sqrt(np.mean((raw-y[train_rows:])**2))) if case["objective"]=="regression_l2" else float(np.mean(np.maximum(raw,0)-raw*y[train_rows:]+np.log1p(np.exp(-np.abs(raw)))))
        return {"preprocessing_seconds":prepared-started,"backend_setup_seconds":initialized-prepared,
            "training_seconds":fitted-initialized,"end_to_end_seconds":fitted-started,"quality":quality,
            "gpu_dispatches":0,"resolved_parameters":params}
    finally:
        if booster.value: reference.check(lib.LGBM_BoosterFree(booster))
        if dataset.value: reference.check(lib.LGBM_DatasetFree(dataset))


def rust_results(path, backend, runs):
    process = subprocess.run(["/usr/bin/time","-l",str(ROOT/"target/release/examples/benchmark"),str(path),backend,str(runs)],cwd=ROOT,text=True,capture_output=True)
    print(process.stderr, end="", flush=True)
    if process.returncode: raise RuntimeError(process.stderr)
    result=json.loads(process.stdout)
    for line in process.stderr.splitlines():
        if "maximum resident set size" in line:
            result["process_peak_rss_bytes"]=int(line.split()[0])
    return result


def verify_inputs(case):
    for name in ("features","labels"):
        if file_hash(ROOT/case[name]) != case["data_sha256"][name]:
            raise RuntimeError(f"benchmark input changed: {case['name']} {name}")


def medians(result):
    warm=[r for r in result["runs"] if r["phase"]=="measured_warm"]
    return {key:statistics.median(r[key] for r in warm) for key in
        ("preprocessing_seconds","backend_setup_seconds","training_seconds","end_to_end_seconds","quality")}


def upstream_results(path, runs):
    case=json.loads(Path(path).read_text())
    reference=Reference()
    x=np.fromfile(ROOT/case["features"],dtype="<f8").reshape(case["rows"],case["cols"])
    y=np.fromfile(ROOT/case["labels"],dtype="<f8")
    result={"backend":"upstream_cpu","runs":[]}
    for i in range(runs+2):
        run=upstream_run(reference,case,x,y)
        run["phase"]="cold_process" if i==0 else "warmup" if i==1 else "measured_warm"
        result["runs"].append(run)
        print(f"upstream {run['phase']}: {run['end_to_end_seconds']:.3f}s",file=sys.stderr,flush=True)
    result["process_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result))


def run_upstream(path, runs):
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--upstream-case",str(path),"--runs",str(runs)],cwd=ROOT,text=True,capture_output=True)
    print(result.stderr,end="",flush=True)
    if result.returncode: raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def benchmark(options):
    WORK.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    reference=Reference()
    subprocess.run(["cargo","build","--release","--locked","--features","metal","--example","benchmark"],cwd=ROOT,check=True)
    sizes={"small":(1000,16),"medium":(100000,32),"large":(1000000,64)}
    cases=[]
    # Freeze all input hashes before measuring any implementation result.
    for size in options.sizes.split(","):
        rows,cols=sizes[size]
        for objective in ("regression_l2","binary_logloss"):
            for bins in (63,255): cases.append(prepare(size,rows,cols,objective,bins))
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"platform":platform.platform(),
        "device":subprocess.check_output(["system_profiler","SPDisplaysDataType"],text=True),
        "rust":subprocess.check_output(["rustc","--version"],text=True).strip(),"numpy":np.__version__,
        "upstream_provenance":reference.provenance,"measured_warm_runs":options.runs,
        "warmup_runs":1,"cold_policy":"first training call in each baseline process; OS shader caches may already be warm",
        "cpu_threads":1,"cases":[]}
    report.update({"expected_cases":len(cases),"complete":False,
        "rust_binary_sha256":file_hash(ROOT/"target/release/examples/benchmark"),
        "cargo_lock_sha256":file_hash(ROOT/"Cargo.lock"),
        "kernel_sha256":file_hash(ROOT/"kernels/histogram.metal")})
    output=RESULTS/options.output
    for path,case in cases:
        print(f"Benchmarking {case['name']}",flush=True)
        verify_inputs(case)
        upstream=run_upstream(path,options.runs)
        verify_inputs(case)
        cpu=rust_results(path,"cpu",options.runs)
        verify_inputs(case)
        metal=rust_results(path,"metal",options.runs)
        verify_inputs(case)
        for result in (upstream,cpu,metal): result["median_warm"]=medians(result)
        report["cases"].append({"input":case,"baselines":[upstream,cpu,metal]})
        temporary=output.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report,indent=2)+"\n")
        temporary.replace(output)
    print(f"Saved {output}")
    report["complete"]=True
    temporary=output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report,indent=2)+"\n")
    temporary.replace(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes",default="small,medium,large")
    parser.add_argument("--runs",type=int,default=5)
    parser.add_argument("--output",default="benchmarks.json")
    parser.add_argument("--upstream-case",help=argparse.SUPPRESS)
    options=parser.parse_args()
    if options.runs<1: parser.error("runs must be positive")
    if options.upstream_case: upstream_results(options.upstream_case,options.runs)
    else: benchmark(options)
