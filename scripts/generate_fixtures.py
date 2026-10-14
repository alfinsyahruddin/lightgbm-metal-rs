#!/usr/bin/env python3
"""Explicitly generate frozen reference fixtures before evaluating the Rust trainer."""
import argparse
import ctypes
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys

from build_reference import ROOT, WORK, source_identity
from reference import Reference

OUT = ROOT / "validation/fixtures"


def f32(value):
    return ctypes.c_float(value).value


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def freeze(name, case):
    path = OUT / f"{name}.json"
    data = encoded(case)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def quality_data(objective, rows=2560, cols=12):
    state = 0x6A09E667F3BCC909
    features, labels = [], []
    def random_value():
        nonlocal state
        state ^= (state << 13) & ((1 << 64) - 1)
        state ^= state >> 7
        state ^= (state << 17) & ((1 << 64) - 1)
        return ((state >> 32) & 0xFFFF) / 65535.0 * 2 - 1
    for _ in range(rows):
        row = [random_value() for _ in range(cols)]
        noise = random_value()
        score = 2 * row[0] - row[1] + 0.7 * row[2] * row[3] + (0.5 if row[4] > 0 else -0.5)
        features.extend(row)
        labels.append(f32(score + 0.15 * noise) if objective == "regression_l2" else float(score + 0.3 * noise > 0))
    return features, labels


def metrics(objective, labels, raw):
    if objective == "regression_l2":
        return {"rmse": math.sqrt(sum((p-y)**2 for p,y in zip(raw, labels)) / len(labels))}
    return {"logloss": sum(max(p, 0) - p*y + math.log1p(math.exp(-abs(p))) for p,y in zip(raw, labels)) / len(labels)}


def bin_probe():
    source = WORK / "source/lightgbm-py"
    probe = WORK / "bin_probe"
    lib = ROOT / json.loads((WORK / "provenance.json").read_text())["library"]
    command = ["c++", "-std=c++17", "-O2", str(ROOT / "validation/bin_probe.cpp"),
               "-I", str(source / "include"), "-I", str(source / "external_libs/fast_double_parser/include"),
               "-I", str(source / "external_libs/fmt/include"), str(lib),
               f"-Wl,-rpath,{lib.parent}", "-o", str(probe)]
    subprocess.run(command, check=True)
    return probe


def generate():
    pin = source_identity()
    reference = Reference()
    probe = bin_probe()
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    base = {"iterations": 4, "learning_rate": 0.1, "num_leaves": 4, "max_bin": 63,
            "min_data_in_leaf": 3, "min_sum_hessian_in_leaf": 0.001}
    values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, None]
    for name, objective, overrides, column in [
        ("regression", "regression_l2", {}, values),
        ("binary", "binary_logloss", {}, values),
        ("regularized", "regression_l2", {"lambda_l1": 1.0, "lambda_l2": 2.0}, values),
        ("depth", "regression_l2", {"max_depth": 1}, values),
        ("leaf_limit", "regression_l2", {"num_leaves": 2}, values),
        ("min_rows", "regression_l2", {"min_data_in_leaf": 14}, values),
        ("min_hessian", "binary_logloss", {"min_sum_hessian_in_leaf": 10.0}, values),
        ("min_gain", "regression_l2", {"min_gain_to_split": 10000.0}, values),
        ("two_bins", "regression_l2", {"max_bin": 2}, values),
        ("constant", "regression_l2", {}, [1.0] * 8),
        ("all_missing", "regression_l2", {}, [None] * 8),
        ("single_class", "binary_logloss", {}, values),
        ("unseen_nan", "regression_l2", {}, values[:-1]),
    ]:
        x = [v for v in column for _ in range(6)]
        y = [float(i//6 >= 4) if objective == "binary_logloss" else float((i//6)**2 - 5) for i in range(len(x))]
        if name == "single_class":
            y = [1.0] * len(x)
        config = {**base, "objective": objective, **overrides}
        prediction_x = x + [None, -10.0, 10.0, 0.0]
        result = reference.train(x, y, 1, config, prediction_x)
        result.pop("timings")
        input_text = f"{len(x)} {config['max_bin']}\n" + " ".join("nan" if v is None else repr(v) for v in x)
        bins = json.loads(subprocess.check_output([str(probe)], input=input_text, text=True))
        initial = sum(y)/len(y)
        if objective == "binary_logloss":
            probability = min(1 - f32(1e-15), max(f32(1e-15), initial))
            initial = math.log(probability/(1-probability))
        gradient = [f32(initial-v) for v in y] if objective == "regression_l2" else [f32((1/(1+math.exp(-initial)))-v) for v in y]
        hessian = [1.0] * len(y) if objective == "regression_l2" else [f32((1/(1+math.exp(-initial)))*(1-(1/(1+math.exp(-initial)))))] * len(y)
        hashes[name] = freeze(name, {"features": x, "labels": y, "cols": 1, "config": config,
                                    "prediction_features": prediction_x, "binning": bins,
                                    "initial_score": initial, "gradients": gradient, "hessians": hessian, **result})
    for objective in ("regression_l2", "binary_logloss"):
        x, y = quality_data(objective)
        train_rows, cols = 2048, 12
        config = {"objective": objective, "iterations": 100, "max_bin": 63}
        result = reference.train(x[:train_rows*cols], y[:train_rows], cols, config, x[train_rows*cols:], dump=False)
        result.pop("timings")
        hashes[f"quality_{objective}"] = freeze(f"quality_{objective}", {"features": x, "labels": y,
            "cols": cols, "train_rows": train_rows, "config": config,
            "quality": metrics(objective, y[train_rows:], result["raw_predictions"]), **result})
    manifest = {"source_tree": pin["source_tree"], "version": pin["version"],
                "reference_provenance": reference.provenance, "fixture_sha256": hashes,
                "generator_python": sys.version, "generator_platform": platform.platform(),
                "generator_sha256": {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                    ("scripts/generate_fixtures.py", "scripts/reference.py", "validation/bin_probe.cpp")},
                "acceptance": {"cpu_raw_atol": 1e-8, "cpu_raw_rtol": 1e-6, "metal_raw_atol": 1e-5,
                    "metal_raw_rtol": 1e-4, "rmse_ratio": 1.02, "rmse_atol": 1e-6, "logloss_atol": 0.01}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(hashes)} fixtures from {pin['source_tree']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", required=True,
                        help="Explicit acknowledgement that frozen evidence will be replaced")
    parser.parse_args()
    generate()
