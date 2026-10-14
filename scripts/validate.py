#!/usr/bin/env python3
"""Run portable and optional hardware validation, keeping explicit evidence and skip status."""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys

from build_reference import ROOT


def command(args, log):
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.append("$ " + " ".join(args) + "\n" + result.stdout)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args)


def validate(require_metal=False):
    log = []
    results = ROOT / "validation/results"
    results.mkdir(exist_ok=True)
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "machine": platform.machine(), "metal": "not_run"}
    try:
        command([sys.executable, "scripts/verify_fixtures.py"], log)
        command(["cargo", "fmt", "--check"], log)
        command(["cargo", "clippy", "--all-targets", "--all-features", "--locked", "--", "-D", "warnings"], log)
        command(["cargo", "test", "--locked", "--no-default-features"], log)
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            metadata["metal"] = "skipped_unsupported_platform"
            print("SKIPPED: real Apple Silicon Metal hardware checks")
            if require_metal:
                raise RuntimeError("required Metal hardware unavailable")
            return
        probe = subprocess.run(["cargo", "run", "--locked", "--features", "metal", "--example", "backend_probe", "--", "--metal"], cwd=ROOT, text=True, capture_output=True)
        log.append(probe.stdout + probe.stderr)
        if probe.returncode:
            if "BackendUnavailable" not in probe.stderr:
                raise RuntimeError(probe.stderr)
            metadata["metal"] = "skipped_unavailable_device"
            print("SKIPPED: no usable Metal device (sandbox access may be restricted)")
            if require_metal:
                raise RuntimeError("required Metal device unavailable")
            return
        metadata["device"] = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
        metadata["rust"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
        command(["cargo", "test", "--locked", "--features", "metal"], log)
        command(["cargo", "test", "--locked", "--features", "metal", "--", "--ignored", "--nocapture", "--test-threads=1"], log)
        command(["cargo", "run", "--locked", "--no-default-features", "--example", "load_metal_model"], log)
        command(["cargo", "run", "--locked", "--features", "metal", "--example", "train", "--", "--metal"], log)
        metadata["metal"] = "passed_on_real_gpu"
    except Exception:
        metadata["status"] = "failed"
        raise
    finally:
        metadata.setdefault("status", "passed" if metadata["metal"] == "passed_on_real_gpu" else "portable_passed_hardware_unverified")
        (results / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (results / "validation.log").write_text("\n".join(log))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-metal", action="store_true")
    validate(parser.parse_args().require_metal)
