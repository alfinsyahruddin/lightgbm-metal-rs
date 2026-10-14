#!/usr/bin/env python3
"""Verify frozen reference evidence; never regenerate expected outputs."""
import hashlib
import json
from pathlib import Path

from build_reference import ROOT, source_identity


def verify():
    source_identity()
    folder = ROOT / "validation/fixtures"
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["fixture_sha256"].items():
        actual = hashlib.sha256((folder / f"{name}.json").read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen fixture changed: {name}")
    for name, expected in manifest["generator_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"fixture generator changed: {name}")
    print(f"Verified {len(manifest['fixture_sha256'])} frozen fixtures")


if __name__ == "__main__":
    verify()
