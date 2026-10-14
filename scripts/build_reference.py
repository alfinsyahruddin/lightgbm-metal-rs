#!/usr/bin/env python3
"""Build the pinned bundled LightGBM source without modifying it (standard library only)."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/reference.json"
WORK = ROOT / ".build/reference"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def source_identity(expected=None):
    pin = json.loads(MANIFEST.read_text())
    actual = subprocess.check_output(["git", "rev-parse", "HEAD:lightgbm-py"], cwd=ROOT, text=True).strip()
    if actual != (expected or pin["source_tree"]):
        raise RuntimeError(f"reference identity mismatch: {actual}")
    run("git", "diff", "--quiet", "HEAD", "--", "lightgbm-py", cwd=ROOT)
    if (ROOT / "lightgbm-py/VERSION.txt").read_text().strip() != pin["version"]:
        raise RuntimeError("reference version mismatch")
    return pin


def build(fetch=False):
    pin = source_identity()
    source = WORK / "source/lightgbm-py"
    deps = WORK / "dependencies"
    deps.mkdir(parents=True, exist_ok=True)
    for name, dep in pin["dependencies"].items():
        checkout = deps / name
        if not checkout.exists():
            if not fetch:
                raise RuntimeError(f"missing {name}; run with --fetch to download pinned reference dependencies")
            checkout.mkdir()
            run("git", "init", "--quiet", str(checkout))
        existing = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, text=True, capture_output=True)
        if existing.returncode or existing.stdout.strip() != dep["commit"]:
            if not fetch:
                raise RuntimeError(f"wrong dependency revision: {name}")
            run("git", "fetch", "--depth", "1", dep["url"], dep["commit"], cwd=checkout)
            run("git", "checkout", "--detach", "FETCH_HEAD", cwd=checkout)
        run("git", "diff", "--quiet", "HEAD", cwd=checkout)
    if source.exists():
        shutil.rmtree(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", "HEAD", "lightgbm-py"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # Only the Git-created, pinned source archive is extracted into the isolated build.
        if sys.version_info >= (3, 12):
            tar.extractall(source.parent, filter="data")
        else:
            tar.extractall(source.parent)
    for name in pin["dependencies"]:
        shutil.copytree(deps / name, source / "external_libs" / name, symlinks=True,
                        ignore=shutil.ignore_patterns(".git"))
    build_dir = WORK / "build"
    arguments = ["cmake", "-S", str(source), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_CLI=OFF", "-DUSE_GPU=OFF", "-DUSE_CUDA=OFF"]
    if platform.system() == "Darwin" and Path("/opt/homebrew/opt/libomp").exists():
        arguments.append("-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/libomp")
    run(*arguments)
    run("cmake", "--build", str(build_dir), "--parallel", str(min(8, os.cpu_count() or 1)))
    candidates = list(source.glob("lib_lightgbm.*")) + list(build_dir.glob("lib_lightgbm.*"))
    libraries = [p for p in candidates if p.suffix in (".so", ".dylib", ".dll")]
    if len(libraries) != 1:
        raise RuntimeError(f"expected one reference library; found {libraries}")
    library = libraries[0].resolve()
    provenance = {**pin, "library": str(library.relative_to(ROOT)),
                  "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                  "python": sys.version, "platform": platform.platform(),
                  "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0]}
    (WORK / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Reference library: {library}")
    return library


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--expected-tree", help="Identity override for negative verification only")
    options = parser.parse_args()
    try:
        if options.check:
            print(source_identity(options.expected_tree)["source_tree"])
        else:
            build(options.fetch)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
