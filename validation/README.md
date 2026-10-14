# Reference evidence

`reference.json` pins the bundled source tree and the external header dependencies missing from the vendored copy. Dependency revisions were resolved from upstream commit `5831ee577776949d26d21efa581ec904bda80483`; the compiled engine source is always the pinned bundled tree, not that upstream checkout. Builds and downloaded dependencies live in ignored `.build/reference/`; the shipped Rust package never loads them.

The generator uses Python's standard library and the compiled reference C API through `ctypes`. It does not import an installed Python LightGBM package, NumPy, or SciPy. Test-only `bin_probe.cpp` queries the upstream `BinMapper` directly without modifying reference code.

```sh
python3 scripts/build_reference.py --check
python3 scripts/build_reference.py --fetch
python3 scripts/generate_fixtures.py --regenerate
python3 scripts/verify_fixtures.py
```

The first build requires Git, CMake, a C++17 compiler, and OpenMP (on this Mac, Homebrew libomp). Subsequent builds can omit `--fetch`. The build provenance records source/dependency identities, library hash, Python, platform, and CMake. Fixture manifests record data/output hashes, generator hashes/environment, configurations, and the declared acceptance thresholds. `--expected-tree` on the identity check exists to verify that a wrong source pin is rejected.

Regeneration is an explicit maintainer action and replaces frozen evidence. Normal tests and fixture verification never regenerate data to repair a failure. Compare old/new hashes when intentionally updating evidence. The two quality cases use a deterministic xorshift generator, 2,560 rows and 12 features, the first 2,048 rows for training, and the remaining 512 for validation. Targets are represented as float32 before comparison, matching upstream's label storage.

Small unambiguous fixtures check intermediate bins/objective values and raw predictions. CPU raw-score tolerance is `1e-8 + 1e-6 * abs(reference)`; Metal is `1e-5 + 1e-4 * abs(reference)`. Metal histogram counts must equal CPU counts exactly. Gradient and Hessian sum errors must each be <= `1e-5 + 1e-4 * sum(abs(contributing values))` for the corresponding bin, including cancelling gradients and empty/missing bins; nonfinite results fail. Frozen quality allows regression RMSE <= `1.02 * baseline + 1e-6` and binary mean logloss <= `baseline + 0.01`, checked against upstream and, for Metal, Rust CPU. Identical trees across backends are not promised. Hardware skips are not acceptance evidence.
