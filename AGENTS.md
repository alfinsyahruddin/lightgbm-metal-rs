# Guidance for coding agents

## Project context

lightgbm-metal-rs is an unofficial, community-maintained open-source implementation of a focused LightGBM training subset in Rust, with optional Apple Silicon Metal histograms. It is independently maintained and is not affiliated with or endorsed by upstream LightGBM. Preserve that distinction in documentation and public-facing descriptions.

Read [README.md](README.md), the affected module, and relevant docs before making changes. Public behavior is described by README, Rust API documentation, and `docs/`; reference thresholds and provenance are described in [validation/README.md](validation/README.md). Keep those sources consistent with the implementation.

## Repository map

- `src/`: public API, input/configuration validation, binning, objectives, histograms, training, and native models.
- `src/histogram/metal.rs` and `kernels/histogram.metal`: native Metal bindings and GPU kernels.
- `examples/`: executable usage and benchmark programs.
- `tests/reference.rs` and `validation/fixtures/`: frozen differential and quality checks.
- `scripts/` and `validation/`: reference build/generation, validation, and benchmark tooling.
- `docs/`: model format, Metal execution, and measured performance.
- `lightgbm-py/`: pinned upstream reference source. Keep it unchanged unless a reference update is explicitly part of the task.
- `.build/` and `target/`: generated local build/data/model artifacts; do not commit them.

## Working conventions

Inspect `git status` and the relevant diff first. Preserve unrelated and uncommitted work; make focused changes rather than repository-wide cleanup. Resolve routine implementation choices using the existing code and task context. Ask when an API contract, scope, or user-facing behavior is materially ambiguous.

Discuss substantial design changes in an issue or the task before implementation. Update affected docs/examples alongside behavior changes. Do not commit, push, publish, tag releases, or send external messages without explicit authorization.

## Rust and Metal conventions

- Use idiomatic ownership and borrowing, small cohesive modules, and typed `Result` errors. Avoid panics and `unwrap`/`expect` in fallible production paths; assertions in tests and examples are acceptable.
- Document public APIs, units, defaults, and invalid domains. Keep `#![deny(missing_docs)]` and `unsafe_op_in_unsafe_fn` checks intact.
- Use checked arithmetic for dimensions, indices, and allocation sizes. Validate before allocating large buffers or starting backend work.
- Keep Metal code and dependencies platform/feature gated. CPU builds and model inference must remain usable without Metal frameworks, Python, or the C++ LightGBM runtime.
- Keep unsafe operations narrowly scoped with comments explaining validity, ownership, lifetime, synchronization, and bounds.
- Retain Metal buffer ownership through command completion. Check completion and execution errors before reading or reusing outputs. A failed backend must not expose a partially successful model or silently fall back to CPU.
- Keep histogram counts exact. Detect nonfinite results and numerical overflow. Preserve learned missing branches, stable objective calculations, deterministic CPU tie-breaking, and validated versioned model loading.
- Account for all simultaneously live host/GPU histogram buffers within the configured workspace cap. Document dataset/model/framework storage separately.
- Preserve upstream MIT notices in translated code. Avoid adding dependencies without a concrete need; review license, platform support, Rust minimum, and package contents, and update the lockfile intentionally.

## Verification

Run checks proportional to the change. Documentation-only edits need link/content checks and `git diff --check`; run any newly documented executable example. Behavior changes need focused regression tests and these portable checks:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --locked --no-default-features
```

For public API documentation changes, also build docs with `cargo doc --locked --all-features --no-deps`. For changes to Cargo metadata, audit `cargo package --list --allow-dirty --offline` and ensure the runtime package excludes vendor/build/cache/result artifacts.

For Metal execution, synchronization, numerical, or integration changes, run on Apple Silicon with a usable GPU:

```sh
python3 scripts/validate.py --require-metal
```

That command runs fixture verification, formatting, Clippy, CPU/Metal-feature tests, ignored hardware tests, CPU-only inference from a Metal-trained model, and the Metal example. Python's standard library is sufficient for this suite. Report unsupported/unavailable hardware as unverified, not passed. Do not treat injected failure tests as proof of successful real GPU execution.

## Reference and benchmark integrity

Run `python3 scripts/verify_fixtures.py` when changing reference tooling or consuming updated fixtures. Fixtures must come from the pinned bundled source, not an arbitrary installed LightGBM package. Fixture generation is an explicit maintainer action documented in `validation/README.md`; never regenerate expected results or loosen tolerances just to make a failure pass. A reference or acceptance change must explain the reason and show old/new evidence.

Preserve the distinction between exact counts, numerical tolerances, model quality, and tree identity. Same-build CPU repeatability and exact native model round trips are separate from cross-backend numerical agreement.

For performance changes, follow `docs/benchmarks.md`: matched data/configuration, explicit CPU thread counts, cold/warm policies, repeated complete-training measurements, validation quality, hardware/software provenance, and all slower cases. Do not present kernel-only timing as training speedup. Keep generated reference dependencies and raw benchmark inputs in ignored `.build/`. Update measured reports only after executing the relevant workload; retain its provenance and avoid running competing workloads during timing.

## Handoff

Summarize what changed, why, and the exact checks performed. State material limitations or hardware checks that remain unverified. Keep pull requests reviewable and update README/API docs when supported behavior or setup changes. Use the MIT license with existing upstream attribution retained.
