# Contributing

## Setup

```bash
git clone https://github.com/Dev-X25874/iron-batch
cd iron-batch
cargo build
cargo test
```

Needs a real network for `cargo build` the first time (this crate has
real dependencies, unlike some others in this account) — see
`Cargo.toml` for pinned versions.

## Running it locally

```bash
cargo build --release
./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

Default backend is `MockBackend` — no GPU or real model needed to run
the server or the test suite.

## Before submitting a PR

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

CI runs the same three checks and will fail on any formatting diff,
clippy warning, or test failure.

## Benchmarks

`cargo bench` runs the Criterion scheduler benchmark (no HTTP, no
mock-sleep overhead) — this is separate from the real vLLM comparison
numbers in `BENCHMARKS.md`, which require a live Modal deployment and
are not part of CI. If you change anything in `src/scheduler.rs` or
`src/kv_cache.rs`, re-run `cargo bench` and note any meaningful
regression in your PR.

## Scope

The core scheduler/allocator logic (`src/scheduler.rs`,
`src/kv_cache.rs`) is the part that actually needs to be correct —
changes there need matching test coverage in `tests/integration.rs`,
specifically for block-leak, double-free, and preemption/resume
correctness (see `AGENTS.md` for what the existing tests check).
