# AGENTS.md

Notes for AI coding assistants (Claude Code, Copilot, Cursor, etc.)
working in this repo.

## Build & test

```bash
cargo build --release
cargo test
cargo bench

./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

Pinned dependency versions in `Cargo.toml` (`tokio = "=1.38.1"`,
`axum = "=0.7.4"`, `clap = "=4.4.18"`, `reqwest = "=0.11.27"`) are
pinned deliberately — don't bump them without checking why they were
pinned first.

## Layout

- `src/scheduler.rs` — continuous batching scheduler: admission,
  preemption, eviction. This is the core logic; changes here need
  matching coverage in `tests/integration.rs`.
- `src/kv_cache.rs` — paged KV cache block allocator. Tracks block IDs
  only, not backing storage (see README "Extending" section).
- `src/backend.rs` — pluggable compute backend trait + `MockBackend`.
  The server runs standalone against the mock by default.
- `src/server.rs` — Axum HTTP layer (`/generate`, `/metrics`).
- `src/driver/mod.rs` — userspace-only device handle shape, no real
  kernel driver.
- `src/main.rs` — server binary entrypoint + CLI args.
- `src/bin/bench_client.rs` — load-generating HTTP client used for
  throughput testing.
- `benches/throughput.rs` — Criterion benchmark of the scheduler step
  in isolation, no HTTP/mock-sleep overhead.
- `app.py` / `vllm_app.py` — Modal deployments used to produce the
  real (non-mock) numbers in `BENCHMARKS.md`.

## Known gap

`README.md`'s CLI flags table lists `--backend` and `--backend-url`,
but `src/main.rs`'s `Args` struct doesn't currently define them — only
the mock backend path is wired into the CLI today. Don't assume this
flag exists when writing code or docs; either implement it or correct
the README, don't let the two continue disagreeing silently.

## Conventions

- No block leaks across admit/evict/preempt cycles, no double-free, no
  under-allocation when a preempted sequence resumes — this is what
  the existing test suite is actually checking; new scheduler/allocator
  changes should extend that coverage, not just add happy-path tests.
- `BENCHMARKS.md` numbers are real, measured against an actual deployed
  vLLM instance on Modal — see the methodology/limitation notes at the
  top of that file before adding or changing any numbers there. Don't
  add a benchmark number that wasn't actually measured and reproduced
  by the commands shown alongside it.
- `max_new_tokens: 0` must still return one `done:true` line, not an
  empty body — this is an explicit, tested contract of `/generate`.

## Before opening a PR

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

CI (`.github/workflows/ci.yml`) runs the same checks and fails the
build on any warning.
