# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- `AGENTS.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `rustfmt.toml`, and a
  GitHub Actions CI workflow (fmt, clippy, build, test, server smoke test).

### Known gaps
- `README.md` documents `--backend` / `--backend-url` CLI flags that
  are not yet implemented in `src/main.rs`'s `Args` struct — only the
  mock backend is currently wired into the CLI. Needs either the flag
  implemented or the README corrected.

## [0.1.0]

Initial version: continuous batching scheduler, paged KV cache block
allocator, Axum HTTP server (`/generate`, `/metrics`), mock backend,
Criterion benchmarks, and real (Modal-deployed) throughput comparison
against vLLM in `BENCHMARKS.md`.
