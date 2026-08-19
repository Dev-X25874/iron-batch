# Iron Batch

A reference-grade LLM inference server written in Rust. It implements the two things that actually determine inference throughput — a **continuous batching scheduler** and a **paged KV cache allocator** — wired to a pluggable compute backend and served over HTTP with streaming responses. The compute backend is a mock by default, so the entire scheduling and allocation stack runs and benchmarks end-to-end without real weights or hardware.

## Contents

- [What it is](#what-it-is)
- [Architecture](#architecture)
- [Build and run](#build-and-run)
- [HTTP API](#http-api)
- [CLI flags](#cli-flags)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Extending toward a real backend](#extending-toward-a-real-backend)
- [What it does not do](#what-it-does-not-do)
- [License](#license)

## What it is

Two ideas make LLM serving fast: batch requests together so the accelerator is never idle, and share KV cache memory across requests instead of copying it. Iron Batch implements both, minimally and readably enough to be a study reference rather than a black box.

- **Continuous batching** (Orca / vLLM style): every decode step, finished sequences are evicted and newly-arrived requests are admitted into the running batch, so utilization never drains to zero waiting for the slowest sequence in a static batch to finish.
- **Paged KV allocation** (PagedAttention style): a fixed pool of physical blocks is handed out by page table per sequence, refcounted so a shared prefix (system prompt, few-shot examples) is reused without copying.

Preemption is implemented end to end: if a running sequence can't grow its KV allocation mid-decode, it's evicted back to the wait queue and its blocks are freed rather than letting it generate tokens with nowhere to store them. On resume, it's re-admitted with blocks reserved for its full context so far (prompt plus everything already generated), not just its original prompt.

## Architecture

| File | What it does |
|---|---|
| `src/kv_cache.rs` | Block-based paged KV allocator. O(1) alloc/free via a free-list stack. Refcounted blocks so a shared prefix is reused across sequences without copying. `fork_seq` gives you cheap beam search. |
| `src/scheduler.rs` | Continuous batching. Every decode step: admit new sequences, advance the running batch by one token, evict finished ones, preempt anything that OOMs mid-grow. Token budget (`max_batch_tokens`) and sequence cap (`max_running_seqs`) are the two knobs. FCFS with no starvation — a request that doesn't fit stays at the head of the queue. |
| `src/backend.rs` | `Backend` trait: one method, `advance_token(seq_id) -> bool`. `MockBackend` simulates per-token latency and a geometric EOS distribution. Swap it for a real kernel call and nothing else in the codebase changes. |
| `src/driver/mod.rs` | Userspace contract for talking to a hardware accelerator: submit-batch / wait-fence / query-free-blocks, shaped like a real Linux char-device ioctl surface. Backed by an in-process `FakeDevice` so it builds without hardware. |
| `src/server.rs` | Axum HTTP server. One background task owns the scheduler and runs the decode loop; request handlers just enqueue work and stream results back over unbounded channels — no scheduler lock contention on the decode hot path. Responses are newline-delimited JSON (`application/x-ndjson`). |
| `src/metrics.rs` | `GET /metrics` returns aggregate tokens/sec and TTFT p50/p99. Both are reported together deliberately — a scheduler that maximizes raw tok/s by starving admission blows up TTFT, so you need both numbers in the same view. |
| `src/bin/bench_client.rs` | Closed-loop-free load generator: fires `--concurrency` requests immediately rather than waiting for each to finish, so it actually exercises continuous batching instead of measuring one sequence at a time. |

## Build and run

Pinned to the versions in `Cargo.lock` (tested on rustc 1.75). Bump freely on newer toolchains.

```bash
cargo build --release
cargo test
cargo bench                   # scheduler + allocator hot path, no network

./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

## HTTP API

### `POST /generate`

Streams newline-delimited JSON, one line per generated token.

```bash
curl -N -X POST localhost:8080/generate \
  -H 'content-type: application/json' \
  -d '{"prompt_tokens": 128, "max_new_tokens": 64}'
```

```json
{"token_index":0,"done":false}
{"token_index":1,"done":false}
{"token_index":2,"done":true}
```

A request with `max_new_tokens: 0` still returns a single `{"token_index":0,"done":true}` line rather than an empty body.

### `GET /metrics`

```json
{
  "elapsed_secs": 12.4,
  "tokens_generated": 48213,
  "requests_completed": 512,
  "tokens_per_sec": 3888.9,
  "ttft_p50_ms": 14,
  "ttft_p99_ms": 61
}
```

## CLI flags

```
--addr                    0.0.0.0:8080
--total-kv-blocks         4096
--kv-block-size           16
--max-batch-tokens        8192
--max-running-seqs        256
--mock-token-latency-us   200
```

## Testing and benchmarking

```bash
cargo test              # unit + integration tests
cargo bench              # scheduler_step benchmark group, isolated from HTTP/mock sleep
```

The test suite covers the correctness properties that matter most for a KV allocator and a continuous-batching scheduler: no block leaks across admit/evict/preempt cycles, no double-free on duplicate sequence IDs, no block leaked on a failed allocation, and no under-allocation when a preempted sequence resumes.

## Extending toward a real backend

1. **Backend**: implement `Backend` against your inference runtime (CUDA kernel, model API, whatever). Nothing else changes.
2. **Driver**: implement `DeviceHandle` against a real device node using `nix::ioctl_*!` macros. `src/driver/mod.rs` sketches the exact shape.
3. **KV memory**: `BlockAllocator` tracks block identity only, not backing storage. Wire block IDs to real HBM offsets.

## What it does not do

- **No real model.** `MockBackend` is a synthetic stand-in; tok/s numbers from the bench client reflect mock sleep latency, not actual inference speed.
- **No auth, multi-tenancy, or persistence.** Single-process reference implementation.
- **Not benchmarked against vLLM / SGLang / TGI.** Not that kind of project.
- **No kernel driver.** `driver/mod.rs` is the userspace side of the contract only; a real kernel module needs the out-of-tree Rust-for-Linux toolchain.

## License

MIT. See [LICENSE](LICENSE).
