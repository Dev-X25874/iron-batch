# infer-server

A reference LLM inference server in Rust: continuous batching scheduler + paged KV cache allocator + a pluggable compute-backend trait.

## What it actually does

- **Paged KV cache** (`kv_cache.rs`) — block-based allocator (same idea as vLLM's PagedAttention): O(1) alloc/free, refcounted blocks so a shared prompt prefix can be reused across requests without copying.
- **Continuous batching scheduler** (`scheduler.rs`) — admits/evicts sequences every decode step instead of waiting for a whole static batch to finish, so throughput doesn't stall on the slowest sequence in the batch.
- **Pluggable backend** (`backend.rs`) — a `Backend` trait with a `MockBackend` behind it that simulates per-token latency and random EOS. There is no real model here. Point `Backend` at real weights/hardware and nothing else in the crate has to change.
- **Driver-shaped interface** (`driver/mod.rs`) — a userspace contract (submit-batch / wait-fence / query-free-blocks) shaped like what a real Linux char-device driver for an accelerator would expose, backed by an in-process fake so it builds without hardware.
- **Streaming HTTP server** (`server.rs`) — axum, one task owns the scheduler, responses stream as newline-delimited JSON.
- **Metrics** (`metrics.rs`) — aggregate tokens/sec plus TTFT p50/p99 on `GET /metrics`.

## What it does NOT do

- No real model, no real weights, no real accelerator. `MockBackend` is a stand-in so the scheduling/allocation logic is testable end to end.
- Not benchmarked against vLLM/SGLang/anything real — the tok/s numbers you'll see running the bench client are an artifact of the mock backend's synthetic sleep, not a claim about actual inference speed.
- No preemption when KV OOMs mid-decode (`StepReport::preempted` is a stub).
- No auth, no multi-tenancy, no persistence — single-process reference implementation only.

## Who this is for

People who want to read or exercise a small, working implementation of continuous batching + paged KV allocation without wading through a full production serving stack (vLLM, SGLang, TGI). Useful as a study reference, a portfolio artifact, or a starting skeleton to wire a real backend into.

## Build & run

Tested with `rustc`/`cargo` 1.75 (crate versions in `Cargo.lock` are pinned accordingly — bump freely on a newer toolchain):

```
cargo build --release
cargo test
cargo bench                # scheduler/allocator hot path only, no network
./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

## Extending toward a real backend

1. Implement `Backend` against real inference (kernel call / model runtime).
2. Implement `driver::DeviceHandle` against a real device node using `nix::ioctl_*!` macros (sketch included in `driver/mod.rs`).
3. `BlockAllocator` currently tracks block identity only, not backing storage — wire block ids to real memory offsets.
4. Add preemption: when `grow_seq` OOMs mid-decode, evict the lowest-priority running sequence back to `waiting` instead of dropping the growth silently.
