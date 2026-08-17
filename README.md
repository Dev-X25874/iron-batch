# Iron Batch

> Iron-fast LLM inference scheduling in Rust — continuous batching, paged KV allocation, streaming HTTP, no model required.

---

## What it is

A reference-grade LLM inference server written in Rust. It implements the two things that actually determine inference throughput — a **continuous batching scheduler** and a **paged KV cache allocator** — wired to a pluggable backend trait and served over HTTP with streaming responses. The compute backend is a mock by default, so the entire scheduling and allocation stack runs and benchmarks end-to-end without real weights or hardware.

---

## What's inside

| File | What it does |
|------|-------------|
| `kv_cache.rs` | Block-based paged KV allocator. O(1) alloc/free via free-list stack. Refcounted blocks so a shared prefix (system prompt, few-shot examples) is reused across sequences without copying. `fork_seq` gives you cheap beam search. |
| `scheduler.rs` | Continuous batching à la Orca/vLLM. Every decode step: admit new sequences, advance the whole running batch by one token, evict finished ones. Token budget (`max_batch_tokens`) and sequence cap (`max_running_seqs`) are the two knobs. FCFS with no starvation — a request that doesn't fit stays at the head of the queue. |
| `backend.rs` | `Backend` trait: one method, `advance_token(seq_id) -> bool`. `MockBackend` simulates per-token latency (configurable) and geometric EOS. Swap it for a real kernel call and nothing else in the codebase changes. |
| `driver/mod.rs` | Userspace contract for talking to a hardware accelerator: submit-batch / wait-fence / query-free-blocks, shaped like a real Linux char-device ioctl surface. Backed by an in-process `FakeDevice` so it builds without hardware. The sketch for a real `/dev/fractile0` + `nix::ioctl_*!` path is in the file. |
| `server.rs` | Axum HTTP server. One background task owns the scheduler and runs the decode loop; request handlers just enqueue work and stream results back over unbounded channels. No scheduler lock contention on the decode hot path. Responses are newline-delimited JSON (`application/x-ndjson`). |
| `metrics.rs` | `GET /metrics` returns aggregate tokens/sec and TTFT p50/p99. Both are reported together deliberately — a scheduler that maximizes raw tok/s by starving admission blows TTFT, so you need both numbers in the same view. |

---

## What it does NOT do

- **No real model.** `MockBackend` is a synthetic stand-in. The tok/s numbers from the bench client reflect mock sleep latency, not actual inference speed.
- **No preemption.** `StepReport::preempted` is a stub. When `grow_seq` OOMs mid-decode, the block growth is silently skipped instead of evicting the lowest-priority sequence back to the wait queue.
- **No auth, multi-tenancy, or persistence.** Single-process reference implementation.
- **Not benchmarked against vLLM/SGLang/TGI.** Not that kind of project.
- **No kernel driver.** `driver/mod.rs` is the userspace side of the contract only. A real kernel module needs the out-of-tree Rust-for-Linux toolchain.

---

## Build & run

Pinned to the versions in `Cargo.lock` (tested on rustc 1.75). Bump freely on newer toolchains.

```bash
cargo build --release
cargo test
cargo bench                   # scheduler + allocator hot path, no network

./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

Flags:
```
--addr               0.0.0.0:8080
--total-kv-blocks    4096
--kv-block-size      16
--max-batch-tokens   8192
--max-running-seqs   256
--mock-token-latency-us  200
```

---

## Extending toward a real backend

1. **Backend**: Implement `Backend` against your inference runtime (CUDA kernel, model API, whatever). Nothing else changes.
2. **Driver**: Implement `DeviceHandle` against a real device node using `nix::ioctl_*!` macros. The `driver/mod.rs` sketch shows the exact shape.
3. **KV memory**: `BlockAllocator` tracks block identity only, not backing storage. Wire block IDs to real HBM offsets.
4. **Preemption**: When `grow_seq` returns `OutOfMemory` mid-decode, evict the lowest-priority running sequence back to `waiting` and retry instead of silently dropping the growth.

---

## Who this is for

If you want to read or stress-test a minimal, self-contained implementation of continuous batching + PagedAttention-style KV allocation without wading through a full production serving stack, this is it. Useful as a study reference, a starting skeleton, or a portfolio artifact showing you understand what actually happens inside vLLM.
