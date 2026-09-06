# Iron Batch

Rust LLM inference server built around **continuous batching** and **paged KV cache allocation** — the two things that actually move the needle on throughput. Pluggable backend, streaming HTTP, no real weights or hardware needed to run.

## Performance

> numbers are from MockBackend (200µs/token). real backend will obviously be different depending on hardware — see [BENCHMARKS.md](BENCHMARKS.md)

| Concurrency | Requests | Tokens Generated | tok/s | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|
| 128 | 2000 | ~48k | ~3900 | 14ms | 61ms |

## Build and run

Tested on rustc 1.75, should be fine on newer.

```bash
cargo build --release
cargo test
cargo bench

./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

## HTTP API

### POST /generate

streams one JSON line per token

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

`max_new_tokens: 0` still returns a single done:true line, not an empty body.

### GET /metrics

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

tok/s and TTFT are reported together on purpose — a scheduler that maximizes throughput by starving admission will tank TTFT, so you need both numbers at once.

## CLI flags

```
--addr                    0.0.0.0:8080
--total-kv-blocks         4096
--kv-block-size           16
--max-batch-tokens        8192
--max-running-seqs        256
--mock-token-latency-us   200
--backend                 mock  # or "real" (requires --backend-url)
--backend-url             ""
```

## Testing and benchmarking

```bash
cargo test
cargo bench    # scheduler_step group, no HTTP or mock sleep overhead
```

test suite covers the stuff that actually matters — no block leaks across admit/evict/preempt cycles, no double-free, no under-allocation when a preempted sequence resumes.

## Extending

three places to plug in, nothing else in the codebase changes:

1. **Backend** — implement `Backend` for your runtime (cuda kernel, model API, whatever)
2. **Driver** — implement `DeviceHandle` with `nix::ioctl_*!` macros against a real device node. `src/driver/mod.rs` has the shape
3. **KV memory** — `BlockAllocator` tracks block IDs only, not backing storage. wire those to real HBM offsets

## what it doesnt do

- mock by default, so tok/s from the bench client is mock latency not real inference. opt in with `--backend real --backend-url <url>`
- no auth, no multi-tenancy, no persistence — single process
- not benchmarked against vllm/sglang/tgi, not the point
- no kernel driver, `driver/mod.rs` is userspace only

## License

MIT
