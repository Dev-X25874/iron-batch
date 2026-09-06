# Iron Batch

Rust LLM inference server built around **continuous batching** and **paged KV cache allocation** — the two things that actually move the needle on throughput. Pluggable backend, streaming HTTP, no real weights or hardware needed to run.

## Performance

Mock backend (200µs/token). Real backend numbers vary by hardware.

| Concurrency | Requests | Tokens Generated | Tok/s | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|
| 128 | 2000 | ~48k | ~3900 | 14ms | 61ms |

## Build and run

```bash
cargo build --release
cargo test
cargo bench

./target/release/server &
./target/release/bench_client --concurrency 128 --total-requests 2000
curl localhost:8080/metrics
```

## HTTP API

### `POST /generate`

Streams one JSON line per token.

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
--backend                 mock  # or "real" (requires --backend-url)
--backend-url             ""
```

## Testing and benchmarking

```bash
cargo test              # unit + integration
cargo bench             # scheduler_step group, isolated from HTTP/mock sleep
```

Covers: no block leaks across admit/evict/preempt cycles, no double-free on duplicate IDs, no leak on failed allocation, correct resume allocation for preempted sequences.

## Extending toward a real backend

Three connection points, nothing else changes:

1. **Backend** — implement `Backend` for your runtime (CUDA kernel, model API, whatever).
2. **Driver** — implement `DeviceHandle` with `nix::ioctl_*!` macros against a real device node.
3. **KV memory** — `BlockAllocator` tracks block IDs only; wire them to real HBM offsets.

## What it doesn't do

- Mock by default — tok/s reflects mock sleep latency. Opt into real backend with `--backend real --backend-url <url>`.
- No auth, multi-tenancy, or persistence.
- Not benchmarked against vLLM / SGLang / TGI.
- No kernel driver — `driver/mod.rs` is userspace only.

## License

MIT. See [LICENSE](LICENSE).
