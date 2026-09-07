# Benchmarks: iron-batch vs vLLM

These are real, measured numbers from iron-batch running against an
actual model, not the synthetic `MockBackend`. All commands below are
exactly what was run to produce these numbers — run them yourself against
your own Modal deployment to reproduce.

**Terminology note:** `vllm_app.py` pins no vLLM version, so the vLLM
instance actually benchmarked here runs whatever was current at deploy
time — which is **Model Runner V2 (MRv2)**, not PagedAttention. vLLM
formally removed PagedAttention as its attention/execution path; MRv2
(announced March 2026) replaced it, decoupling CPU scheduling from GPU
execution (the two now pipeline instead of running sequentially) and
using Triton-compiled kernels instead of Python-dispatched ones. The
underlying idea of block-based, non-contiguous KV memory isn't gone —
vLLM's KV Cache Manager still does that — but it's a different
implementation than the original PagedAttention paper this repo's own
allocator (`src/kv_cache.rs`) was modeled on. Comparisons below are
against **vLLM/MRv2 as actually deployed**, not the older PagedAttention
architecture some earlier drafts of this doc and the accompanying blog
posts incorrectly named.

## Setup

- Model: Qwen/Qwen2.5-3B-Instruct, float16 (both engines, matched precision)
- GPU: Nvidia A10 (Modal), both engines
- vLLM version: latest at deploy time (Model Runner V2 execution path)
- Test: single request, concurrency 1, 64 new tokens, warm container
  (cold-start runs excluded — see raw output below, both cold and warm
  runs are shown)
- Prompt: 3 tokens (minimal prompt — see limitation note below)

**Known limitation, stated plainly:** concurrency 1 tests single-sequence
latency only. It does not exercise continuous batching across multiple
concurrent sequences, which is what both engines are actually built to
optimize — vLLM/MRv2's async, pipelined scheduling and iron-batch's own
scheduler. At concurrency 1, neither engine's core differentiator is
being tested — this measures raw single-sequence decode speed with KV
cache reuse, nothing more. A concurrency > 1 benchmark is the correct
next test and has not been run yet.

## Results (A10, warm container)

| Engine           | Tokens | Time  | Throughput  |
|------------------|--------|-------|-------------|
| vLLM (MRv2)      | 64     | 2.90s | 22.08 tok/s |
| iron-batch       | 64     | 2.98s | 21.5 tok/s  |

At concurrency 1, iron-batch is within ~3% of vLLM/MRv2's throughput on
the same GPU. This is not a general claim that iron-batch matches vLLM —
see the limitation above.

## Raw command output

**iron-batch:**
```
> .\target\release\bench_client.exe --concurrency 1 --total-requests 1 --prompt-tokens 3 --max-new-tokens 64
requests=1 failed=0 tokens=64 elapsed=53.04s throughput=1.2 tok/s     <- cold start
requests=1 failed=0 tokens=64 elapsed=2.98s throughput=21.5 tok/s     <- warm
```

**vLLM:**
```
> python vllm_compare.py --vllm-url "https://sayakmondal56--vllm-compare-fastapi-app.modal.run" --tokens 64
[vLLM] 114.62s for 64 tokens -> 0.56 tok/s     <- cold start (full engine init on A10)
[vLLM] 2.90s for 64 tokens -> 22.08 tok/s      <- warm
```

## How to reproduce this yourself

1. Clone this repo and set up Modal (`pip install modal && modal setup`).
2. Deploy both backends:
   - `modal deploy app.py` (iron-batch's real backend)
   - `modal deploy vllm_app.py` (the vLLM/MRv2 comparison target)
3. Warmup the iron-batch endpoint after deploy to avoid cold-start on the
   first benchmark call: `curl https://<your-app-url>/warmup`
4. `cargo build --release`
5. Run the server:
   `.\target\release\server.exe --backend real --backend-url <your-app.py-url>`
6. Run the iron-batch benchmark twice (first call is cold-start, ignore
   it):
   `.\target\release\bench_client.exe --concurrency 1 --total-requests 1 --prompt-tokens 3 --max-new-tokens 64`
7. For the vLLM comparison, send a POST directly to the vLLM endpoint:
   ```
   curl -X POST https://<your-vllm-url>/generate \
     -H 'content-type: application/json' \
     -d '{"prompt": "Hello", "max_new_tokens": 64}'
   ```
   Time the response manually or wrap it in a script.

Numbers will vary run to run (Modal's GPU allocation, network conditions,
and whatever vLLM version is current when you deploy) but should land in
a similar range.

## Earlier results, for the full history (T4, 4-bit quantized iron-batch)

Before switching to A10 + float16, the same test on a T4 with iron-batch
running 4-bit quantized looked very different:

| Engine                              | Tokens | Time    | Throughput  | vs vLLM      |
|-------------------------------------|--------|---------|-------------|--------------|
| vLLM/MRv2 (T4, float16)             | 64     | 3.52s   | 18.19 tok/s | —            |
| iron-batch v1 (T4, 4-bit)           | 64     | 109.65s | 0.6 tok/s   | ~30x slower  |
| iron-batch v2 (T4, 4-bit, batched)  | 64     | 8.96s   | 7.1 tok/s   | ~2.6x slower |

Two changes closed that gap: batching 16-32 tokens per network call
instead of 1 (cut ~64 round-trips to ~2-4), and moving to an A10 with
matched float16 precision on both sides (removed the quantization
mismatch and used a faster GPU tier).

## Why the earlier gap existed, and what actually closed it

v1's `RealBackend` made one HTTP round-trip to the model per token, and
the model-serving side recomputed the full forward pass from scratch on
every call with no KV cache reuse. v2 fixed both: batched calls, and
`model.generate()` with cache reuse. The remaining T4 gap (v2, ~2.6x) was
partly the GPU tier itself and partly a quantization mismatch (iron-batch
running 4-bit against vLLM's float16) — both resolved by matching
hardware and precision for the A10 comparison above.

## What iron-batch is actually for

Not raw single-sequence throughput — vLLM/MRv2 already matches or beats
that. The design bet is hardware portability: the scheduler and allocator
(`src/scheduler.rs`, `src/kv_cache.rs`) don't assume CUDA or any specific
accelerator, and don't assume any particular execution engine on the
other side of the `Backend` trait — a mock, an HTTP call to any model
server, or eventually a real device driver. That trade matters if you
need to target non-standard hardware; on standard GPUs, vLLM (whichever
execution engine it's currently running — PagedAttention historically,
MRv2 now, likely something else in a year) remains the more mature,
more battle-tested choice, especially at the concurrent-request scale
this benchmark hasn't tested yet.
