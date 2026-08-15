//! Compute backend abstraction. The scheduler doesn't know or care whether
//! a decode step runs on a mocked CPU loop (this file), a CUDA/vLLM
//! reference target, or Fractile's chip via `driver::DeviceHandle`. Swap
//! `MockBackend` for a real implementation of `Backend` behind the same
//! `advance_token` call and nothing else in the crate changes.

use crate::kv_cache::SeqId;
use std::collections::HashMap;
use std::time::Duration;

pub trait Backend: Send + Sync {
    /// Advance one sequence by one decode step. Returns true on EOS.
    fn advance_token(&self, seq_id: &SeqId) -> bool;
}

/// CPU-only stand-in that simulates per-token latency and a geometric EOS
/// distribution, so the scheduler/server/benchmark can be exercised end to
/// end without real weights or hardware. This is the seam a hardware
/// engineer swaps for a call into the actual inference kernel.
pub struct MockBackend {
    per_token_latency: Duration,
    eos_prob: f64,
    rng_state: parking_lot::Mutex<HashMap<SeqId, u64>>,
}

impl MockBackend {
    pub fn new(per_token_latency: Duration, eos_prob: f64) -> Self {
        Self {
            per_token_latency,
            eos_prob,
            rng_state: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    fn next_rand(&self, seq_id: &SeqId) -> f64 {
        // xorshift64 per-seq stream — deterministic, no external RNG dep on
        // the hot path, good enough for a synthetic backend.
        let mut guard = self.rng_state.lock();
        let seed = *seq_id ^ 0x9E3779B97F4A7C15;
        let state = guard.entry(*seq_id).or_insert(seed);
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as f64) / (u64::MAX as f64)
    }
}

impl Backend for MockBackend {
    fn advance_token(&self, seq_id: &SeqId) -> bool {
        if !self.per_token_latency.is_zero() {
            std::thread::sleep(self.per_token_latency);
        }
        self.next_rand(seq_id) < self.eos_prob
    }
}
