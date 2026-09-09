//! Compute backend abstraction. The scheduler doesn't know or care whether
//! a decode step runs on a mocked CPU loop (this file), a CUDA/vLLM
//! reference target, or a real accelerator via `driver::DeviceHandle`. Swap
//! `MockBackend` for a real implementation of `Backend` behind the same
//! `advance_token` call and nothing else in the crate changes.

use crate::kv_cache::SeqId;
use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

pub trait Backend: Send + Sync {
    /// Advance one sequence by one decode step. Returns true on EOS.
    fn advance_token(&self, seq_id: &SeqId) -> bool;

    /// Called when a sequence finishes or is cancelled so the backend can
    /// clean up any per-sequence state it holds (RNG entries, token
    /// history buffers, etc.).
    fn free_seq(&self, seq_id: &SeqId);
}

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
            // tokio::time::sleep is not available here because advance_token
            // is a sync fn called from the scheduler. Use std::thread::sleep
            // only when latency is explicitly requested (benchmark / test use
            // Duration::ZERO to skip it). In production, wire a real async
            // backend instead.
            std::thread::sleep(self.per_token_latency);
        }
        self.next_rand(seq_id) < self.eos_prob
    }

    fn free_seq(&self, seq_id: &SeqId) {
        // Remove the RNG entry so the HashMap doesn't grow unboundedly
        // over the server's lifetime as millions of sequences complete.
        self.rng_state.lock().remove(seq_id);
    }
}

const BATCH_SIZE: usize = 16;

#[derive(Serialize)]
struct BatchRequest {
    token_ids: Vec<u32>,
    num_tokens: usize,
}

#[derive(Deserialize)]
struct BatchResponse {
    token_ids: Vec<u32>,
    eos: bool,
}

struct SeqState {
    /// Only the last HISTORY_CAP tokens are sent to the endpoint on each
    /// refill instead of the full unbounded history, so the HTTP payload
    /// doesn't grow linearly with sequence length.
    history: std::collections::VecDeque<u32>,
    pending: std::collections::VecDeque<u32>,
    eos_pending: bool,
}

/// Cap how many tokens we send to the model endpoint per refill call.
/// Sending the unbounded full history means each refill HTTP request grows
/// linearly with sequence length — at 1000 tokens that's 1000 IDs per call
/// just to get 16 more. A sliding window is sufficient for autoregressive
/// generation; adjust to match what your model endpoint actually needs.
const HISTORY_CAP: usize = 512;

/// Calls out to a real model served on Modal. Fetches BATCH_SIZE tokens
/// per HTTP call and serves them locally from a buffer to amortize
/// network round-trip cost.
pub struct RealBackend {
    endpoint: String,
    // Uses the blocking reqwest::Client. advance_token is a sync fn, so
    // callers must run it inside tokio::task::spawn_blocking or similar
    // to avoid blocking Tokio worker threads.
    client: reqwest::blocking::Client,
    seqs: parking_lot::Mutex<HashMap<SeqId, SeqState>>,
    default_prompt: Vec<u32>,
}

impl RealBackend {
    pub fn new(endpoint: impl Into<String>, default_prompt: Vec<u32>) -> Self {
        Self {
            endpoint: endpoint.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap(),
            seqs: parking_lot::Mutex::new(HashMap::new()),
            default_prompt,
        }
    }

    fn refill(&self, state: &mut SeqState) -> Result<bool, String> {
        let token_ids: Vec<u32> = state.history.iter().cloned().collect();
        let resp = self
            .client
            .post(format!("{}/generate_batch", self.endpoint))
            .json(&BatchRequest {
                token_ids,
                num_tokens: BATCH_SIZE,
            })
            .send()
            .map_err(|e| format!("RealBackend: request failed: {e}"))?
            .json::<BatchResponse>()
            .map_err(|e| format!("RealBackend: bad response: {e}"))?;

        for &t in &resp.token_ids {
            state.history.push_back(t);
            if state.history.len() > HISTORY_CAP {
                state.history.pop_front();
            }
            state.pending.push_back(t);
        }
        Ok(resp.eos)
    }
}

impl Backend for RealBackend {
    fn advance_token(&self, seq_id: &SeqId) -> bool {
        let mut guard = self.seqs.lock();
        let state = guard.entry(*seq_id).or_insert_with(|| SeqState {
            history: self.default_prompt.iter().cloned().collect(),
            pending: std::collections::VecDeque::new(),
            eos_pending: false,
        });

        if state.pending.is_empty() {
            if state.eos_pending {
                return true;
            }
            match self.refill(state) {
                Ok(hit_eos) => state.eos_pending = hit_eos,
                Err(e) => {
                    // Log the error and signal EOS so the sequence drains
                    // cleanly rather than panicking the whole server.
                    tracing::error!("{e}");
                    return true;
                }
            }
        }

        let is_last = state.pending.len() == 1;
        state.pending.pop_front();
        is_last && state.eos_pending
    }

    fn free_seq(&self, seq_id: &SeqId) {
        self.seqs.lock().remove(seq_id);
    }
}
