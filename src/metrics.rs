//! Minimal throughput/latency tracker. Reports aggregate decode tokens/sec
//! across the running batch, plus time-to-first-token (TTFT) as a
//! latency-sensitivity signal — a scheduler that maximizes raw tok/s by
//! starving admission will blow up TTFT, so both need to be visible together.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How many TTFT samples to keep. Older entries are evicted once we hit
/// this cap, keeping memory bounded while still giving statistically
/// stable percentiles for high-traffic servers.
const TTFT_WINDOW: usize = 10_000;

/// Rolling window for tok/s — tracks token counts over the last N seconds
/// so the reported rate reflects recent throughput instead of a lifetime
/// average that gets diluted by idle periods.
const ROLLING_WINDOW_SECS: u64 = 60;

#[derive(Default)]
struct Inner {
    tokens_generated: u64,
    requests_completed: u64,
    /// Bounded ring of TTFT samples. Capped at TTFT_WINDOW entries so
    /// memory stays flat regardless of how long the server runs.
    ttft_samples: VecDeque<Duration>,
    start: Option<Instant>,
    /// (timestamp, token_count) pairs for the rolling tok/s window.
    token_log: VecDeque<(Instant, u64)>,
}

pub struct Metrics {
    inner: Mutex<Inner>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self { inner: Mutex::new(Inner::default()) }
    }

    pub fn mark_start(&self) {
        let mut g = self.inner.lock();
        if g.start.is_none() {
            g.start = Some(Instant::now());
        }
    }

    pub fn record_tokens(&self, n: u64) {
        let mut g = self.inner.lock();
        g.tokens_generated += n;
        let now = Instant::now();
        g.token_log.push_back((now, n));
        // evict entries older than the rolling window
        let cutoff = now - Duration::from_secs(ROLLING_WINDOW_SECS);
        while g.token_log.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            g.token_log.pop_front();
        }
    }

    pub fn record_completion(&self) {
        self.inner.lock().requests_completed += 1;
    }

    pub fn record_ttft(&self, d: Duration) {
        let mut g = self.inner.lock();
        if g.ttft_samples.len() >= TTFT_WINDOW {
            g.ttft_samples.pop_front();
        }
        g.ttft_samples.push_back(d);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let g = self.inner.lock();
        let elapsed = g.start.map(|s| s.elapsed()).unwrap_or_default();

        let tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
            g.tokens_generated as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Rolling tok/s: sum tokens in the window, divide by window duration.
        let tokens_per_sec_rolling = if g.token_log.len() >= 2 {
            let window_tokens: u64 = g.token_log.iter().map(|(_, n)| n).sum();
            let window_secs = g
                .token_log
                .back()
                .unwrap()
                .0
                .duration_since(g.token_log.front().unwrap().0)
                .as_secs_f64();
            if window_secs > 0.0 {
                window_tokens as f64 / window_secs
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Sort a clone for percentiles. With TTFT_WINDOW capped at 10k
        // this is a bounded-cost operation regardless of server uptime.
        let mut sorted: Vec<Duration> = g.ttft_samples.iter().cloned().collect();
        sorted.sort();
        let p50 = percentile(&sorted, 0.50);
        let p99 = percentile(&sorted, 0.99);

        MetricsSnapshot {
            elapsed,
            tokens_generated: g.tokens_generated,
            requests_completed: g.requests_completed,
            tokens_per_sec,
            tokens_per_sec_rolling,
            ttft_p50: p50,
            ttft_p99: p99,
        }
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

#[derive(Debug)]
pub struct MetricsSnapshot {
    pub elapsed: Duration,
    pub tokens_generated: u64,
    pub requests_completed: u64,
    /// Lifetime aggregate tok/s.
    pub tokens_per_sec: f64,
    /// Rolling tok/s over the last 60 seconds — more useful than the
    /// lifetime aggregate when the server has been idle for any stretch.
    pub tokens_per_sec_rolling: f64,
    pub ttft_p50: Duration,
    pub ttft_p99: Duration,
}
