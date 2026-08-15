//! Minimal throughput/latency tracker. Reports the number Fractile's role
//! actually optimizes for: aggregate decode tokens/sec across the running
//! batch, plus time-to-first-token (TTFT) as a latency-sensitivity signal —
//! a scheduler that maximizes raw tok/s by starving admission will blow up
//! TTFT, so both need to be visible together.

use parking_lot::Mutex;
use std::time::{Duration, Instant};

#[derive(Default)]
struct Inner {
    tokens_generated: u64,
    requests_completed: u64,
    ttft_samples: Vec<Duration>,
    start: Option<Instant>,
}

pub struct Metrics {
    inner: Mutex<Inner>,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
        }
    }

    pub fn mark_start(&self) {
        let mut g = self.inner.lock();
        if g.start.is_none() {
            g.start = Some(Instant::now());
        }
    }

    pub fn record_tokens(&self, n: u64) {
        self.inner.lock().tokens_generated += n;
    }

    pub fn record_completion(&self) {
        self.inner.lock().requests_completed += 1;
    }

    pub fn record_ttft(&self, d: Duration) {
        self.inner.lock().ttft_samples.push(d);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let g = self.inner.lock();
        let elapsed = g.start.map(|s| s.elapsed()).unwrap_or_default();
        let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
            g.tokens_generated as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let mut sorted: Vec<Duration> = g.ttft_samples.clone();
        sorted.sort();
        let p50 = percentile(&sorted, 0.50);
        let p99 = percentile(&sorted, 0.99);
        MetricsSnapshot {
            elapsed,
            tokens_generated: g.tokens_generated,
            requests_completed: g.requests_completed,
            tokens_per_sec: tok_per_sec,
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
    pub tokens_per_sec: f64,
    pub ttft_p50: Duration,
    pub ttft_p99: Duration,
}
