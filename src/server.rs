//! HTTP front end. One background task owns the `Scheduler` and drives it
//! in a tight decode loop; per-request handlers only enqueue work and read
//! from a channel that the loop pushes generated tokens into. This keeps
//! the scheduler single-threaded (no lock contention on the hot path) while
//! still serving arbitrarily many concurrent HTTP requests.

use crate::backend::Backend;
use crate::kv_cache::{BlockAllocator, SeqId};
use crate::metrics::Metrics;
use crate::scheduler::{Request as SchedRequest, Scheduler, SchedulerConfig};
use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex as AsyncMutex};

#[derive(Deserialize)]
pub struct GenerateReq {
    pub prompt_tokens: u32,
    pub max_new_tokens: u32,
}

#[derive(Serialize)]
struct TokenEvent {
    token_index: u32,
    done: bool,
}

pub struct AppState {
    next_seq_id: AtomicU64,
    enqueue_tx: mpsc::UnboundedSender<SchedRequest>,
    pub metrics: Arc<Metrics>,
}

pub fn build_router(
    backend: Arc<dyn Backend>,
    total_blocks: u32,
    block_size: u32,
    cfg: SchedulerConfig,
) -> Router {
    let metrics = Arc::new(Metrics::new());
    let allocator = BlockAllocator::new(total_blocks, block_size);
    let scheduler = Scheduler::new(cfg, allocator);

    let (enqueue_tx, mut enqueue_rx) = mpsc::unbounded_channel::<SchedRequest>();
    let subscribers: Arc<AsyncMutex<HashMap<SeqId, mpsc::UnboundedSender<TokenEvent>>>> =
        Arc::new(AsyncMutex::new(HashMap::new()));

    let state = Arc::new(AppState {
        next_seq_id: AtomicU64::new(1),
        enqueue_tx,
        metrics: metrics.clone(),
    });

    // Background decode loop: single owner of the Scheduler, no cross-task
    // locking on the step() hot path itself.
    let subs_for_loop = subscribers.clone();
    let metrics_for_loop = metrics.clone();
    tokio::spawn(async move {
        let mut scheduler = scheduler;
        metrics_for_loop.mark_start();
        loop {
            // drain newly-arrived requests without blocking the step loop
            while let Ok(req) = enqueue_rx.try_recv() {
                scheduler.enqueue(req);
            }
            if !scheduler.has_work() {
                // idle: block on the next arrival instead of busy-spinning
                match enqueue_rx.recv().await {
                    Some(req) => scheduler.enqueue(req),
                    None => break, // all senders dropped, shut down
                }
                continue;
            }

            let backend = backend.clone();
            let report = scheduler.step(|seq_id| backend.advance_token(seq_id));

            metrics_for_loop.record_tokens(report.ran.len() as u64);
            let mut subs = subs_for_loop.lock().await;
            for seq_id in &report.ran {
                if let Some(tx) = subs.get(seq_id) {
                    let done = report.finished.contains(seq_id);
                    let _ = tx.send(TokenEvent {
                        token_index: 0,
                        done,
                    });
                }
            }
            for seq_id in &report.finished {
                subs.remove(seq_id);
                metrics_for_loop.record_completion();
            }
            drop(subs);

            // yield so this doesn't starve the async runtime's other tasks
            tokio::task::yield_now().await;
        }
    });

    Router::new()
        .route("/generate", post(generate))
        .route("/metrics", get(metrics_endpoint))
        .with_state((state, subscribers))
}

async fn generate(
    State((state, subscribers)): State<(
        Arc<AppState>,
        Arc<AsyncMutex<HashMap<SeqId, mpsc::UnboundedSender<TokenEvent>>>>,
    )>,
    Json(req): Json<GenerateReq>,
) -> Response {
    let seq_id = state.next_seq_id.fetch_add(1, Ordering::Relaxed);
    let (tx, mut rx) = mpsc::unbounded_channel::<TokenEvent>();
    subscribers.lock().await.insert(seq_id, tx);

    let start = Instant::now();
    let _ = state.enqueue_tx.send(SchedRequest {
        seq_id,
        prompt_tokens: req.prompt_tokens,
        max_new_tokens: req.max_new_tokens,
        generated: 0,
        arrival: start,
        first_token_at: None,
    });

    let metrics = state.metrics.clone();
    let mut first = true;
    let stream = async_stream::stream! {
        while let Some(ev) = rx.recv().await {
            if first {
                metrics.record_ttft(start.elapsed());
                first = false;
            }
            let line = serde_json::to_string(&ev).unwrap() + "\n";
            yield Ok::<Bytes, std::io::Error>(Bytes::from(line));
            if ev.done {
                break;
            }
        }
    };

    Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
        .into_response()
}

async fn metrics_endpoint(
    State((state, _)): State<(
        Arc<AppState>,
        Arc<AsyncMutex<HashMap<SeqId, mpsc::UnboundedSender<TokenEvent>>>>,
    )>,
) -> Json<serde_json::Value> {
    let snap = state.metrics.snapshot();
    Json(serde_json::json!({
        "elapsed_secs": snap.elapsed.as_secs_f64(),
        "tokens_generated": snap.tokens_generated,
        "requests_completed": snap.requests_completed,
        "tokens_per_sec": snap.tokens_per_sec,
        "ttft_p50_ms": snap.ttft_p50.as_millis(),
        "ttft_p99_ms": snap.ttft_p99.as_millis(),
    }))
}
