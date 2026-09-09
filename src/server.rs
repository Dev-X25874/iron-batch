//! HTTP front end. One background task owns the `Scheduler` and drives the
//! decode loop; request handlers just enqueue and read from a channel the
//! loop pushes into. Keeps the scheduler single-threaded (no lock on the
//! hot path) while still handling as many concurrent requests as you want.

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

const TOKEN_CHANNEL_CAPACITY: usize = 256;

/// Per-sequence sender used to stream token events out to the HTTP handler
/// awaiting them. Shared as Axum app state alongside `AppState`.
type Subscribers = Arc<AsyncMutex<HashMap<SeqId, mpsc::Sender<TokenEvent>>>>;

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
    let subscribers: Subscribers = Arc::new(AsyncMutex::new(HashMap::new()));

    let state =
        Arc::new(AppState { next_seq_id: AtomicU64::new(1), enqueue_tx, metrics: metrics.clone() });

    let subs_for_loop = subscribers.clone();
    let metrics_for_loop = metrics.clone();
    tokio::spawn(async move {
        let mut scheduler = scheduler;
        metrics_for_loop.mark_start();
        loop {
            while let Ok(req) = enqueue_rx.try_recv() {
                let seq_id = req.seq_id;
                if let Err(e) = scheduler.enqueue(req) {
                    tracing::warn!("rejecting request {seq_id}: {e:?}");
                    if let Some(tx) = subs_for_loop.lock().await.remove(&seq_id) {
                        let _ = tx.send(TokenEvent { token_index: 0, done: true }).await;
                    }
                }
            }
            if !scheduler.has_work() {
                match enqueue_rx.recv().await {
                    Some(req) => {
                        let seq_id = req.seq_id;
                        if let Err(e) = scheduler.enqueue(req) {
                            tracing::warn!("rejecting request {seq_id}: {e:?}");
                            if let Some(tx) = subs_for_loop.lock().await.remove(&seq_id) {
                                let _ = tx.send(TokenEvent { token_index: 0, done: true }).await;
                            }
                        }
                    }
                    None => break,
                }
                continue;
            }

            let backend = backend.clone();
            let report = scheduler.step(|seq_id| backend.advance_token(seq_id));

            // Call free_seq on the backend for every finished sequence so
            // backends can clean up per-sequence state (RNG entries, token
            // history buffers). Without this the trait fix does nothing.
            for seq_id in &report.finished {
                backend.free_seq(seq_id);
            }

            metrics_for_loop.record_tokens(report.ran.len() as u64);
            let mut subs = subs_for_loop.lock().await;

            let preempted_set: std::collections::HashSet<SeqId> =
                report.preempted.iter().cloned().collect();

            for seq_id in &report.ran {
                if preempted_set.contains(seq_id) {
                    continue;
                }
                if let Some(tx) = subs.get(seq_id) {
                    let done = report.finished.contains(seq_id);
                    let token_index = report
                        .generated_counts
                        .get(seq_id)
                        .map(|g| g.saturating_sub(1))
                        .unwrap_or(0);
                    if tx.send(TokenEvent { token_index, done }).await.is_err() {
                        tracing::debug!("client disconnected for seq {seq_id}");
                        subs.remove(seq_id);
                    }
                }
            }

            for seq_id in &report.finished {
                if !report.ran.contains(seq_id) || preempted_set.contains(seq_id) {
                    if let Some(tx) = subs.get(seq_id) {
                        let _ = tx.send(TokenEvent { token_index: 0, done: true }).await;
                    }
                }
                subs.remove(seq_id);
                metrics_for_loop.record_completion();
            }
            drop(subs);

            tokio::task::yield_now().await;
        }
    });

    Router::new()
        .route("/generate", post(generate))
        .route("/metrics", get(metrics_endpoint))
        .with_state((state, subscribers))
}

async fn generate(
    State((state, subscribers)): State<(Arc<AppState>, Subscribers)>,
    Json(req): Json<GenerateReq>,
) -> Response {
    let seq_id = state.next_seq_id.fetch_add(1, Ordering::AcqRel);
    let (tx, mut rx) = mpsc::channel::<TokenEvent>(TOKEN_CHANNEL_CAPACITY);
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
            match serde_json::to_string(&ev) {
                Ok(line) => {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(line + "\n"));
                }
                Err(e) => {
                    tracing::error!("failed to serialize token event: {e}");
                    break;
                }
            }
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
    State((state, _)): State<(Arc<AppState>, Subscribers)>,
) -> Json<serde_json::Value> {
    let snap = state.metrics.snapshot();
    Json(serde_json::json!({
        "elapsed_secs": snap.elapsed.as_secs_f64(),
        "tokens_generated": snap.tokens_generated,
        "requests_completed": snap.requests_completed,
        "tokens_per_sec": snap.tokens_per_sec,
        "tokens_per_sec_rolling": snap.tokens_per_sec_rolling,
        "ttft_p50_ms": snap.ttft_p50.as_millis(),
        "ttft_p99_ms": snap.ttft_p99.as_millis(),
    }))
}
