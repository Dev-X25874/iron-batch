use clap::Parser;
use iron_batch::backend::{Backend, MockBackend};
use iron_batch::scheduler::SchedulerConfig;
use iron_batch::server::build_router;
use std::sync::Arc;
use std::time::Duration;

/// Reference inference server: continuous batching + paged KV cache over a
/// pluggable compute backend. Default backend is a synthetic mock so this
/// runs standalone; point `--backend` at a real target once wired up.
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "0.0.0.0:8080")]
    addr: String,
    #[arg(long, default_value_t = 4096)]
    total_kv_blocks: u32,
    #[arg(long, default_value_t = 16)]
    kv_block_size: u32,
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: u32,
    #[arg(long, default_value_t = 256)]
    max_running_seqs: usize,
    /// Simulated per-token compute latency in microseconds (mock backend only).
    #[arg(long, default_value_t = 200)]
    mock_token_latency_us: u64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let backend: Arc<dyn Backend> = Arc::new(MockBackend::new(
        Duration::from_micros(args.mock_token_latency_us),
        0.02, // ~50 token average sequence length
    ));

    let router = build_router(
        backend,
        args.total_kv_blocks,
        args.kv_block_size,
        SchedulerConfig {
            max_batch_tokens: args.max_batch_tokens,
            max_running_seqs: args.max_running_seqs,
        },
    );

    tracing::info!("listening on {}", args.addr);
    let listener = tokio::net::TcpListener::bind(&args.addr).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}
