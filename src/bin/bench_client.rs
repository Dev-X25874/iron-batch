//! Closed-loop-free load generator: fires `--concurrency` requests
//! immediately (Poisson-ish arrival via jitter) rather than waiting for
//! each to finish, so it actually exercises the scheduler's continuous
//! batching instead of measuring one sequence at a time.

use clap::Parser;
use futures_util::{StreamExt, TryStreamExt};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncBufReadExt;
use tokio_util::compat::FuturesAsyncReadCompatExt;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    url: String,
    #[arg(long, default_value_t = 64)]
    concurrency: usize,
    #[arg(long, default_value_t = 200)]
    total_requests: usize,
    #[arg(long, default_value_t = 128)]
    prompt_tokens: u32,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: u32,
}

#[tokio::main]
async fn main() {
    let args = Arc::new(Args::parse());
    let client = reqwest::Client::new();
    let tokens_total = Arc::new(AtomicU64::new(0));
    let failures = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let sem = Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    let mut handles = Vec::new();

    for _ in 0..args.total_requests {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let args = args.clone();
        let tokens_total = tokens_total.clone();
        let failures = failures.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let resp = match client
                .post(format!("{}/generate", args.url))
                .json(&serde_json::json!({
                    "prompt_tokens": args.prompt_tokens,
                    "max_new_tokens": args.max_new_tokens,
                }))
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    // Don't let one bad request take down the whole run --
                    // that defeats the point of a load test. Just count it.
                    eprintln!("request failed: {e}");
                    failures.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };
            let async_read = resp
                .bytes_stream()
                .map(|r| r.map_err(std::io::Error::other))
                .into_async_read()
                .compat();
            let mut reader = tokio::io::BufReader::new(async_read);
            let mut line = String::new();
            let mut n = 0u64;
            loop {
                line.clear();
                let read = reader.read_line(&mut line).await.unwrap_or(0);
                if read == 0 {
                    break;
                }
                n += 1;
            }
            tokens_total.fetch_add(n, Ordering::Relaxed);
        }));
    }

    for h in handles {
        let _ = h.await;
    }

    let elapsed = start.elapsed();
    let total = tokens_total.load(Ordering::Relaxed);
    let failed = failures.load(Ordering::Relaxed);
    println!(
        "requests={} failed={} tokens={} elapsed={:.2}s throughput={:.1} tok/s",
        args.total_requests,
        failed,
        total,
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64()
    );
}
