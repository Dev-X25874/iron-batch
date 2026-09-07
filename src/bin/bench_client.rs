//! Closed-loop-free load generator. Fires all --concurrency requests
//! immediately instead of waiting for each to finish, so it actually
//! exercises continuous batching rather than measuring one sequence at a time.

use clap::Parser;
use futures_util::StreamExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "http://localhost:8080")]
    addr: String,
    #[arg(long, default_value_t = 128)]
    concurrency: usize,
    #[arg(long, default_value_t = 2000)]
    total_requests: usize,
    #[arg(long, default_value_t = 128)]
    prompt_tokens: u32,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: u32,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let client = Arc::new(reqwest::Client::new());
    let tokens_total = Arc::new(AtomicU64::new(0));
    let failures = Arc::new(AtomicU64::new(0));
    let ttfts = Arc::new(parking_lot::Mutex::new(Vec::<Duration>::new()));

    let sem = Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    let mut handles = Vec::with_capacity(args.total_requests);

    // Timer starts when the first request is dispatched, not before,
    // so semaphore wait time doesn't inflate elapsed.
    let start = Instant::now();

    for _ in 0..args.total_requests {
        let permit = sem.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let tokens_total = tokens_total.clone();
        let failures = failures.clone();
        let ttfts = ttfts.clone();
        let url = format!("{}/generate", args.addr);
        let prompt_tokens = args.prompt_tokens;
        let max_new_tokens = args.max_new_tokens;

        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let req_start = Instant::now();

            let resp = match client
                .post(&url)
                .json(&serde_json::json!({
                    "prompt_tokens": prompt_tokens,
                    "max_new_tokens": max_new_tokens,
                }))
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("request failed: {e}");
                    failures.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };

            let mut stream = resp.bytes_stream();
            let mut token_count = 0u64;
            let mut first = true;

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        // Each newline-delimited chunk is one token event.
                        // Count actual token_index values from the JSON
                        // rather than lines, so we're measuring real tokens
                        // not line count (which includes the done:true line).
                        for line in bytes.split(|&b| b == b'\n') {
                            if line.is_empty() {
                                continue;
                            }
                            if first {
                                ttfts.lock().push(req_start.elapsed());
                                first = false;
                            }
                            if let Ok(ev) = serde_json::from_slice::<serde_json::Value>(line) {
                                if ev.get("done").and_then(|d| d.as_bool()) == Some(true) {
                                    // done:true line doesn't carry a new token
                                    break;
                                }
                                token_count += 1;
                            }
                        }
                    }
                    Err(e) => {
                        // Mid-stream error: count what arrived, mark failure.
                        eprintln!("stream error: {e}");
                        failures.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            }

            tokens_total.fetch_add(token_count, Ordering::Relaxed);
        }));
    }

    for h in handles {
        let _ = h.await;
    }

    let elapsed = start.elapsed();
    let total_tokens = tokens_total.load(Ordering::Relaxed);
    let total_failures = failures.load(Ordering::Relaxed);
    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    // Compute client-side TTFT percentiles
    let mut ttft_sorted = ttfts.lock().clone();
    ttft_sorted.sort();
    let p50 = percentile(&ttft_sorted, 0.50);
    let p99 = percentile(&ttft_sorted, 0.99);

    println!("elapsed:    {:.2}s", elapsed.as_secs_f64());
    println!("tokens:     {total_tokens}");
    println!("tok/s:      {tok_per_sec:.1}");
    println!("failures:   {total_failures}");
    println!("ttft p50:   {}ms", p50.as_millis());
    println!("ttft p99:   {}ms", p99.as_millis());
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}
