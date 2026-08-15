//! In-process benchmark of the scheduler/allocator hot path, isolated from
//! HTTP and the mock backend's sleep so it measures pure scheduling
//! overhead (admission + KV bookkeeping per decode step). Run with
//! `cargo bench` — this is the number to watch when changing the
//! allocator or admission policy.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fractile_infer::kv_cache::BlockAllocator;
use fractile_infer::scheduler::{Request, Scheduler, SchedulerConfig};
use std::time::Instant;

fn make_scheduler(n_seqs: u64) -> Scheduler {
    let alloc = BlockAllocator::new(1 << 16, 16);
    let mut sched = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 1 << 20,
            max_running_seqs: n_seqs as usize,
        },
        alloc,
    );
    for i in 0..n_seqs {
        sched.enqueue(Request {
            seq_id: i,
            prompt_tokens: 64,
            max_new_tokens: 128,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        });
    }
    sched
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_step");
    for &n in &[8usize, 64, 256] {
        group.bench_function(format!("running_{n}"), |b| {
            b.iter_batched(
                || make_scheduler(n as u64),
                |mut sched| {
                    let report = sched.step(|seq_id| black_box(*seq_id) % 97 == 0);
                    black_box(report);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
