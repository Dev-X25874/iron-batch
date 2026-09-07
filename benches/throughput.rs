//! In-process benchmark of the scheduler/allocator hot path, isolated from
//! HTTP and the mock backend's sleep. Measures pure scheduling overhead
//! (admission + KV bookkeeping per decode step).
//! Run with `cargo bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iron_batch::kv_cache::BlockAllocator;
use iron_batch::scheduler::{Request, Scheduler, SchedulerConfig};
use std::time::Instant;

fn make_scheduler(n_seqs: u64, prompt_tokens: u32, max_new_tokens: u32) -> Scheduler {
    // Block pool large enough that alloc pressure never masks scheduling cost.
    let alloc = BlockAllocator::new(1 << 16, 16);
    let mut sched = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 1 << 20,
            max_running_seqs: n_seqs as usize,
        },
        alloc,
    );
    for i in 0..n_seqs {
        let _ = sched.enqueue(Request {
            seq_id: i,
            prompt_tokens,
            max_new_tokens,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        });
    }
    sched
}

/// Single-step cost on a fresh batch — the cheapest case (no KV growth,
/// no preemptions). Establishes the lower bound for step() overhead.
fn bench_step_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_step_first");
    for &n in &[8usize, 64, 256] {
        group.bench_function(format!("running_{n}"), |b| {
            b.iter_batched(
                || make_scheduler(n as u64, 64, 128),
                |mut sched| {
                    let report = sched.step(|seq_id| black_box(*seq_id) % 97 == 0);
                    black_box(report);
                },
                // LargeInput runs setup outside the timing window so
                // make_scheduler's enqueue cost doesn't pollute step() time.
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Steady-state cost: run the scheduler for several steps so KV growth and
/// preemption paths get exercised, then measure the Nth step rather than
/// the first. Uses a tight block pool to force preemption.
fn bench_step_steady(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_step_steady");
    for &n in &[8usize, 64, 256] {
        group.bench_function(format!("running_{n}"), |b| {
            b.iter_batched(
                || {
                    // Tight pool: roughly 4 blocks per sequence, forcing
                    // growth and occasional preemption during warmup.
                    let block_pool = (n as u32) * 4;
                    let alloc = BlockAllocator::new(block_pool, 16);
                    let mut sched = Scheduler::new(
                        SchedulerConfig {
                            max_batch_tokens: 1 << 20,
                            max_running_seqs: n,
                        },
                        alloc,
                    );
                    for i in 0..n as u64 {
                        let _ = sched.enqueue(Request {
                            seq_id: i,
                            prompt_tokens: 32,
                            max_new_tokens: 128,
                            generated: 0,
                            arrival: Instant::now(),
                            first_token_at: None,
                        });
                    }
                    // Warm up: run 10 steps so KV growth has happened
                    for _ in 0..10 {
                        sched.step(|seq_id| black_box(*seq_id) % 97 == 0);
                    }
                    sched
                },
                |mut sched| {
                    let report = sched.step(|seq_id| black_box(*seq_id) % 97 == 0);
                    black_box(report);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Allocator microbenchmark in isolation — allocate + free one sequence,
/// so we can attribute what fraction of step() time is pure alloc overhead.
fn bench_allocator(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocator");
    group.bench_function("allocate_grow_free", |b| {
        let alloc = BlockAllocator::new(1 << 16, 16);
        let mut seq_id: u64 = 0;
        b.iter(|| {
            seq_id += 1;
            alloc.allocate_seq(black_box(seq_id), 64).unwrap();
            alloc.grow_seq(black_box(seq_id)).unwrap();
            alloc.free_seq(black_box(seq_id));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_step_first, bench_step_steady, bench_allocator);
criterion_main!(benches);
