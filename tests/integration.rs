use iron_batch::backend::{Backend, MockBackend};
use iron_batch::kv_cache::BlockAllocator;
use iron_batch::scheduler::{Request, Scheduler, SchedulerConfig};
use std::time::{Duration, Instant};

/// Drives the scheduler to completion for a mixed batch of sequences and
/// checks the invariants that actually matter for correctness under
/// continuous batching: every enqueued request finishes exactly once, and
/// the KV allocator returns to a fully-free state once nothing is running
/// (no block leak across admit/evict cycles).
#[test]
fn drains_mixed_batch_without_leaking_kv_blocks() {
    let total_blocks = 256;
    let block_size = 16;
    let alloc = BlockAllocator::new(total_blocks, block_size);
    let mut sched = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 512,
            max_running_seqs: 16,
        },
        alloc,
    );

    let backend = MockBackend::new(Duration::ZERO, 1.0 / 8.0);

    let n_requests = 40u64;
    for i in 0..n_requests {
        sched.enqueue(Request {
            seq_id: i,
            prompt_tokens: 24 + (i % 5) as u32 * 8,
            max_new_tokens: 4 + (i % 3) as u32,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        }).unwrap();
    }

    let mut finished_ids = std::collections::HashSet::new();
    let mut steps = 0;
    while sched.has_work() {
        let report = sched.step(|seq_id| backend.advance_token(seq_id));
        for id in report.finished {
            assert!(finished_ids.insert(id), "seq {id} finished twice");
        }
        steps += 1;
        assert!(steps < 100_000, "scheduler appears stuck");
    }

    assert_eq!(finished_ids.len(), n_requests as usize);
    assert_eq!(sched.free_kv_blocks(), total_blocks, "leaked KV blocks");
}

/// A request whose prompt alone exceeds `max_batch_tokens` can never be
/// admitted under strict FCFS -- it must be rejected at enqueue time,
/// not queued, or it deadlocks every request behind it forever.
#[test]
fn oversized_request_is_rejected_not_deadlocked() {
    let total_blocks = 64;
    let block_size = 16;
    let alloc = BlockAllocator::new(total_blocks, block_size);
    let mut sched = Scheduler::new(
        SchedulerConfig {
            max_batch_tokens: 100,
            max_running_seqs: 8,
        },
        alloc,
    );

    let oversized = Request {
        seq_id: 1,
        prompt_tokens: 200, // exceeds max_batch_tokens on its own
        max_new_tokens: 4,
        generated: 0,
        arrival: Instant::now(),
        first_token_at: None,
    };
    assert!(sched.enqueue(oversized).is_err());
    assert_eq!(sched.waiting_count(), 0, "rejected request must not be queued");

    // A normal request enqueued afterward must still drain to completion,
    // proving the queue isn't wedged.
    sched
        .enqueue(Request {
            seq_id: 2,
            prompt_tokens: 32,
            max_new_tokens: 2,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        })
        .unwrap();

    let backend = MockBackend::new(Duration::ZERO, 1.0);
    let mut steps = 0;
    while sched.has_work() {
        sched.step(|seq_id| backend.advance_token(seq_id));
        steps += 1;
        assert!(steps < 1_000, "scheduler appears stuck");
    }
}
