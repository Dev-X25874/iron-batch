use fractile_infer::backend::{Backend, MockBackend};
use fractile_infer::kv_cache::BlockAllocator;
use fractile_infer::scheduler::{Request, Scheduler, SchedulerConfig};
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
        });
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
