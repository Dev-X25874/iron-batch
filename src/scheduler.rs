//! Continuous batching scheduler (Orca / vLLM style): every decode step,
//! finished sequences are evicted and newly-arrived requests are admitted
//! into the running batch, so GPU/accelerator utilization never drains to
//! zero waiting for the slowest sequence in a static batch to finish.

use crate::kv_cache::{AllocError, BlockAllocator, SeqId};
use std::collections::VecDeque;
use std::time::Instant;

pub struct Request {
    pub seq_id: SeqId,
    pub prompt_tokens: u32,
    pub max_new_tokens: u32,
    pub generated: u32,
    pub arrival: Instant,
    pub first_token_at: Option<Instant>,
}

impl Request {
    pub fn is_done(&self) -> bool {
        self.generated >= self.max_new_tokens
    }
}

pub struct SchedulerConfig {
    /// Cap on total tokens (prompt + already-generated) processed in one
    /// decode step across the whole running batch. This is the knob that
    /// actually bounds accelerator memory + step latency.
    pub max_batch_tokens: u32,
    pub max_running_seqs: usize,
}

pub struct Scheduler {
    cfg: SchedulerConfig,
    allocator: BlockAllocator,
    waiting: VecDeque<Request>,
    running: Vec<Request>,
}

pub struct StepReport {
    pub ran: Vec<SeqId>,
    pub admitted: Vec<SeqId>,
    pub finished: Vec<SeqId>,
    pub preempted: Vec<SeqId>,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig, allocator: BlockAllocator) -> Self {
        Self {
            cfg,
            allocator,
            waiting: VecDeque::new(),
            running: Vec::new(),
        }
    }

    pub fn enqueue(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    fn running_token_budget(&self) -> u32 {
        self.running
            .iter()
            .map(|r| r.prompt_tokens + r.generated)
            .sum()
    }

    /// Admit as many waiting requests as fit under the batch-token and
    /// max-running-seqs caps, respecting KV block availability. FCFS with
    /// no starvation: a request that doesn't fit stays at the front of the
    /// queue rather than letting a later, smaller request jump it.
    fn admit(&mut self) -> Vec<SeqId> {
        let mut admitted = Vec::new();
        while let Some(front) = self.waiting.front() {
            if self.running.len() >= self.cfg.max_running_seqs {
                break;
            }
            let projected = self.running_token_budget() + front.prompt_tokens;
            if projected > self.cfg.max_batch_tokens {
                break;
            }
            let req = self.waiting.pop_front().unwrap();
            match self.allocator.allocate_seq(req.seq_id, req.prompt_tokens) {
                Ok(()) => {
                    admitted.push(req.seq_id);
                    self.running.push(req);
                }
                Err(AllocError::OutOfMemory) => {
                    // put it back, stop admitting this step
                    self.waiting.push_front(req);
                    break;
                }
                Err(e) => panic!("scheduler bug: {e:?}"),
            }
        }
        admitted
    }

    /// One decode step: admit new work, then advance every running sequence
    /// by one token via `step_fn` (the compute backend), growing KV blocks
    /// as needed, and retiring anything that just finished.
    pub fn step<F>(&mut self, mut step_fn: F) -> StepReport
    where
        F: FnMut(&SeqId) -> bool, // returns true if this sequence emitted an EOS
    {
        let admitted = self.admit();
        let mut ran = Vec::with_capacity(self.running.len());
        let mut finished = Vec::new();
        let mut preempted = Vec::new();

        for req in self.running.iter_mut() {
            // A request admitted with max_new_tokens == 0 is already done —
            // finish it without ever calling step_fn, instead of generating
            // one token it never asked for.
            if req.is_done() {
                finished.push(req.seq_id);
                continue;
            }

            let eos = step_fn(&req.seq_id);
            req.generated += 1;
            if req.first_token_at.is_none() {
                req.first_token_at = Some(Instant::now());
            }
            ran.push(req.seq_id);

            // grow KV cache every block_size tokens. If the pool is out of
            // blocks, preempt this sequence back to `waiting` instead of
            // silently letting it keep generating tokens with no KV storage
            // behind them.
            if req.generated % self.allocator.block_size == 0
                && self.allocator.grow_seq(req.seq_id).is_err()
            {
                preempted.push(req.seq_id);
                continue;
            }

            if eos || req.is_done() {
                finished.push(req.seq_id);
            }
        }

        if !finished.is_empty() {
            self.running.retain(|r| !finished.contains(&r.seq_id));
            for id in &finished {
                self.allocator.free_seq(*id);
            }
        }

        if !preempted.is_empty() {
            let mut i = 0;
            while i < self.running.len() {
                if preempted.contains(&self.running[i].seq_id) {
                    let req = self.running.remove(i);
                    self.allocator.free_seq(req.seq_id);
                    self.waiting.push_front(req);
                } else {
                    i += 1;
                }
            }
        }

        StepReport {
            ran,
            admitted,
            finished,
            preempted,
        }
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    pub fn free_kv_blocks(&self) -> u32 {
        self.allocator.num_free_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_until_token_budget_full() {
        let alloc = BlockAllocator::new(64, 16);
        let mut sched = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 100,
                max_running_seqs: 8,
            },
            alloc,
        );
        for i in 0..5 {
            sched.enqueue(Request {
                seq_id: i,
                prompt_tokens: 30,
                max_new_tokens: 4,
                generated: 0,
                arrival: Instant::now(),
                first_token_at: None,
            });
        }
        let report = sched.step(|_| false);
        // 100 / 30 -> 3 admitted this step, 2 stay waiting
        assert_eq!(report.admitted.len(), 3);
        assert_eq!(sched.waiting_count(), 2);
    }

    #[test]
    fn finished_seqs_free_their_blocks() {
        let alloc = BlockAllocator::new(64, 16);
        let mut sched = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 1000,
                max_running_seqs: 8,
            },
            alloc,
        );
        sched.enqueue(Request {
            seq_id: 1,
            prompt_tokens: 16,
            max_new_tokens: 1,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        });
        let r1 = sched.step(|_| false);
        assert_eq!(r1.finished, vec![1]);
        assert_eq!(sched.running_count(), 0);
    }

    #[test]
    fn zero_max_new_tokens_generates_nothing() {
        let alloc = BlockAllocator::new(64, 16);
        let mut sched = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 1000,
                max_running_seqs: 8,
            },
            alloc,
        );
        sched.enqueue(Request {
            seq_id: 1,
            prompt_tokens: 16,
            max_new_tokens: 0,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        });
        let r1 = sched.step(|_| panic!("step_fn must not be called for a 0-token request"));
        assert_eq!(r1.finished, vec![1]);
        assert!(r1.ran.is_empty());
    }

    #[test]
    fn oom_during_decode_preempts_instead_of_silently_continuing() {
        // Only enough blocks for the prompt; the first growth attempt
        // during decode must fail and preempt the sequence.
        let alloc = BlockAllocator::new(1, 16);
        let mut sched = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 1000,
                max_running_seqs: 8,
            },
            alloc,
        );
        sched.enqueue(Request {
            seq_id: 1,
            prompt_tokens: 16, // exactly 1 block, none left to grow into
            max_new_tokens: 20,
            generated: 0,
            arrival: Instant::now(),
            first_token_at: None,
        });
        // step 1: admits, generates token 1..15 without needing to grow
        // (block_size is 16, so growth is only attempted every 16th token)
        let mut report = sched.step(|_| false);
        for _ in 0..15 {
            report = sched.step(|_| false);
        }
        assert!(report.preempted.contains(&1));
        assert_eq!(sched.running_count(), 0);
        assert_eq!(sched.waiting_count(), 1); // back in the queue, not lost
    }
}

// Preemption is now handled: OOM during grow_seq() moves the sequence back
// to `waiting` and frees its blocks rather than silently skipping growth.
