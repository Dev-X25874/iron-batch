//! Continuous batching scheduler, Orca/vLLM style. Every decode step we
//! admit whatever new requests fit, advance the whole running batch by one
//! token, and evict anything that just finished.

use crate::kv_cache::{AllocError, BlockAllocator, SeqId};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Errors returned by [`Scheduler::enqueue`].
#[derive(Debug)]
pub enum SchedulerError {
    /// Prompt alone is bigger than `max_batch_tokens` -- this thing could
    /// never be admitted, ever. We're strict FCFS (a request that doesn't
    /// fit stays at the head of the queue instead of letting something
    /// smaller cut the line), so if we let this one queue up it just
    /// wedges everyone behind it forever. Reject at enqueue time instead.
    PromptExceedsBatchBudget {
        seq_id: SeqId,
        prompt_tokens: u32,
        max_batch_tokens: u32,
    },
}

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
    /// seq_id -> `generated` count *after* this step, for every seq_id in
    /// `ran`. Callers (server.rs) use this to report which token index
    /// they're on instead of making something up.
    pub generated_counts: HashMap<SeqId, u32>,
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

    /// Queue a request for admission. Rejects (without queueing) any
    /// request whose prompt alone exceeds `max_batch_tokens` -- see
    /// `SchedulerError::PromptExceedsBatchBudget`.
    pub fn enqueue(&mut self, req: Request) -> Result<(), SchedulerError> {
        if req.prompt_tokens > self.cfg.max_batch_tokens {
            return Err(SchedulerError::PromptExceedsBatchBudget {
                seq_id: req.seq_id,
                prompt_tokens: req.prompt_tokens,
                max_batch_tokens: self.cfg.max_batch_tokens,
            });
        }
        self.waiting.push_back(req);
        Ok(())
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
            // Reserve blocks for the request's full context so far, not
            // just its prompt. First-time admission and `generated == 0`
            // are the same case, but a request coming back from a
            // preemption already has tokens behind it -- if we only ask
            // for `prompt_tokens` worth of blocks here, it resumes with
            // fewer blocks than the tokens it's already generated need,
            // and the KV bookkeeping silently goes out of sync with the
            // decode step count until it hits the next OOM.
            let context_tokens = req.prompt_tokens + req.generated;
            match self.allocator.allocate_seq(req.seq_id, context_tokens) {
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
        let mut generated_counts = HashMap::new();

        for req in self.running.iter_mut() {
            // max_new_tokens == 0 means this thing was done the moment it
            // was admitted. Don't call step_fn for a token nobody asked for.
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
            generated_counts.insert(req.seq_id, req.generated);

            // Grow the KV cache every block_size tokens. If we're out of
            // blocks, preempt back to `waiting` rather than let the
            // sequence keep decoding with nowhere to put its KV.
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
            generated_counts,
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
            let _ = sched.enqueue(Request {
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
        let _ = sched.enqueue(Request {
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
        let _ = sched.enqueue(Request {
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
        let _ = sched.enqueue(Request {
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

    #[test]
    fn resumed_seq_reserves_blocks_for_tokens_already_generated() {
        // seq 1 needs 1 block for its prompt and will need a 2nd once it's
        // generated 16 tokens. We force that 2nd grow to OOM by parking an
        // unrelated dummy sequence on the only other block, then free the
        // dummy and check what seq 1 actually gets on resume: it should
        // reserve for prompt + generated (32 tokens -> 2 blocks), not just
        // its original prompt (16 tokens -> 1 block).
        let alloc = BlockAllocator::new(2, 16);
        let mut sched = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 1000,
                max_running_seqs: 8,
            },
            alloc,
        );
        sched
            .enqueue(Request {
                seq_id: 1,
                prompt_tokens: 16,
                max_new_tokens: 40,
                generated: 0,
                arrival: Instant::now(),
                first_token_at: None,
            })
            .unwrap();

        sched.step(|_| false); // admits seq 1, takes block 1 of 2
        sched.allocator.allocate_seq(999, 16).unwrap(); // occupy the last block

        let mut report;
        loop {
            report = sched.step(|_| false);
            if report.preempted.contains(&1) {
                break;
            }
        }
        assert!(sched.allocator.page_table(1).is_none()); // freed on preempt

        sched.allocator.free_seq(999); // give seq 1 somewhere to resume into
        let resumed = sched.step(|_| false);
        assert!(resumed.admitted.contains(&1));
        assert_eq!(
            sched.allocator.page_table(1).unwrap().len(),
            2,
            "resumed seq should reserve blocks for prompt + already-generated tokens"
        );
    }
}
