//! Block-based paged KV cache allocator, modeled on the PagedAttention design
//! (vLLM) but written for a fixed-size physical block pool that a real
//! backend would back with device-resident memory (HBM on Fractile's chip).
//!
//! Design goals:
//!   - O(1) block alloc/free via a free-list stack
//!   - reference counting so a shared prompt prefix (system prompt, few-shot
//!     examples) can be reused across requests without copying
//!   - no fragmentation: every block is `block_size` tokens, sequences are
//!     built from a chain of block ids (a page table)

use parking_lot::Mutex;
use std::collections::HashMap;

pub type BlockId = u32;
pub type SeqId = u64;

/// A single physical block's bookkeeping. The actual KV tensor storage is
/// out of scope here — swap `refcount`'s owner for a real device pointer /
/// slab index when wiring this to hardware.
#[derive(Debug, Default, Clone, Copy)]
struct BlockMeta {
    refcount: u32,
}

pub struct BlockAllocator {
    inner: Mutex<AllocatorInner>,
    pub block_size: u32,
    pub total_blocks: u32,
}

struct AllocatorInner {
    meta: Vec<BlockMeta>,
    free_stack: Vec<BlockId>,
    /// seq_id -> ordered list of block ids (the sequence's page table)
    page_tables: HashMap<SeqId, Vec<BlockId>>,
}

#[derive(Debug)]
pub enum AllocError {
    OutOfMemory,
    UnknownSeq(SeqId),
}

impl BlockAllocator {
    pub fn new(total_blocks: u32, block_size: u32) -> Self {
        let free_stack: Vec<BlockId> = (0..total_blocks).rev().collect();
        Self {
            inner: Mutex::new(AllocatorInner {
                meta: vec![BlockMeta::default(); total_blocks as usize],
                free_stack,
                page_tables: HashMap::new(),
            }),
            block_size,
            total_blocks,
        }
    }

    pub fn num_free_blocks(&self) -> u32 {
        self.inner.lock().free_stack.len() as u32
    }

    /// Number of blocks required to hold `num_tokens` tokens for a new seq.
    pub fn blocks_needed(&self, num_tokens: u32) -> u32 {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// Register a new sequence and reserve its first `num_tokens` worth of
    /// blocks. Returns Err(OutOfMemory) without mutating state if the pool
    /// can't satisfy the request (caller should preempt/evict and retry).
    pub fn allocate_seq(&self, seq_id: SeqId, num_tokens: u32) -> Result<(), AllocError> {
        let needed = self.blocks_needed(num_tokens);
        let mut inner = self.inner.lock();
        if inner.free_stack.len() < needed as usize {
            return Err(AllocError::OutOfMemory);
        }
        let mut blocks = Vec::with_capacity(needed as usize);
        for _ in 0..needed {
            let id = inner.free_stack.pop().unwrap();
            inner.meta[id as usize].refcount = 1;
            blocks.push(id);
        }
        inner.page_tables.insert(seq_id, blocks);
        Ok(())
    }

    /// Append one more block to a sequence as it decodes past its current
    /// capacity. This is the steady-state allocation path during
    /// autoregressive decode (one call roughly every `block_size` tokens).
    pub fn grow_seq(&self, seq_id: SeqId) -> Result<BlockId, AllocError> {
        let mut inner = self.inner.lock();
        let id = inner
            .free_stack
            .pop()
            .ok_or(AllocError::OutOfMemory)?;
        inner.meta[id as usize].refcount = 1;
        let table = inner
            .page_tables
            .get_mut(&seq_id)
            .ok_or(AllocError::UnknownSeq(seq_id))?;
        table.push(id);
        Ok(id)
    }

    /// Fork a sequence's page table for beam search / parallel sampling
    /// without copying KV — every shared block's refcount is bumped, and
    /// only diverging (post-fork) blocks get their own allocation.
    pub fn fork_seq(&self, src: SeqId, dst: SeqId) -> Result<(), AllocError> {
        let mut inner = self.inner.lock();
        let src_blocks = inner
            .page_tables
            .get(&src)
            .ok_or(AllocError::UnknownSeq(src))?
            .clone();
        for &b in &src_blocks {
            inner.meta[b as usize].refcount += 1;
        }
        inner.page_tables.insert(dst, src_blocks);
        Ok(())
    }

    /// Free every block owned by a sequence, decrementing shared refcounts
    /// and only returning fully-unreferenced blocks to the free stack.
    pub fn free_seq(&self, seq_id: SeqId) {
        let mut inner = self.inner.lock();
        if let Some(blocks) = inner.page_tables.remove(&seq_id) {
            for b in blocks {
                let m = &mut inner.meta[b as usize];
                m.refcount = m.refcount.saturating_sub(1);
                if m.refcount == 0 {
                    inner.free_stack.push(b);
                }
            }
        }
    }

    pub fn page_table(&self, seq_id: SeqId) -> Option<Vec<BlockId>> {
        self.inner.lock().page_tables.get(&seq_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_roundtrip() {
        let a = BlockAllocator::new(4, 16);
        assert_eq!(a.num_free_blocks(), 4);
        a.allocate_seq(1, 20).unwrap(); // needs 2 blocks
        assert_eq!(a.num_free_blocks(), 2);
        a.free_seq(1);
        assert_eq!(a.num_free_blocks(), 4);
    }

    #[test]
    fn oom_is_clean() {
        let a = BlockAllocator::new(2, 16);
        a.allocate_seq(1, 32).unwrap(); // uses both blocks
        assert!(matches!(a.allocate_seq(2, 16), Err(AllocError::OutOfMemory)));
        assert_eq!(a.num_free_blocks(), 0); // failed alloc didn't leak partial state
    }

    #[test]
    fn fork_shares_blocks_until_freed() {
        let a = BlockAllocator::new(4, 16);
        a.allocate_seq(1, 16).unwrap();
        assert_eq!(a.num_free_blocks(), 3);
        a.fork_seq(1, 2).unwrap();
        assert_eq!(a.num_free_blocks(), 3); // no new blocks on fork
        a.free_seq(1);
        assert_eq!(a.num_free_blocks(), 3); // still referenced by seq 2
        a.free_seq(2);
        assert_eq!(a.num_free_blocks(), 4);
    }
}
