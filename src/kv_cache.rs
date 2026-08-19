//! Block-based paged KV allocator, basically PagedAttention (vLLM) but for
//! a fixed-size block pool that a real backend would back with
//! device-resident memory. O(1) alloc/free via a free-list stack, and
//! blocks are refcounted so a shared prompt prefix (system prompt,
//! few-shot examples) can be reused across sequences without copying.

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
    SeqAlreadyExists(SeqId),
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
    /// Saturating add so a pathological `num_tokens` (e.g. an unvalidated
    /// value from the network) can't wrap around instead of just failing
    /// the allocation like it should.
    pub fn blocks_needed(&self, num_tokens: u32) -> u32 {
        num_tokens.saturating_add(self.block_size - 1) / self.block_size
    }

    /// Register a sequence and reserve `num_tokens` worth of blocks for it.
    /// Fails clean (no state mutated) if we're out of blocks, or if
    /// `seq_id` is already registered -- overwriting it would leak
    /// whatever blocks it was already holding.
    pub fn allocate_seq(&self, seq_id: SeqId, num_tokens: u32) -> Result<(), AllocError> {
        let needed = self.blocks_needed(num_tokens);
        let mut inner = self.inner.lock();
        if inner.page_tables.contains_key(&seq_id) {
            return Err(AllocError::SeqAlreadyExists(seq_id));
        }
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

    /// Append one block to a sequence once it decodes past its current
    /// capacity -- called roughly every `block_size` tokens. We check
    /// `seq_id` exists before popping off the free stack, on purpose: pop
    /// first and you can leak a block on every bad call.
    pub fn grow_seq(&self, seq_id: SeqId) -> Result<BlockId, AllocError> {
        let mut inner = self.inner.lock();
        if !inner.page_tables.contains_key(&seq_id) {
            return Err(AllocError::UnknownSeq(seq_id));
        }
        let id = inner.free_stack.pop().ok_or(AllocError::OutOfMemory)?;
        inner.meta[id as usize].refcount = 1;
        inner
            .page_tables
            .get_mut(&seq_id)
            .expect("checked above")
            .push(id);
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

    #[test]
    fn duplicate_seq_id_is_rejected_not_leaked() {
        let a = BlockAllocator::new(4, 16);
        a.allocate_seq(1, 16).unwrap();
        assert_eq!(a.num_free_blocks(), 3);
        assert!(matches!(
            a.allocate_seq(1, 16),
            Err(AllocError::SeqAlreadyExists(1))
        ));
        assert_eq!(a.num_free_blocks(), 3); // unchanged, original blocks intact
    }

    #[test]
    fn grow_unknown_seq_does_not_leak_a_block() {
        let a = BlockAllocator::new(4, 16);
        assert!(matches!(
            a.grow_seq(99),
            Err(AllocError::UnknownSeq(99))
        ));
        assert_eq!(a.num_free_blocks(), 4); // no block was popped and lost
    }
        }
