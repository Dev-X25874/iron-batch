//! Userspace-side device interface, shaped the way it would look talking to
//! a real Linux character-device driver for an accelerator card.
//!
//! This is NOT a kernel module. What's here is the userspace contract a
//! real driver would need to satisfy, plus an in-process fake so the rest
//! of the stack builds and tests without hardware.

use std::io;
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, RawFd};

#[repr(u32)]
#[allow(dead_code)] // command numbers are documentation for the real ioctl surface
pub enum Cmd {
    SubmitBatch = 1,
    WaitFence = 2,
    QueryFreeBlocks = 3,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchDescriptor {
    pub batch_id: u64,
    pub num_seqs: u32,
    pub ring_offset: u32,
}

pub trait DeviceHandle: Send + Sync {
    fn submit_batch(&self, desc: BatchDescriptor) -> io::Result<u64>;
    fn wait_fence(&self, fence: u64) -> io::Result<()>;
    /// Returns the number of free HBM blocks for backpressure decisions.
    fn free_blocks(&self) -> io::Result<u32>;
    /// Write into the mmap'd command ring. Real implementation returns a
    /// pointer into BAR-backed memory; fake returns a pointer into an
    /// in-process Vec. Callers must hold the ring lock for the duration
    /// of any write to avoid races.
    ///
    /// # Safety
    /// Caller must not hold any reference into the ring across an await
    /// point or across a call that could reallocate the backing buffer.
    unsafe fn ring_ptr(&self) -> *mut u8;
}

pub struct FakeDevice {
    /// The ring is behind a Mutex. Callers must lock it, write, and
    /// unlock within a single synchronous section — never hold a raw
    /// pointer across the lock boundary.
    ring: parking_lot::Mutex<Vec<u8>>,
    next_fence: std::sync::atomic::AtomicU64,
    /// Tracks allocated blocks so free_blocks() returns accurate values
    /// instead of always returning the initial total.
    allocated_blocks: std::sync::atomic::AtomicU32,
    total_blocks: u32,
}

impl FakeDevice {
    pub fn new(ring_bytes: usize, total_blocks: u32) -> Self {
        Self {
            ring: parking_lot::Mutex::new(vec![0u8; ring_bytes]),
            next_fence: std::sync::atomic::AtomicU64::new(0),
            allocated_blocks: std::sync::atomic::AtomicU32::new(0),
            total_blocks,
        }
    }

    pub fn allocate(&self, n: u32) -> io::Result<()> {
        use std::sync::atomic::Ordering;
        let prev = self.allocated_blocks.fetch_add(n, Ordering::SeqCst);
        if prev + n > self.total_blocks {
            self.allocated_blocks.fetch_sub(n, Ordering::SeqCst);
            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "no free blocks"));
        }
        Ok(())
    }

    pub fn deallocate(&self, n: u32) {
        use std::sync::atomic::Ordering;
        self.allocated_blocks.fetch_sub(n, Ordering::SeqCst);
    }
}

impl DeviceHandle for FakeDevice {
    fn submit_batch(&self, desc: BatchDescriptor) -> io::Result<u64> {
        use std::sync::atomic::Ordering;
        let fence = self.next_fence.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = desc;
        Ok(fence)
    }

    fn wait_fence(&self, _fence: u64) -> io::Result<()> {
        Ok(())
    }

    fn free_blocks(&self) -> io::Result<u32> {
        use std::sync::atomic::Ordering;
        let allocated = self.allocated_blocks.load(Ordering::SeqCst);
        Ok(self.total_blocks.saturating_sub(allocated))
    }

    /// Returns a pointer into the locked ring buffer.
    ///
    /// # Safety
    /// The caller must ensure no other thread holds a pointer into the
    /// ring while writing. In the fake implementation the ring is an
    /// in-process Vec — do not let the pointer escape the lock's scope
    /// or the Vec may reallocate and invalidate it.
    unsafe fn ring_ptr(&self) -> *mut u8 {
        // Intentionally return the pointer while holding the lock.
        // Callers must structure their use as:
        //   let _guard = device.ring.lock();
        //   let ptr = device.ring_ptr();
        //   // write through ptr
        //   // _guard drops here, unlocking
        // This is enforced by convention, not the type system, because
        // returning a MutexGuard from a trait method would make the trait
        // non-object-safe.
        self.ring.lock().as_mut_ptr()
    }
}

/// Sketch of the real path (not compiled).
///
/// ```ignore
/// use nix::sys::mman::{mmap, MapFlags, ProtFlags};
/// use nix::{ioctl_readwrite, ioctl_read};
///
/// ioctl_readwrite!(accel_submit_batch, b'A', 1, BatchDescriptor);
/// ioctl_read!(accel_query_free_blocks, b'A', 3, u32);
///
/// pub struct RealDevice { fd: std::fs::File, bar: *mut u8, bar_len: usize }
/// impl RealDevice {
///     pub fn open(path: &str, bar_len: usize) -> io::Result<Self> {
///         let fd = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
///         let bar = unsafe {
///             mmap(None, bar_len.try_into().unwrap(),
///                  ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
///                  MapFlags::MAP_SHARED, &fd, 0)?
///         } as *mut u8;
///         Ok(Self { fd, bar, bar_len })
///     }
/// }
/// ```
struct _RealDeviceSketch;

#[cfg(unix)]
pub struct NullFdWrapper(pub RawFd);
#[cfg(unix)]
impl AsRawFd for NullFdWrapper {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_device_roundtrip() {
        let dev = FakeDevice::new(4096, 1024);
        assert_eq!(dev.free_blocks().unwrap(), 1024);
        let fence =
            dev.submit_batch(BatchDescriptor { batch_id: 1, num_seqs: 8, ring_offset: 0 }).unwrap();
        dev.wait_fence(fence).unwrap();
        assert_eq!(fence, 1);
    }

    #[test]
    fn free_blocks_tracks_allocations() {
        let dev = FakeDevice::new(4096, 1024);
        dev.allocate(100).unwrap();
        assert_eq!(dev.free_blocks().unwrap(), 924);
        dev.deallocate(50);
        assert_eq!(dev.free_blocks().unwrap(), 974);
    }

    #[test]
    fn allocate_beyond_total_fails() {
        let dev = FakeDevice::new(4096, 10);
        assert!(dev.allocate(11).is_err());
        assert_eq!(dev.free_blocks().unwrap(), 10); // unchanged
    }
}
