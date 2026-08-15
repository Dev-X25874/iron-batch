//! Userspace-side device interface, shaped the way it would look talking to
//! a real Linux character-device driver for an accelerator card: open the
//! node, `mmap` a BAR-backed region for the KV cache / activation buffers,
//! and issue `ioctl`s for control-plane ops (submit a batch, wait for a
//! completion fence).
//!
//! This is NOT a kernel module — writing an actual `kernel` crate driver
//! needs the out-of-tree Rust-for-Linux toolchain, which isn't available in
//! this environment. What's here is the userspace contract a real driver
//! would need to satisfy, plus a `/dev/null`-backed fake so the rest of the
//! stack (scheduler, server, benchmark) can be built and tested against a
//! stable `DeviceHandle` trait today. Swapping in the real ioctl numbers
//! and a `/dev/fractile0` node is the only change needed elsewhere.

use std::io;
use std::os::unix::io::{AsRawFd, RawFd};

/// Mirrors what a real driver's ioctl surface would look like: a command
/// number plus an in/out payload struct, matching the
/// `_IOWR`/`_IOW`/`_IOR` convention the kernel side would define with
/// `ioctl_readwrite!` / `ioctl_write_ptr!` macros from the `nix` crate.
#[repr(u32)]
pub enum Cmd {
    /// Submit a decode-step batch descriptor (seq ids + KV block ids).
    SubmitBatch = 1,
    /// Block until the given fence value has been signaled by hardware.
    WaitFence = 2,
    /// Query free HBM block count for backpressure decisions.
    QueryFreeBlocks = 3,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchDescriptor {
    pub batch_id: u64,
    pub num_seqs: u32,
    /// Offset into the mmap'd command ring where the per-seq block-id list
    /// for this batch has already been written by the caller.
    pub ring_offset: u32,
}

pub trait DeviceHandle: Send + Sync {
    fn submit_batch(&self, desc: BatchDescriptor) -> io::Result<u64 /* fence */>;
    fn wait_fence(&self, fence: u64) -> io::Result<()>;
    fn free_blocks(&self) -> io::Result<u32>;
    /// Raw pointer into the mmap'd command ring (device-visible shared
    /// memory). Real implementation: `mmap(fd, MAP_SHARED, PROT_READ|PROT_WRITE)`
    /// over the offset the driver reports via `QueryFreeBlocks`'s sibling
    /// ioctl; unsafe because callers must respect the ring's producer/
    /// consumer protocol to avoid racing the device.
    unsafe fn ring_ptr(&self) -> *mut u8;
}

/// Fake device backed by an in-process buffer instead of a real
/// `/dev/fractile0` node + `ioctl(2)`/`mmap(2)` calls, so this crate
/// builds and tests without kernel-side hardware. The real implementation
/// slots in below it (commented) to show exactly what changes.
pub struct FakeDevice {
    ring: parking_lot::Mutex<Vec<u8>>,
    next_fence: std::sync::atomic::AtomicU64,
    free_blocks: std::sync::atomic::AtomicU32,
}

impl FakeDevice {
    pub fn new(ring_bytes: usize, total_blocks: u32) -> Self {
        Self {
            ring: parking_lot::Mutex::new(vec![0u8; ring_bytes]),
            next_fence: std::sync::atomic::AtomicU64::new(0),
            free_blocks: std::sync::atomic::AtomicU32::new(total_blocks),
        }
    }
}

impl DeviceHandle for FakeDevice {
    fn submit_batch(&self, desc: BatchDescriptor) -> io::Result<u64> {
        use std::sync::atomic::Ordering;
        // real driver: ioctl(fd, IOCTL_SUBMIT_BATCH, &desc as *const _)
        let fence = self.next_fence.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = desc; // descriptor would be validated + copied into the ring here
        Ok(fence)
    }

    fn wait_fence(&self, _fence: u64) -> io::Result<()> {
        // real driver: ioctl(fd, IOCTL_WAIT_FENCE, &fence) blocks in-kernel
        // on the device's completion interrupt; here it's already "done".
        Ok(())
    }

    fn free_blocks(&self) -> io::Result<u32> {
        use std::sync::atomic::Ordering;
        Ok(self.free_blocks.load(Ordering::SeqCst))
    }

    unsafe fn ring_ptr(&self) -> *mut u8 {
        self.ring.lock().as_mut_ptr()
    }
}

/// Sketch of the real path (not compiled — depends on a physical device
/// node). Left in place to show the intended shape:
///
/// ```ignore
/// use nix::sys::mman::{mmap, MapFlags, ProtFlags};
/// use nix::{ioctl_readwrite, ioctl_read};
///
/// ioctl_readwrite!(fractile_submit_batch, b'F', 1, BatchDescriptor);
/// ioctl_read!(fractile_query_free_blocks, b'F', 3, u32);
///
/// pub struct RealDevice { fd: std::fs::File, bar: *mut u8, bar_len: usize }
///
/// impl RealDevice {
///     pub fn open(path: &str, bar_len: usize) -> io::Result<Self> {
///         let fd = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
///         let bar = unsafe {
///             mmap(None, bar_len.try_into().unwrap(), ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
///                  MapFlags::MAP_SHARED, &fd, 0)?
///         } as *mut u8;
///         Ok(Self { fd, bar, bar_len })
///     }
/// }
/// ```
struct _RealDeviceSketch;

pub struct NullFdWrapper(pub RawFd);
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
        let fence = dev
            .submit_batch(BatchDescriptor {
                batch_id: 1,
                num_seqs: 8,
                ring_offset: 0,
            })
            .unwrap();
        dev.wait_fence(fence).unwrap();
        assert_eq!(fence, 1);
    }
}
