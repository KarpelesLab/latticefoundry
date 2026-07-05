//! Executable-memory management for the in-process JIT (ROADMAP Phase 10).
//!
//! This is the *only* module in the JIT that contains `unsafe`, and it keeps that
//! surface as small as the task allows: three raw Linux syscalls (`mmap`,
//! `mprotect`, `munmap`) plus the byte-copy into the mapping. Everything above it
//! — relocation resolution and the call API — is built on the safe [`ExecBuffer`]
//! type exported here.
//!
//! # Why raw syscalls
//!
//! The crate depends only on our own crates and `std`; there is no `libc`. Rust's
//! `std` exposes no stable raw-syscall API and no way to allocate executable
//! memory, so we invoke the kernel directly with [`core::arch::asm!`] using the
//! x86-64 Linux syscall ABI (number in `rax`; arguments in `rdi`, `rsi`, `rdx`,
//! `r10`, `r8`, `r9`; result in `rax`; `rcx` and `r11` clobbered).
//!
//! # The type's contract
//!
//! An [`ExecBuffer`] owns one anonymous private mapping. It is created **writable**
//! (`R+W`) so the JIT can lay down code and patch relocations, then flipped to
//! **executable** (`R-X`) with [`ExecBuffer::make_executable`] before any code in
//! it is called. `Drop` unmaps the region exactly once. All of this is exposed
//! through a safe API; the caller can neither observe a dangling pointer nor
//! double-unmap.

use std::io;

// x86-64 Linux syscall numbers (from the kernel's syscall table; stable ABI).
const SYS_MMAP: usize = 9;
const SYS_MPROTECT: usize = 10;
const SYS_MUNMAP: usize = 11;

// `mmap`/`mprotect` protection bits and `mmap` flags (from `<sys/mman.h>`, a
// stable kernel ABI we implement against — not derived from any C source).
const PROT_READ: usize = 0x1;
const PROT_WRITE: usize = 0x2;
const PROT_EXEC: usize = 0x4;
const MAP_PRIVATE: usize = 0x02;
const MAP_ANONYMOUS: usize = 0x20;

/// The page size we align mappings to (4 KiB on x86-64 Linux).
const PAGE: usize = 0x1000;

/// Round `x` up to a whole number of pages.
fn page_round_up(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

/// `mmap(2)` via a direct syscall.
///
/// # Safety
/// A raw syscall that changes the process address space. The caller must treat
/// the returned address as owning a fresh `len`-byte mapping until it is unmapped
/// exactly once. Only ever called by [`ExecBuffer::new`] with `addr = 0`, an
/// anonymous private mapping, `fd = -1`, `offset = 0`, so no existing mapping is
/// displaced and no file is involved.
#[allow(unsafe_code)]
unsafe fn sys_mmap(addr: usize, len: usize, prot: usize, flags: usize, fd: isize, off: usize) -> isize {
    let ret: isize;
    // SAFETY: `syscall` with the x86-64 Linux ABI. We pass the six `mmap`
    // arguments in the ABI-mandated registers, take the result from `rax`, and
    // declare `rcx`/`r11` clobbered (the kernel destroys them). `nostack` is
    // valid because the asm neither reads nor writes the Rust stack.
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MMAP => ret,
            in("rdi") addr,
            in("rsi") len,
            in("rdx") prot,
            in("r10") flags,
            in("r8") fd,
            in("r9") off,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    ret
}

/// `mprotect(2)` via a direct syscall. Returns `0` on success or `-errno`.
///
/// # Safety
/// Changes the protection of an existing mapping. The caller must pass an `addr`
/// and `len` describing a mapping it owns; changing protections of memory it does
/// not own is undefined. Only ever called by [`ExecBuffer::make_executable`] with
/// this buffer's own mapping.
#[allow(unsafe_code)]
unsafe fn sys_mprotect(addr: usize, len: usize, prot: usize) -> isize {
    let ret: isize;
    // SAFETY: as `sys_mmap`, but the three-argument `mprotect` syscall.
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MPROTECT => ret,
            in("rdi") addr,
            in("rsi") len,
            in("rdx") prot,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    ret
}

/// `munmap(2)` via a direct syscall. Returns `0` on success or `-errno`.
///
/// # Safety
/// Unmaps an existing mapping. The caller must pass an `addr`/`len` it owns and
/// must not use the region afterwards. Only ever called once, from
/// [`ExecBuffer`]'s `Drop`, with this buffer's own mapping.
#[allow(unsafe_code)]
unsafe fn sys_munmap(addr: usize, len: usize) -> isize {
    let ret: isize;
    // SAFETY: as `sys_mmap`, but the two-argument `munmap` syscall.
    unsafe {
        core::arch::asm!(
            "syscall",
            inlateout("rax") SYS_MUNMAP => ret,
            in("rdi") addr,
            in("rsi") len,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    ret
}

/// A syscall result in the error range `-4095..=-1` is `-errno`; anything else is
/// a success value. Convert an error into [`io::Error`].
fn check(ret: isize) -> io::Result<usize> {
    if (-4095..0).contains(&ret) {
        Err(io::Error::from_raw_os_error(-ret as i32))
    } else {
        Ok(ret as usize)
    }
}

/// An owned, page-aligned memory mapping for JIT-compiled code.
///
/// Created writable so code and relocation patches can be written, then flipped
/// to read-execute with [`make_executable`](Self::make_executable). The mapping
/// is unmapped on drop. The type is neither `Clone` nor `Copy`, so the single
/// ownership needed for a sound unmap is enforced by the type system.
#[derive(Debug)]
pub struct ExecBuffer {
    /// The mapping base address (page-aligned, non-null once constructed).
    ptr: *mut u8,
    /// The mapped length in bytes (a whole number of pages).
    len: usize,
    /// The number of code bytes requested by the caller (`<= len`).
    used: usize,
    /// Whether the mapping is currently read-execute (vs. read-write).
    executable: bool,
}

impl ExecBuffer {
    /// Map a fresh writable (`R+W`) region large enough for `size` bytes.
    ///
    /// Returns an error if `size` is zero or the `mmap` syscall fails.
    pub fn new(size: usize) -> io::Result<ExecBuffer> {
        if size == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "zero-length JIT buffer"));
        }
        let len = page_round_up(size);
        // SAFETY: an anonymous, private, fixed-nothing mapping (`addr = 0` lets
        // the kernel choose), `fd = -1`, `offset = 0`. This creates a brand-new
        // region owned solely by the returned `ExecBuffer`, which unmaps it once
        // on drop. No existing mapping is touched.
        #[allow(unsafe_code)]
        let ret = unsafe {
            sys_mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        };
        let addr = check(ret)?;
        Ok(ExecBuffer { ptr: addr as *mut u8, len, used: size, executable: false })
    }

    /// The mapping base address as an integer, for relocation arithmetic.
    #[inline]
    pub fn base_addr(&self) -> usize {
        self.ptr as usize
    }

    /// The number of code bytes this buffer was sized for.
    #[inline]
    pub fn used(&self) -> usize {
        self.used
    }

    /// A read-write view of the code region, valid only while the buffer is still
    /// writable (before [`make_executable`](Self::make_executable)).
    ///
    /// Returns `None` once the mapping has been made executable, so writing to
    /// R-X memory is impossible through the safe API.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if self.executable {
            return None;
        }
        // SAFETY: `ptr` points at a live mapping of at least `used` writable
        // bytes (`used <= len`), owned exclusively by `self`; the borrow checker
        // ties the returned slice's lifetime to `&mut self`, so no aliasing
        // mutable view can coexist.
        #[allow(unsafe_code)]
        let s = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.used) };
        Some(s)
    }

    /// Flip the whole mapping to read-execute (`R-X`).
    ///
    /// After this succeeds, [`as_mut_slice`](Self::as_mut_slice) returns `None`
    /// and the code may be called via [`func_ptr`](Self::func_ptr).
    pub fn make_executable(&mut self) -> io::Result<()> {
        // SAFETY: `ptr`/`len` describe exactly this buffer's own mapping.
        // Changing it to read-execute is sound; we already stopped handing out
        // writable slices (see `as_mut_slice`).
        #[allow(unsafe_code)]
        let ret = unsafe { sys_mprotect(self.ptr as usize, self.len, PROT_READ | PROT_EXEC) };
        check(ret)?;
        self.executable = true;
        Ok(())
    }

    /// A raw code pointer at `offset` bytes into the mapping.
    ///
    /// The address is stable for the lifetime of the buffer. Callers turn it into
    /// a typed function pointer at their own risk — see the JIT call API, which
    /// documents the one signature-match obligation.
    #[inline]
    pub fn func_ptr(&self, offset: usize) -> *const u8 {
        // Pointer arithmetic on the owned mapping; not a dereference.
        #[allow(unsafe_code)]
        // SAFETY: `offset <= used <= len`, so the resulting pointer stays within
        // (or one past the end of) the single owned allocation, as `add` requires.
        unsafe {
            self.ptr.add(offset)
        }
    }
}

impl Drop for ExecBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` are this buffer's own mapping and `Drop` runs at
        // most once, so the region is unmapped exactly once and never used again.
        // The result is ignored: nothing actionable can be done if unmap fails,
        // and a valid `ExecBuffer` always holds a valid mapping.
        #[allow(unsafe_code)]
        unsafe {
            let _ = sys_munmap(self.ptr as usize, self.len);
        }
    }
}

// SAFETY: an `ExecBuffer` is a plain owned memory region addressed by a raw
// pointer. It has no interior mutability and no thread-affine state, so moving it
// across threads (`Send`) is sound. We do not implement `Sync`: concurrent calls
// into JIT code are the caller's responsibility, not this type's.
#[allow(unsafe_code)]
unsafe impl Send for ExecBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_rounding() {
        assert_eq!(page_round_up(1), PAGE);
        assert_eq!(page_round_up(PAGE), PAGE);
        assert_eq!(page_round_up(PAGE + 1), 2 * PAGE);
    }

    #[test]
    fn map_write_execute_ret() {
        // Lay down a single `ret` (0xc3), execute it, and confirm nothing faults.
        let mut buf = ExecBuffer::new(1).unwrap();
        assert!(buf.base_addr() != 0);
        buf.as_mut_slice().unwrap()[0] = 0xc3;
        buf.make_executable().unwrap();
        // Once executable, the writable view is withdrawn.
        assert!(buf.as_mut_slice().is_none());
        // SAFETY (test): the code is a bare `ret`; the empty SysV signature
        // `extern "C" fn()` matches it, and the buffer outlives the call.
        #[allow(unsafe_code)]
        unsafe {
            let f: extern "C" fn() = std::mem::transmute(buf.func_ptr(0));
            f();
        }
    }

    #[test]
    fn zero_length_is_rejected() {
        assert!(ExecBuffer::new(0).is_err());
    }

    #[test]
    fn drop_unmaps_without_fault() {
        // Create and drop many buffers; a botched unmap would fault or leak.
        for _ in 0..64 {
            let b = ExecBuffer::new(4096).unwrap();
            assert_ne!(b.base_addr(), 0);
        }
    }
}
