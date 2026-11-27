// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Rust global allocator wrapper that tracks live allocated bytes.
//!
//! [`TrackingAllocator`] wraps [`std::alloc::System`] as the `#[global_allocator]`
//! so Rust continues to share the underlying C/C++ allocator while maintaining a
//! byte count of live Rust allocations for memory telemetry and JVM heap
//! coordination.
//!
//! Tracking allocations at the Rust `GlobalAlloc` boundary instead of querying
//! system allocator stats (such as TCMalloc's `generic.current_allocated_bytes`,
//! `mallinfo2`, or `malloc_zone_statistics`) isolates Rust heap usage from the
//! JVM's own C++ heap allocations (GC metadata, JIT arenas, metaspace) and works
//! uniformly across platforms (Linux, macOS, Windows). The trade-off is that
//! this tracks requested Rust `Layout::size()` bytes only; allocator overhead
//! (size-class rounding, bookkeeping, fragmentation) and non-Rust native
//! allocations (C/C++ heap, direct `mmap`) are excluded and should be observed
//! via process RSS or native heap profiling.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

// Signed so `realloc` can pass a signed `new_size - old_size` delta directly
// and to remain safe if later sharded across multiple atomics (where
// unsynchronized cross-shard sums can transiently dip below zero).
//
// TODO(chiwang): Shard this counter across a fixed static array of
// cache-line-aligned (`#[repr(align(64))]`) atomic counters in `.bss` indexed
// by Fibonacci-hashed pointer address (summing shards in `allocated_bytes()`)
// once high-rate concurrent Rust allocations make single-atomic cache-line
// bouncing measurable.
static ALLOCATED_BYTES: AtomicIsize = AtomicIsize::new(0);

/// Global allocator wrapper around [`std::alloc::System`] that tracks live Rust
/// heap bytes.
pub struct TrackingAllocator;

impl TrackingAllocator {
    /// Returns the total number of bytes currently allocated across all threads
    /// through the Rust global allocator.
    ///
    /// This sums requested `Layout::size()` bytes only; it excludes system
    /// allocator overhead and non-Rust allocations (see module documentation).
    ///
    /// The result is an unsynchronized snapshot of process-wide usage: earlier
    /// allocations and deallocations by the calling thread are always accounted
    /// for, while those on other threads may not be visible unless the caller
    /// has separately synchronized with them.
    #[must_use]
    pub fn allocated_bytes() -> usize {
        ALLOCATED_BYTES.load(Ordering::Relaxed).max(0) as usize
    }
}

// Technical justification: `GlobalAlloc` is an `unsafe trait` required by Rust
// to interpose on global heap allocations; there is no safe equivalent.
//
// SAFETY:
// - `TrackingAllocator` delegates every allocation, reallocation, and deallocation directly to
//   `std::alloc::System`, which satisfies all `GlobalAlloc` alignment, layout, zero-initialization,
//   and prefix-preservation requirements.
// - Counter updates use only lock-free atomic operations on the static `ALLOCATED_BYTES` in `.bss`,
//   so they never allocate (avoiding reentrancy) and never unwind.
unsafe impl GlobalAlloc for TrackingAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: By `<TrackingAllocator as GlobalAlloc>::alloc`'s precondition,
        // `layout` has non-zero size, discharging `System::alloc`'s precondition.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCATED_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: By `<TrackingAllocator as GlobalAlloc>::alloc_zeroed`'s
        // precondition, `layout` has non-zero size, discharging
        // `System::alloc_zeroed`'s precondition.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            ALLOCATED_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: By `<TrackingAllocator as GlobalAlloc>::dealloc`'s
        // precondition, `ptr` denotes a live block allocated via
        // `TrackingAllocator` with `layout`. Because `TrackingAllocator`
        // delegates all allocations directly to `System`, `ptr` was allocated
        // via `System` with `layout`, discharging `System::dealloc`'s
        // precondition.
        unsafe {
            System.dealloc(ptr, layout);
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: By `<TrackingAllocator as GlobalAlloc>::realloc`'s
        // precondition, `ptr` was allocated via `TrackingAllocator` (and hence
        // via `System`) with `layout`, `new_size > 0`, and `new_size` rounded
        // up to `layout.align()` does not overflow `isize::MAX`. This
        // discharges `System::realloc`'s preconditions.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let delta = (new_size as isize) - (layout.size() as isize);
            ALLOCATED_BYTES.fetch_add(delta, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[global_allocator] // allow_global_allocator
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;
