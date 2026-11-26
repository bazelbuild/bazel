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

//! C FFI bindings for `tracking_allocator`.

use tracking_allocator::TrackingAllocator;

/// Returns the approximate total number of bytes currently allocated across
/// all threads through the Rust global allocator (`TrackingAllocator`). See
/// [`TrackingAllocator::allocated_bytes`] for accounting scope and
/// memory-ordering semantics.
///
/// # Contract
/// - Always returns a non-negative `i64` (saturating at `i64::MAX`).
/// - Thread-safe, lock-free, allocation-free, and never panics or unwinds.
/// - Performs only a single relaxed atomic load with no callbacks, so it is safe to invoke as a
///   critical FFM downcall.
#[unsafe(no_mangle)]
pub extern "C" fn bazel_tracking_allocator_allocated_bytes() -> i64 {
    i64::try_from(TrackingAllocator::allocated_bytes()).unwrap_or(i64::MAX)
}
