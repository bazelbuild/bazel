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

use std::hint::black_box;

use googletest::{expect_eq, gtest};
use tracking_allocator::TrackingAllocator;

#[gtest]
fn alloc_updates_allocated_bytes() {
    const BYTES: usize = 4096;
    let baseline = TrackingAllocator::allocated_bytes();

    let buf: Vec<u8> = black_box(Vec::with_capacity(BYTES));

    expect_eq!(TrackingAllocator::allocated_bytes(), baseline + buf.capacity());
}

#[gtest]
fn dealloc_subtracts_only_freed_bytes() {
    const FREED_BYTES: usize = 4096;
    const RETAINED_BYTES: usize = 8192;
    let baseline = TrackingAllocator::allocated_bytes();
    let retained: Vec<u8> = black_box(Vec::with_capacity(RETAINED_BYTES));

    {
        let _freed: Vec<u8> = black_box(Vec::with_capacity(FREED_BYTES));
    }

    expect_eq!(TrackingAllocator::allocated_bytes(), baseline + retained.capacity());
}

#[gtest]
fn alloc_zeroed_updates_allocated_bytes() {
    const BYTES: usize = 4096;
    let baseline = TrackingAllocator::allocated_bytes();

    let buf = black_box(vec![0u8; BYTES]);

    expect_eq!(TrackingAllocator::allocated_bytes(), baseline + buf.capacity());
    expect_eq!(buf[0], 0);
}

#[gtest]
fn realloc_grow_updates_allocated_bytes() {
    const INITIAL_BYTES: usize = 4096;
    const GROWN_BYTES: usize = 12288;
    let baseline = TrackingAllocator::allocated_bytes();
    let mut buf: Vec<u8> = Vec::with_capacity(INITIAL_BYTES);
    buf.push(0x5a);
    let mut buf = black_box(buf);

    buf.reserve_exact(GROWN_BYTES - buf.len());
    let buf = black_box(buf);

    expect_eq!(TrackingAllocator::allocated_bytes(), baseline + buf.capacity());
    expect_eq!(buf[0], 0x5a);
}

#[gtest]
fn realloc_shrink_updates_allocated_bytes() {
    const INITIAL_BYTES: usize = 12288;
    const SHRUNK_BYTES: usize = 2048;
    let baseline = TrackingAllocator::allocated_bytes();
    let mut buf: Vec<u8> = Vec::with_capacity(INITIAL_BYTES);
    buf.push(0x5a);
    let mut buf = black_box(buf);

    buf.shrink_to(SHRUNK_BYTES);
    let buf = black_box(buf);

    expect_eq!(TrackingAllocator::allocated_bytes(), baseline + buf.capacity());
    expect_eq!(buf[0], 0x5a);
}

#[gtest]
fn ffi_reports_allocated_bytes() {
    const BYTES: usize = 4096;
    let baseline = tracking_allocator_ffi::bazel_tracking_allocator_allocated_bytes();

    let buf: Vec<u8> = black_box(Vec::with_capacity(BYTES));

    expect_eq!(
        tracking_allocator_ffi::bazel_tracking_allocator_allocated_bytes(),
        baseline + i64::try_from(buf.capacity()).expect("capacity fits in i64")
    );
}
