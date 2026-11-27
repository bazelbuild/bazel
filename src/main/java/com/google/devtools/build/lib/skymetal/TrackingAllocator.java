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

package com.google.devtools.build.lib.skymetal;

import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.invoke.MethodHandle;

/** Java binding for the Rust global {@code TrackingAllocator}. */
public final class TrackingAllocator {

  /**
   * Returns the number of bytes currently allocated through the Rust global allocator across all
   * threads, as an unsynchronized snapshot. The returned value is always non-negative.
   *
   * <p>Only the sizes requested by Rust code linked into the server's native library are counted;
   * allocator overhead and non-Rust native allocations (C/C++ heap, {@code mmap}) are excluded.
   */
  public static long getAllocatedBytes() {
    return Native.getAllocatedBytes();
  }

  private static final class Native {
    // Safe as a critical downcall: bazel_tracking_allocator_allocated_bytes performs only a single
    // relaxed atomic load with no allocations, locks, or JVM callbacks.
    private static final MethodHandle ALLOCATED_BYTES =
        SkymetalBridge.downcallHandle(
            "bazel_tracking_allocator_allocated_bytes",
            FunctionDescriptor.of(JAVA_LONG),
            Linker.Option.critical(/* allowHeapAccess= */ false));

    static long getAllocatedBytes() {
      try {
        return (long) ALLOCATED_BYTES.invokeExact();
      } catch (Throwable t) {
        throw new AssertionError(t);
      }
    }

    private Native() {}
  }

  private TrackingAllocator() {}
}
