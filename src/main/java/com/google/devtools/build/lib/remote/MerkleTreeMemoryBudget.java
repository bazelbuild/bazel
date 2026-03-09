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
package com.google.devtools.build.lib.remote;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Limits the total memory used by concurrently retained Merkle trees.
 *
 * <p>Between building a remote action (which creates the Merkle tree) and uploading its inputs
 * (which allows the tree to be released), the tree must be retained in memory. With high {@code
 * --jobs} values, many trees can be retained simultaneously, potentially exhausting the heap.
 *
 * <p>This class implements a memory budget that must be acquired before building a Merkle tree and
 * released after uploading. Since the tree's size is not known until after building, the protocol is
 * three-phase:
 *
 * <ol>
 *   <li>{@link #acquire()} blocks until committed bytes are below the budget, then returns a {@link
 *       Handle}.
 *   <li>{@link #commit(Handle, int)} records the actual tree size after building.
 *   <li>{@link #release(Handle)} returns the committed bytes after uploading.
 * </ol>
 *
 * <p>The budget can be dynamically reduced via {@link #reduceMaxBytes(long)} in response to GC
 * memory pressure events.
 */
@ThreadSafe
final class MerkleTreeMemoryBudget {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** A disabled budget that never blocks. */
  static final MerkleTreeMemoryBudget DISABLED = new MerkleTreeMemoryBudget(0);

  private final long configuredMaxBytes;
  private volatile long effectiveMaxBytes;
  private final AtomicLong committedBytes = new AtomicLong();
  private final ReentrantLock lock = new ReentrantLock(/* fair= */ true);
  private final Condition budgetAvailable = lock.newCondition();

  /**
   * @param maxBytes the maximum number of bytes that can be committed concurrently, or 0 to
   *     disable.
   */
  MerkleTreeMemoryBudget(long maxBytes) {
    this.configuredMaxBytes = maxBytes;
    this.effectiveMaxBytes = maxBytes;
  }

  boolean isEnabled() {
    return configuredMaxBytes > 0;
  }

  /**
   * Acquires a budget slot, blocking until committed bytes are below the budget.
   *
   * <p>The returned handle must eventually be passed to {@link #commit} and then {@link #release}.
   * If the tree build fails, call {@link #release} directly (without commit) to relinquish the
   * slot.
   */
  Handle acquire() throws InterruptedException {
    if (!isEnabled()) {
      return Handle.DISABLED;
    }
    long maxBytes = effectiveMaxBytes;
    if (committedBytes.get() >= maxBytes) {
      lock.lockInterruptibly();
      try {
        try (var c =
            Profiler.instance()
                .profile(ProfilerTask.REMOTE_SETUP, "waiting for Merkle tree memory budget")) {
          while (committedBytes.get() >= effectiveMaxBytes) {
            budgetAvailable.await();
          }
        }
      } finally {
        lock.unlock();
      }
    }
    return new Handle();
  }

  /**
   * Records the actual size of the Merkle tree after building.
   *
   * <p>Must be called exactly once per handle, before {@link #release}.
   */
  void commit(Handle handle, int sizeBytes) {
    if (!isEnabled()) {
      return;
    }
    handle.committedBytes = sizeBytes;
    committedBytes.addAndGet(sizeBytes);
  }

  /**
   * Releases the budget held by the given handle. Wakes up threads waiting in {@link #acquire()}.
   */
  void release(Handle handle) {
    if (!isEnabled() || handle.committedBytes == 0) {
      return;
    }
    long remaining = committedBytes.addAndGet(-handle.committedBytes);
    handle.committedBytes = 0;
    if (remaining < effectiveMaxBytes) {
      lock.lock();
      try {
        budgetAvailable.signalAll();
      } finally {
        lock.unlock();
      }
    }
  }

  /**
   * Reduces the effective budget in response to memory pressure.
   *
   * <p>Does not drop below zero. If the budget was previously reduced, restores to the configured
   * maximum first if {@code newMaxBytes} is higher.
   */
  void reduceMaxBytes(long newMaxBytes) {
    if (!isEnabled()) {
      return;
    }
    long clamped = Math.max(0, Math.min(newMaxBytes, configuredMaxBytes));
    long old = effectiveMaxBytes;
    effectiveMaxBytes = clamped;
    if (clamped > old) {
      // Budget increased, wake up waiters.
      lock.lock();
      try {
        budgetAvailable.signalAll();
      } finally {
        lock.unlock();
      }
    }
    logger.atInfo().log(
        "Merkle tree memory budget adjusted: %,d -> %,d bytes (configured: %,d)",
        old, clamped, configuredMaxBytes);
  }

  /** Restores the effective budget to the configured maximum. */
  void restoreMaxBytes() {
    reduceMaxBytes(configuredMaxBytes);
  }

  long getConfiguredMaxBytes() {
    return configuredMaxBytes;
  }

  @VisibleForTesting
  long getCommittedBytes() {
    return committedBytes.get();
  }

  @VisibleForTesting
  long getEffectiveMaxBytes() {
    return effectiveMaxBytes;
  }

  /** Opaque handle that tracks the memory budget charged for a single Merkle tree. */
  static final class Handle {
    static final Handle DISABLED = new Handle();

    int committedBytes;
  }
}
