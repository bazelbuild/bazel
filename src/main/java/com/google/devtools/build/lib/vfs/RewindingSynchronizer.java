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

package com.google.devtools.build.lib.vfs;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.Collection;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Synchronizes producers that replace their outputs in place with the consumers reading them.
 *
 * <p>Keys identify producers, currently external repository fetches keyed by their {@code
 * RepositoryName}. A producer takes its key's write lock before replacing its outputs, while a
 * consumer takes read locks for the keys of the producers of all its inputs.
 *
 * <p>These locks cannot deadlock: read locks are mutually compatible and a producer only ever takes
 * its own write lock, before holding any read lock. A cycle in the wait-for graph would thus imply
 * a cycle in the producer dependency graph, which Skyframe rules out.
 *
 * <p>Commands that never replace anything don't pay for per-key locks: until {@link
 * #markReplacementsPossible}, write locks are no-ops and consumers only acquire a single shared
 * lock instead of determining the producers of their inputs. The first write lock waits for the
 * holders of that shared lock and switches all later consumers to per-key locks. A command that
 * can't rewind acquires no lock at all.
 */
public final class RewindingSynchronizer {

  /** Whether a producer may replace outputs that consumers could already be reading. */
  private enum Replacements {
    /** Not during this command, so consumers don't have to take any lock. */
    NEVER,
    /** Not so far, so write locks are no-ops and consumers only take the shared lock. */
    NOT_YET,
    /** Possibly, so write locks take effect. */
    POSSIBLE
  }

  // Weakly referenced values so that locks are cleaned up once they are no longer needed.
  private final LoadingCache<Object, StampedLock> locks =
      Caffeine.newBuilder().weakValues().build(unused -> new StampedLock());

  private volatile Replacements replacements = Replacements.NOT_YET;

  // Stands in for the locks of all keys until the first write lock is acquired. Consumers hold
  // its read lock, which is how that producer waits for them without knowing their keys.
  @Nullable private volatile ReentrantReadWriteLock coarseLock = new ReentrantReadWriteLock();

  /**
   * Resets to the state at the beginning of a command, during which producers can only replace
   * their outputs if {@code replacementsEnabled}.
   */
  public void reset(boolean replacementsEnabled) {
    replacements = replacementsEnabled ? Replacements.NOT_YET : Replacements.NEVER;
    coarseLock = new ReentrantReadWriteLock();
  }

  /**
   * Announces that a producer may replace outputs that consumers could already be reading, which
   * makes write locks take effect. Must be called before the producer starts replacing them.
   */
  public void markReplacementsPossible() {
    Preconditions.checkState(replacements != Replacements.NEVER, "replacements are disabled");
    replacements = Replacements.POSSIBLE;
  }

  /**
   * Acquires read locks for the producers of a consumer's inputs, which {@code keys} only has to
   * supply if any producer may replace its outputs.
   */
  public SilentCloseable acquireReadLocks(Supplier<? extends Collection<?>> keys)
      throws InterruptedException {
    if (replacements == Replacements.NEVER) {
      return () -> {};
    }
    var localCoarseLock = coarseLock;
    if (localCoarseLock != null) {
      Lock coarseReadLock = localCoarseLock.readLock();
      coarseReadLock.lockInterruptibly();
      // A producer only switches to per-key locks while holding the write lock, so this check is
      // authoritative.
      if (coarseLock != null) {
        return coarseReadLock::unlock;
      }
      // The switch happened first. Releasing is safe since this consumer hasn't read anything yet.
      coarseReadLock.unlock();
    }
    return acquireReadLocks(keys.get());
  }

  /** Acquires the exclusive lock for the given producer key. */
  public SilentCloseable acquireWriteLock(Object key) throws InterruptedException {
    if (replacements != Replacements.POSSIBLE) {
      // This producer creates its outputs rather than replacing ones that consumers may be reading.
      return () -> {};
    }
    var localCoarseLock = coarseLock;
    if (localCoarseLock != null) {
      // Wait for the holders of the shared lock and switch all later consumers to per-key ones.
      Lock coarseWriteLock = localCoarseLock.writeLock();
      coarseWriteLock.lockInterruptibly();
      try {
        coarseLock = null;
      } finally {
        coarseWriteLock.unlock();
      }
    }
    // StampedLock views are not thread-owned, which allows this lock to be released by a thread
    // other than the one that acquired it.
    Lock writeLock = locks.get(key).asReadWriteLock().writeLock();
    writeLock.lockInterruptibly();
    return writeLock::unlock;
  }

  /**
   * Returns whether a read lock that blocks the write lock of the given key is held, either the
   * key's own lock or the coarse lock standing in for it.
   */
  @VisibleForTesting
  public boolean hasBlockingReadLockForTesting(Object key) {
    var localCoarseLock = coarseLock;
    if (localCoarseLock != null) {
      return localCoarseLock.getReadLockCount() > 0;
    }
    return locks.get(key).isReadLocked();
  }

  private SilentCloseable acquireReadLocks(Collection<?> keys) throws InterruptedException {
    var readLocks = ImmutableList.<Lock>builderWithExpectedSize(keys.size());
    try {
      // Read locks are mutually compatible, even while a writer is queued, so acquisition order
      // doesn't matter. Deduplicate: a lock acquired twice would have to be released twice.
      for (Object key : ImmutableSet.copyOf(keys)) {
        // Unlike the views of ReentrantReadWriteLock, this view strongly references its parent
        // StampedLock and can therefore safely outlive the cache's weak value.
        Lock readLock = locks.get(key).asReadWriteLock().readLock();
        readLock.lockInterruptibly();
        readLocks.add(readLock);
      }
    } catch (InterruptedException e) {
      unlockInReverseOrder(readLocks.build());
      throw e;
    }
    ImmutableList<Lock> locksToRelease = readLocks.build();
    return () -> unlockInReverseOrder(locksToRelease);
  }

  private static void unlockInReverseOrder(ImmutableList<Lock> locks) {
    for (int i = locks.size() - 1; i >= 0; i--) {
      locks.get(i).unlock();
    }
  }
}
