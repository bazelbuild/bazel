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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ReaderPreferringReadWriteLock;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Synchronizes producers that replace their outputs in place with the consumers reading them.
 *
 * <p>Keys identify producers: rewound actions keyed by their {@code ActionLookupData} and external
 * repository fetches keyed by their {@code RepositoryName}, each kind in its own instance. A
 * producer takes its key's write lock before replacing its outputs, while a consumer takes read
 * locks for the keys of the producers of all its inputs, or the lock of a single producer for the
 * duration of a single read of its outputs.
 *
 * <p>A consumer acquires its entire set of read locks in one call, before reading any input, and
 * never holds a partial set while it waits: a producer may hold its write lock while it waits for
 * a dependency to be rebuilt, whose producer must not have to wait for the consumer in turn. A
 * producer takes only its own write lock, before it takes any read lock or waits for a dependency.
 * See the proof of deadlock freedom in RemoteRewoundActionSynchronizer for what else this relies
 * on.
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

  // The per-key locks that are switched to when the first producer acquires its write lock. They
  // keep producers from replacing outputs that are still being read while still allowing producers
  // and consumers to run concurrently (i.e., not force the equivalent of --jobs=1 for as long as a
  // producer is replacing its outputs, as the coarse lock would).
  //
  // ReaderPreferringReadWriteLock is used as java.util.concurrent locks may queue readers behind
  // waiting writers even while other readers hold the lock. A reader thus only ever waits for a
  // writer that holds the lock, which keeps a consumer that waits for reads on its behalf on other
  // threads from deadlocking with a writer that waits for the consumer (see the proof in
  // RemoteRewoundActionSynchronizer).
  //
  // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
  // are no longer needed.
  private final LoadingCache<Object, ReaderPreferringReadWriteLock> locks =
      Caffeine.newBuilder().weakValues().build(unused -> new ReaderPreferringReadWriteLock());
  private final ConcurrentHashMap<Object, ReaderPreferringReadWriteLock> writeLocksKeptForRestart =
      new ConcurrentHashMap<>();

  private volatile Replacements replacements = Replacements.NOT_YET;

  // A single coarse lock is used to synchronize producers (writers) and consumers (readers) as long
  // as no producer has acquired its write lock. Consumers hold its read lock, which is how the
  // first producer waits for them without knowing their keys.
  // This ensures high throughput and low memory footprint for the common case of no replacements.
  // In this case, there won't be any writers and acquiring the read lock amounts to incrementing an
  // atomic counter.
  // Note that it wouldn't be correct to only start using this lock once a producer replaces its
  // outputs, because a consumer of its outputs could have already started reading them.
  // Consumers that arrive while the first producer waits for the holders of this lock take per-key
  // locks right away instead of queuing behind the producer: a read on a holder's behalf on another
  // thread would otherwise wait for the producer, which waits for the holder. Admitting them
  // instead would starve the producer.
  @Nullable
  private volatile ReaderPreferringReadWriteLock coarseLock = new ReaderPreferringReadWriteLock();

  /**
   * Resets to the state at the beginning of a command, during which producers can only replace
   * their outputs if {@code replacementsEnabled}.
   */
  public void reset(boolean replacementsEnabled) {
    replacements = replacementsEnabled ? Replacements.NOT_YET : Replacements.NEVER;
    coarseLock = new ReaderPreferringReadWriteLock();
  }

  /** Releases the write locks that reset evaluations kept but that were never restarted. */
  public void releaseWriteLocksKeptForRestart() {
    writeLocksKeptForRestart.forEach(
        (key, lock) -> {
          if (writeLocksKeptForRestart.remove(key, lock)) {
            lock.unlockWrite();
          }
        });
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
   * supply if any producer may replace its outputs. The keys may repeat.
   */
  public SilentCloseable acquireReadLocks(Supplier<? extends Iterable<?>> keys)
      throws InterruptedException {
    SilentCloseable sharedLock = acquireSharedReadLock();
    return sharedLock != null ? sharedLock : acquireReadLocks(keys.get());
  }

  /**
   * Acquires the read lock for the producer of a single input, which {@code key} only has to supply
   * if any producer may replace its outputs.
   */
  public SilentCloseable acquireReadLock(Supplier<?> key) throws InterruptedException {
    SilentCloseable sharedLock = acquireSharedReadLock();
    if (sharedLock != null) {
      return sharedLock;
    }
    // The lock is strongly referenced from here on and thus outlives the cache's weak value.
    ReaderPreferringReadWriteLock readLock = locks.get(key.get());
    readLock.lockReadInterruptibly();
    return readLock::unlockRead;
  }

  /**
   * Acquires a read lock that stands in for those of all keys as long as no producer has replaced
   * its outputs, otherwise returns {@code null} so that the caller acquires per-key locks.
   */
  @Nullable
  private SilentCloseable acquireSharedReadLock() {
    if (replacements == Replacements.NEVER) {
      return () -> {};
    }
    var localCoarseLock = coarseLock;
    // A producer retires the shared lock while holding its write lock, so a consumer that acquires
    // the read lock and still finds the lock published afterwards is seen by every producer. One
    // that finds it retired, or a producer waiting for or holding it, falls through to per-key
    // locks, which is safe since it hasn't read anything yet.
    if (localCoarseLock != null && localCoarseLock.tryLockReadUnlessWriterWaiting()) {
      if (coarseLock == localCoarseLock) {
        return localCoarseLock::unlockRead;
      }
      localCoarseLock.unlockRead();
    }
    return null;
  }

  /** Acquires the exclusive lock for the given producer key. */
  public TransferableWriteLock acquireWriteLock(Object key) throws InterruptedException {
    if (replacements != Replacements.POSSIBLE) {
      // This producer creates its outputs rather than replacing ones that consumers may be reading.
      return TransferableWriteLock.noop();
    }
    ReaderPreferringReadWriteLock writeLock = writeLocksKeptForRestart.remove(key);
    if (writeLock == null) {
      var localCoarseLock = coarseLock;
      if (localCoarseLock != null) {
        // Wait for the holders of the shared lock and switch all later consumers to per-key ones.
        // Consumers that arrive during the wait already take per-key locks, so it only lasts as
        // long as the holders keep reading. If it is interrupted, the shared lock stays in place
        // and the next producer waits for the holders instead.
        localCoarseLock.lockWriteInterruptibly();
        coarseLock = null;
        localCoarseLock.unlockWrite();
      }
      // The lock is not owned by a thread, which also allows a reset Skyframe evaluation to
      // transfer it to its restarted worker thread.
      writeLock = locks.get(key);
      writeLock.lockWriteInterruptibly();
    }
    return new TransferableWriteLock(this, key, writeLock);
  }

  private SilentCloseable acquireReadLocks(Iterable<?> keys) throws InterruptedException {
    // The lookup collapses repeated keys, so a consumer whose inputs share a producer takes that
    // producer's lock only once. The locks are strongly referenced from here on and thus outlive
    // the cache's weak values.
    var readLocks = ImmutableList.copyOf(locks.getAll(keys).values());
    ReaderPreferringReadWriteLock heldByWriter;
    while ((heldByWriter = tryLockAll(readLocks)) != null) {
      // Wait for the writer without holding any other read lock, then retry the entire set. A
      // producer may hold its write lock while it waits for Skyframe to rebuild one of its
      // dependencies, whose producer would otherwise wait for this consumer's partial set.
      heldByWriter.lockReadInterruptibly();
      heldByWriter.unlockRead();
    }
    return () -> readLocks.forEach(ReaderPreferringReadWriteLock::unlockRead);
  }

  /**
   * Acquires all the given read locks and returns {@code null}, or acquires none of them and
   * returns the first one that a writer holds.
   */
  @Nullable
  private static ReaderPreferringReadWriteLock tryLockAll(
      ImmutableList<ReaderPreferringReadWriteLock> readLocks) {
    for (int i = 0; i < readLocks.size(); i++) {
      if (!readLocks.get(i).tryLockRead()) {
        for (int j = 0; j < i; j++) {
          readLocks.get(j).unlockRead();
        }
        return readLocks.get(i);
      }
    }
    return null;
  }

  private void keepWriteLockForRestart(Object key, ReaderPreferringReadWriteLock writeLock) {
    Preconditions.checkState(
        writeLocksKeptForRestart.putIfAbsent(key, writeLock) == null,
        "write lock already kept for restart: %s",
        key);
  }

  /** A write lock that can be transferred to a restarted Skyframe evaluation. */
  public static final class TransferableWriteLock implements SilentCloseable {
    @Nullable private final RewindingSynchronizer synchronizer;
    @Nullable private final Object key;
    @Nullable private final ReaderPreferringReadWriteLock writeLock;
    private boolean closed;

    private TransferableWriteLock(
        @Nullable RewindingSynchronizer synchronizer,
        @Nullable Object key,
        @Nullable ReaderPreferringReadWriteLock writeLock) {
      this.synchronizer = synchronizer;
      this.key = key;
      this.writeLock = writeLock;
    }

    /** Returns a lock that performs no synchronization. */
    public static TransferableWriteLock noop() {
      return new TransferableWriteLock(
          /* synchronizer= */ null, /* key= */ null, /* writeLock= */ null);
    }

    /** Keeps the lock held for the next evaluation of the same producer after a reset. */
    public synchronized void keepLockedForRestart() {
      if (closed) {
        return;
      }
      if (synchronizer != null) {
        synchronizer.keepWriteLockForRestart(
            Preconditions.checkNotNull(key), Preconditions.checkNotNull(writeLock));
      }
      closed = true;
    }

    @Override
    public synchronized void close() {
      if (closed) {
        return;
      }
      if (writeLock != null) {
        writeLock.unlockWrite();
      }
      closed = true;
    }
  }
}
