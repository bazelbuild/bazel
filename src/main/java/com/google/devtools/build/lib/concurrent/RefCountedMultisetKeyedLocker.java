// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Preconditions;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.util.concurrent.Striped;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import javax.annotation.Nullable;

/**
 * An implementation of {@link KeyedLocker} that uses ref counting to efficiently only store locks
 * that are live.
 */
public class RefCountedMultisetKeyedLocker<K> implements BatchedKeyedLocker<K> {
  // Multiset of keys that have threads waiting on a lock or using a lock.
  private final ConcurrentHashMultiset<K> waiters = ConcurrentHashMultiset.<K>create();

  private static final int NUM_STRIPES = 256;
  // For each key, gives the striped lock to use for atomically managing the waiters on that key
  // internally.
  private final Striped<Lock> waitersLocks = Striped.lazyWeakLock(NUM_STRIPES);

  // Map of key to current lock, for keys that have at least one waiting or live thread.
  private final ConcurrentMap<K, ReentrantLock> locks = new ConcurrentHashMap<>();

  // Used to enforce a consistent ordering in lockBatch.
  @Nullable
  private final Comparator<K> comparator;

  private RefCountedMultisetKeyedLocker(Comparator<K> comparator) {
    this.comparator = comparator;
  }

  /** Factory for {@link RefCountedMultisetKeyedLocker} instances. */ 
  public static class Factory<K> implements BatchedKeyedLocker.Factory<K> {
    @Override
    public BatchedKeyedLocker<K> create(Comparator<K> comparator) {
      return new RefCountedMultisetKeyedLocker<>(comparator);
    }

    public KeyedLocker<K> create() {
      return new RefCountedMultisetKeyedLocker<>(/*comparator=*/null);
    }
  }

  private abstract static class AtMostOnceAutoUnlockerBase<K> implements AutoUnlocker {
    private final AtomicBoolean closeCalled = new AtomicBoolean(false);

    @Override
    public final void close() {
      if (closeCalled.getAndSet(true)) {
        String msg = "'close' can be called at most once per AutoUnlocker instance";
        throw new IllegalUnlockException(msg);
      }
      doClose();
    }

    protected abstract void doClose();
  }

  private class RefCountedAutoUnlocker extends AtMostOnceAutoUnlockerBase<K> {
    private final K key;
    private final ReentrantLock lock;

    private RefCountedAutoUnlocker(K key, ReentrantLock lock) {
      this.key = key;
      this.lock = lock;
    }

    @Override
    protected void doClose() {
      if (!lock.isHeldByCurrentThread()) {
        String msg = String.format("For key %s, the calling thread to 'close' must be the one "
            + "that acquired the AutoUnlocker", key);
        throw new IllegalUnlockException(msg);
      }
      try {
        Lock waitersLock = waitersLocks.get(key);
        try {
          waitersLock.lock();
          // Note that ConcurrentHashMultiset automatically removes removes entries for keys whose
          // count is 0.
          waiters.remove(key);
          if (waiters.count(key) == 0) {
            // No other thread is waiting to access this key, nor does the current thread have
            // another AutoUnlocker instance, so we garbage collect the lock.
            Preconditions.checkState(locks.remove(key, lock), key);
          }
        } finally {
          waitersLock.unlock();
        }
      } finally {
        lock.unlock();
      }
    }
  }

  @Override
  public AutoUnlocker lock(K key) {
    ReentrantLock newLock = new ReentrantLock();
    // Pre-lock our fresh lock, in case we win the race to get access to 'key'.
    newLock.lock();
    Lock waitersLock = waitersLocks.get(key);
    try {
      waitersLock.lock();
      // Add us to the set of waiters, so that in case we lose the race to access 'key', the winner
      // will know that we are waiting. If we already have access to 'key', this simply bumps up
      // the ref count.
      waiters.add(key);
    } finally {
      waitersLock.unlock();
    }
    ReentrantLock lock;
    lock = locks.putIfAbsent(key, newLock);
    if (lock != null) {
      Preconditions.checkState(lock != newLock);
      newLock.unlock();
      // Either another thread won the race to get access to 'key', or we already have exclusive
      // access to 'key'. Either way, we lock; in the former case we wait for our turn and in the
      // latter case the lock's implicit counter is incremented.
      lock.lock();
      return new RefCountedAutoUnlocker(key, lock);
    }
    // We won the race, so the current lock for 'key' is the one we locked and inserted.
    return new RefCountedAutoUnlocker(key, newLock);
  }

  private static void unlockAll(Iterable<KeyedLocker.AutoUnlocker> unlockers) {
    // Note that order doesn't matter here because we always acquire locks in a consistent order.
    for (KeyedLocker.AutoUnlocker unlocker : unlockers) {
      unlocker.close();
    }
  }

  @Override
  public AutoUnlocker lockBatch(Iterable<K> keys) {
    // This indicates the client did some unsafe casting - not our problem.
    Preconditions.checkNotNull(comparator);
    // We acquire locks in a consistent order. This prevents a deadlock that would otherwise occur
    // on two concurrent calls to lockBatch(keys(k1, k2)) if the callers acquired the locks in a
    // different order.
    ImmutableSortedSet<K> sortedKeys = ImmutableSortedSet.copyOf(comparator, keys);
    final List<KeyedLocker.AutoUnlocker> unlockers = new ArrayList<>(sortedKeys.size());
    boolean success = false;
    try {
      for (K key : sortedKeys) {
        unlockers.add(lock(key));
      }
      success = true;
      return new AtMostOnceAutoUnlockerBase<K>() {
        @Override
        public void doClose() {
          unlockAll(unlockers);
        }
      };
    } finally {
      // Just in case we encounter a crash, e.g. if there is a bug in #lock.
      if (!success) {
        unlockAll(unlockers);
      }
    }
  }
}
