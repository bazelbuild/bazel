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
import com.google.common.util.concurrent.Striped;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * An implementation of {@link KeyedLocker} that uses ref counting to efficiently only store locks
 * that are live.
 */
public class RefCountedMultisetKeyedLocker<K> implements KeyedLocker<K> {
  // Multiset of keys that have threads waiting on a lock or using a lock.
  private final ConcurrentHashMultiset<K> waiters = ConcurrentHashMultiset.<K>create();

  private static final int NUM_STRIPES = 2048;
  // For each key, gives the striped lock to use for atomically managing the waiters on that key
  // internally.
  private final Striped<Lock> waitersLocks = Striped.lazyWeakLock(NUM_STRIPES);

  // Map of key to current lock, for keys that have at least one waiting or live thread.
  private final ConcurrentMap<K, ReentrantLock> locks =
      new ConcurrentHashMap<>(1024, .75f, NUM_STRIPES);

  public RefCountedMultisetKeyedLocker() {
  }

  private class RefCountedAutoUnlocker implements AutoUnlocker {
    private final K key;
    private final ReentrantLock lock;
    private final AtomicBoolean closeCalled = new AtomicBoolean(false);

    private RefCountedAutoUnlocker(K key, ReentrantLock lock) {
      this.key = key;
      this.lock = lock;
    }

    @Override
    public void close() {
      if (closeCalled.getAndSet(true)) {
        String msg = String.format("For key %s, 'close' can be called at most once per "
                                       + "AutoUnlocker instance", key);
        throw new IllegalUnlockException(msg);
      }
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
          if (waiters.remove(key, 1) == 1) {
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
}
