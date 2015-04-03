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

import static org.junit.Assert.assertEquals;

import com.google.common.base.Supplier;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.AtomicLongMap;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker;

import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/** Base class for tests for {@link BatchedKeyedLocker} implementations. */
public abstract class BatchedKeyedLockerTest extends KeyedLockerTest {
  private BatchedKeyedLocker<String> batchLocker;

  protected abstract BatchedKeyedLocker.Factory<String> getLockerFactory();

  @Override
  protected BatchedKeyedLocker<String> makeFreshLocker() {
    return getLockerFactory().create(Ordering.<String>natural());
  }

  @Before
  public void setUp_BatchedKeyedLockerTest() {
    batchLocker = makeFreshLocker();
  }

  private Supplier<AutoUnlocker> makeBatchLockInvoker(Set<String> keys) {
    // The collection returned by Collections2#permutations isn't thread-safe.
    ImmutableList<List<String>> perms = ImmutableList.copyOf(Collections2.permutations(keys));
    final Iterator<List<String>> permutationsIter = Iterables.cycle(perms).iterator();
    return new Supplier<KeyedLocker.AutoUnlocker>() {
      @Override
      public AutoUnlocker get() {
        // We call lockBatch with a set whose iteration order is different each time, for the sake
        // of trying to tickle hypothetical concurrency bugs resulting from bad KeyedLocker
        // implementations not being careful about ordering.
        return batchLocker.lockBatch(ImmutableSet.copyOf(permutationsIter.next()));
      }
    };
  }

  private Supplier<AutoUnlocker> makeBatchLockFn1() {
    return makeBatchLockInvoker(ImmutableSet.of("1a", "1b", "1c", "1d", "1e"));
  }

  private Supplier<AutoUnlocker> makeBatchLockFn2() {
    return makeBatchLockInvoker(ImmutableSet.of("2a", "2b", "2c", "2d", "2e"));
  }

  @Test
  public void simpleSingleThreaded_NoUnlocks_Batch() {
    runSimpleSingleThreaded_NoUnlocks(makeBatchLockFn1(), makeBatchLockFn2());
  }

  @Test
  public void simpleSingleThreaded_WithUnlocks_Batch() {
    runSimpleSingleThreaded_WithUnlocks(makeBatchLockFn1(), makeBatchLockFn2());
  }

  @Test
  public void doubleUnlockOnSameAutoUnlockerNotAllowed_Batch() {
    runDoubleUnlockOnSameAutoUnlockerNotAllowed(makeBatchLockFn1());
  }

  @Test
  public void unlockOnDifferentAutoUnlockersAllowed_Batch() {
    runUnlockOnDifferentAutoUnlockersAllowed(makeBatchLockFn1());
  }

  @Test
  public void threadLocksMultipleTimesBeforeUnlocking_Batch() throws Exception {
    runThreadLocksMultipleTimesBeforeUnlocking(makeBatchLockFn1());
  }

  @Test
  public void unlockOnOtherThreadNotAllowed_Batch() throws Exception {
    runUnlockOnOtherThreadNotAllowed(makeBatchLockFn1());
  }

  @Test
  public void refCountingSanity_Batch() {
    runRefCountingSanity(makeBatchLockFn1());
  }

  @Test
  public void simpleMultiThreaded_MutualExclusion_Batch() throws Exception {
    runSimpleMultiThreaded_MutualExclusion(makeBatchLockFn1());
  }

  @Test
  public void testMixOfLockAndLockBatch_MutualExclusion() throws Exception {
    final AtomicInteger count = new AtomicInteger(0);
    final AtomicLongMap<String> mutexCounters = AtomicLongMap.create();
    Set<Set<String>> powerSet = Sets.powerSet(
        ImmutableSet.of("k1", "k2", "k3", "k4", "k5", "k6", "k8", "k9", "k10"));
    for (final Set<String> keys : powerSet) {
      executorService.submit(new Runnable() {
        @Override
        public void run() {
          if (keys.size() == 1) {
            String key = Iterables.getOnlyElement(keys);
            try (AutoUnlocker unlocker = batchLocker.lock(key)) {
              long newCount = mutexCounters.addAndGet(key, 1);
              assertEquals(1, newCount);
              mutexCounters.decrementAndGet(key);
            }
          } else if (keys.size() > 1) {
            try (AutoUnlocker unlocker = batchLocker.lockBatch(keys)) {
              for (String key : keys) {
                long newCount = mutexCounters.addAndGet(key, 1);
                assertEquals(1, newCount);
                mutexCounters.decrementAndGet(key);
              }
            }
          }
          count.incrementAndGet();
        }
      });
    }
    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executorService);
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(powerSet.size(), count.get());
  }
}
