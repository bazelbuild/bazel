// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;

/**
 * A concurrency primitive for managing access to at most K unique things at once, for a fixed K.
 *
 * <p>You can think of this as a pair of a {@link Semaphore} with K total permits and a
 * {@link Multiset}, with permits being doled out and returned based on the current contents of the
 * {@link Multiset}.
 */
@ThreadSafe
public abstract class MultisetSemaphore<T> {
  /**
   * Blocks until permits are available for all the values in {@code valuesToAcquire}, and then
   * atomically acquires these permits.
   *
   * <p>{@code acquireAll(valuesToAcquire)} atomically does the following
   * <ol>
   *   <li>Computes {@code m}, the number of values in {@code valuesToAcquire} that are not
   *   currently in the backing {@link Multiset}.
   *   <li>Adds {@code valuesToAcquire} to the backing {@link Multiset}.
   *   <li>Blocks until {@code m} permits are available from the backing {@link Semaphore}.
   *   <li>Acquires these permits.
   * </ol>
   */
  public abstract void acquireAll(Set<T> valuesToAcquire) throws InterruptedException;

  /**
   * Atomically releases permits for all the values in {@code valuesToAcquire}.
   *
   * <p>{@code releaseAll(valuesToRelease)} atomically does the following
   * <ol>
   *   <li>Computes {@code m}, the number of values in {@code valuesToRelease} that are currently in
   *   the backing {@link Multiset} with multiplicity 1.
   *   <li>Removes {@code valuesToRelease} from the backing {@link Multiset}.
   *   <li>Release {@code m} permits from the backing {@link Semaphore}.
   * </ol>
   *
   * <p>Assumes that this {@link MultisetSemaphore} has already given out permits for all the
   * values in {@code valuesToAcquire}.
   */
  public abstract void releaseAll(Set<T> valuesToRelease);

  /**
   * Returns a {@link MultisetSemaphore} with a backing {@link Semaphore} that has an unbounded
   * number of permits; that is, {@link #acquireAll} will never block.
   */
  public static <T> MultisetSemaphore<T> unbounded() {
    return UnboundedMultisetSemaphore.instance();
  }

  /** Builder for {@link MultisetSemaphore} instances. */
  public static class Builder {
    private static final int UNSET_INT = -1;

    private int maxNumUniqueValues = UNSET_INT;
    private MapMaker mapMaker = new MapMaker();

    private Builder() {
    }

    /**
     * Sets the maximum number of unique values for which permits can be held at once in the
     * to-be-constructed {@link MultisetSemaphore}.
     */
    public Builder maxNumUniqueValues(int maxNumUniqueValues) {
      Preconditions.checkState(
          maxNumUniqueValues > 0,
          "maxNumUniqueValues must be positive (was %d)",
          maxNumUniqueValues);
      this.maxNumUniqueValues = maxNumUniqueValues;
      return this;
    }

    /**
     * Sets the concurrency level (expected number of concurrent usages) of internal data structures
     * of the to-be-constructed {@link MultisetSemaphore}.
     *
     * <p>This is a hint for tweaking performance and lock contention.
     */
    public Builder concurrencyLevel(int concurrencyLevel) {
      mapMaker = mapMaker.concurrencyLevel(concurrencyLevel);
      return this;
    }

    public <T> MultisetSemaphore<T> build() {
      Preconditions.checkState(
          maxNumUniqueValues != UNSET_INT,
          "maxNumUniqueValues(int) must be specified");
      return new BoundedMultisetSemaphore<>(maxNumUniqueValues, mapMaker);
    }
  }

  /** Returns a fresh {@link Builder}. */
  public static Builder newBuilder() {
    return new Builder();
  }

  private static class UnboundedMultisetSemaphore<T> extends MultisetSemaphore<T> {
    private static final UnboundedMultisetSemaphore<Object> INSTANCE =
        new UnboundedMultisetSemaphore<Object>();

    private UnboundedMultisetSemaphore() {
    }

    @SuppressWarnings("unchecked")
    private static <T> UnboundedMultisetSemaphore<T> instance() {
      return (UnboundedMultisetSemaphore<T>) INSTANCE;
    }

    @Override
    public void acquireAll(Set<T> valuesToAcquire) throws InterruptedException {
    }

    @Override
    public void releaseAll(Set<T> valuesToRelease) {
    }
  }

  private static class BoundedMultisetSemaphore<T> extends MultisetSemaphore<T> {
    // Implementation strategy:
    //
    // We have a single Semaphore, access to which is managed by two levels of Multisets, the first
    // of which is an approximate accounting of the current multiplicities, and the second of which
    // is an accurate accounting of the current multiplicities. The first level is used to decide
    // how many permits to acquire from the semaphore on acquireAll and the second level is used to
    // decide how many permits to release from the semaphore on releaseAll. The separation between
    // these two levels ensure the atomicity of acquireAll and releaseAll.

    // We also have a map of CountDownLatches, used to handle the case where there is a not-empty
    // set that is a subset of the set of values for which multiple threads are concurrently trying
    // to acquire permits.

    private final Semaphore semaphore;
    private final ConcurrentHashMultiset<T> tentativeValues;
    private final ConcurrentHashMultiset<T> actualValues;
    private final ConcurrentMap<T, CountDownLatch> latches;

    private BoundedMultisetSemaphore(int maxNumUniqueValues, MapMaker mapMaker) {
      this.semaphore = new Semaphore(maxNumUniqueValues);
      // TODO(nharmata): Use ConcurrentHashMultiset#create(ConcurrentMap<E, AtomicInteger>) when
      // Bazel is switched to use a more recent version of Guava. Until then we'll have unnecessary
      // contention when using these Multisets.
      this.tentativeValues = ConcurrentHashMultiset.create();
      this.actualValues = ConcurrentHashMultiset.create();
      this.latches = mapMaker.makeMap();
    }

    @Override
    public void acquireAll(Set<T> valuesToAcquire) throws InterruptedException {
      int numValuesToAcquire = valuesToAcquire.size();
      HashMap<T, CountDownLatch> latchesToCountDownByValue =
          Maps.newHashMapWithExpectedSize(numValuesToAcquire);
      ArrayList<CountDownLatch> latchesToAwait = new ArrayList<>(numValuesToAcquire);
      for (T value : valuesToAcquire) {
        int oldCount = tentativeValues.add(value, 1);
        if (oldCount == 0) {
          // The value was just uniquely added by us.
          CountDownLatch latch = new CountDownLatch(1);
          Preconditions.checkState(latches.put(value, latch) == null, value);
          latchesToCountDownByValue.put(value, latch);
        } else {
          CountDownLatch latch = latches.get(value);
          if (latch != null) {
            // The value was recently added by another thread, and that thread is still waiting to
            // acquire a permit for it.
            latchesToAwait.add(latch);
          }
        }
      }

      int numUniqueValuesToAcquire = latchesToCountDownByValue.size();
      semaphore.acquire(numUniqueValuesToAcquire);
      for (T value : valuesToAcquire) {
        actualValues.add(value);
      }
      for (Map.Entry<T, CountDownLatch> entry : latchesToCountDownByValue.entrySet()) {
        T value = entry.getKey();
        CountDownLatch latchToCountDown = entry.getValue();
        latchToCountDown.countDown();
        Preconditions.checkState(latchToCountDown == latches.remove(value), value);
      }
      for (CountDownLatch latchToAwait : latchesToAwait) {
        latchToAwait.await();
      }
    }

    @Override
    public void releaseAll(Set<T> valuesToRelease) {
      int numUniqueValuesToRelease = 0;
      for (T value : valuesToRelease) {
        int oldCount = actualValues.remove(value, 1);
        Preconditions.checkState(oldCount >= 0, "%d %s", oldCount, value);
        if (oldCount == 1) {
          numUniqueValuesToRelease++;
        }
        tentativeValues.remove(value, 1);
      }

      semaphore.release(numUniqueValuesToRelease);
    }
  }
}
