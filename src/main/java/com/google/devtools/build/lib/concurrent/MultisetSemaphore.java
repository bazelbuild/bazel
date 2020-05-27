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

import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.Set;
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

  public abstract int estimateCurrentNumUniqueValues();

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

    private Builder() {
    }

    /**
     * Sets the maximum number of unique values for which permits can be held at once in the
     * to-be-constructed {@link MultisetSemaphore}.
     */
    public Builder maxNumUniqueValues(int maxNumUniqueValues) {
      Preconditions.checkState(
          maxNumUniqueValues > 0,
          "maxNumUniqueValues must be positive (was %s)",
          maxNumUniqueValues);
      this.maxNumUniqueValues = maxNumUniqueValues;
      return this;
    }

    public <T> MultisetSemaphore<T> build() {
      Preconditions.checkState(
          maxNumUniqueValues != UNSET_INT,
          "maxNumUniqueValues(int) must be specified");
      return new NaiveMultisetSemaphore<>(maxNumUniqueValues);
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

    @Override
    public int estimateCurrentNumUniqueValues() {
      // We can't give a good estimate since we don't track values at all.
      return 0;
    }
  }

  private static class NaiveMultisetSemaphore<T> extends MultisetSemaphore<T> {
    private final int maxNumUniqueValues;
    private final Semaphore semaphore;
    private final Object lock = new Object();
    // Protected by 'lock'.
    private final HashMultiset<T> actualValues = HashMultiset.create();

    private NaiveMultisetSemaphore(int maxNumUniqueValues) {
      this.maxNumUniqueValues = maxNumUniqueValues;
      this.semaphore = new Semaphore(maxNumUniqueValues);
    }

    @Override
    public void acquireAll(Set<T> valuesToAcquire) throws InterruptedException {
      int oldNumNeededPermits;
      synchronized (lock) {
        oldNumNeededPermits = computeNumNeededPermitsLocked(valuesToAcquire);
      }
      while (true) {
        semaphore.acquire(oldNumNeededPermits);
        synchronized (lock) {
          int newNumNeededPermits = computeNumNeededPermitsLocked(valuesToAcquire);
          if (newNumNeededPermits != oldNumNeededPermits) {
            // While we were doing 'acquire' above, another thread won the race to acquire the first
            // usage of one of the values in 'valuesToAcquire' or release the last usage of one of
            // the values. This means we either acquired too many or too few permits, respectively,
            // above. Release the permits we did acquire, in order to restore the accuracy of the
            // semaphore's current count, and then try again.
            semaphore.release(oldNumNeededPermits);
            oldNumNeededPermits = newNumNeededPermits;
            continue;
          } else {
            // Our modification to the semaphore was correct, so it's sound to update the multiset.
            valuesToAcquire.forEach(actualValues::add);
            return;
          }
        }
      }
    }

    private int computeNumNeededPermitsLocked(Set<T> valuesToAcquire) {
      // We need a permit for each value that is not already in the multiset.
      return (int) valuesToAcquire.stream()
          .filter(v -> actualValues.count(v) == 0)
          .count();
    }

    @Override
    public void releaseAll(Set<T> valuesToRelease) {
      synchronized (lock) {
        // We need to release a permit for each value that currently has multiplicity 1.
        int numPermitsToRelease =
            valuesToRelease
                .stream()
                .mapToInt(v -> actualValues.remove(v, 1) == 1 ? 1 : 0)
                .sum();
        semaphore.release(numPermitsToRelease);
      }
    }

    @Override
    public int estimateCurrentNumUniqueValues() {
      return maxNumUniqueValues - semaphore.availablePermits();
    }
  }
}
