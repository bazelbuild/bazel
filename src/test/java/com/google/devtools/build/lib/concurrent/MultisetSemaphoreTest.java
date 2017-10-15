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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MultisetSemaphore}. */
@RunWith(JUnit4.class)
public class MultisetSemaphoreTest {

  @Test
  public void testSimple_Serial() throws Exception {
    // When we have a MultisetSemaphore
    MultisetSemaphore<String> multisetSemaphore = MultisetSemaphore.newBuilder()
        // with 3 max num unique values,
        .maxNumUniqueValues(3)
        .build();

    // And we serially acquire permits for 3 unique values
    multisetSemaphore.acquireAll(ImmutableSet.of("a", "b", "c"));
    // And then attempt to acquire permits for 2 of those same unique values,
    // Then we don't deadlock.
    multisetSemaphore.acquireAll(ImmutableSet.of("b", "c"));
    // And then we release one of the permit for one of those unique values,
    multisetSemaphore.releaseAll(ImmutableSet.of("c"));
    // And then we release the other permit,
    multisetSemaphore.releaseAll(ImmutableSet.of("c"));
    // We are able to acquire a permit for a 4th unique value.
    multisetSemaphore.acquireAll(ImmutableSet.of("d"));
  }

  @Test
  public void testSimple_Concurrent() throws Exception {
    // When we have N and M, with M > N and M|N.
    final int n = 10;
    int m = n * 2;
    Preconditions.checkState(m > n && m % n == 0, "M=%d N=%d", m, n);
    // When we have a MultisetSemaphore
    final MultisetSemaphore<String> multisetSemaphore = MultisetSemaphore.newBuilder()
        // with N max num unique values,
        .maxNumUniqueValues(n)
        .build();

    // And a ExecutorService with M threads,
    ExecutorService executorService = Executors.newFixedThreadPool(m);
    // And a recorder for thrown exceptions,
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("testSimple_Concurrent");
    final AtomicInteger numThreadsJustAfterAcquireInFirstRound = new AtomicInteger(0);
    final AtomicInteger numThreadsJustAfterAcquireInSecondRound = new AtomicInteger(0);
    final AtomicInteger secondRoundCompleted = new AtomicInteger(0);
    final int napTimeMs = 42;
    for (int i = 0; i < m; i++) {
      final String val = "val" + i;
      // And we submit M Runnables, each of which
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executorService.submit(
              wrapper.wrap(
                  new Runnable() {
                    @Override
                    public void run() {
                      try {
                        // Has two rounds

                        // Wherein the first round
                        //   The Runnable acquire a permit for a unique value (among M values),
                        ImmutableSet<String> valSet = ImmutableSet.of(val);
                        multisetSemaphore.acquireAll(valSet);
                        assertThat(numThreadsJustAfterAcquireInFirstRound.getAndIncrement())
                            .isLessThan(n);
                        //   And then sleeps,
                        Thread.sleep(napTimeMs);
                        numThreadsJustAfterAcquireInFirstRound.decrementAndGet();
                        multisetSemaphore.releaseAll(valSet);

                        // And wherein the second round
                        //   The Runnable again acquires a permit for its unique value,
                        multisetSemaphore.acquireAll(valSet);
                        assertThat(numThreadsJustAfterAcquireInSecondRound.getAndIncrement())
                            .isLessThan(n);
                        //   And then sleeps,
                        Thread.sleep(napTimeMs);
                        numThreadsJustAfterAcquireInSecondRound.decrementAndGet();
                        //   And notes that it has completed the second round,
                        secondRoundCompleted.incrementAndGet();
                        multisetSemaphore.releaseAll(valSet);
                      } catch (InterruptedException e) {
                        throw new IllegalStateException(e);
                      }
                    }
                  }));
    }
    // And we wait for all M Runnables to complete (that is, none of them were deadlocked),
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    // Then none of our Runnables threw any Exceptions.
    assertThat(wrapper.getFirstThrownError()).isNull();
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    // And the counters we used for sanity checks were correctly reset to 0.
    assertThat(numThreadsJustAfterAcquireInFirstRound.get()).isEqualTo(0);
    assertThat(numThreadsJustAfterAcquireInSecondRound.get()).isEqualTo(0);
    // And all M Runnables completed the second round.
    assertThat(secondRoundCompleted.get()).isEqualTo(m);
    Set<String> newVals = new HashSet<>();
    for (int i = 0; i < n; i++) {
      newVals.add("newval" + i);
    }
    // And the main test thread is able to acquire permits for N new unique values (indirectly
    // confirming that the MultisetSemaphore previously had no outstanding permits).
    multisetSemaphore.acquireAll(newVals);
  }

  @Test
  public void testConcurrentAtomicity() throws Exception {
    int n = 100;
    // When we have a MultisetSemaphore
    final MultisetSemaphore<String> multisetSemaphore = MultisetSemaphore.newBuilder()
        // with 2 max num unique values,
        .maxNumUniqueValues(2)
        .build();
    // And a ExecutorService with N threads,
    ExecutorService executorService = Executors.newFixedThreadPool(n);
    // And a recorder for thrown exceptions,
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("testConcurrentAtomicity");
    final int napTimeMs = 42;
    // And a done latch with initial count N,
    final CountDownLatch allDoneLatch = new CountDownLatch(n);
    final String sameVal = "same-val";
    for (int i = 0; i < n; i++) {
      final String differentVal = "different-val" + i;
      // And we submit N Runnables, each of which
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executorService.submit(
              wrapper.wrap(
                  new Runnable() {
                    @Override
                    public void run() {
                      try {
                        Set<String> vals = ImmutableSet.of(sameVal, differentVal);
                        // Tries to acquire a permit for a set of two values, one of which is the same for all
                        // the N Runnables and one of which is unique across all N Runnables.
                        multisetSemaphore.acquireAll(vals);
                        // And then sleeps
                        Thread.sleep(napTimeMs);
                        // And then releases its permits
                        multisetSemaphore.releaseAll(vals);
                        // And then counts down the done latch,
                        allDoneLatch.countDown();
                      } catch (InterruptedException e) {
                        throw new IllegalStateException(e);
                      }
                    }
                  }));
    }
    // Then all of our Runnables completed (without deadlock!), as expected,
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    // And thus were able to count down the done latch,
    allDoneLatch.await();
    // And also none of them threw any Exceptions.
    assertThat(wrapper.getFirstThrownError()).isNull();
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
  }

  @Test
  public void testConcurrentRace() throws Exception {
    // When we have N values
    int n = 6;
    ArrayList<String> vals = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      vals.add("val-" + i);
    }
    // And we have all permutations of these N values
    Collection<List<String>> permutations = Collections2.orderedPermutations(vals);
    int numPermutations = permutations.size();
    // And we have a MultisetSemaphore
    final MultisetSemaphore<String> multisetSemaphore = MultisetSemaphore.newBuilder()
        // with N max num unique values,
        .maxNumUniqueValues(n)
        .build();
    // And a ExecutorService with N! threads,
    ExecutorService executorService = Executors.newFixedThreadPool(numPermutations);
    // And a recorder for thrown exceptions,
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("testConcurrentRace");
    for (List<String> orderedVals : permutations) {
      final Set<String> orderedSet = new LinkedHashSet<>(orderedVals);
      // And we submit N! Runnables, each of which
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError =
          executorService.submit(
              wrapper.wrap(
                  new Runnable() {
                    @Override
                    public void run() {
                      try {
                        // Tries to acquire a permit for the set of N values, with a unique iteration order
                        // (across all the N! different permutations)
                        multisetSemaphore.acquireAll(orderedSet);
                        // And then immediately releases the permit.
                        multisetSemaphore.releaseAll(orderedSet);
                      } catch (InterruptedException e) {
                        throw new IllegalStateException(e);
                      }
                    }
                  }));
    }
    // Then all of our Runnables completed (without deadlock!), as expected,
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    // And also none of them threw any Exceptions.
    assertThat(wrapper.getFirstThrownError()).isNull();
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
  }
}

