// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;

import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Safely await {@link CountDownLatch}es in tests, storing any exceptions that happen. Callers
 * should call {@link #assertNoErrors} at the end of each test method, either manually or using an
 * {@code @After} hook.
 */
public class TrackingAwaiter {
  public static final TrackingAwaiter INSTANCE = new TrackingAwaiter();

  private TrackingAwaiter() {}

  private final ConcurrentLinkedQueue<Pair<String, Throwable>> exceptionsThrown =
      new ConcurrentLinkedQueue<>();

  /**
   * This method fixes a race condition with simply calling {@link CountDownLatch#await}. If this
   * thread is interrupted before {@code latch.await} is called, then {@code latch.await} will throw
   * an {@link InterruptedException} without checking the value of the latch at all. This leads to a
   * race condition in which this thread will throw an InterruptedException if it is slow calling
   * {@code latch.await}, but it will succeed normally otherwise.
   *
   * <p>To avoid this, we wait for the latch uninterruptibly. In the end, if the latch has in fact
   * been released, we do nothing, although the interrupted bit is set, so that the caller can
   * decide to throw an InterruptedException if it wants to. If the latch was not released, then
   * this was not a race condition, but an honest-to-goodness interrupt, and we propagate the
   * exception onward.
   */
  private static void waitAndMaybeThrowInterrupt(CountDownLatch latch, String errorMessage)
      throws InterruptedException {
    if (Uninterruptibles.awaitUninterruptibly(latch, TestUtils.WAIT_TIMEOUT_SECONDS,
        TimeUnit.SECONDS)) {
      // Latch was released. We can ignore the interrupt state.
      return;
    }
    if (!Thread.currentThread().isInterrupted()) {
      // Nobody interrupted us, but latch wasn't released. Failure.
      throw new AssertionError(errorMessage);
    } else {
      // We were interrupted before the latch was released. Propagate this interruption.
      throw new InterruptedException();
    }
  }

  /** Threadpools can swallow exceptions. Make sure they don't get lost. */
  public void awaitLatchAndTrackExceptions(CountDownLatch latch, String errorMessage) {
    try {
      waitAndMaybeThrowInterrupt(latch, errorMessage);
    } catch (Throwable e) {
      // We would expect e to be InterruptedException or AssertionError, but we leave it open so
      // that any throwable gets recorded.
      exceptionsThrown.add(Pair.of(errorMessage, e));
      // Caller will assert exceptionsThrown is empty at end of test and fail, even if this is
      // swallowed.
      Throwables.propagate(e);
    }
  }

  public void assertNoErrors() {
    List<Pair<String, Throwable>> thisEvalExceptionsThrown = ImmutableList.copyOf(exceptionsThrown);
    exceptionsThrown.clear();
    assertThat(thisEvalExceptionsThrown).isEmpty();
  }
}
