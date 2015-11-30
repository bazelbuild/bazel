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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Tests for MoreFutures
 */
@RunWith(JUnit4.class)
public class MoreFuturesTest {

  private ExecutorService executorService;

  @Before
  public final void createExecutor() throws Exception  {
    executorService = Executors.newFixedThreadPool(5);
  }

  @After
  public final void shutdownExecutor() throws Exception  {
    MoreExecutors.shutdownAndAwaitTermination(executorService, TestUtils.WAIT_TIMEOUT_SECONDS,
        TimeUnit.SECONDS);

  }

  /** Test the normal path where everything is successful. */
  @Test
  public void allAsListOrCancelAllHappy() throws ExecutionException, InterruptedException {
    final List<DelayedFuture> futureList = new ArrayList<>();
    for (int i = 0; i < 5; i++) {
      DelayedFuture future = new DelayedFuture(i);
      executorService.execute(future);
      futureList.add(future);
    }
    ListenableFuture<List<Object>> list = MoreFutures.allAsListOrCancelAll(futureList);
    List<Object> result = list.get();
    assertThat(result).hasSize(futureList.size());
    for (DelayedFuture delayedFuture : futureList) {
      assertFalse(delayedFuture.wasCanceled);
      assertFalse(delayedFuture.wasInterrupted);
      assertNotNull(delayedFuture.get());
      assertThat(result).contains(delayedFuture.get());
    }
  }

  /** Test that if any of the futures in the list fails, we cancel all the futures immediately. */
  @Test
  public void allAsListOrCancelAllCancellation() throws InterruptedException {
    final List<DelayedFuture> futureList = new ArrayList<>();
    for (int i = 1; i < 6; i++) {
      DelayedFuture future = new DelayedFuture(i * 1000);
      executorService.execute(future);
      futureList.add(future);
    }
    DelayedFuture toFail = new DelayedFuture(1000);
    futureList.add(toFail);
    toFail.makeItFail();
    ListenableFuture<List<Object>> list = MoreFutures.allAsListOrCancelAll(futureList);

    try {
      list.get();
      fail("This should fail");
    } catch (InterruptedException | ExecutionException ignored) {
    }
    Thread.sleep(100);
    for (DelayedFuture delayedFuture : futureList) {
      assertTrue(delayedFuture.wasCanceled || delayedFuture == toFail);
      assertFalse(delayedFuture.wasInterrupted);
    }
  }

  /**
   * A future that (if added to an executor) waits {@code delay} milliseconds before setting a
   * response.
   */
  private static class DelayedFuture extends AbstractFuture<Object> implements Runnable {

    private final int delay;
    private final CountDownLatch latch = new CountDownLatch(1);
    private boolean wasCanceled;
    private boolean wasInterrupted;

    public DelayedFuture(int delay) {
      this.delay = delay;
    }

    @Override
    public void run() {
      try {
        wasCanceled = latch.await(delay, TimeUnit.MILLISECONDS);
        // Not canceled and not done (makeItFail sets the value, so in that case is done).
        if (!wasCanceled && !isDone()) {
          set(new Object());
        }
      } catch (InterruptedException e) {
        wasInterrupted = true;
      }
    }

    public void makeItFail() {
      setException(new RuntimeException("I like to fail!!"));
      latch.countDown();
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      return super.cancel(mayInterruptIfRunning);
    }

    @Override
    protected void interruptTask() {
      latch.countDown();
    }
  }

}
