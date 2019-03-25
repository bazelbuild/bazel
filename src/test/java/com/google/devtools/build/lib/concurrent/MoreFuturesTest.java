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
import static org.junit.Assert.fail;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
      assertThat(delayedFuture.wasCanceled).isFalse();
      assertThat(delayedFuture.wasInterrupted).isFalse();
      assertThat(delayedFuture.get()).isNotNull();
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
      assertThat(delayedFuture.wasCanceled || delayedFuture == toFail).isTrue();
      assertThat(delayedFuture.wasInterrupted).isFalse();
    }
  }

  @Test
  public void waitForAllInterruptiblyFailFast_AllSuccessful() throws Exception {
    List<DelayedFuture> futureList = new ArrayList<>();
    for (int i = 1; i < 6; i++) {
      DelayedFuture future = new DelayedFuture(i * 1000);
      executorService.execute(future);
      futureList.add(future);
    }
    MoreFutures.waitForAllInterruptiblyFailFast(futureList);
    for (DelayedFuture delayedFuture : futureList) {
      assertThat(delayedFuture.wasCanceled).isFalse();
      assertThat(delayedFuture.wasInterrupted).isFalse();
      assertThat(delayedFuture.get()).isNotNull();
    }
  }

  @Test
  public void waitForAllInterruptiblyFailFast_Interrupt() throws Exception {
    final List<DelayedFuture> futureList = new ArrayList<>();
    for (int i = 1; i < 6; i++) {
      // When we have a bunch of futures that never complete.
      DelayedFuture future = new DelayedFuture(Integer.MAX_VALUE);
      // And submit them to an Executor.
      executorService.execute(future);
      futureList.add(future);
    }
    final Thread testThread = Thread.currentThread();
    // And have a thread that interrupts the current thread (the one running the test) once all the
    // futures were polled at least once via Future#get(long, TimeUnit).
    Thread interruptThread = new Thread() {
      @Override
      public void run() {
        for (DelayedFuture delayedFuture : futureList) {
          try {
            delayedFuture.getLatch.await(
                TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS);
          } catch (InterruptedException ie) {
            throw new IllegalStateException(ie);
          }
        }
        testThread.interrupt();
      }
    };
    // And run this thread in the background.
    interruptThread.start();
    try {
      try {
        // And then wait for all the futures to complete, interruptibly.
        MoreFutures.waitForAllInterruptiblyFailFast(futureList);
        fail();
      } catch (InterruptedException expected) {
        // Then, as expected, waitForAllInterruptiblyFailFast propagates the interrupt sent to the
        // main test thread by our background thread.
      }
    } finally {
      // The @After-annotated shutdownExecutor method blocks on completion of all tasks. Since we
      // submitted a bunch of tasks that never complete, we need to explicitly cancel them.
      for (DelayedFuture delayedFuture : futureList) {
        delayedFuture.cancel(/*mayInterruptIfRunning=*/ true);
      }
      // If we're here and the test were to pass, then the background thread must have already
      // completed. Interrupt it unconditionally - if the test were to pass, this is benign. But if
      // the test were to fail then we'd have a rogue thread in the background which can be very
      // evil (e.g. can interfere with the execution of other test cases).
      interruptThread.interrupt();
    }
  }

  @Test
  public void waitForAllInterruptiblyFailFast_Failure() throws Exception {
    List<DelayedFuture> futureList = new ArrayList<>();
    for (int i = 1; i < 6; i++) {
      DelayedFuture future = new DelayedFuture(i * 1000);
      executorService.execute(future);
      futureList.add(future);
    }
    DelayedFuture toFail = new DelayedFuture(1000);
    futureList.add(toFail);
    toFail.makeItFail();
    try {
      MoreFutures.waitForAllInterruptiblyFailFast(futureList);
      fail();
    } catch (ExecutionException ee) {
      assertThat(ee).hasCauseThat().hasMessageThat().isEqualTo("I like to fail!!");
    }
  }

  /**
   * A future that (if added to an executor) waits {@code delay} milliseconds before setting a
   * response.
   */
  private static class DelayedFuture extends AbstractFuture<Object> implements Runnable {

    private final int delay;
    private final CountDownLatch failOrInterruptLatch = new CountDownLatch(1);
    private final CountDownLatch getLatch = new CountDownLatch(1);
    private boolean wasCanceled;
    private boolean wasInterrupted;

    public DelayedFuture(int delay) {
      this.delay = delay;
    }

    @Override
    public void run() {
      try {
        wasCanceled = failOrInterruptLatch.await(delay, TimeUnit.MILLISECONDS);
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
      failOrInterruptLatch.countDown();
    }

    @Override
    public boolean cancel(boolean mayInterruptIfRunning) {
      return super.cancel(mayInterruptIfRunning);
    }

    @Override
    protected void interruptTask() {
      failOrInterruptLatch.countDown();
    }

    @Override
    public Object get(long timeout, TimeUnit unit)
        throws InterruptedException, TimeoutException, ExecutionException {
      getLatch.countDown();
      return super.get(timeout, unit);
    }
  }

}
