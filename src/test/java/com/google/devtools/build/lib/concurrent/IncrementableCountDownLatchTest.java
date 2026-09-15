// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.util.concurrent.MoreExecutors.shutdownAndAwaitTermination;
import static java.util.concurrent.Executors.newSingleThreadExecutor;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test cases for {@link IncrementableCountDownLatch}
 *
 * @author Shay Raz
 * @author Martin Buchholz
 */
@RunWith(JUnit4.class)
public class IncrementableCountDownLatchTest {
  private ExecutorService executor;

  @Before
  public void setUp() {
    executor = newSingleThreadExecutor();
  }

  @After
  public void tearDown() {
    assertTrue(shutdownAndAwaitTermination(executor, 10, SECONDS));
  }

  @Test
  public void testIncrementableCountDownLatch() throws Exception {
    CountingIncrementableCountDownLatch icdl = new CountingIncrementableCountDownLatch(2, 2);

    Future<?> result = executor.submit(new WaitSuccessfully(icdl));
    icdl.countDown();
    icdl.countDown();

    assertTrue(icdl.match());
    result.get();

    // increment by one
    icdl = new CountingIncrementableCountDownLatch(2, 3);

    result = executor.submit(new WaitSuccessfully(icdl));
    icdl.countDown();
    icdl.increment(1);
    icdl.countDown();
    icdl.countDown();

    assertTrue(icdl.match());
    result.get();
  }

  @Test
  public void testIncrementableCountDownLatchTooLate() throws Exception {
    CountingIncrementableCountDownLatch icdl = new CountingIncrementableCountDownLatch(2, 2);

    Future<?> result = executor.submit(new WaitSuccessfully(icdl));
    icdl.countDown();
    icdl.countDown();
    assertThrows(IllegalStateException.class, () -> icdl.increment(1));
    assertTrue(icdl.match());
    result.get();
  }

  @Test
  public void testIncrementableCountDownLatchWithTimeout() throws Exception {
    CountingIncrementableCountDownLatch icdl = new CountingIncrementableCountDownLatch(2, 2);

    Future<?> result = executor.submit(new WaitSuccessfullyWithTimeout(icdl));
    icdl.countDown();
    icdl.countDown();
    assertTrue(icdl.match());
    result.get();

    // increment by one
    icdl = new CountingIncrementableCountDownLatch(2, 3);

    result = executor.submit(new WaitSuccessfullyWithTimeout(icdl));
    icdl.countDown();
    icdl.increment(1);
    icdl.countDown();
    icdl.countDown();

    assertTrue(icdl.match());
    result.get();
  }

  @Test
  public void testIncrementableCountDownLatchWithTimeoutTimedOut() throws Exception {
    CountingIncrementableCountDownLatch icdl = new CountingIncrementableCountDownLatch(2, 1);

    Future<?> result = executor.submit(new WaitUnsuccessfullyWithTimeout(icdl));
    icdl.countDown();
    assertTrue(icdl.match());
    result.get();

    // increment by one
    icdl = new CountingIncrementableCountDownLatch(2, 2);

    result = executor.submit(new WaitUnsuccessfullyWithTimeout(icdl));
    icdl.countDown();
    icdl.increment(1);
    icdl.countDown();

    assertTrue(icdl.match());
    result.get();
  }

  /** increment() is equivalent to increment(1) */
  @Test
  public void testNullaryIncrement() throws InterruptedException {
    IncrementableCountDownLatch icdl = new IncrementableCountDownLatch(1);
    assertEquals(1, icdl.getCount());
    icdl.increment();
    assertEquals(2, icdl.getCount());
    icdl.increment();
    assertEquals(3, icdl.getCount());
    icdl.countDown();
    icdl.countDown();
    icdl.countDown();
    assertEquals(0, icdl.getCount());
    icdl.await();
  }

  /**
   * Incrementing past Integer.MAX_VALUE throws IllegalStateException, and leaves count unchanged.
   */
  @Test
  public void testCountOverflow() {
    IncrementableCountDownLatch icdl = new IncrementableCountDownLatch(1);
    assertThrows(IllegalArgumentException.class, () -> icdl.increment(Integer.MAX_VALUE));
    assertEquals(1, icdl.getCount());
  }

  /** Incrementing the count to Integer.MAX_VALUE succeeds. */
  @Test
  public void testIncrementCountToMaxValue() {
    IncrementableCountDownLatch icdl = new IncrementableCountDownLatch(42);

    icdl.increment(Integer.MAX_VALUE - 42);
    assertEquals(Integer.MAX_VALUE, icdl.getCount());
  }

  private static class WaitSuccessfully implements Callable<Void> {
    final CountingIncrementableCountDownLatch icdl;

    WaitSuccessfully(CountingIncrementableCountDownLatch icdl) {
      this.icdl = icdl;
    }

    @Override
    public Void call() throws Exception {
      icdl.await();
      return null;
    }
  }

  private static class WaitSuccessfullyWithTimeout implements Callable<Void> {
    final CountingIncrementableCountDownLatch icdl;

    WaitSuccessfullyWithTimeout(CountingIncrementableCountDownLatch icdl) {
      this.icdl = icdl;
    }

    @Override
    public Void call() throws Exception {
      assertTrue(icdl.await(10, SECONDS));
      return null;
    }
  }

  private static class WaitUnsuccessfullyWithTimeout implements Callable<Void> {
    final CountingIncrementableCountDownLatch icdl;

    WaitUnsuccessfullyWithTimeout(CountingIncrementableCountDownLatch icdl) {
      this.icdl = icdl;
    }

    @Override
    public Void call() throws Exception {
      assertFalse(icdl.await(12, MILLISECONDS));
      return null;
    }
  }

  private static class CountingIncrementableCountDownLatch {
    final int expected;
    int actual = 0;
    final IncrementableCountDownLatch latch;

    CountingIncrementableCountDownLatch(int count, int expected) {
      latch = new IncrementableCountDownLatch(count);
      this.expected = expected;
    }

    boolean await(int timeout, TimeUnit unit) throws InterruptedException {
      return latch.await(timeout, unit);
    }

    void await() throws InterruptedException {
      latch.await();
    }

    void increment(int i) {
      latch.increment(i);
    }

    void countDown() {
      actual++;
      latch.countDown();
    }

    boolean match() {
      return expected == actual;
    }
  }
}
