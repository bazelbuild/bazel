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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.ErrorClassifier.ErrorClassification;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tests for AbstractQueueVisitor.
 */
@RunWith(JUnit4.class)
public class AbstractQueueVisitorTest {

  private static final RuntimeException THROWABLE = new RuntimeException();

  @Test
  public void simpleCounter() throws Exception {
    CountingQueueVisitor counter = new CountingQueueVisitor();
    counter.enqueue();
    counter.awaitQuiescence(/*interruptWorkers=*/ false);
    assertSame(10, counter.getCount());
    assertSame(0, counter.activeParallelTasks());
  }

  @Test
  public void callerOwnedPool() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(5, 5, 0, TimeUnit.SECONDS,
                                                         new LinkedBlockingQueue<Runnable>());
    assertSame(0, executor.getActiveCount());

    CountingQueueVisitor counter = new CountingQueueVisitor(executor);
    counter.enqueue();
    counter.awaitQuiescence(/*interruptWorkers=*/ false);
    assertSame(10, counter.getCount());

    executor.shutdown();
    assertTrue(executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
  }

  @Test
  public void doubleCounter() throws Exception {
    CountingQueueVisitor counter = new CountingQueueVisitor();
    counter.enqueue();
    counter.enqueue();
    counter.awaitQuiescence(/*interruptWorkers=*/ false);
    assertSame(10, counter.getCount());
  }

  @Test
  public void exceptionFromWorkerThread() {
    final RuntimeException myException = new IllegalStateException();
    ConcreteQueueVisitor visitor = new ConcreteQueueVisitor();
    visitor.execute(
        new Runnable() {
          @Override
          public void run() {
            throw myException;
          }
        });

    try {
      // The exception from the worker thread should be
      // re-thrown from the main thread.
      visitor.awaitQuiescence(/*interruptWorkers=*/ false);
      fail();
    } catch (Exception e) {
      assertSame(myException, e);
    }
  }

  // Regression test for "AbstractQueueVisitor loses track of jobs if thread allocation fails".
  @Test
  public void threadPoolThrowsSometimes() throws Exception {
    // In certain cases (for example, if the address space is almost entirely consumed by a huge
    // JVM heap), thread allocation can fail with an OutOfMemoryError. If the queue visitor
    // does not handle this gracefully, we lose track of tasks and hang the visitor indefinitely.

    ThreadPoolExecutor executor = new ThreadPoolExecutor(3, 3, 0, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>()) {
      private final AtomicLong count = new AtomicLong();

      @Override
      public void execute(Runnable command) {
        long count = this.count.incrementAndGet();
        if (count == 6) {
          throw new Error("Could not create thread (fakeout)");
        }
        super.execute(command);
      }
    };

    CountingQueueVisitor counter = new CountingQueueVisitor(executor);
    counter.enqueue();
    try {
      counter.awaitQuiescence(/*interruptWorkers=*/ false);
      fail();
    } catch (Error expected) {
      assertThat(expected).hasMessage("Could not create thread (fakeout)");
    }
    assertSame(5, counter.getCount());

    executor.shutdown();
    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
  }

  // Regression test to make sure that AbstractQueueVisitor doesn't swallow unchecked exceptions if
  // it is interrupted concurrently with the unchecked exception being thrown.
  @Test
  public void interruptAndThrownIsInterruptedAndThrown() throws Exception {
    final ConcreteQueueVisitor visitor = new ConcreteQueueVisitor();
    // Use a latch to make sure the thread gets a chance to start.
    final CountDownLatch threadStarted = new CountDownLatch(1);
    visitor.execute(
        new Runnable() {
          @Override
          public void run() {
            threadStarted.countDown();
            assertTrue(
                Uninterruptibles.awaitUninterruptibly(
                    visitor.getInterruptionLatchForTestingOnly(), 2, TimeUnit.SECONDS));
            throw THROWABLE;
          }
        });
    assertTrue(threadStarted.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
    // Interrupt will not be processed until work starts.
    Thread.currentThread().interrupt();
    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ true);
      fail();
    } catch (Exception e) {
      assertEquals(THROWABLE, e);
      assertTrue(Thread.interrupted());
    }
  }

  @Test
  public void interruptionWithoutInterruptingWorkers() throws Exception {
    final Thread mainThread = Thread.currentThread();
    final CountDownLatch latch1 = new CountDownLatch(1);
    final CountDownLatch latch2 = new CountDownLatch(1);
    final boolean[] workerThreadCompleted = { false };
    final ConcreteQueueVisitor visitor = new ConcreteQueueVisitor();

    visitor.execute(
        new Runnable() {
          @Override
          public void run() {
            try {
              latch1.countDown();
              latch2.await();
              workerThreadCompleted[0] = true;
            } catch (InterruptedException e) {
              // Do not set workerThreadCompleted to true
            }
          }
        });

    TestThread interrupterThread =
        new TestThread() {
          @Override
          public void runTest() throws Exception {
            latch1.await();
            mainThread.interrupt();
            assertTrue(
                visitor
                    .getInterruptionLatchForTestingOnly()
                    .await(TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS));
            latch2.countDown();
          }
        };

    interrupterThread.start();

    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ false);
      fail();
    } catch (InterruptedException e) {
      // Expected.
    }

    interrupterThread.joinAndAssertState(400);
    assertTrue(workerThreadCompleted[0]);
  }

  @Test
  public void interruptionWithInterruptingWorkers() throws Exception {
    assertInterruptWorkers(null);

    ThreadPoolExecutor executor = new ThreadPoolExecutor(3, 3, 0, TimeUnit.SECONDS,
                                                         new LinkedBlockingQueue<Runnable>());
    assertInterruptWorkers(executor);
    executor.shutdown();
    executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
  }

  private void assertInterruptWorkers(ThreadPoolExecutor executor) throws Exception {
    final CountDownLatch latch1 = new CountDownLatch(1);
    final CountDownLatch latch2 = new CountDownLatch(1);
    final boolean[] workerThreadInterrupted = { false };
    ConcreteQueueVisitor visitor = (executor == null)
        ? new ConcreteQueueVisitor()
        : new ConcreteQueueVisitor(executor, true);

    visitor.execute(
        new Runnable() {
          @Override
          public void run() {
            try {
              latch1.countDown();
              latch2.await();
            } catch (InterruptedException e) {
              workerThreadInterrupted[0] = true;
            }
          }
        });

    latch1.await();
    Thread.currentThread().interrupt();

    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ true);
      fail();
    } catch (InterruptedException e) {
      // Expected.
    }

    assertTrue(workerThreadInterrupted[0]);
  }

  @Test
  public void failFast() throws Exception {
    // In failFast mode, we only run actions queued before the exception.
    assertFailFast(null, true, false, false, "a", "b");

    // In !failFast mode, we complete all queued actions.
    assertFailFast(null, false, false, false, "a", "b", "1", "2");

    // Now check fail-fast on interrupt:
    assertFailFast(null, false, true, true, "a", "b");
    assertFailFast(null, false, false, true, "a", "b", "1", "2");
  }

  @Test
  public void failFastNoShutdown() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(5, 5, 0, TimeUnit.SECONDS,
                                                         new LinkedBlockingQueue<Runnable>());
    // In failFast mode, we only run actions queued before the exception.
    assertFailFast(executor, true, false, false, "a", "b");

    // In !failFast mode, we complete all queued actions.
    assertFailFast(executor, false, false, false, "a", "b", "1", "2");

    // Now check fail-fast on interrupt:
    assertFailFast(executor, false, true, true, "a", "b");
    assertFailFast(executor, false, false, true, "a", "b", "1", "2");

    executor.shutdown();
    assertTrue(executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
  }

  private void assertFailFast(ThreadPoolExecutor executor,
                              boolean failFastOnException, boolean failFastOnInterrupt,
                              boolean interrupt, String... expectedVisited) throws Exception {
    assertTrue(executor == null || !executor.isShutdown());
    AbstractQueueVisitor visitor = (executor == null)
        ? new ConcreteQueueVisitor(failFastOnException, failFastOnInterrupt)
        : new ConcreteQueueVisitor(executor, failFastOnException, failFastOnInterrupt);

    List<String> visitedList = Collections.synchronizedList(Lists.<String>newArrayList());

    // Runnable "ra" will await the uncaught exception from
    // "throwingRunnable", then add "a" to the list and
    // enqueue "r1". Runnable "r1" should be
    // executed iff !failFast.

    CountDownLatch latchA = new CountDownLatch(1);
    CountDownLatch latchB = new CountDownLatch(1);

    Runnable r1 = awaitAddAndEnqueueRunnable(interrupt, visitor, null, visitedList, "1", null);
    Runnable r2 = awaitAddAndEnqueueRunnable(interrupt, visitor, null, visitedList, "2", null);
    Runnable ra = awaitAddAndEnqueueRunnable(interrupt, visitor, latchA, visitedList, "a", r1);
    Runnable rb = awaitAddAndEnqueueRunnable(interrupt, visitor, latchB, visitedList, "b", r2);

    visitor.execute(ra);
    visitor.execute(rb);
    latchA.await();
    latchB.await();
    visitor.execute(interrupt ? interruptingRunnable(Thread.currentThread()) : throwingRunnable());

    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ false);
      fail();
    } catch (Exception e) {
      if (interrupt) {
        assertThat(e).isInstanceOf(InterruptedException.class);
      } else {
        assertSame(THROWABLE, e);
      }
    }
    assertEquals("got: " + visitedList + "\nwant: " + Arrays.toString(expectedVisited),
        Sets.newHashSet(expectedVisited), Sets.newHashSet(visitedList));

    if (executor != null) {
      assertFalse(executor.isShutdown());
      assertEquals(0, visitor.getTaskCount());
    }
  }

  @Test
  public void jobIsInterruptedWhenOtherFails() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(3, 3, 0, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>());

    final AbstractQueueVisitor visitor =
        createQueueVisitorWithErrorClassification(executor, ErrorClassification.CRITICAL);
    final CountDownLatch latch1 = new CountDownLatch(1);
    final AtomicBoolean wasInterrupted = new AtomicBoolean(false);

    Runnable r1 = new Runnable() {

      @Override
      public void run() {
        latch1.countDown();
        try {
          // Interruption is expected during a sleep. There is no sense in fail or assert call
          // because exception is going to be swallowed inside AbstractQueueVisitior.
          // We are using wasInterrupted flag to assert in the end of test.
          Thread.sleep(1000);
        } catch (InterruptedException e) {
          wasInterrupted.set(true);
        }
      }
    };

    visitor.execute(r1);
    latch1.await();
    visitor.execute(throwingRunnable());

    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ true);
      fail();
    } catch (Exception e) {
      assertSame(THROWABLE, e);
    }

    assertTrue(wasInterrupted.get());
    assertTrue(executor.isShutdown());
  }

  @Test
  public void javaErrorConsideredCriticalNoMatterWhat() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(2, 2, 0, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>());
    AbstractQueueVisitor visitor =
        createQueueVisitorWithErrorClassification(executor, ErrorClassification.NOT_CRITICAL);
    final CountDownLatch latch = new CountDownLatch(1);
    final AtomicBoolean sleepFinished = new AtomicBoolean(false);
    final AtomicBoolean sleepInterrupted = new AtomicBoolean(false);
    final Error error = new Error("bad!");
    Runnable errorRunnable = new Runnable() {
      @Override
      public void run() {
        try {
          latch.await(TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS);
        } catch (InterruptedException expected) {
          // Should only happen if the test itself is interrupted.
        }
        throw error;
      }
    };
    Runnable sleepRunnable = new Runnable() {
      @Override
      public void run() {
        latch.countDown();
        try {
          Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
          sleepFinished.set(true);
        } catch (InterruptedException unexpected) {
          sleepInterrupted.set(true);
        }
      }
    };
    visitor.execute(errorRunnable);
    visitor.execute(sleepRunnable);
    Error thrownError = null;
    // Interrupt workers on a critical error. That way we can test that visitor.work doesn't wait
    // for all workers to finish if one of them already had a critical error.
    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ true);
    } catch (Error e) {
      thrownError = e;
    }
    assertTrue(sleepInterrupted.get());
    assertFalse(sleepFinished.get());
    assertEquals(error, thrownError);
  }

  private Runnable throwingRunnable() {
    return new Runnable() {
      @Override
      public void run() {
        throw THROWABLE;
      }
    };
  }

  private Runnable interruptingRunnable(final Thread thread) {
    return new Runnable() {
      @Override
      public void run() {
        thread.interrupt();
      }
    };
  }

  private static Runnable awaitAddAndEnqueueRunnable(final boolean interrupt,
                                                     final AbstractQueueVisitor visitor,
                                                     final CountDownLatch started,
                                                     final List<String> list,
                                                     final String toAdd,
                                                     final Runnable toEnqueue) {
    return new Runnable() {
      @Override
      public void run() {
        if (started != null) {
          started.countDown();
        }

        try {
          assertTrue(
              interrupt
                  ? visitor.getInterruptionLatchForTestingOnly().await(1, TimeUnit.MINUTES)
                  : visitor.getExceptionLatchForTestingOnly().await(1, TimeUnit.MINUTES));
        } catch (InterruptedException e) {
          // Unexpected.
          throw new RuntimeException(e);
        }
        list.add(toAdd);
        if (toEnqueue != null) {
          visitor.execute(toEnqueue);
        }
      }
    };
  }

  private static class CountingQueueVisitor extends AbstractQueueVisitor {

    private final static String THREAD_NAME = "BlazeTest CountingQueueVisitor";

    private int theInt = 0;
    private final Object lock = new Object();

    public CountingQueueVisitor() {
      super(/*parallelism=*/ 5, /*keepAlive=*/ 3L, TimeUnit.SECONDS, THREAD_NAME);
    }

    public CountingQueueVisitor(ThreadPoolExecutor executor) {
      super(executor, false, true, true);
    }

    public void enqueue() {
      super.execute(
          new Runnable() {
            @Override
            public void run() {
              synchronized (lock) {
                if (theInt < 10) {
                  theInt++;
                  enqueue();
                }
              }
            }
          });
    }

    public int getCount() {
      return theInt;
    }
  }

  private static class ConcreteQueueVisitor extends AbstractQueueVisitor {

    private final static String THREAD_NAME = "BlazeTest ConcreteQueueVisitor";

    public ConcreteQueueVisitor() {
      super(5, 3L, TimeUnit.SECONDS, THREAD_NAME);
    }

    public ConcreteQueueVisitor(boolean failFast, boolean failFastOnInterrupt) {
      super(true, 5, 3L, TimeUnit.SECONDS, failFast, failFastOnInterrupt, THREAD_NAME);
    }

    public ConcreteQueueVisitor(ThreadPoolExecutor executor, boolean failFast,
        boolean failFastOnInterrupt) {
      super(executor, /*shutdownOnCompletion=*/false, failFast, failFastOnInterrupt);
    }

    public ConcreteQueueVisitor(ThreadPoolExecutor executor, boolean failFast) {
      super(executor, /*shutdownOnCompletion=*/false, failFast, true);
    }
  }

  private static AbstractQueueVisitor createQueueVisitorWithErrorClassification(
      ThreadPoolExecutor executor, final ErrorClassification classification) {
    return new AbstractQueueVisitor(
        /*concurrent=*/ true,
        executor,
        /*shutdownOnCompletion=*/ true,
        /*failFastOnException=*/ false,
        /*failFastOnInterrupt=*/ true,
        new ErrorClassifier() {
          @Override
          protected ErrorClassification classifyException(Exception e) {
            return classification;
          }
        });
  }
}
