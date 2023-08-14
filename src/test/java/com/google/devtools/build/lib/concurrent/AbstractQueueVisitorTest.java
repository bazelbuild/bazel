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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor.ExceptionHandlingMode;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor.ExecutorOwnership;
import com.google.devtools.build.lib.concurrent.ErrorClassifier.ErrorClassification;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
    assertThat(counter.getCount()).isSameInstanceAs(10);
  }

  @Test
  public void externalDep() throws Exception {
    SettableFuture<Object> future = SettableFuture.create();
    AbstractQueueVisitor counter =
        new AbstractQueueVisitor(
            /* parallelism= */ 2,
            /* keepAliveTime= */ 3L,
            TimeUnit.SECONDS,
            ExceptionHandlingMode.FAIL_FAST,
            "FOO-BAR",
            ErrorClassifier.DEFAULT);
    counter.dependOnFuture(future);
    new Thread(
            () -> {
              try {
                Thread.sleep(5);
                future.set(new Object());
              } catch (InterruptedException e) {
                throw new RuntimeException(e);
              }
            })
        .start();
    counter.awaitQuiescence(/*interruptWorkers=*/ false);
  }

  @Test
  public void externalDepWithInterrupt() throws Exception {
    SettableFuture<Object> future = SettableFuture.create();
    AbstractQueueVisitor counter =
        new AbstractQueueVisitor(
            /* parallelism= */ 2,
            /* keepAliveTime= */ 3L,
            TimeUnit.SECONDS,
            ExceptionHandlingMode.FAIL_FAST,
            "FOO-BAR",
            ErrorClassifier.DEFAULT);
    counter.dependOnFuture(future);
    Thread.currentThread().interrupt();
    assertThrows(
        InterruptedException.class, () -> counter.awaitQuiescence(/*interruptWorkers=*/ true));
    assertThat(future.isCancelled()).isTrue();
  }

  @Test
  public void callerOwnedPool() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(5, 5, 0, TimeUnit.SECONDS,
                                                         new LinkedBlockingQueue<Runnable>());
    assertThat(executor.getActiveCount()).isSameInstanceAs(0);

    CountingQueueVisitor counter = new CountingQueueVisitor(executor);
    counter.enqueue();
    counter.awaitQuiescence(/*interruptWorkers=*/ false);
    assertThat(counter.getCount()).isSameInstanceAs(10);

    executor.shutdown();
    assertThat(executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        .isTrue();
  }

  @Test
  public void doubleCounter() throws Exception {
    CountingQueueVisitor counter = new CountingQueueVisitor();
    counter.enqueue();
    counter.enqueue();
    counter.awaitQuiescence(/*interruptWorkers=*/ false);
    assertThat(counter.getCount()).isSameInstanceAs(10);
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

    // The exception from the worker thread should be re-thrown from the main thread.
    Exception e =
        assertThrows(Exception.class, () -> visitor.awaitQuiescence(/*interruptWorkers=*/ false));
    assertThat(e).isSameInstanceAs(myException);
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
    Error expected =
        assertThrows(Error.class, () -> counter.awaitQuiescence(/*interruptWorkers=*/ false));
    assertThat(expected).hasMessageThat().isEqualTo("Could not create thread (fakeout)");
    assertThat(counter.getCount()).isSameInstanceAs(5);

    executor.shutdown();
    assertThat(executor.awaitTermination(10, TimeUnit.SECONDS)).isTrue();
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
            assertThat(
                    Uninterruptibles.awaitUninterruptibly(
                        visitor.getInterruptionLatchForTestingOnly(), 2, TimeUnit.SECONDS))
                .isTrue();
            throw THROWABLE;
          }
        });
    assertThat(threadStarted.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
    // Interrupt will not be processed until work starts.
    Thread.currentThread().interrupt();
    Exception e =
        assertThrows(Exception.class, () -> visitor.awaitQuiescence(/*interruptWorkers=*/ true));
    assertThat(e).isEqualTo(THROWABLE);
    assertThat(Thread.interrupted()).isTrue();
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
        new TestThread(
            () -> {
              latch1.await();
              mainThread.interrupt();
              assertThat(
                      visitor
                          .getInterruptionLatchForTestingOnly()
                          .await(TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS))
                  .isTrue();
              latch2.countDown();
            });
    interrupterThread.start();

    assertThrows(
        InterruptedException.class, () -> visitor.awaitQuiescence(/*interruptWorkers=*/ false));

    interrupterThread.joinAndAssertState(400);
    assertThat(workerThreadCompleted[0]).isTrue();
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

  private static void assertInterruptWorkers(ThreadPoolExecutor executor) throws Exception {
    final CountDownLatch latch1 = new CountDownLatch(1);
    final CountDownLatch latch2 = new CountDownLatch(1);
    final boolean[] workerThreadInterrupted = { false };
    ConcreteQueueVisitor visitor =
        (executor == null)
            ? new ConcreteQueueVisitor()
            : new ConcreteQueueVisitor(executor, ExceptionHandlingMode.FAIL_FAST);

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

    assertThrows(
        InterruptedException.class, () -> visitor.awaitQuiescence(/*interruptWorkers=*/ true));

    assertThat(workerThreadInterrupted[0]).isTrue();
  }

  @Test
  public void failFast() throws Exception {
    // In failFast mode, we only run actions queued before the exception.
    assertFailFast(null, ExceptionHandlingMode.FAIL_FAST, false, "a", "b");

    // In !failFast mode, we complete all queued actions.
    assertFailFast(null, ExceptionHandlingMode.KEEP_GOING, false, "a", "b", "1", "2");

    // Now check fail-fast on interrupt:
    assertFailFast(null, ExceptionHandlingMode.KEEP_GOING, true, "a", "b");
  }

  @Test
  public void failFastNoShutdown() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(5, 5, 0, TimeUnit.SECONDS,
                                                         new LinkedBlockingQueue<Runnable>());
    // In failFast mode, we only run actions queued before the exception.
    assertFailFast(executor, ExceptionHandlingMode.FAIL_FAST, false, "a", "b");

    // In !failFast mode, we complete all queued actions.
    assertFailFast(executor, ExceptionHandlingMode.KEEP_GOING, false, "a", "b", "1", "2");

    // Now check fail-fast on interrupt:
    assertFailFast(executor, ExceptionHandlingMode.KEEP_GOING, true, "a", "b");

    executor.shutdown();
    assertThat(executor.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        .isTrue();
  }

  private static void assertFailFast(
      ThreadPoolExecutor executor,
      ExceptionHandlingMode exceptionHandlingMode,
      boolean interrupt,
      String... expectedVisited)
      throws Exception {
    assertThat(executor == null || !executor.isShutdown()).isTrue();
    AbstractQueueVisitor visitor =
        (executor == null)
            ? new ConcreteQueueVisitor(exceptionHandlingMode)
            : new ConcreteQueueVisitor(executor, exceptionHandlingMode);

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

    Exception e =
        assertThrows(Exception.class, () -> visitor.awaitQuiescence(/*interruptWorkers=*/ false));
    if (interrupt) {
        assertThat(e).isInstanceOf(InterruptedException.class);
      } else {
      assertThat(e).isSameInstanceAs(THROWABLE);
    }
    assertWithMessage("got: " + visitedList + "\nwant: " + Arrays.toString(expectedVisited))
        .that(Sets.newHashSet(visitedList))
        .isEqualTo(Sets.newHashSet(expectedVisited));

    if (executor != null) {
      assertThat(executor.isShutdown()).isFalse();
      assertThat(visitor.getTaskCount()).isEqualTo(0);
    }
  }

  @Test
  public void jobIsInterruptedWhenOtherFails() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(3, 3, 0, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>());

    final AbstractQueueVisitor visitor =
        createQueueVisitorWithConstantErrorClassification(executor, ErrorClassification.CRITICAL);
    final CountDownLatch latch1 = new CountDownLatch(1);
    final AtomicBoolean wasInterrupted = new AtomicBoolean(false);

    Runnable r1 = new Runnable() {

      @Override
      public void run() {
        latch1.countDown();
        try {
          // Interruption is expected during a sleep. There is no sense in fail or assert call
          // because exception is going to be swallowed inside AbstractQueueVisitor.
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
    CountDownLatch exnLatch = visitor.getExceptionLatchForTestingOnly();

    Exception e =
        assertThrows(Exception.class, () -> visitor.awaitQuiescence(/*interruptWorkers=*/ true));
    assertThat(e).isSameInstanceAs(THROWABLE);

    assertThat(wasInterrupted.get()).isTrue();
    assertThat(executor.isShutdown()).isTrue();
    assertThat(exnLatch.await(0, TimeUnit.MILLISECONDS)).isTrue();
  }

  @Test
  public void javaErrorConsideredCriticalNoMatterWhat() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(2, 2, 0, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>());
    final Error error = new Error("bad!");
    AbstractQueueVisitor visitor =
        createQueueVisitorWithConstantErrorClassification(
            executor, ErrorClassification.NOT_CRITICAL);
    final CountDownLatch latch = new CountDownLatch(1);
    final AtomicBoolean sleepFinished = new AtomicBoolean(false);
    final AtomicBoolean sleepInterrupted = new AtomicBoolean(false);
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
    CountDownLatch exnLatch = visitor.getExceptionLatchForTestingOnly();
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
    assertThat(sleepInterrupted.get()).isTrue();
    assertThat(sleepFinished.get()).isFalse();
    assertThat(thrownError).isEqualTo(error);
    assertThat(exnLatch.await(0, TimeUnit.MILLISECONDS)).isTrue();
  }

  private static class ClassifiedException extends RuntimeException {
    private final ErrorClassification classification;

    private ClassifiedException(ErrorClassification classification) {
      this.classification = classification;
    }
  }

  @Test
  public void mostSevereErrorPropagated() throws Exception {
    ThreadPoolExecutor executor = new ThreadPoolExecutor(2, 2, 0, TimeUnit.SECONDS,
        new LinkedBlockingQueue<Runnable>());
    final ClassifiedException criticalException =
        new ClassifiedException(ErrorClassification.CRITICAL);
    final ClassifiedException criticalAndLogException =
        new ClassifiedException(ErrorClassification.CRITICAL_AND_LOG);
    final ErrorClassifier errorClassifier = new ErrorClassifier() {
      @Override
      protected ErrorClassification classifyException(Exception e) {
        return (e instanceof ClassifiedException)
            ? ((ClassifiedException) e).classification
            : ErrorClassification.NOT_CRITICAL;
      }
    };
    AbstractQueueVisitor visitor =
        new AbstractQueueVisitor(
            executor, ExecutorOwnership.PRIVATE, ExceptionHandlingMode.KEEP_GOING, errorClassifier);
    final CountDownLatch exnLatch = visitor.getExceptionLatchForTestingOnly();
    Runnable criticalExceptionRunnable = new Runnable() {
      @Override
      public void run() {
        throw criticalException;
      }
    };
    Runnable criticalAndLogExceptionRunnable = new Runnable() {
      @Override
      public void run() {
        // Wait for the critical exception to be thrown. There's a benign race between our 'await'
        // call completing because the exception latch was counted down, and our thread being
        // interrupted by AbstractQueueVisitor because the critical error was encountered. This is
        // completely fine; all that matters is that we have a chance to throw our error _after_
        // the previous one was thrown by the other Runnable.
        try {
          exnLatch.await();
        } catch (InterruptedException e) {
          // Ignored.
        }
        throw criticalAndLogException;
      }
    };
    visitor.execute(criticalExceptionRunnable);
    visitor.execute(criticalAndLogExceptionRunnable);
    ClassifiedException exn = null;
    try {
      visitor.awaitQuiescence(/*interruptWorkers=*/ true);
    } catch (ClassifiedException e) {
      exn = e;
    }
    assertThat(exn).isEqualTo(criticalAndLogException);
  }

  private static Runnable throwingRunnable() {
    return new Runnable() {
      @Override
      public void run() {
        throw THROWABLE;
      }
    };
  }

  private static Runnable interruptingRunnable(final Thread thread) {
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
          assertThat(
                  interrupt
                      ? visitor.getInterruptionLatchForTestingOnly().await(1, TimeUnit.MINUTES)
                      : visitor.getExceptionLatchForTestingOnly().await(1, TimeUnit.MINUTES))
              .isTrue();
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
      super(
          /* parallelism= */ 5,
          /* keepAliveTime= */ 3L,
          TimeUnit.SECONDS,
          ExceptionHandlingMode.KEEP_GOING,
          THREAD_NAME,
          ErrorClassifier.DEFAULT);
    }

    CountingQueueVisitor(ThreadPoolExecutor executor) {
      super(
          executor,
          ExecutorOwnership.SHARED,
          ExceptionHandlingMode.FAIL_FAST,
          ErrorClassifier.DEFAULT);
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

    ConcreteQueueVisitor() {
      super(
          5,
          3L,
          TimeUnit.SECONDS,
          ExceptionHandlingMode.KEEP_GOING,
          THREAD_NAME,
          ErrorClassifier.DEFAULT);
    }

    ConcreteQueueVisitor(ExceptionHandlingMode exceptionHandlingMode) {
      super(5, 3L, TimeUnit.SECONDS, exceptionHandlingMode, THREAD_NAME, ErrorClassifier.DEFAULT);
    }

    ConcreteQueueVisitor(ThreadPoolExecutor executor, ExceptionHandlingMode exceptionHandlingMode) {
      super(executor, ExecutorOwnership.SHARED, exceptionHandlingMode, ErrorClassifier.DEFAULT);
    }
  }

  private static AbstractQueueVisitor createQueueVisitorWithConstantErrorClassification(
      ThreadPoolExecutor executor, final ErrorClassification classification) {
    return new AbstractQueueVisitor(
        executor,
        ExecutorOwnership.PRIVATE,
        ExceptionHandlingMode.KEEP_GOING,
        new ErrorClassifier() {
          @Override
          protected ErrorClassification classifyException(Exception e) {
            return classification;
          }
        });
  }
}
