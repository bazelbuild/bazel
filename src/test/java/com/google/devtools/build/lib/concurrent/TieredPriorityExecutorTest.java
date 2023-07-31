// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.common.util.concurrent.Uninterruptibles.awaitUninterruptibly;
import static com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class TieredPriorityExecutorTest {
  private static final long INTERRUPT_POLL_MS = 100;
  private static final int POOL_SIZE = 10;
  private static final int CPU_PERMITS = 4;

  @Rule public final TestName testName = new TestName();

  private TieredPriorityExecutor executor;

  @Before
  public void setUp() {
    executor =
        new TieredPriorityExecutor(
            testName.getMethodName(), POOL_SIZE, CPU_PERMITS, ErrorClassifier.DEFAULT);
  }

  @Test
  public void constructor_rejectsLargePoolSize() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new TieredPriorityExecutor(
                "pool too large",
                /* poolSize= */ 0x8000,
                /* cpuPermits= */ 100,
                ErrorClassifier.DEFAULT));
  }

  @Test
  public void task_executes() throws InterruptedException {
    var receiver = new AtomicInteger();
    executor.execute(() -> receiver.set(1));
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(receiver.get()).isEqualTo(1);
  }

  @Test
  public void enqueuedTasks_execute() throws InterruptedException {
    var receiver = new AtomicInteger();
    CountDownLatch poolFull = new CountDownLatch(POOL_SIZE);
    CountDownLatch go = new CountDownLatch(1);
    for (int i = 0; i < 100; ++i) {
      executor.execute(
          () -> {
            poolFull.countDown();
            awaitUninterruptibly(go);
            receiver.incrementAndGet();
          });
    }
    // Waits for the pool to fill to ensure tasks have been enqueued.
    if (!poolFull.await(WAIT_TIMEOUT_SECONDS, SECONDS)) {
      fail("timed out waiting for " + POOL_SIZE + " tasks to start: " + executor);
    }
    go.countDown();
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(receiver.get()).isEqualTo(100);
  }

  @Test
  public void cpuHeavyTasks_runInPriorityOrder() throws InterruptedException {
    // Holds all but 1 of the CPU permits. This sequentializes execution on the remaining permit so
    // that the order of execution can be observed. This exercises some of the code that performs
    // CPU permit accounting.
    CountDownLatch allBlockersStarted = new CountDownLatch(CPU_PERMITS);
    CountDownLatch holdCpuPermits = new CountDownLatch(1);
    for (int i = 0; i < CPU_PERMITS - 1; ++i) {
      executor.execute(
          new CpuHeavyRunnable(
              i,
              () -> {
                allBlockersStarted.countDown();
                awaitUninterruptibly(holdCpuPermits);
              }));
    }

    CountDownLatch gate = new CountDownLatch(1);
    executor.execute(
        new CpuHeavyRunnable(
            CPU_PERMITS - 1,
            () -> {
              allBlockersStarted.countDown();
              awaitUninterruptibly(gate);
            }));

    if (!allBlockersStarted.await(WAIT_TIMEOUT_SECONDS, SECONDS)) {
      fail("timed out waiting for initial threads to start: " + executor);
    }

    var sequence = new AtomicInteger(100);
    CountDownLatch sequenceCheckDone = new CountDownLatch(100 - CPU_PERMITS);
    // Tasks are inserted in reverse-priority order. If tasks execute early, they will cause
    // assertion failures.
    for (int i = CPU_PERMITS; i < 100; ++i) {
      final int index = i;
      executor.execute(
          new CpuHeavyRunnable(
              i,
              () -> {
                assertThat(sequence.getAndSet(index)).isEqualTo(index + 1);
                sequenceCheckDone.countDown();
              }));
    }

    gate.countDown(); // Releases a CPU permit now that the test payload has been enqueued.

    while (!sequenceCheckDone.await(INTERRUPT_POLL_MS, MILLISECONDS)) {
      if (executor.isCancelledForTestingOnly()) {
        break; // If the threads are cancelled, remaining checks do not run.
      }
    }
    holdCpuPermits.countDown();
    executor.awaitQuiescence(/* interruptWorkers= */ true);
  }

  @Test
  public void nonCpuHeavyComparableRunnable_ignoresPriority() throws InterruptedException {
    // This execution would crash if the comparator were invoked because it throws
    // UnsupportedOperationException.
    var receiver = new AtomicInteger();
    for (int i = 0; i < 100; ++i) {
      executor.execute(new NonCpuHeavyComparable(receiver::getAndIncrement));
    }
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(receiver.get()).isEqualTo(100);
  }

  @Test
  public void cpuHeavyTasks_haveLowPriority() throws InterruptedException {
    var holdAllButOneThread = new CountDownLatch(1);
    for (int i = 0; i < POOL_SIZE - 1; ++i) {
      executor.execute(() -> awaitUninterruptibly(holdAllButOneThread));
    }

    var gate = new CountDownLatch(1);
    executor.execute(() -> awaitUninterruptibly(gate));

    var load = new CountDownLatch(20);
    var received = new ArrayList<>();

    // Outputs the sequence 9..0 to `received`.
    for (int i = 0; i < 10; ++i) {
      final int index = i;
      executor.execute(
          new CpuHeavyRunnable(
              i,
              () -> {
                received.add(index);
                load.countDown();
              }));
    }

    // Outputs 10 -1s to `received`. Even though these are submitted after, the executor
    // prioritizes them first because they are not CPU-heavy.
    for (int i = 0; i < 10; ++i) {
      executor.execute(
          () -> {
            received.add(-1);
            load.countDown();
          });
    }

    gate.countDown();
    if (!load.await(WAIT_TIMEOUT_SECONDS, SECONDS)) {
      fail("timed out waiting for tasks to execute: " + executor);
    }
    holdAllButOneThread.countDown();
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(received)
        .containsExactly(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  }

  @Test
  public void nonCriticalUncaughtError_propagates() throws InterruptedException {
    executor.execute(
        () -> {
          throw new IllegalStateException("error");
        });
    assertThrows(
        IllegalStateException.class, () -> executor.awaitQuiescence(/* interruptWorkers= */ true));
  }

  @Test
  public void criticalError_interruptsTasksAndPropagates() throws InterruptedException {
    var interrupted = new AtomicBoolean(false);
    var never = new CountDownLatch(1);
    executor.execute(
        () -> {
          try {
            while (!never.await(INTERRUPT_POLL_MS, MILLISECONDS)) {}
          } catch (InterruptedException e) {
            interrupted.set(true);
            return;
          }
        });

    executor.execute(
        () -> {
          throw new AssertionError("critical");
        });
    assertThrows(
        AssertionError.class, () -> executor.awaitQuiescence(/* interruptWorkers= */ true));
    assertThat(interrupted.get()).isTrue();
  }

  @Test
  public void error_upgradesToHigherPriority() throws InterruptedException {
    var holdAllButOneThread = new CountDownLatch(1);
    for (int i = 0; i < POOL_SIZE - 1; ++i) {
      executor.execute(
          () -> {
            try {
              while (!holdAllButOneThread.await(INTERRUPT_POLL_MS, MILLISECONDS)) {}
            } catch (InterruptedException e) {
              return;
            }
          });
    }

    var gate = new CountDownLatch(1);
    executor.execute(() -> awaitUninterruptibly(gate));

    executor.execute(
        new CpuHeavyRunnable(
            10,
            () -> {
              throw new IllegalStateException("lower priority error");
            }));

    // Lower priority to run afterwards, to replace lower priority error.
    executor.execute(
        new CpuHeavyRunnable(
            1,
            () -> {
              throw new AssertionError("higher priority error");
            }));

    gate.countDown();
    assertThrows(
        AssertionError.class, () -> executor.awaitQuiescence(/* interruptWorkers= */ true));
  }

  @Test
  public void catastrophe_exitsBeforeQuiescence() throws InterruptedException {
    var holdAll = new CountDownLatch(1);
    for (int i = 0; i < POOL_SIZE; ++i) {
      executor.execute(() -> awaitUninterruptibly(holdAll));
    }

    executor.execute(new BadTask());
    executor.execute(new BadTask());

    assertThrows(
        UnsupportedOperationException.class,
        () -> executor.awaitQuiescence(/* interruptWorkers= */ true));

    // Reaching here means that awaitQuiescence completed with tasks still blocked. Releasing them
    // so they may be cleaned up after the fact.
    holdAll.countDown();
  }

  @Test
  public void interrupt_interruptsWorkersAndThrowsInterrupted() throws InterruptedException {
    var interrupted = new AtomicBoolean(false);
    var never = new CountDownLatch(1);
    executor.execute(
        () -> {
          try {
            while (!never.await(INTERRUPT_POLL_MS, MILLISECONDS)) {}
          } catch (InterruptedException e) {
            interrupted.set(true);
            return;
          }
        });

    Thread thread =
        new Thread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () -> executor.awaitQuiescence(/* interruptWorkers= */ true)));

    thread.start();
    thread.interrupt();
    thread.join();
    assertThat(interrupted.get()).isTrue();
  }

  @Test
  public void interruptWithoutFlag_doesNotInterruptWorkers() throws InterruptedException {
    var interrupted = new AtomicBoolean(false);
    var holdAll = new CountDownLatch(1);
    for (int i = 0; i < 100; ++i) {
      executor.execute(
          () -> {
            try {
              while (!holdAll.await(INTERRUPT_POLL_MS, MILLISECONDS)) {}
            } catch (InterruptedException e) {
              interrupted.set(true);
              return;
            }
          });
    }

    var thread =
        new Thread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () -> executor.awaitQuiescence(/* interruptWorkers= */ false)));

    thread.start();
    thread.interrupt();
    waitForInterruptPolling(); // Waits for the interrupt to be noticed.
    // There's an unlikely race here that may allow the test to pass even if the worker threads are
    // interrupted.
    // TODO(shahan): think about how to make this tighter.
    holdAll.countDown();
    thread.join();
    assertThat(interrupted.get()).isFalse();
  }

  @Test
  public void interruptWithAdditionalError_throwsError() throws InterruptedException {
    var hold = new CountDownLatch(1);
    executor.execute(() -> awaitUninterruptibly(hold));
    executor.execute(
        () -> {
          throw new IllegalStateException("error");
        });
    var thread =
        new Thread(
            () ->
                assertThrows(
                    IllegalStateException.class,
                    () -> executor.awaitQuiescence(/* interruptWorkers= */ true)));
    thread.start();
    thread.interrupt();
    waitForInterruptPolling(); // Waits for the interrupt to be noticed.
    hold.countDown();
    thread.join();
  }

  private void waitForInterruptPolling() throws InterruptedException {
    Thread.sleep(INTERRUPT_POLL_MS);
  }

  @Test
  public void afterError_poolIsReusable() throws InterruptedException {
    executor.execute(
        () -> {
          throw new IllegalStateException("error");
        });
    assertThrows(
        IllegalStateException.class, () -> executor.awaitQuiescence(/* interruptWorkers= */ true));

    var receiver = new AtomicInteger();
    for (int i = 0; i < 100; ++i) {
      executor.execute(receiver::getAndIncrement);
    }
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(receiver.get()).isEqualTo(100);
  }

  @Test
  public void afterCriticalError_poolIsReusable() throws InterruptedException {
    var never = new CountDownLatch(1);
    var interrupted = new AtomicBoolean();
    executor.execute(
        () -> {
          try {
            while (!never.await(INTERRUPT_POLL_MS, MILLISECONDS)) {}
          } catch (InterruptedException e) {
            interrupted.set(true);
            return;
          }
        });
    executor.execute(
        () -> {
          throw new AssertionError("error");
        });
    assertThrows(
        AssertionError.class, () -> executor.awaitQuiescence(/* interruptWorkers= */ true));
    assertThat(interrupted.get()).isTrue();

    var receiver = new AtomicInteger();
    for (int i = 0; i < 100; ++i) {
      executor.execute(receiver::getAndIncrement);
    }
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(receiver.get()).isEqualTo(100);
  }

  @Test
  public void cleaner_disposesPool() throws InterruptedException {
    var referenceQueue = new ReferenceQueue<ForkJoinPool>();
    var poolRef = executor.registerPoolDisposalMonitorForTesting(referenceQueue);

    // Runs some tasks on the pool to make sure it has some live threads.
    var receiver = new AtomicInteger();
    for (int i = 0; i < 100; ++i) {
      executor.execute(receiver::getAndIncrement);
    }
    executor.awaitQuiescence(/* interruptWorkers= */ true);
    assertThat(receiver.get()).isEqualTo(100);

    // Disposing the TieredPriorityExecutor should cause its cleaner to cleanup the ForkJoinPool.
    executor = null;
    do {
      System.gc();
      var ref = referenceQueue.poll();
      if (ref != null) {
        assertThat(ref).isEqualTo(poolRef);
        break;
      }
      Thread.sleep(INTERRUPT_POLL_MS);
    } while (true);
  }

  @Test
  public void criticalError_disposesPool() throws InterruptedException {
    var referenceQueue = new ReferenceQueue<ForkJoinPool>();
    var poolRef = executor.registerPoolDisposalMonitorForTesting(referenceQueue);

    for (int i = 0; i < 100; ++i) {
      executor.execute(
          () -> {
            throw new IllegalStateException();
          });
    }
    executor.execute(
        () -> {
          throw new AssertionError();
        });

    assertThrows(
        AssertionError.class, () -> executor.awaitQuiescence(/* interruptWorkers= */ true));

    // The critical error should cause the executor to cleanup its internal thread pool.
    do {
      System.gc();
      var ref = referenceQueue.poll();
      if (ref != null) {
        assertThat(ref).isEqualTo(poolRef);
        break;
      }
      Thread.sleep(INTERRUPT_POLL_MS);
    } while (true);
  }

  @Test
  public void settableFuture_respondsToInterrupt() throws InterruptedException {
    var interruptedCount = new AtomicInteger();
    var allStarted = new CountDownLatch(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i) {
      Future<Void> future = SettableFuture.create();
      executor.execute(
          () -> {
            allStarted.countDown();
            try {
              future.get();
            } catch (InterruptedException e) {
              interruptedCount.getAndIncrement();
            } catch (ExecutionException e) {
              throw new IllegalStateException(e);
            }
          });
    }

    Thread thread =
        new Thread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () -> executor.awaitQuiescence(/* interruptWorkers= */ true)));

    thread.start();

    // Waits for all threads to start before interrupting. Object.wait does not appear to respond to
    // the interrupt status unless there is explicit polling. This test verifies that Future.get
    // doesn't require similar polling. For the test to be meaningful, the call to Future.get has
    // to happen before the interrupt happens. It's hard to guarantee this, but chances are small
    // that all the above threads will start and not call Future.get until after the interrupt below
    // propagates.
    if (!allStarted.await(WAIT_TIMEOUT_SECONDS, SECONDS)) {
      fail("timed out waiting for threads to start: " + executor);
    }

    thread.interrupt();
    thread.join();

    assertThat(interruptedCount.get()).isEqualTo(POOL_SIZE);
  }

  @Test
  public void listenableFuture_respondsToInterrupt() throws InterruptedException {
    var interruptedCount = new AtomicInteger();
    var allStarted = new CountDownLatch(POOL_SIZE);

    ListenableFuture<Integer> rootFuture = SettableFuture.create();
    for (int i = 0; i < POOL_SIZE; ++i) {
      ListenableFuture<String> future =
          Futures.transform(rootFuture, x -> x.toString(), directExecutor());
      executor.execute(
          () -> {
            allStarted.countDown();
            try {
              future.get();
            } catch (InterruptedException e) {
              interruptedCount.getAndIncrement();
            } catch (ExecutionException e) {
              throw new IllegalStateException(e);
            }
          });
    }

    Thread thread =
        new Thread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () -> executor.awaitQuiescence(/* interruptWorkers= */ true)));

    thread.start();

    // Waits for all threads to start before interrupting.
    if (!allStarted.await(WAIT_TIMEOUT_SECONDS, SECONDS)) {
      fail("timed out waiting for tasks to start:" + executor);
    }

    thread.interrupt();
    thread.join();

    assertThat(interruptedCount.get()).isEqualTo(POOL_SIZE);
  }

  @Test
  public void taskQueueOverflow_executesTasks() throws InterruptedException {
    var allHoldersStarted = new CountDownLatch(POOL_SIZE);
    var holdAllThreads = new CountDownLatch(1);
    for (int i = 0; i < POOL_SIZE; ++i) {
      executor.execute(
          () -> {
            allHoldersStarted.countDown();
            awaitUninterruptibly(holdAllThreads);
          });
    }

    // Waits for holders to start, otherwise they might race against the filling of the queue below.
    allHoldersStarted.await();

    // Over-fills the queue.
    var executed = Sets.<Integer>newConcurrentHashSet();
    var expected = new ArrayList<Integer>();
    for (int i = 0; i < 2 * PriorityWorkerPool.TASKS_MAX_VALUE; ++i) {
      expected.add(i);

      final int index = i;
      executor.execute(() -> executed.add(index));
    }

    assertThat(executed).isEmpty();

    holdAllThreads.countDown();
    executor.awaitQuiescence(/* interruptWorkers= */ true);

    assertThat(executed).containsExactlyElementsIn(expected);
  }

  @Test
  public void taskQueueOverflow_doesNotExecuteWhenCancelled() throws InterruptedException {
    var holdAllThreads = new CountDownLatch(1);
    for (int i = 0; i < POOL_SIZE; ++i) {
      executor.execute(() -> awaitUninterruptibly(holdAllThreads));
    }

    Thread thread =
        new Thread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () -> executor.awaitQuiescence(/* interruptWorkers= */ true)));
    thread.start();

    // Interrupts the executor and waits for the interrupt to be noticed.
    thread.interrupt();
    do {
      waitForInterruptPolling();
    } while (!executor.isCancelledForTestingOnly());

    // Overfills the queue: none of these should run due to the cancellation, but they are enqueued
    // internally until the queue overflows, after which they are dropped.
    var executed = new ArrayList<Integer>();
    for (int i = 0; i < 2 * PriorityWorkerPool.TASKS_MAX_VALUE; ++i) {
      final int index = i;
      executor.execute(() -> executed.add(index));
    }

    holdAllThreads.countDown();
    thread.join();

    assertThat(executed).isEmpty();
  }

  @Test
  public void fjpExecute_alwaysStartsThreads() throws InterruptedException {
    // This test demonstrates a category of flakes that may rarely occur in other tests in this file
    // caused by a (JDK Bug)[https://bugs.openjdk.org/browse/JDK-8292969] that is fixed in more
    // recent versions of Java.
    ForkJoinPool pool = new ForkJoinPool(10);
    CountDownLatch allStarted = new CountDownLatch(10);
    CountDownLatch gate = new CountDownLatch(1);
    Runnable task =
        () -> {
          allStarted.countDown();
          awaitUninterruptibly(gate);
        };

    for (int i = 0; i < 10; ++i) {
      pool.execute(task);
    }

    if (!allStarted.await(WAIT_TIMEOUT_SECONDS, SECONDS)) {
      fail("timed out waiting: " + pool);
    }

    gate.countDown();
    pool.shutdown();
    assertThat(pool.awaitQuiescence(WAIT_TIMEOUT_SECONDS, SECONDS)).isTrue();
  }

  private static class CpuHeavyRunnable implements ComparableRunnable {
    /** NOTE: must be unique for correct semantics. */
    private final int priority;

    private final Runnable task;

    private CpuHeavyRunnable(int priority, Runnable task) {
      this.priority = priority;
      this.task = task;
    }

    @Override
    public boolean isCpuHeavy() {
      return true;
    }

    @Override
    public void run() {
      task.run();
    }

    @Override
    public int compareTo(ComparableRunnable other) {
      return Integer.compare(((CpuHeavyRunnable) other).priority, priority);
    }

    @Override
    public String toString() {
      return toStringHelper(this).add("priority", priority).toString();
    }
  }

  /**
   * This class has a bad {@link Comparable} implementation.
   *
   * <p>It's used to show that if the {@link ComparableRunnable#isCpuHeavy} flag is not set,
   * prioritization is not used.
   */
  private static class NonCpuHeavyComparable implements ComparableRunnable {
    private final Runnable task;

    private NonCpuHeavyComparable(Runnable task) {
      this.task = task;
    }

    @Override
    public void run() {
      task.run();
    }

    @Override
    public int compareTo(ComparableRunnable other) {
      throw new UnsupportedOperationException();
    }
  }

  /** This class can be used to inject catastrophes into the executor for testing. */
  private static class BadTask implements ComparableRunnable {
    @Override
    public boolean isCpuHeavy() {
      return true;
    }

    @Override
    public void run() {
      throw new UnsupportedOperationException();
    }

    @Override
    public int compareTo(ComparableRunnable other) {
      throw new UnsupportedOperationException();
    }
  }
}
