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
import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Throwables.throwIfUnchecked;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.ListenableFuture;
import java.lang.ref.Cleaner;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import javax.annotation.Nullable;

/**
 * An executor that prioritizes tasks and supports work donation.
 *
 * <p>This executor divides work into two tiers, CPU-heavy and non-CPU-heavy.
 *
 * <ul>
 *   <li><i>Non-CPU-heavy</i> tasks tend to be leaf-level and are serviced first-come first serve
 *       with higher priority than CPU-heavy tasks.
 *   <li><i>CPU-heavy</i> tasks are placed into a priority queue and serviced based on priority,
 *       requiring an additional CPU permit to be scheduled, to avoid oversubscription.
 * </ul>
 *
 * <p>The queue for non-CPUHeavy tasks has a fixed capacity. When full, callers of execute assist
 * with enqueued work.
 */
public final class TieredPriorityExecutor implements QuiescingExecutor {
  /** A common cleaner shared by all executors. */
  private static final Cleaner poolCleaner = Cleaner.create();

  private final PriorityWorkerPool pool;

  /**
   * An unchecked exception when submitting a job is catastrophic.
   *
   * <p>It could mean an inconsistent state so {@link #awaitQuiescence} attempts to throw
   * immediately instead of waiting if this occurs to avoid becoming unresponsive.
   */
  private volatile Throwable catastrophe;

  public TieredPriorityExecutor(
      String name, int poolSize, int cpuPermits, ErrorClassifier errorClassifier) {
    checkArgument(
        poolSize >= cpuPermits, "expected poolSize=%s >= cpuPermits=%s", poolSize, cpuPermits);
    this.pool = new PriorityWorkerPool(poolCleaner, name, poolSize, cpuPermits, errorClassifier);
    // Registers a cleanup procedure for the underlying pool when this object is eligible for
    // garbage collection. This has to be done explicitly because threads are GC roots.
    poolCleaner.register(this, pool::dispose);
  }

  @Override
  public void execute(Runnable task) {
    try {
      pool.execute(task);
    } catch (Throwable uncaught) {
      pool.cancel();
      catastrophe = uncaught;
      synchronized (pool.quiescenceMonitor()) {
        pool.quiescenceMonitor().notify();
      }
    }
  }

  /**
   * Returns after waiting for all pending work to complete.
   *
   * <p>There are various error scenarios. Except in the case of <b>catastrophe</b>, it should be
   * safe to reuse the pool.
   *
   * <ul>
   *   <li><b>Uncaught Error in Task</b>.
   *       <ul>
   *         <li>{@link ErrorClassifier.ErrorClassification#CRITICAL} or higher: Stops processing
   *             the queue and interrupts in-flight threads. Waits for in-flight threads to
   *             complete. Resets the executor to a clean (reusable) state and throws the error
   *             (based on prioritization given by the {@link ErrorClassifier} if there were
   *             multiple) to the caller.
   *         <li>{@link ErrorClassifier.ErrorClassification#NOT_CRITICAL}: completes processing as
   *             usual and throws the exception (unless one with higher priority occurred).
   *       </ul>
   *   <li><b>Interrupted</b>: the calling thread is interrupted.
   *       <ul>
   *         <li>{@code interruptWorkers=true}: Stops processing the queue and interrupts in-flight
   *             threads. Waits for in-flight threads to complete. Resets the executor to a clean
   *             (reusable) state. Finally rethrows the {@link InterruptedException}, unless there
   *             was another uncaught exception, which gets thrown instead.
   *         <li>{@code interruptWorkers=false}: allows work to drain normally. Throws {@link
   *             InterruptedException} unless there is another uncaught exception to throw.
   *       </ul>
   *   <li><b>Catastrophe</b>: uncaught error in the act of submitting a task. Stops queue
   *       processing and interrupts all in-flight threads. Throws the error without waiting for
   *       tasks to drain, leaving this executor in an inconsistent state. The goal here is to avoid
   *       becoming unresponsive.
   * </ul>
   */
  @Override
  public void awaitQuiescence(boolean interruptWorkers) throws InterruptedException {
    InterruptedException interruptedException = null;
    while (true) {
      try {
        synchronized (pool.quiescenceMonitor()) {
          while (!pool.isQuiescent() && catastrophe == null) {
            pool.quiescenceMonitor().wait();
          }
        }
        break;
      } catch (InterruptedException e) {
        interruptedException = e;
        if (interruptWorkers) {
          pool.cancel();
        }
      }
    }
    throwIfNonNull(catastrophe);
    var unhandled = pool.unhandled();
    pool.reset();

    throwIfNonNull(unhandled);
    if (interruptedException != null) {
      throw interruptedException;
    }
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    // TODO(shahan): improve support if needed. This executor is currently only used for analysis,
    // which does not use futures.
    throw new UnsupportedOperationException();
  }

  @Override
  @VisibleForTesting
  public CountDownLatch getExceptionLatchForTestingOnly() {
    throw new UnsupportedOperationException();
  }

  @Override
  @VisibleForTesting
  public CountDownLatch getInterruptionLatchForTestingOnly() {
    throw new UnsupportedOperationException();
  }

  /**
   * The parallelism target of the underlying thread pool.
   *
   * <p>Public to allow clients to examine this executor's configuration and suitability for reuse.
   */
  public int poolSize() {
    return pool.poolSize();
  }

  /**
   * The number of tokens available to CPU-heavy tasks, which must acquire them before execution.
   *
   * <p>This constrains parallelism of CPU-heavy tasks and must be less than {@link #poolSize}.
   *
   * <p>Public to allow clients to examine this executor's configuration and suitability for reuse.
   */
  public int cpuPermits() {
    return pool.cpuPermits();
  }

  /**
   * True if this executor had a catastrophic failure, making it unsuitable for reuse.
   *
   * <p>Public to allow clients to assess this executor's suitability for reuse.
   */
  public boolean hasCatastrophe() {
    return catastrophe != null;
  }

  /**
   * Hook for testing that the underlying {@link ForkJoinPool} is properly garbage collected.
   *
   * <p>This is important for preventing memory leaks and subtle because active threads are garbage
   * collection roots and have back references to their owning pools.
   */
  @VisibleForTesting
  PhantomReference<ForkJoinPool> registerPoolDisposalMonitorForTesting(
      ReferenceQueue<ForkJoinPool> referenceQueue) {
    return pool.registerPoolDisposalMonitorForTesting(referenceQueue);
  }

  @VisibleForTesting
  boolean isCancelledForTestingOnly() {
    return pool.isCancelled();
  }

  @Override
  public String toString() {
    return toStringHelper(this).add("pool", pool).add("catastrophe", catastrophe).toString();
  }

  /**
   * Throws an unchecked exception if {@code e} is non-null.
   *
   * <p>If {@code e} is an unchecked exception, it'll be thrown as-is. Otherwise, it'll be wrapped
   * and thrown as an {@link IllegalArgumentException}, which isn't expected here.
   */
  private static void throwIfNonNull(@Nullable Throwable e) {
    if (e == null) {
      return;
    }
    throwIfUnchecked(e);
    throw new IllegalArgumentException("Unexpected checked exception.", e);
  }
}
