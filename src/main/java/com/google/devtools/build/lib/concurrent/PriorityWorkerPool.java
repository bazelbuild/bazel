// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.DO_CPU_HEAVY_TASK;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.DO_TASK;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.IDLE;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.QUIESCENT;
import static java.lang.Thread.currentThread;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import sun.misc.Unsafe;

/**
 * Inner implementation of {@link TieredPriorityExecutor}.
 *
 * <p>The main motivation for this additional layering is to facilitate garbage collection. The
 * {@link PriorityWorkerPool#WorkerThread}s have references to their enclosing {@link
 * PriorityWorkerPool}. Since threads are garbage collection roots, this makes the entire {@link
 * PriorityWorkerPool} ineligible for garbage collection.
 *
 * <p>The {@link PriorityWorkerPool} has no backreferences to the enclosing {@link
 * TieredPriorityExecutor}, so the {@link TieredPriorityExecutor} is eligible for garbage collection
 * and is able to perform cleanup tasks.
 */
final class PriorityWorkerPool {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final String name;
  private final int poolSize;
  private final int cpuPermits;

  private ForkJoinPool pool;

  /**
   * Cache of workers for interrupt handling.
   *
   * <p>A {@link Cache} allows us to use weak keys, which are the only relevant objects here. The
   * values are an unintentional side effect of the library and populated arbitrarily.
   */
  private final Cache<WorkerThread, Object> workers = Caffeine.newBuilder().weakKeys().build();

  /**
   * Counters describing the state of queues and in-flight work.
   *
   * <p>This field is updated atomically using {@link #tryUpdateAvailable}.
   */
  private volatile AvailableState available;

  private final ErrorClassifier errorClassifier;

  /**
   * Queue for tasks not needing prioritization.
   *
   * <p>An interesting alternative to consider is to place unprioritized tasks directly into {@link
   * #pool}, which could reduce the work performed by the system. Doing this results in about a
   * {@code 4%} end-to-end analysis regression in our benchmark. The reasons are not clear, but
   * perhaps polling from a {@link ConcurrentLinkedQueue}, as implemented in {@link
   * WorkerThread#runLoop} is more efficient than random scanning of {@link ForkJoinPool}, or it
   * could be a domain specific reason having to do with differences in the resulting task ordering.
   */
  private final ConcurrentLinkedQueue<Runnable> queue = new ConcurrentLinkedQueue<>();

  private final ConcurrentSkipListSet<ComparableRunnable> cpuHeavyQueue =
      new ConcurrentSkipListSet<>();

  /**
   * A synchronization mechanism used when waiting for quiescence.
   *
   * <p>Quiescence can be reached if either of the following conditions is satisfied.
   *
   * <ul>
   *   <li>No tasks are enqueued and all workers are idle.
   *   <li>There was a catastrophe (see {@link TieredPriorityExecutor#awaitQuiescence}).
   * </ul>
   */
  private final Object quiescenceMonitor = new Object();

  /**
   * The most severe unhandled exception thrown by a worker thread, according to {@link
   * #errorClassifier}. This exception gets propagated to the calling thread of {@link
   * TieredPriorityExecutor#awaitQuiescence} . We use the most severe error for the sake of not
   * masking e.g. crashes in worker threads after the first critical error that can occur due to
   * race conditions in client code.
   *
   * <p>Field updates happen only in blocks that are synchronized on the {@link
   * AbstractQueueVisitor} object.
   *
   * <p>If {@link AbstractQueueVisitor} clients don't like the semantics of storing and propagating
   * the most severe error, then they should be provide an {@link ErrorClassifier} that does the
   * right thing (e.g. to cause the _first_ error to be propagated, you'd want to provide an {@link
   * ErrorClassifier} that gives all errors the exact same {@link
   * ErrorClassifier.ErrorClassification}).
   *
   * <p>Note that this is not a performance-critical path.
   */
  private final AtomicReference<Throwable> unhandled = new AtomicReference<>();

  PriorityWorkerPool(String name, int poolSize, int cpuPermits, ErrorClassifier errorClassifier) {
    this.name = name;
    this.poolSize = poolSize;
    this.cpuPermits = cpuPermits;

    this.pool = newForkJoinPool();
    this.available = new AvailableState(poolSize, cpuPermits);
    this.errorClassifier = errorClassifier;
  }

  int poolSize() {
    return poolSize;
  }

  int cpuPermits() {
    return cpuPermits;
  }

  void execute(Runnable rawTask) {
    if (rawTask instanceof ComparableRunnable) {
      var task = (ComparableRunnable) rawTask;
      if (task.isCpuHeavy()) {
        cpuHeavyQueue.add(task);
        if (acquireThreadAndCpuPermitElseReleaseCpuHeavyTask()) {
          pool.execute(RUN_CPU_HEAVY_TASK);
        }
        return;
      }
    }

    // The approach here pessimizes task placement into the queue. To understand why, consider the
    // naive, optimistic approach below.
    //
    // Try to acquire a thread.
    // A. If successful, execute in the worker.
    // B. Otherwise, enqueue the task and release a task token.
    //
    // The problem is that B may cause the task to become stranded. While enqueuing the task, but
    // before releasing the task token, all workers could complete without seeing the token.
    // Instead, the queue must be populated first, in case the token needs to be released.
    queue.add(rawTask);
    if (acquireThreadElseReleaseTask()) {
      pool.execute(RUN_TASK);
    }
  }

  /**
   * An object to {@link Object#wait} or {@link Object#notifyAll} on for quiescence.
   *
   * <p>The pool will {@link Object#notifyAll} this object when {@link #isQuiescent} becomes true.
   */
  Object quiescenceMonitor() {
    return quiescenceMonitor;
  }

  boolean isQuiescent() {
    return available.isQuiescent(poolSize);
  }

  @Nullable
  Throwable unhandled() {
    return unhandled.get();
  }

  /**
   * Sets the pool to stop processing tasks and interrupts all workers.
   *
   * <p>Calling cancel is on a cancelled pool is a noop. A cancelled pool can be {@link #reset}.
   */
  void cancel() {
    var target = new AvailableState();
    AvailableState snapshot;
    do {
      snapshot = available;
      if (snapshot.isCancelled()) {
        return;
      }
      snapshot.cancel(poolSize, cpuPermits, target);
    } while (!tryUpdateAvailable(snapshot, target));

    workers.asMap().keySet().forEach(Thread::interrupt);
  }

  boolean isCancelled() {
    return available.isCancelled();
  }

  /**
   * Shuts down a pool and frees all resources.
   *
   * <p>Requires that the pool is quiescent.
   */
  private void cleanup() {
    checkState(isQuiescent(), "cleanup called on pool that was not quiescent: %s", this);

    // There's no particular significance to the teardown order here, given that the pool is
    // quiescent. It has the appearance of a logical ordering for cosmetic reasons only.
    queue.clear();
    cpuHeavyQueue.clear();
    workers.invalidateAll();
    pool.shutdown();
  }

  /** Cleans up then resets this pool. */
  void reset() {
    cleanup();
    unhandled.set(null);

    available = new AvailableState(poolSize, cpuPermits);
    pool = newForkJoinPool();
  }

  /**
   * Makes this pool eligible for garbage collection.
   *
   * <p>Intended for registration with {@link java.lang.ref.Cleaner}.
   */
  void dispose() {
    cancel();
    synchronized (quiescenceMonitor) {
      while (!isQuiescent()) {
        // This should only be reachable if there was a catastrophe because otherwise
        // `TieredPriorityExecutor.awaitQuiescence` would have already ensured quiescence. It's
        // not clear that Bazel can recover from this state. The appropriate action, nonetheless, is
        // to wait for quiescence, then shutdown the pool.
        try {
          quiescenceMonitor.wait();
        } catch (InterruptedException e) {
          // We don't expect this to ever happen, given this is running on a cleaner thread. Logs a
          // warning in case it somehow happens.
          logger.atWarning().withCause(e).log("%s interrupted while cleaning up.", this);
        }
      }
    }
    cleanup();
  }

  @VisibleForTesting
  PhantomReference<ForkJoinPool> registerPoolDisposalMonitorForTesting(
      ReferenceQueue<ForkJoinPool> referenceQueue) {
    return new PhantomReference<>(pool, referenceQueue);
  }

  @Override
  public String toString() {
    var threadStates = new TreeMap<Thread.State, Integer>();
    for (var w : workers.asMap().keySet()) {
      threadStates.compute(w.getState(), (k, v) -> v == null ? 1 : (v + 1));
    }
    return toStringHelper(this)
        .add("available", available)
        .add("|queue|", queue.size())
        .add("|cpu queue|", cpuHeavyQueue.size())
        .add("threads", threadStates)
        .toString();
  }

  /**
   * Handles errors created by submitted tasks.
   *
   * <p>Behavior adheres to documentation of {@link TieredPriorityExecutor#awaitQuiescence}.
   */
  private void handleUncaughtError(Throwable error) {
    boolean critical = false;
    var classification = errorClassifier.classify(error);
    switch (classification) {
      case AS_CRITICAL_AS_POSSIBLE:
      case CRITICAL_AND_LOG:
        logger.atWarning().withCause(error).log("Found critical error in queue visitor");
        // fall through
      case CRITICAL:
        critical = true;
        break;
      case NOT_CRITICAL:
        break;
    }

    Throwable unhandledSnapshot;
    do {
      unhandledSnapshot = unhandled.get();
      if (unhandledSnapshot != null
          && errorClassifier.classify(unhandledSnapshot).compareTo(classification) >= 0) {
        break; // Skips saving anything less severe.
      }
    } while (!unhandled.compareAndSet(unhandledSnapshot, error));

    if (critical) {
      cancel();
    }
  }

  private ForkJoinPool newForkJoinPool() {
    return new ForkJoinPool(
        poolSize,
        pool -> {
          var worker = new WorkerThread(pool, name);
          workers.put(worker, "A non-null value, as required by Caffeine.");
          return worker;
        },
        /* handler= */ null,
        /* asyncMode= */ false);
  }

  private boolean acquireThreadAndCpuPermitElseReleaseCpuHeavyTask() {
    AvailableState snapshot;
    var target = new AvailableState();
    do {
      snapshot = available;
      boolean success = snapshot.tryAcquireThreadAndCpuPermitElseReleaseCpuHeavyTask(target);
      if (tryUpdateAvailable(snapshot, target)) {
        return success;
      }
    } while (true);
  }

  private boolean acquireThreadElseReleaseTask() {
    AvailableState snapshot;
    var target = new AvailableState();
    do {
      snapshot = available;
      boolean acquired = snapshot.tryAcquireThread(target);
      if (!acquired) {
        snapshot.releaseTask(target);
      }
      if (tryUpdateAvailable(snapshot, target)) {
        return acquired;
      }
    } while (true);
  }

  /**
   * {@link WorkerThread#runLoop} implements a small state machine.
   *
   * <p>After completing a task, the worker checks if there are any available tasks that it may
   * execute, subject to CPU permit constraints. On finding and reserving an appropriate task, the
   * worker returns its next planned activity, {@link #IDLE} or {@link #QUIESCENT} if it finds
   * nothing to do.
   */
  enum NextWorkerActivity {
    /** The worker will stop and is the last worker working. */
    QUIESCENT,
    /** The worker will stop. */
    IDLE,
    /** The worker will perform a non-CPU heavy task. */
    DO_TASK,
    /** The worker will perform a CPU heavy task. */
    DO_CPU_HEAVY_TASK
  }

  /**
   * Performs a task in a {@link WorkerThread} then loops.
   *
   * <p>Passed to {@link ForkJoinPool#execute} when a non-CPU-heavy task execution is needed.
   */
  // This could be a static method reference, but this makes it absolutely clear that no per-task
  // garbage is generated and results in nicer stack traces.
  private static final Runnable RUN_TASK = new LoopStarter(DO_TASK);

  /**
   * Performs a CPU heavy task in a {@link WorkerThread} then loops.
   *
   * <p>Passed to {@link ForkJoinPool#execute} when a CPU-heavy task execution is needed.
   */
  private static final Runnable RUN_CPU_HEAVY_TASK = new LoopStarter(DO_CPU_HEAVY_TASK);

  private static class LoopStarter implements Runnable {
    private final NextWorkerActivity activity;

    private LoopStarter(NextWorkerActivity activity) {
      this.activity = activity;
    }

    @Override
    public void run() {
      ((WorkerThread) currentThread()).runLoop(activity);
    }
  }

  class WorkerThread extends ForkJoinWorkerThread {

    private WorkerThread(ForkJoinPool pool, String name) {
      super(pool);
      setName(name + "-" + getPoolIndex());
    }

    /**
     * The worker runs a loop that scans for and runs available tasks.
     *
     * <p>This reduces costs associated with stopping and restarting threads.
     */
    private void runLoop(NextWorkerActivity nextActivity) {
      while (true) {
        switch (nextActivity) {
          case QUIESCENT:
            synchronized (quiescenceMonitor) {
              quiescenceMonitor.notifyAll();
            }
            return;
          case IDLE:
            return;
          case DO_TASK:
            dequeueTaskAndRun();
            nextActivity = getActivityFollowingTask();
            break;
          case DO_CPU_HEAVY_TASK:
            dequeueCpuHeavyTaskAndRun();
            nextActivity = getActivityFollowingCpuHeavyTask();
            break;
        }
      }
    }

    boolean tryDoQueuedWork() {
      AvailableState snapshot;
      var target = new AvailableState();

      do {
        snapshot = available;
        if (!snapshot.tryAcquireTask(target)) {
          return false;
        }
        if (tryUpdateAvailable(snapshot, target)) {
          dequeueTaskAndRun();
          return true;
        }
      } while (true);
    }
  }

  private void dequeueTaskAndRun() {
    try {
      queue.poll().run();
    } catch (Throwable uncaught) {
      handleUncaughtError(uncaught);
    }
  }

  private void dequeueCpuHeavyTaskAndRun() {
    try {
      cpuHeavyQueue.pollFirst().run();
    } catch (Throwable uncaught) {
      handleUncaughtError(uncaught);
    }
  }

  private NextWorkerActivity getActivityFollowingTask() {
    AvailableState snapshot;
    var target = new AvailableState();
    do {
      snapshot = available;
      if (snapshot.tryAcquireTask(target)) { // First, looks for a non-CPU heavy task.
        if (tryUpdateAvailable(snapshot, target)) {
          return DO_TASK;
        }
        // Next looks for a CPU heavy task and permit.
      } else if (snapshot.tryAcquireCpuHeavyTaskAndPermit(target)) {
        if (tryUpdateAvailable(snapshot, target)) {
          return DO_CPU_HEAVY_TASK;
        }
      } else { // Otherwise releases resources and completes.
        snapshot.releaseThread(target);
        if (tryUpdateAvailable(snapshot, target)) {
          return target.isQuiescent(poolSize) ? QUIESCENT : IDLE;
        }
      }
    } while (true);
  }

  private NextWorkerActivity getActivityFollowingCpuHeavyTask() {
    AvailableState snapshot;
    var target = new AvailableState();
    do {
      snapshot = available;
      // First, looks for a non-CPU heavy task.
      if (snapshot.tryAcquireTaskAndReleaseCpuPermit(target)) {
        if (tryUpdateAvailable(snapshot, target)) {
          return DO_TASK;
        }
      } else if (snapshot.tryAcquireCpuHeavyTask(target)) { // Next, looks for a CPU heavy task.
        if (tryUpdateAvailable(snapshot, target)) {
          return DO_CPU_HEAVY_TASK;
        }
      } else { // Otherwise releases resources and completes.
        snapshot.releaseThreadAndCpuPermit(target);
        if (tryUpdateAvailable(snapshot, target)) {
          return target.isQuiescent(poolSize) ? QUIESCENT : IDLE;
        }
      }
    } while (true);
  }

  /** Updates {@link #available} to {@code newValue} if it matches {@code expected}. */
  private boolean tryUpdateAvailable(AvailableState expected, AvailableState newValue) {
    // Profiling indicates that this is much faster than the equivalent VarHandle-based code.
    // TODO(blaze-core): replace with VarHandle if it becomes necessary.
    return UNSAFE.compareAndSwapObject(this, AVAILABLE_OFFSET, expected, newValue);
  }

  private static final Unsafe UNSAFE = UnsafeProvider.unsafe();

  /** Offset of the {@link #available} field. */
  private static final long AVAILABLE_OFFSET;

  static {
    try {
      AVAILABLE_OFFSET =
          UNSAFE.objectFieldOffset(PriorityWorkerPool.class.getDeclaredField("available"));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
