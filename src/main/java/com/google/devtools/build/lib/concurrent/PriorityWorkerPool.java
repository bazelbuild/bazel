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
import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.concurrent.PaddedAddresses.createPaddedBaseAddress;
import static com.google.devtools.build.lib.concurrent.PaddedAddresses.getAlignedAddress;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.DO_CPU_HEAVY_TASK;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.DO_TASK;
import static com.google.devtools.build.lib.concurrent.PriorityWorkerPool.NextWorkerActivity.IDLE;
import static java.lang.Thread.currentThread;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import java.lang.ref.Cleaner;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.TreeMap;
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

  /** The configured size of the thread pool. */
  private final int poolSize;

  /** The number of CPU permits configured. */
  private final int cpuPermits;

  private ForkJoinPool pool;

  /**
   * Queue for non-CPU heavy tasks.
   *
   * <p>An interesting alternative to consider is to place unprioritized tasks directly into {@link
   * #pool}, which could reduce the work performed by the system. Doing this results in about a
   * {@code 4%} end-to-end regression in our benchmark. The likely cause for this is that FIFO
   * behavior is very important for performance because it reflects the ordering of prioritized
   * tasks.
   */
  private final TaskFifo queue;

  private final ConcurrentSkipListSet<ComparableRunnable> cpuHeavyQueue =
      new ConcurrentSkipListSet<>();

  private final String name;

  /**
   * Cache of workers for interrupt handling.
   *
   * <p>A {@link Cache} allows us to use weak keys, which are the only relevant objects here. The
   * values are an unintentional side effect of the library and populated arbitrarily.
   */
  private final Cache<WorkerThread, Object> workers = Caffeine.newBuilder().weakKeys().build();

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

  private final ErrorClassifier errorClassifier;

  /**
   * The most severe unhandled exception thrown by a worker thread, according to {@link
   * #errorClassifier}. This exception gets propagated to the calling thread of {@link
   * TieredPriorityExecutor#awaitQuiescence}. We use the most severe error for the sake of not
   * masking e.g. crashes in worker threads after the first critical error that can occur due to
   * race conditions in client code.
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

  PriorityWorkerPool(
      Cleaner cleaner, String name, int poolSize, int cpuPermits, ErrorClassifier errorClassifier) {
    checkArgument(poolSize <= (THREADS_MASK >> THREADS_BIT_OFFSET), poolSize);
    checkArgument(cpuPermits <= (CPU_PERMITS_MASK >> CPU_PERMITS_BIT_OFFSET), cpuPermits);

    this.name = name;
    this.poolSize = poolSize;
    this.cpuPermits = cpuPermits;

    this.pool = newForkJoinPool();
    this.errorClassifier = errorClassifier;

    long baseAddress = createPaddedBaseAddress(5);
    cleaner.register(this, new AddressFreer(baseAddress));
    this.countersAddress = getAlignedAddress(baseAddress, /* offset= */ 0);

    this.queue =
        new TaskFifo(
            /* sizeAddress= */ getAlignedAddress(baseAddress, /* offset= */ 1),
            /* appendIndexAddress= */ getAlignedAddress(baseAddress, /* offset= */ 2),
            /* takeIndexAddress= */ getAlignedAddress(baseAddress, /* offset= */ 3));

    this.activeWorkerCountAddress = getAlignedAddress(baseAddress, /* offset= */ 4);

    resetExecutionCounters();
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
          UNSAFE.getAndAddInt(null, activeWorkerCountAddress, 1);
          pool.execute(RUN_CPU_HEAVY_TASK);
        }
        return;
      }
    }

    if (!queue.tryAppend(rawTask)) {
      if (!isCancelled()) {
        // The task queue is full (and the pool is not cancelled). Enqueues the task directly in the
        // ForkJoinPool. This should be rare in practice.
        UNSAFE.getAndAddInt(null, activeWorkerCountAddress, 1);
        pool.execute(
            () -> {
              try {
                rawTask.run();
              } catch (Throwable uncaught) {
                handleUncaughtError(uncaught);
              } finally {
                workerBecomingIdle();
              }
            });
      }
      return;
    }

    if (acquireThreadElseReleaseTask()) {
      UNSAFE.getAndAddInt(null, activeWorkerCountAddress, 1);
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
    markCancelled();

    workers.asMap().keySet().forEach(Thread::interrupt);
  }

  /** Shuts down a pool and frees all resources. */
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

    resetExecutionCounters();
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
        .add("available", formatSnapshot(getExecutionCounters()))
        .add("|queue|", queue.size())
        .add("|cpu queue|", cpuHeavyQueue.size())
        .add("threads", threadStates)
        .add("unhandled", unhandled.get())
        .add("pool", pool)
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

  /**
   * {@link WorkerThread#runLoop} implements a small state machine.
   *
   * <p>After completing a task, the worker checks if there are any available tasks that it may
   * execute, subject to CPU permit constraints. On finding and reserving an appropriate task, the
   * worker returns its next planned activity, {@link #IDLE} if it finds nothing to do.
   */
  enum NextWorkerActivity {
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
          case IDLE:
            workerBecomingIdle();
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
  }

  private void dequeueTaskAndRun() {
    try {
      var task = queue.take();
      task.run();
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

  private void workerBecomingIdle() {
    if (UNSAFE.getAndAddInt(null, activeWorkerCountAddress, -1) == 1) {
      synchronized (quiescenceMonitor) {
        quiescenceMonitor.notifyAll();
      }
    }
  }

  // The constants below apply to the 64-bit execution counters value.

  private static final long CANCEL_BIT = 0x8000_0000_0000_0000L;

  private static final long CPU_PERMITS_MASK = 0x7FF0_0000_0000_0000L;
  private static final int CPU_PERMITS_BIT_OFFSET = 52;
  private static final long ONE_CPU_PERMIT = 1L << CPU_PERMITS_BIT_OFFSET;

  private static final long THREADS_MASK = 0x000F_FFE0_0000_0000L;
  private static final int THREADS_BIT_OFFSET = 37;
  private static final long ONE_THREAD = 1L << THREADS_BIT_OFFSET;

  private static final long TASKS_MASK = 0x0000_001F_FF80_0000L;
  private static final int TASKS_BIT_OFFSET = 23;
  private static final long ONE_TASK = 1L << TASKS_BIT_OFFSET;
  static final int TASKS_MAX_VALUE = (int) (TASKS_MASK >> TASKS_BIT_OFFSET);

  private static final long CPU_HEAVY_TASKS_MASK = 0x0000_0000_07F_FFFFL;
  private static final int CPU_HEAVY_TASKS_BIT_OFFSET = 0;
  private static final long ONE_CPU_HEAVY_TASK = 1L << CPU_HEAVY_TASKS_BIT_OFFSET;

  private static final long CPU_HEAVY_RESOURCES = ONE_CPU_PERMIT + ONE_THREAD;

  static {
    checkState(
        ONE_CPU_PERMIT == (CPU_PERMITS_MASK & -CPU_PERMITS_MASK),
        "Inconsistent CPU Permits Constants");
    checkState(ONE_THREAD == (THREADS_MASK & -THREADS_MASK), "Inconistent Threads Constants");
    checkState(
        ONE_CPU_HEAVY_TASK == (CPU_HEAVY_TASKS_MASK & -CPU_HEAVY_TASKS_MASK),
        "Inconsistent CPU Heavy Task Constants");
  }

  /**
   * Address of the execution counters value, consisting of 5 fields packed into a 64-bit long.
   *
   * <ol>
   *   <li>Canceled - (1 bit) true for cancelled.
   *   <li>CPU Permits - (11 bits) how many CPU heavy permits are available.
   *   <li>Threads - (15 bits) how many threads are available.
   *   <li>Tasks - (14 bits) how many non-CPU heavy tasks are inflight.
   *   <li>CPU Heavy Tasks - (23 bits) how many CPU heavy tasks are inflight.
   * </ol>
   *
   * <p>Convenience constants for field access and manipulation are above.
   */
  private final long countersAddress;

  private final long activeWorkerCountAddress;

  boolean isQuiescent() {
    return UNSAFE.getInt(null, activeWorkerCountAddress) == 0;
  }

  boolean isCancelled() {
    return getExecutionCounters() < 0;
  }

  private void markCancelled() {
    long snapshot;
    do {
      snapshot = getExecutionCounters();
      if (snapshot < 0) {
        return; // Already cancelled.
      }
    } while (!tryUpdateExecutionCounters(snapshot, snapshot | CANCEL_BIT));
  }

  private void resetExecutionCounters() {
    UNSAFE.putLong(
        null,
        countersAddress,
        (((long) poolSize) << THREADS_BIT_OFFSET)
            | (((long) cpuPermits) << CPU_PERMITS_BIT_OFFSET));
    UNSAFE.putInt(null, activeWorkerCountAddress, 0);
  }

  private boolean acquireThreadElseReleaseTask() {
    long snapshot;
    do {
      snapshot = UNSAFE.getLongVolatile(null, countersAddress);
      boolean acquired = (snapshot & THREADS_MASK) > 0 && snapshot >= 0;
      long target = snapshot + (acquired ? -ONE_THREAD : ONE_TASK);
      if (UNSAFE.compareAndSwapLong(null, countersAddress, snapshot, target)) {
        return acquired;
      }
    } while (true);
  }

  private boolean acquireThreadAndCpuPermitElseReleaseCpuHeavyTask() {
    long snapshot;
    do {
      snapshot = UNSAFE.getLongVolatile(null, countersAddress);
      boolean acquired =
          (snapshot & (CANCEL_BIT | CPU_PERMITS_MASK)) > 0 && (snapshot & THREADS_MASK) > 0;
      long target = snapshot + (acquired ? -(ONE_THREAD + ONE_CPU_PERMIT) : ONE_CPU_HEAVY_TASK);
      if (UNSAFE.compareAndSwapLong(null, countersAddress, snapshot, target)) {
        return acquired;
      }
    } while (true);
  }

  /**
   * Worker threads determine their next action after completing a task using this method.
   *
   * <p>This acquires a CPU permit when returning {@link NextWorkerActivity#DO_CPU_HEAVY_TASK}.
   */
  private NextWorkerActivity getActivityFollowingTask() {
    long snapshot = UNSAFE.getLongVolatile(null, countersAddress);
    do {
      if ((snapshot & (CANCEL_BIT | TASKS_MASK)) > 0) {
        if (UNSAFE.compareAndSwapLong(null, countersAddress, snapshot, snapshot - ONE_TASK)) {
          return DO_TASK;
        }
      } else if ((snapshot & (CANCEL_BIT | CPU_HEAVY_TASKS_MASK)) > 0
          && (snapshot & CPU_PERMITS_MASK) != 0) {
        if (UNSAFE.compareAndSwapLong(
            null, countersAddress, snapshot, snapshot - (ONE_CPU_HEAVY_TASK + ONE_CPU_PERMIT))) {
          return DO_CPU_HEAVY_TASK;
        }
      } else {
        long target = snapshot + ONE_THREAD;
        if (UNSAFE.compareAndSwapLong(null, countersAddress, snapshot, target)) {
          return IDLE;
        }
      }
      snapshot = UNSAFE.getLong(null, countersAddress);
    } while (true);
  }

  /**
   * Worker threads call this to determine their next action after completing a CPU heavy task.
   *
   * <p>This releases a CPU permit when returning {@link NextWorkerActivity#IDLE} or {@link
   * NextWorkerActivity#DO_TASK}.
   */
  private NextWorkerActivity getActivityFollowingCpuHeavyTask() {
    long snapshot = UNSAFE.getLongVolatile(null, countersAddress);
    do {
      if ((snapshot & (CANCEL_BIT | TASKS_MASK)) > 0) {
        if (UNSAFE.compareAndSwapLong(
            null, countersAddress, snapshot, snapshot + (ONE_CPU_PERMIT - ONE_TASK))) {
          return DO_TASK;
        }
      } else if ((snapshot & (CANCEL_BIT | CPU_HEAVY_TASKS_MASK)) > 0) {
        if (UNSAFE.compareAndSwapLong(
            null, countersAddress, snapshot, snapshot - ONE_CPU_HEAVY_TASK)) {
          return DO_CPU_HEAVY_TASK;
        }
      } else {
        long target = snapshot + CPU_HEAVY_RESOURCES;
        if (UNSAFE.compareAndSwapLong(null, countersAddress, snapshot, target)) {
          return IDLE;
        }
      }
      snapshot = UNSAFE.getLong(null, countersAddress);
    } while (true);
  }

  // Throughout this class, the following wrappers are used where possible, but they are often not
  // inlined by the JVM even though they show up on profiles, so they are inlined explicitly in
  // numerous cases.

  private long getExecutionCounters() {
    return UNSAFE.getLongVolatile(null, countersAddress);
  }

  private boolean tryUpdateExecutionCounters(long snapshot, long target) {
    return UNSAFE.compareAndSwapLong(null, countersAddress, snapshot, target);
  }

  private static String formatSnapshot(long snapshot) {
    return String.format(
        "{cancelled=%b, threads=%d, cpuPermits=%d, tasks=%d, cpuHeavyTasks=%d}",
        snapshot < 0,
        (snapshot & THREADS_MASK) >> THREADS_BIT_OFFSET,
        (snapshot & CPU_PERMITS_MASK) >> CPU_PERMITS_BIT_OFFSET,
        (snapshot & TASKS_MASK) >> TASKS_BIT_OFFSET,
        (snapshot & CPU_HEAVY_TASKS_MASK) >> CPU_HEAVY_TASKS_BIT_OFFSET);
  }

  private static final Unsafe UNSAFE = UnsafeProvider.unsafe();
}
