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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.concurrent.ErrorClassifier.ErrorClassification;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** A {@link QuiescingExecutor} implementation that wraps an {@link ExecutorService}. */
public class AbstractQueueVisitor implements QuiescingExecutor {
  /**
   * The most severe unhandled exception thrown by a worker thread, according to {@link
   * #errorClassifier}. This exception gets propagated to the calling thread of {@link
   * #awaitQuiescence} . We use the most severe error for the sake of not masking e.g. crashes in
   * worker threads after the first critical error that can occur due to race conditions in client
   * code.
   *
   * <p>Field updates happen only in blocks that are synchronized on the {@link
   * AbstractQueueVisitor} object.
   *
   * <p>If {@link AbstractQueueVisitor} clients don't like the semantics of storing and propagating
   * the most severe error, then they should be provide an {@link ErrorClassifier} that does the
   * right thing (e.g. to cause the _first_ error to be propagated, you'd want to provide an {@link
   * ErrorClassifier} that gives all errors the exact same {@link ErrorClassification}).
   *
   * <p>Note that this is not a performance-critical path.
   */
  private volatile Throwable unhandled = null;

  /**
   * An uncaught exception when submitting a job to the {@link ExecutorService} is catastrophic, and
   * usually indicates a lack of stack space on which to allocate a native thread. The {@link
   * ExecutorService} may reach an inconsistent state in such circumstances, so we avoid blocking on
   * its termination when this field is non-{@code null}.
   */
  private volatile Throwable catastrophe;

  /**
   * An object used in the manner of a {@link java.util.concurrent.locks.Condition} object, for the
   * condition {@code remainingTasks.get() == 0 || jobsMustBeStopped}. TODO(bazel-team): Replace
   * with an actual {@link java.util.concurrent.locks.Condition} object.
   */
  private final Object zeroRemainingTasks = new Object();

  /** The number of {@link Runnable}s {@link #execute}-d that have not finished evaluation. */
  private final AtomicLong remainingTasks = new AtomicLong(0);

  /**
   * A thread that wants to add or remove a future from {@code outstandingFutures} must first obtain
   * the <em>read</em> lock and then check the {@code threadInterrupted} flag. The thread that
   * cancels all futures must first set the {@code threadInterrupted} flag, and then obtain the
   * <em>write</em> lock. Only once both have happened is the main thread allowed to iterate over
   * the {@code outstandingFutures}. This allows concurrent future registration, but ensures that
   * canceling only happens when there are no concurrent modifications to the set since that
   * requires iterating over all elements.
   */
  private final ReadWriteLock outstandingFuturesLock = new ReentrantReadWriteLock();

  private final Set<ListenableFuture<?>> outstandingFutures = Sets.newConcurrentHashSet();

  /**
   * Flag used to record when all threads were killed by failed action execution. Only ever
   * transitions from {@code false} to {@code true}.
   *
   * <p>Except for {@link #mustJobsBeStopped}, may only be accessed in a block that is synchronized
   * on {@link #zeroRemainingTasks}.
   */
  private volatile boolean jobsMustBeStopped = false;

  /** Map from thread to number of jobs executing in the thread. Used for interrupt handling. */
  private final Map<Thread, AtomicLong> jobs = new ConcurrentHashMap<>();

  private final ExecutorService executorService;

  /**
   * Flag used to record when the main thread (the thread which called {@link #awaitQuiescence}) is
   * interrupted.
   *
   * <p>When this is {@code true}, adding tasks to the {@link ExecutorService} will fail quietly as
   * a part of the process of shutting down the worker threads.
   */
  private volatile boolean threadInterrupted;

  /**
   * Latches used to signal when the visitor has been interrupted or seen an exception. Used only
   * for testing.
   */
  private final CountDownLatch interruptedLatch = new CountDownLatch(1);

  private final CountDownLatch exceptionLatch = new CountDownLatch(1);

  /** If {@code true}, don't run new actions after an uncaught exception. */
  private final boolean failFastOnException;

  /** If {@code true}, shut down the {@link ExecutorService} on completion. */
  private final boolean ownExecutorService;

  private final ErrorClassifier errorClassifier;

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Default function for constructing {@link ThreadPoolExecutor}s. The {@link ThreadPoolExecutor}s
   * this creates have the same value for {@code corePoolSize} and {@code maximumPoolSize} because
   * that results in a fixed-size thread pool, and the current use cases for {@link
   * AbstractQueueVisitor} don't require any more sophisticated thread pool size management.
   *
   * <p>If client use cases change, they may invoke one of the {@link
   * AbstractQueueVisitor#AbstractQueueVisitor} constructors that accepts a pre-constructed {@link
   * ThreadPoolExecutor}.
   */
  private static ExecutorService createExecutorService(
      int parallelism,
      long keepAliveTime,
      TimeUnit units,
      BlockingQueue<Runnable> workQueue,
      String poolName) {

    if ("1".equals(System.getProperty("experimental_use_fork_join_pool"))) {
      return new NamedForkJoinPool(poolName, parallelism);
    }
    return new ThreadPoolExecutor(
        /*corePoolSize=*/ parallelism,
        /*maximumPoolSize=*/ parallelism,
        keepAliveTime,
        units,
        workQueue,
        new ThreadFactoryBuilder()
            .setNameFormat(Preconditions.checkNotNull(poolName) + " %d")
            .build());
  }

  public static ExecutorService createExecutorService(
      int parallelism, String poolName, boolean useForkJoinPool) {
    if (useForkJoinPool) {
      return new NamedForkJoinPool(poolName, parallelism);
    }
    return createExecutorService(parallelism, poolName);
  }

  public static ExecutorService createExecutorService(int parallelism, String poolName) {
    return createExecutorService(
        parallelism,
        /*keepAliveTime=*/ 1,
        TimeUnit.SECONDS,
        new PriorityBlockingQueue<>(),
        poolName);
  }

  public static AbstractQueueVisitor createWithExecutorService(
      ExecutorService executorService,
      boolean failFastOnException,
      ErrorClassifier errorClassifier) {
    if (executorService instanceof ForkJoinPool) {
      return ForkJoinQuiescingExecutor.newBuilder()
          .withOwnershipOf((ForkJoinPool) executorService)
          .setErrorClassifier(errorClassifier)
          .build();
    }
    return new AbstractQueueVisitor(executorService, true, failFastOnException, errorClassifier);
  }

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param parallelism a measure of parallelism for the {@link ExecutorService}, such as {@code
   *     parallelism} in {@link java.util.concurrent.ForkJoinPool}, or both {@code corePoolSize} and
   *     {@code maximumPoolSize} in {@link ThreadPoolExecutor}.
   * @param keepAliveTime the keep-alive time for the {@link ExecutorService}, if applicable.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param poolName sets the name of threads spawned by the {@link ExecutorService}. If {@code
   *     null}, default thread naming will be used.
   * @param errorClassifier an error classifier used to determine whether to log and/or stop jobs.
   */
  public AbstractQueueVisitor(
      int parallelism,
      long keepAliveTime,
      TimeUnit units,
      boolean failFastOnException,
      String poolName,
      ErrorClassifier errorClassifier) {
    this(
        createExecutorService(parallelism, keepAliveTime, units, new BlockingStack<>(), poolName),
        true,
        failFastOnException,
        errorClassifier);
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param executorService The {@link ExecutorService} to use.
   * @param shutdownOnCompletion If {@code true}, pass ownership of the {@link ExecutorService} to
   *     this class. The service will be shut down after a call to {@link #awaitQuiescence}. Callers
   *     must not shut down the {@link ExecutorService} while queue visitors use it.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param errorClassifier an error classifier used to determine whether to log and/or stop jobs.
   */
  protected AbstractQueueVisitor(
      ExecutorService executorService,
      boolean shutdownOnCompletion,
      boolean failFastOnException,
      ErrorClassifier errorClassifier) {
    this.failFastOnException = failFastOnException;
    this.ownExecutorService = shutdownOnCompletion;
    this.executorService = Preconditions.checkNotNull(executorService);
    this.errorClassifier = Preconditions.checkNotNull(errorClassifier);
  }

  @Override
  public final void awaitQuiescence(boolean interruptWorkers) throws InterruptedException {
    Throwables.propagateIfPossible(catastrophe);
    try {
      synchronized (zeroRemainingTasks) {
        while (remainingTasks.get() != 0 && !jobsMustBeStopped) {
          zeroRemainingTasks.wait();
        }
      }
    } catch (InterruptedException e) {
      // Mark the visitor, so that it's known to be interrupted, and
      // then break out of here, stop the worker threads and return ASAP,
      // sending the interruption to the parent thread.
      setInterrupted();
    }

    awaitTermination(interruptWorkers);
  }

  /** Schedules a call. Called in a worker thread. */
  @Override
  public final void execute(Runnable runnable) {
    executeWithExecutorService(runnable, executorService);
  }

  protected void executeWithExecutorService(Runnable runnable, ExecutorService executorService) {
    WrappedRunnable wrappedRunnable = new WrappedRunnable(runnable);
    try {
      // It's impossible for this increment to result in remainingTasks.get <= 0 because
      // remainingTasks is never negative. Therefore it isn't necessary to check its value for
      // the purpose of updating zeroRemainingTasks.
      long tasks = remainingTasks.incrementAndGet();
      Preconditions.checkState(
          tasks > 0,
          "Incrementing remaining tasks counter resulted in impossible non-positive number.");
      executeWrappedRunnable(wrappedRunnable, executorService);
    } catch (Throwable e) {
      if (!wrappedRunnable.ran) {
        // Note that keeping track of ranTask is necessary to disambiguate the case where
        // execute() itself failed, vs. a caller-runs policy on pool exhaustion, where the
        // runnable threw. To be extra cautious, we decrement the task count in a finally
        // block, even though the CountDownLatch is unlikely to throw.
        recordError(e);
      }
    }
  }

  protected void executeWrappedRunnable(WrappedRunnable runnable, ExecutorService executorService) {
    executorService.execute(runnable);
  }

  private synchronized void maybeSaveUnhandledThrowable(Throwable e, boolean markToStopJobs) {
    boolean critical = false;
    ErrorClassification errorClassification = errorClassifier.classify(e);
    switch (errorClassification) {
      case AS_CRITICAL_AS_POSSIBLE:
      case CRITICAL_AND_LOG:
        critical = true;
        logger.atWarning().withCause(e).log("Found critical error in queue visitor");
        break;
      case CRITICAL:
        critical = true;
        break;
      default:
        break;
    }
    if (unhandled == null
        || errorClassification.compareTo(errorClassifier.classify(unhandled)) > 0) {
      // Save the most severe error.
      unhandled = e;
      exceptionLatch.countDown();
    }
    if (markToStopJobs) {
      synchronized (zeroRemainingTasks) {
        if (critical && !jobsMustBeStopped) {
          jobsMustBeStopped = true;
          // This introduces a benign race, but it's the best we can do. When we have multiple
          // errors of the same severity that is at least CRITICAL, we'll end up saving (above) and
          // propagating (in 'awaitQuiescence') the most severe one we see, but the set of errors we
          // see is non-deterministic and is at the mercy of how quickly the calling thread of
          // 'awaitQuiescence' can do its thing after this 'notify' call.
          zeroRemainingTasks.notify();
        }
      }
    }
  }

  private void recordError(Throwable e) {
    try {
      // If threadInterrupted is true, then RejectedExecutionExceptions are expected. There's no
      // need to remember them, but there is a need to call decrementRemainingTasks, which is
      // satisfied by the finally block below.
      if (e instanceof RejectedExecutionException && threadInterrupted) {
        return;
      }
      catastrophe = e;
      maybeSaveUnhandledThrowable(e, /*markToStopJobs=*/ false);
    } finally {
      decrementRemainingTasks();
    }
  }

  /**
   * A wrapped {@link Runnable} that:
   *
   * <ul>
   *   <li>Sets {@link #run} to {@code true} when {@code WrappedRunnable} is run,
   *   <li>Records the thread evaluating {@code r} in {@link #jobs} while {@code r} is evaluated,
   *   <li>Prevents {@link #originalRunnable} from being invoked if {@link #blockNewActions} returns
   *       {@code true},
   *   <li>Synchronously invokes {@code runnable.run()},
   *   <li>Catches any {@link Throwable} thrown by {@code runnable.run()}, and if it is the most
   *       severe {@link Throwable} seen by this {@link AbstractQueueVisitor}, assigns it to {@link
   *       #unhandled}, and sets {@link #jobsMustBeStopped} if necessary,
   *   <li>And, lastly, calls {@link #decrementRemainingTasks}.
   * </ul>
   */
  protected final class WrappedRunnable implements Runnable, Comparable<WrappedRunnable> {
    private final Runnable originalRunnable;
    private volatile boolean ran;

    private WrappedRunnable(Runnable originalRunnable) {
      this.originalRunnable = originalRunnable;
    }

    @Override
    public void run() {
      ran = true;
      Thread thread = null;
      boolean addedJob = false;
      try {
        thread = Thread.currentThread();
        addJob(thread);
        addedJob = true;
        if (blockNewActions()) {
          // Make any newly enqueued tasks quickly die. We check after adding to the jobs map so
          // that if another thread is racing to kill this thread and didn't make it before this
          // conditional, it will be able to find and kill this thread anyway.
          return;
        }
        originalRunnable.run();
      } catch (Throwable e) {
        maybeSaveUnhandledThrowable(e, /*markToStopJobs=*/ true);
      } finally {
        try {
          if (thread != null && addedJob) {
            removeJob(thread);
          }
        } finally {
          decrementRemainingTasks();
        }
      }
    }

    @SuppressWarnings("unchecked")
    @Override
    public int compareTo(WrappedRunnable o) {
      // This should only be called when the concrete class is submitting comparable runnables.
      return ((Comparable) originalRunnable).compareTo(o.originalRunnable);
    }
  }

  private void addJob(Thread thread) {
    jobs.computeIfAbsent(thread, k -> new AtomicLong()).incrementAndGet();
  }

  private void removeJob(Thread thread) {
    if (jobs.get(thread).decrementAndGet() == 0) {
      jobs.remove(thread);
    }
  }

  /** Set an internal flag to show that an interrupt was detected. */
  protected final void setInterrupted() {
    threadInterrupted = true;
  }

  private void decrementRemainingTasks() {
    // This decrement statement may result in remainingTasks.get() == 0, so it must be checked
    // and the zeroRemainingTasks condition object notified if that condition is obtained.
    long tasks = remainingTasks.decrementAndGet();
    Preconditions.checkState(
        tasks >= 0, "Decrementing remaining tasks counter resulted in impossible negative number.");
    if (tasks == 0) {
      synchronized (zeroRemainingTasks) {
        zeroRemainingTasks.notify();
      }
    }
  }

  /** If this returns true, don't enqueue new actions. */
  protected boolean blockNewActions() {
    return isInterrupted() || (unhandled != null && failFastOnException);
  }

  @VisibleForTesting
  @Override
  public final CountDownLatch getExceptionLatchForTestingOnly() {
    return exceptionLatch;
  }

  @VisibleForTesting
  @Override
  public final CountDownLatch getInterruptionLatchForTestingOnly() {
    return interruptedLatch;
  }

  /** Get the value of the interrupted flag. */
  @ThreadSafety.ThreadSafe
  protected final boolean isInterrupted() {
    return threadInterrupted;
  }

  /**
   * Get number of jobs remaining. Note that this can increase in value if running tasks submit
   * further jobs.
   */
  public final long getTaskCount() {
    return remainingTasks.get();
  }

  protected ExecutorService getExecutorService() {
    return executorService;
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) throws InterruptedException {
    outstandingFuturesLock.readLock().lock();
    try {
      if (threadInterrupted) {
        future.cancel(/*mayInterruptIfRunning=*/ true);
        throw new InterruptedException();
      }
      remainingTasks.incrementAndGet();
      outstandingFutures.add(future);
      future.addListener(() -> markFutureDone(future), MoreExecutors.directExecutor());
    } finally {
      outstandingFuturesLock.readLock().unlock();
    }
  }

  private void markFutureDone(ListenableFuture<?> future) {
    decrementRemainingTasks();
    outstandingFuturesLock.readLock().lock();
    try {
      if (threadInterrupted) {
        // Since futures get worked on asynchronously, there is an inherent race between this method
        // being called and the future getting canceled. If we get here, the other thread either
        // already attempted to cancel the future, or is just about to do so. In either case, no
        // need to throw or do anything with outstandingFutures here.
        return;
      }
      outstandingFutures.remove(future);
    } finally {
      outstandingFuturesLock.readLock().unlock();
    }
  }

  /**
   * Whether all running and pending jobs will be stopped or cancelled. Also newly submitted tasks
   * will be rejected if this is true.
   *
   * <p>This function returns the CURRENT state of whether jobs should be stopped. If the value is
   * false right now, it may be changed to true by another thread later.
   */
  protected final boolean mustJobsBeStopped() {
    return jobsMustBeStopped;
  }

  /**
   * Waits for the task queue to drain. Then if {@code ownExecutorService} is true, shuts down the
   * {@link ExecutorService} and waits for it to terminate. Throws (the same) unchecked exception if
   * any worker thread failed unexpectedly.
   */
  protected final void awaitTermination(boolean interruptWorkers) throws InterruptedException {
    reallyAwaitTermination(interruptWorkers);

    if (isInterrupted()) {
      // Set interrupted bit on current thread so that callers can see that it was interrupted. Note
      // that if the thread was interrupted while awaiting termination, we might not hit this
      // code path, but then the current thread's interrupt bit is already set, so we are fine.
      Thread.currentThread().interrupt();
    }
    // Throw the first unhandled (worker thread) exception in the main thread. We throw an unchecked
    // exception instead of InterruptedException if both are present because an unchecked exception
    // may indicate a catastrophic failure that should shut down the program. The caller can
    // check the interrupted bit if they will handle the unchecked exception without crashing.
    Throwables.propagateIfPossible(unhandled);

    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
  }

  private void reallyAwaitTermination(boolean interruptWorkers) {
    // TODO(bazel-team): verify that interrupt() is safe for every use of
    // AbstractQueueVisitor and remove the interruptWorkers flag.
    if (interruptWorkers && !jobs.isEmpty()) {
      interruptInFlightTasks();
    }
    if (interruptWorkers) {
      // If the computation is done, this does nothing because there are no outstanding futures. Do
      // not predicate on outstandingFutures.isEmpty() here: in the case of an interrupt, there may
      // still be threads concurrently adding futures to the set, and we need to make sure that
      // those are canceled correctly.
      cancelAllFutures();
    }

    if (isInterrupted()) {
      interruptedLatch.countDown();
    }

    Throwables.propagateIfPossible(catastrophe);
    synchronized (zeroRemainingTasks) {
      while (remainingTasks.get() != 0) {
        try {
          zeroRemainingTasks.wait();
        } catch (InterruptedException e) {
          setInterrupted();
        }
      }
    }

    if (ownExecutorService) {
      shutdownExecutorService(catastrophe);
    }
  }

  protected void shutdownExecutorService(Throwable catastrophe) {
      executorService.shutdown();
      for (; ; ) {
        try {
          Throwables.propagateIfPossible(catastrophe);
          executorService.awaitTermination(Integer.MAX_VALUE, TimeUnit.SECONDS);
          break;
        } catch (InterruptedException e) {
          setInterrupted();
        }
      }
  }

  private void interruptInFlightTasks() {
    Thread thisThread = Thread.currentThread();
    for (Thread thread : jobs.keySet()) {
      if (thisThread != thread) {
        thread.interrupt();
      }
    }
  }

  private void cancelAllFutures() {
    // Nobody else can modify outstandingFutures while this thread is holding the write lock.
    outstandingFuturesLock.writeLock().lock();
    try {
      for (ListenableFuture<?> future : outstandingFutures) {
        future.cancel(/*mayInterruptIfRunning=*/ true);
      }
      outstandingFutures.clear();
    } finally {
      outstandingFuturesLock.writeLock().unlock();
    }
  }
}
