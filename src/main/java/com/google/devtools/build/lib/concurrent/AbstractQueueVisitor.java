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
import com.google.common.base.Function;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.AtomicLongMap;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/** A {@link QuiescingExecutor} implementation that wraps an {@link ExecutorService}. */
public class AbstractQueueVisitor implements QuiescingExecutor {

  /**
   * Default factory function for constructing {@link ThreadPoolExecutor}s. The {@link
   * ThreadPoolExecutor}s this creates have the same value for {@code corePoolSize} and {@code
   * maximumPoolSize} because that results in a fixed-size thread pool, and the current use cases
   * for {@link AbstractQueueVisitor} don't require any more sophisticated thread pool size
   * management.
   *
   * <p>If client use cases change, they may invoke one of the {@link
   * AbstractQueueVisitor#AbstractQueueVisitor} constructors that accepts a pre-constructed {@link
   * ThreadPoolExecutor}.
   */
  public static final Function<ExecutorParams, ThreadPoolExecutor> EXECUTOR_FACTORY =
      new Function<ExecutorParams, ThreadPoolExecutor>() {
        @Override
        public ThreadPoolExecutor apply(ExecutorParams p) {
          return new ThreadPoolExecutor(
              /*corePoolSize=*/ p.getParallelism(),
              /*maximumPoolSize=*/ p.getParallelism(),
              p.getKeepAliveTime(),
              p.getUnits(),
              p.getWorkQueue(),
              new ThreadFactoryBuilder().setNameFormat(p.getPoolName() + " %d").build());
        }
      };

  /**
   * The first unhandled exception thrown by a worker thread.  We save it and re-throw it from
   * the main thread to detect bugs faster; otherwise worker threads just quietly die.
   *
   * Field updates happen only in blocks that are synchronized on the {@link
   * AbstractQueueVisitor} object; it's important to save the first one as it may be more
   * informative than a subsequent one, and this is not a performance-critical path.
   */
  private volatile Throwable unhandled = null;

  /**
   * An uncaught exception when submitting a job to the {@link ExecutorService} is catastrophic,
   * and usually indicates a lack of stack space on which to allocate a native thread. The {@link
   * ExecutorService} may reach an inconsistent state in such circumstances, so we avoid blocking
   * on its termination when this field is non-{@code null}.
   */
  private volatile Throwable catastrophe;

  /**
   * Enables concurrency. For debugging or testing, set this to {@code false} to avoid thread
   * creation and concurrency. Any deviation in observed behaviour is a bug.
   */
  private final boolean concurrent;

  /**
   * An object used in the manner of a {@link java.util.concurrent.locks.Condition} object, for the
   * condition {@code remainingTasks.get() == 0 || jobsMustBeStopped}.
   * TODO(bazel-team): Replace with an actual {@link java.util.concurrent.locks.Condition} object.
   */
  private final Object zeroRemainingTasks = new Object();

  /**
   * If {@link #concurrent} is {@code true}, then this is a counter of the number of {@link
   * Runnable}s {@link #execute}-d that have not finished evaluation.
   */
  private final AtomicLong remainingTasks = new AtomicLong(0);

  /**
   * Flag used to record when all threads were killed by failed action execution. Only ever
   * transitions from {@code false} to {@code true}.
   *
   * <p>May only be accessed in a block that is synchronized on {@link #zeroRemainingTasks}.
   */
  private boolean jobsMustBeStopped = false;

  /** Map from thread to number of jobs executing in the thread. Used for interrupt handling. */
  private final AtomicLongMap<Thread> jobs = AtomicLongMap.create();

  /** The {@link ExecutorService}. If !{@code concurrent}, this may be {@code null}. */
  @Nullable private final ExecutorService executorService;

  /**
   * Flag used to record when the main thread (the thread which called {@link #awaitQuiescence})
   * is interrupted.
   *
   * When this is {@code true}, adding tasks to the {@link ExecutorService} will fail quietly as
   * a part of the process of shutting down the worker threads.
   */
  private volatile boolean threadInterrupted = false;

  /**
   * Latches used to signal when the visitor has been interrupted or seen an exception. Used only
   * for testing.
   */
  private final CountDownLatch interruptedLatch = new CountDownLatch(1);
  private final CountDownLatch exceptionLatch = new CountDownLatch(1);

  /** If {@code true}, don't run new actions after an uncaught exception. */
  private final boolean failFastOnException;

  /** If {@code true}, don't run new actions after an interrupt. */
  private final boolean failFastOnInterrupt;

  /** If {@code true}, shut down the {@link ExecutorService} on completion. */
  private final boolean ownExecutorService;

  private final ErrorClassifier errorClassifier;

  private static final Logger LOG = Logger.getLogger(AbstractQueueVisitor.class.getName());

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param concurrent {@code true} if concurrency should be enabled. Only set to {@code false} for
   *                   debugging.
   * @param parallelism a measure of parallelism for the {@link ExecutorService}, such as {@code
   *                    parallelism} in {@link java.util.concurrent.ForkJoinPool}, or both {@code
   *                    corePoolSize} and {@code maximumPoolSize} in {@link ThreadPoolExecutor}.
   * @param keepAliveTime the keep-alive time for the {@link ExecutorService}, if applicable.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if {@code true}, don't run new actions after an interrupt.
   * @param poolName sets the name of threads spawned by the {@link ExecutorService}. If {@code
   *                 null}, default thread naming will be used.
   */
  public AbstractQueueVisitor(
      boolean concurrent,
      int parallelism,
      long keepAliveTime,
      TimeUnit units,
      boolean failFastOnException,
      boolean failFastOnInterrupt,
      String poolName) {
    this(
        concurrent,
        parallelism,
        keepAliveTime,
        units,
        failFastOnException,
        failFastOnInterrupt,
        poolName,
        EXECUTOR_FACTORY,
        ErrorClassifier.DEFAULT);
  }

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param concurrent {@code true} if concurrency should be enabled. Only set to {@code false} for
   *                   debugging.
   * @param parallelism a measure of parallelism for the {@link ExecutorService}, such as {@code
   *                    parallelism} in {@link java.util.concurrent.ForkJoinPool}, or both {@code
   *                    corePoolSize} and {@code maximumPoolSize} in {@link ThreadPoolExecutor}.
   * @param keepAliveTime the keep-alive time for the {@link ExecutorService}, if applicable.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if {@code true}, don't run new actions after an interrupt.
   * @param poolName sets the name of threads spawned by the {@link ExecutorService}. If {@code
   *                 null}, default thread naming will be used.
   * @param errorClassifier an error classifier used to determine whether to log and/or stop jobs.
   */
  public AbstractQueueVisitor(
      boolean concurrent,
      int parallelism,
      long keepAliveTime,
      TimeUnit units,
      boolean failFastOnException,
      boolean failFastOnInterrupt,
      String poolName,
      ErrorClassifier errorClassifier) {
    this(
        concurrent,
        parallelism,
        keepAliveTime,
        units,
        failFastOnException,
        failFastOnInterrupt,
        poolName,
        EXECUTOR_FACTORY,
        errorClassifier);
  }

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param concurrent {@code true} if concurrency should be enabled. Only set to {@code false} for
   *                   debugging.
   * @param parallelism a measure of parallelism for the {@link ExecutorService}, such as {@code
   *                    parallelism} in {@link java.util.concurrent.ForkJoinPool}, or both {@code
   *                    corePoolSize} and {@code maximumPoolSize} in {@link ThreadPoolExecutor}.
   * @param keepAliveTime the keep-alive time for the {@link ExecutorService}, if applicable.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if {@code true}, don't run new actions after interrupt.
   * @param poolName sets the name of threads spawned by the {@link ExecutorService}. If {@code
   *                 null}, default thread naming will be used.
   * @param executorFactory the factory for constructing the executor service if {@code concurrent}
   *                        is {@code true}.
   */
  public AbstractQueueVisitor(
      boolean concurrent,
      int parallelism,
      long keepAliveTime,
      TimeUnit units,
      boolean failFastOnException,
      boolean failFastOnInterrupt,
      String poolName,
      Function<ExecutorParams, ? extends ExecutorService> executorFactory,
      ErrorClassifier errorClassifier) {
    Preconditions.checkNotNull(poolName);
    Preconditions.checkNotNull(executorFactory);
    Preconditions.checkNotNull(errorClassifier);
    this.concurrent = concurrent;
    this.failFastOnException = failFastOnException;
    this.failFastOnInterrupt = failFastOnInterrupt;
    this.ownExecutorService = true;
    this.executorService =
        concurrent
            ? executorFactory.apply(
                new ExecutorParams(
                    parallelism, keepAliveTime, units, poolName, new BlockingStack<Runnable>()))
            : null;
    this.errorClassifier = errorClassifier;
  }

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param concurrent {@code true} if concurrency should be enabled. Only set to {@code false}
   *                   for debugging.
   * @param parallelism a measure of parallelism for the {@link ExecutorService}, such as {@code
   *                    parallelism} in {@link java.util.concurrent.ForkJoinPool}, or both {@code
   *                    corePoolSize} and {@code maximumPoolSize} in {@link ThreadPoolExecutor}.
   * @param keepAliveTime the keep-alive time for the {@link ExecutorService}, if applicable.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param poolName sets the name of threads spawned by the {@link ExecutorService}. If {@code
   *                 null}, default thread naming will be used.
   */
  public AbstractQueueVisitor(
      boolean concurrent,
      int parallelism,
      long keepAliveTime,
      TimeUnit units,
      boolean failFastOnException,
      String poolName) {
    this(
        concurrent,
        parallelism,
        keepAliveTime,
        units,
        failFastOnException,
        true,
        poolName,
        EXECUTOR_FACTORY,
        ErrorClassifier.DEFAULT);
  }

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param executorService The {@link ExecutorService} to use.
   * @param shutdownOnCompletion If {@code true}, pass ownership of the {@link ExecutorService} to
   *                             this class. The service will be shut down after a
   *                             call to {@link #awaitQuiescence}. Callers must not shutdown the
   *                             {@link ExecutorService} while queue visitors use it.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if {@code true}, don't run new actions after an interrupt.
   */
  public AbstractQueueVisitor(
      ExecutorService executorService,
      boolean shutdownOnCompletion,
      boolean failFastOnException,
      boolean failFastOnInterrupt) {
    this(
        /*concurrent=*/ true,
        executorService,
        shutdownOnCompletion,
        failFastOnException,
        failFastOnInterrupt,
        ErrorClassifier.DEFAULT);
  }

  /**
   * Create the {@link AbstractQueueVisitor}.
   *
   * @param concurrent if {@code false}, run tasks inline instead of using the {@link
   *                   ExecutorService}.
   * @param executorService The {@link ExecutorService} to use.
   * @param shutdownOnCompletion If {@code true}, pass ownership of the {@link ExecutorService} to
   *                             this class. The service will be shut down after a
   *                             call to {@link #awaitQuiescence}. Callers must not shut down the
   *                             {@link ExecutorService} while queue visitors use it.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if {@code true}, don't run new actions after an interrupt.
   */
  public AbstractQueueVisitor(
      boolean concurrent,
      ExecutorService executorService,
      boolean shutdownOnCompletion,
      boolean failFastOnException,
      boolean failFastOnInterrupt) {
    Preconditions.checkArgument(executorService != null || !concurrent);
    this.concurrent = concurrent;
    this.failFastOnException = failFastOnException;
    this.failFastOnInterrupt = failFastOnInterrupt;
    this.ownExecutorService = shutdownOnCompletion;
    this.executorService = executorService;
    this.errorClassifier = ErrorClassifier.DEFAULT;
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param concurrent if {@code false}, run tasks inline instead of using the {@link
   *                   ExecutorService}.
   * @param executorService The {@link ExecutorService} to use.
   * @param shutdownOnCompletion If {@code true}, pass ownership of the {@link ExecutorService} to
   *                             this class. The service will be shut down after a
   *                             call to {@link #awaitQuiescence}. Callers must not shut down the
   *                             {@link ExecutorService} while queue visitors use it.
   * @param failFastOnException if {@code true}, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if {@code true}, don't run new actions after an interrupt.
   * @param errorClassifier an error classifier used to determine whether to log and/or stop jobs.
   */
  public AbstractQueueVisitor(
      boolean concurrent,
      ExecutorService executorService,
      boolean shutdownOnCompletion,
      boolean failFastOnException,
      boolean failFastOnInterrupt,
      ErrorClassifier errorClassifier) {
    Preconditions.checkArgument(executorService != null || !concurrent);
    this.concurrent = concurrent;
    this.failFastOnException = failFastOnException;
    this.failFastOnInterrupt = failFastOnInterrupt;
    this.ownExecutorService = shutdownOnCompletion;
    this.executorService = executorService;
    this.errorClassifier = errorClassifier;
  }

  /**
   * Create the {@code AbstractQueueVisitor} with concurrency enabled.
   *
   * @param parallelism a measure of parallelism for the {@link ExecutorService}, such as {@code
   *                    parallelism} in {@link java.util.concurrent.ForkJoinPool}, or both {@code
   *                    corePoolSize} and {@code maximumPoolSize} in {@link ThreadPoolExecutor}.
   * @param keepAlive the keep-alive time for the {@link ExecutorService}, if applicable.
   * @param units the time units of keepAliveTime.
   * @param poolName sets the name of threads spawned by the {@link ExecutorService}. If {@code
   *                 null}, default thread naming will be used.
   */
  public AbstractQueueVisitor(int parallelism, long keepAlive, TimeUnit units, String poolName) {
    this(
        true,
        parallelism,
        keepAlive,
        units,
        false,
        true,
        poolName,
        EXECUTOR_FACTORY,
        ErrorClassifier.DEFAULT);
  }


  @Override
  public final void awaitQuiescence(boolean interruptWorkers) throws InterruptedException {
    if (concurrent) {
      awaitTermination(interruptWorkers);
    } else {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException();
      }
    }
  }

  /** Schedules a call. Called in a worker thread if concurrent. */
  @Override
  public final void execute(Runnable runnable) {
    if (runConcurrently()) {
      WrappedRunnable wrappedRunnable = new WrappedRunnable(runnable);
      try {
        // It's impossible for this increment to result in remainingTasks.get <= 0 because
        // remainingTasks is never negative. Therefore it isn't necessary to check its value for
        // the purpose of updating zeroRemainingTasks.
        long tasks = remainingTasks.incrementAndGet();
        Preconditions.checkState(
            tasks > 0,
            "Incrementing remaining tasks counter resulted in impossible non-positive number.");
        executeRunnable(wrappedRunnable);
      } catch (Throwable e) {
        if (!wrappedRunnable.ran) {
          // Note that keeping track of ranTask is necessary to disambiguate the case where
          // execute() itself failed, vs. a caller-runs policy on pool exhaustion, where the
          // runnable threw. To be extra cautious, we decrement the task count in a finally
          // block, even though the CountDownLatch is unlikely to throw.
          recordError(e);
        }
      }
    } else {
      runnable.run();
    }
  }

  /**
   * Subclasses may override this to make dynamic decisiouns about whether to run tasks
   * asynchronously versus in-thread.
   */
  protected boolean runConcurrently() {
    return concurrent;
  }

  /**
   * Returns an approximate count of how many threads in the queue visitor's thread pool are
   * occupied with tasks.
   */
  protected final int activeParallelTasks() {
    return jobs.asMap().size();
  }

  protected void executeRunnable(Runnable runnable) {
    executorService.execute(runnable);
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
      synchronized (this) {
        if (unhandled == null) { // save only the first one.
          unhandled = e;
          exceptionLatch.countDown();
        }
      }
    } finally {
      decrementRemainingTasks();
    }
  }

  /**
   * A wrapped {@link Runnable} that:
   * <ul>
   *   <li>Sets {@link #run} to {@code true} when {@code WrappedRunnable} is run,
   *   <li>Records the thread evaluating {@code r} in {@link #jobs} while {@code r} is evaluated,
   *   <li>Prevents {@param runnable} from being invoked if {@link #blockNewActions} returns
   *   {@code true},
   *   <li>Synchronously invokes {@code runnable.run()},
   *   <li>Catches any {@link Throwable} thrown by {@code runnable.run()}, and if it is the first
   *   {@link Throwable} seen by this {@link AbstractQueueVisitor}, assigns it to {@link
   *   #unhandled}, and calls {@link #markToStopAllJobsIfNeeded} to set {@link #jobsMustBeStopped}
   *   if necessary,
   *   <li>And, lastly, calls {@link #decrementRemainingTasks}.
   * </ul>
   */
  private final class WrappedRunnable implements Runnable {
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
        synchronized (AbstractQueueVisitor.this) {
          if (unhandled == null) { // save only the first one.
            unhandled = e;
            exceptionLatch.countDown();
          }
          markToStopAllJobsIfNeeded(e);
        }
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
  }

  private void addJob(Thread thread) {
    jobs.incrementAndGet(thread);
  }

  private void removeJob(Thread thread) {
    if (jobs.decrementAndGet(thread) == 0) {
      jobs.remove(thread);
    }
  }

  /** Set an internal flag to show that an interrupt was detected. */
  private void setInterrupted() {
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
    return (failFastOnInterrupt && isInterrupted()) || (unhandled != null && failFastOnException);
  }

  @VisibleForTesting
  public final CountDownLatch getExceptionLatchForTestingOnly() {
    return exceptionLatch;
  }

  @VisibleForTesting
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
  @VisibleForTesting
  protected final long getTaskCount() {
    return remainingTasks.get();
  }

  /**
   * Waits for the task queue to drain, then shuts down the {@link ExecutorService} and
   * waits for it to terminate.  Throws (the same) unchecked exception if any
   * worker thread failed unexpectedly.
   */
  private void awaitTermination(boolean interruptWorkers) throws InterruptedException {
    Preconditions.checkState(failFastOnInterrupt || !interruptWorkers);
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

    reallyAwaitTermination(interruptWorkers);

    if (isInterrupted()) {
      // Set interrupted bit on current thread so that callers can see that it was interrupted. Note
      // that if the thread was interrupted while awaiting termination, we might not hit this
      // codepath, but then the current thread's interrupt bit is already set, so we are fine.
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
      executorService.shutdown();
      for (;;) {
        try {
          Throwables.propagateIfPossible(catastrophe);
          executorService.awaitTermination(Integer.MAX_VALUE, TimeUnit.SECONDS);
          break;
        } catch (InterruptedException e) {
          setInterrupted();
        }
      }
    }
  }

  private void interruptInFlightTasks() {
    Thread thisThread = Thread.currentThread();
    for (Thread thread : jobs.asMap().keySet()) {
      if (thisThread != thread) {
        thread.interrupt();
      }
    }
  }

  /**
   * Classifies a {@link Throwable} {@param e} thrown by a job.
   *
   * <p>If it is classified as critical, then this sets the {@link #jobsMustBeStopped} flag to
   * {@code true} which signals {@link #awaitTermination(boolean)} to stop all jobs.
   *
   * <p>Also logs details about {@param e} if it is classified as something that must be logged.
   */
  private void markToStopAllJobsIfNeeded(Throwable e) {
    boolean critical = false;
    switch (errorClassifier.classify(e)) {
        case CRITICAL_AND_LOG:
          critical = true;
          LOG.log(Level.WARNING, "Found critical error in queue visitor", e);
          break;
        case CRITICAL:
          critical = true;
          break;
        default:
          break;
    }
    synchronized (zeroRemainingTasks) {
      if (critical && !jobsMustBeStopped) {
        jobsMustBeStopped = true;
        zeroRemainingTasks.notify();
      }
    }
  }
}
