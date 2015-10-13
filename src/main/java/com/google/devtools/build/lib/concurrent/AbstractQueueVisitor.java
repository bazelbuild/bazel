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
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ThreadFactoryBuilder;

import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * AbstractQueueVisitor is a wrapper around {@link ThreadPoolExecutor} which
 * delays thread pool shutdown until entire visitation is complete.
 * This is useful for cases in which worker tasks may submit additional tasks.
 *
 * <p>Consider the following example:
 * <pre>
 *   ThreadPoolExecutor executor = <...>
 *   executor.submit(myRunnableTask);
 *   executor.shutdown();
 *   executor.awaitTermination();
 * </pre>
 *
 * <p>This won't work properly if {@code myRunnableTask} submits additional
 * tasks to the executor, because it may already have shut down
 * by that point.
 *
 * <p>AbstractQueueVisitor supports interruption. If the main thread is
 * interrupted, tasks will no longer be added to the queue, and the
 * {@link #work(boolean)} method will throw {@link InterruptedException}.
 */
public class AbstractQueueVisitor {

  /**
   * Default factory function for constructing {@link ThreadPoolExecutor}s.
   */
  public static final Function<ThreadPoolExecutorParams, ThreadPoolExecutor> EXECUTOR_FACTORY =
      new Function<ThreadPoolExecutorParams, ThreadPoolExecutor>() {
        @Override
        public ThreadPoolExecutor apply(ThreadPoolExecutorParams p) {
          return new ThreadPoolExecutor(p.getCorePoolSize(), p.getMaxPoolSize(),
              p.getKeepAliveTime(), p.getUnits(), p.getWorkQueue(),
              new ThreadFactoryBuilder().setNameFormat(p.getPoolName() + " %d").build());
        }
      };

  /**
   * The first unhandled exception thrown by a worker thread.  We save it
   * and re-throw it from the main thread to detect bugs faster;
   * otherwise worker threads just quietly die.
   *
   * Field updates are synchronized; it's
   * important to save the first one as it may be more informative than a
   * subsequent one, and this is not a performance-critical path.
   */
  private volatile Throwable unhandled = null;

  /**
   * An uncaught exception when submitting a job to the ThreadPool is catastrophic, and usually
   * indicates a lack of stack space on which to allocate a native thread. The JDK
   * ThreadPoolExecutor may reach an inconsistent state in such circumstances, so we avoid blocking
   * on its termination when this field is non-null.
   */
  private volatile Throwable catastrophe;

  /**
   * Enables concurrency.  For debugging or testing, set this to false
   * to avoid thread creation and concurrency. Any deviation in observed
   * behaviour is a bug.
   */
  private final boolean concurrent;

  // Condition variable for remainingTasks==0, and a lock for it.
  private final Object zeroRemainingTasks = new Object();
  private long remainingTasks = 0;

  // Map of thread ==> number of jobs executing in the thread.
  // Currently used only for interrupt handling.
  private final Map<Thread, Long> jobs = Maps.newConcurrentMap();

  /**
   * The thread pool. If !concurrent, always null. Created lazily on first
   * call to {@link #enqueue(Runnable)}, and removed after call to
   * {@link #work(boolean)}.
   */
  private final ThreadPoolExecutor pool;

  /**
   * Flag used to record when the main thread (the thread which called
   * {@link #work(boolean)}) is interrupted.
   *
   * When this is true, adding tasks to the thread pool will
   * fail quietly as a part of the process of shutting down the
   * worker threads.
   */
  private volatile boolean threadInterrupted = false;

  /**
   * Latches used to signal when the visitor has been interrupted or
   * seen an exception. Used only for testing.
   */
  private final CountDownLatch interruptedLatch = new CountDownLatch(1);
  private final CountDownLatch exceptionLatch = new CountDownLatch(1);

  /**
   * If true, don't run new actions after an uncaught exception.
   */
  private final boolean failFastOnException;

  /**
   * If true, don't run new actions after an interrupt.
   */
  private final boolean failFastOnInterrupt;

  /**
   * If true, we must shut down the thread pool on completion.
   */
  private final boolean ownThreadPool;

  /**
   * Flag used to record when all threads were killed by failed action execution.
   *
   * <p>May only be accessed in a synchronized block.
   */
  private boolean jobsMustBeStopped = false;

  private static final Logger LOG = Logger.getLogger(AbstractQueueVisitor.class.getName());

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param concurrent true if concurrency should be enabled. Only set to
   *                   false for debugging.
   * @param corePoolSize the core pool size of the thread pool. See
   *                     {@link ThreadPoolExecutor#ThreadPoolExecutor(int, int, long, TimeUnit,
   *                     BlockingQueue)}
   * @param maxPoolSize the max number of threads in the pool.
   * @param keepAliveTime the keep-alive time for the thread pool.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if true, don't run new actions after
   *                            an uncaught exception.
   * @param failFastOnInterrupt if true, don't run new actions after interrupt.
   * @param poolName sets the name of threads spawn by this thread pool. If {@code null}, default
   *                    thread naming will be used.
   */
  public AbstractQueueVisitor(boolean concurrent, int corePoolSize, int maxPoolSize,
      long keepAliveTime, TimeUnit units, boolean failFastOnException,
      boolean failFastOnInterrupt, String poolName) {
    this(concurrent, corePoolSize, maxPoolSize, keepAliveTime, units, failFastOnException,
        failFastOnInterrupt, poolName, EXECUTOR_FACTORY);
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param concurrent true if concurrency should be enabled. Only set to
   *                   false for debugging.
   * @param corePoolSize the core pool size of the thread pool. See
   *                     {@link ThreadPoolExecutor#ThreadPoolExecutor(int, int, long, TimeUnit,
   *                     BlockingQueue)}
   * @param maxPoolSize the max number of threads in the pool.
   * @param keepAliveTime the keep-alive time for the thread pool.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if true, don't run new actions after an uncaught exception.
   * @param failFastOnInterrupt if true, don't run new actions after interrupt.
   * @param poolName sets the name of threads spawn by this thread pool. If {@code null}, default
   *                    thread naming will be used.
   * @param executorFactory the factory for constructing the thread pool if {@code concurrent} is
   *                        true.
   */
  public AbstractQueueVisitor(boolean concurrent, int corePoolSize, int maxPoolSize,
      long keepAliveTime, TimeUnit units, boolean failFastOnException,
      boolean failFastOnInterrupt, String poolName,
      Function<ThreadPoolExecutorParams, ThreadPoolExecutor> executorFactory) {
    Preconditions.checkNotNull(poolName);
    Preconditions.checkNotNull(executorFactory);
    this.concurrent = concurrent;
    this.failFastOnException = failFastOnException;
    this.failFastOnInterrupt = failFastOnInterrupt;
    this.ownThreadPool = true;
    this.pool = concurrent
      ? executorFactory.apply(new ThreadPoolExecutorParams(corePoolSize, maxPoolSize,
        keepAliveTime, units, poolName, getWorkQueue()))
      : null;
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param concurrent true if concurrency should be enabled. Only set to
   *                   false for debugging.
   * @param corePoolSize the core pool size of the thread pool. See
   *                     {@link ThreadPoolExecutor#ThreadPoolExecutor(int, int, long, TimeUnit,
   *                     BlockingQueue)}
   * @param maxPoolSize the max number of threads in the pool.
   * @param keepAliveTime the keep-alive time for the thread pool.
   * @param units the time units of keepAliveTime.
   * @param failFastOnException if true, don't run new actions after
   *                            an uncaught exception.
   * @param poolName sets the name of threads spawn by this thread pool. If {@code null}, default
   *                    thread naming will be used.
   */
  public AbstractQueueVisitor(boolean concurrent, int corePoolSize, int maxPoolSize,
      long keepAliveTime, TimeUnit units, boolean failFastOnException, String poolName) {
    this(concurrent, corePoolSize, maxPoolSize, keepAliveTime, units, failFastOnException, true,
        poolName);
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param executor The ThreadPool to use.
   * @param shutdownOnCompletion If true, pass ownership of the Threadpool to
   *                             this class. The pool will be shut down after a
   *                             call to work(). Callers must not shutdown the
   *                             threadpool while queue visitors use it.
   * @param failFastOnException if true, don't run new actions after
   *                            an uncaught exception.
   * @param failFastOnInterrupt if true, don't run new actions after interrupt.
   */
  public AbstractQueueVisitor(ThreadPoolExecutor executor, boolean shutdownOnCompletion,
                              boolean failFastOnException, boolean failFastOnInterrupt) {
    this(/*concurrent=*/true, executor, shutdownOnCompletion, failFastOnException,
        failFastOnInterrupt);
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param concurrent if false, run tasks inline instead of using the thread pool.
   * @param executor The ThreadPool to use.
   * @param shutdownOnCompletion If true, pass ownership of the Threadpool to
   *                             this class. The pool will be shut down after a
   *                             call to work(). Callers must not shut down the
   *                             threadpool while queue visitors use it.
   * @param failFastOnException if true, don't run new actions after
   *                            an uncaught exception.
   * @param failFastOnInterrupt if true, don't run new actions after interrupt.
   */
  public AbstractQueueVisitor(boolean concurrent, ThreadPoolExecutor executor,
                              boolean shutdownOnCompletion, boolean failFastOnException,
                              boolean failFastOnInterrupt) {
    this.concurrent = concurrent;
    this.failFastOnException = failFastOnException;
    this.failFastOnInterrupt = failFastOnInterrupt;
    this.pool = executor;
    this.ownThreadPool = shutdownOnCompletion;
  }

  public AbstractQueueVisitor(ThreadPoolExecutor executor, boolean failFastOnException) {
    this(executor, true, failFastOnException, true);
  }

  /**
   * Create the AbstractQueueVisitor.
   *
   * @param concurrent true if concurrency should be enabled. Only set to
   *                   false for debugging.
   * @param corePoolSize the core pool size of the thread pool. See
   *                     {@link ThreadPoolExecutor#ThreadPoolExecutor(int, int, long, TimeUnit,
   *                     BlockingQueue)}
   * @param maxPoolSize the max number of threads in the pool.
   * @param keepAliveTime the keep-alive time for the thread pool.
   * @param units the time units of keepAliveTime.
   * @param poolName sets the name of threads spawn by this thread pool. If {@code null}, default
   *                    thread naming will be used.
   */
  public AbstractQueueVisitor(boolean concurrent, int corePoolSize, int maxPoolSize,
      long keepAliveTime, TimeUnit units, String poolName) {
    this(concurrent, corePoolSize, maxPoolSize, keepAliveTime, units, false, poolName);
  }

  /**
   * Create the AbstractQueueVisitor with concurrency enabled.
   *
   * @param corePoolSize the core pool size of the thread pool. See
   *                     {@link ThreadPoolExecutor#ThreadPoolExecutor(int, int, long, TimeUnit,
   *                     BlockingQueue)}
   * @param maxPoolSize the max number of threads in the pool.
   * @param keepAlive the keep-alive time for the thread pool.
   * @param units the time units of keepAliveTime.
   * @param poolName sets the name of threads spawn by this thread pool. If {@code null}, default
   *                    thread naming will be used.
   */
  public AbstractQueueVisitor(int corePoolSize, int maxPoolSize, long keepAlive, TimeUnit units,
      String poolName) {
    this(true, corePoolSize, maxPoolSize, keepAlive, units, poolName);
  }

  protected BlockingQueue<Runnable> getWorkQueue() {
    return new LinkedBlockingQueue<>();
  }

  /**
   * Executes all tasks on the queue, and optionally shuts the pool down and deletes it.
   *
   * <p>Throws (the same) unchecked exception if any worker thread failed unexpectedly. If the pool
   * is interrupted and a worker also throws an unchecked exception, the unchecked exception is
   * rethrown, since it may indicate a programming bug. If callers handle the unchecked exception,
   * they may check the interrupted bit to see if the pool was interrupted.
   *
   * @param interruptWorkers if true, interrupt worker threads if main thread gets an interrupt or
   *        if a worker throws a critical error (see {@link #isCriticalError(Throwable)}). If
   *        false, just wait for them to terminate normally.
   */
  protected void work(boolean interruptWorkers) throws InterruptedException {
    if (concurrent) {
      awaitTermination(interruptWorkers);
    } else {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException();
      }
    }
  }

  /**
   * Schedules a call.
   * Called in a worker thread if concurrent.
   */
  protected void enqueue(Runnable runnable) {
    if (concurrent) {
      AtomicBoolean ranTask = new AtomicBoolean(false);
      try {
        pool.execute(wrapRunnable(runnable, ranTask));
      } catch (Throwable e) {
        if (!ranTask.get()) {
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

  private void recordError(Throwable e) {
    catastrophe = e;
    try {
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

  private Runnable wrapRunnable(final Runnable runnable, final AtomicBoolean ranTask) {
    synchronized (zeroRemainingTasks) {
      remainingTasks++;
    }
    return new Runnable() {
      @Override
      public void run() {
        Thread thread = null;
        boolean addedJob = false;
        try {
          ranTask.set(true);
          thread = Thread.currentThread();
          addJob(thread);
          addedJob = true;
          if (blockNewActions()) {
            // Make any newly enqueued tasks quickly die. We check after adding to the jobs map so
            // that if another thread is racing to kill this thread and didn't make it before this
            // conditional, it will be able to find and kill this thread anyway.
            return;
          }
          runnable.run();
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
    };
  }

  private final void addJob(Thread thread) {
    // Note: this looks like a check-then-act race but it isn't, because each
    // key implies thread-locality.
    long count = jobs.containsKey(thread) ? jobs.get(thread) + 1 : 1;
    jobs.put(thread, count);
  }

  private final void removeJob(Thread thread) {
    Long boxedCount = Preconditions.checkNotNull(jobs.get(thread),
        "Can't retrieve job after successfully adding it");
    long count = boxedCount - 1;
    if (count == 0) {
      jobs.remove(thread);
    } else {
      jobs.put(thread, count);
    }
  }

  /**
   * Set an internal flag to show that an interrupt was detected.
   */
  private void setInterrupted() {
    threadInterrupted = true;
    setRejectedExecutionHandler();
  }

  private final void decrementRemainingTasks() {
    synchronized (zeroRemainingTasks) {
      if (--remainingTasks == 0) {
        zeroRemainingTasks.notify();
      }
    }
  }

  /**
   * If this returns true, don't enqueue new actions.
   */
  protected boolean blockNewActions() {
    return (failFastOnInterrupt && isInterrupted()) || (unhandled != null && failFastOnException);
  }

  /**
   * Await interruption.  Used only in tests.
   */
  @VisibleForTesting
  public boolean awaitInterruptionForTestingOnly(long timeout, TimeUnit units)
      throws InterruptedException {
    return interruptedLatch.await(timeout, units);
  }

  /** Get latch that is released when exception is received by visitor. Used only in tests. */
  @VisibleForTesting
  public CountDownLatch getExceptionLatchForTestingOnly() {
    return exceptionLatch;
  }

  /** Get latch that is released when interruption is received by visitor. Used only in tests. */
  @VisibleForTesting
  public CountDownLatch getInterruptionLatchForTestingOnly() {
    return interruptedLatch;
  }

  /**
   * Get the value of the interrupted flag.
   */
  @ThreadSafety.ThreadSafe
  protected boolean isInterrupted() {
    return threadInterrupted;
  }

  /**
   * Get number of jobs remaining. Note that this can increase in value
   * if running tasks submit further jobs.
   */
  @VisibleForTesting
  protected long getTaskCount() {
    synchronized (zeroRemainingTasks) {
      return remainingTasks;
    }
  }

  /**
   * Waits for the task queue to drain, then shuts down the thread pool and
   * waits for it to terminate.  Throws (the same) unchecked exception if any
   * worker thread failed unexpectedly.
   */
  private void awaitTermination(boolean interruptWorkers) throws InterruptedException {
    Preconditions.checkState(failFastOnInterrupt || !interruptWorkers);
    Throwables.propagateIfPossible(catastrophe);
    try {
      synchronized (zeroRemainingTasks) {
        while (remainingTasks != 0 && !jobsMustBeStopped) {
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
      while (remainingTasks != 0) {
        try {
          zeroRemainingTasks.wait();
        } catch (InterruptedException e) {
          setInterrupted();
        }
      }
    }

    if (ownThreadPool) {
      pool.shutdown();
      for (;;) {
        try {
          Throwables.propagateIfPossible(catastrophe);
          pool.awaitTermination(Integer.MAX_VALUE, TimeUnit.SECONDS);
          break;
        } catch (InterruptedException e) {
          setInterrupted();
        }
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

  /**
   * If this returns true, that means the exception {@code e} is critical
   * and all running actions should be stopped. {@link Error}s are always considered critical.
   *
   * <p>Default value - always false. If different behavior is needed
   * then we should override this method in subclasses.
   *
   * @param e the exception object to check
   */
  protected boolean isCriticalError(Throwable e) {
    return false;
  }

  private boolean isCriticalErrorInternal(Throwable e) {
    boolean isCritical = isCriticalError(e) || (e instanceof Error);
    if (isCritical) {
      LOG.log(Level.WARNING, "Found critical error in queue visitor", e);
    }
    return isCritical;
  }

  private void setRejectedExecutionHandler() {
    if (ownThreadPool) {
      pool.setRejectedExecutionHandler(new RejectedExecutionHandler() {
        @Override
        public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
          decrementRemainingTasks();
        }
      });
    }
  }

  /**
   * If exception is critical then set a flag which signals
   * to stop all jobs inside {@link #awaitTermination(boolean)}.
   */
  private synchronized void markToStopAllJobsIfNeeded(Throwable e) {
    if (isCriticalErrorInternal(e) && !jobsMustBeStopped) {
      jobsMustBeStopped = true;
      synchronized (zeroRemainingTasks) {
        zeroRemainingTasks.notify();
      }
    }
  }
}
