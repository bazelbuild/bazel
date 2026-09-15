// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent.safeexecutor;

import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Ticker;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;

/**
 * Wraps an asynchronous {@link ExecutorService} delegate and implements {@link SafeExecutor}.
 *
 * <p>Intentionally does NOT implement {@link java.util.concurrent.Executor} to prevent un-guarded
 * raw executor dispatches.
 *
 * <p><b>Teardown Strategy & Three Competing Priorities:</b>
 *
 * <ol>
 *   <li><b>Safety (Primary, Non-Negotiable):</b> Zero dropped dispatches or unhandled failure
 *       callbacks. Submission rejections and execution failures MUST always be delivered off-thread
 *       to {@code handleRejection(Throwable)} callbacks so that futures complete, locks release,
 *       and state machines never deadlock. We NEVER forcefully interrupt or drain {@code
 *       rejectionDispatcher} callbacks via {@code shutdownNow()}.
 *   <li><b>Responsiveness (Secondary):</b> Fast execution and bounded teardown within overall
 *       deadline budgets (e.g. 5-second termination window).
 *   <li><b>Isolation (Tertiary / Best-Effort):</b> Preventing thread, memory, or network resource
 *       leaks across Bazel command boundaries by waiting uninterruptibly for worker pool
 *       quiescence.
 * </ol>
 *
 * <p><b>The Trade-Off Rule:</b> When Responsiveness or Safety conflicts with Isolation, we trade
 * off a bit of resource isolation. Specifically, if {@code delegate} or {@code rejectionDispatcher}
 * fails to terminate within the timeout budget, {@link #awaitTermination(Duration)} logs a warning
 * and returns {@code false} early, allowing remaining tasks to drain naturally in the background
 * rather than force-killing them and risking deadlocks.
 */
public final class SafeExecutorOwner implements SafeExecutor {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ExecutorService delegate;
  private final ExecutorService rejectionDispatcher;
  private final Ticker ticker;

  public SafeExecutorOwner(ExecutorService delegate) {
    this(delegate, Executors.newVirtualThreadPerTaskExecutor(), Ticker.systemTicker());
  }

  @VisibleForTesting
  SafeExecutorOwner(ExecutorService delegate, Ticker ticker) {
    this(delegate, Executors.newVirtualThreadPerTaskExecutor(), ticker);
  }

  @VisibleForTesting
  SafeExecutorOwner(ExecutorService delegate, ExecutorService rejectionDispatcher, Ticker ticker) {
    this.delegate = Preconditions.checkNotNull(delegate, "delegate");
    this.rejectionDispatcher =
        Preconditions.checkNotNull(rejectionDispatcher, "rejectionDispatcher");
    this.ticker = Preconditions.checkNotNull(ticker, "ticker");
  }

  @Override
  public Executor getInternalUnsafeExecutor() {
    return delegate;
  }

  @Override
  public void execute(RejectionHandlingRunnable task) {
    Preconditions.checkNotNull(task, "task");
    try {
      delegate.execute(task);
    } catch (RejectedExecutionException e) {
      dispatchRejection(() -> task.handleRejection(e));
    }
  }

  @Override
  public <T> void addCallback(ListenableFuture<T> future, FutureCallback<? super T> callback) {
    Preconditions.checkNotNull(future, "future");
    Preconditions.checkNotNull(callback, "callback");

    var listener = new SafeCallbackListener<T>(future, callback, this);
    future.addListener(listener, listener);
  }

  /** Interrupts active worker threads and unwraps dropped tasks to notify rejections off-thread. */
  public void shutdownNow() {
    List<Runnable> droppedTasks = delegate.shutdownNow();
    var shutdownException =
        new RejectedExecutionException("Executor pool shut down via shutdownNow()");

    for (Runnable r : droppedTasks) {
      if (r instanceof RejectionHandlingRunnable rejectionTask) {
        dispatchRejection(() -> rejectionTask.handleRejection(shutdownException));
      } else if (r instanceof Future<?> futureTask) {
        futureTask.cancel(true);
      } else {
        dispatchRejection(
            () -> {
              // It's an otherwise unknown callback. Runs it on an interrupted thread.
              Thread.currentThread().interrupt();
              try {
                r.run();
              } catch (Throwable t) {
                logger.atWarning().atMostEvery(5, SECONDS).withCause(t).log(
                    "Fallback interrupted execution failed for non-RejectionHandlingRunnable"
                        + " dropped task %s",
                    r.getClass().getName());
              } finally {
                Thread.interrupted(); // Clear interrupt flag on rejectionDispatcher thread.
              }
            });
      }
    }
  }

  /**
   * Monotonic 3-Phase Teardown Protocol.
   *
   * <p><b>General Approach:</b>
   *
   * <p>Teardown allows the underlying {@code delegate} worker threads to drain fully before
   * shutting down {@code rejectionDispatcher}.
   *
   * <ul>
   *   <li><b>Phase 1:</b> Await worker thread termination on {@code delegate} uninterruptibly while
   *       keeping {@code rejectionDispatcher} OPEN so in-flight worker threads can emit
   *       failure/rejection callbacks. Even if the calling thread was interrupted (e.g., Ctrl-C),
   *       we give worker threads (which received {@code shutdownNow()} interrupts) up to the
   *       timeout budget to actually drain, guaranteeing resource isolation. If {@code delegate}
   *       fails to terminate, we return {@code false} early without shutting down {@code
   *       rejectionDispatcher}.
   *   <li><b>Phase 2:</b> Once {@code delegate} worker threads are 100% terminated, transition
   *       {@code rejectionDispatcher} to {@code SHUTDOWN} state.
   *   <li><b>Phase 3:</b> Await completion of all remaining queued rejection callbacks on {@code
   *       rejectionDispatcher} uninterruptibly using remaining budget. If {@code
   *       rejectionDispatcher} fails to terminate within the remaining budget, we log a warning and
   *       return {@code false} without calling {@code shutdownNow()} to avoid deadlocks.
   * </ul>
   *
   * <p><b>Deadline & Trade-Off Enforcement:</b>
   *
   * <p>Enforces a hard deadline bound across all teardown phases using {@link
   * Uninterruptibles#awaitTerminationUninterruptibly}. Adheres strictly to the Trade-Off Rule:
   * prioritizing Safety and Responsiveness by letting residual tasks drain in the background if the
   * timeout budget expires.
   */
  @CanIgnoreReturnValue
  public boolean awaitTermination(Duration timeout) {
    long startTimeNanos = ticker.read();

    // Phase 1: Await delegate worker quiescence uninterruptibly while rejectionDispatcher stays
    // OPEN.
    boolean delegateTerminated =
        Uninterruptibles.awaitTerminationUninterruptibly(delegate, timeout);
    if (!delegateTerminated) {
      logger.atWarning().log(
          "SafeExecutorOwner delegate executor failed to terminate within timeout budget");
      return false;
    }

    // Phase 2: Delegate has 100% terminated. Transition rejectionDispatcher gracefully to SHUTDOWN.
    rejectionDispatcher.shutdown();

    // Phase 3: Await rejectionDispatcher termination using remaining timeout budget.
    long elapsedNanos = ticker.read() - startTimeNanos;
    Duration remaining = timeout.minus(Duration.ofNanos(elapsedNanos));
    if (remaining.isNegative()) {
      remaining = Duration.ZERO;
    }

    boolean dispatcherTerminated =
        Uninterruptibles.awaitTerminationUninterruptibly(rejectionDispatcher, remaining);
    if (!dispatcherTerminated) {
      logger.atWarning().log(
          "SafeExecutorOwner rejectionDispatcher failed to terminate within timeout budget");
    }

    return dispatcherTerminated;
  }

  /** Returns true if this executor has been shut down. */
  @VisibleForTesting
  public boolean isShutdownForTesting() {
    return delegate.isShutdown();
  }

  /** Returns true if all tasks have completed following shut down. */
  @VisibleForTesting
  public boolean isTerminatedForTesting() {
    return delegate.isTerminated() && rejectionDispatcher.isTerminated();
  }

  /**
   * Offloads rejection notification tasks to {@code rejectionDispatcher}.
   *
   * <p><b>Fallback Execution & Thread Safety Mechanics:</b>
   *
   * <p>If {@code rejectionDispatcher.execute} throws an exception (e.g. because {@code
   * rejectionDispatcher} has shut down during Phase 2 of teardown or experienced a virtual thread
   * creation failure), the rejection notification is executed inline via {@code failureTask.run()}.
   *
   * <p><b>Teardown Assumption & Which threads land in this fallback?</b>
   *
   * <p><b>Lifecycle Assumption:</b> By the time {@code SafeExecutorOwner} teardown is invoked (e.g.
   * at the end of a build command hook), we assume there are no external threads (such as Skyframe
   * evaluation threads) actively submitting tasks. Furthermore, because {@code
   * rejectionDispatcher.shutdown()} is deferred until <i>after</i> {@code
   * delegate.awaitTermination()} completes, no active {@code delegate} worker threads exist when
   * {@code rejectionDispatcher} is shut down.
   *
   * <p>Under this assumption, the primary threads that enter this fallback during teardown are:
   *
   * <ul>
   *   <li><b>Rejection Virtual Threads (Cascaded Rejections):</b> Active virtual threads inside
   *       {@code rejectionDispatcher} executing a rejection handler that attempts a sub-rejection
   *       task dispatch after {@code rejectionDispatcher} has transitioned to {@code SHUTDOWN}.
   * </ul>
   *
   * <p><b>Trade-offs & Safety Rationale:</b>
   *
   * <p>This fallback is not 100% risk-free: executing inline carries a small risk of additional
   * <i>sequentialization</i> (cascaded rejection tasks execute serially on the current virtual
   * thread instead of spawning new virtual threads in parallel).
   *
   * <p>However, tasks that would have run asynchronously on the {@code rejectionDispatcher} simply
   * happen to be dispatched inline on a thread that is <i>already executing within the rejection
   * handling context</i> (or a stray submitting thread post-shutdown). This preserves domain
   * isolation from active {@code delegate} worker threads while guaranteeing that rejection
   * callbacks run to completion without being silently dropped.
   */
  private void dispatchRejection(Runnable failureTask) {
    try {
      rejectionDispatcher.execute(failureTask);
    } catch (Throwable t) {
      // If rejectionDispatcher is shut down or unavailable, run inline.
      try {
        failureTask.run();
      } catch (Throwable fatal) {
        // Since this is a double-fault scenario, requiring a rejected task, and a buggy client
        // implementation of handleRejection that throws an unchecked exception, simply logging here
        // is sufficient. It's difficult to inject a BugReport handle here due to how low level
        // this class is and the need for rejectionDispatcher to use virtual threads.
        logger.atSevere().withCause(fatal).log("Inline fallback rejection handler failed");
      }
    }
  }
}
