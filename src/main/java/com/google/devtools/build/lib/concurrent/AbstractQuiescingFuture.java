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
package com.google.devtools.build.lib.concurrent;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.internal.InternalFutures;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.errorprone.annotations.ForOverride;
import com.google.errorprone.annotations.Keep;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.RejectedExecutionException;
import javax.annotation.Nullable;

/**
 * A base class for futures that track in-flight tasks and complete when the tasks quiesce or an
 * error occurs.
 */
public abstract class AbstractQuiescingFuture<T> extends AbstractFuture<T>
    implements RejectionHandlingRunnable {
  /**
   * Handle for {@link #taskCount}.
   *
   * <p>This uses less memory than {@link java.util.concurrent.atomic.AtomicInteger}.
   */
  private static final VarHandle TASK_COUNT_HANDLE;

  private static final VarHandle SECONDARY_EXCEPTIONS_HANDLE;

  private final SafeExecutor getValueExecutor;

  /**
   * Count of in-flight tasks.
   *
   * <p>This is initialized to 1 to support the "pre-increment" pattern, which prevents premature
   * completion during initialization.
   *
   * <p>Use {@link #TASK_COUNT_HANDLE} for atomic operations.
   */
  @Keep // used via TASK_COUNT_HANDLE
  private volatile int taskCount;

  /**
   * The first exception observed by this future.
   *
   * <p>{@code null} if no exceptions occurred or if the future was cancelled.
   */
  @Nullable private volatile Throwable primaryException;

  /** Any additional exceptions observed after the first. */
  @Keep // used via SECONDARY_EXCEPTIONS_HANDLE
  @Nullable
  private volatile ConcurrentLinkedQueue<Throwable> secondaryExceptions;

  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   * @param taskCount initial task count.
   */
  protected AbstractQuiescingFuture(SafeExecutor getValueExecutor, int taskCount) {
    this.getValueExecutor = Preconditions.checkNotNull(getValueExecutor, "getValueExecutor");
    this.taskCount = taskCount;
  }

  /**
   * A unit of subtask work executed via {@link #executeSubtask}.
   *
   * <p>This interface is defined as a distinct functional interface rather than reusing {@link
   * Runnable} to prevent ambiguous subtyping bugs with {@link RejectionHandlingRunnable}. Because
   * {@link RejectionHandlingRunnable} extends {@link Runnable}, accepting a plain {@link Runnable}
   * would allow callers to pass an existing {@link RejectionHandlingRunnable} instance (such as a
   * child {@link QuiescingFutureTask}), under the mistaken assumption that its {@link
   * RejectionHandlingRunnable#handleRejection} method would be invoked upon rejection. By using an
   * independent {@link Subtask} interface, the compiler prevents accidental passing of {@link
   * RejectionHandlingRunnable}.
   */
  @FunctionalInterface
  public interface Subtask {
    /** Executes the subtask. */
    void runSubtask();
  }

  /**
   * Submits a {@link Subtask} to the given {@link SafeExecutor}, incrementing {@link #taskCount}
   * and ensuring proper decrements and exception recording upon execution, exception, or rejection.
   *
   * @param subtask the subtask work to execute
   * @param executor the runner for executing the subtask
   */
  public final void executeSubtask(Subtask subtask, SafeExecutor executor) {
    Preconditions.checkNotNull(subtask, "subtask");
    Preconditions.checkNotNull(executor, "executor");
    increment();
    try {
      executor.execute(new SubtaskRejectionHandlingRunnable(subtask));
    } catch (Throwable t) {
      notifyException(t);
    }
  }

  /**
   * The resulting value of this future.
   *
   * <p>Called after the final decrement. Implementations must guarantee that the value is ready at
   * that time. Not called if there were any errors.
   */
  @ForOverride
  protected abstract T getValue();

  /**
   * Called if there was an error, after all the associated tasks complete.
   *
   * <p>Allows clients to perform cleanup work if there is an error. Receives the primary cause (or
   * null if cancelled without an explicit exception) and any secondary causes recorded during
   * execution.
   */
  @ForOverride
  protected void doneWithError(
      @Nullable Throwable primaryCause, ImmutableList<Throwable> secondaryCauses) {}

  /**
   * Sets the future as failing with {@code t} and decrements the task count.
   *
   * <p>If the client calls this, it should not call {@link #decrement} for the same task.
   */
  protected final void notifyException(Throwable t) {
    Preconditions.checkNotNull(t, "t");
    recordException(t);
    decrement();
  }

  /**
   * Records an exception on the future.
   *
   * <p>If this is the first exception recorded, sets the future's failure cause. Subsequent
   * exceptions are collected into {@link #secondaryExceptions} to preserve multi-failure
   * diagnostics for {@link #doneWithError(Throwable, ImmutableList)}.
   */
  protected final void recordException(Throwable t) {
    Preconditions.checkNotNull(t, "t");
    if (t instanceof CancellationException) {
      // Propagates CancellationException by cancelling this future.
      cancel(/* mayInterruptIfRunning= */ false);
    } else if (setException(t)) {
      primaryException = t;
    } else {
      getOrCreateSecondaryExceptions().add(t);
    }
  }

  /** Increments the task count. */
  protected final void increment() {
    TASK_COUNT_HANDLE.getAndAdd(this, 1);
  }

  /** Decrements the task count. */
  protected final void decrement() {
    int countBeforeDecrement = (int) TASK_COUNT_HANDLE.getAndAdd(this, -1);
    if (countBeforeDecrement == 1) {
      try {
        getValueExecutor.execute(this);
      } catch (RejectedExecutionException e) {
        handleRejection(e);
      }
    }
  }

  /**
   * Decrements the task count without submitting to {@link #getValueExecutor}.
   *
   * <p>For use by {@link QuiescingFutureTask#handleRejection} when a setup task is rejected before
   * running to clear the pre-increment.
   */
  final void decrementWithoutScheduling() {
    TASK_COUNT_HANDLE.getAndAdd(this, -1);
  }

  /**
   * Completes this future upon quiescence (or runs error cleanup if cancelled or failed).
   *
   * <p>Package-private to allow dispatch from {@link QuiescingFuture#run}, {@link
   * QuiescingFutureTask#run}, and {@link QuiescingFutureTask#handleRejection}.
   */
  final void handleQuiescence() {
    Throwable primaryCause = getPrimaryCause();
    if (isCancelled() || primaryCause != null) {
      runDoneWithError(primaryCause);
      return;
    }

    boolean setSuccess = false;
    try {
      T value = getValue();
      setSuccess = set(value);
    } catch (Throwable t) {
      recordException(t);
    } finally {
      if (!setSuccess) {
        // This may occur when cancellation races with set, above, or if getValue() throws an
        // exception.
        runDoneWithError(getPrimaryCause());
      }
    }
  }

  @SuppressWarnings("unchecked") // required for VarHandle usage
  private ConcurrentLinkedQueue<Throwable> getOrCreateSecondaryExceptions() {
    var queue = (ConcurrentLinkedQueue<Throwable>) SECONDARY_EXCEPTIONS_HANDLE.getAcquire(this);
    if (queue == null) {
      queue = new ConcurrentLinkedQueue<>();
      if (!SECONDARY_EXCEPTIONS_HANDLE.compareAndSet(this, null, queue)) {
        queue = (ConcurrentLinkedQueue<Throwable>) SECONDARY_EXCEPTIONS_HANDLE.getAcquire(this);
      }
    }
    return queue;
  }

  private final class SubtaskRejectionHandlingRunnable implements RejectionHandlingRunnable {
    private final Subtask subtask;

    SubtaskRejectionHandlingRunnable(Subtask subtask) {
      this.subtask = subtask;
    }

    @Override
    public void run() {
      try {
        subtask.runSubtask();
      } catch (Throwable t) {
        recordException(t);
      } finally {
        decrement();
      }
    }

    @Override
    public void handleRejection(Throwable t) {
      notifyException(t);
    }
  }

  @Nullable
  private Throwable getPrimaryCause() {
    if (primaryException != null) {
      return primaryException;
    }
    return InternalFutures.tryInternalFastPathGetFailure(this);
  }

  @SuppressWarnings("unchecked") // required by VarHandle usage
  private void runDoneWithError(@Nullable Throwable primaryCause) {
    var queue = (ConcurrentLinkedQueue<Throwable>) SECONDARY_EXCEPTIONS_HANDLE.getAcquire(this);
    ImmutableList<Throwable> secondaryCauses =
        queue == null ? ImmutableList.of() : ImmutableList.copyOf(queue);

    doneWithError(primaryCause, secondaryCauses);
  }

  static {
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      TASK_COUNT_HANDLE =
          lookup.findVarHandle(AbstractQuiescingFuture.class, "taskCount", int.class);
      SECONDARY_EXCEPTIONS_HANDLE =
          lookup.findVarHandle(
              AbstractQuiescingFuture.class, "secondaryExceptions", ConcurrentLinkedQueue.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
