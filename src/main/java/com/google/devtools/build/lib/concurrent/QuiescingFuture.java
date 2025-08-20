// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.util.concurrent.AbstractFuture;
import com.google.errorprone.annotations.DoNotCall;
import com.google.errorprone.annotations.ForOverride;
import com.google.errorprone.annotations.Keep;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.Executor;

/**
 * A future that tracks in-flight tasks and completes when the tasks quiesce or an error occurs.
 *
 * <p>It uses the <em>pre-increment</em> pattern, which is useful in cases where the task count can
 * transiently hit zero during setup or if there are cases where no tasks are created at all. The
 * caller should call {@link #decrement} one additional time after initialization to offset the
 * pre-increment. Typical usage looks like the following.
 *
 * <ol>
 *   <li>Create a {@link QuiescingFuture}. Once created, the future may be used freely, for example,
 *       to chain other tasks.
 *   <li>Start tasks, instrumented with calls to {@link #increment} on creation and {@link
 *       #decrement} on completion. The tasks may recursively create more instrumented tasks. If a
 *       task recursively creates child tasks, it must {@link #increment} for child tasks before
 *       calling {@link #decrement} to mark its own completion to avoid premature completion.
 *   <li>Call {@link #decrement} once to offset the <em>pre-increment</em>.
 *   <li>The future completes once all the tasks complete (but not before step 3 above).
 * </ol>
 */
public abstract class QuiescingFuture<T> extends AbstractFuture<T> implements Runnable {
  /**
   * Handle for {@link #taskCount}.
   *
   * <p>This uses less memory than {@link java.util.concurrent.AtomicInteger}.
   */
  private static final VarHandle TASK_COUNT_HANDLE;

  private static final VarHandle ERROR_COUNT_HANDLE;

  private final Executor getValueExecutor;

  /**
   * Count of in-flight tasks.
   *
   * <p>Pre-incremented to 1 to guarantee completion, even if there are 0 increments or if counts
   * transiently reach 0 during initialization. Clients must call {@link #decrement} one additional
   * time after setup.
   *
   * <p>Use {@link #TASK_COUNT_HANDLE} for atomic operations.
   */
  @Keep // used via TASK_COUNT_HANDLE
  private volatile int taskCount = 1;

  @Keep // used via ERROR_COUNT_HANDLE
  private volatile int errorCount = 0;

  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   */
  public QuiescingFuture(Executor getValueExecutor) {
    this.getValueExecutor = getValueExecutor;
  }

  /** Increments the task count. */
  protected final void increment() {
    TASK_COUNT_HANDLE.getAndAdd(this, 1);
  }

  /** Decrements the task count. */
  protected final void decrement() {
    int countBeforeDecrement = (int) TASK_COUNT_HANDLE.getAndAdd(this, -1);
    if (countBeforeDecrement == 1) {
      getValueExecutor.execute((Runnable) this);
    }
  }

  /**
   * Sets the future as failing with {@code t}.
   *
   * <p>If the client calls this, it should not call {@link #decrement} for the same task. It's
   * already called.
   */
  protected final void notifyException(Throwable t) {
    setException(t);
    ERROR_COUNT_HANDLE.getAndAdd(this, 1);
    decrement();
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
   * <p>Allows clients to perform cleanup work if there is an error.
   */
  @ForOverride
  protected void doneWithError() {}

  /**
   * Called when all tasks are complete.
   *
   * @deprecated only for {@link #decrement}
   */
  @Deprecated
  @Override
  @DoNotCall
  public final void run() {
    if ((int) ERROR_COUNT_HANDLE.getAcquire(this) > 0) {
      doneWithError();
    } else {
      set(getValue());
    }
  }

  static {
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      TASK_COUNT_HANDLE = lookup.findVarHandle(QuiescingFuture.class, "taskCount", int.class);
      ERROR_COUNT_HANDLE = lookup.findVarHandle(QuiescingFuture.class, "errorCount", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
