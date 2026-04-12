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

import com.google.common.util.concurrent.AbstractFuture;
import com.google.errorprone.annotations.ForOverride;
import com.google.errorprone.annotations.Keep;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.Executor;

/**
 * A base class for futures that track in-flight tasks and complete when the tasks quiesce or an
 * error occurs.
 */
public abstract class AbstractQuiescingFuture<T> extends AbstractFuture<T> implements Runnable {
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
   * <p>This is initialized to 1 to support the "pre-increment" pattern, which prevents premature
   * completion during initialization.
   *
   * <p>Use {@link #TASK_COUNT_HANDLE} for atomic operations.
   */
  @Keep // used via TASK_COUNT_HANDLE
  private volatile int taskCount;

  @Keep // used via ERROR_COUNT_HANDLE
  private volatile int errorCount = 0;

  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   * @param taskCount initial task count.
   */
  protected AbstractQuiescingFuture(Executor getValueExecutor, int taskCount) {
    this.getValueExecutor = getValueExecutor;
    this.taskCount = taskCount;
  }

  /** Increments the task count. */
  protected final void increment() {
    TASK_COUNT_HANDLE.getAndAdd(this, 1);
  }

  /** Decrements the task count. */
  protected final void decrement() {
    int countBeforeDecrement = (int) TASK_COUNT_HANDLE.getAndAdd(this, -1);
    if (countBeforeDecrement == 1) {
      getValueExecutor.execute(this);
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

  final void handleQuiescence() {
    if ((int) ERROR_COUNT_HANDLE.getAcquire(this) > 0) {
      doneWithError();
    } else {
      set(getValue());
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
   * <p>Allows clients to perform cleanup work if there is an error.
   */
  @ForOverride
  protected void doneWithError() {}

  static {
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      TASK_COUNT_HANDLE =
          lookup.findVarHandle(AbstractQuiescingFuture.class, "taskCount", int.class);
      ERROR_COUNT_HANDLE =
          lookup.findVarHandle(AbstractQuiescingFuture.class, "errorCount", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
