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

import com.google.errorprone.annotations.ForOverride;
import com.google.errorprone.annotations.Keep;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.Executor;

/**
 * A future that tracks in-flight tasks and completes when the tasks quiesce or an error occurs.
 *
 * <p>Unlike {@link QuiescingFuture}, this class is itself a task that can be submitted to an {@link
 * Executor}.
 *
 * <p>This class uses the "pre-increment" pattern (initializing {@code taskCount} to 1) to prevent
 * premature completion during initialization. However, it <b>automatically</b> offsets this by
 * calling {@link #decrement} at the end of {@link #run} (after {@link #arrangeSubtasks}). Unlike
 * {@link QuiescingFuture}, users of {@link QuiescingFutureTask} do <b>not</b> need to call {@link
 * #decrement} manually to offset the initial count.
 */
public abstract class QuiescingFutureTask<T> extends AbstractQuiescingFuture<T> {
  private static final VarHandle STATE_HANDLE;

  /** State used to distinguish between the initial run and subsequent completion runs. */
  @Keep // used via STATE_HANDLE
  private volatile int state = 0;

  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   */
  public QuiescingFutureTask(Executor getValueExecutor) {
    super(getValueExecutor, /* taskCount= */ 1);
  }

  /**
   * Arranges subtasks.
   *
   * <p>Implementations should call {@link #increment} for each subtask and {@link #decrement} once
   * the subtask completes.
   *
   * <p>Note: The base class's {@link #run} method automatically calls {@link #decrement} to offset
   * the initial count after this method completes.
   *
   * <p>If this method fails with an unchecked exception, the future is failed immediately. In this
   * case, there's no guarantee that {@link #doneWithError} is called.
   */
  @ForOverride
  protected abstract void arrangeSubtasks();

  /**
   * Called to either arrange subtasks or handle quiescence.
   *
   * <p>Unlike {@link QuiescingFuture}, this method is used for both the initial setup (by
   * submitting this task to an executor) and for finalization (when the task count reaches zero).
   *
   * <ul>
   *   <li><b>INITIAL (0):</b> The first call to this method executes {@link #arrangeSubtasks} and
   *       then calls {@link #decrement} to offset the initial count.
   *   <li><b>ARRANGED (1):</b> Subsequent calls to this method (triggered when the task count
   *       reaches zero) will execute the completion logic via {@link #handleQuiescence}.
   * </ul>
   */
  @Override
  public final void run() {
    if (STATE_HANDLE.compareAndSet(this, 0, 1)) {
      try {
        arrangeSubtasks();
        decrement();
      } catch (Throwable t) {
        notifyException(t);
      }
    } else {
      handleQuiescence();
    }
  }

  static {
    MethodHandles.Lookup lookup = MethodHandles.lookup();
    try {
      STATE_HANDLE = lookup.findVarHandle(QuiescingFutureTask.class, "state", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
