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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.errorprone.annotations.DoNotCall;

/**
 * A future that tracks in-flight tasks and completes when the tasks quiesce or an error occurs.
 *
 * <p>It uses the <em>pre-increment</em> pattern (initializing {@code taskCount} to 1), which is
 * useful in cases where the task count can transiently hit zero during setup or if there are cases
 * where no tasks are created at all. The caller should call {@link #decrement} one additional time
 * after initialization to offset the pre-increment.
 *
 * <p>Contrast this with {@link QuiescingFutureTask}, which handles this offset automatically. In
 * this class, the manual call to {@link #decrement} is <b>mandatory</b>. Typical usage looks like
 * the following.
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
public abstract class QuiescingFuture<T> extends AbstractQuiescingFuture<T> {
  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   */
  public QuiescingFuture(SafeExecutor getValueExecutor) {
    super(getValueExecutor, /* taskCount= */ 1);
  }

  /**
   * Direct constructor.
   *
   * <p>This is useful when the total number of tasks is known in advance.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   * @param taskCount initial task count, <i>no pre-increment</i> is applied
   */
  public QuiescingFuture(SafeExecutor getValueExecutor, int taskCount) {
    super(getValueExecutor, taskCount);
  }

  /**
   * Decrements the task count to offset the pre-increment from construction.
   *
   * <p>Must be called after all initial tasks have been registered.
   */
  public final void finishRegistration() {
    decrement();
  }

  /** Called when all tasks are complete. */
  @Override
  @DoNotCall("Called only by SafeExecutor via submission from decrement() at quiescence.")
  public final void run() {
    handleQuiescence();
  }

  @Override
  @DoNotCall("Only called by SafeExecutor upon rejection.")
  public final void handleRejection(Throwable t) {
    Preconditions.checkNotNull(t, "t");
    try {
      recordException(t);
    } finally {
      handleQuiescence();
    }
  }
}
