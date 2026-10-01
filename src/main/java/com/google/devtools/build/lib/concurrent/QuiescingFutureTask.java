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
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.errorprone.annotations.DoNotCall;
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
public abstract class QuiescingFutureTask<T> extends AbstractQuiescingFuture<T>
    implements RejectionHandlingRunnable {
  private static final VarHandle STATE_HANDLE;

  private static final int STATE_INITIAL = 0;
  private static final int STATE_STARTED = 1;

  /** State used to distinguish between the initial run and subsequent completion runs. */
  @Keep // used via STATE_HANDLE
  private volatile int state = STATE_INITIAL;

  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   */
  public QuiescingFutureTask(SafeExecutor getValueExecutor) {
    super(getValueExecutor, /* taskCount= */ 1);
  }

  /**
   * Arranges subtasks.
   *
   * <p>Implementations should schedule subtasks via {@link #executeSubtask} or manually instrument
   * subtasks with {@link #increment} and {@link #decrement}.
   *
   * <p>Note: This class's {@link #run} method automatically calls {@link #decrement} in a finally
   * block after this method completes to offset the initial count.
   */
  @ForOverride
  protected abstract void arrangeSubtasks();

  /**
   * Called to either arrange subtasks or handle quiescence.
   *
   * <ul>
   *   <li><b>INITIAL (0):</b> The first call to this method executes {@link #arrangeSubtasks} and
   *       then calls {@link #decrement} in a {@code finally} block to offset the initial count.
   *   <li><b>STARTED (1):</b> Subsequent calls to this method (triggered when the task count
   *       reaches zero) will execute the completion logic via {@link #handleQuiescence}.
   * </ul>
   *
   * <p>Unlike with other classes of this family, it is okay for clients to call this for
   * synchronous dispatch of {@link #arrangeSubtasks}.
   */
  @Override
  public final void run() {
    if (STATE_HANDLE.compareAndSet(this, STATE_INITIAL, STATE_STARTED)) {
      try {
        arrangeSubtasks();
      } catch (Throwable t) {
        recordException(t);
      } finally {
        decrement();
      }
    } else {
      handleQuiescence();
    }
  }

  @Override
  @DoNotCall("Only called by SafeExecutor upon rejection.")
  public final void handleRejection(Throwable t) {
    // There are 2 points in the lifecycle of QuiescingFutureTask where a rejection is possible.
    // 1. Pre-arrangeSubtasks. The first call to `run` should perform the arrangement, but can be
    // rejected.
    // 2. Upon quiescence, the second call to `run` is meant to trigger handleQuiescence. This can
    // also result in a rejection.
    Preconditions.checkNotNull(t, "t");
    recordException(t);
    if (STATE_HANDLE.compareAndSet(this, STATE_INITIAL, STATE_STARTED)) {
      // This branch is reached if the rejection happened at (1). Since the initial arrangeSubtasks
      // was never called, the pre-increment is still present. We decrement it here for consistency.
      decrementWithoutScheduling();
    }
    handleQuiescence();
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
