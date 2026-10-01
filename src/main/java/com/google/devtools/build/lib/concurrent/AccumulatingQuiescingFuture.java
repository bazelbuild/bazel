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
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.errorprone.annotations.DoNotCall;
import com.google.errorprone.annotations.ForOverride;

/**
 * An abstract base class for futures that accumulate the completion of homogeneous subtask futures
 * directly into an aggregated state.
 *
 * <p>Subclasses implement {@link #accumulateFutureResult} to fold completed subtask results into
 * their state and {@link #getValue()} to synthesize the final result upon quiescence.
 *
 * @param <T> the type of value produced by this future upon quiescence
 * @param <V> the type of value produced by subtask futures accumulated by this future
 */
public abstract class AccumulatingQuiescingFuture<T, V> extends QuiescingFuture<T>
    implements FutureCallback<V> {

  /**
   * Constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   */
  protected AccumulatingQuiescingFuture(SafeExecutor getValueExecutor) {
    super(getValueExecutor);
  }

  /**
   * Direct constructor.
   *
   * @param getValueExecutor runner for running {@link #getValue} or {@link #doneWithError}.
   * @param taskCount initial task count.
   */
  protected AccumulatingQuiescingFuture(SafeExecutor getValueExecutor, int taskCount) {
    super(getValueExecutor, taskCount);
  }

  /**
   * Registers a subtask future whose completion is accumulated directly by this instance,
   * automatically incrementing {@link #taskCount} and rolling back on submission failure.
   *
   * @param future {@code future}'s result will be accumulated upon completion
   * @param executor runner that executes the accumulation
   */
  public final void addFuture(ListenableFuture<? extends V> future, SafeExecutor executor) {
    Preconditions.checkNotNull(future, "future");
    Preconditions.checkNotNull(executor, "executor");
    increment();
    try {
      executor.addCallback(future, this);
    } catch (Throwable t) {
      notifyException(t);
    }
  }

  /**
   * Accumulates the result of a completed subtask future into the aggregated state.
   *
   * <p><b>Thread Safety & Commutativity:</b> This method may be called concurrently from multiple
   * threads when subtask futures complete in parallel, and results may arrive in arbitrary order.
   * Implementations must be thread-safe (e.g. using concurrent collections, atomic operations, or
   * synchronization) and commutative.
   *
   * <p>Decrementing {@link #taskCount} is guaranteed to execute automatically upon return or
   * exception.
   */
  @ForOverride
  protected abstract void accumulateFutureResult(V result);

  /**
   * Implements {@link FutureCallback#onSuccess}.
   *
   * @deprecated Do not call directly; only used by {@link SafeExecutor#addCallback} callback
   *     processing.
   */
  @Deprecated
  @Override
  @DoNotCall("Only for use by SafeExecutor / FutureCallback machinery")
  public final void onSuccess(V result) {
    try {
      accumulateFutureResult(result);
    } catch (Throwable t) {
      notifyException(t);
      return;
    }
    decrement();
  }

  /**
   * Implements {@link FutureCallback#onFailure}.
   *
   * @deprecated Do not call directly; only used by {@link SafeExecutor#addCallback} callback
   *     processing.
   */
  @Deprecated
  @Override
  @DoNotCall("Only for use by SafeExecutor / FutureCallback machinery")
  public final void onFailure(Throwable t) {
    notifyException(t);
  }
}
