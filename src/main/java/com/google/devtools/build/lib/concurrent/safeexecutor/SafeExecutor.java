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

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import com.google.errorprone.annotations.RestrictedApi;
import java.util.concurrent.Executor;

/**
 * A rejection-safe executor interface that handles execution rejections at submission boundaries.
 *
 * <p>There are 3 particularly common operations that are deadlock prone in the presence of task
 * rejections and executor shutdowns.
 *
 * <ul>
 *   <li>{@code Executor.execute}: since {@code RejectedExecutionException} is a {@code
 *       RuntimeException}, there's no systematic enforcement of its handling. {@link #execute}
 *       requires an explicit {@link RejectionHandlingRunnable} instead of a plain {@code Runnable}
 *       so that a handler, in case of task rejection, can be specified.
 *   <li>{@code Futures.addCallback}: Guava handles these by detecting {@code
 *       RejectedExecutionException} and logging, but does nothing further. Implementations forward
 *       the rejection to the callback's {@code onFailure} method.
 *   <li>Upon {@code shutdownNow}, most executors return enqueued, but not yet started tasks to the
 *       caller for processing. Implementations (such as {@code SafeExecutorOwner}) ensure that
 *       rejection notifications for those dropped tasks are dispatched.
 * </ul>
 */
@SkybridgeInterface
public interface SafeExecutor {

  /**
   * Returns a SafeExecutor singleton backed by direct inline execution that never rejects tasks.
   */
  static SafeExecutor safeDirectExecutor() {
    return SafeDirectExecutor.INSTANCE;
  }

  /**
   * Safely executes a RejectionHandlingRunnable task, forwarding submission rejections (e.g. {@code
   * RejectedExecutionException}) to {@code task.handleRejection(t)}.
   */
  void execute(RejectionHandlingRunnable task);

  /**
   * Safely adds a callback to a future, routing listener submission dispatches through SafeExecutor
   * protection.
   */
  <T> void addCallback(ListenableFuture<T> future, FutureCallback<? super T> callback);

  /**
   * Returns the underlying unsafe {@link Executor} delegate for Guava framework integration (e.g.
   * {@link SafeFutures}).
   *
   * <p>Warning: Direct calls to {@code execute(Runnable)} on this executor bypass {@link
   * SafeExecutor} rejection guards and should not be used in general application code.
   *
   * <p>This method exists primarily for use by {@link SafeFutures}, which uses it for intrinsically
   * safe Guava futures operations. {@link SafeFutures} is in the same logical package as this class
   * {@code com.google.devtools.build.lib.concurrent.safeexecutor}. However, {@link SafeFutures}
   * cannot see this method if it is marked package-private.
   *
   * <p>The reason is that this class is marked for Skybridge and in ERW that means this class and
   * {@link SafeFutures} are loaded by different class loaders. Consequently, even though they are
   * logically in the same package, the JVM does not agree due to the class loader difference. To
   * work around this, this method is public and we resort to {@code RestrictedApi} for
   * encapsulation.
   */
  @RestrictedApi(
      explanation =
          "Direct access to unsafe executor bypasses SafeExecutor rejection guards. Use SafeFutures"
              + " instead.",
      allowedOnPath = ".*/com/google/devtools/build/lib/concurrent/safeexecutor/.*")
  Executor getInternalUnsafeExecutor();
}
