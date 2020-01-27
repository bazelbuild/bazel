// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.util.concurrent.ListenableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;

/**
 * QuiescingExecutor is an {@link Executor} which supports waiting until all submitted tasks are
 * complete. This is useful when tasks may submit additional tasks.
 *
 * <p>Consider the following example:
 * <pre>
 *   ThreadPoolExecutor executor = <...>
 *   executor.submit(myRunnableTask);
 *   executor.shutdown();
 *   executor.awaitTermination();
 * </pre>
 *
 * <p>This won't work properly if {@code myRunnableTask} submits additional tasks to the
 * executor, because it may already have shut down by that point.
 *
 * <p>QuiescingExecutor supports interruption. If the main thread is interrupted, tasks will no
 * longer be started, and the {@link #awaitQuiescence} method will throw {@link
 * InterruptedException}.
 */
public interface QuiescingExecutor extends Executor {

  /**
   * Waits for all tasks to complete. If the {@link QuiescingExecutor} owns its own {@link
   * java.util.concurrent.ExecutorService}, the service will also be shutdown.
   *
   * <p>Throws (the same) unchecked exception if any worker thread failed unexpectedly. If the main
   * thread is interrupted and a worker also throws an unchecked exception, the unchecked exception
   * is rethrown, since it may indicate a programming bug. If callers handle the unchecked
   * exception, they may check the interrupted bit to see if the pool was interrupted.
   *
   * @param interruptWorkers if true, interrupt worker threads if main thread gets an interrupt.
   *                         If false, just wait for them to terminate normally.
   */
  void awaitQuiescence(boolean interruptWorkers) throws InterruptedException;

  /**
   * Prevent quiescence of the executor until the given future is completed. If the executor is
   * interrupted, then the executor will call {@link ListenableFuture#cancel} with a parameter of
   * {@code true}.
   */
  void dependOnFuture(ListenableFuture<?> future) throws InterruptedException;

  /** Get latch that is released if a task throws an exception. Used only in tests. */
  @VisibleForTesting
  CountDownLatch getExceptionLatchForTestingOnly();

  /** Get latch that is released if a task is interrupted. Used only in tests. */
  @VisibleForTesting
  CountDownLatch getInterruptionLatchForTestingOnly();
}
