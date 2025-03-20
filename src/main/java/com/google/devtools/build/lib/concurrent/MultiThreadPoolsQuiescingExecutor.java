// Copyright 2021 The Bazel Authors. All rights reserved.
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

/** An Executor that can execute tasks in multiple independent thread pools. */
public interface MultiThreadPoolsQuiescingExecutor extends QuiescingExecutor {
  /** The types of thread pools to use. */
  enum ThreadPoolType {
    // Suitable for CPU-heavy tasks. Ideally the number of threads is close to the machine's number
    // of cores.
    CPU_HEAVY,
    // Reserved for execution-phase tasks.
    EXECUTION_PHASE,
    REGULAR,
  }

  /**
   * Execute the runnable, taking into consideration the preferred thread pool type.
   *
   * <p>WARNING: it is the developer's responsibility to check that the appropriate thread pool is
   * present.
   *
   * @param shouldStallAwaitingSignal whether the execution of the runnable should be stalled. No-op
   *     if the go ahead signal is already given.
   */
  void execute(Runnable runnable, ThreadPoolType threadPoolType, boolean shouldStallAwaitingSignal);

  default void execute(Runnable runnable, ThreadPoolType threadPoolType) {
    execute(runnable, threadPoolType, /* shouldStallAwaitingSignal= */ false);
  }

  /**
   * Begin executing execution phase tasks.
   *
   * <p>The execution phase tasks sent to this executor will be queued up until this method is
   * called. This is called exactly once per build.
   *
   * <p>Skymeld-only.
   */
  void launchQueuedUpExecutionPhaseTasks();

  /** Only true when in Skymeld mode. */
  boolean hasSeparatePoolForExecutionTasks();
}
