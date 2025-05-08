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
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.Lists;
import java.util.List;
import java.util.concurrent.ExecutorService;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * An implementation of MultiThreadPoolsQuiescingExecutor that has 2 ExecutorServices, one with a
 * larger thread pool for IO/Network-bound tasks, and one with a smaller thread pool for CPU-bound
 * tasks.
 *
 * <p>With merged analysis and execution phases, this QueueVisitor is responsible for all 3 phases:
 * loading, analysis and execution. There's an additional 3rd pool for execution tasks. This is done
 * for performance reason: each of these phases has an optimal number of threads for its thread
 * pool.
 *
 * <p>Created anew each build.
 */
public final class MultiExecutorQueueVisitor extends AbstractQueueVisitor
    implements MultiThreadPoolsQuiescingExecutor {
  private final ExecutorService regularPoolExecutorService;
  private final ExecutorService cpuHeavyPoolExecutorService;
  @Nullable private final ExecutorService executionPhaseExecutorService;

  // Whether execution phase tasks should be allowed to move forward.
  private boolean executionPhaseTasksGoAhead;

  @GuardedBy("this")
  @Nullable
  private List<Runnable> queuedPendingGoAhead;

  private MultiExecutorQueueVisitor(
      ExecutorService regularPoolExecutorService,
      ExecutorService cpuHeavyPoolExecutorService,
      @Nullable ExecutorService executionPhaseExecutorService,
      ExceptionHandlingMode exceptionHandlingMode,
      ErrorClassifier errorClassifier) {
    super(
        regularPoolExecutorService,
        ExecutorOwnership.PRIVATE,
        exceptionHandlingMode,
        errorClassifier);
    this.regularPoolExecutorService = super.getExecutorService();
    this.cpuHeavyPoolExecutorService = Preconditions.checkNotNull(cpuHeavyPoolExecutorService);
    this.executionPhaseExecutorService = executionPhaseExecutorService;
    this.executionPhaseTasksGoAhead = executionPhaseExecutorService == null;

    if (executionPhaseExecutorService != null) {
      queuedPendingGoAhead = Lists.newArrayList();
    }
  }

  public static MultiExecutorQueueVisitor createWithExecutorServices(
      ExecutorService regularPoolExecutorService,
      ExecutorService cpuHeavyPoolExecutorService,
      ExceptionHandlingMode exceptionHandlingMode,
      ErrorClassifier errorClassifier) {
    return createWithExecutorServices(
        regularPoolExecutorService,
        cpuHeavyPoolExecutorService,
        /* executionPhaseExecutorService= */ null,
        exceptionHandlingMode,
        errorClassifier);
  }

  public static MultiExecutorQueueVisitor createWithExecutorServices(
      ExecutorService regularPoolExecutorService,
      ExecutorService cpuHeavyPoolExecutorService,
      ExecutorService executionPhaseExecutorService,
      ExceptionHandlingMode exceptionHandlingMode,
      ErrorClassifier errorClassifier) {
    return new MultiExecutorQueueVisitor(
        regularPoolExecutorService,
        cpuHeavyPoolExecutorService,
        executionPhaseExecutorService,
        exceptionHandlingMode,
        errorClassifier);
  }

  @Override
  public void execute(
      Runnable runnable, ThreadPoolType threadPoolType, boolean shouldStallAwaitingSignal) {
    if (shouldStallAwaitingSignal && !executionPhaseTasksGoAhead) {
      synchronized (this) {
        if (!executionPhaseTasksGoAhead) {
          Preconditions.checkNotNull(queuedPendingGoAhead).add(runnable);
          return;
        }
      }
    }
    super.executeWithExecutorService(runnable, getExecutorServiceByThreadPoolType(threadPoolType));
  }

  @VisibleForTesting
  ExecutorService getExecutorServiceByThreadPoolType(ThreadPoolType threadPoolType) {
    return switch (threadPoolType) {
      case REGULAR -> regularPoolExecutorService;
      case CPU_HEAVY -> cpuHeavyPoolExecutorService;
      case EXECUTION_PHASE -> {
        Preconditions.checkNotNull(executionPhaseExecutorService);
        yield executionPhaseExecutorService;
      }
    };
  }

  @Override
  protected void shutdownExecutorService(Throwable catastrophe) {
    if (catastrophe != null) {
      Throwables.throwIfUnchecked(catastrophe);
    }
    internalShutdownExecutorService(regularPoolExecutorService);
    internalShutdownExecutorService(cpuHeavyPoolExecutorService);
    if (executionPhaseExecutorService != null) {
      internalShutdownExecutorService(executionPhaseExecutorService);
    }
  }

  private void internalShutdownExecutorService(ExecutorService executorService) {
    executorService.shutdown();
    while (true) {
      try {
        executorService.awaitTermination(Integer.MAX_VALUE, SECONDS);
        break;
      } catch (InterruptedException e) {
        setInterrupted();
      }
    }
  }

  @Override
  public void launchQueuedUpExecutionPhaseTasks() {
    synchronized (this) {
      executionPhaseTasksGoAhead = true;
      for (Runnable runnable : Preconditions.checkNotNull(queuedPendingGoAhead)) {
        execute(runnable, ThreadPoolType.EXECUTION_PHASE, /* shouldStallAwaitingSignal= */ false);
      }
      queuedPendingGoAhead = null;
    }
  }

  @Override
  public boolean hasSeparatePoolForExecutionTasks() {
    return executionPhaseExecutorService != null;
  }
}
