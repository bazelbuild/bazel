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
import java.util.concurrent.ExecutorService;

/**
 * An implementation of MultiThreadPoolsQuiescingExecutor that has 2 ExecutorServices, one with a
 * larger thread pool for IO/Network-bound tasks, and one with a smaller thread pool for CPU-bound
 * tasks.
 */
public class DualExecutorQueueVisitor extends AbstractQueueVisitor
    implements MultiThreadPoolsQuiescingExecutor {
  private final ExecutorService regularPoolExecutorService;
  private final ExecutorService cpuHeavyPoolExecutorService;

  private DualExecutorQueueVisitor(
      ExecutorService regularPoolExecutorService,
      ExecutorService cpuHeavyPoolExecutorService,
      boolean failFastOnException,
      ErrorClassifier errorClassifier) {
    super(
        regularPoolExecutorService,
        /*shutdownOnCompletion=*/ true,
        failFastOnException,
        errorClassifier);
    this.regularPoolExecutorService = super.getExecutorService();
    this.cpuHeavyPoolExecutorService = Preconditions.checkNotNull(cpuHeavyPoolExecutorService);
  }

  public static AbstractQueueVisitor createWithExecutorServices(
      ExecutorService executorService,
      ExecutorService cpuExecutorService,
      boolean failFastOnException,
      ErrorClassifier errorClassifier) {
    return new DualExecutorQueueVisitor(
        executorService, cpuExecutorService, failFastOnException, errorClassifier);
  }

  @Override
  public final void execute(Runnable runnable, ThreadPoolType threadPoolType) {
    super.executeWithExecutorService(runnable, getExecutorServiceByThreadPoolType(threadPoolType));
  }

  @VisibleForTesting
  ExecutorService getExecutorServiceByThreadPoolType(ThreadPoolType threadPoolType) {
    switch (threadPoolType) {
      case REGULAR:
        return regularPoolExecutorService;
      case CPU_HEAVY:
        return cpuHeavyPoolExecutorService;
    }
    throw new IllegalStateException("Invalid ThreadPoolType: " + threadPoolType);
  }

  @Override
  protected void shutdownExecutorService(Throwable catastrophe) {
    if (catastrophe != null) {
      Throwables.throwIfUnchecked(catastrophe);
    }
    internalShutdownExecutorService(regularPoolExecutorService);
    internalShutdownExecutorService(cpuHeavyPoolExecutorService);
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
}
