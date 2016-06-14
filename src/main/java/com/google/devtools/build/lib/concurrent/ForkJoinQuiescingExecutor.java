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

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

/** A {@link QuiescingExecutor} implementation that wraps a {@link ForkJoinPool}. */
// TODO(bazel-team): This extends AQV to ensure that they share the same semantics for interrupt
// handling, error propagation, and task completion. Because FJP provides a native implementation
// for awaitQuiescence, a careful refactoring would allow FJQE to avoid the overhead of
// maintaining AQV.remainingTasks.
public class ForkJoinQuiescingExecutor extends AbstractQueueVisitor {

  public ForkJoinQuiescingExecutor(ForkJoinPool forkJoinPool, ErrorClassifier errorClassifier,
      ErrorHandler errorHandler) {
    super(
        /*concurrent=*/ true,
        forkJoinPool,
        /*shutdownOnCompletion=*/ true,
        /*failFastOnException=*/ true,
        errorClassifier,
        errorHandler);
  }

  @Override
  protected void executeRunnable(Runnable runnable) {
    if (ForkJoinTask.inForkJoinPool()) {
      ForkJoinTask.adapt(runnable).fork();
    } else {
      super.executeRunnable(runnable);
    }
  }
}
