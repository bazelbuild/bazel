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

import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.Future;

/** A {@link QuiescingExecutor} implementation that wraps a {@link ForkJoinPool}. */
// TODO(bazel-team): This extends AQV to ensure that they share the same semantics for interrupt
// handling, error propagation, and task completion. Because FJP provides a native implementation
// for awaitQuiescence, a careful refactoring would allow FJQE to avoid the overhead of
// maintaining AQV.remainingTasks.
public class ForkJoinQuiescingExecutor extends AbstractQueueVisitor {

  private ForkJoinQuiescingExecutor(ForkJoinPool forkJoinPool, ErrorClassifier errorClassifier) {
    super(
        forkJoinPool, ExecutorOwnership.PRIVATE, ExceptionHandlingMode.FAIL_FAST, errorClassifier);
  }

  /** Builder for {@link ForkJoinQuiescingExecutor}. */
  public static class Builder {
    private ForkJoinPool forkJoinPool = null;
    private ErrorClassifier errorClassifier = ErrorClassifier.DEFAULT;

    private Builder() {
    }

    /**
     * Sets the {@link ForkJoinPool} that will be used by the to-be-built {@link
     * ForkJoinQuiescingExecutor}. The given {@link ForkJoinPool} will be shut down on completion of
     * the {@link ForkJoinQuiescingExecutor}.
     */
    @CanIgnoreReturnValue
    public Builder withOwnershipOf(ForkJoinPool forkJoinPool) {
      Preconditions.checkState(this.forkJoinPool == null);
      this.forkJoinPool = forkJoinPool;
      return this;
    }

    /**
     * Sets the {@link ErrorClassifier} that will be used by the to-be-built {@link
     * ForkJoinQuiescingExecutor}.
     */
    @CanIgnoreReturnValue
    public Builder setErrorClassifier(ErrorClassifier errorClassifier) {
      this.errorClassifier = errorClassifier;
      return this;
    }

    /**
     * Returns a fresh {@link ForkJoinQuiescingExecutor} using the previously given options.
     */
    public ForkJoinQuiescingExecutor build() {
      Preconditions.checkNotNull(forkJoinPool, "fork join pool must be supplied");
      return new ForkJoinQuiescingExecutor(forkJoinPool, errorClassifier);
    }
  }

  /** Returns a fresh {@link Builder}. */
  public static Builder newBuilder() {
    return new Builder();
  }

  @Override
  protected void executeWrappedRunnable(WrappedRunnable runnable, ExecutorService executorService) {
    if (ForkJoinTask.getPool() == executorService) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError = ForkJoinTask.adapt(runnable).fork();
    } else {
      super.executeWrappedRunnable(runnable, executorService);
    }
  }
}
