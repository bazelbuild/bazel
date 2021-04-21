// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Includes options and states used by {@link MemoizingEvaluator#evaluate}, {@link
 * BuildDriver#evaluate} and {@link WalkableGraphFactory#prepareAndGet}
 */
public class EvaluationContext {
  private final int numThreads;
  @Nullable private final Supplier<ExecutorService> executorServiceSupplier;
  private final boolean keepGoing;
  private final ExtendedEventHandler eventHandler;
  private final boolean useForkJoinPool;
  private final boolean isExecutionPhase;

  protected EvaluationContext(
      int numThreads,
      @Nullable Supplier<ExecutorService> executorServiceSupplier,
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      boolean useForkJoinPool,
      boolean isExecutionPhase) {
    Preconditions.checkArgument(0 < numThreads, "numThreads must be positive");
    this.numThreads = numThreads;
    this.executorServiceSupplier = executorServiceSupplier;
    this.keepGoing = keepGoing;
    this.eventHandler = Preconditions.checkNotNull(eventHandler);
    this.useForkJoinPool = useForkJoinPool;
    this.isExecutionPhase = isExecutionPhase;
  }

  public int getParallelism() {
    return numThreads;
  }

  public Optional<Supplier<ExecutorService>> getExecutorServiceSupplier() {
    return Optional.ofNullable(executorServiceSupplier);
  }

  public boolean getKeepGoing() {
    return keepGoing;
  }

  public ExtendedEventHandler getEventHandler() {
    return eventHandler;
  }

  public EvaluationContext getCopyWithKeepGoing(boolean keepGoing) {
    if (this.keepGoing == keepGoing) {
      return this;
    } else {
      return new EvaluationContext(
          this.numThreads,
          this.executorServiceSupplier,
          keepGoing,
          this.eventHandler,
          this.useForkJoinPool,
          this.isExecutionPhase);
    }
  }

  public boolean getUseForkJoinPool() {
    return useForkJoinPool;
  }

  public boolean isExecutionPhase() {
    return isExecutionPhase;
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for {@link EvaluationContext}. */
  public static class Builder {
    private int numThreads;
    private Supplier<ExecutorService> executorServiceSupplier;
    private boolean keepGoing;
    private ExtendedEventHandler eventHandler;
    private boolean useForkJoinPool;
    private boolean isExecutionPhase = false;

    private Builder() {}

    public Builder copyFrom(EvaluationContext evaluationContext) {
      this.numThreads = evaluationContext.numThreads;
      this.executorServiceSupplier = evaluationContext.executorServiceSupplier;
      this.keepGoing = evaluationContext.keepGoing;
      this.eventHandler = evaluationContext.eventHandler;
      this.isExecutionPhase = evaluationContext.isExecutionPhase;
      this.useForkJoinPool = evaluationContext.useForkJoinPool;
      return this;
    }

    public Builder setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    public Builder setExecutorServiceSupplier(Supplier<ExecutorService> executorServiceSupplier) {
      this.executorServiceSupplier = executorServiceSupplier;
      return this;
    }

    public Builder setKeepGoing(boolean keepGoing) {
      this.keepGoing = keepGoing;
      return this;
    }

    public Builder setEventHandler(ExtendedEventHandler eventHandler) {
      this.eventHandler = eventHandler;
      return this;
    }

    public Builder setUseForkJoinPool(boolean useForkJoinPool) {
      this.useForkJoinPool = useForkJoinPool;
      return this;
    }

    public Builder setExecutionPhase() {
      isExecutionPhase = true;
      return this;
    }

    public EvaluationContext build() {
      return new EvaluationContext(
          numThreads,
          executorServiceSupplier,
          keepGoing,
          eventHandler,
          useForkJoinPool,
          isExecutionPhase);
    }
  }
}
