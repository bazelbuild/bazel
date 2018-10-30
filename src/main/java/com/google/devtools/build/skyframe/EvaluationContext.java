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
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Includes options and states used by {@link MemoizingEvaluator#evaluate}, {@link
 * BuildDriver#evaluate} and {@link WalkableGraphFactory#prepareAndGet}
 */
public class EvaluationContext {
  @Nullable private final Integer numThreads;
  @Nullable private final Supplier<ExecutorService> executorService;
  private final boolean keepGoing;
  private final ExtendedEventHandler eventHandler;

  protected EvaluationContext(
      @Nullable Integer numThread,
      @Nullable Supplier<ExecutorService> executorService,
      boolean keepGoing,
      ExtendedEventHandler eventHandler) {
    this.numThreads = numThread;
    this.executorService = executorService;
    this.keepGoing = keepGoing;
    this.eventHandler = eventHandler;
  }

  public Integer getNumThreads() {
    return numThreads;
  }

  public Supplier<ExecutorService> getExecutorService() {
    return executorService;
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
          this.numThreads, this.executorService, keepGoing, this.eventHandler);
    }
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for {@link EvaluationContext}. */
  public static class Builder {
    private Integer numThread;
    private Supplier<ExecutorService> executorService;
    private boolean keepGoing;
    private ExtendedEventHandler eventHandler;

    private Builder() {}

    public Builder copyFrom(EvaluationContext evaluationContext) {
      this.numThread = evaluationContext.numThreads;
      this.executorService = evaluationContext.executorService;
      this.keepGoing = evaluationContext.keepGoing;
      this.eventHandler = evaluationContext.eventHandler;
      return this;
    }

    public Builder setNumThreads(int numThread) {
      this.numThread = numThread;
      this.executorService = null;
      return this;
    }

    public Builder setExecutorServiceSupplier(Supplier<ExecutorService> executorService) {
      this.executorService = executorService;
      this.numThread = null;
      return this;
    }

    public Builder setKeepGoing(boolean keepGoing) {
      this.keepGoing = keepGoing;
      return this;
    }

    public Builder setEventHander(ExtendedEventHandler eventHandler) {
      this.eventHandler = eventHandler;
      return this;
    }

    public EvaluationContext build() {
      Preconditions.checkState(
          (numThread == null && executorService != null)
              || (numThread != null && executorService == null),
          "Exactly one of numThread and executorService must be set. %s %s",
          numThread,
          executorService);
      return new EvaluationContext(numThread, executorService, keepGoing, eventHandler);
    }
  }
}
