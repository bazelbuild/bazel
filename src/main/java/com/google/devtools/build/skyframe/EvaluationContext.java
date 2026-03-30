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
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.WalkableGraph.WalkableGraphFactory;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Includes options and states used by {@link MemoizingEvaluator#evaluate}, {@link
 * MemoizingEvaluator#evaluate} and {@link WalkableGraphFactory#prepareAndGet}
 */
public class EvaluationContext {
  private final int parallelism;
  @Nullable private final QuiescingExecutor executor;
  private final boolean keepGoing;
  private final ExtendedEventHandler eventHandler;
  private final boolean isExecutionPhase;
  private final boolean mergingSkyframeAnalysisExecutionPhases;
  private final boolean storeExactCycles;
  private final UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver;

  private final boolean detectCycles;

  protected EvaluationContext(
      int parallelism,
      @Nullable QuiescingExecutor executor,
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      boolean isExecutionPhase,
      boolean mergingSkyframeAnalysisExecutionPhases,
      boolean storeExactCycles,
      UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver,
      boolean detectCycles) {
    this.parallelism = parallelism;
    this.executor = executor;
    this.keepGoing = keepGoing;
    this.eventHandler = Preconditions.checkNotNull(eventHandler);
    this.isExecutionPhase = isExecutionPhase;
    this.mergingSkyframeAnalysisExecutionPhases = mergingSkyframeAnalysisExecutionPhases;
    this.storeExactCycles = storeExactCycles;
    this.unnecessaryTemporaryStateDropperReceiver = unnecessaryTemporaryStateDropperReceiver;
    this.detectCycles = detectCycles;
  }

  public int getParallelism() {
    return parallelism;
  }

  public Optional<QuiescingExecutor> getExecutor() {
    return Optional.ofNullable(executor);
  }

  public boolean getKeepGoing() {
    return keepGoing;
  }

  public ExtendedEventHandler getEventHandler() {
    return eventHandler;
  }

  public boolean isExecutionPhase() {
    return isExecutionPhase;
  }

  public boolean mergingSkyframeAnalysisExecutionPhases() {
    return mergingSkyframeAnalysisExecutionPhases;
  }

  public boolean storeExactCycles() {
    return storeExactCycles;
  }

  /**
   * Drops unnecessary temporary state used internally by the current evaluation.
   *
   * <p>If the current evaluation is slow because of GC thrashing, and the GC thrashing is partially
   * caused by this temporary state, dropping it may reduce the wall time of the current evaluation.
   * On the other hand, if the current evaluation is not GC thrashing, then dropping this temporary
   * state will probably increase the wall time.
   */
  public interface UnnecessaryTemporaryStateDropper {
    @ThreadSafe
    void drop();
  }

  /**
   * A receiver of a {@link UnnecessaryTemporaryStateDropper} instance tied to the current
   * evaluation.
   */
  public interface UnnecessaryTemporaryStateDropperReceiver {
    UnnecessaryTemporaryStateDropperReceiver NULL =
        new UnnecessaryTemporaryStateDropperReceiver() {
          @Override
          public void onEvaluationStarted(UnnecessaryTemporaryStateDropper dropper) {}

          @Override
          public void onEvaluationFinished() {}
        };

    void onEvaluationStarted(UnnecessaryTemporaryStateDropper dropper);

    void onEvaluationFinished();
  }

  public UnnecessaryTemporaryStateDropperReceiver getUnnecessaryTemporaryStateDropperReceiver() {
    return unnecessaryTemporaryStateDropperReceiver;
  }

  public boolean detectCycles() {
    return detectCycles;
  }

  public Builder builder() {
    return newBuilder().copyFrom(this);
  }

  public static Builder newBuilder() {
    return new Builder();
  }

  /** Builder for {@link EvaluationContext}. */
  public static class Builder {
    protected int parallelism;
    protected QuiescingExecutor executor;
    protected boolean keepGoing;
    protected ExtendedEventHandler eventHandler;
    protected boolean isExecutionPhase = false;
    protected boolean mergingSkyframeAnalysisExecutionPhases;
    protected boolean storeExactCycles = true;
    protected UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver =
        UnnecessaryTemporaryStateDropperReceiver.NULL;

    protected boolean detectCycles = true;

    protected Builder() {}

    @CanIgnoreReturnValue
    protected Builder copyFrom(EvaluationContext evaluationContext) {
      this.parallelism = evaluationContext.parallelism;
      this.executor = evaluationContext.executor;
      this.keepGoing = evaluationContext.keepGoing;
      this.eventHandler = evaluationContext.eventHandler;
      this.isExecutionPhase = evaluationContext.isExecutionPhase;
      this.mergingSkyframeAnalysisExecutionPhases =
          evaluationContext.mergingSkyframeAnalysisExecutionPhases;
      this.storeExactCycles = evaluationContext.storeExactCycles;
      this.unnecessaryTemporaryStateDropperReceiver =
          evaluationContext.unnecessaryTemporaryStateDropperReceiver;
      this.detectCycles = evaluationContext.detectCycles;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setParallelism(int parallelism) {
      this.parallelism = parallelism;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecutor(QuiescingExecutor executor) {
      this.executor = executor;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setKeepGoing(boolean keepGoing) {
      this.keepGoing = keepGoing;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setEventHandler(ExtendedEventHandler eventHandler) {
      this.eventHandler = eventHandler;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecutionPhase() {
      this.isExecutionPhase = true;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setMergingSkyframeAnalysisExecutionPhases(
        boolean mergingSkyframeAnalysisExecutionPhases) {
      this.mergingSkyframeAnalysisExecutionPhases = mergingSkyframeAnalysisExecutionPhases;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setUnnecessaryTemporaryStateDropperReceiver(
        UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver) {
      this.unnecessaryTemporaryStateDropperReceiver = unnecessaryTemporaryStateDropperReceiver;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setStoreExactCycles(boolean storeExactCycles) {
      this.storeExactCycles = storeExactCycles;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setDetectCycles(boolean detectCycles) {
      this.detectCycles = detectCycles;
      return this;
    }

    public EvaluationContext build() {
      return new EvaluationContext(
          parallelism,
          executor,
          keepGoing,
          eventHandler,
          isExecutionPhase,
          mergingSkyframeAnalysisExecutionPhases,
          storeExactCycles,
          unnecessaryTemporaryStateDropperReceiver,
          detectCycles);
    }
  }
}
