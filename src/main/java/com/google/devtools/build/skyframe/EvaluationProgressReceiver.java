// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Receiver for various stages of the lifetime of a skyframe node evaluation. */
@ThreadSafety.ThreadSafe
public interface EvaluationProgressReceiver {
  /**
   * New state of the value entry after evaluation.
   */
  enum EvaluationState {
    /** The value was successfully re-evaluated. */
    BUILT,
    /** The value is clean or re-validated. */
    CLEAN,
  }

  /** Whether or not evaluation of this node succeeded. */
  enum EvaluationSuccessState {
    SUCCESS(true),
    FAILURE(false);

    EvaluationSuccessState(boolean succeeded) {
      this.succeeded = succeeded;
    }

    private final boolean succeeded;

    public boolean succeeded() {
      return succeeded;
    }

    public Supplier<EvaluationSuccessState> supplier() {
      return () -> this;
    }
  }

  /**
   * New state of the value entry after invalidation.
   */
  enum InvalidationState {
    /** The value is dirty, although it might get re-validated again. */
    DIRTY,
    /** The value is dirty and got deleted, cannot get re-validated again. */
    DELETED,
  }

  /**
   * Overall state of the node while it is being evaluated.
   */
  enum NodeState {
    /** The node is undergoing a dirtiness check and may be re-validated. */
    CHECK_DIRTY,
    /** The node is prepping for evaluation. */
    INITIALIZING_ENVIRONMENT,
    /** The node is in compute(). */
    COMPUTE,
    /** The node is done evaluation and committing the result. */
    COMMIT,
  }

  /**
   * Notifies that the node named by {@code key} has been invalidated.
   *
   * <p>{@code state} indicates the new state of the value.
   *
   * <p>May be called concurrently from multiple threads.
   *
   * <p>If {@code state} is {@link InvalidationState#DIRTY}, should only be called after a
   * successful {@link ThinNodeEntry#markDirty} call: a call that returns a non-null value.
   */
  void invalidated(SkyKey skyKey, InvalidationState state);

  /**
   * Notifies that {@code skyKey} is about to get queued for evaluation.
   *
   * <p>Note that we don't guarantee that it actually got enqueued or will, only that if
   * everything "goes well" (e.g. no interrupts happen) it will.
   *
   * <p>This guarantee is intentionally vague to encourage writing robust implementations.
   */
  void enqueueing(SkyKey skyKey);

  /**
   * Notifies that a node corresponding to {@code skyKey} is about to enter the given
   * {@code nodeState}.
   *
   * <p>Notably, this includes {@link SkyFunction#compute} calls due to Skyframe restarts, but also
   * dirtiness checking and node completion.
   */
  void stateStarting(SkyKey skyKey, NodeState nodeState);

  /**
   * Notifies that a node corresponding to {@code skyKey} is about to complete the given
   * {@code nodeState}.
   *
   * <p>Always called symmetrically with {@link #stateStarting(SkyKey, NodeState)}}.
   *
   * <p>{@code elapsedTimeNanos} is either the elapsed time in the {@code nodeState} or -1 if the
   * timing was not recorded.
   */
  void stateEnding(SkyKey skyKey, NodeState nodeState, long elapsedTimeNanos);

  /**
   * Notifies that the node for {@code skyKey} has been evaluated.
   *
   * <p>{@code state} indicates the new state of the node.
   *
   * <p>If the value builder threw an error when building this node, then {@code
   * valueSupplier.get()} evaluates to null.
   *
   * @param value The sky value. Only available if just evaluated, eg. on success *and* <code>
   *     state == EvalutionState.BUILT</code>
   */
  void evaluated(
      SkyKey skyKey,
      @Nullable SkyValue value,
      Supplier<EvaluationSuccessState> evaluationSuccessState,
      EvaluationState state);

  /** An {@link EvaluationProgressReceiver} that does nothing. */
  class NullEvaluationProgressReceiver implements EvaluationProgressReceiver {
    @Override
    public void invalidated(SkyKey skyKey, InvalidationState state) {
    }

    @Override
    public void enqueueing(SkyKey skyKey) {
    }

    @Override
    public void stateStarting(SkyKey skyKey, NodeState nodeState) {
    }

    @Override
    public void stateEnding(SkyKey skyKey, NodeState nodeState, long elapsedTimeNanos) {
    }

    @Override
    public void evaluated(
        SkyKey skyKey,
        @Nullable SkyValue value,
        Supplier<EvaluationSuccessState> evaluationSuccessState,
        EvaluationState state) {}
  }
}
