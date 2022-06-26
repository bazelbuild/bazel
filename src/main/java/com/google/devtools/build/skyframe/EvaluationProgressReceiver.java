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

  /** A no-op {@link EvaluationProgressReceiver}. */
  EvaluationProgressReceiver NULL = new EvaluationProgressReceiver() {};

  /** New state of the value entry after evaluation. */
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

  /** New state of the value entry after invalidation. */
  enum InvalidationState {
    /** The value is dirty, although it might get re-validated again. */
    DIRTY,
    /** The value is dirty and got deleted, cannot get re-validated again. */
    DELETED,
  }

  /** Overall state of the node while it is being evaluated. */
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
   * Notifies that the node for {@code key} has been invalidated.
   *
   * <p>{@code state} indicates the new state of the value.
   *
   * <p>May be called concurrently from multiple threads.
   *
   * <p>If {@code state} is {@link InvalidationState#DIRTY}, should only be called after a
   * successful {@link ThinNodeEntry#markDirty} call: a call that returns a non-null value.
   */
  default void invalidated(SkyKey skyKey, InvalidationState state) {}

  /**
   * Notifies that {@code skyKey} is about to get queued for evaluation.
   *
   * <p>Note that we don't guarantee that it actually got enqueued or will, only that if everything
   * "goes well" (e.g. no interrupts happen) it will.
   *
   * <p>This guarantee is intentionally vague to encourage writing robust implementations.
   */
  default void enqueueing(SkyKey skyKey) {}

  /**
   * Notifies that the node for {@code skyKey} is about to enter the given {@code nodeState}.
   *
   * <p>Notably, this includes {@link SkyFunction#compute} calls due to Skyframe restarts, but also
   * dirtiness checking and node completion.
   */
  default void stateStarting(SkyKey skyKey, NodeState nodeState) {}

  /**
   * Notifies that the node for {@code skyKey} is about to complete the given {@code nodeState}.
   *
   * <p>Always called symmetrically with {@link #stateStarting(SkyKey, NodeState)}}.
   */
  default void stateEnding(SkyKey skyKey, NodeState nodeState) {}

  /**
   * Notifies that the node for {@code skyKey} has been evaluated, or found to not need
   * re-evaluation.
   *
   * @param newValue The new value. Only available if just evaluated, i.e. on success *and* {@code
   *     state == EvaluationState.BUILT}
   * @param newError The new error. Only available if just evaluated, i.e. on error *and* {@code
   *     state == EvaluationState.BUILT}
   * @param evaluationSuccessState whether the node has a value or only an error, behind a {@link
   *     Supplier} for lazy retrieval. Available regardless of whether the node was just evaluated
   * @param state {@code EvaluationState.BUILT} if the node needed to be evaluated and has a new
   *     value or error (i.e., {@code EvaluationState.BUILT} if and only if at least one of newValue
   *     and newError is non-null)
   */
  default void evaluated(
      SkyKey skyKey,
      @Nullable SkyValue newValue,
      @Nullable ErrorInfo newError,
      Supplier<EvaluationSuccessState> evaluationSuccessState,
      EvaluationState state) {}
}
