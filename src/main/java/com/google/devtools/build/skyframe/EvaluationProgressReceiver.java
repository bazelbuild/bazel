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
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import javax.annotation.Nullable;

/** Receiver for various stages of the lifetime of a skyframe node evaluation. */
@ThreadSafety.ThreadSafe
public interface EvaluationProgressReceiver {

  /** A no-op {@link EvaluationProgressReceiver}. */
  EvaluationProgressReceiver NULL = new EvaluationProgressReceiver() {};

  /** The state of a node after it was evaluated. */
  enum EvaluationState {
    SUCCESS_CHANGED(true, true),
    SUCCESS_UNCHANGED(true, false),
    FAIL_CHANGED(false, true),
    FAIL_UNCHANGED(false, true);

    static EvaluationState get(@Nullable SkyValue valueMaybeWithMetadata, boolean changed) {
      boolean success = ValueWithMetadata.justValue(valueMaybeWithMetadata) != null;
      if (changed) {
        return success ? SUCCESS_CHANGED : FAIL_CHANGED;
      } else {
        return success ? SUCCESS_UNCHANGED : FAIL_UNCHANGED;
      }
    }

    private final boolean succeeded;
    private final boolean changed;

    EvaluationState(boolean succeeded, boolean changed) {
      this.succeeded = succeeded;
      this.changed = changed;
    }

    /**
     * Whether the node has a value.
     *
     * <p>If {@code false}, the node has only an error and no value.
     */
    public boolean succeeded() {
      return succeeded;
    }

    /**
     * Whether the node's {@link NodeEntry#getVersion} changed as a result of this evaluation.
     *
     * <p>If {@code true}, the node was built during the current evaluation and its {@link
     * NodeEntry#getVersion} changed. Parents need to be rebuilt.
     *
     * <p>If {@code false}, the node's {@link NodeEntry#getVersion} did not change, either because
     * it was deemed up-to-date and not built or was built and evaluated to the same value as its
     * prior evaluation. Parents do not necessarily need to be rebuilt.
     */
    public boolean changed() {
      return changed;
    }
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
   * Notifies that the node for {@code skyKey} has been {@linkplain NodeEntry#markDirty marked
   * dirty} with the given {@link DirtyType}.
   *
   * <p>May be called concurrently from multiple threads.
   *
   * <p>Only called after a successful {@link NodeEntry#markDirty} call: a call that returns a
   * non-null value.
   */
  default void dirtied(SkyKey skyKey, DirtyType dirtyType) {}

  /** Notifies that the node for {@code skyKey} was deleted. */
  default void deleted(SkyKey skyKey) {}

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
   * Notifies that the node for {@code skyKey} has been evaluated.
   *
   * @param state the current state of the node for {@code skyKey}
   * @param newValue the node's value if {@link EvaluationState#changed()} and {@link
   *     EvaluationState#succeeded()}, otherwise {@code null}
   * @param newError the node's error if it has one and {@link EvaluationState#changed()}
   * @param directDeps direct dependencies of {@code skyKey} if the node was just built, otherwise
   *     {@code null}
   */
  default void evaluated(
      SkyKey skyKey,
      EvaluationState state,
      @Nullable SkyValue newValue,
      @Nullable ErrorInfo newError,
      @Nullable GroupedDeps directDeps) {}
}
