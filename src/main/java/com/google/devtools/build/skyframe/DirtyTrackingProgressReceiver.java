// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import java.util.Set;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A delegating {@link EvaluationProgressReceiver} that tracks inflight nodes, nodes which
 * are being evaluated or scheduled for evaluation, and dirty nodes.
 */
public class DirtyTrackingProgressReceiver implements EvaluationProgressReceiver {

  @Nullable protected final EvaluationProgressReceiver progressReceiver;
  private final Set<SkyKey> dirtyKeys = Sets.newConcurrentHashSet();
  private Set<SkyKey> inflightKeys = Sets.newConcurrentHashSet();

  public DirtyTrackingProgressReceiver(@Nullable EvaluationProgressReceiver progressReceiver) {
    this.progressReceiver = progressReceiver;
  }

  /** Called when a node is injected into the graph, and not evaluated. */
  protected void injected(SkyKey skyKey) {
    // This node was never evaluated, but is now clean and need not be re-evaluated
    inflightKeys.remove(skyKey);
    removeFromDirtySet(skyKey);
  }

  @Override
  public void invalidated(SkyKey skyKey, InvalidationState state) {
    if (progressReceiver != null) {
      progressReceiver.invalidated(skyKey, state);
    }

    switch (state) {
      case DELETED:
        // This key was removed from the graph, so no longer needs to be marked as dirty.
        removeFromDirtySet(skyKey);
        break;
      case DIRTY:
        addToDirtySet(skyKey);
        break;
      default:
        throw new IllegalStateException(state.toString());
    }
  }

  @Override
  public void enqueueing(SkyKey skyKey) {
    enqueueing(skyKey, false);
  }

  private void enqueueing(SkyKey skyKey, boolean afterError) {
    // We unconditionally add the key to the set of in-flight nodes even if evaluation is never
    // scheduled, because we still want to remove the previously created NodeEntry from the graph.
    // Otherwise we would leave the graph in a weird state (wasteful garbage in the best case and
    // inconsistent in the worst case).
    boolean newlyEnqueued = inflightKeys.add(skyKey);
    if (newlyEnqueued) {
      // All nodes enqueued for evaluation will be either verified clean, re-evaluated, or cleaned
      // up after being in-flight when an error happens in nokeep_going mode or in the event of an
      // interrupt. In any of these cases, they won't be dirty anymore.
      removeFromDirtySet(skyKey);
      if (progressReceiver != null && !afterError) {
        // Only tell the external listener the node was enqueued if no there was neither an error
        // or interrupt.
        progressReceiver.enqueueing(skyKey);
      }
    }
  }

  /**
   * Called when a node was requested to be enqueued but wasn't because either an interrupt or an
   * error (in nokeep_going mode) had occurred.
   */
  protected void enqueueAfterError(SkyKey skyKey) {
    enqueueing(skyKey, true);
  }

  @Override
  public void stateStarting(SkyKey skyKey, NodeState nodeState) {
    if (progressReceiver != null) {
      progressReceiver.stateStarting(skyKey, nodeState);
    }
  }

  @Override
  public void stateEnding(SkyKey skyKey, NodeState nodeState) {
    if (progressReceiver != null) {
      progressReceiver.stateEnding(skyKey, nodeState);
    }
  }

  @Override
  public void evaluated(
      SkyKey skyKey,
      @Nullable SkyValue newValue,
      @Nullable ErrorInfo newError,
      Supplier<EvaluationSuccessState> evaluationSuccessState,
      EvaluationState state) {
    if (progressReceiver != null) {
      progressReceiver.evaluated(skyKey, newValue, newError, evaluationSuccessState, state);
    }

    // This key was either built or marked clean, so we can remove it from both the dirty and
    // inflight nodes.
    inflightKeys.remove(skyKey);
    removeFromDirtySet(skyKey);
  }

  /** Returns if the key is enqueued for evaluation. */
  protected boolean isInflight(SkyKey skyKey) {
    return inflightKeys.contains(skyKey);
  }

  /** Returns the set of all keys that are enqueued for evaluation, and resets the set to empty. */
  public Set<SkyKey> getAndClearInflightKeys() {
    Set<SkyKey> keys = inflightKeys;
    inflightKeys = Sets.newConcurrentHashSet();
    return keys;
  }

  /**
   * Returns the set of all dirty keys that have not been enqueued.
   * This is useful for garbage collection, where we would not want to remove dirty nodes that are
   * needed for evaluation (in the downward transitive closure of the set of the evaluation's
   * top level nodes).
   */
  protected Set<SkyKey> getUnenqueuedDirtyKeys(){
    return ImmutableSet.copyOf(dirtyKeys);
  }

  protected void addToDirtySet(SkyKey skyKey) {
    dirtyKeys.add(skyKey);
  }

  protected void removeFromDirtySet(SkyKey skyKey) {
    dirtyKeys.remove(skyKey);
  }
}
