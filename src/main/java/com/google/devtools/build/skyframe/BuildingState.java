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

import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Collections;
import java.util.List;

/**
 * Data the NodeEntry uses to maintain its state before it is done building. It allows the {@link
 * NodeEntry} to keep the current state of the entry across invalidation and successive evaluations.
 * A done node does not contain any of this data. However, if a node is marked dirty, its entry
 * acquires a new {@link DirtyBuildingState} object, which persists until it is done again.
 *
 * <p>This class should be considered a private inner class of {@link InMemoryNodeEntry} -- no other
 * classes should instantiate a {@code BuildingState} object or call any of its methods directly. It
 * is in a separate file solely to keep the {@link NodeEntry} class readable. In particular, the
 * caller must synchronize access to this class.
 *
 * <p>During its life, a node can go through states as follows:
 *
 * <ol>
 * <li>Non-existent
 * <li>Just created ({@link #isEvaluating} is false)
 * <li>Evaluating ({@link #isEvaluating} is true)
 * <li>Done (meaning this buildingState object is null)
 * <li>Just created (when it is dirtied during evaluation)
 * <li>Reset (just before it is re-evaluated)
 * <li>Evaluating
 * <li>Done
 * </ol>
 *
 * <p>The "just created" state is there to allow the {@link EvaluableGraph#createIfAbsentBatch} and
 * {@link NodeEntry#addReverseDepAndCheckIfDone} methods to be separate. All callers have to call
 * both methods in that order if they want to create a node. The second method calls {@link
 * #startEvaluating}, which transitions the current node to the "evaluating" state and returns true
 * only the first time it was called. A caller that gets "true" back from that call must start the
 * evaluation of this node, while any subsequent callers must not.
 *
 * <p>An entry is set to "evaluating" as soon as it is scheduled for evaluation. Thus, even a node
 * that is never actually built (for instance, a dirty node that is verified as clean) is in the
 * "evaluating" state until it is done.
 */
@ThreadCompatible
class BuildingState {
  /**
   * The number of dependencies that are known to be done in a {@link NodeEntry} if it is already
   * evaluating, and a sentinel (-1) indicating that it has not yet started evaluating otherwise.
   * There is a potential check-then-act race here during evaluation, so we need to make sure that
   * when this is increased, we always check if the new value is equal to the number of required
   * dependencies, and if so, we must re-schedule the node for evaluation.
   *
   * <p>There are two potential pitfalls here: 1) If multiple dependencies signal this node in close
   * succession, this node should be scheduled exactly once. 2) If a thread is still working on this
   * node, it should not be scheduled.
   *
   * <p>The first problem is solved by the {@link #signalDep} method, which also returns if the node
   * needs to be re-scheduled, and ensures that only one thread gets a true return value.
   *
   * <p>The second problem is solved by first adding the newly discovered deps to a node's {@link
   * InMemoryNodeEntry#directDeps}, and then looping through the direct deps and registering this
   * node as a reverse dependency. This ensures that the signaledDeps counter can only reach {@link
   * InMemoryNodeEntry#directDeps#numElements} on the very last iteration of the loop, i.e., the
   * thread is not working on the node anymore. Note that this requires that there is no code after
   * the loop in {@code ParallelEvaluator.Evaluate#run}.
   */
  int signaledDeps = -1;

  /**
   * The set of reverse dependencies that are registered before the node has finished building. Upon
   * building, these reverse deps will be signaled and then stored in the permanent {@link
   * InMemoryNodeEntry#reverseDeps}.
   */
  protected Object reverseDepsToSignal = ImmutableList.of();
  private List<Object> reverseDepsDataToConsolidate = null;

  private static final ReverseDepsUtil<BuildingState> REVERSE_DEPS_UTIL =
      new ReverseDepsUtilImpl<BuildingState>() {
        @Override
        void setReverseDepsObject(BuildingState container, Object object) {
          container.reverseDepsToSignal = object;
        }

        @Override
        void setDataToConsolidate(BuildingState container, List<Object> dataToConsolidate) {
          container.reverseDepsDataToConsolidate = dataToConsolidate;
        }

        @Override
        Object getReverseDepsObject(BuildingState container) {
          return container.reverseDepsToSignal;
        }

        @Override
        List<Object> getDataToConsolidate(BuildingState container) {
          return container.reverseDepsDataToConsolidate;
        }

        @Override
        public void consolidateReverseDeps(BuildingState container) {
          // #consolidateReverseDeps is only supported for node entries, not building states.
          throw new UnsupportedOperationException();
        }
      };

  /** Returns whether all known children of this node have signaled that they are done. */
  final boolean isReady(int numDirectDeps) {
    Preconditions.checkState(signaledDeps <= numDirectDeps, "%s %s", numDirectDeps, this);
    return signaledDeps == numDirectDeps;
  }

  /**
   * Returns true if the entry is marked dirty, meaning that at least one of its transitive
   * dependencies is marked changed.
   *
   * @see NodeEntry#isDirty()
   */
  boolean isDirty() {
    return false;
  }

  /**
   * Returns true if the entry is known to require re-evaluation.
   *
   * @see NodeEntry#isChanged()
   */
  boolean isChanged() {
    return false;
  }

  /**
   * Helper method to assert that node has finished building, as far as we can tell. We would
   * actually like to check that the node has been evaluated, but that is not available in this
   * context.
   */
  protected void checkFinishedBuildingWhenAboutToSetValue() {
    Preconditions.checkState(isEvaluating(), "not started building %s", this);
    Preconditions.checkState(!isDirty(), "not done building %s", this);
  }

  /**
   * Puts the node in the "evaluating" state if it is not already in it. Returns true if the node
   * wasn't already evaluating and false otherwise. Should only be called by {@link
   * NodeEntry#addReverseDepAndCheckIfDone}.
   */
  final boolean startEvaluating() {
    boolean result = !isEvaluating();
    if (result) {
      signaledDeps = 0;
    }
    return result;
  }

  final boolean isEvaluating() {
    return signaledDeps > -1;
  }

  /**
   * Increments the number of children known to be finished. Returns true if the number of children
   * finished is equal to the number of known children.
   *
   * <p>If the node is dirty and checking its deps for changes, this also updates dirty state as
   * needed, via {@link #signalDepInternal}.
   *
   * @see NodeEntry#signalDep(Version)
   */
  final boolean signalDep(boolean childChanged, int numDirectDeps) {
    Preconditions.checkState(isEvaluating(), this);
    signaledDeps++;
    signalDepInternal(childChanged, numDirectDeps);
    return isReady(numDirectDeps);
  }

  void signalDepInternal(boolean childChanged, int numDirectDeps) {}

  /**
   * Returns reverse deps to signal that have been registered this build.
   *
   * @see NodeEntry#getReverseDeps()
   */
  final ImmutableSet<SkyKey> getReverseDepsToSignal() {
    return REVERSE_DEPS_UTIL.getReverseDeps(this);
  }

  /**
   * Adds a reverse dependency that should be notified when this entry is done.
   *
   * @see NodeEntry#addReverseDepAndCheckIfDone(SkyKey)
   */
  final void addReverseDepToSignal(SkyKey newReverseDep) {
    REVERSE_DEPS_UTIL.addReverseDeps(this, Collections.singleton(newReverseDep));
  }

  /** @see NodeEntry#removeReverseDep(SkyKey) */
  final void removeReverseDepToSignal(SkyKey reverseDep) {
    REVERSE_DEPS_UTIL.removeReverseDep(this, reverseDep);
  }

  protected ToStringHelper getStringHelper() {
    return MoreObjects.toStringHelper(this)
        .add("hash", System.identityHashCode(this))
        .add("signaledDeps/evaluating state", signaledDeps)
        .add("reverseDepsToSignal", REVERSE_DEPS_UTIL.toString(this));
  }
  @Override
  public final String toString() {
    return getStringHelper().toString();
  }
}
