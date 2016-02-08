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
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Data the NodeEntry uses to maintain its state before it is done building. It allows the
 * {@link NodeEntry} to keep the current state of the entry across invalidation and successive
 * evaluations. A done node does not contain any of this data. However, if a node is marked dirty,
 * its entry acquires a new {@code BuildingState} object, which persists until it is done again.
 *
 * <p>This class should be considered a private inner class of {@link NodeEntry} -- no other
 * classes should instantiate a {@code BuildingState} object or call any of its methods directly.
 * It is in a separate file solely to keep the {@link NodeEntry} class readable. In particular, the
 * caller must synchronize access to this class.
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
@ThreadCompatible
public class BuildingState {
  /**
   * During its life, a node can go through states as follows:
   * <ol>
   * <li>Non-existent
   * <li>Just created ({@code evaluating} is false)
   * <li>Evaluating ({@code evaluating} is true)
   * <li>Done (meaning this buildingState object is null)
   * <li>Just created (when it is dirtied during evaluation)
   * <li>Reset (just before it is re-evaluated)
   * <li>Evaluating
   * <li>Done
   * </ol>
   *
   * <p>The "just created" state is there to allow the {@link EvaluableGraph#createIfAbsent} and
   * {@link NodeEntry#addReverseDepAndCheckIfDone} methods to be separate. All callers have to
   * call both methods in that order if they want to create a node. The second method calls
   * {@link #startEvaluating}, which transitions the current node to the "evaluating" state and
   * returns true only the first time it was called. A caller that gets "true" back from that call
   * must start the evaluation of this node, while any subsequent callers must not.
   *
   * <p>An entry is set to "evaluating" as soon as it is scheduled for evaluation. Thus, even a
   * node that is never actually built (for instance, a dirty node that is verified as clean) is
   * in the "evaluating" state until it is done.
   */
  private boolean evaluating = false;

  /**
   * The state of a dirty node. A node is marked dirty in the BuildingState constructor, and goes
   * into either the state {@link DirtyState#CHECK_DEPENDENCIES} or
   * {@link DirtyState#NEEDS_REBUILDING}, depending on whether the caller specified that the node
   * was itself changed or not. A non-null {@code dirtyState} indicates that the node
   * {@link #isDirty} in some way.
   */
  private DirtyState dirtyState = null;

  /**
   * The number of dependencies that are known to be done in a {@link NodeEntry}. There is a
   * potential check-then-act race here, so we need to make sure that when this is increased, we
   * always check if the new value is equal to the number of required dependencies, and if so, we
   * must re-schedule the node for evaluation.
   *
   * <p>There are two potential pitfalls here: 1) If multiple dependencies signal this node in
   * close succession, this node should be scheduled exactly once. 2) If a thread is still working
   * on this node, it should not be scheduled.
   *
   * <p>The first problem is solved by the {@link #signalDep} method, which also returns if the
   * node needs to be re-scheduled, and ensures that only one thread gets a true return value.
   *
   * <p>The second problem is solved by first adding the newly discovered deps to a node's
   * {@link #directDeps}, and then looping through the direct deps and registering this node as a
   * reverse dependency. This ensures that the signaledDeps counter can only reach
   * {@link #directDeps}.size() on the very last iteration of the loop, i.e., the thread is not
   * working on the node anymore. Note that this requires that there is no code after the loop in
   * {@code ParallelEvaluator.Evaluate#run}.
   */
  private int signaledDeps = 0;

  /**
   * Direct dependencies discovered during the build. They will be written to the immutable field
   * {@code ValueEntry#directDeps} and the dependency group data to {@code ValueEntry#groupData}
   * once the node is finished building. {@link SkyFunction}s can request deps in groups, and these
   * groupings are preserved in this field.
   */
  private final GroupedList<SkyKey> directDeps = new GroupedList<>();

  /**
   * The set of reverse dependencies that are registered before the node has finished building.
   * Upon building, these reverse deps will be signaled and then stored in the permanent
   * {@code ValueEntry#reverseDeps}.
   */
  // TODO(bazel-team): Remove this field. With eager invalidation, all direct deps on this dirty
  // node will be removed by the time evaluation starts, so reverse deps to signal can just be
  // reverse deps in the main ValueEntry object.
  private Object reverseDepsToSignal = ImmutableList.of();
  private List<Object> reverseDepsDataToConsolidate = null;
  private boolean reverseDepIsSingleObject = false;

  private static final ReverseDepsUtil<BuildingState> REVERSE_DEPS_UTIL =
      new ReverseDepsUtilImpl<BuildingState>() {
        @Override
        void setReverseDepsObject(BuildingState container, Object object) {
          container.reverseDepsToSignal = object;
        }

        @Override
        void setSingleReverseDep(BuildingState container, boolean singleObject) {
          container.reverseDepIsSingleObject = singleObject;
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
        boolean isSingleReverseDep(BuildingState container) {
          return container.reverseDepIsSingleObject;
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

  // Below are fields that are used for dirty nodes.

  /**
   * The dependencies requested (with group markers) last time the node was built (and below, the
   * value last time the node was built). They will be compared to dependencies requested on this
   * build to check whether this node has changed in {@link NodeEntry#setValue}. If they are null,
   * it means that this node is being built for the first time. See {@link #directDeps} for more on
   * dependency group storage.
   */
  private final GroupedList<SkyKey> lastBuildDirectDeps;

  /**
   * Which child should be re-evaluated next in the process of determining if this entry needs to
   * be re-evaluated. Used by {@link #getNextDirtyDirectDeps} and {@link #signalDep(boolean)}.
   */
  private Iterator<Collection<SkyKey>> dirtyDirectDepIterator = null;

  BuildingState() {
    lastBuildDirectDeps = null;
  }

  protected BuildingState(boolean isChanged, GroupedList<SkyKey> lastBuildDirectDeps) {
    this.lastBuildDirectDeps = lastBuildDirectDeps;
    Preconditions.checkState(isChanged || !this.lastBuildDirectDeps.isEmpty(),
        "%s is being marked dirty, not changed, but has no children that could have dirtied it",
        this);
    dirtyState = isChanged ? DirtyState.NEEDS_REBUILDING : DirtyState.CHECK_DEPENDENCIES;
    // We need to iterate through the deps to see if they have changed, or to remove them if one
    // has. Initialize the iterator.
    dirtyDirectDepIterator = lastBuildDirectDeps.iterator();
  }

  static BuildingState newDirtyState(boolean isChanged,
      GroupedList<SkyKey> lastBuildDirectDeps, SkyValue lastBuildValue) {
    return new DirtyBuildingState(isChanged, lastBuildDirectDeps, lastBuildValue);
  }

  void markChanged() {
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(!isChanged(), this);
    Preconditions.checkState(!evaluating, this);
    dirtyState = DirtyState.NEEDS_REBUILDING;
  }

  void forceChanged() {
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(!isChanged(), this);
    Preconditions.checkState(evaluating, this);
    Preconditions.checkState(isReady(), this);
    Preconditions.checkState(!dirtyDirectDepIterator.hasNext(), this);
    dirtyState = DirtyState.REBUILDING;
  }

  /**
   * Returns whether all known children of this node have signaled that they are done.
   */
  boolean isReady() {
    int directDepsSize = directDeps.size();
    Preconditions.checkState(signaledDeps <= directDepsSize, "%s %s", directDeps, this);
    return signaledDeps == directDepsSize;
  }

  /**
   * Returns true if the entry is marked dirty, meaning that at least one of its transitive
   * dependencies is marked changed.
   *
   * @see NodeEntry#isDirty()
   */
  boolean isDirty() {
    return dirtyState != null;
  }

  /**
   * Returns true if the entry is known to require re-evaluation.
   *
   * @see NodeEntry#isChanged()
   */
  boolean isChanged() {
    return dirtyState == DirtyState.NEEDS_REBUILDING || dirtyState == DirtyState.REBUILDING;
  }

  /**
   * Helper method to assert that node has finished building, as far as we can tell. We would
   * actually like to check that the node has been evaluated, but that is not available in
   * this context.
   */
  private void checkFinishedBuildingWhenAboutToSetValue() {
    Preconditions.checkState(evaluating, "not started building %s", this);
    Preconditions.checkState(
        !isDirty()
            || dirtyState == DirtyState.VERIFIED_CLEAN
            || dirtyState == DirtyState.REBUILDING,
        "not done building %s",
        this);
    Preconditions.checkState(isReady(), "not done building %s", this);
  }

  /**
   * Puts the node in the "evaluating" state if it is not already in it. Returns true if the
   * node wasn't already evaluating and false otherwise. Should only be called by
   * {@link NodeEntry#addReverseDepAndCheckIfDone}.
   */
  boolean startEvaluating() {
    boolean result = !evaluating;
    evaluating = true;
    return result;
  }

  /**
   * Increments the number of children known to be finished. Returns true if the number of children
   * finished is equal to the number of known children.
   *
   * <p>If the node is dirty and checking its deps for changes, this also updates {@link
   * #dirtyState} as needed -- {@link DirtyState#NEEDS_REBUILDING} if the child has changed,
   * and {@link DirtyState#VERIFIED_CLEAN} if the child has not changed and this was the
   * last child to be checked (as determined by {@link #dirtyDirectDepIterator}.hasNext() and
   * isReady()).
   *
   * @see NodeEntry#signalDep(Version)
   */
  boolean signalDep(boolean childChanged) {
    signaledDeps++;
    if (isDirty() && !isChanged()) {
      // Synchronization isn't needed here because the only caller is NodeEntry, which does it
      // through the synchronized method signalDep(Version).
      if (childChanged) {
        dirtyState = DirtyState.NEEDS_REBUILDING;
      } else if (dirtyState == DirtyState.CHECK_DEPENDENCIES
          && isReady()
          && !dirtyDirectDepIterator.hasNext()) {
        // No other dep already marked this as NEEDS_REBUILDING, no deps outstanding, and this was
        // the last block of deps to be checked.
        dirtyState = DirtyState.VERIFIED_CLEAN;
      }
    }
    return isReady();
  }

  /**
   * Returns true if {@code newValue}.equals the value from the last time this node was built.
   * Should only be used by {@link NodeEntry#setValue}.
   *
   * <p>Changes in direct deps do <i>not</i> force this to return false. Only the value is
   * considered.
   */
  boolean unchangedFromLastBuild(SkyValue newValue) {
    checkFinishedBuildingWhenAboutToSetValue();
    if (newValue instanceof NotComparableSkyValue) {
      return false;
    }
    return getLastBuildValue().equals(newValue);
  }

  /**
   * Returns true if the deps requested during this evaluation are exactly those requested the
   * last time this node was built, in the same order.
   */
  boolean depsUnchangedFromLastBuild() {
    checkFinishedBuildingWhenAboutToSetValue();
    return lastBuildDirectDeps.equals(directDeps);
  }

  boolean noDepsLastBuild() {
    return lastBuildDirectDeps.isEmpty();
  }

  protected SkyValue getLastBuildValue() {
    // Default BuildingState isn't dirty.
    throw new UnsupportedOperationException(this.toString());
  }

  /**
   * Gets the current state of checking this dirty entry to see if it must be re-evaluated. Must be
   * called each time evaluation of a dirty entry starts to find the proper action to perform next,
   * as enumerated by {@link DirtyState}.
   *
   * @see NodeEntry#getDirtyState()
   */
  DirtyState getDirtyState() {
    // Entry may not be ready if being built just for its errors.
    Preconditions.checkState(isDirty(), "must be dirty to get dirty state %s", this);
    Preconditions.checkState(evaluating, "must be evaluating to get dirty state %s", this);
    return dirtyState;
  }

  /**
   * Gets the next children to be re-evaluated to see if this dirty node needs to be re-evaluated.
   *
   * See {@link NodeEntry#getNextDirtyDirectDeps}.
   */
  Collection<SkyKey> getNextDirtyDirectDeps() {
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(dirtyState == DirtyState.CHECK_DEPENDENCIES, this);
    Preconditions.checkState(evaluating, this);
    Preconditions.checkState(dirtyDirectDepIterator.hasNext(), this);
    return dirtyDirectDepIterator.next();
  }

  Collection<SkyKey> getAllRemainingDirtyDirectDeps() {
    Preconditions.checkState(isDirty(), this);
    ImmutableList.Builder<SkyKey> result = ImmutableList.builder();
    while (dirtyDirectDepIterator.hasNext()) {
      result.addAll(dirtyDirectDepIterator.next());
    }
    return result.build();
  }

  protected Collection<SkyKey> markRebuildingAndGetAllRemainingDirtyDirectDeps() {
    Preconditions.checkState(dirtyState == DirtyState.NEEDS_REBUILDING, this);
    Collection<SkyKey> result = getAllRemainingDirtyDirectDeps();
    dirtyState = DirtyState.REBUILDING;
    return result;
  }

  void addDirectDeps(GroupedListHelper<SkyKey> depsThisRun) {
    directDeps.append(depsThisRun);
  }

  void addDirectDepsGroup(Collection<SkyKey> group) {
    directDeps.appendGroup(group);
  }

  /**
   * Returns the direct deps found so far on this build. Should only be called before the node has
   * finished building.
   *
   * @see NodeEntry#getTemporaryDirectDeps()
   */
  Set<SkyKey> getDirectDepsForBuild() {
    return directDeps.toSet();
  }

  /**
   * Returns the direct deps (in groups) found on this build. Should only be called when the node
   * is done.
   *
   * @see InMemoryNodeEntry#setStateFinishedAndReturnReverseDeps
   */
  GroupedList<SkyKey> getFinishedDirectDeps() {
    return directDeps;
  }

  /**
   * Returns reverse deps to signal that have been registered this build.
   *
   * @see NodeEntry#getReverseDeps()
   */
  ImmutableSet<SkyKey> getReverseDepsToSignal() {
    return REVERSE_DEPS_UTIL.getReverseDeps(this);
  }

  /**
   * Adds a reverse dependency that should be notified when this entry is done.
   *
   * @see NodeEntry#addReverseDepAndCheckIfDone(SkyKey)
   */
  void addReverseDepToSignal(SkyKey newReverseDep) {
    REVERSE_DEPS_UTIL.addReverseDeps(this, Collections.singleton(newReverseDep));
  }

  /**
   * @see NodeEntry#removeReverseDep(SkyKey)
   */
  void removeReverseDepToSignal(SkyKey reverseDep) {
    REVERSE_DEPS_UTIL.removeReverseDep(this, reverseDep);
  }

  /**
   * Removes a set of deps from the set of known direct deps. This is complicated by the need
   * to maintain the group data. If we remove a dep that ended a group, then its predecessor's
   * group data must be changed to indicate that it now ends the group.
   *
   * @see NodeEntry#removeUnfinishedDeps
   */
  void removeDirectDeps(Set<SkyKey> unfinishedDeps) {
    directDeps.remove(unfinishedDeps);
  }

  protected ToStringHelper getStringHelper() {
    return MoreObjects.toStringHelper(this)
        .add("hash", System.identityHashCode(this))
        .add("evaluating", evaluating)
        .add("dirtyState", dirtyState)
        .add("signaledDeps", signaledDeps)
        .add("directDeps", directDeps)
        .add("reverseDepsToSignal", REVERSE_DEPS_UTIL.toString(this))
        .add("lastBuildDirectDeps", lastBuildDirectDeps)
        .add("dirtyDirectDepIterator", dirtyDirectDepIterator);
  }
  @Override
  public String toString() {
    return getStringHelper().toString();
  }

  private static class DirtyBuildingState extends BuildingState {
    private final SkyValue lastBuildValue;
    private DirtyBuildingState(boolean isChanged, GroupedList<SkyKey> lastBuildDirectDeps,
        SkyValue lastBuildValue) {
      super(isChanged, lastBuildDirectDeps);
      this.lastBuildValue = Preconditions.checkNotNull(lastBuildValue);
    }

    @Override
    protected SkyValue getLastBuildValue() {
      return lastBuildValue;
    }

    @Override
    public String toString() {
      return getStringHelper().add("lastBuildValue", lastBuildValue).toString();
    }
  }
}
