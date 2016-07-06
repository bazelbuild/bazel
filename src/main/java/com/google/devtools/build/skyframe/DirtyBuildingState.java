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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;

import java.util.Collection;
import java.util.Set;

/**
 * A {@link BuildingState} for a node that has been dirtied, and will be checked to see if it needs
 * re-evaluation, and either marked clean or re-evaluated. See {@link BuildingState} for more.
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
public abstract class DirtyBuildingState extends BuildingState {
  /**
   * The state of a dirty node. A node is marked dirty in the DirtyBuildingState constructor, and
   * goes into either the state {@link DirtyState#CHECK_DEPENDENCIES} or {@link
   * DirtyState#NEEDS_REBUILDING}, depending on whether the caller specified that the node was
   * itself changed or not. Never null.
   */
  private DirtyState dirtyState;

  /**
   * The dependencies requested (with group markers) last time the node was built (and below, the
   * value last time the node was built). They will be compared to dependencies requested on this
   * build to check whether this node has changed in {@link NodeEntry#setValue}. If they are null,
   * it means that this node is being built for the first time. See {@link
   * InMemoryNodeEntry#directDeps} for more on dependency group storage.
   */
  protected final GroupedList<SkyKey> lastBuildDirectDeps;

  /** The value of the node the last time it was built. */
  protected abstract SkyValue getLastBuildValue();

  /**
   * Group of children to be checked next in the process of determining if this entry needs to be
   * re-evaluated. Used by {@link DirtyBuildingState#getNextDirtyDirectDeps} and {@link #signalDep}.
   */
  private int dirtyDirectDepIndex = -1;

  protected DirtyBuildingState(boolean isChanged, GroupedList<SkyKey> lastBuildDirectDeps) {
    this.lastBuildDirectDeps = lastBuildDirectDeps;
    Preconditions.checkState(
        isChanged || !this.lastBuildDirectDeps.isEmpty(),
        "%s is being marked dirty, not changed, but has no children that could have dirtied it",
        this);
    dirtyState = isChanged ? DirtyState.NEEDS_REBUILDING : DirtyState.CHECK_DEPENDENCIES;
    // We need to iterate through the deps to see if they have changed, or to remove them if one
    // has. Initialize the iterating index.
    dirtyDirectDepIndex = 0;
  }

  static BuildingState create(
      boolean isChanged, GroupedList<SkyKey> lastBuildDirectDeps, SkyValue lastBuildValue) {
    return new DirtyBuildingStateWithValue(isChanged, lastBuildDirectDeps, lastBuildValue);
  }

  final void markChanged() {
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(!isChanged(), this);
    Preconditions.checkState(!isEvaluating(), this);
    dirtyState = DirtyState.NEEDS_REBUILDING;
  }

  final void forceChanged() {
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(!isChanged(), this);
    Preconditions.checkState(isEvaluating(), this);
    Preconditions.checkState(lastBuildDirectDeps.listSize() == dirtyDirectDepIndex, this);
    dirtyState = DirtyState.REBUILDING;
  }

  final int getSignaledDeps() {
    return signaledDeps;
  }

  @Override
  final boolean isDirty() {
    return true;
  }

  @Override
  final boolean isChanged() {
    return dirtyState == DirtyState.NEEDS_REBUILDING || dirtyState == DirtyState.REBUILDING;
  }

  @Override
  protected final void checkFinishedBuildingWhenAboutToSetValue() {
    Preconditions.checkState(isEvaluating(), "not started building %s", this);
    Preconditions.checkState(
        !isDirty()
            || dirtyState == DirtyState.VERIFIED_CLEAN
            || dirtyState == DirtyState.REBUILDING,
        "not done building %s",
        this);
  }

  /**
   * If this node is not yet known to need rebuilding, sets {@link #dirtyState} to {@link
   * DirtyState#NEEDS_REBUILDING} if the child has changed, and {@link DirtyState#VERIFIED_CLEAN} if
   * the child has not changed and this was the last child to be checked (as determined by {@link
   * #isReady} and comparing {@link #dirtyDirectDepIndex} and {@link
   * DirtyBuildingState#lastBuildDirectDeps#listSize}.
   */
  @Override
  final void signalDepInternal(boolean childChanged, int numDirectDeps) {
    if (!isChanged()) {
      // Synchronization isn't needed here because the only caller is NodeEntry, which does it
      // through the synchronized method signalDep(Version).
      if (childChanged) {
        dirtyState = DirtyState.NEEDS_REBUILDING;
      } else if (dirtyState == DirtyState.CHECK_DEPENDENCIES
          && isReady(numDirectDeps)
          && lastBuildDirectDeps.listSize() == dirtyDirectDepIndex) {
        // No other dep already marked this as NEEDS_REBUILDING, no deps outstanding, and this was
        // the last block of deps to be checked.
        dirtyState = DirtyState.VERIFIED_CLEAN;
      }
    }
  }

  /**
   * Returns true if {@code newValue}.equals the value from the last time this node was built.
   * Should only be used by {@link NodeEntry#setValue}.
   *
   * <p>Changes in direct deps do <i>not</i> force this to return false. Only the value is
   * considered.
   */
  final boolean unchangedFromLastBuild(SkyValue newValue) {
    checkFinishedBuildingWhenAboutToSetValue();
    return !(newValue instanceof NotComparableSkyValue) && getLastBuildValue().equals(newValue);
  }

  /**
   * Returns true if the deps requested during this evaluation ({@code directDeps}) are exactly
   * those requested the last time this node was built, in the same order.
   */
  final boolean depsUnchangedFromLastBuild(GroupedList<SkyKey> directDeps) {
    checkFinishedBuildingWhenAboutToSetValue();
    return lastBuildDirectDeps.equals(directDeps);
  }

  final boolean noDepsLastBuild() {
    return lastBuildDirectDeps.isEmpty();
  }

  /**
   * Gets the current state of checking this dirty entry to see if it must be re-evaluated. Must be
   * called each time evaluation of a dirty entry starts to find the proper action to perform next,
   * as enumerated by {@link DirtyState}.
   *
   * @see NodeEntry#getDirtyState()
   */
  final DirtyState getDirtyState() {
    // Entry may not be ready if being built just for its errors.
    Preconditions.checkState(isEvaluating(), "must be evaluating to get dirty state %s", this);
    return dirtyState;
  }

  /**
   * Gets the next children to be re-evaluated to see if this dirty node needs to be re-evaluated.
   *
   * <p>See {@link NodeEntry#getNextDirtyDirectDeps}.
   */
  final Collection<SkyKey> getNextDirtyDirectDeps() {
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(dirtyState == DirtyState.CHECK_DEPENDENCIES, this);
    Preconditions.checkState(isEvaluating(), this);
    Preconditions.checkState(dirtyDirectDepIndex < lastBuildDirectDeps.listSize(), this);
    return lastBuildDirectDeps.get(dirtyDirectDepIndex++);
  }

  /**
   * Returns the remaining direct deps that have not been checked. If {@code preservePosition} is
   * true, this method is non-mutating. If {@code preservePosition} is false, the caller must
   * process the returned set, and so subsequent calls to this method will return the empty set.
   */
  Set<SkyKey> getAllRemainingDirtyDirectDeps(boolean preservePosition) {
    Preconditions.checkState(isDirty(), this);
    ImmutableSet.Builder<SkyKey> result = ImmutableSet.builder();

    for (int ind = dirtyDirectDepIndex; ind < lastBuildDirectDeps.listSize(); ind++) {
      result.addAll(lastBuildDirectDeps.get(ind));
    }
    if (!preservePosition) {
      dirtyDirectDepIndex = lastBuildDirectDeps.listSize();
    }
    return result.build();
  }

  protected void markRebuilding() {
    Preconditions.checkState(dirtyState == DirtyState.NEEDS_REBUILDING, this);
    dirtyState = DirtyState.REBUILDING;
  }

  @Override
  protected MoreObjects.ToStringHelper getStringHelper() {
    return super.getStringHelper()
        .add("dirtyState", dirtyState)
        .add("lastBuildDirectDeps", lastBuildDirectDeps)
        .add("dirtyDirectDepIndex", dirtyDirectDepIndex);
  }

  private static class DirtyBuildingStateWithValue extends DirtyBuildingState {
    private final SkyValue lastBuildValue;

    private DirtyBuildingStateWithValue(
        boolean isChanged, GroupedList<SkyKey> lastBuildDirectDeps, SkyValue lastBuildValue) {
      super(isChanged, lastBuildDirectDeps);
      this.lastBuildValue = lastBuildValue;
    }

    @Override
    protected SkyValue getLastBuildValue() {
      return lastBuildValue;
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper().add("lastBuildValue", lastBuildValue);
    }
  }
}
