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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.ThinNodeEntry.DirtyType;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * State for a node that has been dirtied, and will be checked to see if it needs re-evaluation, and
 * either marked clean or re-evaluated.
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
public abstract class DirtyBuildingState {
  private static final int NOT_EVALUATING_SENTINEL = -1;

  static DirtyBuildingState create(
      DirtyType dirtyType, GroupedList<SkyKey> lastBuildDirectDeps, SkyValue lastBuildValue) {
    return new FullDirtyBuildingState(dirtyType, lastBuildDirectDeps, lastBuildValue);
  }

  static DirtyBuildingState createNew() {
    return new FullDirtyBuildingState(DirtyType.CHANGE, null, null);
  }

  /**
   * The state of a dirty node. A node is marked dirty in the DirtyBuildingState constructor, and
   * goes into either the state {@link DirtyState#CHECK_DEPENDENCIES} or {@link
   * DirtyState#NEEDS_REBUILDING}, depending on whether the caller specified that the node was
   * itself changed or not. Never null.
   */
  private DirtyState dirtyState;

  /**
   * The number of dependencies that are known to be done in a {@link NodeEntry}.
   *
   * <p>There is a potential check-then-act race here during evaluation, so we need to make sure
   * that when this is increased, we always check if the new value is equal to the number of
   * required dependencies, and if so, we must re-schedule the node for evaluation.
   *
   * <p>There are two potential pitfalls here: 1) If multiple dependencies signal this node in close
   * succession, this node should be scheduled exactly once. 2) If a thread is still working on this
   * node, it should not be scheduled.
   *
   * <p>To solve the first problem, the {@link NodeEntry#signalDep} method also returns if the node
   * needs to be re-scheduled, and ensures that only one thread gets a true return value.
   *
   * <p>The second problem is solved by first adding the newly discovered deps to a node's {@link
   * InMemoryNodeEntry#directDeps}, and then looping through the direct deps and registering this
   * node as a reverse dependency. This ensures that the signaledDeps counter can only reach {@link
   * InMemoryNodeEntry#directDeps#numElements} on the very last iteration of the loop, i.e., the
   * thread is not working on the node anymore. Note that this requires that there is no code after
   * the loop in {@link ParallelEvaluator.Evaluate#run}.
   */
  private int signaledDeps = NOT_EVALUATING_SENTINEL;

  /**
   * The dependencies requested (with group markers) last time the node was built (and below, the
   * value last time the node was built). They will be compared to dependencies requested on this
   * build to check whether this node has changed in {@link NodeEntry#setValue}. If they are null,
   * it means that this node is being built for the first time. See {@link
   * InMemoryNodeEntry#directDeps} for more on dependency group storage.
   *
   * <p>Public only for the use of alternative graph implementations.
   */
  @Nullable
  public abstract GroupedList<SkyKey> getLastBuildDirectDeps() throws InterruptedException;

  /**
   * The number of groups of the dependencies requested last time when the node was built.
   *
   * <p>Getting the number of last-built dependencies should not throw {@link InterruptedException}.
   */
  protected abstract int getNumOfGroupsInLastBuildDirectDeps();

  /** The number of total dependencies requested the last time the node was built. */
  public abstract int getNumElementsInLastBuildDirectDeps();

  /**
   * The value of the node the last time it was built.
   *
   * <p>Public only for the use of alternative graph implementations.
   */
  @Nullable
  public abstract SkyValue getLastBuildValue() throws InterruptedException;

  /**
   * Group of children to be checked next in the process of determining if this entry needs to be
   * re-evaluated. Used by {@link DirtyBuildingState#getNextDirtyDirectDeps} and {@link
   * #signalDepPostProcess}.
   */
  protected int dirtyDirectDepIndex;

  protected DirtyBuildingState(DirtyType dirtyType) {
    dirtyState = dirtyType.getInitialDirtyState();
    // We need to iterate through the deps to see if they have changed, or to remove them if one
    // has. Initialize the iterating index.
    dirtyDirectDepIndex = 0;
  }

  /** Returns true if this state does have information about a previously built version. */
  protected abstract boolean isDirty();

  final void markChanged() {
    Preconditions.checkState(dirtyState == DirtyState.CHECK_DEPENDENCIES, this);
    Preconditions.checkState(dirtyDirectDepIndex == 0, "Unexpected evaluation: %s", this);
    dirtyState = DirtyState.NEEDS_REBUILDING;
  }

  final void markForceRebuild() {
    if (dirtyState == DirtyState.CHECK_DEPENDENCIES || dirtyState == DirtyState.NEEDS_REBUILDING) {
      dirtyState = DirtyState.NEEDS_REBUILDING;
    }
  }

  final void forceRebuild(int numTemporaryDirectDeps) {
    Preconditions.checkState(numTemporaryDirectDeps == signaledDeps, this);
    Preconditions.checkState(
        (dirtyState == DirtyState.CHECK_DEPENDENCIES
                && getNumOfGroupsInLastBuildDirectDeps() == dirtyDirectDepIndex)
            || dirtyState == DirtyState.NEEDS_FORCED_REBUILDING,
        this);
    dirtyState = DirtyState.FORCED_REBUILDING;
  }

  final boolean isEvaluating() {
    return signaledDeps > NOT_EVALUATING_SENTINEL;
  }

  final boolean isChanged() {
    return dirtyState == DirtyState.NEEDS_REBUILDING
        || dirtyState == DirtyState.NEEDS_FORCED_REBUILDING
        || dirtyState == DirtyState.REBUILDING
        || dirtyState == DirtyState.FORCED_REBUILDING;
  }

  private void checkFinishedBuildingWhenAboutToSetValue() {
    Preconditions.checkState(
        dirtyState == DirtyState.VERIFIED_CLEAN
            || dirtyState == DirtyState.REBUILDING
            || dirtyState == DirtyState.FORCED_REBUILDING,
        "not done building %s",
        this);
  }

  final void signalDep() {
    Preconditions.checkState(isEvaluating());
    signaledDeps++;
  }

  /**
   * If this node is not yet known to need rebuilding, sets {@link #dirtyState} to {@link
   * DirtyState#NEEDS_REBUILDING} if the child has changed, and {@link DirtyState#VERIFIED_CLEAN} if
   * the child has not changed and this was the last child to be checked (as determined by {@code
   * isReady} and comparing {@link #dirtyDirectDepIndex} and {@link
   * DirtyBuildingState#getNumOfGroupsInLastBuildDirectDeps()}.
   */
  final void signalDepPostProcess(boolean childChanged, int numTemporaryDirectDeps) {
    Preconditions.checkState(
        isChanged() || (dirtyState == DirtyState.CHECK_DEPENDENCIES && dirtyDirectDepIndex > 0),
        "Unexpected not evaluating: %s",
        this);
    if (!isChanged()) {
      // Synchronization isn't needed here because the only caller is NodeEntry, which does it
      // through the synchronized method signalDep.
      if (childChanged) {
        dirtyState = DirtyState.NEEDS_REBUILDING;
      } else if (dirtyState == DirtyState.CHECK_DEPENDENCIES
          && isReady(numTemporaryDirectDeps)
          && getNumOfGroupsInLastBuildDirectDeps() == dirtyDirectDepIndex) {
        // No other dep already marked this as NEEDS_REBUILDING, no deps outstanding, and this was
        // the last block of deps to be checked.
        dirtyState = DirtyState.VERIFIED_CLEAN;
      }
    }
  }

  public final void unmarkNeedsRebuilding() {
    Preconditions.checkState(dirtyState == DirtyState.NEEDS_REBUILDING, this);
    if (getNumOfGroupsInLastBuildDirectDeps() == dirtyDirectDepIndex) {
      dirtyState = DirtyState.VERIFIED_CLEAN;
    } else {
      dirtyState = DirtyState.CHECK_DEPENDENCIES;
    }
  }

  /**
   * Returns true if {@code newValue}.equals the value from the last time this node was built.
   * Should only be used by {@link NodeEntry#setValue}.
   *
   * <p>Changes in direct deps do <i>not</i> force this to return false. Only the value is
   * considered.
   */
  public final boolean unchangedFromLastBuild(SkyValue newValue) throws InterruptedException {
    checkFinishedBuildingWhenAboutToSetValue();
    return !(newValue instanceof NotComparableSkyValue)
        && getLastBuildValue() != null
        && getLastBuildValue().equals(newValue);
  }

  /**
   * Returns true if the deps requested during this evaluation ({@code directDeps}) are exactly
   * those requested the last time this node was built, in the same order.
   */
  final boolean depsUnchangedFromLastBuild(GroupedList<SkyKey> directDeps)
      throws InterruptedException {
    checkFinishedBuildingWhenAboutToSetValue();
    return getLastBuildDirectDeps().equals(directDeps);
  }

  final boolean noDepsLastBuild() {
    return getNumOfGroupsInLastBuildDirectDeps() == 0;
  }

  /** @see NodeEntry#getDirtyState() */
  final DirtyState getDirtyState() {
    return dirtyState;
  }

  /**
   * Gets the next children to be re-evaluated to see if this dirty node needs to be re-evaluated.
   *
   * <p>See {@link NodeEntry#getNextDirtyDirectDeps}.
   */
  final List<SkyKey> getNextDirtyDirectDeps() throws InterruptedException {
    Preconditions.checkState(dirtyState == DirtyState.CHECK_DEPENDENCIES, this);
    Preconditions.checkState(dirtyDirectDepIndex < getNumOfGroupsInLastBuildDirectDeps(), this);
    return getLastBuildDirectDeps().get(dirtyDirectDepIndex++);
  }

  /**
   * Returns the remaining direct deps that have not been checked. If {@code preservePosition} is
   * true, this method is non-mutating. If {@code preservePosition} is false, the caller must
   * process the returned set, and so subsequent calls to this method will return the empty set.
   */
  Set<SkyKey> getAllRemainingDirtyDirectDeps(boolean preservePosition) throws InterruptedException {
    if (getLastBuildDirectDeps() == null) {
      return ImmutableSet.of();
    }
    ImmutableSet.Builder<SkyKey> result = ImmutableSet.builder();
    for (int ind = dirtyDirectDepIndex; ind < getNumOfGroupsInLastBuildDirectDeps(); ind++) {
      result.addAll(getLastBuildDirectDeps().get(ind));
    }
    if (!preservePosition) {
      dirtyDirectDepIndex = getNumOfGroupsInLastBuildDirectDeps();
    }
    return result.build();
  }

  final void resetForRestartFromScratch() {
    Preconditions.checkState(
        dirtyState == DirtyState.REBUILDING || dirtyState == DirtyState.FORCED_REBUILDING, this);
    signaledDeps = 0;
    dirtyDirectDepIndex = 0;
  }

  protected void markRebuilding(boolean isEligibleForChangePruning) {
    Preconditions.checkState(dirtyState == DirtyState.NEEDS_REBUILDING, this);
    dirtyState = DirtyState.REBUILDING;
  }

  void startEvaluating() {
    Preconditions.checkState(!isEvaluating(), this);
    signaledDeps = 0;
  }

  public int getLastDirtyDirectDepIndex() {
    return dirtyDirectDepIndex - 1;
  }

  public int getSignaledDeps() {
    return signaledDeps;
  }

  /** Returns whether all known children of this node have signaled that they are done. */
  boolean isReady(int numDirectDeps) {
    Preconditions.checkState(signaledDeps <= numDirectDeps, "%s %s", numDirectDeps, this);
    return signaledDeps == numDirectDeps;
  }

  protected MoreObjects.ToStringHelper getStringHelper() {
    return MoreObjects.toStringHelper(this)
        .add("dirtyState", dirtyState)
        .add("signaledDeps", signaledDeps)
        .add("dirtyDirectDepIndex", dirtyDirectDepIndex);
  }

  @Override
  public String toString() {
    return getStringHelper().toString();
  }

  private static class FullDirtyBuildingState extends DirtyBuildingState {
    private final GroupedList<SkyKey> lastBuildDirectDeps;
    private final SkyValue lastBuildValue;

    private FullDirtyBuildingState(
        DirtyType dirtyType, GroupedList<SkyKey> lastBuildDirectDeps, SkyValue lastBuildValue) {
      super(dirtyType);
      this.lastBuildDirectDeps = lastBuildDirectDeps;
      Preconditions.checkState(
          !dirtyType.equals(DirtyType.DIRTY) || getNumOfGroupsInLastBuildDirectDeps() > 0,
          "%s is being marked dirty but has no children that could have dirtied it",
          this);
      this.lastBuildValue = lastBuildValue;
    }

    @Override
    protected boolean isDirty() {
      return lastBuildDirectDeps != null;
    }

    @Override
    public SkyValue getLastBuildValue() {
      return lastBuildValue;
    }

    @Override
    public GroupedList<SkyKey> getLastBuildDirectDeps() throws InterruptedException {
      return lastBuildDirectDeps;
    }

    @Override
    protected int getNumOfGroupsInLastBuildDirectDeps() {
      return lastBuildDirectDeps == null ? 0 : lastBuildDirectDeps.listSize();
    }

    @Override
    public int getNumElementsInLastBuildDirectDeps() {
      return lastBuildDirectDeps.numElements();
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper()
          .add("lastBuildDirectDeps", lastBuildDirectDeps)
          .add("lastBuildValue", lastBuildValue);
    }
  }
}
