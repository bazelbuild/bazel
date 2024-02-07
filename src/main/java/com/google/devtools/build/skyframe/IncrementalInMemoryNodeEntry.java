// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.skyframe.KeyToConsolidate.Op;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.ForOverride;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** An {@link InMemoryNodeEntry} that {@link #keepsEdges} for use in incremental evaluations. */
public class IncrementalInMemoryNodeEntry extends AbstractInMemoryNodeEntry<DirtyBuildingState> {

  protected volatile NodeVersion version = Version.minimal();

  /**
   * This object represents the direct deps of the node, in groups if the {@code SkyFunction}
   * requested them that way. It contains either the in-progress direct deps, stored as a {@link
   * GroupedDeps} (constructed via {@link GroupedDeps.WithHashSet} if {@code
   * key.supportsPartialReevaluation()}) before the node is finished building, or the full direct
   * deps, compressed in a memory-efficient way (via {@link GroupedDeps#compress}), after the node
   * is done.
   *
   * <p>It is initialized lazily in getTemporaryDirectDeps() to save a little memory.
   */
  protected Object directDeps = null;

  /**
   * This list stores the reverse dependencies of this node that have been declared so far.
   *
   * <p>In case of a single object we store the object unwrapped, without the list, for
   * memory-efficiency.
   *
   * <p>When an entry is being re-evaluated, this object stores the reverse deps from the previous
   * evaluation. At the end of evaluation, the changed reverse dep operations from {@link
   * #reverseDepsDataToConsolidate} are merged in here.
   */
  protected Object reverseDeps = ImmutableList.of();

  /**
   * This list stores objects returned by {@link KeyToConsolidate#create}. Morally they are {@link
   * KeyToConsolidate} objects, but since some operations are stored bare, we can only declare that
   * this list holds {@link Object} references. Created lazily to save memory.
   *
   * <p>This list serves double duty. For a done node, when a reverse dep is removed, checked for
   * presence, or possibly added, we store the mutation in this object instead of immediately doing
   * the operation. That is because removals/checks in reverseDeps are O(N). Originally reverseDeps
   * was a HashSet, but because of memory consumption we switched to a list.
   *
   * <p>Internally, {@link ReverseDepsUtility} consolidates this data periodically, and when the set
   * of reverse deps is requested. While this operation is not free, it can be done more effectively
   * than trying to remove/check each dirty reverse dependency individually (O(N) each time).
   *
   * <p>When the node entry is evaluating, this list serves to declare the reverse dep operations
   * that have taken place on it during this evaluation. When evaluation finishes, this list will be
   * merged into the existing reverse deps if any, but furthermore, this list will also be used to
   * calculate the set of reverse deps to signal when this entry finishes evaluation. That is done
   * by {@link ReverseDepsUtility#consolidateDataAndReturnNewElements}.
   */
  private List<Object> reverseDepsDataToConsolidate = null;

  public IncrementalInMemoryNodeEntry(SkyKey key) {
    super(key);
  }

  @Override
  public final boolean keepsEdges() {
    return true;
  }

  @Override
  public final Iterable<SkyKey> getDirectDeps() {
    return GroupedDeps.compressedToIterable(getCompressedDirectDepsForDoneEntry());
  }

  @Override
  public final boolean hasAtLeastOneDep() {
    return !GroupedDeps.isEmpty(getCompressedDirectDepsForDoneEntry());
  }

  @Override
  public final synchronized @GroupedDeps.Compressed Object getCompressedDirectDepsForDoneEntry() {
    checkState(isDone(), "no deps until done. NodeEntry: %s", this);
    checkNotNull(directDeps, "deps can't be null: %s", this);
    return GroupedDeps.castAsCompressed(directDeps);
  }

  /**
   * Puts entry in "done" state, as checked by {@link #isDone}. Subclasses that override one may
   * need to override the other.
   */
  @ForOverride
  protected void markDone() {
    dirtyBuildingState = null;
  }

  protected final synchronized Set<SkyKey> setStateFinishedAndReturnReverseDepsToSignal() {
    Set<SkyKey> reverseDepsToSignal = ReverseDepsUtility.consolidateDataAndReturnNewElements(this);
    directDeps = getTemporaryDirectDeps().compress();
    markDone();
    return reverseDepsToSignal;
  }

  @Override
  public synchronized Set<SkyKey> getInProgressReverseDeps() {
    checkState(!isDone(), this);
    return ReverseDepsUtility.returnNewElements(this);
  }

  /**
   * {@inheritDoc}
   *
   * <p>In this method it is crucial that {@link #version} is set prior to {@link #value} because
   * although this method itself is synchronized, there are unsynchronized consumers of the version
   * and the value.
   */
  @Override
  @CanIgnoreReturnValue
  public synchronized Set<SkyKey> setValue(
      SkyValue value, Version graphVersion, @Nullable Version maxTransitiveSourceVersion)
      throws InterruptedException {
    checkState(!hasUnsignaledDeps(), "Has unsignaled deps (this=%s, value=%s)", this, value);
    checkState(
        version.lastChanged().atMost(graphVersion) && version.lastEvaluated().atMost(graphVersion),
        "Bad version (this=%s, version=%s, value=%s)",
        this,
        graphVersion,
        value);

    if (dirtyBuildingState.unchangedFromLastBuild(value)) {
      // If the value is the same as before, just use the old value. Note that we don't use the new
      // value, because preserving == equality is even better than .equals() equality.
      Version lastChanged = version.lastChanged();
      version = NodeVersion.of(lastChanged, graphVersion);
      this.value = dirtyBuildingState.getLastBuildValue();
    } else {
      // If this is a new value, or it has changed since the last build, set the version to the
      // current graph version.
      version = graphVersion;
      this.value = value;
    }
    return setStateFinishedAndReturnReverseDepsToSignal();
  }

  @Override
  @CanIgnoreReturnValue
  public DependencyState addReverseDepAndCheckIfDone(SkyKey reverseDep) {
    if (reverseDep == null && isDone()) {
      return DependencyState.DONE;
    }

    synchronized (this) {
      boolean done = isDone();
      if (!done && dirtyBuildingState == null) {
        dirtyBuildingState = new InitialBuildingState();
      }
      if (reverseDep != null) {
        if (done) {
          ReverseDepsUtility.addReverseDep(this, reverseDep);
        } else {
          appendToReverseDepOperations(reverseDep, Op.ADD);
        }
      }
      if (done) {
        return DependencyState.DONE;
      }
      if (dirtyBuildingState.isEvaluating()) {
        return DependencyState.ALREADY_EVALUATING;
      }
      dirtyBuildingState.startEvaluating();
      return DependencyState.NEEDS_SCHEDULING;
    }
  }

  /** Sets {@link #reverseDeps}. Does not alter {@link #reverseDepsDataToConsolidate}. */
  synchronized void setReverseDepsForReverseDepsUtil(Object reverseDeps) {
    this.reverseDeps = reverseDeps;
  }

  /** Sets {@link #reverseDepsDataToConsolidate}. Does not alter {@link #reverseDeps}. */
  synchronized void setReverseDepsDataToConsolidateForReverseDepsUtil(
      List<Object> dataToConsolidate) {
    this.reverseDepsDataToConsolidate = dataToConsolidate;
  }

  synchronized Object getReverseDepsRawForReverseDepsUtil() {
    return this.reverseDeps;
  }

  synchronized List<Object> getReverseDepsDataToConsolidateForReverseDepsUtil() {
    return this.reverseDepsDataToConsolidate;
  }

  private synchronized void appendToReverseDepOperations(SkyKey reverseDep, Op op) {
    checkState(!isDone(), "Don't append to done %s %s %s", this, reverseDep, op);
    if (reverseDepsDataToConsolidate == null) {
      reverseDepsDataToConsolidate = new ArrayList<>();
    }
    checkState(isDirty() || op != Op.CHECK, "Not dirty check %s %s", this, reverseDep);
    reverseDepsDataToConsolidate.add(KeyToConsolidate.create(reverseDep, op, this));
  }

  @Override
  public synchronized DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep) {
    checkNotNull(reverseDep, this);
    if (isDone()) {
      return DependencyState.DONE;
    }
    appendToReverseDepOperations(reverseDep, Op.CHECK);
    return addReverseDepAndCheckIfDone(null);
  }

  @Override
  public synchronized void removeReverseDep(SkyKey reverseDep) {
    if (isDone()) {
      ReverseDepsUtility.removeReverseDep(this, reverseDep);
    } else {
      // Removing a reverse dep from an in-flight node is rare -- it should only happen when there
      // is a cycle or this node is about to be cleaned from the graph.
      appendToReverseDepOperations(reverseDep, Op.REMOVE);
    }
  }

  @Override
  public synchronized void removeReverseDepsFromDoneEntryDueToDeletion(Set<SkyKey> deletedKeys) {
    checkState(isDone(), this);
    ReverseDepsUtility.removeReverseDepsMatching(this, deletedKeys);
  }

  @Override
  public synchronized Collection<SkyKey> getReverseDepsForDoneEntry() {
    checkState(isDone(), "Called on not done %s", this);
    return ReverseDepsUtility.consolidateAndGetReverseDeps(this, /* checkConsistency= */ true);
  }

  @Override
  public synchronized Collection<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    if (!isDone()) {
      // This consolidation loses information about pending reverse deps to signal, but that is
      // unimportant since this node is being deleted.
      ReverseDepsUtility.consolidateDataAndReturnNewElements(this);
    }
    return ReverseDepsUtility.consolidateAndGetReverseDeps(this, /* checkConsistency= */ false);
  }

  @Override
  public synchronized boolean signalDep(Version childVersion, @Nullable SkyKey childForDebugging) {
    checkState(
        !isDone(), "Value must not be done in signalDep %s child=%s", this, childForDebugging);
    checkNotNull(dirtyBuildingState, "%s %s", this, childForDebugging);
    dirtyBuildingState.signalDep(this, version, childVersion, childForDebugging);
    return !hasUnsignaledDeps();
  }

  /**
   * Creates a {@link DirtyBuildingState} for the case where this node is done and is being marked
   * dirty.
   */
  @ForOverride
  protected DirtyBuildingState createDirtyBuildingStateForDoneNode(
      DirtyType dirtyType, GroupedDeps directDeps, SkyValue value) {
    return new IncrementalBuildingState(dirtyType, directDeps, value);
  }

  @Nullable
  @Override
  public synchronized MarkedDirtyResult markDirty(DirtyType dirtyType) {
    checkNotNull(dirtyType, this);

    if (isDone()) {
      if (dirtyType == DirtyType.REWIND && getErrorInfo() != null) {
        return null; // Rewinding errors is no-op.
      }
      GroupedDeps directDeps = GroupedDeps.decompress(getCompressedDirectDepsForDoneEntry());
      checkState(
          dirtyType != DirtyType.DIRTY || !directDeps.isEmpty(),
          "%s is being marked dirty but has no children that could have dirtied it",
          getKey());
      dirtyBuildingState = createDirtyBuildingStateForDoneNode(dirtyType, directDeps, value);
      value = null;
      this.directDeps = null;
      if (dirtyType == DirtyType.REWIND) {
        // For rewinding, the reverse deps don't need to be included in the MarkedDirtyResult, but
        // they do need to be consolidated so that ReverseDepsUtility considers only rdep operations
        // that occur after the rewind to be "new elements." This is important because only rdeps
        // registered after the rewind should be signalled when the rewound evaluation completes.
        ReverseDepsUtility.consolidateData(this);
        return MarkedDirtyResult.forRewinding();
      } else {
        return MarkedDirtyResult.withReverseDeps(
            ReverseDepsUtility.consolidateAndGetReverseDeps(this, /* checkConsistency= */ true));
      }
    }

    // The caller may be simultaneously trying to mark this node dirty and changed, and the dirty
    // thread may have lost the race, but it is the caller's responsibility not to try to mark this
    // node changed twice. The end result of racing markers must be a changed node, since one of the
    // markers is trying to mark the node changed.
    checkState(value == null, "Value should have been reset already %s", this);
    switch (dirtyType) {
      case CHANGE:
        checkState(!isChanged(), "Cannot mark node changed twice: %s", this);
        checkNotNull(dirtyBuildingState, this).markChanged();
        break;
      case DIRTY:
        checkState(isChanged(), "Cannot mark node dirty twice: %s", this);
        break;
      case REWIND:
        // Rewinding is legal at any time in the node's lifecycle, but is no-op when it is not done.
        break;
    }
    return null;
  }

  @Override
  public synchronized NodeValueAndRdepsToSignal markClean() throws InterruptedException {
    checkNotNull(dirtyBuildingState, this);
    this.value = checkNotNull(dirtyBuildingState.getLastBuildValue());
    checkState(!hasUnsignaledDeps(), this);
    checkState(
        dirtyBuildingState.depsUnchangedFromLastBuild(getTemporaryDirectDeps()),
        "Direct deps must be the same as those found last build for node to be marked clean: %s",
        this);
    checkState(isDirty(), this);
    checkState(!dirtyBuildingState.isChanged(), "shouldn't be changed: %s", this);
    Set<SkyKey> rDepsToSignal = setStateFinishedAndReturnReverseDepsToSignal();
    return new NodeValueAndRdepsToSignal(this.value, rDepsToSignal);
  }

  @Override
  public Version getVersion() {
    return version.lastChanged();
  }

  @Override
  public final synchronized ImmutableSet<SkyKey> getAllDirectDepsForIncompleteNode()
      throws InterruptedException {
    checkState(!isDone(), this);
    if (dirtyBuildingState == null) {
      return ImmutableSet.of();
    }
    return ImmutableSet.<SkyKey>builder()
        .addAll(getTemporaryDirectDeps().getAllElementsAsIterable())
        .addAll(dirtyBuildingState.getAllRemainingDirtyDirectDeps(/* preservePosition= */ false))
        .addAll(getResetDirectDeps())
        .build();
  }

  @Override
  public synchronized GroupedDeps getTemporaryDirectDeps() {
    checkState(!isDone(), "temporary shouldn't be done: %s", this);
    if (directDeps == null) {
      // Initialize lazily, to save a little memory.
      directDeps = newGroupedDeps();
    }
    return (GroupedDeps) directDeps;
  }

  @Override
  public final synchronized void forceRebuild() {
    checkNotNull(dirtyBuildingState, this).forceRebuild(getNumTemporaryDirectDeps());
  }

  @Override
  final synchronized int getNumTemporaryDirectDeps() {
    return directDeps == null ? 0 : getTemporaryDirectDeps().numElements();
  }

  @Override
  public final synchronized void resetEvaluationFromScratch() {
    checkState(!hasUnsignaledDeps(), this);

    ImmutableSet<SkyKey> resetDeps =
        ImmutableSet.<SkyKey>builder()
            .addAll(getResetDirectDeps()) // In case this isn't the first reset.
            .addAll(getTemporaryDirectDeps().getAllElementsAsIterable())
            .build();

    if (dirtyBuildingState.isIncremental()) {
      var incrementalBuildingState = (IncrementalBuildingState) dirtyBuildingState;
      dirtyBuildingState =
          new ResetIncrementalBuildingState(
              incrementalBuildingState.lastBuildDirectDeps,
              incrementalBuildingState.lastBuildValue,
              incrementalBuildingState.dirtyDirectDepIndex,
              resetDeps);
    } else {
      dirtyBuildingState = new ResetInitialBuildingState(resetDeps);
    }
    directDeps = null;
  }

  @Override
  public final ImmutableSet<SkyKey> getResetDirectDeps() {
    return checkNotNull(dirtyBuildingState, this).getResetDirectDeps();
  }

  /**
   * For Skyfocus only: clears out all direct dep edges of this node. It is not safe to call this
   * otherwise.
   */
  public final synchronized void clearDirectDepsForSkyfocus() {

    checkState(isDone(), this);
    this.directDeps = GroupedDeps.EMPTY_COMPRESSED;
  }

  /** Flushes pending reverse dep operations, which potentially saves memory. */
  public final synchronized void consolidateReverseDeps() {

    checkState(isDone(), this);
    ReverseDepsUtility.consolidateData(this);
  }

  @Override
  protected synchronized MoreObjects.ToStringHelper toStringHelper() {
    return super.toStringHelper()
        .add("version", version)
        .add(
            "directDeps",
            isDone() ? GroupedDeps.decompress(getCompressedDirectDepsForDoneEntry()) : directDeps)
        .add("reverseDeps", ReverseDepsUtility.toString(this));
  }

  /** {@link DirtyBuildingState} for a node on an incremental build. */
  private static class IncrementalBuildingState extends DirtyBuildingState {
    private final GroupedDeps lastBuildDirectDeps;
    private final SkyValue lastBuildValue;

    private IncrementalBuildingState(
        DirtyType dirtyType, GroupedDeps lastBuildDirectDeps, SkyValue lastBuildValue) {
      super(dirtyType);
      this.lastBuildDirectDeps = lastBuildDirectDeps;
      this.lastBuildValue = lastBuildValue;
    }

    @Override
    protected final boolean isIncremental() {
      return true;
    }

    @Override
    public final SkyValue getLastBuildValue() {
      return lastBuildValue;
    }

    @Override
    public final GroupedDeps getLastBuildDirectDeps() {
      return lastBuildDirectDeps;
    }

    @Override
    protected final int getNumOfGroupsInLastBuildDirectDeps() {
      return lastBuildDirectDeps.numGroups();
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper()
          .add("lastBuildDirectDeps", lastBuildDirectDeps)
          .add("lastBuildValue", lastBuildValue);
    }
  }

  /**
   * Used to track already registered deps when there is a {@linkplain #resetEvaluationFromScratch
   * reset} on a node's initial build.
   */
  private static final class ResetInitialBuildingState extends InitialBuildingState {
    private final ImmutableSet<SkyKey> resetDeps;

    ResetInitialBuildingState(ImmutableSet<SkyKey> resetDeps) {
      this.resetDeps = resetDeps;
      markRebuilding();
      startEvaluating();
    }

    @Override
    ImmutableSet<SkyKey> getResetDirectDeps() {
      return resetDeps;
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper().add("resetDeps", resetDeps);
    }
  }

  /**
   * Used to track already registered deps when there is a {@linkplain #resetEvaluationFromScratch
   * reset} on a node's incremental build.
   */
  private static final class ResetIncrementalBuildingState extends IncrementalBuildingState {
    private final ImmutableSet<SkyKey> resetDeps;

    private ResetIncrementalBuildingState(
        GroupedDeps lastBuildDirectDeps,
        SkyValue lastBuildValue,
        int dirtyDirectDepIndex,
        ImmutableSet<SkyKey> resetDeps) {
      // CHANGE (not DIRTY) since we already know it needs rebuilding.
      super(DirtyType.CHANGE, lastBuildDirectDeps, lastBuildValue);
      this.dirtyDirectDepIndex = dirtyDirectDepIndex;
      this.resetDeps = resetDeps;
      markRebuilding();
      startEvaluating();
    }

    @Override
    ImmutableSet<SkyKey> getResetDirectDeps() {
      return resetDeps;
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper().add("resetDeps", resetDeps);
    }
  }
}
