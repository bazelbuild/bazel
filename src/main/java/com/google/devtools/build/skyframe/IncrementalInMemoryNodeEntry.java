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

  @Nullable
  @Override
  public SkyValue toValue() {
    SkyValue lastBuildValue = value;
    if (lastBuildValue == null) {
      synchronized (this) {
        if (isDone()) {
          lastBuildValue = value;
        } else if (isChanged() || isDirty()) {
          try {
            lastBuildValue = dirtyBuildingState.getLastBuildValue();
          } catch (InterruptedException e) {
            throw new IllegalStateException("Interruption unexpected: " + this, e);
          }
        }
        // If both if statements are escaped, value has not finished evaluating. It's probably about
        // to be cleaned from the graph.
      }
    }

    return lastBuildValue != null ? ValueWithMetadata.justValue(lastBuildValue) : null;
  }

  @Override
  public Iterable<SkyKey> getDirectDeps() {
    return GroupedDeps.compressedToIterable(getCompressedDirectDepsForDoneEntry());
  }

  @Override
  public boolean hasAtLeastOneDep() {
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
        dirtyBuildingState = new InitialBuildingState(getKey().hasLowFanout());
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
      // Removing a reverse dep from an in-flight node is rare -- it should only happen when this
      // node is about to be cleaned from the graph.
      appendToReverseDepOperations(reverseDep, Op.REMOVE_OLD);
    }
  }

  @Override
  public synchronized void removeReverseDepsFromDoneEntryDueToDeletion(Set<SkyKey> deletedKeys) {
    checkState(isDone(), this);
    ReverseDepsUtility.removeReverseDepsMatching(this, deletedKeys);
  }

  @Override
  public synchronized void removeInProgressReverseDep(SkyKey reverseDep) {
    appendToReverseDepOperations(reverseDep, Op.REMOVE);
  }

  @Override
  public synchronized Collection<SkyKey> getReverseDepsForDoneEntry() {
    checkState(isDone(), "Called on not done %s", this);
    return ReverseDepsUtility.getReverseDeps(this, /* checkConsistency= */ true);
  }

  @Override
  public synchronized Collection<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    if (!isDone()) {
      // This consolidation loses information about pending reverse deps to signal, but that is
      // unimportant since this node is being deleted.
      ReverseDepsUtility.consolidateDataAndReturnNewElements(this);
    }
    return ReverseDepsUtility.getReverseDeps(this, /* checkConsistency= */ false);
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
    return new IncrementalDirtyBuildingState(dirtyType, getKey(), directDeps, value);
  }

  @Nullable
  @Override
  public synchronized MarkedDirtyResult markDirty(DirtyType dirtyType) {
    if (isDone()) {
      GroupedDeps directDeps = GroupedDeps.decompress(getCompressedDirectDepsForDoneEntry());
      dirtyBuildingState = createDirtyBuildingStateForDoneNode(dirtyType, directDeps, value);
      value = null;
      this.directDeps = null;
      return MarkedDirtyResult.withReverseDeps(
          ReverseDepsUtility.getReverseDeps(this, /* checkConsistency= */ true));
    }
    if (dirtyType.equals(DirtyType.FORCE_REBUILD)) {
      if (dirtyBuildingState != null) {
        dirtyBuildingState.markForceRebuild();
      }
      return null;
    }
    // The caller may be simultaneously trying to mark this node dirty and changed, and the dirty
    // thread may have lost the race, but it is the caller's responsibility not to try to mark
    // this node changed twice. The end result of racing markers must be a changed node, since one
    // of the markers is trying to mark the node changed.
    checkState(
        dirtyType.equals(DirtyType.CHANGE) != isChanged(),
        "Cannot mark node dirty twice or changed twice: %s",
        this);
    checkState(value == null, "Value should have been reset already %s", this);
    if (dirtyType.equals(DirtyType.CHANGE)) {
      checkNotNull(dirtyBuildingState, this);
      // If the changed marker lost the race, we just need to mark changed in this method -- all
      // other work was done by the dirty marker.
      dirtyBuildingState.markChanged();
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
  public synchronized GroupedDeps getTemporaryDirectDeps() {
    checkState(!isDone(), "temporary shouldn't be done: %s", this);
    if (directDeps == null) {
      // Initialize lazily, to save a little memory.
      directDeps = newGroupedDeps();
    }
    return (GroupedDeps) directDeps;
  }

  @Override
  final synchronized int getNumTemporaryDirectDeps() {
    return directDeps == null ? 0 : getTemporaryDirectDeps().numElements();
  }

  @Override
  public synchronized void resetForRestartFromScratch() {
    checkState(!hasUnsignaledDeps(), this);
    directDeps = null;
    dirtyBuildingState.resetForRestartFromScratch();
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
  private static final class IncrementalDirtyBuildingState extends DirtyBuildingState {
    private final GroupedDeps lastBuildDirectDeps;
    private final SkyValue lastBuildValue;

    private IncrementalDirtyBuildingState(
        DirtyType dirtyType, SkyKey key, GroupedDeps lastBuildDirectDeps, SkyValue lastBuildValue) {
      super(dirtyType, key.hasLowFanout());
      this.lastBuildDirectDeps = lastBuildDirectDeps;
      this.lastBuildValue = lastBuildValue;
      checkState(
          !dirtyType.equals(DirtyType.DIRTY) || getNumOfGroupsInLastBuildDirectDeps() > 0,
          "%s is being marked dirty but has no children that could have dirtied it",
          key);
    }

    @Override
    protected boolean isIncremental() {
      return true;
    }

    @Override
    public SkyValue getLastBuildValue() {
      return lastBuildValue;
    }

    @Override
    public GroupedDeps getLastBuildDirectDeps() {
      return lastBuildDirectDeps;
    }

    @Override
    protected int getNumOfGroupsInLastBuildDirectDeps() {
      return lastBuildDirectDeps.numGroups();
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper()
          .add("lastBuildDirectDeps", lastBuildDirectDeps)
          .add("lastBuildValue", lastBuildValue);
    }
  }
}
