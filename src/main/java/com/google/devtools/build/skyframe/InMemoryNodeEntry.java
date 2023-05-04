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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.skyframe.KeyToConsolidate.Op;
import com.google.errorprone.annotations.ForOverride;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * In-memory implementation of {@link NodeEntry}. All operations on this class are thread-safe.
 *
 * <p>Care was taken to provide certain compound operations to avoid certain check-then-act races.
 * That means this class is somewhat closely tied to the exact Evaluator implementation.
 *
 * <p>Consider the example with two threads working on two nodes, where one depends on the other,
 * say b depends on a. If a completes first, it's done. If it completes second, it needs to signal
 * b, and potentially re-schedule it. If b completes first, it must exit, because it will be
 * signaled (and re-scheduled) by a. If it completes second, it must signal (and re-schedule)
 * itself. However, if the Evaluator supported re-entrancy for a node, then this wouldn't have to be
 * so strict, because duplicate scheduling would be less problematic.
 *
 * <p>During its life, a node can go through states as follows:
 *
 * <ol>
 *   <li>Non-existent
 *   <li>Just created or marked as affected ({@link #isDone} is false; {@link #isDirty} is false)
 *   <li>Evaluating ({@link #isDone} is false; {@link #isDirty} is true)
 *   <li>Done ({@link #isDone} is true; {@link #isDirty} is false)
 * </ol>
 *
 * <p>The "just created" state is there to allow the {@link ProcessableGraph#createIfAbsentBatch}
 * and {@link NodeEntry#addReverseDepAndCheckIfDone} methods to be separate. All callers have to
 * call both methods in that order if they want to create a node. The second method returns the
 * NEEDS_SCHEDULING state only on the first time it was called. A caller that gets NEEDS_SCHEDULING
 * back from that call must start the evaluation of this node, while any subsequent callers must
 * not.
 *
 * <p>An entry is set to ALREADY_EVALUATING as soon as it is scheduled for evaluation. Thus, even a
 * node that is never actually built (for instance, a dirty node that is verified as clean) is in
 * the ALREADY_EVALUATING state until it is DONE.
 *
 * <p>From the DONE state, the node can go back to the "marked as affected" state.
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
public class InMemoryNodeEntry implements NodeEntry {

  private final SkyKey key;

  /** Actual data stored in this entry when it is done. */
  protected volatile SkyValue value = null;

  protected volatile NodeVersion version = Version.minimal();

  /**
   * This object represents the direct deps of the node, in groups if the {@code SkyFunction}
   * requested them that way. It contains either the in-progress direct deps, stored as a {@link
   * GroupedDeps} (constructed via {@link GroupedDeps.WithHashSet} if {@code
   * key.supportsPartialReevaluation()}) before the node is finished building, or the full direct
   * deps, compressed in a memory-efficient way (via {@link GroupedDeps#compress}, after the node is
   * done.
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

  /**
   * Object encapsulating dirty state of the object between when it is marked dirty and
   * re-evaluated.
   */
  @Nullable protected volatile DirtyBuildingState dirtyBuildingState = null;

  public InMemoryNodeEntry(SkyKey key) {
    this.key = checkNotNull(key);
  }

  public final SkyKey getKey() {
    return key;
  }

  /** Whether this node stores edges (deps and rdeps). */
  boolean keepsEdges() {
    return true;
  }

  private boolean isEvaluating() {
    return dirtyBuildingState != null;
  }

  @Override
  public boolean isDone() {
    return value != null && dirtyBuildingState == null;
  }

  @Override
  public synchronized boolean isReadyToEvaluate() {
    return !isDone()
        && isEvaluating()
        && (dirtyBuildingState.isReady(getNumTemporaryDirectDeps())
            || key.supportsPartialReevaluation());
  }

  @Override
  public synchronized boolean hasUnsignaledDeps() {
    checkState(!isDone(), this);
    checkState(isEvaluating(), this);
    return !dirtyBuildingState.isReady(getNumTemporaryDirectDeps());
  }

  @Override
  public synchronized boolean isDirty() {
    return !isDone() && dirtyBuildingState != null;
  }

  @Override
  public synchronized boolean isChanged() {
    return !isDone() && dirtyBuildingState != null && dirtyBuildingState.isChanged();
  }

  @Override
  public synchronized SkyValue getValue() {
    checkState(isDone(), "no value until done. ValueEntry: %s", this);
    return ValueWithMetadata.justValue(value);
  }

  @Override
  @Nullable
  public SkyValue getValueMaybeWithMetadata() {
    return value;
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

  /** Returns the compressed {@link GroupedDeps} of direct deps. Can only be called when done. */
  public final synchronized @GroupedDeps.Compressed Object getCompressedDirectDepsForDoneEntry() {
    assertKeepEdges();
    checkState(isDone(), "no deps until done. NodeEntry: %s", this);
    checkNotNull(directDeps, "deps can't be null: %s", this);
    return GroupedDeps.castAsCompressed(directDeps);
  }

  public int getNumDirectDeps() {
    return GroupedDeps.numElements(getCompressedDirectDepsForDoneEntry());
  }

  @Override
  @Nullable
  public synchronized ErrorInfo getErrorInfo() {
    checkState(isDone(), "no errors until done. NodeEntry: %s", this);
    return ValueWithMetadata.getMaybeErrorInfo(value);
  }

  /**
   * Puts entry in "done" state, as checked by {@link #isDone}. Subclasses that override one may
   * need to override the other.
   */
  protected void markDone() {
    dirtyBuildingState = null;
  }

  @Override
  public synchronized void addExternalDep() {
    checkNotNull(dirtyBuildingState, this);
    dirtyBuildingState.addExternalDep();
  }

  protected final synchronized Set<SkyKey> setStateFinishedAndReturnReverseDepsToSignal() {
    Set<SkyKey> reverseDepsToSignal = ReverseDepsUtility.consolidateDataAndReturnNewElements(this);
    directDeps = keepsEdges() ? getTemporaryDirectDeps().compress() : null;
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
  public DependencyState addReverseDepAndCheckIfDone(SkyKey reverseDep) {
    if ((reverseDep == null || !keepsEdges()) && isDone()) {
      return DependencyState.DONE;
    }

    synchronized (this) {
      boolean done = isDone();
      if (!done && dirtyBuildingState == null) {
        dirtyBuildingState = DirtyBuildingState.createNew(key.hasLowFanout());
      }
      if (reverseDep != null) {
        if (done) {
          if (keepsEdges()) {
            ReverseDepsUtility.addReverseDep(this, reverseDep);
          }
        } else {
          appendToReverseDepOperations(reverseDep, Op.ADD);
        }
      }
      if (done) {
        return DependencyState.DONE;
      }
      boolean wasEvaluating = dirtyBuildingState.isEvaluating();
      if (!wasEvaluating) {
        dirtyBuildingState.startEvaluating();
      }
      return wasEvaluating ? DependencyState.ALREADY_EVALUATING : DependencyState.NEEDS_SCHEDULING;
    }
  }

  /** Sets {@link #reverseDeps}. Does not alter {@link #reverseDepsDataToConsolidate}. */
  synchronized void setSingleReverseDepForReverseDepsUtil(SkyKey reverseDep) {
    this.reverseDeps = reverseDep;
  }

  /** Sets {@link #reverseDeps}. Does not alter {@link #reverseDepsDataToConsolidate}. */
  synchronized void setReverseDepsForReverseDepsUtil(List<SkyKey> reverseDeps) {
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
    checkState(keepsEdges(), "Incremental means keeping edges %s %s", reverseDep, this);
    if (isDone()) {
      ReverseDepsUtility.checkReverseDep(this, reverseDep);
    } else {
      appendToReverseDepOperations(reverseDep, Op.CHECK);
    }
    return addReverseDepAndCheckIfDone(null);
  }

  @Override
  public synchronized void removeReverseDep(SkyKey reverseDep) {
    if (!keepsEdges()) {
      return;
    }
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
    assertKeepEdges();
    checkState(isDone(), this);
    ReverseDepsUtility.removeReverseDepsMatching(this, deletedKeys);
  }

  @Override
  public synchronized void removeInProgressReverseDep(SkyKey reverseDep) {
    appendToReverseDepOperations(reverseDep, Op.REMOVE);
  }

  @Override
  public synchronized Collection<SkyKey> getReverseDepsForDoneEntry() {
    assertKeepEdges();
    checkState(isDone(), "Called on not done %s", this);
    return ReverseDepsUtility.getReverseDeps(this, /*checkConsistency=*/ true);
  }

  @Override
  public synchronized Collection<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    assertKeepEdges();
    if (!isDone()) {
      // This consolidation loses information about pending reverse deps to signal, but that is
      // unimportant since this node is being deleted.
      ReverseDepsUtility.consolidateDataAndReturnNewElements(this);
    }
    return ReverseDepsUtility.getReverseDeps(this, /*checkConsistency=*/ false);
  }

  @Override
  public synchronized boolean signalDep(Version childVersion, @Nullable SkyKey childForDebugging) {
    checkState(
        !isDone(), "Value must not be done in signalDep %s child=%s", this, childForDebugging);
    checkNotNull(dirtyBuildingState, "%s %s", this, childForDebugging);
    dirtyBuildingState.signalDep(this, childVersion, childForDebugging);
    return !hasUnsignaledDeps();
  }

  /** Checks that a caller is not trying to access not-stored graph edges. */
  private void assertKeepEdges() {
    checkState(keepsEdges(), "Not keeping edges: %s", this);
  }

  /**
   * Creates a {@link DirtyBuildingState} for the case where this node is done and is being marked
   * dirty.
   */
  @ForOverride
  protected DirtyBuildingState createDirtyBuildingStateForDoneNode(
      DirtyType dirtyType, GroupedDeps directDeps, SkyValue value) {
    return DirtyBuildingState.create(dirtyType, directDeps, value, key.hasLowFanout());
  }

  private static final GroupedDeps EMPTY_LIST = new GroupedDeps();

  @Nullable
  @Override
  public synchronized MarkedDirtyResult markDirty(DirtyType dirtyType) {
    if (!DirtyType.FORCE_REBUILD.equals(dirtyType)) {
      // A node can't be found to be dirty without deps unless it's force-rebuilt.
      assertKeepEdges();
    }
    if (isDone()) {
      GroupedDeps directDeps =
          keepsEdges() ? GroupedDeps.decompress(getCompressedDirectDepsForDoneEntry()) : EMPTY_LIST;
      dirtyBuildingState = createDirtyBuildingStateForDoneNode(dirtyType, directDeps, value);
      value = null;
      this.directDeps = null;
      return new MarkedDirtyResult(
          keepsEdges()
              ? ReverseDepsUtility.getReverseDeps(this, /* checkConsistency= */ true)
              : ImmutableList.of());
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
  public synchronized void forceRebuild() {
    checkNotNull(dirtyBuildingState, this);
    checkState(isEvaluating(), this);
    dirtyBuildingState.forceRebuild(getNumTemporaryDirectDeps());
  }

  @Override
  public Version getVersion() {
    return version.lastChanged();
  }

  @Override
  public synchronized NodeEntry.DirtyState getDirtyState() {
    checkNotNull(dirtyBuildingState, this);
    return dirtyBuildingState.getDirtyState();
  }

  /**
   * @see DirtyBuildingState#getNextDirtyDirectDeps()
   */
  @Override
  public synchronized List<SkyKey> getNextDirtyDirectDeps() throws InterruptedException {
    checkState(!hasUnsignaledDeps(), this);
    checkNotNull(dirtyBuildingState, this);
    checkState(dirtyBuildingState.isEvaluating(), "Not evaluating during getNextDirty? %s", this);
    return dirtyBuildingState.getNextDirtyDirectDeps();
  }

  @Override
  public synchronized Iterable<SkyKey> getAllDirectDepsForIncompleteNode()
      throws InterruptedException {
    checkState(!isDone(), this);
    if (!isDirty()) {
      return getTemporaryDirectDeps().getAllElementsAsIterable();
    } else {
      // There may be duplicates here. Make sure everything is unique.
      ImmutableSet.Builder<SkyKey> result = ImmutableSet.builder();
      for (List<SkyKey> group : getTemporaryDirectDeps()) {
        result.addAll(group);
      }
      result.addAll(dirtyBuildingState.getAllRemainingDirtyDirectDeps(/*preservePosition=*/ false));
      return result.build();
    }
  }

  @Override
  public synchronized ImmutableSet<SkyKey> getAllRemainingDirtyDirectDeps()
      throws InterruptedException {
    checkNotNull(dirtyBuildingState, this);
    checkState(dirtyBuildingState.isEvaluating(), "Not evaluating for remaining dirty? %s", this);
    if (isDirty()) {
      DirtyState dirtyState = dirtyBuildingState.getDirtyState();
      checkState(
          dirtyState == DirtyState.REBUILDING || dirtyState == DirtyState.FORCED_REBUILDING, this);
      return dirtyBuildingState.getAllRemainingDirtyDirectDeps(/*preservePosition=*/ true);
    } else {
      return ImmutableSet.of();
    }
  }

  @Override
  public synchronized void markRebuilding() {
    checkNotNull(dirtyBuildingState, this).markRebuilding();
  }

  @Override
  public synchronized GroupedDeps getTemporaryDirectDeps() {
    checkState(!isDone(), "temporary shouldn't be done: %s", this);
    if (directDeps == null) {
      // Initialize lazily, to save a little memory.
      //
      // If the key opts into partial reevaluation, tracking deps with a HashSet is worth the extra
      // memory cost -- see SkyFunctionEnvironment.PartialReevaluation.
      directDeps =
          key.supportsPartialReevaluation() ? new GroupedDeps.WithHashSet() : new GroupedDeps();
    }
    return (GroupedDeps) directDeps;
  }

  final synchronized int getNumTemporaryDirectDeps() {
    return directDeps == null ? 0 : getTemporaryDirectDeps().numElements();
  }

  @Override
  public synchronized boolean noDepsLastBuild() {
    checkState(isEvaluating(), this);
    return dirtyBuildingState.noDepsLastBuild();
  }

  @Override
  public synchronized void removeUnfinishedDeps(Set<SkyKey> unfinishedDeps) {
    getTemporaryDirectDeps().remove(unfinishedDeps);
  }

  @Override
  public synchronized void resetForRestartFromScratch() {
    checkState(!hasUnsignaledDeps(), this);
    directDeps = null;
    dirtyBuildingState.resetForRestartFromScratch();
  }

  @Override
  public synchronized void addSingletonTemporaryDirectDep(SkyKey dep) {
    getTemporaryDirectDeps().appendSingleton(dep);
  }

  @Override
  public synchronized void addTemporaryDirectDepGroup(List<SkyKey> group) {
    getTemporaryDirectDeps().appendGroup(group);
  }

  @Override
  public synchronized void addTemporaryDirectDepsInGroups(
      Set<SkyKey> deps, List<Integer> groupSizes) {
    getTemporaryDirectDeps().appendGroups(deps, groupSizes);
  }

  @Override
  public int getPriority() {
    var snapshot = dirtyBuildingState;
    if (snapshot == null) {
      return Integer.MAX_VALUE;
    }
    return snapshot.getPriority();
  }

  @Override
  public int depth() {
    var snapshot = dirtyBuildingState;
    if (snapshot == null) {
      return 0;
    }
    return snapshot.depth();
  }

  @Override
  public void updateDepthIfGreater(int proposedDepth) {
    var snapshot = dirtyBuildingState;
    if (snapshot == null) {
      return;
    }
    snapshot.updateDepthIfGreater(proposedDepth);
  }

  @Override
  public void incrementEvaluationCount() {
    var snapshot = dirtyBuildingState;
    if (snapshot == null) {
      return;
    }
    snapshot.incrementEvaluationCount();
  }

  protected synchronized MoreObjects.ToStringHelper toStringHelper() {
    return MoreObjects.toStringHelper(this)
        .add("key", key)
        .add("identity", System.identityHashCode(this))
        .add("value", value)
        .add("version", version)
        .add(
            "directDeps",
            isDone() && keepsEdges()
                ? GroupedDeps.decompress(getCompressedDirectDepsForDoneEntry())
                : directDeps)
        .add("reverseDeps", ReverseDepsUtility.toString(this))
        .add("dirtyBuildingState", dirtyBuildingState);
  }

  @Override
  public final synchronized String toString() {
    return toStringHelper().toString();
  }

  // Only used for testing hooks.
  protected synchronized InMemoryNodeEntry cloneNodeEntry(InMemoryNodeEntry newEntry) {
    checkState(isDone(), "Only done nodes can be copied: %s", this);
    newEntry.value = value;
    newEntry.version = version;
    for (SkyKey reverseDep : ReverseDepsUtility.getReverseDeps(this, /*checkConsistency=*/ true)) {
      ReverseDepsUtility.addReverseDep(newEntry, reverseDep);
    }
    newEntry.directDeps = directDeps;
    newEntry.dirtyBuildingState = null;
    return newEntry;
  }

  /**
   * Do not use except in custom evaluator implementations! Added only temporarily.
   *
   * <p>Clones a InMemoryMutableNodeEntry iff it is a done node. Otherwise it fails.
   */
  public synchronized InMemoryNodeEntry cloneNodeEntry() {
    return cloneNodeEntry(new InMemoryNodeEntry(key));
  }
}
