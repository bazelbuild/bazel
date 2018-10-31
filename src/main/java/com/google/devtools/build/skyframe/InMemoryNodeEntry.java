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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.KeyToConsolidate.Op;
import com.google.devtools.build.skyframe.KeyToConsolidate.OpToStoreBare;
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
 *   <li>Just created ({@link #isEvaluating} is false)
 *   <li>Evaluating ({@link #isEvaluating} is true)
 *   <li>Done ({@link #isDone} is true)
 *   <li>Just created (when it is dirtied: {@link #dirtyBuildingState} is not null)
 *   <li>Reset (just before it is re-evaluated: {@link #dirtyBuildingState#getDirtyState} returns
 *       {@link DirtyState#NEEDS_REBUILDING})
 *   <li>Evaluating
 *   <li>Done
 * </ol>
 *
 * <p>The "just created" state is there to allow the {@link EvaluableGraph#createIfAbsentBatch} and
 * {@link NodeEntry#addReverseDepAndCheckIfDone} methods to be separate. All callers have to call
 * both methods in that order if they want to create a node. The second method transitions the
 * current node to the "evaluating" state and returns true only the first time it was called. A
 * caller that gets "true" back from that call must start the evaluation of this node, while any
 * subsequent callers must not.
 *
 * <p>An entry is set to "evaluating" as soon as it is scheduled for evaluation. Thus, even a node
 * that is never actually built (for instance, a dirty node that is verified as clean) is in the
 * "evaluating" state until it is done.
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
public class InMemoryNodeEntry implements NodeEntry {

  /** Actual data stored in this entry when it is done. */
  protected SkyValue value = null;

  /**
   * The last version of the graph at which this node's value was changed. In {@link #setValue} it
   * may be determined that the value being written to the graph at a given version is the same as
   * the already-stored value. In that case, the version will remain the same. The version can be
   * thought of as the latest timestamp at which this value was changed.
   */
  protected Version lastChangedVersion = MinimalVersion.INSTANCE;

  /**
   * Returns the last version this entry was evaluated at, even if it re-evaluated to the same
   * value. When a child signals this entry with the last version it was changed at in {@link
   * #signalDep}, this entry need not re-evaluate if the child's version is at most this version,
   * even if the {@link #lastChangedVersion} is less than this one.
   *
   * @see #signalDep(Version, SkyKey)
   */
  protected Version lastEvaluatedVersion = MinimalVersion.INSTANCE;

  /**
   * This object represents the direct deps of the node, in groups if the {@code SkyFunction}
   * requested them that way. It contains either the in-progress direct deps, stored as a {@code
   * GroupedList<SkyKey>} before the node is finished building, or the full direct deps, compressed
   * in a memory-efficient way (via {@link GroupedList#compress}, after the node is done.
   *
   * <p>It is initialized lazily in getTemporaryDirectDeps() to save a little bit more memory.
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
  @VisibleForTesting @Nullable protected volatile DirtyBuildingState dirtyBuildingState = null;

  private static final int NOT_EVALUATING_SENTINEL = -1;

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
   * #directDeps}, and then looping through the direct deps and registering this node as a reverse
   * dependency. This ensures that the signaledDeps counter can only reach {@link
   * #directDeps#numElements} on the very last iteration of the loop, i.e., the thread is not
   * working on the node anymore. Note that this requires that there is no code after the loop in
   * {@code ParallelEvaluator.Evaluate#run}.
   */
  private int signaledDeps = NOT_EVALUATING_SENTINEL;

  /**
   * Construct a InMemoryNodeEntry. Use ONLY in Skyframe evaluation and graph implementations.
   */
  public InMemoryNodeEntry() {
  }

  // Public only for use in alternate graph implementations.
  public KeepEdgesPolicy keepEdges() {
    return KeepEdgesPolicy.ALL;
  }

  private boolean keepReverseDeps() {
    return keepEdges() == KeepEdgesPolicy.ALL;
  }

  @Override
  public boolean isDone() {
    return value != null && !isEvaluating();
  }

  @Override
  public SkyValue getValue() {
    Preconditions.checkState(isDone(), "no value until done. ValueEntry: %s", this);
    return ValueWithMetadata.justValue(value);
  }

  @Override
  @Nullable
  public SkyValue getValueMaybeWithMetadata() {
    return value;
  }

  @Override
  public SkyValue toValue() {
    if (isDone()) {
      return getErrorInfo() == null ? getValue() : null;
    } else if (isChanged() || isDirty()) {
      SkyValue lastBuildValue = null;
      try {
        lastBuildValue = getDirtyBuildingState().getLastBuildValue();
      } catch (InterruptedException e) {
        throw new IllegalStateException("Interruption unexpected: " + this, e);
      }
      return (lastBuildValue == null) ? null : ValueWithMetadata.justValue(lastBuildValue);
    } else {
      // Value has not finished evaluating. It's probably about to be cleaned from the graph.
      return null;
    }
  }

  @Override
  public synchronized Iterable<SkyKey> getDirectDeps() {
    return getGroupedDirectDeps().getAllElementsAsIterable();
  }

  /**
   * If {@code isDone()}, returns the ordered list of sets of grouped direct dependencies that were
   * added in {@link #addTemporaryDirectDeps}.
   */
  public synchronized GroupedList<SkyKey> getGroupedDirectDeps() {
    assertKeepDeps();
    Preconditions.checkState(isDone(), "no deps until done. NodeEntry: %s", this);
    return GroupedList.create(directDeps);
  }

  public int getNumDirectDeps() {
    Preconditions.checkState(isDone(), "no deps until done. NodeEntry: %s", this);
    return GroupedList.numElements(directDeps);
  }

  @Override
  @Nullable
  public synchronized ErrorInfo getErrorInfo() {
    Preconditions.checkState(isDone(), "no errors until done. NodeEntry: %s", this);
    return ValueWithMetadata.getMaybeErrorInfo(value);
  }

  protected DirtyBuildingState getDirtyBuildingState() {
    return Preconditions.checkNotNull(dirtyBuildingState, "Didn't have state: %s", this);
  }

  /**
   * Puts entry in "done" state, as checked by {@link #isDone}. Subclasses that override one may
   * need to override the other.
   */
  protected void markDone() {
    dirtyBuildingState = null;
    signaledDeps = NOT_EVALUATING_SENTINEL;
  }

  protected final synchronized Set<SkyKey> setStateFinishedAndReturnReverseDepsToSignal() {
    Set<SkyKey> reverseDepsToSignal =
        ReverseDepsUtility.consolidateDataAndReturnNewElements(this, getOpToStoreBare());
    this.directDeps = getTemporaryDirectDeps().compress();

    markDone();
    postProcessAfterDone();
    return reverseDepsToSignal;
  }

  protected void postProcessAfterDone() {}

  @Override
  public synchronized Set<SkyKey> getInProgressReverseDeps() {
    Preconditions.checkState(!isDone(), this);
    return ReverseDepsUtility.returnNewElements(this, getOpToStoreBare());
  }

  /**
   * Highly dangerous method. Used only for testing/debugging. Can only be called on an in-progress
   * entry that is not dirty and that will not keep edges. Returns all the entry's reverse deps,
   * which must all be {@link SkyKey}s representing {@link Op#ADD} operations, since that is the
   * operation that is stored bare. Used for speed, since it avoids making any copies, so should be
   * much faster than {@link #getInProgressReverseDeps}.
   */
  @SuppressWarnings("unchecked")
  public synchronized Iterable<SkyKey> unsafeGetUnconsolidatedRdeps() {
    Preconditions.checkState(!isDone(), this);
    Preconditions.checkState(!isDirty(), this);
    Preconditions.checkState(keepEdges().equals(KeepEdgesPolicy.NONE), this);
    Preconditions.checkState(getOpToStoreBare() == OpToStoreBare.ADD, this);
    return (Iterable<SkyKey>) (List<?>) reverseDepsDataToConsolidate;
  }

  @Override
  public synchronized Set<SkyKey> setValue(SkyValue value, Version version)
      throws InterruptedException {
    Preconditions.checkState(isReady(), "%s %s", this, value);
    assertVersionCompatibleWhenSettingValue(version, value);
    this.lastEvaluatedVersion = version;

    if (!isEligibleForChangePruning()) {
      this.lastChangedVersion = version;
      this.value = value;
    } else if (isDirty() && getDirtyBuildingState().unchangedFromLastBuild(value)) {
      // If the value is the same as before, just use the old value. Note that we don't use the new
      // value, because preserving == equality is even better than .equals() equality.
      this.value = getDirtyBuildingState().getLastBuildValue();
    } else {
      boolean forcedRebuild =
          isDirty() && getDirtyBuildingState().getDirtyState() == DirtyState.FORCED_REBUILDING;
      // If this is a new value, or it has changed since the last build, set the version to the
      // current graph version.
      if (!forcedRebuild && this.lastChangedVersion.equals(version)) {
        logError(
            new IllegalStateException(
                "Changed value but with the same version? "
                    + this.lastChangedVersion
                    + " "
                    + version
                    + " "
                    + this));
      }
      // If this is a new value, or it has changed since the last build, set the version to the
      // current graph version.
      this.lastChangedVersion = version;
      this.value = value;
    }
    return setStateFinishedAndReturnReverseDepsToSignal();
  }

  /**
   * Returns {@code true} if this node is eligible to be change pruned when its value has not
   * changed from the last build.
   *
   * <p>Implementations need not check whether the value has changed - this will only be called if
   * the value has not changed.
   */
  protected boolean isEligibleForChangePruning() {
    return true;
  }

  protected void assertVersionCompatibleWhenSettingValue(
      Version version, SkyValue valueForDebugging) {
    if (!this.lastChangedVersion.atMost(version)) {
      logError(
          new IllegalStateException("Bad ch: " + this + ", " + version + ", " + valueForDebugging));
    }
    if (!this.lastEvaluatedVersion.atMost(version)) {
      logError(
          new IllegalStateException("Bad ev: " + this + ", " + version + ", " + valueForDebugging));
    }
  }

  @Override
  public synchronized DependencyState addReverseDepAndCheckIfDone(SkyKey reverseDep) {
    if (reverseDep != null) {
      if (isDone()) {
        if (keepReverseDeps()) {
          ReverseDepsUtility.addReverseDeps(this, ImmutableList.of(reverseDep));
        }
      } else {
        appendToReverseDepOperations(reverseDep, Op.ADD);
      }
    }
    if (isDone()) {
      return DependencyState.DONE;
    }
    boolean result = !isEvaluating();
    if (result) {
      signaledDeps = 0;
    }
    return result ? DependencyState.NEEDS_SCHEDULING : DependencyState.ALREADY_EVALUATING;
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
    Preconditions.checkState(!isDone(), "Don't append to done %s %s %s", this, reverseDep, op);
    if (reverseDepsDataToConsolidate == null) {
      reverseDepsDataToConsolidate = new ArrayList<>();
    }
    Preconditions.checkState(
        isDirty() || op != Op.CHECK, "Not dirty check %s %s", this, reverseDep);
    reverseDepsDataToConsolidate.add(KeyToConsolidate.create(reverseDep, op, getOpToStoreBare()));
  }

  protected OpToStoreBare getOpToStoreBare() {
    return isDirty() ? OpToStoreBare.CHECK : OpToStoreBare.ADD;
  }

  @Override
  public synchronized DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep) {
    Preconditions.checkNotNull(reverseDep, this);
    // Note that implementations of InMemoryNodeEntry that have
    // #keepEdges == KeepEdgesPolicy.JUST_DEPS may override this entire method.
    Preconditions.checkState(
        keepEdges() == KeepEdgesPolicy.ALL,
        "Incremental means keeping edges %s %s",
        reverseDep,
        this);
    if (isDone()) {
      ReverseDepsUtility.checkReverseDep(this, reverseDep);
    } else {
      appendToReverseDepOperations(reverseDep, Op.CHECK);
    }
    return addReverseDepAndCheckIfDone(null);
  }

  @Override
  public synchronized void removeReverseDep(SkyKey reverseDep) {
    if (!keepReverseDeps()) {
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
  public synchronized void removeInProgressReverseDep(SkyKey reverseDep) {
    appendToReverseDepOperations(reverseDep, Op.REMOVE);
  }

  @Override
  public synchronized Set<SkyKey> getReverseDepsForDoneEntry() {
    assertKeepRdeps();
    Preconditions.checkState(isDone(), "Called on not done %s", this);
    return ReverseDepsUtility.getReverseDeps(this);
  }

  @Override
  public synchronized Set<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    assertKeepRdeps();
    if (!isDone()) {
      // This consolidation loses information about pending reverse deps to signal, but that is
      // unimportant since this node is being deleted.
      ReverseDepsUtility.consolidateDataAndReturnNewElements(this, getOpToStoreBare());
    }
    return ReverseDepsUtility.getReverseDeps(this);
  }

  @Override
  public synchronized boolean signalDep() {
    return signalDep(/*childVersion=*/ IntVersion.of(Long.MAX_VALUE), /*childForDebugging=*/ null);
  }

  @Override
  public synchronized boolean signalDep(Version childVersion, @Nullable SkyKey childForDebugging) {
    Preconditions.checkState(
        !isDone(), "Value must not be done in signalDep %s child=%s", this, childForDebugging);
    Preconditions.checkState(isEvaluating(), "%s %s", this, childForDebugging);
    signaledDeps++;
    if (isDirty()) {
      dirtyBuildingState.signalDepInternal(
          childCausesReevaluation(lastEvaluatedVersion, childVersion, childForDebugging),
          isReady());
    }
    return isReady();
  }

  @Override
  public synchronized boolean isDirty() {
    return !isDone() && dirtyBuildingState != null;
  }

  @Override
  public synchronized boolean isChanged() {
    return !isDone() && dirtyBuildingState != null && dirtyBuildingState.isChanged();
  }

  /** Checks that a caller is not trying to access not-stored graph edges. */
  private void assertKeepDeps() {
    Preconditions.checkState(keepEdges() != KeepEdgesPolicy.NONE, "Not keeping deps: %s", this);
  }

  /** Checks that a caller is not trying to access not-stored graph edges. */
  private void assertKeepRdeps() {
    Preconditions.checkState(keepEdges() == KeepEdgesPolicy.ALL, "Not keeping rdeps: %s", this);
  }

  @Override
  public synchronized MarkedDirtyResult markDirty(DirtyType dirtyType) {
    // Can't process a dirty node without its deps.
    assertKeepDeps();
    if (isDone()) {
      dirtyBuildingState =
          DirtyBuildingState.create(dirtyType, GroupedList.create(directDeps), value);
      value = null;
      directDeps = null;
      return new MarkedDirtyResult(ReverseDepsUtility.getReverseDeps(this));
    }
    if (dirtyType.equals(DirtyType.FORCE_REBUILD)) {
      getDirtyBuildingState().markForceRebuild();
      return null;
    }
    // The caller may be simultaneously trying to mark this node dirty and changed, and the dirty
    // thread may have lost the race, but it is the caller's responsibility not to try to mark
    // this node changed twice. The end result of racing markers must be a changed node, since one
    // of the markers is trying to mark the node changed.
    Preconditions.checkState(
        dirtyType.equals(DirtyType.CHANGE) != isChanged(),
        "Cannot mark node dirty twice or changed twice: %s",
        this);
    Preconditions.checkState(value == null, "Value should have been reset already %s", this);
    if (dirtyType.equals(DirtyType.CHANGE)) {
      // If the changed marker lost the race, we just need to mark changed in this method -- all
      // other work was done by the dirty marker.
      getDirtyBuildingState().markChanged();
    }
    return null;
  }

  @Override
  public synchronized Set<SkyKey> markClean() throws InterruptedException {
    this.value = getDirtyBuildingState().getLastBuildValue();
    Preconditions.checkState(isReady(), "Should be ready when clean: %s", this);
    Preconditions.checkState(
        getDirtyBuildingState().depsUnchangedFromLastBuild(getTemporaryDirectDeps()),
        "Direct deps must be the same as those found last build for node to be marked clean: %s",
        this);
    Preconditions.checkState(isDirty(), this);
    Preconditions.checkState(!dirtyBuildingState.isChanged(), "shouldn't be changed: %s", this);
    return setStateFinishedAndReturnReverseDepsToSignal();
  }

  @Override
  public synchronized void forceRebuild() {
    Preconditions.checkState(getNumTemporaryDirectDeps() == signaledDeps, this);
    getDirtyBuildingState().forceRebuild();
  }

  @Override
  public Version getVersion() {
    return lastChangedVersion;
  }

  @Override
  public synchronized NodeEntry.DirtyState getDirtyState() {
    Preconditions.checkState(isEvaluating(), "Not evaluating for dirty state? %s", this);
    return getDirtyBuildingState().getDirtyState();
  }

  /** @see DirtyBuildingState#getNextDirtyDirectDeps() */
  @Override
  public synchronized Collection<SkyKey> getNextDirtyDirectDeps() throws InterruptedException {
    Preconditions.checkState(isReady(), this);
    Preconditions.checkState(isEvaluating(), "Not evaluating during getNextDirty? %s", this);
    return getDirtyBuildingState().getNextDirtyDirectDeps();
  }

  @Override
  public synchronized Iterable<SkyKey> getAllDirectDepsForIncompleteNode()
      throws InterruptedException {
    Preconditions.checkState(!isDone(), this);
    if (!isDirty()) {
      return getTemporaryDirectDeps().getAllElementsAsIterable();
    } else {
      // There may be duplicates here. Make sure everything is unique.
      ImmutableSet.Builder<SkyKey> result = ImmutableSet.builder();
      for (Iterable<SkyKey> group : getTemporaryDirectDeps()) {
        result.addAll(group);
      }
      result.addAll(
          getDirtyBuildingState().getAllRemainingDirtyDirectDeps(/*preservePosition=*/ false));
      return result.build();
    }
  }

  @Override
  public synchronized Set<SkyKey> getAllRemainingDirtyDirectDeps() throws InterruptedException {
    Preconditions.checkState(isEvaluating(), "Not evaluating for remaining dirty? %s", this);
    if (isDirty()) {
      DirtyState dirtyState = getDirtyBuildingState().getDirtyState();
      Preconditions.checkState(
          dirtyState == DirtyState.REBUILDING || dirtyState == DirtyState.FORCED_REBUILDING, this);
      return getDirtyBuildingState().getAllRemainingDirtyDirectDeps(/*preservePosition=*/ true);
    } else {
      return ImmutableSet.of();
    }
  }

  @Override
  public synchronized void markRebuilding() {
    getDirtyBuildingState().markRebuilding(isEligibleForChangePruning());
  }

  @SuppressWarnings("unchecked")
  @Override
  public synchronized GroupedList<SkyKey> getTemporaryDirectDeps() {
    Preconditions.checkState(!isDone(), "temporary shouldn't be done: %s", this);
    if (directDeps == null) {
      // Initialize lazily, to save a little bit of memory.
      directDeps = new GroupedList<SkyKey>();
    }
    return (GroupedList<SkyKey>) directDeps;
  }

  private synchronized int getNumTemporaryDirectDeps() {
    return directDeps == null ? 0 : getTemporaryDirectDeps().numElements();
  }

  @Override
  public synchronized boolean noDepsLastBuild() {
    return getDirtyBuildingState().noDepsLastBuild();
  }

  /**
   * {@inheritDoc}
   *
   * <p>This is complicated by the need to maintain the group data. If we remove a dep that ended a
   * group, then its predecessor's group data must be changed to indicate that it now ends the
   * group.
   */
  @Override
  public synchronized void removeUnfinishedDeps(Set<SkyKey> unfinishedDeps) {
    getTemporaryDirectDeps().remove(unfinishedDeps);
  }

  @Override
  public synchronized void resetForRestartFromScratch() {
    Preconditions.checkState(!isDone(), "Reset entry can't be done: %s", this);
    directDeps = null;
    signaledDeps = 0;
    if (dirtyBuildingState != null) {
      dirtyBuildingState.resetForRestartFromScratch();
    }
  }

  @Override
  public synchronized Set<SkyKey> addTemporaryDirectDeps(GroupedListHelper<SkyKey> helper) {
    Preconditions.checkState(!isDone(), "add temp shouldn't be done: %s %s", helper, this);
    return getTemporaryDirectDeps().append(helper);
  }

  @Override
  public synchronized void addTemporaryDirectDepsGroupToDirtyEntry(Collection<SkyKey> group) {
    Preconditions.checkState(!isDone(), "add group temp shouldn't be done: %s %s", group, this);
    getTemporaryDirectDeps().appendGroup(group);
  }

  @Override
  public synchronized boolean isReady() {
    Preconditions.checkState(!isDone(), "can't be ready if done: %s", this);
    return isReady(getNumTemporaryDirectDeps());
  }

  /** True if the child should cause re-evaluation of this node. */
  protected boolean childCausesReevaluation(
      Version lastEvaluatedVersion,
      Version childVersion,
      @Nullable SkyKey unusedChildForDebugging) {
    // childVersion > lastEvaluatedVersion
    return !childVersion.atMost(lastEvaluatedVersion);
  }

  protected int getSignaledDeps() {
    return signaledDeps;
  }

  protected void logError(RuntimeException error) {
    throw error;
  }

  /** Returns whether all known children of this node have signaled that they are done. */
  private boolean isReady(int numDirectDeps) {
    Preconditions.checkState(signaledDeps <= numDirectDeps, "%s %s", numDirectDeps, this);
    return signaledDeps == numDirectDeps;
  }

  private boolean isEvaluating() {
    return signaledDeps > NOT_EVALUATING_SENTINEL;
  }

  @Override
  public synchronized String toString() {
    return MoreObjects.toStringHelper(this)
        .add("identity", System.identityHashCode(this))
        .add("value", value)
        .add("lastChangedVersion", lastChangedVersion)
        .add("lastEvaluatedVersion", lastEvaluatedVersion)
        .add("directDeps", isDone() ? GroupedList.create(directDeps) : directDeps)
        .add("signaledDeps", signaledDeps)
        .add("reverseDeps", ReverseDepsUtility.toString(this))
        .add("dirtyBuildingState", dirtyBuildingState)
        .toString();
  }

  protected synchronized InMemoryNodeEntry cloneNodeEntry(InMemoryNodeEntry newEntry) {
    // As this is temporary, for now let's limit to done nodes.
    Preconditions.checkState(isDone(), "Only done nodes can be copied: %s", this);
    newEntry.value = value;
    newEntry.lastChangedVersion = this.lastChangedVersion;
    newEntry.lastEvaluatedVersion = this.lastEvaluatedVersion;
    ReverseDepsUtility.addReverseDeps(newEntry, ReverseDepsUtility.getReverseDeps(this));
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
    return cloneNodeEntry(new InMemoryNodeEntry());
  }
}
