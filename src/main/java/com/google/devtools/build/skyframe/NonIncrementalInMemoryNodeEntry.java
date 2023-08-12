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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.skyframe.NonIncrementalInMemoryNodeEntry.NonIncrementalBuildingState;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An {@link InMemoryNodeEntry} that does not store edges (direct deps and reverse deps) once the
 * node is done. Used to save memory when the graph will not be reused for incremental builds.
 *
 * <p>Edges are stored as usual while the node is being built, but are discarded once the node is
 * done.
 *
 * <p>It is illegal to access edges once the node {@link #isDone}.
 */
public class NonIncrementalInMemoryNodeEntry
    extends AbstractInMemoryNodeEntry<NonIncrementalBuildingState> {

  public NonIncrementalInMemoryNodeEntry(SkyKey key) {
    super(key);
  }

  @Override
  public final boolean keepsEdges() {
    return false;
  }

  @Override
  @CanIgnoreReturnValue
  public synchronized ImmutableSet<SkyKey> setValue(
      SkyValue value, Version graphVersion, @Nullable Version maxTransitiveSourceVersion) {
    checkArgument(
        graphVersion.equals(Version.constant()),
        "Non-incremental evaluations must be at a constant version: %s",
        graphVersion);
    checkState(!hasUnsignaledDeps(), "Has unsignaled deps (this=%s, value=%s)", this, value);
    this.value = value;
    ImmutableSet<SkyKey> reverseDepsToSignal = dirtyBuildingState.getReverseDeps(this);
    dirtyBuildingState = null;
    return reverseDepsToSignal;
  }

  @Override
  @Nullable
  public final SkyValue toValue() {
    return ValueWithMetadata.justValue(value);
  }

  @Override
  @CanIgnoreReturnValue
  public final DependencyState addReverseDepAndCheckIfDone(@Nullable SkyKey reverseDep) {
    // Fast path check before locking. If this node is already done, there is nothing to do since we
    // aren't storing reverse deps.
    if (isDone()) {
      return DependencyState.DONE;
    }

    synchronized (this) {
      // Check again under a lock.
      if (isDone()) {
        return DependencyState.DONE;
      }
      if (dirtyBuildingState == null) {
        dirtyBuildingState = newBuildingState();
      }
      if (reverseDep != null) {
        dirtyBuildingState.addReverseDep(reverseDep);
      }
      if (dirtyBuildingState.isEvaluating()) {
        return DependencyState.ALREADY_EVALUATING;
      }
      dirtyBuildingState.startEvaluating();
      return DependencyState.NEEDS_SCHEDULING;
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>A {@link NonIncrementalInMemoryNodeEntry} can only ever be at one of two versions: either
   * {@link Version#constant} when a value is available, or {@link Version#minimal} otherwise.
   *
   * <p>All non-incremental evaluations must use {@link Version#constant} as the graph version. This
   * is enforced in {@link #setValue}.
   */
  @Override
  public final Version getVersion() {
    return value != null ? Version.constant() : Version.minimal();
  }

  @Override
  public final synchronized GroupedDeps getTemporaryDirectDeps() {
    return checkNotNull(dirtyBuildingState, "Not evaluating: %s", this)
        .getTemporaryDirectDeps(this);
  }

  @Override
  public final void resetForRestartFromScratch() {
    checkState(!hasUnsignaledDeps(), this);
    dirtyBuildingState.directDeps = null;
    dirtyBuildingState.resetForRestartFromScratch();
  }

  @Override
  final synchronized int getNumTemporaryDirectDeps() {
    if (dirtyBuildingState == null) {
      return 0;
    }
    GroupedDeps directDeps = dirtyBuildingState.directDeps;
    return directDeps == null ? 0 : directDeps.numElements();
  }

  @Nullable
  @Override
  public final synchronized MarkedDirtyResult markDirty(DirtyType dirtyType) {
    checkState(dirtyType == DirtyType.FORCE_REBUILD, "Unexpected dirty type: %s", dirtyType);
    if (!isDone()) {
      if (dirtyBuildingState != null) {
        dirtyBuildingState.markForceRebuild();
      }
      return null;
    }
    dirtyBuildingState = newBuildingState();
    value = null;
    return MarkedDirtyResult.withReverseDeps(ImmutableList.of());
  }

  @Override
  public final synchronized Set<SkyKey> getInProgressReverseDeps() {
    checkState(!isDone(), this);
    return dirtyBuildingState == null ? ImmutableSet.of() : dirtyBuildingState.getReverseDeps(this);
  }

  @Override
  public final void removeInProgressReverseDep(SkyKey reverseDep) {
    checkNotNull(dirtyBuildingState, "Not evaluating: %s", this).removeReverseDep(reverseDep);
  }

  @Override
  public final synchronized boolean signalDep(
      Version childVersion, @Nullable SkyKey childForDebugging) {
    checkState(
        !isDone(), "Value must not be done in signalDep %s child=%s", this, childForDebugging);
    checkNotNull(dirtyBuildingState, "%s %s", this, childForDebugging)
        .signalDep(this, Version.minimal(), childVersion, childForDebugging);
    return !hasUnsignaledDeps();
  }

  @Override
  public final void removeReverseDep(SkyKey reverseDep) {
    throw unsupported();
  }

  @Override
  public final @GroupedDeps.Compressed Object getCompressedDirectDepsForDoneEntry() {
    throw unsupported();
  }

  @Override
  public final Iterable<SkyKey> getDirectDeps() {
    throw unsupported();
  }

  @Override
  public final boolean hasAtLeastOneDep() {
    throw unsupported();
  }

  @Override
  public final void removeReverseDepsFromDoneEntryDueToDeletion(Set<SkyKey> deletedKeys) {
    throw unsupported();
  }

  @Override
  public final Collection<SkyKey> getReverseDepsForDoneEntry() {
    throw unsupported();
  }

  @Override
  public final Collection<SkyKey> getAllReverseDepsForNodeBeingDeleted() {
    throw unsupported();
  }

  @Override
  public final DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep) {
    throw unsupported();
  }

  @Override
  public final NodeValueAndRdepsToSignal markClean() {
    throw unsupported();
  }

  private UnsupportedOperationException unsupported() {
    return new UnsupportedOperationException("Not keeping edges: " + this);
  }

  private NonIncrementalBuildingState newBuildingState() {
    return new NonIncrementalBuildingState(getKey().hasLowFanout());
  }

  /**
   * Specialized {@link DirtyBuildingState} for a non-incremental node.
   *
   * <p>The {@link #directDeps} and {@link #reverseDeps} fields are stored in this class instead of
   * in {@link NonIncrementalInMemoryNodeEntry} since they are not needed after the node is done.
   * This way we don't pay the memory cost of the fields for a done node.
   */
  static final class NonIncrementalBuildingState extends InitialBuildingState {
    @Nullable private GroupedDeps directDeps = null;
    @Nullable private List<SkyKey> reverseDeps = null;

    private NonIncrementalBuildingState(boolean hasLowFanout) {
      super(hasLowFanout);
    }

    GroupedDeps getTemporaryDirectDeps(NonIncrementalInMemoryNodeEntry entry) {
      if (directDeps == null) {
        directDeps = entry.newGroupedDeps();
      }
      return directDeps;
    }

    void addReverseDep(SkyKey reverseDep) {
      if (reverseDeps == null) {
        reverseDeps = new ArrayList<>();
      }
      reverseDeps.add(reverseDep);
    }

    void removeReverseDep(SkyKey reverseDep) {
      // Reverse dep removal on a non-incremental node is rare (only for cycles), so we can live
      // with inefficiently calling remove on an ArrayList.
      checkState(reverseDeps.remove(reverseDep), "Reverse dep not present: %s", reverseDep);
    }

    ImmutableSet<SkyKey> getReverseDeps(NonIncrementalInMemoryNodeEntry entry) {
      if (reverseDeps == null) {
        return ImmutableSet.of();
      }
      ImmutableSet<SkyKey> result = ImmutableSet.copyOf(reverseDeps);
      ReverseDepsUtility.checkForDuplicates(result, reverseDeps, entry);
      return result;
    }

    @Override
    protected MoreObjects.ToStringHelper getStringHelper() {
      return super.getStringHelper().add("directDeps", directDeps).add("reverseDeps", reverseDeps);
    }
  }
}
