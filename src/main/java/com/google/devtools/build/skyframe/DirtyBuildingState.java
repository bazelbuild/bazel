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

import static com.google.common.base.Preconditions.checkState;
import static java.lang.Math.min;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.unsafe.UnsafeProvider;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import java.util.List;
import javax.annotation.Nullable;
import sun.misc.Unsafe;

/**
 * State for a node that either has not been built yet or has been dirtied.
 *
 * <p>If the node has previously been built, {@link #isIncremental} returns true. Deps are checked
 * to see if re-evaluation is needed, and the node will either marked clean or re-evaluated.
 *
 * <p>This class does not attempt to synchronize operations. It is assumed that the calling {@link
 * InMemoryNodeEntry} performs the appropriate synchronization when necessary.
 *
 * <p>This class is public only for the benefit of alternative graph implementations outside of the
 * package.
 */
public abstract class DirtyBuildingState implements PriorityTracker {
  private static final int NOT_EVALUATING_SENTINEL = -1;

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
   * IncrementalInMemoryNodeEntry#directDeps}, and then looping through the direct deps and
   * registering this node as a reverse dependency. This ensures that the signaledDeps counter can
   * only reach {@link GroupedDeps#numElements} on the very last iteration of the loop, i.e., the
   * thread is not working on the node anymore. Note that this requires that there is no code after
   * the loop in {@link ParallelEvaluator.Evaluate#run}.
   */
  private int signaledDeps = NOT_EVALUATING_SENTINEL;

  /**
   * The number of external dependencies (in contrast to the number of internal dependencies which
   * are tracked in NodeEntry). We never keep information about external dependencies across
   * Skyframe calls.
   */
  // We do not strictly require a counter here; all external deps from one SkyFunction evaluation
  // pass are registered as a single logical dependency, and the SkyFunction is only re-evaluated if
  // all of them complete. Therefore, we only need a single bit to track this fact. If the mere
  // existence of this field turns out to be a significant memory burden, we could change the
  // implementation by moving to a single-bit approach, and then store that bit as part of the
  // dirtyState field, e.g., by adding a REBUILDING_WAITING_FOR_EXTERNAL_DEPS enum value, as this
  // can only happen during evaluation.
  private int externalDeps;

  /**
   * Priority information packed into 32-bits.
   *
   * <p>Packing is used because Java lacks support for native short operations and masking
   * operations are needed to fit 1-bit of information about whether the {@link SkyKey} has low
   * fanout. This field has the following layout.
   *
   * <ol>
   *   <li><i>Has Low Fanout</i> (1-bit) - 1 if {@link SkyKey#hasLowFanout} is true for the
   *       underlying key.
   *   <li><i>Depth</i> (15-bits) - the current estimated depth.
   *   <li><i>Restart Count</i> (16-bits, unsigned) - incremented when this node restarts.
   * </ol>
   */
  private volatile int priority = HAS_LOW_FANOUT_MASK;

  /**
   * Returns the {@link GroupedDeps} requested last time the node was built, or {@code null} if on
   * its initial build.
   *
   * <p>Dependencies from the last build are be compared to dependencies requested on this build to
   * check whether this node has changed in {@link NodeEntry#setValue}. See {@link
   * IncrementalInMemoryNodeEntry#directDeps} for more on dependency group storage.
   *
   * <p>Public only for the use of alternative graph implementations.
   */
  @Nullable
  public abstract GroupedDeps getLastBuildDirectDeps() throws InterruptedException;

  /**
   * The number of groups of the dependencies requested last time when the node was built, or {@code
   * 0} if on its initial build.
   *
   * <p>Getting the number of last-built dependencies should not throw {@link InterruptedException}.
   */
  protected abstract int getNumOfGroupsInLastBuildDirectDeps();

  /**
   * The value of the node the last time it was built, or {@code null} if on its initial build.
   *
   * <p>Public only for the use of alternative graph implementations.
   */
  @Nullable
  public abstract SkyValue getLastBuildValue() throws InterruptedException;

  /**
   * Group of children to be checked next in the process of determining if this entry needs to be
   * re-evaluated. Used by {@link DirtyBuildingState#getNextDirtyDirectDeps} and {@link #signalDep}.
   */
  protected int dirtyDirectDepIndex;

  /**
   * Constructor.
   *
   * @param hasLowFanout indicates that this node is not expected to have high dependency fanout.
   *     Satisfied by by {@link SkyKey#hasLowFanout}.
   */
  protected DirtyBuildingState(DirtyType dirtyType, boolean hasLowFanout) {
    dirtyState = dirtyType.getInitialDirtyState();
    // We need to iterate through the deps to see if they have changed, or to remove them if one
    // has. Initialize the iterating index.
    dirtyDirectDepIndex = 0;
    if (!hasLowFanout) {
      this.priority = 0;
    }
  }

  /** Returns true if this state has information about a previously built version. */
  protected abstract boolean isIncremental();

  final void markChanged() {
    checkState(dirtyState == DirtyState.CHECK_DEPENDENCIES, this);
    checkState(dirtyDirectDepIndex == 0, "Unexpected evaluation: %s", this);
    dirtyState = DirtyState.NEEDS_REBUILDING;
  }

  final void markForceRebuild() {
    if (dirtyState == DirtyState.CHECK_DEPENDENCIES) {
      dirtyState = DirtyState.NEEDS_REBUILDING;
    }
  }

  // TODO(b/228090759): Tighten up state checks for the force rebuild lifecycle.
  final void forceRebuild(int numTemporaryDirectDeps) {
    checkState(numTemporaryDirectDeps + externalDeps == signaledDeps, this);
    switch (dirtyState) {
      case CHECK_DEPENDENCIES:
        checkState(getNumOfGroupsInLastBuildDirectDeps() == dirtyDirectDepIndex, this);
        dirtyState = DirtyState.REBUILDING;
        break;
      case NEEDS_REBUILDING: // Valid for NonIncrementalInMemoryNodeEntry.
      case NEEDS_FORCED_REBUILDING: // Valid for IncrementalInMemoryNodeEntry.
        dirtyState = DirtyState.REBUILDING;
        break;
      default:
        throw new IllegalStateException("Unexpected dirty state " + dirtyState + ": " + this);
    }
  }

  final boolean isEvaluating() {
    return signaledDeps > NOT_EVALUATING_SENTINEL;
  }

  final boolean isChanged() {
    return dirtyState == DirtyState.NEEDS_REBUILDING
        || dirtyState == DirtyState.NEEDS_FORCED_REBUILDING
        || dirtyState == DirtyState.REBUILDING;
  }

  private void checkFinishedBuildingWhenAboutToSetValue() {
    checkState(
        dirtyState == DirtyState.VERIFIED_CLEAN || dirtyState == DirtyState.REBUILDING,
        "not done building %s",
        this);
  }

  /**
   * Signals that a child is done.
   *
   * <p>If this node is not yet known to need rebuilding, sets {@link #dirtyState} to {@link
   * DirtyState#NEEDS_REBUILDING} if the child has changed, and {@link DirtyState#VERIFIED_CLEAN} if
   * the child has not changed and this was the last child to be checked (as determined by {@code
   * isReady} and comparing {@link #dirtyDirectDepIndex} and {@link
   * DirtyBuildingState#getNumOfGroupsInLastBuildDirectDeps()}.
   */
  final void signalDep(
      AbstractInMemoryNodeEntry<?> entry,
      NodeVersion version,
      Version childVersion,
      @Nullable SkyKey childForDebugging) {
    checkState(isEvaluating(), "%s %s", entry, childForDebugging);
    signaledDeps++;
    if (isChanged()) {
      return;
    }

    // childVersion > version.lastEvaluated() means the child has changed since the last evaluation.
    boolean childChanged = !childVersion.atMost(version.lastEvaluated());
    if (childChanged) {
      dirtyState = DirtyState.NEEDS_REBUILDING;
    } else if (dirtyState == DirtyState.CHECK_DEPENDENCIES
        && isReady(entry.getNumTemporaryDirectDeps())
        && getNumOfGroupsInLastBuildDirectDeps() == dirtyDirectDepIndex) {
      // No other dep already marked this as NEEDS_REBUILDING, no deps outstanding, and this was the
      // last block of deps to be checked.
      dirtyState = DirtyState.VERIFIED_CLEAN;
    }
  }

  final void addExternalDep() {
    checkState(isEvaluating());
    externalDeps++;
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
  final boolean depsUnchangedFromLastBuild(GroupedDeps directDeps) throws InterruptedException {
    checkFinishedBuildingWhenAboutToSetValue();
    return getLastBuildDirectDeps().equals(directDeps);
  }

  final boolean noDepsLastBuild() {
    return getNumOfGroupsInLastBuildDirectDeps() == 0;
  }

  /** Returns the {@link DirtyState} as documented by {@link NodeEntry#getDirtyState}. */
  final DirtyState getDirtyState() {
    return dirtyState;
  }

  /**
   * Gets the next children to be re-evaluated to see if this dirty node needs to be re-evaluated.
   *
   * <p>See {@link NodeEntry#getNextDirtyDirectDeps}.
   */
  final List<SkyKey> getNextDirtyDirectDeps() throws InterruptedException {
    checkState(dirtyState == DirtyState.CHECK_DEPENDENCIES, this);
    checkState(dirtyDirectDepIndex < getNumOfGroupsInLastBuildDirectDeps(), this);
    return getLastBuildDirectDeps().getDepGroup(dirtyDirectDepIndex++);
  }

  /**
   * Returns the remaining direct deps that have not been checked. If {@code preservePosition} is
   * true, this method is non-mutating. If {@code preservePosition} is false, the caller must
   * process the returned set, and so subsequent calls to this method will return the empty set.
   */
  final ImmutableSet<SkyKey> getAllRemainingDirtyDirectDeps(boolean preservePosition)
      throws InterruptedException {
    if (getLastBuildDirectDeps() == null) {
      return ImmutableSet.of();
    }
    ImmutableSet.Builder<SkyKey> result = ImmutableSet.builder();
    for (int ind = dirtyDirectDepIndex; ind < getNumOfGroupsInLastBuildDirectDeps(); ind++) {
      result.addAll(getLastBuildDirectDeps().getDepGroup(ind));
    }
    if (!preservePosition) {
      dirtyDirectDepIndex = getNumOfGroupsInLastBuildDirectDeps();
    }
    return result.build();
  }

  /**
   * Resets counters that track evaluation state. May only be called when its corresponding node has
   * no outstanding unsignaled deps, because otherwise this resetting and that signalling would
   * race.
   */
  final void resetForRestartFromScratch() {
    checkState(dirtyState == DirtyState.REBUILDING, this);
    signaledDeps = 0;
    externalDeps = 0;
    dirtyDirectDepIndex = 0;

    // Resets the evaluation count.
    int snapshot;
    do {
      snapshot = priority;
    } while (!UNSAFE.compareAndSwapInt(
        this, PRIORITY_OFFSET, snapshot, snapshot & ~EVALUATION_COUNT_MASK));
  }

  protected void markRebuilding() {
    checkState(dirtyState == DirtyState.NEEDS_REBUILDING, this);
    dirtyState = DirtyState.REBUILDING;
  }

  final void startEvaluating() {
    checkState(!isEvaluating(), this);
    signaledDeps = 0;
  }

  /** Returns whether all known children of this node have signaled that they are done. */
  final boolean isReady(int numDirectDeps) {
    // Avoids calling Preconditions.checkState because it showed up in garbage profiles due to
    // boxing of the int format args.
    if (signaledDeps > numDirectDeps + externalDeps) {
      throw new IllegalStateException(String.format("%s %s %s", numDirectDeps, externalDeps, this));
    }
    return signaledDeps == numDirectDeps + externalDeps;
  }

  /**
   * A bound on depth to avoid overflow.
   *
   * <p>The maximum observed depth in our benchmark is ~150.
   */
  @VisibleForTesting static final int DEPTH_SATURATION_BOUND = 512;

  /**
   * A bound on evaluation count to avoid overflow.
   *
   * <p>The maximum observed evaluation count in our benchmark is 4.
   */
  @VisibleForTesting static final int EVALUATION_COUNT_SATURATION_BOUND = 32;

  @Override
  public final int getPriority() {
    int snapshot = priority;

    // Fanout occupies the top-most bit. Shifts it over one bit so later computations don't need to
    // consider negative priority values. Since this is the highest set bit, low-fanout nodes
    // always have higher priority than high-fanout nodes.
    int fanoutAdjustment = (snapshot & HAS_LOW_FANOUT_MASK) >>> 1;
    int depth = min((snapshot & DEPTH_MASK) >> DEPTH_BIT_OFFSET, DEPTH_SATURATION_BOUND);
    int evaluationCount = min(snapshot & EVALUATION_COUNT_MASK, EVALUATION_COUNT_SATURATION_BOUND);

    // This formula was found to produce good results in our benchmark. There's no deep rationale
    // behind it. It's likely possible to improve it, but iterating is slow.
    return fanoutAdjustment + depth + evaluationCount * evaluationCount;
  }

  @Override
  public final int depth() {
    return (priority & DEPTH_MASK) >> DEPTH_BIT_OFFSET;
  }

  @Override
  public final void updateDepthIfGreater(int proposedDepth) {
    // Shifts the input for comparison instead of the snapshot, which might otherwise need to be
    // shifted repeatedly.
    proposedDepth = (proposedDepth << DEPTH_BIT_OFFSET) & DEPTH_MASK;
    int snapshot;
    do {
      snapshot = priority;
      if (proposedDepth <= (snapshot & DEPTH_MASK)) {
        return;
      }
    } while (!UNSAFE.compareAndSwapInt(
        this, PRIORITY_OFFSET, snapshot, (snapshot & ~DEPTH_MASK) | proposedDepth));
  }

  @Override
  public final void incrementEvaluationCount() {
    UNSAFE.getAndAddInt(this, PRIORITY_OFFSET, ONE_EVALUATION);
  }

  protected MoreObjects.ToStringHelper getStringHelper() {
    int snapshot = priority;
    return MoreObjects.toStringHelper(this)
        .add("dirtyState", dirtyState)
        .add("signaledDeps", signaledDeps)
        .add("externalDeps", externalDeps)
        .add("dirtyDirectDepIndex", dirtyDirectDepIndex)
        .add("has low fanout", (snapshot & HAS_LOW_FANOUT_MASK) != 0)
        .add("depth", (snapshot & DEPTH_MASK) >> DEPTH_BIT_OFFSET)
        .add("evaluation count", (snapshot & EVALUATION_COUNT_MASK));
  }

  @Override
  public final String toString() {
    return getStringHelper().toString();
  }

  // Masks for `priority`.
  private static final int HAS_LOW_FANOUT_MASK = 0x8000_0000;

  private static final int DEPTH_MASK = 0x7FFF_0000;
  private static final int DEPTH_BIT_OFFSET = 16;

  private static final int EVALUATION_COUNT_MASK = 0xFFFF;
  private static final int ONE_EVALUATION = 1;

  static {
    checkState(Integer.numberOfTrailingZeros(DEPTH_MASK) == DEPTH_BIT_OFFSET);
    checkState(
        Integer.numberOfTrailingZeros(EVALUATION_COUNT_MASK)
            == Integer.numberOfTrailingZeros(ONE_EVALUATION));
  }

  private static final Unsafe UNSAFE = UnsafeProvider.unsafe();

  private static final long PRIORITY_OFFSET;

  static {
    try {
      PRIORITY_OFFSET =
          UNSAFE.objectFieldOffset(DirtyBuildingState.class.getDeclaredField("priority"));
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }
}
