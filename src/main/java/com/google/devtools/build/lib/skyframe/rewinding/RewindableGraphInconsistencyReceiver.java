// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.rewinding;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multiset;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.skyframe.NodeDroppingInconsistencyReceiver;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.InconsistencyStats;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.InconsistencyStats.InconsistencyStat;
import java.util.Collection;
import java.util.function.Predicate;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * {@link GraphInconsistencyReceiver} for evaluations that support action rewinding ({@code
 * --rewind_lost_inputs}).
 *
 * <p>Action rewinding results in various kinds of inconsistencies which this receiver tolerates.
 * The first occurrence of each type of tolerated inconsistency is logged. Stats are collected and
 * available through {@link #getInconsistencyStats}.
 *
 * <p>{@link #reset} should be called between commands to clear stats and reset the {@link
 * #rewindingInitiated} state used for consistency checks.
 */
public final class RewindableGraphInconsistencyReceiver implements GraphInconsistencyReceiver {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final int LOGGED_CHILDREN_LIMIT = 50;

  private final Multiset<Inconsistency> selfCounts = ConcurrentHashMultiset.create();
  private final Multiset<Inconsistency> childCounts = ConcurrentHashMultiset.create();
  private boolean rewindingInitiated = false;
  private final boolean heuristicallyDropNodes;
  private final boolean skymeldInconsistenciesExpected;

  public RewindableGraphInconsistencyReceiver(
      boolean heuristicallyDropNodes, boolean skymeldInconsistenciesExpected) {
    this.heuristicallyDropNodes = heuristicallyDropNodes;
    this.skymeldInconsistenciesExpected = skymeldInconsistenciesExpected;
  }

  @Override
  public void noteInconsistencyAndMaybeThrow(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    if (heuristicallyDropNodes
        && NodeDroppingInconsistencyReceiver.isExpectedInconsistency(
            key, otherKeys, inconsistency)) {
      // If `--heuristically_drop_nodes` is enabled, check whether the inconsistency is caused by
      // dropped state node. If so, tolerate the inconsistency and return.
      return;
    }

    if (skymeldInconsistenciesExpected
        && NodeDroppingInconsistencyReceiver.isExpectedInconsistencySkymeld(
            key, otherKeys, inconsistency)) {
      return;
    }

    // RESET_REQUESTED and PARENT_FORCE_REBUILD_OF_CHILD may be the first inconsistencies seen with
    // rewinding. BUILDING_PARENT_FOUND_UNDONE_CHILD may also be seen, but it will not be the first.
    switch (inconsistency) {
      case RESET_REQUESTED:
        checkState(
            RewindingInconsistencyUtils.isTypeThatDependsOnRewindableNodes(key),
            "Unexpected reset requested for: %s",
            key);
        boolean isFirst = noteSelfInconsistency(inconsistency);
        if (isFirst) {
          logger.atInfo().log("Reset requested for: %s", key);
        }
        rewindingInitiated = true;
        return;

      case PARENT_FORCE_REBUILD_OF_CHILD:
        boolean parentMayForceRebuildChildren =
            RewindingInconsistencyUtils.mayForceRebuildChildren(key);
        ImmutableList<SkyKey> unrewindableRebuildChildren =
            otherKeys.stream()
                .filter(Predicate.not(RewindingInconsistencyUtils::isRewindable))
                .collect(toImmutableList());
        checkState(
            parentMayForceRebuildChildren && unrewindableRebuildChildren.isEmpty(),
            "Unexpected force rebuild, parent = %s, children = %s",
            key,
            listChildren(parentMayForceRebuildChildren ? unrewindableRebuildChildren : otherKeys));
        isFirst = noteSelfInconsistency(inconsistency);
        childCounts.add(inconsistency, otherKeys.size());
        if (isFirst) {
          logger.atInfo().log(
              "Parent force rebuild of children: parent = %s, children = %s",
              key, listChildren(otherKeys));
        }
        rewindingInitiated = true;
        return;

      case BUILDING_PARENT_FOUND_UNDONE_CHILD:
        boolean parentDependsOnRewindableNodes =
            RewindingInconsistencyUtils.isTypeThatDependsOnRewindableNodes(key);
        ImmutableList<SkyKey> unrewindableUndoneChildren =
            otherKeys.stream()
                .filter(Predicate.not(RewindingInconsistencyUtils::isRewindable))
                .collect(toImmutableList());
        checkState(
            rewindingInitiated
                && parentDependsOnRewindableNodes
                && unrewindableUndoneChildren.isEmpty(),
            "Unexpected undone children: parent = %s, children = %s",
            key,
            listChildren(
                rewindingInitiated && parentDependsOnRewindableNodes
                    ? unrewindableUndoneChildren
                    : otherKeys));
        isFirst = noteSelfInconsistency(inconsistency);
        childCounts.add(inconsistency, otherKeys.size());
        if (isFirst) {
          logger.atInfo().log(
              "Building parent found undone children: parent = %s, children = %s",
              key, listChildren(otherKeys));
        }
        return;

      default:
        throw new IllegalStateException(
            String.format(
                "Unexpected inconsistency %s, key = %s, otherKeys = %s",
                inconsistency, key, listChildren(otherKeys)));
    }
  }

  /**
   * Returns an object suitable for use as a string format arg in precondition checks or logger
   * statements.
   */
  private static Object listChildren(@Nullable Collection<SkyKey> children) {
    if (children == null) {
      return "null";
    }
    if (children.size() <= LOGGED_CHILDREN_LIMIT) {
      return children;
    }
    return new Object() {
      @Override
      public String toString() {
        return StringUtil.listItemsWithLimit(new StringBuilder(), LOGGED_CHILDREN_LIMIT, children)
            .toString();
      }
    };
  }

  /**
   * Notes in {@link #selfCounts} that an inconsistency occurred and returns true if it was the
   * first one detected.
   */
  private boolean noteSelfInconsistency(Inconsistency inconsistency) {
    return selfCounts.add(inconsistency, 1) == 0;
  }

  @Override
  public InconsistencyStats getInconsistencyStats() {
    InconsistencyStats.Builder builder = InconsistencyStats.newBuilder();
    addInconsistencyStats(selfCounts, builder::addSelfStatsBuilder);
    addInconsistencyStats(childCounts, builder::addChildStatsBuilder);
    return builder.build();
  }

  private static void addInconsistencyStats(
      Multiset<Inconsistency> inconsistencies,
      Supplier<InconsistencyStat.Builder> builderSupplier) {
    inconsistencies.forEachEntry(
        (inconsistency, count) ->
            builderSupplier.get().setInconsistency(inconsistency).setCount(count));
  }

  @Override
  public void reset() {
    selfCounts.clear();
    childCounts.clear();
    rewindingInitiated = false;
  }
}
