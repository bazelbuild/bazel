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

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.util.Collection;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * {@link GraphInconsistencyReceiver} for evaluations operating on graphs that support rewinding (no
 * reverse dependencies, no action cache).
 *
 * <p>Action rewinding results in various kinds of inconsistencies which this receiver tolerates.
 */
public final class RewindableGraphInconsistencyReceiver implements GraphInconsistencyReceiver {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private boolean rewindingInitiated = false;

  @Override
  public void noteInconsistencyAndMaybeThrow(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    String childrenAsString =
        otherKeys != null ? GraphInconsistencyReceiver.listChildren(otherKeys) : "null";

    // RESET_REQUESTED and PARENT_FORCE_REBUILD_OF_CHILD may be the first inconsistencies seen with
    // rewinding. BUILDING_PARENT_FOUND_UNDONE_CHILD may also be seen, but it will not be the first.
    switch (inconsistency) {
      case RESET_REQUESTED:
        checkState(
            RewindingInconsistencyUtils.isTypeThatDependsOnRewindableNodes(key),
            "Unexpected reset requested for: %s",
            key);
        logger.atInfo().log("Reset requested for: %s", key);
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
            parentMayForceRebuildChildren
                ? GraphInconsistencyReceiver.listChildren(unrewindableRebuildChildren)
                : childrenAsString);
        logger.atInfo().log(
            "Parent force rebuild of children: parent = %s, children = %s", key, childrenAsString);
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
            rewindingInitiated && parentDependsOnRewindableNodes
                ? GraphInconsistencyReceiver.listChildren(unrewindableUndoneChildren)
                : childrenAsString);
        logger.atInfo().log(
            "Building parent found undone children: parent = %s, children = %s",
            key, childrenAsString);
        return;

      case PARENT_FORCE_REBUILD_OF_MISSING_CHILD:
      case DIRTY_PARENT_HAD_MISSING_CHILD:
      case ALREADY_DECLARED_CHILD_MISSING:
        throw new IllegalStateException(
            String.format(
                "Unexpected inconsistency %s, key = %s, otherKeys = %s ",
                inconsistency, key, childrenAsString));
      default: // Needed because protobuf creates additional enum values.
        throw new IllegalStateException(
            String.format(
                "Unknown inconsistency %s, key = %s, otherKeys = %s ",
                inconsistency, key, childrenAsString));
    }
  }

  @Override
  public boolean restartPermitted() {
    return true;
  }
}
