// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentNavigableMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;

class ArtifactConflictFinder {
  static final Precomputed<ImmutableMap<ActionAnalysisMetadata, ConflictException>>
      ACTION_CONFLICTS = new Precomputed<>("action_conflicts");

  private ArtifactConflictFinder() {}

  /**
   * Find conflicts between generated artifacts. There are two ways to have conflicts. First, if two
   * (unshareable) actions generate the same output artifact, this will result in an {@link
   * ActionConflictException}. Second, if one action generates an artifact whose path is a prefix of
   * another artifact's path, those two artifacts cannot exist simultaneously in the output tree.
   * This causes an {@link ArtifactPrefixConflictException}.
   *
   * <p>This method must be called if a new action was added to the graph this build, so whenever a
   * new configured target was analyzed this build. It is somewhat expensive (~1s range for a medium
   * build as of 2014), so it should only be called when necessary.
   */
  static ImmutableMap<ActionAnalysisMetadata, ConflictException> findAndStoreArtifactConflicts(
      Iterable<ActionLookupValue> actionLookupValues,
      boolean strictConflictChecks,
      ActionKeyContext actionKeyContext)
      throws InterruptedException {
    ConcurrentMap<ActionAnalysisMetadata, ConflictException> temporaryBadActionMap =
        new ConcurrentHashMap<>();
    Pair<ActionGraph, SortedMap<PathFragment, Artifact>> result;
    result =
        constructActionGraphAndPathMap(actionKeyContext, actionLookupValues, temporaryBadActionMap);
    ActionGraph actionGraph = result.first;
    SortedMap<PathFragment, Artifact> artifactPathMap = result.second;

    Map<ActionAnalysisMetadata, ArtifactPrefixConflictException> actionsWithArtifactPrefixConflict =
        Actions.findArtifactPrefixConflicts(actionGraph, artifactPathMap, strictConflictChecks);
    for (Map.Entry<ActionAnalysisMetadata, ArtifactPrefixConflictException> actionExceptionPair :
        actionsWithArtifactPrefixConflict.entrySet()) {
      temporaryBadActionMap.put(
          actionExceptionPair.getKey(), new ConflictException(actionExceptionPair.getValue()));
    }
    return ImmutableMap.copyOf(temporaryBadActionMap);
  }

  /**
   * Simultaneously construct an action graph for all the actions in Skyframe and a map from {@link
   * PathFragment}s to their respective {@link Artifact}s. We do this in a threadpool to save around
   * 1.5 seconds on a mid-sized build versus a single-threaded operation.
   */
  private static Pair<ActionGraph, SortedMap<PathFragment, Artifact>>
      constructActionGraphAndPathMap(
          ActionKeyContext actionKeyContext,
          Iterable<ActionLookupValue> values,
          ConcurrentMap<ActionAnalysisMetadata, ConflictException> badActionMap)
          throws InterruptedException {
    MutableActionGraph actionGraph = new MapBasedActionGraph(actionKeyContext);
    ConcurrentNavigableMap<PathFragment, Artifact> artifactPathMap =
        new ConcurrentSkipListMap<>(Actions.comparatorForPrefixConflicts());
    // Action graph construction is CPU-bound.
    int numJobs = Runtime.getRuntime().availableProcessors();
    // No great reason for expecting 5000 action lookup values, but not worth counting size of
    // values.
    Sharder<ActionLookupValue> actionShards = new Sharder<>(numJobs, 5000);
    for (ActionLookupValue value : values) {
      actionShards.add(value);
    }

    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper(
            "ArtifactConflictFinder#constructActionGraphAndPathMap");

    ExecutorService executor =
        Executors.newFixedThreadPool(
            numJobs,
            new ThreadFactoryBuilder().setNameFormat("ActionLookupValue Processor %d").build());
    for (List<ActionLookupValue> shard : actionShards) {
      executor.execute(
          wrapper.wrap(actionRegistration(shard, actionGraph, artifactPathMap, badActionMap)));
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      throw new InterruptedException();
    }
    return Pair.of(actionGraph, artifactPathMap);
  }

  private static Runnable actionRegistration(
      final List<ActionLookupValue> values,
      final MutableActionGraph actionGraph,
      final ConcurrentMap<PathFragment, Artifact> artifactPathMap,
      final ConcurrentMap<ActionAnalysisMetadata, ConflictException> badActionMap) {
    return () -> {
      for (ActionLookupValue value : values) {
        for (ActionAnalysisMetadata action : value.getActions()) {
          try {
            actionGraph.registerAction(action);
          } catch (ActionConflictException e) {
            // It may be possible that we detect a conflict for the same action more than once, if
            // that action belongs to multiple aspect values. In this case we will harmlessly
            // overwrite the badActionMap entry.
            badActionMap.put(action, new ConflictException(e));
            // We skip the rest of the loop, and do not add the path->artifact mapping for this
            // artifact below -- we don't need to check it since this action is already in
            // error.
            continue;
          } catch (InterruptedException e) {
            // Bail.
            Thread.currentThread().interrupt();
            return;
          }
          for (Artifact output : action.getOutputs()) {
            artifactPathMap.put(output.getExecPath(), output);
          }
        }
      }
    };
  }

  /**
   * A typed union of {@link ActionConflictException}, which indicates two actions that generate the
   * same {@link Artifact}, and {@link ArtifactPrefixConflictException}, which indicates that the
   * path of one {@link Artifact} is a prefix of another.
   */
  static class ConflictException extends Exception {
    @Nullable private final ActionConflictException ace;
    @Nullable private final ArtifactPrefixConflictException apce;

    ConflictException(ActionConflictException e) {
      super(e);
      this.ace = e;
      this.apce = null;
    }

    ConflictException(ArtifactPrefixConflictException e) {
      super(e);
      this.ace = null;
      this.apce = e;
    }

    void rethrowTyped() throws ActionConflictException, ArtifactPrefixConflictException {
      if (ace == null) {
        throw Preconditions.checkNotNull(apce);
      }
      if (apce == null) {
        throw Preconditions.checkNotNull(ace);
      }
      throw new IllegalStateException();
    }
  }
}
