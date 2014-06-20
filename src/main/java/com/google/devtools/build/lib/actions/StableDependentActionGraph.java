// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Sets;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * A reusable DependentActionGraph: This graph does not release memory or
 * data elements while in use, so that it may be reused for subsequent
 * builds.
 */
final class StableDependentActionGraph extends DependentActionGraph {

  // Thread safety: The dependency checking which discovers new input files
  // is done in parallel, so these must be thread-safe data structures.

  /**
   * A thread-safe multimap of potentially invalid edges.
   */
  private final Multimap<Artifact, DependentAction> toRemoveMap;

  /**
   * A thread-safe set of actions which need to be updated (eg, because of C++ header discovery).
   */
  private final Set<DependentAction> toUpdate;

  /**
   * Creates a new dependency graph with the specified top-level action.
   */
  StableDependentActionGraph(DependentAction topLevelAction, Set<Artifact> rootArtifacts,
      ActionGraph actionGraph) {
    super(topLevelAction, rootArtifacts, actionGraph);
    toRemoveMap = Multimaps.synchronizedListMultimap(
        ArrayListMultimap.<Artifact, DependentAction>create(100, 16));
    toUpdate = Sets.newConcurrentHashSet();
  }

  @Override
  public Collection<DependentAction> getAndRemoveArtifactMaybe(Artifact file) {
    return getArtifactDependenciesInternal(file);
  }

  @Override
  public void beforeChange(Action action) {
    DependentAction depAction = Preconditions.checkNotNull(getActionDependency(action), action);
    Preconditions.checkState(toUpdate.add(depAction), "%s, %s", depAction, action);
    workloadEstimate.addAndGet(-action.estimateWorkload());
    totalEdgeCount.addAndGet(-countActionEdges(action));
    for (Artifact input : action.getInputs()) {
      toRemoveMap.put(input, depAction);
    }
  }

  @Override
  public void clearMiddleman(Action action, Artifact middleman, Action middlemanAction) {
    Preconditions.checkState(middleman.isMiddlemanArtifact());
    artifacts.remove(middleman);
    DependentAction oldMiddlemanAction = dependentActionFromActionMap.remove(middlemanAction);
    if (oldMiddlemanAction != null) {
      for (Artifact input : middlemanAction.getInputs()) {
        dependenciesMultimap.remove(input, oldMiddlemanAction);
      }
    }
    DependentAction depAction = getActionDependency(action);
    if (depAction != null) {
      dependenciesMultimap.remove(middleman, depAction);
      depAction.reset();
    }
  }

  @Override
  public void addMiddleman(Action action, Artifact middleman) {
    Preconditions.checkState(middleman.isMiddlemanArtifact());
    artifacts.add(middleman);
    DependentAction depAction = getActionDependency(action);
    if (depAction == null) {
      return;
    }
    depAction.reset();
    addArtifact(middleman, depAction);
  }

  @Override
  public void sync() {
    // Remove edges which may be invalid, and recreate them using the canonical
    // action graph edges.
    for (Map.Entry<Artifact, DependentAction> entry : toRemoveMap.entries()) {
      removeEdge(entry.getKey(), entry.getValue());
    }
    for (DependentAction action : toUpdate) {
      workloadEstimate.addAndGet(action.getAction().estimateWorkload());
      totalEdgeCount.addAndGet(countActionEdges(action.getAction()));
      for (Artifact input : action.getAction().getInputs()) {
        addEdge(input, action);
      }
      action.reset();
    }
    toRemoveMap.clear();
    toUpdate.clear();

    // Now reset any individual dependent actions that were not reached during the
    // last build because it failed too early.
    for (Action action : getStaleActions()) {
      DependentAction depAction = Preconditions.checkNotNull(getActionDependency(action), action);
      depAction.reset();
    }
    // If last build failed or was interrupted, we need to reset the topLevelAction too. Since
    // resetting is safe, we do it unconditionally.
    getTopLevelAction().reset();
  }
}
