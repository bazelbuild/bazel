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
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.ArtifactMetadataCache;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * An implementation of DependencyChecker and extension of the
 * DatabaseDependencyChecker. This class differs from its superclass in the way
 * it determines if an action needs to be executed in the course of a build.
 *
 * <p>The IncrementalDependencyChecker uses information cached by the
 * ArtifactMTimeCache to avoid doing checks to the action cache unless
 * needed. By knowing the last modification times of artifacts from the previous
 * build, and by constructing a dependency graph, this class can mark artifacts
 * as "dirty" if their mtime changed and propagate this status to any artifacts
 * depending on a dirty one. The dirty state signals that an action may need
 * to be re-run due to changed inputs or outputs, not that it will necessarily
 * be executed. On the other hand, a non-dirty action does not need to be
 * re-executed.
 *
 * <p>Note: the correctness of the results is dependent on the build being
 * executed being incremental, as defined by the {@code isApplicableToBuild}
 * method of the ArtifactMTimeCache class. Namely, the current top-level
 * artifacts requested to build need to be the same as the artifacts of the
 * previous build. The correctness also depends on all actions in the action
 * cache that reference any artifact having the same metadata cached for that
 * artifact. In other words, if any artifacts were externally modified in the
 * previous build such that different actions saw different file metadata, the
 * correctness of the IncrementalDependencyChecker on those files is not guaranteed.
 */
@ThreadSafe
public class IncrementalDependencyChecker extends DatabaseDependencyChecker {

  private ArtifactMTimeCache mTimeCache;
  private Set<Action> dirtyActions;
  private boolean initialized;
  private final EventBus eventBus;

  // The dependent action graph actually used by the builder to execute the build.
  // This may or may not be the same as the forwardGraph passed into init (see
  // maybeCullGraph for details). We maintain separate references to this and
  // dependencyGraph because, while this is the only relevant graph for build
  // execution, dependencyGraph may live across builds (see StableDependentActionGraph)
  // and provide a base for future executionGraph constructions. We need to make sure
  // the persistent graph receives beforeChange updates from discovered header
  // changes, not the transient graph.
  private DependentActionGraph executionGraph;

  private final boolean graphPruningEnabled;
  private long workSavedByGraphCulling = 0;

  public IncrementalDependencyChecker(ActionCache actionCache,
                                      ArtifactResolver artifactResolver,
                                      ArtifactMetadataCache artifactMetadataCache,
                                      ArtifactMTimeCache artifactMTimeCache,
                                      PackageUpToDateChecker packageUpToDateChecker,
                                      Predicate<? super Action> executionFilter,
                                      EventBus eventBus,
                                      boolean verboseExplanations,
                                      boolean enableGraphPruning,
                                      Predicate<PathFragment> allowedMissingInputs) {
    super(actionCache, artifactResolver, artifactMetadataCache, packageUpToDateChecker,
        executionFilter, verboseExplanations, allowedMissingInputs);
    this.mTimeCache = artifactMTimeCache;
    this.dirtyActions = Sets.newConcurrentHashSet();
    this.executionGraph = null;
    this.initialized = false;
    this.eventBus = eventBus;
    this.graphPruningEnabled = enableGraphPruning;
  }

  /**
   * Generates the dependency multimap and set of dirty actions given the
   * set of top-level artifacts the current build is requesting to be built.
   *
   * <p>Note: this must be executed before the build starts, since it needs
   * access to {@code changedArtifacts()} of the ArtifactMTimeCache. Optimally,
   * this is to be run immediately after initialization of this object.
   *
   * Not intended to be run concurrently.
   */
  @ThreadCompatible
  @Override
  public void init(Set<Artifact> topLevelArtifacts, Set<Artifact> builtArtifacts,
      DependentActionGraph forwardGraph, Executor executor, ModifiedFileSet modified,
      ErrorEventListener listener)
          throws InterruptedException {
    super.init(topLevelArtifacts, builtArtifacts, forwardGraph, executor, modified, listener);
    // We dirty all actions that were not successfully executed last build.
    dirtyActions.addAll(forwardGraph.getStaleActions());
    Set<Artifact> artifactsKnownBad = new HashSet<>();
    for (Action action : dirtyActions) {
      artifactsKnownBad.addAll(action.getOutputs());
    }
    Set<Artifact> changedArtifacts =
        mTimeCache.changedArtifacts(dependencyGraph.getArtifacts(), modified, artifactsKnownBad);
    postChangedArtifacts(changedArtifacts);

    maybeCullGraph(topLevelArtifacts, changedArtifacts, executor, builtArtifacts);
    String unreadFiles = "";
    String unsuccessfulActions = "";
    if (!dirtyActions.isEmpty()) {
      unreadFiles = " or unprocessed";
      unsuccessfulActions =
          " and " + dirtyActions.size() + " actions that were not successfully executed last run";
    }
    listener.info(null, String.format("Found %d modified" +
        unreadFiles + " files (using %s dependency graph)" + unsuccessfulActions,
        changedArtifacts.size() - artifactsKnownBad.size(),
        executionGraph != dependencyGraph ? "pruned" : "full"));
    for (Artifact artifact : changedArtifacts) {
      markDirty(actionGraph.getGeneratingAction(artifact));
      for (DependentAction dependency :
           executionGraph.getArtifactDependenciesInternal(artifact)) {
        markDirty(dependency.getAction());
      }
    }
    forwardGraph.markActionsStale(getActionGraphForBuild().getActions());
    // If a build is interrupted at this line, then the set of changed artifacts returned
    // by the next call to mTimeCache.updateCache() will redundantly include inputs to
    // actions already marked stale. However, a subsequent successful build will fix it.
    // See DependentActionGraph#staleActions for more.
    mTimeCache.clearChangedArtifacts();
    initialized = true;
  }

  private void postChangedArtifacts(Set<Artifact> changedArtifacts) {
    eventBus.post(new ChangedArtifactsMessage(Collections.unmodifiableSet(changedArtifacts)));
  }

  /**
   * Adds an action to the list of dirty actions.
   * Some artifacts have null generating actions; for these, this is a no-op.
   */
  private void markDirty(Action action) {
    if (action != null) {
      dirtyActions.add(action);
    }
  }

  @Override
  public long getWorkSavedByDependencyChecker() {
    return workSavedByGraphCulling;
  }

  /**
   * Replaces the dependent action graph with its subgraph of "dirty" actions
   * and artifacts (as defined in the class comments), or leaves the graph
   * as-is if the dirty part would be large enough such that computing / using
   * it offers no performance advantage over using the full graph.
   *
   * For null and incremental builds, this subgraph is often substantially
   * smaller than the original graph and saves us much of the cost of having to
   * visit every graph edge during execution. However, there is also a cost
   * involved in constructing the subgraph, so the larger the subgraph is
   * compared to the full graph the less savings we get. Past some threshold, a
   * net performance loss is incurred. We try to avoid that threshold.
   */
  private void maybeCullGraph(Set<Artifact> topLevelArtifacts, Set<Artifact> changedArtifacts,
      Executor executor, Set<Artifact> builtArtifacts) {
    Preconditions.checkNotNull(dependencyGraph);
    executionGraph = dependencyGraph;
    if (!graphPruningEnabled) {
      return;
    }

    // The set of top-level artifacts connected to one or more dirty paths.
    Set<Artifact> dirtyTopLevelArtifacts = new HashSet<>();
    // The set of actions that need to be part of the dirty graph.
    Set<Action> dirtyActions = new HashSet<>();
    // The set of artifacts that need to be part of the dirty graph.
    Set<Artifact> dirtyArtifacts = new HashSet<>();
    // The set of NotifyOnActionCacheHit actions in the full graph. For these
    // actions, actionCacheHit needs to be called exactly once if execution would
    // hit the action cache, regardless of the action's dirtiness.
    Set<NotifyOnActionCacheHit> notifyOnActionCacheHitActions;
    // Cumulative workload of all non-dirty actions.
    long nonDirtyActionWorkload = dependencyGraph.estimateTotalWorkload();
    long dirtyEdges = 0;
    final long totalEdgeCount = dependencyGraph.getTotalEdgeCount();

    // Initialize the list of dirty actions to traverse with all of the full graph's
    // "volatile" actions. These actions are automatically dirty, regardless of
    // the dirtiness of their inputs. Also populate notifyOnActionCacheHitActions.
    Queue<Action> actionsToTraverse = new LinkedList<>();
    actionsToTraverse.addAll(dependencyGraph.getVolatileActions());
    notifyOnActionCacheHitActions = Sets.newHashSet(dependencyGraph.getCacheHitActions());

    // For each changed artifact: if it's a source artifact, mark it as dirty and
    // enqueue its dependent actions for traversal. If it's a generated artifact,
    // just enqueue its generating action for traversal (this will automatically
    // reach the artifact and its dependencies as the subgraph is traversed).
    for (Artifact changedArtifact : changedArtifacts) {
      if (changedArtifact.isSourceArtifact()) {
        dirtyArtifacts.add(changedArtifact);
        dirtyEdges += enqueueArtifactDependents(changedArtifact, dependencyGraph,
            actionsToTraverse, dirtyTopLevelArtifacts);
        if (dirtyGraphExceedsSizeThreshold(dirtyEdges, totalEdgeCount)) {
          return;
        }
      } else {
        Action action = actionGraph.getGeneratingAction(changedArtifact);
        Preconditions.checkState(action != null, action);
        actionsToTraverse.add(action);
      }
    }

    // Consume the dirty action queue, marking action outputs as dirty and enqueueing
    // their dependent actions, thus propagating "dirtiness" up the graph.
    while (!actionsToTraverse.isEmpty()) {
      Action dirtyAction = actionsToTraverse.remove();
      if (!dirtyActions.add(dirtyAction)) {
        continue;
      }
      nonDirtyActionWorkload -= dirtyAction.estimateWorkload();
      Collection<Artifact> outputs = dirtyAction.getOutputs();
      dirtyArtifacts.addAll(outputs);
      dirtyEdges += outputs.size();
      for (Artifact output : outputs) {
        dirtyEdges += enqueueArtifactDependents(output, dependencyGraph,
            actionsToTraverse, dirtyTopLevelArtifacts);
        if (dirtyGraphExceedsSizeThreshold(dirtyEdges, totalEdgeCount)) {
          return;
        }
      }
    }

    // At the start of execution, ParallelBuilder automatically marks source
    // top-level artifacts as built and decrements the top-level action's
    // unbuilt inputs count accordingly. We need to make sure this relationship
    // is preserved in the subgraph, even if the artifacts aren't dirty, so
    // the top-level action's unbuiltInputs is initialized to the expected value.
    for (Artifact topLevelArtifact : topLevelArtifacts) {
      if (topLevelArtifact.isSourceArtifact()) {
        dirtyTopLevelArtifacts.add(topLevelArtifact);
      }
    }

    // Construct our subgraph. Also traverse the subgraph's actions, decrementing
    // the "unbuilt input" count for each non-dirty generated input (ParallelBuilder
    // will handle the source inputs).
    DependentActionGraph subgraph = DependentActionGraph.newGraph(dirtyTopLevelArtifacts,
        actionGraph, false, new DirtyActionVisitor(dirtyActions, dirtyArtifacts), false);
    for (DependentAction depAction : subgraph.getDependentActions()) {
      for (Artifact inputArtifact : depAction.getAction().getInputs()) {
        if (!dirtyArtifacts.contains(inputArtifact) && !inputArtifact.isSourceArtifact()) {
          depAction.builtOneInput();
        }
      }
    }

    // Mark all non-dirty top-level artifacts as (trivially) "built".
    if (builtArtifacts != null) {
      for (Artifact topLevelArtifact : topLevelArtifacts) {
        if (!dirtyArtifacts.contains(topLevelArtifact)) {
          builtArtifacts.add(topLevelArtifact);
        }
      }
    }

    // Call actionCacheHit on all non-dirty NotifyOnActionCacheHit actions (ParallelBuilder
    // will call the dirty ones during the normal course of execution).
    for (NotifyOnActionCacheHit notifyAction : notifyOnActionCacheHitActions) {
      if (!dirtyActions.contains(notifyAction)) {
        notifyAction.actionCacheHit(executor);
      }
    }

    executionGraph = subgraph;
    workSavedByGraphCulling = nonDirtyActionWorkload;
  }

  @Override
  public DependentActionGraph getActionGraphForBuild() {
    return executionGraph;
  }

  /**
   * Helper routine for getDirtyGraph. Given an artifact, examines all
   * its dependent actions. If an action is a top-level action, adds this artifact
   * to the set of top-level artifacts. Else adds the action to the traversal
   * queue. Returns the number of edges visited as a result of this examination.
   */
  private int enqueueArtifactDependents(Artifact artifact, DependentActionGraph dependencyGraph,
      Queue<Action> actionsToTraverse, Set<Artifact> topLevelArtifacts) {
    Collection<DependentAction> depActions = dependencyGraph.getArtifactDependencies(artifact);
    DependentAction topLevelAction = dependencyGraph.getTopLevelAction();
    for (DependentAction depAction : depActions) {
      if (depAction == topLevelAction) {
        topLevelArtifacts.add(artifact);
      } else {
        actionsToTraverse.add(depAction.getAction());
      }
    }
    return depActions.size();
  }

  /**
   * Returns true if the dirty graph size is a large enough percentage of the
   * full graph size such that constructing the dirty graph as a separate
   * entity isn't worth the time/effort it requires.
   *
   * The condition applied here is chosen heuristically based on simple timing
   * experiments with null builds of test projects.
   */
  private boolean dirtyGraphExceedsSizeThreshold(long dirtyEdges, long allEdges) {
    return dirtyEdges * 10 >= allEdges;
  }

  /**
   * A visitor for {@link DependentActionGraph} construction that only visits
   * dirty actions and artifacts.
   */
  private static final class DirtyActionVisitor implements DependentActionVisitor {
    private final Set<Action> dirtyActions;
    private final Set<Artifact> dirtyArtifacts;

    private DirtyActionVisitor(Set<Action> dirtyActions, Set<Artifact> dirtyArtifacts) {
      this.dirtyActions = dirtyActions;
      this.dirtyArtifacts = dirtyArtifacts;
    }

    @Override
    public boolean visitDependency(Action action, Artifact outputArtifact,
        @Nullable DependentAction dependency, boolean rootArtifact, boolean dependencyVisited) {
      // This condition holds since:
      // 1. action is supposed to be the generating action of outputArtifact (see
      // DependentActionGraph#traverseRecursiveDependencies);
      // 2. Every dirty derived artifact's generating action is also dirty.
      // (2) can be verified by inspecting the above code: an artifact is dirty if either
      // a. it was passed into maybeCullGraph in changedArtifacts (in which case its generating
      //    action was added to actionsToTraverse, and then to dirtyActions);
      // b. it was the output of a dirty action, in which case its generating action is, duh, dirty.
      Preconditions.checkState(dirtyActions.contains(action),
          "%s generates dirty %s, but is not dirty", action, outputArtifact);
      return true;
    }

    @Override
    public boolean visitArtifact(Artifact artifact) {
      return dirtyArtifacts.contains(artifact);
    }

    @Override
    public boolean visitInput(Artifact inputFile, DependentAction dependency) {
      return dirtyArtifacts.contains(inputFile);
    }

    @Override
    public void visitOutputs(Collection<Artifact> outputFiles, DependentAction action) {
      // No-op. No custom activity is required here beyond the default applied
      // in DependentActionGraph.
    }
  }

  private boolean willBeExecuted(Action action, ActionCache.Entry entry,
      DepcheckerListener listener) {
    if (unconditionalExecution(action)) {
      reportUnconditionalExecution(listener, action);
      return true;
    }
    return dirtyActions.contains(action) &&
           super.mustExecute(action, entry, listener, getMetadataHandler());
  }

  @Override
  public Collection<Artifact> getMissingInputs(Action action) {
    Preconditions.checkState(initialized);
    return dirtyActions.contains(action)
        ? super.getMissingInputs(action)
        : ImmutableSet.<Artifact>of();
  }

  /**
   * Returns true if the action needs to be unconditionally executed, and false
   * if it is not marked as "dirty", or defers to the DatabaseDependencyChecker
   * to check via the action cache if the action is marked as "dirty" because
   * one of its dependencies or results was changed.
   */
  @Override
  protected boolean mustExecute(Action action, ActionCache.Entry entry,
      DepcheckerListener listener, MetadataHandler metadataHandler) {
    Preconditions.checkState(initialized);
    Preconditions.checkState(metadataHandler == getMetadataHandler(),
        "%s %s", metadataHandler, getMetadataHandler());
    boolean mustExec = willBeExecuted(action, entry, listener);
    if (mustExec) {
      markDependentActionsDirty(action);
    }
    return mustExec;
  }

  @Override
  protected void checkMiddlemanAction(Action action, DepcheckerListener listener,
      MetadataHandler metadataHandler) {
    Preconditions.checkState(initialized);
    Preconditions.checkState(metadataHandler == getMetadataHandler(),
        "%s %s", metadataHandler, getMetadataHandler());
    if (dirtyActions.contains(action)) {
      markDependentActionsDirty(action);
      super.checkMiddlemanAction(action, listener, metadataHandler);
    }
  }

  /**
   * Marks all actions depending on any output of the given action as dirty.
   */
  private void markDependentActionsDirty(Action action) {
    for (Artifact artifact : action.getOutputs()) {
      for (DependentAction dependency :
           executionGraph.getArtifactDependenciesInternal(artifact)) {
        markDirty(dependency.getAction());
      }
    }
  }
}
