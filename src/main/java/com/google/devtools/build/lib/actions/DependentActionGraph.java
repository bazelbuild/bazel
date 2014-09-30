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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * Represents the dependency graph and implements methods for traversing and
 * altering the graph, including addition of artifact dependency subgraphs.
 */
@ThreadSafety.ThreadCompatible
public class DependentActionGraph {
  // See IMPORTANT NOTE about performance in ParallelBuilder class comment.
  private static final Logger LOG =
      Logger.getLogger(DependentActionGraph.class.getName());

  private final ActionGraph actionGraph;

  private final Deque<Artifact> artifactStack = new ArrayDeque<>();

  /**
   * The action responsible for building all root-level artifacts.
   */
  private final DependentAction topLevelAction;

  /**
   * Multimap of dependencies between generated input artifacts and corresponding DependentActions.
   *
   * <p>This map tells which actions (values) the Artifact (key) is an input of.
   */
  protected final Multimap<Artifact, DependentAction> dependenciesMultimap;

  /**
   * Map of actions which need to be enqueued in the builder.
   */
  protected final Map<Action, DependentAction> dependentActionFromActionMap;

  /**
   * Sets of volatile and notify-on-action-cache-hit actions. These are both needed
   * on incremental rebuilds.
   */
  private Set<Action> volatileActions = new HashSet<>();
  private Set<NotifyOnActionCacheHit> cacheHitActions = new HashSet<>();

  /**
   * Root (top-level) artifacts. All user-specified targets, including any that are depended on by
   * other root targets.
   */
  private final Set<Artifact> rootArtifacts;

  /**
   * The number of "real" (ie, non-Middleman) actions.
   */
  private int numRealActions = 0;

  /**
   * The set of all artifacts visited. May include outputs not in the key set
   * of dependenciesMultimap.
   */
  protected final Set<Artifact> artifacts = Sets.newHashSetWithExpectedSize(30000);

  /**
   * Actions known to have conflicting output files.
   */
  private Map<Action, String> badActions;

  /**
   * Set of actions that were not successfully built during the last build.
   *
   * After a build, there are five possibilities for ArtifactMetadataCache#changedArtifacts
   * and {@link #staleActions}:
   *
   * (1) The build was successful. Then both of these sets are empty.
   * (2) The build failed, but it was a non-incremental build. Then #staleActions will have the
   *     set of all actions that didn't execute successfully, and changedArtifacts will be empty.
   * (3) The build failed, but did so after {@link #markActionsStale} and
   *     {@link ArtifactMTimeCache#clearChangedArtifacts} were called in
   *     {@link IncrementalDependencyChecker#init}. Then staleActions will have the set of all
   *     actions that didn't execute successfully, and changedArtifacts will be empty.
   *     The next run will include those staleActions as dirty ones to be executed in the next
   *     build, regardless of their input files' modification times.
   * (4) The build failed, but did so before #markActionsStale was
   *     called, probably due to an interrupt or failure in
   *     ArtifactMetadataCache#updateCache. Then changedArtifacts will contain the set of
   *     all artifacts that were detected as changed since the previous run, while staleActions
   *     will have whatever value it had at the start of the run. The next run,
   *     ArtifactMetadataCache#updateCache will include this set of changedArtifacts in the
   *     set it returns to IncrementalDependencyChecker, regardless of whether those artifacts
   *     were detected as changed that run. The actions taking those artifacts as inputs will then
   *     be added to the set of dirty (and stale) actions for that run. This ensures that artifact
   *     changes are not lost due to an interrupted build, which would lead to a stable
   *     inconsistent state.
   * (5) The build is magically interrupted between the calls to
   *     #markActionsStale and ArtifactMTimeCache#clearChangedArtifacts. Then
   *     staleActions will have the full set of actions to be executed this run, while
   *     changedArtifacts will have the set of all artifacts detected as changed since the last
   *     run. On the next run, ArtifactMetadataCache#updateCache will redundantly report
   *     those artifacts as having changed, and those artifacts' generating actions will be
   *     redundantly added to the set of dirty (and stale) actions for that run. However, assuming
   *     that next run is not also interrupted at this point, the set of changedArtifacts will
   *     then be cleared, and no redundant steps will be performed in the future. Since
   *     those method calls are consecutive in IncrementalDependencyChecker#init, this case should
   *     practically never happen.
   */
  private final Set<Action> staleActions = new HashSet<>();

  /**
   * Whether or not this graph has data about stale actions from the previous build.
   */
  private boolean hasStaleActionData = false;

  /**
   * Total workload estimate.
   */
  protected AtomicLong workloadEstimate;

  /**
   * Total number of graph edges.
   */
  protected final AtomicLong totalEdgeCount = new AtomicLong();

  // As in the AbstractBuilder, cache the log level at class initialization.
  public static final boolean LOG_FINE = LOG.isLoggable(Level.FINE);
  public static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);
  public static final boolean LOG_FINEST = LOG.isLoggable(Level.FINEST);

  /**
   * Creates a new dependency graph with the specified top-level action.
   */
  @VisibleForTesting
  DependentActionGraph(DependentAction topLevelAction, Set<Artifact> rootArtifacts,
      ActionGraph actionGraph) {
    this.actionGraph = Preconditions.checkNotNull(actionGraph);
    this.rootArtifacts = ImmutableSet.copyOf(rootArtifacts);
    // On average ~90% of artifacts here will have <16 associated records.
    dependenciesMultimap = ArrayListMultimap.create(/*expectedKeys=*/ 10000,
                                                    /*expectedValuesPerKey=*/ 16);
    dependentActionFromActionMap = new LinkedHashMap<Action, DependentAction>(5000);
    this.topLevelAction = topLevelAction;
  }

  /**
   * Constructs a new dependency graph from the given set of artifacts to build.
   * The graph is built by exploring the action graph transitively from the
   * given artifacts, using the specified visitor.
   *
   * @param artifactSet the set of top-level artifacts to build.
   * @param stable if true, returns a reusable graph. if not, the graph may
   *        release memory while in use.
   * @param visitor the visitor to use for determining what parts of the graph
   *        to traverse.
   * @param notifyProfiler whether or not we should log
   *        {@link ProfilerTask#ACTION_GRAPH} messages.
   * @return a new dependency graph.
   */
  public static DependentActionGraph newGraph(Set<Artifact> artifactSet, ActionGraph actionGraph,
      boolean stable, DependentActionVisitor visitor, boolean notifyProfiler) {
    DependentAction topLevelRequest = DependentAction.createTopLevelDependency(artifactSet);
    DependentActionGraph dependencyGraph = stable
        ? new StableDependentActionGraph(topLevelRequest, artifactSet, actionGraph)
        : new DependentActionGraph(topLevelRequest, artifactSet, actionGraph);
    for (Artifact artifact : artifactSet) {
      dependencyGraph.addArtifact(artifact, topLevelRequest, visitor, notifyProfiler);
    }
    return dependencyGraph;
  }

  /**
   * Constructs a new dependency graph from the given set of artifacts to build.
   * The graph is built by exploring the action graph transitively from the
   * given artifacts.
   *
   * @param artifactSet the set of top-level artifacts to build.
   * @param stable if true, returns a reusable graph. if not, the graph may
   *        release memory while in use.
   * @return a new dependency graph.
   */
  public static DependentActionGraph newGraph(Set<Artifact> artifactSet, ActionGraph actionGraph,
      boolean stable) {
    return newGraph(artifactSet, actionGraph, stable, null, true);
  }

  /**
   * Constructs a new dependency graph from the given set of artifacts to build.
   * The graph is built by exploring the action graph transitively from the
   * given artifacts.
   *
   * @param artifactSet the set of top-level artifacts to build.
   * @return a new dependency graph.
   */
  public static DependentActionGraph newGraph(Set<Artifact> artifactSet, ActionGraph actionGraph) {
    return newGraph(artifactSet, actionGraph, false);
  }

  /** Returns the associated action graph. */
  public ActionGraph getActionGraph() {
    return actionGraph;
  }

  public DependentAction getTopLevelAction() {
    return topLevelAction;
  }

  /**
   * Returns the total number of actions left to build in the graph.
   */
  public int getRemainingActionCount() {
    return dependentActionFromActionMap.size();
  }

  /**
   * Returns the DependentActions associated with the given artifact.
   *
   * @param parent parent artifact of dependencies.
   */
  public Collection<DependentAction> getArtifactDependencies(Artifact parent) {
    return Collections.unmodifiableCollection(getArtifactDependenciesInternal(parent));
  }

  /**
   * Returns a modifiable collection of incident DependentActions to the given
   * artifact. Strictly an optimization; only for package-internal use.
   *
   * <p>PERFORMANCE CRITICAL!
   */
  Collection<DependentAction> getArtifactDependenciesInternal(Artifact parent) {
    return dependenciesMultimap.get(parent);
  }

  /**
   * Returns the DependentAction associated with the given action.
   *
   * @param action query action for dependency information.
   */
  @Nullable
  protected final DependentAction getActionDependency(Action action) {
    return dependentActionFromActionMap.get(action);
  }

  /**
   * Returns true iff the specified artifact exists in the graph.
   */
  @VisibleForTesting
  boolean containsArtifact(Artifact artifact) {
    return dependenciesMultimap.containsKey(artifact);
  }

  /**
   * Returns a collection of all artifacts in the graph.
   * This includes all output artifacts reached during the visitation.
   */
  public Set<Artifact> getArtifacts() {
    return Collections.unmodifiableSet(artifacts);
  }

  /**
   * Get the {@link DependentAction}s of a given artifact.
   *
   * <p>As a side effect, the artifact may get removed from this {@link DependentActionGraph} (to
   * save memory). This is acceptable when, for example, the artifact has been built, so the caller
   * no longer cares what the dependent actions are.
   *
   * <p>PERFORMANCE CRITICAL!
   *
   * @param artifact artifact to look up.
   */
  public Collection<DependentAction> getAndRemoveArtifactMaybe(Artifact artifact) {
    return dependenciesMultimap.removeAll(artifact);
  }

  /**
   * Get the {@link DependentAction}s of a given artifact.
   *
   * <p>This method does not change the dependent action graph.
   *
   *  @param artifact artifact to look up.
   */
  Collection<DependentAction> getDependentActions(Artifact artifact) {
    return Collections.unmodifiableCollection(dependenciesMultimap.get(artifact));
  }

  /**
   * Removes an edge from the graph, defined by an artifact to action mapping.
   *
   * <p>PERFORMANCE CRITICAL!
   *
   * @param artifact artifact adjacent to the edge.
   * @param action action adjacent to the edge.
   */
  void removeEdge(Artifact artifact, DependentAction action) {
    dependenciesMultimap.remove(artifact, action);
    if (!dependenciesMultimap.containsKey(artifact)) {
      artifacts.remove(artifact);
    }
  }

  /**
   * Remove a single action from the action graph. Beware! If you use this method, you are
   * responsible for leaving the dependent action graph in a consistent state.
   */
  public void removeAction(Action action) {
    staleActions.remove(action);
    DependentAction dependentAction = dependentActionFromActionMap.remove(action);
    volatileActions.remove(action);
    cacheHitActions.remove(action);
    totalEdgeCount.addAndGet(-countActionEdges(action));

    if (dependentAction == null) {
      return;
    }
    if (workloadEstimate != null) {
      // issue: "workload estimate can be null". This happens if a command that does
      // not use the execution phase but does construct the forward graph was previously run.
      workloadEstimate.addAndGet(-action.estimateWorkload());
    }

    for (Artifact artifact : action.getInputs()) {
      removeEdge(artifact, dependentAction);
    }
  }

  /**
   * Add a single action to the action graph. Beware! If you use this method, you are responsible
   * for leaving the dependent action graph in a consistent state.
   */
  public void addAction(Action action) {
    Preconditions.checkArgument(action != null);
    DependentAction dependentAction = DependentAction.createDependency(action, true);
    addNewActionState(dependentAction, action);
    for (Artifact artifact : action.getInputs()) {
      addEdge(artifact, dependentAction);
    }
    if (workloadEstimate != null) {
      // issue: "workload estimate can be null". This happens if a command that does
      // not use the execution phase but does construct the forward graph was previously run.
      workloadEstimate.addAndGet(action.estimateWorkload());
    }
  }

  private void addNewActionState(DependentAction dependentAction, Action action) {
    DependentAction oldDependentAction = dependentActionFromActionMap.put(action, dependentAction);
    Preconditions.checkState(oldDependentAction == null, action);
    totalEdgeCount.addAndGet(countActionEdges(action));

    if (action instanceof NotifyOnActionCacheHit) {
      cacheHitActions.add((NotifyOnActionCacheHit) action);
    }
    if (action.isVolatile()) {
      volatileActions.add(action);
    }
  }

  protected static int countActionEdges(Action action) {
    return action.getInputCount() + action.getOutputs().size();
  }

  /**
   * Adds the recursive dependencies of the given artifact to the graph.
   *
   * <p>Precondition: (artifact.getGeneratingAction() != null).
   *
   * @param artifact the artifact to use as the root of the dependents.
   * @param parent the action which depends on this artifact being built.
   */
  protected void addArtifact(Artifact artifact, DependentAction parent) {
    addArtifact(artifact, parent, null, true);
  }

  /**
   * Adds the recursive dependencies of the given artifact to the graph,
   * using a custom visitor for determining which parts of the graph
   * to traverse and optionally sending Profiler notifications.
   *
   * <p>Precondition: (artifact.getGeneratingAction() != null).
   *
   * @param artifact the artifact to use as the root of the dependents.
   * @param parent the action which depends on this artifact being built.
   * @param visitor the visitor which determines what parts of the graph to
   *        traverse.
   * @param notifyProfiler whether or not we should log
   *        {@link ProfilerTask#ACTION_GRAPH} messages.
   */
  private void addArtifact(Artifact artifact, DependentAction parent,
      DependentActionVisitor visitor, boolean notifyProfiler) {
    Preconditions.checkNotNull(artifact);
    addArtifactWithVisitor(artifact, parent, visitor, notifyProfiler);
  }

  public void clearMiddleman(Action action, Artifact middleman, Action middlemanAction) {
    // nothing to do here
  }

  public void addMiddleman(Action action, Artifact middleman) {
    // nothing to do here
  }

  /**
   * Adds the recursive dependencies of an artifact using custom behavior
   * defined by the {@link DependentActionVisitor}.
   *
   * @param artifact the artifact to use as the root of the dependents.
   * @param parent the action which depends on this artifact being built.
   * @param visitor a visitor implementing custom behavior to perform during traversal (or null).
   * @param notifyProfiler whether or not we should log
   *        {@link ProfilerTask#ACTION_GRAPH} messages.
   */
  @VisibleForTesting
  void addArtifactWithVisitor(Artifact artifact, DependentAction parent,
      DependentActionVisitor visitor, boolean notifyProfiler) {
    addEdge(artifact, parent);
    traverseRecursiveDependencies(artifact, new AddArtifactVisitor(visitor), notifyProfiler);
    Preconditions.checkState(artifactStack.isEmpty());
  }

  /**
   * Estimates the total workload (so progress can be reported).
   * May only be called while the build graph is not in active use.
   *
   * @return an estimate of the total work that needs to be done to build the
   *         transitive closure.
   */
  public long estimateTotalWorkload() {
    if (workloadEstimate != null) {
      return workloadEstimate.get();
    }
    workloadEstimate = new AtomicLong();
    for (Action action : getActions()) {
      workloadEstimate.getAndAdd(action.estimateWorkload());
    }
    return workloadEstimate.get();
  }

  /**
   * Performs global checks of the action graph.
   *
   * <p>Currently the only check is to ensure that no derived artifact has a path
   * which is a prefix of another, unless they are generated by the same action.
   * (We allow overlapping paths when they are generated by the same action;
   * in that case, one of them will be a directory, and it's no worse than
   * any other directory output.)
   *
   * <p>(This check belongs here, not in the analysis phase, because it's a
   * function of the set of artifacts that are built at the same time, so the
   * error cannot be blamed on one particular target.  This means that such
   * errors cannot be detected and reported during analysis phase using the
   * {@code ConfiguredTarget.hasErrors} mechanism, and it also means the
   * {@code --keep_going} behaviour of discarding faulty targets would not work for
   * this error.)
   *
   * <p>Errors are recorded in {@link #badActions} and reported when the Action
   * is executed.  This is necessary to ensure the proper subgraph
   * succeeds/fails with --keep_going.
   */
  public Map<Action, String> checkActionGraph(Iterable<Artifact> rootArtifacts) {
    if (badActions != null) {
      return badActions;
    }
    badActions = Maps.newConcurrentMap();

    // TODO(bazel-team): (2009) combine this visitation over the graph with others done prior
    // to execution.
    class DerivedArtifactVisitor extends ActionGraphVisitor {
      final Set<Artifact> derivedArtifacts = new HashSet<>();

      public DerivedArtifactVisitor(ActionGraph actionGraph) {
        super(actionGraph);
      }

      @Override
      protected void visitArtifact(Artifact artifact) {
        if (!artifact.isSourceArtifact()) {
          derivedArtifacts.add(artifact);
        }
      }

      // Returns the Action that writes to the specified execPath (which must
      // belong to an artifact in derivedArtifacts).
      // Not very efficient, only called in error codepath.
      Action findAction(PathFragment execPath) {
        for (Artifact x : derivedArtifacts) {
          if (x.getExecPath().equals(execPath)) {
            return actionGraph.getGeneratingAction(x);
          }
        }
        throw new IllegalArgumentException("not found: " + execPath);
      }
    }
    DerivedArtifactVisitor visitor = new DerivedArtifactVisitor(actionGraph);
    visitor.visitWhiteNodes(rootArtifacts);

    List<PathFragment> fragments = Lists.newArrayListWithCapacity(visitor.derivedArtifacts.size());
    for (Artifact a : visitor.derivedArtifacts) {
      fragments.add(a.getExecPath());
    }
    Collections.sort(fragments);
    // Report an error for every derived artifact which is a prefix of another.
    // If x << y << z (where << means "starts with"), then we only report
    // (x,y), (x,z), but not (y,z).
    //
    // i and j denote the indices of pathI (the prefix), and pathJ (the path
    // which has pathI as a prefix), respectively.
    for (int i = 0, len = fragments.size(); i < len - 1;) {
      PathFragment pathI = fragments.get(i);
      // Iterate over all immediately-following paths with pathI as a prefix.
      while (++i < len) {
        PathFragment pathJ = fragments.get(i);
        if (pathJ.startsWith(pathI)) { // prefix conflict
          Action actionI = visitor.findAction(pathI);
          Action actionJ = visitor.findAction(pathJ);
          if (actionI.shouldReportPathPrefixConflict(actionJ)) {
            String error = String.format(
                "output path '%s' (belonging to %s) is a prefix of output path '%s' (belonging "
                + "to %s).  These files cannot be built together; please rename one of them",
                pathI, Label.print(actionI.getOwner().getLabel()),
                pathJ, Label.print(actionJ.getOwner().getLabel()));
            badActions.put(actionI, error);
            badActions.put(actionJ, error);
          }
        } else {
          break;
        }
      }
    }
    return badActions;
  }

  /**
   * May only be called while the build graph is not in active use.
   * @return The number of edges in the graph, counting both action inputs and outputs as edges.
   */
  public long getTotalEdgeCount() {
    return totalEdgeCount.get();
  }

  /**
   * @return the volatile actions in the graph.
   */
  public Collection<Action> getVolatileActions() {
    return Collections.unmodifiableSet(volatileActions);
  }

  /**
   * @return the notify-on-cache-hit actions in the graph.
   */
  public Collection<NotifyOnActionCacheHit> getCacheHitActions() {
    return Collections.unmodifiableSet(cacheHitActions);
  }

  /**
   * Implements the default behavior for adding the subgraph rooted at an
   * artifact to the dependency graph:
   *
   * <p>For each action we'll create a {@link DependentAction} instance (that
   * would immediately record number of action's inputs as a number of "unbuilt"
   * inputs. Then for each source (non-generated) artifact we will
   * immediately mark it as "built" - reducing unbuilt counter for that
   * dependent action instance. For each generated input artifact we will
   * record dependency between it and DependentAction in the list-based
   * {@link #dependenciesMultimap}. Using a list-based multimap is important
   * because Action does not guarantee uniqueness of each input, so we may
   * (correctly) end up holding two identical <Artifact, DependentAction>
   * pairs. What is important is that the number of references to the specific
   * DependentAction instance in the dependenciesMultimap would be equal
   * to the number of generated input artifacts of the related action -
   * which is enforced by the builder.
   *
   * <p>PERFORMANCE CRITICAL!
   */
  private class AddArtifactVisitor implements DependentActionVisitor {
    private final DependentActionVisitor chainedVisitor;

    /**
     * Creates a new visitor with the "add artifact" behavior.
     * If chainedVisitor is not null, forwards to it for custom processing:
     * The return value determines whether to continue.
     * Source artifacts (i.e. those with null generatingActions) will never be
     * traversed further, even if the visitor returns true. They are leaves.
     *
     * @param chainedVisitor The visitor to forward events to. May be null.
     */
    public AddArtifactVisitor(DependentActionVisitor chainedVisitor) {
      this.chainedVisitor = chainedVisitor;
    }

    @Override
    public boolean visitArtifact(Artifact artifact) {
      boolean chainSuccess = (chainedVisitor == null) || chainedVisitor.visitArtifact(artifact);
      return (!artifact.isSourceArtifact() && chainSuccess);
    }

    @Override
    public boolean visitDependency(Action action, Artifact outputArtifact,
        @Nullable DependentAction dependency, boolean rootArtifact, boolean dependencyVisited) {
      Preconditions.checkNotNull(action, outputArtifact);

      // This ensures the uniqueness of DependentActions; do not create a new
      // one if one already exists for this action.
      if (dependencyVisited) {
        if (LOG_FINE) {
          LOG.fine(StringUtilities.indent("Already considered " + outputArtifact,
                                          artifactStack.size()));
        }
      } else {
        if (LOG_FINE) {
          LOG.fine(StringUtilities.indent("Considering " + outputArtifact,
                                          artifactStack.size()));
        }

        // TODO(bazel-team): (2009) save the artifactStack in the DependentAction
        // so that we can use it as context info when reporting errors.
        dependency = DependentAction.createDependency(action, rootArtifact);
        addDependentAction(dependency);
      }

      // If we added a new dependency, this is what is passed, not null.
      // However, the original "dependencyVisited" is preserved, since it
      // represents the state of the world before the action was executed.
      // This behavior is used to detect whether an action has been created.
      boolean chainSuccess = (chainedVisitor == null) || chainedVisitor.visitDependency(
          action, outputArtifact, dependency, rootArtifact, dependencyVisited);

      return !dependencyVisited && chainSuccess;
    }

    @Override
    public boolean visitInput(Artifact inputFile, DependentAction dependency) {
      boolean chainSuccess = (chainedVisitor == null) || chainedVisitor.visitInput(
          inputFile, dependency);

      addEdge(inputFile, dependency);

      return chainSuccess;
    }

    @Override
    public void visitOutputs(Collection<Artifact> outputFiles, DependentAction action) {
      if (chainedVisitor != null) {
        chainedVisitor.visitOutputs(outputFiles, action);
      }
      artifacts.addAll(outputFiles);
    }
  }

  /**
   * Performs a (depth-first) traversal on the transitive dependencies of an
   * artifact, visiting each of its nodes with the given
   * {@link DependentActionVisitor}.
   *
   * @param root the artifact whose dependents are being traversed.
   * @param callback a {@link DependentActionVisitor} which controls the node
   * visitation behavior.
   * @param notifyProfiler whether or not we should log
   *        {@link ProfilerTask#ACTION_GRAPH} messages.
   */
  @VisibleForTesting
  void traverseRecursiveDependencies(Artifact root, DependentActionVisitor callback,
      boolean notifyProfiler) {

    if (!callback.visitArtifact(root)) {
      return;
    }

    Action action = Preconditions.checkNotNull(actionGraph.getGeneratingAction(root), root);
    DependentAction dependentAction = getActionDependency(action);
    artifactStack.push(root);

    final Profiler profiler = Profiler.instance();
    if (notifyProfiler) {
      profiler.startTask(ProfilerTask.ACTION_GRAPH, action);
    }
    try {
      boolean dependencyVisited = (dependentAction != null);
      if (callback.visitDependency(action, root, dependentAction,
          rootArtifacts.contains(root), dependencyVisited)) {
        // The action didn't exist before, but the visitor may have created it.
        dependentAction = getActionDependency(action);
        for (Artifact input : action.getInputs()) {
          if (callback.visitInput(input, dependentAction)) {
            // Recursively process the input artifacts.
            traverseRecursiveDependencies(input, callback, notifyProfiler);
          }
        }
        callback.visitOutputs(action.getOutputs(), dependentAction);
        Artifact top = artifactStack.peek();
        // This compares Artifacts by reference, but it's OK because we put the same Artifact
        // instance in the stack just about the try {} block.
        Preconditions.checkState(root == top, "root=%s / top=%s", root, top);
      }
    } finally {
      if (notifyProfiler) {
        profiler.completeTask(ProfilerTask.ACTION_GRAPH);
      }
      artifactStack.pop();
    }
  }

  /**
   * Explicitly add an edge to the graph, represented by an artifact-action
   * dependency.
   *
   * <p>PERFORMANCE CRITICAL!
   */
  @VisibleForTesting
  protected void addEdge(Artifact inputFile, DependentAction dependentAction) {
    Preconditions.checkNotNull(dependentAction);

    if (LOG_FINEST) {
      LOG.finest(StringUtilities.indent("Recording dependency of " +
          dependentAction.prettyPrint() + " on " + inputFile.prettyPrint(),
          artifactStack.size()));
    }

    dependenciesMultimap.put(inputFile, dependentAction);
    artifacts.add(inputFile);
  }

  /**
   * Adds a dependent action to the map of action dependencies.
   * These will be processed and built in the appropriate Builder.
   */
  private void addDependentAction(DependentAction newAction) {
    Action action = Preconditions.checkNotNull(newAction.getAction());
    addNewActionState(newAction, action);
    // This is actually the number of actions which execute.
    MiddlemanType type = action.getActionType();
    if (type == MiddlemanType.NORMAL || type == MiddlemanType.TARGET_COMPLETION_MIDDLEMAN) {
       numRealActions++;
     }
  }

  /**
   * @return the total number of real, non-Middleman actions in the graph.
   */
  public int getNumRealActions() {
    return numRealActions;
  }

  /**
   * Returns an immutable view of all DependentActions visited in the graph.
   */
  public Collection<DependentAction> getDependentActions() {
    return Collections.unmodifiableCollection(dependentActionFromActionMap.values());
  }

  /**
   * Returns an immutable view of all actions added to the graph.
   */
  public Collection<Action> getActions() {
    return Collections.unmodifiableSet(dependentActionFromActionMap.keySet());
  }

  /**
   * Mark actions to be executed as stale, so if this build fails, the next build will
   * know they need to be re-executed. Not all actions in this graph may be executed,
   * if we are using a pruned forward graph in the {@link IncrementalDependencyChecker}.
   * This method is idempotent, since it may be called twice in one build (from
   * {@link IncrementalDependencyChecker} and {@link ParallelBuilder}).
   * @param actions Actions to be marked stale.
   */
  public void markActionsStale(Collection<Action> actions) {
    if (!hasStaleActionData) {
      staleActions.addAll(actions);
      hasStaleActionData = true;
    }
  }

  /**
   * Returns an immutable view of all stale actions in the graph, and invalidates the
   * current set of stale actions. Note that staleActions may not have been set if
   * this forward graph is newly created.
   */
  public Set<Action> getStaleActions() {
    return Collections.unmodifiableSet(staleActions);
  }

  /**
   * Mark action as successfully executed this build, so no longer stale.
   */
  public boolean markActionNotStale(Action action) {
    Preconditions.checkState(hasStaleActionData);
    boolean result = staleActions.remove(action);
    Preconditions.checkState(result, action);
    return result;
  }

  /**
   * Checks whether or not this graph has stale Actions from the previous build,
   * and tells the graph that its set of stale actions will be updated.
   * @return Whether this graph has stale Actions from the previous build.
   */
  public boolean hasStaleActionDataAndInit() {
    boolean result = hasStaleActionData;
    hasStaleActionData = false;
    return result;
  }

  // The following methods apply only to keeping the graph ready for future
  // builds. This is only necessary for {@link StableDependentActionGraph}.

  /**
   * Called just before an action's inputs are updated.
   */
  @ThreadSafety.ThreadSafe // This is the only method which must be thread-safe.
  public void beforeChange(Action action) {
  }

  /**
   * Reset the graph state in preparation for another build.
   */
  @ThreadSafety.ThreadHostile // Don't call interleaved with any other methods.
  public void sync() {
  }
}
