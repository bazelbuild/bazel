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

package com.google.devtools.build.lib.view;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionRegistry;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactMTimeCache;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.DependentActionGraph;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.TestMiddlemanObserver;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.events.WarningsAsErrorsEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner.LoadingResult;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.skyframe.LabelAndConfiguration;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetCompletionKey;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.ExtraActionArtifactsProvider.ExtraArtifactSet;
import com.google.devtools.build.lib.view.actions.TargetCompletionMiddlemanAction;
import com.google.devtools.build.lib.view.config.BinTools;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.view.test.TestProvider;
import com.google.devtools.build.lib.view.test.TestRunnerAction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * <p>The BuildView presents a semantically-consistent and transitively-closed
 * dependency graph for some set of packages.
 *
 * <h2>Package design</h2>
 *
 * <p>This package contains the Blaze dependency analysis framework (aka
 * "analysis phase").  The goal of this code is to perform semantic analysis of
 * all of the build targets required for a given build, to report
 * errors/warnings for any problems in the input, and to construct an "action
 * graph" (see {@code lib.actions} package) correctly representing the work to
 * be done during the execution phase of the build.
 *
 * <p><b>Configurations</b> the inputs to a build come from two sources: the
 * intrinsic inputs, specified in the BUILD file, are called <em>targets</em>.
 * The environmental inputs, coming from the build tool, the command-line, or
 * configuration files, are called the <em>configuration</em>.  Only when a
 * target and a configuration are combined is there sufficient information to
 * perform a build. </p>
 *
 * <p>Targets are implemented by the {@link Target} hierarchy in the {@code
 * lib.packages} code.  Configurations are implemented by {@link
 * BuildConfiguration}.  The pair of these together is represented by an
 * instance of class {@link ConfiguredTarget}; this is the root of a hierarchy
 * with different implementations for each kind of target: source file, derived
 * file, rules, etc.
 *
 * <p>The framework code in this package (as opposed to its subpackages) is
 * responsible for constructing the {@code ConfiguredTarget} graph for a given
 * target and configuration, taking care of such issues as:
 * <ul>
 *   <li>caching common subgraphs.
 *   <li>detecting and reporting cycles.
 *   <li>correct propagation of errors through the graph.
 *   <li>reporting universal errors, such as dependencies from production code
 *       to tests, or to experimental branches.
 *   <li>capturing and replaying errors.
 *   <li>maintaining the graph from one build to the next to
 *       avoid unnecessary recomputation.
 *   <li>checking software licenses.
 * </ul>
 *
 * <p>See also {@link ConfiguredTarget} which documents some important
 * invariants.
 */
public class BuildView {

  /**
   * Options that affect the <i>mechanism</i> of analysis.  These are distinct from {@link
   * com.google.devtools.build.lib.view.config.BuildOptions}, which affect the <i>value</i>
   * of a BuildConfiguration.
   */
  public static class Options extends OptionsBase {

    @Option(name = "keep_going",
            abbrev = 'k',
            defaultValue = "false",
            category = "strategy",
            help = "Continue as much as possible after an error.  While the "
            + "target that failed, and those that depend on it, cannot be "
            + "analyzed (or built), the other prerequisites of these "
            + "targets can be analyzed (or built) all the same.")
    public boolean keepGoing;

    @Option(name = "analysis_warnings_as_errors",
            defaultValue = "false",
            category = "strategy",
            help = "Treat visible analysis warnings as errors.")
    public boolean analysisWarningsAsErrors;

    @Option(name = "discard_analysis_cache",
        defaultValue = "false",
        category = "strategy",
        help = "Discard the analysis cache immediately after the analysis phase completes. "
        + "Reduces memory usage by ~10%, but makes further incremental builds slower.")
    public boolean discardAnalysisCache;

    @Option(name = "keep_forward_graph",
            defaultValue = "false",
            category = "undocumented",
            help = "Cache the forward action graph across builds for faster "
            + "incremental rebuilds. May slightly increase memory while Blaze "
            + "server is idle."
               )
    public boolean keepForwardGraph;

    @Option(name = "experimental_extra_action_filter",
            defaultValue = "",
            category = "experimental",
            converter = RegexFilter.RegexFilterConverter.class,
            help = "Filters set of targets to schedule extra_actions for.")
    public RegexFilter extraActionFilter;

    @Option(name = "experimental_extra_action_top_level_only",
            defaultValue = "false",
            category = "experimental",
            help = "Only schedules extra_actions for top level targets.")
    public boolean extraActionTopLevelOnly;

    @Option(name = "version_window_for_dirty_node_gc",
            defaultValue = "0",
            category = "undocumented",
            help = "Nodes that have been dirty for more than this many versions will be deleted"
                + " from the graph upon the next update. Values must be non-negative long integers,"
                + " or -1 indicating the maximum possible window.")
    public long versionWindowForDirtyNodeGc;
  }

  private static Logger LOG = Logger.getLogger(BuildView.class.getName());

  private final BlazeDirectories directories;

  private final ImmutableList<OutputFormatter> outputFormatters;

  private EventHandler reporter;

  private final SkyframeExecutor skyframeExecutor;
  private final SkyframeBuildView skyframeBuildView;

  private final PackageManager packageManager;

  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;

  private final BinTools binTools;

  @Nullable
  /** Not used in Skyframe full. */
  private final ArtifactMTimeCache artifactMTimeCache;

  private BuildConfigurationCollection configurations = new BuildConfigurationCollection();

  private BuildConfigurationCollection lastConfigurations = null;

  private ConfiguredRuleClassProvider ruleClassProvider;

  private final ConfiguredTargetFactory factory;

  private final ArtifactFactory artifactFactory;

  /**
   * A union of package roots of all previous incremental analysis results. This is used to detect
   * changes of package roots between incremental analysis instances.
   */
  private final Map<PathFragment, Path> cumulativePackageRoots = new HashMap<>();
  private final ForwardGraphCache forwardGraphCache;

  private final MutableActionGraph legacyActionGraph;

  private WorkspaceStatusArtifacts lastWorkspaceStatusArtifacts = null;

  // This is not accessed on multiple threads
  private final Set<Action> lastTargetCompletionMiddlemen = new HashSet<>();

  private final List<Action> lastExclusiveSchedulingMiddlemen = new ArrayList<>();

  /**
   * Used only for testing that we clear Skyframe caches correctly.
   * TODO(bazel-team): Remove this once we get rid of legacy Skyframe synchronization.
   */
  private boolean skyframeCacheWasInvalidated = false;

  /**
   * If the last build was executed with {@code Options#discard_analysis_cache} and we are not
   * running Skyframe full, we should clear the legacy data since it is out-of-sync.
   */
  private boolean skyframeAnalysisWasDiscarded = false;

  @VisibleForTesting
  public Set<SkyKey> getSkyframeEvaluatedTargetKeysForTesting() {
    return skyframeBuildView.getEvaluatedTargetKeys();
  }

  /** The number of targets freshly evaluated in the last analysis run. */
  public int getTargetsVisited() {
    return skyframeBuildView.getEvaluatedTargetKeys().size();
  }

  /**
   * Returns true iff Skyframe was invalidated during the analysis phase.
   * TODO(bazel-team): Remove this once we do not need to keep legacy in sync with Skyframe.
   */
  @VisibleForTesting
  boolean wasSkyframeCacheInvalidatedDuringAnalysis() {
    return skyframeCacheWasInvalidated;
  }

  public BuildView(BlazeDirectories directories, PackageManager packageManager,
      ConfiguredRuleClassProvider ruleClassProvider, @Nullable SkyframeExecutor skyframeExecutor,
      ImmutableList<OutputFormatter> outputFormatters, BinTools binTools,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory) {
    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.directories = directories;
    this.packageManager = packageManager;
    this.binTools = binTools;
    this.artifactFactory = new ArtifactFactory(directories.getExecRoot());
    this.ruleClassProvider = ruleClassProvider;
    this.factory = new ConfiguredTargetFactory(ruleClassProvider);
    this.skyframeExecutor = Preconditions.checkNotNull(skyframeExecutor);
    boolean skyframeFull = skyframeExecutor.skyframeBuild();
    this.artifactMTimeCache = skyframeFull ? null : new ArtifactMTimeCache();
    this.forwardGraphCache =  skyframeFull ? null : new ForwardGraphCache();
    this.outputFormatters = outputFormatters;
    this.legacyActionGraph = skyframeFull ? null : new MapBasedActionGraph();
    this.skyframeBuildView = new SkyframeBuildView(legacyActionGraph, factory, artifactFactory,
        null, skyframeExecutor, new Runnable() {
      @Override
      public void run() {
        clear();
      }
    }, outputFormatters, binTools);
    skyframeExecutor.setSkyframeBuildView(skyframeBuildView);
  }

  /** Returns the action graph. */
  public ActionGraph getActionGraph() {
    if (skyframeExecutor.skyframeBuild()) {
      return new ActionGraph() {
        @Override
        public Action getGeneratingAction(Artifact artifact) {
          return skyframeExecutor.getGeneratingAction(artifact);
        }
      };
    } else {
      return legacyActionGraph;
    }
  }

  /**
   * Returns whether the given configured target has errors.
   */
  @VisibleForTesting
  public boolean hasErrors(ConfiguredTarget configuredTarget) {
    return configuredTarget == null;
  }

  /**
   * Sets the configurations. Not thread-safe. DO NOT CALL except from tests!
   */
  @VisibleForTesting
  void setConfigurationsForTesting(BuildConfigurationCollection configurations) {
    setConfigurationsInternal(configurations);
  }

  private void setConfigurationsInternal(BuildConfigurationCollection configurations) {
    this.configurations = configurations;
  }

  public BuildConfigurationCollection getConfigurationCollection() {
    return configurations;
  }

  private void clearActionGraph() {
    lastTargetCompletionMiddlemen.clear();
    lastExclusiveSchedulingMiddlemen.clear();

    lastWorkspaceStatusArtifacts = null;
    legacyActionGraph.clear();
  }

  /**
   * Clear the graphs of ConfiguredTargets and Artifacts.
   */
  @VisibleForTesting
  public void clear() {
    cumulativePackageRoots.clear();
    artifactFactory.clear();
    if (forwardGraphCache != null) {
      forwardGraphCache.clear();
    }
    if (artifactMTimeCache != null) {
      artifactMTimeCache.clear();
    }
    if (!skyframeExecutor.skyframeBuild()) {
      clearActionGraph();
    }
  }

  public ArtifactFactory getArtifactFactory() {
    return artifactFactory;
  }

  /**
   * Returns the artifact mtime cache associated with this {@link BuildView} instance.
   */
  @Nullable
  public ArtifactMTimeCache getArtifactMTimeCache() {
    return artifactMTimeCache;
  }

  private void removeFromForwardGraphMaybe(Action action) {
    if (forwardGraphCache != null) {
      forwardGraphCache.removeAction(action);
    }
  }

  private void addToForwardGraphMaybe(Action action) {
    if (forwardGraphCache != null) {
      forwardGraphCache.addAction(action);
    }
  }

  /**
   * Maps each configured target to a list of artifacts that need to be built in order to consider
   * that target as being built successfully.
   *
   * <p>Also takes care to remove all traces of previous target completion middlemen in the action
   * graphs, if they are present.
   *
   * <p>This should only be called once per build.
   */
  private Multimap<ConfiguredTarget, Artifact> createTargetCompletionMiddlemen(
      Iterable<ConfiguredTarget> targets, TopLevelArtifactContext options,
      SkyframeExecutor skyframeExecutor) {
    if (skyframeExecutor.skyframeBuild()) {
      Preconditions.checkState(lastTargetCompletionMiddlemen.isEmpty());
      skyframeExecutor.injectTopLevelContext(options);

      Multimap<ConfiguredTarget, Artifact> result = ArrayListMultimap.create();
      for (ConfiguredTarget target : targets) {
        result.putAll(target, TopLevelArtifactHelper.getAllArtifactsToBuild(target, options));
        if (!(target.getTarget() instanceof Rule)) {
          continue;
        }
        result.put(target, artifactFactory.getDerivedArtifact(
            TopLevelArtifactHelper.getMiddlemanRelativePath(target.getLabel()),
            target.getConfiguration().getMiddlemanDirectory(),
            new TargetCompletionKey(target.getLabel(), target.getConfiguration())));
      }
      return result;
    }

    // First remove the old middlemen from the action graphs
    for (Action oldAction : lastTargetCompletionMiddlemen) {
      removeFromForwardGraphMaybe(oldAction);
      legacyActionGraph.unregisterAction(oldAction);
    }

    lastTargetCompletionMiddlemen.clear();

    Multimap<ConfiguredTarget, Artifact> result = ArrayListMultimap.create();
    for (ConfiguredTarget target : targets) {
      // TODO(bazel-team): Adding the target completion middleman to artifactsToBuild should
      // suffice.
      // TODO(bazel-team): use NestedSet for targetOutputs
      Iterable<Artifact> targetOutputs =
          TopLevelArtifactHelper.getAllArtifactsToBuild(target, options);
      result.putAll(target, targetOutputs);
      if (!(target.getTarget() instanceof Rule)) {
        continue;
      }

      ActionOwner actionOwner = new PostInitializationActionOwner(target);
      BuildConfiguration configuration = target.getConfiguration();
      Preconditions.checkState(configuration != null);

      // These actions serve similar roles to middlemen, but, unlike middlemen,
      // are expected to execute.
      Artifact middleman = artifactFactory.getDerivedArtifact(
          TopLevelArtifactHelper.getMiddlemanRelativePath(target.getLabel()),
          configuration.getMiddlemanDirectory(),
          // Null owner because this artifact's generating action is currently retrieved from the
          // skyframe executor, not from the configured target.
          ArtifactOwner.NULL_OWNER);
      if (middleman != null) {
        artifactFactory.removeSchedulingMiddleman(middleman);
      }
      middleman = artifactFactory.getDerivedArtifact(
          TopLevelArtifactHelper.getMiddlemanRelativePath(target.getLabel()),
          configuration.getMiddlemanDirectory(),
          ArtifactOwner.NULL_OWNER);
      Action newAction = new TargetCompletionMiddlemanAction(target, actionOwner,
          targetOutputs, middleman);

      // Register the new action in the set of target completion middleman actions and in the
      // two action graphs
      registerAction(newAction);
      lastTargetCompletionMiddlemen.add(newAction);
      addToForwardGraphMaybe(newAction);
      result.put(target, middleman);
    }
    return result;
  }

  /**
   * Create the workspace status artifacts (i.e. the ones containing the build info).
   *
   * <p>This method should not be called during the multithreaded portion of the analysis phase.
   *
   * <p>This is complicated, so a little explanation is in order.
   *
   * <p>By the end of this method, the action graph and the dependent action graph should be in a
   * consistent state (both alone and with each other). There are the following cases:
   *
   * <ul>
   * <li>Analysis state was not reused. In this case, <code>lastBuildInfoAction</code> will be null
   * and neither the action graph nor the dependent action graph will contain a build info action,
   * so we create it register it in both graphs.
   * <li>Analysis state was reused in a non-incremental build. <code>lastBuildInfoAction</code>,
   * the action graph and the dependent action graph will all be consistent. In this case, the
   * action will be removed from both of the action graphs, and a new instance will be added to
   * them.
   * <li>Incremental analysis was in effect. In this case, the dependent action graph will contain
   * <code>lastBuildInfoAction</code>, but the action graph will not. We remove it from the
   * dependent action graph, create a new one, and add it to both action graphs.
   * </ul>
   */
  @ThreadHostile  // Mutates lastBuildInfoAction and changes the forward graph
  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  void createWorkspaceStatusArtifacts(UUID buildId)
      throws ViewCreationFailedException {
    // If we re-created the header actions, too, they would always be executed which would be bad
    // for incrementality and increase yet another state for an action in the already too
    // complicated action graph scheme. Therefore, what we do is that we only re-create the actual
    // build info action.
    //
    // This would be incorrect if we ever changed the configurations without clearing the action
    // graph (since the header actions depend on the set of configurations we have), but we
    // currently don't do that, so it's fine.
    boolean createHeaderActions = lastWorkspaceStatusArtifacts == null;
    if (lastWorkspaceStatusArtifacts != null) {
      Action oldAction = lastWorkspaceStatusArtifacts.getBuildInfoAction();
      removeFromForwardGraphMaybe(oldAction);
      legacyActionGraph.unregisterAction(oldAction);
      lastWorkspaceStatusArtifacts = null;
    }

    lastWorkspaceStatusArtifacts = WorkspaceStatusUtils.createWorkspaceStatusArtifacts(
        directories.getBuildDataDirectory(),
        factory.getBuildInfoFactories(), artifactFactory, configurations, buildId,
        workspaceStatusActionFactory);

    skyframeBuildView.setWorkspaceStatusArtifacts(lastWorkspaceStatusArtifacts);
    try {
      // .addAction() always puts the actions in the forward graph, thus, they will always be
      // executed. But that is almost completely okay, since the build info action needs to be
      // executed always anyway and .addAction() is only called on header actions when the graph
      // is cleared (when a lot of actions are executed anyway, so a few extra does not count)

      if (createHeaderActions) {
        for (Action action : lastWorkspaceStatusArtifacts.getActions()) {
          registerAction(action);
          addToForwardGraphMaybe(action);
        }
      }

      Action buildInfoAction = lastWorkspaceStatusArtifacts.getBuildInfoAction();
      registerAction(buildInfoAction);
      addToForwardGraphMaybe(buildInfoAction);
    } catch (ActionConflictException e) {
      // This should never happen. With the old action graph, new actions are always ignored. With
      // the new action graph, there should never be a conflict, because we always start with an
      // empty map, and we only call this method once per build.
      LoggingUtil.logToRemote(Level.SEVERE, "Unexpected duplicate build info action", e);
      throw new ViewCreationFailedException("Unexpected duplicate build info action: "
          + e.getMessage());
    }
  }

  @VisibleForTesting
  WorkspaceStatusArtifacts getLastWorkspaceStatusArtifactsForTesting() {
    Preconditions.checkState(skyframeExecutor == null || !skyframeExecutor.skyframeBuild());
    return lastWorkspaceStatusArtifacts;
  }

  /**
   * Returns a corresponding ConfiguredTarget, if one exists; otherwise throws an {@link
   * NoSuchConfiguredTargetException}.
   */
  @ThreadSafe
  private ConfiguredTarget getConfiguredTarget(Target target, BuildConfiguration config)
      throws NoSuchConfiguredTargetException {
    ConfiguredTarget result =
        getExistingConfiguredTarget(new TargetAndConfiguration(target, config));
    if (result == null) {
      throw new NoSuchConfiguredTargetException(target.getLabel(), config);
    }
    return result;
  }

  /**
   * Obtains a {@link ConfiguredTarget} given a {@code label}, by delegating
   * to the package cache and
   * {@link #getConfiguredTarget(Target, BuildConfiguration)}.
   */
  public ConfiguredTarget getConfiguredTarget(Label label, BuildConfiguration config)
      throws NoSuchPackageException, NoSuchTargetException, NoSuchConfiguredTargetException {
    return getConfiguredTarget(packageManager.getLoadedTarget(label), config);
  }

  public Iterable<ConfiguredTarget> getDirectPrerequisites(ConfiguredTarget ct) {
    return getDirectPrerequisites(ct, null);
  }

  public Iterable<ConfiguredTarget> getDirectPrerequisites(ConfiguredTarget ct,
      @Nullable final LoadingCache<Label, Target> targetCache) {
    if (!(ct.getTarget() instanceof Rule)) {
      return ImmutableList.of();
    }

    class SilentDependencyResolver extends DependencyResolver {
      @Override
      protected void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label) {
        // The error must have been reported already during analysis.
      }

      @Override
      protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
        // The error must have been reported already during analysis.
      }

      @Override
      protected Target getTarget(Label label) throws NoSuchThingException {
        if (targetCache == null) {
          return packageManager.getLoadedTarget(label);
        }

        try {
          return targetCache.get(label);
        } catch (ExecutionException e) {
          // All lookups should succeed because we should not be looking up any targets in error.
          throw new IllegalStateException(e);
        }
      }
    }

    DependencyResolver dependencyResolver = new SilentDependencyResolver();
    TargetAndConfiguration ctgNode =
        new TargetAndConfiguration(ct.getTarget(), ct.getConfiguration());
    return getExistingConfiguredTargets(dependencyResolver.dependentNodeMap(ctgNode).values());
  }

  public TransitiveInfoCollection getGeneratingRule(OutputFileConfiguredTarget target) {
    return target.getGeneratingRule();
  }

  @Deprecated
  public Iterable<ConfiguredTarget> getAllConfiguredTargets() {
    // TODO(bazel-team): Re-enable this if needed for dump command.
    throw new UnsupportedOperationException();
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();  // avoid nondeterminism
  }

  /**
   * Return value for {@link BuildView#update} and {@code BuildTool.prepareToBuild}.
   */
  public static final class AnalysisResult {

    public static final AnalysisResult EMPTY = new AnalysisResult(
        ImmutableList.<ConfiguredTarget>of(), null, null, null, null,
        ImmutableMultimap.<ConfiguredTarget, Artifact>of(), ImmutableList.<Artifact>of(),
        ImmutableList.<Artifact>of());

    private final ImmutableList<ConfiguredTarget> targetsToBuild;
    @Nullable private final ImmutableList<ConfiguredTarget> targetsToTest;
    @Nullable private final String error;
    private final ActionGraph actionGraph;
    private final DependentActionGraph dependentActionGraph;
    private final ImmutableMultimap<ConfiguredTarget, Artifact> targetCompletionMap;
    private final ImmutableSet<Artifact> artifactsToBuild;
    private final ImmutableSet<Artifact> exclusiveTestArtifacts;

    private AnalysisResult(
        Collection<ConfiguredTarget> targetsToBuild, Collection<ConfiguredTarget> targetsToTest,
        @Nullable String error, ActionGraph actionGraph,
        DependentActionGraph dependentActionGraph,
        Multimap<ConfiguredTarget, Artifact> targetCompletionMap,
        Collection<Artifact> artifactsToBuild,
        Collection<Artifact> exclusiveTestArtifacts) {
      this.targetsToBuild = ImmutableList.copyOf(targetsToBuild);
      this.targetsToTest = targetsToTest == null ? null : ImmutableList.copyOf(targetsToTest);
      this.error = error;
      this.actionGraph = actionGraph;
      this.dependentActionGraph = dependentActionGraph;
      this.targetCompletionMap = ImmutableMultimap.copyOf(targetCompletionMap);
      this.artifactsToBuild = ImmutableSet.copyOf(artifactsToBuild);
      this.exclusiveTestArtifacts = ImmutableSet.copyOf(exclusiveTestArtifacts);
    }

    /**
     * Returns configured targets to build.
     */
    public Collection<ConfiguredTarget> getTargetsToBuild() {
      return targetsToBuild;
    }

    /**
     * Returns the configured targets to run as tests, or {@code null} if testing was not
     * requested (e.g. "build" command rather than "test" command).
     */
    @Nullable
    public Collection<ConfiguredTarget> getTargetsToTest() {
      return targetsToTest;
    }

    public Multimap<ConfiguredTarget, Artifact> getTargetCompletionMap() {
      return targetCompletionMap;
    }

    public ImmutableSet<Artifact> getArtifactsToBuild() {
      return artifactsToBuild;
    }

    public ImmutableSet<Artifact> getExclusiveTestArtifacts() {
      return exclusiveTestArtifacts;
    }

    /**
     * Returns an error description (if any).
     */
    @Nullable public String getError() {
      return error;
    }

    /**
     * Returns the action graph.
     */
    public ActionGraph getActionGraph() {
      return actionGraph;
    }

    /**
     * Returns the forward action graph, which is only present for legacy builds.
     */
    @Nullable public DependentActionGraph getDependentActionGraph() {
      return dependentActionGraph;
    }

    public boolean hasStaleActionData() {
      return dependentActionGraph != null
          ? dependentActionGraph.hasStaleActionDataAndInit()
          : false;
    }
  }


  /**
   * Returns the collection of configured targets corresponding to any of the provided targets.
   */
  @VisibleForTesting
  static Iterable<? extends ConfiguredTarget> filterTestsByTargets(
      Collection<? extends ConfiguredTarget> targets,
      final Set<? extends Target> allowedTargets) {
    return Iterables.filter(targets,
        new Predicate<ConfiguredTarget>() {
          @Override
              public boolean apply(ConfiguredTarget rule) {
            return allowedTargets.contains(rule.getTarget());
          }
        });
  }

  private void prepareToBuild() throws ViewCreationFailedException {
    for (BuildConfiguration config : configurations.getTargetConfigurations()) {
      config.prepareToBuild(directories.getExecRoot(), getArtifactFactory());
    }
  }

  @ThreadCompatible
  public AnalysisResult update(@Nullable UUID buildId, LoadingResult loadingResult,
      BuildConfigurationCollection configurations, BuildView.Options viewOptions,
      TopLevelArtifactContext topLevelOptions, EventHandler eventHandler, EventBus eventBus)
          throws ViewCreationFailedException, InterruptedException {

    // Detect errors during analysis and don't attempt a build.
    //
    // (Errors reported during the previous step, package loading, that do
    // not cause the visitation of the transitive closure to abort, are
    // recoverable.  For example, an error encountered while evaluating an
    // irrelevant rule in a visited package causes an error to be reported,
    // but visitation still succeeds.)
    ErrorCollector errorCollector = null;
    if (!viewOptions.keepGoing) {
      eventHandler = errorCollector = new ErrorCollector(eventHandler);
    }

    // Treat analysis warnings as errors, to enable strict builds.
    //
    // Warnings reported during analysis are converted to errors, ultimately
    // triggering failure. This check needs to be added after the keep-going check
    // above so that it is invoked first (FIFO eventHandler chain). This way, detected
    // warnings are converted to errors first, and then the proper error handling
    // logic is invoked.
    WarningsAsErrorsEventHandler warningsHandler = null;
    if (viewOptions.analysisWarningsAsErrors) {
      eventHandler = warningsHandler = new WarningsAsErrorsEventHandler(eventHandler);
    }

    this.reporter = eventHandler;
    skyframeBuildView.setWarningListener(reporter);
    skyframeExecutor.setErrorEventListener(reporter);

    LOG.info("Starting analysis");
    pollInterruptedStatus();

    skyframeBuildView.resetEvaluatedConfiguredTargetKeysSet();

    Collection<Target> targets = loadingResult.getTargets();
    eventBus.post(new AnalysisPhaseStartedEvent(targets));

    skyframeCacheWasInvalidated = false;
    // Clear all cached ConfiguredTargets on configuration change. We need to do this explicitly
    // because we need to make sure that the legacy action graph does not contain multiple actions
    // with different versions of the same (target/host/etc.) configuration.
    // In the future the action graph will be probably be keyed by configurations, which should
    // obviate the need for this workaround.
    //
    // Also if --discard_analysis_cache was used in the last build we want to clear the legacy
    // data.
    if ((lastConfigurations != null && !configurations.equals(this.configurations))
        || skyframeAnalysisWasDiscarded) {
      skyframeExecutor.dropConfiguredTargets();
      skyframeCacheWasInvalidated = true;
      clear();
    }
    skyframeAnalysisWasDiscarded = false;
    ImmutableMap<PathFragment, Path> packageRoots = loadingResult.getPackageRoots();

    if (buildHasIncompatiblePackageRoots(packageRoots)) {
      // When a package root changes source artifacts with the new root will be created, but we
      // cannot be sure that there are no references remaining to the corresponding artifacts
      // with the old root. To avoid that scenario, the analysis cache is simply dropped when
      // a package root change is detected.
      LOG.info("Discarding analysis cache: package roots have changed.");

      skyframeExecutor.dropConfiguredTargets();
      skyframeCacheWasInvalidated = true;
      clear();
    }
    cumulativePackageRoots.putAll(packageRoots);
    lastConfigurations = this.configurations;
    setConfigurationsInternal(configurations);
    setArtifactRoots(packageRoots);

    // Determine the configurations.
    List<TargetAndConfiguration> nodes = nodesForTargets(targets);

    List<LabelAndConfiguration> targetSpecs =
        Lists.transform(nodes, new Function<TargetAndConfiguration, LabelAndConfiguration>() {
          @Override
          public LabelAndConfiguration apply(TargetAndConfiguration node) {
            return new LabelAndConfiguration(node.getLabel(), node.getConfiguration());
          }
        });

    prepareToBuild();
    skyframeBuildView.setWarningListener(warningsHandler);
    if (skyframeExecutor.skyframeBuild()) {
      skyframeExecutor.injectWorkspaceStatusData();
    } else {
      createWorkspaceStatusArtifacts(buildId);
    }
    Collection<ConfiguredTarget> configuredTargets;
    try {
      configuredTargets = skyframeBuildView.configureTargets(
          targetSpecs, eventBus, viewOptions.keepGoing);
    } finally {
      // if skyframeCacheWasInvalidated then we have already invalidated everything.
      // In case of an interrupted exception, if we had invalidated some configured targets we
      // also clear legacy data.
      if (!skyframeCacheWasInvalidated && skyframeBuildView.isSomeConfiguredTargetInvalidated()
          && artifactMTimeCache != null) {
        // ConfiguredTargets have changed. We cannot reuse forwardGraphCache and artifactMTimeCache
        // in Skyframe. ForwardGraphCache should go away once we have full Skyframe. It is not worth
        // it to optimize it now.
        forwardGraphCache.clear();
        artifactMTimeCache.clear();
      }
      skyframeBuildView.clearInvalidatedConfiguredTargets();
      // We also shrink the set of pending actions to avoid clear() being expensive in the nexts
      // builds.
      skyframeBuildView.unregisterPendingActionsAndShrink();
    }

    int numTargetsToAnalyze = nodes.size();
    int numSuccessful = configuredTargets.size();
    boolean analysisSuccessful = (numSuccessful == numTargetsToAnalyze);
    if (0 < numSuccessful && numSuccessful < numTargetsToAnalyze) {
      String msg = String.format("Analysis succeeded for only %d of %d top-level targets",
                                    numSuccessful, numTargetsToAnalyze);
      reporter.handle(Event.info(msg));
      LOG.info(msg);
    }

    postUpdateValidation(errorCollector, warningsHandler);

    AnalysisResult result = createResult(loadingResult, topLevelOptions,
        viewOptions, configuredTargets, analysisSuccessful);
    LOG.info("Finished analysis");
    this.reporter = null;
    return result;
  }

  // Validates that the update has been done correctly
  private void postUpdateValidation(ErrorCollector errorCollector,
      WarningsAsErrorsEventHandler warningsHandler) throws ViewCreationFailedException {
    if (warningsHandler != null && warningsHandler.warningsEncountered()) {
      throw new ViewCreationFailedException("Warnings being treated as errors");
    }

    if (errorCollector != null && !errorCollector.getEvents().isEmpty()) {
      // This assertion ensures that if any errors were reported during the
      // initialization phase, the call to configureTargets will fail with a
      // ViewCreationFailedException.  Violation of this invariant leads to
      // incorrect builds, because the fact that errors were encountered is not
      // properly recorded in the view (i.e. the graph of configured targets).
      // Rule errors must be reported via RuleConfiguredTarget.reportError,
      // which causes the rule's hasErrors() flag to be set, and thus the
      // hasErrors() flag of anything that depends on it transitively.  If the
      // toplevel rule hasErrors, then analysis is aborted and we do not
      // proceed to the execution phase of a build.
      //
      // Reporting errors directly through the Reporter does not set the error
      // flag, so analysis may succeed spuriously, allowing the execution
      // phase to begin with unpredictable consequences.
      //
      // The use of errorCollector (rather than an ErrorSensor) makes the
      // assertion failure messages more informative.
      // Note we tolerate errors iff --keep-going, because some of the
      // requested targets may have had problems during analysis, but that's ok.
      StringBuilder message = new StringBuilder("Unexpected errors reported during analysis:");
      for (Event event : errorCollector.getEvents()) {
        message.append('\n').append(event);
      }
      throw new IllegalStateException(message.toString());
    }
  }

  private AnalysisResult createResult(LoadingResult loadingResult,
      TopLevelArtifactContext topLevelOptions, BuildView.Options viewOptions,
      Collection<ConfiguredTarget> configuredTargets, boolean analysisSuccessful)
          throws InterruptedException {
    Collection<Target> testsToRun = loadingResult.getTestsToRun();
    Collection<ConfiguredTarget> targetsToTest = null;
    if (testsToRun != null) {
      // Determine the subset of configured targets that are meant to be run as tests.
      targetsToTest = Lists.newArrayList(
          filterTestsByTargets(configuredTargets, Sets.newHashSet(testsToRun)));
    }

    Multimap<ConfiguredTarget, Artifact> targetCompletionMap =
        createTargetCompletionMiddlemen(configuredTargets, topLevelOptions, skyframeExecutor);

    Set<Artifact> artifactsToBuild = new HashSet<>();
    Set<Artifact> exclusiveTestArtifacts = new HashSet<>();
    Collection<Artifact> buildInfoArtifacts;
    if (!skyframeExecutor.skyframeBuild()) {
      buildInfoArtifacts = ImmutableList.<Artifact>of(
          lastWorkspaceStatusArtifacts.getStableStatus(),
          lastWorkspaceStatusArtifacts.getVolatileStatus());
    } else {
      Preconditions.checkState(lastWorkspaceStatusArtifacts == null, lastWorkspaceStatusArtifacts);
      buildInfoArtifacts = skyframeExecutor.getWorkspaceStatusArtifacts();
    }
    // build-info and build-changelist.
    Preconditions.checkState(buildInfoArtifacts.size() == 2, buildInfoArtifacts);
    artifactsToBuild.addAll(buildInfoArtifacts);
    artifactsToBuild.addAll(targetCompletionMap.values());

    addExtraActionsIfRequested(viewOptions, artifactsToBuild, configuredTargets);
    // Note that this must come last, so that the tests are scheduled after all artifacts are built.
    scheduleTestsIfRequested(artifactsToBuild, exclusiveTestArtifacts,
        topLevelOptions, configuredTargets, targetsToTest);

    DependentActionGraph dependentActionGraph = null;
    if (!skyframeExecutor.skyframeBuild()) {
      dependentActionGraph = forwardGraphCache.get(artifactsToBuild, legacyActionGraph,
          viewOptions.keepForwardGraph);
    }

    String error = !loadingResult.hasLoadingError()
          ? (analysisSuccessful
            ? null
            : "execution phase succeeded, but not all targets were analyzed")
          : "execution phase succeeded, but there were loading phase errors";
    return new AnalysisResult(configuredTargets, targetsToTest, error, getActionGraph(),
        dependentActionGraph, targetCompletionMap, artifactsToBuild, exclusiveTestArtifacts);
  }

  private void addExtraActionsIfRequested(BuildView.Options viewOptions,
      Set<Artifact> artifactsToBuild, Iterable<ConfiguredTarget> topLevelTargets) {
    NestedSetBuilder<ExtraArtifactSet> builder = NestedSetBuilder.stableOrder();
    for (ConfiguredTarget topLevel : topLevelTargets) {
      ExtraActionArtifactsProvider provider = topLevel.getProvider(
          ExtraActionArtifactsProvider.class);
      if (provider != null) {
        if (viewOptions.extraActionTopLevelOnly) {
          builder.add(ExtraArtifactSet.of(topLevel.getLabel(), provider.getExtraActionArtifacts()));
        } else {
          builder.addTransitive(provider.getTransitiveExtraActionArtifacts());
        }
      }
    }

    RegexFilter filter = viewOptions.extraActionFilter;
    for (ExtraArtifactSet set : builder.build()) {
      boolean filterMatches = filter == null || filter.isIncluded(set.getLabel().toString());
      if (filterMatches) {
        Iterables.addAll(artifactsToBuild, set.getArtifacts());
      }
    }
  }

  private void scheduleTestsIfRequested(Set<Artifact> artifactsToBuild,
      Set<Artifact> exclusiveTestArtifacts, TopLevelArtifactContext topLevelOptions,
      Collection<ConfiguredTarget> targetsToBuild, Collection<ConfiguredTarget> testTargets) {
    // If requested, add test artifacts to the set and ensure correct scheduling dependencies for
    // exclusive tests.
    if (!topLevelOptions.compileOnly() && !topLevelOptions.compilationPrerequisitesOnly()
        && testTargets != null) {
      // Add baseline code coverage artifacts if we are collecting code coverage. We do that only
      // when running tests.
      for (ConfiguredTarget target : targetsToBuild) {
        // It might be slightly faster to first check if any configuration has coverage enabled.
        if (target.getConfiguration() != null
            && target.getConfiguration().isCodeCoverageEnabled()) {
          BaselineCoverageArtifactsProvider provider =
              target.getProvider(BaselineCoverageArtifactsProvider.class);
          if (provider != null) {
            Iterables.addAll(artifactsToBuild, provider.getBaselineCoverageArtifacts());
          }
        }
      }

      // Schedule tests.
      scheduleTests(artifactsToBuild, exclusiveTestArtifacts, testTargets,
                    topLevelOptions.runTestsExclusively());
    }
  }

  @VisibleForTesting
  List<TargetAndConfiguration> nodesForTargets(Collection<Target> targets) {
    // We use a hash set here to remove duplicate nodes; this can happen for input files and package
    // groups.
    LinkedHashSet<TargetAndConfiguration> nodes = new LinkedHashSet<>(targets.size());
    for (BuildConfiguration config : configurations.getTargetConfigurations()) {
      for (Target target : targets) {
        nodes.add(new TargetAndConfiguration(target,
            BuildConfigurationCollection.configureTopLevelTarget(config, target)));
      }
    }
    return ImmutableList.copyOf(nodes);
  }

  /**
   * Detects when a package root changes between instances of incremental analysis.
   *
   * <p>This case is currently problematic for incremental analysis because when a package root
   * changes, source artifacts with the new root will be created, but we can not be sure that there
   * are no references remaining to the corresponding artifacts with the old root.
   */
  private boolean buildHasIncompatiblePackageRoots(Map<PathFragment, Path> packageRoots) {
    for (Map.Entry<PathFragment, Path> entry : packageRoots.entrySet()) {
      Path prevRoot = cumulativePackageRoots.get(entry.getKey());
      if (prevRoot != null && !entry.getValue().equals(prevRoot)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns an existing ConfiguredTarget for the specified target and
   * configuration, or null if none exists.  No validity check is done.
   */
  @ThreadSafe
  public ConfiguredTarget getExistingConfiguredTarget(Target target, BuildConfiguration config) {
    return getExistingConfiguredTarget(new TargetAndConfiguration(target, config));
  }

  /**
   * Returns an existing ConfiguredTarget for the specified node, or null if none exists. No
   * validity check is done.
   */
  @ThreadSafe
  private ConfiguredTarget getExistingConfiguredTarget(TargetAndConfiguration node) {
    return Iterables.getFirst(getExistingConfiguredTargets(ImmutableList.of(node)), null);
  }

  private Iterable<ConfiguredTarget> getExistingConfiguredTargets(
      Iterable<TargetAndConfiguration> nodes) {
    Iterable<LabelAndConfiguration> keys =
        Iterables.transform(nodes, TargetAndConfiguration.TO_LABEL_AND_CONFIGURATION);
    return skyframeExecutor.getConfiguredTargets(keys);
  }

  @VisibleForTesting
  ListMultimap<Attribute, ConfiguredTarget> getPrerequisiteMapForTesting(ConfiguredTarget target) {
    ListMultimap<Attribute, ConfiguredTarget> prerequisiteMap = ArrayListMultimap.create();

    DependencyResolver resolver = new DependencyResolver() {
      @Override
      protected void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label) {
        throw new RuntimeException("bad visibility on " + label + " during testing unexpected");
      }

      @Override
      protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
        throw new RuntimeException(
            "bad package group on " + label + " during testing unexpected");
      }

      @Override
      protected Target getTarget(Label label) throws NoSuchThingException {
        return packageManager.getLoadedTarget(label);
      }
    };
    TargetAndConfiguration ctNode = new TargetAndConfiguration(target);
    for (Map.Entry<Attribute, TargetAndConfiguration> entry :
        resolver.dependentNodeMap(ctNode).entries()) {
      prerequisiteMap.put(entry.getKey(), getExistingConfiguredTarget(entry.getValue()));
    }

    return prerequisiteMap;
  }

  private void scheduleTestsSkyframe(Collection<Artifact> artifactsToBuild,
                                     Set<Artifact> exclusiveTestArtifacts,
                                     Collection<ConfiguredTarget> testTargets,
                                     boolean isExclusive) {
    Preconditions.checkState(lastExclusiveSchedulingMiddlemen.isEmpty());
    for (ConfiguredTarget target : testTargets) {
      if (target.getTarget() instanceof Rule) {
        boolean exclusive =
            isExclusive || TargetUtils.isExclusiveTestRule((Rule) target.getTarget());
        Collection<Artifact> artifacts = exclusive ? exclusiveTestArtifacts : artifactsToBuild;
        artifacts.addAll(TestProvider.getTestStatusArtifacts(target));
      }
    }
  }

  /**
   * Returns set of artifacts representing test results. Also serializes
   * execution of the exclusive tests using scheduling middleman dependencies.
   */
  private void scheduleTests(Collection<Artifact> artifactsToBuild,
                             Set<Artifact> exclusiveTestArtifacts,
                             Collection<ConfiguredTarget> testTargets,
                             boolean isExclusive) {
    if (skyframeExecutor.skyframeBuild()) {
      scheduleTestsSkyframe(artifactsToBuild, exclusiveTestArtifacts, testTargets, isExclusive);
      return;
    }
    Set<Artifact> artifactsToTest = new LinkedHashSet<>();
    MiddlemanFactory middlemanFactory = new MiddlemanFactory(artifactFactory, ActionRegistry.NOP);

    for (Action oldAction : lastExclusiveSchedulingMiddlemen) {
      forwardGraphCache.removeAction(oldAction);
      legacyActionGraph.unregisterAction(oldAction);
    }

    lastExclusiveSchedulingMiddlemen.clear();

    // First process non-exclusive tests.
    if (!isExclusive) {
      for (ConfiguredTarget target : testTargets) {
        if (!(target.getTarget() instanceof Rule)) {
          continue;
        }

        if (!TargetUtils.isExclusiveTestRule((Rule) target.getTarget())) {
          // Non-exclusive tests do not have any scheduling dependencies and
          // can be executed as soon as "normal" prerequisites are built.
          // We need to explicitly set them to null, because of possibility that
          // previous blaze invocation used --test_strategy=exclusive and the
          // current one used analysis caching.
          for (Artifact artifact : TestProvider.getTestStatusArtifacts(target)) {
            TestRunnerAction action = (TestRunnerAction)
                legacyActionGraph.getGeneratingAction(artifact);
            Pair<Artifact, Action> middlemanAndStamp = action.setSchedulingDependencies(
                getArtifactFactory(), middlemanFactory, null, forwardGraphCache);
            if (middlemanAndStamp != null) {
              lastExclusiveSchedulingMiddlemen.add(middlemanAndStamp.getSecond());
              registerAction(middlemanAndStamp.getSecond());
              forwardGraphCache.add(action, middlemanAndStamp.getFirst());
            }
            artifactsToTest.add(artifact);
          }
        }
      }
    }

    // Rest of test targets are exclusive. First of them must depend on the
    // the scheduling middleman covering all generated artifacts and all
    // non-exclusive tests, so it will run only after all other activities are
    // completed. Subsequent exclusive tests will depend on each other, forming
    // sequential dependency.
    List<Artifact> nonExclusiveArtifacts = new ArrayList<>(artifactsToTest);
    for (Artifact buildArtifact : artifactsToBuild) {
      if (!buildArtifact.isSourceArtifact()) {
        nonExclusiveArtifacts.add(buildArtifact);
      }
    }
    Collection<Artifact> dependencies = nonExclusiveArtifacts;

    for (ConfiguredTarget target : testTargets) {
      for (Artifact artifact : TestProvider.getTestStatusArtifacts(target)) {
        // If artifact is already in the artifactsToTest set, then target was
        // already processed as a non-exclusive test and should be skipped.
        if (!artifactsToTest.contains(artifact)) {
          TestRunnerAction action = (TestRunnerAction)
              legacyActionGraph.getGeneratingAction(artifact);
          // This is the heart of the exclusive test scheduling.
          // Test action inputs are then modified to include single additional
          // input (see {@link TestRunnerAction#getInputs} and
          // {@link TestRunnerAction#setSchedulingDependencies} methods), which
          // is a scheduling middleman.
          // While current implementation breaks the implicit rule that whole
          // graph is constructed within BuildView.update() method,  it has been agreed upon as
          // an interim solution. This method is still called prior to the
          // execution phase, so technically it is still considered to be part of
          // the analysis phase.
          Pair<Artifact, Action> middlemanAndStamp = action.setSchedulingDependencies(
              getArtifactFactory(), middlemanFactory, dependencies, forwardGraphCache);
          if (middlemanAndStamp != null) {
            lastExclusiveSchedulingMiddlemen.add(middlemanAndStamp.getSecond());
            registerAction(middlemanAndStamp.getSecond());
            forwardGraphCache.add(action, middlemanAndStamp.getFirst());
          }
          dependencies = Collections.singleton(artifact);
          artifactsToTest.add(artifact);
        }
      }
    }
    artifactsToBuild.addAll(artifactsToTest);
  }

  /**
   * Sets the possible artifact roots in the artifact factory. This allows the
   * factory to resolve paths with unknown roots to artifacts.
   * <p>
   * <em>Note: This must be called before any call to
   * {@link #getConfiguredTarget(Label, BuildConfiguration)}
   * </em>
   */
  @VisibleForTesting // for BuildViewTestCase
  void setArtifactRoots(ImmutableMap<PathFragment, Path> packageRoots) {
    Map<Path, Root> rootMap = new HashMap<>();
    Map<PathFragment, Root> realPackageRoots = new HashMap<>();
    for (Map.Entry<PathFragment, Path> entry : packageRoots.entrySet()) {
      Root root = rootMap.get(entry.getValue());
      if (root == null) {
        root = Root.asSourceRoot(entry.getValue());
        rootMap.put(entry.getValue(), root);
      }
      realPackageRoots.put(entry.getKey(), root);
    }
    // Source Artifact roots:
    artifactFactory.setPackageRoots(realPackageRoots);

    // Derived Artifact roots:
    ImmutableList.Builder<Root> roots = ImmutableList.builder();

    // build-info.txt and friends; this root is not configuration specific.
    roots.add(directories.getBuildDataDirectory());

    // The roots for each configuration - duplicates are automatically removed in the call below.
    for (BuildConfiguration cfg : configurations.getAllConfigurations()) {
      roots.addAll(cfg.getRoots());
    }

    artifactFactory.setDerivedArtifactRoots(roots.build());
  }

  /**
   * Returns a configured target for the specified target and configuration.
   * This should only be called from test cases, and is needed, because
   * plain {@link #getConfiguredTarget(Target, BuildConfiguration)} does not
   * construct the configured target graph, and would thus fail if called from
   * outside an update.
   */
  @VisibleForTesting
  public ConfiguredTarget getConfiguredTargetForTesting(Label label, BuildConfiguration config)
      throws NoSuchPackageException, NoSuchTargetException {
    return getConfiguredTargetForTesting(packageManager.getLoadedTarget(label), config);
  }

  @VisibleForTesting
  public ConfiguredTarget getConfiguredTargetForTesting(Target target, BuildConfiguration config) {
    return skyframeExecutor.getConfiguredTargetForTesting(target.getLabel(), config);
  }

  /**
   * Returns a RuleContext which is the same as the original RuleContext of the target parameter.
   */
  @VisibleForTesting
  public RuleContext getRuleContextForTesting(ConfiguredTarget target,
      StoredEventHandler eventHandler) {
    BuildConfiguration config = target.getConfiguration();
    CachingAnalysisEnvironment analysisEnvironment =
        new CachingAnalysisEnvironment(artifactFactory,
            new LabelAndConfiguration(target.getLabel(), config),
            lastWorkspaceStatusArtifacts, /*isSystemEnv=*/false, config.extendedSanityChecks(),
            eventHandler,
            /*skyframeEnv=*/null, config.isActionsEnabled(), outputFormatters, binTools);
    RuleContext ruleContext = new RuleContext.Builder(analysisEnvironment,
        (Rule) target.getTarget(), config, ruleClassProvider.getPrerequisiteValidator())
            .setVisibility(NestedSetBuilder.<PackageSpecification>create(
                Order.STABLE_ORDER, PackageSpecification.EVERYTHING))
            .setPrerequisites(getPrerequisiteMapForTesting(target))
            .setConfigConditions(ImmutableSet.<ConfigMatchingProvider>of())
            .build();
    return ruleContext;
  }

  /**
   * Tests and clears the current thread's pending "interrupted" status, and
   * throws InterruptedException iff it was set.
   */
  protected final void pollInterruptedStatus() throws InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
  }

  /**
   * Returns a newIdentityHashMap with artifacts stored in the artifact factory.
   * Used to validate and dump internal data structures.
   */
  @VisibleForTesting
  Map<Artifact, Boolean> getArtifactReferences() {
    Map<Artifact, Boolean> artifactMap = Maps.newIdentityHashMap();
    for (Artifact artifact : artifactFactory.getArtifacts()) {
      artifactMap.put(artifact, Boolean.FALSE);
    }
    return artifactMap;
  }

  /**
   * Drops the analysis cache. If building with Skyframe, targets in {@code topLevelTargets} may
   * remain in the cache for use during the execution phase.
   *
   * @see BuildView.Options#discardAnalysisCache
   */
  public void clearAnalysisCache(Collection<ConfiguredTarget> topLevelTargets) {
    // TODO(bazel-team): Consider clearing packages too to save more memory.
    skyframeAnalysisWasDiscarded = true;
    skyframeExecutor.clearAnalysisCache(topLevelTargets);
  }

  /********************************************************************
   *                                                                  *
   *                  'blaze dump' related functions                  *
   *                                                                  *
   ********************************************************************/

  /**
   * Returns developer-friendly artifact name used for dumps.
   */
  private static String dumpArtifact(Artifact artifact) {
    return (artifact.getExecPath() != null ? artifact.getExecPathString() : artifact.prettyPrint())
        + ", " + System.identityHashCode(artifact);
  }

  /**
   * Returns developer-friendly action name used for dumps.
   */
  private static String dumpAction(Action action) {
    return action != null ? action.prettyPrint() + ", " + System.identityHashCode(action) : "";
  }

  /**
   * Dumps state of the artifact factory and referenced actions.
   */
  public void dumpArtifacts(PrintStream out) {
    // Map below associates every artifact with the Boolean.FALSE value. During
    // traversal, values for visited artifacts will be set to the Boolean.TRUE.
    // Any artifacts associated with Boolean.FALSE value will then be reported
    // as not referenced.
    Map<Artifact, Boolean> artifactReferenceMap = getArtifactReferences();
    Map<String, Action> actionMap = new HashMap<>();
    out.println("Artifact factory (" + artifactReferenceMap.size() + " artifacts)");

    List<Artifact> artifacts = new ArrayList<>(artifactReferenceMap.keySet());
    Collections.sort(artifacts, new Comparator<Artifact> () {
      @Override
      public int compare(Artifact o1, Artifact o2) {
        return o1.getPath().compareTo(o2.getPath());
      }
    });
    for (Artifact artifact : artifacts) {
      out.println(dumpArtifact(artifact));
      Action action = legacyActionGraph.getGeneratingAction(artifact);
      if (action != null) {
        out.println("  " + dumpAction(action));
        if (action.getOutputs() == null || action.getOutputs().size() == 0) {
          out.println("    !!! action does not have an output !!!");
        } else {
          // Action annotation is defined in AbstractAction constructor and is
          // used by the profiler. It is assumed to be unique for each action.
          String annotation = action.getClass().getSimpleName()
              + action.getPrimaryOutput();
          Action identicalAction = actionMap.put(annotation, action);
          if (identicalAction != null && identicalAction != action) {
            out.println ("    !!! " + dumpAction(action) + " HAS IDENTICAL ANNOTATION !!!");
          }
        }
        for (Artifact input : action.getInputs()) {
          out.println("    input " + dumpArtifact(input));
          if (artifactReferenceMap.put(input, Boolean.TRUE) == null) {
            artifactReferenceMap.remove(input);
          }
        }
        boolean found = false;
        for (Artifact output : action.getOutputs()) {
          if (output.equals(artifact)) {
            found = true;
          }
          out.println("    output " + dumpArtifact(output));
          if (artifactReferenceMap.put(output, Boolean.TRUE) == null) {
            artifactReferenceMap.remove(output);
          }
        }
        if (!found) {
          out.println("  !!! ACTION DOES NOT HAVE PARENT ARTIFACT AS AN OUTPUT !!!");
        }
      }
    }
    out.println();
    out.println("The following artifacts were NEVER referenced by actions:");
    for (Artifact artifact : artifacts) {
      if (!artifactReferenceMap.get(artifact)) {
        out.println("  " + dumpArtifact(artifact));
        if (!artifact.isSourceArtifact()) {
          out.println("    " + dumpAction(legacyActionGraph.getGeneratingAction(artifact)));
        }
      }
    }
  }

  void registerAction(Action action) {
    legacyActionGraph.registerAction(action);
  }

  /**
   * A cache of the forward action graph.
   */
  private static final class ForwardGraphCache implements TestMiddlemanObserver {
    private Set<Artifact> topLevelArtifacts;
    private DependentActionGraph graph;

    public DependentActionGraph get(Set<Artifact> topLevelArtifacts, ActionGraph actionGraph,
        boolean stable) {
      Preconditions.checkNotNull(topLevelArtifacts);
      if (!stable) {
        clear();
        return DependentActionGraph.newGraph(topLevelArtifacts, actionGraph, false);
      }

      if (topLevelArtifacts.equals(this.topLevelArtifacts)) {
        graph.sync();
        return graph;
      }

      // TODO(bazel-team): raw reference copy seems dangerous here, however the argument is a
      // LinkedHashSet; ImmutableSet.copyOf of that would screw up ordering.
      this.topLevelArtifacts = topLevelArtifacts;
      this.graph = DependentActionGraph.newGraph(topLevelArtifacts, actionGraph, true);
      return graph;
    }

    public void clear() {
      topLevelArtifacts = null;
      graph = null;
    }

    @Override
    public void remove(Action action, Artifact middleman, Action middlemanAction) {
      if (graph != null) {
        graph.clearMiddleman(action, middleman, middlemanAction);
      }
    }

    public void add(Action action, Artifact middleman) {
      if (graph != null) {
        graph.addMiddleman(action, middleman);
      }
    }

    public void addAction(Action action) {
      if (graph != null) {
        graph.addAction(action);
      }
    }

    public void removeAction(Action action) {
      if (graph != null) {
        graph.removeAction(action);
      }
    }
  }

  /**
   * Collects and stores error events while also forwarding them to another eventHandler.
   */
  public static class ErrorCollector extends DelegatingEventHandler {
    private final List<Event> events;

    public ErrorCollector(EventHandler delegate) {
      super(delegate);
      this.events = Lists.newArrayList();
    }

    public List<Event> getEvents() {
      return events;
    }

    @Override
    public void handle(Event e) {
      super.handle(e);
      if (e.getKind() == EventKind.ERROR) {
        events.add(e);
      }
    }
  }
}
