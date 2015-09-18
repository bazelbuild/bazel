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

package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.DependencyResolver.Dependency;
import com.google.devtools.build.lib.analysis.ExtraActionArtifactsProvider.ExtraArtifactSet;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner.LoadingResult;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.rules.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.rules.test.CoverageReportActionFactory.CoverageReportActionsWrapper;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.skyframe.ActionLookupValue;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.CoverageReportValue;
import com.google.devtools.build.lib.skyframe.SkyframeAnalysisResult;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
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
   * com.google.devtools.build.lib.analysis.config.BuildOptions}, which affect the <i>value</i>
   * of a BuildConfiguration.
   */
  public static class Options extends OptionsBase {

    @Option(name = "keep_going",
            abbrev = 'k',
            defaultValue = "false",
            category = "strategy",
            help = "Continue as much as possible after an error.  While the"
                + " target that failed, and those that depend on it, cannot be"
                + " analyzed (or built), the other prerequisites of these"
                + " targets can be analyzed (or built) all the same.")
    public boolean keepGoing;

    @Option(name = "analysis_warnings_as_errors",
            deprecationWarning = "analysis_warnings_as_errors is now a no-op and will be removed in"
                              + " an upcoming Blaze release",
            defaultValue = "false",
            category = "strategy",
            help = "Treat visible analysis warnings as errors.")
    public boolean analysisWarningsAsErrors;

    @Option(name = "discard_analysis_cache",
            defaultValue = "false",
            category = "strategy",
            help = "Discard the analysis cache immediately after the analysis phase completes."
                + " Reduces memory usage by ~10%, but makes further incremental builds slower.")
    public boolean discardAnalysisCache;

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

    @Option(name = "experimental_interleave_loading_and_analysis",
            defaultValue = "false",
            category = "experimental",
            help = "Interleave loading and analysis phases, so that one target may be analyzed at"
                + " the same time as an unrelated target is loaded.")
    public boolean interleaveLoadingAndAnalysis;

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

  private final SkyframeExecutor skyframeExecutor;
  private final SkyframeBuildView skyframeBuildView;

  // Same as skyframeExecutor.getPackageManager().
  private final PackageManager packageManager;

  private final BinTools binTools;

  private BuildConfigurationCollection configurations;

  private final ConfiguredRuleClassProvider ruleClassProvider;

  private final ArtifactFactory artifactFactory;

  /**
   * A factory class to create the coverage report action. May be null.
   */
  @Nullable private final CoverageReportActionFactory coverageReportActionFactory;

  /**
   * Used only for testing that we clear Skyframe caches correctly.
   * TODO(bazel-team): Remove this once we get rid of legacy Skyframe synchronization.
   */
  private boolean skyframeCacheWasInvalidated;

  /**
   * If the last build was executed with {@code Options#discard_analysis_cache} and we are not
   * running Skyframe full, we should clear the legacy data since it is out-of-sync.
   */
  private boolean skyframeAnalysisWasDiscarded;

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

  public BuildView(BlazeDirectories directories,
      ConfiguredRuleClassProvider ruleClassProvider,
      SkyframeExecutor skyframeExecutor,
      BinTools binTools, CoverageReportActionFactory coverageReportActionFactory) {
    this.directories = directories;
    this.packageManager = skyframeExecutor.getPackageManager();
    this.binTools = binTools;
    this.coverageReportActionFactory = coverageReportActionFactory;
    this.artifactFactory = new ArtifactFactory(directories.getExecRoot());
    this.ruleClassProvider = ruleClassProvider;
    this.skyframeExecutor = Preconditions.checkNotNull(skyframeExecutor);
    this.skyframeBuildView =
        new SkyframeBuildView(
            new ConfiguredTargetFactory(ruleClassProvider),
            artifactFactory,
            skyframeExecutor,
            new Runnable() {
              @Override
              public void run() {
                clear();
              }
            },
            binTools,
            ruleClassProvider);
    skyframeExecutor.setSkyframeBuildView(skyframeBuildView);
  }

  /** Returns the action graph. */
  public ActionGraph getActionGraph() {
    return new ActionGraph() {
        @Override
        public Action getGeneratingAction(Artifact artifact) {
          return skyframeExecutor.getGeneratingAction(artifact);
        }
      };
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
  public void setConfigurationsForTesting(BuildConfigurationCollection configurations) {
    this.configurations = configurations;
    skyframeBuildView.setTopLevelHostConfiguration(configurations.getHostConfiguration());
  }

  public BuildConfigurationCollection getConfigurationCollection() {
    return configurations;
  }

  /**
   * Clear the graphs of ConfiguredTargets and Artifacts.
   */
  @VisibleForTesting
  public void clear() {
    artifactFactory.clear();
  }

  public ArtifactFactory getArtifactFactory() {
    return artifactFactory;
  }

  @VisibleForTesting
  WorkspaceStatusAction getLastWorkspaceBuildInfoActionForTesting() {
    return skyframeExecutor.getLastWorkspaceStatusActionForTesting();
  }

  /**
   * Returns a corresponding ConfiguredTarget, if one exists; otherwise throws an {@link
   * NoSuchConfiguredTargetException}.
   */
  @ThreadSafe
  private ConfiguredTarget getConfiguredTarget(Target target, BuildConfiguration config)
      throws NoSuchConfiguredTargetException {
    ConfiguredTarget result =
        getExistingConfiguredTarget(target.getLabel(), config);
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

  public Iterable<ConfiguredTarget> getDirectPrerequisites(ConfiguredTarget ct)
      throws InterruptedException {
    return getDirectPrerequisites(ct, null);
  }

  public Iterable<ConfiguredTarget> getDirectPrerequisites(
      ConfiguredTarget ct, @Nullable final LoadingCache<Label, Target> targetCache)
      throws InterruptedException {
    return skyframeExecutor.getConfiguredTargets(ct.getConfiguration(),
        getDirectPrerequisiteDependencies(ct, targetCache), false);
  }

  public Iterable<Dependency> getDirectPrerequisiteDependencies(
      ConfiguredTarget ct, @Nullable final LoadingCache<Label, Target> targetCache)
      throws InterruptedException {
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
    return dependencyResolver.dependentNodes(ctgNode, configurations.getHostConfiguration(),
        getConfigurableAttributeKeys(ctgNode));
  }

  /**
   * Returns ConfigMatchingProvider instances corresponding to the configurable attribute keys
   * present in this rule's attributes.
   */
  private Set<ConfigMatchingProvider> getConfigurableAttributeKeys(TargetAndConfiguration ctg) {
    if (!(ctg.getTarget() instanceof Rule)) {
      return ImmutableSet.of();
    }
    Rule rule = (Rule) ctg.getTarget();
    ImmutableSet.Builder<ConfigMatchingProvider> keys = ImmutableSet.builder();
    RawAttributeMapper mapper = RawAttributeMapper.of(rule);
    for (Attribute attribute : rule.getAttributes()) {
      for (Label label : mapper.getConfigurabilityKeys(attribute.getName(), attribute.getType())) {
        if (BuildType.Selector.isReservedLabel(label)) {
          continue;
        }
        try {
          ConfiguredTarget ct = getConfiguredTarget(label, ctg.getConfiguration());
          keys.add(Preconditions.checkNotNull(ct.getProvider(ConfigMatchingProvider.class)));
        } catch (
            NoSuchPackageException | NoSuchTargetException | NoSuchConfiguredTargetException e) {
          // All lookups should succeed because we should not be looking up any targets in error.
          throw new IllegalStateException(e);
        }
      }
    }
    return keys.build();
  }

  public TransitiveInfoCollection getGeneratingRule(OutputFileConfiguredTarget target) {
    return target.getGeneratingRule();
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();  // avoid nondeterminism
  }

  /**
   * Return value for {@link BuildView#update} and {@code BuildTool.prepareToBuild}.
   */
  public static final class AnalysisResult {

    public static final AnalysisResult EMPTY =
        new AnalysisResult(
            ImmutableList.<ConfiguredTarget>of(),
            ImmutableList.<AspectValue>of(),
            null,
            null,
            null,
            ImmutableList.<Artifact>of(),
            ImmutableList.<ConfiguredTarget>of(),
            ImmutableList.<ConfiguredTarget>of(),
            null,
            ImmutableMap.<PackageIdentifier, Path>of());

    private final ImmutableList<ConfiguredTarget> targetsToBuild;
    @Nullable private final ImmutableList<ConfiguredTarget> targetsToTest;
    @Nullable private final String error;
    private final ActionGraph actionGraph;
    private final ImmutableSet<Artifact> artifactsToBuild;
    private final ImmutableSet<ConfiguredTarget> parallelTests;
    private final ImmutableSet<ConfiguredTarget> exclusiveTests;
    @Nullable private final TopLevelArtifactContext topLevelContext;
    private final ImmutableList<AspectValue> aspects;
    private final ImmutableMap<PackageIdentifier, Path> packageRoots;

    private AnalysisResult(
        Collection<ConfiguredTarget> targetsToBuild,
        Collection<AspectValue> aspects,
        Collection<ConfiguredTarget> targetsToTest,
        @Nullable String error,
        ActionGraph actionGraph,
        Collection<Artifact> artifactsToBuild,
        Collection<ConfiguredTarget> parallelTests,
        Collection<ConfiguredTarget> exclusiveTests,
        TopLevelArtifactContext topLevelContext,
        ImmutableMap<PackageIdentifier, Path> packageRoots) {
      this.targetsToBuild = ImmutableList.copyOf(targetsToBuild);
      this.aspects = ImmutableList.copyOf(aspects);
      this.targetsToTest = targetsToTest == null ? null : ImmutableList.copyOf(targetsToTest);
      this.error = error;
      this.actionGraph = actionGraph;
      this.artifactsToBuild = ImmutableSet.copyOf(artifactsToBuild);
      this.parallelTests = ImmutableSet.copyOf(parallelTests);
      this.exclusiveTests = ImmutableSet.copyOf(exclusiveTests);
      this.topLevelContext = topLevelContext;
      this.packageRoots = packageRoots;
    }

    /**
     * Returns configured targets to build.
     */
    public Collection<ConfiguredTarget> getTargetsToBuild() {
      return targetsToBuild;
    }

    /**
     * The map from package names to the package root where each package was found; this is used to
     * set up the symlink tree.
     */
    public ImmutableMap<PackageIdentifier, Path> getPackageRoots() {
      return packageRoots;
    }

    /**
     * Returns aspects of configured targets to build.
     *
     * <p>If this list is empty, build the targets returned by {@code getTargetsToBuild()}.
     * Otherwise, only build these aspects of the targets returned by {@code getTargetsToBuild()}.
     */
    public Collection<AspectValue> getAspects() {
      return aspects;
    }

    /**
     * Returns the configured targets to run as tests, or {@code null} if testing was not
     * requested (e.g. "build" command rather than "test" command).
     */
    @Nullable
    public Collection<ConfiguredTarget> getTargetsToTest() {
      return targetsToTest;
    }

    public ImmutableSet<Artifact> getAdditionalArtifactsToBuild() {
      return artifactsToBuild;
    }

    public ImmutableSet<ConfiguredTarget> getExclusiveTests() {
      return exclusiveTests;
    }

    public ImmutableSet<ConfiguredTarget> getParallelTests() {
      return parallelTests;
    }

    /**
     * Returns an error description (if any).
     */
    @Nullable public String getError() {
      return error;
    }

    public boolean hasError() {
      return error != null;
    }

    /**
     * Returns the action graph.
     */
    public ActionGraph getActionGraph() {
      return actionGraph;
    }

    public TopLevelArtifactContext getTopLevelContext() {
      return topLevelContext;
    }
  }


  /**
   * Returns the collection of configured targets corresponding to any of the provided targets.
   */
  @VisibleForTesting
  static Iterable<? extends ConfiguredTarget> filterTestsByTargets(
      Collection<? extends ConfiguredTarget> targets,
      final Set<? extends Target> allowedTargets) {
    return Iterables.filter(
        targets,
        new Predicate<ConfiguredTarget>() {
          @Override
          public boolean apply(ConfiguredTarget rule) {
            return allowedTargets.contains(rule.getTarget());
          }
        });
  }

  private void prepareToBuild(BuildConfigurationCollection configurations,
      PackageRootResolver resolver) throws ViewCreationFailedException {
    for (BuildConfiguration config : configurations.getAllConfigurations()) {
      config.prepareToBuild(directories.getExecRoot(), getArtifactFactory(), resolver);
    }
  }

  @ThreadCompatible
  public AnalysisResult update(
      LoadingResult loadingResult,
      BuildConfigurationCollection configurations,
      List<String> aspects,
      Options viewOptions,
      TopLevelArtifactContext topLevelOptions,
      EventHandler eventHandler,
      EventBus eventBus,
      boolean loadingEnabled)
      throws ViewCreationFailedException, InterruptedException {
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
    if ((this.configurations != null && !configurations.equals(this.configurations))
        || skyframeAnalysisWasDiscarded) {
      LOG.info("Discarding analysis cache: configurations have changed.");

      skyframeExecutor.dropConfiguredTargets();
      skyframeCacheWasInvalidated = true;
      clear();
    }
    skyframeAnalysisWasDiscarded = false;
    this.configurations = configurations;
    skyframeBuildView.setTopLevelHostConfiguration(this.configurations.getHostConfiguration());

    // Determine the configurations.
    List<TargetAndConfiguration> nodes = nodesForTargets(configurations, targets);

    List<ConfiguredTargetKey> targetSpecs =
        Lists.transform(nodes, new Function<TargetAndConfiguration, ConfiguredTargetKey>() {
          @Override
          public ConfiguredTargetKey apply(TargetAndConfiguration node) {
            return new ConfiguredTargetKey(node.getLabel(), node.getConfiguration());
          }
        });

    List<AspectKey> aspectKeys = new ArrayList<>();
    for (String aspect : aspects) {
      @SuppressWarnings("unchecked")
      final Class<? extends ConfiguredAspectFactory> aspectFactoryClass =
          (Class<? extends ConfiguredAspectFactory>)
              ruleClassProvider.getAspectFactoryMap().get(aspect);
      if (aspectFactoryClass != null) {
        for (ConfiguredTargetKey targetSpec : targetSpecs) {
          aspectKeys.add(
              AspectValue.createAspectKey(
                  targetSpec.getLabel(), targetSpec.getConfiguration(), aspectFactoryClass));
        }
      } else {
        throw new ViewCreationFailedException("Aspect '" + aspect + "' is unknown");
      }
    }

    // Configuration of some BuildConfiguration.Fragments may require information about
    // artifactRoots, so we need to set them before calling prepareToBuild. In that case loading
    // phase has to be enabled.
    if (loadingEnabled) {
      setArtifactRoots(loadingResult.getPackageRoots(), configurations);
    }
    prepareToBuild(configurations, new SkyframePackageRootResolver(skyframeExecutor));
    skyframeExecutor.injectWorkspaceStatusData();
    SkyframeAnalysisResult skyframeAnalysisResult;
    try {
      skyframeAnalysisResult =
          skyframeBuildView.configureTargets(
              targetSpecs, aspectKeys, eventBus, viewOptions.keepGoing);
      setArtifactRoots(skyframeAnalysisResult.getPackageRoots(), configurations);
    } finally {
      skyframeBuildView.clearInvalidatedConfiguredTargets();
    }

    int numTargetsToAnalyze = nodes.size();
    int numSuccessful = skyframeAnalysisResult.getConfiguredTargets().size();
    boolean analysisSuccessful = (numSuccessful == numTargetsToAnalyze);
    if (0 < numSuccessful && numSuccessful < numTargetsToAnalyze) {
      String msg = String.format("Analysis succeeded for only %d of %d top-level targets",
                                    numSuccessful, numTargetsToAnalyze);
      eventHandler.handle(Event.info(msg));
      LOG.info(msg);
    }

    AnalysisResult result =
        createResult(
            loadingResult,
            topLevelOptions,
            viewOptions,
            skyframeAnalysisResult.getConfiguredTargets(),
            skyframeAnalysisResult.getAspects(),
            skyframeAnalysisResult.getWalkableGraph(),
            skyframeAnalysisResult.getPackageRoots(),
            analysisSuccessful);
    LOG.info("Finished analysis");
    return result;
  }

  private AnalysisResult createResult(
      LoadingResult loadingResult,
      TopLevelArtifactContext topLevelOptions,
      BuildView.Options viewOptions,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<AspectValue> aspects,
      final WalkableGraph graph,
      ImmutableMap<PackageIdentifier, Path> packageRoots,
      boolean analysisSuccessful)
      throws InterruptedException {
    Collection<Target> testsToRun = loadingResult.getTestsToRun();
    Collection<ConfiguredTarget> allTargetsToTest = null;
    if (testsToRun != null) {
      // Determine the subset of configured targets that are meant to be run as tests.
      allTargetsToTest = Lists.newArrayList(
          filterTestsByTargets(configuredTargets, Sets.newHashSet(testsToRun)));
    }

    Set<Artifact> artifactsToBuild = new HashSet<>();
    Set<ConfiguredTarget> parallelTests = new HashSet<>();
    Set<ConfiguredTarget> exclusiveTests = new HashSet<>();

    // build-info and build-changelist.
    Collection<Artifact> buildInfoArtifacts = skyframeExecutor.getWorkspaceStatusArtifacts();
    Preconditions.checkState(buildInfoArtifacts.size() == 2, buildInfoArtifacts);
    artifactsToBuild.addAll(buildInfoArtifacts);

    // Extra actions
    addExtraActionsIfRequested(viewOptions, artifactsToBuild, configuredTargets);

    // Coverage
    NestedSet<Artifact> baselineCoverageArtifacts = getBaselineCoverageArtifacts(configuredTargets);
    Iterables.addAll(artifactsToBuild, baselineCoverageArtifacts);
    if (coverageReportActionFactory != null) {
      CoverageReportActionsWrapper actionsWrapper;
      actionsWrapper = coverageReportActionFactory.createCoverageReportActionsWrapper(
          allTargetsToTest,
          baselineCoverageArtifacts,
          artifactFactory,
          CoverageReportValue.ARTIFACT_OWNER);
      if (actionsWrapper != null) {
        ImmutableList <Action> actions = actionsWrapper.getActions();
        skyframeExecutor.injectCoverageReportData(actions);
        artifactsToBuild.addAll(actionsWrapper.getCoverageOutputs());
      }
    }

    // Tests. This must come last, so that the exclusive tests are scheduled after everything else.
    scheduleTestsIfRequested(parallelTests, exclusiveTests, topLevelOptions, allTargetsToTest);

    String error = !loadingResult.hasLoadingError()
          ? (analysisSuccessful
            ? null
            : "execution phase succeeded, but not all targets were analyzed")
          : "execution phase succeeded, but there were loading phase errors";

    final ActionGraph actionGraph = new ActionGraph() {
      @Nullable
      @Override
      public Action getGeneratingAction(Artifact artifact) {
        ArtifactOwner artifactOwner = artifact.getArtifactOwner();
        if (artifactOwner instanceof ActionLookupValue.ActionLookupKey) {
          SkyKey key = ActionLookupValue.key((ActionLookupValue.ActionLookupKey) artifactOwner);
          ActionLookupValue val = (ActionLookupValue) graph.getValue(key);
          return val == null ? null : val.getGeneratingAction(artifact);
        }
        return null;
      }
    };
    return new AnalysisResult(
        configuredTargets,
        aspects,
        allTargetsToTest,
        error,
        actionGraph,
        artifactsToBuild,
        parallelTests,
        exclusiveTests,
        topLevelOptions,
        packageRoots);
  }

  private static NestedSet<Artifact> getBaselineCoverageArtifacts(
      Collection<ConfiguredTarget> configuredTargets) {
    NestedSetBuilder<Artifact> baselineCoverageArtifacts = NestedSetBuilder.stableOrder();
    for (ConfiguredTarget target : configuredTargets) {
      InstrumentedFilesProvider provider = target.getProvider(InstrumentedFilesProvider.class);
      if (provider != null) {
        baselineCoverageArtifacts.addTransitive(provider.getBaselineCoverageArtifacts());
      }
    }
    return baselineCoverageArtifacts.build();
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
        artifactsToBuild.addAll(set.getArtifacts());
      }
    }
  }

  private static void scheduleTestsIfRequested(Collection<ConfiguredTarget> targetsToTest,
      Collection<ConfiguredTarget> targetsToTestExclusive, TopLevelArtifactContext topLevelOptions,
      Collection<ConfiguredTarget> allTestTargets) {
    Set<String> outputGroups = topLevelOptions.outputGroups();
    if (!outputGroups.contains(OutputGroupProvider.FILES_TO_COMPILE)
        && !outputGroups.contains(OutputGroupProvider.COMPILATION_PREREQUISITES)
        && allTestTargets != null) {
      scheduleTests(targetsToTest, targetsToTestExclusive, allTestTargets,
          topLevelOptions.runTestsExclusively());
    }
  }


  /**
   * Returns set of artifacts representing test results, writing into targetsToTest and
   * targetsToTestExclusive.
   */
  private static void scheduleTests(Collection<ConfiguredTarget> targetsToTest,
                                    Collection<ConfiguredTarget> targetsToTestExclusive,
                                    Collection<ConfiguredTarget> allTestTargets,
                                    boolean isExclusive) {
    for (ConfiguredTarget target : allTestTargets) {
      if (target.getTarget() instanceof Rule) {
        boolean exclusive =
            isExclusive || TargetUtils.isExclusiveTestRule((Rule) target.getTarget());
        Collection<ConfiguredTarget> testCollection = exclusive
            ? targetsToTestExclusive
            : targetsToTest;
        testCollection.add(target);
      }
    }
  }

  @VisibleForTesting
  List<TargetAndConfiguration> nodesForTargets(BuildConfigurationCollection configurations,
      Collection<Target> targets) {
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
   * Returns an existing ConfiguredTarget for the specified target and
   * configuration, or null if none exists.  No validity check is done.
   */
  @ThreadSafe
  public ConfiguredTarget getExistingConfiguredTarget(Target target, BuildConfiguration config) {
    return getExistingConfiguredTarget(target.getLabel(), config);
  }

  /**
   * Returns an existing ConfiguredTarget for the specified node, or null if none exists. No
   * validity check is done.
   */
  @ThreadSafe
  private ConfiguredTarget getExistingConfiguredTarget(
      Label label, BuildConfiguration configuration) {
    return Iterables.getFirst(
        skyframeExecutor.getConfiguredTargets(
            configuration, ImmutableList.of(new Dependency(label, configuration)), true),
        null);
  }

  @VisibleForTesting
  ListMultimap<Attribute, ConfiguredTarget> getPrerequisiteMapForTesting(ConfiguredTarget target)
      throws InterruptedException {
    DependencyResolver resolver = new DependencyResolver() {
      @Override
      protected void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label) {
        throw new RuntimeException("bad visibility on " + label + " during testing unexpected");
      }

      @Override
      protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
        throw new RuntimeException("bad package group on " + label + " during testing unexpected");
      }

      @Override
      protected Target getTarget(Label label) throws NoSuchThingException {
        return packageManager.getLoadedTarget(label);
      }
    };
    TargetAndConfiguration ctNode = new TargetAndConfiguration(target);
    ListMultimap<Attribute, Dependency> depNodeNames;
    try {
      depNodeNames = resolver.dependentNodeMap(ctNode, configurations.getHostConfiguration(),
          /*aspect=*/null, AspectParameters.EMPTY, getConfigurableAttributeKeys(ctNode));
    } catch (EvalException e) {
      throw new IllegalStateException(e);
    }

    ImmutableMap<Dependency, ConfiguredTarget> cts = skyframeExecutor.getConfiguredTargetMap(
        ctNode.getConfiguration(), ImmutableSet.copyOf(depNodeNames.values()), false);

    ImmutableListMultimap.Builder<Attribute, ConfiguredTarget> builder =
        ImmutableListMultimap.builder();
    for (Map.Entry<Attribute, Dependency> entry : depNodeNames.entries()) {
      builder.put(entry.getKey(), cts.get(entry.getValue()));
    }
    return builder.build();
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
  public void setArtifactRoots(ImmutableMap<PackageIdentifier, Path> packageRoots,
      BuildConfigurationCollection configurations) {
    Map<Path, Root> rootMap = new HashMap<>();
    Map<PackageIdentifier, Root> realPackageRoots = new HashMap<>();
    for (Map.Entry<PackageIdentifier, Path> entry : packageRoots.entrySet()) {
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
  public RuleContext getRuleContextForTesting(
      ConfiguredTarget target, StoredEventHandler eventHandler) throws InterruptedException {
    BuildConfiguration config = target.getConfiguration();
    CachingAnalysisEnvironment analysisEnvironment =
        new CachingAnalysisEnvironment(artifactFactory,
            new ConfiguredTargetKey(target.getLabel(), config),
            /*isSystemEnv=*/false, config.extendedSanityChecks(), eventHandler,
            /*skyframeEnv=*/null, config.isActionsEnabled(), binTools);
    return new RuleContext.Builder(analysisEnvironment,
        (Rule) target.getTarget(), config, configurations.getHostConfiguration(),
        ruleClassProvider.getPrerequisiteValidator())
            .setVisibility(NestedSetBuilder.<PackageSpecification>create(
                Order.STABLE_ORDER, PackageSpecification.EVERYTHING))
            .setPrerequisites(getPrerequisiteMapForTesting(target))
            .setConfigConditions(ImmutableSet.<ConfigMatchingProvider>of())
            .build();
  }

  /**
   * Creates and returns a rule context that is equivalent to the one that was used to create the
   * given configured target.
   */
  @VisibleForTesting
  public RuleContext getRuleContextForTesting(ConfiguredTarget target, AnalysisEnvironment env)
      throws InterruptedException {
    BuildConfiguration targetConfig = target.getConfiguration();
    return new RuleContext.Builder(
        env, (Rule) target.getTarget(), targetConfig, configurations.getHostConfiguration(),
        ruleClassProvider.getPrerequisiteValidator())
            .setVisibility(NestedSetBuilder.<PackageSpecification>create(
                Order.STABLE_ORDER, PackageSpecification.EVERYTHING))
            .setPrerequisites(getPrerequisiteMapForTesting(target))
            .setConfigConditions(ImmutableSet.<ConfigMatchingProvider>of())
            .build();
  }

  /**
   * For a configured target dependentTarget, returns the desired configured target
   * that is depended upon. Useful for obtaining the a target with aspects
   * required by the dependent.
   */
  @VisibleForTesting
  public ConfiguredTarget getPrerequisiteConfiguredTargetForTesting(
      ConfiguredTarget dependentTarget, ConfiguredTarget desiredTarget)
      throws InterruptedException {
    Collection<ConfiguredTarget> configuredTargets =
        getPrerequisiteMapForTesting(dependentTarget).values();
    for (ConfiguredTarget ct : configuredTargets) {
      if (ct.getLabel().equals(desiredTarget.getLabel())) {
        return ct;
      }
    }
    return null;
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
}
