// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.Iterables.concat;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.DependencyResolver.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ComposingSplitTransition;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.DynamicTransitionMapper;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics;
import com.google.devtools.build.lib.analysis.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.analysis.test.CoverageReportActionFactory.CoverageReportActionsWrapper;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.pkgcache.LoadingResult;
import com.google.devtools.build.lib.pkgcache.PackageManager.PackageManagerStatistics;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectValueKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.CoverageReportValue;
import com.google.devtools.build.lib.skyframe.SkyframeAnalysisResult;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.ToolchainContextException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkImport;
import com.google.devtools.build.lib.syntax.SkylarkImports;
import com.google.devtools.build.lib.syntax.SkylarkImports.SkylarkImportSyntaxException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
   * Options that affect the <i>mechanism</i> of analysis. These are distinct from {@link
   * com.google.devtools.build.lib.analysis.config.BuildOptions}, which affect the <i>value</i> of a
   * BuildConfiguration.
   */
  public static class Options extends OptionsBase {
    @Option(
      name = "loading_phase_threads",
      defaultValue = "-1",
      category = "what",
      converter = LoadingPhaseThreadCountConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Number of parallel threads to use for the loading/analysis phase."
    )
    public int loadingPhaseThreads;

    @Option(
      name = "keep_going",
      abbrev = 'k',
      defaultValue = "false",
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Continue as much as possible after an error.  While the target that failed, and those "
              + "that depend on it, cannot be analyzed (or built), the other prerequisites of "
              + "these targets can be analyzed (or built) all the same."
    )
    public boolean keepGoing;

    @Option(
      name = "analysis_warnings_as_errors",
      deprecationWarning =
          "analysis_warnings_as_errors is now a no-op and will be removed in"
              + " an upcoming Blaze release",
      defaultValue = "false",
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Treat visible analysis warnings as errors."
    )
    public boolean analysisWarningsAsErrors;

    @Option(
      name = "discard_analysis_cache",
      defaultValue = "false",
      category = "strategy",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Discard the analysis cache immediately after the analysis phase completes."
              + " Reduces memory usage by ~10%, but makes further incremental builds slower."
    )
    public boolean discardAnalysisCache;

    @Option(
      name = "experimental_extra_action_filter",
      defaultValue = "",
      category = "experimental",
      converter = RegexFilter.RegexFilterConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Filters set of targets to schedule extra_actions for."
    )
    public RegexFilter extraActionFilter;

    @Option(
      name = "experimental_extra_action_top_level_only",
      defaultValue = "false",
      category = "experimental",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Only schedules extra_actions for top level targets."
    )
    public boolean extraActionTopLevelOnly;

    @Option(
      name = "version_window_for_dirty_node_gc",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Nodes that have been dirty for more than this many versions will be deleted"
              + " from the graph upon the next update. Values must be non-negative long integers,"
              + " or -1 indicating the maximum possible window."
    )
    public long versionWindowForDirtyNodeGc;

    @Deprecated
    @Option(
      name = "experimental_interleave_loading_and_analysis",
      defaultValue = "true",
      category = "experimental",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "No-op."
    )
    public boolean interleaveLoadingAndAnalysis;
  }

  private static final Logger logger = Logger.getLogger(BuildView.class.getName());

  private final BlazeDirectories directories;

  private final SkyframeExecutor skyframeExecutor;
  private final SkyframeBuildView skyframeBuildView;

  private final ConfiguredRuleClassProvider ruleClassProvider;

  /**
   * A factory class to create the coverage report action. May be null.
   */
  @Nullable private final CoverageReportActionFactory coverageReportActionFactory;

  @VisibleForTesting
  public Set<SkyKey> getSkyframeEvaluatedTargetKeysForTesting() {
    return skyframeBuildView.getEvaluatedTargetKeys();
  }

  /** The number of targets freshly evaluated in the last analysis run. */
  public int getTargetsVisited() {
    return skyframeBuildView.getEvaluatedTargetKeys().size();
  }

  public PackageManagerStatistics getAndClearPkgManagerStatistics() {
    return skyframeExecutor.getPackageManager().getAndClearStatistics();
  }

  public BuildView(BlazeDirectories directories,
      ConfiguredRuleClassProvider ruleClassProvider,
      SkyframeExecutor skyframeExecutor,
      CoverageReportActionFactory coverageReportActionFactory) {
    this.directories = directories;
    this.coverageReportActionFactory = coverageReportActionFactory;
    this.ruleClassProvider = ruleClassProvider;
    this.skyframeExecutor = Preconditions.checkNotNull(skyframeExecutor);
    this.skyframeBuildView = skyframeExecutor.getSkyframeBuildView();
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
    skyframeBuildView.setConfigurations(configurations);
  }

  public ArtifactFactory getArtifactFactory() {
    return skyframeBuildView.getArtifactFactory();
  }

  @VisibleForTesting
  WorkspaceStatusAction getLastWorkspaceBuildInfoActionForTesting() {
    return skyframeExecutor.getLastWorkspaceStatusActionForTesting();
  }

  @Override
  public int hashCode() {
    throw new UnsupportedOperationException();  // avoid nondeterminism
  }

  /**
   * Return value for {@link BuildView#update} and {@code BuildTool.prepareToBuild}.
   */
  public static final class AnalysisResult {
    private final ImmutableSet<ConfiguredTarget> targetsToBuild;
    @Nullable private final ImmutableList<ConfiguredTarget> targetsToTest;
    private final ImmutableSet<ConfiguredTarget> targetsToSkip;
    @Nullable private final String error;
    private final ActionGraph actionGraph;
    private final ImmutableSet<Artifact> artifactsToBuild;
    private final ImmutableSet<ConfiguredTarget> parallelTests;
    private final ImmutableSet<ConfiguredTarget> exclusiveTests;
    @Nullable private final TopLevelArtifactContext topLevelContext;
    private final ImmutableList<AspectValue> aspects;
    private final ImmutableMap<PackageIdentifier, Path> packageRoots;
    private final String workspaceName;

    private AnalysisResult(
        Collection<ConfiguredTarget> targetsToBuild,
        Collection<AspectValue> aspects,
        Collection<ConfiguredTarget> targetsToTest,
        Collection<ConfiguredTarget> targetsToSkip,
        @Nullable String error,
        ActionGraph actionGraph,
        Collection<Artifact> artifactsToBuild,
        Collection<ConfiguredTarget> parallelTests,
        Collection<ConfiguredTarget> exclusiveTests,
        TopLevelArtifactContext topLevelContext,
        ImmutableMap<PackageIdentifier, Path> packageRoots,
        String workspaceName) {
      this.targetsToBuild = ImmutableSet.copyOf(targetsToBuild);
      this.aspects = ImmutableList.copyOf(aspects);
      this.targetsToTest = targetsToTest == null ? null : ImmutableList.copyOf(targetsToTest);
      this.targetsToSkip = ImmutableSet.copyOf(targetsToSkip);
      this.error = error;
      this.actionGraph = actionGraph;
      this.artifactsToBuild = ImmutableSet.copyOf(artifactsToBuild);
      this.parallelTests = ImmutableSet.copyOf(parallelTests);
      this.exclusiveTests = ImmutableSet.copyOf(exclusiveTests);
      this.topLevelContext = topLevelContext;
      this.packageRoots = packageRoots;
      this.workspaceName = workspaceName;
    }

    /**
     * Returns configured targets to build.
     */
    public ImmutableSet<ConfiguredTarget> getTargetsToBuild() {
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

    /**
     * Returns the configured targets that should not be executed because they're not
     * platform-compatible with the current build.
     *
     * <p>For example: tests that aren't intended for the designated CPU.
     */
    public ImmutableSet<ConfiguredTarget> getTargetsToSkip() {
      return targetsToSkip;
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

    public String getWorkspaceName() {
      return workspaceName;
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

  @ThreadCompatible
  public AnalysisResult update(
      LoadingResult loadingResult,
      BuildConfigurationCollection configurations,
      List<String> aspects,
      Options viewOptions,
      TopLevelArtifactContext topLevelOptions,
      ExtendedEventHandler eventHandler,
      EventBus eventBus)
      throws ViewCreationFailedException, InterruptedException {
    logger.info("Starting analysis");
    pollInterruptedStatus();

    skyframeBuildView.resetEvaluatedConfiguredTargetKeysSet();

    Collection<Target> targets = loadingResult.getTargets();
    eventBus.post(new AnalysisPhaseStartedEvent(targets));

    skyframeBuildView.setConfigurations(configurations);

    // Determine the configurations.
    List<TargetAndConfiguration> topLevelTargetsWithConfigs =
        nodesForTopLevelTargets(configurations, targets, eventHandler);

    // Report the generated association of targets to configurations
    Multimap<Label, BuildConfiguration> byLabel =
        ArrayListMultimap.<Label, BuildConfiguration>create();
    for (TargetAndConfiguration pair : topLevelTargetsWithConfigs) {
      byLabel.put(pair.getLabel(), pair.getConfiguration());
    }
    for (Target target : targets) {
      eventBus.post(new TargetConfiguredEvent(target, byLabel.get(target.getLabel())));
    }

    List<ConfiguredTargetKey> topLevelCtKeys = Lists.transform(topLevelTargetsWithConfigs,
        new Function<TargetAndConfiguration, ConfiguredTargetKey>() {
          @Override
          public ConfiguredTargetKey apply(TargetAndConfiguration node) {
            return new ConfiguredTargetKey(node.getLabel(), node.getConfiguration());
          }
        });

    Multimap<Pair<Label, String>, BuildConfiguration> aspectConfigurations =
        ArrayListMultimap.create();

    List<AspectValueKey> aspectKeys = new ArrayList<>();
    for (String aspect : aspects) {
      // Syntax: label%aspect
      int delimiterPosition = aspect.indexOf('%');
      if (delimiterPosition >= 0) {
        // TODO(jfield): For consistency with Skylark loads, the aspect should be specified
        // as an absolute path. Also, we probably need to do at least basic validation of
        // path well-formedness here.
        String bzlFileLoadLikeString = aspect.substring(0, delimiterPosition);
        if (!bzlFileLoadLikeString.startsWith("//") && !bzlFileLoadLikeString.startsWith("@")) {
          // "Legacy" behavior of '--aspects' parameter.
          bzlFileLoadLikeString = PathFragment.create("/" + bzlFileLoadLikeString).toString();
          if (bzlFileLoadLikeString.endsWith(".bzl")) {
            bzlFileLoadLikeString = bzlFileLoadLikeString.substring(0,
                bzlFileLoadLikeString.length() - ".bzl".length());
          }
        }
        SkylarkImport skylarkImport;
        try {
          skylarkImport = SkylarkImports.create(bzlFileLoadLikeString);
        } catch (SkylarkImportSyntaxException e) {
          throw new ViewCreationFailedException(
              String.format("Invalid aspect '%s': %s", aspect, e.getMessage()), e);
        }

        String skylarkFunctionName = aspect.substring(delimiterPosition + 1);
        for (TargetAndConfiguration targetSpec : topLevelTargetsWithConfigs) {
          aspectConfigurations.put(
              Pair.of(targetSpec.getLabel(), aspect), targetSpec.getConfiguration());
          aspectKeys.add(
              AspectValue.createSkylarkAspectKey(
                  targetSpec.getLabel(),
                  // For invoking top-level aspects, use the top-level configuration for both the
                  // aspect and the base target while the top-level configuration is untrimmed.
                  targetSpec.getConfiguration(),
                  targetSpec.getConfiguration(),
                  skylarkImport,
                  skylarkFunctionName));
        }
      } else {
        final NativeAspectClass aspectFactoryClass =
            ruleClassProvider.getNativeAspectClassMap().get(aspect);

        if (aspectFactoryClass != null) {
          for (TargetAndConfiguration targetSpec : topLevelTargetsWithConfigs) {
            // For invoking top-level aspects, use the top-level configuration for both the
            // aspect and the base target while the top-level configuration is untrimmed.
            BuildConfiguration configuration = targetSpec.getConfiguration();
            aspectConfigurations.put(Pair.of(targetSpec.getLabel(), aspect), configuration);
            aspectKeys.add(
                AspectValue.createAspectKey(
                    targetSpec.getLabel(),
                    configuration,
                    new AspectDescriptor(aspectFactoryClass, AspectParameters.EMPTY),
                    configuration
                ));
          }
        } else {
          throw new ViewCreationFailedException("Aspect '" + aspect + "' is unknown");
        }
      }
    }

    for (Pair<Label, String> target : aspectConfigurations.keys()) {
      eventBus.post(
          new AspectConfiguredEvent(
              target.getFirst(), target.getSecond(), aspectConfigurations.get(target)));
    }

    skyframeExecutor.injectWorkspaceStatusData(loadingResult.getWorkspaceName());
    SkyframeAnalysisResult skyframeAnalysisResult;
    try {
      skyframeAnalysisResult =
          skyframeBuildView.configureTargets(
              eventHandler,
              topLevelCtKeys,
              aspectKeys,
              eventBus,
              viewOptions.keepGoing,
              viewOptions.loadingPhaseThreads);
      setArtifactRoots(skyframeAnalysisResult.getPackageRoots());
    } finally {
      skyframeBuildView.clearInvalidatedConfiguredTargets();
    }

    int numTargetsToAnalyze = topLevelTargetsWithConfigs.size();
    int numSuccessful = skyframeAnalysisResult.getConfiguredTargets().size();
    if (0 < numSuccessful && numSuccessful < numTargetsToAnalyze) {
      String msg = String.format("Analysis succeeded for only %d of %d top-level targets",
                                    numSuccessful, numTargetsToAnalyze);
      eventHandler.handle(Event.info(msg));
      logger.info(msg);
    }

    Set<ConfiguredTarget> targetsToSkip =
        TopLevelConstraintSemantics.checkTargetEnvironmentRestrictions(
            skyframeAnalysisResult.getConfiguredTargets(),
            skyframeExecutor.getPackageManager(),
            eventHandler);

    AnalysisResult result =
        createResult(
            eventHandler,
            loadingResult,
            topLevelOptions,
            viewOptions,
            skyframeAnalysisResult,
            targetsToSkip);
    logger.info("Finished analysis");
    return result;
  }

  private AnalysisResult createResult(
      ExtendedEventHandler eventHandler,
      LoadingResult loadingResult,
      TopLevelArtifactContext topLevelOptions,
      BuildView.Options viewOptions,
      SkyframeAnalysisResult skyframeAnalysisResult,
      Set<ConfiguredTarget> targetsToSkip)
      throws InterruptedException {
    Collection<Target> testsToRun = loadingResult.getTestsToRun();
    Set<ConfiguredTarget> configuredTargets =
        Sets.newLinkedHashSet(skyframeAnalysisResult.getConfiguredTargets());
    Collection<AspectValue> aspects = skyframeAnalysisResult.getAspects();

    Set<ConfiguredTarget> allTargetsToTest = null;
    if (testsToRun != null) {
      // Determine the subset of configured targets that are meant to be run as tests.
      allTargetsToTest = Sets.newLinkedHashSet(
          filterTestsByTargets(configuredTargets, Sets.newHashSet(testsToRun)));
    }

    Set<Artifact> artifactsToBuild = new HashSet<>();
    Set<ConfiguredTarget> parallelTests = new HashSet<>();
    Set<ConfiguredTarget> exclusiveTests = new HashSet<>();

    // build-info and build-changelist.
    Collection<Artifact> buildInfoArtifacts =
        skyframeExecutor.getWorkspaceStatusArtifacts(eventHandler);
    Preconditions.checkState(buildInfoArtifacts.size() == 2, buildInfoArtifacts);
    artifactsToBuild.addAll(buildInfoArtifacts);

    // Extra actions
    addExtraActionsIfRequested(viewOptions, configuredTargets, aspects, artifactsToBuild);

    // Coverage
    NestedSet<Artifact> baselineCoverageArtifacts = getBaselineCoverageArtifacts(configuredTargets);
    Iterables.addAll(artifactsToBuild, baselineCoverageArtifacts);
    if (coverageReportActionFactory != null) {
      CoverageReportActionsWrapper actionsWrapper;
      actionsWrapper = coverageReportActionFactory.createCoverageReportActionsWrapper(
          eventHandler,
          directories,
          allTargetsToTest,
          baselineCoverageArtifacts,
          getArtifactFactory(),
          CoverageReportValue.ARTIFACT_OWNER);
      if (actionsWrapper != null) {
        ImmutableList<ActionAnalysisMetadata> actions = actionsWrapper.getActions();
        skyframeExecutor.injectCoverageReportData(actions);
        artifactsToBuild.addAll(actionsWrapper.getCoverageOutputs());
      }
    }

    // Tests. This must come last, so that the exclusive tests are scheduled after everything else.
    scheduleTestsIfRequested(parallelTests, exclusiveTests, topLevelOptions, allTargetsToTest);

    String error = createErrorMessage(loadingResult, skyframeAnalysisResult);

    final WalkableGraph graph = skyframeAnalysisResult.getWalkableGraph();
    final ActionGraph actionGraph =
        new ActionGraph() {
          @Nullable
          @Override
          public ActionAnalysisMetadata getGeneratingAction(Artifact artifact) {
            ArtifactOwner artifactOwner = artifact.getArtifactOwner();
            if (artifactOwner instanceof ActionLookupValue.ActionLookupKey) {
              SkyKey key = ActionLookupValue.key((ActionLookupValue.ActionLookupKey) artifactOwner);
              ActionLookupValue val;
              try {
                val = (ActionLookupValue) graph.getValue(key);
              } catch (InterruptedException e) {
                throw new IllegalStateException(
                    "Interruption not expected from this graph: " + key, e);
              }
              return val == null ? null : val.getGeneratingActionDangerousReadJavadoc(artifact);
            }
            return null;
          }
        };
    return new AnalysisResult(
        configuredTargets,
        aspects,
        allTargetsToTest,
        targetsToSkip,
        error,
        actionGraph,
        artifactsToBuild,
        parallelTests,
        exclusiveTests,
        topLevelOptions,
        skyframeAnalysisResult.getPackageRoots(),
        loadingResult.getWorkspaceName());
  }

  @Nullable
  public static String createErrorMessage(
      LoadingResult loadingResult, @Nullable SkyframeAnalysisResult skyframeAnalysisResult) {
    return loadingResult.hasTargetPatternError()
        ? "command succeeded, but there were errors parsing the target pattern"
        : loadingResult.hasLoadingError()
                || (skyframeAnalysisResult != null && skyframeAnalysisResult.hasLoadingError())
            ? "command succeeded, but there were loading phase errors"
            : (skyframeAnalysisResult != null && skyframeAnalysisResult.hasAnalysisError())
                ? "command succeeded, but not all targets were analyzed"
                : null;
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

  private void addExtraActionsIfRequested(Options viewOptions,
      Collection<ConfiguredTarget> configuredTargets,
      Collection<AspectValue> aspects,
      Set<Artifact> artifactsToBuild) {
    Iterable<Artifact> extraActionArtifacts =
        concat(
            addExtraActionsFromTargets(viewOptions, configuredTargets),
            addExtraActionsFromAspects(viewOptions, aspects));

    RegexFilter filter = viewOptions.extraActionFilter;
    for (Artifact artifact : extraActionArtifacts) {
      boolean filterMatches =
          filter == null || filter.isIncluded(artifact.getOwnerLabel().toString());
      if (filterMatches) {
        artifactsToBuild.add(artifact);
      }
    }
  }

  private NestedSet<Artifact> addExtraActionsFromTargets(
      BuildView.Options viewOptions, Collection<ConfiguredTarget> configuredTargets) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (ConfiguredTarget target : configuredTargets) {
      ExtraActionArtifactsProvider provider =
          target.getProvider(ExtraActionArtifactsProvider.class);
      if (provider != null) {
        if (viewOptions.extraActionTopLevelOnly) {
          // Collect all aspect-classes that topLevel might inject.
          Set<AspectClass> aspectClasses = new HashSet<>();
          for (Attribute attr : target.getTarget().getAssociatedRule().getAttributes()) {
            aspectClasses.addAll(attr.getAspectClasses());
          }

          builder.addTransitive(provider.getExtraActionArtifacts());
          if (!aspectClasses.isEmpty()) {
            builder.addAll(filterTransitiveExtraActions(provider, aspectClasses));
          }
        } else {
          builder.addTransitive(provider.getTransitiveExtraActionArtifacts());
        }
      }
    }
    return builder.build();
  }

  /**
   * Returns a list of actions from 'provider' that were registered by an aspect from
   * 'aspectClasses'. All actions in 'provider' are considered - both direct and transitive.
   */
  private ImmutableList<Artifact> filterTransitiveExtraActions(
      ExtraActionArtifactsProvider provider, Set<AspectClass> aspectClasses) {
    ImmutableList.Builder<Artifact> artifacts = ImmutableList.builder();
    // Add to 'artifacts' all extra-actions which were registered by aspects which 'topLevel'
    // might have injected.
    for (Artifact artifact : provider.getTransitiveExtraActionArtifacts()) {
      ArtifactOwner owner = artifact.getArtifactOwner();
      if (owner instanceof AspectKey) {
        if (aspectClasses.contains(((AspectKey) owner).getAspectClass())) {
          artifacts.add(artifact);
        }
      }
    }
    return artifacts.build();
  }

  private NestedSet<Artifact> addExtraActionsFromAspects(
      BuildView.Options viewOptions, Collection<AspectValue> aspects) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    for (AspectValue aspect : aspects) {
      ExtraActionArtifactsProvider provider =
          aspect.getConfiguredAspect().getProvider(ExtraActionArtifactsProvider.class);
      if (provider != null) {
        if (viewOptions.extraActionTopLevelOnly) {
          builder.addTransitive(provider.getExtraActionArtifacts());
        } else {
          builder.addTransitive(provider.getTransitiveExtraActionArtifacts());
        }
      }
    }
    return builder.build();
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

  /**
   * Given a set of top-level targets and a configuration collection, returns the appropriate
   * <Target, Configuration> pair for each target.
   *
   * <p>Preserves the original input ordering.
   */
  private List<TargetAndConfiguration> nodesForTopLevelTargets(
      BuildConfigurationCollection configurations,
      Collection<Target> targets,
      ExtendedEventHandler eventHandler)
      throws InterruptedException {
    // We use a hash set here to remove duplicate nodes; this can happen for input files and package
    // groups.
    LinkedHashSet<TargetAndConfiguration> nodes = new LinkedHashSet<>(targets.size());
    for (BuildConfiguration config : configurations.getTargetConfigurations()) {
      for (Target target : targets) {
        nodes.add(new TargetAndConfiguration(target, target.isConfigurable() ? config : null));
      }
    }
    return ImmutableList.copyOf(getDynamicConfigurations(nodes, eventHandler));
  }

  /**
   * If {@link BuildConfiguration.Options#trimConfigurations()} is true, transforms a collection of
   * <Target, Configuration> pairs by trimming each target's configuration to only the fragments the
   * target and its transitive dependencies need.
   *
   * <p>Else returns configurations that unconditionally include all fragments.
   *
   * <p>Preserves the original input order (but merges duplicate nodes that might occur due to
   * top-level configuration transitions) . Uses original (untrimmed) configurations for targets
   * that can't be evaluated (e.g. due to loading phase errors).
   *
   * <p>This is suitable for feeding {@link ConfiguredTargetValue} keys: as general principle {@link
   * ConfiguredTarget}s should have exactly as much information in their configurations as they need
   * to evaluate and no more (e.g. there's no need for Android settings in a C++ configured target).
   */
  // TODO(bazel-team): error out early for targets that fail - untrimmed configurations should
  // never make it through analysis (and especially not seed ConfiguredTargetValues)
  private LinkedHashSet<TargetAndConfiguration> getDynamicConfigurations(
      Iterable<TargetAndConfiguration> inputs, ExtendedEventHandler eventHandler)
      throws InterruptedException {
    Map<Label, Target> labelsToTargets = new LinkedHashMap<>();
    // We'll get the configs from SkyframeExecutor#getConfigurations, which gets configurations
    // for deps including transitions. So to satisfy its API we repackage each target as a
    // Dependency with a NONE transition.
    Multimap<BuildConfiguration, Dependency> asDeps =
        ArrayListMultimap.<BuildConfiguration, Dependency>create();

    for (TargetAndConfiguration targetAndConfig : inputs) {
      labelsToTargets.put(targetAndConfig.getLabel(), targetAndConfig.getTarget());
      if (targetAndConfig.getConfiguration() != null) {
        asDeps.put(targetAndConfig.getConfiguration(),
            Dependency.withTransitionAndAspects(
                targetAndConfig.getLabel(),
                getTopLevelTransition(targetAndConfig,
                    ruleClassProvider.getDynamicTransitionMapper()),
                // TODO(bazel-team): support top-level aspects
                AspectCollection.EMPTY));
      }
    }

    // Maps <target, originalConfig> pairs to <target, dynamicConfig> pairs for targets that
    // could be successfully Skyframe-evaluated.
    Map<TargetAndConfiguration, TargetAndConfiguration> successfullyEvaluatedTargets =
        new LinkedHashMap<>();
    if (!asDeps.isEmpty()) {
      for (BuildConfiguration fromConfig : asDeps.keySet()) {
        Multimap<Dependency, BuildConfiguration> trimmedTargets =
            skyframeExecutor.getConfigurations(eventHandler, fromConfig.getOptions(),
                asDeps.get(fromConfig));
        for (Map.Entry<Dependency, BuildConfiguration> trimmedTarget : trimmedTargets.entries()) {
          Target target = labelsToTargets.get(trimmedTarget.getKey().getLabel());
          successfullyEvaluatedTargets.put(
              new TargetAndConfiguration(target, fromConfig),
              new TargetAndConfiguration(target, trimmedTarget.getValue()));
        }
      }
    }

    LinkedHashSet<TargetAndConfiguration> result = new LinkedHashSet<>();
    for (TargetAndConfiguration originalInput : inputs) {
      if (successfullyEvaluatedTargets.containsKey(originalInput)) {
        // The configuration was successfully trimmed.
        result.add(successfullyEvaluatedTargets.get(originalInput));
      } else {
        // Either the configuration couldn't be determined (e.g. loading phase error) or it's null.
        result.add(originalInput);
      }
    }
    return result;
  }

  /**
   * Returns the transition to apply to the top-level configuration before applying it to this
   * target. This enables support for rule-triggered top-level configuration hooks.
   */
  private static Attribute.Transition getTopLevelTransition(TargetAndConfiguration targetAndConfig,
      DynamicTransitionMapper dynamicTransitionMapper) {
    Target target = targetAndConfig.getTarget();
    BuildConfiguration fromConfig = targetAndConfig.getConfiguration();

    // Top-level transitions (chosen by configuration fragments):
    Transition topLevelTransition = fromConfig.topLevelConfigurationHook(target);
    if (topLevelTransition == null) {
      topLevelTransition = ConfigurationTransition.NONE;
    }

    // Rule class transitions (chosen by rule class definitions):
    if (target.getAssociatedRule() == null) {
      return topLevelTransition;
    }
    Rule associatedRule = target.getAssociatedRule();
    RuleTransitionFactory transitionFactory =
        associatedRule.getRuleClassObject().getTransitionFactory();
    if (transitionFactory == null) {
      return topLevelTransition;
    }
    // dynamicTransitionMapper is only needed because of Attribute.ConfigurationTransition.DATA:
    // this is C++-specific but non-C++ rules declare it. So they can't directly provide the
    // C++-specific patch transition that implements it.
    PatchTransition ruleClassTransition = (PatchTransition)
        dynamicTransitionMapper.map(transitionFactory.buildTransitionFor(associatedRule));
    if (ruleClassTransition == null) {
      return topLevelTransition;
    } else if (topLevelTransition == ConfigurationTransition.NONE) {
      return ruleClassTransition;
    } else {
      return new ComposingSplitTransition(topLevelTransition, ruleClassTransition);
    }
  }


  /**
   * Gets a dynamic configuration for the given target.
   *
   * <p>If {@link BuildConfiguration.Options#trimConfigurations()} is true, the configuration only
   * includes the fragments needed by the fragment and its transitive closure. Else unconditionally
   * includes all fragments.
   */
  @VisibleForTesting
  public BuildConfiguration getDynamicConfigurationForTesting(
      Target target, BuildConfiguration config, ExtendedEventHandler eventHandler)
      throws InterruptedException {
    return Iterables.getOnlyElement(getDynamicConfigurations(
        ImmutableList.<TargetAndConfiguration>of(new TargetAndConfiguration(target, config)),
        eventHandler)).getConfiguration();
  }

  /**
   * Sets the possible artifact roots in the artifact factory. This allows the factory to resolve
   * paths with unknown roots to artifacts.
   */
  @VisibleForTesting // for BuildViewTestCase
  public void setArtifactRoots(ImmutableMap<PackageIdentifier, Path> packageRoots) {
    Map<Path, Root> rootMap = new HashMap<>();
    Map<PackageIdentifier, Root> realPackageRoots = new HashMap<>();
    for (Map.Entry<PackageIdentifier, Path> entry : packageRoots.entrySet()) {
      Root root = rootMap.get(entry.getValue());
      if (root == null) {
        root = Root.asSourceRoot(entry.getValue(), entry.getKey().getRepository().isMain());
        rootMap.put(entry.getValue(), root);
      }
      realPackageRoots.put(entry.getKey(), root);
    }
    // Source Artifact roots:
    getArtifactFactory().setPackageRoots(realPackageRoots);
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

  // For testing
  @VisibleForTesting
  public Iterable<ConfiguredTarget> getDirectPrerequisitesForTesting(
      ExtendedEventHandler eventHandler, ConfiguredTarget ct,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException,
      InterruptedException, InconsistentAspectOrderException {
    return skyframeExecutor.getConfiguredTargets(
        eventHandler, ct.getConfiguration(),
        ImmutableSet.copyOf(
            getDirectPrerequisiteDependenciesForTesting(eventHandler, ct, configurations).values()),
        false);
  }

  @VisibleForTesting
  public OrderedSetMultimap<Attribute, Dependency> getDirectPrerequisiteDependenciesForTesting(
      final ExtendedEventHandler eventHandler,
      final ConfiguredTarget ct,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
             InconsistentAspectOrderException {
    if (!(ct.getTarget() instanceof Rule)) {
      return OrderedSetMultimap.create();
    }

    class SilentDependencyResolver extends DependencyResolver {
      private SilentDependencyResolver() {
        super(ruleClassProvider.getDynamicTransitionMapper());
      }

      @Override
      protected void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label) {
        throw new RuntimeException("bad visibility on " + label + " during testing unexpected");
      }

      @Override
      protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
        throw new RuntimeException("bad package group on " + label + " during testing unexpected");
      }

      @Override
      protected void missingEdgeHook(Target from, Label to, NoSuchThingException e) {
        throw new RuntimeException(
            "missing dependency from " + from.getLabel() + " to " + to + ": " + e.getMessage(),
            e);
      }

      @Override
      protected Target getTarget(Target from, Label label, NestedSetBuilder<Label> rootCauses)
          throws InterruptedException {
        try {
          return skyframeExecutor.getPackageManager().getTarget(eventHandler, label);
        } catch (NoSuchThingException e) {
          throw new IllegalStateException(e);
        }
      }

      @Override
      protected List<BuildConfiguration> getConfigurations(
          Set<Class<? extends BuildConfiguration.Fragment>> fragments,
          Iterable<BuildOptions> buildOptions) {
        Preconditions.checkArgument(ct.getConfiguration().fragmentClasses().equals(fragments));
        Dependency asDep = Dependency.withTransitionAndAspects(ct.getLabel(),
            Attribute.ConfigurationTransition.NONE, AspectCollection.EMPTY);
        ImmutableList.Builder<BuildConfiguration> builder = ImmutableList.builder();
        for (BuildOptions options : buildOptions) {
          builder.add(Iterables.getOnlyElement(
              skyframeExecutor
                  .getConfigurations(eventHandler, options, ImmutableList.<Dependency>of(asDep))
                  .values()
          ));
        }
        return builder.build();
      }
    }

    DependencyResolver dependencyResolver = new SilentDependencyResolver();
    TargetAndConfiguration ctgNode =
        new TargetAndConfiguration(ct.getTarget(), ct.getConfiguration());
    return dependencyResolver.dependentNodeMap(
        ctgNode, configurations.getHostConfiguration(), /*aspect=*/ null,
        getConfigurableAttributeKeysForTesting(eventHandler, ctgNode));
  }

  /**
   * Returns ConfigMatchingProvider instances corresponding to the configurable attribute keys
   * present in this rule's attributes.
   */
  private ImmutableMap<Label, ConfigMatchingProvider> getConfigurableAttributeKeysForTesting(
      ExtendedEventHandler eventHandler, TargetAndConfiguration ctg) {
    if (!(ctg.getTarget() instanceof Rule)) {
      return ImmutableMap.of();
    }
    Rule rule = (Rule) ctg.getTarget();
    Map<Label, ConfigMatchingProvider> keys = new LinkedHashMap<>();
    RawAttributeMapper mapper = RawAttributeMapper.of(rule);
    for (Attribute attribute : rule.getAttributes()) {
      for (Label label : mapper.getConfigurabilityKeys(attribute.getName(), attribute.getType())) {
        if (BuildType.Selector.isReservedLabel(label)) {
          continue;
        }
        ConfiguredTarget ct = getConfiguredTargetForTesting(
            eventHandler, label, ctg.getConfiguration());
        keys.put(label, Preconditions.checkNotNull(ct.getProvider(ConfigMatchingProvider.class)));
      }
    }
    return ImmutableMap.copyOf(keys);
  }

  private OrderedSetMultimap<Attribute, ConfiguredTarget> getPrerequisiteMapForTesting(
      final ExtendedEventHandler eventHandler,
      ConfiguredTarget target,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException,
             InterruptedException, InconsistentAspectOrderException {
    OrderedSetMultimap<Attribute, Dependency> depNodeNames =
        getDirectPrerequisiteDependenciesForTesting(eventHandler, target, configurations);

    ImmutableMultimap<Dependency, ConfiguredTarget> cts = skyframeExecutor.getConfiguredTargetMap(
        eventHandler,
        target.getConfiguration(), ImmutableSet.copyOf(depNodeNames.values()), false);

    OrderedSetMultimap<Attribute, ConfiguredTarget> result = OrderedSetMultimap.create();
    for (Map.Entry<Attribute, Dependency> entry : depNodeNames.entries()) {
      result.putAll(entry.getKey(), cts.get(entry.getValue()));
    }
    return result;
  }

  private Transition getTopLevelTransitionForTarget(Label label, ExtendedEventHandler handler) {
    Rule rule;
    try {
      rule = skyframeExecutor
          .getPackageManager()
          .getTarget(handler, label)
          .getAssociatedRule();
    } catch (NoSuchPackageException | NoSuchTargetException e) {
      return ConfigurationTransition.NONE;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new AssertionError("Configuration of " + label + " interrupted");
    }
    if (rule == null) {
      return ConfigurationTransition.NONE;
    }
    RuleTransitionFactory factory = rule
        .getRuleClassObject()
        .getTransitionFactory();
    if (factory == null) {
      return ConfigurationTransition.NONE;
    }

    // dynamicTransitionMapper is only needed because of Attribute.ConfigurationTransition.DATA:
    // this is C++-specific but non-C++ rules declare it. So they can't directly provide the
    // C++-specific patch transition that implements it.
    PatchTransition transition = (PatchTransition)
        ruleClassProvider.getDynamicTransitionMapper().map(factory.buildTransitionFor(rule));
    return (transition == null) ? ConfigurationTransition.NONE : transition;
  }

  /**
   * Returns a configured target for the specified target and configuration. If dynamic
   * configurations are activated, and the target in question has a top-level rule class transition,
   * that transition is applied in the returned ConfiguredTarget. Returns {@code null} if something
   * goes wrong.
   */
  @VisibleForTesting
  public ConfiguredTarget getConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration config) {
    return skyframeExecutor.getConfiguredTargetForTesting(eventHandler, label, config,
        getTopLevelTransitionForTarget(label, eventHandler));
  }

  /**
   * Returns a RuleContext which is the same as the original RuleContext of the target parameter.
   */
  @VisibleForTesting
  public RuleContext getRuleContextForTesting(
      ConfiguredTarget target,
      StoredEventHandler eventHandler,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, ToolchainContextException {
    BuildConfiguration targetConfig = target.getConfiguration();
    CachingAnalysisEnvironment env =
        new CachingAnalysisEnvironment(getArtifactFactory(),
            new ConfiguredTargetKey(target.getLabel(), targetConfig),
            /*isSystemEnv=*/false, targetConfig.extendedSanityChecks(), eventHandler,
            /*skyframeEnv=*/null, targetConfig.isActionsEnabled());
    return getRuleContextForTesting(eventHandler, target, env, configurations);
  }

  /**
   * Creates and returns a rule context that is equivalent to the one that was used to create the
   * given configured target.
   */
  @VisibleForTesting
  public RuleContext getRuleContextForTesting(
      ExtendedEventHandler eventHandler,
      ConfiguredTarget target,
      AnalysisEnvironment env,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, ToolchainContextException {
    BuildConfiguration targetConfig = target.getConfiguration();
    List<Label> requiredToolchains =
        target.getTarget().getAssociatedRule().getRuleClassObject().getRequiredToolchains();
    ToolchainContext toolchainContext =
        skyframeExecutor.getToolchainContextForTesting(
            requiredToolchains, targetConfig, eventHandler);

    return new RuleContext.Builder(
            env,
            (Rule) target.getTarget(),
            ImmutableList.<AspectDescriptor>of(),
            targetConfig,
            configurations.getHostConfiguration(),
            ruleClassProvider.getPrerequisiteValidator(),
            ((Rule) target.getTarget()).getRuleClassObject().getConfigurationFragmentPolicy())
        .setVisibility(
            NestedSetBuilder.<PackageSpecification>create(
                Order.STABLE_ORDER, PackageSpecification.everything()))
        .setPrerequisites(getPrerequisiteMapForTesting(eventHandler, target, configurations))
        .setConfigConditions(ImmutableMap.<Label, ConfigMatchingProvider>of())
        .setUniversalFragment(ruleClassProvider.getUniversalFragment())
        .setToolchainContext(toolchainContext)
        .build();
  }

  /**
   * For a configured target dependentTarget, returns the desired configured target that is depended
   * upon. Useful for obtaining the a target with aspects required by the dependent.
   */
  @VisibleForTesting
  public ConfiguredTarget getPrerequisiteConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler,
      ConfiguredTarget dependentTarget,
      Label desiredTarget,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
             InconsistentAspectOrderException {
    Collection<ConfiguredTarget> configuredTargets =
        getPrerequisiteMapForTesting(eventHandler, dependentTarget, configurations).values();
    for (ConfiguredTarget ct : configuredTargets) {
      if (ct.getLabel().equals(desiredTarget)) {
        return ct;
      }
    }
    return null;
  }

  /**
   * A converter for loading phase thread count. Since the default is not a true constant, we create
   * a converter here to implement the default logic.
   */
  public static final class LoadingPhaseThreadCountConverter implements Converter<Integer> {
    @Override
    public Integer convert(String input) throws OptionsParsingException {
      if ("-1".equals(input)) {
        // Reduce thread count while running tests. Test cases are typically small, and large thread
        // pools vying for a relatively small number of CPU cores may induce non-optimal
        // performance.
        return System.getenv("TEST_TMPDIR") == null ? 200 : 5;
      }

      try {
        int result = Integer.decode(input);
        if (result < 0) {
          throw new OptionsParsingException("'" + input + "' must be at least -1");
        }
        return result;
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' is not an int");
      }
    }

    @Override
    public String getTypeDescription() {
      return "an integer";
    }
  }
}
