// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.util;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableMultiset.toImmutableMultiset;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultiset;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.LoadingPhaseThreadsOption;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestConstants.InternalTestExecutionMode;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import org.junit.Before;

/**
 * Testing framework for tests of the analysis phase that uses the BuildView and LoadingPhaseRunner
 * APIs correctly (compared to {@link BuildViewTestCase}).
 *
 * <p>The intended usage pattern is to first call {@link #update} with the set of targets, and then
 * assert properties of the configured targets obtained from {@link #getConfiguredTarget}.
 *
 * <p>This class intentionally does not inherit from {@link BuildViewTestCase}; BuildViewTestCase
 * abuses the BuildView API in ways that are incompatible with the goals of this test, i.e. the
 * convenience methods provided there wouldn't work here.
 */
public abstract class AnalysisTestCase extends FoundationTestCase {
  private static final int LOADING_PHASE_THREADS = 20;

  /** All the flags that can be passed to {@link BuildView#update}. */
  public enum Flag {
    // The --keep_going flag.
    KEEP_GOING,
    // Configurations that only include the fragments a target needs to properly analyze.
    TRIMMED_CONFIGURATIONS,
    // The --skyframe_prepare_analysis flag.
    SKYFRAME_PREPARE_ANALYSIS,
    // Flags for visibility to default to public.
    PUBLIC_VISIBILITY,
    // Flags for CPU to work (be set to k8) in test mode.
    CPU_K8,
    // Flags from TestConstants.PRODUCT_SPECIFIC_FLAGS.
    PRODUCT_SPECIFIC_FLAGS
  }

  /** Helper class to make it easy to enable and disable flags. */
  public static final class FlagBuilder {
    private final Set<Flag> flags = new HashSet<>();

    public FlagBuilder with(Flag flag) {
      flags.add(flag);
      return this;
    }

    public FlagBuilder without(Flag flag) {
      flags.remove(flag);
      return this;
    }

    public boolean contains(Flag flag) {
      return flags.contains(flag);
    }
  }

  protected BlazeDirectories directories;
  protected MockToolsConfig mockToolsConfig;

  protected AnalysisMock analysisMock;
  protected BuildOptions buildOptions;
  private OptionsParser optionsParser;
  protected PackageManager packageManager;
  private BuildViewForTesting buildView;
  protected final ActionKeyContext actionKeyContext = new ActionKeyContext();

  // Note that these configurations are virtual (they use only VFS)
  private BuildConfigurationCollection masterConfig;

  private AnalysisResult analysisResult;
  protected SkyframeExecutor skyframeExecutor = null;
  protected ConfiguredRuleClassProvider ruleClassProvider;

  protected AnalysisTestUtil.DummyWorkspaceStatusActionFactory workspaceStatusActionFactory;
  private PathPackageLocator pkgLocator;

  @Before
  public final void createMocks() throws Exception {
    analysisMock = getAnalysisMock();
    pkgLocator =
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(Root.fromPath(rootDirectory)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    directories =
        new BlazeDirectories(
            new ServerDirectories(outputBase, outputBase, outputBase),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());
    workspaceStatusActionFactory = new AnalysisTestUtil.DummyWorkspaceStatusActionFactory();

    mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockToolsRepository(mockToolsConfig);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());

    useRuleClassProvider(analysisMock.createRuleClassProvider());
  }

  protected SkyframeExecutor createSkyframeExecutor(PackageFactory pkgFactory) {
    return BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
        .setPkgFactory(pkgFactory)
        .setFileSystem(fileSystem)
        .setDirectories(directories)
        .setActionKeyContext(actionKeyContext)
        .setDefaultBuildOptions(
            DefaultBuildOptionsForTesting.getDefaultBuildOptionsForTest(ruleClassProvider))
        .setWorkspaceStatusActionFactory(workspaceStatusActionFactory)
        .setExtraSkyFunctions(analysisMock.getSkyFunctions(directories))
        .build();
  }

  /**
   * Changes the rule class provider to be used for the loading and the analysis phase.
   */
  protected void useRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider)
      throws Exception {
    this.ruleClassProvider = ruleClassProvider;
    PackageFactory pkgFactory =
        analysisMock
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);
    useConfiguration();
    skyframeExecutor = createSkyframeExecutor(pkgFactory);
    reinitializeSkyframeExecutor();
    packageManager = skyframeExecutor.getPackageManager();
    buildView = new BuildViewForTesting(directories, ruleClassProvider, skyframeExecutor, null);
  }

  private void reinitializeSkyframeExecutor() {
    SkyframeExecutorTestHelper.process(skyframeExecutor);
    PackageCacheOptions packageCacheOptions = Options.getDefaults(PackageCacheOptions.class);
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 3;
    skyframeExecutor.preparePackageLoading(
        pkgLocator,
        packageCacheOptions,
        Options.getDefaults(StarlarkSemanticsOptions.class),
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
                Optional.<RootedPath>absent()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.REPOSITORY_OVERRIDES,
                ImmutableMap.<RepositoryName, PathFragment>of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY),
            PrecomputedValue.injected(
                PrecomputedValue.BUILD_INFO_FACTORIES,
                ruleClassProvider.getBuildInfoFactoriesAsMap())));
  }

  /** Resets the SkyframeExecutor, as if a clean had been executed. */
  protected void cleanSkyframe() {
    skyframeExecutor.resetEvaluator();
    reinitializeSkyframeExecutor();
  }

  protected AnalysisMock getAnalysisMock() {
    return AnalysisMock.get();
  }

  protected InternalTestExecutionMode getInternalTestExecutionMode() {
    return InternalTestExecutionMode.NORMAL;
  }

  /**
   * Sets host and target configuration using the specified options, falling back to the default
   * options for unspecified ones, and recreates the build view.
   */
  public final void useConfiguration(String... args) throws Exception {
    optionsParser =
        OptionsParser.builder()
            .optionsClasses(
                Iterables.concat(
                    Arrays.asList(
                        ExecutionOptions.class,
                        PackageCacheOptions.class,
                        StarlarkSemanticsOptions.class,
                        BuildRequestOptions.class,
                        AnalysisOptions.class,
                        KeepGoingOption.class,
                        LoadingPhaseThreadsOption.class,
                        LoadingOptions.class),
                    ruleClassProvider.getConfigurationOptions()))
            .build();
    if (defaultFlags().contains(Flag.PUBLIC_VISIBILITY)) {
      optionsParser.parse("--default_visibility=public");
    }
    if (defaultFlags().contains(Flag.CPU_K8)) {
      optionsParser.parse("--cpu=k8", "--host_cpu=k8");
    }
    if (defaultFlags().contains(Flag.PRODUCT_SPECIFIC_FLAGS)) {
      optionsParser.parse(TestConstants.PRODUCT_SPECIFIC_FLAGS);
    }
    optionsParser.parse(args);
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      optionsParser.parse("--experimental_dynamic_configs=on");
    }

    buildOptions = ruleClassProvider.createBuildOptions(optionsParser);
  }

  protected FlagBuilder defaultFlags() {
    return new FlagBuilder()
        .with(Flag.PUBLIC_VISIBILITY)
        .with(Flag.CPU_K8)
        .with(Flag.PRODUCT_SPECIFIC_FLAGS);
  }

  protected Action getGeneratingAction(Artifact artifact) {
    ensureUpdateWasCalled();
    ActionAnalysisMetadata action = analysisResult.getActionGraph().getGeneratingAction(artifact);

    if (action != null) {
      Preconditions.checkState(
          action instanceof Action,
          "%s is not a proper Action object",
          action.prettyPrint());
      return (Action) action;
    } else {
      return null;
    }
  }

  protected BuildConfigurationCollection getBuildConfigurationCollection() {
    return masterConfig;
  }

  /**
   * Returns the target configuration for the most recent build, as created in Blaze's
   * master configuration creation phase.
   */
  protected BuildConfiguration getTargetConfiguration() throws InterruptedException {
    return Iterables.getOnlyElement(masterConfig.getTargetConfigurations());
  }

  protected BuildConfiguration getHostConfiguration() {
    return masterConfig.getHostConfiguration();
  }

  protected final void ensureUpdateWasCalled() {
    Preconditions.checkState(analysisResult != null, "You must run update() first!");
  }

  /**
   * Update the BuildView: syncs the package cache; loads and analyzes the given labels.
   */
  protected AnalysisResult update(
      EventBus eventBus, FlagBuilder config, ImmutableList<String> aspects, String... labels)
          throws Exception {
    Set<Flag> flags = config.flags;

    LoadingOptions loadingOptions = optionsParser.getOptions(LoadingOptions.class);

    AnalysisOptions viewOptions = optionsParser.getOptions(AnalysisOptions.class);
    // update --keep_going option if test requested it.
    boolean keepGoing = flags.contains(Flag.KEEP_GOING);
    boolean discardAnalysisCache = viewOptions.discardAnalysisCache;
    viewOptions.skyframePrepareAnalysis = flags.contains(Flag.SKYFRAME_PREPARE_ANALYSIS);

    PackageCacheOptions packageCacheOptions = optionsParser.getOptions(PackageCacheOptions.class);
    PathPackageLocator pathPackageLocator =
        PathPackageLocator.create(
            outputBase,
            packageCacheOptions.packagePath,
            reporter,
            rootDirectory,
            rootDirectory,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 7;

    StarlarkSemanticsOptions starlarkSemanticsOptions =
        optionsParser.getOptions(StarlarkSemanticsOptions.class);

    skyframeExecutor.preparePackageLoading(
        pathPackageLocator,
        packageCacheOptions,
        starlarkSemanticsOptions,
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, Root.fromPath(rootDirectory));

    TargetPatternPhaseValue loadingResult =
        skyframeExecutor.loadTargetPatternsWithFilters(
            reporter,
            ImmutableList.copyOf(labels),
            PathFragment.EMPTY_FRAGMENT,
            loadingOptions,
            LOADING_PHASE_THREADS,
            keepGoing,
            /*determineTests=*/ false);

    BuildRequestOptions requestOptions = optionsParser.getOptions(BuildRequestOptions.class);
    ImmutableSortedSet<String> multiCpu = ImmutableSortedSet.copyOf(requestOptions.multiCpus);
    analysisResult =
        buildView.update(
            loadingResult,
            buildOptions,
            multiCpu,
            aspects,
            viewOptions,
            keepGoing,
            LOADING_PHASE_THREADS,
            AnalysisTestUtil.TOP_LEVEL_ARTIFACT_CONTEXT,
            reporter,
            eventBus);
    if (discardAnalysisCache) {
      buildView.clearAnalysisCache(analysisResult.getTargetsToBuild(), analysisResult.getAspects());
    }
    masterConfig = analysisResult.getConfigurationCollection();
    return analysisResult;
  }

  protected AnalysisResult update(EventBus eventBus, FlagBuilder config, String... labels)
      throws Exception {
    return update(eventBus, config, /*aspects=*/ImmutableList.<String>of(), labels);
  }

  protected AnalysisResult update(FlagBuilder config, String... labels) throws Exception {
    return update(new EventBus(), config, /*aspects=*/ImmutableList.<String>of(), labels);
  }

  /**
   * Update the BuildView: syncs the package cache; loads and analyzes the given labels.
   */
  protected AnalysisResult update(String... labels) throws Exception {
    return update(new EventBus(), defaultFlags(), /*aspects=*/ImmutableList.<String>of(), labels);
  }

  protected AnalysisResult update(ImmutableList<String> aspects, String... labels)
      throws Exception {
    return update(new EventBus(), defaultFlags(), aspects, labels);
  }

  protected ConfiguredTargetAndData getConfiguredTargetAndTarget(String label)
      throws InterruptedException {
    return getConfiguredTargetAndTarget(label, getTargetConfiguration());
  }

  protected ConfiguredTargetAndData getConfiguredTargetAndTarget(
      String label, BuildConfiguration config) {
    ensureUpdateWasCalled();
    Label parsedLabel;
    try {
      parsedLabel = Label.parseAbsolute(label, ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
    try {
      return skyframeExecutor.getConfiguredTargetAndDataForTesting(reporter, parsedLabel, config);
    } catch (StarlarkTransition.TransitionException
        | InvalidConfigurationException
        | InterruptedException e) {
      throw new AssertionError(e);
    }
  }

  protected Target getTarget(String label) throws InterruptedException {
    try {
      return SkyframeExecutorTestUtils.getExistingTarget(
          skyframeExecutor, Label.parseAbsolute(label, ImmutableMap.of()));
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
  }

  protected final ConfiguredTargetAndData getConfiguredTargetAndData(
      String label, BuildConfiguration configuration) {
    ensureUpdateWasCalled();
    return getConfiguredTargetForSkyframe(label, configuration);
  }

  protected final ConfiguredTargetAndData getConfiguredTargetAndData(String label)
      throws InterruptedException {
    return getConfiguredTargetAndData(label, getTargetConfiguration());
  }

  protected final ConfiguredTarget getConfiguredTarget(
      String label, BuildConfiguration configuration) {
    ConfiguredTargetAndData result = getConfiguredTargetAndData(label, configuration);
    return result == null ? null : result.getConfiguredTarget();
  }

  /**
   * Returns the corresponding configured target, if it exists. Note that this will only return
   * anything useful after a call to update() with the same label.
   */
  protected ConfiguredTarget getConfiguredTarget(String label) throws InterruptedException {
    return getConfiguredTarget(label, getTargetConfiguration());
  }

  private ConfiguredTargetAndData getConfiguredTargetForSkyframe(
      String label, BuildConfiguration configuration) {
    Label parsedLabel;
    try {
      parsedLabel = Label.parseAbsolute(label, ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
    try {
      return skyframeExecutor.getConfiguredTargetAndDataForTesting(
          reporter, parsedLabel, configuration);
    } catch (StarlarkTransition.TransitionException
        | InvalidConfigurationException
        | InterruptedException e) {
      throw new AssertionError(e);
    }
  }

  protected final BuildConfiguration getConfiguration(ConfiguredTarget ct) {
    return skyframeExecutor.getConfiguration(reporter, ct.getConfigurationKey());
  }

  /**
   * Returns the corresponding configured target, if it exists. Note that this will only return
   * anything useful after a call to update() with the same label. The label passed in must
   * represent an input file.
   */
  protected InputFileConfiguredTarget getInputFileConfiguredTarget(String label) {
    return (InputFileConfiguredTarget) getConfiguredTarget(label, null);
  }

  protected boolean hasErrors(ConfiguredTarget configuredTarget) {
    return buildView.hasErrors(configuredTarget);
  }

  protected Artifact getBinArtifact(String packageRelativePath, ConfiguredTarget owner)
      throws InterruptedException {
    Label label = owner.getLabel();
    ActionLookupValue.ActionLookupKey actionLookupKey =
        ConfiguredTargetKey.of(label, owner.getConfigurationKey(), /*isHostConfiguration=*/ false);
    ActionLookupValue actionLookupValue;
    try {
      actionLookupValue =
          (ActionLookupValue)
              skyframeExecutor.getEvaluatorForTesting().getExistingValue(actionLookupKey);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
    PathFragment rootRelativePath = label.getPackageFragment().getRelative(packageRelativePath);
    for (ActionAnalysisMetadata action : actionLookupValue.getActions()) {
      for (Artifact output : action.getOutputs()) {
        if (output.getRootRelativePath().equals(rootRelativePath)) {
          return output;
        }
      }
    }
    // Fall back: some tests don't actually need the right owner.
    return buildView
        .getArtifactFactory()
        .getDerivedArtifact(
            label.getPackageFragment().getRelative(packageRelativePath),
            getTargetConfiguration().getBinDirectory(label.getPackageIdentifier().getRepository()),
            ConfiguredTargetKey.of(
                owner, skyframeExecutor.getConfiguration(reporter, owner.getConfigurationKey())));
  }

  protected Set<SkyKey> getSkyframeEvaluatedTargetKeys() {
    return buildView.getSkyframeEvaluatedTargetKeysForTesting();
  }

  protected void assertNumberOfAnalyzedConfigurationsOfTargets(
      Map<String, Integer> targetsWithCounts) {
    ImmutableMultiset<Label> actualSet =
        getSkyframeEvaluatedTargetKeys().stream()
            .filter(key -> key instanceof ConfiguredTargetKey)
            .map(key -> ((ConfiguredTargetKey) key).getLabel())
            .collect(toImmutableMultiset());
    ImmutableMap<Label, Integer> expected =
        targetsWithCounts.entrySet().stream()
            .collect(
                toImmutableMap(
                    entry -> Label.parseAbsoluteUnchecked(entry.getKey()),
                    entry -> entry.getValue()));
    ImmutableMap<Label, Integer> actual =
        expected.keySet().stream()
            .collect(toImmutableMap(label -> label, label -> actualSet.count(label)));
    assertThat(actual).containsExactlyEntriesIn(expected);
  }

  protected String getAnalysisError() {
    ensureUpdateWasCalled();
    return analysisResult.getError();
  }

  protected BuildViewForTesting getView() {
    return buildView;
  }

  protected ActionGraph getActionGraph() {
    return skyframeExecutor.getActionGraph(reporter);
  }

  protected AnalysisResult getAnalysisResult() {
    return analysisResult;
  }

  protected void clearAnalysisResult() {
    analysisResult = null;
  }

  /**
   * Makes {@code rules} available in tests, in addition to all the rules available to Blaze at
   * running time (e.g., java_library).
   *
   * Also see {@link AnalysisTestCase#setRulesAndAspectsAvailableInTests(Iterable, Iterable)}.
   */
  protected void setRulesAvailableInTests(RuleDefinition... rules) throws Exception {
    // Not all of these aspects are needed for all tests, but it makes it simple to offer them all.
    setRulesAndAspectsAvailableInTests(
        ImmutableList.of(
            TestAspects.SIMPLE_ASPECT,
            TestAspects.PARAMETRIZED_DEFINITION_ASPECT,
            TestAspects.ASPECT_REQUIRING_PROVIDER,
            TestAspects.FALSE_ADVERTISEMENT_ASPECT,
            TestAspects.ALL_ATTRIBUTES_ASPECT,
            TestAspects.ALL_ATTRIBUTES_WITH_TOOL_ASPECT,
            TestAspects.BAR_PROVIDER_ASPECT,
            TestAspects.EXTRA_ATTRIBUTE_ASPECT,
            TestAspects.FOO_PROVIDER_ASPECT,
            TestAspects.ASPECT_REQUIRING_PROVIDER_SETS,
            TestAspects.WARNING_ASPECT,
            TestAspects.ERROR_ASPECT),
        ImmutableList.copyOf(rules));
  }

  /**
   * Makes {@code aspects} and {@code rules} available in tests, in addition to
   * all the rules available to Blaze at running time (e.g., java_library).
   */
  protected final void setRulesAndAspectsAvailableInTests(
      Iterable<NativeAspectClass> aspects,
      Iterable<RuleDefinition> rules) throws Exception {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    for (NativeAspectClass aspect : aspects) {
      builder.addNativeAspectClass(aspect);
    }
    for (RuleDefinition rule : rules) {
      builder.addRuleDefinition(rule);
    }

    useRuleClassProvider(builder.build());
    update();
  }

  /**
   * Makes custom configuration fragments available in tests.
   */
  protected final void setConfigFragmentsAvailableInTests(
      ConfigurationFragmentFactory... factories) throws Exception {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    for (ConfigurationFragmentFactory factory : factories) {
      builder.addConfigurationFragment(factory);
    }
    useRuleClassProvider(builder.build());
  }
}
