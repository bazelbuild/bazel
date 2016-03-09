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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.buildtool.BuildRequest.BuildRequestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseRunner;
import com.google.devtools.build.lib.pkgcache.LoadingResult;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.runtime.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;

import org.junit.Before;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

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
    KEEP_GOING,
    SKYFRAME_LOADING_PHASE,
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

  private OptionsParser optionsParser;
  protected PackageManager packageManager;
  private LoadingPhaseRunner loadingPhaseRunner;
  private ConfigurationFactory configurationFactory;
  private BuildView buildView;

  // Note that these configurations are virtual (they use only VFS)
  private BuildConfigurationCollection masterConfig;

  private AnalysisResult analysisResult;
  protected SkyframeExecutor skyframeExecutor = null;
  protected ConfiguredRuleClassProvider ruleClassProvider;

  protected AnalysisTestUtil.DummyWorkspaceStatusActionFactory workspaceStatusActionFactory;
  private PathPackageLocator pkgLocator;
  private AnalysisMock analysisMock;

  @Before
  public final void createMocks() throws Exception {
    analysisMock = AnalysisMock.get();
    pkgLocator = new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory));
    directories = new BlazeDirectories(outputBase, outputBase, rootDirectory);
    workspaceStatusActionFactory =
        new AnalysisTestUtil.DummyWorkspaceStatusActionFactory(directories);

    mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());
    configurationFactory = analysisMock.createConfigurationFactory();

    useRuleClassProvider(TestRuleClassProvider.getRuleClassProvider());
  }

  /**
   * Changes the rule class provider to be used for the loading and the analysis phase.
   */
  protected void useRuleClassProvider(ConfiguredRuleClassProvider ruleClassProvider)
      throws Exception {
    this.ruleClassProvider = ruleClassProvider;
    PackageFactory pkgFactory = new PackageFactory(ruleClassProvider);
    BinTools binTools = BinTools.forUnitTesting(directories, TestConstants.EMBEDDED_TOOLS);
    skyframeExecutor =
        SequencedSkyframeExecutor.create(
            pkgFactory,
            new TimestampGranularityMonitor(BlazeClock.instance()),
            directories,
            binTools,
            workspaceStatusActionFactory,
            ruleClassProvider.getBuildInfoFactories(),
            ImmutableList.<DiffAwareness.Factory>of(),
            Predicates.<PathFragment>alwaysFalse(),
            Preprocessor.Factory.Supplier.NullSupplier.INSTANCE,
            analysisMock.getSkyFunctions(directories),
            getPrecomputedValues(),
            ImmutableList.<SkyValueDirtinessChecker>of());
    skyframeExecutor.preparePackageLoading(pkgLocator,
        Options.getDefaults(PackageCacheOptions.class).defaultVisibility, true,
        3, ruleClassProvider.getDefaultsPackageContent(), UUID.randomUUID());
    packageManager = skyframeExecutor.getPackageManager();
    loadingPhaseRunner = skyframeExecutor.getLoadingPhaseRunner(
        pkgFactory.getRuleClassNames(), defaultFlags().contains(Flag.SKYFRAME_LOADING_PHASE));
    buildView = new BuildView(directories, ruleClassProvider, skyframeExecutor, null);
    useConfiguration();
  }

  protected ImmutableList<PrecomputedValue.Injected> getPrecomputedValues() {
    return ImmutableList.of();
  }

  protected final void useConfigurationFactory(ConfigurationFactory configurationFactory) {
    this.configurationFactory = configurationFactory;
  }

  /**
   * Sets host and target configuration using the specified options, falling back to the default
   * options for unspecified ones, and recreates the build view.
   */
  protected final void useConfiguration(String... args) throws Exception {
    optionsParser = OptionsParser.newOptionsParser(Iterables.concat(Arrays.asList(
        ExecutionOptions.class,
        PackageCacheOptions.class,
        BuildRequestOptions.class,
        BuildView.Options.class),
        ruleClassProvider.getConfigurationOptions()));
    optionsParser.parse(new String[] {"--default_visibility=public" });
    optionsParser.parse(args);

    InvocationPolicyEnforcer optionsPolicyEnforcer =
        new InvocationPolicyEnforcer(TestConstants.TEST_INVOCATION_POLICY);
    optionsPolicyEnforcer.enforce(optionsParser);
  }

  protected FlagBuilder defaultFlags() {
    return new FlagBuilder();
  }

  protected Action getGeneratingAction(Artifact artifact) {
    ensureUpdateWasCalled();
    return analysisResult.getActionGraph().getGeneratingAction(artifact);
  }

  protected BuildConfigurationCollection getBuildConfigurationCollection() {
    return masterConfig;
  }

  protected BuildConfiguration getTargetConfiguration() {
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

    LoadingOptions loadingOptions = Options.getDefaults(LoadingOptions.class);
    loadingOptions.loadingPhaseThreads = LOADING_PHASE_THREADS;

    BuildView.Options viewOptions = optionsParser.getOptions(BuildView.Options.class);
    viewOptions.keepGoing = flags.contains(Flag.KEEP_GOING);

    BuildOptions buildOptions = ruleClassProvider.createBuildOptions(optionsParser);
    PackageCacheOptions packageCacheOptions = optionsParser.getOptions(PackageCacheOptions.class);

    PathPackageLocator pathPackageLocator = PathPackageLocator.create(
        outputBase, packageCacheOptions.packagePath, reporter, rootDirectory, rootDirectory);
    skyframeExecutor.preparePackageLoading(pathPackageLocator,
        packageCacheOptions.defaultVisibility, true,
        7, ruleClassProvider.getDefaultsPackageContent(), UUID.randomUUID());
    skyframeExecutor.invalidateFilesUnderPathForTesting(reporter,
        ModifiedFileSet.EVERYTHING_MODIFIED, rootDirectory);

    LoadingResult loadingResult = loadingPhaseRunner
        .execute(reporter, eventBus, ImmutableList.copyOf(labels), PathFragment.EMPTY_FRAGMENT,
            loadingOptions, buildOptions.getAllLabels(), viewOptions.keepGoing, isLoadingEnabled(),
            /*determineTests=*/false, /*callback=*/null);

    BuildRequestOptions requestOptions = optionsParser.getOptions(BuildRequestOptions.class);
    ImmutableSortedSet<String> multiCpu = ImmutableSortedSet.copyOf(requestOptions.multiCpus);
    masterConfig = skyframeExecutor.createConfigurations(
        reporter, configurationFactory, buildOptions, directories, multiCpu, false);
    analysisResult =
        buildView.update(
            loadingResult,
            masterConfig,
            aspects,
            viewOptions,
            AnalysisTestUtil.TOP_LEVEL_ARTIFACT_CONTEXT,
            reporter,
            eventBus,
            isLoadingEnabled());
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

  protected Target getTarget(String label) {
    try {
      return SkyframeExecutorTestUtils.getExistingTarget(skyframeExecutor,
          Label.parseAbsolute(label));
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
  }

  protected ConfiguredTarget getConfiguredTarget(String label, BuildConfiguration configuration) {
    ensureUpdateWasCalled();
    return getConfiguredTargetForSkyframe(label, configuration);
  }

  private ConfiguredTarget getConfiguredTargetForSkyframe(String label,
      BuildConfiguration configuration) {
    Label parsedLabel;
    try {
      parsedLabel = Label.parseAbsolute(label);
    } catch (LabelSyntaxException e) {
      throw new AssertionError(e);
    }
    return skyframeExecutor.getConfiguredTargetForTesting(reporter, parsedLabel, configuration);
  }

  /**
   * Returns the corresponding configured target, if it exists. Note that this will only return
   * anything useful after a call to update() with the same label.
   */
  protected ConfiguredTarget getConfiguredTarget(String label) {
    return getConfiguredTarget(label, getTargetConfiguration());
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

  protected Artifact getBinArtifact(String packageRelativePath, ConfiguredTarget owner) {
    Label label = owner.getLabel();
    return buildView.getArtifactFactory().getDerivedArtifact(
        label.getPackageFragment().getRelative(packageRelativePath),
        getTargetConfiguration().getBinDirectory(),
        new ConfiguredTargetKey(owner));
  }

  protected Set<SkyKey> getSkyframeEvaluatedTargetKeys() {
    return buildView.getSkyframeEvaluatedTargetKeysForTesting();
  }

  protected int getTargetsVisited() {
    return buildView.getTargetsVisited();
  }

  protected String getAnalysisError() {
    ensureUpdateWasCalled();
    return analysisResult.getError();
  }

  protected BuildView getView() {
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
}
