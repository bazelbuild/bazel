// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Ascii;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.LostInputsCheck;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLogBufferPathGenerator;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineAndParamFileInfo;
import com.google.devtools.build.lib.actions.MapBasedActionGraph;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ExtraActionArtifactsProvider;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PseudoAction;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.ConfigsMode;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.FileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.extra.ExtraAction;
import com.google.devtools.build.lib.analysis.skylark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.test.BaselineCoverageAction;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.NestedSetExpander;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.PackageValidator;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.skyframe.AspectValueKey;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionFunction;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.PackageRootsNoSymlinkCreation;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import javax.annotation.Nullable;
import org.junit.Before;

/**
 * Common test code that creates a BuildView instance.
 */
public abstract class BuildViewTestCase extends FoundationTestCase {
  protected static final int LOADING_PHASE_THREADS = 20;

  protected AnalysisMock analysisMock;
  protected ConfiguredRuleClassProvider ruleClassProvider;
  protected BuildViewForTesting view;

  protected SequencedSkyframeExecutor skyframeExecutor;

  protected TimestampGranularityMonitor tsgm;
  protected BlazeDirectories directories;
  protected ActionKeyContext actionKeyContext;

  // Note that these configurations are virtual (they use only VFS)
  protected BuildConfigurationCollection masterConfig;
  protected BuildConfiguration targetConfig;  // "target" or "build" config
  private List<String> configurationArgs;
  private ConfigsMode configsMode = ConfigsMode.NOTRIM;

  protected OptionsParser optionsParser;
  private PackageOptions packageOptions;
  private StarlarkSemanticsOptions starlarkSemanticsOptions;
  protected PackageFactory pkgFactory;

  protected MockToolsConfig mockToolsConfig;

  protected WorkspaceStatusAction.Factory workspaceStatusActionFactory;

  private MutableActionGraph mutableActionGraph;

  private LoadingOptions customLoadingOptions = null;
  protected BuildConfigurationValue.Key targetConfigKey;

  private ActionLogBufferPathGenerator actionLogBufferPathGenerator;

  @Before
  public final void initializeSkyframeExecutor() throws Exception {
    initializeSkyframeExecutor(/*doPackageLoadingChecks=*/ true);
  }

  public void initializeSkyframeExecutor(boolean doPackageLoadingChecks) throws Exception {
    analysisMock = getAnalysisMock();
    directories =
        new BlazeDirectories(
            new ServerDirectories(outputBase, outputBase, outputBase),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());

    actionKeyContext = new ActionKeyContext();
    mockToolsConfig = new MockToolsConfig(rootDirectory, false);
    mockToolsConfig.create("bazel_tools_workspace/WORKSPACE", "workspace(name = 'bazel_tools')");
    mockToolsConfig.create("bazel_tools_workspace/tools/build_defs/repo/BUILD");
    mockToolsConfig.create(
        "bazel_tools_workspace/tools/build_defs/repo/utils.bzl",
        "def maybe(repo_rule, name, **kwargs):",
        "  if name not in native.existing_rules():",
        "    repo_rule(name = name, **kwargs)");
    mockToolsConfig.create(
        "bazel_tools_workspace/tools/build_defs/repo/http.bzl",
        "def http_archive(**kwargs):",
        "  pass",
        "",
        "def http_file(**kwargs):",
        "  pass");
    initializeMockClient();

    packageOptions = parsePackageOptions();
    starlarkSemanticsOptions = parseSkylarkSemanticsOptions();
    workspaceStatusActionFactory = new AnalysisTestUtil.DummyWorkspaceStatusActionFactory();
    mutableActionGraph = new MapBasedActionGraph(actionKeyContext);
    ruleClassProvider = getRuleClassProvider();
    getOutputPath().createDirectoryAndParents();
    ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues =
        ImmutableList.of(
            PrecomputedValue.injected(
                PrecomputedValue.STARLARK_SEMANTICS, StarlarkSemantics.DEFAULT_SEMANTICS),
            PrecomputedValue.injected(PrecomputedValue.REPO_ENV, ImmutableMap.<String, String>of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.REPOSITORY_OVERRIDES,
                ImmutableMap.<RepositoryName, PathFragment>of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
                Optional.<RootedPath>absent()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY),
            PrecomputedValue.injected(
                BuildInfoCollectionFunction.BUILD_INFO_FACTORIES,
                ruleClassProvider.getBuildInfoFactoriesAsMap()));
    PackageFactory.BuilderForTesting pkgFactoryBuilder =
        analysisMock
            .getPackageFactoryBuilderForTesting(directories)
            .setExtraPrecomputeValues(extraPrecomputedValues)
            .setEnvironmentExtensions(getEnvironmentExtensions())
            .setPackageValidator(getPackageValidator());
    if (!doPackageLoadingChecks) {
      pkgFactoryBuilder.disableChecks();
    }
    pkgFactory = pkgFactoryBuilder.build(ruleClassProvider, fileSystem);
    tsgm = new TimestampGranularityMonitor(BlazeClock.instance());
    skyframeExecutor =
        BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
            .setPkgFactory(pkgFactory)
            .setFileSystem(fileSystem)
            .setDirectories(directories)
            .setActionKeyContext(actionKeyContext)
            .setDefaultBuildOptions(
                DefaultBuildOptionsForTesting.getDefaultBuildOptionsForTest(ruleClassProvider))
            .setWorkspaceStatusActionFactory(workspaceStatusActionFactory)
            .setExtraSkyFunctions(analysisMock.getSkyFunctions(directories))
            .setManagedDirectoriesKnowledge(getManagedDirectoriesKnowledge())
            .build();
    SkyframeExecutorTestHelper.process(skyframeExecutor);
    skyframeExecutor.injectExtraPrecomputedValues(extraPrecomputedValues);
    packageOptions.defaultVisibility = ConstantRuleVisibility.PUBLIC;
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    skyframeExecutor.preparePackageLoading(
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(root),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY),
        packageOptions,
        starlarkSemanticsOptions,
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        tsgm);
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    useConfiguration();
    setUpSkyframe();
    this.actionLogBufferPathGenerator =
        new ActionLogBufferPathGenerator(
            directories.getActionTempsDirectory(getExecRoot()),
            directories.getPersistentActionOutsDirectory(getExecRoot()));
  }

  public void initializeMockClient() throws IOException {
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());
  }

  protected AnalysisMock getAnalysisMock() {
    return AnalysisMock.get();
  }

  /** Creates or retrieves the rule class provider used in this test. */
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    return getAnalysisMock().createRuleClassProvider();
  }

  protected ManagedDirectoriesKnowledge getManagedDirectoriesKnowledge() {
    return null;
  }

  protected PackageFactory getPackageFactory() {
    return pkgFactory;
  }

  protected Iterable<EnvironmentExtension> getEnvironmentExtensions() {
    return ImmutableList.<EnvironmentExtension>of();
  }

  protected StarlarkSemantics getSkylarkSemantics() {
    return starlarkSemanticsOptions.toSkylarkSemantics();
  }

  protected PackageValidator getPackageValidator() {
    return PackageValidator.NOOP_VALIDATOR;
  }

  protected final BuildConfigurationCollection createConfigurations(
      ImmutableMap<String, Object> skylarkOptions, String... args) throws Exception {
    optionsParser =
        OptionsParser.builder()
            .optionsClasses(
                Iterables.concat(
                    Arrays.asList(ExecutionOptions.class, BuildRequestOptions.class),
                    ruleClassProvider.getConfigurationOptions()))
            .build();
    List<String> allArgs = new ArrayList<>();
    // TODO(dmarting): Add --stamp option only to test that requires it.
    allArgs.add("--stamp");  // Stamp is now defaulted to false.
    allArgs.add("--experimental_extended_sanity_checks");
    allArgs.add("--features=cc_include_scanning");
    // Always default to k8, even on mac and windows. Tests that need different cpu should set it
    // using {@link useConfiguration()} explicitly.
    allArgs.add("--cpu=k8");
    allArgs.add("--host_cpu=k8");

    optionsParser.parse(allArgs);
    optionsParser.parse(args);

    // TODO(juliexxia): when the starlark options parsing work goes in, add type verification here.
    optionsParser.setStarlarkOptions(skylarkOptions);

    BuildOptions buildOptions = ruleClassProvider.createBuildOptions(optionsParser);
    return skyframeExecutor.createConfigurations(
        reporter, buildOptions, ImmutableSet.<String>of(), false);
  }

  protected Target getTarget(String label)
      throws NoSuchPackageException, NoSuchTargetException,
      LabelSyntaxException, InterruptedException {
    return getTarget(Label.parseAbsolute(label, ImmutableMap.of()));
  }

  protected Target getTarget(Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
    return getPackageManager().getTarget(reporter, label);
  }

  /**
   * Checks that loading the given target fails with the expected error message.
   *
   * <p>Fails with an assertion error if this doesn't happen.
   *
   * <p>This method is useful for checking loading phase errors. Analysis phase errors can be
   * checked with {@link #getConfiguredTarget} and related methods.
   */
  protected void assertTargetError(String label, String expectedError)
      throws InterruptedException {
    try {
      getTarget(label);
      fail("Expected loading phase failure for target " + label);
    } catch (NoSuchPackageException | NoSuchTargetException | LabelSyntaxException e) {
      // Target loading failed as expected.
    }
    assertContainsEvent(expectedError);
  }

  private void setUpSkyframe() {
    PathPackageLocator pkgLocator =
        PathPackageLocator.create(
            outputBase,
            packageOptions.packagePath,
            reporter,
            rootDirectory,
            rootDirectory,
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    skyframeExecutor.preparePackageLoading(
        pkgLocator,
        packageOptions,
        starlarkSemanticsOptions,
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        tsgm);
    skyframeExecutor.setActionEnv(ImmutableMap.<String, String>of());
    skyframeExecutor.setDeletedPackages(ImmutableSet.copyOf(packageOptions.getDeletedPackages()));
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE,
                Optional.<RootedPath>absent()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.OUTPUT_VERIFICATION_REPOSITORY_RULES,
                ImmutableSet.<String>of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_FOR_VERIFICATION,
                Optional.<RootedPath>absent())));
  }

  protected void setPackageOptions(String... options) throws Exception {
    packageOptions = parsePackageOptions(options);
    setUpSkyframe();
  }

  protected void setSkylarkSemanticsOptions(String... options) throws Exception {
    starlarkSemanticsOptions = parseSkylarkSemanticsOptions(options);
    setUpSkyframe();
  }

  private static PackageOptions parsePackageOptions(String... options) throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(PackageOptions.class).build();
    parser.parse("--default_visibility=public");
    parser.parse(options);
    return parser.getOptions(PackageOptions.class);
  }

  private static StarlarkSemanticsOptions parseSkylarkSemanticsOptions(String... options)
      throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(StarlarkSemanticsOptions.class).build();
    parser.parse(options);
    return parser.getOptions(StarlarkSemanticsOptions.class);
  }

  /** Used by skyframe-only tests. */
  protected SequencedSkyframeExecutor getSkyframeExecutor() {
    return Preconditions.checkNotNull(skyframeExecutor);
  }

  protected PackageManager getPackageManager() {
    return skyframeExecutor.getPackageManager();
  }

  protected void invalidatePackages() throws InterruptedException {
    invalidatePackages(true);
  }

  /**
   * Invalidates all existing packages. Optionally invalidates configurations too.
   *
   * <p>Tests should invalidate both unless they have specific reason not to.
   *
   * @throws InterruptedException
   */
  protected void invalidatePackages(boolean alsoConfigs) throws InterruptedException {
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter, ModifiedFileSet.EVERYTHING_MODIFIED, Root.fromPath(rootDirectory));
    if (alsoConfigs) {
      try {
        // Also invalidate all configurations. This is important: by invalidating all files we
        // invalidate CROSSTOOL, which invalidates CppConfiguration (and a few other fragments). So
        // we need to invalidate the {@link SkyframeBuildView#hostConfigurationCache} as well.
        // Otherwise we end up with old CppConfiguration instances. Even though they're logically
        // equal to the new ones, CppConfiguration has no .equals() method and some production code
        // expects equality.
        useConfiguration(configurationArgs.toArray(new String[0]));
      } catch (Exception e) {
        // There are enough dependers on this method that don't handle Exception that just passing
        // through the Exception would result in a huge refactoring. As it stands this shouldn't
        // fail anyway because this method only gets called after a successful useConfiguration()
        // call anyway.
        throw new RuntimeException(e);
      }
    }
  }

  /**
   * Sets host and target configuration using the specified options, falling back to the default
   * options for unspecified ones, and recreates the build view.
   *
   * <p>TODO(juliexxia): when Starlark option parsing exists, find a way to combine these parameters
   * into a single parameter so Starlark/native options don't have to be specified separately.
   *
   * @param skylarkOptions map of Starlark-defined options where the keys are option names (in the
   *     form of label-like strings) and the values are option values
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   * @throws IllegalArgumentException
   */
  protected void useConfiguration(ImmutableMap<String, Object> skylarkOptions, String... args)
      throws Exception {
    ImmutableList<String> actualArgs =
        ImmutableList.<String>builder()
            .addAll(TestConstants.PRODUCT_SPECIFIC_FLAGS)
            .add(args)
            .add("--experimental_dynamic_configs=" + Ascii.toLowerCase(configsMode.toString()))
            .build();

    masterConfig = createConfigurations(skylarkOptions, actualArgs.toArray(new String[0]));
    targetConfig = getTargetConfiguration();
    targetConfigKey = BuildConfigurationValue.key(targetConfig);
    configurationArgs = actualArgs;
    createBuildView();
  }

  protected void useConfiguration(String... args) throws Exception {
    useConfiguration(ImmutableMap.of(), args);
  }

  /**
   * Makes subsequent {@link #useConfiguration} calls automatically use the specified style for
   * configurations.
   */
  protected final void useConfigurationMode(ConfigsMode mode) {
    configsMode = mode;
  }

  /**
   * Creates BuildView using current hostConfig/targetConfig values.
   * Ensures that hostConfig is either identical to the targetConfig or has
   * 'host' short name.
   */
  protected final void createBuildView() throws Exception {
    Preconditions.checkNotNull(masterConfig);
    Preconditions.checkState(getHostConfiguration().equals(getTargetConfiguration())
        || getHostConfiguration().isHostConfiguration(),
        "Host configuration %s is not a host configuration' "
        + "and does not match target configuration %s",
        getHostConfiguration(), getTargetConfiguration());

    skyframeExecutor.handleAnalysisInvalidatingChange();

    view = new BuildViewForTesting(directories, ruleClassProvider, skyframeExecutor, null);
    view.setConfigurationsForTesting(event -> {}, masterConfig);

    view.setArtifactRoots(new PackageRootsNoSymlinkCreation(Root.fromPath(rootDirectory)));
  }

  protected CachingAnalysisEnvironment getTestAnalysisEnvironment() {
    return new CachingAnalysisEnvironment(
        view.getArtifactFactory(),
        actionKeyContext,
        new ActionLookupValue.ActionLookupKey() {
          @Override
          public SkyFunctionName functionName() {
            return null;
          }
        },
        /*isSystemEnv=*/ true,
        /*extendedSanityChecks=*/ false,
        /*allowAnalysisFailures=*/ false,
        reporter,
        skyframeExecutor.getSkyFunctionEnvironmentForTesting(reporter));
  }

  /**
   * Allows access to the prerequisites of a configured target. This is currently used in some tests
   * to reach into the internals of RuleCT for white box testing. In principle, this should not be
   * used; instead tests should only assert on properties of the exposed provider instances and / or
   * the action graph.
   */
  protected final Collection<ConfiguredTarget> getDirectPrerequisites(ConfiguredTarget target)
      throws Exception {
    return view.getDirectPrerequisitesForTesting(reporter, target, masterConfig);
  }

  protected final Collection<ConfiguredTargetAndData> getDirectPrerequisites(
      ConfiguredTargetAndData ctad) throws Exception {
    return view.getConfiguredTargetAndDataDirectPrerequisitesForTesting(
        reporter, ctad, masterConfig);
  }

  protected final ConfiguredTarget getDirectPrerequisite(ConfiguredTarget target, String label)
      throws Exception {
    Label candidateLabel = Label.parseAbsolute(label, ImmutableMap.of());
    for (ConfiguredTarget candidate : getDirectPrerequisites(target)) {
      if (candidate.getLabel().equals(candidateLabel)) {
        return candidate;
      }
    }

    return null;
  }

  protected final ConfiguredTargetAndData getConfiguredTargetAndDataDirectPrerequisite(
      ConfiguredTargetAndData ctad, String label) throws Exception {
    Label candidateLabel = Label.parseAbsolute(label, ImmutableMap.of());
    for (ConfiguredTargetAndData candidate : getDirectPrerequisites(ctad)) {
      if (candidate.getConfiguredTarget().getLabel().equals(candidateLabel)) {
        return candidate;
      }
    }
    return null;
  }

  /**
   * Asserts that two configurations are the same.
   *
   * <p>Historically this meant they contained the same object reference. But with upcoming dynamic
   * configurations that may no longer be true (for example, they may have the same values but not
   * the same {@link Fragment}s. So this method abstracts the "configuration equivalency" checking
   * into one place, where the implementation logic can evolve as needed.
   */
  protected void assertConfigurationsEqual(BuildConfiguration config1, BuildConfiguration config2) {
    // BuildOptions and crosstool files determine a configuration's content. Within the context
    // of these tests only the former actually change.
    assertThat(config2.cloneOptions()).isEqualTo(config1.cloneOptions());
  }

  /**
   * Creates and returns a rule context that is equivalent to the one that was used to create the
   * given configured target.
   */
  protected RuleContext getRuleContext(ConfiguredTarget target) throws Exception {
    return view.getRuleContextForTesting(
        reporter, target, new StubAnalysisEnvironment(), masterConfig);
  }

  protected RuleContext getRuleContext(ConfiguredTarget target,
      AnalysisEnvironment analysisEnvironment) throws Exception {
    return view.getRuleContextForTesting(
        reporter, target, analysisEnvironment, masterConfig);
  }

  /**
   * Creates and returns a rule context to use for Starlark tests that is equivalent to the one that
   * was used to create the given configured target.
   */
  protected RuleContext getRuleContextForSkylark(ConfiguredTarget target) throws Exception {
    // TODO(bazel-team): we need this horrible workaround because CachingAnalysisEnvironment
    // only works with StoredErrorEventListener despite the fact it accepts the interface
    // ErrorEventListener, so it's not possible to create it with reporter.
    // See BuildView.getRuleContextForTesting().
    StoredEventHandler eventHandler = new StoredEventHandler() {
      @Override
      public synchronized void handle(Event e) {
        super.handle(e);
        reporter.handle(e);
      }
    };
    return view.getRuleContextForTesting(target, eventHandler, masterConfig);
  }

  /**
   * Allows access to the prerequisites of a configured target. This is currently used in some tests
   * to reach into the internals of RuleCT for white box testing. In principle, this should not be
   * used; instead tests should only assert on properties of the exposed provider instances and / or
   * the action graph.
   */
  protected List<? extends TransitiveInfoCollection> getPrerequisites(ConfiguredTarget target,
      String attributeName) throws Exception {
    return getRuleContext(target).getConfiguredTargetMap().get(attributeName);
  }

  /**
   * Allows access to the prerequisites of a configured target. This is currently used in some tests
   * to reach into the internals of RuleCT for white box testing. In principle, this should not be
   * used; instead tests should only assert on properties of the exposed provider instances and / or
   * the action graph.
   */
  protected <C extends TransitiveInfoProvider> Iterable<C> getPrerequisites(ConfiguredTarget target,
      String attributeName, Class<C> classType) throws Exception {
    return AnalysisUtils.getProviders(getPrerequisites(target, attributeName), classType);
  }

  /**
   * Allows access to the prerequisites of a configured target. This is currently used in some tests
   * to reach into the internals of RuleCT for white box testing. In principle, this should not be
   * used; instead tests should only assert on properties of the exposed provider instances and / or
   * the action graph.
   */
  protected ImmutableList<Artifact> getPrerequisiteArtifacts(
      ConfiguredTarget target, String attributeName) throws Exception {
    Set<Artifact> result = new LinkedHashSet<>();
    for (FileProvider provider : getPrerequisites(target, attributeName, FileProvider.class)) {
      result.addAll(provider.getFilesToBuild().toList());
    }
    return ImmutableList.copyOf(result);
  }

  protected ActionGraph getActionGraph() {
    return skyframeExecutor.getActionGraph(reporter);
  }

  /** Locates the first parameter file used by the action and returns its command line. */
  @Nullable
  protected final CommandLine paramFileCommandLineForAction(Action action) {
    if (action instanceof SpawnAction) {
      CommandLines commandLines = ((SpawnAction) action).getCommandLines();
      for (CommandLineAndParamFileInfo pair : commandLines.getCommandLines()) {
        if (pair.paramFileInfo != null) {
          return pair.commandLine;
        }
      }
    }
    ParameterFileWriteAction parameterFileWriteAction = paramFileWriteActionForAction(action);
    return parameterFileWriteAction != null ? parameterFileWriteAction.getCommandLine() : null;
  }

  /** Locates the first parameter file used by the action and returns its args. */
  @Nullable
  protected final Iterable<String> paramFileArgsForAction(Action action)
      throws CommandLineExpansionException {
    CommandLine commandLine = paramFileCommandLineForAction(action);
    return commandLine != null ? commandLine.arguments() : null;
  }

  /**
   * Locates the first parameter file used by the action and returns its args.
   *
   * <p>If no param file is used, return the action's arguments.
   */
  @Nullable
  protected final Iterable<String> paramFileArgsOrActionArgs(CommandAction action)
      throws CommandLineExpansionException {
    CommandLine commandLine = paramFileCommandLineForAction(action);
    return commandLine != null ? commandLine.arguments() : action.getArguments();
  }

  /** Locates the first parameter file used by the action and returns its contents. */
  @Nullable
  protected final String paramFileStringContentsForAction(Action action)
      throws CommandLineExpansionException, IOException {
    if (action instanceof SpawnAction) {
      CommandLines commandLines = ((SpawnAction) action).getCommandLines();
      for (CommandLineAndParamFileInfo pair : commandLines.getCommandLines()) {
        if (pair.paramFileInfo != null) {
          ByteArrayOutputStream out = new ByteArrayOutputStream();
          ParameterFile.writeParameterFile(
              out,
              pair.commandLine.arguments(),
              pair.paramFileInfo.getFileType(),
              pair.paramFileInfo.getCharset());
          return new String(out.toByteArray(), pair.paramFileInfo.getCharset());
        }
      }
    }
    ParameterFileWriteAction parameterFileWriteAction = paramFileWriteActionForAction(action);
    return parameterFileWriteAction != null ? parameterFileWriteAction.getStringContents() : null;
  }

  @Nullable
  protected ParameterFileWriteAction paramFileWriteActionForAction(Action action) {
    for (Artifact input : action.getInputs().toList()) {
      if (!(input instanceof SpecialArtifact)) {
        Action generatingAction = getGeneratingAction(input);
        if (generatingAction instanceof ParameterFileWriteAction) {
          return (ParameterFileWriteAction) generatingAction;
        }
      }
    }
    return null;
  }

  protected final ActionAnalysisMetadata getGeneratingActionAnalysisMetadata(Artifact artifact) {
    Preconditions.checkNotNull(artifact);
    ActionAnalysisMetadata actionAnalysisMetadata =
        mutableActionGraph.getGeneratingAction(artifact);

    if (actionAnalysisMetadata == null) {
      if (artifact.isSourceArtifact() || !((DerivedArtifact) artifact).hasGeneratingActionKey()) {
        return null;
      }
      actionAnalysisMetadata = getActionGraph().getGeneratingAction(artifact);
    }

    return actionAnalysisMetadata;
  }

  protected Action getGeneratingAction(ConfiguredTarget target, String outputName) {
    NestedSet<Artifact> filesToBuild = getFilesToBuild(target);
    return getGeneratingAction(outputName, filesToBuild, "filesToBuild");
  }

  private Action getGeneratingAction(
      String outputName, NestedSet<Artifact> filesToBuild, String providerName) {
    Artifact artifact = Iterables.find(filesToBuild.toList(), artifactNamed(outputName), null);
    if (artifact == null) {
      fail(
          String.format(
              "Artifact named '%s' not found in %s (%s)", outputName, providerName, filesToBuild));
    }
    return getGeneratingAction(artifact);
  }

  protected final Action getGeneratingAction(Artifact artifact) {
    ActionAnalysisMetadata action = getGeneratingActionAnalysisMetadata(artifact);

    if (action != null) {
      Preconditions.checkState(
          action instanceof Action, "%s is not a proper Action object", action.prettyPrint());
      return (Action) action;
    } else {
      return null;
    }
  }

  protected Action getGeneratingActionInOutputGroup(
      ConfiguredTarget target, String outputName, String outputGroupName) {
    NestedSet<Artifact> outputGroup =
        OutputGroupInfo.get(target).getOutputGroup(outputGroupName);
    return getGeneratingAction(outputName, outputGroup, "outputGroup/" + outputGroupName);
  }

  /**
   * Returns the SpawnAction that generates an artifact.
   * Implicitly assumes the action is a SpawnAction.
   */
  protected final SpawnAction getGeneratingSpawnAction(Artifact artifact) {
    return (SpawnAction) getGeneratingAction(artifact);
  }

  protected SpawnAction getGeneratingSpawnAction(ConfiguredTarget target, String outputName) {
    return getGeneratingSpawnAction(
        Iterables.find(getFilesToBuild(target).toList(), artifactNamed(outputName)));
  }

  protected final List<String> getGeneratingSpawnActionArgs(Artifact artifact)
      throws CommandLineExpansionException {
    SpawnAction a = getGeneratingSpawnAction(artifact);
    return a.getArguments();
  }

  protected ActionsTestUtil actionsTestUtil() {
    return new ActionsTestUtil(getActionGraph());
  }

  // Get a MutableActionGraph for testing purposes.
  protected MutableActionGraph getMutableActionGraph() {
    return mutableActionGraph;
  }

  /**
   * Returns the ConfiguredTarget for the specified label, configured for the "build" (aka "target")
   * configuration. If the label corresponds to a target with a top-level configuration transition,
   * that transition is applied to the given config in the returned ConfiguredTarget.
   */
  public ConfiguredTarget getConfiguredTarget(String label) throws LabelSyntaxException {
    return getConfiguredTarget(label, targetConfig);
  }

  /**
   * Returns the ConfiguredTarget for the specified label, using the given build configuration. If
   * the label corresponds to a target with a top-level configuration transition, that transition is
   * applied to the given config in the returned ConfiguredTarget.
   */
  protected ConfiguredTarget getConfiguredTarget(String label, BuildConfiguration config)
      throws LabelSyntaxException {
    return getConfiguredTarget(Label.parseAbsolute(label, ImmutableMap.of()), config);
  }

  /**
   * Returns the ConfiguredTarget for the specified label, using the given build configuration. If
   * the label corresponds to a target with a top-level configuration transition, that transition is
   * applied to the given config in the returned ConfiguredTarget.
   *
   * <p>If the evaluation of the SkyKey corresponding to the configured target fails, this method
   * may return null. In that case, use a debugger to inspect the {@link ErrorInfo} for the
   * evaluation, which is produced by the {@link MemoizingEvaluator#getExistingValue} call in {@link
   * SkyframeExecutor#getConfiguredTargetForTesting}. See also b/26382502.
   *
   * @throws AssertionError if the target cannot be transitioned into with the given configuration
   */
  protected ConfiguredTarget getConfiguredTarget(Label label, BuildConfiguration config) {
    try {
      return view.getConfiguredTargetForTesting(
          reporter, BlazeTestUtils.convertLabel(label), config);
    } catch (InvalidConfigurationException
        | StarlarkTransition.TransitionException
        | InterruptedException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Returns a ConfiguredTargetAndData for the specified label, using the given build configuration.
   */
  protected ConfiguredTargetAndData getConfiguredTargetAndData(
      Label label, BuildConfiguration config)
      throws StarlarkTransition.TransitionException, InvalidConfigurationException,
          InterruptedException {
    return view.getConfiguredTargetAndDataForTesting(reporter, label, config);
  }

  /**
   * Returns the ConfiguredTargetAndData for the specified label. If the label corresponds to a
   * target with a top-level configuration transition, that transition is applied to the given
   * config in the ConfiguredTargetAndData's ConfiguredTarget.
   */
  public ConfiguredTargetAndData getConfiguredTargetAndData(String label)
      throws LabelSyntaxException, StarlarkTransition.TransitionException,
          InvalidConfigurationException, InterruptedException {
    return getConfiguredTargetAndData(Label.parseAbsolute(label, ImmutableMap.of()), targetConfig);
  }

  /**
   * Returns the ConfiguredTarget for the specified file label, configured for the "build" (aka
   * "target") configuration.
   */
  protected FileConfiguredTarget getFileConfiguredTarget(String label) throws LabelSyntaxException {
    return (FileConfiguredTarget) getConfiguredTarget(label, targetConfig);
  }

  /**
   * Returns the ConfiguredTarget for the specified label, configured for the "host" configuration.
   */
  protected ConfiguredTarget getHostConfiguredTarget(String label) throws LabelSyntaxException {
    return getConfiguredTarget(label, getHostConfiguration());
  }

  /**
   * Returns the ConfiguredTarget for the specified file label, configured for the "host"
   * configuration.
   */
  protected FileConfiguredTarget getHostFileConfiguredTarget(String label)
      throws LabelSyntaxException {
    return (FileConfiguredTarget) getHostConfiguredTarget(label);
  }

  /**
   * Rewrites the WORKSPACE to have the required boilerplate and the given lines of content.
   *
   * <p>Triggers Skyframe to reinitialize everything.
   */
  public void rewriteWorkspace(String... lines) throws Exception {
    scratch.overwriteFile(
        "WORKSPACE",
        new ImmutableList.Builder<String>()
            .addAll(analysisMock.getWorkspaceContents(mockToolsConfig))
            .addAll(ImmutableList.copyOf(lines))
            .build());

    invalidatePackages();
  }

  /**
   * Create and return a configured scratch rule.
   *
   * @param packageName the package name of the rule.
   * @param ruleName the name of the rule.
   * @param lines the text of the rule.
   * @return the configured target instance for the created rule.
   * @throws IOException
   * @throws Exception
   */
  protected ConfiguredTarget scratchConfiguredTarget(
      String packageName, String ruleName, String... lines) throws IOException, Exception {
    return scratchConfiguredTarget(packageName, ruleName, targetConfig, lines);
  }

  /**
   * Create and return a configured scratch rule.
   *
   * @param packageName the package name of the rule.
   * @param ruleName the name of the rule.
   * @param config the configuration to use to construct the configured rule.
   * @param lines the text of the rule.
   * @return the configured target instance for the created rule.
   * @throws IOException
   * @throws Exception
   */
  protected ConfiguredTarget scratchConfiguredTarget(
      String packageName, String ruleName, BuildConfiguration config, String... lines)
      throws IOException, Exception {
    ConfiguredTargetAndData ctad =
        scratchConfiguredTargetAndData(packageName, ruleName, config, lines);
    return ctad == null ? null : ctad.getConfiguredTarget();
  }

  /**
   * Creates and returns a configured scratch rule and its data.
   *
   * @param packageName the package name of the rule.
   * @param rulename the name of the rule.
   * @param lines the text of the rule.
   * @return the configured tatarget and target instance for the created rule.
   * @throws Exception
   */
  protected ConfiguredTargetAndData scratchConfiguredTargetAndData(
      String packageName, String rulename, String... lines) throws Exception {
    return scratchConfiguredTargetAndData(packageName, rulename, targetConfig, lines);
  }

  /**
   * Creates and returns a configured scratch rule and its data.
   *
   * @param packageName the package name of the rule.
   * @param ruleName the name of the rule.
   * @param config the configuration to use to construct the configured rule.
   * @param lines the text of the rule.
   * @return the ConfiguredTargetAndData instance for the created rule.
   * @throws IOException
   * @throws Exception
   */
  protected ConfiguredTargetAndData scratchConfiguredTargetAndData(
      String packageName, String ruleName, BuildConfiguration config, String... lines)
      throws Exception {
    Target rule = scratchRule(packageName, ruleName, lines);
    return view.getConfiguredTargetAndDataForTesting(reporter, rule.getLabel(), config);
  }

  /**
   * Create and return a scratch rule.
   *
   * @param packageName the package name of the rule.
   * @param ruleName the name of the rule.
   * @param lines the text of the rule.
   * @return the rule instance for the created rule.
   * @throws IOException
   * @throws Exception
   */
  protected Rule scratchRule(String packageName, String ruleName, String... lines)
      throws Exception {
    // Allow to create the BUILD file also in the top package.
    String buildFilePathString = packageName.isEmpty() ? "BUILD" : packageName + "/BUILD";
    if (packageName.equals(LabelConstants.EXTERNAL_PACKAGE_NAME.getPathString())) {
      buildFilePathString = "WORKSPACE";
      scratch.overwriteFile(buildFilePathString, lines);
    } else {
      scratch.file(buildFilePathString, lines);
    }
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter,
        new ModifiedFileSet.Builder().modify(PathFragment.create(buildFilePathString)).build(),
        Root.fromPath(rootDirectory));
    return (Rule) getTarget("//" + packageName + ":" + ruleName);
  }

  /**
   * Check that configuration of the target named 'ruleName' in the
   * specified BUILD file fails with an error message containing
   * 'expectedErrorMessage'.
   *
   * @param packageName the package name of the generated BUILD file
   * @param ruleName the rule name for the rule in the generated BUILD file
   * @param expectedErrorMessage the expected error message.
   * @param lines the text of the rule.
   * @return the found error.
   */
  protected Event checkError(String packageName,
                             String ruleName,
                             String expectedErrorMessage,
                             String... lines) throws Exception {
    eventCollector.clear();
    reporter.removeHandler(failFastHandler); // expect errors
    ConfiguredTarget target = scratchConfiguredTarget(packageName, ruleName, lines);
    if (target != null) {
      assertWithMessage(
              "Rule '" + "//" + packageName + ":" + ruleName + "' did not contain an error")
          .that(view.hasErrors(target))
          .isTrue();
    }
    return assertContainsEvent(expectedErrorMessage);
  }

  /**
   * Check that configuration of the target named 'label' fails with an error message containing
   * 'expectedErrorMessage'.
   *
   * @param label the target name to test
   * @param expectedErrorMessage the expected error message.
   * @return the found error.
   */
  protected Event checkError(String label, String expectedErrorMessage) throws Exception {
    eventCollector.clear();
    reporter.removeHandler(failFastHandler); // expect errors
    ConfiguredTarget target = getConfiguredTarget(label);
    if (target != null) {
      assertWithMessage("Rule '" + label + "' did not contain an error")
          .that(view.hasErrors(target))
          .isTrue();
    }
    return assertContainsEvent(expectedErrorMessage);
  }

  /**
   * Checks whether loading the given target results in the specified error message.
   *
   * @param target the name of the target.
   * @param expectedErrorMessage the expected error message.
   */
  protected void checkLoadingPhaseError(String target, String expectedErrorMessage) {
    reporter.removeHandler(failFastHandler);
    // The error happens during the loading of the Starlark file so checkError doesn't work here
    assertThrows(Exception.class, () -> getTarget(target));
    assertContainsEvent(expectedErrorMessage);
  }

  /**
   * Check that configuration of the target named 'ruleName' in the
   * specified BUILD file reports a warning message ending in
   * 'expectedWarningMessage', and that no errors were reported.
   *
   * @param packageName the package name of the generated BUILD file
   * @param ruleName the rule name for the rule in the generated BUILD file
   * @param expectedWarningMessage the expected warning message.
   * @param lines the text of the rule.
   * @return the found error.
   */
  protected Event checkWarning(String packageName,
                               String ruleName,
                               String expectedWarningMessage,
                               String... lines) throws Exception {
    eventCollector.clear();
    ConfiguredTarget target = scratchConfiguredTarget(packageName, ruleName,
        lines);
    assertWithMessage("Rule '" + "//" + packageName + ":" + ruleName + "' did contain an error")
        .that(view.hasErrors(target))
        .isFalse();
    return assertContainsEvent(expectedWarningMessage);
  }

  /**
   * Given a collection of Artifacts, returns a corresponding set of strings of the form "[root]
   * [relpath]", such as "bin x/libx.a". Such strings make assertions easier to write.
   *
   * <p>The returned set preserves the order of the input.
   */
  protected Set<String> artifactsToStrings(NestedSet<? extends Artifact> artifacts) {
    return artifactsToStrings(artifacts.toList());
  }

  /**
   * Given a collection of Artifacts, returns a corresponding set of strings of the form "[root]
   * [relpath]", such as "bin x/libx.a". Such strings make assertions easier to write.
   *
   * <p>The returned set preserves the order of the input.
   */
  protected Set<String> artifactsToStrings(Iterable<? extends Artifact> artifacts) {
    return AnalysisTestUtil.artifactsToStrings(masterConfig, artifacts);
  }

  /**
   * Asserts that targetName's outputs are exactly expectedOuts.
   *
   * @param targetName The label of a rule.
   * @param expectedOuts The labels of the expected outputs of the rule.
   */
  protected void assertOuts(String targetName, String... expectedOuts) throws Exception {
    Rule ruleTarget = (Rule) getTarget(targetName);
    for (String expectedOut : expectedOuts) {
      Target outTarget = getTarget(expectedOut);
      if (!(outTarget instanceof OutputFile)) {
        fail("Target " + outTarget + " is not an output");
        assertThat(((OutputFile) outTarget).getGeneratingRule()).isSameInstanceAs(ruleTarget);
        // This ensures that the output artifact is wired up in the action graph
        getConfiguredTarget(expectedOut);
      }
    }

    Collection<OutputFile> outs = ruleTarget.getOutputFiles();
    assertWithMessage("Mismatched outputs: " + outs)
        .that(outs.size())
        .isEqualTo(expectedOuts.length);
  }

  /**
   * Asserts that there exists a configured target file for the given label.
   */
  protected void assertConfiguredTargetExists(String label) throws Exception {
    assertThat(getFileConfiguredTarget(label)).isNotNull();
  }

  /**
   * Assert that the first label and the second label are both generated
   * by the same command.
   */
  protected void assertSameGeneratingAction(String labelA, String labelB)
      throws Exception {
    assertWithMessage("Action for " + labelA + " did not match " + labelB)
        .that(getGeneratingActionForLabel(labelB))
        .isSameInstanceAs(getGeneratingActionForLabel(labelA));
  }

  protected Artifact getSourceArtifact(PathFragment rootRelativePath, Root root) {
    return view.getArtifactFactory().getSourceArtifact(rootRelativePath, root);
  }

  protected Artifact getSourceArtifact(String name) {
    return getSourceArtifact(PathFragment.create(name), Root.fromPath(rootDirectory));
  }

  /**
   * Gets a derived artifact, creating it if necessary. {@code ArtifactOwner} should be a genuine
   * {@link ConfiguredTargetKey} corresponding to a {@link ConfiguredTarget}. If called from a test
   * that does not exercise the analysis phase, the convenience methods {@link
   * #getBinArtifactWithNoOwner} or {@link #getGenfilesArtifactWithNoOwner} should be used instead.
   */
  protected final Artifact.DerivedArtifact getDerivedArtifact(
      PathFragment rootRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    if ((owner instanceof ActionLookupValue.ActionLookupKey)) {
      ActionLookupValue actionLookupValue;
      try {
        actionLookupValue =
            (ActionLookupValue)
                skyframeExecutor.getEvaluatorForTesting().getExistingValue((SkyKey) owner);
      } catch (InterruptedException e) {
        throw new IllegalStateException(e);
      }
      if (actionLookupValue != null) {
        for (ActionAnalysisMetadata action : actionLookupValue.getActions()) {
          for (Artifact output : action.getOutputs()) {
            if (output.getRootRelativePath().equals(rootRelativePath)
                && output.getRoot().equals(root)) {
              return (Artifact.DerivedArtifact) output;
            }
          }
        }
      }
    }
    // Fall back: some tests don't actually need an artifact with an owner.
    return view.getArtifactFactory().getDerivedArtifact(rootRelativePath, root, owner);
  }

  /**
   * Gets a Tree Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getBinDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.o".
   */
  protected final Artifact getTreeArtifact(String packageRelativePath, ConfiguredTarget owner) {
    ActionLookupValue.ActionLookupKey actionLookupKey =
        ConfiguredTargetKey.of(owner, owner.getConfigurationKey(), /*isHostConfiguration=*/ false);
    return getDerivedArtifact(
        owner.getLabel().getPackageFragment().getRelative(packageRelativePath),
        getConfiguration(owner).getBinDirectory(RepositoryName.MAIN),
        actionLookupKey);
  }

  /**
   * Gets a derived Artifact for testing with path of the form
   * root/owner.getPackageFragment()/packageRelativePath.
   *
   * @see #getDerivedArtifact(PathFragment, ArtifactRoot, ArtifactOwner)
   */
  private Artifact getPackageRelativeDerivedArtifact(
      String packageRelativePath, ArtifactRoot root, ArtifactOwner owner) {
    return getDerivedArtifact(
        owner.getLabel().getPackageFragment().getRelative(packageRelativePath),
        root, owner);
  }

  /** Returns the input {@link Artifact}s to the given {@link Action} with the given exec paths. */
  protected List<Artifact> getInputs(Action owner, Collection<String> execPaths) {
    Set<String> expectedPaths = new HashSet<>(execPaths);
    List<Artifact> result = new ArrayList<>();
    for (Artifact output : owner.getInputs().toList()) {
      if (expectedPaths.remove(output.getExecPathString())) {
        result.add(output);
      }
    }
    assertWithMessage("expected paths not found in: %s", Artifact.asExecPaths(owner.getInputs()))
        .that(expectedPaths)
        .isEmpty();
    return result;
  }

  /**
   * Gets a derived Artifact for testing in the {@link BuildConfiguration#getBinDirectory}. This
   * method should only be used for tests that do no analysis, and so there is no ConfiguredTarget
   * to own this artifact. If the test runs the analysis phase, {@link #getBinArtifact(String,
   * ConfiguredTarget)} or its convenience methods should be used instead.
   */
  protected Artifact.DerivedArtifact getBinArtifactWithNoOwner(String rootRelativePath) {
    return getDerivedArtifact(PathFragment.create(rootRelativePath),
        targetConfig.getBinDirectory(RepositoryName.MAIN),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getBinDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.o".
   */
  protected final Artifact getBinArtifact(String packageRelativePath, ConfiguredTarget owner) {
    return getPackageRelativeDerivedArtifact(
        packageRelativePath,
        getConfiguration(owner).getBinDirectory(RepositoryName.MAIN),
        ConfiguredTargetKey.of(
            owner, skyframeExecutor.getConfiguration(reporter, owner.getConfigurationKey())));
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getBinDirectory} corresponding to the package of {@code owner}, where the
   * given artifact belongs to the given ConfiguredTarget together with the given Aspect. So to
   * specify a file foo/foo.o owned by target //foo:foo with an aspect from FooAspect, {@code
   * packageRelativePath} should just be "foo.o", and aspectOfOwner should be FooAspect.class. This
   * method is necessary when an Aspect of the target, not the target itself, is creating an
   * Artifact.
   */
  protected Artifact getBinArtifact(
      String packageRelativePath, ConfiguredTarget owner, AspectClass creatingAspectFactory) {
    return getBinArtifact(
        packageRelativePath, owner, creatingAspectFactory, AspectParameters.EMPTY);
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getBinDirectory} corresponding to the package of {@code owner}, where the
   * given artifact belongs to the given ConfiguredTarget together with the given Aspect. So to
   * specify a file foo/foo.o owned by target //foo:foo with an aspect from FooAspect, {@code
   * packageRelativePath} should just be "foo.o", and aspectOfOwner should be FooAspect.class. This
   * method is necessary when an Aspect of the target, not the target itself, is creating an
   * Artifact.
   */
  protected Artifact getBinArtifact(
      String packageRelativePath,
      ConfiguredTarget owner,
      AspectClass creatingAspectFactory,
      AspectParameters parameters) {
    return getPackageRelativeDerivedArtifact(
        packageRelativePath,
        getConfiguration(owner).getBinDirectory(RepositoryName.MAIN),
        (AspectKey)
            AspectValueKey.createAspectKey(
                    owner.getLabel(),
                    getConfiguration(owner),
                    new AspectDescriptor(creatingAspectFactory, parameters),
                    getConfiguration(owner))
                .argument());
  }

  /**
   * Gets a derived Artifact for testing in the {@link BuildConfiguration#getGenfilesDirectory}.
   * This method should only be used for tests that do no analysis, and so there is no
   * ConfiguredTarget to own this artifact. If the test runs the analysis phase, {@link
   * #getGenfilesArtifact(String, ConfiguredTarget)} or its convenience methods should be used
   * instead.
   */
  protected Artifact getGenfilesArtifactWithNoOwner(String rootRelativePath) {
    return getDerivedArtifact(PathFragment.create(rootRelativePath),
        targetConfig.getGenfilesDirectory(RepositoryName.MAIN),
        ActionsTestUtil.NULL_ARTIFACT_OWNER);
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getGenfilesDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.o".
   */
  protected Artifact getGenfilesArtifact(String packageRelativePath, String owner) {
    BuildConfiguration config = getConfiguration(owner);
    return getGenfilesArtifact(
        packageRelativePath, ConfiguredTargetKey.of(makeLabel(owner), config), config);
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getGenfilesDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.o".
   */
  protected Artifact getGenfilesArtifact(String packageRelativePath, ConfiguredTarget owner) {
    BuildConfiguration configuration =
        skyframeExecutor.getConfiguration(reporter, owner.getConfigurationKey());
    ConfiguredTargetKey configKey = ConfiguredTargetKey.of(owner, configuration);
    return getGenfilesArtifact(packageRelativePath, configKey, configuration);
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getGenfilesDirectory} corresponding to the package of {@code owner}, where
   * the given artifact belongs to the given ConfiguredTarget together with the given Aspect. So to
   * specify a file foo/foo.o owned by target //foo:foo with an apsect from FooAspect, {@code
   * packageRelativePath} should just be "foo.o", and aspectOfOwner should be FooAspect.class. This
   * method is necessary when an Apsect of the target, not the target itself, is creating an
   * Artifact.
   */
  protected Artifact getGenfilesArtifact(
      String packageRelativePath, ConfiguredTarget owner, NativeAspectClass creatingAspectFactory) {
    return getGenfilesArtifact(
        packageRelativePath, owner, creatingAspectFactory, AspectParameters.EMPTY);
  }

  protected Artifact getGenfilesArtifact(
      String packageRelativePath,
      ConfiguredTarget owner,
      NativeAspectClass creatingAspectFactory,
      AspectParameters params) {
    return getPackageRelativeDerivedArtifact(
        packageRelativePath,
        getConfiguration(owner)
            .getGenfilesDirectory(owner.getLabel().getPackageIdentifier().getRepository()),
        getOwnerForAspect(owner, creatingAspectFactory, params));
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getGenfilesDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.o".
   */
  private Artifact getGenfilesArtifact(
      String packageRelativePath, ArtifactOwner owner, BuildConfiguration config) {
    return getPackageRelativeDerivedArtifact(
        packageRelativePath, config.getGenfilesDirectory(RepositoryName.MAIN), owner);
  }

  protected AspectKey getOwnerForAspect(
      ConfiguredTarget owner, NativeAspectClass creatingAspectFactory, AspectParameters params) {
    return (AspectKey)
        AspectValueKey.createAspectKey(
                owner.getLabel(),
                getConfiguration(owner),
                new AspectDescriptor(creatingAspectFactory, params),
                getConfiguration(owner))
            .argument();
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getIncludeDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.h".
   */
  protected Artifact getIncludeArtifact(String packageRelativePath, String owner) {
    return getIncludeArtifact(packageRelativePath, makeConfiguredTargetKey(owner));
  }

  /**
   * Gets a derived Artifact for testing in the subdirectory of the {@link
   * BuildConfiguration#getIncludeDirectory} corresponding to the package of {@code owner}. So to
   * specify a file foo/foo.o owned by target //foo:foo, {@code packageRelativePath} should just be
   * "foo.h".
   */
  protected Artifact getIncludeArtifact(String packageRelativePath, ArtifactOwner owner) {
    return getPackageRelativeDerivedArtifact(packageRelativePath,
        targetConfig.getIncludeDirectory(owner.getLabel().getPackageIdentifier().getRepository()),
        owner);
  }

  /**
   * @return a shared artifact at the binary-root relative path {@code rootRelativePath} owned by
   *         {@code owner}.
   *
   * @param rootRelativePath the binary-root relative path of the artifact.
   * @param owner the artifact's owner.
   */
  protected Artifact getSharedArtifact(String rootRelativePath, ConfiguredTarget owner) {
    return getDerivedArtifact(
        PathFragment.create(rootRelativePath),
        targetConfig.getBinDirectory(RepositoryName.MAIN),
        ConfiguredTargetKey.of(
            owner, skyframeExecutor.getConfiguration(reporter, owner.getConfigurationKey())));
  }

  protected Action getGeneratingActionForLabel(String label) throws Exception {
    return getGeneratingAction(getFileConfiguredTarget(label).getArtifact());
  }

  /**
   * Strips the C++-contributed prefix out of an output path when tests are run with trimmed
   * configurations. e.g. turns "bazel-out/gcc-X-glibc-Y-k8-fastbuild/ to "bazel-out/fastbuild/".
   *
   * <p>This should be used for targets use configurations with C++ fragments.
   */
  protected String stripCppPrefixForTrimmedConfigs(String outputPath) {
    return targetConfig.trimConfigurations()
        ? AnalysisTestUtil.OUTPUT_PATH_CPP_PREFIX_PATTERN.matcher(outputPath).replaceFirst("")
        : outputPath;
  }

  protected String fileName(Artifact artifact) {
    return artifact.getExecPathString();
  }

  protected String fileName(FileConfiguredTarget target) {
    return fileName(target.getArtifact());
  }

  protected String fileName(String name) throws Exception {
    return fileName(getFileConfiguredTarget(name));
  }

  protected Path getOutputPath() {
    return directories.getOutputPath(ruleClassProvider.getRunfilesPrefix());
  }

  protected String getRelativeOutputPath() {
    return directories.getRelativeOutputPath();
  }

  /**
   * Verifies whether the rule checks the 'srcs' attribute validity.
   *
   * <p>At the call site it expects the {@code packageName} to contain:
   * <ol>
   *   <li>{@code :gvalid} - genrule that outputs a valid file</li>
   *   <li>{@code :ginvalid} - genrule that outputs an invalid file</li>
   *   <li>{@code :gmix} - genrule that outputs a mix of valid and invalid
   *       files</li>
   *   <li>{@code :valid} - rule of type {@code ruleType} that has a valid
   *       file, {@code :gvalid} and {@code :gmix} in the srcs</li>
   *   <li>{@code :invalid} - rule of type {@code ruleType} that has an invalid
   *       file, {@code :ginvalid} in the srcs</li>
   *   <li>{@code :mix} - rule of type {@code ruleType} that has a valid and an
   *       invalid file in the srcs</li>
   * </ol>
   *
   * @param packageName the package where the rules under test are located
   * @param ruleType rules under test types
   * @param expectedTypes expected file types
   */
  protected void assertSrcsValidityForRuleType(String packageName, String ruleType,
      String expectedTypes) throws Exception {
    reporter.removeHandler(failFastHandler);
    String descriptionSingle = ruleType + " srcs file (expected " + expectedTypes + ")";
    String descriptionPlural = ruleType + " srcs files (expected " + expectedTypes + ")";
    String descriptionPluralFile = "(expected " + expectedTypes + ")";
    assertSrcsValidity(ruleType, packageName + ":valid", false,
        "need at least one " + descriptionSingle,
        "'" + packageName + ":gvalid' does not produce any " + descriptionPlural,
        "'" + packageName + ":gmix' does not produce any " + descriptionPlural);
    assertSrcsValidity(ruleType, packageName + ":invalid", true,
        "source file '" + packageName + ":a.foo' is misplaced here " + descriptionPluralFile,
        "'" + packageName + ":ginvalid' does not produce any " + descriptionPlural);
    assertSrcsValidity(ruleType, packageName + ":mix", true,
        "'" + packageName + ":a.foo' does not produce any " + descriptionPlural);
  }

  protected void assertSrcsValidity(String ruleType, String targetName, boolean expectedError,
      String... expectedMessages) throws Exception{
    ConfiguredTarget target = getConfiguredTarget(targetName);
    if (expectedError) {
      assertThat(view.hasErrors(target)).isTrue();
      for (String expectedMessage : expectedMessages) {
        String message = "in srcs attribute of " + ruleType + " rule " + targetName + ": "
            + expectedMessage;
        assertContainsEvent(message);
      }
    } else {
      assertThat(view.hasErrors(target)).isFalse();
      for (String expectedMessage : expectedMessages) {
        String message = "in srcs attribute of " + ruleType + " rule " + target.getLabel() + ": "
            + expectedMessage;
        assertDoesNotContainEvent(message);
      }
    }
  }

  protected static ConfiguredAttributeMapper getMapperFromConfiguredTargetAndTarget(
      ConfiguredTargetAndData ctad) {
    return ConfiguredAttributeMapper.of(
        (Rule) ctad.getTarget(),
        ((RuleConfiguredTarget) ctad.getConfiguredTarget()).getConfigConditions());
  }

  public static Label makeLabel(String label) {
    try {
      return Label.parseAbsolute(label, ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  private ConfiguredTargetKey makeConfiguredTargetKey(String label) {
    return ConfiguredTargetKey.of(makeLabel(label), getConfiguration(label));
  }

  protected static List<String> actionInputsToPaths(NestedSet<? extends ActionInput> actionInputs) {
    return ImmutableList.copyOf(
        Iterables.transform(
            actionInputs.toList(), (actionInput) -> actionInput.getExecPathString()));
  }

  /**
   * Utility method for asserting that the contents of one collection are the
   * same as those in a second plus some set of common elements.
   */
  protected void assertSameContentsWithCommonElements(Iterable<String> artifacts,
      String[] expectedInputs, Iterable<String> common) {
    assertThat(Iterables.concat(Lists.newArrayList(expectedInputs), common))
        .containsExactlyElementsIn(artifacts);
  }

  /**
   * Utility method for asserting that a list contains the elements of a
   * sublist. This is useful for checking that a list of arguments contains a
   * particular set of arguments.
   */
  protected void assertContainsSublist(List<String> list, List<String> sublist) {
    assertContainsSublist(null, list, sublist);
  }

  /**
   * Utility method for asserting that a list contains the elements of a
   * sublist. This is useful for checking that a list of arguments contains a
   * particular set of arguments.
   */
  protected void assertContainsSublist(String message, List<String> list, List<String> sublist) {
    if (Collections.indexOfSubList(list, sublist) == -1) {
      fail((message == null ? "" : (message + ' '))
          + "expected: <" + list + "> to contain sublist: <" + sublist + ">");
    }
  }

  protected void assertContainsSelfEdgeEvent(String label) {
    assertContainsEvent(label + " [self-edge]");
  }

  protected NestedSet<Artifact> collectRunfiles(ConfiguredTarget target) {
    RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);
    if (runfilesProvider != null) {
      return runfilesProvider.getDefaultRunfiles().getAllArtifacts();
    } else {
      return Runfiles.EMPTY.getAllArtifacts();
    }
  }

  protected NestedSet<Artifact> getFilesToBuild(TransitiveInfoCollection target) {
    return target.getProvider(FileProvider.class).getFilesToBuild();
  }

  /**
   * Returns all extra actions for that target (no transitive actions), no duplicate actions.
   */
  protected ImmutableList<Action> getExtraActionActions(ConfiguredTarget target) {
    LinkedHashSet<Action> result = new LinkedHashSet<>();
    for (Artifact artifact : getExtraActionArtifacts(target).toList()) {
      result.add(getGeneratingAction(artifact));
    }
    return ImmutableList.copyOf(result);
  }

  /**
   * Returns all extra actions for that target (including transitive actions).
   */
  protected ImmutableList<ExtraAction> getTransitiveExtraActionActions(ConfiguredTarget target) {
    ImmutableList.Builder<ExtraAction> result = new ImmutableList.Builder<>();
    for (Artifact artifact :
        target
            .getProvider(ExtraActionArtifactsProvider.class)
            .getTransitiveExtraActionArtifacts()
            .toList()) {
      Action action = getGeneratingAction(artifact);
      if (action instanceof ExtraAction) {
        result.add((ExtraAction) action);
      }
    }
    return result.build();
  }

  protected ImmutableList<Action> getFilesToBuildActions(ConfiguredTarget target) {
    List<Action> result = new ArrayList<>();
    for (Artifact artifact : getFilesToBuild(target).toList()) {
      Action action = getGeneratingAction(artifact);
      if (action != null) {
        result.add(action);
      }
    }
    return ImmutableList.copyOf(result);
  }

  protected NestedSet<Artifact> getOutputGroup(
      TransitiveInfoCollection target, String outputGroup) {
    OutputGroupInfo provider = OutputGroupInfo.get(target);
    return provider == null
        ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
        : provider.getOutputGroup(outputGroup);
  }

  protected NestedSet<Artifact.DerivedArtifact> getExtraActionArtifacts(ConfiguredTarget target) {
    return target.getProvider(ExtraActionArtifactsProvider.class).getExtraActionArtifacts();
  }

  protected Artifact getExecutable(String label) throws Exception {
    return getConfiguredTarget(label).getProvider(FilesToRunProvider.class).getExecutable();
  }

  protected Artifact getExecutable(TransitiveInfoCollection target) {
    return target.getProvider(FilesToRunProvider.class).getExecutable();
  }

  protected NestedSet<Artifact> getFilesToRun(TransitiveInfoCollection target) {
    return target.getProvider(FilesToRunProvider.class).getFilesToRun();
  }

  protected NestedSet<Artifact> getFilesToRun(Label label) throws Exception {
    return getConfiguredTarget(label, targetConfig)
        .getProvider(FilesToRunProvider.class).getFilesToRun();
  }

  protected NestedSet<Artifact> getFilesToRun(String label) throws Exception {
    return getConfiguredTarget(label).getProvider(FilesToRunProvider.class).getFilesToRun();
  }

  protected RunfilesSupport getRunfilesSupport(String label) throws Exception {
    return getConfiguredTarget(label).getProvider(FilesToRunProvider.class).getRunfilesSupport();
  }

  protected RunfilesSupport getRunfilesSupport(TransitiveInfoCollection target) {
    return target.getProvider(FilesToRunProvider.class).getRunfilesSupport();
  }

  protected static Runfiles getDefaultRunfiles(ConfiguredTarget target) {
    return target.getProvider(RunfilesProvider.class).getDefaultRunfiles();
  }

  protected static Runfiles getDataRunfiles(ConfiguredTarget target) {
    return target.getProvider(RunfilesProvider.class).getDataRunfiles();
  }

  protected BuildConfiguration getTargetConfiguration() {
    return Iterables.getOnlyElement(masterConfig.getTargetConfigurations());
  }

  protected BuildConfiguration getHostConfiguration() {
    return masterConfig.getHostConfiguration();
  }

  /**
   * Returns the configuration created by applying the given transition to the source configuration.
   *
   * @throws AssertionError if the transition couldn't be evaluated
   */
  protected BuildConfiguration getConfiguration(
      BuildConfiguration fromConfig, PatchTransition transition) throws InterruptedException {
    if (transition == NoTransition.INSTANCE) {
      return fromConfig;
    } else if (transition == NullTransition.INSTANCE) {
      return null;
    } else {
      try {
        return skyframeExecutor.getConfigurationForTesting(
            reporter,
            fromConfig.fragmentClasses(),
            transition.patch(fromConfig.getOptions(), eventCollector));
      } catch (OptionsParsingException | InvalidConfigurationException e) {
        throw new AssertionError(e);
      }
    }
  }

  private BuildConfiguration getConfiguration(String label) {
    BuildConfiguration config;
    try {
      config = getConfiguration(getConfiguredTarget(label));
      config = view.getConfigurationForTesting(getTarget(label), config, reporter);
    } catch (LabelSyntaxException e) {
      throw new IllegalArgumentException(e);
    } catch (Exception e) {
      // TODO(b/36585204): Clean this up
      throw new RuntimeException(e);
    }
    return config;
  }

  protected final BuildConfiguration getConfiguration(ConfiguredTarget ct) {
    return skyframeExecutor.getConfiguration(reporter, ct.getConfigurationKey());
  }

  /** Returns an attribute value retriever for the given rule for the target configuration. */
  protected AttributeMap attributes(RuleConfiguredTarget ct) {
    ConfiguredTargetAndData ctad;
    try {
      ctad = getConfiguredTargetAndData(ct.getLabel().toString());
    } catch (LabelSyntaxException
        | StarlarkTransition.TransitionException
        | InvalidConfigurationException
        | InterruptedException e) {
      throw new RuntimeException(e);
    }
    return getMapperFromConfiguredTargetAndTarget(ctad);
  }

  protected AttributeMap attributes(ConfiguredTarget rule) {
    return attributes((RuleConfiguredTarget) rule);
  }

  protected void useLoadingOptions(String... options) throws OptionsParsingException {
    customLoadingOptions = Options.parse(LoadingOptions.class, options).getOptions();
  }

  protected AnalysisResult update(List<String> targets,
      boolean keepGoing,
      int loadingPhaseThreads,
      boolean doAnalysis,
      EventBus eventBus) throws Exception {
    return update(
        targets, ImmutableList.<String>of(), keepGoing, loadingPhaseThreads, doAnalysis, eventBus);
  }

  protected AnalysisResult update(
      List<String> targets,
      List<String> aspects,
      boolean keepGoing,
      int loadingPhaseThreads,
      boolean doAnalysis,
      EventBus eventBus)
      throws Exception {

    LoadingOptions loadingOptions =
        customLoadingOptions == null
            ? Options.getDefaults(LoadingOptions.class)
            : customLoadingOptions;

    AnalysisOptions viewOptions = Options.getDefaults(AnalysisOptions.class);

    TargetPatternPhaseValue loadingResult =
        skyframeExecutor.loadTargetPatternsWithFilters(
            reporter,
            targets,
            PathFragment.EMPTY_FRAGMENT,
            loadingOptions,
            loadingPhaseThreads,
            keepGoing,
            /*determineTests=*/ false);
    if (!doAnalysis) {
      // TODO(bazel-team): What's supposed to happen in this case?
      return null;
    }
    return view.update(
        loadingResult,
        targetConfig.getOptions(),
        /* multiCpu= */ ImmutableSet.of(),
        aspects,
        viewOptions,
        keepGoing,
        loadingPhaseThreads,
        AnalysisTestUtil.TOP_LEVEL_ARTIFACT_CONTEXT,
        reporter,
        eventBus);
  }

  protected static Predicate<Artifact> artifactNamed(final String name) {
    return new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact input) {
        return name.equals(input.prettyPrint());
      }
    };
  }

  /**
   * Utility method for tests. Converts an array of strings into a set of labels.
   *
   * @param strings the set of strings to be converted to labels.
   * @throws LabelSyntaxException if there are any syntax errors in the strings.
   */
  public static Set<Label> asLabelSet(String... strings) throws LabelSyntaxException {
    return asLabelSet(ImmutableList.copyOf(strings));
  }

  /**
   * Utility method for tests. Converts an array of strings into a set of labels.
   *
   * @param strings the set of strings to be converted to labels.
   * @throws LabelSyntaxException if there are any syntax errors in the strings.
   */
  public static Set<Label> asLabelSet(Iterable<String> strings) throws LabelSyntaxException {
    Set<Label> result = Sets.newTreeSet();
    for (String s : strings) {
      result.add(Label.parseAbsolute(s, ImmutableMap.of()));
    }
    return result;
  }

  protected String getErrorMsgNoGoodFiles(String attrName, String ruleType, String ruleName,
      String depRuleName) {
    return "in " + attrName + " attribute of " + ruleType + " rule " + ruleName + ": '"
        + depRuleName + "' does not produce any " + ruleType + " " + attrName + " files";
  }

  protected String getErrorMsgMisplacedFiles(String attrName, String ruleType, String ruleName,
      String fileName) {
    return "in " + attrName + " attribute of " + ruleType + " rule " + ruleName + ": source file '"
        + fileName + "' is misplaced here";
  }

  protected String getErrorNonExistingTarget(String attrName, String ruleType, String ruleName,
      String targetName) {
    return "in " + attrName + " attribute of " + ruleType + " rule " + ruleName + ": target '"
        + targetName + "' does not exist";
  }

  protected String getErrorNonExistingRule(String attrName, String ruleType, String ruleName,
      String targetName) {
    return "in " + attrName + " attribute of " + ruleType + " rule " + ruleName + ": rule '"
        + targetName + "' does not exist";
  }

  protected String getErrorMsgMisplacedRules(String attrName, String ruleType, String ruleName,
      String depRuleType, String depRuleName) {
    return "in " + attrName + " attribute of " + ruleType + " rule " + ruleName + ": "
        + depRuleType + " rule '" + depRuleName + "' is misplaced here";
  }

  protected String getErrorMsgNonEmptyList(String attrName, String ruleType, String ruleName) {
    return "in " + attrName + " attribute of " + ruleType + " rule " + ruleName + ": attribute "
        + "must be non empty";
  }

  protected String getErrorMsgMandatoryMissing(String attrName, String ruleType) {
    return "missing value for mandatory attribute '" + attrName + "' in '" + ruleType + "' rule";
  }

  protected String getErrorMsgWrongAttributeValue(String value, String... expected) {
    return String.format("has to be one of %s instead of '%s'",
        StringUtil.joinEnglishList(ImmutableSet.copyOf(expected), "or", "'"), value);
  }

  protected String getErrorMsgMandatoryProviderMissing(String offendingRule, String providerName) {
    return String.format("'%s' does not have mandatory providers: '%s'",
        offendingRule, providerName);
  }

  /**
   * Utility method for tests that result in errors early during
   * package loading. Given the name of the package for the test,
   * and the rules for the build file, create a scratch file, load
   * the build file, and produce the package.
   * @param packageName the name of the package for the build file
   * @param lines the rules for the build file as an array of strings
   * @return the loaded package from the populated package cache
   * @throws Exception if there is an error creating the temporary files
   *    for the test.
   */
  protected com.google.devtools.build.lib.packages.Package createScratchPackageForImplicitCycle(
      String packageName, String... lines) throws Exception {
    eventCollector.clear();
    reporter.removeHandler(failFastHandler);
    scratch.file("" + packageName + "/BUILD", lines);
    return getPackageManager()
        .getPackage(reporter, PackageIdentifier.createInMainRepo(packageName));
  }

  /**
   * A stub analysis environment.
   */
  protected class StubAnalysisEnvironment implements AnalysisEnvironment {

    @Override
    public void registerAction(ActionAnalysisMetadata... action) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean hasErrors() {
      return false;
    }

    @Override
    public Artifact getConstantMetadataArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      throw new UnsupportedOperationException();
    }

    @Override
    public SpecialArtifact getTreeArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      throw new UnsupportedOperationException();
    }

    @Override
    public SpecialArtifact getSymlinkArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getSourceArtifactForNinjaBuild(PathFragment execPath, Root root) {
      throw new UnsupportedOperationException();
    }

    @Override
    public ExtendedEventHandler getEventHandler() {
      return reporter;
    }

    @Override
    public MiddlemanFactory getMiddlemanFactory() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Action getLocalGeneratingAction(Artifact artifact) {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableList<ActionAnalysisMetadata> getRegisteredActions() {
      throw new UnsupportedOperationException();
    }

    @Override
    public SkyFunction.Environment getSkyframeEnv() {
      throw new UnsupportedOperationException();
    }

    @Override
    public StarlarkSemantics getSkylarkSemantics() {
      return starlarkSemanticsOptions.toSkylarkSemantics();
    }

    @Override
    public Artifact getFilesetArtifact(PathFragment rootRelativePath, ArtifactRoot root) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact.DerivedArtifact getDerivedArtifact(
        PathFragment rootRelativePath, ArtifactRoot root) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact.DerivedArtifact getDerivedArtifact(
        PathFragment rootRelativePath, ArtifactRoot root, boolean contentBasedPath) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getStableWorkspaceStatusArtifact() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Artifact getVolatileWorkspaceStatusArtifact() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableList<Artifact> getBuildInfo(
        boolean stamp, BuildInfoKey key, BuildConfiguration config) {
      throw new UnsupportedOperationException();
    }

    @Override
    public ActionLookupValue.ActionLookupKey getOwner() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableSet<Artifact> getOrphanArtifacts() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ImmutableSet<Artifact> getTreeArtifactsConflictingWithFiles() {
      throw new UnsupportedOperationException();
    }

    @Override
    public ActionKeyContext getActionKeyContext() {
      return actionKeyContext;
    }
  }

  protected Iterable<String> baselineCoverageArtifactBasenames(ConfiguredTarget target)
      throws Exception {
    ImmutableList.Builder<String> basenames = ImmutableList.builder();
    for (Artifact baselineCoverage :
        target
            .get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR)
            .getBaselineCoverageArtifacts()
            .toList()) {
      BaselineCoverageAction baselineAction =
          (BaselineCoverageAction) getGeneratingAction(baselineCoverage);
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      baselineAction.newDeterministicWriter(ActionsTestUtil.createContext(reporter))
          .writeOutputFile(bytes);

      for (String line : new String(bytes.toByteArray(), StandardCharsets.UTF_8).split("\n")) {
        if (line.startsWith("SF:")) {
          String basename = line.substring(line.lastIndexOf('/') + 1);
          basenames.add(basename);
        }
      }
    }
    return basenames.build();
  }

  /**
   * Finds an artifact in the transitive closure of a set of other artifacts by following a path
   * based on artifact name suffixes.
   *
   * <p>This selects the first artifact in the input set that matches the first suffix, then selects
   * the first artifact in the inputs of its generating action that matches the second suffix etc.,
   * and repeats this until the supplied suffixes run out.
   */
  protected Artifact artifactByPath(NestedSet<Artifact> artifacts, String... suffixes) {
    return artifactByPath(artifacts.toList(), suffixes);
  }

  /**
   * Finds an artifact in the transitive closure of a set of other artifacts by following a path
   * based on artifact name suffixes.
   *
   * <p>This selects the first artifact in the input set that matches the first suffix, then selects
   * the first artifact in the inputs of its generating action that matches the second suffix etc.,
   * and repeats this until the supplied suffixes run out.
   */
  protected Artifact artifactByPath(Iterable<Artifact> artifacts, String... suffixes) {
    Artifact artifact = getFirstArtifactEndingWith(artifacts, suffixes[0]);
    Action action = null;
    for (int i = 1; i < suffixes.length; i++) {
      if (artifact == null) {
        if (action == null) {
          throw new IllegalStateException("No suffix " + suffixes[0] + " among artifacts: "
              + ActionsTestUtil.baseArtifactNames(artifacts));
        } else {
          throw new IllegalStateException("No suffix " + suffixes[i]
              + " among inputs of action " + action.describe() + ": "
              + ActionsTestUtil.baseArtifactNames(artifacts));
        }
      }

      action = getGeneratingAction(artifact);
      artifacts = action.getInputs().toList();
      artifact = getFirstArtifactEndingWith(artifacts, suffixes[i]);
    }

    return artifact;
  }

  /**
   * Retrieves an instance of {@code PseudoAction} that is shadowed by an extra action
   * @param targetLabel Label of the target with an extra action
   * @param actionListenerLabel Label of the action listener
   */
  protected PseudoAction<?> getPseudoActionViaExtraAction(
      String targetLabel, String actionListenerLabel) throws Exception {
    useConfiguration(String.format("--experimental_action_listener=%s", actionListenerLabel));

    ConfiguredTarget target = getConfiguredTarget(targetLabel);
    List<Action> actions = getExtraActionActions(target);

    assertThat(actions).isNotNull();
    assertThat(actions).hasSize(2);

    ExtraAction extraAction = null;

    for (Action action : actions) {
      if (action instanceof ExtraAction) {
        extraAction = (ExtraAction) action;
        break;
      }
    }

    assertWithMessage(actions.toString()).that(extraAction).isNotNull();

    Action pseudoAction = extraAction.getShadowedAction();

    assertThat(pseudoAction).isInstanceOf(PseudoAction.class);
    assertThat(pseudoAction.getPrimaryOutput().getExecPathString())
        .isEqualTo(
            String.format(
                "%s%s.extra_action_dummy",
                targetConfig.getGenfilesFragment(), convertLabelToPath(targetLabel)));

    return (PseudoAction<?>) pseudoAction;
  }

  /**
   * Converts the given label to an output path where double slashes and colons are
   * replaced with single slashes
   * @param label
   */
  private String convertLabelToPath(String label) {
    return label.replace(':', '/').substring(1);
  }

  protected final String getImplicitOutputPath(
      ConfiguredTarget target, SafeImplicitOutputsFunction outputFunction) {
    Rule rule;
    try {
      rule = (Rule) skyframeExecutor.getPackageManager().getTarget(reporter, target.getLabel());
    } catch (NoSuchPackageException | NoSuchTargetException | InterruptedException e) {
      throw new IllegalStateException(e);
    }
    RawAttributeMapper attr = RawAttributeMapper.of(rule.getAssociatedRule());

    return Iterables.getOnlyElement(outputFunction.getImplicitOutputs(eventCollector, attr));
  }

  /**
   * Gets the artifact whose name is derived from {@code outputFunction}. Despite the name, this can
   * be called for artifacts that are not declared as implicit outputs: it just finds the artifact
   * inside the configured target by calling {@link #getBinArtifact(String, ConfiguredTarget)} on
   * the result of the {@code outputFunction}.
   */
  protected final Artifact getImplicitOutputArtifact(
      ConfiguredTarget target, SafeImplicitOutputsFunction outputFunction) {
    return getBinArtifact(getImplicitOutputPath(target, outputFunction), target);
  }

  public Path getExecRoot() {
    return directories.getExecRoot(ruleClassProvider.getRunfilesPrefix());
  }

  /** Returns true iff commandLine contains the option --flagName followed by arg. */
  protected static boolean containsFlag(String flagName, String arg, Iterable<String> commandLine) {
    Iterator<String> iterator = commandLine.iterator();
    while (iterator.hasNext()) {
      if (flagName.equals(iterator.next()) && iterator.hasNext() && arg.equals(iterator.next())) {
        return true;
      }
    }
    return false;
  }

  /** Returns the list of arguments in commandLine that follow after --flagName. */
  protected static ImmutableList<String> flagValue(String flagName, Iterable<String> commandLine) {
    ImmutableList.Builder<String> resultBuilder = ImmutableList.builder();
    Iterator<String> iterator = commandLine.iterator();
    boolean found = false;
    while (iterator.hasNext()) {
      String val = iterator.next();
      if (found) {
        if (val.startsWith("--")) {
          break;
        }
        resultBuilder.add(val);
      } else if (flagName.equals(val)) {
        found = true;
      }
    }
    Preconditions.checkArgument(found);
    return resultBuilder.build();
  }

  /** Creates instances of {@link ActionExecutionContext} consistent with test case. */
  public class ActionExecutionContextBuilder {
    private MetadataProvider actionInputFileCache = null;
    private TreeMap<String, String> clientEnv = new TreeMap<>();
    private ArtifactExpander artifactExpander = null;

    public ActionExecutionContextBuilder setMetadataProvider(
        MetadataProvider actionInputFileCache) {
      this.actionInputFileCache = actionInputFileCache;
      return this;
    }

    public ActionExecutionContextBuilder setArtifactExpander(ArtifactExpander artifactExpander) {
      this.artifactExpander = artifactExpander;
      return this;
    }

    public ActionExecutionContext build() {
      return new ActionExecutionContext(
          new DummyExecutor(fileSystem, getExecRoot()),
          actionInputFileCache,
          /*actionInputPrefetcher=*/ null,
          actionKeyContext,
          /*metadataHandler=*/ null,
          LostInputsCheck.NONE,
          actionLogBufferPathGenerator.generate(ArtifactPathResolver.IDENTITY),
          reporter,
          clientEnv,
          /*topLevelFilesets=*/ ImmutableMap.of(),
          artifactExpander,
          /*actionFileSystem=*/ null,
          /*skyframeDepsResult*/ null,
          NestedSetExpander.DEFAULT);
    }
  }
}
