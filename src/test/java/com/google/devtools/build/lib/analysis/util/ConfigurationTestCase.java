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

import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.runtime.QuiescingExecutorsImpl;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Testing framework for tests which create configuration collections.
 */
@RunWith(JUnit4.class)
public abstract class ConfigurationTestCase extends FoundationTestCase {

  public static final class TestOptions extends OptionsBase {
    @Option(
        name = "multi_cpu",
        converter = Converters.CommaSeparatedOptionListConverter.class,
        allowMultiple = true,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "null",
        help = "Additional target CPUs.")
    public List<String> multiCpus;
  }

  protected MockToolsConfig mockToolsConfig;
  protected Path workspace;
  protected AnalysisMock analysisMock;
  protected SequencedSkyframeExecutor skyframeExecutor;
  protected ImmutableSet<Class<? extends FragmentOptions>> buildOptionClasses;
  protected final ActionKeyContext actionKeyContext = new ActionKeyContext();
  private FragmentFactory fragmentFactory;

  @Before
  public final void initializeSkyframeExecutor() throws Exception {
    workspace = rootDirectory;
    analysisMock = AnalysisMock.get();

    ConfiguredRuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    PathPackageLocator pkgLocator =
        new PathPackageLocator(
            outputBase,
            ImmutableList.of(Root.fromPath(rootDirectory)),
            BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY);
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, outputBase),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            analysisMock.getProductName());

    mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockToolsRepository(mockToolsConfig);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());

    PackageFactory pkgFactory =
        analysisMock
            .getPackageFactoryBuilderForTesting(directories)
            .build(ruleClassProvider, fileSystem);
    AnalysisTestUtil.DummyWorkspaceStatusActionFactory workspaceStatusActionFactory =
        new AnalysisTestUtil.DummyWorkspaceStatusActionFactory();
    skyframeExecutor =
        BazelSkyframeExecutorConstants.newBazelSkyframeExecutorBuilder()
            .setPkgFactory(pkgFactory)
            .setFileSystem(fileSystem)
            .setDirectories(directories)
            .setActionKeyContext(actionKeyContext)
            .setWorkspaceStatusActionFactory(workspaceStatusActionFactory)
            .setExtraSkyFunctions(analysisMock.getSkyFunctions(directories))
            .setSyscallCache(SyscallCache.NO_CACHE)
            .build();
    SkyframeExecutorTestHelper.process(skyframeExecutor);
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                PrecomputedValue.BASELINE_CONFIGURATION,
                BuildOptions.getDefaultBuildOptionsForFragments(
                    ImmutableList.of(CoreOptions.class))),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.REPOSITORY_OVERRIDES, ImmutableMap.of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY),
            PrecomputedValue.injected(
                BuildInfoCollectionFunction.BUILD_INFO_FACTORIES,
                ruleClassProvider.getBuildInfoFactoriesAsMap())));
    PackageOptions packageOptions = Options.getDefaults(PackageOptions.class);
    packageOptions.showLoadingProgress = true;
    packageOptions.globbingThreads = 7;
    skyframeExecutor.preparePackageLoading(
        pkgLocator,
        packageOptions,
        Options.getDefaults(BuildLanguageOptions.class),
        UUID.randomUUID(),
        ImmutableMap.of(),
        QuiescingExecutorsImpl.forTesting(),
        new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.of());

    mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());
    buildOptionClasses = ruleClassProvider.getFragmentRegistry().getOptionsClasses();
    fragmentFactory = new FragmentFactory();
  }

  protected void checkError(String expectedMessage, String... options) {
    reporter.removeHandler(failFastHandler);
    assertThrows(InvalidConfigurationException.class, () -> create(options));
    assertContainsEvent(expectedMessage);
  }

  /**
   * Returns a {@link BuildConfigurationValue} with the given non-default options.
   *
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationValue createConfiguration(String... args) throws Exception {
    return createConfiguration(ImmutableMap.of(), args);
  }

  /**
   * Variation of {@link #createConfiguration(String...)} that also supports Starlark-defined
   * options.
   *
   * @param starlarkOptions map of Starlark-defined options where the keys are option names (in the
   *     form of label-like strings) and the values are option values
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationValue createConfiguration(
      ImmutableMap<String, Object> starlarkOptions, String... args) throws Exception {

    BuildOptions targetOptions = parseBuildOptions(starlarkOptions, args);

    skyframeExecutor.handleDiffsForTesting(reporter);
    skyframeExecutor.setBaselineConfiguration(targetOptions);
    return skyframeExecutor.createConfiguration(reporter, targetOptions, false);
  }

  /** Parses purported commandline options into a BuildOptions (assumes default parsing context.) */
  private Pair<BuildOptions, TestOptions> parseBuildOptionsWithTestOptions(
      ImmutableMap<String, Object> starlarkOptions, String... args) throws Exception {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(
                ImmutableList.<Class<? extends OptionsBase>>builder()
                    .addAll(buildOptionClasses)
                    .add(TestOptions.class)
                    .build())
            .build();
    parser.setStarlarkOptions(starlarkOptions);
    parser.parse(TestConstants.PRODUCT_SPECIFIC_FLAGS);
    parser.parse(args);

    return Pair.of(
        BuildOptions.of(buildOptionClasses, parser), parser.getOptions(TestOptions.class));
  }

  /** Parses purported commandline options into a BuildOptions (assumes default parsing context.) */
  protected BuildOptions parseBuildOptions(
      ImmutableMap<String, Object> starlarkOptions, String... args) throws Exception {
    return parseBuildOptionsWithTestOptions(starlarkOptions, args).getFirst();
  }

  /** Parses purported commandline options into a BuildOptions (assumes default parsing context.) */
  protected BuildOptions parseBuildOptions(String... args) throws Exception {
    return parseBuildOptions(ImmutableMap.of(), args);
  }

  /** Returns a raw {@link BuildConfigurationValue} with the given parameters. */
  protected BuildConfigurationValue createRaw(
      BuildOptions buildOptions,
      String repositoryName,
      boolean siblingRepositoryLayout,
      String transitionDirectoryNameFragment)
      throws Exception {
    return BuildConfigurationValue.create(
        buildOptions,
        RepositoryName.create(repositoryName),
        siblingRepositoryLayout,
        transitionDirectoryNameFragment,
        skyframeExecutor.getBlazeDirectoriesForTesting(),
        skyframeExecutor.getRuleClassProviderForTesting(),
        fragmentFactory);
  }

  /**
   * Returns a target {@link BuildConfigurationValue} with the given non-default options.
   *
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationValue create(String... args) throws Exception {
    return createConfiguration(args);
  }

  /**
   * Variation of {@link #create(String...)} that also supports Starlark-defined options.
   *
   * @param starlarkOptions map of Starlark-defined options where the keys are option names (in the
   *     form of label-like strings) and the values are option values
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationValue create(
      ImmutableMap<String, Object> starlarkOptions, String... args) throws Exception {
    return createConfiguration(starlarkOptions, args);
  }

  /**
   * Returns an exec {@link BuildConfigurationValue} derived from a target configuration with the
   * given non-default options.
   *
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationValue createExec(String... args) throws Exception {
    return skyframeExecutor.getConfiguration(
        reporter,
        AnalysisTestUtil.execOptions(parseBuildOptions(args), reporter),
        /* keepGoing= */ false);
  }
}
