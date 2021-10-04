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
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
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
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.BuildInfoCollectionFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
  protected ImmutableList<Class<? extends FragmentOptions>> buildOptionClasses;
  protected final ActionKeyContext actionKeyContext = new ActionKeyContext();

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
    final PackageFactory pkgFactory;
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

    pkgFactory =
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
            .build();
    SkyframeExecutorTestHelper.process(skyframeExecutor);
    skyframeExecutor.injectExtraPrecomputedValues(
        ImmutableList.of(
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.RESOLVED_FILE_INSTEAD_OF_WORKSPACE, Optional.empty()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.REPOSITORY_OVERRIDES, ImmutableMap.of()),
            PrecomputedValue.injected(
                RepositoryDelegatorFunction.DEPENDENCY_FOR_UNCONDITIONAL_FETCHING,
                RepositoryDelegatorFunction.DONT_FETCH_UNCONDITIONALLY),
            PrecomputedValue.injected(RepositoryDelegatorFunction.ENABLE_BZLMOD, false),
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
        new TimestampGranularityMonitor(BlazeClock.instance()));
    skyframeExecutor.setActionEnv(ImmutableMap.of());

    mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());
    buildOptionClasses = ruleClassProvider.getConfigurationOptions();
  }

  protected void checkError(String expectedMessage, String... options) {
    reporter.removeHandler(failFastHandler);
    assertThrows(InvalidConfigurationException.class, () -> create(options));
    assertContainsEvent(expectedMessage);
  }

  /**
   * Returns a {@link BuildConfigurationCollection} with the given non-default options.
   *
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationCollection createCollection(String... args) throws Exception {
    return createCollection(ImmutableMap.of(), args);
  }

  /**
   * Variation of {@link #createCollection(String...)} that also supports Starlark-defined options.
   *
   * @param starlarkOptions map of Starlark-defined options where the keys are option names (in the
   *     form of label-like strings) and the values are option values
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfigurationCollection createCollection(
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

    ImmutableSortedSet<String> multiCpu = ImmutableSortedSet.copyOf(
        parser.getOptions(TestOptions.class).multiCpus);

    skyframeExecutor.handleDiffsForTesting(reporter);
    return skyframeExecutor.createConfigurations(
        reporter, BuildOptions.of(buildOptionClasses, parser), multiCpu, false);
  }

  /**
   * Returns a target {@link BuildConfiguration} with the given non-default options.
   *
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfiguration create(String... args) throws Exception {
    return Iterables.getOnlyElement(createCollection(args).getTargetConfigurations());
  }

  /**
   * Variation of {@link #create(String...)} that also supports Starlark-defined options.
   *
   * @param starlarkOptions map of Starlark-defined options where the keys are option names (in the
   *     form of label-like strings) and the values are option values
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfiguration create(ImmutableMap<String, Object> starlarkOptions, String... args)
      throws Exception {
    return Iterables.getOnlyElement(
        createCollection(starlarkOptions, args).getTargetConfigurations());
  }

  /**
   * Returns a host {@link BuildConfiguration} derived from a target configuration with the given
   * non-default options.
   *
   * @param args native option name/pair descriptions in command line form (e.g. "--cpu=k8")
   */
  protected BuildConfiguration createHost(String... args) throws Exception {
    return createCollection(args).getHostConfiguration();
  }

  public static void assertConfigurationsHaveUniqueOutputDirectories(
      BuildConfigurationCollection configCollection) {
    Map<ArtifactRoot, BuildConfiguration> outputPaths = new HashMap<>();
    for (BuildConfiguration config : configCollection.getTargetConfigurations()) {
      BuildConfiguration otherConfig =
          outputPaths.get(config.getOutputDirectory(RepositoryName.MAIN));
      if (otherConfig != null) {
        throw new IllegalStateException(
            "The output path '"
                + config.getOutputDirectory(RepositoryName.MAIN)
                + "' is the same for configurations '"
                + config
                + "' and '"
                + otherConfig
                + "'");
      } else {
        outputPaths.put(config.getOutputDirectory(RepositoryName.MAIN), config);
      }
    }
  }
}
