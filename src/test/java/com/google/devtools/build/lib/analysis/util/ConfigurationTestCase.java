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

import static org.junit.Assert.fail;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.flags.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction.CrossRepositoryLabelViolationStrategy;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.BuildFileName;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Testing framework for tests which check ConfigurationFactory.
 */
@RunWith(JUnit4.class)
public abstract class ConfigurationTestCase extends FoundationTestCase {

  public static final class TestOptions extends OptionsBase {
    @Option(name = "multi_cpu",
            converter = Converters.CommaSeparatedOptionListConverter.class,
            allowMultiple = true,
            defaultValue = "",
            category = "semantics",
            help = "Additional target CPUs.")
    public List<String> multiCpus;
  }

  protected MockToolsConfig mockToolsConfig;
  protected Path workspace;
  protected AnalysisMock analysisMock;
  protected SequencedSkyframeExecutor skyframeExecutor;
  protected ConfigurationFactory configurationFactory;
  protected ImmutableList<Class<? extends FragmentOptions>> buildOptionClasses;

  @Before
  public final void initializeSkyframeExecutor() throws Exception {
    workspace = rootDirectory;
    analysisMock = getAnalysisMock();
    ConfiguredRuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    PathPackageLocator pkgLocator =
        new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory));
    final PackageFactory pkgFactory;
    BlazeDirectories directories =
        new BlazeDirectories(outputBase, outputBase, rootDirectory, analysisMock.getProductName());
    pkgFactory =
        analysisMock
            .getPackageFactoryForTesting()
            .create(ruleClassProvider, scratch.getFileSystem());
    AnalysisTestUtil.DummyWorkspaceStatusActionFactory workspaceStatusActionFactory =
        new AnalysisTestUtil.DummyWorkspaceStatusActionFactory(directories);

    skyframeExecutor =
        SequencedSkyframeExecutor.create(
            pkgFactory,
            directories,
            BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools()),
            workspaceStatusActionFactory,
            ruleClassProvider.getBuildInfoFactories(),
            ImmutableList.<DiffAwareness.Factory>of(),
            Predicates.<PathFragment>alwaysFalse(),
            Preprocessor.Factory.Supplier.NullSupplier.INSTANCE,
            analysisMock.getSkyFunctions(),
            ImmutableList.<PrecomputedValue.Injected>of(),
            ImmutableList.<SkyValueDirtinessChecker>of(),
            analysisMock.getProductName(),
            CrossRepositoryLabelViolationStrategy.ERROR,
            ImmutableList.of(BuildFileName.BUILD_DOT_BAZEL, BuildFileName.BUILD));

    PackageCacheOptions packageCacheOptions = Options.getDefaults(PackageCacheOptions.class);
    packageCacheOptions.showLoadingProgress = true;
    packageCacheOptions.globbingThreads = 7;
    skyframeExecutor.preparePackageLoading(
        pkgLocator,
        packageCacheOptions,
        ruleClassProvider.getDefaultsPackageContent(
            analysisMock.getInvocationPolicyEnforcer().getInvocationPolicy()),
        UUID.randomUUID(),
        ImmutableMap.<String, String>of(),
        new TimestampGranularityMonitor(BlazeClock.instance()));

    mockToolsConfig = new MockToolsConfig(rootDirectory);
    analysisMock.setupMockClient(mockToolsConfig);
    analysisMock.setupMockWorkspaceFiles(directories.getEmbeddedBinariesRoot());
    configurationFactory = analysisMock.createConfigurationFactory();
    buildOptionClasses = ruleClassProvider.getConfigurationOptions();
  }

  protected AnalysisMock getAnalysisMock() {
    return AnalysisMock.get();
  }

  protected void checkError(String expectedMessage, String... options) throws Exception {
    reporter.removeHandler(failFastHandler);
    try {
      create(options);
      fail();
    } catch (InvalidConfigurationException e) {
      assertContainsEvent(expectedMessage);
    }
  }

  protected BuildConfigurationCollection createCollection(String... args) throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(
        ImmutableList.<Class<? extends OptionsBase>>builder()
        .addAll(buildOptionClasses)
        .add(TestOptions.class)
        .build());
    parser.parse(args);

    InvocationPolicyEnforcer optionsPolicyEnforcer = analysisMock.getInvocationPolicyEnforcer();
    optionsPolicyEnforcer.enforce(parser);

    ImmutableSortedSet<String> multiCpu = ImmutableSortedSet.copyOf(
        parser.getOptions(TestOptions.class).multiCpus);

    BuildOptions buildOptions = BuildOptions.applyStaticConfigOverride(
         BuildOptions.of(buildOptionClasses, parser));
    skyframeExecutor.handleDiffs(reporter);
    BuildConfigurationCollection collection = skyframeExecutor.createConfigurations(
        reporter, configurationFactory, buildOptions, multiCpu, false);
    return collection;
  }

  protected BuildConfiguration create(String... args) throws Exception {
    return Iterables.getOnlyElement(createCollection(args).getTargetConfigurations());
  }

  protected BuildConfiguration createHost(String... args) throws Exception {
    return createCollection(args).getHostConfiguration();
  }

  public void assertConfigurationsHaveUniqueOutputDirectories(
      BuildConfigurationCollection configCollection) throws Exception {
    Collection<BuildConfiguration> allConfigs = configCollection.getAllConfigurations();
    Map<Root, BuildConfiguration> outputPaths = new HashMap<>();
    for (BuildConfiguration config : allConfigs) {
      if (config.isActionsEnabled()) {
        BuildConfiguration otherConfig = outputPaths.get(
            config.getOutputDirectory(RepositoryName.MAIN));
        if (otherConfig != null) {
          throw new IllegalStateException("The output path '"
              + config.getOutputDirectory(RepositoryName.MAIN)
              + "' is the same for configurations '" + config + "' and '" + otherConfig + "'");
        } else {
          outputPaths.put(config.getOutputDirectory(RepositoryName.MAIN), config);
        }
      }
    }
  }
}
