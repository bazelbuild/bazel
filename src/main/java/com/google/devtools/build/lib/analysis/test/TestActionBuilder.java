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

package com.google.devtools.build.lib.analysis.test;

import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.SingleRunfilesSupplier;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.LazyWriteNestedSetOfPairAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.test.TestProvider.TestParams;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Helper class to create test actions.
 */
public final class TestActionBuilder {
  private static final String CC_CODE_COVERAGE_SCRIPT = "CC_CODE_COVERAGE_SCRIPT";
  private static final String LCOV_MERGER = "LCOV_MERGER";
  // The coverage tool Bazel uses to generate a code coverage report for C++.
  private static final String BAZEL_CC_COVERAGE_TOOL = "BAZEL_CC_COVERAGE_TOOL";
  private static final String GCOV_TOOL = "GCOV";
  // A file that contains a mapping between the reported source file path and the actual source
  // file path, relative to the workspace directory, if the two values are different. If the
  // reported source file is the same as the actual source path it will not be included in the file.
  private static final String COVERAGE_REPORTED_TO_ACTUAL_SOURCES_FILE =
      "COVERAGE_REPORTED_TO_ACTUAL_SOURCES_FILE";

  private final RuleContext ruleContext;
  private final ImmutableList.Builder<Artifact> additionalTools;
  private RunfilesSupport runfilesSupport;
  private Runfiles persistentTestRunnerRunfiles;
  private Artifact executable;
  private ExecutionInfo executionRequirements;
  private InstrumentedFilesInfo instrumentedFiles;
  private int explicitShardCount;
  private final Map<String, String> extraEnv;

  public TestActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.extraEnv = new TreeMap<>();
    this.additionalTools = new ImmutableList.Builder<>();
  }

  /**
   * Creates the test actions and artifacts using the previously set parameters.
   *
   * @return ordered list of test status artifacts
   */
  public TestParams build() throws InterruptedException { // due to TestTargetExecutionSettings
    Preconditions.checkNotNull(runfilesSupport);
    TestShardingStrategy strategy =
        ruleContext.getConfiguration().getFragment(TestConfiguration.class).testShardingStrategy();
    int shards = strategy.getNumberOfShards(explicitShardCount);
    Preconditions.checkState(shards >= 0, "%s returned negative shard count %s", strategy, shards);
    return createTestAction(shards);
  }

  /**
   * Set the runfiles and executable to be run as a test.
   */
  public TestActionBuilder setFilesToRunProvider(FilesToRunProvider provider) {
    Preconditions.checkNotNull(provider.getRunfilesSupport());
    Preconditions.checkNotNull(provider.getExecutable());
    this.runfilesSupport = provider.getRunfilesSupport();
    this.executable = provider.getExecutable();
    return this;
  }

  public TestActionBuilder setPersistentTestRunnerRunfiles(Runfiles runfiles) {
    this.persistentTestRunnerRunfiles = runfiles;
    return this;
  }

  public TestActionBuilder addTools(List<Artifact> tools) {
    this.additionalTools.addAll(tools);
    return this;
  }

  public TestActionBuilder setInstrumentedFiles(@Nullable InstrumentedFilesInfo instrumentedFiles) {
    this.instrumentedFiles = instrumentedFiles;
    return this;
  }

  public TestActionBuilder setExecutionRequirements(
      @Nullable ExecutionInfo executionRequirements) {
    this.executionRequirements = executionRequirements;
    return this;
  }

  public TestActionBuilder addExtraEnv(Map<String, String> extraEnv) {
    this.extraEnv.putAll(extraEnv);
    return this;
  }

  /**
   * Set the explicit shard count. Note that this may be overridden by the sharding strategy.
   */
  public TestActionBuilder setShardCount(int explicitShardCount) {
    this.explicitShardCount = explicitShardCount;
    return this;
  }

  private boolean isPersistentTestRunner() {
    return ruleContext
            .getConfiguration()
            .getFragment(TestConfiguration.class)
            .isPersistentTestRunner()
        && persistentTestRunnerRunfiles != null;
  }

  /**
   * Creates a test action and artifacts for the given rule. The test action will use the specified
   * executable and runfiles.
   *
   * @return ordered list of test artifacts, one per action. These are used to drive execution in
   *     Skyframe, and by AggregatingTestListener and TestResultAnalyzer to keep track of completed
   *     and pending test runs.
   */
  private TestParams createTestAction(int shards)
      throws InterruptedException { // due to TestTargetExecutionSettings
    PathFragment targetName = PathFragment.create(ruleContext.getLabel().getName());
    BuildConfiguration config = ruleContext.getConfiguration();
    TestConfiguration testConfiguration = config.getFragment(TestConfiguration.class);
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    ArtifactRoot root = ruleContext.getTestLogsDirectory();

    // TODO(laszlocsomor), TODO(ulfjack): `isExecutedOnWindows` should use the execution platform,
    // not the host platform. Once Bazel can tell apart these platforms, fix the right side of this
    // initialization.
    final boolean isExecutedOnWindows = OS.getCurrent() == OS.WINDOWS;
    final boolean isUsingTestWrapperInsteadOfTestSetupScript = isExecutedOnWindows;

    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    inputsBuilder.addTransitive(
        NestedSetBuilder.create(Order.STABLE_ORDER, runfilesSupport.getRunfilesMiddleman()));

    if (!isUsingTestWrapperInsteadOfTestSetupScript) {
      NestedSet<Artifact> testRuntime =
          PrerequisiteArtifacts.nestedSet(ruleContext, "$test_runtime");
      inputsBuilder.addTransitive(testRuntime);
    }
    TestTargetProperties testProperties =
        new TestTargetProperties(ruleContext, executionRequirements, isPersistentTestRunner());

    // If the test rule does not provide InstrumentedFilesProvider, there's not much that we can do.
    final boolean collectCodeCoverage = config.isCodeCoverageEnabled()
        && instrumentedFiles != null;

    Artifact testActionExecutable =
        isUsingTestWrapperInsteadOfTestSetupScript
            ? ruleContext.getHostPrerequisiteArtifact("$test_wrapper")
            : ruleContext.getHostPrerequisiteArtifact("$test_setup_script");

    inputsBuilder.add(testActionExecutable);
    Artifact testXmlGeneratorExecutable =
        isUsingTestWrapperInsteadOfTestSetupScript
            ? ruleContext.getHostPrerequisiteArtifact("$xml_writer")
            : ruleContext.getHostPrerequisiteArtifact("$xml_generator_script");
    inputsBuilder.add(testXmlGeneratorExecutable);

    Artifact collectCoverageScript = null;
    TreeMap<String, String> extraTestEnv = new TreeMap<>();

    int runsPerTest = testConfiguration.getRunsPerTestForLabel(ruleContext.getLabel());

    NestedSetBuilder<Artifact> lcovMergerFilesToRun = NestedSetBuilder.compileOrder();
    RunfilesSupplier lcovMergerRunfilesSupplier = null;

    TestTargetExecutionSettings executionSettings;
    if (collectCodeCoverage) {
      collectCoverageScript = ruleContext.getHostPrerequisiteArtifact("$collect_coverage_script");
      inputsBuilder.add(collectCoverageScript);
      inputsBuilder.addTransitive(instrumentedFiles.getCoverageSupportFiles());
      // Add instrumented file manifest artifact to the list of inputs. This file will contain
      // exec paths of all source files that should be included into the code coverage output.
      NestedSet<Artifact> metadataFiles = instrumentedFiles.getInstrumentationMetadataFiles();
      inputsBuilder.addTransitive(metadataFiles);
      inputsBuilder.addTransitive(
          PrerequisiteArtifacts.nestedSet(ruleContext, ":coverage_support"));
      inputsBuilder.addTransitive(
          ruleContext
              .getPrerequisite(":coverage_support", RunfilesProvider.class)
              .getDataRunfiles()
              .getAllArtifacts());

      if (ruleContext.isAttrDefined("$collect_cc_coverage", LABEL)) {
        Artifact collectCcCoverage =
            ruleContext.getHostPrerequisiteArtifact("$collect_cc_coverage");
        inputsBuilder.add(collectCcCoverage);
        extraTestEnv.put(CC_CODE_COVERAGE_SCRIPT, collectCcCoverage.getExecPathString());
      }

      if (!instrumentedFiles.getReportedToActualSources().isEmpty()) {
        Artifact reportedToActualSourcesArtifact =
            ruleContext.getUniqueDirectoryArtifact(
                "_coverage_helpers", "reported_to_actual_sources.txt");
        ruleContext.registerAction(
            new LazyWriteNestedSetOfPairAction(
                ruleContext.getActionOwner(),
                reportedToActualSourcesArtifact,
                instrumentedFiles.getReportedToActualSources()));
        inputsBuilder.add(reportedToActualSourcesArtifact);
        extraTestEnv.put(
            COVERAGE_REPORTED_TO_ACTUAL_SOURCES_FILE,
            reportedToActualSourcesArtifact.getExecPathString());
      }

      // lcov is the default CC coverage tool unless otherwise specified on the command line.
      extraTestEnv.put(BAZEL_CC_COVERAGE_TOOL, GCOV_TOOL);

      // We don't add this attribute to non-supported test target
      String lcovMergerAttr = null;
      if (ruleContext.isAttrDefined(":lcov_merger", LABEL)) {
        lcovMergerAttr = ":lcov_merger";
      } else if (ruleContext.isAttrDefined("$lcov_merger", LABEL)) {
        lcovMergerAttr = "$lcov_merger";
      }
      if (lcovMergerAttr != null) {
        TransitiveInfoCollection lcovMerger = ruleContext.getPrerequisite(lcovMergerAttr);
        FilesToRunProvider lcovFilesToRun = lcovMerger.getProvider(FilesToRunProvider.class);
        if (lcovFilesToRun != null) {
          extraTestEnv.put(LCOV_MERGER, lcovFilesToRun.getExecutable().getExecPathString());
          inputsBuilder.addTransitive(lcovFilesToRun.getFilesToRun());

          lcovMergerFilesToRun.addTransitive(lcovFilesToRun.getFilesToRun());
          if (lcovFilesToRun.getRunfilesSupport() != null) {
            lcovMergerFilesToRun.add(lcovFilesToRun.getRunfilesSupport().getRunfilesMiddleman());
          }
          lcovMergerRunfilesSupplier = lcovFilesToRun.getRunfilesSupplier();
        } else {
          NestedSet<Artifact> filesToBuild =
              lcovMerger.getProvider(FileProvider.class).getFilesToBuild();

          if (filesToBuild.isSingleton()) {
            Artifact lcovMergerArtifact = filesToBuild.getSingleton();
            extraTestEnv.put(LCOV_MERGER, lcovMergerArtifact.getExecPathString());
            inputsBuilder.add(lcovMergerArtifact);
            lcovMergerFilesToRun.add(lcovMergerArtifact);
          } else {
            ruleContext.attributeError(
                lcovMergerAttr,
                "the LCOV merger should be either an executable or a single artifact");
          }
        }
      }

      Artifact instrumentedFileManifest =
          InstrumentedFileManifestAction.getInstrumentedFileManifest(ruleContext,
              instrumentedFiles.getInstrumentedFiles(), metadataFiles);
      executionSettings =
          new TestTargetExecutionSettings(
              ruleContext,
              runfilesSupport,
              executable,
              instrumentedFileManifest,
              /* persistentTestRunnerFlagFile= */ null,
              shards,
              runsPerTest);
      inputsBuilder.add(instrumentedFileManifest);
      // TODO(ulfjack): Is this even ever set? If yes, does this cost us a lot of memory?
      for (Pair<String, String> coverageEnvEntry :
          instrumentedFiles.getCoverageEnvironment().toList()) {
        extraTestEnv.put(coverageEnvEntry.getFirst(), coverageEnvEntry.getSecond());
      }
    } else {
      Artifact flagFile = null;
      // The worker spawn runner expects a flag file containing the worker's flags.
      if (isPersistentTestRunner()) {
        flagFile = ruleContext.getBinArtifact(ruleContext.getLabel().getName() + "_flag_file.txt");
        inputsBuilder.add(flagFile);
      }

      executionSettings =
          new TestTargetExecutionSettings(
              ruleContext, runfilesSupport, executable, null, flagFile, shards, runsPerTest);
    }

    extraTestEnv.putAll(extraEnv);

    if (config.getRunUnder() != null) {
      Artifact runUnderExecutable = executionSettings.getRunUnderExecutable();
      if (runUnderExecutable != null) {
        inputsBuilder.add(runUnderExecutable);
      }
    }

    NestedSet<Artifact> inputs = inputsBuilder.build();
    int shardRuns = (shards > 0 ? shards : 1);
    List<Artifact.DerivedArtifact> results =
        Lists.newArrayListWithCapacity(runsPerTest * shardRuns);
    ImmutableList.Builder<Artifact> coverageArtifacts = ImmutableList.builder();
    ImmutableList.Builder<ActionInput> testOutputs = ImmutableList.builder();

    SingleRunfilesSupplier testRunfilesSupplier;
    if (isPersistentTestRunner()) {
      // Create a RunfilesSupplier from the persistent test runner's runfiles. Pass only the test
      // runner's runfiles to avoid using a different worker for every test run.
      testRunfilesSupplier =
          new SingleRunfilesSupplier(
              /*runfilesDir=*/ persistentTestRunnerRunfiles.getSuffix(),
              /*runfiles=*/ persistentTestRunnerRunfiles,
              /*manifest=*/ null,
              /*buildRunfileLinks=*/ false,
              /*runfileLinksEnabled=*/ false);
    } else if (shardRuns > 1 || runsPerTest > 1) {
      // When creating multiple test actions, cache the runfiles mappings across test actions. This
      // saves a lot of garbage when shard_count and/or runs_per_test is high.
      testRunfilesSupplier =
          SingleRunfilesSupplier.createCaching(
              runfilesSupport.getRunfilesDirectoryExecPath(),
              runfilesSupport.getRunfiles(),
              runfilesSupport.isBuildRunfileLinks(),
              runfilesSupport.isRunfilesEnabled());
    } else {
      testRunfilesSupplier = SingleRunfilesSupplier.create(runfilesSupport);
    }

    // Use 1-based indices for user friendliness.
    for (int shard = 0; shard < shardRuns; shard++) {
      String shardDir = shardRuns > 1 ? String.format("shard_%d_of_%d", shard + 1, shards) : null;
      for (int run = 0; run < runsPerTest; run++) {
        PathFragment dir;
        if (runsPerTest > 1) {
          String runDir = String.format("run_%d_of_%d", run + 1, runsPerTest);
          if (shardDir == null) {
            dir = targetName.getRelative(runDir);
          } else {
            dir = targetName.getRelative(shardDir + "_" + runDir);
          }
        } else if (shardDir == null) {
          dir = targetName;
        } else {
          dir = targetName.getRelative(shardDir);
        }

        Artifact.DerivedArtifact testLog =
            ruleContext.getPackageRelativeArtifact(dir.getRelative("test.log"), root);
        Artifact.DerivedArtifact cacheStatus =
            ruleContext.getPackageRelativeArtifact(dir.getRelative("test.cache_status"), root);

        Artifact.DerivedArtifact coverageArtifact = null;
        Artifact coverageDirectory = null;
        if (collectCodeCoverage) {
          coverageArtifact =
              ruleContext.getPackageRelativeArtifact(dir.getRelative("coverage.dat"), root);
          coverageArtifacts.add(coverageArtifact);
          if (testConfiguration.fetchAllCoverageOutputs()) {
            coverageDirectory =
                ruleContext.getPackageRelativeTreeArtifact(dir.getRelative("_coverage"), root);
          }
        }

        boolean cancelConcurrentTests =
            testConfiguration.runsPerTestDetectsFlakes()
                && testConfiguration.cancelConcurrentTests();


        ImmutableList.Builder<Artifact> tools = new ImmutableList.Builder<>();
        if (isPersistentTestRunner()) {
          tools.add(testActionExecutable);
          tools.add(executionSettings.getExecutable());
          tools.addAll(additionalTools.build());
        }
        boolean splitCoveragePostProcessing = testConfiguration.splitCoveragePostProcessing();
        TestRunnerAction testRunnerAction =
            new TestRunnerAction(
                ruleContext.getActionOwner(),
                inputs,
                testRunfilesSupplier,
                testActionExecutable,
                testXmlGeneratorExecutable,
                collectCoverageScript,
                testLog,
                cacheStatus,
                coverageArtifact,
                coverageDirectory,
                testProperties,
                runfilesSupport.getActionEnvironment().addFixedVariables(extraTestEnv),
                executionSettings,
                shard,
                run,
                config,
                ruleContext.getWorkspaceName(),
                (!isUsingTestWrapperInsteadOfTestSetupScript
                        || executionSettings.needsShell(isExecutedOnWindows))
                    ? ShToolchain.getPathOrError(ruleContext)
                    : null,
                cancelConcurrentTests,
                tools.build(),
                splitCoveragePostProcessing,
                lcovMergerFilesToRun,
                lcovMergerRunfilesSupplier);

        testOutputs.addAll(testRunnerAction.getSpawnOutputs());
        testOutputs.addAll(testRunnerAction.getOutputs());

        env.registerAction(testRunnerAction);

        results.add(cacheStatus);
      }
    }
    // TODO(bazel-team): Passing the reportGenerator to every TestParams is a bit strange.
    FilesToRunProvider reportGenerator = null;
    if (config.isCodeCoverageEnabled()) {
      // It's not enough to add this if the rule has coverage enabled because the command line may
      // contain rules with baseline coverage but no test rules that have coverage enabled, and in
      // that case, we still need the report generator.
      TransitiveInfoCollection reportGeneratorTarget =
          ruleContext.getPrerequisite(":coverage_report_generator");
      reportGenerator = reportGeneratorTarget.getProvider(FilesToRunProvider.class);
    }

    return new TestParams(
        runsPerTest,
        shards,
        testConfiguration.runsPerTestDetectsFlakes(),
        TestTimeout.getTestTimeout(ruleContext.getRule()),
        ruleContext.getRule().getRuleClass(),
        ImmutableList.copyOf(results),
        coverageArtifacts.build(),
        reportGenerator,
        testOutputs.build());
  }
}
