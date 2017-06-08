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

package com.google.devtools.build.lib.rules.test;

import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.rules.test.TestProvider.TestParams;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.EnumConverter;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Helper class to create test actions.
 */
public final class TestActionBuilder {

  private final RuleContext ruleContext;
  private RunfilesSupport runfilesSupport;
  private Artifact executable;
  private ExecutionInfoProvider executionRequirements;
  private InstrumentedFilesProvider instrumentedFiles;
  private int explicitShardCount;
  private Map<String, String> extraEnv;

  public TestActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.extraEnv = new TreeMap<>();
  }

  /**
   * Creates the test actions and artifacts using the previously set parameters.
   *
   * @return ordered list of test status artifacts
   */
  public TestParams build() {
    Preconditions.checkState(runfilesSupport != null);
    boolean local = TargetUtils.isTestRuleAndRunsLocally(ruleContext.getRule());
    TestShardingStrategy strategy = ruleContext.getConfiguration().testShardingStrategy();
    int shards = strategy.getNumberOfShards(
        local, explicitShardCount, isTestShardingCompliant(),
        TestSize.getTestSize(ruleContext.getRule()));
    Preconditions.checkState(shards >= 0);
    return createTestAction(shards);
  }

  private boolean isTestShardingCompliant() {
    // See if it has a data dependency on the special target
    // //tools:test_sharding_compliant. Test runners add this dependency
    // to show they speak the sharding protocol.
    // There are certain cases where this heuristic may fail, giving
    // a "false positive" (where we shard the test even though the
    // it isn't supported). We may want to refine this logic, but
    // heuristically sharding is currently experimental. Also, we do detect
    // false-positive cases and return an error.
    return runfilesSupport.getRunfilesSymlinkNames().contains(
        PathFragment.create("tools/test_sharding_compliant"));
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

  public TestActionBuilder setInstrumentedFiles(
      @Nullable InstrumentedFilesProvider instrumentedFiles) {
    this.instrumentedFiles = instrumentedFiles;
    return this;
  }

  public TestActionBuilder setExecutionRequirements(
      @Nullable ExecutionInfoProvider executionRequirements) {
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

  /**
   * Converts to {@link TestActionBuilder.TestShardingStrategy}.
   */
  public static class ShardingStrategyConverter extends EnumConverter<TestShardingStrategy> {
    public ShardingStrategyConverter() {
      super(TestShardingStrategy.class, "test sharding strategy");
    }
  }

  /**
   * A strategy for running the same tests in many processes.
   */
  public static enum TestShardingStrategy {
    EXPLICIT {
      @Override public int getNumberOfShards(boolean isLocal, int shardCountFromAttr,
          boolean testShardingCompliant, TestSize testSize) {
        return Math.max(shardCountFromAttr, 0);
      }
    },

    EXPERIMENTAL_HEURISTIC {
      @Override public int getNumberOfShards(boolean isLocal, int shardCountFromAttr,
          boolean testShardingCompliant, TestSize testSize) {
        if (shardCountFromAttr >= 0) {
          return shardCountFromAttr;
        }
        if (isLocal || !testShardingCompliant) {
          return 0;
        }
        return testSize.getDefaultShards();
      }
    },

    DISABLED {
      @Override public int getNumberOfShards(boolean isLocal, int shardCountFromAttr,
          boolean testShardingCompliant, TestSize testSize) {
        return 0;
      }
    };

    public abstract int getNumberOfShards(boolean isLocal, int shardCountFromAttr,
        boolean testShardingCompliant, TestSize testSize);
  }

  /**
   * Creates a test action and artifacts for the given rule. The test action will
   * use the specified executable and runfiles.
   *
   * @return ordered list of test artifacts, one per action. These are used to drive
   *    execution in Skyframe, and by AggregatingTestListener and
   *    TestResultAnalyzer to keep track of completed and pending test runs.
   */
  private TestParams createTestAction(int shards) {
    PathFragment targetName = PathFragment.create(ruleContext.getLabel().getName());
    BuildConfiguration config = ruleContext.getConfiguration();
    AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
    Root root = config.getTestLogsDirectory(ruleContext.getRule().getRepository());

    NestedSetBuilder<Artifact> inputsBuilder = NestedSetBuilder.stableOrder();
    inputsBuilder.addTransitive(
        NestedSetBuilder.create(Order.STABLE_ORDER, runfilesSupport.getRunfilesMiddleman()));
    NestedSet<Artifact> testRuntime = PrerequisiteArtifacts.nestedSet(
        ruleContext, "$test_runtime", Mode.HOST);
    inputsBuilder.addTransitive(testRuntime);
    TestTargetProperties testProperties = new TestTargetProperties(
        ruleContext, executionRequirements);

    // If the test rule does not provide InstrumentedFilesProvider, there's not much that we can do.
    final boolean collectCodeCoverage = config.isCodeCoverageEnabled()
        && instrumentedFiles != null;

    TreeMap<String, String> extraTestEnv = new TreeMap<>();

    TestTargetExecutionSettings executionSettings;
    if (collectCodeCoverage) {
      inputsBuilder.addTransitive(instrumentedFiles.getCoverageSupportFiles());
      // Add instrumented file manifest artifact to the list of inputs. This file will contain
      // exec paths of all source files that should be included into the code coverage output.
      NestedSet<Artifact> metadataFiles = instrumentedFiles.getInstrumentationMetadataFiles();
      inputsBuilder.addTransitive(metadataFiles);
      inputsBuilder.addTransitive(PrerequisiteArtifacts.nestedSet(
          ruleContext, "$coverage_support", Mode.DONT_CHECK));
      // We don't add this attribute to non-supported test target
      if (ruleContext.isAttrDefined("$lcov_merger", LABEL)) {
        Artifact lcovMerger = ruleContext.getPrerequisiteArtifact("$lcov_merger", Mode.TARGET);
        if (lcovMerger != null) {
          inputsBuilder.addTransitive(
              PrerequisiteArtifacts.nestedSet(ruleContext, "$lcov_merger", Mode.TARGET));
          // Pass this LcovMerger_deploy.jar path to collect_coverage.sh
          extraTestEnv.put("LCOV_MERGER", lcovMerger.getExecPathString());
        }
      }

      Artifact instrumentedFileManifest =
          InstrumentedFileManifestAction.getInstrumentedFileManifest(ruleContext,
              instrumentedFiles.getInstrumentedFiles(), metadataFiles);
      executionSettings = new TestTargetExecutionSettings(ruleContext, runfilesSupport,
          executable, instrumentedFileManifest, shards);
      inputsBuilder.add(instrumentedFileManifest);
      // TODO(ulfjack): Is this even ever set? If yes, does this cost us a lot of memory?
      for (Pair<String, String> coverageEnvEntry : instrumentedFiles.getCoverageEnvironment()) {
        extraTestEnv.put(coverageEnvEntry.getFirst(), coverageEnvEntry.getSecond());
      }
    } else {
      executionSettings = new TestTargetExecutionSettings(ruleContext, runfilesSupport,
          executable, null, shards);
    }

    extraTestEnv.putAll(extraEnv);

    if (config.getRunUnder() != null) {
      Artifact runUnderExecutable = executionSettings.getRunUnderExecutable();
      if (runUnderExecutable != null) {
        inputsBuilder.add(runUnderExecutable);
      }
    }

    int runsPerTest = config.getRunsPerTestForLabel(ruleContext.getLabel());

    Iterable<Artifact> inputs = inputsBuilder.build();
    int shardRuns = (shards > 0 ? shards : 1);
    List<Artifact> results = Lists.newArrayListWithCapacity(runsPerTest * shardRuns);
    ImmutableList.Builder<Artifact> coverageArtifacts = ImmutableList.builder();

    boolean useTestRunner = false;
    if (ruleContext.attributes().has("use_testrunner", Type.BOOLEAN)) {
      useTestRunner = ruleContext.attributes().get("use_testrunner", Type.BOOLEAN);
    }

    for (int run = 0; run < runsPerTest; run++) {
      // Use a 1-based index for user friendliness.
      String testRunDir =
          runsPerTest > 1 ? String.format("run_%d_of_%d", run + 1, runsPerTest) : "";
      for (int shard = 0; shard < shardRuns; shard++) {
        String shardRunDir =
            (shardRuns > 1 ? String.format("shard_%d_of_%d", shard + 1, shards) : "");
        if (testRunDir.isEmpty()) {
          shardRunDir = shardRunDir.isEmpty() ? "" : shardRunDir + PathFragment.SEPARATOR_CHAR;
        } else {
          testRunDir += PathFragment.SEPARATOR_CHAR;
          shardRunDir = shardRunDir.isEmpty() ? testRunDir : shardRunDir + "_" + testRunDir;
        }
        Artifact testLog =
            ruleContext.getPackageRelativeArtifact(
                targetName.getRelative(shardRunDir + "test.log"), root);
        Artifact cacheStatus =
            ruleContext.getPackageRelativeArtifact(
                targetName.getRelative(shardRunDir + "test.cache_status"), root);

        Artifact coverageArtifact = null;
        if (collectCodeCoverage) {
          coverageArtifact = ruleContext.getPackageRelativeArtifact(
              targetName.getRelative(shardRunDir + "coverage.dat"), root);
          coverageArtifacts.add(coverageArtifact);
        }

        env.registerAction(new TestRunnerAction(
            ruleContext.getActionOwner(), inputs, testRuntime,
            testLog, cacheStatus,
            coverageArtifact,
            testProperties, extraTestEnv, executionSettings,
            shard, run, config, ruleContext.getWorkspaceName(),
            useTestRunner));
        results.add(cacheStatus);
      }
    }
    // TODO(bazel-team): Passing the reportGenerator to every TestParams is a bit strange.
    Artifact reportGenerator = null;
    if (config.isCodeCoverageEnabled()) {
      // It's not enough to add this if the rule has coverage enabled because the command line may
      // contain rules with baseline coverage but no test rules that have coverage enabled, and in
      // that case, we still need the report generator.
      reportGenerator = ruleContext.getPrerequisiteArtifact(
          "$coverage_report_generator", Mode.HOST);
    }

    return new TestParams(runsPerTest, shards, TestTimeout.getTestTimeout(ruleContext.getRule()),
        ruleContext.getRule().getRuleClass(), ImmutableList.copyOf(results),
        coverageArtifacts.build(), reportGenerator);
  }
}
