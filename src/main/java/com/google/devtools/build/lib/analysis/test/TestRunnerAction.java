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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContinuationOrResult;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.SingleRunfilesSupplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.analysis.test.TestActionContext.FailedAttemptResult;
import com.google.devtools.build.lib.analysis.test.TestActionContext.TestAttemptContinuation;
import com.google.devtools.build.lib.analysis.test.TestActionContext.TestAttemptResult;
import com.google.devtools.build.lib.analysis.test.TestActionContext.TestAttemptResult.Result;
import com.google.devtools.build.lib.analysis.test.TestActionContext.TestRunnerSpawn;
import com.google.devtools.build.lib.buildeventstream.TestFileNameConstants;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.ImmutableIterable;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TestAction;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.TriState;
import com.google.protobuf.ExtensionRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * An Action representing a test with the associated environment (runfiles, environment variables,
 * test result, etc). It consumes test executable and runfiles artifacts and produces test result
 * and test status artifacts.
 */
// Not final so that we can mock it in tests.
public class TestRunnerAction extends AbstractAction
    implements NotifyOnActionCacheHit, CommandAction {
  public static final PathFragment COVERAGE_TMP_ROOT = PathFragment.create("_coverage");

  // Used for selecting subset of testcase / testmethods.
  private static final String TEST_BRIDGE_TEST_FILTER_ENV = "TESTBRIDGE_TEST_ONLY";

  private static final String GUID = "cc41f9d0-47a6-11e7-8726-eb6ce83a8cc8";
  public static final String MNEMONIC = "TestRunner";

  private final Artifact testSetupScript;
  private final Artifact testXmlGeneratorScript;
  private final Artifact collectCoverageScript;
  private final BuildConfiguration configuration;
  private final TestConfiguration testConfiguration;
  private final Artifact testLog;
  private final Artifact cacheStatus;
  private final PathFragment testWarningsPath;
  private final PathFragment unusedRunfilesLogPath;
  @Nullable private final PathFragment shExecutable;
  private final PathFragment splitLogsPath;
  private final PathFragment splitLogsDir;
  private final PathFragment undeclaredOutputsDir;
  private final PathFragment undeclaredOutputsZipPath;
  private final PathFragment undeclaredOutputsAnnotationsDir;
  private final PathFragment undeclaredOutputsManifestPath;
  private final PathFragment undeclaredOutputsAnnotationsPath;
  private final PathFragment xmlOutputPath;
  @Nullable private final PathFragment testShard;
  private final PathFragment testExitSafe;
  private final PathFragment testStderr;
  private final PathFragment testInfrastructureFailure;
  private final PathFragment baseDir;

  private final ImmutableSet<PathFragment> filesToDeleteBeforeExecution;
  private final ImmutableSet<PathFragment> directoriesToDeleteBeforeExecution;

  private final Artifact coverageData;
  @Nullable private final Artifact coverageDirectory;
  private final TestTargetProperties testProperties;
  private final TestTargetExecutionSettings executionSettings;
  private final int shardNum;
  private final int runNumber;
  private final String workspaceName;

  // Mutable state related to test caching. Lazily initialized: null indicates unknown.
  private Boolean unconditionalExecution;

  /** Any extra environment variables (and values) added by the rule that created this action. */
  private final ActionEnvironment extraTestEnv;

  /**
   * The set of environment variables that are inherited from the client environment. These are
   * handled explicitly by the ActionCacheChecker and so don't have to be included in the cache key.
   */
  private final ImmutableIterable<String> requiredClientEnvVariables;

  private final boolean cancelConcurrentTestsOnSuccess;

  private final boolean splitCoveragePostProcessing;
  private final NestedSetBuilder<Artifact> lcovMergerFilesToRun;
  private final RunfilesSupplier lcovMergerRunfilesSupplier;

  private static ImmutableSet<Artifact> nonNullAsSet(Artifact... artifacts) {
    ImmutableSet.Builder<Artifact> builder = ImmutableSet.builder();
    for (Artifact artifact : artifacts) {
      if (artifact != null) {
        builder.add(artifact);
      }
    }
    return builder.build();
  }

  /**
   * Create new TestRunnerAction instance. Should not be called directly. Use {@link
   * TestActionBuilder} instead.
   *
   * @param shardNum The shard number. Must be 0 if totalShards == 0 (no sharding). Otherwise, must
   *     be >= 0 and < totalShards.
   * @param runNumber test run number
   */
  TestRunnerAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      SingleRunfilesSupplier runfilesSupplier,
      Artifact testSetupScript, // Must be in inputs
      Artifact testXmlGeneratorScript, // Must be in inputs
      @Nullable Artifact collectCoverageScript, // Must be in inputs, if not null
      Artifact testLog,
      Artifact cacheStatus,
      Artifact coverageArtifact,
      @Nullable Artifact coverageDirectory,
      TestTargetProperties testProperties,
      ActionEnvironment extraTestEnv,
      TestTargetExecutionSettings executionSettings,
      int shardNum,
      int runNumber,
      BuildConfiguration configuration,
      String workspaceName,
      @Nullable PathFragment shExecutable,
      boolean cancelConcurrentTestsOnSuccess,
      Iterable<Artifact> tools,
      boolean splitCoveragePostProcessing,
      NestedSetBuilder<Artifact> lcovMergerFilesToRun,
      RunfilesSupplier lcovMergerRunfilesSupplier) {
    super(
        owner,
        NestedSetBuilder.wrap(Order.STABLE_ORDER, tools),
        inputs,
        runfilesSupplier,
        nonNullAsSet(testLog, cacheStatus, coverageArtifact, coverageDirectory),
        configuration.getActionEnvironment());
    Preconditions.checkState((collectCoverageScript == null) == (coverageArtifact == null));
    this.testSetupScript = testSetupScript;
    this.testXmlGeneratorScript = testXmlGeneratorScript;
    this.collectCoverageScript = collectCoverageScript;
    this.configuration = checkNotNull(configuration);
    this.testConfiguration = checkNotNull(configuration.getFragment(TestConfiguration.class));
    this.testLog = testLog;
    this.cacheStatus = cacheStatus;
    this.coverageData = coverageArtifact;
    this.coverageDirectory = coverageDirectory;
    this.shardNum = shardNum;
    this.runNumber = runNumber;
    this.testProperties = checkNotNull(testProperties);
    this.executionSettings = checkNotNull(executionSettings);

    this.baseDir = cacheStatus.getExecPath().getParentDirectory();

    int totalShards = executionSettings.getTotalShards();
    Preconditions.checkState(
        (totalShards == 0 && shardNum == 0)
            || (totalShards > 0 && 0 <= shardNum && shardNum < totalShards));
    this.testExitSafe = baseDir.getChild("test.exited_prematurely");
    // testShard Path should be set only if sharding is enabled.
    this.testShard = totalShards > 1 ? baseDir.getChild("test.shard") : null;
    this.xmlOutputPath = baseDir.getChild("test.xml");
    this.testWarningsPath = baseDir.getChild("test.warnings");
    this.unusedRunfilesLogPath = baseDir.getChild("test.unused_runfiles_log");
    this.testStderr = baseDir.getChild("test.err");
    this.shExecutable = shExecutable;
    this.splitLogsDir = baseDir.getChild("test.raw_splitlogs");
    // See note in {@link #getSplitLogsPath} on the choice of file name.
    this.splitLogsPath = splitLogsDir.getChild("test.splitlogs");
    this.undeclaredOutputsDir = baseDir.getChild("test.outputs");
    this.undeclaredOutputsZipPath = undeclaredOutputsDir.getChild("outputs.zip");
    this.undeclaredOutputsAnnotationsDir = baseDir.getChild("test.outputs_manifest");
    this.undeclaredOutputsManifestPath = undeclaredOutputsAnnotationsDir.getChild("MANIFEST");
    this.undeclaredOutputsAnnotationsPath = undeclaredOutputsAnnotationsDir.getChild("ANNOTATIONS");
    this.testInfrastructureFailure = baseDir.getChild("test.infrastructure_failure");
    this.workspaceName = workspaceName;

    this.extraTestEnv = extraTestEnv;
    this.requiredClientEnvVariables =
        ImmutableIterable.from(
            Iterables.concat(
                configuration.getActionEnvironment().getInheritedEnv(),
                configuration.getTestActionEnvironment().getInheritedEnv(),
                this.extraTestEnv.getInheritedEnv()));
    this.cancelConcurrentTestsOnSuccess = cancelConcurrentTestsOnSuccess;
    this.splitCoveragePostProcessing = splitCoveragePostProcessing;
    this.lcovMergerFilesToRun = lcovMergerFilesToRun;
    this.lcovMergerRunfilesSupplier = lcovMergerRunfilesSupplier;

    // Mark all possible test outputs for deletion before test execution.
    // TestRunnerAction potentially can create many more non-declared outputs - xml output, coverage
    // data file and logs for failed attempts. All those outputs are uniquely identified by the test
    // log base name with arbitrary prefix and extension.

    // We need to remove *.(xml|data|shard|warnings|zip) files if they are present.
    ImmutableSet.Builder<PathFragment> filesToDeleteBuilder =
        ImmutableSet.<PathFragment>builder()
            .add(
                xmlOutputPath,
                testWarningsPath,
                unusedRunfilesLogPath,
                testStderr,
                testExitSafe,
                testInfrastructureFailure,
                // We cannot use coverageData artifact since it may be null. Generate coverage name
                // instead.
                baseDir.getChild("coverage.dat"),
                baseDir.getChild("test.zip")); // Delete files fetched from remote execution.
    if (testShard != null) {
      filesToDeleteBuilder.add(testShard);
    }
    this.filesToDeleteBeforeExecution = filesToDeleteBuilder.build();
    this.directoriesToDeleteBeforeExecution =
        ImmutableSet.of(
            // Note that splitLogsPath points to a file inside the splitLogsDir so it's not
            // necessary to delete it explicitly.
            splitLogsDir,
            undeclaredOutputsDir,
            undeclaredOutputsAnnotationsDir,
            baseDir.getRelative("test_attempts"));
  }

  public RunfilesSupplier getLcovMergerRunfilesSupplier() {
    return lcovMergerRunfilesSupplier;
  }

  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  public final PathFragment getBaseDir() {
    return baseDir;
  }

  public boolean getSplitCoveragePostProcessing() {
    return splitCoveragePostProcessing;
  }

  public NestedSetBuilder<Artifact> getLcovMergerFilesToRun() {
    return lcovMergerFilesToRun;
  }

  public Artifact getCoverageDirectoryTreeArtifact() {
    return coverageDirectory;
  }

  @Override
  public boolean showsOutputUnconditionally() {
    return true;
  }

  public List<ActionInput> getSpawnOutputs() {
    final List<ActionInput> outputs = new ArrayList<>();
    outputs.add(ActionInputHelper.fromPath(getXmlOutputPath()));
    outputs.add(ActionInputHelper.fromPath(getExitSafeFile()));
    if (isSharded()) {
      outputs.add(ActionInputHelper.fromPath(getTestShard()));
    }
    outputs.add(ActionInputHelper.fromPath(getTestWarningsPath()));
    outputs.add(ActionInputHelper.fromPath(getSplitLogsPath()));
    outputs.add(ActionInputHelper.fromPath(getUnusedRunfilesLogPath()));
    outputs.add(ActionInputHelper.fromPath(getInfrastructureFailureFile()));
    outputs.add(ActionInputHelper.fromPath(getUndeclaredOutputsZipPath()));
    outputs.add(ActionInputHelper.fromPath(getUndeclaredOutputsManifestPath()));
    outputs.add(ActionInputHelper.fromPath(getUndeclaredOutputsAnnotationsPath()));
    if (isCoverageMode()) {
      if (!splitCoveragePostProcessing) {
        outputs.add(coverageData);
      }
      if (coverageDirectory != null) {
        outputs.add(coverageDirectory);
      }
    }
    return outputs;
  }

  /**
   * Returns the list of mappings from file name constants to output files. This method checks the
   * file system for existence of these output files, so it must only be used after test execution.
   */
  // TODO(ulfjack): Instead of going to local disk here, use SpawnResult (add list of files there).
  public ImmutableList<Pair<String, Path>> getTestOutputsMapping(
      ArtifactPathResolver resolver, Path execRoot) {
    ImmutableList.Builder<Pair<String, Path>> builder = ImmutableList.builder();
    if (resolver.toPath(getTestLog()).exists()) {
      builder.add(Pair.of(TestFileNameConstants.TEST_LOG, resolver.toPath(getTestLog())));
    }
    if (getCoverageData() != null && resolver.toPath(getCoverageData()).exists()) {
      builder.add(Pair.of(TestFileNameConstants.TEST_COVERAGE, resolver.toPath(getCoverageData())));
    }
    if (execRoot != null) {
      ResolvedPaths resolvedPaths = resolve(execRoot);
      if (resolvedPaths.getTestStderr().exists()) {
        builder.add(Pair.of(TestFileNameConstants.TEST_STDERR, resolvedPaths.getTestStderr()));
      }
      if (resolvedPaths.getXmlOutputPath().exists()) {
        builder.add(Pair.of(TestFileNameConstants.TEST_XML, resolvedPaths.getXmlOutputPath()));
      }
      if (resolvedPaths.getSplitLogsPath().exists()) {
        builder.add(Pair.of(TestFileNameConstants.SPLIT_LOGS, resolvedPaths.getSplitLogsPath()));
      }
      if (resolvedPaths.getTestWarningsPath().exists()) {
        builder.add(
            Pair.of(TestFileNameConstants.TEST_WARNINGS, resolvedPaths.getTestWarningsPath()));
      }
      if (resolvedPaths.getUndeclaredOutputsZipPath().exists()) {
        builder.add(
            Pair.of(
                TestFileNameConstants.UNDECLARED_OUTPUTS_ZIP,
                resolvedPaths.getUndeclaredOutputsZipPath()));
      }
      if (resolvedPaths.getUndeclaredOutputsManifestPath().exists()) {
        builder.add(
            Pair.of(
                TestFileNameConstants.UNDECLARED_OUTPUTS_MANIFEST,
                resolvedPaths.getUndeclaredOutputsManifestPath()));
      }
      if (resolvedPaths.getUndeclaredOutputsAnnotationsPath().exists()) {
        builder.add(
            Pair.of(
                TestFileNameConstants.UNDECLARED_OUTPUTS_ANNOTATIONS,
                resolvedPaths.getUndeclaredOutputsAnnotationsPath()));
      }
      if (resolvedPaths.getUnusedRunfilesLogPath().exists()) {
        builder.add(
            Pair.of(
                TestFileNameConstants.UNUSED_RUNFILES_LOG,
                resolvedPaths.getUnusedRunfilesLogPath()));
      }
      if (resolvedPaths.getInfrastructureFailureFile().exists()) {
        builder.add(
            Pair.of(
                TestFileNameConstants.TEST_INFRASTRUCTURE_FAILURE,
                resolvedPaths.getInfrastructureFailureFile()));
      }
    }
    return builder.build();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable Artifact.ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    // TODO(b/150305897): use addUUID?
    fp.addString(GUID);
    fp.addIterableStrings(executionSettings.getArgs().arguments());
    fp.addString(Strings.nullToEmpty(executionSettings.getTestFilter()));
    fp.addBoolean(executionSettings.getTestRunnerFailFast());
    RunUnder runUnder = executionSettings.getRunUnder();
    fp.addString(runUnder == null ? "" : runUnder.getValue());
    extraTestEnv.addTo(fp);
    // TODO(ulfjack): It might be better for performance to hash the action and test envs in config,
    // and only add a hash here.
    configuration.getActionEnvironment().addTo(fp);
    configuration.getTestActionEnvironment().addTo(fp);
    // The 'requiredClientEnvVariables' are handled by Skyframe and don't need to be added here.
    fp.addString(testProperties.getSize().toString());
    fp.addString(testProperties.getTimeout().toString());
    fp.addStrings(testProperties.getTags());
    fp.addBoolean(testProperties.isRemotable());
    fp.addInt(shardNum);
    fp.addInt(executionSettings.getTotalShards());
    fp.addInt(runNumber);
    fp.addInt(executionSettings.getTotalRuns());
    fp.addBoolean(configuration.isCodeCoverageEnabled());
    fp.addStringMap(getExecutionInfo());
  }

  @Override
  public boolean executeUnconditionally() {
    // Note: isVolatile must return true if executeUnconditionally can ever return true
    // for this instance.
    if (unconditionalExecution == null) {
      unconditionalExecution = computeExecuteUnconditionallyFromTestStatus();
    }
    return unconditionalExecution;
  }

  @Override // Tighten return type.
  public SingleRunfilesSupplier getRunfilesSupplier() {
    return (SingleRunfilesSupplier) super.getRunfilesSupplier();
  }

  @Override
  public boolean isVolatile() {
    return true;
  }

  /** Saves cache status to disk. */
  public void saveCacheStatus(ActionExecutionContext actionExecutionContext, TestResultData data)
      throws IOException {
    try (OutputStream out = actionExecutionContext.getInputPath(cacheStatus).getOutputStream()) {
      data.writeTo(out);
    }
  }

  /** Returns the cache from disk, or null if the file doesn't exist or if there is an error. */
  @Nullable
  private TestResultData maybeReadCacheStatus() {
    try {
      return readCacheStatus();
    } catch (IOException expected) {
      return null;
    }
  }

  @VisibleForTesting
  TestResultData readCacheStatus() throws IOException {
    try (InputStream in = cacheStatus.getPath().getInputStream()) {
      return TestResultData.parseFrom(in, ExtensionRegistry.getEmptyRegistry());
    }
  }

  private boolean computeExecuteUnconditionallyFromTestStatus() {
    return !canBeCached(
        testConfiguration.cacheTestResults(),
        this::maybeReadCacheStatus,
        testProperties.isExternal(),
        executionSettings.getTotalRuns());
  }

  @VisibleForTesting
  static boolean canBeCached(
      TriState cacheTestResults,
      Supplier<TestResultData> prevStatus, // Lazy evaluation to avoid a disk read if possible.
      boolean isExternal,
      int runsPerTest) {
    if (isExternal || cacheTestResults == TriState.NO) {
      return false;
    }
    if (cacheTestResults == TriState.AUTO && runsPerTest > 1) {
      return false;
    }
    TestResultData status = prevStatus.get();
    if (status != null) {
      if (!status.getCachable()) {
        return false;
      }
      if (cacheTestResults == TriState.AUTO && !status.getTestPassed()) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns whether caching has been deemed safe by looking at the previous test run (for local
   * caching). If the previous run is not present, return "true" here, as remote execution caching
   * should be safe.
   */
  public boolean shouldCacheResult() {
    return !executeUnconditionally();
  }

  @Override
  public void actionCacheHit(ActionCachedContext executor) {
    unconditionalExecution = null;
    try {
      executor
          .getEventHandler()
          .post(
              executor
                  .getContext(TestActionContext.class)
                  .newCachedTestResult(executor.getExecRoot(), this, readCacheStatus()));
    } catch (IOException e) {
      // TODO(b/150311421): Produce a user facing warning/error and a TestResult with information
      //  about the failure to retrieve cached test status.
      LoggingUtil.logToRemote(Level.WARNING, "Failed creating cached protocol buffer", e);
    }
  }

  @Override
  protected String getRawProgressMessage() {
    return "Testing " + getTestName();
  }

  @Override
  protected Iterable<PathFragment> getAdditionalPathOutputsToDelete() {
    return filesToDeleteBeforeExecution;
  }

  @Override
  protected Iterable<PathFragment> getDirectoryOutputsToDelete() {
    return directoriesToDeleteBeforeExecution;
  }

  void createEmptyOutputs(ActionExecutionContext context) throws IOException {
    for (Artifact output : TestRunnerAction.this.getMandatoryOutputs()) {
      FileSystemUtils.touchFile(context.getInputPath(output));
    }
  }

  public void setupEnvVariables(Map<String, String> env, Duration timeout) {
    env.put("TEST_TARGET", Label.print(getOwner().getLabel()));
    env.put("TEST_SIZE", getTestProperties().getSize().toString());
    env.put("TEST_TIMEOUT", Long.toString(timeout.getSeconds()));
    env.put("TEST_WORKSPACE", getRunfilesPrefix());
    env.put(
        "TEST_BINARY",
        getExecutionSettings().getExecutable().getRunfilesPath().getCallablePathString());

    // When we run test multiple times, set different TEST_RANDOM_SEED values for each run.
    // Don't override any previous setting.
    if (executionSettings.getTotalRuns() > 1 && !env.containsKey("TEST_RANDOM_SEED")) {
      env.put("TEST_RANDOM_SEED", Integer.toString(getRunNumber() + 1));
    }

    String testFilter = getExecutionSettings().getTestFilter();
    if (testFilter != null) {
      env.put(TEST_BRIDGE_TEST_FILTER_ENV, testFilter);
    }
    if (testConfiguration.getTestRunnerFailFast()) {
      env.put("TESTBRIDGE_TEST_RUNNER_FAIL_FAST", "1");
    }

    env.put("TEST_WARNINGS_OUTPUT_FILE", getTestWarningsPath().getPathString());
    env.put("TEST_UNUSED_RUNFILES_LOG_FILE", getUnusedRunfilesLogPath().getPathString());

    env.put("TEST_LOGSPLITTER_OUTPUT_FILE", getSplitLogsPath().getPathString());

    env.put("TEST_UNDECLARED_OUTPUTS_ZIP", getUndeclaredOutputsZipPath().getPathString());
    env.put("TEST_UNDECLARED_OUTPUTS_DIR", getUndeclaredOutputsDir().getPathString());
    env.put("TEST_UNDECLARED_OUTPUTS_MANIFEST", getUndeclaredOutputsManifestPath().getPathString());
    env.put(
        "TEST_UNDECLARED_OUTPUTS_ANNOTATIONS",
        getUndeclaredOutputsAnnotationsPath().getPathString());
    env.put(
        "TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR",
        getUndeclaredOutputsAnnotationsDir().getPathString());

    env.put("TEST_PREMATURE_EXIT_FILE", getExitSafeFile().getPathString());
    env.put("TEST_INFRASTRUCTURE_FAILURE_FILE", getInfrastructureFailureFile().getPathString());

    if (isSharded()) {
      env.put("TEST_SHARD_INDEX", Integer.toString(getShardNum()));
      env.put("TEST_TOTAL_SHARDS", Integer.toString(getExecutionSettings().getTotalShards()));
      env.put("TEST_SHARD_STATUS_FILE", getTestShard().getPathString());
    }
    env.put("XML_OUTPUT_FILE", getXmlOutputPath().getPathString());

    if (!isEnableRunfiles()) {
      // If runfiles are disabled, tell remote-runtest.sh/local-runtest.sh about that.
      env.put("RUNFILES_MANIFEST_ONLY", "1");
    }

    if (testProperties.isPersistentTestRunner()) {
      // Let the test runner know it runs persistently within a worker.
      env.put("PERSISTENT_TEST_RUNNER", "true");
    }

    if (isCoverageMode()) {
      // Instruct remote-runtest.sh/local-runtest.sh not to cd into the runfiles directory.
      // TODO(ulfjack): Find a way to avoid setting this variable.
      env.put("RUNTEST_PRESERVE_CWD", "1");

      env.put("COVERAGE_MANIFEST", getCoverageManifest().getExecPathString());
      env.put("COVERAGE_DIR", getCoverageDirectory().getPathString());
      env.put("COVERAGE_OUTPUT_FILE", getCoverageData().getExecPathString());
      env.put("SPLIT_COVERAGE_POST_PROCESSING", splitCoveragePostProcessing ? "1" : "0");
      env.put("IS_COVERAGE_SPAWN", "0");
    }
  }

  /**
   * Gets the test name in a user-friendly format. Will generally include the target name and
   * run/shard numbers, if applicable.
   */
  public String getTestName() {
    String suffix = getTestSuffix();
    String label = Label.print(getOwner().getLabel());
    return suffix.isEmpty() ? label : label + " " + suffix;
  }

  /**
   * Gets the test suffix in a user-friendly format, eg "(shard 1 of 7)". Will include the target
   * name and run/shard numbers, if applicable.
   */
  public String getTestSuffix() {
    int totalShards = executionSettings.getTotalShards();
    // Use a 1-based index for user friendliness.
    int runsPerTest = executionSettings.getTotalRuns();
    if (totalShards > 1 && runsPerTest > 1) {
      return String.format(
          "(shard %d of %d, run %d of %d)", shardNum + 1, totalShards, runNumber + 1, runsPerTest);
    } else if (totalShards > 1) {
      return String.format("(shard %d of %d)", shardNum + 1, totalShards);
    } else if (runsPerTest > 1) {
      return String.format("(run %d of %d)", runNumber + 1, runsPerTest);
    } else {
      return "";
    }
  }

  public Artifact getTestLog() {
    return testLog;
  }

  /** Returns all environment variables which must be set in order to run this test. */
  public ActionEnvironment getExtraTestEnv() {
    return extraTestEnv;
  }

  @Override
  public Iterable<String> getClientEnvironmentVariables() {
    return requiredClientEnvVariables;
  }

  public ResolvedPaths resolve(Path execRoot) {
    return new ResolvedPaths(execRoot);
  }

  public Artifact getCacheStatusArtifact() {
    return cacheStatus;
  }

  public PathFragment getTestWarningsPath() {
    return testWarningsPath;
  }

  public PathFragment getUnusedRunfilesLogPath() {
    return unusedRunfilesLogPath;
  }

  public PathFragment getSplitLogsPath() {
    return splitLogsPath;
  }

  public PathFragment getUndeclaredOutputsDir() {
    return undeclaredOutputsDir;
  }

  /** Returns path to the optional zip file of undeclared test outputs. */
  public PathFragment getUndeclaredOutputsZipPath() {
    return undeclaredOutputsZipPath;
  }

  /** Returns path to the undeclared output manifest file. */
  public PathFragment getUndeclaredOutputsManifestPath() {
    return undeclaredOutputsManifestPath;
  }

  public PathFragment getUndeclaredOutputsAnnotationsDir() {
    return undeclaredOutputsAnnotationsDir;
  }

  /** Returns path to the undeclared output annotations file. */
  public PathFragment getUndeclaredOutputsAnnotationsPath() {
    return undeclaredOutputsAnnotationsPath;
  }

  public PathFragment getTestShard() {
    return testShard;
  }

  public PathFragment getExitSafeFile() {
    return testExitSafe;
  }

  public PathFragment getInfrastructureFailureFile() {
    return testInfrastructureFailure;
  }

  /** Returns path to the optionally created XML output file created by the test. */
  public PathFragment getXmlOutputPath() {
    return xmlOutputPath;
  }

  /** Returns coverage data artifact or null if code coverage was not requested. */
  @Nullable
  public Artifact getCoverageData() {
    return coverageData;
  }

  @Nullable
  public Artifact getCoverageManifest() {
    return getExecutionSettings().getInstrumentedFileManifest();
  }

  /** Returns true if coverage data should be gathered. */
  public boolean isCoverageMode() {
    return coverageData != null;
  }

  /**
   * Returns a directory to temporarily store coverage results for the given action relative to the
   * execution root. This directory is used to store all coverage results related to the test
   * execution with exception of the locally generated *.gcda files. Those are stored separately
   * using relative path within coverage directory.
   *
   * <p>If the coverageDirectory field is set, then its exec path is returned. This is a tree
   * artifact, meaning that all files in the corresponding directories are returned from sandboxed
   * or remote execution.
   *
   * <p>Otherwise, the directory name for the given test runner action is constructed as: {@code
   * _coverage/target_path/test_log_name} where {@code test_log_name} is usually a target name but
   * potentially can include extra suffix, such as a shard number (if test execution was sharded).
   */
  public PathFragment getCoverageDirectory() {
    if (coverageDirectory != null) {
      return coverageDirectory.getExecPath();
    }
    return COVERAGE_TMP_ROOT.getRelative(
        FileSystemUtils.removeExtension(getTestLog().getRootRelativePath()));
  }

  public TestTargetProperties getTestProperties() {
    return testProperties;
  }

  @Override
  public Map<String, String> getExecutionInfo() {
    return testProperties.getExecutionInfo();
  }

  public TestTargetExecutionSettings getExecutionSettings() {
    return executionSettings;
  }

  public boolean isSharded() {
    return testShard != null;
  }

  /**
   * Returns the shard number for this action. If getTotalShards() > 0, must be >= 0 and <
   * getTotalShards(). Otherwise, must be 0.
   */
  public int getShardNum() {
    return shardNum;
  }

  /** Returns run number. */
  public int getRunNumber() {
    return runNumber;
  }

  /** Returns the workspace name. */
  public String getRunfilesPrefix() {
    return workspaceName;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return testLog;
  }

  @Override
  public ActionContinuationOrResult beginExecution(ActionExecutionContext actionExecutionContext)
      throws InterruptedException, ActionExecutionException {
    TestActionContext testActionContext =
        actionExecutionContext.getContext(TestActionContext.class);
    return beginExecution(actionExecutionContext, testActionContext);
  }

  @VisibleForTesting
  public ActionContinuationOrResult beginExecution(
      ActionExecutionContext actionExecutionContext, TestActionContext testActionContext)
      throws InterruptedException, ActionExecutionException {
    try {
      TestRunnerSpawn testRunnerSpawn =
          testActionContext.createTestRunnerSpawn(this, actionExecutionContext);
      ListenableFuture<Void> cancelFuture = null;
      if (cancelConcurrentTestsOnSuccess) {
        cancelFuture = testActionContext.getTestCancelFuture(getOwner(), shardNum);
      }
      TestAttemptContinuation testAttemptContinuation =
          beginIfNotCancelled(testRunnerSpawn, cancelFuture);
      if (testAttemptContinuation == null) {
        testRunnerSpawn.finalizeCancelledTest(ImmutableList.of());
        // We need to create the mandatory output files even if we're not going to run anything.
        createEmptyOutputs(actionExecutionContext);
        return ActionContinuationOrResult.of(ActionResult.create(ImmutableList.of()));
      }
      return new RunAttemptsContinuation(
          this,
          testRunnerSpawn,
          testAttemptContinuation,
          testActionContext.isTestKeepGoing(),
          cancelFuture);
    } catch (ExecException e) {
      throw e.toActionExecutionException(this);
    } catch (IOException e) {
      throw new EnvironmentalExecException(e, Code.TEST_RUNNER_IO_EXCEPTION)
          .toActionExecutionException(this);
    }
  }

  @Nullable
  private static TestAttemptContinuation beginIfNotCancelled(
      TestRunnerSpawn testRunnerSpawn, @Nullable ListenableFuture<Void> cancelFuture)
      throws InterruptedException, IOException {
    if (cancelFuture != null && cancelFuture.isCancelled()) {
      // Don't start another attempt if the action was cancelled. Note that there is a race
      // between checking this and starting the test action. If we loose the race, then we get
      // to cancel the action below when we register a callback with the cancelFuture. Note that
      // cancellation only works with spawn runners supporting async execution, so currently does
      // not work with local execution.
      return null;
    }
    TestAttemptContinuation testAttemptContinuation = testRunnerSpawn.beginExecution();
    if (!testAttemptContinuation.isDone() && cancelFuture != null) {
      cancelFuture.addListener(
          () -> {
            // This is a noop if the future is already done.
            testAttemptContinuation.getFuture().cancel(true);
          },
          MoreExecutors.directExecutor());
    }
    return testAttemptContinuation;
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    TestActionContext context = actionExecutionContext.getContext(TestActionContext.class);
    return execute(actionExecutionContext, context);
  }

  @VisibleForTesting
  public ActionResult execute(
      ActionExecutionContext actionExecutionContext, TestActionContext testActionContext)
      throws ActionExecutionException, InterruptedException {
    try {
      ActionContinuationOrResult continuation =
          beginExecution(actionExecutionContext, testActionContext);
      while (!continuation.isDone()) {
        continuation = continuation.execute();
      }
      return continuation.get();
    } finally {
      unconditionalExecution = null;
    }
  }

  @Override
  public String getMnemonic() {
    return MNEMONIC;
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return getOutputs();
  }

  public Artifact getTestSetupScript() {
    return testSetupScript;
  }

  public Artifact getTestXmlGeneratorScript() {
    return testXmlGeneratorScript;
  }

  @Nullable
  public Artifact getCollectCoverageScript() {
    return collectCoverageScript;
  }

  @Nullable
  public PathFragment getShExecutableMaybe() {
    return shExecutable;
  }

  public ImmutableMap<String, String> getLocalShellEnvironment() {
    return configuration.getLocalShellEnvironment();
  }

  public boolean isEnableRunfiles() {
    return configuration.runfilesEnabled();
  }

  @Override
  public List<String> getArguments() throws CommandLineExpansionException, InterruptedException {
    return TestStrategy.expandedArgsFromAction(this);
  }

  @Override
  public ImmutableMap<String, String> getIncompleteEnvironmentForTesting()
      throws ActionExecutionException {
    return getEnvironment().getFixedEnv().toMap();
  }

  @Override
  public NestedSet<Artifact> getPossibleInputsForTesting() {
    return getInputs();
  }

  /** The same set of paths as the parent test action, resolved against a given exec root. */
  public final class ResolvedPaths {
    private final Path execRoot;

    ResolvedPaths(Path execRoot) {
      this.execRoot = checkNotNull(execRoot);
    }

    private Path getPath(PathFragment relativePath) {
      return execRoot.getRelative(relativePath);
    }

    public final Path getBaseDir() {
      return getPath(baseDir);
    }

    /**
     * In rare cases, error messages will be printed to stderr instead of stdout. The test action is
     * responsible for appending anything in the stderr file to the real test.log.
     */
    public Path getTestStderr() {
      return getPath(testStderr);
    }

    public Path getTestWarningsPath() {
      return getPath(testWarningsPath);
    }

    public Path getSplitLogsPath() {
      return getPath(splitLogsPath);
    }

    public Path getUnusedRunfilesLogPath() {
      return getPath(unusedRunfilesLogPath);
    }

    /** Returns path to the directory containing the split logs (raw and proto file). */
    public Path getSplitLogsDir() {
      return getPath(splitLogsDir);
    }

    /** Returns path to the optional zip file of undeclared test outputs. */
    public Path getUndeclaredOutputsZipPath() {
      return getPath(undeclaredOutputsZipPath);
    }

    /** Returns path to the directory to hold undeclared test outputs. */
    public Path getUndeclaredOutputsDir() {
      return getPath(undeclaredOutputsDir);
    }

    /** Returns path to the directory to hold undeclared output annotations parts. */
    public Path getUndeclaredOutputsAnnotationsDir() {
      return getPath(undeclaredOutputsAnnotationsDir);
    }

    /** Returns path to the undeclared output manifest file. */
    public Path getUndeclaredOutputsManifestPath() {
      return getPath(undeclaredOutputsManifestPath);
    }

    /** Returns path to the undeclared output annotations file. */
    public Path getUndeclaredOutputsAnnotationsPath() {
      return getPath(undeclaredOutputsAnnotationsPath);
    }

    @Nullable
    public Path getTestShard() {
      return testShard == null ? null : getPath(testShard);
    }

    public Path getExitSafeFile() {
      return getPath(testExitSafe);
    }

    public Path getInfrastructureFailureFile() {
      return getPath(testInfrastructureFailure);
    }

    /** Returns path to the optionally created XML output file created by the test. */
    public Path getXmlOutputPath() {
      return getPath(xmlOutputPath);
    }

    public Path getCoverageDirectory() {
      return getPath(TestRunnerAction.this.getCoverageDirectory());
    }

    public Path getCoverageDataPath() {
      return getPath(getCoverageData().getExecPath());
    }
  }

  /** Implements test retries. */
  @VisibleForTesting
  static class RunAttemptsContinuation extends ActionContinuationOrResult {

    private final TestRunnerAction testRunnerAction;
    private final TestRunnerSpawn testRunnerSpawn;
    private final TestAttemptContinuation testContinuation;
    private final boolean keepGoing;
    // Careful: We can only determine this value _after_ the first attempt is done, so we initially
    // set it to 0, but then we need to make sure not to use this value.
    private final int maxAttempts;
    private final List<SpawnResult> spawnResults;
    private final List<FailedAttemptResult> failedAttempts;
    @Nullable private final ListenableFuture<Void> cancelFuture;

    private RunAttemptsContinuation(
        TestRunnerAction testRunnerAction,
        TestRunnerSpawn testRunnerSpawn,
        TestAttemptContinuation testContinuation,
        boolean keepGoing,
        int maxAttempts,
        List<SpawnResult> spawnResults,
        List<FailedAttemptResult> failedAttempts,
        ListenableFuture<Void> cancelFuture) {
      this.testRunnerAction = testRunnerAction;
      this.testRunnerSpawn = testRunnerSpawn;
      this.testContinuation = testContinuation;
      this.keepGoing = keepGoing;
      this.maxAttempts = maxAttempts;
      this.spawnResults = spawnResults;
      this.failedAttempts = failedAttempts;
      this.cancelFuture = cancelFuture;
      if (cancelFuture != null) {
        cancelFuture.addListener(
            () -> {
              // This is a noop if the future is already done.
              testContinuation.getFuture().cancel(true);
            },
            MoreExecutors.directExecutor());
      }
    }

    RunAttemptsContinuation(
        TestRunnerAction testRunnerAction,
        TestRunnerSpawn testRunnerSpawn,
        TestAttemptContinuation testContinuation,
        boolean keepGoing,
        @Nullable ListenableFuture<Void> cancelFuture) {
      this(
          testRunnerAction,
          testRunnerSpawn,
          testContinuation,
          keepGoing,
          0,
          new ArrayList<>(),
          new ArrayList<>(),
          cancelFuture);
    }

    @Nullable
    @Override
    public ListenableFuture<?> getFuture() {
      return testContinuation.getFuture();
    }

    @Override
    public ActionContinuationOrResult execute()
        throws ActionExecutionException, InterruptedException {
      try {
        TestAttemptContinuation nextContinuation;
        try {
          nextContinuation = testContinuation.execute();
        } catch (InterruptedException e) {
          if (cancelFuture != null && cancelFuture.isCancelled()) {
            // Clear the interrupt bit.
            Thread.interrupted();
            testRunnerAction.createEmptyOutputs(testRunnerSpawn.getActionExecutionContext());
            testRunnerSpawn.finalizeCancelledTest(failedAttempts);
            return ActionContinuationOrResult.of(ActionResult.create(spawnResults));
          }
          throw e;
        }
        if (!nextContinuation.isDone()) {
          return new RunAttemptsContinuation(
              testRunnerAction,
              testRunnerSpawn,
              nextContinuation,
              keepGoing,
              maxAttempts,
              spawnResults,
              failedAttempts,
              cancelFuture);
        }

        TestAttemptResult result = nextContinuation.get();
        int actualMaxAttempts =
            failedAttempts.isEmpty() ? testRunnerSpawn.getMaxAttempts(result) : maxAttempts;
        Preconditions.checkState(actualMaxAttempts != 0);
        return process(result, actualMaxAttempts);
      } catch (ExecException e) {
        throw e.toActionExecutionException(this.testRunnerAction);
      } catch (IOException e) {
        throw new EnvironmentalExecException(e, Code.TEST_RUNNER_IO_EXCEPTION)
            .toActionExecutionException(this.testRunnerAction);
      }
    }

    private ActionContinuationOrResult process(TestAttemptResult result, int actualMaxAttempts)
        throws ExecException, IOException, InterruptedException {
      spawnResults.addAll(result.spawnResults());
      TestAttemptResult.Result testResult = result.result();
      if (testResult == TestAttemptResult.Result.PASSED) {
        if (cancelFuture != null) {
          cancelFuture.cancel(true);
        }
      } else {
        TestRunnerSpawnAndMaxAttempts nextRunnerAndAttempts =
            computeNextRunnerAndMaxAttempts(
                testResult, testRunnerSpawn, failedAttempts.size() + 1, actualMaxAttempts);
        if (nextRunnerAndAttempts != null) {
          failedAttempts.add(
              testRunnerSpawn.finalizeFailedTestAttempt(result, failedAttempts.size() + 1));

          TestAttemptContinuation nextContinuation =
              beginIfNotCancelled(nextRunnerAndAttempts.getSpawn(), cancelFuture);
          if (nextContinuation == null) {
            testRunnerSpawn.finalizeCancelledTest(failedAttempts);
            // We need to create the mandatory output files even if we're not going to run anything.
            testRunnerAction.createEmptyOutputs(testRunnerSpawn.getActionExecutionContext());
            return ActionContinuationOrResult.of(ActionResult.create(spawnResults));
          }

          // Change the phase here because we are executing a rerun of the failed attempt.
          this.testRunnerSpawn
              .getActionExecutionContext()
              .getEventHandler()
              .post(new SpawnExecutedEvent.ChangePhase(this.testRunnerAction));

          return new RunAttemptsContinuation(
              testRunnerAction,
              nextRunnerAndAttempts.getSpawn(),
              nextContinuation,
              keepGoing,
              nextRunnerAndAttempts.getMaxAttempts(),
              spawnResults,
              failedAttempts,
              cancelFuture);
        }
      }
      testRunnerSpawn.finalizeTest(result, failedAttempts);

      if (!keepGoing && testResult != TestAttemptResult.Result.PASSED) {
        DetailedExitCode systemFailure = result.primarySystemFailure();
        if (systemFailure != null) {
          throw new TestExecException(
              "Test failed (system error), aborting: "
                  + systemFailure.getFailureDetail().getMessage(),
              systemFailure.getFailureDetail());
        }
        String errorMessage = "Test failed, aborting";
        throw new TestExecException(
            errorMessage,
            FailureDetail.newBuilder()
                .setTestAction(
                    TestAction.newBuilder().setCode(TestAction.Code.NO_KEEP_GOING_TEST_FAILURE))
                .setMessage(errorMessage)
                .build());
      }
      return ActionContinuationOrResult.of(ActionResult.create(spawnResults));
    }

    /**
     * Method used to compute next runner and max attempts. Returns null if there if there is no
     * remaining attempts (including fallback runner).
     */
    @VisibleForTesting
    @Nullable
    static TestRunnerSpawnAndMaxAttempts computeNextRunnerAndMaxAttempts(
        TestAttemptResult.Result result,
        TestRunnerSpawn testRunnerSpawn,
        int numAttempts,
        int maxAttempts)
        throws ExecException, InterruptedException {
      checkState(result != Result.PASSED, "Should not compute retry runner if last result passed");
      if (result.canRetry() && numAttempts < maxAttempts) {
        TestRunnerSpawn nextRunner = testRunnerSpawn.getFlakyRetryRunner();
        if (nextRunner != null) {
          return TestRunnerSpawnAndMaxAttempts.create(nextRunner, maxAttempts);
        }
      } else {
        TestRunnerSpawn nextRunner = testRunnerSpawn.getFallbackRunner();
        if (nextRunner != null) {
          // We only support one level of fallback, in which case maxAttempts gets *added* once. We
          // don't support a different number of max attempts for the fallback strategy.
          return TestRunnerSpawnAndMaxAttempts.create(nextRunner, numAttempts + maxAttempts);
        }
      }
      return null;
    }

    /** Value type used to store computed next runner and max attempts. */
    @AutoValue
    @VisibleForTesting
    abstract static class TestRunnerSpawnAndMaxAttempts {
      public abstract TestRunnerSpawn getSpawn();

      public abstract int getMaxAttempts();

      public static TestRunnerSpawnAndMaxAttempts create(TestRunnerSpawn spawn, int maxAttempts) {
        return new AutoValue_TestRunnerAction_RunAttemptsContinuation_TestRunnerSpawnAndMaxAttempts(
            spawn, maxAttempts);
      }
    }
  }
}
