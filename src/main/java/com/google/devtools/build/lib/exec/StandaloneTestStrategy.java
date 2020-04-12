// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction.ResolvedPaths;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestResult.ExecutionInfo;
import com.google.devtools.build.lib.buildeventstream.TestFileNameConstants;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Runs TestRunnerAction actions. */
// TODO(bazel-team): add tests for this strategy.
public class StandaloneTestStrategy extends TestStrategy {
  private static final ImmutableMap<String, String> ENV_VARS =
      ImmutableMap.<String, String>builder()
          .put("TZ", "UTC")
          .put("TEST_SRCDIR", TestPolicy.RUNFILES_DIR)
          // TODO(lberki): Remove JAVA_RUNFILES and PYTHON_RUNFILES.
          .put("JAVA_RUNFILES", TestPolicy.RUNFILES_DIR)
          .put("PYTHON_RUNFILES", TestPolicy.RUNFILES_DIR)
          .put("RUNFILES_DIR", TestPolicy.RUNFILES_DIR)
          .put("TEST_TMPDIR", TestPolicy.TEST_TMP_DIR)
          .put("RUN_UNDER_RUNFILES", "1")
          .build();

  public static final TestPolicy DEFAULT_LOCAL_POLICY = new TestPolicy(ENV_VARS);

  protected final Path tmpDirRoot;

  public StandaloneTestStrategy(
      ExecutionOptions executionOptions, BinTools binTools, Path tmpDirRoot) {
    super(executionOptions, binTools);
    this.tmpDirRoot = tmpDirRoot;
  }

  @Override
  public TestRunnerSpawn createTestRunnerSpawn(
      TestRunnerAction action, ActionExecutionContext actionExecutionContext) throws ExecException {
    if (action.getExecutionSettings().getInputManifest() == null) {
      throw new TestExecException("cannot run local tests with --nobuild_runfile_manifests");
    }
    Path execRoot = actionExecutionContext.getExecRoot();
    ArtifactPathResolver pathResolver = actionExecutionContext.getPathResolver();
    Path runfilesDir = pathResolver.convertPath(action.getExecutionSettings().getRunfilesDir());
    Path tmpDir = pathResolver.convertPath(tmpDirRoot.getChild(TestStrategy.getTmpDirName(action)));
    Map<String, String> env = setupEnvironment(
        action, actionExecutionContext.getClientEnv(), execRoot, runfilesDir, tmpDir);
    if (executionOptions.splitXmlGeneration) {
      env.put("EXPERIMENTAL_SPLIT_XML_GENERATION", "1");
    }
    Path workingDirectory = runfilesDir.getRelative(action.getRunfilesPrefix());

    Map<String, String> executionInfo =
        new TreeMap<>(action.getTestProperties().getExecutionInfo());
    if (!action.shouldCacheResult()) {
      executionInfo.put(ExecutionRequirements.NO_CACHE, "");
    }
    executionInfo.put(ExecutionRequirements.TIMEOUT, "" + getTimeout(action).getSeconds());
    if (action.getConfiguration().getFragment(TestConfiguration.class).isPersistentTestRunner()) {
      executionInfo.put(ExecutionRequirements.SUPPORTS_WORKERS, "1");
    }

    ResourceSet localResourceUsage =
        action
            .getTestProperties()
            .getLocalResourceUsage(
                action.getOwner().getLabel(), executionOptions.usingLocalTestJobs());

    Spawn spawn =
        new SimpleSpawn(
            action,
            getArgs(action),
            ImmutableMap.copyOf(env),
            ImmutableMap.copyOf(executionInfo),
            action.getRunfilesSupplier(),
            ImmutableMap.of(),
            /*inputs=*/ action.getInputs(),
            action.getConfiguration().getFragment(TestConfiguration.class).isPersistentTestRunner()
                ? action.getTools()
                : NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.copyOf(action.getSpawnOutputs()),
            localResourceUsage);
    return new StandaloneTestRunnerSpawn(
        action, actionExecutionContext, spawn, tmpDir, workingDirectory, execRoot);
  }

  private static ImmutableList<Pair<String, Path>> renameOutputs(
      ActionExecutionContext actionExecutionContext,
      TestRunnerAction action,
      ImmutableList<Pair<String, Path>> testOutputs,
      int attemptId)
      throws IOException {
    // Rename outputs
    String namePrefix =
        FileSystemUtils.removeExtension(action.getTestLog().getExecPath().getBaseName());
    Path testRoot = actionExecutionContext.getInputPath(action.getTestLog()).getParentDirectory();
    Path attemptsDir = testRoot.getChild(namePrefix + "_attempts");
    attemptsDir.createDirectory();
    String attemptPrefix = "attempt_" + attemptId;
    Path testLog = attemptsDir.getChild(attemptPrefix + ".log");

    // Get the normal test output paths, and then update them to use "attempt_N" names, and
    // attemptDir, before adding them to the outputs.
    ImmutableList.Builder<Pair<String, Path>> testOutputsBuilder = new ImmutableList.Builder<>();
    for (Pair<String, Path> testOutput : testOutputs) {
      // e.g. /testRoot/test.dir/file, an example we follow throughout this loop's comments.
      Path testOutputPath = testOutput.getSecond();
      Path destinationPath;
      if (testOutput.getFirst().equals(TestFileNameConstants.TEST_LOG)) {
        // The rename rules for the test log are different than for all the other files.
        destinationPath = testLog;
      } else {
        // e.g. test.dir/file
        PathFragment relativeToTestDirectory = testOutputPath.relativeTo(testRoot);

        // e.g. attempt_1.dir/file
        String destinationPathFragmentStr =
            relativeToTestDirectory.getSafePathString().replaceFirst("test", attemptPrefix);
        PathFragment destinationPathFragment = PathFragment.create(destinationPathFragmentStr);

        // e.g. /attemptsDir/attempt_1.dir/file
        destinationPath = attemptsDir.getRelative(destinationPathFragment);
        destinationPath.getParentDirectory().createDirectory();
      }

      // Move to the destination.
      testOutputPath.renameTo(destinationPath);

      testOutputsBuilder.add(Pair.of(testOutput.getFirst(), destinationPath));
    }
    return testOutputsBuilder.build();
  }

  private StandaloneFailedAttemptResult processFailedTestAttempt(
      int attemptId,
      ActionExecutionContext actionExecutionContext,
      TestRunnerAction action,
      StandaloneTestResult result)
      throws IOException {
    return processTestAttempt(
        attemptId, /*isLastAttempt=*/ false, actionExecutionContext, action, result);
  }

  private void finalizeTest(
      TestRunnerAction action,
      ActionExecutionContext actionExecutionContext,
      StandaloneTestResult standaloneTestResult,
      List<FailedAttemptResult> failedAttempts)
      throws IOException {
    processTestAttempt(
        failedAttempts.size() + 1,
        /*isLastAttempt=*/ true,
        actionExecutionContext,
        action,
        standaloneTestResult);

    TestResultData.Builder dataBuilder = standaloneTestResult.testResultDataBuilder();
    for (FailedAttemptResult failedAttempt : failedAttempts) {
      TestResultData failedAttemptData =
          ((StandaloneFailedAttemptResult) failedAttempt).testResultData;
      dataBuilder.addAllFailedLogs(failedAttemptData.getFailedLogsList());
      dataBuilder.addTestTimes(failedAttemptData.getTestTimes(0));
      dataBuilder.addAllTestProcessTimes(failedAttemptData.getTestProcessTimesList());
    }
    if (dataBuilder.getStatus() == BlazeTestStatus.PASSED && !failedAttempts.isEmpty()) {
      dataBuilder.setStatus(BlazeTestStatus.FLAKY);
    }
    TestResultData data = dataBuilder.build();
    TestResult result = new TestResult(action, data, false);
    postTestResult(actionExecutionContext, result);
  }

  private StandaloneFailedAttemptResult processTestAttempt(
      int attemptId,
      boolean isLastAttempt,
      ActionExecutionContext actionExecutionContext,
      TestRunnerAction action,
      StandaloneTestResult result)
      throws IOException {
    ImmutableList<Pair<String, Path>> testOutputs =
        action.getTestOutputsMapping(
            actionExecutionContext.getPathResolver(), actionExecutionContext.getExecRoot());
    if (!isLastAttempt) {
      testOutputs = renameOutputs(actionExecutionContext, action, testOutputs, attemptId);
    }

    // Recover the test log path, which may have been renamed, and add it to the data builder.
    Path renamedTestLog = null;
    for (Pair<String, Path> pair : testOutputs) {
      if (TestFileNameConstants.TEST_LOG.equals(pair.getFirst())) {
        renamedTestLog = pair.getSecond();
      }
    }

    TestResultData.Builder dataBuilder = result.testResultDataBuilder();
    if (dataBuilder.getStatus() == BlazeTestStatus.PASSED) {
      dataBuilder.setPassedLog(renamedTestLog.toString());
    } else if (dataBuilder.getStatus() == BlazeTestStatus.INCOMPLETE) {
      // Incomplete (cancelled) test runs don't have a log.
      Preconditions.checkState(renamedTestLog == null);
    } else {
      dataBuilder.addFailedLogs(renamedTestLog.toString());
    }

    // Add the test log to the output
    TestResultData data = dataBuilder.build();
    actionExecutionContext
        .getEventHandler()
        .post(
            TestAttempt.forExecutedTestResult(
                action, data, attemptId, testOutputs, result.executionInfo(), isLastAttempt));
    processTestOutput(actionExecutionContext, data, action.getTestName(), renamedTestLog);
    return new StandaloneFailedAttemptResult(data);
  }

  private Map<String, String> setupEnvironment(
      TestRunnerAction action, Map<String, String> clientEnv, Path execRoot, Path runfilesDir,
      Path tmpDir) {
    PathFragment relativeTmpDir;
    if (tmpDir.startsWith(execRoot)) {
      relativeTmpDir = tmpDir.relativeTo(execRoot);
    } else {
      relativeTmpDir = tmpDir.asFragment();
    }
    return DEFAULT_LOCAL_POLICY.computeTestEnvironment(
        action,
        clientEnv,
        getTimeout(action),
        runfilesDir.relativeTo(execRoot),
        relativeTmpDir);
  }

  private TestAttemptContinuation beginTestAttempt(
      TestRunnerAction testAction,
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      Path execRoot)
      throws IOException, InterruptedException {
    ResolvedPaths resolvedPaths = testAction.resolve(execRoot);
    Path out = actionExecutionContext.getInputPath(testAction.getTestLog());
    Path err = resolvedPaths.getTestStderr();
    FileOutErr testOutErr = new FileOutErr(out, err);
    Closeable streamed = null;
    if (executionOptions.testOutput.equals(ExecutionOptions.TestOutputFormat.STREAMED)) {
      streamed =
          createStreamedTestOutput(
              Reporter.outErrForReporter(actionExecutionContext.getEventHandler()), out);
    }
    long startTimeMillis = actionExecutionContext.getClock().currentTimeMillis();
    SpawnContinuation spawnContinuation =
        actionExecutionContext
            .getContext(SpawnStrategy.class)
            .beginExecution(spawn, actionExecutionContext.withFileOutErr(testOutErr));
    return new BazelTestAttemptContinuation(
        testAction,
        actionExecutionContext,
        spawn,
        resolvedPaths,
        testOutErr,
        streamed,
        startTimeMillis,
        spawnContinuation);
  }

  /** In rare cases, we might write something to stderr. Append it to the real test.log. */
  private static void appendStderr(FileOutErr outErr) throws IOException {
    Path stdErr = outErr.getErrorPath();
    FileStatus stat = stdErr.statNullable();
    if (stat != null) {
      try {
        if (stat.getSize() > 0) {
          Path stdOut = outErr.getOutputPath();
          if (stdOut.exists()) {
            stdOut.setWritable(true);
          }
          try (OutputStream out = stdOut.getOutputStream(true);
              InputStream in = stdErr.getInputStream()) {
            ByteStreams.copy(in, out);
          }
        }
      } finally {
        stdErr.delete();
      }
    }
  }

  private static BuildEventStreamProtos.TestResult.ExecutionInfo extractExecutionInfo(
      SpawnResult spawnResult, TestResultData.Builder result) {
    BuildEventStreamProtos.TestResult.ExecutionInfo.Builder executionInfo =
        BuildEventStreamProtos.TestResult.ExecutionInfo.newBuilder();
    if (spawnResult.isCacheHit()) {
      result.setRemotelyCached(true);
      executionInfo.setCachedRemotely(true);
    }
    String strategy = spawnResult.getRunnerName();
    if (strategy != null) {
      executionInfo.setStrategy(strategy);
      result.setIsRemoteStrategy(strategy.equals("remote"));
    }
    if (spawnResult.getExecutorHostName() != null) {
      executionInfo.setHostname(spawnResult.getExecutorHostName());
    }
    return executionInfo.build();
  }

  /**
   * A spawn to generate a test.xml file from the test log. This is only used if the test does not
   * generate a test.xml file itself.
   */
  private static Spawn createXmlGeneratingSpawn(TestRunnerAction action, SpawnResult result) {
    ImmutableList<String> args =
        ImmutableList.of(
            action.getTestXmlGeneratorScript().getExecPath().getCallablePathString(),
            action.getTestLog().getExecPathString(),
            action.getXmlOutputPath().getPathString(),
            Long.toString(result.getWallTime().orElse(Duration.ZERO).getSeconds()),
            Integer.toString(result.exitCode()));

    String testBinaryName =
        action.getExecutionSettings().getExecutable().getRootRelativePath().getCallablePathString();
    return new SimpleSpawn(
        action,
        args,
        ImmutableMap.of(
            "PATH", "/usr/bin:/bin",
            "TEST_SHARD_INDEX", Integer.toString(action.getShardNum()),
            "TEST_TOTAL_SHARDS", Integer.toString(action.getExecutionSettings().getTotalShards()),
            "TEST_NAME", action.getTestName(),
            "TEST_BINARY", testBinaryName),
        // Pass the execution info of the action which is identical to the supported tags set on the
        // test target. In particular, this does not set the test timeout on the spawn.
        ImmutableMap.copyOf(action.getExecutionInfo()),
        null,
        ImmutableMap.of(),
        /*inputs=*/ NestedSetBuilder.create(
            Order.STABLE_ORDER, action.getTestXmlGeneratorScript(), action.getTestLog()),
        /*tools=*/ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /*outputs=*/ ImmutableSet.of(ActionInputHelper.fromPath(action.getXmlOutputPath())),
        SpawnAction.DEFAULT_RESOURCE_SET);
  }

  @Override
  public TestResult newCachedTestResult(
      Path execRoot, TestRunnerAction action, TestResultData data) {
    return new TestResult(action, data, /*cached*/ true, execRoot);
  }

  private static final class StandaloneFailedAttemptResult implements FailedAttemptResult {
    private final TestResultData testResultData;

    StandaloneFailedAttemptResult(TestResultData testResultData) {
      this.testResultData = testResultData;
    }
  }

  private final class StandaloneTestRunnerSpawn implements TestRunnerSpawn {
    private final TestRunnerAction testAction;
    private final ActionExecutionContext actionExecutionContext;
    private final Spawn spawn;
    private final Path tmpDir;
    private final Path workingDirectory;
    private final Path execRoot;

    StandaloneTestRunnerSpawn(
        TestRunnerAction testAction,
        ActionExecutionContext actionExecutionContext,
        Spawn spawn,
        Path tmpDir,
        Path workingDirectory,
        Path execRoot) {
      this.testAction = testAction;
      this.actionExecutionContext = actionExecutionContext;
      this.spawn = spawn;
      this.tmpDir = tmpDir;
      this.workingDirectory = workingDirectory;
      this.execRoot = execRoot;
    }

    @Override
    public ActionExecutionContext getActionExecutionContext() {
      return actionExecutionContext;
    }

    @Override
    public TestAttemptContinuation beginExecution() throws InterruptedException, IOException {
      prepareFileSystem(testAction, execRoot, tmpDir, workingDirectory);
      return beginTestAttempt(testAction, spawn, actionExecutionContext, execRoot);
    }

    @Override
    public int getMaxAttempts(TestAttemptResult firstTestAttemptResult) {
      return getTestAttempts(testAction);
    }

    @Override
    public FailedAttemptResult finalizeFailedTestAttempt(
        TestAttemptResult testAttemptResult, int attempt) throws IOException {
      return processFailedTestAttempt(
          attempt, actionExecutionContext, testAction, (StandaloneTestResult) testAttemptResult);
    }

    @Override
    public void finalizeTest(
        TestAttemptResult finalResult, List<FailedAttemptResult> failedAttempts)
        throws IOException {
      StandaloneTestStrategy.this.finalizeTest(
          testAction, actionExecutionContext, (StandaloneTestResult) finalResult, failedAttempts);
    }

    @Override
    public void finalizeCancelledTest(List<FailedAttemptResult> failedAttempts) throws IOException {
      TestResultData.Builder builder =
          TestResultData.newBuilder().setTestPassed(false).setStatus(BlazeTestStatus.INCOMPLETE);
      StandaloneTestResult standaloneTestResult =
          StandaloneTestResult.builder()
              .setSpawnResults(ImmutableList.of())
              .setTestResultDataBuilder(builder)
              .setExecutionInfo(ExecutionInfo.getDefaultInstance())
              .build();
      finalizeTest(standaloneTestResult, failedAttempts);
    }
  }

  private final class BazelTestAttemptContinuation extends TestAttemptContinuation {
    private final TestRunnerAction testAction;
    private final ActionExecutionContext actionExecutionContext;
    private final Spawn spawn;
    private final ResolvedPaths resolvedPaths;
    private final FileOutErr fileOutErr;
    private final Closeable streamed;
    private final long startTimeMillis;
    private final SpawnContinuation spawnContinuation;

    BazelTestAttemptContinuation(
        TestRunnerAction testAction,
        ActionExecutionContext actionExecutionContext,
        Spawn spawn,
        ResolvedPaths resolvedPaths,
        FileOutErr fileOutErr,
        Closeable streamed,
        long startTimeMillis,
        SpawnContinuation spawnContinuation) {
      this.testAction = testAction;
      this.actionExecutionContext = actionExecutionContext;
      this.spawn = spawn;
      this.resolvedPaths = resolvedPaths;
      this.fileOutErr = fileOutErr;
      this.streamed = streamed;
      this.startTimeMillis = startTimeMillis;
      this.spawnContinuation = spawnContinuation;
    }

    @Nullable
    @Override
    public ListenableFuture<?> getFuture() {
      return spawnContinuation.getFuture();
    }

    @Override
    public TestAttemptContinuation execute() throws InterruptedException, ExecException {
      // We have two protos to represent test attempts:
      // 1. com.google.devtools.build.lib.view.test.TestStatus.TestResultData represents both failed
      //    attempts and finished tests. Bazel stores this to disk to persist cached test result
      //    information across server restarts.
      // 2. com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestResult
      //    represents only individual attempts (failed or not). Bazel reports this as an event to
      //    the Build Event Protocol, but never saves it to disk.
      //
      // The TestResult proto is always constructed from a TestResultData instance, either one that
      // is created right here, or one that is read back from disk.
      TestResultData.Builder builder;
      ImmutableList<SpawnResult> spawnResults;
      try {
        SpawnContinuation nextContinuation = spawnContinuation.execute();
        if (!nextContinuation.isDone()) {
          return new BazelTestAttemptContinuation(
              testAction,
              actionExecutionContext,
              spawn,
              resolvedPaths,
              fileOutErr,
              streamed,
              startTimeMillis,
              nextContinuation);
        }
        spawnResults = nextContinuation.get();
        builder = TestResultData.newBuilder();
        builder.setTestPassed(true).setStatus(BlazeTestStatus.PASSED);
      } catch (SpawnExecException e) {
        if (e.isCatastrophic()) {
          closeSuppressed(e, streamed);
          closeSuppressed(e, fileOutErr);
          throw e;
        }
        if (!e.getSpawnResult().setupSuccess()) {
          closeSuppressed(e, streamed);
          closeSuppressed(e, fileOutErr);
          // Rethrow as the test could not be run and thus there's no point in retrying.
          throw e;
        }
        spawnResults = ImmutableList.of(e.getSpawnResult());
        builder = TestResultData.newBuilder();
        builder
            .setTestPassed(false)
            .setStatus(e.hasTimedOut() ? BlazeTestStatus.TIMEOUT : BlazeTestStatus.FAILED);
      }
      long endTimeMillis = actionExecutionContext.getClock().currentTimeMillis();

      try {
        if (!fileOutErr.hasRecordedOutput()) {
          // Make sure that the test.log exists.
          FileSystemUtils.touchFile(fileOutErr.getOutputPath());
        }
        // Append any error output to the test.log. This is very rare.
        appendStderr(fileOutErr);
        fileOutErr.close();
        if (streamed != null) {
          streamed.close();
        }
      } catch (IOException e) {
        throw new EnvironmentalExecException(e);
      }

      // SpawnActionContext guarantees the first entry to correspond to the spawn passed in (there
      // may be additional entries due to tree artifact handling).
      SpawnResult primaryResult = spawnResults.get(0);

      // The SpawnResult of a remotely cached or remotely executed action may not have walltime
      // set. We fall back to the time measured here for backwards compatibility.
      long durationMillis = endTimeMillis - startTimeMillis;
      durationMillis =
          primaryResult.getWallTime().orElse(Duration.ofMillis(durationMillis)).toMillis();

      builder.setStartTimeMillisEpoch(startTimeMillis);
      builder.addTestTimes(durationMillis);
      builder.addTestProcessTimes(durationMillis);
      builder.setRunDurationMillis(durationMillis);
      if (testAction.isCoverageMode()) {
        builder.setHasCoverage(true);
      }

      // If the test did not create a test.xml, and --experimental_split_xml_generation is enabled,
      // then we run a separate action to create a test.xml from test.log. We do this as a spawn
      // rather than doing it locally in-process, as the test.log file may only exist remotely (when
      // remote execution is enabled), and we do not want to have to download it.
      Path xmlOutputPath = resolvedPaths.getXmlOutputPath();
      if (executionOptions.splitXmlGeneration
          && fileOutErr.getOutputPath().exists()
          && !xmlOutputPath.exists()) {
        Spawn xmlGeneratingSpawn = createXmlGeneratingSpawn(testAction, primaryResult);
        SpawnStrategy strategy = actionExecutionContext.getContext(SpawnStrategy.class);
        // We treat all failures to generate the test.xml here as catastrophic, and won't rerun
        // the test if this fails. We redirect the output to a temporary file.
        FileOutErr xmlSpawnOutErr = actionExecutionContext.getFileOutErr().childOutErr();
        try {
          SpawnContinuation xmlContinuation =
              strategy.beginExecution(
                  xmlGeneratingSpawn, actionExecutionContext.withFileOutErr(xmlSpawnOutErr));
          return new BazelXmlCreationContinuation(
              resolvedPaths, xmlSpawnOutErr, builder, spawnResults, xmlContinuation);
        } catch (InterruptedException e) {
          closeSuppressed(e, xmlSpawnOutErr);
          throw e;
        }
      }

      TestCase details = parseTestResult(xmlOutputPath);
      if (details != null) {
        builder.setTestCase(details);
      }

      BuildEventStreamProtos.TestResult.ExecutionInfo executionInfo =
          extractExecutionInfo(primaryResult, builder);
      StandaloneTestResult standaloneTestResult =
          StandaloneTestResult.builder()
              .setSpawnResults(spawnResults)
              // We return the TestResultData.Builder rather than the finished TestResultData
              // instance, as we may have to rename the output files in case the test needs to be
              // rerun (if it failed here _and_ is marked flaky _and_ the number of flaky attempts
              // is larger than 1).
              .setTestResultDataBuilder(builder)
              .setExecutionInfo(executionInfo)
              .build();
      return TestAttemptContinuation.of(standaloneTestResult);
    }
  }

  private final class BazelXmlCreationContinuation extends TestAttemptContinuation {
    private final ResolvedPaths resolvedPaths;
    private final FileOutErr fileOutErr;
    private final TestResultData.Builder builder;
    private final List<SpawnResult> primarySpawnResults;
    private final SpawnContinuation spawnContinuation;

    BazelXmlCreationContinuation(
        ResolvedPaths resolvedPaths,
        FileOutErr fileOutErr,
        TestResultData.Builder builder,
        List<SpawnResult> primarySpawnResults,
        SpawnContinuation spawnContinuation) {
      this.resolvedPaths = resolvedPaths;
      this.fileOutErr = fileOutErr;
      this.builder = builder;
      this.primarySpawnResults = primarySpawnResults;
      this.spawnContinuation = spawnContinuation;
    }

    @Nullable
    @Override
    public ListenableFuture<?> getFuture() {
      return spawnContinuation.getFuture();
    }

    @Override
    public TestAttemptContinuation execute() throws InterruptedException, ExecException {
      SpawnContinuation nextContinuation;
      try {
        nextContinuation = spawnContinuation.execute();
        if (!nextContinuation.isDone()) {
          return new BazelXmlCreationContinuation(
              resolvedPaths, fileOutErr, builder, primarySpawnResults, nextContinuation);
        }
      } catch (ExecException | InterruptedException e) {
        closeSuppressed(e, fileOutErr);
        throw e;
      }

      ImmutableList.Builder<SpawnResult> spawnResults = ImmutableList.builder();
      spawnResults.addAll(primarySpawnResults);
      spawnResults.addAll(nextContinuation.get());

      Path xmlOutputPath = resolvedPaths.getXmlOutputPath();
      TestCase details = parseTestResult(xmlOutputPath);
      if (details != null) {
        builder.setTestCase(details);
      }

      BuildEventStreamProtos.TestResult.ExecutionInfo executionInfo =
          extractExecutionInfo(primarySpawnResults.get(0), builder);
      StandaloneTestResult standaloneTestResult =
          StandaloneTestResult.builder()
              .setSpawnResults(spawnResults.build())
              // We return the TestResultData.Builder rather than the finished TestResultData
              // instance, as we may have to rename the output files in case the test needs to be
              // rerun (if it failed here _and_ is marked flaky _and_ the number of flaky attempts
              // is larger than 1).
              .setTestResultDataBuilder(builder)
              .setExecutionInfo(executionInfo)
              .build();
      return TestAttemptContinuation.of(standaloneTestResult);
    }
  }
}
