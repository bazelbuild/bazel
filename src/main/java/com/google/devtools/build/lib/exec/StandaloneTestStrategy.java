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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction.ResolvedPaths;
import com.google.devtools.build.lib.buildeventstream.TestFileNameConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.rules.test.TestAttempt;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData.Builder;
import java.io.Closeable;
import java.io.IOException;
import java.util.Map;

/** Runs TestRunnerAction actions. */
@ExecutionStrategy(
  contextType = TestActionContext.class,
  name = {"standalone"}
)
public class StandaloneTestStrategy extends TestStrategy {
  // TODO(bazel-team) - add tests for this strategy.
  public static final String COLLECT_COVERAGE =
      "external/bazel_tools/tools/test/collect_coverage.sh";

  private static final ImmutableMap<String, String> ENV_VARS =
      ImmutableMap.<String, String>builder()
          .put("TZ", "UTC")
          .put("USER", TestPolicy.SYSTEM_USER_NAME)
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
  public void exec(TestRunnerAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    Path execRoot = actionExecutionContext.getExecRoot();
    Path coverageDir = execRoot.getRelative(action.getCoverageDirectory());
    Path runfilesDir =
        getLocalRunfilesDirectory(
            action,
            actionExecutionContext,
            binTools,
            action.getLocalShellEnvironment(),
            action.isEnableRunfiles());
    Path tmpDir =
        tmpDirRoot.getChild(
            getTmpDirName(
                action.getExecutionSettings().getExecutable().getExecPath(),
                action.getShardNum(),
                action.getRunNumber()));
    Map<String, String> env = setupEnvironment(
        action, actionExecutionContext.getClientEnv(), execRoot, runfilesDir, tmpDir);
    Path workingDirectory = runfilesDir.getRelative(action.getRunfilesPrefix());

    ResolvedPaths resolvedPaths = action.resolve(execRoot);

    ImmutableMap.Builder<String, String> executionInfo = ImmutableMap.builder();
    if (!action.shouldCacheResult()) {
      executionInfo.put(ExecutionRequirements.NO_CACHE, "");
    }
    // This key is only understood by StandaloneSpawnStrategy.
    executionInfo.put("timeout", "" + getTimeout(action).getSeconds());
    executionInfo.putAll(action.getTestProperties().getExecutionInfo());

    ResourceSet localResourceUsage =
        action
            .getTestProperties()
            .getLocalResourceUsage(
                action.getOwner().getLabel(), executionOptions.usingLocalTestJobs());

    Spawn spawn =
        new SimpleSpawn(
            action,
            getArgs(COLLECT_COVERAGE, action),
            ImmutableMap.copyOf(env),
            executionInfo.build(),
            new RunfilesSupplierImpl(
                runfilesDir.relativeTo(execRoot), action.getExecutionSettings().getRunfiles()),
            /*inputs=*/ ImmutableList.copyOf(action.getInputs()),
            /*tools=*/ ImmutableList.<Artifact>of(),
            /*filesetManifests=*/ ImmutableList.<Artifact>of(),
            ImmutableList.copyOf(action.getSpawnOutputs()),
            localResourceUsage);

    TestResultData.Builder dataBuilder = TestResultData.newBuilder();

    try {
      int maxAttempts = getTestAttempts(action);
      TestResultData data =
          executeTestAttempt(
              action,
              spawn,
              actionExecutionContext,
              execRoot,
              coverageDir,
              tmpDir,
              workingDirectory);
      int attempt;
      for (attempt = 1;
          data.getStatus() != BlazeTestStatus.PASSED && attempt < maxAttempts;
          attempt++) {
        processFailedTestAttempt(
            attempt, actionExecutionContext, action, dataBuilder, data);
        data =
            executeTestAttempt(
                action,
                spawn,
                actionExecutionContext,
                execRoot,
                coverageDir,
                tmpDir,
                workingDirectory);
      }
      processLastTestAttempt(attempt, dataBuilder, data);
      ImmutableList.Builder<Pair<String, Path>> testOutputsBuilder = new ImmutableList.Builder<>();
      if (action.getTestLog().getPath().exists()) {
        testOutputsBuilder.add(
            Pair.of(TestFileNameConstants.TEST_LOG, action.getTestLog().getPath()));
      }
      testOutputsBuilder.addAll(TestResult.testOutputsFromPaths(resolvedPaths));
      actionExecutionContext
          .getEventBus()
          .post(
              new TestAttempt(
                  action,
                  attempt,
                  data.getStatus(),
                  data.getStartTimeMillisEpoch(),
                  data.getRunDurationMillis(),
                  testOutputsBuilder.build(),
                  data.getWarningList(),
                  true));
      finalizeTest(actionExecutionContext, action, dataBuilder.build());
    } catch (IOException e) {
      actionExecutionContext.getEventHandler().handle(Event.error("Caught I/O exception: " + e));
      throw new EnvironmentalExecException("unexpected I/O exception", e);
    }
  }

  private void processFailedTestAttempt(
      int attempt,
      ActionExecutionContext actionExecutionContext,
      TestRunnerAction action,
      Builder dataBuilder,
      TestResultData data)
      throws IOException {
    ImmutableList.Builder<Pair<String, Path>> testOutputsBuilder = new ImmutableList.Builder<>();
    // Rename outputs
    String namePrefix =
        FileSystemUtils.removeExtension(action.getTestLog().getExecPath().getBaseName());
    Path testRoot = action.getTestLog().getPath().getParentDirectory();
    Path attemptsDir = testRoot.getChild(namePrefix + "_attempts");
    attemptsDir.createDirectory();
    String attemptPrefix = "attempt_" + attempt;
    Path testLog = attemptsDir.getChild(attemptPrefix + ".log");
    if (action.getTestLog().getPath().exists()) {
      action.getTestLog().getPath().renameTo(testLog);
      testOutputsBuilder.add(Pair.of(TestFileNameConstants.TEST_LOG, testLog));
    }

    // Get the normal test output paths, and then update them to use "attempt_N" names, and
    // attemptDir, before adding them to the outputs.
    ResolvedPaths resolvedPaths = action.resolve(actionExecutionContext.getExecRoot());
    ImmutableList<Pair<String, Path>> testOutputs = TestResult.testOutputsFromPaths(resolvedPaths);
    for (Pair<String, Path> testOutput : testOutputs) {
      // e.g. /testRoot/test.dir/file, an example we follow throughout this loop's comments.
      Path testOutputPath = testOutput.getSecond();

      // e.g. test.dir/file
      PathFragment relativeToTestDirectory = testOutputPath.relativeTo(testRoot);
      
      // e.g. attempt_1.dir/file
      String destinationPathFragmentStr =
          relativeToTestDirectory.getSafePathString().replaceFirst("test", attemptPrefix);
      PathFragment destinationPathFragment = PathFragment.create(destinationPathFragmentStr);
      
      // e.g. /attemptsDir/attempt_1.dir/file
      Path destinationPath = attemptsDir.getRelative(destinationPathFragment);
      destinationPath.getParentDirectory().createDirectory();

      // Copy to the destination.
      testOutputPath.renameTo(destinationPath);

      testOutputsBuilder.add(Pair.of(testOutput.getFirst(), destinationPath));
    }

    // Add the test log to the output
    dataBuilder.addFailedLogs(testLog.toString());
    dataBuilder.addTestTimes(data.getTestTimes(0));
    dataBuilder.addAllTestProcessTimes(data.getTestProcessTimesList());
    actionExecutionContext
        .getEventBus()
        .post(
            new TestAttempt(
                action,
                attempt,
                data.getStatus(),
                data.getStartTimeMillisEpoch(),
                data.getRunDurationMillis(),
                testOutputsBuilder.build(),
                data.getWarningList(),
                false));
    processTestOutput(actionExecutionContext, new TestResult(action, data, false), testLog);
  }

  private void processLastTestAttempt(int attempt, Builder dataBuilder, TestResultData data) {
    dataBuilder.setHasCoverage(data.getHasCoverage());
    dataBuilder.setStatus(
        data.getStatus() == BlazeTestStatus.PASSED && attempt > 1
            ? BlazeTestStatus.FLAKY
            : data.getStatus());
    dataBuilder.setTestPassed(data.getTestPassed());
    for (int i = 0; i < data.getFailedLogsCount(); i++) {
      dataBuilder.addFailedLogs(data.getFailedLogs(i));
    }
    if (data.getTestPassed()) {
      dataBuilder.setPassedLog(data.getPassedLog());
    }
    dataBuilder.addTestTimes(data.getTestTimes(0));
    dataBuilder.addAllTestProcessTimes(data.getTestProcessTimesList());
    dataBuilder.setStartTimeMillisEpoch(data.getStartTimeMillisEpoch());
    dataBuilder.setRunDurationMillis(data.getRunDurationMillis());
    if (data.hasTestCase()) {
      dataBuilder.setTestCase(data.getTestCase());
    }
  }

  private TestResultData executeTestAttempt(
      TestRunnerAction action,
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      Path execRoot,
      Path coverageDir,
      Path tmpDir,
      Path workingDirectory)
      throws IOException, ExecException, InterruptedException {
    prepareFileSystem(action, tmpDir, coverageDir, workingDirectory);

    try (FileOutErr fileOutErr =
            new FileOutErr(
                action.getTestLog().getPath(), action.resolve(execRoot).getTestStderr())) {
      TestResultData data =
          executeTest(
              action,
              spawn,
              actionExecutionContext.withFileOutErr(fileOutErr));
      appendStderr(fileOutErr.getOutputPath(), fileOutErr.getErrorPath());
      return data;
    }
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

  protected TestResultData executeTest(
      TestRunnerAction action,
      Spawn spawn,
      ActionExecutionContext actionExecutionContext)
          throws ExecException, InterruptedException, IOException {
    Closeable streamed = null;
    Path testLogPath = action.getTestLog().getPath();
    TestResultData.Builder builder = TestResultData.newBuilder();

    long startTime = actionExecutionContext.getClock().currentTimeMillis();
    SpawnActionContext spawnActionContext =
        actionExecutionContext.getSpawnActionContext(action.getMnemonic());
    try {
      try {
        if (executionOptions.testOutput.equals(TestOutputFormat.STREAMED)) {
          streamed =
              new StreamedTestOutput(
                  Reporter.outErrForReporter(actionExecutionContext.getEventHandler()),
                  testLogPath);
        }
        spawnActionContext.exec(spawn, actionExecutionContext);

        builder
            .setTestPassed(true)
            .setStatus(BlazeTestStatus.PASSED)
            .setPassedLog(testLogPath.getPathString());
      } catch (SpawnExecException e) {
        // If this method returns normally, then the higher level will rerun the test (up to
        // --flaky_test_attempts times). We don't catch any other ExecException here, so those never
        // get retried.
        builder
            .setTestPassed(false)
            .setStatus(e.hasTimedOut() ? BlazeTestStatus.TIMEOUT : BlazeTestStatus.FAILED)
            .addFailedLogs(testLogPath.getPathString());
      } finally {
        long duration = actionExecutionContext.getClock().currentTimeMillis() - startTime;
        builder.setStartTimeMillisEpoch(startTime);
        builder.addTestTimes(duration);
        builder.addTestProcessTimes(duration);
        builder.setRunDurationMillis(duration);
        if (streamed != null) {
          streamed.close();
        }
      }

      TestCase details =
          parseTestResult(
              action
                  .resolve(actionExecutionContext.getExecRoot())
                  .getXmlOutputPath());
      if (details != null) {
        builder.setTestCase(details);
      }

      if (action.isCoverageMode()) {
        builder.setHasCoverage(true);
      }

      return builder.build();
    } catch (IOException e) {
      throw new TestExecException(e.getMessage());
    }
  }

  /**
   * Outputs test result to the stdout after test has finished (e.g. for --test_output=all or
   * --test_output=errors). Will also try to group output lines together (up to 10000 lines) so
   * parallel test outputs will not get interleaved.
   */
  protected void processTestOutput(
      ActionExecutionContext actionExecutionContext, TestResult result, Path testLogPath)
          throws IOException {
    Path testOutput = actionExecutionContext.getExecRoot().getRelative(testLogPath.asFragment());
    boolean isPassed = result.getData().getTestPassed();
    try {
      if (TestLogHelper.shouldOutputTestLog(executionOptions.testOutput, isPassed)) {
        TestLogHelper.writeTestLog(
            testOutput,
            result.getTestName(),
            actionExecutionContext.getFileOutErr().getOutputStream());
      }
    } finally {
      if (isPassed) {
        actionExecutionContext
            .getEventHandler().handle(Event.of(EventKind.PASS, null, result.getTestName()));
      } else {
        if (result.getData().getStatus() == BlazeTestStatus.TIMEOUT) {
          actionExecutionContext
              .getEventHandler()
              .handle(
                  Event.of(
                      EventKind.TIMEOUT, null, result.getTestName() + " (see " + testOutput + ")"));
        } else {
          actionExecutionContext
              .getEventHandler()
              .handle(
                  Event.of(
                      EventKind.FAIL, null, result.getTestName() + " (see " + testOutput + ")"));
        }
      }
    }
  }

  private final void finalizeTest(
      ActionExecutionContext actionExecutionContext, TestRunnerAction action, TestResultData data)
      throws IOException, ExecException {
    TestResult result = new TestResult(action, data, false);
    postTestResult(actionExecutionContext, result);

    processTestOutput(
        actionExecutionContext,
        result,
        result.getTestLogPath());
    // TODO(bazel-team): handle --test_output=errors, --test_output=all.

    if (!executionOptions.testKeepGoing
        && data.getStatus() != BlazeTestStatus.FLAKY
        && data.getStatus() != BlazeTestStatus.PASSED) {
      throw new TestExecException("Test failed: aborting");
    }
  }

  @Override
  public TestResult newCachedTestResult(
      Path execRoot, TestRunnerAction action, TestResultData data) {
    return new TestResult(action, data, /*cached*/ true, execRoot);
  }
}
