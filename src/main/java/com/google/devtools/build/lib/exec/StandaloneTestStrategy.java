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
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.TestFileNameConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.io.Closeable;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Runs TestRunnerAction actions. */
// TODO(bazel-team): add tests for this strategy.
@ExecutionStrategy(
  contextType = TestActionContext.class,
  name = {"standalone"}
)
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
  public List<SpawnResult> exec(
      TestRunnerAction action, ActionExecutionContext actionExecutionContext)
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
    Path tmpDir = tmpDirRoot.getChild(TestStrategy.getTmpDirName(action));
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
            new RunfilesSupplierImpl(
                runfilesDir.relativeTo(execRoot), action.getExecutionSettings().getRunfiles()),
            ImmutableMap.of(),
            /*inputs=*/ ImmutableList.copyOf(action.getInputs()),
            /*tools=*/ ImmutableList.<Artifact>of(),
            ImmutableList.copyOf(action.getSpawnOutputs()),
            localResourceUsage);

    TestResultData.Builder dataBuilder = TestResultData.newBuilder();

    try {
      int maxAttempts = getTestAttempts(action);
      StandaloneTestResult standaloneTestResult =
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
          standaloneTestResult.testResultData().getStatus() != BlazeTestStatus.PASSED
              && attempt < maxAttempts;
          attempt++) {
        processFailedTestAttempt(
            attempt, actionExecutionContext, action, dataBuilder, standaloneTestResult);
        standaloneTestResult =
            executeTestAttempt(
                action,
                spawn,
                actionExecutionContext,
                execRoot,
                coverageDir,
                tmpDir,
                workingDirectory);
      }
      processLastTestAttempt(attempt, dataBuilder, standaloneTestResult.testResultData());
      ImmutableList<Pair<String, Path>> testOutputs =
          action.getTestOutputsMapping(actionExecutionContext.getPathResolver(), execRoot);
      actionExecutionContext
          .getEventHandler()
          .post(
              TestAttempt.forExecutedTestResult(
                  action,
                  standaloneTestResult.testResultData(),
                  attempt,
                  testOutputs,
                  standaloneTestResult.executionInfo(),
                  true));
      finalizeTest(actionExecutionContext, action, dataBuilder.build());

      // TODO(b/62588075): Should we accumulate SpawnResults across test attempts instead of only
      // returning the last list?
      return standaloneTestResult.spawnResults();
    } catch (IOException e) {
      // Print the stack trace, otherwise the unexpected I/O error is hard to diagnose.
      // A stack trace could help with bugs like https://github.com/bazelbuild/bazel/issues/4924
      StringBuilder sb = new StringBuilder();
      sb.append("Caught I/O exception: ").append(e.getMessage());
      for (Object s : e.getStackTrace()) {
        sb.append("\n\t").append(s);
      }
      actionExecutionContext.getEventHandler().handle(Event.error(sb.toString()));
      throw new EnvironmentalExecException("unexpected I/O exception", e);
    }
  }

  private void processFailedTestAttempt(
      int attempt,
      ActionExecutionContext actionExecutionContext,
      TestRunnerAction action,
      TestResultData.Builder dataBuilder,
      StandaloneTestResult result)
      throws IOException {
    ImmutableList.Builder<Pair<String, Path>> testOutputsBuilder = new ImmutableList.Builder<>();
    // Rename outputs
    String namePrefix =
        FileSystemUtils.removeExtension(action.getTestLog().getExecPath().getBaseName());
    Path testRoot = actionExecutionContext.getInputPath(action.getTestLog()).getParentDirectory();
    Path attemptsDir = testRoot.getChild(namePrefix + "_attempts");
    attemptsDir.createDirectory();
    String attemptPrefix = "attempt_" + attempt;
    Path testLog = attemptsDir.getChild(attemptPrefix + ".log");

    // Get the normal test output paths, and then update them to use "attempt_N" names, and
    // attemptDir, before adding them to the outputs.
    ImmutableList<Pair<String, Path>> testOutputs =
        action.getTestOutputsMapping(actionExecutionContext.getPathResolver(),
            actionExecutionContext.getExecRoot());
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

    // Add the test log to the output
    TestResultData data = result.testResultData();
    dataBuilder.addFailedLogs(testLog.toString());
    dataBuilder.addTestTimes(data.getTestTimes(0));
    dataBuilder.addAllTestProcessTimes(data.getTestProcessTimesList());
    actionExecutionContext
        .getEventHandler()
        .post(
            TestAttempt.forExecutedTestResult(
                action,
                data,
                attempt,
                testOutputsBuilder.build(),
                result.executionInfo(),
                false));
    processTestOutput(actionExecutionContext, new TestResult(action, data, false), testLog);
  }

  private void processLastTestAttempt(
      int attempt, TestResultData.Builder dataBuilder, TestResultData data) {
    dataBuilder.setHasCoverage(data.getHasCoverage());
    dataBuilder.setRemotelyCached(data.getRemotelyCached());
    dataBuilder.setIsRemoteStrategy(data.getIsRemoteStrategy());
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

  private StandaloneTestResult executeTestAttempt(
      TestRunnerAction action,
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      Path execRoot,
      Path coverageDir,
      Path tmpDir,
      Path workingDirectory)
      throws IOException, ExecException, InterruptedException {
    prepareFileSystem(action, tmpDir, coverageDir, workingDirectory);

    Path out = actionExecutionContext.getInputPath(action.getTestLog());
    Path err = action.resolve(execRoot).getTestStderr();
    StandaloneTestResult standaloneTestResult = null;
    try (FileOutErr fileOutErr = new FileOutErr(out, err)) {
      standaloneTestResult =
          executeTest(action, spawn, actionExecutionContext.withFileOutErr(fileOutErr));
      if (!fileOutErr.hasRecordedOutput()) {
        // Touch the output file so that test.log can get created.
        FileSystemUtils.touchFile(fileOutErr.getOutputPath());
      }
    }
    appendStderr(out, err);
    return standaloneTestResult;
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

  protected StandaloneTestResult executeTest(
      TestRunnerAction action, Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException, IOException {
    Closeable streamed = null;
    Path testLogPath = actionExecutionContext.getInputPath(action.getTestLog());
    TestResultData.Builder builder = TestResultData.newBuilder();

    long startTime = actionExecutionContext.getClock().currentTimeMillis();
    SpawnActionContext spawnActionContext =
        actionExecutionContext.getContext(SpawnActionContext.class);
    Path xmlOutputPath = action.resolve(actionExecutionContext.getExecRoot()).getXmlOutputPath();
    List<SpawnResult> spawnResults = new ArrayList<>();
    BuildEventStreamProtos.TestResult.ExecutionInfo.Builder executionInfo =
        BuildEventStreamProtos.TestResult.ExecutionInfo.newBuilder();
    try {
      try {
        if (executionOptions.testOutput.equals(TestOutputFormat.STREAMED)) {
          streamed =
              new StreamedTestOutput(
                  Reporter.outErrForReporter(actionExecutionContext.getEventHandler()),
                  testLogPath);
        }
        try {
          spawnResults.addAll(spawnActionContext.exec(spawn, actionExecutionContext));
          builder
              .setTestPassed(true)
              .setStatus(BlazeTestStatus.PASSED)
              .setPassedLog(testLogPath.getPathString());
        } catch (SpawnExecException e) {
          // If this method returns normally, then the higher level will rerun the test (up to
          // --flaky_test_attempts times).
          if (e.isCatastrophic()) {
            // Rethrow as the error was catastrophic and thus the build has to be halted.
            throw e;
          }
          if (!e.getSpawnResult().setupSuccess()) {
            // Rethrow as the test could not be run and thus there's no point in retrying.
            throw e;
          }
          builder
              .setTestPassed(false)
              .setStatus(e.hasTimedOut() ? BlazeTestStatus.TIMEOUT : BlazeTestStatus.FAILED)
              .addFailedLogs(testLogPath.getPathString());
          spawnResults.add(e.getSpawnResult());
        }
        // If the test did not create a test.xml, and --experimental_split_xml_generation is
        // enabled, then we run a separate action to create a test.xml from test.log.
        if (executionOptions.splitXmlGeneration
            && action.getTestLog().getPath().exists()
            && !xmlOutputPath.exists()) {
          SpawnResult result = Iterables.getOnlyElement(spawnResults);
          Spawn xmlGeneratingSpawn = createXmlGeneratingSpawn(action, result);
          // We treat all failures to generate the test.xml here as catastrophic, and won't rerun
          // the test if this fails.
          spawnResults.addAll(spawnActionContext.exec(xmlGeneratingSpawn, actionExecutionContext));
        }
      } finally {
        long endTime = actionExecutionContext.getClock().currentTimeMillis();
        long duration = endTime - startTime;
        // If execution fails with an exception other SpawnExecException, there is no result here.
        if (!spawnResults.isEmpty()) {
          // The SpawnResult of a remotely cached or remotely executed action may not have walltime
          // set. We fall back to the time measured here for backwards compatibility.
          SpawnResult primaryResult = spawnResults.iterator().next();
          duration = primaryResult.getWallTime().orElse(Duration.ofMillis(duration)).toMillis();
          extractExecutionInfo(primaryResult, builder, executionInfo);
        }

        builder.setStartTimeMillisEpoch(startTime);
        builder.addTestTimes(duration);
        builder.addTestProcessTimes(duration);
        builder.setRunDurationMillis(duration);
        if (streamed != null) {
          streamed.close();
        }
      }

      TestCase details = parseTestResult(xmlOutputPath);
      if (details != null) {
        builder.setTestCase(details);
      }

      if (action.isCoverageMode()) {
        builder.setHasCoverage(true);
      }

      return StandaloneTestResult.builder()
          .setSpawnResults(spawnResults)
          .setTestResultData(builder.build())
          .setExecutionInfo(executionInfo.build())
          .build();
    } catch (IOException e) {
      throw new TestExecException(e.getMessage());
    }
  }

  private static void extractExecutionInfo(
      SpawnResult spawnResult,
      TestResultData.Builder result,
      BuildEventStreamProtos.TestResult.ExecutionInfo.Builder executionInfo) {
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
  }

  /**
   * A spawn to generate a test.xml file from the test log. This is only used if the test does not
   * generate a test.xml file itself.
   */
  private Spawn createXmlGeneratingSpawn(TestRunnerAction action, SpawnResult result) {
    List<String> args = Lists.newArrayList();
    // TODO(ulfjack): This is incorrect for remote execution, where we need to consider the target
    // configuration, not the machine Bazel happens to run on. Change this to something like:
    // testAction.getConfiguration().getExecOS() == OS.WINDOWS
    if (OS.getCurrent() == OS.WINDOWS && !action.isUsingTestWrapperInsteadOfTestSetupScript()) {
      args.add(action.getShExecutable().getPathString());
      args.add("-c");
      args.add("$0 $*");
    }
    args.add(action.getTestXmlGeneratorScript().getExecPath().getCallablePathString());
    args.add(action.getTestLog().getExecPathString());
    args.add(action.getXmlOutputPath().getPathString());
    args.add(Long.toString(result.getWallTime().orElse(Duration.ZERO).getSeconds()));
    args.add(Integer.toString(result.exitCode()));

    return new SimpleSpawn(
        action,
        ImmutableList.copyOf(args),
        ImmutableMap.of(
            "PATH", "/usr/bin:/bin",
            "TEST_SHARD_INDEX", Integer.toString(action.getShardNum()),
            "TEST_TOTAL_SHARDS", Integer.toString(action.getExecutionSettings().getTotalShards()),
            "TEST_NAME", action.getTestName()),
        ImmutableMap.of(),
        null,
        ImmutableMap.of(),
        /*inputs=*/ ImmutableList.of(action.getTestXmlGeneratorScript(), action.getTestLog()),
        /*tools=*/ ImmutableList.<Artifact>of(),
        /*outputs=*/ ImmutableList.of(ActionInputHelper.fromPath(action.getXmlOutputPath())),
        SpawnAction.DEFAULT_RESOURCE_SET);
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
