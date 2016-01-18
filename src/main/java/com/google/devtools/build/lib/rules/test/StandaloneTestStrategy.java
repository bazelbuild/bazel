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

package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.OptionsClassProvider;

import java.io.Closeable;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Runs TestRunnerAction actions.
 */
@ExecutionStrategy(contextType = TestActionContext.class, name = { "standalone" })
public class StandaloneTestStrategy extends TestStrategy {
  // TODO(bazel-team) - add tests for this strategy.

  private final Path workspace;

  public StandaloneTestStrategy(
      OptionsClassProvider requestOptions,
      BinTools binTools,
      Map<String, String> clientEnv,
      Path workspace) {
    super(requestOptions, binTools, clientEnv);
    this.workspace = workspace;
  }

  @Override
  public void exec(TestRunnerAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    Path runfilesDir = null;
    try {
      runfilesDir = TestStrategy.getLocalRunfilesDirectory(
          action, actionExecutionContext, binTools);
    } catch (ExecException e) {
      throw new TestExecException(e.getMessage());
    }

    Path testTmpDir = TestStrategy.getTmpRoot(
        workspace, actionExecutionContext.getExecutor().getExecRoot(), executionOptions)
        .getChild(getTmpDirName(action.getExecutionSettings().getExecutable().getExecPath()));
    Path workingDirectory = runfilesDir.getRelative(action.getRunfilesPrefix());

    Path execRoot = actionExecutionContext.getExecutor().getExecRoot();
    TestRunnerAction.ResolvedPaths resolvedPaths = action.resolve(execRoot);
    Map<String, String> env = getEnv(action, runfilesDir, testTmpDir, resolvedPaths);

    Map<String, String> info = new HashMap<>();

    // This key is only understood by StandaloneSpawnStrategy.
    info.put("timeout", "" + getTimeout(action));
    info.putAll(action.getTestProperties().getExecutionInfo());

    Artifact testSetup = action.getRuntimeArtifact(TEST_SETUP_BASENAME);
    Spawn spawn =
        new BaseSpawn(
            // Bazel lacks much of the tooling for coverage, so we don't attempt to pass a coverage
            // script here.
            getArgs(testSetup.getExecPathString(), "", action),
            env,
            info,
            new RunfilesSupplierImpl(
                runfilesDir.asFragment(), action.getExecutionSettings().getRunfiles()),
            action,
            action.getTestProperties().getLocalResourceUsage(executionOptions.usingLocalTestJobs()),
            ImmutableSet.of(resolvedPaths.getXmlOutputPath().relativeTo(execRoot)));

    Executor executor = actionExecutionContext.getExecutor();

    try {
      if (testTmpDir.exists(Symlinks.NOFOLLOW)) {
        FileSystemUtils.deleteTree(testTmpDir);
      }
      FileSystemUtils.createDirectoryAndParents(testTmpDir);
    } catch (IOException e) {
      executor.getEventHandler().handle(Event.error("Could not create TEST_TMPDIR: " + e));
      throw new EnvironmentalExecException("Could not create TEST_TMPDIR " + testTmpDir, e);
    }

    ResourceSet resources = null;
    FileOutErr fileOutErr = null;
    try {
      FileSystemUtils.createDirectoryAndParents(workingDirectory);
      fileOutErr = new FileOutErr(action.getTestLog().getPath(),
          action.resolve(actionExecutionContext.getExecutor().getExecRoot()).getTestStderr());

      resources = action.getTestProperties()
          .getLocalResourceUsage(executionOptions.usingLocalTestJobs());
      ResourceManager.instance().acquireResources(action, resources);
      TestResultData data = execute(
          actionExecutionContext.withFileOutErr(fileOutErr), spawn, action);
      appendStderr(fileOutErr.getOutputFile(), fileOutErr.getErrorFile());
      finalizeTest(actionExecutionContext, action, data);
    } catch (IOException e) {
      executor.getEventHandler().handle(Event.error("Caught I/O exception: " + e));
      throw new EnvironmentalExecException("unexpected I/O exception", e);
    } finally {
      if (resources != null) {
        ResourceManager.instance().releaseResources(action, resources);
      }
      try {
        if (fileOutErr != null) {
          fileOutErr.close();
        }
      } catch (IOException e) {
        // If the close fails, there is little we can do.
      }
    }
  }

  private Map<String, String> getEnv(
      TestRunnerAction action,
      Path runfilesDir,
      Path tmpDir,
      TestRunnerAction.ResolvedPaths resolvedPaths) {
    Map<String, String> vars = getDefaultTestEnvironment(action);
    BuildConfiguration config = action.getConfiguration();

    vars.putAll(config.getDefaultShellEnvironment());
    vars.putAll(action.getTestEnv());

    /*
     * TODO(bazel-team): the paths below are absolute,
     * making test actions impossible to cache remotely.
     */
    vars.put("TEST_SRCDIR", runfilesDir.getPathString());
    vars.put("TEST_TMPDIR", tmpDir.getPathString());
    vars.put("TEST_WORKSPACE", action.getRunfilesPrefix());
    vars.put("XML_OUTPUT_FILE", resolvedPaths.getXmlOutputPath().getPathString());

    return vars;
  }

  private TestResultData execute(
      ActionExecutionContext actionExecutionContext, Spawn spawn, TestRunnerAction action)
      throws TestExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    Closeable streamed = null;
    Path testLogPath = action.getTestLog().getPath();
    TestResultData.Builder builder = TestResultData.newBuilder();

    long startTime = executor.getClock().currentTimeMillis();
    try {
      try {
        if (executionOptions.testOutput.equals(TestOutputFormat.STREAMED)) {
          streamed = new StreamedTestOutput(
              Reporter.outErrForReporter(
                  actionExecutionContext.getExecutor().getEventHandler()), testLogPath);
        }
        executor.getSpawnActionContext(action.getMnemonic()).exec(spawn, actionExecutionContext);

        builder.setTestPassed(true)
            .setStatus(BlazeTestStatus.PASSED)
            .setCachable(true)
            .setPassedLog(testLogPath.getPathString());
      } catch (ExecException e) {
        // Execution failed, which we consider a test failure.

        // TODO(bazel-team): set cachable==true for relevant statuses (failure, but not for
        // timeout, etc.)
        builder
            .setTestPassed(false)
            .setStatus(e.hasTimedOut() ? BlazeTestStatus.TIMEOUT : BlazeTestStatus.FAILED)
            .addFailedLogs(testLogPath.getPathString());
      } finally {
        long duration = executor.getClock().currentTimeMillis() - startTime;
        builder.addTestTimes(duration);
        builder.setRunDurationMillis(duration);
        if (streamed != null) {
          streamed.close();
        }
      }

      TestCase details = parseTestResult(
          action.resolve(actionExecutionContext.getExecutor().getExecRoot()).getXmlOutputPath());
      if (details != null) {
        builder.setTestCase(details);
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
  protected void processTestOutput(Executor executor, FileOutErr outErr, TestResult result)
      throws IOException {
    Path testOutput = executor.getExecRoot().getRelative(result.getTestLogPath().asFragment());
    boolean isPassed = result.getData().getTestPassed();
    try {
      if (TestLogHelper.shouldOutputTestLog(executionOptions.testOutput, isPassed)) {
        TestLogHelper.writeTestLog(testOutput, result.getTestName(), outErr.getOutputStream());
      }
    } finally {
      if (isPassed) {
        executor.getEventHandler().handle(new Event(EventKind.PASS, null, result.getTestName()));
      } else {
        if (result.getData().getStatus() == BlazeTestStatus.TIMEOUT) {
          executor.getEventHandler().handle(
              new Event(EventKind.TIMEOUT, null, result.getTestName()
                  + " (see " + testOutput + ")"));
        } else {
          executor.getEventHandler().handle(
              new Event(EventKind.FAIL, null, result.getTestName() + " (see " + testOutput + ")"));
        }
      }
    }
  }

  private final void finalizeTest(ActionExecutionContext actionExecutionContext,
      TestRunnerAction action, TestResultData data) throws IOException, ExecException {
    TestResult result = new TestResult(action, data, false);
    postTestResult(actionExecutionContext.getExecutor(), result);

    processTestOutput(actionExecutionContext.getExecutor(),
        actionExecutionContext.getFileOutErr(), result);
    // TODO(bazel-team): handle --test_output=errors, --test_output=all.

    if (!executionOptions.testKeepGoing && data.getStatus() != BlazeTestStatus.PASSED) {
      throw new TestExecException("Test failed: aborting");
    }
  }

  @Override
  public String strategyLocality(TestRunnerAction action) { return "standalone"; }

  @Override
  public TestResult newCachedTestResult(
      Path execRoot, TestRunnerAction action, TestResultData data) {
    return new TestResult(action, data, /*cached*/ true);
  }
}
