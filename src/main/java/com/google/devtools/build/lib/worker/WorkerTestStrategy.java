// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.worker;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.StandaloneTestStrategy;
import com.google.devtools.build.lib.rules.test.TestActionContext;
import com.google.devtools.build.lib.rules.test.TestRunnerAction;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.devtools.common.options.OptionsClassProvider;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Runs TestRunnerAction actions in a worker. This is still experimental WIP.
 * Do not use this strategy to run tests. <br>
 *
 * TODO(kush): List to things to cosider: <br>
 * 1. Figure out if/how to honor the actions's execution info:
 *      action.getTestProperties().getExecutionInfo() <br>
 * 2. Figure out how to stream intermediate output when running in a Worker or block streamed
 *      outputs for this strategy. <br>
 * 3. Figure out how to add timeout facility. <br>
 */
@ExecutionStrategy(contextType = TestActionContext.class, name = { "experimental_worker" })
public class WorkerTestStrategy extends StandaloneTestStrategy {
  private final WorkerPool workerPool;
  private final int maxRetries;
  private final Multimap<String, String> extraFlags;

  public WorkerTestStrategy(
      CommandEnvironment env,
      OptionsClassProvider requestOptions,
      WorkerPool workerPool,
      int maxRetries,
      Multimap<String, String> extraFlags) {
    super(
        requestOptions,
        env.getBlazeWorkspace().getBinTools(),
        env.getClientEnv(),
        env.getWorkspace());
    this.workerPool = workerPool;
    this.maxRetries = maxRetries;
    this.extraFlags = extraFlags;
  }

  @Override
  protected TestResultData executeTest(
      TestRunnerAction action,
      ActionExecutionContext actionExecutionContext,
      Map<String, String> environment,
      Path execRoot,
      Path runfilesDir)
      throws ExecException, InterruptedException, IOException {
    List<String> startupArgs = getStartUpArgs(action);

    return execInWorker(
        action, actionExecutionContext, environment, startupArgs, execRoot, maxRetries);
  }

  private TestResultData execInWorker(
      TestRunnerAction action,
      ActionExecutionContext actionExecutionContext,
      Map<String, String> environment,
      List<String> startupArgs,
      Path execRoot,
      int retriesLeft)
      throws ExecException, InterruptedException, IOException {
    Executor executor = actionExecutionContext.getExecutor();
    TestResultData.Builder builder = TestResultData.newBuilder();

    Path testLogPath = action.getTestLog().getPath();
    Worker worker = null;
    WorkerKey key = null;
    long startTime = executor.getClock().currentTimeMillis();
    try {
      HashCode workerFilesHash = WorkerFilesHash.getWorkerFilesHash(
          action.getTools(), actionExecutionContext);
      key =
          new WorkerKey(
              startupArgs,
              environment,
              execRoot,
              action.getMnemonic(),
              workerFilesHash,
              ImmutableMap.<PathFragment, Path>of(),
              ImmutableSet.<PathFragment>of(),
              /*mustBeSandboxed=*/false);
      worker = workerPool.borrowObject(key);

      WorkRequest request = WorkRequest.getDefaultInstance();
      request.writeDelimitedTo(worker.getOutputStream());
      worker.getOutputStream().flush();

      WorkResponse response = WorkResponse.parseDelimitedFrom(worker.getInputStream());
      actionExecutionContext.getFileOutErr().getErrorStream().write(
          response.getOutputBytes().toByteArray());

      long duration = executor.getClock().currentTimeMillis() - startTime;
      builder.addTestTimes(duration);
      builder.setRunDurationMillis(duration);
      if (response.getExitCode() == 0) {
        builder
            .setTestPassed(true)
            .setStatus(BlazeTestStatus.PASSED)
            .setCachable(true)
            .setPassedLog(testLogPath.getPathString());
      } else {
        builder
            .setTestPassed(false)
            .setStatus(BlazeTestStatus.FAILED)
            .addFailedLogs(testLogPath.getPathString());
      }
      TestCase details = parseTestResult(
          action.resolve(actionExecutionContext.getExecutor().getExecRoot()).getXmlOutputPath());
      if (details != null) {
        builder.setTestCase(details);
      }

      return builder.build();
    } catch (IOException | InterruptedException e) {
      if (e instanceof InterruptedException) {
        // The user pressed Ctrl-C. Get out here quick.
        retriesLeft = 0;
      }

      if (worker != null) {
        workerPool.invalidateObject(key, worker);
        worker = null;
      }
      if (retriesLeft > 0) {
        // The worker process failed, but we still have some retries left. Let's retry with a fresh
        // worker.
        executor
            .getEventHandler()
            .handle(
                Event.warn(
                    key.getMnemonic()
                        + " worker failed ("
                        + e
                        + "), invalidating and retrying with new worker..."));
        return execInWorker(
            action, actionExecutionContext, environment, startupArgs, execRoot, retriesLeft - 1);
      } else {
        throw new TestExecException(e.getMessage());
      }
    } finally {
      if (worker != null) {
        workerPool.returnObject(key, worker);
      }
    }
  }

  private List<String> getStartUpArgs(TestRunnerAction action) throws ExecException {
    Artifact testSetup = action.getRuntimeArtifact(TEST_SETUP_BASENAME);
    List<String> args = getArgs(testSetup.getExecPathString(), "", action);
    ImmutableList.Builder<String> startupArgs = ImmutableList.builder();
    // Add test setup with no echo to prevent stdout corruption.
    startupArgs.add(args.get(0)).add("--no_echo");
    // Add remaining of the original args.
    startupArgs.addAll(args.subList(1, args.size()));
    // Make the Test runner run persistently.
    startupArgs.add("--persistent_test_runner");
    // Add additional flags requested for this invocation.
    startupArgs.addAll(MoreObjects.firstNonNull(
            extraFlags.get(action.getMnemonic()), ImmutableList.<String>of()));
    return startupArgs.build();
  }
}
