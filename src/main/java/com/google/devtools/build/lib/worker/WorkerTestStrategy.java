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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.UserExecException;
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
import com.google.protobuf.InvalidProtocolBufferException;
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
  private final Multimap<String, String> extraFlags;

  public WorkerTestStrategy(
      CommandEnvironment env,
      OptionsClassProvider requestOptions,
      WorkerPool workerPool,
      Multimap<String, String> extraFlags) {
    super(
        requestOptions,
        env.getBlazeWorkspace().getBinTools(),
        env.getClientEnv(),
        env.getWorkspace());
    this.workerPool = workerPool;
    this.extraFlags = extraFlags;
  }

  @Override
  protected TestResultData executeTest(
      TestRunnerAction action,
      Spawn spawn,
      ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException, IOException {
    if (!action.getConfiguration().compatibleWithStrategy("experimental_worker")) {
      throw new UserExecException(
          "Build configuration not compatible with experimental_worker "
              + "strategy. Make sure you set the explicit_java_test_deps and "
              + "experimental_testrunner flags to true.");
    }

    if (!action.useTestRunner()) {
      throw new UserExecException(
          "Tests that do not use the experimental test runner are incompatible with the persistent"
              + " worker test strategy. Please use another test strategy");
    }
    if (action.isCoverageMode()) {
      throw new UserExecException("Coverage is currently incompatible"
          + " with the persistent worker test strategy. Please use another test strategy");
    }
    List<String> startupArgs = getStartUpArgs(action);

    return execInWorker(
        action,
        actionExecutionContext,
        addPersistentRunnerVars(spawn.getEnvironment()),
        startupArgs,
        actionExecutionContext.getExecutor().getExecRoot());
  }

  private TestResultData execInWorker(
      TestRunnerAction action,
      ActionExecutionContext actionExecutionContext,
      Map<String, String> environment,
      List<String> startupArgs,
      Path execRoot)
      throws ExecException, InterruptedException, IOException {
    Executor executor = actionExecutionContext.getExecutor();

    // TODO(kush): Remove once we're out of the experimental phase.
    executor
        .getEventHandler()
        .handle(
            Event.warn(
                "RUNNING TEST IN AN EXPERIMENTAL PERSISTENT WORKER. RESULTS MAY BE INACCURATE"));

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

      RecordingInputStream recordingStream = new RecordingInputStream(worker.getInputStream());
      recordingStream.startRecording(4096);
      WorkResponse response;
      try {
        // response can be null when the worker has already closed stdout at this point and thus the
        // InputStream is at EOF.
        response = WorkResponse.parseDelimitedFrom(recordingStream);
      } catch (InvalidProtocolBufferException e) {
        // If protobuf couldn't parse the response, try to print whatever the failing worker wrote
        // to stdout - it's probably a stack trace or some kind of error message that will help the
        // user figure out why the compiler is failing.
        recordingStream.readRemaining();
        String data = recordingStream.getRecordedDataAsString();
        ErrorMessage errorMessage =
            ErrorMessage.builder()
                .message("Worker process returned an unparseable WorkResponse:")
                .logText(data)
                .build();
        executor.getEventHandler().handle(Event.warn(errorMessage.toString()));
        throw e;
      }

      worker.finishExecution(key);

      if (response == null) {
        ErrorMessage errorMessage =
            ErrorMessage.builder()
                .message(
                    "Worker process did not return a WorkResponse. This is usually caused by a bug"
                        + " in the worker, thus dumping its log file for debugging purposes:")
                .logFile(worker.getLogFile())
                .logSizeLimit(4096)
                .build();
        throw new UserExecException(errorMessage.toString());
      }

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
      if (worker != null) {
        workerPool.invalidateObject(key, worker);
        worker = null;
      }

      throw new TestExecException(e.getMessage());
    } finally {
      if (worker != null) {
        workerPool.returnObject(key, worker);
      }
    }
  }

  private static Map<String, String> addPersistentRunnerVars(Map<String, String> originalEnv)
      throws UserExecException {
    if (originalEnv.containsKey("PERSISTENT_TEST_RUNNER")) {
      throw new UserExecException(
          "Found clashing environment variable with persistent_test_runner."
              + " Please use another test strategy");
    }
    return ImmutableMap.<String, String>builder()
        .putAll(originalEnv)
        .put("PERSISTENT_TEST_RUNNER", "true")
        .build();
  }

  private List<String> getStartUpArgs(TestRunnerAction action) throws ExecException {
    List<String> args = getArgs(/*coverageScript=*/ "coverage-is-not-supported", action);
    ImmutableList.Builder<String> startupArgs = ImmutableList.builder();
    // Add test setup with no echo to prevent stdout corruption.
    startupArgs.add(args.get(0)).add("--no_echo");
    // Add remaining of the original args.
    startupArgs.addAll(args.subList(1, args.size()));
    // Add additional flags requested for this invocation.
    startupArgs.addAll(MoreObjects.firstNonNull(
            extraFlags.get(action.getMnemonic()), ImmutableList.<String>of()));
    return startupArgs.build();
  }
}
