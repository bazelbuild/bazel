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
package com.google.devtools.build.lib.worker;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.eventbus.EventBus;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.standalone.StandaloneSpawnStrategy;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;

/**
 * A spawn action context that launches Spawns the first time they are used in a persistent mode and
 * then shards work over all the processes.
 */
@ExecutionStrategy(
  name = {"worker"},
  contextType = SpawnActionContext.class
)
public final class WorkerSpawnStrategy implements SandboxedSpawnActionContext {

  public static final String ERROR_MESSAGE_PREFIX =
      "Worker strategy cannot execute this %s action, ";
  public static final String REASON_NO_FLAGFILE =
      "because the command-line arguments do not contain at least one @flagfile or --flagfile=";
  public static final String REASON_NO_TOOLS = "because the action has no tools";
  public static final String REASON_NO_EXECUTION_INFO =
      "because the action's execution info does not contain 'supports-workers=1'";

  /** Pattern for @flagfile.txt and --flagfile=flagfile.txt */
  private static final Pattern FLAG_FILE_PATTERN = Pattern.compile("(?:@|--?flagfile=)(.+)");

  private final WorkerPool workers;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final Multimap<String, String> extraFlags;
  private final SpawnInputExpander spawnInputExpander;

  public WorkerSpawnStrategy(
      Path execRoot,
      WorkerPool workers,
      boolean verboseFailures,
      Multimap<String, String> extraFlags) {
    Preconditions.checkNotNull(workers);
    this.workers = Preconditions.checkNotNull(workers);
    this.execRoot = execRoot;
    this.verboseFailures = verboseFailures;
    this.extraFlags = extraFlags;
    this.spawnInputExpander = new SpawnInputExpander(false);
  }

  @Override
  public void exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    exec(spawn, actionExecutionContext, null);
  }

  @Override
  public void exec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    if (!spawn.getExecutionInfo().containsKey(ExecutionRequirements.SUPPORTS_WORKERS)
        || !spawn.getExecutionInfo().get(ExecutionRequirements.SUPPORTS_WORKERS).equals("1")) {
      StandaloneSpawnStrategy standaloneStrategy =
          Preconditions.checkNotNull(executor.getContext(StandaloneSpawnStrategy.class));
      executor.getEventHandler().handle(
          Event.warn(
              String.format(ERROR_MESSAGE_PREFIX + REASON_NO_EXECUTION_INFO, spawn.getMnemonic())));
      standaloneStrategy.exec(spawn, actionExecutionContext);
      return;
    }

    EventBus eventBus = actionExecutionContext.getExecutor().getEventBus();
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    eventBus.post(ActionStatusMessage.schedulingStrategy(owner));
    try (ResourceHandle handle =
        ResourceManager.instance().acquireResources(owner, spawn.getLocalResources())) {
      eventBus.post(ActionStatusMessage.runningStrategy(spawn.getResourceOwner(), "worker"));
      actuallyExec(spawn, actionExecutionContext, writeOutputFiles);
    }
  }

  private void actuallyExec(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws ExecException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();

    if (executor.reportsSubcommands()) {
      executor.reportSubcommand(spawn);
    }

    if (Iterables.isEmpty(spawn.getToolFiles())) {
      throw new UserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_TOOLS, spawn.getMnemonic()));
    }

    // We assume that the spawn to be executed always gets at least one @flagfile.txt or
    // --flagfile=flagfile.txt argument, which contains the flags related to the work itself (as
    // opposed to start-up options for the executed tool). Thus, we can extract those elements from
    // its args and put them into the WorkRequest instead.
    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> workerArgs = splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);
    ImmutableMap<String, String> env = spawn.getEnvironment();

    try {
      ActionInputFileCache inputFileCache = actionExecutionContext.getActionInputFileCache();

      HashCode workerFilesHash = WorkerFilesHash.getWorkerFilesHash(
          spawn.getToolFiles(), actionExecutionContext);
      Map<PathFragment, Path> inputFiles =
          SandboxHelpers.getInputFiles(spawnInputExpander, execRoot, spawn, actionExecutionContext);
      Set<PathFragment> outputFiles = SandboxHelpers.getOutputFiles(spawn);

      WorkerKey key =
          new WorkerKey(
              workerArgs,
              env,
              execRoot,
              spawn.getMnemonic(),
              workerFilesHash,
              inputFiles,
              outputFiles,
              writeOutputFiles != null);

      WorkRequest workRequest =
          createWorkRequest(spawn, actionExecutionContext, flagFiles, inputFileCache);

      WorkResponse response = execInWorker(key, workRequest, writeOutputFiles);

      FileOutErr outErr = actionExecutionContext.getFileOutErr();
      response.getOutputBytes().writeTo(outErr.getErrorStream());

      if (response.getExitCode() != 0) {
        throw new UserExecException(
            String.format(
                "Worker process sent response with exit code: %d.", response.getExitCode()));
      }
    } catch (IOException e) {
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures, spawn.getArguments(), env, execRoot.getPathString());
      throw new UserExecException(
          ErrorMessage.builder().message(message).exception(e).build().toString());
    }
  }

  /**
   * Splits the command-line arguments of the {@code Spawn} into the part that is used to start the
   * persistent worker ({@code workerArgs}) and the part that goes into the {@code WorkRequest}
   * protobuf ({@code flagFiles}).
   */
  private ImmutableList<String> splitSpawnArgsIntoWorkerArgsAndFlagFiles(
      Spawn spawn, List<String> flagFiles) throws UserExecException {
    ImmutableList.Builder<String> workerArgs = ImmutableList.builder();
    for (String arg : spawn.getArguments()) {
      if (FLAG_FILE_PATTERN.matcher(arg).matches()) {
        flagFiles.add(arg);
      } else {
        workerArgs.add(arg);
      }
    }

    if (flagFiles.isEmpty()) {
      throw new UserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_FLAGFILE, spawn.getMnemonic()));
    }

    return workerArgs
        .add("--persistent_worker")
        .addAll(
            MoreObjects.firstNonNull(
                extraFlags.get(spawn.getMnemonic()), ImmutableList.<String>of()))
        .build();
  }

  private WorkRequest createWorkRequest(
      Spawn spawn,
      ActionExecutionContext actionExecutionContext,
      List<String> flagfiles,
      ActionInputFileCache inputFileCache)
      throws IOException {
    WorkRequest.Builder requestBuilder = WorkRequest.newBuilder();
    for (String flagfile : flagfiles) {
      expandArgument(requestBuilder, flagfile);
    }

    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(
            spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());

    for (ActionInput input : inputs) {
      byte[] digestBytes = inputFileCache.getDigest(input);
      ByteString digest;
      if (digestBytes == null) {
        digest = ByteString.EMPTY;
      } else {
        digest = ByteString.copyFromUtf8(HashCode.fromBytes(digestBytes).toString());
      }

      requestBuilder
          .addInputsBuilder()
          .setPath(input.getExecPathString())
          .setDigest(digest)
          .build();
    }
    return requestBuilder.build();
  }

  /**
   * Recursively expands arguments by replacing @filename args with the contents of the referenced
   * files. The @ itself can be escaped with @@. This deliberately does not expand --flagfile= style
   * arguments, because we want to get rid of the expansion entirely at some point in time.
   *
   * @param requestBuilder the WorkRequest.Builder that the arguments should be added to.
   * @param arg the argument to expand.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  private void expandArgument(WorkRequest.Builder requestBuilder, String arg) throws IOException {
    if (arg.startsWith("@") && !arg.startsWith("@@")) {
      for (String line : Files.readAllLines(
          Paths.get(execRoot.getRelative(arg.substring(1)).getPathString()), UTF_8)) {
        if (line.length() > 0) {
          expandArgument(requestBuilder, line);
        }
      }
    } else {
      requestBuilder.addArguments(arg);
    }
  }

  private WorkResponse execInWorker(
      WorkerKey key,
      WorkRequest request,
      AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
      throws InterruptedException, ExecException {
    Worker worker = null;
    WorkResponse response;

    try {
      try {
        worker = workers.borrowObject(key);
      } catch (IOException e) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message("IOException while borrowing a worker from the pool:")
                .exception(e)
                .build()
                .toString());
      }

      try {
        worker.prepareExecution(key);
      } catch (IOException e) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message("IOException while preparing the execution environment of a worker:")
                .logFile(worker.getLogFile())
                .exception(e)
                .build()
                .toString());
      }

      try {
        request.writeDelimitedTo(worker.getOutputStream());
        worker.getOutputStream().flush();
      } catch (IOException e) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message(
                    "Worker process quit or closed its stdin stream when we tried to send a"
                        + " WorkRequest:")
                .logFile(worker.getLogFile())
                .exception(e)
                .build()
                .toString());
      }

      RecordingInputStream recordingStream = new RecordingInputStream(worker.getInputStream());
      recordingStream.startRecording(4096);
      try {
        // response can be null when the worker has already closed stdout at this point and thus the
        // InputStream is at EOF.
        response = WorkResponse.parseDelimitedFrom(recordingStream);
      } catch (IOException e) {
        // If protobuf couldn't parse the response, try to print whatever the failing worker wrote
        // to stdout - it's probably a stack trace or some kind of error message that will help the
        // user figure out why the compiler is failing.
        recordingStream.readRemaining();
        throw new UserExecException(
            ErrorMessage.builder()
                .message("Worker process returned an unparseable WorkResponse:")
                .logText(recordingStream.getRecordedDataAsString())
                .exception(e)
                .build()
                .toString());
      }

      if (writeOutputFiles != null
          && !writeOutputFiles.compareAndSet(null, WorkerSpawnStrategy.class)) {
        throw new InterruptedException();
      }

      if (response == null) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message("Worker process did not return a WorkResponse:")
                .logFile(worker.getLogFile())
                .logSizeLimit(4096)
                .build()
                .toString());
      }

      try {
        worker.finishExecution(key);
      } catch (IOException e) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message("IOException while finishing worker execution:")
                .exception(e)
                .build()
                .toString());
      }
    } catch (ExecException e) {
      if (worker != null) {
        try {
          workers.invalidateObject(key, worker);
        } catch (IOException e1) {
          // The original exception is more important / helpful, so we'll just ignore this one.
        }
        worker = null;
      }

      throw e;
    } finally {
      if (worker != null) {
        workers.returnObject(key, worker);
      }
    }

    return response;
  }

  @Override
  public String toString() {
    return "worker";
  }

  @Override
  public boolean shouldPropagateExecException() {
    return false;
  }
}
