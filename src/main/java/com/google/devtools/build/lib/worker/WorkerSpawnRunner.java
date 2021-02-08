// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

/**
 * A spawn runner that launches Spawns the first time they are used in a persistent mode and then
 * shards work over all the processes.
 */
final class WorkerSpawnRunner implements SpawnRunner {
  public static final String ERROR_MESSAGE_PREFIX =
      "Worker strategy cannot execute this %s action, ";
  public static final String REASON_NO_FLAGFILE =
      "because the command-line arguments do not contain at least one @flagfile or --flagfile=";
  public static final String REASON_NO_TOOLS = "because the action has no tools";
  public static final String REASON_NO_EXECUTION_INFO =
      "because the action's execution info does not contain 'supports-workers=1'";

  /** Pattern for @flagfile.txt and --flagfile=flagfile.txt */
  private static final Pattern FLAG_FILE_PATTERN = Pattern.compile("(?:@|--?flagfile=)(.+)");

  private final SandboxHelpers helpers;
  private final Path execRoot;
  private final WorkerPool workers;
  private final boolean multiplex;
  private final ExtendedEventHandler reporter;
  private final LocalEnvProvider localEnvProvider;
  private final BinTools binTools;
  private final ResourceManager resourceManager;
  private final RunfilesTreeUpdater runfilesTreeUpdater;
  private final WorkerOptions workerOptions;
  private final AtomicInteger requestIdCounter = new AtomicInteger(1);

  public WorkerSpawnRunner(
      SandboxHelpers helpers,
      Path execRoot,
      WorkerPool workers,
      boolean multiplex,
      ExtendedEventHandler reporter,
      LocalEnvProvider localEnvProvider,
      BinTools binTools,
      ResourceManager resourceManager,
      RunfilesTreeUpdater runfilesTreeUpdater,
      WorkerOptions workerOptions) {
    this.helpers = helpers;
    this.execRoot = execRoot;
    this.workers = Preconditions.checkNotNull(workers);
    this.multiplex = multiplex;
    this.reporter = reporter;
    this.localEnvProvider = localEnvProvider;
    this.binTools = binTools;
    this.resourceManager = resourceManager;
    this.runfilesTreeUpdater = runfilesTreeUpdater;
    this.workerOptions = workerOptions;
  }

  @Override
  public String getName() {
    return "worker";
  }

  @Override
  public boolean canExec(Spawn spawn) {
    if (!Spawns.supportsWorkers(spawn) && !Spawns.supportsMultiplexWorkers(spawn)) {
      return false;
    }
    if (spawn.getToolFiles().isEmpty()) {
      return false;
    }
    return true;
  }

  @Override
  public boolean handlesCaching() {
    return false;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException {
    context.report(
        ProgressStatus.SCHEDULING,
        WorkerKey.makeWorkerTypeName(Spawns.supportsMultiplexWorkers(spawn)));
    if (spawn.getToolFiles().isEmpty()) {
      throw createUserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_TOOLS, spawn.getMnemonic()),
          Code.NO_TOOLS);
    }

    Instant startTime = Instant.now();

    runfilesTreeUpdater.updateRunfilesDirectory(
        execRoot,
        spawn.getRunfilesSupplier(),
        binTools,
        spawn.getEnvironment(),
        context.getFileOutErr());

    // We assume that the spawn to be executed always gets at least one @flagfile.txt or
    // --flagfile=flagfile.txt argument, which contains the flags related to the work itself (as
    // opposed to start-up options for the executed tool). Thus, we can extract those elements from
    // its args and put them into the WorkRequest instead.
    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> workerArgs = splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);
    ImmutableMap<String, String> env =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

    MetadataProvider inputFileCache = context.getMetadataProvider();

    SortedMap<PathFragment, HashCode> workerFiles =
        WorkerFilesHash.getWorkerFilesWithHashes(
            spawn, context.getArtifactExpander(), context.getMetadataProvider());

    HashCode workerFilesCombinedHash = WorkerFilesHash.getCombinedHash(workerFiles);

    SandboxInputs inputFiles =
        helpers.processInputFiles(
            context.getInputMapping(PathFragment.EMPTY_FRAGMENT),
            spawn,
            context.getArtifactExpander(),
            execRoot);
    SandboxOutputs outputs = helpers.getOutputs(spawn);

    WorkerProtocolFormat protocolFormat = Spawns.getWorkerProtocolFormat(spawn);
    if (!workerOptions.experimentalJsonWorkerProtocol) {
      if (protocolFormat == WorkerProtocolFormat.JSON) {
        throw new IOException(
            "Persistent worker protocol format must be set to proto unless"
                + " --experimental_worker_allow_json_protocol is used");
      }
    }

    WorkerKey key =
        new WorkerKey(
            workerArgs,
            env,
            execRoot,
            Spawns.getWorkerKeyMnemonic(spawn),
            workerFilesCombinedHash,
            workerFiles,
            context.speculating(),
            multiplex && Spawns.supportsMultiplexWorkers(spawn),
            protocolFormat);

    SpawnMetrics.Builder spawnMetrics =
        SpawnMetrics.Builder.forWorkerExec()
            .setInputFiles(inputFiles.getFiles().size() + inputFiles.getSymlinks().size());
    WorkResponse response =
        execInWorker(
            spawn, key, context, inputFiles, outputs, flagFiles, inputFileCache, spawnMetrics);

    FileOutErr outErr = context.getFileOutErr();
    response.getOutputBytes().writeTo(outErr.getErrorStream());

    Duration wallTime = Duration.between(startTime, Instant.now());

    int exitCode = response.getExitCode();
    SpawnResult.Builder builder =
        new SpawnResult.Builder()
            .setRunnerName(getName())
            .setExitCode(exitCode)
            .setStatus(exitCode == 0 ? Status.SUCCESS : Status.NON_ZERO_EXIT)
            .setWallTime(wallTime)
            .setSpawnMetrics(spawnMetrics.setTotalTime(wallTime).build());
    if (exitCode != 0) {
      builder.setFailureDetail(
          FailureDetail.newBuilder()
              .setMessage("worker spawn failed for " + spawn.getMnemonic())
              .setSpawn(
                  FailureDetails.Spawn.newBuilder()
                      .setCode(FailureDetails.Spawn.Code.NON_ZERO_EXIT)
                      .setSpawnExitCode(exitCode))
              .build());
    }
    SpawnResult result = builder.build();
    reporter.post(new SpawnExecutedEvent(spawn, result, startTime));
    return result;
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
      throw createUserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_FLAGFILE, spawn.getMnemonic()),
          Code.NO_FLAGFILE);
    }

    ImmutableList.Builder<String> mnemonicFlags = ImmutableList.builder();

    workerOptions.workerExtraFlags.stream()
        .filter(entry -> entry.getKey().equals(spawn.getMnemonic()))
        .forEach(entry -> mnemonicFlags.add(entry.getValue()));

    return workerArgs
        .add("--persistent_worker")
        .addAll(MoreObjects.firstNonNull(mnemonicFlags.build(), ImmutableList.<String>of()))
        .build();
  }

  private WorkRequest createWorkRequest(
      Spawn spawn,
      SpawnExecutionContext context,
      List<String> flagfiles,
      MetadataProvider inputFileCache,
      WorkerKey key)
      throws IOException {
    WorkRequest.Builder requestBuilder = WorkRequest.newBuilder();
    for (String flagfile : flagfiles) {
      expandArgument(execRoot, flagfile, requestBuilder);
    }

    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(spawn.getInputFiles(), context.getArtifactExpander());

    for (ActionInput input : inputs) {
      byte[] digestBytes = inputFileCache.getMetadata(input).getDigest();
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
    if (key.getProxied()) {
      requestBuilder.setRequestId(requestIdCounter.getAndIncrement());
    }
    return requestBuilder.build();
  }

  /**
   * Recursively expands arguments by replacing @filename args with the contents of the referenced
   * files. The @ itself can be escaped with @@. This deliberately does not expand --flagfile= style
   * arguments, because we want to get rid of the expansion entirely at some point in time.
   *
   * <p>Also check that the argument is not an external repository label, because they start with
   * `@` and are not flagfile locations.
   *
   * @param execRoot the current execroot of the build (relative paths will be assumed to be
   *     relative to this directory).
   * @param arg the argument to expand.
   * @param requestBuilder the WorkRequest to whose arguments the expanded arguments will be added.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  static void expandArgument(Path execRoot, String arg, WorkRequest.Builder requestBuilder)
      throws IOException {
    if (arg.startsWith("@") && !arg.startsWith("@@") && !isExternalRepositoryLabel(arg)) {
      for (String line :
          Files.readAllLines(
              Paths.get(execRoot.getRelative(arg.substring(1)).getPathString()), UTF_8)) {
        expandArgument(execRoot, line, requestBuilder);
      }
    } else {
      requestBuilder.addArguments(arg);
    }
  }

  private static boolean isExternalRepositoryLabel(String arg) {
    return arg.matches("^@.*//.*");
  }

  private static UserExecException createEmptyResponseException(Path logfile) {
    String message =
        ErrorMessage.builder()
            .message("Worker process did not return a WorkResponse:")
            .logFile(logfile)
            .logSizeLimit(4096)
            .build()
            .toString();
    return createUserExecException(message, Code.NO_RESPONSE);
  }

  private static UserExecException createUnparsableResponseException(
      String recordingStreamMessage, Path logfile, Exception e) {
    String message =
        ErrorMessage.builder()
            .message(
                "Worker process returned an unparseable WorkResponse!\n\n"
                    + "Did you try to print something to stdout? Workers aren't allowed to "
                    + "do this, as it breaks the protocol between Bazel and the worker "
                    + "process.\n\n"
                    + "---8<---8<--- Start of response ---8<---8<---\n"
                    + recordingStreamMessage
                    + "---8<---8<--- End of response ---8<---8<---\n\n")
            .logFile(logfile)
            .logSizeLimit(8192)
            .exception(e)
            .build()
            .toString();
    return createUserExecException(message, Code.PARSE_RESPONSE_FAILURE);
  }

  @VisibleForTesting
  WorkResponse execInWorker(
      Spawn spawn,
      WorkerKey key,
      SpawnExecutionContext context,
      SandboxInputs inputFiles,
      SandboxOutputs outputs,
      List<String> flagFiles,
      MetadataProvider inputFileCache,
      SpawnMetrics.Builder spawnMetrics)
      throws InterruptedException, ExecException {
    Worker worker = null;
    WorkResponse response;
    WorkRequest request;

    ActionExecutionMetadata owner = spawn.getResourceOwner();
    try {
      Stopwatch setupInputsStopwatch = Stopwatch.createStarted();
      try {
        inputFiles.materializeVirtualInputs(execRoot);
      } catch (IOException e) {
        restoreInterrupt(e);
        String message = "IOException while materializing virtual inputs:";
        throw createUserExecException(e, message, Code.VIRTUAL_INPUT_MATERIALIZATION_FAILURE);
      }

      try {
        context.prefetchInputs();
      } catch (IOException e) {
        restoreInterrupt(e);
        String message = "IOException while prefetching for worker:";
        throw createUserExecException(e, message, Code.PREFETCH_FAILURE);
      }
      Duration setupInputsTime = setupInputsStopwatch.elapsed();

      Stopwatch queueStopwatch = Stopwatch.createStarted();
      try {
        worker = workers.borrowObject(key);
        worker.setReporter(workerOptions.workerVerbose ? reporter : null);
        request = createWorkRequest(spawn, context, flagFiles, inputFileCache, key);
      } catch (IOException e) {
        restoreInterrupt(e);
        String message = "IOException while borrowing a worker from the pool:";
        throw createUserExecException(e, message, Code.BORROW_FAILURE);
      }

      try (ResourceHandle handle =
          resourceManager.acquireResources(owner, spawn.getLocalResources())) {
        // We acquired a worker and resources -- mark that as queuing time.
        spawnMetrics.setQueueTime(queueStopwatch.elapsed());

        context.report(ProgressStatus.EXECUTING, WorkerKey.makeWorkerTypeName(key.getProxied()));
        try {
          // We consider `prepareExecution` to be also part of setup.
          Stopwatch prepareExecutionStopwatch = Stopwatch.createStarted();
          worker.prepareExecution(inputFiles, outputs, key.getWorkerFilesWithHashes().keySet());
          spawnMetrics.setSetupTime(setupInputsTime.plus(prepareExecutionStopwatch.elapsed()));
        } catch (IOException e) {
          restoreInterrupt(e);
          String message =
              ErrorMessage.builder()
                  .message("IOException while preparing the execution environment of a worker:")
                  .logFile(worker.getLogFile())
                  .exception(e)
                  .build()
                  .toString();
          throw createUserExecException(message, Code.PREPARE_FAILURE);
        }

        Stopwatch executionStopwatch = Stopwatch.createStarted();
        try {
          worker.putRequest(request);
        } catch (IOException e) {
          restoreInterrupt(e);
          String message =
              ErrorMessage.builder()
                  .message(
                      "Worker process quit or closed its stdin stream when we tried to send a"
                          + " WorkRequest:")
                  .logFile(worker.getLogFile())
                  .exception(e)
                  .build()
                  .toString();
          throw createUserExecException(message, Code.REQUEST_FAILURE);
        }

        try {
          response = worker.getResponse(request.getRequestId());
        } catch (IOException e) {
          restoreInterrupt(e);
          // If protobuf or json reader couldn't parse the response, try to print whatever the
          // failing worker wrote to stdout - it's probably a stack trace or some kind of error
          // message that will help the user figure out why the compiler is failing.
          String recordingStreamMessage = worker.getRecordingStreamMessage();
          if (recordingStreamMessage.isEmpty()) {
            throw createEmptyResponseException(worker.getLogFile());
          } else {
            throw createUnparsableResponseException(recordingStreamMessage, worker.getLogFile(), e);
          }
        }
        spawnMetrics.setExecutionWallTime(executionStopwatch.elapsed());
      }

      if (response == null) {
        throw createEmptyResponseException(worker.getLogFile());
      }

      try {
        Stopwatch processOutputsStopwatch = Stopwatch.createStarted();
        context.lockOutputFiles();
        worker.finishExecution(execRoot);
        spawnMetrics.setProcessOutputsTime(processOutputsStopwatch.elapsed());
      } catch (IOException e) {
        restoreInterrupt(e);
        String message =
            ErrorMessage.builder()
                .message("IOException while finishing worker execution:")
                .logFile(worker.getLogFile())
                .exception(e)
                .build()
                .toString();
        throw createUserExecException(message, Code.FINISH_FAILURE);
      }
    } catch (UserExecException e) {
      if (worker != null) {
        try {
          workers.invalidateObject(key, worker);
        } catch (IOException e1) {
          // The original exception is more important / helpful, so we'll just ignore this one.
          restoreInterrupt(e1);
        } finally {
          worker = null;
        }
      }

      throw e;
    } finally {
      if (worker != null) {
        workers.returnObject(key, worker);
      }
    }

    return response;
  }

  private static void restoreInterrupt(IOException e) {
    if (e instanceof InterruptedIOException) {
      Thread.currentThread().interrupt();
    }
  }

  private static UserExecException createUserExecException(
      IOException e, String message, Code detailedCode) {
    return createUserExecException(
        ErrorMessage.builder().message(message).exception(e).build().toString(), detailedCode);
  }

  private static UserExecException createUserExecException(String message, Code detailedCode) {
    return new UserExecException(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setWorker(FailureDetails.Worker.newBuilder().setCode(detailedCode))
            .build());
  }
}
