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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multimap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
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

  private final Path execRoot;
  private final WorkerPool workers;
  private final Multimap<String, String> extraFlags;
  private final EventHandler reporter;
  private final SpawnRunner fallbackRunner;
  private final LocalEnvProvider localEnvProvider;
  private final boolean sandboxUsesExpandedTreeArtifactsInRunfiles;
  private final BinTools binTools;
  private final ResourceManager resourceManager;
  private final RunfilesTreeUpdater runfilesTreeUpdater;

  public WorkerSpawnRunner(
      Path execRoot,
      WorkerPool workers,
      Multimap<String, String> extraFlags,
      EventHandler reporter,
      SpawnRunner fallbackRunner,
      LocalEnvProvider localEnvProvider,
      boolean sandboxUsesExpandedTreeArtifactsInRunfiles,
      BinTools binTools,
      ResourceManager resourceManager,
      RunfilesTreeUpdater runfilesTreeUpdater) {
    this.execRoot = execRoot;
    this.workers = Preconditions.checkNotNull(workers);
    this.extraFlags = extraFlags;
    this.reporter = reporter;
    this.fallbackRunner = fallbackRunner;
    this.localEnvProvider = localEnvProvider;
    this.sandboxUsesExpandedTreeArtifactsInRunfiles = sandboxUsesExpandedTreeArtifactsInRunfiles;
    this.binTools = binTools;
    this.resourceManager = resourceManager;
    this.runfilesTreeUpdater = runfilesTreeUpdater;
  }

  @Override
  public String getName() {
    return "worker";
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException {
    if (!Spawns.supportsWorkers(spawn) && !Spawns.supportsMultiplexWorkers(spawn)) {
      // TODO(ulfjack): Don't circumvent SpawnExecutionPolicy. Either drop the warning here, or
      // provide a mechanism in SpawnExecutionPolicy to report warnings.
      reporter.handle(
          Event.warn(
              String.format(ERROR_MESSAGE_PREFIX + REASON_NO_EXECUTION_INFO, spawn.getMnemonic())));
      return fallbackRunner.exec(spawn, context);
    }

    context.report(ProgressStatus.SCHEDULING, getName());
    return actuallyExec(spawn, context);
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return Spawns.supportsWorkers(spawn) || Spawns.supportsMultiplexWorkers(spawn);
  }

  private SpawnResult actuallyExec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException {
    if (spawn.getToolFiles().isEmpty()) {
      throw new UserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_TOOLS, spawn.getMnemonic()));
    }

    runfilesTreeUpdater.updateRunfilesDirectory(
        execRoot,
        spawn.getRunfilesSupplier(),
        context.getPathResolver(),
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
            spawn,
            context.getArtifactExpander(),
            context.getPathResolver(),
            context.getMetadataProvider());

    HashCode workerFilesCombinedHash = WorkerFilesHash.getCombinedHash(workerFiles);

    SandboxInputs inputFiles =
        SandboxHelpers.processInputFiles(
            context.getInputMapping(sandboxUsesExpandedTreeArtifactsInRunfiles),
            spawn,
            context.getArtifactExpander(),
            execRoot);
    SandboxOutputs outputs = SandboxHelpers.getOutputs(spawn);

    WorkerKey key =
        new WorkerKey(
            workerArgs,
            env,
            execRoot,
            spawn.getMnemonic(),
            workerFilesCombinedHash,
            workerFiles,
            context.speculating(),
            Spawns.supportsMultiplexWorkers(spawn));

    long startTime = System.currentTimeMillis();
    WorkResponse response =
        execInWorker(spawn, key, context, inputFiles, outputs, flagFiles, inputFileCache);
    Duration wallTime = Duration.ofMillis(System.currentTimeMillis() - startTime);

    FileOutErr outErr = context.getFileOutErr();
    response.getOutputBytes().writeTo(outErr.getErrorStream());

    int exitCode = response.getExitCode();
    return new SpawnResult.Builder()
        .setRunnerName(getName())
        .setExitCode(exitCode)
        .setStatus(exitCode == 0 ? SpawnResult.Status.SUCCESS : SpawnResult.Status.NON_ZERO_EXIT)
        .setWallTime(wallTime)
        .build();
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
      SpawnExecutionContext context,
      List<String> flagfiles,
      MetadataProvider inputFileCache,
      int workerId)
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
    return requestBuilder.setRequestId(workerId).build();
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

  private WorkResponse execInWorker(
      Spawn spawn,
      WorkerKey key,
      SpawnExecutionContext context,
      SandboxInputs inputFiles,
      SandboxOutputs outputs,
      List<String> flagFiles,
      MetadataProvider inputFileCache)
      throws InterruptedException, ExecException {
    Worker worker = null;
    WorkResponse response;
    WorkRequest request;

    ActionExecutionMetadata owner = spawn.getResourceOwner();
    try {
      try {
        worker = workers.borrowObject(key);
        request =
            createWorkRequest(spawn, context, flagFiles, inputFileCache, worker.getWorkerId());
      } catch (IOException e) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message("IOException while borrowing a worker from the pool:")
                .exception(e)
                .build()
                .toString());
      }

      try {
        context.prefetchInputs();
      } catch (IOException e) {
        throw new UserExecException(
            ErrorMessage.builder()
                .message("IOException while prefetching for worker:")
                .exception(e)
                .build()
                .toString());
      }

      try (ResourceHandle handle =
          resourceManager.acquireResources(owner, spawn.getLocalResources())) {
        context.report(ProgressStatus.EXECUTING, getName());
        try {
          worker.prepareExecution(inputFiles, outputs, key.getWorkerFilesWithHashes().keySet());
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
          worker.putRequest(request);
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

        try {
          response = worker.getResponse();
        } catch (IOException e) {
          // If protobuf couldn't parse the response, try to print whatever the failing worker wrote
          // to stdout - it's probably a stack trace or some kind of error message that will help
          // the user figure out why the compiler is failing.
          String recordingStreamMessage = worker.getRecordingStreamMessage();
          throw new UserExecException(
              ErrorMessage.builder()
                  .message(
                      "Worker process returned an unparseable WorkResponse!\n\n"
                          + "Did you try to print something to stdout? Workers aren't allowed to "
                          + "do this, as it breaks the protocol between Bazel and the worker "
                          + "process.")
                  .logText(recordingStreamMessage)
                  .exception(e)
                  .build()
                  .toString());
        }
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
        context.lockOutputFiles();
        worker.finishExecution(execRoot);
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
}
