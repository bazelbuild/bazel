// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.LogFile;
import build.bazel.remote.execution.v2.Platform;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.remote.Retrier.RetryException;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import io.grpc.Context;
import io.grpc.Status.Code;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/** A client for the remote execution service. */
@ThreadSafe
class RemoteSpawnRunner implements SpawnRunner {
  private static final int POSIX_TIMEOUT_EXIT_CODE = /*SIGNAL_BASE=*/128 + /*SIGALRM=*/14;

  private final Path execRoot;
  private final RemoteOptions remoteOptions;
  private final ExecutionOptions executionOptions;
  private final AtomicReference<SpawnRunner> fallbackRunner;
  private final boolean verboseFailures;

  @Nullable private final Reporter cmdlineReporter;
  @Nullable private final AbstractRemoteActionCache remoteCache;
  @Nullable private final GrpcRemoteExecutor remoteExecutor;
  @Nullable private final RemoteRetrier retrier;
  private final String buildRequestId;
  private final String commandId;
  private final DigestUtil digestUtil;
  private final Path logDir;

  // Used to ensure that a warning is reported only once.
  private final AtomicBoolean warningReported = new AtomicBoolean();

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions remoteOptions,
      ExecutionOptions executionOptions,
      AtomicReference<SpawnRunner> fallbackRunner,
      boolean verboseFailures,
      @Nullable Reporter cmdlineReporter,
      String buildRequestId,
      String commandId,
      @Nullable AbstractRemoteActionCache remoteCache,
      @Nullable GrpcRemoteExecutor remoteExecutor,
      @Nullable RemoteRetrier retrier,
      DigestUtil digestUtil,
      Path logDir) {
    this.execRoot = execRoot;
    this.remoteOptions = remoteOptions;
    this.executionOptions = executionOptions;
    this.fallbackRunner = fallbackRunner;
    this.remoteCache = remoteCache;
    this.remoteExecutor = remoteExecutor;
    this.verboseFailures = verboseFailures;
    this.cmdlineReporter = cmdlineReporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.retrier = retrier;
    this.digestUtil = digestUtil;
    this.logDir = logDir;
  }

  @Override
  public String getName() {
    return "remote";
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    if (!Spawns.mayBeExecutedRemotely(spawn) || remoteCache == null) {
      return execLocally(spawn, context);
    }

    context.report(ProgressStatus.EXECUTING, getName());
    // Temporary hack: the TreeNodeRepository should be created and maintained upstream!
    MetadataProvider inputFileCache = context.getMetadataProvider();
    TreeNodeRepository repository = new TreeNodeRepository(execRoot, inputFileCache, digestUtil);
    SortedMap<PathFragment, ActionInput> inputMap = context.getInputMapping(true);
    TreeNode inputRoot = repository.buildFromActionInputs(inputMap);
    repository.computeMerkleDigests(inputRoot);
    maybeWriteParamFilesLocally(spawn);
    Command command = buildCommand(
        spawn.getOutputFiles(),
        spawn.getArguments(),
        spawn.getEnvironment(),
        spawn.getExecutionPlatform());
    Action action =
        buildAction(
            digestUtil.compute(command),
            repository.getMerkleDigest(inputRoot),
            context.getTimeout(),
            Spawns.mayBeCached(spawn));

    // Look up action cache, and reuse the action output if it is found.
    ActionKey actionKey = digestUtil.computeActionKey(action);
    Context withMetadata =
        TracingMetadataUtils.contextWithMetadata(buildRequestId, commandId, actionKey);
    Context previous = withMetadata.attach();
    try {
      boolean acceptCachedResult = remoteOptions.remoteAcceptCached && Spawns.mayBeCached(spawn);
      boolean uploadLocalResults = remoteOptions.remoteUploadLocalResults;

      try {
        // Try to lookup the action in the action cache.
        ActionResult cachedResult =
            acceptCachedResult ? remoteCache.getCachedActionResult(actionKey) : null;
        if (cachedResult != null) {
          if (cachedResult.getExitCode() != 0) {
            // The remote cache must never serve a failed action.
            throw new EnvironmentalExecException(
                "The remote cache is in an invalid state as it"
                    + " served a failed action. Hash of the action: "
                    + actionKey.getDigest());
          }
          try {
            return downloadRemoteResults(cachedResult, context.getFileOutErr())
                .setCacheHit(true)
                .setRunnerName("remote cache hit")
                .build();
          } catch (RetryException e) {
            if (!AbstractRemoteActionCache.causedByCacheMiss(e)) {
              throw e;
            }
            // No cache hit, so we fall through to local or remote execution.
            // We set acceptCachedResult to false in order to force the action re-execution.
            acceptCachedResult = false;
          }
        }
      } catch (IOException e) {
        return execLocallyAndUploadOrFail(
            spawn, context, inputMap, actionKey, action, command, uploadLocalResults, e);
      }

      if (remoteExecutor == null) {
        // Remote execution is disabled and so execute the spawn on the local machine.
        return execLocallyAndUpload(
            spawn, context, inputMap, remoteCache, actionKey, action, command, uploadLocalResults);
       }

      ExecuteRequest request =
          ExecuteRequest.newBuilder()
              .setInstanceName(remoteOptions.remoteInstanceName)
              .setActionDigest(actionKey.getDigest())
              .setSkipCacheLookup(!acceptCachedResult)
              .build();
      try {
        return retrier.execute(
            () -> {
              // Upload the command and all the inputs into the remote cache.
             remoteCache.ensureInputsPresent(repository, execRoot, inputRoot, action, command);
              ExecuteResponse reply = remoteExecutor.executeRemotely(request);
              maybeDownloadServerLogs(reply, actionKey);

              return downloadRemoteResults(reply.getResult(), context.getFileOutErr())
                  .setRunnerName(reply.getCachedResult() ? "remote cache hit" : getName())
                  .setCacheHit(reply.getCachedResult())
                  .build();
            });
      } catch (IOException e) {
        return execLocallyAndUploadOrFail(
            spawn, context, inputMap, actionKey, action, command, uploadLocalResults, e);
      }

    } finally {
      withMetadata.detach(previous);
    }
  }

  private void maybeWriteParamFilesLocally(Spawn spawn) throws IOException {
    if (!executionOptions.materializeParamFiles) {
      return;
    }
    for (ActionInput actionInput : spawn.getInputFiles()) {
      if (actionInput instanceof ParamFileActionInput) {
        ParamFileActionInput paramFileActionInput = (ParamFileActionInput) actionInput;
        Path outputPath = execRoot.getRelative(paramFileActionInput.getExecPath());
        if (outputPath.exists()) {
          outputPath.delete();
        }
        outputPath.getParentDirectory().createDirectoryAndParents();
        try (OutputStream out = outputPath.getOutputStream()) {
          paramFileActionInput.writeTo(out);
        }
      }
    }
  }

  private void maybeDownloadServerLogs(ExecuteResponse resp, ActionKey actionKey)
      throws InterruptedException {
    ActionResult result = resp.getResult();
    if (resp.getServerLogsCount() > 0
        && (result.getExitCode() != 0 || resp.getStatus().getCode() != Code.OK.value())) {
      Path parent = logDir.getRelative(actionKey.getDigest().getHash());
      Path logPath = null;
      int logCount = 0;
      for (Map.Entry<String, LogFile> e : resp.getServerLogsMap().entrySet()) {
        if (e.getValue().getHumanReadable()) {
          logPath = parent.getRelative(e.getKey());
          logCount++;
          try {
            getFromFuture(remoteCache.downloadFile(logPath, e.getValue().getDigest()));
          } catch (IOException ex) {
            reportOnce(Event.warn("Failed downloading server logs from the remote cache."));
          }
        }
      }
      if (logCount > 0 && verboseFailures) {
        report(
            Event.info("Server logs of failing action:\n   " + (logCount > 1 ? parent : logPath)));
      }
    }
  }

  private SpawnResult.Builder downloadRemoteResults(ActionResult result, FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    remoteCache.download(result, execRoot, outErr);
    int exitCode = result.getExitCode();
    return new SpawnResult.Builder()
        .setStatus(exitCode == 0 ? Status.SUCCESS : Status.NON_ZERO_EXIT)
        .setExitCode(exitCode);
  }

  private SpawnResult execLocally(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    return fallbackRunner.get().exec(spawn, context);
  }

  private SpawnResult execLocallyAndUploadOrFail(
      Spawn spawn,
      SpawnExecutionContext context,
      SortedMap<PathFragment, ActionInput> inputMap,
      ActionKey actionKey,
      Action action,
      Command command,
      boolean uploadLocalResults,
      IOException cause)
      throws ExecException, InterruptedException, IOException {
    // Regardless of cause, if we are interrupted, we should stop without displaying a user-visible
    // failure/stack trace.
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException();
    }
    if (remoteOptions.remoteLocalFallback
        && !(cause instanceof RetryException
            && RemoteRetrierUtils.causedByExecTimeout((RetryException) cause))) {
      return execLocallyAndUpload(
          spawn, context, inputMap, remoteCache, actionKey, action, command, uploadLocalResults);
    }
    return handleError(cause, context.getFileOutErr(), actionKey);
  }

  private SpawnResult handleError(IOException exception, FileOutErr outErr, ActionKey actionKey)
      throws ExecException, InterruptedException, IOException {
    final Throwable cause = exception.getCause();
    if (cause instanceof ExecutionStatusException) {
      ExecutionStatusException e = (ExecutionStatusException) cause;
      if (e.getResponse() != null) {
        ExecuteResponse resp = e.getResponse();
        maybeDownloadServerLogs(resp, actionKey);
        if (resp.hasResult()) {
          // We try to download all (partial) results even on server error, for debuggability.
          remoteCache.download(resp.getResult(), execRoot, outErr);
        }
      }
      if (e.isExecutionTimeout()) {
        return new SpawnResult.Builder()
            .setRunnerName(getName())
            .setStatus(Status.TIMEOUT)
            .setExitCode(POSIX_TIMEOUT_EXIT_CODE)
            .build();
      }
    }
    final Status status;
    if (exception instanceof RetryException
        && RemoteRetrierUtils.causedByStatus((RetryException) exception, Code.UNAVAILABLE)) {
      status = Status.EXECUTION_FAILED_CATASTROPHICALLY;
    } else if (exception instanceof CacheNotFoundException
        || cause instanceof CacheNotFoundException) {
      status = Status.REMOTE_CACHE_FAILED;
    } else {
      status = Status.EXECUTION_FAILED;
    }
    throw new SpawnExecException(
        verboseFailures ? Throwables.getStackTraceAsString(exception) : exception.getMessage(),
        new SpawnResult.Builder()
            .setRunnerName(getName())
            .setStatus(status)
            .setExitCode(ExitCode.REMOTE_ERROR.getNumericExitCode())
            .setFailureMessage(exception.getMessage())
            .build(),
        /* forciblyRunRemotely= */ false);
  }

  static Action buildAction(
      Digest command,
      Digest inputRoot,
      Duration timeout,
      boolean cacheable) {

    Action.Builder action = Action.newBuilder();
    action.setCommandDigest(command);
    action.setInputRootDigest(inputRoot);
    if (!timeout.isZero()) {
      action.setTimeout(com.google.protobuf.Duration.newBuilder().setSeconds(timeout.getSeconds()));
    }
    if (!cacheable) {
      action.setDoNotCache(true);
    }
    return action.build();
  }

  static Platform parsePlatform(Label platformLabel, @Nullable String platformDescription) {
    Platform.Builder platformBuilder = Platform.newBuilder();
    try {
      if (platformDescription != null) {
        TextFormat.getParser().merge(platformDescription, platformBuilder);
      }
    } catch (ParseException e) {
      throw new IllegalArgumentException(
          String.format(
              "Failed to parse remote_execution_properties from platform %s", platformLabel),
          e);
    }
    List<Platform.Property> properties = platformBuilder.getPropertiesList();
    platformBuilder.clearProperties();
    platformBuilder.addAllProperties(
        Ordering.from(Comparator.comparing(Platform.Property::getName)).sortedCopy(properties));
    return platformBuilder.build();
  }

  static Command buildCommand(
      Collection<? extends ActionInput> outputs,
      List<String> arguments,
      ImmutableMap<String, String> env,
      @Nullable PlatformInfo executionPlatform) {
    Command.Builder command = Command.newBuilder();
    ArrayList<String> outputFiles = new ArrayList<>();
    ArrayList<String> outputDirectories = new ArrayList<>();
    for (ActionInput output : outputs) {
      String pathString = output.getExecPathString();
      if (output instanceof Artifact && ((Artifact) output).isTreeArtifact()) {
        outputDirectories.add(pathString);
      } else {
        outputFiles.add(pathString);
      }
    }
    Collections.sort(outputFiles);
    Collections.sort(outputDirectories);
    command.addAllOutputFiles(outputFiles);
    command.addAllOutputDirectories(outputDirectories);

    // Get the remote platform properties.
    if (executionPlatform != null) {
      Platform platform =
          parsePlatform(executionPlatform.label(), executionPlatform.remoteExecutionProperties());
      command.setPlatform(platform);
    }
    command.addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(env.keySet());
    for (String var : variables) {
      command.addEnvironmentVariablesBuilder().setName(var).setValue(env.get(var));
    }
    return command.build();
  }

  private Map<Path, Long> getInputCtimes(SortedMap<PathFragment, ActionInput> inputMap) {
    HashMap<Path, Long>  ctimes = new HashMap<>();
    for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
      ActionInput input = e.getValue();
      if (input instanceof VirtualActionInput) {
        continue;
      }
      Path path = execRoot.getRelative(input.getExecPathString());
      try {
        ctimes.put(path, path.stat().getLastChangeTime());
      } catch (IOException ex) {
        // Put a token value indicating an exception; this is used so that if the exception
        // is raised both before and after the execution, it is ignored, but if it is raised only
        // one of the times, it triggers a remote cache upload skip.
        ctimes.put(path, -1L);
      }
    }
    return ctimes;
  }

  @VisibleForTesting
  SpawnResult execLocallyAndUpload(
      Spawn spawn,
      SpawnExecutionContext context,
      SortedMap<PathFragment, ActionInput> inputMap,
      AbstractRemoteActionCache remoteCache,
      ActionKey actionKey,
      Action action,
      Command command,
      boolean uploadLocalResults)
      throws ExecException, IOException, InterruptedException {
    Map<Path, Long> ctimesBefore = getInputCtimes(inputMap);
    SpawnResult result = execLocally(spawn, context);
    Map<Path, Long> ctimesAfter = getInputCtimes(inputMap);
    for (Map.Entry<Path, Long> e : ctimesBefore.entrySet()) {
      // Skip uploading to remote cache, because an input was modified during execution.
      if (!ctimesAfter.get(e.getKey()).equals(e.getValue())) {
        return result;
      }
    }
    if (!uploadLocalResults) {
      return result;
    }
    boolean uploadAction =
        Spawns.mayBeCached(spawn)
            && Status.SUCCESS.equals(result.status())
            && result.exitCode() == 0;
    Collection<Path> outputFiles = resolveActionInputs(execRoot, spawn.getOutputFiles());
    try {
      remoteCache.upload(
          actionKey, action, command, execRoot, outputFiles, context.getFileOutErr(), uploadAction);
    } catch (IOException e) {
      if (verboseFailures) {
        report(Event.debug("Upload to remote cache failed: " + e.getMessage()));
      } else {
        reportOnce(Event.warn("Some artifacts failed be uploaded to the remote cache."));
      }
    }
    return result;
  }

  private void reportOnce(Event evt) {
    if (warningReported.compareAndSet(false, true)) {
      report(evt);
    }
  }

  private void report(Event evt) {
    if (cmdlineReporter != null) {
      cmdlineReporter.handle(evt);
    }
  }

  /**
   * Resolve a collection of {@link com.google.devtools.build.lib.actions.ActionInput}s to {@link
   * Path}s.
   */
  static Collection<Path> resolveActionInputs(
      Path execRoot, Collection<? extends ActionInput> actionInputs) {
    return actionInputs
        .stream()
        .map((inp) -> execRoot.getRelative(inp.getExecPath()))
        .collect(ImmutableList.toImmutableList());
  }
}
