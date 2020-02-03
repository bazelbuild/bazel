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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_DOWNLOAD;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_EXECUTION;
import static com.google.devtools.build.lib.profiler.ProfilerTask.UPLOAD_TIME;
import static com.google.devtools.build.lib.remote.util.Utils.createSpawnResult;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getInMemoryOutputPath;
import static com.google.devtools.build.lib.remote.util.Utils.hasFilesToDownload;
import static com.google.devtools.build.lib.remote.util.Utils.shouldDownloadAllSpawnOutputs;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.LogFile;
import build.bazel.remote.execution.v2.Platform;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.analysis.platform.PlatformUtils;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.RemoteLocalFallbackRegistry;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Message;
import com.google.rpc.PreconditionFailure;
import com.google.rpc.PreconditionFailure.Violation;
import io.grpc.Context;
import io.grpc.Status.Code;
import io.grpc.protobuf.StatusProto;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** A client for the remote execution service. */
@ThreadSafe
public class RemoteSpawnRunner implements SpawnRunner {

  private static final int POSIX_TIMEOUT_EXIT_CODE = /*SIGNAL_BASE=*/ 128 + /*SIGALRM=*/ 14;

  private static final String VIOLATION_TYPE_MISSING = "MISSING";

  private static boolean retriableExecErrors(Exception e) {
    if (e instanceof CacheNotFoundException || e.getCause() instanceof CacheNotFoundException) {
      return true;
    }
    if (!RemoteRetrierUtils.causedByStatus(e, Code.FAILED_PRECONDITION)) {
      return false;
    }
    com.google.rpc.Status status = StatusProto.fromThrowable(e);
    if (status == null || status.getDetailsCount() == 0) {
      return false;
    }
    for (Any details : status.getDetailsList()) {
      PreconditionFailure f;
      try {
        f = details.unpack(PreconditionFailure.class);
      } catch (InvalidProtocolBufferException protoEx) {
        return false;
      }
      if (f.getViolationsCount() == 0) {
        return false; // Generally shouldn't happen
      }
      for (Violation v : f.getViolationsList()) {
        if (!v.getType().equals(VIOLATION_TYPE_MISSING)) {
          return false;
        }
      }
    }
    return true; // if *all* > 0 violations have type MISSING
  }

  private final Path execRoot;
  private final RemoteOptions remoteOptions;
  private final ExecutionOptions executionOptions;
  private final boolean verboseFailures;

  @Nullable private final Reporter cmdlineReporter;
  private final RemoteExecutionCache remoteCache;
  @Nullable private final GrpcRemoteExecutor remoteExecutor;
  private final RemoteRetrier retrier;
  private final String buildRequestId;
  private final String commandId;
  private final DigestUtil digestUtil;
  private final Path logDir;

  /**
   * If {@link RemoteOutputsMode#TOPLEVEL} is specified it contains the artifacts that should be
   * downloaded.
   */
  private final ImmutableSet<ActionInput> filesToDownload;

  // Used to ensure that a warning is reported only once.
  private final AtomicBoolean warningReported = new AtomicBoolean();

  RemoteSpawnRunner(
      Path execRoot,
      RemoteOptions remoteOptions,
      ExecutionOptions executionOptions,
      boolean verboseFailures,
      @Nullable Reporter cmdlineReporter,
      String buildRequestId,
      String commandId,
      RemoteExecutionCache remoteCache,
      GrpcRemoteExecutor remoteExecutor,
      ListeningScheduledExecutorService retryService,
      DigestUtil digestUtil,
      Path logDir,
      ImmutableSet<ActionInput> filesToDownload) {
    this.execRoot = execRoot;
    this.remoteOptions = remoteOptions;
    this.executionOptions = executionOptions;
    this.remoteCache = Preconditions.checkNotNull(remoteCache, "remoteCache");
    this.remoteExecutor = Preconditions.checkNotNull(remoteExecutor, "remoteExecutor");
    this.verboseFailures = verboseFailures;
    this.cmdlineReporter = cmdlineReporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.retrier = createExecuteRetrier(remoteOptions, retryService);
    this.digestUtil = digestUtil;
    this.logDir = logDir;
    this.filesToDownload = Preconditions.checkNotNull(filesToDownload, "filesToDownload");
  }

  @Override
  public String getName() {
    return "remote";
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {

    boolean spawnCacheableRemotely = Spawns.mayBeCachedRemotely(spawn);
    boolean uploadLocalResults = remoteOptions.remoteUploadLocalResults && spawnCacheableRemotely;
    boolean acceptCachedResult = remoteOptions.remoteAcceptCached && spawnCacheableRemotely;

    context.report(ProgressStatus.EXECUTING, getName());
    RemoteOutputsMode remoteOutputsMode = remoteOptions.remoteOutputsMode;
    SortedMap<PathFragment, ActionInput> inputMap = context.getInputMapping(true);
    final MerkleTree merkleTree =
        MerkleTree.build(inputMap, context.getMetadataProvider(), execRoot, digestUtil);
    maybeWriteParamFilesLocally(spawn);

    // Get the remote platform properties.
    Platform platform = PlatformUtils.getPlatformProto(spawn, remoteOptions);

    Command command =
        buildCommand(
            spawn.getOutputFiles(),
            spawn.getArguments(),
            spawn.getEnvironment(),
            platform,
            /* workingDirectory= */ null);
    Digest commandHash = digestUtil.compute(command);
    Action action =
        buildAction(
            commandHash, merkleTree.getRootDigest(), context.getTimeout(), spawnCacheableRemotely);
    ActionKey actionKey = digestUtil.computeActionKey(action);

    Preconditions.checkArgument(
        Spawns.mayBeExecutedRemotely(spawn), "Spawn can't be executed remotely. This is a bug.");
    // Look up action cache, and reuse the action output if it is found.
    Context withMetadata =
        TracingMetadataUtils.contextWithMetadata(buildRequestId, commandId, actionKey);
    Context previous = withMetadata.attach();
    Profiler prof = Profiler.instance();
    try {
      try {
        // Try to lookup the action in the action cache.
        ActionResult cachedResult;
        try (SilentCloseable c = prof.profile(ProfilerTask.REMOTE_CACHE_CHECK, "check cache hit")) {
          cachedResult = acceptCachedResult ? remoteCache.downloadActionResult(actionKey) : null;
        }
        if (cachedResult != null) {
          if (cachedResult.getExitCode() != 0) {
            // Failed actions are treated as a cache miss mostly in order to avoid caching flaky
            // actions (tests).
            // Set acceptCachedResult to false in order to force the action re-execution
            acceptCachedResult = false;
          } else {
            try {
              return downloadAndFinalizeSpawnResult(
                  cachedResult, /* cacheHit= */ true, spawn, context, remoteOutputsMode);
            } catch (CacheNotFoundException e) {
              // No cache hit, so we fall through to local or remote execution.
              // We set acceptCachedResult to false in order to force the action re-execution.
              acceptCachedResult = false;
            }
          }
        }
      } catch (IOException e) {
        return execLocallyAndUploadOrFail(
            spawn, context, inputMap, actionKey, action, command, uploadLocalResults, e);
      }

      ExecuteRequest.Builder requestBuilder =
          ExecuteRequest.newBuilder()
              .setInstanceName(remoteOptions.remoteInstanceName)
              .setActionDigest(actionKey.getDigest())
              .setSkipCacheLookup(!acceptCachedResult);
      if (remoteOptions.remoteResultCachePriority != 0) {
        requestBuilder
            .getResultsCachePolicyBuilder()
            .setPriority(remoteOptions.remoteResultCachePriority);
      }
      if (remoteOptions.remoteExecutionPriority != 0) {
        requestBuilder
            .getExecutionPolicyBuilder()
            .setPriority(remoteOptions.remoteExecutionPriority);
      }
      try {
        return retrier.execute(
            () -> {
              ExecuteRequest request = requestBuilder.build();

              // Upload the command and all the inputs into the remote cache.
              try (SilentCloseable c = prof.profile(UPLOAD_TIME, "upload missing inputs")) {
                Map<Digest, Message> additionalInputs = Maps.newHashMapWithExpectedSize(2);
                additionalInputs.put(actionKey.getDigest(), action);
                additionalInputs.put(commandHash, command);
                remoteCache.ensureInputsPresent(merkleTree, additionalInputs);
              }
              ExecuteResponse reply;
              try (SilentCloseable c = prof.profile(REMOTE_EXECUTION, "execute remotely")) {
                reply = remoteExecutor.executeRemotely(request);
              }

              FileOutErr outErr = context.getFileOutErr();
              String message = reply.getMessage();
              ActionResult actionResult = reply.getResult();
              if ((actionResult.getExitCode() != 0
                      || reply.getStatus().getCode() != Code.OK.value())
                  && !message.isEmpty()) {
                outErr.printErr(message + "\n");
              }

              try (SilentCloseable c = prof.profile(REMOTE_DOWNLOAD, "download server logs")) {
                maybeDownloadServerLogs(reply, actionKey);
              }

              try {
                return downloadAndFinalizeSpawnResult(
                    actionResult, reply.getCachedResult(), spawn, context, remoteOutputsMode);
              } catch (CacheNotFoundException e) {
                // No cache hit, so if we retry this execution, we must no longer accept
                // cached results, it must be reexecuted
                requestBuilder.setSkipCacheLookup(true);
                throw e;
              }
            });
      } catch (IOException e) {
        return execLocallyAndUploadOrFail(
            spawn, context, inputMap, actionKey, action, command, uploadLocalResults, e);
      }
    } finally {
      withMetadata.detach(previous);
    }
  }

  private SpawnResult downloadAndFinalizeSpawnResult(
      ActionResult actionResult,
      boolean cacheHit,
      Spawn spawn,
      SpawnExecutionContext context,
      RemoteOutputsMode remoteOutputsMode)
      throws ExecException, IOException, InterruptedException {
    boolean downloadOutputs =
        shouldDownloadAllSpawnOutputs(
            remoteOutputsMode,
            /* exitCode = */ actionResult.getExitCode(),
            hasFilesToDownload(spawn.getOutputFiles(), filesToDownload));
    InMemoryOutput inMemoryOutput = null;
    if (downloadOutputs) {
      try (SilentCloseable c = Profiler.instance().profile(REMOTE_DOWNLOAD, "download outputs")) {
        remoteCache.download(
            actionResult, execRoot, context.getFileOutErr(), context::lockOutputFiles);
      }
    } else {
      PathFragment inMemoryOutputPath = getInMemoryOutputPath(spawn);
      try (SilentCloseable c =
          Profiler.instance().profile(REMOTE_DOWNLOAD, "download outputs minimal")) {
        inMemoryOutput =
            remoteCache.downloadMinimal(
                actionResult,
                spawn.getOutputFiles(),
                inMemoryOutputPath,
                context.getFileOutErr(),
                execRoot,
                context.getMetadataInjector(),
                context::lockOutputFiles);
      }
    }
    return createSpawnResult(actionResult.getExitCode(), cacheHit, getName(), inMemoryOutput);
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return Spawns.mayBeExecutedRemotely(spawn);
  }

  private void maybeWriteParamFilesLocally(Spawn spawn) throws IOException {
    if (!executionOptions.shouldMaterializeParamFiles()) {
      return;
    }
    for (ActionInput actionInput : spawn.getInputFiles().toList()) {
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

  private SpawnResult execLocally(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    RemoteLocalFallbackRegistry localFallbackRegistry =
        context.getContext(RemoteLocalFallbackRegistry.class);
    checkNotNull(localFallbackRegistry, "Expected a RemoteLocalFallbackRegistry to be registered");
    AbstractSpawnStrategy remoteLocalFallbackStrategy =
        localFallbackRegistry.getRemoteLocalFallbackStrategy();
    checkNotNull(
        remoteLocalFallbackStrategy,
        "A remote local fallback strategy must be set if using remote fallback.");
    return remoteLocalFallbackStrategy.getSpawnRunner().exec(spawn, context);
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
    if (remoteOptions.remoteLocalFallback && !RemoteRetrierUtils.causedByExecTimeout(cause)) {
      return execLocallyAndUpload(
          spawn, context, inputMap, actionKey, action, command, uploadLocalResults);
    }
    return handleError(cause, context.getFileOutErr(), actionKey, context);
  }

  private SpawnResult handleError(
      IOException exception, FileOutErr outErr, ActionKey actionKey, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    if (exception.getCause() instanceof ExecutionStatusException) {
      ExecutionStatusException e = (ExecutionStatusException) exception.getCause();
      if (e.getResponse() != null) {
        ExecuteResponse resp = e.getResponse();
        maybeDownloadServerLogs(resp, actionKey);
        if (resp.hasResult()) {
          // We try to download all (partial) results even on server error, for debuggability.
          remoteCache.download(resp.getResult(), execRoot, outErr, context::lockOutputFiles);
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
    if (RemoteRetrierUtils.causedByStatus(exception, Code.UNAVAILABLE)) {
      status = Status.EXECUTION_FAILED_CATASTROPHICALLY;
    } else if (exception instanceof CacheNotFoundException) {
      status = Status.REMOTE_CACHE_FAILED;
    } else {
      status = Status.EXECUTION_FAILED;
    }

    final String errorMessage;
    if (!verboseFailures) {
      errorMessage = Utils.grpcAwareErrorMessage(exception);
    } else {
      // On --verbose_failures print the whole stack trace
      errorMessage = Throwables.getStackTraceAsString(exception);
    }

    return new SpawnResult.Builder()
        .setRunnerName(getName())
        .setStatus(status)
        .setExitCode(ExitCode.REMOTE_ERROR.getNumericExitCode())
        .setFailureMessage(errorMessage)
        .build();
  }


  static Action buildAction(Digest command, Digest inputRoot, Duration timeout, boolean cacheable) {

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

  static Command buildCommand(
      Collection<? extends ActionInput> outputs,
      List<String> arguments,
      ImmutableMap<String, String> env,
      @Nullable Platform platform,
      @Nullable String workingDirectory) {
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

    if (platform != null) {
      command.setPlatform(platform);
    }
    command.addAllArguments(arguments);
    // Sorting the environment pairs by variable name.
    TreeSet<String> variables = new TreeSet<>(env.keySet());
    for (String var : variables) {
      command.addEnvironmentVariablesBuilder().setName(var).setValue(env.get(var));
    }

    if (!Strings.isNullOrEmpty(workingDirectory)) {
      command.setWorkingDirectory(workingDirectory);
    }
    return command.build();
  }

  private Map<Path, Long> getInputCtimes(SortedMap<PathFragment, ActionInput> inputMap) {
    HashMap<Path, Long> ctimes = new HashMap<>();
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
      ActionKey actionKey,
      Action action,
      Command command,
      boolean uploadLocalResults)
      throws ExecException, IOException, InterruptedException {
    Map<Path, Long> ctimesBefore = getInputCtimes(inputMap);
    SpawnResult result = execLocally(spawn, context);
    Map<Path, Long> ctimesAfter = getInputCtimes(inputMap);
    uploadLocalResults =
        uploadLocalResults && Status.SUCCESS.equals(result.status()) && result.exitCode() == 0;
    if (!uploadLocalResults) {
      return result;
    }

    for (Map.Entry<Path, Long> e : ctimesBefore.entrySet()) {
      // Skip uploading to remote cache, because an input was modified during execution.
      if (!ctimesAfter.get(e.getKey()).equals(e.getValue())) {
        return result;
      }
    }

    Collection<Path> outputFiles = resolveActionInputs(execRoot, spawn.getOutputFiles());
    try (SilentCloseable c = Profiler.instance().profile(UPLOAD_TIME, "upload outputs")) {
      remoteCache.upload(
          actionKey, action, command, execRoot, outputFiles, context.getFileOutErr());
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
    return actionInputs.stream()
        .map((inp) -> execRoot.getRelative(inp.getExecPath()))
        .collect(ImmutableList.toImmutableList());
  }

  private static RemoteRetrier createExecuteRetrier(
      RemoteOptions options, ListeningScheduledExecutorService retryService) {
    return new RemoteRetrier(
        options.remoteMaxRetryAttempts > 0
            ? () -> new Retrier.ZeroBackoff(options.remoteMaxRetryAttempts)
            : () -> Retrier.RETRIES_DISABLED,
        RemoteSpawnRunner::retriableExecErrors,
        retryService,
        Retrier.ALLOW_ALL_CALLS);
  }
}
