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
import build.bazel.remote.execution.v2.ExecuteOperationMetadata;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutedActionMetadata;
import build.bazel.remote.execution.v2.ExecutionStage.Value;
import build.bazel.remote.execution.v2.LogFile;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
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
import com.google.devtools.build.lib.actions.SpawnMetrics;
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
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.common.RemoteExecutionClient;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.longrunning.Operation;
import com.google.protobuf.Message;
import com.google.protobuf.Timestamp;
import com.google.protobuf.util.Durations;
import com.google.protobuf.util.Timestamps;
import io.grpc.Status.Code;
import java.io.IOException;
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
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** A client for the remote execution service. */
@ThreadSafe
public class RemoteSpawnRunner implements SpawnRunner {

  private final Path execRoot;
  private final RemoteOptions remoteOptions;
  private final ExecutionOptions executionOptions;
  private final boolean verboseFailures;

  @Nullable private final Reporter cmdlineReporter;
  private final RemoteExecutionCache remoteCache;
  private final RemoteExecutionClient remoteExecutor;
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
      RemoteExecutionClient remoteExecutor,
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

  class ExecutingStatusReporter implements OperationObserver {
    private boolean reportedExecuting = false;
    private final SpawnExecutionContext context;

    ExecutingStatusReporter(SpawnExecutionContext context) {
      this.context = context;
    }

    @Override
    public void onNext(Operation o) throws IOException {
      if (!reportedExecuting) {
        if (o.getMetadata().is(ExecuteOperationMetadata.class)) {
          ExecuteOperationMetadata metadata =
              o.getMetadata().unpack(ExecuteOperationMetadata.class);
          if (metadata.getStage() == Value.EXECUTING) {
            reportExecuting();
          }
        } else {
          // If the server didn't return metadata, we can't know the accurate execution status, so
          // assuming that the action is accepted by the server and will be executed ASAP.
          reportExecuting();
        }
      }
    }

    public void reportExecuting() {
      context.report(ProgressStatus.EXECUTING, getName());
      reportedExecuting = true;
    }

    public void reportExecutingIfNot() {
      if (!reportedExecuting) {
        reportExecuting();
      }
    }
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    Stopwatch totalTime = Stopwatch.createStarted();
    boolean spawnCacheableRemotely = Spawns.mayBeCachedRemotely(spawn);
    boolean uploadLocalResults = remoteOptions.remoteUploadLocalResults && spawnCacheableRemotely;
    boolean acceptCachedResult = remoteOptions.remoteAcceptCached && spawnCacheableRemotely;

    context.report(ProgressStatus.SCHEDULING, getName());
    RemoteOutputsMode remoteOutputsMode = remoteOptions.remoteOutputsMode;
    // The "root directory" of the action from the point of view of RBE is the parent directory of
    // the execroot locally. This is so that paths of artifacts in external repositories don't
    // start with an uplevel reference...
    SortedMap<PathFragment, ActionInput> inputMap =
        context.getInputMapping(PathFragment.create(execRoot.getBaseName()));

    // ...however, MerkleTree.build() uses its execRoot parameter to resolve artifacts based on
    // ActionInput.getExecPath(), so it needs the execroot and not its parent directory.
    final MerkleTree merkleTree =
        MerkleTree.build(inputMap, context.getMetadataProvider(), execRoot, digestUtil);
    SpawnMetrics.Builder spawnMetrics =
        SpawnMetrics.Builder.forRemoteExec()
            .setInputBytes(merkleTree.getInputBytes())
            .setInputFiles(merkleTree.getInputFiles());
    maybeWriteParamFilesLocally(spawn);

    // Get the remote platform properties.
    Platform platform = PlatformUtils.getPlatformProto(spawn, remoteOptions);

    Command command =
        buildCommand(
            spawn.getOutputFiles(),
            spawn.getArguments(),
            spawn.getEnvironment(),
            platform,
            execRoot.getBaseName());
    Digest commandHash = digestUtil.compute(command);
    Action action =
        buildAction(
            commandHash,
            merkleTree.getRootDigest(),
            platform,
            context.getTimeout(),
            spawnCacheableRemotely);

    spawnMetrics.setParseTime(totalTime.elapsed());

    Preconditions.checkArgument(
        Spawns.mayBeExecutedRemotely(spawn), "Spawn can't be executed remotely. This is a bug.");
    // Look up action cache, and reuse the action output if it is found.
    ActionKey actionKey = digestUtil.computeActionKey(action);

    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(
            buildRequestId, commandId, actionKey.getDigest().getHash(), spawn.getResourceOwner());
    RemoteActionExecutionContext remoteActionExecutionContext =
        RemoteActionExecutionContext.create(metadata);
    Profiler prof = Profiler.instance();
    try {
      // Try to lookup the action in the action cache.
      ActionResult cachedResult;
      try (SilentCloseable c = prof.profile(ProfilerTask.REMOTE_CACHE_CHECK, "check cache hit")) {
        cachedResult =
            acceptCachedResult
                ? remoteCache.downloadActionResult(
                    remoteActionExecutionContext, actionKey, /* inlineOutErr= */ false)
                : null;
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
                remoteActionExecutionContext,
                actionKey.getDigest().getHash(),
                cachedResult,
                /* cacheHit= */ true,
                spawn,
                context,
                remoteOutputsMode,
                totalTime,
                () -> remoteActionExecutionContext.getNetworkTime().getDuration(),
                spawnMetrics);
          } catch (BulkTransferException e) {
            if (!e.onlyCausedByCacheNotFoundException()) {
              throw e;
            }
            // No cache hit, so we fall through to local or remote execution.
            // We set acceptCachedResult to false in order to force the action re-execution.
            acceptCachedResult = false;
          }
        }
      }
    } catch (IOException e) {
      return execLocallyAndUploadOrFail(
          remoteActionExecutionContext,
          spawn,
          context,
          inputMap,
          actionKey,
          action,
          command,
          uploadLocalResults,
          e);
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
      requestBuilder.getExecutionPolicyBuilder().setPriority(remoteOptions.remoteExecutionPriority);
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
              Duration networkTimeStart =
                  remoteActionExecutionContext.getNetworkTime().getDuration();
              Stopwatch uploadTime = Stopwatch.createStarted();
              remoteCache.ensureInputsPresent(
                  remoteActionExecutionContext, merkleTree, additionalInputs);
              // subtract network time consumed here to ensure wall clock during upload is not
              // double
              // counted, and metrics time computation does not exceed total time
              spawnMetrics.setUploadTime(
                  uploadTime
                      .elapsed()
                      .minus(
                          remoteActionExecutionContext
                              .getNetworkTime()
                              .getDuration()
                              .minus(networkTimeStart)));
            }

            ExecutingStatusReporter reporter = new ExecutingStatusReporter(context);
            ExecuteResponse reply;
            try (SilentCloseable c = prof.profile(REMOTE_EXECUTION, "execute remotely")) {
              reply =
                  remoteExecutor.executeRemotely(remoteActionExecutionContext, request, reporter);
            }
            // In case of replies from server contains metadata, but none of them has EXECUTING
            // status.
            // It's already late at this stage, but we should at least report once.
            reporter.reportExecutingIfNot();

            FileOutErr outErr = context.getFileOutErr();
            String message = reply.getMessage();
            ActionResult actionResult = reply.getResult();
            if ((actionResult.getExitCode() != 0 || reply.getStatus().getCode() != Code.OK.value())
                && !message.isEmpty()) {
              outErr.printErr(message + "\n");
            }

            spawnMetricsAccounting(spawnMetrics, actionResult.getExecutionMetadata());

            try (SilentCloseable c = prof.profile(REMOTE_DOWNLOAD, "download server logs")) {
              maybeDownloadServerLogs(remoteActionExecutionContext, reply, actionKey);
            }

            try {
              return downloadAndFinalizeSpawnResult(
                  remoteActionExecutionContext,
                  actionKey.getDigest().getHash(),
                  actionResult,
                  reply.getCachedResult(),
                  spawn,
                  context,
                  remoteOutputsMode,
                  totalTime,
                  () -> remoteActionExecutionContext.getNetworkTime().getDuration(),
                  spawnMetrics);
            } catch (BulkTransferException e) {
              if (e.onlyCausedByCacheNotFoundException()) {
                // No cache hit, so if we retry this execution, we must no longer accept
                // cached results, it must be reexecuted
                requestBuilder.setSkipCacheLookup(true);
              }
              throw e;
            }
          });
    } catch (IOException e) {
      return execLocallyAndUploadOrFail(
          remoteActionExecutionContext,
          spawn,
          context,
          inputMap,
          actionKey,
          action,
          command,
          uploadLocalResults,
          e);
    }
  }

  /** conversion utility for protobuf Timestamp difference to java.time.Duration */
  private static Duration between(Timestamp from, Timestamp to) {
    return Duration.ofNanos(Durations.toNanos(Timestamps.between(from, to)));
  }

  @VisibleForTesting
  static void spawnMetricsAccounting(
      SpawnMetrics.Builder spawnMetrics, ExecutedActionMetadata executionMetadata) {
    // Expect that a non-empty worker indicates that all fields are populated.
    // If the bounded sides of these checkpoints are default timestamps, i.e. unset,
    // the phase durations can be extremely large. Unset pairs, or a fully unset
    // collection of timestamps, will result in zeroed durations, and no metrics
    // contributions for a phase or phases.
    if (!executionMetadata.getWorker().isEmpty()) {
      // Accumulate queueTime from any previous attempts
      Duration remoteQueueTime =
          spawnMetrics
              .build()
              .queueTime()
              .plus(
                  between(
                      executionMetadata.getQueuedTimestamp(),
                      executionMetadata.getWorkerStartTimestamp()));
      spawnMetrics.setQueueTime(remoteQueueTime);
      // setup time does not include failed attempts
      Duration setupTime =
          between(
              executionMetadata.getWorkerStartTimestamp(),
              executionMetadata.getExecutionStartTimestamp());
      spawnMetrics.setSetupTime(setupTime);
      // execution time is unspecified for failures
      Duration executionWallTime =
          between(
              executionMetadata.getExecutionStartTimestamp(),
              executionMetadata.getExecutionCompletedTimestamp());
      spawnMetrics.setExecutionWallTime(executionWallTime);
      // remoteProcessOutputs time is unspecified for failures
      Duration remoteProcessOutputsTime =
          between(
              executionMetadata.getOutputUploadStartTimestamp(),
              executionMetadata.getOutputUploadCompletedTimestamp());
      spawnMetrics.setProcessOutputsTime(remoteProcessOutputsTime);
    }
  }

  private SpawnResult downloadAndFinalizeSpawnResult(
      RemoteActionExecutionContext remoteActionExecutionContext,
      String actionId,
      ActionResult actionResult,
      boolean cacheHit,
      Spawn spawn,
      SpawnExecutionContext context,
      RemoteOutputsMode remoteOutputsMode,
      Stopwatch totalTime,
      Supplier<Duration> networkTime,
      SpawnMetrics.Builder spawnMetrics)
      throws ExecException, IOException, InterruptedException {
    boolean downloadOutputs =
        shouldDownloadAllSpawnOutputs(
            remoteOutputsMode,
            /* exitCode = */ actionResult.getExitCode(),
            hasFilesToDownload(spawn.getOutputFiles(), filesToDownload));
    InMemoryOutput inMemoryOutput = null;
    Duration networkTimeStart = networkTime.get();
    Stopwatch fetchTime = Stopwatch.createStarted();
    if (downloadOutputs) {
      try (SilentCloseable c = Profiler.instance().profile(REMOTE_DOWNLOAD, "download outputs")) {
        remoteCache.download(
            remoteActionExecutionContext,
            actionResult,
            execRoot,
            context.getFileOutErr(),
            context::lockOutputFiles);
      }
    } else {
      PathFragment inMemoryOutputPath = getInMemoryOutputPath(spawn);
      try (SilentCloseable c =
          Profiler.instance().profile(REMOTE_DOWNLOAD, "download outputs minimal")) {
        inMemoryOutput =
            remoteCache.downloadMinimal(
                remoteActionExecutionContext,
                actionId,
                actionResult,
                spawn.getOutputFiles(),
                inMemoryOutputPath,
                context.getFileOutErr(),
                execRoot,
                context.getMetadataInjector(),
                context::lockOutputFiles);
      }
    }
    fetchTime.stop();
    totalTime.stop();
    Duration networkTimeEnd = networkTime.get();
    // subtract network time consumed here to ensure wall clock during fetch is not double
    // counted, and metrics time computation does not exceed total time
    return createSpawnResult(
        actionResult.getExitCode(),
        cacheHit,
        getName(),
        inMemoryOutput,
        spawnMetrics
            .setFetchTime(fetchTime.elapsed().minus(networkTimeEnd.minus(networkTimeStart)))
            .setTotalTime(totalTime.elapsed())
            .setNetworkTime(networkTimeEnd)
            .build(),
        spawn.getMnemonic());
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return Spawns.mayBeExecutedRemotely(spawn);
  }

  @Override
  public boolean handlesCaching() {
    return true;
  }

  private void maybeWriteParamFilesLocally(Spawn spawn) throws IOException {
    if (!executionOptions.shouldMaterializeParamFiles()) {
      return;
    }
    for (ActionInput actionInput : spawn.getInputFiles().toList()) {
      if (actionInput instanceof ParamFileActionInput) {
        ParamFileActionInput paramFileActionInput = (ParamFileActionInput) actionInput;
        Path outputPath = execRoot.getRelative(paramFileActionInput.getExecPath());
        SandboxHelpers.atomicallyWriteVirtualInput(paramFileActionInput, outputPath, ".remote");
      }
    }
  }

  private void maybeDownloadServerLogs(
      RemoteActionExecutionContext context, ExecuteResponse resp, ActionKey actionKey)
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
            getFromFuture(remoteCache.downloadFile(context, logPath, e.getValue().getDigest()));
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
      RemoteActionExecutionContext remoteActionExecutionContext,
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
          remoteActionExecutionContext,
          spawn,
          context,
          inputMap,
          actionKey,
          action,
          command,
          uploadLocalResults);
    }
    return handleError(
        remoteActionExecutionContext, cause, context.getFileOutErr(), actionKey, context);
  }

  private SpawnResult handleError(
      RemoteActionExecutionContext remoteActionExecutionContext,
      IOException exception,
      FileOutErr outErr,
      ActionKey actionKey,
      SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    boolean remoteCacheFailed =
        BulkTransferException.isOnlyCausedByCacheNotFoundException(exception);
    if (exception.getCause() instanceof ExecutionStatusException) {
      ExecutionStatusException e = (ExecutionStatusException) exception.getCause();
      if (e.getResponse() != null) {
        ExecuteResponse resp = e.getResponse();
        maybeDownloadServerLogs(remoteActionExecutionContext, resp, actionKey);
        if (resp.hasResult()) {
          try {
            // We try to download all (partial) results even on server error, for debuggability.
            remoteCache.download(
                remoteActionExecutionContext,
                resp.getResult(),
                execRoot,
                outErr,
                context::lockOutputFiles);
          } catch (BulkTransferException bulkTransferEx) {
            exception.addSuppressed(bulkTransferEx);
          }
        }
      }
      if (e.isExecutionTimeout()) {
        return new SpawnResult.Builder()
            .setRunnerName(getName())
            .setStatus(Status.TIMEOUT)
            .setExitCode(SpawnResult.POSIX_TIMEOUT_EXIT_CODE)
            .setFailureDetail(
                FailureDetail.newBuilder()
                    .setMessage("remote spawn timed out")
                    .setSpawn(
                        FailureDetails.Spawn.newBuilder()
                            .setCode(FailureDetails.Spawn.Code.TIMEOUT))
                    .build())
            .build();
      }
    }
    final Status status;
    FailureDetails.Spawn.Code detailedCode;
    boolean catastrophe;
    if (RemoteRetrierUtils.causedByStatus(exception, Code.UNAVAILABLE)) {
      status = Status.EXECUTION_FAILED_CATASTROPHICALLY;
      detailedCode = FailureDetails.Spawn.Code.EXECUTION_FAILED;
      catastrophe = true;
    } else if (remoteCacheFailed) {
      status = Status.REMOTE_CACHE_FAILED;
      detailedCode = FailureDetails.Spawn.Code.REMOTE_CACHE_FAILED;
      catastrophe = false;
    } else {
      status = Status.EXECUTION_FAILED;
      detailedCode = FailureDetails.Spawn.Code.EXECUTION_FAILED;
      catastrophe = false;
    }

    String errorMessage = Utils.grpcAwareErrorMessage(exception);
    if (verboseFailures) {
      // On --verbose_failures print the whole stack trace
      errorMessage += "\n" + Throwables.getStackTraceAsString(exception);
    }

    return new SpawnResult.Builder()
        .setRunnerName(getName())
        .setStatus(status)
        .setExitCode(ExitCode.REMOTE_ERROR.getNumericExitCode())
        .setFailureMessage(errorMessage)
        .setFailureDetail(
            FailureDetail.newBuilder()
                .setMessage("remote spawn failed: " + errorMessage)
                .setSpawn(
                    FailureDetails.Spawn.newBuilder()
                        .setCode(detailedCode)
                        .setCatastrophic(catastrophe))
                .build())
        .build();
  }

  static Action buildAction(
      Digest command,
      Digest inputRoot,
      @Nullable Platform platform,
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
    if (platform != null) {
      action.setPlatform(platform);
    }
    return action.build();
  }

  static Command buildCommand(
      Collection<? extends ActionInput> outputs,
      List<String> arguments,
      ImmutableMap<String, String> env,
      @Nullable Platform platform,
      @Nullable String workingDirectoryString) {
    Command.Builder command = Command.newBuilder();
    ArrayList<String> outputFiles = new ArrayList<>();
    ArrayList<String> outputDirectories = new ArrayList<>();
    PathFragment workingDirectoryPathFragment =
        workingDirectoryString == null
            ? PathFragment.EMPTY_FRAGMENT
            : PathFragment.create(workingDirectoryString);
    for (ActionInput output : outputs) {
      String pathString =
          workingDirectoryPathFragment.getRelative(output.getExecPath()).getPathString();
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

    if (!Strings.isNullOrEmpty(workingDirectoryString)) {
      command.setWorkingDirectory(workingDirectoryString);
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
      RemoteActionExecutionContext remoteActionExecutionContext,
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
          remoteActionExecutionContext,
          actionKey,
          action,
          command,
          execRoot,
          outputFiles,
          context.getFileOutErr());
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
    return new ExecuteRetrier(
        options.remoteMaxRetryAttempts, retryService, Retrier.ALLOW_ALL_CALLS);
  }
}
