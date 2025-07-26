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
import static com.google.devtools.build.lib.profiler.ProfilerTask.FETCH;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_DOWNLOAD;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_EXECUTION;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_PROCESS_TIME;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_QUEUE;
import static com.google.devtools.build.lib.profiler.ProfilerTask.REMOTE_SETUP;
import static com.google.devtools.build.lib.profiler.ProfilerTask.UPLOAD_TIME;
import static com.google.devtools.build.lib.remote.util.Utils.createExecExceptionForCredentialHelperException;
import static com.google.devtools.build.lib.remote.util.Utils.createExecExceptionFromRemoteExecutionCapabilitiesException;
import static com.google.devtools.build.lib.remote.util.Utils.createSpawnResult;
import static java.lang.Math.max;

import build.bazel.remote.execution.v2.ExecuteOperationMetadata;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ExecutedActionMetadata;
import build.bazel.remote.execution.v2.ExecutionStage.Value;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialHelperException;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.BlazeClock.MillisSinceEpochToNanosConverter;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.RemoteLocalFallbackRegistry;
import com.google.devtools.build.lib.exec.SpawnCheckingCacheEvent;
import com.google.devtools.build.lib.exec.SpawnExecutingEvent;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnSchedulingEvent;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.RemoteExecutionService.RemoteActionResult;
import com.google.devtools.build.lib.remote.RemoteExecutionService.ServerLogs;
import com.google.devtools.build.lib.remote.circuitbreaker.CircuitBreakerFactory;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.OperationObserver;
import com.google.devtools.build.lib.remote.common.RemoteExecutionCapabilitiesException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.remote.util.Utils.InMemoryOutput;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.longrunning.Operation;
import com.google.protobuf.Timestamp;
import com.google.protobuf.util.Durations;
import com.google.protobuf.util.Timestamps;
import io.grpc.Status.Code;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** A client for the remote execution service. */
@ThreadSafe
public class RemoteSpawnRunner implements SpawnRunner {

  private static final SpawnCheckingCacheEvent SPAWN_CHECKING_CACHE_EVENT =
      SpawnCheckingCacheEvent.create("remote");

  private static final SpawnSchedulingEvent SPAWN_SCHEDULING_EVENT =
      SpawnSchedulingEvent.create("remote");

  private static final SpawnExecutingEvent SPAWN_EXECUTING_EVENT =
      SpawnExecutingEvent.create("remote");

  private final RemoteOptions remoteOptions;
  private final boolean verboseFailures;
  @Nullable private final Reporter cmdlineReporter;
  private final RemoteRetrier retrier;
  private final Path logDir;
  private final RemoteExecutionService remoteExecutionService;
  private final DigestUtil digestUtil;

  // Used to ensure that a warning is reported only once.
  private final AtomicBoolean warningReported = new AtomicBoolean();

  RemoteSpawnRunner(
      RemoteOptions remoteOptions,
      boolean verboseFailures,
      @Nullable Reporter cmdlineReporter,
      ListeningScheduledExecutorService retryService,
      Path logDir,
      RemoteExecutionService remoteExecutionService,
      DigestUtil digestUtil) {
    this.remoteOptions = remoteOptions;
    this.verboseFailures = verboseFailures;
    this.cmdlineReporter = cmdlineReporter;
    this.retrier = createExecuteRetrier(remoteOptions, retryService);
    this.logDir = logDir;
    this.remoteExecutionService = remoteExecutionService;
    this.digestUtil = digestUtil;
  }

  @VisibleForTesting
  RemoteExecutionService getRemoteExecutionService() {
    return remoteExecutionService;
  }

  @Override
  public String getName() {
    return "remote";
  }

  static class ExecutingStatusReporter implements OperationObserver {
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
      context.report(SPAWN_EXECUTING_EVENT);
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
    Preconditions.checkArgument(
        remoteExecutionService.mayBeExecutedRemotely(spawn),
        "Spawn can't be executed remotely. This is a bug.");

    Stopwatch totalTime = Stopwatch.createStarted();
    boolean acceptCachedResult = remoteExecutionService.getReadCachePolicy(spawn).allowAnyCache();
    boolean uploadLocalResults = remoteExecutionService.getWriteCachePolicy(spawn).allowAnyCache();

    RemoteAction action = remoteExecutionService.buildRemoteAction(spawn, context);

    context.setDigest(digestUtil.asSpawnLogProto(action.getActionKey()));

    SpawnMetrics.Builder spawnMetrics =
        SpawnMetrics.Builder.forRemoteExec()
            .setInputBytes(action.getInputBytes())
            .setInputFiles(action.getInputFiles());

    remoteExecutionService.maybeWriteParamFilesLocally(spawn);

    spawnMetrics.setParseTime(totalTime.elapsed());

    Profiler prof = Profiler.instance();
    try {
      context.report(SPAWN_CHECKING_CACHE_EVENT);

      // Try to lookup the action in the action cache.
      RemoteActionResult cachedResult;
      try (SilentCloseable c = prof.profile(ProfilerTask.REMOTE_CACHE_CHECK, "check cache hit")) {
        cachedResult = acceptCachedResult ? remoteExecutionService.lookupCache(action) : null;
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
                action,
                cachedResult,
                /* cacheHit= */ true,
                cachedResult.cacheName(),
                spawn,
                totalTime,
                () -> action.getNetworkTime().getDuration(),
                spawnMetrics);
          } catch (BulkTransferException e) {
            if (!e.allCausedByCacheNotFoundException()) {
              throw e;
            }
            // No cache hit, so we fall through to local or remote execution.
            // We set acceptCachedResult to false in order to force the action re-execution.
            acceptCachedResult = false;
          }
        }
      }
    } catch (CredentialHelperException e) {
      throw createExecExceptionForCredentialHelperException(e);
    } catch (IOException e) {
      return execLocallyAndUploadOrFail(
          action, spawn, context, uploadLocalResults, e, FailureReason.DOWNLOAD);
    }

    if (remoteOptions.remoteRequireCached) {
      return new SpawnResult.Builder()
          .setStatus(SpawnResult.Status.EXECUTION_DENIED)
          .setExitCode(1)
          .setFailureMessage(
              "Action must be cached due to --experimental_remote_require_cached but it is not")
          .setFailureDetail(
              FailureDetail.newBuilder()
                  .setSpawn(
                      FailureDetails.Spawn.newBuilder()
                          .setCode(FailureDetails.Spawn.Code.EXECUTION_DENIED))
                  .build())
          .setRunnerName("remote")
          .build();
    }

    AtomicBoolean useCachedResult = new AtomicBoolean(acceptCachedResult);
    AtomicBoolean forceUploadInput = new AtomicBoolean(false);
    AtomicBoolean ioExceptionCausedByUpload = new AtomicBoolean(false);
    try {
      return retrier.execute(
          () -> {
            ioExceptionCausedByUpload.set(false);
            // Upload the command and all the inputs into the remote cache.
            try (SilentCloseable c = prof.profile(UPLOAD_TIME, "upload missing inputs")) {
              Duration networkTimeStart = action.getNetworkTime().getDuration();
              Stopwatch uploadTime = Stopwatch.createStarted();
              // Upon retry, we force upload inputs
              try {
                remoteExecutionService.uploadInputsIfNotPresent(
                    action, forceUploadInput.getAndSet(true));
              } catch (IOException e) {
                ioExceptionCausedByUpload.set(true);
                throw e;
              }

              // subtract network time consumed here to ensure wall clock during upload is not
              // double
              // counted, and metrics time computation does not exceed total time
              spawnMetrics.setUploadTime(
                  uploadTime
                      .elapsed()
                      .minus(action.getNetworkTime().getDuration().minus(networkTimeStart)));
            }

            context.report(SPAWN_SCHEDULING_EVENT);

            ExecutingStatusReporter reporter = new ExecutingStatusReporter(context);
            long clampTimeNanos; // See comment in logProfileTask.
            RemoteActionResult result;
            try (SilentCloseable c = prof.profile(REMOTE_EXECUTION, "execute remotely")) {
              clampTimeNanos = Profiler.nanoTimeMaybe();
              result =
                  remoteExecutionService.executeRemotely(action, useCachedResult.get(), reporter);
            }
            // In case of replies from server contains metadata, but none of them has EXECUTING
            // status.
            // It's already late at this stage, but we should at least report once.
            reporter.reportExecutingIfNot();

            maybePrintExecutionMessages(context, result.getMessage(), result.success());

            profileAccounting(clampTimeNanos, result.getExecutionMetadata());
            spawnMetricsAccounting(spawnMetrics, result.getExecutionMetadata());

            try (SilentCloseable c = prof.profile(REMOTE_DOWNLOAD, "download server logs")) {
              maybeDownloadServerLogs(action, result.getResponse());
            }

            try {
              return downloadAndFinalizeSpawnResult(
                  action,
                  result,
                  result.cacheHit(),
                  getName(),
                  spawn,
                  totalTime,
                  () -> action.getNetworkTime().getDuration(),
                  spawnMetrics);
            } catch (BulkTransferException e) {
              if (e.allCausedByCacheNotFoundException()) {
                // No cache hit, so if we retry this execution, we must no longer accept
                // cached results, it must be reexecuted
                useCachedResult.set(false);
              }
              throw e;
            }
          });
    } catch (CredentialHelperException e) {
      throw createExecExceptionForCredentialHelperException(e);
    } catch (IOException e) {
      return execLocallyAndUploadOrFail(
          action,
          spawn,
          context,
          uploadLocalResults,
          e,
          ioExceptionCausedByUpload.get() ? FailureReason.UPLOAD : FailureReason.DOWNLOAD);
    }
  }

  private static void profileAccounting(
      long clampTimeNanos, ExecutedActionMetadata executedActionMetadata) {
    MillisSinceEpochToNanosConverter converter =
        BlazeClock.createMillisSinceEpochToNanosConverter();

    logProfileTask(
        converter,
        executedActionMetadata.getQueuedTimestamp(),
        executedActionMetadata.getWorkerStartTimestamp(),
        clampTimeNanos,
        REMOTE_QUEUE,
        "queue");
    logProfileTask(
        converter,
        executedActionMetadata.getWorkerStartTimestamp(),
        executedActionMetadata.getInputFetchStartTimestamp(),
        clampTimeNanos,
        REMOTE_SETUP,
        "pre-fetch");
    logProfileTask(
        converter,
        executedActionMetadata.getInputFetchStartTimestamp(),
        executedActionMetadata.getInputFetchCompletedTimestamp(),
        clampTimeNanos,
        FETCH,
        "fetch");
    logProfileTask(
        converter,
        executedActionMetadata.getInputFetchCompletedTimestamp(),
        executedActionMetadata.getExecutionStartTimestamp(),
        clampTimeNanos,
        REMOTE_SETUP,
        "pre-execute");
    logProfileTask(
        converter,
        executedActionMetadata.getExecutionStartTimestamp(),
        executedActionMetadata.getExecutionCompletedTimestamp(),
        clampTimeNanos,
        REMOTE_PROCESS_TIME,
        "execute");
    logProfileTask(
        converter,
        executedActionMetadata.getExecutionCompletedTimestamp(),
        executedActionMetadata.getOutputUploadStartTimestamp(),
        clampTimeNanos,
        REMOTE_SETUP,
        "pre-upload");
    logProfileTask(
        converter,
        executedActionMetadata.getOutputUploadStartTimestamp(),
        executedActionMetadata.getOutputUploadCompletedTimestamp(),
        clampTimeNanos,
        UPLOAD_TIME,
        "upload");
  }

  private static void logProfileTask(
      MillisSinceEpochToNanosConverter converter,
      Timestamp start,
      Timestamp end,
      long clampTimeNanos,
      ProfilerTask type,
      String description) {
    // If the remote execution request is deduped against an earlier request for the same action,
    // the start and end times may predate the start of the execution on our side. To avoid
    // confusion, clamp them so that they nest inside the parent profile span.
    long startTimeNanos = converter.toNanos(Timestamps.toMillis(start));
    long endTimeNanos = converter.toNanos(Timestamps.toMillis(end));
    if (endTimeNanos <= clampTimeNanos) {
      // Span lies entirely outside the parent.
      return;
    }
    startTimeNanos = max(startTimeNanos, clampTimeNanos);
    Profiler.instance().logSimpleTask(startTimeNanos, endTimeNanos, type, description);
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
          between(
              executionMetadata.getQueuedTimestamp(), executionMetadata.getWorkerStartTimestamp());
      spawnMetrics.addQueueTime(remoteQueueTime);
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
      RemoteAction action,
      RemoteActionResult result,
      boolean cacheHit,
      String cacheName,
      Spawn spawn,
      Stopwatch totalTime,
      Supplier<Duration> networkTime,
      SpawnMetrics.Builder spawnMetrics)
      throws ExecException, IOException, InterruptedException {
    Duration networkTimeStart = networkTime.get();
    Stopwatch fetchTime = Stopwatch.createStarted();

    InMemoryOutput inMemoryOutput;
    try (SilentCloseable c = Profiler.instance().profile(REMOTE_DOWNLOAD, "download outputs")) {
      inMemoryOutput = remoteExecutionService.downloadOutputs(action, result);
    }

    fetchTime.stop();
    totalTime.stop();
    Duration networkTimeEnd = networkTime.get();
    // subtract network time consumed here to ensure wall clock during fetch is not double
    // counted, and metrics time computation does not exceed total time
    return createSpawnResult(
        digestUtil,
        action.getActionKey(),
        result.getExitCode(),
        cacheHit,
        cacheName,
        inMemoryOutput,
        result.getExecutionMetadata().getExecutionStartTimestamp(),
        result.getExecutionMetadata().getExecutionCompletedTimestamp(),
        spawnMetrics
            .setFetchTimeInMs(
                (int) fetchTime.elapsed().minus(networkTimeEnd.minus(networkTimeStart)).toMillis())
            .setTotalTimeInMs((int) totalTime.elapsed().toMillis())
            .setNetworkTimeInMs((int) networkTimeEnd.toMillis())
            .build(),
        spawn.getMnemonic());
  }

  @Override
  public boolean canExec(Spawn spawn) {
    return remoteExecutionService.mayBeExecutedRemotely(spawn);
  }

  @Override
  public boolean handlesCaching() {
    return true;
  }

  private void maybePrintExecutionMessages(
      SpawnExecutionContext context, String message, boolean success) {
    FileOutErr outErr = context.getFileOutErr();
    boolean printMessage =
        remoteOptions.remotePrintExecutionMessages.shouldPrintMessages(success)
            && !message.isEmpty();
    if (printMessage) {
      outErr.printErr("Remote server execution message: " + message + "\n");
    }
  }

  private void maybeDownloadServerLogs(RemoteAction action, ExecuteResponse resp)
      throws InterruptedException {
    try {
      ServerLogs serverLogs = remoteExecutionService.maybeDownloadServerLogs(action, resp, logDir);
      if (serverLogs.logCount > 0 && verboseFailures) {
        report(
            Event.info(
                "Remote server log of failing action:\n   "
                    + (serverLogs.logCount > 1 ? serverLogs.directory : serverLogs.lastLogPath)));
      }
    } catch (IOException e) {
      reportOnce(Event.warn("Failed downloading server logs from the remote cache."));
    }
  }

  private static SpawnResult execLocally(Spawn spawn, SpawnExecutionContext context)
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

  private enum FailureReason {
    // The failure occurred during the upload of the action's input for remote execution.
    UPLOAD,
    // The failure occurred during the download of the action's output from the remote cache.
    DOWNLOAD,
  }

  private SpawnResult execLocallyAndUploadOrFail(
      RemoteAction action,
      Spawn spawn,
      SpawnExecutionContext context,
      boolean uploadLocalResults,
      IOException cause,
      FailureReason reason)
      throws ExecException, InterruptedException, IOException {
    // Regardless of cause, if we are interrupted, we should stop without displaying a user-visible
    // failure/stack trace.
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException();
    }
    // If the failure is caused by eviction of inputs to the current action that are only available
    // remotely, try to regenerate the lost inputs. This doesn't make sense for outputs of the
    // current action.
    if (reason == FailureReason.UPLOAD && cause instanceof BulkTransferException e) {
      e.getLostArtifacts(context.getInputMetadataProvider()::getInput).throwIfNotEmpty();
    }
    if (remoteOptions.remoteLocalFallback && !RemoteRetrierUtils.causedByExecTimeout(cause)) {
      return execLocallyAndUpload(action, spawn, context, uploadLocalResults);
    }
    return handleError(action, cause, context);
  }

  private SpawnResult handleError(
      RemoteAction action, IOException exception, SpawnExecutionContext context)
      throws ExecException, InterruptedException, IOException {
    if (exception instanceof RemoteExecutionCapabilitiesException e) {
      throw createExecExceptionFromRemoteExecutionCapabilitiesException(e);
    }
    if (exception.getCause() instanceof ExecutionStatusException e) {
      RemoteActionResult result = null;
      if (e.getResponse() != null) {
        ExecuteResponse resp = e.getResponse();
        maybeDownloadServerLogs(action, resp);
        if (resp.hasResult()) {
          result = RemoteActionResult.createFromResponse(resp);
          try {
            remoteExecutionService.downloadOutputs(action, result);
          } catch (BulkTransferException bulkTransferEx) {
            exception.addSuppressed(bulkTransferEx);
          }
        }
      }
      if (e.isExecutionTimeout()) {
        maybePrintExecutionMessages(context, e.getResponse().getMessage(), /* success= */ false);
        SpawnResult.Builder resultBuilder =
            new SpawnResult.Builder()
                .setRunnerName(getName())
                .setStatus(Status.TIMEOUT)
                .setExitCode(SpawnResult.POSIX_TIMEOUT_EXIT_CODE)
                .setFailureDetail(
                    FailureDetail.newBuilder()
                        .setMessage("remote spawn timed out")
                        .setSpawn(
                            FailureDetails.Spawn.newBuilder()
                                .setCode(FailureDetails.Spawn.Code.TIMEOUT))
                        .build());
        if (result != null) {
          resultBuilder
              .setWallTimeInMs(
                  (int)
                      Duration.between(
                              Utils.timestampToInstant(
                                  result.getExecutionMetadata().getExecutionStartTimestamp()),
                              Utils.timestampToInstant(
                                  result.getExecutionMetadata().getExecutionCompletedTimestamp()))
                          .toMillis())
              .setStartTime(
                  Utils.timestampToInstant(
                      result.getExecutionMetadata().getExecutionStartTimestamp()));
        }
        return resultBuilder.build();
      }
    }
    final Status status;
    FailureDetails.Spawn.Code detailedCode;
    boolean catastrophe;
    if (RemoteRetrierUtils.causedByStatus(exception, Code.UNAVAILABLE)) {
      status = Status.EXECUTION_FAILED_CATASTROPHICALLY;
      detailedCode = FailureDetails.Spawn.Code.EXECUTION_FAILED;
      catastrophe = true;
    } else if (BulkTransferException.allCausedByCacheNotFoundException(exception)) {
      // At this point, cache evictions that affect uploaded inputs have already been handled.
      // Cache evictions that affect the outputs of the current actions have also been retried with
      // a request that disallows reusing cached results. This means that there is no point in
      // retrying the entire build.
      status = Status.REMOTE_CACHE_FAILED;
      detailedCode = FailureDetails.Spawn.Code.REMOTE_CACHE_FAILED;
      catastrophe = false;
    } else {
      status = Status.EXECUTION_FAILED;
      detailedCode = FailureDetails.Spawn.Code.EXECUTION_FAILED;
      catastrophe = false;
    }

    String errorMessage = Utils.grpcAwareErrorMessage(exception, verboseFailures);

    if (exception.getCause() instanceof ExecutionStatusException e) {
      if (e.getResponse() != null) {
        if (!e.getResponse().getMessage().isEmpty()) {
          errorMessage += "\n" + e.getResponse().getMessage();
        }
      }
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

  @VisibleForTesting
  SpawnResult execLocallyAndUpload(
      RemoteAction action, Spawn spawn, SpawnExecutionContext context, boolean uploadLocalResults)
      throws ExecException, IOException, InterruptedException {
    SpawnResult result = execLocally(spawn, context);
    if (uploadLocalResults && Status.SUCCESS.equals(result.status()) && result.exitCode() == 0) {
      remoteExecutionService.uploadOutputs(
          action, result, () -> {}, remoteOptions.guardAgainstConcurrentChanges);
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

  private static RemoteRetrier createExecuteRetrier(
      RemoteOptions options, ListeningScheduledExecutorService retryService) {
    return new ExecuteRetrier(
        options.remoteMaxRetryAttempts,
        retryService,
        CircuitBreakerFactory.createCircuitBreaker(options));
  }
}
