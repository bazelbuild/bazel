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
package com.google.devtools.build.lib.buildeventservice;


import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.buildeventservice.BuildEventServiceOptions.BesUploadMode;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.Aborted.AbortReason;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.transports.BinaryFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildeventstream.transports.JsonFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.TextFormatFileTransport;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.network.ConnectivityStatus;
import com.google.devtools.build.lib.network.ConnectivityStatus.Status;
import com.google.devtools.build.lib.network.ConnectivityStatusProvider;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BuildEventStreamer;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CountingArtifactGroupNamer;
import com.google.devtools.build.lib.runtime.SynchronizedOutputStream;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Module responsible for the Build Event Transport (BEP) and Build Event Service (BES)
 * functionality.
 */
public abstract class BuildEventServiceModule<BESOptionsT extends BuildEventServiceOptions>
    extends BlazeModule {

  private static final Logger logger = Logger.getLogger(BuildEventServiceModule.class.getName());
  private static final GoogleLogger googleLogger = GoogleLogger.forEnclosingClass();

  /**
   * TargetComplete BEP events scale with the value of --runs_per_tests, thus setting a very large
   * value for can result in BEP events that are too big for BES to handle.
   */
  @VisibleForTesting static final int RUNS_PER_TEST_LIMIT = 100000;

  private BuildEventProtocolOptions bepOptions;
  private AuthAndTLSOptions authTlsOptions;
  private BuildEventStreamOptions besStreamOptions;
  private boolean isRunsPerTestOverTheLimit;

  /**
   * Holds the close futures for the upload of each transport with timeouts attached to them using
   * {@link #constructCloseFuturesMapWithTimeouts(ImmutableMap)} obtained from {@link
   * BuildEventTransport#getTimeout()}.
   */
  private ImmutableMap<BuildEventTransport, ListenableFuture<Void>> closeFuturesWithTimeoutsMap =
      ImmutableMap.of();

  /**
   * Holds the half-close futures for the upload of each transport with timeouts attached to them
   * using {@link #constructCloseFuturesMapWithTimeouts(ImmutableMap)} obtained from {@link
   * BuildEventTransport#getTimeout()}.
   *
   * <p>The completion of the half-close indicates that the client has sent all of the data to the
   * server and is just waiting for acknowledgement. The client must still keep the data buffered
   * locally in case acknowledgement fails.
   */
  private ImmutableMap<BuildEventTransport, ListenableFuture<Void>>
      halfCloseFuturesWithTimeoutsMap = ImmutableMap.of();

  private BesUploadMode previousUploadMode = BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE;

  // TODO(lpino): Use Optional instead of @Nullable for the members below.
  @Nullable private OutErr outErr;
  @Nullable private ImmutableSet<BuildEventTransport> bepTransports;
  @Nullable private String buildRequestId;
  @Nullable private String invocationId;
  @Nullable private Reporter reporter;
  @Nullable private BuildEventStreamer streamer;
  @Nullable private ConnectivityStatusProvider connectivityProvider;
  private static final String CONNECTIVITY_CACHE_KEY = "BES";

  protected BESOptionsT besOptions;

  protected void reportCommandLineError(EventHandler commandLineReporter, Exception exception) {
    // Don't hide unchecked exceptions as part of the error reporting.
    Throwables.throwIfUnchecked(exception);
    commandLineReporter.handle(Event.error(exception.getMessage()));
  }

  /** Maximum duration Bazel waits for the previous invocation to finish before cancelling it. */
  protected Duration getMaxWaitForPreviousInvocation() {
    return Duration.ofSeconds(5);
  }

  /** Report errors in the command line and possibly fail the build. */
  private void reportError(
      EventHandler commandLineReporter,
      ModuleEnvironment moduleEnvironment,
      String msg,
      Exception exception,
      ExitCode exitCode) {
    // Don't hide unchecked exceptions as part of the error reporting.
    Throwables.throwIfUnchecked(exception);

    googleLogger.atSevere().withCause(exception).log(msg);
    AbruptExitException abruptException = new AbruptExitException(msg, exitCode, exception);
    reportCommandLineError(commandLineReporter, exception);
    moduleEnvironment.exit(abruptException);
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(
        optionsClass(),
        AuthAndTLSOptions.class,
        BuildEventStreamOptions.class,
        BuildEventProtocolOptions.class);
  }

  // Resets the maps tracking the state of closing/half-closing BES transports.
  private void resetPendingUploads() {
    closeFuturesWithTimeoutsMap = ImmutableMap.of();
    halfCloseFuturesWithTimeoutsMap = ImmutableMap.of();
  }

  // Cancels and interrupts any in-flight threads closing BES transports, then resets the maps
  // tracking in-flight close operations.
  private void cancelAndResetPendingUploads() {
    closeFuturesWithTimeoutsMap
        .values()
        .forEach(closeFuture -> closeFuture.cancel(/* mayInterruptIfRunning= */ true));
    resetPendingUploads();
  }

  private static boolean isTimeoutException(ExecutionException e) {
    return e.getCause() instanceof TimeoutException;
  }

  private void waitForPreviousInvocation(boolean isShutdown) {
    if (closeFuturesWithTimeoutsMap.isEmpty()) {
      return;
    }

    ConnectivityStatus status = connectivityProvider.getStatus(CONNECTIVITY_CACHE_KEY);
    if (status.status != ConnectivityStatus.Status.OK) {
      reporter.handle(
          Event.info(
              String.format(
                  "The Build Event Protocol encountered a connectivity problem: %s. Cancelling"
                      + " previous background uploads",
                  status)));
      cancelAndResetPendingUploads();
      return;
    }

    ImmutableMap<BuildEventTransport, ListenableFuture<Void>> waitingFutureMap = null;
    boolean cancelCloseFutures = true;
    switch (previousUploadMode) {
      case FULLY_ASYNC:
        waitingFutureMap =
            isShutdown ? closeFuturesWithTimeoutsMap : halfCloseFuturesWithTimeoutsMap;
        cancelCloseFutures = false;
        break;
      case WAIT_FOR_UPLOAD_COMPLETE:
      case NOWAIT_FOR_UPLOAD_COMPLETE:
        waitingFutureMap = closeFuturesWithTimeoutsMap;
        cancelCloseFutures = true;
        break;
    }

    Stopwatch stopwatch = Stopwatch.createStarted();
    try {
      // TODO(b/234994611): It would be better to report before we wait, but the current
      //  infrastructure does not support that. At least we can report it afterwards.
      Uninterruptibles.getUninterruptibly(
          Futures.allAsList(waitingFutureMap.values()),
          getMaxWaitForPreviousInvocation().toMillis(),
          TimeUnit.MILLISECONDS);
      long waitedMillis = stopwatch.elapsed().toMillis();
      if (waitedMillis > 100) {
        reporter.handle(
            Event.info(
                String.format(
                    "Waited for the background upload of the Build Event Protocol for "
                        + "%d.%03d seconds.",
                    waitedMillis / 1000, waitedMillis % 1000)));
      }
    } catch (TimeoutException exception) {
      long waitedMillis = stopwatch.elapsed().toMillis();
      String msg =
          String.format(
              "The background upload of the Build Event Protocol for the previous invocation "
                  + "failed to complete in %d.%03d seconds. "
                  + "Cancelling and starting a new invocation...",
              waitedMillis / 1000, waitedMillis % 1000);
      reporter.handle(Event.warn(msg));
      googleLogger.atWarning().withCause(exception).log(msg);
      cancelCloseFutures = true;
    } catch (ExecutionException e) {
      String msg;
      // Futures.withTimeout wraps the TimeoutException in an ExecutionException when the future
      // times out.
      if (isTimeoutException(e)) {
        msg =
            "The background upload of the Build Event Protocol for the previous invocation "
                + "failed due to a network timeout. Ignoring the failure and starting a new "
                + "invocation...";
      } else {
        msg =
            String.format(
                "The background upload of the Build Event Protocol for the previous invocation "
                    + "failed with the following exception: '%s'. "
                    + "Ignoring the failure and starting a new invocation...",
                e.getMessage());
      }
      reporter.handle(Event.warn(msg));
      googleLogger.atWarning().withCause(e).log(msg);
      cancelCloseFutures = true;
    } finally {
      if (cancelCloseFutures) {
        cancelAndResetPendingUploads();
      } else {
        resetPendingUploads();
      }
    }
  }

  @Override
  public void beforeCommand(CommandEnvironment cmdEnv) {
    this.invocationId = cmdEnv.getCommandId().toString();
    this.buildRequestId = cmdEnv.getBuildRequestId();
    this.reporter = cmdEnv.getReporter();

    this.connectivityProvider =
        Preconditions.checkNotNull(
            cmdEnv.getRuntime().getBlazeModule(ConnectivityStatusProvider.class),
            "No ConnectivityStatusProvider found in modules list");

    OptionsParsingResult parsingResult = cmdEnv.getOptions();
    this.besOptions = Preconditions.checkNotNull(parsingResult.getOptions(optionsClass()));
    this.bepOptions =
        Preconditions.checkNotNull(parsingResult.getOptions(BuildEventProtocolOptions.class));
    this.authTlsOptions =
        Preconditions.checkNotNull(parsingResult.getOptions(AuthAndTLSOptions.class));
    this.besStreamOptions =
        Preconditions.checkNotNull(parsingResult.getOptions(BuildEventStreamOptions.class));
    this.isRunsPerTestOverTheLimit =
        parsingResult.getOptions(TestOptions.class) != null
            && parsingResult.getOptions(TestOptions.class).runsPerTest.stream()
                .anyMatch(
                    (perLabelOptions) ->
                        Integer.parseInt(Iterables.getOnlyElement(perLabelOptions.getOptions()))
                            > RUNS_PER_TEST_LIMIT);

    ConnectivityStatus status = connectivityProvider.getStatus(CONNECTIVITY_CACHE_KEY);
    String buildEventUploadStrategy =
        status.status.equals(ConnectivityStatus.Status.OK)
            ? this.bepOptions.buildEventUploadStrategy
            : "local";

    CountingArtifactGroupNamer artifactGroupNamer = new CountingArtifactGroupNamer();
    ThrowingBuildEventArtifactUploaderSupplier uploaderSupplier =
        new ThrowingBuildEventArtifactUploaderSupplier(
            () ->
                cmdEnv
                    .getRuntime()
                    .getBuildEventArtifactUploaderFactoryMap()
                    .select(buildEventUploadStrategy)
                    .create(cmdEnv));

    // We need to wait for the previous invocation before we check the whitelist of commands to
    // allow completing previous runs using BES, for example:
    //   bazel build (..run with async BES..)
    //   bazel info <-- Doesn't run with BES unless we wait before checking the whitelist.
    waitForPreviousInvocation("shutdown".equals(cmdEnv.getCommandName()));

    if (!whitelistedCommands(besOptions).contains(cmdEnv.getCommandName())) {
      // Exit early if the running command isn't supported.
      return;
    }

    try {
      bepTransports = createBepTransports(cmdEnv, uploaderSupplier, artifactGroupNamer);
    } catch (IOException e) {
      cmdEnv
          .getBlazeModuleEnvironment()
          .exit(new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e));
      return;
    }
    if (bepTransports.isEmpty()) {
      // Exit early if there are no transports to stream to.
      return;
    }

    streamer =
        new BuildEventStreamer.Builder()
            .buildEventTransports(bepTransports)
            .besStreamOptions(besStreamOptions)
            .artifactGroupNamer(artifactGroupNamer)
            .build();

    cmdEnv.getEventBus().register(streamer);
    registerOutAndErrOutputStreams();

    // This event should probably be posted in a more general place (e.g. {@link BuildTool};
    // however, so far the BES module is the only module that requires extra work after the build
    // so we post it here until it's needed for other modules.
    reporter.post(new AnnounceBuildEventTransportsEvent(bepTransports));
  }

  private void registerOutAndErrOutputStreams() {
    int bufferSize = besOptions.besOuterrBufferSize;
    int chunkSize = besOptions.besOuterrChunkSize;
    final SynchronizedOutputStream out = new SynchronizedOutputStream(bufferSize, chunkSize);
    final SynchronizedOutputStream err = new SynchronizedOutputStream(bufferSize, chunkSize);

    this.outErr = OutErr.create(out, err);
    streamer.registerOutErrProvider(
        new BuildEventStreamer.OutErrProvider() {
          @Override
          public Iterable<String> getOut() {
            return out.readAndReset();
          }

          @Override
          public Iterable<String> getErr() {
            return err.readAndReset();
          }
        });
    err.registerStreamer(streamer);
    out.registerStreamer(streamer);
  }

  @Override
  public OutErr getOutputListener() {
    return outErr;
  }

  private void forceShutdownBuildEventStreamer() {
    streamer.close(AbortReason.INTERNAL);
    closeFuturesWithTimeoutsMap =
        constructCloseFuturesMapWithTimeouts(streamer.getCloseFuturesMap());
    try {
      googleLogger.atInfo().log("Closing pending build event transports");
      Uninterruptibles.getUninterruptibly(Futures.allAsList(closeFuturesWithTimeoutsMap.values()));
    } catch (ExecutionException e) {
      googleLogger.atSevere().withCause(e).log("Failed to close a build event transport");
    } finally {
      cancelAndResetPendingUploads();
    }
  }

  @Override
  public void blazeShutdownOnCrash() {
    if (streamer != null) {
      googleLogger.atWarning().log("Attempting to close BES streamer on crash");
      forceShutdownBuildEventStreamer();
    }
  }

  @Override
  public void blazeShutdown() {
    if (closeFuturesWithTimeoutsMap.isEmpty()) {
      return;
    }

    try {
      Uninterruptibles.getUninterruptibly(
          Futures.allAsList(closeFuturesWithTimeoutsMap.values()),
          getMaxWaitForPreviousInvocation().getSeconds(),
          TimeUnit.SECONDS);
    } catch (TimeoutException | ExecutionException exception) {
      googleLogger.atWarning().withCause(exception).log(
          "Encountered Exception when closing BEP transports in Blaze's shutting down sequence");
    } finally {
      cancelAndResetPendingUploads();
    }
  }

  private void waitForBuildEventTransportsToClose(
      Map<BuildEventTransport, ListenableFuture<Void>> transportFutures,
      boolean besUploadModeIsSynchronous)
      throws AbruptExitException {
    final ScheduledExecutorService executor =
        Executors.newSingleThreadScheduledExecutor(
            new ThreadFactoryBuilder().setNameFormat("bes-notify-ui-%d").build());
    ScheduledFuture<?> waitMessageFuture = null;

    try {
      // Notify the UI handler when a transport finished closing.
      transportFutures.forEach(
          (bepTransport, closeFuture) ->
              closeFuture.addListener(
                  () -> {
                    reporter.post(new BuildEventTransportClosedEvent(bepTransport));
                  },
                  executor));

      try (AutoProfiler p = AutoProfiler.logged("waiting for BES close", logger)) {
        Uninterruptibles.getUninterruptibly(Futures.allAsList(transportFutures.values()));
      }
    } catch (ExecutionException e) {
      // Futures.withTimeout wraps the TimeoutException in an ExecutionException when the future
      // times out.
      if (isTimeoutException(e)) {
        throw new AbruptExitException(
            "The Build Event Protocol upload timed out",
            ExitCode.TRANSIENT_BUILD_EVENT_SERVICE_UPLOAD_ERROR,
            e);
      }

      Throwables.throwIfInstanceOf(e.getCause(), AbruptExitException.class);
      throw new RuntimeException(
          String.format(
              "Unexpected Exception '%s' when closing BEP transports, this is a bug.",
              e.getCause().getMessage()));
    } finally {
      if (besUploadModeIsSynchronous) {
        cancelAndResetPendingUploads();
      }
      if (waitMessageFuture != null) {
        waitMessageFuture.cancel(/* mayInterruptIfRunning= */ true);
      }
      executor.shutdown();
    }
  }

  private static ImmutableMap<BuildEventTransport, ListenableFuture<Void>>
      constructCloseFuturesMapWithTimeouts(
          ImmutableMap<BuildEventTransport, ListenableFuture<Void>> bepTransportToCloseFuturesMap) {
    ImmutableMap.Builder<BuildEventTransport, ListenableFuture<Void>> builder =
        ImmutableMap.builder();

    bepTransportToCloseFuturesMap.forEach(
        (bepTransport, closeFuture) -> {
          final ListenableFuture<Void> closeFutureWithTimeout;
          if (bepTransport.getTimeout().isZero() || bepTransport.getTimeout().isNegative()) {
            closeFutureWithTimeout = closeFuture;
          } else {
            final ScheduledExecutorService timeoutExecutor =
                Executors.newSingleThreadScheduledExecutor(
                    new ThreadFactoryBuilder()
                        .setNameFormat("bes-close-" + bepTransport.name() + "-%d")
                        .build());

            // Make sure to avoid propagating the cancellation to the enclosing future since
            // we handle cancellation ourselves in this class.
            // Futures.withTimeout may cancel the enclosing future when the timeout is
            // reached.
            final ListenableFuture<Void> enclosingFuture =
                Futures.nonCancellationPropagating(closeFuture);

            closeFutureWithTimeout =
                Futures.withTimeout(
                    enclosingFuture,
                    bepTransport.getTimeout().toMillis(),
                    TimeUnit.MILLISECONDS,
                    timeoutExecutor);
            closeFutureWithTimeout.addListener(
                () -> timeoutExecutor.shutdown(), MoreExecutors.directExecutor());
          }
          builder.put(bepTransport, closeFutureWithTimeout);
        });

    return builder.build();
  }

  private void closeBepTransports() throws AbruptExitException {
    previousUploadMode = besOptions.besUploadMode;
    closeFuturesWithTimeoutsMap =
        constructCloseFuturesMapWithTimeouts(streamer.getCloseFuturesMap());
    halfCloseFuturesWithTimeoutsMap =
        constructCloseFuturesMapWithTimeouts(streamer.getHalfClosedMap());
    boolean besUploadModeIsSynchronous =
        besOptions.besUploadMode == BesUploadMode.WAIT_FOR_UPLOAD_COMPLETE;
    Map<BuildEventTransport, ListenableFuture<Void>> blockingTransportFutures = new HashMap<>();
    for (Map.Entry<BuildEventTransport, ListenableFuture<Void>> entry :
        closeFuturesWithTimeoutsMap.entrySet()) {
      BuildEventTransport bepTransport = entry.getKey();
      if (!bepTransport.mayBeSlow() || besUploadModeIsSynchronous) {
        blockingTransportFutures.put(bepTransport, entry.getValue());
      } else {
        // When running asynchronously notify the UI immediately since we won't wait for the
        // uploads to close.
        reporter.post(new BuildEventTransportClosedEvent(bepTransport));
      }
    }
    if (!blockingTransportFutures.isEmpty()) {
      waitForBuildEventTransportsToClose(blockingTransportFutures, besUploadModeIsSynchronous);
    }
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    if (streamer != null) {
      if (!streamer.isClosed()) {
        // This should not occur, but close with an internal error if a {@link BuildEventStreamer}
        // bug manifests as an unclosed streamer.
        googleLogger.atWarning().log("Attempting to close BES streamer after command");
        reporter.handle(Event.warn("BES was not properly closed"));
        forceShutdownBuildEventStreamer();
      }

      closeBepTransports();

      if (!Strings.isNullOrEmpty(besOptions.besBackend)) {
        constructAndMaybeReportInvocationIdUrl();
      } else if (!bepTransports.isEmpty()) {
        reporter.handle(Event.info("Build Event Protocol files produced successfully."));
      }
    }

    if (!besStreamOptions.keepBackendConnections) {
      clearBesClient();
    }
  }

  @Override
  public void commandComplete() {
    this.outErr = null;
    this.bepTransports = null;
    this.invocationId = null;
    this.buildRequestId = null;
    this.reporter = null;
    this.streamer = null;
  }

  private void constructAndMaybeReportInvocationIdUrl() {
    if (!getInvocationIdPrefix().isEmpty()) {
      reporter.handle(
          Event.info("Streaming build results to: " + getInvocationIdPrefix() + invocationId));
    }
  }

  private void constructAndMaybeReportBuildRequestIdUrl() {
    if (!getBuildRequestIdPrefix().isEmpty()) {
      reporter.handle(
          Event.info(
              "See "
                  + getBuildRequestIdPrefix()
                  + buildRequestId
                  + " for more information about your request."));
    }
  }

  private void logIds() {
    googleLogger.atInfo().log(
        "Streaming Build Event Protocol to '%s' with build_request_id: '%s'"
            + " and invocation_id: '%s'",
        besOptions.besBackend, buildRequestId, invocationId);
  }

  @Nullable
  private BuildEventServiceTransport createBesTransport(
      CommandEnvironment cmdEnv,
      ThrowingBuildEventArtifactUploaderSupplier uploaderSupplier,
      CountingArtifactGroupNamer artifactGroupNamer)
      throws IOException {
    if (Strings.isNullOrEmpty(besOptions.besBackend)) {
      clearBesClient();
      return null;
    }

    if (isRunsPerTestOverTheLimit) {
      String msg =
          String.format(
              "The value of --runs_per_test is bigger than %d and it will produce build events "
                  + "that are too big for the Build Event Service to handle.",
              RUNS_PER_TEST_LIMIT);
      reportError(
          reporter,
          cmdEnv.getBlazeModuleEnvironment(),
          msg,
          new OptionsParsingException(msg),
          ExitCode.COMMAND_LINE_ERROR);
      return null;
    }

    logIds();

    ConnectivityStatus status = connectivityProvider.getStatus(CONNECTIVITY_CACHE_KEY);
    if (status.status != Status.OK) {
      clearBesClient();
      String message =
          String.format(
              "Build Event Service uploads disabled due to a connectivity problem: %s", status);
      reporter.handle(Event.warn(message));
      googleLogger.atWarning().log(message);
      return null;
    }

    final BuildEventServiceClient besClient;
    try {
      besClient = getBesClient(besOptions, authTlsOptions);
    } catch (IOException | OptionsParsingException e) {
      reportError(
          reporter,
          cmdEnv.getBlazeModuleEnvironment(),
          e.getMessage(),
          e,
          ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
      return null;
    }

    BuildEventServiceProtoUtil besProtoUtil =
        new BuildEventServiceProtoUtil.Builder()
            .buildRequestId(buildRequestId)
            .invocationId(invocationId)
            .projectId(besOptions.projectId)
            .commandName(cmdEnv.getCommandName())
            .keywords(getBesKeywords(besOptions, cmdEnv.getRuntime().getStartupOptionsProvider()))
            .build();

    return new BuildEventServiceTransport.Builder()
        .localFileUploader(uploaderSupplier.get())
        .besClient(besClient)
        .besOptions(besOptions)
        .besProtoUtil(besProtoUtil)
        .artifactGroupNamer(artifactGroupNamer)
        .bepOptions(bepOptions)
        .clock(cmdEnv.getRuntime().getClock())
        .eventBus(cmdEnv.getEventBus())
        .build();
  }

  private ImmutableSet<BuildEventTransport> createBepTransports(
      CommandEnvironment cmdEnv,
      ThrowingBuildEventArtifactUploaderSupplier uploaderSupplier,
      CountingArtifactGroupNamer artifactGroupNamer)
      throws IOException {
    ImmutableSet.Builder<BuildEventTransport> bepTransportsBuilder = new ImmutableSet.Builder<>();

    if (!Strings.isNullOrEmpty(besStreamOptions.buildEventTextFile)) {
      try {
        BufferedOutputStream bepTextOutputStream =
            new BufferedOutputStream(
                Files.newOutputStream(Paths.get(besStreamOptions.buildEventTextFile)));

        BuildEventArtifactUploader localFileUploader =
            besStreamOptions.buildEventTextFilePathConversion
                ? uploaderSupplier.get()
                : new LocalFilesArtifactUploader();
        bepTransportsBuilder.add(
            new TextFormatFileTransport(
                bepTextOutputStream,
                bepOptions,
                localFileUploader,
                artifactGroupNamer));
      } catch (IOException exception) {
        // TODO(b/125216340): Consider making this a warning instead of an error once the
        //  associated bug has been resolved.
        reportError(
            reporter,
            cmdEnv.getBlazeModuleEnvironment(),
            "Unable to write to '"
                + besStreamOptions.buildEventTextFile
                + "'. Omitting --build_event_text_file.",
            exception,
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
      }
    }

    if (!Strings.isNullOrEmpty(besStreamOptions.buildEventBinaryFile)) {
      try {
        BufferedOutputStream bepBinaryOutputStream =
            new BufferedOutputStream(
                Files.newOutputStream(Paths.get(besStreamOptions.buildEventBinaryFile)));

        BuildEventArtifactUploader localFileUploader =
            besStreamOptions.buildEventBinaryFilePathConversion
                ? uploaderSupplier.get()
                : new LocalFilesArtifactUploader();
        bepTransportsBuilder.add(
            new BinaryFormatFileTransport(
                bepBinaryOutputStream,
                bepOptions,
                localFileUploader,
                artifactGroupNamer));
      } catch (IOException exception) {
        // TODO(b/125216340): Consider making this a warning instead of an error once the
        //  associated bug has been resolved.
        reportError(
            reporter,
            cmdEnv.getBlazeModuleEnvironment(),
            "Unable to write to '"
                + besStreamOptions.buildEventBinaryFile
                + "'. Omitting --build_event_binary_file.",
            exception,
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
      }
    }

    if (!Strings.isNullOrEmpty(besStreamOptions.buildEventJsonFile)) {
      try {
        BufferedOutputStream bepJsonOutputStream =
            new BufferedOutputStream(
                Files.newOutputStream(Paths.get(besStreamOptions.buildEventJsonFile)));
        BuildEventArtifactUploader localFileUploader =
            besStreamOptions.buildEventJsonFilePathConversion
                ? uploaderSupplier.get()
                : new LocalFilesArtifactUploader();
        bepTransportsBuilder.add(
            new JsonFormatFileTransport(
                bepJsonOutputStream,
                bepOptions,
                localFileUploader,
                artifactGroupNamer));
      } catch (IOException exception) {
        // TODO(b/125216340): Consider making this a warning instead of an error once the
        //  associated bug has been resolved.
        reportError(
            reporter,
            cmdEnv.getBlazeModuleEnvironment(),
            "Unable to write to '"
                + besStreamOptions.buildEventJsonFile
                + "'. Omitting --build_event_json_file.",
            exception,
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
      }
    }

    BuildEventServiceTransport besTransport =
        createBesTransport(cmdEnv, uploaderSupplier, artifactGroupNamer);
    if (besTransport != null) {
      constructAndMaybeReportInvocationIdUrl();
      constructAndMaybeReportBuildRequestIdUrl();
      bepTransportsBuilder.add(besTransport);
    }

    return bepTransportsBuilder.build();
  }

  protected abstract Class<BESOptionsT> optionsClass();

  protected abstract BuildEventServiceClient getBesClient(
      BESOptionsT besOptions, AuthAndTLSOptions authAndTLSOptions)
      throws IOException, OptionsParsingException;

  protected abstract void clearBesClient();

  protected abstract Set<String> whitelistedCommands(BESOptionsT besOptions);

  protected Set<String> getBesKeywords(
      BESOptionsT besOptions, @Nullable OptionsParsingResult startupOptionsProvider) {
    return besOptions.besKeywords.stream()
        .map(keyword -> "user_keyword=" + keyword)
        .collect(ImmutableSet.toImmutableSet());
  }

  /** A prefix used when printing the invocation ID in the command line */
  protected abstract String getInvocationIdPrefix();

  /** A prefix used when printing the build request ID in the command line */
  protected abstract String getBuildRequestIdPrefix();

  // TODO(b/115961387): This method shouldn't exist. It only does because some tests are relying on
  //  the transport creation logic of this module directly.
  @VisibleForTesting
  ImmutableSet<BuildEventTransport> getBepTransports() {
    return bepTransports;
  }

  private static class ThrowingBuildEventArtifactUploaderSupplier {
    private final Callable<BuildEventArtifactUploader> callable;
    @Nullable private BuildEventArtifactUploader memoizedValue;
    @Nullable private IOException exception;

    ThrowingBuildEventArtifactUploaderSupplier(Callable<BuildEventArtifactUploader> callable) {
      this.callable = callable;
    }

    BuildEventArtifactUploader get() throws IOException {
      if (memoizedValue == null && exception == null) {
        try {
          memoizedValue = callable.call();
        } catch (IOException e) {
          exception = e;
        } catch (Exception e) {
          Throwables.throwIfUnchecked(e);
          throw new IllegalStateException(e);
        }
      }
      if (memoizedValue != null) {
        return memoizedValue;
      }
      throw exception;
    }
  }
}
