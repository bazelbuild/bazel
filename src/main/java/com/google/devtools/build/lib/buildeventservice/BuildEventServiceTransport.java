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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.listeningDecorator;
import static com.google.devtools.build.lib.events.EventKind.ERROR;
import static com.google.devtools.build.lib.events.EventKind.INFO;
import static com.google.devtools.build.lib.events.EventKind.WARNING;
import static com.google.devtools.build.v1.BuildEvent.EventCase.COMPONENT_STREAM_FINISHED;
import static com.google.devtools.build.v1.BuildStatus.Result.COMMAND_FAILED;
import static com.google.devtools.build.v1.BuildStatus.Result.COMMAND_SUCCEEDED;
import static com.google.devtools.build.v1.BuildStatus.Result.UNKNOWN_STATUS;
import static java.lang.String.format;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent.PayloadCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildFinished;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.runtime.BlazeModule.ModuleEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import com.google.devtools.build.v1.BuildStatus.Result;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishBuildToolEventStreamResponse;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.protobuf.Any;
import io.grpc.Status;
import java.time.Duration;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/** A {@link BuildEventTransport} that streams {@link BuildEvent}s to BuildEventService. */
public class BuildEventServiceTransport implements BuildEventTransport {

  static final String UPLOAD_FAILED_MESSAGE = "Build Event Protocol upload failed: %s";
  static final String UPLOAD_SUCCEEDED_MESSAGE =
      "Build Event Protocol upload finished successfully.";

  private static final Logger logger = Logger.getLogger(BuildEventServiceTransport.class.getName());

  /** Max wait time until for the Streaming RPC to finish after all events were sent. */
  private static final Duration PUBLISH_EVENT_STREAM_FINISHED_TIMEOUT = Duration.ofSeconds(30);
  /** Max wait time between isStreamActive checks of the PublishBuildToolEventStream RPC. */
  private static final int STREAMING_RPC_POLL_IN_SECS = 1;

  private final ListeningExecutorService uploaderExecutorService;
  private final Duration uploadTimeout;
  private final boolean publishLifecycleEvents;
  private final boolean bestEffortUpload;
  private final BuildEventServiceClient besClient;
  private final BuildEventServiceProtoUtil besProtoUtil;
  private final ModuleEnvironment moduleEnvironment;
  private final EventHandler commandLineReporter;
  private final PathConverter pathConverter;
  private final Sleeper sleeper;
  /** Contains all pendingAck events that might be retried in case of failures. */
  private ConcurrentLinkedDeque<PublishBuildToolEventStreamRequest> pendingAck;
  /** Contains all events should be sent ordered by sequence number. */
  private final BlockingDeque<PublishBuildToolEventStreamRequest> pendingSend;
  /** Holds the result status of the BuildEventStreamProtos BuildFinished event. */
  private Result invocationResult;
  /** Used to block until all events have been uploaded. */
  private ListenableFuture<?> uploadComplete;
  /** Used to ensure that the close logic is only invoked once. */
  private SettableFuture<Void> shutdownFuture;
  /**
   * If the call before the current call threw an exception, this field points to it. If the
   * previous call was successful, this field is null. This is useful for error reporting, when an
   * upload times out due to having had to retry several times.
   */
  private volatile Exception lastKnownError;
  /** Returns true if we already reported a warning or error to UI. */
  private volatile boolean errorsReported;
  /**
   * Returns the number of ACKs received since the last time {@link #publishEventStream()} was
   * retried due to a failure.
   */
  private volatile int acksReceivedSinceLastRetry;

  public BuildEventServiceTransport(
      BuildEventServiceClient besClient,
      Duration uploadTimeout,
      boolean bestEffortUpload,
      boolean publishLifecycleEvents,
      String buildRequestId,
      String invocationId,
      String command,
      ModuleEnvironment moduleEnvironment,
      Clock clock,
      PathConverter pathConverter,
      EventHandler commandLineReporter,
      @Nullable String projectId,
      List<String> keywords) {
    this(besClient, uploadTimeout, bestEffortUpload, publishLifecycleEvents, buildRequestId,
        invocationId, command, moduleEnvironment, clock, pathConverter, commandLineReporter,
        projectId, keywords, new JavaSleeper());
  }

  @VisibleForTesting
  public BuildEventServiceTransport(
      BuildEventServiceClient besClient,
      Duration uploadTimeout,
      boolean bestEffortUpload,
      boolean publishLifecycleEvents,
      String buildRequestId,
      String invocationId,
      String command,
      ModuleEnvironment moduleEnvironment,
      Clock clock,
      PathConverter pathConverter,
      EventHandler commandLineReporter,
      @Nullable String projectId,
      List<String> keywords,
      Sleeper sleeper) {
    this.besClient = besClient;
    this.besProtoUtil = new BuildEventServiceProtoUtil(
        buildRequestId, invocationId, projectId, command, clock, keywords);
    this.publishLifecycleEvents = publishLifecycleEvents;
    this.moduleEnvironment = moduleEnvironment;
    this.commandLineReporter = commandLineReporter;
    this.pendingAck = new ConcurrentLinkedDeque<>();
    this.pendingSend = new LinkedBlockingDeque<>();
    // Setting the thread count to 2 instead of 1 is a hack, but necessary as publishEventStream
    // blocks one thread permanently and thus we can't do any other work on the executor. A proper
    // fix would be to remove the spinning loop from publishEventStream and instead implement the
    // loop by publishEventStream re-submitting itself to the executor.
    // TODO(buchgr): Fix it.
    this.uploaderExecutorService = listeningDecorator(Executors.newFixedThreadPool(2));
    this.pathConverter = pathConverter;
    this.invocationResult = UNKNOWN_STATUS;
    this.uploadTimeout = uploadTimeout;
    this.bestEffortUpload = bestEffortUpload;
    this.sleeper = sleeper;
  }

  public boolean isStreaming() {
    return besClient.isStreamActive();
  }

  @Override
  public ListenableFuture<Void> close() {
    return close(/*now=*/false);
  }

  @Override
  @SuppressWarnings("FutureReturnValueIgnored")
  public void closeNow() {
    close(/*now=*/true);
  }

  private synchronized ListenableFuture<Void> close(boolean now) {
    if (shutdownFuture != null) {
      if (now) {
        cancelUpload();
        if (!shutdownFuture.isDone()) {
          shutdownFuture.set(null);
        }
      }
      return shutdownFuture;
    }

    logger.log(Level.INFO, "Closing the build event service transport.");

    // The future is completed once the close succeeded or failed.
    shutdownFuture = SettableFuture.create();

    if (now) {
      cancelUpload();
      shutdownFuture.set(null);
      return shutdownFuture;
    }

    uploaderExecutorService.execute(
        () -> {
          try {
            sendOrderedBuildEvent(besProtoUtil.streamFinished());

            if (errorsReported) {
              // If we encountered errors before and have already reported them, then we should
              // not report them a second time.
              return;
            }

            if (bestEffortUpload) {
              // TODO(buchgr): The code structure currently doesn't allow to enforce a timeout for
              // best effort upload.
              if (!uploadComplete.isDone()) {
                report(INFO, "Asynchronous Build Event Protocol upload.");
              } else {
                Throwable uploadError = fromFuture(uploadComplete);

                if (uploadError != null) {
                  report(WARNING, UPLOAD_FAILED_MESSAGE, uploadError.getMessage());
                } else {
                  report(INFO, UPLOAD_SUCCEEDED_MESSAGE);
                }
              }
            } else {
              report(INFO, "Waiting for Build Event Protocol upload to finish.");
              try {
                if (Duration.ZERO.equals(uploadTimeout)) {
                  uploadComplete.get();
                } else {
                  uploadComplete.get(uploadTimeout.toMillis(), MILLISECONDS);
                }
                report(INFO, UPLOAD_SUCCEEDED_MESSAGE);
              } catch (Exception e) {
                uploadComplete.cancel(true);
                reportErrorAndFailBuild(e);
              }
            }
          } finally {
            shutdownFuture.set(null);
            uploaderExecutorService.shutdown();
          }
        });

    return shutdownFuture;
  }

  private void cancelUpload() {
    if (!uploaderExecutorService.isShutdown()) {
      logger.log(Level.INFO, "Forcefully closing the build event service transport.");
      // This will interrupt the thread doing the BES upload.
      if (uploadComplete != null) {
        uploadComplete.cancel(true);
      }
      uploaderExecutorService.shutdownNow();
      try {
        uploaderExecutorService.awaitTermination(100, TimeUnit.MILLISECONDS);
      } catch (InterruptedException e) {
        // Ignore this exception. We are shutting down independently no matter what the BES
        // upload does.
      }
    }
  }

  @Override
  public String name() {
    // TODO(buchgr): Also display the hostname / IP.
    return "Build Event Service";
  }

  @Override
  public synchronized void sendBuildEvent(BuildEvent event, final ArtifactGroupNamer namer) {
    BuildEventStreamProtos.BuildEvent eventProto = event.asStreamProto(
        new BuildEventConverters() {
          @Override
          public PathConverter pathConverter() {
            return pathConverter;
          }
          @Override
          public ArtifactGroupNamer artifactGroupNamer() {
            return namer;
          }
        });
    if (PayloadCase.FINISHED.equals(eventProto.getPayloadCase())) {
      BuildFinished finished = eventProto.getFinished();
      if (finished.hasExitCode() && finished.getExitCode().getCode() == 0) {
        invocationResult = COMMAND_SUCCEEDED;
      } else {
        invocationResult = COMMAND_FAILED;
      }
    }

    sendOrderedBuildEvent(besProtoUtil.bazelEvent(Any.pack(eventProto)));
  }

  private String errorMessageFromException(Throwable t) {
    String message;
    if (t instanceof TimeoutException) {
      message = "Build Event Protocol upload timed out.";
      Exception lastKnownError0 = lastKnownError;
      if (lastKnownError0 != null) {
        // We may at times get a timeout exception due to an underlying error that was retried
        // several times. If such an error exists, report it.
        message += " Transport errors caused the upload to be retried.";
        message += " Last known reason for retry: ";
        message += besClient.userReadableError(lastKnownError0);
        return message;
      }
      return message;
    } else if (t instanceof ExecutionException) {
      message = format(UPLOAD_FAILED_MESSAGE,
          t.getCause() != null
              ? besClient.userReadableError(t.getCause())
              : t.getMessage());
      return message;
    } else {
      message = format(UPLOAD_FAILED_MESSAGE, besClient.userReadableError(t));
      return message;
    }
  }

  private void reportErrorAndFailBuild(Throwable t) {
    checkState(!bestEffortUpload);

    String message = errorMessageFromException(t);

    report(ERROR, message);
    moduleEnvironment.exit(new AbruptExitException(ExitCode.PUBLISH_ERROR));
  }

  private void maybeReportUploadError() {
    if (errorsReported) {
      return;
    }

    Throwable uploadError = fromFuture(uploadComplete);
    if (uploadError != null) {
      errorsReported = true;
      if (bestEffortUpload) {
        report(WARNING, UPLOAD_FAILED_MESSAGE, uploadError.getMessage());
      } else {
        reportErrorAndFailBuild(uploadError);
      }
    }
  }

  private synchronized void sendOrderedBuildEvent(
      PublishBuildToolEventStreamRequest serialisedEvent) {
    if (uploadComplete != null && uploadComplete.isDone()) {
      maybeReportUploadError();
      return;
    }

    pendingSend.add(serialisedEvent);
    if (uploadComplete == null) {
      uploadComplete = uploaderExecutorService.submit(new BuildEventServiceUpload());
    }
  }

  private synchronized Result getInvocationResult() {
    return invocationResult;
  }

  /**
   * Method responsible for sending all requests to BuildEventService.
   */
  private class BuildEventServiceUpload implements Callable<Void> {
    @Override
    public Void call() throws Exception {
      try {
        publishBuildEnqueuedEvent();
        publishInvocationStartedEvent();
        try {
          publishEventStream0();
        } finally {
          Result result = getInvocationResult();
          publishInvocationFinishedEvent(result);
          publishBuildFinishedEvent(result);
        }
      } finally {
        besClient.shutdown();
      }
      return null;
    }

    private void publishBuildEnqueuedEvent() throws Exception {
      retryOnException(
          () -> {
            publishLifecycleEvent(besProtoUtil.buildEnqueued());
            return null;
          });
    }

    private void publishInvocationStartedEvent() throws Exception {
      retryOnException(
          () -> {
            publishLifecycleEvent(besProtoUtil.invocationStarted());
            return null;
          });
    }

    private void publishEventStream0() throws Exception {
      retryOnException(
          () -> {
            publishEventStream();
            return null;
          });
    }

    private void publishInvocationFinishedEvent(final Result result) throws Exception {
      retryOnException(
          () -> {
            publishLifecycleEvent(besProtoUtil.invocationFinished(result));
            return null;
          });
    }

    private void publishBuildFinishedEvent(final Result result) throws Exception {
      retryOnException(
          () -> {
            publishLifecycleEvent(besProtoUtil.buildFinished(result));
            return null;
          });
    }
  }

  /** Responsible for publishing lifecycle evnts RPC. Safe to retry. */
  private Status publishLifecycleEvent(PublishLifecycleEventRequest request) throws Exception {
    if (publishLifecycleEvents) {
      // Change the status based on BEP data
      return besClient.publish(request);
    }
    return Status.OK;
  }

  /**
   * Used as method reference, responsible for the entire Streaming RPC. Safe to retry. This method
   * carries over the state between consecutive calls (pendingAck messages will be added to the head
   * of the pendingSend queue), but that is intended behavior.
   */
  private void publishEventStream() throws Exception {
    // Reschedule unacked messages if required, keeping its original order.
    PublishBuildToolEventStreamRequest unacked;
    while ((unacked = pendingAck.pollLast()) != null) {
      pendingSend.addFirst(unacked);
    }
    pendingAck = new ConcurrentLinkedDeque<>();
    publishEventStream(pendingAck, pendingSend, besClient);
  }

  /** Method responsible for a single Streaming RPC. */
  private void publishEventStream(
      final ConcurrentLinkedDeque<PublishBuildToolEventStreamRequest> pendingAck,
      final BlockingDeque<PublishBuildToolEventStreamRequest> pendingSend,
      final BuildEventServiceClient besClient)
      throws Exception {
    ListenableFuture<Status> streamDone = besClient
        .openStream(ackCallback(pendingAck, besClient));
    try {
      @Nullable PublishBuildToolEventStreamRequest event;
      do {
        event = pendingSend.pollFirst(STREAMING_RPC_POLL_IN_SECS, TimeUnit.SECONDS);
        if (event != null) {
          pendingAck.add(event);
          besClient.sendOverStream(event);
        }
        checkState(besClient.isStreamActive(), "Stream was closed prematurely.");
      } while (!isLastEvent(event));
      logger.log(
          Level.INFO,
          String.format(
              "Will end publishEventStream() isLastEvent: %s isStreamActive: %s",
              isLastEvent(event), besClient.isStreamActive()));
    } catch (InterruptedException e) {
      // By convention the interrupted flag should have been cleared,
      // but just to be sure clear it.
      Thread.interrupted();
      String additionalDetails = "Sending build events.";
      besClient.abortStream(Status.CANCELLED.augmentDescription(additionalDetails));
      throw e;
    } catch (Exception e) {
      Status status = streamDone.isDone() ? streamDone.get() : null;
      String additionalDetail = e.getMessage();
      logger.log(
          Level.WARNING,
          String.format(
              "Aborting publishBuildToolEventStream RPC (status=%s): %s", status, additionalDetail),
          e);
      besClient.abortStream(Status.INTERNAL.augmentDescription(additionalDetail));
      throw e;
    }

    try {
      Status status =
          streamDone.get(PUBLISH_EVENT_STREAM_FINISHED_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS);
      logger.log(Level.INFO, "Done with publishEventStream(). Status: " + status);
    } catch (InterruptedException e) {
      // By convention the interrupted flag should have been cleared,
      // but just to be sure clear it.
      Thread.interrupted();
      String additionalDetails = "Waiting for ACK messages.";
      besClient.abortStream(Status.CANCELLED.augmentDescription(additionalDetails));
      throw e;
    } catch (TimeoutException e) {
      String additionalDetail = "Build Event Protocol upload timed out waiting for ACK messages";
      logger
          .log(Level.WARNING, "Cancelling publishBuildToolEventStream RPC: " + additionalDetail);
      besClient.abortStream(Status.CANCELLED.augmentDescription(additionalDetail));
      throw e;
    }
  }

  private static boolean isLastEvent(@Nullable PublishBuildToolEventStreamRequest event) {
    return event != null
        && event.getOrderedBuildEvent().getEvent().getEventCase() == COMPONENT_STREAM_FINISHED;
  }

  @SuppressWarnings("NonAtomicVolatileUpdate")
  private Function<PublishBuildToolEventStreamResponse, Void> ackCallback(
      final Deque<PublishBuildToolEventStreamRequest> pendingAck,
      final BuildEventServiceClient besClient) {
    return ack -> {
      long pendingSeq =
          pendingAck.isEmpty()
              ? -1
              : pendingAck.peekFirst().getOrderedBuildEvent().getSequenceNumber();
      long ackSeq = ack.getSequenceNumber();
      if (pendingSeq != ackSeq) {
        besClient.abortStream(
            Status.INTERNAL.augmentDescription(
                format("Expected ACK %s but was %s.", pendingSeq, ackSeq)));
        return null;
      }
      PublishBuildToolEventStreamRequest event = pendingAck.removeFirst();
      if (isLastEvent(event)) {
        logger.log(Level.INFO, "Last ACK received.");
        besClient.closeStream();
      }
      acksReceivedSinceLastRetry++;
      return null;
    };
  }

  /**
   * Executes a {@link Callable} retrying on exception thrown.
   */
  // TODO(eduardocolaco): Implement transient/persistent failures
  private void retryOnException(Callable<?> c) throws Exception {
    final int maxRetries = 5;
    final long initialDelayMillis = 0;
    final long delayMillis = 1000;

    int tries = 0;
    while (tries <= maxRetries) {
      try {
        acksReceivedSinceLastRetry = 0;
        c.call();
        lastKnownError = null;
        return;
      } catch (InterruptedException e) {
        throw e;
      } catch (Exception e) {
        if (acksReceivedSinceLastRetry > 0) {
          logger.fine(String.format("ACKs received since last retry %d.",
              acksReceivedSinceLastRetry));
          tries = 0;
        }
        tries++;
        lastKnownError = e;
        long sleepMillis;
        if (tries == 1) {
          sleepMillis = initialDelayMillis;
        } else {
          // This roughly matches the gRPC connection backoff.
          sleepMillis = (long) (delayMillis * Math.pow(1.6, tries));
        }
        String message = String.format("Retrying RPC to BES. Backoff %s ms.", sleepMillis);
        logger.log(Level.INFO, message, lastKnownError);
        sleeper.sleepMillis(sleepMillis);
      }
    }
    Preconditions.checkNotNull(lastKnownError);
    throw lastKnownError;
  }

  private void report(EventKind eventKind, String msg, Object... parameters) {
    commandLineReporter.handle(Event.of(eventKind, null, format(msg, parameters)));
  }

  @Nullable
  private static Throwable fromFuture(Future<?> f) {
    if (!f.isDone()) {
      return null;
    }
    try {
      f.get();
      return null;
    } catch (Throwable t) {
      return t;
    }
  }
}
