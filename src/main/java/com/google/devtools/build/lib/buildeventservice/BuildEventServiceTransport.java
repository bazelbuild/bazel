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
import static com.google.devtools.build.v1.BuildStatus.Result.COMMAND_FAILED;
import static com.google.devtools.build.v1.BuildStatus.Result.COMMAND_SUCCEEDED;
import static com.google.devtools.build.v1.BuildStatus.Result.UNKNOWN_STATUS;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildCompletingEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.v1.BuildStatus.Result;
import com.google.devtools.build.v1.PublishBuildToolEventStreamRequest;
import com.google.devtools.build.v1.PublishLifecycleEventRequest;
import com.google.protobuf.Any;
import com.google.protobuf.Timestamp;
import com.google.protobuf.util.Timestamps;
import io.grpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusException;
import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.concurrent.GuardedBy;
import javax.annotation.concurrent.Immutable;

/** A {@link BuildEventTransport} that streams {@link BuildEvent}s to BuildEventService. */
public class BuildEventServiceTransport implements BuildEventTransport {

  private final BuildEventServiceUploader besUploader;

  /** A builder for {@link BuildEventServiceTransport}. */
  public static class Builder {
    private boolean publishLifecycleEvents;
    private Duration closeTimeout;
    private Sleeper sleeper;
    private BuildEventLogger buildEventLogger;

    /** Whether to publish lifecycle events. */
    public Builder publishLifecycleEvents(boolean publishLifecycleEvents) {
      this.publishLifecycleEvents = publishLifecycleEvents;
      return this;
    }

    /** The time to wait for the build event upload after the build has completed. */
    public Builder closeTimeout(Duration closeTimeout) {
      this.closeTimeout = closeTimeout;
      return this;
    }

    public Builder buildEventLogger(BuildEventLogger buildEventLogger) {
      this.buildEventLogger = buildEventLogger;
      return this;
    }

    @VisibleForTesting
    public Builder sleeper(Sleeper sleeper) {
      this.sleeper = sleeper;
      return this;
    }

    public BuildEventServiceTransport build(
        BuildEventServiceClient besClient,
        BuildEventArtifactUploader localFileUploader,
        BuildEventProtocolOptions bepOptions,
        BuildEventServiceProtoUtil besProtoUtil,
        Clock clock,
        ExitFunction exitFunction) {

      return new BuildEventServiceTransport(
          besClient,
          localFileUploader,
          bepOptions,
          besProtoUtil,
          clock,
          exitFunction,
          publishLifecycleEvents,
          closeTimeout != null ? closeTimeout : Duration.ZERO,
          sleeper != null ? sleeper : new JavaSleeper(),
          buildEventLogger != null ? buildEventLogger : (e) -> {});
    }
  }

  private BuildEventServiceTransport(
      BuildEventServiceClient besClient,
      BuildEventArtifactUploader localFileUploader,
      BuildEventProtocolOptions bepOptions,
      BuildEventServiceProtoUtil besProtoUtil,
      Clock clock,
      ExitFunction exitFunc,
      boolean publishLifecycleEvents,
      Duration closeTimeout,
      Sleeper sleeper,
      BuildEventLogger buildEventLogger) {
    this.besUploader =
        new BuildEventServiceUploader(
            besClient,
            localFileUploader,
            besProtoUtil,
            bepOptions,
            publishLifecycleEvents,
            closeTimeout,
            exitFunc,
            sleeper,
            clock,
            buildEventLogger);
  }

  @Override
  public ListenableFuture<Void> close() {
    // This future completes once the upload has finished. As
    // per API contract it is expected to never fail.
    SettableFuture<Void> closeFuture = SettableFuture.create();
    ListenableFuture<Void> uploaderCloseFuture = besUploader.close();
    uploaderCloseFuture.addListener(() -> closeFuture.set(null), MoreExecutors.directExecutor());
    return closeFuture;
  }

  @Override
  public void closeNow() {
    besUploader.closeNow(/*causedByTimeout=*/ false);
  }

  @Override
  public String name() {
    return "Build Event Service";
  }

  @Override
  public void sendBuildEvent(BuildEvent event, final ArtifactGroupNamer namer) {
    besUploader.enqueueEvent(event, namer);
  }

  /** BuildEventLogger can be used to log build event (stats). */
  @FunctionalInterface
  public interface BuildEventLogger {
    void log(BuildEventStreamProtos.BuildEvent buildEvent);
  }

  /**
   * Called by the {@link BuildEventServiceUploader} in case of error to asynchronously notify Bazel
   * of an error.
   */
  @FunctionalInterface
  public interface ExitFunction {
    void accept(String message, Exception cause);
  }

  /**
   * This method is only used in tests. Once TODO(b/113035235) is fixed the close future will also
   * carry error messages.
   */
  @VisibleForTesting
  public void throwUploaderError() throws Exception {
    synchronized (besUploader.lock) {
      checkState(besUploader.closeFuture != null && besUploader.closeFuture.isDone());
      try {
        besUploader.closeFuture.get();
      } catch (ExecutionException e) {
        throw (Exception) e.getCause();
      }
    }
  }

  /** Implements the BES upload which includes uploading the lifecycle and build events. */
  private static class BuildEventServiceUploader implements Runnable {

    private static final Logger logger =
        Logger.getLogger(BuildEventServiceUploader.class.getName());

    /** Configuration knobs related to RPC retries. Values chosen by good judgement. */
    private static final int MAX_NUM_RETRIES = 4;

    private static final int DELAY_MILLIS = 1000;

    private final BuildEventServiceClient besClient;
    private final BuildEventArtifactUploader localFileUploader;
    private final BuildEventServiceProtoUtil besProtoUtil;
    private final BuildEventProtocolOptions protocolOptions;
    private final boolean publishLifecycleEvents;
    private final Duration closeTimeout;
    private final ExitFunction exitFunc;
    private final Sleeper sleeper;
    private final Clock clock;
    private final BuildEventLogger buildEventLogger;

    /**
     * The event queue contains two types of events: - Build events, sorted by sequence number, that
     * should be sent to the server - Command events that are used by {@link #publishBuildEvents()}
     * to change state.
     */
    private final BlockingDeque<EventLoopCommand> eventQueue = new LinkedBlockingDeque<>();

    /**
     * Computes sequence numbers for build events. As per the BES protocol, sequence numbers must be
     * consecutive monotonically increasing natural numbers.
     */
    private final AtomicLong nextSeqNum = new AtomicLong(1);

    private final Object lock = new Object();

    @GuardedBy("lock")
    private Result buildStatus = UNKNOWN_STATUS;

    /**
     * Initialized only after the first call to {@link #close()} or if the upload fails before that.
     * The {@code null} state is used throughout the code to make multiple calls to {@link #close()}
     * idempotent.
     */
    @GuardedBy("lock")
    private SettableFuture<Void> closeFuture;

    /**
     * The thread that calls the lifecycle RPCs and does the build event upload. It's started lazily
     * on the first call to {@link #enqueueEvent(BuildEvent, ArtifactGroupNamer)} or {@link
     * #close()} (which ever comes first).
     */
    @GuardedBy("lock")
    private Thread uploadThread;

    @GuardedBy("lock")
    private boolean interruptCausedByTimeout;

    public BuildEventServiceUploader(
        BuildEventServiceClient besClient,
        BuildEventArtifactUploader localFileUploader,
        BuildEventServiceProtoUtil besProtoUtil,
        BuildEventProtocolOptions protocolOptions,
        boolean publishLifecycleEvents,
        Duration closeTimeout,
        ExitFunction exitFunc,
        Sleeper sleeper,
        Clock clock,
        BuildEventLogger buildEventLogger) {
      this.besClient = Preconditions.checkNotNull(besClient);
      this.localFileUploader = Preconditions.checkNotNull(localFileUploader);
      this.besProtoUtil = Preconditions.checkNotNull(besProtoUtil);
      this.protocolOptions = Preconditions.checkNotNull(protocolOptions);
      this.publishLifecycleEvents = publishLifecycleEvents;
      this.closeTimeout = Preconditions.checkNotNull(closeTimeout);
      this.exitFunc = Preconditions.checkNotNull(exitFunc);
      this.sleeper = Preconditions.checkNotNull(sleeper);
      this.clock = Preconditions.checkNotNull(clock);
      this.buildEventLogger = Preconditions.checkNotNull(buildEventLogger);
    }

    /** Enqueues an event for uploading to a BES backend. */
    public void enqueueEvent(BuildEvent event, ArtifactGroupNamer namer) {
      // This needs to happen outside a synchronized block as it may trigger
      // stdout/stderr and lead to a deadlock. See b/109725432
      ListenableFuture<PathConverter> localFileUploadFuture =
          uploadReferencedLocalFiles(event.referencedLocalFiles());

      synchronized (lock) {
        if (closeFuture != null) {
          // Close has been called and thus we silently ignore any further events and cancel
          // any pending file uploads
          closeFuture.addListener(
              () -> {
                if (!localFileUploadFuture.isDone()) {
                  localFileUploadFuture.cancel(true);
                }
              },
              MoreExecutors.directExecutor());
          return;
        }
        // BuildCompletingEvent marks the end of the build in the BEP event stream.
        if (event instanceof BuildCompletingEvent) {
          this.buildStatus = extractBuildStatus((BuildCompletingEvent) event);
        }
        ensureUploadThreadStarted();
        eventQueue.addLast(
            new SendRegularBuildEventCommand(
                event, namer, localFileUploadFuture, nextSeqNum.getAndIncrement(), currentTime()));
      }
    }

    /**
     * Gracefully stops the BES upload. All events enqueued before the call to close will be
     * uploaded and events enqueued after the call will be discarded.
     *
     * <p>The returned future completes when the upload completes. It's guaranteed to never fail.
     */
    public ListenableFuture<Void> close() {
      synchronized (lock) {
        if (closeFuture != null) {
          return closeFuture;
        }
        ensureUploadThreadStarted();

        closeFuture = SettableFuture.create();

        // Enqueue the last event which will terminate the upload.
        eventQueue.addLast(
            new SendLastBuildEventCommand(nextSeqNum.getAndIncrement(), currentTime()));

        if (!closeTimeout.isZero()) {
          startCloseTimer(closeFuture, closeTimeout);
        }
        return closeFuture;
      }
    }

    /** Stops the upload immediately. Enqueued events that have not been sent yet will be lost. */
    public void closeNow(boolean causedByTimeout) {
      synchronized (lock) {
        if (uploadThread != null) {
          if (uploadThread.isInterrupted()) {
            return;
          }

          interruptCausedByTimeout = causedByTimeout;
          uploadThread.interrupt();
        }
      }
    }

    @Override
    public void run() {
      try {
        if (publishLifecycleEvents) {
          publishLifecycleEvent(besProtoUtil.buildEnqueued(currentTime()));
          publishLifecycleEvent(besProtoUtil.invocationStarted(currentTime()));
        }

        try {
          publishBuildEvents();
        } finally {
          if (publishLifecycleEvents) {
            Result buildStatus;
            synchronized (lock) {
              buildStatus = this.buildStatus;
            }
            publishLifecycleEvent(besProtoUtil.invocationFinished(currentTime(), buildStatus));
            publishLifecycleEvent(besProtoUtil.buildFinished(currentTime(), buildStatus));
          }
        }
        exitFunc.accept("The Build Event Protocol upload finished successfully", null);
        synchronized (lock) {
          // Invariant: closeFuture is not null.
          // publishBuildEvents() only terminates successfully after SendLastBuildEventCommand
          // has been sent successfully and that event is only added to the eventQueue during a
          // call to close() which initializes the closeFuture.
          closeFuture.set(null);
        }
      } catch (InterruptedException e) {
        try {
          logInfo(e, "Aborting the BES upload due to having received an interrupt");
          synchronized (lock) {
            if (interruptCausedByTimeout) {
              exitFunc.accept("The Build Event Protocol upload timed out", e);
            }
          }
        } finally {
          // TODO(buchgr): Due to b/113035235 exitFunc needs to be called before the close future
          // completes.
          failCloseFuture(e);
        }
      } catch (StatusException e) {
        try {
          String message =
              "The Build Event Protocol upload failed: " + besClient.userReadableError(e);
          logInfo(e, message);
          exitFunc.accept(message, e);
        } finally {
          failCloseFuture(e);
        }
      } catch (LocalFileUploadException e) {
        try {
          String message =
              "The Build Event Protocol local file upload failed: " + e.getCause().getMessage();
          logInfo((Exception) e.getCause(), message);
          exitFunc.accept(message, (Exception) e.getCause());
        } finally {
          failCloseFuture((Exception) e.getCause());
        }
      } catch (RuntimeException e) {
        failCloseFuture(e);
        logError(e, "BES upload failed due to a RuntimeException. This is a bug.");
        throw e;
      } finally {
        try {
          besClient.shutdown();
        } finally {
          localFileUploader.shutdown();
        }
      }
    }

    private void publishBuildEvents()
        throws StatusException, LocalFileUploadException, InterruptedException {
      eventQueue.addFirst(new OpenStreamCommand());

      // Every build event sent to the server needs to be acknowledged by it. This queue stores
      // the build events that have been sent and still have to be acknowledged by the server.
      // The build events are stored in the order they were sent.
      ConcurrentLinkedDeque<SendBuildEventCommand> ackQueue = new ConcurrentLinkedDeque<>();
      boolean lastEventSent = false;
      int acksReceived = 0;
      int retryAttempt = 0;

      try {
        // OPEN_STREAM is the first event and opens a bidi streaming RPC for sending build events
        // and receiving ACKs.
        // SEND_BUILD_EVENT sends a build event to the server. Sending of the Nth build event does
        // does not wait for the ACK of the N-1th build event to have been received.
        // SEND_LAST_EVENT sends the last build event and half closes the RPC.
        // ACK_RECEIVED is executed for every ACK from the server and checks that the ACKs are in
        // the correct order.
        // STREAM_COMPLETE checks that all build events have been sent and all ACKs have been
        // received. If not it invokes a retry logic that may decide to re-send every build event
        // for which an ACK has not been received. If so, it adds an OPEN_STREAM event.
        while (true) {
          EventLoopCommand event = eventQueue.takeFirst();
          switch (event.type()) {
            case OPEN_STREAM:
              {
                // Invariant: the eventQueue only contains events of type SEND_BUILD_EVENT
                // or SEND_LAST_EVENT
                logInfo("Starting publishBuildEvents: eventQueue=%d", eventQueue.size());
                ListenableFuture<Status> streamFuture =
                    besClient.openStream(
                        (ack) ->
                            eventQueue.addLast(new AckReceivedCommand(ack.getSequenceNumber())));
                addStreamStatusListener(
                    streamFuture,
                    (status) -> eventQueue.addLast(new StreamCompleteCommand(status)));
              }
              break;

            case SEND_BUILD_EVENT:
              {
                // Invariant: the eventQueue may contain events of any type
                SendRegularBuildEventCommand buildEvent = (SendRegularBuildEventCommand) event;
                ackQueue.addLast(buildEvent);
                PathConverter pathConverter = waitForLocalFileUploads(buildEvent);
                besClient.sendOverStream(buildEvent.serialize(pathConverter));
              }
              break;

            case SEND_LAST_EVENT:
              {
                // Invariant: the eventQueue may contain events of any type
                SendLastBuildEventCommand lastEvent = (SendLastBuildEventCommand) event;
                ackQueue.addLast(lastEvent);
                lastEventSent = true;
                besClient.sendOverStream(lastEvent.serialize());
                besClient.halfCloseStream();
              }
              break;

            case ACK_RECEIVED:
              {
                // Invariant: the eventQueue may contain events of any type
                AckReceivedCommand ackEvent = (AckReceivedCommand) event;
                if (!ackQueue.isEmpty()) {
                  SendBuildEventCommand expected = ackQueue.removeFirst();
                  long actualSeqNum = ackEvent.getSequenceNumber();
                  if (expected.getSequenceNumber() == actualSeqNum) {
                    acksReceived++;
                  } else {
                    ackQueue.addFirst(expected);
                    String message =
                        String.format(
                            "Expected ACK with seqNum=%d but received ACK with seqNum=%d",
                            expected.getSequenceNumber(), actualSeqNum);
                    logInfo(message);
                    besClient.abortStream(Status.FAILED_PRECONDITION.withDescription(message));
                  }
                } else {
                  String message =
                      String.format(
                          "Received ACK (seqNum=%d) when no ACK was expected",
                          ackEvent.getSequenceNumber());
                  logInfo(message);
                  besClient.abortStream(Status.FAILED_PRECONDITION.withDescription(message));
                }
              }
              break;

            case STREAM_COMPLETE:
              {
                // Invariant: the eventQueue only contains events of type SEND_BUILD_EVENT
                // or SEND_LAST_EVENT
                StreamCompleteCommand completeEvent = (StreamCompleteCommand) event;
                Status streamStatus = completeEvent.status();
                if (streamStatus.isOk()) {
                  if (lastEventSent && ackQueue.isEmpty()) {
                    logInfo("publishBuildEvents was successful");
                    // Upload successful. Break out from the while(true) loop.
                    return;
                  } else {
                    throw (lastEventSent
                            ? ackQueueNotEmptyStatus(ackQueue.size())
                            : lastEventNotSentStatus())
                        .asException();
                  }
                }

                if (!shouldRetryStatus(streamStatus)) {
                  logInfo("Not retrying publishBuildEvents: status='%s'", streamStatus);
                  throw streamStatus.asException();
                }
                if (retryAttempt == MAX_NUM_RETRIES) {
                  logInfo(
                      "Not retrying publishBuildEvents, no more attempts left: status='%s'",
                      streamStatus);
                  throw streamStatus.asException();
                }

                // Retry logic
                // Adds events from the ackQueue to the front of the eventQueue, so that the
                // events in the eventQueue are sorted by sequence number (ascending).
                SendBuildEventCommand unacked;
                while ((unacked = ackQueue.pollLast()) != null) {
                  eventQueue.addFirst(unacked);
                }

                long sleepMillis = retrySleepMillis(retryAttempt);
                logInfo(
                    "Retrying publishLifecycleEvent: status='%s', sleepMillis=%d",
                    streamStatus, sleepMillis);
                sleeper.sleepMillis(sleepMillis);

                // If we made progress, meaning the server ACKed events that we sent, then reset
                // the retry counter to 0.
                if (acksReceived > 0) {
                  retryAttempt = 0;
                } else {
                  retryAttempt++;
                }
                acksReceived = 0;
                eventQueue.addFirst(new OpenStreamCommand());
              }
              break;
          }
        }
      } catch (InterruptedException | LocalFileUploadException e) {
        int limit = 30;
        logInfo(
            String.format(
                "Publish interrupt. Showing up to %d items from queues: ack_queue_size: %d, "
                    + "ack_queue: %s, event_queue_size: %d, event_queue: %s",
                limit,
                ackQueue.size(),
                Iterables.limit(ackQueue, limit),
                eventQueue.size(),
                Iterables.limit(eventQueue, limit)));
        besClient.abortStream(Status.CANCELLED);
        throw e;
      } finally {
        // Cancel all local file uploads that may still be running
        // of events that haven't been uploaded.
        EventLoopCommand event;
        while ((event = ackQueue.pollFirst()) != null) {
          if (event instanceof SendRegularBuildEventCommand) {
            cancelLocalFileUpload((SendRegularBuildEventCommand) event);
          }
        }
        while ((event = eventQueue.pollFirst()) != null) {
          if (event instanceof SendRegularBuildEventCommand) {
            cancelLocalFileUpload((SendRegularBuildEventCommand) event);
          }
        }
      }
    }

    private void cancelLocalFileUpload(SendRegularBuildEventCommand event) {
      ListenableFuture<PathConverter> localFileUploaderFuture = event.localFileUploadProgress();
      if (!localFileUploaderFuture.isDone()) {
        localFileUploaderFuture.cancel(true);
      }
    }

    /** Sends a {@link PublishLifecycleEventRequest} to the BES backend. */
    private void publishLifecycleEvent(PublishLifecycleEventRequest request)
        throws StatusException, InterruptedException {
      int retryAttempt = 0;
      StatusException cause = null;
      while (retryAttempt <= MAX_NUM_RETRIES) {
        try {
          besClient.publish(request);
          return;
        } catch (StatusException e) {
          if (!shouldRetryStatus(e.getStatus())) {
            logInfo("Not retrying publishLifecycleEvent: status='%s'", e.getStatus().toString());
            throw e;
          }

          cause = e;

          long sleepMillis = retrySleepMillis(retryAttempt);
          logInfo(
              "Retrying publishLifecycleEvent: status='%s', sleepMillis=%d",
              e.getStatus().toString(), sleepMillis);
          sleeper.sleepMillis(sleepMillis);
          retryAttempt++;
        }
      }

      // All retry attempts failed
      throw cause;
    }

    private ListenableFuture<PathConverter> uploadReferencedLocalFiles(
        Collection<LocalFile> localFiles) {
      Map<Path, LocalFile> localFileMap = new TreeMap<>();
      for (LocalFile localFile : localFiles) {
        // It is possible for targets to have duplicate artifacts (same path but different owners)
        // in their output groups. Since they didn't trigger an artifact conflict they are the
        // same file, so just skip either one
        localFileMap.putIfAbsent(localFile.path, localFile);
      }
      return localFileUploader.upload(localFileMap);
    }

    private void ensureUploadThreadStarted() {
      synchronized (lock) {
        if (uploadThread == null) {
          uploadThread = new Thread(this, "bes-uploader");
          uploadThread.start();
        }
      }
    }

    private void startCloseTimer(ListenableFuture<Void> closeFuture, Duration closeTimeout) {
      Thread closeTimer =
          new Thread(
              () -> {
                // Call closeNow() if the future does not complete within closeTimeout
                try {
                  closeFuture.get(closeTimeout.toMillis(), TimeUnit.MILLISECONDS);
                } catch (InterruptedException | TimeoutException e) {
                  closeNow(/*causedByTimeout=*/ true);
                } catch (ExecutionException e) {
                  // Intentionally left empty, because this code only cares about
                  // calling closeNow() if the closeFuture does not complete within
                  // closeTimeout.
                }
              },
              "bes-uploader-close-timer");
      closeTimer.start();
    }

    private void failCloseFuture(Exception cause) {
      synchronized (lock) {
        if (closeFuture == null) {
          closeFuture = SettableFuture.create();
        }
        closeFuture.setException(cause);
      }
    }

    private PathConverter waitForLocalFileUploads(SendRegularBuildEventCommand orderedBuildEvent)
        throws LocalFileUploadException, InterruptedException {
      try {
        // Wait for the local file upload to have been completed.
        return orderedBuildEvent.localFileUploadProgress().get();
      } catch (ExecutionException e) {
        logger.log(
            Level.WARNING,
            String.format(
                "Failed to upload local files referenced by build event: %s", e.getMessage()),
            e);
        throw new LocalFileUploadException((Exception) e.getCause());
      }
    }

    private Timestamp currentTime() {
      return Timestamps.fromMillis(clock.currentTimeMillis());
    }

    private static Result extractBuildStatus(BuildCompletingEvent event) {
      if (event.getExitCode() != null && event.getExitCode().getNumericExitCode() == 0) {
        return COMMAND_SUCCEEDED;
      } else {
        return COMMAND_FAILED;
      }
    }

    private static Status lastEventNotSentStatus() {
      return Status.FAILED_PRECONDITION.withDescription(
          "Server closed stream with status OK but not all events have been sent");
    }

    private static Status ackQueueNotEmptyStatus(int ackQueueSize) {
      return Status.FAILED_PRECONDITION.withDescription(
          String.format(
              "Server closed stream with status OK but not all ACKs have been"
                  + " received (ackQueue=%d)",
              ackQueueSize));
    }

    private static void addStreamStatusListener(
        ListenableFuture<Status> stream, Consumer<Status> onDone) {
      Futures.addCallback(
          stream,
          new FutureCallback<Status>() {
            @Override
            public void onSuccess(Status result) {
              onDone.accept(result);
            }

            @Override
            public void onFailure(Throwable t) {}
          },
          MoreExecutors.directExecutor());
    }

    private static boolean shouldRetryStatus(Status status) {
      return !status.getCode().equals(Code.INVALID_ARGUMENT)
          && !status.getCode().equals(Code.FAILED_PRECONDITION);
    }

    private static long retrySleepMillis(int attempt) {
      // This somewhat matches the backoff used for gRPC connection backoffs.
      return (long) (DELAY_MILLIS * Math.pow(1.6, attempt));
    }

    private static void logInfo(String message, Object... args) {
      logger.log(Level.INFO, String.format(message, args));
    }

    private static void logInfo(Exception cause, String message, Object... args) {
      logger.log(Level.INFO, String.format(message, args), cause);
    }

    private static void logError(Exception cause, String message, Object... args) {
      logger.log(Level.SEVERE, String.format(message, args), cause);
    }

    /** A command that may be added to the {@code eventQueue}. */
    private interface EventLoopCommand {

      /**
       * The event types are used to switch between states in the event loop in {@link
       * #publishBuildEvents()}
       */
      enum Type {
        /** Tells the event loop to open a new BES stream */
        OPEN_STREAM,
        /** Tells the event loop to send the build event */
        SEND_BUILD_EVENT,
        /** Tells the event loop that an ACK was received */
        ACK_RECEIVED,
        /** Tells the event loop that this is the last event of the stream */
        SEND_LAST_EVENT,
        /** Tells the event loop that the streaming RPC completed */
        STREAM_COMPLETE
      }

      Type type();
    }

    @Immutable
    private static final class OpenStreamCommand implements EventLoopCommand {

      @Override
      public Type type() {
        return Type.OPEN_STREAM;
      }
    }

    @Immutable
    private static final class StreamCompleteCommand implements EventLoopCommand {

      private final Status status;

      public StreamCompleteCommand(Status status) {
        this.status = status;
      }

      public Status status() {
        return status;
      }

      @Override
      public Type type() {
        return Type.STREAM_COMPLETE;
      }
    }

    @Immutable
    private static final class AckReceivedCommand implements EventLoopCommand {

      private final long sequenceNumber;

      public AckReceivedCommand(long sequenceNumber) {
        this.sequenceNumber = sequenceNumber;
      }

      public long getSequenceNumber() {
        return sequenceNumber;
      }

      @Override
      public Type type() {
        return Type.ACK_RECEIVED;
      }

      @Override
      public String toString() {
        return MoreObjects.toStringHelper(this).add("seq_num", getSequenceNumber()).toString();
      }
    }

    private abstract static class SendBuildEventCommand implements EventLoopCommand {

      abstract long getSequenceNumber();

      @Override
      public String toString() {
        return MoreObjects.toStringHelper(this).add("seq_num", getSequenceNumber()).toString();
      }
    }

    private final class SendRegularBuildEventCommand extends SendBuildEventCommand {

      private final BuildEvent event;
      private final ArtifactGroupNamer namer;
      private final ListenableFuture<PathConverter> localFileUpload;
      private final long sequenceNumber;
      private final Timestamp creationTime;

      private SendRegularBuildEventCommand(
          BuildEvent event,
          ArtifactGroupNamer namer,
          ListenableFuture<PathConverter> localFileUpload,
          long sequenceNumber,
          Timestamp creationTime) {
        this.event = event;
        this.namer = namer;
        this.localFileUpload = localFileUpload;
        this.sequenceNumber = sequenceNumber;
        this.creationTime = creationTime;
      }

      public ListenableFuture<PathConverter> localFileUploadProgress() {
        return localFileUpload;
      }

      public PublishBuildToolEventStreamRequest serialize(PathConverter pathConverter) {
        BuildEventContext ctx =
            new BuildEventContext() {
              @Override
              public PathConverter pathConverter() {
                return pathConverter;
              }

              @Override
              public ArtifactGroupNamer artifactGroupNamer() {
                return namer;
              }

              @Override
              public BuildEventProtocolOptions getOptions() {
                return protocolOptions;
              }
            };
        BuildEventStreamProtos.BuildEvent serializedBepEvent = event.asStreamProto(ctx);
        buildEventLogger.log(serializedBepEvent);
        return besProtoUtil.bazelEvent(sequenceNumber, creationTime, Any.pack(serializedBepEvent));
      }

      @Override
      public long getSequenceNumber() {
        return sequenceNumber;
      }

      @Override
      public Type type() {
        return Type.SEND_BUILD_EVENT;
      }

      @Override
      public String toString() {
        return super.toString() + " - [" + event + "]";
      }
    }

    @Immutable
    private final class SendLastBuildEventCommand extends SendBuildEventCommand {

      private final long sequenceNumber;
      private final Timestamp creationTime;

      SendLastBuildEventCommand(long sequenceNumber, Timestamp creationTime) {
        this.sequenceNumber = sequenceNumber;
        this.creationTime = creationTime;
      }

      public PublishBuildToolEventStreamRequest serialize() {
        return besProtoUtil.streamFinished(sequenceNumber, creationTime);
      }

      @Override
      public long getSequenceNumber() {
        return sequenceNumber;
      }

      @Override
      public Type type() {
        return Type.SEND_LAST_EVENT;
      }
    }

    private static class LocalFileUploadException extends Exception {

      public LocalFileUploadException(Exception cause) {
        super(cause);
      }
    }
  }
}
