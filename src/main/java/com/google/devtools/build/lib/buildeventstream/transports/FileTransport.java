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

package com.google.devtools.build.lib.buildeventstream.transports;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.server.FailureDetails.BuildProgress;
import com.google.devtools.build.lib.server.FailureDetails.BuildProgress.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Non-blocking file transport.
 *
 * <p>Implementors of this class need to implement {@code #sendBuildEvent(BuildEvent)} which
 * serializes the build event and writes it to a file.
 */
abstract class FileTransport implements BuildEventTransport {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final BuildEventProtocolOptions options;
  private final BuildEventArtifactUploader uploader;
  private final SequentialWriter writer;
  private final ArtifactGroupNamer namer;

  private final ScheduledExecutorService timeoutExecutor =
      MoreExecutors.listeningDecorator(
          Executors.newSingleThreadScheduledExecutor(
              new ThreadFactoryBuilder().setNameFormat("file-uploader-timeout-%d").build()));

  FileTransport(
      BufferedOutputStream outputStream,
      BuildEventProtocolOptions options,
      BuildEventArtifactUploader uploader,
      ArtifactGroupNamer namer) {
    this.uploader = uploader;
    this.options = options;
    this.writer =
        new SequentialWriter(outputStream, this::serializeEvent, uploader, timeoutExecutor);
    this.namer = namer;
  }

  @ThreadSafe
  @VisibleForTesting
  static final class SequentialWriter implements Runnable {
    private static final ListenableFuture<BuildEventStreamProtos.BuildEvent> CLOSE_EVENT_FUTURE =
        Futures.immediateFailedFuture(
            new IllegalStateException(
                "A FileTransport is trying to write CLOSE_EVENT_FUTURE, this is a bug."));
    private static final Duration FLUSH_INTERVAL =
        Duration.ofMillis(
            Long.parseLong(System.getProperty("EXPERIMENTAL_BEP_FILE_FLUSH_MILLIS", "250")));

    private final Thread writerThread;
    private final BufferedOutputStream out;
    private final Function<BuildEventStreamProtos.BuildEvent, byte[]> serializeFunc;
    private final BuildEventArtifactUploader uploader;
    private final AtomicBoolean isClosed = new AtomicBoolean();
    private final SettableFuture<Void> closeFuture = SettableFuture.create();

    @VisibleForTesting
    final BlockingQueue<ListenableFuture<BuildEventStreamProtos.BuildEvent>> pendingWrites =
        new LinkedBlockingDeque<>();

    private ScheduledExecutorService timeoutExecutor;

    SequentialWriter(
        BufferedOutputStream outputStream,
        Function<BuildEventStreamProtos.BuildEvent, byte[]> serializeFunc,
        BuildEventArtifactUploader uploader,
        ScheduledExecutorService timeoutExecutor) {
      checkNotNull(uploader);

      this.out = checkNotNull(outputStream);
      this.writerThread = new Thread(this, "bep-local-writer");
      this.serializeFunc = checkNotNull(serializeFunc);
      this.uploader = checkNotNull(uploader);
      this.timeoutExecutor = checkNotNull(timeoutExecutor);
      writerThread.start();
    }

    @Override
    public void run() {
      ListenableFuture<BuildEventStreamProtos.BuildEvent> buildEventF;
      try {
        Instant prevFlush = Instant.now();
        while ((buildEventF = pendingWrites.poll(FLUSH_INTERVAL.toMillis(), TimeUnit.MILLISECONDS))
            != CLOSE_EVENT_FUTURE) {
          if (buildEventF != null) {
            BuildEventStreamProtos.BuildEvent buildEvent = buildEventF.get();
            byte[] serialized = serializeFunc.apply(buildEvent);
            out.write(serialized);
          }
          Instant now = Instant.now();
          if (buildEventF == null || now.compareTo(prevFlush.plus(FLUSH_INTERVAL)) > 0) {
            // Some users, e.g. Tulsi, expect prompt BEP stream flushes for interactive use.
            out.flush();
            prevFlush = now;
          }
        }
      } catch (ExecutionException e) {
        if (e.getCause() instanceof RuntimeException || e.getCause() instanceof Error) {
          closeFuture.setException(e.getCause());
        }
        exitFailure(e);
      } catch (IOException | InterruptedException | CancellationException e) {
        exitFailure(e);
      } finally {
        try {
          out.flush();
          out.close();
        } catch (IOException e) {
          logger.atSevere().withCause(e).log("Failed to close BEP file output stream.");
        } finally {
          uploader.shutdown();
          timeoutExecutor.shutdown();
        }
        closeFuture.set(null);
      }
    }

    private void exitFailure(Throwable e) {
      final String message;
      // Print a more useful error message when the upload times out.
      // An {@link ExecutionException} may be wrapping a {@link TimeoutException} if the
      // Future was created with {@link Futures#withTimeout}.
      if (e instanceof ExecutionException
          && e.getCause() instanceof TimeoutException) {
        message = "Unable to write all BEP events to file due to timeout";
      } else {
        message =
            String.format("Unable to write all BEP events to file due to '%s'", e.getMessage());
      }
      closeFuture.setException(
          new AbruptExitException(
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(message)
                      .setBuildProgress(BuildProgress.newBuilder().setCode(getBuildProgressCode(e)))
                      .build()),
              e));
      pendingWrites.clear();
      logger.atSevere().withCause(e).log(message);
    }

    private static BuildProgress.Code getBuildProgressCode(Throwable e) {
      if (e instanceof ExecutionException && e.getCause() instanceof TimeoutException) {
        return Code.BES_FILE_WRITE_TIMEOUT;
      }
      Throwable maybeUnwrappedFailure = e instanceof ExecutionException ? e.getCause() : e;
      if (maybeUnwrappedFailure instanceof IOException) {
        return Code.BES_FILE_WRITE_IO_ERROR;
      }
      if (maybeUnwrappedFailure instanceof InterruptedException) {
        return Code.BES_FILE_WRITE_INTERRUPTED;
      }
      if (maybeUnwrappedFailure instanceof CancellationException) {
        return Code.BES_FILE_WRITE_CANCELED;
      }
      return Code.BES_FILE_WRITE_UNKNOWN_ERROR;
    }

    private void closeNow() {
      if (closeFuture.isDone()) {
        return;
      }
      try {
        pendingWrites.clear();
        pendingWrites.put(CLOSE_EVENT_FUTURE);
      } catch (InterruptedException e) {
        logger.atSevere().withCause(e).log("Failed to immediately close the sequential writer.");
      }
    }

    ListenableFuture<Void> close() {
      if (isClosed.getAndSet(true)) {
        return closeFuture;
      } else if (closeFuture.isDone()) {
        return closeFuture;
      }

      // Close abruptly if the closing future is cancelled.
      closeFuture.addListener(
          () -> {
            if (closeFuture.isCancelled()) {
              closeNow();
            }
          },
          MoreExecutors.directExecutor());

      try {
        pendingWrites.put(CLOSE_EVENT_FUTURE);
      } catch (InterruptedException e) {
        closeNow();
        logger.atSevere().withCause(e).log("Failed to close the sequential writer.");
        closeFuture.set(null);
      }
      return closeFuture;
    }

    private Duration getFlushInterval() {
      return FLUSH_INTERVAL;
    }
  }

  @Override
  public void sendBuildEvent(BuildEvent event) {
    if (writer.isClosed.get()) {
      return;
    }
    try {
      if (!writer.pendingWrites.add(asStreamProto(event, namer))) {
        logger.atSevere().log("Failed to add BEP event to the write queue");
      }
    } catch (RejectedExecutionException e) {
      // If early shutdown races with this event, log but otherwise ignore.
      logger.atWarning().withCause(e).log("Event upload started after shutdown");
    }
  }

  protected abstract byte[] serializeEvent(BuildEventStreamProtos.BuildEvent buildEvent);

  @Override
  public ListenableFuture<Void> close() {
    return writer.close();
  }

  /**
   * Converts the given event into a proto object; this may trigger uploading of referenced files as
   * a side effect. May return {@code null} if there was an interrupt. This method is not
   * thread-safe.
   */
  private ListenableFuture<BuildEventStreamProtos.BuildEvent> asStreamProto(
      BuildEvent event, ArtifactGroupNamer namer) {
    checkNotNull(event);

    ListenableFuture<PathConverter> converterFuture =
        uploader.uploadReferencedLocalFiles(event.referencedLocalFiles());
    ListenableFuture<?> remoteUploads =
        uploader.waitForRemoteUploads(event.remoteUploads(), timeoutExecutor);
    return Futures.transform(
        Futures.allAsList(converterFuture, remoteUploads),
        results -> {
          BuildEventContext context =
              new BuildEventContext() {
                @Override
                public PathConverter pathConverter() {
                  return Futures.getUnchecked(converterFuture);
                }

                @Override
                public ArtifactGroupNamer artifactGroupNamer() {
                  return namer;
                }

                @Override
                public BuildEventProtocolOptions getOptions() {
                  return options;
                }
              };
          try {
            return event.asStreamProto(context);
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
          }
        },
        MoreExecutors.directExecutor());
  }

  @Override
  public boolean mayBeSlow() {
    return uploader.mayBeSlow();
  }

  @Override
  public BuildEventArtifactUploader getUploader() {
    return uploader;
  }

  /** Determines how often the {@link FileTransport} flushes events. */
  Duration getFlushInterval() {
    return writer.getFlushInterval();
  }
}

