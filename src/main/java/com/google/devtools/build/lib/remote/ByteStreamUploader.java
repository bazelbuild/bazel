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

import static com.google.devtools.build.lib.util.Preconditions.checkArgument;
import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.util.Preconditions.checkState;
import static java.lang.String.format;
import static java.util.Collections.singletonList;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListenableScheduledFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.ByteString;
import io.grpc.CallCredentials;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A client implementing the {@code Write} method of the {@code ByteStream} gRPC service.
 *
 * <p>Users must call {@link #shutdown()} before exiting.
 */
final class ByteStreamUploader {

  private final String instanceName;
  private final Channel channel;
  private final CallCredentials callCredentials;
  private final long callTimeoutSecs;
  private final Retrier retrier;
  private final ListeningScheduledExecutorService retryService;

  private final Object lock = new Object();

  @GuardedBy("lock")
  private final Map<Digest, ListenableFuture<Void>> uploadsInProgress = new HashMap<>();

  @GuardedBy("lock")
  private boolean isShutdown;

  /**
   * Creates a new instance.
   *
   * @param instanceName the instance name to be prepended to resource name of the {@code Write}
   *     call. See the {@code ByteStream} service definition for details
   * @param channel the {@link io.grpc.Channel} to use for calls
   * @param callCredentials the credentials to use for authentication. May be {@code null}, in which
   *     case no authentication is performed
   * @param callTimeoutSecs the timeout in seconds after which a {@code Write} gRPC call must be
   *     complete. The timeout resets between retries
   * @param retrier the {@link Retrier} whose backoff strategy to use for retry timings.
   * @param retryService the executor service to schedule retries on. It's the responsibility of the
   *     caller to properly shutdown the service after use. Users should avoid shutting down the
   *     service before {@link #shutdown()} has been called
   */
  public ByteStreamUploader(
      @Nullable String instanceName,
      Channel channel,
      @Nullable CallCredentials callCredentials,
      long callTimeoutSecs,
      Retrier retrier,
      ListeningScheduledExecutorService retryService) {
    checkArgument(callTimeoutSecs > 0, "callTimeoutSecs must be gt 0.");

    this.instanceName = instanceName;
    this.channel = channel;
    this.callCredentials = callCredentials;
    this.callTimeoutSecs = callTimeoutSecs;
    this.retrier = retrier;
    this.retryService = retryService;
  }

  /**
   * Uploads a BLOB, as provided by the {@link Chunker.SingleSourceBuilder}, to the remote {@code
   * ByteStream} service. The call blocks until the upload is complete, or throws an {@link
   * Exception} in case of error.
   *
   * <p>Uploads are retried according to the specified {@link Retrier}. Retrying is transparent to
   * the user of this API.
   *
   * <p>Trying to upload the same BLOB multiple times concurrently, results in only one upload being
   * performed. This is transparent to the user of this API.
   *
   * @throws IOException when reading of the {@link Chunker}s input source fails
   * @throws RetryException when the upload failed after a retry
   */
  public void uploadBlob(Chunker.SingleSourceBuilder chunkerBuilder)
      throws IOException, InterruptedException {
    uploadBlobs(singletonList(chunkerBuilder));
  }

  /**
   * Uploads a list of BLOBs concurrently to the remote {@code ByteStream} service. The call blocks
   * until the upload of all BLOBs is complete, or throws an {@link Exception} after the first
   * upload failed. Any other uploads will continue uploading in the background, until they complete
   * or the {@link #shutdown()} method is called. Errors encountered by these uploads are swallowed.
   *
   * <p>Uploads are retried according to the specified {@link Retrier}. Retrying is transparent to
   * the user of this API.
   *
   * <p>Trying to upload the same BLOB multiple times concurrently, results in only one upload being
   * performed. This is transparent to the user of this API.
   *
   * @throws IOException when reading of the {@link Chunker}s input source fails
   * @throws RetryException when the upload failed after a retry
   */
  public void uploadBlobs(Iterable<Chunker.SingleSourceBuilder> chunkerBuilders)
      throws IOException, InterruptedException {
    List<ListenableFuture<Void>> uploads = new ArrayList<>();

    for (Chunker.SingleSourceBuilder chunkerBuilder : chunkerBuilders) {
      uploads.add(uploadBlobAsync(chunkerBuilder));
    }

    try {
      for (ListenableFuture<Void> upload : uploads) {
        upload.get();
      }
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof RetryException) {
        throw (RetryException) cause;
      } else {
        throw Throwables.propagate(cause);
      }
    } catch (InterruptedException e) {
      Thread.interrupted();
      throw e;
    }
  }

  /**
   * Cancels all running uploads. The method returns immediately and does NOT wait for the uploads
   * to be cancelled.
   *
   * <p>This method must be the last method called.
   */
  public void shutdown() {
    synchronized (lock) {
      if (isShutdown) {
        return;
      }
      isShutdown = true;
      // Before cancelling, copy the futures to a separate list in order to avoid concurrently
      // iterating over and modifying the map (cancel triggers a listener that removes the entry
      // from the map. the listener is executed in the same thread.).
      List<Future<Void>> uploadsToCancel = new ArrayList<>(uploadsInProgress.values());
      for (Future<Void> upload : uploadsToCancel) {
        upload.cancel(true);
      }
    }
  }

  @VisibleForTesting
  ListenableFuture<Void> uploadBlobAsync(Chunker.SingleSourceBuilder chunkerBuilder)
      throws IOException {
    Digest digest = checkNotNull(chunkerBuilder.getDigest());

    synchronized (lock) {
      checkState(!isShutdown, "Must not call uploadBlobs after shutdown.");

      ListenableFuture<Void> uploadResult = uploadsInProgress.get(digest);
      if (uploadResult == null) {
        uploadResult = SettableFuture.create();
        uploadResult.addListener(
            () -> {
              synchronized (lock) {
                uploadsInProgress.remove(digest);
              }
            },
            MoreExecutors.directExecutor());
        startAsyncUploadWithRetry(
            chunkerBuilder, retrier.newBackoff(), (SettableFuture<Void>) uploadResult);
        uploadsInProgress.put(digest, uploadResult);
      }
      return uploadResult;
    }
  }

  @VisibleForTesting
  boolean uploadsInProgress() {
    synchronized (lock) {
      return !uploadsInProgress.isEmpty();
    }
  }

  private void startAsyncUploadWithRetry(
      Chunker.SingleSourceBuilder chunkerBuilder,
      Retrier.Backoff backoffTimes,
      SettableFuture<Void> overallUploadResult) {

    AsyncUpload.Listener listener =
        new AsyncUpload.Listener() {
          @Override
          public void success() {
            overallUploadResult.set(null);
          }

          @Override
          public void failure(Status status) {
            StatusException cause = status.asException();
            long nextDelayMillis = backoffTimes.nextDelayMillis();
            if (nextDelayMillis < 0 || !retrier.isRetriable(status)) {
              // Out of retries or status not retriable.
              RetryException error = new RetryException(cause, backoffTimes.getRetryAttempts());
              overallUploadResult.setException(error);
            } else {
              retryAsyncUpload(nextDelayMillis, chunkerBuilder, backoffTimes, overallUploadResult);
            }
          }

          private void retryAsyncUpload(
              long nextDelayMillis,
              Chunker.SingleSourceBuilder chunkerBuilder,
              Retrier.Backoff backoffTimes,
              SettableFuture<Void> overallUploadResult) {
            try {
              ListenableScheduledFuture<?> schedulingResult =
                  retryService.schedule(
                      () ->
                          startAsyncUploadWithRetry(
                              chunkerBuilder, backoffTimes, overallUploadResult),
                      nextDelayMillis,
                      MILLISECONDS);
              // In case the scheduled execution errors, we need to notify the overallUploadResult.
              schedulingResult.addListener(
                  () -> {
                    try {
                      schedulingResult.get();
                    } catch (Exception e) {
                      overallUploadResult.setException(
                          new RetryException(e, backoffTimes.getRetryAttempts()));
                    }
                  },
                  MoreExecutors.directExecutor());
            } catch (RejectedExecutionException e) {
              // May be thrown by .schedule(...) if i.e. the executor is shutdown.
              overallUploadResult.setException(
                  new RetryException(e, backoffTimes.getRetryAttempts()));
            }
          }
        };

    Chunker chunker;
    try {
      chunker = chunkerBuilder.build();
    } catch (IOException e) {
      overallUploadResult.setException(e);
      return;
    }

    AsyncUpload newUpload =
        new AsyncUpload(channel, callCredentials, callTimeoutSecs, instanceName, chunker, listener);
    overallUploadResult.addListener(
        () -> {
          if (overallUploadResult.isCancelled()) {
            newUpload.cancel();
          }
        },
        MoreExecutors.directExecutor());
    newUpload.start();
  }

  private static class AsyncUpload {

    interface Listener {
      void success();

      void failure(Status status);
    }

    private final Channel channel;
    private final CallCredentials callCredentials;
    private final long callTimeoutSecs;
    private final String instanceName;
    private final Chunker chunker;
    private final Listener listener;

    private ClientCall<WriteRequest, WriteResponse> call;

    AsyncUpload(
        Channel channel,
        CallCredentials callCredentials,
        long callTimeoutSecs,
        String instanceName,
        Chunker chunker,
        Listener listener) {
      this.channel = channel;
      this.callCredentials = callCredentials;
      this.callTimeoutSecs = callTimeoutSecs;
      this.instanceName = instanceName;
      this.chunker = chunker;
      this.listener = listener;
    }

    void start() {
      CallOptions callOptions =
          CallOptions.DEFAULT
              .withCallCredentials(callCredentials)
              .withDeadlineAfter(callTimeoutSecs, SECONDS);
      call = channel.newCall(ByteStreamGrpc.METHOD_WRITE, callOptions);

      ClientCall.Listener<WriteResponse> callListener =
          new ClientCall.Listener<WriteResponse>() {

            private final WriteRequest.Builder requestBuilder = WriteRequest.newBuilder();
            private boolean callHalfClosed = false;

            @Override
            public void onMessage(WriteResponse response) {
              // TODO(buchgr): The ByteStream API allows to resume the upload at the committedSize.
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              if (!status.isOk()) {
                listener.failure(status);
              } else {
                listener.success();
              }
            }

            @Override
            public void onReady() {
              while (call.isReady()) {
                if (!chunker.hasNext()) {
                  // call.halfClose() may only be called once. Guard against it being called more
                  // often.
                  // See: https://github.com/grpc/grpc-java/issues/3201
                  if (!callHalfClosed) {
                    callHalfClosed = true;
                    // Every chunk has been written. No more work to do.
                    call.halfClose();
                  }
                  return;
                }

                try {
                  requestBuilder.clear();
                  Chunker.Chunk chunk = chunker.next();

                  if (chunk.getOffset() == 0) {
                    // Resource name only needs to be set on the first write for each file.
                    requestBuilder.setResourceName(newResourceName(chunk.getDigest()));
                  }

                  boolean isLastChunk = !chunker.hasNext();
                  WriteRequest request =
                      requestBuilder
                          .setData(ByteString.copyFrom(chunk.getData()))
                          .setWriteOffset(chunk.getOffset())
                          .setFinishWrite(isLastChunk)
                          .build();

                  call.sendMessage(request);
                } catch (IOException e) {
                  call.cancel("Failed to read next chunk.", e);
                }
              }
            }

            private String newResourceName(Digest digest) {
              String resourceName =
                  format(
                      "uploads/%s/blobs/%s/%d",
                      UUID.randomUUID(), digest.getHash(), digest.getSizeBytes());
              if (!Strings.isNullOrEmpty(instanceName)) {
                resourceName = instanceName + "/" + resourceName;
              }
              return resourceName;
            }
          };
      call.start(callListener, new Metadata());
      call.request(1);
    }

    void cancel() {
      if (call != null) {
        call.cancel("Cancelled by user.", null);
      }
    }
  }
}
