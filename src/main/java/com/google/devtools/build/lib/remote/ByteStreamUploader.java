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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static java.lang.String.format;
import static java.util.Collections.singletonMap;
import static java.util.concurrent.TimeUnit.SECONDS;

import build.bazel.remote.execution.v2.Digest;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamFutureStub;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusRequest;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.remote.RemoteRetrier.ProgressiveBackoff;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import io.netty.util.AbstractReferenceCounted;
import io.netty.util.ReferenceCounted;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A client implementing the {@code Write} method of the {@code ByteStream} gRPC service.
 *
 * <p>The uploader supports reference counting to easily be shared between components with different
 * lifecyles. After instantiation the reference count is {@code 1}.
 *
 * <p>See {@link ReferenceCounted} for more information on reference counting.
 */
class ByteStreamUploader extends AbstractReferenceCounted {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final String instanceName;
  private final ReferenceCountedChannel channel;
  private final CallCredentialsProvider callCredentialsProvider;
  private final long callTimeoutSecs;
  private final RemoteRetrier retrier;

  private final Object lock = new Object();

  /** Contains the hash codes of already uploaded blobs. * */
  @GuardedBy("lock")
  private final Set<HashCode> uploadedBlobs = new HashSet<>();

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
   * @param callCredentialsProvider the credentials provider to use for authentication.
   * @param callTimeoutSecs the timeout in seconds after which a {@code Write} gRPC call must be
   *     complete. The timeout resets between retries
   * @param retrier the {@link RemoteRetrier} whose backoff strategy to use for retry timings.
   */
  public ByteStreamUploader(
      @Nullable String instanceName,
      ReferenceCountedChannel channel,
      CallCredentialsProvider callCredentialsProvider,
      long callTimeoutSecs,
      RemoteRetrier retrier) {
    checkArgument(callTimeoutSecs > 0, "callTimeoutSecs must be gt 0.");

    this.instanceName = instanceName;
    this.channel = channel;
    this.callCredentialsProvider = callCredentialsProvider;
    this.callTimeoutSecs = callTimeoutSecs;
    this.retrier = retrier;
  }

  /**
   * Uploads a BLOB, as provided by the {@link Chunker}, to the remote {@code ByteStream} service.
   * The call blocks until the upload is complete, or throws an {@link Exception} in case of error.
   *
   * <p>Uploads are retried according to the specified {@link RemoteRetrier}. Retrying is
   * transparent to the user of this API.
   *
   * <p>Trying to upload the same BLOB multiple times concurrently, results in only one upload being
   * performed. This is transparent to the user of this API.
   *
   * @param hash the hash of the data to upload.
   * @param chunker the data to upload.
   * @param forceUpload if {@code false} the blob is not uploaded if it has previously been
   *     uploaded, if {@code true} the blob is uploaded.
   * @throws IOException when reading of the {@link Chunker}s input source fails
   */
  public void uploadBlob(
      RemoteActionExecutionContext context, HashCode hash, Chunker chunker, boolean forceUpload)
      throws IOException, InterruptedException {
    uploadBlobs(context, singletonMap(hash, chunker), forceUpload);
  }

  /**
   * Uploads a list of BLOBs concurrently to the remote {@code ByteStream} service. The call blocks
   * until the upload of all BLOBs is complete, or throws an {@link Exception} after the first
   * upload failed. Any other uploads will continue uploading in the background, until they complete
   * or the {@link #shutdown()} method is called. Errors encountered by these uploads are swallowed.
   *
   * <p>Uploads are retried according to the specified {@link RemoteRetrier}. Retrying is
   * transparent to the user of this API.
   *
   * <p>Trying to upload the same BLOB multiple times concurrently, results in only one upload being
   * performed. This is transparent to the user of this API.
   *
   * @param chunkers the data to upload.
   * @param forceUpload if {@code false} the blob is not uploaded if it has previously been
   *     uploaded, if {@code true} the blob is uploaded.
   * @throws IOException when reading of the {@link Chunker}s input source or uploading fails
   */
  public void uploadBlobs(
      RemoteActionExecutionContext context, Map<HashCode, Chunker> chunkers, boolean forceUpload)
      throws IOException, InterruptedException {
    List<ListenableFuture<Void>> uploads = new ArrayList<>();

    for (Map.Entry<HashCode, Chunker> chunkerEntry : chunkers.entrySet()) {
      uploads.add(
          uploadBlobAsync(context, chunkerEntry.getKey(), chunkerEntry.getValue(), forceUpload));
    }

    try {
      for (ListenableFuture<Void> upload : uploads) {
        upload.get();
      }
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.propagateIfInstanceOf(cause, IOException.class);
      Throwables.propagateIfInstanceOf(cause, InterruptedException.class);
      Throwables.propagate(cause);
    }
  }

  /**
   * Cancels all running uploads. The method returns immediately and does NOT wait for the uploads
   * to be cancelled.
   *
   * <p>This method should not be called directly, but will be called implicitly when the reference
   * count reaches {@code 0}.
   */
  @VisibleForTesting
  void shutdown() {
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

  /**
   * @deprecated Use {@link #uploadBlobAsync(RemoteActionExecutionContext, Digest, Chunker,
   *     boolean)} instead.
   */
  @Deprecated
  public ListenableFuture<Void> uploadBlobAsync(
      RemoteActionExecutionContext context, HashCode hash, Chunker chunker, boolean forceUpload) {
    Digest digest =
        Digest.newBuilder().setHash(hash.toString()).setSizeBytes(chunker.getSize()).build();
    return uploadBlobAsync(context, digest, chunker, forceUpload);
  }

  /**
   * Uploads a BLOB asynchronously to the remote {@code ByteStream} service. The call returns
   * immediately and one can listen to the returned future for the success/failure of the upload.
   *
   * <p>Uploads are retried according to the specified {@link RemoteRetrier}. Retrying is
   * transparent to the user of this API.
   *
   * <p>Trying to upload the same BLOB multiple times concurrently, results in only one upload being
   * performed. This is transparent to the user of this API.
   *
   * @param digest the {@link Digest} of the data to upload.
   * @param chunker the data to upload.
   * @param forceUpload if {@code false} the blob is not uploaded if it has previously been
   *     uploaded, if {@code true} the blob is uploaded.
   * @throws IOException when reading of the {@link Chunker}s input source fails
   */
  public ListenableFuture<Void> uploadBlobAsync(
      RemoteActionExecutionContext context, Digest digest, Chunker chunker, boolean forceUpload) {
    synchronized (lock) {
      checkState(!isShutdown, "Must not call uploadBlobs after shutdown.");

      if (!forceUpload && uploadedBlobs.contains(HashCode.fromString(digest.getHash()))) {
        return Futures.immediateFuture(null);
      }

      ListenableFuture<Void> inProgress = uploadsInProgress.get(digest);
      if (inProgress != null) {
        return inProgress;
      }

      ListenableFuture<Void> uploadResult =
          Futures.transform(
              startAsyncUpload(context, digest, chunker),
              (v) -> {
                synchronized (lock) {
                  uploadedBlobs.add(HashCode.fromString(digest.getHash()));
                }
                return null;
              },
              MoreExecutors.directExecutor());

      uploadResult =
          Futures.catchingAsync(
              uploadResult,
              StatusRuntimeException.class,
              (sre) ->
                  Futures.immediateFailedFuture(
                      new IOException(
                          String.format(
                              "Error while uploading artifact with digest '%s/%s'",
                              digest.getHash(), digest.getSizeBytes()),
                          sre)),
              MoreExecutors.directExecutor());

      uploadsInProgress.put(digest, uploadResult);
      uploadResult.addListener(
          () -> {
            synchronized (lock) {
              uploadsInProgress.remove(digest);
            }
          },
          MoreExecutors.directExecutor());

      return uploadResult;
    }
  }

  @VisibleForTesting
  boolean uploadsInProgress() {
    synchronized (lock) {
      return !uploadsInProgress.isEmpty();
    }
  }

  private static String buildUploadResourceName(String instanceName, UUID uuid, Digest digest) {
    String resourceName =
        format("uploads/%s/blobs/%s/%d", uuid, digest.getHash(), digest.getSizeBytes());
    if (!Strings.isNullOrEmpty(instanceName)) {
      resourceName = instanceName + "/" + resourceName;
    }
    return resourceName;
  }

  /** Starts a file upload and returns a future representing the upload. */
  private ListenableFuture<Void> startAsyncUpload(
      RemoteActionExecutionContext context, Digest digest, Chunker chunker) {
    try {
      chunker.reset();
    } catch (IOException e) {
      return Futures.immediateFailedFuture(e);
    }

    if (chunker.getSize() != digest.getSizeBytes()) {
      return Futures.immediateFailedFuture(
          new IllegalStateException(
              String.format(
                  "Expected chunker size of %d, got %d",
                  digest.getSizeBytes(), chunker.getSize())));
    }

    UUID uploadId = UUID.randomUUID();
    String resourceName = buildUploadResourceName(instanceName, uploadId, digest);
    AsyncUpload newUpload =
        new AsyncUpload(
            context,
            channel,
            callCredentialsProvider,
            callTimeoutSecs,
            retrier,
            resourceName,
            chunker);
    ListenableFuture<Void> currUpload = newUpload.start();
    currUpload.addListener(
        () -> {
          if (currUpload.isCancelled()) {
            newUpload.cancel();
          }
        },
        MoreExecutors.directExecutor());
    return currUpload;
  }

  @Override
  public ByteStreamUploader retain() {
    return (ByteStreamUploader) super.retain();
  }

  @Override
  public ByteStreamUploader retain(int increment) {
    return (ByteStreamUploader) super.retain(increment);
  }

  @Override
  protected void deallocate() {
    shutdown();
    channel.release();
  }

  @Override
  public ReferenceCounted touch(Object o) {
    return this;
  }

  private static class AsyncUpload {

    private final RemoteActionExecutionContext context;
    private final Channel channel;
    private final CallCredentialsProvider callCredentialsProvider;
    private final long callTimeoutSecs;
    private final Retrier retrier;
    private final String resourceName;
    private final Chunker chunker;

    private ClientCall<WriteRequest, WriteResponse> call;

    AsyncUpload(
        RemoteActionExecutionContext context,
        Channel channel,
        CallCredentialsProvider callCredentialsProvider,
        long callTimeoutSecs,
        Retrier retrier,
        String resourceName,
        Chunker chunker) {
      this.context = context;
      this.channel = channel;
      this.callCredentialsProvider = callCredentialsProvider;
      this.callTimeoutSecs = callTimeoutSecs;
      this.retrier = retrier;
      this.resourceName = resourceName;
      this.chunker = chunker;
    }

    ListenableFuture<Void> start() {
      ProgressiveBackoff progressiveBackoff = new ProgressiveBackoff(retrier::newBackoff);
      AtomicLong committedOffset = new AtomicLong(0);

      ListenableFuture<Void> callFuture =
          Utils.refreshIfUnauthenticatedAsync(
              () ->
                  retrier.executeAsync(
                      () -> {
                        if (committedOffset.get() < chunker.getSize()) {
                          return callAndQueryOnFailure(committedOffset, progressiveBackoff);
                        }
                        return Futures.immediateFuture(null);
                      },
                      progressiveBackoff),
              callCredentialsProvider);

      return Futures.transformAsync(
          callFuture,
          (result) -> {
            long committedSize = committedOffset.get();
            long expected = chunker.getSize();
            if (committedSize != expected) {
              String message =
                  format(
                      "write incomplete: committed_size %d for %d total", committedSize, expected);
              return Futures.immediateFailedFuture(new IOException(message));
            }
            return Futures.immediateFuture(null);
          },
          MoreExecutors.directExecutor());
    }

    private ByteStreamFutureStub bsFutureStub() {
      return ByteStreamGrpc.newFutureStub(channel)
          .withInterceptors(
              TracingMetadataUtils.attachMetadataInterceptor(context.getRequestMetadata()))
          .withCallCredentials(callCredentialsProvider.getCallCredentials())
          .withDeadlineAfter(callTimeoutSecs, SECONDS);
    }

    private ListenableFuture<Void> callAndQueryOnFailure(
        AtomicLong committedOffset, ProgressiveBackoff progressiveBackoff) {
      return Futures.catchingAsync(
          call(committedOffset),
          Exception.class,
          (e) -> guardQueryWithSuppression(e, committedOffset, progressiveBackoff),
          MoreExecutors.directExecutor());
    }

    private ListenableFuture<Void> guardQueryWithSuppression(
        Exception e, AtomicLong committedOffset, ProgressiveBackoff progressiveBackoff) {
      // we are destined to return this, avoid recreating it
      ListenableFuture<Void> exceptionFuture = Futures.immediateFailedFuture(e);

      // TODO(buchgr): we should also return immediately without the query if
      // we were out of retry attempts for the underlying backoff. This
      // is meant to be an only in-between-retries query request.
      if (!retrier.isRetriable(e)) {
        return exceptionFuture;
      }

      ListenableFuture<Void> suppressedQueryFuture =
          Futures.catchingAsync(
              query(committedOffset, progressiveBackoff),
              Exception.class,
              (queryException) -> {
                // if the query threw an exception, add it to the suppressions
                // for the destined exception
                e.addSuppressed(queryException);
                return exceptionFuture;
              },
              MoreExecutors.directExecutor());
      return Futures.transformAsync(
          suppressedQueryFuture, (result) -> exceptionFuture, MoreExecutors.directExecutor());
    }

    private ListenableFuture<Void> query(
        AtomicLong committedOffset, ProgressiveBackoff progressiveBackoff) {
      ListenableFuture<Long> committedSizeFuture =
          Futures.transform(
              bsFutureStub()
                  .queryWriteStatus(
                      QueryWriteStatusRequest.newBuilder().setResourceName(resourceName).build()),
              (response) -> response.getCommittedSize(),
              MoreExecutors.directExecutor());
      ListenableFuture<Long> guardedCommittedSizeFuture =
          Futures.catchingAsync(
              committedSizeFuture,
              Exception.class,
              (e) -> {
                Status status = Status.fromThrowable(e);
                if (status.getCode() == Code.UNIMPLEMENTED) {
                  // if the bytestream server does not implement the query, insist
                  // that we should reset the upload
                  return Futures.immediateFuture(0L);
                }
                return Futures.immediateFailedFuture(e);
              },
              MoreExecutors.directExecutor());
      return Futures.transformAsync(
          guardedCommittedSizeFuture,
          (committedSize) -> {
            if (committedSize > committedOffset.get()) {
              // we have made progress on this upload in the last request,
              // reset the backoff so that this request has a full deck of retries
              progressiveBackoff.reset();
            }
            committedOffset.set(committedSize);
            return Futures.immediateFuture(null);
          },
          MoreExecutors.directExecutor());
    }

    private ListenableFuture<Void> call(AtomicLong committedOffset) {
      CallOptions callOptions =
          CallOptions.DEFAULT
              .withCallCredentials(callCredentialsProvider.getCallCredentials())
              .withDeadlineAfter(callTimeoutSecs, SECONDS);
      call = channel.newCall(ByteStreamGrpc.getWriteMethod(), callOptions);

      try {
        chunker.seek(committedOffset.get());
      } catch (IOException e) {
        try {
          chunker.reset();
        } catch (IOException resetException) {
          e.addSuppressed(resetException);
        }
        return Futures.immediateFailedFuture(e);
      }

      SettableFuture<Void> uploadResult = SettableFuture.create();
      ClientCall.Listener<WriteResponse> callListener =
          new ClientCall.Listener<WriteResponse>() {

            private final WriteRequest.Builder requestBuilder = WriteRequest.newBuilder();
            private boolean callHalfClosed = false;

            void halfClose() {
              // call.halfClose() may only be called once. Guard against it being called more
              // often.
              // See: https://github.com/grpc/grpc-java/issues/3201
              if (!callHalfClosed) {
                callHalfClosed = true;
                // Every chunk has been written. No more work to do.
                call.halfClose();
              }
            }

            @Override
            public void onMessage(WriteResponse response) {
              // upload was completed either by us or someone else
              committedOffset.set(response.getCommittedSize());
              halfClose();
            }

            @Override
            public void onClose(Status status, Metadata trailers) {
              if (status.isOk()) {
                uploadResult.set(null);
              } else {
                uploadResult.setException(status.asRuntimeException());
              }
            }

            @Override
            public void onReady() {
              while (call.isReady()) {
                if (!chunker.hasNext()) {
                  halfClose();
                  return;
                }

                if (callHalfClosed) {
                  return;
                }

                try {
                  requestBuilder.clear();
                  Chunker.Chunk chunk = chunker.next();

                  if (chunk.getOffset() == committedOffset.get()) {
                    // Resource name only needs to be set on the first write for each file.
                    requestBuilder.setResourceName(resourceName);
                  }

                  boolean isLastChunk = !chunker.hasNext();
                  WriteRequest request =
                      requestBuilder
                          .setData(chunk.getData())
                          .setWriteOffset(chunk.getOffset())
                          .setFinishWrite(isLastChunk)
                          .build();

                  call.sendMessage(request);
                } catch (IOException e) {
                  try {
                    chunker.reset();
                  } catch (IOException e1) {
                    // This exception indicates that closing the underlying input stream failed.
                    // We don't expect this to ever happen, but don't want to swallow the exception
                    // completely.
                    logger.atWarning().withCause(e1).log("Chunker failed closing data source.");
                  } finally {
                    call.cancel("Failed to read next chunk.", e);
                  }
                }
              }
            }
          };
      call.start(
          callListener,
          TracingMetadataUtils.headersFromRequestMetadata(context.getRequestMetadata()));
      call.request(1);
      return uploadResult;
    }

    void cancel() {
      if (call != null) {
        call.cancel("Cancelled by user.", null);
      }
    }
  }
}
