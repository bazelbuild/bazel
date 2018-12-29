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
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.lang.String.format;
import static java.util.Collections.singletonList;
import static java.util.concurrent.TimeUnit.SECONDS;

import build.bazel.remote.execution.v2.Digest;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.remote.Retrier.RetryException;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import io.grpc.CallCredentials;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.Context;
import io.grpc.Metadata;
import io.grpc.Status;
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
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A client implementing the {@code Write} method of the {@code ByteStream} gRPC service.
 *
 * <p>The uploader supports reference counting to easily be shared between components with
 * different lifecyles. After instantiation the reference count is {@code 1}.
 *
 * See {@link ReferenceCounted} for more information on reference counting.
 */
class ByteStreamUploader extends AbstractReferenceCounted {

  private static final Logger logger = Logger.getLogger(ByteStreamUploader.class.getName());

  private final String instanceName;
  private final ReferenceCountedChannel channel;
  private final CallCredentials callCredentials;
  private final long callTimeoutSecs;
  private final RemoteRetrier retrier;

  private final Object lock = new Object();

  /** Contains the hash codes of already uploaded blobs. **/
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
   * @param callCredentials the credentials to use for authentication. May be {@code null}, in which
   *     case no authentication is performed
   * @param callTimeoutSecs the timeout in seconds after which a {@code Write} gRPC call must be
   *     complete. The timeout resets between retries
   * @param retrier the {@link RemoteRetrier} whose backoff strategy to use for retry timings.
   */
  public ByteStreamUploader(
      @Nullable String instanceName,
      ReferenceCountedChannel channel,
      @Nullable CallCredentials callCredentials,
      long callTimeoutSecs,
      RemoteRetrier retrier) {
    checkArgument(callTimeoutSecs > 0, "callTimeoutSecs must be gt 0.");

    this.instanceName = instanceName;
    this.channel = channel;
    this.callCredentials = callCredentials;
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
   * @param chunker the data to upload.
   * @param forceUpload if {@code false} the blob is not uploaded if it has previously been
   *        uploaded, if {@code true} the blob is uploaded.
   * @throws IOException when reading of the {@link Chunker}s input source fails
   * @throws RetryException when the upload failed after a retry
   */
  public void uploadBlob(Chunker chunker, boolean forceUpload) throws IOException,
      InterruptedException {
    uploadBlobs(singletonList(chunker), forceUpload);
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
   *        uploaded, if {@code true} the blob is uploaded.
   * @throws IOException when reading of the {@link Chunker}s input source fails
   * @throws RetryException when the upload failed after a retry
   */
  public void uploadBlobs(Iterable<Chunker> chunkers, boolean forceUpload) throws IOException,
      InterruptedException {
    List<ListenableFuture<Void>> uploads = new ArrayList<>();

    for (Chunker chunker : chunkers) {
      uploads.add(uploadBlobAsync(chunker, forceUpload));
    }

    try {
      for (ListenableFuture<Void> upload : uploads) {
        upload.get();
      }
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.propagateIfInstanceOf(cause, IOException.class);
      throw new RuntimeException(cause);
    }
  }

  /**
   * Cancels all running uploads. The method returns immediately and does NOT wait for the uploads
   * to be cancelled.
   *
   * <p>This method should not be called directly, but will be called implicitly when the
   * reference count reaches {@code 0}.
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
   * Uploads a BLOB asynchronously to the remote {@code ByteStream} service. The call returns
   * immediately and one can listen to the returned future for the success/failure of the upload.
   *
   * <p>Uploads are retried according to the specified {@link RemoteRetrier}. Retrying is
   * transparent to the user of this API.
   *
   * <p>Trying to upload the same BLOB multiple times concurrently, results in only one upload being
   * performed. This is transparent to the user of this API.
   *
   * @param chunker the data to upload.
   * @param forceUpload if {@code false} the blob is not uploaded if it has previously been
   *        uploaded, if {@code true} the blob is uploaded.
   * @throws IOException when reading of the {@link Chunker}s input source fails
   * @throws RetryException when the upload failed after a retry
   */
  public ListenableFuture<Void> uploadBlobAsync(Chunker chunker, boolean forceUpload) {
    Digest digest = checkNotNull(chunker.digest());
    HashCode hash = HashCode.fromString(digest.getHash());

    synchronized (lock) {
      checkState(!isShutdown, "Must not call uploadBlobs after shutdown.");

      if (!forceUpload && uploadedBlobs.contains(hash)) {
        return Futures.immediateFuture(null);
      }

      ListenableFuture<Void> inProgress = uploadsInProgress.get(digest);
      if (inProgress != null) {
        return inProgress;
      }

      final SettableFuture<Void> uploadResult = SettableFuture.create();
      uploadResult.addListener(
          () -> {
            synchronized (lock) {
              uploadsInProgress.remove(digest);
              uploadedBlobs.add(hash);
            }
          },
          MoreExecutors.directExecutor());
      uploadsInProgress.put(digest, uploadResult);
      Context ctx = Context.current();
      retrier.executeAsync(
          () -> ctx.call(() -> startAsyncUpload(chunker, uploadResult)), uploadResult);
      return uploadResult;
    }
  }

  @VisibleForTesting
  boolean uploadsInProgress() {
    synchronized (lock) {
      return !uploadsInProgress.isEmpty();
    }
  }

  /**
   * Starts a file upload an returns a future representing the upload. The {@code
   * overallUploadResult} future propagates cancellations from the caller to the upload.
   */
  private ListenableFuture<Void> startAsyncUpload(
      Chunker chunker, ListenableFuture<Void> overallUploadResult) {
    SettableFuture<Void> currUpload = SettableFuture.create();
    try {
      chunker.reset();
    } catch (IOException e) {
      currUpload.setException(e);
      return currUpload;
    }

    AsyncUpload newUpload =
        new AsyncUpload(
            channel, callCredentials, callTimeoutSecs, instanceName, chunker, currUpload);
    overallUploadResult.addListener(
        () -> {
          if (overallUploadResult.isCancelled()) {
            newUpload.cancel();
          }
        },
        MoreExecutors.directExecutor());
    newUpload.start();
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

    private final Channel channel;
    private final CallCredentials callCredentials;
    private final long callTimeoutSecs;
    private final String instanceName;
    private final Chunker chunker;
    private final SettableFuture<Void> uploadResult;

    private ClientCall<WriteRequest, WriteResponse> call;

    AsyncUpload(
        Channel channel,
        CallCredentials callCredentials,
        long callTimeoutSecs,
        String instanceName,
        Chunker chunker,
        SettableFuture<Void> uploadResult) {
      this.channel = channel;
      this.callCredentials = callCredentials;
      this.callTimeoutSecs = callTimeoutSecs;
      this.instanceName = instanceName;
      this.chunker = chunker;
      this.uploadResult = uploadResult;
    }

    void start() {
      CallOptions callOptions =
          CallOptions.DEFAULT
              .withCallCredentials(callCredentials)
              .withDeadlineAfter(callTimeoutSecs, SECONDS);
      call = channel.newCall(ByteStreamGrpc.getWriteMethod(), callOptions);

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
                    logger.log(Level.WARNING, "Chunker failed closing data source.", e1);
                  } finally {
                    call.cancel("Failed to read next chunk.", e);
                  }
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
      call.start(callListener, TracingMetadataUtils.headersFromCurrentContext());
      call.request(1);
    }

    void cancel() {
      if (call != null) {
        call.cancel("Cancelled by user.", null);
      }
    }
  }
}
