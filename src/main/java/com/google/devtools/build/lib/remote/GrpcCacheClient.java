// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Strings.isNullOrEmpty;

import build.bazel.remote.execution.v2.ActionCacheGrpc;
import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheBlockingStub;
import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheFutureStub;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageFutureStub;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import build.bazel.remote.execution.v2.GetActionResultRequest;
import build.bazel.remote.execution.v2.UpdateActionResultRequest;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamStub;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashingOutputStream;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.RemoteRetrier.ProgressiveBackoff;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.MissingDigestsFinder;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import io.grpc.CallCredentials;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** A RemoteActionCache implementation that uses gRPC calls to a remote cache server. */
@ThreadSafe
public class GrpcCacheClient implements RemoteCacheClient, MissingDigestsFinder {
  private final CallCredentials credentials;
  private final ReferenceCountedChannel channel;
  private final RemoteOptions options;
  private final DigestUtil digestUtil;
  private final RemoteRetrier retrier;
  private final ByteStreamUploader uploader;
  private final int maxMissingBlobsDigestsPerMessage;

  private AtomicBoolean closed = new AtomicBoolean();

  @VisibleForTesting
  public GrpcCacheClient(
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteOptions options,
      RemoteRetrier retrier,
      DigestUtil digestUtil,
      ByteStreamUploader uploader) {
    this.credentials = credentials;
    this.channel = channel;
    this.options = options;
    this.digestUtil = digestUtil;
    this.retrier = retrier;
    this.uploader = uploader;
    maxMissingBlobsDigestsPerMessage = computeMaxMissingBlobsDigestsPerMessage();
    Preconditions.checkState(
        maxMissingBlobsDigestsPerMessage > 0, "Error: gRPC message size too small.");
  }

  private int computeMaxMissingBlobsDigestsPerMessage() {
    final int overhead =
        FindMissingBlobsRequest.newBuilder()
            .setInstanceName(options.remoteInstanceName)
            .build()
            .getSerializedSize();
    final int tagSize =
        FindMissingBlobsRequest.newBuilder()
                .addBlobDigests(Digest.getDefaultInstance())
                .build()
                .getSerializedSize()
            - FindMissingBlobsRequest.getDefaultInstance().getSerializedSize();
    // We assume all non-empty digests have the same size. This is true for fixed-length hashes.
    final int digestSize = digestUtil.compute(new byte[] {1}).getSerializedSize() + tagSize;
    return (options.maxOutboundMessageSize - overhead) / digestSize;
  }

  private ContentAddressableStorageFutureStub casFutureStub() {
    return ContentAddressableStorageGrpc.newFutureStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withInterceptors(TracingMetadataUtils.newCacheHeadersInterceptor(options))
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ByteStreamStub bsAsyncStub() {
    return ByteStreamGrpc.newStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withInterceptors(TracingMetadataUtils.newCacheHeadersInterceptor(options))
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ActionCacheBlockingStub acBlockingStub() {
    return ActionCacheGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withInterceptors(TracingMetadataUtils.newCacheHeadersInterceptor(options))
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ActionCacheFutureStub acFutureStub() {
    return ActionCacheGrpc.newFutureStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withInterceptors(TracingMetadataUtils.newCacheHeadersInterceptor(options))
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  @Override
  public void close() {
    if (closed.getAndSet(true)) {
      return;
    }
    uploader.release();
    channel.release();
  }

  /** Returns true if 'options.remoteCache' uses 'grpc' or an empty scheme */
  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    if (isNullOrEmpty(options.remoteCache)) {
      return false;
    }
    // TODO(ishikhman): add proper URI validation/parsing for remote options
    return !(Ascii.toLowerCase(options.remoteCache).startsWith("http://")
        || Ascii.toLowerCase(options.remoteCache).startsWith("https://"));
  }

  @Override
  public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
    if (Iterables.isEmpty(digests)) {
      return Futures.immediateFuture(ImmutableSet.of());
    }
    // Need to potentially split the digests into multiple requests.
    FindMissingBlobsRequest.Builder requestBuilder =
        FindMissingBlobsRequest.newBuilder().setInstanceName(options.remoteInstanceName);
    List<ListenableFuture<FindMissingBlobsResponse>> getMissingDigestCalls = new ArrayList<>();
    for (Digest digest : digests) {
      requestBuilder.addBlobDigests(digest);
      if (requestBuilder.getBlobDigestsCount() == maxMissingBlobsDigestsPerMessage) {
        getMissingDigestCalls.add(getMissingDigests(requestBuilder.build()));
        requestBuilder.clearBlobDigests();
      }
    }

    if (requestBuilder.getBlobDigestsCount() > 0) {
      getMissingDigestCalls.add(getMissingDigests(requestBuilder.build()));
    }

    ListenableFuture<ImmutableSet<Digest>> success =
        Futures.whenAllSucceed(getMissingDigestCalls)
            .call(
                () -> {
                  ImmutableSet.Builder<Digest> result = ImmutableSet.builder();
                  for (ListenableFuture<FindMissingBlobsResponse> callFuture :
                      getMissingDigestCalls) {
                    result.addAll(callFuture.get().getMissingBlobDigestsList());
                  }
                  return result.build();
                },
                MoreExecutors.directExecutor());
    return Futures.catchingAsync(
        success,
        RuntimeException.class,
        (e) ->
            Futures.immediateFailedFuture(
                new IOException(
                    String.format(
                        "findMissingBlobs(%d) for %s",
                        requestBuilder.getBlobDigestsCount(),
                        TracingMetadataUtils.fromCurrentContext().getActionId()),
                    e)),
        MoreExecutors.directExecutor());
  }

  private ListenableFuture<FindMissingBlobsResponse> getMissingDigests(
      FindMissingBlobsRequest request) {
    Context ctx = Context.current();
    return retrier.executeAsync(() -> ctx.call(() -> casFutureStub().findMissingBlobs(request)));
  }

  private ListenableFuture<ActionResult> handleStatus(ListenableFuture<ActionResult> download) {
    return Futures.catchingAsync(
        download,
        StatusRuntimeException.class,
        (sre) ->
            sre.getStatus().getCode() == Code.NOT_FOUND
                // Return null to indicate that it was a cache miss.
                ? Futures.immediateFuture(null)
                : Futures.immediateFailedFuture(new IOException(sre)),
        MoreExecutors.directExecutor());
  }

  @Override
  public ListenableFuture<ActionResult> downloadActionResult(ActionKey actionKey) {
    GetActionResultRequest request =
        GetActionResultRequest.newBuilder()
            .setInstanceName(options.remoteInstanceName)
            .setActionDigest(actionKey.getDigest())
            .build();
    Context ctx = Context.current();
    return retrier.executeAsync(
        () -> ctx.call(() -> handleStatus(acFutureStub().getActionResult(request))));
  }

  private static String digestToString(Digest digest) {
    return digest.getHash() + "/" + digest.getSizeBytes();
  }

  @Override
  public void uploadActionResult(ActionKey actionKey, ActionResult actionResult)
      throws IOException, InterruptedException {
    try {
      retrier.execute(
          () ->
              acBlockingStub()
                  .updateActionResult(
                      UpdateActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(actionKey.getDigest())
                          .setActionResult(actionResult)
                          .build()));
    } catch (StatusRuntimeException e) {
      throw new IOException(e);
    }
  }

  @Override
  public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    if (digest.getSizeBytes() == 0) {
      return Futures.immediateFuture(null);
    }
    String resourceName = "";
    if (!options.remoteInstanceName.isEmpty()) {
      resourceName += options.remoteInstanceName + "/";
    }
    resourceName += "blobs/" + digestToString(digest);

    @Nullable Supplier<HashCode> hashSupplier = null;
    if (options.remoteVerifyDownloads) {
      HashingOutputStream hashOut = digestUtil.newHashingOutputStream(out);
      hashSupplier = hashOut::hash;
      out = hashOut;
    }

    SettableFuture<Void> outerF = SettableFuture.create();
    Futures.addCallback(
        downloadBlob(resourceName, digest, out, hashSupplier),
        new FutureCallback<Void>() {
          @Override
          public void onSuccess(Void result) {
            outerF.set(null);
          }

          @Override
          public void onFailure(Throwable t) {
            if (t instanceof StatusRuntimeException) {
              t = new IOException(t);
            }
            outerF.setException(t);
          }
        },
        Context.current().fixedContextExecutor(MoreExecutors.directExecutor()));
    return outerF;
  }

  private ListenableFuture<Void> downloadBlob(
      String resourceName,
      Digest digest,
      OutputStream out,
      @Nullable Supplier<HashCode> hashSupplier) {
    Context ctx = Context.current();
    AtomicLong offset = new AtomicLong(0);
    ProgressiveBackoff progressiveBackoff = new ProgressiveBackoff(retrier::newBackoff);
    return Futures.catchingAsync(
        retrier.executeAsync(
            () ->
                ctx.call(
                    () ->
                        requestRead(
                            resourceName, offset, progressiveBackoff, digest, out, hashSupplier)),
            progressiveBackoff),
        StatusRuntimeException.class,
        (e) -> Futures.immediateFailedFuture(new IOException(e)),
        MoreExecutors.directExecutor());
  }

  private ListenableFuture<Void> requestRead(
      String resourceName,
      AtomicLong offset,
      ProgressiveBackoff progressiveBackoff,
      Digest digest,
      OutputStream out,
      @Nullable Supplier<HashCode> hashSupplier) {
    SettableFuture<Void> future = SettableFuture.create();
    bsAsyncStub()
        .read(
            ReadRequest.newBuilder()
                .setResourceName(resourceName)
                .setReadOffset(offset.get())
                .build(),
            new StreamObserver<ReadResponse>() {
              @Override
              public void onNext(ReadResponse readResponse) {
                ByteString data = readResponse.getData();
                try {
                  data.writeTo(out);
                  offset.addAndGet(data.size());
                } catch (IOException e) {
                  future.setException(e);
                  // Cancel the call.
                  throw new RuntimeException(e);
                }
                // reset the stall backoff because we've made progress or been kept alive
                progressiveBackoff.reset();
              }

              @Override
              public void onError(Throwable t) {
                Status status = Status.fromThrowable(t);
                if (status.getCode() == Status.Code.NOT_FOUND) {
                  future.setException(new CacheNotFoundException(digest));
                } else {
                  future.setException(t);
                }
              }

              @Override
              public void onCompleted() {
                try {
                  if (hashSupplier != null) {
                    Utils.verifyBlobContents(
                        digest.getHash(), DigestUtil.hashCodeToString(hashSupplier.get()));
                  }
                  out.flush();
                  future.set(null);
                } catch (IOException e) {
                  future.setException(e);
                }
              }
            });
    return future;
  }

  @Override
  public ListenableFuture<Void> uploadFile(Digest digest, Path path) {
    return uploader.uploadBlobAsync(
        HashCode.fromString(digest.getHash()),
        Chunker.builder().setInput(digest.getSizeBytes(), path).build(),
        /* forceUpload= */ true);
  }

  @Override
  public ListenableFuture<Void> uploadBlob(Digest digest, ByteString data) {
    return uploader.uploadBlobAsync(
        HashCode.fromString(digest.getHash()),
        Chunker.builder().setInput(data.toByteArray()).build(),
        /* forceUpload= */ true);
  }
}
