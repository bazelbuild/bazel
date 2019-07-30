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

import static java.lang.String.format;

import build.bazel.remote.execution.v2.ActionCacheGrpc;
import build.bazel.remote.execution.v2.ActionCacheGrpc.ActionCacheBlockingStub;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc.ContentAddressableStorageFutureStub;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import build.bazel.remote.execution.v2.GetActionResultRequest;
import build.bazel.remote.execution.v2.UpdateActionResultRequest;
import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamStub;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashingOutputStream;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.remote.RemoteRetrier.ProgressiveBackoff;
import com.google.devtools.build.lib.remote.common.Chunker;
import com.google.devtools.build.lib.remote.common.SimpleBlobStore;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import io.grpc.CallCredentials;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/** A {@link SimpleBlobStore} implementation that uses gRPC calls to a remote cache server. */
public class GrpcBlobStore implements SimpleBlobStore {
  private final CallCredentials credentials;
  private final ReferenceCountedChannel channel;
  private final RemoteRetrier retrier;
  private final ByteStreamUploader uploader;
  private final int maxMissingBlobsDigestsPerMessage;
  private AtomicBoolean closed = new AtomicBoolean();
  private RemoteOptions options;
  private DigestUtil digestUtil;

  public GrpcBlobStore(
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteOptions options,
      RemoteRetrier retrier,
      DigestUtil digestUtil,
      ByteStreamUploader uploader) {
    this.credentials = credentials;
    this.channel = channel;
    this.retrier = retrier;
    this.uploader = uploader;
    this.options = options;
    this.digestUtil = digestUtil;

    maxMissingBlobsDigestsPerMessage = computeMaxMissingBlobsDigestsPerMessage();
    Preconditions.checkState(
        maxMissingBlobsDigestsPerMessage > 0, "Error: gRPC message size too small.");
  }

  @Override
  public boolean contains(String key) throws IOException, InterruptedException {
    throw new UnsupportedOperationException("gRPC Caching does not use this method.");
  }

  @Override
  public boolean containsActionResult(String key) throws IOException, InterruptedException {
    throw new UnsupportedOperationException("gRPC Caching does not use this method.");
  }

  @Override
  public ListenableFuture<Boolean> get(String key, Digest digest, OutputStream out) {
    String resourceName = "";
    if (!options.remoteInstanceName.isEmpty()) {
      resourceName += options.remoteInstanceName + "/";
    }
    resourceName += "blobs/" + digestUtil.toString(digest);

    return downloadBlob(resourceName, digest, out);
  }

  @Override
  public ListenableFuture<Boolean> getActionResult(Digest digest, OutputStream out) {
    SettableFuture<Boolean> f = SettableFuture.create();
    try {
      ActionResult actionResult = retrier.execute(
          () ->
              acBlockingStub()
                  .getActionResult(
                      GetActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(digest)
                          .build()));
      try {
        actionResult.writeTo(out);
        f.set(true);
      } catch (IOException e) {
        f.setException(e);
      }
    } catch (IOException|InterruptedException e) {
      f.setException(e);
    } catch (StatusRuntimeException e) {
      if (e.getStatus().getCode() == Status.Code.NOT_FOUND) {
        // Set false to indicate that it was a cache miss.
        f.set(false);
      }
      f.setException(new IOException(e));
    }
    return f;
  }

  @Override
  public void put(String key, Digest digest, long length, Chunker chunker, InputStream in) throws IOException, InterruptedException {
    ImmutableSet<Digest> missing = getMissingDigests(ImmutableList.of(digest));
    if (missing.isEmpty()) {
      return;
    }
    if (chunker == null) {
      String message = "call returned an unknown digest: " + digest;
      throw new IOException(message);
    }
    uploader.uploadBlob(HashCode.fromString(digest.getHash()), chunker, /*forceUpload=*/true);
  }

  @Override
  public void putActionResult(Digest digest, ActionResult actionResult) throws IOException, InterruptedException {
    try {
      retrier.execute(
          () ->
              acBlockingStub()
                  .updateActionResult(
                      UpdateActionResultRequest.newBuilder()
                          .setInstanceName(options.remoteInstanceName)
                          .setActionDigest(digest)
                          .setActionResult(actionResult)
                          .build()));
    } catch (StatusRuntimeException e) {
      throw new IOException(e);
    }
  }

  @Override
  public void close() {
    if (closed.getAndSet(true)) {
      return;
    }
    uploader.release();
    channel.release();
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

  private ActionCacheBlockingStub acBlockingStub() {
    return ActionCacheGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ByteStreamStub bsAsyncStub() {
    return ByteStreamGrpc.newStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ContentAddressableStorageFutureStub casFutureStub() {
    return ContentAddressableStorageGrpc.newFutureStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  private ListenableFuture<Boolean> downloadBlob(
      String resourceName,
      Digest digest,
      OutputStream out) {
    Context ctx = Context.current();
    AtomicLong offset = new AtomicLong(0);
    ProgressiveBackoff progressiveBackoff = new ProgressiveBackoff(retrier::newBackoff);
    return Futures.catchingAsync(
        retrier.executeAsync(
            () ->
                ctx.call(
                    () ->
                        requestRead(
                            resourceName, offset, progressiveBackoff, digest, out)),
            progressiveBackoff),
        StatusRuntimeException.class,
        (e) -> Futures.immediateFailedFuture(new IOException(e)),
        MoreExecutors.directExecutor());
  }

  private ListenableFuture<Boolean> requestRead(
      String resourceName,
      AtomicLong offset,
      ProgressiveBackoff progressiveBackoff,
      Digest digest,
      OutputStream out) {
    SettableFuture<Boolean> future = SettableFuture.create();
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
                  future.setException(new CacheNotFoundException(digest, digestUtil));
                } else {
                  future.setException(t);
                }
              }

              @Override
              public void onCompleted() {
                try {
                  out.flush();
                  future.set(true);
                } catch (IOException e) {
                  future.setException(e);
                }
              }
            });
    return future;
  }

  private ListenableFuture<FindMissingBlobsResponse> getMissingDigests(
      FindMissingBlobsRequest request) throws IOException, InterruptedException {
    Context ctx = Context.current();
    try {
      return retrier.executeAsync(() -> ctx.call(() -> casFutureStub().findMissingBlobs(request)));
    } catch (StatusRuntimeException e) {
      throw new IOException(e);
    }
  }

  @Override
  public ImmutableSet<Digest> getMissingDigests(Iterable<Digest> digests)
      throws IOException, InterruptedException {
    if (Iterables.isEmpty(digests)) {
      return ImmutableSet.of();
    }
    // Need to potentially split the digests into multiple requests.
    FindMissingBlobsRequest.Builder requestBuilder =
        FindMissingBlobsRequest.newBuilder().setInstanceName(options.remoteInstanceName);
    List<ListenableFuture<FindMissingBlobsResponse>> callFutures = new ArrayList<>();
    for (Digest digest : digests) {
      requestBuilder.addBlobDigests(digest);
      if (requestBuilder.getBlobDigestsCount() == maxMissingBlobsDigestsPerMessage) {
        callFutures.add(getMissingDigests(requestBuilder.build()));
        requestBuilder.clearBlobDigests();
      }
    }
    if (requestBuilder.getBlobDigestsCount() > 0) {
      callFutures.add(getMissingDigests(requestBuilder.build()));
    }
    ImmutableSet.Builder<Digest> result = ImmutableSet.builder();
    try {
      for (ListenableFuture<FindMissingBlobsResponse> callFuture : callFutures) {
        result.addAll(callFuture.get().getMissingBlobDigestsList());
      }
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.propagateIfInstanceOf(cause, IOException.class);
      throw new RuntimeException(cause);
    }
    return result.build();
  }

  @Override
  public void ensureInputsPresent(
      MerkleTree merkleTree, Map<Digest, Message> additionalInputs, Path execRoot)
      throws IOException, InterruptedException {
    ImmutableSet<Digest> missingDigests =
        getMissingDigests(Iterables.concat(merkleTree.getAllDigests(), additionalInputs.keySet()));
    Map<HashCode, Chunker> inputsToUpload = Maps.newHashMapWithExpectedSize(missingDigests.size());
    for (Digest missingDigest : missingDigests) {
      Directory node = merkleTree.getDirectoryByDigest(missingDigest);
      HashCode hash = HashCode.fromString(missingDigest.getHash());
      if (node != null) {
        Chunker c = Chunker.builder().setInput(node.toByteArray()).build();
        inputsToUpload.put(hash, c);
        continue;
      }

      ActionInput file = merkleTree.getInputByDigest(missingDigest);
      if (file != null) {
        Chunker c =
            Chunker.builder().setInput(missingDigest.getSizeBytes(), file, execRoot).build();
        inputsToUpload.put(hash, c);
        continue;
      }

      Message message = additionalInputs.get(missingDigest);
      if (message != null) {
        Chunker c = Chunker.builder().setInput(message.toByteArray()).build();
        inputsToUpload.put(hash, c);
        continue;
      }

      throw new IOException(
          format(
              "getMissingDigests returned a missing digest that has not been requested: %s",
              missingDigest));
    }

    uploader.uploadBlobs(inputsToUpload, /* forceUpload= */ true);
  }
}
