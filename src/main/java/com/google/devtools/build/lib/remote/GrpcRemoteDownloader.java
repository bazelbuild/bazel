// Copyright 2019 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.Digest;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.Downloader;
import com.google.devtools.build.lib.bazel.repository.downloader.ProgressInputStream;
import com.google.devtools.build.lib.buildeventstream.FetchEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.repositorycache.DownloadRequest;
import com.google.devtools.build.lib.remote.repositorycache.DownloadResponse;
import com.google.devtools.build.lib.remote.repositorycache.RepositoryCacheGrpc;
import com.google.devtools.build.lib.remote.repositorycache.RepositoryCacheGrpc.RepositoryCacheBlockingStub;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;

import com.google.bytestream.ByteStreamGrpc;
import com.google.bytestream.ByteStreamGrpc.ByteStreamStub;
import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.hash.HashCode;
import com.google.common.hash.HashingOutputStream;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.remote.RemoteRetrier.ProgressiveBackoff;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import io.grpc.CallCredentials;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import javax.annotation.Nullable;

public class GrpcRemoteDownloader extends Downloader {

  private static final int MAX_PARALLEL_DOWNLOADS = 8;
  private static final Semaphore semaphore = new Semaphore(MAX_PARALLEL_DOWNLOADS, true);

  private final ReferenceCountedChannel channel;
  private final CallCredentials credentials;
  private final RemoteOptions options;
  private final RemoteRetrier retrier;
  private final DigestUtil digestUtil;

  private AtomicBoolean closed = new AtomicBoolean();

  @VisibleForTesting
  public GrpcRemoteDownloader(
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteOptions options,
      RemoteRetrier retrier,
      DigestUtil digestUtil) {
    this.channel = channel;
    this.credentials = credentials;
    this.options = options;
    this.retrier = retrier;
    this.digestUtil = digestUtil;
  }

  private RepositoryCacheBlockingStub cacheBlockingStub() {
    return RepositoryCacheGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withCallCredentials(credentials)
        .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
  }

  @Override
  public Path download(
      List<URL> urls,
      Map<URI, Map<String, String>> authHeaders,
      Optional<Checksum> checksum,
      String canonicalId,
      Optional<String> type,
      Path output,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      String repo)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }

    DownloadRequest.Builder requestBuilder = DownloadRequest.newBuilder()
        .setInstanceName(options.remoteInstanceName)
        .setCanonicalId(canonicalId);
    for (URL url : urls) {
      requestBuilder.addUrls(url.toString());
    }
    if (checksum.isPresent()) {
      requestBuilder.setIntegrity(checksum.get().toSubresourceIntegrity());
    }

    Clock clock = new JavaClock();
    Locale locale = Locale.getDefault();
    ProgressInputStream.Factory progressInputStreamFactory =
        new ProgressInputStream.Factory(locale, clock, eventHandler);

    URL mainUrl; // The "main" URL for this request
    // Used for reporting only and determining the file name only.
    if (urls.isEmpty()) {
      if (type.isPresent() && !Strings.isNullOrEmpty(type.get())) {
        mainUrl = new URL("http://nonexistent.example.org/cacheprobe." + type.get());
      } else {
        mainUrl = new URL("http://nonexistent.example.org/cacheprobe");
      }
    } else {
      mainUrl = urls.get(0);
    }
    Path destination = getDownloadDestination(mainUrl, type, output);

    Digest digest = getRemoteDigest(requestBuilder.build());

    // Download the file from the cache, while reporting progress to the CLI.
    semaphore.acquire();
    boolean success = false;
    try (OutputStream out = destination.getOutputStream()) {
      ListenableFuture<Void> future = (new CodeThatShouldBeSharedWithGrpcRemoteCache(
        channel, credentials, options, retrier, digestUtil)).downloadBlob(digest, out);
      future.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      Throwables.propagateIfInstanceOf(cause, IOException.class);
      throw new RuntimeException(cause);
    } finally {
      semaphore.release();
      eventHandler.post(new FetchEvent(urls.get(0).toString(), success));
    }

    return destination;
  }

  private Digest getRemoteDigest(DownloadRequest request)
      throws IOException, InterruptedException {
    DownloadResponse response;
    try {
      response = retrier.execute(() -> cacheBlockingStub().download(request));
    } catch (StatusRuntimeException e) {
      throw new IOException(e);
    }
    return response.getDigest();
  }

  @Override
  public void close() {
    if (closed.getAndSet(true)) {
      return;
    }
    channel.release();
  }

  private class CodeThatShouldBeSharedWithGrpcRemoteCache {
    private final CallCredentials credentials;
    private final ReferenceCountedChannel channel;
    private final RemoteOptions options;
    private final RemoteRetrier retrier;
    private final DigestUtil digestUtil;

    public CodeThatShouldBeSharedWithGrpcRemoteCache(
      ReferenceCountedChannel channel,
      CallCredentials credentials,
      RemoteOptions options,
      RemoteRetrier retrier,
      DigestUtil digestUtil
    ) {
      this.channel = channel;
      this.credentials = credentials;
      this.options = options;
      this.retrier = retrier;
      this.digestUtil = digestUtil;
    }

    private ByteStreamStub bsAsyncStub() {
      return ByteStreamGrpc.newStub(channel)
          .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
          .withCallCredentials(credentials)
          .withDeadlineAfter(options.remoteTimeout, TimeUnit.SECONDS);
    }

    public ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
      if (digest.getSizeBytes() == 0) {
        return Futures.immediateFuture(null);
      }
      String resourceName = "";
      if (!options.remoteInstanceName.isEmpty()) {
        resourceName += options.remoteInstanceName + "/";
      }
      resourceName += "blobs/" + digestUtil.toString(digest);

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
                    future.setException(new CacheNotFoundException(digest, digestUtil));
                  } else {
                    future.setException(t);
                  }
                }

                @Override
                public void onCompleted() {
                  try {
                    if (hashSupplier != null) {
                      verifyContents(
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

    protected void verifyContents(String expectedHash, String actualHash) throws IOException {
      if (!expectedHash.equals(actualHash)) {
        String msg =
            String.format(
                "An output download failed, because the expected hash"
                    + "'%s' did not match the received hash '%s'.",
                expectedHash, actualHash);
        throw new IOException(msg);
      }
    }

  }
}
