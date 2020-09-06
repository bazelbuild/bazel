// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.downloader;

import build.bazel.remote.asset.v1.FetchBlobRequest;
import build.bazel.remote.asset.v1.FetchBlobResponse;
import build.bazel.remote.asset.v1.FetchGrpc;
import build.bazel.remote.asset.v1.FetchGrpc.FetchBlockingStub;
import build.bazel.remote.asset.v1.Qualifier;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.Downloader;
import com.google.devtools.build.lib.bazel.repository.downloader.HashOutputStream;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.ReferenceCountedChannel;
import com.google.devtools.build.lib.remote.RemoteRetrier;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import io.grpc.CallCredentials;
import io.grpc.Context;
import io.grpc.StatusRuntimeException;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A Downloader implementation that uses Bazel's Remote Execution APIs to delegate downloads of
 * external files to a remote service.
 *
 * <p>See https://github.com/bazelbuild/remote-apis for more details on the exact capabilities and
 * semantics of the Remote Execution API.
 */
public class GrpcRemoteDownloader implements AutoCloseable, Downloader {

  private final ReferenceCountedChannel channel;
  private final Optional<CallCredentials> credentials;
  private final RemoteRetrier retrier;
  private final Context requestCtx;
  private final RemoteCacheClient cacheClient;
  private final RemoteOptions options;

  private final AtomicBoolean closed = new AtomicBoolean();

  // The `Qualifier::name` field uses well-known string keys to attach arbitrary
  // key-value metadata to download requests. These are the qualifier names
  // supported by Bazel.
  private static final String QUALIFIER_CHECKSUM_SRI = "checksum.sri";
  private static final String QUALIFIER_CANONICAL_ID = "bazel.canonical_id";
  private static final String QUALIFIER_AUTH_HEADERS = "bazel.auth_headers";

  public GrpcRemoteDownloader(
      ReferenceCountedChannel channel,
      Optional<CallCredentials> credentials,
      RemoteRetrier retrier,
      Context requestCtx,
      RemoteCacheClient cacheClient,
      RemoteOptions options) {
    this.channel = channel;
    this.credentials = credentials;
    this.retrier = retrier;
    this.cacheClient = cacheClient;
    this.requestCtx = requestCtx;
    this.options = options;
  }

  @Override
  public void close() {
    if (closed.getAndSet(true)) {
      return;
    }
    cacheClient.close();
    channel.release();
  }

  @Override
  public void download(
      List<URL> urls,
      Map<URI, Map<String, String>> authHeaders,
      com.google.common.base.Optional<Checksum> checksum,
      String canonicalId,
      Path destination,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv)
      throws IOException, InterruptedException {
    final FetchBlobRequest request =
        newFetchBlobRequest(options.remoteInstanceName, urls, authHeaders, checksum, canonicalId);
    try {
      FetchBlobResponse response =
          retrier.execute(() -> requestCtx.call(() -> fetchBlockingStub().fetchBlob(request)));
      final Digest blobDigest = response.getBlobDigest();

      retrier.execute(
          () ->
              requestCtx.call(
                  () -> {
                    try (OutputStream out = newOutputStream(destination, checksum)) {
                      Utils.getFromFuture(cacheClient.downloadBlob(blobDigest, out));
                    }
                    return null;
                  }));
    } catch (StatusRuntimeException e) {
      throw new IOException(e);
    }
  }

  @VisibleForTesting
  static FetchBlobRequest newFetchBlobRequest(
      String instanceName,
      List<URL> urls,
      Map<URI, Map<String, String>> authHeaders,
      com.google.common.base.Optional<Checksum> checksum,
      String canonicalId) {
    FetchBlobRequest.Builder requestBuilder =
        FetchBlobRequest.newBuilder().setInstanceName(instanceName);
    for (URL url : urls) {
      requestBuilder.addUris(url.toString());
    }
    if (checksum.isPresent()) {
      requestBuilder.addQualifiers(
          Qualifier.newBuilder()
              .setName(QUALIFIER_CHECKSUM_SRI)
              .setValue(checksum.get().toSubresourceIntegrity())
              .build());
    }
    if (!Strings.isNullOrEmpty(canonicalId)) {
      requestBuilder.addQualifiers(
          Qualifier.newBuilder().setName(QUALIFIER_CANONICAL_ID).setValue(canonicalId).build());
    }
    if (!authHeaders.isEmpty()) {
      requestBuilder.addQualifiers(
          Qualifier.newBuilder()
              .setName(QUALIFIER_AUTH_HEADERS)
              .setValue(authHeadersJson(authHeaders))
              .build());
    }

    return requestBuilder.build();
  }

  private FetchBlockingStub fetchBlockingStub() {
    return FetchGrpc.newBlockingStub(channel)
        .withInterceptors(TracingMetadataUtils.attachMetadataFromContextInterceptor())
        .withInterceptors(TracingMetadataUtils.newDownloaderHeadersInterceptor(options))
        .withCallCredentials(credentials.orElse(null))
        .withDeadlineAfter(options.remoteTimeout.getSeconds(), TimeUnit.SECONDS);
  }

  private OutputStream newOutputStream(
      Path destination, com.google.common.base.Optional<Checksum> checksum) throws IOException {
    OutputStream out = destination.getOutputStream();
    if (checksum.isPresent()) {
      out = new HashOutputStream(out, checksum.get());
    }
    return out;
  }

  private static String authHeadersJson(Map<URI, Map<String, String>> authHeaders) {
    Map<String, JsonObject> subObjects = new TreeMap<>();
    for (Map.Entry<URI, Map<String, String>> entry : authHeaders.entrySet()) {
      JsonObject subObject = new JsonObject();
      Map<String, String> orderedHeaders = new TreeMap<>(entry.getValue());
      for (Map.Entry<String, String> subEntry : orderedHeaders.entrySet()) {
        subObject.addProperty(subEntry.getKey(), subEntry.getValue());
      }
      subObjects.put(entry.getKey().toString(), subObject);
    }

    JsonObject authHeadersJson = new JsonObject();
    for (Map.Entry<String, JsonObject> entry : subObjects.entrySet()) {
      authHeadersJson.add(entry.getKey(), entry.getValue());
    }

    return new Gson().toJson(authHeadersJson);
  }
}
