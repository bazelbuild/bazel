// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote.worker;

import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.asset.v1.FetchBlobRequest;
import build.bazel.remote.asset.v1.FetchBlobResponse;
import build.bazel.remote.asset.v1.FetchDirectoryRequest;
import build.bazel.remote.asset.v1.FetchDirectoryResponse;
import build.bazel.remote.asset.v1.FetchGrpc.FetchImplBase;
import build.bazel.remote.asset.v1.Qualifier;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.HashOutputStream;
import com.google.devtools.build.lib.bazel.repository.downloader.UnrecoverableHttpException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestOutputStream;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.rpc.Code;
import io.grpc.StatusException;
import io.grpc.stub.StreamObserver;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URI;
import java.net.URISyntaxException;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Optional;
import java.util.SequencedMap;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** A basic implementation of a {@link FetchImplBase} service. */
final class FetchServer extends FetchImplBase {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final String QUALIFIER_CANONICAL_ID = "bazel.canonical_id";
  private static final String QUALIFIER_CHECKSUM_SRI = "checksum.sri";
  private static final String QUALIFIER_HTTP_HEADER_PREFIX = "http_header:";
  private static final String QUALIFIER_HTTP_HEADER_URL_PREFIX = "http_header_url:";

  private final OnDiskBlobStoreCache cache;
  private final DigestUtil digestUtil;
  private final Path tempPath;
  private final ConcurrentHashMap<CacheKey, CacheValue> knownUrls = new ConcurrentHashMap<>();

  private record CacheKey(String url, @Nullable String canonicalId) {}

  private record CacheValue(Digest digest, Instant downloadedAt) {}

  private record Qualifiers(
      @Nullable Checksum expectedChecksum,
      ImmutableMap<String, String> globalHeaders,
      ImmutableTable<Integer, String, String> urlSpecificHeaders,
      @Nullable String canonicalId) {}

  @Override
  public void fetchBlob(
      FetchBlobRequest request, StreamObserver<FetchBlobResponse> responseObserver) {
    if (request.getUrisCount() == 0) {
      responseObserver.onError(
          StatusUtils.invalidArgumentError("uris", "at least one URI must be provided"));
      return;
    }

    Qualifiers qualifiers;
    try {
      qualifiers = parseQualifiers(request.getQualifiersList());
    } catch (StatusException e) {
      responseObserver.onError(e);
      return;
    }

    Instant cutoff =
        request.hasOldestContentAccepted()
            ? Instant.now()
                .minus(Duration.ofSeconds(request.getOldestContentAccepted().getSeconds()))
            : Instant.MIN;
    Optional<Digest> cacheHit = checkCache(request.getUrisList(), qualifiers.canonicalId(), cutoff);
    if (cacheHit.isPresent()) {
      responseObserver.onNext(
          FetchBlobResponse.newBuilder()
              .setStatus(com.google.rpc.Status.newBuilder().setCode(Code.OK_VALUE).build())
              .setUri(request.getUris(0))
              .setBlobDigest(cacheHit.get())
              .setDigestFunction(digestUtil.getDigestFunction())
              .build());
      responseObserver.onCompleted();
      return;
    }

    Path tempDownloadDir;
    try {
      tempPath.createDirectoryAndParents();
      tempDownloadDir = tempPath.createTempDirectory("download-");
    } catch (IOException e) {
      responseObserver.onError(StatusUtils.internalError(e));
      return;
    }
    try {
      DownloadResult result = tryDownload(request, qualifiers, tempDownloadDir);

      RequestMetadata requestMetadata = TracingMetadataUtils.fromCurrentContext();
      RemoteActionExecutionContext context = RemoteActionExecutionContext.create(requestMetadata);
      getFromFuture(cache.uploadFile(context, result.digest(), result.path()));
      addToCache(result.uri(), qualifiers.canonicalId(), result.digest());

      responseObserver.onNext(
          FetchBlobResponse.newBuilder()
              .setStatus(com.google.rpc.Status.newBuilder().setCode(Code.OK_VALUE).build())
              .setUri(result.uri())
              .setBlobDigest(result.digest())
              .setDigestFunction(digestUtil.getDigestFunction())
              .build());
      responseObserver.onCompleted();
    } catch (IOException e) {
      responseObserver.onNext(
          FetchBlobResponse.newBuilder()
              .setStatus(
                  com.google.rpc.Status.newBuilder()
                      .setCode(determineCode(e).getNumber())
                      .setMessage("Failed to fetch from any URI: " + e.getMessage())
                      .build())
              .setUri(request.getUris(0))
              .build());
      responseObserver.onCompleted();
    } catch (Exception e) {
      if (e instanceof InterruptedException) {
        Thread.currentThread().interrupt();
      }
      logger.atWarning().withCause(e).log("Failed to upload blob to CAS");
      responseObserver.onError(StatusUtils.internalError(e));
    } finally {
      try {
        tempDownloadDir.deleteTree();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log(
            "Failed to delete temporary download directory %s", tempDownloadDir);
      }
    }
  }

  @Override
  public void fetchDirectory(
      FetchDirectoryRequest request, StreamObserver<FetchDirectoryResponse> responseObserver) {
    // FetchDirectory is not used by Bazel's GrpcRemoteDownloader client.
    responseObserver.onError(
        io.grpc.Status.UNIMPLEMENTED
            .withDescription("FetchDirectory is not implemented")
            .asRuntimeException());
  }

  public FetchServer(OnDiskBlobStoreCache cache, DigestUtil digestUtil, Path tempPath) {
    this.cache = cache;
    this.digestUtil = digestUtil;
    this.tempPath = tempPath;
  }

  private static Qualifiers parseQualifiers(Iterable<Qualifier> qualifiersList)
      throws StatusException {
    Checksum expectedChecksum = null;
    var globalHeaders = ImmutableMap.<String, String>builder();
    var urlSpecificHeaders = ImmutableTable.<Integer, String, String>builder();
    String canonicalId = null;

    for (var qualifier : qualifiersList) {
      String name = qualifier.getName();
      String value = qualifier.getValue();

      if (name.equals(QUALIFIER_CANONICAL_ID)) {
        canonicalId = value;
      } else if (name.equals(QUALIFIER_CHECKSUM_SRI)) {
        try {
          expectedChecksum = Checksum.fromSubresourceIntegrity(value);
        } catch (Checksum.InvalidChecksumException e) {
          throw StatusUtils.invalidArgumentError(
              "qualifiers",
              "invalid '%s' qualifier: %s".formatted(QUALIFIER_CHECKSUM_SRI, e.getMessage()));
        }
      } else if (name.startsWith(QUALIFIER_HTTP_HEADER_URL_PREFIX)) {
        // Format: http_header_url:<url_index>:<header_name>
        String remainder = name.substring(QUALIFIER_HTTP_HEADER_URL_PREFIX.length());
        int colonIndex = remainder.indexOf(':');
        if (colonIndex > 0) {
          try {
            int urlIndex = Integer.parseInt(remainder.substring(0, colonIndex));
            String headerName = remainder.substring(colonIndex + 1);
            urlSpecificHeaders.put(urlIndex, headerName, value);
          } catch (NumberFormatException e) {
            throw StatusUtils.invalidArgumentError(
                "qualifiers",
                "invalid '%s' qualifier: %s"
                    .formatted(QUALIFIER_HTTP_HEADER_URL_PREFIX, e.getMessage()));
          }
        }
      } else if (name.startsWith(QUALIFIER_HTTP_HEADER_PREFIX)) {
        String headerName = name.substring(QUALIFIER_HTTP_HEADER_PREFIX.length());
        globalHeaders.put(headerName, value);
      } else {
        throw StatusUtils.invalidArgumentError(
            "qualifiers", "unknown qualifier: '%s'".formatted(name));
      }
    }

    return new Qualifiers(
        expectedChecksum,
        globalHeaders.buildOrThrow(),
        urlSpecificHeaders.buildOrThrow(),
        canonicalId);
  }

  private Optional<Digest> checkCache(
      Iterable<String> uris, @Nullable String canonicalId, Instant cutoff) {
    for (var uri : uris) {
      var cacheValue = knownUrls.get(new CacheKey(uri, canonicalId));
      if (cacheValue != null && cacheValue.downloadedAt.isAfter(cutoff)) {
        return Optional.of(cacheValue.digest);
      }
    }
    return Optional.empty();
  }

  private void addToCache(String uri, @Nullable String canonicalId, Digest digest) {
    knownUrls.put(new CacheKey(uri, canonicalId), new CacheValue(digest, Instant.now()));
  }

  private record DownloadResult(String uri, Path path, Digest digest) {}

  private DownloadResult tryDownload(
      FetchBlobRequest request, Qualifiers qualifiers, Path tempDownloadDir) throws IOException {
    IOException lastException = null;

    for (int i = 0; i < request.getUrisCount(); i++) {
      String uri = request.getUris(i);
      Path downloadPath = tempDownloadDir.getChild("attempt_" + i);
      try {
        var out = downloadPath.getOutputStream();
        var digestOut =
            new DigestOutputStream(
                downloadPath.getFileSystem().getDigestFunction().getHashFunction(), out);
        var maybeChecksumOut =
            qualifiers.expectedChecksum() != null
                ? new HashOutputStream(digestOut, qualifiers.expectedChecksum())
                : digestOut;
        try (maybeChecksumOut) {
          var headers = new LinkedHashMap<>(qualifiers.globalHeaders());
          headers.putAll(qualifiers.urlSpecificHeaders().row(i));
          fetchFromUrl(
              uri,
              headers,
              Duration.ofSeconds(request.getTimeout().getSeconds()),
              maybeChecksumOut);
          return new DownloadResult(uri, downloadPath, digestOut.digest());
        }
      } catch (IOException e) {
        try {
          downloadPath.delete();
        } catch (IOException ex) {
          logger.atWarning().withCause(ex).log(
              "Failed to delete partially downloaded file %s", downloadPath);
        }
        lastException = e;
        logger.atFine().withCause(e).log("Failed to fetch from %s", uri);
      }
    }

    throw lastException != null ? lastException : new IOException("No URIs to fetch");
  }

  private Code determineCode(@Nullable IOException lastException) {
    return switch (lastException) {
      case SocketTimeoutException e -> Code.DEADLINE_EXCEEDED;
      case FileNotFoundException e -> Code.NOT_FOUND;
      // See HashOutputStream#verifyHash.
      case UnrecoverableHttpException e when e.getMessage().startsWith("Checksum was ") ->
          Code.ABORTED;
      case null, default -> Code.UNKNOWN;
    };
  }

  private void fetchFromUrl(
      String urlString, SequencedMap<String, String> headers, Duration timeout, OutputStream out)
      throws IOException {
    HttpURLConnection connection;
    try {
      connection = (HttpURLConnection) new URI(urlString).toURL().openConnection();
    } catch (URISyntaxException e) {
      throw new IOException("Invalid URI: " + urlString, e);
    }
    var timeoutMillis = timeout.equals(Duration.ZERO) ? 30000 : (int) timeout.toMillis();
    try {
      connection.setRequestMethod("GET");
      connection.setConnectTimeout(timeoutMillis);
      connection.setReadTimeout(timeoutMillis);
      headers.forEach(connection::setRequestProperty);

      int responseCode = connection.getResponseCode();
      if (responseCode != HttpURLConnection.HTTP_OK) {
        throw new IOException("HTTP request failed with status " + responseCode);
      }

      try (var in = connection.getInputStream()) {
        in.transferTo(out);
      }
    } finally {
      connection.disconnect();
    }
  }
}
