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

package com.google.devtools.build.lib.remote.downloader;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import build.bazel.remote.asset.v1.FetchBlobRequest;
import build.bazel.remote.asset.v1.FetchBlobResponse;
import build.bazel.remote.asset.v1.FetchGrpc.FetchImplBase;
import build.bazel.remote.asset.v1.Qualifier;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.bazel.repository.downloader.UnrecoverableHttpException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.ReferenceCountedChannel;
import com.google.devtools.build.lib.remote.RemoteRetrier;
import com.google.devtools.build.lib.remote.RemoteRetrier.ExponentialBackoff;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.CallCredentials;
import io.grpc.Context;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GrpcRemoteDownloader}. */
@RunWith(JUnit4.class)
public class GrpcRemoteDownloaderTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private final String fakeServerName = "fake server for " + getClass();
  private Server fakeServer;
  private Context withEmptyMetadata;
  private Context prevContext;
  private ListeningScheduledExecutorService retryService;

  @Before
  public final void setUp() throws Exception {
    // Use a mutable service registry for later registering the service impl for each test case.
    fakeServer =
        InProcessServerBuilder.forName(fakeServerName)
            .fallbackHandlerRegistry(serviceRegistry)
            .directExecutor()
            .build()
            .start();
    withEmptyMetadata =
        TracingMetadataUtils.contextWithMetadata(
            "none", "none", DIGEST_UTIL.asActionKey(Digest.getDefaultInstance()));

    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));

    prevContext = withEmptyMetadata.attach();
  }

  @After
  public void tearDown() throws Exception {
    withEmptyMetadata.detach(prevContext);

    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);

    fakeServer.shutdownNow();
    fakeServer.awaitTermination();
  }

  private GrpcRemoteDownloader newDownloader(RemoteCacheClient cacheClient) throws IOException {
    final RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    final RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new ExponentialBackoff(remoteOptions),
            RemoteRetrier.RETRIABLE_GRPC_ERRORS,
            retryService);
    final ReferenceCountedChannel channel =
        new ReferenceCountedChannel(
            InProcessChannelBuilder.forName(fakeServerName).directExecutor().build());
    return new GrpcRemoteDownloader(
        channel.retain(),
        Optional.<CallCredentials>empty(),
        retrier,
        withEmptyMetadata,
        cacheClient,
        remoteOptions);
  }

  private static byte[] downloadBlob(
      GrpcRemoteDownloader downloader, URL url, Optional<Checksum> checksum)
      throws IOException, InterruptedException {
    final List<URL> urls = ImmutableList.of(url);
    com.google.common.base.Optional<Checksum> guavaChecksum =
        com.google.common.base.Optional.<Checksum>absent();
    if (checksum.isPresent()) {
      guavaChecksum = com.google.common.base.Optional.<Checksum>of(checksum.get());
    }

    final Map<URI, Map<String, String>> authHeaders = ImmutableMap.of();
    final String canonicalId = "";
    final ExtendedEventHandler eventHandler = mock(ExtendedEventHandler.class);
    final Map<String, String> clientEnv = ImmutableMap.of();

    Scratch scratch = new Scratch();
    final Path destination = scratch.resolve("output file path");
    downloader.download(
        urls,
        authHeaders,
        guavaChecksum,
        canonicalId,
        destination,
        eventHandler,
        clientEnv,
        com.google.common.base.Optional.<String>absent());

    try (InputStream in = destination.getInputStream()) {
      return ByteStreams.toByteArray(in);
    }
  }

  @Test
  public void testDownload() throws Exception {
    final byte[] content = "example content".getBytes(UTF_8);
    final Digest contentDigest = DIGEST_UTIL.compute(content);

    serviceRegistry.addService(
        new FetchImplBase() {
          @Override
          public void fetchBlob(
              FetchBlobRequest request, StreamObserver<FetchBlobResponse> responseObserver) {
            assertThat(request)
                .isEqualTo(
                    FetchBlobRequest.newBuilder()
                        .addUris("http://example.com/content.txt")
                        .build());
            responseObserver.onNext(
                FetchBlobResponse.newBuilder().setBlobDigest(contentDigest).build());
            responseObserver.onCompleted();
          }
        });

    final RemoteCacheClient cacheClient = new InMemoryCacheClient();
    final GrpcRemoteDownloader downloader = newDownloader(cacheClient);

    getFromFuture(cacheClient.uploadBlob(contentDigest, ByteString.copyFrom(content)));
    final byte[] downloaded =
        downloadBlob(
            downloader, new URL("http://example.com/content.txt"), Optional.<Checksum>empty());

    assertThat(downloaded).isEqualTo(content);
  }

  @Test
  public void testPropagateChecksum() throws Exception {
    final byte[] content = "example content".getBytes(UTF_8);
    final Digest contentDigest = DIGEST_UTIL.compute(content);

    serviceRegistry.addService(
        new FetchImplBase() {
          @Override
          public void fetchBlob(
              FetchBlobRequest request, StreamObserver<FetchBlobResponse> responseObserver) {
            assertThat(request)
                .isEqualTo(
                    FetchBlobRequest.newBuilder()
                        .addUris("http://example.com/content.txt")
                        .addQualifiers(
                            Qualifier.newBuilder()
                                .setName("checksum.sri")
                                .setValue("sha256-ot7ke6YmiSXal3UKt0K69n8C4vtUziPUmftmpbAiKQM="))
                        .build());
            responseObserver.onNext(
                FetchBlobResponse.newBuilder().setBlobDigest(contentDigest).build());
            responseObserver.onCompleted();
          }
        });

    final RemoteCacheClient cacheClient = new InMemoryCacheClient();
    final GrpcRemoteDownloader downloader = newDownloader(cacheClient);

    getFromFuture(cacheClient.uploadBlob(contentDigest, ByteString.copyFrom(content)));
    final byte[] downloaded =
        downloadBlob(
            downloader,
            new URL("http://example.com/content.txt"),
            Optional.<Checksum>of(Checksum.fromString(KeyType.SHA256, contentDigest.getHash())));

    assertThat(downloaded).isEqualTo(content);
  }

  @Test
  public void testRejectChecksumMismatch() throws Exception {
    final byte[] content = "example content".getBytes(UTF_8);
    final Digest contentDigest = DIGEST_UTIL.compute(content);

    serviceRegistry.addService(
        new FetchImplBase() {
          @Override
          public void fetchBlob(
              FetchBlobRequest request, StreamObserver<FetchBlobResponse> responseObserver) {
            assertThat(request)
                .isEqualTo(
                    FetchBlobRequest.newBuilder()
                        .addUris("http://example.com/content.txt")
                        .addQualifiers(
                            Qualifier.newBuilder()
                                .setName("checksum.sri")
                                .setValue("sha256-ot7ke6YmiSXal3UKt0K69n8C4vtUziPUmftmpbAiKQM="))
                        .build());
            responseObserver.onNext(
                FetchBlobResponse.newBuilder().setBlobDigest(contentDigest).build());
            responseObserver.onCompleted();
          }
        });

    final RemoteCacheClient cacheClient = new InMemoryCacheClient();
    final GrpcRemoteDownloader downloader = newDownloader(cacheClient);

    getFromFuture(cacheClient.uploadBlob(contentDigest, ByteString.copyFromUtf8("wrong content")));

    IOException e =
        assertThrows(
            UnrecoverableHttpException.class,
            () ->
                downloadBlob(
                    downloader,
                    new URL("http://example.com/content.txt"),
                    Optional.<Checksum>of(
                        Checksum.fromString(KeyType.SHA256, contentDigest.getHash()))));

    assertThat(e).hasMessageThat().contains(contentDigest.getHash());
    assertThat(e).hasMessageThat().contains(DIGEST_UTIL.computeAsUtf8("wrong content").getHash());
  }

  @Test
  public void testFetchBlobRequest() throws Exception {
    FetchBlobRequest request =
        GrpcRemoteDownloader.newFetchBlobRequest(
            "instance name",
            ImmutableList.of(
                new URL("http://example.com/a"),
                new URL("http://example.com/b"),
                new URL("file:/not/limited/to/http")),
            ImmutableMap.of(
                new URI("http://example.com"),
                ImmutableMap.of(
                    "Some-Header", "some header content",
                    "Another-Header", "another header content"),
                new URI("http://example.org"),
                ImmutableMap.of("Org-Header", "org header content")),
            com.google.common.base.Optional.<Checksum>of(
                Checksum.fromSubresourceIntegrity(
                    "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")),
            "canonical ID");

    final String expectedAuthHeadersJson =
        "{"
            + "\"http://example.com\":{"
            + "\"Another-Header\":\"another header content\","
            + "\"Some-Header\":\"some header content\""
            + "},"
            + "\"http://example.org\":{"
            + "\"Org-Header\":\"org header content\""
            + "}"
            + "}";

    assertThat(request)
        .isEqualTo(
            FetchBlobRequest.newBuilder()
                .setInstanceName("instance name")
                .addUris("http://example.com/a")
                .addUris("http://example.com/b")
                .addUris("file:/not/limited/to/http")
                .addQualifiers(
                    Qualifier.newBuilder()
                        .setName("checksum.sri")
                        .setValue("sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="))
                .addQualifiers(
                    Qualifier.newBuilder().setName("bazel.canonical_id").setValue("canonical ID"))
                .addQualifiers(
                    Qualifier.newBuilder()
                        .setName("bazel.auth_headers")
                        .setValue(expectedAuthHeadersJson))
                .build());
  }
}
