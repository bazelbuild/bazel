// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceImplBase;
import com.google.devtools.build.lib.remote.RemoteProtocol.ActionResult;
import com.google.devtools.build.lib.remote.RemoteProtocol.BlobChunk;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasStatus;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.ContentDigest;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.ConcurrentMap;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Tests for {@link GrpcActionCache}. */
@RunWith(JUnit4.class)
public class GrpcActionCacheTest {
  private final FakeRemoteCacheService fakeRemoteCacheService = new FakeRemoteCacheService();

  private final Server server =
      InProcessServerBuilder.forName(getClass().getSimpleName())
          .directExecutor()
          .addService(fakeRemoteCacheService)
          .build();

  private final ManagedChannel channel =
      InProcessChannelBuilder.forName(getClass().getSimpleName()).directExecutor().build();
  private Scratch scratch;
  private Root rootDir;

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    scratch = new Scratch();
    rootDir = Root.asDerivedRoot(scratch.dir("/exec/root"));
    server.start();
  }

  @After
  public void tearDown() {
    server.shutdownNow();
    channel.shutdownNow();
  }

  @Test
  public void testDownloadEmptyBlobs() throws Exception {
    GrpcActionCache client = new GrpcActionCache(channel, Options.getDefaults(RemoteOptions.class));
    ContentDigest fooDigest = fakeRemoteCacheService.put("foo".getBytes(UTF_8));
    ContentDigest emptyDigest = ContentDigests.computeDigest(new byte[0]);
    ImmutableList<byte[]> results =
        client.downloadBlobs(ImmutableList.<ContentDigest>of(emptyDigest, fooDigest, emptyDigest));
    // Will not query the server for empty blobs.
    assertThat(new String(results.get(0), UTF_8)).isEmpty();
    assertThat(new String(results.get(1), UTF_8)).isEqualTo("foo");
    assertThat(new String(results.get(2), UTF_8)).isEmpty();
    // Will not call the server at all.
    assertThat(new String(client.downloadBlob(emptyDigest), UTF_8)).isEmpty();
  }

  @Test
  public void testDownloadBlobs() throws Exception {
    GrpcActionCache client = new GrpcActionCache(channel, Options.getDefaults(RemoteOptions.class));
    ContentDigest fooDigest = fakeRemoteCacheService.put("foo".getBytes(UTF_8));
    ContentDigest barDigest = fakeRemoteCacheService.put("bar".getBytes(UTF_8));
    ImmutableList<byte[]> results =
        client.downloadBlobs(ImmutableList.<ContentDigest>of(fooDigest, barDigest));
    assertThat(new String(results.get(0), UTF_8)).isEqualTo("foo");
    assertThat(new String(results.get(1), UTF_8)).isEqualTo("bar");
  }

  @Test
  public void testDownloadBlobsBatchChunk() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.grpcMaxBatchInputs = 10;
    options.grpcMaxChunkSizeBytes = 2;
    options.grpcMaxBatchSizeBytes = 10;
    options.grpcTimeoutSeconds = 10;
    GrpcActionCache client = new GrpcActionCache(channel, options);
    ContentDigest fooDigest = fakeRemoteCacheService.put("fooooooo".getBytes(UTF_8));
    ContentDigest barDigest = fakeRemoteCacheService.put("baaaar".getBytes(UTF_8));
    ContentDigest s1Digest = fakeRemoteCacheService.put("1".getBytes(UTF_8));
    ContentDigest s2Digest = fakeRemoteCacheService.put("2".getBytes(UTF_8));
    ContentDigest s3Digest = fakeRemoteCacheService.put("3".getBytes(UTF_8));
    ImmutableList<byte[]> results =
        client.downloadBlobs(
            ImmutableList.<ContentDigest>of(fooDigest, barDigest, s1Digest, s2Digest, s3Digest));
    assertThat(new String(results.get(0), UTF_8)).isEqualTo("fooooooo");
    assertThat(new String(results.get(1), UTF_8)).isEqualTo("baaaar");
    assertThat(new String(results.get(2), UTF_8)).isEqualTo("1");
    assertThat(new String(results.get(3), UTF_8)).isEqualTo("2");
    assertThat(new String(results.get(4), UTF_8)).isEqualTo("3");
  }

  @Test
  public void testUploadBlobs() throws Exception {
    GrpcActionCache client = new GrpcActionCache(channel, Options.getDefaults(RemoteOptions.class));
    byte[] foo = "foo".getBytes(UTF_8);
    byte[] bar = "bar".getBytes(UTF_8);
    ContentDigest fooDigest = ContentDigests.computeDigest(foo);
    ContentDigest barDigest = ContentDigests.computeDigest(bar);
    ImmutableList<ContentDigest> digests = client.uploadBlobs(ImmutableList.<byte[]>of(foo, bar));
    assertThat(digests).containsExactly(fooDigest, barDigest);
    assertThat(fakeRemoteCacheService.get(fooDigest)).isEqualTo(foo);
    assertThat(fakeRemoteCacheService.get(barDigest)).isEqualTo(bar);
  }

  @Test
  public void testUploadBlobsBatchChunk() throws Exception {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.grpcMaxBatchInputs = 10;
    options.grpcMaxChunkSizeBytes = 2;
    options.grpcMaxBatchSizeBytes = 10;
    options.grpcTimeoutSeconds = 10;
    GrpcActionCache client = new GrpcActionCache(channel, options);

    byte[] foo = "fooooooo".getBytes(UTF_8);
    byte[] bar = "baaaar".getBytes(UTF_8);
    byte[] s1 = "1".getBytes(UTF_8);
    byte[] s2 = "2".getBytes(UTF_8);
    byte[] s3 = "3".getBytes(UTF_8);
    ContentDigest fooDigest = ContentDigests.computeDigest(foo);
    ContentDigest barDigest = ContentDigests.computeDigest(bar);
    ContentDigest s1Digest = ContentDigests.computeDigest(s1);
    ContentDigest s2Digest = ContentDigests.computeDigest(s2);
    ContentDigest s3Digest = ContentDigests.computeDigest(s3);
    ImmutableList<ContentDigest> digests =
        client.uploadBlobs(ImmutableList.<byte[]>of(foo, bar, s1, s2, s3));
    assertThat(digests).containsExactly(fooDigest, barDigest, s1Digest, s2Digest, s3Digest);
    assertThat(fakeRemoteCacheService.get(fooDigest)).isEqualTo(foo);
    assertThat(fakeRemoteCacheService.get(barDigest)).isEqualTo(bar);
    assertThat(fakeRemoteCacheService.get(s1Digest)).isEqualTo(s1);
    assertThat(fakeRemoteCacheService.get(s2Digest)).isEqualTo(s2);
    assertThat(fakeRemoteCacheService.get(s3Digest)).isEqualTo(s3);
  }

  @Test
  public void testUploadAllResults() throws Exception {
    GrpcActionCache client = new GrpcActionCache(channel, Options.getDefaults(RemoteOptions.class));
    byte[] foo = "foo".getBytes(UTF_8);
    byte[] bar = "bar".getBytes(UTF_8);
    Path fooFile = scratch.file("/exec/root/a/foo", foo);
    Path emptyFile = scratch.file("/exec/root/b/empty");
    Path barFile = scratch.file("/exec/root/a/bar", bar);
    ContentDigest fooDigest = ContentDigests.computeDigest(fooFile);
    ContentDigest barDigest = ContentDigests.computeDigest(barFile);
    ContentDigest emptyDigest = ContentDigests.computeDigest(new byte[0]);
    ActionResult.Builder result = ActionResult.newBuilder();
    client.uploadAllResults(
        rootDir.getPath(), ImmutableList.<Path>of(fooFile, emptyFile, barFile), result);
    assertThat(fakeRemoteCacheService.get(fooDigest)).isEqualTo(foo);
    assertThat(fakeRemoteCacheService.get(barDigest)).isEqualTo(bar);
    ActionResult.Builder expectedResult = ActionResult.newBuilder();
    expectedResult
        .addOutputBuilder()
        .setPath("a/foo")
        .getFileMetadataBuilder()
        .setDigest(fooDigest);
    expectedResult
        .addOutputBuilder()
        .setPath("b/empty")
        .getFileMetadataBuilder()
        .setDigest(emptyDigest);
    expectedResult
        .addOutputBuilder()
        .setPath("a/bar")
        .getFileMetadataBuilder()
        .setDigest(barDigest);
    assertThat(result.build()).isEqualTo(expectedResult.build());
  }

  @Test
  public void testDownloadAllResults() throws Exception {
    GrpcActionCache client = new GrpcActionCache(channel, Options.getDefaults(RemoteOptions.class));
    ContentDigest fooDigest = fakeRemoteCacheService.put("foo".getBytes(UTF_8));
    ContentDigest barDigest = fakeRemoteCacheService.put("bar".getBytes(UTF_8));
    ContentDigest emptyDigest = ContentDigests.computeDigest(new byte[0]);
    ActionResult.Builder result = ActionResult.newBuilder();
    result.addOutputBuilder().setPath("a/foo").getFileMetadataBuilder().setDigest(fooDigest);
    result.addOutputBuilder().setPath("b/empty").getFileMetadataBuilder().setDigest(emptyDigest);
    result.addOutputBuilder().setPath("a/bar").getFileMetadataBuilder().setDigest(barDigest);
    client.downloadAllResults(result.build(), rootDir.getPath());
    Path fooFile = rootDir.getPath().getRelative("a/foo");
    Path emptyFile = rootDir.getPath().getRelative("b/empty");
    Path barFile = rootDir.getPath().getRelative("a/bar");
    assertThat(ContentDigests.computeDigest(fooFile)).isEqualTo(fooDigest);
    assertThat(ContentDigests.computeDigest(emptyFile)).isEqualTo(emptyDigest);
    assertThat(ContentDigests.computeDigest(barFile)).isEqualTo(barDigest);
  }

  private static class FakeRemoteCacheService extends CasServiceImplBase {
    private final ConcurrentMap<String, byte[]> cache = Maps.newConcurrentMap();

    public ContentDigest put(byte[] blob) {
      ContentDigest digest = ContentDigests.computeDigest(blob);
      cache.put(ContentDigests.toHexString(digest), blob);
      return digest;
    }

    public byte[] get(ContentDigest digest) {
      return cache.get(ContentDigests.toHexString(digest));
    }

    public void clear() {
      cache.clear();
    }

    @Override
    public void lookup(CasLookupRequest request, StreamObserver<CasLookupReply> observer) {
      CasLookupReply.Builder reply = CasLookupReply.newBuilder();
      CasStatus.Builder status = reply.getStatusBuilder();
      for (ContentDigest digest : request.getDigestList()) {
        if (get(digest) == null) {
          status.addMissingDigest(digest);
        }
      }
      status.setSucceeded(true);
      observer.onNext(reply.build());
      observer.onCompleted();
    }

    @Override
    public void downloadBlob(
        CasDownloadBlobRequest request, StreamObserver<CasDownloadReply> observer) {
      CasDownloadReply.Builder reply = CasDownloadReply.newBuilder();
      CasStatus.Builder status = reply.getStatusBuilder();
      boolean success = true;
      for (ContentDigest digest : request.getDigestList()) {
        if (get(digest) == null) {
          status.addMissingDigest(digest);
          success = false;
        }
      }
      if (!success) {
        status.setError(CasStatus.ErrorCode.MISSING_DIGEST);
        status.setSucceeded(false);
        observer.onNext(reply.build());
        observer.onCompleted();
        return;
      }
      for (ContentDigest digest : request.getDigestList()) {
        observer.onNext(
            CasDownloadReply.newBuilder()
                .setStatus(CasStatus.newBuilder().setSucceeded(true))
                .setData(
                    BlobChunk.newBuilder()
                        .setDigest(digest)
                        .setData(ByteString.copyFrom(get(digest))))
                .build());
      }
      observer.onCompleted();
    }

    @Override
    public StreamObserver<CasUploadBlobRequest> uploadBlob(
        final StreamObserver<CasUploadBlobReply> responseObserver) {
      return new StreamObserver<CasUploadBlobRequest>() {
        byte[] blob = null;
        ContentDigest digest = null;
        long offset = 0;

        @Override
        public void onNext(CasUploadBlobRequest request) {
          BlobChunk chunk = request.getData();
          try {
            if (chunk.hasDigest()) {
              // Check if the previous chunk was really done.
              Preconditions.checkArgument(
                  digest == null || offset == 0,
                  "Missing input chunk for digest %s",
                  digest == null ? "" : ContentDigests.toString(digest));
              digest = chunk.getDigest();
              blob = new byte[(int) digest.getSizeBytes()];
            }
            Preconditions.checkArgument(digest != null, "First chunk contains no digest");
            Preconditions.checkArgument(
                offset == chunk.getOffset(),
                "Missing input chunk for digest %s",
                ContentDigests.toString(digest));
            chunk.getData().copyTo(blob, (int) offset);
            offset = (offset + chunk.getData().size()) % digest.getSizeBytes();
            if (offset == 0) {
              ContentDigest uploadedDigest = put(blob);
              Preconditions.checkArgument(
                  uploadedDigest.equals(digest),
                  "Digest mismatch: client sent %s, server computed %s",
                  ContentDigests.toString(digest),
                  ContentDigests.toString(uploadedDigest));
            }
          } catch (Exception e) {
            CasUploadBlobReply.Builder reply = CasUploadBlobReply.newBuilder();
            reply
                .getStatusBuilder()
                .setSucceeded(false)
                .setError(
                    e instanceof IllegalArgumentException
                        ? CasStatus.ErrorCode.INVALID_ARGUMENT
                        : CasStatus.ErrorCode.UNKNOWN)
                .setErrorDetail(e.toString());
            responseObserver.onNext(reply.build());
          }
        }

        @Override
        public void onError(Throwable t) {}

        @Override
        public void onCompleted() {
          responseObserver.onCompleted();
        }
      };
    }
  }
}
