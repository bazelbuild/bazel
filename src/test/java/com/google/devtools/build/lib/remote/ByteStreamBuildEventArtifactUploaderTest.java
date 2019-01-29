// Copyright 2018 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

import build.bazel.remote.execution.v2.Digest;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.io.BaseEncoding;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.ByteStreamUploaderTest.FixedBackoff;
import com.google.devtools.build.lib.remote.ByteStreamUploaderTest.MaybeFailOnceUploadService;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Test for {@link ByteStreamBuildEventArtifactUploader}. */
@RunWith(JUnit4.class)
public class ByteStreamBuildEventArtifactUploaderTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private static ListeningScheduledExecutorService retryService;

  private Server server;
  private ManagedChannel channel;
  private Context withEmptyMetadata;
  private Context prevContext;
  private final FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);

    String serverName = "Server for " + this.getClass();
    server =
        InProcessServerBuilder.forName(serverName)
            .fallbackHandlerRegistry(serviceRegistry)
            .build()
            .start();
    channel = InProcessChannelBuilder.forName(serverName).build();
    withEmptyMetadata =
        TracingMetadataUtils.contextWithMetadata(
            "none", "none", DIGEST_UTIL.asActionKey(Digest.getDefaultInstance()));
    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the setUp.
    prevContext = withEmptyMetadata.attach();
  }

  @After
  public void tearDown() throws Exception {
    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the tearDown.
    withEmptyMetadata.detach(prevContext);

    server.shutdownNow();
    server.awaitTermination();
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Before
  public void setup() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void uploadsShouldWork() throws Exception {
    int numUploads = 2;
    Map<String, byte[]> blobsByHash = new HashMap<>();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/file" + i);
      OutputStream out = file.getOutputStream();
      int blobSize = rand.nextInt(100) + 1;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      out.write(blob);
      out.close();
      blobsByHash.put(DIGEST_UTIL.compute(file).getHash(), blob);
      filesToUpload.put(file, new LocalFile(file, LocalFileType.OUTPUT));
    }
    serviceRegistry.addService(new MaybeFailOnceUploadService(blobsByHash));

    RemoteRetrier retrier =
        new RemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> true, retryService, Retrier.ALLOW_ALL_CALLS);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channel);
    ByteStreamUploader uploader =
        new ByteStreamUploader("instance", refCntChannel, null, 3, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader =
        new ByteStreamBuildEventArtifactUploader(
            uploader, "localhost", withEmptyMetadata, "instance", /* maxUploadThreads= */ 100);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    for (Path file : filesToUpload.keySet()) {
      String hash = BaseEncoding.base16().lowerCase().encode(file.getDigest());
      long size = file.getFileSize();
      String conversion = pathConverter.apply(file);
      assertThat(conversion)
          .isEqualTo("bytestream://localhost/instance/blobs/" + hash + "/" + size);
    }

    artifactUploader.shutdown();

    assertThat(uploader.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void testUploadDirectoryDoesNotCrash() throws Exception {
    Path dir = fs.getPath("/dir");
    dir.createDirectoryAndParents();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(dir, new LocalFile(dir, LocalFileType.OUTPUT));
    ByteStreamUploader uploader = mock(ByteStreamUploader.class);
    ByteStreamBuildEventArtifactUploader artifactUploader =
        new ByteStreamBuildEventArtifactUploader(
            uploader, "localhost", withEmptyMetadata, "instance", /* maxUploadThreads= */ 100);
    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(dir)).isNull();
    artifactUploader.shutdown();
  }

  @Test
  public void someUploadsFail() throws Exception {
    // Test that if one of multiple file uploads fails, the upload future fails and that the
    // error is propagated correctly.

    int numUploads = 10;
    Map<String, byte[]> blobsByHash = new HashMap<>();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/file" + i);
      OutputStream out = file.getOutputStream();
      int blobSize = rand.nextInt(100) + 1;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      out.write(blob);
      out.flush();
      out.close();
      blobsByHash.put(DIGEST_UTIL.compute(file).getHash(), blob);
      filesToUpload.put(file, new LocalFile(file, LocalFileType.OUTPUT));
    }
    String hashOfBlobThatShouldFail = blobsByHash.keySet().iterator().next();
    serviceRegistry.addService(new MaybeFailOnceUploadService(blobsByHash) {
      @Override
      public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
        StreamObserver<WriteRequest> delegate = super.write(response);
        return new StreamObserver<WriteRequest>() {
          @Override
          public void onNext(WriteRequest value) {
            if (value.getResourceName().contains(hashOfBlobThatShouldFail)) {
              response.onError(Status.CANCELLED.asException());
            } else {
              delegate.onNext(value);
            }
          }

          @Override
          public void onError(Throwable t) {
            delegate.onError(t);
          }

          @Override
          public void onCompleted() {
            delegate.onCompleted();
          }
        };
      }
    });

    RemoteRetrier retrier =
        new RemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> true, retryService, Retrier.ALLOW_ALL_CALLS);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channel);
    ByteStreamUploader uploader =
        new ByteStreamUploader("instance", refCntChannel, null, 3, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader =
        new ByteStreamBuildEventArtifactUploader(
            uploader, "localhost", withEmptyMetadata, "instance", /* maxUploadThreads= */ 100);

    try {
      artifactUploader.upload(filesToUpload).get();
      fail("exception expected.");
    } catch (ExecutionException e) {
      assertThat(Status.fromThrowable(e).getCode()).isEqualTo(Status.CANCELLED.getCode());
    }

    artifactUploader.shutdown();

    assertThat(uploader.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }
}
