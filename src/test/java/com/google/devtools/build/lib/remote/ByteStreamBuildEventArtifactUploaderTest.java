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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashCode;
import com.google.common.io.BaseEncoding;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.ByteStreamUploaderTest.FixedBackoff;
import com.google.devtools.build.lib.remote.ByteStreamUploaderTest.MaybeFailOnceUploadService;
import com.google.devtools.build.lib.remote.common.MissingDigestsFinder;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import io.grpc.Context;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Test for {@link ByteStreamBuildEventArtifactUploader}. */
@RunWith(JUnit4.class)
public class ByteStreamBuildEventArtifactUploaderTest {

  private static final DigestUtil DIGEST_UTIL = new DigestUtil(DigestHashFunction.SHA256);

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private ListeningScheduledExecutorService retryService;

  private Server server;
  private ManagedChannel channel;
  private Context withEmptyMetadata;
  private Context prevContext;
  private final FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);

  private final Path execRoot = fs.getPath("/execroot");
  private ArtifactRoot outputRoot;

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

    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, "out");
    outputRoot.getRoot().asPath().createDirectoryAndParents();

    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @After
  public void tearDown() throws Exception {
    // Needs to be repeated in every test that uses the timeout setting, since the tests run
    // on different threads than the tearDown.
    withEmptyMetadata.detach(prevContext);

    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);

    channel.shutdownNow();
    channel.awaitTermination(5, TimeUnit.SECONDS);
    server.shutdownNow();
    server.awaitTermination();
  }

  @Before
  public void setup() {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void uploadsShouldWork() throws Exception {
    int numUploads = 2;
    Map<HashCode, byte[]> blobsByHash = new HashMap<>();
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
      blobsByHash.put(HashCode.fromString(DIGEST_UTIL.compute(file).getHash()), blob);
      filesToUpload.put(file, new LocalFile(file, LocalFileType.OUTPUT));
    }
    serviceRegistry.addService(new MaybeFailOnceUploadService(blobsByHash));

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 0), (e) -> true, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channel);
    ByteStreamUploader uploader =
        new ByteStreamUploader("instance", refCntChannel, null, 3, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(uploader);

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
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(uploader);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(dir)).isNull();
    artifactUploader.shutdown();
  }

  @Test
  public void someUploadsFail() throws Exception {
    // Test that if one of multiple file uploads fails, the upload future fails and that the
    // error is propagated correctly.

    int numUploads = 10;
    Map<HashCode, byte[]> blobsByHash = new HashMap<>();
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
      blobsByHash.put(HashCode.fromString(DIGEST_UTIL.compute(file).getHash()), blob);
      filesToUpload.put(file, new LocalFile(file, LocalFileType.OUTPUT));
    }
    String hashOfBlobThatShouldFail = blobsByHash.keySet().iterator().next().toString();
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
        TestUtils.newRemoteRetrier(() -> new FixedBackoff(1, 0), (e) -> true, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channel);
    ByteStreamUploader uploader =
        new ByteStreamUploader("instance", refCntChannel, null, 3, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(uploader);

    ExecutionException e =
        assertThrows(ExecutionException.class, () -> artifactUploader.upload(filesToUpload).get());
    // The gRPC library uses StatusRuntimeException to raise errors. However, throughout the Bazel
    // codebase runtime exceptions are considered bugs. This test ensures that a SRE is converted
    // to a checked exception type.
    assertThat(e.getCause()).isInstanceOf(IOException.class);
    assertThat(e.getCause().getCause()).isInstanceOf(StatusRuntimeException.class);
    assertThat(Status.fromThrowable(e).getCode()).isEqualTo(Status.CANCELLED.getCode());

    artifactUploader.shutdown();

    assertThat(uploader.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void remoteFileShouldNotBeUploaded_actionFs() throws Exception {
    // Test that we don't attempt to upload remotely stored file but convert the remote path
    // to a bytestream:// URI.

    // arrange

    ByteStreamUploader uploader = Mockito.mock(ByteStreamUploader.class);
    RemoteActionInputFetcher actionInputFetcher = Mockito.mock(RemoteActionInputFetcher.class);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(uploader);

    ActionInputMap outputs = new ActionInputMap(2);
    Artifact artifact = createRemoteArtifact("file1.txt", "foo", outputs);

    RemoteActionFileSystem remoteFs =
        new RemoteActionFileSystem(
            fs,
            execRoot.asFragment(),
            outputRoot.getRoot().asPath().relativeTo(execRoot).getPathString(),
            outputs,
            actionInputFetcher);
    Path remotePath = remoteFs.getPath(artifact.getPath().getPathString());
    assertThat(remotePath.getFileSystem()).isEqualTo(remoteFs);
    LocalFile file = new LocalFile(remotePath, LocalFileType.OUTPUT);

    // act

    PathConverter pathConverter = artifactUploader.upload(ImmutableMap.of(remotePath, file)).get();

    FileArtifactValue metadata = outputs.getMetadata(artifact);
    Digest digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());

    // assert

    String conversion = pathConverter.apply(remotePath);
    assertThat(conversion)
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    verifyNoMoreInteractions(uploader);
  }

  @Test
  public void remoteFileShouldNotBeUploaded_findMissingDigests() throws Exception {
    // Test that findMissingDigests is called to check which files exist remotely
    // and that those are not uploaded.

    // arrange
    Path remoteFile = fs.getPath("/remote-file");
    FileSystemUtils.writeContent(remoteFile, StandardCharsets.UTF_8, "hello world");
    Digest remoteDigest = DIGEST_UTIL.compute(remoteFile);
    Path localFile = fs.getPath("/local-file");
    FileSystemUtils.writeContent(localFile, StandardCharsets.UTF_8, "foo bar");
    Digest localDigest = DIGEST_UTIL.compute(localFile);

    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of(remoteDigest)));
    ByteStreamUploader uploader = Mockito.mock(ByteStreamUploader.class);
    when(uploader.uploadBlobAsync(any(), any(), anyBoolean()))
        .thenReturn(Futures.immediateFuture(null));
    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(uploader, digestQuerier);

    // act
    Map<Path, LocalFile> files =
        ImmutableMap.of(
            remoteFile,
            new LocalFile(remoteFile, LocalFileType.OUTPUT),
            localFile,
            new LocalFile(localFile, LocalFileType.OUTPUT));
    PathConverter pathConverter = artifactUploader.upload(files).get();

    // assert
    verify(digestQuerier).findMissingDigests(any());
    verify(uploader)
        .uploadBlobAsync(eq(HashCode.fromString(localDigest.getHash())), any(), anyBoolean());
    assertThat(pathConverter.apply(remoteFile)).contains(remoteDigest.getHash());
    assertThat(pathConverter.apply(localFile)).contains(localDigest.getHash());
  }

  /** Returns a remote artifact and puts its metadata into the action input map. */
  private Artifact createRemoteArtifact(
      String pathFragment, String contents, ActionInputMap inputs) {
    Path p = outputRoot.getRoot().asPath().getRelative(pathFragment);
    Artifact a = ActionsTestUtil.createArtifact(outputRoot, p);
    byte[] b = contents.getBytes(StandardCharsets.UTF_8);
    HashCode h = HashCode.fromString(DIGEST_UTIL.compute(b).getHash());
    FileArtifactValue f =
        new RemoteFileArtifactValue(h.asBytes(), b.length, /* locationIndex= */ 1, "action-id");
    inputs.putWithNoDepOwner(a, f);
    return a;
  }

  private ByteStreamBuildEventArtifactUploader newArtifactUploader(
      ByteStreamUploader uploader, MissingDigestsFinder missingDigestsFinder) {
    return new ByteStreamBuildEventArtifactUploader(
        uploader,
        missingDigestsFinder,
        "localhost",
        withEmptyMetadata,
        "instance",
        /* maxUploadThreads= */ 100);
  }

  private ByteStreamBuildEventArtifactUploader newArtifactUploader(ByteStreamUploader uploader) {
    return newArtifactUploader(uploader, AllMissingDigestsFinder.INSTANCE);
  }

  private static class StaticMissingDigestsFinder implements MissingDigestsFinder {

    private final ImmutableSet<Digest> knownDigests;

    public StaticMissingDigestsFinder(ImmutableSet<Digest> knownDigests) {
      this.knownDigests = knownDigests;
    }

    @Override
    public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
      ImmutableSet.Builder<Digest> missingDigests = ImmutableSet.builder();
      for (Digest digest : digests) {
        if (!knownDigests.contains(digest)) {
          missingDigests.add(digest);
        }
      }
      return Futures.immediateFuture(missingDigests.build());
    }
  }

  private static class AllMissingDigestsFinder implements MissingDigestsFinder {

    public static final AllMissingDigestsFinder INSTANCE = new AllMissingDigestsFinder();

    @Override
    public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(Iterable<Digest> digests) {
      return Futures.immediateFuture(ImmutableSet.copyOf(digests));
    }
  }
}
