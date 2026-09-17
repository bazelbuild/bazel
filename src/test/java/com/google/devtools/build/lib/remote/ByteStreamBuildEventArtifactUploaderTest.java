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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ServerCapabilities;
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
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.EventBusEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.remote.ByteStreamUploaderTest.FixedBackoff;
import com.google.devtools.build.lib.remote.ByteStreamUploaderTest.MaybeFailOnceUploadService;
import com.google.devtools.build.lib.remote.Retrier.ResultClassifier.Result;
import com.google.devtools.build.lib.remote.common.MissingDigestsFinder;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.options.RemoteBuildEventUploadMode;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.RxNoGlobalErrorsRule;
import com.google.devtools.build.lib.remote.util.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.bazel.BazelHashFunctions;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.util.MutableHandlerRegistry;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;
import org.openjdk.jol.info.GraphLayout;

/** Test for {@link ByteStreamBuildEventArtifactUploader}. */
@RunWith(JUnit4.class)
public class ByteStreamBuildEventArtifactUploaderTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  @Rule public final RxNoGlobalErrorsRule rxNoGlobalErrorsRule = new RxNoGlobalErrorsRule();

  private final Reporter reporter = new Reporter(EventBusEventHandler.createWithNewEventBus());
  private final StoredEventHandler eventHandler = new StoredEventHandler();

  private final MutableHandlerRegistry serviceRegistry = new MutableHandlerRegistry();
  private ListeningScheduledExecutorService retryService;

  private Server server;
  private ChannelConnectionWithServerCapabilitiesFactory channelConnectionFactory;

  private final FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);

  private final Path execRoot = fs.getPath("/execroot");
  private ArtifactRoot outputRoot;

  @Before
  public final void setUp() throws Exception {
    reporter.addHandler(eventHandler);

    String serverName = "Server for " + this.getClass();
    server =
        InProcessServerBuilder.forName(serverName)
            .fallbackHandlerRegistry(serviceRegistry)
            .build()
            .start();
    channelConnectionFactory =
        new ChannelConnectionWithServerCapabilitiesFactory() {
          @Override
          public Single<ChannelConnectionWithServerCapabilities> create() {
            return Single.just(
                new ChannelConnectionWithServerCapabilities(
                    InProcessChannelBuilder.forName(serverName).build(),
                    Single.just(ServerCapabilities.getDefaultInstance())));
          }

          @Override
          public int maxConcurrency() {
            return 100;
          }
        };

    outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    outputRoot.getRoot().asPath().createDirectoryAndParents();

    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @After
  public void tearDown() throws Exception {

    retryService.shutdownNow();
    retryService.awaitTermination(
        com.google.devtools.build.lib.testutil.TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);

    server.shutdownNow();
    server.awaitTermination();
  }

  @Test
  public void uploadsShouldWork() throws Exception {
    int numUploads = 2;
    Map<HashCode, byte[]> blobsByHash = new HashMap<>();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/file" + i);
      int blobSize = rand.nextInt(100) + 1;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      FileSystemUtils.writeContent(file, blob);
      blobsByHash.put(HashCode.fromString(DIGEST_UTIL.compute(file).getHash()), blob);
      filesToUpload.put(
          file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    }
    serviceRegistry.addService(new MaybeFailOnceUploadService(blobsByHash));

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    for (Path file : filesToUpload.keySet()) {
      String hash = BaseEncoding.base16().lowerCase().encode(file.getDigest());
      long size = file.getFileSize();
      String conversion = pathConverter.apply(file);
      assertThat(conversion)
          .isEqualTo("bytestream://localhost/instance/blobs/" + hash + "/" + size);
    }

    artifactUploader.release();

    assertThat(combinedCache.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void uploadsShouldIgnoreSpecialFiles() throws Exception {
    Path file = Mockito.spy(fs.getPath("/fifo"));
    Mockito.doReturn(true).when(file).isSpecialFile();

    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(file, new LocalFile(file, LocalFileType.LOG, /* artifactMetadata= */ null));

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    String conversion = pathConverter.apply(file);
    assertThat(conversion).isEqualTo("file:///fifo");

    artifactUploader.release();
  }

  @Test
  public void uploadsShouldWork_fewerPermitsThanUploads() throws Exception {
    int numUploads = 2;
    Map<HashCode, byte[]> blobsByHash = new HashMap<>();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/file" + i);
      int blobSize = rand.nextInt(100) + 1;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      FileSystemUtils.writeContent(file, blob);
      blobsByHash.put(HashCode.fromString(DIGEST_UTIL.compute(file).getHash()), blob);
      filesToUpload.put(
          file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    }
    serviceRegistry.addService(new MaybeFailOnceUploadService(blobsByHash));

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    // number of permits is less than number of uploads to affirm permit is released
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(
            combinedCache, RemoteBuildEventUploadMode.ALL, /* maximumOpenFiles= */ 1);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    for (Path file : filesToUpload.keySet()) {
      String hash = BaseEncoding.base16().lowerCase().encode(file.getDigest());
      long size = file.getFileSize();
      String conversion = pathConverter.apply(file);
      assertThat(conversion)
          .isEqualTo("bytestream://localhost/instance/blobs/" + hash + "/" + size);
    }

    artifactUploader.release();

    assertThat(combinedCache.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void uploadsRespectMaxConcurrency() throws Exception {
    int numUploads = 5;
    int maxConcurrency = 2;
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/concurrency_file" + i);
      FileSystemUtils.writeContent(file, new byte[] {(byte) i});
      filesToUpload.put(
          file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    }

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));

    AtomicInteger inFlightUploads = new AtomicInteger(0);
    AtomicInteger maxInFlightUploads = new AtomicInteger(0);
    List<SettableFuture<Void>> futures = new ArrayList<>();

    doAnswer(
            invocation -> {
              int current = inFlightUploads.incrementAndGet();
              maxInFlightUploads.accumulateAndGet(current, Math::max);
              SettableFuture<Void> f = SettableFuture.create();
              synchronized (futures) {
                futures.add(f);
              }
              f.addListener(inFlightUploads::decrementAndGet, MoreExecutors.directExecutor());
              return f;
            })
        .when(combinedCache)
        .uploadFile(any(), any(), any());

    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(combinedCache, RemoteBuildEventUploadMode.ALL, maxConcurrency);

    ListenableFuture<PathConverter> uploadFuture = artifactUploader.upload(filesToUpload);

    // Initial subscription should only request maxConcurrency uploads concurrently.
    assertThat(inFlightUploads.get()).isEqualTo(maxConcurrency);

    // Complete futures one by one to allow subsequent uploads to proceed.
    while (true) {
      SettableFuture<Void> toComplete = null;
      synchronized (futures) {
        for (SettableFuture<Void> f : futures) {
          if (!f.isDone()) {
            toComplete = f;
            break;
          }
        }
      }
      if (toComplete == null) {
        break;
      }
      toComplete.set(null);
    }

    PathConverter pathConverter = uploadFuture.get();
    assertThat(pathConverter).isNotNull();
    assertThat(maxInFlightUploads.get()).isEqualTo(maxConcurrency);

    artifactUploader.release();
    assertThat(combinedCache.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void uploadsWithDefaultConcurrency_unbounded() throws Exception {
    int numUploads = 5;
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/unbounded_concurrency_file" + i);
      FileSystemUtils.writeContent(file, new byte[] {(byte) i});
      filesToUpload.put(
          file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    }

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));

    AtomicInteger inFlightUploads = new AtomicInteger(0);
    AtomicInteger maxInFlightUploads = new AtomicInteger(0);
    List<SettableFuture<Void>> futures = new ArrayList<>();

    doAnswer(
            invocation -> {
              int current = inFlightUploads.incrementAndGet();
              maxInFlightUploads.accumulateAndGet(current, Math::max);
              SettableFuture<Void> f = SettableFuture.create();
              synchronized (futures) {
                futures.add(f);
              }
              f.addListener(inFlightUploads::decrementAndGet, MoreExecutors.directExecutor());
              return f;
            })
        .when(combinedCache)
        .uploadFile(any(), any(), any());

    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(
            combinedCache, RemoteBuildEventUploadMode.ALL, /* maximumOpenFiles= */ -1);

    ListenableFuture<PathConverter> uploadFuture = artifactUploader.upload(filesToUpload);

    // Unbounded concurrency should initiate all uploads concurrently.
    assertThat(inFlightUploads.get()).isEqualTo(numUploads);

    synchronized (futures) {
      for (SettableFuture<Void> f : futures) {
        f.set(null);
      }
    }

    PathConverter pathConverter = uploadFuture.get();
    assertThat(pathConverter).isNotNull();
    assertThat(maxInFlightUploads.get()).isEqualTo(numUploads);

    artifactUploader.release();
    assertThat(combinedCache.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void directory_notUploaded() throws Exception {
    Path dir = fs.getPath("/dir");
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(
        dir, new LocalFile(dir, LocalFileType.OUTPUT_DIRECTORY, /* artifactMetadata= */ null));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(dir)).isNull();
    artifactUploader.release();
  }

  @Test
  public void symlink_notUploaded() throws Exception {
    Path sym = fs.getPath("/sym");
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(
        sym, new LocalFile(sym, LocalFileType.OUTPUT_SYMLINK, /* artifactMetadata= */ null));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(sym)).isNull();
    artifactUploader.release();
  }

  @Test
  public void customHashFunction_uploaded() throws Exception {
    assumeNotNull(BazelHashFunctions.BLAKE3);

    FileSystem fs = new InMemoryFileSystem(new JavaClock(), BazelHashFunctions.BLAKE3);
    Path file = fs.getPath("/file");
    FileSystemUtils.createEmptyFile(file);
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(
        file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    String hash = BaseEncoding.base16().lowerCase().encode(file.getDigest());
    long size = file.getFileSize();
    String conversion = pathConverter.apply(file);
    assertThat(conversion)
        .isEqualTo("bytestream://localhost/instance/blobs/blake3/" + hash + "/" + size);
    artifactUploader.release();
  }

  @Test
  public void testOutputs_uploadedIfFiles() throws Exception {
    var successfulFile = fs.getPath("/test.file.passed");
    FileSystemUtils.createEmptyFile(successfulFile);
    var successfulDir = fs.getPath("/test.dir.passed");
    successfulDir.createDirectory();
    var failedFile = fs.getPath("/test.file.failed");
    FileSystemUtils.createEmptyFile(failedFile);
    var failedDir = fs.getPath("/test.dir.failed");
    failedDir.createDirectory();
    var filesToUpload =
        ImmutableMap.of(
            successfulFile,
            new LocalFile(
                successfulFile, LocalFileType.SUCCESSFUL_TEST_OUTPUT, /* artifactMetadata= */ null),
            failedFile,
            new LocalFile(
                failedFile, LocalFileType.FAILED_TEST_OUTPUT, /* artifactMetadata= */ null),
            successfulDir,
            new LocalFile(
                successfulDir, LocalFileType.SUCCESSFUL_TEST_OUTPUT, /* artifactMetadata= */ null),
            failedDir,
            new LocalFile(
                failedDir, LocalFileType.FAILED_TEST_OUTPUT, /* artifactMetadata= */ null));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(successfulFile)).isNotNull();
    assertThat(pathConverter.apply(failedFile)).isNotNull();
    assertThat(pathConverter.apply(successfulDir)).isNull();
    assertThat(pathConverter.apply(failedDir)).isNull();
    assertThat(eventHandler.getEvents()).isEmpty();
    artifactUploader.release();
  }

  @Test
  public void someUploadsFail_succeedsWithWarningMessages() throws Exception {
    // Test that if one of multiple file uploads fails, the upload future succeeds but the
    // error is reported correctly.

    int numUploads = 10;
    Map<HashCode, byte[]> blobsByHash = new HashMap<>();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    Random rand = new Random();
    for (int i = 0; i < numUploads; i++) {
      Path file = fs.getPath("/file" + i);
      int blobSize = rand.nextInt(100) + 1;
      byte[] blob = new byte[blobSize];
      rand.nextBytes(blob);
      FileSystemUtils.writeContent(file, blob);
      blobsByHash.put(HashCode.fromString(DIGEST_UTIL.compute(file).getHash()), blob);
      filesToUpload.put(
          file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    }
    String hashOfBlobThatShouldFail = blobsByHash.keySet().iterator().next().toString();
    serviceRegistry.addService(
        new MaybeFailOnceUploadService(blobsByHash) {
          @Override
          public StreamObserver<WriteRequest> write(StreamObserver<WriteResponse> response) {
            StreamObserver<WriteRequest> delegate = super.write(response);
            return new StreamObserver<>() {
              private boolean failed;

              @Override
              public void onNext(WriteRequest value) {
                if (value.getResourceName().contains(hashOfBlobThatShouldFail)) {
                  response.onError(Status.CANCELLED.asException());
                  failed = true;
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
                if (failed) {
                  return;
                }
                delegate.onCompleted();
              }
            };
          }
        });

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();

    assertThat(eventHandler.getEvents()).isNotEmpty();
    assertThat(eventHandler.getEvents().get(0).getMessage())
        .contains("Uploading BEP referenced local file /file");
    for (Path file : filesToUpload.keySet()) {
      String hash = BaseEncoding.base16().lowerCase().encode(file.getDigest());
      if (hash.equals(hashOfBlobThatShouldFail)) {
        // In ALL mode a file whose upload failed is reported with a file:// URI.
        assertThat(pathConverter.apply(file)).isEqualTo("file://" + file.getPathString());
      } else {
        assertThat(pathConverter.apply(file))
            .isEqualTo("bytestream://localhost/instance/blobs/" + hash + "/" + file.getFileSize());
      }
    }

    artifactUploader.release();

    assertThat(combinedCache.refCnt()).isEqualTo(0);
    assertThat(refCntChannel.isShutdown()).isTrue();
  }

  @Test
  public void remoteFileShouldNotBeUploaded_actionFs() throws Exception {
    // Test that we don't attempt to upload remotely stored file but convert the remote path
    // to a bytestream:// URI.

    // arrange

    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));
    RemoteActionInputFetcher actionInputFetcher = mock(RemoteActionInputFetcher.class);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

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
    LocalFile file =
        new LocalFile(remotePath, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null);

    // act

    PathConverter pathConverter = artifactUploader.upload(ImmutableMap.of(remotePath, file)).get();

    FileArtifactValue metadata = outputs.getInputMetadata(artifact);
    Digest digest = DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());

    // assert

    String conversion = pathConverter.apply(remotePath);
    assertThat(conversion)
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    verify(combinedCache, times(0)).uploadFile(any(), any(), any());
    verify(combinedCache, times(0)).uploadBlob(any(), any(), any(ByteString.class));
    verify(combinedCache, times(0)).uploadBlob(any(), any(), any(Blob.class));
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
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    doAnswer(invocationOnMock -> Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    // act
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(
            remoteFile,
            new LocalFile(remoteFile, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
            localFile,
            new LocalFile(localFile, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));
    PathConverter pathConverter = artifactUploader.upload(files).get();

    // assert
    verify(digestQuerier).findMissingDigests(any(), any());
    verify(combinedCache).uploadFile(any(), eq(localDigest), any());
    assertThat(pathConverter.apply(remoteFile)).contains(remoteDigest.getHash());
    assertThat(pathConverter.apply(localFile)).contains(localDigest.getHash());
  }

  @Test
  public void fileWithMetadata_digestReusedAndFileNotRead() throws Exception {
    // arrange
    byte[] blob = "contents of a file that is not present locally".getBytes(StandardCharsets.UTF_8);
    Digest digest = DIGEST_UTIL.compute(blob);
    Path file = fs.getPath("/file");
    FileArtifactValue metadata =
        FileArtifactValue.createForVirtualActionInput(
            HashCode.fromString(digest.getHash()).asBytes(), digest.getSizeBytes());

    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of(digest)));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    // act
    PathConverter pathConverter =
        artifactUploader
            .upload(ImmutableMap.of(file, new LocalFile(file, LocalFileType.OUTPUT_FILE, metadata)))
            .get();

    // assert
    verify(digestQuerier).findMissingDigests(any(), any());
    verify(combinedCache, times(0)).uploadFile(any(), any(), any());
    assertThat(pathConverter.apply(file))
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void fileWithRemoteMetadata_notQueriedOrUploaded() throws Exception {
    // arrange
    byte[] blob = "contents of a remote file".getBytes(StandardCharsets.UTF_8);
    Digest digest = DIGEST_UTIL.compute(blob);
    Path file = fs.getPath("/file");
    FileArtifactValue metadata =
        FileArtifactValue.createForRemoteFile(
            HashCode.fromString(digest.getHash()).asBytes(),
            digest.getSizeBytes(),
            /* locationIndex= */ 1);

    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of()));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    // act
    PathConverter pathConverter =
        artifactUploader
            .upload(ImmutableMap.of(file, new LocalFile(file, LocalFileType.OUTPUT_FILE, metadata)))
            .get();

    // assert
    verify(digestQuerier, times(0)).findMissingDigests(any(), any());
    verify(combinedCache, times(0)).uploadFile(any(), any(), any());
    assertThat(pathConverter.apply(file))
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void fileWithMetadata_minimalMode_digestReusedAndFileNotRead() throws Exception {
    // arrange
    byte[] blob = "contents of a file that is not present locally".getBytes(StandardCharsets.UTF_8);
    Digest digest = DIGEST_UTIL.compute(blob);
    Path file = fs.getPath("/file");
    FileArtifactValue metadata =
        FileArtifactValue.createForVirtualActionInput(
            HashCode.fromString(digest.getHash()).asBytes(), digest.getSizeBytes());

    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of()));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(combinedCache, RemoteBuildEventUploadMode.MINIMAL);

    // act
    PathConverter pathConverter =
        artifactUploader
            .upload(ImmutableMap.of(file, new LocalFile(file, LocalFileType.OUTPUT_FILE, metadata)))
            .get();

    // assert
    verify(digestQuerier, times(0)).findMissingDigests(any(), any());
    verify(combinedCache, times(0)).uploadFile(any(), any(), any());
    assertThat(pathConverter.apply(file))
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void fileWithDirectoryMetadata_notUploaded() throws Exception {
    Path dir = fs.getPath("/dir");
    FileArtifactValue metadata = FileArtifactValue.createForDirectoryWithMtime(0);
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(dir, new LocalFile(dir, LocalFileType.SUCCESSFUL_TEST_OUTPUT, metadata));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(dir)).isNull();
    assertThat(eventHandler.getEvents()).isEmpty();
    artifactUploader.release();
  }

  @Test
  public void fileWithSymlinkMetadata_notUploaded() throws Exception {
    Path sym = fs.getPath("/sym");
    sym.createSymbolicLink(PathFragment.create("target"));
    FileArtifactValue metadata = FileArtifactValue.createForUnresolvedSymlink(sym);
    sym.delete();
    Map<Path, LocalFile> filesToUpload = new HashMap<>();
    filesToUpload.put(sym, new LocalFile(sym, LocalFileType.OUTPUT_FILE, metadata));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter = artifactUploader.upload(filesToUpload).get();
    assertThat(pathConverter.apply(sym)).isNull();
    assertThat(eventHandler.getEvents()).isEmpty();
    artifactUploader.release();
  }

  @Test
  public void pathConverter_storesPrecomputedUri() throws Exception {
    // The converter is retained by the build event until the event has been sent to the BES
    // backend, so it must store the final URI rather than the metadata needed to compute it.
    Path file = fs.getPath("/file");
    FileSystemUtils.writeContent(file, new byte[] {1, 2, 3});
    Digest digest = DIGEST_UTIL.compute(file);
    StaticMissingDigestsFinder digestQuerier =
        new StaticMissingDigestsFinder(ImmutableSet.of(digest));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier, digestQuerier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter =
        artifactUploader
            .upload(
                ImmutableMap.of(
                    file,
                    new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null)))
            .get();

    String uri = pathConverter.apply(file);
    assertThat(uri)
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    assertThat(pathConverter.apply(file)).isSameInstanceAs(uri);
    artifactUploader.release();
  }

  @Test
  public void pathConverter_matchesPathsByFragmentAcrossFileSystems() throws Exception {
    // Outputs of remotely executed actions are referenced through per-action file systems. The
    // converter must not depend on (and thus not retain) the file system of the declared paths.
    Path remoteFile = fs.getPath("/remote-file");
    FileSystemUtils.writeContent(remoteFile, StandardCharsets.UTF_8, "hello world");
    Digest remoteDigest = DIGEST_UTIL.compute(remoteFile);
    Path dir = fs.getPath("/dir");
    dir.createDirectory();
    StaticMissingDigestsFinder digestQuerier =
        new StaticMissingDigestsFinder(ImmutableSet.of(remoteDigest));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier, digestQuerier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter =
        artifactUploader
            .upload(
                ImmutableMap.of(
                    remoteFile,
                    new LocalFile(
                        remoteFile, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
                    dir,
                    new LocalFile(
                        dir, LocalFileType.OUTPUT_DIRECTORY, /* artifactMetadata= */ null)))
            .get();

    FileSystem otherFs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    Path remoteFileOnOtherFs = otherFs.getPath(remoteFile.asFragment());
    assertThat(remoteFileOnOtherFs).isNotEqualTo(remoteFile);
    assertThat(pathConverter.apply(remoteFileOnOtherFs))
        .isEqualTo(pathConverter.apply(remoteFile));
    assertThat(pathConverter.apply(remoteFileOnOtherFs))
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + remoteDigest.getHash()
                + "/"
                + remoteDigest.getSizeBytes());
    assertThat(pathConverter.apply(otherFs.getPath(dir.asFragment()))).isNull();
    artifactUploader.release();
  }

  @Test
  public void pathConverter_doesNotRetainPathMetadata() throws Exception {
    // The converter of a pending build event may be retained for a long time. It must only hold
    // the resulting URI strings, not the per-file metadata (Digest, DigestFunction) or the Path
    // objects (which pin their FileSystem) that were used to compute them.
    Path remoteFile = fs.getPath("/remote-file");
    FileSystemUtils.writeContent(remoteFile, StandardCharsets.UTF_8, "hello world");
    Digest remoteDigest = DIGEST_UTIL.compute(remoteFile);
    Path localFile = fs.getPath("/local-file");
    FileSystemUtils.writeContent(localFile, StandardCharsets.UTF_8, "foo bar");
    Path dir = fs.getPath("/dir");
    dir.createDirectory();
    StaticMissingDigestsFinder digestQuerier =
        new StaticMissingDigestsFinder(ImmutableSet.of(remoteDigest));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    doAnswer(invocationOnMock -> Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter =
        artifactUploader
            .upload(
                ImmutableMap.of(
                    remoteFile,
                    new LocalFile(
                        remoteFile, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
                    localFile,
                    new LocalFile(
                        localFile, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null),
                    dir,
                    new LocalFile(
                        dir, LocalFileType.OUTPUT_DIRECTORY, /* artifactMetadata= */ null)))
            .get();

    assertThat(pathConverter.apply(remoteFile)).contains(remoteDigest.getHash());
    assertThat(pathConverter.apply(localFile)).startsWith("bytestream://");
    assertThat(pathConverter.apply(dir)).isNull();
    ImmutableSet<String> retainedClasses =
        GraphLayout.parseInstance(pathConverter).getClasses().stream()
            .map(Class::getName)
            .collect(toImmutableSet());
    assertThat(retainedClasses)
        .containsNoneOf(
            ByteStreamBuildEventArtifactUploader.class.getName() + "$PathMetadata",
            Digest.class.getName(),
            DigestFunction.Value.class.getName(),
            Path.class.getName(),
            fs.getClass().getName());
    artifactUploader.release();
  }

  @Test
  public void pathConverter_undeclaredPath_throws() throws Exception {
    Path file = fs.getPath("/file");
    FileSystemUtils.writeContent(file, new byte[] {1, 2, 3});
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));
    doAnswer(invocationOnMock -> Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter pathConverter =
        artifactUploader
            .upload(
                ImmutableMap.of(
                    file,
                    new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null)))
            .get();

    assertThat(pathConverter.apply(file)).isNotNull();
    IllegalStateException e =
        assertThrows(
            IllegalStateException.class, () -> pathConverter.apply(fs.getPath("/undeclared")));
    assertThat(e).hasMessageThat().contains("Illegal file reference: '/undeclared'");
    artifactUploader.release();
  }

  @Test
  public void sameFileAcrossEvents_queriedAndUploadedOnce_sharesUri() throws Exception {
    // Every TargetCompleteEvent references its full transitive output set, so the same file is
    // passed to upload() once per depending target. Within a build it only needs handling once.
    Path file = fs.getPath("/shared-file");
    FileSystemUtils.writeContent(file, StandardCharsets.UTF_8, "shared contents");
    Digest digest = DIGEST_UTIL.compute(file);
    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of()));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    doAnswer(invocationOnMock -> Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(
            file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));

    PathConverter first = artifactUploader.upload(files).get();
    PathConverter second = artifactUploader.upload(files).get();

    verify(digestQuerier, times(1)).findMissingDigests(any(), any());
    verify(combinedCache, times(1)).uploadFile(any(), eq(digest), any());
    String uri = first.apply(file);
    assertThat(uri)
        .isEqualTo(
            "bytestream://localhost/instance/blobs/"
                + digest.getHash()
                + "/"
                + digest.getSizeBytes());
    // All pending events share a single URI string per file.
    assertThat(second.apply(file)).isSameInstanceAs(uri);
    assertThat(eventHandler.getEvents()).isEmpty();
    artifactUploader.release();
  }

  @Test
  public void sameFileAcrossEvents_minimalMode_sharesUriWithoutQueryOrUpload() throws Exception {
    byte[] blob = "contents of a file that is not present locally".getBytes(StandardCharsets.UTF_8);
    Digest digest = DIGEST_UTIL.compute(blob);
    Path file = fs.getPath("/file");
    FileArtifactValue metadata =
        FileArtifactValue.createForVirtualActionInput(
            HashCode.fromString(digest.getHash()).asBytes(), digest.getSizeBytes());
    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of()));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(combinedCache, RemoteBuildEventUploadMode.MINIMAL);
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(file, new LocalFile(file, LocalFileType.OUTPUT_FILE, metadata));

    PathConverter first = artifactUploader.upload(files).get();
    PathConverter second = artifactUploader.upload(files).get();

    verify(digestQuerier, times(0)).findMissingDigests(any(), any());
    verify(combinedCache, times(0)).uploadFile(any(), any(), any());
    assertThat(first.apply(file)).contains(digest.getHash());
    assertThat(second.apply(file)).isSameInstanceAs(first.apply(file));
    artifactUploader.release();
  }

  @Test
  public void sameFileAcrossEvents_differentContent_notShared() throws Exception {
    Path file = fs.getPath("/file");
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));
    doAnswer(invocationOnMock -> Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(
            file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));

    FileSystemUtils.writeContent(file, StandardCharsets.UTF_8, "version 1");
    Digest digest1 = DIGEST_UTIL.compute(file);
    PathConverter first = artifactUploader.upload(files).get();
    FileSystemUtils.writeContent(file, StandardCharsets.UTF_8, "version 2");
    Digest digest2 = DIGEST_UTIL.compute(file);
    PathConverter second = artifactUploader.upload(files).get();

    verify(combinedCache, times(1)).uploadFile(any(), eq(digest1), any());
    verify(combinedCache, times(1)).uploadFile(any(), eq(digest2), any());
    assertThat(first.apply(file)).contains(digest1.getHash());
    assertThat(second.apply(file)).contains(digest2.getHash());
    artifactUploader.release();
  }

  @Test
  public void buildToolLogAcrossEvents_notShared() throws Exception {
    Path log = fs.getPath("/command.log");
    FileSystemUtils.writeContent(log, StandardCharsets.UTF_8, "log contents");
    Digest digest = DIGEST_UTIL.compute(log);
    StaticMissingDigestsFinder digestQuerier =
        Mockito.spy(new StaticMissingDigestsFinder(ImmutableSet.of()));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier, digestQuerier));
    doAnswer(invocationOnMock -> Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(log, new LocalFile(log, LocalFileType.LOG, /* artifactMetadata= */ null));

    PathConverter first = artifactUploader.upload(files).get();
    PathConverter second = artifactUploader.upload(files).get();

    verify(digestQuerier, times(2)).findMissingDigests(any(), any());
    verify(combinedCache, times(2)).uploadFile(any(), eq(digest), any());
    assertThat(first.apply(log)).isEqualTo(second.apply(log));
    artifactUploader.release();
  }

  @Test
  public void failedUploadAcrossEvents_notShared_retriedByNextEvent() throws Exception {
    Path file = fs.getPath("/flaky-file");
    FileSystemUtils.writeContent(file, StandardCharsets.UTF_8, "contents");
    Digest digest = DIGEST_UTIL.compute(file);
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));
    AtomicInteger uploadAttempts = new AtomicInteger();
    doAnswer(
            invocationOnMock ->
                uploadAttempts.getAndIncrement() == 0
                    ? Futures.immediateFailedFuture(new IOException("upload failed"))
                    : Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(
            file, new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null));

    PathConverter first = artifactUploader.upload(files).get();
    PathConverter second = artifactUploader.upload(files).get();

    verify(combinedCache, times(2)).uploadFile(any(), eq(digest), any());
    // The failed upload falls back to file:// and is not remembered; the next event retries.
    assertThat(first.apply(file)).isEqualTo("file://" + file.getPathString());
    assertThat(second.apply(file)).contains(digest.getHash());
    assertThat(eventHandler.getEvents()).hasSize(1);
    artifactUploader.release();
  }

  @Test
  public void minimalMode_failedTestLogUpload_retriedByNextEvent() throws Exception {
    // In MINIMAL mode a file is reported with a bytestream:// URI whether or not its upload
    // succeeded, so a failed upload must not be remembered as present for later events.
    Path log = fs.getPath("/execroot/bazel-out/k8-fastbuild/testlogs/foo/test.log");
    log.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(log, StandardCharsets.UTF_8, "test log");
    Digest digest = DIGEST_UTIL.compute(log);
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = spy(newCombinedCache(refCntChannel, retrier));
    AtomicInteger uploadAttempts = new AtomicInteger();
    doAnswer(
            invocationOnMock ->
                uploadAttempts.getAndIncrement() == 0
                    ? Futures.immediateFailedFuture(new IOException("upload failed"))
                    : Futures.immediateFuture(null))
        .when(combinedCache)
        .uploadFile(any(), any(), any());
    ByteStreamBuildEventArtifactUploader artifactUploader =
        newArtifactUploader(combinedCache, RemoteBuildEventUploadMode.MINIMAL);
    ImmutableMap<Path, LocalFile> files =
        ImmutableMap.of(
            log,
            new LocalFile(log, LocalFileType.SUCCESSFUL_TEST_OUTPUT, /* artifactMetadata= */ null));

    PathConverter first = artifactUploader.upload(files).get();
    PathConverter second = artifactUploader.upload(files).get();
    PathConverter third = artifactUploader.upload(files).get();

    // The first upload failed and was retried by the second event; the third event then hit the
    // cache.
    verify(combinedCache, times(2)).uploadFile(any(), eq(digest), any());
    assertThat(first.apply(log)).contains(digest.getHash());
    assertThat(second.apply(log)).contains(digest.getHash());
    assertThat(third.apply(log)).isSameInstanceAs(second.apply(log));
    assertThat(eventHandler.getEvents()).hasSize(1);
    artifactUploader.release();
  }

  @Test
  public void pathConvertersAcrossEvents_shareKeysAndUris() throws Exception {
    // Every event resolves its paths through its own file system, so unless the converters
    // intern their keys, each pending event retains its own copy of every referenced path.
    Path file = fs.getPath("/shared-file");
    FileSystemUtils.writeContent(file, StandardCharsets.UTF_8, "shared contents");
    Digest digest = DIGEST_UTIL.compute(file);
    FileSystem otherFs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    Path fileOnOtherFs = otherFs.getPath(file.getPathString());
    FileSystemUtils.writeContent(fileOnOtherFs, StandardCharsets.UTF_8, "shared contents");
    assertThat(fileOnOtherFs.asFragment()).isNotSameInstanceAs(file.asFragment());
    StaticMissingDigestsFinder digestQuerier =
        new StaticMissingDigestsFinder(ImmutableSet.of(digest));
    RemoteRetrier retrier =
        TestUtils.newRemoteRetrier(
            () -> new FixedBackoff(1, 0), (e) -> Result.TRANSIENT_FAILURE, retryService);
    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(channelConnectionFactory);
    CombinedCache combinedCache = newCombinedCache(refCntChannel, retrier, digestQuerier);
    ByteStreamBuildEventArtifactUploader artifactUploader = newArtifactUploader(combinedCache);

    PathConverter first =
        artifactUploader
            .upload(
                ImmutableMap.of(
                    file,
                    new LocalFile(file, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null)))
            .get();
    PathConverter second =
        artifactUploader
            .upload(
                ImmutableMap.of(
                    fileOnOtherFs,
                    new LocalFile(
                        fileOnOtherFs, LocalFileType.OUTPUT_FILE, /* artifactMetadata= */ null)))
            .get();

    assertThat(second.apply(fileOnOtherFs)).isSameInstanceAs(first.apply(file));
    // Both converters together retain a single PathFragment (the key) and two Strings (the path
    // and the URI).
    GraphLayout layout = GraphLayout.parseInstance(first, second);
    long pathFragments =
        layout.getClasses().stream()
            .filter(PathFragment.class::isAssignableFrom)
            .mapToLong(cls -> layout.getClassCounts().count(cls))
            .sum();
    assertThat(pathFragments).isEqualTo(1);
    assertThat(layout.getClassCounts().count(String.class)).isEqualTo(2);
    artifactUploader.release();
  }

  /** Returns a remote artifact and puts its metadata into the action input map. */
  private Artifact createRemoteArtifact(
      String pathFragment, String contents, ActionInputMap inputs) {
    Path p = outputRoot.getRoot().asPath().getRelative(pathFragment);
    Artifact a = ActionsTestUtil.createArtifact(outputRoot, p);
    byte[] b = contents.getBytes(StandardCharsets.UTF_8);
    HashCode h = HashCode.fromString(DIGEST_UTIL.compute(b).getHash());
    FileArtifactValue f =
        FileArtifactValue.createForRemoteFile(h.asBytes(), b.length, /* locationIndex= */ 1);
    inputs.put(a, f);
    return a;
  }

  private static CombinedCache newCombinedCache(
      ReferenceCountedChannel channel, RemoteRetrier retrier) {
    return newCombinedCache(channel, retrier, new AllMissingDigestsFinder());
  }

  private static CombinedCache newCombinedCache(
      ReferenceCountedChannel channel,
      RemoteRetrier retrier,
      MissingDigestsFinder missingDigestsFinder) {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.setRemoteInstanceName("instance");
    GrpcCacheClient cacheClient =
        spy(
            new GrpcCacheClient(
                channel,
                CallCredentialsProvider.NO_CREDENTIALS,
                remoteOptions,
                retrier,
                DIGEST_UTIL));
    doAnswer(
            invocationOnMock ->
                missingDigestsFinder.findMissingDigests(
                    invocationOnMock.getArgument(0), invocationOnMock.getArgument(1)))
        .when(cacheClient)
        .findMissingDigests(any(), any());

    return new CombinedCache(
        cacheClient,
        /* diskCacheClient= */ null,
        /* symlinkTemplate= */ null,
        DIGEST_UTIL,
        /* chunkingFunction= */ null,
        new ChunkLocationMap());
  }

  private ByteStreamBuildEventArtifactUploader newArtifactUploader(CombinedCache combinedCache) {
    return newArtifactUploader(
        combinedCache, RemoteBuildEventUploadMode.ALL, /* maximumOpenFiles= */ -1);
  }

  private ByteStreamBuildEventArtifactUploader newArtifactUploader(
      CombinedCache combinedCache, RemoteBuildEventUploadMode remoteBuildEventUploadMode) {
    return newArtifactUploader(
        combinedCache, remoteBuildEventUploadMode, /* maximumOpenFiles= */ -1);
  }

  private ByteStreamBuildEventArtifactUploader newArtifactUploader(
      CombinedCache combinedCache,
      RemoteBuildEventUploadMode remoteBuildEventUploadMode,
      int maximumOpenFiles) {

    return new ByteStreamBuildEventArtifactUploader(
        MoreExecutors.directExecutor(),
        reporter,
        /* verboseFailures= */ true,
        combinedCache,
        /* remoteInstanceName= */ "",
        /* remoteBytestreamUriPrefix= */ "localhost/instance",
        /* buildRequestId= */ "none",
        /* commandId= */ "none",
        SyscallCache.NO_CACHE,
        remoteBuildEventUploadMode,
        maximumOpenFiles);
  }

  private static class StaticMissingDigestsFinder implements MissingDigestsFinder {

    private final ImmutableSet<Digest> knownDigests;

    public StaticMissingDigestsFinder(ImmutableSet<Digest> knownDigests) {
      this.knownDigests = knownDigests;
    }

    @Override
    public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
        RemoteActionExecutionContext context, Iterable<Digest> digests) {
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

    @Override
    public ListenableFuture<ImmutableSet<Digest>> findMissingDigests(
        RemoteActionExecutionContext context, Iterable<Digest> digests) {
      return Futures.immediateFuture(ImmutableSet.copyOf(digests));
    }
  }
}
