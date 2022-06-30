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
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Tests for {@link RemoteCache}. */
@RunWith(JUnit4.class)
public class RemoteCacheTest {
  private RemoteActionExecutionContext context;
  private FileSystem fs;
  private Path execRoot;
  ArtifactRoot artifactRoot;
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private FakeActionInputFileCache fakeFileCache;

  private ListeningScheduledExecutorService retryService;

  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata("none", "none", "action-id", null);
    Spawn spawn =
        new SimpleSpawn(
            new FakeOwner("foo", "bar", "//dummy:label"),
            /*arguments=*/ ImmutableList.of(),
            /*environment=*/ ImmutableMap.of(),
            /*executionInfo=*/ ImmutableMap.of(),
            /*inputs=*/ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /*outputs=*/ ImmutableSet.of(),
            ResourceSet.ZERO);
    context = RemoteActionExecutionContext.create(spawn, metadata);
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/main");
    execRoot.createDirectoryAndParents();
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.Output, "outputs");
    artifactRoot.getRoot().asPath().createDirectoryAndParents();
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @After
  public void afterEverything() throws InterruptedException {
    retryService.shutdownNow();
    retryService.awaitTermination(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
  }

  @Test
  public void testDownloadEmptyBlobAndFile() throws Exception {
    // Test that downloading an empty BLOB/file does not try to perform a download.

    // arrange
    Path file = fs.getPath("/execroot/file");
    RemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);

    // act and assert
    assertThat(getFromFuture(remoteCache.downloadBlob(context, emptyDigest))).isEmpty();

    try (OutputStream out = file.getOutputStream()) {
      getFromFuture(remoteCache.downloadFile(context, file, emptyDigest));
    }
    assertThat(file.exists()).isTrue();
    assertThat(file.getFileSize()).isEqualTo(0);
  }

  @Test
  public void downloadFile_cancelled_cancelDownload() throws Exception {
    // Test that if a download future is cancelled, the download itself is also cancelled.

    // arrange
    RemoteCacheClient remoteCacheClient = mock(RemoteCacheClient.class);
    SettableFuture<Void> future = SettableFuture.create();
    // Return a future that never completes
    doAnswer(invocationOnMock -> future).when(remoteCacheClient).downloadBlob(any(), any(), any());
    RemoteCache remoteCache = newRemoteCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");

    // act
    ListenableFuture<Void> download = remoteCache.downloadFile(context, file, digest);
    download.cancel(/* mayInterruptIfRunning= */ true);

    // assert
    assertThat(future.isCancelled()).isTrue();
  }

  @Test
  public void downloadOutErr_empty_doNotPerformDownload() throws Exception {
    // Test that downloading empty stdout/stderr does not try to perform a download.

    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);
    ActionResult.Builder result = ActionResult.newBuilder();
    result.setStdoutDigest(emptyDigest);
    result.setStderrDigest(emptyDigest);

    waitForBulkTransfer(
        remoteCache.downloadOutErr(
            context,
            result.build(),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr"))),
        true);

    assertThat(remoteCache.getNumSuccessfulDownloads()).isEqualTo(0);
    assertThat(remoteCache.getNumFailedDownloads()).isEqualTo(0);
  }

  @Test
  public void testDownloadFileWithSymlinkTemplate() throws Exception {
    // Test that when a symlink template is provided, we don't actually download files to disk.
    // Instead, a symbolic link should be created that points to a location where the file may
    // actually be found. That location could, for example, be backed by a FUSE file system that
    // exposes the Content Addressable Storage.

    // arrange
    final ConcurrentMap<Digest, byte[]> cas = new ConcurrentHashMap<>();

    Digest helloDigest = digestUtil.computeAsUtf8("hello-contents");
    cas.put(helloDigest, "hello-contents".getBytes(StandardCharsets.UTF_8));

    Path file = fs.getPath("/execroot/symlink-to-file");
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteDownloadSymlinkTemplate = "/home/alice/cas/{hash}-{size_bytes}";
    RemoteCache remoteCache = new InMemoryRemoteCache(cas, options, digestUtil);

    // act
    getFromFuture(remoteCache.downloadFile(context, file, helloDigest));

    // assert
    assertThat(file.isSymbolicLink()).isTrue();
    assertThat(PathFragment.create(file.readSymbolicLink()))
        .isEqualTo(
            PathFragment.create(
                "/home/alice/cas/a378b939ad2e1d470a9a28b34b0e256b189e85cb236766edc1d46ec3b6ca82e5-14"));
  }

  @Test
  public void upload_emptyBlobAndFile_doNotPerformUpload() throws Exception {
    // Test that uploading an empty BLOB/file does not try to perform an upload.
    InMemoryRemoteCache remoteCache = newRemoteCache();
    Digest emptyDigest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "");
    Path file = execRoot.getRelative("file");

    getFromFuture(remoteCache.uploadBlob(context, emptyDigest, ByteString.EMPTY));
    assertThat(getFromFuture(remoteCache.findMissingDigests(context, ImmutableSet.of(emptyDigest))))
        .containsExactly(emptyDigest);

    getFromFuture(remoteCache.uploadFile(context, emptyDigest, file));
    assertThat(getFromFuture(remoteCache.findMissingDigests(context, ImmutableSet.of(emptyDigest))))
        .containsExactly(emptyDigest);
  }

  @Test
  @SuppressWarnings("FutureReturnValueIgnored")
  public void upload_deduplicationWorks() throws IOException {
    RemoteCacheClient remoteCacheClient = mock(RemoteCacheClient.class);
    AtomicInteger times = new AtomicInteger(0);
    doAnswer(
            invocationOnMock -> {
              times.incrementAndGet();
              return SettableFuture.create();
            })
        .when(remoteCacheClient)
        .uploadFile(any(), any(), any());
    RemoteCache remoteCache = newRemoteCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");

    remoteCache.uploadFile(context, digest, file);
    remoteCache.uploadFile(context, digest, file);

    assertThat(times.get()).isEqualTo(1);
  }

  @Test
  public void upload_failedUploads_doNotDeduplicate() throws Exception {
    AtomicBoolean failRequest = new AtomicBoolean(true);
    InMemoryCacheClient inMemoryCacheClient = new InMemoryCacheClient();
    RemoteCacheClient remoteCacheClient = mock(RemoteCacheClient.class);
    doAnswer(
            invocationOnMock -> {
              if (failRequest.getAndSet(false)) {
                return Futures.immediateFailedFuture(new IOException("Failed"));
              }
              return inMemoryCacheClient.uploadFile(
                  invocationOnMock.getArgument(0),
                  invocationOnMock.getArgument(1),
                  invocationOnMock.getArgument(2));
            })
        .when(remoteCacheClient)
        .uploadFile(any(), any(), any());
    doAnswer(
            invocationOnMock ->
                inMemoryCacheClient.findMissingDigests(
                    invocationOnMock.getArgument(0), invocationOnMock.getArgument(1)))
        .when(remoteCacheClient)
        .findMissingDigests(any(), any());
    RemoteCache remoteCache = newRemoteCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");
    assertThat(getFromFuture(remoteCache.findMissingDigests(context, ImmutableList.of(digest))))
        .containsExactly(digest);

    Exception thrown = null;
    try {
      getFromFuture(remoteCache.uploadFile(context, digest, file));
    } catch (IOException e) {
      thrown = e;
    }
    assertThat(thrown).isNotNull();
    assertThat(thrown).isInstanceOf(IOException.class);
    getFromFuture(remoteCache.uploadFile(context, digest, file));

    assertThat(getFromFuture(remoteCache.findMissingDigests(context, ImmutableList.of(digest))))
        .isEmpty();
  }

  @Test
  public void ensureInputsPresent_interrupted_cancelInProgressUploadTasks() throws Exception {
    // arrange
    InMemoryRemoteCache remoteCache = spy(newRemoteCache());

    CountDownLatch findMissingDigestsCalled = new CountDownLatch(1);
    doAnswer(
            invocationOnMock -> {
              findMissingDigestsCalled.countDown();
              return SettableFuture.create();
            })
        .when(remoteCache)
        .findMissingDigests(any(), any());

    Path path = fs.getPath("/execroot/foo");
    FileSystemUtils.writeContentAsLatin1(path, "bar");
    SortedMap<PathFragment, Path> inputs = new TreeMap<>();
    inputs.put(PathFragment.create("foo"), path);
    MerkleTree merkleTree = MerkleTree.build(inputs, digestUtil);

    CountDownLatch ensureInputsPresentReturned = new CountDownLatch(1);
    Thread thread =
        new Thread(
            () -> {
              try {
                remoteCache.ensureInputsPresent(context, merkleTree, ImmutableMap.of(), false);
              } catch (IOException | InterruptedException ignored) {
                // ignored
              } finally {
                ensureInputsPresentReturned.countDown();
              }
            });

    // act
    thread.start();
    findMissingDigestsCalled.await();
    assertThat(remoteCache.casUploadCache.getInProgressTasks()).isNotEmpty();

    thread.interrupt();
    ensureInputsPresentReturned.await();

    // assert
    assertThat(remoteCache.casUploadCache.getInProgressTasks()).isEmpty();
  }

  @Test
  public void shutdownNow_cancelInProgressUploads() throws Exception {
    RemoteCacheClient remoteCacheClient = mock(RemoteCacheClient.class);
    // Return a future that never completes
    doAnswer(invocationOnMock -> SettableFuture.create())
        .when(remoteCacheClient)
        .uploadFile(any(), any(), any());
    RemoteCache remoteCache = newRemoteCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");

    ListenableFuture<Void> upload = remoteCache.uploadFile(context, digest, file);
    assertThat(remoteCache.casUploadCache.getInProgressTasks()).contains(digest);
    remoteCache.shutdownNow();

    assertThat(upload.isCancelled()).isTrue();
  }

  private InMemoryRemoteCache newRemoteCache() {
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    return new InMemoryRemoteCache(options, digestUtil);
  }

  private RemoteCache newRemoteCache(RemoteCacheClient remoteCacheClient) {
    return new RemoteCache(remoteCacheClient, Options.getDefaults(RemoteOptions.class), digestUtil);
  }
}
