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
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.FastCdc2020Params;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.authandtls.CallCredentialsProvider;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.SpawnCheckingCacheEvent;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.Blob;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.merkletree.MerkleTree;
import com.google.devtools.build.lib.remote.merkletree.MerkleTreeComputer;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.FakeSpawnExecutionContext;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.RxNoGlobalErrorsRule;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
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
import java.util.Deque;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.MockitoAnnotations;

/** Tests for {@link CombinedCache}. */
@RunWith(JUnit4.class)
public class CombinedCacheTest {
  @Rule public final RxNoGlobalErrorsRule rxNoGlobalErrorsRule = new RxNoGlobalErrorsRule();

  private RequestMetadata metadata;
  private RemoteActionExecutionContext remoteActionExecutionContext;
  private FileSystem fs;
  private Path execRoot;
  ArtifactRoot artifactRoot;
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private final MerkleTreeComputer merkleTreeComputer =
      new MerkleTreeComputer(
          digestUtil,
          /* remoteExecutionCache= */ null,
          "buildRequestId",
          "commandId",
          TestConstants.WORKSPACE_NAME);
  private FakeActionInputFileCache fakeFileCache;

  private ListeningScheduledExecutorService retryService;

  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    metadata = TracingMetadataUtils.buildMetadata("none", "none", "action-id", null);
    Spawn spawn =
        new SimpleSpawn(
            new FakeOwner("foo", "bar", "//dummy:label"),
            /* arguments= */ ImmutableList.of(),
            /* environment= */ ImmutableMap.of(),
            /* executionInfo= */ ImmutableMap.of(),
            /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* outputs= */ ImmutableSet.of(),
            ResourceSet.ZERO);
    SpawnExecutionContext spawnExecutionContext = mock(SpawnExecutionContext.class);
    remoteActionExecutionContext =
        RemoteActionExecutionContext.create(spawn, spawnExecutionContext, metadata);
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/main");
    execRoot.createDirectoryAndParents();
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "outputs");
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
    InMemoryCombinedCache combinedCache = newCombinedCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);

    // act and assert
    assertThat(getFromFuture(combinedCache.downloadBlob(remoteActionExecutionContext, emptyDigest)))
        .isEmpty();

    try (OutputStream out = file.getOutputStream()) {
      getFromFuture(combinedCache.downloadFile(remoteActionExecutionContext, file, emptyDigest));
    }
    assertThat(file.exists()).isTrue();
    assertThat(file.getFileSize()).isEqualTo(0);
  }

  @Test
  public void downloadActionResult_reportsSpawnCheckingCacheEvent() throws Exception {
    var combinedCache = newCombinedCache();
    var unused =
        combinedCache.downloadActionResult(
            remoteActionExecutionContext,
            digestUtil.asActionKey(digestUtil.computeAsUtf8("key")),
            /* inlineOutErr= */ false,
            /* inlineOutputFiles= */ ImmutableSet.of());

    verify(remoteActionExecutionContext.getSpawnExecutionContext())
        .report(SpawnCheckingCacheEvent.create("remote-cache"));
  }

  @Test
  public void downloadFile_cancelled_cancelDownload() throws Exception {
    // Test that if a download future is cancelled, the download itself is also cancelled.

    // arrange
    RemoteCacheClient remoteCacheClient = mock(RemoteCacheClient.class);
    SettableFuture<Void> future = SettableFuture.create();
    // Return a future that never completes
    doAnswer(invocationOnMock -> future).when(remoteCacheClient).downloadBlob(any(), any(), any());
    CombinedCache remoteCache = newCombinedCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");

    // act
    ListenableFuture<Void> download =
        remoteCache.downloadFile(remoteActionExecutionContext, file, digest);
    download.cancel(/* mayInterruptIfRunning= */ true);

    // assert
    assertThat(future.isCancelled()).isTrue();
  }

  @Test
  public void downloadOutErr_empty_doNotPerformDownload() throws Exception {
    // Test that downloading empty stdout/stderr does not try to perform a download.

    InMemoryCombinedCache combinedCache = newCombinedCache();
    Digest emptyDigest = digestUtil.compute(new byte[0]);
    ActionResult.Builder result = ActionResult.newBuilder();
    result.setStdoutDigest(emptyDigest);
    result.setStderrDigest(emptyDigest);

    waitForBulkTransfer(
        combinedCache.downloadOutErr(
            remoteActionExecutionContext,
            result.build(),
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr"))));

    assertThat(combinedCache.getNumSuccessfulDownloads()).isEqualTo(0);
    assertThat(combinedCache.getNumFailedDownloads()).isEqualTo(0);
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
    cas.put(helloDigest, "hello-contents".getBytes(UTF_8));

    Path file = fs.getPath("/execroot/symlink-to-file");
    InMemoryCombinedCache combinedCache =
        newCombinedCache(cas, digestUtil, "/home/alice/cas/{hash}-{size_bytes}");

    // act
    getFromFuture(combinedCache.downloadFile(remoteActionExecutionContext, file, helloDigest));

    // assert
    assertThat(file.isSymbolicLink()).isTrue();
    assertThat(file.readSymbolicLink())
        .isEqualTo(
            PathFragment.create(
                "/home/alice/cas/a378b939ad2e1d470a9a28b34b0e256b189e85cb236766edc1d46ec3b6ca82e5-14"));
  }

  @Test
  public void upload_emptyBlobAndFile_doNotPerformUpload() throws Exception {
    // Test that uploading an empty BLOB/file does not try to perform an upload.
    InMemoryCombinedCache combinedCache = newCombinedCache();
    Digest emptyDigest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "");
    Path file = execRoot.getRelative("file");

    getFromFuture(
        combinedCache.uploadBlob(remoteActionExecutionContext, emptyDigest, ByteString.EMPTY));
    assertThat(
            getFromFuture(
                combinedCache.findMissingDigests(
                    remoteActionExecutionContext, ImmutableSet.of(emptyDigest))))
        .containsExactly(emptyDigest);

    getFromFuture(combinedCache.uploadFile(remoteActionExecutionContext, emptyDigest, file));
    assertThat(
            getFromFuture(
                combinedCache.findMissingDigests(
                    remoteActionExecutionContext, ImmutableSet.of(emptyDigest))))
        .containsExactly(emptyDigest);
  }

  @Test
  @SuppressWarnings("FutureReturnValueIgnored")
  public void upload_deduplicationWorks() throws IOException {
    RemoteCacheClient remoteCacheClient = spy(new InMemoryCacheClient());
    AtomicInteger times = new AtomicInteger(0);
    doAnswer(
            invocationOnMock -> {
              times.incrementAndGet();
              return SettableFuture.create();
            })
        .when(remoteCacheClient)
        .uploadBlobImpl(any(), any(), any());
    CombinedCache combinedCache = newCombinedCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");

    var unused1 = combinedCache.uploadFile(remoteActionExecutionContext, digest, file);
    var unused2 = combinedCache.uploadFile(remoteActionExecutionContext, digest, file);

    assertThat(times.get()).isEqualTo(1);
  }

  @Test
  public void upload_failedUploads_doNotDeduplicate() throws Exception {
    AtomicBoolean failRequest = new AtomicBoolean(true);
    RemoteCacheClient remoteCacheClient = spy(new InMemoryCacheClient());
    doAnswer(
            invocationOnMock -> {
              if (failRequest.getAndSet(false)) {
                return Futures.immediateFailedFuture(new IOException("Failed"));
              }
              return invocationOnMock.callRealMethod();
            })
        .when(remoteCacheClient)
        .uploadBlobImpl(any(), any(), any());
    CombinedCache combinedCache = newCombinedCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");
    assertThat(
            getFromFuture(
                combinedCache.findMissingDigests(
                    remoteActionExecutionContext, ImmutableList.of(digest))))
        .containsExactly(digest);

    Exception thrown = null;
    try {
      getFromFuture(combinedCache.uploadFile(remoteActionExecutionContext, digest, file));
    } catch (IOException e) {
      thrown = e;
    }
    assertThat(thrown).isNotNull();
    assertThat(thrown).isInstanceOf(IOException.class);
    getFromFuture(combinedCache.uploadFile(remoteActionExecutionContext, digest, file));

    assertThat(
            getFromFuture(
                combinedCache.findMissingDigests(
                    remoteActionExecutionContext, ImmutableList.of(digest))))
        .isEmpty();
  }

  @Test
  public void ensureInputsPresent_missingInputs_exceptionHasLostInputs() throws Exception {
    RemoteCacheClient cacheProtocol = spy(new InMemoryCacheClient());
    RemoteExecutionCache remoteCache = spy(newRemoteExecutionCache(cacheProtocol));
    remoteActionExecutionContext = RemoteActionExecutionContext.create(metadata);
    remoteCache.setRemotePathChecker(
        (context, path) ->
            immediateFuture(!path.relativeTo(execRoot).equals(PathFragment.create("foo"))));

    Path path = execRoot.getRelative("foo");
    FileSystemUtils.writeContentAsLatin1(path, "bar");
    SortedMap<PathFragment, Path> inputs = new TreeMap<>();
    inputs.put(PathFragment.create("foo"), path);
    var merkleTree = merkleTreeComputer.buildForFiles(inputs);
    path.delete();

    var e =
        assertThrows(
            BulkTransferException.class,
            () ->
                remoteCache.ensureInputsPresent(
                    remoteActionExecutionContext,
                    merkleTree,
                    ImmutableMap.of(),
                    false,
                    new RemotePathResolver.DefaultRemotePathResolver(execRoot)));
    assertThat(e.getLostArtifacts(ActionInputHelper::fromPath).byDigest())
        .containsExactly(
            DigestUtil.toString(digestUtil.computeAsUtf8("bar")),
            ActionInputHelper.fromPath("foo"));
  }

  @Test
  public void ensureInputsPresent_sharedMissingDigest_exceptionsHaveOwnLostInputs()
      throws Exception {
    RemoteCacheClient cacheProtocol = spy(new InMemoryCacheClient());
    RemoteExecutionCache remoteCache = spy(newRemoteExecutionCache(cacheProtocol));

    CountDownLatch findMissingDigestsCalls = new CountDownLatch(2);
    doAnswer(
            invocationOnMock -> {
              findMissingDigestsCalls.countDown();
              return invocationOnMock.callRealMethod();
            })
        .when(cacheProtocol)
        .findMissingDigests(any(), any());

    SettableFuture<Boolean> missingInputAvailable = SettableFuture.create();
    CountDownLatch remotePathChecked = new CountDownLatch(1);
    remoteCache.setRemotePathChecker(
        (context, path) -> {
          PathFragment execPath = path.relativeTo(execRoot);
          if (execPath.equals(PathFragment.create("outputs/foo"))
              || execPath.equals(PathFragment.create("outputs/bar"))) {
            remotePathChecked.countDown();
            return missingInputAvailable;
          }
          return immediateFuture(true);
        });

    Artifact foo = ActionsTestUtil.createArtifact(artifactRoot, "foo");
    Artifact bar = ActionsTestUtil.createArtifact(artifactRoot, "bar");
    Digest digest = fakeFileCache.createScratchInput(foo, "same");
    assertThat(fakeFileCache.createScratchInput(bar, "same")).isEqualTo(digest);

    Spawn fooSpawn = new SpawnBuilder().withInput(foo).build();
    var fooContext =
        new FakeSpawnExecutionContext(
            fooSpawn,
            fakeFileCache,
            execRoot,
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")),
            ImmutableClassToInstanceMap.of(),
            /* actionFileSystem= */ null);
    var fooRemoteContext = RemoteActionExecutionContext.create(fooSpawn, fooContext, metadata);
    var fooTree =
        (MerkleTree.Uploadable)
            merkleTreeComputer.buildForSpawn(
                fooSpawn,
                ImmutableSet.of(),
                /* scrubber= */ null,
                fooContext,
                RemotePathResolver.createDefault(execRoot),
                MerkleTreeComputer.BlobPolicy.KEEP_AND_REUPLOAD);

    Spawn barSpawn = new SpawnBuilder().withInput(bar).build();
    var barContext =
        new FakeSpawnExecutionContext(
            barSpawn,
            fakeFileCache,
            execRoot,
            new FileOutErr(execRoot.getRelative("stdout"), execRoot.getRelative("stderr")),
            ImmutableClassToInstanceMap.of(),
            /* actionFileSystem= */ null);
    var barRemoteContext = RemoteActionExecutionContext.create(barSpawn, barContext, metadata);
    var barTree =
        (MerkleTree.Uploadable)
            merkleTreeComputer.buildForSpawn(
                barSpawn,
                ImmutableSet.of(),
                /* scrubber= */ null,
                barContext,
                RemotePathResolver.createDefault(execRoot),
                MerkleTreeComputer.BlobPolicy.KEEP_AND_REUPLOAD);

    var fooFailure = new AtomicReference<Throwable>();
    Thread fooThread =
        new Thread(
            () -> {
              try {
                remoteCache.ensureInputsPresent(
                    fooRemoteContext,
                    fooTree,
                    ImmutableMap.of(),
                    /* force= */ false,
                    RemotePathResolver.createDefault(execRoot));
              } catch (Throwable t) {
                if (t instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                fooFailure.set(t);
              }
            });
    fooThread.start();
    assertThat(remotePathChecked.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();

    var barFailure = new AtomicReference<Throwable>();
    Thread barThread =
        new Thread(
            () -> {
              try {
                remoteCache.ensureInputsPresent(
                    barRemoteContext,
                    barTree,
                    ImmutableMap.of(),
                    /* force= */ false,
                    RemotePathResolver.createDefault(execRoot));
              } catch (Throwable t) {
                if (t instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                barFailure.set(t);
              }
            });
    barThread.start();
    assertThat(findMissingDigestsCalls.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        .isTrue();

    missingInputAvailable.set(false);
    fooThread.join();
    barThread.join();

    assertThat(fooFailure.get()).isInstanceOf(BulkTransferException.class);
    assertThat(
            ((BulkTransferException) fooFailure.get())
                .getLostArtifacts(execPath -> execPath.equals(foo.getExecPath()) ? foo : null)
                .byDigest())
        .containsExactly(DigestUtil.toString(digest), foo);

    assertThat(barFailure.get()).isInstanceOf(BulkTransferException.class);
    assertThat(
            ((BulkTransferException) barFailure.get())
                .getLostArtifacts(execPath -> execPath.equals(bar.getExecPath()) ? bar : null)
                .byDigest())
        .containsExactly(DigestUtil.toString(digest), bar);
  }

  @Test
  public void ensureInputsPresent_interruptedDuringUploadBlobs_cancelInProgressUploadTasks()
      throws Exception {
    // arrange
    RemoteCacheClient cacheProtocol = spy(new InMemoryCacheClient());
    RemoteExecutionCache remoteCache = spy(newRemoteExecutionCache(cacheProtocol));
    remoteActionExecutionContext = RemoteActionExecutionContext.create(metadata);

    Deque<SettableFuture<Void>> futures = new ConcurrentLinkedDeque<>();
    CountDownLatch uploadBlobCalls = new CountDownLatch(2);
    doAnswer(
            invocationOnMock -> {
              SettableFuture<Void> future = SettableFuture.create();
              futures.add(future);
              uploadBlobCalls.countDown();
              return future;
            })
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), (Blob) any());
    doAnswer(
            invocationOnMock -> {
              SettableFuture<Void> future = SettableFuture.create();
              futures.add(future);
              uploadBlobCalls.countDown();
              return future;
            })
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), any());

    Path path = execRoot.getRelative("foo");
    FileSystemUtils.writeContentAsLatin1(path, "bar");
    SortedMap<PathFragment, Path> inputs = new TreeMap<>();
    inputs.put(PathFragment.create("foo"), path);
    var merkleTree = merkleTreeComputer.buildForFiles(inputs);

    CountDownLatch ensureInputsPresentReturned = new CountDownLatch(1);
    Thread thread =
        new Thread(
            () -> {
              try {
                remoteCache.ensureInputsPresent(
                    remoteActionExecutionContext,
                    merkleTree,
                    ImmutableMap.of(),
                    false,
                    /* remotePathResolver= */ null);
              } catch (IOException | InterruptedException ignored) {
                // ignored
              } finally {
                ensureInputsPresentReturned.countDown();
              }
            });

    // act
    thread.start();
    uploadBlobCalls.await();
    assertThat(futures).hasSize(2);
    assertThat(cacheProtocol.getInProgressUploads()).isNotEmpty();

    thread.interrupt();
    ensureInputsPresentReturned.await();

    // assert
    assertThat(cacheProtocol.getInProgressUploads()).isEmpty();
    assertThat(cacheProtocol.getFinishedUploads()).isEmpty();
    for (SettableFuture<Void> future : futures) {
      assertThat(future.isCancelled()).isTrue();
    }
  }

  @Test
  public void
      ensureInputsPresent_multipleConsumers_interruptedOneDuringFindMissingBlobs_keepAndFinishInProgressUploadTasks()
          throws Exception {
    // arrange
    RemoteCacheClient cacheProtocol = spy(new InMemoryCacheClient());
    RemoteExecutionCache remoteCache = newRemoteExecutionCache(cacheProtocol);
    remoteActionExecutionContext = RemoteActionExecutionContext.create(metadata);

    SettableFuture<ImmutableSet<Digest>> findMissingDigestsFuture = SettableFuture.create();
    CountDownLatch findMissingDigestsCalled = new CountDownLatch(1);
    doAnswer(
            invocationOnMock -> {
              findMissingDigestsCalled.countDown();
              return findMissingDigestsFuture;
            })
        .when(cacheProtocol)
        .findMissingDigests(any(), any());
    Deque<SettableFuture<Void>> futures = new ConcurrentLinkedDeque<>();
    CountDownLatch uploadBlobCalls = new CountDownLatch(2);
    doAnswer(
            invocationOnMock -> {
              SettableFuture<Void> future = SettableFuture.create();
              futures.add(future);
              uploadBlobCalls.countDown();
              return future;
            })
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), (Blob) any());
    doAnswer(
            invocationOnMock -> {
              SettableFuture<Void> future = SettableFuture.create();
              futures.add(future);
              uploadBlobCalls.countDown();
              return future;
            })
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), any());

    Path path = execRoot.getRelative("foo");
    FileSystemUtils.writeContentAsLatin1(path, "bar");
    SortedMap<PathFragment, Path> inputs = new TreeMap<>();
    inputs.put(PathFragment.create("foo"), path);
    var merkleTree = merkleTreeComputer.buildForFiles(inputs);

    CountDownLatch ensureInputsPresentReturned = new CountDownLatch(2);
    CountDownLatch ensureInterrupted = new CountDownLatch(1);
    Runnable work =
        () -> {
          try {
            remoteCache.ensureInputsPresent(
                remoteActionExecutionContext,
                merkleTree,
                ImmutableMap.of(),
                false,
                /* remotePathResolver= */ null);
          } catch (IOException ignored) {
            // ignored
          } catch (InterruptedException e) {
            ensureInterrupted.countDown();
          } finally {
            ensureInputsPresentReturned.countDown();
          }
        };
    Thread thread1 = new Thread(work);
    Thread thread2 = new Thread(work);
    thread1.start();
    thread2.start();
    findMissingDigestsCalled.await();

    // act
    thread1.interrupt();
    ensureInterrupted.await();
    findMissingDigestsFuture.set(ImmutableSet.copyOf(merkleTree.allDigests()));

    uploadBlobCalls.await();
    assertThat(futures).hasSize(2);

    // assert
    assertThat(cacheProtocol.getInProgressUploads()).hasSize(2);
    assertThat(cacheProtocol.getFinishedUploads()).isEmpty();
    for (SettableFuture<Void> future : futures) {
      assertThat(future.isCancelled()).isFalse();
    }

    for (SettableFuture<Void> future : futures) {
      future.set(null);
    }
    ensureInputsPresentReturned.await();
    assertThat(cacheProtocol.getInProgressUploads()).isEmpty();
    assertThat(cacheProtocol.getFinishedUploads()).hasSize(2);
  }

  @Test
  public void
      ensureInputsPresent_multipleConsumers_interruptedOneDuringUploadBlobs_keepInProgressUploadTasks()
          throws Exception {
    // arrange
    RemoteCacheClient cacheProtocol = spy(new InMemoryCacheClient());
    RemoteExecutionCache remoteCache = spy(newRemoteExecutionCache(cacheProtocol));
    remoteActionExecutionContext = RemoteActionExecutionContext.create(metadata);

    Map<Digest, SettableFuture<Void>> uploadFutures = Maps.newConcurrentMap();
    // 3 unique file digests + 2 unique directory blob digests = 5 uploads total.
    CountDownLatch uploadCalls = new CountDownLatch(5);
    doAnswer(
            invocationOnMock -> {
              Digest digest = invocationOnMock.getArgument(1, Digest.class);
              SettableFuture<Void> future = SettableFuture.create();
              uploadFutures.put(digest, future);
              uploadCalls.countDown();
              return future;
            })
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), (Blob) any());

    Path foo = execRoot.getRelative("foo");
    FileSystemUtils.writeContentAsLatin1(foo, "foo");
    Path bar = execRoot.getRelative("bar");
    FileSystemUtils.writeContentAsLatin1(bar, "bar");
    Path qux = execRoot.getRelative("qux");
    FileSystemUtils.writeContentAsLatin1(qux, "qux");
    Digest fooDigest = digestUtil.computeAsUtf8("foo");
    Digest barDigest = digestUtil.computeAsUtf8("bar");
    Digest quxDigest = digestUtil.computeAsUtf8("qux");

    SortedMap<PathFragment, Path> input1 = new TreeMap<>();
    input1.put(PathFragment.create("foo"), foo);
    input1.put(PathFragment.create("bar"), bar);
    var merkleTree1 = merkleTreeComputer.buildForFiles(input1);

    SortedMap<PathFragment, Path> input2 = new TreeMap<>();
    input2.put(PathFragment.create("bar"), bar);
    input2.put(PathFragment.create("qux"), qux);
    var merkleTree2 = merkleTreeComputer.buildForFiles(input2);

    CountDownLatch ensureInputsPresentReturned = new CountDownLatch(2);
    CountDownLatch ensureInterrupted = new CountDownLatch(1);
    Thread thread1 =
        new Thread(
            () -> {
              try {
                remoteCache.ensureInputsPresent(
                    remoteActionExecutionContext,
                    merkleTree1,
                    ImmutableMap.of(),
                    false,
                    /* remotePathResolver= */ null);
              } catch (IOException ignored) {
                // ignored
              } catch (InterruptedException e) {
                ensureInterrupted.countDown();
              } finally {
                ensureInputsPresentReturned.countDown();
              }
            });
    Thread thread2 =
        new Thread(
            () -> {
              try {
                remoteCache.ensureInputsPresent(
                    remoteActionExecutionContext,
                    merkleTree2,
                    ImmutableMap.of(),
                    false,
                    /* remotePathResolver= */ null);
              } catch (InterruptedException | IOException ignored) {
                // ignored
              } finally {
                ensureInputsPresentReturned.countDown();
              }
            });

    // act
    thread1.start();
    thread2.start();
    uploadCalls.await();
    assertThat(uploadFutures).hasSize(5);
    assertThat(cacheProtocol.getInProgressUploads()).hasSize(5);

    thread1.interrupt();
    ensureInterrupted.await();

    // assert
    assertThat(cacheProtocol.getInProgressUploads()).hasSize(3);
    assertThat(cacheProtocol.getFinishedUploads()).isEmpty();
    // foo is only in tree1, so interrupting thread1 cancels it; bar is shared and qux is only in
    // tree2, so both are kept.
    assertThat(uploadFutures.get(fooDigest).isCancelled()).isTrue();
    assertThat(uploadFutures.get(barDigest).isCancelled()).isFalse();
    assertThat(uploadFutures.get(quxDigest).isCancelled()).isFalse();

    for (SettableFuture<Void> future : uploadFutures.values()) {
      future.set(null);
    }
    ensureInputsPresentReturned.await();
    assertThat(cacheProtocol.getInProgressUploads()).isEmpty();
    assertThat(cacheProtocol.getFinishedUploads()).hasSize(3);
  }

  @Test
  public void ensureInputsPresent_uploadFailed_propagateErrors() throws Exception {
    RemoteCacheClient cacheProtocol = spy(new InMemoryCacheClient());
    remoteActionExecutionContext = RemoteActionExecutionContext.create(metadata);
    doAnswer(invocationOnMock -> Futures.immediateFailedFuture(new IOException("upload failed")))
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), (Blob) any());
    doAnswer(invocationOnMock -> Futures.immediateFailedFuture(new IOException("upload failed")))
        .when(cacheProtocol)
        .uploadBlobImpl(any(), any(), any());
    RemoteExecutionCache remoteCache = spy(newRemoteExecutionCache(cacheProtocol));
    Path path = execRoot.getRelative("foo");
    FileSystemUtils.writeContentAsLatin1(path, "bar");
    SortedMap<PathFragment, Path> inputs = ImmutableSortedMap.of(PathFragment.create("foo"), path);
    var merkleTree = merkleTreeComputer.buildForFiles(inputs);

    IOException e =
        assertThrows(
            IOException.class,
            () ->
                remoteCache.ensureInputsPresent(
                    remoteActionExecutionContext,
                    merkleTree,
                    ImmutableMap.of(),
                    false,
                    /* remotePathResolver= */ null));

    assertThat(e).hasMessageThat().contains("upload failed");
  }

  @Test
  public void shutdownNow_cancelInProgressUploads() throws Exception {
    RemoteCacheClient remoteCacheClient = spy(new InMemoryCacheClient());
    // Return a future that never completes
    doAnswer(invocationOnMock -> SettableFuture.create())
        .when(remoteCacheClient)
        .uploadBlobImpl(any(), any(), any());
    CombinedCache combinedCache = newCombinedCache(remoteCacheClient);
    Digest digest = fakeFileCache.createScratchInput(ActionInputHelper.fromPath("file"), "content");
    Path file = execRoot.getRelative("file");

    ListenableFuture<Void> upload =
        combinedCache.uploadFile(remoteActionExecutionContext, digest, file);
    assertThat(remoteCacheClient.getInProgressUploads()).contains(digest);
    combinedCache.shutdownNow();

    assertThat(upload.isCancelled()).isTrue();
  }

  @Test
  public void uploadFile_chunkedUpload_deduplicatesRemoteUpload() throws Exception {
    // Spy on a real GrpcCacheClient so that final methods on the RemoteCacheClient base class
    // (e.g. dedupUpload, uploadFile) execute their real implementations against a properly
    // initialized casUploadCache.
    GrpcCacheClient grpcCacheClient =
        spy(
            new GrpcCacheClient(
                mock(ReferenceCountedChannel.class),
                mock(CallCredentialsProvider.class),
                Options.getDefaults(RemoteOptions.class),
                mock(RemoteRetrier.class),
                digestUtil));
    doAnswer(unused -> chunkingCapabilities()).when(grpcCacheClient).getServerCapabilities();
    doAnswer(unused -> immediateFuture(ImmutableSet.of()))
        .when(grpcCacheClient)
        .findMissingDigests(any(), any());

    CountDownLatch spliceStarted = new CountDownLatch(1);
    SettableFuture<Void> spliceFuture = SettableFuture.create();
    doAnswer(
            unused -> {
              spliceStarted.countDown();
              return spliceFuture;
            })
        .when(grpcCacheClient)
        .spliceBlob(any(), any(), any());

    CombinedCache combinedCache =
        new CombinedCache(
            grpcCacheClient,
            /* diskCacheClient= */ null,
            /* symlinkTemplate= */ null,
            digestUtil,
            /* chunkingEnabled= */ true);
    byte[] data = new byte[8192];
    Path file = execRoot.getRelative("chunked-output");
    try (var out = file.getOutputStream()) {
      out.write(data);
    }
    Digest digest = digestUtil.compute(data);

    try {
      ListenableFuture<Void> firstUpload =
          combinedCache.uploadFile(remoteActionExecutionContext, digest, file);
      assertThat(spliceStarted.await(1, TimeUnit.SECONDS)).isTrue();

      ListenableFuture<Void> secondUpload =
          combinedCache.uploadFile(remoteActionExecutionContext, digest, file);

      assertThat(grpcCacheClient.getUploadSubscriberCount(digest)).isEqualTo(2);
      verify(grpcCacheClient).findMissingDigests(any(), any());
      verify(grpcCacheClient).spliceBlob(any(), any(), any());

      spliceFuture.set(null);
      getFromFuture(firstUpload);
      getFromFuture(secondUpload);
    } finally {
      combinedCache.release();
    }
  }

  private InMemoryCombinedCache newCombinedCache() {
    return new InMemoryCombinedCache(digestUtil);
  }

  private InMemoryCombinedCache newCombinedCache(
      Map<Digest, byte[]> casEntries, DigestUtil digestUtil, @Nullable String symlinkTemplate) {
    return new InMemoryCombinedCache(casEntries, digestUtil, symlinkTemplate);
  }

  private CombinedCache newCombinedCache(RemoteCacheClient remoteCacheClient) {
    return new CombinedCache(
        remoteCacheClient,
        /* diskCacheClient= */ null,
        /* symlinkTemplate= */ null,
        digestUtil,
        /* chunkingEnabled= */ false);
  }

  private RemoteExecutionCache newRemoteExecutionCache(RemoteCacheClient remoteCacheClient) {
    return new RemoteExecutionCache(
        remoteCacheClient,
        /* diskCacheClient= */ null,
        /* symlinkTemplate= */ null,
        digestUtil,
        /* chunkingEnabled= */ false);
  }

  private static ServerCapabilities chunkingCapabilities() {
    return ServerCapabilities.newBuilder()
        .setCacheCapabilities(
            CacheCapabilities.newBuilder()
                .setFastCdc2020Params(
                    FastCdc2020Params.newBuilder().setAvgChunkSizeBytes(1024).build()))
        .build();
  }
}
