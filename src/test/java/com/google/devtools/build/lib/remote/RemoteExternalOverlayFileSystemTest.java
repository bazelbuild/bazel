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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventBusEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RewindableRepoFileSystem;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.OutputStream;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Tests for {@link RemoteExternalOverlayFileSystem}. */
@RunWith(JUnit4.class)
public final class RemoteExternalOverlayFileSystemTest {
  private static final long TIMEOUT_MILLIS = 10_000;
  private static final PathFragment EXTERNAL_DIR = PathFragment.create("/output_base/external");
  private static final RepositoryName REPO = RepositoryName.createUnvalidated("repo");
  private static final PathFragment REPO_DIR = EXTERNAL_DIR.getChild(REPO.getName());
  private static final PathFragment SRC_FILE = REPO_DIR.getChild("src.txt");
  private static final PathFragment MARKER_FILE = EXTERNAL_DIR.getChild(REPO.getMarkerFileName());

  private final InMemoryFileSystem nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private final Map<Digest, byte[]> cas = new HashMap<>();
  private final Reporter reporter = new Reporter(EventBusEventHandler.createWithNewEventBus());
  private final CountDownLatch downloadStarted = new CountDownLatch(1);
  private final CountDownLatch downloadGate = new CountDownLatch(1);
  private final RemoteExternalOverlayFileSystem overlayFs =
      new RemoteExternalOverlayFileSystem(EXTERNAL_DIR, nativeFs);

  /** A cache client whose downloads announce themselves and then wait for a gate to open. */
  private static final class GatedCacheClient extends InMemoryCacheClient {
    private final CountDownLatch started;
    private final CountDownLatch gate;

    GatedCacheClient(Map<Digest, byte[]> cas, CountDownLatch started, CountDownLatch gate) {
      super(cas);
      this.started = started;
      this.gate = gate;
    }

    @Override
    public ListenableFuture<Void> downloadBlob(
        RemoteActionExecutionContext context, Digest digest, OutputStream out) {
      started.countDown();
      try {
        if (!gate.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS)) {
          return Futures.immediateFailedFuture(
              new IllegalStateException("download gate stayed shut"));
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return Futures.immediateFailedFuture(e);
      }
      return super.downloadBlob(context, digest, out);
    }
  }

  @Test
  public void ensureMaterialized_refetchWaitsForMaterialization() throws Exception {
    setUpWithInjectedRepo("old");
    RewindingSynchronizer synchronizer = overlayFs.getRewindingSynchronizer();
    synchronizer.markReplacementsPossible();

    TestThread materializer = new TestThread(() -> overlayFs.ensureMaterialized(REPO, reporter));
    materializer.start();
    // The copy holds the repo's read lock while its downloads are in flight.
    assertThat(downloadStarted.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS)).isTrue();

    AtomicBoolean markerSeenByRefetch = new AtomicBoolean();
    TestThread refetcher =
        new TestThread(
            () -> {
              try (var writeLock = overlayFs.acquireRepoWriteLock(REPO)) {
                markerSeenByRefetch.set(nativeFs.getPath(MARKER_FILE).exists());
              }
            });
    refetcher.start();
    awaitCondition(() -> refetcher.getState() == Thread.State.WAITING);

    downloadGate.countDown();
    materializer.joinAndAssertState(TIMEOUT_MILLIS);
    refetcher.joinAndAssertState(TIMEOUT_MILLIS);
    assertThat(markerSeenByRefetch.get()).isTrue();
    assertThat(FileSystemUtils.readContent(nativeFs.getPath(SRC_FILE), UTF_8)).isEqualTo("old");
  }

  @Test
  public void ensureMaterialized_duringRefetch_waitsForRefetchedRepo() throws Exception {
    downloadGate.countDown();
    setUpWithInjectedRepo("old");
    overlayFs.getRewindingSynchronizer().markReplacementsPossible();

    TestThread materializer = new TestThread(() -> overlayFs.ensureMaterialized(REPO, reporter));
    try (var writeLock = overlayFs.acquireRepoWriteLock(REPO)) {
      // A refetch deletes the repo root, which drops the in-memory contents and, on the next
      // access, their marker, and then recreates the repo on the native file system.
      overlayFs.getPath(REPO_DIR).deleteTree();
      assertThat(overlayFs.getPath(SRC_FILE).exists()).isFalse();
      materializer.start();
      // There is nothing left to materialize, but the caller must still not get to read the repo
      // before the refetch has finished.
      awaitCondition(() -> materializer.getState() == Thread.State.WAITING);
      overlayFs.getPath(REPO_DIR).createDirectoryAndParents();
      FileSystemUtils.writeContent(overlayFs.getPath(SRC_FILE), UTF_8, "new");
    }

    materializer.joinAndAssertState(TIMEOUT_MILLIS);
    assertThat(FileSystemUtils.readContent(overlayFs.getPath(SRC_FILE), UTF_8)).isEqualTo("new");
    assertThat(nativeFs.getPath(MARKER_FILE).exists()).isFalse();
  }

  @Test
  public void readUnderRepoLock_duringRefetch_observesRefetchedContents() throws Exception {
    downloadGate.countDown();
    setUpWithInjectedRepo("old");
    overlayFs.getRewindingSynchronizer().markReplacementsPossible();
    AtomicReference<String> result = new AtomicReference<>();
    TestThread reader =
        new TestThread(
            () ->
                result.set(
                    RewindableRepoFileSystem.readUnderRepoLock(
                        overlayFs.getPath(SRC_FILE),
                        () -> FileSystemUtils.readContent(overlayFs.getPath(SRC_FILE), UTF_8))));

    try (var writeLock = overlayFs.acquireRepoWriteLock(REPO)) {
      reader.start();
      // The read waits for the refetch to finish rather than observing it halfway.
      awaitCondition(() -> reader.getState() == Thread.State.WAITING);
      overlayFs.getPath(REPO_DIR).deleteTree();
      overlayFs.getPath(REPO_DIR).createDirectoryAndParents();
      FileSystemUtils.writeContent(overlayFs.getPath(SRC_FILE), UTF_8, "new");
    }
    reader.joinAndAssertState(TIMEOUT_MILLIS);

    assertThat(result.get()).isEqualTo("new");
  }

  @Test
  public void readUnderRepoLock_outsideRepos_readsDirectly() throws Exception {
    downloadGate.countDown();
    setUpWithInjectedRepo("old");
    overlayFs.getRewindingSynchronizer().markReplacementsPossible();
    PathFragment mainRepoFile = PathFragment.create("/workspace/file.txt");
    nativeFs.getPath(mainRepoFile).getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(overlayFs.getPath(mainRepoFile), UTF_8, "main");

    try (var writeLock = overlayFs.acquireRepoWriteLock(REPO)) {
      assertThat(
              RewindableRepoFileSystem.readUnderRepoLock(
                  overlayFs.getPath(mainRepoFile),
                  () -> FileSystemUtils.readContent(overlayFs.getPath(mainRepoFile), UTF_8)))
          .isEqualTo("main");
    }
  }

  @Test
  public void readUnderRepoLock_otherRepo_readsDirectly() throws Exception {
    downloadGate.countDown();
    setUpWithInjectedRepo("old");
    overlayFs.getRewindingSynchronizer().markReplacementsPossible();
    PathFragment otherRepoFile = EXTERNAL_DIR.getChild("other_repo").getChild("file.txt");
    nativeFs.getPath(otherRepoFile).getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(overlayFs.getPath(otherRepoFile), UTF_8, "other");

    try (var writeLock = overlayFs.acquireRepoWriteLock(REPO)) {
      assertThat(
              RewindableRepoFileSystem.readUnderRepoLock(
                  overlayFs.getPath(otherRepoFile),
                  () -> FileSystemUtils.readContent(overlayFs.getPath(otherRepoFile), UTF_8)))
          .isEqualTo("other");
    }
  }

  @Test
  public void ensureMaterialized_previousCallerFinishesBeforeTaskSubmission() throws Exception {
    var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    var cache = new InMemoryCombinedCache(digestUtil);
    var nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var externalRoot = PathFragment.create("/output/external");
    var overlay = new RemoteExternalOverlayFileSystem(externalRoot, nativeFs);
    var reporter = new Reporter();
    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(isNull(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());
    overlay.beforeCommand(
        cache,
        prefetcher,
        reporter,
        "build-request",
        "command",
        mock(MemoizingEvaluator.class),
        Duration.ofMinutes(1),
        /* rewindingEnabled= */ false);
    try {
      var repo = RepositoryName.create("repo");
      assertThat(overlay.injectRemoteRepo(repo, Tree.getDefaultInstance(), "marker")).isTrue();
      var delayedRepo = mock(RepositoryName.class);
      when(delayedRepo.getMarkerFileName()).thenReturn(repo.getMarkerFileName());
      var calls = new AtomicInteger();
      when(delayedRepo.getName())
          .thenAnswer(
              unused -> {
                // Finish another caller after the presence check but before task submission.
                if (calls.incrementAndGet() == 2) {
                  overlay.ensureMaterialized(repo, reporter);
                }
                return repo.getName();
              });

      overlay.ensureMaterialized(delayedRepo, reporter);

      assertThat(
              FileSystemUtils.readContent(
                  nativeFs.getPath(externalRoot.getChild(repo.getMarkerFileName())), UTF_8))
          .isEqualTo("marker");
    } finally {
      overlay.afterCommand();
    }
  }

  @Test
  public void markLostRepoFile_untilRefetched_reportsCacheMiss() throws Exception {
    setUpWithInjectedRepo("content");

    overlayFs.markLostRepoFile(REPO);
    assertThat(overlayFs.shouldRefetch(REPO)).isTrue();

    // A refetch deletes the injected contents and runs the repo rule, which writes to disk.
    overlayFs.getPath(REPO_DIR).deleteTree();
    overlayFs.getPath(REPO_DIR).createDirectoryAndParents();
    FileSystemUtils.writeContent(overlayFs.getPath(SRC_FILE), UTF_8, "content");
    overlayFs.repoRefetched(REPO);
    assertThat(overlayFs.shouldRefetch(REPO)).isFalse();

    // A loss reported after the contents have been replaced concerns the old contents.
    overlayFs.markLostRepoFile(REPO);
    assertThat(overlayFs.shouldRefetch(REPO)).isFalse();
  }

  @Test
  public void afterCommand_lostFileNotRefetched_dropsRepoAndInvalidatesFetch() throws Exception {
    MemoizingEvaluator evaluator = mock(MemoizingEvaluator.class);
    setUpWithInjectedRepo("content", evaluator);
    overlayFs.markLostRepoFile(REPO);

    overlayFs.afterCommand();

    // The next command refetches the repo, which the marker makes a cache miss.
    @SuppressWarnings("unchecked")
    ArgumentCaptor<Predicate<SkyKey>> deleted = ArgumentCaptor.forClass(Predicate.class);
    verify(evaluator).delete(deleted.capture());
    assertThat(deleted.getValue().test(RepositoryDirectoryValue.key(REPO))).isTrue();
    assertThat(
            deleted
                .getValue()
                .test(RepositoryDirectoryValue.key(RepositoryName.createUnvalidated("other"))))
        .isFalse();
    assertThat(overlayFs.getPath(SRC_FILE).exists()).isFalse();
    assertThat(overlayFs.shouldRefetch(REPO)).isTrue();
  }

  @Test
  public void afterCommand_lostFileRefetched_keepsRepoAndFetch() throws Exception {
    MemoizingEvaluator evaluator = mock(MemoizingEvaluator.class);
    setUpWithInjectedRepo("content", evaluator);
    overlayFs.markLostRepoFile(REPO);
    // The refetch replaced the injected contents with the repo rule's output on disk.
    overlayFs.getPath(REPO_DIR).deleteTree();
    overlayFs.getPath(REPO_DIR).createDirectoryAndParents();
    FileSystemUtils.writeContent(overlayFs.getPath(SRC_FILE), UTF_8, "content");
    overlayFs.repoRefetched(REPO);

    overlayFs.afterCommand();

    verifyNoInteractions(evaluator);
    assertThat(overlayFs.getPath(SRC_FILE).exists()).isTrue();
    assertThat(overlayFs.shouldRefetch(REPO)).isFalse();
  }

  private void setUpWithInjectedRepo(String srcContent) throws Exception {
    setUpWithInjectedRepo(srcContent, /* evaluator= */ null);
  }

  private void setUpWithInjectedRepo(String srcContent, @Nullable MemoizingEvaluator evaluator)
      throws Exception {
    byte[] srcBytes = srcContent.getBytes(UTF_8);
    Digest srcDigest = digestUtil.compute(srcBytes);
    cas.put(srcDigest, srcBytes);
    CombinedCache cache =
        new CombinedCache(
            new GatedCacheClient(cas, downloadStarted, downloadGate),
            /* diskCacheClient= */ null,
            /* symlinkTemplate= */ null,
            digestUtil,
            /* chunkingFunction= */ null,
            new ChunkLocationMap());
    nativeFs.getPath("/tmp").createDirectoryAndParents();
    RemoteActionInputFetcher prefetcher =
        new RemoteActionInputFetcher(
            reporter,
            "none",
            "none",
            cache,
            overlayFs.getPath("/exec"),
            new TempPathGenerator(nativeFs.getPath("/tmp")),
            new RemoteOutputChecker("build", RemoteOutputsMode.MINIMAL, ImmutableList.of()),
            ActionOutputDirectoryHelper.createForTesting(),
            OutputPermissions.READONLY);
    overlayFs.beforeCommand(
        cache,
        prefetcher,
        reporter,
        "none",
        "none",
        evaluator,
        /* remoteCacheTtl= */ Duration.ofHours(1),
        /* rewindingEnabled= */ true);

    Tree tree =
        Tree.newBuilder()
            .setRoot(
                Directory.newBuilder()
                    .addFiles(FileNode.newBuilder().setName("src.txt").setDigest(srcDigest)))
            .build();
    assertThat(overlayFs.injectRemoteRepo(REPO, tree, "MARKER\n")).isTrue();
  }

  private static void awaitCondition(BooleanSupplier condition) throws InterruptedException {
    long deadline = System.currentTimeMillis() + TIMEOUT_MILLIS;
    while (!condition.getAsBoolean()) {
      if (System.currentTimeMillis() > deadline) {
        fail("condition not met within " + TIMEOUT_MILLIS + "ms");
      }
      Thread.sleep(10);
    }
  }
}
