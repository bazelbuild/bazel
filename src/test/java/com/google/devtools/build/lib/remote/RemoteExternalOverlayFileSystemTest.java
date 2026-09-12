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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;

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
import java.io.OutputStream;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
  private final CountDownLatch downloadGate = new CountDownLatch(1);
  private final RemoteExternalOverlayFileSystem overlayFs =
      new RemoteExternalOverlayFileSystem(EXTERNAL_DIR, nativeFs);

  /** A cache client whose downloads wait for a gate to open. */
  private static final class GatedCacheClient extends InMemoryCacheClient {
    private final CountDownLatch gate;

    GatedCacheClient(Map<Digest, byte[]> cas, CountDownLatch gate) {
      super(cas);
      this.gate = gate;
    }

    @Override
    public ListenableFuture<Void> downloadBlob(
        RemoteActionExecutionContext context, Digest digest, OutputStream out) {
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
    awaitCondition(() -> synchronizer.hasBlockingReadLockForTesting(REPO));

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
  public void ensureMaterialized_duringRefetch_skipsRefetchedRepo() throws Exception {
    downloadGate.countDown();
    setUpWithInjectedRepo("old");
    overlayFs.getRewindingSynchronizer().markReplacementsPossible();

    TestThread materializer = new TestThread(() -> overlayFs.ensureMaterialized(REPO, reporter));
    try (var writeLock = overlayFs.acquireRepoWriteLock(REPO)) {
      materializer.start();
      // A refetch deletes the repo root and recreates it on the native file system.
      overlayFs.getPath(REPO_DIR).deleteTree();
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

  private void setUpWithInjectedRepo(String srcContent) throws Exception {
    byte[] srcBytes = srcContent.getBytes(UTF_8);
    Digest srcDigest = digestUtil.compute(srcBytes);
    cas.put(srcDigest, srcBytes);
    CombinedCache cache =
        new CombinedCache(
            new GatedCacheClient(cas, downloadGate),
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
        /* evaluator= */ null,
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
