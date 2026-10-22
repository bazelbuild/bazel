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
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RemoteExternalOverlayFileSystemTest {
  @Test
  public void getInputStream_lostFileDownloadedBefore_readFromDisk() throws Exception {
    var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    var nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var externalRoot = PathFragment.create("/output/external");
    var overlay = new RemoteExternalOverlayFileSystem(externalRoot, nativeFs);
    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(isNull(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());
    when(prefetcher.isAvailable(any(), any())).thenCallRealMethod();
    overlay.beforeCommand(
        new InMemoryCombinedCache(digestUtil),
        prefetcher,
        new Reporter(),
        "build-request",
        "command",
        () -> mock(MemoizingEvaluator.class),
        Duration.ofMinutes(1));
    try {
      // The remote cache doesn't have the contents of the files.
      var repo = RepositoryName.create("repo");
      var root = Directory.newBuilder();
      for (var name : ImmutableList.of("downloaded", "missing", "modified")) {
        root.addFiles(
            FileNode.newBuilder()
                .setName(name)
                .setDigest(digestUtil.compute("contents".getBytes(UTF_8))));
      }
      assertThat(overlay.injectRemoteRepo(repo, Tree.newBuilder().setRoot(root).build(), "marker"))
          .isTrue();
      // The repo is still served from memory, but some of its files have been downloaded, e.g. as
      // inputs of an action.
      var repoDir = externalRoot.getChild(repo.getName());
      nativeFs.getPath(repoDir).createDirectoryAndParents();
      FileSystemUtils.writeContent(
          nativeFs.getPath(repoDir.getChild("downloaded")), UTF_8, "contents");
      FileSystemUtils.writeContent(
          nativeFs.getPath(repoDir.getChild("modified")), UTF_8, "CONTENTS");

      assertThat(
              FileSystemUtils.readContent(overlay.getPath(repoDir.getChild("downloaded")), UTF_8))
          .isEqualTo("contents");
      assertThrows(
          IOException.class,
          () -> FileSystemUtils.readContent(overlay.getPath(repoDir.getChild("missing"))));
      assertThrows(
          IOException.class,
          () -> FileSystemUtils.readContent(overlay.getPath(repoDir.getChild("modified"))));
    } finally {
      overlay.afterCommand();
    }
  }

  @Test
  public void getInputStream_downloadFailsForFileDownloadedBefore_readFromDisk() throws Exception {
    var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    var cacheClient = new InMemoryCacheClient();
    // The remote cache has the contents, but fails to serve them.
    cacheClient.addDownloadFailure(
        digestUtil.compute("contents".getBytes(UTF_8)), new IOException("503 Service Unavailable"));
    var nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var externalRoot = PathFragment.create("/output/external");
    var overlay = new RemoteExternalOverlayFileSystem(externalRoot, nativeFs);
    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(isNull(), any(), any(), any(), any()))
        .thenReturn(immediateVoidFuture());
    when(prefetcher.isAvailable(any(), any())).thenCallRealMethod();
    overlay.beforeCommand(
        new InMemoryCombinedCache(cacheClient, digestUtil),
        prefetcher,
        new Reporter(),
        "build-request",
        "command",
        () -> mock(MemoizingEvaluator.class),
        Duration.ofMinutes(1));
    try {
      var repo = RepositoryName.create("repo");
      var root = Directory.newBuilder();
      for (var name : ImmutableList.of("downloaded", "missing", "modified")) {
        root.addFiles(
            FileNode.newBuilder()
                .setName(name)
                .setDigest(digestUtil.compute("contents".getBytes(UTF_8))));
      }
      assertThat(overlay.injectRemoteRepo(repo, Tree.newBuilder().setRoot(root).build(), "marker"))
          .isTrue();
      // The repo is still served from memory, but some of its files have been downloaded, e.g. as
      // inputs of an action.
      var repoDir = externalRoot.getChild(repo.getName());
      nativeFs.getPath(repoDir).createDirectoryAndParents();
      FileSystemUtils.writeContent(
          nativeFs.getPath(repoDir.getChild("downloaded")), UTF_8, "contents");
      FileSystemUtils.writeContent(
          nativeFs.getPath(repoDir.getChild("modified")), UTF_8, "CONTENTS");

      assertThat(
              FileSystemUtils.readContent(overlay.getPath(repoDir.getChild("downloaded")), UTF_8))
          .isEqualTo("contents");
      assertThrows(
          IOException.class,
          () -> FileSystemUtils.readContent(overlay.getPath(repoDir.getChild("missing"))));
      assertThrows(
          IOException.class,
          () -> FileSystemUtils.readContent(overlay.getPath(repoDir.getChild("modified"))));
    } finally {
      overlay.afterCommand();
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
        () -> mock(MemoizingEvaluator.class),
        Duration.ofMinutes(1));
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
  public void ensureMaterialized_nativeRepoWithSymlinkIntoCachedRepo_materializesTargetRepo()
      throws Exception {
    var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    var cache = new InMemoryCombinedCache(digestUtil);
    var nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var externalRoot = PathFragment.create("/output/external");
    var overlay = new RemoteExternalOverlayFileSystem(externalRoot, nativeFs);
    var reporter = new Reporter();
    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(isNull(), any(), any(), any(), any()))
        .thenAnswer(
            invocation -> {
              Iterable<? extends ActionInput> inputs = invocation.getArgument(1);
              for (var input : inputs) {
                var path = nativeFs.getPath(input.getExecPath());
                path.getParentDirectory().createDirectoryAndParents();
                FileSystemUtils.writeContent(path, UTF_8, "binary_contents");
              }
              return immediateVoidFuture();
            });
    overlay.beforeCommand(
        cache,
        prefetcher,
        reporter,
        "build-request",
        "command",
        () -> mock(MemoizingEvaluator.class),
        Duration.ofMinutes(1));
    try {
      var cachedRepo = RepositoryName.create("cached_repo");
      var binDir =
          Directory.newBuilder()
              .addFiles(
                  FileNode.newBuilder()
                      .setName("python3")
                      .setDigest(digestUtil.compute("binary_contents".getBytes(UTF_8)))
                      .setIsExecutable(true))
              .build();
      var rootDir =
          Directory.newBuilder()
              .addDirectories(
                  DirectoryNode.newBuilder().setName("bin").setDigest(digestUtil.compute(binDir)))
              .build();
      var tree = Tree.newBuilder().setRoot(rootDir).addChildren(binDir).build();
      assertThat(overlay.injectRemoteRepo(cachedRepo, tree, "cached_marker")).isTrue();

      var unrelatedCachedRepo = RepositoryName.create("unrelated_cached_repo");
      assertThat(overlay.injectRemoteRepo(unrelatedCachedRepo, tree, "unrelated_marker")).isTrue();

      // Non-existent repo directory is ignored.
      overlay.ensureMaterialized(RepositoryName.create("nonexistent_repo"), reporter);

      var nativeRepo = RepositoryName.create("native_repo");
      var nativeRepoDir = externalRoot.getChild(nativeRepo.getName());
      nativeFs.getPath(nativeRepoDir.getChild("sub")).createDirectoryAndParents();
      FileSystemUtils.writeContent(
          nativeFs.getPath(nativeRepoDir.getChild("regular_file")), UTF_8, "native");
      nativeFs
          .getPath(nativeRepoDir.getChild("link_to_regular_file"))
          .createSymbolicLink(PathFragment.create("regular_file"));
      nativeFs
          .getPath(nativeRepoDir.getChild("dangling_link"))
          .createSymbolicLink(PathFragment.create("does_not_exist"));
      nativeFs
          .getPath(nativeRepoDir.getChild("loop_a"))
          .createSymbolicLink(PathFragment.create("loop_b"));
      nativeFs
          .getPath(nativeRepoDir.getChild("loop_b"))
          .createSymbolicLink(PathFragment.create("loop_a"));
      nativeFs
          .getPath(nativeRepoDir.getRelative("sub/python_target"))
          .createSymbolicLink(PathFragment.create("../../cached_repo/bin/python3"));
      nativeFs
          .getPath(nativeRepoDir.getChild("python"))
          .createSymbolicLink(nativeRepoDir.getRelative("sub/python_target"));

      overlay.ensureMaterialized(nativeRepo, reporter);

      assertThat(
              FileSystemUtils.readContent(
                  nativeFs.getPath(externalRoot.getChild(cachedRepo.getMarkerFileName())), UTF_8))
          .isEqualTo("cached_marker");
      assertThat(
              FileSystemUtils.readContent(
                  nativeFs.getPath(nativeRepoDir.getChild("python")).resolveSymbolicLinks(), UTF_8))
          .isEqualTo("binary_contents");
      assertThat(
              nativeFs
                  .getPath(externalRoot.getChild(unrelatedCachedRepo.getMarkerFileName()))
                  .exists())
          .isFalse();

      // Already-materialized repos are not walked for symlinks.
      nativeFs
          .getPath(externalRoot.getChild(cachedRepo.getName()).getChild("extra_link"))
          .createSymbolicLink(PathFragment.create("../unrelated_cached_repo/bin/python3"));
      overlay.ensureMaterialized(cachedRepo, reporter);
      assertThat(
              nativeFs
                  .getPath(externalRoot.getChild(unrelatedCachedRepo.getMarkerFileName()))
                  .exists())
          .isFalse();

      // Once the last remaining cached repo is materialized during a directory walk, the walk
      // stops early and subsequent walks return immediately.
      var nativeRepo2 = RepositoryName.create("native_repo2");
      var nativeRepo2Dir = externalRoot.getChild(nativeRepo2.getName());
      nativeFs.getPath(nativeRepo2Dir).createDirectoryAndParents();
      nativeFs
          .getPath(nativeRepo2Dir.getChild("a_link"))
          .createSymbolicLink(PathFragment.create("../unrelated_cached_repo/bin/python3"));
      nativeFs
          .getPath(nativeRepo2Dir.getChild("z_link"))
          .createSymbolicLink(PathFragment.create("../unrelated_cached_repo/bin/python3"));
      overlay.ensureMaterialized(nativeRepo2, reporter);
      assertThat(
              FileSystemUtils.readContent(
                  nativeFs.getPath(externalRoot.getChild(unrelatedCachedRepo.getMarkerFileName())),
                  UTF_8))
          .isEqualTo("unrelated_marker");
      overlay.ensureMaterialized(nativeRepo2, reporter);
    } finally {
      overlay.afterCommand();
    }
  }
}
