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
import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Reason;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.actions.FileStatusWithMetadata;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RemoteExternalOverlayFileSystemTest {
  private static final PathFragment EXTERNAL_ROOT = PathFragment.create("/output/external");
  private static final RepositoryName REPO = RepositoryName.createUnvalidated("repo");

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
  public void prefetch_fileBelowSymlinkedDirectory_reproducesSymlink() throws Exception {
    var repoDir = EXTERNAL_ROOT.getChild(REPO.getName());
    try (var env = new RepoWithSymlinkedDirectory()) {
      var input = ActionInputHelper.fromPath(repoDir.getRelative("alias/file"));

      var unused =
          getFromFuture(
              env.prefetcher.prefetchFilesInterruptibly(
                  /* action= */ null,
                  ImmutableList.of(input),
                  unusedInput ->
                      ((FileStatusWithMetadata) env.overlay.getPath(input.getExecPath()).stat())
                          .getMetadata(),
                  Priority.MEDIUM,
                  Reason.INPUTS));

      var nativeRepoDir = env.nativeFs.getPath(repoDir);
      assertThat(nativeRepoDir.getChild("alias").readSymbolicLink())
          .isEqualTo(PathFragment.create("real"));
      assertThat(FileSystemUtils.readContent(nativeRepoDir.getRelative("real/file"), UTF_8))
          .isEqualTo("file contents");

      // Materializing the entire repo plants the symlink again, which only succeeds if the symlink
      // hasn't been created as a directory.
      env.overlay.ensureMaterialized(REPO, env.reporter);

      assertThat(nativeRepoDir.getChild("alias").readSymbolicLink())
          .isEqualTo(PathFragment.create("real"));
      assertThat(
              FileSystemUtils.readContent(nativeRepoDir.getRelative("alias/subdir/nested"), UTF_8))
          .isEqualTo("nested contents");
    }
  }

  @Test
  public void ensureSubtreeMaterialized_belowSymlinkedDirectory_reproducesSymlink()
      throws Exception {
    var repoDir = EXTERNAL_ROOT.getChild(REPO.getName());
    try (var env = new RepoWithSymlinkedDirectory()) {
      env.overlay.ensureSubtreeMaterialized(repoDir.getRelative("alias/subdir"));

      var nativeRepoDir = env.nativeFs.getPath(repoDir);
      assertThat(nativeRepoDir.getChild("alias").readSymbolicLink())
          .isEqualTo(PathFragment.create("real"));
      assertThat(
              FileSystemUtils.readContent(nativeRepoDir.getRelative("real/subdir/nested"), UTF_8))
          .isEqualTo("nested contents");

      // Materializing the entire repo plants the symlink again, which only succeeds if the symlink
      // hasn't been created as a directory.
      env.overlay.ensureMaterialized(REPO, env.reporter);

      assertThat(FileSystemUtils.readContent(nativeRepoDir.getRelative("alias/file"), UTF_8))
          .isEqualTo("file contents");
    }
  }

  @Test
  public void ensureMaterialized_symlinkLoop_reproduced() throws Exception {
    var repoDir = EXTERNAL_ROOT.getChild(REPO.getName());
    try (var env =
        new RepoWithSymlinkedDirectory(
            root ->
                root.addSymlinks(SymlinkNode.newBuilder().setName("loop_a").setTarget("loop_b"))
                    .addSymlinks(
                        SymlinkNode.newBuilder().setName("loop_b").setTarget("loop_a")))) {
      env.overlay.ensureMaterialized(REPO, env.reporter);

      var nativeRepoDir = env.nativeFs.getPath(repoDir);
      assertThat(nativeRepoDir.getChild("loop_a").readSymbolicLink())
          .isEqualTo(PathFragment.create("loop_b"));
      assertThat(nativeRepoDir.getChild("loop_b").readSymbolicLink())
          .isEqualTo(PathFragment.create("loop_a"));
      assertThat(FileSystemUtils.readContent(nativeRepoDir.getRelative("real/file"), UTF_8))
          .isEqualTo("file contents");
    }
  }

  @Test
  public void ensureMaterialized_symlinkLoopThroughItself_reproduced() throws Exception {
    var repoDir = EXTERNAL_ROOT.getChild(REPO.getName());
    try (var env =
        new RepoWithSymlinkedDirectory(
            root ->
                root.addSymlinks(
                    SymlinkNode.newBuilder().setName("loop").setTarget("loop/subdir")))) {
      env.overlay.ensureMaterialized(REPO, env.reporter);

      var nativeRepoDir = env.nativeFs.getPath(repoDir);
      assertThat(nativeRepoDir.getChild("loop").readSymbolicLink())
          .isEqualTo(PathFragment.create("loop/subdir"));
      assertThat(FileSystemUtils.readContent(nativeRepoDir.getRelative("real/file"), UTF_8))
          .isEqualTo("file contents");
    }
  }

  @Test
  public void ensureMaterialized_symlinkToItself_reproduced() throws Exception {
    var repoDir = EXTERNAL_ROOT.getChild(REPO.getName());
    // Some file systems only support absolute symlink targets.
    var target = repoDir.getChild("loop");
    try (var env =
        new RepoWithSymlinkedDirectory(
            root ->
                root.addSymlinks(
                    SymlinkNode.newBuilder().setName("loop").setTarget(target.getPathString())))) {
      env.overlay.ensureMaterialized(REPO, env.reporter);

      assertThat(env.nativeFs.getPath(target).readSymbolicLink()).isEqualTo(target);
    }
  }

  @Test
  public void prefetch_pathThroughSameSymlinkTwice_notMistakenForLoop() throws Exception {
    var repoDir = EXTERNAL_ROOT.getChild(REPO.getName());
    try (var env =
        new RepoWithSymlinkedDirectory(
            root -> root.addSymlinks(SymlinkNode.newBuilder().setName("self").setTarget(".")))) {
      var input = ActionInputHelper.fromPath(repoDir.getRelative("self/self/real/file"));

      var unused =
          getFromFuture(
              env.prefetcher.prefetchFilesInterruptibly(
                  /* action= */ null,
                  ImmutableList.of(input),
                  unusedInput ->
                      ((FileStatusWithMetadata) env.overlay.getPath(input.getExecPath()).stat())
                          .getMetadata(),
                  Priority.MEDIUM,
                  Reason.INPUTS));

      var nativeRepoDir = env.nativeFs.getPath(repoDir);
      assertThat(nativeRepoDir.getChild("self").readSymbolicLink())
          .isEqualTo(PathFragment.create("."));
      assertThat(nativeRepoDir.getRelative("real/file").isFile(Symlinks.NOFOLLOW)).isTrue();
      assertThat(
              FileSystemUtils.readContent(nativeRepoDir.getRelative("self/self/real/file"), UTF_8))
          .isEqualTo("file contents");
    }
  }

  /**
   * An overlay file system with an injected repo that consists of a directory {@code real}
   * containing {@code file} and {@code subdir/nested} as well as a symlink {@code alias} to {@code
   * real}, backed by a real prefetcher.
   */
  private static final class RepoWithSymlinkedDirectory implements AutoCloseable {
    final InMemoryFileSystem nativeFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    final RemoteExternalOverlayFileSystem overlay =
        new RemoteExternalOverlayFileSystem(EXTERNAL_ROOT, nativeFs);
    final Reporter reporter = new Reporter();
    final RemoteActionInputFetcher prefetcher;

    RepoWithSymlinkedDirectory() throws Exception {
      this(root -> {});
    }

    /**
     * @param rootCustomizer adds further entries to the root directory of the repo
     */
    RepoWithSymlinkedDirectory(Consumer<Directory.Builder> rootCustomizer) throws Exception {
      var digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
      var casEntries = new HashMap<Digest, byte[]>();
      var subdir =
          Directory.newBuilder()
              .addFiles(fileNode(digestUtil, casEntries, "nested", "nested contents"))
              .build();
      var real =
          Directory.newBuilder()
              .addFiles(fileNode(digestUtil, casEntries, "file", "file contents"))
              .addDirectories(
                  DirectoryNode.newBuilder().setName("subdir").setDigest(digestUtil.compute(subdir)))
              .build();
      var root =
          Directory.newBuilder()
              .addDirectories(
                  DirectoryNode.newBuilder().setName("real").setDigest(digestUtil.compute(real)))
              .addSymlinks(SymlinkNode.newBuilder().setName("alias").setTarget("real"));
      rootCustomizer.accept(root);
      var tree = Tree.newBuilder().setRoot(root).addChildren(real).addChildren(subdir).build();

      var cache = new InMemoryCombinedCache(casEntries, digestUtil);
      var execRoot = overlay.getPath("/output/execroot");
      execRoot.createDirectoryAndParents();
      var tempDir = overlay.getPath("/output/tmp");
      tempDir.createDirectoryAndParents();
      prefetcher =
          new RemoteActionInputFetcher(
              reporter,
              "build-request",
              "command",
              cache,
              execRoot,
              new TempPathGenerator(tempDir),
              new RemoteOutputChecker("build", RemoteOutputsMode.MINIMAL, ImmutableList.of()),
              ActionOutputDirectoryHelper.createForTesting(),
              OutputPermissions.READONLY);
      overlay.beforeCommand(
          cache,
          prefetcher,
          reporter,
          "build-request",
          "command",
          mock(MemoizingEvaluator.class),
          Duration.ofMinutes(1));
      assertThat(overlay.injectRemoteRepo(REPO, tree, "marker")).isTrue();
    }

    private static FileNode fileNode(
        DigestUtil digestUtil, Map<Digest, byte[]> casEntries, String name, String contents) {
      var bytes = contents.getBytes(UTF_8);
      var digest = digestUtil.compute(bytes);
      casEntries.put(digest, bytes);
      return FileNode.newBuilder().setName(name).setDigest(digest).build();
    }

    @Override
    public void close() {
      overlay.afterCommand();
    }
  }
}
