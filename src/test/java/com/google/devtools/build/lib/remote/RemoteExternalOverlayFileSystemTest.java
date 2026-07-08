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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.AbstractActionInputPrefetcher.MetadataSupplier;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.testing.vfs.SpiedFileSystem;
import com.google.devtools.build.lib.vfs.DetailedIOException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.invocation.InvocationOnMock;

/**
 * Tests for {@link RemoteExternalOverlayFileSystem}, with a focus on symlink chains that cross the
 * in-memory and native backings.
 */
@RunWith(JUnit4.class)
public final class RemoteExternalOverlayFileSystemTest {

  private static final PathFragment OUTPUT_BASE = PathFragment.create("/output_base");
  private static final PathFragment EXTERNAL_DIR = OUTPUT_BASE.getRelative("external");
  private static final PathFragment WORKSPACE_DIR = PathFragment.create("/workspace");

  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private final Map<Digest, byte[]> casEntries = new ConcurrentHashMap<>();
  private final InMemoryCacheClient cacheClient = new InMemoryCacheClient();
  private final InMemoryCombinedCache cache = new InMemoryCombinedCache(cacheClient, digestUtil);
  private final RemoteActionExecutionContext context =
      RemoteActionExecutionContext.create(
          TracingMetadataUtils.buildMetadata("build-request-id", "command-id", "action-id"));
  private final Reporter reporter = new Reporter();

  private SpiedFileSystem nativeFs;
  private RemoteExternalOverlayFileSystem overlayFs;

  @Before
  public void setUp() throws Exception {
    nativeFs = SpiedFileSystem.createInMemorySpy();
    nativeFs.getPath(EXTERNAL_DIR).createDirectoryAndParents();
    nativeFs.getPath(WORKSPACE_DIR).createDirectoryAndParents();
    overlayFs = new RemoteExternalOverlayFileSystem(EXTERNAL_DIR, nativeFs);
    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(any(), any(), any(), any(), any()))
        .thenAnswer(this::fakePrefetch);
    overlayFs.beforeCommand(
        cache,
        prefetcher,
        reporter,
        "build-request-id",
        "command-id",
        mock(MemoizingEvaluator.class),
        Duration.ofHours(1));
  }

  /** Materializes files and symlinks to the native file system, like the real prefetcher. */
  private ListenableFuture<Void> fakePrefetch(InvocationOnMock invocation) throws Exception {
    Iterable<? extends ActionInput> inputs = invocation.getArgument(1);
    MetadataSupplier metadataSupplier = invocation.getArgument(2);
    for (ActionInput input : inputs) {
      Path path = nativeFs.getPath(input.getExecPath());
      path.getParentDirectory().createDirectoryAndParents();
      FileArtifactValue metadata = metadataSupplier.getMetadata(input);
      if (metadata.getType().isSymlink()) {
        path.createSymbolicLink(PathFragment.create(metadata.getUnresolvedSymlinkTarget()));
      } else {
        byte[] contents =
            casEntries.get(DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize()));
        FileSystemUtils.writeContent(path, contents);
      }
    }
    return immediateVoidFuture();
  }

  private FileNode fileNode(String name, String contents) throws Exception {
    byte[] bytes = contents.getBytes(UTF_8);
    Digest digest = cache.addContents(context, bytes);
    casEntries.put(digest, bytes);
    return FileNode.newBuilder().setName(name).setDigest(digest).build();
  }

  private static SymlinkNode symlinkNode(String name, String target) {
    return SymlinkNode.newBuilder().setName(name).setTarget(target).build();
  }

  private DirectoryNode dirNode(String name, Directory dir) {
    return DirectoryNode.newBuilder().setName(name).setDigest(digestUtil.compute(dir)).build();
  }

  private void injectRepo(String repoName, Directory root, Directory... children)
      throws Exception {
    var tree = Tree.newBuilder().setRoot(root);
    for (Directory child : children) {
      tree.addChildren(child);
    }
    assertThat(
            overlayFs.injectRemoteRepo(
                RepositoryName.createUnvalidated(repoName), tree.build(), "marker contents"))
        .isTrue();
  }

  private Path externalPath(String relativePath) {
    return overlayFs.getPath(EXTERNAL_DIR.getRelative(relativePath));
  }

  private String readContent(Path path) throws Exception {
    return new String(FileSystemUtils.readContent(path), UTF_8);
  }

  @Test
  public void inMemorySymlink_relativeTargetInSameRepo() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("file.txt", "hello"))
            .addSymlinks(symlinkNode("link", "file.txt"))
            .build());

    Path link = externalPath("repo/link");
    assertThat(link.isSymbolicLink()).isTrue();
    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(link.getFileSize()).isEqualTo(5);
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("repo/file.txt"));
    assertThat(readContent(link)).isEqualTo("hello");
    assertThat(link.getDigest())
        .isEqualTo(DigestUtil.toBinaryDigest(digestUtil.compute("hello".getBytes(UTF_8))));
  }

  @Test
  public void inMemorySymlink_targetInOtherInMemoryRepo() throws Exception {
    injectRepo("dep+repo", Directory.newBuilder().addFiles(fileNode("file.txt", "hello")).build());
    injectRepo(
        "repo",
        Directory.newBuilder().addSymlinks(symlinkNode("link", "../dep+repo/file.txt")).build());

    Path link = externalPath("repo/link");
    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("dep+repo/file.txt"));
    assertThat(readContent(link)).isEqualTo("hello");
  }

  @Test
  public void inMemorySymlink_targetInNativeRepo() throws Exception {
    Path nativeRepoFile = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/file.txt"));
    nativeRepoFile.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(nativeRepoFile, UTF_8, "native");
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addSymlinks(symlinkNode("link", "../native_repo/file.txt"))
            .build());

    Path link = externalPath("repo/link");
    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("native_repo/file.txt"));
    assertThat(readContent(link)).isEqualTo("native");
  }

  @Test
  public void nativeSymlink_targetInInMemoryRepo() throws Exception {
    injectRepo("repo", Directory.newBuilder().addFiles(fileNode("file.txt", "hello")).build());
    Path nativeLink = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/link"));
    nativeLink.getParentDirectory().createDirectoryAndParents();
    nativeLink.createSymbolicLink(PathFragment.create("../repo/file.txt"));

    Path link = externalPath("native_repo/link");
    assertThat(link.isSymbolicLink()).isTrue();
    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(link.stat(Symlinks.FOLLOW).getSize()).isEqualTo(5);
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("repo/file.txt"));
    assertThat(readContent(link)).isEqualTo("hello");
    assertThat(link.getDigest())
        .isEqualTo(DigestUtil.toBinaryDigest(digestUtil.compute("hello".getBytes(UTF_8))));
  }

  @Test
  public void inMemorySymlink_escapesIntoWorkspaceViaWorkspaceSymlink() throws Exception {
    Path workspaceFile = nativeFs.getPath(WORKSPACE_DIR.getRelative("foo.txt"));
    FileSystemUtils.writeContent(workspaceFile, UTF_8, "workspace");
    nativeFs.getPath(EXTERNAL_DIR.getRelative("_main")).createSymbolicLink(WORKSPACE_DIR);
    injectRepo(
        "repo",
        Directory.newBuilder().addSymlinks(symlinkNode("link", "../_main/foo.txt")).build());

    Path link = externalPath("repo/link");
    assertThat(link.isFile(Symlinks.FOLLOW)).isTrue();
    assertThat(link.resolveSymbolicLinks())
        .isEqualTo(overlayFs.getPath(WORKSPACE_DIR.getRelative("foo.txt")));
    assertThat(readContent(link)).isEqualTo("workspace");
  }

  @Test
  public void symlinkedBzlFile_readFromPrefetchedNativeFile() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("real_helper.bzl", "load stuff"))
            .addSymlinks(symlinkNode("helper.bzl", "real_helper.bzl"))
            .build());

    // Remove the blob from the CAS: reading the symlinked .bzl file must be served from the
    // native copy prefetched during injection, without any remote cache access.
    cacheClient.removeCasEntry(digestUtil.compute("load stuff".getBytes(UTF_8)));

    assertThat(readContent(externalPath("repo/helper.bzl"))).isEqualTo("load stuff");
    assertThat(readContent(externalPath("repo/real_helper.bzl"))).isEqualTo("load stuff");
    assertThat(cacheClient.getNumSuccessfulDownloads()).isEqualTo(0);
  }

  @Test
  public void danglingSymlink_behavesLikeDanglingLink() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder().addSymlinks(symlinkNode("link", "../gone/file.txt")).build());

    Path link = externalPath("repo/link");
    assertThat(link.isSymbolicLink()).isTrue();
    assertThat(link.exists(Symlinks.FOLLOW)).isFalse();
    assertThat(link.statIfFound(Symlinks.FOLLOW)).isNull();
    assertThat(link.readSymbolicLink()).isEqualTo(PathFragment.create("../gone/file.txt"));
    assertThrows(FileNotFoundException.class, link::resolveSymbolicLinks);
  }

  @Test
  public void symlinkLoopAcrossBackings_throws() throws Exception {
    Path nativeLink = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/loop"));
    nativeLink.getParentDirectory().createDirectoryAndParents();
    nativeLink.createSymbolicLink(PathFragment.create("../repo/loop"));
    injectRepo(
        "repo",
        Directory.newBuilder().addSymlinks(symlinkNode("loop", "../native_repo/loop")).build());

    Path link = externalPath("repo/loop");
    assertThrows(FileSymlinkLoopException.class, link::resolveSymbolicLinks);
    assertThrows(FileSymlinkLoopException.class, () -> link.stat(Symlinks.FOLLOW));
    assertThat(link.statNullable(Symlinks.FOLLOW)).isNull();
  }

  @Test
  public void readdir_followClassifiesCrossBackingSymlinks() throws Exception {
    Path nativeRepoFile = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/file.txt"));
    nativeRepoFile.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(nativeRepoFile, UTF_8, "native");
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addSymlinks(symlinkNode("link", "../native_repo/file.txt"))
            .addSymlinks(symlinkNode("dangling", "../gone"))
            .build());

    Path repoDir = externalPath("repo");
    assertThat(repoDir.readdir(Symlinks.NOFOLLOW))
        .containsExactly(
            new Dirent("link", Dirent.Type.SYMLINK), new Dirent("dangling", Dirent.Type.SYMLINK));
    assertThat(repoDir.readdir(Symlinks.FOLLOW))
        .containsExactly(
            new Dirent("link", Dirent.Type.FILE), new Dirent("dangling", Dirent.Type.UNKNOWN));
  }

  @Test
  public void resolveSymbolicLinks_neverLeavesOverlayFileSystem() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("file.txt", "hello"))
            .addSymlinks(symlinkNode("link", "file.txt"))
            .build());
    Path workspaceFile = nativeFs.getPath(WORKSPACE_DIR.getRelative("foo.txt"));
    FileSystemUtils.writeContent(workspaceFile, UTF_8, "workspace");

    // In scope.
    assertThat(externalPath("repo/link").resolveSymbolicLinks().getFileSystem())
        .isSameInstanceAs(overlayFs);
    // Out of scope, delegated wholesale.
    assertThat(
            overlayFs
                .getPath(WORKSPACE_DIR.getRelative("foo.txt"))
                .resolveSymbolicLinks()
                .getFileSystem())
        .isSameInstanceAs(overlayFs);
  }

  @Test
  public void deleteTreeAndRecreate_resolutionReflectsNewState() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("file.txt", "hello"))
            .addSymlinks(symlinkNode("link", "file.txt"))
            .build());
    // Populate the canonicalization cache.
    assertThat(externalPath("repo/link").resolveSymbolicLinks())
        .isEqualTo(externalPath("repo/file.txt"));

    // Simulate a refetch: the repo is deleted and recreated natively with a regular file where
    // the symlink used to be.
    overlayFs.deleteTree(EXTERNAL_DIR.getRelative("repo"));
    Path newFile = externalPath("repo/link");
    newFile.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(newFile, UTF_8, "now a file");

    assertThat(newFile.isSymbolicLink()).isFalse();
    assertThat(newFile.resolveSymbolicLinks()).isEqualTo(newFile);
    assertThat(readContent(newFile)).isEqualTo("now a file");
  }

  @Test
  public void lostFile_reportsEvictionAndRecoversAfterRefetch() throws Exception {
    injectRepo("repo", Directory.newBuilder().addFiles(fileNode("file.txt", "hello")).build());
    Path file = externalPath("repo/file.txt");
    // Populate the canonicalization cache.
    assertThat(file.resolveSymbolicLinks()).isEqualTo(file);

    cacheClient.removeCasEntry(digestUtil.compute("hello".getBytes(UTF_8)));
    var exception = assertThrows(DetailedIOException.class, file::getInputStream);
    assertThat(exception.getDetailedExitCode().getFailureDetail().getFilesystem().getCode())
        .isEqualTo(FailureDetails.Filesystem.Code.REMOTE_FILE_EVICTED);

    // Simulate the refetch triggered by the invalidation of the repository directory.
    overlayFs.deleteTree(EXTERNAL_DIR.getRelative("repo"));
    assertThat(file.statIfFound(Symlinks.FOLLOW)).isNull();
    assertThat(file.exists()).isFalse();
  }

  @Test
  public void materialization_preservesResolutionAcrossOwnershipFlip() throws Exception {
    Path workspaceFile = nativeFs.getPath(WORKSPACE_DIR.getRelative("foo.txt"));
    FileSystemUtils.writeContent(workspaceFile, UTF_8, "workspace");
    nativeFs.getPath(EXTERNAL_DIR.getRelative("_main")).createSymbolicLink(WORKSPACE_DIR);
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("file.txt", "hello"))
            .addSymlinks(symlinkNode("link", "file.txt"))
            .addSymlinks(symlinkNode("workspace_link", "../_main/foo.txt"))
            .build());

    Path link = externalPath("repo/link");
    Path workspaceLink = externalPath("repo/workspace_link");
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("repo/file.txt"));
    assertThat(workspaceLink.resolveSymbolicLinks())
        .isEqualTo(overlayFs.getPath(WORKSPACE_DIR.getRelative("foo.txt")));

    overlayFs.ensureMaterialized(RepositoryName.createUnvalidated("repo"), reporter);

    // The same results must be obtained now that the native file system owns the repo.
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("repo/file.txt"));
    assertThat(readContent(link)).isEqualTo("hello");
    assertThat(workspaceLink.resolveSymbolicLinks())
        .isEqualTo(overlayFs.getPath(WORKSPACE_DIR.getRelative("foo.txt")));
    // The materialized repo is now backed by the native file system.
    assertThat(nativeFs.getPath(EXTERNAL_DIR.getRelative("repo/file.txt")).exists()).isTrue();
    assertThat(
            nativeFs.getPath(EXTERNAL_DIR.getRelative("repo/link")).isSymbolicLink())
        .isTrue();
  }

  @Test
  public void notifyNoCacheAvailable_evictsInMemoryReposAndResolutionCache() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("file.txt", "hello"))
            .addSymlinks(symlinkNode("link", "file.txt"))
            .build());
    Path link = externalPath("repo/link");
    // Populate the canonicalization cache.
    assertThat(link.resolveSymbolicLinks()).isEqualTo(externalPath("repo/file.txt"));

    overlayFs.afterCommand();
    overlayFs.notifyNoCacheAvailable(mock(MemoizingEvaluator.class));

    assertThat(link.statIfFound(Symlinks.FOLLOW)).isNull();
    assertThat(link.exists(Symlinks.NOFOLLOW)).isFalse();
  }

  @Test
  public void nativeRepoMutatedBypassingOverlay_reflectedImmediately() throws Exception {
    // Native repo directories are mutated during fetching by processes run by repo rules, which
    // bypass this file system entirely. Their resolution must not be cached.
    Path realFile = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/real.txt"));
    realFile.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(realFile, UTF_8, "real");
    Path conf = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/conf"));
    FileSystemUtils.writeContent(conf, UTF_8, "regular file");

    Path overlayConf = externalPath("native_repo/conf");
    // Resolve through the overlay so that a (hypothetical) cache would be populated.
    assertThat(overlayConf.isSymbolicLink()).isFalse();
    assertThat(overlayConf.resolveSymbolicLinks()).isEqualTo(overlayConf);

    // Replace the regular file with a symlink directly on the native file system, bypassing the
    // overlay (like a process run by a repo rule would).
    assertThat(conf.delete()).isTrue();
    conf.createSymbolicLink(PathFragment.create("real.txt"));

    assertThat(overlayConf.isSymbolicLink()).isTrue();
    assertThat(overlayConf.resolveSymbolicLinks()).isEqualTo(externalPath("native_repo/real.txt"));
    assertThat(readContent(overlayConf)).isEqualTo("real");
  }

  @Test
  public void renameAcrossBackings_throws() throws Exception {
    injectRepo("repo", Directory.newBuilder().addFiles(fileNode("file.txt", "hello")).build());
    Path nativeRepoDir = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo"));
    nativeRepoDir.createDirectoryAndParents();
    FileSystemUtils.writeContent(nativeRepoDir.getChild("file.txt"), UTF_8, "native");

    var exception =
        assertThrows(
            IOException.class,
            () ->
                externalPath("native_repo/file.txt")
                    .renameTo(externalPath("repo/moved.txt")));
    assertThat(exception).hasMessageThat().contains("Cross-device link");
  }

  @Test
  public void getInputStreamOnInMemoryDirectory_throwsIOException() throws Exception {
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addDirectories(dirNode("subdir", Directory.getDefaultInstance()))
            .build(),
        Directory.getDefaultInstance());

    var exception =
        assertThrows(IOException.class, () -> externalPath("repo/subdir").getInputStream());
    assertThat(exception).hasMessageThat().contains("Is a directory");
  }

  @Test
  public void getxattr_neverThrowsAndFollowsCrossBackingSymlinks() throws Exception {
    Path nativeRepoFile = nativeFs.getPath(EXTERNAL_DIR.getRelative("native_repo/file.txt"));
    nativeRepoFile.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(nativeRepoFile, UTF_8, "native");
    injectRepo(
        "repo",
        Directory.newBuilder()
            .addFiles(fileNode("file.txt", "hello"))
            .addSymlinks(symlinkNode("link", "../native_repo/file.txt"))
            .build());

    // In-memory files have no extended attributes, and a cross-backing symlink is resolved to the
    // native file rather than dispatched to the in-memory backing.
    assertThat(externalPath("repo/file.txt").getxattr("user.foo")).isNull();
    assertThat(externalPath("repo/link").getxattr("user.foo")).isNull();
  }

  @Test
  public void outOfScopePaths_delegatedWholesaleToNativeFs() throws Exception {
    Path workspaceFile = nativeFs.getPath(WORKSPACE_DIR.getRelative("foo.txt"));
    FileSystemUtils.writeContent(workspaceFile, UTF_8, "workspace");

    Path throughOverlay = overlayFs.getPath(WORKSPACE_DIR.getRelative("foo.txt"));
    assertThat(readContent(throughOverlay)).isEqualTo("workspace");
    assertThat(throughOverlay.stat(Symlinks.FOLLOW).getSize()).isEqualTo(9);
    assertThat(throughOverlay.getDigest())
        .isEqualTo(DigestUtil.toBinaryDigest(digestUtil.compute("workspace".getBytes(UTF_8))));
  }
}
